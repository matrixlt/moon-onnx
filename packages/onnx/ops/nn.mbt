///|
/// 根据 ONNX 规范计算单个空间维度的输出尺寸与前/后填充量。
fn compute_spatial_out_and_pads(
  in_size : Int32,
  kernel : Int32,
  stride : Int32,
  dilation : Int32,
  explicit_pad_begin : Int32,
  explicit_pad_end : Int32,
  ceil_mode : Bool,
  auto_pad : @nn_ir.AutoPad,
) -> (Int32, Int32, Int32) raise @tensor.TensorError {
  if stride <= 0 || dilation <= 0 || kernel <= 0 {
    raise @tensor.InvalidShape(
      "spatial params must be positive: kernel=\{kernel}, stride=\{stride}, dilation=\{dilation}",
    )
  }
  let eff_kernel = (kernel - 1) * dilation + 1
  match auto_pad {
    @nn_ir.Valid => {
      let out = (in_size - eff_kernel) / stride + 1
      if out <= 0 {
        raise @tensor.InvalidShape(
          "VALID padding yields non-positive output size \{out}",
        )
      }
      (out, 0, 0)
    }
    @nn_ir.SameUpper | @nn_ir.SameLower => {
      let out = (in_size + stride - 1) / stride
      let pad_needed = (out - 1) * stride + eff_kernel - in_size
      let total_pad = if pad_needed > 0 { pad_needed } else { 0 }
      let pad_begin = match auto_pad {
        @nn_ir.SameUpper => total_pad / 2
        _ => (total_pad + 1) / 2
      }
      let pad_end = total_pad - pad_begin
      (out, pad_begin, pad_end)
    }
    @nn_ir.NotSet => {
      let numer = in_size + explicit_pad_begin + explicit_pad_end - eff_kernel
      let out = if ceil_mode {
        (numer + stride - 1) / stride + 1
      } else {
        numer / stride + 1
      }
      if out <= 0 {
        raise @tensor.InvalidShape(
          "computed non-positive spatial output size \{out}",
        )
      }
      (out, explicit_pad_begin, explicit_pad_end)
    }
  }
}

///|
/// 将 IR 层的 `FusedActivation` 映射为 `ml_kernels/gemm` 的寄存器级 `Epilogue`。
fn to_gemm_epilogue(act : @nn_ir.FusedActivation) -> @gemm.Epilogue {
  match act {
    @nn_ir.NoAct => @gemm.NoEpilogue
    @nn_ir.ReluAct => @gemm.Relu
    @nn_ir.ClipAct(min~, max~) => @gemm.Clip(min~, max~)
  }
}

///|
/// 预打包后的 2D 卷积静态权重：
/// - 常规/分组卷积：按 `group` 将 `W_g[C_out_g, C_in_g * kH * kW]` 转置并打包为 16 字节对齐的 `PackedMatrixB(K, C_out_g)`；
/// - 深度可分离卷积（`group == c_out && c_in_per_group == 1`）：直接保留紧凑 `depthwise_weight` 走零 `im2col` 专用 V128 内核。
pub struct PackedConvWeight {
  c_out : Int32
  c_in_per_group : Int32
  kh : Int32
  kw : Int32
  group : Int32
  groups : Array[@gemm.PackedMatrixB]
  depthwise_weight : FixedArray[Float]?
}

///|
/// 返回预打包卷积权重对应的原始 OIHW 形状 `[C_out, C_in_per_group, kH, kW]`。
pub fn PackedConvWeight::shape(self : PackedConvWeight) -> Array[Int32] {
  [self.c_out, self.c_in_per_group, self.kh, self.kw]
}

///|
/// 返回预打包卷积权重对应的分组数 `group`。
pub fn PackedConvWeight::group(self : PackedConvWeight) -> Int32 {
  self.group
}

///|
/// 返回该预打包权重是否命中深度可分离卷积（Depthwise Conv）专用路径。
pub fn PackedConvWeight::is_depthwise(self : PackedConvWeight) -> Bool {
  match self.depthwise_weight {
    Some(_) => true
    None => false
  }
}

///|
/// 在模型加载/会话构建期将 4D OIHW 或 3D OIK（Conv1D，自动视为 H=1）Float32 卷积权重一次性预打包为 `PackedConvWeight`。
pub fn pack_conv2d_weight_f32(
  w : @tensor.Tensor,
  group : Int32,
) -> PackedConvWeight raise @tensor.TensorError {
  let ws = w.shape()
  if ws.length() != 4 && ws.length() != 3 {
    raise @tensor.ShapeMismatch(
      "pack_conv2d_weight_f32 requires 3D OIK or 4D OIHW weight, got rank \{ws.length()}",
    )
  }
  let c_out = ws[0]
  let c_in_per_group = ws[1]
  let kh = if ws.length() == 4 { ws[2] } else { 1 }
  let kw = if ws.length() == 4 { ws[3] } else { ws[2] }
  if group <= 0 || c_out % group != 0 {
    raise @tensor.ShapeMismatch(
      "pack_conv2d_weight_f32 invalid group=\{group} for c_out=\{c_out}",
    )
  }
  let c_out_per_group = c_out / group
  let w_buf = w.to_f32_buffer()
  // Depthwise Conv（group == c_out && c_in_per_group == 1）：无需打包 GEMM 面板，直接走零 im2col 专用内核
  if group == c_out && c_in_per_group == 1 && c_out_per_group == 1 {
    return {
      c_out,
      c_in_per_group,
      kh,
      kw,
      group,
      groups: [],
      depthwise_weight: Some(w_buf),
    }
  }
  let k = c_in_per_group * kh * kw
  let w_group_stride = c_out_per_group * k
  let groups = Array::makei(group, fn(g) {
    @gemm.pack_matrix_b_transposed_f32(
      w_buf,
      g * w_group_stride,
      c_out_per_group,
      k,
    )
  })
  { c_out, c_in_per_group, kh, kw, group, groups, depthwise_weight: None, }
}

///|
/// 将 1D 卷积参数 `[s]` / `[p_l, p_r]` / `[d]` 规范化为等价的 H=1 2D 卷积参数，复用全部 V128 SIMD 卷积内核。
fn normalize_conv1d_params(
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
) -> (Array[Int32], Array[Int32], Array[Int32]) {
  let sw = if strides.length() == 1 {
    strides[0]
  } else if strides.length() >= 2 {
    strides[1]
  } else {
    1
  }
  let (pl, pr) = if pads.length() == 2 {
    (pads[0], pads[1])
  } else if pads.length() >= 4 {
    (pads[1], pads[3])
  } else {
    (0, 0)
  }
  let dw = if dilations.length() == 1 {
    dilations[0]
  } else if dilations.length() >= 2 {
    dilations[1]
  } else {
    1
  }
  ([1, sw], [0, pl, 0, pr], [1, dw])
}

///|
/// 使用预打包静态权重 `PackedConvWeight` 执行 2D/1D Conv（V128 SIMD 零运行期打包 + 偏置/激活融合 + Depthwise 直算 + BufferPool 复用）。
pub fn conv2d_prepacked_f32(
  x : @tensor.Tensor,
  packed_w : PackedConvWeight,
  bias : @tensor.Tensor?,
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  auto_pad : @nn_ir.AutoPad,
  activation~ : @nn_ir.FusedActivation = @nn_ir.NoAct,
  pool~ : @tensor.BufferPool? = None,
  scratch_pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let xs = x.shape()
  if xs.length() == 3 {
    let x4 = x.reshape([xs[0], xs[1], 1, xs[2]])
    let (st2, pd2, dl2) = normalize_conv1d_params(strides, pads, dilations)
    let y4 = conv2d_prepacked_f32(
      x4,
      packed_w,
      bias,
      st2,
      pd2,
      dl2,
      auto_pad,
      activation~,
      pool~,
      scratch_pool~,
    )
    let ys = y4.shape()
    return y4.reshape([ys[0], ys[1], ys[3]])
  }
  if xs.length() != 4 {
    raise @tensor.ShapeMismatch(
      "conv2d_prepacked_f32 requires 3D NCL or 4D NCHW input, got rank \{xs.length()}",
    )
  }
  let n_batch = xs[0]
  let c_in = xs[1]
  let in_h = xs[2]
  let in_w = xs[3]

  let c_out = packed_w.c_out
  let c_in_per_group = packed_w.c_in_per_group
  let kh = packed_w.kh
  let kw = packed_w.kw
  let group = packed_w.group

  if c_in != c_in_per_group * group {
    raise @tensor.ShapeMismatch(
      "conv2d_prepacked_f32 invalid c_in=\{c_in} for group=\{group}, w_in=\{c_in_per_group}",
    )
  }
  let c_out_per_group = c_out / group

  let bias_buf : FixedArray[Float]? = match bias {
    Some(b_tensor) => {
      let b_buf = b_tensor.to_f32_buffer()
      if b_buf.length() != c_out {
        raise @tensor.ShapeMismatch(
          "conv2d_prepacked_f32 bias length \{b_buf.length()} != c_out \{c_out}",
        )
      }
      Some(b_buf)
    }
    None => None
  }

  let stride_h = if strides.length() >= 1 { strides[0] } else { 1 }
  let stride_w = if strides.length() >= 2 { strides[1] } else { stride_h }
  let dil_h = if dilations.length() >= 1 { dilations[0] } else { 1 }
  let dil_w = if dilations.length() >= 2 { dilations[1] } else { dil_h }
  let pad_top_in = if pads.length() >= 1 { pads[0] } else { 0 }
  let pad_left_in = if pads.length() >= 2 { pads[1] } else { 0 }
  let pad_bottom_in = if pads.length() >= 3 { pads[2] } else { pad_top_in }
  let pad_right_in = if pads.length() >= 4 { pads[3] } else { pad_left_in }

  let (out_h, pad_top, _) = compute_spatial_out_and_pads(
    in_h, kh, stride_h, dil_h, pad_top_in, pad_bottom_in, false, auto_pad,
  )
  let (out_w, pad_left, _) = compute_spatial_out_and_pads(
    in_w, kw, stride_w, dil_w, pad_left_in, pad_right_in, false, auto_pad,
  )

  let x_buf = x.to_f32_buffer()
  let out_spatial = out_h * out_w
  let epilogue = to_gemm_epilogue(activation)
  let out_buf = alloc_f32(pool, n_batch * c_out * out_spatial)
  let in_batch_stride = c_in * in_h * in_w
  let out_batch_stride = c_out * out_spatial

  // 专用快速路径：Depthwise 2D Conv（零 im2col 分配 + 4-wide V128 向量化 + 融合偏置与激活）
  match packed_w.depthwise_weight {
    Some(dw_w) => {
      for b = 0; b < n_batch; b = b + 1 {
        @conv.depthwise_conv2d_f32(
          x_buf,
          b * in_batch_stride,
          dw_w,
          bias_buf,
          c_out,
          in_h,
          in_w,
          kh,
          kw,
          stride_h,
          stride_w,
          pad_top,
          pad_left,
          dil_h,
          dil_w,
          out_h,
          out_w,
          out_buf,
          b * out_batch_stride,
          epilogue~,
        )
      }
      return @tensor.Tensor::from_f32([n_batch, c_out, out_h, out_w], out_buf)
    }
    None => ()
  }

  let col_k = c_in_per_group * kh * kw
  let work_pool = match scratch_pool {
    Some(_) => scratch_pool
    None => pool
  }
  let col_t_buf = alloc_f32(work_pool, out_spatial * col_k)
  let in_group_stride = c_in_per_group * in_h * in_w
  let out_group_stride = c_out_per_group * out_spatial

  for b = 0; b < n_batch; b = b + 1 {
    for g = 0; g < group; g = g + 1 {
      let in_off = b * in_batch_stride + g * in_group_stride
      @conv.im2col_2d_nhwc_f32(
        x_buf, in_off, c_in_per_group, in_h, in_w, kh, kw, stride_h, stride_w, pad_top,
        pad_left, dil_h, dil_w, out_h, out_w, col_t_buf,
      )
      let out_off = b * out_batch_stride + g * out_group_stride
      @gemm.matmul_2d_prepacked_nchw_simd_f32(
        col_t_buf,
        0,
        packed_w.groups[g],
        bias_buf,
        g * c_out_per_group,
        out_buf,
        out_off,
        out_spatial,
        epilogue~,
      )
    }
  }

  free_f32(work_pool, col_t_buf)
  @tensor.Tensor::from_f32([n_batch, c_out, out_h, out_w], out_buf)
}

///|
/// ONNX 2D Conv 算子：底层复用 `matrixlt/ml_kernels/conv::im2col_2d_f32`
/// 与 `matrixlt/ml_kernels/gemm`（支持标量、V128 SIMD、融合激活与 `BufferPool` 缓冲复用）。
pub fn conv2d_f32(
  x : @tensor.Tensor,
  w : @tensor.Tensor,
  bias : @tensor.Tensor?,
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  group : Int32,
  auto_pad : @nn_ir.AutoPad,
  use_simd~ : Bool = false,
  activation~ : @nn_ir.FusedActivation = @nn_ir.NoAct,
  pool~ : @tensor.BufferPool? = None,
  scratch_pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let xs = x.shape()
  let ws = w.shape()
  if xs.length() == 3 && ws.length() == 3 {
    let x4 = x.reshape([xs[0], xs[1], 1, xs[2]])
    let w4 = w.reshape([ws[0], ws[1], 1, ws[2]])
    let (st2, pd2, dl2) = normalize_conv1d_params(strides, pads, dilations)
    let y4 = conv2d_f32(
      x4,
      w4,
      bias,
      st2,
      pd2,
      dl2,
      group,
      auto_pad,
      use_simd~,
      activation~,
      pool~,
      scratch_pool~,
    )
    let ys = y4.shape()
    return y4.reshape([ys[0], ys[1], ys[3]])
  }
  if xs.length() != 4 || ws.length() != 4 {
    raise @tensor.ShapeMismatch(
      "conv2d_f32 requires 3D NCL/OIK or 4D NCHW/OIHW inputs, got ranks \{xs.length()} and \{ws.length()}",
    )
  }
  let n_batch = xs[0]
  let c_in = xs[1]
  let in_h = xs[2]
  let in_w = xs[3]

  let c_out = ws[0]
  let c_in_per_group = ws[1]
  let kh = ws[2]
  let kw = ws[3]

  if group <= 0 || c_in != c_in_per_group * group || c_out % group != 0 {
    raise @tensor.ShapeMismatch(
      "conv2d_f32 invalid group=\{group} for c_in=\{c_in}, w_in=\{c_in_per_group}, c_out=\{c_out}",
    )
  }
  let c_out_per_group = c_out / group

  let bias_buf : FixedArray[Float]? = match bias {
    Some(b_tensor) => {
      let b_buf = b_tensor.to_f32_buffer()
      if b_buf.length() != c_out {
        raise @tensor.ShapeMismatch(
          "conv2d_f32 bias length \{b_buf.length()} != c_out \{c_out}",
        )
      }
      Some(b_buf)
    }
    None => None
  }

  let stride_h = if strides.length() >= 1 { strides[0] } else { 1 }
  let stride_w = if strides.length() >= 2 { strides[1] } else { stride_h }
  let dil_h = if dilations.length() >= 1 { dilations[0] } else { 1 }
  let dil_w = if dilations.length() >= 2 { dilations[1] } else { dil_h }
  let pad_top_in = if pads.length() >= 1 { pads[0] } else { 0 }
  let pad_left_in = if pads.length() >= 2 { pads[1] } else { 0 }
  let pad_bottom_in = if pads.length() >= 3 { pads[2] } else { pad_top_in }
  let pad_right_in = if pads.length() >= 4 { pads[3] } else { pad_left_in }

  let (out_h, pad_top, _) = compute_spatial_out_and_pads(
    in_h, kh, stride_h, dil_h, pad_top_in, pad_bottom_in, false, auto_pad,
  )
  let (out_w, pad_left, _) = compute_spatial_out_and_pads(
    in_w, kw, stride_w, dil_w, pad_left_in, pad_right_in, false, auto_pad,
  )

  let x_buf = x.to_f32_buffer()
  let w_buf = w.to_f32_buffer()
  let out_spatial = out_h * out_w
  let col_rows = c_in_per_group * kh * kw
  let work_pool = match scratch_pool {
    Some(_) => scratch_pool
    None => pool
  }

  let out_buf = alloc_f32(pool, n_batch * c_out * out_spatial)
  let col_buf = alloc_f32(work_pool, col_rows * out_spatial)

  let in_batch_stride = c_in * in_h * in_w
  let in_group_stride = c_in_per_group * in_h * in_w
  let w_group_stride = c_out_per_group * col_rows
  let out_batch_stride = c_out * out_spatial
  let out_group_stride = c_out_per_group * out_spatial

  for b = 0; b < n_batch; b = b + 1 {
    for g = 0; g < group; g = g + 1 {
      let in_off = b * in_batch_stride + g * in_group_stride
      @conv.im2col_2d_f32(
        x_buf, in_off, c_in_per_group, in_h, in_w, kh, kw, stride_h, stride_w, pad_top,
        pad_left, dil_h, dil_w, out_h, out_w, col_buf,
      )
      let w_off = g * w_group_stride
      let out_off = b * out_batch_stride + g * out_group_stride
      run_gemm_2d(
        w_buf, w_off, col_buf, 0, out_buf, out_off, c_out_per_group, col_rows, out_spatial,
        use_simd,
      )
    }
  }

  free_f32(work_pool, col_buf)

  match bias_buf {
    Some(b_buf) =>
      for b = 0; b < n_batch; b = b + 1 {
        let b_base = b * out_batch_stride
        for oc = 0; oc < c_out; oc = oc + 1 {
          let bv = b_buf[oc]
          let ch_base = b_base + oc * out_spatial
          for s = 0; s < out_spatial; s = s + 1 {
            out_buf[ch_base + s] = out_buf[ch_base + s] + bv
          }
        }
      }
    None => ()
  }

  // 标量/未预打包路径下的尾部融合激活
  match activation {
    @nn_ir.NoAct => ()
    @nn_ir.ReluAct => {
      let zero : Float = 0.0
      for i = 0; i < out_buf.length(); i = i + 1 {
        if out_buf[i] < zero {
          out_buf[i] = zero
        }
      }
    }
    @nn_ir.ClipAct(min~, max~) =>
      for i = 0; i < out_buf.length(); i = i + 1 {
        let v = out_buf[i]
        let lo = if v > min { v } else { min }
        out_buf[i] = if lo < max { lo } else { max }
      }
  }

  @tensor.Tensor::from_f32([n_batch, c_out, out_h, out_w], out_buf)
}

///|
/// ONNX 2D MaxPool 算子（支持无填充无空洞快速路径与可选 `BufferPool` 复用）。
pub fn maxpool_2d_f32(
  x : @tensor.Tensor,
  kernel_shape : Array[Int32],
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  ceil_mode : Bool,
  auto_pad : @nn_ir.AutoPad,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let xs = x.shape()
  if xs.length() != 4 || kernel_shape.length() != 2 {
    raise @tensor.ShapeMismatch(
      "maxpool_2d_f32 requires 4D NCHW input and 2D kernel_shape",
    )
  }
  let n_batch = xs[0]
  let channels = xs[1]
  let in_h = xs[2]
  let in_w = xs[3]
  let kh = kernel_shape[0]
  let kw = kernel_shape[1]

  let stride_h = if strides.length() >= 1 { strides[0] } else { 1 }
  let stride_w = if strides.length() >= 2 { strides[1] } else { stride_h }
  let dil_h = if dilations.length() >= 1 { dilations[0] } else { 1 }
  let dil_w = if dilations.length() >= 2 { dilations[1] } else { dil_h }
  let pad_top_in = if pads.length() >= 1 { pads[0] } else { 0 }
  let pad_left_in = if pads.length() >= 2 { pads[1] } else { 0 }
  let pad_bottom_in = if pads.length() >= 3 { pads[2] } else { pad_top_in }
  let pad_right_in = if pads.length() >= 4 { pads[3] } else { pad_left_in }

  let (out_h, pad_top, _) = compute_spatial_out_and_pads(
    in_h, kh, stride_h, dil_h, pad_top_in, pad_bottom_in, ceil_mode, auto_pad,
  )
  let (out_w, pad_left, _) = compute_spatial_out_and_pads(
    in_w, kw, stride_w, dil_w, pad_left_in, pad_right_in, ceil_mode, auto_pad,
  )

  let src = x.to_f32_buffer()
  let out = alloc_f32(pool, n_batch * channels * out_h * out_w)
  let in_hw = in_h * in_w
  let out_hw = out_h * out_w

  // 快速路径：无填充且 dilation=1 且所有窗口严格位于输入范围内（消除内层边界判断分支）
  if pad_top == 0 &&
    pad_left == 0 &&
    dil_h == 1 &&
    dil_w == 1 &&
    (out_h - 1) * stride_h + kh <= in_h &&
    (out_w - 1) * stride_w + kw <= in_w {
    for b = 0; b < n_batch; b = b + 1 {
      for c = 0; c < channels; c = c + 1 {
        let in_ch = (b * channels + c) * in_hw
        let out_ch = (b * channels + c) * out_hw
        for oy = 0; oy < out_h; oy = oy + 1 {
          let iy0 = oy * stride_h
          let out_row = out_ch + oy * out_w
          for ox = 0; ox < out_w; ox = ox + 1 {
            let mut row_base = in_ch + iy0 * in_w + ox * stride_w
            let mut best = src[row_base]
            for ky = 0; ky < kh; ky = ky + 1 {
              for kx = 0; kx < kw; kx = kx + 1 {
                let v = src[row_base + kx]
                if v > best {
                  best = v
                }
              }
              row_base = row_base + in_w
            }
            out[out_row + ox] = best
          }
        }
      }
    }
    return @tensor.Tensor::from_f32([n_batch, channels, out_h, out_w], out)
  }

  for b = 0; b < n_batch; b = b + 1 {
    for c = 0; c < channels; c = c + 1 {
      let in_ch = (b * channels + c) * in_hw
      let out_ch = (b * channels + c) * out_hw
      for oy = 0; oy < out_h; oy = oy + 1 {
        for ox = 0; ox < out_w; ox = ox + 1 {
          let mut has_val = false
          let mut best : Float = 0.0
          for ky = 0; ky < kh; ky = ky + 1 {
            let iy = oy * stride_h - pad_top + ky * dil_h
            if iy < 0 || iy >= in_h {
              continue
            }
            for kx = 0; kx < kw; kx = kx + 1 {
              let ix = ox * stride_w - pad_left + kx * dil_w
              if ix < 0 || ix >= in_w {
                continue
              }
              let v = src[in_ch + iy * in_w + ix]
              if !has_val || v > best {
                best = v
                has_val = true
              }
            }
          }
          out[out_ch + oy * out_w + ox] = best
        }
      }
    }
  }
  @tensor.Tensor::from_f32([n_batch, channels, out_h, out_w], out)
}

///|
/// ONNX 2D AveragePool 算子（支持可选 `BufferPool` 复用）。
pub fn avgpool_2d_f32(
  x : @tensor.Tensor,
  kernel_shape : Array[Int32],
  strides : Array[Int32],
  pads : Array[Int32],
  ceil_mode : Bool,
  count_include_pad : Bool,
  auto_pad : @nn_ir.AutoPad,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let xs = x.shape()
  if xs.length() != 4 || kernel_shape.length() != 2 {
    raise @tensor.ShapeMismatch(
      "avgpool_2d_f32 requires 4D NCHW input and 2D kernel_shape",
    )
  }
  let n_batch = xs[0]
  let channels = xs[1]
  let in_h = xs[2]
  let in_w = xs[3]
  let kh = kernel_shape[0]
  let kw = kernel_shape[1]

  let stride_h = if strides.length() >= 1 { strides[0] } else { 1 }
  let stride_w = if strides.length() >= 2 { strides[1] } else { stride_h }
  let pad_top_in = if pads.length() >= 1 { pads[0] } else { 0 }
  let pad_left_in = if pads.length() >= 2 { pads[1] } else { 0 }
  let pad_bottom_in = if pads.length() >= 3 { pads[2] } else { pad_top_in }
  let pad_right_in = if pads.length() >= 4 { pads[3] } else { pad_left_in }

  let (out_h, pad_top, _) = compute_spatial_out_and_pads(
    in_h, kh, stride_h, 1, pad_top_in, pad_bottom_in, ceil_mode, auto_pad,
  )
  let (out_w, pad_left, _) = compute_spatial_out_and_pads(
    in_w, kw, stride_w, 1, pad_left_in, pad_right_in, ceil_mode, auto_pad,
  )

  let src = x.to_f32_buffer()
  let out = alloc_f32(pool, n_batch * channels * out_h * out_w)
  let in_hw = in_h * in_w
  let out_hw = out_h * out_w
  let full_window = kh * kw

  for b = 0; b < n_batch; b = b + 1 {
    for c = 0; c < channels; c = c + 1 {
      let in_ch = (b * channels + c) * in_hw
      let out_ch = (b * channels + c) * out_hw
      for oy = 0; oy < out_h; oy = oy + 1 {
        for ox = 0; ox < out_w; ox = ox + 1 {
          let mut sum : Float = 0.0
          let mut valid_count = 0
          for ky = 0; ky < kh; ky = ky + 1 {
            let iy = oy * stride_h - pad_top + ky
            if iy < 0 || iy >= in_h {
              continue
            }
            for kx = 0; kx < kw; kx = kx + 1 {
              let ix = ox * stride_w - pad_left + kx
              if ix < 0 || ix >= in_w {
                continue
              }
              sum = sum + src[in_ch + iy * in_w + ix]
              valid_count = valid_count + 1
            }
          }
          let denom = if count_include_pad { full_window } else { valid_count }
          out[out_ch + oy * out_w + ox] = if denom > 0 {
            sum / Float::from_int(denom)
          } else {
            (0.0 : Float)
          }
        }
      }
    }
  }
  @tensor.Tensor::from_f32([n_batch, channels, out_h, out_w], out)
}

///|
/// ONNX 推理态 BatchNormalization 算子（支持可选 `BufferPool` 复用）。
pub fn batch_norm_f32(
  x : @tensor.Tensor,
  scale : @tensor.Tensor,
  bias : @tensor.Tensor,
  mean : @tensor.Tensor,
  variance : @tensor.Tensor,
  epsilon~ : Float = (1.0e-5 : Float),
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let xs = x.shape()
  let rank = xs.length()
  if rank < 2 {
    raise @tensor.ShapeMismatch(
      "batch_norm_f32 expects input rank >= 2 ([N, C, ...]), got rank \{rank}",
    )
  }
  let n_batch = xs[0]
  let channels = xs[1]
  let mut spatial = 1
  for i = 2; i < rank; i = i + 1 {
    spatial = spatial * xs[i]
  }

  let s_buf = scale.to_f32_buffer()
  let b_buf = bias.to_f32_buffer()
  let m_buf = mean.to_f32_buffer()
  let v_buf = variance.to_f32_buffer()
  if s_buf.length() != channels ||
    b_buf.length() != channels ||
    m_buf.length() != channels ||
    v_buf.length() != channels {
    raise @tensor.ShapeMismatch(
      "batch_norm_f32 parameter lengths must match channel count \{channels}",
    )
  }

  let a_coef = FixedArray::make(channels, (0.0 : Float))
  let shift_coef = FixedArray::make(channels, (0.0 : Float))
  let eps_d = epsilon.to_double()
  for c = 0; c < channels; c = c + 1 {
    let inv_std = 1.0 / (v_buf[c].to_double() + eps_d).sqrt()
    let a = s_buf[c].to_double() * inv_std
    let shift = b_buf[c].to_double() - m_buf[c].to_double() * a
    a_coef[c] = Float::from_double(a)
    shift_coef[c] = Float::from_double(shift)
  }

  let src = x.to_f32_buffer()
  let out = alloc_f32(pool, src.length())
  for b = 0; b < n_batch; b = b + 1 {
    for c = 0; c < channels; c = c + 1 {
      let a = a_coef[c]
      let shift = shift_coef[c]
      let base = (b * channels + c) * spatial
      for s = 0; s < spatial; s = s + 1 {
        out[base + s] = src[base + s] * a + shift
      }
    }
  }
  @tensor.Tensor::from_f32(xs, out)
}
