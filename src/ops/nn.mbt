///|
/// 根据 ONNX 规范计算单个空间维度的输出尺寸与前/后填充量（pad_begin, pad_end）。
/// 支持 NOTSET（显式 pads）、VALID、SAME_UPPER、SAME_LOWER 以及 ceil_mode。
fn compute_spatial_out_and_pads(
  in_size : Int32,
  kernel : Int32,
  stride : Int32,
  dilation : Int32,
  explicit_pad_begin : Int32,
  explicit_pad_end : Int32,
  ceil_mode : Bool,
  auto_pad : @ir.AutoPad,
) -> (Int32, Int32, Int32) raise @tensor.TensorError {
  if stride <= 0 || dilation <= 0 || kernel <= 0 {
    raise @tensor.InvalidShape(
      "spatial params must be positive: kernel=\{kernel}, stride=\{stride}, dilation=\{dilation}",
    )
  }
  let eff_kernel = (kernel - 1) * dilation + 1
  match auto_pad {
    @ir.Valid => {
      let out = (in_size - eff_kernel) / stride + 1
      if out <= 0 {
        raise @tensor.InvalidShape(
          "VALID padding yields non-positive output size \{out}",
        )
      }
      (out, 0, 0)
    }
    @ir.SameUpper | @ir.SameLower => {
      let out = (in_size + stride - 1) / stride
      let pad_needed = (out - 1) * stride + eff_kernel - in_size
      let total_pad = if pad_needed > 0 { pad_needed } else { 0 }
      let pad_begin = match auto_pad {
        @ir.SameUpper => total_pad / 2
        _ => (total_pad + 1) / 2
      }
      let pad_end = total_pad - pad_begin
      (out, pad_begin, pad_end)
    }
    @ir.NotSet => {
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
/// 将单张输入特征图的单个分组 `[C_in_g, H, W]` 按 `im2col` 展开为 2D 矩阵
/// `[C_in_g * kH * kW, out_h * out_w]`。
/// 展开后卷积运算即可完全等价于一次矩阵乘法：
/// `W_group[C_out_g, K] @ col[K, out_h * out_w]`，从而直接复用标量或 V128 SIMD GEMM 内核。
pub fn im2col_2d_f32(
  input : FixedArray[Float],
  in_base : Int32,
  channels : Int32,
  in_h : Int32,
  in_w : Int32,
  kh : Int32,
  kw : Int32,
  stride_h : Int32,
  stride_w : Int32,
  pad_top : Int32,
  pad_left : Int32,
  dil_h : Int32,
  dil_w : Int32,
  out_h : Int32,
  out_w : Int32,
  col : FixedArray[Float],
) -> Unit {
  let out_spatial = out_h * out_w
  let in_hw = in_h * in_w
  let zero : Float = 0.0
  let mut col_row = 0

  for c = 0; c < channels; c = c + 1 {
    let ch_base = in_base + c * in_hw
    for ky = 0; ky < kh; ky = ky + 1 {
      for kx = 0; kx < kw; kx = kx + 1 {
        let row_off = col_row * out_spatial
        let mut out_idx = 0
        for oy = 0; oy < out_h; oy = oy + 1 {
          let iy = oy * stride_h - pad_top + ky * dil_h
          if iy < 0 || iy >= in_h {
            // 整行越界（处于上下 padding 区域），直接填 0
            for _ox = 0; _ox < out_w; _ox = _ox + 1 {
              col[row_off + out_idx] = zero
              out_idx = out_idx + 1
            }
          } else {
            let row_in_base = ch_base + iy * in_w
            for ox = 0; ox < out_w; ox = ox + 1 {
              let ix = ox * stride_w - pad_left + kx * dil_w
              col[row_off + out_idx] = if ix >= 0 && ix < in_w {
                input[row_in_base + ix]
              } else {
                zero
              }
              out_idx = out_idx + 1
            }
          }
        }
        col_row = col_row + 1
      }
    }
  }
}

///|
/// 2D 卷积核心实现，允许通过 `gemm_fn` 注入不同的矩阵乘法内核（标量或 V128 SIMD），
/// 避免在不同后端间重复编写 padding / dilation / group / im2col 逻辑。
pub fn conv2d_f32_with_gemm(
  x : @tensor.Tensor,
  w : @tensor.Tensor,
  bias : @tensor.Tensor?,
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  group : Int32,
  auto_pad : @ir.AutoPad,
  gemm_fn : (
    FixedArray[Float],
    Int32,
    FixedArray[Float],
    Int32,
    FixedArray[Float],
    Int32,
    Int32,
    Int32,
    Int32,
  ) -> Unit,
) -> @tensor.Tensor raise @tensor.TensorError {
  let xs = x.shape()
  let ws = w.shape()
  if xs.length() != 4 || ws.length() != 4 {
    raise @tensor.ShapeMismatch(
      "conv2d_f32 requires 4D NCHW input and 4D OIHW weight, got ranks \{xs.length()} and \{ws.length()}",
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
  let out_buf = FixedArray::make(n_batch * c_out * out_spatial, (0.0 : Float))

  // 复用单个 im2col 工作缓冲，避免在 batch / group 循环中重复分配大数组
  let col_rows = c_in_per_group * kh * kw
  let col_buf = FixedArray::make(col_rows * out_spatial, (0.0 : Float))

  let in_batch_stride = c_in * in_h * in_w
  let in_group_stride = c_in_per_group * in_h * in_w
  let w_group_stride = c_out_per_group * col_rows
  let out_batch_stride = c_out * out_spatial
  let out_group_stride = c_out_per_group * out_spatial

  for b = 0; b < n_batch; b = b + 1 {
    for g = 0; g < group; g = g + 1 {
      let in_off = b * in_batch_stride + g * in_group_stride
      im2col_2d_f32(
        x_buf, in_off, c_in_per_group, in_h, in_w, kh, kw, stride_h, stride_w, pad_top,
        pad_left, dil_h, dil_w, out_h, out_w, col_buf,
      )
      let w_off = g * w_group_stride
      let out_off = b * out_batch_stride + g * out_group_stride
      // W_group[c_out_per_group, col_rows] @ col[col_rows, out_spatial]
      gemm_fn(
        w_buf, w_off, col_buf, 0, out_buf, out_off, c_out_per_group, col_rows, out_spatial,
      )
    }
  }

  // 若提供了可选 1D 偏置 B[C_out]，叠加到对应输出通道
  match bias {
    Some(b_tensor) => {
      let b_buf = b_tensor.to_f32_buffer()
      if b_buf.length() != c_out {
        raise @tensor.ShapeMismatch(
          "conv2d_f32 bias length \{b_buf.length()} != c_out \{c_out}",
        )
      }
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
    }
    None => ()
  }

  @tensor.Tensor::from_f32([n_batch, c_out, out_h, out_w], out_buf)
}

///|
/// ONNX 2D Conv 算子（标量 GEMM 默认入口）。
pub fn conv2d_f32(
  x : @tensor.Tensor,
  w : @tensor.Tensor,
  bias : @tensor.Tensor?,
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  group : Int32,
  auto_pad : @ir.AutoPad,
) -> @tensor.Tensor raise @tensor.TensorError {
  conv2d_f32_with_gemm(
    x, w, bias, strides, pads, dilations, group, auto_pad, matmul_2d_raw_f32,
  )
}

///|
/// ONNX 2D MaxPool 算子（NCHW 布局，支持 stride / pads / dilations / ceil_mode / auto_pad）。
pub fn maxpool_2d_f32(
  x : @tensor.Tensor,
  kernel_shape : Array[Int32],
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  ceil_mode : Bool,
  auto_pad : @ir.AutoPad,
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
  let out = FixedArray::make(n_batch * channels * out_h * out_w, (0.0 : Float))
  let in_hw = in_h * in_w
  let out_hw = out_h * out_w

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
/// ONNX 2D AveragePool 算子（NCHW 布局，支持 count_include_pad 与 ceil_mode）。
pub fn avgpool_2d_f32(
  x : @tensor.Tensor,
  kernel_shape : Array[Int32],
  strides : Array[Int32],
  pads : Array[Int32],
  ceil_mode : Bool,
  count_include_pad : Bool,
  auto_pad : @ir.AutoPad,
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
  let out = FixedArray::make(n_batch * channels * out_h * out_w, (0.0 : Float))
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
/// ONNX 推理态 BatchNormalization 算子：
/// `y = scale * (x - mean) / sqrt(var + epsilon) + bias`
/// 预计算每通道仿射系数 `(a_c, b_c)` 使热循环退化为一次乘加 `a_c * x + b_c`。
pub fn batch_norm_f32(
  x : @tensor.Tensor,
  scale : @tensor.Tensor,
  bias : @tensor.Tensor,
  mean : @tensor.Tensor,
  variance : @tensor.Tensor,
  epsilon~ : Float = (1.0e-5 : Float),
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

  // 预计算每个通道的等效线性系数 y = a[c] * x + shift[c]
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
  let out = FixedArray::make(src.length(), (0.0 : Float))
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
