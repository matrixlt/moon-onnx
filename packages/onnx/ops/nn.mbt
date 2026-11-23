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
/// ONNX 2D Conv 算子：底层复用 `matrixlt/ml_kernels/conv::im2col_2d_f32`
/// 与 `matrixlt/ml_kernels/gemm`（支持标量与 V128 SIMD 分派）。
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
/// ONNX 2D MaxPool 算子。
pub fn maxpool_2d_f32(
  x : @tensor.Tensor,
  kernel_shape : Array[Int32],
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  ceil_mode : Bool,
  auto_pad : @nn_ir.AutoPad,
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
/// ONNX 2D AveragePool 算子。
pub fn avgpool_2d_f32(
  x : @tensor.Tensor,
  kernel_shape : Array[Int32],
  strides : Array[Int32],
  pads : Array[Int32],
  ceil_mode : Bool,
  count_include_pad : Bool,
  auto_pad : @nn_ir.AutoPad,
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
/// ONNX 推理态 BatchNormalization 算子。
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
