///|
/// 与 ONNX Slice 规范一致的静态切片端点规范化函数（供形状推导、动态算子与 AOT 代码生成共享）。
pub fn clamp_slice_bound_static(v : Int64, dim : Int32, step : Int32) -> Int32 {
  let d64 = dim.to_int64()
  let norm = if v < (0 : Int64) { v + d64 } else { v }
  if step > 0 {
    if norm < (0 : Int64) {
      0
    } else if norm > d64 {
      dim
    } else {
      norm.to_int()
    }
  } else if norm < (-1 : Int64) {
    -1
  } else if norm >= d64 {
    dim - 1
  } else {
    norm.to_int()
  }
}

///|
/// 根据 ONNX 规范计算单个空间维度的 `(out_size, pad_begin, pad_end)`（全项目唯一空间填充与尺寸计算真相源）。
pub fn compute_spatial_out_and_pads(
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
/// 计算 1D/2D 卷积与池化的静态输出空间维度。
pub fn infer_spatial_dim(
  in_dim : Int32,
  k : Int32,
  stride : Int32,
  dilation : Int32,
  pad_begin : Int32,
  pad_end : Int32,
  auto_pad : @nn_ir.AutoPad,
  ceil_mode : Bool,
) -> Int32 {
  let eff_k = (k - 1) * dilation + 1
  match auto_pad {
    @nn_ir.Valid => (in_dim - eff_k) / stride + 1
    @nn_ir.SameUpper | @nn_ir.SameLower => (in_dim + stride - 1) / stride
    @nn_ir.NotSet => {
      let num = in_dim + pad_begin + pad_end - eff_k
      if ceil_mode {
        (num + stride - 1) / stride + 1
      } else {
        num / stride + 1
      }
    }
  }
}

///|
/// 编译期解析 2D 卷积/池化最终四向填充 `(pad_top, pad_left, pad_bottom, pad_right)`，
/// 供 AOT 静态代码生成消除运行期 `AutoPad` 分支与数组分配。
pub fn resolve_2d_pads(
  ih : Int32,
  iw : Int32,
  kh : Int32,
  kw : Int32,
  sh : Int32,
  sw : Int32,
  dh : Int32,
  dw : Int32,
  pads : Array[Int32],
  auto_pad : @nn_ir.AutoPad,
  oh : Int32,
  ow : Int32,
) -> (Int32, Int32, Int32, Int32) {
  let eff_kh = (kh - 1) * dh + 1
  let eff_kw = (kw - 1) * dw + 1
  match auto_pad {
    @nn_ir.NotSet => (pads[0], pads[1], pads[2], pads[3])
    @nn_ir.Valid => (0, 0, 0, 0)
    @nn_ir.SameUpper | @nn_ir.SameLower => {
      let th = (oh - 1) * sh + eff_kh - ih
      let tw = (ow - 1) * sw + eff_kw - iw
      let pad_h = if th > 0 { th } else { 0 }
      let pad_w = if tw > 0 { tw } else { 0 }
      match auto_pad {
        @nn_ir.SameUpper =>
          (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
        _ => (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2)
      }
    }
  }
}
