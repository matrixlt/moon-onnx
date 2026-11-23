///|
/// 将 Int64 / Int32 类型的 1D 张量提取为 Int 数组（供 Reshape / Squeeze / Unsqueeze 使用）。
fn tensor_to_int_array(
  t : @tensor.Tensor,
) -> Array[Int32] raise @tensor.TensorError {
  let buf = t.to_i64_buffer()
  Array::makei(buf.length(), fn(i) { buf[i].to_int() })
}

///|
/// ONNX Reshape 算子：支持 `-1` 自动推导与 `0` 复制维度。
pub fn reshape_op(
  data : @tensor.Tensor,
  shape_tensor : @tensor.Tensor,
  allowzero~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  let target_dims = tensor_to_int_array(shape_tensor)
  data.reshape(target_dims, allowzero~)
}

///|
/// ONNX Flatten 算子：按 axis 展平为 2D 矩阵。
pub fn flatten_op(
  data : @tensor.Tensor,
  axis~ : Int32 = 1,
) -> @tensor.Tensor raise @tensor.TensorError {
  data.flatten(axis~)
}

///|
/// ONNX Transpose 算子：空 perm 按规范默认反转全部维度。
pub fn transpose_op(
  data : @tensor.Tensor,
  perm : Array[Int32],
) -> @tensor.Tensor raise @tensor.TensorError {
  let resolved_perm = if perm.length() == 0 {
    let r = data.rank()
    Array::makei(r, fn(i) { r - 1 - i })
  } else {
    perm
  }
  data.transpose(resolved_perm)
}

///|
/// ONNX Squeeze 算子（兼容 Opset < 13 属性与 Opset >= 13 输入张量）。
pub fn squeeze_op(
  data : @tensor.Tensor,
  attr_axes : Array[Int32]?,
  axes_tensor : @tensor.Tensor?,
) -> @tensor.Tensor raise @tensor.TensorError {
  let resolved : Array[Int32]? = match axes_tensor {
    Some(t) => Some(tensor_to_int_array(t))
    None => attr_axes
  }
  data.squeeze(resolved)
}

///|
/// ONNX Unsqueeze 算子（兼容 Opset < 13 属性与 Opset >= 13 输入张量）。
pub fn unsqueeze_op(
  data : @tensor.Tensor,
  attr_axes : Array[Int32]?,
  axes_tensor : @tensor.Tensor?,
) -> @tensor.Tensor raise @tensor.TensorError {
  let resolved = match axes_tensor {
    Some(t) => tensor_to_int_array(t)
    None =>
      match attr_axes {
        Some(a) => a
        None =>
          raise @tensor.InvalidAxis(
            "unsqueeze_op: missing required axes attribute or input tensor",
          )
      }
  }
  data.unsqueeze(resolved)
}
