///|
/// 测试 ONNX Reshape 算子从 Int64 形状张量读取目标维度（含 -1 推导）。
test "reshape_op reads Int64 shape tensor and handles non-contiguous input" {
  let x = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  // 先转置为非连续张量 [3, 2]，再 Reshape 为 [1, -1]
  let xt = @ops.transpose_op(x, [])
  assert_eq(xt.shape(), [3, 2])

  let shape_t = @tensor.Tensor::from_i64([2], [(1 : Int64), (-1 : Int64)])
  let r = @ops.reshape_op(xt, shape_t)
  assert_eq(r.shape(), [1, 6])
  // 转置后行主序应为 [1, 4, 2, 5, 3, 6]
  assert_eq(r.get_f32([0, 1]), (4.0 : Float))
  assert_eq(r.get_f32([0, 2]), (2.0 : Float))
}

///|
/// 测试 Squeeze / Unsqueeze 兼容 Opset >= 13 通过输入张量传递 axes。
test "squeeze_op and unsqueeze_op support axes input tensor (Opset 13+)" {
  let x = @tensor.Tensor::zeros_f32([2, 4])
  let axes_in = @tensor.Tensor::from_i64([2], [(0 : Int64), (-1 : Int64)])

  let u = @ops.unsqueeze_op(x, None, Some(axes_in))
  assert_eq(u.shape(), [1, 2, 4, 1])

  let sq_axes = @tensor.Tensor::from_i64([1], [(0 : Int64)])
  let s = @ops.squeeze_op(u, None, Some(sq_axes))
  assert_eq(s.shape(), [2, 4, 1])

  let flat = @ops.flatten_op(u, axis=1)
  assert_eq(flat.shape(), [1, 8])
}
