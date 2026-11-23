///|
/// Tensor 基础构造、步长计算与索引读取测试。
test "tensor creation, strides, and indexing" {
  let t = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  assert_eq(t.rank(), 2)
  assert_eq(t.numel(), 6)
  assert_eq(t.shape(), [2, 3])
  assert_eq(t.strides(), [3, 1])
  assert_true(t.is_contiguous())
  assert_eq(t.get_f32([0, 2]), (3.0 : Float))
  assert_eq(t.get_f32([1, 0]), (4.0 : Float))
  assert_eq(t.summary(), "Tensor(f32, shape=[2, 3], numel=6)")
}

///|
/// 验证 Transpose 零拷贝视图与 contiguous 连续化结果完全一致。
test "transpose zero-copy view and contiguous materialization" {
  let t = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let tr = t.transpose([1, 0])
  assert_eq(tr.shape(), [3, 2])
  assert_eq(tr.strides(), [1, 3])
  assert_false(tr.is_contiguous())
  // 转置后的矩阵为 [[1, 4], [2, 5], [3, 6]]
  assert_eq(tr.get_f32([0, 1]), (4.0 : Float))
  assert_eq(tr.get_f32([2, 0]), (3.0 : Float))
  assert_eq(tr.get_f32_flat(1), (4.0 : Float))
  assert_eq(tr.get_f32_flat(4), (3.0 : Float))

  let c = tr.contiguous()
  assert_true(c.is_contiguous())
  let buf = c.to_f32_buffer()
  assert_eq(buf[0], (1.0 : Float))
  assert_eq(buf[1], (4.0 : Float))
  assert_eq(buf[2], (2.0 : Float))
  assert_eq(buf[3], (5.0 : Float))
  assert_eq(buf[4], (3.0 : Float))
  assert_eq(buf[5], (6.0 : Float))
}

///|
/// 验证 ONNX Reshape 的 -1 推导、0 维复制与 Flatten 语义。
test "reshape with -1 and 0, plus flatten" {
  let t = @tensor.Tensor::from_f32_array([2, 3, 2], [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
  ])
  let r1 = t.reshape([0, -1])
  assert_eq(r1.shape(), [2, 6])
  let r2 = t.reshape([-1, 4])
  assert_eq(r2.shape(), [3, 4])

  let f = t.flatten(axis=1)
  assert_eq(f.shape(), [2, 6])
  let f0 = t.flatten(axis=0)
  assert_eq(f0.shape(), [1, 12])
}

///|
/// 验证 Squeeze / Unsqueeze（含负轴）。
test "squeeze and unsqueeze" {
  let t = @tensor.Tensor::zeros_f32([2, 3])
  let u = t.unsqueeze([0, -1])
  assert_eq(u.shape(), [1, 2, 3, 1])
  assert_true(u.is_contiguous())

  let s1 = u.squeeze(Some([0]))
  assert_eq(s1.shape(), [2, 3, 1])
  let s_all = u.squeeze(None)
  assert_eq(s_all.shape(), [2, 3])
}

///|
/// 验证 NumPy / ONNX 多向广播形状与虚拟步长计算。
test "broadcast shape and strides" {
  let out = @tensor.broadcast_shape([1, 8, 28, 28], [8, 1, 1])
  assert_eq(out, [1, 8, 28, 28])

  let bias_strides = @tensor.compute_strides([8, 1, 1])
  let b_strides = @tensor.broadcast_strides([8, 1, 1], bias_strides, out)
  // 第 0 轴前导补齐步长为 0，第 1 轴通道维度步长为 1，第 2/3 轴广播步长为 0
  assert_eq(b_strides, [0, 1, 0, 0])
}

///|
/// 验证 argmax_f32 正确返回最大类别下标及对应分值。
test "argmax_f32 finds peak logit" {
  let logits = @tensor.Tensor::from_f32_array([1, 4], [-1.5, 0.2, 3.75, 1.1])
  let (idx, val) = logits.argmax_f32()
  assert_eq(idx, 2)
  assert_eq(val, (3.75 : Float))
}
