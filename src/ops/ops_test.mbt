///|
/// 浮点近似相等断言辅助函数。
fn assert_approx(actual : Float, expected : Float, tol : Float) -> Unit raise {
  let diff = (actual - expected).abs()
  if diff > tol {
    fail("expected \{expected}, got \{actual} (diff=\{diff} > tol=\{tol})")
  }
}

///|
/// 测试同形状 Add / Sub / Mul 以及 4D+3D / 2D+1D 偏置广播。
test "add, sub, mul with same-shape and broadcasting" {
  let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = @tensor.Tensor::from_f32_array([3], [10.0, 20.0, 30.0])

  let added = @ops.add_f32(a, b)
  assert_eq(added.shape(), [2, 3])
  assert_eq(added.get_f32([0, 0]), (11.0 : Float))
  assert_eq(added.get_f32([1, 2]), (36.0 : Float))

  let subbed = @ops.sub_f32(added, a)
  assert_eq(subbed.get_f32([0, 1]), (20.0 : Float))
  assert_eq(subbed.get_f32([1, 2]), (30.0 : Float))

  // 模拟 CNN 卷积输出 [1, 2, 2, 2] 加通道偏置 [2, 1, 1]
  let feat = @tensor.Tensor::from_f32_array([1, 2, 2, 2], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
  ])
  let bias = @tensor.Tensor::from_f32_array([2, 1, 1], [10.0, 100.0])
  let cnn_out = @ops.add_f32(feat, bias)
  assert_eq(cnn_out.shape(), [1, 2, 2, 2])
  assert_eq(cnn_out.get_f32([0, 0, 1, 1]), (14.0 : Float))
  assert_eq(cnn_out.get_f32([0, 1, 0, 0]), (105.0 : Float))

  let scaled = @ops.mul_f32(feat, bias)
  assert_eq(scaled.get_f32([0, 1, 1, 1]), (800.0 : Float))
}

///|
/// 测试 Relu 对正、零、负值的截断行为。
test "relu clamps negative values to zero" {
  let x = @tensor.Tensor::from_f32_array([4], [-3.5, 0.0, 2.25, -0.001])
  let y = @ops.relu_f32(x)
  assert_eq(y.get_f32([0]), (0.0 : Float))
  assert_eq(y.get_f32([1]), (0.0 : Float))
  assert_eq(y.get_f32([2]), (2.25 : Float))
  assert_eq(y.get_f32([3]), (0.0 : Float))
}

///|
/// 测试 Softmax 的概率归一化与大数值稳定性（防止 exp 上溢）。
test "softmax normalizes probabilities and resists overflow" {
  // 大数值输入 [1000, 1000, 1000] 应稳定输出均匀分布 1/3
  let big = @tensor.Tensor::from_f32_array([1, 3], [1000.0, 1000.0, 1000.0])
  let p = @ops.softmax_f32(big, -1)
  assert_approx(p.get_f32([0, 0]), (0.33333334 : Float), (1.0e-5 : Float))
  assert_approx(p.get_f32([0, 1]), (0.33333334 : Float), (1.0e-5 : Float))
  assert_approx(p.get_f32([0, 2]), (0.33333334 : Float), (1.0e-5 : Float))

  // 测试沿 axis=0（列方向）计算 Softmax
  let m = @tensor.Tensor::from_f32_array([2, 2], [0.0, 1.0, 0.0, 1.0])
  let col_p = @ops.softmax_f32(m, 0)
  assert_approx(col_p.get_f32([0, 0]), (0.5 : Float), (1.0e-6 : Float))
  assert_approx(col_p.get_f32([1, 1]), (0.5 : Float), (1.0e-6 : Float))
}
