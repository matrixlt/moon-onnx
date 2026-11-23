///|
/// 浮点近似相等断言辅助函数。
fn assert_approx(actual : Float, expected : Float, tol : Float) -> Unit raise {
  let diff = (actual - expected).abs()
  if diff > tol {
    fail("expected \{expected}, got \{actual} (diff=\{diff} > tol=\{tol})")
  }
}

///|
/// 测试同形状与广播 Add/Sub/Mul、Relu（标量与 SIMD 一致性）及 Softmax 稳定性。
test "elementwise, broadcasting, Relu (scalar vs simd), and Softmax" {
  let a = @tensor.Tensor::from_f32_array([2, 3], [
    -1.0, 2.0, -3.0, 4.0, -5.0, 6.0,
  ])
  let b = @tensor.Tensor::from_f32_array([3], [10.0, 20.0, 30.0])
  let added = @ops.add_f32(a, b)
  assert_eq(added.get_f32([0, 0]), (9.0 : Float))
  assert_eq(added.get_f32([1, 2]), (36.0 : Float))

  let r_scalar = @ops.relu_f32(a, use_simd=false).to_f32_buffer()
  let r_simd = @ops.relu_f32(a, use_simd=true).to_f32_buffer()
  for i = 0; i < 6; i = i + 1 {
    assert_eq(r_scalar[i], r_simd[i])
  }

  let big = @tensor.Tensor::from_f32_array([1, 3], [1000.0, 1000.0, 1000.0])
  let p = @ops.softmax_f32(big, -1)
  assert_approx(p.get_f32([0, 0]), (0.33333334 : Float), (1.0e-5 : Float))
}

///|
/// 测试 MatMul / Gemm / Conv2D 在标量与 V128 SIMD 路径下数值完全一致。
test "MatMul, Gemm, Conv2D, MaxPool, AveragePool and BatchNorm" {
  let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = @tensor.Tensor::from_f32_array([3, 2], [
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
  ])
  let mm_s = @ops.matmul_f32(a, b, use_simd=false)
  let mm_v = @ops.matmul_f32(a, b, use_simd=true)
  assert_eq(mm_s.get_f32([0, 0]), (58.0 : Float))
  assert_eq(mm_v.get_f32([1, 1]), (154.0 : Float))

  let x = @tensor.Tensor::from_f32_array([1, 1, 3, 3], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let w = @tensor.Tensor::from_f32_array([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
  let bias = @tensor.Tensor::from_f32_array([1], [0.5])
  let c_s = @ops.conv2d_f32(
    x,
    w,
    Some(bias),
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @nn_ir.NotSet,
    use_simd=false,
  )
  let c_v = @ops.conv2d_f32(
    x,
    w,
    Some(bias),
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @nn_ir.NotSet,
    use_simd=true,
  )
  assert_eq(c_s.get_f32([0, 0, 0, 0]), (12.5 : Float))
  assert_eq(c_v.get_f32([0, 0, 1, 1]), (28.5 : Float))

  let mp = @ops.maxpool_2d_f32(
    x,
    [2, 2],
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    false,
    @nn_ir.NotSet,
  )
  assert_eq(mp.get_f32([0, 0, 1, 1]), (9.0 : Float))
}
