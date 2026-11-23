///|
/// 测试标准 2D MatMul 与 Gemm（含 transB=true、alpha/beta 与 1D 偏置广播）。
test "matmul_f32 and gemm_f32 numerical correctness" {
  // A = [[1, 2, 3], [4, 5, 6]] (2x3)
  let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  // B = [[7, 8], [9, 10], [11, 12]] (3x2)
  let b = @tensor.Tensor::from_f32_array([3, 2], [
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
  ])
  let mm = @ops.matmul_f32(a, b)
  assert_eq(mm.shape(), [2, 2])
  // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
  // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
  assert_eq(mm.get_f32([0, 0]), (58.0 : Float))
  assert_eq(mm.get_f32([0, 1]), (64.0 : Float))
  assert_eq(mm.get_f32([1, 0]), (139.0 : Float))
  assert_eq(mm.get_f32([1, 1]), (154.0 : Float))

  // Gemm with transB=true (B^T shape = [2, 3]), alpha=0.5, beta=2.0, bias=[1, 2]
  let bt = b.transpose([1, 0]).contiguous()
  let bias = @tensor.Tensor::from_f32_array([2], [1.0, 2.0])
  let g = @ops.gemm_f32(
    a,
    bt,
    Some(bias),
    alpha=(0.5 : Float),
    beta=(2.0 : Float),
    trans_b=true,
  )
  assert_eq(g.shape(), [2, 2])
  assert_eq(g.get_f32([0, 0]), (31.0 : Float)) // 0.5*58 + 2*1
  assert_eq(g.get_f32([0, 1]), (36.0 : Float)) // 0.5*64 + 2*2
  assert_eq(g.get_f32([1, 0]), (71.5 : Float)) // 0.5*139 + 2*1
  assert_eq(g.get_f32([1, 1]), (81.0 : Float)) // 0.5*154 + 2*2
}

///|
/// 测试高维批量 MatMul（含 batch 广播）以及 1D x 1D 向量点积。
test "matmul_f32 supports batched broadcasting and 1D dot product" {
  // [2, 1, 2] @ [1, 2, 2] -> [2, 1, 2]
  let a = @tensor.Tensor::from_f32_array([2, 1, 2], [1.0, 2.0, 3.0, 4.0])
  let b = @tensor.Tensor::from_f32_array([1, 2, 2], [1.0, 0.0, 0.0, 2.0])
  let y = @ops.matmul_f32(a, b)
  assert_eq(y.shape(), [2, 1, 2])
  assert_eq(y.get_f32([0, 0, 0]), (1.0 : Float))
  assert_eq(y.get_f32([0, 0, 1]), (4.0 : Float))
  assert_eq(y.get_f32([1, 0, 0]), (3.0 : Float))
  assert_eq(y.get_f32([1, 0, 1]), (8.0 : Float))

  let v1 = @tensor.Tensor::from_f32_array([3], [1.0, 2.0, 3.0])
  let v2 = @tensor.Tensor::from_f32_array([3], [4.0, 5.0, 6.0])
  let dot = @ops.matmul_f32(v1, v2)
  assert_eq(dot.shape(), [])
  assert_eq(dot.get_f32([]), (32.0 : Float))
}
