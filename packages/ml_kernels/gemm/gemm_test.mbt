///|
/// 验证非对齐及大尺寸下 `matmul_2d_simd_f32` 与 `matmul_2d_prepacked_simd_f32` 均与标量内核精确对齐。
test "V128 GEMM (on-the-fly and prepacked) matches scalar GEMM across odd dimensions" {
  let shapes = [(1, 17, 10), (3, 13, 7), (4, 16, 8), (13, 29, 71), (32, 64, 65)]
  for idx = 0; idx < shapes.length(); idx = idx + 1 {
    let (m, k, n) = shapes[idx]
    let a = FixedArray::makei(m * k, fn(i) {
      Float::from_int(i % 19 - 9) * (0.125 : Float)
    })
    let b = FixedArray::makei(k * n, fn(i) {
      Float::from_int(i % 23 - 11) * (0.0625 : Float)
    })
    let out_scalar = FixedArray::make(m * n, (0.0 : Float))
    let out_simd = FixedArray::make(m * n, (0.0 : Float))
    let out_prepacked = FixedArray::make(m * n, (0.0 : Float))

    @gemm.matmul_2d_scalar_f32(a, 0, b, 0, out_scalar, 0, m, k, n)
    @gemm.matmul_2d_simd_f32(a, 0, b, 0, out_simd, 0, m, k, n)

    let packed_b = @gemm.pack_matrix_b_f32(b, 0, k, n)
    assert_eq(packed_b.dims(), (k, n))
    @gemm.matmul_2d_prepacked_simd_f32(a, 0, packed_b, out_prepacked, 0, m)

    for i = 0; i < m * n; i = i + 1 {
      assert_true((out_scalar[i] - out_simd[i]).abs() <= (1.0e-5 : Float))
      assert_true((out_simd[i] - out_prepacked[i]).abs() == (0.0 : Float))
    }
  }
}
