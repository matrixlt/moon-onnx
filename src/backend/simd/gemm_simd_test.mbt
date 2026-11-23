///|
/// 测试各种非对齐尺寸（涵盖 4x8 主块、4x4 边块、1x8 与 1x4 尾部行）的 V128 GEMM 与标量内核数值完全对齐。
test "V128 tiled GEMM matches scalar GEMM across odd and large dimensions" {
  let shapes = [(1, 17, 10), (3, 13, 7), (4, 16, 8), (13, 29, 71), (32, 64, 65)]
  for s_idx = 0; s_idx < shapes.length(); s_idx = s_idx + 1 {
    let (m, k, n) = shapes[s_idx]
    let a = FixedArray::makei(m * k, fn(i) {
      Float::from_int(i % 19 - 9) * (0.125 : Float)
    })
    let b = FixedArray::makei(k * n, fn(i) {
      Float::from_int(i % 23 - 11) * (0.0625 : Float)
    })
    let out_scalar = FixedArray::make(m * n, (0.0 : Float))
    let out_simd = FixedArray::make(m * n, (0.0 : Float))

    @ops.matmul_2d_raw_f32(a, 0, b, 0, out_scalar, 0, m, k, n)
    @simd.matmul_2d_raw_simd_f32(a, 0, b, 0, out_simd, 0, m, k, n)

    let mut max_diff : Float = 0.0
    for i = 0; i < m * n; i = i + 1 {
      let d = (out_scalar[i] - out_simd[i]).abs()
      if d > max_diff {
        max_diff = d
      }
    }
    assert_true(max_diff <= (1.0e-5 : Float))
  }
}

///|
/// 测试 V128 Conv2D（im2col + V128 GEMM）与标量 Conv2D 数值对齐。
test "V128 Conv2D matches scalar Conv2D within 1e-5" {
  let x = @tensor.Tensor::from_f32(
    [1, 3, 12, 12],
    FixedArray::makei(3 * 144, fn(i) {
      Float::from_int(i % 17 - 8) * (0.1 : Float)
    }),
  )
  let w = @tensor.Tensor::from_f32(
    [8, 3, 3, 3],
    FixedArray::makei(8 * 27, fn(i) {
      Float::from_int(i % 13 - 6) * (0.05 : Float)
    }),
  )
  let b = @tensor.Tensor::from_f32(
    [8],
    FixedArray::makei(8, fn(i) { Float::from_int(i) * (0.02 : Float) }),
  )

  let y_scalar = @ops.conv2d_f32(
    x,
    w,
    Some(b),
    [1, 1],
    [1, 1, 1, 1],
    [1, 1],
    1,
    @ir.NotSet,
  )
  let y_simd = @simd.conv2d_f32_simd(
    x,
    w,
    Some(b),
    [1, 1],
    [1, 1, 1, 1],
    [1, 1],
    1,
    @ir.NotSet,
  )

  assert_eq(y_simd.shape(), y_scalar.shape())
  let a = y_scalar.to_f32_buffer()
  let c = y_simd.to_f32_buffer()
  for i = 0; i < a.length(); i = i + 1 {
    assert_true((a[i] - c[i]).abs() <= (1.0e-5 : Float))
  }
}
