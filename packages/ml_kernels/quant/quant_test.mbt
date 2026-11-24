///|
/// 验证 V128 SIMD `QLinearMatMul`（Int8 与 UInt8）在非 16 倍数 K=37 下与标量参考实现逐位完全一致，
/// 以及对称 Weight-Only INT8 矩阵乘法的精度与 4× 权重压缩比。
test "V128 QLinearMatMul (i8 & u8) matches scalar bit-exact and W8A32 preserves accuracy" {
  let m = 5
  let k = 37 // 非 16 倍数，完整覆盖 SIMD 主循环 + 标量尾部
  let n = 9

  // 1. Int8 QLinearMatMul
  let a_i8 = FixedArray::makei(m * k, fn(i) { @quant.i8_to_byte(i % 200 - 100) })
  let b_i8 = FixedArray::makei(k * n, fn(i) { @quant.i8_to_byte(i % 150 - 75) })
  let out_i8_sc = FixedArray::make(m * n, b'\x00')
  let out_i8_v128 = FixedArray::make(m * n, b'\x00')
  @quant.qlinear_matmul_i8_scalar(
    a_i8,
    (0.05 : Float),
    @quant.i8_to_byte(-3),
    b_i8,
    (0.04 : Float),
    @quant.i8_to_byte(5),
    (0.25 : Float),
    @quant.i8_to_byte(-2),
    out_i8_sc,
    m,
    k,
    n,
  )
  @quant.qlinear_matmul_i8_v128(
    a_i8,
    (0.05 : Float),
    @quant.i8_to_byte(-3),
    b_i8,
    (0.04 : Float),
    @quant.i8_to_byte(5),
    (0.25 : Float),
    @quant.i8_to_byte(-2),
    out_i8_v128,
    m,
    k,
    n,
  )
  assert_eq(out_i8_v128, out_i8_sc)

  // 2. UInt8 QLinearMatMul
  let a_u8 = FixedArray::makei(m * k, fn(i) { @quant.u8_to_byte(i % 251) })
  let b_u8 = FixedArray::makei(k * n, fn(i) { @quant.u8_to_byte(i * 7 % 251) })
  let out_u8_sc = FixedArray::make(m * n, b'\x00')
  let out_u8_v128 = FixedArray::make(m * n, b'\x00')
  @quant.qlinear_matmul_u8_scalar(
    a_u8,
    (0.03 : Float),
    (128).to_byte(),
    b_u8,
    (0.02 : Float),
    (120).to_byte(),
    (0.2 : Float),
    (128).to_byte(),
    out_u8_sc,
    m,
    k,
    n,
  )
  @quant.qlinear_matmul_u8_v128(
    a_u8,
    (0.03 : Float),
    (128).to_byte(),
    b_u8,
    (0.02 : Float),
    (120).to_byte(),
    (0.2 : Float),
    (128).to_byte(),
    out_u8_v128,
    m,
    k,
    n,
  )
  assert_eq(out_u8_v128, out_u8_sc)

  // 3. Weight-Only INT8 对称量化（4× 体积压缩）精度验证
  let a_f32 = FixedArray::makei(m * k, fn(i) {
    Float::from_int(i % 19 - 9) * (0.15 : Float)
  })
  let w_f32 = FixedArray::makei(k * n, fn(i) {
    Float::from_int(i % 23 - 11) * (0.1 : Float)
  })
  let (w_q, scales) = @quant.quantize_weights_symmetric_i8(w_f32, k, n)
  assert_eq(w_q.length(), k * n) // 1 byte / weight vs 4 bytes / Float32 = 4x 压缩
  let y_w8 = FixedArray::make(m * n, (0.0 : Float))
  @quant.matmul_f32_w8_i8(a_f32, w_q, scales, y_w8, m, k, n)

  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut ref_v : Float = 0.0
      for kk = 0; kk < k; kk = kk + 1 {
        ref_v = ref_v + a_f32[i * k + kk] * w_f32[kk * n + j]
      }
      assert_true((y_w8[i * n + j] - ref_v).abs() < (0.08 : Float))
    }
  }
}
