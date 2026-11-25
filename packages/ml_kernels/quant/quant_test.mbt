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

///|
/// 验证 `dynamic_quantize_linear_u8` 与 4 列展开 `V128` `MatMulInteger`（含零/非零每列 `b_zp`）与标量参考逐位一致。
test "DynamicQuantizeLinear and V128 MatMulInteger match scalar reference bit-for-bit" {
  // 1. DynamicQuantizeLinear：值域 [-2.0, 6.0]，scale = 8/255，zp = round(2 / (8/255)) = 64
  let src : FixedArray[Float] = [-2.0, 0.0, 2.0, 6.0]
  let dst = FixedArray::make(4, b'\x00')
  let (sc, zp) = @quant.dynamic_quantize_linear_u8(src, 0, dst, 0, 4)
  assert_true((sc - Float::from_double(8.0 / 255.0)).abs() < (1.0e-6 : Float))
  assert_eq(zp.to_int(), 64)
  assert_eq(dst[0].to_int(), 0)
  assert_eq(dst[1].to_int(), 64)
  assert_eq(dst[3].to_int(), 255)

  // 2. MatMulInteger：非 16/4 整倍数尺寸 [5, 37] @ [37, 11]，验证尾部处理与每列非零 b_zp
  let m = 5
  let k = 37
  let n = 11
  let a = FixedArray::makei(m * k, fn(i) { (i * 13 % 256).to_byte() })
  let b = FixedArray::makei(k * n, fn(i) {
    @quant.i8_to_byte(i * 17 % 200 - 100)
  })
  let a_zp = (123).to_byte()
  let b_zp = FixedArray::makei(n, fn(j) { @quant.i8_to_byte(j - 5) })
  let packed = @quant.pack_int8_matrix_b(b, k, n)

  let out_i32 = FixedArray::make(m * n, 0)
  let out_f32_v128 = FixedArray::make(m * n, (0.0 : Float))
  let out_f32_sc = FixedArray::make(m * n, (0.0 : Float))
  @quant.matmul_integer_u8_i8_prepacked_v128(
    a,
    0,
    a_zp,
    packed,
    Some(b_zp),
    out_i32,
    0,
    m,
  )
  @quant.matmul_integer_u8_i8_prepacked_to_f32_v128(
    a,
    0,
    a_zp,
    packed,
    Some(b_zp),
    out_f32_v128,
    0,
    m,
  )
  @quant.matmul_integer_u8_i8_prepacked_to_f32_scalar(
    a,
    0,
    a_zp,
    packed,
    Some(b_zp),
    out_f32_sc,
    0,
    m,
  )

  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut ref_acc = 0
      let bz = @quant.byte_to_i8(b_zp[j])
      for kk = 0; kk < k; kk = kk + 1 {
        ref_acc = ref_acc +
          (a[i * k + kk].to_int() - 123) *
          (@quant.byte_to_i8(b[kk * n + j]) - bz)
      }
      assert_eq(out_i32[i * n + j], ref_acc)
      assert_eq(out_f32_v128[i * n + j], Float::from_int(ref_acc))
      assert_eq(out_f32_sc[i * n + j], Float::from_int(ref_acc))
    }
  }
}
