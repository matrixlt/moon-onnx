///|
/// 验证 `MatMulNBits` Q4 标量与 V128 奇偶拆分 4 列展开内核在对称/非对称零点、`block_size=32/64`、非对齐列尾与偏置下精确对齐。
test "matmul_nbits_q4_v128_f32 matches scalar across block_size 32/64, zp, and bias" {
  let m = 3
  let k = 64
  let n = 7
  let a = FixedArray::makei(m * k, fn(i) {
    Float::from_int((i * 13 + 5) % 23 - 11) * (0.125 : Float)
  })

  // Case 1: block_size=32, default symmetric zp=8, with bias
  let bs1 = 32
  let nb1 = k / bs1
  let b1 = FixedArray::makei(n * nb1 * (bs1 / 2), fn(i) {
    ((i * 37 + 11) & 0xFF).to_byte()
  })
  let sc1 = FixedArray::makei(n * nb1, fn(i) {
    Float::from_int(i % 7 + 1) * (0.025 : Float)
  })
  let bias1 = FixedArray::makei(n, fn(i) {
    Float::from_int(i - 3) * (0.1 : Float)
  })
  let out_s1 = FixedArray::make(m * n, (0.0 : Float))
  let out_v1 = FixedArray::make(m * n, (0.0 : Float))
  @quant.matmul_nbits_q4_scalar_f32(
    a,
    0,
    b1,
    sc1,
    None,
    None,
    Some(bias1),
    out_s1,
    0,
    m,
    k,
    n,
    bs1,
  )
  @quant.matmul_nbits_q4_v128_f32(
    a,
    0,
    b1,
    sc1,
    None,
    None,
    Some(bias1),
    out_v1,
    0,
    m,
    k,
    n,
    bs1,
  )
  for i = 0; i < m * n; i = i + 1 {
    assert_true((out_s1[i] - out_v1[i]).abs() <= (1.0e-5 : Float))
  }

  // Case 2: block_size=64, packed uint8 zero_points, no bias
  let bs2 = 64
  let nb2 = k / bs2
  let b2 = FixedArray::makei(n * nb2 * (bs2 / 2), fn(i) {
    ((i * 53 + 7) & 0xFF).to_byte()
  })
  let sc2 = FixedArray::makei(n * nb2, fn(i) {
    Float::from_int(i % 5 + 2) * (0.03 : Float)
  })
  let zp2 = FixedArray::makei(n * ((nb2 + 1) / 2), fn(i) {
    ((i * 3 + 5) & 0x0F).to_byte()
  })
  let out_s2 = FixedArray::make(m * n, (0.0 : Float))
  let out_v2 = FixedArray::make(m * n, (0.0 : Float))
  @quant.matmul_nbits_q4_scalar_f32(
    a,
    0,
    b2,
    sc2,
    Some(zp2),
    None,
    None,
    out_s2,
    0,
    m,
    k,
    n,
    bs2,
  )
  @quant.matmul_nbits_q4_v128_f32(
    a,
    0,
    b2,
    sc2,
    Some(zp2),
    None,
    None,
    out_v2,
    0,
    m,
    k,
    n,
    bs2,
  )
  for i = 0; i < m * n; i = i + 1 {
    assert_true((out_s2[i] - out_v2[i]).abs() <= (1.0e-5 : Float))
  }
}
