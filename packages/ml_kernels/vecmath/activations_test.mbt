///|
/// 验证 V128 快速多项式 `exp_bytes_v128`、`sigmoid_bytes_v128` 与 `tanh_bytes_v128`
/// 在 [-10, 10] 区间内与 `core/math` 标量双精度参考值的精度误差满足深度学习推理标准（< 2e-6）。
test "vecmath V128 exp, sigmoid, tanh match double-precision reference" {
  let n = 41
  let xs = FixedArray::makei(n, fn(i) {
    Float::from_int(i - 20) * (0.5 : Float)
  })
  let bx = @vecmath.pack_f32_to_bytes(xs)
  let out_exp = FixedArray::make(n * 4, b'\x00')
  let out_sig = FixedArray::make(n * 4, b'\x00')
  let out_tanh = FixedArray::make(n * 4, b'\x00')

  @vecmath.exp_bytes_v128(bx, out_exp, n)
  @vecmath.sigmoid_bytes_v128(bx, out_sig, n)
  @vecmath.tanh_bytes_v128(bx, out_tanh, n)

  let r_exp = @vecmath.unpack_bytes_to_f32(out_exp, n)
  let r_sig = @vecmath.unpack_bytes_to_f32(out_sig, n)
  let r_tanh = @vecmath.unpack_bytes_to_f32(out_tanh, n)

  for i = 0; i < n; i = i + 1 {
    let xd = xs[i].to_double()
    let ref_exp = @math.exp(xd)
    let rel_exp = (r_exp[i].to_double() - ref_exp).abs() /
      (if ref_exp > 1.0e-12 { ref_exp } else { 1.0 })
    assert_true(rel_exp < 2.0e-6)

    let ref_sig = 1.0 / (1.0 + @math.exp(-xd))
    assert_true((r_sig[i].to_double() - ref_sig).abs() < 2.0e-6)

    let ref_tanh = @math.tanh(xd)
    assert_true((r_tanh[i].to_double() - ref_tanh).abs() < 2.0e-6)
  }
}

///|
/// 验证 V128 Clip（含字节缓冲与 Float 数组双版本）对上下界截断的精确性。
test "vecmath V128 clip clamps values to [min, max]" {
  let xs = FixedArray::makei(9, fn(i) { Float::from_int(i - 4) * (2.0 : Float) })
  let out_f = FixedArray::make(9, (0.0 : Float))
  @vecmath.clip_f32_v128(xs, out_f, 9, (-3.0 : Float), (3.0 : Float))

  let bx = @vecmath.pack_f32_to_bytes(xs)
  let out_b = FixedArray::make(36, b'\x00')
  @vecmath.clip_bytes_v128(bx, out_b, 9, (-3.0 : Float), (3.0 : Float))
  let ub = @vecmath.unpack_bytes_to_f32(out_b, 9)

  for i = 0; i < 9; i = i + 1 {
    let exp = if xs[i] < (-3.0 : Float) {
      (-3.0 : Float)
    } else if xs[i] > (3.0 : Float) {
      (3.0 : Float)
    } else {
      xs[i]
    }
    assert_eq(out_f[i], exp)
    assert_eq(ub[i], exp)
  }
}
