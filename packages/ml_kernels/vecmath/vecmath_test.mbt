///|
/// 验证字节缓冲版与浮点数组版 V128 vecmath 在非 4 倍数长度下与标量计算逐位一致。
test "vecmath V128 add/sub/mul/relu match scalar reference" {
  let n = 11
  let a = FixedArray::makei(n, fn(i) { Float::from_int(i - 5) * (0.75 : Float) })
  let b = FixedArray::makei(n, fn(i) { Float::from_int(i - 3) * (0.5 : Float) })

  // Float 数组版
  let fa_add = FixedArray::make(n, (0.0 : Float))
  let fa_sub = FixedArray::make(n, (0.0 : Float))
  let fa_mul = FixedArray::make(n, (0.0 : Float))
  let fa_relu = FixedArray::make(n, (0.0 : Float))
  @vecmath.add_f32_v128(a, b, fa_add, n)
  @vecmath.sub_f32_v128(a, b, fa_sub, n)
  @vecmath.mul_f32_v128(a, b, fa_mul, n)
  @vecmath.relu_f32_v128(a, fa_relu, n)

  // 字节缓冲零转换版
  let ba = @vecmath.pack_f32_to_bytes(a)
  let bb = @vecmath.pack_f32_to_bytes(b)
  let b_add = FixedArray::make(n * 4, b'\x00')
  let b_sub = FixedArray::make(n * 4, b'\x00')
  let b_mul = FixedArray::make(n * 4, b'\x00')
  let b_relu = FixedArray::make(n * 4, b'\x00')
  @vecmath.add_bytes_v128(ba, bb, b_add, n)
  @vecmath.sub_bytes_v128(ba, bb, b_sub, n)
  @vecmath.mul_bytes_v128(ba, bb, b_mul, n)
  @vecmath.relu_bytes_v128(ba, b_relu, n)

  let u_add = @vecmath.unpack_bytes_to_f32(b_add, n)
  let u_sub = @vecmath.unpack_bytes_to_f32(b_sub, n)
  let u_mul = @vecmath.unpack_bytes_to_f32(b_mul, n)
  let u_relu = @vecmath.unpack_bytes_to_f32(b_relu, n)

  for i = 0; i < n; i = i + 1 {
    let exp_relu = if a[i] > (0.0 : Float) { a[i] } else { (0.0 : Float) }
    assert_eq(fa_add[i], a[i] + b[i])
    assert_eq(fa_sub[i], a[i] - b[i])
    assert_eq(fa_mul[i], a[i] * b[i])
    assert_eq(fa_relu[i], exp_relu)
    assert_eq(u_add[i], a[i] + b[i])
    assert_eq(u_sub[i], a[i] - b[i])
    assert_eq(u_mul[i], a[i] * b[i])
    assert_eq(u_relu[i], exp_relu)
  }
}
