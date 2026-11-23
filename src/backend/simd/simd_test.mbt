///|
/// 验证 V128 SIMD 版 Add / Sub / Mul / Relu 在非 4 倍数长度（含尾部标量）下与标量内核完全一致。
test "V128 SIMD elementwise and Relu match scalar reference exactly" {
  // 长度 11（= 2 * 4 + 3 尾部），同时测试主循环与标量尾部
  let a = @tensor.Tensor::from_f32_array([11], [
    -3.5, -1.0, 0.0, 1.25, 2.5, -0.1, 4.0, 5.5, -9.0, 0.5, 7.25,
  ])
  let b = @tensor.Tensor::from_f32_array([11], [
    1.5, 2.0, -0.5, 3.75, -1.5, 0.1, 2.0, -2.5, 10.0, -0.5, 0.75,
  ])

  let s_add = @ops.add_f32(a, b).to_f32_buffer()
  let v_add = @simd.add_f32_simd(a, b).to_f32_buffer()
  let s_sub = @ops.sub_f32(a, b).to_f32_buffer()
  let v_sub = @simd.sub_f32_simd(a, b).to_f32_buffer()
  let s_mul = @ops.mul_f32(a, b).to_f32_buffer()
  let v_mul = @simd.mul_f32_simd(a, b).to_f32_buffer()
  let s_relu = @ops.relu_f32(a).to_f32_buffer()
  let v_relu = @simd.relu_f32_simd(a).to_f32_buffer()

  for i = 0; i < 11; i = i + 1 {
    assert_eq(v_add[i], s_add[i])
    assert_eq(v_sub[i], s_sub[i])
    assert_eq(v_mul[i], s_mul[i])
    assert_eq(v_relu[i], s_relu[i])
  }
}

///|
/// 验证字节缓冲零转换版 `add_bytes_v128` 与 `relu_bytes_v128` 正确性。
test "V128 byte-packed add and relu match scalar values" {
  let a = FixedArray::makei(10, fn(i) { Float::from_int(i - 5) * (0.5 : Float) })
  let b = FixedArray::makei(10, fn(i) { Float::from_int(i) * (0.25 : Float) })
  let ba = @simd.pack_f32_to_bytes(a)
  let bb = @simd.pack_f32_to_bytes(b)
  let out_add = FixedArray::make(40, b'\x00')
  let out_relu = FixedArray::make(40, b'\x00')

  @simd.add_bytes_v128(ba, bb, out_add, 10)
  @simd.relu_bytes_v128(ba, out_relu, 10)

  let res_add = @simd.unpack_bytes_to_f32(out_add, 10)
  let res_relu = @simd.unpack_bytes_to_f32(out_relu, 10)
  for i = 0; i < 10; i = i + 1 {
    assert_eq(res_add[i], a[i] + b[i])
    let exp_relu = if a[i] > (0.0 : Float) { a[i] } else { (0.0 : Float) }
    assert_eq(res_relu[i], exp_relu)
  }
}
