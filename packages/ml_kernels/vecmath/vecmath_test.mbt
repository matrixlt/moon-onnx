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

///|
/// 验证所有 `*_f32_v128` 向量内核在共享 Arena 缓冲区上使用非零 `a_off`/`b_off`/`src_off`/`out_off` 偏移时结果精确且不越界污染相邻槽位。
test "vecmath V128 Arena non-zero offsets" {
  let n = 9
  let arena = FixedArray::make(48, (99.0 : Float))
  let a_off = 3
  let b_off = 15
  let out_off = 28
  for i = 0; i < n; i = i + 1 {
    arena[a_off + i] = Float::from_int(i - 4) * (0.5 : Float) + (2.5 : Float)
    arena[b_off + i] = Float::from_int(i + 1) * (0.25 : Float)
  }

  @vecmath.add_f32_v128(arena, arena, arena, n, a_off~, b_off~, out_off~)
  for i = 0; i < n; i = i + 1 {
    assert_eq(arena[out_off + i], arena[a_off + i] + arena[b_off + i])
  }
  // 哨兵位置未被污染
  assert_eq(arena[out_off - 1], (99.0 : Float))
  assert_eq(arena[out_off + n], (99.0 : Float))

  @vecmath.div_f32_v128(arena, arena, arena, n, a_off~, b_off~, out_off~)
  for i = 0; i < n; i = i + 1 {
    let diff = (arena[out_off + i] - arena[a_off + i] / arena[b_off + i])
      .to_double()
      .abs()
    assert_true(diff < 1.0e-6)
  }

  @vecmath.clip_f32_v128(
    arena,
    arena,
    n,
    (1.0 : Float),
    (3.0 : Float),
    src_off=a_off,
    out_off~,
  )
  for i = 0; i < n; i = i + 1 {
    let x = arena[a_off + i]
    let exp = if x < (1.0 : Float) {
      (1.0 : Float)
    } else if x > (3.0 : Float) {
      (3.0 : Float)
    } else {
      x
    }
    assert_eq(arena[out_off + i], exp)
  }

  @vecmath.sqrt_f32_v128(arena, arena, n, src_off=a_off, out_off~)
  for i = 0; i < n; i = i + 1 {
    let exp = arena[a_off + i].to_double().sqrt()
    let diff = (arena[out_off + i].to_double() - exp).abs()
    assert_true(diff < 1.0e-5)
  }

  @vecmath.sigmoid_f32_v128(arena, arena, n, src_off=a_off, out_off~)
  for i = 0; i < n; i = i + 1 {
    let x = arena[a_off + i].to_double()
    let exp = 1.0 / (1.0 + @math.exp(-x))
    let diff = (arena[out_off + i].to_double() - exp).abs()
    assert_true(diff < 1.0e-5)
  }

  @vecmath.hard_swish_f32_v128(arena, arena, n, src_off=a_off, out_off~)
  for i = 0; i < n; i = i + 1 {
    let x = arena[a_off + i].to_double()
    let t = x / 6.0 + 0.5
    let c = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t }
    let diff = (arena[out_off + i].to_double() - x * c).abs()
    assert_true(diff < 1.0e-5)
  }
  assert_eq(arena[out_off - 1], (99.0 : Float))
  assert_eq(arena[out_off + n], (99.0 : Float))
}
