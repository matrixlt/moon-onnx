///|
/// 测试 SymDim / SymShape 的静态判定、格式化与运行期符号绑定解析。
test "SymShape formats and resolves symbolic dimensions" {
  let s = @tensor.SymShape::from_dims([
    @tensor.Sym("batch"),
    @tensor.Static(1),
    @tensor.Static(28),
    @tensor.Static(28),
  ])
  assert_false(s.is_fully_static())
  assert_eq(s.format(), "[batch, 1, 28, 28]")

  let bindings : Map[String, Int32] = Map([("batch", 8)])
  assert_eq(s.resolve(bindings), [8, 1, 28, 28])

  let static_s = @tensor.SymShape::from_static([1, 10])
  assert_true(static_s.is_fully_static())
  assert_eq(static_s.resolve(Map([])), [1, 10])
}

///|
/// 测试 BufferPool 的缓存命中、容量上限控制与清零重置行为。
test "BufferPool reuses released f32 and byte buffers" {
  let pool = @tensor.BufferPool::new(max_per_bucket=2)
  let b1 = pool.acquire_f32(16)
  b1[0] = (42.0 : Float)
  pool.release_f32(b1)

  let b2 = pool.acquire_f32(16)
  // 命中缓存复用了同一底层缓冲区
  assert_eq(b2[0], (42.0 : Float))

  let raw = pool.acquire_bytes(64)
  pool.release_bytes(raw)
  let _ = pool.acquire_bytes(64)

  let st = pool.stats()
  assert_eq(st.misses, 2)
  assert_eq(st.hits, 2)
  assert_eq(st.pooled_f32_buffers, 0)
  assert_eq(st.pooled_byte_buffers, 0)
}

///|
/// 测试 Float32 张量与小端序字节流的无损往返转换。
test "Tensor from_f32_bytes_le and to_f32_bytes_le round-trip" {
  let t = @tensor.Tensor::from_f32_array([2, 2], [-1.5, 0.0, 2.25, 100.5])
  let bytes_arr = t.to_f32_bytes_le()
  let t2 = @tensor.Tensor::from_f32_bytes_le(
    [2, 2],
    Bytes::from_array(bytes_arr),
  )
  assert_eq(t2.get_f32([0, 0]), (-1.5 : Float))
  assert_eq(t2.get_f32([1, 0]), (2.25 : Float))
  assert_eq(t2.get_f32([1, 1]), (100.5 : Float))
}
