///|
/// 将连续 Float32 缓冲区按小端序打包为 16 字节对齐的 `FixedArray[Byte]`，
/// 供 `@v128.v128_load` / `@v128.v128_store` 直接以 128-bit（4 x Float32）宽度批量访存。
pub fn pack_f32_to_bytes(src : FixedArray[Float]) -> FixedArray[Byte] {
  let n = src.length()
  let out = FixedArray::make(n * 4, b'\x00')
  let mut i = 0
  // 主循环每次将 4 个 Float 装入一个 V128 后单条 v128_store 写回 16 字节
  while i + 4 <= n {
    let v = @v128.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    @v128.v128_store(out, i * 4, v)
    i = i + 4
  }
  // 处理不足 4 个元素的尾部
  while i < n {
    let bits = src[i].reinterpret_as_uint()
    let off = i * 4
    out[off] = (bits & 0xFFU).to_byte()
    out[off + 1] = ((bits >> 8) & 0xFFU).to_byte()
    out[off + 2] = ((bits >> 16) & 0xFFU).to_byte()
    out[off + 3] = ((bits >> 24) & 0xFFU).to_byte()
    i = i + 1
  }
  out
}

///|
/// 将小端序 `FixedArray[Byte]` 解包为 `FixedArray[Float]`。
pub fn unpack_bytes_to_f32(
  src : FixedArray[Byte],
  count : Int32,
) -> FixedArray[Float] {
  let out = FixedArray::make(count, (0.0 : Float))
  let mut i = 0
  while i + 4 <= count {
    let v = @v128.v128_load(src, i * 4)
    out[i] = @v128.f32x4_extract_lane(v, 0)
    out[i + 1] = @v128.f32x4_extract_lane(v, 1)
    out[i + 2] = @v128.f32x4_extract_lane(v, 2)
    out[i + 3] = @v128.f32x4_extract_lane(v, 3)
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let bits = src[off].to_uint() |
      (src[off + 1].to_uint() << 8) |
      (src[off + 2].to_uint() << 16) |
      (src[off + 3].to_uint() << 24)
    out[i] = Float::reinterpret_from_uint(bits)
    i = i + 1
  }
  out
}

///|
/// 零转换字节缓冲版 V128 向量加法（用于评估纯 `v128_load + f32x4_add + v128_store` 吞吐）。
/// 要求 `a`、`b`、`out` 长度至少为 `count * 4` 字节。
pub fn add_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let va = @v128.v128_load(a, off)
    let vb = @v128.v128_load(b, off)
    @v128.v128_store(out, off, @v128.f32x4_add(va, vb))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let fa = Float::reinterpret_from_uint(
      a[off].to_uint() |
      (a[off + 1].to_uint() << 8) |
      (a[off + 2].to_uint() << 16) |
      (a[off + 3].to_uint() << 24),
    )
    let fb = Float::reinterpret_from_uint(
      b[off].to_uint() |
      (b[off + 1].to_uint() << 8) |
      (b[off + 2].to_uint() << 16) |
      (b[off + 3].to_uint() << 24),
    )
    let bits = (fa + fb).reinterpret_as_uint()
    out[off] = (bits & 0xFFU).to_byte()
    out[off + 1] = ((bits >> 8) & 0xFFU).to_byte()
    out[off + 2] = ((bits >> 16) & 0xFFU).to_byte()
    out[off + 3] = ((bits >> 24) & 0xFFU).to_byte()
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量 Relu（`f32x4_max(v, 0.0)`）。
pub fn relu_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let zero_vec = @v128.f32x4_splat((0.0 : Float))
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @v128.v128_load(src, off)
    // 使用 f32x4_pmax(0.0, v) 直接映射硬件 MAXPS/FMAX，避免 f32x4_max 的 IEEE NaN 分支回退开销
    @v128.v128_store(out, off, @v128.f32x4_pmax(zero_vec, v))
    i = i + 4
  }
  let zero : Float = 0.0
  while i < count {
    let off = i * 4
    let fx = Float::reinterpret_from_uint(
      src[off].to_uint() |
      (src[off + 1].to_uint() << 8) |
      (src[off + 2].to_uint() << 16) |
      (src[off + 3].to_uint() << 24),
    )
    let r = if fx > zero { fx } else { zero }
    let bits = r.reinterpret_as_uint()
    out[off] = (bits & 0xFFU).to_byte()
    out[off + 1] = ((bits >> 8) & 0xFFU).to_byte()
    out[off + 2] = ((bits >> 16) & 0xFFU).to_byte()
    out[off + 3] = ((bits >> 24) & 0xFFU).to_byte()
    i = i + 1
  }
}

///|
/// V128 SIMD 版 ONNX Add 算子：
/// - 同形状张量使用 4-wide `f32x4_add` 向量化主循环 + 尾部标量处理；
/// - 广播形状回退到 `@ops.add_f32` 保持完整 ONNX 多向广播语义。
pub fn add_f32_simd(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.shape() != b.shape() {
    return @ops.add_f32(a, b)
  }
  let a_buf = a.to_f32_buffer()
  let b_buf = b.to_f32_buffer()
  let n = a_buf.length()
  let out = FixedArray::make(n, (0.0 : Float))
  let mut i = 0
  while i + 4 <= n {
    let va = @v128.f32x4_const(
      a_buf[i],
      a_buf[i + 1],
      a_buf[i + 2],
      a_buf[i + 3],
    )
    let vb = @v128.f32x4_const(
      b_buf[i],
      b_buf[i + 1],
      b_buf[i + 2],
      b_buf[i + 3],
    )
    let vr = @v128.f32x4_add(va, vb)
    out[i] = @v128.f32x4_extract_lane(vr, 0)
    out[i + 1] = @v128.f32x4_extract_lane(vr, 1)
    out[i + 2] = @v128.f32x4_extract_lane(vr, 2)
    out[i + 3] = @v128.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = a_buf[i] + b_buf[i]
    i = i + 1
  }
  @tensor.Tensor::from_f32(a.shape(), out)
}

///|
/// V128 SIMD 版 ONNX Sub 算子。
pub fn sub_f32_simd(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.shape() != b.shape() {
    return @ops.sub_f32(a, b)
  }
  let a_buf = a.to_f32_buffer()
  let b_buf = b.to_f32_buffer()
  let n = a_buf.length()
  let out = FixedArray::make(n, (0.0 : Float))
  let mut i = 0
  while i + 4 <= n {
    let va = @v128.f32x4_const(
      a_buf[i],
      a_buf[i + 1],
      a_buf[i + 2],
      a_buf[i + 3],
    )
    let vb = @v128.f32x4_const(
      b_buf[i],
      b_buf[i + 1],
      b_buf[i + 2],
      b_buf[i + 3],
    )
    let vr = @v128.f32x4_sub(va, vb)
    out[i] = @v128.f32x4_extract_lane(vr, 0)
    out[i + 1] = @v128.f32x4_extract_lane(vr, 1)
    out[i + 2] = @v128.f32x4_extract_lane(vr, 2)
    out[i + 3] = @v128.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = a_buf[i] - b_buf[i]
    i = i + 1
  }
  @tensor.Tensor::from_f32(a.shape(), out)
}

///|
/// V128 SIMD 版 ONNX Mul 算子。
pub fn mul_f32_simd(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.shape() != b.shape() {
    return @ops.mul_f32(a, b)
  }
  let a_buf = a.to_f32_buffer()
  let b_buf = b.to_f32_buffer()
  let n = a_buf.length()
  let out = FixedArray::make(n, (0.0 : Float))
  let mut i = 0
  while i + 4 <= n {
    let va = @v128.f32x4_const(
      a_buf[i],
      a_buf[i + 1],
      a_buf[i + 2],
      a_buf[i + 3],
    )
    let vb = @v128.f32x4_const(
      b_buf[i],
      b_buf[i + 1],
      b_buf[i + 2],
      b_buf[i + 3],
    )
    let vr = @v128.f32x4_mul(va, vb)
    out[i] = @v128.f32x4_extract_lane(vr, 0)
    out[i + 1] = @v128.f32x4_extract_lane(vr, 1)
    out[i + 2] = @v128.f32x4_extract_lane(vr, 2)
    out[i + 3] = @v128.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = a_buf[i] * b_buf[i]
    i = i + 1
  }
  @tensor.Tensor::from_f32(a.shape(), out)
}

///|
/// V128 SIMD 版 ONNX Relu 算子：利用 `f32x4_max(v, 0.0)` 一次处理 4 个单精度浮点，
/// 并自动处理非 4 倍数长度的尾部元素。
pub fn relu_f32_simd(
  x : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = FixedArray::make(n, (0.0 : Float))
  let zero_vec = @v128.f32x4_splat((0.0 : Float))
  let mut i = 0
  while i + 4 <= n {
    let vx = @v128.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let vr = @v128.f32x4_pmax(zero_vec, vx)
    out[i] = @v128.f32x4_extract_lane(vr, 0)
    out[i + 1] = @v128.f32x4_extract_lane(vr, 1)
    out[i + 2] = @v128.f32x4_extract_lane(vr, 2)
    out[i + 3] = @v128.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  let zero : Float = 0.0
  while i < n {
    let v = src[i]
    out[i] = if v > zero { v } else { zero }
    i = i + 1
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}
