///|
/// 从字节数组指定小端序偏移处读取单个 Float32（用于非 4 倍数尾部标量收尾）。
fn read_f32_le(bytes : FixedArray[Byte], off : Int32) -> Float {
  let bits = bytes[off].to_uint() |
    (bytes[off + 1].to_uint() << 8) |
    (bytes[off + 2].to_uint() << 16) |
    (bytes[off + 3].to_uint() << 24)
  Float::reinterpret_from_uint(bits)
}

///|
/// 将单个 Float32 按小端序写入字节数组指定偏移处。
fn write_f32_le(bytes : FixedArray[Byte], off : Int32, v : Float) -> Unit {
  let bits = v.reinterpret_as_uint()
  bytes[off] = (bits & 0xFFU).to_byte()
  bytes[off + 1] = ((bits >> 8) & 0xFFU).to_byte()
  bytes[off + 2] = ((bits >> 16) & 0xFFU).to_byte()
  bytes[off + 3] = ((bits >> 24) & 0xFFU).to_byte()
}

///|
/// 将连续 Float32 数组按小端序打包为 16 字节友好的 `FixedArray[Byte]`。
pub fn pack_f32_to_bytes(src : FixedArray[Float]) -> FixedArray[Byte] {
  let n = src.length()
  let out = FixedArray::make(n * 4, b'\x00')
  let mut i = 0
  while i + 4 <= n {
    let v = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    @simd.v128_store(out, i * 4, v)
    i = i + 4
  }
  while i < n {
    write_f32_le(out, i * 4, src[i])
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
    let v = @simd.v128_load(src, i * 4)
    out[i] = @simd.f32x4_extract_lane(v, 0)
    out[i + 1] = @simd.f32x4_extract_lane(v, 1)
    out[i + 2] = @simd.f32x4_extract_lane(v, 2)
    out[i + 3] = @simd.f32x4_extract_lane(v, 3)
    i = i + 4
  }
  while i < count {
    out[i] = read_f32_le(src, i * 4)
    i = i + 1
  }
  out
}

///|
/// 零转换字节缓冲版 V128 向量加法（`out[i] = a[i] + b[i]`）。
pub fn add_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let va = @simd.v128_load(a, off)
    let vb = @simd.v128_load(b, off)
    @simd.v128_store(out, off, @simd.f32x4_add(va, vb))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    write_f32_le(out, off, read_f32_le(a, off) + read_f32_le(b, off))
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量减法（`out[i] = a[i] - b[i]`）。
pub fn sub_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let va = @simd.v128_load(a, off)
    let vb = @simd.v128_load(b, off)
    @simd.v128_store(out, off, @simd.f32x4_sub(va, vb))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    write_f32_le(out, off, read_f32_le(a, off) - read_f32_le(b, off))
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量乘法（`out[i] = a[i] * b[i]`）。
pub fn mul_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let va = @simd.v128_load(a, off)
    let vb = @simd.v128_load(b, off)
    @simd.v128_store(out, off, @simd.f32x4_mul(va, vb))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    write_f32_le(out, off, read_f32_le(a, off) * read_f32_le(b, off))
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量 Relu（使用 `f32x4_pmax(0.0, v)` 直接映射硬件 MAXPS/FMAX）。
pub fn relu_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let zero_vec = @simd.f32x4_splat((0.0 : Float))
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    @simd.v128_store(out, off, @simd.f32x4_pmax(zero_vec, v))
    i = i + 4
  }
  let zero : Float = 0.0
  while i < count {
    let off = i * 4
    let fx = read_f32_le(src, off)
    write_f32_le(out, off, if fx > zero { fx } else { zero })
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量加法内核（自动处理尾部元素）。
pub fn add_f32_v128(
  a : FixedArray[Float],
  b : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  if n <= 0 {
    return
  }
  if n > a.length() || n > b.length() || n > out.length() {
    abort("ml_kernels/vecmath: add_f32_v128 out of bounds")
  }
  let mut i = 0
  while i + 4 <= n {
    let va = @simd.f32x4_const(
      a.unsafe_get(i),
      a.unsafe_get(i + 1),
      a.unsafe_get(i + 2),
      a.unsafe_get(i + 3),
    )
    let vb = @simd.f32x4_const(
      b.unsafe_get(i),
      b.unsafe_get(i + 1),
      b.unsafe_get(i + 2),
      b.unsafe_get(i + 3),
    )
    let vr = @simd.f32x4_add(va, vb)
    out.unsafe_set(i, @simd.f32x4_extract_lane(vr, 0))
    out.unsafe_set(i + 1, @simd.f32x4_extract_lane(vr, 1))
    out.unsafe_set(i + 2, @simd.f32x4_extract_lane(vr, 2))
    out.unsafe_set(i + 3, @simd.f32x4_extract_lane(vr, 3))
    i = i + 4
  }
  while i < n {
    out.unsafe_set(i, a.unsafe_get(i) + b.unsafe_get(i))
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量减法内核。
pub fn sub_f32_v128(
  a : FixedArray[Float],
  b : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  if n <= 0 {
    return
  }
  if n > a.length() || n > b.length() || n > out.length() {
    abort("ml_kernels/vecmath: sub_f32_v128 out of bounds")
  }
  let mut i = 0
  while i + 4 <= n {
    let va = @simd.f32x4_const(
      a.unsafe_get(i),
      a.unsafe_get(i + 1),
      a.unsafe_get(i + 2),
      a.unsafe_get(i + 3),
    )
    let vb = @simd.f32x4_const(
      b.unsafe_get(i),
      b.unsafe_get(i + 1),
      b.unsafe_get(i + 2),
      b.unsafe_get(i + 3),
    )
    let vr = @simd.f32x4_sub(va, vb)
    out.unsafe_set(i, @simd.f32x4_extract_lane(vr, 0))
    out.unsafe_set(i + 1, @simd.f32x4_extract_lane(vr, 1))
    out.unsafe_set(i + 2, @simd.f32x4_extract_lane(vr, 2))
    out.unsafe_set(i + 3, @simd.f32x4_extract_lane(vr, 3))
    i = i + 4
  }
  while i < n {
    out.unsafe_set(i, a.unsafe_get(i) - b.unsafe_get(i))
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量乘法内核。
pub fn mul_f32_v128(
  a : FixedArray[Float],
  b : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  if n <= 0 {
    return
  }
  if n > a.length() || n > b.length() || n > out.length() {
    abort("ml_kernels/vecmath: mul_f32_v128 out of bounds")
  }
  let mut i = 0
  while i + 4 <= n {
    let va = @simd.f32x4_const(
      a.unsafe_get(i),
      a.unsafe_get(i + 1),
      a.unsafe_get(i + 2),
      a.unsafe_get(i + 3),
    )
    let vb = @simd.f32x4_const(
      b.unsafe_get(i),
      b.unsafe_get(i + 1),
      b.unsafe_get(i + 2),
      b.unsafe_get(i + 3),
    )
    let vr = @simd.f32x4_mul(va, vb)
    out.unsafe_set(i, @simd.f32x4_extract_lane(vr, 0))
    out.unsafe_set(i + 1, @simd.f32x4_extract_lane(vr, 1))
    out.unsafe_set(i + 2, @simd.f32x4_extract_lane(vr, 2))
    out.unsafe_set(i + 3, @simd.f32x4_extract_lane(vr, 3))
    i = i + 4
  }
  while i < n {
    out.unsafe_set(i, a.unsafe_get(i) * b.unsafe_get(i))
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 Relu 内核（使用 `f32x4_pmax`）。
pub fn relu_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  if n <= 0 {
    return
  }
  if n > src.length() || n > out.length() {
    abort("ml_kernels/vecmath: relu_f32_v128 out of bounds")
  }
  let zero_vec = @simd.f32x4_splat((0.0 : Float))
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(
      src.unsafe_get(i),
      src.unsafe_get(i + 1),
      src.unsafe_get(i + 2),
      src.unsafe_get(i + 3),
    )
    let vr = @simd.f32x4_pmax(zero_vec, vx)
    out.unsafe_set(i, @simd.f32x4_extract_lane(vr, 0))
    out.unsafe_set(i + 1, @simd.f32x4_extract_lane(vr, 1))
    out.unsafe_set(i + 2, @simd.f32x4_extract_lane(vr, 2))
    out.unsafe_set(i + 3, @simd.f32x4_extract_lane(vr, 3))
    i = i + 4
  }
  let zero : Float = 0.0
  while i < n {
    let v = src.unsafe_get(i)
    out.unsafe_set(i, if v > zero { v } else { zero })
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量除法（`out[i] = a[i] / b[i]`）。
pub fn div_bytes_v128(
  a : FixedArray[Byte],
  b : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let va = @simd.v128_load(a, off)
    let vb = @simd.v128_load(b, off)
    @simd.v128_store(out, off, @simd.f32x4_div(va, vb))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    write_f32_le(out, off, read_f32_le(a, off) / read_f32_le(b, off))
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量平方根（`out[i] = sqrt(src[i])`）。
pub fn sqrt_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    @simd.v128_store(out, off, @simd.f32x4_sqrt(v))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let x = read_f32_le(src, off)
    let r = @simd.f32x4_extract_lane(@simd.f32x4_sqrt(@simd.f32x4_splat(x)), 0)
    write_f32_le(out, off, r)
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量除法内核。
pub fn div_f32_v128(
  a : FixedArray[Float],
  b : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  if n <= 0 {
    return
  }
  if n > a.length() || n > b.length() || n > out.length() {
    abort("ml_kernels/vecmath: div_f32_v128 out of bounds")
  }
  let mut i = 0
  while i + 4 <= n {
    let va = @simd.f32x4_const(
      a.unsafe_get(i),
      a.unsafe_get(i + 1),
      a.unsafe_get(i + 2),
      a.unsafe_get(i + 3),
    )
    let vb = @simd.f32x4_const(
      b.unsafe_get(i),
      b.unsafe_get(i + 1),
      b.unsafe_get(i + 2),
      b.unsafe_get(i + 3),
    )
    let vr = @simd.f32x4_div(va, vb)
    out.unsafe_set(i, @simd.f32x4_extract_lane(vr, 0))
    out.unsafe_set(i + 1, @simd.f32x4_extract_lane(vr, 1))
    out.unsafe_set(i + 2, @simd.f32x4_extract_lane(vr, 2))
    out.unsafe_set(i + 3, @simd.f32x4_extract_lane(vr, 3))
    i = i + 4
  }
  while i < n {
    out.unsafe_set(i, a.unsafe_get(i) / b.unsafe_get(i))
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量平方根内核。
pub fn sqrt_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  if n <= 0 {
    return
  }
  if n > src.length() || n > out.length() {
    abort("ml_kernels/vecmath: sqrt_f32_v128 out of bounds")
  }
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(
      src.unsafe_get(i),
      src.unsafe_get(i + 1),
      src.unsafe_get(i + 2),
      src.unsafe_get(i + 3),
    )
    let vr = @simd.f32x4_sqrt(vx)
    out.unsafe_set(i, @simd.f32x4_extract_lane(vr, 0))
    out.unsafe_set(i + 1, @simd.f32x4_extract_lane(vr, 1))
    out.unsafe_set(i + 2, @simd.f32x4_extract_lane(vr, 2))
    out.unsafe_set(i + 3, @simd.f32x4_extract_lane(vr, 3))
    i = i + 4
  }
  while i < n {
    out.unsafe_set(
      i,
      @simd.f32x4_extract_lane(
        @simd.f32x4_sqrt(@simd.f32x4_splat(src.unsafe_get(i))),
        0,
      ),
    )
    i = i + 1
  }
}
