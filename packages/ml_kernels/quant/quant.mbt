///|
/// 将补码存储的 `Byte` 解释为有符号 8 位整数 `[-128, 127]`（无分支位运算实现）。
pub fn byte_to_i8(b : Byte) -> Int32 {
  (b.to_int() ^ 128) - 128
}

///|
/// 将有符号整数裁剪至 `[-128, 127]` 并按补码编码为 `Byte`。
pub fn i8_to_byte(v : Int32) -> Byte {
  let clamped = if v < -128 { -128 } else if v > 127 { 127 } else { v }
  (clamped & 0xFF).to_byte()
}

///|
/// 将整数裁剪至无符号 8 位范围 `[0, 255]` 并编码为 `Byte`。
pub fn u8_to_byte(v : Int32) -> Byte {
  let clamped = if v < 0 { 0 } else if v > 255 { 255 } else { v }
  clamped.to_byte()
}

///|
/// IEEE 754 银行家舍入（round-half-to-even），严格匹配 ONNX `QuantizeLinear` / `QLinearMatMul` 规范。
pub fn round_half_to_even(x : Double) -> Int32 {
  let fl = x.floor()
  let diff = x - fl
  let base = fl.to_int()
  if diff > 0.5 {
    base + 1
  } else if diff < 0.5 {
    base
  } else if (base & 1) == 0 {
    base
  } else {
    base + 1
  }
}

///|
/// 校验量子化缓冲切片区间 `[off, off + len)` 不越界，供内层 `unsafe_get` / `unsafe_set` 保证内存安全。
fn check_quant_bounds(
  arr_len : Int32,
  off : Int32,
  len : Int32,
  label : String,
) -> Unit {
  if off < 0 || len < 0 || off + len > arr_len {
    abort("ml_kernels/quant: " + label + " out of bounds")
  }
}

///|
/// 逐元素线性量化（`Float32 -> UInt8`）：`y = saturate_u8(round_even(x / scale) + zp)`，支持 Arena 零拷贝偏移。
pub fn quantize_u8_per_tensor(
  src : FixedArray[Float],
  scale : Float,
  zero_point : Byte,
  dst : FixedArray[Byte],
  n : Int32,
  src_off~ : Int32 = 0,
  dst_off~ : Int32 = 0,
) -> Unit {
  if n <= 0 {
    return
  }
  check_quant_bounds(src.length(), src_off, n, "quantize_u8_per_tensor(src)")
  check_quant_bounds(dst.length(), dst_off, n, "quantize_u8_per_tensor(dst)")
  let inv_s = 1.0 / scale.to_double()
  let zp = zero_point.to_int()
  for i = 0; i < n; i = i + 1 {
    let q = round_half_to_even(src.unsafe_get(src_off + i).to_double() * inv_s) +
      zp
    dst.unsafe_set(dst_off + i, u8_to_byte(q))
  }
}

///|
/// 逐元素线性量化（`Float32 -> Int8`）：`y = saturate_i8(round_even(x / scale) + zp)`，支持 Arena 零拷贝偏移。
pub fn quantize_i8_per_tensor(
  src : FixedArray[Float],
  scale : Float,
  zero_point : Byte,
  dst : FixedArray[Byte],
  n : Int32,
  src_off~ : Int32 = 0,
  dst_off~ : Int32 = 0,
) -> Unit {
  if n <= 0 {
    return
  }
  check_quant_bounds(src.length(), src_off, n, "quantize_i8_per_tensor(src)")
  check_quant_bounds(dst.length(), dst_off, n, "quantize_i8_per_tensor(dst)")
  let inv_s = 1.0 / scale.to_double()
  let zp = byte_to_i8(zero_point)
  for i = 0; i < n; i = i + 1 {
    let q = round_half_to_even(src.unsafe_get(src_off + i).to_double() * inv_s) +
      zp
    dst.unsafe_set(dst_off + i, i8_to_byte(q))
  }
}

///|
/// 逐元素线性反量化（`UInt8 -> Float32`）：`y = (x - zp) * scale`，支持 Arena 零拷贝偏移。
pub fn dequantize_u8_per_tensor(
  src : FixedArray[Byte],
  scale : Float,
  zero_point : Byte,
  dst : FixedArray[Float],
  n : Int32,
  src_off~ : Int32 = 0,
  dst_off~ : Int32 = 0,
) -> Unit {
  if n <= 0 {
    return
  }
  check_quant_bounds(src.length(), src_off, n, "dequantize_u8_per_tensor(src)")
  check_quant_bounds(dst.length(), dst_off, n, "dequantize_u8_per_tensor(dst)")
  let s = scale.to_double()
  let zp = zero_point.to_int()
  for i = 0; i < n; i = i + 1 {
    dst.unsafe_set(
      dst_off + i,
      Float::from_double(
        (src.unsafe_get(src_off + i).to_int() - zp).to_double() * s,
      ),
    )
  }
}

///|
/// 逐元素线性反量化（`Int8 -> Float32`）：`y = (x - zp) * scale`，支持 Arena 零拷贝偏移。
pub fn dequantize_i8_per_tensor(
  src : FixedArray[Byte],
  scale : Float,
  zero_point : Byte,
  dst : FixedArray[Float],
  n : Int32,
  src_off~ : Int32 = 0,
  dst_off~ : Int32 = 0,
) -> Unit {
  if n <= 0 {
    return
  }
  check_quant_bounds(src.length(), src_off, n, "dequantize_i8_per_tensor(src)")
  check_quant_bounds(dst.length(), dst_off, n, "dequantize_i8_per_tensor(dst)")
  let s = scale.to_double()
  let zp = byte_to_i8(zero_point)
  for i = 0; i < n; i = i + 1 {
    dst.unsafe_set(
      dst_off + i,
      Float::from_double(
        (byte_to_i8(src.unsafe_get(src_off + i)) - zp).to_double() * s,
      ),
    )
  }
}

///|
/// 将右矩阵 `B[K, N]` 转置为按列连续的 `BT[N, K]`，使 `V128` 16 字节向量加载能够沿 `K` 轴全速连续读取。
pub fn transpose_bytes_kn_to_nk(
  b : FixedArray[Byte],
  k : Int32,
  n : Int32,
) -> FixedArray[Byte] {
  let bt = FixedArray::make(n * k, b'\x00')
  for kk = 0; kk < k; kk = kk + 1 {
    let row = kk * n
    for j = 0; j < n; j = j + 1 {
      bt[j * k + kk] = b[row + j]
    }
  }
  bt
}

///|
/// 提取 `i32x4` 向量中全部 4 个有符号 32 位车道并求水平总和。
fn hsum_i32x4(v : V128) -> Int32 {
  @simd.i32x4_extract_lane(v, 0).reinterpret_as_int() +
  @simd.i32x4_extract_lane(v, 1).reinterpret_as_int() +
  @simd.i32x4_extract_lane(v, 2).reinterpret_as_int() +
  @simd.i32x4_extract_lane(v, 3).reinterpret_as_int()
}

///|
/// ONNX `DynamicQuantizeLinear` 内核：实时扫描 `src[src_off .. +n]` 的值域并量化为 `UInt8`，
/// 返回 `(y_scale, y_zero_point)`，严格遵循 ONNX 规范（值域包含 0.0、银行家舍入）。
pub fn dynamic_quantize_linear_u8(
  src : FixedArray[Float],
  src_off : Int32,
  dst : FixedArray[Byte],
  dst_off : Int32,
  n : Int32,
) -> (Float, Byte) {
  if n <= 0 {
    return ((1.0 : Float), b'\x00')
  }
  check_quant_bounds(
    src.length(),
    src_off,
    n,
    "dynamic_quantize_linear_u8(src)",
  )
  check_quant_bounds(
    dst.length(),
    dst_off,
    n,
    "dynamic_quantize_linear_u8(dst)",
  )
  let mut min_v = 0.0
  let mut max_v = 0.0
  for i = 0; i < n; i = i + 1 {
    let v = src.unsafe_get(src_off + i).to_double()
    if v < min_v {
      min_v = v
    }
    if v > max_v {
      max_v = v
    }
  }
  let range = max_v - min_v
  if range <= 0.0 {
    for i = 0; i < n; i = i + 1 {
      dst.unsafe_set(dst_off + i, b'\x00')
    }
    return ((1.0 : Float), b'\x00')
  }
  let scale_d = range / 255.0
  let scale = Float::from_double(scale_d)
  let zp_i = round_half_to_even(-min_v / scale.to_double())
  let zp_clamped = if zp_i < 0 { 0 } else if zp_i > 255 { 255 } else { zp_i }
  let zp_byte = zp_clamped.to_byte()
  let inv_s = 1.0 / scale.to_double()
  for i = 0; i < n; i = i + 1 {
    let q = round_half_to_even(src.unsafe_get(src_off + i).to_double() * inv_s) +
      zp_clamped
    dst.unsafe_set(dst_off + i, u8_to_byte(q))
  }
  (scale, zp_byte)
}
