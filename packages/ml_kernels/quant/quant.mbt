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
/// 标量参考实现：有符号 `Int8` 量化矩阵乘法 `Y[M, N] = QLinearMatMul(A[M, K], B[K, N])`。
pub fn qlinear_matmul_i8_scalar(
  a : FixedArray[Byte],
  a_scale : Float,
  a_zp : Byte,
  b : FixedArray[Byte],
  b_scale : Float,
  b_zp : Byte,
  y_scale : Float,
  y_zp : Byte,
  out : FixedArray[Byte],
  m : Int32,
  k : Int32,
  n : Int32,
  a_off~ : Int32 = 0,
  out_off~ : Int32 = 0,
) -> Unit {
  let az = byte_to_i8(a_zp)
  let bz = byte_to_i8(b_zp)
  let yz = byte_to_i8(y_zp)
  let mul_scale = a_scale.to_double() *
    b_scale.to_double() /
    y_scale.to_double()

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    for j = 0; j < n; j = j + 1 {
      let mut acc = 0
      for kk = 0; kk < k; kk = kk + 1 {
        let av = byte_to_i8(a[a_row + kk]) - az
        let bv = byte_to_i8(b[kk * n + j]) - bz
        acc = acc + av * bv
      }
      let q = round_half_to_even(acc.to_double() * mul_scale) + yz
      out[y_row + j] = i8_to_byte(q)
    }
  }
}

///|
/// 标量参考实现：无符号 `UInt8` 量化矩阵乘法 `Y[M, N] = QLinearMatMul(A[M, K], B[K, N])`。
pub fn qlinear_matmul_u8_scalar(
  a : FixedArray[Byte],
  a_scale : Float,
  a_zp : Byte,
  b : FixedArray[Byte],
  b_scale : Float,
  b_zp : Byte,
  y_scale : Float,
  y_zp : Byte,
  out : FixedArray[Byte],
  m : Int32,
  k : Int32,
  n : Int32,
  a_off~ : Int32 = 0,
  out_off~ : Int32 = 0,
) -> Unit {
  let az = a_zp.to_int()
  let bz = b_zp.to_int()
  let yz = y_zp.to_int()
  let mul_scale = a_scale.to_double() *
    b_scale.to_double() /
    y_scale.to_double()

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    for j = 0; j < n; j = j + 1 {
      let mut acc = 0
      for kk = 0; kk < k; kk = kk + 1 {
        let av = a[a_row + kk].to_int() - az
        let bv = b[kk * n + j].to_int() - bz
        acc = acc + av * bv
      }
      let q = round_half_to_even(acc.to_double() * mul_scale) + yz
      out[y_row + j] = u8_to_byte(q)
    }
  }
}

///|
/// V128 SIMD 硬件加速版有符号 `Int8` 量化矩阵乘法（预转置右矩阵 `BT[N, K]`，零运行期转置开销）。
pub fn qlinear_matmul_i8_pretransposed_v128(
  a : FixedArray[Byte],
  a_off : Int32,
  a_scale : Float,
  a_zp : Byte,
  bt : FixedArray[Byte],
  b_scale : Float,
  b_zp : Byte,
  y_scale : Float,
  y_zp : Byte,
  out : FixedArray[Byte],
  out_off : Int32,
  m : Int32,
  k : Int32,
  n : Int32,
) -> Unit {
  let az = byte_to_i8(a_zp)
  let bz = byte_to_i8(b_zp)
  let yz = byte_to_i8(y_zp)
  let v_azp = @simd.i16x8_splat((az & 0xFFFF).to_uint16())
  let v_bzp = @simd.i16x8_splat((bz & 0xFFFF).to_uint16())
  let v_zero = @simd.i32x4_splat(0U)
  let mul_scale = a_scale.to_double() *
    b_scale.to_double() /
    y_scale.to_double()
  let k_vec = k - k % 16

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    for j = 0; j < n; j = j + 1 {
      let bt_row = j * k
      let mut acc_v = v_zero
      let mut kk = 0
      while kk < k_vec {
        let va = @simd.v128_load(a, a_row + kk)
        let vb = @simd.v128_load(bt, bt_row + kk)
        let va_lo = @simd.i16x8_sub(@simd.i16x8_extend_low_i8x16_s(va), v_azp)
        let vb_lo = @simd.i16x8_sub(@simd.i16x8_extend_low_i8x16_s(vb), v_bzp)
        let va_hi = @simd.i16x8_sub(@simd.i16x8_extend_high_i8x16_s(va), v_azp)
        let vb_hi = @simd.i16x8_sub(@simd.i16x8_extend_high_i8x16_s(vb), v_bzp)
        acc_v = @simd.i32x4_add(acc_v, @simd.i32x4_dot_i16x8_s(va_lo, vb_lo))
        acc_v = @simd.i32x4_add(acc_v, @simd.i32x4_dot_i16x8_s(va_hi, vb_hi))
        kk = kk + 16
      }
      let mut acc = hsum_i32x4(acc_v)
      while kk < k {
        let av = byte_to_i8(a[a_row + kk]) - az
        let bv = byte_to_i8(bt[bt_row + kk]) - bz
        acc = acc + av * bv
        kk = kk + 1
      }
      let q = round_half_to_even(acc.to_double() * mul_scale) + yz
      out[y_row + j] = i8_to_byte(q)
    }
  }
}

///|
/// V128 SIMD 硬件加速版有符号 `Int8` 量化矩阵乘法：
/// 每轮迭代通过 `v128.load` + `i16x8.extend_*_i8x16_s` + `i16x8.sub` + 2× `i32x4.dot_i16x8_s`
/// 一次完成 16 路 INT8 乘加累加，与标量参考实现逐位完全一致。
pub fn qlinear_matmul_i8_v128(
  a : FixedArray[Byte],
  a_scale : Float,
  a_zp : Byte,
  b : FixedArray[Byte],
  b_scale : Float,
  b_zp : Byte,
  y_scale : Float,
  y_zp : Byte,
  out : FixedArray[Byte],
  m : Int32,
  k : Int32,
  n : Int32,
  a_off~ : Int32 = 0,
  out_off~ : Int32 = 0,
) -> Unit {
  let bt = transpose_bytes_kn_to_nk(b, k, n)
  qlinear_matmul_i8_pretransposed_v128(
    a, a_off, a_scale, a_zp, bt, b_scale, b_zp, y_scale, y_zp, out, out_off, m, k,
    n,
  )
}

///|
/// V128 SIMD 硬件加速版无符号 `UInt8` 量化矩阵乘法（预转置右矩阵 `BT[N, K]`，零运行期转置开销）。
pub fn qlinear_matmul_u8_pretransposed_v128(
  a : FixedArray[Byte],
  a_off : Int32,
  a_scale : Float,
  a_zp : Byte,
  bt : FixedArray[Byte],
  b_scale : Float,
  b_zp : Byte,
  y_scale : Float,
  y_zp : Byte,
  out : FixedArray[Byte],
  out_off : Int32,
  m : Int32,
  k : Int32,
  n : Int32,
) -> Unit {
  let az = a_zp.to_int()
  let bz = b_zp.to_int()
  let yz = y_zp.to_int()
  let v_azp = @simd.i16x8_splat((az & 0xFFFF).to_uint16())
  let v_bzp = @simd.i16x8_splat((bz & 0xFFFF).to_uint16())
  let v_zero = @simd.i32x4_splat(0U)
  let mul_scale = a_scale.to_double() *
    b_scale.to_double() /
    y_scale.to_double()
  let k_vec = k - k % 16

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    for j = 0; j < n; j = j + 1 {
      let bt_row = j * k
      let mut acc_v = v_zero
      let mut kk = 0
      while kk < k_vec {
        let va = @simd.v128_load(a, a_row + kk)
        let vb = @simd.v128_load(bt, bt_row + kk)
        let va_lo = @simd.i16x8_sub(@simd.i16x8_extend_low_i8x16_u(va), v_azp)
        let vb_lo = @simd.i16x8_sub(@simd.i16x8_extend_low_i8x16_u(vb), v_bzp)
        let va_hi = @simd.i16x8_sub(@simd.i16x8_extend_high_i8x16_u(va), v_azp)
        let vb_hi = @simd.i16x8_sub(@simd.i16x8_extend_high_i8x16_u(vb), v_bzp)
        acc_v = @simd.i32x4_add(acc_v, @simd.i32x4_dot_i16x8_s(va_lo, vb_lo))
        acc_v = @simd.i32x4_add(acc_v, @simd.i32x4_dot_i16x8_s(va_hi, vb_hi))
        kk = kk + 16
      }
      let mut acc = hsum_i32x4(acc_v)
      while kk < k {
        let av = a[a_row + kk].to_int() - az
        let bv = bt[bt_row + kk].to_int() - bz
        acc = acc + av * bv
        kk = kk + 1
      }
      let q = round_half_to_even(acc.to_double() * mul_scale) + yz
      out[y_row + j] = u8_to_byte(q)
    }
  }
}

///|
/// V128 SIMD 硬件加速版无符号 `UInt8` 量化矩阵乘法：
/// 每轮迭代通过 `v128.load` + `i16x8.extend_*_i8x16_u` + `i16x8.sub` + 2× `i32x4.dot_i16x8_s`
/// 一次完成 16 路 UINT8 乘加累加，与标量参考实现逐位完全一致。
pub fn qlinear_matmul_u8_v128(
  a : FixedArray[Byte],
  a_scale : Float,
  a_zp : Byte,
  b : FixedArray[Byte],
  b_scale : Float,
  b_zp : Byte,
  y_scale : Float,
  y_zp : Byte,
  out : FixedArray[Byte],
  m : Int32,
  k : Int32,
  n : Int32,
  a_off~ : Int32 = 0,
  out_off~ : Int32 = 0,
) -> Unit {
  let bt = transpose_bytes_kn_to_nk(b, k, n)
  qlinear_matmul_u8_pretransposed_v128(
    a, a_off, a_scale, a_zp, bt, b_scale, b_zp, y_scale, y_zp, out, out_off, m, k,
    n,
  )
}

///|
/// 将 2D Float32 权重矩阵 `W[K, N]` 按列对称量化为紧凑 `Int8`（零点固定为 0，实现 4× 权重内存压缩）。
/// 返回 `(w_i8_col_major[N, K], col_scales[N])`，直接供 `matmul_f32_w8_i8` 零转置消费。
pub fn quantize_weights_symmetric_i8(
  w_f32 : FixedArray[Float],
  k : Int32,
  n : Int32,
) -> (FixedArray[Byte], FixedArray[Float]) {
  let w_q = FixedArray::make(n * k, b'\x00')
  let scales = FixedArray::make(n, (1.0 : Float))
  for j = 0; j < n; j = j + 1 {
    let mut max_abs = 0.0
    for kk = 0; kk < k; kk = kk + 1 {
      let a = w_f32[kk * n + j].abs().to_double()
      if a > max_abs {
        max_abs = a
      }
    }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 }
    let inv_scale = 1.0 / scale
    scales[j] = Float::from_double(scale)
    let col_off = j * k
    for kk = 0; kk < k; kk = kk + 1 {
      let q = round_half_to_even(w_f32[kk * n + j].to_double() * inv_scale)
      w_q[col_off + kk] = i8_to_byte(q)
    }
  }
  (w_q, scales)
}

///|
/// 仅权重量化（Weight-Only INT8）矩阵乘法：`Y[M, N] = A_f32[M, K] × dequant(W_i8_col_major[N, K], col_scales[N])`。
/// 适合边缘 Wasm 场景下以 4× 更小模型体积直接运行 Float32 激活推理。
pub fn matmul_f32_w8_i8(
  a_f32 : FixedArray[Float],
  w_i8_nk : FixedArray[Byte],
  col_scales : FixedArray[Float],
  out_f32 : FixedArray[Float],
  m : Int32,
  k : Int32,
  n : Int32,
) -> Unit {
  for i = 0; i < m; i = i + 1 {
    let a_row = i * k
    let y_row = i * n
    for j = 0; j < n; j = j + 1 {
      let w_col = j * k
      let mut acc : Float = 0.0
      for kk = 0; kk < k; kk = kk + 1 {
        let wv = Float::from_int(byte_to_i8(w_i8_nk[w_col + kk]))
        acc = acc + a_f32[a_row + kk] * wv
      }
      out_f32[y_row + j] = acc * col_scales[j]
    }
  }
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

///|
/// 预打包的有符号 `Int8` 权重矩阵 `B[K, N]`：
/// - `bt` 以列优先连续布局 `[N, K]` 存储，保证 `V128` 沿 `K` 维 16 字节全速顺序读取；
/// - `col_sums[N]` 离线预计算每列有符号总和 `sum_k B[k, j]`，将 `a_zp` 零点补偿移出内层 K 循环。
pub struct PackedInt8MatrixB {
  bt : FixedArray[Byte]
  col_sums : FixedArray[Int32]
  k : Int32
  n : Int32
}

///|
/// 返回预打包 INT8 权重的逻辑维度 `(K, N)`。
pub fn PackedInt8MatrixB::dims(self : PackedInt8MatrixB) -> (Int32, Int32) {
  (self.k, self.n)
}

///|
/// 离线打包行主序有符号 `Int8` 权重矩阵 `B[K, N]` 为 `PackedInt8MatrixB`。
pub fn pack_int8_matrix_b(
  b : FixedArray[Byte],
  k : Int32,
  n : Int32,
) -> PackedInt8MatrixB {
  let bt = FixedArray::make(n * k, b'\x00')
  let col_sums = FixedArray::make(n, 0)
  for kk = 0; kk < k; kk = kk + 1 {
    let row = kk * n
    for j = 0; j < n; j = j + 1 {
      let bv = b[row + j]
      bt[j * k + kk] = bv
      col_sums[j] = col_sums[j] + byte_to_i8(bv)
    }
  }
  { bt, col_sums, k, n, }
}

///|
/// 检查可选的 `b_zp` 数组是否包含非零元素（对称每通道量化权重通常全 0，可跳过 `row_sum` 计算）。
fn has_nonzero_byte(arr : FixedArray[Byte]?) -> Bool {
  match arr {
    None => false
    Some(b) => {
      for i = 0; i < b.length(); i = i + 1 {
        if b[i] != b'\x00' {
          return true
        }
      }
      false
    }
  }
}

///|
/// 提取第 `j` 列的有符号 `Int8` 零点值（支持标量长度 1 与每列长度 `N` 两种布局）。
fn get_b_zp_i8(b_zp : FixedArray[Byte]?, j : Int32) -> Int32 {
  match b_zp {
    None => 0
    Some(arr) =>
      if arr.length() == 0 {
        0
      } else if arr.length() == 1 {
        byte_to_i8(arr[0])
      } else {
        byte_to_i8(arr[j])
      }
  }
}

///|
/// 共享 4 列寄存器展开 `V128` `UInt8 × Int8` 点积微内核：
/// 1 次 `A` 向量加载 + 扩展并发服务 4 列预打包 `BT`，并叠加 `col_sums` / `b_zp` 零点补偿。
fn dot4_u8_i8_v128(
  a : FixedArray[Byte],
  a_row : Int32,
  bt : FixedArray[Byte],
  col_sums : FixedArray[Int32],
  j : Int32,
  k : Int32,
  k_vec : Int32,
  az : Int32,
  need_bzp : Bool,
  b_zp : FixedArray[Byte]?,
  row_sum : Int32,
) -> (Int32, Int32, Int32, Int32) {
  let v_zero = @simd.i32x4_splat(0U)
  let b0 = j * k
  let b1 = b0 + k
  let b2 = b1 + k
  let b3 = b2 + k
  let mut acc0 = v_zero
  let mut acc1 = v_zero
  let mut acc2 = v_zero
  let mut acc3 = v_zero
  let mut kk = 0
  while kk < k_vec {
    let va = @simd.v128_load(a, a_row + kk)
    let va_lo = @simd.i16x8_extend_low_i8x16_u(va)
    let va_hi = @simd.i16x8_extend_high_i8x16_u(va)

    let vb0 = @simd.v128_load(bt, b0 + kk)
    acc0 = @simd.i32x4_add(
      acc0,
      @simd.i32x4_dot_i16x8_s(va_lo, @simd.i16x8_extend_low_i8x16_s(vb0)),
    )
    acc0 = @simd.i32x4_add(
      acc0,
      @simd.i32x4_dot_i16x8_s(va_hi, @simd.i16x8_extend_high_i8x16_s(vb0)),
    )

    let vb1 = @simd.v128_load(bt, b1 + kk)
    acc1 = @simd.i32x4_add(
      acc1,
      @simd.i32x4_dot_i16x8_s(va_lo, @simd.i16x8_extend_low_i8x16_s(vb1)),
    )
    acc1 = @simd.i32x4_add(
      acc1,
      @simd.i32x4_dot_i16x8_s(va_hi, @simd.i16x8_extend_high_i8x16_s(vb1)),
    )

    let vb2 = @simd.v128_load(bt, b2 + kk)
    acc2 = @simd.i32x4_add(
      acc2,
      @simd.i32x4_dot_i16x8_s(va_lo, @simd.i16x8_extend_low_i8x16_s(vb2)),
    )
    acc2 = @simd.i32x4_add(
      acc2,
      @simd.i32x4_dot_i16x8_s(va_hi, @simd.i16x8_extend_high_i8x16_s(vb2)),
    )

    let vb3 = @simd.v128_load(bt, b3 + kk)
    acc3 = @simd.i32x4_add(
      acc3,
      @simd.i32x4_dot_i16x8_s(va_lo, @simd.i16x8_extend_low_i8x16_s(vb3)),
    )
    acc3 = @simd.i32x4_add(
      acc3,
      @simd.i32x4_dot_i16x8_s(va_hi, @simd.i16x8_extend_high_i8x16_s(vb3)),
    )
    kk = kk + 16
  }
  let mut s0 = hsum_i32x4(acc0)
  let mut s1 = hsum_i32x4(acc1)
  let mut s2 = hsum_i32x4(acc2)
  let mut s3 = hsum_i32x4(acc3)
  while kk < k {
    let av = a.unsafe_get(a_row + kk).to_int()
    s0 = s0 + av * byte_to_i8(bt.unsafe_get(b0 + kk))
    s1 = s1 + av * byte_to_i8(bt.unsafe_get(b1 + kk))
    s2 = s2 + av * byte_to_i8(bt.unsafe_get(b2 + kk))
    s3 = s3 + av * byte_to_i8(bt.unsafe_get(b3 + kk))
    kk = kk + 1
  }
  s0 = s0 - az * col_sums.unsafe_get(j)
  s1 = s1 - az * col_sums.unsafe_get(j + 1)
  s2 = s2 - az * col_sums.unsafe_get(j + 2)
  s3 = s3 - az * col_sums.unsafe_get(j + 3)
  if need_bzp {
    let delta = row_sum - k * az
    s0 = s0 - get_b_zp_i8(b_zp, j) * delta
    s1 = s1 - get_b_zp_i8(b_zp, j + 1) * delta
    s2 = s2 - get_b_zp_i8(b_zp, j + 2) * delta
    s3 = s3 - get_b_zp_i8(b_zp, j + 3) * delta
  }
  (s0, s1, s2, s3)
}

///|
/// 共享单列尾部 `V128` `UInt8 × Int8` 点积微内核。
fn dot1_u8_i8_v128(
  a : FixedArray[Byte],
  a_row : Int32,
  bt : FixedArray[Byte],
  col_sums : FixedArray[Int32],
  j : Int32,
  k : Int32,
  k_vec : Int32,
  az : Int32,
  need_bzp : Bool,
  b_zp : FixedArray[Byte]?,
  row_sum : Int32,
) -> Int32 {
  let b0 = j * k
  let mut acc0 = @simd.i32x4_splat(0U)
  let mut kk = 0
  while kk < k_vec {
    let va = @simd.v128_load(a, a_row + kk)
    let vb = @simd.v128_load(bt, b0 + kk)
    acc0 = @simd.i32x4_add(
      acc0,
      @simd.i32x4_dot_i16x8_s(
        @simd.i16x8_extend_low_i8x16_u(va),
        @simd.i16x8_extend_low_i8x16_s(vb),
      ),
    )
    acc0 = @simd.i32x4_add(
      acc0,
      @simd.i32x4_dot_i16x8_s(
        @simd.i16x8_extend_high_i8x16_u(va),
        @simd.i16x8_extend_high_i8x16_s(vb),
      ),
    )
    kk = kk + 16
  }
  let mut s0 = hsum_i32x4(acc0)
  while kk < k {
    s0 = s0 +
      a.unsafe_get(a_row + kk).to_int() * byte_to_i8(bt.unsafe_get(b0 + kk))
    kk = kk + 1
  }
  s0 = s0 - az * col_sums.unsafe_get(j)
  if need_bzp {
    s0 = s0 - get_b_zp_i8(b_zp, j) * (row_sum - k * az)
  }
  s0
}

///|
/// V128 SIMD 4 列展开 `MatMulInteger(A_u8[M, K], B_i8[K, N]) -> Int32[M, N]`：
/// - 复用同一条 `A` 向量并发点积 4 列预打包 `BT`，减少 4× 的 `A` 加载与扩展开销；
/// - 借助离线 `col_sums` 将零点减法完全移出热循环，内循环仅执行纯 `i32x4_dot_i16x8_s` 硬件点积。
pub fn matmul_integer_u8_i8_prepacked_v128(
  a : FixedArray[Byte],
  a_off : Int32,
  a_zp : Byte,
  packed_b : PackedInt8MatrixB,
  b_zp : FixedArray[Byte]?,
  out : FixedArray[Int32],
  out_off : Int32,
  m : Int32,
) -> Unit {
  let k = packed_b.k
  let n = packed_b.n
  if m <= 0 || n <= 0 {
    return
  }
  let bt = packed_b.bt
  let col_sums = packed_b.col_sums
  check_quant_bounds(a.length(), a_off, m * k, "matmul_integer_v128(a)")
  check_quant_bounds(bt.length(), 0, n * k, "matmul_integer_v128(bt)")
  check_quant_bounds(col_sums.length(), 0, n, "matmul_integer_v128(col_sums)")
  check_quant_bounds(out.length(), out_off, m * n, "matmul_integer_v128(out)")
  let az = a_zp.to_int()
  let need_bzp = has_nonzero_byte(b_zp)
  let k_vec = k - k % 16

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    let mut row_sum = 0
    if need_bzp {
      for kk = 0; kk < k; kk = kk + 1 {
        row_sum = row_sum + a.unsafe_get(a_row + kk).to_int()
      }
    }
    let mut j = 0
    while j + 4 <= n {
      let (s0, s1, s2, s3) = dot4_u8_i8_v128(
        a, a_row, bt, col_sums, j, k, k_vec, az, need_bzp, b_zp, row_sum,
      )
      out.unsafe_set(y_row + j, s0)
      out.unsafe_set(y_row + j + 1, s1)
      out.unsafe_set(y_row + j + 2, s2)
      out.unsafe_set(y_row + j + 3, s3)
      j = j + 4
    }
    while j < n {
      out.unsafe_set(
        y_row + j,
        dot1_u8_i8_v128(
          a, a_row, bt, col_sums, j, k, k_vec, az, need_bzp, b_zp, row_sum,
        ),
      )
      j = j + 1
    }
  }
}

///|
/// V128 SIMD 融合输出版 `MatMulInteger(A_u8, B_i8) -> Float32`（直接将 `Int32` 累加器转为 `Float32` 写回 Arena，消除中间 `Int32` 缓冲）。
pub fn matmul_integer_u8_i8_prepacked_to_f32_v128(
  a : FixedArray[Byte],
  a_off : Int32,
  a_zp : Byte,
  packed_b : PackedInt8MatrixB,
  b_zp : FixedArray[Byte]?,
  out : FixedArray[Float],
  out_off : Int32,
  m : Int32,
) -> Unit {
  let k = packed_b.k
  let n = packed_b.n
  if m <= 0 || n <= 0 {
    return
  }
  let bt = packed_b.bt
  let col_sums = packed_b.col_sums
  check_quant_bounds(a.length(), a_off, m * k, "matmul_integer_to_f32_v128(a)")
  check_quant_bounds(bt.length(), 0, n * k, "matmul_integer_to_f32_v128(bt)")
  check_quant_bounds(
    col_sums.length(),
    0,
    n,
    "matmul_integer_to_f32_v128(col_sums)",
  )
  check_quant_bounds(
    out.length(),
    out_off,
    m * n,
    "matmul_integer_to_f32_v128(out)",
  )
  let az = a_zp.to_int()
  let need_bzp = has_nonzero_byte(b_zp)
  let k_vec = k - k % 16

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    let mut row_sum = 0
    if need_bzp {
      for kk = 0; kk < k; kk = kk + 1 {
        row_sum = row_sum + a.unsafe_get(a_row + kk).to_int()
      }
    }
    let mut j = 0
    while j + 4 <= n {
      let (s0, s1, s2, s3) = dot4_u8_i8_v128(
        a, a_row, bt, col_sums, j, k, k_vec, az, need_bzp, b_zp, row_sum,
      )
      out.unsafe_set(y_row + j, Float::from_int(s0))
      out.unsafe_set(y_row + j + 1, Float::from_int(s1))
      out.unsafe_set(y_row + j + 2, Float::from_int(s2))
      out.unsafe_set(y_row + j + 3, Float::from_int(s3))
      j = j + 4
    }
    while j < n {
      out.unsafe_set(
        y_row + j,
        Float::from_int(
          dot1_u8_i8_v128(
            a, a_row, bt, col_sums, j, k, k_vec, az, need_bzp, b_zp, row_sum,
          ),
        ),
      )
      j = j + 1
    }
  }
}

///|
/// 4 列展开标量快速路径 `MatMulInteger(A_u8, B_i8) -> Float32`（专为 Wasm-GC 优化：
/// 避免 GC 字节数组 `v128_load` 退化，单次加载 `a` 并发点积 4 列预打包 `BT` 并配合 `unsafe_get` / `unsafe_set`）。
pub fn matmul_integer_u8_i8_prepacked_to_f32_scalar(
  a : FixedArray[Byte],
  a_off : Int32,
  a_zp : Byte,
  packed_b : PackedInt8MatrixB,
  b_zp : FixedArray[Byte]?,
  out : FixedArray[Float],
  out_off : Int32,
  m : Int32,
) -> Unit {
  let k = packed_b.k
  let n = packed_b.n
  if m <= 0 || n <= 0 {
    return
  }
  let bt = packed_b.bt
  let col_sums = packed_b.col_sums
  check_quant_bounds(
    a.length(),
    a_off,
    m * k,
    "matmul_integer_to_f32_scalar(a)",
  )
  check_quant_bounds(bt.length(), 0, n * k, "matmul_integer_to_f32_scalar(bt)")
  check_quant_bounds(
    col_sums.length(),
    0,
    n,
    "matmul_integer_to_f32_scalar(col_sums)",
  )
  check_quant_bounds(
    out.length(),
    out_off,
    m * n,
    "matmul_integer_to_f32_scalar(out)",
  )
  let az = a_zp.to_int()
  let need_bzp = has_nonzero_byte(b_zp)

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let y_row = out_off + i * n
    let mut row_sum = 0
    if need_bzp {
      for kk = 0; kk < k; kk = kk + 1 {
        row_sum = row_sum + a.unsafe_get(a_row + kk).to_int()
      }
    }
    let delta = row_sum - k * az
    let mut j = 0
    while j + 4 <= n {
      let b0 = j * k
      let b1 = b0 + k
      let b2 = b1 + k
      let b3 = b2 + k
      let mut s0 = 0
      let mut s1 = 0
      let mut s2 = 0
      let mut s3 = 0
      for kk = 0; kk < k; kk = kk + 1 {
        let av = a.unsafe_get(a_row + kk).to_int()
        s0 = s0 + av * byte_to_i8(bt.unsafe_get(b0 + kk))
        s1 = s1 + av * byte_to_i8(bt.unsafe_get(b1 + kk))
        s2 = s2 + av * byte_to_i8(bt.unsafe_get(b2 + kk))
        s3 = s3 + av * byte_to_i8(bt.unsafe_get(b3 + kk))
      }
      s0 = s0 - az * col_sums.unsafe_get(j)
      s1 = s1 - az * col_sums.unsafe_get(j + 1)
      s2 = s2 - az * col_sums.unsafe_get(j + 2)
      s3 = s3 - az * col_sums.unsafe_get(j + 3)
      if need_bzp {
        s0 = s0 - get_b_zp_i8(b_zp, j) * delta
        s1 = s1 - get_b_zp_i8(b_zp, j + 1) * delta
        s2 = s2 - get_b_zp_i8(b_zp, j + 2) * delta
        s3 = s3 - get_b_zp_i8(b_zp, j + 3) * delta
      }
      out.unsafe_set(y_row + j, Float::from_int(s0))
      out.unsafe_set(y_row + j + 1, Float::from_int(s1))
      out.unsafe_set(y_row + j + 2, Float::from_int(s2))
      out.unsafe_set(y_row + j + 3, Float::from_int(s3))
      j = j + 4
    }
    while j < n {
      let b0 = j * k
      let mut acc = 0
      for kk = 0; kk < k; kk = kk + 1 {
        acc = acc +
          a.unsafe_get(a_row + kk).to_int() * byte_to_i8(bt.unsafe_get(b0 + kk))
      }
      acc = acc - az * col_sums.unsafe_get(j)
      if need_bzp {
        acc = acc - get_b_zp_i8(b_zp, j) * delta
      }
      out.unsafe_set(y_row + j, Float::from_int(acc))
      j = j + 1
    }
  }
}
