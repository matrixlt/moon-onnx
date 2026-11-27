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
