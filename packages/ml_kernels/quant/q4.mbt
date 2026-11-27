///|
/// 将 `Float32` 写入小端字节缓冲指定偏移（供 `MatMulNBits` Q4 奇偶拆分 V128 加载使用）。
fn write_f32_le(dst : FixedArray[Byte], byte_off : Int32, v : Float) -> Unit {
  let bits = v.reinterpret_as_uint()
  dst[byte_off] = (bits & 0xFFU).to_byte()
  dst[byte_off + 1] = ((bits >> 8) & 0xFFU).to_byte()
  dst[byte_off + 2] = ((bits >> 16) & 0xFFU).to_byte()
  dst[byte_off + 3] = ((bits >> 24) & 0xFFU).to_byte()
}

///|
/// 读取 `MatMulNBits` 4-bit 权重的指定块零点值（默认对称零点为 `8.0`）。
/// - 当 `zp_bytes` 存在时按 ONNX 规范每字节打包 2 个 4-bit 零点（`n_blocks_zp = (n_blocks + 1) / 2`）；
/// - 当 `zp_f32` 存在时直接读取 `Float32` 零点；
/// - 均为空时使用默认对称零点 `8`。
fn read_q4_zero_point(
  zp_bytes : FixedArray[Byte]?,
  zp_f32 : FixedArray[Float]?,
  j : Int32,
  blk : Int32,
  n_blocks : Int32,
) -> Float {
  match zp_bytes {
    Some(zb) => {
      let row_bytes = (n_blocks + 1) / 2
      let b = zb[j * row_bytes + blk / 2].to_int()
      let nib = if (blk & 1) == 0 { b & 0x0F } else { (b >> 4) & 0x0F }
      Float::from_int(nib)
    }
    None =>
      match zp_f32 {
        Some(zf) => zf[j * n_blocks + blk]
        None => (8.0 : Float)
      }
  }
}

///|
/// 读取 4-bit 整数零点（用于 V128 整型路径 `i8x16_sub`；当提供 `zp_f32` 时返回 `None` 走浮点路径）。
fn read_q4_zero_point_i8(
  zp_bytes : FixedArray[Byte]?,
  j : Int32,
  blk : Int32,
  n_blocks : Int32,
) -> Byte {
  match zp_bytes {
    Some(zb) => {
      let row_bytes = (n_blocks + 1) / 2
      let b = zb[j * row_bytes + blk / 2].to_int()
      let nib = if (blk & 1) == 0 { b & 0x0F } else { (b >> 4) & 0x0F }
      nib.to_byte()
    }
    None => b'\x08'
  }
}

///|
/// 校验 Q4 量化矩阵乘缓冲切片区间不越界。
fn check_q4_bounds(
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
/// 标量参考实现：ONNX `com.microsoft::MatMulNBits`（4-bit 分块量化矩阵乘法）。
/// - `a`: `Float32[M, K]`（起始偏移 `a_off`）
/// - `b_q4`: `UInt8[N, n_blocks, block_size / 2]`（每字节低 4 位为偶数位，高 4 位为奇数位）
/// - `scales`: `Float32[N * n_blocks]`
/// - `zp_bytes` / `zp_f32`: 可选 4-bit 打包字节或 Float32 零点（缺省对称零点 `8`）
/// - `bias`: 可选 `Float32[N]`
/// - `out`: `Float32[M, N]`（起始偏移 `out_off`）
pub fn matmul_nbits_q4_scalar_f32(
  a : FixedArray[Float],
  a_off : Int32,
  b_q4 : FixedArray[Byte],
  scales : FixedArray[Float],
  zp_bytes : FixedArray[Byte]?,
  zp_f32 : FixedArray[Float]?,
  bias : FixedArray[Float]?,
  out : FixedArray[Float],
  out_off : Int32,
  m : Int32,
  k : Int32,
  n : Int32,
  block_size : Int32,
) -> Unit {
  if m <= 0 || n <= 0 {
    return
  }
  let n_blocks = (k + block_size - 1) / block_size
  let blob_bytes = block_size / 2
  let col_stride = n_blocks * blob_bytes
  check_q4_bounds(a.length(), a_off, m * k, "matmul_nbits_q4_scalar(a)")
  check_q4_bounds(
    b_q4.length(),
    0,
    n * col_stride,
    "matmul_nbits_q4_scalar(b_q4)",
  )
  check_q4_bounds(
    scales.length(),
    0,
    n * n_blocks,
    "matmul_nbits_q4_scalar(scales)",
  )
  check_q4_bounds(out.length(), out_off, m * n, "matmul_nbits_q4_scalar(out)")
  match bias {
    Some(bv) =>
      check_q4_bounds(bv.length(), 0, n, "matmul_nbits_q4_scalar(bias)")
    None => ()
  }

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let o_row = out_off + i * n
    for j = 0; j < n; j = j + 1 {
      let b_col = j * col_stride
      let s_col = j * n_blocks
      let mut acc : Float = match bias {
        Some(bv) => bv.unsafe_get(j)
        None => (0.0 : Float)
      }
      for blk = 0; blk < n_blocks; blk = blk + 1 {
        let sc = scales.unsafe_get(s_col + blk)
        let zp = read_q4_zero_point(zp_bytes, zp_f32, j, blk, n_blocks)
        let k_base = blk * block_size
        let b_blk = b_col + blk * blob_bytes
        let mut blk_sum : Float = (0.0 : Float)
        let mut pair = 0
        let mut kk = k_base
        while kk < k && pair < blob_bytes {
          let byte_val = b_q4.unsafe_get(b_blk + pair).to_int()
          let lo = Float::from_int(byte_val & 0x0F) - zp
          blk_sum = blk_sum + a.unsafe_get(a_row + kk) * lo
          if kk + 1 < k {
            let hi = Float::from_int((byte_val >> 4) & 0x0F) - zp
            blk_sum = blk_sum + a.unsafe_get(a_row + kk + 1) * hi
          }
          kk = kk + 2
          pair = pair + 1
        }
        acc = acc + blk_sum * sc
      }
      out.unsafe_set(o_row + j, acc)
    }
  }
}

///|
/// 将 `f32x4` 向量中 4 个浮点车道求水平和。
fn hsum_f32x4(v : V128) -> Float {
  @simd.f32x4_extract_lane(v, 0) +
  @simd.f32x4_extract_lane(v, 1) +
  @simd.f32x4_extract_lane(v, 2) +
  @simd.f32x4_extract_lane(v, 3)
}

///|
/// 对单块 16 字节（32 个 4-bit 权重）执行零洗牌 `V128` 奇偶解包并与已加载的 8 条 `A` 向量做 `f32x4_relaxed_madd` 累加。
fn q4_dot32_v128(
  vb : V128,
  mask0f : V128,
  zp_v : V128,
  ae0 : V128,
  ae1 : V128,
  ae2 : V128,
  ae3 : V128,
  ao0 : V128,
  ao1 : V128,
  ao2 : V128,
  ao3 : V128,
) -> Float {
  // 低 4 位对应偶数位置 k=0,2,...,30（与 ae0..ae3 严格按车道对齐，0 次 shuffle！）
  let lo_i8 = @simd.i8x16_sub(@simd.v128_and_(vb, mask0f), zp_v)
  let lo_i16_0 = @simd.i16x8_extend_low_i8x16_s(lo_i8)
  let lo_i16_1 = @simd.i16x8_extend_high_i8x16_s(lo_i8)
  let w0 = @simd.f32x4_convert_i32x4_s(@simd.i32x4_extend_low_i16x8_s(lo_i16_0))
  let w1 = @simd.f32x4_convert_i32x4_s(
    @simd.i32x4_extend_high_i16x8_s(lo_i16_0),
  )
  let w2 = @simd.f32x4_convert_i32x4_s(@simd.i32x4_extend_low_i16x8_s(lo_i16_1))
  let w3 = @simd.f32x4_convert_i32x4_s(
    @simd.i32x4_extend_high_i16x8_s(lo_i16_1),
  )

  let mut acc = @simd.f32x4_mul(ae0, w0)
  acc = @simd.f32x4_relaxed_madd(ae1, w1, acc)
  acc = @simd.f32x4_relaxed_madd(ae2, w2, acc)
  acc = @simd.f32x4_relaxed_madd(ae3, w3, acc)

  // 高 4 位对应奇数位置 k=1,3,...,31（与 ao0..ao3 严格按车道对齐，0 次 shuffle！）
  let hi_i8 = @simd.i8x16_sub(@simd.i8x16_shr_u(vb, 4), zp_v)
  let hi_i16_0 = @simd.i16x8_extend_low_i8x16_s(hi_i8)
  let hi_i16_1 = @simd.i16x8_extend_high_i8x16_s(hi_i8)
  let u0 = @simd.f32x4_convert_i32x4_s(@simd.i32x4_extend_low_i16x8_s(hi_i16_0))
  let u1 = @simd.f32x4_convert_i32x4_s(
    @simd.i32x4_extend_high_i16x8_s(hi_i16_0),
  )
  let u2 = @simd.f32x4_convert_i32x4_s(@simd.i32x4_extend_low_i16x8_s(hi_i16_1))
  let u3 = @simd.f32x4_convert_i32x4_s(
    @simd.i32x4_extend_high_i16x8_s(hi_i16_1),
  )

  acc = @simd.f32x4_relaxed_madd(ao0, u0, acc)
  acc = @simd.f32x4_relaxed_madd(ao1, u1, acc)
  acc = @simd.f32x4_relaxed_madd(ao2, u2, acc)
  acc = @simd.f32x4_relaxed_madd(ao3, u3, acc)
  hsum_f32x4(acc)
}

///|
/// V128 SIMD 高性能 4-bit 分块量化矩阵乘法 `MatMulNBits`（支持 `block_size` 为 32 的倍数，如 32/64/128）：
/// - 对输入行 `A` 一次性执行奇偶拆分打包为字节视图 `a_even` / `a_odd`，彻底消除内循环 nibble 交错洗牌开销；
/// - 4 列寄存器展开复用 `A` 的 8 条 `f32x4` 向量，直接以硬件 `v128_load` + `f32x4_convert_i32x4_s` + `f32x4_relaxed_madd` 累加，
///   在获得接近理论带宽上限速度的同时保持与 ONNX Runtime `~1e-6` 级 IEEE-754 数值对齐。
pub fn matmul_nbits_q4_v128_f32(
  a : FixedArray[Float],
  a_off : Int32,
  b_q4 : FixedArray[Byte],
  scales : FixedArray[Float],
  zp_bytes : FixedArray[Byte]?,
  zp_f32 : FixedArray[Float]?,
  bias : FixedArray[Float]?,
  out : FixedArray[Float],
  out_off : Int32,
  m : Int32,
  k : Int32,
  n : Int32,
  block_size : Int32,
) -> Unit {
  if m <= 0 || n <= 0 {
    return
  }
  // 非 32 倍数或浮点零点回退至精确标量实现
  if block_size <= 0 || block_size % 32 != 0 || k % 32 != 0 || !(zp_f32 is None) {
    matmul_nbits_q4_scalar_f32(
      a, a_off, b_q4, scales, zp_bytes, zp_f32, bias, out, out_off, m, k, n, block_size,
    )
    return
  }

  let n_blocks = k / block_size
  let sub_per_blk = block_size / 32
  let blob_bytes = block_size / 2
  let col_stride = n_blocks * blob_bytes
  check_q4_bounds(a.length(), a_off, m * k, "matmul_nbits_q4_v128(a)")
  check_q4_bounds(
    b_q4.length(),
    0,
    n * col_stride,
    "matmul_nbits_q4_v128(b_q4)",
  )
  check_q4_bounds(
    scales.length(),
    0,
    n * n_blocks,
    "matmul_nbits_q4_v128(scales)",
  )
  check_q4_bounds(out.length(), out_off, m * n, "matmul_nbits_q4_v128(out)")
  match bias {
    Some(bv) => check_q4_bounds(bv.length(), 0, n, "matmul_nbits_q4_v128(bias)")
    None => ()
  }

  let half_k = k / 2
  let a_even = FixedArray::make(half_k * 4, b'\x00')
  let a_odd = FixedArray::make(half_k * 4, b'\x00')
  let mask0f = @simd.i8x16_splat(b'\x0F')
  let default_zp_v = @simd.i8x16_splat(b'\x08')
  let has_zp = !(zp_bytes is None)

  for i = 0; i < m; i = i + 1 {
    let a_row = a_off + i * k
    let o_row = out_off + i * n
    // 一次性奇偶拆分打包当前输入行 A[i, 0..K)
    for p = 0; p < half_k; p = p + 1 {
      let bo = p * 4
      write_f32_le(a_even, bo, a.unsafe_get(a_row + p * 2))
      write_f32_le(a_odd, bo, a.unsafe_get(a_row + p * 2 + 1))
    }

    let mut j = 0
    while j + 4 <= n {
      let bc0 = j * col_stride
      let bc1 = bc0 + col_stride
      let bc2 = bc1 + col_stride
      let bc3 = bc2 + col_stride
      let sc0 = j * n_blocks
      let sc1 = sc0 + n_blocks
      let sc2 = sc1 + n_blocks
      let sc3 = sc2 + n_blocks
      let (b0, b1, b2, b3) = match bias {
        Some(bv) =>
          (
            bv.unsafe_get(j),
            bv.unsafe_get(j + 1),
            bv.unsafe_get(j + 2),
            bv.unsafe_get(j + 3),
          )
        None => ((0.0 : Float), (0.0 : Float), (0.0 : Float), (0.0 : Float))
      }
      let mut acc0 : Float = b0
      let mut acc1 : Float = b1
      let mut acc2 : Float = b2
      let mut acc3 : Float = b3

      for blk = 0; blk < n_blocks; blk = blk + 1 {
        let zp0 = if has_zp {
          @simd.i8x16_splat(read_q4_zero_point_i8(zp_bytes, j, blk, n_blocks))
        } else {
          default_zp_v
        }
        let zp1 = if has_zp {
          @simd.i8x16_splat(
            read_q4_zero_point_i8(zp_bytes, j + 1, blk, n_blocks),
          )
        } else {
          default_zp_v
        }
        let zp2 = if has_zp {
          @simd.i8x16_splat(
            read_q4_zero_point_i8(zp_bytes, j + 2, blk, n_blocks),
          )
        } else {
          default_zp_v
        }
        let zp3 = if has_zp {
          @simd.i8x16_splat(
            read_q4_zero_point_i8(zp_bytes, j + 3, blk, n_blocks),
          )
        } else {
          default_zp_v
        }

        let mut bsum0 : Float = (0.0 : Float)
        let mut bsum1 : Float = (0.0 : Float)
        let mut bsum2 : Float = (0.0 : Float)
        let mut bsum3 : Float = (0.0 : Float)
        let blk_byte_off = blk * blob_bytes

        for sub = 0; sub < sub_per_blk; sub = sub + 1 {
          let a_sub_off = (blk * sub_per_blk + sub) * 64
          let ae0 = @simd.v128_load(a_even, a_sub_off)
          let ae1 = @simd.v128_load(a_even, a_sub_off + 16)
          let ae2 = @simd.v128_load(a_even, a_sub_off + 32)
          let ae3 = @simd.v128_load(a_even, a_sub_off + 48)
          let ao0 = @simd.v128_load(a_odd, a_sub_off)
          let ao1 = @simd.v128_load(a_odd, a_sub_off + 16)
          let ao2 = @simd.v128_load(a_odd, a_sub_off + 32)
          let ao3 = @simd.v128_load(a_odd, a_sub_off + 48)

          let bo = blk_byte_off + sub * 16
          bsum0 = bsum0 +
            q4_dot32_v128(
              @simd.v128_load(b_q4, bc0 + bo),
              mask0f,
              zp0,
              ae0,
              ae1,
              ae2,
              ae3,
              ao0,
              ao1,
              ao2,
              ao3,
            )
          bsum1 = bsum1 +
            q4_dot32_v128(
              @simd.v128_load(b_q4, bc1 + bo),
              mask0f,
              zp1,
              ae0,
              ae1,
              ae2,
              ae3,
              ao0,
              ao1,
              ao2,
              ao3,
            )
          bsum2 = bsum2 +
            q4_dot32_v128(
              @simd.v128_load(b_q4, bc2 + bo),
              mask0f,
              zp2,
              ae0,
              ae1,
              ae2,
              ae3,
              ao0,
              ao1,
              ao2,
              ao3,
            )
          bsum3 = bsum3 +
            q4_dot32_v128(
              @simd.v128_load(b_q4, bc3 + bo),
              mask0f,
              zp3,
              ae0,
              ae1,
              ae2,
              ae3,
              ao0,
              ao1,
              ao2,
              ao3,
            )
        }
        acc0 = acc0 + bsum0 * scales.unsafe_get(sc0 + blk)
        acc1 = acc1 + bsum1 * scales.unsafe_get(sc1 + blk)
        acc2 = acc2 + bsum2 * scales.unsafe_get(sc2 + blk)
        acc3 = acc3 + bsum3 * scales.unsafe_get(sc3 + blk)
      }
      out.unsafe_set(o_row + j, acc0)
      out.unsafe_set(o_row + j + 1, acc1)
      out.unsafe_set(o_row + j + 2, acc2)
      out.unsafe_set(o_row + j + 3, acc3)
      j = j + 4
    }

    // 处理不足 4 列的尾部
    while j < n {
      let bc0 = j * col_stride
      let sc0 = j * n_blocks
      let mut acc0 : Float = match bias {
        Some(bv) => bv.unsafe_get(j)
        None => (0.0 : Float)
      }
      for blk = 0; blk < n_blocks; blk = blk + 1 {
        let zp0 = if has_zp {
          @simd.i8x16_splat(read_q4_zero_point_i8(zp_bytes, j, blk, n_blocks))
        } else {
          default_zp_v
        }
        let mut bsum0 : Float = (0.0 : Float)
        let blk_byte_off = blk * blob_bytes
        for sub = 0; sub < sub_per_blk; sub = sub + 1 {
          let a_sub_off = (blk * sub_per_blk + sub) * 64
          let bo = blk_byte_off + sub * 16
          bsum0 = bsum0 +
            q4_dot32_v128(
              @simd.v128_load(b_q4, bc0 + bo),
              mask0f,
              zp0,
              @simd.v128_load(a_even, a_sub_off),
              @simd.v128_load(a_even, a_sub_off + 16),
              @simd.v128_load(a_even, a_sub_off + 32),
              @simd.v128_load(a_even, a_sub_off + 48),
              @simd.v128_load(a_odd, a_sub_off),
              @simd.v128_load(a_odd, a_sub_off + 16),
              @simd.v128_load(a_odd, a_sub_off + 32),
              @simd.v128_load(a_odd, a_sub_off + 48),
            )
        }
        acc0 = acc0 + bsum0 * scales.unsafe_get(sc0 + blk)
      }
      out.unsafe_set(o_row + j, acc0)
      j = j + 1
    }
  }
}
