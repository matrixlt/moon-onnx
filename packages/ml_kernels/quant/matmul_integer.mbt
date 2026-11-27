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
