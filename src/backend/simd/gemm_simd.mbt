///|
/// 将矩阵 B[K, N] 打包为列数按 4 向上对齐（16 字节对齐）的小端序 `FixedArray[Byte]`。
/// 对齐后的列宽 `n_pad` 保证了热循环内部每次 `@v128.v128_load` 均不会越界，
/// 无需在最内层 `k` 循环中插入任何边界分支判断。
fn pack_b_aligned4(
  b : FixedArray[Float],
  b_offset : Int32,
  k : Int32,
  n : Int32,
) -> (FixedArray[Byte], Int32) {
  let n_pad = (n + 3) / 4 * 4
  let row_bytes = n_pad * 4
  let packed = FixedArray::make(k * row_bytes, b'\x00')
  let zero : Float = 0.0

  for p = 0; p < k; p = p + 1 {
    let src_row = b_offset + p * n
    let dst_row = p * row_bytes
    let mut j = 0
    while j + 4 <= n {
      let v = @v128.f32x4_const(
        b[src_row + j],
        b[src_row + j + 1],
        b[src_row + j + 2],
        b[src_row + j + 3],
      )
      @v128.v128_store(packed, dst_row + j * 4, v)
      j = j + 4
    }
    if j < n {
      // 尾部不足 4 列的部分用 0.0 补齐至一个完整 V128 向量
      let f0 = b[src_row + j]
      let f1 = if j + 1 < n { b[src_row + j + 1] } else { zero }
      let f2 = if j + 2 < n { b[src_row + j + 2] } else { zero }
      let v = @v128.f32x4_const(f0, f1, f2, zero)
      @v128.v128_store(packed, dst_row + j * 4, v)
    }
  }
  (packed, n_pad)
}

///|
/// 将一个 V128 累加器写回输出矩阵行 `out[row_off + col ..]`，自动裁剪超过 `n` 的尾部通道。
fn store_v128_clamped(
  out : FixedArray[Float],
  row_off : Int32,
  col : Int32,
  n : Int32,
  v : V128,
) -> Unit {
  if col + 4 <= n {
    out[row_off + col] = @v128.f32x4_extract_lane(v, 0)
    out[row_off + col + 1] = @v128.f32x4_extract_lane(v, 1)
    out[row_off + col + 2] = @v128.f32x4_extract_lane(v, 2)
    out[row_off + col + 3] = @v128.f32x4_extract_lane(v, 3)
  } else {
    if col < n {
      out[row_off + col] = @v128.f32x4_extract_lane(v, 0)
    }
    if col + 1 < n {
      out[row_off + col + 1] = @v128.f32x4_extract_lane(v, 1)
    }
    if col + 2 < n {
      out[row_off + col + 2] = @v128.f32x4_extract_lane(v, 2)
    }
  }
}

///|
/// 基于 MoonBit `V128`（`f32x4`）的分块寄存器微内核矩阵乘法：`out[M, N] = a[M, K] @ b[K, N]`。
///
/// 核心优化策略（参考 BLIS / GotoBLAS 思路）：
/// 1. 预先将 B 打包为 4 列对齐的连续字节数组 `b_packed`，热循环直接用 `v128_load` 批量载入；
/// 2. 外层对 N 维度按 `NC = 64` 切块，保持当前 B 面板常驻 L1/L2 缓存；
/// 3. 主微内核采用 `4 x 8` 寄存器分块：同时维护 8 个 `V128` 累加器（覆盖 4 行 × 8 列），
///    每次 K 步进仅需 2 次 `v128_load` + 4 次 `f32x4_splat` 即可完成 8 条 `f32x4_relaxed_madd`
///    （共 32 次浮点乘加），整个 K 归约过程中对输出矩阵 C 零写回。
pub fn matmul_2d_raw_simd_f32(
  a : FixedArray[Float],
  a_offset : Int32,
  b : FixedArray[Float],
  b_offset : Int32,
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  k : Int32,
  n : Int32,
) -> Unit {
  if m == 0 || n == 0 {
    return
  }
  if k == 0 {
    let zero : Float = 0.0
    for idx = 0; idx < m * n; idx = idx + 1 {
      out[out_offset + idx] = zero
    }
    return
  }

  let (b_packed, n_pad) = pack_b_aligned4(b, b_offset, k, n)
  let b_row_bytes = n_pad * 4
  let zero_vec = @v128.f32x4_splat((0.0 : Float))

  // 列分块宽度（64 列 = 16 个 V128 向量），提升大矩阵下的缓存局部性
  let nc = 64
  let mut jc = 0
  while jc < n_pad {
    let j_end = if jc + nc < n_pad { jc + nc } else { n_pad }

    // --- 主循环：每次处理 4 行 A ---
    let mut i = 0
    while i + 4 <= m {
      let a_r0 = a_offset + i * k
      let a_r1 = a_r0 + k
      let a_r2 = a_r1 + k
      let a_r3 = a_r2 + k
      let c_r0 = out_offset + i * n
      let c_r1 = c_r0 + n
      let c_r2 = c_r1 + n
      let c_r3 = c_r2 + n

      let mut j = jc
      // 4 x 8 寄存器分块微内核（8 个 V128 累加器）
      while j + 8 <= j_end {
        let mut c00 = zero_vec
        let mut c01 = zero_vec
        let mut c10 = zero_vec
        let mut c11 = zero_vec
        let mut c20 = zero_vec
        let mut c21 = zero_vec
        let mut c30 = zero_vec
        let mut c31 = zero_vec

        let j_byte = j * 4
        let mut b_ptr = j_byte
        for p = 0; p < k; p = p + 1 {
          let bv0 = @v128.v128_load(b_packed, b_ptr)
          let bv1 = @v128.v128_load(b_packed, b_ptr + 16)
          let av0 = @v128.f32x4_splat(a[a_r0 + p])
          let av1 = @v128.f32x4_splat(a[a_r1 + p])
          let av2 = @v128.f32x4_splat(a[a_r2 + p])
          let av3 = @v128.f32x4_splat(a[a_r3 + p])

          c00 = @v128.f32x4_relaxed_madd(av0, bv0, c00)
          c01 = @v128.f32x4_relaxed_madd(av0, bv1, c01)
          c10 = @v128.f32x4_relaxed_madd(av1, bv0, c10)
          c11 = @v128.f32x4_relaxed_madd(av1, bv1, c11)
          c20 = @v128.f32x4_relaxed_madd(av2, bv0, c20)
          c21 = @v128.f32x4_relaxed_madd(av2, bv1, c21)
          c30 = @v128.f32x4_relaxed_madd(av3, bv0, c30)
          c31 = @v128.f32x4_relaxed_madd(av3, bv1, c31)
          b_ptr = b_ptr + b_row_bytes
        }

        store_v128_clamped(out, c_r0, j, n, c00)
        store_v128_clamped(out, c_r0, j + 4, n, c01)
        store_v128_clamped(out, c_r1, j, n, c10)
        store_v128_clamped(out, c_r1, j + 4, n, c11)
        store_v128_clamped(out, c_r2, j, n, c20)
        store_v128_clamped(out, c_r2, j + 4, n, c21)
        store_v128_clamped(out, c_r3, j, n, c30)
        store_v128_clamped(out, c_r3, j + 4, n, c31)
        j = j + 8
      }

      // 4 x 4 尾部列块（当 j_end 不是 8 的倍数时执行一次）
      while j + 4 <= j_end {
        let mut c0 = zero_vec
        let mut c1 = zero_vec
        let mut c2 = zero_vec
        let mut c3 = zero_vec
        let mut b_ptr = j * 4
        for p = 0; p < k; p = p + 1 {
          let bv = @v128.v128_load(b_packed, b_ptr)
          c0 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r0 + p]), bv, c0)
          c1 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r1 + p]), bv, c1)
          c2 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r2 + p]), bv, c2)
          c3 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r3 + p]), bv, c3)
          b_ptr = b_ptr + b_row_bytes
        }
        store_v128_clamped(out, c_r0, j, n, c0)
        store_v128_clamped(out, c_r1, j, n, c1)
        store_v128_clamped(out, c_r2, j, n, c2)
        store_v128_clamped(out, c_r3, j, n, c3)
        j = j + 4
      }
      i = i + 4
    }

    // --- 尾部不足 4 行的剩余行（逐行使用 1 x 8 + 1 x 4 V128 微内核，典型如 Batch=1 的全连接层）---
    while i < m {
      let a_r = a_offset + i * k
      let c_r = out_offset + i * n
      let mut j = jc
      while j + 8 <= j_end {
        let mut c0 = zero_vec
        let mut c1 = zero_vec
        let mut b_ptr = j * 4
        for p = 0; p < k; p = p + 1 {
          let av = @v128.f32x4_splat(a[a_r + p])
          let bv0 = @v128.v128_load(b_packed, b_ptr)
          let bv1 = @v128.v128_load(b_packed, b_ptr + 16)
          c0 = @v128.f32x4_relaxed_madd(av, bv0, c0)
          c1 = @v128.f32x4_relaxed_madd(av, bv1, c1)
          b_ptr = b_ptr + b_row_bytes
        }
        store_v128_clamped(out, c_r, j, n, c0)
        store_v128_clamped(out, c_r, j + 4, n, c1)
        j = j + 8
      }
      while j + 4 <= j_end {
        let mut c0 = zero_vec
        let mut b_ptr = j * 4
        for p = 0; p < k; p = p + 1 {
          let av = @v128.f32x4_splat(a[a_r + p])
          let bv = @v128.v128_load(b_packed, b_ptr)
          c0 = @v128.f32x4_relaxed_madd(av, bv, c0)
          b_ptr = b_ptr + b_row_bytes
        }
        store_v128_clamped(out, c_r, j, n, c0)
        j = j + 4
      }
      i = i + 1
    }

    jc = jc + nc
  }
}

///|
/// V128 SIMD 版 ONNX Gemm 算子（复用 V128 分块矩阵乘法内核）。
pub fn gemm_f32_simd(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  c : @tensor.Tensor?,
  alpha~ : Float = (1.0 : Float),
  beta~ : Float = (1.0 : Float),
  trans_a~ : Bool = false,
  trans_b~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.rank() != 2 || b.rank() != 2 {
    raise @tensor.ShapeMismatch(
      "gemm_f32_simd: A and B must be 2D matrices, got ranks \{a.rank()} and \{b.rank()}",
    )
  }
  let a_mat = if trans_a { a.transpose([1, 0]).contiguous() } else { a }
  let b_mat = if trans_b { b.transpose([1, 0]).contiguous() } else { b }
  let sa = a_mat.shape()
  let sb = b_mat.shape()
  let m = sa[0]
  let k = sa[1]
  let n = sb[1]
  if k != sb[0] {
    raise @tensor.ShapeMismatch(
      "gemm_f32_simd: inner dimension mismatch (\{k} vs \{sb[0]})",
    )
  }

  let out = FixedArray::make(m * n, (0.0 : Float))
  matmul_2d_raw_simd_f32(
    a_mat.to_f32_buffer(),
    0,
    b_mat.to_f32_buffer(),
    0,
    out,
    0,
    m,
    k,
    n,
  )

  let one : Float = 1.0
  if alpha != one {
    for i = 0; i < m * n; i = i + 1 {
      out[i] = out[i] * alpha
    }
  }
  let prod = @tensor.Tensor::from_f32([m, n], out)
  let zero : Float = 0.0
  match c {
    None => prod
    Some(bias) =>
      if beta == zero {
        prod
      } else if beta == one {
        add_f32_simd(prod, bias)
      } else {
        let b_buf = bias.to_f32_buffer()
        let scaled = FixedArray::makei(b_buf.length(), fn(i) { b_buf[i] * beta })
        add_f32_simd(prod, @tensor.Tensor::from_f32(bias.shape(), scaled))
      }
  }
}

///|
/// V128 SIMD 版 ONNX MatMul 算子（2D 走 V128 分块内核，高维/1D 回退至通用批量广播）。
pub fn matmul_f32_simd(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.rank() == 2 && b.rank() == 2 {
    let sa = a.shape()
    let sb = b.shape()
    let m = sa[0]
    let k = sa[1]
    if k != sb[0] {
      raise @tensor.ShapeMismatch(
        "matmul_f32_simd: inner dimension mismatch (\{k} vs \{sb[0]})",
      )
    }
    let n = sb[1]
    let out = FixedArray::make(m * n, (0.0 : Float))
    matmul_2d_raw_simd_f32(
      a.to_f32_buffer(),
      0,
      b.to_f32_buffer(),
      0,
      out,
      0,
      m,
      k,
      n,
    )
    return @tensor.Tensor::from_f32([m, n], out)
  }
  @ops.matmul_f32(a, b)
}

///|
/// V128 SIMD 版 ONNX Conv2D 算子：直接将 V128 分块 GEMM 注入 `im2col + GEMM` 卷积框架。
pub fn conv2d_f32_simd(
  x : @tensor.Tensor,
  w : @tensor.Tensor,
  bias : @tensor.Tensor?,
  strides : Array[Int32],
  pads : Array[Int32],
  dilations : Array[Int32],
  group : Int32,
  auto_pad : @ir.AutoPad,
) -> @tensor.Tensor raise @tensor.TensorError {
  @ops.conv2d_f32_with_gemm(
    x, w, bias, strides, pads, dilations, group, auto_pad, matmul_2d_raw_simd_f32,
  )
}
