///|
/// 标量核心矩阵乘法内核：计算 `out[M, N] = a[M, K] @ b[K, N]`（行主序连续缓冲区）。
/// 采用 `i-k-j` 循环顺序，保证内层循环对 `b` 与 `out` 均为顺序连续访存，
/// 作为全平台标量回退路径及数值正确性黄金基准。
pub fn matmul_2d_scalar_f32(
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
  let zero : Float = 0.0
  for i = 0; i < m * n; i = i + 1 {
    out[out_offset + i] = zero
  }
  for i = 0; i < m; i = i + 1 {
    let a_row = a_offset + i * k
    let out_row = out_offset + i * n
    for p = 0; p < k; p = p + 1 {
      let a_ip = a[a_row + p]
      if a_ip == zero {
        continue
      }
      let b_row = b_offset + p * n
      for j = 0; j < n; j = j + 1 {
        out[out_row + j] = out[out_row + j] + a_ip * b[b_row + j]
      }
    }
  }
}

///|
/// 预打包后的右矩阵 `B[K, N]`（按 4 列 / 16 字节对齐存储在小端序 `FixedArray[Byte]` 中）。
/// 面向推理引擎静态权重场景：模型加载期打包一次，运行期反复复用，彻底消除每次推理的打包拷贝开销。
pub struct PackedMatrixB {
  data : FixedArray[Byte]
  k : Int32
  n : Int32
  /// 向上对齐到 4 的倍数的列宽，保证热循环每次 `v128_load` 均无越界分支
  n_pad : Int32
}

///|
/// 将行主序矩阵 `B[K, N]` 打包为 16 字节对齐的 `PackedMatrixB`。
pub fn pack_matrix_b_f32(
  b : FixedArray[Float],
  b_offset : Int32,
  k : Int32,
  n : Int32,
) -> PackedMatrixB {
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
      // 尾部不足 4 列的部分以 0.0 补齐至一个完整 V128 向量
      let f0 = b[src_row + j]
      let f1 = if j + 1 < n { b[src_row + j + 1] } else { zero }
      let f2 = if j + 2 < n { b[src_row + j + 2] } else { zero }
      let v = @v128.f32x4_const(f0, f1, f2, zero)
      @v128.v128_store(packed, dst_row + j * 4, v)
    }
  }
  { data: packed, k, n, n_pad, }
}

///|
/// 将行主序存储的转置矩阵 `B^T[N, K]`（如 `Gemm(trans_b=true)` 或 `Conv` 权重 `[C_out_g, K]`）
/// 一步转置并打包为 16 字节对齐的 `PackedMatrixB(K, N)`，避免中间转置缓冲区分配。
pub fn pack_matrix_b_transposed_f32(
  b_t : FixedArray[Float],
  b_offset : Int32,
  n : Int32,
  k : Int32,
) -> PackedMatrixB {
  let n_pad = (n + 3) / 4 * 4
  let row_bytes = n_pad * 4
  let packed = FixedArray::make(k * row_bytes, b'\x00')
  let zero : Float = 0.0

  for p = 0; p < k; p = p + 1 {
    let dst_row = p * row_bytes
    let mut j = 0
    while j + 4 <= n {
      let v = @v128.f32x4_const(
        b_t[b_offset + j * k + p],
        b_t[b_offset + (j + 1) * k + p],
        b_t[b_offset + (j + 2) * k + p],
        b_t[b_offset + (j + 3) * k + p],
      )
      @v128.v128_store(packed, dst_row + j * 4, v)
      j = j + 4
    }
    if j < n {
      let f0 = b_t[b_offset + j * k + p]
      let f1 = if j + 1 < n { b_t[b_offset + (j + 1) * k + p] } else { zero }
      let f2 = if j + 2 < n { b_t[b_offset + (j + 2) * k + p] } else { zero }
      let v = @v128.f32x4_const(f0, f1, f2, zero)
      @v128.v128_store(packed, dst_row + j * 4, v)
    }
  }
  { data: packed, k, n, n_pad, }
}

///|
/// 返回预打包矩阵的逻辑维度 `(K, N)`。
pub fn PackedMatrixB::dims(self : PackedMatrixB) -> (Int32, Int32) {
  (self.k, self.n)
}

///|
/// 将单个 V128 累加器写回输出矩阵行 `out[row_off + col ..]`，自动裁剪超过 `n` 的尾部通道。
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
/// 从可选通道偏置数组加载 `[col .. col+4)` 的 V128 初始累加向量（无偏置时返回全 0 向量）。
fn load_bias_v128(
  bias : FixedArray[Float]?,
  bias_offset : Int32,
  col : Int32,
  n : Int32,
) -> V128 {
  let zero : Float = 0.0
  match bias {
    None => @v128.f32x4_splat(zero)
    Some(b) =>
      if col + 4 <= n {
        let base = bias_offset + col
        @v128.f32x4_const(b[base], b[base + 1], b[base + 2], b[base + 3])
      } else {
        let f0 = if col < n { b[bias_offset + col] } else { zero }
        let f1 = if col + 1 < n { b[bias_offset + col + 1] } else { zero }
        let f2 = if col + 2 < n { b[bias_offset + col + 2] } else { zero }
        @v128.f32x4_const(f0, f1, f2, zero)
      }
  }
}

///|
/// 将 4 个连续空间位置（`row .. row+3`）在 4 个输出通道（`col .. col+3`）上的累加结果
/// 直接转置写回 NCHW 布局 `out[out_offset + ch * m + row .. +3]`（每通道连续 4 次顺序写入）。
fn store_v128_4rows_nchw(
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  row : Int32,
  col : Int32,
  n : Int32,
  v0 : V128,
  v1 : V128,
  v2 : V128,
  v3 : V128,
) -> Unit {
  if col + 4 <= n {
    let r0 = out_offset + col * m + row
    out[r0] = @v128.f32x4_extract_lane(v0, 0)
    out[r0 + 1] = @v128.f32x4_extract_lane(v1, 0)
    out[r0 + 2] = @v128.f32x4_extract_lane(v2, 0)
    out[r0 + 3] = @v128.f32x4_extract_lane(v3, 0)

    let r1 = r0 + m
    out[r1] = @v128.f32x4_extract_lane(v0, 1)
    out[r1 + 1] = @v128.f32x4_extract_lane(v1, 1)
    out[r1 + 2] = @v128.f32x4_extract_lane(v2, 1)
    out[r1 + 3] = @v128.f32x4_extract_lane(v3, 1)

    let r2 = r1 + m
    out[r2] = @v128.f32x4_extract_lane(v0, 2)
    out[r2 + 1] = @v128.f32x4_extract_lane(v1, 2)
    out[r2 + 2] = @v128.f32x4_extract_lane(v2, 2)
    out[r2 + 3] = @v128.f32x4_extract_lane(v3, 2)

    let r3 = r2 + m
    out[r3] = @v128.f32x4_extract_lane(v0, 3)
    out[r3 + 1] = @v128.f32x4_extract_lane(v1, 3)
    out[r3 + 2] = @v128.f32x4_extract_lane(v2, 3)
    out[r3 + 3] = @v128.f32x4_extract_lane(v3, 3)
  } else {
    if col < n {
      let r0 = out_offset + col * m + row
      out[r0] = @v128.f32x4_extract_lane(v0, 0)
      out[r0 + 1] = @v128.f32x4_extract_lane(v1, 0)
      out[r0 + 2] = @v128.f32x4_extract_lane(v2, 0)
      out[r0 + 3] = @v128.f32x4_extract_lane(v3, 0)
    }
    if col + 1 < n {
      let r1 = out_offset + (col + 1) * m + row
      out[r1] = @v128.f32x4_extract_lane(v0, 1)
      out[r1 + 1] = @v128.f32x4_extract_lane(v1, 1)
      out[r1 + 2] = @v128.f32x4_extract_lane(v2, 1)
      out[r1 + 3] = @v128.f32x4_extract_lane(v3, 1)
    }
    if col + 2 < n {
      let r2 = out_offset + (col + 2) * m + row
      out[r2] = @v128.f32x4_extract_lane(v0, 2)
      out[r2 + 1] = @v128.f32x4_extract_lane(v1, 2)
      out[r2 + 2] = @v128.f32x4_extract_lane(v2, 2)
      out[r2 + 3] = @v128.f32x4_extract_lane(v3, 2)
    }
  }
}

///|
/// 将单个空间位置 `row` 在最多 4 个输出通道上的累加结果写回 NCHW 输出缓冲。
fn store_v128_1row_nchw(
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  row : Int32,
  col : Int32,
  n : Int32,
  v : V128,
) -> Unit {
  if col < n {
    out[out_offset + col * m + row] = @v128.f32x4_extract_lane(v, 0)
  }
  if col + 1 < n {
    out[out_offset + (col + 1) * m + row] = @v128.f32x4_extract_lane(v, 1)
  }
  if col + 2 < n {
    out[out_offset + (col + 2) * m + row] = @v128.f32x4_extract_lane(v, 2)
  }
  if col + 3 < n {
    out[out_offset + (col + 3) * m + row] = @v128.f32x4_extract_lane(v, 3)
  }
}

///|
/// 使用已预打包的 `PackedMatrixB` 执行 `NC = 64` 列分块 + `4×8` 寄存器分块 `V128` SIMD GEMM。
///
/// 核心微内核设计（参考 BLIS / GotoBLAS）：
/// - 主循环每步处理 4 行 A × 8 列 B，在寄存器中维护 8 个 `V128` 累加器；
/// - 每次 K 步进仅需 2 次 `v128_load` + 4 次 `f32x4_splat` 即可完成 8 条 `f32x4_relaxed_madd`
///   （32 次浮点乘加），整个 K 归约过程中对输出矩阵 C 零写回；
/// - 自动处理 `4×4` 边列块与 `< 4` 行尾部（典型如 Batch=1 全连接层）。
pub fn matmul_2d_prepacked_simd_f32(
  a : FixedArray[Float],
  a_offset : Int32,
  packed_b : PackedMatrixB,
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
) -> Unit {
  let k = packed_b.k
  let n = packed_b.n
  let n_pad = packed_b.n_pad
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

  let b_packed = packed_b.data
  let b_row_bytes = n_pad * 4
  let zero_vec = @v128.f32x4_splat((0.0 : Float))

  // 列分块宽度 NC = 64（16 个 V128 向量），使当前 B 面板常驻 L1/L2 缓存
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
      while j + 8 <= j_end {
        let mut c00 = zero_vec
        let mut c01 = zero_vec
        let mut c10 = zero_vec
        let mut c11 = zero_vec
        let mut c20 = zero_vec
        let mut c21 = zero_vec
        let mut c30 = zero_vec
        let mut c31 = zero_vec

        let mut b_ptr = j * 4
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

    // --- 尾部不足 4 行的剩余行（典型如 Batch=1 全连接层）---
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
/// 预打包 NCHW 卷积微内核写回前的寄存器级融合激活配置。
/// 在 `V128` 累加器写回内存前原地执行，避免独立激活算子的整张特征图读写开销。
#derive(Debug, Eq)
pub(all) enum Epilogue {
  /// 无融合激活
  NoEpilogue
  /// 融合 ReLU（使用硬件 `f32x4_pmax(0.0, acc)`）
  Relu
  /// 融合区间裁剪（`min(max, max(min, acc))`，典型如 MobileNetV2 ReLU6 `[0.0, 6.0]`）
  Clip(min~ : Float, max~ : Float)
}

///|
/// 对单个 V128 累加器应用融合激活（无激活时零开销透传）。
fn apply_epilogue_v128(v : V128, epilogue : Epilogue) -> V128 {
  match epilogue {
    NoEpilogue => v
    Relu => @v128.f32x4_pmax(@v128.f32x4_splat((0.0 : Float)), v)
    Clip(min~, max~) =>
      @v128.f32x4_pmin(
        @v128.f32x4_splat(max),
        @v128.f32x4_pmax(@v128.f32x4_splat(min), v),
      )
  }
}

///|
/// 对单个标量 Float32 应用融合激活（用于 `k == 0` 等边界路径）。
pub fn apply_epilogue_scalar(x : Float, epilogue : Epilogue) -> Float {
  match epilogue {
    NoEpilogue => x
    Relu => if x > (0.0 : Float) { x } else { (0.0 : Float) }
    Clip(min~, max~) => {
      let lo = if x > min { x } else { min }
      if lo < max {
        lo
      } else {
        max
      }
    }
  }
}

///|
/// 面向预打包 2D 卷积的专用 V128 SIMD GEMM 内核：
/// 计算 `out_nchw[N, M] = epilogue((A[M, K] @ packed_b[K, N])^T + bias[N])`，其中：
/// - `A[M, K]` 为 `im2col_2d_nhwc_f32` 生成的转置感受野矩阵（`M = out_spatial`）；
/// - `packed_b[K, N]` 为模型加载期一次性预打包的静态卷积权重 `W_g^T[K, C_out_g]`；
/// - 可选 `bias` 直接融合为 V128 累加器初值，`epilogue` 在写回内存前原地执行于 SIMD 寄存器中；
/// - 结果直接按 NCHW 通道优先顺序写回 `out`，无需中间转置缓冲。
pub fn matmul_2d_prepacked_nchw_simd_f32(
  a : FixedArray[Float],
  a_offset : Int32,
  packed_b : PackedMatrixB,
  bias : FixedArray[Float]?,
  bias_offset : Int32,
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  epilogue~ : Epilogue = NoEpilogue,
) -> Unit {
  let k = packed_b.k
  let n = packed_b.n
  let n_pad = packed_b.n_pad
  if m == 0 || n == 0 {
    return
  }
  if k == 0 {
    let zero : Float = 0.0
    for ch = 0; ch < n; ch = ch + 1 {
      let raw = match bias {
        Some(b) => b[bias_offset + ch]
        None => zero
      }
      let bv = apply_epilogue_scalar(raw, epilogue)
      let ch_base = out_offset + ch * m
      for i = 0; i < m; i = i + 1 {
        out[ch_base + i] = bv
      }
    }
    return
  }

  let b_packed = packed_b.data
  let b_row_bytes = n_pad * 4

  // 外层按 8 通道面板遍历：bias 加载外提、当前 8 列 B 微面板常驻 L1、同时按通道顺序流式写回 NCHW 输出
  let mut j = 0
  while j + 8 <= n_pad {
    let init0 = load_bias_v128(bias, bias_offset, j, n)
    let init1 = load_bias_v128(bias, bias_offset, j + 4, n)
    let b_col_off = j * 4

    let mut i = 0
    while i + 4 <= m {
      let a_r0 = a_offset + i * k
      let a_r1 = a_r0 + k
      let a_r2 = a_r1 + k
      let a_r3 = a_r2 + k

      let mut c00 = init0
      let mut c01 = init1
      let mut c10 = init0
      let mut c11 = init1
      let mut c20 = init0
      let mut c21 = init1
      let mut c30 = init0
      let mut c31 = init1

      let mut b_ptr = b_col_off
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

      match epilogue {
        NoEpilogue => ()
        _ => {
          c00 = apply_epilogue_v128(c00, epilogue)
          c01 = apply_epilogue_v128(c01, epilogue)
          c10 = apply_epilogue_v128(c10, epilogue)
          c11 = apply_epilogue_v128(c11, epilogue)
          c20 = apply_epilogue_v128(c20, epilogue)
          c21 = apply_epilogue_v128(c21, epilogue)
          c30 = apply_epilogue_v128(c30, epilogue)
          c31 = apply_epilogue_v128(c31, epilogue)
        }
      }

      store_v128_4rows_nchw(out, out_offset, m, i, j, n, c00, c10, c20, c30)
      store_v128_4rows_nchw(out, out_offset, m, i, j + 4, n, c01, c11, c21, c31)
      i = i + 4
    }

    while i < m {
      let a_r = a_offset + i * k
      let mut c0 = init0
      let mut c1 = init1
      let mut b_ptr = b_col_off
      for p = 0; p < k; p = p + 1 {
        let av = @v128.f32x4_splat(a[a_r + p])
        let bv0 = @v128.v128_load(b_packed, b_ptr)
        let bv1 = @v128.v128_load(b_packed, b_ptr + 16)
        c0 = @v128.f32x4_relaxed_madd(av, bv0, c0)
        c1 = @v128.f32x4_relaxed_madd(av, bv1, c1)
        b_ptr = b_ptr + b_row_bytes
      }
      c0 = apply_epilogue_v128(c0, epilogue)
      c1 = apply_epilogue_v128(c1, epilogue)
      store_v128_1row_nchw(out, out_offset, m, i, j, n, c0)
      store_v128_1row_nchw(out, out_offset, m, i, j + 4, n, c1)
      i = i + 1
    }

    j = j + 8
  }

  // 尾部不足 8 通道的 4 通道面板
  while j + 4 <= n_pad {
    let init0 = load_bias_v128(bias, bias_offset, j, n)
    let b_col_off = j * 4

    let mut i = 0
    while i + 4 <= m {
      let a_r0 = a_offset + i * k
      let a_r1 = a_r0 + k
      let a_r2 = a_r1 + k
      let a_r3 = a_r2 + k

      let mut c0 = init0
      let mut c1 = init0
      let mut c2 = init0
      let mut c3 = init0
      let mut b_ptr = b_col_off
      for p = 0; p < k; p = p + 1 {
        let bv = @v128.v128_load(b_packed, b_ptr)
        c0 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r0 + p]), bv, c0)
        c1 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r1 + p]), bv, c1)
        c2 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r2 + p]), bv, c2)
        c3 = @v128.f32x4_relaxed_madd(@v128.f32x4_splat(a[a_r3 + p]), bv, c3)
        b_ptr = b_ptr + b_row_bytes
      }
      match epilogue {
        NoEpilogue => ()
        _ => {
          c0 = apply_epilogue_v128(c0, epilogue)
          c1 = apply_epilogue_v128(c1, epilogue)
          c2 = apply_epilogue_v128(c2, epilogue)
          c3 = apply_epilogue_v128(c3, epilogue)
        }
      }
      store_v128_4rows_nchw(out, out_offset, m, i, j, n, c0, c1, c2, c3)
      i = i + 4
    }

    while i < m {
      let a_r = a_offset + i * k
      let mut c0 = init0
      let mut b_ptr = b_col_off
      for p = 0; p < k; p = p + 1 {
        let av = @v128.f32x4_splat(a[a_r + p])
        let bv = @v128.v128_load(b_packed, b_ptr)
        c0 = @v128.f32x4_relaxed_madd(av, bv, c0)
        b_ptr = b_ptr + b_row_bytes
      }
      c0 = apply_epilogue_v128(c0, epilogue)
      store_v128_1row_nchw(out, out_offset, m, i, j, n, c0)
      i = i + 1
    }

    j = j + 4
  }
}

///|
/// 即时打包版 `V128` SIMD 矩阵乘法：`out[M, N] = a[M, K] @ b[K, N]`。
/// 适用于动态矩阵 B（如卷积 `im2col` 展开后的临时矩阵或 Attention Q@K^T）。
pub fn matmul_2d_simd_f32(
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
  let packed_b = pack_matrix_b_f32(b, b_offset, k, n)
  matmul_2d_prepacked_simd_f32(a, a_offset, packed_b, out, out_offset, m)
}
