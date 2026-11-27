///|
/// 标量核心矩阵乘法内核：计算 `out[M, N] = a[M, K] @ b[K, N]`（行主序连续缓冲区）。
/// 采用 `i-k-j` 循环顺序与入口一次性边界校验，保证内层循环对 `b` 与 `out` 均为零边界检查的顺序连续访存，
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
  if m == 0 || n == 0 {
    return
  }
  check_slice_bounds(out.length(), out_offset, m * n, "matmul_2d_scalar(out)")
  let zero : Float = 0.0
  for i = 0; i < m * n; i = i + 1 {
    out.unsafe_set(out_offset + i, zero)
  }
  if k == 0 {
    return
  }
  check_slice_bounds(a.length(), a_offset, m * k, "matmul_2d_scalar(a)")
  check_slice_bounds(b.length(), b_offset, k * n, "matmul_2d_scalar(b)")
  for i = 0; i < m; i = i + 1 {
    let a_row = a_offset + i * k
    let out_row = out_offset + i * n
    for p = 0; p < k; p = p + 1 {
      let a_ip = a.unsafe_get(a_row + p)
      if a_ip == zero {
        continue
      }
      let b_row = b_offset + p * n
      for j = 0; j < n; j = j + 1 {
        let oi = out_row + j
        out.unsafe_set(oi, out.unsafe_get(oi) + a_ip * b.unsafe_get(b_row + j))
      }
    }
  }
}

///|
/// 使用已预打包的 BLIS 8 列连续微面板 `PackedMatrixB` 执行 `6×8` 寄存器分块 + `K` 维 2× 展开 `V128` SIMD GEMM。
///
/// 核心微内核设计（参考 BLIS / GotoBLAS / XNNPACK Wasm SIMD）：
/// - 入口执行一次性 O(1) 边界校验，内层循环全部使用 `unsafe_get` / `unsafe_set`，消除 Wasm 线性内存热路径边界检查；
/// - 主循环采用 `6×8` 寄存器分块（12 个 `V128` 累加器 + 2 条 B 向量 + 1 个复用 A 广播 = 15 XMM 寄存器，零寄存器溢出），
///   相比 `4×8` 减少 33% 的 `B` 面板内存加载；
/// - 内循环按 `K` 维 2× 展开降低分支与指针自增开销，自动平滑衔接 `4×8` 与 `1×8` 尾部路径。
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
  check_slice_bounds(
    out.length(),
    out_offset,
    m * n,
    "matmul_2d_prepacked_simd(out)",
  )
  if k == 0 {
    let zero : Float = 0.0
    for idx = 0; idx < m * n; idx = idx + 1 {
      out.unsafe_set(out_offset + idx, zero)
    }
    return
  }
  check_slice_bounds(a.length(), a_offset, m * k, "matmul_2d_prepacked_simd(a)")

  let b_packed = packed_b.data
  let panel_bytes = k * 32
  let k2 = k - k % 2
  let zero_vec = @simd.f32x4_splat((0.0 : Float))

  let mut j = 0
  let mut panel_base = 0
  while j < n_pad {
    // --- 主循环：每次处理 6 行 A × 8 列 B（12 个 V128 累加器，K 维 2× 展开）---
    let mut i = 0
    while i + 6 <= m {
      let a_r0 = a_offset + i * k
      let a_r1 = a_r0 + k
      let a_r2 = a_r1 + k
      let a_r3 = a_r2 + k
      let a_r4 = a_r3 + k
      let a_r5 = a_r4 + k

      let mut c00 = zero_vec
      let mut c01 = zero_vec
      let mut c10 = zero_vec
      let mut c11 = zero_vec
      let mut c20 = zero_vec
      let mut c21 = zero_vec
      let mut c30 = zero_vec
      let mut c31 = zero_vec
      let mut c40 = zero_vec
      let mut c41 = zero_vec
      let mut c50 = zero_vec
      let mut c51 = zero_vec

      let mut b_ptr = panel_base
      let mut p = 0
      while p < k2 {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let mut av = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        c00 = @simd.f32x4_relaxed_madd(av, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av, bv1, c01)
        av = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        c10 = @simd.f32x4_relaxed_madd(av, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av, bv1, c11)
        av = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        c20 = @simd.f32x4_relaxed_madd(av, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av, bv1, c21)
        av = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c30 = @simd.f32x4_relaxed_madd(av, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av, bv1, c31)
        av = @simd.f32x4_splat(a.unsafe_get(a_r4 + p))
        c40 = @simd.f32x4_relaxed_madd(av, bv0, c40)
        c41 = @simd.f32x4_relaxed_madd(av, bv1, c41)
        av = @simd.f32x4_splat(a.unsafe_get(a_r5 + p))
        c50 = @simd.f32x4_relaxed_madd(av, bv0, c50)
        c51 = @simd.f32x4_relaxed_madd(av, bv1, c51)

        let bv2 = @simd.v128_load(b_packed, b_ptr + 32)
        let bv3 = @simd.v128_load(b_packed, b_ptr + 48)
        let p1 = p + 1
        av = @simd.f32x4_splat(a.unsafe_get(a_r0 + p1))
        c00 = @simd.f32x4_relaxed_madd(av, bv2, c00)
        c01 = @simd.f32x4_relaxed_madd(av, bv3, c01)
        av = @simd.f32x4_splat(a.unsafe_get(a_r1 + p1))
        c10 = @simd.f32x4_relaxed_madd(av, bv2, c10)
        c11 = @simd.f32x4_relaxed_madd(av, bv3, c11)
        av = @simd.f32x4_splat(a.unsafe_get(a_r2 + p1))
        c20 = @simd.f32x4_relaxed_madd(av, bv2, c20)
        c21 = @simd.f32x4_relaxed_madd(av, bv3, c21)
        av = @simd.f32x4_splat(a.unsafe_get(a_r3 + p1))
        c30 = @simd.f32x4_relaxed_madd(av, bv2, c30)
        c31 = @simd.f32x4_relaxed_madd(av, bv3, c31)
        av = @simd.f32x4_splat(a.unsafe_get(a_r4 + p1))
        c40 = @simd.f32x4_relaxed_madd(av, bv2, c40)
        c41 = @simd.f32x4_relaxed_madd(av, bv3, c41)
        av = @simd.f32x4_splat(a.unsafe_get(a_r5 + p1))
        c50 = @simd.f32x4_relaxed_madd(av, bv2, c50)
        c51 = @simd.f32x4_relaxed_madd(av, bv3, c51)

        b_ptr = b_ptr + 64
        p = p + 2
      }
      if p < k {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let mut av = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        c00 = @simd.f32x4_relaxed_madd(av, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av, bv1, c01)
        av = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        c10 = @simd.f32x4_relaxed_madd(av, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av, bv1, c11)
        av = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        c20 = @simd.f32x4_relaxed_madd(av, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av, bv1, c21)
        av = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c30 = @simd.f32x4_relaxed_madd(av, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av, bv1, c31)
        av = @simd.f32x4_splat(a.unsafe_get(a_r4 + p))
        c40 = @simd.f32x4_relaxed_madd(av, bv0, c40)
        c41 = @simd.f32x4_relaxed_madd(av, bv1, c41)
        av = @simd.f32x4_splat(a.unsafe_get(a_r5 + p))
        c50 = @simd.f32x4_relaxed_madd(av, bv0, c50)
        c51 = @simd.f32x4_relaxed_madd(av, bv1, c51)
      }

      let c_r0 = out_offset + i * n
      let c_r1 = c_r0 + n
      let c_r2 = c_r1 + n
      let c_r3 = c_r2 + n
      let c_r4 = c_r3 + n
      let c_r5 = c_r4 + n
      store_v128_clamped(out, c_r0, j, n, c00)
      store_v128_clamped(out, c_r0, j + 4, n, c01)
      store_v128_clamped(out, c_r1, j, n, c10)
      store_v128_clamped(out, c_r1, j + 4, n, c11)
      store_v128_clamped(out, c_r2, j, n, c20)
      store_v128_clamped(out, c_r2, j + 4, n, c21)
      store_v128_clamped(out, c_r3, j, n, c30)
      store_v128_clamped(out, c_r3, j + 4, n, c31)
      store_v128_clamped(out, c_r4, j, n, c40)
      store_v128_clamped(out, c_r4, j + 4, n, c41)
      store_v128_clamped(out, c_r5, j, n, c50)
      store_v128_clamped(out, c_r5, j + 4, n, c51)
      i = i + 6
    }

    // --- 4 行尾部路径（典型如 seq=4 prefill 或剩余 4~5 行）---
    if i + 4 <= m {
      let a_r0 = a_offset + i * k
      let a_r1 = a_r0 + k
      let a_r2 = a_r1 + k
      let a_r3 = a_r2 + k
      let mut c00 = zero_vec
      let mut c01 = zero_vec
      let mut c10 = zero_vec
      let mut c11 = zero_vec
      let mut c20 = zero_vec
      let mut c21 = zero_vec
      let mut c30 = zero_vec
      let mut c31 = zero_vec
      let mut b_ptr = panel_base
      let mut p = 0
      while p < k2 {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let av0 = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        let av1 = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        let av2 = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        let av3 = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c00 = @simd.f32x4_relaxed_madd(av0, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av0, bv1, c01)
        c10 = @simd.f32x4_relaxed_madd(av1, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av1, bv1, c11)
        c20 = @simd.f32x4_relaxed_madd(av2, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av2, bv1, c21)
        c30 = @simd.f32x4_relaxed_madd(av3, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av3, bv1, c31)

        let bv2 = @simd.v128_load(b_packed, b_ptr + 32)
        let bv3 = @simd.v128_load(b_packed, b_ptr + 48)
        let p1 = p + 1
        let av0b = @simd.f32x4_splat(a.unsafe_get(a_r0 + p1))
        let av1b = @simd.f32x4_splat(a.unsafe_get(a_r1 + p1))
        let av2b = @simd.f32x4_splat(a.unsafe_get(a_r2 + p1))
        let av3b = @simd.f32x4_splat(a.unsafe_get(a_r3 + p1))
        c00 = @simd.f32x4_relaxed_madd(av0b, bv2, c00)
        c01 = @simd.f32x4_relaxed_madd(av0b, bv3, c01)
        c10 = @simd.f32x4_relaxed_madd(av1b, bv2, c10)
        c11 = @simd.f32x4_relaxed_madd(av1b, bv3, c11)
        c20 = @simd.f32x4_relaxed_madd(av2b, bv2, c20)
        c21 = @simd.f32x4_relaxed_madd(av2b, bv3, c21)
        c30 = @simd.f32x4_relaxed_madd(av3b, bv2, c30)
        c31 = @simd.f32x4_relaxed_madd(av3b, bv3, c31)

        b_ptr = b_ptr + 64
        p = p + 2
      }
      if p < k {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let av0 = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        let av1 = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        let av2 = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        let av3 = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c00 = @simd.f32x4_relaxed_madd(av0, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av0, bv1, c01)
        c10 = @simd.f32x4_relaxed_madd(av1, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av1, bv1, c11)
        c20 = @simd.f32x4_relaxed_madd(av2, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av2, bv1, c21)
        c30 = @simd.f32x4_relaxed_madd(av3, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av3, bv1, c31)
      }
      let c_r0 = out_offset + i * n
      let c_r1 = c_r0 + n
      let c_r2 = c_r1 + n
      let c_r3 = c_r2 + n
      store_v128_clamped(out, c_r0, j, n, c00)
      store_v128_clamped(out, c_r0, j + 4, n, c01)
      store_v128_clamped(out, c_r1, j, n, c10)
      store_v128_clamped(out, c_r1, j + 4, n, c11)
      store_v128_clamped(out, c_r2, j, n, c20)
      store_v128_clamped(out, c_r2, j + 4, n, c21)
      store_v128_clamped(out, c_r3, j, n, c30)
      store_v128_clamped(out, c_r3, j + 4, n, c31)
      i = i + 4
    }

    // --- 尾部单行路径（典型如 Batch=1 全连接或 LLM Decode）---
    while i < m {
      let a_r = a_offset + i * k
      let c_r = out_offset + i * n
      let mut c0 = zero_vec
      let mut c1 = zero_vec
      let mut b_ptr = panel_base
      let mut p = 0
      while p < k2 {
        let av0 = @simd.f32x4_splat(a.unsafe_get(a_r + p))
        c0 = @simd.f32x4_relaxed_madd(av0, @simd.v128_load(b_packed, b_ptr), c0)
        c1 = @simd.f32x4_relaxed_madd(
          av0,
          @simd.v128_load(b_packed, b_ptr + 16),
          c1,
        )
        let av1 = @simd.f32x4_splat(a.unsafe_get(a_r + p + 1))
        c0 = @simd.f32x4_relaxed_madd(
          av1,
          @simd.v128_load(b_packed, b_ptr + 32),
          c0,
        )
        c1 = @simd.f32x4_relaxed_madd(
          av1,
          @simd.v128_load(b_packed, b_ptr + 48),
          c1,
        )
        b_ptr = b_ptr + 64
        p = p + 2
      }
      if p < k {
        let av = @simd.f32x4_splat(a.unsafe_get(a_r + p))
        c0 = @simd.f32x4_relaxed_madd(av, @simd.v128_load(b_packed, b_ptr), c0)
        c1 = @simd.f32x4_relaxed_madd(
          av,
          @simd.v128_load(b_packed, b_ptr + 16),
          c1,
        )
      }
      store_v128_clamped(out, c_r, j, n, c0)
      store_v128_clamped(out, c_r, j + 4, n, c1)
      i = i + 1
    }

    j = j + 8
    panel_base = panel_base + panel_bytes
  }
}

///|
/// 面向预打包 2D/1D 卷积的专用 `6×8` K-unroll-2 V128 SIMD GEMM 内核：
/// 计算 `out_nchw[N, M] = epilogue((A[M, K] @ packed_b[K, N])^T + bias[N])`，其中：
/// - `A[M, K]` 为 `im2col_2d_nhwc_f32` 生成的转置感受野矩阵（`M = out_spatial`）；
/// - `packed_b[K, N]` 为模型加载期一次性预打包的 BLIS 8 列连续微面板卷积权重 `W_g^T[K, C_out_g]`；
/// - 可选 `bias` 直接融合为 V128 累加器初值，`epilogue`（`Relu` / `Clip` / `Silu`）在写回前原地执行于 SIMD 寄存器中；
/// - 入口一次性边界校验 + 内部零边界检查 `unsafe_get` / `unsafe_set`，显著提升 YOLOv8 / MobileNetV2 / Whisper Wasm 吞吐。
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
  check_slice_bounds(
    out.length(),
    out_offset,
    m * n,
    "matmul_2d_prepacked_nchw(out)",
  )
  match bias {
    Some(b) =>
      check_slice_bounds(
        b.length(),
        bias_offset,
        n,
        "matmul_2d_prepacked_nchw(bias)",
      )
    None => ()
  }
  if k == 0 {
    let zero : Float = 0.0
    for ch = 0; ch < n; ch = ch + 1 {
      let raw = match bias {
        Some(b) => b.unsafe_get(bias_offset + ch)
        None => zero
      }
      let bv = apply_epilogue_scalar(raw, epilogue)
      let ch_base = out_offset + ch * m
      for i = 0; i < m; i = i + 1 {
        out.unsafe_set(ch_base + i, bv)
      }
    }
    return
  }
  check_slice_bounds(a.length(), a_offset, m * k, "matmul_2d_prepacked_nchw(a)")

  let b_packed = packed_b.data
  let panel_bytes = k * 32
  let k2 = k - k % 2
  let has_ep = match epilogue {
    NoEpilogue => false
    _ => true
  }

  let mut j = 0
  let mut panel_base = 0
  while j < n_pad {
    let init0 = load_bias_v128(bias, bias_offset, j, n)
    let init1 = load_bias_v128(bias, bias_offset, j + 4, n)

    // --- 主循环：6 个空间像素 × 8 个输出通道，K 维 2× 展开 ---
    let mut i = 0
    while i + 6 <= m {
      let a_r0 = a_offset + i * k
      let a_r1 = a_r0 + k
      let a_r2 = a_r1 + k
      let a_r3 = a_r2 + k
      let a_r4 = a_r3 + k
      let a_r5 = a_r4 + k

      let mut c00 = init0
      let mut c01 = init1
      let mut c10 = init0
      let mut c11 = init1
      let mut c20 = init0
      let mut c21 = init1
      let mut c30 = init0
      let mut c31 = init1
      let mut c40 = init0
      let mut c41 = init1
      let mut c50 = init0
      let mut c51 = init1

      let mut b_ptr = panel_base
      let mut p = 0
      while p < k2 {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let mut av = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        c00 = @simd.f32x4_relaxed_madd(av, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av, bv1, c01)
        av = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        c10 = @simd.f32x4_relaxed_madd(av, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av, bv1, c11)
        av = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        c20 = @simd.f32x4_relaxed_madd(av, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av, bv1, c21)
        av = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c30 = @simd.f32x4_relaxed_madd(av, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av, bv1, c31)
        av = @simd.f32x4_splat(a.unsafe_get(a_r4 + p))
        c40 = @simd.f32x4_relaxed_madd(av, bv0, c40)
        c41 = @simd.f32x4_relaxed_madd(av, bv1, c41)
        av = @simd.f32x4_splat(a.unsafe_get(a_r5 + p))
        c50 = @simd.f32x4_relaxed_madd(av, bv0, c50)
        c51 = @simd.f32x4_relaxed_madd(av, bv1, c51)

        let bv2 = @simd.v128_load(b_packed, b_ptr + 32)
        let bv3 = @simd.v128_load(b_packed, b_ptr + 48)
        let p1 = p + 1
        av = @simd.f32x4_splat(a.unsafe_get(a_r0 + p1))
        c00 = @simd.f32x4_relaxed_madd(av, bv2, c00)
        c01 = @simd.f32x4_relaxed_madd(av, bv3, c01)
        av = @simd.f32x4_splat(a.unsafe_get(a_r1 + p1))
        c10 = @simd.f32x4_relaxed_madd(av, bv2, c10)
        c11 = @simd.f32x4_relaxed_madd(av, bv3, c11)
        av = @simd.f32x4_splat(a.unsafe_get(a_r2 + p1))
        c20 = @simd.f32x4_relaxed_madd(av, bv2, c20)
        c21 = @simd.f32x4_relaxed_madd(av, bv3, c21)
        av = @simd.f32x4_splat(a.unsafe_get(a_r3 + p1))
        c30 = @simd.f32x4_relaxed_madd(av, bv2, c30)
        c31 = @simd.f32x4_relaxed_madd(av, bv3, c31)
        av = @simd.f32x4_splat(a.unsafe_get(a_r4 + p1))
        c40 = @simd.f32x4_relaxed_madd(av, bv2, c40)
        c41 = @simd.f32x4_relaxed_madd(av, bv3, c41)
        av = @simd.f32x4_splat(a.unsafe_get(a_r5 + p1))
        c50 = @simd.f32x4_relaxed_madd(av, bv2, c50)
        c51 = @simd.f32x4_relaxed_madd(av, bv3, c51)

        b_ptr = b_ptr + 64
        p = p + 2
      }
      if p < k {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let mut av = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        c00 = @simd.f32x4_relaxed_madd(av, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av, bv1, c01)
        av = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        c10 = @simd.f32x4_relaxed_madd(av, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av, bv1, c11)
        av = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        c20 = @simd.f32x4_relaxed_madd(av, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av, bv1, c21)
        av = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c30 = @simd.f32x4_relaxed_madd(av, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av, bv1, c31)
        av = @simd.f32x4_splat(a.unsafe_get(a_r4 + p))
        c40 = @simd.f32x4_relaxed_madd(av, bv0, c40)
        c41 = @simd.f32x4_relaxed_madd(av, bv1, c41)
        av = @simd.f32x4_splat(a.unsafe_get(a_r5 + p))
        c50 = @simd.f32x4_relaxed_madd(av, bv0, c50)
        c51 = @simd.f32x4_relaxed_madd(av, bv1, c51)
      }

      if has_ep {
        c00 = apply_epilogue_v128(c00, epilogue)
        c01 = apply_epilogue_v128(c01, epilogue)
        c10 = apply_epilogue_v128(c10, epilogue)
        c11 = apply_epilogue_v128(c11, epilogue)
        c20 = apply_epilogue_v128(c20, epilogue)
        c21 = apply_epilogue_v128(c21, epilogue)
        c30 = apply_epilogue_v128(c30, epilogue)
        c31 = apply_epilogue_v128(c31, epilogue)
        c40 = apply_epilogue_v128(c40, epilogue)
        c41 = apply_epilogue_v128(c41, epilogue)
        c50 = apply_epilogue_v128(c50, epilogue)
        c51 = apply_epilogue_v128(c51, epilogue)
      }

      store_v128_6rows_nchw(
        out, out_offset, m, i, j, n, c00, c10, c20, c30, c40, c50,
      )
      store_v128_6rows_nchw(
        out,
        out_offset,
        m,
        i,
        j + 4,
        n,
        c01,
        c11,
        c21,
        c31,
        c41,
        c51,
      )
      i = i + 6
    }

    // --- 4 行尾部路径 ---
    if i + 4 <= m {
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
      let mut b_ptr = panel_base
      for p = 0; p < k; p = p + 1 {
        let bv0 = @simd.v128_load(b_packed, b_ptr)
        let bv1 = @simd.v128_load(b_packed, b_ptr + 16)
        let av0 = @simd.f32x4_splat(a.unsafe_get(a_r0 + p))
        let av1 = @simd.f32x4_splat(a.unsafe_get(a_r1 + p))
        let av2 = @simd.f32x4_splat(a.unsafe_get(a_r2 + p))
        let av3 = @simd.f32x4_splat(a.unsafe_get(a_r3 + p))
        c00 = @simd.f32x4_relaxed_madd(av0, bv0, c00)
        c01 = @simd.f32x4_relaxed_madd(av0, bv1, c01)
        c10 = @simd.f32x4_relaxed_madd(av1, bv0, c10)
        c11 = @simd.f32x4_relaxed_madd(av1, bv1, c11)
        c20 = @simd.f32x4_relaxed_madd(av2, bv0, c20)
        c21 = @simd.f32x4_relaxed_madd(av2, bv1, c21)
        c30 = @simd.f32x4_relaxed_madd(av3, bv0, c30)
        c31 = @simd.f32x4_relaxed_madd(av3, bv1, c31)
        b_ptr = b_ptr + 32
      }
      if has_ep {
        c00 = apply_epilogue_v128(c00, epilogue)
        c01 = apply_epilogue_v128(c01, epilogue)
        c10 = apply_epilogue_v128(c10, epilogue)
        c11 = apply_epilogue_v128(c11, epilogue)
        c20 = apply_epilogue_v128(c20, epilogue)
        c21 = apply_epilogue_v128(c21, epilogue)
        c30 = apply_epilogue_v128(c30, epilogue)
        c31 = apply_epilogue_v128(c31, epilogue)
      }
      store_v128_4rows_nchw(out, out_offset, m, i, j, n, c00, c10, c20, c30)
      store_v128_4rows_nchw(out, out_offset, m, i, j + 4, n, c01, c11, c21, c31)
      i = i + 4
    }

    // --- 尾部单行路径 ---
    while i < m {
      let a_r = a_offset + i * k
      let mut c0 = init0
      let mut c1 = init1
      let mut b_ptr = panel_base
      for p = 0; p < k; p = p + 1 {
        let av = @simd.f32x4_splat(a.unsafe_get(a_r + p))
        c0 = @simd.f32x4_relaxed_madd(av, @simd.v128_load(b_packed, b_ptr), c0)
        c1 = @simd.f32x4_relaxed_madd(
          av,
          @simd.v128_load(b_packed, b_ptr + 16),
          c1,
        )
        b_ptr = b_ptr + 32
      }
      if has_ep {
        c0 = apply_epilogue_v128(c0, epilogue)
        c1 = apply_epilogue_v128(c1, epilogue)
      }
      store_v128_1row_nchw(out, out_offset, m, i, j, n, c0)
      store_v128_1row_nchw(out, out_offset, m, i, j + 4, n, c1)
      i = i + 1
    }

    j = j + 8
    panel_base = panel_base + panel_bytes
  }
}

///|
/// 即时打包版 `V128` SIMD 矩阵乘法：`out[M, N] = a[M, K] @ b[K, N]`。
/// 适用于动态矩阵 B（如 Attention `Q@K^T` 与 `P@V`）。
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

///|
/// V128 SIMD 零拷贝转置右矩阵乘法：`out[M, N] = A[M, K] @ B^T`（其中 `B` 直接以行主序 `[N, K]` 传入，无需百兆级转置拷贝）。
/// 针对 LLM 共享词表投影 `lm_head`（如 `[1, 576] @ [49152, 576]^T`）采用 4 列寄存器展开复用 `A` 向量，
/// 配合入口一次性边界校验与零边界检查访存，消除 100% 权重拷贝开销并实现 4× 浮点向量化吞吐。
pub fn matmul_2d_nt_simd_f32(
  a : FixedArray[Float],
  a_offset : Int32,
  b_nk : FixedArray[Float],
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
  check_slice_bounds(out.length(), out_offset, m * n, "matmul_2d_nt_simd(out)")
  if k == 0 {
    let zero : Float = 0.0
    for i = 0; i < m * n; i = i + 1 {
      out.unsafe_set(out_offset + i, zero)
    }
    return
  }
  check_slice_bounds(a.length(), a_offset, m * k, "matmul_2d_nt_simd(a)")
  check_slice_bounds(b_nk.length(), b_offset, n * k, "matmul_2d_nt_simd(b_nk)")

  let k4 = k - k % 4
  let v_zero = @simd.f32x4_splat((0.0 : Float))
  for i = 0; i < m; i = i + 1 {
    let a_row = a_offset + i * k
    let o_row = out_offset + i * n
    let mut j = 0
    while j + 4 <= n {
      let b0 = b_offset + j * k
      let b1 = b0 + k
      let b2 = b1 + k
      let b3 = b2 + k
      let mut acc0 = v_zero
      let mut acc1 = v_zero
      let mut acc2 = v_zero
      let mut acc3 = v_zero
      let mut kk = 0
      while kk < k4 {
        let va = @simd.f32x4_const(
          a.unsafe_get(a_row + kk),
          a.unsafe_get(a_row + kk + 1),
          a.unsafe_get(a_row + kk + 2),
          a.unsafe_get(a_row + kk + 3),
        )
        let vb0 = @simd.f32x4_const(
          b_nk.unsafe_get(b0 + kk),
          b_nk.unsafe_get(b0 + kk + 1),
          b_nk.unsafe_get(b0 + kk + 2),
          b_nk.unsafe_get(b0 + kk + 3),
        )
        let vb1 = @simd.f32x4_const(
          b_nk.unsafe_get(b1 + kk),
          b_nk.unsafe_get(b1 + kk + 1),
          b_nk.unsafe_get(b1 + kk + 2),
          b_nk.unsafe_get(b1 + kk + 3),
        )
        let vb2 = @simd.f32x4_const(
          b_nk.unsafe_get(b2 + kk),
          b_nk.unsafe_get(b2 + kk + 1),
          b_nk.unsafe_get(b2 + kk + 2),
          b_nk.unsafe_get(b2 + kk + 3),
        )
        let vb3 = @simd.f32x4_const(
          b_nk.unsafe_get(b3 + kk),
          b_nk.unsafe_get(b3 + kk + 1),
          b_nk.unsafe_get(b3 + kk + 2),
          b_nk.unsafe_get(b3 + kk + 3),
        )
        acc0 = @simd.f32x4_relaxed_madd(va, vb0, acc0)
        acc1 = @simd.f32x4_relaxed_madd(va, vb1, acc1)
        acc2 = @simd.f32x4_relaxed_madd(va, vb2, acc2)
        acc3 = @simd.f32x4_relaxed_madd(va, vb3, acc3)
        kk = kk + 4
      }
      let mut s0 = @simd.f32x4_extract_lane(acc0, 0) +
        @simd.f32x4_extract_lane(acc0, 1) +
        (@simd.f32x4_extract_lane(acc0, 2) + @simd.f32x4_extract_lane(acc0, 3))
      let mut s1 = @simd.f32x4_extract_lane(acc1, 0) +
        @simd.f32x4_extract_lane(acc1, 1) +
        (@simd.f32x4_extract_lane(acc1, 2) + @simd.f32x4_extract_lane(acc1, 3))
      let mut s2 = @simd.f32x4_extract_lane(acc2, 0) +
        @simd.f32x4_extract_lane(acc2, 1) +
        (@simd.f32x4_extract_lane(acc2, 2) + @simd.f32x4_extract_lane(acc2, 3))
      let mut s3 = @simd.f32x4_extract_lane(acc3, 0) +
        @simd.f32x4_extract_lane(acc3, 1) +
        (@simd.f32x4_extract_lane(acc3, 2) + @simd.f32x4_extract_lane(acc3, 3))
      while kk < k {
        let av = a.unsafe_get(a_row + kk)
        s0 = s0 + av * b_nk.unsafe_get(b0 + kk)
        s1 = s1 + av * b_nk.unsafe_get(b1 + kk)
        s2 = s2 + av * b_nk.unsafe_get(b2 + kk)
        s3 = s3 + av * b_nk.unsafe_get(b3 + kk)
        kk = kk + 1
      }
      out.unsafe_set(o_row + j, s0)
      out.unsafe_set(o_row + j + 1, s1)
      out.unsafe_set(o_row + j + 2, s2)
      out.unsafe_set(o_row + j + 3, s3)
      j = j + 4
    }
    while j < n {
      let b0 = b_offset + j * k
      let mut acc0 = v_zero
      let mut kk = 0
      while kk < k4 {
        let va = @simd.f32x4_const(
          a.unsafe_get(a_row + kk),
          a.unsafe_get(a_row + kk + 1),
          a.unsafe_get(a_row + kk + 2),
          a.unsafe_get(a_row + kk + 3),
        )
        let vb0 = @simd.f32x4_const(
          b_nk.unsafe_get(b0 + kk),
          b_nk.unsafe_get(b0 + kk + 1),
          b_nk.unsafe_get(b0 + kk + 2),
          b_nk.unsafe_get(b0 + kk + 3),
        )
        acc0 = @simd.f32x4_relaxed_madd(va, vb0, acc0)
        kk = kk + 4
      }
      let mut s0 = @simd.f32x4_extract_lane(acc0, 0) +
        @simd.f32x4_extract_lane(acc0, 1) +
        (@simd.f32x4_extract_lane(acc0, 2) + @simd.f32x4_extract_lane(acc0, 3))
      while kk < k {
        s0 = s0 + a.unsafe_get(a_row + kk) * b_nk.unsafe_get(b0 + kk)
        kk = kk + 1
      }
      out.unsafe_set(o_row + j, s0)
      j = j + 1
    }
  }
}
