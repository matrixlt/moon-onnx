///|
/// 根据后端开关选择 `matrixlt/ml_kernels/gemm` 中的标量或 V128 SIMD 矩阵乘法内核。
fn run_gemm_2d(
  a : FixedArray[Float],
  a_offset : Int32,
  b : FixedArray[Float],
  b_offset : Int32,
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  k : Int32,
  n : Int32,
  use_simd : Bool,
) -> Unit {
  if use_simd {
    @gemm.matmul_2d_simd_f32(a, a_offset, b, b_offset, out, out_offset, m, k, n)
  } else {
    @gemm.matmul_2d_scalar_f32(
      a, a_offset, b, b_offset, out, out_offset, m, k, n,
    )
  }
}

///|
/// 将 ONNX Gemm 的单向广播偏置 `beta * C` 原地累加至 `out[m, n]`，避免额外分配临时输出张量。
fn add_gemm_bias_inplace(
  out : FixedArray[Float],
  m : Int32,
  n : Int32,
  bias : @tensor.Tensor,
  beta : Float,
) -> Unit raise @tensor.TensorError {
  let zero : Float = 0.0
  if beta == zero {
    return
  }
  let cs = bias.shape()
  let b_buf = bias.to_f32_buffer()
  let one : Float = 1.0
  let total = m * n

  if cs.length() == 0 ||
    (cs.length() == 1 && cs[0] == 1) ||
    (cs.length() == 2 && cs[0] == 1 && cs[1] == 1) {
    let bv = b_buf[0] * beta
    for i = 0; i < total; i = i + 1 {
      out[i] = out[i] + bv
    }
    return
  }
  if (cs.length() == 1 && cs[0] == n) ||
    (cs.length() == 2 && cs[0] == 1 && cs[1] == n) {
    if beta == one {
      for r = 0; r < m; r = r + 1 {
        let off = r * n
        for c = 0; c < n; c = c + 1 {
          out[off + c] = out[off + c] + b_buf[c]
        }
      }
    } else {
      for r = 0; r < m; r = r + 1 {
        let off = r * n
        for c = 0; c < n; c = c + 1 {
          out[off + c] = out[off + c] + b_buf[c] * beta
        }
      }
    }
    return
  }
  if cs.length() == 2 && cs[0] == m && cs[1] == 1 {
    for r = 0; r < m; r = r + 1 {
      let bv = b_buf[r] * beta
      let off = r * n
      for c = 0; c < n; c = c + 1 {
        out[off + c] = out[off + c] + bv
      }
    }
    return
  }
  if cs.length() == 2 && cs[0] == m && cs[1] == n {
    if beta == one {
      for i = 0; i < total; i = i + 1 {
        out[i] = out[i] + b_buf[i]
      }
    } else {
      for i = 0; i < total; i = i + 1 {
        out[i] = out[i] + b_buf[i] * beta
      }
    }
    return
  }
  raise @tensor.ShapeMismatch(
    "gemm: bias shape \{@tensor.format_shape(cs)} is not unidirectional-broadcastable to [\{m}, \{n}]",
  )
}

///|
/// ONNX Gemm 算子：`Y = alpha * (A' @ B') + beta * C`（复用 `matrixlt/ml_kernels/gemm` 与可选 `BufferPool`）。
pub fn gemm_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  c : @tensor.Tensor?,
  alpha~ : Float = (1.0 : Float),
  beta~ : Float = (1.0 : Float),
  trans_a~ : Bool = false,
  trans_b~ : Bool = false,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.rank() != 2 || b.rank() != 2 {
    raise @tensor.ShapeMismatch(
      "gemm_f32: A and B must be 2D matrices, got ranks \{a.rank()} and \{b.rank()}",
    )
  }
  let a_mat = if trans_a { a.transpose([1, 0]).contiguous() } else { a }
  let b_mat = if trans_b { b.transpose([1, 0]).contiguous() } else { b }

  let a_shape = a_mat.shape()
  let b_shape = b_mat.shape()
  let m = a_shape[0]
  let k = a_shape[1]
  let kb = b_shape[0]
  let n = b_shape[1]
  if k != kb {
    raise @tensor.ShapeMismatch(
      "gemm_f32: inner dimension mismatch (\{k} vs \{kb})",
    )
  }

  let a_buf = a_mat.to_f32_buffer()
  let b_buf = b_mat.to_f32_buffer()
  let out = alloc_f32(pool, m * n)
  run_gemm_2d(a_buf, 0, b_buf, 0, out, 0, m, k, n, use_simd)

  let one : Float = 1.0
  if alpha != one {
    for i = 0; i < m * n; i = i + 1 {
      out[i] = out[i] * alpha
    }
  }

  match c {
    None => ()
    Some(bias) => add_gemm_bias_inplace(out, m, n, bias, beta)
  }
  @tensor.Tensor::from_f32([m, n], out)
}

///|
/// ONNX MatMul 算子（支持 2D、1D 点积及高维批量广播，底层复用 `matrixlt/ml_kernels/gemm` 与可选 `BufferPool`）。
pub fn matmul_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let ra = a.rank()
  let rb = b.rank()
  if ra == 0 || rb == 0 {
    raise @tensor.ShapeMismatch("matmul_f32: inputs must have rank >= 1")
  }

  // 零拷贝快速路径：当 B 为 2D 转置视图（如 LLM 共享词表权重 `Transpose(embed_tokens[N, K]) -> [K, N]`）时，
  // 直接按行连续点积底层 `[N, K]` 缓冲，避免百兆级权重拷贝（SmolLM2 lm_head 节省 113 MB 内存与拷贝耗时）
  if rb == 2 && ra >= 2 && b.offset == 0 {
    let sb = b.shape()
    let b_st = b.strides()
    let k = sb[0]
    let n = sb[1]
    if b_st.length() == 2 && b_st[0] == 1 && b_st[1] == k {
      match b.data {
        @tensor.F32(b_raw) => {
          let sa = a.shape()
          if sa[ra - 1] != k {
            raise @tensor.ShapeMismatch(
              "matmul_f32: inner dimension mismatch (\{sa[ra - 1]} vs \{k})",
            )
          }
          let mut m_tot = 1
          for d = 0; d < ra - 1; d = d + 1 {
            m_tot = m_tot * sa[d]
          }
          let a_buf = a.to_f32_buffer()
          let out_buf = alloc_f32(pool, m_tot * n)
          if use_simd {
            @gemm.matmul_2d_nt_simd_f32(
              a_buf, 0, b_raw, 0, out_buf, 0, m_tot, k, n,
            )
          } else {
            let k4 = k - k % 4
            for i = 0; i < m_tot; i = i + 1 {
              let a_row = i * k
              let o_row = i * n
              for j = 0; j < n; j = j + 1 {
                let b_row = j * k
                let mut s0 : Float = (0.0 : Float)
                let mut s1 : Float = (0.0 : Float)
                let mut s2 : Float = (0.0 : Float)
                let mut s3 : Float = (0.0 : Float)
                let mut kk = 0
                while kk < k4 {
                  s0 = s0 + a_buf[a_row + kk] * b_raw[b_row + kk]
                  s1 = s1 + a_buf[a_row + kk + 1] * b_raw[b_row + kk + 1]
                  s2 = s2 + a_buf[a_row + kk + 2] * b_raw[b_row + kk + 2]
                  s3 = s3 + a_buf[a_row + kk + 3] * b_raw[b_row + kk + 3]
                  kk = kk + 4
                }
                let mut acc = s0 + s1 + (s2 + s3)
                while kk < k {
                  acc = acc + a_buf[a_row + kk] * b_raw[b_row + kk]
                  kk = kk + 1
                }
                out_buf[o_row + j] = acc
              }
            }
          }
          let out_shape = sa.copy()
          out_shape[ra - 1] = n
          return @tensor.Tensor::from_f32(out_shape, out_buf)
        }
        _ => ()
      }
    }
  }

  if ra == 2 && rb == 2 {
    let sa = a.shape()
    let sb = b.shape()
    let m = sa[0]
    let k = sa[1]
    if k != sb[0] {
      raise @tensor.ShapeMismatch(
        "matmul_f32: inner dimension mismatch (\{k} vs \{sb[0]})",
      )
    }
    let n = sb[1]
    let a_buf = a.to_f32_buffer()
    let b_buf = b.to_f32_buffer()
    let out = alloc_f32(pool, m * n)
    run_gemm_2d(a_buf, 0, b_buf, 0, out, 0, m, k, n, use_simd)
    return @tensor.Tensor::from_f32([m, n], out)
  }

  let squeeze_a = ra == 1
  let squeeze_b = rb == 1
  let a2 = if squeeze_a { a.unsqueeze([0]) } else { a }
  let b2 = if squeeze_b { b.unsqueeze([1]) } else { b }

  let sa = a2.shape()
  let sb = b2.shape()
  let rank_a = sa.length()
  let rank_b = sb.length()
  let m = sa[rank_a - 2]
  let k = sa[rank_a - 1]
  let kb = sb[rank_b - 2]
  let n = sb[rank_b - 1]
  if k != kb {
    raise @tensor.ShapeMismatch(
      "matmul_f32: inner dimension mismatch (\{k} vs \{kb})",
    )
  }

  let batch_a = Array::makei(rank_a - 2, fn(i) { sa[i] })
  let batch_b = Array::makei(rank_b - 2, fn(i) { sb[i] })
  let batch_out = @tensor.broadcast_shape(batch_a, batch_b)
  let batch_total = @tensor.num_elements(batch_out)

  let a_buf = a2.to_f32_buffer()
  let b_buf = b2.to_f32_buffer()
  let out_buf = alloc_f32(pool, batch_total * m * n)

  let mat_size_a = m * k
  let mat_size_b = k * n
  let mat_size_out = m * n

  let batch_a_strides = @tensor.broadcast_strides(
    batch_a,
    @tensor.compute_strides(batch_a),
    batch_out,
  )
  let batch_b_strides = @tensor.broadcast_strides(
    batch_b,
    @tensor.compute_strides(batch_b),
    batch_out,
  )
  let batch_rank = batch_out.length()

  for bi = 0; bi < batch_total; bi = bi + 1 {
    let mut rem = bi
    let mut off_a = 0
    let mut off_b = 0
    for ax = batch_rank - 1; ax >= 0; ax = ax - 1 {
      let dim = batch_out[ax]
      let coord = rem % dim
      rem = rem / dim
      off_a = off_a + coord * batch_a_strides[ax]
      off_b = off_b + coord * batch_b_strides[ax]
    }
    run_gemm_2d(
      a_buf,
      off_a * mat_size_a,
      b_buf,
      off_b * mat_size_b,
      out_buf,
      bi * mat_size_out,
      m,
      k,
      n,
      use_simd,
    )
  }

  let final_shape : Array[Int32] = batch_out.copy()
  if !squeeze_a {
    final_shape.push(m)
  }
  if !squeeze_b {
    final_shape.push(n)
  }
  @tensor.Tensor::from_f32(final_shape, out_buf)
}

///|
/// 使用预打包权重矩阵 `PackedMatrixB` 执行 V128 SIMD `Gemm`（运行期零权重打包开销，支持原地偏置融合与 BufferPool）。
/// 要求 `packed_b` 已经对应有效的 `[K, N]` 右矩阵（无论原权重是 `trans_b=false` 还是已转置预打包的 `trans_b=true`）。
pub fn gemm_prepacked_f32(
  a : @tensor.Tensor,
  packed_b : @gemm.PackedMatrixB,
  c : @tensor.Tensor?,
  alpha~ : Float = (1.0 : Float),
  beta~ : Float = (1.0 : Float),
  trans_a~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.rank() != 2 {
    raise @tensor.ShapeMismatch(
      "gemm_prepacked_f32: A must be 2D matrix, got rank \{a.rank()}",
    )
  }
  let a_mat = if trans_a { a.transpose([1, 0]).contiguous() } else { a }
  let sa = a_mat.shape()
  let m = sa[0]
  let k = sa[1]
  let (kb, n) = packed_b.dims()
  if k != kb {
    raise @tensor.ShapeMismatch(
      "gemm_prepacked_f32: inner dimension mismatch (\{k} vs \{kb})",
    )
  }

  let a_buf = a_mat.to_f32_buffer()
  let out = alloc_f32(pool, m * n)
  @gemm.matmul_2d_prepacked_simd_f32(a_buf, 0, packed_b, out, 0, m)

  let one : Float = 1.0
  if alpha != one {
    for i = 0; i < m * n; i = i + 1 {
      out[i] = out[i] * alpha
    }
  }
  match c {
    None => ()
    Some(bias) => add_gemm_bias_inplace(out, m, n, bias, beta)
  }
  @tensor.Tensor::from_f32([m, n], out)
}

///|
/// 使用预打包权重矩阵 `PackedMatrixB` 执行 V128 SIMD `MatMul`（支持 2D 与任意高维 `[..., M, K] @ [K, N]`，零权重打包开销）。
pub fn matmul_prepacked_f32(
  a : @tensor.Tensor,
  packed_b : @gemm.PackedMatrixB,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sa = a.shape()
  let rank = sa.length()
  if rank < 2 {
    raise @tensor.ShapeMismatch(
      "matmul_prepacked_f32: A must have rank >= 2, got rank \{rank}",
    )
  }
  let k = sa[rank - 1]
  let (kb, n) = packed_b.dims()
  if k != kb {
    raise @tensor.ShapeMismatch(
      "matmul_prepacked_f32: inner dimension mismatch (\{k} vs \{kb})",
    )
  }
  let mut m = 1
  for d = 0; d < rank - 1; d = d + 1 {
    m = m * sa[d]
  }
  let out_shape = sa.copy()
  out_shape[rank - 1] = n

  let a_buf = a.to_f32_buffer()
  let out = alloc_f32(pool, m * n)
  @gemm.matmul_2d_prepacked_simd_f32(a_buf, 0, packed_b, out, 0, m)
  @tensor.Tensor::from_f32(out_shape, out)
}
