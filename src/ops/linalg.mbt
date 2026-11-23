///|
/// 标量核心矩阵乘法内核：计算 `out[M, N] = a[M, K] * b[K, N]`（行主序连续缓冲区）。
/// 采用 `i-k-j` 循环顺序，使内层循环对 `b` 和 `out` 均为顺序连续访存，
/// 相比朴素 `i-j-k` 顺序可显著提高 CPU L1 Cache 命中率。
pub fn matmul_2d_raw_f32(
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
  // 先清零输出块，保证多次调用同一缓冲区时结果干净
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
/// ONNX Gemm 算子：计算 `Y = alpha * (A' @ B') + beta * C`。
/// 其中 A' 为 A（或 trans_a=true 时为 A^T），B' 为 B（或 trans_b=true 时为 B^T），
/// 可选偏置张量 C 支持单向广播到输出形状 `[M, N]`。
pub fn gemm_f32(
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
  let out = FixedArray::make(m * n, (0.0 : Float))
  matmul_2d_raw_f32(a_buf, 0, b_buf, 0, out, 0, m, k, n)

  // 应用 alpha 缩放因子
  let one : Float = 1.0
  if alpha != one {
    for i = 0; i < m * n; i = i + 1 {
      out[i] = out[i] * alpha
    }
  }

  let prod = @tensor.Tensor::from_f32([m, n], out)

  // 若提供可选偏置 C 且 beta != 0，按广播语义叠加 beta * C
  let zero : Float = 0.0
  match c {
    None => prod
    Some(bias) =>
      if beta == zero {
        prod
      } else if beta == one {
        add_f32(prod, bias)
      } else {
        let b_buf = bias.to_f32_buffer()
        let scaled = FixedArray::makei(b_buf.length(), fn(i) { b_buf[i] * beta })
        let scaled_bias = @tensor.Tensor::from_f32(bias.shape(), scaled)
        add_f32(prod, scaled_bias)
      }
  }
}

///|
/// ONNX MatMul 算子（标量参考实现），兼容 NumPy 规则：
/// - 2D x 2D：`[M, K] @ [K, N] -> [M, N]`
/// - 1D 向量自动升维/降维（1D x 1D 得标量，1D x ND / ND x 1D 消去缩并维）
/// - 高维批量矩阵乘法：前导 batch 维度遵循多向广播规则。
pub fn matmul_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  let ra = a.rank()
  let rb = b.rank()
  if ra == 0 || rb == 0 {
    raise @tensor.ShapeMismatch("matmul_f32: inputs must have rank >= 1")
  }

  // 最常见快速路径：标准 2D x 2D 矩阵乘法
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
    let out = FixedArray::make(m * n, (0.0 : Float))
    matmul_2d_raw_f32(
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

  // 若输入为 1D，按 NumPy 规范临时补齐到 2D，计算完成后再剥离补齐的维度
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

  // 提取并广播前导 batch 维度
  let batch_a = Array::makei(rank_a - 2, fn(i) { sa[i] })
  let batch_b = Array::makei(rank_b - 2, fn(i) { sb[i] })
  let batch_out = @tensor.broadcast_shape(batch_a, batch_b)
  let batch_total = @tensor.num_elements(batch_out)

  let a_buf = a2.to_f32_buffer()
  let b_buf = b2.to_f32_buffer()
  let out_buf = FixedArray::make(batch_total * m * n, (0.0 : Float))

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
    matmul_2d_raw_f32(
      a_buf,
      off_a * mat_size_a,
      b_buf,
      off_b * mat_size_b,
      out_buf,
      bi * mat_size_out,
      m,
      k,
      n,
    )
  }

  // 组装最终输出形状（若原输入为 1D，剥离临时添加的维度）
  let final_shape : Array[Int32] = batch_out.copy()
  if !squeeze_a {
    final_shape.push(m)
  }
  if !squeeze_b {
    final_shape.push(n)
  }
  @tensor.Tensor::from_f32(final_shape, out_buf)
}
