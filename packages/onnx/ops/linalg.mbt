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
/// ONNX Gemm 算子：`Y = alpha * (A' @ B') + beta * C`（复用 `matrixlt/ml_kernels/gemm`）。
pub fn gemm_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  c : @tensor.Tensor?,
  alpha~ : Float = (1.0 : Float),
  beta~ : Float = (1.0 : Float),
  trans_a~ : Bool = false,
  trans_b~ : Bool = false,
  use_simd~ : Bool = false,
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

  let out = FixedArray::make(m * n, (0.0 : Float))
  run_gemm_2d(
    a_mat.to_f32_buffer(),
    0,
    b_mat.to_f32_buffer(),
    0,
    out,
    0,
    m,
    k,
    n,
    use_simd,
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
        add_f32(prod, bias, use_simd~)
      } else {
        let b_buf = bias.to_f32_buffer()
        let scaled = FixedArray::makei(b_buf.length(), fn(i) { b_buf[i] * beta })
        add_f32(prod, @tensor.Tensor::from_f32(bias.shape(), scaled), use_simd~)
      }
  }
}

///|
/// ONNX MatMul 算子（支持 2D、1D 点积及高维批量广播，底层复用 `matrixlt/ml_kernels/gemm`）。
pub fn matmul_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  let ra = a.rank()
  let rb = b.rank()
  if ra == 0 || rb == 0 {
    raise @tensor.ShapeMismatch("matmul_f32: inputs must have rank >= 1")
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
    let out = FixedArray::make(m * n, (0.0 : Float))
    run_gemm_2d(
      a.to_f32_buffer(),
      0,
      b.to_f32_buffer(),
      0,
      out,
      0,
      m,
      k,
      n,
      use_simd,
    )
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
/// 使用预打包权重矩阵 `PackedMatrixB` 执行 V128 SIMD `Gemm`（运行期零权重打包开销）。
/// 要求 `packed_b` 已经对应未转置的 `[K, N]` 静态权重矩阵。
pub fn gemm_prepacked_f32(
  a : @tensor.Tensor,
  packed_b : @gemm.PackedMatrixB,
  c : @tensor.Tensor?,
  alpha~ : Float = (1.0 : Float),
  beta~ : Float = (1.0 : Float),
  trans_a~ : Bool = false,
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

  let out = FixedArray::make(m * n, (0.0 : Float))
  @gemm.matmul_2d_prepacked_simd_f32(
    a_mat.to_f32_buffer(),
    0,
    packed_b,
    out,
    0,
    m,
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
        add_f32(prod, bias, use_simd=true)
      } else {
        let b_buf = bias.to_f32_buffer()
        let scaled = FixedArray::makei(b_buf.length(), fn(i) { b_buf[i] * beta })
        add_f32(
          prod,
          @tensor.Tensor::from_f32(bias.shape(), scaled),
          use_simd=true,
        )
      }
  }
}

///|
/// 使用预打包权重矩阵 `PackedMatrixB` 执行 2D V128 SIMD `MatMul`（运行期零权重打包开销）。
pub fn matmul_prepacked_f32(
  a : @tensor.Tensor,
  packed_b : @gemm.PackedMatrixB,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.rank() != 2 {
    raise @tensor.ShapeMismatch(
      "matmul_prepacked_f32: A must be 2D matrix, got rank \{a.rank()}",
    )
  }
  let sa = a.shape()
  let m = sa[0]
  let k = sa[1]
  let (kb, n) = packed_b.dims()
  if k != kb {
    raise @tensor.ShapeMismatch(
      "matmul_prepacked_f32: inner dimension mismatch (\{k} vs \{kb})",
    )
  }
  let out = FixedArray::make(m * n, (0.0 : Float))
  @gemm.matmul_2d_prepacked_simd_f32(a.to_f32_buffer(), 0, packed_b, out, 0, m)
  @tensor.Tensor::from_f32([m, n], out)
}
