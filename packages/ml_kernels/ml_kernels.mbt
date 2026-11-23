///|
/// `matrixlt/ml_kernels` 根包门面：导出常用内核便捷入口。
/// 如需细粒度控制，也可直接导入 `matrixlt/ml_kernels/gemm`、`vecmath` 或 `conv`。
pub type PackedMatrixB = @gemm.PackedMatrixB

///|
/// 标量缓存友好 2D 矩阵乘法便捷入口。
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
  @gemm.matmul_2d_scalar_f32(a, a_offset, b, b_offset, out, out_offset, m, k, n)
}

///|
/// V128 SIMD 分块 2D 矩阵乘法便捷入口。
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
  @gemm.matmul_2d_simd_f32(a, a_offset, b, b_offset, out, out_offset, m, k, n)
}

///|
/// 将静态权重矩阵 B 预打包为 16 字节对齐的 `PackedMatrixB`。
pub fn pack_matrix_b_f32(
  b : FixedArray[Float],
  b_offset : Int32,
  k : Int32,
  n : Int32,
) -> PackedMatrixB {
  @gemm.pack_matrix_b_f32(b, b_offset, k, n)
}

///|
/// 将转置存储的静态权重矩阵 `B^T[N, K]` 一步打包为 `PackedMatrixB(K, N)`。
pub fn pack_matrix_b_transposed_f32(
  b_t : FixedArray[Float],
  b_offset : Int32,
  n : Int32,
  k : Int32,
) -> PackedMatrixB {
  @gemm.pack_matrix_b_transposed_f32(b_t, b_offset, n, k)
}

///|
/// 使用预打包权重矩阵执行 V128 SIMD GEMM。
pub fn matmul_2d_prepacked_simd_f32(
  a : FixedArray[Float],
  a_offset : Int32,
  packed_b : PackedMatrixB,
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
) -> Unit {
  @gemm.matmul_2d_prepacked_simd_f32(a, a_offset, packed_b, out, out_offset, m)
}

///|
/// 使用预打包转置权重执行 NCHW 输出 + 融合偏置的 V128 SIMD GEMM（供 2D 卷积热路径使用）。
pub fn matmul_2d_prepacked_nchw_simd_f32(
  a : FixedArray[Float],
  a_offset : Int32,
  packed_b : PackedMatrixB,
  bias : FixedArray[Float]?,
  bias_offset : Int32,
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
) -> Unit {
  @gemm.matmul_2d_prepacked_nchw_simd_f32(
    a, a_offset, packed_b, bias, bias_offset, out, out_offset, m,
  )
}

///|
/// 将 Float32 缓冲打包为小端序字节缓冲（供零转换 V128 运算使用）。
pub fn pack_f32_to_bytes(src : FixedArray[Float]) -> FixedArray[Byte] {
  @vecmath.pack_f32_to_bytes(src)
}

///|
/// 2D 卷积 `im2col` 展开便捷入口。
pub fn im2col_2d_f32(
  input : FixedArray[Float],
  in_base : Int32,
  channels : Int32,
  in_h : Int32,
  in_w : Int32,
  kh : Int32,
  kw : Int32,
  stride_h : Int32,
  stride_w : Int32,
  pad_top : Int32,
  pad_left : Int32,
  dil_h : Int32,
  dil_w : Int32,
  out_h : Int32,
  out_w : Int32,
  col : FixedArray[Float],
) -> Unit {
  @conv.im2col_2d_f32(
    input, in_base, channels, in_h, in_w, kh, kw, stride_h, stride_w, pad_top, pad_left,
    dil_h, dil_w, out_h, out_w, col,
  )
}

///|
/// 2D 卷积转置 `im2col` 展开便捷入口（输出 `[out_h * out_w, channels * kh * kw]`）。
pub fn im2col_2d_nhwc_f32(
  input : FixedArray[Float],
  in_base : Int32,
  channels : Int32,
  in_h : Int32,
  in_w : Int32,
  kh : Int32,
  kw : Int32,
  stride_h : Int32,
  stride_w : Int32,
  pad_top : Int32,
  pad_left : Int32,
  dil_h : Int32,
  dil_w : Int32,
  out_h : Int32,
  out_w : Int32,
  col_t : FixedArray[Float],
) -> Unit {
  @conv.im2col_2d_nhwc_f32(
    input, in_base, channels, in_h, in_w, kh, kw, stride_h, stride_w, pad_top, pad_left,
    dil_h, dil_w, out_h, out_w, col_t,
  )
}
