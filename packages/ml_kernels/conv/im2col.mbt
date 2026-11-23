///|
/// 将单张输入特征图的单个分组 `[C_in_g, H, W]` 按 `im2col` 展开为 2D 矩阵
/// `[C_in_g * kH * kW, out_h * out_w]`。
/// 展开后 2D 卷积等价于一次矩阵乘法 `W_group[C_out_g, K] @ col[K, out_h * out_w]`，
/// 可直接复用 `matrixlt/ml_kernels/gemm` 中的标量或 V128 SIMD 分块内核。
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
  let out_spatial = out_h * out_w
  let in_hw = in_h * in_w
  let zero : Float = 0.0
  let mut col_row = 0

  for c = 0; c < channels; c = c + 1 {
    let ch_base = in_base + c * in_hw
    for ky = 0; ky < kh; ky = ky + 1 {
      for kx = 0; kx < kw; kx = kx + 1 {
        let row_off = col_row * out_spatial
        let mut out_idx = 0
        for oy = 0; oy < out_h; oy = oy + 1 {
          let iy = oy * stride_h - pad_top + ky * dil_h
          if iy < 0 || iy >= in_h {
            // 整行处于上下 padding 区域，直接批量填 0
            for _ox = 0; _ox < out_w; _ox = _ox + 1 {
              col[row_off + out_idx] = zero
              out_idx = out_idx + 1
            }
          } else {
            let row_in_base = ch_base + iy * in_w
            for ox = 0; ox < out_w; ox = ox + 1 {
              let ix = ox * stride_w - pad_left + kx * dil_w
              col[row_off + out_idx] = if ix >= 0 && ix < in_w {
                input[row_in_base + ix]
              } else {
                zero
              }
              out_idx = out_idx + 1
            }
          }
        }
        col_row = col_row + 1
      }
    }
  }
}
