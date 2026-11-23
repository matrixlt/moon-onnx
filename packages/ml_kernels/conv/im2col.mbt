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

///|
/// 将单个分组的 NCHW 输入特征图 `[channels, in_h, in_w]` 按转置 `im2col` 布局展开为
/// 2D 矩阵 `col_t[out_h * out_w, channels * kh * kw]`：
/// - 每一行对应一个输出空间像素 `(oy, ox)` 的完整感受野（按 `(c, ky, kx)` 顺序排列）；
/// - 配合预打包转置权重 `W_g^T[K, C_out_g]` 可直接执行零运行期打包的 `col_t @ W_g^T` SIMD GEMM；
/// - 内置 `1×1` 点卷积快速路径与内部无越界像素矩形（零边界检查分支）优化。
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
  let in_hw = in_h * in_w
  let out_spatial = out_h * out_w
  let k_cols = channels * kh * kw

  // 快速路径 1：1x1 点卷积（stride=1, pad=0），直接执行 NCHW -> NHWC 紧凑转置
  if kh == 1 &&
    kw == 1 &&
    stride_h == 1 &&
    stride_w == 1 &&
    pad_top == 0 &&
    pad_left == 0 &&
    out_h == in_h &&
    out_w == in_w {
    for s = 0; s < out_spatial; s = s + 1 {
      let dst_row = s * channels
      for c = 0; c < channels; c = c + 1 {
        col_t[dst_row + c] = input[in_base + c * in_hw + s]
      }
    }
    return
  }

  // 计算无需边界判断的内部有效输出像素矩形 [oy_min, oy_max) x [ox_min, ox_max)
  let raw_oy_min = if pad_top <= 0 {
    0
  } else {
    (pad_top + stride_h - 1) / stride_h
  }
  let oy_min = if raw_oy_min > out_h { out_h } else { raw_oy_min }
  let max_top_y = in_h - 1 - (kh - 1) * dil_h + pad_top
  let raw_oy_max = if max_top_y < 0 { 0 } else { max_top_y / stride_h + 1 }
  let oy_max = if raw_oy_max > out_h { out_h } else { raw_oy_max }

  let raw_ox_min = if pad_left <= 0 {
    0
  } else {
    (pad_left + stride_w - 1) / stride_w
  }
  let ox_min = if raw_ox_min > out_w { out_w } else { raw_ox_min }
  let max_left_x = in_w - 1 - (kw - 1) * dil_w + pad_left
  let raw_ox_max = if max_left_x < 0 { 0 } else { max_left_x / stride_w + 1 }
  let ox_max = if raw_ox_max > out_w { out_w } else { raw_ox_max }

  let zero : Float = 0.0
  let row_step_y = dil_h * in_w

  for oy = 0; oy < out_h; oy = oy + 1 {
    let iy0 = oy * stride_h - pad_top
    let y_interior = oy >= oy_min && oy < oy_max
    let oy_out_base = oy * out_w

    for ox = 0; ox < out_w; ox = ox + 1 {
      let ix0 = ox * stride_w - pad_left
      let mut dst = (oy_out_base + ox) * k_cols

      if y_interior && ox >= ox_min && ox < ox_max {
        // 内部像素快速路径：整块感受野严格位于输入范围内，零越界分支
        if dil_w == 1 {
          for c = 0; c < channels; c = c + 1 {
            let mut src_row = in_base + c * in_hw + iy0 * in_w + ix0
            for _ky = 0; _ky < kh; _ky = _ky + 1 {
              for kx = 0; kx < kw; kx = kx + 1 {
                col_t[dst + kx] = input[src_row + kx]
              }
              dst = dst + kw
              src_row = src_row + row_step_y
            }
          }
        } else {
          for c = 0; c < channels; c = c + 1 {
            let mut src_row = in_base + c * in_hw + iy0 * in_w + ix0
            for _ky = 0; _ky < kh; _ky = _ky + 1 {
              for kx = 0; kx < kw; kx = kx + 1 {
                col_t[dst + kx] = input[src_row + kx * dil_w]
              }
              dst = dst + kw
              src_row = src_row + row_step_y
            }
          }
        }
      } else {
        // 边界像素路径：按行提升 iy 越界判断，越界整行直接补 0
        for c = 0; c < channels; c = c + 1 {
          let ch_base = in_base + c * in_hw
          for ky = 0; ky < kh; ky = ky + 1 {
            let iy = iy0 + ky * dil_h
            if iy < 0 || iy >= in_h {
              for kx = 0; kx < kw; kx = kx + 1 {
                col_t[dst + kx] = zero
              }
            } else {
              let row_in_base = ch_base + iy * in_w
              for kx = 0; kx < kw; kx = kx + 1 {
                let ix = ix0 + kx * dil_w
                col_t[dst + kx] = if ix >= 0 && ix < in_w {
                  input[row_in_base + ix]
                } else {
                  zero
                }
              }
            }
            dst = dst + kw
          }
        }
      }
    }
  }
}
