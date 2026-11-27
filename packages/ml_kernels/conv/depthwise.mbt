///|
/// 专用 NCHW 深度可分离 2D 卷积内核（Depthwise Conv2D，`group == c_in == c_out`）：
/// - 零 `im2col` 临时内存分配，直接在输入特征图上滑动计算；
/// - 入口一次性边界校验 + 热循环零边界检查 `unsafe_get` / `unsafe_set`；
/// - 当 `stride_w == 1 && dil_w == 1` 时，对内部安全输出列采用 4-wide `V128` 向量化乘加（`f32x4_relaxed_madd`）；
/// - 原生支持融合通道偏置与尾部激活（`NoEpilogue` / `Relu` / `Clip` / `Silu`）。
pub fn depthwise_conv2d_f32(
  input : FixedArray[Float],
  in_base : Int32,
  weight : FixedArray[Float],
  bias : FixedArray[Float]?,
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
  out : FixedArray[Float],
  out_base : Int32,
  epilogue~ : @gemm.Epilogue = @gemm.NoEpilogue,
) -> Unit {
  let in_hw = in_h * in_w
  let out_hw = out_h * out_w
  let k_hw = kh * kw
  if channels == 0 || out_hw == 0 {
    return
  }
  check_conv_bounds(
    input.length(),
    in_base,
    channels * in_hw,
    "depthwise_conv2d_f32(input)",
  )
  check_conv_bounds(
    weight.length(),
    0,
    channels * k_hw,
    "depthwise_conv2d_f32(weight)",
  )
  check_conv_bounds(
    out.length(),
    out_base,
    channels * out_hw,
    "depthwise_conv2d_f32(out)",
  )
  match bias {
    Some(b) =>
      check_conv_bounds(b.length(), 0, channels, "depthwise_conv2d_f32(bias)")
    None => ()
  }
  let zero : Float = 0.0

  // 计算水平方向无需左右边界判断的内部输出列区间 [ox_min, ox_max)
  let raw_ox_min = if pad_left <= 0 {
    0
  } else {
    (pad_left + stride_w - 1) / stride_w
  }
  let ox_min = if raw_ox_min > out_w { out_w } else { raw_ox_min }
  let max_left_x = in_w - 1 - (kw - 1) * dil_w + pad_left
  let raw_ox_max = if max_left_x < 0 { 0 } else { max_left_x / stride_w + 1 }
  let ox_max = if raw_ox_max > out_w { out_w } else { raw_ox_max }
  let can_vec_x = stride_w == 1 && dil_w == 1 && ox_max - ox_min >= 4

  for c = 0; c < channels; c = c + 1 {
    let ch_in = in_base + c * in_hw
    let ch_out = out_base + c * out_hw
    let w_ch = c * k_hw
    let b_val = match bias {
      Some(b) => b.unsafe_get(c)
      None => zero
    }
    let b_vec = @simd.f32x4_splat(b_val)

    for oy = 0; oy < out_h; oy = oy + 1 {
      let iy0 = oy * stride_h - pad_top
      let out_row = ch_out + oy * out_w

      if can_vec_x {
        // 左侧边界列（标量带越界检查）
        for ox = 0; ox < ox_min; ox = ox + 1 {
          let ix0 = ox - pad_left
          let mut acc = b_val
          for ky = 0; ky < kh; ky = ky + 1 {
            let iy = iy0 + ky * dil_h
            if iy < 0 || iy >= in_h {
              continue
            }
            let in_row = ch_in + iy * in_w
            let w_row = w_ch + ky * kw
            for kx = 0; kx < kw; kx = kx + 1 {
              let ix = ix0 + kx
              if ix >= 0 && ix < in_w {
                acc = acc +
                  input.unsafe_get(in_row + ix) * weight.unsafe_get(w_row + kx)
              }
            }
          }
          out.unsafe_set(
            out_row + ox,
            @gemm.apply_epilogue_scalar(acc, epilogue),
          )
        }

        // 中部安全列：4 像素并行 V128 向量化乘加，无水平边界分支
        let mut ox = ox_min
        while ox + 4 <= ox_max {
          let ix0 = ox - pad_left
          let mut acc = b_vec
          for ky = 0; ky < kh; ky = ky + 1 {
            let iy = iy0 + ky * dil_h
            if iy < 0 || iy >= in_h {
              continue
            }
            let in_row = ch_in + iy * in_w + ix0
            let w_row = w_ch + ky * kw
            for kx = 0; kx < kw; kx = kx + 1 {
              let wv = @simd.f32x4_splat(weight.unsafe_get(w_row + kx))
              let xv = @simd.f32x4_const(
                input.unsafe_get(in_row + kx),
                input.unsafe_get(in_row + kx + 1),
                input.unsafe_get(in_row + kx + 2),
                input.unsafe_get(in_row + kx + 3),
              )
              acc = @simd.f32x4_relaxed_madd(xv, wv, acc)
            }
          }
          acc = @gemm.apply_epilogue_v128(acc, epilogue)
          out.unsafe_set(out_row + ox, @simd.f32x4_extract_lane(acc, 0))
          out.unsafe_set(out_row + ox + 1, @simd.f32x4_extract_lane(acc, 1))
          out.unsafe_set(out_row + ox + 2, @simd.f32x4_extract_lane(acc, 2))
          out.unsafe_set(out_row + ox + 3, @simd.f32x4_extract_lane(acc, 3))
          ox = ox + 4
        }

        // 中部尾部 + 右侧边界列
        while ox < out_w {
          let ix0 = ox - pad_left
          let mut acc = b_val
          for ky = 0; ky < kh; ky = ky + 1 {
            let iy = iy0 + ky * dil_h
            if iy < 0 || iy >= in_h {
              continue
            }
            let in_row = ch_in + iy * in_w
            let w_row = w_ch + ky * kw
            for kx = 0; kx < kw; kx = kx + 1 {
              let ix = ix0 + kx
              if ix >= 0 && ix < in_w {
                acc = acc +
                  input.unsafe_get(in_row + ix) * weight.unsafe_get(w_row + kx)
              }
            }
          }
          out.unsafe_set(
            out_row + ox,
            @gemm.apply_epilogue_scalar(acc, epilogue),
          )
          ox = ox + 1
        }
      } else {
        // 通用路径（stride_w > 1 或 dil_w > 1，例如 stride=2 下采样 Depthwise Conv）
        for ox = 0; ox < out_w; ox = ox + 1 {
          let ix0 = ox * stride_w - pad_left
          let mut acc = b_val
          if ox >= ox_min && ox < ox_max {
            for ky = 0; ky < kh; ky = ky + 1 {
              let iy = iy0 + ky * dil_h
              if iy < 0 || iy >= in_h {
                continue
              }
              let in_row = ch_in + iy * in_w + ix0
              let w_row = w_ch + ky * kw
              for kx = 0; kx < kw; kx = kx + 1 {
                acc = acc +
                  input.unsafe_get(in_row + kx * dil_w) *
                  weight.unsafe_get(w_row + kx)
              }
            }
          } else {
            for ky = 0; ky < kh; ky = ky + 1 {
              let iy = iy0 + ky * dil_h
              if iy < 0 || iy >= in_h {
                continue
              }
              let in_row = ch_in + iy * in_w
              let w_row = w_ch + ky * kw
              for kx = 0; kx < kw; kx = kx + 1 {
                let ix = ix0 + kx * dil_w
                if ix >= 0 && ix < in_w {
                  acc = acc +
                    input.unsafe_get(in_row + ix) *
                    weight.unsafe_get(w_row + kx)
                }
              }
            }
          }
          out.unsafe_set(
            out_row + ox,
            @gemm.apply_epilogue_scalar(acc, epilogue),
          )
        }
      }
    }
  }
}
