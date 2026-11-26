///|
/// 零分配高斯误差函数 Erf（Abramowitz & Stegun 7.1.26，最大误差 < 1.5e-7）。
pub fn erf_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
) -> Unit {
  for i = 0; i < n; i = i + 1 {
    let v = x[x_off + i].to_double()
    let sign = if v < 0.0 { -1.0 } else { 1.0 }
    let ax = if v < 0.0 { -v } else { v }
    let t = 1.0 / (1.0 + 0.3275911 * ax)
    let poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592
      ) *
      t
    y[y_off + i] = Float::from_double(sign * (1.0 - poly * @math.exp(-ax * ax)))
  }
}

///|
/// 零分配 GELU（Tanh 近似），支持 Arena 偏移读写。
pub fn gelu_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
) -> Unit {
  let c = 0.7978845608028654
  for i = 0; i < n; i = i + 1 {
    let d = x[x_off + i].to_double()
    y[y_off + i] = Float::from_double(
      0.5 * d * (1.0 + @math.tanh(c * (d + 0.044715 * d * d * d))),
    )
  }
}

///|
/// 零分配数值稳定 Softmax，沿 `[outer, axis_dim, inner]` 的中间轴执行归一化。
pub fn softmax_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  outer : Int32,
  axis_dim : Int32,
  inner : Int32,
) -> Unit {
  for o = 0; o < outer; o = o + 1 {
    let ob = o * axis_dim * inner
    for j = 0; j < inner; j = j + 1 {
      let mut mx = x[x_off + ob + j]
      for k = 1; k < axis_dim; k = k + 1 {
        let v = x[x_off + ob + k * inner + j]
        if v > mx {
          mx = v
        }
      }
      let mut s = 0.0
      for k = 0; k < axis_dim; k = k + 1 {
        let idx = ob + k * inner + j
        let e = @math.exp((x[x_off + idx] - mx).to_double())
        y[y_off + idx] = Float::from_double(e)
        s = s + e
      }
      let inv = Float::from_double(1.0 / s)
      for k = 0; k < axis_dim; k = k + 1 {
        let idx = ob + k * inner + j
        y[y_off + idx] = y[y_off + idx] * inv
      }
    }
  }
}

///|
/// 零分配双精度累加 LayerNormalization（对 `[outer, norm_sz]` 逐行归一化）。
pub fn layernorm_f32(
  x : FixedArray[Float],
  x_off : Int32,
  scale : FixedArray[Float],
  bias : FixedArray[Float]?,
  y : FixedArray[Float],
  y_off : Int32,
  outer : Int32,
  norm_sz : Int32,
  eps : Float,
) -> Unit {
  let inv_n = 1.0 / norm_sz.to_double()
  let eps_d = eps.to_double()
  for o = 0; o < outer; o = o + 1 {
    let b = o * norm_sz
    let mut sum = 0.0
    for i = 0; i < norm_sz; i = i + 1 {
      sum = sum + x[x_off + b + i].to_double()
    }
    let mean = sum * inv_n
    let mut sq = 0.0
    for i = 0; i < norm_sz; i = i + 1 {
      let d = x[x_off + b + i].to_double() - mean
      sq = sq + d * d
    }
    let inv_std = 1.0 / (sq * inv_n + eps_d).sqrt()
    for i = 0; i < norm_sz; i = i + 1 {
      let norm = (x[x_off + b + i].to_double() - mean) *
        inv_std *
        scale[i].to_double()
      let bv = match bias {
        Some(bb) => bb[i].to_double()
        None => 0.0
      }
      y[y_off + b + i] = Float::from_double(norm + bv)
    }
  }
}

///|
/// 零分配 2D MaxPool（直接在静态 Arena 偏移间读写）。
pub fn maxpool2d_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
  c : Int32,
  ih : Int32,
  iw : Int32,
  kh : Int32,
  kw : Int32,
  sh : Int32,
  sw : Int32,
  pt : Int32,
  pl : Int32,
  dh : Int32,
  dw : Int32,
  oh : Int32,
  ow : Int32,
) -> Unit {
  let neg_inf = Float::from_double(-1.0e38)
  let nc = n * c
  let in_plane = ih * iw
  let out_plane = oh * ow
  for idx = 0; idx < nc; idx = idx + 1 {
    let ib = x_off + idx * in_plane
    let ob = y_off + idx * out_plane
    for r = 0; r < oh; r = r + 1 {
      let hs = r * sh - pt
      for col = 0; col < ow; col = col + 1 {
        let ws = col * sw - pl
        let mut best = neg_inf
        for kr = 0; kr < kh; kr = kr + 1 {
          let ir = hs + kr * dh
          if ir >= 0 && ir < ih {
            let row_off = ib + ir * iw
            for kc = 0; kc < kw; kc = kc + 1 {
              let ic = ws + kc * dw
              if ic >= 0 && ic < iw {
                let v = x[row_off + ic]
                if v > best {
                  best = v
                }
              }
            }
          }
        }
        y[ob + r * ow + col] = best
      }
    }
  }
}

///|
/// 零分配 2D AveragePool（直接在静态 Arena 偏移间读写）。
pub fn avgpool2d_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
  c : Int32,
  ih : Int32,
  iw : Int32,
  kh : Int32,
  kw : Int32,
  sh : Int32,
  sw : Int32,
  pt : Int32,
  pl : Int32,
  oh : Int32,
  ow : Int32,
  include_pad : Bool,
) -> Unit {
  let nc = n * c
  let in_plane = ih * iw
  let out_plane = oh * ow
  for idx = 0; idx < nc; idx = idx + 1 {
    let ib = x_off + idx * in_plane
    let ob = y_off + idx * out_plane
    for r = 0; r < oh; r = r + 1 {
      let hs = r * sh - pt
      for col = 0; col < ow; col = col + 1 {
        let ws = col * sw - pl
        let mut sum = 0.0
        let mut cnt = 0
        for kr = 0; kr < kh; kr = kr + 1 {
          let ir = hs + kr
          if ir >= 0 && ir < ih {
            let row_off = ib + ir * iw
            for kc = 0; kc < kw; kc = kc + 1 {
              let ic = ws + kc
              if ic >= 0 && ic < iw {
                sum = sum + x[row_off + ic].to_double()
                cnt = cnt + 1
              }
            }
          }
        }
        let denom = if include_pad {
          kh * kw
        } else if cnt > 0 {
          cnt
        } else {
          1
        }
        y[ob + r * ow + col] = Float::from_double(sum / denom.to_double())
      }
    }
  }
}

///|
/// 零分配 GlobalAveragePool（对 `[nc, spatial]` 沿空间维度求均值）。
pub fn gap_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  nc : Int32,
  spatial : Int32,
) -> Unit {
  let inv = 1.0 / spatial.to_double()
  for idx = 0; idx < nc; idx = idx + 1 {
    let base = x_off + idx * spatial
    let mut sum = 0.0
    for s = 0; s < spatial; s = s + 1 {
      sum = sum + x[base + s].to_double()
    }
    y[y_off + idx] = Float::from_double(sum * inv)
  }
}

///|
/// 零分配 `reflect` 填充镜像坐标映射。
pub fn reflect_idx(idx : Int32, dim : Int32) -> Int32 {
  if dim <= 1 {
    return 0
  }
  let mut cur = idx
  while cur < 0 || cur >= dim {
    if cur < 0 {
      cur = -cur
    }
    if cur >= dim {
      cur = 2 * dim - 2 - cur
    }
  }
  cur
}

///|
/// Wasm-GC 原生 Float 数组友好的 NCHW 卷积矩阵乘（避免 GC 字节数组 `v128_load` 退化）。
pub fn conv_nchw_scalar_f32(
  col : FixedArray[Float],
  w : FixedArray[Float],
  bias : FixedArray[Float]?,
  out : FixedArray[Float],
  out_off : Int32,
  spatial : Int32,
  k : Int32,
  c_out : Int32,
  ep : @gemm.Epilogue,
) -> Unit {
  for oc = 0; oc < c_out; oc = oc + 1 {
    let w_row = oc * k
    let out_ch = out_off + oc * spatial
    let b0 : Float = match bias {
      Some(b) => b[oc]
      None => (0.0 : Float)
    }
    for s = 0; s < spatial; s = s + 1 {
      let c_row = s * k
      let mut acc = b0
      for kk = 0; kk < k; kk = kk + 1 {
        acc = acc + col[c_row + kk] * w[w_row + kk]
      }
      out[out_ch + s] = @gemm.apply_epilogue_scalar(acc, ep)
    }
  }
}
