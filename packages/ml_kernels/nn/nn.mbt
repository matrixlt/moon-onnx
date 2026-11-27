///|
/// 对数组切片 `[offset, offset + count)` 执行一次性 O(1) 入口边界校验。
fn check_nn_bounds(
  len : Int32,
  offset : Int32,
  count : Int32,
  label : String,
) -> Unit {
  if count < 0 || offset < 0 || offset + count > len {
    abort(
      "ml_kernels/nn: out of bounds in \{label} (offset=\{offset}, count=\{count}, len=\{len})",
    )
  }
}

///|
/// 零分配高斯误差函数 Erf（Abramowitz & Stegun 7.1.26，最大误差 < 1.5e-7）。
pub fn erf_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
) -> Unit {
  if n == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, n, "erf_f32(x)")
  check_nn_bounds(y.length(), y_off, n, "erf_f32(y)")
  for i = 0; i < n; i = i + 1 {
    let v = x.unsafe_get(x_off + i).to_double()
    let sign = if v < 0.0 { -1.0 } else { 1.0 }
    let ax = if v < 0.0 { -v } else { v }
    let t = 1.0 / (1.0 + 0.3275911 * ax)
    let poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592
      ) *
      t
    y.unsafe_set(
      y_off + i,
      Float::from_double(sign * (1.0 - poly * @math.exp(-ax * ax))),
    )
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
  if n == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, n, "gelu_f32(x)")
  check_nn_bounds(y.length(), y_off, n, "gelu_f32(y)")
  let c = 0.7978845608028654
  for i = 0; i < n; i = i + 1 {
    let d = x.unsafe_get(x_off + i).to_double()
    y.unsafe_set(
      y_off + i,
      Float::from_double(
        0.5 * d * (1.0 + @math.tanh(c * (d + 0.044715 * d * d * d))),
      ),
    )
  }
}

///|
/// 零分配精确 GELU（`0.5 * x * (1 + erf(x / sqrt(2)))`，单遍融合计算，最大误差 < 1.5e-7，供 Whisper/BERT/ViT 融合使用）。
pub fn gelu_exact_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
) -> Unit {
  if n == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, n, "gelu_exact_f32(x)")
  check_nn_bounds(y.length(), y_off, n, "gelu_exact_f32(y)")
  let inv_sqrt2 = 0.7071067811865475
  for i = 0; i < n; i = i + 1 {
    let xd = x.unsafe_get(x_off + i).to_double()
    let v = xd * inv_sqrt2
    let sign = if v < 0.0 { -1.0 } else { 1.0 }
    let ax = if v < 0.0 { -v } else { v }
    let t = 1.0 / (1.0 + 0.3275911 * ax)
    let poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592
      ) *
      t
    let erf_v = sign * (1.0 - poly * @math.exp(-ax * ax))
    y.unsafe_set(y_off + i, Float::from_double(0.5 * xd * (1.0 + erf_v)))
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
  let total = outer * axis_dim * inner
  if total == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, total, "softmax_f32(x)")
  check_nn_bounds(y.length(), y_off, total, "softmax_f32(y)")
  if inner == 1 {
    // 末轴连续快速路径（Attention / 分类头最常见）
    for o = 0; o < outer; o = o + 1 {
      let xb = x_off + o * axis_dim
      let yb = y_off + o * axis_dim
      let mut mx = x.unsafe_get(xb)
      for k = 1; k < axis_dim; k = k + 1 {
        let v = x.unsafe_get(xb + k)
        if v > mx {
          mx = v
        }
      }
      let mut s = 0.0
      for k = 0; k < axis_dim; k = k + 1 {
        let e = @math.exp((x.unsafe_get(xb + k) - mx).to_double())
        y.unsafe_set(yb + k, Float::from_double(e))
        s = s + e
      }
      let inv = Float::from_double(1.0 / s)
      for k = 0; k < axis_dim; k = k + 1 {
        y.unsafe_set(yb + k, y.unsafe_get(yb + k) * inv)
      }
    }
    return
  }
  for o = 0; o < outer; o = o + 1 {
    let ob = o * axis_dim * inner
    for j = 0; j < inner; j = j + 1 {
      let mut mx = x.unsafe_get(x_off + ob + j)
      for k = 1; k < axis_dim; k = k + 1 {
        let v = x.unsafe_get(x_off + ob + k * inner + j)
        if v > mx {
          mx = v
        }
      }
      let mut s = 0.0
      for k = 0; k < axis_dim; k = k + 1 {
        let idx = ob + k * inner + j
        let e = @math.exp((x.unsafe_get(x_off + idx) - mx).to_double())
        y.unsafe_set(y_off + idx, Float::from_double(e))
        s = s + e
      }
      let inv = Float::from_double(1.0 / s)
      for k = 0; k < axis_dim; k = k + 1 {
        let idx = ob + k * inner + j
        y.unsafe_set(y_off + idx, y.unsafe_get(y_off + idx) * inv)
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
  let total = outer * norm_sz
  if total == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, total, "layernorm_f32(x)")
  check_nn_bounds(y.length(), y_off, total, "layernorm_f32(y)")
  check_nn_bounds(scale.length(), 0, norm_sz, "layernorm_f32(scale)")
  match bias {
    Some(bb) => check_nn_bounds(bb.length(), 0, norm_sz, "layernorm_f32(bias)")
    None => ()
  }
  let inv_n = 1.0 / norm_sz.to_double()
  let eps_d = eps.to_double()
  for o = 0; o < outer; o = o + 1 {
    let xb = x_off + o * norm_sz
    let yb = y_off + o * norm_sz
    let mut sum = 0.0
    for i = 0; i < norm_sz; i = i + 1 {
      sum = sum + x.unsafe_get(xb + i).to_double()
    }
    let mean = sum * inv_n
    let mut sq = 0.0
    for i = 0; i < norm_sz; i = i + 1 {
      let d = x.unsafe_get(xb + i).to_double() - mean
      sq = sq + d * d
    }
    let inv_std = 1.0 / (sq * inv_n + eps_d).sqrt()
    match bias {
      Some(bb) =>
        for i = 0; i < norm_sz; i = i + 1 {
          let norm = (x.unsafe_get(xb + i).to_double() - mean) *
            inv_std *
            scale.unsafe_get(i).to_double()
          y.unsafe_set(
            yb + i,
            Float::from_double(norm + bb.unsafe_get(i).to_double()),
          )
        }
      None =>
        for i = 0; i < norm_sz; i = i + 1 {
          let norm = (x.unsafe_get(xb + i).to_double() - mean) *
            inv_std *
            scale.unsafe_get(i).to_double()
          y.unsafe_set(yb + i, Float::from_double(norm))
        }
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
  let nc = n * c
  let in_plane = ih * iw
  let out_plane = oh * ow
  if nc == 0 || out_plane == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, nc * in_plane, "maxpool2d_f32(x)")
  check_nn_bounds(y.length(), y_off, nc * out_plane, "maxpool2d_f32(y)")
  let neg_inf = Float::from_double(-1.0e38)
  for idx = 0; idx < nc; idx = idx + 1 {
    let ib = x_off + idx * in_plane
    let ob = y_off + idx * out_plane
    for r = 0; r < oh; r = r + 1 {
      let hs = r * sh - pt
      let dst_row = ob + r * ow
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
                let v = x.unsafe_get(row_off + ic)
                if v > best {
                  best = v
                }
              }
            }
          }
        }
        y.unsafe_set(dst_row + col, best)
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
  if nc == 0 || out_plane == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, nc * in_plane, "avgpool2d_f32(x)")
  check_nn_bounds(y.length(), y_off, nc * out_plane, "avgpool2d_f32(y)")
  for idx = 0; idx < nc; idx = idx + 1 {
    let ib = x_off + idx * in_plane
    let ob = y_off + idx * out_plane
    for r = 0; r < oh; r = r + 1 {
      let hs = r * sh - pt
      let dst_row = ob + r * ow
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
                sum = sum + x.unsafe_get(row_off + ic).to_double()
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
        y.unsafe_set(dst_row + col, Float::from_double(sum / denom.to_double()))
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
  if nc == 0 || spatial == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, nc * spatial, "gap_f32(x)")
  check_nn_bounds(y.length(), y_off, nc, "gap_f32(y)")
  let inv = 1.0 / spatial.to_double()
  for idx = 0; idx < nc; idx = idx + 1 {
    let base = x_off + idx * spatial
    let mut sum = 0.0
    for s = 0; s < spatial; s = s + 1 {
      sum = sum + x.unsafe_get(base + s).to_double()
    }
    y.unsafe_set(y_off + idx, Float::from_double(sum * inv))
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
/// Wasm-GC 原生 Float 数组友好的 4 输出通道展开 NCHW 卷积矩阵乘：
/// 复用每次加载的感受野向量 `col[s, kk]` 并发累加 4 个输出通道（减少 75% 输入读取）并配合 `unsafe_get` / `unsafe_set`。
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
  if spatial == 0 || c_out == 0 {
    return
  }
  check_nn_bounds(col.length(), 0, spatial * k, "conv_nchw_scalar_f32(col)")
  check_nn_bounds(w.length(), 0, c_out * k, "conv_nchw_scalar_f32(w)")
  check_nn_bounds(
    out.length(),
    out_off,
    c_out * spatial,
    "conv_nchw_scalar_f32(out)",
  )
  match bias {
    Some(b) => check_nn_bounds(b.length(), 0, c_out, "conv_nchw_scalar_f32(b)")
    None => ()
  }
  let zero : Float = 0.0
  let mut oc = 0
  while oc + 4 <= c_out {
    let w0 = oc * k
    let w1 = w0 + k
    let w2 = w1 + k
    let w3 = w2 + k
    let o0 = out_off + oc * spatial
    let o1 = o0 + spatial
    let o2 = o1 + spatial
    let o3 = o2 + spatial
    let (b0, b1, b2, b3) = match bias {
      Some(b) =>
        (
          b.unsafe_get(oc),
          b.unsafe_get(oc + 1),
          b.unsafe_get(oc + 2),
          b.unsafe_get(oc + 3),
        )
      None => (zero, zero, zero, zero)
    }
    for s = 0; s < spatial; s = s + 1 {
      let c_row = s * k
      let mut a0 = b0
      let mut a1 = b1
      let mut a2 = b2
      let mut a3 = b3
      for kk = 0; kk < k; kk = kk + 1 {
        let cv = col.unsafe_get(c_row + kk)
        a0 = a0 + cv * w.unsafe_get(w0 + kk)
        a1 = a1 + cv * w.unsafe_get(w1 + kk)
        a2 = a2 + cv * w.unsafe_get(w2 + kk)
        a3 = a3 + cv * w.unsafe_get(w3 + kk)
      }
      out.unsafe_set(o0 + s, @gemm.apply_epilogue_scalar(a0, ep))
      out.unsafe_set(o1 + s, @gemm.apply_epilogue_scalar(a1, ep))
      out.unsafe_set(o2 + s, @gemm.apply_epilogue_scalar(a2, ep))
      out.unsafe_set(o3 + s, @gemm.apply_epilogue_scalar(a3, ep))
    }
    oc = oc + 4
  }
  while oc < c_out {
    let w_row = oc * k
    let out_ch = out_off + oc * spatial
    let b0 : Float = match bias {
      Some(b) => b.unsafe_get(oc)
      None => zero
    }
    for s = 0; s < spatial; s = s + 1 {
      let c_row = s * k
      let mut acc = b0
      for kk = 0; kk < k; kk = kk + 1 {
        acc = acc + col.unsafe_get(c_row + kk) * w.unsafe_get(w_row + kk)
      }
      out.unsafe_set(out_ch + s, @gemm.apply_epilogue_scalar(acc, ep))
    }
    oc = oc + 1
  }
}
