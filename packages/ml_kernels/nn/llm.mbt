///|
/// 零分配 SiLU / Swish 激活函数：`y = x / (1 + exp(-x))`（Llama / Qwen / SmolLM2 / YOLOv8 标配）。
pub fn silu_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
) -> Unit {
  for i = 0; i < n; i = i + 1 {
    let v = x[x_off + i].to_double()
    let sig = 1.0 / (1.0 + @math.exp(-v))
    y[y_off + i] = Float::from_double(v * sig)
  }
}

///|
/// 零分配双精度累加 RMSNormalization（ONNX `SimplifiedLayerNormalization` / Opset 23 `RMSNormalization`）：
/// `y[i] = x[i] / sqrt(mean(x^2) + eps) * scale[i]`。
pub fn rmsnorm_f32(
  x : FixedArray[Float],
  x_off : Int32,
  scale : FixedArray[Float],
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
    let mut sq = 0.0
    for i = 0; i < norm_sz; i = i + 1 {
      let v = x[x_off + b + i].to_double()
      sq = sq + v * v
    }
    let inv_rms = 1.0 / (sq * inv_n + eps_d).sqrt()
    for i = 0; i < norm_sz; i = i + 1 {
      y[y_off + b + i] = Float::from_double(
        x[x_off + b + i].to_double() * inv_rms * scale[i].to_double(),
      )
    }
  }
}

///|
/// 零分配融合残差加 + RMSNormalization（ONNX `com.microsoft::SkipSimplifiedLayerNormalization`）：
/// - `sum = x + skip (+ bias)`（若 `sum_out` 为 `Some` 则写出供下游残差连接复用）；
/// - `y = rmsnorm(sum, scale, eps)`。
pub fn skip_rmsnorm_f32(
  x : FixedArray[Float],
  x_off : Int32,
  skip : FixedArray[Float],
  skip_off : Int32,
  scale : FixedArray[Float],
  bias : FixedArray[Float]?,
  y : FixedArray[Float],
  y_off : Int32,
  sum_out : FixedArray[Float]?,
  sum_off : Int32,
  outer : Int32,
  norm_sz : Int32,
  eps : Float,
) -> Unit {
  let inv_n = 1.0 / norm_sz.to_double()
  let eps_d = eps.to_double()
  for o = 0; o < outer; o = o + 1 {
    let b = o * norm_sz
    let mut sq = 0.0
    for i = 0; i < norm_sz; i = i + 1 {
      let bv = match bias {
        Some(bb) => bb[i].to_double()
        None => 0.0
      }
      let s = x[x_off + b + i].to_double() +
        skip[skip_off + b + i].to_double() +
        bv
      match sum_out {
        Some(so) => so[sum_off + b + i] = Float::from_double(s)
        None => ()
      }
      sq = sq + s * s
    }
    let inv_rms = 1.0 / (sq * inv_n + eps_d).sqrt()
    for i = 0; i < norm_sz; i = i + 1 {
      let s = match sum_out {
        Some(so) => so[sum_off + b + i].to_double()
        None => {
          let bv = match bias {
            Some(bb) => bb[i].to_double()
            None => 0.0
          }
          x[x_off + b + i].to_double() + skip[skip_off + b + i].to_double() + bv
        }
      }
      y[y_off + b + i] = Float::from_double(s * inv_rms * scale[i].to_double())
    }
  }
}

///|
/// 零分配旋转位置编码（ONNX `com.microsoft::RotaryEmbedding`）：
/// 支持 3D `[B, S, num_heads * head_dim]`（`is_bhsd = false`）与 4D `[B, num_heads, S, head_dim]`（`is_bhsd = true`）布局，
/// 兼容半切分（`interleaved = false`，Llama/Qwen/SmolLM2）与交错对（`interleaved = true`，GPT-J）两种模式，
/// 以及部分维度旋转（`rotary_dim < head_dim` 时尾部通道原样透传）。
pub fn rotary_embedding_f32(
  x : FixedArray[Float],
  x_off : Int32,
  pos_ids : FixedArray[Int64],
  pos_is_scalar : Bool,
  cos_cache : FixedArray[Float],
  sin_cache : FixedArray[Float],
  cache_half_dim : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  batch : Int32,
  seq_len : Int32,
  num_heads : Int32,
  head_dim : Int32,
  rotary_dim : Int32,
  interleaved : Bool,
  is_bhsd : Bool,
) -> Unit {
  let half = rotary_dim / 2
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq_len; s = s + 1 {
      let pos = if pos_is_scalar {
        pos_ids[0].to_int() + s
      } else {
        pos_ids[b * seq_len + s].to_int()
      }
      let c_base = pos * cache_half_dim
      for h = 0; h < num_heads; h = h + 1 {
        let base = if is_bhsd {
          ((b * num_heads + h) * seq_len + s) * head_dim
        } else {
          ((b * seq_len + s) * num_heads + h) * head_dim
        }
        if interleaved {
          for i = 0; i < half; i = i + 1 {
            let c = cos_cache[c_base + i].to_double()
            let sn = sin_cache[c_base + i].to_double()
            let i0 = base + i * 2
            let i1 = i0 + 1
            let x0 = x[x_off + i0].to_double()
            let x1 = x[x_off + i1].to_double()
            y[y_off + i0] = Float::from_double(x0 * c - x1 * sn)
            y[y_off + i1] = Float::from_double(x0 * sn + x1 * c)
          }
        } else {
          for i = 0; i < half; i = i + 1 {
            let c = cos_cache[c_base + i].to_double()
            let sn = sin_cache[c_base + i].to_double()
            let i0 = base + i
            let i1 = base + i + half
            let x0 = x[x_off + i0].to_double()
            let x1 = x[x_off + i1].to_double()
            y[y_off + i0] = Float::from_double(x0 * c - x1 * sn)
            y[y_off + i1] = Float::from_double(x0 * sn + x1 * c)
          }
        }
        // 未参与旋转的尾部维度直接透传
        for i = rotary_dim; i < head_dim; i = i + 1 {
          y[y_off + base + i] = x[x_off + base + i]
        }
      }
    }
  }
}

///|
/// 零分配分组查询注意力（ONNX `com.microsoft::GroupQueryAttention`）：
/// 1. 将 `past_key` / `past_value` `[B, kv_heads, past_seq, D]` 与当前 `key` / `value` `[B, S, kv_heads * D]`
///    合并写入 `present_key` / `present_value` `[B, kv_heads, present_seq, D]`；
/// 2. 采用在线流式 Softmax（FlashAttention 风格，$O(D)$ 栈缓冲、零 $O(S^2)$ 分数矩阵分配）执行因果 GQA，
///    原生支持 `local_window_size` 滑动窗口与 `softcap`（Gemma-2）。
pub fn gqa_f32(
  q : FixedArray[Float],
  q_off : Int32,
  k : FixedArray[Float],
  k_off : Int32,
  v : FixedArray[Float],
  v_off : Int32,
  past_k : FixedArray[Float],
  past_k_off : Int32,
  past_v : FixedArray[Float],
  past_v_off : Int32,
  seqlens_k : FixedArray[Int32],
  out : FixedArray[Float],
  out_off : Int32,
  present_k : FixedArray[Float],
  present_k_off : Int32,
  present_v : FixedArray[Float],
  present_v_off : Int32,
  batch : Int32,
  seq_len : Int32,
  past_seq : Int32,
  present_seq : Int32,
  num_heads : Int32,
  kv_num_heads : Int32,
  head_dim : Int32,
  scale : Float,
  local_window_size : Int32,
  softcap : Float,
) -> Unit {
  let eff_scale = if scale == (0.0 : Float) {
    1.0 / head_dim.to_double().sqrt()
  } else {
    scale.to_double()
  }
  let softcap_d = softcap.to_double()
  let has_softcap = softcap_d > 0.0
  let group_size = num_heads / kv_num_heads
  let kv_stride = kv_num_heads * head_dim
  let q_stride = num_heads * head_dim

  // 1. 构造 present_k / present_v：[B, kv_num_heads, present_seq, head_dim]
  for b = 0; b < batch; b = b + 1 {
    let valid_total = seqlens_k[b] + 1
    let write_start = if valid_total >= seq_len {
      valid_total - seq_len
    } else {
      0
    }
    for kh = 0; kh < kv_num_heads; kh = kh + 1 {
      let pk_base = present_k_off +
        (b * kv_num_heads + kh) * present_seq * head_dim
      let pv_base = present_v_off +
        (b * kv_num_heads + kh) * present_seq * head_dim
      let past_k_base = past_k_off +
        (b * kv_num_heads + kh) * past_seq * head_dim
      let past_v_base = past_v_off +
        (b * kv_num_heads + kh) * past_seq * head_dim
      let copy_past = if past_seq < present_seq {
        past_seq
      } else {
        present_seq
      }
      for t = 0; t < copy_past; t = t + 1 {
        let p_off = t * head_dim
        for d = 0; d < head_dim; d = d + 1 {
          present_k[pk_base + p_off + d] = past_k[past_k_base + p_off + d]
          present_v[pv_base + p_off + d] = past_v[past_v_base + p_off + d]
        }
      }
      for s = 0; s < seq_len; s = s + 1 {
        let dst_t = write_start + s
        if dst_t >= 0 && dst_t < present_seq {
          let dst_off = dst_t * head_dim
          let src_off = (b * seq_len + s) * kv_stride + kh * head_dim
          for d = 0; d < head_dim; d = d + 1 {
            present_k[pk_base + dst_off + d] = k[k_off + src_off + d]
            present_v[pv_base + dst_off + d] = v[v_off + src_off + d]
          }
        }
      }
    }
  }

  // 2. 在线流式 Softmax 执行因果 GQA（仅 O(head_dim) 临时累加器）
  let acc = FixedArray::make(head_dim, 0.0)
  for b = 0; b < batch; b = b + 1 {
    let valid_total = seqlens_k[b] + 1
    let max_valid = if valid_total < present_seq {
      valid_total
    } else {
      present_seq
    }
    let q_pos_base = if valid_total >= seq_len {
      valid_total - seq_len
    } else {
      0
    }
    for s = 0; s < seq_len; s = s + 1 {
      let q_pos = q_pos_base + s
      let tk_end = if q_pos < max_valid { q_pos } else { max_valid - 1 }
      let tk_start = if local_window_size > 0 &&
        q_pos - local_window_size + 1 > 0 {
        q_pos - local_window_size + 1
      } else {
        0
      }
      for h = 0; h < num_heads; h = h + 1 {
        let kh = h / group_size
        let q_base = q_off + (b * seq_len + s) * q_stride + h * head_dim
        let o_base = out_off + (b * seq_len + s) * q_stride + h * head_dim
        let pk_base = present_k_off +
          (b * kv_num_heads + kh) * present_seq * head_dim
        let pv_base = present_v_off +
          (b * kv_num_heads + kh) * present_seq * head_dim

        for d = 0; d < head_dim; d = d + 1 {
          acc[d] = 0.0
        }
        let mut max_score = -1.0e300
        let mut sum_exp = 0.0

        for tk = tk_start; tk <= tk_end; tk = tk + 1 {
          let k_row = pk_base + tk * head_dim
          let mut dot = 0.0
          for d = 0; d < head_dim; d = d + 1 {
            dot = dot +
              q[q_base + d].to_double() * present_k[k_row + d].to_double()
          }
          let mut sc = dot * eff_scale
          if has_softcap {
            sc = softcap_d * @math.tanh(sc / softcap_d)
          }
          let v_row = pv_base + tk * head_dim
          if sc > max_score {
            let rescale = if sum_exp == 0.0 {
              0.0
            } else {
              @math.exp(max_score - sc)
            }
            sum_exp = sum_exp * rescale + 1.0
            for d = 0; d < head_dim; d = d + 1 {
              acc[d] = acc[d] * rescale + present_v[v_row + d].to_double()
            }
            max_score = sc
          } else {
            let w = @math.exp(sc - max_score)
            sum_exp = sum_exp + w
            for d = 0; d < head_dim; d = d + 1 {
              acc[d] = acc[d] + w * present_v[v_row + d].to_double()
            }
          }
        }

        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 }
        for d = 0; d < head_dim; d = d + 1 {
          out[o_base + d] = Float::from_double(acc[d] * inv_sum)
        }
      }
    }
  }
}
