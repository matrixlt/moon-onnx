///|
/// ONNX `Silu` / Swish 激活算子：`y = x * sigmoid(x)`，直接复用 `matrixlt/ml_kernels/nn`。
pub fn silu_f32(
  x : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  @nn.silu_f32(src, 0, out, 0, n)
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX `RMSNormalization` / `SimplifiedLayerNormalization` 算子（双精度累加，支持自定义归一化轴）。
pub fn rms_norm_f32(
  x : @tensor.Tensor,
  scale : @tensor.Tensor,
  axis~ : Int32 = -1,
  epsilon~ : Float = (1.0e-5 : Float),
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sx = x.shape()
  let rank = sx.length()
  let ax = @tensor.normalize_axis(axis, rank)
  let mut outer = 1
  for d = 0; d < ax; d = d + 1 {
    outer = outer * sx[d]
  }
  let mut norm_sz = 1
  for d = ax; d < rank; d = d + 1 {
    norm_sz = norm_sz * sx[d]
  }
  let src = x.to_f32_buffer()
  let sc = scale.to_f32_buffer()
  if sc.length() != norm_sz {
    raise @tensor.ShapeMismatch(
      "rms_norm_f32: scale size (\{sc.length()}) != norm_sz (\{norm_sz})",
    )
  }
  let out = alloc_f32(pool, outer * norm_sz)
  @nn.rmsnorm_f32(src, 0, sc, out, 0, outer, norm_sz, epsilon)
  @tensor.Tensor::from_f32(sx, out)
}

///|
/// ONNX `com.microsoft::SkipSimplifiedLayerNormalization` 多输出算子：
/// 返回 `(output, dummy_mean, dummy_inv_std, sum_out)`，与官方 4 输出槽位（含空位 `""`）严格对齐。
pub fn skip_rms_norm_f32(
  x : @tensor.Tensor,
  skip : @tensor.Tensor,
  scale : @tensor.Tensor,
  bias : @tensor.Tensor?,
  epsilon~ : Float = (1.0e-5 : Float),
  need_sum_out~ : Bool = true,
  pool~ : @tensor.BufferPool? = None,
) -> Array[@tensor.Tensor] raise @tensor.TensorError {
  let sx = x.shape()
  let rank = sx.length()
  if rank < 1 {
    raise @tensor.InvalidShape("skip_rms_norm_f32 requires rank >= 1")
  }
  let norm_sz = sx[rank - 1]
  let total = @tensor.num_elements(sx)
  let outer = total / norm_sz
  let x_buf = x.to_f32_buffer()
  let skip_buf = skip.to_f32_buffer()
  let sc_buf = scale.to_f32_buffer()
  let b_opt : FixedArray[Float]? = match bias {
    Some(bt) => Some(bt.to_f32_buffer())
    None => None
  }
  let out_buf = alloc_f32(pool, total)
  let sum_buf : FixedArray[Float]? = if need_sum_out {
    Some(alloc_f32(pool, total))
  } else {
    None
  }
  @nn.skip_rmsnorm_f32(
    x_buf, 0, skip_buf, 0, sc_buf, b_opt, out_buf, 0, sum_buf, 0, outer, norm_sz,
    epsilon,
  )
  let out_t = @tensor.Tensor::from_f32(sx, out_buf)
  let empty_t = @tensor.Tensor::from_f32(
    [0],
    FixedArray::make(0, (0.0 : Float)),
  )
  let sum_t = match sum_buf {
    Some(sb) => @tensor.Tensor::from_f32(sx, sb)
    None => empty_t
  }
  [out_t, empty_t, empty_t, sum_t]
}

///|
/// ONNX `com.microsoft::RotaryEmbedding` 算子：支持 3D `[B, S, H*D]` 与 4D `[B, H, S, D]` 输入。
pub fn rotary_embedding_f32(
  x : @tensor.Tensor,
  position_ids : @tensor.Tensor,
  cos_cache : @tensor.Tensor,
  sin_cache : @tensor.Tensor,
  interleaved~ : Int32 = 0,
  num_heads~ : Int32 = 0,
  rotary_embedding_dim~ : Int32 = 0,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sx = x.shape()
  let rank = sx.length()
  let cs = cos_cache.shape()
  if cs.length() != 2 {
    raise @tensor.InvalidShape(
      "rotary_embedding_f32: cos_cache must be 2D, got \{@tensor.format_shape(cs)}",
    )
  }
  let cache_half = cs[1]
  let is_bhsd = rank == 4
  let batch = sx[0]
  let seq_len = if is_bhsd { sx[2] } else { sx[1] }
  let (eff_heads, head_dim) = if is_bhsd {
    (sx[1], sx[3])
  } else if rank == 3 {
    let hidden = sx[2]
    if num_heads > 0 {
      (num_heads, hidden / num_heads)
    } else {
      let hd = cache_half * 2
      (hidden / hd, hd)
    }
  } else {
    raise @tensor.InvalidShape(
      "rotary_embedding_f32 requires 3D or 4D input, got rank \{rank}",
    )
  }
  let rot_dim = if rotary_embedding_dim > 0 {
    rotary_embedding_dim
  } else {
    cache_half * 2
  }
  let pos_buf = position_ids.to_i64_buffer()
  let pos_is_scalar = pos_buf.length() == 1 && batch * seq_len > 1
  let src = x.to_f32_buffer()
  let cos_buf = cos_cache.to_f32_buffer()
  let sin_buf = sin_cache.to_f32_buffer()
  let out = alloc_f32(pool, src.length())
  @nn.rotary_embedding_f32(
    src,
    0,
    pos_buf,
    pos_is_scalar,
    cos_buf,
    sin_buf,
    cache_half,
    out,
    0,
    batch,
    seq_len,
    eff_heads,
    head_dim,
    rot_dim,
    interleaved != 0,
    is_bhsd,
  )
  @tensor.Tensor::from_f32(sx, out)
}

///|
/// ONNX `com.microsoft::GroupQueryAttention` 多输出算子：
/// 返回 `[output, present_key, present_value]`，完整支持 KV-Cache 拼接、因果掩码与滑动窗口/Softcap。
pub fn gqa_f32(
  query : @tensor.Tensor,
  key : @tensor.Tensor?,
  value : @tensor.Tensor?,
  past_key : @tensor.Tensor?,
  past_value : @tensor.Tensor?,
  seqlens_k : @tensor.Tensor,
  total_sequence_length : @tensor.Tensor,
  cos_cache : @tensor.Tensor?,
  sin_cache : @tensor.Tensor?,
  num_heads~ : Int32 = 1,
  kv_num_heads~ : Int32 = 1,
  scale~ : Float = (0.0 : Float),
  local_window_size~ : Int32 = -1,
  softcap~ : Float = (0.0 : Float),
  do_rotary~ : Int32 = 0,
  rotary_interleaved~ : Int32 = 0,
  pool~ : @tensor.BufferPool? = None,
) -> Array[@tensor.Tensor] raise @tensor.TensorError {
  let _ = (cos_cache, sin_cache, do_rotary, rotary_interleaved)
  let sq = query.shape()
  if sq.length() != 3 {
    raise @tensor.InvalidShape(
      "gqa_f32: query must be 3D [B, S, H*D], got \{@tensor.format_shape(sq)}",
    )
  }
  let batch = sq[0]
  let seq_len = sq[1]
  let head_dim = sq[2] / num_heads

  let (k_t, v_t) = match (key, value) {
    (Some(kt), Some(vt)) => (kt, vt)
    _ =>
      raise @tensor.Unsupported(
        "gqa_f32: packed QKV without separate key/value inputs is not supported yet",
      )
  }

  let slk_raw = match seqlens_k.dtype() {
    @tensor.Int32 => seqlens_k.to_i32_buffer()
    @tensor.Int64 => {
      let b = seqlens_k.to_i64_buffer()
      FixedArray::makei(b.length(), fn(i) { b[i].to_int() })
    }
    other =>
      raise @tensor.DTypeMismatch(
        "gqa_f32: seqlens_k must be Int32/Int64, got \{other}",
      )
  }

  let tot_seq = match total_sequence_length.dtype() {
    @tensor.Int32 => total_sequence_length.to_i32_buffer()[0]
    @tensor.Int64 => total_sequence_length.to_i64_buffer()[0].to_int()
    other =>
      raise @tensor.DTypeMismatch(
        "gqa_f32: total_sequence_length must be Int32/Int64, got \{other}",
      )
  }

  let (past_k_buf, past_v_buf, past_seq) = match (past_key, past_value) {
    (Some(pk), Some(pv)) => {
      let spk = pk.shape()
      let ps = if spk.length() >= 3 { spk[2] } else { 0 }
      let pk_b = if pk.numel() == 0 {
        FixedArray::make(0, (0.0 : Float))
      } else {
        pk.to_f32_buffer()
      }
      let pv_b = if pv.numel() == 0 {
        FixedArray::make(0, (0.0 : Float))
      } else {
        pv.to_f32_buffer()
      }
      (pk_b, pv_b, ps)
    }
    _ =>
      (
        FixedArray::make(0, (0.0 : Float)),
        FixedArray::make(0, (0.0 : Float)),
        0,
      )
  }
  let present_seq = if tot_seq > past_seq + seq_len {
    tot_seq
  } else {
    past_seq + seq_len
  }

  let q_buf = query.to_f32_buffer()
  let k_buf = k_t.to_f32_buffer()
  let v_buf = v_t.to_f32_buffer()
  let out_buf = alloc_f32(pool, batch * seq_len * num_heads * head_dim)
  let pk_shape : Array[Int32] = [batch, kv_num_heads, present_seq, head_dim]
  let pk_numel = batch * kv_num_heads * present_seq * head_dim
  let present_k_buf = FixedArray::make(pk_numel, (0.0 : Float))
  let present_v_buf = FixedArray::make(pk_numel, (0.0 : Float))

  @nn.gqa_f32(
    q_buf, 0, k_buf, 0, v_buf, 0, past_k_buf, 0, past_v_buf, 0, slk_raw, out_buf,
    0, present_k_buf, 0, present_v_buf, 0, batch, seq_len, past_seq, present_seq,
    num_heads, kv_num_heads, head_dim, scale, local_window_size, softcap,
  )

  [
    @tensor.Tensor::from_f32(sq, out_buf),
    @tensor.Tensor::from_f32(pk_shape, present_k_buf),
    @tensor.Tensor::from_f32(pk_shape.copy(), present_v_buf),
  ]
}

///|
/// ONNX `com.microsoft::MatMulNBits`（4-bit 分块量化矩阵乘法）算子：
/// 自动根据 `use_simd` 分派至 V128 奇偶拆分 4 列展开内核或标量参考内核，支持 N-D × Q4 权重、可选零点与偏置。
pub fn matmul_nbits_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  scales : @tensor.Tensor,
  zero_points : @tensor.Tensor?,
  bias : @tensor.Tensor?,
  k~ : Int32 = 0,
  n~ : Int32 = 0,
  bits~ : Int32 = 4,
  block_size~ : Int32 = 32,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  if bits != 4 {
    raise @tensor.Unsupported(
      "matmul_nbits_op: currently only bits=4 is supported, got bits=\{bits}",
    )
  }
  let sa = a.shape()
  let rank = sa.length()
  if rank < 2 {
    raise @tensor.InvalidShape(
      "matmul_nbits_op: input A must have rank >= 2, got rank \{rank}",
    )
  }
  let eff_k = if k > 0 { k } else { sa[rank - 1] }
  let eff_n = if n > 0 { n } else { b.shape()[0] }
  let mut m = 1
  for d = 0; d < rank - 1; d = d + 1 {
    m = m * sa[d]
  }
  let out_shape = sa.copy()
  out_shape[rank - 1] = eff_n

  let a_buf = a.to_f32_buffer()
  let b_buf = b.to_u8_buffer()
  let sc_buf = scales.to_f32_buffer()
  let (zp_bytes, zp_f32) : (FixedArray[Byte]?, FixedArray[Float]?) = match
    zero_points {
    Some(zt) =>
      match zt.dtype() {
        @tensor.UInt8 => (Some(zt.to_u8_buffer()), None)
        @tensor.Float32 => (None, Some(zt.to_f32_buffer()))
        other =>
          raise @tensor.DTypeMismatch(
            "matmul_nbits_op: unsupported zero_points dtype \{other}",
          )
      }
    None => (None, None)
  }
  let bias_buf : FixedArray[Float]? = match bias {
    Some(bt) => Some(bt.to_f32_buffer())
    None => None
  }
  let out_buf = alloc_f32(pool, m * eff_n)
  if use_simd {
    @quant.matmul_nbits_q4_v128_f32(
      a_buf, 0, b_buf, sc_buf, zp_bytes, zp_f32, bias_buf, out_buf, 0, m, eff_k,
      eff_n, block_size,
    )
  } else {
    @quant.matmul_nbits_q4_scalar_f32(
      a_buf, 0, b_buf, sc_buf, zp_bytes, zp_f32, bias_buf, out_buf, 0, m, eff_k,
      eff_n, block_size,
    )
  }
  @tensor.Tensor::from_f32(out_shape, out_buf)
}
