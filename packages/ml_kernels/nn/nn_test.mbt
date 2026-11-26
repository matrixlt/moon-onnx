///|
test "ml_kernels/nn zero-allocation kernels match analytical references" {
  // 1. Erf
  let x : FixedArray[Float] = [0.0, 1.0, -1.0]
  let y = FixedArray::make(3, (0.0 : Float))
  @nn.erf_f32(x, 0, y, 0, 3)
  assert_true(y[0].abs() < (1.0e-6 : Float))
  assert_true((y[1] - (0.8427008 : Float)).abs() < (1.0e-5 : Float))
  assert_true((y[2] + (0.8427008 : Float)).abs() < (1.0e-5 : Float))

  // 2. Softmax
  let sx : FixedArray[Float] = [1.0, 2.0, 3.0]
  let sy = FixedArray::make(3, (0.0 : Float))
  @nn.softmax_f32(sx, 0, sy, 0, 1, 3, 1)
  assert_true((sy[0] + sy[1] + sy[2] - (1.0 : Float)).abs() < (1.0e-6 : Float))
  assert_true(sy[2] > sy[1] && sy[1] > sy[0])

  // 3. LayerNorm
  let lx : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0]
  let sc : FixedArray[Float] = [1.0, 1.0, 1.0, 1.0]
  let ly = FixedArray::make(4, (0.0 : Float))
  @nn.layernorm_f32(lx, 0, sc, None, ly, 0, 1, 4, (1.0e-5 : Float))
  assert_true((ly[0] + ly[3]).abs() < (1.0e-5 : Float))

  // 4. reflect_idx
  assert_eq(@nn.reflect_idx(-1, 4), 1)
  assert_eq(@nn.reflect_idx(4, 4), 2)
  assert_eq(@nn.reflect_idx(0, 1), 0)

  // 5. GAP
  let gx : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
  let gy = FixedArray::make(2, (0.0 : Float))
  @nn.gap_f32(gx, 0, gy, 0, 2, 4)
  assert_true((gy[0] - (2.5 : Float)).abs() < (1.0e-6 : Float))
  assert_true((gy[1] - (25.0 : Float)).abs() < (1.0e-6 : Float))
}

///|
test "ml_kernels/nn LLM kernels (SiLU, RMSNorm, SkipRMSNorm, RoPE, GQA)" {
  // 1. SiLU
  let sx : FixedArray[Float] = [0.0, 2.0]
  let sy = FixedArray::make(2, (0.0 : Float))
  @nn.silu_f32(sx, 0, sy, 0, 2)
  assert_true(sy[0].abs() < (1.0e-6 : Float))
  assert_true((sy[1] - (1.7615942 : Float)).abs() < (1.0e-5 : Float))

  // 2. RMSNorm & SkipRMSNorm
  let rx : FixedArray[Float] = [3.0, 4.0]
  let rskip : FixedArray[Float] = [0.0, 0.0]
  let rscale : FixedArray[Float] = [1.0, 2.0]
  let ry = FixedArray::make(2, (0.0 : Float))
  let rsy = FixedArray::make(2, (0.0 : Float))
  let rsum = FixedArray::make(2, (0.0 : Float))
  @nn.rmsnorm_f32(rx, 0, rscale, ry, 0, 1, 2, (0.0 : Float))
  @nn.skip_rmsnorm_f32(
    rx,
    0,
    rskip,
    0,
    rscale,
    None,
    rsy,
    0,
    Some(rsum),
    0,
    1,
    2,
    (0.0 : Float),
  )
  // rms = sqrt((9+16)/2) = sqrt(12.5) = 3.5355339
  assert_true((ry[0] - (0.8485281 : Float)).abs() < (1.0e-5 : Float))
  assert_true((ry[1] - (2.2627417 : Float)).abs() < (1.0e-5 : Float))
  assert_true((rsy[0] - ry[0]).abs() < (1.0e-6 : Float))
  assert_true((rsum[1] - (4.0 : Float)).abs() < (1.0e-6 : Float))

  // 3. RoPE (half-split, pos=1 -> 90 deg rotation cos=0, sin=1)
  let rope_x : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0]
  let rope_y = FixedArray::make(4, (0.0 : Float))
  let pos : FixedArray[Int64] = [1]
  let cos_c : FixedArray[Float] = [1.0, 1.0, 0.0, 0.0]
  let sin_c : FixedArray[Float] = [0.0, 0.0, 1.0, 1.0]
  @nn.rotary_embedding_f32(
    rope_x, 0, pos, false, cos_c, sin_c, 2, rope_y, 0, 1, 1, 1, 4, 4, false, false,
  )
  // [x0*0 - x2*1, x1*0 - x3*1, x0*1 + x2*0, x1*1 + x3*0] = [-3, -4, 1, 2]
  assert_true((rope_y[0] + (3.0 : Float)).abs() < (1.0e-6 : Float))
  assert_true((rope_y[1] + (4.0 : Float)).abs() < (1.0e-6 : Float))
  assert_true((rope_y[2] - (1.0 : Float)).abs() < (1.0e-6 : Float))
  assert_true((rope_y[3] - (2.0 : Float)).abs() < (1.0e-6 : Float))

  // 4. GQA (B=1, S=2, num_heads=2, kv_num_heads=1, head_dim=2, past_seq=0)
  let q : FixedArray[Float] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
  let k : FixedArray[Float] = [1.0, 0.0, 0.0, 1.0]
  let v : FixedArray[Float] = [10.0, 20.0, 30.0, 40.0]
  let empty : FixedArray[Float] = []
  let seqlens : FixedArray[Int32] = [1]
  let gqa_out = FixedArray::make(8, (0.0 : Float))
  let pk = FixedArray::make(4, (0.0 : Float))
  let pv = FixedArray::make(4, (0.0 : Float))
  @nn.gqa_f32(
    q,
    0,
    k,
    0,
    v,
    0,
    empty,
    0,
    empty,
    0,
    seqlens,
    gqa_out,
    0,
    pk,
    0,
    pv,
    0,
    1,
    2,
    0,
    2,
    2,
    1,
    2,
    (1.0 : Float),
    -1,
    (0.0 : Float),
  )
  // Token 0 only attends to token 0 -> output equals v[0] = [10, 20] for both heads
  assert_true((gqa_out[0] - (10.0 : Float)).abs() < (1.0e-5 : Float))
  assert_true((gqa_out[1] - (20.0 : Float)).abs() < (1.0e-5 : Float))
  assert_true((gqa_out[2] - (10.0 : Float)).abs() < (1.0e-5 : Float))
  // Token 1 head 0 prefers k[0] (dot=1 vs 0), head 1 prefers k[1] (dot=1 vs 0)
  assert_true(gqa_out[4] < gqa_out[6])
  assert_true((pk[2] - (0.0 : Float)).abs() < (1.0e-6 : Float))
  assert_true((pv[3] - (40.0 : Float)).abs() < (1.0e-6 : Float))
}
