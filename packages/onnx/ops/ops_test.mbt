///|
/// 浮点近似相等断言辅助函数。
fn assert_approx(actual : Float, expected : Float, tol : Float) -> Unit raise {
  let diff = (actual - expected).abs()
  if diff > tol {
    fail("expected \{expected}, got \{actual} (diff=\{diff} > tol=\{tol})")
  }
}

///|
/// 测试同形状与广播 Add/Sub/Mul、Relu（标量与 SIMD 一致性）及 Softmax 稳定性。
test "elementwise, broadcasting, Relu (scalar vs simd), and Softmax" {
  let a = @tensor.Tensor::from_f32_array([2, 3], [
    -1.0, 2.0, -3.0, 4.0, -5.0, 6.0,
  ])
  let b = @tensor.Tensor::from_f32_array([3], [10.0, 20.0, 30.0])
  let added = @ops.add_f32(a, b)
  assert_eq(added.get_f32([0, 0]), (9.0 : Float))
  assert_eq(added.get_f32([1, 2]), (36.0 : Float))

  let r_scalar = @ops.relu_f32(a, use_simd=false).to_f32_buffer()
  let r_simd = @ops.relu_f32(a, use_simd=true).to_f32_buffer()
  for i = 0; i < 6; i = i + 1 {
    assert_eq(r_scalar[i], r_simd[i])
  }

  let big = @tensor.Tensor::from_f32_array([1, 3], [1000.0, 1000.0, 1000.0])
  let p = @ops.softmax_f32(big, -1)
  assert_approx(p.get_f32([0, 0]), (0.33333334 : Float), (1.0e-5 : Float))
}

///|
/// 测试 MatMul / Gemm / Conv2D 在标量与 V128 SIMD 路径下数值完全一致。
test "MatMul, Gemm, Conv2D, MaxPool, AveragePool and BatchNorm" {
  let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = @tensor.Tensor::from_f32_array([3, 2], [
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
  ])
  let mm_s = @ops.matmul_f32(a, b, use_simd=false)
  let mm_v = @ops.matmul_f32(a, b, use_simd=true)
  assert_eq(mm_s.get_f32([0, 0]), (58.0 : Float))
  assert_eq(mm_v.get_f32([1, 1]), (154.0 : Float))

  let x = @tensor.Tensor::from_f32_array([1, 1, 3, 3], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let w = @tensor.Tensor::from_f32_array([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
  let bias = @tensor.Tensor::from_f32_array([1], [0.5])
  let c_s = @ops.conv2d_f32(
    x,
    w,
    Some(bias),
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @nn_ir.NotSet,
    use_simd=false,
  )
  let c_v = @ops.conv2d_f32(
    x,
    w,
    Some(bias),
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @nn_ir.NotSet,
    use_simd=true,
  )
  assert_eq(c_s.get_f32([0, 0, 0, 0]), (12.5 : Float))
  assert_eq(c_v.get_f32([0, 0, 1, 1]), (28.5 : Float))

  let mp = @ops.maxpool_2d_f32(
    x,
    [2, 2],
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    false,
    @nn_ir.NotSet,
  )
  assert_eq(mp.get_f32([0, 0, 1, 1]), (9.0 : Float))
}

///|
/// 验证预打包卷积 `conv2d_prepacked_f32`（含多分组、填充、偏置融合与 BufferPool 复用）与标量参考精确一致。
test "conv2d_prepacked_f32 matches scalar conv2d across grouped, padded, and 1x1 cases with BufferPool" {
  let pool = @tensor.BufferPool::new()

  // 用例 1：多通道 + 5x5 + padding=2 + bias（与 MNIST CNN 特征形态一致）
  let x1 = @tensor.Tensor::from_f32(
    [2, 4, 7, 7],
    FixedArray::makei(2 * 4 * 7 * 7, fn(i) {
      Float::from_int(i % 19 - 9) * (0.125 : Float)
    }),
  )
  let w1 = @tensor.Tensor::from_f32(
    [10, 4, 5, 5],
    FixedArray::makei(10 * 4 * 5 * 5, fn(i) {
      Float::from_int(i % 13 - 6) * (0.0625 : Float)
    }),
  )
  let b1 = @tensor.Tensor::from_f32(
    [10],
    FixedArray::makei(10, fn(i) { Float::from_int(i - 5) * (0.1 : Float) }),
  )
  let ref1 = @ops.conv2d_f32(
    x1,
    w1,
    Some(b1),
    [1, 1],
    [2, 2, 2, 2],
    [1, 1],
    1,
    @nn_ir.NotSet,
    use_simd=false,
  )
  let pw1 = @ops.pack_conv2d_weight_f32(w1, 1)
  assert_eq(pw1.shape(), [10, 4, 5, 5])
  assert_eq(pw1.group(), 1)
  let pre1_a = @ops.conv2d_prepacked_f32(
    x1,
    pw1,
    Some(b1),
    [1, 1],
    [2, 2, 2, 2],
    [1, 1],
    @nn_ir.NotSet,
    scratch_pool=Some(pool),
  )
  let pre1_b = @ops.conv2d_prepacked_f32(
    x1,
    pw1,
    Some(b1),
    [1, 1],
    [2, 2, 2, 2],
    [1, 1],
    @nn_ir.NotSet,
    scratch_pool=Some(pool),
  )
  // 第二次执行时 col_t 临时工作缓冲应直接命中 BufferPool
  assert_true(pool.stats().hits >= 1)
  let r1_buf = ref1.to_f32_buffer()
  let p1a_buf = pre1_a.to_f32_buffer()
  let p1b_buf = pre1_b.to_f32_buffer()
  for i = 0; i < r1_buf.length(); i = i + 1 {
    assert_approx(p1a_buf[i], r1_buf[i], (1.0e-5 : Float))
    assert_eq(p1a_buf[i], p1b_buf[i])
  }

  // 用例 2：Grouped Conv（group=2, stride=2, dilation=1, 无偏置）
  let x2 = @tensor.Tensor::from_f32(
    [1, 6, 8, 8],
    FixedArray::makei(6 * 8 * 8, fn(i) {
      Float::from_int(i % 11 - 5) * (0.25 : Float)
    }),
  )
  let w2 = @tensor.Tensor::from_f32(
    [6, 3, 3, 3],
    FixedArray::makei(6 * 3 * 3 * 3, fn(i) {
      Float::from_int(i % 7 - 3) * (0.125 : Float)
    }),
  )
  let ref2 = @ops.conv2d_f32(
    x2,
    w2,
    None,
    [2, 2],
    [1, 1, 1, 1],
    [1, 1],
    2,
    @nn_ir.NotSet,
    use_simd=false,
  )
  let pw2 = @ops.pack_conv2d_weight_f32(w2, 2)
  let pre2 = @ops.conv2d_prepacked_f32(
    x2,
    pw2,
    None,
    [2, 2],
    [1, 1, 1, 1],
    [1, 1],
    @nn_ir.NotSet,
    scratch_pool=Some(pool),
  )
  let r2_buf = ref2.to_f32_buffer()
  let p2_buf = pre2.to_f32_buffer()
  for i = 0; i < r2_buf.length(); i = i + 1 {
    assert_approx(p2_buf[i], r2_buf[i], (1.0e-5 : Float))
  }
}

///|
/// 验证预打包卷积的寄存器级融合激活（ReLU / ReLU6）与零 im2col 深度可分离卷积（Depthwise Conv2D，含 stride=1 V128 与 stride=2）。
test "conv2d_prepacked_f32 fused Relu/Clip and V128 Depthwise Conv2D match scalar reference" {
  // 1. 常规卷积 + 融合 ReLU / ReLU6（Clip [0, 6]）
  let x = @tensor.Tensor::from_f32(
    [1, 3, 6, 6],
    FixedArray::makei(3 * 6 * 6, fn(i) {
      Float::from_int(i % 23 - 11) * (0.5 : Float)
    }),
  )
  let w = @tensor.Tensor::from_f32(
    [9, 3, 3, 3],
    FixedArray::makei(9 * 3 * 3 * 3, fn(i) {
      Float::from_int(i % 15 - 7) * (0.2 : Float)
    }),
  )
  let b = @tensor.Tensor::from_f32(
    [9],
    FixedArray::makei(9, fn(i) { Float::from_int(i - 4) * (0.5 : Float) }),
  )
  let pw = @ops.pack_conv2d_weight_f32(w, 1)
  assert_true(!pw.is_depthwise())

  let acts = [
    @nn_ir.ReluAct,
    @nn_ir.ClipAct(min=(0.0 : Float), max=(6.0 : Float)),
  ]
  for a_idx = 0; a_idx < acts.length(); a_idx = a_idx + 1 {
    let act = acts[a_idx]
    let ref_out = @ops.conv2d_f32(
      x,
      w,
      Some(b),
      [1, 1],
      [1, 1, 1, 1],
      [1, 1],
      1,
      @nn_ir.NotSet,
      use_simd=false,
      activation=act,
    ).to_f32_buffer()
    let pre_out = @ops.conv2d_prepacked_f32(
      x,
      pw,
      Some(b),
      [1, 1],
      [1, 1, 1, 1],
      [1, 1],
      @nn_ir.NotSet,
      activation=act,
    ).to_f32_buffer()
    for i = 0; i < ref_out.length(); i = i + 1 {
      assert_approx(pre_out[i], ref_out[i], (1.0e-5 : Float))
    }
  }

  // 2. Depthwise Conv2D（group == c_in == c_out == 8）：测试 stride=1 (V128) 与 stride=2 两条路径
  let dw_x = @tensor.Tensor::from_f32(
    [2, 8, 9, 10],
    FixedArray::makei(2 * 8 * 9 * 10, fn(i) {
      Float::from_int(i % 19 - 9) * (0.25 : Float)
    }),
  )
  let dw_w = @tensor.Tensor::from_f32(
    [8, 1, 3, 3],
    FixedArray::makei(8 * 9, fn(i) {
      Float::from_int(i % 11 - 5) * (0.2 : Float)
    }),
  )
  let dw_b = @tensor.Tensor::from_f32(
    [8],
    FixedArray::makei(8, fn(i) { Float::from_int(i - 3) * (0.3 : Float) }),
  )
  let dw_pw = @ops.pack_conv2d_weight_f32(dw_w, 8)
  assert_true(dw_pw.is_depthwise())

  let stride_cases = [[1, 1], [2, 2]]
  for s_idx = 0; s_idx < stride_cases.length(); s_idx = s_idx + 1 {
    let st = stride_cases[s_idx]
    let act = @nn_ir.ClipAct(min=(0.0 : Float), max=(6.0 : Float))
    let ref_dw = @ops.conv2d_f32(
      dw_x,
      dw_w,
      Some(dw_b),
      st,
      [1, 1, 1, 1],
      [1, 1],
      8,
      @nn_ir.NotSet,
      use_simd=false,
      activation=act,
    ).to_f32_buffer()
    let fast_dw = @ops.conv2d_prepacked_f32(
      dw_x,
      dw_pw,
      Some(dw_b),
      st,
      [1, 1, 1, 1],
      [1, 1],
      @nn_ir.NotSet,
      activation=act,
    ).to_f32_buffer()
    for i = 0; i < ref_dw.length(); i = i + 1 {
      assert_approx(fast_dw[i], ref_dw[i], (1.0e-5 : Float))
    }
  }
}

///|
/// 验证 Tier-2a 逐元素数学与激活算子（Div / Pow / Sqrt / Exp / Neg / Clip / Sigmoid / Tanh / Gelu / HardSwish / LeakyRelu）
/// 的标量与 V128 SIMD 路径数值一致性及广播正确性。
test "Tier-2a elementwise math and activations match scalar and SIMD" {
  let n = 13
  let x = @tensor.Tensor::from_f32(
    [n],
    FixedArray::makei(n, fn(i) { Float::from_int(i - 6) * (0.6 : Float) }),
  )
  let pos = @tensor.Tensor::from_f32(
    [n],
    FixedArray::makei(n, fn(i) { Float::from_int(i + 1) * (0.5 : Float) }),
  )
  let two = @tensor.Tensor::from_f32([1], FixedArray::make(1, (2.0 : Float)))

  // Div（同形 SIMD 与广播标量）
  let div_sc = @ops.div_f32(x, pos, use_simd=false).to_f32_buffer()
  let div_si = @ops.div_f32(x, pos, use_simd=true).to_f32_buffer()
  for i = 0; i < n; i = i + 1 {
    assert_approx(div_si[i], div_sc[i], (1.0e-6 : Float))
  }
  let pow_out = @ops.pow_f32(pos, two).to_f32_buffer()
  for i = 0; i < n; i = i + 1 {
    let v = pos.get_f32_flat(i)
    assert_approx(pow_out[i], v * v, (1.0e-5 : Float))
  }

  // Sqrt / Exp / Neg
  let sqrt_sc = @ops.sqrt_f32(pos, use_simd=false).to_f32_buffer()
  let sqrt_si = @ops.sqrt_f32(pos, use_simd=true).to_f32_buffer()
  let exp_sc = @ops.exp_f32(x, use_simd=false).to_f32_buffer()
  let exp_si = @ops.exp_f32(x, use_simd=true).to_f32_buffer()
  let neg_out = @ops.neg_f32(x).to_f32_buffer()
  for i = 0; i < n; i = i + 1 {
    assert_approx(sqrt_si[i], sqrt_sc[i], (1.0e-6 : Float))
    assert_approx(exp_si[i], exp_sc[i], (1.0e-4 : Float))
    assert_eq(neg_out[i], -x.get_f32_flat(i))
  }

  // Clip / Sigmoid / Tanh / Gelu / HardSwish / LeakyRelu
  let clip_sc = @ops.clip_f32(
    x,
    Some((-1.0 : Float)),
    Some((2.0 : Float)),
    use_simd=false,
  ).to_f32_buffer()
  let clip_si = @ops.clip_f32(
    x,
    Some((-1.0 : Float)),
    Some((2.0 : Float)),
    use_simd=true,
  ).to_f32_buffer()
  let sig_sc = @ops.sigmoid_f32(x, use_simd=false).to_f32_buffer()
  let sig_si = @ops.sigmoid_f32(x, use_simd=true).to_f32_buffer()
  let tanh_sc = @ops.tanh_f32(x, use_simd=false).to_f32_buffer()
  let tanh_si = @ops.tanh_f32(x, use_simd=true).to_f32_buffer()
  let gelu_sc = @ops.gelu_f32(x, use_simd=false).to_f32_buffer()
  let gelu_si = @ops.gelu_f32(x, use_simd=true).to_f32_buffer()
  let hs_sc = @ops.hard_swish_f32(x, use_simd=false).to_f32_buffer()
  let hs_si = @ops.hard_swish_f32(x, use_simd=true).to_f32_buffer()
  let lr_out = @ops.leaky_relu_f32(x, alpha=(0.1 : Float)).to_f32_buffer()

  for i = 0; i < n; i = i + 1 {
    assert_eq(clip_si[i], clip_sc[i])
    assert_approx(sig_si[i], sig_sc[i], (2.0e-4 : Float))
    assert_approx(tanh_si[i], tanh_sc[i], (2.0e-4 : Float))
    assert_approx(gelu_si[i], gelu_sc[i], (5.0e-4 : Float))
    assert_approx(hs_si[i], hs_sc[i], (1.0e-5 : Float))
    let v = x.get_f32_flat(i)
    let expected_lr = if v >= (0.0 : Float) { v } else { v * (0.1 : Float) }
    assert_approx(lr_out[i], expected_lr, (1.0e-6 : Float))
  }
}

///|
/// 验证 Tier-2b 张量变形、索引与填充算子（Shape / Cast / Concat / Split / Slice / Pad / Gather）。
test "Tier-2b tensor manipulation ops (Shape/Cast/Concat/Split/Slice/Pad/Gather)" {
  // 1. Shape（含 start/end 切片）
  let x = @tensor.Tensor::from_f32_array([2, 3, 4], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
  ])
  let sh_full = @ops.shape_op(x)
  assert_eq(sh_full.shape(), [3])
  assert_eq(sh_full.to_i64_buffer(), [2, 3, 4])
  let sh_sub = @ops.shape_op(x, start=Some(1), end=Some(-1))
  assert_eq(sh_sub.to_i64_buffer(), [3])

  // 2. Cast（F32 -> I64 -> F32 与 Bool）
  let c_i64 = @ops.cast_op(
    @tensor.Tensor::from_f32_array([3], [0.0, 2.7, -3.2]),
    @tensor.Int64,
  )
  assert_eq(c_i64.to_i64_buffer(), [0, 2, -3])
  let c_f32 = @ops.cast_op(c_i64, @tensor.Float32)
  assert_eq(c_f32.to_f32_buffer(), [0.0, 2.0, -3.0])
  let c_bool = @ops.cast_op(c_i64, @tensor.Bool)
  let c_back = @ops.cast_op(c_bool, @tensor.Int32)
  assert_eq(c_back.to_i64_buffer(), [0, 1, 1])

  // 3. Split + Concat 往返一致
  let parts = @ops.split_op(x, 1, Some([1, 2]), None, 2)
  assert_eq(parts.length(), 2)
  assert_eq(parts[0].shape(), [2, 1, 4])
  assert_eq(parts[1].shape(), [2, 2, 4])
  let rejoined = @ops.concat_op(parts, 1)
  assert_eq(rejoined.shape(), [2, 3, 4])
  assert_eq(rejoined.to_f32_buffer(), x.to_f32_buffer())

  // 4. Slice（含负索引与步长 2）
  let sl = @ops.slice_op(x, [0, 1], [2, 4], Some([1, 2]), Some([1, 2]))
  assert_eq(sl.shape(), [2, 2, 2])
  // [0, 0, 1] -> 2.0, [0, 0, 3] -> 4.0, [0, 1, 1] -> 6.0, [0, 1, 3] -> 8.0
  assert_eq(sl.get_f32([0, 0, 0]), (2.0 : Float))
  assert_eq(sl.get_f32([0, 0, 1]), (4.0 : Float))
  assert_eq(sl.get_f32([0, 1, 0]), (6.0 : Float))
  assert_eq(sl.get_f32([0, 1, 1]), (8.0 : Float))

  // 5. Pad（constant / reflect / edge）
  let p_in = @tensor.Tensor::from_f32_array([1, 3], [1.0, 2.0, 3.0])
  let p_const = @ops.pad_f32(
    p_in,
    [0, 1, 0, 1],
    "constant",
    (-1.0 : Float),
    None,
  )
  assert_eq(p_const.shape(), [1, 5])
  assert_eq(p_const.to_f32_buffer(), [-1.0, 1.0, 2.0, 3.0, -1.0])
  let p_refl = @ops.pad_f32(p_in, [0, 1, 0, 1], "reflect", (0.0 : Float), None)
  assert_eq(p_refl.to_f32_buffer(), [2.0, 1.0, 2.0, 3.0, 2.0])
  let p_edge = @ops.pad_f32(p_in, [0, 1, 0, 1], "edge", (0.0 : Float), None)
  assert_eq(p_edge.to_f32_buffer(), [1.0, 1.0, 2.0, 3.0, 3.0])

  // 6. Gather（嵌入查表与负索引）
  let emb = @tensor.Tensor::from_f32_array([3, 2], [
    10.0, 11.0, 20.0, 21.0, 30.0, 31.0,
  ])
  let idx = @tensor.Tensor::from_i64([2], [2, -3])
  let gathered = @ops.gather_op(emb, idx, 0)
  assert_eq(gathered.shape(), [2, 2])
  assert_eq(gathered.to_f32_buffer(), [30.0, 31.0, 10.0, 11.0])
}

///|
/// 验证 Tier-2b 归约与归一化算子（GlobalAveragePool / ReduceMean / ReduceSum / LayerNormalization）。
test "Tier-2b reduction and normalization ops (GlobalAvgPool/ReduceMean/ReduceSum/LayerNorm)" {
  // 1. GlobalAveragePool
  let feat = @tensor.Tensor::from_f32_array([1, 2, 2, 2], [
    1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0,
  ])
  let gap = @ops.global_avgpool_f32(feat)
  assert_eq(gap.shape(), [1, 2, 1, 1])
  assert_approx(gap.get_f32([0, 0, 0, 0]), (2.5 : Float), (1.0e-6 : Float))
  assert_approx(gap.get_f32([0, 1, 0, 0]), (25.0 : Float), (1.0e-6 : Float))

  // 2. ReduceMean & ReduceSum
  let m = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let r_mean = @ops.reduce_mean_f32(m, Some([1]), None, keepdims=false)
  assert_eq(r_mean.shape(), [2])
  assert_approx(r_mean.get_f32([0]), (2.0 : Float), (1.0e-6 : Float))
  assert_approx(r_mean.get_f32([1]), (5.0 : Float), (1.0e-6 : Float))
  let r_sum = @ops.reduce_sum_f32(m, Some([0]), None, keepdims=true)
  assert_eq(r_sum.shape(), [1, 3])
  assert_eq(r_sum.to_f32_buffer(), [5.0, 7.0, 9.0])

  // 3. LayerNormalization（[1, 4] 均值 2.5，方差 1.25，std = sqrt(1.25)）
  let ln_in = @tensor.Tensor::from_f32_array([1, 4], [1.0, 2.0, 3.0, 4.0])
  let ln_s = @tensor.Tensor::from_f32_array([4], [2.0, 2.0, 2.0, 2.0])
  let ln_b = @tensor.Tensor::from_f32_array([4], [0.5, 0.5, 0.5, 0.5])
  let ln_out = @ops.layer_norm_f32(
    ln_in,
    ln_s,
    Some(ln_b),
    axis=-1,
    epsilon=(0.0 : Float),
  )
  let inv_std = 1.0 / 1.25.sqrt()
  let expected0 = Float::from_double((1.0 - 2.5) * inv_std * 2.0 + 0.5)
  let expected3 = Float::from_double((4.0 - 2.5) * inv_std * 2.0 + 0.5)
  assert_approx(ln_out.get_f32([0, 0]), expected0, (1.0e-5 : Float))
  assert_approx(ln_out.get_f32([0, 3]), expected3, (1.0e-5 : Float))
}

///|
/// 验证原生 1D 卷积（普通 Conv1D 与 Depthwise Conv1D，含融合 ReLU/Clip）的标量与预打包 V128 SIMD 一致性。
test "Conv1D (standard & depthwise) prepacked V128 matches scalar reference" {
  let x = @tensor.Tensor::from_f32(
    [2, 4, 15],
    FixedArray::makei(2 * 4 * 15, fn(i) {
      Float::from_int(i % 13 - 6) * (0.25 : Float)
    }),
  )
  let w = @tensor.Tensor::from_f32(
    [6, 4, 3],
    FixedArray::makei(6 * 4 * 3, fn(i) {
      Float::from_int(i % 9 - 4) * (0.2 : Float)
    }),
  )
  let b = @tensor.Tensor::from_f32_array([6], [0.1, -0.2, 0.3, 0.0, -0.1, 0.2])

  let pw = @ops.pack_conv2d_weight_f32(w, 1)
  let ref_y = @ops.conv2d_f32(
    x,
    w,
    Some(b),
    [2],
    [1, 1],
    [1],
    1,
    @nn_ir.NotSet,
    use_simd=false,
    activation=@nn_ir.ReluAct,
  )
  let pre_y = @ops.conv2d_prepacked_f32(
    x,
    pw,
    Some(b),
    [2],
    [1, 1],
    [1],
    @nn_ir.NotSet,
    activation=@nn_ir.ReluAct,
  )
  assert_eq(pre_y.shape(), [2, 6, 8])
  let r_buf = ref_y.to_f32_buffer()
  let p_buf = pre_y.to_f32_buffer()
  for i = 0; i < r_buf.length(); i = i + 1 {
    assert_approx(p_buf[i], r_buf[i], (1.0e-5 : Float))
  }

  // Depthwise Conv1D (group == c_in == c_out == 4)
  let dw_w = @tensor.Tensor::from_f32(
    [4, 1, 3],
    FixedArray::makei(12, fn(i) { Float::from_int(i - 5) * (0.15 : Float) }),
  )
  let dw_pw = @ops.pack_conv2d_weight_f32(dw_w, 4)
  assert_true(dw_pw.is_depthwise())
  let dw_ref = @ops.conv2d_f32(
    x,
    dw_w,
    None,
    [1],
    [1, 1],
    [1],
    4,
    @nn_ir.NotSet,
    use_simd=false,
    activation=@nn_ir.ClipAct(min=(0.0 : Float), max=(6.0 : Float)),
  ).to_f32_buffer()
  let dw_fast = @ops.conv2d_prepacked_f32(
    x,
    dw_pw,
    None,
    [1],
    [1, 1],
    [1],
    @nn_ir.NotSet,
    activation=@nn_ir.ClipAct(min=(0.0 : Float), max=(6.0 : Float)),
  ).to_f32_buffer()
  for i = 0; i < dw_ref.length(); i = i + 1 {
    assert_approx(dw_fast[i], dw_ref[i], (1.0e-5 : Float))
  }
}

///|
/// 验证 ONNX `Resize` 算子：4D Nearest 2× 上采样、4D Bilinear（`align_corners`）与 3D 线性插值。
test "Resize supports 4D nearest/bilinear and 3D linear interpolation" {
  let img = @tensor.Tensor::from_f32_array([1, 1, 2, 2], [1.0, 3.0, 5.0, 7.0])
  let scales_2x = @tensor.Tensor::from_f32_array([4], [1.0, 1.0, 2.0, 2.0])

  // 1. Nearest 2x（asymmetric + floor：每个像素复制为 2x2 块）
  let near = @ops.resize_f32(
    img,
    Some(scales_2x),
    None,
    mode="nearest",
    coordinate_transformation_mode="asymmetric",
    nearest_mode="floor",
  )
  assert_eq(near.shape(), [1, 1, 4, 4])
  assert_eq(near.get_f32([0, 0, 0, 0]), (1.0 : Float))
  assert_eq(near.get_f32([0, 0, 0, 1]), (1.0 : Float))
  assert_eq(near.get_f32([0, 0, 0, 2]), (3.0 : Float))
  assert_eq(near.get_f32([0, 0, 3, 3]), (7.0 : Float))

  // 2. Bilinear align_corners [1,1,2,2] -> [1,1,3,3]（中点精确为四角均值 4.0）
  let target_sz = @tensor.Tensor::from_i64([4], [1, 1, 3, 3])
  let bilin = @ops.resize_f32(
    img,
    None,
    Some(target_sz),
    mode="linear",
    coordinate_transformation_mode="align_corners",
  )
  assert_eq(bilin.shape(), [1, 1, 3, 3])
  assert_approx(bilin.get_f32([0, 0, 0, 0]), (1.0 : Float), (1.0e-5 : Float))
  assert_approx(bilin.get_f32([0, 0, 0, 1]), (2.0 : Float), (1.0e-5 : Float))
  assert_approx(bilin.get_f32([0, 0, 1, 1]), (4.0 : Float), (1.0e-5 : Float))
  assert_approx(bilin.get_f32([0, 0, 2, 2]), (7.0 : Float), (1.0e-5 : Float))

  // 3. 3D Linear align_corners [1,1,2] -> [1,1,3]
  let sig = @tensor.Tensor::from_f32_array([1, 1, 2], [2.0, 6.0])
  let sig_sz = @tensor.Tensor::from_i64([3], [1, 1, 3])
  let sig_out = @ops.resize_f32(
    sig,
    None,
    Some(sig_sz),
    mode="linear",
    coordinate_transformation_mode="align_corners",
  )
  assert_eq(sig_out.shape(), [1, 1, 3])
  assert_approx(sig_out.get_f32([0, 0, 1]), (4.0 : Float), (1.0e-5 : Float))
}

///|
/// 验证 ONNX `LSTM` 算子（门序 `i, o, f, c`）：单向解析解对齐、初始状态传递与双向输出形状。
test "LSTM matches analytical single-step reference and supports bidirectional & layout=1" {
  // 单步、H=1、input_size=2、W=全 1、R=全 0、B=全 0、x=[1.0, 0.0]
  // 所有 4 门预激活均为 1.0 => i=o=f=sigmoid(1.0), c_tilde=tanh(1.0)
  // c_1 = i * c_tilde, h_1 = o * tanh(c_1)
  let x = @tensor.Tensor::from_f32_array([1, 1, 2], [1.0, 0.0])
  let w = @tensor.Tensor::full_f32([1, 4, 2], (1.0 : Float))
  let r = @tensor.Tensor::zeros_f32([1, 4, 1])
  let outs = @ops.lstm_f32(
    x,
    w,
    r,
    None,
    None,
    None,
    None,
    None,
    hidden_size=Some(1),
  )
  assert_eq(outs.length(), 3)
  assert_eq(outs[0].shape(), [1, 1, 1, 1]) // Y
  assert_eq(outs[1].shape(), [1, 1, 1]) // Y_h
  assert_eq(outs[2].shape(), [1, 1, 1]) // Y_c

  let sig1 = 1.0 / (1.0 + @math.exp(-1.0))
  let tanh1 = @math.tanh(1.0)
  let exp_c = sig1 * tanh1
  let exp_h = sig1 * @math.tanh(exp_c)
  assert_approx(
    outs[2].get_f32([0, 0, 0]),
    Float::from_double(exp_c),
    (1.0e-5 : Float),
  )
  assert_approx(
    outs[1].get_f32([0, 0, 0]),
    Float::from_double(exp_h),
    (1.0e-5 : Float),
  )
  assert_approx(
    outs[0].get_f32([0, 0, 0, 0]),
    Float::from_double(exp_h),
    (1.0e-5 : Float),
  )

  // 双向 + layout=1（batch-first: [B=2, S=4, D=3]）
  let xb = @tensor.Tensor::full_f32([2, 4, 3], (0.5 : Float))
  let wb = @tensor.Tensor::full_f32([2, 8, 3], (0.2 : Float))
  let rb = @tensor.Tensor::full_f32([2, 8, 2], (0.1 : Float))
  let b_outs = @ops.lstm_f32(
    xb,
    wb,
    rb,
    None,
    None,
    None,
    None,
    None,
    hidden_size=Some(2),
    direction="bidirectional",
    layout=1,
  )
  assert_eq(b_outs[0].shape(), [2, 4, 2, 2]) // [B, S, num_dirs, H]
  assert_eq(b_outs[1].shape(), [2, 2, 2]) // [B, num_dirs, H]
  assert_eq(b_outs[2].shape(), [2, 2, 2])
}

///|
/// 验证 `Equal`（含标量广播与多类型支持）及 `ConstantOfShape`（`Int64` / `Float32` 填充）。
test "Equal and ConstantOfShape match ONNX semantics" {
  fn read_bools(t : @tensor.Tensor) -> FixedArray[Bool] {
    let c = t.contiguous() catch { _ => abort("contiguous failed") }
    match c.data {
      @tensor.Bool(buf) => buf
      _ => abort("expected Bool tensor")
    }
  }

  let sr = @tensor.Tensor::from_i64([], FixedArray::make(1, (16000 : Int64)))
  let sr_16k = @tensor.Tensor::from_i64(
    [],
    FixedArray::make(1, (16000 : Int64)),
  )
  let sr_8k = @tensor.Tensor::from_i64([], FixedArray::make(1, (8000 : Int64)))
  let eq_t = @ops.equal_op(sr, sr_16k)
  let eq_f = @ops.equal_op(sr, sr_8k)
  assert_eq(eq_t.dtype(), @tensor.Bool)
  assert_eq(eq_t.shape(), [])
  assert_eq(read_bools(eq_t)[0], true)
  assert_eq(read_bools(eq_f)[0], false)

  // 广播比较：[2, 3] vs [3]
  let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 1.0, 5.0, 3.0])
  let b = @tensor.Tensor::from_f32_array([3], [1.0, 2.0, 3.0])
  let eq_b = @ops.equal_op(a, b)
  assert_eq(eq_b.shape(), [2, 3])
  let bb = read_bools(eq_b)
  assert_eq(bb[0], true)
  assert_eq(bb[1], true)
  assert_eq(bb[2], true)
  assert_eq(bb[3], true)
  assert_eq(bb[4], false)
  assert_eq(bb[5], true)

  // ConstantOfShape：构造 [2, 3] 的 Int64 常量 7 张量
  let sh = @tensor.Tensor::from_i64([2], [2, 3])
  let fill_i64 = @tensor.Tensor::from_i64([1], [7])
  let c_i64 = @ops.constant_of_shape_op(sh, fill_i64)
  assert_eq(c_i64.dtype(), @tensor.Int64)
  assert_eq(c_i64.shape(), [2, 3])
  let cb = c_i64.to_i64_buffer()
  assert_eq(cb.length(), 6)
  assert_eq(cb[0], (7 : Int64))
  assert_eq(cb[5], (7 : Int64))
}

///|
/// 验证 Transformer/NLP 核心算子套件：`Erf`、关系比较、逻辑 `Not`/`And`、`Where`、`Range`、`Expand`、`Tile` 及 3D 预打包 `MatMul`。
test "Transformer/NLP ops match ONNX semantics" {
  fn read_bools(t : @tensor.Tensor) -> FixedArray[Bool] {
    let c = t.contiguous() catch { _ => abort("contiguous failed") }
    match c.data {
      @tensor.Bool(buf) => buf
      _ => abort("expected Bool tensor")
    }
  }

  // 1. Erf（高斯误差函数，校验对称性与经典真值）
  let ex = @tensor.Tensor::from_f32_array([4], [0.0, 1.0, -1.0, 2.0])
  let ey = @ops.erf_f32(ex)
  assert_approx(ey.get_f32([0]), (0.0 : Float), (1.0e-6 : Float))
  assert_approx(ey.get_f32([1]), (0.8427008 : Float), (1.0e-5 : Float))
  assert_approx(ey.get_f32([2]), (-0.8427008 : Float), (1.0e-5 : Float))
  assert_approx(ey.get_f32([3]), (0.9953223 : Float), (1.0e-5 : Float))

  // 2. 比较与逻辑算子 + Where（含广播）
  let a = @tensor.Tensor::from_f32_array([3], [1.0, 2.0, 3.0])
  let b = @tensor.Tensor::from_f32_array([1], [2.0])
  let lt = @ops.less_op(a, b)
  let gt = @ops.greater_op(a, b)
  let le = @ops.less_or_equal_op(a, b)
  let ge = @ops.greater_or_equal_op(a, b)
  assert_eq(read_bools(lt), [true, false, false])
  assert_eq(read_bools(gt), [false, false, true])
  assert_eq(read_bools(le), [true, true, false])
  assert_eq(read_bools(ge), [false, true, true])

  let not_lt = @ops.not_op(lt)
  assert_eq(read_bools(not_lt), [false, true, true])
  let and_r = @ops.and_op(le, ge)
  assert_eq(read_bools(and_r), [false, true, false])

  let neg_inf = @tensor.Tensor::from_f32_array([1], [-10000.0])
  let w_out = @ops.where_op(le, a, neg_inf)
  assert_approx(w_out.get_f32([0]), (1.0 : Float), (1.0e-5 : Float))
  assert_approx(w_out.get_f32([1]), (2.0 : Float), (1.0e-5 : Float))
  assert_approx(w_out.get_f32([2]), (-10000.0 : Float), (1.0e-5 : Float))

  // 3. Range（Int64 与 Float32，含负步长）
  let r_i64 = @ops.range_op(
    @tensor.Tensor::from_i64([], [0]),
    @tensor.Tensor::from_i64([], [5]),
    @tensor.Tensor::from_i64([], [2]),
  )
  assert_eq(r_i64.shape(), [3])
  assert_eq(r_i64.to_i64_buffer(), [(0 : Int64), 2, 4])

  let r_f32 = @ops.range_op(
    @tensor.Tensor::from_f32_array([], [3.0]),
    @tensor.Tensor::from_f32_array([], [0.0]),
    @tensor.Tensor::from_f32_array([], [-1.5]),
  )
  assert_eq(r_f32.shape(), [2])
  assert_approx(r_f32.get_f32([0]), (3.0 : Float), (1.0e-5 : Float))
  assert_approx(r_f32.get_f32([1]), (1.5 : Float), (1.0e-5 : Float))

  // 4. Expand 与 Tile
  let base = @tensor.Tensor::from_f32_array([1, 3], [10.0, 20.0, 30.0])
  let exp_t = @ops.expand_op(base, @tensor.Tensor::from_i64([2], [2, 3]))
  assert_eq(exp_t.shape(), [2, 3])
  assert_approx(exp_t.get_f32([1, 2]), (30.0 : Float), (1.0e-5 : Float))

  let tiled = @ops.tile_op(base, @tensor.Tensor::from_i64([2], [2, 2]))
  assert_eq(tiled.shape(), [2, 6])
  assert_approx(tiled.get_f32([1, 4]), (20.0 : Float), (1.0e-5 : Float))

  // 5. 3D 预打包 MatMul（[B=2, S=3, K=4] @ [K=4, N=5]）对齐通用 matmul_f32
  let ma = @tensor.Tensor::from_f32_array(
    [2, 3, 4],
    Array::makei(24, fn(i) { Float::from_int(i - 10) * (0.25 : Float) }),
  )
  let mb = @tensor.Tensor::from_f32_array(
    [4, 5],
    Array::makei(20, fn(i) { Float::from_int(i + 1) * (0.1 : Float) }),
  )
  let packed_b = @gemm.pack_matrix_b_f32(mb.to_f32_buffer(), 0, 4, 5)
  let ref_mm = @ops.matmul_f32(ma, mb)
  let simd_mm = @ops.matmul_prepacked_f32(ma, packed_b)
  assert_eq(simd_mm.shape(), [2, 3, 5])
  for i = 0; i < 30; i = i + 1 {
    assert_approx(
      simd_mm.get_f32_flat(i),
      ref_mm.get_f32_flat(i),
      (1.0e-5 : Float),
    )
  }
}

///|
/// 验证 `DynamicQuantizeLinear`、3D `MatMulInteger`（V128 预打包 vs 标量参考）以及 `UInt8`/`Int8` `Gather`。
test "DynamicQuantizeLinear, MatMulInteger, and byte Gather match ONNX semantics" {
  // 1. UInt8 词表 Gather + DequantizeLinear
  let table = @tensor.Tensor::from_u8([3, 2], [
    (10).to_byte(),
    (20).to_byte(),
    (30).to_byte(),
    (40).to_byte(),
    (50).to_byte(),
    (60).to_byte(),
  ])
  let ids = @tensor.Tensor::from_i64([1, 2], [2, 0])
  let gathered = @ops.gather_op(table, ids, 0)
  assert_eq(gathered.dtype(), @tensor.UInt8)
  assert_eq(gathered.shape(), [1, 2, 2])
  let gb = gathered.to_u8_buffer()
  assert_eq(gb[0].to_int(), 50)
  assert_eq(gb[1].to_int(), 60)
  assert_eq(gb[2].to_int(), 10)
  assert_eq(gb[3].to_int(), 20)

  // 2. DynamicQuantizeLinear + 3D MatMulInteger ([1, 3, 19] @ [19, 7])
  let x = @tensor.Tensor::from_f32_array(
    [1, 3, 19],
    Array::makei(57, fn(i) { Float::from_int(i - 28) * (0.1 : Float) }),
  )
  let (xq, x_sc, x_zp) = @ops.dynamic_quantize_linear_op(x)
  assert_eq(xq.dtype(), @tensor.UInt8)
  assert_eq(xq.shape(), [1, 3, 19])
  assert_eq(x_sc.shape(), [])
  assert_eq(x_zp.shape(), [])

  let w = @tensor.Tensor::from_i8(
    [19, 7],
    FixedArray::makei(133, fn(i) { @quant.i8_to_byte(i * 11 % 160 - 80) }),
  )
  let b_zp = @tensor.Tensor::from_i8([7], FixedArray::make(7, b'\x00'))
  let y_simd = @ops.matmul_integer_op(
    xq,
    w,
    Some(x_zp),
    Some(b_zp),
    use_simd=true,
  )
  let y_scalar = @ops.matmul_integer_op(
    xq,
    w,
    Some(x_zp),
    Some(b_zp),
    use_simd=false,
  )
  assert_eq(y_simd.dtype(), @tensor.Int32)
  assert_eq(y_simd.shape(), [1, 3, 7])
  let c1 = @ops.cast_op(y_simd, @tensor.Float32)
  let c2 = @ops.cast_op(y_scalar, @tensor.Float32)
  for i = 0; i < 21; i = i + 1 {
    assert_eq(c1.get_f32_flat(i), c2.get_f32_flat(i))
  }
}
