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
