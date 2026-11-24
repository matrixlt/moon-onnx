///|
/// 构造纯内存微型 CNN 计算图（`Conv -> Relu -> MaxPool -> Flatten -> MatMul -> Add`），
/// 用于在无外部文件依赖下完整验证 Scalar/SIMD 对齐、静态权重预打包、`RunProfile` 与 `BufferPool` 100% 复用。
fn build_mini_cnn_graph() -> @nn_ir.IrGraph raise {
  let conv_w = @tensor.Tensor::from_f32(
    [8, 2, 3, 3],
    FixedArray::makei(8 * 2 * 3 * 3, fn(i) {
      Float::from_int(i % 13 - 6) * (0.0625 : Float)
    }),
  )
  let conv_b = @tensor.Tensor::from_f32(
    [8],
    FixedArray::makei(8, fn(i) { Float::from_int(i - 4) * (0.1 : Float) }),
  )
  // 经过 [1, 2, 6, 6] -> Conv(pad=1, 8ch) -> [1, 8, 6, 6] -> MaxPool(2x2, s=2) -> [1, 8, 3, 3] -> Flatten [1, 72]
  let fc_w = @tensor.Tensor::from_f32(
    [72, 4],
    FixedArray::makei(72 * 4, fn(i) {
      Float::from_int(i % 17 - 8) * (0.03125 : Float)
    }),
  )
  let fc_b = @tensor.Tensor::from_f32_array([4], [0.1, -0.2, 0.3, -0.1])
  {
    name: "mini_cnn_in_memory",
    inputs: ["x"],
    outputs: ["logits"],
    initializers: Map([
      ("conv.w", conv_w),
      ("conv.b", conv_b),
      ("fc.w", fc_w),
      ("fc.b", fc_b),
    ]),
    nodes: [
      {
        name: "conv1",
        op: @nn_ir.Conv(
          kernel_shape=Some([3, 3]),
          strides=[1, 1],
          pads=[1, 1, 1, 1],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["x", "conv.w", "conv.b"],
        outputs: ["c1"],
      },
      { name: "relu1", op: @nn_ir.Relu, inputs: ["c1"], outputs: ["r1"], },
      {
        name: "pool1",
        op: @nn_ir.MaxPool(
          kernel_shape=[2, 2],
          strides=[2, 2],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          ceil_mode=false,
          auto_pad=@nn_ir.NotSet,
        ),
        inputs: ["r1"],
        outputs: ["p1"],
      },
      {
        name: "flat1",
        op: @nn_ir.Flatten(axis=1),
        inputs: ["p1"],
        outputs: ["f1"],
      },
      {
        name: "matmul1",
        op: @nn_ir.MatMul,
        inputs: ["f1", "fc.w"],
        outputs: ["mm"],
      },
      {
        name: "add1",
        op: @nn_ir.Add,
        inputs: ["mm", "fc.b"],
        outputs: ["logits"],
      },
    ],
  }
}

///|
/// 验证微型 CNN 在 Scalar 与 V128 SIMD 后端数值一致、自动预打包 Conv/MatMul 权重，
/// 并在连续推理中实现 `BufferPool` 100% 复用且不污染历史输出张量。
test "Session mini-CNN matches Scalar vs Simd, prepacks Conv/MatMul weights, and reuses BufferPool" {
  let graph = build_mini_cnn_graph()
  let sess_scalar = @session.Session::from_graph(graph, backend=@session.Scalar)
  let sess_simd = @session.Session::from_graph(graph, backend=@session.Simd)

  // 包含 1 个 Conv 静态权重 + 1 个 MatMul 静态权重
  assert_eq(sess_simd.packed_weight_count(), 2)
  assert_eq(sess_simd.packed_conv_weight_count(), 1)
  assert_eq(sess_scalar.packed_weight_count(), 0)

  let inp = @tensor.Tensor::from_f32(
    [1, 2, 6, 6],
    FixedArray::makei(2 * 6 * 6, fn(i) {
      Float::from_int(i % 19 - 9) * (0.125 : Float)
    }),
  )
  let ref_out = sess_scalar.run_single(inp)
  let (out1, profile) = sess_simd.run_single_profiled(inp)
  assert_eq(out1.shape(), [1, 4])
  assert_eq(profile.ops.length(), 6)
  assert_true(profile.total_us >= 0.0)
  assert_true(profile.format_table().contains("Conv"))
  assert_true(@session.max_abs_diff_f32(out1, ref_out) <= (1.0e-5 : Float))

  let stats_after_1 = sess_simd.pool_stats()
  assert_true(stats_after_1.misses > 0)
  assert_true(stats_after_1.pooled_f32_buffers > 0)

  // 第 2 次前向推理：全部中间张量与 im2col 工作区应 100% 命中 BufferPool（零新增 miss）
  let out2 = sess_simd.run_single(inp)
  let stats_after_2 = sess_simd.pool_stats()
  assert_eq(stats_after_2.misses, stats_after_1.misses)
  assert_true(stats_after_2.hits > stats_after_1.hits)

  // 用全零输入执行第 3 次推理，验证 out1/out2 结果未被后续池复用篡改
  let zero_in = @tensor.Tensor::zeros_f32([1, 2, 6, 6])
  let _out3 = sess_simd.run_single(zero_in)
  assert_true(@session.max_abs_diff_f32(out1, ref_out) <= (1.0e-5 : Float))
  assert_true(@session.max_abs_diff_f32(out2, ref_out) <= (1.0e-5 : Float))

  sess_simd.clear_pool()
  assert_eq(sess_simd.pool_stats().pooled_f32_buffers, 0)
}

///|
/// 验证 `Gemm(trans_b=true)` 静态权重预打包与零拷贝视图作为图输出时的 BufferPool 安全锁定。
test "Session prepacks Gemm(trans_b=true) weights and pins view-aliased graph outputs" {
  let w = @tensor.Tensor::from_f32_array([3, 4], [
    1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5, 0.25, -1.5, 1.0, 3.0,
  ])
  let bias = @tensor.Tensor::from_f32_array([3], [0.5, -0.25, 1.0])
  let graph : @nn_ir.IrGraph = {
    name: "gemm_transb_view_test",
    inputs: ["x"],
    outputs: ["y_view"],
    initializers: Map([("w", w), ("b", bias)]),
    nodes: [
      {
        name: "fc",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["x", "w", "b"],
        outputs: ["h"],
      },
      { name: "act", op: @nn_ir.Relu, inputs: ["h"], outputs: ["r"], },
      {
        name: "view",
        op: @nn_ir.Unsqueeze(axes=Some([0])),
        inputs: ["r"],
        outputs: ["y_view"],
      },
    ],
  }
  let sess_scalar = @session.Session::from_graph(graph, backend=@session.Scalar)
  let sess_simd = @session.Session::from_graph(graph, backend=@session.Simd)
  assert_eq(sess_simd.packed_weight_count(), 1)

  let x1 = @tensor.Tensor::from_f32_array([2, 4], [
    1.0, -2.0, 0.5, 3.0, -1.0, 4.0, 2.0, -0.5,
  ])
  let x2 = @tensor.Tensor::from_f32_array([2, 4], [
    -3.0, 1.0, 2.0, -1.0, 0.5, -0.5, 1.5, 2.5,
  ])
  let ref1 = sess_scalar.run_single(x1)
  let out1 = sess_simd.run_single(x1)
  let _out2 = sess_simd.run_single(x2)

  // 验证 out1（视图输出）在第二次推理后仍保持与标量结果严格一致，未被池覆盖
  assert_true(@session.max_abs_diff_f32(out1, ref1) <= (1.0e-5 : Float))
  // 中间缓冲 h（非视图输出）应在第二次推理时命中 BufferPool
  assert_true(sess_simd.pool_stats().hits >= 1)
}

///|
/// 自定义包装 `ExecutionProvider`（测试用）：统计经由该后端调度的算子执行次数并委托给 `CpuSimdProvider`。
priv struct CountingProvider {
  inner : @session.CpuSimdProvider
  mut executed_ops : Int32
}

///|
impl @session.ExecutionProvider for CountingProvider with fn name(_self) {
  "counting-simd"
}

///|
impl @session.ExecutionProvider for CountingProvider with fn uses_prepacked_weights(
  _self,
) {
  true
}

///|
impl @session.ExecutionProvider for CountingProvider with fn eval_node(
  self,
  node,
  ctx,
) {
  self.executed_ops = self.executed_ops + 1
  @session.ExecutionProvider::eval_node(self.inner, node, ctx)
}

///|
/// 验证用户可通过 `Session::with_provider` 零侵入接入自定义 `ExecutionProvider` 并无缝共享静态预打包与 `BufferPool`。
test "Session::with_provider supports pluggable custom ExecutionProvider" {
  let graph = build_mini_cnn_graph()
  let custom : CountingProvider = {
    inner: @session.CpuSimdProvider::new(),
    executed_ops: 0,
  }
  let sess = @session.Session::with_provider(graph, custom)
  assert_eq(sess.provider_name(), "counting-simd")
  assert_eq(sess.packed_weight_count(), 2)
  assert_eq(sess.packed_conv_weight_count(), 1)

  let inp = @tensor.Tensor::zeros_f32([1, 2, 6, 6])
  let out = sess.run_single(inp)
  assert_eq(out.shape(), [1, 4])
  assert_eq(custom.executed_ops, 6)
}
