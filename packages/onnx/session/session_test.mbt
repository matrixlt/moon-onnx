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
impl @session.ExecutionProvider for CountingProvider with fn is_simd(_self) {
  true
}

///|
impl @session.ExecutionProvider for CountingProvider with fn prepare_graph(
  self,
  graph,
) {
  @session.ExecutionProvider::prepare_graph(self.inner, graph)
}

///|
impl @session.ExecutionProvider for CountingProvider with fn packed_weight_count(
  self,
) {
  @session.ExecutionProvider::packed_weight_count(self.inner)
}

///|
impl @session.ExecutionProvider for CountingProvider with fn packed_conv_weight_count(
  self,
) {
  @session.ExecutionProvider::packed_conv_weight_count(self.inner)
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

///|
/// 端到端验证 MobileNetV2 倒残差瓶颈（Expand 1x1 + BN + ReLU6 -> Depthwise 3x3 + BN + ReLU6
/// -> Project 1x1 + BN + Residual Add -> GlobalAveragePool -> Flatten -> Gemm -> Softmax）：
/// 验证图优化折叠（13 节点 -> 7 节点）、预打包 Depthwise+Pointwise SIMD 精度及 BufferPool 复用。
test "End-to-end MobileNetV2 inverted residual bottleneck + classifier (opt + SIMD + pool)" {
  let w_exp = @tensor.Tensor::from_f32(
    [12, 4, 1, 1],
    FixedArray::makei(12 * 4, fn(i) {
      Float::from_int(i % 11 - 5) * (0.15 : Float)
    }),
  )
  let s_exp = @tensor.Tensor::full_f32([12], (1.2 : Float))
  let b_exp = @tensor.Tensor::full_f32([12], (0.1 : Float))
  let m_exp = @tensor.Tensor::full_f32([12], (0.05 : Float))
  let v_exp = @tensor.Tensor::full_f32([12], (0.9 : Float))

  let w_dw = @tensor.Tensor::from_f32(
    [12, 1, 3, 3],
    FixedArray::makei(12 * 9, fn(i) {
      Float::from_int(i % 9 - 4) * (0.12 : Float)
    }),
  )
  let s_dw = @tensor.Tensor::full_f32([12], (0.9 : Float))
  let b_dw = @tensor.Tensor::full_f32([12], (-0.05 : Float))
  let m_dw = @tensor.Tensor::full_f32([12], (0.1 : Float))
  let v_dw = @tensor.Tensor::full_f32([12], (1.1 : Float))

  let w_proj = @tensor.Tensor::from_f32(
    [4, 12, 1, 1],
    FixedArray::makei(4 * 12, fn(i) {
      Float::from_int(i % 7 - 3) * (0.1 : Float)
    }),
  )
  let s_proj = @tensor.Tensor::full_f32([4], (1.1 : Float))
  let b_proj = @tensor.Tensor::full_f32([4], (0.02 : Float))
  let m_proj = @tensor.Tensor::full_f32([4], (-0.05 : Float))
  let v_proj = @tensor.Tensor::full_f32([4], (0.8 : Float))

  let w_cls = @tensor.Tensor::from_f32(
    [5, 4],
    FixedArray::makei(20, fn(i) { Float::from_int(i % 9 - 4) * (0.2 : Float) }),
  )
  let b_cls = @tensor.Tensor::from_f32_array([5], [0.1, -0.1, 0.2, 0.0, -0.2])

  let c0 = @tensor.Tensor::from_f32([], FixedArray::make(1, (0.0 : Float)))
  let c6 = @tensor.Tensor::from_f32([], FixedArray::make(1, (6.0 : Float)))

  let inits : Map[String, @tensor.Tensor] = Map([
    ("w_exp", w_exp),
    ("s_exp", s_exp),
    ("b_exp", b_exp),
    ("m_exp", m_exp),
    ("v_exp", v_exp),
    ("w_dw", w_dw),
    ("s_dw", s_dw),
    ("b_dw", b_dw),
    ("m_dw", m_dw),
    ("v_dw", v_dw),
    ("w_proj", w_proj),
    ("s_proj", s_proj),
    ("b_proj", b_proj),
    ("m_proj", m_proj),
    ("v_proj", v_proj),
    ("w_cls", w_cls),
    ("b_cls", b_cls),
  ])

  let raw_graph : @nn_ir.IrGraph = {
    name: "mobilenetv2_mini",
    inputs: ["x"],
    outputs: ["probs"],
    initializers: inits,
    nodes: [
      {
        name: "c_zero",
        op: @nn_ir.Constant(value=c0),
        inputs: [],
        outputs: ["k0"],
      },
      {
        name: "c_six",
        op: @nn_ir.Constant(value=c6),
        inputs: [],
        outputs: ["k6"],
      },
      {
        name: "expand_conv",
        op: @nn_ir.Conv(
          kernel_shape=Some([1, 1]),
          strides=[1, 1],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["x", "w_exp"],
        outputs: ["e_c"],
      },
      {
        name: "expand_bn",
        op: @nn_ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["e_c", "s_exp", "b_exp", "m_exp", "v_exp"],
        outputs: ["e_bn"],
      },
      {
        name: "expand_relu6",
        op: @nn_ir.Clip(min=None, max=None),
        inputs: ["e_bn", "k0", "k6"],
        outputs: ["e_out"],
      },
      {
        name: "dw_conv",
        op: @nn_ir.Conv(
          kernel_shape=Some([3, 3]),
          strides=[1, 1],
          pads=[1, 1, 1, 1],
          dilations=[1, 1],
          group=12,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["e_out", "w_dw"],
        outputs: ["dw_c"],
      },
      {
        name: "dw_bn",
        op: @nn_ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["dw_c", "s_dw", "b_dw", "m_dw", "v_dw"],
        outputs: ["dw_bn"],
      },
      {
        name: "dw_relu6",
        op: @nn_ir.Clip(min=None, max=None),
        inputs: ["dw_bn", "k0", "k6"],
        outputs: ["dw_out"],
      },
      {
        name: "proj_conv",
        op: @nn_ir.Conv(
          kernel_shape=Some([1, 1]),
          strides=[1, 1],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["dw_out", "w_proj"],
        outputs: ["p_c"],
      },
      {
        name: "proj_bn",
        op: @nn_ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["p_c", "s_proj", "b_proj", "m_proj", "v_proj"],
        outputs: ["p_out"],
      },
      {
        name: "res_add",
        op: @nn_ir.Add,
        inputs: ["x", "p_out"],
        outputs: ["res"],
      },
      {
        name: "gap",
        op: @nn_ir.GlobalAveragePool,
        inputs: ["res"],
        outputs: ["pooled"],
      },
      {
        name: "flat",
        op: @nn_ir.Flatten(axis=1),
        inputs: ["pooled"],
        outputs: ["f"],
      },
      {
        name: "fc",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["f", "w_cls", "b_cls"],
        outputs: ["logits"],
      },
      {
        name: "sm",
        op: @nn_ir.Softmax(axis=-1),
        inputs: ["logits"],
        outputs: ["probs"],
      },
    ],
  }

  let (opt_graph, stats) = @opt.optimize_graph(raw_graph)
  assert_eq(stats.lifted_constants, 2)
  assert_eq(stats.folded_conv_bn, 3)
  assert_eq(stats.folded_conv_activation, 2)
  assert_eq(stats.pretransposed_gemm_weights, 1)
  // 原始 15 节点折叠后仅余 8 节点（Expand Conv+Clip, DW Conv+Clip, Proj Conv, Add, GAP, Flatten, Gemm, Softmax）
  assert_eq(opt_graph.nodes.length(), 8)

  let sess_ref = @session.Session::from_graph(
    raw_graph,
    backend=@session.Scalar,
  )
  let sess_opt = @session.Session::from_graph(opt_graph, backend=@session.Simd)
  assert_eq(sess_opt.packed_conv_weight_count(), 3)

  let inp = @tensor.Tensor::from_f32(
    [1, 4, 8, 8],
    FixedArray::makei(4 * 8 * 8, fn(i) {
      Float::from_int(i % 17 - 8) * (0.25 : Float)
    }),
  )
  let out_ref = sess_ref.run_single(inp)
  let out_opt1 = sess_opt.run_single(inp)
  let out_opt2 = sess_opt.run_single(inp)

  assert_eq(out_opt1.shape(), [1, 5])
  assert_true(@session.max_abs_diff_f32(out_opt1, out_ref) <= (1.0e-5 : Float))
  assert_true(@session.max_abs_diff_f32(out_opt2, out_ref) <= (1.0e-5 : Float))
  // 第二次运行充分命中 BufferPool
  assert_true(sess_opt.pool_stats().hits >= 5)
}

///|
/// 端到端验证多输出 `Split` + 并行门控分支（`Sigmoid` / `HardSwish`）+ `Concat` + Transformer `LayerNormalization + Gelu` 块。
test "End-to-end multi-output Split/Concat and Transformer LayerNorm+Gelu FFN block" {
  // 1. Split -> (Sigmoid, HardSwish) -> Concat
  let g_split : @nn_ir.IrGraph = {
    name: "gated_split_concat",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([]),
    nodes: [
      {
        name: "sp",
        op: @nn_ir.Split(axis=1, split=Some([4, 4]), num_outputs=None),
        inputs: ["x"],
        outputs: ["left", "right"],
      },
      { name: "sig", op: @nn_ir.Sigmoid, inputs: ["left"], outputs: ["l_act"], },
      {
        name: "hsw",
        op: @nn_ir.HardSwish,
        inputs: ["right"],
        outputs: ["r_act"],
      },
      {
        name: "cat",
        op: @nn_ir.Concat(axis=1),
        inputs: ["l_act", "r_act"],
        outputs: ["y"],
      },
    ],
  }
  let x_sp = @tensor.Tensor::from_f32(
    [2, 8],
    FixedArray::makei(16, fn(i) { Float::from_int(i - 8) * (0.4 : Float) }),
  )
  let s_sc = @session.Session::from_graph(g_split, backend=@session.Scalar)
  let s_si = @session.Session::from_graph(g_split, backend=@session.Simd)
  let y_sc = s_sc.run_single(x_sp)
  let y_si1 = s_si.run_single(x_sp)
  let y_si2 = s_si.run_single(x_sp)
  assert_eq(y_si1.shape(), [2, 8])
  assert_true(@session.max_abs_diff_f32(y_si1, y_sc) <= (2.0e-4 : Float))
  assert_true(@session.max_abs_diff_f32(y_si2, y_sc) <= (2.0e-4 : Float))
  assert_true(s_si.pool_stats().hits >= 2)

  // 2. Transformer Pre-Norm FFN: LayerNorm -> MatMul -> Gelu -> MatMul -> Residual Add
  let ln_s = @tensor.Tensor::full_f32([8], (1.0 : Float))
  let ln_b = @tensor.Tensor::full_f32([8], (0.0 : Float))
  let w1 = @tensor.Tensor::from_f32(
    [8, 16],
    FixedArray::makei(8 * 16, fn(i) {
      Float::from_int(i % 13 - 6) * (0.1 : Float)
    }),
  )
  let w2 = @tensor.Tensor::from_f32(
    [16, 8],
    FixedArray::makei(16 * 8, fn(i) {
      Float::from_int(i % 11 - 5) * (0.1 : Float)
    }),
  )
  let g_ffn : @nn_ir.IrGraph = {
    name: "transformer_ffn",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([("ln_s", ln_s), ("ln_b", ln_b), ("w1", w1), ("w2", w2)]),
    nodes: [
      {
        name: "ln",
        op: @nn_ir.LayerNormalization(axis=-1, epsilon=(1.0e-5 : Float)),
        inputs: ["x", "ln_s", "ln_b"],
        outputs: ["h0"],
      },
      { name: "fc1", op: @nn_ir.MatMul, inputs: ["h0", "w1"], outputs: ["h1"], },
      { name: "act", op: @nn_ir.Gelu, inputs: ["h1"], outputs: ["h2"], },
      { name: "fc2", op: @nn_ir.MatMul, inputs: ["h2", "w2"], outputs: ["h3"], },
      { name: "res", op: @nn_ir.Add, inputs: ["x", "h3"], outputs: ["y"], },
    ],
  }
  let x_ffn = @tensor.Tensor::from_f32(
    [3, 8],
    FixedArray::makei(24, fn(i) { Float::from_int(i - 12) * (0.2 : Float) }),
  )
  let ffn_sc = @session.Session::from_graph(g_ffn, backend=@session.Scalar)
  let ffn_si = @session.Session::from_graph(g_ffn, backend=@session.Simd)
  let o_sc = ffn_sc.run_single(x_ffn)
  let o_si = ffn_si.run_single(x_ffn)
  assert_eq(o_si.shape(), [3, 8])
  assert_true(@session.max_abs_diff_f32(o_si, o_sc) <= (5.0e-4 : Float))
}

///|
/// 端到端验证 INT8 量化推理流水线：`QuantizeLinear -> QLinearMatMul -> DequantizeLinear`，
/// 确认 V128 `i32x4.dot_i16x8_s` 与标量后端逐位完全一致，且反量化结果与 Float32 基准高精度吻合。
test "End-to-end INT8 QuantizeLinear -> QLinearMatMul -> DequantizeLinear (Scalar vs V128 SIMD)" {
  let x_f32 = @tensor.Tensor::from_f32(
    [3, 24],
    FixedArray::makei(3 * 24, fn(i) {
      Float::from_int(i % 15 - 7) * (0.2 : Float)
    }),
  )
  let w_f32 = @tensor.Tensor::from_f32(
    [24, 6],
    FixedArray::makei(24 * 6, fn(i) {
      Float::from_int(i % 13 - 6) * (0.15 : Float)
    }),
  )

  // 对称 INT8 量化参数
  let a_scale = @tensor.Tensor::from_f32(
    [],
    FixedArray::make(1, (0.02 : Float)),
  )
  let a_zp = @tensor.Tensor::from_i8([], FixedArray::make(1, b'\x00'))
  let b_scale = @tensor.Tensor::from_f32(
    [],
    FixedArray::make(1, (0.01 : Float)),
  )
  let b_zp = @tensor.Tensor::from_i8([], FixedArray::make(1, b'\x00'))
  let y_scale = @tensor.Tensor::from_f32(
    [],
    FixedArray::make(1, (0.05 : Float)),
  )
  let y_zp = @tensor.Tensor::from_i8([], FixedArray::make(1, b'\x00'))

  // 离线量化权重 B 为 Int8（单元素 1 字节，相比 Float32 实现 4× 权重压缩）
  let w_q = @ops.quantize_linear_op(w_f32, b_scale, Some(b_zp))
  assert_eq(w_q.dtype(), @tensor.Int8)
  assert_eq(w_q.dtype().byte_size(), 1)

  let q_graph : @nn_ir.IrGraph = {
    name: "int8_qlinear_mlp",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([
      ("a_s", a_scale),
      ("a_zp", a_zp),
      ("w_q", w_q),
      ("b_s", b_scale),
      ("b_zp", b_zp),
      ("y_s", y_scale),
      ("y_zp", y_zp),
    ]),
    nodes: [
      {
        name: "quant_x",
        op: @nn_ir.QuantizeLinear(axis=1),
        inputs: ["x", "a_s", "a_zp"],
        outputs: ["x_q"],
      },
      {
        name: "qmm",
        op: @nn_ir.QLinearMatMul,
        inputs: ["x_q", "a_s", "a_zp", "w_q", "b_s", "b_zp", "y_s", "y_zp"],
        outputs: ["y_q"],
      },
      {
        name: "dequant_y",
        op: @nn_ir.DequantizeLinear(axis=1),
        inputs: ["y_q", "y_s", "y_zp"],
        outputs: ["y"],
      },
    ],
  }

  let s_sc = @session.Session::from_graph(q_graph, backend=@session.Scalar)
  let s_si = @session.Session::from_graph(q_graph, backend=@session.Simd)
  let out_sc = s_sc.run_single(x_f32)
  let out_si = s_si.run_single(x_f32)
  let ref_f32 = @ops.matmul_f32(x_f32, w_f32)

  // V128 INT8 SIMD 与标量参考逐位完全一致（误差为 0）
  assert_eq(@session.max_abs_diff_f32(out_si, out_sc), (0.0 : Float))
  // 量化推理结果与 Float32 基准误差在半个量化步长附近
  assert_true(@session.max_abs_diff_f32(out_si, ref_f32) <= (0.06 : Float))
}

///|
/// 端到端验证 Silero VAD 风格流式语音端点检测流水线：
/// `Audio[1,1,32] -> Conv1D+BN+Relu -> Depthwise Conv1D+Relu -> Transpose -> LSTM(流式 h/c 状态传递) -> Gemm -> Sigmoid`。
/// 校验图优化折叠 Conv1D+BN+Relu、多输出 LSTM 状态跨帧传递及 Scalar vs V128 SIMD 数值一致。
test "End-to-end Silero VAD-style streaming audio pipeline (Conv1D + LSTM + stateful h/c)" {
  let w_c1 = @tensor.Tensor::from_f32(
    [8, 1, 5],
    FixedArray::makei(40, fn(i) { Float::from_int(i % 11 - 5) * (0.15 : Float) }),
  )
  let bn_s = @tensor.Tensor::full_f32([8], (1.1 : Float))
  let bn_b = @tensor.Tensor::full_f32([8], (0.05 : Float))
  let bn_m = @tensor.Tensor::full_f32([8], (0.0 : Float))
  let bn_v = @tensor.Tensor::full_f32([8], (1.0 : Float))
  let w_dw = @tensor.Tensor::from_f32(
    [8, 1, 3],
    FixedArray::makei(24, fn(i) { Float::from_int(i % 7 - 3) * (0.1 : Float) }),
  )
  let lstm_w = @tensor.Tensor::from_f32(
    [1, 24, 8],
    FixedArray::makei(24 * 8, fn(i) {
      Float::from_int(i % 9 - 4) * (0.1 : Float)
    }),
  )
  let lstm_r = @tensor.Tensor::from_f32(
    [1, 24, 6],
    FixedArray::makei(24 * 6, fn(i) {
      Float::from_int(i % 7 - 3) * (0.08 : Float)
    }),
  )
  let lstm_b = @tensor.Tensor::full_f32([1, 48], (0.01 : Float))
  let sh_h = @tensor.Tensor::from_i64([2], [1, 6])
  let w_fc = @tensor.Tensor::from_f32_array([1, 6], [
    0.5, -0.3, 0.4, 0.2, -0.5, 0.3,
  ])
  let b_fc = @tensor.Tensor::from_f32_array([1], [0.05])

  let raw_graph : @nn_ir.IrGraph = {
    name: "silero_vad_mini",
    inputs: ["audio", "h_in", "c_in"],
    outputs: ["speech_prob", "h_out", "c_out"],
    initializers: Map([
      ("w_c1", w_c1),
      ("bn_s", bn_s),
      ("bn_b", bn_b),
      ("bn_m", bn_m),
      ("bn_v", bn_v),
      ("w_dw", w_dw),
      ("lstm_w", lstm_w),
      ("lstm_r", lstm_r),
      ("lstm_b", lstm_b),
      ("sh_h", sh_h),
      ("w_fc", w_fc),
      ("b_fc", b_fc),
    ]),
    nodes: [
      {
        name: "enc1",
        op: @nn_ir.Conv(
          kernel_shape=Some([5]),
          strides=[2],
          pads=[2, 2],
          dilations=[1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["audio", "w_c1"],
        outputs: ["c1"],
      },
      {
        name: "bn1",
        op: @nn_ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["c1", "bn_s", "bn_b", "bn_m", "bn_v"],
        outputs: ["c1_bn"],
      },
      { name: "act1", op: @nn_ir.Relu, inputs: ["c1_bn"], outputs: ["c1_act"], },
      {
        name: "enc2_dw",
        op: @nn_ir.Conv(
          kernel_shape=Some([3]),
          strides=[2],
          pads=[1, 1],
          dilations=[1],
          group=8,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["c1_act", "w_dw"],
        outputs: ["c2"],
      },
      { name: "act2", op: @nn_ir.Relu, inputs: ["c2"], outputs: ["c2_act"], },
      // [N=1, C=8, L=8] -> [L=8, N=1, C=8] 供 LSTM 消费
      {
        name: "to_seq",
        op: @nn_ir.Transpose(perm=[2, 0, 1]),
        inputs: ["c2_act"],
        outputs: ["seq_in"],
      },
      {
        name: "rnn",
        op: @nn_ir.Lstm(
          hidden_size=Some(6),
          direction="forward",
          layout=0,
          clip=None,
        ),
        inputs: ["seq_in", "lstm_w", "lstm_r", "lstm_b", "", "h_in", "c_in"],
        outputs: ["rnn_y", "h_out", "c_out"],
      },
      {
        name: "flat_h",
        op: @nn_ir.Reshape(allowzero=false),
        inputs: ["h_out", "sh_h"],
        outputs: ["h_2d"],
      },
      {
        name: "head",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["h_2d", "w_fc", "b_fc"],
        outputs: ["logit"],
      },
      {
        name: "prob",
        op: @nn_ir.Sigmoid,
        inputs: ["logit"],
        outputs: ["speech_prob"],
      },
    ],
  }

  let (opt_graph, stats) = @opt.optimize_graph(raw_graph)
  assert_eq(stats.folded_conv_bn, 1)
  assert_eq(stats.folded_conv_activation, 2)
  assert_eq(opt_graph.nodes.length(), 7)

  let s_ref = @session.Session::from_graph(raw_graph, backend=@session.Scalar)
  let s_opt = @session.Session::from_graph(opt_graph, backend=@session.Simd)
  assert_eq(s_opt.packed_conv_weight_count(), 2)

  // 连续模拟 2 帧流式音频推理（将第 1 帧输出的 h_out/c_out 喂给第 2 帧）
  let frame1 = @tensor.Tensor::from_f32(
    [1, 1, 32],
    FixedArray::makei(32, fn(i) { Float::from_int(i % 9 - 4) * (0.2 : Float) }),
  )
  let frame2 = @tensor.Tensor::from_f32(
    [1, 1, 32],
    FixedArray::makei(32, fn(i) {
      Float::from_int((i + 3) % 9 - 4) * (0.25 : Float)
    }),
  )
  let h0 = @tensor.Tensor::zeros_f32([1, 1, 6])
  let c0 = @tensor.Tensor::zeros_f32([1, 1, 6])

  let r1 = s_ref.run(Map([("audio", frame1), ("h_in", h0), ("c_in", c0)]))
  let o1 = s_opt.run(Map([("audio", frame1), ("h_in", h0), ("c_in", c0)]))
  let r2 = s_ref.run(
    Map([
      ("audio", frame2),
      ("h_in", r1.get("h_out").unwrap()),
      ("c_in", r1.get("c_out").unwrap()),
    ]),
  )
  let o2 = s_opt.run(
    Map([
      ("audio", frame2),
      ("h_in", o1.get("h_out").unwrap()),
      ("c_in", o1.get("c_out").unwrap()),
    ]),
  )

  let prob2 = o2.get("speech_prob").unwrap()
  let prob2_val = prob2.get_f32([0, 0])
  assert_true(prob2_val > (0.0 : Float) && prob2_val < (1.0 : Float))
  assert_true(
    @session.max_abs_diff_f32(prob2, r2.get("speech_prob").unwrap()) <=
    (1.0e-5 : Float),
  )
  assert_true(
    @session.max_abs_diff_f32(
      o2.get("h_out").unwrap(),
      r2.get("h_out").unwrap(),
    ) <=
    (1.0e-5 : Float),
  )
}

///|
/// 端到端验证 YOLO / UNet 常见多尺度特征融合颈部（`Conv2D(s=2) -> Resize(2× bilinear) -> Concat -> Conv2D`）。
test "End-to-end FPN/UNet Conv2D + Resize(2x bilinear) + Concat skip connection" {
  let w_down = @tensor.Tensor::from_f32(
    [4, 2, 3, 3],
    FixedArray::makei(4 * 2 * 9, fn(i) {
      Float::from_int(i % 7 - 3) * (0.1 : Float)
    }),
  )
  let w_fuse = @tensor.Tensor::from_f32(
    [3, 6, 1, 1],
    FixedArray::makei(18, fn(i) { Float::from_int(i % 9 - 4) * (0.1 : Float) }),
  )
  let scales = @tensor.Tensor::from_f32_array([4], [1.0, 1.0, 2.0, 2.0])

  let g : @nn_ir.IrGraph = {
    name: "fpn_neck",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([
      ("w_down", w_down),
      ("w_fuse", w_fuse),
      ("scales", scales),
    ]),
    nodes: [
      {
        name: "down",
        op: @nn_ir.Conv(
          kernel_shape=Some([3, 3]),
          strides=[2, 2],
          pads=[1, 1, 1, 1],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.ReluAct,
        ),
        inputs: ["x", "w_down"],
        outputs: ["d1"],
      },
      {
        name: "up",
        op: @nn_ir.Resize(
          mode="linear",
          coordinate_transformation_mode="half_pixel",
          nearest_mode="round_prefer_floor",
        ),
        inputs: ["d1", "", "scales"],
        outputs: ["u1"],
      },
      {
        name: "skip_cat",
        op: @nn_ir.Concat(axis=1),
        inputs: ["x", "u1"],
        outputs: ["cat"],
      },
      {
        name: "fuse",
        op: @nn_ir.Conv(
          kernel_shape=Some([1, 1]),
          strides=[1, 1],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["cat", "w_fuse"],
        outputs: ["y"],
      },
    ],
  }

  let inp = @tensor.Tensor::from_f32(
    [1, 2, 8, 8],
    FixedArray::makei(128, fn(i) { Float::from_int(i % 13 - 6) * (0.2 : Float) }),
  )
  let s_sc = @session.Session::from_graph(g, backend=@session.Scalar)
  let s_si = @session.Session::from_graph(g, backend=@session.Simd)
  let y_sc = s_sc.run_single(inp)
  let y_si = s_si.run_single(inp)
  assert_eq(y_si.shape(), [1, 3, 8, 8])
  assert_true(@session.max_abs_diff_f32(y_si, y_sc) <= (1.0e-5 : Float))
}

///|
/// 验证 `Equal + If` 控制流子图执行（模拟 Silero VAD 16kHz / 8kHz 双分支架构）：
/// 确认子图词法捕获外层 `input`/`state`、两分支各自完成 V128 权重预打包，且多输出状态递推精度一致。
test "If control-flow subgraphs capture outer scope and prepack weights on Scalar & SIMD" {
  let w_16k = @tensor.Tensor::from_f32_array([2, 2], [2.0, 0.0, 0.0, 2.0])
  let w_8k = @tensor.Tensor::from_f32_array([2, 2], [0.5, 0.0, 0.0, 0.5])
  let sr_16k = @tensor.Tensor::from_i64(
    [],
    FixedArray::make(1, (16000 : Int64)),
  )

  let then_branch : @nn_ir.IrGraph = {
    name: "branch_16k",
    inputs: [],
    outputs: ["y_16k", "s_16k"],
    initializers: Map([("w_16k", w_16k)]),
    nodes: [
      {
        name: "mm_16k",
        op: @nn_ir.MatMul,
        inputs: ["x", "w_16k"],
        outputs: ["y_16k"],
      },
      {
        name: "add_16k",
        op: @nn_ir.Add,
        inputs: ["state", "y_16k"],
        outputs: ["s_16k"],
      },
    ],
  }
  let else_branch : @nn_ir.IrGraph = {
    name: "branch_8k",
    inputs: [],
    outputs: ["y_8k", "s_8k"],
    initializers: Map([("w_8k", w_8k)]),
    nodes: [
      {
        name: "mm_8k",
        op: @nn_ir.MatMul,
        inputs: ["x", "w_8k"],
        outputs: ["y_8k"],
      },
      {
        name: "add_8k",
        op: @nn_ir.Add,
        inputs: ["state", "y_8k"],
        outputs: ["s_8k"],
      },
    ],
  }
  let top_graph : @nn_ir.IrGraph = {
    name: "dual_rate_if_model",
    inputs: ["x", "state", "sr"],
    outputs: ["y", "stateN"],
    initializers: Map([("sr_16k", sr_16k)]),
    nodes: [
      {
        name: "cmp_sr",
        op: @nn_ir.Equal,
        inputs: ["sr", "sr_16k"],
        outputs: ["is_16k"],
      },
      {
        name: "dispatch_rate",
        op: @nn_ir.If(then_branch~, else_branch~),
        inputs: ["is_16k"],
        outputs: ["y", "stateN"],
      },
    ],
  }

  let (opt_g, _) = @opt.optimize_graph(top_graph)
  let sess_simd = @session.Session::from_graph(opt_g, backend=@session.Simd)
  let sess_scalar = @session.Session::from_graph(opt_g, backend=@session.Scalar)
  // 两个子分支的静态权重均已在初始化期预打包
  assert_eq(sess_simd.packed_weight_count(), 2)

  let x = @tensor.Tensor::from_f32_array([1, 2], [3.0, -4.0])
  let st0 = @tensor.Tensor::from_f32_array([1, 2], [1.0, 2.0])

  // 走 then_branch (16kHz): y = [6, -8], stateN = [7, -6]
  let f16 : Map[String, @tensor.Tensor] = Map([
    ("x", x),
    ("state", st0),
    ("sr", @tensor.Tensor::from_i64([], FixedArray::make(1, (16000 : Int64)))),
  ])
  let r16_si = sess_simd.run(f16)
  let r16_sc = sess_scalar.run(f16)
  assert_true(
    @session.max_abs_diff_f32(
      r16_si.get("y").unwrap(),
      @tensor.Tensor::from_f32_array([1, 2], [6.0, -8.0]),
    ) <=
    (1.0e-5 : Float),
  )
  assert_true(
    @session.max_abs_diff_f32(
      r16_si.get("stateN").unwrap(),
      r16_sc.get("stateN").unwrap(),
    ) <=
    (1.0e-5 : Float),
  )

  // 走 else_branch (8kHz): y = [1.5, -2.0], stateN = [2.5, 0.0]
  let f8 : Map[String, @tensor.Tensor] = Map([
    ("x", x),
    ("state", st0),
    ("sr", @tensor.Tensor::from_i64([], FixedArray::make(1, (8000 : Int64)))),
  ])
  let r8_si = sess_simd.run(f8)
  assert_true(
    @session.max_abs_diff_f32(
      r8_si.get("y").unwrap(),
      @tensor.Tensor::from_f32_array([1, 2], [1.5, -2.0]),
    ) <=
    (1.0e-5 : Float),
  )
  assert_true(
    @session.max_abs_diff_f32(
      r8_si.get("stateN").unwrap(),
      @tensor.Tensor::from_f32_array([1, 2], [2.5, 0.0]),
    ) <=
    (1.0e-5 : Float),
  )
}

///|
/// 端到端验证现代 Q4 大模型单层 Transformer Block（RMSNorm -> Q/K/V MatMulNBits(4-bit) -> RoPE -> GQA KV-Cache
/// -> O_proj MatMulNBits -> SkipRMSNorm -> SwiGLU(SiLU 融合) -> Down_proj MatMulNBits）：
/// 在纯内存四端覆盖 prefill (`past_seq=0, seq=2`) + decode (`past_seq=2, seq=1`) 与 Scalar vs V128 SIMD 数值对齐。
test "End-to-end Q4 LLM Transformer block (MatMulNBits + RoPE + GQA KV-Cache + SkipRMSNorm + SiLU)" {
  // hidden=32, num_heads=4, kv_num_heads=2, head_dim=8, block_size=32
  let hidden = 32
  let kv_dim = 16
  let make_q4 = fn(
    n_out : Int32,
    seed : Int32,
  ) -> (@tensor.Tensor, @tensor.Tensor) raise {
    let w_b = @tensor.Tensor::from_u8(
      [n_out, 1, 16],
      FixedArray::makei(n_out * 16, fn(i) {
        ((i * 19 + seed * 7) & 0xFF).to_byte()
      }),
    )
    let w_s = @tensor.Tensor::from_f32(
      [n_out],
      FixedArray::makei(n_out, fn(i) {
        Float::from_int(i % 5 + 1) * (0.02 : Float)
      }),
    )
    (w_b, w_s)
  }
  let (wq_b, wq_s) = make_q4(hidden, 1)
  let (wk_b, wk_s) = make_q4(kv_dim, 2)
  let (wv_b, wv_s) = make_q4(kv_dim, 3)
  let (wo_b, wo_s) = make_q4(hidden, 4)
  let (wg_b, wg_s) = make_q4(hidden, 5)
  let (wu_b, wu_s) = make_q4(hidden, 6)
  let (wd_b, wd_s) = make_q4(hidden, 7)

  let norm1_w = @tensor.Tensor::full_f32([hidden], (1.0 : Float))
  let norm2_w = @tensor.Tensor::full_f32([hidden], (1.0 : Float))
  let cos_cache = @tensor.Tensor::from_f32(
    [8, 4],
    FixedArray::makei(32, fn(i) {
      if i / 4 % 2 == 0 {
        (1.0 : Float)
      } else {
        (0.6 : Float)
      }
    }),
  )
  let sin_cache = @tensor.Tensor::from_f32(
    [8, 4],
    FixedArray::makei(32, fn(i) {
      if i / 4 % 2 == 0 {
        (0.0 : Float)
      } else {
        (0.8 : Float)
      }
    }),
  )

  let g : @nn_ir.IrGraph = {
    name: "mini_q4_llama_block",
    inputs: ["x", "pos", "past_k", "past_v", "seqlens_k", "tot_seq"],
    outputs: ["out", "present_k", "present_v"],
    initializers: Map([
      ("norm1.w", norm1_w),
      ("norm2.w", norm2_w),
      ("cos", cos_cache),
      ("sin", sin_cache),
      ("wq.b", wq_b),
      ("wq.s", wq_s),
      ("wk.b", wk_b),
      ("wk.s", wk_s),
      ("wv.b", wv_b),
      ("wv.s", wv_s),
      ("wo.b", wo_b),
      ("wo.s", wo_s),
      ("wg.b", wg_b),
      ("wg.s", wg_s),
      ("wu.b", wu_b),
      ("wu.s", wu_s),
      ("wd.b", wd_b),
      ("wd.s", wd_s),
    ]),
    nodes: [
      {
        name: "rms1",
        op: @nn_ir.RmsNormalization(axis=-1, epsilon=(1.0e-5 : Float)),
        inputs: ["x", "norm1.w"],
        outputs: ["h0"],
      },
      {
        name: "q_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=hidden, bits=4, block_size=32),
        inputs: ["h0", "wq.b", "wq.s"],
        outputs: ["q"],
      },
      {
        name: "k_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=kv_dim, bits=4, block_size=32),
        inputs: ["h0", "wk.b", "wk.s"],
        outputs: ["k"],
      },
      {
        name: "v_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=kv_dim, bits=4, block_size=32),
        inputs: ["h0", "wv.b", "wv.s"],
        outputs: ["v"],
      },
      {
        name: "rope_q",
        op: @nn_ir.RotaryEmbedding(
          interleaved=0,
          num_heads=0,
          rotary_embedding_dim=0,
        ),
        inputs: ["q", "pos", "cos", "sin"],
        outputs: ["qr"],
      },
      {
        name: "rope_k",
        op: @nn_ir.RotaryEmbedding(
          interleaved=0,
          num_heads=0,
          rotary_embedding_dim=0,
        ),
        inputs: ["k", "pos", "cos", "sin"],
        outputs: ["kr"],
      },
      {
        name: "gqa",
        op: @nn_ir.GroupQueryAttention(
          num_heads=4,
          kv_num_heads=2,
          scale=(0.35355338 : Float),
          local_window_size=-1,
          softcap=(0.0 : Float),
          do_rotary=0,
          rotary_interleaved=0,
        ),
        inputs: ["qr", "kr", "v", "past_k", "past_v", "seqlens_k", "tot_seq"],
        outputs: ["attn_out", "present_k", "present_v"],
      },
      {
        name: "o_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=hidden, bits=4, block_size=32),
        inputs: ["attn_out", "wo.b", "wo.s"],
        outputs: ["o"],
      },
      {
        name: "skip_rms",
        op: @nn_ir.SkipRmsNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["x", "o", "norm2.w"],
        outputs: ["h1", "", "", "res1"],
      },
      {
        name: "gate_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=hidden, bits=4, block_size=32),
        inputs: ["h1", "wg.b", "wg.s"],
        outputs: ["gate"],
      },
      {
        name: "up_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=hidden, bits=4, block_size=32),
        inputs: ["h1", "wu.b", "wu.s"],
        outputs: ["up"],
      },
      {
        name: "silu",
        op: @nn_ir.Silu,
        inputs: ["gate"],
        outputs: ["gate_act"],
      },
      {
        name: "swiglu",
        op: @nn_ir.Mul,
        inputs: ["gate_act", "up"],
        outputs: ["mlp_mid"],
      },
      {
        name: "down_proj",
        op: @nn_ir.MatMulNBits(k=hidden, n=hidden, bits=4, block_size=32),
        inputs: ["mlp_mid", "wd.b", "wd.s"],
        outputs: ["mlp_out"],
      },
      {
        name: "res_add",
        op: @nn_ir.Add,
        inputs: ["res1", "mlp_out"],
        outputs: ["out"],
      },
    ],
  }

  let s_scalar = @session.Session::from_graph(g, backend=@session.Scalar)
  let s_simd = @session.Session::from_graph(g, backend=@session.Simd)

  // Step 1: Prefill (seq=2, past=0)
  let x_pre = @tensor.Tensor::from_f32(
    [1, 2, hidden],
    FixedArray::makei(2 * hidden, fn(i) {
      Float::from_int(i % 13 - 6) * (0.1 : Float)
    }),
  )
  let pre_feeds : Map[String, @tensor.Tensor] = Map([
    ("x", x_pre),
    ("pos", @tensor.Tensor::from_i64([1, 2], [0, 1])),
    ("past_k", @tensor.Tensor::from_f32([1, 2, 0, 8], [])),
    ("past_v", @tensor.Tensor::from_f32([1, 2, 0, 8], [])),
    ("seqlens_k", @tensor.Tensor::from_i32([1], [1])),
    ("tot_seq", @tensor.Tensor::from_i32([1], [2])),
  ])
  let r_pre_sc = s_scalar.run(pre_feeds)
  let r_pre_si = s_simd.run(pre_feeds)
  assert_eq(r_pre_si.get("out").unwrap().shape(), [1, 2, 32])
  assert_eq(r_pre_si.get("present_k").unwrap().shape(), [1, 2, 2, 8])
  assert_true(
    @session.max_abs_diff_f32(
      r_pre_si.get("out").unwrap(),
      r_pre_sc.get("out").unwrap(),
    ) <=
    (1.0e-5 : Float),
  )

  // Step 2: Decode (seq=1, past=2)
  let x_dec = @tensor.Tensor::from_f32(
    [1, 1, hidden],
    FixedArray::makei(hidden, fn(i) {
      Float::from_int(i % 11 - 5) * (0.12 : Float)
    }),
  )
  let dec_feeds : Map[String, @tensor.Tensor] = Map([
    ("x", x_dec),
    ("pos", @tensor.Tensor::from_i64([1, 1], [2])),
    ("past_k", r_pre_si.get("present_k").unwrap()),
    ("past_v", r_pre_si.get("present_v").unwrap()),
    ("seqlens_k", @tensor.Tensor::from_i32([1], [2])),
    ("tot_seq", @tensor.Tensor::from_i32([1], [3])),
  ])
  let r_dec_sc = s_scalar.run(dec_feeds)
  let r_dec_si = s_simd.run(dec_feeds)
  assert_eq(r_dec_si.get("out").unwrap().shape(), [1, 1, 32])
  assert_eq(r_dec_si.get("present_k").unwrap().shape(), [1, 2, 3, 8])
  assert_true(
    @session.max_abs_diff_f32(
      r_dec_si.get("out").unwrap(),
      r_dec_sc.get("out").unwrap(),
    ) <=
    (1.0e-5 : Float),
  )
}

///|
/// 端到端验证 YOLOv8 典型结构（`Conv -> SiLU 融合 -> MaxPool -> Resize(nearest) -> Concat -> Split -> Conv`）
/// 及标准 ONNX `NonMaxSuppression` 算子执行。
test "End-to-end YOLOv8 FPN block (SiLU fusion + Resize + Split + Concat) and NonMaxSuppression" {
  // 1. Mini YOLOv8 neck: Conv -> Sigmoid+Mul(fused to Silu) -> Split -> Concat
  let w1 = @tensor.Tensor::from_f32(
    [4, 2, 1, 1],
    FixedArray::makei(8, fn(i) { Float::from_int(i - 3) * (0.2 : Float) }),
  )
  let scales = @tensor.Tensor::from_f32_array([4], [1.0, 1.0, 2.0, 2.0])
  let raw_g : @nn_ir.IrGraph = {
    name: "mini_yolov8_fpn",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([("w1", w1), ("scales", scales)]),
    nodes: [
      {
        name: "c0",
        op: @nn_ir.Conv(
          kernel_shape=Some([1, 1]),
          strides=[1, 1],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["x", "w1"],
        outputs: ["c0_out"],
      },
      { name: "sig0", op: @nn_ir.Sigmoid, inputs: ["c0_out"], outputs: ["s0"], },
      {
        name: "mul0",
        op: @nn_ir.Mul,
        inputs: ["c0_out", "s0"],
        outputs: ["a0"],
      },
      {
        name: "pool0",
        op: @nn_ir.MaxPool(
          kernel_shape=[2, 2],
          strides=[2, 2],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          ceil_mode=false,
          auto_pad=@nn_ir.NotSet,
        ),
        inputs: ["a0"],
        outputs: ["p0"],
      },
      {
        name: "up0",
        op: @nn_ir.Resize(
          mode="nearest",
          coordinate_transformation_mode="asymmetric",
          nearest_mode="floor",
        ),
        inputs: ["p0", "", "scales"],
        outputs: ["u0"],
      },
      {
        name: "sp0",
        op: @nn_ir.Split(axis=1, split=Some([2, 2]), num_outputs=None),
        inputs: ["u0"],
        outputs: ["sp_a", "sp_b"],
      },
      {
        name: "cat0",
        op: @nn_ir.Concat(axis=1),
        inputs: ["sp_a", "sp_b", "x"],
        outputs: ["y"],
      },
    ],
  }
  let (opt_g, stats) = @opt.optimize_graph(raw_g)
  assert_eq(stats.folded_silu, 1)
  let s_sc = @session.Session::from_graph(raw_g, backend=@session.Scalar)
  let s_si = @session.Session::from_graph(opt_g, backend=@session.Simd)
  let inp = @tensor.Tensor::from_f32(
    [1, 2, 4, 4],
    FixedArray::makei(32, fn(i) { Float::from_int(i - 16) * (0.15 : Float) }),
  )
  let o_sc = s_sc.run_single(inp)
  let o_si = s_si.run_single(inp)
  assert_eq(o_si.shape(), [1, 6, 4, 4])
  assert_true(@session.max_abs_diff_f32(o_si, o_sc) <= (1.0e-5 : Float))

  // 2. NonMaxSuppression 算子端到端验证
  let nms_g : @nn_ir.IrGraph = {
    name: "nms_post",
    inputs: ["boxes", "scores"],
    outputs: ["selected"],
    initializers: Map([
      ("max_boxes", @tensor.Tensor::from_i64([1], [5])),
      ("iou_th", @tensor.Tensor::from_f32_array([1], [0.5])),
      ("score_th", @tensor.Tensor::from_f32_array([1], [0.3])),
    ]),
    nodes: [
      {
        name: "nms0",
        op: @nn_ir.NonMaxSuppression(center_point_box=1),
        inputs: ["boxes", "scores", "max_boxes", "iou_th", "score_th"],
        outputs: ["selected"],
      },
    ],
  }
  let nms_sess = @session.Session::from_graph(nms_g, backend=@session.Simd)
  let boxes = @tensor.Tensor::from_f32_array([1, 3, 4], [
    10.0, 10.0, 20.0, 20.0, 11.0, 11.0, 20.0, 20.0, 100.0, 100.0, 20.0, 20.0,
  ])
  let scores = @tensor.Tensor::from_f32_array([1, 2, 3], [
    0.9, 0.85, 0.1, // class 0: box 0 kept, box 1 suppressed, box 2 below score_th
     0.2, 0.4, 0.8, // class 1: box 2 kept, box 1 kept, box 0 below score_th
  ])
  let res = nms_sess.run(Map([("boxes", boxes), ("scores", scores)]))
  let sel = res.get("selected").unwrap()
  assert_eq(sel.shape(), [3, 3])
  let sbuf = sel.to_i64_buffer()
  // (0, 0, 0), (0, 1, 2), (0, 1, 1)
  assert_eq(sbuf[0], (0 : Int64))
  assert_eq(sbuf[1], (0 : Int64))
  assert_eq(sbuf[2], (0 : Int64))
  assert_eq(sbuf[3], (0 : Int64))
  assert_eq(sbuf[4], (1 : Int64))
  assert_eq(sbuf[5], (2 : Int64))
  assert_eq(sbuf[6], (0 : Int64))
  assert_eq(sbuf[7], (1 : Int64))
  assert_eq(sbuf[8], (1 : Int64))
}
