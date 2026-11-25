///|
/// 测试 Constant 提升、Identity 消除与 DCE 死节点清除。
test "optimize_graph lifts constants, eliminates identity and dead nodes" {
  let c_val = @tensor.Tensor::from_f32_array([2], [1.0, 2.0])
  let g : @nn_ir.IrGraph = {
    name: "opt_basic",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([]),
    nodes: [
      {
        name: "c0",
        op: @nn_ir.Constant(value=c_val),
        inputs: [],
        outputs: ["c"],
      },
      { name: "id0", op: @nn_ir.Identity, inputs: ["x"], outputs: ["x_id"], },
      { name: "dead", op: @nn_ir.Relu, inputs: ["x_id"], outputs: ["unused"], },
      { name: "add0", op: @nn_ir.Add, inputs: ["x_id", "c"], outputs: ["y"], },
    ],
  }

  let (opt_g, stats) = @opt.optimize_graph(g)
  assert_eq(stats.lifted_constants, 1)
  assert_eq(stats.eliminated_identities, 1)
  assert_eq(stats.eliminated_dead_nodes, 1)
  assert_eq(opt_g.nodes.length(), 1)
  assert_eq(opt_g.nodes[0].inputs, ["x", "c"])
  assert_true(opt_g.initializers.contains("c"))
}

///|
/// 测试 Conv + BatchNormalization 静态折叠及静态 Gemm(trans_b=true) 权重预转置。
test "optimize_graph folds Conv+BN, Conv+bias, and pre-transposes Gemm weights" {
  let w_conv = @tensor.Tensor::from_f32_array([2, 1, 2, 2], [
    1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5,
  ])
  let scale = @tensor.Tensor::from_f32_array([2], [2.0, 0.5])
  let bias = @tensor.Tensor::from_f32_array([2], [0.5, -1.0])
  let mean = @tensor.Tensor::from_f32_array([2], [1.0, 2.0])
  let variance = @tensor.Tensor::from_f32_array([2], [4.0, 1.0])
  let w_fc = @tensor.Tensor::from_f32_array([3, 2], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  ])

  let inits : Map[String, @tensor.Tensor] = Map([
    ("w_conv", w_conv),
    ("s", scale),
    ("b", bias),
    ("m", mean),
    ("v", variance),
    ("w_fc", w_fc),
  ])
  let g : @nn_ir.IrGraph = {
    name: "conv_bn_gemm",
    inputs: ["x"],
    outputs: ["y"],
    initializers: inits,
    nodes: [
      {
        name: "conv0",
        op: @nn_ir.Conv(
          kernel_shape=Some([2, 2]),
          strides=[1, 1],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["x", "w_conv"],
        outputs: ["c0"],
      },
      {
        name: "bn0",
        op: @nn_ir.BatchNormalization(epsilon=(0.0 : Float)),
        inputs: ["c0", "s", "b", "m", "v"],
        outputs: ["c1"],
      },
      { name: "relu0", op: @nn_ir.Relu, inputs: ["c1"], outputs: ["r1"], },
      {
        name: "gemm0",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["r1", "w_fc"],
        outputs: ["y"],
      },
    ],
  }

  let (opt_g, stats) = @opt.optimize_graph(g)
  assert_eq(stats.folded_conv_bn, 1)
  assert_eq(stats.folded_conv_activation, 1)
  assert_eq(stats.pretransposed_gemm_weights, 1)
  assert_eq(opt_g.nodes.length(), 2)
  assert_eq(opt_g.nodes[0].op.name(), "Conv+Relu")

  // ch0: factor = 2.0 / sqrt(4.0) = 1.0, fused_b[0] = (0 - 1) * 1 + 0.5 = -0.5
  let fused_b = opt_g.initializers.get(opt_g.nodes[0].inputs[2]).unwrap()
  assert_eq(fused_b.get_f32([0]), (-0.5 : Float))
  // ch1: factor = 0.5 / sqrt(1.0) = 0.5, fused_b[1] = (0 - 2) * 0.5 - 1.0 = -2.0
  assert_eq(fused_b.get_f32([1]), (-2.0 : Float))

  // Gemm 权重从 [3, 2] 预转置为 [2, 3]
  let pt_w = opt_g.initializers.get(opt_g.nodes[1].inputs[1]).unwrap()
  assert_eq(pt_w.shape(), [2, 3])
}

///|
/// 测试 MobileNetV2 典型结构：`Conv + BatchNormalization + Clip(0, 6)`（Opset 11+ 常量输入形式）
/// 被一次性折叠为单个 `Conv+Clip` 节点，消除 BN 与 Clip 两次整图读写。
test "optimize_graph folds Conv+BN+Clip(ReLU6) into single fused Conv+Clip" {
  let w_conv = @tensor.Tensor::from_f32_array([2, 1, 3, 3], [
    1.0, 0.0, -1.0, 0.5, 1.0, 0.5, -0.5, 0.0, 0.5, 0.2, 0.3, 0.4, -0.1, 0.5, -0.2,
    0.1, 0.2, 0.3,
  ])
  let scale = @tensor.Tensor::from_f32_array([2], [1.0, 2.0])
  let bias = @tensor.Tensor::from_f32_array([2], [0.0, 0.5])
  let mean = @tensor.Tensor::from_f32_array([2], [0.0, 0.0])
  let variance = @tensor.Tensor::from_f32_array([2], [1.0, 4.0])
  let c_zero = @tensor.Tensor::from_f32([], FixedArray::make(1, (0.0 : Float)))
  let c_six = @tensor.Tensor::from_f32([], FixedArray::make(1, (6.0 : Float)))

  let inits : Map[String, @tensor.Tensor] = Map([
    ("w", w_conv),
    ("s", scale),
    ("b", bias),
    ("m", mean),
    ("v", variance),
  ])
  let g : @nn_ir.IrGraph = {
    name: "mobilenet_block",
    inputs: ["x"],
    outputs: ["y"],
    initializers: inits,
    nodes: [
      {
        name: "c_min",
        op: @nn_ir.Constant(value=c_zero),
        inputs: [],
        outputs: ["min_v"],
      },
      {
        name: "c_max",
        op: @nn_ir.Constant(value=c_six),
        inputs: [],
        outputs: ["max_v"],
      },
      {
        name: "conv0",
        op: @nn_ir.Conv(
          kernel_shape=Some([3, 3]),
          strides=[1, 1],
          pads=[1, 1, 1, 1],
          dilations=[1, 1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["x", "w"],
        outputs: ["t_conv"],
      },
      {
        name: "bn0",
        op: @nn_ir.BatchNormalization(epsilon=(0.0 : Float)),
        inputs: ["t_conv", "s", "b", "m", "v"],
        outputs: ["t_bn"],
      },
      {
        name: "clip0",
        op: @nn_ir.Clip(min=None, max=None),
        inputs: ["t_bn", "min_v", "max_v"],
        outputs: ["y"],
      },
    ],
  }

  let (opt_g, stats) = @opt.optimize_graph(g)
  assert_eq(stats.lifted_constants, 2)
  assert_eq(stats.folded_conv_bn, 1)
  assert_eq(stats.folded_conv_activation, 1)
  assert_eq(opt_g.nodes.length(), 1)
  assert_eq(opt_g.nodes[0].op.name(), "Conv+Clip")
  match opt_g.nodes[0].op {
    @nn_ir.Conv(activation=@nn_ir.ClipAct(min~, max~), ..) => {
      assert_eq(min, (0.0 : Float))
      assert_eq(max, (6.0 : Float))
    }
    _ => fail("expected fused Conv+Clip")
  }
}

///|
/// 验证 9 节点 PyTorch 分解版 LayerNorm 子图被自动模式融合为单个 `@nn_ir.LayerNormalization` 算子。
test "pass_fold_layernorm fuses 9-node PyTorch decomposed LayerNorm" {
  let c_two = @tensor.Tensor::from_f32_array([], [2.0])
  let c_eps = @tensor.Tensor::from_f32_array([], [1.0e-12])
  let gamma = @tensor.Tensor::from_f32_array([4], [1.0, 2.0, 0.5, 1.5])
  let beta = @tensor.Tensor::from_f32_array([4], [0.1, -0.1, 0.2, 0.0])
  let g : @nn_ir.IrGraph = {
    name: "decomposed_ln",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([
      ("two", c_two),
      ("eps", c_eps),
      ("gamma", gamma),
      ("beta", beta),
    ]),
    nodes: [
      {
        name: "m1",
        op: @nn_ir.ReduceMean(
          axes=Some([-1]),
          keepdims=true,
          noop_with_empty_axes=false,
        ),
        inputs: ["x"],
        outputs: ["mean"],
      },
      { name: "s1", op: @nn_ir.Sub, inputs: ["x", "mean"], outputs: ["diff"], },
      { name: "p1", op: @nn_ir.Pow, inputs: ["diff", "two"], outputs: ["sq"], },
      {
        name: "m2",
        op: @nn_ir.ReduceMean(
          axes=Some([-1]),
          keepdims=true,
          noop_with_empty_axes=false,
        ),
        inputs: ["sq"],
        outputs: ["var"],
      },
      {
        name: "a1",
        op: @nn_ir.Add,
        inputs: ["var", "eps"],
        outputs: ["var_eps"],
      },
      { name: "sq1", op: @nn_ir.Sqrt, inputs: ["var_eps"], outputs: ["std"], },
      {
        name: "d1",
        op: @nn_ir.Div,
        inputs: ["diff", "std"],
        outputs: ["norm"],
      },
      {
        name: "mul1",
        op: @nn_ir.Mul,
        inputs: ["norm", "gamma"],
        outputs: ["scaled"],
      },
      {
        name: "add2",
        op: @nn_ir.Add,
        inputs: ["scaled", "beta"],
        outputs: ["y"],
      },
    ],
  }

  let (opt_g, stats) = @opt.optimize_graph(g)
  assert_eq(stats.folded_layernorm, 1)
  assert_eq(opt_g.nodes.length(), 1)
  assert_eq(opt_g.nodes[0].inputs, ["x", "gamma", "beta"])
  assert_eq(opt_g.nodes[0].outputs, ["y"])
  match opt_g.nodes[0].op {
    @nn_ir.LayerNormalization(axis~, epsilon~) => {
      assert_eq(axis, -1)
      assert_true((epsilon - (1.0e-12 : Float)).abs() < (1.0e-15 : Float))
    }
    _ => fail("expected fused LayerNormalization")
  }
}
