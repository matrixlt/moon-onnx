///|
/// 测试 Constant 提升、Identity 消除与 DCE 死节点清除。
test "optimize_graph lifts constants, eliminates identity and dead nodes" {
  let c_val = @tensor.Tensor::from_f32_array([2], [1.0, 2.0])
  let g : @ir.IrGraph = {
    name: "opt_basic",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([]),
    nodes: [
      { name: "c0", op: @ir.Constant(value=c_val), inputs: [], outputs: ["c"], },
      { name: "id0", op: @ir.Identity, inputs: ["x"], outputs: ["x_id"], },
      { name: "dead", op: @ir.Relu, inputs: ["x_id"], outputs: ["unused"], },
      { name: "add0", op: @ir.Add, inputs: ["x_id", "c"], outputs: ["y"], },
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
/// 测试 Conv + BatchNormalization 静态折叠前后输出数值完全等价。
test "optimize_graph folds Conv + BatchNormalization with numerical parity" {
  let x = @tensor.Tensor::from_f32_array([1, 1, 3, 3], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  let w = @tensor.Tensor::from_f32_array([2, 1, 2, 2], [
    0.5, -0.25, 1.0, 0.75, -0.5, 0.5, 0.25, 1.25,
  ])
  let scale = @tensor.Tensor::from_f32_array([2], [1.5, 0.8])
  let bias = @tensor.Tensor::from_f32_array([2], [0.2, -0.3])
  let mean = @tensor.Tensor::from_f32_array([2], [2.0, 4.0])
  let variance = @tensor.Tensor::from_f32_array([2], [1.2, 2.5])

  // 未折叠的参考结果：Conv -> BatchNorm
  let conv_ref = @ops.conv2d_f32(
    x,
    w,
    None,
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @ir.NotSet,
  )
  let bn_ref = @ops.batch_norm_f32(
    conv_ref,
    scale,
    bias,
    mean,
    variance,
    epsilon=(1.0e-5 : Float),
  )

  // 构造 IR 并执行图优化折叠
  let inits : Map[String, @tensor.Tensor] = Map([
    ("w", w),
    ("s", scale),
    ("b", bias),
    ("m", mean),
    ("v", variance),
  ])
  let g : @ir.IrGraph = {
    name: "conv_bn",
    inputs: ["x"],
    outputs: ["y"],
    initializers: inits,
    nodes: [
      {
        name: "conv0",
        op: @ir.Conv(
          kernel_shape=Some([2, 2]),
          strides=[1, 1],
          pads=[0, 0, 0, 0],
          dilations=[1, 1],
          group=1,
          auto_pad=@ir.NotSet,
        ),
        inputs: ["x", "w"],
        outputs: ["c_out"],
      },
      {
        name: "bn0",
        op: @ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["c_out", "s", "b", "m", "v"],
        outputs: ["y"],
      },
    ],
  }

  let (opt_g, stats) = @opt.optimize_graph(g)
  assert_eq(stats.folded_conv_bn, 1)
  assert_eq(opt_g.nodes.length(), 1)

  let fused_w = opt_g.initializers.get(opt_g.nodes[0].inputs[1]).unwrap()
  let fused_b = opt_g.initializers.get(opt_g.nodes[0].inputs[2]).unwrap()
  let fused_out = @ops.conv2d_f32(
    x,
    fused_w,
    Some(fused_b),
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @ir.NotSet,
  )

  let a = fused_out.to_f32_buffer()
  let b = bn_ref.to_f32_buffer()
  for i = 0; i < a.length(); i = i + 1 {
    assert_true((a[i] - b[i]).abs() < (1.0e-5 : Float))
  }
}

///|
/// 测试真实 mnist_cnn.onnx 经优化后两处 Conv+Add(bias) 自动折叠（11 节点 -> 9 节点）。
test "optimize_graph folds Conv + channel bias Add in mnist_cnn.onnx" {
  let m = @model.load_model_from_file("tests/fixtures/mnist_cnn.onnx")
  let g = @ir.lower_model(m)
  assert_eq(g.nodes.length(), 11)

  let (opt_g, stats) = @opt.optimize_graph(g)
  assert_eq(stats.folded_conv_bias, 2)
  assert_eq(opt_g.nodes.length(), 9)
}
