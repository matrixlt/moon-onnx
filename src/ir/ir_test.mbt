///|
/// 验证真实生成的 mnist_mlp.onnx 与 mnist_cnn.onnx 能完整 Lowering 为强类型 IrGraph。
test "lower_model succeeds on real MNIST MLP and CNN fixtures" {
  let mlp = @model.load_model_from_file("tests/fixtures/mnist_mlp.onnx")
  let mlp_ir = @ir.lower_model(mlp)
  assert_eq(mlp_ir.nodes.length(), 7)
  assert_eq(mlp_ir.nodes[0].op.name(), "Flatten")
  assert_eq(mlp_ir.nodes[1].op.name(), "Gemm")
  assert_eq(mlp_ir.nodes[6].op.name(), "Softmax")

  let cnn = @model.load_model_from_file("tests/fixtures/mnist_cnn.onnx")
  let cnn_ir = @ir.lower_model(cnn)
  assert_eq(cnn_ir.nodes.length(), 11)
  assert_eq(cnn_ir.nodes[0].op.name(), "Conv")
  assert_eq(cnn_ir.nodes[3].op.name(), "MaxPool")
  assert_eq(cnn_ir.nodes[8].op.name(), "Reshape")
  assert_eq(cnn_ir.nodes[9].op.name(), "MatMul")
}

///|
/// 验证 topo_sort_nodes 能自动纠正乱序节点排列。
test "topo_sort_nodes reorders out-of-order nodes" {
  let n_relu : @ir.IrNode = {
    name: "relu_after",
    op: @ir.Relu,
    inputs: ["mid"],
    outputs: ["out"],
  }
  let n_add : @ir.IrNode = {
    name: "add_before",
    op: @ir.Add,
    inputs: ["x", "x"],
    outputs: ["mid"],
  }
  let sorted = @ir.topo_sort_nodes([n_relu, n_add], ["x"], Map([]))
  assert_eq(sorted[0].name, "add_before")
  assert_eq(sorted[1].name, "relu_after")
}

///|
/// 验证存在环路的计算图会被准确检测并抛出 CyclicGraph 错误。
test "topo_sort_nodes detects cycles" {
  let n1 : @ir.IrNode = {
    name: "n1",
    op: @ir.Relu,
    inputs: ["b"],
    outputs: ["a"],
  }
  let n2 : @ir.IrNode = {
    name: "n2",
    op: @ir.Relu,
    inputs: ["a"],
    outputs: ["b"],
  }
  let mut caught_cycle = false
  try @ir.topo_sort_nodes([n1, n2], [], Map([])) |> ignore catch {
    @ir.CyclicGraph(_) => caught_cycle = true
    _ => ()
  }
  assert_true(caught_cycle)
}
