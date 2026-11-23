///|
/// 验证真实 MNIST MLP 与 CNN ONNX 模型能完整 Lowering 为强类型 `nn_ir::IrGraph`。
test "lower_onnx_model succeeds on real MNIST MLP and CNN fixtures" {
  let mlp = @onnx_fs.load_model_from_file("../../tests/fixtures/mnist_mlp.onnx")
  let mlp_ir = @frontend.lower_onnx_model(mlp)
  assert_eq(mlp_ir.nodes.length(), 7)
  assert_eq(mlp_ir.nodes[0].op.name(), "Flatten")
  assert_eq(mlp_ir.nodes[6].op.name(), "Softmax")

  let cnn = @onnx_fs.load_model_from_file("../../tests/fixtures/mnist_cnn.onnx")
  let cnn_ir = @frontend.lower_onnx_model(cnn)
  assert_eq(cnn_ir.nodes.length(), 11)
  assert_eq(cnn_ir.nodes[0].op.name(), "Conv")
  assert_eq(cnn_ir.nodes[9].op.name(), "MatMul")
}
