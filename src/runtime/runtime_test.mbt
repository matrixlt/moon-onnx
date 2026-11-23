///|
/// 端到端集成测试：加载真实 mnist_mlp.onnx 与测试输入，推理结果与 ONNX Runtime 黄金输出逐元素对齐。
test "end-to-end MNIST MLP matches ONNX Runtime within 1e-5" {
  let session = @runtime.Session::from_file("tests/fixtures/mnist_mlp.onnx")
  let input = @runtime.load_f32_file("tests/fixtures/mnist_mlp_input.f32", [
    1, 1, 28, 28,
  ])
  let golden = @runtime.load_f32_file("tests/fixtures/mnist_mlp_golden.f32", [
    1, 10,
  ])

  let output = session.run_single(input)
  assert_eq(output.shape(), [1, 10])

  let diff = @runtime.max_abs_diff_f32(output, golden)
  assert_true(diff <= (1.0e-5 : Float))

  let (pred_cls, _) = output.argmax_f32()
  let (ort_cls, _) = golden.argmax_f32()
  assert_eq(pred_cls, ort_cls)
}
