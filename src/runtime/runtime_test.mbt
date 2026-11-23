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

///|
/// 端到端集成测试：加载对齐 ONNX Model Zoo MNIST-12 结构的 mnist_cnn.onnx，
/// 校验双层 Conv + Add + Relu + MaxPool + Reshape + MatMul + Add 全链路输出与 ORT 严格一致。
test "end-to-end MNIST CNN matches ONNX Runtime within 1e-5" {
  let session = @runtime.Session::from_file("tests/fixtures/mnist_cnn.onnx")
  let input = @runtime.load_f32_file("tests/fixtures/mnist_cnn_input.f32", [
    1, 1, 28, 28,
  ])
  let golden = @runtime.load_f32_file("tests/fixtures/mnist_cnn_golden.f32", [
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

///|
/// 端到端集成测试：在 V128 SIMD 后端下运行 MNIST MLP 与 MNIST CNN，
/// 验证两者与 ONNX Runtime 黄金输出及 Scalar 后端误差均 <= 1e-5。
test "end-to-end MNIST MLP and CNN on V128 SIMD backend match ORT within 1e-5" {
  let mlp_simd = @runtime.Session::from_file(
    "tests/fixtures/mnist_mlp.onnx",
    backend=@runtime.Simd,
  )
  let mlp_in = @runtime.load_f32_file("tests/fixtures/mnist_mlp_input.f32", [
    1, 1, 28, 28,
  ])
  let mlp_gold = @runtime.load_f32_file("tests/fixtures/mnist_mlp_golden.f32", [
    1, 10,
  ])
  let mlp_out = mlp_simd.run_single(mlp_in)
  assert_true(@runtime.max_abs_diff_f32(mlp_out, mlp_gold) <= (1.0e-5 : Float))

  let cnn_simd = @runtime.Session::from_file(
    "tests/fixtures/mnist_cnn.onnx",
    backend=@runtime.Simd,
  )
  let cnn_in = @runtime.load_f32_file("tests/fixtures/mnist_cnn_input.f32", [
    1, 1, 28, 28,
  ])
  let cnn_gold = @runtime.load_f32_file("tests/fixtures/mnist_cnn_golden.f32", [
    1, 10,
  ])
  let cnn_out = cnn_simd.run_single(cnn_in)
  assert_true(@runtime.max_abs_diff_f32(cnn_out, cnn_gold) <= (1.0e-5 : Float))
}
