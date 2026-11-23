///|
/// 端到端集成测试：在 Scalar 与 V128 SIMD 双后端下验证 MNIST MLP 与 MNIST CNN 均与 ONNX Runtime 严格对齐。
test "end-to-end MNIST MLP and CNN match ORT on Scalar and V128 SIMD backends" {
  let backends = [@session.Scalar, @session.Simd]
  for i = 0; i < backends.length(); i = i + 1 {
    let backend = backends[i]

    let mlp = @session.Session::from_file(
      "../../tests/fixtures/mnist_mlp.onnx",
      backend~,
    )
    let mlp_in = @session.load_f32_file(
      "../../tests/fixtures/mnist_mlp_input.f32",
      [1, 1, 28, 28],
    )
    let mlp_gold = @session.load_f32_file(
      "../../tests/fixtures/mnist_mlp_golden.f32",
      [1, 10],
    )
    let mlp_out = mlp.run_single(mlp_in)
    assert_true(
      @session.max_abs_diff_f32(mlp_out, mlp_gold) <= (1.0e-5 : Float),
    )
    let (mlp_pred, _) = mlp_out.argmax_f32()
    let (mlp_ort, _) = mlp_gold.argmax_f32()
    assert_eq(mlp_pred, mlp_ort)

    let cnn = @session.Session::from_file(
      "../../tests/fixtures/mnist_cnn.onnx",
      backend~,
    )
    let cnn_in = @session.load_f32_file(
      "../../tests/fixtures/mnist_cnn_input.f32",
      [1, 1, 28, 28],
    )
    let cnn_gold = @session.load_f32_file(
      "../../tests/fixtures/mnist_cnn_golden.f32",
      [1, 10],
    )
    let cnn_out = cnn.run_single(cnn_in)
    assert_true(
      @session.max_abs_diff_f32(cnn_out, cnn_gold) <= (1.0e-5 : Float),
    )
    let (cnn_pred, _) = cnn_out.argmax_f32()
    let (cnn_ort, _) = cnn_gold.argmax_f32()
    assert_eq(cnn_pred, cnn_ort)
  }
}
