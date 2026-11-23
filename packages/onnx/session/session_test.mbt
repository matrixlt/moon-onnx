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

///|
/// 验证 SIMD 会话自动缓存静态权重预打包矩阵，并且 `run_single_profiled` 准确记录全部节点耗时。
test "Session caches PackedMatrixB static weights and produces per-op RunProfile" {
  let mlp_simd = @session.Session::from_file(
    "../../tests/fixtures/mnist_mlp.onnx",
    backend=@session.Simd,
  )
  // MNIST MLP 包含 3 个全连接 Gemm 权重，均应在初始化期预打包
  assert_eq(mlp_simd.packed_weight_count(), 3)

  let cnn_simd = @session.Session::from_file(
    "../../tests/fixtures/mnist_cnn.onnx",
    backend=@session.Simd,
  )
  // MNIST CNN 包含 1 个全连接 MatMul 权重
  assert_eq(cnn_simd.packed_weight_count(), 1)

  let cnn_in = @session.load_f32_file(
    "../../tests/fixtures/mnist_cnn_input.f32",
    [1, 1, 28, 28],
  )
  let (out, profile) = cnn_simd.run_single_profiled(cnn_in)
  assert_eq(out.shape(), [1, 10])
  assert_eq(profile.ops.length(), 9)
  assert_true(profile.total_us >= 0.0)
  assert_true(profile.format_table().contains("Conv"))
}
