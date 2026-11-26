///|
/// 端到端前端集成测试：验证真实 MNIST MLP 与 CNN `.onnx` 文件能完整 Lowering 为强类型 `nn_ir::IrGraph`。
test "lower_onnx_model succeeds on real MNIST MLP and CNN fixtures" {
  let mlp = @onnx_fs.load_model_from_file("fixtures/mnist_mlp.onnx")
  let mlp_ir = @frontend.lower_onnx_model(mlp)
  assert_eq(mlp_ir.nodes.length(), 7)
  assert_eq(mlp_ir.nodes[0].op.name(), "Flatten")
  assert_eq(mlp_ir.nodes[6].op.name(), "Softmax")

  let cnn = @onnx_fs.load_model_from_file("fixtures/mnist_cnn.onnx")
  let cnn_ir = @frontend.lower_onnx_model(cnn)
  assert_eq(cnn_ir.nodes.length(), 11)
  assert_eq(cnn_ir.nodes[0].op.name(), "Conv")
  assert_eq(cnn_ir.nodes[9].op.name(), "MatMul")
}

///|
/// 端到端集成测试：在 Scalar 与 V128 SIMD 双后端下验证真实 MNIST MLP 与 CNN 均与 ONNX Runtime 黄金结果严格对齐。
test "end-to-end MNIST MLP and CNN match ORT on Scalar and V128 SIMD backends" {
  let backends = [@session.Scalar, @session.Simd]
  for i = 0; i < backends.length(); i = i + 1 {
    let backend = backends[i]

    let mlp = @onnx_fs.session_from_file("fixtures/mnist_mlp.onnx", backend~)
    let mlp_in = @onnx_fs.load_f32_file("fixtures/mnist_mlp_input.f32", [
      1, 1, 28, 28,
    ])
    let mlp_gold = @onnx_fs.load_f32_file("fixtures/mnist_mlp_golden.f32", [
      1, 10,
    ])
    let mlp_out = mlp.run_single(mlp_in)
    assert_true(
      @session.max_abs_diff_f32(mlp_out, mlp_gold) <= (1.0e-5 : Float),
    )
    let (mlp_pred, _) = mlp_out.argmax_f32()
    let (mlp_ort, _) = mlp_gold.argmax_f32()
    assert_eq(mlp_pred, mlp_ort)

    let cnn = @onnx_fs.session_from_file("fixtures/mnist_cnn.onnx", backend~)
    let cnn_in = @onnx_fs.load_f32_file("fixtures/mnist_cnn_input.f32", [
      1, 1, 28, 28,
    ])
    let cnn_gold = @onnx_fs.load_f32_file("fixtures/mnist_cnn_golden.f32", [
      1, 10,
    ])
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
/// 验证真实 MNIST 模型下静态权重预打包数量、逐算子剖析与 100% BufferPool 内存复用。
test "real MNIST fixtures prepack static weights and achieve 100% BufferPool reuse" {
  let mlp_simd = @onnx_fs.session_from_file(
    "fixtures/mnist_mlp.onnx",
    backend=@session.Simd,
  )
  assert_eq(mlp_simd.packed_weight_count(), 3)
  assert_eq(mlp_simd.packed_conv_weight_count(), 0)

  let cnn_simd = @onnx_fs.session_from_file(
    "fixtures/mnist_cnn.onnx",
    backend=@session.Simd,
  )
  assert_eq(cnn_simd.packed_weight_count(), 3)
  assert_eq(cnn_simd.packed_conv_weight_count(), 2)

  let cnn_in = @onnx_fs.load_f32_file("fixtures/mnist_cnn_input.f32", [
    1, 1, 28, 28,
  ])
  let cnn_gold = @onnx_fs.load_f32_file("fixtures/mnist_cnn_golden.f32", [1, 10])

  let (out1, profile) = cnn_simd.run_single_profiled(cnn_in)
  assert_eq(out1.shape(), [1, 10])
  // 优化后 2 组 Conv+bias+Relu 均融合为单个 Conv+Relu 节点（11 -> 7 节点）
  assert_eq(profile.ops.length(), 7)
  assert_true(profile.total_us >= 0.0)
  assert_true(profile.format_table().contains("Conv+Relu"))
  let stats1 = cnn_simd.pool_stats()
  assert_true(stats1.misses > 0)

  // 第 2 次推理应 100% 命中 BufferPool（零新增 miss）
  let out2 = cnn_simd.run_single(cnn_in)
  let stats2 = cnn_simd.pool_stats()
  assert_eq(stats2.misses, stats1.misses)
  assert_true(stats2.hits > stats1.hits)

  // 第 3 次用全零输入推理，验证 out1/out2 不受缓冲复用影响
  let zero_in = @tensor.Tensor::zeros_f32([1, 1, 28, 28])
  let _out3 = cnn_simd.run_single(zero_in)
  assert_true(@session.max_abs_diff_f32(out1, cnn_gold) <= (1.0e-5 : Float))
  assert_true(@session.max_abs_diff_f32(out2, cnn_gold) <= (1.0e-5 : Float))
}
