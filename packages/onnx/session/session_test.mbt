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
/// 验证 SIMD 会话自动缓存 MatMul/Gemm/Conv 静态预打包权重、`run_single_profiled` 剖析报告，
/// 以及连续推理下 `BufferPool` 100% 零未命中复用且不污染先前返回的输出张量。
test "Session caches PackedMatrixB/PackedConvWeight static weights and achieves 100% BufferPool reuse" {
  let mlp_simd = @session.Session::from_file(
    "../../tests/fixtures/mnist_mlp.onnx",
    backend=@session.Simd,
  )
  // MNIST MLP 包含 3 个全连接 Gemm 权重，均应在初始化期预打包
  assert_eq(mlp_simd.packed_weight_count(), 3)
  assert_eq(mlp_simd.packed_conv_weight_count(), 0)

  let cnn_simd = @session.Session::from_file(
    "../../tests/fixtures/mnist_cnn.onnx",
    backend=@session.Simd,
  )
  // MNIST CNN 包含 1 个全连接 MatMul 权重 + 2 个 Conv 静态权重
  assert_eq(cnn_simd.packed_weight_count(), 3)
  assert_eq(cnn_simd.packed_conv_weight_count(), 2)

  let cnn_in = @session.load_f32_file(
    "../../tests/fixtures/mnist_cnn_input.f32",
    [1, 1, 28, 28],
  )
  let cnn_gold = @session.load_f32_file(
    "../../tests/fixtures/mnist_cnn_golden.f32",
    [1, 10],
  )
  // 第 1 次前向推理：冷启动填充 BufferPool
  let (out1, profile) = cnn_simd.run_single_profiled(cnn_in)
  assert_eq(out1.shape(), [1, 10])
  assert_eq(profile.ops.length(), 9)
  assert_true(profile.total_us >= 0.0)
  assert_true(profile.format_table().contains("Conv"))
  let stats_after_1 = cnn_simd.pool_stats()
  assert_true(stats_after_1.misses > 0)
  assert_true(stats_after_1.pooled_f32_buffers > 0)

  // 第 2 次前向推理：全部中间张量与 Conv im2col 工作区应 100% 命中 BufferPool（零新增 miss）
  let out2 = cnn_simd.run_single(cnn_in)
  let stats_after_2 = cnn_simd.pool_stats()
  assert_eq(stats_after_2.misses, stats_after_1.misses)
  assert_true(stats_after_2.hits > stats_after_1.hits)

  // 用全零输入执行第 3 次推理，验证 out1/out2 结果未被后续池复用篡改
  let zero_in = @tensor.Tensor::zeros_f32([1, 1, 28, 28])
  let _out3 = cnn_simd.run_single(zero_in)
  assert_true(@session.max_abs_diff_f32(out1, cnn_gold) <= (1.0e-5 : Float))
  assert_true(@session.max_abs_diff_f32(out2, cnn_gold) <= (1.0e-5 : Float))

  cnn_simd.clear_pool()
  assert_eq(cnn_simd.pool_stats().pooled_f32_buffers, 0)
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
