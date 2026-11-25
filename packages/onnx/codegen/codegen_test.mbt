///|
/// 构造包含 `Conv+BN+Clip(ReLU6) -> GAP -> Flatten -> Gemm -> Softmax` 的视觉测试图。
fn build_tiny_vision_graph() -> @nn_ir.IrGraph raise @tensor.TensorError {
  let w_conv = @tensor.Tensor::from_f32(
    [4, 2, 3, 3],
    FixedArray::makei(4 * 2 * 9, fn(i) {
      Float::from_int(i % 7 - 3) * (0.1 : Float)
    }),
  )
  let bn_s = @tensor.Tensor::full_f32([4], (1.0 : Float))
  let bn_b = @tensor.Tensor::full_f32([4], (0.2 : Float))
  let bn_m = @tensor.Tensor::full_f32([4], (0.0 : Float))
  let bn_v = @tensor.Tensor::full_f32([4], (1.0 : Float))
  let w_fc = @tensor.Tensor::from_f32(
    [3, 4],
    FixedArray::makei(12, fn(i) { Float::from_int(i - 5) * (0.1 : Float) }),
  )
  let b_fc = @tensor.Tensor::from_f32_array([3], [0.1, 0.0, -0.1])
  let c0 = @tensor.Tensor::from_f32([], FixedArray::make(1, (0.0 : Float)))
  let c6 = @tensor.Tensor::from_f32([], FixedArray::make(1, (6.0 : Float)))

  {
    name: "tiny_vision_aot",
    inputs: ["x"],
    outputs: ["probs"],
    initializers: Map([
      ("w_conv", w_conv),
      ("bn_s", bn_s),
      ("bn_b", bn_b),
      ("bn_m", bn_m),
      ("bn_v", bn_v),
      ("w_fc", w_fc),
      ("b_fc", b_fc),
    ]),
    nodes: [
      {
        name: "c0",
        op: @nn_ir.Constant(value=c0),
        inputs: [],
        outputs: ["k0"],
      },
      {
        name: "c6",
        op: @nn_ir.Constant(value=c6),
        inputs: [],
        outputs: ["k6"],
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
        inputs: ["x", "w_conv"],
        outputs: ["c_out"],
      },
      {
        name: "bn0",
        op: @nn_ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["c_out", "bn_s", "bn_b", "bn_m", "bn_v"],
        outputs: ["bn_out"],
      },
      {
        name: "clip0",
        op: @nn_ir.Clip(min=None, max=None),
        inputs: ["bn_out", "k0", "k6"],
        outputs: ["act_out"],
      },
      {
        name: "gap0",
        op: @nn_ir.GlobalAveragePool,
        inputs: ["act_out"],
        outputs: ["gap_out"],
      },
      {
        name: "flat0",
        op: @nn_ir.Flatten(axis=1),
        inputs: ["gap_out"],
        outputs: ["flat_out"],
      },
      {
        name: "fc0",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["flat_out", "w_fc", "b_fc"],
        outputs: ["logits"],
      },
      {
        name: "sm0",
        op: @nn_ir.Softmax(axis=-1),
        inputs: ["logits"],
        outputs: ["probs"],
      },
    ],
  }
}

///|
/// 验证 `onnx2mbt` v2 默认 Arena 零分配模式：
/// - 零 `matrixlt/tensor`、零 `matrixlt/nn_ir`、零 `matrixlt/onnx/ops`、零 `BufferPool` 依赖；
/// - 静态 Arena 偏移复用 + `--embed-weights` 内嵌权重 + 目标平台自适应（`TargetWasm` vs `TargetWasmGc`）。
test "compile_graph_to_mbt Arena mode emits zero-allocation, zero-Tensor code with embedded weights" {
  let graph = build_tiny_vision_graph()
  let input_shapes : Map[String, Array[Int32]] = Map([("x", [1, 2, 6, 6])])

  // 1. TargetWasm + Arena + embed_weights
  let art_wasm = @codegen.compile_graph_to_mbt(graph, input_shapes, config={
    struct_name: "TinyVisionWasm",
    use_simd: true,
    optimize: true,
    target: @codegen.TargetWasm,
    mode: @codegen.Arena,
    embed_weights: true,
  })
  assert_eq(art_wasm.mode_used, @codegen.Arena)
  assert_eq(art_wasm.node_count, 5)
  assert_eq(art_wasm.weights.length(), 4)
  assert_eq(art_wasm.weights_bin.length(), 91 * 4)
  assert_true(art_wasm.arena_f32_elems > 0)
  assert_true(art_wasm.arena_f32_elems <= art_wasm.arena_f32_naive_elems)
  assert_true(art_wasm.mbt_source.contains("const WEIGHTS_DATA : Bytes = b\""))
  assert_true(art_wasm.mbt_source.contains("pub fn TinyVisionWasm::new()"))
  assert_true(
    art_wasm.mbt_source.contains("pub fn TinyVisionWasm::forward_into("),
  )
  assert_true(
    art_wasm.mbt_source.contains("@gemm.matmul_2d_prepacked_nchw_simd_f32"),
  )
  assert_true(!art_wasm.pkg_config.contains("matrixlt/tensor"))
  assert_true(!art_wasm.pkg_config.contains("matrixlt/onnx/ops"))
  assert_true(!art_wasm.mbt_source.contains("BufferPool"))
  assert_true(!art_wasm.mbt_source.contains("Map["))

  // 2. TargetWasmGc 自动切至原生 Float 数组路径，避开 GC 字节数组 v128_load 退化
  let art_gc = @codegen.compile_graph_to_mbt(graph, input_shapes, config={
    struct_name: "TinyVisionGc",
    use_simd: true,
    optimize: true,
    target: @codegen.TargetWasmGc,
    mode: @codegen.Arena,
    embed_weights: false,
  })
  assert_eq(art_gc.mode_used, @codegen.Arena)
  assert_true(art_gc.mbt_source.contains("conv_nchw_scalar_f32"))
  assert_true(art_gc.mbt_source.contains("@gemm.matmul_2d_scalar_f32"))
  assert_true(!art_gc.mbt_source.contains("PackedMatrixB"))

  // 3. Tensor 兼容模式仍正常工作
  let art_tensor = @codegen.compile_graph_to_mbt(graph, input_shapes, config={
    struct_name: "TinyVisionTensor",
    use_simd: true,
    optimize: true,
    target: @codegen.TargetWasm,
    mode: @codegen.Tensor,
    embed_weights: false,
  })
  assert_eq(art_tensor.mode_used, @codegen.Tensor)
  assert_true(art_tensor.mbt_source.contains("@ops.conv2d_prepacked_f32"))
}

///|
/// 验证 `onnx2mbt` v2 对 INT8 量化计算图（`QuantizeLinear -> QLinearMatMul -> DequantizeLinear`）的 AOT 编译：
/// 确认权重以 1 字节紧凑打包（4× 体积压缩）、生成零拷贝预转置 V128 `i32x4.dot` 调用与双 Arena。
test "compile_graph_to_mbt compiles INT8 quantized QLinearMatMul graph in Arena mode" {
  let w_q = @tensor.Tensor::from_i8(
    [16, 4],
    FixedArray::makei(64, fn(i) { (i % 100 - 50).to_byte() }),
  )
  let a_s = @tensor.Tensor::from_f32([], FixedArray::make(1, (0.02 : Float)))
  let a_zp = @tensor.Tensor::from_i8([], FixedArray::make(1, b'\x00'))
  let b_s = @tensor.Tensor::from_f32([], FixedArray::make(1, (0.01 : Float)))
  let b_zp = @tensor.Tensor::from_i8([], FixedArray::make(1, b'\x00'))
  let y_s = @tensor.Tensor::from_f32([], FixedArray::make(1, (0.05 : Float)))
  let y_zp = @tensor.Tensor::from_i8([], FixedArray::make(1, b'\x00'))

  let q_graph : @nn_ir.IrGraph = {
    name: "int8_aot_mlp",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([
      ("a_s", a_s),
      ("a_zp", a_zp),
      ("w_q", w_q),
      ("b_s", b_s),
      ("b_zp", b_zp),
      ("y_s", y_s),
      ("y_zp", y_zp),
    ]),
    nodes: [
      {
        name: "q_in",
        op: @nn_ir.QuantizeLinear(axis=1),
        inputs: ["x", "a_s", "a_zp"],
        outputs: ["x_q"],
      },
      {
        name: "qmm",
        op: @nn_ir.QLinearMatMul,
        inputs: ["x_q", "a_s", "a_zp", "w_q", "b_s", "b_zp", "y_s", "y_zp"],
        outputs: ["y_q"],
      },
      {
        name: "dq_out",
        op: @nn_ir.DequantizeLinear(axis=1),
        inputs: ["y_q", "y_s", "y_zp"],
        outputs: ["y"],
      },
    ],
  }

  let art = @codegen.compile_graph_to_mbt(q_graph, Map([("x", [2, 16])]), config={
    struct_name: "Int8Mlp",
    use_simd: true,
    optimize: true,
    target: @codegen.TargetWasm,
    mode: @codegen.Arena,
    embed_weights: true,
  })
  assert_eq(art.mode_used, @codegen.Arena)
  // 权重仅 64 字节（1 byte/element，相对 Float32 的 256 字节实现 4x 压缩）
  assert_eq(art.weights_bin.length(), 64)
  assert_true(art.arena_byte_elems >= 40)
  assert_true(art.arena_f32_elems >= 8)
  assert_true(
    art.mbt_source.contains("@quant.qlinear_matmul_i8_pretransposed_v128"),
  )
  assert_true(art.mbt_source.contains("@quant.transpose_bytes_kn_to_nk"))
  assert_true(art.pkg_config.contains("matrixlt/ml_kernels/quant"))
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}

///|
/// 验证 `onnx2mbt` v2 Arena 零分配模式原生支持 3D `Conv1D`（含 `Conv1D+BN+Relu` 融合与 Depthwise 1D）。
test "compile_graph_to_mbt compiles 3D Conv1D audio classifier in zero-allocation Arena mode" {
  let w1 = @tensor.Tensor::from_f32(
    [8, 1, 5],
    FixedArray::makei(40, fn(i) { Float::from_int(i % 7 - 3) * (0.1 : Float) }),
  )
  let bn_s = @tensor.Tensor::full_f32([8], (1.0 : Float))
  let bn_b = @tensor.Tensor::full_f32([8], (0.0 : Float))
  let bn_m = @tensor.Tensor::full_f32([8], (0.0 : Float))
  let bn_v = @tensor.Tensor::full_f32([8], (1.0 : Float))
  let w_dw = @tensor.Tensor::from_f32(
    [8, 1, 3],
    FixedArray::makei(24, fn(i) { Float::from_int(i % 5 - 2) * (0.1 : Float) }),
  )
  let w_fc = @tensor.Tensor::from_f32(
    [2, 8],
    FixedArray::makei(16, fn(i) { Float::from_int(i - 8) * (0.1 : Float) }),
  )

  let g : @nn_ir.IrGraph = {
    name: "audio_kws_conv1d",
    inputs: ["audio"],
    outputs: ["probs"],
    initializers: Map([
      ("w1", w1),
      ("bn_s", bn_s),
      ("bn_b", bn_b),
      ("bn_m", bn_m),
      ("bn_v", bn_v),
      ("w_dw", w_dw),
      ("w_fc", w_fc),
    ]),
    nodes: [
      {
        name: "c1",
        op: @nn_ir.Conv(
          kernel_shape=Some([5]),
          strides=[2],
          pads=[2, 2],
          dilations=[1],
          group=1,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.NoAct,
        ),
        inputs: ["audio", "w1"],
        outputs: ["t0"],
      },
      {
        name: "bn1",
        op: @nn_ir.BatchNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["t0", "bn_s", "bn_b", "bn_m", "bn_v"],
        outputs: ["t1"],
      },
      { name: "r1", op: @nn_ir.Relu, inputs: ["t1"], outputs: ["t2"], },
      {
        name: "dw1",
        op: @nn_ir.Conv(
          kernel_shape=Some([3]),
          strides=[1],
          pads=[1, 1],
          dilations=[1],
          group=8,
          auto_pad=@nn_ir.NotSet,
          activation=@nn_ir.ReluAct,
        ),
        inputs: ["t2", "w_dw"],
        outputs: ["t3"],
      },
      {
        name: "gap",
        op: @nn_ir.GlobalAveragePool,
        inputs: ["t3"],
        outputs: ["t4"],
      },
      {
        name: "flat",
        op: @nn_ir.Flatten(axis=1),
        inputs: ["t4"],
        outputs: ["t5"],
      },
      {
        name: "fc",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["t5", "w_fc"],
        outputs: ["logits"],
      },
      {
        name: "sig",
        op: @nn_ir.Sigmoid,
        inputs: ["logits"],
        outputs: ["probs"],
      },
    ],
  }

  let art = @codegen.compile_graph_to_mbt(g, Map([("audio", [1, 1, 32])]), config={
    struct_name: "AudioKws",
    use_simd: true,
    optimize: true,
    target: @codegen.TargetWasm,
    mode: @codegen.Arena,
    embed_weights: true,
  })
  assert_eq(art.mode_used, @codegen.Arena)
  assert_eq(art.node_count, 6) // Conv1D+BN+Relu 折叠为 1 个节点
  assert_eq(art.output_shapes, [("probs", [1, 2])])
  assert_true(art.mbt_source.contains("@conv.depthwise_conv2d_f32"))
  assert_true(
    art.mbt_source.contains("@gemm.matmul_2d_prepacked_nchw_simd_f32"),
  )
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}
