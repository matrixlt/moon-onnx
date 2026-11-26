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
  let art_wasm = @onnx_aot.compile_graph_to_mbt(graph, input_shapes, config={
    ..@onnx_aot.CodegenConfig::default(),
    struct_name: "TinyVisionWasm",
    target: @onnx_aot.TargetWasm,
    mode: @onnx_aot.Arena,
    embed_weights: true,
  })
  assert_eq(art_wasm.mode_used, @onnx_aot.Arena)
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
  let art_gc = @onnx_aot.compile_graph_to_mbt(graph, input_shapes, config={
    ..@onnx_aot.CodegenConfig::default(),
    struct_name: "TinyVisionGc",
    target: @onnx_aot.TargetWasmGc,
    mode: @onnx_aot.Arena,
    embed_weights: false,
  })
  assert_eq(art_gc.mode_used, @onnx_aot.Arena)
  assert_true(art_gc.mbt_source.contains("@nn.conv_nchw_scalar_f32"))
  assert_true(art_gc.mbt_source.contains("@gemm.matmul_2d_scalar_f32"))
  assert_true(!art_gc.mbt_source.contains("PackedMatrixB"))

  // 3. Tensor 兼容模式仍正常工作
  let art_tensor = @onnx_aot.compile_graph_to_mbt(graph, input_shapes, config={
    ..@onnx_aot.CodegenConfig::default(),
    struct_name: "TinyVisionTensor",
    target: @onnx_aot.TargetWasm,
    mode: @onnx_aot.Tensor,
    embed_weights: false,
  })
  assert_eq(art_tensor.mode_used, @onnx_aot.Tensor)
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

  let art = @onnx_aot.compile_graph_to_mbt(q_graph, Map([("x", [2, 16])]), config={
    ..@onnx_aot.CodegenConfig::default(),
    struct_name: "Int8Mlp",
    target: @onnx_aot.TargetWasm,
    mode: @onnx_aot.Arena,
    embed_weights: true,
  })
  assert_eq(art.mode_used, @onnx_aot.Arena)
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

  let art = @onnx_aot.compile_graph_to_mbt(g, Map([("audio", [1, 1, 32])]), config={
    ..@onnx_aot.CodegenConfig::default(),
    struct_name: "AudioKws",
    target: @onnx_aot.TargetWasm,
    mode: @onnx_aot.Arena,
    embed_weights: true,
  })
  assert_eq(art.mode_used, @onnx_aot.Arena)
  assert_eq(art.node_count, 6) // Conv1D+BN+Relu 折叠为 1 个节点
  assert_eq(art.output_shapes, [("probs", [1, 2])])
  assert_true(art.mbt_source.contains("@conv.depthwise_conv2d_f32"))
  assert_true(
    art.mbt_source.contains("@gemm.matmul_2d_prepacked_nchw_simd_f32"),
  )
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}

///|
/// 验证编译期静态形状子图折叠（MobileNetV2 典型 `GlobalAveragePool -> Shape -> Gather -> Unsqueeze -> Concat -> Reshape -> Gemm`）
/// 可将元数据节点全部消除，使整图直接落入零分配 Arena 模式。
test "specialize_graph folds MobileNetV2 Shape-Gather-Unsqueeze-Concat chain into pure Arena mode" {
  let w_fc = @tensor.Tensor::full_f32([4, 2], (0.5 : Float))
  let idx0 = @tensor.Tensor::from_i64([], FixedArray::make(1, (0 : Int64)))
  let neg1 = @tensor.Tensor::from_i64([1], [-1])

  let g : @nn_ir.IrGraph = {
    name: "mobilenet_tail",
    inputs: ["feat"],
    outputs: ["logits"],
    initializers: Map([("w_fc", w_fc), ("idx0", idx0), ("neg1", neg1)]),
    nodes: [
      {
        name: "gap",
        op: @nn_ir.GlobalAveragePool,
        inputs: ["feat"],
        outputs: ["pooled"],
      },
      {
        name: "sh",
        op: @nn_ir.Shape(start=None, end=None),
        inputs: ["feat"],
        outputs: ["dims"],
      },
      {
        name: "g0",
        op: @nn_ir.Gather(axis=0),
        inputs: ["dims", "idx0"],
        outputs: ["b_scalar"],
      },
      {
        name: "unsq",
        op: @nn_ir.Unsqueeze(axes=Some([0])),
        inputs: ["b_scalar"],
        outputs: ["b_1d"],
      },
      {
        name: "cat_sh",
        op: @nn_ir.Concat(axis=0),
        inputs: ["b_1d", "neg1"],
        outputs: ["flat_sh"],
      },
      {
        name: "rsh",
        op: @nn_ir.Reshape(allowzero=false),
        inputs: ["pooled", "flat_sh"],
        outputs: ["flat"],
      },
      {
        name: "fc",
        op: @nn_ir.Gemm(
          alpha=(1.0 : Float),
          beta=(1.0 : Float),
          trans_a=false,
          trans_b=true,
        ),
        inputs: ["flat", "w_fc"],
        outputs: ["logits"],
      },
    ],
  }

  let art = @onnx_aot.compile_graph_to_mbt(g, Map([("feat", [1, 2, 4, 4])]))
  assert_eq(art.mode_used, @onnx_aot.Arena)
  // Shape/Gather/Unsqueeze/Concat 全部被编译期常量折叠消除，仅剩 GAP + Reshape(alias) + Gemm
  assert_eq(art.node_count, 3)
  assert_eq(art.output_shapes, [("logits", [1, 4])])
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}

///|
/// 验证编译期 `frozen_inputs` + `If` 分支特化与多输入多输出 Zero-Allocation Arena 代码生成
/// （覆盖 Silero VAD 关键算子：`Pad(reflect)`、`Conv1D`、`Slice`、`Pow(x,2)`、`Split`、`Concat`、`ReduceMean`）。
test "compile_graph_to_mbt specializes If branch and emits multi-IO Arena code" {
  let sr_16k = @tensor.Tensor::from_i64(
    [],
    FixedArray::make(1, (16000 : Int64)),
  )
  let pads = @tensor.Tensor::from_i64([4], [0, 0, 0, 2])
  let s0 = @tensor.Tensor::from_i64([1], [0])
  let e0 = @tensor.Tensor::from_i64([1], [4])
  let ax1 = @tensor.Tensor::from_i64([1], [1])
  let two = @tensor.Tensor::from_f32([], FixedArray::make(1, (2.0 : Float)))
  let sp = @tensor.Tensor::from_i64([2], [2, 2])

  let then_branch : @nn_ir.IrGraph = {
    name: "vad_16k",
    inputs: [],
    outputs: ["out_16k", "st_16k"],
    initializers: Map([
      ("pads", pads),
      ("s0", s0),
      ("e0", e0),
      ("ax1", ax1),
      ("two", two),
      ("sp", sp),
    ]),
    nodes: [
      {
        name: "pad",
        op: @nn_ir.Pad(mode="reflect", pads=None, constant_value=None),
        inputs: ["audio", "pads"],
        outputs: ["padded"],
      },
      {
        name: "sl",
        op: @nn_ir.Slice(starts=None, ends=None, axes=None, steps=None),
        inputs: ["padded", "s0", "e0", "ax1"],
        outputs: ["sliced"],
      },
      {
        name: "sq",
        op: @nn_ir.Pow,
        inputs: ["sliced", "two"],
        outputs: ["sq_out"],
      },
      {
        name: "sp_n",
        op: @nn_ir.Split(axis=1, split=None, num_outputs=None),
        inputs: ["sq_out", "sp"],
        outputs: ["p0", "p1"],
      },
      {
        name: "mix",
        op: @nn_ir.Add,
        inputs: ["p0", "state"],
        outputs: ["st_next_part"],
      },
      {
        name: "cat",
        op: @nn_ir.Concat(axis=1),
        inputs: ["st_next_part", "p1"],
        outputs: ["st_16k"],
      },
      {
        name: "mean",
        op: @nn_ir.ReduceMean(
          axes=Some([1]),
          keepdims=true,
          noop_with_empty_axes=false,
        ),
        inputs: ["st_16k"],
        outputs: ["out_16k"],
      },
    ],
  }
  let else_branch : @nn_ir.IrGraph = {
    name: "vad_8k",
    inputs: [],
    outputs: ["audio", "state"],
    initializers: Map([]),
    nodes: [],
  }
  let top : @nn_ir.IrGraph = {
    name: "silero_like",
    inputs: ["audio", "state", "sr"],
    outputs: ["prob", "stateN"],
    initializers: Map([("sr_16k", sr_16k)]),
    nodes: [
      {
        name: "eq",
        op: @nn_ir.Equal,
        inputs: ["sr", "sr_16k"],
        outputs: ["is_16k"],
      },
      {
        name: "branch",
        op: @nn_ir.If(then_branch~, else_branch~),
        inputs: ["is_16k"],
        outputs: ["prob", "stateN"],
      },
    ],
  }

  let art = @onnx_aot.compile_graph_to_mbt(
    top,
    Map([("audio", [1, 4]), ("state", [1, 2])]),
    config={
      ..@onnx_aot.CodegenConfig::default(),
      struct_name: "MiniVad16k",
      frozen_inputs: Map([
        (
          "sr",
          @tensor.Tensor::from_i64([], FixedArray::make(1, (16000 : Int64))),
        ),
      ]),
    },
  )
  assert_eq(art.mode_used, @onnx_aot.Arena)
  assert_eq(art.input_shapes, [("audio", [1, 4]), ("state", [1, 2])])
  assert_eq(art.output_shapes, [("prob", [1, 1]), ("stateN", [1, 4])])
  assert_true(art.mbt_source.contains("@nn.reflect_idx("))
  assert_true(art.mbt_source.contains("v * v"))
  assert_true(art.mbt_source.contains("pub fn MiniVad16k::forward_into("))
  assert_true(art.mbt_source.contains("in_1 : FixedArray[Float]"))
  assert_true(art.mbt_source.contains("out_1 : FixedArray[Float]"))
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}

///|
/// 验证迷你 BERT/Transformer 编码块（含 `Int64` 输入、动态 `Gather` 词嵌入、
/// `Unsqueeze -> Cast` 掩码、9 节点融合 `LayerNormalization`、多头 `Transpose`、
/// 批量动态 `Q@K^T` 与 4D 广播掩码 `Add`、`Softmax` 以及 `Erf` GELU）
/// 可完整编译为零堆分配 `Arena` 模式且无 `matrixlt/tensor` 依赖。
test "Transformer encoder block compiles to zero-allocation Arena mode with Int64 inputs" {
  let emb_w = @tensor.Tensor::full_f32([16, 8], (0.1 : Float))
  let w_q = @tensor.Tensor::full_f32([8, 8], (0.05 : Float))
  let b_q = @tensor.Tensor::full_f32([8], (0.01 : Float))
  let ln_g = @tensor.Tensor::full_f32([8], (1.0 : Float))
  let ln_b = @tensor.Tensor::full_f32([8], (0.0 : Float))
  let sh_4d = @tensor.Tensor::from_i64([4], [1, 4, 2, 4])
  let sh_3d = @tensor.Tensor::from_i64([3], [1, 4, 8])
  let inv_sqrt_d = @tensor.Tensor::from_f32_array([], [0.5])
  let mask_scale = @tensor.Tensor::from_f32_array([], [-10000.0])
  let one_f = @tensor.Tensor::from_f32_array([], [1.0])
  let ax1 = @tensor.Tensor::from_i64([1], [1])
  let ax2 = @tensor.Tensor::from_i64([1], [2])

  let g : @nn_ir.IrGraph = {
    name: "mini_bert",
    inputs: ["input_ids", "attention_mask"],
    outputs: ["hidden"],
    initializers: Map([
      ("emb_w", emb_w),
      ("w_q", w_q),
      ("b_q", b_q),
      ("ln_g", ln_g),
      ("ln_b", ln_b),
      ("sh_4d", sh_4d),
      ("sh_3d", sh_3d),
      ("inv_sqrt_d", inv_sqrt_d),
      ("mask_scale", mask_scale),
      ("one_f", one_f),
      ("ax1", ax1),
      ("ax2", ax2),
    ]),
    nodes: [
      {
        name: "emb",
        op: @nn_ir.Gather(axis=0),
        inputs: ["emb_w", "input_ids"],
        outputs: ["x0"],
      },
      {
        name: "ln0",
        op: @nn_ir.LayerNormalization(axis=-1, epsilon=(1.0e-12 : Float)),
        inputs: ["x0", "ln_g", "ln_b"],
        outputs: ["x1"],
      },
      // attention_mask [1, 4] -> [1, 1, 1, 4] -> f32 -> (1 - m) * -10000
      {
        name: "u1",
        op: @nn_ir.Unsqueeze(axes=None),
        inputs: ["attention_mask", "ax1"],
        outputs: ["m1"],
      },
      {
        name: "u2",
        op: @nn_ir.Unsqueeze(axes=None),
        inputs: ["m1", "ax2"],
        outputs: ["m2"],
      },
      {
        name: "cast_m",
        op: @nn_ir.Cast(to=@tensor.Float32),
        inputs: ["m2"],
        outputs: ["mf"],
      },
      {
        name: "sub_m",
        op: @nn_ir.Sub,
        inputs: ["one_f", "mf"],
        outputs: ["inv_m"],
      },
      {
        name: "mul_m",
        op: @nn_ir.Mul,
        inputs: ["inv_m", "mask_scale"],
        outputs: ["ext_mask"],
      },
      // Q projection + symmetric bias [8] + [1, 4, 8]
      {
        name: "mm_q",
        op: @nn_ir.MatMul,
        inputs: ["x1", "w_q"],
        outputs: ["q_raw"],
      },
      {
        name: "add_bq",
        op: @nn_ir.Add,
        inputs: ["b_q", "q_raw"],
        outputs: ["q_lin"],
      },
      {
        name: "r_q",
        op: @nn_ir.Reshape(allowzero=false),
        inputs: ["q_lin", "sh_4d"],
        outputs: ["q_4d"],
      },
      {
        name: "t_q",
        op: @nn_ir.Transpose(perm=[0, 2, 1, 3]),
        inputs: ["q_4d"],
        outputs: ["q_h"],
      },
      {
        name: "t_k",
        op: @nn_ir.Transpose(perm=[0, 2, 3, 1]),
        inputs: ["q_4d"],
        outputs: ["k_h"],
      },
      // Batched dynamic Q @ K^T -> [1, 2, 4, 4]
      {
        name: "qk",
        op: @nn_ir.MatMul,
        inputs: ["q_h", "k_h"],
        outputs: ["scores"],
      },
      {
        name: "sc",
        op: @nn_ir.Mul,
        inputs: ["scores", "inv_sqrt_d"],
        outputs: ["scores_sc"],
      },
      // 4D broadcast add: [1, 2, 4, 4] + [1, 1, 1, 4]
      {
        name: "add_mask",
        op: @nn_ir.Add,
        inputs: ["scores_sc", "ext_mask"],
        outputs: ["masked"],
      },
      {
        name: "sm",
        op: @nn_ir.Softmax(axis=-1),
        inputs: ["masked"],
        outputs: ["probs"],
      },
      // Batched dynamic P @ V -> [1, 2, 4, 4]
      {
        name: "pv",
        op: @nn_ir.MatMul,
        inputs: ["probs", "q_h"],
        outputs: ["ctx"],
      },
      {
        name: "t_c",
        op: @nn_ir.Transpose(perm=[0, 2, 1, 3]),
        inputs: ["ctx"],
        outputs: ["ctx_t"],
      },
      {
        name: "r_c",
        op: @nn_ir.Reshape(allowzero=false),
        inputs: ["ctx_t", "sh_3d"],
        outputs: ["ctx_3d"],
      },
      { name: "erf0", op: @nn_ir.Erf, inputs: ["ctx_3d"], outputs: ["hidden"], },
    ],
  }

  let art = @onnx_aot.compile_graph_to_mbt(
    g,
    Map([("input_ids", [1, 4]), ("attention_mask", [1, 4])]),
    config={
      ..@onnx_aot.CodegenConfig::default(),
      struct_name: "MiniBertL1",
      target: @onnx_aot.TargetWasm,
      input_dtypes: Map([
        ("input_ids", @tensor.Int64),
        ("attention_mask", @tensor.Int64),
      ]),
    },
  )
  assert_eq(art.mode_used, @onnx_aot.Arena)
  assert_eq(art.input_shapes, [
    ("input_ids", [1, 4]),
    ("attention_mask", [1, 4]),
  ])
  assert_eq(art.output_shapes, [("hidden", [1, 4, 8])])
  assert_true(art.arena_f32_elems < art.arena_f32_naive_elems)
  assert_true(art.mbt_source.contains("in_0 : FixedArray[Int64]"))
  assert_true(art.mbt_source.contains("in_1 : FixedArray[Int64]"))
  assert_true(art.mbt_source.contains("@nn.erf_f32("))
  assert_true(art.mbt_source.contains("@nn.layernorm_f32("))
  assert_true(art.mbt_source.contains("@nn.softmax_f32("))
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}

///|
/// 验证 INT8 动态量化 Transformer 子图（`UInt8 Gather` 词表 + `DequantizeLinear` +
/// `DynamicQuantizeLinear` + 融合 `MatMulInteger -> Cast(Float32)` + 缩放 `Mul`）
/// 可完整编译为零堆分配 `Arena` 模式（含 `PackedInt8MatrixB` 预打包与 `arena_u8` 静态复用）。
test "Quantized INT8 Transformer block compiles to zero-allocation Arena mode" {
  let emb_u8 = @tensor.Tensor::from_u8(
    [16, 8],
    FixedArray::makei(128, fn(i) { (i % 200).to_byte() }),
  )
  let emb_sc = @tensor.Tensor::from_f32_array([], [0.02])
  let emb_zp = @tensor.Tensor::from_u8([], [(100).to_byte()])
  let w_i8 = @tensor.Tensor::from_i8(
    [8, 8],
    FixedArray::makei(64, fn(i) { ((i % 100 - 50) & 0xFF).to_byte() }),
  )
  let w_zp = @tensor.Tensor::from_i8([8], FixedArray::make(8, b'\x00'))
  let w_sc = @tensor.Tensor::full_f32([8], (0.01 : Float))

  let g : @nn_ir.IrGraph = {
    name: "mini_bert_q8",
    inputs: ["input_ids"],
    outputs: ["out"],
    initializers: Map([
      ("emb_u8", emb_u8),
      ("emb_sc", emb_sc),
      ("emb_zp", emb_zp),
      ("w_i8", w_i8),
      ("w_zp", w_zp),
      ("w_sc", w_sc),
    ]),
    nodes: [
      {
        name: "g0",
        op: @nn_ir.Gather(axis=0),
        inputs: ["emb_u8", "input_ids"],
        outputs: ["emb_q"],
      },
      {
        name: "dq0",
        op: @nn_ir.DequantizeLinear(axis=1),
        inputs: ["emb_q", "emb_sc", "emb_zp"],
        outputs: ["x"],
      },
      {
        name: "dynq",
        op: @nn_ir.DynamicQuantizeLinear,
        inputs: ["x"],
        outputs: ["xq", "x_sc", "x_zp"],
      },
      {
        name: "sc_mul",
        op: @nn_ir.Mul,
        inputs: ["x_sc", "w_sc"],
        outputs: ["comb_sc"],
      },
      {
        name: "mmi",
        op: @nn_ir.MatMulInteger,
        inputs: ["xq", "w_i8", "x_zp", "w_zp"],
        outputs: ["acc_i32"],
      },
      {
        name: "cast0",
        op: @nn_ir.Cast(to=@tensor.Float32),
        inputs: ["acc_i32"],
        outputs: ["acc_f32"],
      },
      {
        name: "out_mul",
        op: @nn_ir.Mul,
        inputs: ["acc_f32", "comb_sc"],
        outputs: ["out"],
      },
    ],
  }

  let art = @onnx_aot.compile_graph_to_mbt(g, Map([("input_ids", [1, 4])]), config={
    ..@onnx_aot.CodegenConfig::default(),
    struct_name: "MiniBertQ8",
    target: @onnx_aot.TargetWasm,
    input_dtypes: Map([("input_ids", @tensor.Int64)]),
  })
  assert_eq(art.mode_used, @onnx_aot.Arena)
  assert_true(art.arena_byte_elems > 0)
  assert_true(art.mbt_source.contains("@quant.PackedInt8MatrixB"))
  assert_true(art.mbt_source.contains("@quant.dynamic_quantize_linear_u8"))
  assert_true(
    art.mbt_source.contains("@quant.matmul_integer_u8_i8_prepacked_to_f32_v128"),
  )
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}

///|
/// 验证 `onnx_aot` Arena 模式支持现代 Q4 LLM 算子组合（`RMSNorm -> MatMulNBits(Q4) -> RotaryEmbedding -> SkipRMSNorm -> SiLU`），
/// 直接发射 `@nn.*` 与 `@quant.matmul_nbits_q4_v128_f32` 零堆分配代码。
test "onnx_aot Arena mode compiles Q4 MatMulNBits + RMSNorm + SkipRMSNorm + RoPE + SiLU" {
  let wq_b = @tensor.Tensor::from_u8(
    [32, 1, 16],
    FixedArray::make(32 * 16, b'\x88'),
  )
  let wq_s = @tensor.Tensor::full_f32([32], (0.05 : Float))
  let norm_w = @tensor.Tensor::full_f32([32], (1.0 : Float))
  let cos_c = @tensor.Tensor::full_f32([8, 4], (1.0 : Float))
  let sin_c = @tensor.Tensor::full_f32([8, 4], (0.0 : Float))
  let g : @nn_ir.IrGraph = {
    name: "mini_q4_aot",
    inputs: ["x", "pos"],
    outputs: ["out"],
    initializers: Map([
      ("norm_w", norm_w),
      ("wq.b", wq_b),
      ("wq.s", wq_s),
      ("cos", cos_c),
      ("sin", sin_c),
    ]),
    nodes: [
      {
        name: "rms0",
        op: @nn_ir.RmsNormalization(axis=-1, epsilon=(1.0e-5 : Float)),
        inputs: ["x", "norm_w"],
        outputs: ["h0"],
      },
      {
        name: "q4_0",
        op: @nn_ir.MatMulNBits(k=32, n=32, bits=4, block_size=32),
        inputs: ["h0", "wq.b", "wq.s"],
        outputs: ["q"],
      },
      {
        name: "rope0",
        op: @nn_ir.RotaryEmbedding(
          interleaved=0,
          num_heads=0,
          rotary_embedding_dim=0,
        ),
        inputs: ["q", "pos", "cos", "sin"],
        outputs: ["qr"],
      },
      {
        name: "skip0",
        op: @nn_ir.SkipRmsNormalization(epsilon=(1.0e-5 : Float)),
        inputs: ["x", "qr", "norm_w"],
        outputs: ["h1", "", "", "res1"],
      },
      { name: "sig0", op: @nn_ir.Sigmoid, inputs: ["h1"], outputs: ["s0"], },
      { name: "mul0", op: @nn_ir.Mul, inputs: ["h1", "s0"], outputs: ["act"], },
      {
        name: "add0",
        op: @nn_ir.Add,
        inputs: ["res1", "act"],
        outputs: ["out"],
      },
    ],
  }
  let art = @onnx_aot.compile_graph_to_mbt(
    g,
    Map([("x", [1, 2, 32]), ("pos", [1, 2])]),
    config={
      ..@onnx_aot.CodegenConfig::default(),
      struct_name: "MiniQ4Block",
      target: @onnx_aot.TargetWasm,
      input_dtypes: Map([("x", @tensor.Float32), ("pos", @tensor.Int64)]),
    },
  )
  assert_eq(art.mode_used, @onnx_aot.Arena)
  assert_true(art.mbt_source.contains("@nn.rmsnorm_f32("))
  assert_true(art.mbt_source.contains("@quant.matmul_nbits_q4_v128_f32("))
  assert_true(art.mbt_source.contains("@nn.rotary_embedding_f32("))
  assert_true(art.mbt_source.contains("@nn.skip_rmsnorm_f32("))
  assert_true(art.mbt_source.contains("@nn.silu_f32("))
  assert_true(art.pkg_config.contains("matrixlt/ml_kernels/nn"))
  assert_true(art.pkg_config.contains("matrixlt/ml_kernels/quant"))
  assert_true(!art.pkg_config.contains("matrixlt/tensor"))
}
