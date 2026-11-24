///|
/// 验证 `onnx2mbt` AOT 静态编译器：对包含 `Conv+BN+Clip(ReLU6) -> GAP -> Flatten -> Gemm -> Softmax`
/// 的完整视觉网络执行图优化、静态形状推导与零 Map 纯 MoonBit 代码生成。
test "compile_graph_to_mbt optimizes Conv+BN+Clip and emits zero-Map prepacked MoonBit code" {
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

  let graph : @nn_ir.IrGraph = {
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

  let input_shapes : Map[String, Array[Int32]] = Map([("x", [1, 2, 6, 6])])
  let art = @codegen.compile_graph_to_mbt(graph, input_shapes, config={
    struct_name: "TinyVision",
    use_simd: true,
    optimize: true,
  })

  // 优化后 9 节点折叠为 5 节点：Conv+Clip, GAP, Flatten, Gemm, Softmax
  assert_eq(art.node_count, 5)
  assert_eq(art.input_shapes, [("x", [1, 2, 6, 6])])
  assert_eq(art.output_shapes, [("probs", [1, 3])])

  // 权重仅保留折叠后的 Conv W (72) + Conv B (4) + 预转置 FC W (12) + FC B (3) = 91 floats = 364 bytes
  assert_eq(art.weights.length(), 4)
  assert_eq(art.weights_bin.length(), 91 * 4)

  // 验证生成代码包含预期预打包调用且完全不包含动态 Map 或 Protobuf 依赖
  assert_true(art.mbt_source.contains("pub struct TinyVision"))
  assert_true(art.mbt_source.contains("@ops.conv2d_prepacked_f32"))
  assert_true(art.mbt_source.contains("@nn_ir.ClipAct"))
  assert_true(art.mbt_source.contains("@ops.gemm_prepacked_f32"))
  assert_true(art.mbt_source.contains("pub fn TinyVision::forward_f32"))
  assert_true(art.mbt_source.contains("release_buf(self.pool,"))
  assert_true(!art.mbt_source.contains("Map["))
  assert_true(!art.pkg_config.contains("onnx_proto"))
  assert_true(!art.pkg_config.contains("onnx/session"))
}
