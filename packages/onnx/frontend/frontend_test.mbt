///|
/// 构造最小 `ValueInfoProto`（Float32 静态形状），用于纯内存前端 Lowering 单元测试。
fn make_f32_value_info(
  name : String,
  dims : Array[Int64],
) -> @schema.ValueInfoProto {
  let shape_dims = Array::makei(dims.length(), fn(i) {
    let d = @schema.TensorShapeProto_Dimension::default()
    d.value = @schema.TensorShapeProto_Dimension_Value::DimValue(dims[i])
    d
  })
  let shape = @schema.TensorShapeProto::default()
  shape.dim = shape_dims
  let tt = @schema.TypeProto_Tensor::default()
  tt.elem_type = Some(1)
  tt.shape = Some(shape)
  let tp = @schema.TypeProto::default()
  tp.value = @schema.TypeProto_Value::TensorType(tt)
  let vi = @schema.ValueInfoProto::default()
  vi.name = Some(name)
  vi.type_ = Some(tp)
  vi
}

///|
/// 构造 Float32 类型 `TensorProto` 初始化器。
fn make_f32_initializer(
  name : String,
  dims : Array[Int64],
  values : Array[Float],
) -> @schema.TensorProto {
  let tp = @schema.TensorProto::default()
  tp.name = Some(name)
  tp.data_type = Some(1)
  tp.dims = dims
  tp.float_data = values
  tp
}

///|
/// 构造整数组属性（如 `strides`、`pads`、`kernel_shape`）。
fn make_ints_attr(name : String, vals : Array[Int64]) -> @schema.AttributeProto {
  let a = @schema.AttributeProto::default()
  a.name = Some(name)
  a.type_ = Some(@schema.AttributeProto_AttributeType::INTS)
  a.ints = vals
  a
}

///|
/// 构造单整数属性（如 `transB`、`axis`、`group`）。
fn make_int_attr(name : String, val : Int64) -> @schema.AttributeProto {
  let a = @schema.AttributeProto::default()
  a.name = Some(name)
  a.type_ = Some(@schema.AttributeProto_AttributeType::INT)
  a.i = Some(val)
  a
}

///|
/// 验证纯内存构造的 CNN+MLP ONNX 模型可完整 Lowering 为强类型 `nn_ir::IrGraph`（零文件 I/O，四端可测）。
test "lower_onnx_model lowers in-memory CNN+Gemm ONNX model with attributes and topo sort" {
  let conv_w = make_f32_initializer(
    "conv.w",
    [2, 1, 3, 3],
    Array::make(18, (0.25 : Float)),
  )
  let conv_b = make_f32_initializer("conv.b", [2], [
    (0.1 : Float),
    (-0.1 : Float),
  ])
  let fc_w = make_f32_initializer(
    "fc.w",
    [3, 8],
    Array::make(24, (0.5 : Float)),
  )
  let fc_b = make_f32_initializer("fc.b", [3], [
    (0.0 : Float),
    (0.1 : Float),
    (0.2 : Float),
  ])

  let n_conv = @schema.NodeProto::default()
  n_conv.op_type = Some("Conv")
  n_conv.name = Some("conv1")
  n_conv.input = ["input", "conv.w", "conv.b"]
  n_conv.output = ["c1"]
  n_conv.attribute = [
    make_ints_attr("strides", [1, 1]),
    make_ints_attr("pads", [1, 1, 1, 1]),
    make_int_attr("group", 1),
  ]

  let n_relu = @schema.NodeProto::default()
  n_relu.op_type = Some("Relu")
  n_relu.name = Some("relu1")
  n_relu.input = ["c1"]
  n_relu.output = ["r1"]

  let n_pool = @schema.NodeProto::default()
  n_pool.op_type = Some("MaxPool")
  n_pool.name = Some("pool1")
  n_pool.input = ["r1"]
  n_pool.output = ["p1"]
  n_pool.attribute = [
    make_ints_attr("kernel_shape", [2, 2]),
    make_ints_attr("strides", [2, 2]),
  ]

  let n_flat = @schema.NodeProto::default()
  n_flat.op_type = Some("Flatten")
  n_flat.name = Some("flat1")
  n_flat.input = ["p1"]
  n_flat.output = ["f1"]
  n_flat.attribute = [make_int_attr("axis", 1)]

  let n_gemm = @schema.NodeProto::default()
  n_gemm.op_type = Some("Gemm")
  n_gemm.name = Some("fc1")
  n_gemm.input = ["f1", "fc.w", "fc.b"]
  n_gemm.output = ["logits"]
  n_gemm.attribute = [make_int_attr("transB", 1)]

  let n_sm = @schema.NodeProto::default()
  n_sm.op_type = Some("Softmax")
  n_sm.name = Some("sm1")
  n_sm.input = ["logits"]
  n_sm.output = ["probs"]
  n_sm.attribute = [make_int_attr("axis", -1)]

  let graph = @schema.GraphProto::default()
  graph.name = Some("tiny_cnn_frontend_test")
  graph.input = [make_f32_value_info("input", [1, 1, 4, 4])]
  graph.output = [make_f32_value_info("probs", [1, 3])]
  graph.initializer = [conv_w, conv_b, fc_w, fc_b]
  // 故意打乱节点顺序，验证前端自动执行拓扑排序
  graph.node = [n_sm, n_gemm, n_flat, n_pool, n_relu, n_conv]

  let opset = @schema.OperatorSetIdProto::default()
  opset.domain = Some("")
  opset.version = Some((17 : Int64))

  let model_proto = @schema.ModelProto::default()
  model_proto.ir_version = Some((9 : Int64))
  model_proto.producer_name = Some("moon-onnx-unit-test")
  model_proto.opset_import = [opset]
  model_proto.graph = Some(graph)

  let model = @onnx_proto.load_model_from_proto(model_proto)
  let ir = @frontend.lower_onnx_model(model)
  assert_eq(ir.nodes.length(), 6)
  assert_eq(ir.nodes[0].op.name(), "Conv")
  assert_eq(ir.nodes[1].op.name(), "Relu")
  assert_eq(ir.nodes[2].op.name(), "MaxPool")
  assert_eq(ir.nodes[3].op.name(), "Flatten")
  assert_eq(ir.nodes[4].op.name(), "Gemm")
  assert_eq(ir.nodes[5].op.name(), "Softmax")
}
