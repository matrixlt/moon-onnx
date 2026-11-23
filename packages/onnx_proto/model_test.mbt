///|
/// 构造小端序 Float32 raw_data 字节流，用于测试 TensorProto 二进制权重解码。
fn encode_f32_le(values : Array[Float]) -> Bytes {
  let out = FixedArray::make(values.length() * 4, b'\x00')
  for i = 0; i < values.length(); i = i + 1 {
    let bits = values[i].reinterpret_as_uint()
    out[i * 4] = (bits & 0xFFU).to_byte()
    out[i * 4 + 1] = ((bits >> 8) & 0xFFU).to_byte()
    out[i * 4 + 2] = ((bits >> 16) & 0xFFU).to_byte()
    out[i * 4 + 3] = ((bits >> 24) & 0xFFU).to_byte()
  }
  Bytes::from_array(out)
}

///|
/// 验证 TensorProto 同时支持 raw_data 与 float_data 两种解码路径。
test "decode_tensor_proto supports raw_data and typed float_data" {
  let tp_raw = @schema_test.TensorProto::default()
  tp_raw.name = Some("w_raw")
  tp_raw.data_type = Some(1) // FLOAT
  tp_raw.dims = [(2 : Int64), (2 : Int64)]
  tp_raw.raw_data = Some(encode_f32_le([1.5, -2.25, 0.0, 4.0]))

  let t_raw = @onnx_proto.decode_tensor_proto(tp_raw)
  assert_eq(t_raw.shape(), [2, 2])
  assert_eq(t_raw.get_f32([0, 0]), (1.5 : Float))
  assert_eq(t_raw.get_f32([0, 1]), (-2.25 : Float))
  assert_eq(t_raw.get_f32([1, 1]), (4.0 : Float))

  let tp_typed = @schema_test.TensorProto::default()
  tp_typed.name = Some("w_typed")
  tp_typed.data_type = Some(1) // FLOAT
  tp_typed.dims = [(3 : Int64)]
  tp_typed.float_data = [(10.0 : Float), (20.0 : Float), (30.0 : Float)]

  let t_typed = @onnx_proto.decode_tensor_proto(tp_typed)
  assert_eq(t_typed.shape(), [3])
  assert_eq(t_typed.get_f32([2]), (30.0 : Float))
}

///|
/// 验证构造完整 ModelProto 序列化后再由 load_model_from_bytes 反序列化闭环。
test "load_model_from_bytes round-trips a minimal ONNX graph" {
  // 构造一个最小 Relu 单节点模型
  let input_dim0 = @schema_test.TensorShapeProto_Dimension::default()
  input_dim0.value = @schema_test.TensorShapeProto_Dimension_Value::DimValue(
    (1 : Int64),
  )
  let input_dim1 = @schema_test.TensorShapeProto_Dimension::default()
  input_dim1.value = @schema_test.TensorShapeProto_Dimension_Value::DimValue(
    (4 : Int64),
  )
  let shape_proto = @schema_test.TensorShapeProto::default()
  shape_proto.dim = [input_dim0, input_dim1]

  let tt = @schema_test.TypeProto_Tensor::default()
  tt.elem_type = Some(1)
  tt.shape = Some(shape_proto)

  let tp = @schema_test.TypeProto::default()
  tp.value = @schema_test.TypeProto_Value::TensorType(tt)

  let vi_in = @schema_test.ValueInfoProto::default()
  vi_in.name = Some("x")
  vi_in.type_ = Some(tp)

  let vi_out = @schema_test.ValueInfoProto::default()
  vi_out.name = Some("y")
  vi_out.type_ = Some(tp)

  let node = @schema_test.NodeProto::default()
  node.op_type = Some("Relu")
  node.name = Some("relu_0")
  node.input = ["x"]
  node.output = ["y"]

  let graph = @schema_test.GraphProto::default()
  graph.name = Some("tiny_relu")
  graph.input = [vi_in]
  graph.output = [vi_out]
  graph.node = [node]

  let opset = @schema_test.OperatorSetIdProto::default()
  opset.domain = Some("")
  opset.version = Some((17 : Int64))

  let model_proto = @schema_test.ModelProto::default()
  model_proto.ir_version = Some((9 : Int64))
  model_proto.producer_name = Some("moon-onnx-test")
  model_proto.opset_import = [opset]
  model_proto.graph = Some(graph)

  // 序列化为二进制 Bytes
  let writer = Buffer()
  model_proto.write(writer)
  let encoded = writer.to_bytes()

  // 反序列化并校验摘要字段
  let loaded = @onnx_proto.load_model_from_bytes(encoded)
  assert_eq(loaded.graph_name, "tiny_relu")
  assert_eq(loaded.ir_version, (9 : Int64))
  assert_eq(loaded.default_opset_version(), Some((17 : Int64)))
  assert_eq(loaded.inputs.length(), 1)
  assert_eq(loaded.inputs[0].format(), "x: f32[1, 4]")
  assert_eq(loaded.outputs.length(), 1)
  assert_eq(loaded.nodes.length(), 1)
  assert_true(loaded.format_summary().contains("Relu (relu_0)"))
}
