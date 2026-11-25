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

///|
/// 验证 IEEE 754 半精度浮点（FLOAT16, code=10）的 `raw_data` 与 `int32_data` 解码精度，
/// 涵盖常规值（1.0, -2.5, 0.5）、最小规格化数、非规格化数（subnormal）与 ±0.0。
test "decode_tensor_proto decodes IEEE 754 FLOAT16 raw_data and int32_data" {
  // 0x3C00 = 1.0, 0xC100 = -2.5, 0x3800 = 0.5, 0x0000 = 0.0, 0x0001 = 2^-24 (subnormal)
  let h_bits : Array[Int32] = [0x3C00, 0xC100, 0x3800, 0x0000, 0x0001]
  let raw = FixedArray::make(h_bits.length() * 2, b'\x00')
  for i = 0; i < h_bits.length(); i = i + 1 {
    raw[i * 2] = (h_bits[i] & 0xFF).to_byte()
    raw[i * 2 + 1] = ((h_bits[i] >> 8) & 0xFF).to_byte()
  }

  let tp_raw = @schema_test.TensorProto::default()
  tp_raw.name = Some("w_f16_raw")
  tp_raw.data_type = Some(10) // FLOAT16
  tp_raw.dims = [(5 : Int64)]
  tp_raw.raw_data = Some(Bytes::from_array(raw))

  let t_raw = @onnx_proto.decode_tensor_proto(tp_raw)
  assert_eq(t_raw.dtype(), @tensor.Float32)
  assert_eq(t_raw.get_f32([0]), (1.0 : Float))
  assert_eq(t_raw.get_f32([1]), (-2.5 : Float))
  assert_eq(t_raw.get_f32([2]), (0.5 : Float))
  assert_eq(t_raw.get_f32([3]), (0.0 : Float))
  // 2^-24 ≈ 5.9604645e-8
  let expected_sub = Float::from_double(1.0 / 16777216.0)
  assert_eq(t_raw.get_f32([4]), expected_sub)

  let tp_i32 = @schema_test.TensorProto::default()
  tp_i32.name = Some("w_f16_i32")
  tp_i32.data_type = Some(10)
  tp_i32.dims = [(2 : Int64)]
  tp_i32.int32_data = [0x3C00, 0xC100]
  let t_i32 = @onnx_proto.decode_tensor_proto(tp_i32)
  assert_eq(t_i32.get_f32([0]), (1.0 : Float))
  assert_eq(t_i32.get_f32([1]), (-2.5 : Float))
}
