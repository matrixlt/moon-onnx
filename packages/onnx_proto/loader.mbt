///|
/// 从 Bytes 中按小端序（Little-Endian）读取 32 位无符号整数。
/// ONNX 规范规定 TensorProto.raw_data 统一使用小端序字节排列。
fn read_u32_le(bytes : Bytes, offset : Int32) -> UInt32 {
  let b0 = bytes[offset].to_uint()
  let b1 = bytes[offset + 1].to_uint()
  let b2 = bytes[offset + 2].to_uint()
  let b3 = bytes[offset + 3].to_uint()
  b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

///|
/// 从 Bytes 中按小端序读取 64 位无符号整数。
fn read_u64_le(bytes : Bytes, offset : Int32) -> UInt64 {
  let lo = read_u32_le(bytes, offset).to_uint64()
  let hi = read_u32_le(bytes, offset + 4).to_uint64()
  lo | (hi << 32)
}

///|
/// 将 IEEE 754 半精度浮点（binary16 / FP16）的 16 位比特模式精确转换为单精度 `Float`（binary32）。
/// 完整覆盖规格化数、非规格化数（subnormal）、有符号零（±0.0）与无穷大/NaN。
pub fn f16_bits_to_f32(h : UInt32) -> Float {
  let sign = (h >> 15) & 1U
  let exp = (h >> 10) & 0x1FU
  let frac = h & 0x3FFU
  let bits : UInt32 = if exp == 0U {
    if frac == 0U {
      sign << 31
    } else {
      // 非规格化半精度数：左移直至隐含高位 1 落入 bit 10
      let mut f = frac
      let mut e = 113U // 127 - 14
      while (f & 0x400U) == 0U {
        f = f << 1
        e = e - 1U
      }
      (sign << 31) | (e << 23) | ((f & 0x3FFU) << 13)
    }
  } else if exp == 31U {
    (sign << 31) | 0x7F800000U | (frac << 13)
  } else {
    (sign << 31) | ((exp + 112U) << 23) | (frac << 13)
  }
  Float::reinterpret_from_uint(bits)
}

///|
/// 将 ONNX TensorProto.DataType 数值枚举映射为引擎内部的 DType。
/// 参考 proto/onnx.proto 中 TensorProto.DataType 定义：
/// 1=FLOAT, 2=UINT8, 3=INT8, 6=INT32, 7=INT64, 9=BOOL, 10=FLOAT16（加载期自动无损提升为 Float32）, 11=DOUBLE。
pub fn onnx_dtype_to_dtype(code : Int32) -> @tensor.DType raise ModelError {
  match code {
    1 | 10 => @tensor.Float32
    2 => @tensor.UInt8
    3 => @tensor.Int8
    6 => @tensor.Int32
    7 => @tensor.Int64
    9 => @tensor.Bool
    11 => @tensor.Float64
    _ => raise UnsupportedDataType(code)
  }
}

///|
/// 将 ONNX TensorProto 解码为引擎核心的 Tensor 对象。
/// 自动兼容 ONNX 允许的多种存储方式：
/// 1. 优先读取紧凑二进制 `raw_data`（含 2 字节 IEEE 754 `FLOAT16` 自动提升至 `Float32`）
/// 2. 回退读取类型化 repeated 字段（`float_data` / `int64_data` / `int32_data` / `double_data`）
pub fn decode_tensor_proto(
  tp : @schema.TensorProto,
) -> @tensor.Tensor raise ModelError {
  let name = match tp.name {
    Some(n) => n
    None => "<unnamed>"
  }

  // 检查是否使用了外部权重文件（目前 MNIST 等轻量模型均为内嵌权重，后续可扩展外部文件加载）
  match tp.data_location {
    Some(@schema.TensorProto_DataLocation::EXTERNAL) =>
      raise InvalidTensor(
        "tensor '\{name}' uses external_data, which is not yet supported in MVP",
      )
    _ => ()
  }

  let dtype_code = match tp.data_type {
    Some(c) => c
    None => raise InvalidTensor("tensor '\{name}' is missing data_type field")
  }

  // 将 Int64 维度数组转换为 Int 形状数组并校验非负性
  let shape = Array::make(tp.dims.length(), 0)
  let mut expected = 1
  for i = 0; i < tp.dims.length(); i = i + 1 {
    let d = tp.dims[i].to_int()
    if d < 0 {
      raise InvalidTensor("tensor '\{name}' has negative dim \{d} at axis \{i}")
    }
    shape[i] = d
    expected = expected * d
  }

  // 特殊处理 ONNX FLOAT16 (code=10)：每元素占 2 字节，加载期精确转为 Float32
  if dtype_code == 10 {
    match tp.raw_data {
      Some(raw) if raw.length() > 0 => {
        let expected_bytes = expected * 2
        if raw.length() != expected_bytes {
          raise InvalidTensor(
            "tensor '\{name}' float16 raw_data has \{raw.length()} bytes, expected \{expected_bytes}",
          )
        }
        let buf = FixedArray::makei(expected, fn(i) {
          let lo = raw[i * 2].to_uint()
          let hi = raw[i * 2 + 1].to_uint()
          f16_bits_to_f32(lo | (hi << 8))
        })
        return @tensor.Tensor::from_f32(shape, buf) catch {
          e => raise InvalidTensor("tensor '\{name}': \{e}")
        }
      }
      _ => {
        // ONNX 规范规定非 raw_data 的 FLOAT16 存放在 int32_data 的低 16 位中
        if tp.int32_data.length() != expected {
          raise InvalidTensor(
            "tensor '\{name}' float16 int32_data length \{tp.int32_data.length()} != expected \{expected}",
          )
        }
        let buf = FixedArray::makei(expected, fn(i) {
          f16_bits_to_f32(tp.int32_data[i].reinterpret_as_uint() & 0xFFFFU)
        })
        return @tensor.Tensor::from_f32(shape, buf) catch {
          e => raise InvalidTensor("tensor '\{name}': \{e}")
        }
      }
    }
  }

  let dtype = onnx_dtype_to_dtype(dtype_code)

  // 优先解析 raw_data（小端序二进制）
  match tp.raw_data {
    Some(raw) if raw.length() > 0 =>
      return decode_raw_data(name, shape, expected, dtype, raw)
    _ => ()
  }

  // 回退解析类型化 repeated 字段
  match dtype {
    @tensor.Float32 => {
      if tp.float_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' float_data length \{tp.float_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) { tp.float_data[i] })
      @tensor.Tensor::from_f32(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Float64 => {
      if tp.double_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' double_data length \{tp.double_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) { tp.double_data[i] })
      @tensor.Tensor::from_f64(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Int64 => {
      if tp.int64_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' int64_data length \{tp.int64_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) { tp.int64_data[i] })
      @tensor.Tensor::from_i64(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Int32 => {
      if tp.int32_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' int32_data length \{tp.int32_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) { tp.int32_data[i] })
      @tensor.Tensor::from_i32(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Bool => {
      // ONNX 规范允许 Bool 存储在 int32_data 中（非 0 为 true）
      if tp.int32_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' bool int32_data length \{tp.int32_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) { tp.int32_data[i] != 0 })
      @tensor.Tensor::from_bool(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.UInt8 => {
      if tp.int32_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' uint8 int32_data length \{tp.int32_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) {
        (tp.int32_data[i] & 0xFF).to_byte()
      })
      @tensor.Tensor::from_u8(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Int8 => {
      if tp.int32_data.length() != expected {
        raise InvalidTensor(
          "tensor '\{name}' int8 int32_data length \{tp.int32_data.length()} != expected \{expected}",
        )
      }
      let buf = FixedArray::makei(expected, fn(i) {
        (tp.int32_data[i] & 0xFF).to_byte()
      })
      @tensor.Tensor::from_i8(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
  }
}

///|
/// 将小端序 `raw_data` 字节流按目标 DType 解码为 Tensor。
fn decode_raw_data(
  name : String,
  shape : Array[Int32],
  expected : Int32,
  dtype : @tensor.DType,
  raw : Bytes,
) -> @tensor.Tensor raise ModelError {
  let elem_bytes = dtype.byte_size()
  let expected_bytes = expected * elem_bytes
  if raw.length() != expected_bytes {
    raise InvalidTensor(
      "tensor '\{name}' raw_data has \{raw.length()} bytes, expected \{expected_bytes} bytes for \{expected} x \{dtype}",
    )
  }
  match dtype {
    @tensor.Float32 => {
      let buf = FixedArray::makei(expected, fn(i) {
        Float::reinterpret_from_uint(read_u32_le(raw, i * 4))
      })
      @tensor.Tensor::from_f32(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Float64 => {
      let buf = FixedArray::makei(expected, fn(i) {
        read_u64_le(raw, i * 8).reinterpret_as_double()
      })
      @tensor.Tensor::from_f64(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Int64 => {
      let buf = FixedArray::makei(expected, fn(i) {
        read_u64_le(raw, i * 8).reinterpret_as_int64()
      })
      @tensor.Tensor::from_i64(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Int32 => {
      let buf = FixedArray::makei(expected, fn(i) {
        read_u32_le(raw, i * 4).reinterpret_as_int()
      })
      @tensor.Tensor::from_i32(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Bool => {
      let buf = FixedArray::makei(expected, fn(i) { raw[i] != b'\x00' })
      @tensor.Tensor::from_bool(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.UInt8 => {
      let buf = FixedArray::makei(expected, fn(i) { raw[i] })
      @tensor.Tensor::from_u8(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
    @tensor.Int8 => {
      let buf = FixedArray::makei(expected, fn(i) { raw[i] })
      @tensor.Tensor::from_i8(shape, buf) catch {
        e => raise InvalidTensor("tensor '\{name}': \{e}")
      }
    }
  }
}

///|
/// 将 ONNX ValueInfoProto 转换为内部 ValueSpec。
fn parse_value_info(vi : @schema.ValueInfoProto) -> ValueSpec {
  let name = match vi.name {
    Some(n) => n
    None => "<unnamed>"
  }
  let mut dtype : @tensor.DType? = None
  let shape : Array[DimSpec] = []
  match vi.type_ {
    Some(tp) =>
      match tp.value {
        @schema.TypeProto_Value::TensorType(tt) => {
          match tt.elem_type {
            Some(code) =>
              dtype = Some(onnx_dtype_to_dtype(code)) catch { _ => None }
            None => ()
          }
          match tt.shape {
            Some(ts) =>
              for i = 0; i < ts.dim.length(); i = i + 1 {
                let d = ts.dim[i]
                match d.value {
                  @schema.TensorShapeProto_Dimension_Value::DimValue(v) =>
                    shape.push(Static(v.to_int()))
                  @schema.TensorShapeProto_Dimension_Value::DimParam(p) =>
                    shape.push(Symbolic(p))
                  @schema.TensorShapeProto_Dimension_Value::NotSet =>
                    shape.push(Unknown)
                }
              }
            None => ()
          }
        }
        _ => ()
      }
    None => ()
  }
  { name, dtype, shape, }
}

///|
/// 从内存中的 `.onnx` 二进制字节流解析完整的 OnnxModel。
/// 该函数不依赖文件系统，天然支持 Native / Wasm / JS 全平台复用。
pub fn load_model_from_bytes(bytes : Bytes) -> OnnxModel raise ModelError {
  let reader = @protobuf.BytesReader::from_bytes(bytes)
  let proto_model = @schema.ModelProto::read(reader) catch {
    e => raise ParseError("failed to decode ONNX ModelProto: \{e}")
  }
  load_model_from_proto(proto_model)
}

///|
/// 从已构造或反序列化的 `@schema.ModelProto` 构建高层 `OnnxModel`。
/// 便于模型转换工具与单元测试在内存中直接构造模型，无需落盘或手动编码 Protobuf。
pub fn load_model_from_proto(
  proto_model : @schema.ModelProto,
) -> OnnxModel raise ModelError {
  let graph = match proto_model.graph {
    Some(g) => g
    None => raise MissingGraph("ONNX ModelProto does not contain a graph")
  }

  // 解析 Opset 导入列表
  let opsets : Array[OpsetImport] = []
  for i = 0; i < proto_model.opset_import.length(); i = i + 1 {
    let op = proto_model.opset_import[i]
    let domain = match op.domain {
      Some(d) => d
      None => ""
    }
    let version = match op.version {
      Some(v) => v
      None => (0 : Int64)
    }
    opsets.push({ domain, version, })
  }

  // 解码全部 Initializer 权重张量
  let initializers : Map[String, @tensor.Tensor] = Map([])
  let initializer_order : Array[String] = []
  for i = 0; i < graph.initializer.length(); i = i + 1 {
    let init_proto = graph.initializer[i]
    let init_name = match init_proto.name {
      Some(n) => n
      None => raise InvalidTensor("initializer at index \{i} has no name")
    }
    let tensor = decode_tensor_proto(init_proto)
    initializers[init_name] = tensor
    initializer_order.push(init_name)
  }

  // 提取图输入签名；旧版 ONNX 会把 Initializer 同时列在 graph.input 中，这里自动过滤
  let inputs : Array[ValueSpec] = []
  for i = 0; i < graph.input.length(); i = i + 1 {
    let spec = parse_value_info(graph.input[i])
    if !initializers.contains(spec.name) {
      inputs.push(spec)
    }
  }

  // 提取图输出签名
  let outputs : Array[ValueSpec] = []
  for i = 0; i < graph.output.length(); i = i + 1 {
    outputs.push(parse_value_info(graph.output[i]))
  }

  let ir_version = match proto_model.ir_version {
    Some(v) => v
    None => (0 : Int64)
  }
  let producer_name = match proto_model.producer_name {
    Some(s) => s
    None => "<unknown>"
  }
  let producer_version = match proto_model.producer_version {
    Some(s) => s
    None => ""
  }
  let graph_name = match graph.name {
    Some(s) => s
    None => "<unnamed>"
  }

  {
    ir_version,
    producer_name,
    producer_version,
    opsets,
    graph_name,
    inputs,
    outputs,
    initializers,
    initializer_order,
    nodes: graph.node,
  }
}

///|
/// 生成人类可读的 ONNX 模型结构摘要报告，供 CLI `inspect` 与调试日志使用。
pub fn OnnxModel::format_summary(self : OnnxModel) -> String {
  let buf = StringBuilder()
  buf.write_string("=== ONNX Model Summary ===\n")
  buf.write_string("Graph Name  : \{self.graph_name}\n")
  buf.write_string("IR Version  : \{self.ir_version}\n")
  if self.producer_version != "" {
    buf.write_string(
      "Producer    : \{self.producer_name} (\{self.producer_version})\n",
    )
  } else {
    buf.write_string("Producer    : \{self.producer_name}\n")
  }

  // Opset 列表
  buf.write_string("Opsets      : ")
  for i = 0; i < self.opsets.length(); i = i + 1 {
    if i > 0 {
      buf.write_string(", ")
    }
    let op = self.opsets[i]
    let dom = if op.domain == "" { "ai.onnx" } else { op.domain }
    buf.write_string("\{dom}@\{op.version}")
  }
  buf.write_char('\n')

  // 输入签名
  buf.write_string("\nInputs (\{self.inputs.length()}):\n")
  for i = 0; i < self.inputs.length(); i = i + 1 {
    buf.write_string("  - \{self.inputs[i].format()}\n")
  }

  // 输出签名
  buf.write_string("\nOutputs (\{self.outputs.length()}):\n")
  for i = 0; i < self.outputs.length(); i = i + 1 {
    buf.write_string("  - \{self.outputs[i].format()}\n")
  }

  // 权重统计
  let mut total_params = 0
  buf.write_string("\nInitializers (\{self.initializer_order.length()}):\n")
  for i = 0; i < self.initializer_order.length(); i = i + 1 {
    let name = self.initializer_order[i]
    match self.initializers.get(name) {
      Some(t) => {
        total_params = total_params + t.numel()
        buf.write_string("  - \{name}: \{t.summary()}\n")
      }
      None => ()
    }
  }
  buf.write_string("  Total Parameters: \{total_params}\n")

  // 计算图节点与算子统计
  buf.write_string("\nNodes (\{self.nodes.length()}):\n")
  for i = 0; i < self.nodes.length(); i = i + 1 {
    let node = self.nodes[i]
    let op_type = match node.op_type {
      Some(s) => s
      None => "<unknown>"
    }
    let node_name = match node.name {
      Some(s) if s != "" => s
      _ => "node_\{i}"
    }
    buf.write_string("  [\{i}] \{op_type} (\{node_name}): ")
    buf.write_string(join_names(node.input))
    buf.write_string(" -> ")
    buf.write_string(join_names(node.output))
    buf.write_char('\n')
  }
  buf.to_string()
}

///|
/// 将节点输入或输出名称数组拼接为 `[a, b]` 形式的字符串。
fn join_names(items : Array[String]) -> String {
  let buf = StringBuilder()
  buf.write_char('[')
  for i = 0; i < items.length(); i = i + 1 {
    if i > 0 {
      buf.write_string(", ")
    }
    buf.write_string(items[i])
  }
  buf.write_char(']')
  buf.to_string()
}
