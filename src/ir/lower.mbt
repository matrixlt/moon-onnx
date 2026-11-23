///|
/// 在节点的 AttributeProto 列表中按名称查找属性。
fn find_attr(
  attrs : Array[@proto.AttributeProto],
  target_name : String,
) -> @proto.AttributeProto? {
  for i = 0; i < attrs.length(); i = i + 1 {
    match attrs[i].name {
      Some(n) if n == target_name => return Some(attrs[i])
      _ => ()
    }
  }
  None
}

///|
/// 读取整数属性（Int64 -> Int），缺失时返回默认值。
fn get_attr_int(
  attrs : Array[@proto.AttributeProto],
  name : String,
  default_val : Int32,
) -> Int32 {
  match find_attr(attrs, name) {
    Some(a) =>
      match a.i {
        Some(v) => v.to_int()
        None => default_val
      }
    None => default_val
  }
}

///|
/// 读取浮点属性，缺失时返回默认值。
fn get_attr_float(
  attrs : Array[@proto.AttributeProto],
  name : String,
  default_val : Float,
) -> Float {
  match find_attr(attrs, name) {
    Some(a) =>
      match a.f {
        Some(v) => v
        None => default_val
      }
    None => default_val
  }
}

///|
/// 读取整数列表属性，缺失时返回 None。
fn get_attr_ints_opt(
  attrs : Array[@proto.AttributeProto],
  name : String,
) -> Array[Int32]? {
  match find_attr(attrs, name) {
    Some(a) => {
      let out = Array::make(a.ints.length(), 0)
      for i = 0; i < a.ints.length(); i = i + 1 {
        out[i] = a.ints[i].to_int()
      }
      Some(out)
    }
    None => None
  }
}

///|
/// 读取字节串类型的字符串属性（ONNX Protobuf 中 AttributeProto.s 存储为 UTF-8 Bytes）。
fn get_attr_string(
  attrs : Array[@proto.AttributeProto],
  name : String,
  default_val : String,
) -> String {
  match find_attr(attrs, name) {
    Some(a) =>
      match a.s {
        Some(raw) => bytes_ascii_to_string(raw)
        None => default_val
      }
    None => default_val
  }
}

///|
/// 将 ONNX 属性中常用的 ASCII/UTF-8 字节串（如 "NOTSET"、"SAME_UPPER"）转为 String。
fn bytes_ascii_to_string(raw : Bytes) -> String {
  let buf = StringBuilder()
  for i = 0; i < raw.length(); i = i + 1 {
    buf.write_char(Int32::unsafe_to_char(raw[i].to_int()))
  }
  buf.to_string()
}

///|
/// 解析 ONNX auto_pad 属性字符串为强类型 AutoPad 枚举。
fn parse_auto_pad(
  attrs : Array[@proto.AttributeProto],
  node_name : String,
) -> AutoPad raise IrError {
  let raw = get_attr_string(attrs, "auto_pad", "NOTSET")
  match raw {
    "" | "NOTSET" => NotSet
    "SAME_UPPER" => SameUpper
    "SAME_LOWER" => SameLower
    "VALID" => Valid
    other =>
      raise InvalidAttribute(
        "node '\{node_name}': unsupported auto_pad value '\{other}'",
      )
  }
}

///|
/// 将单个 ONNX NodeProto Lowering 为强类型 IrNode。
fn lower_node(
  node : @proto.NodeProto,
  idx : Int32,
  opset_version : Int64,
) -> IrNode raise IrError {
  let op_type = match node.op_type {
    Some(s) => s
    None => raise InvalidGraph("node at index \{idx} has empty op_type")
  }
  let node_name = match node.name {
    Some(s) if s != "" => s
    _ => "\{op_type}_\{idx}"
  }
  let attrs = node.attribute

  let op : Op = match op_type {
    "Add" => Add
    "Sub" => Sub
    "Mul" => Mul
    "Relu" => Relu
    "Identity" => Identity
    "MatMul" => MatMul
    "Softmax" => {
      // ONNX 规范：Opset < 13 默认 axis = 1；Opset >= 13 默认 axis = -1
      let default_axis = if opset_version > 0 && opset_version < 13 {
        1
      } else {
        -1
      }
      Softmax(axis=get_attr_int(attrs, "axis", default_axis))
    }
    "Flatten" => Flatten(axis=get_attr_int(attrs, "axis", 1))
    "Reshape" => Reshape(allowzero=get_attr_int(attrs, "allowzero", 0) != 0)
    "Transpose" => {
      let perm = match get_attr_ints_opt(attrs, "perm") {
        Some(p) => p
        None => []
      }
      Transpose(perm~)
    }
    "Squeeze" => Squeeze(axes=get_attr_ints_opt(attrs, "axes"))
    "Unsqueeze" => Unsqueeze(axes=get_attr_ints_opt(attrs, "axes"))
    "Constant" => {
      // 支持最常见的 value (TensorProto) 以及 value_float / value_int / value_ints / value_floats
      let tensor = match find_attr(attrs, "value") {
        Some(a) =>
          match a.t {
            Some(tp) =>
              @model.decode_tensor_proto(tp) catch {
                e =>
                  raise InvalidAttribute(
                    "Constant '\{node_name}' failed to decode value tensor: \{e}",
                  )
              }
            None =>
              raise InvalidAttribute(
                "Constant '\{node_name}' attribute 'value' has no tensor",
              )
          }
        None =>
          match find_attr(attrs, "value_ints") {
            Some(a) => {
              let buf = FixedArray::makei(a.ints.length(), fn(i) { a.ints[i] })
              @tensor.Tensor::from_i64([a.ints.length()], buf) catch {
                e => raise InvalidAttribute("Constant '\{node_name}': \{e}")
              }
            }
            None =>
              match find_attr(attrs, "value_floats") {
                Some(a) => {
                  let buf = FixedArray::makei(a.floats.length(), fn(i) {
                    a.floats[i]
                  })
                  @tensor.Tensor::from_f32([a.floats.length()], buf) catch {
                    e => raise InvalidAttribute("Constant '\{node_name}': \{e}")
                  }
                }
                None =>
                  raise InvalidAttribute(
                    "Constant '\{node_name}' missing supported value attribute",
                  )
              }
          }
      }
      Constant(value=tensor)
    }
    "Gemm" => {
      let alpha = get_attr_float(attrs, "alpha", (1.0 : Float))
      let beta = get_attr_float(attrs, "beta", (1.0 : Float))
      let trans_a = get_attr_int(attrs, "transA", 0) != 0
      let trans_b = get_attr_int(attrs, "transB", 0) != 0
      Gemm(alpha~, beta~, trans_a~, trans_b~)
    }
    "Conv" => {
      let kernel_shape = get_attr_ints_opt(attrs, "kernel_shape")
      let strides = match get_attr_ints_opt(attrs, "strides") {
        Some(v) => v
        None => [1, 1]
      }
      let pads = match get_attr_ints_opt(attrs, "pads") {
        Some(v) => v
        None => [0, 0, 0, 0]
      }
      let dilations = match get_attr_ints_opt(attrs, "dilations") {
        Some(v) => v
        None => [1, 1]
      }
      let group = get_attr_int(attrs, "group", 1)
      let auto_pad = parse_auto_pad(attrs, node_name)
      Conv(kernel_shape~, strides~, pads~, dilations~, group~, auto_pad~)
    }
    "MaxPool" => {
      let kernel_shape = match get_attr_ints_opt(attrs, "kernel_shape") {
        Some(v) => v
        None =>
          raise InvalidAttribute(
            "MaxPool '\{node_name}' requires kernel_shape attribute",
          )
      }
      let strides = match get_attr_ints_opt(attrs, "strides") {
        Some(v) => v
        None => [1, 1]
      }
      let pads = match get_attr_ints_opt(attrs, "pads") {
        Some(v) => v
        None => [0, 0, 0, 0]
      }
      let dilations = match get_attr_ints_opt(attrs, "dilations") {
        Some(v) => v
        None => [1, 1]
      }
      let ceil_mode = get_attr_int(attrs, "ceil_mode", 0) != 0
      let auto_pad = parse_auto_pad(attrs, node_name)
      MaxPool(kernel_shape~, strides~, pads~, dilations~, ceil_mode~, auto_pad~)
    }
    "AveragePool" => {
      let kernel_shape = match get_attr_ints_opt(attrs, "kernel_shape") {
        Some(v) => v
        None =>
          raise InvalidAttribute(
            "AveragePool '\{node_name}' requires kernel_shape attribute",
          )
      }
      let strides = match get_attr_ints_opt(attrs, "strides") {
        Some(v) => v
        None => [1, 1]
      }
      let pads = match get_attr_ints_opt(attrs, "pads") {
        Some(v) => v
        None => [0, 0, 0, 0]
      }
      let ceil_mode = get_attr_int(attrs, "ceil_mode", 0) != 0
      let count_include_pad = get_attr_int(attrs, "count_include_pad", 0) != 0
      let auto_pad = parse_auto_pad(attrs, node_name)
      AveragePool(
        kernel_shape~,
        strides~,
        pads~,
        ceil_mode~,
        count_include_pad~,
        auto_pad~,
      )
    }
    "BatchNormalization" => {
      let epsilon = get_attr_float(attrs, "epsilon", (1.0e-5 : Float))
      BatchNormalization(epsilon~)
    }
    other => raise UnsupportedOp(other)
  }

  {
    name: node_name,
    op,
    inputs: node.input.copy(),
    outputs: node.output.copy(),
  }
}

///|
/// 对一组 IrNode 执行依赖校验与 Kahn 拓扑排序。
/// 1. 校验每个非空输入名称均由图输入、权重初始化张量或某个节点产出。
/// 2. 在存在多个就绪节点时优先保持原始声明顺序，保证执行顺序稳定可复现。
/// 3. 若图中存在环路，抛出 CyclicGraph 错误。
pub fn topo_sort_nodes(
  nodes : Array[IrNode],
  graph_inputs : Array[String],
  initializers : Map[String, @tensor.Tensor],
) -> Array[IrNode] raise IrError {
  let n = nodes.length()
  // 记录每个张量名由哪个节点产出（tensor_name -> producer_node_idx）
  let producer : Map[String, Int32] = Map([])
  for i = 0; i < n; i = i + 1 {
    let outs = nodes[i].outputs
    for j = 0; j < outs.length(); j = j + 1 {
      let out_name = outs[j]
      if out_name != "" {
        producer[out_name] = i
      }
    }
  }

  // 记录已存在的外部输入集合
  let known_external : Map[String, Bool] = Map([])
  for i = 0; i < graph_inputs.length(); i = i + 1 {
    known_external[graph_inputs[i]] = true
  }

  // 构建邻接表与入度数组
  let in_degree = Array::make(n, 0)
  let adj : Array[Array[Int32]] = Array::makei(n, fn(_) { [] })
  for i = 0; i < n; i = i + 1 {
    let ins = nodes[i].inputs
    for j = 0; j < ins.length(); j = j + 1 {
      let in_name = ins[j]
      // ONNX 允许可选输入为空字符串 ""，直接跳过
      if in_name == "" {
        continue
      }
      match producer.get(in_name) {
        Some(p_idx) => {
          adj[p_idx].push(i)
          in_degree[i] = in_degree[i] + 1
        }
        None =>
          if !known_external.contains(in_name) &&
            !initializers.contains(in_name) {
            raise MissingInput(
              "node '\{nodes[i].name}' references undefined tensor '\{in_name}'",
            )
          }
      }
    }
  }

  // Kahn 算法：按索引顺序扫描就绪队列，确保稳定拓扑序
  let queue : Array[Int32] = []
  for i = 0; i < n; i = i + 1 {
    if in_degree[i] == 0 {
      queue.push(i)
    }
  }

  let sorted : Array[IrNode] = []
  let mut head = 0
  while head < queue.length() {
    let u = queue[head]
    head = head + 1
    sorted.push(nodes[u])
    let neighbors = adj[u]
    for k = 0; k < neighbors.length(); k = k + 1 {
      let v = neighbors[k]
      in_degree[v] = in_degree[v] - 1
      if in_degree[v] == 0 {
        queue.push(v)
      }
    }
  }

  if sorted.length() != n {
    raise CyclicGraph(
      "graph contains a dependency cycle (sorted \{sorted.length()} of \{n} nodes)",
    )
  }
  sorted
}

///|
/// 将解析完成的 OnnxModel 转换为经过校验与拓扑排序的强类型 IrGraph。
pub fn lower_model(m : @model.OnnxModel) -> IrGraph raise IrError {
  let opset_version = match m.default_opset_version() {
    Some(v) => v
    None => (17 : Int64)
  }

  let raw_nodes : Array[IrNode] = []
  for i = 0; i < m.nodes.length(); i = i + 1 {
    raw_nodes.push(lower_node(m.nodes[i], i, opset_version))
  }

  let input_names = Array::makei(m.inputs.length(), fn(i) { m.inputs[i].name })
  let output_names = Array::makei(m.outputs.length(), fn(i) {
    m.outputs[i].name
  })

  // 复制 initializer 映射，避免图优化阶段直接修改原始模型对象
  let initializers : Map[String, @tensor.Tensor] = Map([])
  for i = 0; i < m.initializer_order.length(); i = i + 1 {
    let key = m.initializer_order[i]
    match m.initializers.get(key) {
      Some(t) => initializers[key] = t
      None => ()
    }
  }

  let sorted_nodes = topo_sort_nodes(raw_nodes, input_names, initializers)
  {
    name: m.graph_name,
    inputs: input_names,
    outputs: output_names,
    initializers,
    nodes: sorted_nodes,
  }
}
