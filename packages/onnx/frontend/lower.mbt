///|
/// 在节点的 AttributeProto 列表中按名称查找属性。
fn find_attr(
  attrs : Array[@schema.AttributeProto],
  target_name : String,
) -> @schema.AttributeProto? {
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
  attrs : Array[@schema.AttributeProto],
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
  attrs : Array[@schema.AttributeProto],
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
/// 读取可选整数属性，缺失时返回 None。
fn get_attr_int_opt(
  attrs : Array[@schema.AttributeProto],
  name : String,
) -> Int32? {
  match find_attr(attrs, name) {
    Some(a) =>
      match a.i {
        Some(v) => Some(v.to_int())
        None => None
      }
    None => None
  }
}

///|
/// 读取可选浮点属性，缺失时返回 None。
fn get_attr_float_opt(
  attrs : Array[@schema.AttributeProto],
  name : String,
) -> Float? {
  match find_attr(attrs, name) {
    Some(a) => a.f
    None => None
  }
}

///|
/// 将 ONNX `TensorProto.DataType` 整型编码解析为 `@tensor.DType`。
fn parse_onnx_dtype(
  code : Int32,
  node_name : String,
) -> @tensor.DType raise @nn_ir.IrError {
  match code {
    1 | 10 => @tensor.Float32
    2 => @tensor.UInt8
    3 => @tensor.Int8
    6 => @tensor.Int32
    7 => @tensor.Int64
    9 => @tensor.Bool
    11 => @tensor.Float64
    other =>
      raise @nn_ir.InvalidAttribute(
        "Cast '\{node_name}': unsupported ONNX dtype code \{other}",
      )
  }
}

///|
/// 读取整数列表属性，缺失时返回 None。
fn get_attr_ints_opt(
  attrs : Array[@schema.AttributeProto],
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
/// 将 ONNX 属性中常用的 ASCII 字节串（如 "NOTSET"、"SAME_UPPER"）转为 String。
fn bytes_ascii_to_string(raw : Bytes) -> String {
  let buf = StringBuilder()
  for i = 0; i < raw.length(); i = i + 1 {
    buf.write_char(Int32::unsafe_to_char(raw[i].to_int()))
  }
  buf.to_string()
}

///|
/// 读取字节串类型的字符串属性。
fn get_attr_string(
  attrs : Array[@schema.AttributeProto],
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
/// 解析 ONNX `auto_pad` 属性字符串为强类型 `@nn_ir.AutoPad`。
fn parse_auto_pad(
  attrs : Array[@schema.AttributeProto],
  node_name : String,
) -> @nn_ir.AutoPad raise @nn_ir.IrError {
  let raw = get_attr_string(attrs, "auto_pad", "NOTSET")
  match raw {
    "" | "NOTSET" => @nn_ir.NotSet
    "SAME_UPPER" => @nn_ir.SameUpper
    "SAME_LOWER" => @nn_ir.SameLower
    "VALID" => @nn_ir.Valid
    other =>
      raise @nn_ir.InvalidAttribute(
        "node '\{node_name}': unsupported auto_pad value '\{other}'",
      )
  }
}

///|
/// 将单个 ONNX `NodeProto` Lowering 为格式无关的强类型 `@nn_ir.IrNode`。
fn lower_node(
  node : @schema.NodeProto,
  idx : Int32,
  opset_version : Int64,
  outer_scope : Map[String, Bool],
) -> @nn_ir.IrNode raise @nn_ir.IrError {
  let op_type = match node.op_type {
    Some(s) => s
    None => raise @nn_ir.InvalidGraph("node at index \{idx} has empty op_type")
  }
  let node_name = match node.name {
    Some(s) if s != "" => s
    _ => "\{op_type}_\{idx}"
  }
  let attrs = node.attribute

  let op : @nn_ir.Op = match op_type {
    "Add" => @nn_ir.Add
    "Sub" => @nn_ir.Sub
    "Mul" => @nn_ir.Mul
    "Div" => @nn_ir.Div
    "Pow" => @nn_ir.Pow
    "Sqrt" => @nn_ir.Sqrt
    "Exp" => @nn_ir.Exp
    "Neg" => @nn_ir.Neg
    "Relu" => @nn_ir.Relu
    "Clip" =>
      @nn_ir.Clip(
        min=get_attr_float_opt(attrs, "min"),
        max=get_attr_float_opt(attrs, "max"),
      )
    "Sigmoid" => @nn_ir.Sigmoid
    "Tanh" => @nn_ir.Tanh
    "Gelu" => @nn_ir.Gelu
    "HardSwish" => @nn_ir.HardSwish
    "LeakyRelu" =>
      @nn_ir.LeakyRelu(alpha=get_attr_float(attrs, "alpha", (0.01 : Float)))
    "Identity" => @nn_ir.Identity
    "MatMul" => @nn_ir.MatMul
    "Softmax" => {
      // ONNX 规范：Opset < 13 默认 axis = 1；Opset >= 13 默认 axis = -1
      let default_axis = if opset_version > 0 && opset_version < 13 {
        1
      } else {
        -1
      }
      @nn_ir.Softmax(axis=get_attr_int(attrs, "axis", default_axis))
    }
    "Flatten" => @nn_ir.Flatten(axis=get_attr_int(attrs, "axis", 1))
    "Reshape" =>
      @nn_ir.Reshape(allowzero=get_attr_int(attrs, "allowzero", 0) != 0)
    "Transpose" => {
      let perm = match get_attr_ints_opt(attrs, "perm") {
        Some(p) => p
        None => []
      }
      @nn_ir.Transpose(perm~)
    }
    "Squeeze" => @nn_ir.Squeeze(axes=get_attr_ints_opt(attrs, "axes"))
    "Unsqueeze" => @nn_ir.Unsqueeze(axes=get_attr_ints_opt(attrs, "axes"))
    "Constant" => {
      let tensor = match find_attr(attrs, "value") {
        Some(a) =>
          match a.t {
            Some(tp) =>
              @onnx_proto.decode_tensor_proto(tp) catch {
                e =>
                  raise @nn_ir.InvalidAttribute(
                    "Constant '\{node_name}' failed to decode value tensor: \{e}",
                  )
              }
            None =>
              raise @nn_ir.InvalidAttribute(
                "Constant '\{node_name}' attribute 'value' has no tensor",
              )
          }
        None =>
          match find_attr(attrs, "value_ints") {
            Some(a) => {
              let buf = FixedArray::makei(a.ints.length(), fn(i) { a.ints[i] })
              @tensor.Tensor::from_i64([a.ints.length()], buf) catch {
                e =>
                  raise @nn_ir.InvalidAttribute("Constant '\{node_name}': \{e}")
              }
            }
            None =>
              match find_attr(attrs, "value_floats") {
                Some(a) => {
                  let buf = FixedArray::makei(a.floats.length(), fn(i) {
                    a.floats[i]
                  })
                  @tensor.Tensor::from_f32([a.floats.length()], buf) catch {
                    e =>
                      raise @nn_ir.InvalidAttribute(
                        "Constant '\{node_name}': \{e}",
                      )
                  }
                }
                None =>
                  match find_attr(attrs, "value_float") {
                    Some(a) =>
                      match a.f {
                        Some(v) =>
                          @tensor.Tensor::from_f32([], FixedArray::make(1, v)) catch {
                            e =>
                              raise @nn_ir.InvalidAttribute(
                                "Constant '\{node_name}': \{e}",
                              )
                          }
                        None =>
                          raise @nn_ir.InvalidAttribute(
                            "Constant '\{node_name}' value_float empty",
                          )
                      }
                    None =>
                      match find_attr(attrs, "value_int") {
                        Some(a) =>
                          match a.i {
                            Some(v) =>
                              @tensor.Tensor::from_i64(
                                [],
                                FixedArray::make(1, v),
                              ) catch {
                                e =>
                                  raise @nn_ir.InvalidAttribute(
                                    "Constant '\{node_name}': \{e}",
                                  )
                              }
                            None =>
                              raise @nn_ir.InvalidAttribute(
                                "Constant '\{node_name}' value_int empty",
                              )
                          }
                        None =>
                          raise @nn_ir.InvalidAttribute(
                            "Constant '\{node_name}' missing supported value attribute",
                          )
                      }
                  }
              }
          }
      }
      @nn_ir.Constant(value=tensor)
    }
    "Gemm" => {
      let alpha = get_attr_float(attrs, "alpha", (1.0 : Float))
      let beta = get_attr_float(attrs, "beta", (1.0 : Float))
      let trans_a = get_attr_int(attrs, "transA", 0) != 0
      let trans_b = get_attr_int(attrs, "transB", 0) != 0
      @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~)
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
      @nn_ir.Conv(
        kernel_shape~,
        strides~,
        pads~,
        dilations~,
        group~,
        auto_pad~,
        activation=@nn_ir.NoAct,
      )
    }
    "MaxPool" => {
      let kernel_shape = match get_attr_ints_opt(attrs, "kernel_shape") {
        Some(v) => v
        None =>
          raise @nn_ir.InvalidAttribute(
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
      @nn_ir.MaxPool(
        kernel_shape~,
        strides~,
        pads~,
        dilations~,
        ceil_mode~,
        auto_pad~,
      )
    }
    "AveragePool" => {
      let kernel_shape = match get_attr_ints_opt(attrs, "kernel_shape") {
        Some(v) => v
        None =>
          raise @nn_ir.InvalidAttribute(
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
      @nn_ir.AveragePool(
        kernel_shape~,
        strides~,
        pads~,
        ceil_mode~,
        count_include_pad~,
        auto_pad~,
      )
    }
    "GlobalAveragePool" => @nn_ir.GlobalAveragePool
    "BatchNormalization" => {
      let epsilon = get_attr_float(attrs, "epsilon", (1.0e-5 : Float))
      @nn_ir.BatchNormalization(epsilon~)
    }
    "LayerNormalization" => {
      let axis = get_attr_int(attrs, "axis", -1)
      let epsilon = get_attr_float(attrs, "epsilon", (1.0e-5 : Float))
      @nn_ir.LayerNormalization(axis~, epsilon~)
    }
    "Concat" => @nn_ir.Concat(axis=get_attr_int(attrs, "axis", 0))
    "Split" =>
      @nn_ir.Split(
        axis=get_attr_int(attrs, "axis", 0),
        split=get_attr_ints_opt(attrs, "split"),
        num_outputs=get_attr_int_opt(attrs, "num_outputs"),
      )
    "Slice" =>
      @nn_ir.Slice(
        starts=get_attr_ints_opt(attrs, "starts"),
        ends=get_attr_ints_opt(attrs, "ends"),
        axes=get_attr_ints_opt(attrs, "axes"),
        steps=get_attr_ints_opt(attrs, "steps"),
      )
    "Pad" =>
      @nn_ir.Pad(
        mode=get_attr_string(attrs, "mode", "constant"),
        pads=get_attr_ints_opt(attrs, "pads"),
        constant_value=get_attr_float_opt(attrs, "value"),
      )
    "Gather" => @nn_ir.Gather(axis=get_attr_int(attrs, "axis", 0))
    "Cast" => {
      let code = match get_attr_int_opt(attrs, "to") {
        Some(c) => c
        None =>
          raise @nn_ir.InvalidAttribute(
            "Cast '\{node_name}' requires 'to' attribute",
          )
      }
      @nn_ir.Cast(to=parse_onnx_dtype(code, node_name))
    }
    "ReduceMean" =>
      @nn_ir.ReduceMean(
        axes=get_attr_ints_opt(attrs, "axes"),
        keepdims=get_attr_int(attrs, "keepdims", 1) != 0,
        noop_with_empty_axes=get_attr_int(attrs, "noop_with_empty_axes", 0) != 0,
      )
    "ReduceSum" =>
      @nn_ir.ReduceSum(
        axes=get_attr_ints_opt(attrs, "axes"),
        keepdims=get_attr_int(attrs, "keepdims", 1) != 0,
        noop_with_empty_axes=get_attr_int(attrs, "noop_with_empty_axes", 0) != 0,
      )
    "Shape" =>
      @nn_ir.Shape(
        start=get_attr_int_opt(attrs, "start"),
        end=get_attr_int_opt(attrs, "end"),
      )
    "QuantizeLinear" =>
      @nn_ir.QuantizeLinear(axis=get_attr_int(attrs, "axis", 1))
    "DequantizeLinear" =>
      @nn_ir.DequantizeLinear(axis=get_attr_int(attrs, "axis", 1))
    "QLinearMatMul" => @nn_ir.QLinearMatMul
    "Resize" =>
      @nn_ir.Resize(
        mode=get_attr_string(attrs, "mode", "nearest"),
        coordinate_transformation_mode=get_attr_string(
          attrs, "coordinate_transformation_mode", "half_pixel",
        ),
        nearest_mode=get_attr_string(
          attrs, "nearest_mode", "round_prefer_floor",
        ),
      )
    "LSTM" =>
      @nn_ir.Lstm(
        hidden_size=get_attr_int_opt(attrs, "hidden_size"),
        direction=get_attr_string(attrs, "direction", "forward"),
        layout=get_attr_int(attrs, "layout", 0),
        clip=get_attr_float_opt(attrs, "clip"),
      )
    "Equal" => @nn_ir.Equal
    "Less" => @nn_ir.Less
    "Greater" => @nn_ir.Greater
    "LessOrEqual" => @nn_ir.LessOrEqual
    "GreaterOrEqual" => @nn_ir.GreaterOrEqual
    "Not" => @nn_ir.Not
    "And" => @nn_ir.And
    "Where" => @nn_ir.Where
    "Erf" => @nn_ir.Erf
    "Range" => @nn_ir.Range
    "Expand" => @nn_ir.Expand
    "Tile" => @nn_ir.Tile
    "ConstantOfShape" => {
      // ONNX 规范：未提供 `value` 属性时默认填充单元素 `Float32(0.0)`
      let fill_val = match find_attr(attrs, "value") {
        Some(a) =>
          match a.t {
            Some(tp) =>
              @onnx_proto.decode_tensor_proto(tp) catch {
                e =>
                  raise @nn_ir.InvalidAttribute(
                    "ConstantOfShape '\{node_name}' failed to decode value: \{e}",
                  )
              }
            None =>
              raise @nn_ir.InvalidAttribute(
                "ConstantOfShape '\{node_name}' attribute 'value' has no tensor",
              )
          }
        None =>
          @tensor.Tensor::from_f32([1], FixedArray::make(1, (0.0 : Float))) catch {
            e =>
              raise @nn_ir.InvalidAttribute(
                "ConstantOfShape '\{node_name}': \{e}",
              )
          }
      }
      @nn_ir.ConstantOfShape(value=fill_val)
    }
    "If" => {
      let then_proto = match find_attr(attrs, "then_branch") {
        Some(a) =>
          match a.g {
            Some(g) => g
            None =>
              raise @nn_ir.InvalidAttribute(
                "If '\{node_name}' attribute 'then_branch' has no GraphProto",
              )
          }
        None =>
          raise @nn_ir.InvalidAttribute(
            "If '\{node_name}' missing required 'then_branch' attribute",
          )
      }
      let else_proto = match find_attr(attrs, "else_branch") {
        Some(a) =>
          match a.g {
            Some(g) => g
            None =>
              raise @nn_ir.InvalidAttribute(
                "If '\{node_name}' attribute 'else_branch' has no GraphProto",
              )
          }
        None =>
          raise @nn_ir.InvalidAttribute(
            "If '\{node_name}' missing required 'else_branch' attribute",
          )
      }
      let then_branch = lower_graph_proto(
        then_proto, opset_version, outer_scope,
      )
      let else_branch = lower_graph_proto(
        else_proto, opset_version, outer_scope,
      )
      @nn_ir.If(then_branch~, else_branch~)
    }
    other => raise @nn_ir.UnsupportedOp(other)
  }

  {
    name: node_name,
    op,
    inputs: node.input.copy(),
    outputs: node.output.copy(),
  }
}

///|
/// 将子图 `@schema.GraphProto`（如 `If` 的分支子图）递归 Lowering 为带词法作用域感知的 `@nn_ir.IrGraph`。
fn lower_graph_proto(
  graph : @schema.GraphProto,
  opset_version : Int64,
  outer_scope : Map[String, Bool],
) -> @nn_ir.IrGraph raise @nn_ir.IrError {
  let initializers : Map[String, @tensor.Tensor] = Map([])
  for i = 0; i < graph.initializer.length(); i = i + 1 {
    let init_proto = graph.initializer[i]
    let init_name = match init_proto.name {
      Some(n) => n
      None =>
        raise @nn_ir.InvalidGraph(
          "subgraph initializer at index \{i} has no name",
        )
    }
    let t = @onnx_proto.decode_tensor_proto(init_proto) catch {
      e =>
        raise @nn_ir.InvalidGraph(
          "subgraph initializer '\{init_name}' decode failed: \{e}",
        )
    }
    initializers[init_name] = t
  }

  let input_names : Array[String] = []
  for i = 0; i < graph.input.length(); i = i + 1 {
    match graph.input[i].name {
      Some(n) if !initializers.contains(n) => input_names.push(n)
      _ => ()
    }
  }
  let output_names : Array[String] = []
  for i = 0; i < graph.output.length(); i = i + 1 {
    match graph.output[i].name {
      Some(n) => output_names.push(n)
      None => ()
    }
  }

  // 构建嵌套子图可见的完整外层作用域（继承上层 + 当前子图输入/权重/节点输出）
  let child_scope : Map[String, Bool] = Map([])
  outer_scope.each(fn(k, v) { if v { child_scope[k] = true } })
  for i = 0; i < input_names.length(); i = i + 1 {
    child_scope[input_names[i]] = true
  }
  initializers.each(fn(k, _) { child_scope[k] = true })
  for i = 0; i < graph.node.length(); i = i + 1 {
    let outs = graph.node[i].output
    for k = 0; k < outs.length(); k = k + 1 {
      if outs[k] != "" {
        child_scope[outs[k]] = true
      }
    }
  }

  let raw_nodes : Array[@nn_ir.IrNode] = []
  for i = 0; i < graph.node.length(); i = i + 1 {
    raw_nodes.push(lower_node(graph.node[i], i, opset_version, child_scope))
  }

  let sorted_nodes = @nn_ir.topo_sort_nodes_with_outer(
    raw_nodes, input_names, initializers, outer_scope,
  )
  let g_name = match graph.name {
    Some(n) => n
    None => "<subgraph>"
  }
  {
    name: g_name,
    inputs: input_names,
    outputs: output_names,
    initializers,
    nodes: sorted_nodes,
  }
}

///|
/// 将解析完成的 ONNX 模型 Lowering 为经过依赖校验与 Kahn 拓扑排序的强类型 `@nn_ir.IrGraph`。
pub fn lower_onnx_model(
  m : @onnx_proto.OnnxModel,
) -> @nn_ir.IrGraph raise @nn_ir.IrError {
  let opset_version = match m.default_opset_version() {
    Some(v) => v
    None => (17 : Int64)
  }

  let input_names = Array::makei(m.inputs.length(), fn(i) { m.inputs[i].name })
  let output_names = Array::makei(m.outputs.length(), fn(i) {
    m.outputs[i].name
  })

  let initializers : Map[String, @tensor.Tensor] = Map([])
  for i = 0; i < m.initializer_order.length(); i = i + 1 {
    let key = m.initializer_order[i]
    match m.initializers.get(key) {
      Some(t) => initializers[key] = t
      None => ()
    }
  }

  // 构建顶层词法作用域，供 `If` 子图合法引用外层输入、权重与上游节点输出
  let top_scope : Map[String, Bool] = Map([])
  for i = 0; i < input_names.length(); i = i + 1 {
    top_scope[input_names[i]] = true
  }
  initializers.each(fn(k, _) { top_scope[k] = true })
  for i = 0; i < m.nodes.length(); i = i + 1 {
    let outs = m.nodes[i].output
    for k = 0; k < outs.length(); k = k + 1 {
      if outs[k] != "" {
        top_scope[outs[k]] = true
      }
    }
  }

  let raw_nodes : Array[@nn_ir.IrNode] = []
  for i = 0; i < m.nodes.length(); i = i + 1 {
    raw_nodes.push(lower_node(m.nodes[i], i, opset_version, top_scope))
  }

  let sorted_nodes = @nn_ir.topo_sort_nodes(
    raw_nodes, input_names, initializers,
  )
  {
    name: m.graph_name,
    inputs: input_names,
    outputs: output_names,
    initializers,
    nodes: sorted_nodes,
  }
}
