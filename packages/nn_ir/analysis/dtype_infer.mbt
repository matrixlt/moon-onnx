///|
/// 推导单个 `IrNode` 各输出张量的静态数据类型并写入 `dtypes`（全项目唯一类型推断真相源）。
pub fn infer_node_output_dtypes(
  node : @nn_ir.IrNode,
  dtypes : Map[String, @tensor.DType],
) -> Unit {
  if node.outputs.length() == 0 {
    return
  }
  match node.op {
    @nn_ir.DynamicQuantizeLinear => {
      if node.outputs.length() >= 1 && node.outputs[0] != "" {
        dtypes[node.outputs[0]] = @tensor.UInt8
      }
      if node.outputs.length() >= 2 && node.outputs[1] != "" {
        dtypes[node.outputs[1]] = @tensor.Float32
      }
      if node.outputs.length() >= 3 && node.outputs[2] != "" {
        dtypes[node.outputs[2]] = @tensor.UInt8
      }
      return
    }
    _ => ()
  }
  let out_dt : @tensor.DType = match node.op {
    @nn_ir.Cast(to~) => to
    @nn_ir.Shape(..) | @nn_ir.NonMaxSuppression(..) => @tensor.Int64
    @nn_ir.MatMulInteger => @tensor.Int32
    @nn_ir.MatMulNBits(..)
    | @nn_ir.Silu
    | @nn_ir.RmsNormalization(..)
    | @nn_ir.SkipRmsNormalization(..)
    | @nn_ir.RotaryEmbedding(..)
    | @nn_ir.GroupQueryAttention(..) => @tensor.Float32
    @nn_ir.Equal
    | @nn_ir.Less
    | @nn_ir.Greater
    | @nn_ir.LessOrEqual
    | @nn_ir.GreaterOrEqual
    | @nn_ir.Not
    | @nn_ir.And => @tensor.Bool
    @nn_ir.Where =>
      if node.inputs.length() >= 2 {
        match dtypes.get(node.inputs[1]) {
          Some(dt) => dt
          None => @tensor.Float32
        }
      } else {
        @tensor.Float32
      }
    @nn_ir.Constant(value~) => value.dtype()
    @nn_ir.ConstantOfShape(value~) => value.dtype()
    @nn_ir.DequantizeLinear(..) => @tensor.Float32
    @nn_ir.QuantizeLinear(..) =>
      if node.inputs.length() >= 3 && node.inputs[2] != "" {
        match dtypes.get(node.inputs[2]) {
          Some(dt) => dt
          None => @tensor.UInt8
        }
      } else {
        @tensor.UInt8
      }
    @nn_ir.QLinearMatMul =>
      if node.inputs.length() >= 8 && node.inputs[7] != "" {
        match dtypes.get(node.inputs[7]) {
          Some(dt) => dt
          None => @tensor.Int8
        }
      } else {
        @tensor.Int8
      }
    _ =>
      if node.inputs.length() > 0 {
        match dtypes.get(node.inputs[0]) {
          Some(dt) => dt
          None => @tensor.Float32
        }
      } else {
        @tensor.Float32
      }
  }
  for k = 0; k < node.outputs.length(); k = k + 1 {
    if node.outputs[k] != "" {
      dtypes[node.outputs[k]] = out_dt
    }
  }
}

///|
/// 推导图中每个张量的静态数据类型（全项目唯一入口，供图分析、Arena 规划与 AOT 编译器共享）。
pub fn infer_static_dtypes(
  graph : @nn_ir.IrGraph,
  input_dtypes : Map[String, @tensor.DType],
) -> Map[String, @tensor.DType] {
  let dtypes : Map[String, @tensor.DType] = Map([])
  graph.initializers.each(fn(k, t) { dtypes[k] = t.dtype() })
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    dtypes[name] = match input_dtypes.get(name) {
      Some(dt) => dt
      None => @tensor.Float32
    }
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    infer_node_output_dtypes(graph.nodes[i], dtypes)
  }
  dtypes
}
