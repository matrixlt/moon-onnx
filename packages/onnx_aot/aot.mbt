///|
/// 对已拓扑排序的 `IrGraph` 执行编译期静态形状推导，返回所有张量名称对应的确定形状表（复用 `matrixlt/nn_ir/analysis` 单一真相源）。
pub fn infer_static_shapes(
  graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
) -> Map[String, Array[Int32]] raise CodegenError {
  @analysis.infer_static_shapes(graph, input_shapes) catch {
    @nn_ir.InvalidGraph(msg) => raise UnsupportedGraph(msg)
    @nn_ir.UnsupportedOp(msg) => raise UnsupportedOp(msg)
    e => raise ShapeError("\{e}")
  }
}

///|
/// 将 `IrGraph` 编译为独立、零 `Map`、零 Protobuf 的纯 MoonBit 静态推理代码与紧凑权重二进制包。
/// 默认启用 `Arena` 模式（编译期静态内存区间复用 + 零堆分配 + 零 `Tensor` 依赖），遇到特殊算子时自动回退至 `Tensor` 模式。
pub fn compile_graph_to_mbt(
  raw_graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
  config~ : CodegenConfig = CodegenConfig::default(),
) -> AotArtifact raise CodegenError {
  let base_graph = if config.optimize {
    let (opt_g, _) = @opt.optimize_graph(raw_graph)
    opt_g
  } else {
    raw_graph
  }
  // 执行编译期特化（冻结输入、`If` 分支内联、静态 `Shape`/元数据常量折叠、冗余 `Cast` 消除）
  let graph = specialize_graph(
    base_graph,
    input_shapes,
    config.frozen_inputs,
    input_dtypes=config.input_dtypes,
  )
  if graph.inputs.length() == 0 || graph.outputs.length() == 0 {
    raise UnsupportedGraph("graph must have at least 1 input and 1 output")
  }

  let shapes = infer_static_shapes(graph, input_shapes)
  let dtypes = infer_static_dtypes(graph, config.input_dtypes)

  if config.mode == Arena && can_emit_arena(graph, shapes, dtypes) {
    let (f32_plan, byte_plan) = plan_static_arenas(graph, shapes, dtypes)
    return emit_arena_model(graph, shapes, dtypes, f32_plan, byte_plan, config)
  }

  emit_tensor_model(graph, shapes, config)
}

///|
/// 将已解析的 `OnnxModel` 直接静态编译为独立的 MoonBit 推理源码与权重包（`onnx2mbt` 一站式入口）。
pub fn compile_model_to_mbt(
  m : @onnx_proto.OnnxModel,
  input_shapes : Map[String, Array[Int32]],
  config~ : CodegenConfig = CodegenConfig::default(),
) -> AotArtifact raise CodegenError {
  let graph = @frontend.lower_onnx_model(m) catch {
    e => raise UnsupportedGraph("lowering failed: \{e}")
  }
  let merged_dtypes : Map[String, @tensor.DType] = Map([])
  for i = 0; i < m.inputs.length(); i = i + 1 {
    match m.inputs[i].dtype {
      Some(dt) => merged_dtypes[m.inputs[i].name] = dt
      None => ()
    }
  }
  config.input_dtypes.each(fn(k, v) { merged_dtypes[k] = v })
  let eff_cfg : CodegenConfig = { ..config, input_dtypes: merged_dtypes, }
  compile_graph_to_mbt(graph, input_shapes, config=eff_cfg)
}
