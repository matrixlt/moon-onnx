///|
/// 尝试在编译期对全常量输入的纯元算子执行常量折叠求值；若不可折叠返回 `None`。
fn try_eval_const_node(
  node : @nn_ir.IrNode,
  consts : Map[String, @tensor.Tensor],
) -> @tensor.Tensor? {
  let ins = node.inputs
  if node.outputs.length() != 1 || node.outputs[0] == "" {
    return None
  }
  // 检查所有非空输入是否均已是编译期常量
  for j = 0; j < ins.length(); j = j + 1 {
    if ins[j] != "" && !consts.contains(ins[j]) {
      return None
    }
  }
  try {
    match node.op {
      @nn_ir.Constant(value~) => Some(value)
      @nn_ir.Identity => consts.get(ins[0])
      @nn_ir.Equal =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.equal_op(a, b))
          _ => None
        }
      @nn_ir.ConstantOfShape(value~) =>
        match consts.get(ins[0]) {
          Some(sh) => Some(@ops.constant_of_shape_op(sh, value))
          None => None
        }
      @nn_ir.Cast(to~) =>
        match consts.get(ins[0]) {
          Some(x) => Some(@ops.cast_op(x, to))
          None => None
        }
      @nn_ir.Gather(axis~) =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(d), Some(idx)) => Some(@ops.gather_op(d, idx, axis))
          _ => None
        }
      @nn_ir.Unsqueeze(axes~) =>
        match consts.get(ins[0]) {
          Some(x) => {
            let ax_t = if ins.length() >= 2 && ins[1] != "" {
              consts.get(ins[1])
            } else {
              None
            }
            Some(@ops.unsqueeze_op(x, axes, ax_t))
          }
          None => None
        }
      @nn_ir.Squeeze(axes~) =>
        match consts.get(ins[0]) {
          Some(x) => {
            let ax_t = if ins.length() >= 2 && ins[1] != "" {
              consts.get(ins[1])
            } else {
              None
            }
            Some(@ops.squeeze_op(x, axes, ax_t))
          }
          None => None
        }
      @nn_ir.Concat(axis~) => {
        let ts : Array[@tensor.Tensor] = []
        for k = 0; k < ins.length(); k = k + 1 {
          match consts.get(ins[k]) {
            Some(t) => ts.push(t)
            None => return None
          }
        }
        Some(@ops.concat_op(ts, axis))
      }
      @nn_ir.Reshape(allowzero~) =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(x), Some(sh)) => Some(@ops.reshape_op(x, sh, allowzero~))
          _ => None
        }
      @nn_ir.Transpose(perm~) =>
        match consts.get(ins[0]) {
          Some(x) => Some(@ops.transpose_op(x, perm))
          None => None
        }
      @nn_ir.Slice(starts~, ends~, axes~, steps~) =>
        match consts.get(ins[0]) {
          Some(x) => {
            let s_arr : Array[Int64] = if ins.length() >= 2 && ins[1] != "" {
              match consts.get(ins[1]) {
                Some(t) => {
                  let b = t.to_i64_buffer()
                  Array::makei(b.length(), fn(k) { b[k] })
                }
                None => return None
              }
            } else {
              match starts {
                Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
                None => return None
              }
            }
            let e_arr : Array[Int64] = if ins.length() >= 3 && ins[2] != "" {
              match consts.get(ins[2]) {
                Some(t) => {
                  let b = t.to_i64_buffer()
                  Array::makei(b.length(), fn(k) { b[k] })
                }
                None => return None
              }
            } else {
              match ends {
                Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
                None => return None
              }
            }
            let a_opt : Array[Int32]? = if ins.length() >= 4 && ins[3] != "" {
              match consts.get(ins[3]) {
                Some(t) => {
                  let b = t.to_i64_buffer()
                  Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
                }
                None => return None
              }
            } else {
              axes
            }
            let st_opt : Array[Int32]? = if ins.length() >= 5 && ins[4] != "" {
              match consts.get(ins[4]) {
                Some(t) => {
                  let b = t.to_i64_buffer()
                  Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
                }
                None => return None
              }
            } else {
              steps
            }
            Some(@ops.slice_op(x, s_arr, e_arr, a_opt, st_opt))
          }
          None => None
        }
      @nn_ir.Add =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.add_f32(a, b))
          _ => None
        }
      @nn_ir.Sub =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.sub_f32(a, b))
          _ => None
        }
      @nn_ir.Mul =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.mul_f32(a, b))
          _ => None
        }
      @nn_ir.Div =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.div_f32(a, b))
          _ => None
        }
      @nn_ir.Less =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.less_op(a, b))
          _ => None
        }
      @nn_ir.Greater =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.greater_op(a, b))
          _ => None
        }
      @nn_ir.LessOrEqual =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.less_or_equal_op(a, b))
          _ => None
        }
      @nn_ir.GreaterOrEqual =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.greater_or_equal_op(a, b))
          _ => None
        }
      @nn_ir.Not =>
        match consts.get(ins[0]) {
          Some(x) => Some(@ops.not_op(x))
          None => None
        }
      @nn_ir.And =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.and_op(a, b))
          _ => None
        }
      @nn_ir.Where =>
        match (consts.get(ins[0]), consts.get(ins[1]), consts.get(ins[2])) {
          (Some(c), Some(x), Some(y)) => Some(@ops.where_op(c, x, y))
          _ => None
        }
      @nn_ir.Range =>
        match (consts.get(ins[0]), consts.get(ins[1]), consts.get(ins[2])) {
          (Some(s), Some(l), Some(d)) => Some(@ops.range_op(s, l, d))
          _ => None
        }
      @nn_ir.Expand =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(x), Some(sh)) => Some(@ops.expand_op(x, sh))
          _ => None
        }
      @nn_ir.Tile =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(x), Some(reps)) => Some(@ops.tile_op(x, reps))
          _ => None
        }
      @nn_ir.DequantizeLinear(axis~) =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(x), Some(sc)) => {
            let zp = if ins.length() >= 3 && ins[2] != "" {
              consts.get(ins[2])
            } else {
              None
            }
            Some(@ops.dequantize_linear_op(x, sc, zp, axis~))
          }
          _ => None
        }
      _ => None
    }
  } catch {
    _ => None
  }
}

///|
/// 从 `Bool` 标量常量张量中读取布尔值（用于 `If` 编译期分支特化）。
fn try_read_const_bool(t : @tensor.Tensor) -> Bool? {
  let c = t.contiguous() catch { _ => return None }
  if c.numel() != 1 {
    return None
  }
  match c.data {
    @tensor.Bool(buf) => Some(buf[c.offset])
    _ => None
  }
}

///|
/// 编译期图特化与静态形状/元数据常量折叠管线（AOT 专用）：
/// 1. 冻结 `frozen_inputs`（如 Silero VAD 的 `sr = 16000`），将其转为静态常量；
/// 2. 对条件为编译期常量的 `If` 节点执行分支内联，彻底消除控制流与未使用分支权重；
/// 3. 在已知 `input_shapes` 基础上折叠 `Shape` 及全常量元子图（如 MobileNetV2 的 `Shape->Gather->Unsqueeze->Concat` 与 Silero VAD 的 `ConstantOfShape->...->Cast` 填充参数计算）；
/// 4. 将无操作同类型 `Cast`（如 `f32 -> f32`）重写为 `Identity` 以便后续优化消除。
pub fn specialize_graph(
  graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
  frozen_inputs : Map[String, @tensor.Tensor],
  input_dtypes~ : Map[String, @tensor.DType] = Map([]),
) -> @nn_ir.IrGraph raise CodegenError {
  // 1. 克隆初始权重并吸收冻结输入
  let initializers : Map[String, @tensor.Tensor] = Map([])
  graph.initializers.each(fn(k, v) { initializers[k] = v })
  frozen_inputs.each(fn(k, v) { initializers[k] = v })

  let remaining_inputs : Array[String] = []
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    if !frozen_inputs.contains(name) {
      remaining_inputs.push(name)
    }
  }

  let is_graph_out : Map[String, Bool] = Map([])
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    is_graph_out[graph.outputs[i]] = true
  }

  // 2. 迭代执行静态 `If` 分支内联 + `Equal`/常量折叠（支持嵌套 `If`）
  let mut cur_nodes = graph.nodes.copy()
  for _iter = 0; _iter < 8; _iter = _iter + 1 {
    let mut changed = false
    let next_nodes : Array[@nn_ir.IrNode] = []
    for i = 0; i < cur_nodes.length(); i = i + 1 {
      let node = cur_nodes[i]
      match node.op {
        @nn_ir.If(then_branch~, else_branch~) =>
          if node.inputs.length() >= 1 {
            match initializers.get(node.inputs[0]) {
              Some(cond_t) =>
                match try_read_const_bool(cond_t) {
                  Some(cond) => {
                    let chosen = if cond { then_branch } else { else_branch }
                    chosen.initializers.each(fn(k, v) { initializers[k] = v })
                    for k = 0; k < chosen.nodes.length(); k = k + 1 {
                      next_nodes.push(chosen.nodes[k])
                    }
                    // 用 Identity 桥接子图输出与外层 If 输出名，交由后续 optimize_graph 零成本消除
                    for k = 0
                        k < node.outputs.length() && k < chosen.outputs.length()
                        k = k + 1 {
                      if node.outputs[k] != "" &&
                        chosen.outputs[k] != node.outputs[k] {
                        next_nodes.push({
                          name: "\{node.name}_inline_out_\{k}",
                          op: @nn_ir.Identity,
                          inputs: [chosen.outputs[k]],
                          outputs: [node.outputs[k]],
                        })
                      }
                    }
                    changed = true
                    continue
                  }
                  None => ()
                }
              None => ()
            }
          }
        _ =>
          // 尝试折叠仅依赖静态常量的前置条件节点（例如 `Constant` 与 `Equal`）
          match try_eval_const_node(node, initializers) {
            Some(val) if !is_graph_out.contains(node.outputs[0]) => {
              initializers[node.outputs[0]] = val
              changed = true
              continue
            }
            _ => ()
          }
      }
      next_nodes.push(node)
    }
    cur_nodes = next_nodes
    if !changed {
      break
    }
  }

  // 3. 在已知静态输入形状的前提下，单遍前向传播形状 + 折叠 `Shape` / 常量元子图 + 消除冗余 `Cast`
  let shapes : Map[String, Array[Int32]] = Map([])
  let dtypes : Map[String, @tensor.DType] = Map([])
  initializers.each(fn(k, t) {
    shapes[k] = t.shape()
    dtypes[k] = t.dtype()
  })
  for i = 0; i < remaining_inputs.length(); i = i + 1 {
    let name = remaining_inputs[i]
    match input_shapes.get(name) {
      Some(s) => shapes[name] = s.copy()
      None => ()
    }
    dtypes[name] = match input_dtypes.get(name) {
      Some(dt) => dt
      None => @tensor.Float32
    }
  }

  let folded_nodes : Array[@nn_ir.IrNode] = []
  for i = 0; i < cur_nodes.length(); i = i + 1 {
    let node = cur_nodes[i]
    // 3a. 静态 `Shape` 算子：直接由编译期已知形状求值为 `Int64` 常量张量
    match node.op {
      @nn_ir.Shape(start~, end~) =>
        if node.inputs.length() >= 1 && node.outputs.length() == 1 {
          match shapes.get(node.inputs[0]) {
            Some(dims) => {
              let rank = dims.length()
              let s = match start {
                Some(v) => {
                  let norm = if v < 0 { v + rank } else { v }
                  if norm < 0 {
                    0
                  } else if norm > rank {
                    rank
                  } else {
                    norm
                  }
                }
                None => 0
              }
              let e = match end {
                Some(v) => {
                  let norm = if v < 0 { v + rank } else { v }
                  if norm < 0 {
                    0
                  } else if norm > rank {
                    rank
                  } else {
                    norm
                  }
                }
                None => rank
              }
              let len = if e > s { e - s } else { 0 }
              let buf = FixedArray::makei(len, fn(k) { dims[s + k].to_int64() })
              let val = @tensor.Tensor::from_i64([len], buf) catch {
                err => raise TensorErr("\{err}")
              }
              let out_name = node.outputs[0]
              shapes[out_name] = [len]
              dtypes[out_name] = @tensor.Int64
              if !is_graph_out.contains(out_name) {
                initializers[out_name] = val
                continue
              } else {
                folded_nodes.push({
                  name: node.name,
                  op: @nn_ir.Constant(value=val),
                  inputs: [],
                  outputs: node.outputs.copy(),
                })
                continue
              }
            }
            None => ()
          }
        }
      _ => ()
    }

    // 3b. 全常量输入纯元算子求值折叠
    match try_eval_const_node(node, initializers) {
      Some(val) => {
        let out_name = node.outputs[0]
        shapes[out_name] = val.shape()
        dtypes[out_name] = val.dtype()
        if !is_graph_out.contains(out_name) {
          initializers[out_name] = val
          continue
        }
      }
      None => ()
    }

    // 3c. 同类型无操作 `Cast` 重写为 `Identity`，供后续优化直接消除
    let eff_node : @nn_ir.IrNode = match node.op {
      @nn_ir.Cast(to~) if node.inputs.length() >= 1 =>
        match dtypes.get(node.inputs[0]) {
          Some(src_dt) if src_dt == to =>
            {
              name: node.name,
              op: @nn_ir.Identity,
              inputs: node.inputs.copy(),
              outputs: node.outputs.copy(),
            }
          _ => node
        }
      _ => node
    }

    // 记录保留节点的静态输出形状与数据类型，供下游 `Shape`/`Cast` 继续推导
    // 记录保留节点的静态输出形状与数据类型（复用 `matrixlt/nn_ir/analysis` 单一真相源）
    let out_shapes = @analysis.infer_node_output_shapes(
      eff_node, shapes, initializers,
    ) catch {
      _ => []
    }
    for k = 0
        k < out_shapes.length() && k < eff_node.outputs.length()
        k = k + 1 {
      if eff_node.outputs[k] != "" {
        shapes[eff_node.outputs[k]] = out_shapes[k]
      }
    }
    @analysis.infer_node_output_dtypes(eff_node, dtypes)
    folded_nodes.push(eff_node)
  }

  let specialized : @nn_ir.IrGraph = {
    name: graph.name,
    inputs: remaining_inputs,
    outputs: graph.outputs.copy(),
    initializers,
    nodes: folded_nodes,
  }
  let (opt_g, _) = @opt.optimize_graph(specialized)
  opt_g
}
