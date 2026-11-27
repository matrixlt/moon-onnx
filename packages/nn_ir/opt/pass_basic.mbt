///|
/// Pass 1：Constant 节点提升（将 Constant 节点的静态张量提升到 initializers，消除运行期节点）。
fn pass_lift_constants(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let is_output : Map[String, Bool] = Map([])
  for i = 0; i < graph_outputs.length(); i = i + 1 {
    is_output[graph_outputs[i]] = true
  }
  let kept : Array[@nn_ir.IrNode] = []
  let mut count = 0
  for i = 0; i < nodes.length(); i = i + 1 {
    let node = nodes[i]
    match node.op {
      @nn_ir.Constant(value~) =>
        if node.outputs.length() == 1 &&
          node.outputs[0] != "" &&
          !is_output.contains(node.outputs[0]) {
          initializers[node.outputs[0]] = value
          count = count + 1
        } else {
          kept.push(node)
        }
      _ => kept.push(node)
    }
  }
  (kept, count)
}

///|
/// Pass 2：Identity 节点消除（重写下游引用指向源张量，保留作为图直接输出的情况）。
fn pass_eliminate_identity(
  nodes : Array[@nn_ir.IrNode],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let is_output : Map[String, Bool] = Map([])
  for i = 0; i < graph_outputs.length(); i = i + 1 {
    is_output[graph_outputs[i]] = true
  }
  let alias_map : Map[String, String] = Map([])
  let kept : Array[@nn_ir.IrNode] = []
  let mut removed = 0

  fn resolve(name : String) -> String {
    let mut cur = name
    while alias_map.contains(cur) {
      match alias_map.get(cur) {
        Some(next) => cur = next
        None => break
      }
    }
    cur
  }

  for i = 0; i < nodes.length(); i = i + 1 {
    let node = nodes[i]
    let rewritten_inputs = Array::makei(node.inputs.length(), fn(j) {
      resolve(node.inputs[j])
    })
    match node.op {
      @nn_ir.Identity =>
        if rewritten_inputs.length() == 1 &&
          node.outputs.length() == 1 &&
          !is_output.contains(node.outputs[0]) {
          alias_map[node.outputs[0]] = rewritten_inputs[0]
          removed = removed + 1
        } else {
          kept.push({
            name: node.name,
            op: node.op,
            inputs: rewritten_inputs,
            outputs: node.outputs.copy(),
          })
        }
      _ =>
        kept.push({
          name: node.name,
          op: node.op,
          inputs: rewritten_inputs,
          outputs: node.outputs.copy(),
        })
    }
  }
  (kept, removed)
}

///|
/// Pass 5：死代码消除（DCE）——从图输出反向标记可达节点，剔除对最终输出无贡献的冗余节点。
fn pass_dce(
  nodes : Array[@nn_ir.IrNode],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let needed : Map[String, Bool] = Map([])
  for i = 0; i < graph_outputs.length(); i = i + 1 {
    needed[graph_outputs[i]] = true
  }
  let live = Array::make(nodes.length(), false)
  // 节点已按拓扑序排列，从后向前单遍扫描即可精确确定全部存活节点
  for i = nodes.length() - 1; i >= 0; i = i - 1 {
    let node = nodes[i]
    let mut is_live = false
    for j = 0; j < node.outputs.length(); j = j + 1 {
      if needed.contains(node.outputs[j]) {
        is_live = true
        break
      }
    }
    if is_live {
      live[i] = true
      for j = 0; j < node.inputs.length(); j = j + 1 {
        if node.inputs[j] != "" {
          needed[node.inputs[j]] = true
        }
      }
      match node.op {
        @nn_ir.If(then_branch~, else_branch~) => {
          let t_refs = then_branch.external_refs()
          for k = 0; k < t_refs.length(); k = k + 1 {
            needed[t_refs[k]] = true
          }
          let e_refs = else_branch.external_refs()
          for k = 0; k < e_refs.length(); k = k + 1 {
            needed[e_refs[k]] = true
          }
        }
        _ => ()
      }
    }
  }
  let kept : Array[@nn_ir.IrNode] = []
  let mut removed = 0
  for i = 0; i < nodes.length(); i = i + 1 {
    if live[i] {
      kept.push(nodes[i])
    } else {
      removed = removed + 1
    }
  }
  (kept, removed)
}

///|
/// Pass 5：静态 Gemm(trans_b=true) 权重预转置。
/// PyTorch 导出的全连接层权重通常以 [N, K] + transB=1 形式存储；
/// 在加载期一次性预转置为连续 [K, N] 布局并置 trans_b=false，可消除每次推理的 O(K*N) 转置复制开销。
fn pass_pretranspose_gemm_weights(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
) -> (Array[@nn_ir.IrNode], Int32) {
  let kept : Array[@nn_ir.IrNode] = []
  let mut count = 0
  for i = 0; i < nodes.length(); i = i + 1 {
    let node = nodes[i]
    match node.op {
      @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) if trans_b &&
        node.inputs.length() >= 2 =>
        match initializers.get(node.inputs[1]) {
          Some(w_t) if w_t.rank() == 2 =>
            try {
              let wt = w_t.transpose([1, 0]).contiguous()
              let new_w_name = "\{node.inputs[1]}__pretransposed_\{i}"
              initializers[new_w_name] = wt
              let new_inputs = node.inputs.copy()
              new_inputs[1] = new_w_name
              kept.push({
                name: node.name,
                op: @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b=false),
                inputs: new_inputs,
                outputs: node.outputs.copy(),
              })
              count = count + 1
              continue
            } catch {
              _ => ()
            }
          _ => ()
        }
      _ => ()
    }
    kept.push(node)
  }
  (kept, count)
}

///|
/// 从静态初始化器中提取单个 Float32 标量值（用于解析 Opset 11+ `Clip` 的 `min`/`max` 静态输入）。
fn extract_static_scalar_f32(
  initializers : Map[String, @tensor.Tensor],
  name : String,
) -> Float? {
  if name == "" {
    return None
  }
  match initializers.get(name) {
    Some(t) if t.numel() == 1 && t.dtype() == @tensor.Float32 =>
      Some(t.get_f32_flat(0)) catch {
        _ => None
      }
    _ => None
  }
}
