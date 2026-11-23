///|
/// 图优化执行统计摘要，便于 CLI 日志与 DEVLOG 记录每项 Pass 的收益。
#derive(Debug, Eq)
pub(all) struct OptStats {
  lifted_constants : Int32
  eliminated_identities : Int32
  folded_conv_bn : Int32
  folded_conv_bias : Int32
  pretransposed_gemm_weights : Int32
  eliminated_dead_nodes : Int32
}

///|
/// 复制 Initializer 字典，确保图优化为纯变换，不修改输入 IrGraph。
fn clone_initializers(
  src : Map[String, @tensor.Tensor],
) -> Map[String, @tensor.Tensor] {
  let dst : Map[String, @tensor.Tensor] = Map([])
  src.each(fn(k, v) { dst[k] = v })
  dst
}

///|
/// 统计图中每个张量名称被节点输入与图输出引用的总次数（用于安全判断是否可融合或消除）。
fn count_tensor_uses(
  nodes : Array[@nn_ir.IrNode],
  graph_outputs : Array[String],
) -> Map[String, Int32] {
  let counts : Map[String, Int32] = Map([])
  for i = 0; i < nodes.length(); i = i + 1 {
    let ins = nodes[i].inputs
    for j = 0; j < ins.length(); j = j + 1 {
      let name = ins[j]
      if name != "" {
        let prev = match counts.get(name) {
          Some(c) => c
          None => 0
        }
        counts[name] = prev + 1
      }
    }
  }
  for i = 0; i < graph_outputs.length(); i = i + 1 {
    let name = graph_outputs[i]
    let prev = match counts.get(name) {
      Some(c) => c
      None => 0
    }
    counts[name] = prev + 1
  }
  counts
}

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
/// Pass 3：Conv + BatchNormalization 静态参数融合。
/// 当 Conv 的输出仅被紧邻的推理态 BatchNormalization 消费，且 Conv 权重与 BN 参数均为静态 Initializer 时，
/// 可在 Double 精度下将 BN 缩放与偏移直接吸收到卷积核与偏置中，彻底消除运行期 BN 开销。
fn pass_fold_conv_bn(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let uses = count_tensor_uses(nodes, graph_outputs)
  let kept : Array[@nn_ir.IrNode] = []
  let mut folded = 0
  let mut i = 0

  while i < nodes.length() {
    let cur = nodes[i]
    if i + 1 < nodes.length() {
      let next = nodes[i + 1]
      match (cur.op, next.op) {
        (@nn_ir.Conv(..), @nn_ir.BatchNormalization(epsilon~)) =>
          if cur.outputs.length() == 1 &&
            next.inputs.length() >= 5 &&
            next.inputs[0] == cur.outputs[0] &&
            uses.get(cur.outputs[0]) == Some(1) {
            let w_name = cur.inputs[1]
            let b_opt_name = if cur.inputs.length() >= 3 && cur.inputs[2] != "" {
              Some(cur.inputs[2])
            } else {
              None
            }
            match
              (
                initializers.get(w_name),
                initializers.get(next.inputs[1]),
                initializers.get(next.inputs[2]),
                initializers.get(next.inputs[3]),
                initializers.get(next.inputs[4]),
              ) {
              (
                Some(w_t),
                Some(scale_t),
                Some(bn_bias_t),
                Some(mean_t),
                Some(var_t),
              ) =>
                try {
                  let ws = w_t.shape()
                  if ws.length() == 4 {
                    let c_out = ws[0]
                    let per_oc = ws[1] * ws[2] * ws[3]
                    let w_buf = w_t.to_f32_buffer()
                    let s_buf = scale_t.to_f32_buffer()
                    let bb_buf = bn_bias_t.to_f32_buffer()
                    let m_buf = mean_t.to_f32_buffer()
                    let v_buf = var_t.to_f32_buffer()
                    let old_b_buf : FixedArray[Float]? = match b_opt_name {
                      Some(bn) =>
                        match initializers.get(bn) {
                          Some(bt) => Some(bt.to_f32_buffer())
                          None => None
                        }
                      None => None
                    }
                    let new_w = FixedArray::make(w_buf.length(), (0.0 : Float))
                    let new_b = FixedArray::make(c_out, (0.0 : Float))
                    let eps_d = epsilon.to_double()

                    for oc = 0; oc < c_out; oc = oc + 1 {
                      let factor = s_buf[oc].to_double() /
                        (v_buf[oc].to_double() + eps_d).sqrt()
                      let base = oc * per_oc
                      for j = 0; j < per_oc; j = j + 1 {
                        new_w[base + j] = Float::from_double(
                          w_buf[base + j].to_double() * factor,
                        )
                      }
                      let b0 = match old_b_buf {
                        Some(ob) => ob[oc].to_double()
                        None => 0.0
                      }
                      new_b[oc] = Float::from_double(
                        (b0 - m_buf[oc].to_double()) * factor +
                        bb_buf[oc].to_double(),
                      )
                    }

                    let fused_w_name = "\{w_name}__bn_fused_w_\{i}"
                    let fused_b_name = "\{w_name}__bn_fused_b_\{i}"
                    initializers[fused_w_name] = @tensor.Tensor::from_f32(
                      ws, new_w,
                    )
                    initializers[fused_b_name] = @tensor.Tensor::from_f32(
                      [c_out],
                      new_b,
                    )

                    kept.push({
                      name: "\{cur.name}_fused_bn",
                      op: cur.op,
                      inputs: [cur.inputs[0], fused_w_name, fused_b_name],
                      outputs: next.outputs.copy(),
                    })
                    folded = folded + 1
                    i = i + 2
                    continue
                  }
                } catch {
                  _ => ()
                }
              _ => ()
            }
          }
        _ => ()
      }
    }
    kept.push(cur)
    i = i + 1
  }
  (kept, folded)
}

///|
/// Pass 4：Conv + 通道偏置 Add 融合。
/// ONNX Model Zoo MNIST 等模型常将偏置拆为 `Conv(x, W)` + `Add(c, bias[C_out, 1, 1])`；
/// 合并为单个带偏置的 `Conv(x, W, bias_1d[C_out])` 可消除整张特征图的额外读写开销。
fn pass_fold_conv_channel_add(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let uses = count_tensor_uses(nodes, graph_outputs)
  let kept : Array[@nn_ir.IrNode] = []
  let mut folded = 0
  let mut i = 0

  while i < nodes.length() {
    let cur = nodes[i]
    if i + 1 < nodes.length() {
      let next = nodes[i + 1]
      match (cur.op, next.op) {
        (@nn_ir.Conv(..), @nn_ir.Add) => {
          let conv_has_no_bias = cur.inputs.length() == 2 ||
            (cur.inputs.length() >= 3 && cur.inputs[2] == "")
          if conv_has_no_bias &&
            cur.outputs.length() == 1 &&
            next.inputs.length() == 2 &&
            uses.get(cur.outputs[0]) == Some(1) {
            let conv_out = cur.outputs[0]
            let bias_name_opt : String? = if next.inputs[0] == conv_out {
              Some(next.inputs[1])
            } else if next.inputs[1] == conv_out {
              Some(next.inputs[0])
            } else {
              None
            }
            match (bias_name_opt, initializers.get(cur.inputs[1])) {
              (Some(b_name), Some(w_t)) =>
                match initializers.get(b_name) {
                  Some(b_t) => {
                    let ws = w_t.shape()
                    let bs = b_t.shape()
                    let c_out = if ws.length() == 4 { ws[0] } else { -1 }
                    // 仅当偏置形状明确为通道级 [C_out, 1, 1] 或 [1, C_out, 1, 1] 时才安全折叠
                    let is_channel_bias = (
                        bs.length() == 3 &&
                        bs[0] == c_out &&
                        bs[1] == 1 &&
                        bs[2] == 1
                      ) ||
                      (
                        bs.length() == 4 &&
                        bs[0] == 1 &&
                        bs[1] == c_out &&
                        bs[2] == 1 &&
                        bs[3] == 1
                      )
                    if is_channel_bias {
                      try {
                        let b1d = b_t.reshape([c_out])
                        let fused_b_name = "\{b_name}__conv_bias_1d_\{i}"
                        initializers[fused_b_name] = b1d
                        kept.push({
                          name: "\{cur.name}_fused_bias",
                          op: cur.op,
                          inputs: [cur.inputs[0], cur.inputs[1], fused_b_name],
                          outputs: next.outputs.copy(),
                        })
                        folded = folded + 1
                        i = i + 2
                        continue
                      } catch {
                        _ => ()
                      }
                    }
                  }
                  None => ()
                }
              _ => ()
            }
          }
        }
        _ => ()
      }
    }
    kept.push(cur)
    i = i + 1
  }
  (kept, folded)
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
/// 对 IrGraph 执行完整图优化管线，返回优化后的新 IrGraph 与各 Pass 统计信息。
pub fn optimize_graph(graph : @nn_ir.IrGraph) -> (@nn_ir.IrGraph, OptStats) {
  let initializers = clone_initializers(graph.initializers)
  let (n1, lifted_constants) = pass_lift_constants(
    graph.nodes,
    initializers,
    graph.outputs,
  )
  let (n2, eliminated_identities) = pass_eliminate_identity(n1, graph.outputs)
  let (n3, folded_conv_bn) = pass_fold_conv_bn(n2, initializers, graph.outputs)
  let (n4, folded_conv_bias) = pass_fold_conv_channel_add(
    n3,
    initializers,
    graph.outputs,
  )
  let (n5, pretransposed_gemm_weights) = pass_pretranspose_gemm_weights(
    n4, initializers,
  )
  let (n6, eliminated_dead_nodes) = pass_dce(n5, graph.outputs)

  let optimized : @nn_ir.IrGraph = {
    name: graph.name,
    inputs: graph.inputs.copy(),
    outputs: graph.outputs.copy(),
    initializers,
    nodes: n6,
  }
  let stats : OptStats = {
    lifted_constants,
    eliminated_identities,
    folded_conv_bn,
    folded_conv_bias,
    pretransposed_gemm_weights,
    eliminated_dead_nodes,
  }
  (optimized, stats)
}
