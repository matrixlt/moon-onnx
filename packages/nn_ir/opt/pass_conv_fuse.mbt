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
                  if ws.length() == 4 || ws.length() == 3 {
                    let c_out = ws[0]
                    let per_oc = if ws.length() == 4 {
                      ws[1] * ws[2] * ws[3]
                    } else {
                      ws[1] * ws[2]
                    }
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
/// 解析 `Clip` 节点的静态边界 `(min, max)`；若边界来自非常量运行期张量则返回 `None` 表示不可静态融合。
fn resolve_static_clip_bounds(
  node : @nn_ir.IrNode,
  attr_min : Float?,
  attr_max : Float?,
  initializers : Map[String, @tensor.Tensor],
) -> (Float?, Float?)? {
  let lo = if node.inputs.length() >= 2 && node.inputs[1] != "" {
    match extract_static_scalar_f32(initializers, node.inputs[1]) {
      Some(v) => Some(v)
      None => return None
    }
  } else {
    attr_min
  }
  let hi = if node.inputs.length() >= 3 && node.inputs[2] != "" {
    match extract_static_scalar_f32(initializers, node.inputs[2]) {
      Some(v) => Some(v)
      None => return None
    }
  } else {
    attr_max
  }
  Some((lo, hi))
}

///|
/// Pass 5：Conv + 尾部激活（Relu / Clip / ReLU6）融合。
/// 当 `Conv(activation=NoAct)` 的输出仅被下游单个 `Relu`、`SiLU` 或静态边界 `Clip` 消费一次时（支持并行分支非紧邻拓扑），
/// 将激活内联至卷积微内核写回阶段，直接消除整张特征图的额外读写与中间缓冲。
fn pass_fold_conv_activation(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let uses = count_tensor_uses(nodes, graph_outputs)
  // 索引所有可融合激活节点（输入张量名 -> 节点下标）
  let act_consumer : Map[String, Int32] = Map([])
  for i = 0; i < nodes.length(); i = i + 1 {
    let nd = nodes[i]
    if nd.outputs.length() == 1 &&
      nd.outputs[0] != "" &&
      nd.inputs.length() >= 1 &&
      nd.inputs[0] != "" {
      match nd.op {
        @nn_ir.Relu | @nn_ir.Silu | @nn_ir.Clip(..) =>
          act_consumer[nd.inputs[0]] = i
        _ => ()
      }
    }
  }

  let skip = Array::make(nodes.length(), false)
  let kept : Array[@nn_ir.IrNode] = []
  let mut folded = 0

  for i = 0; i < nodes.length(); i = i + 1 {
    if skip[i] {
      continue
    }
    let cur = nodes[i]
    match cur.op {
      @nn_ir.Conv(
        kernel_shape~,
        strides~,
        pads~,
        dilations~,
        group~,
        auto_pad~,
        activation=@nn_ir.NoAct,
      ) if cur.outputs.length() == 1 && uses.get(cur.outputs[0]) == Some(1) =>
        match act_consumer.get(cur.outputs[0]) {
          Some(ci) if ci > i && !skip[ci] => {
            let next = nodes[ci]
            let fused_act : (@nn_ir.FusedActivation, String)? = match next.op {
              @nn_ir.Relu => Some((@nn_ir.ReluAct, "_relu"))
              @nn_ir.Silu => Some((@nn_ir.SiluAct, "_silu"))
              @nn_ir.Clip(min~, max~) =>
                match resolve_static_clip_bounds(next, min, max, initializers) {
                  Some((Some(lo), Some(hi))) =>
                    Some((@nn_ir.ClipAct(min=lo, max=hi), "_clip"))
                  Some((Some(lo), None)) if lo == (0.0 : Float) =>
                    Some((@nn_ir.ReluAct, "_relu"))
                  _ => None
                }
              _ => None
            }
            match fused_act {
              Some((act, suffix)) => {
                skip[ci] = true
                kept.push({
                  name: "\{cur.name}\{suffix}",
                  op: @nn_ir.Conv(
                    kernel_shape~,
                    strides~,
                    pads~,
                    dilations~,
                    group~,
                    auto_pad~,
                    activation=act,
                  ),
                  inputs: cur.inputs.copy(),
                  outputs: next.outputs.copy(),
                })
                folded = folded + 1
                continue
              }
              None => ()
            }
          }
          _ => ()
        }
      _ => ()
    }
    kept.push(cur)
  }
  (kept, folded)
}
