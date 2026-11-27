///|
/// Pass 6：Transformer 分解态 `LayerNormalization` 9 节点模式融合。
/// PyTorch Opset 11~16 导出的 BERT / MiniLM / RoBERTa / ViT 模型会将 `LayerNorm` 展开为：
/// `ReduceMean(ax=-1) -> Sub -> Pow(2) -> ReduceMean(ax=-1) -> Add(eps) -> Sqrt -> Div -> Mul(gamma) -> [Add(beta)]`
/// 将该子图融合为单个双精度累加的 `@nn_ir.LayerNormalization` 节点，可大幅削减节点数与 8 份中间激活读写。
fn pass_fold_layernorm(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let uses = count_tensor_uses(nodes, graph_outputs)
  let kept : Array[@nn_ir.IrNode] = []
  let mut folded = 0
  let mut i = 0
  let n = nodes.length()

  while i < n {
    if i + 7 < n {
      let n_rm1 = nodes[i]
      let n_sub = nodes[i + 1]
      let n_pow = nodes[i + 2]
      let n_rm2 = nodes[i + 3]
      let n_add_eps = nodes[i + 4]
      let n_sqrt = nodes[i + 5]
      let n_div = nodes[i + 6]
      let n_mul = nodes[i + 7]

      match
        (
          n_rm1.op,
          n_sub.op,
          n_pow.op,
          n_rm2.op,
          n_add_eps.op,
          n_sqrt.op,
          n_div.op,
          n_mul.op,
        ) {
        (
          @nn_ir.ReduceMean(axes=Some(ax1), keepdims=true, ..),
          @nn_ir.Sub,
          @nn_ir.Pow,
          @nn_ir.ReduceMean(axes=Some(ax2), keepdims=true, ..),
          @nn_ir.Add,
          @nn_ir.Sqrt,
          @nn_ir.Div,
          @nn_ir.Mul,
        ) if ax1.length() == 1 && ax2.length() == 1 && ax1[0] == ax2[0] => {
          let x_name = n_rm1.inputs[0]
          let mean_name = n_rm1.outputs[0]
          let diff_name = n_sub.outputs[0]
          let sq_name = n_pow.outputs[0]
          let var_name = n_rm2.outputs[0]
          let veps_name = n_add_eps.outputs[0]
          let std_name = n_sqrt.outputs[0]
          let norm_name = n_div.outputs[0]
          let scaled_name = n_mul.outputs[0]

          // 校验拓扑连通性与中间张量使用计数，确保无外部旁路消费
          let chain_ok = n_sub.inputs.length() == 2 &&
            n_sub.inputs[0] == x_name &&
            n_sub.inputs[1] == mean_name &&
            uses.get(mean_name) == Some(1) &&
            n_pow.inputs.length() == 2 &&
            n_pow.inputs[0] == diff_name &&
            extract_static_scalar_f32(initializers, n_pow.inputs[1]) ==
            Some((2.0 : Float)) &&
            uses.get(diff_name) == Some(2) &&
            n_rm2.inputs.length() >= 1 &&
            n_rm2.inputs[0] == sq_name &&
            uses.get(sq_name) == Some(1) &&
            uses.get(var_name) == Some(1) &&
            n_sqrt.inputs.length() == 1 &&
            n_sqrt.inputs[0] == veps_name &&
            uses.get(veps_name) == Some(1) &&
            n_div.inputs.length() == 2 &&
            n_div.inputs[0] == diff_name &&
            n_div.inputs[1] == std_name &&
            uses.get(std_name) == Some(1) &&
            uses.get(norm_name) == Some(1)

          // 解析 Add(var, eps) 或 Add(eps, var) 中的静态 epsilon 标量
          let eps_opt : Float? = if chain_ok && n_add_eps.inputs.length() == 2 {
            if n_add_eps.inputs[0] == var_name {
              extract_static_scalar_f32(initializers, n_add_eps.inputs[1])
            } else if n_add_eps.inputs[1] == var_name {
              extract_static_scalar_f32(initializers, n_add_eps.inputs[0])
            } else {
              None
            }
          } else {
            None
          }

          // 解析 Mul(norm, gamma) 或 Mul(gamma, norm) 中的 1D 静态缩放权重
          let gamma_opt : String? = if chain_ok && n_mul.inputs.length() == 2 {
            let cand = if n_mul.inputs[0] == norm_name {
              n_mul.inputs[1]
            } else if n_mul.inputs[1] == norm_name {
              n_mul.inputs[0]
            } else {
              ""
            }
            match initializers.get(cand) {
              Some(gt) if gt.rank() == 1 && gt.dtype() == @tensor.Float32 =>
                Some(cand)
              _ => None
            }
          } else {
            None
          }

          match (eps_opt, gamma_opt) {
            (Some(eps), Some(gamma_name)) if eps >= (0.0 : Float) => {
              // 检查紧邻的第 9 个节点是否为可选 bias Add(scaled, beta)
              let mut fused_with_bias = false
              if i + 8 < n && uses.get(scaled_name) == Some(1) {
                let n_add_b = nodes[i + 8]
                match n_add_b.op {
                  @nn_ir.Add if n_add_b.inputs.length() == 2 => {
                    let cand_b = if n_add_b.inputs[0] == scaled_name {
                      n_add_b.inputs[1]
                    } else if n_add_b.inputs[1] == scaled_name {
                      n_add_b.inputs[0]
                    } else {
                      ""
                    }
                    match initializers.get(cand_b) {
                      Some(bt) if bt.rank() == 1 &&
                        bt.dtype() == @tensor.Float32 => {
                        kept.push({
                          name: "\{n_rm1.name}_fused_ln",
                          op: @nn_ir.LayerNormalization(
                            axis=ax1[0],
                            epsilon=eps,
                          ),
                          inputs: [x_name, gamma_name, cand_b],
                          outputs: n_add_b.outputs.copy(),
                        })
                        folded = folded + 1
                        i = i + 9
                        fused_with_bias = true
                      }
                      _ => ()
                    }
                  }
                  _ => ()
                }
              }
              if fused_with_bias {
                continue
              }
              kept.push({
                name: "\{n_rm1.name}_fused_ln",
                op: @nn_ir.LayerNormalization(axis=ax1[0], epsilon=eps),
                inputs: [x_name, gamma_name],
                outputs: n_mul.outputs.copy(),
              })
              folded = folded + 1
              i = i + 8
              continue
            }
            _ => ()
          }
        }
        _ => ()
      }
    }
    kept.push(nodes[i])
    i = i + 1
  }
  (kept, folded)
}

///|
/// SiLU 算子融合 Pass：将 `Mul(x, Sigmoid(x))` 或 `Mul(Sigmoid(x), x)`（当 `Sigmoid` 输出仅被该 `Mul` 消费时）
/// 自动折叠为单节点 `Silu(x)`，消除 50% 激活节点与中间激活内存往返（覆盖 Llama / Qwen / SmolLM2 / YOLOv8）。
fn pass_fold_silu(
  nodes : Array[@nn_ir.IrNode],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let use_counts = count_tensor_uses(nodes, graph_outputs)
  let sig_info : Map[String, (String, Int32)] = Map([])
  for i = 0; i < nodes.length(); i = i + 1 {
    let nd = nodes[i]
    match nd.op {
      @nn_ir.Sigmoid if nd.inputs.length() == 1 &&
        nd.outputs.length() == 1 &&
        nd.outputs[0] != "" => sig_info[nd.outputs[0]] = (nd.inputs[0], i)
      _ => ()
    }
  }
  let skip = Array::make(nodes.length(), false)
  let replaced : Map[Int32, @nn_ir.IrNode] = Map([])
  let mut folded = 0
  for i = 0; i < nodes.length(); i = i + 1 {
    let nd = nodes[i]
    match nd.op {
      @nn_ir.Mul if nd.inputs.length() == 2 && nd.outputs.length() == 1 => {
        let a = nd.inputs[0]
        let b = nd.inputs[1]
        let matched : (String, Int32)? = match sig_info.get(b) {
          Some((x, si)) if x == a =>
            match use_counts.get(b) {
              Some(1) => Some((x, si))
              _ => None
            }
          _ =>
            match sig_info.get(a) {
              Some((x, si)) if x == b =>
                match use_counts.get(a) {
                  Some(1) => Some((x, si))
                  _ => None
                }
              _ => None
            }
        }
        match matched {
          Some((x, si)) => {
            skip[si] = true
            replaced[i] = {
              name: nd.name,
              op: @nn_ir.Silu,
              inputs: [x],
              outputs: nd.outputs.copy(),
            }
            folded = folded + 1
          }
          None => ()
        }
      }
      _ => ()
    }
  }
  let out : Array[@nn_ir.IrNode] = []
  for i = 0; i < nodes.length(); i = i + 1 {
    if skip[i] {
      continue
    }
    match replaced.get(i) {
      Some(r) => out.push(r)
      None => out.push(nodes[i])
    }
  }
  (out, folded)
}

///|
/// 读取单元素标量 Float32 初始化常量值并提升为 Double，供精确 GELU 常量匹配复用。
fn read_scalar_f32_init(
  initializers : Map[String, @tensor.Tensor],
  name : String,
) -> Double? {
  match extract_static_scalar_f32(initializers, name) {
    Some(v) => Some(v.to_double())
    None => None
  }
}

///|
/// 精确 GELU 模式融合 Pass：将 PyTorch / HuggingFace 导出的 5 节点精确 Erf-GELU 子图：
/// - 模式 A（Whisper / Optimum）：`Div(x, sqrt(2)) -> Erf -> Add(1.0) -> Mul(x, _) -> Mul(_, 0.5)`
/// - 模式 B（BERT / MiniLM / ViT）：`Div(x, sqrt(2)) -> Erf -> Add(1.0)` 与 `Mul(x, 0.5)` 相乘
/// 折叠为单节点 `@nn_ir.GeluExact(x)`，消除 4 个中间激活张量及 4 趟整图访存。
fn pass_fold_gelu(
  nodes : Array[@nn_ir.IrNode],
  initializers : Map[String, @tensor.Tensor],
  graph_outputs : Array[String],
) -> (Array[@nn_ir.IrNode], Int32) {
  let uses = count_tensor_uses(nodes, graph_outputs)
  let producer : Map[String, Int32] = Map([])
  for i = 0; i < nodes.length(); i = i + 1 {
    let nd = nodes[i]
    if nd.outputs.length() == 1 && nd.outputs[0] != "" {
      producer[nd.outputs[0]] = i
    }
  }

  fn is_approx(v : Double, target : Double) -> Bool {
    let d = v - target
    (if d < 0.0 { -d } else { d }) < 1.0e-4
  }

  // 检查 `add_name` 是否为 `Add(Erf(Div(x, sqrt2)), 1.0)`，若是则返回 `(x, div_idx, erf_idx, add_idx)`
  fn match_one_plus_erf(add_name : String) -> (String, Int32, Int32, Int32)? {
    if uses.get(add_name) != Some(1) {
      return None
    }
    let add_idx = match producer.get(add_name) {
      Some(idx) => idx
      None => return None
    }
    let add_nd = nodes[add_idx]
    match add_nd.op {
      @nn_ir.Add if add_nd.inputs.length() == 2 => {
        let a0 = add_nd.inputs[0]
        let a1 = add_nd.inputs[1]
        let erf_out : String? = match read_scalar_f32_init(initializers, a1) {
          Some(v) if is_approx(v, 1.0) => Some(a0)
          _ =>
            match read_scalar_f32_init(initializers, a0) {
              Some(v) if is_approx(v, 1.0) => Some(a1)
              _ => None
            }
        }
        let erf_name = match erf_out {
          Some(s) if uses.get(s) == Some(1) => s
          _ => return None
        }
        let erf_idx = match producer.get(erf_name) {
          Some(idx) => idx
          None => return None
        }
        let erf_nd = nodes[erf_idx]
        match erf_nd.op {
          @nn_ir.Erf if erf_nd.inputs.length() == 1 => {
            let div_name = erf_nd.inputs[0]
            if uses.get(div_name) != Some(1) {
              return None
            }
            let div_idx = match producer.get(div_name) {
              Some(idx) => idx
              None => return None
            }
            let div_nd = nodes[div_idx]
            match div_nd.op {
              @nn_ir.Div if div_nd.inputs.length() == 2 =>
                match read_scalar_f32_init(initializers, div_nd.inputs[1]) {
                  Some(v) if is_approx(v, 1.4142135623730951) =>
                    Some((div_nd.inputs[0], div_idx, erf_idx, add_idx))
                  _ => None
                }
              _ => None
            }
          }
          _ => None
        }
      }
      _ => None
    }
  }

  let skip = Array::make(nodes.length(), false)
  let replaced : Map[Int32, @nn_ir.IrNode] = Map([])
  let mut folded = 0

  for i = 0; i < nodes.length(); i = i + 1 {
    if skip[i] {
      continue
    }
    let nd = nodes[i]
    match nd.op {
      @nn_ir.Mul if nd.inputs.length() == 2 && nd.outputs.length() == 1 => {
        let m0 = nd.inputs[0]
        let m1 = nd.inputs[1]
        // 模式 A：`Mul(Mul(x, 1 + erf(x/sqrt2)), 0.5)`
        let half_other : String? = match
          read_scalar_f32_init(initializers, m1) {
          Some(v) if is_approx(v, 0.5) => Some(m0)
          _ =>
            match read_scalar_f32_init(initializers, m0) {
              Some(v) if is_approx(v, 0.5) => Some(m1)
              _ => None
            }
        }
        let mut matched_a = false
        match half_other {
          Some(prev_name) if uses.get(prev_name) == Some(1) =>
            match producer.get(prev_name) {
              Some(p_idx) if !skip[p_idx] => {
                let p_nd = nodes[p_idx]
                match p_nd.op {
                  @nn_ir.Mul if p_nd.inputs.length() == 2 => {
                    let pa = p_nd.inputs[0]
                    let pb = p_nd.inputs[1]
                    let hit : (String, Int32, Int32, Int32)? = match
                      match_one_plus_erf(pb) {
                      Some((x, di, ei, ai)) if x == pa => Some((x, di, ei, ai))
                      _ =>
                        match match_one_plus_erf(pa) {
                          Some((x, di, ei, ai)) if x == pb =>
                            Some((x, di, ei, ai))
                          _ => None
                        }
                    }
                    match hit {
                      Some((x, di, ei, ai)) => {
                        skip[di] = true
                        skip[ei] = true
                        skip[ai] = true
                        skip[p_idx] = true
                        replaced[i] = {
                          name: nd.name,
                          op: @nn_ir.GeluExact,
                          inputs: [x],
                          outputs: nd.outputs.copy(),
                        }
                        folded = folded + 1
                        matched_a = true
                      }
                      None => ()
                    }
                  }
                  _ => ()
                }
              }
              _ => ()
            }
          _ => ()
        }
        if matched_a {
          continue
        }

        // 模式 B：`Mul(Mul(x, 0.5), 1 + erf(x/sqrt2))`
        let hit_b : (String, String, Int32, Int32, Int32)? = match
          match_one_plus_erf(m1) {
          Some((x, di, ei, ai)) => Some((m0, x, di, ei, ai))
          None =>
            match match_one_plus_erf(m0) {
              Some((x, di, ei, ai)) => Some((m1, x, di, ei, ai))
              None => None
            }
        }
        match hit_b {
          Some((xh_name, x, di, ei, ai)) if uses.get(xh_name) == Some(1) =>
            match producer.get(xh_name) {
              Some(xh_idx) if !skip[xh_idx] => {
                let xh_nd = nodes[xh_idx]
                match xh_nd.op {
                  @nn_ir.Mul if xh_nd.inputs.length() == 2 => {
                    let xa = xh_nd.inputs[0]
                    let xb = xh_nd.inputs[1]
                    let half_b = match read_scalar_f32_init(initializers, xb) {
                      Some(v) => is_approx(v, 0.5)
                      None => false
                    }
                    let half_a = match read_scalar_f32_init(initializers, xa) {
                      Some(v) => is_approx(v, 0.5)
                      None => false
                    }
                    let ok = (xa == x && half_b) || (xb == x && half_a)
                    if ok {
                      skip[di] = true
                      skip[ei] = true
                      skip[ai] = true
                      skip[xh_idx] = true
                      replaced[i] = {
                        name: nd.name,
                        op: @nn_ir.GeluExact,
                        inputs: [x],
                        outputs: nd.outputs.copy(),
                      }
                      folded = folded + 1
                    }
                  }
                  _ => ()
                }
              }
              _ => ()
            }
          _ => ()
        }
      }
      _ => ()
    }
  }

  let out : Array[@nn_ir.IrNode] = []
  for i = 0; i < nodes.length(); i = i + 1 {
    if skip[i] {
      continue
    }
    match replaced.get(i) {
      Some(r) => out.push(r)
      None => out.push(nodes[i])
    }
  }
  (out, folded)
}
