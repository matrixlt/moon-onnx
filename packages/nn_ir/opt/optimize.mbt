///|
/// 图优化执行统计摘要，便于 CLI 日志与 DEVLOG 记录每项 Pass 的收益。
#derive(Debug, Eq)
pub(all) struct OptStats {
  lifted_constants : Int32
  eliminated_identities : Int32
  folded_conv_bn : Int32
  folded_conv_bias : Int32
  folded_conv_activation : Int32
  folded_layernorm : Int32
  folded_silu : Int32
  folded_gelu : Int32
  pretransposed_gemm_weights : Int32
  eliminated_dead_nodes : Int32
}

///|
/// 将图优化统计结果格式化为单行人类可读摘要，便于 CLI 日志与文档示例输出。
pub fn OptStats::format_summary(self : OptStats) -> String {
  "OptStats(const=\{self.lifted_constants}, id=\{self.eliminated_identities}, conv_bn=\{self.folded_conv_bn}, conv_bias=\{self.folded_conv_bias}, conv_act=\{self.folded_conv_activation}, ln=\{self.folded_layernorm}, silu=\{self.folded_silu}, gelu=\{self.folded_gelu}, pretrans_gemm=\{self.pretransposed_gemm_weights}, dce=\{self.eliminated_dead_nodes})"
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
/// 统计图中每个张量名称被节点输入、控制流子图捕获及图输出引用的总次数（用于安全判断是否可融合或消除）。
fn count_tensor_uses(
  nodes : Array[@nn_ir.IrNode],
  graph_outputs : Array[String],
) -> Map[String, Int32] {
  let counts : Map[String, Int32] = Map([])
  fn bump(name : String) -> Unit {
    if name != "" {
      let prev = match counts.get(name) {
        Some(c) => c
        None => 0
      }
      counts[name] = prev + 1
    }
  }

  for i = 0; i < nodes.length(); i = i + 1 {
    let node = nodes[i]
    for j = 0; j < node.inputs.length(); j = j + 1 {
      bump(node.inputs[j])
    }
    match node.op {
      @nn_ir.If(then_branch~, else_branch~) => {
        let t_refs = then_branch.external_refs()
        for k = 0; k < t_refs.length(); k = k + 1 {
          bump(t_refs[k])
        }
        let e_refs = else_branch.external_refs()
        for k = 0; k < e_refs.length(); k = k + 1 {
          bump(e_refs[k])
        }
      }
      _ => ()
    }
  }
  for i = 0; i < graph_outputs.length(); i = i + 1 {
    bump(graph_outputs[i])
  }
  counts
}

///|
/// 递归优化 `If` 控制流节点内部的 `then_branch` 与 `else_branch` 子图，累加统计信息。
fn pass_optimize_subgraphs(
  nodes : Array[@nn_ir.IrNode],
) -> (Array[@nn_ir.IrNode], OptStats) {
  let out : Array[@nn_ir.IrNode] = []
  let mut lifted = 0
  let mut id_elim = 0
  let mut conv_bn = 0
  let mut conv_bias = 0
  let mut conv_act = 0
  let mut ln_fold = 0
  let mut silu_fold = 0
  let mut gelu_fold = 0
  let mut gemm_pt = 0
  let mut dead = 0

  for i = 0; i < nodes.length(); i = i + 1 {
    let node = nodes[i]
    match node.op {
      @nn_ir.If(then_branch~, else_branch~) => {
        let (opt_then, s1) = optimize_graph(then_branch)
        let (opt_else, s2) = optimize_graph(else_branch)
        lifted = lifted + s1.lifted_constants + s2.lifted_constants
        id_elim = id_elim + s1.eliminated_identities + s2.eliminated_identities
        conv_bn = conv_bn + s1.folded_conv_bn + s2.folded_conv_bn
        conv_bias = conv_bias + s1.folded_conv_bias + s2.folded_conv_bias
        conv_act = conv_act +
          s1.folded_conv_activation +
          s2.folded_conv_activation
        ln_fold = ln_fold + s1.folded_layernorm + s2.folded_layernorm
        silu_fold = silu_fold + s1.folded_silu + s2.folded_silu
        gelu_fold = gelu_fold + s1.folded_gelu + s2.folded_gelu
        gemm_pt = gemm_pt +
          s1.pretransposed_gemm_weights +
          s2.pretransposed_gemm_weights
        dead = dead + s1.eliminated_dead_nodes + s2.eliminated_dead_nodes
        out.push({
          name: node.name,
          op: @nn_ir.If(then_branch=opt_then, else_branch=opt_else),
          inputs: node.inputs.copy(),
          outputs: node.outputs.copy(),
        })
      }
      _ => out.push(node)
    }
  }
  (
    out,
    {
      lifted_constants: lifted,
      eliminated_identities: id_elim,
      folded_conv_bn: conv_bn,
      folded_conv_bias: conv_bias,
      folded_conv_activation: conv_act,
      folded_layernorm: ln_fold,
      folded_silu: silu_fold,
      folded_gelu: gelu_fold,
      pretransposed_gemm_weights: gemm_pt,
      eliminated_dead_nodes: dead,
    },
  )
}

///|
/// 对 IrGraph 执行完整图优化管线，返回优化后的新 IrGraph 与各 Pass 统计信息。
pub fn optimize_graph(graph : @nn_ir.IrGraph) -> (@nn_ir.IrGraph, OptStats) {
  let initializers = clone_initializers(graph.initializers)
  let (n0, sub_stats) = pass_optimize_subgraphs(graph.nodes)
  let (n1, lifted_constants) = pass_lift_constants(
    n0,
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
  let (n4b, folded_silu) = pass_fold_silu(n4, graph.outputs)
  let (n5, folded_conv_activation) = pass_fold_conv_activation(
    n4b,
    initializers,
    graph.outputs,
  )
  let (n6, folded_layernorm) = pass_fold_layernorm(
    n5,
    initializers,
    graph.outputs,
  )
  let (n6b, folded_gelu) = pass_fold_gelu(n6, initializers, graph.outputs)
  let (n7, pretransposed_gemm_weights) = pass_pretranspose_gemm_weights(
    n6b, initializers,
  )
  let (n8, eliminated_dead_nodes) = pass_dce(n7, graph.outputs)

  let optimized : @nn_ir.IrGraph = {
    name: graph.name,
    inputs: graph.inputs.copy(),
    outputs: graph.outputs.copy(),
    initializers,
    nodes: n8,
  }
  let stats : OptStats = {
    lifted_constants: lifted_constants + sub_stats.lifted_constants,
    eliminated_identities: eliminated_identities +
    sub_stats.eliminated_identities,
    folded_conv_bn: folded_conv_bn + sub_stats.folded_conv_bn,
    folded_conv_bias: folded_conv_bias + sub_stats.folded_conv_bias,
    folded_conv_activation: folded_conv_activation +
    sub_stats.folded_conv_activation,
    folded_layernorm: folded_layernorm + sub_stats.folded_layernorm,
    folded_silu: folded_silu + sub_stats.folded_silu,
    folded_gelu: folded_gelu + sub_stats.folded_gelu,
    pretransposed_gemm_weights: pretransposed_gemm_weights +
    sub_stats.pretransposed_gemm_weights,
    eliminated_dead_nodes: eliminated_dead_nodes +
    sub_stats.eliminated_dead_nodes,
  }
  (optimized, stats)
}
