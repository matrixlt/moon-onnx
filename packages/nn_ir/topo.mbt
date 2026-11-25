///|
/// 对一组 IrNode 执行依赖校验与 Kahn 拓扑排序（支持子图词法捕获外层作用域变量）。
/// 1. 校验每个非空输入名称均由图输入、权重初始化张量、外层作用域或某个节点产出。
/// 2. 自动追踪 `If` 节点内子图引用的外层张量，确保外层生产者节点严格排在 `If` 之前。
/// 3. 在存在多个就绪节点时优先保持原始声明顺序，保证执行顺序稳定可复现。
/// 4. 若图中存在环路，抛出 CyclicGraph 错误。
pub fn topo_sort_nodes_with_outer(
  nodes : Array[IrNode],
  graph_inputs : Array[String],
  initializers : Map[String, @tensor.Tensor],
  outer_scope : Map[String, Bool],
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

  // 记录已存在的外部输入集合（图输入 + 外层词法作用域）
  let known_external : Map[String, Bool] = Map([])
  for i = 0; i < graph_inputs.length(); i = i + 1 {
    known_external[graph_inputs[i]] = true
  }
  outer_scope.each(fn(k, v) { if v { known_external[k] = true } })

  // 构建邻接表与入度数组
  let in_degree = Array::make(n, 0)
  let adj : Array[Array[Int32]] = Array::makei(n, fn(_) { [] })

  fn add_dep(consumer_idx : Int32, in_name : String) -> Unit raise IrError {
    if in_name == "" {
      return
    }
    match producer.get(in_name) {
      Some(p_idx) => {
        adj[p_idx].push(consumer_idx)
        in_degree[consumer_idx] = in_degree[consumer_idx] + 1
      }
      None =>
        if !known_external.contains(in_name) && !initializers.contains(in_name) {
          raise MissingInput(
            "node '\{nodes[consumer_idx].name}' references undefined tensor '\{in_name}'",
          )
        }
    }
  }

  for i = 0; i < n; i = i + 1 {
    let ins = nodes[i].inputs
    for j = 0; j < ins.length(); j = j + 1 {
      add_dep(i, ins[j])
    }
    // 追踪控制流子图对当前层节点输出的隐式依赖
    match nodes[i].op {
      If(then_branch~, else_branch~) => {
        let t_refs = then_branch.external_refs()
        for k = 0; k < t_refs.length(); k = k + 1 {
          add_dep(i, t_refs[k])
        }
        let e_refs = else_branch.external_refs()
        for k = 0; k < e_refs.length(); k = k + 1 {
          add_dep(i, e_refs[k])
        }
      }
      _ => ()
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
/// 对顶层计算图的一组 IrNode 执行依赖校验与 Kahn 拓扑排序。
pub fn topo_sort_nodes(
  nodes : Array[IrNode],
  graph_inputs : Array[String],
  initializers : Map[String, @tensor.Tensor],
) -> Array[IrNode] raise IrError {
  topo_sort_nodes_with_outer(nodes, graph_inputs, initializers, Map([]))
}
