///|
/// 判断算子是否为零拷贝视图算子（其输出可能共享第 0 个输入的底层缓冲区，供动态 Session 与 AOT 复用）。
pub fn is_view_op(op : @nn_ir.Op) -> Bool {
  match op {
    @nn_ir.Identity
    | @nn_ir.Flatten(..)
    | @nn_ir.Reshape(..)
    | @nn_ir.Transpose(..)
    | @nn_ir.Squeeze(..)
    | @nn_ir.Unsqueeze(..) => true
    _ => false
  }
}

///|
/// 别名感知的静态生命周期分析结果（供动态 `Session` 与 `onnx_aot` Tensor 模式共享）。
pub(all) struct LivenessPlan {
  /// 每个拓扑步执行完毕后可安全回收至 `BufferPool` 的物理根张量名称列表
  release_plan : Array[Array[String]]
  /// 每个节点的输出是否被锁定（不可从 `BufferPool` 分配），长度等于节点数
  pinned_outputs : Array[Bool]
  /// 所有被永久锁定的张量名称（图输入、权重初始化器、常量、图输出及子图外层捕获张量）
  pinned : Map[String, Bool]
  /// 张量名到其物理底层根缓冲张量名的映射（穿透零拷贝视图链）
  root_of : Map[String, String]
}

///|
/// 对已拓扑排序的计算图执行别名感知的静态生命周期分析：
/// - 追踪零拷贝视图算子对源张量缓冲区的别名引用，确保在所有别名最后一次消费完毕后才回收缓冲；
/// - 永久锁定图输入、权重初始化器、常量节点输出、子图捕获外部引用与图最终输出（及其源缓冲），防止跨推理数据污染。
pub fn build_release_plan(graph : @nn_ir.IrGraph) -> LivenessPlan {
  let num_nodes = graph.nodes.length()
  let pinned : Map[String, Bool] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    pinned[graph.inputs[i]] = true
  }
  graph.initializers.each(fn(k, _) { pinned[k] = true })
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    pinned[graph.outputs[i]] = true
  }
  // 子图捕获的外层作用域张量必须锁定，禁止在子图执行期间被归还至局部复用池
  let ext_refs = graph.external_refs()
  for i = 0; i < ext_refs.length(); i = i + 1 {
    pinned[ext_refs[i]] = true
  }

  let root_of : Map[String, String] = Map([])
  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    for k = 0; k < node.outputs.length(); k = k + 1 {
      let out_name = node.outputs[k]
      if out_name == "" {
        continue
      }
      match node.op {
        @nn_ir.Constant(..) => {
          root_of[out_name] = out_name
          pinned[out_name] = true
        }
        op if is_view_op(op) => {
          let in_name = if node.inputs.length() > 0 {
            node.inputs[0]
          } else {
            ""
          }
          let root = match root_of.get(in_name) {
            Some(prev) => prev
            None => in_name
          }
          root_of[out_name] = root
          if pinned.contains(out_name) && root != "" {
            pinned[root] = true
          }
        }
        _ => root_of[out_name] = out_name
      }
    }
  }

  let last_use : Map[String, Int32] = Map([])
  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    if !is_view_op(node.op) {
      match node.op {
        @nn_ir.Constant(..) => ()
        _ =>
          for k = 0; k < node.outputs.length(); k = k + 1 {
            let out_name = node.outputs[k]
            if out_name != "" && !pinned.contains(out_name) {
              last_use[out_name] = i
            }
          }
      }
    }
  }

  fn mark_use(name : String, step : Int32) -> Unit {
    if name == "" {
      return
    }
    let root = match root_of.get(name) {
      Some(prev) => prev
      None => name
    }
    if last_use.contains(root) {
      last_use[root] = step
    }
  }

  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    for j = 0; j < node.inputs.length(); j = j + 1 {
      mark_use(node.inputs[j], i)
    }
    // 延长被 `If` 分支子图词法捕获的外层中间张量生命周期至该 `If` 节点
    match node.op {
      @nn_ir.If(then_branch~, else_branch~) => {
        let t_refs = then_branch.external_refs()
        for k = 0; k < t_refs.length(); k = k + 1 {
          mark_use(t_refs[k], i)
        }
        let e_refs = else_branch.external_refs()
        for k = 0; k < e_refs.length(); k = k + 1 {
          mark_use(e_refs[k], i)
        }
      }
      _ => ()
    }
  }

  let release_plan : Array[Array[String]] = Array::makei(num_nodes, fn(_) { [] })
  last_use.each(fn(root_name, step) {
    if !pinned.contains(root_name) {
      release_plan[step].push(root_name)
    }
  })

  let pinned_outputs = Array::make(num_nodes, false)
  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    if node.outputs.length() == 0 {
      pinned_outputs[i] = true
    } else {
      for k = 0; k < node.outputs.length(); k = k + 1 {
        let out_name = node.outputs[k]
        if out_name == "" {
          pinned_outputs[i] = true
          break
        }
        let root = match root_of.get(out_name) {
          Some(prev) => prev
          None => out_name
        }
        if pinned.contains(out_name) || pinned.contains(root) {
          pinned_outputs[i] = true
          break
        }
      }
    }
  }

  { release_plan, pinned_outputs, pinned, root_of, }
}
