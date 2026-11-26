///|
/// 推理会话构建与执行阶段可能抛出的结构化错误。
#derive(Debug, Eq)
pub(all) suberror RuntimeError {
  ModelErr(String)
  IrErr(String)
  TensorErr(String)
  MissingTensor(String)
  UnsupportedOp(String)
}

///|
/// 为 RuntimeError 提供人类可读的错误描述。
pub impl Show for RuntimeError with fn output(self, logger) {
  match self {
    ModelErr(msg) => logger.write_string("ModelErr(\{msg})")
    IrErr(msg) => logger.write_string("IrErr(\{msg})")
    TensorErr(msg) => logger.write_string("TensorErr(\{msg})")
    MissingTensor(msg) => logger.write_string("MissingTensor(\{msg})")
    UnsupportedOp(msg) => logger.write_string("UnsupportedOp(\{msg})")
  }
}

///|
/// 推理计算后端选择：
/// - `Scalar`：100% 纯 MoonBit 标量参考实现
/// - `Simd`  ：基于 `matrixlt/ml_kernels` 的 MoonBit V128 SIMD 向量化内核 + 静态权重预打包缓存
#derive(Debug, Eq)
pub(all) enum BackendKind {
  Scalar
  Simd
}

///|
/// 为 BackendKind 提供可读名称。
pub impl Show for BackendKind with fn output(self, logger) {
  match self {
    Scalar => logger.write_string("scalar")
    Simd => logger.write_string("simd-v128")
  }
}

///|
/// 单个节点的逐算子耗时剖析记录。
pub(all) struct OpTiming {
  node_name : String
  op_type : String
  output_shape : Array[Int32]
  elapsed_us : Double
}

///|
/// 单次推理的完整剖析报告，包含总耗时与每个节点的耗时分布。
pub(all) struct RunProfile {
  backend : BackendKind
  total_us : Double
  ops : Array[OpTiming]
}

///|
/// 将剖析报告格式化为人类易读的对齐表格（供 CLI `profile` 与性能诊断使用）。
pub fn RunProfile::format_table(self : RunProfile) -> String {
  let buf = StringBuilder()
  buf.write_string("=== Inference Profile (backend=\{self.backend}) ===\n")
  let denom = if self.total_us > 0.0 { self.total_us } else { 1.0 }
  for i = 0; i < self.ops.length(); i = i + 1 {
    let op = self.ops[i]
    let pct = op.elapsed_us / denom * 100.0
    // 保留 1 位小数展示占比，避免冗长浮点尾数影响阅读
    let pct_10 = (pct * 10.0).to_int()
    let pct_whole = pct_10 / 10
    let pct_frac = pct_10 % 10
    let us_10 = (op.elapsed_us * 10.0).to_int()
    let us_whole = us_10 / 10
    let us_frac = us_10 % 10
    buf.write_string(
      "  [\{i}] \{op.op_type} (\{op.node_name}) -> \{@tensor.format_shape(op.output_shape)} : \{us_whole}.\{us_frac} us (\{pct_whole}.\{pct_frac}%)\n",
    )
  }
  let tot_10 = (self.total_us * 10.0).to_int()
  buf.write_string("Total: \{tot_10 / 10}.\{tot_10 % 10} us\n")
  buf.to_string()
}

///|
/// ONNX 推理会话（Session，100% 纯内存、四端通用）：
/// 持有已完成 Lowering、可选图优化与拓扑排序的计算图、可插拔 `ExecutionProvider` 执行后端
/// 以及基于静态生命周期分析的 `BufferPool` 复用池。
pub struct Session {
  graph : @nn_ir.IrGraph
  backend : BackendKind
  /// 可插拔执行后端（默认根据 `backend` 构造 `CpuScalarProvider` 或 `CpuSimdProvider`）
  provider : &ExecutionProvider
  /// 静态生命周期回收计划：`release_plan[i]` 为第 i 个节点执行完后可安全归还至 `BufferPool` 的源张量名
  release_plan : Array[Array[String]]
  /// 标记每个节点的输出是否被图输入/输出/常量锁定（锁定输出不从池分配，避免跨推理结果被覆盖）
  pinned_outputs : Array[Bool]
  /// 会话级缓冲区复用池（跨推理复用 `im2col` 工作区与中间激活张量缓冲）
  pool : @tensor.BufferPool
  /// 预构建的 `If` 控制流分支子会话缓存（键为节点名，值为 `(then_session, else_session)`），
  /// 使子图在初始化期同样完成后端准备与生命周期分析，推理热路径零重复构建开销
  sub_sessions : Map[String, (Session, Session)]
}

///|
/// 判断算子是否为零拷贝视图算子（其输出可能共享第 0 个输入的底层缓冲区）。
fn is_view_op(op : @nn_ir.Op) -> Bool {
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
/// 对已拓扑排序的计算图执行别名感知的静态生命周期分析：
/// - 追踪零拷贝视图算子对源张量缓冲区的别名引用，确保在所有别名最后一次消费完毕后才回收缓冲；
/// - 永久锁定图输入、权重初始化器、常量节点输出与图最终输出（及其源缓冲），防止跨推理数据污染。
fn build_release_plan(
  graph : @nn_ir.IrGraph,
) -> (Array[Array[String]], Array[Bool]) {
  let num_nodes = graph.nodes.length()
  let pinned : Map[String, Bool] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    pinned[graph.inputs[i]] = true
  }
  graph.initializers.each(fn(k, _) { pinned[k] = true })
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    pinned[graph.outputs[i]] = true
  }
  // 子图捕获的外层作用域张量必须锁定，禁止在子图执行期间被归还至子图局部复用池
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

  (release_plan, pinned_outputs)
}

///|
/// 从已构建的 `IrGraph` 与自定义 `ExecutionProvider` 创建推理会话。
pub fn Session::with_provider(
  graph : @nn_ir.IrGraph,
  provider : &ExecutionProvider,
) -> Session {
  let backend = if provider.is_simd() { Simd } else { Scalar }
  provider.prepare_graph(graph)
  let (release_plan, pinned_outputs) = build_release_plan(graph)
  let sub_sessions : Map[String, (Session, Session)] = Map([])
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    match node.op {
      @nn_ir.If(then_branch~, else_branch~) => {
        let then_sess = Session::with_provider(then_branch, provider)
        let else_sess = Session::with_provider(else_branch, provider)
        sub_sessions[node.name] = (then_sess, else_sess)
      }
      _ => ()
    }
  }
  {
    graph,
    backend,
    provider,
    release_plan,
    pinned_outputs,
    pool: @tensor.BufferPool::new(),
    sub_sessions,
  }
}

///|
/// 从已解析的 OnnxModel 构建推理会话（默认启用图优化 Pass、SIMD 权重预打包缓存与 BufferPool 复用）。
pub fn Session::from_model(
  m : @onnx_proto.OnnxModel,
  backend~ : BackendKind = Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  let raw_graph = @frontend.lower_onnx_model(m) catch {
    e => raise IrErr("\{e}")
  }
  let graph = if optimize {
    let (opt_g, _) = @opt.optimize_graph(raw_graph)
    opt_g
  } else {
    raw_graph
  }
  Session::with_provider(graph, default_provider_for_backend(backend))
}

///|
/// 从内存 `.onnx` 字节流直接构建推理会话（全平台通用）。
pub fn Session::from_bytes(
  bytes : Bytes,
  backend~ : BackendKind = Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  let m = @onnx_proto.load_model_from_bytes(bytes) catch {
    e => raise ModelErr("\{e}")
  }
  Session::from_model(m, backend~, optimize~)
}

///|
/// 从已构建的 IrGraph 直接创建推理会话。
pub fn Session::from_graph(
  graph : @nn_ir.IrGraph,
  backend~ : BackendKind = Scalar,
) -> Session {
  Session::with_provider(graph, default_provider_for_backend(backend))
}

///|
/// 返回当前会话的计算图。
pub fn Session::graph(self : Session) -> @nn_ir.IrGraph {
  self.graph
}

///|
/// 返回当前会话使用的计算后端类型。
pub fn Session::backend(self : Session) -> BackendKind {
  self.backend
}

///|
/// 返回当前会话所绑定 `ExecutionProvider` 的名称。
pub fn Session::provider_name(self : Session) -> String {
  self.provider.name()
}

///|
/// 返回当前会话后端已完成预打包的静态权重总数量（含 MatMul / Gemm / Conv / MatMulInteger 及 `If` 分支子图权重）。
pub fn Session::packed_weight_count(self : Session) -> Int32 {
  self.provider.packed_weight_count()
}

///|
/// 返回当前会话后端已完成预打包的静态卷积权重数量（含 `If` 分支子图）。
pub fn Session::packed_conv_weight_count(self : Session) -> Int32 {
  self.provider.packed_conv_weight_count()
}

///|
/// 返回当前会话内部 `BufferPool` 的缓存命中率与滞留缓冲统计信息。
pub fn Session::pool_stats(self : Session) -> @tensor.PoolStats {
  self.pool.stats()
}

///|
/// 清空当前会话的 `BufferPool` 缓存缓冲区以主动释放峰值内存。
pub fn Session::clear_pool(self : Session) -> Unit {
  self.pool.clear()
}

///|
/// 将在当前执行步骤结束后不再存活的中间张量底层 `Float32` 缓冲区归还至会话复用池。
fn release_dead_buffers(
  dead_names : Array[String],
  env : Map[String, @tensor.Tensor],
  pool : @tensor.BufferPool,
) -> Unit {
  for d = 0; d < dead_names.length(); d = d + 1 {
    let name = dead_names[d]
    match env.get(name) {
      Some(t) =>
        match t.raw_f32_buffer() {
          Some(buf) => pool.release_f32(buf)
          None => ()
        }
      None => ()
    }
  }
}

///|
/// 执行单个节点：若节点为 `If` 且已预构建分支子会话，则优先复用预打包权重的子会话零开销执行。
fn Session::eval_step(
  self : Session,
  node : @nn_ir.IrNode,
  ctx : EvalContext,
) -> Array[@tensor.Tensor] raise RuntimeError {
  match node.op {
    @nn_ir.If(..) =>
      match self.sub_sessions.get(node.name) {
        Some((then_sess, else_sess)) => {
          let cond_t = ctx.require_tensor(node.name, node.inputs, 0)
          let cond = read_scalar_bool(node.name, cond_t)
          let branch_sess = if cond { then_sess } else { else_sess }
          branch_sess.run_subgraph(ctx.env)
        }
        None => self.provider.eval_node_multi(node, ctx)
      }
    _ => self.provider.eval_node_multi(node, ctx)
  }
}

///|
/// 统一节点执行主循环（由 `run`、`run_profiled` 与 `run_subgraph` 共享）：
/// 当 `timings` 为 `None` 时走零计时开销热路径；为 `Some(arr)` 时记录每个算子的精确微秒耗时。
fn Session::exec_nodes(
  self : Session,
  env : Map[String, @tensor.Tensor],
  timings : Array[OpTiming]?,
) -> Unit raise RuntimeError {
  for i = 0; i < self.graph.nodes.length(); i = i + 1 {
    let node = self.graph.nodes[i]
    let out_pool : @tensor.BufferPool? = if self.pinned_outputs[i] {
      None
    } else {
      Some(self.pool)
    }
    let ctx : EvalContext = { env, scratch_pool: self.pool, out_pool, }
    let out_tensors = match timings {
      Some(arr) => {
        let t0 = @bench.monotonic_clock_start()
        let outs = self.eval_step(node, ctx)
        let elapsed_us = @bench.monotonic_clock_end(t0)
        let out_shape = if outs.length() > 0 { outs[0].shape() } else { [] }
        arr.push({
          node_name: node.name,
          op_type: node.op.name(),
          output_shape: out_shape,
          elapsed_us,
        })
        outs
      }
      None => self.eval_step(node, ctx)
    }
    for k = 0; k < out_tensors.length() && k < node.outputs.length(); k = k + 1 {
      if node.outputs[k] != "" {
        env[node.outputs[k]] = out_tensors[k]
      }
    }
    release_dead_buffers(self.release_plan[i], env, self.pool)
  }
}

///|
/// 根据图初始化器与用户 `feeds` 构建执行环境字典。
fn Session::init_env(
  self : Session,
  feeds : Map[String, @tensor.Tensor],
) -> Map[String, @tensor.Tensor] raise RuntimeError {
  let env : Map[String, @tensor.Tensor] = Map([])
  self.graph.initializers.each(fn(k, v) { env[k] = v })
  for i = 0; i < self.graph.inputs.length(); i = i + 1 {
    let in_name = self.graph.inputs[i]
    match feeds.get(in_name) {
      Some(t) => env[in_name] = t
      None => raise MissingTensor("missing required graph input '\{in_name}'")
    }
  }
  env
}

///|
/// 从执行环境中收集全部图输出张量。
fn Session::collect_outputs(
  self : Session,
  env : Map[String, @tensor.Tensor],
) -> Map[String, @tensor.Tensor] raise RuntimeError {
  let results : Map[String, @tensor.Tensor] = Map([])
  for i = 0; i < self.graph.outputs.length(); i = i + 1 {
    let out_name = self.graph.outputs[i]
    match env.get(out_name) {
      Some(t) => results[out_name] = t
      None => raise MissingTensor("graph output '\{out_name}' was not produced")
    }
  }
  results
}

///|
/// 在外层词法环境 `outer_env` 上执行当前子会话（供 `If` 分支调用），返回按子图 `outputs` 顺序排列的结果张量。
fn Session::run_subgraph(
  self : Session,
  outer_env : Map[String, @tensor.Tensor],
) -> Array[@tensor.Tensor] raise RuntimeError {
  let env : Map[String, @tensor.Tensor] = Map([])
  outer_env.each(fn(k, v) { env[k] = v })
  self.graph.initializers.each(fn(k, v) { env[k] = v })
  self.exec_nodes(env, None)
  let results : Array[@tensor.Tensor] = []
  for i = 0; i < self.graph.outputs.length(); i = i + 1 {
    let out_name = self.graph.outputs[i]
    match env.get(out_name) {
      Some(t) => results.push(t)
      None =>
        raise MissingTensor("subgraph output '\{out_name}' was not produced")
    }
  }
  results
}

///|
/// 执行一次完整的前向推理（零计时开销热路径，自动通过 `ExecutionProvider` 分派并复用 `BufferPool`）。
pub fn Session::run(
  self : Session,
  feeds : Map[String, @tensor.Tensor],
) -> Map[String, @tensor.Tensor] raise RuntimeError {
  let env = self.init_env(feeds)
  self.exec_nodes(env, None)
  self.collect_outputs(env)
}

///|
/// 执行带逐算子计时剖析的前向推理，返回输出张量字典与 `RunProfile` 报告。
pub fn Session::run_profiled(
  self : Session,
  feeds : Map[String, @tensor.Tensor],
) -> (Map[String, @tensor.Tensor], RunProfile) raise RuntimeError {
  let t_total = @bench.monotonic_clock_start()
  let env = self.init_env(feeds)
  let timings : Array[OpTiming] = []
  self.exec_nodes(env, Some(timings))
  let results = self.collect_outputs(env)
  let total_us = @bench.monotonic_clock_end(t_total)
  (results, { backend: self.backend, total_us, ops: timings, })
}

///|
/// 面向单输入、单输出模型的便捷前向推理接口。
pub fn Session::run_single(
  self : Session,
  input : @tensor.Tensor,
) -> @tensor.Tensor raise RuntimeError {
  if self.graph.inputs.length() != 1 || self.graph.outputs.length() != 1 {
    raise IrErr(
      "run_single requires 1 input and 1 output, got \{self.graph.inputs.length()} inputs and \{self.graph.outputs.length()} outputs",
    )
  }
  let feeds : Map[String, @tensor.Tensor] = Map([])
  feeds[self.graph.inputs[0]] = input
  let outputs = self.run(feeds)
  match outputs.get(self.graph.outputs[0]) {
    Some(t) => t
    None => raise MissingTensor("output tensor missing")
  }
}

///|
/// 面向单输入、单输出模型的带剖析前向推理接口。
pub fn Session::run_single_profiled(
  self : Session,
  input : @tensor.Tensor,
) -> (@tensor.Tensor, RunProfile) raise RuntimeError {
  if self.graph.inputs.length() != 1 || self.graph.outputs.length() != 1 {
    raise IrErr(
      "run_single_profiled requires 1 input and 1 output, got \{self.graph.inputs.length()} inputs and \{self.graph.outputs.length()} outputs",
    )
  }
  let feeds : Map[String, @tensor.Tensor] = Map([])
  feeds[self.graph.inputs[0]] = input
  let (outputs, profile) = self.run_profiled(feeds)
  match outputs.get(self.graph.outputs[0]) {
    Some(t) => (t, profile)
    None => raise MissingTensor("output tensor missing")
  }
}

///|
/// 计算两个同形状 Float32 张量之间的最大绝对误差（`max_abs_diff`）。
pub fn max_abs_diff_f32(
  actual : @tensor.Tensor,
  expected : @tensor.Tensor,
) -> Float raise RuntimeError {
  if actual.shape() != expected.shape() {
    raise TensorErr(
      "shape mismatch: actual=\{@tensor.format_shape(actual.shape())} vs expected=\{@tensor.format_shape(expected.shape())}",
    )
  }
  let a = actual.to_f32_buffer() catch { e => raise TensorErr("\{e}") }
  let b = expected.to_f32_buffer() catch { e => raise TensorErr("\{e}") }
  let mut max_diff : Float = 0.0
  for i = 0; i < a.length(); i = i + 1 {
    let d = (a[i] - b[i]).abs()
    if d > max_diff {
      max_diff = d
    }
  }
  max_diff
}
