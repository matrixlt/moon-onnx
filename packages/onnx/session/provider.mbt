///|
/// 单节点执行上下文，向 `ExecutionProvider` 传递纯运行期张量环境与 `BufferPool`（不泄露具体后端私有权重格式）。
pub struct EvalContext {
  env : Map[String, @tensor.Tensor]
  scratch_pool : @tensor.BufferPool
  out_pool : @tensor.BufferPool?
}

///|
/// 从执行上下文中读取节点的必选输入张量（缺失时返回 `MissingTensor`）。
pub fn EvalContext::require_tensor(
  self : EvalContext,
  node_name : String,
  inputs : Array[String],
  idx : Int32,
) -> @tensor.Tensor raise RuntimeError {
  if idx >= inputs.length() || inputs[idx] == "" {
    raise MissingTensor(
      "node '\{node_name}' missing required input at index \{idx}",
    )
  }
  let name = inputs[idx]
  match self.env.get(name) {
    Some(t) => t
    None =>
      raise MissingTensor(
        "node '\{node_name}' could not find tensor '\{name}' in runtime env",
      )
  }
}

///|
/// 从执行上下文中读取节点的可选输入张量（空字符串或下标越界时返回 `None`）。
pub fn EvalContext::optional_tensor(
  self : EvalContext,
  node_name : String,
  inputs : Array[String],
  idx : Int32,
) -> @tensor.Tensor? raise RuntimeError {
  if idx >= inputs.length() || inputs[idx] == "" {
    return None
  }
  let name = inputs[idx]
  match self.env.get(name) {
    Some(t) => Some(t)
    None =>
      raise MissingTensor(
        "node '\{node_name}' could not find optional tensor '\{name}'",
      )
  }
}

///|
/// 可插拔推理执行后端接口（参考 ONNX Runtime `ExecutionProvider`）。
/// 允许用户在不修改 `Session` 调度与静态生命周期 `BufferPool` 复用逻辑的前提下，
/// 接入自定义算子加速后端（如 WebGPU Compute Shader、Native BLAS 或专用量化内核）。
pub(open) trait ExecutionProvider {
  /// 返回后端唯一标识名称（如 `"scalar"`、`"simd-v128"`、`"webgpu"`）
  fn name(Self) -> String
  /// 指示后端是否采用 SIMD 加速路径（默认 `false`）
  fn is_simd(Self) -> Bool = _
  /// 会话初始化期图准备钩子：由后端私有完成静态权重预打包或设备上传（默认空操作）
  fn prepare_graph(Self, @nn_ir.IrGraph) -> Unit = _
  /// 查询后端已预打包的静态权重总数（供诊断与测试使用，默认返回 0）
  fn packed_weight_count(Self) -> Int32 = _
  /// 查询后端已预打包的静态卷积权重数（供诊断与测试使用，默认返回 0）
  fn packed_conv_weight_count(Self) -> Int32 = _
  /// 执行单个强类型 IR 节点并返回主输出张量
  fn eval_node(Self, @nn_ir.IrNode, EvalContext) -> @tensor.Tensor raise RuntimeError
  /// 执行可能产生多个输出张量的 IR 节点（例如 `Split`、`LSTM`、`DynamicQuantizeLinear`、`If`），默认退化为单输出包装
  fn eval_node_multi(Self, @nn_ir.IrNode, EvalContext) -> Array[@tensor.Tensor] raise RuntimeError = _
}

///|
impl ExecutionProvider with fn is_simd(_self) {
  false
}

///|
impl ExecutionProvider with fn prepare_graph(_self, _graph) {
  ()
}

///|
impl ExecutionProvider with fn packed_weight_count(_self) {
  0
}

///|
impl ExecutionProvider with fn packed_conv_weight_count(_self) {
  0
}

///|
/// `ExecutionProvider::eval_node_multi` 默认实现：直接复用单输出 `eval_node`。
impl ExecutionProvider with fn eval_node_multi(self, node, ctx) {
  [self.eval_node(node, ctx)]
}

///|
/// 内置纯 MoonBit 标量参考执行后端（100% 跨四端，适合作为数值验证黄金基准）。
pub struct CpuScalarProvider {}

///|
/// 创建标量 CPU 执行后端实例。
pub fn CpuScalarProvider::new() -> CpuScalarProvider {
  CpuScalarProvider::{ }
}

///|
/// 内置基于 MoonBit 原生 `V128` SIMD 与私有静态权重预打包缓存的高性能 CPU/Wasm 执行后端。
pub struct CpuSimdProvider {
  priv cache : CpuWeightCache
}

///|
/// 创建 `V128` SIMD 执行后端实例（私有权重缓存在 `Session::with_provider` 调用 `prepare_graph` 时填充）。
pub fn CpuSimdProvider::new() -> CpuSimdProvider {
  {
    cache: {
      packed_weights: Map([]),
      packed_conv_weights: Map([]),
      packed_i8_weights: Map([]),
    },
  }
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn name(_self) {
  "scalar"
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn eval_node(
  _self,
  node,
  ctx,
) {
  eval_cpu_node(node, ctx, false, None)
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn eval_node_multi(
  _self,
  node,
  ctx,
) {
  eval_cpu_node_multi(node, ctx, false, None)
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn name(_self) {
  "simd-v128"
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn is_simd(_self) {
  true
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn prepare_graph(
  self,
  graph,
) {
  self.cache.prepare(graph)
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn packed_weight_count(self) {
  let mut count = 0
  self.cache.packed_weights.each(fn(_, _) { count = count + 1 })
  self.cache.packed_conv_weights.each(fn(_, _) { count = count + 1 })
  self.cache.packed_i8_weights.each(fn(_, _) { count = count + 1 })
  count
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn packed_conv_weight_count(
  self,
) {
  let mut count = 0
  self.cache.packed_conv_weights.each(fn(_, _) { count = count + 1 })
  count
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn eval_node(
  self,
  node,
  ctx,
) {
  eval_cpu_node(node, ctx, true, Some(self.cache))
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn eval_node_multi(
  self,
  node,
  ctx,
) {
  eval_cpu_node_multi(node, ctx, true, Some(self.cache))
}

///|
/// 将 `BackendKind` 枚举转换为对应的默认内置 `ExecutionProvider` 对象。
pub fn default_provider_for_backend(
  backend : BackendKind,
) -> &ExecutionProvider {
  match backend {
    Scalar => (CpuScalarProvider::new() : &ExecutionProvider)
    Simd => (CpuSimdProvider::new() : &ExecutionProvider)
  }
}
