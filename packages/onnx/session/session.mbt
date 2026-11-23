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
/// ONNX 推理会话（Session）：
/// 持有已完成 Lowering、可选图优化与拓扑排序的计算图、所选后端以及 SIMD 静态权重预打包缓存。
pub struct Session {
  graph : @nn_ir.IrGraph
  backend : BackendKind
  /// 静态 2D Float32 权重的预打包缓存（键为权重张量名），运行期直接走零打包 SIMD GEMM
  packed_weights : Map[String, @gemm.PackedMatrixB]
}

///|
/// 在会话初始化期扫描计算图，将所有用于 `Gemm(trans_b=false)` 与 `MatMul` 的静态 2D Float32 权重
/// 一次性打包为 16 字节对齐的 `PackedMatrixB`，彻底消除推理热路径上的重复打包开销。
fn build_packed_weight_cache(
  graph : @nn_ir.IrGraph,
  backend : BackendKind,
) -> Map[String, @gemm.PackedMatrixB] {
  let cache : Map[String, @gemm.PackedMatrixB] = Map([])
  if backend != Simd {
    return cache
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let should_pack = match node.op {
      @nn_ir.MatMul => true
      @nn_ir.Gemm(trans_b~, ..) => !trans_b
      _ => false
    }
    if should_pack && node.inputs.length() >= 2 {
      let w_name = node.inputs[1]
      if !cache.contains(w_name) {
        match graph.initializers.get(w_name) {
          Some(w_t) if w_t.rank() == 2 && w_t.dtype() == @tensor.Float32 => {
            let ws = w_t.shape()
            try {
              let w_buf = w_t.to_f32_buffer()
              cache[w_name] = @gemm.pack_matrix_b_f32(w_buf, 0, ws[0], ws[1])
            } catch {
              _ => ()
            }
          }
          _ => ()
        }
      }
    }
  }
  cache
}

///|
/// 从已解析的 OnnxModel 构建推理会话（默认启用图优化 Pass 与 SIMD 权重预打包缓存）。
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
  let packed_weights = build_packed_weight_cache(graph, backend)
  { graph, backend, packed_weights, }
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
/// 从本地 `.onnx` 文件路径构建推理会话（Native 入口）。
pub fn Session::from_file(
  path : String,
  backend~ : BackendKind = Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  let m = @onnx_fs.load_model_from_file(path) catch {
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
  let packed_weights = build_packed_weight_cache(graph, backend)
  { graph, backend, packed_weights, }
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
/// 返回当前会话中已完成预打包的静态权重矩阵数量（便于测试与诊断）。
pub fn Session::packed_weight_count(self : Session) -> Int32 {
  let mut count = 0
  self.packed_weights.each(fn(_, _) { count = count + 1 })
  count
}

///|
/// 从运行期张量环境表中读取必选输入张量。
fn require_tensor(
  env : Map[String, @tensor.Tensor],
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
  match env.get(name) {
    Some(t) => t
    None =>
      raise MissingTensor(
        "node '\{node_name}' could not find tensor '\{name}' in runtime env",
      )
  }
}

///|
/// 从运行期张量环境表中读取可选输入张量。
fn optional_tensor(
  env : Map[String, @tensor.Tensor],
  node_name : String,
  inputs : Array[String],
  idx : Int32,
) -> @tensor.Tensor? raise RuntimeError {
  if idx >= inputs.length() || inputs[idx] == "" {
    return None
  }
  let name = inputs[idx]
  match env.get(name) {
    Some(t) => Some(t)
    None =>
      raise MissingTensor(
        "node '\{node_name}' could not find optional tensor '\{name}'",
      )
  }
}

///|
/// 执行单个强类型 IR 节点并返回输出张量（SIMD 后端下优先命中静态 `PackedMatrixB` 缓存）。
fn eval_node(
  node : @nn_ir.IrNode,
  env : Map[String, @tensor.Tensor],
  backend : BackendKind,
  packed_weights : Map[String, @gemm.PackedMatrixB],
) -> @tensor.Tensor raise RuntimeError {
  let n = node.name
  let ins = node.inputs
  let use_simd = backend == Simd
  try {
    match node.op {
      @nn_ir.Add => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.add_f32(a, b, use_simd~)
      }
      @nn_ir.Sub => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.sub_f32(a, b, use_simd~)
      }
      @nn_ir.Mul => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.mul_f32(a, b, use_simd~)
      }
      @nn_ir.Relu => {
        let x = require_tensor(env, n, ins, 0)
        @ops.relu_f32(x, use_simd~)
      }
      @nn_ir.Identity => require_tensor(env, n, ins, 0)
      @nn_ir.Softmax(axis~) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.softmax_f32(x, axis)
      }
      @nn_ir.Flatten(axis~) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.flatten_op(x, axis~)
      }
      @nn_ir.Reshape(allowzero~) => {
        let data = require_tensor(env, n, ins, 0)
        let shape_t = require_tensor(env, n, ins, 1)
        @ops.reshape_op(data, shape_t, allowzero~)
      }
      @nn_ir.Transpose(perm~) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.transpose_op(x, perm)
      }
      @nn_ir.Squeeze(axes~) => {
        let x = require_tensor(env, n, ins, 0)
        let axes_t = optional_tensor(env, n, ins, 1)
        @ops.squeeze_op(x, axes, axes_t)
      }
      @nn_ir.Unsqueeze(axes~) => {
        let x = require_tensor(env, n, ins, 0)
        let axes_t = optional_tensor(env, n, ins, 1)
        @ops.unsqueeze_op(x, axes, axes_t)
      }
      @nn_ir.Constant(value~) => value
      @nn_ir.MatMul => {
        let a = require_tensor(env, n, ins, 0)
        if use_simd && a.rank() == 2 && ins.length() >= 2 {
          match packed_weights.get(ins[1]) {
            Some(pb) => return @ops.matmul_prepacked_f32(a, pb)
            None => ()
          }
        }
        let b = require_tensor(env, n, ins, 1)
        @ops.matmul_f32(a, b, use_simd~)
      }
      @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
        let a = require_tensor(env, n, ins, 0)
        let c = optional_tensor(env, n, ins, 2)
        if use_simd && !trans_b && ins.length() >= 2 {
          match packed_weights.get(ins[1]) {
            Some(pb) =>
              return @ops.gemm_prepacked_f32(a, pb, c, alpha~, beta~, trans_a~)
            None => ()
          }
        }
        let b = require_tensor(env, n, ins, 1)
        @ops.gemm_f32(a, b, c, alpha~, beta~, trans_a~, trans_b~, use_simd~)
      }
      @nn_ir.Conv(strides~, pads~, dilations~, group~, auto_pad~, ..) => {
        let x = require_tensor(env, n, ins, 0)
        let w = require_tensor(env, n, ins, 1)
        let b = optional_tensor(env, n, ins, 2)
        @ops.conv2d_f32(
          x,
          w,
          b,
          strides,
          pads,
          dilations,
          group,
          auto_pad,
          use_simd~,
        )
      }
      @nn_ir.MaxPool(
        kernel_shape~,
        strides~,
        pads~,
        dilations~,
        ceil_mode~,
        auto_pad~,
      ) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.maxpool_2d_f32(
          x, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad,
        )
      }
      @nn_ir.AveragePool(
        kernel_shape~,
        strides~,
        pads~,
        ceil_mode~,
        count_include_pad~,
        auto_pad~,
      ) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.avgpool_2d_f32(
          x, kernel_shape, strides, pads, ceil_mode, count_include_pad, auto_pad,
        )
      }
      @nn_ir.BatchNormalization(epsilon~) => {
        let x = require_tensor(env, n, ins, 0)
        let scale = require_tensor(env, n, ins, 1)
        let bias = require_tensor(env, n, ins, 2)
        let mean = require_tensor(env, n, ins, 3)
        let variance = require_tensor(env, n, ins, 4)
        @ops.batch_norm_f32(x, scale, bias, mean, variance, epsilon~)
      }
    }
  } catch {
    MissingTensor(msg) => raise MissingTensor(msg)
    UnsupportedOp(msg) => raise UnsupportedOp(msg)
    e => raise TensorErr("node '\{n}' (\{node.op.name()}): \{e}")
  }
}

///|
/// 执行一次完整的前向推理（零计时开销热路径）。
pub fn Session::run(
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

  for i = 0; i < self.graph.nodes.length(); i = i + 1 {
    let node = self.graph.nodes[i]
    let out_tensor = eval_node(node, env, self.backend, self.packed_weights)
    if node.outputs.length() > 0 && node.outputs[0] != "" {
      env[node.outputs[0]] = out_tensor
    }
  }

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
/// 执行带逐算子计时剖析的前向推理，返回输出张量字典与 `RunProfile` 报告。
pub fn Session::run_profiled(
  self : Session,
  feeds : Map[String, @tensor.Tensor],
) -> (Map[String, @tensor.Tensor], RunProfile) raise RuntimeError {
  let t_total = @bench.monotonic_clock_start()
  let env : Map[String, @tensor.Tensor] = Map([])
  self.graph.initializers.each(fn(k, v) { env[k] = v })

  for i = 0; i < self.graph.inputs.length(); i = i + 1 {
    let in_name = self.graph.inputs[i]
    match feeds.get(in_name) {
      Some(t) => env[in_name] = t
      None => raise MissingTensor("missing required graph input '\{in_name}'")
    }
  }

  let timings : Array[OpTiming] = []
  for i = 0; i < self.graph.nodes.length(); i = i + 1 {
    let node = self.graph.nodes[i]
    let t0 = @bench.monotonic_clock_start()
    let out_tensor = eval_node(node, env, self.backend, self.packed_weights)
    let elapsed_us = @bench.monotonic_clock_end(t0)
    timings.push({
      node_name: node.name,
      op_type: node.op.name(),
      output_shape: out_tensor.shape(),
      elapsed_us,
    })
    if node.outputs.length() > 0 && node.outputs[0] != "" {
      env[node.outputs[0]] = out_tensor
    }
  }

  let results : Map[String, @tensor.Tensor] = Map([])
  for i = 0; i < self.graph.outputs.length(); i = i + 1 {
    let out_name = self.graph.outputs[i]
    match env.get(out_name) {
      Some(t) => results[out_name] = t
      None => raise MissingTensor("graph output '\{out_name}' was not produced")
    }
  }
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
/// 从原始小端序 float32 二进制文件（`.f32`）加载张量。
pub fn load_f32_file(
  path : String,
  shape : Array[Int32],
) -> @tensor.Tensor raise RuntimeError {
  let bytes = @fs.read_file_to_bytes(path) catch {
    @fs.IOError(msg) => raise ModelErr("failed to read '\{path}': \{msg}")
  }
  @tensor.Tensor::from_f32_bytes_le(shape, bytes) catch {
    e => raise TensorErr("file '\{path}': \{e}")
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
