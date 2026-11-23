///|
/// 推理会话构建与执行阶段可能抛出的结构化错误。
#derive(Debug, Eq)
pub(all) suberror RuntimeError {
  /// 模型解析错误
  ModelErr(String)
  /// 计算图 Lowering 或校验错误
  IrErr(String)
  /// 张量运算错误
  TensorErr(String)
  /// 运行期缺少所需的输入或中间张量
  MissingTensor(String)
  /// 遇到当前执行后端尚未实现的算子
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
/// 推理计算后端类型选择。
/// - `Scalar`：100% 纯 MoonBit 标量参考内核（正确性黄金基准）
/// - `Simd`  ：基于 MoonBit V128（f32x4）的向量化加速内核
#derive(Debug, Eq)
pub(all) enum BackendKind {
  Scalar
  Simd
}

///|
/// 为 BackendKind 提供可读字符串名称。
pub impl Show for BackendKind with fn output(self, logger) {
  match self {
    Scalar => logger.write_string("scalar")
    Simd => logger.write_string("simd-v128")
  }
}

///|
/// ONNX 推理会话（Session）：持有已完成 Lowering 与拓扑排序的计算图及所选后端。
pub struct Session {
  graph : @ir.IrGraph
  backend : BackendKind
}

///|
/// 从已解析的 OnnxModel 构建推理会话（默认启用图优化 Pass）。
pub fn Session::from_model(
  m : @model.OnnxModel,
  backend~ : BackendKind = Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  let raw_graph = @ir.lower_model(m) catch { e => raise IrErr("\{e}") }
  let graph = if optimize {
    let (opt_g, _) = @opt.optimize_graph(raw_graph)
    opt_g
  } else {
    raw_graph
  }
  { graph, backend, }
}

///|
/// 从内存 `.onnx` 字节流直接构建推理会话（全平台通用）。
pub fn Session::from_bytes(
  bytes : Bytes,
  backend~ : BackendKind = Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  let m = @model.load_model_from_bytes(bytes) catch {
    e => raise ModelErr("\{e}")
  }
  Session::from_model(m, backend~, optimize~)
}

///|
/// 从本地 `.onnx` 文件路径构建推理会话（Native 目标入口）。
pub fn Session::from_file(
  path : String,
  backend~ : BackendKind = Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  let m = @model.load_model_from_file(path) catch {
    e => raise ModelErr("\{e}")
  }
  Session::from_model(m, backend~, optimize~)
}

///|
/// 从已构建的 IrGraph 创建推理会话（供图优化 Pass 产物直接执行）。
pub fn Session::from_graph(
  graph : @ir.IrGraph,
  backend~ : BackendKind = Scalar,
) -> Session {
  { graph, backend, }
}

///|
/// 返回当前会话使用的计算图。
pub fn Session::graph(self : Session) -> @ir.IrGraph {
  self.graph
}

///|
/// 返回当前会话选择的后端类型。
pub fn Session::backend(self : Session) -> BackendKind {
  self.backend
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
/// 从运行期张量环境表中读取可选输入张量（索引越界或名称为空字符串均返回 None）。
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
/// 执行单个强类型 IR 节点并返回输出张量。
fn eval_node(
  node : @ir.IrNode,
  env : Map[String, @tensor.Tensor],
  _backend : BackendKind,
) -> @tensor.Tensor raise RuntimeError {
  let n = node.name
  let ins = node.inputs
  try {
    match node.op {
      @ir.Add => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.add_f32(a, b)
      }
      @ir.Sub => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.sub_f32(a, b)
      }
      @ir.Mul => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.mul_f32(a, b)
      }
      @ir.Relu => {
        let x = require_tensor(env, n, ins, 0)
        @ops.relu_f32(x)
      }
      @ir.Identity => require_tensor(env, n, ins, 0)
      @ir.Softmax(axis~) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.softmax_f32(x, axis)
      }
      @ir.Flatten(axis~) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.flatten_op(x, axis~)
      }
      @ir.Reshape(allowzero~) => {
        let data = require_tensor(env, n, ins, 0)
        let shape_t = require_tensor(env, n, ins, 1)
        @ops.reshape_op(data, shape_t, allowzero~)
      }
      @ir.Transpose(perm~) => {
        let x = require_tensor(env, n, ins, 0)
        @ops.transpose_op(x, perm)
      }
      @ir.Squeeze(axes~) => {
        let x = require_tensor(env, n, ins, 0)
        let axes_t = optional_tensor(env, n, ins, 1)
        @ops.squeeze_op(x, axes, axes_t)
      }
      @ir.Unsqueeze(axes~) => {
        let x = require_tensor(env, n, ins, 0)
        let axes_t = optional_tensor(env, n, ins, 1)
        @ops.unsqueeze_op(x, axes, axes_t)
      }
      @ir.Constant(value~) => value
      @ir.MatMul => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        @ops.matmul_f32(a, b)
      }
      @ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
        let a = require_tensor(env, n, ins, 0)
        let b = require_tensor(env, n, ins, 1)
        let c = optional_tensor(env, n, ins, 2)
        @ops.gemm_f32(a, b, c, alpha~, beta~, trans_a~, trans_b~)
      }
      @ir.Conv(strides~, pads~, dilations~, group~, auto_pad~, ..) => {
        let x = require_tensor(env, n, ins, 0)
        let w = require_tensor(env, n, ins, 1)
        let b = optional_tensor(env, n, ins, 2)
        @ops.conv2d_f32(x, w, b, strides, pads, dilations, group, auto_pad)
      }
      @ir.MaxPool(
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
      @ir.AveragePool(
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
      @ir.BatchNormalization(epsilon~) => {
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
/// 执行一次完整的前向推理，传入输入张量字典并返回所有图输出张量。
pub fn Session::run(
  self : Session,
  feeds : Map[String, @tensor.Tensor],
) -> Map[String, @tensor.Tensor] raise RuntimeError {
  // 1. 初始化运行期张量环境表：先加载全部权重 Initializer，再写入外部输入
  let env : Map[String, @tensor.Tensor] = Map([])
  self.graph.initializers.each(fn(k, v) { env[k] = v })

  for i = 0; i < self.graph.inputs.length(); i = i + 1 {
    let in_name = self.graph.inputs[i]
    match feeds.get(in_name) {
      Some(t) => env[in_name] = t
      None => raise MissingTensor("missing required graph input '\{in_name}'")
    }
  }

  // 2. 按已验证的拓扑序逐节点求值
  for i = 0; i < self.graph.nodes.length(); i = i + 1 {
    let node = self.graph.nodes[i]
    let out_tensor = eval_node(node, env, self.backend)
    if node.outputs.length() > 0 && node.outputs[0] != "" {
      env[node.outputs[0]] = out_tensor
    }
  }

  // 3. 收集声明的图输出张量
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
/// 面向单输入、单输出模型的便捷前向推理接口（如 MNIST 图像分类）。
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
/// 从原始小端序 float32 二进制文件（`.f32`）加载张量，供 CLI 与差分测试使用。
pub fn load_f32_file(
  path : String,
  shape : Array[Int32],
) -> @tensor.Tensor raise RuntimeError {
  let bytes = @fs.read_file_to_bytes(path) catch {
    @fs.IOError(msg) => raise ModelErr("failed to read '\{path}': \{msg}")
  }
  let expected = @tensor.num_elements(shape) catch {
    e => raise TensorErr("\{e}")
  }
  if bytes.length() != expected * 4 {
    raise TensorErr(
      "file '\{path}' has \{bytes.length()} bytes, expected \{expected * 4} bytes for shape \{@tensor.format_shape(shape)}",
    )
  }
  let buf = FixedArray::makei(expected, fn(i) {
    let off = i * 4
    let bits = bytes[off].to_uint() |
      (bytes[off + 1].to_uint() << 8) |
      (bytes[off + 2].to_uint() << 16) |
      (bytes[off + 3].to_uint() << 24)
    Float::reinterpret_from_uint(bits)
  })
  @tensor.Tensor::from_f32(shape, buf) catch {
    e => raise TensorErr("\{e}")
  }
}

///|
/// 计算两个同形状 Float32 张量之间的最大绝对误差（max_abs_diff），用于与 ONNX Runtime 数值对齐校验。
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
