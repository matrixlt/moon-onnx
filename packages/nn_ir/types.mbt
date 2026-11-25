///|
/// 计算图 Lowering、校验与拓扑排序阶段可能抛出的结构化错误。
#derive(Debug, Eq)
pub(all) suberror IrError {
  /// 模型包含引擎尚未实现的算子类型
  UnsupportedOp(String)
  /// 算子属性缺失、类型不匹配或取值非法
  InvalidAttribute(String)
  /// 节点引用的输入张量既不是图输入/权重，也不是上游节点输出
  MissingInput(String)
  /// 计算图存在环路，无法拓扑排序
  CyclicGraph(String)
  /// 其他图结构错误
  InvalidGraph(String)
}

///|
/// 为 IrError 提供清晰可读的错误信息。
pub impl Show for IrError with fn output(self, logger) {
  match self {
    UnsupportedOp(op) => logger.write_string("UnsupportedOp(\{op})")
    InvalidAttribute(msg) => logger.write_string("InvalidAttribute(\{msg})")
    MissingInput(msg) => logger.write_string("MissingInput(\{msg})")
    CyclicGraph(msg) => logger.write_string("CyclicGraph(\{msg})")
    InvalidGraph(msg) => logger.write_string("InvalidGraph(\{msg})")
  }
}

///|
/// ONNX Conv / Pool 算子的自动填充模式。
#derive(Debug, Eq)
pub(all) enum AutoPad {
  NotSet
  SameUpper
  SameLower
  Valid
}

///|
/// 融合至卷积算子尾部的逐元素激活函数（在 SIMD 累加器写回前原地执行，消除独立激活节点读写）。
#derive(Debug, Eq)
pub(all) enum FusedActivation {
  /// 无融合激活
  NoAct
  /// ReLU 激活：`max(0.0, x)`
  ReluAct
  /// 区间裁剪激活：`min(max, max(min, x))`（例如 MobileNetV2 的 ReLU6 即 `[0.0, 6.0]`）
  ClipAct(min~ : Float, max~ : Float)
}

///|
/// 强类型算子代数数据类型（ADT）。
/// 在模型加载期即完成字符串 op_type 与动态 AttributeProto 的解析与校验，
/// 运行期只需通过穷尽模式匹配分派，无字符串查找开销。
pub(all) enum Op {
  Add
  Sub
  Mul
  Div
  Pow
  Sqrt
  Exp
  Neg
  Relu
  Clip(min~ : Float?, max~ : Float?)
  Sigmoid
  Tanh
  Gelu
  HardSwish
  LeakyRelu(alpha~ : Float)
  Identity
  Softmax(axis~ : Int32)
  Flatten(axis~ : Int32)
  Reshape(allowzero~ : Bool)
  Transpose(perm~ : Array[Int32])
  /// Opset < 13 时 axes 通过属性传入；Opset >= 13 时 axes 作为第 2 个输入张量传入（此处为 None）
  Squeeze(axes~ : Array[Int32]?)
  Unsqueeze(axes~ : Array[Int32]?)
  Constant(value~ : @tensor.Tensor)
  MatMul
  Gemm(alpha~ : Float, beta~ : Float, trans_a~ : Bool, trans_b~ : Bool)
  Conv(
    kernel_shape~ : Array[Int32]?,
    strides~ : Array[Int32],
    pads~ : Array[Int32],
    dilations~ : Array[Int32],
    group~ : Int32,
    auto_pad~ : AutoPad,
    activation~ : FusedActivation,
  )
  MaxPool(
    kernel_shape~ : Array[Int32],
    strides~ : Array[Int32],
    pads~ : Array[Int32],
    dilations~ : Array[Int32],
    ceil_mode~ : Bool,
    auto_pad~ : AutoPad,
  )
  AveragePool(
    kernel_shape~ : Array[Int32],
    strides~ : Array[Int32],
    pads~ : Array[Int32],
    ceil_mode~ : Bool,
    count_include_pad~ : Bool,
    auto_pad~ : AutoPad,
  )
  GlobalAveragePool
  BatchNormalization(epsilon~ : Float)
  LayerNormalization(axis~ : Int32, epsilon~ : Float)
  Concat(axis~ : Int32)
  /// Opset < 13 通过属性传入 split；Opset >= 13 通过第 2 个输入传入；Opset >= 18 支持 num_outputs
  Split(axis~ : Int32, split~ : Array[Int32]?, num_outputs~ : Int32?)
  /// Opset < 10 通过属性传入 starts/ends/axes；Opset >= 10 通过输入张量传入
  Slice(
    starts~ : Array[Int32]?,
    ends~ : Array[Int32]?,
    axes~ : Array[Int32]?,
    steps~ : Array[Int32]?,
  )
  /// Opset < 11 通过属性传入 pads/constant_value；Opset >= 11 通过输入张量传入
  Pad(mode~ : String, pads~ : Array[Int32]?, constant_value~ : Float?)
  Gather(axis~ : Int32)
  Cast(to~ : @tensor.DType)
  ReduceMean(
    axes~ : Array[Int32]?,
    keepdims~ : Bool,
    noop_with_empty_axes~ : Bool,
  )
  ReduceSum(
    axes~ : Array[Int32]?,
    keepdims~ : Bool,
    noop_with_empty_axes~ : Bool,
  )
  Shape(start~ : Int32?, end~ : Int32?)
  QuantizeLinear(axis~ : Int32)
  DequantizeLinear(axis~ : Int32)
  QLinearMatMul
  Resize(
    mode~ : String,
    coordinate_transformation_mode~ : String,
    nearest_mode~ : String,
  )
  Lstm(
    hidden_size~ : Int32?,
    direction~ : String,
    layout~ : Int32,
    clip~ : Float?,
  )
  /// 逐元素相等比较（支持 NumPy 广播，输出 `Bool` 张量）
  Equal
  /// 逐元素小于比较（支持 NumPy 广播，输出 `Bool` 张量）
  Less
  /// 逐元素大于比较（支持 NumPy 广播，输出 `Bool` 张量）
  Greater
  /// 逐元素小于等于比较（支持 NumPy 广播，输出 `Bool` 张量）
  LessOrEqual
  /// 逐元素大于等于比较（支持 NumPy 广播，输出 `Bool` 张量）
  GreaterOrEqual
  /// 逐元素布尔取反
  Not
  /// 逐元素布尔与（支持 NumPy 广播）
  And
  /// 逐元素三目选择 `Where(condition, X, Y)`（支持三向 NumPy 广播）
  Where
  /// 高斯误差函数 `erf(x)`（PyTorch/HuggingFace 精确 GELU 核心算子）
  Erf
  /// 等差一维序列生成 `Range(start, limit, delta)`
  Range
  /// 多向广播张量扩展 `Expand(input, shape)`
  Expand
  /// 沿各维度重复平铺张量 `Tile(input, repeats)`
  Tile
  /// 按输入 1D 形状张量填充标量常量的张量生成算子（默认 `Float32(0.0)`）
  ConstantOfShape(value~ : @tensor.Tensor)
  /// ONNX 条件控制流算子：根据标量布尔条件选择执行 `then_branch` 或 `else_branch` 子图
  If(then_branch~ : IrGraph, else_branch~ : IrGraph)
  /// 动态线性量化 `DynamicQuantizeLinear(x) -> (y: UInt8, y_scale: Float32, y_zero_point: UInt8)`
  DynamicQuantizeLinear
  /// 整数矩阵乘法 `MatMulInteger(A, B, a_zero_point?, b_zero_point?) -> Int32`（支持 N-D × 2D 与 INT8/UINT8 混合）
  MatMulInteger
}

///|
/// 返回算子的简短类型名，用于 Profiler、日志与图摘要展示。
pub fn Op::name(self : Op) -> String {
  match self {
    Add => "Add"
    Sub => "Sub"
    Mul => "Mul"
    Div => "Div"
    Pow => "Pow"
    Sqrt => "Sqrt"
    Exp => "Exp"
    Neg => "Neg"
    Relu => "Relu"
    Clip(..) => "Clip"
    Sigmoid => "Sigmoid"
    Tanh => "Tanh"
    Gelu => "Gelu"
    HardSwish => "HardSwish"
    LeakyRelu(..) => "LeakyRelu"
    Identity => "Identity"
    Softmax(..) => "Softmax"
    Flatten(..) => "Flatten"
    Reshape(..) => "Reshape"
    Transpose(..) => "Transpose"
    Squeeze(..) => "Squeeze"
    Unsqueeze(..) => "Unsqueeze"
    Constant(..) => "Constant"
    MatMul => "MatMul"
    Gemm(..) => "Gemm"
    Conv(activation~, ..) =>
      match activation {
        NoAct => "Conv"
        ReluAct => "Conv+Relu"
        ClipAct(..) => "Conv+Clip"
      }
    MaxPool(..) => "MaxPool"
    AveragePool(..) => "AveragePool"
    GlobalAveragePool => "GlobalAveragePool"
    BatchNormalization(..) => "BatchNormalization"
    LayerNormalization(..) => "LayerNormalization"
    Concat(..) => "Concat"
    Split(..) => "Split"
    Slice(..) => "Slice"
    Pad(..) => "Pad"
    Gather(..) => "Gather"
    Cast(..) => "Cast"
    ReduceMean(..) => "ReduceMean"
    ReduceSum(..) => "ReduceSum"
    Shape(..) => "Shape"
    QuantizeLinear(..) => "QuantizeLinear"
    DequantizeLinear(..) => "DequantizeLinear"
    QLinearMatMul => "QLinearMatMul"
    Resize(..) => "Resize"
    Lstm(..) => "LSTM"
    Equal => "Equal"
    Less => "Less"
    Greater => "Greater"
    LessOrEqual => "LessOrEqual"
    GreaterOrEqual => "GreaterOrEqual"
    Not => "Not"
    And => "And"
    Where => "Where"
    Erf => "Erf"
    Range => "Range"
    Expand => "Expand"
    Tile => "Tile"
    ConstantOfShape(..) => "ConstantOfShape"
    If(..) => "If"
    DynamicQuantizeLinear => "DynamicQuantizeLinear"
    MatMulInteger => "MatMulInteger"
  }
}

///|
/// 强类型计算图节点。
pub(all) struct IrNode {
  name : String
  op : Op
  inputs : Array[String]
  outputs : Array[String]
}

///|
/// 已校验并按拓扑序排好序的计算图中间表示（Typed IR）。
pub(all) struct IrGraph {
  name : String
  inputs : Array[String]
  outputs : Array[String]
  initializers : Map[String, @tensor.Tensor]
  nodes : Array[IrNode]
}

///|
/// 收集子图（如 `If` 的 `then_branch`/`else_branch`）中引用自外层作用域的全部自由变量名称。
/// 用于拓扑排序、死代码消除（DCE）与静态生命周期 `BufferPool` 分析正确追踪词法捕获依赖。
pub fn IrGraph::external_refs(self : IrGraph) -> Array[String] {
  let local_defs : Map[String, Bool] = Map([])
  for i = 0; i < self.inputs.length(); i = i + 1 {
    local_defs[self.inputs[i]] = true
  }
  self.initializers.each(fn(k, _) { local_defs[k] = true })
  for i = 0; i < self.nodes.length(); i = i + 1 {
    let outs = self.nodes[i].outputs
    for k = 0; k < outs.length(); k = k + 1 {
      if outs[k] != "" {
        local_defs[outs[k]] = true
      }
    }
  }

  let seen : Map[String, Bool] = Map([])
  let refs : Array[String] = []
  fn record(name : String) -> Unit {
    if name != "" && !local_defs.contains(name) && !seen.contains(name) {
      seen[name] = true
      refs.push(name)
    }
  }

  for i = 0; i < self.nodes.length(); i = i + 1 {
    let node = self.nodes[i]
    for j = 0; j < node.inputs.length(); j = j + 1 {
      record(node.inputs[j])
    }
    match node.op {
      If(then_branch~, else_branch~) => {
        let t_refs = then_branch.external_refs()
        for k = 0; k < t_refs.length(); k = k + 1 {
          record(t_refs[k])
        }
        let e_refs = else_branch.external_refs()
        for k = 0; k < e_refs.length(); k = k + 1 {
          record(e_refs[k])
        }
      }
      _ => ()
    }
  }
  for i = 0; i < self.outputs.length(); i = i + 1 {
    record(self.outputs[i])
  }
  refs
}
