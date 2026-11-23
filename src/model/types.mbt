///|
/// 模型加载与解析阶段可能抛出的结构化错误。
#derive(Debug, Eq)
pub(all) suberror ModelError {
  /// 文件系统读取失败
  IoError(String)
  /// Protobuf 二进制反序列化失败
  ParseError(String)
  /// 遇到尚未实现的 ONNX TensorProto.DataType 枚举值
  UnsupportedDataType(Int32)
  /// TensorProto 字段不完整或字节长度与 dims 不一致
  InvalidTensor(String)
  /// ModelProto 缺少必需的 GraphProto
  MissingGraph(String)
}

///|
/// 为 ModelError 提供清晰可读的错误描述。
pub impl Show for ModelError with fn output(self, logger) {
  match self {
    IoError(msg) => logger.write_string("IoError(\{msg})")
    ParseError(msg) => logger.write_string("ParseError(\{msg})")
    UnsupportedDataType(code) =>
      logger.write_string("UnsupportedDataType(onnx_dtype=\{code})")
    InvalidTensor(msg) => logger.write_string("InvalidTensor(\{msg})")
    MissingGraph(msg) => logger.write_string("MissingGraph(\{msg})")
  }
}

///|
/// ONNX 张量输入/输出的维度规格（支持静态整数维度、符号动态维度与未知维度）。
#derive(Debug, Eq)
pub(all) enum DimSpec {
  Static(Int32)
  Symbolic(String)
  Unknown
}

///|
/// 将单个维度规格格式化为简洁字符串。
pub fn DimSpec::to_label(self : DimSpec) -> String {
  match self {
    Static(n) => n.to_string()
    Symbolic(name) => name
    Unknown => "?"
  }
}

///|
/// 计算图的输入或输出签名（名称、数据类型与维度规格）。
pub struct ValueSpec {
  name : String
  dtype : @tensor.DType?
  shape : Array[DimSpec]
}

///|
/// 将 ValueSpec 格式化为类似 `input: f32[1, 1, 28, 28]` 的可读签名。
pub fn ValueSpec::format(self : ValueSpec) -> String {
  let dtype_str = match self.dtype {
    Some(dt) => dt.short_name()
    None => "unknown"
  }
  let buf = StringBuilder()
  buf.write_string(self.name)
  buf.write_string(": ")
  buf.write_string(dtype_str)
  buf.write_char('[')
  for i = 0; i < self.shape.length(); i = i + 1 {
    if i > 0 {
      buf.write_string(", ")
    }
    buf.write_string(self.shape[i].to_label())
  }
  buf.write_char(']')
  buf.to_string()
}

///|
/// 模型声明的 Opset 版本信息。
pub struct OpsetImport {
  domain : String
  version : Int64
}

///|
/// 解析并解码完成后的 ONNX 模型对象。
/// 隔离了底层自动生成的 Protobuf 细节，向上层 IR 与推理引擎提供干净接口。
pub struct OnnxModel {
  ir_version : Int64
  producer_name : String
  producer_version : String
  opsets : Array[OpsetImport]
  graph_name : String
  /// 计算图的真实运行期输入（已自动排除同时出现在 initializer 中的权重项）
  inputs : Array[ValueSpec]
  /// 计算图的输出签名
  outputs : Array[ValueSpec]
  /// 已解码为 Tensor 的全部权重初始化张量（按名称索引）
  initializers : Map[String, @tensor.Tensor]
  /// 保持初始权重在模型文件中的原始顺序，便于稳定输出摘要
  initializer_order : Array[String]
  /// 保留原始 NodeProto 列表，交由 src/ir 进一步 Lowering 为强类型算子图
  nodes : Array[@proto.NodeProto]
}

///|
/// 返回模型主域（空字符串或 "ai.onnx"）的 Opset 版本号；若未声明则返回 None。
pub fn OnnxModel::default_opset_version(self : OnnxModel) -> Int64? {
  for i = 0; i < self.opsets.length(); i = i + 1 {
    let op = self.opsets[i]
    if op.domain == "" || op.domain == "ai.onnx" {
      return Some(op.version)
    }
  }
  None
}
