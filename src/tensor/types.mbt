///|
/// 张量运算与构建过程中可能抛出的结构化错误类型。
#derive(Debug, Eq)
pub(all) suberror TensorError {
  /// 形状非法（例如包含负数维度或元素总数溢出）
  InvalidShape(String)
  /// 张量形状不兼容（例如广播或矩阵乘法维度不匹配）
  ShapeMismatch(String)
  /// 数据类型不匹配（例如期望 Float32 但实际为 Int64）
  DTypeMismatch(String)
  /// 索引越界
  IndexOutOfBounds(String)
  /// 轴参数非法（例如轴超出范围或重复）
  InvalidAxis(String)
  /// 尚未支持的张量类型或操作
  Unsupported(String)
}

///|
/// 为 TensorError 提供人类可读的错误文本。
pub impl Show for TensorError with fn output(self, logger) {
  match self {
    InvalidShape(msg) => logger.write_string("InvalidShape(\{msg})")
    ShapeMismatch(msg) => logger.write_string("ShapeMismatch(\{msg})")
    DTypeMismatch(msg) => logger.write_string("DTypeMismatch(\{msg})")
    IndexOutOfBounds(msg) => logger.write_string("IndexOutOfBounds(\{msg})")
    InvalidAxis(msg) => logger.write_string("InvalidAxis(\{msg})")
    Unsupported(msg) => logger.write_string("Unsupported(\{msg})")
  }
}

///|
/// 支持的张量基础数据类型，与 ONNX TensorProto.DataType 常用子集一一对应。
#derive(Debug, Eq)
pub(all) enum DType {
  Float32
  Float64
  Int64
  Int32
  Bool
}

///|
/// 为 DType 实现人类可读输出。
pub impl Show for DType with fn output(self, logger) {
  logger.write_string(self.short_name())
}

///|
/// 将整数形状数组格式化为 `[d0, d1, ...]` 字符串，供日志、摘要与错误信息使用。
pub fn format_shape(shape : Array[Int32]) -> String {
  let buf = StringBuilder()
  buf.write_char('[')
  for i = 0; i < shape.length(); i = i + 1 {
    if i > 0 {
      buf.write_string(", ")
    }
    buf.write_string(shape[i].to_string())
  }
  buf.write_char(']')
  buf.to_string()
}

///|
/// 返回单个元素占用的字节数，便于内存估算与 raw_data 解码校验。
pub fn DType::byte_size(self : DType) -> Int32 {
  match self {
    Float32 => 4
    Float64 => 8
    Int64 => 8
    Int32 => 4
    Bool => 1
  }
}

///|
/// 返回面向日志和 CLI 展示的简短类型名称。
pub fn DType::short_name(self : DType) -> String {
  match self {
    Float32 => "f32"
    Float64 => "f64"
    Int64 => "i64"
    Int32 => "i32"
    Bool => "bool"
  }
}

///|
/// 张量底层紧凑存储。
/// 使用非装箱的 FixedArray 以减少 GC 对象开销，并为后续 SIMD 分块加载奠定基础。
pub(all) enum TensorData {
  F32(FixedArray[Float])
  F64(FixedArray[Double])
  I64(FixedArray[Int64])
  I32(FixedArray[Int32])
  Bool(FixedArray[Bool])
}

///|
/// 返回底层存储缓冲区的元素总长度。
pub fn TensorData::len(self : TensorData) -> Int32 {
  match self {
    F32(buf) => buf.length()
    F64(buf) => buf.length()
    I64(buf) => buf.length()
    I32(buf) => buf.length()
    Bool(buf) => buf.length()
  }
}

///|
/// 多维张量核心结构。
/// 采用 (data, offset, shape, strides) 视图模型，使 Transpose / Squeeze / Reshape（连续时）
/// 能够在不复制底层缓冲的前提下零拷贝完成。
pub struct Tensor {
  data : TensorData
  offset : Int32
  shape : Array[Int32]
  strides : Array[Int32]
  dtype : DType
}

///|
/// 校验形状合法性并计算元素总数（空形状 `[]` 表示 0 维标量，元素数为 1）。
pub fn num_elements(shape : Array[Int32]) -> Int32 raise TensorError {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    let dim = shape[i]
    if dim < 0 {
      raise InvalidShape(
        "dimension at axis \{i} must be non-negative, got \{dim}",
      )
    }
    total = total * dim
  }
  total
}

///|
/// 按 C 风格行主序（Row-Major）计算给定形状的默认步长。
/// 例如 shape = [2, 3, 4] 对应 strides = [12, 4, 1]。
pub fn compute_strides(shape : Array[Int32]) -> Array[Int32] {
  let rank = shape.length()
  if rank == 0 {
    return []
  }
  let strides = Array::make(rank, 1)
  let mut acc = 1
  for i = rank - 1; i >= 0; i = i - 1 {
    strides[i] = acc
    acc = acc * shape[i]
  }
  strides
}

///|
/// 从紧凑的 FixedArray[Float] 构造 Float32 张量。
pub fn Tensor::from_f32(
  shape : Array[Int32],
  data : FixedArray[Float],
) -> Tensor raise TensorError {
  let expected = num_elements(shape)
  if data.length() != expected {
    raise ShapeMismatch(
      "from_f32: shape \{format_shape(shape)} requires \{expected} elements, got \{data.length()}",
    )
  }
  {
    data: F32(data),
    offset: 0,
    shape: shape.copy(),
    strides: compute_strides(shape),
    dtype: Float32,
  }
}

///|
/// 从动态 Array[Float] 构造 Float32 张量（便于测试与外部数据导入）。
pub fn Tensor::from_f32_array(
  shape : Array[Int32],
  values : Array[Float],
) -> Tensor raise TensorError {
  let buf = FixedArray::makei(values.length(), fn(i) { values[i] })
  Tensor::from_f32(shape, buf)
}

///|
/// 从 FixedArray[Double] 构造 Float64 张量。
pub fn Tensor::from_f64(
  shape : Array[Int32],
  data : FixedArray[Double],
) -> Tensor raise TensorError {
  let expected = num_elements(shape)
  if data.length() != expected {
    raise ShapeMismatch(
      "from_f64: shape \{format_shape(shape)} requires \{expected} elements, got \{data.length()}",
    )
  }
  {
    data: F64(data),
    offset: 0,
    shape: shape.copy(),
    strides: compute_strides(shape),
    dtype: Float64,
  }
}

///|
/// 从 FixedArray[Int64] 构造 Int64 张量（常用于 ONNX Reshape 的 shape 输入及索引）。
pub fn Tensor::from_i64(
  shape : Array[Int32],
  data : FixedArray[Int64],
) -> Tensor raise TensorError {
  let expected = num_elements(shape)
  if data.length() != expected {
    raise ShapeMismatch(
      "from_i64: shape \{format_shape(shape)} requires \{expected} elements, got \{data.length()}",
    )
  }
  {
    data: I64(data),
    offset: 0,
    shape: shape.copy(),
    strides: compute_strides(shape),
    dtype: Int64,
  }
}

///|
/// 从 FixedArray[Int] 构造 Int32 张量。
pub fn Tensor::from_i32(
  shape : Array[Int32],
  data : FixedArray[Int32],
) -> Tensor raise TensorError {
  let expected = num_elements(shape)
  if data.length() != expected {
    raise ShapeMismatch(
      "from_i32: shape \{format_shape(shape)} requires \{expected} elements, got \{data.length()}",
    )
  }
  {
    data: I32(data),
    offset: 0,
    shape: shape.copy(),
    strides: compute_strides(shape),
    dtype: Int32,
  }
}

///|
/// 从 FixedArray[Bool] 构造 Bool 张量。
pub fn Tensor::from_bool(
  shape : Array[Int32],
  data : FixedArray[Bool],
) -> Tensor raise TensorError {
  let expected = num_elements(shape)
  if data.length() != expected {
    raise ShapeMismatch(
      "from_bool: shape \{format_shape(shape)} requires \{expected} elements, got \{data.length()}",
    )
  }
  {
    data: Bool(data),
    offset: 0,
    shape: shape.copy(),
    strides: compute_strides(shape),
    dtype: Bool,
  }
}

///|
/// 创建全 0 初始化的 Float32 张量。
pub fn Tensor::zeros_f32(shape : Array[Int32]) -> Tensor raise TensorError {
  let total = num_elements(shape)
  Tensor::from_f32(shape, FixedArray::make(total, (0.0 : Float)))
}

///|
/// 创建指定常量填充的 Float32 张量。
pub fn Tensor::full_f32(
  shape : Array[Int32],
  value : Float,
) -> Tensor raise TensorError {
  let total = num_elements(shape)
  Tensor::from_f32(shape, FixedArray::make(total, value))
}

///|
/// 返回张量维度阶数（Rank）。
pub fn Tensor::rank(self : Tensor) -> Int32 {
  self.shape.length()
}

///|
/// 返回张量逻辑元素总数。
pub fn Tensor::numel(self : Tensor) -> Int32 {
  let mut total = 1
  for i = 0; i < self.shape.length(); i = i + 1 {
    total = total * self.shape[i]
  }
  total
}

///|
/// 返回张量形状的拷贝（防止外部直接修改破坏内部不变量）。
pub fn Tensor::shape(self : Tensor) -> Array[Int32] {
  self.shape.copy()
}

///|
/// 返回张量步长的拷贝。
pub fn Tensor::strides(self : Tensor) -> Array[Int32] {
  self.strides.copy()
}

///|
/// 返回张量数据类型。
pub fn Tensor::dtype(self : Tensor) -> DType {
  self.dtype
}

///|
/// 返回张量在底层缓冲区中的起始偏移。
pub fn Tensor::offset(self : Tensor) -> Int32 {
  self.offset
}
