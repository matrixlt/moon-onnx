///|
/// `matrixlt/onnx` 根包门面：对外提供简洁稳定的推理引擎入口。
pub type Session = @session.Session

///|
pub type BackendKind = @session.BackendKind

///|
pub type RuntimeError = @session.RuntimeError

///|
pub type OpTiming = @session.OpTiming

///|
pub type RunProfile = @session.RunProfile

///|
pub type OnnxModel = @onnx_proto.OnnxModel

///|
pub type Tensor = @tensor.Tensor

///|
pub type PoolStats = @tensor.PoolStats

///|
/// 返回当前引擎版本字符串。
pub fn engine_version() -> String {
  "moon-onnx 0.1.0"
}

///|
/// 从内存 `.onnx` 字节流加载模型元信息与权重。
pub fn load_model_from_bytes(
  bytes : Bytes,
) -> OnnxModel raise @onnx_proto.ModelError {
  @onnx_proto.load_model_from_bytes(bytes)
}

///|
/// 从本地文件路径加载 `.onnx` 模型。
pub fn load_model_from_file(
  path : String,
) -> OnnxModel raise @onnx_proto.ModelError {
  @onnx_fs.load_model_from_file(path)
}

///|
/// 从本地 `.onnx` 文件直接创建推理会话。
pub fn session_from_file(
  path : String,
  backend~ : BackendKind = @session.Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  @session.Session::from_file(path, backend~, optimize~)
}

///|
/// 从内存 `.onnx` 字节流直接创建推理会话（全平台通用）。
pub fn session_from_bytes(
  bytes : Bytes,
  backend~ : BackendKind = @session.Scalar,
  optimize~ : Bool = true,
) -> Session raise RuntimeError {
  @session.Session::from_bytes(bytes, backend~, optimize~)
}

///|
/// 从小端序 `.f32` 文件读取张量。
pub fn load_f32_file(
  path : String,
  shape : Array[Int32],
) -> Tensor raise RuntimeError {
  @session.load_f32_file(path, shape)
}

///|
/// 计算两个 Float32 张量之间的最大绝对误差。
pub fn max_abs_diff_f32(
  actual : Tensor,
  expected : Tensor,
) -> Float raise RuntimeError {
  @session.max_abs_diff_f32(actual, expected)
}
