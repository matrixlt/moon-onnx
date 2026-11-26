///|
/// `matrixlt/onnx` 根包门面（100% 纯内存、四端通用）：对外提供简洁稳定的 ONNX 推理引擎入口。
/// 本地文件系统读取入口请使用 `matrixlt/onnx/fs`，静态 AOT 编译（`onnx2mbt`）请使用独立模块 `matrixlt/onnx_aot`。
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
pub type EvalContext = @session.EvalContext

///|
pub type CpuScalarProvider = @session.CpuScalarProvider

///|
pub type CpuSimdProvider = @session.CpuSimdProvider

///|
/// 返回当前引擎版本字符串。
pub fn engine_version() -> String {
  "moon-onnx 0.4.0"
}

///|
/// 从内存 `.onnx` 字节流加载模型元信息与权重（全平台通用）。
pub fn load_model_from_bytes(
  bytes : Bytes,
) -> OnnxModel raise @onnx_proto.ModelError {
  @onnx_proto.load_model_from_bytes(bytes)
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
/// 计算两个 Float32 张量之间的最大绝对误差。
pub fn max_abs_diff_f32(
  actual : Tensor,
  expected : Tensor,
) -> Float raise RuntimeError {
  @session.max_abs_diff_f32(actual, expected)
}
