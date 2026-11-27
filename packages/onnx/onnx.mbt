///|
/// `matrixlt/onnx` 根包门面（100% 纯内存、四端通用）：对外提供简洁稳定的 ONNX 推理引擎入口。
/// 本地文件系统读取入口请使用 `matrixlt/onnx/fs`，静态 AOT 编译（`onnx2mbt`）请使用独立模块 `matrixlt/onnx_aot`。
pub type Session = @session.Session

///|
/// 推理执行后端类型别名（`Scalar` / `Simd`）。
pub type BackendKind = @session.BackendKind

///|
/// 运行时错误类型别名（封装模型加载、IR、形状与算子执行错误）。
pub type RuntimeError = @session.RuntimeError

///|
/// 单算子性能剖析耗时记录类型别名。
pub type OpTiming = @session.OpTiming

///|
/// 完整单次前向推理性能剖析报告类型别名。
pub type RunProfile = @session.RunProfile

///|
/// 解析后的 ONNX 模型元信息与权重容器类型别名。
pub type OnnxModel = @onnx_proto.OnnxModel

///|
/// 多维张量核心类型别名。
pub type Tensor = @tensor.Tensor

///|
/// 内存池命中/未命中统计信息类型别名。
pub type PoolStats = @tensor.PoolStats

///|
/// 执行后端算子求值上下文类型别名。
pub type EvalContext = @session.EvalContext

///|
/// 标量 CPU 执行后端实现类型别名。
pub type CpuScalarProvider = @session.CpuScalarProvider

///|
/// V128 SIMD 预打包加速 CPU 执行后端实现类型别名。
pub type CpuSimdProvider = @session.CpuSimdProvider

///|
/// 返回当前引擎版本字符串（与工作区 `moon.mod` 版本保持一致）。
pub fn engine_version() -> String {
  "moon-onnx 0.2.0"
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
