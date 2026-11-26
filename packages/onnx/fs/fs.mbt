///|
/// 本地文件系统便捷入口（独立隔离在 `matrixlt/onnx/fs`，使 `matrixlt/onnx` 与 `matrixlt/onnx/session`
/// 保持 100% 纯内存、零 `moonbitlang/x/fs` 依赖，可无缝运行于浏览器与边缘 Wasm）。

///|
/// 从本地 `.onnx` 文件路径加载模型元信息与权重。
pub fn load_model_from_file(
  path : String,
) -> @onnx_proto.OnnxModel raise @onnx_proto.ModelError {
  @onnx_fs.load_model_from_file(path)
}

///|
/// 从本地 `.onnx` 文件路径直接构建推理会话。
pub fn session_from_file(
  path : String,
  backend~ : @session.BackendKind = @session.Scalar,
  optimize~ : Bool = true,
) -> @session.Session raise @session.RuntimeError {
  let m = @onnx_fs.load_model_from_file(path) catch {
    e => raise @session.ModelErr("\{e}")
  }
  @session.Session::from_model(m, backend~, optimize~)
}

///|
/// 从原始小端序 float32 二进制文件（`.f32`）加载张量。
pub fn load_f32_file(
  path : String,
  shape : Array[Int32],
) -> @tensor.Tensor raise @session.RuntimeError {
  let bytes = @xfs.read_file_to_bytes(path) catch {
    @xfs.IOError(msg) =>
      raise @session.ModelErr("failed to read '\{path}': \{msg}")
  }
  @tensor.Tensor::from_f32_bytes_le(shape, bytes) catch {
    e => raise @session.TensorErr("file '\{path}': \{e}")
  }
}
