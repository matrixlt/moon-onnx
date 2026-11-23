///|
/// 从本地文件路径读取并解析 `.onnx` 模型（与纯内存核心包解耦，避免 Wasm/JS 消费者引入文件系统依赖）。
pub fn load_model_from_file(
  path : String,
) -> @onnx_proto.OnnxModel raise @onnx_proto.ModelError {
  let bytes = @xfs.read_file_to_bytes(path) catch {
    @xfs.IOError(msg) =>
      raise @onnx_proto.IoError("failed to read file '\{path}': \{msg}")
  }
  @onnx_proto.load_model_from_bytes(bytes)
}
