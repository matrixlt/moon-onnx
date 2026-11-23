# matrixlt/onnx_proto

纯 MoonBit 的官方 ONNX 模型解析、权重解码与结构检视库，不绑定任何推理运行时，可供模型检查器、可视化工具、格式转换器与推理引擎独立复用。

## 安装

```bash
moon add matrixlt/onnx_proto
```

## 子包结构

- `matrixlt/onnx_proto`（根包，**纯内存、零文件系统依赖，四端通用**）：
  - `load_model_from_bytes(Bytes) -> OnnxModel`
  - `load_model_from_proto(@schema.ModelProto) -> OnnxModel`
  - `decode_tensor_proto(TensorProto) -> @tensor.Tensor`（支持小端序 `raw_data` 及类型化 repeated 字段）
  - `OnnxModel::format_summary()`（人类可读模型结构摘要）
- `matrixlt/onnx_proto/schema`：通过 `protoc` + `moonbitlang/protoc-gen-mbt` 从官方 `onnx.proto` 自动生成的完整消息与二进制编解码器。
- `matrixlt/onnx_proto/fs`：封装本地 `.onnx` 文件读取（`load_model_from_file(path)`），与纯内存根包解耦。

## 快速示例

```moonbit
let model = @onnx_proto.load_model_from_bytes(onnx_bytes)
println(model.format_summary())
```

## 许可证

Apache-2.0
