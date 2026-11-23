# matrixlt/onnx

纯 MoonBit 编写的跨端 ONNX 推理引擎与统一门面库，组合了 `matrixlt/tensor`、`matrixlt/ml_kernels`、`matrixlt/onnx_proto` 与 `matrixlt/nn_ir`。

## 子包结构

- `matrixlt/onnx`（根包门面 `@onnx`）：重导出 `Session`、`BackendKind`、`RuntimeError`、模型加载与校验入口。
- `matrixlt/onnx/frontend`：ONNX `OnnxModel` → `nn_ir::IrGraph` Lowering（属性解析、Opset 语义适配与拓扑排序）。
- `matrixlt/onnx/ops`：面向 `Tensor` 的 ONNX 算子实现（逐元素/广播、激活、形状变换、MatMul/Gemm、Conv2D/Pool/BatchNorm），底层复用 `matrixlt/ml_kernels`。
- `matrixlt/onnx/session`：`Session` 执行调度引擎（支持 `Scalar` 与 `Simd` V128 后端切换及可选图优化）。
