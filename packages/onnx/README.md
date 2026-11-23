# matrixlt/onnx

纯 MoonBit 编写的跨端 ONNX 推理引擎与统一门面库，组合了 `matrixlt/tensor`、`matrixlt/ml_kernels`、`matrixlt/onnx_proto` 与 `matrixlt/nn_ir`，无外部 C/C++ 运行时依赖。

## 安装

```bash
moon add matrixlt/onnx
```

## 子包结构

- `matrixlt/onnx`（根包门面 `@onnx`）：重导出 `Session`、`BackendKind`、`RuntimeError`、`OpTiming`、`RunProfile`、`PoolStats`、模型加载与校验入口。
- `matrixlt/onnx/frontend`：ONNX `OnnxModel` → `nn_ir::IrGraph` Lowering（属性解析、Opset 语义适配与拓扑排序）。
- `matrixlt/onnx/ops`：面向 `Tensor` 的 ONNX 算子实现（逐元素/广播、激活、形状变换、`MatMul`/`Gemm`、`Conv2D`/`PackedConvWeight`/`MaxPool`/`AveragePool`/`BatchNorm`），底层复用 `matrixlt/ml_kernels` 与 `BufferPool`。
- `matrixlt/onnx/session`：`Session` 执行调度引擎：
  - 支持 `Scalar` 与 `Simd`（原生 V128）双后端切换；
  - 初始化期自动预打包 `MatMul`、`Gemm`（含 `trans_b=true`）与 `Conv` 静态权重；
  - 内置别名感知的静态生命周期分析与 `BufferPool` 内存复用（预热后中间激活与 `im2col` 工作区 100% 命中）；
  - 提供逐算子延迟剖析接口 `run_profiled` / `run_single_profiled` 与 `RunProfile::format_table()`。

## 快速示例

```moonbit
let session = @onnx.session_from_bytes(model_bytes, backend=@session.Simd)
let (output, profile) = session.run_single_profiled(input_tensor)
println(profile.format_table())
```

## 许可证

Apache-2.0
