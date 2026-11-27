# matrixlt/onnx

纯 MoonBit 编写的跨端 ONNX 动态推理引擎与算子库，组合了 `matrixlt/tensor`、`matrixlt/ml_kernels`、`matrixlt/onnx_proto` 与 `matrixlt/nn_ir`，无外部 C/C++ 运行时依赖；根包与 `session` 100% 纯内存、支持 `native` / `wasm` / `wasm-gc` / `js` 四端全量运行（静态 AOT 编译器 `onnx2mbt` 已独立为 `matrixlt/onnx_aot`）。

## 安装

```bash
moon add matrixlt/onnx
```

## 子包结构

- `matrixlt/onnx`（根包门面 `@onnx`，100% 纯内存）：重导出 `Session`、`BackendKind`、`RuntimeError`、`OpTiming`、`RunProfile`、`PoolStats`、`EvalContext`、`CpuScalarProvider`、`CpuSimdProvider` 与内存模型/会话入口（`load_model_from_bytes`、`session_from_bytes`、`max_abs_diff_f32`）。
- `matrixlt/onnx/fs`（Native 可选文件 I/O 隔离层）：封装 `load_model_from_file`、`session_from_file`、`load_f32_file`，与纯内存核心完全解耦。
- `matrixlt/onnx/frontend`：ONNX `OnnxModel` → `nn_ir::IrGraph` Lowering（覆盖 69 个强类型 IR 算子 / 72 个 ONNX 算子类型，兼容 Opset 6~18 语义、`If` 子图控制流与 `FLOAT16` / `INT8` / `UINT8` / 4-bit `MatMulNBits`）。
- `matrixlt/onnx/ops`：面向 `Tensor` 的 ONNX 算子库（按职责拆分为 `elementwise`、`linalg`、`nn`、`reduce`、`quant`、`rnn`、`layout_ops`、`logic_ops`、`shape_ops`、`llm_ops`、`vision`）。
- `matrixlt/onnx/session`（100% 纯内存四端通用）：`Session` 执行调度引擎：
  - 可插拔 `ExecutionProvider` trait（`prepare_graph` 钩子由 `CpuSimdProvider` 私有持有 Float32/Int8 权重预打包缓存，`EvalContext` 零后端泄漏）；
  - 别名感知的静态生命周期分析与 `BufferPool` 内存复用，内置 `run_profiled` 逐算子剖析。

## 快速示例

```moonbit
// 纯内存跨端动态推理（Native / Wasm / Wasm-GC / JS 通用）
let session = @onnx.session_from_bytes(model_bytes, backend=@session.Simd)
let (output, profile) = session.run_single_profiled(input_tensor)
println(profile.format_table())
```

> 如需将 `.onnx` 模型静态编译为零堆分配的独立 MoonBit 源码（`onnx2mbt`），请引入独立编译器模块 `matrixlt/onnx_aot`。

## 许可证

Apache-2.0
