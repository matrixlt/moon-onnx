# matrixlt/onnx

纯 MoonBit 编写的跨端 ONNX 推理引擎、算子库与 **AOT 静态模型编译器（`onnx2mbt`）**，组合了 `matrixlt/tensor`、`matrixlt/ml_kernels`、`matrixlt/onnx_proto` 与 `matrixlt/nn_ir`，无外部 C/C++ 运行时依赖，支持 `native` / `wasm` / `wasm-gc` / `js` 四端。

## 安装

```bash
moon add matrixlt/onnx
```

## 子包结构（v0.2.0）

- `matrixlt/onnx`（根包门面 `@onnx`）：重导出 `Session`、`BackendKind`、`RuntimeError`、`OpTiming`、`RunProfile`、`PoolStats`、`CodegenConfig`、`AotArtifact`、模型加载与 `compile_model_to_mbt` AOT 编译入口。
- `matrixlt/onnx/frontend`：ONNX `OnnxModel` → `nn_ir::IrGraph` Lowering（覆盖 45 个高频算子，兼容 Opset 6~18 语义与 `FLOAT16` / `INT8` / `UINT8`）。
- `matrixlt/onnx/ops`：面向 `Tensor` 的 ONNX 算子库：
  - 数学与激活：`Add`/`Sub`/`Mul`/`Div`/`Pow`/`Sqrt`/`Exp`/`Neg`/`Relu`/`Clip`/`Sigmoid`/`Tanh`/`Gelu`/`HardSwish`/`LeakyRelu`/`Softmax`
  - 视觉与序列：`Conv1D`/`Conv2D`/Depthwise（支持融合 `Relu`/`Clip`）、`MaxPool`/`AveragePool`/`GlobalAveragePool`、`Resize`（nearest/bilinear）、`BatchNormalization`、`LayerNormalization`、`LSTM`（单向/双向多输出）
  - 张量变换与量化：`MatMul`/`Gemm`、`Reshape`/`Flatten`/`Transpose`/`Squeeze`/`Unsqueeze`/`Shape`/`Cast`/`Concat`/`Split`/`Slice`/`Pad`/`Gather`/`ReduceMean`/`ReduceSum`、`QuantizeLinear`/`DequantizeLinear`/`QLinearMatMul`
- `matrixlt/onnx/session`：`Session` 执行调度引擎：
  - 可插拔 `ExecutionProvider` trait（内置 `CpuScalarProvider` 与 `CpuSimdProvider`，支持多输出 `eval_node_multi`）；
  - 初始化期自动预打包 `MatMul`、`Gemm` 与 `Conv1D`/`Conv2D` 静态权重；
  - 别名感知的静态生命周期分析与 `BufferPool` 内存复用，内置 `run_profiled` 逐算子剖析。
- `matrixlt/onnx/codegen`（**`onnx2mbt` v2**）：
  - 编译期静态形状推导与 **Static Memory Arena 区间规划**（推理热路径 0 堆分配、0 `Map`、0 `Tensor` 依赖）；
  - 支持 `--target=wasm|wasm-gc|native` 自适应代码生成、`--embed-weights` 单文件内嵌权重 Wasm 与 INT8 / Conv1D / Conv2D 模型 AOT 编译。

## 快速示例

```moonbit
// 1. 动态推理会话
let session = @onnx.session_from_bytes(model_bytes, backend=@session.Simd)
let (output, profile) = session.run_single_profiled(input_tensor)
println(profile.format_table())

// 2. AOT 静态编译为独立零分配 MoonBit 源码 + 权重包
let model = @onnx.load_model_from_bytes(model_bytes)
let artifact = @onnx.compile_model_to_mbt(
  model,
  Map([("x", [1, 1, 28, 28])]),
)
```

## 许可证

Apache-2.0
