# moon-onnx

使用纯 MoonBit 开发的 ONNX 模型解析、跨端推理引擎与 **AOT 静态模型编译器（`onnx2mbt`）**，单一代码库原生支持 `native` / `wasm` / `wasm-gc` / `js` 四端，内置 MoonBit `V128` SIMD（`f32x4` 与 INT8 `i32x4.dot_i16x8_s`）加速，**零 C/C++ FFI 依赖**。

仓库采用 MoonBit 官方 `moon.work` 多模块工作区组织，拆分为 **5 个严格分层、无循环依赖、可独立发布至 `mooncakes.io` 的开源库**：

## 模块总览（v0.2.0）

| 模块 | 路径 | 外部依赖 | 职责定位 |
| :--- | :--- | :--- | :--- |
| **`matrixlt/tensor`** | `packages/tensor` | **零依赖** | 多维张量底座（`Float32`/`Float64`/`Int64`/`Int32`/`Bool`/`UInt8`/`Int8`、零拷贝视图、多向广播、`BufferPool`） |
| **`matrixlt/ml_kernels`** | `packages/ml_kernels` | **零依赖** | 纯 MoonBit 原生 `V128` SIMD 数值内核（预打包 `4×8` GEMM、零 `im2col` Depthwise Conv2D、`vecmath`、INT8 `i32x4.dot` 量化矩阵乘） |
| **`matrixlt/onnx_proto`** | `packages/onnx_proto` | `protobuf`, `tensor` | 官方 `onnx.proto` 绑定、纯内存模型解析与权重解码（支持 `FLOAT32`/`FLOAT16`/`INT8`/`UINT8`/`INT64` 等，文件 I/O 隔离在 `/fs`） |
| **`matrixlt/nn_ir`** | `packages/nn_ir` | `tensor` | 强类型计算图 IR（45 算子）、Kahn 拓扑排序与图优化管线（常量提升 / `Conv+BN`（含 1D/2D）/ `Conv+Bias` / `Conv+Relu` / `Conv+Clip(ReLU6)` / `Gemm` 预转置 / DCE） |
| **`matrixlt/onnx`** | `packages/onnx` | 上述 4 库 | 跨端 ONNX 推理引擎（`frontend`、`ops`、可插拔 `ExecutionProvider` `Session` 与 `codegen` AOT 编译器 `onnx2mbt`） |

工作区应用与基准：
- `apps/cli`：Native 命令行工具（`inspect`、`run`、`profile`、`compile`）
- `apps/wasm_bench` & `scripts/bench_wasm.mjs`：纯内存 Wasm / Wasm-GC `V128` SIMD 算子基准
- `scripts/bench_aot_wasm.mjs`：真实 MNIST CNN 的 `onnx2mbt` AOT 5 模式端到端 Wasm 体积与性能基准
- `benchmarks`：Native Release 微基准与 MNIST 端到端基准

## 核心亮点与实测数据（v0.2.0）

- **45 个高频 ONNX 算子**：覆盖视觉（`Conv1D`/`Conv2D`/Depthwise、`MaxPool`/`AveragePool`/`GlobalAveragePool`、`Resize`、`BatchNormalization`）、语音/时序（多输出 `LSTM`）、Transformer（`LayerNormalization`、`Gelu`、`HardSwish`、`MatMul`/`Gemm`、`Softmax`）、张量操作（`Split`/`Concat`/`Slice`/`Pad`/`Gather`/`Cast`/`Shape`/`ReduceMean`/`ReduceSum`）与量化（`QuantizeLinear`/`DequantizeLinear`/`QLinearMatMul`）。
- **`onnx2mbt` v2 零分配 AOT 静态编译器**：编译期完成图融合 + 静态内存 Arena 区间规划 + 目标平台自适应 + 可选 `--embed-weights` 单文件内嵌权重。真实 `mnist_cnn.onnx`（与 ORT Golden 误差 `5.36e-7`）在 Node.js V8 Wasm 中实测：
  - `wasm-gc` 外置权重：**3.2 KB raw / 1.8 KB gzip**，**641 FPS**，单次推理 **0 堆分配**；
  - `wasm-gc` 单文件内嵌全量 24 KB 权重：**26.8 KB raw / 24.1 KB gzip**，冷启动 **0.77 ms**，**614 FPS**；
  - `wasm` V128 SIMD 外置权重：**16.6 KB raw / 8.7 KB gzip**，**534 FPS**。
- **Node.js V8 Wasm `V128` SIMD 算子实测加速**：`GEMM 128³` **19.4×**，`INT8 QLinearMatMul 128³ (i32x4.dot)` **25.4×**，`Conv2D + Relu` **10.9×**，`Relu 65K` **37.8×**。

## 常用命令

```bash
# 1. 全工作区格式化 + 四端检查（native/wasm/wasm-gc/js）+ 测试
./scripts/pre_commit_check.sh

# 2. 查看 ONNX 模型结构摘要
moon run apps/cli --target native -- inspect tests/fixtures/mnist_cnn.onnx

# 3. 运行 MNIST CNN 推理并与 ONNX Runtime 黄金输出对齐
moon run apps/cli --target native -- run \
  tests/fixtures/mnist_cnn.onnx \
  tests/fixtures/mnist_cnn_input.f32 \
  tests/fixtures/mnist_cnn_golden.f32 \
  --backend=simd

# 4. 使用 onnx2mbt AOT 编译为单文件零分配 MoonBit/Wasm 推理模块
moon run apps/cli --target native -- compile \
  tests/fixtures/mnist_cnn.onnx ./out_mnist_aot \
  --input-shape=1,1,28,28 --struct=MnistCnn --target=wasm --embed-weights

# 5. 运行 Wasm V128 算子基准与 AOT 端到端基准
node scripts/bench_wasm.mjs
node scripts/bench_aot_wasm.mjs
```

## 架构文档

模块分层约束、依赖图与长期演进设计参见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
