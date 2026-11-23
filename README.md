# moon-onnx

使用纯 MoonBit 开发的 ONNX 模型解析与跨端推理引擎生态（支持 `native` / `wasm` / `wasm-gc` / `js`，内置 MoonBit `V128` SIMD 加速）。

仓库采用 MoonBit 官方 `moon.work` 多模块工作区组织，拆分为 **5 个严格分层、无循环依赖、可独立发布至 `mooncakes.io` 的开源库**：

## 模块总览

| 模块 | 路径 | 外部依赖 | 职责定位 |
| :--- | :--- | :--- | :--- |
| **`matrixlt/tensor`** | `packages/tensor` | **零依赖** | 面向推理与数值计算的多维张量底座（非装箱 `FixedArray`、零拷贝视图、多向广播） |
| **`matrixlt/ml_kernels`** | `packages/ml_kernels` | **零依赖** | 纯 MoonBit 原生 `V128` SIMD 数值内核（`4×8` 分块预打包 GEMM、`vecmath`、`im2col`） |
| **`matrixlt/onnx_proto`** | `packages/onnx_proto` | `protobuf`, `tensor` | 官方 `onnx.proto` 代码生成绑定、纯内存模型解析与权重解码（本地文件 I/O 隔离在 `/fs` 子包） |
| **`matrixlt/nn_ir`** | `packages/nn_ir` | `tensor` | 格式无关的强类型计算图 IR、Kahn 拓扑校验与分级图优化管线（常量提升 / Conv+BN / Conv+Bias / Gemm 预转置 / DCE） |
| **`matrixlt/onnx`** | `packages/onnx` | 上述 4 库 | 跨端 ONNX 推理引擎（ONNX Lowering、算子库、Scalar/`V128` SIMD `Session` 与 `@onnx` 统一门面） |

工作区应用与基准：
- `apps/cli`：Native 命令行工具（`inspect`、`run --backend=scalar|simd`）
- `benchmarks`：Scalar vs MoonBit `V128` SIMD 微基准与 MNIST 端到端基准套件

## 常用命令

```bash
# 全工作区格式化 + 四端检查（native/wasm/wasm-gc/js）+ 测试
./scripts/pre_commit_check.sh

# 查看 ONNX 模型结构摘要
moon run apps/cli --target native -- inspect tests/fixtures/mnist_cnn.onnx

# 运行 MNIST CNN 推理并与 ONNX Runtime 黄金输出对齐
moon run apps/cli --target native -- run \
  tests/fixtures/mnist_cnn.onnx \
  tests/fixtures/mnist_cnn_input.f32 \
  tests/fixtures/mnist_cnn_golden.f32 \
  --backend=simd

# 运行 Native Release 性能基准
moon bench --target native --release -p matrixlt/moon_onnx_benchmarks
```

## 架构文档

模块分层约束、依赖图与长期演进设计参见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
