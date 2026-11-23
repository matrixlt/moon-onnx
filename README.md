# moon-onnx

使用 MoonBit 开发的 ONNX 模型解析与跨端推理引擎（支持 `native` / `wasm` / `wasm-gc` / `js`）。

## 核心特性

- **官方 ONNX Protobuf 代码生成**：基于 `proto/onnx.proto` 与 `moonbitlang/protoc-gen-mbt` 生成完整编解码层，由 `src/model` 统一解码 `raw_data` 与类型化 repeated 权重。
- **强类型计算图 IR 与图优化（`src/ir` + `src/opt`）**：
  - 将弱类型 `NodeProto` Lowering 为强类型 `Op` ADT，执行输入引用校验与 Kahn 拓扑排序。
  - 内置 6 项图优化 Pass：Constant 提升、Identity 消除、Conv+BatchNorm 参数静态折叠、Conv+通道偏置 Add 折叠、静态 `Gemm(transB=1)` 权重预转置与死代码消除（DCE）。
- **双计算后端（Scalar + MoonBit `V128` SIMD）**：
  - `Scalar`：100% 纯 MoonBit 标量参考内核，与 ONNX Runtime 逐元素误差 `< 1.2e-6`。
  - `Simd`：基于 MoonBit 内建 `V128`（`f32x4`）的 `4x8` 寄存器分块 GEMM 微内核、`im2col + SIMD GEMM` 2D 卷积与向量化逐元素算子。
- **四端原生可编译**：全仓库通过 `native` / `wasm` / `wasm-gc` / `js` 四目标编译检查，核心张量、算子与 `V128` SIMD 测试在四端全部通过。

## 快速开始

```bash
# 1. 查看模型元信息、权重与计算图结构
moon run src/cli --target native -- inspect tests/fixtures/mnist_cnn.onnx

# 2. 执行 MNIST CNN 推理并与 ONNX Runtime 黄金输出对齐（可选 --backend=scalar|simd）
moon run src/cli --target native -- run \
  tests/fixtures/mnist_cnn.onnx \
  tests/fixtures/mnist_cnn_input.f32 \
  tests/fixtures/mnist_cnn_golden.f32 \
  --backend=simd

# 3. 运行提交前全套质量门禁（moon fmt + moon check + moon test）
./scripts/pre_commit_check.sh

# 4. 运行 Native Release 性能基准测试（Scalar vs V128 SIMD）
moon bench --target native --release -p local/moon_onnx/benchmarks
```

## Native Release 性能摘要（Scalar vs MoonBit `V128` SIMD）

| 基准项 | Scalar | MoonBit `V128` SIMD | 加速比 |
| :--- | ---: | ---: | ---: |
| `Add` (`n=65536`, `FixedArray[Byte]`) | 54.04 µs | **8.49 µs** | **6.37×** |
| `Relu` (`n=65536`, `FixedArray[Byte]`) | 55.29 µs | **7.23 µs** | **7.65×** |
| `GEMM 64×64×64` | 205.74 µs | **21.63 µs** | **9.51×** |
| `GEMM 128×128×128` | 1.48 ms | **164.09 µs** | **9.02×** |
| `GEMM 256×256×256` | 11.26 ms | **1.23 ms** | **9.15×** |
| E2E `MNIST MLP`（含 ORT 对齐 `1.19e-7`） | 44.13 µs | **34.70 µs** | **1.27×**（较优化前 6.48×） |
| E2E `MNIST CNN`（含 ORT 对齐 `1.19e-6`） | 722.90 µs (未优化) | **195.55 µs** | **3.70×** |

## 开发日志

完整开发历程、架构决策、踩坑记录与每阶段测试数据详见 [docs/DEVLOG.md](docs/DEVLOG.md)。
