# moon-onnx

使用纯 MoonBit 开发的 ONNX 模型解析、跨端推理引擎与 **AOT 静态模型编译器（`onnx2mbt`）**，单一代码库原生支持 `native` / `wasm` / `wasm-gc` / `js` 四端，内置 MoonBit `V128` SIMD（`f32x4` 与 INT8 `i32x4.dot_i16x8_s`）加速，**零 C/C++ FFI 依赖**。

仓库采用 MoonBit 官方 `moon.work` 多模块工作区组织，拆分为 **6 个严格分层、无循环依赖、可独立发布至 `mooncakes.io` 的开源库**：

## 模块总览

| 模块 | 路径 | 外部依赖 | 职责定位 |
| :--- | :--- | :--- | :--- |
| **`matrixlt/tensor`** | `packages/tensor` | **零依赖** | 多维张量底座（`Float32`/`Float64`/`Int64`/`Int32`/`Bool`/`UInt8`/`Int8`、`SymDim`、零拷贝视图、多向广播、`BufferPool`） |
| **`matrixlt/ml_kernels`** | `packages/ml_kernels` | **零依赖** | 纯 MoonBit 原生 `V128` SIMD 与零分配内核集合（入口 O(1) 边界校验 + 内层 `unsafe_get`/`unsafe_set`、`/gemm` BLIS `6×8` K-unroll-2 + Relu/Clip/SiLU 融合、`/conv` 16×16 分块 1×1 / 全展开 3×3 与零 `im2col` Depthwise、`/vecmath`、`/quant` INT8 `i32x4.dot` + Wasm-GC 4 列展开 + 4-bit `MatMulNBits`、`/nn` 零分配 LLM/视觉/语音内核） |
| **`matrixlt/onnx_proto`** | `packages/onnx_proto` | `protobuf`, `tensor` | 官方 `onnx.proto` 绑定、纯内存模型解析与权重解码（文件 I/O 严格隔离在 `/fs`） |
| **`matrixlt/nn_ir`** | `packages/nn_ir` | `tensor` | 强类型计算图 IR（69 算子）、Kahn 拓扑排序、分级图优化 Pass（`/opt`）与静态形状/类型推断单一真相源（`/analysis`） |
| **`matrixlt/onnx`** | `packages/onnx` | 上述 4 库 | 纯内存跨端动态推理引擎（`/frontend`、`/ops`、纯内存四端通用 `/session` + 私有权重缓存 `ExecutionProvider`，文件 I/O 严格隔离在 `/fs`） |
| **`matrixlt/onnx_aot`** | `packages/onnx_aot` | `tensor`, `onnx_proto`, `nn_ir`, `onnx` | 独立 AOT 静态编译器（`onnx2mbt` v2，纯内存四端通用）：图特化、Static Arena 区间规划与零堆分配 Wasm/Native 代码生成 |

工作区应用与基准：
- `apps/cli`：Native 命令行工具（`inspect`、`run`、`profile`、`compile`）
- `apps/wasm_bench` & `scripts/bench_wasm.mjs`：纯内存 Wasm / Wasm-GC `V128` SIMD 算子基准
- `scripts/bench_aot_wasm.mjs`：真实 MNIST CNN 的 `onnx2mbt` AOT 5 模式端到端 Wasm 体积与性能基准
- `scripts/bench_real_models.mjs`：真实开源生产模型（Silero VAD v5 + MobileNetV2-12 + YOLOv8n 640×640 + all-MiniLM-L6-v2 FP32/INT8）的 `onnx2mbt` 零分配 Wasm AOT 精度对齐（vs ORT）与性能基准
- `scripts/bench_whisper.mjs`：真实 OpenAI Whisper-tiny（Encoder FP32 + Decoder INT8 Prefill + Decoder INT8 KV-Cache Step）全链路零分配 Wasm AOT 精度对齐（vs ORT）与性能基准
- `benchmarks`：Native Release 微基准与 MNIST 端到端基准

## 核心亮点与实测数据

- **69 个高频 ONNX 算子（覆盖生成式 Q4 LLM、视觉检测 YOLOv8、端侧语音 Whisper 与控制流/动态量化）**：
  - **生成式 Q4 LLM 套件**：`MatMulNBits`（4-bit 奇偶拆分 4 列展开 V128 内核）、`RMSNormalization` / `SkipSimplifiedLayerNormalization`、`RotaryEmbedding`、`GroupQueryAttention`（流式在线 Softmax + KV-Cache）、`Silu` 自动融合与零拷贝 `A @ B^T` `lm_head`；
  - **视觉检测与分类**：`Conv1D`/`Conv2D`/Depthwise（含 `16×16` 分块 `1×1` / 全展开 `3×3` `im2col` 与寄存器级 `Conv+Relu/Clip/SiLU` 融合）、`MaxPool`/`AveragePool`/`GlobalAveragePool`、`Resize`（支持 Arena AOT）、`NonMaxSuppression`、`BatchNormalization`；
  - **语音 ASR / 流式 VAD 与 Transformer**：多输出 `LSTM`、`If` 条件分支子图、`LayerNormalization`（9 节点自动融合）、`GeluExact`（5 节点 Erf-GELU 自动融合）、`Erf`/`Gelu`/`HardSwish`、静态/动态 INT8 量化族（`QuantizeLinear`/`DequantizeLinear`/`QLinearMatMul`/`DynamicQuantizeLinear`/`MatMulInteger`）与完整逻辑/索引算子。
- **真实开源生产模型端到端打通 + 实测数据（MoonBit 1.0 RC1，单线程 Node.js V8 Wasm，对比官方 ONNX Runtime 1.32.0）**：

| 真实开源模型 | 运行模式 / 目标 | 产物体积（raw / gzip） | 静态 Arena RAM | 误差（vs ORT） | 实测速度 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SmolLM2-135M-Instruct Q4** (30 层, 174 MB) | Native V128 `Session` (Q4 + KV-Cache) | 纯 MoonBit 零 FFI | 零拷贝 `lm_head` | **`5.44e-5`**（Token 100% 一致） | **107.0 ms/tok（9.4 tok/s）** |
| **YOLOv8n COCO 80-class** (640×640, 12.2 MB) | `wasm-gc` Arena (`Conv+SiLU` 融合) | **25.6 KB / 9.8 KB** | 28.1 MB（Arena 省 92.7%） | **`box 1.9e-3px / cls 2.8e-7`** | 5590.9 ms（Native 1498 ms） |
| **YOLOv8n COCO 80-class** (640×640, 12.2 MB) | `wasm` V128 Arena (`6×8` GEMM + `Conv+SiLU`) | 55.5 KB / 27.1 KB | 28.1 MB（Arena 省 92.7%） | **`box 1.9e-3px / cls 2.5e-7`** | **1276.7 ms（0.78 FPS，4.6× 提速）** |
| **Whisper-tiny Encoder FP32** (30s 音频, 32 MB) | `wasm-gc` / `wasm` V128 Arena | **21.2 KB / 8.5 KB** (`52.4 KB`) | 114.0 MB（省 85%） | **`1.87e-3`**（576K 输出） | **7.80 s（3.85× 实时，3.3× 提速）** |
| **Whisper-tiny Decoder INT8 Prefill** (Cross+Self Attn) | `wasm` V128 Arena | 91.0 KB / 45.0 KB | 20.5 MB（省 73%） | `tok1=542`（与 ORT 一致） | **712.0 ms** |
| **Whisper-tiny Decoder INT8 KV-Cache Step** | `wasm-gc` / `wasm` V128 Arena | **36.3 KB / 13.2 KB** (`80.7 KB`) | **2.29 MB（省 76.5%）** | `tok2=33`（与 ORT 一致） | **72.0–72.6 ms/tok（13.9 tok/s）** |
| **Silero VAD v5** (16kHz 流式) | `wasm-gc` / `wasm` V128 Arena | **13.6 KB / 6.3 KB** (`38.4 KB`) | **13.8 KB** | `1.17e-7` | **0.40 ms/帧（80.9× 实时）** |
| **MobileNetV2-12** (3.49M, 224²) | `wasm-gc` / `wasm` V128 Arena | **20.0 KB / 8.4 KB** (`45.6 KB`) | 7.64 MB（省 78%） | `1.06e-5`（Top-1 一致） | **147.7 ms（6.8 FPS，4.4× 提速）** |
| **all-MiniLM-L6-v2 FP32** (89.5 MB) | `wasm-gc` / `wasm` V128 Arena | **24.4 KB / 9.2 KB** (`53.7 KB`) | **216.1 KB（省 97.0%）** | `4.05e-6` | **45.7 ms（21.9 emb/s，4.4× 提速）** |
| **all-MiniLM-L6-v2 INT8** (**21.5 MB**, 4× 压缩) | `wasm-gc` / `wasm` V128 `i32x4.dot` | **37.5 KB / 13.4 KB** (`83.1 KB`) | **246.1 KB（省 97.4%）** | **`1.07e-6`** | **69.4 ms（14.4 emb/s）** |
| **MNIST CNN** (外置 / 单文件内嵌) | `wasm-gc` Arena | **3.2 KB / 1.8 KB**（内嵌 26.8 KB） | 12.9 KB | `8.34e-7` | **651–654 FPS** |

- **Node.js V8 Wasm `V128` 微基准实测（`6×8` K-unroll-2 + O(1) 边界校验零检查热路径）**：`GEMM 64³` **47.1 µs**、`GEMM 128³` **390.8 µs**（含在线 Pack，约 10.7 GFLOP/s）、`INT8 QLinearMatMul 128³ (i32x4.dot)` **1295.8 µs (22.7×)**、`Conv2D + Relu` **213.6 µs**、`Relu 65K` **28.4 µs (36.6×)**。

## 常用命令

```bash
# 1. 全工作区格式化 + 四端检查（native/wasm/wasm-gc/js）+ 测试
./scripts/pre_commit_check.sh

# 2. 查看 ONNX 模型结构摘要
moon run apps/cli --target native -- inspect tests/fixtures/mnist_cnn.onnx

# 3. 运行推理并与 ONNX Runtime 黄金输出对齐（支持自定义 --input-shape）
moon run apps/cli --target native -- run \
  tests/fixtures/mnist_cnn.onnx \
  tests/fixtures/mnist_cnn_input.f32 \
  tests/fixtures/mnist_cnn_golden.f32 \
  --input-shape=1,1,28,28 --backend=simd

# 4. 使用 onnx2mbt AOT 编译真实模型（支持多输入形状与 --freeze-i64 分支特化）
moon run apps/cli --target native -- compile \
  silero_vad.onnx ./out_vad_16k \
  --input-shape="input:1,512;state:2,1,128" \
  --freeze-i64="sr:16000" \
  --struct=SileroVad16k --target=wasm-gc --mode=arena

# 5. 运行 Wasm V128 算子基准与 AOT 端到端基准
node scripts/bench_wasm.mjs
node scripts/bench_aot_wasm.mjs
```

## 架构文档

模块分层约束、依赖图与长期演进设计参见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
