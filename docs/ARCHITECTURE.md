# moon-onnx 架构设计文档

本文档记录 `matrixlt/moon-onnx` 工作区的模块分层边界、依赖约束与关键工程决策。架构设计参考了 **ONNX Runtime**（Graph / Optimizer / MLAS / ExecutionProvider / Session）、**Tract**（独立 `data` / `linalg` / 格式无关 `core` / ONNX 前端）与 **RTen**（独立 `tensor` / `gemm` / `vecmath` / `onnx`）。

## 1. 模块依赖图（严格无环 DAG）

```text
Layer 0 (零外部依赖，可被整个 MoonBit 社区独立复用):
  matrixlt/tensor        N 维张量、多 DType、零拷贝视图、多向广播
  matrixlt/ml_kernels    原生 V128 SIMD GEMM (含 PackedMatrixB)、vecmath、im2col

Layer 1 (模型格式与格式无关编译中端):
  matrixlt/onnx_proto    官方 onnx.proto 绑定 + 纯内存加载器 + /fs 可选文件读取
    └─ depends on: moonbitlang/protobuf, matrixlt/tensor
  matrixlt/nn_ir         强类型 Op IR、Kahn 拓扑排序、分级图优化 Pass (/opt)
    └─ depends on: matrixlt/tensor

Layer 2 & Facade (跨端推理引擎与统一入口):
  matrixlt/onnx          /frontend (ONNX->IR)、/ops、/session + 根包 @onnx 门面
    └─ depends on: tensor, ml_kernels, onnx_proto, nn_ir
```

## 2. 核心设计约束

1. **零 C-FFI 核心**：所有 5 个发布库均为 100% 纯 MoonBit 实现，在 `native`、`wasm`、`wasm-gc`、`js` 四个目标上均通过编译与核心测试。
2. **文件 I/O 边界隔离**：`matrixlt/onnx_proto` 根包仅处理内存 `Bytes`，本地文件读取隔离在 `matrixlt/onnx_proto/fs` 子包，确保浏览器与边缘 Wasm 场景无多余依赖。
3. **自动生成代码隔离**：`packages/onnx_proto/schema/top.mbt` 由 `scripts/gen_proto.sh` 自动生成（`json=false`），禁止手工修改；所有业务封装位于 `packages/onnx_proto/*.mbt`。
4. **原生 `V128` SIMD 规范**：
   - Relu 使用 `@v128.f32x4_pmax(0.0, v)` 而非 `f32x4_max`，避免 IEEE NaN 传播回退导致的性能劣化；
   - GEMM 采用 `NC=64` 列分块 + `4×8` 寄存器分块 `f32x4_relaxed_madd` 微内核，并提供 `PackedMatrixB` 预打包接口消除运行期重复打包。
