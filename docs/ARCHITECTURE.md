# moon-onnx 架构设计文档

本文档记录 `matrixlt/moon-onnx` 工作区的模块分层边界、依赖约束与关键工程决策。架构设计参考了 **ONNX Runtime**（Graph / Optimizer / MLAS / ExecutionProvider / Session）、**Tract**（独立 `data` / `linalg` / 格式无关 `core` / ONNX 前端）与 **RTen**（独立 `tensor` / `gemm` / `vecmath` / `onnx`）。

## 1. 模块依赖图（严格无环 DAG，6 大独立模块）

```text
Layer 0 (零外部依赖，可被整个 MoonBit 社区独立复用):
  matrixlt/tensor        N 维张量、多 DType、SymDim 符号维度、零拷贝视图、多向广播、BufferPool
  matrixlt/ml_kernels    原生 V128 SIMD 内核集合（入口 O(1) 边界校验 + 内层 unsafe_get/unsafe_set 零检查热路径）：
                         /gemm (BLIS 6×8 K-unroll-2 微内核 + PackedMatrixB + 零拷贝 NT + Relu/Clip/SiLU 寄存器级融合)、
                         /vecmath、/conv (16×16 分块 1×1 + 全展开 3×3 im2col + 零 im2col Depthwise)、
                         /quant (INT8 i32x4.dot_i16x8_s + Wasm-GC 4 列展开标量 + Q4 MatMulNBits 奇偶拆分 4 列展开 V128)、
                         /nn (零分配 Pool/Erf/GELU/GeluExact/SiLU/Softmax/LayerNorm/RMSNorm/RoPE/GQA/Resize/NMS + Wasm-GC 4 通道展开 Conv)

Layer 1 (模型格式与格式无关编译中端):
  matrixlt/onnx_proto    官方 onnx.proto 绑定 + 纯内存加载器（DimSpec = @tensor.SymDim）+ /fs 可选文件读取
    └─ depends on: moonbitlang/protobuf, matrixlt/tensor
  matrixlt/nn_ir         强类型 69 算子 IR、Kahn 拓扑排序、分级图优化 Pass (/opt: Conv+BN+Bias+Relu/Clip/SiLU、LayerNorm、SiLU、GeluExact)、静态形状/类型推断单一真相源 (/analysis)
    └─ depends on: matrixlt/tensor

Layer 2 (动态推理引擎运行时与独立 AOT 静态编译器):
  matrixlt/onnx          纯内存运行时核心：/frontend (ONNX->IR)、/ops、/session + 根包 @onnx 门面 + /fs 可选文件读取
    └─ depends on: tensor, ml_kernels, onnx_proto, nn_ir
  matrixlt/onnx_aot      独立 AOT 静态编译器 (`onnx2mbt` v2)：图特化、常量折叠、Static Arena 规划、Arena/Tensor 代码生成
    └─ depends on: tensor, onnx_proto, nn_ir (/opt, /analysis), onnx (/frontend, /ops)
```

## 2. 核心设计约束

1. **零 C-FFI 核心**：全部 6 个核心库均为 100% 纯 MoonBit 实现，所有纯内存包（含 `matrixlt/onnx/session`、`matrixlt/onnx` 根包与 `matrixlt/onnx_aot`）均在 `native`、`wasm`、`wasm-gc`、`js` 四个目标上全量通过编译与单元测试。
2. **文件 I/O 严格边界隔离**：所有本地文件系统依赖（`moonbitlang/x/fs`）严格隔离在 `matrixlt/onnx_proto/fs` 与 `matrixlt/onnx/fs` 两个可选子包中，确保浏览器、Cloudflare Workers、WASI 与边缘 Wasm 场景下核心推理与 AOT 编译器 100% 纯内存零 I/O 污染。
3. **运行时与 AOT 编译器解耦**：动态推理运行时（`matrixlt/onnx`）与静态代码生成器（`matrixlt/onnx_aot`）拆分为独立模块；静态形状/数据类型推断统一收敛在 `matrixlt/nn_ir/analysis`（Single Source of Truth）；AOT 生成代码直接调用经四端验证的 `matrixlt/ml_kernels/{gemm,conv,quant,nn}`，杜绝脆弱的内联字符串算子模板。
4. **干净的 `ExecutionProvider` 抽象**：`EvalContext` 仅向后端暴露运行期张量环境与 `BufferPool`，不泄露后端私有权重结构；`CpuSimdProvider` 通过 `prepare_graph` 钩子私有维护 `CpuWeightCache`（覆盖 Float32 MatMul/Gemm/Conv 与 Int8 MatMulInteger 预打包缓存）。
5. **自动生成代码隔离**：`packages/onnx_proto/schema/top.mbt` 由 `scripts/gen_proto.sh` 自动生成（`json=false`），禁止手工修改；所有业务封装位于 `packages/onnx_proto/*.mbt`。
6. **原生 `V128` SIMD 与安全零边界检查规范**：
   - 全部 `ml_kernels` 热路径内核在入口执行一次性 O(1) 切片边界校验（非法越界安全 `abort`），内层循环统一使用 `unsafe_get`/`unsafe_set`，消除 Wasm 线性内存每元素边界检查开销；
   - Relu 使用 `@simd.f32x4_pmax(0.0, v)` 而非 `f32x4_max`，避免 IEEE NaN 传播回退导致的性能劣化；
   - FP32 GEMM 采用 BLIS 8 列微面板 + **`6×8` 寄存器分块 + `K` 维 2× 展开** `f32x4_relaxed_madd` 微内核（12 条 V128 累加器零寄存器溢出），并内置 `Relu` / `Clip` / `SiLU` 寄存器级 Epilogue 融合；
   - INT8 `MatMulInteger` 采用 4 列展开 `i32x4.dot_i16x8_s` 指令 + 预计算列和零减法公式（`PackedInt8MatrixB`），Wasm-GC 标量路径采用 4 列展开 + 无分支 `byte_to_i8`；
   - 4-bit `MatMulNBits` 采用 16 字节批量奇偶 nibble 拆分 + 4 列展开 `f32x4_relaxed_madd` 微内核。
