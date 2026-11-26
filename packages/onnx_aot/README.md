# matrixlt/onnx_aot

纯 MoonBit ONNX AOT 静态编译器（`onnx2mbt` v2），独立于运行时包 `matrixlt/onnx`，100% 纯内存、四端通用（Native / Wasm / Wasm-GC / JS）。

## 核心特性

- **编译期图特化**（`specialize_graph`）：冻结静态输入（如采样率）、内联常量 `If` 分支、折叠静态 `Shape`/元算子子图、消除同类型冗余 `Cast`。
- **静态内存区间规划**（`plan_static_arenas`）：基于 Greedy-by-Size First-Fit 区间着色算法复用 Float32 / Byte 中间激活 Arena，单次 `forward_into` 推理实现 **0 堆分配**。
- **极简零运行时代码生成**（`Arena` 模式）：生成的 `.mbt` 代码直接调用 `matrixlt/ml_kernels/{gemm,conv,quant,nn}`，不依赖 `onnx_proto`、`nn_ir`、`Tensor` 或 `BufferPool`，支持 Wasm V128 与 Wasm-GC 双目标自适应策略，以及 `Tensor` 回退模式。

## 快速开始

```moonbit
let artifact = @onnx_aot.compile_model_to_mbt(
  model,
  Map([("input", [1, 3, 224, 224])]),
  config={ ..@onnx_aot.CodegenConfig::default(), target: @onnx_aot.TargetWasm },
)
// artifact.mbt_source  -> 生成的独立 MoonBit 推理源码
// artifact.pkg_config  -> 最小 moon.pkg 依赖声明
// artifact.weights_bin -> 紧凑打包的二进制权重
```
