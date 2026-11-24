// wasm_bench — 纯内存 Wasm / Wasm-GC 端到端与核心算子性能基准导出模块（零文件系统依赖）

name = "matrixlt/moon_onnx_wasm_bench"

version = "0.1.0"

license = "Apache-2.0"

preferred_target = "wasm-gc"

supported_targets = "+all"

description = "Pure-memory WebAssembly (wasm & wasm-gc) V128 SIMD benchmark harness for moon-onnx"

import {
  "matrixlt/tensor@0.1.0",
  "matrixlt/ml_kernels@0.1.0",
  "matrixlt/nn_ir@0.1.0",
  "matrixlt/onnx@0.1.0",
}
