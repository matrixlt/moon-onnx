// matrixlt/onnx — 纯 MoonBit 跨端 ONNX 推理引擎（Native / Wasm / Wasm-GC / JS，支持 V128 SIMD）

name = "matrixlt/onnx"

version = "0.2.0"

readme = "README.md"

repository = "https://github.com/matrixlt/moon-onnx"

license = "Apache-2.0"

keywords = [ "onnx", "inference", "ml", "simd", "v128", "wasm", "native" ]

preferred_target = "native"

supported_targets = "+all"

description = "Pure-MoonBit cross-platform ONNX inference engine with typed IR, graph optimizations, and V128 SIMD kernels"

import {
  "moonbitlang/x@0.5.9",
  "matrixlt/tensor@0.2.0",
  "matrixlt/ml_kernels@0.2.0",
  "matrixlt/onnx_proto@0.2.0",
  "matrixlt/nn_ir@0.2.0",
}
