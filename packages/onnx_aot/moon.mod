// matrixlt/onnx_aot — 纯 MoonBit ONNX AOT 静态编译器（onnx2mbt，支持静态内存区间复用 Arena 与 Wasm V128/GC 代码生成）

name = "matrixlt/onnx_aot"

version = "0.2.0"

readme = "README.md"

repository = "https://github.com/matrixlt/moon-onnx"

license = "Apache-2.0"

keywords = [ "onnx", "aot", "compiler", "codegen", "wasm", "simd", "moonbit" ]

preferred_target = "native"

supported_targets = "+all"

description = "Pure-MoonBit ONNX AOT compiler (onnx2mbt) generating zero-allocation static-arena MoonBit inference code"

import {
  "matrixlt/tensor@0.2.0",
  "matrixlt/onnx_proto@0.2.0",
  "matrixlt/nn_ir@0.2.0",
  "matrixlt/onnx@0.2.0",
}
