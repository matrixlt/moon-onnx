// matrixlt/ml_kernels — 基于 MoonBit 原生 V128 SIMD 的跨端 ML 数值内核库（零外部依赖，零 FFI）

name = "matrixlt/ml_kernels"

version = "0.1.0"

readme = "README.md"

repository = "https://github.com/matrixlt/moon-onnx"

license = "Apache-2.0"

keywords = [
  "simd",
  "v128",
  "gemm",
  "matmul",
  "vecmath",
  "im2col",
  "ml",
  "wasm",
  "native",
]

preferred_target = "native"

supported_targets = "+all"

description = "Portable pure-MoonBit V128 SIMD GEMM, vecmath, and im2col numerical kernels (zero FFI, 4-target)"
