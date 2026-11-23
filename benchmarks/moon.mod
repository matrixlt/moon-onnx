// benchmarks — Scalar vs MoonBit V128 SIMD 微基准与端到端推理基准套件

name = "matrixlt/moon_onnx_benchmarks"

version = "0.1.0"

license = "Apache-2.0"

preferred_target = "native"

description = "Benchmarks comparing Scalar vs MoonBit V128 SIMD kernels and MNIST end-to-end inference"

import {
  "matrixlt/tensor@0.1.0",
  "matrixlt/ml_kernels@0.1.0",
  "matrixlt/onnx@0.1.0",
}
