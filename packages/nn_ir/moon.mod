// matrixlt/nn_ir — 格式无关的强类型神经网络计算图 IR 与分级图优化器

name = "matrixlt/nn_ir"

version = "0.1.0"

readme = "README.md"

repository = "https://github.com/matrixlt/moon-onnx"

license = "Apache-2.0"

keywords = [
  "ir",
  "compiler",
  "graph",
  "optimizer",
  "ml",
  "nn",
  "wasm",
  "native",
]

preferred_target = "native"

supported_targets = "+all"

description = "Format-agnostic typed neural network computation graph IR, topological validator, and graph optimizer for MoonBit"

import {
  "matrixlt/tensor@0.1.0",
}
