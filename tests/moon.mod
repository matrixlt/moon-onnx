// tests — 端到端真实 ONNX 模型（MNIST MLP / CNN）与 ONNX Runtime 差分集成测试套件（仅工作区内部使用，不发布）

name = "matrixlt/moon_onnx_tests"

version = "0.2.0"

license = "Apache-2.0"

preferred_target = "native"

supported_targets = "native"

description = "End-to-end integration tests verifying MNIST MLP/CNN ONNX fixtures against ONNX Runtime golden outputs"

import {
  "matrixlt/tensor@0.2.0",
  "matrixlt/onnx_proto@0.2.0",
  "matrixlt/onnx@0.2.0",
}
