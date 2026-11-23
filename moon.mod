// MoonBit ONNX 模型解析与跨端推理引擎模块配置
// 优先以 native 为默认开发与验证目标，核心计算保持纯 MoonBit 以支持跨端编译。

name = "local/moon_onnx"

version = "0.1.0"

readme = "README.md"

repository = ""

license = "Apache-2.0"

keywords = [ "onnx", "inference", "ml", "simd", "wasm", "native" ]

preferred_target = "native"

description = "ONNX model parser and cross-platform inference engine written in MoonBit"

import {
  "moonbitlang/protobuf@0.2.0",
  "moonbitlang/x@0.5.9",
}
