// matrixlt/onnx_proto — 官方 ONNX Protobuf 绑定、纯内存模型解析与结构检视库

name = "matrixlt/onnx_proto"

version = "0.2.0"

readme = "README.md"

repository = "https://github.com/matrixlt/moon-onnx"

license = "Apache-2.0"

keywords = [ "onnx", "protobuf", "parser", "ml", "wasm", "native" ]

preferred_target = "native"

supported_targets = "+all"

description = "Official ONNX Protobuf bindings, pure-memory model loader, TensorProto decoder, and inspector for MoonBit"

import {
  "moonbitlang/protobuf@0.2.0",
  "moonbitlang/x@0.5.9",
  "matrixlt/tensor@0.2.0",
}
