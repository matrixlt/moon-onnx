# moon-onnx 开发日志（DEVLOG）

本文档按阶段持续记录 `moon-onnx` 从零搭建到完成 MNIST 端到端推理与 `V128` SIMD 加速的全过程，包括工具链版本、架构决策、踩坑记录、与 ONNX Runtime 的数值误差对齐结果，以及各阶段基准测试数据。

---

## 2026-11-23 · Phase 0：工具链安装与工程脚手架

### 1. 工具链与环境版本

- **MoonBit**：`moon 0.1.20261119` / `moonc v0.10.22+124d61d05` / `moonrun 0.1.20261119`
- **Protobuf 编译器**：`libprotoc 3.21.12`（`/usr/bin/protoc`）
- **C 编译器**：`gcc` / `clang`（用于 MoonBit Native 后端链接）
- **Python 差分验证环境（`.venv`）**：Python 3.11.2、`onnx==1.23.2`、`onnxruntime==1.32.0`、`numpy==2.4.6`

### 2. 前期关键发现（影响架构设计）

1. **MoonBit 0.10.22 模块配置格式**：新版工具链采用 DSL 风格的 `moon.mod` 与 `moon.pkg`（支持 `preferred_target = "native"` 与 `pkgtype(kind: "executable")`）。
2. **MoonBit 内建 `V128` SIMD 原语**：标准库 `moonbitlang/core/v128` 已原生提供完整的 `f32x4` 指令集（`f32x4_add`、`f32x4_mul`、`f32x4_max`、`f32x4_relaxed_madd` 等），其中 `v128_load` / `v128_store` 直接作用于 `FixedArray[Byte]`。
   - **设计启示**：在标量阶段（Phase 1-2）使用 `FixedArray[Float]` 保证算子实现清晰易读；进入 Phase 3 SIMD 阶段时，GEMM 分块打包（Panel Packing）可直接将块数据打包到连续的 `FixedArray[Byte]` 缓冲区中，与 `v128_load` / `v128_store` 无缝衔接，并可在 Elementwise 路径测试零拷贝字节缓冲方案。
3. **官方 `onnx.proto` 来源**：Python `onnx==1.23.2` 包内自带标准 `onnx.proto`（`syntax = "proto2"`），接下来将在 Commit 2 中直接引入并验证 `moonbitlang/protoc-gen-mbt` 生成效果。

### 3. 本次提交内容（Commit 1）

- 初始化 Git 仓库与 `moon.mod`（设定 `preferred_target = "native"`）。
- 建立根包入口与基础 smoke test。
- 添加提交前强制门禁脚本 `scripts/pre_commit_check.sh`（依次执行 `moon fmt`、`moon check --target native`、`moon test --target native`）。

---

## 2026-11-23 · Phase 0（Commit 2）：引入 `onnx.proto` 与 `protoc-gen-mbt` 代码生成

### 1. 依赖与生成链路

- 添加依赖：`moonbitlang/protobuf@0.2.0`（Protobuf 运行时）、`moonbitlang/x@0.5.9`（Native 文件 I/O 等扩展能力）。
- 从 Python `onnx==1.23.2` 提取官方 `proto/onnx.proto`。
- 编写 `scripts/protoc-gen-mbt.sh`（基于 `moonx moonbitlang/protoc-gen-mbt@0.3.0`）与 `scripts/gen_proto.sh`，将生成结果落盘到 `src/proto_gen/`。

### 2. 踩坑记录与解决方案

- **问题**：默认参数运行 `protoc-gen-mbt@0.3.0` 时，插件会开启 `json=true`，对 `onnx.proto` 中两处 `repeated bytes` 字段（`AttributeProto.strings` 与 `TensorProto.string_data`）生成了 `@lib.base64_encode(Array[Bytes])`，触发 MoonBit 编译错误（`base64_encode` 期望单个 `Bytes`）。
- **解决**：ONNX 模型解析仅依赖二进制 Protobuf，在 `--mbt_opt` 中显式指定 `json=false`，同时在生成脚本中移除多余的 `moonbitlang/core/json` 导入。重新生成后 `moon check --target native` 与 `moon test --target native` 0 错误通过。

---

## 2026-11-23 · Phase 1（Commit 3）：实现核心张量模块 `src/tensor`

- 实现 `Tensor` 视图模型（`data` + `offset` + `shape` + `strides` + `dtype`）与非装箱 `FixedArray` 存储（`Float32`/`Float64`/`Int64`/`Int32`/`Bool`）。
- 实现行主序步长计算、多维/线性下标映射、NumPy/ONNX 多向广播形状与虚拟步长（步长置 0 零拷贝广播）。
- 实现 `contiguous`、`reshape`（支持 ONNX `-1` 自动推导与 `0` 复制维度）、`transpose`（零拷贝）、`flatten`、`squeeze`、`unsqueeze`、`argmax_f32` 与 `summary`。
- 遵循 MoonBit 0.10.22 规范采用 `derive(Debug, Eq)` + 显式 `Show` 实现，消除全部废弃 API 警告；`src/tensor` 6 项核心单元测试全部通过。

---

## 2026-11-23 · Phase 1（Commit 4）：实现 ONNX 模型加载器与 `TensorProto` 权重解码（`src/model`）

- 封装 `src/proto_gen`，实现跨平台 `load_model_from_bytes`（纯内存解析，Wasm/JS/Native 通用）与 Native 端 `load_model_from_file`。
- 实现 `decode_tensor_proto`：
  - 优先解码小端序 `raw_data` 二进制权重（通过 `Float::reinterpret_from_uint` 与 `UInt64::reinterpret_as_double` 零拷贝位重解释）。
  - 自动回退支持 `float_data` / `double_data` / `int64_data` / `int32_data` 等类型化 repeated 字段。
- 自动过滤旧版 ONNX 导出模型中重复出现在 `graph.input` 中的 Initializer 项。
- 提供 `OnnxModel::format_summary()` 人类可读模型摘要输出；完成 `TensorProto` 解码及 `ModelProto` 二进制序列化/反序列化闭环测试（累计 9 项测试全部通过，自写代码 0 警告）。

---

## 2026-11-23 · Phase 1（Commit 5）：Native CLI `inspect` 与 MNIST Fixture 验证

- 新增 `scripts/gen_mnist_fixtures.py`，利用 Python `onnx` + `onnxruntime` 生成两套通过 `onnx.checker` 严格校验的真实模型及黄金参考数据：
  - `tests/fixtures/mnist_mlp.onnx`（211 KB，52,650 参数，7 节点：Flatten / Gemm / Relu / Softmax）
  - `tests/fixtures/mnist_cnn.onnx`（24.8 KB，5,996 参数，11 节点：对齐 ONNX Model Zoo MNIST-12 的双层 Conv / Add / Relu / MaxPool / Reshape / MatMul / Add）
- 实现 Native CLI `src/cli`（采用 `moonbitlang/core/env` 替代旧版 `x/sys` 保持 0 警告）。
- 实测 `moon run src/cli --target native -- inspect` 成功解析两份真实 `.onnx` 文件，准确还原 IR Version、Opset 17、输入输出签名、全部 `raw_data` 权重张量形状与节点连接拓扑。
- **Phase 1（ONNX 模型解析与检视器）全部达成。**
