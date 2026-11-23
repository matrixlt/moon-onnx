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

---

## 2026-11-23 · Phase 2（Commit 6）：强类型计算图 IR 与拓扑排序（`src/ir`）

- 定义强类型算子 ADT `Op`（覆盖 MNIST MLP/CNN 及常用视觉算子共 18 种）与结构化错误 `IrError`。
- 实现 `lower_model`：解析 ONNX 动态 `AttributeProto`（兼容 `Softmax` 在 Opset < 13 / >= 13 的默认轴差异与 `auto_pad` 枚举），自动检测缺失输入引用。
- 实现稳定 Kahn 拓扑排序 `topo_sort_nodes` 与环路检测（`CyclicGraph`）。
- 通过真实 `mnist_mlp.onnx` 与 `mnist_cnn.onnx` Lowering 测试、乱序重排测试与环检测测试（累计 12 项测试全部通过）。

---

## 2026-11-23 · Phase 2（Commit 7）：实现逐元素、多向广播与激活算子（`src/ops/elementwise.mbt`）

- 实现 `add_f32`、`sub_f32`、`mul_f32`：覆盖同形状连续快速路径、2D+1D 偏置快速路径与基于虚拟广播步长的任意阶 NumPy/ONNX 多向广播通用路径。
- 实现 `relu_f32` 与数值稳定版 `softmax_f32`（支持任意正负轴，先减最大值并在 `Double` 精度累计指数和避免溢出错差）。
- 新增单元测试覆盖 4D+3D CNN 通道偏置广播、2D+1D 偏置广播与大数值 Softmax 稳定性（累计 15 项测试全部通过）。

---

## 2026-11-23 · Phase 2（Commit 8）：实现形状变换算子（`src/ops/shape_ops.mbt`）

- 实现 `reshape_op`（解析 Int64 形状张量，支持非连续输入自动连续化）、`flatten_op`、`transpose_op`（空 `perm` 按 ONNX 规范默认反转全部维度）。
- 实现 `squeeze_op` / `unsqueeze_op`，同时兼容 Opset < 13（属性传轴）与 Opset >= 13（第二输入张量传轴）。
- 累计 17 项测试全部通过。

---

## 2026-11-23 · Phase 2（Commit 9）：实现 `MatMul` 与 `Gemm` 标量内核（`src/ops/linalg.mbt`）

- 实现缓存友好型 `i-k-j` 循环顺序的 `matmul_2d_raw_f32` 内核，抽离为独立可替换的原始矩阵乘接口，便于 Phase 3 直接对比与接入 `V128` 分块 SIMD 内核。
- 实现完整 ONNX `gemm_f32`（支持 `transA`、`transB`、`alpha`、`beta` 及可选偏置 `C` 广播）与 `matmul_f32`（支持 2D、1D 向量点积与任意高维 Batch 广播）。
- 累计 19 项测试全部通过。

---

## 2026-11-23 · Phase 2（Commit 10）：实现 `Session` 执行引擎并跑通 MNIST MLP 端到端对齐

- 新增 `src/runtime`：`Session`（支持 `from_file` / `from_bytes` / `from_model` / `from_graph`）、`BackendKind`（`Scalar` / `Simd`）、`.f32` 原始张量加载器与 `max_abs_diff_f32` 数值校验器。
- 扩展 Native CLI `src/cli`：新增 `run <model.onnx> <input.f32> [golden.f32] [--backend=scalar|simd]` 子命令。
- **首个真实模型端到端验证结果（`mnist_mlp.onnx` vs ONNX Runtime 1.32.0）**：
  - 预测类别：`9`（与 ORT 完全一致，概率 `0.495124`）
  - 最大绝对误差（`max_abs_diff`）：**`1.1920929e-7`**（达到单精度浮点 1 ULP 精度）
  - 累计 20 项测试全部通过。

---

## 2026-11-23 · Phase 2（Commit 11）：实现 `Conv2D`、`MaxPool`、`AveragePool` 与 `BatchNormalization`（`src/ops/nn.mbt`）

- 实现基于 `im2col + 可插拔 GEMM` 的 2D 卷积 `conv2d_f32_with_gemm` / `conv2d_f32`，支持 `strides`、`pads`、`dilations`、`group`（含分组与深度可分离卷积）、`auto_pad`（`NOTSET`/`VALID`/`SAME_UPPER`/`SAME_LOWER`）以及可选偏置 `B`。
- 将 `gemm_fn` 抽离为回调参数，确保 Phase 3 接入 `V128` SIMD GEMM 时卷积算子自动共享加速能力、零重复逻辑。
- 实现 `maxpool_2d_f32`、`avgpool_2d_f32`（支持 `ceil_mode` / `count_include_pad`）与推理态 `batch_norm_f32`（每通道预计算等效仿射系数）。
- 累计 23 项测试全部通过。

---

## 2026-11-23 · Phase 2（Commit 12）：MNIST CNN 端到端对齐 ONNX Runtime

- 在 `src/runtime/session.mbt` 中接入 `Conv`、`MaxPool`、`AveragePool`、`BatchNormalization`，至此 `IrGraph` 的 18 个强类型 `Op` 变体已全部穷尽匹配，无未实现分支。
- **MNIST CNN（11 节点，对齐 ONNX Model Zoo MNIST-12 结构）端到端差分测试结果**：
  - 预测类别：`0`（与 ORT 完全一致，logit=`2.5831244`）
  - 最大绝对误差（`max_abs_diff`）：**`1.1920929e-6`**（远优于 `1e-5` 验收标准）
  - 累计 24 项测试全部通过。
- **Phase 2（标量推理闭环与 MNIST 端到端正确性对齐）全部达成。**

---

## 2026-11-23 · Phase 3（Commit 13）：实现图优化 Pass 管线（`src/opt`）

- 新增 `src/opt/optimize.mbt`，实现 5 个确定性图优化 Pass：
  1. **Constant Lifting**：将 `Constant` 节点静态张量直接提升为 Initializer。
  2. **Identity Elimination**：重写下游张量别名并剔除冗余 `Identity` 节点。
  3. **Conv + BatchNormalization 静态折叠**：在 `Double` 精度下将推理态 BN 参数合并入卷积核与偏置，单测验证与未融合结果误差 `< 1e-5`。
  4. **Conv + 通道偏置 Add 折叠**：针对 ONNX Model Zoo MNIST 常见导出形态，将 `Conv(x, W) -> Add(c, bias[C_out, 1, 1])` 合并为带偏置单节点 `Conv`，使 `mnist_cnn.onnx` 从 **11 节点缩减为 9 节点**。
  5. **Dead Code Elimination (DCE)**：基于拓扑序反向标记消除不可达死节点。
- 将 `optimize_graph` 默认接入 `Session`（提供 `optimize? : Bool = true` 开关），优化后 `mnist_cnn.onnx` 依然与 ONNX Runtime 保持完全一致（累计 27 项测试全部通过）。
