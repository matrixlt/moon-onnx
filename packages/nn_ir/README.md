# matrixlt/nn_ir

格式无关的强类型机器学习计算图中间表示（Typed IR）与分级图优化器，仅依赖 `matrixlt/tensor`，零 ONNX Protobuf 依赖，支持四端编译。

## 安装

```bash
moon add matrixlt/nn_ir
```

## 子包结构

- `matrixlt/nn_ir`（根包）：
  - 覆盖 69 种强类型算子代数数据类型 `Op`（含 Q4 `MatMulNBits`、`RMSNormalization`/`SkipRmsNormalization`、`RotaryEmbedding`、`GroupQueryAttention`、1D/2D `Conv`、`Resize`、`NonMaxSuppression`、`LSTM`、`If` 控制流、`LayerNormalization`、`Gelu`/`GeluExact`/`Silu`、静态与动态 INT8 量化族等）、`FusedActivation`、`AutoPad`、`IrNode`、`IrGraph`、`IrError`
  - 稳定 Kahn 拓扑排序与依赖/环路校验（`topo_sort_nodes`）
- `matrixlt/nn_ir/analysis`：
  - 全项目唯一的编译期静态形状与数据类型推断真相源（`infer_static_shapes` / `infer_static_dtypes`）
- `matrixlt/nn_ir/opt`：
  - 分级图优化管线 `optimize_graph`：
    1. `Constant` 节点提升为静态初始化器
    2. `Identity` 别名消除
    3. `Conv (1D/2D) + BatchNormalization` 静态参数折叠
    4. `Conv + 通道偏置 Add` 融合
    5. `Mul(x, Sigmoid(x))` -> `Silu` 融合
    6. `Conv + Relu / Clip(ReLU6) / SiLU` 激活融合（支持并行分支非紧邻拓扑）
    7. 9 节点 PyTorch 分解 `LayerNormalization` 模式融合
    8. 5 节点精确 Erf `GeluExact` 模式融合（Whisper / BERT / ViT）
    9. 静态 `Gemm(transB=1)` 权重预转置
    10. 死代码消除（DCE）与 `If` 子图递归优化

## 快速示例

```moonbit
let (optimized_graph, report) = @opt.optimize_graph(raw_graph)
println(report.format_summary())
```

## 许可证

Apache-2.0
