# matrixlt/nn_ir

格式无关的强类型机器学习计算图中间表示（Typed IR）与分级图优化器，仅依赖 `matrixlt/tensor`，零 ONNX Protobuf 依赖，支持四端编译。

## 安装

```bash
moon add matrixlt/nn_ir
```

## 子包结构

- `matrixlt/nn_ir`（根包）：
  - 覆盖 45 种强类型算子代数数据类型 `Op`（含 1D/2D `Conv`、`Resize`、`LSTM`、`LayerNormalization`、`Gelu`、`QuantizeLinear` / `DequantizeLinear` / `QLinearMatMul` 等）、`ActivationKind`、`AutoPad`、`IrNode`、`IrGraph`、`IrError`
  - 稳定 Kahn 拓扑排序与依赖/环路校验（`topo_sort_nodes`）
- `matrixlt/nn_ir/opt`：
  - 分级图优化管线 `optimize_graph`：
    1. `Constant` 节点提升为静态初始化器
    2. `Identity` 别名消除
    3. `Conv (1D/2D) + BatchNormalization` 静态参数折叠
    4. `Conv + 通道偏置 Add` 融合
    5. `Conv + Relu / Clip(ReLU6)` 激活融合
    6. 静态 `Gemm(transB=1)` 权重预转置
    7. 死代码消除（DCE）

## 快速示例

```moonbit
let (optimized_graph, report) = @opt.optimize_graph(raw_graph)
println(report.format_summary())
```

## 许可证

Apache-2.0
