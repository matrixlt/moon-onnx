# matrixlt/nn_ir

格式无关的强类型机器学习计算图中间表示（Typed IR）与分级图优化器，仅依赖 `matrixlt/tensor`，零 ONNX Protobuf 依赖。

## 子包结构

- `matrixlt/nn_ir`（根包）：
  - 强类型算子代数数据类型 `Op`、`AutoPad`、`IrNode`、`IrGraph`、`IrError`
  - 稳定 Kahn 拓扑排序与依赖/环路校验（`topo_sort_nodes`）
- `matrixlt/nn_ir/opt`：
  - 分级图优化管线 `optimize_graph`：
    1. Constant 节点提升
    2. Identity 别名消除
    3. `Conv + BatchNormalization` 静态参数折叠
    4. `Conv + 通道偏置 Add` 融合
    5. 静态 `Gemm(transB=1)` 权重预转置
    6. 死代码消除（DCE）
