# matrixlt/tensor

零外部依赖、支持 `native` / `wasm` / `wasm-gc` / `js` 四端编译的纯 MoonBit 多维张量库，面向机器学习推理与高性能数值计算。

## 核心能力

- **紧凑非装箱存储**：基于 `FixedArray` 封装 `Float32` / `Float64` / `Int64` / `Int32` / `Bool` 多类型存储，无泛型装箱开销。
- **零拷贝视图模型**：基于 `(data, offset, shape, strides)` 支持 `transpose`、`squeeze`、`unsqueeze` 与连续 `reshape` 的零拷贝视图变换。
- **完整 NumPy / ONNX 语义**：支持多向广播形状与虚拟步长（`broadcast_shape` / `broadcast_strides`）、ONNX `reshape`（`-1` 自动推导与 `0` 复制维度）、`flatten` 与 `argmax_f32`。
- **结构化错误检查**：全部边界/形状/类型错误通过 checked `TensorError` 返回，不静默越界。

## 快速示例

```moonbit
let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
let b = a.transpose([1, 0]).contiguous()
let c = b.reshape([1, -1])
```
