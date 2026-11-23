# matrixlt/tensor

零外部依赖、支持 `native` / `wasm` / `wasm-gc` / `js` 四端编译的纯 MoonBit 多维张量库，面向机器学习推理与高性能数值计算。

## 安装

```bash
moon add matrixlt/tensor
```

## 核心能力

- **紧凑非装箱存储**：基于 `FixedArray` 封装 `Float32` / `Float64` / `Int64` / `Int32` / `Bool` 多类型存储，无泛型装箱开销；内置小端序原始字节流互转（`from_f32_bytes_le` / `to_f32_bytes_le`）。
- **零拷贝视图模型**：基于 `(data, offset, shape, strides)` 支持 `transpose`、`squeeze`、`unsqueeze` 与连续 `reshape` 的零拷贝视图变换。
- **符号维度与形状推断底座**：提供 `SymDim` / `SymShape` 符号维度建模，支持动态 Batch/序列维度绑定与解析。
- **推理缓冲复用池（`BufferPool`）**：按精确长度分桶缓存 `FixedArray[Float]` 与 `FixedArray[Byte]`，并提供 `PoolStats` 命中率统计，消除高频推理的中间激活 GC 抖动。
- **完整 NumPy / ONNX 语义**：支持多向广播形状与虚拟步长（`broadcast_shape` / `broadcast_strides`）、ONNX `reshape`（`-1` 自动推导与 `0` 复制维度）、`flatten` 与 `argmax_f32`。

## 快速示例

```moonbit
let a = @tensor.Tensor::from_f32_array([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
let b = a.transpose([1, 0]).contiguous()
let c = b.reshape([1, -1])

let pool = @tensor.BufferPool::new()
let scratch = pool.acquire_f32(1024)
pool.release_f32(scratch)
```

## 许可证

Apache-2.0
