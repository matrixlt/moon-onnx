# matrixlt/ml_kernels

零外部依赖、零 C-FFI、100% 基于 MoonBit 标准库原生 `V128`（`f32x4`）构建的跨端机器学习数值计算内核库，单一代码库即可编译运行于 `native`、`wasm`、`wasm-gc` 与 `js` 四个目标平台。

## 安装

```bash
moon add matrixlt/ml_kernels
```

## 子包结构

- `matrixlt/ml_kernels/gemm`：
  - 缓存友好 `i-k-j` 标量矩阵乘法（`matmul_2d_scalar_f32`）
  - `NC = 64` 列分块 + `4×8` 寄存器分块 `V128` SIMD GEMM 微内核（`matmul_2d_simd_f32`）
  - 静态权重预打包矩阵：`PackedMatrixB`、`pack_matrix_b_f32`、`pack_matrix_b_transposed_f32`、`matmul_2d_prepacked_simd_f32`
  - 面向 2D 卷积的专用 NCHW + 偏置融合预打包微内核：`matmul_2d_prepacked_nchw_simd_f32`
- `matrixlt/ml_kernels/vecmath`：
  - 基于 `FixedArray[Byte]` 零转换 `v128_load`/`v128_store` 的 4-wide `add` / `sub` / `mul` / `relu`（`f32x4_pmax`）
  - V128 向量化激活与超越函数：`exp`（6 阶 Cephes 多项式）、`sigmoid`、`tanh`、`clip`
  - 面向 `FixedArray[Float]` 缓冲的便捷 `V128` 运算与小端序打包/解包工具
- `matrixlt/ml_kernels/conv`：
  - `im2col_2d_f32`（`[K, out_spatial]` 布局）
  - `im2col_2d_nhwc_f32`（`[out_spatial, K]` 转置布局，内置 `1×1` 点卷积与内部像素零边界检查快速路径）

## Native Release 实测加速效果（Scalar vs `V128` SIMD）

- `GEMM 256×256`：**7.3×–9.15×**（纯 MoonBit 单线程）
- `Relu (n=65536, byte-packed)`：**7.6×**
- `Add (n=65536, byte-packed)`：**6.4×–10.9×**
- `Conv2D 1×8×14×14 → 16×14×14 (k=5, 预打包 + BufferPool)`：**4.78×** vs Scalar

## 许可证

Apache-2.0
