# matrixlt/ml_kernels

零外部依赖、零 C-FFI、100% 基于 MoonBit 标准库原生 `V128`（`f32x4`）构建的跨端机器学习数值计算内核库，单一代码库即可编译运行于 `native`、`wasm`、`wasm-gc` 与 `js` 四个目标平台。

## 子包结构

- `matrixlt/ml_kernels/gemm`：
  - 缓存友好 `i-k-j` 标量矩阵乘法（`matmul_2d_scalar_f32`）
  - `NC = 64` 缓存列分块 + `4×8` 寄存器分块 `V128` SIMD GEMM 微内核（`matmul_2d_simd_f32`）
  - 静态权重预打包矩阵（`PackedMatrixB` + `pack_matrix_b_f32` + `matmul_2d_prepacked_simd_f32`）
- `matrixlt/ml_kernels/vecmath`：
  - 基于 `FixedArray[Byte]` 零转换 `v128_load`/`v128_store` 的 4-wide `add` / `sub` / `mul` / `relu`（`f32x4_pmax`）
  - 面向 `FixedArray[Float]` 缓冲的便捷 `V128` 运算与小端序打包/解包工具
- `matrixlt/ml_kernels/conv`：
  - 2D 卷积 `im2col_2d_f32` 展开内核（支持 `strides` / `pads` / `dilations`）

## Native Release 实测加速效果（Scalar vs `V128` SIMD）

- `GEMM 64×64`：**9.51×**（205.7 µs → 21.6 µs）
- `GEMM 256×256`：**9.15×**（11.26 ms → 1.23 ms，~27.3 GFLOP/s 单线程）
- `Relu (n=65536, byte-packed)`：**7.65×**（55.3 µs → 7.23 µs）
- `Add (n=65536, byte-packed)`：**6.37×**（54.0 µs → 8.49 µs）
