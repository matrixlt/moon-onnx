# matrixlt/ml_kernels

零外部依赖、零 C-FFI、100% 基于 MoonBit 标准库原生 `V128`（`f32x4` 与 INT8 `i32x4.dot_i16x8_s`）构建的跨端机器学习数值计算内核库，单一代码库即可编译运行于 `native`、`wasm`、`wasm-gc` 与 `js` 四个目标平台。

## 安装

```bash
moon add matrixlt/ml_kernels
```

## 子包结构（v0.2.0）

- `matrixlt/ml_kernels/gemm`：
  - 缓存友好 `i-k-j` 标量矩阵乘法（`matmul_2d_scalar_f32`）与零拷贝 `A @ B^T` V128 内核（`matmul_2d_nt_simd_f32`）
  - `NC = 64` 列分块 + `4×8` 寄存器分块 `V128` SIMD GEMM 微内核（`matmul_2d_simd_f32`）
  - 静态权重预打包矩阵：`PackedMatrixB`、`pack_matrix_b_f32`、`pack_matrix_b_transposed_f32`、`matmul_2d_prepacked_simd_f32`
  - 面向卷积的 NCHW + 偏置 + 尾部激活（`Epilogue::NoEpilogue` / `Relu` / `Clip` / `Silu`）寄存器级融合微内核：`matmul_2d_prepacked_nchw_simd_f32`
- `matrixlt/ml_kernels/conv`：
  - `im2col_2d_f32` 与 `im2col_2d_nhwc_f32`（内置 `16×16` 缓存分块 `1×1` 点卷积与内部像素零边界检查快速路径）
  - 零 `im2col` 专用 V128 深度可分离卷积：`depthwise_conv2d_f32`（支持融合偏置与 `Relu`/`Clip`/`Silu`）
- `matrixlt/ml_kernels/vecmath`：
  - 4-wide V128 逐元素算术与超越函数：`add` / `sub` / `mul` / `div` / `sqrt` / `relu` / `clip` / `exp` / `sigmoid` / `tanh` / `gelu` / `hard_swish`
- `matrixlt/ml_kernels/quant`：
  - 线性与动态量化/反量化（`quantize_*` / `dequantize_*` / `dynamic_quantize_linear_u8`，支持 Arena 偏移）
  - 硬件 V128 `i32x4.dot_i16x8_s` INT8/UINT8 量化矩阵乘法（`qlinear_matmul_*` 与预打包 `matmul_integer_u8_i8_prepacked_*`）
  - 4-bit 分块量化矩阵乘法：`matmul_nbits_q4_v128_f32`（16 字节奇偶 nibble 拆分 + 4 列展开 V128，支持 `block_size=32/64/128`）
  - 对称 Weight-Only INT8（`quantize_weights_symmetric_i8` / `matmul_f32_w8_i8`）
- `matrixlt/ml_kernels/nn`：
  - 零堆分配通用算子内核：`maxpool2d_f32`、`avgpool2d_f32`、`gap_f32`、`erf_f32`、`gelu_f32`、`gelu_exact_f32`、`softmax_f32`、`layernorm_f32`
  - 现代 LLM 内核：`silu_f32`、`rmsnorm_f32`、`skip_rmsnorm_f32`、`rotary_embedding_f32`、`gqa_f32`（在线 Softmax + KV-Cache）
  - 视觉检测内核：`resize_2d_f32`（nearest/bilinear）与 `nms_f32`（标准 ONNX `NonMaxSuppression`）

## 实测 V128 SIMD 加速效果（Node.js V8 `--target wasm --release`）

| 基准 | Scalar | V128 SIMD | 加速比 |
| :--- | ---: | ---: | ---: |
| `GEMM 64×64×64 (f32)` | 5668 µs | **289 µs** | **19.58×** |
| `GEMM 128×128×128 (f32)` | 42968 µs | **2214 µs** | **19.41×** |
| `QLinearMatMul 128×128×128 (int8, i32x4.dot)` | 31862 µs | **1255 µs** | **25.39×** |
| `Conv2D 1×8×14×14→16 k=5 +Relu` | 13517 µs | **1240 µs** | **10.90×** |
| `Relu n=65536 (byte V128 pmax)` | 1142 µs | **30.2 µs** | **37.78×** |

## 许可证

Apache-2.0
