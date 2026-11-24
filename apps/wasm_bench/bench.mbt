///|
/// 构造确定性非零测试矩阵，避免全零输入触发标量零跳过捷径导致基准失真。
fn make_mat(rows : Int32, cols : Int32, seed : Int32) -> FixedArray[Float] {
  FixedArray::makei(rows * cols, fn(i) {
    Float::from_int((i * 17 + seed) % 29 - 14) * (0.05 : Float) + (0.01 : Float)
  })
}

///|
/// 64×64×64 GEMM 标量基准（循环 `iters` 次返回累加校验值防止 DCE）。
pub fn bench_gemm64_scalar(iters : Int32) -> Float {
  let a = make_mat(64, 64, 1)
  let b = make_mat(64, 64, 2)
  let c = FixedArray::make(64 * 64, (0.0 : Float))
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    @gemm.matmul_2d_scalar_f32(a, 0, b, 0, c, 0, 64, 64, 64)
    acc = acc + c[0]
  }
  acc
}

///|
/// 64×64×64 GEMM 预打包 V128 SIMD 基准。
pub fn bench_gemm64_simd(iters : Int32) -> Float {
  let a = make_mat(64, 64, 1)
  let pb = @gemm.pack_matrix_b_f32(make_mat(64, 64, 2), 0, 64, 64)
  let c = FixedArray::make(64 * 64, (0.0 : Float))
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    @gemm.matmul_2d_prepacked_simd_f32(a, 0, pb, c, 0, 64)
    acc = acc + c[0]
  }
  acc
}

///|
/// 128×128×128 GEMM 标量基准。
pub fn bench_gemm128_scalar(iters : Int32) -> Float {
  let a = make_mat(128, 128, 3)
  let b = make_mat(128, 128, 5)
  let c = FixedArray::make(128 * 128, (0.0 : Float))
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    @gemm.matmul_2d_scalar_f32(a, 0, b, 0, c, 0, 128, 128, 128)
    acc = acc + c[0]
  }
  acc
}

///|
/// 128×128×128 GEMM 预打包 V128 SIMD 基准。
pub fn bench_gemm128_simd(iters : Int32) -> Float {
  let a = make_mat(128, 128, 3)
  let pb = @gemm.pack_matrix_b_f32(make_mat(128, 128, 5), 0, 128, 128)
  let c = FixedArray::make(128 * 128, (0.0 : Float))
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    @gemm.matmul_2d_prepacked_simd_f32(a, 0, pb, c, 0, 128)
    acc = acc + c[0]
  }
  acc
}

///|
/// Conv2D (`1×8×14×14 → 16×14×14, k=5, pad=2`) + ReLU 标量基准。
pub fn bench_conv2d_scalar(iters : Int32) -> Float {
  let x = @tensor.Tensor::from_f32([1, 8, 14, 14], make_mat(8, 196, 3)) catch {
    _ => abort("tensor init failed")
  }
  let w = @tensor.Tensor::from_f32([16, 8, 5, 5], make_mat(16, 200, 5)) catch {
    _ => abort("tensor init failed")
  }
  let b = @tensor.Tensor::from_f32([16], make_mat(1, 16, 7)) catch {
    _ => abort("tensor init failed")
  }
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    let out = @ops.conv2d_f32(
      x,
      w,
      Some(b),
      [1, 1],
      [2, 2, 2, 2],
      [1, 1],
      1,
      @nn_ir.NotSet,
      use_simd=false,
      activation=@nn_ir.ReluAct,
    ) catch {
      _ => abort("conv scalar failed")
    }
    acc = acc + (out.get_f32_flat(0) catch { _ => (0.0 : Float) })
  }
  acc
}

///|
/// Conv2D (`1×8×14×14 → 16×14×14, k=5, pad=2`) + 融合 ReLU 预打包 V128 + BufferPool 基准。
pub fn bench_conv2d_simd(iters : Int32) -> Float {
  let x = @tensor.Tensor::from_f32([1, 8, 14, 14], make_mat(8, 196, 3)) catch {
    _ => abort("tensor init failed")
  }
  let w = @tensor.Tensor::from_f32([16, 8, 5, 5], make_mat(16, 200, 5)) catch {
    _ => abort("tensor init failed")
  }
  let b = @tensor.Tensor::from_f32([16], make_mat(1, 16, 7)) catch {
    _ => abort("tensor init failed")
  }
  let pw = @ops.pack_conv2d_weight_f32(w, 1) catch {
    _ => abort("pack conv failed")
  }
  let pool = @tensor.BufferPool::new()
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    let out = @ops.conv2d_prepacked_f32(
      x,
      pw,
      Some(b),
      [1, 1],
      [2, 2, 2, 2],
      [1, 1],
      @nn_ir.NotSet,
      activation=@nn_ir.ReluAct,
      pool=Some(pool),
      scratch_pool=Some(pool),
    ) catch {
      _ => abort("conv simd failed")
    }
    acc = acc + (out.get_f32_flat(0) catch { _ => (0.0 : Float) })
    match out.raw_f32_buffer() {
      Some(buf) => pool.release_f32(buf)
      None => ()
    }
  }
  acc
}

///|
/// Depthwise Conv2D (`1×32×14×14, k=3, pad=1, group=32`) + ReLU6 通用标量基准。
pub fn bench_depthwise_scalar(iters : Int32) -> Float {
  let x = @tensor.Tensor::from_f32([1, 32, 14, 14], make_mat(32, 196, 2)) catch {
    _ => abort("tensor init failed")
  }
  let w = @tensor.Tensor::from_f32([32, 1, 3, 3], make_mat(32, 9, 4)) catch {
    _ => abort("tensor init failed")
  }
  let b = @tensor.Tensor::from_f32([32], make_mat(1, 32, 6)) catch {
    _ => abort("tensor init failed")
  }
  let act = @nn_ir.ClipAct(min=(0.0 : Float), max=(6.0 : Float))
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    let out = @ops.conv2d_f32(
      x,
      w,
      Some(b),
      [1, 1],
      [1, 1, 1, 1],
      [1, 1],
      32,
      @nn_ir.NotSet,
      use_simd=false,
      activation=act,
    ) catch {
      _ => abort("dw scalar failed")
    }
    acc = acc + (out.get_f32_flat(0) catch { _ => (0.0 : Float) })
  }
  acc
}

///|
/// Depthwise Conv2D (`1×32×14×14, k=3, pad=1, group=32`) + ReLU6 零 im2col 专用 V128 基准。
pub fn bench_depthwise_simd(iters : Int32) -> Float {
  let x_buf = make_mat(32, 196, 2)
  let w_buf = make_mat(32, 9, 4)
  let b_buf = make_mat(1, 32, 6)
  let out_buf = FixedArray::make(32 * 196, (0.0 : Float))
  let ep = @gemm.Clip(min=(0.0 : Float), max=(6.0 : Float))
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    @conv.depthwise_conv2d_f32(
      x_buf,
      0,
      w_buf,
      Some(b_buf),
      32,
      14,
      14,
      3,
      3,
      1,
      1,
      1,
      1,
      1,
      1,
      14,
      14,
      out_buf,
      0,
      epilogue=ep,
    )
    acc = acc + out_buf[0]
  }
  acc
}

///|
/// Relu (`n = 65536`) 标量基准。
pub fn bench_relu65k_scalar(iters : Int32) -> Float {
  let src = make_mat(256, 256, 11)
  let out = FixedArray::make(65536, (0.0 : Float))
  let zero : Float = 0.0
  let mut acc : Float = 0.0
  for _i = 0; _i < iters; _i = _i + 1 {
    for j = 0; j < 65536; j = j + 1 {
      let v = src[j]
      out[j] = if v > zero { v } else { zero }
    }
    acc = acc + out[0]
  }
  acc
}

///|
/// Relu (`n = 65536`) 零转换字节缓冲 V128 `f32x4_pmax` 基准。
pub fn bench_relu65k_simd_bytes(iters : Int32) -> Float {
  let src = @vecmath.pack_f32_to_bytes(make_mat(256, 256, 11))
  let out = FixedArray::make(65536 * 4, b'\x00')
  let mut acc : Int32 = 0
  for _i = 0; _i < iters; _i = _i + 1 {
    @vecmath.relu_bytes_v128(src, out, 65536)
    acc = acc + out[0].to_int()
  }
  Float::from_int(acc)
}
