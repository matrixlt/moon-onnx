///|
/// 从可选 `BufferPool` 获取指定长度的 `FixedArray[Float]`（无池时直接分配）。
fn alloc_f32(pool : @tensor.BufferPool?, len : Int32) -> FixedArray[Float] {
  match pool {
    Some(p) => p.acquire_f32(len)
    None => FixedArray::make(len, (0.0 : Float))
  }
}

///|
/// 将临时工作缓冲区归还至可选 `BufferPool`。
fn free_f32(pool : @tensor.BufferPool?, buf : FixedArray[Float]) -> Unit {
  match pool {
    Some(p) => p.release_f32(buf)
    None => ()
  }
}

///|
/// 二元逐元素算术操作类型（内部复用快速路径与通用多向广播逻辑）。
priv enum BinaryOp {
  BAdd
  BSub
  BMul
  BDiv
  BPow
}

///|
/// 标量二元运算核心分派。
fn apply_binary(op : BinaryOp, x : Float, y : Float) -> Float {
  match op {
    BAdd => x + y
    BSub => x - y
    BMul => x * y
    BDiv => x / y
    BPow => Float::from_double(@math.pow(x.to_double(), y.to_double()))
  }
}

///|
/// 通用二元 Float32 算子，支持同形状 V128 SIMD / 标量快速路径与完整 NumPy/ONNX 多向广播。
fn binary_f32(
  op : BinaryOp,
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd : Bool,
  pool : @tensor.BufferPool?,
) -> @tensor.Tensor raise @tensor.TensorError {
  let a_shape = a.shape()
  let b_shape = b.shape()
  let out_shape = @tensor.broadcast_shape(a_shape, b_shape)
  let total = @tensor.num_elements(out_shape)

  // Int64 / Int32 整型广播算术支持（用于 LLM attention_mask / shape 索引计算子图）
  if a.dtype() == @tensor.Int64 && b.dtype() == @tensor.Int64 {
    let a_buf = a.to_i64_buffer()
    let b_buf = b.to_i64_buffer()
    let a_bs = @tensor.broadcast_strides(
      a_shape,
      @tensor.compute_strides(a_shape),
      out_shape,
    )
    let b_bs = @tensor.broadcast_strides(
      b_shape,
      @tensor.compute_strides(b_shape),
      out_shape,
    )
    let rank = out_shape.length()
    let out = FixedArray::make(total, (0 : Int64))
    for linear = 0; linear < total; linear = linear + 1 {
      let mut rem = linear
      let mut ia = 0
      let mut ib = 0
      for axis = rank - 1; axis >= 0; axis = axis - 1 {
        let dim = out_shape[axis]
        let coord = rem % dim
        rem = rem / dim
        ia = ia + coord * a_bs[axis]
        ib = ib + coord * b_bs[axis]
      }
      let va = a_buf[ia]
      let vb = b_buf[ib]
      out[linear] = match op {
        BAdd => va + vb
        BSub => va - vb
        BMul => va * vb
        BDiv => va / vb
        BPow => @math.pow(va.to_double(), vb.to_double()).to_int64()
      }
    }
    return @tensor.Tensor::from_i64(out_shape, out)
  }
  if a.dtype() == @tensor.Int32 && b.dtype() == @tensor.Int32 {
    let a_buf = a.to_i32_buffer()
    let b_buf = b.to_i32_buffer()
    let a_bs = @tensor.broadcast_strides(
      a_shape,
      @tensor.compute_strides(a_shape),
      out_shape,
    )
    let b_bs = @tensor.broadcast_strides(
      b_shape,
      @tensor.compute_strides(b_shape),
      out_shape,
    )
    let rank = out_shape.length()
    let out = FixedArray::make(total, 0)
    for linear = 0; linear < total; linear = linear + 1 {
      let mut rem = linear
      let mut ia = 0
      let mut ib = 0
      for axis = rank - 1; axis >= 0; axis = axis - 1 {
        let dim = out_shape[axis]
        let coord = rem % dim
        rem = rem / dim
        ia = ia + coord * a_bs[axis]
        ib = ib + coord * b_bs[axis]
      }
      let va = a_buf[ia]
      let vb = b_buf[ib]
      out[linear] = match op {
        BAdd => va + vb
        BSub => va - vb
        BMul => va * vb
        BDiv => va / vb
        BPow => @math.pow(va.to_double(), vb.to_double()).to_int()
      }
    }
    return @tensor.Tensor::from_i32(out_shape, out)
  }

  let a_buf = a.to_f32_buffer()
  let b_buf = b.to_f32_buffer()
  let out = alloc_f32(pool, total)

  // 快速路径 1：同形状连续张量（支持 V128 SIMD 分派）
  if a_shape == b_shape {
    if use_simd {
      match op {
        BAdd => @vecmath.add_f32_v128(a_buf, b_buf, out, total)
        BSub => @vecmath.sub_f32_v128(a_buf, b_buf, out, total)
        BMul => @vecmath.mul_f32_v128(a_buf, b_buf, out, total)
        BDiv => @vecmath.div_f32_v128(a_buf, b_buf, out, total)
        BPow =>
          for i = 0; i < total; i = i + 1 {
            out[i] = apply_binary(op, a_buf[i], b_buf[i])
          }
      }
    } else {
      for i = 0; i < total; i = i + 1 {
        out[i] = apply_binary(op, a_buf[i], b_buf[i])
      }
    }
    return @tensor.Tensor::from_f32(out_shape, out)
  }

  // 快速路径 2：2D 特征 [M, N] 与 1D 偏置 [N] 广播
  if a_shape.length() == 2 && b_shape.length() == 1 && a_shape[1] == b_shape[0] {
    let rows = a_shape[0]
    let cols = a_shape[1]
    for r = 0; r < rows; r = r + 1 {
      let row_off = r * cols
      for c = 0; c < cols; c = c + 1 {
        out[row_off + c] = apply_binary(op, a_buf[row_off + c], b_buf[c])
      }
    }
    return @tensor.Tensor::from_f32(out_shape, out)
  }

  // 通用路径：利用虚拟广播步长（被广播轴 stride = 0）逐元素计算
  let a_bstrides = @tensor.broadcast_strides(
    a_shape,
    @tensor.compute_strides(a_shape),
    out_shape,
  )
  let b_bstrides = @tensor.broadcast_strides(
    b_shape,
    @tensor.compute_strides(b_shape),
    out_shape,
  )
  let rank = out_shape.length()

  for linear = 0; linear < total; linear = linear + 1 {
    let mut rem = linear
    let mut ia = 0
    let mut ib = 0
    for axis = rank - 1; axis >= 0; axis = axis - 1 {
      let dim = out_shape[axis]
      let coord = rem % dim
      rem = rem / dim
      ia = ia + coord * a_bstrides[axis]
      ib = ib + coord * b_bstrides[axis]
    }
    out[linear] = apply_binary(op, a_buf[ia], b_buf[ib])
  }
  @tensor.Tensor::from_f32(out_shape, out)
}

///|
/// ONNX Add 算子（支持多向广播与可选 V128 SIMD 加速及 BufferPool 复用）。
pub fn add_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BAdd, a, b, use_simd, pool)
}

///|
/// ONNX Sub 算子（支持多向广播与可选 V128 SIMD 加速及 BufferPool 复用）。
pub fn sub_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BSub, a, b, use_simd, pool)
}

///|
/// ONNX Mul 算子（支持多向广播与可选 V128 SIMD 加速及 BufferPool 复用）。
pub fn mul_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BMul, a, b, use_simd, pool)
}

///|
/// ONNX Div 算子（支持多向广播与可选 V128 SIMD 加速及 BufferPool 复用）。
pub fn div_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BDiv, a, b, use_simd, pool)
}

///|
/// ONNX Pow 算子（支持多向广播与 BufferPool 复用）。
pub fn pow_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BPow, a, b, false, pool)
}

///|
/// ONNX Sqrt 算子（支持标量与 V128 `f32x4_sqrt` 向量化路径及 BufferPool 复用）。
pub fn sqrt_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.sqrt_f32_v128(src, out, n)
  } else {
    for i = 0; i < n; i = i + 1 {
      out[i] = Float::from_double(src[i].to_double().sqrt())
    }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Exp 算子（支持标量与 V128 向量化路径及 BufferPool 复用）。
pub fn exp_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.exp_f32_v128(src, out, n)
  } else {
    for i = 0; i < n; i = i + 1 {
      out[i] = Float::from_double(@math.exp(src[i].to_double()))
    }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Neg 算子（逐元素取反，支持 BufferPool 复用）。
pub fn neg_f32(
  x : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  for i = 0; i < n; i = i + 1 {
    out[i] = -src[i]
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Relu 算子（支持标量与 V128 `f32x4_pmax` 向量化路径及 BufferPool 复用）。
pub fn relu_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.relu_f32_v128(src, out, n)
  } else {
    let zero : Float = 0.0
    for i = 0; i < n; i = i + 1 {
      let v = src[i]
      out[i] = if v > zero { v } else { zero }
    }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Clip 算子（支持可选 `min`/`max` 上下界、V128 向量化快速路径与 BufferPool 复用）。
pub fn clip_f32(
  x : @tensor.Tensor,
  min_val : Float?,
  max_val : Float?,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  match (min_val, max_val) {
    (Some(lo), Some(hi)) =>
      if use_simd {
        @vecmath.clip_f32_v128(src, out, n, lo, hi)
      } else {
        for i = 0; i < n; i = i + 1 {
          let v = src[i]
          let c1 = if v < lo { lo } else { v }
          out[i] = if c1 > hi { hi } else { c1 }
        }
      }
    (Some(lo), None) =>
      if use_simd && lo == (0.0 : Float) {
        @vecmath.relu_f32_v128(src, out, n)
      } else {
        for i = 0; i < n; i = i + 1 {
          let v = src[i]
          out[i] = if v < lo { lo } else { v }
        }
      }
    (None, Some(hi)) =>
      for i = 0; i < n; i = i + 1 {
        let v = src[i]
        out[i] = if v > hi { hi } else { v }
      }
    (None, None) =>
      for i = 0; i < n; i = i + 1 {
        out[i] = src[i]
      }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Sigmoid 算子（支持标量与 V128 向量化路径及 BufferPool 复用）。
pub fn sigmoid_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.sigmoid_f32_v128(src, out, n)
  } else {
    for i = 0; i < n; i = i + 1 {
      let d = src[i].to_double()
      out[i] = Float::from_double(1.0 / (1.0 + @math.exp(-d)))
    }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Tanh 算子（支持标量与 V128 向量化路径及 BufferPool 复用）。
pub fn tanh_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.tanh_f32_v128(src, out, n)
  } else {
    for i = 0; i < n; i = i + 1 {
      out[i] = Float::from_double(@math.tanh(src[i].to_double()))
    }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Gelu 算子（Tanh 近似：`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`）。
pub fn gelu_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.gelu_f32_v128(src, out, n)
  } else {
    @nn.gelu_f32(src, 0, out, 0, n)
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX 精确 GELU 算子：`0.5 * x * (1 + erf(x / sqrt(2)))`（复用 `ml_kernels/nn` 单遍融合内核）。
pub fn gelu_exact_f32(
  x : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  @nn.gelu_exact_f32(src, 0, out, 0, n)
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX HardSwish 算子：`x * clamp(x / 6 + 0.5, 0, 1)`（MobileNetV3 高频激活）。
pub fn hard_swish_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  if use_simd {
    @vecmath.hard_swish_f32_v128(src, out, n)
  } else {
    let zero : Float = 0.0
    let one : Float = 1.0
    let sixth : Float = Float::from_double(1.0 / 6.0)
    let half : Float = 0.5
    for i = 0; i < n; i = i + 1 {
      let v = src[i]
      let t = v * sixth + half
      let clamped = if t < zero { zero } else if t > one { one } else { t }
      out[i] = v * clamped
    }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX LeakyRelu 算子：`x >= 0 ? x : alpha * x`。
pub fn leaky_relu_f32(
  x : @tensor.Tensor,
  alpha~ : Float = (0.01 : Float),
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  let zero : Float = 0.0
  for i = 0; i < n; i = i + 1 {
    let v = src[i]
    out[i] = if v >= zero { v } else { alpha * v }
  }
  @tensor.Tensor::from_f32(x.shape(), out)
}

///|
/// ONNX Softmax 算子（数值稳定版，支持任意正负轴与 BufferPool 复用，委托 `ml_kernels/nn` 内核）。
pub fn softmax_f32(
  x : @tensor.Tensor,
  axis : Int32,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  if rank == 0 {
    return @tensor.Tensor::from_f32([], FixedArray::make(1, (1.0 : Float)))
  }
  let ax = @tensor.normalize_axis(axis, rank)

  let mut outer = 1
  for i = 0; i < ax; i = i + 1 {
    outer = outer * shape[i]
  }
  let axis_dim = shape[ax]
  let mut inner = 1
  for i = ax + 1; i < rank; i = i + 1 {
    inner = inner * shape[i]
  }

  let src = x.to_f32_buffer()
  let out = alloc_f32(pool, src.length())
  @nn.softmax_f32(src, 0, out, 0, outer, axis_dim, inner)
  @tensor.Tensor::from_f32(shape, out)
}

///|
/// ONNX `Erf` 算子：逐元素计算高斯误差函数 `erf(x)`（支持 BufferPool 复用，委托 `ml_kernels/nn` 内核）。
pub fn erf_f32(
  x : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = alloc_f32(pool, n)
  @nn.erf_f32(src, 0, out, 0, n)
  @tensor.Tensor::from_f32(x.shape(), out)
}
