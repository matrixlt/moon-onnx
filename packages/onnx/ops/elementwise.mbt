///|
/// 二元逐元素算术操作类型（内部复用快速路径与通用多向广播逻辑）。
priv enum BinaryOp {
  BAdd
  BSub
  BMul
}

///|
/// 标量二元运算核心分派。
fn apply_binary(op : BinaryOp, x : Float, y : Float) -> Float {
  match op {
    BAdd => x + y
    BSub => x - y
    BMul => x * y
  }
}

///|
/// 通用二元 Float32 算子，支持同形状 V128 SIMD / 标量快速路径与完整 NumPy/ONNX 多向广播。
fn binary_f32(
  op : BinaryOp,
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd : Bool,
) -> @tensor.Tensor raise @tensor.TensorError {
  let a_shape = a.shape()
  let b_shape = b.shape()
  let out_shape = @tensor.broadcast_shape(a_shape, b_shape)
  let total = @tensor.num_elements(out_shape)

  let a_buf = a.to_f32_buffer()
  let b_buf = b.to_f32_buffer()
  let out = FixedArray::make(total, (0.0 : Float))

  // 快速路径 1：同形状连续张量（支持 V128 SIMD 分派）
  if a_shape == b_shape {
    if use_simd {
      match op {
        BAdd => @vecmath.add_f32_v128(a_buf, b_buf, out, total)
        BSub => @vecmath.sub_f32_v128(a_buf, b_buf, out, total)
        BMul => @vecmath.mul_f32_v128(a_buf, b_buf, out, total)
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
/// ONNX Add 算子（支持多向广播与可选 V128 SIMD 加速）。
pub fn add_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BAdd, a, b, use_simd)
}

///|
/// ONNX Sub 算子（支持多向广播与可选 V128 SIMD 加速）。
pub fn sub_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BSub, a, b, use_simd)
}

///|
/// ONNX Mul 算子（支持多向广播与可选 V128 SIMD 加速）。
pub fn mul_f32(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  use_simd~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  binary_f32(BMul, a, b, use_simd)
}

///|
/// ONNX Relu 算子（支持标量与 V128 `f32x4_pmax` 向量化路径）。
pub fn relu_f32(
  x : @tensor.Tensor,
  use_simd~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  let src = x.to_f32_buffer()
  let n = src.length()
  let out = FixedArray::make(n, (0.0 : Float))
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
/// ONNX Softmax 算子（数值稳定版，支持任意正负轴）。
pub fn softmax_f32(
  x : @tensor.Tensor,
  axis : Int32,
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
  let out = FixedArray::make(src.length(), (0.0 : Float))

  for o = 0; o < outer; o = o + 1 {
    let outer_base = o * axis_dim * inner
    for in_idx = 0; in_idx < inner; in_idx = in_idx + 1 {
      let mut max_val = src[outer_base + in_idx]
      for k = 1; k < axis_dim; k = k + 1 {
        let v = src[outer_base + k * inner + in_idx]
        if v > max_val {
          max_val = v
        }
      }
      let mut sum_exp = 0.0
      for k = 0; k < axis_dim; k = k + 1 {
        let idx = outer_base + k * inner + in_idx
        let e = @math.exp((src[idx] - max_val).to_double())
        out[idx] = Float::from_double(e)
        sum_exp = sum_exp + e
      }
      let inv_sum = Float::from_double(1.0 / sum_exp)
      for k = 0; k < axis_dim; k = k + 1 {
        let idx = outer_base + k * inner + in_idx
        out[idx] = out[idx] * inv_sum
      }
    }
  }
  @tensor.Tensor::from_f32(shape, out)
}
