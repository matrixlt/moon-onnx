///|
/// ONNX `QuantizeLinear` 算子：将 `Float32` 张量按每张量（per-tensor）或每通道（per-axis）线性量化为 `UInt8` 或 `Int8`。
/// - 若省略 `y_zero_point`，按 ONNX 规范默认输出 `UInt8` 且零点为 `0`。
/// - 若提供 `y_zero_point`，输出数据类型与 `y_zero_point` 保持一致（`UInt8` 或 `Int8`）。
pub fn quantize_linear_op(
  x : @tensor.Tensor,
  y_scale : @tensor.Tensor,
  y_zero_point : @tensor.Tensor?,
  axis~ : Int32 = 1,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let src = x.to_f32_buffer()
  let n = src.length()
  let scales = y_scale.to_f32_buffer()

  let (out_dtype, zps) : (@tensor.DType, FixedArray[Byte]) = match
    y_zero_point {
    Some(zp) =>
      match zp.dtype() {
        @tensor.UInt8 => (@tensor.UInt8, zp.to_u8_buffer())
        @tensor.Int8 => (@tensor.Int8, zp.to_i8_buffer())
        other =>
          raise @tensor.DTypeMismatch(
            "quantize_linear_op: y_zero_point must be UInt8 or Int8, got \{other}",
          )
      }
    None => (@tensor.UInt8, FixedArray::make(scales.length(), b'\x00'))
  }

  let out = FixedArray::make(n, b'\x00')
  // 快速路径：每张量标量量化
  if scales.length() == 1 {
    match out_dtype {
      @tensor.UInt8 =>
        @quant.quantize_u8_per_tensor(src, scales[0], zps[0], out, n)
      @tensor.Int8 =>
        @quant.quantize_i8_per_tensor(src, scales[0], zps[0], out, n)
      _ => ()
    }
  } else {
    // 每轴（per-axis）量化：沿指定 axis 索引对应通道的 scale 与 zero_point
    let rank = shape.length()
    let ax = @tensor.normalize_axis(axis, rank)
    let axis_dim = shape[ax]
    if scales.length() != axis_dim || zps.length() != axis_dim {
      raise @tensor.ShapeMismatch(
        "quantize_linear_op: per-axis scale/zp length (\{scales.length()}) != axis dim (\{axis_dim})",
      )
    }
    let mut inner = 1
    for d = ax + 1; d < rank; d = d + 1 {
      inner = inner * shape[d]
    }
    for i = 0; i < n; i = i + 1 {
      let c = i / inner % axis_dim
      let inv_s = 1.0 / scales[c].to_double()
      match out_dtype {
        @tensor.UInt8 => {
          let q = @quant.round_half_to_even(src[i].to_double() * inv_s) +
            zps[c].to_int()
          out[i] = @quant.u8_to_byte(q)
        }
        @tensor.Int8 => {
          let q = @quant.round_half_to_even(src[i].to_double() * inv_s) +
            @quant.byte_to_i8(zps[c])
          out[i] = @quant.i8_to_byte(q)
        }
        _ => ()
      }
    }
  }

  match out_dtype {
    @tensor.UInt8 => @tensor.Tensor::from_u8(shape, out)
    @tensor.Int8 => @tensor.Tensor::from_i8(shape, out)
    _ => raise @tensor.Unsupported("quantize_linear_op: invalid output dtype")
  }
}

///|
/// ONNX `DequantizeLinear` 算子：将 `UInt8` / `Int8` / `Int32` 量化张量反量化为 `Float32`（支持每张量与每通道）。
pub fn dequantize_linear_op(
  x : @tensor.Tensor,
  x_scale : @tensor.Tensor,
  x_zero_point : @tensor.Tensor?,
  axis~ : Int32 = 1,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let n = x.numel()
  let scales = x_scale.to_f32_buffer()
  let out = alloc_f32(pool, n)

  match x.dtype() {
    @tensor.UInt8 => {
      let src = x.to_u8_buffer()
      let zps = match x_zero_point {
        Some(zp) => zp.to_u8_buffer()
        None => FixedArray::make(scales.length(), b'\x00')
      }
      if scales.length() == 1 {
        @quant.dequantize_u8_per_tensor(src, scales[0], zps[0], out, n)
      } else {
        let rank = shape.length()
        let ax = @tensor.normalize_axis(axis, rank)
        let axis_dim = shape[ax]
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * shape[d]
        }
        for i = 0; i < n; i = i + 1 {
          let c = i / inner % axis_dim
          let v = (src[i].to_int() - zps[c].to_int()).to_double() *
            scales[c].to_double()
          out[i] = Float::from_double(v)
        }
      }
    }
    @tensor.Int8 => {
      let src = x.to_i8_buffer()
      let zps = match x_zero_point {
        Some(zp) => zp.to_i8_buffer()
        None => FixedArray::make(scales.length(), b'\x00')
      }
      if scales.length() == 1 {
        @quant.dequantize_i8_per_tensor(src, scales[0], zps[0], out, n)
      } else {
        let rank = shape.length()
        let ax = @tensor.normalize_axis(axis, rank)
        let axis_dim = shape[ax]
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * shape[d]
        }
        for i = 0; i < n; i = i + 1 {
          let c = i / inner % axis_dim
          let v = (@quant.byte_to_i8(src[i]) - @quant.byte_to_i8(zps[c])).to_double() *
            scales[c].to_double()
          out[i] = Float::from_double(v)
        }
      }
    }
    @tensor.Int32 => {
      // ONNX 规范允许 Int32 偏置反量化（零点恒为 0）
      let src = x.to_i64_buffer()
      if scales.length() == 1 {
        let s = scales[0].to_double()
        for i = 0; i < n; i = i + 1 {
          out[i] = Float::from_double(src[i].to_double() * s)
        }
      } else {
        let rank = shape.length()
        let ax = @tensor.normalize_axis(axis, rank)
        let axis_dim = shape[ax]
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * shape[d]
        }
        for i = 0; i < n; i = i + 1 {
          let c = i / inner % axis_dim
          out[i] = Float::from_double(
            src[i].to_double() * scales[c].to_double(),
          )
        }
      }
    }
    other =>
      raise @tensor.DTypeMismatch(
        "dequantize_linear_op: expected UInt8/Int8/Int32, got \{other}",
      )
  }
  @tensor.Tensor::from_f32(shape, out)
}

///|
/// ONNX `QLinearMatMul` 算子：8 输入端到端量化矩阵乘法（支持 2D `Int8` 与 `UInt8`，`use_simd=true` 时走 V128 `i32x4.dot_i16x8_s`）。
pub fn qlinear_matmul_op(
  a : @tensor.Tensor,
  a_scale : @tensor.Tensor,
  a_zp : @tensor.Tensor,
  b : @tensor.Tensor,
  b_scale : @tensor.Tensor,
  b_zp : @tensor.Tensor,
  y_scale : @tensor.Tensor,
  y_zp : @tensor.Tensor,
  use_simd~ : Bool = false,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sa = a.shape()
  let sb = b.shape()
  if sa.length() != 2 || sb.length() != 2 {
    raise @tensor.InvalidShape(
      "qlinear_matmul_op currently requires 2D inputs, got \{@tensor.format_shape(sa)} and \{@tensor.format_shape(sb)}",
    )
  }
  let m = sa[0]
  let k = sa[1]
  let kb = sb[0]
  let n = sb[1]
  if k != kb {
    raise @tensor.ShapeMismatch(
      "qlinear_matmul_op: inner dim mismatch (\{k} vs \{kb})",
    )
  }
  if a.dtype() != b.dtype() {
    raise @tensor.DTypeMismatch(
      "qlinear_matmul_op: a dtype (\{a.dtype()}) != b dtype (\{b.dtype()})",
    )
  }
  let as_v = a_scale.get_f32_flat(0)
  let bs_v = b_scale.get_f32_flat(0)
  let ys_v = y_scale.get_f32_flat(0)
  let out = FixedArray::make(m * n, b'\x00')

  match a.dtype() {
    @tensor.Int8 => {
      let ab = a.to_i8_buffer()
      let bb = b.to_i8_buffer()
      let az = a_zp.to_i8_buffer()[0]
      let bz = b_zp.to_i8_buffer()[0]
      let yz = y_zp.to_i8_buffer()[0]
      if use_simd {
        @quant.qlinear_matmul_i8_v128(
          ab, as_v, az, bb, bs_v, bz, ys_v, yz, out, m, k, n,
        )
      } else {
        @quant.qlinear_matmul_i8_scalar(
          ab, as_v, az, bb, bs_v, bz, ys_v, yz, out, m, k, n,
        )
      }
      @tensor.Tensor::from_i8([m, n], out)
    }
    @tensor.UInt8 => {
      let ab = a.to_u8_buffer()
      let bb = b.to_u8_buffer()
      let az = a_zp.to_u8_buffer()[0]
      let bz = b_zp.to_u8_buffer()[0]
      let yz = y_zp.to_u8_buffer()[0]
      if use_simd {
        @quant.qlinear_matmul_u8_v128(
          ab, as_v, az, bb, bs_v, bz, ys_v, yz, out, m, k, n,
        )
      } else {
        @quant.qlinear_matmul_u8_scalar(
          ab, as_v, az, bb, bs_v, bz, ys_v, yz, out, m, k, n,
        )
      }
      @tensor.Tensor::from_u8([m, n], out)
    }
    other =>
      raise @tensor.DTypeMismatch(
        "qlinear_matmul_op: expected Int8 or UInt8, got \{other}",
      )
  }
}

///|
/// ONNX `DynamicQuantizeLinear` 算子：动态量化 `Float32` 激活为 `(y: UInt8, y_scale: Float32[], y_zero_point: UInt8[])`。
pub fn dynamic_quantize_linear_op(
  x : @tensor.Tensor,
) -> (@tensor.Tensor, @tensor.Tensor, @tensor.Tensor) raise @tensor.TensorError {
  let shape = x.shape()
  let src = x.to_f32_buffer()
  let n = src.length()
  let dst = FixedArray::make(n, b'\x00')
  let (scale, zp) = @quant.dynamic_quantize_linear_u8(src, 0, dst, 0, n)
  let y = @tensor.Tensor::from_u8(shape, dst)
  let y_scale = @tensor.Tensor::from_f32([], FixedArray::make(1, scale))
  let y_zp = @tensor.Tensor::from_u8([], FixedArray::make(1, zp))
  (y, y_scale, y_zp)
}

///|
/// 预打包加速版 ONNX `MatMulInteger`：`A_u8[..., M, K] @ PackedInt8MatrixB[K, N] -> Int32[..., M, N]`。
/// 支持 `A` 任意 `rank >= 2` 与标量/每列 `[N]` `b_zero_point`，直接复用 V128 `i32x4.dot_i16x8_s` 硬件点积。
pub fn matmul_integer_prepacked_op(
  a : @tensor.Tensor,
  packed_b : @quant.PackedInt8MatrixB,
  a_zp : @tensor.Tensor?,
  b_zp : @tensor.Tensor?,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sa = a.shape()
  let rank = sa.length()
  if rank < 2 {
    raise @tensor.InvalidShape(
      "matmul_integer_prepacked_op requires rank >= 2, got \{@tensor.format_shape(sa)}",
    )
  }
  let (kb, n) = packed_b.dims()
  let k = sa[rank - 1]
  if k != kb {
    raise @tensor.ShapeMismatch(
      "matmul_integer_prepacked_op: inner dim mismatch (\{k} vs \{kb})",
    )
  }
  let mut m_tot = 1
  let out_shape : Array[Int32] = []
  for d = 0; d < rank - 1; d = d + 1 {
    m_tot = m_tot * sa[d]
    out_shape.push(sa[d])
  }
  out_shape.push(n)

  let ab = a.to_u8_buffer()
  let az : Byte = match a_zp {
    Some(t) => t.to_u8_buffer()[0]
    None => b'\x00'
  }
  let bz_opt : FixedArray[Byte]? = match b_zp {
    Some(t) => Some(t.to_i8_buffer())
    None => None
  }
  let out = FixedArray::make(m_tot * n, 0)
  @quant.matmul_integer_u8_i8_prepacked_v128(
    ab, 0, az, packed_b, bz_opt, out, 0, m_tot,
  )
  @tensor.Tensor::from_i32(out_shape, out)
}

///|
/// 通用 ONNX `MatMulInteger` 算子：`A[..., M, K] @ B[K, N] -> Int32[..., M, N]`。
/// 支持 `UInt8 × Int8`（走预打包 V128 快速路径）、`UInt8 × UInt8` 与 `Int8 × Int8`，并支持标量或每列 `[N]` 零点。
pub fn matmul_integer_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
  a_zp : @tensor.Tensor?,
  b_zp : @tensor.Tensor?,
  use_simd~ : Bool = true,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sa = a.shape()
  let sb = b.shape()
  if sa.length() < 2 || sb.length() != 2 {
    raise @tensor.InvalidShape(
      "matmul_integer_op requires A rank>=2 and B 2D, got \{@tensor.format_shape(sa)} and \{@tensor.format_shape(sb)}",
    )
  }
  let rank = sa.length()
  let k = sa[rank - 1]
  let kb = sb[0]
  let n = sb[1]
  if k != kb {
    raise @tensor.ShapeMismatch(
      "matmul_integer_op: inner dim mismatch (\{k} vs \{kb})",
    )
  }
  let mut m_tot = 1
  let out_shape : Array[Int32] = []
  for d = 0; d < rank - 1; d = d + 1 {
    m_tot = m_tot * sa[d]
    out_shape.push(sa[d])
  }
  out_shape.push(n)

  match (a.dtype(), b.dtype()) {
    (@tensor.UInt8, @tensor.Int8) if use_simd => {
      let packed = @quant.pack_int8_matrix_b(b.to_i8_buffer(), k, n)
      matmul_integer_prepacked_op(a, packed, a_zp, b_zp)
    }
    (@tensor.UInt8, @tensor.Int8) => {
      let ab = a.to_u8_buffer()
      let bb = b.to_i8_buffer()
      let az = match a_zp {
        Some(t) => t.to_u8_buffer()[0].to_int()
        None => 0
      }
      let bzs : FixedArray[Byte]? = match b_zp {
        Some(t) => Some(t.to_i8_buffer())
        None => None
      }
      let out = FixedArray::make(m_tot * n, 0)
      for i = 0; i < m_tot; i = i + 1 {
        let ar = i * k
        let yr = i * n
        for j = 0; j < n; j = j + 1 {
          let bz = match bzs {
            None => 0
            Some(arr) =>
              if arr.length() == 1 {
                @quant.byte_to_i8(arr[0])
              } else {
                @quant.byte_to_i8(arr[j])
              }
          }
          let mut acc = 0
          for kk = 0; kk < k; kk = kk + 1 {
            acc = acc +
              (ab[ar + kk].to_int() - az) *
              (@quant.byte_to_i8(bb[kk * n + j]) - bz)
          }
          out[yr + j] = acc
        }
      }
      @tensor.Tensor::from_i32(out_shape, out)
    }
    (@tensor.UInt8, @tensor.UInt8) => {
      let ab = a.to_u8_buffer()
      let bb = b.to_u8_buffer()
      let az = match a_zp {
        Some(t) => t.to_u8_buffer()[0].to_int()
        None => 0
      }
      let bzs : FixedArray[Byte]? = match b_zp {
        Some(t) => Some(t.to_u8_buffer())
        None => None
      }
      let out = FixedArray::make(m_tot * n, 0)
      for i = 0; i < m_tot; i = i + 1 {
        let ar = i * k
        let yr = i * n
        for j = 0; j < n; j = j + 1 {
          let bz = match bzs {
            None => 0
            Some(arr) =>
              if arr.length() == 1 {
                arr[0].to_int()
              } else {
                arr[j].to_int()
              }
          }
          let mut acc = 0
          for kk = 0; kk < k; kk = kk + 1 {
            acc = acc +
              (ab[ar + kk].to_int() - az) * (bb[kk * n + j].to_int() - bz)
          }
          out[yr + j] = acc
        }
      }
      @tensor.Tensor::from_i32(out_shape, out)
    }
    (@tensor.Int8, @tensor.Int8) => {
      let ab = a.to_i8_buffer()
      let bb = b.to_i8_buffer()
      let az = match a_zp {
        Some(t) => @quant.byte_to_i8(t.to_i8_buffer()[0])
        None => 0
      }
      let bzs : FixedArray[Byte]? = match b_zp {
        Some(t) => Some(t.to_i8_buffer())
        None => None
      }
      let out = FixedArray::make(m_tot * n, 0)
      for i = 0; i < m_tot; i = i + 1 {
        let ar = i * k
        let yr = i * n
        for j = 0; j < n; j = j + 1 {
          let bz = match bzs {
            None => 0
            Some(arr) =>
              if arr.length() == 1 {
                @quant.byte_to_i8(arr[0])
              } else {
                @quant.byte_to_i8(arr[j])
              }
          }
          let mut acc = 0
          for kk = 0; kk < k; kk = kk + 1 {
            acc = acc +
              (@quant.byte_to_i8(ab[ar + kk]) - az) *
              (@quant.byte_to_i8(bb[kk * n + j]) - bz)
          }
          out[yr + j] = acc
        }
      }
      @tensor.Tensor::from_i32(out_shape, out)
    }
    (da, db) =>
      raise @tensor.DTypeMismatch(
        "matmul_integer_op: unsupported dtypes (\{da}, \{db})",
      )
  }
}
