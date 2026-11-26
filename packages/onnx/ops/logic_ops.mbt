///|
/// 将连续张量的第 `i` 个元素统一读取为 `Double`（供 `Cast` 算子通用转换使用）。
fn read_elem_double(c : @tensor.Tensor, i : Int32) -> Double {
  let idx = c.offset + i
  match c.data {
    @tensor.F32(src) => src[idx].to_double()
    @tensor.F64(src) => src[idx]
    @tensor.I64(src) => src[idx].to_double()
    @tensor.I32(src) => src[idx].to_double()
    @tensor.Bool(src) => if src[idx] { 1.0 } else { 0.0 }
    @tensor.U8(src) => src[idx].to_int().to_double()
    @tensor.I8(src) => @quant.byte_to_i8(src[idx]).to_double()
  }
}

///|
/// ONNX `Cast` 算子：在 `Float32` / `Float64` / `Int64` / `Int32` / `Bool` / `UInt8` / `Int8` 之间执行逐元素类型转换。
pub fn cast_op(
  x : @tensor.Tensor,
  to : @tensor.DType,
) -> @tensor.Tensor raise @tensor.TensorError {
  let c = x.contiguous()
  let shape = c.shape()
  let n = c.numel()
  match to {
    @tensor.Float32 => {
      let out = FixedArray::makei(n, fn(i) {
        Float::from_double(read_elem_double(c, i))
      })
      @tensor.Tensor::from_f32(shape, out)
    }
    @tensor.Float64 => {
      let out = FixedArray::makei(n, fn(i) { read_elem_double(c, i) })
      @tensor.Tensor::from_f64(shape, out)
    }
    @tensor.Int64 => {
      let out = FixedArray::makei(n, fn(i) { read_elem_double(c, i).to_int64() })
      @tensor.Tensor::from_i64(shape, out)
    }
    @tensor.Int32 => {
      let out = FixedArray::makei(n, fn(i) { read_elem_double(c, i).to_int() })
      @tensor.Tensor::from_i32(shape, out)
    }
    @tensor.Bool => {
      let out = FixedArray::makei(n, fn(i) { read_elem_double(c, i) != 0.0 })
      @tensor.Tensor::from_bool(shape, out)
    }
    @tensor.UInt8 => {
      let out = FixedArray::makei(n, fn(i) {
        (read_elem_double(c, i).to_int() & 0xFF).to_byte()
      })
      @tensor.Tensor::from_u8(shape, out)
    }
    @tensor.Int8 => {
      let out = FixedArray::makei(n, fn(i) {
        (read_elem_double(c, i).to_int() & 0xFF).to_byte()
      })
      @tensor.Tensor::from_i8(shape, out)
    }
  }
}

///|
/// 内部比较算子种类（统一复用多向广播与全类型分派逻辑）。
priv enum CmpOp {
  CmpEq
  CmpLt
  CmpGt
  CmpLe
  CmpGe
}

///|
/// 通用逐元素广播比较内核，输出 `Bool` 张量。
fn compare_op(
  kind : CmpOp,
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  if a.dtype() != b.dtype() {
    raise @tensor.DTypeMismatch(
      "compare_op: dtype mismatch (\{a.dtype()} vs \{b.dtype()})",
    )
  }
  let a_shape = a.shape()
  let b_shape = b.shape()
  let out_shape = @tensor.broadcast_shape(a_shape, b_shape)
  let total = @tensor.num_elements(out_shape)
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
  let ac = a.contiguous()
  let bc = b.contiguous()
  let out = FixedArray::make(total, false)

  for linear = 0; linear < total; linear = linear + 1 {
    let mut rem = linear
    let mut ia = ac.offset
    let mut ib = bc.offset
    for axis = rank - 1; axis >= 0; axis = axis - 1 {
      let dim = out_shape[axis]
      let coord = rem % dim
      rem = rem / dim
      ia = ia + coord * a_bstrides[axis]
      ib = ib + coord * b_bstrides[axis]
    }
    out[linear] = match (ac.data, bc.data) {
      (@tensor.F32(sa), @tensor.F32(sb)) =>
        match kind {
          CmpEq => sa[ia] == sb[ib]
          CmpLt => sa[ia] < sb[ib]
          CmpGt => sa[ia] > sb[ib]
          CmpLe => sa[ia] <= sb[ib]
          CmpGe => sa[ia] >= sb[ib]
        }
      (@tensor.F64(sa), @tensor.F64(sb)) =>
        match kind {
          CmpEq => sa[ia] == sb[ib]
          CmpLt => sa[ia] < sb[ib]
          CmpGt => sa[ia] > sb[ib]
          CmpLe => sa[ia] <= sb[ib]
          CmpGe => sa[ia] >= sb[ib]
        }
      (@tensor.I64(sa), @tensor.I64(sb)) =>
        match kind {
          CmpEq => sa[ia] == sb[ib]
          CmpLt => sa[ia] < sb[ib]
          CmpGt => sa[ia] > sb[ib]
          CmpLe => sa[ia] <= sb[ib]
          CmpGe => sa[ia] >= sb[ib]
        }
      (@tensor.I32(sa), @tensor.I32(sb)) =>
        match kind {
          CmpEq => sa[ia] == sb[ib]
          CmpLt => sa[ia] < sb[ib]
          CmpGt => sa[ia] > sb[ib]
          CmpLe => sa[ia] <= sb[ib]
          CmpGe => sa[ia] >= sb[ib]
        }
      (@tensor.Bool(sa), @tensor.Bool(sb)) =>
        match kind {
          CmpEq => sa[ia] == sb[ib]
          CmpLt => !sa[ia] && sb[ib]
          CmpGt => sa[ia] && !sb[ib]
          CmpLe => !sa[ia] || sb[ib]
          CmpGe => sa[ia] || !sb[ib]
        }
      (@tensor.U8(sa), @tensor.U8(sb)) => {
        let va = sa[ia].to_int()
        let vb = sb[ib].to_int()
        match kind {
          CmpEq => va == vb
          CmpLt => va < vb
          CmpGt => va > vb
          CmpLe => va <= vb
          CmpGe => va >= vb
        }
      }
      (@tensor.I8(sa), @tensor.I8(sb)) => {
        let va = @quant.byte_to_i8(sa[ia])
        let vb = @quant.byte_to_i8(sb[ib])
        match kind {
          CmpEq => va == vb
          CmpLt => va < vb
          CmpGt => va > vb
          CmpLe => va <= vb
          CmpGe => va >= vb
        }
      }
      _ => false
    }
  }
  @tensor.Tensor::from_bool(out_shape, out)
}

///|
/// ONNX `Equal` 算子：按 NumPy/ONNX 多向广播规则逐元素比较是否相等，输出 `Bool` 张量。
pub fn equal_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  compare_op(CmpEq, a, b)
}

///|
/// ONNX `Less` 算子：按多向广播规则逐元素计算 `a < b`，输出 `Bool` 张量。
pub fn less_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  compare_op(CmpLt, a, b)
}

///|
/// ONNX `Greater` 算子：按多向广播规则逐元素计算 `a > b`，输出 `Bool` 张量。
pub fn greater_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  compare_op(CmpGt, a, b)
}

///|
/// ONNX `LessOrEqual` 算子：按多向广播规则逐元素计算 `a <= b`，输出 `Bool` 张量。
pub fn less_or_equal_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  compare_op(CmpLe, a, b)
}

///|
/// ONNX `GreaterOrEqual` 算子：按多向广播规则逐元素计算 `a >= b`，输出 `Bool` 张量。
pub fn greater_or_equal_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  compare_op(CmpGe, a, b)
}

///|
/// ONNX `Not` 算子：对 `Bool` 张量执行逐元素逻辑取反。
pub fn not_op(x : @tensor.Tensor) -> @tensor.Tensor raise @tensor.TensorError {
  let c = x.contiguous()
  match c.data {
    @tensor.Bool(src) => {
      let n = c.numel()
      let out = FixedArray::makei(n, fn(i) { !src[c.offset + i] })
      @tensor.Tensor::from_bool(c.shape(), out)
    }
    _ =>
      raise @tensor.DTypeMismatch(
        "not_op: expected Bool tensor, got \{x.dtype()}",
      )
  }
}

///|
/// ONNX `And` 算子：对两个 `Bool` 张量执行多向广播逻辑与。
pub fn and_op(
  a : @tensor.Tensor,
  b : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  let ac = a.contiguous()
  let bc = b.contiguous()
  match (ac.data, bc.data) {
    (@tensor.Bool(sa), @tensor.Bool(sb)) => {
      let a_shape = ac.shape()
      let b_shape = bc.shape()
      let out_shape = @tensor.broadcast_shape(a_shape, b_shape)
      let total = @tensor.num_elements(out_shape)
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
      let out = FixedArray::make(total, false)
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut ia = ac.offset
        let mut ib = bc.offset
        for ax = rank - 1; ax >= 0; ax = ax - 1 {
          let dim = out_shape[ax]
          let coord = rem % dim
          rem = rem / dim
          ia = ia + coord * a_bs[ax]
          ib = ib + coord * b_bs[ax]
        }
        out[linear] = sa[ia] && sb[ib]
      }
      @tensor.Tensor::from_bool(out_shape, out)
    }
    _ =>
      raise @tensor.DTypeMismatch(
        "and_op: expected Bool inputs, got \{a.dtype()} and \{b.dtype()}",
      )
  }
}

///|
/// ONNX `Where` 算子：按三向 NumPy 广播规则执行 `cond ? x : y`（支持 Float32 / Int64 / Int32 / Bool）。
pub fn where_op(
  cond : @tensor.Tensor,
  x : @tensor.Tensor,
  y : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  if cond.dtype() != @tensor.Bool {
    raise @tensor.DTypeMismatch(
      "where_op: condition must be Bool, got \{cond.dtype()}",
    )
  }
  if x.dtype() != y.dtype() {
    raise @tensor.DTypeMismatch(
      "where_op: X and Y dtype mismatch (\{x.dtype()} vs \{y.dtype()})",
    )
  }
  let cs = cond.shape()
  let xs = x.shape()
  let ys = y.shape()
  let xy_shape = @tensor.broadcast_shape(xs, ys)
  let out_shape = @tensor.broadcast_shape(cs, xy_shape)
  let total = @tensor.num_elements(out_shape)
  let c_bs = @tensor.broadcast_strides(
    cs,
    @tensor.compute_strides(cs),
    out_shape,
  )
  let x_bs = @tensor.broadcast_strides(
    xs,
    @tensor.compute_strides(xs),
    out_shape,
  )
  let y_bs = @tensor.broadcast_strides(
    ys,
    @tensor.compute_strides(ys),
    out_shape,
  )
  let rank = out_shape.length()

  let cc = cond.contiguous()
  let xc = x.contiguous()
  let yc = y.contiguous()
  let c_buf = match cc.data {
    @tensor.Bool(b) => b
    _ => raise @tensor.DTypeMismatch("where_op: condition must be Bool")
  }

  match (xc.data, yc.data) {
    (@tensor.F32(sx), @tensor.F32(sy)) => {
      let out = alloc_f32(pool, total)
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut ic = cc.offset
        let mut ix = xc.offset
        let mut iy = yc.offset
        for ax = rank - 1; ax >= 0; ax = ax - 1 {
          let dim = out_shape[ax]
          let coord = rem % dim
          rem = rem / dim
          ic = ic + coord * c_bs[ax]
          ix = ix + coord * x_bs[ax]
          iy = iy + coord * y_bs[ax]
        }
        out[linear] = if c_buf[ic] { sx[ix] } else { sy[iy] }
      }
      @tensor.Tensor::from_f32(out_shape, out)
    }
    (@tensor.I64(sx), @tensor.I64(sy)) => {
      let out = FixedArray::make(total, (0 : Int64))
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut ic = cc.offset
        let mut ix = xc.offset
        let mut iy = yc.offset
        for ax = rank - 1; ax >= 0; ax = ax - 1 {
          let dim = out_shape[ax]
          let coord = rem % dim
          rem = rem / dim
          ic = ic + coord * c_bs[ax]
          ix = ix + coord * x_bs[ax]
          iy = iy + coord * y_bs[ax]
        }
        out[linear] = if c_buf[ic] { sx[ix] } else { sy[iy] }
      }
      @tensor.Tensor::from_i64(out_shape, out)
    }
    (@tensor.I32(sx), @tensor.I32(sy)) => {
      let out = FixedArray::make(total, 0)
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut ic = cc.offset
        let mut ix = xc.offset
        let mut iy = yc.offset
        for ax = rank - 1; ax >= 0; ax = ax - 1 {
          let dim = out_shape[ax]
          let coord = rem % dim
          rem = rem / dim
          ic = ic + coord * c_bs[ax]
          ix = ix + coord * x_bs[ax]
          iy = iy + coord * y_bs[ax]
        }
        out[linear] = if c_buf[ic] { sx[ix] } else { sy[iy] }
      }
      @tensor.Tensor::from_i32(out_shape, out)
    }
    (@tensor.Bool(sx), @tensor.Bool(sy)) => {
      let out = FixedArray::make(total, false)
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut ic = cc.offset
        let mut ix = xc.offset
        let mut iy = yc.offset
        for ax = rank - 1; ax >= 0; ax = ax - 1 {
          let dim = out_shape[ax]
          let coord = rem % dim
          rem = rem / dim
          ic = ic + coord * c_bs[ax]
          ix = ix + coord * x_bs[ax]
          iy = iy + coord * y_bs[ax]
        }
        out[linear] = if c_buf[ic] { sx[ix] } else { sy[iy] }
      }
      @tensor.Tensor::from_bool(out_shape, out)
    }
    _ =>
      raise @tensor.Unsupported(
        "where_op: unsupported dtype \{x.dtype()} (supports Float32/Int64/Int32/Bool)",
      )
  }
}

///|
/// ONNX `Range` 算子：生成等差一维张量 `[start, start + delta, ...)`，支持 `Int64` / `Int32` / `Float32`。
pub fn range_op(
  start : @tensor.Tensor,
  limit : @tensor.Tensor,
  delta : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  let sc = start.contiguous()
  let lc = limit.contiguous()
  let dc = delta.contiguous()
  if sc.numel() != 1 || lc.numel() != 1 || dc.numel() != 1 {
    raise @tensor.InvalidShape("range_op: start, limit, delta must be scalars")
  }
  match (sc.data, lc.data, dc.data) {
    (@tensor.I64(sb), @tensor.I64(lb), @tensor.I64(db)) => {
      let s = sb[sc.offset]
      let l = lb[lc.offset]
      let d = db[dc.offset]
      if d == (0 : Int64) {
        raise @tensor.InvalidShape("range_op: delta cannot be 0")
      }
      let n_raw = if d > (0 : Int64) {
        if l <= s {
          0
        } else {
          ((l - s + d - (1 : Int64)) / d).to_int()
        }
      } else if l >= s {
        0
      } else {
        let nd = -d
        ((s - l + nd - (1 : Int64)) / nd).to_int()
      }
      let out = FixedArray::makei(n_raw, fn(i) { s + i.to_int64() * d })
      @tensor.Tensor::from_i64([n_raw], out)
    }
    (@tensor.I32(sb), @tensor.I32(lb), @tensor.I32(db)) => {
      let s = sb[sc.offset]
      let l = lb[lc.offset]
      let d = db[dc.offset]
      if d == 0 {
        raise @tensor.InvalidShape("range_op: delta cannot be 0")
      }
      let n_raw = if d > 0 {
        if l <= s {
          0
        } else {
          (l - s + d - 1) / d
        }
      } else if l >= s {
        0
      } else {
        let nd = -d
        (s - l + nd - 1) / nd
      }
      let out = FixedArray::makei(n_raw, fn(i) { s + i * d })
      @tensor.Tensor::from_i32([n_raw], out)
    }
    (@tensor.F32(sb), @tensor.F32(lb), @tensor.F32(db)) => {
      let s = sb[sc.offset].to_double()
      let l = lb[lc.offset].to_double()
      let d = db[dc.offset].to_double()
      if d == 0.0 {
        raise @tensor.InvalidShape("range_op: delta cannot be 0")
      }
      let ratio = (l - s) / d
      let n_raw = if ratio <= 0.0 { 0 } else { ratio.ceil().to_int() }
      let out = FixedArray::makei(n_raw, fn(i) {
        Float::from_double(s + i.to_double() * d)
      })
      @tensor.Tensor::from_f32([n_raw], out)
    }
    _ =>
      raise @tensor.Unsupported(
        "range_op: unsupported or mismatched dtypes (\{start.dtype()}, \{limit.dtype()}, \{delta.dtype()})",
      )
  }
}

///|
/// ONNX `ConstantOfShape` 算子：根据输入 1D 形状张量分配目标形状张量，并填充 `fill_value` 的标量值。
pub fn constant_of_shape_op(
  shape_tensor : @tensor.Tensor,
  fill_value : @tensor.Tensor,
) -> @tensor.Tensor raise @tensor.TensorError {
  if shape_tensor.rank() != 1 {
    raise @tensor.InvalidShape(
      "constant_of_shape_op: shape input must be 1D, got rank \{shape_tensor.rank()}",
    )
  }
  let raw_dims = shape_tensor.to_i64_buffer()
  let out_shape = Array::makei(raw_dims.length(), fn(i) { raw_dims[i].to_int() })
  for i = 0; i < out_shape.length(); i = i + 1 {
    if out_shape[i] < 0 {
      raise @tensor.InvalidShape(
        "constant_of_shape_op: negative dimension \{out_shape[i]} at axis \{i}",
      )
    }
  }
  let total = @tensor.num_elements(out_shape)
  let vc = fill_value.contiguous()
  if vc.numel() != 1 {
    raise @tensor.InvalidShape(
      "constant_of_shape_op: value attribute must be a 1-element tensor, got numel=\{vc.numel()}",
    )
  }
  match vc.data {
    @tensor.F32(buf) =>
      @tensor.Tensor::from_f32(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
    @tensor.F64(buf) =>
      @tensor.Tensor::from_f64(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
    @tensor.I64(buf) =>
      @tensor.Tensor::from_i64(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
    @tensor.I32(buf) =>
      @tensor.Tensor::from_i32(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
    @tensor.Bool(buf) =>
      @tensor.Tensor::from_bool(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
    @tensor.U8(buf) =>
      @tensor.Tensor::from_u8(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
    @tensor.I8(buf) =>
      @tensor.Tensor::from_i8(
        out_shape,
        FixedArray::make(total, buf[vc.offset]),
      )
  }
}
