///|
/// ONNX `Shape` 算子：返回输入张量形状的 1D `Int64` 张量，支持 Opset 15+ 的 `start`/`end` 切片属性。
pub fn shape_op(
  data : @tensor.Tensor,
  start~ : Int32? = None,
  end~ : Int32? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let dims = data.shape()
  let rank = dims.length()
  let s = match start {
    Some(v) => {
      let norm = if v < 0 { v + rank } else { v }
      if norm < 0 {
        0
      } else if norm > rank {
        rank
      } else {
        norm
      }
    }
    None => 0
  }
  let e = match end {
    Some(v) => {
      let norm = if v < 0 { v + rank } else { v }
      if norm < 0 {
        0
      } else if norm > rank {
        rank
      } else {
        norm
      }
    }
    None => rank
  }
  let len = if e > s { e - s } else { 0 }
  let out = FixedArray::makei(len, fn(i) { dims[s + i].to_int64() })
  @tensor.Tensor::from_i64([len], out)
}

///|
/// ONNX `Cast` 算子：在 `Float32` / `Float64` / `Int64` / `Int32` / `Bool` 之间执行逐元素类型转换。
pub fn cast_op(
  x : @tensor.Tensor,
  to : @tensor.DType,
) -> @tensor.Tensor raise @tensor.TensorError {
  let c = x.contiguous()
  let shape = c.shape()
  let n = c.numel()
  match to {
    @tensor.Float32 => {
      let out = FixedArray::make(n, (0.0 : Float))
      match c.data {
        @tensor.F32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i]
          }
        @tensor.F64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = Float::from_double(src[c.offset + i])
          }
        @tensor.I64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = Float::from_double(src[c.offset + i].to_double())
          }
        @tensor.I32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = Float::from_int(src[c.offset + i])
          }
        @tensor.Bool(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = if src[c.offset + i] {
              (1.0 : Float)
            } else {
              (0.0 : Float)
            }
          }
      }
      @tensor.Tensor::from_f32(shape, out)
    }
    @tensor.Float64 => {
      let out = FixedArray::make(n, 0.0)
      match c.data {
        @tensor.F32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_double()
          }
        @tensor.F64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i]
          }
        @tensor.I64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_double()
          }
        @tensor.I32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_double()
          }
        @tensor.Bool(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = if src[c.offset + i] { 1.0 } else { 0.0 }
          }
      }
      @tensor.Tensor::from_f64(shape, out)
    }
    @tensor.Int64 => {
      let out = FixedArray::make(n, (0 : Int64))
      match c.data {
        @tensor.F32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_double().to_int64()
          }
        @tensor.F64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_int64()
          }
        @tensor.I64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i]
          }
        @tensor.I32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_int64()
          }
        @tensor.Bool(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = if src[c.offset + i] { (1 : Int64) } else { (0 : Int64) }
          }
      }
      @tensor.Tensor::from_i64(shape, out)
    }
    @tensor.Int32 => {
      let out = FixedArray::make(n, 0)
      match c.data {
        @tensor.F32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_double().to_int()
          }
        @tensor.F64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_int()
          }
        @tensor.I64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i].to_int()
          }
        @tensor.I32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i]
          }
        @tensor.Bool(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = if src[c.offset + i] { 1 } else { 0 }
          }
      }
      @tensor.Tensor::from_i32(shape, out)
    }
    @tensor.Bool => {
      let out = FixedArray::make(n, false)
      match c.data {
        @tensor.F32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i] != (0.0 : Float)
          }
        @tensor.F64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i] != 0.0
          }
        @tensor.I64(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i] != (0 : Int64)
          }
        @tensor.I32(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i] != 0
          }
        @tensor.Bool(src) =>
          for i = 0; i < n; i = i + 1 {
            out[i] = src[c.offset + i]
          }
      }
      @tensor.Tensor::from_bool(shape, out)
    }
  }
}

///|
/// ONNX `Concat` 算子：沿指定 `axis` 拼接多个同秩同类型张量（支持 `Float32` 与 `Int64`）。
pub fn concat_op(
  tensors : Array[@tensor.Tensor],
  axis : Int32,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  if tensors.length() == 0 {
    raise @tensor.InvalidShape("concat_op requires at least one input tensor")
  }
  let first = tensors[0]
  let ref_shape = first.shape()
  let rank = ref_shape.length()
  let ax = @tensor.normalize_axis(axis, rank)
  let dtype = first.dtype()

  // 校验所有输入张量秩、数据类型与非拼接轴维度一致，并累计拼接轴总长度
  let mut total_axis = 0
  for t_idx = 0; t_idx < tensors.length(); t_idx = t_idx + 1 {
    let t = tensors[t_idx]
    if t.dtype() != dtype {
      raise @tensor.DTypeMismatch(
        "concat_op: input \{t_idx} dtype \{t.dtype()} != \{dtype}",
      )
    }
    let s = t.shape()
    if s.length() != rank {
      raise @tensor.ShapeMismatch(
        "concat_op: input \{t_idx} rank \{s.length()} != \{rank}",
      )
    }
    for d = 0; d < rank; d = d + 1 {
      if d != ax && s[d] != ref_shape[d] {
        raise @tensor.ShapeMismatch(
          "concat_op: dim \{d} mismatch (\{s[d]} vs \{ref_shape[d]})",
        )
      }
    }
    total_axis = total_axis + s[ax]
  }

  let out_shape = ref_shape.copy()
  out_shape[ax] = total_axis
  let mut outer = 1
  for d = 0; d < ax; d = d + 1 {
    outer = outer * ref_shape[d]
  }
  let mut inner = 1
  for d = ax + 1; d < rank; d = d + 1 {
    inner = inner * ref_shape[d]
  }
  let total = outer * total_axis * inner

  match dtype {
    @tensor.Float32 => {
      let out = alloc_f32(pool, total)
      let bufs = Array::makei(tensors.length(), i => tensors[i].to_f32_buffer())
      for o = 0; o < outer; o = o + 1 {
        let mut dst_off = o * total_axis * inner
        for t_idx = 0; t_idx < tensors.length(); t_idx = t_idx + 1 {
          let chunk = tensors[t_idx].shape()[ax] * inner
          let src_off = o * chunk
          let src = bufs[t_idx]
          for k = 0; k < chunk; k = k + 1 {
            out[dst_off + k] = src[src_off + k]
          }
          dst_off = dst_off + chunk
        }
      }
      @tensor.Tensor::from_f32(out_shape, out)
    }
    @tensor.Int64 => {
      let out = FixedArray::make(total, (0 : Int64))
      let bufs = Array::makei(tensors.length(), i => tensors[i].to_i64_buffer())
      for o = 0; o < outer; o = o + 1 {
        let mut dst_off = o * total_axis * inner
        for t_idx = 0; t_idx < tensors.length(); t_idx = t_idx + 1 {
          let chunk = tensors[t_idx].shape()[ax] * inner
          let src_off = o * chunk
          let src = bufs[t_idx]
          for k = 0; k < chunk; k = k + 1 {
            out[dst_off + k] = src[src_off + k]
          }
          dst_off = dst_off + chunk
        }
      }
      @tensor.Tensor::from_i64(out_shape, out)
    }
    other =>
      raise @tensor.Unsupported(
        "concat_op: unsupported dtype \{other} (supports Float32/Int64)",
      )
  }
}

///|
/// ONNX `Split` 算子：沿 `axis` 将输入张量切分为多个子张量（兼容 Opset 2~18 的 `split` / `num_outputs` 规范）。
pub fn split_op(
  x : @tensor.Tensor,
  axis : Int32,
  attr_split : Array[Int32]?,
  split_tensor : @tensor.Tensor?,
  num_outputs : Int32,
  pool~ : @tensor.BufferPool? = None,
) -> Array[@tensor.Tensor] raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  let ax = @tensor.normalize_axis(axis, rank)
  let axis_dim = shape[ax]

  // 决定各输出分片沿 axis 的长度
  let sizes : Array[Int32] = match split_tensor {
    Some(st) => tensor_to_int_array(st)
    None =>
      match attr_split {
        Some(sp) => sp
        None => {
          if num_outputs <= 0 {
            raise @tensor.InvalidShape(
              "split_op: num_outputs must be positive, got \{num_outputs}",
            )
          }
          // Opset 18 规范：无法整除时最后一个分片取余数，其余分片取 ceil(axis_dim / num_outputs)
          let base = (axis_dim + num_outputs - 1) / num_outputs
          let out_sizes : Array[Int32] = []
          let mut rem = axis_dim
          for i = 0; i < num_outputs; i = i + 1 {
            let sz = if rem >= base { base } else { rem }
            out_sizes.push(sz)
            rem = rem - sz
          }
          out_sizes
        }
      }
  }

  let mut sum = 0
  for i = 0; i < sizes.length(); i = i + 1 {
    if sizes[i] < 0 {
      raise @tensor.InvalidShape("split_op: negative split size \{sizes[i]}")
    }
    sum = sum + sizes[i]
  }
  if sum != axis_dim {
    raise @tensor.ShapeMismatch(
      "split_op: sum of split sizes (\{sum}) != axis dim (\{axis_dim})",
    )
  }

  let mut outer = 1
  for d = 0; d < ax; d = d + 1 {
    outer = outer * shape[d]
  }
  let mut inner = 1
  for d = ax + 1; d < rank; d = d + 1 {
    inner = inner * shape[d]
  }

  let results : Array[@tensor.Tensor] = []
  match x.dtype() {
    @tensor.Float32 => {
      let src = x.to_f32_buffer()
      let mut axis_off = 0
      for p = 0; p < sizes.length(); p = p + 1 {
        let part_dim = sizes[p]
        let out_shape = shape.copy()
        out_shape[ax] = part_dim
        let part_inner = part_dim * inner
        let out = alloc_f32(pool, outer * part_inner)
        for o = 0; o < outer; o = o + 1 {
          let src_base = (o * axis_dim + axis_off) * inner
          let dst_base = o * part_inner
          for k = 0; k < part_inner; k = k + 1 {
            out[dst_base + k] = src[src_base + k]
          }
        }
        results.push(@tensor.Tensor::from_f32(out_shape, out))
        axis_off = axis_off + part_dim
      }
    }
    @tensor.Int64 => {
      let src = x.to_i64_buffer()
      let mut axis_off = 0
      for p = 0; p < sizes.length(); p = p + 1 {
        let part_dim = sizes[p]
        let out_shape = shape.copy()
        out_shape[ax] = part_dim
        let part_inner = part_dim * inner
        let out = FixedArray::make(outer * part_inner, (0 : Int64))
        for o = 0; o < outer; o = o + 1 {
          let src_base = (o * axis_dim + axis_off) * inner
          let dst_base = o * part_inner
          for k = 0; k < part_inner; k = k + 1 {
            out[dst_base + k] = src[src_base + k]
          }
        }
        results.push(@tensor.Tensor::from_i64(out_shape, out))
        axis_off = axis_off + part_dim
      }
    }
    other =>
      raise @tensor.Unsupported(
        "split_op: unsupported dtype \{other} (supports Float32/Int64)",
      )
  }
  results
}

///|
/// 将 `Int64` 切片端点按 ONNX 规范（支持负索引与大值钳位）裁剪到合法范围。
fn clamp_slice_bound(v : Int64, dim : Int32, step : Int32) -> Int32 {
  let d64 = dim.to_int64()
  let norm = if v < (0 : Int64) { v + d64 } else { v }
  if step > 0 {
    if norm < (0 : Int64) {
      0
    } else if norm > d64 {
      dim
    } else {
      norm.to_int()
    }
  } else if norm < (-1 : Int64) {
    -1
  } else if norm >= d64 {
    dim - 1
  } else {
    norm.to_int()
  }
}

///|
/// ONNX `Slice` 算子（兼容 Opset < 10 属性与 Opset >= 10 张量输入，支持负索引与负步长）。
pub fn slice_op(
  x : @tensor.Tensor,
  starts_in : Array[Int64],
  ends_in : Array[Int64],
  axes_in : Array[Int32]?,
  steps_in : Array[Int32]?,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  let num_slices = starts_in.length()
  if ends_in.length() != num_slices {
    raise @tensor.InvalidShape(
      "slice_op: starts length (\{num_slices}) != ends length (\{ends_in.length()})",
    )
  }

  let axis_start = Array::make(rank, 0)
  let axis_step = Array::make(rank, 1)
  let out_shape = shape.copy()

  for k = 0; k < num_slices; k = k + 1 {
    let raw_ax = match axes_in {
      Some(a) => a[k]
      None => k
    }
    let ax = @tensor.normalize_axis(raw_ax, rank)
    let step = match steps_in {
      Some(st) => st[k]
      None => 1
    }
    if step == 0 {
      raise @tensor.InvalidShape("slice_op: step cannot be 0")
    }
    let dim = shape[ax]
    let s = clamp_slice_bound(starts_in[k], dim, step)
    let e = clamp_slice_bound(ends_in[k], dim, step)
    let out_dim = if step > 0 {
      if e > s {
        (e - s + step - 1) / step
      } else {
        0
      }
    } else if s > e {
      let neg_step = -step
      (s - e + neg_step - 1) / neg_step
    } else {
      0
    }
    axis_start[ax] = s
    axis_step[ax] = step
    out_shape[ax] = out_dim
  }

  let total = @tensor.num_elements(out_shape)
  let in_strides = @tensor.compute_strides(shape)
  match x.dtype() {
    @tensor.Float32 => {
      let src = x.to_f32_buffer()
      let out = alloc_f32(pool, total)
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut src_idx = 0
        for d = rank - 1; d >= 0; d = d - 1 {
          let coord = rem % out_shape[d]
          rem = rem / out_shape[d]
          let in_coord = axis_start[d] + coord * axis_step[d]
          src_idx = src_idx + in_coord * in_strides[d]
        }
        out[linear] = src[src_idx]
      }
      @tensor.Tensor::from_f32(out_shape, out)
    }
    @tensor.Int64 => {
      let src = x.to_i64_buffer()
      let out = FixedArray::make(total, (0 : Int64))
      for linear = 0; linear < total; linear = linear + 1 {
        let mut rem = linear
        let mut src_idx = 0
        for d = rank - 1; d >= 0; d = d - 1 {
          let coord = rem % out_shape[d]
          rem = rem / out_shape[d]
          let in_coord = axis_start[d] + coord * axis_step[d]
          src_idx = src_idx + in_coord * in_strides[d]
        }
        out[linear] = src[src_idx]
      }
      @tensor.Tensor::from_i64(out_shape, out)
    }
    other =>
      raise @tensor.Unsupported(
        "slice_op: unsupported dtype \{other} (supports Float32/Int64)",
      )
  }
}

///|
/// 将越界坐标按 `reflect` 模式镜像映射回 `[0, dim - 1]` 合法区间。
fn reflect_index(idx : Int32, dim : Int32) -> Int32 {
  if dim <= 1 {
    return 0
  }
  let mut cur = idx
  while cur < 0 || cur >= dim {
    if cur < 0 {
      cur = -cur
    }
    if cur >= dim {
      cur = 2 * dim - 2 - cur
    }
  }
  cur
}

///|
/// ONNX `Pad` 算子：支持 `constant` / `reflect` / `edge` 三种填充模式及任意秩张量（含 Opset 18 `axes`）。
pub fn pad_f32(
  x : @tensor.Tensor,
  pads_in : Array[Int32],
  mode : String,
  constant_value : Float,
  axes_in : Array[Int32]?,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  let pad_begin = Array::make(rank, 0)
  let pad_end = Array::make(rank, 0)

  match axes_in {
    Some(axes) => {
      let k = axes.length()
      if pads_in.length() != 2 * k {
        raise @tensor.InvalidShape(
          "pad_f32: pads length (\{pads_in.length()}) != 2 * axes length (\{2 * k})",
        )
      }
      for i = 0; i < k; i = i + 1 {
        let ax = @tensor.normalize_axis(axes[i], rank)
        pad_begin[ax] = pads_in[i]
        pad_end[ax] = pads_in[k + i]
      }
    }
    None => {
      if pads_in.length() != 2 * rank {
        raise @tensor.InvalidShape(
          "pad_f32: pads length (\{pads_in.length()}) != 2 * rank (\{2 * rank})",
        )
      }
      for d = 0; d < rank; d = d + 1 {
        pad_begin[d] = pads_in[d]
        pad_end[d] = pads_in[rank + d]
      }
    }
  }

  let out_shape = Array::makei(rank, fn(d) {
    shape[d] + pad_begin[d] + pad_end[d]
  })
  for d = 0; d < rank; d = d + 1 {
    if out_shape[d] < 0 {
      raise @tensor.InvalidShape("pad_f32: negative output dim at axis \{d}")
    }
  }

  let total = @tensor.num_elements(out_shape)
  let src = x.to_f32_buffer()
  let in_strides = @tensor.compute_strides(shape)
  let out = alloc_f32(pool, total)

  let mode_id = match mode {
    "" | "constant" => 0
    "reflect" => 1
    "edge" => 2
    other => raise @tensor.Unsupported("pad_f32: unsupported mode '\{other}'")
  }

  for linear = 0; linear < total; linear = linear + 1 {
    let mut rem = linear
    let mut src_idx = 0
    let mut is_oob = false
    for d = rank - 1; d >= 0; d = d - 1 {
      let coord = rem % out_shape[d]
      rem = rem / out_shape[d]
      let in_c = coord - pad_begin[d]
      let dim = shape[d]
      if in_c < 0 || in_c >= dim {
        if mode_id == 0 {
          is_oob = true
          break
        } else if mode_id == 1 {
          src_idx = src_idx + reflect_index(in_c, dim) * in_strides[d]
        } else {
          let clamped = if in_c < 0 { 0 } else { dim - 1 }
          src_idx = src_idx + clamped * in_strides[d]
        }
      } else {
        src_idx = src_idx + in_c * in_strides[d]
      }
    }
    out[linear] = if is_oob { constant_value } else { src[src_idx] }
  }
  @tensor.Tensor::from_f32(out_shape, out)
}

///|
/// ONNX `Gather` 算子：沿指定 `axis` 按 `indices` 张量索引切片（同时支持 `Float32` 嵌入查表与 `Int64` 形状索引）。
pub fn gather_op(
  data : @tensor.Tensor,
  indices : @tensor.Tensor,
  axis : Int32,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let d_shape = data.shape()
  let d_rank = d_shape.length()
  let ax = @tensor.normalize_axis(axis, d_rank)
  let axis_dim = d_shape[ax]

  let i_shape = indices.shape()
  let i_buf = indices.to_i64_buffer()
  let i_numel = i_buf.length()

  // 输出形状 = data.shape[:ax] + indices.shape + data.shape[ax+1:]
  let out_shape : Array[Int32] = []
  for d = 0; d < ax; d = d + 1 {
    out_shape.push(d_shape[d])
  }
  for d = 0; d < i_shape.length(); d = d + 1 {
    out_shape.push(i_shape[d])
  }
  for d = ax + 1; d < d_rank; d = d + 1 {
    out_shape.push(d_shape[d])
  }

  let mut outer = 1
  for d = 0; d < ax; d = d + 1 {
    outer = outer * d_shape[d]
  }
  let mut inner = 1
  for d = ax + 1; d < d_rank; d = d + 1 {
    inner = inner * d_shape[d]
  }

  // 预归一化并校验全部索引，避免内层循环重复边界检查
  let norm_idx = FixedArray::make(i_numel, 0)
  for k = 0; k < i_numel; k = k + 1 {
    let raw = i_buf[k].to_int()
    let idx = if raw < 0 { raw + axis_dim } else { raw }
    if idx < 0 || idx >= axis_dim {
      raise @tensor.IndexOutOfBounds(
        "gather_op: index \{raw} out of bounds for axis dim \{axis_dim}",
      )
    }
    norm_idx[k] = idx
  }

  let total = outer * i_numel * inner
  match data.dtype() {
    @tensor.Float32 => {
      let src = data.to_f32_buffer()
      let out = alloc_f32(pool, total)
      for o = 0; o < outer; o = o + 1 {
        let src_outer = o * axis_dim * inner
        let dst_outer = o * i_numel * inner
        for k = 0; k < i_numel; k = k + 1 {
          let s_off = src_outer + norm_idx[k] * inner
          let d_off = dst_outer + k * inner
          for j = 0; j < inner; j = j + 1 {
            out[d_off + j] = src[s_off + j]
          }
        }
      }
      @tensor.Tensor::from_f32(out_shape, out)
    }
    @tensor.Int64 => {
      let src = data.to_i64_buffer()
      let out = FixedArray::make(total, (0 : Int64))
      for o = 0; o < outer; o = o + 1 {
        let src_outer = o * axis_dim * inner
        let dst_outer = o * i_numel * inner
        for k = 0; k < i_numel; k = k + 1 {
          let s_off = src_outer + norm_idx[k] * inner
          let d_off = dst_outer + k * inner
          for j = 0; j < inner; j = j + 1 {
            out[d_off + j] = src[s_off + j]
          }
        }
      }
      @tensor.Tensor::from_i64(out_shape, out)
    }
    other =>
      raise @tensor.Unsupported(
        "gather_op: unsupported data dtype \{other} (supports Float32/Int64)",
      )
  }
}
