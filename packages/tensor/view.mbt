///|
/// 读取指定多维坐标处的 Float32 元素值。
pub fn Tensor::get_f32(
  self : Tensor,
  indices : Array[Int32],
) -> Float raise TensorError {
  let idx = self.flat_index(indices)
  match self.data {
    F32(buf) => buf[idx]
    _ => raise DTypeMismatch("get_f32: expected Float32, got \{self.dtype}")
  }
}

///|
/// 按逻辑行主序下标读取 Float32 元素值（透明兼容非连续视图）。
pub fn Tensor::get_f32_flat(
  self : Tensor,
  linear : Int32,
) -> Float raise TensorError {
  let n = self.numel()
  if linear < 0 || linear >= n {
    raise IndexOutOfBounds(
      "linear index \{linear} out of bounds for numel \{n}",
    )
  }
  let phys = if self.is_contiguous() {
    linear
  } else {
    self.physical_index_from_linear(linear)
  }
  match self.data {
    F32(buf) => buf[phys]
    _ =>
      raise DTypeMismatch("get_f32_flat: expected Float32, got \{self.dtype}")
  }
}

///|
/// 将张量转换为内存连续的拷贝；若当前已处于连续状态（offset=0 且行主序步长），直接返回自身。
pub fn Tensor::contiguous(self : Tensor) -> Tensor raise TensorError {
  if self.is_contiguous() {
    return self
  }
  let n = self.numel()
  match self.data {
    F32(src) => {
      let dst = FixedArray::make(n, (0.0 : Float))
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_f32(self.shape, dst)
    }
    F64(src) => {
      let dst = FixedArray::make(n, 0.0)
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_f64(self.shape, dst)
    }
    I64(src) => {
      let dst = FixedArray::make(n, (0 : Int64))
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_i64(self.shape, dst)
    }
    I32(src) => {
      let dst = FixedArray::make(n, 0)
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_i32(self.shape, dst)
    }
    Bool(src) => {
      let dst = FixedArray::make(n, false)
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_bool(self.shape, dst)
    }
    U8(src) => {
      let dst = FixedArray::make(n, b'\x00')
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_u8(self.shape, dst)
    }
    I8(src) => {
      let dst = FixedArray::make(n, b'\x00')
      for i = 0; i < n; i = i + 1 {
        dst[i] = src[self.physical_index_from_linear(i)]
      }
      Tensor::from_i8(self.shape, dst)
    }
  }
}

///|
/// 获取底层连续的 `FixedArray[Byte]`（用于 `UInt8` 量化张量）。
pub fn Tensor::to_u8_buffer(
  self : Tensor,
) -> FixedArray[Byte] raise TensorError {
  let contig = self.contiguous()
  match contig.data {
    U8(buf) => buf
    _ => raise DTypeMismatch("to_u8_buffer: expected UInt8, got \{self.dtype}")
  }
}

///|
/// 获取底层连续的补码 `FixedArray[Byte]`（用于 `Int8` 量化张量）。
pub fn Tensor::to_i8_buffer(
  self : Tensor,
) -> FixedArray[Byte] raise TensorError {
  let contig = self.contiguous()
  match contig.data {
    I8(buf) => buf
    _ => raise DTypeMismatch("to_i8_buffer: expected Int8, got \{self.dtype}")
  }
}

///|
/// 获取底层连续的 FixedArray[Float] 缓冲区。
/// 若当前张量非连续，会先执行连续化复制以保证返回数组按行主序紧密排列。
pub fn Tensor::to_f32_buffer(
  self : Tensor,
) -> FixedArray[Float] raise TensorError {
  let contig = self.contiguous()
  match contig.data {
    F32(buf) => buf
    _ =>
      raise DTypeMismatch("to_f32_buffer: expected Float32, got \{self.dtype}")
  }
}

///|
/// 直接返回底层 `Float32` 缓冲区（不触发复制），供推理会话在生命周期结束后归还至 `BufferPool`。
/// 调用方必须确保该张量及其所有共享视图不再被后续使用。
pub fn Tensor::raw_f32_buffer(self : Tensor) -> FixedArray[Float]? {
  match self.data {
    F32(buf) => Some(buf)
    _ => None
  }
}

///|
/// 获取底层连续的 FixedArray[Int64] 缓冲区（常用于读取 ONNX Reshape 的目标 shape 张量）。
pub fn Tensor::to_i64_buffer(
  self : Tensor,
) -> FixedArray[Int64] raise TensorError {
  let contig = self.contiguous()
  match contig.data {
    I64(buf) => buf
    I32(buf) => FixedArray::makei(buf.length(), fn(i) { buf[i].to_int64() })
    _ =>
      raise DTypeMismatch(
        "to_i64_buffer: expected Int64 or Int32, got \{self.dtype}",
      )
  }
}

///|
/// 将张量转换为连续的 `FixedArray[Int32]` 缓冲（兼容 `Int32` 与 `Int64`）。
pub fn Tensor::to_i32_buffer(
  self : Tensor,
) -> FixedArray[Int32] raise TensorError {
  let contig = self.contiguous()
  match contig.data {
    I32(buf) => buf
    I64(buf) => FixedArray::makei(buf.length(), fn(i) { buf[i].to_int() })
    _ =>
      raise DTypeMismatch(
        "to_i32_buffer: expected Int32 or Int64, got \{self.dtype}",
      )
  }
}

///|
/// 按 ONNX Reshape 语义重塑张量形状：
/// - 维度值 `0`（当 allowzero == false 时）：从输入张量对应轴复制维度大小。
/// - 维度值 `-1`：最多出现一次，根据总元素数自动推导。
/// - 若当前张量已连续，返回零拷贝视图；否则先连续化后再重塑。
pub fn Tensor::reshape(
  self : Tensor,
  target : Array[Int32],
  allowzero~ : Bool = false,
) -> Tensor raise TensorError {
  let total = self.numel()
  let resolved = Array::make(target.length(), 0)
  let mut infer_axis = -1
  let mut known_product = 1

  for i = 0; i < target.length(); i = i + 1 {
    let dim = target[i]
    if dim == -1 {
      if infer_axis != -1 {
        raise InvalidShape("reshape: at most one -1 dimension is allowed")
      }
      infer_axis = i
    } else if dim == 0 && !allowzero {
      if i >= self.shape.length() {
        raise InvalidShape(
          "reshape: 0 dimension at axis \{i} exceeds input rank \{self.shape.length()}",
        )
      }
      let copied = self.shape[i]
      resolved[i] = copied
      known_product = known_product * copied
    } else if dim < 0 {
      raise InvalidShape("reshape: invalid negative dimension \{dim}")
    } else {
      resolved[i] = dim
      known_product = known_product * dim
    }
  }

  if infer_axis != -1 {
    if known_product == 0 || total % known_product != 0 {
      raise ShapeMismatch(
        "reshape: cannot infer dimension for total=\{total} with known product=\{known_product}",
      )
    }
    resolved[infer_axis] = total / known_product
  } else if known_product != total {
    raise ShapeMismatch(
      "reshape: cannot reshape \{total} elements into shape \{format_shape(resolved)}",
    )
  }

  let base = self.contiguous()
  {
    data: base.data,
    offset: base.offset,
    shape: resolved,
    strides: compute_strides(resolved),
    dtype: base.dtype,
  }
}

///|
/// 零拷贝维度重排（Transpose）。
/// perm 为长度等于张量 rank 的置换数组，例如 2D 矩阵转置为 `[1, 0]`。
pub fn Tensor::transpose(
  self : Tensor,
  perm : Array[Int32],
) -> Tensor raise TensorError {
  let rank = self.shape.length()
  if perm.length() != rank {
    raise InvalidAxis(
      "transpose: permutation length \{perm.length()} != rank \{rank}",
    )
  }
  let seen = Array::make(rank, false)
  let new_shape = Array::make(rank, 0)
  let new_strides = Array::make(rank, 0)
  for i = 0; i < rank; i = i + 1 {
    let axis = perm[i]
    if axis < 0 || axis >= rank {
      raise InvalidAxis("transpose: axis \{axis} out of range for rank \{rank}")
    }
    if seen[axis] {
      raise InvalidAxis(
        "transpose: duplicate axis \{axis} in \{format_shape(perm)}",
      )
    }
    seen[axis] = true
    new_shape[i] = self.shape[axis]
    new_strides[i] = self.strides[axis]
  }
  {
    data: self.data,
    offset: self.offset,
    shape: new_shape,
    strides: new_strides,
    dtype: self.dtype,
  }
}

///|
/// ONNX Flatten 算子语义：将张量展平为 2D 矩阵 `[d0, d1]`，
/// 其中 `d0` 是 `[0, axis)` 维度的乘积，`d1` 是 `[axis, rank)` 维度的乘积。
pub fn Tensor::flatten(
  self : Tensor,
  axis~ : Int32 = 1,
) -> Tensor raise TensorError {
  let rank = self.shape.length()
  // ONNX 允许 axis 取值范围为 [-rank, rank]
  let resolved = if axis < 0 { axis + rank } else { axis }
  if resolved < 0 || resolved > rank {
    raise InvalidAxis("flatten: axis \{axis} out of range [-\{rank}, \{rank}]")
  }
  let mut outer = 1
  for i = 0; i < resolved; i = i + 1 {
    outer = outer * self.shape[i]
  }
  let mut inner = 1
  for i = resolved; i < rank; i = i + 1 {
    inner = inner * self.shape[i]
  }
  self.reshape([outer, inner], allowzero=true)
}

///|
/// 移除大小为 1 的指定轴；若 axes 为 None，则移除所有大小为 1 的轴。
pub fn Tensor::squeeze(
  self : Tensor,
  axes : Array[Int32]?,
) -> Tensor raise TensorError {
  let rank = self.shape.length()
  let remove = Array::make(rank, false)
  match axes {
    None =>
      for i = 0; i < rank; i = i + 1 {
        if self.shape[i] == 1 {
          remove[i] = true
        }
      }
    Some(list) =>
      for i = 0; i < list.length(); i = i + 1 {
        let ax = normalize_axis(list[i], rank)
        if self.shape[ax] != 1 {
          raise InvalidAxis(
            "squeeze: cannot squeeze axis \{ax} with size \{self.shape[ax]} != 1",
          )
        }
        if remove[ax] {
          raise InvalidAxis("squeeze: duplicate axis \{ax}")
        }
        remove[ax] = true
      }
  }
  let new_shape : Array[Int32] = []
  let new_strides : Array[Int32] = []
  for i = 0; i < rank; i = i + 1 {
    if !remove[i] {
      new_shape.push(self.shape[i])
      new_strides.push(self.strides[i])
    }
  }
  {
    data: self.data,
    offset: self.offset,
    shape: new_shape,
    strides: new_strides,
    dtype: self.dtype,
  }
}

///|
/// 在指定轴位置插入大小为 1 的新维度（支持负轴，按插入后新 rank 规范化）。
pub fn Tensor::unsqueeze(
  self : Tensor,
  axes : Array[Int32],
) -> Tensor raise TensorError {
  let new_rank = self.shape.length() + axes.length()
  let is_inserted = Array::make(new_rank, false)
  for i = 0; i < axes.length(); i = i + 1 {
    let ax = normalize_axis(axes[i], new_rank)
    if is_inserted[ax] {
      raise InvalidAxis("unsqueeze: duplicate axis \{ax}")
    }
    is_inserted[ax] = true
  }
  let new_shape = Array::make(new_rank, 1)
  let new_strides = Array::make(new_rank, 0)
  let mut src_idx = 0
  for i = 0; i < new_rank; i = i + 1 {
    if !is_inserted[i] {
      new_shape[i] = self.shape[src_idx]
      new_strides[i] = self.strides[src_idx]
      src_idx = src_idx + 1
    } else {
      // 大小为 1 的维度步长对索引无影响，但设为后继步长可保持 is_contiguous 为 true
      new_strides[i] = if src_idx < self.strides.length() {
        self.strides[src_idx]
      } else {
        1
      }
    }
  }
  {
    data: self.data,
    offset: self.offset,
    shape: new_shape,
    strides: new_strides,
    dtype: self.dtype,
  }
}

///|
/// 查找 Float32 张量中最大值的逻辑下标与数值（用于分类模型输出预测类别）。
pub fn Tensor::argmax_f32(self : Tensor) -> (Int32, Float) raise TensorError {
  let n = self.numel()
  if n == 0 {
    raise InvalidShape("argmax_f32: tensor is empty")
  }
  let buf = self.to_f32_buffer()
  let mut best_idx = 0
  let mut best_val = buf[0]
  for i = 1; i < n; i = i + 1 {
    if buf[i] > best_val {
      best_val = buf[i]
      best_idx = i
    }
  }
  (best_idx, best_val)
}

///|
/// 返回简洁易读的张量摘要字符串，便于日志输出与 CLI 调试。
pub fn Tensor::summary(self : Tensor) -> String {
  "Tensor(\{self.dtype.short_name()}, shape=\{format_shape(self.shape)}, numel=\{self.numel()})"
}

///|
/// 从小端序原始字节流直接构造 Float32 张量（与 ONNX `raw_data` 及 SIMD 字节缓冲格式一致）。
pub fn Tensor::from_f32_bytes_le(
  shape : Array[Int32],
  raw : Bytes,
) -> Tensor raise TensorError {
  let expected = num_elements(shape)
  if raw.length() != expected * 4 {
    raise ShapeMismatch(
      "from_f32_bytes_le: shape \{format_shape(shape)} requires \{expected * 4} bytes, got \{raw.length()}",
    )
  }
  let buf = FixedArray::makei(expected, fn(i) {
    let off = i * 4
    let bits = raw[off].to_uint() |
      (raw[off + 1].to_uint() << 8) |
      (raw[off + 2].to_uint() << 16) |
      (raw[off + 3].to_uint() << 24)
    Float::reinterpret_from_uint(bits)
  })
  Tensor::from_f32(shape, buf)
}

///|
/// 将 Float32 张量导出为小端序 `FixedArray[Byte]`，便于直接传入 `ml_kernels/vecmath` 字节级 SIMD 内核。
pub fn Tensor::to_f32_bytes_le(
  self : Tensor,
) -> FixedArray[Byte] raise TensorError {
  let buf = self.to_f32_buffer()
  let n = buf.length()
  let out = FixedArray::make(n * 4, b'\x00')
  for i = 0; i < n; i = i + 1 {
    let bits = buf[i].reinterpret_as_uint()
    let off = i * 4
    out[off] = (bits & 0xFFU).to_byte()
    out[off + 1] = ((bits >> 8) & 0xFFU).to_byte()
    out[off + 2] = ((bits >> 16) & 0xFFU).to_byte()
    out[off + 3] = ((bits >> 24) & 0xFFU).to_byte()
  }
  out
}
