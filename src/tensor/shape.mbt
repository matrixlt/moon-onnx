///|
/// 判断张量在内存中是否为标准 C 风格行主序连续布局（且起始偏移为 0）。
/// 注意：长度为 1 的维度其步长不影响逻辑连续性，但若元素总数为 0 则直接视为连续。
pub fn Tensor::is_contiguous(self : Tensor) -> Bool {
  if self.offset != 0 {
    return false
  }
  let rank = self.shape.length()
  if rank == 0 {
    return true
  }
  let mut expected_stride = 1
  for i = rank - 1; i >= 0; i = i - 1 {
    let dim = self.shape[i]
    if dim == 0 {
      return true
    }
    // 当 dim == 1 时，该维度不会发生步进，因此任何 stride 值都等价于连续布局
    if dim > 1 && self.strides[i] != expected_stride {
      return false
    }
    expected_stride = expected_stride * dim
  }
  true
}

///|
/// 将多维坐标转换为底层缓冲区的一维物理下标。
pub fn Tensor::flat_index(
  self : Tensor,
  indices : Array[Int32],
) -> Int32 raise TensorError {
  let rank = self.shape.length()
  if indices.length() != rank {
    raise IndexOutOfBounds(
      "expected \{rank} indices for shape \{format_shape(self.shape)}, got \{indices.length()}",
    )
  }
  let mut idx = self.offset
  for i = 0; i < rank; i = i + 1 {
    let coord = indices[i]
    let dim = self.shape[i]
    if coord < 0 || coord >= dim {
      raise IndexOutOfBounds(
        "index \{coord} out of bounds for axis \{i} with size \{dim}",
      )
    }
    idx = idx + coord * self.strides[i]
  }
  idx
}

///|
/// 将行主序逻辑线性下标展开为多维坐标。
/// 例如 shape = [2, 3], linear = 4 -> [1, 1]。
pub fn unravel_index(linear : Int32, shape : Array[Int32]) -> Array[Int32] {
  let rank = shape.length()
  let coords = Array::make(rank, 0)
  let mut rem = linear
  for i = rank - 1; i >= 0; i = i - 1 {
    let dim = shape[i]
    if dim > 0 {
      coords[i] = rem % dim
      rem = rem / dim
    }
  }
  coords
}

///|
/// 根据逻辑线性下标计算物理缓冲区下标（自动兼容非连续视图与广播步长）。
pub fn Tensor::physical_index_from_linear(
  self : Tensor,
  linear : Int32,
) -> Int32 {
  let rank = self.shape.length()
  if rank == 0 {
    return self.offset
  }
  let mut rem = linear
  let mut phys = self.offset
  for i = rank - 1; i >= 0; i = i - 1 {
    let dim = self.shape[i]
    let coord = rem % dim
    rem = rem / dim
    phys = phys + coord * self.strides[i]
  }
  phys
}

///|
/// 将可能为负数的轴索引规范化到 `[0, rank)` 区间内。
/// 例如 rank = 4, axis = -1 -> 3。
pub fn normalize_axis(axis : Int32, rank : Int32) -> Int32 raise TensorError {
  let resolved = if axis < 0 { axis + rank } else { axis }
  if resolved < 0 || resolved >= rank {
    raise InvalidAxis("axis \{axis} out of range for rank \{rank}")
  }
  resolved
}

///|
/// 按 NumPy / ONNX 多向广播规则计算两个形状广播后的目标形状。
/// 从末尾维度向右对齐：两维度相等，或其中之一为 1 时可广播；否则报错。
pub fn broadcast_shape(
  a : Array[Int32],
  b : Array[Int32],
) -> Array[Int32] raise TensorError {
  let ra = a.length()
  let rb = b.length()
  let out_rank = if ra > rb { ra } else { rb }
  let out = Array::make(out_rank, 1)
  for i = 0; i < out_rank; i = i + 1 {
    let da = if i < out_rank - ra { 1 } else { a[i - (out_rank - ra)] }
    let db = if i < out_rank - rb { 1 } else { b[i - (out_rank - rb)] }
    if da == db {
      out[i] = da
    } else if da == 1 {
      out[i] = db
    } else if db == 1 {
      out[i] = da
    } else {
      raise ShapeMismatch(
        "cannot broadcast shapes \{format_shape(a)} and \{format_shape(b)} at output axis \{i} (\{da} vs \{db})",
      )
    }
  }
  out
}

///|
/// 计算将源张量广播到 target_shape 后的虚拟步长数组。
/// 发生广播（源维度为 1 或前导补齐维度）的轴步长置为 0，从而无需复制数据即可按目标坐标寻址。
pub fn broadcast_strides(
  src_shape : Array[Int32],
  src_strides : Array[Int32],
  target_shape : Array[Int32],
) -> Array[Int32] raise TensorError {
  let src_rank = src_shape.length()
  let tgt_rank = target_shape.length()
  if src_rank > tgt_rank {
    raise ShapeMismatch(
      "cannot broadcast source shape \{format_shape(src_shape)} to lower-rank target \{format_shape(target_shape)}",
    )
  }
  let out = Array::make(tgt_rank, 0)
  let pad = tgt_rank - src_rank
  for i = 0; i < src_rank; i = i + 1 {
    let sdim = src_shape[i]
    let tdim = target_shape[i + pad]
    if sdim == tdim {
      out[i + pad] = src_strides[i]
    } else if sdim == 1 {
      // 源维度为 1，重复沿该轴读取同一元素，因此步长为 0
      out[i + pad] = 0
    } else {
      raise ShapeMismatch(
        "incompatible dimension at axis \{i + pad}: source=\{sdim}, target=\{tdim}",
      )
    }
  }
  out
}
