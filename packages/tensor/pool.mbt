///|
/// 缓冲区复用池统计信息，用于监控高频推理场景下的内存复用命中率与 GC 压力。
#derive(Debug, Eq)
pub(all) struct PoolStats {
  hits : Int32
  misses : Int32
  pooled_f32_buffers : Int32
  pooled_byte_buffers : Int32
}

///|
/// 面向推理热路径的轻量级缓冲区复用池（参考 `rten::TensorPool` / ONNX Runtime Arena）。
/// 按精确元素/字节长度分桶缓存已归还的 `FixedArray[Float]` 与 `FixedArray[Byte]`，
/// 避免每次前向推理反复分配大尺寸中间激活与 `im2col` 工作缓冲。
pub struct BufferPool {
  f32_buckets : Map[Int32, Array[FixedArray[Float]]]
  byte_buckets : Map[Int32, Array[FixedArray[Byte]]]
  mut hits : Int32
  mut misses : Int32
  max_per_bucket : Int32
}

///|
/// 创建新的缓冲区复用池；`max_per_bucket` 限制单个尺寸桶最多缓存的缓冲数量，防止峰值内存滞留。
pub fn BufferPool::new(max_per_bucket~ : Int32 = 8) -> BufferPool {
  {
    f32_buckets: Map([]),
    byte_buckets: Map([]),
    hits: 0,
    misses: 0,
    max_per_bucket,
  }
}

///|
/// 获取长度为 `len` 的 `FixedArray[Float]`（命中缓存时零分配复用；未命中时新分配）。
/// 注意：复用返回的缓冲区可能包含旧数据，调用方负责覆盖写入或在需要时清零。
pub fn BufferPool::acquire_f32(
  self : BufferPool,
  len : Int32,
) -> FixedArray[Float] {
  match self.f32_buckets.get(len) {
    Some(stack) =>
      match stack.pop() {
        Some(buf) => {
          self.hits = self.hits + 1
          return buf
        }
        None => ()
      }
    None => ()
  }
  self.misses = self.misses + 1
  FixedArray::make(len, (0.0 : Float))
}

///|
/// 归还不再使用的 `FixedArray[Float]` 至复用池中供后续算子复用。
pub fn BufferPool::release_f32(
  self : BufferPool,
  buf : FixedArray[Float],
) -> Unit {
  let len = buf.length()
  if len == 0 {
    return
  }
  match self.f32_buckets.get(len) {
    Some(stack) => if stack.length() < self.max_per_bucket { stack.push(buf) }
    None => self.f32_buckets[len] = [buf]
  }
}

///|
/// 获取长度为 `byte_len` 的 `FixedArray[Byte]`（供 SIMD 对齐缓冲与临时序列化复用）。
pub fn BufferPool::acquire_bytes(
  self : BufferPool,
  byte_len : Int32,
) -> FixedArray[Byte] {
  match self.byte_buckets.get(byte_len) {
    Some(stack) =>
      match stack.pop() {
        Some(buf) => {
          self.hits = self.hits + 1
          return buf
        }
        None => ()
      }
    None => ()
  }
  self.misses = self.misses + 1
  FixedArray::make(byte_len, b'\x00')
}

///|
/// 归还 `FixedArray[Byte]` 至复用池。
pub fn BufferPool::release_bytes(
  self : BufferPool,
  buf : FixedArray[Byte],
) -> Unit {
  let len = buf.length()
  if len == 0 {
    return
  }
  match self.byte_buckets.get(len) {
    Some(stack) => if stack.length() < self.max_per_bucket { stack.push(buf) }
    None => self.byte_buckets[len] = [buf]
  }
}

///|
/// 清空池内全部缓存缓冲区以释放内存。
pub fn BufferPool::clear(self : BufferPool) -> Unit {
  self.f32_buckets.clear()
  self.byte_buckets.clear()
}

///|
/// 返回当前复用池的命中/未命中与已缓存缓冲区统计。
pub fn BufferPool::stats(self : BufferPool) -> PoolStats {
  let mut f32_cnt = 0
  self.f32_buckets.each(fn(_, stack) { f32_cnt = f32_cnt + stack.length() })
  let mut byte_cnt = 0
  self.byte_buckets.each(fn(_, stack) { byte_cnt = byte_cnt + stack.length() })
  {
    hits: self.hits,
    misses: self.misses,
    pooled_f32_buffers: f32_cnt,
    pooled_byte_buffers: byte_cnt,
  }
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
