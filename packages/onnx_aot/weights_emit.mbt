///|
/// 将多个权重字节块线性拼接为单个紧凑 `Bytes`（供 Arena 与 Tensor 模式共享，消除重复拷贝逻辑）。
fn concat_byte_chunks(
  chunks : Array[FixedArray[Byte]],
  total_len : Int32,
) -> Bytes {
  let flat_bytes = FixedArray::make(total_len, b'\x00')
  let mut pos = 0
  for c = 0; c < chunks.length(); c = c + 1 {
    let chunk = chunks[c]
    for b = 0; b < chunk.length(); b = b + 1 {
      flat_bytes[pos + b] = chunk[b]
    }
    pos = pos + chunk.length()
  }
  Bytes::makei(total_len, fn(i) { flat_bytes[i] })
}

///|
/// 将单个字节切片 `[start, end)` 格式化为 MoonBit 原始字节串字面量 `b"\x.."`。
fn format_bytes_slice_literal(
  data : Bytes,
  start : Int32,
  end : Int32,
) -> String {
  let hex = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
  ]
  let sb = StringBuilder()
  sb.write_string("b\"")
  for i = start; i < end; i = i + 1 {
    let v = data[i].to_int()
    sb.write_string("\\x")
    sb.write_char(hex[(v >> 4) & 0xF])
    sb.write_char(hex[v & 0xF])
  }
  sb.write_char('"')
  sb.to_string()
}

///|
/// 输出内嵌模型权重常量段与零参数 `new()` 构造器：
/// 当权重超过 4096 字节时自动分块为多个 `const WEIGHTS_CHUNK_i : Bytes`（每行 <= 16 KB，彻底避免 MoonBit 超长行告警 `[0033]`）。
fn emit_embedded_weights_section(
  src : StringBuilder,
  sname : String,
  weights_bin : Bytes,
  raises_tensor_err : Bool,
) -> Unit {
  let total = weights_bin.length()
  let chunk_size = 4096
  let ret_sig = if raises_tensor_err {
    "\{sname} raise @tensor.TensorError"
  } else {
    sname
  }
  if total <= chunk_size {
    src.write_string("///|\n")
    src.write_string(
      "/// 内嵌模型权重数据段（直接编译进 Wasm Data Segment，单文件零外部请求加载）。\n",
    )
    src.write_string(
      "const WEIGHTS_DATA : Bytes = \{format_bytes_slice_literal(weights_bin, 0, total)}\n\n",
    )
    src.write_string("///|\n")
    src.write_string(
      "/// 零参数构造器：直接从内嵌 Wasm 数据段加载全部权重并初始化运行时。\n",
    )
    src.write_string(
      "pub fn \{sname}::new() -> \{ret_sig} {\n  \{sname}::load(WEIGHTS_DATA)\n}\n\n",
    )
  } else {
    let num_chunks = (total + chunk_size - 1) / chunk_size
    for c = 0; c < num_chunks; c = c + 1 {
      let s = c * chunk_size
      let e = if s + chunk_size < total { s + chunk_size } else { total }
      src.write_string("///|\n")
      src.write_string(
        "const WEIGHTS_CHUNK_\{c} : Bytes = \{format_bytes_slice_literal(weights_bin, s, e)}\n\n",
      )
    }
    src.write_string("///|\n")
    src.write_string(
      "/// 零参数构造器：拼接内嵌 Wasm 数据段分块并一次性加载全部权重。\n",
    )
    src.write_string("pub fn \{sname}::new() -> \{ret_sig} {\n")
    src.write_string("  let raw = FixedArray::make(\{total}, b'\\x00')\n")
    for c = 0; c < num_chunks; c = c + 1 {
      let s = c * chunk_size
      let e = if s + chunk_size < total { s + chunk_size } else { total }
      let len = e - s
      src.write_string(
        "  for i = 0; i < \{len}; i = i + 1 { raw[\{s} + i] = WEIGHTS_CHUNK_\{c}[i] }\n",
      )
    }
    src.write_string(
      "  \{sname}::load(Bytes::makei(\{total}, fn(i) { raw[i] }))\n}\n\n",
    )
  }
}
