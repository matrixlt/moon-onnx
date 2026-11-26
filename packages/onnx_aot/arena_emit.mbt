///|
/// Arena 代码生成上下文：统一维护静态权重打包表、双 Arena 偏移解析、按需最小依赖标记与指令流缓冲。
priv struct ArenaEmitter {
  graph : @nn_ir.IrGraph
  shapes : Map[String, Array[Int32]]
  dtypes : Map[String, @tensor.DType]
  f32_plan : ArenaPlan
  byte_plan : ArenaPlan
  input_param_of : Map[String, String]
  use_packed_f32_simd : Bool
  use_quant_simd : Bool
  mmi_to_cast : Map[String, String]
  skipped_cast : Map[String, Bool]
  weight_entries : Array[WeightEntry]
  weight_key_to_field : Map[String, String]
  raw_byte_chunks : Array[FixedArray[Byte]]
  mut cur_byte_offset : Int32
  mut need_gemm : Bool
  mut need_conv : Bool
  mut need_quant : Bool
  mut need_nn : Bool
  mut need_math : Bool
  mut need_f32_loader : Bool
  mut need_byte_loader : Bool
  mut need_trans_f32 : Bool
  body : StringBuilder
}

///|
/// 注册并去重静态权重字段，将其原始小端字节追加至紧凑 `weights.bin` 缓冲。
fn ArenaEmitter::reg_weight(
  self : ArenaEmitter,
  name : String,
  kind : WeightKind,
) -> String raise CodegenError {
  let key = match kind {
    RawF32 => "\{name}#f32"
    PackedMatB(trans_b~) =>
      if trans_b {
        "\{name}#pmatT"
      } else {
        "\{name}#pmat"
      }
    PackedConv(group~) => "\{name}#pconv_g\{group}"
    RawBytes(trans_b~) =>
      if trans_b {
        "\{name}#bytesT"
      } else {
        "\{name}#bytes"
      }
    PackedInt8MatB => "\{name}#pi8mat"
  }
  match self.weight_key_to_field.get(key) {
    Some(f) => return f
    None => ()
  }
  let t = match self.graph.initializers.get(name) {
    Some(tensor) => tensor
    None => raise UnsupportedGraph("missing initializer '\{name}'")
  }
  let bytes_chunk : FixedArray[Byte] = match kind {
    RawBytes(..) | PackedInt8MatB =>
      match t.dtype() {
        @tensor.Int8 => t.to_i8_buffer() catch { e => raise TensorErr("\{e}") }
        @tensor.UInt8 => t.to_u8_buffer() catch { e => raise TensorErr("\{e}") }
        other =>
          raise TensorErr(
            "expected Int8/UInt8 weight for '\{name}', got \{other}",
          )
      }
    _ => t.to_f32_bytes_le() catch { e => raise TensorErr("\{e}") }
  }
  let field_name = "w_\{self.weight_entries.length()}"
  let byte_len = bytes_chunk.length()
  self.weight_entries.push({
    name,
    field_name,
    kind,
    shape: t.shape(),
    byte_offset: self.cur_byte_offset,
    byte_len,
  })
  self.weight_key_to_field[key] = field_name
  self.raw_byte_chunks.push(bytes_chunk)
  self.cur_byte_offset = self.cur_byte_offset + byte_len
  field_name
}

///|
/// 解析 Float32 张量在静态 Arena 模式下的 `(缓冲区表达式, 元素起始偏移)`。
fn ArenaEmitter::resolve_f32_loc(
  self : ArenaEmitter,
  name : String,
) -> (String, Int32) raise CodegenError {
  match self.input_param_of.get(name) {
    Some(pname) => return (pname, 0)
    None => ()
  }
  match self.f32_plan.allocs.get(name) {
    Some(a) => return ("self.arena_f32", a.offset)
    None => ()
  }
  if self.graph.initializers.contains(name) {
    let f = self.reg_weight(name, RawF32)
    return ("self.\{f}", 0)
  }
  raise UnsupportedGraph("arena: unresolved f32 tensor '\{name}'")
}

///|
/// 解析 Int8 / UInt8 张量在静态字节 Arena 模式下的 `(缓冲区表达式, 字节起始偏移)`。
fn ArenaEmitter::resolve_byte_loc(
  self : ArenaEmitter,
  name : String,
) -> (String, Int32) raise CodegenError {
  match self.byte_plan.allocs.get(name) {
    Some(a) => return ("self.arena_u8", a.offset)
    None => ()
  }
  if self.graph.initializers.contains(name) {
    let f = self.reg_weight(name, RawBytes(trans_b=false))
    return ("self.\{f}", 0)
  }
  raise UnsupportedGraph("arena: unresolved byte tensor '\{name}'")
}

///|
/// Arena 模式下生成零堆分配、零 `Tensor`、零 `BufferPool`、零 `onnx/ops` 依赖的极简高速 MoonBit 推理代码。
/// 所有常见 NN 辅助算子（MaxPool/AvgPool/GAP/Erf/GELU/Softmax/LayerNorm/Reflect/ScalarConv）均直接调用
/// 经过全四端类型检查与单测验证的 `matrixlt/ml_kernels/nn`，彻底消除字符串内联样板代码。
fn emit_arena_model(
  graph : @nn_ir.IrGraph,
  shapes : Map[String, Array[Int32]],
  dtypes : Map[String, @tensor.DType],
  f32_plan : ArenaPlan,
  byte_plan : ArenaPlan,
  config : CodegenConfig,
) -> AotArtifact raise CodegenError {
  let single_io = graph.inputs.length() == 1 && graph.outputs.length() == 1
  let in_names = graph.inputs
  let out_names = graph.outputs
  let in_shapes_arr : Array[(String, Array[Int32])] = []
  let in_lens : Array[Int32] = []
  for idx = 0; idx < in_names.length(); idx = idx + 1 {
    let sh = shapes.get(in_names[idx]).unwrap()
    in_shapes_arr.push((in_names[idx], sh))
    in_lens.push(
      @tensor.num_elements(sh) catch {
        e => raise ShapeError("\{e}")
      },
    )
  }
  let out_shapes_arr : Array[(String, Array[Int32])] = []
  let out_lens : Array[Int32] = []
  for idx = 0; idx < out_names.length(); idx = idx + 1 {
    let sh = shapes.get(out_names[idx]).unwrap()
    out_shapes_arr.push((out_names[idx], sh))
    out_lens.push(
      @tensor.num_elements(sh) catch {
        e => raise ShapeError("\{e}")
      },
    )
  }

  // 建立图输入及其零拷贝别名视图到函数参数名的映射（单输入用 `input`，多输入用 `in_0`, `in_1`, ...）
  let input_param_of : Map[String, String] = Map([])
  for idx = 0; idx < in_names.length(); idx = idx + 1 {
    let pname = if single_io { "input" } else { "in_\{idx}" }
    input_param_of[in_names[idx]] = pname
  }
  for idx = 0; idx < graph.nodes.length(); idx = idx + 1 {
    let nd = graph.nodes[idx]
    if is_arena_alias_op(nd.op) &&
      nd.inputs.length() > 0 &&
      nd.outputs.length() > 0 &&
      nd.outputs[0] != "" {
      match input_param_of.get(nd.inputs[0]) {
        Some(p) => input_param_of[nd.outputs[0]] = p
        None => ()
      }
    }
  }

  // 根据目标平台自适应选择 SIMD 策略：
  // - TargetWasm / TargetNative：使用预打包字节 PackedMatrixB（V128 提速 19x）
  // - TargetWasmGc：避免 GC 字节数组 v128_load 退化，改用原生 FixedArray[Float] 路径
  let use_packed_f32_simd = config.use_simd && config.target != TargetWasmGc
  let use_quant_simd = config.use_simd && config.target != TargetWasmGc
  let (mmi_to_cast, skipped_cast) = find_fused_mmi_casts(graph)

  let em : ArenaEmitter = {
    graph,
    shapes,
    dtypes,
    f32_plan,
    byte_plan,
    input_param_of,
    use_packed_f32_simd,
    use_quant_simd,
    mmi_to_cast,
    skipped_cast,
    weight_entries: [],
    weight_key_to_field: Map([]),
    raw_byte_chunks: [],
    cur_byte_offset: 0,
    need_gemm: false,
    need_conv: false,
    need_quant: false,
    need_nn: false,
    need_math: false,
    need_f32_loader: false,
    need_byte_loader: false,
    need_trans_f32: false,
    body: StringBuilder(),
  }

  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let out_n = node.outputs[0]
    let out_sh = shapes.get(out_n).unwrap()
    let out_numel = @tensor.num_elements(out_sh) catch {
      e => raise ShapeError("\{e}")
    }

    // 零拷贝逻辑视图算子：编译期偏移直接复用，运行期 0 指令！
    if is_arena_alias_op(node.op) {
      em.body.write_string(
        "  // [\{i}] \{node.op.name()} -> \{@tensor.format_shape(out_sh)} (zero-cost compile-time alias)\n",
      )
      continue
    }

    em.body.write_string(
      "  // [\{i}] \{node.op.name()} -> \{@tensor.format_shape(out_sh)}\n",
    )

    if !em.emit_nn_op(i, node, out_n, out_sh, out_numel) {
      em.emit_tensor_op(node, out_n, out_sh, out_numel)
    }
  }

  // 检查是否需要读取 f32 / byte 权重辅助函数
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    match em.weight_entries[i].kind {
      RawBytes(..) | PackedInt8MatB => em.need_byte_loader = true
      _ => em.need_f32_loader = true
    }
  }

  // 合并权重二进制块
  let flat_bytes = FixedArray::make(em.cur_byte_offset, b'\x00')
  let mut pos = 0
  for c = 0; c < em.raw_byte_chunks.length(); c = c + 1 {
    let chunk = em.raw_byte_chunks[c]
    for b = 0; b < chunk.length(); b = b + 1 {
      flat_bytes[pos + b] = chunk[b]
    }
    pos = pos + chunk.length()
  }
  let weights_bin = Bytes::makei(em.cur_byte_offset, fn(i) { flat_bytes[i] })

  // 组装完整零分配 Arena MoonBit 源码
  let src = StringBuilder()
  let sname = config.struct_name
  let arena_f32_bytes = f32_plan.peak_elems * 4
  let naive_f32_bytes = f32_plan.naive_elems * 4
  src.write_string(
    "// Auto-generated by moon-onnx AOT compiler v2 (`onnx2mbt` Arena Mode).\n",
  )
  src.write_string(
    "// Graph: \{graph.name} | Nodes: \{graph.nodes.length()} | Weights: \{em.weight_entries.length()} (\{em.cur_byte_offset} bytes)\n",
  )
  src.write_string(
    "// Static F32 Arena: \{f32_plan.peak_elems} elems (\{arena_f32_bytes} B) vs Naive \{f32_plan.naive_elems} elems (\{naive_f32_bytes} B) | Shared Scratch: \{f32_plan.scratch_elems} elems\n",
  )
  src.write_string(
    "// Runtime heap allocations per `forward_into`: 0 (Zero-Allocation Inference)\n\n",
  )

  if em.need_f32_loader {
    src.write_string("///|\n")
    src.write_string(
      "/// 从小端序权重字节块解码 Float32 数组。\n",
    )
    src.write_string(
      "fn load_f32_slice(weights : Bytes, byte_off : Int32, count : Int32) -> FixedArray[Float] {\n",
    )
    src.write_string("  FixedArray::makei(count, fn(i) {\n")
    src.write_string("    let o = byte_off + i * 4\n")
    src.write_string(
      "    let bits = weights[o].to_uint() | (weights[o + 1].to_uint() << 8) | (weights[o + 2].to_uint() << 16) | (weights[o + 3].to_uint() << 24)\n",
    )
    src.write_string("    Float::reinterpret_from_uint(bits)\n")
    src.write_string("  })\n}\n\n")
  }

  if em.need_byte_loader {
    src.write_string("///|\n")
    src.write_string(
      "/// 从紧凑权重字节块切出 1 字节量化权重数组。\n",
    )
    src.write_string(
      "fn load_byte_slice(weights : Bytes, byte_off : Int32, count : Int32) -> FixedArray[Byte] {\n",
    )
    src.write_string(
      "  FixedArray::makei(count, fn(i) { weights[byte_off + i] })\n}\n\n",
    )
  }

  if em.need_trans_f32 {
    src.write_string("///|\n")
    src.write_string("/// 离线转置 2D Float32 权重 [N, K] -> [K, N]。\n")
    src.write_string(
      "fn transpose_f32_nk_to_kn(src : FixedArray[Float], n : Int32, k : Int32) -> FixedArray[Float] {\n",
    )
    src.write_string("  let dst = FixedArray::make(k * n, (0.0 : Float))\n")
    src.write_string("  for r = 0; r < n; r = r + 1 {\n")
    src.write_string("    for c = 0; c < k; c = c + 1 {\n")
    src.write_string("      dst[c * n + r] = src[r * k + c]\n")
    src.write_string("    }\n  }\n  dst\n}\n\n")
  }

  // 结构体定义
  src.write_string("///|\n")
  src.write_string(
    "/// 零堆分配静态编译模型运行时：预打包权重常驻，所有中间激活静态复用单一 Arena 缓冲区。\n",
  )
  src.write_string("pub struct \{sname} {\n")
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    let w = em.weight_entries[i]
    let ty = match w.kind {
      RawF32 => "FixedArray[Float]"
      PackedMatB(..) =>
        if use_packed_f32_simd {
          "@gemm.PackedMatrixB"
        } else {
          "FixedArray[Float]"
        }
      PackedConv(..) => "@gemm.PackedMatrixB"
      RawBytes(..) => "FixedArray[Byte]"
      PackedInt8MatB => "@quant.PackedInt8MatrixB"
    }
    src.write_string("  \{w.field_name} : \{ty}\n")
  }
  if f32_plan.peak_elems > 0 {
    src.write_string("  arena_f32 : FixedArray[Float]\n")
  }
  if byte_plan.peak_elems > 0 {
    src.write_string("  arena_u8 : FixedArray[Byte]\n")
  }
  if f32_plan.scratch_elems > 0 {
    src.write_string("  scratch : FixedArray[Float]\n")
  }
  src.write_string("}\n\n")

  // 元信息查询方法
  if single_io {
    src.write_string("///|\n")
    src.write_string("/// 返回模型静态输入 Float32 元素总数。\n")
    src.write_string(
      "pub fn \{sname}::input_len() -> Int32 {\n  \{in_lens[0]}\n}\n\n",
    )
    src.write_string("///|\n")
    src.write_string("/// 返回模型静态输出 Float32 元素总数。\n")
    src.write_string(
      "pub fn \{sname}::output_len() -> Int32 {\n  \{out_lens[0]}\n}\n\n",
    )
  } else {
    src.write_string("///|\n")
    src.write_string(
      "/// 返回模型各静态输入的 Float32 元素数列表。\n",
    )
    src.write_string(
      "pub fn \{sname}::input_lens() -> Array[Int32] {\n  \{fmt_int_array(in_lens)}\n}\n\n",
    )
    src.write_string("///|\n")
    src.write_string(
      "/// 返回模型各静态输出的 Float32 元素数列表。\n",
    )
    src.write_string(
      "pub fn \{sname}::output_lens() -> Array[Int32] {\n  \{fmt_int_array(out_lens)}\n}\n\n",
    )
  }
  src.write_string("///|\n")
  src.write_string(
    "/// 返回静态激活 Arena 与卷积共享工作区总内存占用（字节）。\n",
  )
  src.write_string(
    "pub fn \{sname}::arena_bytes() -> Int32 {\n  \{(f32_plan.peak_elems + f32_plan.scratch_elems) * 4 + byte_plan.peak_elems}\n}\n\n",
  )

  // load(weights : Bytes)
  src.write_string("///|\n")
  src.write_string(
    "/// 从紧凑 `weights.bin` 一次性加载、预打包权重并分配静态 Arena（后续推理 0 堆分配）。\n",
  )
  src.write_string("pub fn \{sname}::load(weights : Bytes) -> \{sname} {\n")
  if em.weight_entries.length() == 0 {
    src.write_string("  let _ = weights\n")
  }
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    let w = em.weight_entries[i]
    let numel = @tensor.num_elements(w.shape) catch {
      e => raise ShapeError("\{e}")
    }
    match w.kind {
      RawF32 =>
        src.write_string(
          "  let \{w.field_name} = load_f32_slice(weights, \{w.byte_offset}, \{numel})\n",
        )
      PackedMatB(trans_b~) => {
        src.write_string(
          "  let \{w.field_name}_raw = load_f32_slice(weights, \{w.byte_offset}, \{numel})\n",
        )
        if use_packed_f32_simd {
          if trans_b {
            src.write_string(
              "  let \{w.field_name} = @gemm.pack_matrix_b_transposed_f32(\{w.field_name}_raw, 0, \{w.shape[0]}, \{w.shape[1]})\n",
            )
          } else {
            src.write_string(
              "  let \{w.field_name} = @gemm.pack_matrix_b_f32(\{w.field_name}_raw, 0, \{w.shape[0]}, \{w.shape[1]})\n",
            )
          }
        } else if trans_b {
          src.write_string(
            "  let \{w.field_name} = transpose_f32_nk_to_kn(\{w.field_name}_raw, \{w.shape[0]}, \{w.shape[1]})\n",
          )
        } else {
          src.write_string("  let \{w.field_name} = \{w.field_name}_raw\n")
        }
      }
      PackedConv(..) => {
        let c_out = w.shape[0]
        let k_dim = if w.shape.length() == 4 {
          w.shape[1] * w.shape[2] * w.shape[3]
        } else {
          w.shape[1] * w.shape[2]
        }
        src.write_string(
          "  let \{w.field_name}_raw = load_f32_slice(weights, \{w.byte_offset}, \{numel})\n",
        )
        src.write_string(
          "  let \{w.field_name} = @gemm.pack_matrix_b_transposed_f32(\{w.field_name}_raw, 0, \{c_out}, \{k_dim})\n",
        )
      }
      RawBytes(trans_b~) => {
        src.write_string(
          "  let \{w.field_name}_raw = load_byte_slice(weights, \{w.byte_offset}, \{numel})\n",
        )
        if trans_b {
          src.write_string(
            "  let \{w.field_name} = @quant.transpose_bytes_kn_to_nk(\{w.field_name}_raw, \{w.shape[0]}, \{w.shape[1]})\n",
          )
        } else {
          src.write_string("  let \{w.field_name} = \{w.field_name}_raw\n")
        }
      }
      PackedInt8MatB => {
        src.write_string(
          "  let \{w.field_name}_raw = load_byte_slice(weights, \{w.byte_offset}, \{numel})\n",
        )
        src.write_string(
          "  let \{w.field_name} = @quant.pack_int8_matrix_b(\{w.field_name}_raw, \{w.shape[0]}, \{w.shape[1]})\n",
        )
      }
    }
  }
  src.write_string("  {\n")
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    src.write_string("    \{em.weight_entries[i].field_name},\n")
  }
  if f32_plan.peak_elems > 0 {
    src.write_string(
      "    arena_f32: FixedArray::make(\{f32_plan.peak_elems}, (0.0 : Float)),\n",
    )
  }
  if byte_plan.peak_elems > 0 {
    src.write_string(
      "    arena_u8: FixedArray::make(\{byte_plan.peak_elems}, b'\\x00'),\n",
    )
  }
  if f32_plan.scratch_elems > 0 {
    src.write_string(
      "    scratch: FixedArray::make(\{f32_plan.scratch_elems}, (0.0 : Float)),\n",
    )
  }
  src.write_string("  }\n}\n\n")

  if config.embed_weights {
    emit_embedded_weights_section(src, sname, weights_bin, false)
  }

  // forward_into（真正 0 堆分配热路径）
  fn input_arr_type(name : String) -> String {
    match dtypes.get(name) {
      Some(@tensor.Int64) => "FixedArray[Int64]"
      _ => "FixedArray[Float]"
    }
  }

  if single_io {
    let (final_buf, final_off) = em.resolve_f32_loc(out_names[0])
    let in_ty = input_arr_type(in_names[0])
    src.write_string("///|\n")
    src.write_string(
      "/// 零堆分配前向推理入口：将输入 `input[\{in_lens[0]}]` 推理结果直接写入调用方缓冲 `output[\{out_lens[0]}]`。\n",
    )
    src.write_string(
      "pub fn \{sname}::forward_into(\n  self : \{sname},\n  input : \{in_ty},\n  output : FixedArray[Float],\n) -> Unit {\n",
    )
    src.write_string(em.body.to_string())
    src.write_string("  for k_i = 0; k_i < \{out_lens[0]}; k_i = k_i + 1 {\n")
    src.write_string("    output[k_i] = \{final_buf}[\{final_off} + k_i]\n")
    src.write_string("  }\n}\n\n")

    src.write_string("///|\n")
    src.write_string(
      "/// 便捷前向推理入口：返回新分配的 `FixedArray[Float]` 输出结果。\n",
    )
    src.write_string(
      "pub fn \{sname}::forward_f32(\n  self : \{sname},\n  input : \{in_ty},\n) -> FixedArray[Float] {\n",
    )
    src.write_string(
      "  let out = FixedArray::make(\{out_lens[0]}, (0.0 : Float))\n",
    )
    src.write_string("  self.forward_into(input, out)\n")
    src.write_string("  out\n}\n")
  } else {
    src.write_string("///|\n")
    src.write_string(
      "/// 零堆分配多输入多输出前向推理入口：将结果直接写入调用方预分配的输出缓冲。\n",
    )
    src.write_string("pub fn \{sname}::forward_into(\n  self : \{sname},\n")
    for k = 0; k < in_names.length(); k = k + 1 {
      let in_ty = input_arr_type(in_names[k])
      src.write_string("  in_\{k} : \{in_ty},\n")
    }
    for k = 0; k < out_names.length(); k = k + 1 {
      src.write_string("  out_\{k} : FixedArray[Float],\n")
    }
    src.write_string(") -> Unit {\n")
    src.write_string(em.body.to_string())
    for k = 0; k < out_names.length(); k = k + 1 {
      let (fb, fo) = em.resolve_f32_loc(out_names[k])
      src.write_string(
        "  for k_i = 0; k_i < \{out_lens[k]}; k_i = k_i + 1 { out_\{k}[k_i] = \{fb}[\{fo} + k_i] }\n",
      )
    }
    src.write_string("}\n\n")

    src.write_string("///|\n")
    src.write_string(
      "/// 便捷多输入多输出前向推理入口：返回各输出的 `FixedArray[Float]` 数组。\n",
    )
    src.write_string("pub fn \{sname}::forward_f32(\n  self : \{sname},\n")
    for k = 0; k < in_names.length(); k = k + 1 {
      let in_ty = input_arr_type(in_names[k])
      src.write_string("  in_\{k} : \{in_ty},\n")
    }
    src.write_string(") -> Array[FixedArray[Float]] {\n")
    for k = 0; k < out_names.length(); k = k + 1 {
      src.write_string(
        "  let out_\{k} = FixedArray::make(\{out_lens[k]}, (0.0 : Float))\n",
      )
    }
    src.write_string("  self.forward_into(")
    for k = 0; k < in_names.length(); k = k + 1 {
      if k > 0 {
        src.write_string(", ")
      }
      src.write_string("in_\{k}")
    }
    for k = 0; k < out_names.length(); k = k + 1 {
      src.write_string(", out_\{k}")
    }
    src.write_string(")\n  [")
    for k = 0; k < out_names.length(); k = k + 1 {
      if k > 0 {
        src.write_string(", ")
      }
      src.write_string("out_\{k}")
    }
    src.write_string("]\n}\n")
  }

  // 生成极简 moon.pkg（零 tensor / 零 nn_ir / 零 onnx/ops / 零 onnx_proto！）
  let pkg = StringBuilder()
  let has_any_import = em.need_math ||
    em.need_gemm ||
    em.need_conv ||
    em.need_quant ||
    em.need_nn
  if has_any_import {
    pkg.write_string("import {\n")
    if em.need_math {
      pkg.write_string("  \"moonbitlang/core/math\",\n")
    }
    if em.need_gemm {
      pkg.write_string("  \"matrixlt/ml_kernels/gemm\",\n")
    }
    if em.need_conv {
      pkg.write_string("  \"matrixlt/ml_kernels/conv\",\n")
    }
    if em.need_quant {
      pkg.write_string("  \"matrixlt/ml_kernels/quant\",\n")
    }
    if em.need_nn {
      pkg.write_string("  \"matrixlt/ml_kernels/nn\",\n")
    }
    pkg.write_string("}\n")
  }

  {
    mbt_source: src.to_string(),
    pkg_config: pkg.to_string(),
    weights_bin,
    weights: em.weight_entries,
    input_shapes: in_shapes_arr,
    output_shapes: out_shapes_arr,
    node_count: graph.nodes.length(),
    arena_f32_elems: f32_plan.peak_elems,
    arena_f32_naive_elems: f32_plan.naive_elems,
    arena_byte_elems: byte_plan.peak_elems,
    scratch_f32_elems: f32_plan.scratch_elems,
    mode_used: Arena,
  }
}
