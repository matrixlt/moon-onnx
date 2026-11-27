///|
/// Tensor 兼容模式代码生成上下文：维护静态权重表、SSA 变量映射、最小依赖标记与语句缓冲。
priv struct TensorEmitter {
  graph : @nn_ir.IrGraph
  shapes : Map[String, Array[Int32]]
  var_of : Map[String, String]
  use_simd : Bool
  simd_str : String
  weight_entries : Array[WeightEntry]
  weight_key_to_field : Map[String, String]
  raw_byte_chunks : Array[FixedArray[Byte]]
  mut cur_byte_offset : Int32
  mut needs_nn_ir : Bool
  mut needs_gemm : Bool
  body : StringBuilder
}

///|
/// 注册并去重静态 Float32 权重字段，将其小端序字节追加至紧凑 `weights.bin` 缓冲。
fn TensorEmitter::register_f32_weight(
  self : TensorEmitter,
  name : String,
  kind : WeightKind,
) -> String raise CodegenError {
  let key = match kind {
    RawF32 => "\{name}#raw"
    PackedMatB(trans_b~) => if trans_b { "\{name}#matT" } else { "\{name}#mat" }
    PackedConv(group~) => "\{name}#conv_g\{group}"
    RawBytes(..) | PackedInt8MatB => "\{name}#bytes"
  }
  match self.weight_key_to_field.get(key) {
    Some(f) => return f
    None => ()
  }
  let t = match self.graph.initializers.get(name) {
    Some(tensor) => tensor
    None =>
      raise UnsupportedGraph("expected static initializer '\{name}' not found")
  }
  if t.dtype() != @tensor.Float32 {
    raise TensorErr("weight '\{name}' expected Float32 dtype, got \{t.dtype()}")
  }
  let bytes_le = t.to_f32_bytes_le() catch { e => raise TensorErr("\{e}") }
  let field_name = "w_\{self.weight_entries.length()}"
  let byte_len = bytes_le.length()
  self.weight_entries.push({
    name,
    field_name,
    kind,
    shape: t.shape(),
    byte_offset: self.cur_byte_offset,
    byte_len,
  })
  self.weight_key_to_field[key] = field_name
  self.raw_byte_chunks.push(bytes_le)
  self.cur_byte_offset = self.cur_byte_offset + byte_len
  field_name
}

///|
/// 解析节点输入的 MoonBit 表达式（支持图输入、上游 SSA 变量、静态 Float32 权重或内联 Int64 常量）。
fn TensorEmitter::resolve_expr(
  self : TensorEmitter,
  name : String,
) -> String raise CodegenError {
  match self.var_of.get(name) {
    Some(v) => v
    None =>
      match self.graph.initializers.get(name) {
        Some(t) =>
          match t.dtype() {
            @tensor.Float32 => {
              let f = self.register_f32_weight(name, RawF32)
              "self.\{f}"
            }
            @tensor.Int64 => {
              let b = t.to_i64_buffer() catch { e => raise TensorErr("\{e}") }
              let arr = Array::makei(b.length(), fn(k) { b[k] })
              "@tensor.Tensor::from_i64(\{fmt_int_array(t.shape())}, \{fmt_i64_array(arr)})"
            }
            other =>
              raise TensorErr(
                "unsupported static initializer dtype \{other} for '\{name}'",
              )
          }
        None => raise UnsupportedGraph("unresolved tensor reference '\{name}'")
      }
  }
}

///|
/// 将已完成特化与形状推导的 `IrGraph` 按 v1 `Tensor` 回退模式生成 MoonBit 源码与紧凑权重包。
fn emit_tensor_model(
  graph : @nn_ir.IrGraph,
  shapes : Map[String, Array[Int32]],
  config : CodegenConfig,
) -> AotArtifact raise CodegenError {
  // 1. 为每个节点分配 SSA 变量名并复用 `nn_ir/analysis` 统一别名感知生命周期计划
  let var_of : Map[String, String] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    var_of[graph.inputs[i]] = "in_\{i}"
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let out_name = if node.outputs.length() > 0 { node.outputs[0] } else { "" }
    if out_name != "" {
      var_of[out_name] = "v_\{i}"
    }
  }
  let liveness = @analysis.build_release_plan(graph)
  let pinned = liveness.pinned
  let root_of = liveness.root_of
  let release_at : Array[Array[String]] = Array::makei(graph.nodes.length(), fn(
    step,
  ) {
    let vars : Array[String] = []
    let names = liveness.release_plan[step]
    for k = 0; k < names.length(); k = k + 1 {
      match var_of.get(names[k]) {
        Some(v) => vars.push(v)
        None => ()
      }
    }
    vars
  })

  // 2. 逐节点生成直线型 SSA 推理语句体
  let em : TensorEmitter = {
    graph,
    shapes,
    var_of,
    use_simd: config.use_simd,
    simd_str: if config.use_simd {
      "true"
    } else {
      "false"
    },
    weight_entries: [],
    weight_key_to_field: Map([]),
    raw_byte_chunks: [],
    cur_byte_offset: 0,
    needs_nn_ir: false,
    needs_gemm: false,
    body: StringBuilder(),
  }

  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let out_name = node.outputs[0]
    let out_var = match var_of.get(out_name) {
      Some(v) => v
      None =>
        raise UnsupportedGraph(
          "tensor_emit: missing var name for '\{out_name}'",
        )
    }
    let out_root = match root_of.get(out_name) {
      Some(r) => r
      None => out_name
    }
    let is_pinned = pinned.contains(out_name) || pinned.contains(out_root)
    let pool_expr = if is_pinned { "None" } else { "Some(self.pool)" }
    let sh_comment = match shapes.get(out_name) {
      Some(sh) => @tensor.format_shape(sh)
      None => "?"
    }

    em.body.write_string("  // [\{i}] \{node.op.name()} -> \{sh_comment}\n")
    em.emit_node(node, out_name, out_var, pool_expr)

    // 在每个节点执行完毕后立即释放死亡中间张量缓冲回 BufferPool
    let dead = release_at[i]
    for d = 0; d < dead.length(); d = d + 1 {
      em.body.write_string("  release_buf(self.pool, \{dead[d]})\n")
    }
  }

  // 3. 拼接完整 MoonBit 源码文件
  let src = StringBuilder()
  let sname = config.struct_name
  src.write_string(
    "// Auto-generated by moon-onnx AOT compiler (`onnx2mbt`).\n",
  )
  src.write_string(
    "// Graph: \{graph.name} | Nodes: \{graph.nodes.length()} | Weights: \{em.weight_entries.length()} (\{em.cur_byte_offset} bytes)\n\n",
  )

  // Helper: slice Float32 tensor from weights.bin
  if em.weight_entries.length() > 0 {
    src.write_string("///|\n")
    src.write_string(
      "/// 从紧凑权重二进制块按静态偏移解码 Float32 张量。\n",
    )
    src.write_string(
      "fn load_weight_f32(\n  weights : Bytes,\n  offset : Int32,\n  shape : Array[Int32],\n) -> @tensor.Tensor raise @tensor.TensorError {\n",
    )
    src.write_string("  let n = @tensor.num_elements(shape)\n")
    src.write_string(
      "  let raw = Bytes::makei(n * 4, i => weights[offset + i])\n",
    )
    src.write_string("  @tensor.Tensor::from_f32_bytes_le(shape, raw)\n")
    src.write_string("}\n\n")
  }

  // Helper: release buffer back to pool
  let mut has_releases = false
  for i = 0; i < release_at.length(); i = i + 1 {
    if release_at[i].length() > 0 {
      has_releases = true
      break
    }
  }
  if has_releases {
    src.write_string("///|\n")
    src.write_string(
      "/// 将死亡中间张量的底层 Float32 缓冲区归还至复用池。\n",
    )
    src.write_string(
      "fn release_buf(pool : @tensor.BufferPool, t : @tensor.Tensor) -> Unit {\n",
    )
    src.write_string("  match t.raw_f32_buffer() {\n")
    src.write_string("    Some(buf) => pool.release_f32(buf)\n")
    src.write_string("    None => ()\n")
    src.write_string("  }\n")
    src.write_string("}\n\n")
  }

  // Struct definition
  src.write_string("///|\n")
  src.write_string(
    "/// 静态编译后的推理运行时实例（零 Map 查找、零 Protobuf 解析、预打包权重常驻）。\n",
  )
  src.write_string("pub struct \{sname} {\n")
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    let w = em.weight_entries[i]
    let ty = match w.kind {
      RawF32 | RawBytes(..) | PackedInt8MatB => "@tensor.Tensor"
      PackedMatB(..) => "@gemm.PackedMatrixB"
      PackedConv(..) => "@ops.PackedConvWeight"
    }
    src.write_string("  \{w.field_name} : \{ty}\n")
  }
  src.write_string("  pool : @tensor.BufferPool\n")
  src.write_string("}\n\n")

  // load(weights : Bytes) -> StructName
  src.write_string("///|\n")
  src.write_string(
    "/// 从紧凑 `weights.bin` 字节流一次性加载并预打包全部模型权重。\n",
  )
  src.write_string(
    "pub fn \{sname}::load(weights : Bytes) -> \{sname} raise @tensor.TensorError {\n",
  )
  if em.weight_entries.length() == 0 {
    src.write_string("  let _ = weights\n")
  }
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    let w = em.weight_entries[i]
    let sh_str = fmt_int_array(w.shape)
    match w.kind {
      RawF32 | RawBytes(..) | PackedInt8MatB =>
        src.write_string(
          "  let \{w.field_name} = load_weight_f32(weights, \{w.byte_offset}, \{sh_str})\n",
        )
      PackedMatB(trans_b~) => {
        src.write_string(
          "  let \{w.field_name}_raw = load_weight_f32(weights, \{w.byte_offset}, \{sh_str}).to_f32_buffer()\n",
        )
        if trans_b {
          src.write_string(
            "  let \{w.field_name} = @gemm.pack_matrix_b_transposed_f32(\{w.field_name}_raw, 0, \{w.shape[0]}, \{w.shape[1]})\n",
          )
        } else {
          src.write_string(
            "  let \{w.field_name} = @gemm.pack_matrix_b_f32(\{w.field_name}_raw, 0, \{w.shape[0]}, \{w.shape[1]})\n",
          )
        }
      }
      PackedConv(group~) => {
        src.write_string(
          "  let \{w.field_name}_t = load_weight_f32(weights, \{w.byte_offset}, \{sh_str})\n",
        )
        src.write_string(
          "  let \{w.field_name} = @ops.pack_conv2d_weight_f32(\{w.field_name}_t, \{group})\n",
        )
      }
    }
  }
  src.write_string("  {\n")
  for i = 0; i < em.weight_entries.length(); i = i + 1 {
    src.write_string("    \{em.weight_entries[i].field_name},\n")
  }
  src.write_string("    pool: @tensor.BufferPool::new(),\n")
  src.write_string("  }\n")
  src.write_string("}\n\n")

  // forward(...)
  src.write_string("///|\n")
  src.write_string("/// 执行静态编译后的直线型前向推理。\n")
  src.write_string("pub fn \{sname}::forward(\n  self : \{sname},\n")
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    src.write_string("  in_\{i} : @tensor.Tensor,\n")
  }
  if graph.outputs.length() == 1 {
    src.write_string(") -> @tensor.Tensor raise @tensor.TensorError {\n")
  } else {
    src.write_string(") -> Array[@tensor.Tensor] raise @tensor.TensorError {\n")
  }
  src.write_string(em.body.to_string())
  if graph.outputs.length() == 1 {
    let out_v = em.resolve_expr(graph.outputs[0])
    src.write_string("  \{out_v}\n")
  } else {
    src.write_string("  [")
    for i = 0; i < graph.outputs.length(); i = i + 1 {
      if i > 0 {
        src.write_string(", ")
      }
      src.write_string(em.resolve_expr(graph.outputs[i]))
    }
    src.write_string("]\n")
  }
  src.write_string("}\n")

  // 单输入单输出时额外生成面向 Wasm/FFI 的平坦 FixedArray[Float] 快捷入口
  if graph.inputs.length() == 1 && graph.outputs.length() == 1 {
    let in_sh = require_shape_from(shapes, graph.inputs[0])
    src.write_string("\n///|\n")
    src.write_string(
      "/// 平坦 `FixedArray[Float]` 快捷前向入口（输入形状固定为 \{@tensor.format_shape(in_sh)}）。\n",
    )
    src.write_string(
      "pub fn \{sname}::forward_f32(\n  self : \{sname},\n  input : FixedArray[Float],\n) -> FixedArray[Float] raise @tensor.TensorError {\n",
    )
    src.write_string(
      "  let t_in = @tensor.Tensor::from_f32(\{fmt_int_array(in_sh)}, input)\n",
    )
    src.write_string("  self.forward(t_in).to_f32_buffer()\n")
    src.write_string("}\n")
  }

  // 4. 生成最小依赖 moon.pkg
  let pkg = StringBuilder()
  pkg.write_string("import {\n")
  pkg.write_string("  \"matrixlt/tensor\",\n")
  if em.needs_nn_ir {
    pkg.write_string("  \"matrixlt/nn_ir\",\n")
  }
  if em.needs_gemm {
    pkg.write_string("  \"matrixlt/ml_kernels/gemm\",\n")
  }
  pkg.write_string("  \"matrixlt/onnx/ops\",\n")
  pkg.write_string("}\n")

  // 5. 合并所有权重字节块为单个紧凑 Bytes
  let weights_bin = concat_byte_chunks(em.raw_byte_chunks, em.cur_byte_offset)

  if config.embed_weights {
    src.write_string("\n")
    emit_embedded_weights_section(src, sname, weights_bin, true)
  }

  let in_sh_list : Array[(String, Array[Int32])] = []
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    in_sh_list.push((name, require_shape_from(shapes, name)))
  }
  let out_sh_list : Array[(String, Array[Int32])] = []
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    let name = graph.outputs[i]
    out_sh_list.push((name, require_shape_from(shapes, name)))
  }

  {
    mbt_source: src.to_string(),
    pkg_config: pkg.to_string(),
    weights_bin,
    weights: em.weight_entries,
    input_shapes: in_sh_list,
    output_shapes: out_sh_list,
    node_count: graph.nodes.length(),
    arena_f32_elems: 0,
    arena_f32_naive_elems: 0,
    arena_byte_elems: 0,
    scratch_f32_elems: 0,
    mode_used: Tensor,
  }
}
