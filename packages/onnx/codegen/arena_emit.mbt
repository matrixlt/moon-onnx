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

///|
/// 编译期解析 2D 卷积/池化最终四向填充 `(pad_top, pad_left, pad_bottom, pad_right)`，
/// 彻底消除运行期的 `AutoPad` 分支与数组分配。
fn resolve_2d_pads(
  ih : Int32,
  iw : Int32,
  kh : Int32,
  kw : Int32,
  sh : Int32,
  sw : Int32,
  dh : Int32,
  dw : Int32,
  pads : Array[Int32],
  auto_pad : @nn_ir.AutoPad,
  oh : Int32,
  ow : Int32,
) -> (Int32, Int32, Int32, Int32) {
  let eff_kh = (kh - 1) * dh + 1
  let eff_kw = (kw - 1) * dw + 1
  match auto_pad {
    @nn_ir.NotSet => (pads[0], pads[1], pads[2], pads[3])
    @nn_ir.Valid => (0, 0, 0, 0)
    @nn_ir.SameUpper | @nn_ir.SameLower => {
      let th = (oh - 1) * sh + eff_kh - ih
      let tw = (ow - 1) * sw + eff_kw - iw
      let pad_h = if th > 0 { th } else { 0 }
      let pad_w = if tw > 0 { tw } else { 0 }
      match auto_pad {
        @nn_ir.SameUpper =>
          (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
        _ => (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2)
      }
    }
  }
}

///|
/// 将 `@nn_ir.FusedActivation` 格式化为 `@gemm.Epilogue` 表达式（Arena 模式不再依赖 `matrixlt/nn_ir`）。
fn fmt_gemm_epilogue(act : @nn_ir.FusedActivation) -> String {
  match act {
    @nn_ir.NoAct => "@gemm.NoEpilogue"
    @nn_ir.ReluAct => "@gemm.Relu"
    @nn_ir.ClipAct(min~, max~) =>
      "@gemm.Clip(min=\{fmt_float_literal(min)}, max=\{fmt_float_literal(max)})"
  }
}

///|
/// 识别图中可零拷贝融合的 `MatMulInteger -> Cast(to=Float32)` 节点对：
/// 当 `MatMulInteger` 的 `Int32` 输出仅被单个 `Cast(to=Float32)` 消费（且不是图输出）时，
/// 返回 `(mmi_out -> cast_out, cast_out -> true)`，避免分配任何中间 `Int32` Arena 缓冲。
fn find_fused_mmi_casts(
  graph : @nn_ir.IrGraph,
) -> (Map[String, String], Map[String, Bool]) {
  let use_cnt : Map[String, Int32] = Map([])
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let ins = graph.nodes[i].inputs
    for k = 0; k < ins.length(); k = k + 1 {
      if ins[k] != "" {
        let prev = match use_cnt.get(ins[k]) {
          Some(c) => c
          None => 0
        }
        use_cnt[ins[k]] = prev + 1
      }
    }
  }
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    let n = graph.outputs[i]
    let prev = match use_cnt.get(n) {
      Some(c) => c
      None => 0
    }
    use_cnt[n] = prev + 1
  }
  let mmi_outs : Map[String, Bool] = Map([])
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let nd = graph.nodes[i]
    match nd.op {
      @nn_ir.MatMulInteger if nd.outputs.length() == 1 && nd.outputs[0] != "" =>
        mmi_outs[nd.outputs[0]] = true
      _ => ()
    }
  }
  let mmi_to_cast : Map[String, String] = Map([])
  let skipped_cast : Map[String, Bool] = Map([])
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let nd = graph.nodes[i]
    match nd.op {
      @nn_ir.Cast(to=@tensor.Float32) if nd.inputs.length() >= 1 &&
        nd.outputs.length() == 1 &&
        nd.outputs[0] != "" => {
        let src = nd.inputs[0]
        if mmi_outs.contains(src) && use_cnt.get(src) == Some(1) {
          mmi_to_cast[src] = nd.outputs[0]
          skipped_cast[nd.outputs[0]] = true
        }
      }
      _ => ()
    }
  }
  (mmi_to_cast, skipped_cast)
}

///|
/// 检查当前图是否满足 Zero-Allocation Arena 模式编译条件（支持 Float32/Int64 输入、Float32 输出与 Transformer/INT8 量化全算子）。
fn can_emit_arena(
  graph : @nn_ir.IrGraph,
  shapes : Map[String, Array[Int32]],
  dtypes : Map[String, @tensor.DType],
) -> Bool {
  if graph.inputs.length() == 0 || graph.outputs.length() == 0 {
    return false
  }
  let (mmi_to_cast, skipped_cast) = find_fused_mmi_casts(graph)
  let input_or_alias_i64 : Map[String, Bool] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    match dtypes.get(graph.inputs[i]) {
      Some(@tensor.Float32) | None => ()
      Some(@tensor.Int64) => input_or_alias_i64[graph.inputs[i]] = true
      _ => return false
    }
  }
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    match dtypes.get(graph.outputs[i]) {
      Some(@tensor.Float32) | None => ()
      _ => return false
    }
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let ins = node.inputs
    if node.outputs.length() == 0 {
      return false
    }
    let is_multi_out = match node.op {
      @nn_ir.Split(..) | @nn_ir.DynamicQuantizeLinear => true
      _ => false
    }
    if !is_multi_out && (node.outputs.length() != 1 || node.outputs[0] == "") {
      return false
    }
    if is_arena_alias_op(node.op) &&
      ins.length() > 0 &&
      input_or_alias_i64.contains(ins[0]) {
      input_or_alias_i64[node.outputs[0]] = true
    }
    let ok = match node.op {
      @nn_ir.Identity
      | @nn_ir.Flatten(..)
      | @nn_ir.Reshape(..)
      | @nn_ir.Squeeze(..)
      | @nn_ir.Unsqueeze(..)
      | @nn_ir.Relu
      | @nn_ir.Sigmoid
      | @nn_ir.Tanh
      | @nn_ir.Gelu
      | @nn_ir.Erf
      | @nn_ir.HardSwish
      | @nn_ir.LeakyRelu(..)
      | @nn_ir.Sqrt
      | @nn_ir.Exp
      | @nn_ir.Neg
      | @nn_ir.GlobalAveragePool
      | @nn_ir.Softmax(..)
      | @nn_ir.LayerNormalization(..) => true
      @nn_ir.DynamicQuantizeLinear =>
        ins.length() >= 1 &&
        node.outputs.length() >= 3 &&
        (
          dtypes.get(ins[0]) == Some(@tensor.Float32) ||
          dtypes.get(ins[0]) is None
        )
      @nn_ir.MatMulInteger =>
        if ins.length() >= 2 && mmi_to_cast.contains(node.outputs[0]) {
          match
            (
              shapes.get(ins[0]),
              shapes.get(ins[1]),
              dtypes.get(ins[0]),
              dtypes.get(ins[1]),
            ) {
            (Some(sa), Some(sb), Some(@tensor.UInt8), Some(@tensor.Int8)) => {
              let b_zp_ok = ins.length() < 4 ||
                ins[3] == "" ||
                graph.initializers.contains(ins[3])
              sa.length() >= 2 &&
              sb.length() == 2 &&
              graph.initializers.contains(ins[1]) &&
              b_zp_ok
            }
            _ => false
          }
        } else {
          false
        }
      @nn_ir.Transpose(..) =>
        match (shapes.get(ins[0]), dtypes.get(ins[0])) {
          (Some(sx), Some(@tensor.Float32) | None) =>
            sx.length() >= 1 && sx.length() <= 4
          _ => false
        }
      @nn_ir.Cast(to~) =>
        to == @tensor.Float32 &&
        ins.length() >= 1 &&
        (
          input_or_alias_i64.contains(ins[0]) ||
          skipped_cast.contains(node.outputs[0])
        )
      @nn_ir.Clip(..) => {
        let lo_ok = ins.length() < 2 ||
          ins[1] == "" ||
          graph.initializers.contains(ins[1])
        let hi_ok = ins.length() < 3 ||
          ins[2] == "" ||
          graph.initializers.contains(ins[2])
        lo_ok && hi_ok
      }
      @nn_ir.Add | @nn_ir.Sub | @nn_ir.Mul | @nn_ir.Div =>
        match shapes.get(node.outputs[0]) {
          Some(so) => so.length() <= 4
          None => false
        }
      @nn_ir.Pow =>
        match (shapes.get(ins[0]), shapes.get(ins[1])) {
          (Some(sa), Some(sb)) => {
            let same = sa == sb
            let scalar_b = sb.length() == 0 || (sb.length() == 1 && sb[0] == 1)
            same || scalar_b
          }
          _ => false
        }
      @nn_ir.Conv(group~, ..) =>
        match (shapes.get(ins[0]), shapes.get(ins[1])) {
          (Some(sx), Some(sw)) =>
            (
              (sx.length() == 4 && sw.length() == 4) ||
              (sx.length() == 3 && sw.length() == 3)
            ) &&
            graph.initializers.contains(ins[1]) &&
            (group == 1 || (group > 1 && group == sx[1] && group == sw[0]))
          _ => false
        }
      @nn_ir.MaxPool(..) | @nn_ir.AveragePool(..) =>
        match shapes.get(ins[0]) {
          Some(sx) => sx.length() == 4
          None => false
        }
      @nn_ir.MatMul =>
        match (shapes.get(ins[0]), shapes.get(ins[1])) {
          (Some(sa), Some(sb)) =>
            (
              sa.length() >= 2 &&
              sb.length() == 2 &&
              graph.initializers.contains(ins[1])
            ) ||
            (sa.length() >= 2 && sa.length() == sb.length())
          _ => false
        }
      @nn_ir.Gemm(..) =>
        match (shapes.get(ins[0]), shapes.get(ins[1])) {
          (Some(sa), Some(sb)) =>
            sa.length() == 2 &&
            sb.length() == 2 &&
            graph.initializers.contains(ins[1])
          _ => false
        }
      @nn_ir.Pad(pads~, ..) =>
        match shapes.get(ins[0]) {
          Some(sx) if sx.length() >= 1 && sx.length() <= 4 => {
            let pads_ok = (
                ins.length() >= 2 &&
                ins[1] != "" &&
                graph.initializers.contains(ins[1])
              ) ||
              pads is Some(_)
            let cval_ok = ins.length() < 3 ||
              ins[2] == "" ||
              graph.initializers.contains(ins[2])
            let axes_ok = ins.length() < 4 ||
              ins[3] == "" ||
              graph.initializers.contains(ins[3])
            pads_ok && cval_ok && axes_ok
          }
          _ => false
        }
      @nn_ir.Slice(starts~, ends~, ..) =>
        match shapes.get(ins[0]) {
          Some(sx) if sx.length() >= 1 && sx.length() <= 4 => {
            let s_ok = (
                ins.length() >= 2 &&
                ins[1] != "" &&
                graph.initializers.contains(ins[1])
              ) ||
              starts is Some(_)
            let e_ok = (
                ins.length() >= 3 &&
                ins[2] != "" &&
                graph.initializers.contains(ins[2])
              ) ||
              ends is Some(_)
            let a_ok = ins.length() < 4 ||
              ins[3] == "" ||
              graph.initializers.contains(ins[3])
            let st_ok = ins.length() < 5 ||
              ins[4] == "" ||
              graph.initializers.contains(ins[4])
            s_ok && e_ok && a_ok && st_ok
          }
          _ => false
        }
      @nn_ir.Gather(..) => {
        let data_ok = match dtypes.get(ins[0]) {
          Some(@tensor.Float32)
          | Some(@tensor.UInt8)
          | Some(@tensor.Int8)
          | None => true
          _ => false
        }
        let idx_ok = graph.initializers.contains(ins[1]) ||
          input_or_alias_i64.contains(ins[1])
        data_ok && idx_ok
      }
      @nn_ir.Split(..) => {
        let dt_ok = match dtypes.get(ins[0]) {
          Some(@tensor.Float32) | None => true
          _ => false
        }
        let sp_ok = ins.length() < 2 ||
          ins[1] == "" ||
          graph.initializers.contains(ins[1])
        dt_ok && sp_ok
      }
      @nn_ir.Concat(..) => {
        let mut all_f32 = ins.length() >= 1
        for k = 0; k < ins.length(); k = k + 1 {
          match dtypes.get(ins[k]) {
            Some(@tensor.Float32) | None => ()
            _ => all_f32 = false
          }
        }
        all_f32
      }
      @nn_ir.ReduceMean(..) =>
        match shapes.get(ins[0]) {
          Some(sx) if sx.length() >= 1 && sx.length() <= 4 =>
            ins.length() < 2 ||
            ins[1] == "" ||
            graph.initializers.contains(ins[1])
          _ => false
        }
      @nn_ir.QuantizeLinear(..) | @nn_ir.DequantizeLinear(..) => {
        let sc_ok = match graph.initializers.get(ins[1]) {
          Some(t) => t.numel() == 1
          None => false
        }
        let zp_ok = if ins.length() < 3 || ins[2] == "" {
          true
        } else {
          match graph.initializers.get(ins[2]) {
            Some(t) => t.numel() == 1
            None => false
          }
        }
        sc_ok && zp_ok
      }
      @nn_ir.QLinearMatMul =>
        if ins.length() >= 8 {
          match (shapes.get(ins[0]), shapes.get(ins[3]), dtypes.get(ins[0])) {
            (Some(sa), Some(sb), Some(@tensor.Int8 | @tensor.UInt8)) =>
              sa.length() == 2 &&
              sb.length() == 2 &&
              graph.initializers.contains(ins[1]) &&
              graph.initializers.contains(ins[2]) &&
              graph.initializers.contains(ins[3]) &&
              graph.initializers.contains(ins[4]) &&
              graph.initializers.contains(ins[5]) &&
              graph.initializers.contains(ins[6]) &&
              graph.initializers.contains(ins[7])
            _ => false
          }
        } else {
          false
        }
      _ => false
    }
    if !ok {
      return false
    }
  }
  true
}

///|
/// Arena 模式下生成零堆分配、零 `Tensor`、零 `BufferPool`、零 `onnx/ops` 依赖的极简高速 MoonBit 推理代码。
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

  let weight_entries : Array[WeightEntry] = []
  let weight_key_to_field : Map[String, String] = Map([])
  let raw_byte_chunks : Array[FixedArray[Byte]] = []
  let mut cur_byte_offset = 0

  fn reg_weight(name : String, kind : WeightKind) -> String raise CodegenError {
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
    match weight_key_to_field.get(key) {
      Some(f) => return f
      None => ()
    }
    let t = match graph.initializers.get(name) {
      Some(tensor) => tensor
      None => raise UnsupportedGraph("missing initializer '\{name}'")
    }
    let bytes_chunk : FixedArray[Byte] = match kind {
      RawBytes(..) | PackedInt8MatB =>
        match t.dtype() {
          @tensor.Int8 =>
            t.to_i8_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          @tensor.UInt8 =>
            t.to_u8_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          other =>
            raise TensorErr(
              "expected Int8/UInt8 weight for '\{name}', got \{other}",
            )
        }
      _ => t.to_f32_bytes_le() catch { e => raise TensorErr("\{e}") }
    }
    let field_name = "w_\{weight_entries.length()}"
    let byte_len = bytes_chunk.length()
    weight_entries.push({
      name,
      field_name,
      kind,
      shape: t.shape(),
      byte_offset: cur_byte_offset,
      byte_len,
    })
    weight_key_to_field[key] = field_name
    raw_byte_chunks.push(bytes_chunk)
    cur_byte_offset = cur_byte_offset + byte_len
    field_name
  }

  // 解析张量在 Arena 模式下的 (缓冲表达式, 起始偏移)
  fn resolve_f32_loc(name : String) -> (String, Int32) raise CodegenError {
    match input_param_of.get(name) {
      Some(pname) => return (pname, 0)
      None => ()
    }
    match f32_plan.allocs.get(name) {
      Some(a) => return ("self.arena_f32", a.offset)
      None => ()
    }
    if graph.initializers.contains(name) {
      let f = reg_weight(name, RawF32)
      return ("self.\{f}", 0)
    }
    raise UnsupportedGraph("arena: unresolved f32 tensor '\{name}'")
  }

  fn resolve_byte_loc(name : String) -> (String, Int32) raise CodegenError {
    match byte_plan.allocs.get(name) {
      Some(a) => return ("self.arena_u8", a.offset)
      None => ()
    }
    if graph.initializers.contains(name) {
      let f = reg_weight(name, RawBytes(trans_b=false))
      return ("self.\{f}", 0)
    }
    raise UnsupportedGraph("arena: unresolved byte tensor '\{name}'")
  }

  // 追踪需要按需生成的内联微辅助函数与最小导入集
  let mut need_gemm = false
  let mut need_conv = false
  let mut need_quant = false
  let mut need_math = false
  let mut need_maxpool = false
  let mut need_avgpool = false
  let mut need_gap = false
  let mut need_conv_scalar_nchw = false
  let mut need_softmax = false
  let mut need_layernorm = false
  let mut need_gelu = false
  let mut need_erf = false
  let mut need_reflect = false
  let mut need_f32_loader = false
  let mut need_byte_loader = false
  let mut need_trans_f32 = false

  let body = StringBuilder()

  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let ins = node.inputs
    let out_n = node.outputs[0]
    let out_sh = shapes.get(out_n).unwrap()
    let out_numel = @tensor.num_elements(out_sh) catch {
      e => raise ShapeError("\{e}")
    }

    // 零拷贝逻辑视图算子：编译期偏移直接复用，运行期 0 指令！
    if is_arena_alias_op(node.op) {
      body.write_string(
        "  // [\{i}] \{node.op.name()} -> \{@tensor.format_shape(out_sh)} (zero-cost compile-time alias)\n",
      )
      continue
    }

    body.write_string(
      "  // [\{i}] \{node.op.name()} -> \{@tensor.format_shape(out_sh)}\n",
    )

    match node.op {
      @nn_ir.Conv(
        strides~,
        pads~,
        dilations~,
        group~,
        auto_pad~,
        activation~,
        ..
      ) => {
        need_conv = true
        need_gemm = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let sw = shapes.get(ins[1]).unwrap()
        let is_1d = sx.length() == 3
        let n_batch = sx[0]
        let c_in = sx[1]
        let ih = if is_1d { 1 } else { sx[2] }
        let iw = if is_1d { sx[2] } else { sx[3] }
        let c_out = sw[0]
        let kh = if is_1d { 1 } else { sw[2] }
        let kw = if is_1d { sw[2] } else { sw[3] }
        let oh = if is_1d { 1 } else { out_sh[2] }
        let ow = if is_1d { out_sh[2] } else { out_sh[3] }
        let sh = if is_1d { 1 } else { strides[0] }
        let sw_s = if is_1d {
          if strides.length() == 1 {
            strides[0]
          } else {
            strides[1]
          }
        } else {
          strides[1]
        }
        let dh = if is_1d { 1 } else { dilations[0] }
        let dw_s = if is_1d {
          if dilations.length() == 1 {
            dilations[0]
          } else {
            dilations[1]
          }
        } else {
          dilations[1]
        }
        let pads2 = if is_1d {
          if pads.length() == 2 {
            [0, pads[0], 0, pads[1]]
          } else {
            [0, pads[1], 0, pads[3]]
          }
        } else {
          pads
        }
        let (pt, pl, _pb, _pr) = resolve_2d_pads(
          ih, iw, kh, kw, sh, sw_s, dh, dw_s, pads2, auto_pad, oh, ow,
        )
        let bias_expr = if ins.length() >= 3 && ins[2] != "" {
          need_f32_loader = true
          let bf = reg_weight(ins[2], RawF32)
          "Some(self.\{bf})"
        } else {
          "None"
        }
        let ep_expr = fmt_gemm_epilogue(activation)
        let is_dw = group > 1 && group == c_in && group == c_out
        if is_dw {
          need_f32_loader = true
          let wf = reg_weight(ins[1], RawF32)
          for b = 0; b < n_batch; b = b + 1 {
            let xb = x_off + b * c_in * ih * iw
            let yb = y_off + b * c_out * oh * ow
            body.write_string(
              "  @conv.depthwise_conv2d_f32(\{x_buf}, \{xb}, self.\{wf}, \{bias_expr}, \{c_in}, \{ih}, \{iw}, \{kh}, \{kw}, \{sh}, \{sw_s}, \{pt}, \{pl}, \{dh}, \{dw_s}, \{oh}, \{ow}, \{y_buf}, \{yb}, epilogue=\{ep_expr})\n",
            )
          }
        } else {
          let k_dim = c_in * kh * kw
          let spatial = oh * ow
          need_f32_loader = true
          if use_packed_f32_simd {
            let wf = reg_weight(ins[1], PackedConv(group=1))
            for b = 0; b < n_batch; b = b + 1 {
              let xb = x_off + b * c_in * ih * iw
              let yb = y_off + b * c_out * spatial
              body.write_string(
                "  @conv.im2col_2d_nhwc_f32(\{x_buf}, \{xb}, \{c_in}, \{ih}, \{iw}, \{kh}, \{kw}, \{sh}, \{sw_s}, \{pt}, \{pl}, \{dh}, \{dw_s}, \{oh}, \{ow}, self.scratch)\n",
              )
              body.write_string(
                "  @gemm.matmul_2d_prepacked_nchw_simd_f32(self.scratch, 0, self.\{wf}, \{bias_expr}, 0, \{y_buf}, \{yb}, \{spatial}, epilogue=\{ep_expr})\n",
              )
            }
          } else {
            need_conv_scalar_nchw = true
            let wf = reg_weight(ins[1], RawF32)
            for b = 0; b < n_batch; b = b + 1 {
              let xb = x_off + b * c_in * ih * iw
              let yb = y_off + b * c_out * spatial
              body.write_string(
                "  @conv.im2col_2d_nhwc_f32(\{x_buf}, \{xb}, \{c_in}, \{ih}, \{iw}, \{kh}, \{kw}, \{sh}, \{sw_s}, \{pt}, \{pl}, \{dh}, \{dw_s}, \{oh}, \{ow}, self.scratch)\n",
              )
              body.write_string(
                "  conv_nchw_scalar_f32(self.scratch, self.\{wf}, \{bias_expr}, \{y_buf}, \{yb}, \{spatial}, \{k_dim}, \{c_out}, \{ep_expr})\n",
              )
            }
          }
        }
      }
      @nn_ir.MaxPool(kernel_shape~, strides~, pads~, dilations~, auto_pad~, ..) => {
        need_maxpool = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let (pt, pl, _pb, _pr) = resolve_2d_pads(
          sx[2],
          sx[3],
          kernel_shape[0],
          kernel_shape[1],
          strides[0],
          strides[1],
          dilations[0],
          dilations[1],
          pads,
          auto_pad,
          out_sh[2],
          out_sh[3],
        )
        body.write_string(
          "  maxpool2d_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0]}, \{sx[1]}, \{sx[2]}, \{sx[3]}, \{kernel_shape[0]}, \{kernel_shape[1]}, \{strides[0]}, \{strides[1]}, \{pt}, \{pl}, \{dilations[0]}, \{dilations[1]}, \{out_sh[2]}, \{out_sh[3]})\n",
        )
      }
      @nn_ir.AveragePool(
        kernel_shape~,
        strides~,
        pads~,
        count_include_pad~,
        auto_pad~,
        ..
      ) => {
        need_avgpool = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let (pt, pl, _pb, _pr) = resolve_2d_pads(
          sx[2],
          sx[3],
          kernel_shape[0],
          kernel_shape[1],
          strides[0],
          strides[1],
          1,
          1,
          pads,
          auto_pad,
          out_sh[2],
          out_sh[3],
        )
        body.write_string(
          "  avgpool2d_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0]}, \{sx[1]}, \{sx[2]}, \{sx[3]}, \{kernel_shape[0]}, \{kernel_shape[1]}, \{strides[0]}, \{strides[1]}, \{pt}, \{pl}, \{out_sh[2]}, \{out_sh[3]}, \{count_include_pad})\n",
        )
      }
      @nn_ir.GlobalAveragePool => {
        need_gap = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let mut spatial = 1
        for d = 2; d < sx.length(); d = d + 1 {
          spatial = spatial * sx[d]
        }
        body.write_string(
          "  gap_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0] * sx[1]}, \{spatial})\n",
        )
      }
      @nn_ir.MatMul => {
        need_gemm = true
        let (a_buf, a_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sa = shapes.get(ins[0]).unwrap()
        let sb = shapes.get(ins[1]).unwrap()
        if sb.length() == 2 && graph.initializers.contains(ins[1]) {
          // N-D × 2D 静态权重投影（[B, S, K] @ [K, N] -> [B*S, K] @ [K, N]）
          need_f32_loader = true
          let k_dim = sa[sa.length() - 1]
          let n_dim = sb[1]
          let mut m_tot = 1
          for d = 0; d < sa.length() - 1; d = d + 1 {
            m_tot = m_tot * sa[d]
          }
          if use_packed_f32_simd {
            let wf = reg_weight(ins[1], PackedMatB(trans_b=false))
            body.write_string(
              "  @gemm.matmul_2d_prepacked_simd_f32(\{a_buf}, \{a_off}, self.\{wf}, \{y_buf}, \{y_off}, \{m_tot})\n",
            )
          } else {
            let wf = reg_weight(ins[1], RawF32)
            body.write_string(
              "  @gemm.matmul_2d_scalar_f32(\{a_buf}, \{a_off}, self.\{wf}, 0, \{y_buf}, \{y_off}, \{m_tot}, \{k_dim}, \{n_dim})\n",
            )
          }
        } else {
          // 2D/3D/4D 动态批量矩阵乘（多头自注意力 Q@K^T 与 P@V）
          let (b_buf, b_off) = resolve_f32_loc(ins[1])
          let r = sa.length()
          let m_dim = sa[r - 2]
          let k_dim = sa[r - 1]
          let n_dim = sb[r - 1]
          let mut batches = 1
          for d = 0; d < r - 2; d = d + 1 {
            batches = batches * sa[d]
          }
          let mat_a = m_dim * k_dim
          let mat_b = k_dim * n_dim
          let mat_y = m_dim * n_dim
          let mm_fn = if use_packed_f32_simd {
            "@gemm.matmul_2d_simd_f32"
          } else {
            "@gemm.matmul_2d_scalar_f32"
          }
          if batches == 1 {
            body.write_string(
              "  \{mm_fn}(\{a_buf}, \{a_off}, \{b_buf}, \{b_off}, \{y_buf}, \{y_off}, \{m_dim}, \{k_dim}, \{n_dim})\n",
            )
          } else {
            body.write_string(
              "  for bm_i = 0; bm_i < \{batches}; bm_i = bm_i + 1 { \{mm_fn}(\{a_buf}, \{a_off} + bm_i * \{mat_a}, \{b_buf}, \{b_off} + bm_i * \{mat_b}, \{y_buf}, \{y_off} + bm_i * \{mat_y}, \{m_dim}, \{k_dim}, \{n_dim}) }\n",
            )
          }
        }
      }
      @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
        need_gemm = true
        need_f32_loader = true
        let (a_buf, a_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sa = shapes.get(ins[0]).unwrap()
        let sb = shapes.get(ins[1]).unwrap()
        let m = if trans_a { sa[1] } else { sa[0] }
        let k = if trans_a { sa[0] } else { sa[1] }
        let n = if trans_b { sb[0] } else { sb[1] }
        if trans_a {
          raise UnsupportedOp("Arena Gemm does not support trans_a=true yet")
        }
        if use_packed_f32_simd {
          let wf = reg_weight(ins[1], PackedMatB(trans_b~))
          body.write_string(
            "  @gemm.matmul_2d_prepacked_simd_f32(\{a_buf}, \{a_off}, self.\{wf}, \{y_buf}, \{y_off}, \{m})\n",
          )
        } else {
          let wf = if trans_b {
            need_trans_f32 = true
            reg_weight(ins[1], PackedMatB(trans_b=true))
          } else {
            reg_weight(ins[1], RawF32)
          }
          body.write_string(
            "  @gemm.matmul_2d_scalar_f32(\{a_buf}, \{a_off}, self.\{wf}, 0, \{y_buf}, \{y_off}, \{m}, \{k}, \{n})\n",
          )
        }
        if alpha != (1.0 : Float) {
          body.write_string(
            "  for k_i = 0; k_i < \{m * n}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{y_buf}[\{y_off} + k_i] * \{fmt_float_literal(alpha)} }\n",
          )
        }
        if ins.length() >= 3 && ins[2] != "" {
          let (c_buf, c_off) = resolve_f32_loc(ins[2])
          let sc = shapes.get(ins[2]).unwrap()
          let c_numel = @tensor.num_elements(sc) catch {
            e => raise ShapeError("\{e}")
          }
          let beta_mul = if beta == (1.0 : Float) {
            ""
          } else {
            " * \{fmt_float_literal(beta)}"
          }
          if c_numel == 1 {
            body.write_string(
              "  for k_i = 0; k_i < \{m * n}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{y_buf}[\{y_off} + k_i] + \{c_buf}[\{c_off}]\{beta_mul} }\n",
            )
          } else if c_numel == n {
            body.write_string(
              "  for r_i = 0; r_i < \{m}; r_i = r_i + 1 { for c_i = 0; c_i < \{n}; c_i = c_i + 1 { \{y_buf}[\{y_off} + r_i * \{n} + c_i] = \{y_buf}[\{y_off} + r_i * \{n} + c_i] + \{c_buf}[\{c_off} + c_i]\{beta_mul} } }\n",
            )
          } else {
            body.write_string(
              "  for k_i = 0; k_i < \{m * n}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{y_buf}[\{y_off} + k_i] + \{c_buf}[\{c_off} + k_i]\{beta_mul} }\n",
            )
          }
        }
      }
      @nn_ir.Add | @nn_ir.Sub | @nn_ir.Mul | @nn_ir.Div => {
        let (a_buf, a_off) = resolve_f32_loc(ins[0])
        let (b_buf, b_off) = resolve_f32_loc(ins[1])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sa = shapes.get(ins[0]).unwrap()
        let sb = shapes.get(ins[1]).unwrap()
        let a_numel = @tensor.num_elements(sa) catch {
          e => raise ShapeError("\{e}")
        }
        let b_numel = @tensor.num_elements(sb) catch {
          e => raise ShapeError("\{e}")
        }
        let op_sym = match node.op {
          @nn_ir.Add => "+"
          @nn_ir.Sub => "-"
          @nn_ir.Mul => "*"
          _ => "/"
        }
        if sa == sb {
          body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{a_buf}[\{a_off} + k_i] \{op_sym} \{b_buf}[\{b_off} + k_i] }\n",
          )
        } else if b_numel == 1 {
          body.write_string("  let bs_v = \{b_buf}[\{b_off}]\n")
          body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{a_buf}[\{a_off} + k_i] \{op_sym} bs_v }\n",
          )
        } else if a_numel == 1 {
          body.write_string("  let as_v = \{a_buf}[\{a_off}]\n")
          body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = as_v \{op_sym} \{b_buf}[\{b_off} + k_i] }\n",
          )
        } else if sb.length() == 1 &&
          out_sh.length() >= 1 &&
          sa == out_sh &&
          sb[0] == out_sh[out_sh.length() - 1] {
          let cols = out_sh[out_sh.length() - 1]
          let rows = out_numel / cols
          body.write_string(
            "  for r_i = 0; r_i < \{rows}; r_i = r_i + 1 { let base_i = r_i * \{cols}; for c_i = 0; c_i < \{cols}; c_i = c_i + 1 { \{y_buf}[\{y_off} + base_i + c_i] = \{a_buf}[\{a_off} + base_i + c_i] \{op_sym} \{b_buf}[\{b_off} + c_i] } }\n",
          )
        } else if sa.length() == 1 &&
          out_sh.length() >= 1 &&
          sb == out_sh &&
          sa[0] == out_sh[out_sh.length() - 1] {
          // 对称偏置加：[C] op [..., C]（PyTorch Transformer 线性层偏置常见顺序）
          let cols = out_sh[out_sh.length() - 1]
          let rows = out_numel / cols
          body.write_string(
            "  for r_i = 0; r_i < \{rows}; r_i = r_i + 1 { let base_i = r_i * \{cols}; for c_i = 0; c_i < \{cols}; c_i = c_i + 1 { \{y_buf}[\{y_off} + base_i + c_i] = \{a_buf}[\{a_off} + c_i] \{op_sym} \{b_buf}[\{b_off} + base_i + c_i] } }\n",
          )
        } else {
          // 通用 1D~4D 静态展开广播（例如 Attention Mask [1,12,S,S] + [1,1,1,S]）
          let rank = out_sh.length()
          let a_bs = @tensor.broadcast_strides(
            sa,
            @tensor.compute_strides(sa),
            out_sh,
          ) catch {
            e => raise TensorErr("\{e}")
          }
          let b_bs = @tensor.broadcast_strides(
            sb,
            @tensor.compute_strides(sb),
            out_sh,
          ) catch {
            e => raise TensorErr("\{e}")
          }
          let o_st = @tensor.compute_strides(out_sh)
          for d = 0; d < rank; d = d + 1 {
            body.write_string(
              "  for bd_\{d} = 0; bd_\{d} < \{out_sh[d]}; bd_\{d} = bd_\{d} + 1 {\n",
            )
          }
          let ea = StringBuilder()
          let eb = StringBuilder()
          let ey = StringBuilder()
          ea.write_string("\{a_off}")
          eb.write_string("\{b_off}")
          ey.write_string("\{y_off}")
          for d = 0; d < rank; d = d + 1 {
            if a_bs[d] != 0 {
              ea.write_string(" + bd_\{d} * \{a_bs[d]}")
            }
            if b_bs[d] != 0 {
              eb.write_string(" + bd_\{d} * \{b_bs[d]}")
            }
            ey.write_string(" + bd_\{d} * \{o_st[d]}")
          }
          body.write_string(
            "    \{y_buf}[\{ey.to_string()}] = \{a_buf}[\{ea.to_string()}] \{op_sym} \{b_buf}[\{eb.to_string()}]\n",
          )
          for _d = 0; _d < rank; _d = _d + 1 {
            body.write_string("  }\n")
          }
        }
      }
      @nn_ir.Relu => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; \{y_buf}[\{y_off} + k_i] = if v > (0.0 : Float) { v } else { (0.0 : Float) } }\n",
        )
      }
      @nn_ir.Clip(min~, max~) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let lo_v : Float? = if ins.length() >= 2 && ins[1] != "" {
          Some(
            graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
              e => raise TensorErr("\{e}")
            },
          )
        } else {
          min
        }
        let hi_v : Float? = if ins.length() >= 3 && ins[2] != "" {
          Some(
            graph.initializers.get(ins[2]).unwrap().get_f32_flat(0) catch {
              e => raise TensorErr("\{e}")
            },
          )
        } else {
          max
        }
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 {\n",
        )
        body.write_string("    let mut v = \{x_buf}[\{x_off} + k_i]\n")
        match lo_v {
          Some(lo) =>
            body.write_string(
              "    if v < \{fmt_float_literal(lo)} { v = \{fmt_float_literal(lo)} }\n",
            )
          None => ()
        }
        match hi_v {
          Some(hi) =>
            body.write_string(
              "    if v > \{fmt_float_literal(hi)} { v = \{fmt_float_literal(hi)} }\n",
            )
          None => ()
        }
        body.write_string("    \{y_buf}[\{y_off} + k_i] = v\n  }\n")
      }
      @nn_ir.LeakyRelu(alpha~) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let al = fmt_float_literal(alpha)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; \{y_buf}[\{y_off} + k_i] = if v >= (0.0 : Float) { v } else { v * \{al} } }\n",
        )
      }
      @nn_ir.HardSwish => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sixth = fmt_float_literal(Float::from_double(1.0 / 6.0))
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; let t = v * \{sixth} + (0.5 : Float); let c = if t < (0.0 : Float) { (0.0 : Float) } else if t > (1.0 : Float) { (1.0 : Float) } else { t }; \{y_buf}[\{y_off} + k_i] = v * c }\n",
        )
      }
      @nn_ir.Neg => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = -\{x_buf}[\{x_off} + k_i] }\n",
        )
      }
      @nn_ir.Sqrt => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(\{x_buf}[\{x_off} + k_i].to_double().sqrt()) }\n",
        )
      }
      @nn_ir.Exp => {
        need_math = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.exp(\{x_buf}[\{x_off} + k_i].to_double())) }\n",
        )
      }
      @nn_ir.Sigmoid => {
        need_math = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(1.0 / (1.0 + @math.exp(-\{x_buf}[\{x_off} + k_i].to_double()))) }\n",
        )
      }
      @nn_ir.Tanh => {
        need_math = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.tanh(\{x_buf}[\{x_off} + k_i].to_double())) }\n",
        )
      }
      @nn_ir.Gelu => {
        need_math = true
        need_gelu = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  gelu_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
        )
      }
      @nn_ir.Erf => {
        need_math = true
        need_erf = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        body.write_string(
          "  erf_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
        )
      }
      @nn_ir.Softmax(axis~) => {
        need_math = true
        need_softmax = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let rank = out_sh.length()
        let ax = if axis < 0 { axis + rank } else { axis }
        let mut outer = 1
        for d = 0; d < ax; d = d + 1 {
          outer = outer * out_sh[d]
        }
        let axis_dim = out_sh[ax]
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * out_sh[d]
        }
        body.write_string(
          "  softmax_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{outer}, \{axis_dim}, \{inner})\n",
        )
      }
      @nn_ir.LayerNormalization(axis~, epsilon~) => {
        need_layernorm = true
        need_f32_loader = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sf = reg_weight(ins[1], RawF32)
        let bias_expr = if ins.length() >= 3 && ins[2] != "" {
          let bf = reg_weight(ins[2], RawF32)
          "Some(self.\{bf})"
        } else {
          "None"
        }
        let rank = out_sh.length()
        let ax = if axis < 0 { axis + rank } else { axis }
        let mut outer = 1
        for d = 0; d < ax; d = d + 1 {
          outer = outer * out_sh[d]
        }
        let mut norm_sz = 1
        for d = ax; d < rank; d = d + 1 {
          norm_sz = norm_sz * out_sh[d]
        }
        body.write_string(
          "  layernorm_f32(\{x_buf}, \{x_off}, self.\{sf}, \{bias_expr}, \{y_buf}, \{y_off}, \{outer}, \{norm_sz}, \{fmt_float_literal(epsilon)})\n",
        )
      }
      @nn_ir.QuantizeLinear(..) => {
        need_quant = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_byte_loc(out_n)
        let sc = graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          }
        let out_dt = dtypes.get(out_n).unwrap()
        let zp_byte : Byte = if ins.length() >= 3 && ins[2] != "" {
          let zt = graph.initializers.get(ins[2]).unwrap()
          match zt.dtype() {
            @tensor.Int8 =>
              (zt.to_i8_buffer() catch { e => raise TensorErr("\{e}") })[0]
            _ => (zt.to_u8_buffer() catch { e => raise TensorErr("\{e}") })[0]
          }
        } else {
          b'\x00'
        }
        let q_fn = match out_dt {
          @tensor.Int8 => "@quant.quantize_i8_per_tensor"
          _ => "@quant.quantize_u8_per_tensor"
        }
        body.write_string(
          "  \{q_fn}(\{x_buf}, \{fmt_float_literal(sc)}, (\{zp_byte.to_int()}).to_byte(), \{y_buf}, \{out_numel}, src_off=\{x_off}, dst_off=\{y_off})\n",
        )
      }
      @nn_ir.DequantizeLinear(..) => {
        need_quant = true
        let (x_buf, x_off) = resolve_byte_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sc = graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          }
        let in_dt = dtypes.get(ins[0]).unwrap()
        let zp_byte : Byte = if ins.length() >= 3 && ins[2] != "" {
          let zt = graph.initializers.get(ins[2]).unwrap()
          match zt.dtype() {
            @tensor.Int8 =>
              (zt.to_i8_buffer() catch { e => raise TensorErr("\{e}") })[0]
            _ => (zt.to_u8_buffer() catch { e => raise TensorErr("\{e}") })[0]
          }
        } else {
          b'\x00'
        }
        let dq_fn = match in_dt {
          @tensor.Int8 => "@quant.dequantize_i8_per_tensor"
          _ => "@quant.dequantize_u8_per_tensor"
        }
        body.write_string(
          "  \{dq_fn}(\{x_buf}, \{fmt_float_literal(sc)}, (\{zp_byte.to_int()}).to_byte(), \{y_buf}, \{out_numel}, src_off=\{x_off}, dst_off=\{y_off})\n",
        )
      }
      @nn_ir.QLinearMatMul => {
        need_quant = true
        need_byte_loader = true
        let (a_buf, a_off) = resolve_byte_loc(ins[0])
        let (y_buf, y_off) = resolve_byte_loc(out_n)
        let sa = shapes.get(ins[0]).unwrap()
        let sb = shapes.get(ins[3]).unwrap()
        let m = sa[0]
        let k = sa[1]
        let n = sb[1]
        let a_dt = dtypes.get(ins[0]).unwrap()
        let a_sc = graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          }
        let b_sc = graph.initializers.get(ins[4]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          }
        let y_sc = graph.initializers.get(ins[6]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          }
        let a_zp : Int32 = match a_dt {
          @tensor.Int8 =>
            (graph.initializers.get(ins[2]).unwrap().to_i8_buffer() catch {
              e => raise TensorErr("\{e}")
            })[0].to_int()
          _ =>
            (graph.initializers.get(ins[2]).unwrap().to_u8_buffer() catch {
              e => raise TensorErr("\{e}")
            })[0].to_int()
        }
        let b_zp : Int32 = match a_dt {
          @tensor.Int8 =>
            (graph.initializers.get(ins[5]).unwrap().to_i8_buffer() catch {
              e => raise TensorErr("\{e}")
            })[0].to_int()
          _ =>
            (graph.initializers.get(ins[5]).unwrap().to_u8_buffer() catch {
              e => raise TensorErr("\{e}")
            })[0].to_int()
        }
        let y_zp : Int32 = match a_dt {
          @tensor.Int8 =>
            (graph.initializers.get(ins[7]).unwrap().to_i8_buffer() catch {
              e => raise TensorErr("\{e}")
            })[0].to_int()
          _ =>
            (graph.initializers.get(ins[7]).unwrap().to_u8_buffer() catch {
              e => raise TensorErr("\{e}")
            })[0].to_int()
        }
        if use_quant_simd {
          let wf = reg_weight(ins[3], RawBytes(trans_b=true))
          let qmm_fn = match a_dt {
            @tensor.Int8 => "@quant.qlinear_matmul_i8_pretransposed_v128"
            _ => "@quant.qlinear_matmul_u8_pretransposed_v128"
          }
          body.write_string(
            "  \{qmm_fn}(\{a_buf}, \{a_off}, \{fmt_float_literal(a_sc)}, (\{a_zp}).to_byte(), self.\{wf}, \{fmt_float_literal(b_sc)}, (\{b_zp}).to_byte(), \{fmt_float_literal(y_sc)}, (\{y_zp}).to_byte(), \{y_buf}, \{y_off}, \{m}, \{k}, \{n})\n",
          )
        } else {
          let wf = reg_weight(ins[3], RawBytes(trans_b=false))
          let qmm_fn = match a_dt {
            @tensor.Int8 => "@quant.qlinear_matmul_i8_scalar"
            _ => "@quant.qlinear_matmul_u8_scalar"
          }
          body.write_string(
            "  \{qmm_fn}(\{a_buf}, \{fmt_float_literal(a_sc)}, (\{a_zp}).to_byte(), self.\{wf}, \{fmt_float_literal(b_sc)}, (\{b_zp}).to_byte(), \{fmt_float_literal(y_sc)}, (\{y_zp}).to_byte(), \{y_buf}, \{m}, \{k}, \{n}, a_off=\{a_off}, out_off=\{y_off})\n",
          )
        }
      }
      @nn_ir.DynamicQuantizeLinear => {
        need_quant = true
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_byte_loc(node.outputs[0])
        let (sc_buf, sc_off) = resolve_f32_loc(node.outputs[1])
        let (zp_buf, zp_off) = resolve_byte_loc(node.outputs[2])
        body.write_string(
          "  let (dq_s_\{i}, dq_z_\{i}) = @quant.dynamic_quantize_linear_u8(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
        )
        body.write_string("  \{sc_buf}[\{sc_off}] = dq_s_\{i}\n")
        body.write_string("  \{zp_buf}[\{zp_off}] = dq_z_\{i}\n")
      }
      @nn_ir.MatMulInteger => {
        need_quant = true
        need_byte_loader = true
        let (a_buf, a_off) = resolve_byte_loc(ins[0])
        let cast_out = mmi_to_cast.get(out_n).unwrap()
        let (y_buf, y_off) = resolve_f32_loc(cast_out)
        let sa = shapes.get(ins[0]).unwrap()
        let mut m_tot = 1
        for d = 0; d < sa.length() - 1; d = d + 1 {
          m_tot = m_tot * sa[d]
        }
        let wf = reg_weight(ins[1], PackedInt8MatB)
        let a_zp_expr = if ins.length() >= 3 && ins[2] != "" {
          let (az_buf, az_off) = resolve_byte_loc(ins[2])
          "\{az_buf}[\{az_off}]"
        } else {
          "b'\\x00'"
        }
        let b_zp_expr = if ins.length() >= 4 && ins[3] != "" {
          let bzt = graph.initializers.get(ins[3]).unwrap()
          let bzb = bzt.to_i8_buffer() catch { e => raise TensorErr("\{e}") }
          let mut any_nz = false
          for bi = 0; bi < bzb.length(); bi = bi + 1 {
            if bzb[bi] != b'\x00' {
              any_nz = true
              break
            }
          }
          if any_nz {
            let bzf = reg_weight(ins[3], RawBytes(trans_b=false))
            "Some(self.\{bzf})"
          } else {
            "None"
          }
        } else {
          "None"
        }
        let mmi_fn = if use_quant_simd {
          "@quant.matmul_integer_u8_i8_prepacked_to_f32_v128"
        } else {
          "@quant.matmul_integer_u8_i8_prepacked_to_f32_scalar"
        }
        body.write_string(
          "  \{mmi_fn}(\{a_buf}, \{a_off}, \{a_zp_expr}, self.\{wf}, \{b_zp_expr}, \{y_buf}, \{y_off}, \{m_tot})\n",
        )
      }
      @nn_ir.Pow => {
        let (a_buf, a_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sa = shapes.get(ins[0]).unwrap()
        let sb = shapes.get(ins[1]).unwrap()
        let mut emitted_sq = false
        if sb.length() == 0 || (sb.length() == 1 && sb[0] == 1) {
          match graph.initializers.get(ins[1]) {
            Some(bt) => {
              let exp_v = bt.get_f32_flat(0) catch {
                e => raise TensorErr("\{e}")
              }
              if exp_v == (2.0 : Float) {
                // 快速路径：x^2 直接展开为乘法（如 STFT 幅值计算），避免调用 `@math.pow`
                body.write_string(
                  "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{a_buf}[\{a_off} + k_i]; \{y_buf}[\{y_off} + k_i] = v * v }\n",
                )
                emitted_sq = true
              } else {
                need_math = true
                let exp_d = exp_v.to_double()
                body.write_string(
                  "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.pow(\{a_buf}[\{a_off} + k_i].to_double(), \{exp_d})) }\n",
                )
                emitted_sq = true
              }
            }
            None => ()
          }
        }
        if !emitted_sq {
          need_math = true
          let (b_buf, b_off) = resolve_f32_loc(ins[1])
          if sa == sb {
            body.write_string(
              "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.pow(\{a_buf}[\{a_off} + k_i].to_double(), \{b_buf}[\{b_off} + k_i].to_double())) }\n",
            )
          } else {
            body.write_string("  let exp_s = \{b_buf}[\{b_off}].to_double()\n")
            body.write_string(
              "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.pow(\{a_buf}[\{a_off} + k_i].to_double(), exp_s)) }\n",
            )
          }
        }
      }
      @nn_ir.Pad(mode~, pads~, constant_value~) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let rank = sx.length()
        let raw_pads : Array[Int32] = if ins.length() >= 2 && ins[1] != "" {
          let b = graph.initializers.get(ins[1]).unwrap().to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          Array::makei(b.length(), fn(k) { b[k].to_int() })
        } else {
          pads.unwrap()
        }
        let pb = Array::make(rank, 0)
        if ins.length() >= 4 && ins[3] != "" {
          let ab = graph.initializers.get(ins[3]).unwrap().to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          let k_ax = ab.length()
          for k = 0; k < k_ax; k = k + 1 {
            let ax = ab[k].to_int()
            let norm = if ax < 0 { ax + rank } else { ax }
            pb[norm] = raw_pads[k]
          }
        } else {
          for d = 0; d < rank; d = d + 1 {
            pb[d] = raw_pads[d]
          }
        }
        let cval : Float = if ins.length() >= 3 && ins[2] != "" {
          graph.initializers.get(ins[2]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          }
        } else {
          match constant_value {
            Some(v) => v
            None => (0.0 : Float)
          }
        }
        let mode_id = match mode {
          "" | "constant" => 0
          "reflect" => {
            need_reflect = true
            1
          }
          "edge" => 2
          other => raise UnsupportedOp("arena Pad unsupported mode '\{other}'")
        }
        let in_strides = @tensor.compute_strides(sx)
        let out_strides = @tensor.compute_strides(out_sh)
        // 展开 1D~4D 静态循环实现严格 0 堆分配的 Pad
        for d = 0; d < rank; d = d + 1 {
          body.write_string(
            "  for p_\{d} = 0; p_\{d} < \{out_sh[d]}; p_\{d} = p_\{d} + 1 {\n",
          )
        }
        if mode_id == 0 {
          body.write_string("    let mut oob = false\n")
        }
        body.write_string("    let mut s_idx = \{x_off}\n")
        for d = 0; d < rank; d = d + 1 {
          body.write_string("    let ic_\{d} = p_\{d} - \{pb[d]}\n")
          if mode_id == 0 {
            body.write_string(
              "    if ic_\{d} < 0 || ic_\{d} >= \{sx[d]} { oob = true } else { s_idx = s_idx + ic_\{d} * \{in_strides[d]} }\n",
            )
          } else if mode_id == 1 {
            body.write_string(
              "    let rc_\{d} = if ic_\{d} < 0 || ic_\{d} >= \{sx[d]} { reflect_idx(ic_\{d}, \{sx[d]}) } else { ic_\{d} }\n",
            )
            body.write_string(
              "    s_idx = s_idx + rc_\{d} * \{in_strides[d]}\n",
            )
          } else {
            body.write_string(
              "    let ec_\{d} = if ic_\{d} < 0 { 0 } else if ic_\{d} >= \{sx[d]} { \{sx[d] - 1} } else { ic_\{d} }\n",
            )
            body.write_string(
              "    s_idx = s_idx + ec_\{d} * \{in_strides[d]}\n",
            )
          }
        }
        let dst_expr = StringBuilder()
        dst_expr.write_string("\{y_off}")
        for d = 0; d < rank; d = d + 1 {
          dst_expr.write_string(" + p_\{d} * \{out_strides[d]}")
        }
        if mode_id == 0 {
          body.write_string(
            "    \{y_buf}[\{dst_expr.to_string()}] = if oob { \{fmt_float_literal(cval)} } else { \{x_buf}[s_idx] }\n",
          )
        } else {
          body.write_string(
            "    \{y_buf}[\{dst_expr.to_string()}] = \{x_buf}[s_idx]\n",
          )
        }
        for _d = 0; _d < rank; _d = _d + 1 {
          body.write_string("  }\n")
        }
      }
      @nn_ir.Slice(starts~, ends~, axes~, steps~) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let rank = sx.length()
        let s_arr : Array[Int64] = if ins.length() >= 2 && ins[1] != "" {
          let b = graph.initializers.get(ins[1]).unwrap().to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          Array::makei(b.length(), fn(k) { b[k] })
        } else {
          let v = starts.unwrap()
          Array::makei(v.length(), fn(k) { v[k].to_int64() })
        }
        let a_opt : Array[Int32]? = if ins.length() >= 4 && ins[3] != "" {
          let b = graph.initializers.get(ins[3]).unwrap().to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
        } else {
          axes
        }
        let st_opt : Array[Int32]? = if ins.length() >= 5 && ins[4] != "" {
          let b = graph.initializers.get(ins[4]).unwrap().to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
        } else {
          steps
        }
        let _ = ends
        let ax_start = Array::make(rank, 0)
        let ax_step = Array::make(rank, 1)
        for k = 0; k < s_arr.length(); k = k + 1 {
          let raw_ax = match a_opt {
            Some(a) => a[k]
            None => k
          }
          let ax = if raw_ax < 0 { raw_ax + rank } else { raw_ax }
          let step = match st_opt {
            Some(st) => st[k]
            None => 1
          }
          ax_start[ax] = clamp_slice_bound_static(s_arr[k], sx[ax], step)
          ax_step[ax] = step
        }
        let in_strides = @tensor.compute_strides(sx)
        let out_strides = @tensor.compute_strides(out_sh)
        for d = 0; d < rank; d = d + 1 {
          body.write_string(
            "  for sl_\{d} = 0; sl_\{d} < \{out_sh[d]}; sl_\{d} = sl_\{d} + 1 {\n",
          )
        }
        let src_expr = StringBuilder()
        let dst_expr = StringBuilder()
        src_expr.write_string("\{x_off}")
        dst_expr.write_string("\{y_off}")
        for d = 0; d < rank; d = d + 1 {
          src_expr.write_string(
            " + (\{ax_start[d]} + sl_\{d} * \{ax_step[d]}) * \{in_strides[d]}",
          )
          dst_expr.write_string(" + sl_\{d} * \{out_strides[d]}")
        }
        body.write_string(
          "    \{y_buf}[\{dst_expr.to_string()}] = \{x_buf}[\{src_expr.to_string()}]\n",
        )
        for _d = 0; _d < rank; _d = _d + 1 {
          body.write_string("  }\n")
        }
      }
      @nn_ir.Transpose(perm~) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let rank = sx.length()
        let eff_perm = if perm.length() == rank {
          perm
        } else {
          Array::makei(rank, fn(d) { rank - 1 - d })
        }
        let in_strides = @tensor.compute_strides(sx)
        let out_strides = @tensor.compute_strides(out_sh)
        // 展开 1D~4D 静态转置循环，零堆分配完成多头注意力 [B,S,H,D] <-> [B,H,S,D] 等重排
        for d = 0; d < rank; d = d + 1 {
          body.write_string(
            "  for tr_\{d} = 0; tr_\{d} < \{out_sh[d]}; tr_\{d} = tr_\{d} + 1 {\n",
          )
        }
        let src_expr = StringBuilder()
        let dst_expr = StringBuilder()
        src_expr.write_string("\{x_off}")
        dst_expr.write_string("\{y_off}")
        for d = 0; d < rank; d = d + 1 {
          src_expr.write_string(" + tr_\{d} * \{in_strides[eff_perm[d]]}")
          dst_expr.write_string(" + tr_\{d} * \{out_strides[d]}")
        }
        body.write_string(
          "    \{y_buf}[\{dst_expr.to_string()}] = \{x_buf}[\{src_expr.to_string()}]\n",
        )
        for _d = 0; _d < rank; _d = _d + 1 {
          body.write_string("  }\n")
        }
      }
      @nn_ir.Cast(to~) => {
        let _ = to
        if skipped_cast.contains(out_n) {
          body.write_string("  // fused directly into MatMulInteger output\n")
        } else {
          let in_param = input_param_of.get(ins[0]).unwrap()
          let (y_buf, y_off) = resolve_f32_loc(out_n)
          body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_int(\{in_param}[k_i].to_int()) }\n",
          )
        }
      }
      @nn_ir.Gather(axis~) => {
        let is_byte_gather = match dtypes.get(ins[0]) {
          Some(@tensor.UInt8) | Some(@tensor.Int8) => true
          _ => false
        }
        let (x_buf, x_off) = if is_byte_gather {
          resolve_byte_loc(ins[0])
        } else {
          resolve_f32_loc(ins[0])
        }
        let (y_buf, y_off) = if is_byte_gather {
          resolve_byte_loc(out_n)
        } else {
          resolve_f32_loc(out_n)
        }
        let sd = shapes.get(ins[0]).unwrap()
        let rank = sd.length()
        let ax = if axis < 0 { axis + rank } else { axis }
        let axis_dim = sd[ax]
        let mut outer = 1
        for d = 0; d < ax; d = d + 1 {
          outer = outer * sd[d]
        }
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * sd[d]
        }
        match graph.initializers.get(ins[1]) {
          Some(idx_t) => {
            let idx_buf = idx_t.to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
            let idx_len = idx_buf.length()
            for k = 0; k < idx_len; k = k + 1 {
              let raw = idx_buf[k].to_int()
              let norm = if raw < 0 { raw + axis_dim } else { raw }
              if outer == 1 {
                body.write_string(
                  "  for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[\{y_off + k * inner} + g_j] = \{x_buf}[\{x_off + norm * inner} + g_j] }\n",
                )
              } else {
                body.write_string(
                  "  for g_o = 0; g_o < \{outer}; g_o = g_o + 1 { for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[\{y_off} + (g_o * \{idx_len} + \{k}) * \{inner} + g_j] = \{x_buf}[\{x_off} + (g_o * \{axis_dim} + \{norm}) * \{inner} + g_j] } }\n",
                )
              }
            }
          }
          None => {
            // 动态 Int64 输入索引查表（如 Transformer Embedding 表查表）
            let idx_param = input_param_of.get(ins[1]).unwrap()
            let idx_sh = shapes.get(ins[1]).unwrap()
            let idx_len = @tensor.num_elements(idx_sh) catch {
              e => raise ShapeError("\{e}")
            }
            if outer == 1 {
              body.write_string(
                "  for g_k = 0; g_k < \{idx_len}; g_k = g_k + 1 { let raw_idx = \{idx_param}[g_k].to_int(); let norm_idx = if raw_idx < 0 { raw_idx + \{axis_dim} } else { raw_idx }; let src_b = \{x_off} + norm_idx * \{inner}; let dst_b = \{y_off} + g_k * \{inner}; for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[dst_b + g_j] = \{x_buf}[src_b + g_j] } }\n",
              )
            } else {
              body.write_string(
                "  for g_o = 0; g_o < \{outer}; g_o = g_o + 1 { for g_k = 0; g_k < \{idx_len}; g_k = g_k + 1 { let raw_idx = \{idx_param}[g_k].to_int(); let norm_idx = if raw_idx < 0 { raw_idx + \{axis_dim} } else { raw_idx }; let src_b = \{x_off} + (g_o * \{axis_dim} + norm_idx) * \{inner}; let dst_b = \{y_off} + (g_o * \{idx_len} + g_k) * \{inner}; for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[dst_b + g_j] = \{x_buf}[src_b + g_j] } } }\n",
              )
            }
          }
        }
      }
      @nn_ir.Split(axis~, ..) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let sx = shapes.get(ins[0]).unwrap()
        let rank = sx.length()
        let ax = if axis < 0 { axis + rank } else { axis }
        let axis_dim = sx[ax]
        let mut outer = 1
        for d = 0; d < ax; d = d + 1 {
          outer = outer * sx[d]
        }
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * sx[d]
        }
        let mut axis_off = 0
        for p = 0; p < node.outputs.length(); p = p + 1 {
          let pn = node.outputs[p]
          if pn == "" {
            continue
          }
          let psh = shapes.get(pn).unwrap()
          let part_dim = psh[ax]
          let part_inner = part_dim * inner
          let (p_buf, p_off) = resolve_f32_loc(pn)
          if outer == 1 {
            body.write_string(
              "  for sp_k = 0; sp_k < \{part_inner}; sp_k = sp_k + 1 { \{p_buf}[\{p_off} + sp_k] = \{x_buf}[\{x_off + axis_off * inner} + sp_k] }\n",
            )
          } else {
            body.write_string(
              "  for sp_o = 0; sp_o < \{outer}; sp_o = sp_o + 1 { for sp_k = 0; sp_k < \{part_inner}; sp_k = sp_k + 1 { \{p_buf}[\{p_off} + sp_o * \{part_inner} + sp_k] = \{x_buf}[\{x_off} + (sp_o * \{axis_dim} + \{axis_off}) * \{inner} + sp_k] } }\n",
            )
          }
          axis_off = axis_off + part_dim
        }
      }
      @nn_ir.Concat(axis~) => {
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let s0 = shapes.get(ins[0]).unwrap()
        let rank = s0.length()
        let ax = if axis < 0 { axis + rank } else { axis }
        let total_axis = out_sh[ax]
        let mut outer = 1
        for d = 0; d < ax; d = d + 1 {
          outer = outer * s0[d]
        }
        let mut inner = 1
        for d = ax + 1; d < rank; d = d + 1 {
          inner = inner * s0[d]
        }
        let mut dst_axis_off = 0
        for t_idx = 0; t_idx < ins.length(); t_idx = t_idx + 1 {
          let (s_buf, s_off) = resolve_f32_loc(ins[t_idx])
          let ts = shapes.get(ins[t_idx]).unwrap()
          let chunk = ts[ax] * inner
          if outer == 1 {
            body.write_string(
              "  for cc_k = 0; cc_k < \{chunk}; cc_k = cc_k + 1 { \{y_buf}[\{y_off + dst_axis_off * inner} + cc_k] = \{s_buf}[\{s_off} + cc_k] }\n",
            )
          } else {
            body.write_string(
              "  for cc_o = 0; cc_o < \{outer}; cc_o = cc_o + 1 { for cc_k = 0; cc_k < \{chunk}; cc_k = cc_k + 1 { \{y_buf}[\{y_off} + (cc_o * \{total_axis} + \{dst_axis_off}) * \{inner} + cc_k] = \{s_buf}[\{s_off} + cc_o * \{chunk} + cc_k] } }\n",
            )
          }
          dst_axis_off = dst_axis_off + ts[ax]
        }
      }
      @nn_ir.ReduceMean(axes~, ..) => {
        let (x_buf, x_off) = resolve_f32_loc(ins[0])
        let (y_buf, y_off) = resolve_f32_loc(out_n)
        let sx = shapes.get(ins[0]).unwrap()
        let rank = sx.length()
        let ax_opt : Array[Int32]? = if ins.length() >= 2 && ins[1] != "" {
          let b = graph.initializers.get(ins[1]).unwrap().to_i64_buffer() catch {
              e => raise TensorErr("\{e}")
            }
          Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
        } else {
          axes
        }
        let is_red = Array::make(rank, false)
        match ax_opt {
          Some(ax_arr) =>
            for k = 0; k < ax_arr.length(); k = k + 1 {
              let norm = if ax_arr[k] < 0 {
                ax_arr[k] + rank
              } else {
                ax_arr[k]
              }
              is_red[norm] = true
            }
          None =>
            for k = 0; k < rank; k = k + 1 {
              is_red[k] = true
            }
        }
        let mut red_count = 1
        for d = 0; d < rank; d = d + 1 {
          if is_red[d] {
            red_count = red_count * sx[d]
          }
        }
        let in_strides = @tensor.compute_strides(sx)
        // 计算保留轴的输出步长
        let keep_axes : Array[Int32] = []
        let red_axes : Array[Int32] = []
        for d = 0; d < rank; d = d + 1 {
          if is_red[d] {
            red_axes.push(d)
          } else {
            keep_axes.push(d)
          }
        }
        let keep_dims = Array::makei(keep_axes.length(), fn(k) {
          sx[keep_axes[k]]
        })
        let keep_strides = @tensor.compute_strides(keep_dims)
        for k = 0; k < keep_axes.length(); k = k + 1 {
          let d = keep_axes[k]
          body.write_string(
            "  for rk_\{d} = 0; rk_\{d} < \{sx[d]}; rk_\{d} = rk_\{d} + 1 {\n",
          )
        }
        body.write_string("    let mut r_sum = 0.0\n")
        for k = 0; k < red_axes.length(); k = k + 1 {
          let d = red_axes[k]
          body.write_string(
            "    for rr_\{d} = 0; rr_\{d} < \{sx[d]}; rr_\{d} = rr_\{d} + 1 {\n",
          )
        }
        let src_expr = StringBuilder()
        src_expr.write_string("\{x_off}")
        for d = 0; d < rank; d = d + 1 {
          if is_red[d] {
            src_expr.write_string(" + rr_\{d} * \{in_strides[d]}")
          } else {
            src_expr.write_string(" + rk_\{d} * \{in_strides[d]}")
          }
        }
        body.write_string(
          "      r_sum = r_sum + \{x_buf}[\{src_expr.to_string()}].to_double()\n",
        )
        for _k = 0; _k < red_axes.length(); _k = _k + 1 {
          body.write_string("    }\n")
        }
        let dst_expr = StringBuilder()
        dst_expr.write_string("\{y_off}")
        for k = 0; k < keep_axes.length(); k = k + 1 {
          let d = keep_axes[k]
          dst_expr.write_string(" + rk_\{d} * \{keep_strides[k]}")
        }
        let inv_red = 1.0 / red_count.to_double()
        body.write_string(
          "    \{y_buf}[\{dst_expr.to_string()}] = Float::from_double(r_sum * \{inv_red})\n",
        )
        for _k = 0; _k < keep_axes.length(); _k = _k + 1 {
          body.write_string("  }\n")
        }
      }
      other =>
        raise UnsupportedOp("arena emit unsupported op '\{other.name()}'")
    }
  }

  // 检查是否需要读取 f32 权重辅助函数（包含 RawF32 / PackedMatB / PackedConv）
  for i = 0; i < weight_entries.length(); i = i + 1 {
    match weight_entries[i].kind {
      RawBytes(..) | PackedInt8MatB => need_byte_loader = true
      _ => need_f32_loader = true
    }
  }

  // 合并权重二进制块
  let flat_bytes = FixedArray::make(cur_byte_offset, b'\x00')
  let mut pos = 0
  for c = 0; c < raw_byte_chunks.length(); c = c + 1 {
    let chunk = raw_byte_chunks[c]
    for b = 0; b < chunk.length(); b = b + 1 {
      flat_bytes[pos + b] = chunk[b]
    }
    pos = pos + chunk.length()
  }
  let weights_bin = Bytes::makei(cur_byte_offset, fn(i) { flat_bytes[i] })

  // 组装完整零分配 Arena MoonBit 源码
  let src = StringBuilder()
  let sname = config.struct_name
  let arena_f32_bytes = f32_plan.peak_elems * 4
  let naive_f32_bytes = f32_plan.naive_elems * 4
  src.write_string(
    "// Auto-generated by moon-onnx AOT compiler v2 (`onnx2mbt` Arena Mode).\n",
  )
  src.write_string(
    "// Graph: \{graph.name} | Nodes: \{graph.nodes.length()} | Weights: \{weight_entries.length()} (\{cur_byte_offset} bytes)\n",
  )
  src.write_string(
    "// Static F32 Arena: \{f32_plan.peak_elems} elems (\{arena_f32_bytes} B) vs Naive \{f32_plan.naive_elems} elems (\{naive_f32_bytes} B) | Shared Scratch: \{f32_plan.scratch_elems} elems\n",
  )
  src.write_string(
    "// Runtime heap allocations per `forward_into`: 0 (Zero-Allocation Inference)\n\n",
  )

  if need_f32_loader {
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

  if need_byte_loader {
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

  if need_trans_f32 {
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

  if need_conv_scalar_nchw {
    src.write_string("///|\n")
    src.write_string(
      "/// Wasm-GC 原生 Float 数组友好的 NCHW 卷积矩阵乘（避免 GC 字节数组 `v128_load` 退化）。\n",
    )
    src.write_string(
      "fn conv_nchw_scalar_f32(\n  col : FixedArray[Float],\n  w : FixedArray[Float],\n  bias : FixedArray[Float]?,\n  out : FixedArray[Float],\n  out_off : Int32,\n  spatial : Int32,\n  k : Int32,\n  c_out : Int32,\n  ep : @gemm.Epilogue,\n) -> Unit {\n",
    )
    src.write_string("  for oc = 0; oc < c_out; oc = oc + 1 {\n")
    src.write_string("    let w_row = oc * k\n")
    src.write_string("    let out_ch = out_off + oc * spatial\n")
    src.write_string(
      "    let b0 : Float = match bias { Some(b) => b[oc]\n None => (0.0 : Float) }\n",
    )
    src.write_string("    for s = 0; s < spatial; s = s + 1 {\n")
    src.write_string("      let c_row = s * k\n")
    src.write_string("      let mut acc = b0\n")
    src.write_string("      for kk = 0; kk < k; kk = kk + 1 {\n")
    src.write_string("        acc = acc + col[c_row + kk] * w[w_row + kk]\n")
    src.write_string("      }\n")
    src.write_string(
      "      out[out_ch + s] = @gemm.apply_epilogue_scalar(acc, ep)\n",
    )
    src.write_string("    }\n  }\n}\n\n")
  }

  if need_maxpool {
    src.write_string("///|\n")
    src.write_string(
      "/// 零分配 2D MaxPool（直接在静态 Arena 偏移间读写）。\n",
    )
    src.write_string(
      "fn maxpool2d_f32(\n  x : FixedArray[Float],\n  x_off : Int32,\n  y : FixedArray[Float],\n  y_off : Int32,\n  n : Int32,\n  c : Int32,\n  ih : Int32,\n  iw : Int32,\n  kh : Int32,\n  kw : Int32,\n  sh : Int32,\n  sw : Int32,\n  pt : Int32,\n  pl : Int32,\n  dh : Int32,\n  dw : Int32,\n  oh : Int32,\n  ow : Int32,\n) -> Unit {\n",
    )
    src.write_string("  let neg_inf = Float::from_double(-1.0e38)\n")
    src.write_string("  let nc = n * c\n")
    src.write_string("  let in_plane = ih * iw\n")
    src.write_string("  let out_plane = oh * ow\n")
    src.write_string("  for idx = 0; idx < nc; idx = idx + 1 {\n")
    src.write_string("    let ib = x_off + idx * in_plane\n")
    src.write_string("    let ob = y_off + idx * out_plane\n")
    src.write_string("    for r = 0; r < oh; r = r + 1 {\n")
    src.write_string("      let hs = r * sh - pt\n")
    src.write_string("      for col = 0; col < ow; col = col + 1 {\n")
    src.write_string("        let ws = col * sw - pl\n")
    src.write_string("        let mut best = neg_inf\n")
    src.write_string("        for kr = 0; kr < kh; kr = kr + 1 {\n")
    src.write_string("          let ir = hs + kr * dh\n")
    src.write_string("          if ir >= 0 && ir < ih {\n")
    src.write_string("            let row_off = ib + ir * iw\n")
    src.write_string("            for kc = 0; kc < kw; kc = kc + 1 {\n")
    src.write_string("              let ic = ws + kc * dw\n")
    src.write_string("              if ic >= 0 && ic < iw {\n")
    src.write_string("                let v = x[row_off + ic]\n")
    src.write_string("                if v > best { best = v }\n")
    src.write_string("              }\n            }\n          }\n        }\n")
    src.write_string("        y[ob + r * ow + col] = best\n")
    src.write_string("      }\n    }\n  }\n}\n\n")
  }

  if need_avgpool {
    src.write_string("///|\n")
    src.write_string(
      "/// 零分配 2D AveragePool（直接在静态 Arena 偏移间读写）。\n",
    )
    src.write_string(
      "fn avgpool2d_f32(\n  x : FixedArray[Float],\n  x_off : Int32,\n  y : FixedArray[Float],\n  y_off : Int32,\n  n : Int32,\n  c : Int32,\n  ih : Int32,\n  iw : Int32,\n  kh : Int32,\n  kw : Int32,\n  sh : Int32,\n  sw : Int32,\n  pt : Int32,\n  pl : Int32,\n  oh : Int32,\n  ow : Int32,\n  include_pad : Bool,\n) -> Unit {\n",
    )
    src.write_string("  let nc = n * c\n")
    src.write_string("  let in_plane = ih * iw\n")
    src.write_string("  let out_plane = oh * ow\n")
    src.write_string("  for idx = 0; idx < nc; idx = idx + 1 {\n")
    src.write_string("    let ib = x_off + idx * in_plane\n")
    src.write_string("    let ob = y_off + idx * out_plane\n")
    src.write_string("    for r = 0; r < oh; r = r + 1 {\n")
    src.write_string("      let hs = r * sh - pt\n")
    src.write_string("      for col = 0; col < ow; col = col + 1 {\n")
    src.write_string("        let ws = col * sw - pl\n")
    src.write_string("        let mut sum = 0.0\n")
    src.write_string("        let mut cnt = 0\n")
    src.write_string("        for kr = 0; kr < kh; kr = kr + 1 {\n")
    src.write_string("          let ir = hs + kr\n")
    src.write_string("          if ir >= 0 && ir < ih {\n")
    src.write_string("            let row_off = ib + ir * iw\n")
    src.write_string("            for kc = 0; kc < kw; kc = kc + 1 {\n")
    src.write_string("              let ic = ws + kc\n")
    src.write_string("              if ic >= 0 && ic < iw {\n")
    src.write_string(
      "                sum = sum + x[row_off + ic].to_double()\n",
    )
    src.write_string("                cnt = cnt + 1\n")
    src.write_string("              }\n            }\n          }\n        }\n")
    src.write_string(
      "        let denom = if include_pad { kh * kw } else if cnt > 0 { cnt } else { 1 }\n",
    )
    src.write_string(
      "        y[ob + r * ow + col] = Float::from_double(sum / denom.to_double())\n",
    )
    src.write_string("      }\n    }\n  }\n}\n\n")
  }

  if need_gap {
    src.write_string("///|\n")
    src.write_string(
      "/// 零分配 GlobalAveragePool（直接操作静态 Arena 偏移）。\n",
    )
    src.write_string(
      "fn gap_f32(\n  x : FixedArray[Float],\n  x_off : Int32,\n  y : FixedArray[Float],\n  y_off : Int32,\n  nc : Int32,\n  spatial : Int32,\n) -> Unit {\n",
    )
    src.write_string("  let inv = 1.0 / spatial.to_double()\n")
    src.write_string("  for idx = 0; idx < nc; idx = idx + 1 {\n")
    src.write_string("    let base = x_off + idx * spatial\n")
    src.write_string("    let mut sum = 0.0\n")
    src.write_string("    for s = 0; s < spatial; s = s + 1 {\n")
    src.write_string("      sum = sum + x[base + s].to_double()\n")
    src.write_string("    }\n")
    src.write_string("    y[y_off + idx] = Float::from_double(sum * inv)\n")
    src.write_string("  }\n}\n\n")
  }

  if need_gelu {
    src.write_string("///|\n")
    src.write_string("/// 零分配 GELU（Tanh 近似）。\n")
    src.write_string(
      "fn gelu_f32(x : FixedArray[Float], x_off : Int32, y : FixedArray[Float], y_off : Int32, n : Int32) -> Unit {\n",
    )
    src.write_string("  let c = 0.7978845608028654\n")
    src.write_string("  for i = 0; i < n; i = i + 1 {\n")
    src.write_string("    let d = x[x_off + i].to_double()\n")
    src.write_string(
      "    y[y_off + i] = Float::from_double(0.5 * d * (1.0 + @math.tanh(c * (d + 0.044715 * d * d * d))))\n",
    )
    src.write_string("  }\n}\n\n")
  }

  if need_erf {
    src.write_string("///|\n")
    src.write_string(
      "/// 零分配高斯误差函数 Erf（Abramowitz & Stegun 7.1.26，最大误差 < 1.5e-7）。\n",
    )
    src.write_string(
      "fn erf_f32(x : FixedArray[Float], x_off : Int32, y : FixedArray[Float], y_off : Int32, n : Int32) -> Unit {\n",
    )
    src.write_string("  for i = 0; i < n; i = i + 1 {\n")
    src.write_string("    let v = x[x_off + i].to_double()\n")
    src.write_string("    let sign = if v < 0.0 { -1.0 } else { 1.0 }\n")
    src.write_string("    let ax = if v < 0.0 { -v } else { v }\n")
    src.write_string("    let t = 1.0 / (1.0 + 0.3275911 * ax)\n")
    src.write_string(
      "    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t\n",
    )
    src.write_string(
      "    y[y_off + i] = Float::from_double(sign * (1.0 - poly * @math.exp(-ax * ax)))\n",
    )
    src.write_string("  }\n}\n\n")
  }

  if need_softmax {
    src.write_string("///|\n")
    src.write_string("/// 零分配数值稳定 Softmax。\n")
    src.write_string(
      "fn softmax_f32(\n  x : FixedArray[Float],\n  x_off : Int32,\n  y : FixedArray[Float],\n  y_off : Int32,\n  outer : Int32,\n  axis_dim : Int32,\n  inner : Int32,\n) -> Unit {\n",
    )
    src.write_string("  for o = 0; o < outer; o = o + 1 {\n")
    src.write_string("    let ob = o * axis_dim * inner\n")
    src.write_string("    for j = 0; j < inner; j = j + 1 {\n")
    src.write_string("      let mut mx = x[x_off + ob + j]\n")
    src.write_string("      for k = 1; k < axis_dim; k = k + 1 {\n")
    src.write_string("        let v = x[x_off + ob + k * inner + j]\n")
    src.write_string("        if v > mx { mx = v }\n")
    src.write_string("      }\n")
    src.write_string("      let mut s = 0.0\n")
    src.write_string("      for k = 0; k < axis_dim; k = k + 1 {\n")
    src.write_string("        let idx = ob + k * inner + j\n")
    src.write_string(
      "        let e = @math.exp((x[x_off + idx] - mx).to_double())\n",
    )
    src.write_string("        y[y_off + idx] = Float::from_double(e)\n")
    src.write_string("        s = s + e\n")
    src.write_string("      }\n")
    src.write_string("      let inv = Float::from_double(1.0 / s)\n")
    src.write_string("      for k = 0; k < axis_dim; k = k + 1 {\n")
    src.write_string("        let idx = ob + k * inner + j\n")
    src.write_string("        y[y_off + idx] = y[y_off + idx] * inv\n")
    src.write_string("      }\n    }\n  }\n}\n\n")
  }

  if need_layernorm {
    src.write_string("///|\n")
    src.write_string("/// 零分配双精度累加 LayerNormalization。\n")
    src.write_string(
      "fn layernorm_f32(\n  x : FixedArray[Float],\n  x_off : Int32,\n  scale : FixedArray[Float],\n  bias : FixedArray[Float]?,\n  y : FixedArray[Float],\n  y_off : Int32,\n  outer : Int32,\n  norm_sz : Int32,\n  eps : Float,\n) -> Unit {\n",
    )
    src.write_string("  let inv_n = 1.0 / norm_sz.to_double()\n")
    src.write_string("  let eps_d = eps.to_double()\n")
    src.write_string("  for o = 0; o < outer; o = o + 1 {\n")
    src.write_string("    let b = o * norm_sz\n")
    src.write_string("    let mut sum = 0.0\n")
    src.write_string(
      "    for i = 0; i < norm_sz; i = i + 1 { sum = sum + x[x_off + b + i].to_double() }\n",
    )
    src.write_string("    let mean = sum * inv_n\n")
    src.write_string("    let mut sq = 0.0\n")
    src.write_string(
      "    for i = 0; i < norm_sz; i = i + 1 { let d = x[x_off + b + i].to_double() - mean; sq = sq + d * d }\n",
    )
    src.write_string("    let inv_std = 1.0 / (sq * inv_n + eps_d).sqrt()\n")
    src.write_string("    for i = 0; i < norm_sz; i = i + 1 {\n")
    src.write_string(
      "      let norm = (x[x_off + b + i].to_double() - mean) * inv_std * scale[i].to_double()\n",
    )
    src.write_string(
      "      let bv = match bias { Some(bb) => bb[i].to_double()\n None => 0.0 }\n",
    )
    src.write_string("      y[y_off + b + i] = Float::from_double(norm + bv)\n")
    src.write_string("    }\n  }\n}\n\n")
  }

  if need_reflect {
    src.write_string("///|\n")
    src.write_string("/// 零分配 `reflect` 填充镜像坐标映射。\n")
    src.write_string("fn reflect_idx(idx : Int32, dim : Int32) -> Int32 {\n")
    src.write_string("  if dim <= 1 { return 0 }\n")
    src.write_string("  let mut cur = idx\n")
    src.write_string("  while cur < 0 || cur >= dim {\n")
    src.write_string("    if cur < 0 { cur = -cur }\n")
    src.write_string("    if cur >= dim { cur = 2 * dim - 2 - cur }\n")
    src.write_string("  }\n  cur\n}\n\n")
  }

  // 结构体定义
  src.write_string("///|\n")
  src.write_string(
    "/// 零堆分配静态编译模型运行时：预打包权重常驻，所有中间激活静态复用单一 Arena 缓冲区。\n",
  )
  src.write_string("pub struct \{sname} {\n")
  for i = 0; i < weight_entries.length(); i = i + 1 {
    let w = weight_entries[i]
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
  if weight_entries.length() == 0 {
    src.write_string("  let _ = weights\n")
  }
  for i = 0; i < weight_entries.length(); i = i + 1 {
    let w = weight_entries[i]
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
  for i = 0; i < weight_entries.length(); i = i + 1 {
    src.write_string("    \{weight_entries[i].field_name},\n")
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
    let (final_buf, final_off) = resolve_f32_loc(out_names[0])
    let in_ty = input_arr_type(in_names[0])
    src.write_string("///|\n")
    src.write_string(
      "/// 零堆分配前向推理入口：将输入 `input[\{in_lens[0]}]` 推理结果直接写入调用方缓冲 `output[\{out_lens[0]}]`。\n",
    )
    src.write_string(
      "pub fn \{sname}::forward_into(\n  self : \{sname},\n  input : \{in_ty},\n  output : FixedArray[Float],\n) -> Unit {\n",
    )
    src.write_string(body.to_string())
    src.write_string("  for k_i = 0; k_i < \{out_lens[0]}; k_i = k_i + 1 {\n")
    src.write_string("    output[k_i] = \{final_buf}[\{final_off} + k_i]\n")
    src.write_string("  }\n}\n\n")

    // forward_f32 便捷入口
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
    src.write_string(body.to_string())
    for k = 0; k < out_names.length(); k = k + 1 {
      let (fb, fo) = resolve_f32_loc(out_names[k])
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
  let has_any_import = need_math || need_gemm || need_conv || need_quant
  if has_any_import {
    pkg.write_string("import {\n")
    if need_math {
      pkg.write_string("  \"moonbitlang/core/math\",\n")
    }
    if need_gemm {
      pkg.write_string("  \"matrixlt/ml_kernels/gemm\",\n")
    }
    if need_conv {
      pkg.write_string("  \"matrixlt/ml_kernels/conv\",\n")
    }
    if need_quant {
      pkg.write_string("  \"matrixlt/ml_kernels/quant\",\n")
    }
    pkg.write_string("}\n")
  }

  {
    mbt_source: src.to_string(),
    pkg_config: pkg.to_string(),
    weights_bin,
    weights: weight_entries,
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
