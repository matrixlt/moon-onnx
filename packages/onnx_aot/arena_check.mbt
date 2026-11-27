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
      @nn_ir.Split(..)
      | @nn_ir.DynamicQuantizeLinear
      | @nn_ir.SkipRmsNormalization(..) => true
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
      | @nn_ir.GeluExact
      | @nn_ir.Silu
      | @nn_ir.Erf
      | @nn_ir.HardSwish
      | @nn_ir.LeakyRelu(..)
      | @nn_ir.Sqrt
      | @nn_ir.Exp
      | @nn_ir.Neg
      | @nn_ir.GlobalAveragePool
      | @nn_ir.Softmax(..)
      | @nn_ir.LayerNormalization(..)
      | @nn_ir.RmsNormalization(..) => true
      @nn_ir.SkipRmsNormalization(..) =>
        ins.length() >= 3 &&
        graph.initializers.contains(ins[2]) &&
        (
          ins.length() < 4 ||
          ins[3] == "" ||
          graph.initializers.contains(ins[3])
        )
      @nn_ir.RotaryEmbedding(..) =>
        ins.length() >= 4 &&
        input_or_alias_i64.contains(ins[1]) &&
        graph.initializers.contains(ins[2]) &&
        graph.initializers.contains(ins[3])
      @nn_ir.MatMulNBits(bits~, ..) =>
        bits == 4 &&
        ins.length() >= 3 &&
        graph.initializers.contains(ins[1]) &&
        graph.initializers.contains(ins[2]) &&
        (
          ins.length() < 4 ||
          ins[3] == "" ||
          graph.initializers.contains(ins[3])
        ) &&
        (
          ins.length() < 6 ||
          ins[5] == "" ||
          graph.initializers.contains(ins[5])
        )
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
      @nn_ir.Resize(mode~, ..) => {
        let mode_ok = match mode {
          "" | "nearest" | "linear" | "bilinear" => true
          _ => false
        }
        match (shapes.get(ins[0]), shapes.get(node.outputs[0])) {
          (Some(sx), Some(sy)) =>
            mode_ok &&
            sx.length() == 4 &&
            sy.length() == 4 &&
            sx[0] == sy[0] &&
            sx[1] == sy[1]
          _ => false
        }
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
