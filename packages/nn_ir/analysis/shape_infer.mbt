///|
/// 推导单个 `IrNode` 的静态输出形状列表（全项目唯一形状推断真相源，支持单输出与多输出算子）。
pub fn infer_node_output_shapes(
  node : @nn_ir.IrNode,
  shapes : Map[String, Array[Int32]],
  consts : Map[String, @tensor.Tensor],
) -> Array[Array[Int32]] raise @nn_ir.IrError {
  let ins = node.inputs
  let outs = node.outputs
  fn req_shape(name : String) -> Array[Int32] raise @nn_ir.IrError {
    match shapes.get(name) {
      Some(s) => s
      None =>
        raise @nn_ir.InvalidGraph(
          "node '\{node.name}' (\{node.op.name()}): unknown input shape for '\{name}'",
        )
    }
  }

  fn read_const_ints(name : String) -> Array[Int32]? {
    if name == "" {
      return None
    }
    match consts.get(name) {
      Some(t) => {
        let buf = t.to_i64_buffer() catch { _ => return None }
        Some(Array::makei(buf.length(), fn(k) { buf[k].to_int() }))
      }
      None => None
    }
  }

  fn read_const_i64s(name : String) -> Array[Int64]? {
    if name == "" {
      return None
    }
    match consts.get(name) {
      Some(t) => {
        let buf = t.to_i64_buffer() catch { _ => return None }
        Some(Array::makei(buf.length(), fn(k) { buf[k] }))
      }
      None => None
    }
  }

  match node.op {
    @nn_ir.Split(axis~, split~, num_outputs~) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let ax = if axis < 0 { axis + rank } else { axis }
      let axis_dim = sx[ax]
      let parts : Array[Int32] = if ins.length() >= 2 && ins[1] != "" {
        match read_const_ints(ins[1]) {
          Some(v) => v
          None =>
            raise @nn_ir.InvalidGraph(
              "Split '\{node.name}': split tensor '\{ins[1]}' must be static initializer",
            )
        }
      } else {
        match split {
          Some(v) => v
          None => {
            let n_out = match num_outputs {
              Some(k) => k
              None => outs.length()
            }
            if n_out <= 0 {
              raise @nn_ir.InvalidGraph(
                "Split '\{node.name}': invalid num_outputs",
              )
            }
            let base = (axis_dim + n_out - 1) / n_out
            let out_s : Array[Int32] = []
            let mut rem = axis_dim
            for k = 0; k < n_out; k = k + 1 {
              let sz = if rem >= base { base } else { rem }
              out_s.push(sz)
              rem = rem - sz
            }
            out_s
          }
        }
      }
      let out_shapes : Array[Array[Int32]] = []
      for k = 0; k < parts.length(); k = k + 1 {
        let s = sx.copy()
        s[ax] = parts[k]
        out_shapes.push(s)
      }
      return out_shapes
    }
    @nn_ir.DynamicQuantizeLinear => return [req_shape(ins[0]).copy(), [], []]
    _ => ()
  }

  let out_shape : Array[Int32] = match node.op {
    @nn_ir.Add | @nn_ir.Sub | @nn_ir.Mul | @nn_ir.Div | @nn_ir.Pow => {
      let sa = req_shape(ins[0])
      let sb = req_shape(ins[1])
      @tensor.broadcast_shape(sa, sb) catch {
        e => raise @nn_ir.InvalidGraph("node '\{node.name}': \{e}")
      }
    }
    @nn_ir.Equal
    | @nn_ir.Less
    | @nn_ir.Greater
    | @nn_ir.LessOrEqual
    | @nn_ir.GreaterOrEqual
    | @nn_ir.And => {
      let sa = req_shape(ins[0])
      let sb = req_shape(ins[1])
      @tensor.broadcast_shape(sa, sb) catch {
        e => raise @nn_ir.InvalidGraph("node '\{node.name}': \{e}")
      }
    }
    @nn_ir.Where => {
      let sc = req_shape(ins[0])
      let sx = req_shape(ins[1])
      let sy = req_shape(ins[2])
      let xy = @tensor.broadcast_shape(sx, sy) catch {
        e => raise @nn_ir.InvalidGraph("node '\{node.name}': \{e}")
      }
      @tensor.broadcast_shape(sc, xy) catch {
        e => raise @nn_ir.InvalidGraph("node '\{node.name}': \{e}")
      }
    }
    @nn_ir.Expand => {
      let sx = req_shape(ins[0])
      match read_const_ints(ins[1]) {
        Some(target) =>
          @tensor.broadcast_shape(sx, target) catch {
            e => raise @nn_ir.InvalidGraph("node '\{node.name}': \{e}")
          }
        None =>
          raise @nn_ir.InvalidGraph(
            "Expand '\{node.name}': shape tensor '\{ins[1]}' must be static",
          )
      }
    }
    @nn_ir.Tile => {
      let sx = req_shape(ins[0])
      match read_const_ints(ins[1]) {
        Some(reps) => Array::makei(sx.length(), fn(d) { sx[d] * reps[d] })
        None =>
          raise @nn_ir.InvalidGraph(
            "Tile '\{node.name}': repeats tensor '\{ins[1]}' must be static",
          )
      }
    }
    @nn_ir.Sqrt
    | @nn_ir.Exp
    | @nn_ir.Neg
    | @nn_ir.Relu
    | @nn_ir.Clip(..)
    | @nn_ir.Sigmoid
    | @nn_ir.Tanh
    | @nn_ir.Gelu
    | @nn_ir.GeluExact
    | @nn_ir.Erf
    | @nn_ir.Not
    | @nn_ir.HardSwish
    | @nn_ir.LeakyRelu(..)
    | @nn_ir.Identity
    | @nn_ir.Softmax(..)
    | @nn_ir.BatchNormalization(..)
    | @nn_ir.LayerNormalization(..)
    | @nn_ir.Cast(..) => req_shape(ins[0]).copy()
    @nn_ir.Constant(value~) => value.shape()
    @nn_ir.ConstantOfShape(..) =>
      match read_const_ints(ins[0]) {
        Some(dims) => dims
        None =>
          raise @nn_ir.InvalidGraph(
            "ConstantOfShape '\{node.name}': shape input '\{ins[0]}' must be static",
          )
      }
    @nn_ir.Shape(start~, end~) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
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
      [len]
    }
    @nn_ir.Flatten(axis~) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let ax = if axis < 0 { axis + rank } else { axis }
      let mut outer = 1
      for d = 0; d < ax; d = d + 1 {
        outer = outer * sx[d]
      }
      let mut inner = 1
      for d = ax; d < rank; d = d + 1 {
        inner = inner * sx[d]
      }
      [outer, inner]
    }
    @nn_ir.Reshape(allowzero~) => {
      let sx = req_shape(ins[0])
      let target = match read_const_ints(ins[1]) {
        Some(v) => v
        None =>
          raise @nn_ir.InvalidGraph(
            "Reshape '\{node.name}': shape tensor '\{ins[1]}' must be static initializer",
          )
      }
      let total = @tensor.num_elements(sx) catch {
        e => raise @nn_ir.InvalidGraph("Reshape '\{node.name}': \{e}")
      }
      let resolved = target.copy()
      let mut neg_idx = -1
      let mut prod = 1
      for d = 0; d < resolved.length(); d = d + 1 {
        if resolved[d] == 0 && !allowzero {
          resolved[d] = sx[d]
        }
        if resolved[d] == -1 {
          neg_idx = d
        } else {
          prod = prod * resolved[d]
        }
      }
      if neg_idx >= 0 {
        resolved[neg_idx] = total / prod
      }
      resolved
    }
    @nn_ir.Transpose(perm~) => {
      let sx = req_shape(ins[0])
      let r = sx.length()
      let p = if perm.length() == 0 {
        Array::makei(r, fn(k) { r - 1 - k })
      } else {
        perm
      }
      Array::makei(r, fn(k) { sx[p[k]] })
    }
    @nn_ir.Squeeze(axes~) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let ax_opt : Array[Int32]? = if ins.length() >= 2 && ins[1] != "" {
        match read_const_ints(ins[1]) {
          Some(v) => Some(v)
          None =>
            raise @nn_ir.InvalidGraph(
              "Squeeze '\{node.name}': axes tensor must be static",
            )
        }
      } else {
        axes
      }
      let drop = Array::make(rank, false)
      match ax_opt {
        Some(a) =>
          for k = 0; k < a.length(); k = k + 1 {
            let norm = if a[k] < 0 { a[k] + rank } else { a[k] }
            drop[norm] = true
          }
        None =>
          for k = 0; k < rank; k = k + 1 {
            if sx[k] == 1 {
              drop[k] = true
            }
          }
      }
      let res : Array[Int32] = []
      for k = 0; k < rank; k = k + 1 {
        if !drop[k] {
          res.push(sx[k])
        }
      }
      res
    }
    @nn_ir.Unsqueeze(axes~) => {
      let sx = req_shape(ins[0])
      let ax_arr : Array[Int32] = if ins.length() >= 2 && ins[1] != "" {
        match read_const_ints(ins[1]) {
          Some(v) => v
          None =>
            raise @nn_ir.InvalidGraph(
              "Unsqueeze '\{node.name}': axes tensor must be static",
            )
        }
      } else {
        match axes {
          Some(a) => a
          None =>
            raise @nn_ir.InvalidGraph("Unsqueeze '\{node.name}': missing axes")
        }
      }
      let out_rank = sx.length() + ax_arr.length()
      let is_new = Array::make(out_rank, false)
      for k = 0; k < ax_arr.length(); k = k + 1 {
        let norm = if ax_arr[k] < 0 { ax_arr[k] + out_rank } else { ax_arr[k] }
        is_new[norm] = true
      }
      let res = Array::make(out_rank, 1)
      let mut src_i = 0
      for k = 0; k < out_rank; k = k + 1 {
        if !is_new[k] {
          res[k] = sx[src_i]
          src_i = src_i + 1
        }
      }
      res
    }
    @nn_ir.MatMul | @nn_ir.MatMulInteger => {
      let sa = req_shape(ins[0])
      let sb = req_shape(ins[1])
      if sa.length() == 2 && sb.length() == 2 {
        [sa[0], sb[1]]
      } else if sa.length() >= 2 && sb.length() == 2 {
        let out = sa.copy()
        out[out.length() - 1] = sb[1]
        out
      } else {
        let out = sa.copy()
        out[out.length() - 1] = sb[sb.length() - 1]
        out
      }
    }
    @nn_ir.Gemm(trans_a~, trans_b~, ..) => {
      let sa = req_shape(ins[0])
      let sb = req_shape(ins[1])
      let m = if trans_a { sa[1] } else { sa[0] }
      let n = if trans_b { sb[0] } else { sb[1] }
      [m, n]
    }
    @nn_ir.Conv(strides~, pads~, dilations~, auto_pad~, ..) => {
      let sx = req_shape(ins[0])
      let sw = req_shape(ins[1])
      if sx.length() == 3 && sw.length() == 3 {
        let s_w = if strides.length() == 1 { strides[0] } else { strides[1] }
        let d_w = if dilations.length() == 1 {
          dilations[0]
        } else {
          dilations[1]
        }
        let p_l = if pads.length() == 2 { pads[0] } else { pads[1] }
        let p_r = if pads.length() == 2 { pads[1] } else { pads[3] }
        let ol = infer_spatial_dim(
          sx[2],
          sw[2],
          s_w,
          d_w,
          p_l,
          p_r,
          auto_pad,
          false,
        )
        [sx[0], sw[0], ol]
      } else {
        let oh = infer_spatial_dim(
          sx[2],
          sw[2],
          strides[0],
          dilations[0],
          pads[0],
          pads[2],
          auto_pad,
          false,
        )
        let ow = infer_spatial_dim(
          sx[3],
          sw[3],
          strides[1],
          dilations[1],
          pads[1],
          pads[3],
          auto_pad,
          false,
        )
        [sx[0], sw[0], oh, ow]
      }
    }
    @nn_ir.MaxPool(
      kernel_shape~,
      strides~,
      pads~,
      dilations~,
      ceil_mode~,
      auto_pad~,
    ) => {
      let sx = req_shape(ins[0])
      let oh = infer_spatial_dim(
        sx[2],
        kernel_shape[0],
        strides[0],
        dilations[0],
        pads[0],
        pads[2],
        auto_pad,
        ceil_mode,
      )
      let ow = infer_spatial_dim(
        sx[3],
        kernel_shape[1],
        strides[1],
        dilations[1],
        pads[1],
        pads[3],
        auto_pad,
        ceil_mode,
      )
      [sx[0], sx[1], oh, ow]
    }
    @nn_ir.AveragePool(
      kernel_shape~,
      strides~,
      pads~,
      ceil_mode~,
      auto_pad~,
      ..
    ) => {
      let sx = req_shape(ins[0])
      let oh = infer_spatial_dim(
        sx[2],
        kernel_shape[0],
        strides[0],
        1,
        pads[0],
        pads[2],
        auto_pad,
        ceil_mode,
      )
      let ow = infer_spatial_dim(
        sx[3],
        kernel_shape[1],
        strides[1],
        1,
        pads[1],
        pads[3],
        auto_pad,
        ceil_mode,
      )
      [sx[0], sx[1], oh, ow]
    }
    @nn_ir.GlobalAveragePool => {
      let sx = req_shape(ins[0])
      let out = Array::make(sx.length(), 1)
      out[0] = sx[0]
      out[1] = sx[1]
      out
    }
    @nn_ir.Concat(axis~) => {
      let s0 = req_shape(ins[0])
      let ax = if axis < 0 { axis + s0.length() } else { axis }
      let out = s0.copy()
      let mut total_ax = s0[ax]
      for k = 1; k < ins.length(); k = k + 1 {
        total_ax = total_ax + req_shape(ins[k])[ax]
      }
      out[ax] = total_ax
      out
    }
    @nn_ir.Gather(axis~) => {
      let sd = req_shape(ins[0])
      let si = req_shape(ins[1])
      let rank = sd.length()
      let ax = if axis < 0 { axis + rank } else { axis }
      let out : Array[Int32] = []
      for d = 0; d < ax; d = d + 1 {
        out.push(sd[d])
      }
      for d = 0; d < si.length(); d = d + 1 {
        out.push(si[d])
      }
      for d = ax + 1; d < rank; d = d + 1 {
        out.push(sd[d])
      }
      out
    }
    @nn_ir.Pad(pads~, ..) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let resolved_pads : Array[Int32] = if ins.length() >= 2 && ins[1] != "" {
        match read_const_ints(ins[1]) {
          Some(v) => v
          None =>
            raise @nn_ir.InvalidGraph(
              "Pad '\{node.name}': pads tensor must be static",
            )
        }
      } else {
        match pads {
          Some(p) => p
          None => raise @nn_ir.InvalidGraph("Pad '\{node.name}': missing pads")
        }
      }
      let out = sx.copy()
      if ins.length() >= 4 && ins[3] != "" {
        match read_const_ints(ins[3]) {
          Some(axes) => {
            let k = axes.length()
            for i = 0; i < k; i = i + 1 {
              let ax = if axes[i] < 0 { axes[i] + rank } else { axes[i] }
              out[ax] = out[ax] + resolved_pads[i] + resolved_pads[k + i]
            }
          }
          None =>
            raise @nn_ir.InvalidGraph(
              "Pad '\{node.name}': axes tensor must be static",
            )
        }
      } else {
        for d = 0; d < rank; d = d + 1 {
          out[d] = out[d] + resolved_pads[d] + resolved_pads[rank + d]
        }
      }
      out
    }
    @nn_ir.Slice(starts~, ends~, axes~, steps~) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let s_arr : Array[Int64] = if ins.length() >= 2 && ins[1] != "" {
        match read_const_i64s(ins[1]) {
          Some(v) => v
          None =>
            raise @nn_ir.InvalidGraph(
              "Slice '\{node.name}': starts tensor must be static",
            )
        }
      } else {
        match starts {
          Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
          None =>
            raise @nn_ir.InvalidGraph("Slice '\{node.name}': missing starts")
        }
      }
      let e_arr : Array[Int64] = if ins.length() >= 3 && ins[2] != "" {
        match read_const_i64s(ins[2]) {
          Some(v) => v
          None =>
            raise @nn_ir.InvalidGraph(
              "Slice '\{node.name}': ends tensor must be static",
            )
        }
      } else {
        match ends {
          Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
          None =>
            raise @nn_ir.InvalidGraph("Slice '\{node.name}': missing ends")
        }
      }
      let a_opt : Array[Int32]? = if ins.length() >= 4 && ins[3] != "" {
        read_const_ints(ins[3])
      } else {
        axes
      }
      let st_opt : Array[Int32]? = if ins.length() >= 5 && ins[4] != "" {
        read_const_ints(ins[4])
      } else {
        steps
      }
      let out = sx.copy()
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
        let dim = sx[ax]
        let s = clamp_slice_bound_static(s_arr[k], dim, step)
        let e = clamp_slice_bound_static(e_arr[k], dim, step)
        out[ax] = if step > 0 {
          if e > s {
            (e - s + step - 1) / step
          } else {
            0
          }
        } else if s > e {
          let ns = -step
          (s - e + ns - 1) / ns
        } else {
          0
        }
      }
      out
    }
    @nn_ir.ReduceMean(axes~, keepdims~, ..)
    | @nn_ir.ReduceSum(axes~, keepdims~, ..) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let drop = Array::make(rank, false)
      let ax_opt : Array[Int32]? = if ins.length() >= 2 && ins[1] != "" {
        read_const_ints(ins[1])
      } else {
        axes
      }
      match ax_opt {
        Some(ax_arr) =>
          for k = 0; k < ax_arr.length(); k = k + 1 {
            let a = if ax_arr[k] < 0 { ax_arr[k] + rank } else { ax_arr[k] }
            drop[a] = true
          }
        None =>
          for k = 0; k < rank; k = k + 1 {
            drop[k] = true
          }
      }
      let out : Array[Int32] = []
      for k = 0; k < rank; k = k + 1 {
        if drop[k] {
          if keepdims {
            out.push(1)
          }
        } else {
          out.push(sx[k])
        }
      }
      out
    }
    @nn_ir.QuantizeLinear(..) | @nn_ir.DequantizeLinear(..) =>
      req_shape(ins[0]).copy()
    @nn_ir.QLinearMatMul => {
      let sa = req_shape(ins[0])
      let sb = req_shape(ins[3])
      if sa.length() == 2 && sb.length() == 2 {
        [sa[0], sb[1]]
      } else {
        let out = sa.copy()
        out[out.length() - 1] = sb[sb.length() - 1]
        out
      }
    }
    @nn_ir.Silu | @nn_ir.RmsNormalization(..) | @nn_ir.RotaryEmbedding(..) =>
      req_shape(ins[0]).copy()
    @nn_ir.SkipRmsNormalization(..) => {
      let sx = req_shape(ins[0])
      let res : Array[Array[Int32]] = [sx.copy()]
      if node.outputs.length() >= 2 {
        res.push([])
      }
      if node.outputs.length() >= 3 {
        res.push([])
      }
      if node.outputs.length() >= 4 {
        res.push(sx.copy())
      }
      return res
    }
    @nn_ir.GroupQueryAttention(num_heads~, kv_num_heads~, ..) => {
      let sq = req_shape(ins[0])
      let b = sq[0]
      let s = sq[1]
      let head_dim = sq[2] / num_heads
      let past_seq = if ins.length() >= 4 && ins[3] != "" {
        match shapes.get(ins[3]) {
          Some(pk) if pk.length() >= 3 => pk[2]
          _ => 0
        }
      } else {
        0
      }
      let present_seq = past_seq + s
      let pk_sh : Array[Int32] = [b, kv_num_heads, present_seq, head_dim]
      return [sq.copy(), pk_sh, pk_sh.copy()]
    }
    @nn_ir.Resize(..) => {
      let sx = req_shape(ins[0])
      let rank = sx.length()
      let out = sx.copy()
      let mut resolved = false
      if ins.length() >= 4 && ins[3] != "" {
        match read_const_ints(ins[3]) {
          Some(sz) if sz.length() == rank => {
            for d = 0; d < rank; d = d + 1 {
              out[d] = sz[d]
            }
            resolved = true
          }
          _ => ()
        }
      }
      if !resolved && ins.length() >= 3 && ins[2] != "" {
        match consts.get(ins[2]) {
          Some(sc_t) if sc_t.numel() == rank => {
            let sc_buf = sc_t.to_f32_buffer() catch {
              e => raise @nn_ir.InvalidGraph("Resize '\{node.name}': \{e}")
            }
            for d = 0; d < rank; d = d + 1 {
              let od = (sx[d].to_double() * sc_buf[d].to_double())
                .floor()
                .to_int()
              out[d] = if od > 0 { od } else { 1 }
            }
            resolved = true
          }
          _ => ()
        }
      }
      if !resolved {
        raise @nn_ir.InvalidGraph(
          "Resize '\{node.name}': static scales or sizes initializer required",
        )
      }
      out
    }
    @nn_ir.MatMulNBits(n~, ..) => {
      let sa = req_shape(ins[0])
      if sa.length() < 2 {
        raise @nn_ir.InvalidGraph(
          "MatMulNBits '\{node.name}' requires rank >= 2",
        )
      }
      let out = sa.copy()
      out[out.length() - 1] = n
      out
    }
    other =>
      raise @nn_ir.UnsupportedOp(
        "shape inference not yet implemented for '\{other.name()}'",
      )
  }
  [out_shape]
}

///|
/// 对已拓扑排序的 `IrGraph` 执行编译期静态形状推导，返回全部张量对应的确定形状表。
pub fn infer_static_shapes(
  graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
) -> Map[String, Array[Int32]] raise @nn_ir.IrError {
  let shapes : Map[String, Array[Int32]] = Map([])
  graph.initializers.each(fn(k, t) { shapes[k] = t.shape() })
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    match input_shapes.get(name) {
      Some(sh) => shapes[name] = sh.copy()
      None =>
        raise @nn_ir.InvalidGraph(
          "missing static input shape for graph input '\{name}'",
        )
    }
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    if node.outputs.length() == 0 {
      continue
    }
    let out_shapes = infer_node_output_shapes(node, shapes, graph.initializers)
    for k = 0; k < out_shapes.length() && k < node.outputs.length(); k = k + 1 {
      if node.outputs[k] != "" {
        shapes[node.outputs[k]] = out_shapes[k]
      }
    }
  }
  shapes
}
