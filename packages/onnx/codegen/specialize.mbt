///|
/// 计算单个 `IrNode` 的静态输出形状列表（支持单输出与多输出算子如 `Split`）。
/// 被 `specialize_graph`（编译期常量/形状折叠）与 `infer_static_shapes` 共同复用，避免重复实现。
fn infer_node_output_shapes(
  node : @nn_ir.IrNode,
  shapes : Map[String, Array[Int32]],
  consts : Map[String, @tensor.Tensor],
) -> Array[Array[Int32]] raise CodegenError {
  let ins = node.inputs
  let outs = node.outputs
  fn req_shape(name : String) -> Array[Int32] raise CodegenError {
    match shapes.get(name) {
      Some(s) => s
      None =>
        raise ShapeError(
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
            raise ShapeError(
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
              raise ShapeError("Split '\{node.name}': invalid num_outputs")
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
    _ => ()
  }

  let out_shape : Array[Int32] = match node.op {
    @nn_ir.Add | @nn_ir.Sub | @nn_ir.Mul | @nn_ir.Div | @nn_ir.Pow => {
      let sa = req_shape(ins[0])
      let sb = req_shape(ins[1])
      @tensor.broadcast_shape(sa, sb) catch {
        e => raise ShapeError("node '\{node.name}': \{e}")
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
        e => raise ShapeError("node '\{node.name}': \{e}")
      }
    }
    @nn_ir.Where => {
      let sc = req_shape(ins[0])
      let sx = req_shape(ins[1])
      let sy = req_shape(ins[2])
      let xy = @tensor.broadcast_shape(sx, sy) catch {
        e => raise ShapeError("node '\{node.name}': \{e}")
      }
      @tensor.broadcast_shape(sc, xy) catch {
        e => raise ShapeError("node '\{node.name}': \{e}")
      }
    }
    @nn_ir.Expand => {
      let sx = req_shape(ins[0])
      match read_const_ints(ins[1]) {
        Some(target) =>
          @tensor.broadcast_shape(sx, target) catch {
            e => raise ShapeError("node '\{node.name}': \{e}")
          }
        None =>
          raise ShapeError(
            "Expand '\{node.name}': shape tensor '\{ins[1]}' must be static",
          )
      }
    }
    @nn_ir.Tile => {
      let sx = req_shape(ins[0])
      match read_const_ints(ins[1]) {
        Some(reps) => Array::makei(sx.length(), fn(d) { sx[d] * reps[d] })
        None =>
          raise ShapeError(
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
          raise ShapeError(
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
          raise ShapeError(
            "Reshape '\{node.name}': shape tensor '\{ins[1]}' must be static initializer",
          )
      }
      let total = @tensor.num_elements(sx) catch {
        e => raise ShapeError("Reshape '\{node.name}': \{e}")
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
            raise ShapeError(
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
            raise ShapeError(
              "Unsqueeze '\{node.name}': axes tensor must be static",
            )
        }
      } else {
        match axes {
          Some(a) => a
          None => raise ShapeError("Unsqueeze '\{node.name}': missing axes")
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
    @nn_ir.MatMul => {
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
            raise ShapeError("Pad '\{node.name}': pads tensor must be static")
        }
      } else {
        match pads {
          Some(p) => p
          None => raise ShapeError("Pad '\{node.name}': missing pads")
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
            raise ShapeError("Pad '\{node.name}': axes tensor must be static")
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
            raise ShapeError(
              "Slice '\{node.name}': starts tensor must be static",
            )
        }
      } else {
        match starts {
          Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
          None => raise ShapeError("Slice '\{node.name}': missing starts")
        }
      }
      let e_arr : Array[Int64] = if ins.length() >= 3 && ins[2] != "" {
        match read_const_i64s(ins[2]) {
          Some(v) => v
          None =>
            raise ShapeError("Slice '\{node.name}': ends tensor must be static")
        }
      } else {
        match ends {
          Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
          None => raise ShapeError("Slice '\{node.name}': missing ends")
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
    other =>
      raise UnsupportedOp(
        "shape inference not yet implemented for '\{other.name()}' in AOT compiler",
      )
  }
  [out_shape]
}

///|
/// 与 `ops/tensor_ops.mbt` 一致的静态切片端点规范化函数（供编译期形状推导与 Arena Slice 代码生成共享）。
fn clamp_slice_bound_static(v : Int64, dim : Int32, step : Int32) -> Int32 {
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
/// 尝试在编译期对全常量输入的纯元算子执行常量折叠求值；若不可折叠返回 `None`。
fn try_eval_const_node(
  node : @nn_ir.IrNode,
  consts : Map[String, @tensor.Tensor],
) -> @tensor.Tensor? {
  let ins = node.inputs
  if node.outputs.length() != 1 || node.outputs[0] == "" {
    return None
  }
  // 检查所有非空输入是否均已是编译期常量
  for j = 0; j < ins.length(); j = j + 1 {
    if ins[j] != "" && !consts.contains(ins[j]) {
      return None
    }
  }
  try {
    match node.op {
      @nn_ir.Constant(value~) => Some(value)
      @nn_ir.Identity => consts.get(ins[0])
      @nn_ir.Equal =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(a), Some(b)) => Some(@ops.equal_op(a, b))
          _ => None
        }
      @nn_ir.ConstantOfShape(value~) =>
        match consts.get(ins[0]) {
          Some(sh) => Some(@ops.constant_of_shape_op(sh, value))
          None => None
        }
      @nn_ir.Cast(to~) =>
        match consts.get(ins[0]) {
          Some(x) => Some(@ops.cast_op(x, to))
          None => None
        }
      @nn_ir.Gather(axis~) =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(d), Some(idx)) => Some(@ops.gather_op(d, idx, axis))
          _ => None
        }
      @nn_ir.Unsqueeze(axes~) =>
        match consts.get(ins[0]) {
          Some(x) => {
            let ax_t = if ins.length() >= 2 && ins[1] != "" {
              consts.get(ins[1])
            } else {
              None
            }
            Some(@ops.unsqueeze_op(x, axes, ax_t))
          }
          None => None
        }
      @nn_ir.Squeeze(axes~) =>
        match consts.get(ins[0]) {
          Some(x) => {
            let ax_t = if ins.length() >= 2 && ins[1] != "" {
              consts.get(ins[1])
            } else {
              None
            }
            Some(@ops.squeeze_op(x, axes, ax_t))
          }
          None => None
        }
      @nn_ir.Concat(axis~) => {
        let ts : Array[@tensor.Tensor] = []
        for k = 0; k < ins.length(); k = k + 1 {
          match consts.get(ins[k]) {
            Some(t) => ts.push(t)
            None => return None
          }
        }
        Some(@ops.concat_op(ts, axis))
      }
      @nn_ir.Reshape(allowzero~) =>
        match (consts.get(ins[0]), consts.get(ins[1])) {
          (Some(x), Some(sh)) => Some(@ops.reshape_op(x, sh, allowzero~))
          _ => None
        }
      @nn_ir.Transpose(perm~) =>
        match consts.get(ins[0]) {
          Some(x) => Some(@ops.transpose_op(x, perm))
          None => None
        }
      @nn_ir.Slice(starts~, ends~, axes~, steps~) =>
        match consts.get(ins[0]) {
          Some(x) => {
            let s_arr : Array[Int64] = if ins.length() >= 2 && ins[1] != "" {
              let b = consts.get(ins[1]).unwrap().to_i64_buffer()
              Array::makei(b.length(), fn(k) { b[k] })
            } else {
              match starts {
                Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
                None => return None
              }
            }
            let e_arr : Array[Int64] = if ins.length() >= 3 && ins[2] != "" {
              let b = consts.get(ins[2]).unwrap().to_i64_buffer()
              Array::makei(b.length(), fn(k) { b[k] })
            } else {
              match ends {
                Some(v) => Array::makei(v.length(), fn(k) { v[k].to_int64() })
                None => return None
              }
            }
            let a_opt : Array[Int32]? = if ins.length() >= 4 && ins[3] != "" {
              let b = consts.get(ins[3]).unwrap().to_i64_buffer()
              Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
            } else {
              axes
            }
            let st_opt : Array[Int32]? = if ins.length() >= 5 && ins[4] != "" {
              let b = consts.get(ins[4]).unwrap().to_i64_buffer()
              Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
            } else {
              steps
            }
            Some(@ops.slice_op(x, s_arr, e_arr, a_opt, st_opt))
          }
          None => None
        }
      _ => None
    }
  } catch {
    _ => None
  }
}

///|
/// 从 `Bool` 标量常量张量中读取布尔值（用于 `If` 编译期分支特化）。
fn try_read_const_bool(t : @tensor.Tensor) -> Bool? {
  let c = t.contiguous() catch { _ => return None }
  if c.numel() != 1 {
    return None
  }
  match c.data {
    @tensor.Bool(buf) => Some(buf[c.offset])
    _ => None
  }
}

///|
/// 编译期图特化与静态形状/元数据常量折叠管线（AOT 专用）：
/// 1. 冻结 `frozen_inputs`（如 Silero VAD 的 `sr = 16000`），将其转为静态常量；
/// 2. 对条件为编译期常量的 `If` 节点执行分支内联，彻底消除控制流与未使用分支权重；
/// 3. 在已知 `input_shapes` 基础上折叠 `Shape` 及全常量元子图（如 MobileNetV2 的 `Shape->Gather->Unsqueeze->Concat` 与 Silero VAD 的 `ConstantOfShape->...->Cast` 填充参数计算）；
/// 4. 将无操作同类型 `Cast`（如 `f32 -> f32`）重写为 `Identity` 以便后续优化消除。
pub fn specialize_graph(
  graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
  frozen_inputs : Map[String, @tensor.Tensor],
  input_dtypes~ : Map[String, @tensor.DType] = Map([]),
) -> @nn_ir.IrGraph raise CodegenError {
  // 1. 克隆初始权重并吸收冻结输入
  let initializers : Map[String, @tensor.Tensor] = Map([])
  graph.initializers.each(fn(k, v) { initializers[k] = v })
  frozen_inputs.each(fn(k, v) { initializers[k] = v })

  let remaining_inputs : Array[String] = []
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    if !frozen_inputs.contains(name) {
      remaining_inputs.push(name)
    }
  }

  let is_graph_out : Map[String, Bool] = Map([])
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    is_graph_out[graph.outputs[i]] = true
  }

  // 2. 迭代执行静态 `If` 分支内联 + `Equal`/常量折叠（支持嵌套 `If`）
  let mut cur_nodes = graph.nodes.copy()
  for _iter = 0; _iter < 8; _iter = _iter + 1 {
    let mut changed = false
    let next_nodes : Array[@nn_ir.IrNode] = []
    for i = 0; i < cur_nodes.length(); i = i + 1 {
      let node = cur_nodes[i]
      match node.op {
        @nn_ir.If(then_branch~, else_branch~) =>
          if node.inputs.length() >= 1 {
            match initializers.get(node.inputs[0]) {
              Some(cond_t) =>
                match try_read_const_bool(cond_t) {
                  Some(cond) => {
                    let chosen = if cond { then_branch } else { else_branch }
                    chosen.initializers.each(fn(k, v) { initializers[k] = v })
                    for k = 0; k < chosen.nodes.length(); k = k + 1 {
                      next_nodes.push(chosen.nodes[k])
                    }
                    // 用 Identity 桥接子图输出与外层 If 输出名，交由后续 optimize_graph 零成本消除
                    for k = 0
                        k < node.outputs.length() && k < chosen.outputs.length()
                        k = k + 1 {
                      if node.outputs[k] != "" &&
                        chosen.outputs[k] != node.outputs[k] {
                        next_nodes.push({
                          name: "\{node.name}_inline_out_\{k}",
                          op: @nn_ir.Identity,
                          inputs: [chosen.outputs[k]],
                          outputs: [node.outputs[k]],
                        })
                      }
                    }
                    changed = true
                    continue
                  }
                  None => ()
                }
              None => ()
            }
          }
        _ =>
          // 尝试折叠仅依赖静态常量的前置条件节点（例如 `Constant` 与 `Equal`）
          match try_eval_const_node(node, initializers) {
            Some(val) if !is_graph_out.contains(node.outputs[0]) => {
              initializers[node.outputs[0]] = val
              changed = true
              continue
            }
            _ => ()
          }
      }
      next_nodes.push(node)
    }
    cur_nodes = next_nodes
    if !changed {
      break
    }
  }

  // 3. 在已知静态输入形状的前提下，单遍前向传播形状 + 折叠 `Shape` / 常量元子图 + 消除冗余 `Cast`
  let shapes : Map[String, Array[Int32]] = Map([])
  let dtypes : Map[String, @tensor.DType] = Map([])
  initializers.each(fn(k, t) {
    shapes[k] = t.shape()
    dtypes[k] = t.dtype()
  })
  for i = 0; i < remaining_inputs.length(); i = i + 1 {
    let name = remaining_inputs[i]
    match input_shapes.get(name) {
      Some(s) => shapes[name] = s.copy()
      None => ()
    }
    dtypes[name] = match input_dtypes.get(name) {
      Some(dt) => dt
      None => @tensor.Float32
    }
  }

  let folded_nodes : Array[@nn_ir.IrNode] = []
  for i = 0; i < cur_nodes.length(); i = i + 1 {
    let node = cur_nodes[i]
    // 3a. 静态 `Shape` 算子：直接由编译期已知形状求值为 `Int64` 常量张量
    match node.op {
      @nn_ir.Shape(start~, end~) =>
        if node.inputs.length() >= 1 && node.outputs.length() == 1 {
          match shapes.get(node.inputs[0]) {
            Some(dims) => {
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
              let buf = FixedArray::makei(len, fn(k) { dims[s + k].to_int64() })
              let val = @tensor.Tensor::from_i64([len], buf) catch {
                err => raise TensorErr("\{err}")
              }
              let out_name = node.outputs[0]
              shapes[out_name] = [len]
              dtypes[out_name] = @tensor.Int64
              if !is_graph_out.contains(out_name) {
                initializers[out_name] = val
                continue
              } else {
                folded_nodes.push({
                  name: node.name,
                  op: @nn_ir.Constant(value=val),
                  inputs: [],
                  outputs: node.outputs.copy(),
                })
                continue
              }
            }
            None => ()
          }
        }
      _ => ()
    }

    // 3b. 全常量输入纯元算子求值折叠
    match try_eval_const_node(node, initializers) {
      Some(val) => {
        let out_name = node.outputs[0]
        shapes[out_name] = val.shape()
        dtypes[out_name] = val.dtype()
        if !is_graph_out.contains(out_name) {
          initializers[out_name] = val
          continue
        }
      }
      None => ()
    }

    // 3c. 同类型无操作 `Cast` 重写为 `Identity`，供后续优化直接消除
    let eff_node : @nn_ir.IrNode = match node.op {
      @nn_ir.Cast(to~) if node.inputs.length() >= 1 =>
        match dtypes.get(node.inputs[0]) {
          Some(src_dt) if src_dt == to =>
            {
              name: node.name,
              op: @nn_ir.Identity,
              inputs: node.inputs.copy(),
              outputs: node.outputs.copy(),
            }
          _ => node
        }
      _ => node
    }

    // 记录保留节点的静态输出形状与数据类型，供下游 `Shape`/`Cast` 继续推导
    let out_shapes = infer_node_output_shapes(eff_node, shapes, initializers) catch {
      _ => []
    }
    for k = 0
        k < out_shapes.length() && k < eff_node.outputs.length()
        k = k + 1 {
      if eff_node.outputs[k] != "" {
        shapes[eff_node.outputs[k]] = out_shapes[k]
      }
    }
    let out_dt : @tensor.DType = match eff_node.op {
      @nn_ir.Cast(to~) => to
      @nn_ir.Shape(..) => @tensor.Int64
      @nn_ir.Equal => @tensor.Bool
      @nn_ir.Constant(value~) => value.dtype()
      @nn_ir.ConstantOfShape(value~) => value.dtype()
      @nn_ir.DequantizeLinear(..) => @tensor.Float32
      _ =>
        if eff_node.inputs.length() > 0 {
          match dtypes.get(eff_node.inputs[0]) {
            Some(dt) => dt
            None => @tensor.Float32
          }
        } else {
          @tensor.Float32
        }
    }
    for k = 0; k < eff_node.outputs.length(); k = k + 1 {
      if eff_node.outputs[k] != "" {
        dtypes[eff_node.outputs[k]] = out_dt
      }
    }
    folded_nodes.push(eff_node)
  }

  let specialized : @nn_ir.IrGraph = {
    name: graph.name,
    inputs: remaining_inputs,
    outputs: graph.outputs.copy(),
    initializers,
    nodes: folded_nodes,
  }
  let (opt_g, _) = @opt.optimize_graph(specialized)
  opt_g
}
