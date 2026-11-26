///|
/// Arena 模式张量变换、索引、裁剪与归约算子发射器（Pow / Pad / Slice / Transpose / Cast / Gather / Split / Concat / ReduceMean）。
fn ArenaEmitter::emit_tensor_op(
  self : ArenaEmitter,
  node : @nn_ir.IrNode,
  out_n : String,
  out_sh : Array[Int32],
  out_numel : Int32,
) -> Unit raise CodegenError {
  let ins = node.inputs
  let _ = out_numel
  match node.op {
    @nn_ir.Pow => {
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.shapes.get(ins[0]).unwrap()
      let sb = self.shapes.get(ins[1]).unwrap()
      let mut emitted_sq = false
      if sb.length() == 0 || (sb.length() == 1 && sb[0] == 1) {
        match self.graph.initializers.get(ins[1]) {
          Some(bt) => {
            let exp_v = bt.get_f32_flat(0) catch {
              e => raise TensorErr("\{e}")
            }
            if exp_v == (2.0 : Float) {
              // 快速路径：x^2 直接展开为乘法（如 STFT 幅值计算），避免调用 `@math.pow`
              self.body.write_string(
                "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{a_buf}[\{a_off} + k_i]; \{y_buf}[\{y_off} + k_i] = v * v }\n",
              )
              emitted_sq = true
            } else {
              self.need_math = true
              let exp_d = exp_v.to_double()
              self.body.write_string(
                "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.pow(\{a_buf}[\{a_off} + k_i].to_double(), \{exp_d})) }\n",
              )
              emitted_sq = true
            }
          }
          None => ()
        }
      }
      if !emitted_sq {
        self.need_math = true
        let (b_buf, b_off) = self.resolve_f32_loc(ins[1])
        if sa == sb {
          self.body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.pow(\{a_buf}[\{a_off} + k_i].to_double(), \{b_buf}[\{b_off} + k_i].to_double())) }\n",
          )
        } else {
          self.body.write_string(
            "  let exp_s = \{b_buf}[\{b_off}].to_double()\n",
          )
          self.body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.pow(\{a_buf}[\{a_off} + k_i].to_double(), exp_s)) }\n",
          )
        }
      }
    }
    @nn_ir.Pad(mode~, pads~, constant_value~) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
      let rank = sx.length()
      let raw_pads : Array[Int32] = if ins.length() >= 2 && ins[1] != "" {
        let b = self.graph.initializers.get(ins[1]).unwrap().to_i64_buffer() catch {
            e => raise TensorErr("\{e}")
          }
        Array::makei(b.length(), fn(k) { b[k].to_int() })
      } else {
        pads.unwrap()
      }
      let pb = Array::make(rank, 0)
      if ins.length() >= 4 && ins[3] != "" {
        let ab = self.graph.initializers.get(ins[3]).unwrap().to_i64_buffer() catch {
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
        self.graph.initializers.get(ins[2]).unwrap().get_f32_flat(0) catch {
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
          self.need_nn = true
          1
        }
        "edge" => 2
        other => raise UnsupportedOp("arena Pad unsupported mode '\{other}'")
      }
      let in_strides = @tensor.compute_strides(sx)
      let out_strides = @tensor.compute_strides(out_sh)
      // 展开 1D~4D 静态循环实现严格 0 堆分配的 Pad
      for d = 0; d < rank; d = d + 1 {
        self.body.write_string(
          "  for p_\{d} = 0; p_\{d} < \{out_sh[d]}; p_\{d} = p_\{d} + 1 {\n",
        )
      }
      if mode_id == 0 {
        self.body.write_string("    let mut oob = false\n")
      }
      self.body.write_string("    let mut s_idx = \{x_off}\n")
      for d = 0; d < rank; d = d + 1 {
        self.body.write_string("    let ic_\{d} = p_\{d} - \{pb[d]}\n")
        if mode_id == 0 {
          self.body.write_string(
            "    if ic_\{d} < 0 || ic_\{d} >= \{sx[d]} { oob = true } else { s_idx = s_idx + ic_\{d} * \{in_strides[d]} }\n",
          )
        } else if mode_id == 1 {
          self.body.write_string(
            "    let rc_\{d} = if ic_\{d} < 0 || ic_\{d} >= \{sx[d]} { @nn.reflect_idx(ic_\{d}, \{sx[d]}) } else { ic_\{d} }\n",
          )
          self.body.write_string(
            "    s_idx = s_idx + rc_\{d} * \{in_strides[d]}\n",
          )
        } else {
          self.body.write_string(
            "    let ec_\{d} = if ic_\{d} < 0 { 0 } else if ic_\{d} >= \{sx[d]} { \{sx[d] - 1} } else { ic_\{d} }\n",
          )
          self.body.write_string(
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
        self.body.write_string(
          "    \{y_buf}[\{dst_expr.to_string()}] = if oob { \{fmt_float_literal(cval)} } else { \{x_buf}[s_idx] }\n",
        )
      } else {
        self.body.write_string(
          "    \{y_buf}[\{dst_expr.to_string()}] = \{x_buf}[s_idx]\n",
        )
      }
      for _d = 0; _d < rank; _d = _d + 1 {
        self.body.write_string("  }\n")
      }
    }
    @nn_ir.Slice(starts~, ends~, axes~, steps~) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
      let rank = sx.length()
      let s_arr : Array[Int64] = if ins.length() >= 2 && ins[1] != "" {
        let b = self.graph.initializers.get(ins[1]).unwrap().to_i64_buffer() catch {
            e => raise TensorErr("\{e}")
          }
        Array::makei(b.length(), fn(k) { b[k] })
      } else {
        let v = starts.unwrap()
        Array::makei(v.length(), fn(k) { v[k].to_int64() })
      }
      let a_opt : Array[Int32]? = if ins.length() >= 4 && ins[3] != "" {
        let b = self.graph.initializers.get(ins[3]).unwrap().to_i64_buffer() catch {
            e => raise TensorErr("\{e}")
          }
        Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
      } else {
        axes
      }
      let st_opt : Array[Int32]? = if ins.length() >= 5 && ins[4] != "" {
        let b = self.graph.initializers.get(ins[4]).unwrap().to_i64_buffer() catch {
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
        ax_start[ax] = @analysis.clamp_slice_bound_static(
          s_arr[k],
          sx[ax],
          step,
        )
        ax_step[ax] = step
      }
      let in_strides = @tensor.compute_strides(sx)
      let out_strides = @tensor.compute_strides(out_sh)
      for d = 0; d < rank; d = d + 1 {
        self.body.write_string(
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
      self.body.write_string(
        "    \{y_buf}[\{dst_expr.to_string()}] = \{x_buf}[\{src_expr.to_string()}]\n",
      )
      for _d = 0; _d < rank; _d = _d + 1 {
        self.body.write_string("  }\n")
      }
    }
    @nn_ir.Transpose(perm~) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
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
        self.body.write_string(
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
      self.body.write_string(
        "    \{y_buf}[\{dst_expr.to_string()}] = \{x_buf}[\{src_expr.to_string()}]\n",
      )
      for _d = 0; _d < rank; _d = _d + 1 {
        self.body.write_string("  }\n")
      }
    }
    @nn_ir.Cast(to~) => {
      let _ = to
      if self.skipped_cast.contains(out_n) {
        self.body.write_string(
          "  // fused directly into MatMulInteger output\n",
        )
      } else {
        let in_param = self.input_param_of.get(ins[0]).unwrap()
        let (y_buf, y_off) = self.resolve_f32_loc(out_n)
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_int(\{in_param}[k_i].to_int()) }\n",
        )
      }
    }
    @nn_ir.Gather(axis~) => {
      let is_byte_gather = match self.dtypes.get(ins[0]) {
        Some(@tensor.UInt8) | Some(@tensor.Int8) => true
        _ => false
      }
      let (x_buf, x_off) = if is_byte_gather {
        self.resolve_byte_loc(ins[0])
      } else {
        self.resolve_f32_loc(ins[0])
      }
      let (y_buf, y_off) = if is_byte_gather {
        self.resolve_byte_loc(out_n)
      } else {
        self.resolve_f32_loc(out_n)
      }
      let sd = self.shapes.get(ins[0]).unwrap()
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
      match self.graph.initializers.get(ins[1]) {
        Some(idx_t) => {
          let idx_buf = idx_t.to_i64_buffer() catch {
            e => raise TensorErr("\{e}")
          }
          let idx_len = idx_buf.length()
          for k = 0; k < idx_len; k = k + 1 {
            let raw = idx_buf[k].to_int()
            let norm = if raw < 0 { raw + axis_dim } else { raw }
            if outer == 1 {
              self.body.write_string(
                "  for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[\{y_off + k * inner} + g_j] = \{x_buf}[\{x_off + norm * inner} + g_j] }\n",
              )
            } else {
              self.body.write_string(
                "  for g_o = 0; g_o < \{outer}; g_o = g_o + 1 { for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[\{y_off} + (g_o * \{idx_len} + \{k}) * \{inner} + g_j] = \{x_buf}[\{x_off} + (g_o * \{axis_dim} + \{norm}) * \{inner} + g_j] } }\n",
              )
            }
          }
        }
        None => {
          // 动态 Int64 输入索引查表（如 Transformer Embedding 表查表）
          let idx_param = self.input_param_of.get(ins[1]).unwrap()
          let idx_sh = self.shapes.get(ins[1]).unwrap()
          let idx_len = @tensor.num_elements(idx_sh) catch {
            e => raise ShapeError("\{e}")
          }
          if outer == 1 {
            self.body.write_string(
              "  for g_k = 0; g_k < \{idx_len}; g_k = g_k + 1 { let raw_idx = \{idx_param}[g_k].to_int(); let norm_idx = if raw_idx < 0 { raw_idx + \{axis_dim} } else { raw_idx }; let src_b = \{x_off} + norm_idx * \{inner}; let dst_b = \{y_off} + g_k * \{inner}; for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[dst_b + g_j] = \{x_buf}[src_b + g_j] } }\n",
            )
          } else {
            self.body.write_string(
              "  for g_o = 0; g_o < \{outer}; g_o = g_o + 1 { for g_k = 0; g_k < \{idx_len}; g_k = g_k + 1 { let raw_idx = \{idx_param}[g_k].to_int(); let norm_idx = if raw_idx < 0 { raw_idx + \{axis_dim} } else { raw_idx }; let src_b = \{x_off} + (g_o * \{axis_dim} + norm_idx) * \{inner}; let dst_b = \{y_off} + (g_o * \{idx_len} + g_k) * \{inner}; for g_j = 0; g_j < \{inner}; g_j = g_j + 1 { \{y_buf}[dst_b + g_j] = \{x_buf}[src_b + g_j] } } }\n",
            )
          }
        }
      }
    }
    @nn_ir.Split(axis~, ..) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let sx = self.shapes.get(ins[0]).unwrap()
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
        let psh = self.shapes.get(pn).unwrap()
        let part_dim = psh[ax]
        let part_inner = part_dim * inner
        let (p_buf, p_off) = self.resolve_f32_loc(pn)
        if outer == 1 {
          self.body.write_string(
            "  for sp_k = 0; sp_k < \{part_inner}; sp_k = sp_k + 1 { \{p_buf}[\{p_off} + sp_k] = \{x_buf}[\{x_off + axis_off * inner} + sp_k] }\n",
          )
        } else {
          self.body.write_string(
            "  for sp_o = 0; sp_o < \{outer}; sp_o = sp_o + 1 { for sp_k = 0; sp_k < \{part_inner}; sp_k = sp_k + 1 { \{p_buf}[\{p_off} + sp_o * \{part_inner} + sp_k] = \{x_buf}[\{x_off} + (sp_o * \{axis_dim} + \{axis_off}) * \{inner} + sp_k] } }\n",
          )
        }
        axis_off = axis_off + part_dim
      }
    }
    @nn_ir.Concat(axis~) => {
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let s0 = self.shapes.get(ins[0]).unwrap()
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
        let (s_buf, s_off) = self.resolve_f32_loc(ins[t_idx])
        let ts = self.shapes.get(ins[t_idx]).unwrap()
        let chunk = ts[ax] * inner
        if outer == 1 {
          self.body.write_string(
            "  for cc_k = 0; cc_k < \{chunk}; cc_k = cc_k + 1 { \{y_buf}[\{y_off + dst_axis_off * inner} + cc_k] = \{s_buf}[\{s_off} + cc_k] }\n",
          )
        } else {
          self.body.write_string(
            "  for cc_o = 0; cc_o < \{outer}; cc_o = cc_o + 1 { for cc_k = 0; cc_k < \{chunk}; cc_k = cc_k + 1 { \{y_buf}[\{y_off} + (cc_o * \{total_axis} + \{dst_axis_off}) * \{inner} + cc_k] = \{s_buf}[\{s_off} + cc_o * \{chunk} + cc_k] } }\n",
          )
        }
        dst_axis_off = dst_axis_off + ts[ax]
      }
    }
    @nn_ir.ReduceMean(axes~, ..) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
      let rank = sx.length()
      let ax_opt : Array[Int32]? = if ins.length() >= 2 && ins[1] != "" {
        let b = self.graph.initializers.get(ins[1]).unwrap().to_i64_buffer() catch {
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
            let norm = if ax_arr[k] < 0 { ax_arr[k] + rank } else { ax_arr[k] }
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
        self.body.write_string(
          "  for rk_\{d} = 0; rk_\{d} < \{sx[d]}; rk_\{d} = rk_\{d} + 1 {\n",
        )
      }
      self.body.write_string("    let mut r_sum = 0.0\n")
      for k = 0; k < red_axes.length(); k = k + 1 {
        let d = red_axes[k]
        self.body.write_string(
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
      self.body.write_string(
        "      r_sum = r_sum + \{x_buf}[\{src_expr.to_string()}].to_double()\n",
      )
      for _k = 0; _k < red_axes.length(); _k = _k + 1 {
        self.body.write_string("    }\n")
      }
      let dst_expr = StringBuilder()
      dst_expr.write_string("\{y_off}")
      for k = 0; k < keep_axes.length(); k = k + 1 {
        let d = keep_axes[k]
        dst_expr.write_string(" + rk_\{d} * \{keep_strides[k]}")
      }
      let inv_red = 1.0 / red_count.to_double()
      self.body.write_string(
        "    \{y_buf}[\{dst_expr.to_string()}] = Float::from_double(r_sum * \{inv_red})\n",
      )
      for _k = 0; _k < keep_axes.length(); _k = _k + 1 {
        self.body.write_string("  }\n")
      }
    }
    other => raise UnsupportedOp("arena emit unsupported op '\{other.name()}'")
  }
}
