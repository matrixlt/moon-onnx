///|
/// Arena 模式逐元素广播、激活、归一化与 LLM 算子发射器（Add/Sub/Mul/Div、Relu/Clip/Sigmoid/Tanh/Gelu/Erf/SiLU、Softmax/LayerNorm/RMSNorm/RoPE）。
fn ArenaEmitter::emit_elem_llm_op(
  self : ArenaEmitter,
  node : @nn_ir.IrNode,
  out_n : String,
  out_sh : Array[Int32],
  out_numel : Int32,
) -> Bool raise CodegenError {
  let ins = node.inputs
  match node.op {
    @nn_ir.Add | @nn_ir.Sub | @nn_ir.Mul | @nn_ir.Div => {
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (b_buf, b_off) = self.resolve_f32_loc(ins[1])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.require_shape(ins[0])
      let sb = self.require_shape(ins[1])
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
        if self.use_packed_f32_simd {
          self.need_vecmath = true
          let vfn = match node.op {
            @nn_ir.Add => "add_f32_v128"
            @nn_ir.Sub => "sub_f32_v128"
            @nn_ir.Mul => "mul_f32_v128"
            _ => "div_f32_v128"
          }
          self.body.write_string(
            "  @vecmath.\{vfn}(\{a_buf}, \{b_buf}, \{y_buf}, \{out_numel}, a_off=\{a_off}, b_off=\{b_off}, out_off=\{y_off})\n",
          )
        } else {
          self.body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{a_buf}[\{a_off} + k_i] \{op_sym} \{b_buf}[\{b_off} + k_i] }\n",
          )
        }
      } else if b_numel == 1 {
        self.body.write_string("  let bs_v = \{b_buf}[\{b_off}]\n")
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{a_buf}[\{a_off} + k_i] \{op_sym} bs_v }\n",
        )
      } else if a_numel == 1 {
        self.body.write_string("  let as_v = \{a_buf}[\{a_off}]\n")
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = as_v \{op_sym} \{b_buf}[\{b_off} + k_i] }\n",
        )
      } else if sb.length() == 1 &&
        out_sh.length() >= 1 &&
        sa == out_sh &&
        sb[0] == out_sh[out_sh.length() - 1] {
        let cols = out_sh[out_sh.length() - 1]
        let rows = out_numel / cols
        self.body.write_string(
          "  for r_i = 0; r_i < \{rows}; r_i = r_i + 1 { let base_i = r_i * \{cols}; for c_i = 0; c_i < \{cols}; c_i = c_i + 1 { \{y_buf}[\{y_off} + base_i + c_i] = \{a_buf}[\{a_off} + base_i + c_i] \{op_sym} \{b_buf}[\{b_off} + c_i] } }\n",
        )
      } else if sa.length() == 1 &&
        out_sh.length() >= 1 &&
        sb == out_sh &&
        sa[0] == out_sh[out_sh.length() - 1] {
        // 对称偏置加：[C] op [..., C]（PyTorch Transformer 线性层偏置常见顺序）
        let cols = out_sh[out_sh.length() - 1]
        let rows = out_numel / cols
        self.body.write_string(
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
          self.body.write_string(
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
        self.body.write_string(
          "    \{y_buf}[\{ey.to_string()}] = \{a_buf}[\{ea.to_string()}] \{op_sym} \{b_buf}[\{eb.to_string()}]\n",
        )
        for _d = 0; _d < rank; _d = _d + 1 {
          self.body.write_string("  }\n")
        }
      }
    }
    @nn_ir.Relu => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.relu_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; \{y_buf}[\{y_off} + k_i] = if v > (0.0 : Float) { v } else { (0.0 : Float) } }\n",
        )
      }
    }
    @nn_ir.Clip(min~, max~) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let lo_v : Float? = if ins.length() >= 2 && ins[1] != "" {
        Some(self.require_init_f32_scalar(ins[1]))
      } else {
        min
      }
      let hi_v : Float? = if ins.length() >= 3 && ins[2] != "" {
        Some(self.require_init_f32_scalar(ins[2]))
      } else {
        max
      }
      match (self.use_packed_f32_simd, lo_v, hi_v) {
        (true, Some(lo), Some(hi)) => {
          self.need_vecmath = true
          self.body.write_string(
            "  @vecmath.clip_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, \{fmt_float_literal(lo)}, \{fmt_float_literal(hi)}, src_off=\{x_off}, out_off=\{y_off})\n",
          )
        }
        _ => {
          self.body.write_string(
            "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 {\n",
          )
          self.body.write_string("    let mut v = \{x_buf}[\{x_off} + k_i]\n")
          match lo_v {
            Some(lo) =>
              self.body.write_string(
                "    if v < \{fmt_float_literal(lo)} { v = \{fmt_float_literal(lo)} }\n",
              )
            None => ()
          }
          match hi_v {
            Some(hi) =>
              self.body.write_string(
                "    if v > \{fmt_float_literal(hi)} { v = \{fmt_float_literal(hi)} }\n",
              )
            None => ()
          }
          self.body.write_string("    \{y_buf}[\{y_off} + k_i] = v\n  }\n")
        }
      }
    }
    @nn_ir.LeakyRelu(alpha~) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let al = fmt_float_literal(alpha)
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; \{y_buf}[\{y_off} + k_i] = if v >= (0.0 : Float) { v } else { v * \{al} } }\n",
      )
    }
    @nn_ir.HardSwish => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.hard_swish_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        let sixth = fmt_float_literal(Float::from_double(1.0 / 6.0))
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; let t = v * \{sixth} + (0.5 : Float); let c = if t < (0.0 : Float) { (0.0 : Float) } else if t > (1.0 : Float) { (1.0 : Float) } else { t }; \{y_buf}[\{y_off} + k_i] = v * c }\n",
        )
      }
    }
    @nn_ir.Neg => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = -\{x_buf}[\{x_off} + k_i] }\n",
      )
    }
    @nn_ir.Sqrt => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.sqrt_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(\{x_buf}[\{x_off} + k_i].to_double().sqrt()) }\n",
        )
      }
    }
    @nn_ir.Exp => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.exp_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        self.need_math = true
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.exp(\{x_buf}[\{x_off} + k_i].to_double())) }\n",
        )
      }
    }
    @nn_ir.Sigmoid => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.sigmoid_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        self.need_math = true
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(1.0 / (1.0 + @math.exp(-\{x_buf}[\{x_off} + k_i].to_double()))) }\n",
        )
      }
    }
    @nn_ir.Tanh => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.tanh_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        self.need_math = true
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.tanh(\{x_buf}[\{x_off} + k_i].to_double())) }\n",
        )
      }
    }
    @nn_ir.Gelu => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      if self.use_packed_f32_simd {
        self.need_vecmath = true
        self.body.write_string(
          "  @vecmath.gelu_f32_v128(\{x_buf}, \{y_buf}, \{out_numel}, src_off=\{x_off}, out_off=\{y_off})\n",
        )
      } else {
        self.need_nn = true
        self.body.write_string(
          "  @nn.gelu_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
        )
      }
    }
    @nn_ir.GeluExact => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  @nn.gelu_exact_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
      )
    }
    @nn_ir.Erf => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  @nn.erf_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
      )
    }
    @nn_ir.Softmax(axis~) => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
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
      self.body.write_string(
        "  @nn.softmax_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{outer}, \{axis_dim}, \{inner})\n",
      )
    }
    @nn_ir.LayerNormalization(axis~, epsilon~) => {
      self.need_nn = true
      self.need_f32_loader = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sf = self.reg_weight(ins[1], RawF32)
      let bias_expr = if ins.length() >= 3 && ins[2] != "" {
        let bf = self.reg_weight(ins[2], RawF32)
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
      self.body.write_string(
        "  @nn.layernorm_f32(\{x_buf}, \{x_off}, self.\{sf}, \{bias_expr}, \{y_buf}, \{y_off}, \{outer}, \{norm_sz}, \{fmt_float_literal(epsilon)})\n",
      )
    }
    @nn_ir.Silu => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  @nn.silu_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
      )
    }
    @nn_ir.RmsNormalization(axis~, epsilon~) => {
      self.need_nn = true
      self.need_f32_loader = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sf = self.reg_weight(ins[1], RawF32)
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
      self.body.write_string(
        "  @nn.rmsnorm_f32(\{x_buf}, \{x_off}, self.\{sf}, \{y_buf}, \{y_off}, \{outer}, \{norm_sz}, \{fmt_float_literal(epsilon)})\n",
      )
    }
    @nn_ir.SkipRmsNormalization(epsilon~) => {
      self.need_nn = true
      self.need_f32_loader = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (sk_buf, sk_off) = self.resolve_f32_loc(ins[1])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sf = self.reg_weight(ins[2], RawF32)
      let bias_expr = if ins.length() >= 4 && ins[3] != "" {
        let bf = self.reg_weight(ins[3], RawF32)
        "Some(self.\{bf})"
      } else {
        "None"
      }
      let (sum_expr, sum_off) = if node.outputs.length() >= 4 &&
        node.outputs[3] != "" {
        let (sb, so) = self.resolve_f32_loc(node.outputs[3])
        ("Some(\{sb})", so)
      } else {
        ("None", 0)
      }
      let norm_sz = out_sh[out_sh.length() - 1]
      let outer = out_numel / norm_sz
      self.body.write_string(
        "  @nn.skip_rmsnorm_f32(\{x_buf}, \{x_off}, \{sk_buf}, \{sk_off}, self.\{sf}, \{bias_expr}, \{y_buf}, \{y_off}, \{sum_expr}, \{sum_off}, \{outer}, \{norm_sz}, \{fmt_float_literal(epsilon)})\n",
      )
    }
    @nn_ir.RotaryEmbedding(interleaved~, num_heads~, rotary_embedding_dim~) => {
      self.need_nn = true
      self.need_f32_loader = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let pos_param = self.require_input_param(ins[1])
      let cos_f = self.reg_weight(ins[2], RawF32)
      let sin_f = self.reg_weight(ins[3], RawF32)
      let cs = self.require_shape(ins[2])
      let cache_half = cs[1]
      let is_bhsd = out_sh.length() == 4
      let batch = out_sh[0]
      let seq_len = if is_bhsd { out_sh[2] } else { out_sh[1] }
      let (eff_heads, head_dim) = if is_bhsd {
        (out_sh[1], out_sh[3])
      } else if num_heads > 0 {
        (num_heads, out_sh[2] / num_heads)
      } else {
        let hd = cache_half * 2
        (out_sh[2] / hd, hd)
      }
      let rot_dim = if rotary_embedding_dim > 0 {
        rotary_embedding_dim
      } else {
        cache_half * 2
      }
      let pos_sh = self.require_shape(ins[1])
      let pos_numel = @tensor.num_elements(pos_sh) catch {
        e => raise ShapeError("\{e}")
      }
      let pos_is_scalar = pos_numel == 1 && batch * seq_len > 1
      self.body.write_string(
        "  @nn.rotary_embedding_f32(\{x_buf}, \{x_off}, \{pos_param}, \{pos_is_scalar}, self.\{cos_f}, self.\{sin_f}, \{cache_half}, \{y_buf}, \{y_off}, \{batch}, \{seq_len}, \{eff_heads}, \{head_dim}, \{rot_dim}, \{interleaved != 0}, \{is_bhsd})\n",
      )
    }
    _ => return false
  }
  true
}
