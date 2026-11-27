///|
/// Arena 模式矩阵乘与量化算子发射器（MatMul / Gemm / QuantizeLinear / DequantizeLinear / QLinearMatMul / DynamicQuantizeLinear / MatMulInteger / MatMulNBits）。
fn ArenaEmitter::emit_linalg_quant_op(
  self : ArenaEmitter,
  i : Int32,
  node : @nn_ir.IrNode,
  out_n : String,
  out_sh : Array[Int32],
  out_numel : Int32,
) -> Bool raise CodegenError {
  let _ = out_sh
  let ins = node.inputs
  match node.op {
    @nn_ir.MatMul => {
      self.need_gemm = true
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.require_shape(ins[0])
      let sb = self.require_shape(ins[1])
      if sb.length() == 2 && self.graph.initializers.contains(ins[1]) {
        // N-D × 2D 静态权重投影（[B, S, K] @ [K, N] -> [B*S, K] @ [K, N]）
        self.need_f32_loader = true
        let k_dim = sa[sa.length() - 1]
        let n_dim = sb[1]
        let mut m_tot = 1
        for d = 0; d < sa.length() - 1; d = d + 1 {
          m_tot = m_tot * sa[d]
        }
        if self.use_packed_f32_simd {
          let wf = self.reg_weight(ins[1], PackedMatB(trans_b=false))
          self.body.write_string(
            "  @gemm.matmul_2d_prepacked_simd_f32(\{a_buf}, \{a_off}, self.\{wf}, \{y_buf}, \{y_off}, \{m_tot})\n",
          )
        } else {
          let wf = self.reg_weight(ins[1], RawF32)
          self.body.write_string(
            "  @gemm.matmul_2d_scalar_f32(\{a_buf}, \{a_off}, self.\{wf}, 0, \{y_buf}, \{y_off}, \{m_tot}, \{k_dim}, \{n_dim})\n",
          )
        }
      } else {
        // 2D/3D/4D 动态批量矩阵乘（多头自注意力 Q@K^T 与 P@V）
        let (b_buf, b_off) = self.resolve_f32_loc(ins[1])
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
        let mm_fn = if self.use_packed_f32_simd {
          "@gemm.matmul_2d_simd_f32"
        } else {
          "@gemm.matmul_2d_scalar_f32"
        }
        if batches == 1 {
          self.body.write_string(
            "  \{mm_fn}(\{a_buf}, \{a_off}, \{b_buf}, \{b_off}, \{y_buf}, \{y_off}, \{m_dim}, \{k_dim}, \{n_dim})\n",
          )
        } else {
          self.body.write_string(
            "  for bm_i = 0; bm_i < \{batches}; bm_i = bm_i + 1 { \{mm_fn}(\{a_buf}, \{a_off} + bm_i * \{mat_a}, \{b_buf}, \{b_off} + bm_i * \{mat_b}, \{y_buf}, \{y_off} + bm_i * \{mat_y}, \{m_dim}, \{k_dim}, \{n_dim}) }\n",
          )
        }
      }
    }
    @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
      self.need_gemm = true
      self.need_f32_loader = true
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.require_shape(ins[0])
      let sb = self.require_shape(ins[1])
      let m = if trans_a { sa[1] } else { sa[0] }
      let k = if trans_a { sa[0] } else { sa[1] }
      let n = if trans_b { sb[0] } else { sb[1] }
      if trans_a {
        raise UnsupportedOp("Arena Gemm does not support trans_a=true yet")
      }
      if self.use_packed_f32_simd {
        let wf = self.reg_weight(ins[1], PackedMatB(trans_b~))
        self.body.write_string(
          "  @gemm.matmul_2d_prepacked_simd_f32(\{a_buf}, \{a_off}, self.\{wf}, \{y_buf}, \{y_off}, \{m})\n",
        )
      } else {
        let wf = if trans_b {
          self.need_trans_f32 = true
          self.reg_weight(ins[1], PackedMatB(trans_b=true))
        } else {
          self.reg_weight(ins[1], RawF32)
        }
        self.body.write_string(
          "  @gemm.matmul_2d_scalar_f32(\{a_buf}, \{a_off}, self.\{wf}, 0, \{y_buf}, \{y_off}, \{m}, \{k}, \{n})\n",
        )
      }
      if alpha != (1.0 : Float) {
        self.body.write_string(
          "  for k_i = 0; k_i < \{m * n}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{y_buf}[\{y_off} + k_i] * \{fmt_float_literal(alpha)} }\n",
        )
      }
      if ins.length() >= 3 && ins[2] != "" {
        let (c_buf, c_off) = self.resolve_f32_loc(ins[2])
        let sc = self.require_shape(ins[2])
        let c_numel = @tensor.num_elements(sc) catch {
          e => raise ShapeError("\{e}")
        }
        let beta_mul = if beta == (1.0 : Float) {
          ""
        } else {
          " * \{fmt_float_literal(beta)}"
        }
        if c_numel == 1 {
          self.body.write_string(
            "  for k_i = 0; k_i < \{m * n}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{y_buf}[\{y_off} + k_i] + \{c_buf}[\{c_off}]\{beta_mul} }\n",
          )
        } else if c_numel == n {
          self.body.write_string(
            "  for r_i = 0; r_i < \{m}; r_i = r_i + 1 { for c_i = 0; c_i < \{n}; c_i = c_i + 1 { \{y_buf}[\{y_off} + r_i * \{n} + c_i] = \{y_buf}[\{y_off} + r_i * \{n} + c_i] + \{c_buf}[\{c_off} + c_i]\{beta_mul} } }\n",
          )
        } else {
          self.body.write_string(
            "  for k_i = 0; k_i < \{m * n}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{y_buf}[\{y_off} + k_i] + \{c_buf}[\{c_off} + k_i]\{beta_mul} }\n",
          )
        }
      }
    }
    @nn_ir.QuantizeLinear(..) => {
      self.need_quant = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_byte_loc(out_n)
      let sc = self.require_init_f32_scalar(ins[1])
      let out_dt = self.require_dtype(out_n)
      let zp_byte : Byte = if ins.length() >= 3 && ins[2] != "" {
        let zt = self.require_init(ins[2])
        self.require_init_zp_byte(ins[2], zt.dtype() == @tensor.Int8)
      } else {
        b'\x00'
      }
      let q_fn = match out_dt {
        @tensor.Int8 => "@quant.quantize_i8_per_tensor"
        _ => "@quant.quantize_u8_per_tensor"
      }
      self.body.write_string(
        "  \{q_fn}(\{x_buf}, \{fmt_float_literal(sc)}, (\{zp_byte.to_int()}).to_byte(), \{y_buf}, \{out_numel}, src_off=\{x_off}, dst_off=\{y_off})\n",
      )
    }
    @nn_ir.DequantizeLinear(..) => {
      self.need_quant = true
      let (x_buf, x_off) = self.resolve_byte_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sc = self.require_init_f32_scalar(ins[1])
      let in_dt = self.require_dtype(ins[0])
      let zp_byte : Byte = if ins.length() >= 3 && ins[2] != "" {
        let zt = self.require_init(ins[2])
        self.require_init_zp_byte(ins[2], zt.dtype() == @tensor.Int8)
      } else {
        b'\x00'
      }
      let dq_fn = match in_dt {
        @tensor.Int8 => "@quant.dequantize_i8_per_tensor"
        _ => "@quant.dequantize_u8_per_tensor"
      }
      self.body.write_string(
        "  \{dq_fn}(\{x_buf}, \{fmt_float_literal(sc)}, (\{zp_byte.to_int()}).to_byte(), \{y_buf}, \{out_numel}, src_off=\{x_off}, dst_off=\{y_off})\n",
      )
    }
    @nn_ir.QLinearMatMul => {
      self.need_quant = true
      self.need_byte_loader = true
      let (a_buf, a_off) = self.resolve_byte_loc(ins[0])
      let (y_buf, y_off) = self.resolve_byte_loc(out_n)
      let sa = self.require_shape(ins[0])
      let sb = self.require_shape(ins[3])
      let m = sa[0]
      let k = sa[1]
      let n = sb[1]
      let a_dt = self.require_dtype(ins[0])
      let is_i8 = a_dt == @tensor.Int8
      let a_sc = self.require_init_f32_scalar(ins[1])
      let b_sc = self.require_init_f32_scalar(ins[4])
      let y_sc = self.require_init_f32_scalar(ins[6])
      let a_zp = self.require_init_zp_byte(ins[2], is_i8).to_int()
      let b_zp = self.require_init_zp_byte(ins[5], is_i8).to_int()
      let y_zp = self.require_init_zp_byte(ins[7], is_i8).to_int()
      if self.use_quant_simd {
        let wf = self.reg_weight(ins[3], RawBytes(trans_b=true))
        let qmm_fn = match a_dt {
          @tensor.Int8 => "@quant.qlinear_matmul_i8_pretransposed_v128"
          _ => "@quant.qlinear_matmul_u8_pretransposed_v128"
        }
        self.body.write_string(
          "  \{qmm_fn}(\{a_buf}, \{a_off}, \{fmt_float_literal(a_sc)}, (\{a_zp}).to_byte(), self.\{wf}, \{fmt_float_literal(b_sc)}, (\{b_zp}).to_byte(), \{fmt_float_literal(y_sc)}, (\{y_zp}).to_byte(), \{y_buf}, \{y_off}, \{m}, \{k}, \{n})\n",
        )
      } else {
        let wf = self.reg_weight(ins[3], RawBytes(trans_b=false))
        let qmm_fn = match a_dt {
          @tensor.Int8 => "@quant.qlinear_matmul_i8_scalar"
          _ => "@quant.qlinear_matmul_u8_scalar"
        }
        self.body.write_string(
          "  \{qmm_fn}(\{a_buf}, \{fmt_float_literal(a_sc)}, (\{a_zp}).to_byte(), self.\{wf}, \{fmt_float_literal(b_sc)}, (\{b_zp}).to_byte(), \{fmt_float_literal(y_sc)}, (\{y_zp}).to_byte(), \{y_buf}, \{m}, \{k}, \{n}, a_off=\{a_off}, out_off=\{y_off})\n",
        )
      }
    }
    @nn_ir.DynamicQuantizeLinear => {
      self.need_quant = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_byte_loc(node.outputs[0])
      let (sc_buf, sc_off) = self.resolve_f32_loc(node.outputs[1])
      let (zp_buf, zp_off) = self.resolve_byte_loc(node.outputs[2])
      self.body.write_string(
        "  let (dq_s_\{i}, dq_z_\{i}) = @quant.dynamic_quantize_linear_u8(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
      )
      self.body.write_string("  \{sc_buf}[\{sc_off}] = dq_s_\{i}\n")
      self.body.write_string("  \{zp_buf}[\{zp_off}] = dq_z_\{i}\n")
    }
    @nn_ir.MatMulInteger => {
      self.need_quant = true
      self.need_byte_loader = true
      let (a_buf, a_off) = self.resolve_byte_loc(ins[0])
      let cast_out = self.require_mmi_cast(out_n)
      let (y_buf, y_off) = self.resolve_f32_loc(cast_out)
      let sa = self.require_shape(ins[0])
      let mut m_tot = 1
      for d = 0; d < sa.length() - 1; d = d + 1 {
        m_tot = m_tot * sa[d]
      }
      let wf = self.reg_weight(ins[1], PackedInt8MatB)
      let a_zp_expr = if ins.length() >= 3 && ins[2] != "" {
        let (az_buf, az_off) = self.resolve_byte_loc(ins[2])
        "\{az_buf}[\{az_off}]"
      } else {
        "b'\\x00'"
      }
      let b_zp_expr = if ins.length() >= 4 && ins[3] != "" {
        let bzt = self.require_init(ins[3])
        let bzb = bzt.to_i8_buffer() catch { e => raise TensorErr("\{e}") }
        let mut any_nz = false
        for bi = 0; bi < bzb.length(); bi = bi + 1 {
          if bzb[bi] != b'\x00' {
            any_nz = true
            break
          }
        }
        if any_nz {
          let bzf = self.reg_weight(ins[3], RawBytes(trans_b=false))
          "Some(self.\{bzf})"
        } else {
          "None"
        }
      } else {
        "None"
      }
      let mmi_fn = if self.use_quant_simd {
        "@quant.matmul_integer_u8_i8_prepacked_to_f32_v128"
      } else {
        "@quant.matmul_integer_u8_i8_prepacked_to_f32_scalar"
      }
      self.body.write_string(
        "  \{mmi_fn}(\{a_buf}, \{a_off}, \{a_zp_expr}, self.\{wf}, \{b_zp_expr}, \{y_buf}, \{y_off}, \{m_tot})\n",
      )
    }
    @nn_ir.MatMulNBits(k~, n=out_n_dim, block_size~, ..) => {
      self.need_quant = true
      self.need_byte_loader = true
      self.need_f32_loader = true
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.require_shape(ins[0])
      let mut m_tot = 1
      for d = 0; d < sa.length() - 1; d = d + 1 {
        m_tot = m_tot * sa[d]
      }
      let eff_k = if k > 0 { k } else { sa[sa.length() - 1] }
      let eff_n = if out_n_dim > 0 {
        out_n_dim
      } else {
        self.require_shape(ins[1])[0]
      }
      let wf = self.reg_weight(ins[1], RawBytes(trans_b=false))
      let sf = self.reg_weight(ins[2], RawF32)
      let (zp_b_expr, zp_f_expr) = if ins.length() >= 4 && ins[3] != "" {
        let zt = self.require_init(ins[3])
        match zt.dtype() {
          @tensor.UInt8 => {
            let zf = self.reg_weight(ins[3], RawBytes(trans_b=false))
            ("Some(self.\{zf})", "None")
          }
          _ => {
            let zf = self.reg_weight(ins[3], RawF32)
            ("None", "Some(self.\{zf})")
          }
        }
      } else {
        ("None", "None")
      }
      let bias_expr = if ins.length() >= 6 && ins[5] != "" {
        let bf = self.reg_weight(ins[5], RawF32)
        "Some(self.\{bf})"
      } else {
        "None"
      }
      let q4_fn = if self.use_quant_simd {
        "@quant.matmul_nbits_q4_v128_f32"
      } else {
        "@quant.matmul_nbits_q4_scalar_f32"
      }
      self.body.write_string(
        "  \{q4_fn}(\{a_buf}, \{a_off}, self.\{wf}, self.\{sf}, \{zp_b_expr}, \{zp_f_expr}, \{bias_expr}, \{y_buf}, \{y_off}, \{m_tot}, \{eff_k}, \{eff_n}, \{block_size})\n",
      )
    }
    _ => return false
  }
  true
}
