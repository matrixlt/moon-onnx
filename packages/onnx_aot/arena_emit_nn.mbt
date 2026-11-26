///|
/// Arena 模式神经网络核心算子发射器（Conv / Pool / MatMul / Gemm / 逐元素广播与激活 / Norm / 量化算子）。
/// 若当前节点已处理返回 `true`，否则交由 `emit_tensor_op` 处理布局与归约算子。
fn ArenaEmitter::emit_nn_op(
  self : ArenaEmitter,
  i : Int32,
  node : @nn_ir.IrNode,
  out_n : String,
  out_sh : Array[Int32],
  out_numel : Int32,
) -> Bool raise CodegenError {
  let ins = node.inputs
  match node.op {
    @nn_ir.Conv(strides~, pads~, dilations~, group~, auto_pad~, activation~, ..) => {
      self.need_conv = true
      self.need_gemm = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
      let sw = self.shapes.get(ins[1]).unwrap()
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
        self.need_f32_loader = true
        let bf = self.reg_weight(ins[2], RawF32)
        "Some(self.\{bf})"
      } else {
        "None"
      }
      let ep_expr = fmt_gemm_epilogue(activation)
      let is_dw = group > 1 && group == c_in && group == c_out
      if is_dw {
        self.need_f32_loader = true
        let wf = self.reg_weight(ins[1], RawF32)
        for b = 0; b < n_batch; b = b + 1 {
          let xb = x_off + b * c_in * ih * iw
          let yb = y_off + b * c_out * oh * ow
          self.body.write_string(
            "  @conv.depthwise_conv2d_f32(\{x_buf}, \{xb}, self.\{wf}, \{bias_expr}, \{c_in}, \{ih}, \{iw}, \{kh}, \{kw}, \{sh}, \{sw_s}, \{pt}, \{pl}, \{dh}, \{dw_s}, \{oh}, \{ow}, \{y_buf}, \{yb}, epilogue=\{ep_expr})\n",
          )
        }
      } else {
        let k_dim = c_in * kh * kw
        let spatial = oh * ow
        self.need_f32_loader = true
        if self.use_packed_f32_simd {
          let wf = self.reg_weight(ins[1], PackedConv(group=1))
          for b = 0; b < n_batch; b = b + 1 {
            let xb = x_off + b * c_in * ih * iw
            let yb = y_off + b * c_out * spatial
            self.body.write_string(
              "  @conv.im2col_2d_nhwc_f32(\{x_buf}, \{xb}, \{c_in}, \{ih}, \{iw}, \{kh}, \{kw}, \{sh}, \{sw_s}, \{pt}, \{pl}, \{dh}, \{dw_s}, \{oh}, \{ow}, self.scratch)\n",
            )
            self.body.write_string(
              "  @gemm.matmul_2d_prepacked_nchw_simd_f32(self.scratch, 0, self.\{wf}, \{bias_expr}, 0, \{y_buf}, \{yb}, \{spatial}, epilogue=\{ep_expr})\n",
            )
          }
        } else {
          self.need_nn = true
          let wf = self.reg_weight(ins[1], RawF32)
          for b = 0; b < n_batch; b = b + 1 {
            let xb = x_off + b * c_in * ih * iw
            let yb = y_off + b * c_out * spatial
            self.body.write_string(
              "  @conv.im2col_2d_nhwc_f32(\{x_buf}, \{xb}, \{c_in}, \{ih}, \{iw}, \{kh}, \{kw}, \{sh}, \{sw_s}, \{pt}, \{pl}, \{dh}, \{dw_s}, \{oh}, \{ow}, self.scratch)\n",
            )
            self.body.write_string(
              "  @nn.conv_nchw_scalar_f32(self.scratch, self.\{wf}, \{bias_expr}, \{y_buf}, \{yb}, \{spatial}, \{k_dim}, \{c_out}, \{ep_expr})\n",
            )
          }
        }
      }
    }
    @nn_ir.MaxPool(kernel_shape~, strides~, pads~, dilations~, auto_pad~, ..) => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
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
      self.body.write_string(
        "  @nn.maxpool2d_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0]}, \{sx[1]}, \{sx[2]}, \{sx[3]}, \{kernel_shape[0]}, \{kernel_shape[1]}, \{strides[0]}, \{strides[1]}, \{pt}, \{pl}, \{dilations[0]}, \{dilations[1]}, \{out_sh[2]}, \{out_sh[3]})\n",
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
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
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
      self.body.write_string(
        "  @nn.avgpool2d_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0]}, \{sx[1]}, \{sx[2]}, \{sx[3]}, \{kernel_shape[0]}, \{kernel_shape[1]}, \{strides[0]}, \{strides[1]}, \{pt}, \{pl}, \{out_sh[2]}, \{out_sh[3]}, \{count_include_pad})\n",
      )
    }
    @nn_ir.GlobalAveragePool => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
      let mut spatial = 1
      for d = 2; d < sx.length(); d = d + 1 {
        spatial = spatial * sx[d]
      }
      self.body.write_string(
        "  @nn.gap_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0] * sx[1]}, \{spatial})\n",
      )
    }
    @nn_ir.Resize(mode~, coordinate_transformation_mode~, nearest_mode~) => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sx = self.shapes.get(ins[0]).unwrap()
      let is_linear = mode == "linear" || mode == "bilinear"
      self.body.write_string(
        "  @nn.resize_2d_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0]}, \{sx[1]}, \{sx[2]}, \{sx[3]}, \{out_sh[2]}, \{out_sh[3]}, \{is_linear}, \"\{coordinate_transformation_mode}\", \"\{nearest_mode}\")\n",
      )
    }
    @nn_ir.MatMul => {
      self.need_gemm = true
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.shapes.get(ins[0]).unwrap()
      let sb = self.shapes.get(ins[1]).unwrap()
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
      let sa = self.shapes.get(ins[0]).unwrap()
      let sb = self.shapes.get(ins[1]).unwrap()
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
        let sc = self.shapes.get(ins[2]).unwrap()
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
    @nn_ir.Add | @nn_ir.Sub | @nn_ir.Mul | @nn_ir.Div => {
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (b_buf, b_off) = self.resolve_f32_loc(ins[1])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.shapes.get(ins[0]).unwrap()
      let sb = self.shapes.get(ins[1]).unwrap()
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
        self.body.write_string(
          "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = \{a_buf}[\{a_off} + k_i] \{op_sym} \{b_buf}[\{b_off} + k_i] }\n",
        )
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
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; \{y_buf}[\{y_off} + k_i] = if v > (0.0 : Float) { v } else { (0.0 : Float) } }\n",
      )
    }
    @nn_ir.Clip(min~, max~) => {
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let lo_v : Float? = if ins.length() >= 2 && ins[1] != "" {
        Some(
          self.graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          },
        )
      } else {
        min
      }
      let hi_v : Float? = if ins.length() >= 3 && ins[2] != "" {
        Some(
          self.graph.initializers.get(ins[2]).unwrap().get_f32_flat(0) catch {
            e => raise TensorErr("\{e}")
          },
        )
      } else {
        max
      }
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
      let sixth = fmt_float_literal(Float::from_double(1.0 / 6.0))
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { let v = \{x_buf}[\{x_off} + k_i]; let t = v * \{sixth} + (0.5 : Float); let c = if t < (0.0 : Float) { (0.0 : Float) } else if t > (1.0 : Float) { (1.0 : Float) } else { t }; \{y_buf}[\{y_off} + k_i] = v * c }\n",
      )
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
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(\{x_buf}[\{x_off} + k_i].to_double().sqrt()) }\n",
      )
    }
    @nn_ir.Exp => {
      self.need_math = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.exp(\{x_buf}[\{x_off} + k_i].to_double())) }\n",
      )
    }
    @nn_ir.Sigmoid => {
      self.need_math = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(1.0 / (1.0 + @math.exp(-\{x_buf}[\{x_off} + k_i].to_double()))) }\n",
      )
    }
    @nn_ir.Tanh => {
      self.need_math = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  for k_i = 0; k_i < \{out_numel}; k_i = k_i + 1 { \{y_buf}[\{y_off} + k_i] = Float::from_double(@math.tanh(\{x_buf}[\{x_off} + k_i].to_double())) }\n",
      )
    }
    @nn_ir.Gelu => {
      self.need_nn = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      self.body.write_string(
        "  @nn.gelu_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{out_numel})\n",
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
    @nn_ir.QuantizeLinear(..) => {
      self.need_quant = true
      let (x_buf, x_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_byte_loc(out_n)
      let sc = self.graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
          e => raise TensorErr("\{e}")
        }
      let out_dt = self.dtypes.get(out_n).unwrap()
      let zp_byte : Byte = if ins.length() >= 3 && ins[2] != "" {
        let zt = self.graph.initializers.get(ins[2]).unwrap()
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
      self.body.write_string(
        "  \{q_fn}(\{x_buf}, \{fmt_float_literal(sc)}, (\{zp_byte.to_int()}).to_byte(), \{y_buf}, \{out_numel}, src_off=\{x_off}, dst_off=\{y_off})\n",
      )
    }
    @nn_ir.DequantizeLinear(..) => {
      self.need_quant = true
      let (x_buf, x_off) = self.resolve_byte_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sc = self.graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
          e => raise TensorErr("\{e}")
        }
      let in_dt = self.dtypes.get(ins[0]).unwrap()
      let zp_byte : Byte = if ins.length() >= 3 && ins[2] != "" {
        let zt = self.graph.initializers.get(ins[2]).unwrap()
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
      self.body.write_string(
        "  \{dq_fn}(\{x_buf}, \{fmt_float_literal(sc)}, (\{zp_byte.to_int()}).to_byte(), \{y_buf}, \{out_numel}, src_off=\{x_off}, dst_off=\{y_off})\n",
      )
    }
    @nn_ir.QLinearMatMul => {
      self.need_quant = true
      self.need_byte_loader = true
      let (a_buf, a_off) = self.resolve_byte_loc(ins[0])
      let (y_buf, y_off) = self.resolve_byte_loc(out_n)
      let sa = self.shapes.get(ins[0]).unwrap()
      let sb = self.shapes.get(ins[3]).unwrap()
      let m = sa[0]
      let k = sa[1]
      let n = sb[1]
      let a_dt = self.dtypes.get(ins[0]).unwrap()
      let a_sc = self.graph.initializers.get(ins[1]).unwrap().get_f32_flat(0) catch {
          e => raise TensorErr("\{e}")
        }
      let b_sc = self.graph.initializers.get(ins[4]).unwrap().get_f32_flat(0) catch {
          e => raise TensorErr("\{e}")
        }
      let y_sc = self.graph.initializers.get(ins[6]).unwrap().get_f32_flat(0) catch {
          e => raise TensorErr("\{e}")
        }
      let a_zp : Int32 = match a_dt {
        @tensor.Int8 =>
          (self.graph.initializers.get(ins[2]).unwrap().to_i8_buffer() catch {
            e => raise TensorErr("\{e}")
          })[0].to_int()
        _ =>
          (self.graph.initializers.get(ins[2]).unwrap().to_u8_buffer() catch {
            e => raise TensorErr("\{e}")
          })[0].to_int()
      }
      let b_zp : Int32 = match a_dt {
        @tensor.Int8 =>
          (self.graph.initializers.get(ins[5]).unwrap().to_i8_buffer() catch {
            e => raise TensorErr("\{e}")
          })[0].to_int()
        _ =>
          (self.graph.initializers.get(ins[5]).unwrap().to_u8_buffer() catch {
            e => raise TensorErr("\{e}")
          })[0].to_int()
      }
      let y_zp : Int32 = match a_dt {
        @tensor.Int8 =>
          (self.graph.initializers.get(ins[7]).unwrap().to_i8_buffer() catch {
            e => raise TensorErr("\{e}")
          })[0].to_int()
        _ =>
          (self.graph.initializers.get(ins[7]).unwrap().to_u8_buffer() catch {
            e => raise TensorErr("\{e}")
          })[0].to_int()
      }
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
      let cast_out = self.mmi_to_cast.get(out_n).unwrap()
      let (y_buf, y_off) = self.resolve_f32_loc(cast_out)
      let sa = self.shapes.get(ins[0]).unwrap()
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
        let bzt = self.graph.initializers.get(ins[3]).unwrap()
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
      let pos_param = self.input_param_of.get(ins[1]).unwrap()
      let cos_f = self.reg_weight(ins[2], RawF32)
      let sin_f = self.reg_weight(ins[3], RawF32)
      let cs = self.shapes.get(ins[2]).unwrap()
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
      let pos_sh = self.shapes.get(ins[1]).unwrap()
      let pos_numel = @tensor.num_elements(pos_sh) catch {
        e => raise ShapeError("\{e}")
      }
      let pos_is_scalar = pos_numel == 1 && batch * seq_len > 1
      self.body.write_string(
        "  @nn.rotary_embedding_f32(\{x_buf}, \{x_off}, \{pos_param}, \{pos_is_scalar}, self.\{cos_f}, self.\{sin_f}, \{cache_half}, \{y_buf}, \{y_off}, \{batch}, \{seq_len}, \{eff_heads}, \{head_dim}, \{rot_dim}, \{interleaved != 0}, \{is_bhsd})\n",
      )
    }
    @nn_ir.MatMulNBits(k~, n=out_n_dim, block_size~, ..) => {
      self.need_quant = true
      self.need_byte_loader = true
      self.need_f32_loader = true
      let (a_buf, a_off) = self.resolve_f32_loc(ins[0])
      let (y_buf, y_off) = self.resolve_f32_loc(out_n)
      let sa = self.shapes.get(ins[0]).unwrap()
      let mut m_tot = 1
      for d = 0; d < sa.length() - 1; d = d + 1 {
        m_tot = m_tot * sa[d]
      }
      let eff_k = if k > 0 { k } else { sa[sa.length() - 1] }
      let eff_n = if out_n_dim > 0 {
        out_n_dim
      } else {
        self.shapes.get(ins[1]).unwrap()[0]
      }
      let wf = self.reg_weight(ins[1], RawBytes(trans_b=false))
      let sf = self.reg_weight(ins[2], RawF32)
      let (zp_b_expr, zp_f_expr) = if ins.length() >= 4 && ins[3] != "" {
        let zt = self.graph.initializers.get(ins[3]).unwrap()
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
