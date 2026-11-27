///|
/// Arena 模式神经网络核心算子发射器入口（Conv / Pool / Resize，并委托线性代数/量化与逐元素/LLM 发射器）。
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
      let sx = self.require_shape(ins[0])
      let sw = self.require_shape(ins[1])
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
      let (pt, pl, _pb, _pr) = @analysis.resolve_2d_pads(
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
      let sx = self.require_shape(ins[0])
      let (pt, pl, _pb, _pr) = @analysis.resolve_2d_pads(
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
      let sx = self.require_shape(ins[0])
      let (pt, pl, _pb, _pr) = @analysis.resolve_2d_pads(
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
      let sx = self.require_shape(ins[0])
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
      let sx = self.require_shape(ins[0])
      let is_linear = mode == "linear" || mode == "bilinear"
      self.body.write_string(
        "  @nn.resize_2d_f32(\{x_buf}, \{x_off}, \{y_buf}, \{y_off}, \{sx[0]}, \{sx[1]}, \{sx[2]}, \{sx[3]}, \{out_sh[2]}, \{out_sh[3]}, \{is_linear}, \"\{coordinate_transformation_mode}\", \"\{nearest_mode}\")\n",
      )
    }
    _ =>
      return self.emit_linalg_quant_op(i, node, out_n, out_sh, out_numel) ||
        self.emit_elem_llm_op(node, out_n, out_sh, out_numel)
  }
  true
}
