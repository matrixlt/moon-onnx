///|
/// Tensor 兼容模式单节点 SSA 代码发射器：按算子类型将 `IrNode` 转换为 `@ops.*` 调用语句。
fn TensorEmitter::emit_node(
  self : TensorEmitter,
  node : @nn_ir.IrNode,
  out_name : String,
  out_var : String,
  pool_expr : String,
) -> Unit raise CodegenError {
  let ins = node.inputs
  let simd_str = self.simd_str
  match node.op {
    @nn_ir.Add => {
      let a = self.resolve_expr(ins[0])
      let b = self.resolve_expr(ins[1])
      self.body.write_string(
        "  let \{out_var} = @ops.add_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Sub => {
      let a = self.resolve_expr(ins[0])
      let b = self.resolve_expr(ins[1])
      self.body.write_string(
        "  let \{out_var} = @ops.sub_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Mul => {
      let a = self.resolve_expr(ins[0])
      let b = self.resolve_expr(ins[1])
      self.body.write_string(
        "  let \{out_var} = @ops.mul_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Div => {
      let a = self.resolve_expr(ins[0])
      let b = self.resolve_expr(ins[1])
      self.body.write_string(
        "  let \{out_var} = @ops.div_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Pow => {
      let a = self.resolve_expr(ins[0])
      let b = self.resolve_expr(ins[1])
      self.body.write_string(
        "  let \{out_var} = @ops.pow_f32(\{a}, \{b}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Sqrt => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.sqrt_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Exp => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.exp_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Neg => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.neg_f32(\{x}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Relu => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.relu_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Clip(min~, max~) => {
      let x = self.resolve_expr(ins[0])
      let lo_val : Float? = if ins.length() >= 2 && ins[1] != "" {
        match self.graph.initializers.get(ins[1]) {
          Some(t) =>
            Some(t.get_f32_flat(0) catch { e => raise TensorErr("\{e}") })
          None =>
            raise UnsupportedGraph("Clip '\{node.name}' requires static min")
        }
      } else {
        min
      }
      let hi_val : Float? = if ins.length() >= 3 && ins[2] != "" {
        match self.graph.initializers.get(ins[2]) {
          Some(t) =>
            Some(t.get_f32_flat(0) catch { e => raise TensorErr("\{e}") })
          None =>
            raise UnsupportedGraph("Clip '\{node.name}' requires static max")
        }
      } else {
        max
      }
      let lo_expr = match lo_val {
        Some(v) => "Some(\{fmt_float_literal(v)})"
        None => "None"
      }
      let hi_expr = match hi_val {
        Some(v) => "Some(\{fmt_float_literal(v)})"
        None => "None"
      }
      self.body.write_string(
        "  let \{out_var} = @ops.clip_f32(\{x}, \{lo_expr}, \{hi_expr}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Sigmoid => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.sigmoid_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Tanh => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.tanh_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Gelu => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.gelu_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.GeluExact => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.gelu_exact_f32(\{x}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.HardSwish => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.hard_swish_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.LeakyRelu(alpha~) => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.leaky_relu_f32(\{x}, alpha=\{fmt_float_literal(alpha)}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Identity => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string("  let \{out_var} = \{x}\n")
    }
    @nn_ir.Softmax(axis~) => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.softmax_f32(\{x}, \{axis}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Flatten(axis~) => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.flatten_op(\{x}, axis=\{axis})\n",
      )
    }
    @nn_ir.Reshape(allowzero~) => {
      let x = self.resolve_expr(ins[0])
      let sh = require_shape_from(self.shapes, out_name)
      self.body.write_string(
        "  let \{out_var} = \{x}.reshape(\{fmt_int_array(sh)}, allowzero=\{allowzero})\n",
      )
    }
    @nn_ir.Transpose(perm~) => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.transpose_op(\{x}, \{fmt_int_array(perm)})\n",
      )
    }
    @nn_ir.Squeeze(axes~) => {
      let x = self.resolve_expr(ins[0])
      let sh = require_shape_from(self.shapes, out_name)
      let _ = axes
      self.body.write_string(
        "  let \{out_var} = \{x}.reshape(\{fmt_int_array(sh)})\n",
      )
    }
    @nn_ir.Unsqueeze(axes~) => {
      let x = self.resolve_expr(ins[0])
      let sh = require_shape_from(self.shapes, out_name)
      let _ = axes
      self.body.write_string(
        "  let \{out_var} = \{x}.reshape(\{fmt_int_array(sh)})\n",
      )
    }
    @nn_ir.MatMul => {
      let a = self.resolve_expr(ins[0])
      let a_sh = require_shape_from(self.shapes, ins[0])
      if self.use_simd &&
        a_sh.length() == 2 &&
        self.graph.initializers.contains(ins[1]) {
        self.needs_gemm = true
        let wf = self.register_f32_weight(ins[1], PackedMatB(trans_b=false))
        self.body.write_string(
          "  let \{out_var} = @ops.matmul_prepacked_f32(\{a}, self.\{wf}, pool=\{pool_expr})\n",
        )
      } else {
        let b = self.resolve_expr(ins[1])
        self.body.write_string(
          "  let \{out_var} = @ops.matmul_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
    }
    @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
      let a = self.resolve_expr(ins[0])
      let c_expr = if ins.length() >= 3 && ins[2] != "" {
        "Some(\{self.resolve_expr(ins[2])})"
      } else {
        "None"
      }
      if self.use_simd && self.graph.initializers.contains(ins[1]) {
        self.needs_gemm = true
        let wf = self.register_f32_weight(ins[1], PackedMatB(trans_b~))
        self.body.write_string(
          "  let \{out_var} = @ops.gemm_prepacked_f32(\{a}, self.\{wf}, \{c_expr}, alpha=\{fmt_float_literal(alpha)}, beta=\{fmt_float_literal(beta)}, trans_a=\{trans_a}, pool=\{pool_expr})\n",
        )
      } else {
        let b = self.resolve_expr(ins[1])
        self.body.write_string(
          "  let \{out_var} = @ops.gemm_f32(\{a}, \{b}, \{c_expr}, alpha=\{fmt_float_literal(alpha)}, beta=\{fmt_float_literal(beta)}, trans_a=\{trans_a}, trans_b=\{trans_b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
    }
    @nn_ir.Conv(strides~, pads~, dilations~, group~, auto_pad~, activation~, ..) => {
      self.needs_nn_ir = true
      let x = self.resolve_expr(ins[0])
      let b_expr = if ins.length() >= 3 && ins[2] != "" {
        "Some(\{self.resolve_expr(ins[2])})"
      } else {
        "None"
      }
      if self.use_simd && self.graph.initializers.contains(ins[1]) {
        let wf = self.register_f32_weight(ins[1], PackedConv(group~))
        self.body.write_string(
          "  let \{out_var} = @ops.conv2d_prepacked_f32(\{x}, self.\{wf}, \{b_expr}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{fmt_int_array(dilations)}, \{fmt_auto_pad(auto_pad)}, activation=\{fmt_activation(activation)}, pool=\{pool_expr}, scratch_pool=Some(self.pool))\n",
        )
      } else {
        let w = self.resolve_expr(ins[1])
        self.body.write_string(
          "  let \{out_var} = @ops.conv2d_f32(\{x}, \{w}, \{b_expr}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{fmt_int_array(dilations)}, \{group}, \{fmt_auto_pad(auto_pad)}, use_simd=\{simd_str}, activation=\{fmt_activation(activation)}, pool=\{pool_expr}, scratch_pool=Some(self.pool))\n",
        )
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
      self.needs_nn_ir = true
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.maxpool_2d_f32(\{x}, \{fmt_int_array(kernel_shape)}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{fmt_int_array(dilations)}, \{ceil_mode}, \{fmt_auto_pad(auto_pad)}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.AveragePool(
      kernel_shape~,
      strides~,
      pads~,
      ceil_mode~,
      count_include_pad~,
      auto_pad~,
    ) => {
      self.needs_nn_ir = true
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.avgpool_2d_f32(\{x}, \{fmt_int_array(kernel_shape)}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{ceil_mode}, \{count_include_pad}, \{fmt_auto_pad(auto_pad)}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.GlobalAveragePool => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.global_avgpool_f32(\{x}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.BatchNormalization(epsilon~) => {
      let x = self.resolve_expr(ins[0])
      let s = self.resolve_expr(ins[1])
      let b = self.resolve_expr(ins[2])
      let m = self.resolve_expr(ins[3])
      let v = self.resolve_expr(ins[4])
      self.body.write_string(
        "  let \{out_var} = @ops.batch_norm_f32(\{x}, \{s}, \{b}, \{m}, \{v}, epsilon=\{fmt_float_literal(epsilon)}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.LayerNormalization(axis~, epsilon~) => {
      let x = self.resolve_expr(ins[0])
      let s = self.resolve_expr(ins[1])
      let b_expr = if ins.length() >= 3 && ins[2] != "" {
        "Some(\{self.resolve_expr(ins[2])})"
      } else {
        "None"
      }
      self.body.write_string(
        "  let \{out_var} = @ops.layer_norm_f32(\{x}, \{s}, \{b_expr}, axis=\{axis}, epsilon=\{fmt_float_literal(epsilon)}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Concat(axis~) => {
      let parts = StringBuilder()
      parts.write_char('[')
      for k = 0; k < ins.length(); k = k + 1 {
        if k > 0 {
          parts.write_string(", ")
        }
        parts.write_string(self.resolve_expr(ins[k]))
      }
      parts.write_char(']')
      self.body.write_string(
        "  let \{out_var} = @ops.concat_op(\{parts.to_string()}, \{axis}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.Cast(to~) => {
      let x = self.resolve_expr(ins[0])
      self.body.write_string(
        "  let \{out_var} = @ops.cast_op(\{x}, \{fmt_dtype(to)})\n",
      )
    }
    @nn_ir.ReduceMean(axes~, keepdims~, noop_with_empty_axes~) => {
      let x = self.resolve_expr(ins[0])
      let ax_expr = match axes {
        Some(a) => "Some(\{fmt_int_array(a)})"
        None => "None"
      }
      self.body.write_string(
        "  let \{out_var} = @ops.reduce_mean_f32(\{x}, \{ax_expr}, None, keepdims=\{keepdims}, noop_with_empty_axes=\{noop_with_empty_axes}, pool=\{pool_expr})\n",
      )
    }
    @nn_ir.ReduceSum(axes~, keepdims~, noop_with_empty_axes~) => {
      let x = self.resolve_expr(ins[0])
      let ax_expr = match axes {
        Some(a) => "Some(\{fmt_int_array(a)})"
        None => "None"
      }
      self.body.write_string(
        "  let \{out_var} = @ops.reduce_sum_f32(\{x}, \{ax_expr}, None, keepdims=\{keepdims}, noop_with_empty_axes=\{noop_with_empty_axes}, pool=\{pool_expr})\n",
      )
    }
    other =>
      raise UnsupportedOp(
        "AOT codegen does not yet support op '\{other.name()}'",
      )
  }
}
