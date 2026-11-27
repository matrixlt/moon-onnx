///|
/// 内置 CPU 单输出算子分派核心实现，由 `CpuScalarProvider` 与 `CpuSimdProvider` 共享。
fn eval_cpu_node_single(
  node : @nn_ir.IrNode,
  ctx : EvalContext,
  use_simd : Bool,
  cache : CpuWeightCache?,
) -> @tensor.Tensor raise RuntimeError {
  let n = node.name
  let ins = node.inputs
  let out_pool = ctx.out_pool
  let scratch_pool = ctx.scratch_pool
  try {
    match node.op {
      @nn_ir.Add => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.add_f32(a, b, use_simd~, pool=out_pool)
      }
      @nn_ir.Sub => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.sub_f32(a, b, use_simd~, pool=out_pool)
      }
      @nn_ir.Mul => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.mul_f32(a, b, use_simd~, pool=out_pool)
      }
      @nn_ir.Div => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.div_f32(a, b, use_simd~, pool=out_pool)
      }
      @nn_ir.Pow => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.pow_f32(a, b, pool=out_pool)
      }
      @nn_ir.Sqrt => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.sqrt_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.Exp => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.exp_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.Neg => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.neg_f32(x, pool=out_pool)
      }
      @nn_ir.Relu => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.relu_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.Clip(min~, max~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let min_val = match ctx.optional_tensor(n, ins, 1) {
          Some(t) => Some(t.get_f32_flat(0))
          None => min
        }
        let max_val = match ctx.optional_tensor(n, ins, 2) {
          Some(t) => Some(t.get_f32_flat(0))
          None => max
        }
        @ops.clip_f32(x, min_val, max_val, use_simd~, pool=out_pool)
      }
      @nn_ir.Sigmoid => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.sigmoid_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.Tanh => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.tanh_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.Gelu => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.gelu_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.GeluExact => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.gelu_exact_f32(x, pool=out_pool)
      }
      @nn_ir.HardSwish => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.hard_swish_f32(x, use_simd~, pool=out_pool)
      }
      @nn_ir.LeakyRelu(alpha~) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.leaky_relu_f32(x, alpha~, pool=out_pool)
      }
      @nn_ir.Identity => ctx.require_tensor(n, ins, 0)
      @nn_ir.Softmax(axis~) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.softmax_f32(x, axis, pool=out_pool)
      }
      @nn_ir.Flatten(axis~) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.flatten_op(x, axis~)
      }
      @nn_ir.Reshape(allowzero~) => {
        let data = ctx.require_tensor(n, ins, 0)
        let shape_t = ctx.require_tensor(n, ins, 1)
        @ops.reshape_op(data, shape_t, allowzero~)
      }
      @nn_ir.Transpose(perm~) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.transpose_op(x, perm)
      }
      @nn_ir.Squeeze(axes~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let axes_t = ctx.optional_tensor(n, ins, 1)
        @ops.squeeze_op(x, axes, axes_t)
      }
      @nn_ir.Unsqueeze(axes~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let axes_t = ctx.optional_tensor(n, ins, 1)
        @ops.unsqueeze_op(x, axes, axes_t)
      }
      @nn_ir.Constant(value~) => value
      @nn_ir.MatMul => {
        let a = ctx.require_tensor(n, ins, 0)
        if use_simd && a.rank() >= 2 && ins.length() >= 2 {
          match cache {
            Some(c) =>
              match c.packed_weights.get(ins[1]) {
                Some(pb) =>
                  return @ops.matmul_prepacked_f32(a, pb, pool=out_pool)
                None => ()
              }
            None => ()
          }
        }
        let b = ctx.require_tensor(n, ins, 1)
        @ops.matmul_f32(a, b, use_simd~, pool=out_pool)
      }
      @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
        let a = ctx.require_tensor(n, ins, 0)
        let c = ctx.optional_tensor(n, ins, 2)
        if use_simd && ins.length() >= 2 {
          let key = if trans_b { "\{ins[1]}#T" } else { ins[1] }
          match cache {
            Some(cc) =>
              match cc.packed_weights.get(key) {
                Some(pb) =>
                  return @ops.gemm_prepacked_f32(
                    a,
                    pb,
                    c,
                    alpha~,
                    beta~,
                    trans_a~,
                    pool=out_pool,
                  )
                None => ()
              }
            None => ()
          }
        }
        let b = ctx.require_tensor(n, ins, 1)
        @ops.gemm_f32(
          a,
          b,
          c,
          alpha~,
          beta~,
          trans_a~,
          trans_b~,
          use_simd~,
          pool=out_pool,
        )
      }
      @nn_ir.Conv(
        strides~,
        pads~,
        dilations~,
        group~,
        auto_pad~,
        activation~,
        ..
      ) => {
        let x = ctx.require_tensor(n, ins, 0)
        let b = ctx.optional_tensor(n, ins, 2)
        if use_simd && ins.length() >= 2 {
          let key = "\{ins[1]}#g\{group}"
          match cache {
            Some(cc) =>
              match cc.packed_conv_weights.get(key) {
                Some(pw) =>
                  return @ops.conv2d_prepacked_f32(
                    x,
                    pw,
                    b,
                    strides,
                    pads,
                    dilations,
                    auto_pad,
                    activation~,
                    pool=out_pool,
                    scratch_pool=Some(scratch_pool),
                  )
                None => ()
              }
            None => ()
          }
        }
        let w = ctx.require_tensor(n, ins, 1)
        @ops.conv2d_f32(
          x,
          w,
          b,
          strides,
          pads,
          dilations,
          group,
          auto_pad,
          use_simd~,
          activation~,
          pool=out_pool,
          scratch_pool=Some(scratch_pool),
        )
      }
      @nn_ir.MaxPool(
        kernel_shape~,
        strides~,
        pads~,
        dilations~,
        ceil_mode~,
        auto_pad~,
      ) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.maxpool_2d_f32(
          x,
          kernel_shape,
          strides,
          pads,
          dilations,
          ceil_mode,
          auto_pad,
          pool=out_pool,
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
        let x = ctx.require_tensor(n, ins, 0)
        @ops.avgpool_2d_f32(
          x,
          kernel_shape,
          strides,
          pads,
          ceil_mode,
          count_include_pad,
          auto_pad,
          pool=out_pool,
        )
      }
      @nn_ir.GlobalAveragePool => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.global_avgpool_f32(x, pool=out_pool)
      }
      @nn_ir.BatchNormalization(epsilon~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let scale = ctx.require_tensor(n, ins, 1)
        let bias = ctx.require_tensor(n, ins, 2)
        let mean = ctx.require_tensor(n, ins, 3)
        let variance = ctx.require_tensor(n, ins, 4)
        @ops.batch_norm_f32(
          x,
          scale,
          bias,
          mean,
          variance,
          epsilon~,
          pool=out_pool,
        )
      }
      @nn_ir.LayerNormalization(axis~, epsilon~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let scale = ctx.require_tensor(n, ins, 1)
        let bias = ctx.optional_tensor(n, ins, 2)
        @ops.layer_norm_f32(x, scale, bias, axis~, epsilon~, pool=out_pool)
      }
      @nn_ir.Concat(axis~) => {
        let tensors : Array[@tensor.Tensor] = []
        for idx = 0; idx < ins.length(); idx = idx + 1 {
          tensors.push(ctx.require_tensor(n, ins, idx))
        }
        @ops.concat_op(tensors, axis, pool=out_pool)
      }
      @nn_ir.Split(..)
      | @nn_ir.Lstm(..)
      | @nn_ir.DynamicQuantizeLinear
      | @nn_ir.If(..) =>
        raise TensorErr(
          "internal error: multi-output op '\{node.op.name()}' must be dispatched via eval_cpu_node_multi",
        )
      @nn_ir.Slice(starts~, ends~, axes~, steps~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let resolved_starts : Array[Int64] = match
          ctx.optional_tensor(n, ins, 1) {
          Some(t) => {
            let b = t.to_i64_buffer()
            Array::makei(b.length(), fn(k) { b[k] })
          }
          None =>
            match starts {
              Some(s) => Array::makei(s.length(), fn(k) { s[k].to_int64() })
              None =>
                raise MissingTensor("node '\{n}' (Slice) missing starts input")
            }
        }
        let resolved_ends : Array[Int64] = match
          ctx.optional_tensor(n, ins, 2) {
          Some(t) => {
            let b = t.to_i64_buffer()
            Array::makei(b.length(), fn(k) { b[k] })
          }
          None =>
            match ends {
              Some(e) => Array::makei(e.length(), fn(k) { e[k].to_int64() })
              None =>
                raise MissingTensor("node '\{n}' (Slice) missing ends input")
            }
        }
        let resolved_axes : Array[Int32]? = match
          ctx.optional_tensor(n, ins, 3) {
          Some(t) => {
            let b = t.to_i64_buffer()
            Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
          }
          None => axes
        }
        let resolved_steps : Array[Int32]? = match
          ctx.optional_tensor(n, ins, 4) {
          Some(t) => {
            let b = t.to_i64_buffer()
            Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
          }
          None => steps
        }
        @ops.slice_op(
          x,
          resolved_starts,
          resolved_ends,
          resolved_axes,
          resolved_steps,
          pool=out_pool,
        )
      }
      @nn_ir.Pad(mode~, pads~, constant_value~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let resolved_pads : Array[Int32] = match
          ctx.optional_tensor(n, ins, 1) {
          Some(t) => {
            let b = t.to_i64_buffer()
            Array::makei(b.length(), fn(k) { b[k].to_int() })
          }
          None =>
            match pads {
              Some(p) => p
              None =>
                raise MissingTensor("node '\{n}' (Pad) missing pads input")
            }
        }
        let resolved_val : Float = match ctx.optional_tensor(n, ins, 2) {
          Some(t) => t.get_f32_flat(0)
          None =>
            match constant_value {
              Some(v) => v
              None => (0.0 : Float)
            }
        }
        let resolved_axes : Array[Int32]? = match
          ctx.optional_tensor(n, ins, 3) {
          Some(t) => {
            let b = t.to_i64_buffer()
            Some(Array::makei(b.length(), fn(k) { b[k].to_int() }))
          }
          None => None
        }
        @ops.pad_f32(
          x,
          resolved_pads,
          mode,
          resolved_val,
          resolved_axes,
          pool=out_pool,
        )
      }
      @nn_ir.Gather(axis~) => {
        let data = ctx.require_tensor(n, ins, 0)
        let indices = ctx.require_tensor(n, ins, 1)
        @ops.gather_op(data, indices, axis, pool=out_pool)
      }
      @nn_ir.Cast(to~) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.cast_op(x, to)
      }
      @nn_ir.ReduceMean(axes~, keepdims~, noop_with_empty_axes~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let axes_t = ctx.optional_tensor(n, ins, 1)
        @ops.reduce_mean_f32(
          x,
          axes,
          axes_t,
          keepdims~,
          noop_with_empty_axes~,
          pool=out_pool,
        )
      }
      @nn_ir.ReduceSum(axes~, keepdims~, noop_with_empty_axes~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let axes_t = ctx.optional_tensor(n, ins, 1)
        @ops.reduce_sum_f32(
          x,
          axes,
          axes_t,
          keepdims~,
          noop_with_empty_axes~,
          pool=out_pool,
        )
      }
      @nn_ir.Shape(start~, end~) => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.shape_op(x, start~, end~)
      }
      @nn_ir.QuantizeLinear(axis~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let scale = ctx.require_tensor(n, ins, 1)
        let zp = ctx.optional_tensor(n, ins, 2)
        @ops.quantize_linear_op(x, scale, zp, axis~)
      }
      @nn_ir.DequantizeLinear(axis~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let scale = ctx.require_tensor(n, ins, 1)
        let zp = ctx.optional_tensor(n, ins, 2)
        @ops.dequantize_linear_op(x, scale, zp, axis~, pool=out_pool)
      }
      @nn_ir.QLinearMatMul => {
        let a = ctx.require_tensor(n, ins, 0)
        let a_scale = ctx.require_tensor(n, ins, 1)
        let a_zp = ctx.require_tensor(n, ins, 2)
        let b = ctx.require_tensor(n, ins, 3)
        let b_scale = ctx.require_tensor(n, ins, 4)
        let b_zp = ctx.require_tensor(n, ins, 5)
        let y_scale = ctx.require_tensor(n, ins, 6)
        let y_zp = ctx.require_tensor(n, ins, 7)
        @ops.qlinear_matmul_op(
          a,
          a_scale,
          a_zp,
          b,
          b_scale,
          b_zp,
          y_scale,
          y_zp,
          use_simd~,
        )
      }
      @nn_ir.Resize(mode~, coordinate_transformation_mode~, nearest_mode~) => {
        let x = ctx.require_tensor(n, ins, 0)
        // Opset 11+ 输入顺序：X, roi, scales, sizes
        let scales = ctx.optional_tensor(n, ins, 2)
        let sizes = ctx.optional_tensor(n, ins, 3)
        @ops.resize_f32(
          x,
          scales,
          sizes,
          mode~,
          coordinate_transformation_mode~,
          nearest_mode~,
          pool=out_pool,
        )
      }
      @nn_ir.NonMaxSuppression(center_point_box~) => {
        let boxes = ctx.require_tensor(n, ins, 0)
        let scores = ctx.require_tensor(n, ins, 1)
        let max_boxes = ctx.optional_tensor(n, ins, 2)
        let iou_th = ctx.optional_tensor(n, ins, 3)
        let score_th = ctx.optional_tensor(n, ins, 4)
        @ops.nms_op(
          boxes,
          scores,
          max_boxes,
          iou_th,
          score_th,
          center_point_box~,
        )
      }
      @nn_ir.Equal => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.equal_op(a, b)
      }
      @nn_ir.Less => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.less_op(a, b)
      }
      @nn_ir.Greater => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.greater_op(a, b)
      }
      @nn_ir.LessOrEqual => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.less_or_equal_op(a, b)
      }
      @nn_ir.GreaterOrEqual => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.greater_or_equal_op(a, b)
      }
      @nn_ir.Not => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.not_op(x)
      }
      @nn_ir.And => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        @ops.and_op(a, b)
      }
      @nn_ir.Where => {
        let cond = ctx.require_tensor(n, ins, 0)
        let x = ctx.require_tensor(n, ins, 1)
        let y = ctx.require_tensor(n, ins, 2)
        @ops.where_op(cond, x, y, pool=out_pool)
      }
      @nn_ir.Erf => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.erf_f32(x, pool=out_pool)
      }
      @nn_ir.Range => {
        let start = ctx.require_tensor(n, ins, 0)
        let limit = ctx.require_tensor(n, ins, 1)
        let delta = ctx.require_tensor(n, ins, 2)
        @ops.range_op(start, limit, delta)
      }
      @nn_ir.Expand => {
        let x = ctx.require_tensor(n, ins, 0)
        let shape_t = ctx.require_tensor(n, ins, 1)
        @ops.expand_op(x, shape_t, pool=out_pool)
      }
      @nn_ir.Tile => {
        let x = ctx.require_tensor(n, ins, 0)
        let reps_t = ctx.require_tensor(n, ins, 1)
        @ops.tile_op(x, reps_t, pool=out_pool)
      }
      @nn_ir.ConstantOfShape(value~) => {
        let shape_t = ctx.require_tensor(n, ins, 0)
        @ops.constant_of_shape_op(shape_t, value)
      }
      @nn_ir.MatMulInteger => {
        let a = ctx.require_tensor(n, ins, 0)
        let a_zp = ctx.optional_tensor(n, ins, 2)
        let b_zp = ctx.optional_tensor(n, ins, 3)
        match cache {
          Some(c) if use_simd && ins.length() >= 2 =>
            match c.packed_i8_weights.get(ins[1]) {
              Some(packed_b) if a.dtype() == @tensor.UInt8 =>
                @ops.matmul_integer_prepacked_op(a, packed_b, a_zp, b_zp)
              _ => {
                let b = ctx.require_tensor(n, ins, 1)
                @ops.matmul_integer_op(a, b, a_zp, b_zp, use_simd~)
              }
            }
          _ => {
            let b = ctx.require_tensor(n, ins, 1)
            @ops.matmul_integer_op(a, b, a_zp, b_zp, use_simd~)
          }
        }
      }
      @nn_ir.Silu => {
        let x = ctx.require_tensor(n, ins, 0)
        @ops.silu_f32(x, pool=out_pool)
      }
      @nn_ir.RmsNormalization(axis~, epsilon~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let scale = ctx.require_tensor(n, ins, 1)
        @ops.rms_norm_f32(x, scale, axis~, epsilon~, pool=out_pool)
      }
      @nn_ir.RotaryEmbedding(interleaved~, num_heads~, rotary_embedding_dim~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let pos_ids = ctx.require_tensor(n, ins, 1)
        let cos_c = ctx.require_tensor(n, ins, 2)
        let sin_c = ctx.require_tensor(n, ins, 3)
        @ops.rotary_embedding_f32(
          x,
          pos_ids,
          cos_c,
          sin_c,
          interleaved~,
          num_heads~,
          rotary_embedding_dim~,
          pool=out_pool,
        )
      }
      @nn_ir.MatMulNBits(k~, n=out_n, bits~, block_size~) => {
        let a = ctx.require_tensor(n, ins, 0)
        let b = ctx.require_tensor(n, ins, 1)
        let scales = ctx.require_tensor(n, ins, 2)
        let zp = ctx.optional_tensor(n, ins, 3)
        let bias = ctx.optional_tensor(n, ins, 5)
        @ops.matmul_nbits_op(
          a,
          b,
          scales,
          zp,
          bias,
          k~,
          n=out_n,
          bits~,
          block_size~,
          use_simd~,
          pool=out_pool,
        )
      }
      @nn_ir.SkipRmsNormalization(..) | @nn_ir.GroupQueryAttention(..) =>
        raise TensorErr(
          "internal error: multi-output op '\{node.op.name()}' must be dispatched via eval_cpu_node_multi",
        )
    }
  } catch {
    MissingTensor(msg) => raise MissingTensor(msg)
    UnsupportedOp(msg) => raise UnsupportedOp(msg)
    e => raise TensorErr("node '\{n}' (\{node.op.name()}): \{e}")
  }
}
