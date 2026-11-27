///|
/// 从单元素 `Bool` 张量中读取标量布尔值（供 `If` 条件分支判断使用）。
fn read_scalar_bool(
  node_name : String,
  cond_t : @tensor.Tensor,
) -> Bool raise RuntimeError {
  let c = cond_t.contiguous() catch {
    e => raise TensorErr("node '\{node_name}' (If): \{e}")
  }
  if c.numel() != 1 {
    raise TensorErr(
      "node '\{node_name}' (If): condition must have 1 element, got \{c.numel()}",
    )
  }
  match c.data {
    @tensor.Bool(buf) => buf[c.offset]
    _ =>
      raise TensorErr(
        "node '\{node_name}' (If): condition must be Bool tensor, got \{c.dtype()}",
      )
  }
}

///|
/// 内置 CPU 统一算子分派入口：集中处理 `Split`、`LSTM`、`DynamicQuantizeLinear`、`SkipRmsNormalization`、`GroupQueryAttention` 与 `If` 多输出节点，
/// 其余单输出算子委托给 `eval_cpu_node_single`，彻底消除单/多输出重复分派代码。
fn eval_cpu_node_multi(
  node : @nn_ir.IrNode,
  ctx : EvalContext,
  use_simd : Bool,
  cache : CpuWeightCache?,
) -> Array[@tensor.Tensor] raise RuntimeError {
  match node.op {
    @nn_ir.Split(axis~, split~, num_outputs~) => {
      let n = node.name
      let ins = node.inputs
      try {
        let x = ctx.require_tensor(n, ins, 0)
        let split_t = ctx.optional_tensor(n, ins, 1)
        let n_out = match num_outputs {
          Some(k) => k
          None => node.outputs.length()
        }
        @ops.split_op(x, axis, split, split_t, n_out, pool=ctx.out_pool)
      } catch {
        MissingTensor(msg) => raise MissingTensor(msg)
        UnsupportedOp(msg) => raise UnsupportedOp(msg)
        e => raise TensorErr("node '\{n}' (Split): \{e}")
      }
    }
    @nn_ir.Lstm(hidden_size~, direction~, layout~, clip~) => {
      let n = node.name
      let ins = node.inputs
      try {
        let x = ctx.require_tensor(n, ins, 0)
        let w = ctx.require_tensor(n, ins, 1)
        let r = ctx.require_tensor(n, ins, 2)
        let b = ctx.optional_tensor(n, ins, 3)
        let seq_lens = ctx.optional_tensor(n, ins, 4)
        let init_h = ctx.optional_tensor(n, ins, 5)
        let init_c = ctx.optional_tensor(n, ins, 6)
        let peepholes = ctx.optional_tensor(n, ins, 7)
        @ops.lstm_f32(
          x,
          w,
          r,
          b,
          seq_lens,
          init_h,
          init_c,
          peepholes,
          hidden_size~,
          direction~,
          layout~,
          clip~,
        )
      } catch {
        MissingTensor(msg) => raise MissingTensor(msg)
        UnsupportedOp(msg) => raise UnsupportedOp(msg)
        e => raise TensorErr("node '\{n}' (LSTM): \{e}")
      }
    }
    @nn_ir.DynamicQuantizeLinear => {
      let n = node.name
      let ins = node.inputs
      try {
        let x = ctx.require_tensor(n, ins, 0)
        let (y, y_scale, y_zp) = @ops.dynamic_quantize_linear_op(x)
        [y, y_scale, y_zp]
      } catch {
        MissingTensor(msg) => raise MissingTensor(msg)
        UnsupportedOp(msg) => raise UnsupportedOp(msg)
        e => raise TensorErr("node '\{n}' (DynamicQuantizeLinear): \{e}")
      }
    }
    @nn_ir.SkipRmsNormalization(epsilon~) => {
      let n = node.name
      let ins = node.inputs
      try {
        let x = ctx.require_tensor(n, ins, 0)
        let skip = ctx.require_tensor(n, ins, 1)
        let scale = ctx.require_tensor(n, ins, 2)
        let bias = ctx.optional_tensor(n, ins, 3)
        let need_sum_out = node.outputs.length() >= 4 && node.outputs[3] != ""
        @ops.skip_rms_norm_f32(
          x,
          skip,
          scale,
          bias,
          epsilon~,
          need_sum_out~,
          pool=ctx.out_pool,
        )
      } catch {
        MissingTensor(msg) => raise MissingTensor(msg)
        UnsupportedOp(msg) => raise UnsupportedOp(msg)
        e => raise TensorErr("node '\{n}' (SkipRMSNormalization): \{e}")
      }
    }
    @nn_ir.GroupQueryAttention(
      num_heads~,
      kv_num_heads~,
      scale~,
      local_window_size~,
      softcap~,
      do_rotary~,
      rotary_interleaved~,
    ) => {
      let n = node.name
      let ins = node.inputs
      try {
        let q = ctx.require_tensor(n, ins, 0)
        let k = ctx.optional_tensor(n, ins, 1)
        let v = ctx.optional_tensor(n, ins, 2)
        let pk = ctx.optional_tensor(n, ins, 3)
        let pv = ctx.optional_tensor(n, ins, 4)
        let seqlens_k = ctx.require_tensor(n, ins, 5)
        let tot_seq = ctx.require_tensor(n, ins, 6)
        let cos_c = ctx.optional_tensor(n, ins, 7)
        let sin_c = ctx.optional_tensor(n, ins, 8)
        @ops.gqa_f32(
          q,
          k,
          v,
          pk,
          pv,
          seqlens_k,
          tot_seq,
          cos_c,
          sin_c,
          num_heads~,
          kv_num_heads~,
          scale~,
          local_window_size~,
          softcap~,
          do_rotary~,
          rotary_interleaved~,
          pool=ctx.out_pool,
        )
      } catch {
        MissingTensor(msg) => raise MissingTensor(msg)
        UnsupportedOp(msg) => raise UnsupportedOp(msg)
        e => raise TensorErr("node '\{n}' (GroupQueryAttention): \{e}")
      }
    }
    @nn_ir.If(then_branch~, else_branch~) => {
      let cond_t = ctx.require_tensor(node.name, node.inputs, 0)
      let cond = read_scalar_bool(node.name, cond_t)
      let branch = if cond { then_branch } else { else_branch }
      let sub_sess = Session::from_graph(
        branch,
        backend=if use_simd { Simd } else { Scalar },
      )
      sub_sess.run_subgraph(ctx.env)
    }
    _ => [eval_cpu_node_single(node, ctx, use_simd, cache)]
  }
}

///|
/// 内置 CPU 单输出分派包装：复用 `eval_cpu_node_multi` 并取第 0 个输出张量。
fn eval_cpu_node(
  node : @nn_ir.IrNode,
  ctx : EvalContext,
  use_simd : Bool,
  cache : CpuWeightCache?,
) -> @tensor.Tensor raise RuntimeError {
  let outs = eval_cpu_node_multi(node, ctx, use_simd, cache)
  if outs.length() == 0 {
    raise TensorErr(
      "node '\{node.name}' (\{node.op.name()}) produced 0 outputs",
    )
  }
  outs[0]
}
