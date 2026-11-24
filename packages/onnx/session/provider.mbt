///|
/// 单节点执行上下文，向 `ExecutionProvider` 传递运行期张量环境、预打包权重缓存与 `BufferPool`。
pub struct EvalContext {
  env : Map[String, @tensor.Tensor]
  packed_weights : Map[String, @gemm.PackedMatrixB]
  packed_conv_weights : Map[String, @ops.PackedConvWeight]
  scratch_pool : @tensor.BufferPool
  out_pool : @tensor.BufferPool?
}

///|
/// 从执行上下文中读取节点的必选输入张量（缺失时返回 `MissingTensor`）。
pub fn EvalContext::require_tensor(
  self : EvalContext,
  node_name : String,
  inputs : Array[String],
  idx : Int32,
) -> @tensor.Tensor raise RuntimeError {
  if idx >= inputs.length() || inputs[idx] == "" {
    raise MissingTensor(
      "node '\{node_name}' missing required input at index \{idx}",
    )
  }
  let name = inputs[idx]
  match self.env.get(name) {
    Some(t) => t
    None =>
      raise MissingTensor(
        "node '\{node_name}' could not find tensor '\{name}' in runtime env",
      )
  }
}

///|
/// 从执行上下文中读取节点的可选输入张量（空字符串或下标越界时返回 `None`）。
pub fn EvalContext::optional_tensor(
  self : EvalContext,
  node_name : String,
  inputs : Array[String],
  idx : Int32,
) -> @tensor.Tensor? raise RuntimeError {
  if idx >= inputs.length() || inputs[idx] == "" {
    return None
  }
  let name = inputs[idx]
  match self.env.get(name) {
    Some(t) => Some(t)
    None =>
      raise MissingTensor(
        "node '\{node_name}' could not find optional tensor '\{name}'",
      )
  }
}

///|
/// 可插拔推理执行后端接口（参考 ONNX Runtime `ExecutionProvider`）。
/// 允许用户在不修改 `Session` 调度与静态生命周期 `BufferPool` 复用逻辑的前提下，
/// 接入自定义算子加速后端（如 WebGPU Compute Shader、Native BLAS 或专用量化内核）。
pub(open) trait ExecutionProvider {
  /// 返回后端唯一标识名称（如 `"scalar"`、`"simd-v128"`、`"webgpu"`）
  fn name(Self) -> String
  /// 指示会话初始化期是否需要预打包静态 2D/4D Float32 权重
  fn uses_prepacked_weights(Self) -> Bool
  /// 执行单个强类型 IR 节点并返回主输出张量
  fn eval_node(Self, @nn_ir.IrNode, EvalContext) -> @tensor.Tensor raise RuntimeError
  /// 执行可能产生多个输出张量的 IR 节点（例如 `Split`），默认退化为单输出包装
  fn eval_node_multi(Self, @nn_ir.IrNode, EvalContext) -> Array[@tensor.Tensor] raise RuntimeError = _
}

///|
/// `ExecutionProvider::eval_node_multi` 默认实现：直接复用单输出 `eval_node`。
impl ExecutionProvider with fn eval_node_multi(self, node, ctx) {
  [self.eval_node(node, ctx)]
}

///|
/// 内置纯 MoonBit 标量参考执行后端（100% 跨四端，适合作为数值验证黄金基准）。
pub struct CpuScalarProvider {}

///|
/// 创建标量 CPU 执行后端实例。
pub fn CpuScalarProvider::new() -> CpuScalarProvider {
  CpuScalarProvider::{ }
}

///|
/// 内置基于 MoonBit 原生 `V128` SIMD 与静态权重预打包的高性能 CPU/Wasm 执行后端。
pub struct CpuSimdProvider {}

///|
/// 创建 `V128` SIMD 执行后端实例。
pub fn CpuSimdProvider::new() -> CpuSimdProvider {
  CpuSimdProvider::{ }
}

///|
/// 内置 CPU 算子分派核心实现，由 `CpuScalarProvider` 与 `CpuSimdProvider` 共享。
fn eval_cpu_node(
  node : @nn_ir.IrNode,
  ctx : EvalContext,
  use_simd : Bool,
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
        if use_simd && a.rank() == 2 && ins.length() >= 2 {
          match ctx.packed_weights.get(ins[1]) {
            Some(pb) => return @ops.matmul_prepacked_f32(a, pb, pool=out_pool)
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
          match ctx.packed_weights.get(key) {
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
          match ctx.packed_conv_weights.get(key) {
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
      @nn_ir.Split(axis~, split~, num_outputs~) => {
        let x = ctx.require_tensor(n, ins, 0)
        let split_t = ctx.optional_tensor(n, ins, 1)
        let n_out = match num_outputs {
          Some(k) => k
          None => node.outputs.length()
        }
        let parts = @ops.split_op(x, axis, split, split_t, n_out, pool=out_pool)
        if parts.length() == 0 {
          raise TensorErr("node '\{n}' (Split) produced 0 outputs")
        }
        parts[0]
      }
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
    }
  } catch {
    MissingTensor(msg) => raise MissingTensor(msg)
    UnsupportedOp(msg) => raise UnsupportedOp(msg)
    e => raise TensorErr("node '\{n}' (\{node.op.name()}): \{e}")
  }
}

///|
/// 内置 CPU 多输出节点分派入口（针对 `Split` 返回全部子张量，其余节点复用 `eval_cpu_node`）。
fn eval_cpu_node_multi(
  node : @nn_ir.IrNode,
  ctx : EvalContext,
  use_simd : Bool,
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
    _ => [eval_cpu_node(node, ctx, use_simd)]
  }
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn name(_self) {
  "scalar"
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn uses_prepacked_weights(
  _self,
) {
  false
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn eval_node(
  _self,
  node,
  ctx,
) {
  eval_cpu_node(node, ctx, false)
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn eval_node_multi(
  _self,
  node,
  ctx,
) {
  eval_cpu_node_multi(node, ctx, false)
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn name(_self) {
  "simd-v128"
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn uses_prepacked_weights(
  _self,
) {
  true
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn eval_node(
  _self,
  node,
  ctx,
) {
  eval_cpu_node(node, ctx, true)
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn eval_node_multi(
  _self,
  node,
  ctx,
) {
  eval_cpu_node_multi(node, ctx, true)
}

///|
/// 将 `BackendKind` 枚举转换为对应的默认内置 `ExecutionProvider` 对象。
pub fn default_provider_for_backend(
  backend : BackendKind,
) -> &ExecutionProvider {
  match backend {
    Scalar => (CpuScalarProvider::new() : &ExecutionProvider)
    Simd => (CpuSimdProvider::new() : &ExecutionProvider)
  }
}
