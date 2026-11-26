///|
/// 单节点执行上下文，向 `ExecutionProvider` 传递纯运行期张量环境与 `BufferPool`（不泄露具体后端私有权重格式）。
pub struct EvalContext {
  env : Map[String, @tensor.Tensor]
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
  /// 指示后端是否采用 SIMD 加速路径（默认 `false`）
  fn is_simd(Self) -> Bool = _
  /// 会话初始化期图准备钩子：由后端私有完成静态权重预打包或设备上传（默认空操作）
  fn prepare_graph(Self, @nn_ir.IrGraph) -> Unit = _
  /// 查询后端已预打包的静态权重总数（供诊断与测试使用，默认返回 0）
  fn packed_weight_count(Self) -> Int32 = _
  /// 查询后端已预打包的静态卷积权重数（供诊断与测试使用，默认返回 0）
  fn packed_conv_weight_count(Self) -> Int32 = _
  /// 执行单个强类型 IR 节点并返回主输出张量
  fn eval_node(Self, @nn_ir.IrNode, EvalContext) -> @tensor.Tensor raise RuntimeError
  /// 执行可能产生多个输出张量的 IR 节点（例如 `Split`、`LSTM`、`DynamicQuantizeLinear`、`If`），默认退化为单输出包装
  fn eval_node_multi(Self, @nn_ir.IrNode, EvalContext) -> Array[@tensor.Tensor] raise RuntimeError = _
}

///|
impl ExecutionProvider with fn is_simd(_self) {
  false
}

///|
impl ExecutionProvider with fn prepare_graph(_self, _graph) {
  ()
}

///|
impl ExecutionProvider with fn packed_weight_count(_self) {
  0
}

///|
impl ExecutionProvider with fn packed_conv_weight_count(_self) {
  0
}

///|
/// `ExecutionProvider::eval_node_multi` 默认实现：直接复用单输出 `eval_node`。
impl ExecutionProvider with fn eval_node_multi(self, node, ctx) {
  [self.eval_node(node, ctx)]
}

///|
/// `CpuSimdProvider` 私有静态权重预打包缓存（不暴露给 `Session` 或 `EvalContext`）。
priv struct CpuWeightCache {
  packed_weights : Map[String, @gemm.PackedMatrixB]
  packed_conv_weights : Map[String, @ops.PackedConvWeight]
  packed_i8_weights : Map[String, @quant.PackedInt8MatrixB]
}

///|
/// 扫描计算图并将所有静态 2D/4D Float32 权重与 2D Int8 量化权重离线预打包写入 `CpuWeightCache`。
fn CpuWeightCache::prepare(
  self : CpuWeightCache,
  graph : @nn_ir.IrGraph,
) -> Unit {
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    if node.inputs.length() < 2 {
      continue
    }
    let w_name = node.inputs[1]
    match node.op {
      @nn_ir.MatMul =>
        if !self.packed_weights.contains(w_name) {
          match graph.initializers.get(w_name) {
            Some(w_t) if w_t.rank() == 2 && w_t.dtype() == @tensor.Float32 => {
              let ws = w_t.shape()
              try {
                let w_buf = w_t.to_f32_buffer()
                self.packed_weights[w_name] = @gemm.pack_matrix_b_f32(
                  w_buf,
                  0,
                  ws[0],
                  ws[1],
                )
              } catch {
                _ => ()
              }
            }
            _ => ()
          }
        }
      @nn_ir.Gemm(trans_b~, ..) => {
        let key = if trans_b { "\{w_name}#T" } else { w_name }
        if !self.packed_weights.contains(key) {
          match graph.initializers.get(w_name) {
            Some(w_t) if w_t.rank() == 2 && w_t.dtype() == @tensor.Float32 => {
              let ws = w_t.shape()
              try {
                let w_buf = w_t.to_f32_buffer()
                self.packed_weights[key] = if trans_b {
                  @gemm.pack_matrix_b_transposed_f32(w_buf, 0, ws[0], ws[1])
                } else {
                  @gemm.pack_matrix_b_f32(w_buf, 0, ws[0], ws[1])
                }
              } catch {
                _ => ()
              }
            }
            _ => ()
          }
        }
      }
      @nn_ir.Conv(group~, ..) => {
        let key = "\{w_name}#g\{group}"
        if !self.packed_conv_weights.contains(key) {
          match graph.initializers.get(w_name) {
            Some(w_t) if (w_t.rank() == 4 || w_t.rank() == 3) &&
              w_t.dtype() == @tensor.Float32 =>
              try {
                self.packed_conv_weights[key] = @ops.pack_conv2d_weight_f32(
                  w_t, group,
                )
              } catch {
                _ => ()
              }
            _ => ()
          }
        }
      }
      @nn_ir.MatMulInteger =>
        if !self.packed_i8_weights.contains(w_name) {
          match graph.initializers.get(w_name) {
            Some(w_t) if w_t.rank() == 2 && w_t.dtype() == @tensor.Int8 => {
              let ws = w_t.shape()
              try {
                self.packed_i8_weights[w_name] = @quant.pack_int8_matrix_b(
                  w_t.to_i8_buffer(),
                  ws[0],
                  ws[1],
                )
              } catch {
                _ => ()
              }
            }
            _ => ()
          }
        }
      _ => ()
    }
  }
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
/// 内置基于 MoonBit 原生 `V128` SIMD 与私有静态权重预打包缓存的高性能 CPU/Wasm 执行后端。
pub struct CpuSimdProvider {
  priv cache : CpuWeightCache
}

///|
/// 创建 `V128` SIMD 执行后端实例（私有权重缓存在 `Session::with_provider` 调用 `prepare_graph` 时填充）。
pub fn CpuSimdProvider::new() -> CpuSimdProvider {
  {
    cache: {
      packed_weights: Map([]),
      packed_conv_weights: Map([]),
      packed_i8_weights: Map([]),
    },
  }
}

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

///|
pub impl ExecutionProvider for CpuScalarProvider with fn name(_self) {
  "scalar"
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn eval_node(
  _self,
  node,
  ctx,
) {
  eval_cpu_node(node, ctx, false, None)
}

///|
pub impl ExecutionProvider for CpuScalarProvider with fn eval_node_multi(
  _self,
  node,
  ctx,
) {
  eval_cpu_node_multi(node, ctx, false, None)
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn name(_self) {
  "simd-v128"
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn is_simd(_self) {
  true
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn prepare_graph(
  self,
  graph,
) {
  self.cache.prepare(graph)
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn packed_weight_count(self) {
  let mut count = 0
  self.cache.packed_weights.each(fn(_, _) { count = count + 1 })
  self.cache.packed_conv_weights.each(fn(_, _) { count = count + 1 })
  self.cache.packed_i8_weights.each(fn(_, _) { count = count + 1 })
  count
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn packed_conv_weight_count(
  self,
) {
  let mut count = 0
  self.cache.packed_conv_weights.each(fn(_, _) { count = count + 1 })
  count
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn eval_node(
  self,
  node,
  ctx,
) {
  eval_cpu_node(node, ctx, true, Some(self.cache))
}

///|
pub impl ExecutionProvider for CpuSimdProvider with fn eval_node_multi(
  self,
  node,
  ctx,
) {
  eval_cpu_node_multi(node, ctx, true, Some(self.cache))
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
