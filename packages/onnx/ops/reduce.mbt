///|
/// ONNX `GlobalAveragePool` 算子：对空间维度 `[2..rank-1]` 全局求均值，输出形状 `[N, C, 1, ..., 1]`。
/// 是 ResNet、MobileNetV2/V3 等现代视觉主干分类头的核心池化层。
pub fn global_avgpool_f32(
  x : @tensor.Tensor,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  if rank < 2 {
    raise @tensor.InvalidShape(
      "global_avgpool_f32 requires rank >= 2, got \{rank}",
    )
  }
  let n = shape[0]
  let c = shape[1]
  let mut spatial = 1
  for d = 2; d < rank; d = d + 1 {
    spatial = spatial * shape[d]
  }
  if spatial <= 0 {
    raise @tensor.InvalidShape(
      "global_avgpool_f32: spatial size must be positive",
    )
  }
  let out_shape = Array::make(rank, 1)
  out_shape[0] = n
  out_shape[1] = c

  let src = x.to_f32_buffer()
  let nc = n * c
  let out = alloc_f32(pool, nc)
  @nn.gap_f32(src, 0, out, 0, nc, spatial)
  @tensor.Tensor::from_f32(out_shape, out)
}

///|
/// 内部通用归约内核（同时服务于 `ReduceMean` 与 `ReduceSum`，使用 Double 精度累加避免大张量漂移）。
fn reduce_f32_impl(
  x : @tensor.Tensor,
  axes_in : Array[Int32]?,
  keepdims : Bool,
  noop_with_empty_axes : Bool,
  is_mean : Bool,
  pool : @tensor.BufferPool?,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  let is_reduced = Array::make(rank, false)

  match axes_in {
    Some(axes) =>
      if axes.length() == 0 {
        if noop_with_empty_axes {
          let src = x.to_f32_buffer()
          let out = alloc_f32(pool, src.length())
          for i = 0; i < src.length(); i = i + 1 {
            out[i] = src[i]
          }
          return @tensor.Tensor::from_f32(shape, out)
        } else {
          for d = 0; d < rank; d = d + 1 {
            is_reduced[d] = true
          }
        }
      } else {
        for i = 0; i < axes.length(); i = i + 1 {
          let ax = @tensor.normalize_axis(axes[i], rank)
          is_reduced[ax] = true
        }
      }
    None =>
      if noop_with_empty_axes {
        let src = x.to_f32_buffer()
        let out = alloc_f32(pool, src.length())
        for i = 0; i < src.length(); i = i + 1 {
          out[i] = src[i]
        }
        return @tensor.Tensor::from_f32(shape, out)
      } else {
        for d = 0; d < rank; d = d + 1 {
          is_reduced[d] = true
        }
      }
  }

  let out_shape : Array[Int32] = []
  let kept_dims : Array[Int32] = []
  let mut reduce_count = 1
  for d = 0; d < rank; d = d + 1 {
    if is_reduced[d] {
      reduce_count = reduce_count * shape[d]
      if keepdims {
        out_shape.push(1)
      }
      kept_dims.push(1)
    } else {
      out_shape.push(shape[d])
      kept_dims.push(shape[d])
    }
  }

  let out_total = @tensor.num_elements(out_shape)
  let out_strides = @tensor.compute_strides(kept_dims)
  let acc = FixedArray::make(out_total, 0.0)
  let src = x.to_f32_buffer()
  let in_total = src.length()

  for linear = 0; linear < in_total; linear = linear + 1 {
    let mut rem = linear
    let mut out_idx = 0
    for d = rank - 1; d >= 0; d = d - 1 {
      let coord = rem % shape[d]
      rem = rem / shape[d]
      if !is_reduced[d] {
        out_idx = out_idx + coord * out_strides[d]
      }
    }
    acc[out_idx] = acc[out_idx] + src[linear].to_double()
  }

  let out = alloc_f32(pool, out_total)
  if is_mean {
    let inv = 1.0 / reduce_count.to_double()
    for i = 0; i < out_total; i = i + 1 {
      out[i] = Float::from_double(acc[i] * inv)
    }
  } else {
    for i = 0; i < out_total; i = i + 1 {
      out[i] = Float::from_double(acc[i])
    }
  }
  @tensor.Tensor::from_f32(out_shape, out)
}

///|
/// ONNX `ReduceMean` 算子（兼容 Opset 1~18 属性与输入张量形式的 `axes`）。
pub fn reduce_mean_f32(
  x : @tensor.Tensor,
  attr_axes : Array[Int32]?,
  axes_tensor : @tensor.Tensor?,
  keepdims~ : Bool = true,
  noop_with_empty_axes~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let resolved = match axes_tensor {
    Some(t) => Some(tensor_to_int_array(t))
    None => attr_axes
  }
  reduce_f32_impl(x, resolved, keepdims, noop_with_empty_axes, true, pool)
}

///|
/// ONNX `ReduceSum` 算子（兼容 Opset 1~18 属性与输入张量形式的 `axes`，支持 `Float32` / `Int64` / `Int32`）。
pub fn reduce_sum_f32(
  x : @tensor.Tensor,
  attr_axes : Array[Int32]?,
  axes_tensor : @tensor.Tensor?,
  keepdims~ : Bool = true,
  noop_with_empty_axes~ : Bool = false,
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let resolved = match axes_tensor {
    Some(t) => Some(tensor_to_int_array(t))
    None => attr_axes
  }
  if x.dtype() == @tensor.Int64 || x.dtype() == @tensor.Int32 {
    let shape = x.shape()
    let rank = shape.length()
    let is_reduced = Array::make(rank, false)
    match resolved {
      Some(axes) =>
        if axes.length() == 0 {
          if noop_with_empty_axes {
            return x.contiguous()
          }
          for d = 0; d < rank; d = d + 1 {
            is_reduced[d] = true
          }
        } else {
          for i = 0; i < axes.length(); i = i + 1 {
            is_reduced[@tensor.normalize_axis(axes[i], rank)] = true
          }
        }
      None =>
        if noop_with_empty_axes {
          return x.contiguous()
        } else {
          for d = 0; d < rank; d = d + 1 {
            is_reduced[d] = true
          }
        }
    }
    let out_shape : Array[Int32] = []
    let kept_dims : Array[Int32] = []
    for d = 0; d < rank; d = d + 1 {
      if is_reduced[d] {
        if keepdims {
          out_shape.push(1)
        }
        kept_dims.push(1)
      } else {
        out_shape.push(shape[d])
        kept_dims.push(shape[d])
      }
    }
    let out_total = @tensor.num_elements(out_shape)
    let out_strides = @tensor.compute_strides(kept_dims)
    if x.dtype() == @tensor.Int64 {
      let src = x.to_i64_buffer()
      let acc = FixedArray::make(out_total, (0 : Int64))
      for linear = 0; linear < src.length(); linear = linear + 1 {
        let mut rem = linear
        let mut out_idx = 0
        for d = rank - 1; d >= 0; d = d - 1 {
          let coord = rem % shape[d]
          rem = rem / shape[d]
          if !is_reduced[d] {
            out_idx = out_idx + coord * out_strides[d]
          }
        }
        acc[out_idx] = acc[out_idx] + src[linear]
      }
      return @tensor.Tensor::from_i64(out_shape, acc)
    } else {
      let src = x.to_i32_buffer()
      let acc = FixedArray::make(out_total, 0)
      for linear = 0; linear < src.length(); linear = linear + 1 {
        let mut rem = linear
        let mut out_idx = 0
        for d = rank - 1; d >= 0; d = d - 1 {
          let coord = rem % shape[d]
          rem = rem / shape[d]
          if !is_reduced[d] {
            out_idx = out_idx + coord * out_strides[d]
          }
        }
        acc[out_idx] = acc[out_idx] + src[linear]
      }
      return @tensor.Tensor::from_i32(out_shape, acc)
    }
  }
  reduce_f32_impl(x, resolved, keepdims, noop_with_empty_axes, false, pool)
}

///|
/// ONNX `LayerNormalization` 算子（Opset 17）：
/// 对 `[axis..rank-1]` 归一化：`y = ((x - mean) / sqrt(var + epsilon)) * scale + bias`。
/// 使用双精度两遍扫描计算均值与方差，确保 Transformer / ViT / 语音模型深层数值稳定。
pub fn layer_norm_f32(
  x : @tensor.Tensor,
  scale : @tensor.Tensor,
  bias : @tensor.Tensor?,
  axis~ : Int32 = -1,
  epsilon~ : Float = (1.0e-5 : Float),
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let shape = x.shape()
  let rank = shape.length()
  let ax = @tensor.normalize_axis(axis, rank)

  let mut outer = 1
  for d = 0; d < ax; d = d + 1 {
    outer = outer * shape[d]
  }
  let mut norm_size = 1
  for d = ax; d < rank; d = d + 1 {
    norm_size = norm_size * shape[d]
  }

  let s_buf = scale.to_f32_buffer()
  if s_buf.length() != norm_size {
    raise @tensor.ShapeMismatch(
      "layer_norm_f32: scale elements (\{s_buf.length()}) != normalized size (\{norm_size})",
    )
  }
  let b_buf : FixedArray[Float]? = match bias {
    Some(b) => {
      let bb = b.to_f32_buffer()
      if bb.length() != norm_size {
        raise @tensor.ShapeMismatch(
          "layer_norm_f32: bias elements (\{bb.length()}) != normalized size (\{norm_size})",
        )
      }
      Some(bb)
    }
    None => None
  }

  let src = x.to_f32_buffer()
  let out = alloc_f32(pool, src.length())
  @nn.layernorm_f32(src, 0, s_buf, b_buf, out, 0, outer, norm_size, epsilon)
  @tensor.Tensor::from_f32(shape, out)
}
