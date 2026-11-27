///|
/// ONNX `Resize` 算子（Opset 11~18）：
/// 支持 4D `[N, C, H, W]`（视觉检测/分割常用双线性或最近邻上采样）与 3D `[N, C, L]`，
/// 兼容 `scales` 与 `sizes` 两种目标尺寸指定方式及 `BufferPool` 复用，统一委托 `@nn.resize_2d_f32`。
pub fn resize_f32(
  x : @tensor.Tensor,
  scales_opt : @tensor.Tensor?,
  sizes_opt : @tensor.Tensor?,
  mode~ : String = "nearest",
  coordinate_transformation_mode~ : String = "half_pixel",
  nearest_mode~ : String = "round_prefer_floor",
  pool~ : @tensor.BufferPool? = None,
) -> @tensor.Tensor raise @tensor.TensorError {
  let in_shape = x.shape()
  let rank = in_shape.length()
  if rank != 4 && rank != 3 {
    raise @tensor.Unsupported(
      "resize_f32 currently supports 3D NCL and 4D NCHW tensors, got rank \{rank}",
    )
  }

  // 解析输出形状与各轴真实缩放比例
  let out_shape = in_shape.copy()
  let axis_scales = Array::make(rank, 1.0)
  match sizes_opt {
    Some(sz_t) if sz_t.numel() > 0 => {
      let sz_buf = sz_t.to_i64_buffer()
      if sz_buf.length() != rank {
        raise @tensor.ShapeMismatch(
          "resize_f32: sizes length \{sz_buf.length()} != rank \{rank}",
        )
      }
      for d = 0; d < rank; d = d + 1 {
        let od = sz_buf[d].to_int()
        if od <= 0 {
          raise @tensor.InvalidShape("resize_f32: non-positive size \{od}")
        }
        out_shape[d] = od
        axis_scales[d] = od.to_double() / in_shape[d].to_double()
      }
    }
    _ =>
      match scales_opt {
        Some(sc_t) if sc_t.numel() > 0 => {
          let sc_buf = sc_t.to_f32_buffer()
          if sc_buf.length() != rank {
            raise @tensor.ShapeMismatch(
              "resize_f32: scales length \{sc_buf.length()} != rank \{rank}",
            )
          }
          for d = 0; d < rank; d = d + 1 {
            let s = sc_buf[d].to_double()
            if s <= 0.0 {
              raise @tensor.InvalidShape(
                "resize_f32: non-positive scale at axis \{d}",
              )
            }
            axis_scales[d] = s
            let od = (in_shape[d].to_double() * s).floor().to_int()
            out_shape[d] = if od > 0 { od } else { 1 }
          }
        }
        _ =>
          raise @tensor.InvalidShape(
            "resize_f32 requires either scales or sizes input",
          )
      }
  }

  if out_shape[0] != in_shape[0] || out_shape[1] != in_shape[1] {
    raise @tensor.Unsupported(
      "resize_f32 currently resizes spatial dimensions only (N and C must remain unchanged)",
    )
  }

  let is_linear = match mode {
    "linear" | "bilinear" => true
    "" | "nearest" => false
    other =>
      raise @tensor.Unsupported(
        "resize_f32: unsupported mode '\{other}' (supports nearest/linear)",
      )
  }

  let n = in_shape[0]
  let c = in_shape[1]
  let src = x.to_f32_buffer()
  let out_total = @tensor.num_elements(out_shape)
  let out = alloc_f32(pool, out_total)

  // 3D [N, C, L] 等价于 ih=1, oh=1 的 4D [N, C, 1, L]，统一复用 @nn.resize_2d_f32
  let (ih, iw, oh, ow, sc_h, sc_w) = if rank == 3 {
    (1, in_shape[2], 1, out_shape[2], 1.0, axis_scales[2])
  } else {
    (
      in_shape[2],
      in_shape[3],
      out_shape[2],
      out_shape[3],
      axis_scales[2],
      axis_scales[3],
    )
  }
  @nn.resize_2d_f32(
    src,
    0,
    out,
    0,
    n,
    c,
    ih,
    iw,
    oh,
    ow,
    is_linear,
    coordinate_transformation_mode,
    nearest_mode,
    scale_h=sc_h,
    scale_w=sc_w,
  )
  @tensor.Tensor::from_f32(out_shape, out)
}

///|
/// ONNX `NonMaxSuppression` 算子（Opset 10~19）：
/// - `boxes`: `[num_batches, spatial_dimension, 4]`（Float32）
/// - `scores`: `[num_batches, num_classes, spatial_dimension]`（Float32）
/// - `max_output_boxes_per_class`: 可选标量 Int64（默认 0，按 ONNX 规范表示不输出任何框）
/// - `iou_threshold`: 可选标量 Float32（默认 0.0）
/// - `score_threshold`: 可选标量 Float32（未提供时不按分数过滤）
/// 返回 `Int64` 张量，形状 `[num_selected_indices, 3]`，每行为 `[batch_index, class_index, box_index]`。
pub fn nms_op(
  boxes : @tensor.Tensor,
  scores : @tensor.Tensor,
  max_output_boxes_per_class : @tensor.Tensor?,
  iou_threshold : @tensor.Tensor?,
  score_threshold : @tensor.Tensor?,
  center_point_box~ : Int32 = 0,
) -> @tensor.Tensor raise @tensor.TensorError {
  let b_sh = boxes.shape()
  let s_sh = scores.shape()
  if b_sh.length() != 3 || b_sh[2] != 4 {
    raise @tensor.InvalidShape(
      "NonMaxSuppression: boxes must have shape [B, N, 4], got rank=\{b_sh.length()}",
    )
  }
  if s_sh.length() != 3 || s_sh[0] != b_sh[0] || s_sh[2] != b_sh[1] {
    raise @tensor.ShapeMismatch(
      "NonMaxSuppression: scores shape incompatible with boxes (rank=\{s_sh.length()})",
    )
  }
  let max_boxes = match max_output_boxes_per_class {
    Some(t) if t.numel() > 0 => t.to_i64_buffer()[0].to_int()
    _ => 0
  }
  let iou_th : Float = match iou_threshold {
    Some(t) if t.numel() > 0 => t.to_f32_buffer()[0]
    _ => (0.0 : Float)
  }
  let (has_score_th, score_th) : (Bool, Float) = match score_threshold {
    Some(t) if t.numel() > 0 => (true, t.to_f32_buffer()[0])
    _ => (false, (0.0 : Float))
  }
  let b_buf = boxes.to_f32_buffer()
  let s_buf = scores.to_f32_buffer()
  let sel = @nn.nms_f32(
    b_buf,
    s_buf,
    b_sh[0],
    s_sh[1],
    b_sh[1],
    max_boxes,
    iou_th,
    score_th,
    has_score_th,
    center_point_box,
  )
  let count = sel.length()
  let out = FixedArray::make(count * 3, (0 : Int64))
  for i = 0; i < count; i = i + 1 {
    let (bi, ci, idx) = sel[i]
    out[i * 3] = bi.to_int64()
    out[i * 3 + 1] = ci.to_int64()
    out[i * 3 + 2] = idx.to_int64()
  }
  @tensor.Tensor::from_i64([count, 3], out)
}
