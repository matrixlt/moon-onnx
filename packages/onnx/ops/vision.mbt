///|
/// 将输出坐标 `x_resized` 按 ONNX `coordinate_transformation_mode` 映射回原始输入浮点坐标。
fn map_resize_coord(
  x_resized : Int32,
  in_len : Int32,
  out_len : Int32,
  scale : Double,
  coord_mode : String,
) -> Double {
  let xr = x_resized.to_double()
  match coord_mode {
    "align_corners" =>
      if out_len <= 1 {
        0.0
      } else {
        xr * (in_len - 1).to_double() / (out_len - 1).to_double()
      }
    "asymmetric" => xr / scale
    "pytorch_half_pixel" =>
      if out_len <= 1 {
        0.0
      } else {
        (xr + 0.5) / scale - 0.5
      }
    _ => (xr + 0.5) / scale - 0.5 // 默认 "half_pixel"
  }
}

///|
/// 将原始浮点坐标按 ONNX `nearest_mode` 离散化为合法输入整数下标 `[0, in_len - 1]`。
fn map_nearest_index(
  x_orig : Double,
  in_len : Int32,
  nearest_mode : String,
) -> Int32 {
  let raw = match nearest_mode {
    "floor" => x_orig.floor().to_int()
    "ceil" => x_orig.ceil().to_int()
    "round_prefer_ceil" => (x_orig + 0.5).floor().to_int()
    _ => {
      // 默认 "round_prefer_floor"（半值向下取整，符合 ONNX Opset 11+ 默认值）
      let fl = x_orig.floor()
      let diff = x_orig - fl
      if diff > 0.5 {
        fl.to_int() + 1
      } else {
        fl.to_int()
      }
    }
  }
  if raw < 0 {
    0
  } else if raw >= in_len {
    in_len - 1
  } else {
    raw
  }
}

///|
/// ONNX `Resize` 算子（Opset 11~18）：
/// 支持 4D `[N, C, H, W]`（视觉检测/分割常用双线性或最近邻上采样）与 3D `[N, C, L]`，
/// 兼容 `scales` 与 `sizes` 两种目标尺寸指定方式及 `BufferPool` 复用。
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
  let nc = n * c
  let src = x.to_f32_buffer()
  let out_total = @tensor.num_elements(out_shape)
  let out = alloc_f32(pool, out_total)

  if rank == 3 {
    let in_l = in_shape[2]
    let out_l = out_shape[2]
    let sc_l = axis_scales[2]
    if !is_linear {
      let idx_map = FixedArray::makei(out_l, fn(i) {
        let orig = map_resize_coord(
          i, in_l, out_l, sc_l, coordinate_transformation_mode,
        )
        map_nearest_index(orig, in_l, nearest_mode)
      })
      for ch = 0; ch < nc; ch = ch + 1 {
        let ib = ch * in_l
        let ob = ch * out_l
        for i = 0; i < out_l; i = i + 1 {
          out[ob + i] = src[ib + idx_map[i]]
        }
      }
    } else {
      let i0_arr = FixedArray::make(out_l, 0)
      let i1_arr = FixedArray::make(out_l, 0)
      let w_arr = FixedArray::make(out_l, 0.0)
      for i = 0; i < out_l; i = i + 1 {
        let mut orig = map_resize_coord(
          i, in_l, out_l, sc_l, coordinate_transformation_mode,
        )
        if orig < 0.0 {
          orig = 0.0
        }
        let max_c = (in_l - 1).to_double()
        if orig > max_c {
          orig = max_c
        }
        let i0 = orig.floor().to_int()
        let i1 = if i0 + 1 < in_l { i0 + 1 } else { in_l - 1 }
        i0_arr[i] = i0
        i1_arr[i] = i1
        w_arr[i] = orig - i0.to_double()
      }
      for ch = 0; ch < nc; ch = ch + 1 {
        let ib = ch * in_l
        let ob = ch * out_l
        for i = 0; i < out_l; i = i + 1 {
          let w1 = w_arr[i]
          let v0 = src[ib + i0_arr[i]].to_double()
          let v1 = src[ib + i1_arr[i]].to_double()
          out[ob + i] = Float::from_double(v0 * (1.0 - w1) + v1 * w1)
        }
      }
    }
    return @tensor.Tensor::from_f32(out_shape, out)
  }

  // 4D [N, C, H, W]
  let in_h = in_shape[2]
  let in_w = in_shape[3]
  let out_h = out_shape[2]
  let out_w = out_shape[3]
  let sc_h = axis_scales[2]
  let sc_w = axis_scales[3]
  let in_plane = in_h * in_w
  let out_plane = out_h * out_w

  if !is_linear {
    let h_map = FixedArray::makei(out_h, fn(r) {
      let orig = map_resize_coord(
        r, in_h, out_h, sc_h, coordinate_transformation_mode,
      )
      map_nearest_index(orig, in_h, nearest_mode)
    })
    let w_map = FixedArray::makei(out_w, fn(col) {
      let orig = map_resize_coord(
        col, in_w, out_w, sc_w, coordinate_transformation_mode,
      )
      map_nearest_index(orig, in_w, nearest_mode)
    })
    for ch = 0; ch < nc; ch = ch + 1 {
      let ib = ch * in_plane
      let ob = ch * out_plane
      for r = 0; r < out_h; r = r + 1 {
        let src_row = ib + h_map[r] * in_w
        let dst_row = ob + r * out_w
        for col = 0; col < out_w; col = col + 1 {
          out[dst_row + col] = src[src_row + w_map[col]]
        }
      }
    }
  } else {
    // 预计算行/列双线性插值索引与权重，彻底消除内层浮点坐标映射开销
    let y0_arr = FixedArray::make(out_h, 0)
    let y1_arr = FixedArray::make(out_h, 0)
    let wy_arr = FixedArray::make(out_h, 0.0)
    let max_y = (in_h - 1).to_double()
    for r = 0; r < out_h; r = r + 1 {
      let mut orig = map_resize_coord(
        r, in_h, out_h, sc_h, coordinate_transformation_mode,
      )
      if orig < 0.0 {
        orig = 0.0
      } else if orig > max_y {
        orig = max_y
      }
      let y0 = orig.floor().to_int()
      let y1 = if y0 + 1 < in_h { y0 + 1 } else { in_h - 1 }
      y0_arr[r] = y0
      y1_arr[r] = y1
      wy_arr[r] = orig - y0.to_double()
    }

    let x0_arr = FixedArray::make(out_w, 0)
    let x1_arr = FixedArray::make(out_w, 0)
    let wx_arr = FixedArray::make(out_w, 0.0)
    let max_x = (in_w - 1).to_double()
    for col = 0; col < out_w; col = col + 1 {
      let mut orig = map_resize_coord(
        col, in_w, out_w, sc_w, coordinate_transformation_mode,
      )
      if orig < 0.0 {
        orig = 0.0
      } else if orig > max_x {
        orig = max_x
      }
      let x0 = orig.floor().to_int()
      let x1 = if x0 + 1 < in_w { x0 + 1 } else { in_w - 1 }
      x0_arr[col] = x0
      x1_arr[col] = x1
      wx_arr[col] = orig - x0.to_double()
    }

    for ch = 0; ch < nc; ch = ch + 1 {
      let ib = ch * in_plane
      let ob = ch * out_plane
      for r = 0; r < out_h; r = r + 1 {
        let row0 = ib + y0_arr[r] * in_w
        let row1 = ib + y1_arr[r] * in_w
        let wy = wy_arr[r]
        let wy0 = 1.0 - wy
        let dst_row = ob + r * out_w
        for col = 0; col < out_w; col = col + 1 {
          let x0 = x0_arr[col]
          let x1 = x1_arr[col]
          let wx = wx_arr[col]
          let wx0 = 1.0 - wx
          let v00 = src[row0 + x0].to_double()
          let v01 = src[row0 + x1].to_double()
          let v10 = src[row1 + x0].to_double()
          let v11 = src[row1 + x1].to_double()
          let top = v00 * wx0 + v01 * wx
          let bot = v10 * wx0 + v11 * wx
          out[dst_row + col] = Float::from_double(top * wy0 + bot * wy)
        }
      }
    }
  }
  @tensor.Tensor::from_f32(out_shape, out)
}
