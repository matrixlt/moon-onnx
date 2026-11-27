///|
/// 将输出坐标 `x_resized` 按 ONNX `coordinate_transformation_mode` 映射回原始输入浮点坐标。
fn map_resize_coord_nn(
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
    _ => (xr + 0.5) / scale - 0.5
  }
}

///|
/// 将原始浮点坐标按 ONNX `nearest_mode` 离散化为合法输入整数下标 `[0, in_len - 1]`。
fn map_nearest_idx_nn(
  x_orig : Double,
  in_len : Int32,
  nearest_mode : String,
) -> Int32 {
  let raw = match nearest_mode {
    "floor" => x_orig.floor().to_int()
    "ceil" => x_orig.ceil().to_int()
    "round_prefer_ceil" => (x_orig + 0.5).floor().to_int()
    _ => {
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
/// 零分配 4D NCHW `Resize`（支持 `nearest` 与 `bilinear` 模式及全套 ONNX 坐标变换模式，供 YOLOv8/分割 AOT Arena 直接调用）。
pub fn resize_2d_f32(
  x : FixedArray[Float],
  x_off : Int32,
  y : FixedArray[Float],
  y_off : Int32,
  n : Int32,
  c : Int32,
  ih : Int32,
  iw : Int32,
  oh : Int32,
  ow : Int32,
  is_linear : Bool,
  coord_mode : String,
  nearest_mode : String,
  scale_h~ : Double = 0.0,
  scale_w~ : Double = 0.0,
) -> Unit {
  let nc = n * c
  let in_plane = ih * iw
  let out_plane = oh * ow
  if nc == 0 || out_plane == 0 {
    return
  }
  check_nn_bounds(x.length(), x_off, nc * in_plane, "resize_2d_f32(x)")
  check_nn_bounds(y.length(), y_off, nc * out_plane, "resize_2d_f32(y)")
  let sc_h = if scale_h > 0.0 {
    scale_h
  } else {
    oh.to_double() / ih.to_double()
  }
  let sc_w = if scale_w > 0.0 {
    scale_w
  } else {
    ow.to_double() / iw.to_double()
  }

  if !is_linear {
    let y_map = FixedArray::makei(oh, fn(r) {
      map_nearest_idx_nn(
        map_resize_coord_nn(r, ih, oh, sc_h, coord_mode),
        ih,
        nearest_mode,
      )
    })
    let x_map = FixedArray::makei(ow, fn(col) {
      map_nearest_idx_nn(
        map_resize_coord_nn(col, iw, ow, sc_w, coord_mode),
        iw,
        nearest_mode,
      )
    })
    for ch = 0; ch < nc; ch = ch + 1 {
      let ib = x_off + ch * in_plane
      let ob = y_off + ch * out_plane
      for r = 0; r < oh; r = r + 1 {
        let src_row = ib + y_map.unsafe_get(r) * iw
        let dst_row = ob + r * ow
        for col = 0; col < ow; col = col + 1 {
          y.unsafe_set(
            dst_row + col,
            x.unsafe_get(src_row + x_map.unsafe_get(col)),
          )
        }
      }
    }
  } else {
    let y0_arr = FixedArray::make(oh, 0)
    let y1_arr = FixedArray::make(oh, 0)
    let wy_arr = FixedArray::make(oh, 0.0)
    let max_h = (ih - 1).to_double()
    for r = 0; r < oh; r = r + 1 {
      let mut oy = map_resize_coord_nn(r, ih, oh, sc_h, coord_mode)
      if oy < 0.0 {
        oy = 0.0
      }
      if oy > max_h {
        oy = max_h
      }
      let y0 = oy.floor().to_int()
      let y1 = if y0 + 1 < ih { y0 + 1 } else { ih - 1 }
      y0_arr.unsafe_set(r, y0)
      y1_arr.unsafe_set(r, y1)
      wy_arr.unsafe_set(r, oy - y0.to_double())
    }
    let x0_arr = FixedArray::make(ow, 0)
    let x1_arr = FixedArray::make(ow, 0)
    let wx_arr = FixedArray::make(ow, 0.0)
    let max_w = (iw - 1).to_double()
    for col = 0; col < ow; col = col + 1 {
      let mut ox = map_resize_coord_nn(col, iw, ow, sc_w, coord_mode)
      if ox < 0.0 {
        ox = 0.0
      }
      if ox > max_w {
        ox = max_w
      }
      let x0 = ox.floor().to_int()
      let x1 = if x0 + 1 < iw { x0 + 1 } else { iw - 1 }
      x0_arr.unsafe_set(col, x0)
      x1_arr.unsafe_set(col, x1)
      wx_arr.unsafe_set(col, ox - x0.to_double())
    }
    for ch = 0; ch < nc; ch = ch + 1 {
      let ib = x_off + ch * in_plane
      let ob = y_off + ch * out_plane
      for r = 0; r < oh; r = r + 1 {
        let r0 = ib + y0_arr.unsafe_get(r) * iw
        let r1 = ib + y1_arr.unsafe_get(r) * iw
        let wy = wy_arr.unsafe_get(r)
        let one_my = 1.0 - wy
        let dst_row = ob + r * ow
        for col = 0; col < ow; col = col + 1 {
          let x0 = x0_arr.unsafe_get(col)
          let x1 = x1_arr.unsafe_get(col)
          let wx = wx_arr.unsafe_get(col)
          let top = x.unsafe_get(r0 + x0).to_double() * (1.0 - wx) +
            x.unsafe_get(r0 + x1).to_double() * wx
          let bot = x.unsafe_get(r1 + x0).to_double() * (1.0 - wx) +
            x.unsafe_get(r1 + x1).to_double() * wx
          y.unsafe_set(
            dst_row + col,
            Float::from_double(top * one_my + bot * wy),
          )
        }
      }
    }
  }
}

///|
/// 计算两个角点矩形框 `(y1, x1, y2, x2)` 之间的交并比（IoU），严格兼容 ONNX `NonMaxSuppression` 规范。
fn box_iou_corners(
  ay1 : Double,
  ax1 : Double,
  ay2 : Double,
  ax2 : Double,
  by1 : Double,
  bx1 : Double,
  by2 : Double,
  bx2 : Double,
) -> Double {
  let a_ymin = if ay1 < ay2 { ay1 } else { ay2 }
  let a_ymax = if ay1 > ay2 { ay1 } else { ay2 }
  let a_xmin = if ax1 < ax2 { ax1 } else { ax2 }
  let a_xmax = if ax1 > ax2 { ax1 } else { ax2 }
  let b_ymin = if by1 < by2 { by1 } else { by2 }
  let b_ymax = if by1 > by2 { by1 } else { by2 }
  let b_xmin = if bx1 < bx2 { bx1 } else { bx2 }
  let b_xmax = if bx1 > bx2 { bx1 } else { bx2 }

  let area_a = (a_ymax - a_ymin) * (a_xmax - a_xmin)
  let area_b = (b_ymax - b_ymin) * (b_xmax - b_xmin)
  if area_a <= 0.0 || area_b <= 0.0 {
    return 0.0
  }
  let inter_y1 = if a_ymin > b_ymin { a_ymin } else { b_ymin }
  let inter_x1 = if a_xmin > b_xmin { a_xmin } else { b_xmin }
  let inter_y2 = if a_ymax < b_ymax { a_ymax } else { b_ymax }
  let inter_x2 = if a_xmax < b_xmax { a_xmax } else { b_xmax }
  let ih = inter_y2 - inter_y1
  let iw = inter_x2 - inter_x1
  if ih <= 0.0 || iw <= 0.0 {
    return 0.0
  }
  let inter = ih * iw
  inter / (area_a + area_b - inter)
}

///|
/// 标准 ONNX `NonMaxSuppression` 核心内核：
/// - `boxes`: `[num_batches, spatial_dimension, 4]`（`center_point_box=0` 时为 `[y1, x1, y2, x2]`，`=1` 时为 `[cx, cy, w, h]`）
/// - `scores`: `[num_batches, num_classes, spatial_dimension]`
/// - 返回选中结果三元组数组 `[(batch_idx, class_idx, box_idx)]`（按 ONNX 规范每类按分数降序贪心抑制）。
pub fn nms_f32(
  boxes : FixedArray[Float],
  scores : FixedArray[Float],
  num_batches : Int32,
  num_classes : Int32,
  num_boxes : Int32,
  max_output_boxes_per_class : Int32,
  iou_threshold : Float,
  score_threshold : Float,
  has_score_threshold : Bool,
  center_point_box : Int32,
) -> Array[(Int32, Int32, Int32)] {
  let results : Array[(Int32, Int32, Int32)] = []
  if max_output_boxes_per_class <= 0 || num_boxes <= 0 {
    return results
  }
  let iou_th = iou_threshold.to_double()
  let sc_th = score_threshold.to_double()

  fn get_box(b : Int32, idx : Int32) -> (Double, Double, Double, Double) {
    let off = (b * num_boxes + idx) * 4
    let v0 = boxes[off].to_double()
    let v1 = boxes[off + 1].to_double()
    let v2 = boxes[off + 2].to_double()
    let v3 = boxes[off + 3].to_double()
    if center_point_box != 0 {
      let hx = v2 * 0.5
      let hy = v3 * 0.5
      (v1 - hy, v0 - hx, v1 + hy, v0 + hx)
    } else {
      (v0, v1, v2, v3)
    }
  }

  for b = 0; b < num_batches; b = b + 1 {
    for c = 0; c < num_classes; c = c + 1 {
      let sc_base = (b * num_classes + c) * num_boxes
      let cands : Array[(Int32, Double)] = []
      for i = 0; i < num_boxes; i = i + 1 {
        let s = scores[sc_base + i].to_double()
        if !has_score_threshold || s > sc_th {
          cands.push((i, s))
        }
      }
      // 按分数降序排序（同分保留低索引优先，确定性对齐 ORT）
      cands.sort_by(fn(a, b_item) {
        if a.1 > b_item.1 {
          -1
        } else if a.1 < b_item.1 {
          1
        } else if a.0 < b_item.0 {
          -1
        } else if a.0 > b_item.0 {
          1
        } else {
          0
        }
      })
      let selected : Array[Int32] = []
      for ci = 0; ci < cands.length(); ci = ci + 1 {
        if selected.length() >= max_output_boxes_per_class {
          break
        }
        let idx = cands[ci].0
        let (ay1, ax1, ay2, ax2) = get_box(b, idx)
        let mut keep = true
        for si = 0; si < selected.length(); si = si + 1 {
          let (by1, bx1, by2, bx2) = get_box(b, selected[si])
          if box_iou_corners(ay1, ax1, ay2, ax2, by1, bx1, by2, bx2) > iou_th {
            keep = false
            break
          }
        }
        if keep {
          selected.push(idx)
          results.push((b, c, idx))
        }
      }
    }
  }
  results
}
