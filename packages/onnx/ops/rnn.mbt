///|
/// 数值稳定双精度 Sigmoid（供 LSTM 门控计算使用）。
fn sigmoid_f64(x : Double) -> Double {
  1.0 / (1.0 + @math.exp(-x))
}

///|
/// 可选的门控输入区间裁剪（对应 ONNX LSTM `clip` 属性）。
fn apply_gate_clip(v : Double, clip : Float?) -> Double {
  match clip {
    Some(c) => {
      let lim = c.to_double()
      if v > lim {
        lim
      } else if v < -lim {
        -lim
      } else {
        v
      }
    }
    None => v
  }
}

///|
/// ONNX 标准 `LSTM` 算子（Opset 7~18，语音端点检测 Silero VAD / 时序模型核心层）：
/// - 支持 `direction = "forward" | "reverse" | "bidirectional"`；
/// - 支持 `layout = 0`（`[seq_len, batch, input_size]`）与 `layout = 1`（`[batch, seq_len, input_size]`）；
/// - 严格遵循 ONNX 门控顺序 **`i, o, f, c`**（输入门、输出门、遗忘门、候选细胞态）；
/// - 支持可选偏置 `B`、变长序列 `sequence_lens`、初始状态 `initial_h`/`initial_c`、窥视孔权重 `P`（peepholes）及 `clip`；
/// - 返回 3 元张量数组 `[Y, Y_h, Y_c]`，配合 `ExecutionProvider::eval_node_multi` 无缝支持任意输出组合。
pub fn lstm_f32(
  x : @tensor.Tensor,
  w : @tensor.Tensor,
  r : @tensor.Tensor,
  bias : @tensor.Tensor?,
  seq_lens : @tensor.Tensor?,
  initial_h : @tensor.Tensor?,
  initial_c : @tensor.Tensor?,
  peepholes : @tensor.Tensor?,
  hidden_size~ : Int32? = None,
  direction~ : String = "forward",
  layout~ : Int32 = 0,
  clip~ : Float? = None,
) -> Array[@tensor.Tensor] raise @tensor.TensorError {
  let xs = x.shape()
  let ws = w.shape()
  let rs = r.shape()
  if xs.length() != 3 || ws.length() != 3 || rs.length() != 3 {
    raise @tensor.ShapeMismatch(
      "lstm_f32 requires 3D X, W, R tensors, got ranks \{xs.length()}, \{ws.length()}, \{rs.length()}",
    )
  }

  let (seq_len, batch_size, input_size) = if layout == 1 {
    (xs[1], xs[0], xs[2])
  } else {
    (xs[0], xs[1], xs[2])
  }
  let num_dirs = match direction {
    "bidirectional" => 2
    "forward" | "reverse" | "" => 1
    other =>
      raise @tensor.Unsupported("lstm_f32: unsupported direction '\{other}'")
  }
  let h_size = match hidden_size {
    Some(h) => h
    None => rs[2]
  }
  if ws[0] != num_dirs ||
    ws[1] != 4 * h_size ||
    ws[2] != input_size ||
    rs[0] != num_dirs ||
    rs[1] != 4 * h_size ||
    rs[2] != h_size {
    raise @tensor.ShapeMismatch(
      "lstm_f32 weight shape mismatch: W=\{@tensor.format_shape(ws)}, R=\{@tensor.format_shape(rs)}, H=\{h_size}",
    )
  }

  let x_buf = x.to_f32_buffer()
  let w_buf = w.to_f32_buffer()
  let r_buf = r.to_f32_buffer()
  let b_buf : FixedArray[Float]? = match bias {
    Some(bt) => Some(bt.to_f32_buffer())
    None => None
  }
  let p_buf : FixedArray[Float]? = match peepholes {
    Some(pt) => Some(pt.to_f32_buffer())
    None => None
  }
  let lens : FixedArray[Int32] = match seq_lens {
    Some(lt) => {
      let lb = lt.to_i64_buffer()
      FixedArray::makei(batch_size, fn(b) { lb[b].to_int() })
    }
    None => FixedArray::make(batch_size, seq_len)
  }

  // 输出张量缓冲：
  // layout == 0: Y [seq_len, num_dirs, batch, H], Y_h/Y_c [num_dirs, batch, H]
  // layout == 1: Y [batch, seq_len, num_dirs, H], Y_h/Y_c [batch, num_dirs, H]
  let y_buf = FixedArray::make(
    seq_len * num_dirs * batch_size * h_size,
    (0.0 : Float),
  )
  let yh_buf = FixedArray::make(num_dirs * batch_size * h_size, (0.0 : Float))
  let yc_buf = FixedArray::make(num_dirs * batch_size * h_size, (0.0 : Float))

  let init_h_buf : FixedArray[Float]? = match initial_h {
    Some(ht) => Some(ht.to_f32_buffer())
    None => None
  }
  let init_c_buf : FixedArray[Float]? = match initial_c {
    Some(ct) => Some(ct.to_f32_buffer())
    None => None
  }

  // 复用单步隐藏状态与 4 门预激活工作缓冲
  let h_cur = FixedArray::make(h_size, 0.0)
  let c_cur = FixedArray::make(h_size, 0.0)
  let gates = FixedArray::make(4 * h_size, 0.0)

  for dir = 0; dir < num_dirs; dir = dir + 1 {
    let is_rev = direction == "reverse" ||
      (direction == "bidirectional" && dir == 1)
    let w_dir_off = dir * 4 * h_size * input_size
    let r_dir_off = dir * 4 * h_size * h_size
    let b_dir_off = dir * 8 * h_size
    let p_dir_off = dir * 3 * h_size

    for b = 0; b < batch_size; b = b + 1 {
      let valid_len = lens[b]
      // 加载该 batch 样本的初始隐状态 h_0 与细胞态 c_0
      let init_off = if layout == 1 {
        (b * num_dirs + dir) * h_size
      } else {
        (dir * batch_size + b) * h_size
      }
      for h = 0; h < h_size; h = h + 1 {
        h_cur[h] = match init_h_buf {
          Some(hb) => hb[init_off + h].to_double()
          None => 0.0
        }
        c_cur[h] = match init_c_buf {
          Some(cb) => cb[init_off + h].to_double()
          None => 0.0
        }
      }

      for step = 0; step < valid_len; step = step + 1 {
        let t = if is_rev { valid_len - 1 - step } else { step }
        let x_off = if layout == 1 {
          (b * seq_len + t) * input_size
        } else {
          (t * batch_size + b) * input_size
        }

        // 计算 4 门线性变换：gates[g] = (W * x_t + R * h_{t-1} + Wb + Rb)[g]
        // ONNX 门序：0..H = i (input), H..2H = o (output), 2H..3H = f (forget), 3H..4H = c (cell)
        for g = 0; g < 4 * h_size; g = g + 1 {
          let mut acc = match b_buf {
            Some(bb) =>
              bb[b_dir_off + g].to_double() +
              bb[b_dir_off + 4 * h_size + g].to_double()
            None => 0.0
          }
          let w_row = w_dir_off + g * input_size
          for k = 0; k < input_size; k = k + 1 {
            acc = acc +
              w_buf[w_row + k].to_double() * x_buf[x_off + k].to_double()
          }
          let r_row = r_dir_off + g * h_size
          for k = 0; k < h_size; k = k + 1 {
            acc = acc + r_buf[r_row + k].to_double() * h_cur[k]
          }
          gates[g] = acc
        }

        let y_step_off = if layout == 1 {
          ((b * seq_len + t) * num_dirs + dir) * h_size
        } else {
          ((t * num_dirs + dir) * batch_size + b) * h_size
        }

        // 更新细胞态 c_t 与隐藏态 h_t
        for h = 0; h < h_size; h = h + 1 {
          let c_prev = c_cur[h]
          let (pi, po, pf) = match p_buf {
            Some(pb) =>
              (
                pb[p_dir_off + h].to_double(),
                pb[p_dir_off + h_size + h].to_double(),
                pb[p_dir_off + 2 * h_size + h].to_double(),
              )
            None => (0.0, 0.0, 0.0)
          }
          let i_t = sigmoid_f64(apply_gate_clip(gates[h] + pi * c_prev, clip))
          let f_t = sigmoid_f64(
            apply_gate_clip(gates[2 * h_size + h] + pf * c_prev, clip),
          )
          let c_tilde = @math.tanh(apply_gate_clip(gates[3 * h_size + h], clip))
          let c_new = f_t * c_prev + i_t * c_tilde
          let o_t = sigmoid_f64(
            apply_gate_clip(gates[h_size + h] + po * c_new, clip),
          )
          let h_new = o_t * @math.tanh(c_new)
          c_cur[h] = c_new
          h_cur[h] = h_new
          y_buf[y_step_off + h] = Float::from_double(h_new)
        }
      }

      // 写回最终隐状态 Y_h 与最终细胞态 Y_c
      for h = 0; h < h_size; h = h + 1 {
        yh_buf[init_off + h] = Float::from_double(h_cur[h])
        yc_buf[init_off + h] = Float::from_double(c_cur[h])
      }
    }
  }

  let y_shape = if layout == 1 {
    [batch_size, seq_len, num_dirs, h_size]
  } else {
    [seq_len, num_dirs, batch_size, h_size]
  }
  let state_shape = if layout == 1 {
    [batch_size, num_dirs, h_size]
  } else {
    [num_dirs, batch_size, h_size]
  }

  [
    @tensor.Tensor::from_f32(y_shape, y_buf),
    @tensor.Tensor::from_f32(state_shape, yh_buf),
    @tensor.Tensor::from_f32(state_shape, yc_buf),
  ]
}
