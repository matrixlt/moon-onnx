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
