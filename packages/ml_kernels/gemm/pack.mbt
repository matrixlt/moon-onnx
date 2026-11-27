///|
/// 对数组切片 `[offset, offset + count)` 执行一次性 O(1) 入口边界校验，
/// 确保后续热循环使用 `unsafe_get` / `unsafe_set` 时严格内存安全。
fn check_slice_bounds(
  len : Int32,
  offset : Int32,
  count : Int32,
  label : String,
) -> Unit {
  if count < 0 || offset < 0 || offset + count > len {
    abort(
      "ml_kernels/gemm: out of bounds in \{label} (offset=\{offset}, count=\{count}, len=\{len})",
    )
  }
}

///|
/// 预打包后的右矩阵 `B[K, N]`：按 BLIS 8 列微面板布局 `[N_pad / 8, K, 8]` 连续存储于小端序 `FixedArray[Byte]`。
/// 每个 8 列微面板沿 `K` 维步长恒为 32 字节（2 条 `V128` 向量），彻底消除大矩阵跨行访存造成的 L1/TLB 缓存未命中。
pub struct PackedMatrixB {
  data : FixedArray[Byte]
  k : Int32
  n : Int32
  /// 向上对齐到 8 的倍数的列宽，保证每个微面板均为完整 8 列（32 字节）
  n_pad : Int32
}

///|
/// 返回预打包矩阵的逻辑维度 `(K, N)`。
pub fn PackedMatrixB::dims(self : PackedMatrixB) -> (Int32, Int32) {
  (self.k, self.n)
}

///|
/// 将行主序矩阵 `B[K, N]` 打包为 8 列连续微面板布局的 `PackedMatrixB`。
pub fn pack_matrix_b_f32(
  b : FixedArray[Float],
  b_offset : Int32,
  k : Int32,
  n : Int32,
) -> PackedMatrixB {
  if k > 0 && n > 0 {
    check_slice_bounds(b.length(), b_offset, k * n, "pack_matrix_b_f32(b)")
  }
  let n_pad = (n + 7) / 8 * 8
  let num_panels = n_pad / 8
  let panel_bytes = k * 32
  let packed = FixedArray::make(num_panels * panel_bytes, b'\x00')
  let zero : Float = 0.0

  for jb = 0; jb < num_panels; jb = jb + 1 {
    let col = jb * 8
    let panel_base = jb * panel_bytes
    if col + 8 <= n {
      let mut s = b_offset + col
      let mut d = panel_base
      for _p = 0; _p < k; _p = _p + 1 {
        let v0 = @simd.f32x4_const(
          b.unsafe_get(s),
          b.unsafe_get(s + 1),
          b.unsafe_get(s + 2),
          b.unsafe_get(s + 3),
        )
        let v1 = @simd.f32x4_const(
          b.unsafe_get(s + 4),
          b.unsafe_get(s + 5),
          b.unsafe_get(s + 6),
          b.unsafe_get(s + 7),
        )
        @simd.v128_store(packed, d, v0)
        @simd.v128_store(packed, d + 16, v1)
        s = s + n
        d = d + 32
      }
    } else {
      let mut s = b_offset + col
      let mut d = panel_base
      for _p = 0; _p < k; _p = _p + 1 {
        let f0 = if col < n { b.unsafe_get(s) } else { zero }
        let f1 = if col + 1 < n { b.unsafe_get(s + 1) } else { zero }
        let f2 = if col + 2 < n { b.unsafe_get(s + 2) } else { zero }
        let f3 = if col + 3 < n { b.unsafe_get(s + 3) } else { zero }
        let f4 = if col + 4 < n { b.unsafe_get(s + 4) } else { zero }
        let f5 = if col + 5 < n { b.unsafe_get(s + 5) } else { zero }
        let f6 = if col + 6 < n { b.unsafe_get(s + 6) } else { zero }
        let f7 = if col + 7 < n { b.unsafe_get(s + 7) } else { zero }
        @simd.v128_store(packed, d, @simd.f32x4_const(f0, f1, f2, f3))
        @simd.v128_store(packed, d + 16, @simd.f32x4_const(f4, f5, f6, f7))
        s = s + n
        d = d + 32
      }
    }
  }
  { data: packed, k, n, n_pad, }
}

///|
/// 将行主序存储的转置矩阵 `B^T[N, K]`（如 `Gemm(trans_b=true)` 或 `Conv` 权重 `[C_out_g, K]`）
/// 一步转置并打包为 8 列连续微面板布局的 `PackedMatrixB(K, N)`，避免中间转置缓冲区分配。
pub fn pack_matrix_b_transposed_f32(
  b_t : FixedArray[Float],
  b_offset : Int32,
  n : Int32,
  k : Int32,
) -> PackedMatrixB {
  if k > 0 && n > 0 {
    check_slice_bounds(
      b_t.length(),
      b_offset,
      n * k,
      "pack_matrix_b_transposed_f32(b_t)",
    )
  }
  let n_pad = (n + 7) / 8 * 8
  let num_panels = n_pad / 8
  let panel_bytes = k * 32
  let packed = FixedArray::make(num_panels * panel_bytes, b'\x00')
  let zero : Float = 0.0

  for jb = 0; jb < num_panels; jb = jb + 1 {
    let col = jb * 8
    let panel_base = jb * panel_bytes
    if col + 8 <= n {
      let r0 = b_offset + col * k
      let r1 = r0 + k
      let r2 = r1 + k
      let r3 = r2 + k
      let r4 = r3 + k
      let r5 = r4 + k
      let r6 = r5 + k
      let r7 = r6 + k
      let mut d = panel_base
      for p = 0; p < k; p = p + 1 {
        let v0 = @simd.f32x4_const(
          b_t.unsafe_get(r0 + p),
          b_t.unsafe_get(r1 + p),
          b_t.unsafe_get(r2 + p),
          b_t.unsafe_get(r3 + p),
        )
        let v1 = @simd.f32x4_const(
          b_t.unsafe_get(r4 + p),
          b_t.unsafe_get(r5 + p),
          b_t.unsafe_get(r6 + p),
          b_t.unsafe_get(r7 + p),
        )
        @simd.v128_store(packed, d, v0)
        @simd.v128_store(packed, d + 16, v1)
        d = d + 32
      }
    } else {
      let mut d = panel_base
      for p = 0; p < k; p = p + 1 {
        let f0 = if col < n {
          b_t.unsafe_get(b_offset + col * k + p)
        } else {
          zero
        }
        let f1 = if col + 1 < n {
          b_t.unsafe_get(b_offset + (col + 1) * k + p)
        } else {
          zero
        }
        let f2 = if col + 2 < n {
          b_t.unsafe_get(b_offset + (col + 2) * k + p)
        } else {
          zero
        }
        let f3 = if col + 3 < n {
          b_t.unsafe_get(b_offset + (col + 3) * k + p)
        } else {
          zero
        }
        let f4 = if col + 4 < n {
          b_t.unsafe_get(b_offset + (col + 4) * k + p)
        } else {
          zero
        }
        let f5 = if col + 5 < n {
          b_t.unsafe_get(b_offset + (col + 5) * k + p)
        } else {
          zero
        }
        let f6 = if col + 6 < n {
          b_t.unsafe_get(b_offset + (col + 6) * k + p)
        } else {
          zero
        }
        let f7 = if col + 7 < n {
          b_t.unsafe_get(b_offset + (col + 7) * k + p)
        } else {
          zero
        }
        @simd.v128_store(packed, d, @simd.f32x4_const(f0, f1, f2, f3))
        @simd.v128_store(packed, d + 16, @simd.f32x4_const(f4, f5, f6, f7))
        d = d + 32
      }
    }
  }
  { data: packed, k, n, n_pad, }
}

///|
/// 预打包 NCHW 卷积微内核写回前的寄存器级融合激活配置。
/// 在 `V128` 累加器写回内存前原地执行，避免独立激活算子的整张特征图读写开销。
#derive(Debug, Eq)
pub(all) enum Epilogue {
  /// 无融合激活
  NoEpilogue
  /// 融合 ReLU（使用硬件 `f32x4_pmax(0.0, acc)`）
  Relu
  /// 融合区间裁剪（`min(max, max(min, acc))`，典型如 MobileNetV2 ReLU6 `[0.0, 6.0]`）
  Clip(min~ : Float, max~ : Float)
  /// 融合 SiLU / Swish（`x * sigmoid(x)`，在 V128 寄存器内直接调用 4-wide 快速 `sigmoid_f32x4`，消除 YOLOv8 独立 SiLU 访存与 libm 开销）
  Silu
}

///|
/// 对单个 V128 累加器应用融合激活（无激活时零开销透传）。
pub fn apply_epilogue_v128(v : V128, epilogue : Epilogue) -> V128 {
  match epilogue {
    NoEpilogue => v
    Relu => @simd.f32x4_pmax(@simd.f32x4_splat((0.0 : Float)), v)
    Clip(min~, max~) =>
      @simd.f32x4_pmin(
        @simd.f32x4_splat(max),
        @simd.f32x4_pmax(@simd.f32x4_splat(min), v),
      )
    Silu => @simd.f32x4_mul(v, @vecmath.sigmoid_f32x4(v))
  }
}

///|
/// 对单个标量 Float32 应用融合激活（用于 `k == 0` 等边界路径）。
pub fn apply_epilogue_scalar(x : Float, epilogue : Epilogue) -> Float {
  match epilogue {
    NoEpilogue => x
    Relu => if x > (0.0 : Float) { x } else { (0.0 : Float) }
    Clip(min~, max~) => {
      let lo = if x > min { x } else { min }
      if lo < max {
        lo
      } else {
        max
      }
    }
    Silu => {
      let vd = x.to_double()
      Float::from_double(vd / (1.0 + @math.exp(-vd)))
    }
  }
}

///|
/// 将单个 V128 累加器写回输出矩阵行 `out[row_off + col ..]`，自动裁剪超过 `n` 的尾部通道（调用前已校验边界）。
fn store_v128_clamped(
  out : FixedArray[Float],
  row_off : Int32,
  col : Int32,
  n : Int32,
  v : V128,
) -> Unit {
  if col + 4 <= n {
    let base = row_off + col
    out.unsafe_set(base, @simd.f32x4_extract_lane(v, 0))
    out.unsafe_set(base + 1, @simd.f32x4_extract_lane(v, 1))
    out.unsafe_set(base + 2, @simd.f32x4_extract_lane(v, 2))
    out.unsafe_set(base + 3, @simd.f32x4_extract_lane(v, 3))
  } else {
    if col < n {
      out.unsafe_set(row_off + col, @simd.f32x4_extract_lane(v, 0))
    }
    if col + 1 < n {
      out.unsafe_set(row_off + col + 1, @simd.f32x4_extract_lane(v, 1))
    }
    if col + 2 < n {
      out.unsafe_set(row_off + col + 2, @simd.f32x4_extract_lane(v, 2))
    }
  }
}

///|
/// 从可选通道偏置数组加载 `[col .. col+4)` 的 V128 初始累加向量（无偏置时返回全 0 向量）。
fn load_bias_v128(
  bias : FixedArray[Float]?,
  bias_offset : Int32,
  col : Int32,
  n : Int32,
) -> V128 {
  let zero : Float = 0.0
  match bias {
    None => @simd.f32x4_splat(zero)
    Some(b) =>
      if col + 4 <= n {
        let base = bias_offset + col
        @simd.f32x4_const(
          b.unsafe_get(base),
          b.unsafe_get(base + 1),
          b.unsafe_get(base + 2),
          b.unsafe_get(base + 3),
        )
      } else {
        let f0 = if col < n { b.unsafe_get(bias_offset + col) } else { zero }
        let f1 = if col + 1 < n {
          b.unsafe_get(bias_offset + col + 1)
        } else {
          zero
        }
        let f2 = if col + 2 < n {
          b.unsafe_get(bias_offset + col + 2)
        } else {
          zero
        }
        @simd.f32x4_const(f0, f1, f2, zero)
      }
  }
}

///|
/// 将 6 个连续空间位置（`row .. row+5`）在 4 个输出通道（`col .. col+3`）上的累加结果
/// 直接转置写回 NCHW 布局 `out[out_offset + ch * m + row .. +5]`（每通道连续 6 次顺序写入）。
fn store_v128_6rows_nchw(
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  row : Int32,
  col : Int32,
  n : Int32,
  v0 : V128,
  v1 : V128,
  v2 : V128,
  v3 : V128,
  v4 : V128,
  v5 : V128,
) -> Unit {
  if col + 4 <= n {
    let r0 = out_offset + col * m + row
    out.unsafe_set(r0, @simd.f32x4_extract_lane(v0, 0))
    out.unsafe_set(r0 + 1, @simd.f32x4_extract_lane(v1, 0))
    out.unsafe_set(r0 + 2, @simd.f32x4_extract_lane(v2, 0))
    out.unsafe_set(r0 + 3, @simd.f32x4_extract_lane(v3, 0))
    out.unsafe_set(r0 + 4, @simd.f32x4_extract_lane(v4, 0))
    out.unsafe_set(r0 + 5, @simd.f32x4_extract_lane(v5, 0))

    let r1 = r0 + m
    out.unsafe_set(r1, @simd.f32x4_extract_lane(v0, 1))
    out.unsafe_set(r1 + 1, @simd.f32x4_extract_lane(v1, 1))
    out.unsafe_set(r1 + 2, @simd.f32x4_extract_lane(v2, 1))
    out.unsafe_set(r1 + 3, @simd.f32x4_extract_lane(v3, 1))
    out.unsafe_set(r1 + 4, @simd.f32x4_extract_lane(v4, 1))
    out.unsafe_set(r1 + 5, @simd.f32x4_extract_lane(v5, 1))

    let r2 = r1 + m
    out.unsafe_set(r2, @simd.f32x4_extract_lane(v0, 2))
    out.unsafe_set(r2 + 1, @simd.f32x4_extract_lane(v1, 2))
    out.unsafe_set(r2 + 2, @simd.f32x4_extract_lane(v2, 2))
    out.unsafe_set(r2 + 3, @simd.f32x4_extract_lane(v3, 2))
    out.unsafe_set(r2 + 4, @simd.f32x4_extract_lane(v4, 2))
    out.unsafe_set(r2 + 5, @simd.f32x4_extract_lane(v5, 2))

    let r3 = r2 + m
    out.unsafe_set(r3, @simd.f32x4_extract_lane(v0, 3))
    out.unsafe_set(r3 + 1, @simd.f32x4_extract_lane(v1, 3))
    out.unsafe_set(r3 + 2, @simd.f32x4_extract_lane(v2, 3))
    out.unsafe_set(r3 + 3, @simd.f32x4_extract_lane(v3, 3))
    out.unsafe_set(r3 + 4, @simd.f32x4_extract_lane(v4, 3))
    out.unsafe_set(r3 + 5, @simd.f32x4_extract_lane(v5, 3))
  } else {
    if col < n {
      let r0 = out_offset + col * m + row
      out.unsafe_set(r0, @simd.f32x4_extract_lane(v0, 0))
      out.unsafe_set(r0 + 1, @simd.f32x4_extract_lane(v1, 0))
      out.unsafe_set(r0 + 2, @simd.f32x4_extract_lane(v2, 0))
      out.unsafe_set(r0 + 3, @simd.f32x4_extract_lane(v3, 0))
      out.unsafe_set(r0 + 4, @simd.f32x4_extract_lane(v4, 0))
      out.unsafe_set(r0 + 5, @simd.f32x4_extract_lane(v5, 0))
    }
    if col + 1 < n {
      let r1 = out_offset + (col + 1) * m + row
      out.unsafe_set(r1, @simd.f32x4_extract_lane(v0, 1))
      out.unsafe_set(r1 + 1, @simd.f32x4_extract_lane(v1, 1))
      out.unsafe_set(r1 + 2, @simd.f32x4_extract_lane(v2, 1))
      out.unsafe_set(r1 + 3, @simd.f32x4_extract_lane(v3, 1))
      out.unsafe_set(r1 + 4, @simd.f32x4_extract_lane(v4, 1))
      out.unsafe_set(r1 + 5, @simd.f32x4_extract_lane(v5, 1))
    }
    if col + 2 < n {
      let r2 = out_offset + (col + 2) * m + row
      out.unsafe_set(r2, @simd.f32x4_extract_lane(v0, 2))
      out.unsafe_set(r2 + 1, @simd.f32x4_extract_lane(v1, 2))
      out.unsafe_set(r2 + 2, @simd.f32x4_extract_lane(v2, 2))
      out.unsafe_set(r2 + 3, @simd.f32x4_extract_lane(v3, 2))
      out.unsafe_set(r2 + 4, @simd.f32x4_extract_lane(v4, 2))
      out.unsafe_set(r2 + 5, @simd.f32x4_extract_lane(v5, 2))
    }
  }
}

///|
/// 将 4 个连续空间位置（`row .. row+3`）在 4 个输出通道（`col .. col+3`）上的累加结果
/// 直接转置写回 NCHW 布局 `out[out_offset + ch * m + row .. +3]`。
fn store_v128_4rows_nchw(
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  row : Int32,
  col : Int32,
  n : Int32,
  v0 : V128,
  v1 : V128,
  v2 : V128,
  v3 : V128,
) -> Unit {
  if col + 4 <= n {
    let r0 = out_offset + col * m + row
    out.unsafe_set(r0, @simd.f32x4_extract_lane(v0, 0))
    out.unsafe_set(r0 + 1, @simd.f32x4_extract_lane(v1, 0))
    out.unsafe_set(r0 + 2, @simd.f32x4_extract_lane(v2, 0))
    out.unsafe_set(r0 + 3, @simd.f32x4_extract_lane(v3, 0))

    let r1 = r0 + m
    out.unsafe_set(r1, @simd.f32x4_extract_lane(v0, 1))
    out.unsafe_set(r1 + 1, @simd.f32x4_extract_lane(v1, 1))
    out.unsafe_set(r1 + 2, @simd.f32x4_extract_lane(v2, 1))
    out.unsafe_set(r1 + 3, @simd.f32x4_extract_lane(v3, 1))

    let r2 = r1 + m
    out.unsafe_set(r2, @simd.f32x4_extract_lane(v0, 2))
    out.unsafe_set(r2 + 1, @simd.f32x4_extract_lane(v1, 2))
    out.unsafe_set(r2 + 2, @simd.f32x4_extract_lane(v2, 2))
    out.unsafe_set(r2 + 3, @simd.f32x4_extract_lane(v3, 2))

    let r3 = r2 + m
    out.unsafe_set(r3, @simd.f32x4_extract_lane(v0, 3))
    out.unsafe_set(r3 + 1, @simd.f32x4_extract_lane(v1, 3))
    out.unsafe_set(r3 + 2, @simd.f32x4_extract_lane(v2, 3))
    out.unsafe_set(r3 + 3, @simd.f32x4_extract_lane(v3, 3))
  } else {
    if col < n {
      let r0 = out_offset + col * m + row
      out.unsafe_set(r0, @simd.f32x4_extract_lane(v0, 0))
      out.unsafe_set(r0 + 1, @simd.f32x4_extract_lane(v1, 0))
      out.unsafe_set(r0 + 2, @simd.f32x4_extract_lane(v2, 0))
      out.unsafe_set(r0 + 3, @simd.f32x4_extract_lane(v3, 0))
    }
    if col + 1 < n {
      let r1 = out_offset + (col + 1) * m + row
      out.unsafe_set(r1, @simd.f32x4_extract_lane(v0, 1))
      out.unsafe_set(r1 + 1, @simd.f32x4_extract_lane(v1, 1))
      out.unsafe_set(r1 + 2, @simd.f32x4_extract_lane(v2, 1))
      out.unsafe_set(r1 + 3, @simd.f32x4_extract_lane(v3, 1))
    }
    if col + 2 < n {
      let r2 = out_offset + (col + 2) * m + row
      out.unsafe_set(r2, @simd.f32x4_extract_lane(v0, 2))
      out.unsafe_set(r2 + 1, @simd.f32x4_extract_lane(v1, 2))
      out.unsafe_set(r2 + 2, @simd.f32x4_extract_lane(v2, 2))
      out.unsafe_set(r2 + 3, @simd.f32x4_extract_lane(v3, 2))
    }
  }
}

///|
/// 将单个空间位置 `row` 在最多 4 个输出通道上的累加结果写回 NCHW 输出缓冲。
fn store_v128_1row_nchw(
  out : FixedArray[Float],
  out_offset : Int32,
  m : Int32,
  row : Int32,
  col : Int32,
  n : Int32,
  v : V128,
) -> Unit {
  if col < n {
    out.unsafe_set(out_offset + col * m + row, @simd.f32x4_extract_lane(v, 0))
  }
  if col + 1 < n {
    out.unsafe_set(
      out_offset + (col + 1) * m + row,
      @simd.f32x4_extract_lane(v, 1),
    )
  }
  if col + 2 < n {
    out.unsafe_set(
      out_offset + (col + 2) * m + row,
      @simd.f32x4_extract_lane(v, 2),
    )
  }
  if col + 3 < n {
    out.unsafe_set(
      out_offset + (col + 3) * m + row,
      @simd.f32x4_extract_lane(v, 3),
    )
  }
}
