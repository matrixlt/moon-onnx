///|
/// 4-wide `V128` 向量化快速指数函数 `exp(x)`（Cody-Waite 范围归约 + 5 阶 Horner 多项式逼近）。
///
/// 算法步骤（参考 Cephes / Eigen / RTen vecmath）：
/// 1. 将输入钳位在 `[-87.33654, 88.37626]`，防止 IEEE-754 单精度指数位溢出产生非规格化噪声；
/// 2. 计算 `k = round(x / ln(2))` 与归约余数 `r = x - k * ln2_hi - k * ln2_lo`（`r ∈ [-ln2/2, ln2/2]`）；
/// 3. 在 `r` 上使用 5 阶泰勒/极小极大逼近多项式 `p(r) ≈ exp(r)`（全链路 `f32x4_relaxed_madd`）；
/// 4. 通过 IEEE-754 阶码位操作 `((int32(k) + 127) << 23)` 零开销构造 `2^k` 并与 `p(r)` 相乘。
///
/// 精度：在常规区间 `[-20, 20]` 内相对误差 `< 1e-6`，满足全部深度学习激活函数（Sigmoid/Tanh/Softmax/Gelu）精度要求。
pub fn exp_f32x4(x : V128) -> V128 {
  let lo_clamp = @simd.f32x4_splat((-87.33654 : Float))
  let hi_clamp = @simd.f32x4_splat((88.37626 : Float))
  let xc = @simd.f32x4_pmax(lo_clamp, @simd.f32x4_pmin(hi_clamp, x))

  let log2e = @simd.f32x4_splat((1.4426950408889634 : Float))
  // Cody-Waite 高低双段 ln(2) 常量，减少大 |x| 时的范围归约相消误差
  let neg_ln2_hi = @simd.f32x4_splat((-0.693145751953125 : Float))
  let neg_ln2_lo = @simd.f32x4_splat((-1.428606765330187e-6 : Float))

  let kf = @simd.f32x4_nearest(@simd.f32x4_mul(xc, log2e))
  let r0 = @simd.f32x4_relaxed_madd(kf, neg_ln2_hi, xc)
  let r = @simd.f32x4_relaxed_madd(kf, neg_ln2_lo, r0)

  // 6 阶多项式：1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6!
  // 在 |r| <= ln(2)/2 ≈ 0.3466 上截断误差约 1.2e-7，达到 Float32 单精度全精度
  let c6 = @simd.f32x4_splat((0.0013888889225199819 : Float))
  let c5 = @simd.f32x4_splat((0.008333333767950535 : Float))
  let c4 = @simd.f32x4_splat((0.0416666679084301 : Float))
  let c3 = @simd.f32x4_splat((0.1666666716337204 : Float))
  let c2 = @simd.f32x4_splat((0.5 : Float))
  let one = @simd.f32x4_splat((1.0 : Float))

  let p = @simd.f32x4_relaxed_madd(c6, r, c5)
  let p = @simd.f32x4_relaxed_madd(p, r, c4)
  let p = @simd.f32x4_relaxed_madd(p, r, c3)
  let p = @simd.f32x4_relaxed_madd(p, r, c2)
  let p = @simd.f32x4_relaxed_madd(p, r, one)
  let poly = @simd.f32x4_relaxed_madd(p, r, one)

  // 构造 2^k：IEEE-754 单精度偏移阶码 = (k + 127) << 23
  let bias = @simd.i32x4_splat(127U)
  let ki = @simd.i32x4_trunc_sat_f32x4_s(kf)
  let pow2k = @simd.i32x4_shl(@simd.i32x4_add(ki, bias), 23)
  @simd.f32x4_mul(poly, pow2k)
}

///|
/// 4-wide `V128` 向量化 Sigmoid：`1 / (1 + exp(-x))`。
pub fn sigmoid_f32x4(x : V128) -> V128 {
  let one = @simd.f32x4_splat((1.0 : Float))
  let denom = @simd.f32x4_add(one, exp_f32x4(@simd.f32x4_neg(x)))
  @simd.f32x4_div(one, denom)
}

///|
/// 4-wide `V128` 向量化 Tanh：`2 * sigmoid(2x) - 1`，在 `|x| > 10` 时平滑饱和至 `±1.0`。
pub fn tanh_f32x4(x : V128) -> V128 {
  let two = @simd.f32x4_splat((2.0 : Float))
  let one = @simd.f32x4_splat((1.0 : Float))
  let s = sigmoid_f32x4(@simd.f32x4_mul(two, x))
  @simd.f32x4_sub(@simd.f32x4_mul(two, s), one)
}

///|
/// 零转换字节缓冲版 V128 向量 Clip（`clamp(x, min_val, max_val)`）。
pub fn clip_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
  min_val : Float,
  max_val : Float,
) -> Unit {
  let vmin = @simd.f32x4_splat(min_val)
  let vmax = @simd.f32x4_splat(max_val)
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    let clamped = @simd.f32x4_pmax(vmin, @simd.f32x4_pmin(vmax, v))
    @simd.v128_store(out, off, clamped)
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let x = read_f32_le(src, off)
    let c = if x < min_val {
      min_val
    } else if x > max_val {
      max_val
    } else {
      x
    }
    write_f32_le(out, off, c)
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 Clip。
pub fn clip_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
  min_val : Float,
  max_val : Float,
) -> Unit {
  let vmin = @simd.f32x4_splat(min_val)
  let vmax = @simd.f32x4_splat(max_val)
  let mut i = 0
  while i + 4 <= n {
    let v = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let r = @simd.f32x4_pmax(vmin, @simd.f32x4_pmin(vmax, v))
    out[i] = @simd.f32x4_extract_lane(r, 0)
    out[i + 1] = @simd.f32x4_extract_lane(r, 1)
    out[i + 2] = @simd.f32x4_extract_lane(r, 2)
    out[i + 3] = @simd.f32x4_extract_lane(r, 3)
    i = i + 4
  }
  while i < n {
    let x = src[i]
    out[i] = if x < min_val {
      min_val
    } else if x > max_val {
      max_val
    } else {
      x
    }
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量 `exp(x)`。
pub fn exp_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    @simd.v128_store(out, off, exp_f32x4(v))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let x = read_f32_le(src, off)
    let r = @simd.f32x4_extract_lane(exp_f32x4(@simd.f32x4_splat(x)), 0)
    write_f32_le(out, off, r)
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量 `sigmoid(x)`。
pub fn sigmoid_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    @simd.v128_store(out, off, sigmoid_f32x4(v))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let x = read_f32_le(src, off)
    let r = @simd.f32x4_extract_lane(sigmoid_f32x4(@simd.f32x4_splat(x)), 0)
    write_f32_le(out, off, r)
    i = i + 1
  }
}

///|
/// 零转换字节缓冲版 V128 向量 `tanh(x)`。
pub fn tanh_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    @simd.v128_store(out, off, tanh_f32x4(v))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let x = read_f32_le(src, off)
    let r = @simd.f32x4_extract_lane(tanh_f32x4(@simd.f32x4_splat(x)), 0)
    write_f32_le(out, off, r)
    i = i + 1
  }
}

///|
/// 4-wide `V128` 向量化近似 GELU（tanh 近似，对应 ONNX `Gelu(approximate="tanh")` 与 GPT/ViT 标准实现）：
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`。
pub fn gelu_f32x4(x : V128) -> V128 {
  let half = @simd.f32x4_splat((0.5 : Float))
  let one = @simd.f32x4_splat((1.0 : Float))
  let sqrt_2_over_pi = @simd.f32x4_splat((0.7978845608028654 : Float))
  let coef = @simd.f32x4_splat((0.044715 : Float))
  let x2 = @simd.f32x4_mul(x, x)
  let x3 = @simd.f32x4_mul(x2, x)
  let inner = @simd.f32x4_mul(
    sqrt_2_over_pi,
    @simd.f32x4_relaxed_madd(coef, x3, x),
  )
  let t = tanh_f32x4(inner)
  @simd.f32x4_mul(@simd.f32x4_mul(half, x), @simd.f32x4_add(one, t))
}

///|
/// 4-wide `V128` 向量化 HardSwish：`x * clamp(x / 6 + 0.5, 0, 1)`（MobileNetV3 核心激活）。
pub fn hard_swish_f32x4(x : V128) -> V128 {
  let inv6 = @simd.f32x4_splat((0.16666667 : Float))
  let half = @simd.f32x4_splat((0.5 : Float))
  let zero = @simd.f32x4_splat((0.0 : Float))
  let one = @simd.f32x4_splat((1.0 : Float))
  let gate = @simd.f32x4_pmax(
    zero,
    @simd.f32x4_pmin(one, @simd.f32x4_relaxed_madd(x, inv6, half)),
  )
  @simd.f32x4_mul(x, gate)
}

///|
/// 零转换字节缓冲版 V128 向量 GELU。
pub fn gelu_bytes_v128(
  src : FixedArray[Byte],
  out : FixedArray[Byte],
  count : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= count {
    let off = i * 4
    let v = @simd.v128_load(src, off)
    @simd.v128_store(out, off, gelu_f32x4(v))
    i = i + 4
  }
  while i < count {
    let off = i * 4
    let x = read_f32_le(src, off)
    let r = @simd.f32x4_extract_lane(gelu_f32x4(@simd.f32x4_splat(x)), 0)
    write_f32_le(out, off, r)
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 `exp(x)`。
pub fn exp_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let vr = exp_f32x4(vx)
    out[i] = @simd.f32x4_extract_lane(vr, 0)
    out[i + 1] = @simd.f32x4_extract_lane(vr, 1)
    out[i + 2] = @simd.f32x4_extract_lane(vr, 2)
    out[i + 3] = @simd.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = @simd.f32x4_extract_lane(exp_f32x4(@simd.f32x4_splat(src[i])), 0)
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 `sigmoid(x)`。
pub fn sigmoid_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let vr = sigmoid_f32x4(vx)
    out[i] = @simd.f32x4_extract_lane(vr, 0)
    out[i + 1] = @simd.f32x4_extract_lane(vr, 1)
    out[i + 2] = @simd.f32x4_extract_lane(vr, 2)
    out[i + 3] = @simd.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = @simd.f32x4_extract_lane(
      sigmoid_f32x4(@simd.f32x4_splat(src[i])),
      0,
    )
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 `tanh(x)`。
pub fn tanh_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let vr = tanh_f32x4(vx)
    out[i] = @simd.f32x4_extract_lane(vr, 0)
    out[i + 1] = @simd.f32x4_extract_lane(vr, 1)
    out[i + 2] = @simd.f32x4_extract_lane(vr, 2)
    out[i + 3] = @simd.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = @simd.f32x4_extract_lane(tanh_f32x4(@simd.f32x4_splat(src[i])), 0)
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 `gelu(x)`（tanh 近似）。
pub fn gelu_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let vr = gelu_f32x4(vx)
    out[i] = @simd.f32x4_extract_lane(vr, 0)
    out[i + 1] = @simd.f32x4_extract_lane(vr, 1)
    out[i + 2] = @simd.f32x4_extract_lane(vr, 2)
    out[i + 3] = @simd.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = @simd.f32x4_extract_lane(gelu_f32x4(@simd.f32x4_splat(src[i])), 0)
    i = i + 1
  }
}

///|
/// 面向 `FixedArray[Float]` 的 V128 向量 `hard_swish(x)`。
pub fn hard_swish_f32_v128(
  src : FixedArray[Float],
  out : FixedArray[Float],
  n : Int32,
) -> Unit {
  let mut i = 0
  while i + 4 <= n {
    let vx = @simd.f32x4_const(src[i], src[i + 1], src[i + 2], src[i + 3])
    let vr = hard_swish_f32x4(vx)
    out[i] = @simd.f32x4_extract_lane(vr, 0)
    out[i + 1] = @simd.f32x4_extract_lane(vr, 1)
    out[i + 2] = @simd.f32x4_extract_lane(vr, 2)
    out[i + 3] = @simd.f32x4_extract_lane(vr, 3)
    i = i + 4
  }
  while i < n {
    out[i] = @simd.f32x4_extract_lane(
      hard_swish_f32x4(@simd.f32x4_splat(src[i])),
      0,
    )
    i = i + 1
  }
}
