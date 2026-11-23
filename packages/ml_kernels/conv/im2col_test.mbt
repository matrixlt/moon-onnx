///|
/// 验证 3x3 特征图按 2x2 核（stride=1, pad=0）展开为 [4, 4] im2col 矩阵的精确布局。
test "im2col_2d_f32 unfolds 3x3 input with 2x2 kernel accurately" {
  let inp = FixedArray::makei(9, fn(i) { Float::from_int(i + 1) })
  let col = FixedArray::make(4 * 4, (0.0 : Float))
  @conv.im2col_2d_f32(inp, 0, 1, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, col)
  // 第 0 行 (ky=0, kx=0): [1, 2, 4, 5]
  // 第 3 行 (ky=1, kx=1): [5, 6, 8, 9]
  assert_eq(col[0], (1.0 : Float))
  assert_eq(col[1], (2.0 : Float))
  assert_eq(col[2], (4.0 : Float))
  assert_eq(col[3], (5.0 : Float))
  assert_eq(col[12], (5.0 : Float))
  assert_eq(col[15], (9.0 : Float))
}

///|
/// 验证 `im2col_2d_nhwc_f32` 在常规填充、步长、空洞及 1×1 点卷积场景下均与 `im2col_2d_f32` 转置精确一致。
test "im2col_2d_nhwc_f32 matches transpose of im2col_2d_f32 across configs" {
  // (channels, in_h, in_w, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w, out_h, out_w)
  let configs = [
    (1, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2),
    (2, 6, 6, 5, 5, 1, 1, 2, 2, 1, 1, 6, 6),
    (3, 7, 5, 3, 3, 2, 1, 1, 1, 1, 2, 4, 3),
    (4, 5, 4, 1, 1, 1, 1, 0, 0, 1, 1, 5, 4),
    (2, 3, 3, 5, 5, 1, 1, 2, 2, 1, 1, 3, 3),
  ]
  for idx = 0; idx < configs.length(); idx = idx + 1 {
    let (c, ih, iw, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow) = configs[idx]
    let inp = FixedArray::makei(c * ih * iw, fn(i) {
      Float::from_int((i * 7 + 3) % 29 - 14) * (0.25 : Float)
    })
    let k = c * kh * kw
    let s = oh * ow
    let col = FixedArray::make(k * s, (-99.0 : Float))
    let col_t = FixedArray::make(s * k, (-99.0 : Float))
    @conv.im2col_2d_f32(
      inp, 0, c, ih, iw, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow, col,
    )
    @conv.im2col_2d_nhwc_f32(
      inp, 0, c, ih, iw, kh, kw, sh, sw, pt, pl, dh, dw, oh, ow, col_t,
    )
    for row = 0; row < s; row = row + 1 {
      for col_i = 0; col_i < k; col_i = col_i + 1 {
        assert_eq(col_t[row * k + col_i], col[col_i * s + row])
      }
    }
  }
}
