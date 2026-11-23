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
