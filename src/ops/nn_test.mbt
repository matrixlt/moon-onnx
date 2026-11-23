///|
/// 测试 2D Conv（含 padding 与 bias）输出形状与手算卷积值一致。
test "conv2d_f32 computes valid and padded 2D convolution" {
  // 输入 [1, 1, 3, 3] = [[1,2,3],[4,5,6],[7,8,9]]
  let x = @tensor.Tensor::from_f32_array([1, 1, 3, 3], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  ])
  // 全 1 的 2x2 卷积核 [1, 1, 2, 2]
  let w = @tensor.Tensor::from_f32_array([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
  let b = @tensor.Tensor::from_f32_array([1], [0.5])

  let y = @ops.conv2d_f32(
    x,
    w,
    Some(b),
    [1, 1],
    [0, 0, 0, 0],
    [1, 1],
    1,
    @ir.NotSet,
  )
  assert_eq(y.shape(), [1, 1, 2, 2])
  // 1+2+4+5 + 0.5 = 12.5, 2+3+5+6 + 0.5 = 16.5, 4+5+7+8 + 0.5 = 24.5, 5+6+8+9 + 0.5 = 28.5
  assert_eq(y.get_f32([0, 0, 0, 0]), (12.5 : Float))
  assert_eq(y.get_f32([0, 0, 0, 1]), (16.5 : Float))
  assert_eq(y.get_f32([0, 0, 1, 0]), (24.5 : Float))
  assert_eq(y.get_f32([0, 0, 1, 1]), (28.5 : Float))
}

///|
/// 测试 MaxPool 与 AveragePool 的 2x2 stride=2 下采样结果。
test "maxpool_2d_f32 and avgpool_2d_f32" {
  let x = @tensor.Tensor::from_f32_array([1, 1, 4, 4], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    16.0,
  ])
  let mp = @ops.maxpool_2d_f32(
    x,
    [2, 2],
    [2, 2],
    [0, 0, 0, 0],
    [1, 1],
    false,
    @ir.NotSet,
  )
  assert_eq(mp.shape(), [1, 1, 2, 2])
  assert_eq(mp.get_f32([0, 0, 0, 0]), (6.0 : Float))
  assert_eq(mp.get_f32([0, 0, 0, 1]), (8.0 : Float))
  assert_eq(mp.get_f32([0, 0, 1, 0]), (14.0 : Float))
  assert_eq(mp.get_f32([0, 0, 1, 1]), (16.0 : Float))

  let ap = @ops.avgpool_2d_f32(
    x,
    [2, 2],
    [2, 2],
    [0, 0, 0, 0],
    false,
    false,
    @ir.NotSet,
  )
  assert_eq(ap.get_f32([0, 0, 0, 0]), (3.5 : Float))
  assert_eq(ap.get_f32([0, 0, 1, 1]), (13.5 : Float))
}

///|
/// 测试 BatchNormalization 通道仿射变换精度。
test "batch_norm_f32 normalizes per channel" {
  let x = @tensor.Tensor::from_f32_array([1, 2, 1, 2], [2.0, 4.0, 10.0, 14.0])
  let scale = @tensor.Tensor::from_f32_array([2], [2.0, 0.5])
  let bias = @tensor.Tensor::from_f32_array([2], [1.0, -1.0])
  let mean = @tensor.Tensor::from_f32_array([2], [3.0, 12.0])
  let variance = @tensor.Tensor::from_f32_array([2], [1.0, 4.0])

  let y = @ops.batch_norm_f32(
    x,
    scale,
    bias,
    mean,
    variance,
    epsilon=(0.0 : Float),
  )
  // ch0: 2*(x-3)/1 + 1 -> [-1.0, 3.0]
  // ch1: 0.5*(x-12)/2 - 1 -> [-1.5, -0.5]
  assert_eq(y.get_f32([0, 0, 0, 0]), (-1.0 : Float))
  assert_eq(y.get_f32([0, 0, 0, 1]), (3.0 : Float))
  assert_eq(y.get_f32([0, 1, 0, 0]), (-1.5 : Float))
  assert_eq(y.get_f32([0, 1, 0, 1]), (-0.5 : Float))
}
