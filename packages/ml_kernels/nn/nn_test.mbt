///|
test "ml_kernels/nn zero-allocation kernels match analytical references" {
  // 1. Erf
  let x : FixedArray[Float] = [0.0, 1.0, -1.0]
  let y = FixedArray::make(3, (0.0 : Float))
  @nn.erf_f32(x, 0, y, 0, 3)
  assert_true(y[0].abs() < (1.0e-6 : Float))
  assert_true((y[1] - (0.8427008 : Float)).abs() < (1.0e-5 : Float))
  assert_true((y[2] + (0.8427008 : Float)).abs() < (1.0e-5 : Float))

  // 2. Softmax
  let sx : FixedArray[Float] = [1.0, 2.0, 3.0]
  let sy = FixedArray::make(3, (0.0 : Float))
  @nn.softmax_f32(sx, 0, sy, 0, 1, 3, 1)
  assert_true((sy[0] + sy[1] + sy[2] - (1.0 : Float)).abs() < (1.0e-6 : Float))
  assert_true(sy[2] > sy[1] && sy[1] > sy[0])

  // 3. LayerNorm
  let lx : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0]
  let sc : FixedArray[Float] = [1.0, 1.0, 1.0, 1.0]
  let ly = FixedArray::make(4, (0.0 : Float))
  @nn.layernorm_f32(lx, 0, sc, None, ly, 0, 1, 4, (1.0e-5 : Float))
  assert_true((ly[0] + ly[3]).abs() < (1.0e-5 : Float))

  // 4. reflect_idx
  assert_eq(@nn.reflect_idx(-1, 4), 1)
  assert_eq(@nn.reflect_idx(4, 4), 2)
  assert_eq(@nn.reflect_idx(0, 1), 0)

  // 5. GAP
  let gx : FixedArray[Float] = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
  let gy = FixedArray::make(2, (0.0 : Float))
  @nn.gap_f32(gx, 0, gy, 0, 2, 4)
  assert_true((gy[0] - (2.5 : Float)).abs() < (1.0e-6 : Float))
  assert_true((gy[1] - (25.0 : Float)).abs() < (1.0e-6 : Float))
}
