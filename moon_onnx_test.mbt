///|
/// 根包基础冒烟测试，验证测试框架与 Native 目标编译正常。
test "engine_version returns non-empty version string" {
  assert_eq(@moon_onnx.engine_version(), "moon-onnx 0.1.0")
}
