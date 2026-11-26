///|
test "nn_ir/analysis infers shapes and mixed dtypes for DynamicQuantizeLinear + MatMulInteger" {
  let w = @tensor.Tensor::from_i8([8, 4], FixedArray::make(32, b'\x00'))
  let g : @nn_ir.IrGraph = {
    name: "q_analysis",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([("w", w)]),
    nodes: [
      {
        name: "dq",
        op: @nn_ir.DynamicQuantizeLinear,
        inputs: ["x"],
        outputs: ["xq", "x_sc", "x_zp"],
      },
      {
        name: "mm",
        op: @nn_ir.MatMulInteger,
        inputs: ["xq", "w", "x_zp"],
        outputs: ["acc"],
      },
      {
        name: "cast",
        op: @nn_ir.Cast(to=@tensor.Float32),
        inputs: ["acc"],
        outputs: ["y"],
      },
    ],
  }
  let shapes = @analysis.infer_static_shapes(g, Map([("x", [1, 6, 8])]))
  assert_eq(shapes.get("xq"), Some([1, 6, 8]))
  assert_eq(shapes.get("x_sc"), Some([]))
  assert_eq(shapes.get("acc"), Some([1, 6, 4]))
  assert_eq(shapes.get("y"), Some([1, 6, 4]))

  let dtypes = @analysis.infer_static_dtypes(g, Map([("x", @tensor.Float32)]))
  assert_eq(dtypes.get("xq"), Some(@tensor.UInt8))
  assert_eq(dtypes.get("x_sc"), Some(@tensor.Float32))
  assert_eq(dtypes.get("x_zp"), Some(@tensor.UInt8))
  assert_eq(dtypes.get("acc"), Some(@tensor.Int32))
  assert_eq(dtypes.get("y"), Some(@tensor.Float32))
}
