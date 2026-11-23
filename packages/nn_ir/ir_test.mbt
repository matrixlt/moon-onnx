///|
/// 验证 `topo_sort_nodes` 能自动纠正乱序节点排列。
test "topo_sort_nodes reorders out-of-order nodes" {
  let n_relu : @nn_ir.IrNode = {
    name: "relu_after",
    op: @nn_ir.Relu,
    inputs: ["mid"],
    outputs: ["out"],
  }
  let n_add : @nn_ir.IrNode = {
    name: "add_before",
    op: @nn_ir.Add,
    inputs: ["x", "x"],
    outputs: ["mid"],
  }
  let sorted = @nn_ir.topo_sort_nodes([n_relu, n_add], ["x"], Map([]))
  assert_eq(sorted[0].name, "add_before")
  assert_eq(sorted[1].name, "relu_after")
}

///|
/// 验证存在环路的计算图会被准确检测并抛出 `CyclicGraph` 错误。
test "topo_sort_nodes detects cycles" {
  let n1 : @nn_ir.IrNode = {
    name: "n1",
    op: @nn_ir.Relu,
    inputs: ["b"],
    outputs: ["a"],
  }
  let n2 : @nn_ir.IrNode = {
    name: "n2",
    op: @nn_ir.Relu,
    inputs: ["a"],
    outputs: ["b"],
  }
  let mut caught_cycle = false
  try @nn_ir.topo_sort_nodes([n1, n2], [], Map([])) |> ignore catch {
    @nn_ir.CyclicGraph(_) => caught_cycle = true
    _ => ()
  }
  assert_true(caught_cycle)
}
