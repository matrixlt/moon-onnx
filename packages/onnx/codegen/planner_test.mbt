///|
/// 验证编译期静态内存规划器在链式网络、残差分支与 Reshape 零拷贝别名三种场景下的内存复用与安全隔离。
test "plan_static_arenas reuses dead intervals safely across chains, residuals, and aliases" {
  // 1. 4 层链式激活：x -> a(100) -> b(80) -> c(100) -> d(50)
  //    a 在第 1 步后死亡，c 在第 2 步诞生，可完美复用 a 的偏移 0，峰值仅 180（朴素需 330，节省 > 45%）
  let g_chain : @nn_ir.IrGraph = {
    name: "chain",
    inputs: ["x"],
    outputs: ["d"],
    initializers: Map([]),
    nodes: [
      { name: "n0", op: @nn_ir.Relu, inputs: ["x"], outputs: ["a"], },
      { name: "n1", op: @nn_ir.Relu, inputs: ["a"], outputs: ["b"], },
      { name: "n2", op: @nn_ir.Relu, inputs: ["b"], outputs: ["c"], },
      { name: "n3", op: @nn_ir.Relu, inputs: ["c"], outputs: ["d"], },
    ],
  }
  let sh_chain : Map[String, Array[Int32]] = Map([
    ("x", [100]),
    ("a", [100]),
    ("b", [80]),
    ("c", [100]),
    ("d", [50]),
  ])
  let dt_chain = @codegen.infer_static_dtypes(g_chain, Map([]))
  let (f32_plan, _) = @codegen.plan_static_arenas(g_chain, sh_chain, dt_chain)
  assert_eq(f32_plan.naive_elems, 330)
  assert_true(f32_plan.peak_elems <= 184)
  assert_true(f32_plan.savings_pct() > 40.0)

  // 验证相邻存活区间绝不重叠（a 与 b、b 与 c、c 与 d 各自物理范围互斥）
  let aa = f32_plan.allocs.get("a").unwrap()
  let ab = f32_plan.allocs.get("b").unwrap()
  let ac = f32_plan.allocs.get("c").unwrap()
  let ad = f32_plan.allocs.get("d").unwrap()
  assert_true(
    aa.offset + aa.length <= ab.offset || ab.offset + ab.length <= aa.offset,
  )
  assert_true(
    ab.offset + ab.length <= ac.offset || ac.offset + ac.length <= ab.offset,
  )
  assert_true(
    ac.offset + ac.length <= ad.offset || ad.offset + ad.length <= ac.offset,
  )

  // 2. 残差分支 + Reshape 别名：h0 经 Reshape 别名 h0_flat 后在第 3 步被残差 Add 消费，
  //    规划器必须确保 h0 在第 1~2 步（h1 / h2 计算期间）绝不被覆盖。
  let g_res : @nn_ir.IrGraph = {
    name: "residual_alias",
    inputs: ["x"],
    outputs: ["y"],
    initializers: Map([]),
    nodes: [
      { name: "n0", op: @nn_ir.Relu, inputs: ["x"], outputs: ["h0"], },
      {
        name: "n1",
        op: @nn_ir.Reshape(allowzero=false),
        inputs: ["h0", "sh"],
        outputs: ["h0_alias"],
      },
      { name: "n2", op: @nn_ir.Sigmoid, inputs: ["h0_alias"], outputs: ["h1"], },
      {
        name: "n3",
        op: @nn_ir.Add,
        inputs: ["h0_alias", "h1"],
        outputs: ["y"],
      },
    ],
  }
  let sh_res : Map[String, Array[Int32]] = Map([
    ("x", [1, 16]),
    ("h0", [1, 16]),
    ("h0_alias", [16]),
    ("h1", [16]),
    ("y", [16]),
  ])
  let dt_res = @codegen.infer_static_dtypes(g_res, Map([]))
  let (res_plan, _) = @codegen.plan_static_arenas(g_res, sh_res, dt_res)
  let r_h0 = res_plan.allocs.get("h0").unwrap()
  let r_alias = res_plan.allocs.get("h0_alias").unwrap()
  let r_h1 = res_plan.allocs.get("h1").unwrap()
  // Reshape 别名与原张量共享完全一致的物理偏移
  assert_eq(r_alias.offset, r_h0.offset)
  // h0 存活至第 3 步，与第 2 步诞生的 h1 重叠，绝不能共享区间
  assert_eq(r_h0.death, 3)
  assert_true(
    r_h0.offset + r_h0.length <= r_h1.offset ||
    r_h1.offset + r_h1.length <= r_h0.offset,
  )
}
