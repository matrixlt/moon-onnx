///|
/// 打印 CLI 用法说明。
fn print_usage() -> Unit {
  println("\{@onnx.engine_version()} — ONNX Model Parser & Inference Engine")
  println("")
  println("Usage:")
  println("  moon-onnx inspect <model.onnx>")
  println("      Parse and display ONNX model metadata, weights, and graph")
  println(
    "  moon-onnx run <model.onnx> <input.f32> [golden.f32] [--backend=scalar|simd]",
  )
  println(
    "      Run MNIST [1,1,28,28] inference and optionally compare against ORT golden output",
  )
  println(
    "  moon-onnx profile <model.onnx> <input.f32> [--backend=scalar|simd]",
  )
  println(
    "      Run inference and print per-operator execution latency breakdown",
  )
  println("  moon-onnx version")
  println("      Show engine version")
}

///|
/// 解析命令行中的 `--backend=...` 选项，默认使用 scalar 后端。
fn parse_backend_flag(args : Array[String]) -> @onnx.BackendKind {
  for i = 0; i < args.length(); i = i + 1 {
    if args[i] == "--backend=simd" {
      return @session.Simd
    }
  }
  @session.Scalar
}

///|
/// 过滤掉以 `--` 开头的标志参数，提取位置参数列表。
fn positional_args(args : Array[String], start : Int32) -> Array[String] {
  let out : Array[String] = []
  for i = start; i < args.length(); i = i + 1 {
    if !args[i].has_prefix("--") {
      out.push(args[i])
    }
  }
  out
}

///|
/// 执行 `run` 子命令：加载模型与输入张量、执行推理并打印预测结果与误差统计。
fn handle_run(args : Array[String]) -> Unit {
  let pos = positional_args(args, 2)
  if pos.length() < 2 {
    println("Error: run requires <model.onnx> and <input.f32>")
    print_usage()
    @env.exit(1)
    return
  }
  let model_path = pos[0]
  let input_path = pos[1]
  let golden_path : String? = if pos.length() >= 3 {
    Some(pos[2])
  } else {
    None
  }
  let backend = parse_backend_flag(args)

  try {
    let session = @onnx.session_from_file(model_path, backend~)
    let input = @onnx.load_f32_file(input_path, [1, 1, 28, 28])
    let output = session.run_single(input)
    let (pred_cls, pred_val) = output.argmax_f32() catch {
      e => {
        println("Failed to compute argmax: \{e}")
        @env.exit(1)
        return
      }
    }
    println("Backend         : \{backend}")
    println("Model           : \{model_path}")
    println("Packed Weights  : \{session.packed_weight_count()}")
    println("Output          : \{output.summary()}")
    println("Predicted Class : \{pred_cls} (score=\{pred_val})")

    match golden_path {
      Some(gp) => {
        let golden = @onnx.load_f32_file(gp, output.shape())
        let max_diff = @onnx.max_abs_diff_f32(output, golden)
        let (ort_cls, _) = golden.argmax_f32() catch {
          _ => (-1, (0.0 : Float))
        }
        println("ORT Golden Class: \{ort_cls}")
        println("Max Abs Diff    : \{max_diff}")
        if pred_cls == ort_cls && max_diff <= (1.0e-4 : Float) {
          println("Verification    : PASS")
        } else {
          println("Verification    : FAIL")
          @env.exit(1)
        }
      }
      None => ()
    }
  } catch {
    err => {
      println("Inference failed: \{err}")
      @env.exit(1)
    }
  }
}

///|
/// 执行 `profile` 子命令：预热一次后输出逐算子耗时分布表。
fn handle_profile(args : Array[String]) -> Unit {
  let pos = positional_args(args, 2)
  if pos.length() < 2 {
    println("Error: profile requires <model.onnx> and <input.f32>")
    print_usage()
    @env.exit(1)
    return
  }
  let model_path = pos[0]
  let input_path = pos[1]
  let backend = parse_backend_flag(args)

  try {
    let session = @onnx.session_from_file(model_path, backend~)
    let input = @onnx.load_f32_file(input_path, [1, 1, 28, 28])
    // 先预热 1 次消除冷启动抖动
    session.run_single(input) |> ignore
    let (_, profile) = session.run_single_profiled(input)
    let ps = session.pool_stats()
    println(
      "Model: \{model_path} (packed_weights=\{session.packed_weight_count()}, packed_conv=\{session.packed_conv_weight_count()}, pool_hits=\{ps.hits}, pool_misses=\{ps.misses}, pooled_f32=\{ps.pooled_f32_buffers})",
    )
    println(profile.format_table())
  } catch {
    err => {
      println("Profile failed: \{err}")
      @env.exit(1)
    }
  }
}

///|
/// Native CLI 主入口。
fn main {
  let args = @env.args()
  if args.length() < 2 {
    print_usage()
    return
  }
  match args[1] {
    "version" | "--version" | "-v" => println(@onnx.engine_version())
    "inspect" => {
      if args.length() < 3 {
        println("Error: missing <model.onnx> path")
        print_usage()
        @env.exit(1)
        return
      }
      let path = args[2]
      try {
        let m = @onnx.load_model_from_file(path)
        println(m.format_summary())
      } catch {
        err => {
          println("Failed to inspect '\{path}': \{err}")
          @env.exit(1)
        }
      }
    }
    "run" => handle_run(args)
    "profile" => handle_profile(args)
    other => {
      println("Error: unknown command '\{other}'")
      print_usage()
      @env.exit(1)
    }
  }
}
