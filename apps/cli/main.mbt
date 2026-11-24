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
  println(
    "  moon-onnx compile <model.onnx> <out_dir> [--input-shape=1,1,28,28] [--struct=CompiledModel] [--backend=scalar|simd]",
  )
  println(
    "      AOT-compile an ONNX model into standalone zero-Map MoonBit code + weights.bin (onnx2mbt)",
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
/// 将逗号分隔的维度字符串（如 `"1,1,28,28"`）解析为整型数组。
fn parse_shape_csv(spec : String) -> Array[Int32] {
  let dims : Array[Int32] = []
  let mut cur = 0
  let mut has_digit = false
  for i = 0; i < spec.length(); i = i + 1 {
    let ch = spec.code_unit_at(i).to_int()
    if ch >= 48 && ch <= 57 {
      cur = cur * 10 + (ch - 48)
      has_digit = true
    } else if ch == 44 || ch == 120 || ch == 88 {
      // 支持 ',' / 'x' / 'X' 作为维度分隔符
      if has_digit {
        dims.push(cur)
        cur = 0
        has_digit = false
      }
    }
  }
  if has_digit {
    dims.push(cur)
  }
  dims
}

///|
/// 提取 `--key=value` 形式命令行参数的值。
fn find_flag_value(args : Array[String], prefix : String) -> String? {
  for i = 0; i < args.length(); i = i + 1 {
    if args[i].has_prefix(prefix) {
      let sb = StringBuilder()
      for k = prefix.length(); k < args[i].length(); k = k + 1 {
        sb.write_char(Int32::unsafe_to_char(args[i].code_unit_at(k).to_int()))
      }
      return Some(sb.to_string())
    }
  }
  None
}

///|
/// 执行 `compile` 子命令（`onnx2mbt`）：将 `.onnx` 模型静态编译为独立 MoonBit 源码 + `weights.bin`。
fn handle_compile(args : Array[String]) -> Unit {
  let pos = positional_args(args, 2)
  if pos.length() < 2 {
    println("Error: compile requires <model.onnx> and <out_dir>")
    print_usage()
    @env.exit(1)
    return
  }
  let model_path = pos[0]
  let out_dir = pos[1]
  let shape_spec = match find_flag_value(args, "--input-shape=") {
    Some(s) => s
    None => "1,1,28,28"
  }
  let struct_name = match find_flag_value(args, "--struct=") {
    Some(s) if s != "" => s
    _ => "CompiledModel"
  }
  let use_simd = match find_flag_value(args, "--backend=") {
    Some("scalar") => false
    _ => true
  }

  try {
    let m = @onnx.load_model_from_file(model_path)
    if m.inputs.length() == 0 {
      println("Error: model has no graph inputs")
      @env.exit(1)
      return
    }
    let in_dims = parse_shape_csv(shape_spec)
    let input_shapes : Map[String, Array[Int32]] = Map([])
    input_shapes[m.inputs[0].name] = in_dims

    let cfg : @codegen.CodegenConfig = {
      struct_name,
      use_simd,
      optimize: true,
    }
    let art = @onnx.compile_model_to_mbt(m, input_shapes, config=cfg)

    if !@fs.path_exists(out_dir) {
      @fs.create_dir(out_dir) catch {
        @fs.IOError(msg) => {
          println("Failed to create dir '\{out_dir}': \{msg}")
          @env.exit(1)
          return
        }
      }
    }
    @fs.write_string_to_file("\{out_dir}/model.mbt", art.mbt_source) catch {
      @fs.IOError(msg) => {
        println("Failed to write model.mbt: \{msg}")
        @env.exit(1)
        return
      }
    }
    @fs.write_string_to_file("\{out_dir}/moon.pkg", art.pkg_config) catch {
      @fs.IOError(msg) => {
        println("Failed to write moon.pkg: \{msg}")
        @env.exit(1)
        return
      }
    }
    @fs.write_bytes_to_file("\{out_dir}/weights.bin", art.weights_bin) catch {
      @fs.IOError(msg) => {
        println("Failed to write weights.bin: \{msg}")
        @env.exit(1)
        return
      }
    }
    println("onnx2mbt AOT Compilation Successful!")
    println("  Model        : \{model_path}")
    println("  Output Dir   : \{out_dir}")
    println("  Struct Name  : \{struct_name}")
    println("  SIMD Enabled : \{use_simd}")
    println("  Fused Nodes  : \{art.node_count}")
    println(
      "  Weights Blob : \{art.weights.length()} tensors (\{art.weights_bin.length()} bytes)",
    )
    println("  Generated    : model.mbt, moon.pkg, weights.bin")
  } catch {
    err => {
      println("Compile failed: \{err}")
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
    "compile" => handle_compile(args)
    other => {
      println("Error: unknown command '\{other}'")
      print_usage()
      @env.exit(1)
    }
  }
}
