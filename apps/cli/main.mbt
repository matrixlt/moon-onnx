///|
/// 打印 CLI 用法说明。
fn print_usage() -> Unit {
  println("\{@onnx.engine_version()} — ONNX Model Parser & Inference Engine")
  println("")
  println("Usage:")
  println("  moon-onnx inspect <model.onnx>")
  println("      Parse and display ONNX model metadata, weights, and graph")
  println(
    "  moon-onnx run <model.onnx> <input.f32> [golden.f32] [--input-shape=1,1,28,28] [--backend=scalar|simd]",
  )
  println(
    "      Run single-input inference and optionally compare against ORT golden output",
  )
  println(
    "  moon-onnx profile <model.onnx> <input.f32> [--input-shape=1,1,28,28] [--backend=scalar|simd]",
  )
  println(
    "      Run inference and print per-operator execution latency breakdown",
  )
  println(
    "  moon-onnx compile <model.onnx> <out_dir> [--input-shape=1,1,28,28] [--struct=CompiledModel] [--target=wasm|wasm-gc|native] [--mode=arena|tensor] [--embed-weights]",
  )
  println(
    "      AOT-compile an ONNX model into zero-allocation Arena MoonBit code + weights.bin (or embedded single-file Wasm)",
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
  let shape_spec = match find_flag_value(args, "--input-shape=") {
    Some(s) => s
    None => "1,1,28,28"
  }
  let in_dims = parse_shape_csv(shape_spec)

  try {
    let session = @onnx.session_from_file(model_path, backend~)
    let input = @onnx.load_f32_file(input_path, in_dims)
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
  let shape_spec = match find_flag_value(args, "--input-shape=") {
    Some(s) => s
    None => "1,1,28,28"
  }
  let in_dims = parse_shape_csv(shape_spec)

  try {
    let session = @onnx.session_from_file(model_path, backend~)
    let input = @onnx.load_f32_file(input_path, in_dims)
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
/// 检查命令行参数数组中是否存在指定布尔开关标志。
fn has_flag(args : Array[String], flag : String) -> Bool {
  for i = 0; i < args.length(); i = i + 1 {
    if args[i] == flag {
      return true
    }
  }
  false
}

///|
/// 执行 `compile` 子命令（`onnx2mbt`）：将 `.onnx` 模型静态编译为独立零分配 MoonBit 源码 + `weights.bin`。
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
  let target : @codegen.CodegenTarget = match
    find_flag_value(args, "--target=") {
    Some("wasm-gc") => @codegen.TargetWasmGc
    Some("native") => @codegen.TargetNative
    _ => @codegen.TargetWasm
  }
  let mode : @codegen.CodegenMode = match find_flag_value(args, "--mode=") {
    Some("tensor") => @codegen.Tensor
    _ => @codegen.Arena
  }
  let embed_weights = has_flag(args, "--embed-weights")

  try {
    let m = @onnx.load_model_from_file(model_path)
    if m.inputs.length() == 0 {
      println("Error: model has no graph inputs")
      @env.exit(1)
      return
    }
    let input_shapes : Map[String, Array[Int32]] = Map([])
    // 支持单输入 `--input-shape=1,3,224,224` 与多输入 `--input-shape=input:1,512;state:2,1,128`
    if shape_spec.contains(":") {
      let mut seg_start = 0
      for idx = 0; idx <= shape_spec.length(); idx = idx + 1 {
        let is_end = idx == shape_spec.length() ||
          shape_spec.code_unit_at(idx).to_int() == 59 // ';'
        if is_end && idx > seg_start {
          let name_sb = StringBuilder()
          let dims_sb = StringBuilder()
          let mut seen_colon = false
          for k = seg_start; k < idx; k = k + 1 {
            let cu = shape_spec.code_unit_at(k).to_int()
            if cu == 58 && !seen_colon {
              seen_colon = true
            } else if !seen_colon {
              name_sb.write_char(Int32::unsafe_to_char(cu))
            } else {
              dims_sb.write_char(Int32::unsafe_to_char(cu))
            }
          }
          let in_n = name_sb.to_string()
          if in_n != "" {
            input_shapes[in_n] = parse_shape_csv(dims_sb.to_string())
          }
          seg_start = idx + 1
        }
      }
    } else {
      input_shapes[m.inputs[0].name] = parse_shape_csv(shape_spec)
    }

    // 解析 `--freeze-i64=name:val`（支持多个以 ';' 分隔，例如 `sr:16000`）
    let frozen_inputs : Map[String, @tensor.Tensor] = Map([])
    match find_flag_value(args, "--freeze-i64=") {
      Some(spec) => {
        let mut seg_start = 0
        for idx = 0; idx <= spec.length(); idx = idx + 1 {
          let is_end = idx == spec.length() ||
            spec.code_unit_at(idx).to_int() == 59
          if is_end && idx > seg_start {
            let name_sb = StringBuilder()
            let mut val : Int64 = 0
            let mut neg = false
            let mut seen_colon = false
            for k = seg_start; k < idx; k = k + 1 {
              let cu = spec.code_unit_at(k).to_int()
              if cu == 58 && !seen_colon {
                seen_colon = true
              } else if !seen_colon {
                name_sb.write_char(Int32::unsafe_to_char(cu))
              } else if cu == 45 {
                neg = true
              } else if cu >= 48 && cu <= 57 {
                val = val * (10 : Int64) + (cu - 48).to_int64()
              }
            }
            let fn_name = name_sb.to_string()
            if fn_name != "" && seen_colon {
              let final_v = if neg { -val } else { val }
              let t = @tensor.Tensor::from_i64([], FixedArray::make(1, final_v)) catch {
                e => {
                  println("Failed to freeze input '\{fn_name}': \{e}")
                  @env.exit(1)
                  return
                }
              }
              frozen_inputs[fn_name] = t
            }
            seg_start = idx + 1
          }
        }
      }
      None => ()
    }

    let cfg : @codegen.CodegenConfig = {
      struct_name,
      use_simd,
      optimize: true,
      target,
      mode,
      embed_weights,
      frozen_inputs,
      input_dtypes: Map([]),
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
    if !embed_weights {
      @fs.write_bytes_to_file("\{out_dir}/weights.bin", art.weights_bin) catch {
        @fs.IOError(msg) => {
          println("Failed to write weights.bin: \{msg}")
          @env.exit(1)
          return
        }
      }
    }
    println("onnx2mbt v2 AOT Compilation Successful!")
    println("  Model        : \{model_path}")
    println("  Output Dir   : \{out_dir}")
    println("  Struct Name  : \{struct_name}")
    println(
      "  Codegen Mode : \{art.mode_used} | Target: \{target} | SIMD: \{use_simd}",
    )
    println("  Fused Nodes  : \{art.node_count}")
    if art.mode_used == @codegen.Arena {
      let peak_b = art.arena_f32_elems * 4
      let naive_b = art.arena_f32_naive_elems * 4
      let scratch_b = art.scratch_f32_elems * 4
      println(
        "  Static Arena : \{peak_b} B peak vs \{naive_b} B naive (Byte Arena: \{art.arena_byte_elems} B, Shared Scratch: \{scratch_b} B)",
      )
    }
    println(
      "  Weights Blob : \{art.weights.length()} tensors (\{art.weights_bin.length()} bytes, embedded=\{embed_weights})",
    )
    if embed_weights {
      println("  Generated    : model.mbt (weights embedded), moon.pkg")
    } else {
      println("  Generated    : model.mbt, moon.pkg, weights.bin")
    }
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
