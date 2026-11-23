///|
/// 打印 CLI 用法说明。
fn print_usage() -> Unit {
  println(
    "\{@engine.engine_version()} — ONNX Model Parser & Inference Engine",
  )
  println("")
  println("Usage:")
  println(
    "  moon-onnx inspect <model.onnx>   Parse and display model structure",
  )
  println("  moon-onnx version                Show engine version")
}

///|
/// Native CLI 主入口：当前支持 `inspect` 与 `version` 子命令，后续扩展 `run` 与 `bench`。
fn main {
  let args = @env.args()
  // args[0] 为可执行文件路径，真实子命令从 args[1] 开始
  if args.length() < 2 {
    print_usage()
    return
  }
  match args[1] {
    "version" | "--version" | "-v" => println(@engine.engine_version())
    "inspect" => {
      if args.length() < 3 {
        println("Error: missing <model.onnx> path")
        print_usage()
        @env.exit(1)
        return
      }
      let path = args[2]
      try {
        let m = @model.load_model_from_file(path)
        println(m.format_summary())
      } catch {
        err => {
          println("Failed to inspect '\{path}': \{err}")
          @env.exit(1)
        }
      }
    }
    other => {
      println("Error: unknown command '\{other}'")
      print_usage()
      @env.exit(1)
    }
  }
}
