///|
/// AOT 静态编译阶段可能抛出的结构化错误。
#derive(Debug, Eq)
pub(all) suberror CodegenError {
  /// 计算图包含当前 AOT 编译器尚未支持生成代码的算子
  UnsupportedOp(String)
  /// 图结构不符合静态编译要求（如缺少输入形状或非常量形状输入）
  UnsupportedGraph(String)
  /// 静态形状推导失败
  ShapeError(String)
  /// 权重序列化或张量处理失败
  TensorErr(String)
}

///|
/// 为 CodegenError 提供清晰可读的错误描述。
pub impl Show for CodegenError with fn output(self, logger) {
  match self {
    UnsupportedOp(msg) => logger.write_string("UnsupportedOp(\{msg})")
    UnsupportedGraph(msg) => logger.write_string("UnsupportedGraph(\{msg})")
    ShapeError(msg) => logger.write_string("ShapeError(\{msg})")
    TensorErr(msg) => logger.write_string("TensorErr(\{msg})")
  }
}

///|
/// AOT 编译目标平台（用于自适应选择最优 SIMD / 标量内核路径）：
/// - `TargetWasm`   ：线性内存 WebAssembly，启用预打包字节 V128 SIMD（19×~25× 提速）
/// - `TargetWasmGc` ：Wasm-GC 托管数组目标，避开 GC 字节数组 `v128_load` 退化，采用原生 Float 数组路径
/// - `TargetNative` ：本地 Native 目标，启用完整 V128 SIMD 与预打包权重
#derive(Debug, Eq)
pub(all) enum CodegenTarget {
  TargetWasm
  TargetWasmGc
  TargetNative
}

///|
pub impl Show for CodegenTarget with fn output(self, logger) {
  match self {
    TargetWasm => logger.write_string("wasm")
    TargetWasmGc => logger.write_string("wasm-gc")
    TargetNative => logger.write_string("native")
  }
}

///|
/// AOT 代码生成模式：
/// - `Arena`  ：编译期静态内存 Arena 规划，零 `Tensor` / 零 `BufferPool` / 推理热路径 0 堆分配（推荐默认）
/// - `Tensor` ：基于 `@ops` 的通用张量模式（支持全部算子与多输入多输出图）
#derive(Debug, Eq)
pub(all) enum CodegenMode {
  Arena
  Tensor
}

///|
pub impl Show for CodegenMode with fn output(self, logger) {
  match self {
    Arena => logger.write_string("arena")
    Tensor => logger.write_string("tensor")
  }
}

///|
/// AOT 代码生成配置选项。
pub(all) struct CodegenConfig {
  /// 生成的运行时结构体名称（默认 `"CompiledModel"`）
  struct_name : String
  /// 是否启用 V128 SIMD 预打包权重与 SIMD 算子路径（默认 `true`）
  use_simd : Bool
  /// 编译前是否先执行 `@opt.optimize_graph` 图优化管线（默认 `true`）
  optimize : Bool
  /// 目标平台（默认 `TargetWasm`）
  target : CodegenTarget
  /// 代码生成模式（默认 `Arena`，遇到不支持算子时自动回退至 `Tensor`）
  mode : CodegenMode
  /// 是否将权重二进制块直接内嵌为 MoonBit `Bytes` 常量段（实现单文件 `.wasm` 零外部加载）
  embed_weights : Bool
  /// 编译期冻结的静态输入常量（例如 Silero VAD 的 `sr = 16000`），用于折叠 `Equal`/`If` 分支特化
  frozen_inputs : Map[String, @tensor.Tensor]
}

///|
/// 返回默认 AOT 代码生成配置（结构体名 `"CompiledModel"`、Arena 零分配模式、开启图优化与 V128 SIMD）。
pub fn CodegenConfig::default() -> CodegenConfig {
  {
    struct_name: "CompiledModel",
    use_simd: true,
    optimize: true,
    target: TargetWasm,
    mode: Arena,
    embed_weights: false,
    frozen_inputs: Map([]),
  }
}

///|
/// 编译期静态权重的存储与加载形态。
#derive(Debug, Eq)
pub(all) enum WeightKind {
  /// 原始 Float32 张量（偏置、BN/LN 缩放、Depthwise 卷积核等）
  RawF32
  /// 预打包 2D 矩阵权重（`MatMul` / `Gemm` 右矩阵，支持预转置标志）
  PackedMatB(trans_b~ : Bool)
  /// 预打包 4D 卷积核（含普通卷积与零 `im2col` Depthwise 卷积）
  PackedConv(group~ : Int32)
  /// 原始 1 字节量化权重（`Int8` / `UInt8`，`trans_b=true` 时在加载期预转置为 `[N, K]`）
  RawBytes(trans_b~ : Bool)
}

///|
/// 二进制权重包 `weights.bin` 中单个张量的布局条目。
#derive(Debug, Eq)
pub(all) struct WeightEntry {
  name : String
  field_name : String
  kind : WeightKind
  shape : Array[Int32]
  byte_offset : Int32
  byte_len : Int32
}

///|
/// `onnx2mbt` AOT 静态编译产物：
/// - `mbt_source`：零 `Map`、零 Protobuf、零动态分派的纯 MoonBit 推理源码
/// - `pkg_config`：仅包含该模型实际依赖的最小 `moon.pkg`
/// - `weights_bin`：按小端序紧凑拼接的权重二进制块（Float32 为 4 字节，Int8/UInt8 为 1 字节）
pub struct AotArtifact {
  mbt_source : String
  pkg_config : String
  weights_bin : Bytes
  weights : Array[WeightEntry]
  input_shapes : Array[(String, Array[Int32])]
  output_shapes : Array[(String, Array[Int32])]
  node_count : Int32
  arena_f32_elems : Int32
  arena_f32_naive_elems : Int32
  arena_byte_elems : Int32
  scratch_f32_elems : Int32
  mode_used : CodegenMode
}

///|
/// 将整数数组格式化为 MoonBit 字面量（如 `[1, 1, 28, 28]`）。
fn fmt_int_array(arr : Array[Int32]) -> String {
  let sb = StringBuilder()
  sb.write_char('[')
  for i = 0; i < arr.length(); i = i + 1 {
    if i > 0 {
      sb.write_string(", ")
    }
    sb.write_string("\{arr[i]}")
  }
  sb.write_char(']')
  sb.to_string()
}

///|
/// 将 `Int64` 数组格式化为 MoonBit 字面量（带显式类型标注以便空/非空均无歧义）。
fn fmt_i64_array(arr : Array[Int64]) -> String {
  let sb = StringBuilder()
  sb.write_char('[')
  for i = 0; i < arr.length(); i = i + 1 {
    if i > 0 {
      sb.write_string(", ")
    }
    sb.write_string("(\{arr[i]} : Int64)")
  }
  sb.write_char(']')
  sb.to_string()
}

///|
/// 将浮点常量格式化为合法 MoonBit `Float` 表达式（通过 IEEE754 位模式确保零精度损失）。
fn fmt_float_literal(v : Float) -> String {
  let bits = v.reinterpret_as_int()
  "Float::reinterpret_from_int(\{bits})"
}

///|
/// 将 `@nn_ir.AutoPad` 格式化为源代码表达式。
fn fmt_auto_pad(ap : @nn_ir.AutoPad) -> String {
  match ap {
    @nn_ir.NotSet => "@nn_ir.NotSet"
    @nn_ir.SameUpper => "@nn_ir.SameUpper"
    @nn_ir.SameLower => "@nn_ir.SameLower"
    @nn_ir.Valid => "@nn_ir.Valid"
  }
}

///|
/// 将 `@nn_ir.FusedActivation` 格式化为源代码表达式。
fn fmt_activation(act : @nn_ir.FusedActivation) -> String {
  match act {
    @nn_ir.NoAct => "@nn_ir.NoAct"
    @nn_ir.ReluAct => "@nn_ir.ReluAct"
    @nn_ir.ClipAct(min~, max~) =>
      "@nn_ir.ClipAct(min=\{fmt_float_literal(min)}, max=\{fmt_float_literal(max)})"
  }
}

///|
/// 将 `@tensor.DType` 格式化为源代码表达式。
fn fmt_dtype(dt : @tensor.DType) -> String {
  match dt {
    @tensor.Float32 => "@tensor.Float32"
    @tensor.Float64 => "@tensor.Float64"
    @tensor.Int64 => "@tensor.Int64"
    @tensor.Int32 => "@tensor.Int32"
    @tensor.Bool => "@tensor.Bool"
    @tensor.UInt8 => "@tensor.UInt8"
    @tensor.Int8 => "@tensor.Int8"
  }
}

///|
/// 计算 2D 卷积/池化的静态输出空间维度（与 `ops/nn.mbt` 保持严格一致）。
fn infer_spatial_dim(
  in_dim : Int32,
  k : Int32,
  stride : Int32,
  dilation : Int32,
  pad_begin : Int32,
  pad_end : Int32,
  auto_pad : @nn_ir.AutoPad,
  ceil_mode : Bool,
) -> Int32 {
  let eff_k = (k - 1) * dilation + 1
  match auto_pad {
    @nn_ir.Valid => (in_dim - eff_k) / stride + 1
    @nn_ir.SameUpper | @nn_ir.SameLower => (in_dim + stride - 1) / stride
    @nn_ir.NotSet => {
      let num = in_dim + pad_begin + pad_end - eff_k
      if ceil_mode {
        (num + stride - 1) / stride + 1
      } else {
        num / stride + 1
      }
    }
  }
}

///|
/// 对已拓扑排序的 `IrGraph` 执行编译期静态形状推导，返回所有张量名称对应的确定形状表（支持多输出节点如 `Split`）。
pub fn infer_static_shapes(
  graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
) -> Map[String, Array[Int32]] raise CodegenError {
  let shapes : Map[String, Array[Int32]] = Map([])
  graph.initializers.each(fn(k, t) { shapes[k] = t.shape() })
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    match input_shapes.get(name) {
      Some(sh) => shapes[name] = sh.copy()
      None =>
        raise UnsupportedGraph(
          "missing static input shape for graph input '\{name}'",
        )
    }
  }

  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    if node.outputs.length() == 0 {
      continue
    }
    let out_shapes = infer_node_output_shapes(node, shapes, graph.initializers)
    for k = 0; k < out_shapes.length() && k < node.outputs.length(); k = k + 1 {
      if node.outputs[k] != "" {
        shapes[node.outputs[k]] = out_shapes[k]
      }
    }
  }
  shapes
}

///|
/// 判断算子是否为零拷贝视图算子（供 AOT 静态生命周期回收分析复用，防止回收别名底层缓冲）。
fn is_aot_view_op(op : @nn_ir.Op) -> Bool {
  match op {
    @nn_ir.Identity
    | @nn_ir.Flatten(..)
    | @nn_ir.Reshape(..)
    | @nn_ir.Transpose(..)
    | @nn_ir.Squeeze(..)
    | @nn_ir.Unsqueeze(..) => true
    _ => false
  }
}

///|
/// 将 `IrGraph` 编译为独立、零 `Map`、零 Protobuf 的纯 MoonBit 静态推理代码与紧凑权重二进制包。
/// 默认启用 `Arena` 模式（编译期静态内存区间复用 + 零堆分配 + 零 `Tensor` 依赖），遇到特殊算子时自动回退至 `Tensor` 模式。
pub fn compile_graph_to_mbt(
  raw_graph : @nn_ir.IrGraph,
  input_shapes : Map[String, Array[Int32]],
  config~ : CodegenConfig = CodegenConfig::default(),
) -> AotArtifact raise CodegenError {
  let base_graph = if config.optimize {
    let (opt_g, _) = @opt.optimize_graph(raw_graph)
    opt_g
  } else {
    raw_graph
  }
  // 执行编译期特化（冻结输入、`If` 分支内联、静态 `Shape`/元数据常量折叠、冗余 `Cast` 消除）
  let graph = specialize_graph(base_graph, input_shapes, config.frozen_inputs)
  if graph.inputs.length() == 0 || graph.outputs.length() == 0 {
    raise UnsupportedGraph("graph must have at least 1 input and 1 output")
  }

  let shapes = infer_static_shapes(graph, input_shapes)
  let dtypes = infer_static_dtypes(graph, Map([]))

  if config.mode == Arena && can_emit_arena(graph, shapes, dtypes) {
    let (f32_plan, byte_plan) = plan_static_arenas(graph, shapes, dtypes)
    return emit_arena_model(graph, shapes, dtypes, f32_plan, byte_plan, config)
  }

  // 1. 扫描所有节点，收集真正被消费的静态权重及其打包方式，同时构建紧凑 weights.bin
  let weight_entries : Array[WeightEntry] = []
  let weight_key_to_field : Map[String, String] = Map([])
  let raw_byte_chunks : Array[FixedArray[Byte]] = []
  let mut cur_byte_offset = 0

  fn register_f32_weight(
    name : String,
    kind : WeightKind,
  ) -> String raise CodegenError {
    let key = match kind {
      RawF32 => "\{name}#raw"
      PackedMatB(trans_b~) =>
        if trans_b {
          "\{name}#matT"
        } else {
          "\{name}#mat"
        }
      PackedConv(group~) => "\{name}#conv_g\{group}"
      RawBytes(..) => "\{name}#bytes"
    }
    match weight_key_to_field.get(key) {
      Some(f) => return f
      None => ()
    }
    let t = match graph.initializers.get(name) {
      Some(tensor) => tensor
      None =>
        raise UnsupportedGraph(
          "expected static initializer '\{name}' not found",
        )
    }
    if t.dtype() != @tensor.Float32 {
      raise TensorErr(
        "weight '\{name}' expected Float32 dtype, got \{t.dtype()}",
      )
    }
    let bytes_le = t.to_f32_bytes_le() catch { e => raise TensorErr("\{e}") }
    let field_name = "w_\{weight_entries.length()}"
    let byte_len = bytes_le.length()
    weight_entries.push({
      name,
      field_name,
      kind,
      shape: t.shape(),
      byte_offset: cur_byte_offset,
      byte_len,
    })
    weight_key_to_field[key] = field_name
    raw_byte_chunks.push(bytes_le)
    cur_byte_offset = cur_byte_offset + byte_len
    field_name
  }

  let mut needs_nn_ir = false
  let mut needs_gemm = false

  // 2. 为每个节点分配 SSA 变量名与静态生命周期回收计划
  let var_of : Map[String, String] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    var_of[graph.inputs[i]] = "in_\{i}"
  }

  let pinned : Map[String, Bool] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    pinned[graph.inputs[i]] = true
  }
  graph.initializers.each(fn(k, _) { pinned[k] = true })
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    pinned[graph.outputs[i]] = true
  }

  let root_of : Map[String, String] = Map([])
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let out_name = if node.outputs.length() > 0 { node.outputs[0] } else { "" }
    if out_name == "" {
      continue
    }
    var_of[out_name] = "v_\{i}"
    if is_aot_view_op(node.op) {
      let in_name = if node.inputs.length() > 0 { node.inputs[0] } else { "" }
      let root = match root_of.get(in_name) {
        Some(r) => r
        None => in_name
      }
      root_of[out_name] = root
      if pinned.contains(out_name) && root != "" {
        pinned[root] = true
      }
    } else {
      root_of[out_name] = out_name
    }
  }

  let last_use : Map[String, Int32] = Map([])
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let out_name = if node.outputs.length() > 0 { node.outputs[0] } else { "" }
    if out_name != "" && !is_aot_view_op(node.op) && !pinned.contains(out_name) {
      last_use[out_name] = i
    }
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    for j = 0; j < node.inputs.length(); j = j + 1 {
      let in_name = node.inputs[j]
      if in_name == "" {
        continue
      }
      let root = match root_of.get(in_name) {
        Some(r) => r
        None => in_name
      }
      if last_use.contains(root) {
        last_use[root] = i
      }
    }
  }
  let release_at : Array[Array[String]] = Array::makei(graph.nodes.length(), fn(
    _,
  ) {
    []
  })
  last_use.each(fn(root_name, step) {
    if !pinned.contains(root_name) {
      match var_of.get(root_name) {
        Some(v) => release_at[step].push(v)
        None => ()
      }
    }
  })

  // 辅助：解析节点输入的 MoonBit 表达式（支持图输入、上游 SSA 变量、静态 Float32 权重或内联 Int64 常量）
  fn resolve_expr(name : String) -> String raise CodegenError {
    match var_of.get(name) {
      Some(v) => v
      None =>
        match graph.initializers.get(name) {
          Some(t) =>
            match t.dtype() {
              @tensor.Float32 => {
                let f = register_f32_weight(name, RawF32)
                "self.\{f}"
              }
              @tensor.Int64 => {
                let b = t.to_i64_buffer() catch { e => raise TensorErr("\{e}") }
                let arr = Array::makei(b.length(), fn(k) { b[k] })
                "@tensor.Tensor::from_i64(\{fmt_int_array(t.shape())}, \{fmt_i64_array(arr)})"
              }
              other =>
                raise TensorErr(
                  "unsupported static initializer dtype \{other} for '\{name}'",
                )
            }
          None =>
            raise UnsupportedGraph("unresolved tensor reference '\{name}'")
        }
    }
  }

  // 3. 逐节点生成直线型 SSA 推理语句体
  let body = StringBuilder()
  let simd_str = if config.use_simd { "true" } else { "false" }

  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    let ins = node.inputs
    let out_name = node.outputs[0]
    let out_var = var_of.get(out_name).unwrap()
    let out_root = match root_of.get(out_name) {
      Some(r) => r
      None => out_name
    }
    let is_pinned = pinned.contains(out_name) || pinned.contains(out_root)
    let pool_expr = if is_pinned { "None" } else { "Some(self.pool)" }
    let sh_comment = match shapes.get(out_name) {
      Some(sh) => @tensor.format_shape(sh)
      None => "?"
    }

    body.write_string("  // [\{i}] \{node.op.name()} -> \{sh_comment}\n")
    match node.op {
      @nn_ir.Add => {
        let a = resolve_expr(ins[0])
        let b = resolve_expr(ins[1])
        body.write_string(
          "  let \{out_var} = @ops.add_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Sub => {
        let a = resolve_expr(ins[0])
        let b = resolve_expr(ins[1])
        body.write_string(
          "  let \{out_var} = @ops.sub_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Mul => {
        let a = resolve_expr(ins[0])
        let b = resolve_expr(ins[1])
        body.write_string(
          "  let \{out_var} = @ops.mul_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Div => {
        let a = resolve_expr(ins[0])
        let b = resolve_expr(ins[1])
        body.write_string(
          "  let \{out_var} = @ops.div_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Pow => {
        let a = resolve_expr(ins[0])
        let b = resolve_expr(ins[1])
        body.write_string(
          "  let \{out_var} = @ops.pow_f32(\{a}, \{b}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Sqrt => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.sqrt_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Exp => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.exp_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Neg => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.neg_f32(\{x}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Relu => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.relu_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Clip(min~, max~) => {
        let x = resolve_expr(ins[0])
        let lo_val : Float? = if ins.length() >= 2 && ins[1] != "" {
          match graph.initializers.get(ins[1]) {
            Some(t) =>
              Some(t.get_f32_flat(0) catch { e => raise TensorErr("\{e}") })
            None =>
              raise UnsupportedGraph("Clip '\{node.name}' requires static min")
          }
        } else {
          min
        }
        let hi_val : Float? = if ins.length() >= 3 && ins[2] != "" {
          match graph.initializers.get(ins[2]) {
            Some(t) =>
              Some(t.get_f32_flat(0) catch { e => raise TensorErr("\{e}") })
            None =>
              raise UnsupportedGraph("Clip '\{node.name}' requires static max")
          }
        } else {
          max
        }
        let lo_expr = match lo_val {
          Some(v) => "Some(\{fmt_float_literal(v)})"
          None => "None"
        }
        let hi_expr = match hi_val {
          Some(v) => "Some(\{fmt_float_literal(v)})"
          None => "None"
        }
        body.write_string(
          "  let \{out_var} = @ops.clip_f32(\{x}, \{lo_expr}, \{hi_expr}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Sigmoid => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.sigmoid_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Tanh => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.tanh_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Gelu => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.gelu_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.HardSwish => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.hard_swish_f32(\{x}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.LeakyRelu(alpha~) => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.leaky_relu_f32(\{x}, alpha=\{fmt_float_literal(alpha)}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Identity => {
        let x = resolve_expr(ins[0])
        body.write_string("  let \{out_var} = \{x}\n")
      }
      @nn_ir.Softmax(axis~) => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.softmax_f32(\{x}, \{axis}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Flatten(axis~) => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.flatten_op(\{x}, axis=\{axis})\n",
        )
      }
      @nn_ir.Reshape(allowzero~) => {
        let x = resolve_expr(ins[0])
        let sh = shapes.get(out_name).unwrap()
        body.write_string(
          "  let \{out_var} = \{x}.reshape(\{fmt_int_array(sh)}, allowzero=\{allowzero})\n",
        )
      }
      @nn_ir.Transpose(perm~) => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.transpose_op(\{x}, \{fmt_int_array(perm)})\n",
        )
      }
      @nn_ir.Squeeze(axes~) => {
        let x = resolve_expr(ins[0])
        let sh = shapes.get(out_name).unwrap()
        let _ = axes
        body.write_string(
          "  let \{out_var} = \{x}.reshape(\{fmt_int_array(sh)})\n",
        )
      }
      @nn_ir.Unsqueeze(axes~) => {
        let x = resolve_expr(ins[0])
        let sh = shapes.get(out_name).unwrap()
        let _ = axes
        body.write_string(
          "  let \{out_var} = \{x}.reshape(\{fmt_int_array(sh)})\n",
        )
      }
      @nn_ir.MatMul => {
        let a = resolve_expr(ins[0])
        let a_sh = shapes.get(ins[0]).unwrap()
        if config.use_simd &&
          a_sh.length() == 2 &&
          graph.initializers.contains(ins[1]) {
          needs_gemm = true
          let wf = register_f32_weight(ins[1], PackedMatB(trans_b=false))
          body.write_string(
            "  let \{out_var} = @ops.matmul_prepacked_f32(\{a}, self.\{wf}, pool=\{pool_expr})\n",
          )
        } else {
          let b = resolve_expr(ins[1])
          body.write_string(
            "  let \{out_var} = @ops.matmul_f32(\{a}, \{b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
          )
        }
      }
      @nn_ir.Gemm(alpha~, beta~, trans_a~, trans_b~) => {
        let a = resolve_expr(ins[0])
        let c_expr = if ins.length() >= 3 && ins[2] != "" {
          "Some(\{resolve_expr(ins[2])})"
        } else {
          "None"
        }
        if config.use_simd && graph.initializers.contains(ins[1]) {
          needs_gemm = true
          let wf = register_f32_weight(ins[1], PackedMatB(trans_b~))
          body.write_string(
            "  let \{out_var} = @ops.gemm_prepacked_f32(\{a}, self.\{wf}, \{c_expr}, alpha=\{fmt_float_literal(alpha)}, beta=\{fmt_float_literal(beta)}, trans_a=\{trans_a}, pool=\{pool_expr})\n",
          )
        } else {
          let b = resolve_expr(ins[1])
          body.write_string(
            "  let \{out_var} = @ops.gemm_f32(\{a}, \{b}, \{c_expr}, alpha=\{fmt_float_literal(alpha)}, beta=\{fmt_float_literal(beta)}, trans_a=\{trans_a}, trans_b=\{trans_b}, use_simd=\{simd_str}, pool=\{pool_expr})\n",
          )
        }
      }
      @nn_ir.Conv(
        strides~,
        pads~,
        dilations~,
        group~,
        auto_pad~,
        activation~,
        ..
      ) => {
        needs_nn_ir = true
        let x = resolve_expr(ins[0])
        let b_expr = if ins.length() >= 3 && ins[2] != "" {
          "Some(\{resolve_expr(ins[2])})"
        } else {
          "None"
        }
        if config.use_simd && graph.initializers.contains(ins[1]) {
          let wf = register_f32_weight(ins[1], PackedConv(group~))
          body.write_string(
            "  let \{out_var} = @ops.conv2d_prepacked_f32(\{x}, self.\{wf}, \{b_expr}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{fmt_int_array(dilations)}, \{fmt_auto_pad(auto_pad)}, activation=\{fmt_activation(activation)}, pool=\{pool_expr}, scratch_pool=Some(self.pool))\n",
          )
        } else {
          let w = resolve_expr(ins[1])
          body.write_string(
            "  let \{out_var} = @ops.conv2d_f32(\{x}, \{w}, \{b_expr}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{fmt_int_array(dilations)}, \{group}, \{fmt_auto_pad(auto_pad)}, use_simd=\{simd_str}, activation=\{fmt_activation(activation)}, pool=\{pool_expr}, scratch_pool=Some(self.pool))\n",
          )
        }
      }
      @nn_ir.MaxPool(
        kernel_shape~,
        strides~,
        pads~,
        dilations~,
        ceil_mode~,
        auto_pad~,
      ) => {
        needs_nn_ir = true
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.maxpool_2d_f32(\{x}, \{fmt_int_array(kernel_shape)}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{fmt_int_array(dilations)}, \{ceil_mode}, \{fmt_auto_pad(auto_pad)}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.AveragePool(
        kernel_shape~,
        strides~,
        pads~,
        ceil_mode~,
        count_include_pad~,
        auto_pad~,
      ) => {
        needs_nn_ir = true
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.avgpool_2d_f32(\{x}, \{fmt_int_array(kernel_shape)}, \{fmt_int_array(strides)}, \{fmt_int_array(pads)}, \{ceil_mode}, \{count_include_pad}, \{fmt_auto_pad(auto_pad)}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.GlobalAveragePool => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.global_avgpool_f32(\{x}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.BatchNormalization(epsilon~) => {
        let x = resolve_expr(ins[0])
        let s = resolve_expr(ins[1])
        let b = resolve_expr(ins[2])
        let m = resolve_expr(ins[3])
        let v = resolve_expr(ins[4])
        body.write_string(
          "  let \{out_var} = @ops.batch_norm_f32(\{x}, \{s}, \{b}, \{m}, \{v}, epsilon=\{fmt_float_literal(epsilon)}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.LayerNormalization(axis~, epsilon~) => {
        let x = resolve_expr(ins[0])
        let s = resolve_expr(ins[1])
        let b_expr = if ins.length() >= 3 && ins[2] != "" {
          "Some(\{resolve_expr(ins[2])})"
        } else {
          "None"
        }
        body.write_string(
          "  let \{out_var} = @ops.layer_norm_f32(\{x}, \{s}, \{b_expr}, axis=\{axis}, epsilon=\{fmt_float_literal(epsilon)}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Concat(axis~) => {
        let parts = StringBuilder()
        parts.write_char('[')
        for k = 0; k < ins.length(); k = k + 1 {
          if k > 0 {
            parts.write_string(", ")
          }
          parts.write_string(resolve_expr(ins[k]))
        }
        parts.write_char(']')
        body.write_string(
          "  let \{out_var} = @ops.concat_op(\{parts.to_string()}, \{axis}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.Cast(to~) => {
        let x = resolve_expr(ins[0])
        body.write_string(
          "  let \{out_var} = @ops.cast_op(\{x}, \{fmt_dtype(to)})\n",
        )
      }
      @nn_ir.ReduceMean(axes~, keepdims~, noop_with_empty_axes~) => {
        let x = resolve_expr(ins[0])
        let ax_expr = match axes {
          Some(a) => "Some(\{fmt_int_array(a)})"
          None => "None"
        }
        body.write_string(
          "  let \{out_var} = @ops.reduce_mean_f32(\{x}, \{ax_expr}, None, keepdims=\{keepdims}, noop_with_empty_axes=\{noop_with_empty_axes}, pool=\{pool_expr})\n",
        )
      }
      @nn_ir.ReduceSum(axes~, keepdims~, noop_with_empty_axes~) => {
        let x = resolve_expr(ins[0])
        let ax_expr = match axes {
          Some(a) => "Some(\{fmt_int_array(a)})"
          None => "None"
        }
        body.write_string(
          "  let \{out_var} = @ops.reduce_sum_f32(\{x}, \{ax_expr}, None, keepdims=\{keepdims}, noop_with_empty_axes=\{noop_with_empty_axes}, pool=\{pool_expr})\n",
        )
      }
      other =>
        raise UnsupportedOp(
          "AOT codegen does not yet support op '\{other.name()}'",
        )
    }

    // 在每个节点执行完毕后立即释放死亡中间张量缓冲回 BufferPool
    let dead = release_at[i]
    for d = 0; d < dead.length(); d = d + 1 {
      body.write_string("  release_buf(self.pool, \{dead[d]})\n")
    }
  }

  // 4. 拼接完整 MoonBit 源码文件
  let src = StringBuilder()
  let sname = config.struct_name
  src.write_string(
    "// Auto-generated by moon-onnx AOT compiler (`onnx2mbt`).\n",
  )
  src.write_string(
    "// Graph: \{graph.name} | Nodes: \{graph.nodes.length()} | Weights: \{weight_entries.length()} (\{cur_byte_offset} bytes)\n\n",
  )

  // Helper: slice Float32 tensor from weights.bin
  if weight_entries.length() > 0 {
    src.write_string("///|\n")
    src.write_string(
      "/// 从紧凑权重二进制块按静态偏移解码 Float32 张量。\n",
    )
    src.write_string(
      "fn load_weight_f32(\n  weights : Bytes,\n  offset : Int32,\n  shape : Array[Int32],\n) -> @tensor.Tensor raise @tensor.TensorError {\n",
    )
    src.write_string("  let n = @tensor.num_elements(shape)\n")
    src.write_string(
      "  let raw = Bytes::makei(n * 4, i => weights[offset + i])\n",
    )
    src.write_string("  @tensor.Tensor::from_f32_bytes_le(shape, raw)\n")
    src.write_string("}\n\n")
  }

  // Helper: release buffer back to pool
  let mut has_releases = false
  for i = 0; i < release_at.length(); i = i + 1 {
    if release_at[i].length() > 0 {
      has_releases = true
      break
    }
  }
  if has_releases {
    src.write_string("///|\n")
    src.write_string(
      "/// 将死亡中间张量的底层 Float32 缓冲区归还至复用池。\n",
    )
    src.write_string(
      "fn release_buf(pool : @tensor.BufferPool, t : @tensor.Tensor) -> Unit {\n",
    )
    src.write_string("  match t.raw_f32_buffer() {\n")
    src.write_string("    Some(buf) => pool.release_f32(buf)\n")
    src.write_string("    None => ()\n")
    src.write_string("  }\n")
    src.write_string("}\n\n")
  }

  // Struct definition
  src.write_string("///|\n")
  src.write_string(
    "/// 静态编译后的推理运行时实例（零 Map 查找、零 Protobuf 解析、预打包权重常驻）。\n",
  )
  src.write_string("pub struct \{sname} {\n")
  for i = 0; i < weight_entries.length(); i = i + 1 {
    let w = weight_entries[i]
    let ty = match w.kind {
      RawF32 | RawBytes(..) => "@tensor.Tensor"
      PackedMatB(..) => "@gemm.PackedMatrixB"
      PackedConv(..) => "@ops.PackedConvWeight"
    }
    src.write_string("  \{w.field_name} : \{ty}\n")
  }
  src.write_string("  pool : @tensor.BufferPool\n")
  src.write_string("}\n\n")

  // load(weights : Bytes) -> StructName
  src.write_string("///|\n")
  src.write_string(
    "/// 从紧凑 `weights.bin` 字节流一次性加载并预打包全部模型权重。\n",
  )
  src.write_string(
    "pub fn \{sname}::load(weights : Bytes) -> \{sname} raise @tensor.TensorError {\n",
  )
  if weight_entries.length() == 0 {
    src.write_string("  let _ = weights\n")
  }
  for i = 0; i < weight_entries.length(); i = i + 1 {
    let w = weight_entries[i]
    let sh_str = fmt_int_array(w.shape)
    match w.kind {
      RawF32 | RawBytes(..) =>
        src.write_string(
          "  let \{w.field_name} = load_weight_f32(weights, \{w.byte_offset}, \{sh_str})\n",
        )
      PackedMatB(trans_b~) => {
        src.write_string(
          "  let \{w.field_name}_raw = load_weight_f32(weights, \{w.byte_offset}, \{sh_str}).to_f32_buffer()\n",
        )
        if trans_b {
          src.write_string(
            "  let \{w.field_name} = @gemm.pack_matrix_b_transposed_f32(\{w.field_name}_raw, 0, \{w.shape[0]}, \{w.shape[1]})\n",
          )
        } else {
          src.write_string(
            "  let \{w.field_name} = @gemm.pack_matrix_b_f32(\{w.field_name}_raw, 0, \{w.shape[0]}, \{w.shape[1]})\n",
          )
        }
      }
      PackedConv(group~) => {
        src.write_string(
          "  let \{w.field_name}_t = load_weight_f32(weights, \{w.byte_offset}, \{sh_str})\n",
        )
        src.write_string(
          "  let \{w.field_name} = @ops.pack_conv2d_weight_f32(\{w.field_name}_t, \{group})\n",
        )
      }
    }
  }
  src.write_string("  {\n")
  for i = 0; i < weight_entries.length(); i = i + 1 {
    src.write_string("    \{weight_entries[i].field_name},\n")
  }
  src.write_string("    pool: @tensor.BufferPool::new(),\n")
  src.write_string("  }\n")
  src.write_string("}\n\n")

  // forward(...)
  src.write_string("///|\n")
  src.write_string("/// 执行静态编译后的直线型前向推理。\n")
  src.write_string("pub fn \{sname}::forward(\n  self : \{sname},\n")
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    src.write_string("  in_\{i} : @tensor.Tensor,\n")
  }
  if graph.outputs.length() == 1 {
    src.write_string(") -> @tensor.Tensor raise @tensor.TensorError {\n")
  } else {
    src.write_string(") -> Array[@tensor.Tensor] raise @tensor.TensorError {\n")
  }
  src.write_string(body.to_string())
  if graph.outputs.length() == 1 {
    let out_v = resolve_expr(graph.outputs[0])
    src.write_string("  \{out_v}\n")
  } else {
    src.write_string("  [")
    for i = 0; i < graph.outputs.length(); i = i + 1 {
      if i > 0 {
        src.write_string(", ")
      }
      src.write_string(resolve_expr(graph.outputs[i]))
    }
    src.write_string("]\n")
  }
  src.write_string("}\n")

  // 单输入单输出时额外生成面向 Wasm/FFI 的平坦 FixedArray[Float] 快捷入口
  if graph.inputs.length() == 1 && graph.outputs.length() == 1 {
    let in_sh = shapes.get(graph.inputs[0]).unwrap()
    src.write_string("\n///|\n")
    src.write_string(
      "/// 平坦 `FixedArray[Float]` 快捷前向入口（输入形状固定为 \{@tensor.format_shape(in_sh)}）。\n",
    )
    src.write_string(
      "pub fn \{sname}::forward_f32(\n  self : \{sname},\n  input : FixedArray[Float],\n) -> FixedArray[Float] raise @tensor.TensorError {\n",
    )
    src.write_string(
      "  let t_in = @tensor.Tensor::from_f32(\{fmt_int_array(in_sh)}, input)\n",
    )
    src.write_string("  self.forward(t_in).to_f32_buffer()\n")
    src.write_string("}\n")
  }

  // 5. 生成最小依赖 moon.pkg
  let pkg = StringBuilder()
  pkg.write_string("import {\n")
  pkg.write_string("  \"matrixlt/tensor\",\n")
  if needs_nn_ir {
    pkg.write_string("  \"matrixlt/nn_ir\",\n")
  }
  if needs_gemm {
    pkg.write_string("  \"matrixlt/ml_kernels/gemm\",\n")
  }
  pkg.write_string("  \"matrixlt/onnx/ops\",\n")
  pkg.write_string("}\n")

  // 6. 合并所有权重字节块为单个紧凑 Bytes
  let flat_bytes = FixedArray::make(cur_byte_offset, b'\x00')
  let mut pos = 0
  for c = 0; c < raw_byte_chunks.length(); c = c + 1 {
    let chunk = raw_byte_chunks[c]
    for b = 0; b < chunk.length(); b = b + 1 {
      flat_bytes[pos + b] = chunk[b]
    }
    pos = pos + chunk.length()
  }
  let weights_bin = Bytes::makei(cur_byte_offset, fn(i) { flat_bytes[i] })

  if config.embed_weights {
    src.write_string("\n")
    emit_embedded_weights_section(src, sname, weights_bin, true)
  }

  let in_sh_list = Array::makei(graph.inputs.length(), fn(i) {
    let name = graph.inputs[i]
    (name, shapes.get(name).unwrap())
  })
  let out_sh_list = Array::makei(graph.outputs.length(), fn(i) {
    let name = graph.outputs[i]
    (name, shapes.get(name).unwrap())
  })

  {
    mbt_source: src.to_string(),
    pkg_config: pkg.to_string(),
    weights_bin,
    weights: weight_entries,
    input_shapes: in_sh_list,
    output_shapes: out_sh_list,
    node_count: graph.nodes.length(),
    arena_f32_elems: 0,
    arena_f32_naive_elems: 0,
    arena_byte_elems: 0,
    scratch_f32_elems: 0,
    mode_used: Tensor,
  }
}
