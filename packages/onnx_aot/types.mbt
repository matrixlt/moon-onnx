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
  /// 可选输入张量数据类型声明（缺失项默认 `Float32`；NLP 模型如 `input_ids` / `attention_mask` 为 `Int64`）
  input_dtypes : Map[String, @tensor.DType]
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
    input_dtypes: Map([]),
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
  /// 预打包 2D `Int8` 权重（`MatMulInteger` 离线预转置 `[N, K]` + 预计算列和 `col_sums[N]`）
  PackedInt8MatB
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
    @nn_ir.SiluAct => "@nn_ir.SiluAct"
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
