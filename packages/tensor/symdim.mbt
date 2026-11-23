///|
/// 符号维度规格（参考 Tract `TDim` 与 ONNX `TensorShapeProto.Dimension`）。
/// 用于在图编译与形状推断阶段同时表达静态维度、命名动态维度（如 `batch` / `seq_len`）与未知维度。
#derive(Debug, Eq)
pub(all) enum SymDim {
  Static(Int32)
  Sym(String)
  Unknown
}

///|
/// 将单个符号维度格式化为人类可读字符串。
pub fn SymDim::to_label(self : SymDim) -> String {
  match self {
    Static(n) => n.to_string()
    Sym(name) => name
    Unknown => "?"
  }
}

///|
/// 使用运行期符号绑定表（如 `{"batch": 4}`）尝试将符号维度解析为具体整数。
pub fn SymDim::resolve(self : SymDim, bindings : Map[String, Int32]) -> Int32? {
  match self {
    Static(n) => Some(n)
    Sym(name) => bindings.get(name)
    Unknown => None
  }
}

///|
/// 符号形状包装类型，支持静态化解析、格式化与静态形状判定。
#derive(Debug, Eq)
pub struct SymShape {
  dims : Array[SymDim]
}

///|
/// 从静态整数形状数组构造 SymShape。
pub fn SymShape::from_static(shape : Array[Int32]) -> SymShape {
  { dims: Array::makei(shape.length(), fn(i) { Static(shape[i]) }), }
}

///|
/// 从符号维度数组构造 SymShape。
pub fn SymShape::from_dims(dims : Array[SymDim]) -> SymShape {
  { dims: dims.copy(), }
}

///|
/// 返回形状阶数（Rank）。
pub fn SymShape::rank(self : SymShape) -> Int32 {
  self.dims.length()
}

///|
/// 判断该形状是否全部为已知静态维度。
pub fn SymShape::is_fully_static(self : SymShape) -> Bool {
  for i = 0; i < self.dims.length(); i = i + 1 {
    match self.dims[i] {
      Static(_) => ()
      _ => return false
    }
  }
  true
}

///|
/// 根据符号绑定表尝试将全部维度解析为静态 `Array[Int]`；若有任一维度无法解析则报错。
pub fn SymShape::resolve(
  self : SymShape,
  bindings : Map[String, Int32],
) -> Array[Int32] raise TensorError {
  let out = Array::make(self.dims.length(), 0)
  for i = 0; i < self.dims.length(); i = i + 1 {
    match self.dims[i].resolve(bindings) {
      Some(v) => {
        if v < 0 {
          raise InvalidShape("resolved negative dim \{v} at axis \{i}")
        }
        out[i] = v
      }
      None =>
        raise InvalidShape(
          "cannot resolve symbolic dimension '\{self.dims[i].to_label()}' at axis \{i}",
        )
    }
  }
  out
}

///|
/// 将 SymShape 格式化为 `[batch, 1, 28, 28]` 形式字符串。
pub fn SymShape::format(self : SymShape) -> String {
  let buf = StringBuilder()
  buf.write_char('[')
  for i = 0; i < self.dims.length(); i = i + 1 {
    if i > 0 {
      buf.write_string(", ")
    }
    buf.write_string(self.dims[i].to_label())
  }
  buf.write_char(']')
  buf.to_string()
}
