///|
/// 单个中间张量在编译期静态内存 Arena 中的分配区间与生命周期。
#derive(Debug, Eq)
pub(all) struct TensorAlloc {
  name : String
  /// 在所属 Arena 缓冲区中的起始元素下标（Float32 按 4 元素对齐，Byte 按 16 字节对齐）
  offset : Int32
  /// 占用元素数量
  length : Int32
  /// 产生该张量的节点拓扑索引
  birth : Int32
  /// 最后一次消费该张量（含其全部零拷贝别名视图）的节点拓扑索引
  death : Int32
}

///|
/// 编译期静态内存规划结果（Unified Static Memory Planning / Arena Plan）。
/// 将所有中间激活张量映射到单一预分配线性缓冲区的固定偏移，彻底消除运行期堆分配与 `BufferPool` 哈希查找。
pub struct ArenaPlan {
  /// 每个中间张量名（含零拷贝视图别名）对应的静态偏移与长度
  allocs : Map[String, TensorAlloc]
  /// Arena 所需总元素数（峰值同时存活内存）
  peak_elems : Int32
  /// 不做内存复用时的朴素累加元素数（用于计算内存节省比例）
  naive_elems : Int32
  /// 所有卷积层共享的单个 `im2col` 临时工作区峰值元素数
  scratch_elems : Int32
}

///|
/// 返回静态 Arena 复用相较朴素独立分配节省的内存百分比（0.0 ~ 100.0）。
pub fn ArenaPlan::savings_pct(self : ArenaPlan) -> Double {
  if self.naive_elems <= 0 {
    return 0.0
  }
  let saved = self.naive_elems - self.peak_elems
  if saved <= 0 {
    0.0
  } else {
    saved.to_double() / self.naive_elems.to_double() * 100.0
  }
}

///|
/// 将非负长度向上对齐到 `align` 的整数倍（`align` 须为正整数）。
fn align_up(val : Int32, align : Int32) -> Int32 {
  if align <= 1 {
    val
  } else {
    (val + align - 1) / align * align
  }
}

///|
/// 判断算子在 AOT Arena 模式下是否为纯零拷贝逻辑视图（只需静态复用输入偏移，无需分配内存或生成运行期代码）。
pub fn is_arena_alias_op(op : @nn_ir.Op) -> Bool {
  match op {
    @nn_ir.Identity
    | @nn_ir.Flatten(..)
    | @nn_ir.Reshape(..)
    | @nn_ir.Squeeze(..)
    | @nn_ir.Unsqueeze(..) => true
    _ => false
  }
}

///|
/// 推导图中每个张量的静态数据类型（供双 Arena 分流：`Float32` vs `Int8`/`UInt8`）。
pub fn infer_static_dtypes(
  graph : @nn_ir.IrGraph,
  input_dtypes : Map[String, @tensor.DType],
) -> Map[String, @tensor.DType] {
  let dtypes : Map[String, @tensor.DType] = Map([])
  graph.initializers.each(fn(k, t) { dtypes[k] = t.dtype() })
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    let name = graph.inputs[i]
    dtypes[name] = match input_dtypes.get(name) {
      Some(dt) => dt
      None => @tensor.Float32
    }
  }
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    if node.outputs.length() == 0 || node.outputs[0] == "" {
      continue
    }
    let out_dt : @tensor.DType = match node.op {
      @nn_ir.Cast(to~) => to
      @nn_ir.Shape(..) => @tensor.Int64
      @nn_ir.Equal
      | @nn_ir.Less
      | @nn_ir.Greater
      | @nn_ir.LessOrEqual
      | @nn_ir.GreaterOrEqual
      | @nn_ir.Not
      | @nn_ir.And => @tensor.Bool
      @nn_ir.Where =>
        if node.inputs.length() >= 2 {
          match dtypes.get(node.inputs[1]) {
            Some(dt) => dt
            None => @tensor.Float32
          }
        } else {
          @tensor.Float32
        }
      @nn_ir.Constant(value~) => value.dtype()
      @nn_ir.ConstantOfShape(value~) => value.dtype()
      @nn_ir.DequantizeLinear(..) => @tensor.Float32
      @nn_ir.QuantizeLinear(..) =>
        if node.inputs.length() >= 3 && node.inputs[2] != "" {
          match dtypes.get(node.inputs[2]) {
            Some(dt) => dt
            None => @tensor.UInt8
          }
        } else {
          @tensor.UInt8
        }
      @nn_ir.QLinearMatMul =>
        if node.inputs.length() >= 8 && node.inputs[7] != "" {
          match dtypes.get(node.inputs[7]) {
            Some(dt) => dt
            None => @tensor.Int8
          }
        } else {
          @tensor.Int8
        }
      _ =>
        if node.inputs.length() > 0 {
          match dtypes.get(node.inputs[0]) {
            Some(dt) => dt
            None => @tensor.Float32
          }
        } else {
          @tensor.Float32
        }
    }
    for k = 0; k < node.outputs.length(); k = k + 1 {
      if node.outputs[k] != "" {
        dtypes[node.outputs[k]] = out_dt
      }
    }
  }
  dtypes
}

///|
/// 内部候选张量记录，用于贪心按大小优先的首次适配（Greedy-by-Size First-Fit）区间分配。
priv struct IntervalItem {
  name : String
  length : Int32
  birth : Int32
  death : Int32
}

///|
/// 对一组带生命周期区间的张量执行 Greedy-by-Size First-Fit 静态偏移分配（与 TFLite Micro / TVM USMP 一致）：
/// - 按张量体积从大到小排序优先放置，最大限度减少内存碎片；
/// - 两个生命周期不重叠的张量自动共享物理偏移；
/// - 起始偏移按 `align` 对齐以保证 V128 SIMD 友好访问。
fn pack_intervals_greedy(
  items : Array[IntervalItem],
  align : Int32,
) -> (Map[String, TensorAlloc], Int32, Int32) {
  let n = items.length()
  let mut naive = 0
  for i = 0; i < n; i = i + 1 {
    naive = naive + items[i].length
  }

  // 插入排序：按 length 降序，同长度按 birth 升序（确定性排序，结果完全可复现）
  let sorted = items.copy()
  for i = 1; i < n; i = i + 1 {
    let key = sorted[i]
    let mut j = i - 1
    while j >= 0 &&
          (
            sorted[j].length < key.length ||
            (sorted[j].length == key.length && sorted[j].birth > key.birth)
          ) {
      sorted[j + 1] = sorted[j]
      j = j - 1
    }
    sorted[j + 1] = key
  }

  let placed : Array[TensorAlloc] = []
  let allocs : Map[String, TensorAlloc] = Map([])
  let mut peak = 0

  for i = 0; i < n; i = i + 1 {
    let item = sorted[i]
    // 收集所有与当前张量生命周期重叠的已放置区间 [start, end)
    let busy_start : Array[Int32] = []
    let busy_end : Array[Int32] = []
    for p = 0; p < placed.length(); p = p + 1 {
      let other = placed[p]
      // 闭区间 [birth, death] 重叠条件：a.birth <= b.death && b.birth <= a.death
      if item.birth <= other.death && other.birth <= item.death {
        busy_start.push(other.offset)
        busy_end.push(other.offset + other.length)
      }
    }

    // 按起始偏移排序冲突区间
    let m = busy_start.length()
    for a = 1; a < m; a = a + 1 {
      let ks = busy_start[a]
      let ke = busy_end[a]
      let mut b = a - 1
      while b >= 0 && busy_start[b] > ks {
        busy_start[b + 1] = busy_start[b]
        busy_end[b + 1] = busy_end[b]
        b = b - 1
      }
      busy_start[b + 1] = ks
      busy_end[b + 1] = ke
    }

    // 扫描空闲缝隙，寻找最小的对齐可用偏移
    let mut cand = 0
    for b = 0; b < m; b = b + 1 {
      if cand + item.length <= busy_start[b] {
        break
      }
      if busy_end[b] > cand {
        cand = align_up(busy_end[b], align)
      }
    }

    let alloc : TensorAlloc = {
      name: item.name,
      offset: cand,
      length: item.length,
      birth: item.birth,
      death: item.death,
    }
    placed.push(alloc)
    allocs[item.name] = alloc
    let end_pos = cand + item.length
    if end_pos > peak {
      peak = end_pos
    }
  }

  (allocs, align_up(peak, align), naive)
}

///|
/// 为计算图分别规划 `Float32` 静态激活 Arena 与 `Byte`（`Int8`/`UInt8`）静态量化激活 Arena，
/// 并计算所有非 Depthwise 卷积层共享的 `im2col` 临时工作区峰值大小。
pub fn plan_static_arenas(
  graph : @nn_ir.IrGraph,
  shapes : Map[String, Array[Int32]],
  dtypes : Map[String, @tensor.DType],
) -> (ArenaPlan, ArenaPlan) raise CodegenError {
  let num_nodes = graph.nodes.length()
  let is_external : Map[String, Bool] = Map([])
  for i = 0; i < graph.inputs.length(); i = i + 1 {
    is_external[graph.inputs[i]] = true
  }
  graph.initializers.each(fn(k, _) { is_external[k] = true })

  let is_graph_out : Map[String, Bool] = Map([])
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    is_graph_out[graph.outputs[i]] = true
  }

  // 1. 建立零拷贝视图别名到物理根张量的映射，以及每个根张量的诞生/死亡步号
  let root_of : Map[String, String] = Map([])
  let birth_of : Map[String, Int32] = Map([])
  let death_of : Map[String, Int32] = Map([])
  let mut max_im2col = 0

  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    // 估算非 Depthwise 2D 卷积所需的 im2col 临时工作区峰值
    match node.op {
      @nn_ir.Conv(group~, ..) =>
        if node.inputs.length() >= 2 && node.outputs.length() >= 1 {
          match
            (
              shapes.get(node.inputs[0]),
              shapes.get(node.inputs[1]),
              shapes.get(node.outputs[0]),
            ) {
            (Some(sx), Some(sw), Some(sy)) if (
                sx.length() == 4 && sw.length() == 4 && sy.length() == 4
              ) ||
              (sx.length() == 3 && sw.length() == 3 && sy.length() == 3) => {
              let c_in = sx[1]
              let c_out = sw[0]
              let is_dw = group > 1 && group == c_in && group == c_out
              if !is_dw {
                let k_dim = if sw.length() == 4 {
                  sw[1] * sw[2] * sw[3]
                } else {
                  sw[1] * sw[2]
                }
                let spatial = if sy.length() == 4 {
                  sy[2] * sy[3]
                } else {
                  sy[2]
                }
                let need = spatial * k_dim
                if need > max_im2col {
                  max_im2col = need
                }
              }
            }
            _ => ()
          }
        }
      _ => ()
    }

    for k = 0; k < node.outputs.length(); k = k + 1 {
      let out_name = node.outputs[k]
      if out_name == "" {
        continue
      }
      if is_arena_alias_op(node.op) && k == 0 && node.inputs.length() > 0 {
        let in_name = node.inputs[0]
        let root = match root_of.get(in_name) {
          Some(r) => r
          None => in_name
        }
        root_of[out_name] = root
      } else {
        root_of[out_name] = out_name
        birth_of[out_name] = i
        death_of[out_name] = i
      }
    }
  }

  // 2. 扫描所有消费点与图输出，延长对应根张量的死亡步号
  for i = 0; i < num_nodes; i = i + 1 {
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
      if death_of.contains(root) {
        death_of[root] = i
      }
    }
  }
  for i = 0; i < graph.outputs.length(); i = i + 1 {
    let out_name = graph.outputs[i]
    let root = match root_of.get(out_name) {
      Some(r) => r
      None => out_name
    }
    if death_of.contains(root) {
      // 图输出在整个前向计算结束前均保持存活，禁止被后续节点覆盖
      death_of[root] = num_nodes
    }
  }

  // 3. 拆分 Float32 与 Byte（Int8/UInt8）中间张量，分别执行区间着色分配
  let f32_items : Array[IntervalItem] = []
  let byte_items : Array[IntervalItem] = []

  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    for k = 0; k < node.outputs.length(); k = k + 1 {
      let out_name = node.outputs[k]
      if out_name == "" ||
        (is_arena_alias_op(node.op) && k == 0) ||
        is_external.contains(out_name) {
        continue
      }
      let sh = match shapes.get(out_name) {
        Some(s) => s
        None => raise ShapeError("planner: missing shape for '\{out_name}'")
      }
      let numel = @tensor.num_elements(sh) catch {
        e => raise ShapeError("planner: \{e}")
      }
      let b = birth_of.get(out_name).unwrap()
      let d = death_of.get(out_name).unwrap()
      let dt = match dtypes.get(out_name) {
        Some(t) => t
        None => @tensor.Float32
      }
      match dt {
        @tensor.Float32 =>
          f32_items.push({ name: out_name, length: numel, birth: b, death: d, })
        @tensor.Int8 | @tensor.UInt8 =>
          byte_items.push({ name: out_name, length: numel, birth: b, death: d, })
        _ => ()
      }
    }
  }

  let (f32_allocs, f32_peak, f32_naive) = pack_intervals_greedy(f32_items, 4)
  let (byte_allocs, byte_peak, byte_naive) = pack_intervals_greedy(
    byte_items, 16,
  )

  // 4. 将零拷贝别名视图映射到其物理根张量的同一偏移，便于代码生成器统一查表
  for i = 0; i < num_nodes; i = i + 1 {
    let node = graph.nodes[i]
    if node.outputs.length() == 0 || node.outputs[0] == "" {
      continue
    }
    let out_name = node.outputs[0]
    if is_arena_alias_op(node.op) {
      match root_of.get(out_name) {
        Some(root) => {
          match f32_allocs.get(root) {
            Some(ra) => f32_allocs[out_name] = ra
            None => ()
          }
          match byte_allocs.get(root) {
            Some(ra) => byte_allocs[out_name] = ra
            None => ()
          }
        }
        None => ()
      }
    }
  }
  let _ = is_graph_out

  let f32_plan : ArenaPlan = {
    allocs: f32_allocs,
    peak_elems: f32_peak,
    naive_elems: f32_naive,
    scratch_elems: align_up(max_im2col, 4),
  }
  let byte_plan : ArenaPlan = {
    allocs: byte_allocs,
    peak_elems: byte_peak,
    naive_elems: byte_naive,
    scratch_elems: 0,
  }
  (f32_plan, byte_plan)
}
