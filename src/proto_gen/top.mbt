///|
#derive(Eq, Debug)
pub(all) enum Version {
  XSTART_VERSION
  IR_VERSION_2017_10_10
  IR_VERSION_2017_10_30
  IR_VERSION_2017_11_3
  IR_VERSION_2019_1_22
  IR_VERSION_2019_3_18
  IR_VERSION_2019_9_19
  IR_VERSION_2020_5_8
  IR_VERSION_2021_7_30
  IR_VERSION_2023_5_5
  IR_VERSION_2024_3_25
  IR_VERSION_2025_05_12
  IR_VERSION_2025_08_26
  IR_VERSION_2025_11_06
  IR_VERSION
}

///|
pub fn Version::to_enum(self : Version) -> @lib.Enum {
  match self {
    Version::XSTART_VERSION => 0
    Version::IR_VERSION_2017_10_10 => 1
    Version::IR_VERSION_2017_10_30 => 2
    Version::IR_VERSION_2017_11_3 => 3
    Version::IR_VERSION_2019_1_22 => 4
    Version::IR_VERSION_2019_3_18 => 5
    Version::IR_VERSION_2019_9_19 => 6
    Version::IR_VERSION_2020_5_8 => 7
    Version::IR_VERSION_2021_7_30 => 8
    Version::IR_VERSION_2023_5_5 => 9
    Version::IR_VERSION_2024_3_25 => 10
    Version::IR_VERSION_2025_05_12 => 11
    Version::IR_VERSION_2025_08_26 => 12
    Version::IR_VERSION_2025_11_06 => 13
    Version::IR_VERSION => 14
  }
}

///|
pub fn Version::from_enum(i : @lib.Enum) -> Version {
  match i.0 {
    0 => Version::XSTART_VERSION
    1 => Version::IR_VERSION_2017_10_10
    2 => Version::IR_VERSION_2017_10_30
    3 => Version::IR_VERSION_2017_11_3
    4 => Version::IR_VERSION_2019_1_22
    5 => Version::IR_VERSION_2019_3_18
    6 => Version::IR_VERSION_2019_9_19
    7 => Version::IR_VERSION_2020_5_8
    8 => Version::IR_VERSION_2021_7_30
    9 => Version::IR_VERSION_2023_5_5
    10 => Version::IR_VERSION_2024_3_25
    11 => Version::IR_VERSION_2025_05_12
    12 => Version::IR_VERSION_2025_08_26
    13 => Version::IR_VERSION_2025_11_06
    14 => Version::IR_VERSION
    _ => Default::default()
  }
}

///|
pub impl Default for Version with fn default() -> Version {
  Version::XSTART_VERSION
}

///|
pub impl @lib.Sized for Version with fn size_of(self : Version) {
  @lib.Sized::size_of(self.to_enum())
}

///|
#deprecated("Use `==` instead")
pub extend Version with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend Version with Eq::{not_equal}

///|
pub extend Version with Debug::{to_repr}

///|
pub extend Version with Default::{default}

///|
pub extend Version with @lib.Sized::{size_of}

///|
#derive(Eq, Debug)
pub(all) enum OperatorStatus {
  EXPERIMENTAL
  STABLE
}

///|
pub fn OperatorStatus::to_enum(self : OperatorStatus) -> @lib.Enum {
  match self {
    OperatorStatus::EXPERIMENTAL => 0
    OperatorStatus::STABLE => 1
  }
}

///|
pub fn OperatorStatus::from_enum(i : @lib.Enum) -> OperatorStatus {
  match i.0 {
    0 => OperatorStatus::EXPERIMENTAL
    1 => OperatorStatus::STABLE
    _ => Default::default()
  }
}

///|
pub impl Default for OperatorStatus with fn default() -> OperatorStatus {
  OperatorStatus::EXPERIMENTAL
}

///|
pub impl @lib.Sized for OperatorStatus with fn size_of(self : OperatorStatus) {
  @lib.Sized::size_of(self.to_enum())
}

///|
#deprecated("Use `==` instead")
pub extend OperatorStatus with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend OperatorStatus with Eq::{not_equal}

///|
pub extend OperatorStatus with Debug::{to_repr}

///|
pub extend OperatorStatus with Default::{default}

///|
pub extend OperatorStatus with @lib.Sized::{size_of}

///|
#derive(Eq, Debug)
pub(all) enum AttributeProto_AttributeType {
  UNDEFINED
  FLOAT
  INT
  STRING
  TENSOR
  GRAPH
  SPARSE_TENSOR
  TYPE_PROTO
  FLOATS
  INTS
  STRINGS
  TENSORS
  GRAPHS
  SPARSE_TENSORS
  TYPE_PROTOS
}

///|
pub fn AttributeProto_AttributeType::to_enum(
  self : AttributeProto_AttributeType,
) -> @lib.Enum {
  match self {
    AttributeProto_AttributeType::UNDEFINED => 0
    AttributeProto_AttributeType::FLOAT => 1
    AttributeProto_AttributeType::INT => 2
    AttributeProto_AttributeType::STRING => 3
    AttributeProto_AttributeType::TENSOR => 4
    AttributeProto_AttributeType::GRAPH => 5
    AttributeProto_AttributeType::SPARSE_TENSOR => 11
    AttributeProto_AttributeType::TYPE_PROTO => 13
    AttributeProto_AttributeType::FLOATS => 6
    AttributeProto_AttributeType::INTS => 7
    AttributeProto_AttributeType::STRINGS => 8
    AttributeProto_AttributeType::TENSORS => 9
    AttributeProto_AttributeType::GRAPHS => 10
    AttributeProto_AttributeType::SPARSE_TENSORS => 12
    AttributeProto_AttributeType::TYPE_PROTOS => 14
  }
}

///|
pub fn AttributeProto_AttributeType::from_enum(
  i : @lib.Enum,
) -> AttributeProto_AttributeType {
  match i.0 {
    0 => AttributeProto_AttributeType::UNDEFINED
    1 => AttributeProto_AttributeType::FLOAT
    2 => AttributeProto_AttributeType::INT
    3 => AttributeProto_AttributeType::STRING
    4 => AttributeProto_AttributeType::TENSOR
    5 => AttributeProto_AttributeType::GRAPH
    11 => AttributeProto_AttributeType::SPARSE_TENSOR
    13 => AttributeProto_AttributeType::TYPE_PROTO
    6 => AttributeProto_AttributeType::FLOATS
    7 => AttributeProto_AttributeType::INTS
    8 => AttributeProto_AttributeType::STRINGS
    9 => AttributeProto_AttributeType::TENSORS
    10 => AttributeProto_AttributeType::GRAPHS
    12 => AttributeProto_AttributeType::SPARSE_TENSORS
    14 => AttributeProto_AttributeType::TYPE_PROTOS
    _ => Default::default()
  }
}

///|
pub impl Default for AttributeProto_AttributeType with fn default() -> AttributeProto_AttributeType {
  AttributeProto_AttributeType::UNDEFINED
}

///|
pub impl @lib.Sized for AttributeProto_AttributeType with fn size_of(
  self : AttributeProto_AttributeType,
) {
  @lib.Sized::size_of(self.to_enum())
}

///|
#deprecated("Use `==` instead")
pub extend AttributeProto_AttributeType with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend AttributeProto_AttributeType with Eq::{not_equal}

///|
pub extend AttributeProto_AttributeType with Debug::{to_repr}

///|
pub extend AttributeProto_AttributeType with Default::{default}

///|
pub extend AttributeProto_AttributeType with @lib.Sized::{size_of}

///|
#derive(Eq, Debug)
pub(all) struct AttributeProto {
  mut name : String?
  mut ref_attr_name : String?
  mut doc_string : String?
  mut type_ : AttributeProto_AttributeType?
  mut f : Float?
  mut i : Int64?
  mut s : Bytes?
  mut t : TensorProto?
  mut g : GraphProto?
  mut sparse_tensor : SparseTensorProto?
  mut tp : TypeProto?
  mut floats : Array[Float]
  mut ints : Array[Int64]
  mut strings : Array[Bytes]
  mut tensors : Array[TensorProto]
  mut graphs : Array[GraphProto]
  mut sparse_tensors : Array[SparseTensorProto]
  mut type_protos : Array[TypeProto]
}

///|
pub impl @lib.Sized for AttributeProto with fn size_of(self) {
  let mut size = 0U
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.ref_attr_name is Some(v) {
    size += @lib.size_of_length_delimited_field(21U, @lib.size_of(v))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(13U, @lib.size_of(v))
  }
  if self.type_ is Some(v) {
    size += @lib.size_of_field(20U, @lib.size_of(v))
  }
  if self.f is Some(_) {
    size += @lib.size_of_field(2U, 4U)
  }
  if self.i is Some(v) {
    size += @lib.size_of_field(3U, @lib.size_of(v))
  }
  if self.s is Some(v) {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(v))
  }
  if self.t is Some(v) {
    size += @lib.size_of_length_delimited_field(5U, @lib.size_of(v))
  }
  if self.g is Some(v) {
    size += @lib.size_of_length_delimited_field(6U, @lib.size_of(v))
  }
  if self.sparse_tensor is Some(v) {
    size += @lib.size_of_length_delimited_field(22U, @lib.size_of(v))
  }
  if self.tp is Some(v) {
    size += @lib.size_of_length_delimited_field(14U, @lib.size_of(v))
  }
  for s in self.floats {
    size += @lib.size_of_field(7U, 4U)
  }
  for s in self.ints {
    size += @lib.size_of_field(8U, @lib.size_of(s))
  }
  for s in self.strings {
    size += @lib.size_of_length_delimited_field(9U, @lib.size_of(s))
  }
  for s in self.tensors {
    size += @lib.size_of_length_delimited_field(10U, @lib.size_of(s))
  }
  for s in self.graphs {
    size += @lib.size_of_length_delimited_field(11U, @lib.size_of(s))
  }
  for s in self.sparse_tensors {
    size += @lib.size_of_length_delimited_field(23U, @lib.size_of(s))
  }
  for s in self.type_protos {
    size += @lib.size_of_length_delimited_field(15U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for AttributeProto with fn default() -> AttributeProto {
  AttributeProto::{
    name: None,
    ref_attr_name: None,
    doc_string: None,
    type_: None,
    f: None,
    i: None,
    s: None,
    t: None,
    g: None,
    sparse_tensor: None,
    tp: None,
    floats: [],
    ints: [],
    strings: [],
    tensors: [],
    graphs: [],
    sparse_tensors: [],
    type_protos: [],
  }
}

///|
#alias(new, deprecated)
pub fn AttributeProto::AttributeProto(
  name? : String,
  ref_attr_name? : String,
  doc_string? : String,
  type_? : AttributeProto_AttributeType,
  f? : Float,
  i? : Int64,
  s? : Bytes,
  t? : TensorProto,
  g? : GraphProto,
  sparse_tensor? : SparseTensorProto,
  tp? : TypeProto,
  floats : Array[Float],
  ints : Array[Int64],
  strings : Array[Bytes],
  tensors : Array[TensorProto],
  graphs : Array[GraphProto],
  sparse_tensors : Array[SparseTensorProto],
  type_protos : Array[TypeProto],
) -> AttributeProto {
  AttributeProto::{
    name,
    ref_attr_name,
    doc_string,
    type_,
    f,
    i,
    s,
    t,
    g,
    sparse_tensor,
    tp,
    floats,
    ints,
    strings,
    tensors,
    graphs,
    sparse_tensors,
    type_protos,
  }
}

///|
pub impl @lib.Read for AttributeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> AttributeProto raise {
  let msg : AttributeProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.read_string() |> Some
      (21, _) => msg.ref_attr_name = reader |> @lib.read_string() |> Some
      (13, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (20, _) =>
        msg.type_ = reader
          |> @lib.read_enum()
          |> AttributeProto_AttributeType::from_enum
          |> Some
      (2, _) => msg.f = reader |> @lib.read_float() |> Some
      (3, _) => msg.i = reader |> @lib.read_int64() |> Some
      (4, _) => msg.s = reader |> @lib.read_bytes() |> Some
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.t = (reader |> @lib.read_message() : TensorProto) |> Some
      (6, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.g = (reader |> @lib.read_message() : GraphProto) |> Some
      (22, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sparse_tensor = (reader |> @lib.read_message() : SparseTensorProto)
          |> Some
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.tp = (reader |> @lib.read_message() : TypeProto) |> Some
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.floats.push_iter(
          (reader |> @lib.read_packed(@lib.read_float, Some(4))).iter(),
        )
      (7, @lib.WIRE_TYPE_FIXED32) =>
        msg.floats.push(reader |> @lib.read_float())
      (8, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.ints.push_iter(
          (reader |> @lib.read_packed(@lib.read_int64, None)).iter(),
        )
      (8, @lib.WIRE_TYPE_VARINT) => msg.ints.push(reader |> @lib.read_int64())
      (9, _) => msg.strings.push(reader |> @lib.read_bytes())
      (10, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.tensors.push((reader |> @lib.read_message() : TensorProto))
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.graphs.push((reader |> @lib.read_message() : GraphProto))
      (23, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sparse_tensors.push(
          (reader |> @lib.read_message() : SparseTensorProto),
        )
      (15, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.type_protos.push((reader |> @lib.read_message() : TypeProto))
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for AttributeProto with fn write(
  self : AttributeProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  if self.ref_attr_name is Some(v) {
    writer |> @lib.write_varint(170UL)
    writer |> @lib.write_string(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(106UL)
    writer |> @lib.write_string(v)
  }
  if self.type_ is Some(v) {
    writer |> @lib.write_varint(160UL)
    writer |> @lib.write_enum(v.to_enum())
  }
  if self.f is Some(v) {
    writer |> @lib.write_varint(21UL)
    writer |> @lib.write_float(v)
  }
  if self.i is Some(v) {
    writer |> @lib.write_varint(24UL)
    writer |> @lib.write_int64(v)
  }
  if self.s is Some(v) {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_bytes(v)
  }
  if self.t is Some(v) {
    writer |> @lib.write_varint(42UL)
    writer |> @lib.write_message(v)
  }
  if self.g is Some(v) {
    writer |> @lib.write_varint(50UL)
    writer |> @lib.write_message(v)
  }
  if self.sparse_tensor is Some(v) {
    writer |> @lib.write_varint(178UL)
    writer |> @lib.write_message(v)
  }
  if self.tp is Some(v) {
    writer |> @lib.write_varint(114UL)
    writer |> @lib.write_message(v)
  }
  for item in self.floats {
    writer |> @lib.write_varint(61UL)
    writer |> @lib.write_float(item)
  }
  for item in self.ints {
    writer |> @lib.write_varint(64UL)
    writer |> @lib.write_int64(item)
  }
  for item in self.strings {
    writer |> @lib.write_varint(74UL)
    writer |> @lib.write_bytes(item)
  }
  for item in self.tensors {
    writer |> @lib.write_varint(82UL)
    writer |> @lib.write_message(item)
  }
  for item in self.graphs {
    writer |> @lib.write_varint(90UL)
    writer |> @lib.write_message(item)
  }
  for item in self.sparse_tensors {
    writer |> @lib.write_varint(186UL)
    writer |> @lib.write_message(item)
  }
  for item in self.type_protos {
    writer |> @lib.write_varint(122UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for AttributeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> AttributeProto raise {
  let msg : AttributeProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (21, _) => msg.ref_attr_name = reader |> @lib.async_read_string() |> Some
      (13, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (20, _) =>
        msg.type_ = reader
          |> @lib.async_read_enum()
          |> AttributeProto_AttributeType::from_enum
          |> Some
      (2, _) => msg.f = reader |> @lib.async_read_float() |> Some
      (3, _) => msg.i = reader |> @lib.async_read_int64() |> Some
      (4, _) => msg.s = reader |> @lib.async_read_bytes() |> Some
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.t = (reader |> @lib.async_read_message() : TensorProto) |> Some
      (6, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.g = (reader |> @lib.async_read_message() : GraphProto) |> Some
      (22, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sparse_tensor = (
            reader |> @lib.async_read_message() : SparseTensorProto)
          |> Some
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.tp = (reader |> @lib.async_read_message() : TypeProto) |> Some
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.floats.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_float, Some(4))).iter(),
        )
      (7, @lib.WIRE_TYPE_FIXED32) =>
        msg.floats.push(reader |> @lib.async_read_float())
      (8, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.ints.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int64, None)).iter(),
        )
      (8, @lib.WIRE_TYPE_VARINT) =>
        msg.ints.push(reader |> @lib.async_read_int64())
      (9, _) => msg.strings.push(reader |> @lib.async_read_bytes())
      (10, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.tensors.push((reader |> @lib.async_read_message() : TensorProto))
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.graphs.push((reader |> @lib.async_read_message() : GraphProto))
      (23, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sparse_tensors.push(
          (reader |> @lib.async_read_message() : SparseTensorProto),
        )
      (15, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.type_protos.push((reader |> @lib.async_read_message() : TypeProto))
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for AttributeProto with fn write(
  self : AttributeProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  if self.ref_attr_name is Some(v) {
    writer |> @lib.async_write_varint(170UL)
    writer |> @lib.async_write_string(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(106UL)
    writer |> @lib.async_write_string(v)
  }
  if self.type_ is Some(v) {
    writer |> @lib.async_write_varint(160UL)
    writer |> @lib.async_write_enum(v.to_enum())
  }
  if self.f is Some(v) {
    writer |> @lib.async_write_varint(21UL)
    writer |> @lib.async_write_float(v)
  }
  if self.i is Some(v) {
    writer |> @lib.async_write_varint(24UL)
    writer |> @lib.async_write_int64(v)
  }
  if self.s is Some(v) {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_bytes(v)
  }
  if self.t is Some(v) {
    writer |> @lib.async_write_varint(42UL)
    writer |> @lib.async_write_message(v)
  }
  if self.g is Some(v) {
    writer |> @lib.async_write_varint(50UL)
    writer |> @lib.async_write_message(v)
  }
  if self.sparse_tensor is Some(v) {
    writer |> @lib.async_write_varint(178UL)
    writer |> @lib.async_write_message(v)
  }
  if self.tp is Some(v) {
    writer |> @lib.async_write_varint(114UL)
    writer |> @lib.async_write_message(v)
  }
  for item in self.floats {
    writer |> @lib.async_write_varint(61UL)
    writer |> @lib.async_write_float(item)
  }
  for item in self.ints {
    writer |> @lib.async_write_varint(64UL)
    writer |> @lib.async_write_int64(item)
  }
  for item in self.strings {
    writer |> @lib.async_write_varint(74UL)
    writer |> @lib.async_write_bytes(item)
  }
  for item in self.tensors {
    writer |> @lib.async_write_varint(82UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.graphs {
    writer |> @lib.async_write_varint(90UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.sparse_tensors {
    writer |> @lib.async_write_varint(186UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.type_protos {
    writer |> @lib.async_write_varint(122UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend AttributeProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend AttributeProto with Eq::{not_equal}

///|
pub extend AttributeProto with Debug::{to_repr}

///|
pub extend AttributeProto with Default::{default}

///|
pub extend AttributeProto with @lib.Sized::{size_of}

///|
pub extend AttributeProto with @lib.Read::{read, read_with_limit}

///|
pub extend AttributeProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct ValueInfoProto {
  mut name : String?
  mut type_ : TypeProto?
  mut doc_string : String?
  mut metadata_props : Array[StringStringEntryProto]
}

///|
pub impl @lib.Sized for ValueInfoProto with fn size_of(self) {
  let mut size = 0U
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.type_ is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(v))
  }
  for s in self.metadata_props {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for ValueInfoProto with fn default() -> ValueInfoProto {
  ValueInfoProto::{
    name: None,
    type_: None,
    doc_string: None,
    metadata_props: [],
  }
}

///|
#alias(new, deprecated)
pub fn ValueInfoProto::ValueInfoProto(
  name? : String,
  type_? : TypeProto,
  doc_string? : String,
  metadata_props : Array[StringStringEntryProto],
) -> ValueInfoProto {
  ValueInfoProto::{ name, type_, doc_string, metadata_props, }
}

///|
pub impl @lib.Read for ValueInfoProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> ValueInfoProto raise {
  let msg : ValueInfoProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.type_ = (reader |> @lib.read_message() : TypeProto) |> Some
      (3, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for ValueInfoProto with fn write(
  self : ValueInfoProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  if self.type_ is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_string(v)
  }
  for item in self.metadata_props {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for ValueInfoProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> ValueInfoProto raise {
  let msg : ValueInfoProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.type_ = (reader |> @lib.async_read_message() : TypeProto) |> Some
      (3, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for ValueInfoProto with fn write(
  self : ValueInfoProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  if self.type_ is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.metadata_props {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend ValueInfoProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend ValueInfoProto with Eq::{not_equal}

///|
pub extend ValueInfoProto with Debug::{to_repr}

///|
pub extend ValueInfoProto with Default::{default}

///|
pub extend ValueInfoProto with @lib.Sized::{size_of}

///|
pub extend ValueInfoProto with @lib.Read::{read, read_with_limit}

///|
pub extend ValueInfoProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct NodeProto {
  mut input : Array[String]
  mut output : Array[String]
  mut name : String?
  mut op_type : String?
  mut domain : String?
  mut overload : String?
  mut attribute : Array[AttributeProto]
  mut doc_string : String?
  mut metadata_props : Array[StringStringEntryProto]
  mut device_configurations : Array[NodeDeviceConfigurationProto]
}

///|
pub impl @lib.Sized for NodeProto with fn size_of(self) {
  let mut size = 0U
  for s in self.input {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(s))
  }
  for s in self.output {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(s))
  }
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(v))
  }
  if self.op_type is Some(v) {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(v))
  }
  if self.domain is Some(v) {
    size += @lib.size_of_length_delimited_field(7U, @lib.size_of(v))
  }
  if self.overload is Some(v) {
    size += @lib.size_of_length_delimited_field(8U, @lib.size_of(v))
  }
  for s in self.attribute {
    size += @lib.size_of_length_delimited_field(5U, @lib.size_of(s))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(6U, @lib.size_of(v))
  }
  for s in self.metadata_props {
    size += @lib.size_of_length_delimited_field(9U, @lib.size_of(s))
  }
  for s in self.device_configurations {
    size += @lib.size_of_length_delimited_field(10U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for NodeProto with fn default() -> NodeProto {
  NodeProto::{
    input: [],
    output: [],
    name: None,
    op_type: None,
    domain: None,
    overload: None,
    attribute: [],
    doc_string: None,
    metadata_props: [],
    device_configurations: [],
  }
}

///|
#alias(new, deprecated)
pub fn NodeProto::NodeProto(
  input : Array[String],
  output : Array[String],
  name? : String,
  op_type? : String,
  domain? : String,
  overload? : String,
  attribute : Array[AttributeProto],
  doc_string? : String,
  metadata_props : Array[StringStringEntryProto],
  device_configurations : Array[NodeDeviceConfigurationProto],
) -> NodeProto {
  NodeProto::{
    input,
    output,
    name,
    op_type,
    domain,
    overload,
    attribute,
    doc_string,
    metadata_props,
    device_configurations,
  }
}

///|
pub impl @lib.Read for NodeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> NodeProto raise {
  let msg : NodeProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.input.push(reader |> @lib.read_string())
      (2, _) => msg.output.push(reader |> @lib.read_string())
      (3, _) => msg.name = reader |> @lib.read_string() |> Some
      (4, _) => msg.op_type = reader |> @lib.read_string() |> Some
      (7, _) => msg.domain = reader |> @lib.read_string() |> Some
      (8, _) => msg.overload = reader |> @lib.read_string() |> Some
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.attribute.push((reader |> @lib.read_message() : AttributeProto))
      (6, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (9, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      (10, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.device_configurations.push(
          (reader |> @lib.read_message() : NodeDeviceConfigurationProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for NodeProto with fn write(
  self : NodeProto,
  writer : &@lib.Writer,
) -> Unit raise {
  for item in self.input {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(item)
  }
  for item in self.output {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_string(item)
  }
  if self.name is Some(v) {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_string(v)
  }
  if self.op_type is Some(v) {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_string(v)
  }
  if self.domain is Some(v) {
    writer |> @lib.write_varint(58UL)
    writer |> @lib.write_string(v)
  }
  if self.overload is Some(v) {
    writer |> @lib.write_varint(66UL)
    writer |> @lib.write_string(v)
  }
  for item in self.attribute {
    writer |> @lib.write_varint(42UL)
    writer |> @lib.write_message(item)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(50UL)
    writer |> @lib.write_string(v)
  }
  for item in self.metadata_props {
    writer |> @lib.write_varint(74UL)
    writer |> @lib.write_message(item)
  }
  for item in self.device_configurations {
    writer |> @lib.write_varint(82UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for NodeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> NodeProto raise {
  let msg : NodeProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.input.push(reader |> @lib.async_read_string())
      (2, _) => msg.output.push(reader |> @lib.async_read_string())
      (3, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (4, _) => msg.op_type = reader |> @lib.async_read_string() |> Some
      (7, _) => msg.domain = reader |> @lib.async_read_string() |> Some
      (8, _) => msg.overload = reader |> @lib.async_read_string() |> Some
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.attribute.push(
          (reader |> @lib.async_read_message() : AttributeProto),
        )
      (6, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (9, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      (10, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.device_configurations.push(
          (reader |> @lib.async_read_message() : NodeDeviceConfigurationProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for NodeProto with fn write(
  self : NodeProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  for item in self.input {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(item)
  }
  for item in self.output {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_string(item)
  }
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_string(v)
  }
  if self.op_type is Some(v) {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_string(v)
  }
  if self.domain is Some(v) {
    writer |> @lib.async_write_varint(58UL)
    writer |> @lib.async_write_string(v)
  }
  if self.overload is Some(v) {
    writer |> @lib.async_write_varint(66UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.attribute {
    writer |> @lib.async_write_varint(42UL)
    writer |> @lib.async_write_message(item)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(50UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.metadata_props {
    writer |> @lib.async_write_varint(74UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.device_configurations {
    writer |> @lib.async_write_varint(82UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend NodeProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend NodeProto with Eq::{not_equal}

///|
pub extend NodeProto with Debug::{to_repr}

///|
pub extend NodeProto with Default::{default}

///|
pub extend NodeProto with @lib.Sized::{size_of}

///|
pub extend NodeProto with @lib.Read::{read, read_with_limit}

///|
pub extend NodeProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct IntIntListEntryProto {
  mut key : Int64?
  mut value : Array[Int64]
}

///|
pub impl @lib.Sized for IntIntListEntryProto with fn size_of(self) {
  let mut size = 0U
  if self.key is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  for s in self.value {
    size += @lib.size_of_field(2U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for IntIntListEntryProto with fn default() -> IntIntListEntryProto {
  IntIntListEntryProto::{ key: None, value: [], }
}

///|
#alias(new, deprecated)
pub fn IntIntListEntryProto::IntIntListEntryProto(
  key? : Int64,
  value : Array[Int64],
) -> IntIntListEntryProto {
  IntIntListEntryProto::{ key, value, }
}

///|
pub impl @lib.Read for IntIntListEntryProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> IntIntListEntryProto raise {
  let msg : IntIntListEntryProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.key = reader |> @lib.read_int64() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value.push_iter(
          (reader |> @lib.read_packed(@lib.read_int64, None)).iter(),
        )
      (2, @lib.WIRE_TYPE_VARINT) => msg.value.push(reader |> @lib.read_int64())
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for IntIntListEntryProto with fn write(
  self : IntIntListEntryProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.key is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int64(v)
  }
  for item in self.value {
    writer |> @lib.write_varint(16UL)
    writer |> @lib.write_int64(item)
  }
}

///|
pub impl @lib.AsyncRead for IntIntListEntryProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> IntIntListEntryProto raise {
  let msg : IntIntListEntryProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.key = reader |> @lib.async_read_int64() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int64, None)).iter(),
        )
      (2, @lib.WIRE_TYPE_VARINT) =>
        msg.value.push(reader |> @lib.async_read_int64())
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for IntIntListEntryProto with fn write(
  self : IntIntListEntryProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.key is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int64(v)
  }
  for item in self.value {
    writer |> @lib.async_write_varint(16UL)
    writer |> @lib.async_write_int64(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend IntIntListEntryProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend IntIntListEntryProto with Eq::{not_equal}

///|
pub extend IntIntListEntryProto with Debug::{to_repr}

///|
pub extend IntIntListEntryProto with Default::{default}

///|
pub extend IntIntListEntryProto with @lib.Sized::{size_of}

///|
pub extend IntIntListEntryProto with @lib.Read::{read, read_with_limit}

///|
pub extend IntIntListEntryProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct NodeDeviceConfigurationProto {
  mut configuration_id : String?
  mut sharding_spec : Array[ShardingSpecProto]
  mut pipeline_stage : Int32?
}

///|
pub impl @lib.Sized for NodeDeviceConfigurationProto with fn size_of(self) {
  let mut size = 0U
  if self.configuration_id is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  for s in self.sharding_spec {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(s))
  }
  if self.pipeline_stage is Some(v) {
    size += @lib.size_of_field(3U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for NodeDeviceConfigurationProto with fn default() -> NodeDeviceConfigurationProto {
  NodeDeviceConfigurationProto::{
    configuration_id: None,
    sharding_spec: [],
    pipeline_stage: None,
  }
}

///|
#alias(new, deprecated)
pub fn NodeDeviceConfigurationProto::NodeDeviceConfigurationProto(
  configuration_id? : String,
  sharding_spec : Array[ShardingSpecProto],
  pipeline_stage? : Int32,
) -> NodeDeviceConfigurationProto {
  NodeDeviceConfigurationProto::{
    configuration_id,
    sharding_spec,
    pipeline_stage,
  }
}

///|
pub impl @lib.Read for NodeDeviceConfigurationProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> NodeDeviceConfigurationProto raise {
  let msg : NodeDeviceConfigurationProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.configuration_id = reader |> @lib.read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sharding_spec.push(
          (reader |> @lib.read_message() : ShardingSpecProto),
        )
      (3, _) => msg.pipeline_stage = reader |> @lib.read_int32() |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for NodeDeviceConfigurationProto with fn write(
  self : NodeDeviceConfigurationProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.configuration_id is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  for item in self.sharding_spec {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(item)
  }
  if self.pipeline_stage is Some(v) {
    writer |> @lib.write_varint(24UL)
    writer |> @lib.write_int32(v)
  }
}

///|
pub impl @lib.AsyncRead for NodeDeviceConfigurationProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> NodeDeviceConfigurationProto raise {
  let msg : NodeDeviceConfigurationProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) =>
        msg.configuration_id = reader |> @lib.async_read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sharding_spec.push(
          (reader |> @lib.async_read_message() : ShardingSpecProto),
        )
      (3, _) => msg.pipeline_stage = reader |> @lib.async_read_int32() |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for NodeDeviceConfigurationProto with fn write(
  self : NodeDeviceConfigurationProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.configuration_id is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.sharding_spec {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(item)
  }
  if self.pipeline_stage is Some(v) {
    writer |> @lib.async_write_varint(24UL)
    writer |> @lib.async_write_int32(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend NodeDeviceConfigurationProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend NodeDeviceConfigurationProto with Eq::{not_equal}

///|
pub extend NodeDeviceConfigurationProto with Debug::{to_repr}

///|
pub extend NodeDeviceConfigurationProto with Default::{default}

///|
pub extend NodeDeviceConfigurationProto with @lib.Sized::{size_of}

///|
pub extend NodeDeviceConfigurationProto with @lib.Read::{read, read_with_limit}

///|
pub extend NodeDeviceConfigurationProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct ShardingSpecProto {
  mut tensor_name : String?
  mut device : Array[Int64]
  mut index_to_device_group_map : Array[IntIntListEntryProto]
  mut sharded_dim : Array[ShardedDimProto]
}

///|
pub impl @lib.Sized for ShardingSpecProto with fn size_of(self) {
  let mut size = 0U
  if self.tensor_name is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  for s in self.device {
    size += @lib.size_of_field(2U, @lib.size_of(s))
  }
  for s in self.index_to_device_group_map {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(s))
  }
  for s in self.sharded_dim {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for ShardingSpecProto with fn default() -> ShardingSpecProto {
  ShardingSpecProto::{
    tensor_name: None,
    device: [],
    index_to_device_group_map: [],
    sharded_dim: [],
  }
}

///|
#alias(new, deprecated)
pub fn ShardingSpecProto::ShardingSpecProto(
  tensor_name? : String,
  device : Array[Int64],
  index_to_device_group_map : Array[IntIntListEntryProto],
  sharded_dim : Array[ShardedDimProto],
) -> ShardingSpecProto {
  ShardingSpecProto::{
    tensor_name,
    device,
    index_to_device_group_map,
    sharded_dim,
  }
}

///|
pub impl @lib.Read for ShardingSpecProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> ShardingSpecProto raise {
  let msg : ShardingSpecProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.tensor_name = reader |> @lib.read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.device.push_iter(
          (reader |> @lib.read_packed(@lib.read_int64, None)).iter(),
        )
      (2, @lib.WIRE_TYPE_VARINT) => msg.device.push(reader |> @lib.read_int64())
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.index_to_device_group_map.push(
          (reader |> @lib.read_message() : IntIntListEntryProto),
        )
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sharded_dim.push((reader |> @lib.read_message() : ShardedDimProto))
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for ShardingSpecProto with fn write(
  self : ShardingSpecProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.tensor_name is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  for item in self.device {
    writer |> @lib.write_varint(16UL)
    writer |> @lib.write_int64(item)
  }
  for item in self.index_to_device_group_map {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_message(item)
  }
  for item in self.sharded_dim {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for ShardingSpecProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> ShardingSpecProto raise {
  let msg : ShardingSpecProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.tensor_name = reader |> @lib.async_read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.device.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int64, None)).iter(),
        )
      (2, @lib.WIRE_TYPE_VARINT) =>
        msg.device.push(reader |> @lib.async_read_int64())
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.index_to_device_group_map.push(
          (reader |> @lib.async_read_message() : IntIntListEntryProto),
        )
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sharded_dim.push(
          (reader |> @lib.async_read_message() : ShardedDimProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for ShardingSpecProto with fn write(
  self : ShardingSpecProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.tensor_name is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.device {
    writer |> @lib.async_write_varint(16UL)
    writer |> @lib.async_write_int64(item)
  }
  for item in self.index_to_device_group_map {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.sharded_dim {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend ShardingSpecProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend ShardingSpecProto with Eq::{not_equal}

///|
pub extend ShardingSpecProto with Debug::{to_repr}

///|
pub extend ShardingSpecProto with Default::{default}

///|
pub extend ShardingSpecProto with @lib.Sized::{size_of}

///|
pub extend ShardingSpecProto with @lib.Read::{read, read_with_limit}

///|
pub extend ShardingSpecProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct ShardedDimProto {
  mut axis : Int64?
  mut simple_sharding : Array[SimpleShardedDimProto]
}

///|
pub impl @lib.Sized for ShardedDimProto with fn size_of(self) {
  let mut size = 0U
  if self.axis is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  for s in self.simple_sharding {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for ShardedDimProto with fn default() -> ShardedDimProto {
  ShardedDimProto::{ axis: None, simple_sharding: [], }
}

///|
#alias(new, deprecated)
pub fn ShardedDimProto::ShardedDimProto(
  axis? : Int64,
  simple_sharding : Array[SimpleShardedDimProto],
) -> ShardedDimProto {
  ShardedDimProto::{ axis, simple_sharding, }
}

///|
pub impl @lib.Read for ShardedDimProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> ShardedDimProto raise {
  let msg : ShardedDimProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.axis = reader |> @lib.read_int64() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.simple_sharding.push(
          (reader |> @lib.read_message() : SimpleShardedDimProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for ShardedDimProto with fn write(
  self : ShardedDimProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.axis is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int64(v)
  }
  for item in self.simple_sharding {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for ShardedDimProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> ShardedDimProto raise {
  let msg : ShardedDimProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.axis = reader |> @lib.async_read_int64() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.simple_sharding.push(
          (reader |> @lib.async_read_message() : SimpleShardedDimProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for ShardedDimProto with fn write(
  self : ShardedDimProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.axis is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int64(v)
  }
  for item in self.simple_sharding {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend ShardedDimProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend ShardedDimProto with Eq::{not_equal}

///|
pub extend ShardedDimProto with Debug::{to_repr}

///|
pub extend ShardedDimProto with Default::{default}

///|
pub extend ShardedDimProto with @lib.Sized::{size_of}

///|
pub extend ShardedDimProto with @lib.Read::{read, read_with_limit}

///|
pub extend ShardedDimProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct SimpleShardedDimProto {
  mut num_shards : Int64?
  mut dim : SimpleShardedDimProto_Dim
}

///|
#derive(Eq, Debug)
pub(all) enum SimpleShardedDimProto_Dim {
  DimValue(Int64)
  DimParam(String)
  NotSet
}

///|
pub impl Default for SimpleShardedDimProto_Dim with fn default() -> SimpleShardedDimProto_Dim {
  NotSet
}

///|
#deprecated("Use `==` instead")
pub extend SimpleShardedDimProto_Dim with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend SimpleShardedDimProto_Dim with Eq::{not_equal}

///|
pub extend SimpleShardedDimProto_Dim with Debug::{to_repr}

///|
pub extend SimpleShardedDimProto_Dim with Default::{default}

///|
pub impl @lib.Sized for SimpleShardedDimProto with fn size_of(self) {
  let mut size = 0U
  if self.num_shards is Some(v) {
    size += @lib.size_of_field(3U, @lib.size_of(v))
  }
  match self.dim {
    DimValue(v) => size += @lib.size_of_field(1U, @lib.size_of(v))
    DimParam(v) =>
      size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
    NotSet => ()
  }
  size
}

///|
pub impl Default for SimpleShardedDimProto with fn default() -> SimpleShardedDimProto {
  SimpleShardedDimProto::{
    num_shards: None,
    dim: SimpleShardedDimProto_Dim::NotSet,
  }
}

///|
#alias(new, deprecated)
pub fn SimpleShardedDimProto::SimpleShardedDimProto(
  num_shards? : Int64,
  dim~ : SimpleShardedDimProto_Dim = SimpleShardedDimProto_Dim::NotSet,
) -> SimpleShardedDimProto {
  SimpleShardedDimProto::{ num_shards, dim, }
}

///|
pub impl @lib.Read for SimpleShardedDimProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> SimpleShardedDimProto raise {
  let msg : SimpleShardedDimProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (3, _) => msg.num_shards = reader |> @lib.read_int64() |> Some
      (1, _) =>
        msg.dim = reader
          |> @lib.read_int64()
          |> SimpleShardedDimProto_Dim::DimValue
      (2, _) =>
        msg.dim = reader
          |> @lib.read_string()
          |> SimpleShardedDimProto_Dim::DimParam
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for SimpleShardedDimProto with fn write(
  self : SimpleShardedDimProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.num_shards is Some(v) {
    writer |> @lib.write_varint(24UL)
    writer |> @lib.write_int64(v)
  }
  match self.dim {
    SimpleShardedDimProto_Dim::DimValue(v) => {
      writer |> @lib.write_varint(8UL)
      writer |> @lib.write_int64(v)
    }
    SimpleShardedDimProto_Dim::DimParam(v) => {
      writer |> @lib.write_varint(18UL)
      writer |> @lib.write_string(v)
    }
    SimpleShardedDimProto_Dim::NotSet => ()
  }
}

///|
pub impl @lib.AsyncRead for SimpleShardedDimProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> SimpleShardedDimProto raise {
  let msg : SimpleShardedDimProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (3, _) => msg.num_shards = reader |> @lib.async_read_int64() |> Some
      (1, _) =>
        msg.dim = reader
          |> @lib.async_read_int64()
          |> SimpleShardedDimProto_Dim::DimValue
      (2, _) =>
        msg.dim = reader
          |> @lib.async_read_string()
          |> SimpleShardedDimProto_Dim::DimParam
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for SimpleShardedDimProto with fn write(
  self : SimpleShardedDimProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.num_shards is Some(v) {
    writer |> @lib.async_write_varint(24UL)
    writer |> @lib.async_write_int64(v)
  }
  match self.dim {
    SimpleShardedDimProto_Dim::DimValue(v) => {
      writer |> @lib.async_write_varint(8UL)
      writer |> @lib.async_write_int64(v)
    }
    SimpleShardedDimProto_Dim::DimParam(v) => {
      writer |> @lib.async_write_varint(18UL)
      writer |> @lib.async_write_string(v)
    }
    SimpleShardedDimProto_Dim::NotSet => ()
  }
}

///|
#deprecated("Use `==` instead")
pub extend SimpleShardedDimProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend SimpleShardedDimProto with Eq::{not_equal}

///|
pub extend SimpleShardedDimProto with Debug::{to_repr}

///|
pub extend SimpleShardedDimProto with Default::{default}

///|
pub extend SimpleShardedDimProto with @lib.Sized::{size_of}

///|
pub extend SimpleShardedDimProto with @lib.Read::{read, read_with_limit}

///|
pub extend SimpleShardedDimProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TrainingInfoProto {
  mut initialization : GraphProto?
  mut algorithm : GraphProto?
  mut initialization_binding : Array[StringStringEntryProto]
  mut update_binding : Array[StringStringEntryProto]
}

///|
pub impl @lib.Sized for TrainingInfoProto with fn size_of(self) {
  let mut size = 0U
  if self.initialization is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.algorithm is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  for s in self.initialization_binding {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(s))
  }
  for s in self.update_binding {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for TrainingInfoProto with fn default() -> TrainingInfoProto {
  TrainingInfoProto::{
    initialization: None,
    algorithm: None,
    initialization_binding: [],
    update_binding: [],
  }
}

///|
#alias(new, deprecated)
pub fn TrainingInfoProto::TrainingInfoProto(
  initialization? : GraphProto,
  algorithm? : GraphProto,
  initialization_binding : Array[StringStringEntryProto],
  update_binding : Array[StringStringEntryProto],
) -> TrainingInfoProto {
  TrainingInfoProto::{
    initialization,
    algorithm,
    initialization_binding,
    update_binding,
  }
}

///|
pub impl @lib.Read for TrainingInfoProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TrainingInfoProto raise {
  let msg : TrainingInfoProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.initialization = (reader |> @lib.read_message() : GraphProto)
          |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.algorithm = (reader |> @lib.read_message() : GraphProto) |> Some
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.initialization_binding.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.update_binding.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TrainingInfoProto with fn write(
  self : TrainingInfoProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.initialization is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_message(v)
  }
  if self.algorithm is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(v)
  }
  for item in self.initialization_binding {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_message(item)
  }
  for item in self.update_binding {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for TrainingInfoProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TrainingInfoProto raise {
  let msg : TrainingInfoProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.initialization = (reader |> @lib.async_read_message() : GraphProto)
          |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.algorithm = (reader |> @lib.async_read_message() : GraphProto)
          |> Some
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.initialization_binding.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.update_binding.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TrainingInfoProto with fn write(
  self : TrainingInfoProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.initialization is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_message(v)
  }
  if self.algorithm is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(v)
  }
  for item in self.initialization_binding {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.update_binding {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TrainingInfoProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TrainingInfoProto with Eq::{not_equal}

///|
pub extend TrainingInfoProto with Debug::{to_repr}

///|
pub extend TrainingInfoProto with Default::{default}

///|
pub extend TrainingInfoProto with @lib.Sized::{size_of}

///|
pub extend TrainingInfoProto with @lib.Read::{read, read_with_limit}

///|
pub extend TrainingInfoProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct ModelProto {
  mut ir_version : Int64?
  mut opset_import : Array[OperatorSetIdProto]
  mut producer_name : String?
  mut producer_version : String?
  mut domain : String?
  mut model_version : Int64?
  mut doc_string : String?
  mut graph : GraphProto?
  mut metadata_props : Array[StringStringEntryProto]
  mut training_info : Array[TrainingInfoProto]
  mut functions : Array[FunctionProto]
  mut configuration : Array[DeviceConfigurationProto]
}

///|
pub impl @lib.Sized for ModelProto with fn size_of(self) {
  let mut size = 0U
  if self.ir_version is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  for s in self.opset_import {
    size += @lib.size_of_length_delimited_field(8U, @lib.size_of(s))
  }
  if self.producer_name is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  if self.producer_version is Some(v) {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(v))
  }
  if self.domain is Some(v) {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(v))
  }
  if self.model_version is Some(v) {
    size += @lib.size_of_field(5U, @lib.size_of(v))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(6U, @lib.size_of(v))
  }
  if self.graph is Some(v) {
    size += @lib.size_of_length_delimited_field(7U, @lib.size_of(v))
  }
  for s in self.metadata_props {
    size += @lib.size_of_length_delimited_field(14U, @lib.size_of(s))
  }
  for s in self.training_info {
    size += @lib.size_of_length_delimited_field(20U, @lib.size_of(s))
  }
  for s in self.functions {
    size += @lib.size_of_length_delimited_field(25U, @lib.size_of(s))
  }
  for s in self.configuration {
    size += @lib.size_of_length_delimited_field(26U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for ModelProto with fn default() -> ModelProto {
  ModelProto::{
    ir_version: None,
    opset_import: [],
    producer_name: None,
    producer_version: None,
    domain: None,
    model_version: None,
    doc_string: None,
    graph: None,
    metadata_props: [],
    training_info: [],
    functions: [],
    configuration: [],
  }
}

///|
#alias(new, deprecated)
pub fn ModelProto::ModelProto(
  ir_version? : Int64,
  opset_import : Array[OperatorSetIdProto],
  producer_name? : String,
  producer_version? : String,
  domain? : String,
  model_version? : Int64,
  doc_string? : String,
  graph? : GraphProto,
  metadata_props : Array[StringStringEntryProto],
  training_info : Array[TrainingInfoProto],
  functions : Array[FunctionProto],
  configuration : Array[DeviceConfigurationProto],
) -> ModelProto {
  ModelProto::{
    ir_version,
    opset_import,
    producer_name,
    producer_version,
    domain,
    model_version,
    doc_string,
    graph,
    metadata_props,
    training_info,
    functions,
    configuration,
  }
}

///|
pub impl @lib.Read for ModelProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> ModelProto raise {
  let msg : ModelProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.ir_version = reader |> @lib.read_int64() |> Some
      (8, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.opset_import.push(
          (reader |> @lib.read_message() : OperatorSetIdProto),
        )
      (2, _) => msg.producer_name = reader |> @lib.read_string() |> Some
      (3, _) => msg.producer_version = reader |> @lib.read_string() |> Some
      (4, _) => msg.domain = reader |> @lib.read_string() |> Some
      (5, _) => msg.model_version = reader |> @lib.read_int64() |> Some
      (6, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.graph = (reader |> @lib.read_message() : GraphProto) |> Some
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      (20, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.training_info.push(
          (reader |> @lib.read_message() : TrainingInfoProto),
        )
      (25, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.functions.push((reader |> @lib.read_message() : FunctionProto))
      (26, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.configuration.push(
          (reader |> @lib.read_message() : DeviceConfigurationProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for ModelProto with fn write(
  self : ModelProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.ir_version is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int64(v)
  }
  for item in self.opset_import {
    writer |> @lib.write_varint(66UL)
    writer |> @lib.write_message(item)
  }
  if self.producer_name is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_string(v)
  }
  if self.producer_version is Some(v) {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_string(v)
  }
  if self.domain is Some(v) {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_string(v)
  }
  if self.model_version is Some(v) {
    writer |> @lib.write_varint(40UL)
    writer |> @lib.write_int64(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(50UL)
    writer |> @lib.write_string(v)
  }
  if self.graph is Some(v) {
    writer |> @lib.write_varint(58UL)
    writer |> @lib.write_message(v)
  }
  for item in self.metadata_props {
    writer |> @lib.write_varint(114UL)
    writer |> @lib.write_message(item)
  }
  for item in self.training_info {
    writer |> @lib.write_varint(162UL)
    writer |> @lib.write_message(item)
  }
  for item in self.functions {
    writer |> @lib.write_varint(202UL)
    writer |> @lib.write_message(item)
  }
  for item in self.configuration {
    writer |> @lib.write_varint(210UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for ModelProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> ModelProto raise {
  let msg : ModelProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.ir_version = reader |> @lib.async_read_int64() |> Some
      (8, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.opset_import.push(
          (reader |> @lib.async_read_message() : OperatorSetIdProto),
        )
      (2, _) => msg.producer_name = reader |> @lib.async_read_string() |> Some
      (3, _) =>
        msg.producer_version = reader |> @lib.async_read_string() |> Some
      (4, _) => msg.domain = reader |> @lib.async_read_string() |> Some
      (5, _) => msg.model_version = reader |> @lib.async_read_int64() |> Some
      (6, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.graph = (reader |> @lib.async_read_message() : GraphProto) |> Some
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      (20, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.training_info.push(
          (reader |> @lib.async_read_message() : TrainingInfoProto),
        )
      (25, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.functions.push(
          (reader |> @lib.async_read_message() : FunctionProto),
        )
      (26, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.configuration.push(
          (reader |> @lib.async_read_message() : DeviceConfigurationProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for ModelProto with fn write(
  self : ModelProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.ir_version is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int64(v)
  }
  for item in self.opset_import {
    writer |> @lib.async_write_varint(66UL)
    writer |> @lib.async_write_message(item)
  }
  if self.producer_name is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_string(v)
  }
  if self.producer_version is Some(v) {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_string(v)
  }
  if self.domain is Some(v) {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_string(v)
  }
  if self.model_version is Some(v) {
    writer |> @lib.async_write_varint(40UL)
    writer |> @lib.async_write_int64(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(50UL)
    writer |> @lib.async_write_string(v)
  }
  if self.graph is Some(v) {
    writer |> @lib.async_write_varint(58UL)
    writer |> @lib.async_write_message(v)
  }
  for item in self.metadata_props {
    writer |> @lib.async_write_varint(114UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.training_info {
    writer |> @lib.async_write_varint(162UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.functions {
    writer |> @lib.async_write_varint(202UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.configuration {
    writer |> @lib.async_write_varint(210UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend ModelProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend ModelProto with Eq::{not_equal}

///|
pub extend ModelProto with Debug::{to_repr}

///|
pub extend ModelProto with Default::{default}

///|
pub extend ModelProto with @lib.Sized::{size_of}

///|
pub extend ModelProto with @lib.Read::{read, read_with_limit}

///|
pub extend ModelProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct DeviceConfigurationProto {
  mut name : String?
  mut num_devices : Int32?
  mut device : Array[String]
}

///|
pub impl @lib.Sized for DeviceConfigurationProto with fn size_of(self) {
  let mut size = 0U
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.num_devices is Some(v) {
    size += @lib.size_of_field(2U, @lib.size_of(v))
  }
  for s in self.device {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for DeviceConfigurationProto with fn default() -> DeviceConfigurationProto {
  DeviceConfigurationProto::{ name: None, num_devices: None, device: [], }
}

///|
#alias(new, deprecated)
pub fn DeviceConfigurationProto::DeviceConfigurationProto(
  name? : String,
  num_devices? : Int32,
  device : Array[String],
) -> DeviceConfigurationProto {
  DeviceConfigurationProto::{ name, num_devices, device, }
}

///|
pub impl @lib.Read for DeviceConfigurationProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> DeviceConfigurationProto raise {
  let msg : DeviceConfigurationProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.read_string() |> Some
      (2, _) => msg.num_devices = reader |> @lib.read_int32() |> Some
      (3, _) => msg.device.push(reader |> @lib.read_string())
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for DeviceConfigurationProto with fn write(
  self : DeviceConfigurationProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  if self.num_devices is Some(v) {
    writer |> @lib.write_varint(16UL)
    writer |> @lib.write_int32(v)
  }
  for item in self.device {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_string(item)
  }
}

///|
pub impl @lib.AsyncRead for DeviceConfigurationProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> DeviceConfigurationProto raise {
  let msg : DeviceConfigurationProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (2, _) => msg.num_devices = reader |> @lib.async_read_int32() |> Some
      (3, _) => msg.device.push(reader |> @lib.async_read_string())
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for DeviceConfigurationProto with fn write(
  self : DeviceConfigurationProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  if self.num_devices is Some(v) {
    writer |> @lib.async_write_varint(16UL)
    writer |> @lib.async_write_int32(v)
  }
  for item in self.device {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_string(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend DeviceConfigurationProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend DeviceConfigurationProto with Eq::{not_equal}

///|
pub extend DeviceConfigurationProto with Debug::{to_repr}

///|
pub extend DeviceConfigurationProto with Default::{default}

///|
pub extend DeviceConfigurationProto with @lib.Sized::{size_of}

///|
pub extend DeviceConfigurationProto with @lib.Read::{read, read_with_limit}

///|
pub extend DeviceConfigurationProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct StringStringEntryProto {
  mut key : String?
  mut value : String?
}

///|
pub impl @lib.Sized for StringStringEntryProto with fn size_of(self) {
  let mut size = 0U
  if self.key is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.value is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for StringStringEntryProto with fn default() -> StringStringEntryProto {
  StringStringEntryProto::{ key: None, value: None, }
}

///|
#alias(new, deprecated)
pub fn StringStringEntryProto::StringStringEntryProto(
  key? : String,
  value? : String,
) -> StringStringEntryProto {
  StringStringEntryProto::{ key, value, }
}

///|
pub impl @lib.Read for StringStringEntryProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> StringStringEntryProto raise {
  let msg : StringStringEntryProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.key = reader |> @lib.read_string() |> Some
      (2, _) => msg.value = reader |> @lib.read_string() |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for StringStringEntryProto with fn write(
  self : StringStringEntryProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.key is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  if self.value is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_string(v)
  }
}

///|
pub impl @lib.AsyncRead for StringStringEntryProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> StringStringEntryProto raise {
  let msg : StringStringEntryProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.key = reader |> @lib.async_read_string() |> Some
      (2, _) => msg.value = reader |> @lib.async_read_string() |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for StringStringEntryProto with fn write(
  self : StringStringEntryProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.key is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  if self.value is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_string(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend StringStringEntryProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend StringStringEntryProto with Eq::{not_equal}

///|
pub extend StringStringEntryProto with Debug::{to_repr}

///|
pub extend StringStringEntryProto with Default::{default}

///|
pub extend StringStringEntryProto with @lib.Sized::{size_of}

///|
pub extend StringStringEntryProto with @lib.Read::{read, read_with_limit}

///|
pub extend StringStringEntryProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TensorAnnotation {
  mut tensor_name : String?
  mut quant_parameter_tensor_names : Array[StringStringEntryProto]
}

///|
pub impl @lib.Sized for TensorAnnotation with fn size_of(self) {
  let mut size = 0U
  if self.tensor_name is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  for s in self.quant_parameter_tensor_names {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for TensorAnnotation with fn default() -> TensorAnnotation {
  TensorAnnotation::{ tensor_name: None, quant_parameter_tensor_names: [], }
}

///|
#alias(new, deprecated)
pub fn TensorAnnotation::TensorAnnotation(
  tensor_name? : String,
  quant_parameter_tensor_names : Array[StringStringEntryProto],
) -> TensorAnnotation {
  TensorAnnotation::{ tensor_name, quant_parameter_tensor_names, }
}

///|
pub impl @lib.Read for TensorAnnotation with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TensorAnnotation raise {
  let msg : TensorAnnotation = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.tensor_name = reader |> @lib.read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.quant_parameter_tensor_names.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TensorAnnotation with fn write(
  self : TensorAnnotation,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.tensor_name is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  for item in self.quant_parameter_tensor_names {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for TensorAnnotation with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TensorAnnotation raise {
  let msg : TensorAnnotation = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.tensor_name = reader |> @lib.async_read_string() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.quant_parameter_tensor_names.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TensorAnnotation with fn write(
  self : TensorAnnotation,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.tensor_name is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.quant_parameter_tensor_names {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TensorAnnotation with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorAnnotation with Eq::{not_equal}

///|
pub extend TensorAnnotation with Debug::{to_repr}

///|
pub extend TensorAnnotation with Default::{default}

///|
pub extend TensorAnnotation with @lib.Sized::{size_of}

///|
pub extend TensorAnnotation with @lib.Read::{read, read_with_limit}

///|
pub extend TensorAnnotation with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct GraphProto {
  mut node : Array[NodeProto]
  mut name : String?
  mut initializer : Array[TensorProto]
  mut sparse_initializer : Array[SparseTensorProto]
  mut doc_string : String?
  mut input : Array[ValueInfoProto]
  mut output : Array[ValueInfoProto]
  mut value_info : Array[ValueInfoProto]
  mut quantization_annotation : Array[TensorAnnotation]
  mut metadata_props : Array[StringStringEntryProto]
}

///|
pub impl @lib.Sized for GraphProto with fn size_of(self) {
  let mut size = 0U
  for s in self.node {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(s))
  }
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  for s in self.initializer {
    size += @lib.size_of_length_delimited_field(5U, @lib.size_of(s))
  }
  for s in self.sparse_initializer {
    size += @lib.size_of_length_delimited_field(15U, @lib.size_of(s))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(10U, @lib.size_of(v))
  }
  for s in self.input {
    size += @lib.size_of_length_delimited_field(11U, @lib.size_of(s))
  }
  for s in self.output {
    size += @lib.size_of_length_delimited_field(12U, @lib.size_of(s))
  }
  for s in self.value_info {
    size += @lib.size_of_length_delimited_field(13U, @lib.size_of(s))
  }
  for s in self.quantization_annotation {
    size += @lib.size_of_length_delimited_field(14U, @lib.size_of(s))
  }
  for s in self.metadata_props {
    size += @lib.size_of_length_delimited_field(16U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for GraphProto with fn default() -> GraphProto {
  GraphProto::{
    node: [],
    name: None,
    initializer: [],
    sparse_initializer: [],
    doc_string: None,
    input: [],
    output: [],
    value_info: [],
    quantization_annotation: [],
    metadata_props: [],
  }
}

///|
#alias(new, deprecated)
pub fn GraphProto::GraphProto(
  node : Array[NodeProto],
  name? : String,
  initializer : Array[TensorProto],
  sparse_initializer : Array[SparseTensorProto],
  doc_string? : String,
  input : Array[ValueInfoProto],
  output : Array[ValueInfoProto],
  value_info : Array[ValueInfoProto],
  quantization_annotation : Array[TensorAnnotation],
  metadata_props : Array[StringStringEntryProto],
) -> GraphProto {
  GraphProto::{
    node,
    name,
    initializer,
    sparse_initializer,
    doc_string,
    input,
    output,
    value_info,
    quantization_annotation,
    metadata_props,
  }
}

///|
pub impl @lib.Read for GraphProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> GraphProto raise {
  let msg : GraphProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.node.push((reader |> @lib.read_message() : NodeProto))
      (2, _) => msg.name = reader |> @lib.read_string() |> Some
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.initializer.push((reader |> @lib.read_message() : TensorProto))
      (15, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sparse_initializer.push(
          (reader |> @lib.read_message() : SparseTensorProto),
        )
      (10, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.input.push((reader |> @lib.read_message() : ValueInfoProto))
      (12, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.output.push((reader |> @lib.read_message() : ValueInfoProto))
      (13, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value_info.push((reader |> @lib.read_message() : ValueInfoProto))
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.quantization_annotation.push(
          (reader |> @lib.read_message() : TensorAnnotation),
        )
      (16, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for GraphProto with fn write(
  self : GraphProto,
  writer : &@lib.Writer,
) -> Unit raise {
  for item in self.node {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_message(item)
  }
  if self.name is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_string(v)
  }
  for item in self.initializer {
    writer |> @lib.write_varint(42UL)
    writer |> @lib.write_message(item)
  }
  for item in self.sparse_initializer {
    writer |> @lib.write_varint(122UL)
    writer |> @lib.write_message(item)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(82UL)
    writer |> @lib.write_string(v)
  }
  for item in self.input {
    writer |> @lib.write_varint(90UL)
    writer |> @lib.write_message(item)
  }
  for item in self.output {
    writer |> @lib.write_varint(98UL)
    writer |> @lib.write_message(item)
  }
  for item in self.value_info {
    writer |> @lib.write_varint(106UL)
    writer |> @lib.write_message(item)
  }
  for item in self.quantization_annotation {
    writer |> @lib.write_varint(114UL)
    writer |> @lib.write_message(item)
  }
  for item in self.metadata_props {
    writer |> @lib.write_varint(130UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for GraphProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> GraphProto raise {
  let msg : GraphProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.node.push((reader |> @lib.async_read_message() : NodeProto))
      (2, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.initializer.push(
          (reader |> @lib.async_read_message() : TensorProto),
        )
      (15, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.sparse_initializer.push(
          (reader |> @lib.async_read_message() : SparseTensorProto),
        )
      (10, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.input.push((reader |> @lib.async_read_message() : ValueInfoProto))
      (12, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.output.push((reader |> @lib.async_read_message() : ValueInfoProto))
      (13, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value_info.push(
          (reader |> @lib.async_read_message() : ValueInfoProto),
        )
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.quantization_annotation.push(
          (reader |> @lib.async_read_message() : TensorAnnotation),
        )
      (16, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for GraphProto with fn write(
  self : GraphProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  for item in self.node {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_message(item)
  }
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.initializer {
    writer |> @lib.async_write_varint(42UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.sparse_initializer {
    writer |> @lib.async_write_varint(122UL)
    writer |> @lib.async_write_message(item)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(82UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.input {
    writer |> @lib.async_write_varint(90UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.output {
    writer |> @lib.async_write_varint(98UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.value_info {
    writer |> @lib.async_write_varint(106UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.quantization_annotation {
    writer |> @lib.async_write_varint(114UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.metadata_props {
    writer |> @lib.async_write_varint(130UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend GraphProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend GraphProto with Eq::{not_equal}

///|
pub extend GraphProto with Debug::{to_repr}

///|
pub extend GraphProto with Default::{default}

///|
pub extend GraphProto with @lib.Sized::{size_of}

///|
pub extend GraphProto with @lib.Read::{read, read_with_limit}

///|
pub extend GraphProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) enum TensorProto_DataType {
  UNDEFINED
  FLOAT
  UINT8
  INT8
  UINT16
  INT16
  INT32
  INT64
  STRING
  BOOL
  FLOAT16
  DOUBLE
  UINT32
  UINT64
  COMPLEX64
  COMPLEX128
  BFLOAT16
  FLOAT8E4M3FN
  FLOAT8E4M3FNUZ
  FLOAT8E5M2
  FLOAT8E5M2FNUZ
  UINT4
  INT4
  FLOAT4E2M1
  FLOAT8E8M0
  UINT2
  INT2
  FLOAT6E2M3
  FLOAT6E3M2
}

///|
pub fn TensorProto_DataType::to_enum(self : TensorProto_DataType) -> @lib.Enum {
  match self {
    TensorProto_DataType::UNDEFINED => 0
    TensorProto_DataType::FLOAT => 1
    TensorProto_DataType::UINT8 => 2
    TensorProto_DataType::INT8 => 3
    TensorProto_DataType::UINT16 => 4
    TensorProto_DataType::INT16 => 5
    TensorProto_DataType::INT32 => 6
    TensorProto_DataType::INT64 => 7
    TensorProto_DataType::STRING => 8
    TensorProto_DataType::BOOL => 9
    TensorProto_DataType::FLOAT16 => 10
    TensorProto_DataType::DOUBLE => 11
    TensorProto_DataType::UINT32 => 12
    TensorProto_DataType::UINT64 => 13
    TensorProto_DataType::COMPLEX64 => 14
    TensorProto_DataType::COMPLEX128 => 15
    TensorProto_DataType::BFLOAT16 => 16
    TensorProto_DataType::FLOAT8E4M3FN => 17
    TensorProto_DataType::FLOAT8E4M3FNUZ => 18
    TensorProto_DataType::FLOAT8E5M2 => 19
    TensorProto_DataType::FLOAT8E5M2FNUZ => 20
    TensorProto_DataType::UINT4 => 21
    TensorProto_DataType::INT4 => 22
    TensorProto_DataType::FLOAT4E2M1 => 23
    TensorProto_DataType::FLOAT8E8M0 => 24
    TensorProto_DataType::UINT2 => 25
    TensorProto_DataType::INT2 => 26
    TensorProto_DataType::FLOAT6E2M3 => 27
    TensorProto_DataType::FLOAT6E3M2 => 28
  }
}

///|
pub fn TensorProto_DataType::from_enum(i : @lib.Enum) -> TensorProto_DataType {
  match i.0 {
    0 => TensorProto_DataType::UNDEFINED
    1 => TensorProto_DataType::FLOAT
    2 => TensorProto_DataType::UINT8
    3 => TensorProto_DataType::INT8
    4 => TensorProto_DataType::UINT16
    5 => TensorProto_DataType::INT16
    6 => TensorProto_DataType::INT32
    7 => TensorProto_DataType::INT64
    8 => TensorProto_DataType::STRING
    9 => TensorProto_DataType::BOOL
    10 => TensorProto_DataType::FLOAT16
    11 => TensorProto_DataType::DOUBLE
    12 => TensorProto_DataType::UINT32
    13 => TensorProto_DataType::UINT64
    14 => TensorProto_DataType::COMPLEX64
    15 => TensorProto_DataType::COMPLEX128
    16 => TensorProto_DataType::BFLOAT16
    17 => TensorProto_DataType::FLOAT8E4M3FN
    18 => TensorProto_DataType::FLOAT8E4M3FNUZ
    19 => TensorProto_DataType::FLOAT8E5M2
    20 => TensorProto_DataType::FLOAT8E5M2FNUZ
    21 => TensorProto_DataType::UINT4
    22 => TensorProto_DataType::INT4
    23 => TensorProto_DataType::FLOAT4E2M1
    24 => TensorProto_DataType::FLOAT8E8M0
    25 => TensorProto_DataType::UINT2
    26 => TensorProto_DataType::INT2
    27 => TensorProto_DataType::FLOAT6E2M3
    28 => TensorProto_DataType::FLOAT6E3M2
    _ => Default::default()
  }
}

///|
pub impl Default for TensorProto_DataType with fn default() -> TensorProto_DataType {
  TensorProto_DataType::UNDEFINED
}

///|
pub impl @lib.Sized for TensorProto_DataType with fn size_of(
  self : TensorProto_DataType,
) {
  @lib.Sized::size_of(self.to_enum())
}

///|
#deprecated("Use `==` instead")
pub extend TensorProto_DataType with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorProto_DataType with Eq::{not_equal}

///|
pub extend TensorProto_DataType with Debug::{to_repr}

///|
pub extend TensorProto_DataType with Default::{default}

///|
pub extend TensorProto_DataType with @lib.Sized::{size_of}

///|
#derive(Eq, Debug)
pub(all) enum TensorProto_DataLocation {
  DEFAULT
  EXTERNAL
}

///|
pub fn TensorProto_DataLocation::to_enum(
  self : TensorProto_DataLocation,
) -> @lib.Enum {
  match self {
    TensorProto_DataLocation::DEFAULT => 0
    TensorProto_DataLocation::EXTERNAL => 1
  }
}

///|
pub fn TensorProto_DataLocation::from_enum(
  i : @lib.Enum,
) -> TensorProto_DataLocation {
  match i.0 {
    0 => TensorProto_DataLocation::DEFAULT
    1 => TensorProto_DataLocation::EXTERNAL
    _ => Default::default()
  }
}

///|
pub impl Default for TensorProto_DataLocation with fn default() -> TensorProto_DataLocation {
  TensorProto_DataLocation::DEFAULT
}

///|
pub impl @lib.Sized for TensorProto_DataLocation with fn size_of(
  self : TensorProto_DataLocation,
) {
  @lib.Sized::size_of(self.to_enum())
}

///|
#deprecated("Use `==` instead")
pub extend TensorProto_DataLocation with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorProto_DataLocation with Eq::{not_equal}

///|
pub extend TensorProto_DataLocation with Debug::{to_repr}

///|
pub extend TensorProto_DataLocation with Default::{default}

///|
pub extend TensorProto_DataLocation with @lib.Sized::{size_of}

///|
#derive(Eq, Debug)
pub(all) struct TensorProto_Segment {
  mut begin : Int64?
  mut end : Int64?
}

///|
pub impl @lib.Sized for TensorProto_Segment with fn size_of(self) {
  let mut size = 0U
  if self.begin is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  if self.end is Some(v) {
    size += @lib.size_of_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TensorProto_Segment with fn default() -> TensorProto_Segment {
  TensorProto_Segment::{ begin: None, end: None, }
}

///|
#alias(new, deprecated)
pub fn TensorProto_Segment::TensorProto_Segment(
  begin? : Int64,
  end? : Int64,
) -> TensorProto_Segment {
  TensorProto_Segment::{ begin, end, }
}

///|
pub impl @lib.Read for TensorProto_Segment with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TensorProto_Segment raise {
  let msg : TensorProto_Segment = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.begin = reader |> @lib.read_int64() |> Some
      (2, _) => msg.end = reader |> @lib.read_int64() |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TensorProto_Segment with fn write(
  self : TensorProto_Segment,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.begin is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int64(v)
  }
  if self.end is Some(v) {
    writer |> @lib.write_varint(16UL)
    writer |> @lib.write_int64(v)
  }
}

///|
pub impl @lib.AsyncRead for TensorProto_Segment with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TensorProto_Segment raise {
  let msg : TensorProto_Segment = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.begin = reader |> @lib.async_read_int64() |> Some
      (2, _) => msg.end = reader |> @lib.async_read_int64() |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TensorProto_Segment with fn write(
  self : TensorProto_Segment,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.begin is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int64(v)
  }
  if self.end is Some(v) {
    writer |> @lib.async_write_varint(16UL)
    writer |> @lib.async_write_int64(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TensorProto_Segment with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorProto_Segment with Eq::{not_equal}

///|
pub extend TensorProto_Segment with Debug::{to_repr}

///|
pub extend TensorProto_Segment with Default::{default}

///|
pub extend TensorProto_Segment with @lib.Sized::{size_of}

///|
pub extend TensorProto_Segment with @lib.Read::{read, read_with_limit}

///|
pub extend TensorProto_Segment with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TensorProto {
  mut dims : Array[Int64]
  mut data_type : Int32?
  mut segment : TensorProto_Segment?
  mut float_data : Array[Float]
  mut int32_data : Array[Int32]
  mut string_data : Array[Bytes]
  mut int64_data : Array[Int64]
  mut name : String?
  mut doc_string : String?
  mut raw_data : Bytes?
  mut external_data : Array[StringStringEntryProto]
  mut data_location : TensorProto_DataLocation?
  mut double_data : Array[Double]
  mut uint64_data : Array[UInt64]
  mut metadata_props : Array[StringStringEntryProto]
}

///|
pub impl @lib.Sized for TensorProto with fn size_of(self) {
  let mut size = 0U
  for s in self.dims {
    size += @lib.size_of_field(1U, @lib.size_of(s))
  }
  if self.data_type is Some(v) {
    size += @lib.size_of_field(2U, @lib.size_of(v))
  }
  if self.segment is Some(v) {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(v))
  }
  if !self.float_data.is_empty() {
    size += @lib.size_of_length_delimited_field(
      4U,
      self.float_data.length().reinterpret_as_uint() * 4,
    )
  }
  if !self.int32_data.is_empty() {
    size += @lib.size_of_length_delimited_field(
      5U,
      self.int32_data.iter().map(@lib.size_of).fold(init=0U, UInt32::add),
    )
  }
  for s in self.string_data {
    size += @lib.size_of_length_delimited_field(6U, @lib.size_of(s))
  }
  if !self.int64_data.is_empty() {
    size += @lib.size_of_length_delimited_field(
      7U,
      self.int64_data.iter().map(@lib.size_of).fold(init=0U, UInt32::add),
    )
  }
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(8U, @lib.size_of(v))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(12U, @lib.size_of(v))
  }
  if self.raw_data is Some(v) {
    size += @lib.size_of_length_delimited_field(9U, @lib.size_of(v))
  }
  for s in self.external_data {
    size += @lib.size_of_length_delimited_field(13U, @lib.size_of(s))
  }
  if self.data_location is Some(v) {
    size += @lib.size_of_field(14U, @lib.size_of(v))
  }
  if !self.double_data.is_empty() {
    size += @lib.size_of_length_delimited_field(
      10U,
      self.double_data.length().reinterpret_as_uint() * 8,
    )
  }
  if !self.uint64_data.is_empty() {
    size += @lib.size_of_length_delimited_field(
      11U,
      self.uint64_data.iter().map(@lib.size_of).fold(init=0U, UInt32::add),
    )
  }
  for s in self.metadata_props {
    size += @lib.size_of_length_delimited_field(16U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for TensorProto with fn default() -> TensorProto {
  TensorProto::{
    dims: [],
    data_type: None,
    segment: None,
    float_data: [],
    int32_data: [],
    string_data: [],
    int64_data: [],
    name: None,
    doc_string: None,
    raw_data: None,
    external_data: [],
    data_location: None,
    double_data: [],
    uint64_data: [],
    metadata_props: [],
  }
}

///|
#alias(new, deprecated)
pub fn TensorProto::TensorProto(
  dims : Array[Int64],
  data_type? : Int32,
  segment? : TensorProto_Segment,
  float_data : Array[Float],
  int32_data : Array[Int32],
  string_data : Array[Bytes],
  int64_data : Array[Int64],
  name? : String,
  doc_string? : String,
  raw_data? : Bytes,
  external_data : Array[StringStringEntryProto],
  data_location? : TensorProto_DataLocation,
  double_data : Array[Double],
  uint64_data : Array[UInt64],
  metadata_props : Array[StringStringEntryProto],
) -> TensorProto {
  TensorProto::{
    dims,
    data_type,
    segment,
    float_data,
    int32_data,
    string_data,
    int64_data,
    name,
    doc_string,
    raw_data,
    external_data,
    data_location,
    double_data,
    uint64_data,
    metadata_props,
  }
}

///|
pub impl @lib.Read for TensorProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TensorProto raise {
  let msg : TensorProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.dims.push_iter(
          (reader |> @lib.read_packed(@lib.read_int64, None)).iter(),
        )
      (1, @lib.WIRE_TYPE_VARINT) => msg.dims.push(reader |> @lib.read_int64())
      (2, _) => msg.data_type = reader |> @lib.read_int32() |> Some
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.segment = (reader |> @lib.read_message() : TensorProto_Segment)
          |> Some
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.float_data.push_iter(
          (reader |> @lib.read_packed(@lib.read_float, Some(4))).iter(),
        )
      (4, @lib.WIRE_TYPE_FIXED32) =>
        msg.float_data.push(reader |> @lib.read_float())
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.int32_data.push_iter(
          (reader |> @lib.read_packed(@lib.read_int32, None)).iter(),
        )
      (5, @lib.WIRE_TYPE_VARINT) =>
        msg.int32_data.push(reader |> @lib.read_int32())
      (6, _) => msg.string_data.push(reader |> @lib.read_bytes())
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.int64_data.push_iter(
          (reader |> @lib.read_packed(@lib.read_int64, None)).iter(),
        )
      (7, @lib.WIRE_TYPE_VARINT) =>
        msg.int64_data.push(reader |> @lib.read_int64())
      (8, _) => msg.name = reader |> @lib.read_string() |> Some
      (12, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (9, _) => msg.raw_data = reader |> @lib.read_bytes() |> Some
      (13, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.external_data.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      (14, _) =>
        msg.data_location = reader
          |> @lib.read_enum()
          |> TensorProto_DataLocation::from_enum
          |> Some
      (10, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.double_data.push_iter(
          (reader |> @lib.read_packed(@lib.read_double, Some(8))).iter(),
        )
      (10, @lib.WIRE_TYPE_FIXED64) =>
        msg.double_data.push(reader |> @lib.read_double())
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.uint64_data.push_iter(
          (reader |> @lib.read_packed(@lib.read_uint64, None)).iter(),
        )
      (11, @lib.WIRE_TYPE_VARINT) =>
        msg.uint64_data.push(reader |> @lib.read_uint64())
      (16, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TensorProto with fn write(
  self : TensorProto,
  writer : &@lib.Writer,
) -> Unit raise {
  for item in self.dims {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int64(item)
  }
  if self.data_type is Some(v) {
    writer |> @lib.write_varint(16UL)
    writer |> @lib.write_int32(v)
  }
  if self.segment is Some(v) {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_message(v)
  }
  if !self.float_data.is_empty() {
    writer |> @lib.write_varint(34UL)
    let size = self.float_data.length().reinterpret_as_uint() * 4
    writer |> @lib.write_uint32(size)
    for item in self.float_data {
      writer |> @lib.write_float(item)
    }
  }
  if !self.int32_data.is_empty() {
    writer |> @lib.write_varint(42UL)
    let size = self.int32_data
      .iter()
      .map(@lib.size_of)
      .fold(init=0U, UInt32::add)
    writer |> @lib.write_uint32(size)
    for item in self.int32_data {
      writer |> @lib.write_int32(item)
    }
  }
  for item in self.string_data {
    writer |> @lib.write_varint(50UL)
    writer |> @lib.write_bytes(item)
  }
  if !self.int64_data.is_empty() {
    writer |> @lib.write_varint(58UL)
    let size = self.int64_data
      .iter()
      .map(@lib.size_of)
      .fold(init=0U, UInt32::add)
    writer |> @lib.write_uint32(size)
    for item in self.int64_data {
      writer |> @lib.write_int64(item)
    }
  }
  if self.name is Some(v) {
    writer |> @lib.write_varint(66UL)
    writer |> @lib.write_string(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(98UL)
    writer |> @lib.write_string(v)
  }
  if self.raw_data is Some(v) {
    writer |> @lib.write_varint(74UL)
    writer |> @lib.write_bytes(v)
  }
  for item in self.external_data {
    writer |> @lib.write_varint(106UL)
    writer |> @lib.write_message(item)
  }
  if self.data_location is Some(v) {
    writer |> @lib.write_varint(112UL)
    writer |> @lib.write_enum(v.to_enum())
  }
  if !self.double_data.is_empty() {
    writer |> @lib.write_varint(82UL)
    let size = self.double_data.length().reinterpret_as_uint() * 8
    writer |> @lib.write_uint32(size)
    for item in self.double_data {
      writer |> @lib.write_double(item)
    }
  }
  if !self.uint64_data.is_empty() {
    writer |> @lib.write_varint(90UL)
    let size = self.uint64_data
      .iter()
      .map(@lib.size_of)
      .fold(init=0U, UInt32::add)
    writer |> @lib.write_uint32(size)
    for item in self.uint64_data {
      writer |> @lib.write_uint64(item)
    }
  }
  for item in self.metadata_props {
    writer |> @lib.write_varint(130UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for TensorProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TensorProto raise {
  let msg : TensorProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.dims.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int64, None)).iter(),
        )
      (1, @lib.WIRE_TYPE_VARINT) =>
        msg.dims.push(reader |> @lib.async_read_int64())
      (2, _) => msg.data_type = reader |> @lib.async_read_int32() |> Some
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.segment = (reader |> @lib.async_read_message() : TensorProto_Segment)
          |> Some
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.float_data.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_float, Some(4))).iter(),
        )
      (4, @lib.WIRE_TYPE_FIXED32) =>
        msg.float_data.push(reader |> @lib.async_read_float())
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.int32_data.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int32, None)).iter(),
        )
      (5, @lib.WIRE_TYPE_VARINT) =>
        msg.int32_data.push(reader |> @lib.async_read_int32())
      (6, _) => msg.string_data.push(reader |> @lib.async_read_bytes())
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.int64_data.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int64, None)).iter(),
        )
      (7, @lib.WIRE_TYPE_VARINT) =>
        msg.int64_data.push(reader |> @lib.async_read_int64())
      (8, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (12, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (9, _) => msg.raw_data = reader |> @lib.async_read_bytes() |> Some
      (13, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.external_data.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      (14, _) =>
        msg.data_location = reader
          |> @lib.async_read_enum()
          |> TensorProto_DataLocation::from_enum
          |> Some
      (10, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.double_data.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_double, Some(8))).iter(),
        )
      (10, @lib.WIRE_TYPE_FIXED64) =>
        msg.double_data.push(reader |> @lib.async_read_double())
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.uint64_data.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_uint64, None)).iter(),
        )
      (11, @lib.WIRE_TYPE_VARINT) =>
        msg.uint64_data.push(reader |> @lib.async_read_uint64())
      (16, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TensorProto with fn write(
  self : TensorProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  for item in self.dims {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int64(item)
  }
  if self.data_type is Some(v) {
    writer |> @lib.async_write_varint(16UL)
    writer |> @lib.async_write_int32(v)
  }
  if self.segment is Some(v) {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_message(v)
  }
  if !self.float_data.is_empty() {
    writer |> @lib.async_write_varint(34UL)
    let size = self.float_data.length().reinterpret_as_uint() * 4
    writer |> @lib.async_write_uint32(size)
    for item in self.float_data {
      writer |> @lib.async_write_float(item)
    }
  }
  if !self.int32_data.is_empty() {
    writer |> @lib.async_write_varint(42UL)
    let size = self.int32_data
      .iter()
      .map(@lib.size_of)
      .fold(init=0U, UInt32::add)
    writer |> @lib.async_write_uint32(size)
    for item in self.int32_data {
      writer |> @lib.async_write_int32(item)
    }
  }
  for item in self.string_data {
    writer |> @lib.async_write_varint(50UL)
    writer |> @lib.async_write_bytes(item)
  }
  if !self.int64_data.is_empty() {
    writer |> @lib.async_write_varint(58UL)
    let size = self.int64_data
      .iter()
      .map(@lib.size_of)
      .fold(init=0U, UInt32::add)
    writer |> @lib.async_write_uint32(size)
    for item in self.int64_data {
      writer |> @lib.async_write_int64(item)
    }
  }
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(66UL)
    writer |> @lib.async_write_string(v)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(98UL)
    writer |> @lib.async_write_string(v)
  }
  if self.raw_data is Some(v) {
    writer |> @lib.async_write_varint(74UL)
    writer |> @lib.async_write_bytes(v)
  }
  for item in self.external_data {
    writer |> @lib.async_write_varint(106UL)
    writer |> @lib.async_write_message(item)
  }
  if self.data_location is Some(v) {
    writer |> @lib.async_write_varint(112UL)
    writer |> @lib.async_write_enum(v.to_enum())
  }
  if !self.double_data.is_empty() {
    writer |> @lib.async_write_varint(82UL)
    let size = self.double_data.length().reinterpret_as_uint() * 8
    writer |> @lib.async_write_uint32(size)
    for item in self.double_data {
      writer |> @lib.async_write_double(item)
    }
  }
  if !self.uint64_data.is_empty() {
    writer |> @lib.async_write_varint(90UL)
    let size = self.uint64_data
      .iter()
      .map(@lib.size_of)
      .fold(init=0U, UInt32::add)
    writer |> @lib.async_write_uint32(size)
    for item in self.uint64_data {
      writer |> @lib.async_write_uint64(item)
    }
  }
  for item in self.metadata_props {
    writer |> @lib.async_write_varint(130UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TensorProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorProto with Eq::{not_equal}

///|
pub extend TensorProto with Debug::{to_repr}

///|
pub extend TensorProto with Default::{default}

///|
pub extend TensorProto with @lib.Sized::{size_of}

///|
pub extend TensorProto with @lib.Read::{read, read_with_limit}

///|
pub extend TensorProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct SparseTensorProto {
  mut values : TensorProto?
  mut indices : TensorProto?
  mut dims : Array[Int64]
}

///|
pub impl @lib.Sized for SparseTensorProto with fn size_of(self) {
  let mut size = 0U
  if self.values is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.indices is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  for s in self.dims {
    size += @lib.size_of_field(3U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for SparseTensorProto with fn default() -> SparseTensorProto {
  SparseTensorProto::{ values: None, indices: None, dims: [], }
}

///|
#alias(new, deprecated)
pub fn SparseTensorProto::SparseTensorProto(
  values? : TensorProto,
  indices? : TensorProto,
  dims : Array[Int64],
) -> SparseTensorProto {
  SparseTensorProto::{ values, indices, dims, }
}

///|
pub impl @lib.Read for SparseTensorProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> SparseTensorProto raise {
  let msg : SparseTensorProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.values = (reader |> @lib.read_message() : TensorProto) |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.indices = (reader |> @lib.read_message() : TensorProto) |> Some
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.dims.push_iter(
          (reader |> @lib.read_packed(@lib.read_int64, None)).iter(),
        )
      (3, @lib.WIRE_TYPE_VARINT) => msg.dims.push(reader |> @lib.read_int64())
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for SparseTensorProto with fn write(
  self : SparseTensorProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.values is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_message(v)
  }
  if self.indices is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(v)
  }
  for item in self.dims {
    writer |> @lib.write_varint(24UL)
    writer |> @lib.write_int64(item)
  }
}

///|
pub impl @lib.AsyncRead for SparseTensorProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> SparseTensorProto raise {
  let msg : SparseTensorProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.values = (reader |> @lib.async_read_message() : TensorProto) |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.indices = (reader |> @lib.async_read_message() : TensorProto)
          |> Some
      (3, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.dims.push_iter(
          (reader |> @lib.async_read_packed(@lib.async_read_int64, None)).iter(),
        )
      (3, @lib.WIRE_TYPE_VARINT) =>
        msg.dims.push(reader |> @lib.async_read_int64())
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for SparseTensorProto with fn write(
  self : SparseTensorProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.values is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_message(v)
  }
  if self.indices is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(v)
  }
  for item in self.dims {
    writer |> @lib.async_write_varint(24UL)
    writer |> @lib.async_write_int64(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend SparseTensorProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend SparseTensorProto with Eq::{not_equal}

///|
pub extend SparseTensorProto with Debug::{to_repr}

///|
pub extend SparseTensorProto with Default::{default}

///|
pub extend SparseTensorProto with @lib.Sized::{size_of}

///|
pub extend SparseTensorProto with @lib.Read::{read, read_with_limit}

///|
pub extend SparseTensorProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TensorShapeProto_Dimension {
  mut denotation : String?
  mut value : TensorShapeProto_Dimension_Value
}

///|
#derive(Eq, Debug)
pub(all) enum TensorShapeProto_Dimension_Value {
  DimValue(Int64)
  DimParam(String)
  NotSet
}

///|
pub impl Default for TensorShapeProto_Dimension_Value with fn default() -> TensorShapeProto_Dimension_Value {
  NotSet
}

///|
#deprecated("Use `==` instead")
pub extend TensorShapeProto_Dimension_Value with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorShapeProto_Dimension_Value with Eq::{not_equal}

///|
pub extend TensorShapeProto_Dimension_Value with Debug::{to_repr}

///|
pub extend TensorShapeProto_Dimension_Value with Default::{default}

///|
pub impl @lib.Sized for TensorShapeProto_Dimension with fn size_of(self) {
  let mut size = 0U
  if self.denotation is Some(v) {
    size += @lib.size_of_length_delimited_field(3U, @lib.size_of(v))
  }
  match self.value {
    DimValue(v) => size += @lib.size_of_field(1U, @lib.size_of(v))
    DimParam(v) =>
      size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
    NotSet => ()
  }
  size
}

///|
pub impl Default for TensorShapeProto_Dimension with fn default() -> TensorShapeProto_Dimension {
  TensorShapeProto_Dimension::{
    denotation: None,
    value: TensorShapeProto_Dimension_Value::NotSet,
  }
}

///|
#alias(new, deprecated)
pub fn TensorShapeProto_Dimension::TensorShapeProto_Dimension(
  denotation? : String,
  value~ : TensorShapeProto_Dimension_Value = TensorShapeProto_Dimension_Value::NotSet,
) -> TensorShapeProto_Dimension {
  TensorShapeProto_Dimension::{ denotation, value, }
}

///|
pub impl @lib.Read for TensorShapeProto_Dimension with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TensorShapeProto_Dimension raise {
  let msg : TensorShapeProto_Dimension = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (3, _) => msg.denotation = reader |> @lib.read_string() |> Some
      (1, _) =>
        msg.value = reader
          |> @lib.read_int64()
          |> TensorShapeProto_Dimension_Value::DimValue
      (2, _) =>
        msg.value = reader
          |> @lib.read_string()
          |> TensorShapeProto_Dimension_Value::DimParam
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TensorShapeProto_Dimension with fn write(
  self : TensorShapeProto_Dimension,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.denotation is Some(v) {
    writer |> @lib.write_varint(26UL)
    writer |> @lib.write_string(v)
  }
  match self.value {
    TensorShapeProto_Dimension_Value::DimValue(v) => {
      writer |> @lib.write_varint(8UL)
      writer |> @lib.write_int64(v)
    }
    TensorShapeProto_Dimension_Value::DimParam(v) => {
      writer |> @lib.write_varint(18UL)
      writer |> @lib.write_string(v)
    }
    TensorShapeProto_Dimension_Value::NotSet => ()
  }
}

///|
pub impl @lib.AsyncRead for TensorShapeProto_Dimension with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TensorShapeProto_Dimension raise {
  let msg : TensorShapeProto_Dimension = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (3, _) => msg.denotation = reader |> @lib.async_read_string() |> Some
      (1, _) =>
        msg.value = reader
          |> @lib.async_read_int64()
          |> TensorShapeProto_Dimension_Value::DimValue
      (2, _) =>
        msg.value = reader
          |> @lib.async_read_string()
          |> TensorShapeProto_Dimension_Value::DimParam
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TensorShapeProto_Dimension with fn write(
  self : TensorShapeProto_Dimension,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.denotation is Some(v) {
    writer |> @lib.async_write_varint(26UL)
    writer |> @lib.async_write_string(v)
  }
  match self.value {
    TensorShapeProto_Dimension_Value::DimValue(v) => {
      writer |> @lib.async_write_varint(8UL)
      writer |> @lib.async_write_int64(v)
    }
    TensorShapeProto_Dimension_Value::DimParam(v) => {
      writer |> @lib.async_write_varint(18UL)
      writer |> @lib.async_write_string(v)
    }
    TensorShapeProto_Dimension_Value::NotSet => ()
  }
}

///|
#deprecated("Use `==` instead")
pub extend TensorShapeProto_Dimension with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorShapeProto_Dimension with Eq::{not_equal}

///|
pub extend TensorShapeProto_Dimension with Debug::{to_repr}

///|
pub extend TensorShapeProto_Dimension with Default::{default}

///|
pub extend TensorShapeProto_Dimension with @lib.Sized::{size_of}

///|
pub extend TensorShapeProto_Dimension with @lib.Read::{read, read_with_limit}

///|
pub extend TensorShapeProto_Dimension with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TensorShapeProto {
  mut dim : Array[TensorShapeProto_Dimension]
}

///|
pub impl @lib.Sized for TensorShapeProto with fn size_of(self) {
  let mut size = 0U
  for s in self.dim {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for TensorShapeProto with fn default() -> TensorShapeProto {
  TensorShapeProto::{ dim: [], }
}

///|
#alias(new, deprecated)
pub fn TensorShapeProto::TensorShapeProto(
  dim : Array[TensorShapeProto_Dimension],
) -> TensorShapeProto {
  TensorShapeProto::{ dim, }
}

///|
pub impl @lib.Read for TensorShapeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TensorShapeProto raise {
  let msg : TensorShapeProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.dim.push(
          (reader |> @lib.read_message() : TensorShapeProto_Dimension),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TensorShapeProto with fn write(
  self : TensorShapeProto,
  writer : &@lib.Writer,
) -> Unit raise {
  for item in self.dim {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for TensorShapeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TensorShapeProto raise {
  let msg : TensorShapeProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.dim.push(
          (reader |> @lib.async_read_message() : TensorShapeProto_Dimension),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TensorShapeProto with fn write(
  self : TensorShapeProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  for item in self.dim {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TensorShapeProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TensorShapeProto with Eq::{not_equal}

///|
pub extend TensorShapeProto with Debug::{to_repr}

///|
pub extend TensorShapeProto with Default::{default}

///|
pub extend TensorShapeProto with @lib.Sized::{size_of}

///|
pub extend TensorShapeProto with @lib.Read::{read, read_with_limit}

///|
pub extend TensorShapeProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto_Tensor {
  mut elem_type : Int32?
  mut shape : TensorShapeProto?
}

///|
pub impl @lib.Sized for TypeProto_Tensor with fn size_of(self) {
  let mut size = 0U
  if self.elem_type is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  if self.shape is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TypeProto_Tensor with fn default() -> TypeProto_Tensor {
  TypeProto_Tensor::{ elem_type: None, shape: None, }
}

///|
#alias(new, deprecated)
pub fn TypeProto_Tensor::TypeProto_Tensor(
  elem_type? : Int32,
  shape? : TensorShapeProto,
) -> TypeProto_Tensor {
  TypeProto_Tensor::{ elem_type, shape, }
}

///|
pub impl @lib.Read for TypeProto_Tensor with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto_Tensor raise {
  let msg : TypeProto_Tensor = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.elem_type = reader |> @lib.read_int32() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.shape = (reader |> @lib.read_message() : TensorShapeProto) |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto_Tensor with fn write(
  self : TypeProto_Tensor,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int32(v)
  }
  if self.shape is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(v)
  }
}

///|
pub impl @lib.AsyncRead for TypeProto_Tensor with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto_Tensor raise {
  let msg : TypeProto_Tensor = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.elem_type = reader |> @lib.async_read_int32() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.shape = (reader |> @lib.async_read_message() : TensorShapeProto)
          |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto_Tensor with fn write(
  self : TypeProto_Tensor,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int32(v)
  }
  if self.shape is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_Tensor with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_Tensor with Eq::{not_equal}

///|
pub extend TypeProto_Tensor with Debug::{to_repr}

///|
pub extend TypeProto_Tensor with Default::{default}

///|
pub extend TypeProto_Tensor with @lib.Sized::{size_of}

///|
pub extend TypeProto_Tensor with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto_Tensor with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto_Sequence {
  mut elem_type : TypeProto?
}

///|
pub impl @lib.Sized for TypeProto_Sequence with fn size_of(self) {
  let mut size = 0U
  if self.elem_type is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TypeProto_Sequence with fn default() -> TypeProto_Sequence {
  TypeProto_Sequence::{ elem_type: None, }
}

///|
#alias(new, deprecated)
pub fn TypeProto_Sequence::TypeProto_Sequence(
  elem_type? : TypeProto,
) -> TypeProto_Sequence {
  TypeProto_Sequence::{ elem_type, }
}

///|
pub impl @lib.Read for TypeProto_Sequence with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto_Sequence raise {
  let msg : TypeProto_Sequence = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.elem_type = (reader |> @lib.read_message() : TypeProto) |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto_Sequence with fn write(
  self : TypeProto_Sequence,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_message(v)
  }
}

///|
pub impl @lib.AsyncRead for TypeProto_Sequence with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto_Sequence raise {
  let msg : TypeProto_Sequence = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.elem_type = (reader |> @lib.async_read_message() : TypeProto)
          |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto_Sequence with fn write(
  self : TypeProto_Sequence,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_message(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_Sequence with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_Sequence with Eq::{not_equal}

///|
pub extend TypeProto_Sequence with Debug::{to_repr}

///|
pub extend TypeProto_Sequence with Default::{default}

///|
pub extend TypeProto_Sequence with @lib.Sized::{size_of}

///|
pub extend TypeProto_Sequence with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto_Sequence with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto_Map {
  mut key_type : Int32?
  mut value_type : TypeProto?
}

///|
pub impl @lib.Sized for TypeProto_Map with fn size_of(self) {
  let mut size = 0U
  if self.key_type is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  if self.value_type is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TypeProto_Map with fn default() -> TypeProto_Map {
  TypeProto_Map::{ key_type: None, value_type: None, }
}

///|
#alias(new, deprecated)
pub fn TypeProto_Map::TypeProto_Map(
  key_type? : Int32,
  value_type? : TypeProto,
) -> TypeProto_Map {
  TypeProto_Map::{ key_type, value_type, }
}

///|
pub impl @lib.Read for TypeProto_Map with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto_Map raise {
  let msg : TypeProto_Map = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.key_type = reader |> @lib.read_int32() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value_type = (reader |> @lib.read_message() : TypeProto) |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto_Map with fn write(
  self : TypeProto_Map,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.key_type is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int32(v)
  }
  if self.value_type is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(v)
  }
}

///|
pub impl @lib.AsyncRead for TypeProto_Map with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto_Map raise {
  let msg : TypeProto_Map = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.key_type = reader |> @lib.async_read_int32() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value_type = (reader |> @lib.async_read_message() : TypeProto)
          |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto_Map with fn write(
  self : TypeProto_Map,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.key_type is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int32(v)
  }
  if self.value_type is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_Map with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_Map with Eq::{not_equal}

///|
pub extend TypeProto_Map with Debug::{to_repr}

///|
pub extend TypeProto_Map with Default::{default}

///|
pub extend TypeProto_Map with @lib.Sized::{size_of}

///|
pub extend TypeProto_Map with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto_Map with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto_Optional {
  mut elem_type : TypeProto?
}

///|
pub impl @lib.Sized for TypeProto_Optional with fn size_of(self) {
  let mut size = 0U
  if self.elem_type is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TypeProto_Optional with fn default() -> TypeProto_Optional {
  TypeProto_Optional::{ elem_type: None, }
}

///|
#alias(new, deprecated)
pub fn TypeProto_Optional::TypeProto_Optional(
  elem_type? : TypeProto,
) -> TypeProto_Optional {
  TypeProto_Optional::{ elem_type, }
}

///|
pub impl @lib.Read for TypeProto_Optional with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto_Optional raise {
  let msg : TypeProto_Optional = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.elem_type = (reader |> @lib.read_message() : TypeProto) |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto_Optional with fn write(
  self : TypeProto_Optional,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_message(v)
  }
}

///|
pub impl @lib.AsyncRead for TypeProto_Optional with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto_Optional raise {
  let msg : TypeProto_Optional = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.elem_type = (reader |> @lib.async_read_message() : TypeProto)
          |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto_Optional with fn write(
  self : TypeProto_Optional,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_message(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_Optional with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_Optional with Eq::{not_equal}

///|
pub extend TypeProto_Optional with Debug::{to_repr}

///|
pub extend TypeProto_Optional with Default::{default}

///|
pub extend TypeProto_Optional with @lib.Sized::{size_of}

///|
pub extend TypeProto_Optional with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto_Optional with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto_SparseTensor {
  mut elem_type : Int32?
  mut shape : TensorShapeProto?
}

///|
pub impl @lib.Sized for TypeProto_SparseTensor with fn size_of(self) {
  let mut size = 0U
  if self.elem_type is Some(v) {
    size += @lib.size_of_field(1U, @lib.size_of(v))
  }
  if self.shape is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TypeProto_SparseTensor with fn default() -> TypeProto_SparseTensor {
  TypeProto_SparseTensor::{ elem_type: None, shape: None, }
}

///|
#alias(new, deprecated)
pub fn TypeProto_SparseTensor::TypeProto_SparseTensor(
  elem_type? : Int32,
  shape? : TensorShapeProto,
) -> TypeProto_SparseTensor {
  TypeProto_SparseTensor::{ elem_type, shape, }
}

///|
pub impl @lib.Read for TypeProto_SparseTensor with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto_SparseTensor raise {
  let msg : TypeProto_SparseTensor = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.elem_type = reader |> @lib.read_int32() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.shape = (reader |> @lib.read_message() : TensorShapeProto) |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto_SparseTensor with fn write(
  self : TypeProto_SparseTensor,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.write_varint(8UL)
    writer |> @lib.write_int32(v)
  }
  if self.shape is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_message(v)
  }
}

///|
pub impl @lib.AsyncRead for TypeProto_SparseTensor with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto_SparseTensor raise {
  let msg : TypeProto_SparseTensor = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.elem_type = reader |> @lib.async_read_int32() |> Some
      (2, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.shape = (reader |> @lib.async_read_message() : TensorShapeProto)
          |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto_SparseTensor with fn write(
  self : TypeProto_SparseTensor,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.elem_type is Some(v) {
    writer |> @lib.async_write_varint(8UL)
    writer |> @lib.async_write_int32(v)
  }
  if self.shape is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_message(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_SparseTensor with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_SparseTensor with Eq::{not_equal}

///|
pub extend TypeProto_SparseTensor with Debug::{to_repr}

///|
pub extend TypeProto_SparseTensor with Default::{default}

///|
pub extend TypeProto_SparseTensor with @lib.Sized::{size_of}

///|
pub extend TypeProto_SparseTensor with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto_SparseTensor with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto_Opaque {
  mut domain : String?
  mut name : String?
}

///|
pub impl @lib.Sized for TypeProto_Opaque with fn size_of(self) {
  let mut size = 0U
  if self.domain is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for TypeProto_Opaque with fn default() -> TypeProto_Opaque {
  TypeProto_Opaque::{ domain: None, name: None, }
}

///|
#alias(new, deprecated)
pub fn TypeProto_Opaque::TypeProto_Opaque(
  domain? : String,
  name? : String,
) -> TypeProto_Opaque {
  TypeProto_Opaque::{ domain, name, }
}

///|
pub impl @lib.Read for TypeProto_Opaque with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto_Opaque raise {
  let msg : TypeProto_Opaque = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.domain = reader |> @lib.read_string() |> Some
      (2, _) => msg.name = reader |> @lib.read_string() |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto_Opaque with fn write(
  self : TypeProto_Opaque,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.domain is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  if self.name is Some(v) {
    writer |> @lib.write_varint(18UL)
    writer |> @lib.write_string(v)
  }
}

///|
pub impl @lib.AsyncRead for TypeProto_Opaque with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto_Opaque raise {
  let msg : TypeProto_Opaque = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.domain = reader |> @lib.async_read_string() |> Some
      (2, _) => msg.name = reader |> @lib.async_read_string() |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto_Opaque with fn write(
  self : TypeProto_Opaque,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.domain is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(18UL)
    writer |> @lib.async_write_string(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_Opaque with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_Opaque with Eq::{not_equal}

///|
pub extend TypeProto_Opaque with Debug::{to_repr}

///|
pub extend TypeProto_Opaque with Default::{default}

///|
pub extend TypeProto_Opaque with @lib.Sized::{size_of}

///|
pub extend TypeProto_Opaque with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto_Opaque with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct TypeProto {
  mut denotation : String?
  mut value : TypeProto_Value
}

///|
#derive(Eq, Debug)
pub(all) enum TypeProto_Value {
  TensorType(TypeProto_Tensor)
  SequenceType(TypeProto_Sequence)
  MapType(TypeProto_Map)
  OptionalType(TypeProto_Optional)
  SparseTensorType(TypeProto_SparseTensor)
  OpaqueType(TypeProto_Opaque)
  NotSet
}

///|
pub impl Default for TypeProto_Value with fn default() -> TypeProto_Value {
  NotSet
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto_Value with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto_Value with Eq::{not_equal}

///|
pub extend TypeProto_Value with Debug::{to_repr}

///|
pub extend TypeProto_Value with Default::{default}

///|
pub impl @lib.Sized for TypeProto with fn size_of(self) {
  let mut size = 0U
  if self.denotation is Some(v) {
    size += @lib.size_of_length_delimited_field(6U, @lib.size_of(v))
  }
  match self.value {
    TensorType(v) =>
      size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
    SequenceType(v) =>
      size += @lib.size_of_length_delimited_field(4U, @lib.size_of(v))
    MapType(v) =>
      size += @lib.size_of_length_delimited_field(5U, @lib.size_of(v))
    OptionalType(v) =>
      size += @lib.size_of_length_delimited_field(9U, @lib.size_of(v))
    SparseTensorType(v) =>
      size += @lib.size_of_length_delimited_field(8U, @lib.size_of(v))
    OpaqueType(v) =>
      size += @lib.size_of_length_delimited_field(7U, @lib.size_of(v))
    NotSet => ()
  }
  size
}

///|
pub impl Default for TypeProto with fn default() -> TypeProto {
  TypeProto::{ denotation: None, value: TypeProto_Value::NotSet, }
}

///|
#alias(new, deprecated)
pub fn TypeProto::TypeProto(
  denotation? : String,
  value~ : TypeProto_Value = TypeProto_Value::NotSet,
) -> TypeProto {
  TypeProto::{ denotation, value, }
}

///|
pub impl @lib.Read for TypeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> TypeProto raise {
  let msg : TypeProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (6, _) => msg.denotation = reader |> @lib.read_string() |> Some
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.read_message() : TypeProto_Tensor)
          |> TypeProto_Value::TensorType
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.read_message() : TypeProto_Sequence)
          |> TypeProto_Value::SequenceType
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.read_message() : TypeProto_Map)
          |> TypeProto_Value::MapType
      (9, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.read_message() : TypeProto_Optional)
          |> TypeProto_Value::OptionalType
      (8, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.read_message() : TypeProto_SparseTensor)
          |> TypeProto_Value::SparseTensorType
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.read_message() : TypeProto_Opaque)
          |> TypeProto_Value::OpaqueType
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for TypeProto with fn write(
  self : TypeProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.denotation is Some(v) {
    writer |> @lib.write_varint(50UL)
    writer |> @lib.write_string(v)
  }
  match self.value {
    TypeProto_Value::TensorType(v) => {
      writer |> @lib.write_varint(10UL)
      writer |> @lib.write_message(v)
    }
    TypeProto_Value::SequenceType(v) => {
      writer |> @lib.write_varint(34UL)
      writer |> @lib.write_message(v)
    }
    TypeProto_Value::MapType(v) => {
      writer |> @lib.write_varint(42UL)
      writer |> @lib.write_message(v)
    }
    TypeProto_Value::OptionalType(v) => {
      writer |> @lib.write_varint(74UL)
      writer |> @lib.write_message(v)
    }
    TypeProto_Value::SparseTensorType(v) => {
      writer |> @lib.write_varint(66UL)
      writer |> @lib.write_message(v)
    }
    TypeProto_Value::OpaqueType(v) => {
      writer |> @lib.write_varint(58UL)
      writer |> @lib.write_message(v)
    }
    TypeProto_Value::NotSet => ()
  }
}

///|
pub impl @lib.AsyncRead for TypeProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> TypeProto raise {
  let msg : TypeProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (6, _) => msg.denotation = reader |> @lib.async_read_string() |> Some
      (1, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.async_read_message() : TypeProto_Tensor)
          |> TypeProto_Value::TensorType
      (4, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.async_read_message() : TypeProto_Sequence)
          |> TypeProto_Value::SequenceType
      (5, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.async_read_message() : TypeProto_Map)
          |> TypeProto_Value::MapType
      (9, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.async_read_message() : TypeProto_Optional)
          |> TypeProto_Value::OptionalType
      (8, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (
            reader |> @lib.async_read_message() : TypeProto_SparseTensor)
          |> TypeProto_Value::SparseTensorType
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value = (reader |> @lib.async_read_message() : TypeProto_Opaque)
          |> TypeProto_Value::OpaqueType
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for TypeProto with fn write(
  self : TypeProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.denotation is Some(v) {
    writer |> @lib.async_write_varint(50UL)
    writer |> @lib.async_write_string(v)
  }
  match self.value {
    TypeProto_Value::TensorType(v) => {
      writer |> @lib.async_write_varint(10UL)
      writer |> @lib.async_write_message(v)
    }
    TypeProto_Value::SequenceType(v) => {
      writer |> @lib.async_write_varint(34UL)
      writer |> @lib.async_write_message(v)
    }
    TypeProto_Value::MapType(v) => {
      writer |> @lib.async_write_varint(42UL)
      writer |> @lib.async_write_message(v)
    }
    TypeProto_Value::OptionalType(v) => {
      writer |> @lib.async_write_varint(74UL)
      writer |> @lib.async_write_message(v)
    }
    TypeProto_Value::SparseTensorType(v) => {
      writer |> @lib.async_write_varint(66UL)
      writer |> @lib.async_write_message(v)
    }
    TypeProto_Value::OpaqueType(v) => {
      writer |> @lib.async_write_varint(58UL)
      writer |> @lib.async_write_message(v)
    }
    TypeProto_Value::NotSet => ()
  }
}

///|
#deprecated("Use `==` instead")
pub extend TypeProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend TypeProto with Eq::{not_equal}

///|
pub extend TypeProto with Debug::{to_repr}

///|
pub extend TypeProto with Default::{default}

///|
pub extend TypeProto with @lib.Sized::{size_of}

///|
pub extend TypeProto with @lib.Read::{read, read_with_limit}

///|
pub extend TypeProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct OperatorSetIdProto {
  mut domain : String?
  mut version : Int64?
}

///|
pub impl @lib.Sized for OperatorSetIdProto with fn size_of(self) {
  let mut size = 0U
  if self.domain is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  if self.version is Some(v) {
    size += @lib.size_of_field(2U, @lib.size_of(v))
  }
  size
}

///|
pub impl Default for OperatorSetIdProto with fn default() -> OperatorSetIdProto {
  OperatorSetIdProto::{ domain: None, version: None, }
}

///|
#alias(new, deprecated)
pub fn OperatorSetIdProto::OperatorSetIdProto(
  domain? : String,
  version? : Int64,
) -> OperatorSetIdProto {
  OperatorSetIdProto::{ domain, version, }
}

///|
pub impl @lib.Read for OperatorSetIdProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> OperatorSetIdProto raise {
  let msg : OperatorSetIdProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.domain = reader |> @lib.read_string() |> Some
      (2, _) => msg.version = reader |> @lib.read_int64() |> Some
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for OperatorSetIdProto with fn write(
  self : OperatorSetIdProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.domain is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  if self.version is Some(v) {
    writer |> @lib.write_varint(16UL)
    writer |> @lib.write_int64(v)
  }
}

///|
pub impl @lib.AsyncRead for OperatorSetIdProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> OperatorSetIdProto raise {
  let msg : OperatorSetIdProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.domain = reader |> @lib.async_read_string() |> Some
      (2, _) => msg.version = reader |> @lib.async_read_int64() |> Some
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for OperatorSetIdProto with fn write(
  self : OperatorSetIdProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.domain is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  if self.version is Some(v) {
    writer |> @lib.async_write_varint(16UL)
    writer |> @lib.async_write_int64(v)
  }
}

///|
#deprecated("Use `==` instead")
pub extend OperatorSetIdProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend OperatorSetIdProto with Eq::{not_equal}

///|
pub extend OperatorSetIdProto with Debug::{to_repr}

///|
pub extend OperatorSetIdProto with Default::{default}

///|
pub extend OperatorSetIdProto with @lib.Sized::{size_of}

///|
pub extend OperatorSetIdProto with @lib.Read::{read, read_with_limit}

///|
pub extend OperatorSetIdProto with @lib.Write::{write}

///|
#derive(Eq, Debug)
pub(all) struct FunctionProto {
  mut name : String?
  mut input : Array[String]
  mut output : Array[String]
  mut attribute : Array[String]
  mut attribute_proto : Array[AttributeProto]
  mut node : Array[NodeProto]
  mut doc_string : String?
  mut opset_import : Array[OperatorSetIdProto]
  mut domain : String?
  mut overload : String?
  mut value_info : Array[ValueInfoProto]
  mut metadata_props : Array[StringStringEntryProto]
}

///|
pub impl @lib.Sized for FunctionProto with fn size_of(self) {
  let mut size = 0U
  if self.name is Some(v) {
    size += @lib.size_of_length_delimited_field(1U, @lib.size_of(v))
  }
  for s in self.input {
    size += @lib.size_of_length_delimited_field(4U, @lib.size_of(s))
  }
  for s in self.output {
    size += @lib.size_of_length_delimited_field(5U, @lib.size_of(s))
  }
  for s in self.attribute {
    size += @lib.size_of_length_delimited_field(6U, @lib.size_of(s))
  }
  for s in self.attribute_proto {
    size += @lib.size_of_length_delimited_field(11U, @lib.size_of(s))
  }
  for s in self.node {
    size += @lib.size_of_length_delimited_field(7U, @lib.size_of(s))
  }
  if self.doc_string is Some(v) {
    size += @lib.size_of_length_delimited_field(8U, @lib.size_of(v))
  }
  for s in self.opset_import {
    size += @lib.size_of_length_delimited_field(9U, @lib.size_of(s))
  }
  if self.domain is Some(v) {
    size += @lib.size_of_length_delimited_field(10U, @lib.size_of(v))
  }
  if self.overload is Some(v) {
    size += @lib.size_of_length_delimited_field(13U, @lib.size_of(v))
  }
  for s in self.value_info {
    size += @lib.size_of_length_delimited_field(12U, @lib.size_of(s))
  }
  for s in self.metadata_props {
    size += @lib.size_of_length_delimited_field(14U, @lib.size_of(s))
  }
  size
}

///|
pub impl Default for FunctionProto with fn default() -> FunctionProto {
  FunctionProto::{
    name: None,
    input: [],
    output: [],
    attribute: [],
    attribute_proto: [],
    node: [],
    doc_string: None,
    opset_import: [],
    domain: None,
    overload: None,
    value_info: [],
    metadata_props: [],
  }
}

///|
#alias(new, deprecated)
pub fn FunctionProto::FunctionProto(
  name? : String,
  input : Array[String],
  output : Array[String],
  attribute : Array[String],
  attribute_proto : Array[AttributeProto],
  node : Array[NodeProto],
  doc_string? : String,
  opset_import : Array[OperatorSetIdProto],
  domain? : String,
  overload? : String,
  value_info : Array[ValueInfoProto],
  metadata_props : Array[StringStringEntryProto],
) -> FunctionProto {
  FunctionProto::{
    name,
    input,
    output,
    attribute,
    attribute_proto,
    node,
    doc_string,
    opset_import,
    domain,
    overload,
    value_info,
    metadata_props,
  }
}

///|
pub impl @lib.Read for FunctionProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.Reader],
) -> FunctionProto raise {
  let msg : FunctionProto = Default::default()
  while (reader |> @lib.read_next_message_field()) is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.read_string() |> Some
      (4, _) => msg.input.push(reader |> @lib.read_string())
      (5, _) => msg.output.push(reader |> @lib.read_string())
      (6, _) => msg.attribute.push(reader |> @lib.read_string())
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.attribute_proto.push(
          (reader |> @lib.read_message() : AttributeProto),
        )
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.node.push((reader |> @lib.read_message() : NodeProto))
      (8, _) => msg.doc_string = reader |> @lib.read_string() |> Some
      (9, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.opset_import.push(
          (reader |> @lib.read_message() : OperatorSetIdProto),
        )
      (10, _) => msg.domain = reader |> @lib.read_string() |> Some
      (13, _) => msg.overload = reader |> @lib.read_string() |> Some
      (12, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value_info.push((reader |> @lib.read_message() : ValueInfoProto))
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.Write for FunctionProto with fn write(
  self : FunctionProto,
  writer : &@lib.Writer,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.write_varint(10UL)
    writer |> @lib.write_string(v)
  }
  for item in self.input {
    writer |> @lib.write_varint(34UL)
    writer |> @lib.write_string(item)
  }
  for item in self.output {
    writer |> @lib.write_varint(42UL)
    writer |> @lib.write_string(item)
  }
  for item in self.attribute {
    writer |> @lib.write_varint(50UL)
    writer |> @lib.write_string(item)
  }
  for item in self.attribute_proto {
    writer |> @lib.write_varint(90UL)
    writer |> @lib.write_message(item)
  }
  for item in self.node {
    writer |> @lib.write_varint(58UL)
    writer |> @lib.write_message(item)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.write_varint(66UL)
    writer |> @lib.write_string(v)
  }
  for item in self.opset_import {
    writer |> @lib.write_varint(74UL)
    writer |> @lib.write_message(item)
  }
  if self.domain is Some(v) {
    writer |> @lib.write_varint(82UL)
    writer |> @lib.write_string(v)
  }
  if self.overload is Some(v) {
    writer |> @lib.write_varint(106UL)
    writer |> @lib.write_string(v)
  }
  for item in self.value_info {
    writer |> @lib.write_varint(98UL)
    writer |> @lib.write_message(item)
  }
  for item in self.metadata_props {
    writer |> @lib.write_varint(114UL)
    writer |> @lib.write_message(item)
  }
}

///|
pub impl @lib.AsyncRead for FunctionProto with fn read_with_limit(
  reader : @lib.LimitedReader[&@lib.AsyncReader],
) -> FunctionProto raise {
  let msg : FunctionProto = Default::default()
  while (reader |> @lib.async_read_next_message_field())
        is Some((field_number, wire)) {
    match (field_number, wire) {
      (1, _) => msg.name = reader |> @lib.async_read_string() |> Some
      (4, _) => msg.input.push(reader |> @lib.async_read_string())
      (5, _) => msg.output.push(reader |> @lib.async_read_string())
      (6, _) => msg.attribute.push(reader |> @lib.async_read_string())
      (11, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.attribute_proto.push(
          (reader |> @lib.async_read_message() : AttributeProto),
        )
      (7, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.node.push((reader |> @lib.async_read_message() : NodeProto))
      (8, _) => msg.doc_string = reader |> @lib.async_read_string() |> Some
      (9, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.opset_import.push(
          (reader |> @lib.async_read_message() : OperatorSetIdProto),
        )
      (10, _) => msg.domain = reader |> @lib.async_read_string() |> Some
      (13, _) => msg.overload = reader |> @lib.async_read_string() |> Some
      (12, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.value_info.push(
          (reader |> @lib.async_read_message() : ValueInfoProto),
        )
      (14, @lib.WIRE_TYPE_LENGTH_DELIMITED) =>
        msg.metadata_props.push(
          (reader |> @lib.async_read_message() : StringStringEntryProto),
        )
      _ => reader |> @lib.async_skip_message_field_by_number(field_number, wire)
    }
  }
  msg
}

///|
pub impl @lib.AsyncWrite for FunctionProto with fn write(
  self : FunctionProto,
  writer : &@lib.AsyncWriter,
) -> Unit raise {
  if self.name is Some(v) {
    writer |> @lib.async_write_varint(10UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.input {
    writer |> @lib.async_write_varint(34UL)
    writer |> @lib.async_write_string(item)
  }
  for item in self.output {
    writer |> @lib.async_write_varint(42UL)
    writer |> @lib.async_write_string(item)
  }
  for item in self.attribute {
    writer |> @lib.async_write_varint(50UL)
    writer |> @lib.async_write_string(item)
  }
  for item in self.attribute_proto {
    writer |> @lib.async_write_varint(90UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.node {
    writer |> @lib.async_write_varint(58UL)
    writer |> @lib.async_write_message(item)
  }
  if self.doc_string is Some(v) {
    writer |> @lib.async_write_varint(66UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.opset_import {
    writer |> @lib.async_write_varint(74UL)
    writer |> @lib.async_write_message(item)
  }
  if self.domain is Some(v) {
    writer |> @lib.async_write_varint(82UL)
    writer |> @lib.async_write_string(v)
  }
  if self.overload is Some(v) {
    writer |> @lib.async_write_varint(106UL)
    writer |> @lib.async_write_string(v)
  }
  for item in self.value_info {
    writer |> @lib.async_write_varint(98UL)
    writer |> @lib.async_write_message(item)
  }
  for item in self.metadata_props {
    writer |> @lib.async_write_varint(114UL)
    writer |> @lib.async_write_message(item)
  }
}

///|
#deprecated("Use `==` instead")
pub extend FunctionProto with Eq::{equal}

///|
#deprecated("Use `!=` instead")
pub extend FunctionProto with Eq::{not_equal}

///|
pub extend FunctionProto with Debug::{to_repr}

///|
pub extend FunctionProto with Default::{default}

///|
pub extend FunctionProto with @lib.Sized::{size_of}

///|
pub extend FunctionProto with @lib.Read::{read, read_with_limit}

///|
pub extend FunctionProto with @lib.Write::{write}
