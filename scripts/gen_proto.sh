#!/usr/bin/env bash
# 使用官方 protoc 与 moonbitlang/protoc-gen-mbt 从 packages/onnx_proto/proto/onnx.proto
# 重新生成 packages/onnx_proto/schema/ 下的 MoonBit Protobuf 绑定代码。

set -euo pipefail

export PATH="$HOME/.moon/bin:$PATH"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="${ROOT_DIR}/packages/onnx_proto/proto"
SCHEMA_DIR="${ROOT_DIR}/packages/onnx_proto/schema"
TMP_OUT="$(mktemp -d)"
trap 'rm -rf "$TMP_OUT"' EXIT

echo "==> Generating MoonBit ONNX protobuf bindings into packages/onnx_proto/schema/..."
protoc \
  --plugin="protoc-gen-mbt=${ROOT_DIR}/scripts/protoc-gen-mbt.sh" \
  --mbt_out="$TMP_OUT" \
  --mbt_opt="project_name=onnx_gen,json=false" \
  -I "${PROTO_DIR}" \
  "${PROTO_DIR}/onnx.proto"

rm -rf "${SCHEMA_DIR}"
mkdir -p "${SCHEMA_DIR}"

cat > "${SCHEMA_DIR}/moon.pkg" <<'EOF'
import {
  "moonbitlang/protobuf" @lib,
}
EOF

cp "${TMP_OUT}/onnx_gen/src/onnx/top.mbt" "${SCHEMA_DIR}/top.mbt"

echo "==> Formatting workspace..."
(cd "${ROOT_DIR}" && moon fmt)

echo "==> Generated packages/onnx_proto/schema/ successfully."
