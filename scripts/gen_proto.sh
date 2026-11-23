#!/usr/bin/env bash
# 使用官方 protoc 与 moonbitlang/protoc-gen-mbt 从 proto/onnx.proto 重新生成
# src/proto_gen/ 下的 MoonBit Protobuf 结构体与二进制编解码代码。
#
# 注意：
# 1. 显式传入 json=false，规避 protoc-gen-mbt@0.3.0 对 repeated bytes 字段生成 to_json 时的类型不匹配问题
#    （ONNX 模型仅使用二进制 Protobuf 格式，不需要 Protobuf JSON 编解码）。
# 2. src/proto_gen/ 属于自动生成产物，请勿手工修改；业务适配统一写在 src/model/ 中。

set -euo pipefail

export PATH="$HOME/.moon/bin:$PATH"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_OUT="$(mktemp -d)"
trap 'rm -rf "$TMP_OUT"' EXIT

echo "==> Generating MoonBit ONNX protobuf bindings from proto/onnx.proto..."
protoc \
  --plugin="protoc-gen-mbt=${ROOT_DIR}/scripts/protoc-gen-mbt.sh" \
  --mbt_out="$TMP_OUT" \
  --mbt_opt="project_name=onnx_gen,json=false" \
  -I "${ROOT_DIR}/proto" \
  "${ROOT_DIR}/proto/onnx.proto"

rm -rf "${ROOT_DIR}/src/proto_gen"
mkdir -p "${ROOT_DIR}/src/proto_gen"

# 禁用 json=false 后不再需要 moonbitlang/core/json 依赖，避免 unused_package 警告
cat > "${ROOT_DIR}/src/proto_gen/moon.pkg" <<'EOF'
import {
  "moonbitlang/protobuf" @lib,
}
EOF

cp "${TMP_OUT}/onnx_gen/src/onnx/top.mbt" "${ROOT_DIR}/src/proto_gen/top.mbt"

echo "==> Formatting generated package..."
(cd "${ROOT_DIR}" && moon fmt)

echo "==> Generated src/proto_gen/ successfully."
