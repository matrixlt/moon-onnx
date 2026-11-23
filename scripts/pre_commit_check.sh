#!/usr/bin/env bash
# 提交前强制质量检查脚本：
# 1. 使用 moon fmt 格式化全部 MoonBit 源码
# 2. 执行 Native 目标类型与静态检查
# 3. 执行 Native 目标全量单元测试

set -euo pipefail

export PATH="$HOME/.moon/bin:$PATH"

echo "==> [1/3] Formatting MoonBit code (moon fmt)..."
moon fmt

echo "==> [2/3] Checking Native target (moon check --target native)..."
moon check --target native

echo "==> [3/3] Running Native tests (moon test --target native)..."
moon test --target native

echo "==> All pre-commit checks passed!"
