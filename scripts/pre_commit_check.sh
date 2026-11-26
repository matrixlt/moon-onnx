#!/usr/bin/env bash
# 提交前强制质量检查脚本（全 moon.work 多模块工作区）：
# 1. 使用 moon fmt 格式化全部模块的 MoonBit 源码
# 2. 对全部 4 个目标平台（native / wasm / wasm-gc / js）执行静态类型检查
# 3. 执行 Native 全量单元测试（含真实 MNIST MLP / CNN 与 ORT 差分对齐）
# 4. 在 wasm / wasm-gc / js 三端验证全部纯内存模块（tensor / ml_kernels / onnx_proto / nn_ir / onnx / onnx_aot）单测

set -euo pipefail

export PATH="$HOME/.moon/bin:$PATH"

echo "==> [1/4] Formatting MoonBit workspace (moon fmt)..."
moon fmt

echo "==> [2/4] Checking all 4 targets (moon check --target all)..."
moon check --target all

echo "==> [3/4] Running Native tests across workspace (moon test --target native)..."
moon test --target native

echo "==> [4/4] Verifying all pure-memory modules on wasm, wasm-gc, and js..."
for target in wasm wasm-gc js; do
  moon test --target "$target" \
    -p matrixlt/tensor \
    -p matrixlt/ml_kernels/gemm \
    -p matrixlt/ml_kernels/vecmath \
    -p matrixlt/ml_kernels/conv \
    -p matrixlt/ml_kernels/quant \
    -p matrixlt/ml_kernels/nn \
    -p matrixlt/onnx_proto \
    -p matrixlt/nn_ir \
    -p matrixlt/nn_ir/opt \
    -p matrixlt/nn_ir/analysis \
    -p matrixlt/onnx/frontend \
    -p matrixlt/onnx/ops \
    -p matrixlt/onnx/session \
    -p matrixlt/onnx \
    -p matrixlt/onnx_aot
done

echo "==> All workspace pre-commit checks passed!"
