# moon-onnx

使用 MoonBit 开发的 ONNX 模型解析与跨端推理引擎。

## 项目定位

- **Native 优先验证**：优先在 `--target native` 上跑通完整解析、算子执行与 ONNX Runtime 数值对齐，同时保持核心代码为纯 MoonBit，兼顾 Wasm / Wasm-GC / JS 跨端可编译性。
- **Protobuf 代码生成**：基于官方 `onnx.proto` 与 `moonbitlang/protoc-gen-mbt` 生成编解码代码，并由 `src/model` 提供清晰的业务封装层。
- **渐进式验证**：从最简 MNIST（MLP → CNN）模型端到端闭环起步，随后落地基于 MoonBit 内建 `V128`（`f32x4`）的跨端 SIMD 内核并量化评估加速效果。

## 常用命令

```bash
# 格式化 + 类型检查 + 单元测试（每次提交前必跑）
./scripts/pre_commit_check.sh

# 单独运行 Native 测试
moon test --target native
```

## 开发日志

详细的设计演进、踩坑记录、数值对齐数据与性能测试结果均持续记录在 [docs/DEVLOG.md](docs/DEVLOG.md)。
