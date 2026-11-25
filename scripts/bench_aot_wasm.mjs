#!/usr/bin/env node
// 端到端验证并对比 moon-onnx AOT 编译器 (`onnx2mbt`) 各模式在 Node.js V8 Wasm / Wasm-GC 下的产物体积、冷启动与推理性能。
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const repoRoot = process.cwd();
const workDir = "/tmp/moon_onnx_aot_bench";
const moonEnv = {
  ...process.env,
  PATH: `${process.env.HOME}/.moon/bin:${process.env.PATH ?? ""}`,
};

function runCmd(cmd, args, cwd = repoRoot) {
  return execFileSync(cmd, args, {
    cwd,
    env: moonEnv,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

fs.rmSync(workDir, { recursive: true, force: true });
fs.mkdirSync(workDir, { recursive: true });

// 1. 生成 4 种 AOT 编译产物（均来自同一真实模型 tests/fixtures/mnist_cnn.onnx）
const modes = [
  {
    id: "v1_tensor_wasm",
    label: "v1 Tensor (wasm, ext weights)",
    cliArgs: ["--target=wasm", "--mode=tensor"],
    buildTarget: "wasm",
    embedded: false,
    isTensorMode: true,
  },
  {
    id: "v2_arena_wasm",
    label: "v2 Arena  (wasm V128, ext weights)",
    cliArgs: ["--target=wasm", "--mode=arena"],
    buildTarget: "wasm",
    embedded: false,
    isTensorMode: false,
  },
  {
    id: "v2_arena_wasm_emb",
    label: "v2 Arena  (wasm V128, 1-file embedded)",
    cliArgs: ["--target=wasm", "--mode=arena", "--embed-weights"],
    buildTarget: "wasm",
    embedded: true,
    isTensorMode: false,
  },
  {
    id: "v2_arena_wasmgc_ext",
    label: "v2 Arena  (wasm-gc adaptive, ext)",
    cliArgs: ["--target=wasm-gc", "--mode=arena"],
    buildTarget: "wasm-gc",
    embedded: false,
    isTensorMode: false,
  },
  {
    id: "v2_arena_wasmgc_emb",
    label: "v2 Arena  (wasm-gc adaptive, 1-file)",
    cliArgs: ["--target=wasm-gc", "--mode=arena", "--embed-weights"],
    buildTarget: "wasm-gc",
    embedded: true,
    isTensorMode: false,
  },
];

for (const m of modes) {
  const outDir = path.join(workDir, m.id);
  runCmd("moon", [
    "run",
    "apps/cli",
    "--",
    "compile",
    "tests/fixtures/mnist_cnn.onnx",
    outDir,
    "--input-shape=1,1,28,28",
    "--struct=MnistCnn",
    ...m.cliArgs,
  ]);
}

// 2. 组装临时多模块工作区，分别编译为 --release .wasm
fs.writeFileSync(
  path.join(workDir, "moon.work"),
  `members = [
  "${path.join(repoRoot, "packages/tensor")}",
  "${path.join(repoRoot, "packages/ml_kernels")}",
  "${path.join(repoRoot, "packages/onnx_proto")}",
  "${path.join(repoRoot, "packages/nn_ir")}",
  "${path.join(repoRoot, "packages/onnx")}",
${modes.map((m) => `  "./${m.id}",`).join("\n")}
]
`
);

for (const m of modes) {
  const dir = path.join(workDir, m.id);
  const importsMod = m.isTensorMode
    ? `  "matrixlt/tensor@0.2.0",\n  "matrixlt/ml_kernels@0.2.0",\n  "matrixlt/nn_ir@0.2.0",\n  "matrixlt/onnx@0.2.0",`
    : `  "matrixlt/ml_kernels@0.2.0",`;
  fs.writeFileSync(
    path.join(dir, "moon.mod"),
    `name = "demo/${m.id}"
version = "0.2.0"
license = "Apache-2.0"
supported_targets = "+all"
import {
${importsMod}
}
`
  );

  const exportsList = m.embedded
    ? `["init_model", "set_input_f32", "run_infer", "get_output_f32"]`
    : `["set_weight_byte", "init_model", "set_input_f32", "run_infer", "get_output_f32"]`;
  const basePkg = fs.readFileSync(path.join(dir, "moon.pkg"), "utf8");
  fs.writeFileSync(
    path.join(dir, "moon.pkg"),
    `${basePkg}
options(
  link: {
    "wasm": { "exports": ${exportsList} },
    "wasm-gc": { "exports": ${exportsList} },
  },
)
`
  );

  const entrySrc = m.isTensorMode
    ? `let w_buf : FixedArray[Byte] = FixedArray::make(23976, b'\\x00')
let in_buf : FixedArray[Float] = FixedArray::make(784, (0.0 : Float))
let out_buf : FixedArray[Float] = FixedArray::make(10, (0.0 : Float))
let model_ref : Ref[MnistCnn?] = { val: None }

pub fn set_weight_byte(idx : Int32, val : Int32) -> Unit { w_buf[idx] = val.to_byte() }
pub fn set_input_f32(idx : Int32, val : Float) -> Unit { in_buf[idx] = val }
pub fn init_model() -> Int32 {
  try {
    model_ref.val = Some(MnistCnn::load(Bytes::makei(w_buf.length(), i => w_buf[i])))
    1
  } catch { _ => -1 }
}
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => try {
      let y = m.forward_f32(in_buf)
      for i = 0; i < 10; i = i + 1 { out_buf[i] = y[i] }
      1
    } catch { _ => -1 }
    None => -2
  }
}
pub fn get_output_f32(idx : Int32) -> Float { out_buf[idx] }
`
    : m.embedded
    ? `let in_buf : FixedArray[Float] = FixedArray::make(784, (0.0 : Float))
let out_buf : FixedArray[Float] = FixedArray::make(10, (0.0 : Float))
let model_ref : Ref[MnistCnn?] = { val: None }

pub fn set_input_f32(idx : Int32, val : Float) -> Unit { in_buf[idx] = val }
pub fn init_model() -> Int32 { model_ref.val = Some(MnistCnn::new()); 1 }
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => { m.forward_into(in_buf, out_buf); 1 }
    None => -1
  }
}
pub fn get_output_f32(idx : Int32) -> Float { out_buf[idx] }
`
    : `let w_buf : FixedArray[Byte] = FixedArray::make(23976, b'\\x00')
let in_buf : FixedArray[Float] = FixedArray::make(784, (0.0 : Float))
let out_buf : FixedArray[Float] = FixedArray::make(10, (0.0 : Float))
let model_ref : Ref[MnistCnn?] = { val: None }

pub fn set_weight_byte(idx : Int32, val : Int32) -> Unit { w_buf[idx] = val.to_byte() }
pub fn set_input_f32(idx : Int32, val : Float) -> Unit { in_buf[idx] = val }
pub fn init_model() -> Int32 {
  model_ref.val = Some(MnistCnn::load(Bytes::makei(w_buf.length(), i => w_buf[i])))
  1
}
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => { m.forward_into(in_buf, out_buf); 1 }
    None => -1
  }
}
pub fn get_output_f32(idx : Int32) -> Float { out_buf[idx] }
`;
  fs.writeFileSync(path.join(dir, "entry.mbt"), entrySrc);
}

// 3. 编译并在 V8 中逐一验证精度与性能
const weights = fs.readFileSync(path.join(workDir, "v2_arena_wasm", "weights.bin"));
const inBuf = fs.readFileSync(path.join(repoRoot, "tests/fixtures/mnist_cnn_input.f32"));
const inF32 = new Float32Array(inBuf.buffer, inBuf.byteOffset, inBuf.byteLength / 4);
const goldBuf = fs.readFileSync(path.join(repoRoot, "tests/fixtures/mnist_cnn_golden.f32"));
const goldF32 = new Float32Array(goldBuf.buffer, goldBuf.byteOffset, goldBuf.byteLength / 4);

console.log("=== onnx2mbt AOT Wasm Benchmark (Real MNIST CNN vs ORT Golden) ===");
for (const m of modes) {
  const buildOut = runCmd(
    "moon",
    ["-C", workDir, "build", "--target", m.buildTarget, "--release", m.id],
    repoRoot
  );
  if (buildOut.includes("Warning:")) {
    throw new Error(`Unexpected compiler warning in generated ${m.id}:\n${buildOut}`);
  }
  const wasmPath = path.join(
    workDir,
    "_build",
    m.buildTarget,
    "release/build/demo",
    m.id,
    `${m.id}.wasm`
  );
  const raw = fs.readFileSync(wasmPath);
  const gz = zlib.gzipSync(raw);

  const t0 = performance.now();
  const { instance } = await WebAssembly.instantiate(raw, {});
  const instMs = performance.now() - t0;
  const ex = instance.exports;

  if (!m.embedded) {
    for (let i = 0; i < weights.length; i++) ex.set_weight_byte(i, weights[i]);
  }
  for (let i = 0; i < inF32.length; i++) ex.set_input_f32(i, inF32[i]);
  if (ex.init_model() !== 1) throw new Error(`${m.id} init_model failed`);

  // 预热
  for (let i = 0; i < 10; i++) ex.run_infer();

  let maxDiff = 0;
  let predCls = 0;
  let predVal = -1e9;
  for (let i = 0; i < 10; i++) {
    const v = ex.get_output_f32(i);
    maxDiff = Math.max(maxDiff, Math.abs(v - goldF32[i]));
    if (v > predVal) {
      predVal = v;
      predCls = i;
    }
  }
  if (predCls !== 0 || maxDiff > 1e-5) {
    throw new Error(`${m.id} verification failed: cls=${predCls}, diff=${maxDiff}`);
  }

  const runs = 200;
  const b0 = performance.now();
  for (let i = 0; i < runs; i++) ex.run_infer();
  const avgUs = ((performance.now() - b0) * 1000) / runs;
  const fps = 1e6 / avgUs;

  console.log(
    `  ${m.label.padEnd(39)} | raw=${(raw.length / 1024)
      .toFixed(1)
      .padStart(5)} KB | gz=${(gz.length / 1024)
      .toFixed(1)
      .padStart(4)} KB | inst=${instMs.toFixed(2).padStart(4)} ms | diff=${maxDiff.toExponential(
      2
    )} | ${avgUs.toFixed(1).padStart(6)} us (${fps.toFixed(0).padStart(3)} FPS)`
  );
}

fs.rmSync(workDir, { recursive: true, force: true });
