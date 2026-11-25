#!/usr/bin/env node
// 真实开源生产模型（Silero VAD v5 + MobileNetV2-12）Wasm AOT 端到端编译、精度对齐（vs ORT）与性能基准脚本。
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const repoRoot = process.cwd();
const modelDir = process.env.REAL_MODELS_DIR ?? "/tmp/real_models";
const workDir = "/tmp/moon_onnx_real_aot_bench";
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

function readF32(filePath) {
  const buf = fs.readFileSync(filePath);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function maxAbsDiff(a, b) {
  if (a.length !== b.length) {
    throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
  }
  let mx = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > mx) mx = d;
  }
  return mx;
}

function argmax(arr) {
  let bestIdx = 0;
  let bestVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bestVal) {
      bestVal = arr[i];
      bestIdx = i;
    }
  }
  return { idx: bestIdx, val: bestVal };
}

fs.rmSync(workDir, { recursive: true, force: true });
fs.mkdirSync(workDir, { recursive: true });

const targets = [
  {
    id: "silero_vad_wasm",
    label: "Silero VAD 16kHz (wasm V128 Arena)",
    model: path.join(modelDir, "silero_vad.onnx"),
    struct: "SileroVad16k",
    cliArgs: [
      "--input-shape=input:1,512;state:2,1,128",
      "--freeze-i64=sr:16000",
      "--target=wasm",
      "--mode=arena",
    ],
    buildTarget: "wasm",
    kind: "silero",
  },
  {
    id: "silero_vad_wasmgc",
    label: "Silero VAD 16kHz (wasm-gc Arena)",
    model: path.join(modelDir, "silero_vad.onnx"),
    struct: "SileroVad16k",
    cliArgs: [
      "--input-shape=input:1,512;state:2,1,128",
      "--freeze-i64=sr:16000",
      "--target=wasm-gc",
      "--mode=arena",
    ],
    buildTarget: "wasm-gc",
    kind: "silero",
  },
  {
    id: "mobilenetv2_wasm",
    label: "MobileNetV2-12   (wasm V128 Arena)",
    model: path.join(modelDir, "mobilenetv2-12.onnx"),
    struct: "MobileNetV2",
    cliArgs: [
      "--input-shape=1,3,224,224",
      "--target=wasm",
      "--mode=arena",
    ],
    buildTarget: "wasm",
    kind: "mobilenet",
  },
  {
    id: "mobilenetv2_wasmgc",
    label: "MobileNetV2-12   (wasm-gc Arena)",
    model: path.join(modelDir, "mobilenetv2-12.onnx"),
    struct: "MobileNetV2",
    cliArgs: [
      "--input-shape=1,3,224,224",
      "--target=wasm-gc",
      "--mode=arena",
    ],
    buildTarget: "wasm-gc",
    kind: "mobilenet",
  },
];

// 1. 使用 onnx2mbt AOT 编译所有目标
for (const t of targets) {
  const outDir = path.join(workDir, t.id);
  runCmd("moon", [
    "run",
    "--release",
    "apps/cli",
    "--",
    "compile",
    t.model,
    outDir,
    `--struct=${t.struct}`,
    ...t.cliArgs,
  ]);
}

// 2. 组装临时多模块工作区
fs.writeFileSync(
  path.join(workDir, "moon.work"),
  `members = [
  "${path.join(repoRoot, "packages/ml_kernels")}",
${targets.map((t) => `  "./${t.id}",`).join("\n")}
]
`
);

for (const t of targets) {
  const dir = path.join(workDir, t.id);
  const weightBytes = fs.readFileSync(path.join(dir, "weights.bin")).length;
  fs.writeFileSync(
    path.join(dir, "moon.mod"),
    `name = "real_aot/${t.id}"
version = "0.2.0"
license = "Apache-2.0"
supported_targets = "+all"
import {
  "matrixlt/ml_kernels@0.2.0",
}
`
  );

  const basePkg = fs.readFileSync(path.join(dir, "moon.pkg"), "utf8");
  const exportsList =
    t.kind === "silero"
      ? `["set_weight_u32", "init_model", "set_audio_f32", "set_state_f32", "run_infer", "get_prob_f32", "get_state_f32", "get_arena_bytes"]`
      : `["set_weight_u32", "init_model", "set_input_f32", "run_infer", "get_output_f32", "get_arena_bytes"]`;
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

  const numWords = Math.ceil(weightBytes / 4);
  const weightLoaderSnippet = `let w_buf : FixedArray[Byte] = FixedArray::make(${weightBytes}, b'\\x00')
pub fn set_weight_u32(word_idx : Int32, val : Int32) -> Unit {
  let off = word_idx * 4
  if off < ${weightBytes} { w_buf[off] = (val & 0xFF).to_byte() }
  if off + 1 < ${weightBytes} { w_buf[off + 1] = ((val >> 8) & 0xFF).to_byte() }
  if off + 2 < ${weightBytes} { w_buf[off + 2] = ((val >> 16) & 0xFF).to_byte() }
  if off + 3 < ${weightBytes} { w_buf[off + 3] = ((val >> 24) & 0xFF).to_byte() }
}
pub fn get_arena_bytes() -> Int32 { ${t.struct}::arena_bytes() }
`;

  const entrySrc =
    t.kind === "silero"
      ? `${weightLoaderSnippet}
let audio_buf : FixedArray[Float] = FixedArray::make(512, (0.0 : Float))
let state_in : FixedArray[Float] = FixedArray::make(256, (0.0 : Float))
let prob_out : FixedArray[Float] = FixedArray::make(1, (0.0 : Float))
let state_out : FixedArray[Float] = FixedArray::make(256, (0.0 : Float))
let model_ref : Ref[${t.struct}?] = { val: None }

pub fn init_model() -> Int32 {
  model_ref.val = Some(${t.struct}::load(Bytes::makei(${weightBytes}, fn(i) { w_buf[i] })))
  1
}
pub fn set_audio_f32(idx : Int32, val : Float) -> Unit { audio_buf[idx] = val }
pub fn set_state_f32(idx : Int32, val : Float) -> Unit { state_in[idx] = val }
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => {
      m.forward_into(audio_buf, state_in, prob_out, state_out)
      1
    }
    None => -1
  }
}
pub fn get_prob_f32() -> Float { prob_out[0] }
pub fn get_state_f32(idx : Int32) -> Float { state_out[idx] }
`
      : `${weightLoaderSnippet}
let in_buf : FixedArray[Float] = FixedArray::make(150528, (0.0 : Float))
let out_buf : FixedArray[Float] = FixedArray::make(1000, (0.0 : Float))
let model_ref : Ref[${t.struct}?] = { val: None }

pub fn init_model() -> Int32 {
  model_ref.val = Some(${t.struct}::load(Bytes::makei(${weightBytes}, fn(i) { w_buf[i] })))
  1
}
pub fn set_input_f32(idx : Int32, val : Float) -> Unit { in_buf[idx] = val }
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => {
      m.forward_into(in_buf, out_buf)
      1
    }
    None => -1
  }
}
pub fn get_output_f32(idx : Int32) -> Float { out_buf[idx] }
`;
  fs.writeFileSync(path.join(dir, "entry.mbt"), entrySrc);
  t.numWords = numWords;
  t.weightBytes = weightBytes;
}

// 3. 编译为 --release .wasm 并校验零警告
for (const t of targets) {
  const dir = path.join(workDir, t.id);
  const out = runCmd("moon", ["build", "--release", "--target", t.buildTarget], dir);
  if (out.includes("Warning")) {
    throw new Error(`Unexpected compiler warning in ${t.id}:\n${out}`);
  }
}

// 4. 在 Node.js V8 中加载并对比 ORT Golden 输出 + 基准测试
console.log("=== Real Open-Source Models Wasm AOT Benchmark (vs Official ORT 1.32.0) ===");

const audio1 = readF32(path.join(modelDir, "silero_audio1.f32"));
const audio2 = readF32(path.join(modelDir, "silero_audio2.f32"));
const out1Gold = readF32(path.join(modelDir, "silero_out1_golden.f32"));
const st1Gold = readF32(path.join(modelDir, "silero_state1_golden.f32"));
const out2Gold = readF32(path.join(modelDir, "silero_out2_golden.f32"));
const st2Gold = readF32(path.join(modelDir, "silero_state2_golden.f32"));

const mnIn = readF32(path.join(modelDir, "mobilenetv2_input.f32"));
const mnGold = readF32(path.join(modelDir, "mobilenetv2_golden.f32"));
const mnGoldTop = argmax(mnGold);

for (const t of targets) {
  const dir = path.join(workDir, t.id);
  const wasmPath = path.join(
    workDir,
    "_build",
    t.buildTarget,
    "release",
    "build",
    "real_aot",
    t.id,
    `${t.id}.wasm`
  );
  const wasmBytes = fs.readFileSync(wasmPath);
  const gzBytes = zlib.gzipSync(wasmBytes, { level: 9 });

  const tInst0 = performance.now();
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exp = instance.exports;
  const instMs = performance.now() - tInst0;

  // 写入权重字节（按 4 字节字写入，初始化更快）
  const weightsRaw = fs.readFileSync(path.join(dir, "weights.bin"));
  const view = new DataView(weightsRaw.buffer, weightsRaw.byteOffset, weightsRaw.byteLength);
  for (let w = 0; w < t.numWords; w++) {
    const off = w * 4;
    let word = 0;
    if (off + 4 <= weightsRaw.length) {
      word = view.getInt32(off, true);
    } else {
      for (let b = 0; b < weightsRaw.length - off; b++) {
        word |= weightsRaw[off + b] << (b * 8);
      }
    }
    exp.set_weight_u32(w, word);
  }
  const tLoad0 = performance.now();
  if (exp.init_model() !== 1) throw new Error(`init_model failed for ${t.id}`);
  const loadMs = performance.now() - tLoad0;
  const arenaKB = (exp.get_arena_bytes() / 1024).toFixed(1);

  if (t.kind === "silero") {
    // Step 1
    for (let i = 0; i < 512; i++) exp.set_audio_f32(i, audio1[i]);
    for (let i = 0; i < 256; i++) exp.set_state_f32(i, 0.0);
    exp.run_infer();
    const p1 = new Float32Array([exp.get_prob_f32()]);
    const s1 = new Float32Array(256);
    for (let i = 0; i < 256; i++) s1[i] = exp.get_state_f32(i);

    // Step 2（流式状态递推）
    for (let i = 0; i < 512; i++) exp.set_audio_f32(i, audio2[i]);
    for (let i = 0; i < 256; i++) exp.set_state_f32(i, s1[i]);
    exp.run_infer();
    const p2 = new Float32Array([exp.get_prob_f32()]);
    const s2 = new Float32Array(256);
    for (let i = 0; i < 256; i++) s2[i] = exp.get_state_f32(i);

    const d1 = maxAbsDiff(p1, out1Gold);
    const ds1 = maxAbsDiff(s1, st1Gold);
    const d2 = maxAbsDiff(p2, out2Gold);
    const ds2 = maxAbsDiff(s2, st2Gold);
    const worst = Math.max(d1, ds1, d2, ds2);
    if (worst > 1e-4) {
      throw new Error(`${t.id} accuracy mismatch: worst=${worst}`);
    }

    // 预热 + 50 帧计时（每帧对应 32ms 16kHz 音频）
    for (let r = 0; r < 5; r++) exp.run_infer();
    const iters = 50;
    const t0 = performance.now();
    for (let r = 0; r < iters; r++) exp.run_infer();
    const avgMs = (performance.now() - t0) / iters;
    const rtf = avgMs / 32.0;

    console.log(
      `  ${t.label.padEnd(36)} | wasm=${(wasmBytes.length / 1024).toFixed(1).padStart(5)} KB (gz ${(gzBytes.length / 1024).toFixed(1).padStart(4)} KB) | arena=${arenaKB.padStart(6)} KB | inst=${instMs.toFixed(2)}ms | diff=${worst.toExponential(2)} | ${avgMs.toFixed(2)} ms/frame (${(1 / rtf).toFixed(1)}x RT)`
    );
  } else {
    for (let i = 0; i < mnIn.length; i++) exp.set_input_f32(i, mnIn[i]);
    exp.run_infer();
    const out = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) out[i] = exp.get_output_f32(i);
    const diff = maxAbsDiff(out, mnGold);
    const top = argmax(out);
    if (top.idx !== mnGoldTop.idx || diff > 1e-4) {
      throw new Error(
        `${t.id} mismatch: top=${top.idx} vs ${mnGoldTop.idx}, diff=${diff}`
      );
    }

    // 预热 + 5 次完整 224x224 ImageNet 推理计时
    exp.run_infer();
    const iters = 5;
    const t0 = performance.now();
    for (let r = 0; r < iters; r++) exp.run_infer();
    const avgMs = (performance.now() - t0) / iters;

    console.log(
      `  ${t.label.padEnd(36)} | wasm=${(wasmBytes.length / 1024).toFixed(1).padStart(5)} KB (gz ${(gzBytes.length / 1024).toFixed(1).padStart(4)} KB) | arena=${arenaKB.padStart(6)} KB | inst=${instMs.toFixed(2)}ms | diff=${diff.toExponential(2)} | top1=${top.idx} | ${avgMs.toFixed(1)} ms (${(1000 / avgMs).toFixed(1)} FPS)`
    );
  }
}
