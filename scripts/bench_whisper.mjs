#!/usr/bin/env node
// 真实 OpenAI Whisper-tiny（Encoder FP32 + Decoder INT8 Prefill + Decoder INT8 KV-Cache Step）
// 全链路 Zero-Allocation Wasm AOT 编译、精度对齐（vs ORT 1.32.0）与 V8 性能基准脚本。
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const repoRoot = process.cwd();
const whisperDir = process.env.WHISPER_DIR ?? "/tmp/real_models/whisper";
const workDir = "/tmp/moon_onnx_whisper_aot_bench";
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

const decStepShapes = [
  "input_ids:1,1",
  ...Array.from({ length: 4 }, (_, l) => [
    `past_key_values.${l}.decoder.key:1,6,4,64`,
    `past_key_values.${l}.decoder.value:1,6,4,64`,
    `past_key_values.${l}.encoder.key:1,6,1500,64`,
    `past_key_values.${l}.encoder.value:1,6,1500,64`,
  ]).flat(),
].join(";");

const buildTargets = [
  {
    id: "whisper_enc_wasm",
    label: "Whisper-tiny Enc FP32   (wasm V128)",
    model: path.join(whisperDir, "encoder_model.onnx"),
    struct: "WhisperTinyEnc",
    inputShape: "input_features:1,80,3000",
    buildTarget: "wasm",
    kind: "enc",
  },
  {
    id: "whisper_enc_wasmgc",
    label: "Whisper-tiny Enc FP32   (wasm-gc)",
    model: path.join(whisperDir, "encoder_model.onnx"),
    struct: "WhisperTinyEnc",
    inputShape: "input_features:1,80,3000",
    buildTarget: "wasm-gc",
    kind: "enc",
  },
  {
    id: "whisper_dec_pre_wasm",
    label: "Whisper-tiny DecPre Q8  (wasm V128)",
    model: path.join(whisperDir, "decoder_model_int8.onnx"),
    struct: "WhisperTinyDecPre",
    inputShape: "input_ids:1,4;encoder_hidden_states:1,1500,384",
    buildTarget: "wasm",
    kind: "dec_pre",
  },
  {
    id: "whisper_dec_step_wasm",
    label: "Whisper-tiny DecStep Q8 (wasm V128)",
    model: path.join(whisperDir, "decoder_with_past_model_int8.onnx"),
    struct: "WhisperTinyDecStep",
    inputShape: decStepShapes,
    buildTarget: "wasm",
    kind: "dec_step",
  },
  {
    id: "whisper_dec_step_wasmgc",
    label: "Whisper-tiny DecStep Q8 (wasm-gc)",
    model: path.join(whisperDir, "decoder_with_past_model_int8.onnx"),
    struct: "WhisperTinyDecStep",
    inputShape: decStepShapes,
    buildTarget: "wasm-gc",
    kind: "dec_step",
  },
];

// 1. AOT 编译全部 Whisper-tiny 子模型
for (const t of buildTargets) {
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
    `--input-shape=${t.inputShape}`,
    `--target=${t.buildTarget}`,
    "--mode=arena",
  ]);
}

// 2. 组装临时多模块工作区与导出包装器
fs.writeFileSync(
  path.join(workDir, "moon.work"),
  `members = [
  "${path.join(repoRoot, "packages/ml_kernels")}",
${buildTargets.map((t) => `  "./${t.id}",`).join("\n")}
]
`
);

for (const t of buildTargets) {
  const dir = path.join(workDir, t.id);
  const weightBytes = fs.readFileSync(path.join(dir, "weights.bin")).length;
  t.weightBytes = weightBytes;
  t.numWords = Math.ceil(weightBytes / 4);

  fs.writeFileSync(
    path.join(dir, "moon.mod"),
    `name = "whisper_aot/${t.id}"
version = "0.2.0"
license = "Apache-2.0"
supported_targets = "+all"
import {
  "matrixlt/ml_kernels@0.2.0",
}
`
  );
  const basePkg = fs.readFileSync(path.join(dir, "moon.pkg"), "utf8");
  fs.writeFileSync(
    path.join(dir, "moon.pkg"),
    `${basePkg}
pkgtype(kind: "foreign_library")
`
  );

  const wHeader = `let w_buf : FixedArray[Byte] = FixedArray::make(${weightBytes}, b'\\x00')
#export_name("set_weight_u32")
pub fn set_weight_u32(word_idx : Int32, val : Int32) -> Unit {
  let off = word_idx * 4
  if off < ${weightBytes} { w_buf[off] = (val & 0xFF).to_byte() }
  if off + 1 < ${weightBytes} { w_buf[off + 1] = ((val >> 8) & 0xFF).to_byte() }
  if off + 2 < ${weightBytes} { w_buf[off + 2] = ((val >> 16) & 0xFF).to_byte() }
  if off + 3 < ${weightBytes} { w_buf[off + 3] = ((val >> 24) & 0xFF).to_byte() }
}
#export_name("get_arena_bytes")
pub fn get_arena_bytes() -> Int32 { ${t.struct}::arena_bytes() }
let model_ref : Ref[${t.struct}?] = { val: None }
#export_name("init_model")
pub fn init_model() -> Int32 {
  model_ref.val = Some(${t.struct}::load(Bytes::makei(${weightBytes}, fn(i) { w_buf[i] })))
  1
}
`;

  let body = "";
  if (t.kind === "enc") {
    body = `let mel_buf : FixedArray[Float] = FixedArray::make(240000, (0.0 : Float))
let enc_out : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
#export_name("set_in_f32")
pub fn set_in_f32(idx : Int32, val : Float) -> Unit { mel_buf[idx] = val }
#export_name("run_infer")
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => { m.forward_into(mel_buf, enc_out); 1 }
    None => -1
  }
}
#export_name("get_out_f32")
pub fn get_out_f32(idx : Int32) -> Float { enc_out[idx] }
`;
  } else if (t.kind === "dec_pre") {
    body = `let ids_buf : FixedArray[Int64] = FixedArray::make(4, (0 : Int64))
let enc_buf : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let logits_buf : FixedArray[Float] = FixedArray::make(207460, (0.0 : Float))
let kv_dk0 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_dv0 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_ek0 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_ev0 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_dk1 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_dv1 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_ek1 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_ev1 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_dk2 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_dv2 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_ek2 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_ev2 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_dk3 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_dv3 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let kv_ek3 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let kv_ev3 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))

#export_name("set_id_i32")
pub fn set_id_i32(idx : Int32, val : Int32) -> Unit { ids_buf[idx] = val.to_int64() }
#export_name("set_enc_f32")
pub fn set_enc_f32(idx : Int32, val : Float) -> Unit { enc_buf[idx] = val }
#export_name("run_infer")
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => {
      m.forward_into(
        ids_buf, enc_buf, logits_buf,
        kv_dk0, kv_dv0, kv_ek0, kv_ev0,
        kv_dk1, kv_dv1, kv_ek1, kv_ev1,
        kv_dk2, kv_dv2, kv_ek2, kv_ev2,
        kv_dk3, kv_dv3, kv_ek3, kv_ev3,
      )
      1
    }
    None => -1
  }
}
#export_name("get_logit_f32")
pub fn get_logit_f32(idx : Int32) -> Float { logits_buf[idx] }
#export_name("get_kv_f32")
pub fn get_kv_f32(slot : Int32, idx : Int32) -> Float {
  match slot {
    0 => kv_dk0[idx]
    1 => kv_dv0[idx]
    2 => kv_ek0[idx]
    3 => kv_ev0[idx]
    4 => kv_dk1[idx]
    5 => kv_dv1[idx]
    6 => kv_ek1[idx]
    7 => kv_ev1[idx]
    8 => kv_dk2[idx]
    9 => kv_dv2[idx]
    10 => kv_ek2[idx]
    11 => kv_ev2[idx]
    12 => kv_dk3[idx]
    13 => kv_dv3[idx]
    14 => kv_ek3[idx]
    _ => kv_ev3[idx]
  }
}
`;
  } else {
    body = `let id_buf : FixedArray[Int64] = FixedArray::make(1, (0 : Int64))
let p_dk0 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_dv0 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_ek0 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_ev0 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_dk1 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_dv1 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_ek1 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_ev1 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_dk2 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_dv2 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_ek2 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_ev2 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_dk3 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_dv3 : FixedArray[Float] = FixedArray::make(1536, (0.0 : Float))
let p_ek3 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))
let p_ev3 : FixedArray[Float] = FixedArray::make(576000, (0.0 : Float))

let logits_buf : FixedArray[Float] = FixedArray::make(51865, (0.0 : Float))
let o_dk0 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dv0 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dk1 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dv1 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dk2 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dv2 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dk3 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))
let o_dv3 : FixedArray[Float] = FixedArray::make(1920, (0.0 : Float))

#export_name("set_id_i32")
pub fn set_id_i32(val : Int32) -> Unit { id_buf[0] = val.to_int64() }
#export_name("set_kv_f32")
pub fn set_kv_f32(slot : Int32, idx : Int32, val : Float) -> Unit {
  match slot {
    0 => p_dk0[idx] = val
    1 => p_dv0[idx] = val
    2 => p_ek0[idx] = val
    3 => p_ev0[idx] = val
    4 => p_dk1[idx] = val
    5 => p_dv1[idx] = val
    6 => p_ek1[idx] = val
    7 => p_ev1[idx] = val
    8 => p_dk2[idx] = val
    9 => p_dv2[idx] = val
    10 => p_ek2[idx] = val
    11 => p_ev2[idx] = val
    12 => p_dk3[idx] = val
    13 => p_dv3[idx] = val
    14 => p_ek3[idx] = val
    _ => p_ev3[idx] = val
  }
}
#export_name("run_infer")
pub fn run_infer() -> Int32 {
  match model_ref.val {
    Some(m) => {
      m.forward_into(
        id_buf,
        p_dk0, p_dv0, p_ek0, p_ev0,
        p_dk1, p_dv1, p_ek1, p_ev1,
        p_dk2, p_dv2, p_ek2, p_ev2,
        p_dk3, p_dv3, p_ek3, p_ev3,
        logits_buf,
        o_dk0, o_dv0, o_dk1, o_dv1, o_dk2, o_dv2, o_dk3, o_dv3,
      )
      1
    }
    None => -1
  }
}
#export_name("get_logit_f32")
pub fn get_logit_f32(idx : Int32) -> Float { logits_buf[idx] }
`;
  }
  fs.writeFileSync(path.join(dir, "entry.mbt"), wHeader + body);
}

// 3. 编译为 --release .wasm 并校验零警告
for (const t of buildTargets) {
  const dir = path.join(workDir, t.id);
  const out = runCmd("moon", ["build", "--release", "--target", t.buildTarget], dir);
  if (out.includes("Warning")) {
    throw new Error(`Unexpected compiler warning in ${t.id}:\n${out}`);
  }
}

async function loadWasmTarget(t) {
  const dir = path.join(workDir, t.id);
  const wasmPath = path.join(
    workDir,
    "_build",
    t.buildTarget,
    "release",
    "build",
    "whisper_aot",
    t.id,
    `${t.id}.wasm`
  );
  const wasmBytes = fs.readFileSync(wasmPath);
  const gzBytes = zlib.gzipSync(wasmBytes, { level: 9 });
  const tInst0 = performance.now();
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exp = instance.exports;
  const instMs = performance.now() - tInst0;

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
  if (exp.init_model() !== 1) throw new Error(`init_model failed for ${t.id}`);
  return { exp, wasmBytes, gzBytes, instMs, arenaKB: exp.get_arena_bytes() / 1024 };
}

console.log("=== Real OpenAI Whisper-tiny E2E Wasm AOT Benchmark (vs Official ORT 1.32.0) ===");

const melIn = readF32(path.join(whisperDir, "mel_input.f32"));
const encGold = readF32(path.join(whisperDir, "enc_fp32_golden.f32"));
const decPreGold = readF32(path.join(whisperDir, "dec_int8_prefill_logits.f32"));
const decStepGold = readF32(path.join(whisperDir, "dec_int8_step_logits.f32"));
const promptIds = [50258, 50259, 50359, 50363];

let sharedEncOut = null;
let sharedKvs = null;
let nextTok1 = -1;

// Step 1: Whisper-tiny Encoder (wasm V128 & wasm-gc)
for (const t of buildTargets.filter((x) => x.kind === "enc")) {
  const { exp, wasmBytes, gzBytes, instMs, arenaKB } = await loadWasmTarget(t);
  for (let i = 0; i < melIn.length; i++) exp.set_in_f32(i, melIn[i]);
  const t0 = performance.now();
  exp.run_infer();
  const runMs = performance.now() - t0;
  const out = new Float32Array(576000);
  for (let i = 0; i < 576000; i++) out[i] = exp.get_out_f32(i);
  const diff = maxAbsDiff(out, encGold);
  if (diff > 5e-3) throw new Error(`${t.id} mismatch: diff=${diff}`);
  if (!sharedEncOut) sharedEncOut = out;
  console.log(
    `  ${t.label.padEnd(36)} | wasm=${(wasmBytes.length / 1024).toFixed(1).padStart(5)} KB (gz ${(gzBytes.length / 1024).toFixed(1).padStart(4)} KB) | arena=${arenaKB.toFixed(1).padStart(8)} KB | inst=${instMs.toFixed(2)}ms | diff=${diff.toExponential(2)} | ${runMs.toFixed(1)} ms`
  );
}

// Step 2: Whisper-tiny Decoder INT8 Prefill (stage-isolated vs ORT + passing Wasm KV-Cache to Step 3)
{
  const t = buildTargets.find((x) => x.kind === "dec_pre");
  const { exp, wasmBytes, gzBytes, instMs, arenaKB } = await loadWasmTarget(t);
  for (let i = 0; i < 4; i++) exp.set_id_i32(i, promptIds[i]);
  for (let i = 0; i < 576000; i++) exp.set_enc_f32(i, encGold[i]);
  exp.run_infer();
  const t0 = performance.now();
  exp.run_infer();
  const runMs = performance.now() - t0;
  const logits = new Float32Array(207460);
  for (let i = 0; i < 207460; i++) logits[i] = exp.get_logit_f32(i);
  const diff = maxAbsDiff(logits, decPreGold);
  const lastLogits = logits.subarray(3 * 51865, 4 * 51865);
  const goldLast = decPreGold.subarray(3 * 51865, 4 * 51865);
  const pred = argmax(lastLogits);
  const goldPred = argmax(goldLast);
  if (pred.idx !== goldPred.idx) {
    throw new Error(`${t.id} token mismatch: ${pred.idx} vs ORT ${goldPred.idx}`);
  }
  nextTok1 = pred.idx;

  // 提取 16 个自注意力 + 交叉注意力 KV-Cache 张量传递给解码步进模型
  sharedKvs = Array.from({ length: 16 }, (_, slot) => {
    const len = slot % 4 < 2 ? 1536 : 576000;
    const arr = new Float32Array(len);
    for (let i = 0; i < len; i++) arr[i] = exp.get_kv_f32(slot, i);
    return arr;
  });

  console.log(
    `  ${t.label.padEnd(36)} | wasm=${(wasmBytes.length / 1024).toFixed(1).padStart(5)} KB (gz ${(gzBytes.length / 1024).toFixed(1).padStart(4)} KB) | arena=${arenaKB.toFixed(1).padStart(8)} KB | inst=${instMs.toFixed(2)}ms | diff=${diff.toExponential(2)} | tok1=${pred.idx} (ORT ${goldPred.idx}) | ${runMs.toFixed(1)} ms`
  );
}

// Step 3: Whisper-tiny Decoder INT8 Autoregressive KV-Cache Step (wasm V128 & wasm-gc)
for (const t of buildTargets.filter((x) => x.kind === "dec_step")) {
  const { exp, wasmBytes, gzBytes, instMs, arenaKB } = await loadWasmTarget(t);
  exp.set_id_i32(nextTok1);
  for (let slot = 0; slot < 16; slot++) {
    const arr = sharedKvs[slot];
    for (let i = 0; i < arr.length; i++) exp.set_kv_f32(slot, i, arr[i]);
  }
  exp.run_infer();
  const iters = 5;
  const t0 = performance.now();
  for (let r = 0; r < iters; r++) exp.run_infer();
  const avgMs = (performance.now() - t0) / iters;

  const logits = new Float32Array(51865);
  for (let i = 0; i < 51865; i++) logits[i] = exp.get_logit_f32(i);
  const diff = maxAbsDiff(logits, decStepGold);
  const pred = argmax(logits);
  const goldPred = argmax(decStepGold);
  if (pred.idx !== goldPred.idx) {
    throw new Error(`${t.id} token mismatch: ${pred.idx} vs ORT ${goldPred.idx}`);
  }
  console.log(
    `  ${t.label.padEnd(36)} | wasm=${(wasmBytes.length / 1024).toFixed(1).padStart(5)} KB (gz ${(gzBytes.length / 1024).toFixed(1).padStart(4)} KB) | arena=${arenaKB.toFixed(1).padStart(8)} KB | inst=${instMs.toFixed(2)}ms | diff=${diff.toExponential(2)} | tok2=${pred.idx} (ORT ${goldPred.idx}) | ${avgMs.toFixed(1)} ms/tok (${(1000 / avgMs).toFixed(1)} tok/s)`
  );
}
