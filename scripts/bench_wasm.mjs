#!/usr/bin/env node
// 在 Node.js V8 引擎中实测 moon-onnx 的 wasm 与 wasm-gc V128 SIMD 性能与产物体积。
import fs from "node:fs";
import zlib from "node:zlib";
import { performance } from "node:perf_hooks";

async function runSuite(label, wasmPath) {
  const bytes = fs.readFileSync(wasmPath);
  const gzBytes = zlib.gzipSync(bytes);
  const tCompile0 = performance.now();
  const { instance } = await WebAssembly.instantiate(bytes, {
    spectest: { print_char: () => {} },
  });
  const compileMs = performance.now() - tCompile0;
  const ex = instance.exports;

  console.log(
    `\n=== ${label} (raw=${(bytes.length / 1024).toFixed(1)} KB, gzip=${(
      gzBytes.length / 1024
    ).toFixed(1)} KB, instantiate=${compileMs.toFixed(2)} ms) ===`
  );

  const pairs = [
    ["GEMM 64x64x64 (f32)", "bench_gemm64_scalar", "bench_gemm64_simd", 200],
    ["GEMM 128x128x128 (f32)", "bench_gemm128_scalar", "bench_gemm128_simd", 30],
    ["QLinearMatMul 128x128x128 (int8)", "bench_qgemm128_i8_scalar", "bench_qgemm128_i8_simd", 30],
    ["Conv2D 1x8x14x14->16 k=5 +Relu", "bench_conv2d_scalar", "bench_conv2d_simd", 80],
    ["Depthwise 1x32x14x14 k=3 +ReLU6", "bench_depthwise_scalar", "bench_depthwise_simd", 100],
    ["Relu n=65536 (byte V128 pmax)", "bench_relu65k_scalar", "bench_relu65k_simd_bytes", 300],
  ];

  for (const [name, fnScalar, fnSimd, iters] of pairs) {
    // 预热 V8 TurboFan JIT
    ex[fnScalar](5);
    ex[fnSimd](5);

    const t0 = performance.now();
    ex[fnScalar](iters);
    const scalarUs = ((performance.now() - t0) * 1000) / iters;

    const t1 = performance.now();
    ex[fnSimd](iters);
    const simdUs = ((performance.now() - t1) * 1000) / iters;

    const speedup = scalarUs / simdUs;
    console.log(
      `  ${name.padEnd(34)} scalar=${scalarUs
        .toFixed(1)
        .padStart(8)} us | simd=${simdUs
        .toFixed(1)
        .padStart(8)} us | speedup=${speedup.toFixed(2)}x`
    );
  }
}

await runSuite(
  "Wasm (linear memory, --release)",
  "_build/wasm/release/build/matrixlt/moon_onnx_wasm_bench/moon_onnx_wasm_bench.wasm"
);
await runSuite(
  "Wasm-GC (--release)",
  "_build/wasm-gc/release/build/matrixlt/moon_onnx_wasm_bench/moon_onnx_wasm_bench.wasm"
);
