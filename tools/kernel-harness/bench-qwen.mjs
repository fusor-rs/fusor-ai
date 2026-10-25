// Qwen3.5's kernels in headless Chrome (the e2e's Playwright): residual stores, matmul rows,
// the GELU store, quantization, exponentials, the delta rule's step,
// convolution and the lane kernel's pass width. Prints JSON.
//   node bench-qwen.mjs <module.wasm> [residual|rows|gelu|quantize|quant2|exp|delta|conv|pass ...]
import { readFileSync } from 'node:fs';

const { chromium } = await import('../../packages/e2e/node_modules/playwright/index.mjs');
const bytes = readFileSync(process.argv[2]).toString('base64');
const which = process.argv.slice(3);
const flags = process.env.JS_FLAGS ? [`--js-flags=${process.env.JS_FLAGS}`] : [];
const browser = await chromium.launch({ headless: true, args: ['--enable-features=SharedArrayBuffer', ...flags] });
const page = await browser.newPage();
page.on('console', (m) => console.error('console:', m.text()));
const result = await page.evaluate(async ([b64, which]) => {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const { instance } = await WebAssembly.instantiate(raw);
  const e = instance.exports;
  const want = (name) => which.length === 0 || which.includes(name);
  const time = (fn) => { const t0 = performance.now(); fn(); return performance.now() - t0; };
  const out = {};

  // Runs fn(iters) with growing iters until a call takes >= 80 ms, then takes the best of 5.
  const steady = (fn) => {
    let iters = 1;
    for (;;) {
      const ms = time(() => fn(iters));
      if (ms >= 80) break;
      iters *= 2;
    }
    let best = Infinity;
    for (let i = 0; i < 5; i++) best = Math.min(best, time(() => fn(iters)));
    return { iters, ms: best };
  };
  const gmac = (m, k, n, fn) => { const { iters, ms } = steady(fn); return +((m * k * n * iters) / ms / 1e6).toFixed(1); };
  const ns = (count, fn) => { const { iters, ms } = steady(fn); return +((ms * 1e6) / (iters * count)).toFixed(3); };
  const bits = (values) => Array.from(new Uint32Array(Float32Array.from(values).buffer));
  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

  let seed = 1;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const alloc = (n) => { const p = e.arena(n + 64); return (p + 63) & ~63; };

  if (want('residual')) {
    out.residual = [];
    const variants = ['real_matmul_residual_scalar', 'real_matmul_residual_array', 'real_matmul_residual_simd'];
    for (const [m, k, n, bits_] of [[512, 2048, 256, 7], [512, 6144, 256, 7], [512, 1024, 128, 7]]) {
      const row = { m, k, n, bits: bits_, gmac_per_s: {} };
      if (!which.includes('identity')) {
        const slot = e.real_prepare(m, k, n, bits_);
        for (const name of ['real_matmul', ...variants]) row.gmac_per_s[name] = gmac(m, k, n, (r) => e[name](slot, r));
      }
      const outs = variants.map((name) => {
        const slot = e.real_prepare(m, k, n, bits_);
        e[name](slot, 1);
        return Array.from(new Uint32Array(e.memory.buffer, e.real_out_ptr(slot), 8));
      });
      row.out_bits = outs[0];
      row.identical = outs.every((o) => same(o, outs[0]));
      out.residual.push(row);
    }
  }

  if (want('rows')) {
    out.rows = [];
    for (const [k, n, bits_] of [[2048, 768, 7], [6144, 256, 7]]) {
      for (const m of [64, 128, 256, 384, 512]) {
        const slot = e.real_prepare(m, k, n, bits_);
        out.rows.push({ m, k, n, bits: bits_, gmac_per_s: gmac(m, k, n, (r) => e.real_matmul(slot, r)) });
      }
    }
  }

  if (want('gelu')) {
    const [m, k, n] = [3472, 1024, 512];
    out.gelu = { m, k, n, bits: 7, group: 32, gmac_per_s: {} };
    if (!which.includes('identity')) {
      const slot = e.real_prepare(m, k, n, 7);
      for (const name of ['real_matmul', 'real_matmul_gelu', 'real_matmul_gelu_scalar']) out.gelu.gmac_per_s[name] = gmac(m, k, n, (r) => e[name](slot, r));
    }
    const outs = ['real_matmul_gelu', 'real_matmul_gelu_scalar'].map((name) => {
      const slot = e.real_prepare(m, k, n, 7);
      e[name](slot, 1);
      return Array.from(new Uint32Array(e.memory.buffer, e.real_out_ptr(slot), 8));
    });
    out.gelu.out_bits = outs[0];
    out.gelu.identical = same(outs[0], outs[1]);
  }

  if (want('quantize')) {
    out.quantize = [];
    for (const [rows, width, perRow] of [[3472, 1024, 0], [3472, 1024, 1], [32, 3488, 0]]) {
      out.quantize.push({ rows, width, per_row: perRow, ns_per_value: ns(rows * width, (r) => e.quantize_bench(rows, width, perRow, r)) });
    }
  }

  if (want('quant2')) {
    out.quant2 = [];
    const replicas = ['quantize_replica_inline', 'quantize_replica_group_noinline', 'quantize_replica_fused', 'quantize_replica_fused_noinline'];
    for (const [rows, width] of [[3472, 1024], [32, 3488]]) {
      const row = { rows, width, ns_per_value: {} };
      if (!which.includes('identity')) {
        row.ns_per_value.quantize_bench = ns(rows * width, (r) => e.quantize_bench(rows, width, 0, r));
        for (const name of replicas) row.ns_per_value[name] = ns(rows * width, (r) => e[name](rows, width, r));
      }
      const mask = e.quantize_check(rows, width);
      row.identical = replicas.map((_, i) => (mask & (1 << i)) !== 0);
      out.quant2.push(row);
    }
  }

  if (want('exp')) {
    const n = 3472;
    const scores = alloc(n * 4), outp = alloc(n * 4);
    const f32 = new Float32Array(e.memory.buffer);
    for (let i = 0; i < n; i++) f32[scores / 4 + i] = rnd() * 30 - 20;
    out.exp = { ns_per_exp: {}, max_rel_error: {}, identity: [] };
    const variants = [
      'exponentials_real', 'exponentials_simd', 'exponentials_two_runs',
      'exponentials_noclamp', 'exponentials_madd', 'exponentials_fast4', 'exponentials_fast3',
    ];
    if (!which.includes('identity')) {
      for (const name of variants) out.exp.ns_per_exp[name] = ns(n, (r) => e[name](scores, outp, n, r));
    }
    const inputs = Array.from(new Float32Array(e.memory.buffer, scores, n));
    const max = Math.max(...inputs);
    for (const name of variants) {
      e[name](scores, outp, n, 1);
      const got = new Float32Array(e.memory.buffer, outp, n);
      out.exp.max_rel_error[name] = inputs.reduce((worst, score, i) => {
        const want = Math.exp(score - max);
        return Math.max(worst, Math.abs(got[i] - want) / want);
      }, 0);
    }
    for (const [reference, variant] of [['exponentials_real', 'exponentials_two_runs'], ['exponentials_two_runs', 'exponentials_noclamp']]) {
      for (const count of [3472, 37]) {
        const runs = [reference, variant].map((name) => {
          const sum = e[name](scores, outp, count, 1);
          return { sum: bits([sum])[0], out: Array.from(new Uint32Array(e.memory.buffer, outp, count)) };
        });
        const [first, second] = runs;
        out.exp.identity.push({ n: count, variants: [reference, variant], sum_bits: [first.sum, second.sum], identical: first.sum === second.sum && same(first.out, second.out) });
      }
    }
  }

  if (want('delta')) {
    const variants = ['delta_step_ref', 'delta_step_many', 'delta_step_many2'];
    out.delta = { ns_per_step: {} };
    if (!which.includes('identity')) {
      for (const name of ['delta_step_ref', 'delta_step_many2', 'delta_step_many']) {
        e.delta_reset(1);
        out.delta.ns_per_step[name] = ns(1, (r) => e[name](r));
      }
    }
    const checksums = variants.map((name, variant) => {
      e.delta_reset(1);
      e[name](5);
      return e.delta_checksum(variant) >>> 0;
    });
    out.delta.checksums = Object.fromEntries(variants.map((name, i) => [name, checksums[i]]));
    out.delta.identical = checksums.every((c) => c === checksums[0]);
  }

  if (want('conv')) {
    const variants = ['conv_ref', 'conv_straight'];
    out.conv = { ns_per_row: {} };
    if (!which.includes('identity')) {
      for (const name of variants) out.conv.ns_per_row[name] = ns(1, (r) => e[name](r));
    }
    const checksums = variants.map((name, variant) => {
      e[name](1);
      return e.conv_checksum(variant) >>> 0;
    });
    out.conv.checksums = Object.fromEntries(variants.map((name, i) => [name, checksums[i]]));
    out.conv.identical = checksums.every((c) => c === checksums[0]);
  }

  if (want('pass')) {
    out.pass = [];
    // The rowscale and int variants are hypothetical scalings: their outputs differ by construction.
    const variants = {
      mm_lanes_q7_r4: 4, mm_lanes_q7_r4g5: 5, mm_lanes_q7_r4g6: 6, mm_lanes_q7_r4g8: 8,
      mm_lanes_q7_r4_rowscale: 4, mm_lanes_q7_r4_int: 4,
    };
    for (const [m, k, n] of [[512, 2048, 768], [3472, 1024, 512]]) {
      const blocks = k / 32;
      const a = alloc(m * k), aScales = alloc(m * blocks * 4), aCorr = alloc(m * blocks * 4);
      const w = alloc(n * blocks * 32), wScales = alloc(n * blocks * 4);
      const outp = alloc(m * n * 4), params = alloc(9 * 4);
      const u8 = new Uint8Array(e.memory.buffer);
      for (let i = 0; i < m * k; i++) u8[a + i] = (Math.floor(rnd() * 255) - 127) & 0xff;
      for (let i = 0; i < n * blocks * 32; i++) u8[w + i] = 1 + Math.floor(rnd() * 127);
      const f32 = new Float32Array(e.memory.buffer);
      const i32 = new Int32Array(e.memory.buffer);
      for (let i = 0; i < m * blocks; i++) { f32[aScales / 4 + i] = 0.01 + rnd() * 0.02; i32[aCorr / 4 + i] = -64 * Math.floor(rnd() * 200 - 100); }
      for (let i = 0; i < n * blocks; i++) f32[wScales / 4 + i] = 0.001 + rnd() * 0.01;
      new Uint32Array(e.memory.buffer, params, 9).set([a, aScales, aCorr, w, wScales, outp, m, blocks, n]);
      const row = { m, k, n, gmac_per_s: {}, skipped: {}, out_bits: {} };
      for (const [name, groups] of Object.entries(variants)) {
        if (n % (4 * groups) !== 0) {
          row.skipped[name] = `n = ${n} is not a multiple of ${4 * groups} (4 columns × ${groups} groups per pass)`;
          continue;
        }
        if (!which.includes('identity')) row.gmac_per_s[name] = gmac(m, k, n, (r) => e[name](params, r));
        new Float32Array(e.memory.buffer, outp, m * n).fill(0);
        e[name](params, 1);
        row.out_bits[name] = Array.from(new Uint32Array(e.memory.buffer, outp, 8));
      }
      const outs = Object.values(row.out_bits);
      row.identical = outs.every((o) => same(o, outs[0]));
      out.pass.push(row);
    }
  }
  return out;
}, [bytes, which]);
await browser.close();
console.log(JSON.stringify(result, null, 2));
