// Runs the microbench wasm in headless Chrome (the e2e's Playwright) and prints JSON.
//   node bench.mjs <module.wasm> [tier|dots|mm|exp ...]
import { readFileSync } from 'node:fs';

const { chromium } = await import('../../packages/e2e/node_modules/playwright/index.mjs');
const bytes = readFileSync(process.argv[2]).toString('base64');
const which = process.argv.slice(3);
const flags = process.env.JS_FLAGS ? [`--js-flags=${process.env.JS_FLAGS}`] : [];
const browser = await chromium.launch({ headless: true, args: flags });
const page = await browser.newPage();
page.on('console', (m) => console.error('console:', m.text()));
const attnCount = Number(process.env.COUNT ?? 1200);
const result = await page.evaluate(async ([b64, which, attnCount]) => {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const { instance } = await WebAssembly.instantiate(raw);
  const e = instance.exports;
  const want = (name) => which.length === 0 || which.includes(name);
  const time = (fn) => { const t0 = performance.now(); fn(); return performance.now() - t0; };
  const out = {};

  if (want('tier')) {
    const calls = [];
    for (let i = 0; i < 300; i++) calls.push(time(() => e.tier_work(2_000_000, i)));
    const once = time(() => e.tier_work(300 * 2_000_000, 7));
    const sorted = calls.slice(200).sort((a, b) => a - b);
    out.tier = { first_calls_ms: calls.slice(0, 12).map((v) => +v.toFixed(3)), steady_ms: sorted[sorted.length >> 1], one_call_300x_ms: once, calls_total_ms: calls.reduce((a, b) => a + b, 0) };
  }

  // Runs fn(iters) with growing iters until a call takes >= 80 ms, then takes the best of 5.
  const steady = (fn) => {
    let iters = 1000;
    for (;;) {
      const ms = time(() => fn(iters));
      if (ms >= 80) break;
      iters *= 2;
    }
    let best = Infinity;
    for (let i = 0; i < 5; i++) best = Math.min(best, time(() => fn(iters)));
    return { iters, ms: best };
  };

  if (want('dots')) {
    out.dots = {};
    for (const name of ['dot_regs', 'dot_regs_i16']) {
      const { iters, ms } = steady((n) => e[name](n, 12345));
      // 8 dot instructions per iteration, 16 MACs each
      out.dots[name] = { gmac_per_s: (iters * 8 * 16) / ms / 1e6, ns_per_dot: (ms * 1e6) / (iters * 8) };
    }
  }

  const mem = () => new Uint8Array(e.memory.buffer);
  let seed = 1;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const alloc = (n) => { const p = e.arena(n + 64); return (p + 63) & ~63; };

  if (want('mm')) {
    out.mm = [];
    const configs = [[48, 1152, 432], [48, 4320, 144], [48, 96, 4640]];
    for (const [m, k, n] of configs) {
      const blocks = k / 32;
      const a = alloc(m * k), aScales = alloc(m * blocks * 4), aCorr = alloc(m * blocks * 4);
      const w7 = alloc(n * blocks * 32), w4 = alloc(n * blocks * 16), wScales = alloc(n * blocks * 4);
      const outp = alloc(m * n * 4), params = alloc(9 * 4);
      const u8 = mem();
      for (let i = 0; i < m * k; i++) u8[a + i] = (Math.floor(rnd() * 255) - 127) & 0xff;
      for (let i = 0; i < n * blocks * 32; i++) u8[w7 + i] = 1 + Math.floor(rnd() * 127);
      for (let i = 0; i < n * blocks * 16; i++) u8[w4 + i] = Math.floor(rnd() * 256);
      const f32 = new Float32Array(e.memory.buffer);
      const i32 = new Int32Array(e.memory.buffer);
      for (let i = 0; i < m * blocks; i++) { f32[aScales / 4 + i] = 0.01 + rnd() * 0.02; i32[aCorr / 4 + i] = -64 * Math.floor(rnd() * 200 - 100); }
      for (let i = 0; i < n * blocks; i++) f32[wScales / 4 + i] = 0.001 + rnd() * 0.01;
      const variants = ['mm_1x2_q7', 'mm_1x2_q4', 'mm_4x2_q7', 'mm_4x2_q7_madd', 'mm_4x4_q7', 'mm_4x4_q7_rowscale', 'mm_4x4_q7_int', 'mm_4x2_q4', 'mm_4x2_q4_madd', 'mm_4x4_q4', 'mm_4x4_q4_rowscale', 'mm_4x4_q4_int', 'mm_lanes_q7_r1g1', 'mm_lanes_q7_r1g2', 'mm_lanes_q4_r1g1', 'mm_lanes_q7_r1', 'mm_lanes_q7_r2', 'mm_lanes_q7_r4', 'mm_lanes_q4_r1', 'mm_lanes_q4_r2', 'mm_lanes_q4_r4', 'mm_lanes_q7_r6', 'mm_lanes_q7_r8', 'mm_lanes_q4_r6', 'mm_lanes_q4_r8', 'mm_lanes_q4_r4g2', 'mm_lanes_q4_r8g2'];
      const row = { m, k, n, variants: {} };
      for (const name of variants) {
        if (name.includes('lanes') && n % 16 !== 0) continue;
        if ((name.includes('4x4') || name.includes('4x2') || name.includes('_r4')) && m % 4 !== 0) continue;
        if (m > 1 && n > 20000) continue;
        if (name.includes('1x2') && m !== 1) continue;
        if (name.includes('_r2') && m % 2 !== 0) continue;
        if (name.includes('_r6') && m % 6 !== 0) continue;
        if (name.includes('_r8') && m % 8 !== 0) continue;
        if (name.includes('g1') || name.includes('g2')) { if (m !== 1) continue; }
        const w = name.includes('q7') ? w7 : w4;
        new Uint32Array(e.memory.buffer, params, 9).set([a, aScales, aCorr, w, wScales, outp, m, blocks, n]);
        const { iters, ms } = steady((r) => e[name](params, r));
        const macs = m * k * n * iters;
        row.variants[name] = { gmac_per_s: +(macs / ms / 1e6).toFixed(1), weight_gb_per_s: +((n * k * (name.includes('q7') ? 1 : 0.5) * iters) / ms / 1e6).toFixed(1), ns_per_block_iter: +((ms * 1e6) / (iters * (m / 4) * (n / 2) * blocks)).toFixed(2) };
      }
      out.mm.push(row);
    }
  }

  if (want('attn')) {
    const [count, capacity, head] = [attnCount, 3360, 128];
    const q = alloc(head * 4), keys = alloc(capacity * head * 4), values = alloc(capacity * head * 4), scores = alloc(count * 4), outp = alloc(head * 4);
    const f32 = new Float32Array(e.memory.buffer);
    for (let i = 0; i < head; i++) f32[q / 4 + i] = rnd() - 0.5;
    for (let i = 0; i < capacity * head; i++) { f32[keys / 4 + i] = rnd() - 0.5; f32[values / 4 + i] = rnd() - 0.5; }
    out.attn = {};
    const variants = {
      real_attend: (r) => e.real_attend(q, keys, values, count, capacity, 0, scores, outp, r),
      attend_dyn_real: (r) => e.attend_dyn_real(q, keys, values, count, capacity, head, scores, outp, r),
      attend_dyn_newsoftmax_oldpv: (r) => e.attend_dyn_newsoftmax_oldpv(q, keys, values, count, capacity, head, scores, outp, r),
      attend_dyn_oldsoftmax_newpv: (r) => e.attend_dyn_oldsoftmax_newpv(q, keys, values, count, capacity, head, scores, outp, r),
      attend_closure: (r) => e.attend_closure(q, keys, values, count, capacity, 0, scores, outp, r),
      attend_strided: (r) => e.attend_strided(q, keys, values, count, scores, outp, r),
      attend_blocked: (r) => e.attend_blocked(q, keys, values, count, scores, outp, r),
    };
    for (const [name, fn] of Object.entries(variants)) {
      const { iters, ms } = steady(fn);
      fn(1);
      const sample = Array.from(new Float32Array(e.memory.buffer, outp, 4)).map((v) => +v.toFixed(6));
      out.attn[name] = { ns_per_key: +((ms * 1e6) / (iters * count)).toFixed(1), sample };
    }
  }

  if (want('real')) {
    out.real = [];
    for (const [m, k, n, bits] of [[32, 1152, 432, 4], [32, 1152, 432, 7], [4, 1152, 432, 4], [32, 96, 4640, 7], [32, 4640, 72, 7], [1, 1024, 12928, 7]]) {
      const slot = e.real_prepare(m, k, n, bits);
      const { iters, ms } = steady((r) => e.real_matmul(slot, r));
      const bare = steady((r) => e.real_matmul_nostore(slot, r));
      out.real.push({ m, k, n, bits, gmac_per_s: +((m * k * n * iters) / ms / 1e6).toFixed(1), gmac_per_s_nostore: +((m * k * n * bare.iters) / bare.ms / 1e6).toFixed(1) });
    }
  }

  if (want('exp')) {
    const n = 4640;
    const scores = alloc(n * 4), outp = alloc(n * 4);
    const f32 = new Float32Array(e.memory.buffer);
    for (let i = 0; i < n; i++) f32[scores / 4 + i] = rnd() * 30 - 20;
    out.exp = {};
    for (const name of ['exponentials_ref', 'exponentials_simd', 'exponentials_simd_scalarmax']) {
      const { iters, ms } = steady((r) => e[name](scores, outp, n, r));
      const sum = e[name](scores, outp, n, 1);
      out.exp[name] = { gexp_per_s: +((n * iters) / ms / 1e6).toFixed(3), ns_per_exp: +((ms * 1e6) / (n * iters)).toFixed(3), sum };
    }
  }
  return out;
}, [bytes, which, attnCount]);
await browser.close();
console.log(JSON.stringify(result, null, 2));
