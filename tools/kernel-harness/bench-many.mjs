// N pages at once, each running the quad-layout kernel on the two vision shapes: the all-core rate.
//   node bench-many.mjs <module.wasm> <pages>
import { readFileSync } from 'node:fs';

const { chromium } = await import('../../packages/e2e/node_modules/playwright/index.mjs');
const bytes = readFileSync(process.argv[2]).toString('base64');
const pages = Number(process.argv[3] ?? 1);
const stream = process.argv[4] === 'stream';
const browser = await chromium.launch({ headless: true, args: ['--enable-features=SharedArrayBuffer'] });

const run = async ([b64, stream]) => {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const { instance } = await WebAssembly.instantiate(raw);
  const e = instance.exports;
  const time = (fn) => { const t0 = performance.now(); fn(); return performance.now() - t0; };
  const mem = () => new Uint8Array(e.memory.buffer);
  let seed = 1;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const alloc = (n) => { const p = e.arena(n + 64); return (p + 63) & ~63; };
  const out = {};
  for (const [name, m, k, n] of [['mm_lanes_q4_r4', 48, 1152, 432], ['mm_lanes_q7_r4', 32, 96, 4640]]) {
    const blocks = k / 32;
    // stream: the activations cycle through 5.3 MB and the outputs through 64 MB, as a vision layer's do.
    const aSlots = stream ? Math.ceil(5.3e6 / (m * k)) : 1, outSlots = stream ? Math.ceil(64e6 / (m * n * 4)) : 1;
    const a = alloc(m * k * aSlots), aScales = alloc(m * blocks * 4), aCorr = alloc(m * blocks * 4);
    const w7 = alloc(n * blocks * 32), w4 = alloc(n * blocks * 16), wScales = alloc(n * blocks * 4);
    const outp = alloc(m * n * 4 * outSlots), params = alloc(9 * 4);
    const u8 = mem();
    for (let i = 0; i < m * k * aSlots; i++) u8[a + i] = (i * 7) & 0x7f;
    for (let i = 0; i < n * blocks * 32; i++) u8[w7 + i] = 1 + Math.floor(rnd() * 127);
    for (let i = 0; i < n * blocks * 16; i++) u8[w4 + i] = Math.floor(rnd() * 256);
    const f32 = new Float32Array(e.memory.buffer);
    const i32 = new Int32Array(e.memory.buffer);
    for (let i = 0; i < m * blocks; i++) { f32[aScales / 4 + i] = 0.01 + rnd() * 0.02; i32[aCorr / 4 + i] = -64 * Math.floor(rnd() * 200 - 100); }
    for (let i = 0; i < n * blocks; i++) f32[wScales / 4 + i] = 0.001 + rnd() * 0.01;
    const w = name.includes('q7') ? w7 : w4;
    const p = new Uint32Array(e.memory.buffer, params, 9);
    p.set([a, aScales, aCorr, w, wScales, outp, m, blocks, n]);
    let call = 0;
    const calls = (count) => { for (let i = 0; i < count; i++, call++) { p[0] = a + (call % aSlots) * m * k; p[5] = outp + (call % outSlots) * m * n * 4; e[name](params, 1); } };
    let iters = 100;
    for (;;) { if (time(() => calls(iters)) >= 200) break; iters *= 2; }
    let best = Infinity;
    for (let i = 0; i < 5; i++) best = Math.min(best, time(() => calls(iters)));
    out[`${name} ${m}x${k}x${n}`] = +((m * k * n * iters) / best / 1e6).toFixed(1);
  }
  // The real kernel through kernels::matmul, as the page calls it.
  for (const [m, k, n, bits] of [[48, 1152, 432, 4], [32, 96, 4640, 7], [32, 4640, 72, 7]]) {
    const slot = e.real_prepare(m, k, n, bits);
    for (const store of ['', '_scaled', '_scaled_map']) {
      const fn = e['real_matmul' + store];
      let iters = 100;
      for (;;) { if (time(() => fn(slot, iters)) >= 200) break; iters *= 2; }
      let best = Infinity;
      for (let i = 0; i < 5; i++) best = Math.min(best, time(() => fn(slot, iters)));
      out[`real${store || '_copy'} q${bits} ${m}x${k}x${n}`] = +((m * k * n * iters) / best / 1e6).toFixed(1);
    }
  }
  return out;
};

const results = await Promise.all(Array.from({ length: pages }, async () => {
  const page = await browser.newPage();
  return page.evaluate(run, [bytes, stream]);
}));
await browser.close();
for (const key of Object.keys(results[0])) {
  const rates = results.map((r) => r[key]).sort((a, b) => a - b);
  console.log(`${pages} pages ${stream ? 'stream' : 'hot   '} ${key}: GMAC/s per page ${rates.join(' ')}`);
}
