// Tier-up latency of a matmul tile, call by call, on the main thread, in one worker and in
// N concurrent workers that received the compiled module by postMessage (as the runtime's
// compute workers do). JS_FLAGS=... to pass V8 flags.
//   node bench-worker.mjs <module.wasm> [workers]
import { readFileSync } from 'node:fs';

const { chromium } = await import('../../packages/e2e/node_modules/playwright/index.mjs');
const bytes = readFileSync(process.argv[2]).toString('base64');
const workers = Number(process.argv[3] ?? 8);
const flags = process.env.JS_FLAGS ? [`--js-flags=${process.env.JS_FLAGS}`] : [];
const browser = await chromium.launch({ headless: true, args: flags });
const page = await browser.newPage();
page.on('console', (m) => console.error('console:', m.text()));

// Runs in the page or in a worker: times `calls` calls of mm_4x2_q7 on a 32x1152x432 problem.
const probeSource = `
  async function probe(module, calls) {
    const instance = module instanceof WebAssembly.Module ? await WebAssembly.instantiate(module) : (await WebAssembly.instantiate(module)).instance;
    const e = instance.exports;
    const [m, k, n] = [32, 1152, 432];
    const blocks = k / 32;
    let seed = 1;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
    const alloc = (bytes) => { const p = e.arena(bytes + 64); return (p + 63) & ~63; };
    const a = alloc(m * k), aScales = alloc(m * blocks * 4), aCorr = alloc(m * blocks * 4);
    const w = alloc(n * blocks * 32), wScales = alloc(n * blocks * 4), out = alloc(m * n * 4), params = alloc(36);
    const u8 = new Uint8Array(e.memory.buffer);
    for (let i = 0; i < m * k; i++) u8[a + i] = (Math.floor(rnd() * 255) - 127) & 0xff;
    for (let i = 0; i < n * blocks * 32; i++) u8[w + i] = 1 + Math.floor(rnd() * 127);
    const f32 = new Float32Array(e.memory.buffer);
    for (let i = 0; i < m * blocks; i++) f32[aScales / 4 + i] = 0.01;
    for (let i = 0; i < n * blocks; i++) f32[wScales / 4 + i] = 0.01;
    new Uint32Array(e.memory.buffer, params, 9).set([a, aScales, aCorr, w, wScales, out, m, blocks, n]);
    const times = [];
    const t0 = performance.now();
    for (let i = 0; i < calls; i++) {
      const t = performance.now();
      e.mm_4x2_q7(params, 1);
      times.push(performance.now() - t);
    }
    const total = performance.now() - t0;
    const sorted = times.slice(calls >> 1).sort((x, y) => x - y);
    const steady = sorted[sorted.length >> 1];
    const fastFrom = times.findIndex((t, i) => i > 2 && times.slice(i, i + 5).every((v) => v < steady * 1.3));
    return { first: times.slice(0, 8).map((v) => +v.toFixed(3)), steady_ms: +steady.toFixed(3), fast_from_call: fastFrom, total_ms: +total.toFixed(1), gmac_steady: +((m * k * n) / steady / 1e6).toFixed(1) };
  }
`;

const result = await page.evaluate(async ([b64, probeSource, workers]) => {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const module = await WebAssembly.compile(raw);
  const probe = new Function(`${probeSource}; return probe;`)();
  const out = {};
  out.main = await probe(module, 400);
  const workerSource = `${probeSource}; onmessage = async ({ data }) => { postMessage(await probe(data.module, data.calls)); };`;
  const url = URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' }));
  const runWorker = (calls) => new Promise((resolve) => {
    const worker = new Worker(url);
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(data); };
    worker.postMessage({ module, calls });
  });
  out.one_worker = await runWorker(400);
  out.concurrent = await Promise.all(Array.from({ length: workers }, () => runWorker(400)));
  return out;
}, [bytes, probeSource, workers]);
await browser.close();
console.log(JSON.stringify(result, null, 2));
