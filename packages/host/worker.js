// The shared half of a model's engine worker: the module Worker that loads the model and runs it
// off the page's main thread. A product's worker imports this beside the loader and keeps only
// its own message handlers.
//
// `load` overlaps the manifest download with the runtime's start, streams every part into the
// runtime (from OPFS when cached there, else from the network), and afterwards deletes the cached
// parts of earlier bundles. `serve` routes the page's messages to the handlers and reports their
// failures. After the model is loaded nothing else should be fetched: `requestsSinceReady` counts
// what was, for the page to show.

import { describe, errorKind, fetchOk, loadParts, startRuntime } from './loader.js';

export const post = (message, transfer) => self.postMessage(message, transfer ?? []);

// A macrotask without setTimeout's 4 ms clamp on nested timers. Resolvers queue, so overlapping
// awaits each resume.
const ticks = new MessageChannel();
const waiting = [];
ticks.port1.onmessage = () => waiting.shift()();
export const yieldTask = () => new Promise((resolve) => {
  waiting.push(resolve);
  ticks.port2.postMessage(null);
});

/** The loaded runtime, `{ rt, glue }`: null until `load` created it. */
export let engine = null;
let resolveReady;
/** Resolves once every part is in and the model is built. */
export const ready = new Promise((resolve) => (resolveReady = resolve));
let readyAt = Infinity;

/** The requests this worker made after the model was ready. */
export const requestsSinceReady = () => performance.getEntriesByType('resource').filter((entry) => entry.startTime >= readyAt).length;

/**
 * Loads the bundle at `modelUrl` into the best runtime build of `glueName` under `distUrl`.
 * `onRuntime(started)` runs once the runtime exists, `onReady({ ms, ...progress })` once the model
 * is built; `progress` and `warning` messages are posted on the way.
 */
export async function load({ modelUrl, threads, build, glueName, cacheName, distUrl, onRuntime, onReady }) {
  const t0 = performance.now();
  const [manifestText, started] = await Promise.all([
    fetchOk(new URL('manifest.json', modelUrl)).then((response) => response.text()),
    startRuntime({ distUrl, glueName, threads, requested: build, onWarning: (message) => post({ type: 'warning', message }) }),
  ]);
  const manifest = JSON.parse(manifestText);
  engine = { rt: new started.glue.Runtime(manifestText), glue: started.glue };
  onRuntime(started);
  const progress = await loadParts({
    runtime: engine.rt,
    manifest,
    baseUrl: modelUrl,
    cacheName,
    onProgress: (p) => post({ type: 'progress', ...p }),
  });
  readyAt = performance.now();
  onReady({ ms: readyAt - t0, ...progress });
  resolveReady();
  await evictStaleParts(manifest, cacheName);
}

/** Deletes cached parts this bundle does not name: those of earlier bundles. */
async function evictStaleParts(manifest, cacheName) {
  const keep = new Set(manifest.parts.map((part) => part.blake3));
  let dir;
  try {
    dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(cacheName);
  } catch {
    return; // no OPFS, or nothing was cached
  }
  for await (const name of dir.keys()) {
    if (!keep.has(name)) await dir.removeEntry(name);
  }
}

/**
 * Posts a failure of `job`. A runtime error is `{"kind","message"}` JSON and leaves the engine
 * usable: it is posted as `error`. Anything else is a trap (the runtimes abort on panic) or a bug
 * that lost the module's state: it is posted as `fatal`.
 */
export function report(error, job) {
  post({ type: errorKind(error) ? 'error' : 'fatal', job, message: describe(error, engine?.glue) });
}

/** Routes each page message to `handlers[type]`, and reports what fails. */
export function serve(handlers) {
  self.onmessage = async ({ data }) => {
    try {
      await handlers[data.type]?.(data);
    } catch (error) {
      report(error, data.job);
    }
  };
}
