// Starting a fusor-ai runtime in a module Worker and streaming a bundle into it.
//
// `startRuntime` picks the best runtime build this browser runs (threads need a cross-origin
// isolated page, relaxed SIMD is probed), instantiates it and, for threaded builds, starts its
// compute pool; when threads cannot be set up it falls back to a single-threaded build.
//
// `loadParts` streams every part of a bundle into the runtime — from OPFS when cached there,
// else from the network while caching it — through the runtime's part protocol:
// `write_part(index, bytes)` in order, `end_part(index)` (which checks size and BLAKE3 and
// throws `{"kind","message"}` JSON), and `restart_part(index)`. Only an `integrity` failure
// evicts a cached part; other errors are reported as they are.

const PARALLEL_PARTS = 4;
const PUSH_BYTES = 4 << 20; // coalesce network chunks into writes of this size

// Relaxed SIMD: a function using `i32x4.relaxed_dot_i8x16_i7x16_add_s` (0xfd 0x93 0x02), the
// instruction the relaxed kernels are built on. Engines without the proposal reject it.
const RELAXED_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 19, 1, 17, 0, 65, 1, 253, 15, 65, 2, 253, 15, 65, 3, 253, 15, 253, 147, 2, 11]);

// Runtime builds, best first (`scripts/build-runtime.sh` builds each into `<dist>/<name>/`).
export const BUILDS = [
  { name: 'web-threaded-relaxed', threads: true, relaxed: true },
  { name: 'web-threaded', threads: true, relaxed: false },
  { name: 'web-single-relaxed', threads: false, relaxed: true },
  { name: 'web-single', threads: false, relaxed: false },
];

/** A runtime's `{"kind","message"}` error, or null for any other error. */
function runtimeError(error) {
  try {
    return JSON.parse(String(error?.message ?? error));
  } catch {
    return null; // not thrown by the runtime as JSON
  }
}

/** The `kind` of a runtime error (`format`, `integrity`, …), or null for other errors. */
export function errorKind(error) {
  return runtimeError(error)?.kind ?? null;
}

/** A runtime error's message, or the runtime's last Rust panic behind a wasm trap. */
export function describe(error, glue) {
  const runtime = runtimeError(error);
  if (runtime) return runtime.message;
  const text = String(error?.message ?? error);
  const panic = glue?.last_panic?.();
  return panic ? `${text} (panic: ${panic})` : text;
}

/** The best build this browser runs, or `requested` (a `BUILDS` name) when it can run that. */
export function chooseBuild(threads, requested) {
  const threaded = self.crossOriginIsolated === true && threads > 1;
  const relaxed = WebAssembly.validate(RELAXED_PROBE);
  const usable = BUILDS.filter((b) => (!b.threads || threaded) && (!b.relaxed || relaxed));
  return usable.find((b) => b.name === requested) ?? usable[0];
}

/**
 * Imports and instantiates `<distUrl>/<build>/<glueName>.js`. Returns `{ glue, build, threads }`.
 * `onWarning(message)` hears about a fallback to one thread.
 */
export async function startRuntime({ distUrl, glueName, threads, requested, onWarning }) {
  const build = chooseBuild(threads, requested);
  try {
    return { glue: await instantiate(distUrl, glueName, build, threads), build, threads: build.threads ? threads : 1 };
  } catch (error) {
    // The shared memory or the compute workers could not be set up: one thread still works.
    if (!build.threads) throw error;
    onWarning?.(`threads unavailable (${describe(error)}); running on one thread`);
    const single = chooseBuild(1);
    return { glue: await instantiate(distUrl, glueName, single, 1), build: single, threads: 1 };
  }
}

async function instantiate(distUrl, glueName, build, threads) {
  const base = new URL(`${build.name}/`, distUrl);
  const glue = await import(new URL(`${glueName}.js`, base));
  await glue.default({ module_or_path: new URL(`${glueName}_bg.wasm`, base) });
  if (build.threads) await glue.initThreadPool(threads);
  return glue;
}

/**
 * Streams every part of `manifest` (relative to `baseUrl`) into `runtime`. `onProgress` gets
 * `{ done, total, network, cache }` byte counts. Parts are cached in the OPFS directory
 * `cacheName`, one file per part named by its BLAKE3.
 */
export async function loadParts({ runtime, manifest, baseUrl, cacheName, onProgress }) {
  const store = await openStore(cacheName);
  const total = manifest.parts.reduce((n, part) => n + part.bytes, 0);
  const progress = { done: 0, total, network: 0, cache: 0 };
  const report = (bytes, source) => {
    progress.done += bytes;
    progress[source] += bytes;
    onProgress?.({ ...progress });
  };
  const queue = manifest.parts.map((part, index) => ({ part, index }));
  const lane = async () => {
    for (let item = queue.shift(); item; item = queue.shift()) {
      await loadPart({ runtime, store, item, baseUrl, report });
    }
  };
  await Promise.all(Array.from({ length: PARALLEL_PARTS }, lane));
  return progress;
}

async function loadPart({ runtime, store, item: { part, index }, baseUrl, report }) {
  if (await store?.has(part)) {
    const pushed = await pushStream(runtime, index, await store.read(part), (n) => report(n, 'cache'));
    try {
      runtime.end_part(index);
      return;
    } catch (error) {
      // Only a hash mismatch means the cached copy is bad; anything else (the runtime failing
      // to build from a complete bundle) is reported as is.
      if (errorKind(error) !== 'integrity') throw error;
      await store.remove(part);
      runtime.restart_part(index);
      report(-pushed, 'cache');
    }
  }
  const response = await fetchOk(new URL(part.file, baseUrl));
  const sink = await store?.writer(part);
  try {
    await pushStream(runtime, index, response.body, (n) => report(n, 'network'), sink);
    runtime.end_part(index);
  } catch (error) {
    sink?.close();
    if (errorKind(error) !== 'format') await store?.remove(part); // bad bytes, or a cut-off download
    throw error;
  }
  sink?.close();
}

/** Pushes a byte stream into part `index` in writes of about PUSH_BYTES; returns the bytes. */
async function pushStream(runtime, index, stream, onBytes, sink) {
  const reader = stream.getReader();
  let pending = [];
  let pendingBytes = 0;
  let total = 0;
  const flush = () => {
    if (!pendingBytes) return;
    const bytes = pending.length === 1 ? pending[0] : concat(pending, pendingBytes);
    runtime.write_part(index, bytes);
    sink?.write(bytes);
    onBytes(bytes.length);
    total += bytes.length;
    pending = [];
    pendingBytes = 0;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    pending.push(value);
    pendingBytes += value.length;
    if (pendingBytes >= PUSH_BYTES) flush();
  }
  flush();
  return total;
}

function concat(chunks, length) {
  const out = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

export async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

/**
 * Parts cached in the origin-private file system. A file whose part fails verification is
 * deleted; a cached file is verified again on every load (an interrupted write leaves a short
 * file, which `has` rejects). Without OPFS everything is fetched.
 */
async function openStore(cacheName) {
  let dir;
  try {
    dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(cacheName, { create: true });
  } catch {
    return null;
  }
  return {
    async has(part) {
      try {
        return (await (await dir.getFileHandle(part.blake3)).getFile()).size === part.bytes;
      } catch {
        return false;
      }
    },
    async read(part) {
      return (await (await dir.getFileHandle(part.blake3)).getFile()).stream();
    },
    async remove(part) {
      await dir.removeEntry(part.blake3).catch(() => {});
    },
    async writer(part) {
      let access;
      try {
        access = await (await dir.getFileHandle(part.blake3, { create: true })).createSyncAccessHandle();
      } catch {
        return null; // no sync access handles (or another tab holds the file): don't cache
      }
      access.truncate(0);
      let offset = 0;
      let failed = false; // e.g. quota exceeded: keep loading, just stop caching this part
      return {
        write(bytes) {
          if (failed) return;
          try {
            offset += access.write(bytes, { at: offset });
          } catch {
            failed = true;
          }
        },
        close() {
          try {
            access.flush();
          } finally {
            access.close();
          }
        },
      };
    },
  };
}
