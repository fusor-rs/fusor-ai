// The thread host for wasm-bindgen-rayon 1.3.0. `scripts/build-runtime.sh` copies it over the
// adapter's `workerHelpers.no-bundler.js`, keeping the adapter's public PoolBuilder ABI (the same
// approach as fusor's worker host, crates/fusor-cli/src/pipeline/workers/rayon.js in the fusor
// repository).
//
// Two differences from the stock adapter:
//  - it fetches nothing (the stock one re-downloads itself to make a blob URL);
//  - the page, not the runtime's worker, creates every compute worker, so the page can always
//    terminate the whole pool, and a blocked coordinator cannot leak threads.
//
// As a compute worker: wait for an init message, instantiate the same verified module on the
// shared memory, then run Rayon's worker loop. As a module imported by the runtime glue:
// export `startWorkers`, which asks the page for workers and waits until they are all ready.

import { RAYON_ERROR, RAYON_INIT, RAYON_READY, SPAWN_COMPUTE } from './protocol.js';

self.addEventListener('message', ({ data }) => {
  if (data?.type !== RAYON_INIT) return;
  initialize(data).catch((error) => {
    postMessage({ type: RAYON_ERROR, message: String(error) });
    self.close();
  });
});

async function initialize(data) {
  const pkg = await import(data.mainJS);
  pkg.initSync({ module: data.module, memory: data.memory });
  postMessage({ type: RAYON_READY });
  pkg.wbg_rayon_start_worker(data.receiver);
}

export async function startWorkers(module, memory, builder) {
  const count = builder.numThreads();
  if (count === 0) throw new Error('num_threads must be > 0');
  const channel = new MessageChannel();
  try {
    await new Promise((resolve, reject) => {
      channel.port1.onmessage = ({ data }) => (data.ready ? resolve() : reject(new Error(data.error)));
      const spawn = {
        type: SPAWN_COMPUTE,
        url: import.meta.url,
        count,
        module,
        memory,
        receiver: builder.receiver(),
        mainJS: builder.mainJS(),
        port: channel.port2,
      };
      postMessage(spawn, [channel.port2]);
    });
    builder.build();
  } finally {
    channel.port1.close();
  }
}
