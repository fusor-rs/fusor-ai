// The page side of a threaded fusor-ai runtime: its compute workers.
//
// The runtime's thread host (thread-host.js, running in the engine worker) asks the page for
// workers with a `SPAWN_COMPUTE` message; the page owns them, so it can always terminate the
// whole pool. A compute thread that fails after start-up (a Rust panic traps it) would leave the
// others waiting at a barrier forever, so it is reported to `onCrash` instead of ignored.

import { RAYON_ERROR, RAYON_INIT, RAYON_READY } from './protocol.js';

export { SPAWN_COMPUTE } from './protocol.js';

export class ComputePool {
  #workers = [];
  #onCrash;

  constructor(onCrash) {
    this.#onCrash = onCrash;
  }

  /** Answers a `SPAWN_COMPUTE` message from the engine worker. */
  async spawn({ url, count, module, memory, receiver, mainJS, port }) {
    try {
      const ready = Array.from({ length: count }, (_, i) => this.#start(i, url, { module, memory, receiver, mainJS }));
      await Promise.all(ready);
      port.postMessage({ ready: true });
    } catch (error) {
      this.terminate();
      port.postMessage({ ready: false, error: String(error?.message ?? error) });
    }
  }

  terminate() {
    this.#workers.forEach((w) => w.terminate());
    this.#workers = [];
  }

  #start(index, url, init) {
    const worker = new Worker(url, { type: 'module', name: `fusor-ai-compute-${index}` });
    this.#workers.push(worker);
    return new Promise((resolve, reject) => {
      let started = false;
      worker.onmessage = ({ data }) => {
        if (data?.type === RAYON_READY) {
          started = true;
          resolve();
        }
        if (data?.type === RAYON_ERROR) reject(new Error(data.message));
      };
      worker.onerror = (e) => {
        const message = e.message || 'a compute thread failed';
        if (started) this.#onCrash(message);
        else reject(new Error(message));
      };
      worker.postMessage({ type: RAYON_INIT, ...init });
    });
  }
}
