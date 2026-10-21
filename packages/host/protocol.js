// The messages between a threaded runtime's thread host (thread-host.js, in the engine worker
// and in each compute worker) and the page's compute pool (compute.js). `build-runtime.sh` copies
// this file next to the thread host.

export const SPAWN_COMPUTE = 'fusor-ai-spawn-compute';
export const RAYON_INIT = 'fusor-ai-rayon-init';
export const RAYON_READY = 'fusor-ai-rayon-ready';
export const RAYON_ERROR = 'fusor-ai-rayon-error';
