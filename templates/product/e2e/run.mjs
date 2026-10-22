// End-to-end run in headless Chrome, in a fresh profile: open the page and wait for the model,
// then reload: the revisit must load every part from OPFS.
//
//   PORT=8787 node e2e/run.mjs [--threads 8] [--build web-threaded]
//   node e2e/run.mjs --url https://example.org    # a deployed page, no local server

import { fileURLToPath } from 'node:url';
import { flags, run } from '../vendor/fusor-ai/packages/e2e/kit.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const values = flags({});
const DEADLINE_MS = 10 * 60 * 1000;

// The page's state machine is `main[data-phase]`: idle, broken.
const loadedOrBroken = () => document.querySelector('main')?.dataset.loaded === 'true' || document.querySelector('main')?.dataset.phase === 'broken';

const launch = {
  root: ROOT,
  name: '__name__',
  values,
};
await run(launch, async ({ page, query, base, results, check, revisit }) => {
  const t0 = Date.now();
  await page.goto(`${base}/${query}`);
  await page.waitForFunction(loadedOrBroken, null, { timeout: DEADLINE_MS, polling: 250 });
  results.loadMs = Date.now() - t0;
  results.runtime = await page.textContent('#runtime');
  check((await page.getAttribute('main', 'data-loaded')) === 'true', `model loaded in ${(results.loadMs / 1000).toFixed(1)} s (${results.runtime})`);
  await revisit(loadedOrBroken, DEADLINE_MS);
});
