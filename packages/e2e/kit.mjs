// The shared half of a product's end-to-end run in headless Chrome (its `e2e/run.mjs` drives the
// page): the flags every run takes, fusor-ai's server when no page answers on PORT, a fresh
// profile with every request recorded, the revisit check, and the results file and exit status.
//
// Results (JSON, outputs, screenshots) go to <product>/e2e/results/<STAMP>-*, or to `--out <dir>`.

import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const PORT = process.env.PORT ?? '8787';
const SERVE = fileURLToPath(new URL('../../scripts/serve.mjs', import.meta.url));
export const STAMP = new Date().toISOString().replace(/[:.]/g, '-');
export const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** `--threads`, `--build`, `--url`, `--model` and `--out`, with the product's own `options`. */
export function flags(options) {
  const { values } = parseArgs({
    options: {
      threads: { type: 'string' },
      build: { type: 'string' },
      ...options,
      url: { type: 'string' },
      model: { type: 'string' },
      out: { type: 'string' },
    },
  });
  return values;
}

/** The requests made since `mark` (a length of `requests`), but the page's own `blob:` URLs. */
export const fetched = (requests, mark) => requests.slice(mark).filter((u) => !u.startsWith('blob:'));

/**
 * Opens a fresh Chrome profile named after `name` (with the Chrome `args`, and `init`, a
 * `[function, argument]` run in the page before its scripts) and runs `body` with
 * `{ page, requests, query, base, runDir, results, check, revisit }`; `results` starts with
 * `fields`. Then writes the results to `<runDir>/<STAMP>.json` and exits 1 if a check failed.
 * `body` may return a note for the verdict of a run that passed.
 */
export async function run({ root, name, values, fields, args = [], init }, body) {
  const base = (values.url ?? `http://127.0.0.1:${PORT}`).replace(/\/$/, '');
  const runDir = values.out ?? join(root, 'e2e/results');
  const up = async () => {
    try {
      return (await fetch(`${base}/model/manifest.json`)).ok;
    } catch {
      return false;
    }
  };
  let server = null;
  if (!values.url && !(await up())) {
    log(`starting fusor-ai's serve.mjs on :${PORT}`);
    const mounts = ['--mount', '/=app/dist', '--mount', '/model/=dist/model'];
    server = spawn(process.execPath, [SERVE, '--port', PORT, ...mounts], { cwd: root, stdio: 'inherit' });
    for (let i = 0; i < 50 && !(await up()); i++) await new Promise((r) => setTimeout(r, 200));
  }

  mkdirSync(runDir, { recursive: true });
  const profile = mkdtempSync(join(tmpdir(), `${name}-e2e-`));
  const results = { started: new Date().toISOString(), base, ...fields };
  const failures = [];
  const check = (ok, message) => {
    log(`${ok ? 'ok  ' : 'FAIL'} ${message}`);
    if (!ok) failures.push(message);
  };
  let note = '';
  try {
    const context = await chromium.launchPersistentContext(profile, { headless: true, args });
    if (init) await context.addInitScript(...init);
    const requests = [];
    context.on('request', (r) => requests.push(r.url()));
    const page = context.pages()[0] ?? (await context.newPage());
    page.on('console', (m) => m.type() === 'error' && log('console:', m.text()));
    page.on('pageerror', (e) => log('pageerror:', e.message));
    const query = `?${new URLSearchParams(Object.entries({ threads: values.threads, build: values.build, model: values.model }).filter(([, v]) => v))}`;

    /** Reload: the page must load every part from OPFS once `ready` holds in it. */
    const revisit = async (ready, timeout) => {
      const t0 = Date.now();
      await page.reload();
      await page.waitForFunction(ready, null, { timeout, polling: 250 });
      results.revisitMs = Date.now() - t0;
      results.revisitDetail = await page.textContent('#runtime');
      check(/model loaded from this device/.test(results.revisitDetail), `revisit in ${(results.revisitMs / 1000).toFixed(1)} s: ${results.revisitDetail}`);
    };
    note = (await body({ page, requests, query, base, runDir, results, check, revisit })) ?? '';
    await context.close();
  } finally {
    rmSync(profile, { recursive: true, force: true });
    server?.kill();
  }

  results.failures = failures;
  writeFileSync(join(runDir, `${STAMP}.json`), JSON.stringify(results, null, 2));
  log(failures.length ? `e2e FAILED (${failures.length})` : `e2e passed${note}`, `-> ${values.out ?? 'e2e/results'}/${STAMP}.json`);
  process.exit(failures.length ? 1 : 0);
}
