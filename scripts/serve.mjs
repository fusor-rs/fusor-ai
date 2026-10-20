// Static dev server for fusor-ai pages: serves directories under URL prefixes, sends COOP/COEP
// so the page is cross-origin isolated (shared memory, hence threaded runtimes) and answers
// Range requests. Nothing is computed here.
//
//   node serve.mjs --port 8787 --mount /=packages/web --mount /model/=dist/model
//
// Mount directories are relative to the working directory; the longest matching prefix wins.

import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: { port: { type: 'string', default: '8787' }, mount: { type: 'string', multiple: true, default: [] } } });
const mounts = values.mount
  .map((spec) => {
    const [prefix, dir] = spec.split('=');
    if (!prefix?.startsWith('/') || !prefix.endsWith('/') || !dir) throw new Error(`--mount ${spec}: expected /prefix/=directory`);
    return [prefix, resolve(dir)];
  })
  .sort(([a], [b]) => b.length - a.length);
if (!mounts.length) throw new Error('nothing to serve: pass --mount /=<directory>');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.bin': 'application/octet-stream',
};
const ISOLATION = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
};

function resolvePath(url) {
  const path = decodeURIComponent(new URL(url, 'http://x').pathname);
  for (const [prefix, dir] of mounts) {
    if (!path.startsWith(prefix)) continue;
    const file = normalize(join(dir, path.slice(prefix.length) || 'index.html'));
    return file.startsWith(dir) ? file : null;
  }
  return null;
}

createServer((req, res) => {
  const file = resolvePath(req.url);
  let stat;
  try {
    stat = file && statSync(file);
  } catch {}
  if (!stat?.isFile()) {
    res.writeHead(404, ISOLATION).end('not found');
    return;
  }
  const headers = { ...ISOLATION, 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache' };
  const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
  if (range) {
    const start = Number(range[1]);
    const end = range[2] ? Math.min(Number(range[2]), stat.size - 1) : stat.size - 1;
    res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${stat.size}`, 'Content-Length': end - start + 1 });
    createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { ...headers, 'Content-Length': stat.size });
  if (req.method === 'HEAD') res.end();
  else createReadStream(file).pipe(res);
}).listen(Number(values.port), () => console.log(`serving http://localhost:${values.port}/ (${mounts.map(([p, d]) => `${p} → ${d}`).join(', ')})`));
