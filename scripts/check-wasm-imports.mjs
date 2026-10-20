#!/usr/bin/env node
// The sealed-runtime check: a runtime module may only import wasm-bindgen intrinsics, never
// anything network-, storage-, clock- or entropy-capable. Exits non-zero on any import outside
// the allowlist.
import { readFileSync } from 'node:fs';

const USAGE = `usage: check-wasm-imports.mjs <module.wasm> [--glue <glue.js>] [--none] [--no-relaxed] [--threaded]
       check-wasm-imports.mjs --selftest

  --none        require an empty import section (a module that needs nothing from its host)
  --glue        print the JS body behind each allowed import, for review
  --no-relaxed  the module must not use relaxed SIMD (its \`target_features\` section must not list
                it), so the plain-SIMD fallback builds validate on engines without the proposal
  --threaded    the shared-memory build: additionally allow exactly the imports wasm-bindgen-rayon
                needs (listed in THREADED with the reason for each), and check that the Rayon
                snippet next to the glue is our thread host (no fetch, page-owned workers)
  --selftest    run the check on synthesised modules that must fail (and controls that must pass)`;

const args = process.argv.slice(2);

// --selftest: synthesise modules and make sure the check rejects every forbidden one and accepts
// the controls, so the check is known to fail when it should (a test that cannot fail proves
// nothing).
if (args[0] === '--selftest') {
  const enc = (s) => [s.length, ...Buffer.from(s)];
  const HEADER = [0, 97, 115, 109, 1, 0, 0, 0];
  const TYPE_SECTION = [1, 4, 1, 96, 0, 0]; // one type: () -> ()
  const importing = (name) => {
    const imp = [1, ...enc('./glue.js'), ...enc(name), 0, 0];
    return [...TYPE_SECTION, 2, imp.length, ...imp];
  };
  // A `target_features` custom section listing one feature with the `+` (used) prefix.
  const usingFeature = (feature) => {
    const body = [...enc('target_features'), 1, '+'.charCodeAt(0), ...enc(feature)];
    return [0, body.length, ...body];
  };
  const cases = [
    ...['__wbg_fetch_0123456789abcdef', '__wbg_getRandomValues_0123456789abcdef', '__wbg_now_0123456789abcdef', 'something_else']
      .map((name) => ({ name, flags: [], sections: importing(name), pass: false })),
    { name: 'allowed intrinsic (control)', flags: [], sections: importing('__wbindgen_init_externref_table'), pass: true },
    { name: '--none with an allowed intrinsic', flags: ['--none'], sections: importing('__wbindgen_init_externref_table'), pass: false },
    { name: '--none without imports (control)', flags: ['--none'], sections: [], pass: true },
    { name: '--no-relaxed with +relaxed-simd', flags: ['--no-relaxed'], sections: usingFeature('relaxed-simd'), pass: false },
    { name: '--no-relaxed with +simd128 (control)', flags: ['--no-relaxed'], sections: usingFeature('simd128'), pass: true },
    { name: '--threaded with fetch', flags: ['--threaded'], sections: importing('__wbg_fetch_0123456789abcdef'), pass: false },
  ];
  const { execFileSync } = await import('node:child_process');
  const { writeFileSync, mkdtempSync } = await import('node:fs');
  const { join } = await import('node:path');
  const dir = mkdtempSync(join((await import('node:os')).tmpdir(), 'fusor-ai-imports-'));
  let failures = 0;
  for (const c of cases) {
    const f = join(dir, 'm.wasm');
    writeFileSync(f, new Uint8Array([...HEADER, ...c.sections]));
    let passed = true;
    try {
      execFileSync(process.execPath, [new URL(import.meta.url).pathname, f, ...c.flags], { stdio: 'pipe' });
    } catch {
      passed = false;
    }
    const ok = passed === c.pass;
    console.log(`selftest ${ok ? 'ok  ' : 'FAIL'}: ${c.name} ${passed ? 'accepted' : 'rejected'}`);
    if (!ok) failures++;
  }
  process.exit(failures ? 1 : 0);
}
const file = args[0];
const glueIdx = args.indexOf('--glue');
const glue = glueIdx >= 0 ? readFileSync(args[glueIdx + 1], 'utf8') : null;
const requireNone = args.includes('--none');
const threaded = args.includes('--threaded');
const noRelaxed = args.includes('--no-relaxed');
if (!file) {
  console.error(USAGE);
  process.exit(2);
}

const mod = new WebAssembly.Module(readFileSync(file));
const imports = WebAssembly.Module.imports(mod);

// wasm-bindgen names imported JS functions `__wbg_<name>_<hash>` and intrinsics `__wbindgen_*`.
// Each allowed entry names what the glue does with it.
const ALLOW = [
  [/^__wbindgen_init_externref_table$/, 'initialises the externref table (no I/O)'],
  [/^__wbg___wbindgen_throw_[0-9a-f]+$/, 'throw new Error(message)'],
  [/^__wbg_Error_[0-9a-f]+$/, 'new Error(message) for JsError'],
  [/^__wbindgen_object_drop_ref$/, 'drops a JS object reference'],
  [/^__wbindgen_string_new$/, 'creates a JS string'],
];
// Extra imports of the threaded build. None can call an arbitrary JS function or method: the
// global accessors return an object the module can only pass back to these same imports.
const THREADED = [
  [/^memory$/, 'the shared WebAssembly.Memory compute workers run on', 'memory'],
  [/^__wbg_startWorkers_[0-9a-f]+$/, 'thread-host.js: asks the page to spawn compute workers'],
  [/^__wbg___wbindgen_module_[0-9a-f]+$/, 'this module, passed to compute workers'],
  [/^__wbg___wbindgen_memory_[0-9a-f]+$/, 'this memory, passed to compute workers'],
  [/^__wbg___wbindgen_is_undefined_[0-9a-f]+$/, 'typeof x === "undefined"'],
  [/^__wbg_static_accessor_URL_[0-9a-f]+$/, 'import.meta.url of the glue (a string), for compute workers'],
  [/^__wbg_instanceof_Window_[0-9a-f]+$/, 'Rayon web_spin_lock: is this the (non-blocking) main thread?'],
  [/^__wbg_static_accessor_(GLOBAL_THIS|SELF|GLOBAL|WINDOW)_[0-9a-f]+$/, 'js-sys global() lookup for the check above'],
];

// Belt and braces: names that must never appear, whatever wasm-bindgen does in future.
const DENY = /fetch|xmlhttp|websocket|eventsource|beacon|indexeddb|caches|storage|opfs|filesystem|crypto|random|date|now|performance|worker|postmessage|import|eval|function|navigator|location|document|window|self|global|process|require|socket|request|send|open/i;

let bad = 0;
console.log(`${file}: ${imports.length} import(s)`);
for (const imp of imports) {
  const name = imp.name;
  const threadedOk = threaded && THREADED.find(([re, , kind]) => re.test(name) && imp.kind === (kind ?? 'function'));
  const allowed = threadedOk || ALLOW.find(([re]) => re.test(name));
  const stripped = name.replace(/^__wbg_/, '').replace(/_[0-9a-f]{8,}$/, '');
  const denied = !threadedOk && DENY.test(stripped.replace(/^__wbindgen_(throw|init_externref_table)$/, ''));
  if (requireNone || !allowed || (!threadedOk && imp.kind !== 'function') || denied) {
    bad++;
    console.log(`  FAIL ${imp.module} :: ${name} (${imp.kind})`);
  } else {
    console.log(`  ok   ${imp.module} :: ${name} — ${allowed[1]}`);
  }
  if (glue) {
    const m = glue.match(new RegExp(`${name}:\\s*function\\(([^)]*)\\)\\s*\\{([\\s\\S]*?)\\n\\s*\\},`));
    if (m) console.log(`         glue: function(${m[1]}) {${m[2].replace(/\s+/g, ' ')} }`);
  }
}
// Features the module was compiled with: the `target_features` custom section is a LEB128 count
// of (prefix byte, LEB128 length, name) entries.
function targetFeatures(m) {
  const sec = WebAssembly.Module.customSections(m, 'target_features')[0];
  if (!sec) return [];
  const b = new Uint8Array(sec);
  let p = 0;
  const leb = () => { let v = 0, s = 0, x; do { x = b[p++]; v |= (x & 0x7f) << s; s += 7; } while (x & 0x80); return v; };
  const out = [];
  for (let n = leb(); n > 0; n--) {
    const prefix = String.fromCharCode(b[p++]);
    const len = leb();
    out.push(prefix + new TextDecoder().decode(b.subarray(p, p + len)));
    p += len;
  }
  return out;
}
if (noRelaxed) {
  const feats = targetFeatures(mod);
  const relaxed = feats.some((f) => f.endsWith('relaxed-simd') && !f.startsWith('-'));
  console.log(`  ${relaxed ? 'FAIL' : 'ok  '} target features: ${feats.join(' ') || '(none)'}${relaxed ? ' — uses relaxed SIMD' : ' — no relaxed SIMD'}`);
  if (relaxed) bad++;
}
if (threaded && glue) {
  // The thread host must be ours: page-owned workers, nothing fetched.
  const { readdirSync, existsSync } = await import('node:fs');
  const { dirname, join } = await import('node:path');
  const snippets = join(dirname(args[glueIdx + 1]), 'snippets');
  const hosts = existsSync(snippets)
    ? readdirSync(snippets).map((d) => join(snippets, d, 'src/workerHelpers.no-bundler.js')).filter((f) => existsSync(f))
    : [];
  for (const f of hosts) {
    const src = readFileSync(f, 'utf8');
    const ours = src.includes('fusor-ai-spawn-compute') && !/\bfetch\s*\(/.test(src) && !/importScripts|XMLHttpRequest|WebSocket/.test(src);
    console.log(`  ${ours ? 'ok  ' : 'FAIL'} thread host ${f.split('/snippets/')[1]}${ours ? ' — thread-host.js (no fetch)' : ' — not fusor-ai packages/host/thread-host.js'}`);
    if (!ours) bad++;
  }
  if (hosts.length !== 1) { console.log(`  FAIL expected exactly one Rayon thread host, found ${hosts.length}`); bad++; }
}
if (bad) {
  console.error(`import check FAILED: ${bad} problem(s) (see FAIL lines above)`);
  process.exit(1);
}
console.log(`import check passed: no network, storage, clock or entropy capability${threaded ? ' (threaded: shared memory + page-owned compute workers)' : ''}`);
