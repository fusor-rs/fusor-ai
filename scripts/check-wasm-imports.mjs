#!/usr/bin/env node
// The sealed-runtime check: a runtime module may only import wasm-bindgen intrinsics, never
// anything network-, storage-, clock- or entropy-capable. Exits non-zero on any import outside
// the allowlist.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const USAGE = `usage: check-wasm-imports.mjs <module.wasm> [--glue <glue.js>] [--none]
                             [--no-relaxed] [--threaded]
       check-wasm-imports.mjs --selftest

  --none        require an empty import section (a module that needs nothing from its host)
  --glue        print the JS body behind each allowed import, for review
  --no-relaxed  the module must not use relaxed SIMD (its \`target_features\` section must not
                list it), so the plain-SIMD fallback builds validate on engines without the
                proposal
  --threaded    the shared-memory build: additionally allow exactly the imports
                wasm-bindgen-rayon needs (listed in THREADED with the reason for each), and
                check that the Rayon snippet next to the glue is our thread host
  --selftest    run the check on synthesised modules that must fail (and controls that must
                pass)`;

const HOST = join(dirname(fileURLToPath(import.meta.url)), '../packages/host');
// The thread host build-runtime.sh puts in place of wasm-bindgen-rayon's, and its protocol.
const THREAD_HOST_FILES = [
  ['workerHelpers.no-bundler.js', 'thread-host.js'],
  ['protocol.js', 'protocol.js'],
];

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
  [
    /^__wbg_static_accessor_URL_[0-9a-f]+$/,
    'import.meta.url of the glue (a string), for compute workers',
  ],
  [
    /^__wbg_instanceof_Window_[0-9a-f]+$/,
    'Rayon web_spin_lock: is this the (non-blocking) main thread?',
  ],
  [
    /^__wbg_static_accessor_(GLOBAL_THIS|SELF|GLOBAL|WINDOW)_[0-9a-f]+$/,
    'js-sys global() lookup for the check above',
  ],
];

// Belt and braces: names that must never appear, whatever wasm-bindgen does in future.
const DENY = new RegExp(
  [
    'fetch|xmlhttp|websocket|eventsource|beacon|indexeddb|caches|storage|opfs|filesystem',
    'crypto|random|date|now|performance|worker|postmessage|import|eval|function|navigator',
    'location|document|window|self|global|process|require|socket|request|send|open',
  ].join('|'),
  'i',
);

const args = process.argv.slice(2);
if (args[0] === '--selftest') {
  process.exit(selftest() ? 0 : 1);
}
const file = args[0];
if (!file) {
  console.error(USAGE);
  process.exit(2);
}
const gluePath = args.includes('--glue') ? args[args.indexOf('--glue') + 1] : null;
const options = {
  glue: gluePath ? readFileSync(gluePath, 'utf8') : null,
  none: args.includes('--none'),
  threaded: args.includes('--threaded'),
  noRelaxed: args.includes('--no-relaxed'),
};
const mod = new WebAssembly.Module(readFileSync(file));
console.log(`${file}: ${WebAssembly.Module.imports(mod).length} import(s)`);
let bad = checkImports(mod, options);
if (options.noRelaxed) bad += checkNoRelaxed(mod);
if (options.threaded && gluePath) bad += checkThreadHost(gluePath);
if (bad) {
  console.error(`import check FAILED: ${bad} problem(s) (see FAIL lines above)`);
  process.exit(1);
}
const shared = options.threaded ? ' (threaded: shared memory + page-owned compute workers)' : '';
console.log(`import check passed: no network, storage, clock or entropy capability${shared}`);

/** Prints every import as allowed or not; returns how many are not. */
function checkImports(mod, { glue, none, threaded }) {
  let bad = 0;
  for (const imp of WebAssembly.Module.imports(mod)) {
    const name = imp.name;
    const threadedOk =
      threaded &&
      THREADED.find(([re, , kind]) => re.test(name) && imp.kind === (kind ?? 'function'));
    const allowed = threadedOk || ALLOW.find(([re]) => re.test(name));
    const stripped = name.replace(/^__wbg_/, '').replace(/_[0-9a-f]{8,}$/, '');
    const intrinsic = stripped.replace(/^__wbindgen_(throw|init_externref_table)$/, '');
    const denied = !threadedOk && DENY.test(intrinsic);
    if (none || !allowed || (!threadedOk && imp.kind !== 'function') || denied) {
      bad++;
      console.log(`  FAIL ${imp.module} :: ${name} (${imp.kind})`);
    } else {
      console.log(`  ok   ${imp.module} :: ${name} — ${allowed[1]}`);
    }
    const pattern = `${name}:\\s*function\\(([^)]*)\\)\\s*\\{([\\s\\S]*?)\\n\\s*\\},`;
    const body = glue?.match(new RegExp(pattern));
    if (body) {
      console.log(`         glue: function(${body[1]}) {${body[2].replace(/\s+/g, ' ')} }`);
    }
  }
  return bad;
}

/** Fails a module that uses relaxed SIMD. */
function checkNoRelaxed(mod) {
  const features = targetFeatures(mod);
  const relaxed = features.some(
    (feature) => feature.endsWith('relaxed-simd') && !feature.startsWith('-'),
  );
  const listed = features.join(' ') || '(none)';
  const verdict = relaxed ? 'FAIL' : 'ok  ';
  const uses = relaxed ? 'uses' : 'no';
  console.log(`  ${verdict} target features: ${listed} — ${uses} relaxed SIMD`);
  return relaxed ? 1 : 0;
}

/** The features the module was compiled with: its `target_features` custom section is a
 * LEB128 count of (prefix byte, LEB128 length, name) entries. */
function targetFeatures(mod) {
  const section = WebAssembly.Module.customSections(mod, 'target_features')[0];
  if (!section) return [];
  const bytes = new Uint8Array(section);
  let at = 0;
  const leb128 = () => {
    let value = 0;
    let shift = 0;
    for (;;) {
      const byte = bytes[at++];
      value |= (byte & 0x7f) << shift;
      shift += 7;
      if (!(byte & 0x80)) return value;
    }
  };
  const features = [];
  for (let count = leb128(); count > 0; count--) {
    const prefix = String.fromCharCode(bytes[at++]);
    const length = leb128();
    features.push(prefix + new TextDecoder().decode(bytes.subarray(at, at + length)));
    at += length;
  }
  return features;
}

/** The Rayon snippet next to the glue must be our thread host (page-owned workers, nothing
 * fetched): byte for byte packages/host's, with its protocol beside it. */
function checkThreadHost(gluePath) {
  const snippets = join(dirname(gluePath), 'snippets');
  const hosts = existsSync(snippets)
    ? readdirSync(snippets)
        .map((dir) => join(snippets, dir, 'src'))
        .filter((dir) => existsSync(join(dir, THREAD_HOST_FILES[0][0])))
    : [];
  let bad = 0;
  for (const dir of hosts) {
    const ours = THREAD_HOST_FILES.every(([snippet, host]) => {
      const path = join(dir, snippet);
      const expected = readFileSync(join(HOST, host), 'utf8');
      return existsSync(path) && readFileSync(path, 'utf8') === expected;
    });
    const where = relative(snippets, dir);
    const verdict = ours ? 'ok   thread host' : 'FAIL thread host';
    console.log(`  ${verdict} ${where} — ${ours ? '' : 'not '}packages/host/thread-host.js`);
    if (!ours) bad++;
  }
  if (hosts.length !== 1) {
    console.log(`  FAIL expected exactly one Rayon thread host, found ${hosts.length}`);
    bad++;
  }
  return bad;
}

/** Runs the check on synthesised modules and makes sure it rejects every forbidden one and
 * accepts the controls, so the check is known to fail when it should (a test that cannot fail
 * proves nothing). Returns whether every case came out as expected. */
function selftest() {
  const encode = (text) => [text.length, ...Buffer.from(text)];
  const HEADER = [0, 97, 115, 109, 1, 0, 0, 0];
  const TYPE_SECTION = [1, 4, 1, 96, 0, 0]; // one type: () -> ()
  const importing = (name) => {
    const entry = [1, ...encode('./glue.js'), ...encode(name), 0, 0];
    return [...TYPE_SECTION, 2, entry.length, ...entry];
  };
  // A `target_features` custom section listing one feature with the `+` (used) prefix.
  const usingFeature = (feature) => {
    const body = [...encode('target_features'), 1, '+'.charCodeAt(0), ...encode(feature)];
    return [0, body.length, ...body];
  };
  const forbidden = [
    '__wbg_fetch_0123456789abcdef',
    '__wbg_getRandomValues_0123456789abcdef',
    '__wbg_now_0123456789abcdef',
    'something_else',
  ];
  const allowed = importing('__wbindgen_init_externref_table');
  const cases = [
    ...forbidden.map((name) => ({ name, flags: [], sections: importing(name), pass: false })),
    { name: 'allowed intrinsic (control)', flags: [], sections: allowed, pass: true },
    { name: '--none with an allowed intrinsic', flags: ['--none'], sections: allowed, pass: false },
    { name: '--none without imports (control)', flags: ['--none'], sections: [], pass: true },
    {
      name: '--no-relaxed with +relaxed-simd',
      flags: ['--no-relaxed'],
      sections: usingFeature('relaxed-simd'),
      pass: false,
    },
    {
      name: '--no-relaxed with +simd128 (control)',
      flags: ['--no-relaxed'],
      sections: usingFeature('simd128'),
      pass: true,
    },
    {
      name: '--threaded with fetch',
      flags: ['--threaded'],
      sections: importing('__wbg_fetch_0123456789abcdef'),
      pass: false,
    },
  ];
  const module = join(mkdtempSync(join(tmpdir(), 'fusor-ai-imports-')), 'm.wasm');
  let failures = 0;
  for (const { name, flags, sections, pass } of cases) {
    writeFileSync(module, new Uint8Array([...HEADER, ...sections]));
    let passed = true;
    try {
      execFileSync(process.execPath, [fileURLToPath(import.meta.url), module, ...flags], {
        stdio: 'pipe',
      });
    } catch {
      passed = false;
    }
    const expected = passed === pass;
    const verdict = expected ? 'ok  ' : 'FAIL';
    console.log(`selftest ${verdict}: ${name} ${passed ? 'accepted' : 'rejected'}`);
    if (!expected) failures++;
  }
  return failures === 0;
}
