# Ming-3.0-tts-0.5B

Text-to-speech with [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)
running entirely in the browser: Rust compiled to WebAssembly, on the CPU, with WASM threads.
Type text, press Generate, get 44.1 kHz speech. The server only serves static files; after the
model has loaded the page makes no network requests.

On an Apple M4 Pro, Chrome generates speech 1.5–2× faster than real time (8 threads, relaxed
SIMD); the numbers for every build are in [`results.md`](results.md).

## Run a release

A release holds the `ming` CLI (macOS on Apple silicon, Linux on x86-64) and
`ming-web-<version>.tar.gz`: the page with its four runtimes, a dev server and the checkpoint
download. Without Rust, with Node.js ≥ 20, curl, about 4 GB of disk and 5 GB of free RAM:

```sh
curl -fsSL https://github.com/fusor-rs/fusor-ai/releases/latest/download/install.sh | sh
version=0.1.0
curl -fsSL https://github.com/fusor-rs/fusor-ai/releases/download/v$version/ming-web-$version.tar.gz | tar -xz
cd ming-web-$version
./fetch-checkpoint.sh checkpoint    # 3 GB, sha256-checked
ming compile checkpoint model       # the 850 MB bundle the page loads from model/
node serve.mjs --mount /=.          # http://localhost:8787/
```

`install.sh` puts `ming` in `~/.fusor-ai/bin` (`FUSOR_AI_INSTALL` changes it) and takes a
version (`sh install.sh v0.1.0`); rerunning it upgrades. Any static server works in place of
`serve.mjs` if it sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (without them the page uses the single-threaded
builds).

## Run it from source

Prerequisites: Rust via [rustup](https://rustup.rs) (`rust-toolchain.toml` pins 1.99.0; the
threaded builds use `nightly-2025-11-15` with `rust-src`, installed by `just tools`), Node.js
≥ 20, [`just`](https://github.com/casey/just), curl, about 4 GB of disk for the checkpoint and
the bundle, about 5 GB of free RAM while the model compiles (the compiler reads the 3 GB
checkpoint into memory), and a recent desktop browser (Chrome/Edge 114+ for the fastest build;
any browser with WebAssembly SIMD works).

```sh
git clone https://github.com/fusor-rs/fusor-ai && cd fusor-ai
just ming-build    # tools, native CLI, downloads + compiles the model (850 MB), four browser runtimes
just ming-serve    # http://localhost:8787/
```

`just ming-build` downloads the pinned checkpoint (3 GB, sha256-checked) into
`.cache/Ming-3.0-tts-0.5B` unless `CHECKPOINT=<dir>` already holds `model.safetensors` and
`tokenizer.json`. `PORT=… just ming-serve` changes the port.

## What to expect

The page loads the model (progress per MB; a revisit loads it from the browser's
origin-private storage in about a second), then offers:

- a text box (long text is split into fragments and synthesized one after another; English
  numbers, money, percentages and units are read out),
- a voice: five English and two Chinese built-in voices, **Describe a voice…** (a voice
  designed from a description) or **Free-form instruction…** (voice and delivery in words),
- a speed preset (quality, balanced, fast) and a seed,
- Generate / Stop. Audio plays while it is generated when generation is faster than real time,
  otherwise when it is done; **Download WAV** saves it.

The footer names the runtime build in use and counts network requests since the model loaded.
`?threads=1` forces the single-threaded build, `?build=web-single` pins a build, `?model=<url>`
loads another bundle (another origin must allow CORS and send `Cross-Origin-Resource-Policy`,
since the page is cross-origin isolated).

## How it works

```
text → fragments → prompt (built-in voice caption) → Qwen2-0.5B prefill
     → per 80 ms frame: z → flow-matching DiT (10 Euler steps, CFG) → 64-d latent
                         latent → aggregator DiT → next LLM input → z
                         latent → streaming VAE decoder (Qwen2 backbone + iSTFT) → 3528 samples
```

`ming::pipeline::Stage` names these stages; `ming bench` times them and `ming parity` checks
them one by one.

- `crates/models/ming`: the model on fusor-ai's blocks, no I/O. `model` is the table of
  contents (each component loads itself from the bundle), `flow_head` the flow head and the
  aggregator on `blocks::dit` and `blocks::flow`, `vocoder` the streaming VAE decoder on
  `blocks::qwen2` and `blocks::istft`, `session` the synthesis loop, `prompt` and `text` the
  prompt and the text frontend.
- `crates/models/ming-compile`: the checkpoint → bundle compiler (what to keep, renames, q/k/v
  fusion, the dtype of every tensor).
- `crates/hosts/ming-runtime`: the wasm-bindgen surface (parts in, PCM out). Its modules import
  no network, storage, clock or entropy capability (`scripts/check-wasm-imports.mjs`).
- `crates/hosts/ming-cli`: the native CLI `ming`: `compile`, `say`, `bench`, `parity`.
- `packages/ming-web`: the page and the engine worker (generation loop); loading and the compute
  workers are `packages/host`, copied to `packages/ming-web/dist/host/` by
  `just ming-runtimes`.
- `docs/ming/`: [`plan.md`](plan.md) (the design), [`decisions.md`](decisions.md),
  [`results.md`](results.md).

## Checks

```sh
just check          # fmt, tests, clippy (native + wasm, -D warnings), docs, host scripts, layers
just ming-runtimes  # the four runtime builds and their import checks
just ming-parity    # every stage against the PyTorch goldens (needs `just ming-goldens` first)
just ming-smoke     # both single-threaded runtimes in Node
just ming-e2e       # headless Chrome: load, generate, 0 requests during synthesis, ASR check, revisit from cache
```

`just ming-goldens` runs the PyTorch reference: `REFERENCE=<dir>` must hold a checkout of
[inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) at `e999c39` as
`Ming-3.0-tts/`, the checkpoint as `Ming-3.0-tts-0.5b/`, and a `.venv/` with that repository's
requirements. `just ming-parity` then compiles a 4.3 GB f32 bundle. The e2e's ASR check uses
faster-whisper from `ASR_PYTHON` (skipped if absent).

Native use: `target/release/ming say dist/ming/model "Hello there." -o hello.wav
[--voice UK_George] [--design "…"] [--freeform "…"] [--preset fast]`,
`target/release/ming bench dist/ming/model`.

## Limits

- No voice cloning from a reference recording (it needs the VAE encoder and the CAM++ speaker
  model in the browser; decisions D2).
- The reference's FST text normalizer is not ported (decisions D10): dates, times and
  abbreviations are read as written.
- Speed depends on the device: without cross-origin isolation (COOP/COEP) or relaxed SIMD the
  page falls back to slower builds (single-threaded SIMD128 runs at about a quarter of real
  time on an M4 Pro).
- The model's license (MIT) and its usage disclaimer apply: label synthetic speech as synthetic.
