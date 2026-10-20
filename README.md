# Compile AI models into sealed WebAssembly runtimes

Ship a model as static files. The browser downloads a bundle of quantized weights and a
3 MB runtime compiled from Rust, and runs inference on the device's CPU with WebAssembly SIMD
and threads. No server, no API keys, no WebGPU required, and no way for the runtime to reach
the network.

fusor-ai holds the toolkit and the models built on it: Ming-3.0-tts-0.5B text-to-speech runs
in Chrome faster than real time ([docs/ming](docs/ming/README.md)). It is the pattern behind
SealedKB's in-browser question answering too, and sits next to
[fusor](https://github.com/fusor-rs/fusor): it reuses fusor's thread recipe for workers that
share one WebAssembly memory.

**In development · v0.1**: this is experimental software. APIs will change; don't rely on it for
production yet. See [status and known limitations](#status-and-known-limitations).

[Get started](#get-started) · [How it works](#how-it-works) · [Layers](#layers) ·
[Models](#models) · [Status](#status-and-known-limitations) · [Contributing](CONTRIBUTING.md)

## A model component in one function

A component of a model is one function that every thread of the team runs. Each thread
computes its share of the output columns; barriers separate the phases that depend on each
other. Here, a projection plus residual, as every transformer layer does twice:

```rust
use kernels::{Linear, QuantizedRows, SharedRows, project_residual};
use team::{PerThread, SharedBuf, Team};

fn project(team: &Team, layer: &Linear, input: &SharedBuf, residual: &SharedBuf, rows: usize) {
    let scratch = PerThread::new(team.threads(), QuantizedRows::default);
    team.run(|ctx| {
        let input = SharedRows { values: input, rows };
        project_residual(ctx, layer, input, residual, scratch.get(ctx));
        ctx.barrier();
        // the next phase may now read every column of `residual`
    });
}
```

`project_residual` quantizes the input rows to int8 per block of 32 and multiplies them with
the 7- or 4-bit weights using the target's dot-product instruction: relaxed SIMD or SIMD128 in
WebAssembly, `sdot` on Apple silicon. The same function is the single-threaded build when the
`threads` feature is off.

## Get started

You need Rust 1.99 or newer ([rustup](https://rustup.rs)), Node.js 20 or newer,
[`just`](https://github.com/casey/just), and, for threaded runtimes, the `nightly-2025-11-15`
toolchain with `rust-src` (`just tools` installs it with the pinned wasm-bindgen and binaryen).

Run Ming in the browser: about 4 GB of disk for the checkpoint and the bundle, and 5 GB of free
RAM while the model compiles.

```sh
git clone https://github.com/fusor-rs/fusor-ai && cd fusor-ai
just ming-build    # tools, native CLI, downloads and compiles the model (850 MB), four browser runtimes
just ming-serve    # http://localhost:8787/
```

A model is four crates and a page, each in its layer (see [Layers](#layers)):

1. **The model** (`crates/models/<model>`): its dimensions, its wiring of blocks and its
   sampling loop. Each component loads itself from a `bundle::Store` (`bundle::Load`) under
   the names its compiler writes.
2. **The compiler** (`crates/models/<model>-compile`): reads the checkpoint with
   `compile::Checkpoint`, keeps, renames and fuses the tensors the model needs, picks a dtype
   per tensor (`kernels::Tensor::encode`) and writes the bundle with `BundleWriter`.
3. **The runtime** (`crates/hosts/<model>-runtime`, a `cdylib`): exposes `write_part`,
   `end_part` and `restart_part` plus the model's inference calls through wasm-bindgen, and has
   a `threads` feature that enables `team/threads` and re-exports
   `wasm_bindgen_rayon::init_thread_pool`. A native CLI (`crates/hosts/<model>-cli`) compiles,
   benchmarks and checks parity.
4. **The page** (`packages/<model>-web`): a module worker that calls `startRuntime` and
   `loadParts` from `packages/host/loader.js`, and a page that hands compute workers to
   `packages/host/compute.js`.

`scripts/build-runtime.sh <runtime crate> <variant> <dist>` builds each of the four runtime
variants, and `scripts/check-wasm-imports.mjs` checks each one is sealed; the `ming-runtimes`
recipe in the `justfile` shows both.

## How it works

- **Ahead of time.** The compiler decides everything a framework would decide at run time: which
  tensors exist, their fused layout, their quantization. The runtime is model code over a
  handful of kernels, with no graph, no dispatcher and no dynamic shapes.
- **Quantized by block.** Weights are 7-bit (`Q7b32`) or 4-bit (`Q4b32`, the ggml Q4_0 layout)
  with one f32 scale per 32 values; activations are quantized to int8 per block on the fly. The
  7-bit layout keeps every weight byte inside the defined range of relaxed SIMD's
  `i32x4.relaxed_dot_i8x16_i7x16_add_s`, so results agree across engines.
- **One thread start per component.** A `Team::run` is one Rayon broadcast; inside it, static
  column splits and spin barriers. Small matrices (a few rows, as in autoregressive decoding or
  a diffusion sampler) stay fast because nothing forks per matmul.
- **Streamed and verified.** A bundle is a manifest of parts with BLAKE3 hashes. The loader
  streams parts into the runtime as they download, the runtime copies each byte range straight
  into its tensor and checks the hash, and verified parts are cached in the browser's
  origin-private storage for the next visit.
- **Sealed.** A runtime imports only wasm-bindgen intrinsics (plus shared memory and the worker
  handles in threaded builds): `scripts/check-wasm-imports.mjs` proves it has no network,
  storage, clock or entropy capability.
- **Four builds, picked by the browser.** Threads when the page is cross-origin isolated,
  relaxed SIMD when the engine validates it, plain SIMD128 everywhere else.

## Layers

Every crate depends only on its own layer or lower ones; `just check` enforces it.

| Layer | Crate or package | Holds |
| --- | --- | --- |
| Primitives | `team` | `Team`, `Ctx` (`split`, `barrier`), `SharedBuf`, `PerThread` |
| | `kernels` | `Tensor` and its formats, `matmul` over per-target backends, `ops` (RMSNorm, RoPE, GELU, SiLU, `exp`, attention), `Linear`, `rng` |
| | `bundle` | `Manifest`, `Store` (streaming, hashed parts; tensors and blobs), the `Load` trait |
| | `compile` | `Checkpoint` (safetensors), `BundleWriter` |
| Blocks | `blocks` | `qwen2` (the Qwen2 decoder stack, causal or sliding-window KV cache), `dit` (diffusion-transformer blocks), `flow` (flow-matching Euler steps with classifier-free guidance, sway sampling), `istft` (streaming inverse STFT) |
| Models | `ming`, `ming-compile` | Ming-3.0-tts-0.5B and its compiler |
| Hosts | `ming-runtime`, `ming-cli` | Ming's wasm-bindgen surface and native CLI |
| | `packages/host` | `loader.js` (build choice, part streaming, OPFS cache), `compute.js` (the compute-worker pool), `thread-host.js` |
| | `packages/ming-web` | Ming's page and engine worker |

`scripts/` holds `build-runtime.sh`, `check-wasm-imports.mjs`, `check-layers.mjs`, `serve.mjs`
and `install-tools.sh`; `scripts/ming/` the checkpoint download, the goldens and the Node
smoke test.

## Models

- [Ming-3.0-tts-0.5B](docs/ming/README.md): text-to-speech (a Qwen2 LLM, a
  diffusion-transformer flow head and a streaming VAE vocoder) in the browser, with parity
  checks against the PyTorch reference.

## Status and known limitations

fusor-ai is v0.1 and experimental. Until 1.0, each minor release may change the crates' APIs and
the bundle format.

| Platform or capability | Status |
| --- | --- |
| Chrome / Edge, threads + relaxed SIMD | Headless Chrome end-to-end check (`just ming-e2e`) |
| Browsers without relaxed SIMD (Safari, older Firefox) | SIMD128 builds; exercised in Node |
| Pages without COOP/COEP | Single-threaded builds |
| Native (aarch64 `sdot`, scalar elsewhere) | Unit tests; the compiler and parity tools run natively |
| WebGPU | Not implemented |
| x86 SIMD natively | Scalar fallback only |

**Not supported yet:**

- Models above roughly 1.5B parameters: wasm32 memory is capped at 4 GiB (threaded builds
  reserve 2 GiB), and on that scale WebGPU runtimes are much faster.
- GGUF checkpoints; only safetensors (bf16 or f32) are read.
- Signed bundles. Parts are content-hashed, but the manifest is not signed yet.

## How is this different from ONNX Runtime Web and transformers.js?

They run a model graph through a general-purpose executor shipped as a large WebAssembly
module, and support many architectures without per-model work. fusor-ai compiles each model to
its own small runtime: there is no graph to interpret, so small-batch work such as a diffusion
sampler's ten passes per audio frame does not pay per-operator dispatch, and the runtime's
import section proves what it cannot do. The cost is per-model engineering.

## How is this different from WebLLM?

WebLLM runs large language models on the GPU through WebGPU and is much faster for models of
several billion parameters. fusor-ai targets models up to about 1.5B that must run on any
device, including those without WebGPU, offline and with an auditable runtime.

## Contributing

fusor-ai is an open source project in early development. The [contributing guide](CONTRIBUTING.md)
covers the setup, checks and workflow.

fusor-ai is built with the help of AI tools, mainly Claude Code, under the same rules the
project asks of contributors: see [Using AI tools](CONTRIBUTING.md#using-ai-tools).

## License

Licensed under the [MIT License](LICENSE).
