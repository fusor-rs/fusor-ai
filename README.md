# Compile AI models into sealed WebAssembly runtimes

Ship a model as static files. The browser downloads a bundle of quantized weights and a
3 MB runtime compiled from Rust, and runs inference on the device's CPU with WebAssembly SIMD
and threads. No server, no API keys, no WebGPU required, and no way for the runtime to reach
the network.

fusor-ai is the toolkit behind [ming-wasm](https://github.com/fusor-rs/ming-wasm), which runs
the Ming-3.0-tts-0.5B speech model in Chrome faster than real time, and the pattern behind
SealedKB's in-browser question answering. It sits next to
[fusor](https://github.com/fusor-rs/fusor): it reuses fusor's thread recipe for workers that
share one WebAssembly memory.

**In development · v0.1**: this is experimental software. APIs will change; don't rely on it for
production yet. See [status and known limitations](#status-and-known-limitations).

[Get started](#get-started) · [How it works](#how-it-works) · [Crates](#crates) ·
[Examples](#examples) · [Status](#status-and-known-limitations) · [Contributing](CONTRIBUTING.md)

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

You need Rust 1.99 or newer ([rustup](https://rustup.rs)), Node.js 20 or newer, and, for
threaded runtimes, the `nightly-2025-11-15` toolchain with `rust-src`. fusor-ai is not on
crates.io yet: add it to your model's repository as a submodule, so the crates and the host
scripts move together.

```sh
git submodule add https://github.com/fusor-rs/fusor-ai vendor/fusor-ai
```

```toml
# Cargo.toml of your workspace
[workspace]
exclude = ["vendor"]

[workspace.dependencies]
team = { path = "vendor/fusor-ai/crates/team" }
kernels = { path = "vendor/fusor-ai/crates/kernels" }
nn = { path = "vendor/fusor-ai/crates/nn" }
bundle = { path = "vendor/fusor-ai/crates/bundle" }
compile = { path = "vendor/fusor-ai/crates/compile" }
```

A model built on fusor-ai has three parts, as in ming-wasm:

1. **A compiler** (native): reads the checkpoint with `compile::Checkpoint`, keeps,
   renames and fuses the tensors it needs, picks a dtype per tensor
   (`kernels::Tensor::encode`) and writes the bundle with `BundleWriter`.
2. **A runtime crate** (`cdylib`): builds the model from a `bundle::Store`, exposes
   `write_part`, `end_part` and `restart_part` plus its own inference calls through
   wasm-bindgen, and has a `threads` feature that enables `team/threads` and
   re-exports `wasm_bindgen_rayon::init_thread_pool`.
3. **A page**: a module worker that calls `startRuntime` and `loadParts` from
   `packages/host/loader.js`, and a page that hands compute workers to
   `packages/host/compute.js`.

Build the four runtime variants and serve the page:

```sh
vendor/fusor-ai/scripts/install-tools.sh            # pinned wasm-bindgen and wasm-opt into .tools/
for variant in web-single web-single-relaxed web-threaded web-threaded-relaxed; do
  vendor/fusor-ai/scripts/build-runtime.sh my-model-runtime $variant web/dist
done
node vendor/fusor-ai/scripts/check-wasm-imports.mjs web/dist/web-single/my_model_runtime_bg.wasm --no-relaxed
node vendor/fusor-ai/scripts/serve.mjs --mount /=web --mount /model/=dist/model
```

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

## Crates

| Crate or package | Holds |
| --- | --- |
| `team` | `Team`, `Ctx` (`split`, `barrier`), `SharedBuf`, `PerThread` |
| `kernels` | `Tensor` and its formats, `matmul` and its per-target backends, `ops` (RMSNorm, RoPE, GELU, SiLU, `exp`, attention), `Linear` |
| `nn` | `qwen2`: the Qwen2 decoder stack with a causal or sliding-window KV cache |
| `bundle` | `Manifest`, `Store` (streaming, hashed parts; tensors and blobs) |
| `compile` | `Checkpoint` (safetensors), `BundleWriter` |
| `packages/host` | `loader.js` (build choice, part streaming, OPFS cache), `compute.js` (the compute-worker pool), `thread-host.js` |
| `scripts/` | `build-runtime.sh`, `check-wasm-imports.mjs`, `serve.mjs`, `install-tools.sh` |

## Examples

- [ming-wasm](https://github.com/fusor-rs/ming-wasm): Ming-3.0-tts-0.5B text-to-speech (a Qwen2
  LLM, a diffusion-transformer flow head and a streaming VAE vocoder) in the browser, with
  parity checks against the PyTorch reference.

## Status and known limitations

fusor-ai is v0.1 and experimental. Until 1.0, each minor release may change the crates' APIs and
the bundle format.

| Platform or capability | Status |
| --- | --- |
| Chrome / Edge, threads + relaxed SIMD | Headless Chrome end-to-end gate in ming-wasm |
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
