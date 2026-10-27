<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/fusor-ai-horizontal-dark.svg">
    <img src="assets/brand/fusor-ai-horizontal.svg" alt="fusor-ai" width="260">
  </picture>
</p>

# Build AI runtimes with Rust and WebAssembly

Compile a model into a small WebAssembly runtime and a bundle of quantized weights, and run it on the device's CPU. No server, no API keys, no GPU required, and no way for the runtime to reach the network.

fusor-ai is a toolkit for on-device inference: SIMD kernels for quantized weights, a thread team for WebAssembly workers, reusable model blocks, a streamed and hashed weight format, and the browser side that loads it all. It sits next to [fusor](https://github.com/fusor-rs/fusor) and uses its recipe for workers that share one WebAssembly memory.

**In development · v0.1**: this is experimental software. APIs will change; don't rely on it for production yet. See [status and known limitations](#status-and-known-limitations).

[Get started](#get-started) · [How it works](#how-it-works) · [The toolkit](#the-toolkit) · [Models](#models) · [Status](#status-and-known-limitations) · [Contributing](CONTRIBUTING.md)

## A model in one file

A model is ordinary Rust over the toolkit's blocks. Here, a Qwen2-0.5B language model that loads its weights from a bundle and runs one token through every layer:

```rust
use blocks::transformer::{Architecture, Buffers, KvCache, Mask, Shape, Span, Transformer};
use bundle::{Load, Store};
use kernels::ops::rms_norm;
use kernels::{Table, Vector};
use team::Team;

const QWEN2_0_5B: Shape = Shape {
    architecture: Architecture::Qwen2,
    hidden: 896,
    layers: 24,
    heads: 14,
    kv_heads: 2,
    head_dim: 64,
    feed_forward: 4864,
    rope_base: 1e6,
    norm_eps: 1e-6,
};
const VOCABULARY: usize = 151_936;

pub struct LanguageModel {
    embed: Table,
    body: Transformer,
    norm: Vector,
}

impl LanguageModel {
    pub fn load(store: &mut Store) -> Result<Self, bundle::Error> {
        let hidden = QWEN2_0_5B.hidden;
        Ok(Self {
            embed: Table::load(store, "embed", [VOCABULARY, hidden])?,
            body: Transformer::load(store, "llm", QWEN2_0_5B)?,
            norm: Vector::load(store, "llm.norm", hidden)?,
        })
    }

    /// The hidden state after `token` at `position`, the positions before it already cached.
    pub fn step(&self, team: &Team, cache: &KvCache, buffers: &mut Buffers, token: u32, position: usize) -> Vec<f32> {
        let hidden = QWEN2_0_5B.hidden;
        let row = &mut buffers.hidden.get_mut()[..hidden];
        self.embed.matrix().dequantize_row(token as usize, row);
        let span = Span { rows: 1, first: position, mask: Mask::Causal };
        team.run(|ctx| self.body.forward(ctx, cache, buffers, span));
        let mut state = vec![0.0; hidden];
        let last = &buffers.hidden.read()[..hidden];
        rms_norm(last, self.norm.values(), QWEN2_0_5B.norm_eps, &mut state);
        state
    }
}
```

`Transformer::load` reads the stack's tensors under the `llm` prefix and checks each against the shape. `forward` runs inside one `Team::run`: every thread takes its share of each layer's heads and columns, with barriers between the phases that depend on each other. The matrix products quantize activations to int8 on the fly and use the target's dot-product instruction: relaxed SIMD or SIMD128 in WebAssembly, `sdot` on Apple silicon, scalar code elsewhere. The same file builds natively, for single-threaded WebAssembly, and for WebAssembly workers that share memory.

## Get started

You need Rust 1.99 or newer, installed with [rustup](https://rustup.rs), Node.js 20 or newer and [`just`](https://github.com/casey/just). Threaded WebAssembly builds also need the `nightly-2026-10-20` toolchain with `rust-src`; `scripts/install-tools.sh` installs the pinned wasm-bindgen and binaryen beside it. Each release publishes the crates to crates.io (`fusor-ai-kernels`, `fusor-ai-blocks`, …); a model's app also needs this repository as a submodule for the browser host and scripts.

### Run a model

[Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B), text to speech, is the first model built with fusor-ai. Its app, [ming-wasm](https://github.com/fusor-rs/ming-wasm), uses this repository as a submodule. It needs about 4 GB of disk and 5 GB of free RAM while the model compiles:

```sh
git clone --recurse-submodules https://github.com/fusor-rs/ming-wasm && cd ming-wasm
just build    # downloads the checkpoint, compiles an 850 MB bundle, builds four browser runtimes
just serve    # http://localhost:8787/
```

Type some text and press Generate: the page speaks it, 1.5–2× faster than real time in Chrome on an Apple M4 Pro, and makes no network requests after the model has loaded.

### Build a model

A model's crates live here and its app in its own repository. `just new-model <name>` scaffolds the crates in `models/<name>` on the shared shells, and `just new-product <name> <directory>` the app's repository with fusor-ai as its submodule. The crates are [the models layer](#the-toolkit):

1. **The model** (`models/<model>/model`): its dimensions, its wiring of blocks and its sampling loop, with each component loading itself from a `bundle::Store`.
2. **The compiler** (`models/<model>/compile`): reads the checkpoint with `compile::Checkpoint`, keeps, renames and fuses the tensors the model needs, picks a format per tensor and writes the bundle with `BundleWriter`.
3. **The runtime** (`models/<model>/runtime`): a wasm-bindgen `cdylib` that takes the bundle's parts and runs inference. `scripts/build-runtime.sh` builds its four variants and `scripts/check-wasm-imports.mjs` proves each one is sealed.
4. **A native CLI** (`models/<model>/cli`) that compiles bundles, benchmarks the model and checks parity with the reference implementation.

The app depends on fusor-ai as a submodule and holds the rest:

5. **The page**: a worker that loads the runtime with `packages/host/loader.js`, and a page that hands it compute workers with `packages/host/compute.js`.

ming-wasm is a complete worked example, and [AGENTS.md](AGENTS.md#repository) says where each piece of new code goes.

## How it works

- **Ahead of time.** The compiler decides everything a framework would decide at run time: which tensors exist, their fused layout, their quantization. The runtime is model code over a handful of kernels, with no graph, no dispatcher and no dynamic shapes.
- **Quantized by block.** Weights are 7-bit (`Q7b32`) or 4-bit (`Q4b32`, the ggml Q4_0 layout) with one f32 scale per 32 values, or ternary (`T2b128`: −1, 0 or +1 as two-bit codes, one scale per 128 values); activations are quantized to int8 per scale group on the fly, and a group's integer products add up before one conversion to f32. Every stored weight code stays inside the defined range of relaxed SIMD's `i32x4.relaxed_dot_i8x16_i7x16_add_s`, so results agree across engines.
- **One thread start per component.** A `Team::run` is one Rayon broadcast; inside it, static column splits and spin barriers. Small matrices, as in autoregressive decoding or a diffusion sampler, stay fast because nothing forks per matrix product.
- **Streamed and verified.** A bundle is a manifest of parts with BLAKE3 hashes. The loader streams parts into the runtime as they download, the runtime copies each byte range straight into its tensor and checks the hash, and verified parts are cached in the browser's origin-private storage for the next visit.
- **Sealed.** A runtime imports only wasm-bindgen intrinsics, plus shared memory and the worker handles in threaded builds: `scripts/check-wasm-imports.mjs` proves it has no network, storage, clock or entropy capability.
- **Four builds, picked by the browser.** Threads when the page is cross-origin isolated, relaxed SIMD when the engine validates it, plain SIMD128 everywhere else.
- **Layers that point down.** Primitives know no architecture, blocks know no model, and models know no host. A block names no tensor and fixes no dimension, so the next model reuses it unchanged; `just check` enforces the direction.

## The toolkit

| Layer | Crate or package | Holds |
| --- | --- | --- |
| Primitives | `team` | `Team`, `Ctx` (`split`, `barrier`), `SharedBuf`, `PerThread` |
| | `kernels` | `Tensor` and its formats, the checked views `Vector` and `Table`, `matmul` over per-target backends, `ops` (RMSNorm, LayerNorm, RoPE, GELU, SiLU, ELU, HardSwish, `exp`, `erf`, softmax, attention), `Linear`, `conv` (products over the pixels of images split by rows, 2-D convolutions as such products, nearest upsampling folded in; depthwise convolutions), `QuantizedMatrix` (weights quantized at run time), `numpy` (NumPy's float32 reductions: pairwise sums, means, medians), `rng` (normals and categorical draws) |
| | `media` | `wav` (16-bit PCM WAV encoding, and decoding to mono), `image` (Pillow's bilinear and bicubic resizes of 8-bit images, bit-exact; PyTorch's bilinear and bicubic resampling of a grid of vectors, and bilinear with the corners aligned), `opencv` (OpenCV's grayscale conversion, area and linear resizes, rotations and quarter turns of 8-bit images, bit-exact with its arm64 builds) |
| | `bundle` | `Manifest`, `Store` (streamed, hashed parts; tensors and blobs), the `Load` trait that checks every tensor's shape, the `Engine` and `Stages` traits every model's session and pipeline implement, `Tokenizer` (with the `tokenizer` feature), `python` (prompt text as Python references write it: `json.dumps`, `str()`, `str.strip`) |
| | `compile` | `Checkpoint` (safetensors), `Onnx` (an ONNX graph's nodes and tensors), `read_npz` (NumPy archives), `BundleWriter`, `Profile` and `Tensors` (each tensor in its profile's format) |
| | `runtime` | The browser runtime shell: `Shell` and the `runtime!` macro (a model's engine behind wasm-bindgen: the part protocol, the panic message, JSON errors, the thread-pool export) and the deterministic getrandom backend that keeps entropy out of a runtime's imports |
| | `harness` | The native host of a model's CLI: the thread pool, the bundle loader (`load`), the `compile` command, image files, `parity` comparisons (`relative_l2`, `max_abs`, `compare`) and `bench` timings |
| Blocks | `blocks` | `transformer` (decoder stacks with a causal or sliding-window KV cache, in the Qwen2, Qwen3, Kyutai and CLIP architectures, and encoders whose rows all attend to each other), `hybrid` (Qwen3.5's hybrid decoder stacks: gated attention over a KV cache, causal or reading the whole prompt both ways, and the gated delta rule, a linear attention whose state carries from call to call and can be saved to read on again from an earlier row), `rotary` (rotary embeddings by sequence position, Qwen2-VL's multimodal sections, Qwen3-VL's interleaved ones, or a patch's row and column), `lm` (a causal language model: its stack and final norm, the head, prefill in chunks and greedy decoding; the prediction from a row of any stack), `vit` (NaViT vision encoders over one image of any size, with the int8 attention; the vision tower; the patch merger; the image and video preprocessing of vision-language models and their tokens' positions in a prompt), `reading` (the engine of a vision-language model reading an image to text, step by step), `dit` (diffusion-transformer blocks), `norm` (LayerNorm, GroupNorm), `adaln_mlp` (the adaLN residual MLP of MAR-style flow heads), `flow` (flow-matching Euler steps with classifier-free guidance), `istft` (streaming inverse STFT), `seanet` (SEANet's streaming decoder, every conv one matrix), `conformer` (NVIDIA's FastConformer encoder: the convolutional subsampler, Transformer-XL relative-position attention, the convolution module), `unet` (latent diffusion's UNet: residual blocks, spatial transformers with int8 self- and cross-attention), `vae` (the KL autoencoder's decoder), `dpm` (DPM-Solver++(2M) at Karras noise levels) |
| Models | `ming`, `ming-compile` | Ming-3.0-tts-0.5B and the compiler that writes its bundles |
| | `phonon`, `phonon-compile` | Phonon-31m-en and the compiler that writes its bundles |
| | `parakeet`, `parakeet-compile` | Moondream's Parakeet Redux and the compiler that writes its bundles |
| | `realistic-vision`, `realistic-vision-compile` | Realistic Vision V6.0 B1 (Stable Diffusion 1.5) with the SD VAE ft-MSE, and the compiler that writes its bundles |
| | `paddle-ocr`, `paddle-ocr-compile` | PaddleOCR-VL-1.6 and the compiler that writes its bundles |
| | `oxr`, `oxr-compile` | SoMark's OXR-1.0 and the compiler that writes its bundles |
| | `intern-decision`, `intern-decision-compile` | InternLM's Intern-Decision-2B and the compiler that writes its bundles |
| | `indra-ocr`, `indra-ocr-compile` | Datalab's Indra OCR and the compiler that writes its bundles from its ONNX networks and language models |
| | `d3`, `d3-compile` | vLLM Semantic Router's d3-nano and d3-lite (Decision 3.0), and the compiler that writes either's bundles |
| | `qwen3-5`, `qwen3-5-compile` | Qwen's Qwen3.5-2B and the compiler that writes its bundles |
| | `spark-x2-6`, `spark-x2-6-compile` | XHToken's Spark-X2.6-1.7B and the compiler that writes its bundles |
| Browser | `packages/host` | `loader.js` (build choice, part streaming, OPFS cache), `worker.js` (the shared half of a model's engine worker: loading with stale-part eviction, the clamp-free yield, error routing, the request count), `compute.js` (the compute-worker pool), `thread-host.js` and the `protocol.js` they share |
| | `packages/e2e` | `kit.mjs`: the shared half of a product's headless Chrome run (its flags, the local server, a fresh profile with every request recorded, the revisit check, the results file and exit status) |
| | `packages/ui` | `base.css`: the palette, layout, masthead, cards, controls and footer every product's page shares, linked before its own `app.css` |
| Tools | `scripts/` | `build-runtime.sh`, `check-wasm-imports.mjs`, `check-layers.mjs`, `serve.mjs`, `install-tools.sh`, `integrate.mjs` (every model's outputs and timings against its baseline) |
| | `tools/kernel-harness` | A page that replays the kernels with a page's run-time sizes (`node bench.mjs <module.wasm>`), for explaining kernel speed (`knowledge.md`) |

On crates.io each toolkit crate is `fusor-ai-<name>`, such as `fusor-ai-kernels`, and its library keeps the short name (`use kernels::…`). The model crates are not published.

## Models

| Model | Task | Blocks | In the browser |
| --- | --- | --- | --- |
| Ming-3.0-tts-0.5B: `ming`, `ming-compile`; app in [ming-wasm](https://github.com/fusor-rs/ming-wasm) | Text to speech, 44.1 kHz | `transformer`, `dit`, `flow`, `istft` | 1.5–2× faster than real time in Chrome on an Apple M4 Pro; 850 MB bundle |
| Phonon-31m-en: `phonon`, `phonon-compile`; app in [phonon-wasm](https://github.com/fusor-rs/phonon-wasm) | Text to speech, 24 kHz, English | `transformer`, `norm`, `adaln_mlp`, `seanet` | 9× faster than real time on one thread and 40× on eight in Chrome on an Apple M4 Pro; 46 MB bundle |
| Parakeet Redux: `parakeet`, `parakeet-compile`; app in [parakeet-redux-wasm](https://github.com/fusor-rs/parakeet-redux-wasm) | Speech to text, 16 kHz, 25 European languages, live or from a recording | `conformer` | 34× faster than real time on eight threads and 7× on one in Chrome on an Apple M4 Pro; 199 MB bundle |
| Realistic Vision V6.0 B1: `realistic-vision`, `realistic-vision-compile`; app in [realistic-vision](https://github.com/fusor-rs/realistic-vision) | Text to image, Stable Diffusion 1.5 with the SD VAE ft-MSE, up to 512 × 768 | `transformer`, `unet`, `vae`, `dpm` | A 512 × 512 image in 20 steps in 57 s on eight threads and 6 minutes on one in Chrome on an Apple M4 Pro; 1.16 GB bundle |
| PaddleOCR-VL-1.6: `paddle-ocr`, `paddle-ocr-compile`; app in [paddle-ocr-wasm](https://github.com/fusor-rs/paddle-ocr-wasm) | Document reading: text, tables, formulas, charts and seals off an image | `vit`, `rotary`, `transformer` | A formula's first token in 0.9 s and a book page's in 9.3 s on eight threads in Chrome on an Apple M4 Pro, then 270–370 tokens per second; 831 MB bundle |
| OXR-1.0: `oxr`, `oxr-compile`; app in [oxr-wasm](https://github.com/fusor-rs/oxr-wasm) | Document parsing to Markdown: a whole page, text, tables, formulas, code and chemical structures off an image | `vit`, `rotary`, `transformer` | A formula's first token in 1.3 s and a book page's in 5.3 s on eight threads in Chrome on an Apple M4 Pro, then 170–210 tokens per second; 1420 MB bundle |
| Intern-Decision-2B: `intern-decision`, `intern-decision-compile`; app in [intern-decision-wasm](https://github.com/fusor-rs/intern-decision-wasm) | Structured decisions: calibrated answers to a request's questions about a state and images, in one pass | `vit`, `hybrid`, `rotary` | Four questions about a scanned seal (755 prompt tokens, 400 of them the image's) in 3.4 s on eight threads in Chrome on an Apple M4 Pro; 2068 MB bundle |
| Indra OCR: `indra-ocr`, `indra-ocr-compile`; app in [indra-ocr-wasm](https://github.com/fusor-rs/indra-ocr-wasm) | Pages to Markdown: line detection, CTC recognition in 80+ languages with n-gram language models, math as LaTeX, reading order and layout | `transformer` | A scanned page in 0.61 s and a page of equations in 0.53 s on eight threads in Chrome on an Apple M4 Pro, each read exactly as the reference reads it; 155 MB bundle |
| Qwen3.5-2B: `qwen3-5`, `qwen3-5-compile`; app in [qwen3-5-wasm](https://github.com/fusor-rs/qwen3-5-wasm) | A conversation about text, images and videos, each reply streamed, thinking first or not | `vit`, `hybrid`, `lm`, `rotary` | A reply about a receipt (896 prompt tokens) starts in 7.0 s and writes 84 tokens/s on eight threads in Chrome on an Apple M4 Pro; a follow-up starts in 0.14 s; 2517 MB bundle |

Each model's library, compiler, runtime and CLI live in `models/<model>/`. Its page, documentation and releases live in its app's repository.

## Documentation

| Guide | Covers |
| --- | --- |
| [ming-wasm](https://github.com/fusor-rs/ming-wasm) | Running the Ming text-to-speech page and the `ming` CLI, releases, and benchmarks of the first model against PyTorch and other runtimes |
| [phonon-wasm](https://github.com/fusor-rs/phonon-wasm) | Running the Phonon text-to-speech page and the `phonon` CLI, releases and benchmarks |
| [parakeet-redux-wasm](https://github.com/fusor-rs/parakeet-redux-wasm) | Running the Parakeet Redux speech-to-text page and the `parakeet` CLI, releases and benchmarks |
| [realistic-vision](https://github.com/fusor-rs/realistic-vision) | Running the Realistic Vision text-to-image page and the `realistic-vision` CLI, releases and benchmarks |
| [paddle-ocr-wasm](https://github.com/fusor-rs/paddle-ocr-wasm) | Running the PaddleOCR-VL document-reading page and the `paddle-ocr` CLI, and benchmarks against PyTorch |
| [oxr-wasm](https://github.com/fusor-rs/oxr-wasm) | Running the OXR document-parsing page and the `oxr` CLI, and benchmarks against PyTorch |
| [intern-decision-wasm](https://github.com/fusor-rs/intern-decision-wasm) | Running the Intern-Decision page and the `intern-decision` CLI, and benchmarks against PyTorch |
| [indra-ocr-wasm](https://github.com/fusor-rs/indra-ocr-wasm) | Running the Indra OCR page and the `indra-ocr` CLI, and benchmarks against ONNX Runtime |
| [d3-wasm](https://github.com/fusor-rs/d3-wasm) | Running the Decision 3.0 page (d3-nano and d3-lite) and the `d3` CLI, and benchmarks against PyTorch |
| [qwen3-5-wasm](https://github.com/fusor-rs/qwen3-5-wasm) | Running the Qwen3.5 conversation page and the `qwen3-5` CLI, and benchmarks |
| [Repository rules](AGENTS.md) | The layers, where new code goes, and the rules every change follows |
| [Contributing](CONTRIBUTING.md) | Setup, checks and releasing |
| [Brand](assets/brand/BRANDING.md) | The logo, its construction and its colors |

## Status and known limitations

fusor-ai is v0.1 and experimental. Until 1.0, each minor release may change the crates' APIs and the bundle format.

### Platform status

| Platform or capability | Status |
| --- | --- |
| macOS on Apple silicon (NEON `sdot`) | CI gate |
| Linux on x86-64 (scalar kernels) | CI gate |
| Runtime builds and the sealed-import check | CI gate in each model's repository |
| Chrome and Edge, threads and relaxed SIMD | Headless Chrome end-to-end check in each model's repository, run before each change to a kernel, block or model it uses |
| Browsers without relaxed SIMD (Safari, older Firefox) | SIMD128 builds, exercised in Node by each model's repository |
| Pages without cross-origin isolation | Single-threaded builds |
| x86 SIMD natively | Scalar kernels only |
| WebGPU | Not implemented |

**Not supported yet:**

- Models above roughly 1.5B parameters: wasm32 memory is capped at 4 GiB (threaded builds reserve 2 GiB), and at that scale WebGPU runtimes are much faster.
- Loading an architecture from a checkpoint alone. Each model is its own crate, written against the blocks.
- GGUF checkpoints; only safetensors (bf16 or f32) are read.
- Signed bundles. Parts are content-hashed, but the manifest is not signed yet.

## How is this different from ONNX Runtime Web and transformers.js?

[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) and [transformers.js](https://github.com/huggingface/transformers.js) run a model graph through a general-purpose executor shipped as a large WebAssembly module, and support many architectures without per-model work. fusor-ai compiles each model to its own small runtime: there is no graph to interpret, so small-batch work such as a diffusion sampler's ten passes per audio frame does not pay per-operator dispatch, and the runtime's import section proves what it cannot do. The cost is per-model engineering, which the blocks keep small.

## How is this different from WebLLM?

[WebLLM](https://github.com/mlc-ai/web-llm) runs large language models on the GPU through WebGPU and is much faster for models of several billion parameters. fusor-ai targets models up to about 1.5B that must run on any device, including those without WebGPU, offline and with an auditable runtime.

## How is this different from Candle and Burn?

[Candle](https://github.com/huggingface/candle) and [Burn](https://github.com/tracel-ai/burn) are Rust machine-learning frameworks with a tensor type, operators dispatched at run time and several backends, and both compile to WebAssembly. fusor-ai has no tensor framework: model code calls a few kernels directly, splits each one across a thread team that shares WebAssembly memory, and reads weights in formats chosen ahead of time for the browser's SIMD instructions.

## Contributing

fusor-ai is an open source project in early development. The [contributing guide](CONTRIBUTING.md) covers the setup, checks and workflow. The toolkit lives in `crates/`, `packages/` and `scripts/`, and the models in `models/`.

fusor-ai is built with the help of AI tools, mainly Claude Code, under the same rules the project asks of contributors: see [Using AI tools](CONTRIBUTING.md#using-ai-tools).

## License

Licensed under the [MIT License](LICENSE).
