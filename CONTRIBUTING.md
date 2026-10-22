# Contributing

Use Rust 1.99 or later, edition 2024, Node.js 20 or later, and `just`. `unsafe` is denied
except in the modules that state why they need it (see `AGENTS.md`).

Run `just check` before considering a change complete. It formats, tests and lints the crates
natively and for `wasm32-unknown-unknown` (SIMD128 and relaxed SIMD), checks the host
package and the import-check self-test, and checks that every crate depends only on its own
layer or lower ones (`AGENTS.md`, "Repository").

A change to a primitive, a block or a model also has to run through the runtimes of every model
it reaches, because the wasm backends only execute there. A model's runtimes, page and checks
live in its own repository ([ming-wasm](https://github.com/fusor-rs/ming-wasm),
[phonon-wasm](https://github.com/fusor-rs/phonon-wasm),
[parakeet-redux-wasm](https://github.com/fusor-rs/parakeet-redux-wasm),
[realistic-vision](https://github.com/fusor-rs/realistic-vision),
[paddle-ocr-wasm](https://github.com/fusor-rs/paddle-ocr-wasm),
[oxr-wasm](https://github.com/fusor-rs/oxr-wasm)), checked out beside this one. `just integrate
<model>…` (or `--all`) fetches the model's goldens from the
[fusorlabs/fusor-ai-goldens](https://huggingface.co/datasets/fusorlabs/fusor-ai-goldens)
dataset, pinned by revision and sha256, points its product's `vendor/fusor-ai` submodule at the
current commit, builds its bundles, runtimes and page, runs its parity, Node smoke and
headless-Chrome e2e checks, restores the pin, and compares every output with the model's
baseline in `models/<model>/baseline.json`; `integration.json` says how each product is built and
checked. A change that moves no number leaves every output identical. A change meant to move
them records the new outputs with `--bless`, so the baseline's diff shows what moved. A change
to a kernel backend reaches every model whose weight formats it reads, the `formats` its
`models/<model>/model.toml` lists. Attach the report, `target/integrate/report.md`, to the pull
request.

Each model's repository runs its `just check` and `just runtimes` on GitHub through this
repository's reusable [Product workflow](.github/workflows/product.yml).

The threaded runtimes need `nightly-2025-11-15` with `rust-src`
(`rustup toolchain install nightly-2025-11-15 --component rust-src --target
wasm32-unknown-unknown`); `scripts/install-tools.sh` installs the pinned wasm-bindgen and
binaryen into the model repository's `.tools/`.

Review each cohesive implementation change independently. Check for correctness, the barrier
discipline of every new phase, unbounded work, duplicated paths, speculative abstractions,
misleading comments and tests that mirror implementation. Resolve blocking findings before
advancing.

## Releasing

Every crate shares one version: `version` under `[workspace.package]` in the root
`Cargo.toml`. A release publishes every crate to crates.io under its `fusor-ai-*` package
name; the library names stay `team`, `kernels`, `blocks`, `bundle`, `compile`, `ming`,
`ming_compile`, `phonon`, `phonon_compile`, `parakeet`, `parakeet_compile`, `realistic_vision`
and `realistic_vision_compile`. A model's CLI,
page and installer are released from its own repository.

1. Bump `version` and the `=x.y.z` pins under `[workspace.dependencies]`, run `just check`
   so `Cargo.lock` follows, and merge that to `main` with CI passing.
2. On GitHub, publish a release whose tag is the version with a `v`, such as `v0.2.0`, on
   that commit.

The [Release workflow](.github/workflows/release.yml) then runs Check, checks that the tag
matches the version, packages and verifies every crate (`cargo publish --dry-run`), and
publishes the crates that are not on crates.io at this version yet, so rerunning a release
that stopped halfway finishes it. The `crates-io` environment holds the
`CARGO_REGISTRY_TOKEN` secret; add required reviewers to it to approve each upload. Running
the workflow by hand packages and verifies the crates without publishing.

## Using AI tools

You're welcome to use AI tools, but you're responsible for everything you submit. Read and
understand every line, make sure the tests you add would fail without your change, and be ready
to explain your reasoning in review. Please don't open pull requests or issues generated
without that review.

fusor-ai's own development uses AI tools heavily too, mainly Claude Code, and the maintainer
works under the same rules. Coding agents follow `AGENTS.md`, which is also the code standard
for every change.
