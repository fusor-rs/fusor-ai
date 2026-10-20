# Contributing

Use Rust 1.99 or later, edition 2024, Node.js 20 or later, and `just`. `unsafe` is denied
except in the modules that state why they need it (see `AGENTS.md`).

Run `just check` before considering a change complete. It formats, tests and lints the crates
natively and for `wasm32-unknown-unknown` (SIMD128 and relaxed SIMD), and checks the host
package and the import-check self-test.

A kernel, team or bundle change also has to be run through a consumer, because the wasm
backends only execute in a runtime: in `ming-wasm`, point its `vendor/fusor-ai` submodule at
your change and run `make wasm`, `make smoke`, `make parity` and `make e2e`. Say in the pull
request what they showed (parity numbers, real-time factor, word error rate).

The threaded runtimes need `nightly-2025-11-15` with `rust-src`
(`rustup toolchain install nightly-2025-11-15 --component rust-src --target
wasm32-unknown-unknown`); `scripts/install-tools.sh` installs the pinned wasm-bindgen and
binaryen into the consumer's `.tools/`.

Review each cohesive implementation change independently. Check for correctness, the barrier
discipline of every new phase, unbounded work, duplicated paths, speculative abstractions,
misleading comments and tests that mirror implementation. Resolve blocking findings before
advancing.

## Releasing

Every crate shares one version: `version` under `[workspace.package]` in the root
`Cargo.toml`. The crates are not published to crates.io yet; consumers pin a commit of this
repository (a git submodule, so the Rust crates and the host scripts move together). To
release, bump the version, run `just check`, merge to `main` and tag the commit `v<version>`.

## Using AI tools

You're welcome to use AI tools, but you're responsible for everything you submit. Read and
understand every line, make sure the tests you add would fail without your change, and be ready
to explain your reasoning in review. Please don't open pull requests or issues generated
without that review.

fusor-ai's own development uses AI tools heavily too, mainly Claude Code, and the maintainer
works under the same rules. Coding agents follow `AGENTS.md`, which is also the code standard
for every change.
