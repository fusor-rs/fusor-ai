# Contributing

Use Rust 1.99 or later, edition 2024, Node.js 20 or later, and `just`. `unsafe` is denied
except in the modules that state why they need it (see `AGENTS.md`).

Run `just check` before considering a change complete. It formats, tests and lints the crates
natively and for `wasm32-unknown-unknown` (SIMD128 and relaxed SIMD), checks the host
package and the import-check self-test, and checks that every crate depends only on its own
layer or lower ones (`AGENTS.md`, "Repository").

A change to a primitive, a block or a model also has to run through the model's runtimes,
because the wasm backends only execute there: for Ming, `just ming-build` once, then
`just ming-runtimes`, `just ming-smoke`, `just ming-parity` and `just ming-e2e`. Say in the
pull request what they showed (parity numbers, real-time factor, word error rate).

The threaded runtimes need `nightly-2025-11-15` with `rust-src`
(`rustup toolchain install nightly-2025-11-15 --component rust-src --target
wasm32-unknown-unknown`); `scripts/install-tools.sh` installs the pinned wasm-bindgen and
binaryen into `.tools/` (`just tools` does both).

Review each cohesive implementation change independently. Check for correctness, the barrier
discipline of every new phase, unbounded work, duplicated paths, speculative abstractions,
misleading comments and tests that mirror implementation. Resolve blocking findings before
advancing.

## Releasing

Every crate shares one version: `version` under `[workspace.package]` in the root
`Cargo.toml`. A release publishes the library crates and the `ming` CLI to crates.io under
`fusor-ai-*` package names (their library names stay `team`, `kernels`, `blocks`, `bundle`,
`compile`, `ming` and `ming_compile`), and attaches the CLI archives, the Ming page with its
runtimes and `install.sh` to the GitHub release. `ming-runtime` is not published.

1. Bump `version` and the `=x.y.z` pins under `[workspace.dependencies]`, run `just check`
   so `Cargo.lock` follows, and merge that to `main` with CI passing.
2. On GitHub, publish a release whose tag is the version with a `v`, such as `v0.2.0`, on
   that commit.

The [Release workflow](.github/workflows/release.yml) then runs Check, checks that the tag
matches the version, and builds and tests every archive:

- `just archive` builds the `ming` CLI for the runner's target (macOS on Apple silicon, for
  every Apple silicon CPU; Linux on x86-64) and packages it as
  `ming-VERSION-TARGET.tar.gz`, holding `bin/ming`, `LICENSE` and `README.md`, with a
  `.sha256` file.
- `just archive-check` installs that archive with [install.sh](install.sh) from the local
  files (`FUSOR_AI_DOWNLOAD_BASE`), checks the installed version, and checks that a corrupt
  download is refused.
- `just ming-web-archive` builds the four runtimes through the import check and packages the
  page with them, `serve.mjs`, `fetch-checkpoint.sh` and Ming's README as
  `ming-web-VERSION.tar.gz`, checking that every runtime is in it.

It attaches the archives to the release and publishes the crates that are not on crates.io
at this version yet, so rerunning a release that stopped halfway finishes it. The
`crates-io` environment holds the `CARGO_REGISTRY_TOKEN` secret; add required reviewers to it
to approve each upload. `install.sh` is attached last: its presence marks the release
complete. Running the workflow by hand builds and tests the archives without publishing.
