default:
    @just --list

check: lint wasm host

lint:
    cargo fmt --all -- --check
    cargo test --workspace --features fusor-ai-team/threads --locked
    cargo clippy --workspace --all-targets --features fusor-ai-team/threads --locked -- -D warnings
    cargo doc --workspace --no-deps --locked

# The kernels compile a different backend per target feature set: lint each.
wasm:
    cargo clippy --workspace --target wasm32-unknown-unknown --locked -- -D warnings
    cargo clippy --workspace --target wasm32-unknown-unknown --locked --target-dir target/relaxed \
      --config 'target.wasm32-unknown-unknown.rustflags=["-C", "target-feature=+simd128,+relaxed-simd"]' -- -D warnings

host:
    node --check packages/host/loader.js
    node --check packages/host/compute.js
    node --check packages/host/thread-host.js
    node --check scripts/serve.mjs
    bash -n scripts/build-runtime.sh
    node scripts/check-wasm-imports.mjs --selftest
