default:
    @just --list

check: lint wasm host layers

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
    for file in packages/host/*.js scripts/*.mjs; do node --check "$file" || exit 1; done
    for file in scripts/*.sh; do bash -n "$file" || exit 1; done
    for file in models/*/reference/*.py; do python3 -m py_compile "$file" || exit 1; done
    node scripts/check-wasm-imports.mjs --selftest

# Models → blocks → primitives (AGENTS.md, "Repository").
layers:
    node scripts/check-layers.mjs

# Builds the models in their product checkouts and compares every output with its baseline
# (scripts/integrate.mjs): `just integrate ming oxr`, `just integrate --all`.
integrate *args:
    node scripts/integrate.mjs {{args}}
