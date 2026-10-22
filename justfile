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
    for file in packages/host/*.js packages/e2e/*.mjs scripts/*.mjs; do node --check "$file" || exit 1; done
    for file in scripts/*.sh; do bash -n "$file" || exit 1; done
    for file in models/*/reference/*.py; do python3 -m py_compile "$file" || exit 1; done
    node scripts/check-wasm-imports.mjs --selftest

# Models → blocks → primitives (AGENTS.md, "Repository").
layers:
    node scripts/check-layers.mjs

# A new model on the shared shells, in models/<name>: `just new-model whisper-tiny`.
new-model name:
    node scripts/new-model.mjs {{name}}

# A new product repository for a model, in <directory>, with fusor-ai as its submodule.
new-product name directory:
    node scripts/new-product.mjs {{name}} {{directory}}

# Builds the models in their product checkouts and compares every output with its baseline
# (scripts/integrate.mjs): `just integrate ming oxr`, `just integrate --all`.
integrate *args:
    node scripts/integrate.mjs {{args}}
