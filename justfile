default:
    @just --list

check: lint wasm host layers

lint:
    cargo fmt --all -- --check
    cargo test --workspace --features team/threads --locked
    cargo clippy --workspace --all-targets --features team/threads --locked -- -D warnings
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

# Hosts → models → blocks → primitives (AGENTS.md, "Layers").
layers:
    node scripts/check-layers.mjs

# Ming-3.0-tts-0.5B (crates/models/ming): text-to-speech in the browser.
# `just ming-build` then `just ming-serve`; docs/ming/README.md describes it.

ming_checkpoint := env_var_or_default("CHECKPOINT", ".cache/Ming-3.0-tts-0.5B")
ming_reference := env_var_or_default("REFERENCE", "../ming-05b-experiment")
ming_profile := env_var_or_default("PROFILE", "mix")
ming_model := "dist/ming/model"
ming_web := "packages/ming-web"
port := env_var_or_default("PORT", "8787")
nice := "nice -n 19"
threads_toolchain := "nightly-2025-11-15"
wasm_bindgen_version := "0.2.129"
variants := "web-single web-single-relaxed web-threaded web-threaded-relaxed"

ming-build: tools ming-native ming-model ming-runtimes

tools:
    rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
    rustup run {{threads_toolchain}} rustc --version >/dev/null 2>&1 || rustup toolchain install {{threads_toolchain}} --profile minimal --component rust-src --target wasm32-unknown-unknown --no-self-update
    scripts/install-tools.sh {{wasm_bindgen_version}} > /dev/null

ming-native:
    {{nice}} cargo build --release -p ming-cli

# Downloads the pinned checkpoint (3 GB) unless it is already there, then compiles the bundle.
ming-model: ming-native
    scripts/ming/fetch-checkpoint.sh {{ming_checkpoint}}
    target/release/ming compile {{ming_checkpoint}} {{ming_model}} --profile {{ming_profile}}

# The four browser runtimes, the host package next to them, and the sealed-runtime check.
ming-runtimes: tools
    for variant in {{variants}}; do {{nice}} scripts/build-runtime.sh ming-runtime $variant {{ming_web}}/dist || exit 1; done
    mkdir -p {{ming_web}}/dist/host
    cp packages/host/loader.js packages/host/compute.js {{ming_web}}/dist/host/
    for variant in {{variants}}; do \
      flags=""; case $variant in *threaded*) flags="--threaded";; esac; case $variant in *relaxed*) ;; *) flags="$flags --no-relaxed";; esac; \
      node scripts/check-wasm-imports.mjs {{ming_web}}/dist/$variant/ming_runtime_bg.wasm --glue {{ming_web}}/dist/$variant/ming_runtime.js $flags || exit 1; \
    done

ming-serve:
    node scripts/serve.mjs --port {{port}} --mount /={{ming_web}} --mount /model/={{ming_model}}

# Golden tensors from the fp32 PyTorch reference (see scripts/ming/dump_goldens.py).
ming-goldens:
    {{ming_reference}}/.venv/bin/python scripts/ming/dump_goldens.py --reference {{ming_reference}} --out fixtures/goldens/ming/ref_f32_A.safetensors

# Every component against the goldens, with the f32 bundle (4.3 GB, native only).
ming-parity: ming-native
    test -f dist/ming/model-f32/manifest.json || target/release/ming compile {{ming_checkpoint}} dist/ming/model-f32 --profile f32
    target/release/ming parity dist/ming/model-f32 fixtures/goldens/ming/ref_f32_A.safetensors

ming-bench: ming-native
    target/release/ming bench {{ming_model}}

ming-smoke:
    node scripts/ming/node-smoke.mjs {{ming_model}} --build web-single
    node scripts/ming/node-smoke.mjs {{ming_model}} --build web-single-relaxed

ming-e2e:
    cd e2e && npm install --no-audit --no-fund --silent && npx playwright install chromium
    PORT={{port}} node e2e/ming.mjs
