# The recipes every product repository shares. A product's justfile starts with
#
#   import 'vendor/fusor-ai/just/product.just'
#   name := "paddle-ocr"
#
# where `name` names its model in fusor-ai, `models/<name>`, whose crates `<name>-runtime` and
# `<name>-cli` are built in fusor-ai's workspace into the product's `target/`. It adds its own
# `host` (the syntax checks of its scripts), `model` (the bundle in dist/model), `fixtures` (the
# fixed inputs its checks read; an empty recipe when there are none) and whatever else is its
# own. A runtime that needs more shared memory than build-runtime.sh's 2 GB sets
# `export MAX_MEMORY := "…"`.

fusor_ai := "vendor/fusor-ai"
model := "dist/model"
app := "app"
engine := "app/engine/dist"
fusor := justfile_directory() / ".tools/bin/fusor"
fusor_version := "0.1.9"
port := env_var_or_default("PORT", "8787")
nice := "nice -n 19"
threads_toolchain := "nightly-2025-11-15"
wasm_bindgen_version := "0.2.129"
variants := "web-single web-single-relaxed web-threaded web-threaded-relaxed"
glue := replace(name, "-", "_") + "_runtime"
version := shell('v=$(cargo pkgid --manifest-path "$1/Cargo.toml" -p "$2"); echo "${v##*[#@]}"', fusor_ai, name + "-cli")

default:
    @just --list

# The runtime and CLI are checked in fusor-ai with the model they carry; a product checks its page.
check: lint host

lint:
    cargo fmt --manifest-path {{app}}/Cargo.toml -- --check
    cargo clippy --manifest-path {{app}}/Cargo.toml --target wasm32-unknown-unknown --locked -- -D warnings

build: tools native model runtimes page

# The threads toolchain, and the pinned wasm-bindgen, binaryen and fusor CLI (into .tools/).
tools:
    test -f {{fusor_ai}}/Cargo.toml || git submodule update --init
    rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
    rustup run {{threads_toolchain}} rustc --version >/dev/null 2>&1 || rustup toolchain install {{threads_toolchain}} --profile minimal --component rust-src --target wasm32-unknown-unknown --no-self-update
    {{fusor_ai}}/scripts/install-tools.sh {{wasm_bindgen_version}} > /dev/null
    {{fusor}} --version 2>/dev/null | grep -qx 'fusor {{fusor_version}}' || cargo install --root .tools fusor-cli --version {{fusor_version}} --locked
    cd {{app}} && {{fusor}} install

native:
    cd {{fusor_ai}} && {{nice}} cargo build --release -p {{name}}-cli --target-dir {{justfile_directory()}}/target

# The four browser runtimes, fusor-ai's loader and worker kit next to them, and the
# sealed-runtime check.
runtimes: tools
    for variant in {{variants}}; do {{nice}} {{fusor_ai}}/scripts/build-runtime.sh {{name}}-runtime $variant {{engine}} || exit 1; done
    mkdir -p {{engine}}/host
    cp {{fusor_ai}}/packages/host/loader.js {{fusor_ai}}/packages/host/worker.js {{engine}}/host/
    for variant in {{variants}}; do \
      flags=""; case $variant in *threaded*) flags="--threaded";; esac; case $variant in *relaxed*) ;; *) flags="$flags --no-relaxed";; esac; \
      node {{fusor_ai}}/scripts/check-wasm-imports.mjs {{engine}}/$variant/{{glue}}_bg.wasm --glue {{engine}}/$variant/{{glue}}.js $flags || exit 1; \
    done

# The page, a fusor app, with the runtimes, into app/dist.
page:
    cd {{app}} && {{nice}} {{fusor}} build

serve:
    @echo "open http://localhost:{{port}}/"
    node {{fusor_ai}}/scripts/serve.mjs --port {{port}} --mount /={{app}}/dist --mount /model/={{model}}

# The bundle and the runtimes are rebuilt first (seconds when nothing changed): a runtime only
# loads the bundle compiled with the same fusor-ai.
smoke: model runtimes fixtures
    node scripts/node-smoke.mjs {{model}} --build web-single
    node scripts/node-smoke.mjs {{model}} --build web-single-relaxed

e2e: model runtimes page fixtures
    cd e2e && npm ci --no-audit --no-fund --silent && npx playwright install chromium
    PORT={{port}} node e2e/run.mjs
