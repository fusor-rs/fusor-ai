#!/usr/bin/env bash
# Builds one browser runtime variant of a wasm-bindgen `cdylib` crate into <dist>/<variant>/:
# cargo, wasm-bindgen (plus fusor-ai's page-owned thread host for threaded variants), wasm-opt.
#
#   scripts/build-runtime.sh <package> <variant> <dist>
#
# Variants: web-single (SIMD128), web-single-relaxed (+ relaxed SIMD), web-threaded and
# web-threaded-relaxed (shared memory and Rayon workers: the `threads` feature of <package>,
# built with THREADS_TOOLCHAIN, build-std and the atomics flags fusor pins for its workers).
# The crate is built with the Cargo profile PROFILE (default `wasm`), each variant in its own
# target directory. TOOLS holds wasm-bindgen and wasm-opt (scripts/install-tools.sh).
set -euo pipefail
PACKAGE="$1"
VARIANT="$2"
DIST="$3"
TOOLS="${TOOLS:-.tools/bin}"
PROFILE="${PROFILE:-wasm}"
THREADS_TOOLCHAIN="${THREADS_TOOLCHAIN:-nightly-2025-11-15}"
MAX_MEMORY="${MAX_MEMORY:-2147483648}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LIB="${PACKAGE//-/_}"

case "$VARIANT" in
  web-single | web-single-relaxed | web-threaded | web-threaded-relaxed) ;;
  *) echo "build-runtime.sh: unknown variant $VARIANT" >&2; exit 2 ;;
esac

flags=()
cargo=(cargo)
args=(build -p "$PACKAGE" --target wasm32-unknown-unknown --profile "$PROFILE" --target-dir "target/$VARIANT")
opt=(-O3 --enable-simd --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int
  --enable-mutable-globals --enable-reference-types --enable-multivalue)
if [[ "$VARIANT" == *threaded* ]]; then
  cargo=(cargo "+$THREADS_TOOLCHAIN")
  # The crates' rust-version (1.99) is what their aarch64 `sdot` kernels need; the wasm path
  # builds with the pinned nightly, which predates it.
  args+=(--features threads -Z build-std=std,panic_abort --ignore-rust-version)
  flags+=('"-C", "target-feature=+atomics,+bulk-memory"' '"-C", "link-arg=--shared-memory"'
    "\"-C\", \"link-arg=--max-memory=$MAX_MEMORY\"" '"-C", "link-arg=--import-memory"'
    '"-C", "link-arg=--export=__wasm_init_tls"' '"-C", "link-arg=--export=__tls_size"'
    '"-C", "link-arg=--export=__tls_align"' '"-C", "link-arg=--export=__tls_base"')
  opt+=(--enable-threads)
fi
if [[ "$VARIANT" == *relaxed* ]]; then
  flags+=('"-C", "target-feature=+relaxed-simd"')
  opt+=(--enable-relaxed-simd)
fi
if ((${#flags[@]})); then
  joined="$(IFS=,; echo "${flags[*]}")"
  cargo+=(--config "target.wasm32-unknown-unknown.rustflags=[$joined]")
fi

"${cargo[@]}" "${args[@]}"
out="$DIST/$VARIANT"
rm -rf "$out"
"$TOOLS/wasm-bindgen" --target web --no-typescript --out-dir "$out" --out-name "$LIB" \
  "target/$VARIANT/wasm32-unknown-unknown/$PROFILE/$LIB.wasm"
if [[ "$VARIANT" == *threaded* ]]; then
  for helper in "$out"/snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js; do
    cp "$HERE/../packages/host/thread-host.js" "$helper"
  done
fi
"$TOOLS/wasm-opt" "${opt[@]}" "$out/${LIB}_bg.wasm" -o "$out/${LIB}_bg.wasm.opt"
mv "$out/${LIB}_bg.wasm.opt" "$out/${LIB}_bg.wasm"
