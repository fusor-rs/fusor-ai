#!/usr/bin/env bash
# Installs the pinned wasm-bindgen CLI (must match the wasm-bindgen crate in Cargo.lock) and,
# optionally, binaryen's wasm-opt into .tools/bin of the working directory (the consumer's
# repository). Nothing is installed globally.
set -euo pipefail
WB_VERSION="${1:-0.2.129}"
BINARYEN_VERSION="${2:-version_133}"
ROOT="$PWD"
BIN="$ROOT/.tools/bin"
mkdir -p "$BIN"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) WB_TRIPLE=aarch64-apple-darwin; BY_TRIPLE=arm64-macos ;;
  Darwin-x86_64) WB_TRIPLE=x86_64-apple-darwin; BY_TRIPLE=x86_64-macos ;;
  Linux-x86_64) WB_TRIPLE=x86_64-unknown-linux-musl; BY_TRIPLE=x86_64-linux ;;
  Linux-aarch64) WB_TRIPLE=aarch64-unknown-linux-gnu; BY_TRIPLE=aarch64-linux ;;
  *) WB_TRIPLE=""; BY_TRIPLE="" ;;
esac

if ! "$BIN/wasm-bindgen" --version 2>/dev/null | grep -q "$WB_VERSION"; then
  echo "installing wasm-bindgen $WB_VERSION into .tools/bin"
  ok=0
  if [ -n "$WB_TRIPLE" ]; then
    for org in wasm-bindgen rustwasm; do
      url="https://github.com/$org/wasm-bindgen/releases/download/$WB_VERSION/wasm-bindgen-$WB_VERSION-$WB_TRIPLE.tar.gz"
      if curl -sSfL -o "$TMP/wb.tgz" "$url"; then
        tar -xzf "$TMP/wb.tgz" -C "$TMP" && cp "$TMP/wasm-bindgen-$WB_VERSION-$WB_TRIPLE/wasm-bindgen" "$BIN/" && ok=1 && break
      fi
    done
  fi
  if [ "$ok" != 1 ]; then
    echo "no prebuilt binary; building with cargo install (a few minutes)"
    cargo install --root "$ROOT/.tools" wasm-bindgen-cli --version "$WB_VERSION" --locked
  fi
fi
"$BIN/wasm-bindgen" --version

if [ ! -x "$BIN/wasm-opt" ] && [ -n "$BY_TRIPLE" ] && [ "${SKIP_WASM_OPT:-0}" != 1 ]; then
  echo "installing binaryen $BINARYEN_VERSION (wasm-opt) into .tools"
  base="https://github.com/WebAssembly/binaryen/releases/download/$BINARYEN_VERSION/binaryen-$BINARYEN_VERSION-$BY_TRIPLE.tar.gz"
  if curl -sSfL -o "$TMP/by.tgz" "$base" && curl -sSfL -o "$TMP/by.sha256" "$base.sha256"; then
    want="$(cut -d' ' -f1 < "$TMP/by.sha256")"
    got="$(shasum -a 256 "$TMP/by.tgz" | cut -d' ' -f1)"
    if [ "$want" = "$got" ]; then
      tar -xzf "$TMP/by.tgz" -C "$TMP"
      mkdir -p "$ROOT/.tools/lib"
      cp "$TMP/binaryen-$BINARYEN_VERSION/bin/wasm-opt" "$BIN/"
      cp -R "$TMP/binaryen-$BINARYEN_VERSION/lib/." "$ROOT/.tools/lib/" 2>/dev/null || true
    else
      echo "binaryen sha256 mismatch ($got != $want); skipping wasm-opt"
    fi
  else
    echo "could not download binaryen; wasm-opt is optional, skipping"
  fi
fi
[ -x "$BIN/wasm-opt" ] && "$BIN/wasm-opt" --version || true
