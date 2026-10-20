#!/usr/bin/env node
// Enforces the crate layers of AGENTS.md: hosts → models → blocks → primitives, never upwards.
// A model may depend on its own family (`ming-compile` → `ming`), never on another model; a host
// never on another host. Blocks must not name tensors or hard-code dimensions: those belong to
// a model.

import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

const LAYERS = [
  { name: "primitive", match: /^crates\/(team|kernels|bundle|compile)$/ },
  { name: "block", match: /^crates\/blocks$/ },
  { name: "model", match: /^crates\/models\/[^/]+$/ },
  { name: "host", match: /^crates\/hosts\/[^/]+$/ },
];
const BLOCK_SOURCES = "crates/blocks/src";
const LITMUS = [
  { what: "a tensor name", pattern: /"[a-z_]+\./ },
  { what: "a hard-coded dimension", pattern: /const [A-Z_]+: usize = [0-9]/ },
];

const metadata = JSON.parse(
  execFileSync("cargo", ["metadata", "--format-version", "1", "--no-deps", "--locked"], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  }),
);
const root = metadata.workspace_root;
const failures = [];

function layerOf(manifestPath) {
  const dir = relative(root, manifestPath).replace(/\/Cargo\.toml$/, "");
  const layer = LAYERS.findIndex(({ match }) => match.test(dir));
  if (layer < 0) failures.push(`${dir}: not in a layer directory (see AGENTS.md)`);
  return layer;
}

const family = (name) => name.split("-")[0];
const crates = new Map(
  metadata.packages.map((pkg) => [pkg.name, { pkg, layer: layerOf(pkg.manifest_path) }]),
);

for (const { pkg, layer } of crates.values()) {
  for (const dep of pkg.dependencies) {
    const target = crates.get(dep.name);
    if (!target || dep.kind === "dev") continue;
    const from = `${LAYERS[layer]?.name} ${pkg.name}`;
    const to = `${LAYERS[target.layer]?.name} ${dep.name}`;
    if (target.layer > layer) failures.push(`${from} depends on ${to}, a higher layer`);
    const sameLayer = target.layer === layer;
    if (sameLayer && LAYERS[layer].name === "model" && family(pkg.name) !== family(dep.name)) {
      failures.push(`${from} depends on ${to}: a model never depends on another model`);
    }
    if (sameLayer && LAYERS[layer].name === "host") {
      failures.push(`${from} depends on ${to}: a host never depends on another host`);
    }
  }
}

for (const file of readdirSync(join(root, BLOCK_SOURCES), { recursive: true })) {
  if (!file.endsWith(".rs")) continue;
  const path = join(BLOCK_SOURCES, file);
  readFileSync(join(root, path), "utf8")
    .split("\n")
    .forEach((line, index) => {
      for (const { what, pattern } of LITMUS) {
        if (pattern.test(line)) {
          failures.push(`${path}:${index + 1}: ${what} in a block belongs to a model`);
        }
      }
    });
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`layers: ${crates.size} crates, every dependency points down`);
