#!/usr/bin/env node
// Enforces the crate layers of AGENTS.md: models → blocks → primitives, never upwards. A model's
// crates may depend on each other (`models/ming/cli` → `models/ming/model`), never on another
// model. Blocks must not name tensors or hard-code dimensions: those belong to a model.

import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

const LAYERS = [
  { name: "primitive", match: /^crates\/(team|kernels|media|bundle|compile|runtime|harness)$/ },
  { name: "block", match: /^crates\/blocks$/ },
  { name: "model", match: /^models\/[^/]+\/(model|compile|runtime|cli)$/ },
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

const directoryOf = (pkg) => relative(root, pkg.manifest_path).replace(/\/Cargo\.toml$/, "");

function layerOf(dir) {
  const layer = LAYERS.findIndex(({ match }) => match.test(dir));
  if (layer < 0) failures.push(`${dir}: not in a layer directory (see AGENTS.md)`);
  return layer;
}

// `models/ming/compile` belongs to the `ming` family.
const familyOf = (dir) => dir.split("/")[1];
const crates = new Map(
  metadata.packages.map((pkg) => {
    const dir = directoryOf(pkg);
    return [pkg.name, { pkg, layer: layerOf(dir), family: familyOf(dir) }];
  }),
);

for (const { pkg, layer, family } of crates.values()) {
  for (const dep of pkg.dependencies) {
    const target = crates.get(dep.name);
    if (!target || dep.kind === "dev") continue;
    const from = `${LAYERS[layer]?.name} ${pkg.name}`;
    const to = `${LAYERS[target.layer]?.name} ${dep.name}`;
    if (target.layer > layer) failures.push(`${from} depends on ${to}, a higher layer`);
    const sameLayer = target.layer === layer;
    if (sameLayer && LAYERS[layer].name === "model" && family !== target.family) {
      failures.push(`${from} depends on ${to}: a model never depends on another model`);
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
