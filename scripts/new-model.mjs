#!/usr/bin/env node
// Scaffolds a model on the shared shells (AGENTS.md, "Repository"): templates/model copied to
// models/<name>, its name filled in, and its crates added to the workspace's members and
// dependencies after the last model's.
//
//   node scripts/new-model.mjs <name>    # kebab-case: `whisper-tiny`

import { cpSync, existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const KINDS = ["model", "compile", "runtime", "cli"];
const name = process.argv[2] ?? "";
if (!/^[a-z][a-z0-9]*(-[a-z0-9]+)*$/.test(name)) fail("usage: new-model.mjs <name>, in kebab-case");
const target = join(ROOT, "models", name);
if (existsSync(target)) fail(`models/${name} exists`);

const fill = (text) => text.replaceAll("__name__", name).replaceAll("__crate__", name.replaceAll("-", "_"));
cpSync(join(ROOT, "templates/model"), target, { recursive: true });
for (const file of readdirSync(target, { recursive: true })) {
  const path = join(target, file);
  if (file.endsWith(".rs") || file.endsWith(".toml")) writeFileSync(path, fill(readFileSync(path, "utf8")));
}

const manifestPath = join(ROOT, "Cargo.toml");
const lines = readFileSync(manifestPath, "utf8").split("\n");
const version = /^version = "(.+)"$/m.exec(lines.join("\n"))[1];
const lastIndex = (pattern) => lines.findLastIndex((line) => pattern.test(line));
const members = KINDS.map((kind) => `    "models/${name}/${kind}",`);
lines.splice(lastIndex(/^ {4}"models\//) + 1, 0, ...members);
const dependencies = [
  `${name} = { package = "fusor-ai-${name}", path = "models/${name}/model", version = "=${version}" }`,
  `${name}-compile = { package = "fusor-ai-${name}-compile", path = "models/${name}/compile", version = "=${version}" }`,
];
lines.splice(lastIndex(/^\S+ = \{ package = "fusor-ai-[^"]+", path = "models\//) + 1, 0, ...dependencies);
writeFileSync(manifestPath, lines.join("\n"));
console.log(`models/${name}: ${KINDS.join(", ")} and model.toml; fill in the model, its compiler's table and its runtime's exports`);

function fail(message) {
  console.error(message);
  process.exit(1);
}
