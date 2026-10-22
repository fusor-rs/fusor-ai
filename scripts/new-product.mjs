#!/usr/bin/env node
// Scaffolds a product repository for a model (templates/product): its page on fusor-ai's page
// shell and worker kit, its e2e run on the e2e kit, its justfile importing product.just, its
// workflows and Dependabot, a short AGENTS.md and a CLAUDE.md importing fusor-ai's, and
// fusor-ai as the `vendor/fusor-ai` submodule at this checkout's commit. It commits nothing.
//
//   node scripts/new-product.mjs <name> <directory>    # <name>: the model, `models/<name>`

import { execFileSync } from "node:child_process";
import { cpSync, existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const REMOTE = "https://github.com/fusor-rs/fusor-ai.git";
const [name = "", directory] = process.argv.slice(2);
if (!/^[a-z][a-z0-9]*(-[a-z0-9]+)*$/.test(name) || !directory) {
  fail("usage: new-product.mjs <name> <directory>, the name in kebab-case");
}
const target = resolve(directory);
if (existsSync(target)) fail(`${target} exists`);

const fill = (text) => text.replaceAll("__name__", name).replaceAll("__crate__", name.replaceAll("-", "_"));
cpSync(join(ROOT, "templates/product"), target, { recursive: true });
for (const file of readdirSync(target, { recursive: true })) {
  const path = join(target, file);
  if (statSync(path).isFile()) writeFileSync(path, fill(readFileSync(path, "utf8")));
}

const run = (command, ...args) => execFileSync(command, args, { cwd: target, stdio: "inherit" });
const commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
run("git", "init", "--quiet", "--initial-branch=main");
run("git", "submodule", "add", "--quiet", REMOTE, "vendor/fusor-ai");
run("git", "-C", "vendor/fusor-ai", "checkout", "--quiet", commit);
// The template's page lock names its package `__name__-app`; cargo reorders it under the new name.
execFileSync("cargo", ["metadata", "--format-version", "1", "--manifest-path", "app/Cargo.toml"], {
  cwd: target,
  stdio: "ignore",
});
console.log(`${target}: ${name}'s product on fusor-ai ${commit.slice(0, 7)}; review it, then commit`);

function fail(message) {
  console.error(message);
  process.exit(1);
}
