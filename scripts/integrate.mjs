#!/usr/bin/env node
// Integration: builds each model in its product checkout against this fusor-ai commit, and
// compares everything it produces with the model's baseline, `integration/<model>.json`: the
// bundles' manifests, the parity lines, and the smoke and e2e outputs and metrics. A refactor
// must match it exactly; `--bless` records a change meant to move it.
//
//   node scripts/integrate.mjs <model>… [--bless] [--commit <rev>]
//   node scripts/integrate.mjs --all
//
// `--commit` builds another fusor-ai commit than the one checked out, such as a product's pin.
// The products are checkouts beside fusor-ai (integration.json). Each product's
// `vendor/fusor-ai` is pointed at this commit for the run and restored to its pin afterwards.
// The report and each model's outputs and log go to target/integrate/.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { parseArgs } from "node:util";

const ROOT = git(process.cwd(), "rev-parse", "--show-toplevel");
const CONFIG = JSON.parse(readFileSync(join(ROOT, "integration.json"), "utf8"));
const BASELINES = join(ROOT, "integration");
const RUNS = join(ROOT, "target/integrate");
const PLATFORM = `${process.platform}-${process.arch}`;
// `CER 0.0123 (3/244)`, `WER 0.0000 (0/52)`, `WER=0.05`: the metrics the products' checks print.
const METRIC = /\b(?:CER|WER)[ =][0-9.]+(?: \(\d+\/\d+\))?/g;
// The e2e runs prefix their files with the run's time: `2026-10-22T11-02-33-123Z-book.txt`.
const STAMP = /^\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d-\d+Z-?/;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { all: { type: "boolean" }, bless: { type: "boolean" }, commit: { type: "string" } },
});
const models = values.all ? Object.keys(CONFIG) : positionals;
const unknown = models.filter((model) => !(model in CONFIG));
if (models.length === 0 || unknown.length > 0) {
  fail(`expected models from integration.json (${Object.keys(CONFIG).join(", ")}) or --all`);
}
if (!values.commit && git(ROOT, "status", "--porcelain", "--untracked-files=no") !== "") {
  fail("fusor-ai has uncommitted changes: commit them, the products build from a commit");
}
const commit = git(ROOT, "rev-parse", "--verify", `${values.commit ?? "HEAD"}^{commit}`);

const sections = [];
let identical = true;
for (const model of models) {
  let record;
  try {
    record = integrate(model, CONFIG[model]);
  } catch (error) {
    identical = false;
    sections.push(section(model, [error.message]));
    continue;
  }
  const baselinePath = join(BASELINES, `${model}.json`);
  if (values.bless) {
    mkdirSync(BASELINES, { recursive: true });
    writeFileSync(baselinePath, `${JSON.stringify(record, null, 2)}\n`);
  }
  const baseline = existsSync(baselinePath) ? JSON.parse(readFileSync(baselinePath, "utf8")) : null;
  const differences = baseline ? compare(baseline, record) : ["no baseline: run with --bless"];
  identical &&= differences.length === 0;
  sections.push(section(model, differences));
}
const report = [`# Integration of fusor-ai ${commit}`, "", `Platform: ${PLATFORM}.`, "", ...sections];
writeFileSync(join(RUNS, "report.md"), `${report.join("\n")}\n`);
console.log(`${identical ? "every output identical" : "outputs differ"}: ${relative(process.cwd(), join(RUNS, "report.md"))}`);
process.exit(identical ? 0 : 1);

/** Builds `model` in its product at this commit and records what it produced. */
function integrate(model, spec) {
  const product = join(dirname(ROOT), spec.product);
  const vendor = join(product, "vendor/fusor-ai");
  if (git(product, "status", "--porcelain") !== "") {
    throw new Error(`${spec.product} has uncommitted changes: commit or stash them first`);
  }
  const pinned = git(product, "rev-parse", "HEAD:vendor/fusor-ai");
  const run = join(RUNS, model);
  rmSync(run, { recursive: true, force: true });
  mkdirSync(run, { recursive: true });
  const log = join(run, "log.txt");
  git(vendor, "fetch", "--quiet", ROOT, commit);
  git(vendor, "checkout", "--quiet", commit);
  try {
    for (const command of spec.build) {
      const { status } = shell(model, product, command, log);
      if (status !== 0) throw new Error(`\`${command}\` failed with ${status}; see ${log}`);
    }
    return {
      platform: PLATFORM,
      bundles: Object.fromEntries(spec.bundles.map((bundle) => [bundle, sha256(join(product, bundle, "manifest.json"))])),
      parity: lines(shell(model, product, spec.parity, log).stdout),
      smoke: check(model, product, spec, "smoke", log),
      e2e: check(model, product, spec, "e2e", log),
    };
  } finally {
    git(vendor, "checkout", "--quiet", pinned);
  }
}

/** Runs a check's commands with `{out}` in each replaced by its output directory: the hashes of
 * the outputs it wrote there and the metrics it printed, whatever its exit status. */
function check(model, product, spec, name, log) {
  const out = join(RUNS, model, name);
  mkdirSync(out, { recursive: true });
  const printed = spec[name].map((command) => shell(model, product, command.replaceAll("{out}", out), log).stdout);
  const outputs = readdirSync(out, { recursive: true })
    .filter((file) => spec.outputs.some((extension) => file.endsWith(extension)))
    .sort()
    .map((file) => [join(dirname(file), file.split("/").pop().replace(STAMP, "")), sha256(join(out, file))]);
  return { outputs: Object.fromEntries(outputs), metrics: printed.join("").match(METRIC) ?? [] };
}

function compare(baseline, record) {
  const differences = [];
  const walk = (path, want, got) => {
    if (typeof want === "object" && want !== null && typeof got === "object" && got !== null) {
      for (const key of new Set([...Object.keys(want), ...Object.keys(got)])) walk([...path, key], want[key], got[key]);
    } else if (want !== got) {
      differences.push(`\`${path.join(".")}\`: baseline ${JSON.stringify(want)}, now ${JSON.stringify(got)}`);
    }
  };
  walk([], baseline, record);
  return differences;
}

function section(model, differences) {
  const verdict = differences.length === 0 ? "identical to its baseline" : `${differences.length} differences`;
  return [`## ${model}: ${verdict}`, "", ...differences.map((difference) => `- ${difference}`), ""].join("\n");
}

function shell(model, product, command, log) {
  console.log(`${model}: ${command}`);
  appendFileSync(log, `$ ${command}\n`);
  const result = spawnSync("sh", ["-c", command], { cwd: product, encoding: "utf8", maxBuffer: 1 << 30 });
  appendFileSync(log, `${result.stdout}${result.stderr}exit ${result.status}\n\n`);
  return result;
}

function lines(text) {
  return text.split("\n").filter((line) => line !== "");
}

function sha256(path) {
  return `sha256:${createHash("sha256").update(readFileSync(path)).digest("hex")}`;
}

function git(cwd, ...args) {
  const result = spawnSync("git", ["-C", cwd, ...args], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(`git ${args.join(" ")} in ${cwd}: ${result.stderr.trim()}`);
  return result.stdout.trim();
}

function fail(message) {
  console.error(`integrate: ${message}`);
  process.exit(2);
}
