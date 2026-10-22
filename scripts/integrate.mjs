#!/usr/bin/env node
// Integration: builds each model in its product checkout against this fusor-ai commit, and
// compares everything it produces with the model's baseline, `models/<model>/baseline.json`: the
// bundles' manifests, the parity lines, and the smoke and e2e outputs and metrics. A refactor
// must match it exactly; `--bless` records a change meant to move it.
//
// It also times the model: its native bench, and the figures its page measured in the e2e runs.
// The report sets them beside the baseline's and flags those that moved beyond their noise;
// timings never fail the run.
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
import { basename, dirname, join, relative } from "node:path";
import { parseArgs } from "node:util";

const ROOT = git(process.cwd(), "rev-parse", "--show-toplevel");
const CONFIG = JSON.parse(readFileSync(join(ROOT, "integration.json"), "utf8"));
const RUNS = join(ROOT, "target/integrate");
const PLATFORM = `${process.platform}-${process.arch}`;
// `CER 0.0123 (3/244)`, `live WER 0.0512`, `WER=0.05`: the metrics the products' checks print,
// with the word naming the case when one precedes them.
const METRIC = /(?:\b[a-z]+ )?\b(?:CER|WER)[ =][0-9.]+(?: \(\d+\/\d+\))?/g;
// The e2e runs prefix their files with the run's time: `2026-10-22T11-02-33-123Z-book.txt`.
const STAMP = /^\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d-\d+Z-?/;
// A bench line's label and its first time: `  encode   12.34 ms/frame`,
// `threads 8: prefill 12 tokens 345 ms`, `a generation of 20 steps …: 27.5 s`. The figures after
// it on the line derive from that one.
const BENCH_LINE = /^\s*(.*?)[\s:]+(\d+(?:\.\d+)?) (ms|s)\b/;
// A page figure as the page shows it: `2.0 ms`, `4.86 s`, `378/s`, or a ratio (`0.024`, a
// real-time factor). A number is milliseconds (`loadMs`).
const PAGE_FIGURE = /^(\d+(?:\.\d+)?)\s*(ms|s|\/s)?$/;
const BENCH_RUNS = 3;
// A timing moved when it changed by more than this share of its baseline, or by more than twice
// the spread of the bench runs behind either when that is wider.
const NOISE = 0.1;
const TIMING_TABLE = ["| Timing | Baseline | Now | Change |", "| --- | --- | --- | --- |"];

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
    sections.push(section(model, [error.message], null, null));
    continue;
  }
  const baselinePath = join(ROOT, "models", model, "baseline.json");
  if (values.bless) {
    writeFileSync(baselinePath, `${JSON.stringify(record, null, 2)}\n`);
  }
  const baseline = existsSync(baselinePath) ? JSON.parse(readFileSync(baselinePath, "utf8")) : null;
  const differences = baseline ? compare(baseline, record) : ["no baseline: run with --bless"];
  identical &&= differences.length === 0;
  sections.push(section(model, differences, record, baseline));
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
  fetchGoldens(model, product, spec.goldens, log);
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
      parity: parity(model, product, spec.parity, log),
      smoke: check(model, product, spec, "smoke", log),
      e2e: check(model, product, spec, "e2e", log),
      timings: { ...bench(model, product, spec.bench, log), ...pageTimings(model, spec.page) },
    };
  } finally {
    git(vendor, "checkout", "--quiet", pinned);
  }
}

/** Downloads each golden file that is missing or differs from its pinned hash. */
function fetchGoldens(model, product, goldens, log) {
  for (const { file, url, sha256: pinned } of goldens) {
    if (existsSync(join(product, file)) && sha256(join(product, file)) === pinned) continue;
    mkdirSync(dirname(join(product, file)), { recursive: true });
    shell(model, product, `curl --fail --silent --show-error --location --output ${file} ${url}`, log);
    if (sha256(join(product, file)) !== pinned) throw new Error(`${file}: not the goldens pinned in integration.json`);
  }
}

function parity(model, product, command, log) {
  const { status, stdout } = shell(model, product, command, log);
  return { status, lines: stdout.split("\n").filter((line) => line !== "") };
}

/** Runs a check's commands with `{out}` in each replaced by its output directory, and records
 * each command's exit status, the hashes of the outputs written there and the metrics printed.
 * The outputs and metrics whose `<check>/<name>` matches the model's `unstable` pattern come from
 * a check that is not reproducible (live audio fed in real time): they are recorded under
 * `unstable`, which the comparison skips, so only the exit statuses hold them to their
 * thresholds. */
function check(model, product, spec, name, log) {
  const out = join(RUNS, model, name);
  mkdirSync(out, { recursive: true });
  const runs = spec[name].map((command) => shell(model, product, command.replaceAll("{out}", out), log));
  const files = readdirSync(out, { recursive: true })
    .filter((file) => spec.outputs.some((extension) => file.endsWith(extension)))
    .sort();
  const unstable = (text) => spec.unstable !== undefined && new RegExp(spec.unstable).test(`${name}/${text}`);
  const outputs = files.map((file) => [join(dirname(file), basename(file).replace(STAMP, "")), sha256(join(out, file))]);
  const metrics = runs.flatMap(({ stdout }) => stdout.match(METRIC) ?? []);
  return {
    statuses: runs.map(({ status }) => status),
    outputs: Object.fromEntries(outputs.filter(([file]) => !unstable(file))),
    metrics: metrics.filter((metric) => !unstable(metric)),
    unstable: {
      outputs: Object.fromEntries(outputs.filter(([file]) => unstable(file))),
      metrics: metrics.filter(unstable),
    },
  };
}

/** The bench's times over `BENCH_RUNS` runs: each line's median, and the runs' spread around it. */
function bench(model, product, command, log) {
  const runs = Array.from({ length: BENCH_RUNS }, () => {
    const { status, stdout } = shell(model, product, command, log);
    if (status !== 0) throw new Error(`\`${command}\` failed with ${status}; see ${log}`);
    return stdout.split("\n").flatMap((line) => {
      const [, label, number, unit] = BENCH_LINE.exec(line) ?? [];
      if (!label) return [];
      return [[`bench/${label.replace(/\s+/g, " ")}`, milliseconds(Number(number), unit)]];
    });
  });
  return Object.fromEntries(
    runs[0].map(([name], line) => {
      const times = runs.map((run) => run[line][1]).sort((a, b) => a - b);
      const median = times[Math.floor(times.length / 2)];
      return [name, { value: median, unit: "ms", spread: (times.at(-1) - times[0]) / median }];
    }),
  );
}

/** The figures `fields` names in each e2e run's results, by build: `web-threaded/loadMs`. A
 * figure the run did not reach (its check failed first) is null. */
function pageTimings(model, fields) {
  const runs = join(RUNS, model, "e2e");
  const results = readdirSync(runs, { recursive: true }).filter((file) => file.endsWith(".json"));
  return Object.fromEntries(
    results.flatMap((file) => {
      const shown = JSON.parse(readFileSync(join(runs, file), "utf8"));
      return fields.map((field) => {
        const value = field.split(".").reduce((parent, key) => parent?.[key], shown);
        return [`${dirname(file)}/${field}`, value === undefined ? null : figure(value)];
      });
    }),
  );
}

function figure(shown) {
  if (typeof shown === "number") return { value: shown, unit: "ms" };
  const [, number, unit = ""] = PAGE_FIGURE.exec(shown) ?? [];
  if (number === undefined) throw new Error(`${JSON.stringify(shown)}: not a page figure`);
  if (unit === "s" || unit === "ms") {
    return { value: milliseconds(Number(number), unit), unit: "ms" };
  }
  return { value: Number(number), unit };
}

function milliseconds(number, unit) {
  return unit === "s" ? number * 1000 : number;
}

function compare(baseline, record) {
  const differences = [];
  const walk = (path, want, got) => {
    if (typeof want === "object" && want !== null && typeof got === "object" && got !== null) {
      for (const key of new Set([...Object.keys(want), ...Object.keys(got)])) {
        if (key !== "unstable" && key !== "timings") walk([...path, key], want[key], got[key]);
      }
    } else if (want !== got) {
      differences.push(`\`${path.join(".")}\`: baseline ${JSON.stringify(want)}, now ${JSON.stringify(got)}`);
    }
  };
  walk([], baseline, record);
  return differences;
}

function section(model, differences, record, baseline) {
  const verdict = differences.length === 0 ? "identical to its baseline" : `${differences.length} differences`;
  const unstable = record ? [...record.smoke.unstable.metrics, ...record.e2e.unstable.metrics] : [];
  const timings = record
    ? [...TIMING_TABLE, ...timingRows(baseline?.timings, record.timings), ""]
    : [];
  return [
    `## ${model}: ${verdict}`,
    "",
    ...differences.map((difference) => `- ${difference}`),
    ...unstable.map((metric) => `- not reproducible, held to its threshold only: ${metric}`),
    "",
    ...timings,
  ].join("\n");
}

/** Each timing beside its baseline's, flagged when it moved beyond its noise. */
function timingRows(baseline, timings) {
  return Object.entries(timings).map(([name, now]) => {
    const was = baseline?.[name];
    if (!was || !now) return `| ${name} | ${shown(was)} | ${shown(now)} | |`;
    const change = now.value / was.value - 1;
    const noise = Math.max(NOISE, 2 * (now.spread ?? 0), 2 * (was.spread ?? 0));
    const slower = now.unit === "/s" ? change < 0 : change > 0;
    const moved = Math.abs(change) > noise ? (slower ? " **slower**" : " **faster**") : "";
    return `| ${name} | ${shown(was)} | ${shown(now)} | ${(change * 100).toFixed(1)}%${moved} |`;
  });
}

function shown(figure) {
  if (!figure) return "–";
  const value = Number(figure.value.toPrecision(4));
  return figure.unit === "/s" ? `${value}/s` : `${value} ${figure.unit}`.trim();
}

function shell(model, product, command, log) {
  console.log(`${model}: ${command}`);
  appendFileSync(log, `$ ${command}\n`);
  const result = spawnSync("sh", ["-c", command], { cwd: product, encoding: "utf8", maxBuffer: 1 << 30 });
  appendFileSync(log, `${result.stdout}${result.stderr}exit ${result.status}\n\n`);
  return result;
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
