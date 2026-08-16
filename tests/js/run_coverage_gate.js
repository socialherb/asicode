#!/usr/bin/env node
/**
 * JS suite runner + coverage gate (P3 — round 32-5).
 *
 * Runs every tests/js/test_*.js under the coverage preload and gates that
 * every LIVE handlers-map key of agent-panel.js (and every
 * sse.addEventListener listener of design-chat.js) is EXERCISED by at least
 * one test.
 *
 * Why not V8 native coverage: tests extract code from the static files as
 * STRINGS and compile it via `new Function(...)`. NODE_V8_COVERAGE /
 * --experimental-test-coverage attribute Function-constructed code to
 * empty-url scripts, so the real files measure 0% no matter how much code
 * the tests execute (empirically verified 2026-08-16). The preload instead
 * records the exact body text of every Function that was compiled AND
 * called; a key is covered iff a recorded body contains the key's source
 * slice as it appears in the real file.
 *
 * Gate semantics (baseline-diff, mirroring scripts/check_f821_no_new.py):
 *   - tests/js/coverage_baseline.json stores each surface's uncovered keys.
 *   - The gate FAILS only on NET-NEW uncovered keys — i.e. a handler added
 *     to the live map (or a listener added in design-chat.js) without any
 *     test that compiles and calls it. This automates the round 32 manual
 *     dead-code hunts.
 *   - Removing a key or adding a test shrinks the uncovered set → pass.
 *   - `--update` re-baselines the current state (use after deliberate
 *     coverage improvements or surface changes, with the report as
 *     justification).
 *
 * Usage:
 *   node tests/js/run_coverage_gate.js           # run suite + gate
 *   node tests/js/run_coverage_gate.js --update  # run suite + re-baseline
 *   node tests/js/run_coverage_gate.js --list    # report only, never fails
 *   node tests/js/run_coverage_gate.js --verbose # per-key call counts
 */
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const TESTS_DIR = __dirname;
const REPO_ROOT = path.resolve(TESTS_DIR, "..", "..");
const PRELOAD = path.join(TESTS_DIR, "coverage_preload.js");
const BASELINE_FILE = path.join(TESTS_DIR, "coverage_baseline.json");

const args = process.argv.slice(2);
const UPDATE = args.includes("--update");
const LIST_ONLY = args.includes("--list");
const VERBOSE = args.includes("--verbose");

// ── Surfaces ──────────────────────────────────────────────────────────────
// agent-panel.js: live handlers-map entries `    name: (data) => {` inside
//   the region between the two markers (identical anchors to the round 32
//   dead-map gate).
// design-chat.js: sse.addEventListener("name", (e) => { ... }) listeners
//   (no EventSequencer override — direct addEventListener IS the live path).
const SURFACES = [
  {
    file: "webapp/ui/static/agent-panel.js",
    kind: "map",
    region: ["// Define event handlers", "// Create EventSequencer"],
    entryRe: /^    ([a-z_0-9]+): \(data\) => \{$/gm,
  },
  {
    file: "webapp/ui/static/design-chat.js",
    kind: "listener",
    entryRe: /sse\.addEventListener\("([a-z_0-9]+)"/g,
  },
];

// Brace-balanced close of the opening brace at `openIdx`.
function braceClose(src, openIdx) {
  let depth = 0;
  let i = openIdx;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") { depth--; if (depth === 0) break; }
  }
  if (i >= src.length) throw new Error(`unbalanced braces at offset ${openIdx}`);
  return i;
}

function inventory(surface) {
  const src = fs.readFileSync(path.join(REPO_ROOT, surface.file), "utf8");
  let scope = src;
  if (surface.region) {
    const rs = src.indexOf(surface.region[0]);
    const re_ = src.indexOf(surface.region[1]);
    if (rs < 0 || re_ <= rs) {
      throw new Error(`region markers not found in ${surface.file}`);
    }
    scope = src.slice(rs, re_);
  }
  surface.entryRe.lastIndex = 0;
  const items = [];
  let m;
  while ((m = surface.entryRe.exec(scope)) !== null) {
    items.push(m[1]);
  }
  // Exact source slice per entry — the text tests embed in `new Function`.
  const slices = new Map();
  for (const name of items) {
    if (surface.kind === "map") {
      const anchor = `    ${name}: (data) => {`;
      const idx = src.indexOf(anchor);
      if (idx < 0) throw new Error(`anchor ${anchor} not found in ${surface.file}`);
      if (src.indexOf(anchor, idx + 1) >= 0) {
        throw new Error(`anchor ${anchor} is not unique in ${surface.file} — coverage attribution ambiguous`);
      }
      const open = idx + anchor.length - 1;
      slices.set(name, src.slice(idx, braceClose(src, open) + 1));
    } else {
      const anchor = `sse.addEventListener("${name}"`;
      const idx = src.indexOf(anchor);
      if (idx < 0) throw new Error(`anchor ${anchor} not found in ${surface.file}`);
      const arrow = src.indexOf("=>", idx);
      if (arrow < 0 || arrow > idx + anchor.length + 80) {
        throw new Error(`listener ${name} is not an arrow function in ${surface.file}`);
      }
      const open = src.indexOf("{", arrow);
      slices.set(name, src.slice(open + 1, braceClose(src, open)));
    }
  }
  return { src, names: items, slices };
}

function runSuite(covDir) {
  const tests = fs.readdirSync(TESTS_DIR)
    .filter((f) => /^test_.*\.js$/.test(f))
    .sort();
  const failures = [];
  let passed = 0;
  for (const t of tests) {
    const covOut = path.join(covDir, t.replace(/\.js$/, ".json"));
    const res = spawnSync(process.execPath, ["--require", PRELOAD, path.join(TESTS_DIR, t)], {
      env: { ...process.env, COV_OUT: covOut },
      cwd: REPO_ROOT,
      encoding: "utf8",
      timeout: 120000,
    });
    if (res.status === 0) {
      passed++;
    } else {
      const tail = (res.stderr || res.stdout || "").trim().split("\n").slice(-15).join("\n");
      failures.push({ test: t, status: res.status, signal: res.signal, tail });
    }
  }
  return { passed, total: tests.length, failures };
}

function mergeEvidence(covDir) {
  const bodies = new Map(); // body -> call count
  for (const f of fs.readdirSync(covDir)) {
    if (!f.endsWith(".json")) continue;
    let data;
    try {
      data = JSON.parse(fs.readFileSync(path.join(covDir, f), "utf8"));
    } catch {
      continue;
    }
    for (const r of data.records || []) {
      if (typeof r.body === "string" && r.body.length > 0) {
        bodies.set(r.body, (bodies.get(r.body) || 0) + (r.count || 1));
      }
    }
  }
  return bodies;
}

function computeCoverage(surface, inv, bodies) {
  const covered = [];
  const uncovered = [];
  const callCounts = new Map();
  for (const name of inv.names) {
    const slice = inv.slices.get(name);
    let total = 0;
    for (const [body, count] of bodies) {
      if (body.includes(slice)) total += count;
    }
    if (total > 0) {
      covered.push(name);
      callCounts.set(name, total);
    } else {
      uncovered.push(name);
    }
  }
  return { covered, uncovered, callCounts };
}

function loadBaseline() {
  try {
    return JSON.parse(fs.readFileSync(BASELINE_FILE, "utf8"));
  } catch {
    return { surfaces: {} };
  }
}

function main() {
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), "js-cov-"));
  const results = new Map(); // file -> { inv, cov }
  const failures = [];
  let suitePassed = 0;
  try {
    const suite = runSuite(covDir);
    suitePassed = suite.passed;
    failures.push(...suite.failures);
    const bodies = mergeEvidence(covDir);

    for (const surface of SURFACES) {
      const inv = inventory(surface);
      const cov = computeCoverage(surface, inv, bodies);
      results.set(surface.file, { surface, inv, cov });
    }
  } finally {
    fs.rmSync(covDir, { recursive: true, force: true });
  }

  // ── Report ─────────────────────────────────────────────────────────────
  console.log(`\nJS suite: ${suitePassed}/${suitePassed + failures.length} passed`);
  const baseline = loadBaseline();
  const baselineSurfaces = baseline.surfaces || {};
  let gateFailed = false;

  for (const [file, { surface, inv, cov }] of results) {
    const pct = inv.names.length ? (100 * cov.covered.length / inv.names.length).toFixed(1) : "100.0";
    console.log(`\n${surface.kind === "map" ? "handlers-map" : "listeners"}  ${file}  (${surface.kind})`);
    console.log(`  exercised ${cov.covered.length}/${inv.names.length}  (${pct}%)`);
    if (VERBOSE && cov.covered.length) {
      console.log(`  covered (call count): ${cov.covered.map((n) => `${n}(${cov.callCounts.get(n)})`).join(", ")}`);
    }
    if (cov.uncovered.length) {
      console.log(`  NOT exercised: ${cov.uncovered.join(", ")}`);
    }

    const prev = baselineSurfaces[file] || {};
    const prevUncovered = new Set(prev.uncovered || []);
    const newUncovered = cov.uncovered.filter((n) => !prevUncovered.has(n));
    if (newUncovered.length && !LIST_ONLY) {
      gateFailed = true;
      console.log(`  ❌ GATE: ${newUncovered.length} NET-NEW uncovered key(s): ${newUncovered.join(", ")}`);
      console.log(`     (add a behavior test that compiles+calls these, or remove the dead handler;`);
      console.log(`      deliberate baseline change → node tests/js/run_coverage_gate.js --update)`);
    }
  }

  for (const f of failures) {
    console.log(`\n❌ TEST FAILED: ${f.test} (status=${f.status} signal=${f.signal})`);
    console.log(f.tail);
    gateFailed = true;
  }

  if (UPDATE) {
    const out = { surfaces: {} };
    for (const [file, { surface, cov }] of results) {
      out.surfaces[file] = { kind: surface.kind, uncovered: cov.uncovered };
    }
    fs.writeFileSync(BASELINE_FILE, JSON.stringify(out, null, 2) + "\n");
    console.log(`\n✅ Baseline updated → ${path.relative(REPO_ROOT, BASELINE_FILE)}`);
  }

  if (gateFailed) {
    process.exitCode = 1;
  } else if (failures.length === 0) {
    console.log("\n✅ Coverage gate passed — every live handler is exercised by a test");
  }
}

main();
