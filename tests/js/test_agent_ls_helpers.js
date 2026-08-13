#!/usr/bin/env node
/**
 * Regression harness — P9-2: every localStorage access routes through
 * _lsGet/_lsSet/_lsRemove so blocked/private storage never breaks callers.
 *
 * Bug class: bare localStorage access. Worst case was the ui.js MODULE
 * top-level state literal — `llmUseCtxPack: (localStorage.getItem(...) ===
 * "1")` — where a throwing storage (private mode, blocked cookies) killed
 * ui.js at load → every panel on the page broke. Handler functions with
 * bare reads/writes died mid-way for the same reason.
 *
 * Fix under test: try-wrapped helpers in ALL THREE UI files (null / no-op
 * on failure), and a source gate that forbids direct localStorage. access
 * anywhere outside those helper bodies.
 *
 * The REAL helpers are sliced out of ui.js and executed against throwing /
 * working storage stubs (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_ls_helpers.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const UI_FILES = [
  ["ui.js", path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js")],
  ["agent-panel.js", path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js")],
  ["design-chat.js", path.join(__dirname, "..", "..", "webapp", "ui", "static", "design-chat.js")],
];

// ── Slice a function out of a source file (brace-balanced) ──
function sliceFunction(fileSrc, name) {
  const start = fileSrc.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found`);
  let depth = 0;
  let i = fileSrc.indexOf("{", start);
  for (; i < fileSrc.length; i++) {
    if (fileSrc[i] === "{") depth++;
    else if (fileSrc[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < fileSrc.length, `unbalanced braces extracting ${name}`);
  return fileSrc.slice(start, i + 1);
}

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── Helper semantics against a THROWING storage stub ──
const uiSrc = fs.readFileSync(UI_FILES[0][1], "utf8");
const helpersText = ["_lsGet", "_lsSet", "_lsRemove"].map((n) => sliceFunction(uiSrc, n)).join("\n");
const helpers = new Function(`${helpersText}\nreturn { _lsGet, _lsSet, _lsRemove };`)();

check("helpers tolerate a throwing storage (blocked/private mode)", () => {
  globalThis.localStorage = {
    getItem() { throw new Error("SecurityError: storage blocked"); },
    setItem() { throw new Error("SecurityError: storage blocked"); },
    removeItem() { throw new Error("SecurityError: storage blocked"); },
  };
  assert.strictEqual(helpers._lsGet("any.key"), null, "getItem failure → null");
  assert.doesNotThrow(() => helpers._lsSet("any.key", "v"));
  assert.doesNotThrow(() => helpers._lsRemove("any.key"));
});

check("helpers work when storage is available", () => {
  const store = new Map();
  globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  assert.strictEqual(helpers._lsGet("k"), null, "missing key → null");
  helpers._lsSet("k", "1");
  assert.strictEqual(helpers._lsGet("k"), "1");
  helpers._lsRemove("k");
  assert.strictEqual(helpers._lsGet("k"), null);
});

// ── SOURCE GATE: no direct localStorage. access outside the helper bodies ──
// Helper set is per-file (agent-panel.js additionally has _lsKeys for
// key-scan helpers like _tlSweepStale); strip whichever exist, then flag
// anything left.
const HELPER_NAMES = ["_lsGet", "_lsSet", "_lsRemove", "_lsKeys"];
function stripHelpers(fileSrc) {
  let out = fileSrc;
  for (const h of HELPER_NAMES) {
    const start = out.indexOf(`function ${h}(`);
    if (start >= 0) out = out.replace(sliceFunction(out, h), "");
  }
  return out;
}

check("source gate — no direct localStorage access outside helper bodies", () => {
  for (const [name, p] of UI_FILES) {
    const text = fs.readFileSync(p, "utf8");
    const rest = stripHelpers(text);
    const hits = rest.split("\n").filter((l) => l.includes("localStorage."));
    assert.deepStrictEqual(
      hits,
      [],
      `${name}: direct localStorage access outside helpers:\n${hits.join("\n")}`
    );
  }
});

// ── SOURCE GATE: helpers exist in ALL three files ──
check("source gate — all three UI files define the three helpers", () => {
  for (const [name, p] of UI_FILES) {
    const text = fs.readFileSync(p, "utf8");
    for (const fn of ["function _lsGet(", "function _lsSet(", "function _lsRemove("]) {
      assert.ok(text.includes(fn), `${name} missing ${fn}`);
    }
  }
});

console.log(`\n${passed}/4 checks passed (P9-2 localStorage helpers)`);
process.exit(passed === 4 ? 0 : 1);
