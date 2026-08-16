#!/usr/bin/env node
/**
 * Regression harness — F4: startPerformanceStream must not throw on a
 * malformed SSE frame.
 *
 * Every other SSE handler in the app guards JSON.parse with try/catch and a
 * `|| "{}"` fallback; this one raw-parsed. One truncated frame threw out of
 * the handler, and since the throw happens inside the EventSource dispatch
 * the stream's error/reconnect wiring never ran — the dashboard froze on
 * stale values silently.
 *
 * The REAL function text is sliced out of ui.js and executed against a stub
 * EventSource (no test framework, no browser).
 *
 * Run: node tests/js/test_ui_performance_stream.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const uiPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const src = fs.readFileSync(uiPath, "utf8");

// ── Slice the REAL function out of ui.js (brace-balanced) ──
const start = src.indexOf("function startPerformanceStream() {");
assert.ok(start >= 0, "startPerformanceStream not found in ui.js");
let depth = 0;
let i = src.indexOf("{", start);
for (; i < src.length; i++) {
  if (src[i] === "{") depth++;
  else if (src[i] === "}") {
    depth--;
    if (depth === 0) break;
  }
}
assert.ok(i < src.length, "unbalanced braces extracting startPerformanceStream");
const fnText = src.slice(start, i + 1);

// ── Stub EventSource / free variables ──
let lastSource = null;
globalThis.EventSource = class {
  constructor(url) { this.url = url; this._handlers = {}; lastSource = this; }
  addEventListener(type, fn) { this._handlers[type] = fn; }
  close() {}
};
globalThis.performanceStream = null;   // module-level let in ui.js
let metricCalls = [];
globalThis.updatePerformanceMetrics = (d) => { metricCalls.push(d); };
globalThis.console = console;

// Compile the real function; free variables resolve against globalThis.
const api = new Function(fnText + "\nreturn { startPerformanceStream };")();

let passed = 0;
function check(name, fn) { fn(); passed++; console.log(`PASS: ${name}`); }

// ── Scenario 1: malformed frame — no throw, no update, stream survives ──
check("malformed frame does not throw and does not paint metrics", () => {
  metricCalls = [];
  api.startPerformanceStream();
  assert.ok(lastSource, "EventSource created");
  const h = lastSource._handlers.performance_metrics;
  assert.ok(typeof h === "function", "performance_metrics handler wired");
  assert.doesNotThrow(() => h({ data: "{broken json" }), "malformed frame must not throw");
  assert.equal(metricCalls.length, 0, "no metrics painted from garbage");
});

// ── Scenario 2: empty frame — codebase parity: || '{}' fallback → harmless {} ──
check("empty frame falls back to {} (parity with all other SSE handlers)", () => {
  metricCalls = [];
  api.startPerformanceStream();
  const h = lastSource._handlers.performance_metrics;
  assert.doesNotThrow(() => h({ data: "" }), "empty frame must not throw");
  assert.equal(metricCalls.length, 1, "empty frame paints the {} fallback");
  assert.deepEqual(metricCalls[0], {}, "fallback payload is an empty object");
});

// ── Scenario 3: valid frame still updates ──
check("valid frame parses and updates metrics", () => {
  metricCalls = [];
  api.startPerformanceStream();
  lastSource._handlers.performance_metrics({ data: '{"cpu": 12, "mem": 34}' });
  assert.equal(metricCalls.length, 1, "valid frame updates metrics");
  assert.equal(metricCalls[0].cpu, 12);
  assert.equal(metricCalls[0].mem, 34);
});

// ── Scenario 3: error handler keeps the identity-guarded reconnect ──
check("error handler still identity-guards reconnect", () => {
  api.startPerformanceStream();
  const errHandler = lastSource._handlers.error;
  assert.ok(typeof errHandler === "function", "error handler wired");
  const body = errHandler.toString();
  assert.ok(body.includes("performanceStream === stream"), "identity check retained");
});

// ── Source gates ──
check("source gate: parse is try-wrapped with || '{}' fallback", () => {
  assert.ok(fnText.includes("try { data = JSON.parse(event.data || \"{}\")"),
    "must parse with fallback inside try");
  assert.ok(fnText.includes("catch"), "must catch parse failures");
  assert.ok(fnText.includes("updatePerformanceMetrics(data)"), "valid path still updates");
  assert.ok(fnText.indexOf("catch") < fnText.indexOf("updatePerformanceMetrics(data)"),
    "update must only run after a successful parse");
});

console.log(`\nAll ${passed} checks passed.`);
