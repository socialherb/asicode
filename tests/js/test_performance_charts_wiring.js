#!/usr/bin/env node
/**
 * Regression harness — 28차 감사 #1: the Performance tab's two canvases
 * (cache-hit-chart, tool-time-chart) were DEAD — ui.html declares them and
 * updatePerformanceMetrics() updates every card + raw JSON, but
 * updatePerformanceCharts() was a no-op placeholder ("Could use Chart.js if
 * included"), so the user sees two permanently blank chart areas.
 *
 * The backend /agent/performance summary is a live SNAPSHOT (no timeseries), so
 * the cache-hit chart accumulates a client-side rolling history from the 2 s
 * SSE stream / manual refreshes, and the tool-time chart renders the per-tool
 * avg_execution_time_ms snapshot as horizontal bars.
 *
 * Run: node tests/js/test_performance_charts_wiring.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const JS_PATH = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const HTML_PATH = path.join(__dirname, "..", "..", "webapp", "ui", "templates", "ui.html");
const uiSrc = fs.readFileSync(JS_PATH, "utf8");
const htmlSrc = fs.readFileSync(HTML_PATH, "utf8");

let passed = 0;
function check(name, fn) { fn(); passed++; console.log(`PASS: ${name}`); }

// Extract a top-level function body from source (brace-matched).
function extractFunction(src, name) {
  const re = new RegExp(`(?:^|\\n)(?:async )?function ${name}\\(`);
  const start = src.search(re);
  if (start < 0) return null;
  let depth = 0;
  let i = src.indexOf("{", start);
  if (i < 0) return null;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  return null;
}

// --- DOM anchors (these existed all along — the bug is the missing wiring) ---
check("cache-hit-chart canvas exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="cache-hit-chart"'), 'ui.html has no id="cache-hit-chart"');
});

check("tool-time-chart canvas exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="tool-time-chart"'), 'ui.html has no id="tool-time-chart"');
});

// --- Source gates: the placeholder must be gone, real draw code present ---
check("updatePerformanceCharts is not a no-op placeholder anymore", () => {
  assert.ok(uiSrc.includes("function updatePerformanceCharts(summary) {"),
    "updatePerformanceCharts not found");
  assert.ok(!uiSrc.includes("Placeholder for chart updates"),
    "placeholder body still present — charts are still dead");
});

check("_drawCacheHitChart exists in ui.js", () => {
  assert.ok(uiSrc.includes("function _drawCacheHitChart(summary) {"),
    "_drawCacheHitChart not found");
});

check("_drawToolTimeChart exists in ui.js", () => {
  assert.ok(uiSrc.includes("function _drawToolTimeChart(summary) {"),
    "_drawToolTimeChart not found");
});

check("rolling history buffer declared (ring-buffer cap constant)", () => {
  assert.ok(uiSrc.includes("let _perfCacheHistory = [];"), "_perfCacheHistory not declared");
  assert.ok(uiSrc.includes("_PERF_HISTORY_MAX"), "_PERF_HISTORY_MAX missing");
});

// --- Execution gates: slice the REAL functions and run them on stubs ---
function buildApi() {
  const fns = ["updatePerformanceCharts", "_drawCacheHitChart", "_drawToolTimeChart"];
  let blob = "let _perfCacheHistory = [];\nconst _PERF_HISTORY_MAX = 60;\n";
  for (const name of fns) {
    const text = extractFunction(uiSrc, name);
    assert.ok(text, `could not extract ${name} from ui.js`);
    blob += text + "\n";
  }
  blob += "\nreturn { updatePerformanceCharts, _drawCacheHitChart, _drawToolTimeChart };\n";
  return new Function(blob)();
}

// Fake 2D context: records draw calls.
function fakeCtx() {
  const calls = { lines: 0, rects: 0, texts: [], strokeStyle: null, fillStyle: null };
  return {
    calls,
    setTransform() {}, clearRect() {},
    beginPath() {}, moveTo() {}, lineTo() { calls.lines++; }, stroke() {},
    fillRect() { calls.rects++; }, fillText(t) { calls.texts.push(String(t)); },
    measureText(t) { return { width: String(t).length * 6 }; },
    set fillStyle(v) { calls.fillStyle = v; },
    set strokeStyle(v) { calls.strokeStyle = v; },
  };
}

function makeCanvas(ctx) {
  return { width: 400, height: 200, clientWidth: 400, clientHeight: 200, getContext: () => ctx };
}

const summary = (overall, tools) => ({
  cache_metrics: {
    overall_hit_rate: overall,
    tool_result_cache: { hit_rate: 0.5, hits: 5, misses: 5 },
    rag_cache: { hit_rate: 0.25, hits: 2, misses: 6 },
    vector_cache: { hit_rate: 0.1, hits: 1, misses: 9 },
  },
  tool_metrics: tools,
});

check("cache-hit chart draws polylines from the rolling history", () => {
  const hitCtx = fakeCtx();
  const hitCanvas = makeCanvas(hitCtx);
  globalThis.el = (id) => (id === "cache-hit-chart" ? hitCanvas : null);
  const api = buildApi();
  // First snapshot: only 1 sample → 'collecting' text, no line yet.
  api.updatePerformanceCharts(summary(0.4, {}));
  assert.equal(hitCtx.calls.lines, 0, "single sample must not draw a line");
  // Second snapshot: 2 samples → polylines for 4 channels.
  api.updatePerformanceCharts(summary(0.6, {}));
  assert.ok(hitCtx.calls.lines >= 4, `expected >=4 channel line segments, got ${hitCtx.calls.lines}`);
});

check("cache-hit chart keeps the ring buffer bounded", () => {
  const hitCtx = fakeCtx();
  globalThis.el = (id) => (id === "cache-hit-chart" ? makeCanvas(hitCtx) : null);
  const api = buildApi();
  for (let i = 0; i < 75; i++) api.updatePerformanceCharts(summary(0.5, {}));
  assert.ok(hitCtx.calls.lines > 0, "lines drawn after 75 samples");
  // Boundedness is a source-level property (splice to _PERF_HISTORY_MAX);
  // verify the cap constant is referenced by the draw function.
  const draw = extractFunction(uiSrc, "_drawCacheHitChart");
  assert.ok(draw.includes("_PERF_HISTORY_MAX"), "ring buffer must be capped by _PERF_HISTORY_MAX");
});

check("tool-time chart draws horizontal bars from tool_metrics snapshot", () => {
  const toolCtx = fakeCtx();
  globalThis.el = (id) => (id === "tool-time-chart" ? makeCanvas(toolCtx) : null);
  const api = buildApi();
  const tools = {
    read_file: { call_count: 10, avg_execution_time_ms: 12.5 },
    apply_patch: { call_count: 4, avg_execution_time_ms: 220.0 },
    grep: { call_count: 8, avg_execution_time_ms: 3.2 },
  };
  api.updatePerformanceCharts(summary(0.5, tools));
  assert.equal(toolCtx.calls.rects, 3, `expected 3 bars, got ${toolCtx.calls.rects}`);
  const texts = toolCtx.calls.texts.join("|");
  assert.ok(texts.includes("apply_patch"), "tool name labels drawn");
  assert.ok(texts.includes("220.0ms"), "ms value labels drawn");
});

check("empty tool_metrics draws a no-data hint without throwing", () => {
  const toolCtx = fakeCtx();
  globalThis.el = (id) => (id === "tool-time-chart" ? makeCanvas(toolCtx) : null);
  const api = buildApi();
  assert.doesNotThrow(() => api.updatePerformanceCharts(summary(0.5, {})));
  assert.ok(toolCtx.calls.texts.some((t) => t.includes("no tool calls")), "no-data hint text drawn");
});

check("missing canvas element is a harmless no-op", () => {
  globalThis.el = () => null;
  const api = buildApi();
  assert.doesNotThrow(() => api.updatePerformanceCharts(summary(0.5, { read_file: { call_count: 1, avg_execution_time_ms: 1 } })));
});

console.log(`\nAll ${passed} checks passed.`);
