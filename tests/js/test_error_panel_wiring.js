#!/usr/bin/env node
/**
 * Regression harness — 27차 감사: the "Error Monitoring & Diagnostics" panel
 * in ui.html was a DEAD CONTRACT — error-refresh-btn / error-clear-btn /
 * error-auto-refresh / error-table-body had ZERO listeners or consumers in
 * ui.js (the whole panel was unwired, like the Load button in #1 and the
 * Commit button in #2, only worse: an entire panel).  This test locks in the
 * wiring: refresh/clear/dismiss handlers + auto-refresh toggle, backed by the
 * per-repo failure-pattern store (.asicode/failure_patterns.json) served via
 * GET /ui/api/errors/overview, POST /ui/api/errors/clear, POST
 * /ui/api/errors/drop in ui_tools.py.
 *
 * Run: node tests/js/test_error_panel_wiring.js
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
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

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

// --- checks --------------------------------------------------------------

// 1-3. The panel DOM exists (it always did — it just had no wiring).
check("error-refresh-btn exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="error-refresh-btn"'), "ui.html has no id=\"error-refresh-btn\"");
});

check("error-clear-btn exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="error-clear-btn"'), "ui.html has no id=\"error-clear-btn\"");
});

check("error-table-body exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="error-table-body"'), "ui.html has no id=\"error-table-body\"");
});

// 4. ui.js wires the refresh button to a click listener (the dead-contract
//    point: before the fix there were ZERO error-* references in ui.js).
check("error-refresh-btn wired to a click listener in ui.js", () => {
  assert.ok(
    uiSrc.includes('el("error-refresh-btn")?.addEventListener("click"'),
    "ui.js does not register a click listener on error-refresh-btn"
  );
});

// 5. The wired handler is actually defined (no dangling reference).
check("wired refresh handler is defined in ui.js", () => {
  const m = uiSrc.match(/el\("error-refresh-btn"\)\?\.addEventListener\("click",\s*(\w+)/);
  assert.ok(m, "cannot parse error-refresh-btn click listener registration");
  const handler = m[1];
  const defined = extractFunction(uiSrc, handler) !== null || uiSrc.includes(`const ${handler} =`);
  assert.ok(defined, `handler ${handler} is not defined in ui.js`);
});

// 6. The refresh handler reads the overview endpoint (dead-end guard).
check("refreshErrorPanel fetches /ui/api/errors/overview", () => {
  const body = extractFunction(uiSrc, "refreshErrorPanel");
  assert.ok(body, "refreshErrorPanel not found in ui.js");
  assert.ok(
    body.includes('apiGet(`/ui/api/errors/overview?repo_root='),
    "refreshErrorPanel does not call apiGet(/ui/api/errors/overview)"
  );
});

// 7. Rendering updates the summary cards + table + suggestions + charts.
//    renderErrorPanel delegates the table to renderErrorTable; the dead-end
//    guard is the full path renderErrorPanel → renderErrorTable →
//    error-table-body (if either hop is unwired the table stays empty).
check("renderErrorPanel updates cards, table, suggestions and charts", () => {
  const body = extractFunction(uiSrc, "renderErrorPanel");
  assert.ok(body, "renderErrorPanel not found in ui.js");
  assert.ok(body.includes("error-total-count"), "renderErrorPanel never updates error-total-count");
  assert.ok(body.includes("renderErrorTable"), "renderErrorPanel never renders the table");
  assert.ok(body.includes("renderErrorSuggestions"), "renderErrorPanel never renders suggestions");
  assert.ok(body.includes("drawErrorCharts"), "renderErrorPanel never draws the charts");
  const tbl = extractFunction(uiSrc, "renderErrorTable");
  assert.ok(tbl, "renderErrorTable not found in ui.js");
  assert.ok(tbl.includes("error-table-body"), "renderErrorTable never updates error-table-body");
});

// 8. Clear POSTs to the clear endpoint and refreshes afterwards.
check("clearErrorLogs POSTs to /ui/api/errors/clear and refreshes", () => {
  const body = extractFunction(uiSrc, "clearErrorLogs");
  assert.ok(body, "clearErrorLogs not found in ui.js");
  assert.ok(
    body.includes('apiPost("/ui/api/errors/clear"'),
    "clearErrorLogs does not call apiPost(/ui/api/errors/clear)"
  );
  assert.ok(body.includes("refreshErrorPanel"), "clearErrorLogs never refreshes after clearing");
});

// 9. Per-row dismiss POSTs the pattern key to the drop endpoint.
check("dismissErrorPattern POSTs to /ui/api/errors/drop with the key", () => {
  const body = extractFunction(uiSrc, "dismissErrorPattern");
  assert.ok(body, "dismissErrorPattern not found in ui.js");
  assert.ok(
    body.includes('apiPost("/ui/api/errors/drop"'),
    "dismissErrorPattern does not call apiPost(/ui/api/errors/drop)"
  );
  assert.ok(body.includes("key"), "dismissErrorPattern does not send the pattern key");
});

// 10. Auto-refresh checkbox is wired: change listener + interval toggle.
check("error-auto-refresh wired with change listener + interval toggle", () => {
  assert.ok(
    uiSrc.includes('el("error-auto-refresh")?.addEventListener("change"'),
    "error-auto-refresh has no change listener"
  );
  const body = extractFunction(uiSrc, "setErrorAutoRefresh");
  assert.ok(body, "setErrorAutoRefresh not found in ui.js");
  assert.ok(body.includes("setInterval"), "setErrorAutoRefresh never starts a poll interval");
  assert.ok(body.includes("clearInterval"), "setErrorAutoRefresh never stops the poll interval");
});

// 11. Initial load refreshes the panel (dead-end guard: without this the
//     panel stays empty until the user clicks refresh).
check("DOMContentLoaded calls refreshErrorPanel on initial load", () => {
  const m = uiSrc.match(/document\.addEventListener\("DOMContentLoaded",[\s\S]{0,4000}?refreshErrorPanel\(\)/);
  assert.ok(m, "refreshErrorPanel() is never called on initial load");
});

// 12. Table rows: executed sort (last_seen desc), status threshold, cap 50.
check("buildErrorTableRows sorts by last_seen desc and marks recurrence", () => {
  const fnSrc = extractFunction(uiSrc, "buildErrorTableRows");
  assert.ok(fnSrc, "buildErrorTableRows not found in ui.js");
  const buildErrorTableRows = eval(`(${fnSrc})`); // eslint-disable-line no-eval
  const rows = buildErrorTableRows([
    { key: "a::x", tool: "a", reason: "x", effective: 1, last_seen: 100 },
    { key: "b::y", tool: "b", reason: "y", effective: 5, last_seen: 300 },
    { key: "c::z", tool: "c", reason: "z", effective: 2.7, last_seen: 200 },
  ]);
  assert.deepStrictEqual(
    rows.map((r) => [r.key, r.status, r.count]),
    [
      ["b::y", "recurrent", 5],
      ["c::z", "new", 3],
      ["a::x", "new", 1],
    ]
  );
  const many = buildErrorTableRows(
    Array.from({ length: 70 }, (_, i) => ({ key: `t::${i}`, tool: "t", reason: `r${i}`, effective: 1, last_seen: i }))
  );
  assert.strictEqual(many.length, 50, "table rows must be capped at 50");
});

// 13. Time formatting: relative within 24h, absolute date beyond (executed).
check("formatErrorTime renders relative and absolute times", () => {
  const fnSrc = extractFunction(uiSrc, "formatErrorTime");
  assert.ok(fnSrc, "formatErrorTime not found in ui.js");
  const formatErrorTime = eval(`(${fnSrc})`); // eslint-disable-line no-eval
  const now = 1_700_000_000; // 2023-11-14T22:13:20Z
  assert.strictEqual(formatErrorTime(now - 30, now), "30s ago");
  assert.strictEqual(formatErrorTime(now - 3600, now), "1h ago");
  assert.strictEqual(formatErrorTime(now - 2 * 86400, now).length, 16); // "2023-11-12 22:13"
  assert.strictEqual(formatErrorTime(0, now), "—");
  assert.strictEqual(formatErrorTime(null, now), "—");
});

// 14. Frequency histogram buckets last_seen into N daily buckets (executed).
check("bucketByDay produces N daily buckets", () => {
  const fnSrc = extractFunction(uiSrc, "bucketByDay");
  assert.ok(fnSrc, "bucketByDay not found in ui.js");
  const bucketByDay = eval(`(${fnSrc})`); // eslint-disable-line no-eval
  const now = 7 * 86400;
  const out = bucketByDay(
    [
      { last_seen: now },         // today
      { last_seen: now - 86400 }, // yesterday
      { last_seen: now - 6 * 86400 },
      { last_seen: now - 7 * 86400 }, // out of window → dropped
      { last_seen: 0 },
    ],
    now,
    7
  );
  assert.strictEqual(out.length, 7);
  assert.deepStrictEqual(out.map((b) => b.count), [1, 0, 0, 0, 0, 1, 1]);
});

// 15. Chart aggregation: top reasons/tools by effective count (executed).
check("topReasons and topTools aggregate by effective count", () => {
  const src1 = extractFunction(uiSrc, "topReasons");
  const src2 = extractFunction(uiSrc, "topTools");
  assert.ok(src1 && src2, "topReasons/topTools not found in ui.js");
  const topReasons = eval(`(${src1})`); // eslint-disable-line no-eval
  const topTools = eval(`(${src2})`); // eslint-disable-line no-eval
  const pats = [
    { tool: "read_file", reason: "file missing", effective: 4.2 },
    { tool: "bash", reason: "transient failure", effective: 1.0 },
    { tool: "read_file", reason: "permission denied", effective: 2.5 },
  ];
  assert.deepStrictEqual(topReasons(pats, 6), [
    { label: "file missing", value: 4 },
    { label: "permission denied", value: 3 },
    { label: "transient failure", value: 1 },
  ]);
  assert.deepStrictEqual(topTools(pats, 6), [
    { label: "read_file", value: 7 },
    { label: "bash", value: 1 },
  ]);
});

// 16. Suggestions render from server-provided hints (dead-end guard).
check("renderErrorSuggestions renders hints and keeps the empty-state", () => {
  const body = extractFunction(uiSrc, "renderErrorSuggestions");
  assert.ok(body, "renderErrorSuggestions not found in ui.js");
  assert.ok(body.includes("error-suggestions-list"), "renderErrorSuggestions never touches the list");
  assert.ok(body.includes("hint"), "renderErrorSuggestions ignores the server hint field");
});

console.log(`\n${passed} checks passed`);
