#!/usr/bin/env node
/**
 * Regression harness — P15-9: the agent panel's agentRunStream() bounds
 * requestText client-side, mirroring the server's /agent/run request_text cap
 * (256 KiB, agent_stream.py P15-4). ui.js's _launchAgent got its own gate in
 * P15-7; agent-panel.js's run entry (continue bar / subagent spawn / panel
 * run) is a separate call site and had none.
 *
 * Fix under test: agentRunStream() rejects params.requestText over
 * _AGENT_RUN_TEXT_MAX_BYTES with an addTL error BEFORE touching the SSE
 * source or any downstream work.
 *
 * The REAL function text is sliced out of agent-panel.js (brace-balanced) and
 * executed against stub globals (no test framework, no browser).
 * Run: node tests/js/test_agent_run_stream_cap.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "agent-panel.js"), "utf8");

function sliceFunction(src, anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found`);
  const open = src.indexOf("{", start);
  assert.ok(open >= 0, `no body for ${anchor}`);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces while extracting ${anchor}`);
  return src.slice(start, i + 1);
}

const fnText = sliceFunction(src, "async function agentRunStream(params)");

// ── stub environment ──
let lastError = null;
let sseTouches = 0;
let passed = 0;
function check(name, ok, detail) {
  passed++;
  if (!ok) {
    console.error(`  FAIL: ${name}${detail ? ` (${detail})` : ""}`);
    process.exit(1);
  }
  console.log(`  ok: ${name}`);
}

function addTL(level, msg) { if (level === "error") lastError = msg; }
const _AGENT_RUN_TEXT_MAX_BYTES = 256 * 1024;  // must mirror the source constant

// The REAL function reads `_agentSSESource` from its scope chain (Function
// scope → global). Expose it as a global getter so "touched" means the gate
// did NOT return early.
Object.defineProperty(globalThis, "_agentSSESource", {
  get() { sseTouches++; return null; },
  set(v) { sseTouches++; },
  configurable: true,
});

const env = {};

// P20-1 added a supersede block (cancel POST for the previous run) between the
// size gate and the SSE-cancel block; the gate harness must stub its globals
// or the exact-cap/missing-requestText cases would crash on sessionStorage.
globalThis.sessionStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis._agentSessionId = null;
globalThis._agentSetStatus = () => {};
globalThis.fetch = async (url, opts) => {
  if (typeof url === "string" && url.startsWith("/agent/cancel/")) {
    return { ok: true, status: 200, json: async () => ({}) };
  }
  if (typeof url === "string" && url.startsWith("/agent/run")) {
    return { ok: false, status: 500, json: async () => ({}) };
  }
  throw new Error("unexpected fetch: " + url);
};

function loadRun(overrides) {
  return new Function(
    "params", "addTL", "_AGENT_RUN_TEXT_MAX_BYTES", "TextEncoder", "console",
    "_UI_DEBUG", "_lsGet", "_AGENT_LS", "el", "normalizeRouteMode",
    "_isAgentRunning", "_wireAgentSSE",
    `${fnText}; return agentRunStream;`
  )(env, overrides.addTL || addTL, overrides.maxBytes || _AGENT_RUN_TEXT_MAX_BYTES,
    TextEncoder, console, overrides._UI_DEBUG || false,
    overrides._lsGet || (() => "0"), overrides._AGENT_LS || {},
    overrides.el || (() => null), overrides.normalizeRouteMode || (() => "auto"),
    overrides._isAgentRunning || (() => false), overrides._wireAgentSSE || (() => { throw new Error("should not be reached"); }));
}

(async () => {
  // ── oversize: reject before any SSE touch ──
  lastError = null; sseTouches = 0;
  const big = "x".repeat(_AGENT_RUN_TEXT_MAX_BYTES + 1);
  await loadRun({}).call(null, { requestText: big });
  check("oversize requestText surfaces an addTL error naming the limit",
    typeof lastError === "string" && /256 KiB/.test(lastError), lastError || "no error");
  check("oversize requestText never touches the SSE source", sseTouches === 0, `sseTouches=${sseTouches}`);

  // ── boundary: exactly the cap proceeds to the first downstream step ──
  lastError = null; sseTouches = 0;
  const exact = "y".repeat(_AGENT_RUN_TEXT_MAX_BYTES);
  await loadRun({}).call(null, { requestText: exact });
  check("exact-cap requestText passes the guard (no client error)", lastError === null, lastError || "unexpected error");
  check("exact-cap requestText proceeds to the SSE source", sseTouches >= 1, `sseTouches=${sseTouches}`);

  // ── missing requestText passes the guard (server 400 is the authority) ──
  lastError = null; sseTouches = 0;
  await loadRun({}).call(null, {});
  check("missing requestText does not error client-side", lastError === null, lastError || "unexpected error");

  // ── source gate: the cap constant is pinned near the other client caps ──
  const srcHasConst = /const _AGENT_RUN_TEXT_MAX_BYTES = 256 \* 1024;/.test(src);
  check("source gate: _AGENT_RUN_TEXT_MAX_BYTES constant exists in agent-panel.js", srcHasConst);
  const gateInsideFn = /async function agentRunStream\(params\) \{[\s\S]*?_AGENT_RUN_TEXT_MAX_BYTES[\s\S]*?Cancel any in-progress stream/.test(src);
  check("source gate: the size check sits before the SSE-cancel block", gateInsideFn);

  console.log(`P15-9 agent run stream cap gate: ${passed} checks PASS`);
})().catch((e) => {
  console.error(`P15-9 agent run stream cap gate FAILED: ${e.message}`);
  process.exit(1);
});
