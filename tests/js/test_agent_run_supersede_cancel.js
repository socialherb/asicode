#!/usr/bin/env node
/**
 * Regression harness — P20-1: starting a NEW agent run asks the server to
 * cancel the previous run instead of only closing the SSE stream.
 *
 * Bug: agentRunStream closed _agentSSESource but never POSTed
 * /agent/cancel/{prevSessionId}. The old run kept executing (LLM calls, tool
 * runs, file edits) until the SSE-disconnect grace timer (~30s) fired — a
 * quick re-run ran two agents in parallel. P18-1 fixed the same class of bug
 * for design-chat only; the agent panel's run entry still had it.
 *
 * Fix under test: right after closing the old SSE, a fresh run (continueMode
 * !== true) reads the previous run's session id from _agentSessionId OR
 * sessionStorage ("asr_agent_session_id", which survives a page reload so a
 * stale run started before the reload is still cancelled) and fires a
 * fire-and-forget POST /agent/cancel/{id}. continueMode re-attaches to the
 * existing gated run (approval/user-input pause) — cancelling it would kill
 * the run being resumed, so it is exempt. First run (no id) → no-op.
 *
 * REAL function text of agentRunStream is sliced out (brace-balanced) and
 * executed against stub globals (no test framework, no browser).
 * Run: node tests/js/test_agent_run_supersede_cancel.js
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
let passed = 0;
function check(name, ok, detail) {
  passed++;
  if (!ok) {
    console.error(`  FAIL: ${name}${detail ? ` (${detail})` : ""}`);
    process.exit(1);
  }
  console.log(`  ok: ${name}`);
}

const _AGENT_RUN_TEXT_MAX_BYTES = 256 * 1024;  // must mirror the source constant
const calls = [];  // every fetch: { url, method }

// The REAL function closes _agentSSESource from its scope chain (Function
// scope → global). Expose a no-op global so the SSE-cancel block runs.
Object.defineProperty(globalThis, "_agentSSESource", {
  get() { return null; },
  set(v) {},
  configurable: true,
});

function makeFetch() {
  return async (url, opts) => {
    calls.push({ url: String(url), method: (opts && opts.method) || "GET" });
    if (typeof url === "string" && url.startsWith("/agent/cancel/")) {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (typeof url === "string" && url.startsWith("/agent/run")) {
      // stop the run-start path early (the harness only tests the supersede)
      return { ok: false, status: 500, json: async () => ({}) };
    }
    throw new Error("unexpected fetch: " + url);
  };
}

function makeSessionStorage(initial) {
  const store = Object.assign({}, initial || {});
  return {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
  };
}

function loadRun(overrides) {
  return new Function(
    "params", "fetch", "_agentSessionId", "sessionStorage", "addTL",
    "_AGENT_RUN_TEXT_MAX_BYTES", "TextEncoder", "console", "_UI_DEBUG",
    "_lsGet", "_AGENT_LS", "el", "normalizeRouteMode", "_isAgentRunning",
    "_wireAgentSSE", "_agentSetStatus", "_fetchTimeoutSignal",
    `${fnText}; return agentRunStream;`
  )(null, overrides.fetch || makeFetch(), overrides.agentSessionId === undefined ? null : overrides.agentSessionId,
    overrides.sessionStorage || makeSessionStorage(),
    overrides.addTL || (() => {}), overrides.maxBytes || _AGENT_RUN_TEXT_MAX_BYTES,
    TextEncoder, console, overrides._UI_DEBUG || false,
    overrides._lsGet || (() => "0"), overrides._AGENT_LS || {},
    overrides.el || (() => null), overrides.normalizeRouteMode || (() => "auto"),
    overrides._isAgentRunning || (() => false),
    overrides._wireAgentSSE || (() => { throw new Error("should not be reached"); }),
    overrides._agentSetStatus || (() => {}),
    overrides._fetchTimeoutSignal || (() => undefined));
}

const freshParams = { requestText: "hi", repoRoot: "/tmp/repo" };

(async () => {
  // ── fresh run with a previous id → cancel POST fires FIRST ──
  calls.length = 0;
  await loadRun({ agentSessionId: "prev-1" }).call(null, freshParams);
  const cancels = calls.filter((c) => c.url.startsWith("/agent/cancel/"));
  check("fresh run cancels the previous run via /agent/cancel POST",
    cancels.length === 1 && cancels[0].url === "/agent/cancel/prev-1" && cancels[0].method === "POST",
    JSON.stringify(calls));
  check("cancel is the first request (starts before the run-start POST)",
    calls.length >= 1 && calls[0].url === "/agent/cancel/prev-1", JSON.stringify(calls));

  // ── continueMode re-attaches to the gated run → no cancel ──
  calls.length = 0;
  await loadRun({ agentSessionId: "prev-1" }).call(null, Object.assign({ continueMode: true }, freshParams));
  const cancels2 = calls.filter((c) => c.url.startsWith("/agent/cancel/"));
  check("continueMode does NOT cancel the previous run (it is the run being resumed)",
    cancels2.length === 0, JSON.stringify(calls));

  // ── first run (no id anywhere) → no-op ──
  calls.length = 0;
  await loadRun({ agentSessionId: null, sessionStorage: makeSessionStorage() }).call(null, freshParams);
  const cancels3 = calls.filter((c) => c.url.startsWith("/agent/cancel/"));
  check("first run (no stored id) sends no cancel", cancels3.length === 0, JSON.stringify(calls));

  // ── reload: module variable lost, sessionStorage survives → still cancels ──
  calls.length = 0;
  await loadRun({
    agentSessionId: null,
    sessionStorage: makeSessionStorage({ asr_agent_session_id: "stale-9" }),
  }).call(null, freshParams);
  const cancels4 = calls.filter((c) => c.url.startsWith("/agent/cancel/"));
  check("reload case: stale run id from sessionStorage is still cancelled",
    cancels4.length === 1 && cancels4[0].url === "/agent/cancel/stale-9" && cancels4[0].method === "POST",
    JSON.stringify(calls));

  // ── source gates: the supersede block exists and sits AFTER the SSE-cancel ──
  const hasCancel = /agent\/cancel\/" \+ encodeURIComponent\(_prevSid\)/.test(src);
  check("source gate: agentRunStream contains the /agent/cancel POST", hasCancel);
  const hasContinueExempt = /continueMode === true/.test(src);
  check("source gate: continueMode exemption is present", hasContinueExempt);
  const hasSessionStorage = /sessionStorage\.getItem\("asr_agent_session_id"\)/.test(src);
  check("source gate: sessionStorage reload fallback is present", hasSessionStorage);
  const orderOk = /Cancel any in-progress stream[\s\S]*?supersede any in-flight agent run/.test(src);
  check("source gate: supersede block sits after the SSE-cancel block", orderOk);

  console.log(`P20-1 agent run supersede-cancel gate: ${passed} checks PASS`);
})().catch((e) => {
  console.error(`P20-1 agent run supersede-cancel gate FAILED: ${e.message}`);
  process.exit(1);
});
