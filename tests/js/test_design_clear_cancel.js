#!/usr/bin/env node
/**
 * Regression harness — P20-2: "new session" (clear chat) asks the server to
 * cancel the previous design run instead of only closing the SSE stream.
 *
 * Bug: _designClearChat closed _designChat.sse but never cancelled the run.
 * The old run kept executing (LLM calls, tool runs, file edits) until the
 * SSE-disconnect grace timer (~30s) fired, and its add_turn could write into
 * the session file right after the user believed the chat was cleared.
 * (P18-1 covered the send path only; the clear path was a second entry.)
 *
 * Fix under test: right after closing the SSE, the clear path reads the
 * persisted previous RUN session id (_designRunKey, localStorage) and fires
 * a fire-and-forget POST /agent/cancel/{id}, then removes the stored run id.
 * No stored id (nothing running) → no-op.
 *
 * REAL function text of _designClearChat, _designRunKey and _designSessionKey
 * is sliced out (brace-balanced) and executed against stub globals.
 * Run: node tests/js/test_design_clear_cancel.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "design-chat.js"), "utf8");

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

const clearFnText = sliceFunction(src, "function _designClearChat(silent)");
const runKeyFnText = sliceFunction(src, "function _designRunKey()");
const sessKeyFnText = sliceFunction(src, "function _designSessionKey()");

// ── stub environment ──
const lsStore = {};
const ls = {
  getItem: (k) => (k in lsStore ? lsStore[k] : null),
  setItem: (k, v) => { lsStore[k] = String(v); },
  removeItem: (k) => { delete lsStore[k]; },
};

let passed = 0;
function check(name, ok, detail) {
  passed++;
  if (!ok) {
    console.error(`  FAIL: ${name}${detail ? ` (${detail})` : ""}`);
    process.exit(1);
  }
  console.log(`  ok: ${name}`);
}

const calls = [];
function makeFetch() {
  return async (url, opts) => {
    calls.push({ url: String(url), method: (opts && opts.method) || "GET" });
    if (typeof url === "string" && url.startsWith("/agent/cancel/")) {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (typeof url === "string" && url.startsWith("/design/session/")) {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    throw new Error("unexpected fetch: " + url);
  };
}

function loadClear(overrides) {
  return new Function(
    "silent", "confirm", "_designCleanupListeners", "initPromptFeatureToggles",
    "_designChat", "_lsRemove", "_lsGet", "_designSessionKey", "_designRunKey",
    "_designRenderImagePreviews", "document", "fetch", "_fetchTimeoutSignal",
    "currentRepoRoot", "_UI_DEBUG", "console", "state",
    `${clearFnText}\n${runKeyFnText}\n${sessKeyFnText}\nreturn _designClearChat;`
  )(true, overrides.confirm || (() => true),
    overrides._designCleanupListeners || (() => {}),
    overrides.initPromptFeatureToggles || (() => {}),
    overrides._designChat || { history: [], sessionId: null, attachedImages: [], attachedTotalBytes: 0, sse: null },
    overrides._lsRemove || ls.removeItem, overrides._lsGet || ls.getItem,
    overrides._designSessionKey, overrides._designRunKey,
    overrides._designRenderImagePreviews || (() => {}),
    overrides.document || { getElementById: () => null },
    overrides.fetch || makeFetch(), overrides._fetchTimeoutSignal || (() => undefined),
    overrides.currentRepoRoot || (() => "/tmp/repo"), overrides._UI_DEBUG || false,
    console, overrides.state || { repoRoot: "/tmp/repo" });
}

// Deterministic localStorage keys derived from the same repo path the real
// functions use (currentRepoRoot stub → "/tmp/repo").
const runKey = "asicode.design.run_session_id." +
  btoa(unescape(encodeURIComponent("/tmp/repo"))).slice(0, 16).replace(/[/+=]/g, "_");
const sessKey = "asicode.design.session_id." +
  btoa(unescape(encodeURIComponent("/tmp/repo"))).slice(0, 16).replace(/[/+=]/g, "_");

(async () => {
  // ── stored run id → cancel POST + stored id removed ──
  Object.keys(lsStore).forEach((k) => delete lsStore[k]);
  lsStore[runKey] = "run-9";
  lsStore[sessKey] = "sess-old";
  calls.length = 0;
  const chat = { history: [], sessionId: "sess-old", attachedImages: [], attachedTotalBytes: 0, sse: null };
  await loadClear({ _designChat: chat })();
  const cancels = calls.filter((c) => c.url.startsWith("/agent/cancel/"));
  check("clear cancels the previous run via /agent/cancel POST",
    cancels.length === 1 && cancels[0].url === "/agent/cancel/run-9" && cancels[0].method === "POST",
    JSON.stringify(calls));
  check("clear removes the stored run id (no stale cancel on next message)",
    !(runKey in lsStore), `runKey still present: ${lsStore[runKey]}`);
  check("clear still deletes the server-side session file",
    calls.some((c) => c.url.startsWith("/design/session/sess-old") && c.method === "DELETE"),
    JSON.stringify(calls));

  // ── no stored run id (nothing running) → no cancel, no-op ──
  Object.keys(lsStore).forEach((k) => delete lsStore[k]);
  lsStore[sessKey] = "sess-old";
  calls.length = 0;
  await loadClear({ _designChat: { history: [], sessionId: "sess-old", attachedImages: [], attachedTotalBytes: 0, sse: null } })();
  const cancels2 = calls.filter((c) => c.url.startsWith("/agent/cancel/"));
  check("clear without a stored run id sends no cancel", cancels2.length === 0, JSON.stringify(calls));

  // ── source gates ──
  const hasCancel = /_lsGet\(_designRunKey\(\)\)/.test(src) && /\/agent\/cancel\//.test(src);
  check("source gate: clear path reads _designRunKey and POSTs /agent/cancel", hasCancel);
  const hasRemove = /_lsRemove\(_designRunKey\(\)\)/.test(src);
  check("source gate: clear path removes the stored run id", hasRemove);
  const orderOk = /_designChat\.sse\) \{[\s\S]*?\/agent\/cancel\//.test(src);
  check("source gate: cancel block sits after the SSE close", orderOk);

  console.log(`P20-2 design clear-cancel gate: ${passed} checks PASS`);
})().catch((e) => {
  console.error(`P20-2 design clear-cancel gate FAILED: ${e.message}`);
  process.exit(1);
});
