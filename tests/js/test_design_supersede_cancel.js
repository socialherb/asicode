#!/usr/bin/env node
/**
 * Regression harness — P18-1: sending a new design-chat message asks the
 * server to cancel the previous run instead of only closing the SSE.
 *
 * Bug: _designSendMessage closed _designChat.sse but never POSTed
 * /agent/cancel/{prevRunId}. The old run kept executing (LLM calls, tool
 * runs, file edits) until the SSE-disconnect grace timer (~30s) fired and its
 * cancel_event propagated — meanwhile its assistant turn landed AFTER the new
 * user turn (turn-order skew on reload).
 *
 * Fix under test: right after closing the old SSE, the send path reads the
 * persisted previous RUN session id (_designRunKey, localStorage) and fires a
 * fire-and-forget POST /agent/cancel/{id}, mirroring the cancel-button
 * handler. The server's per-design-session registry (P18-1) cancels the old
 * run anyway when the new POST lands; this starts the cancel immediately,
 * before the upload round-trips. First message (no stored id) → no-op.
 *
 * REAL function text of _designSendMessage, _designRunKey and _lsGet is
 * sliced out (brace-balanced) and executed against stub globals.
 * Run: node tests/js/test_design_supersede_cancel.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "design-chat.js"), "utf8");

// ── Slice helper: brace-balanced from `anchor` to its closing } ──
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

const sendFnText = sliceFunction(src, "async function _designSendMessage()");
const runKeyFnText = sliceFunction(src, "function _designRunKey()");
const lsGetFnText = sliceFunction(src, "function _lsGet(key)");

// ── stub environment ──
const lsStore = {};
global.localStorage = {
  getItem: (k) => (k in lsStore ? lsStore[k] : null),
  setItem: (k, v) => { lsStore[k] = String(v); },
  removeItem: (k) => { delete lsStore[k]; },
};

let lastError = null;
let history = [];
let appended = [];
const calls = [];        // every fetch: { url, method }
let lastSseUrl = null;
let sseClosed = false;

function addTL(level, msg) { if (level === "error") lastError = msg; }
function _designCancelTypewriter() {}
function _designTrimHistory() {}  // no-op stub — P18-2 trim is covered by test_design_history_trim.js
function _designRenderImagePreviews() {}
function _designAppendUserMsg(m, imgs) { appended.push({ m, imgs }); }
function _designShowTypingIndicator() {}
function _designShowRunStartError() {}
function _fetchTimeoutSignal() { return undefined; }
function _getProviderApiKey() { return ""; }
function _getAgentSettings() { return { design_max_turns: 2 }; }

class FakeEventSource {
  constructor(url) { this.url = url; this.closed = false; }
  close() { this.closed = true; sseClosed = true; }
}
function _designSetupSSEHandlers(sse) { lastSseUrl = sse.url; }

const _designChat = { attachedImages: [], history, sessionId: null, sse: null };
const state = { repoRoot: "/tmp/repo" };
let _designUserScrolledUp = true;

// Must mirror the source constants; the send path checks them before anything else.
const _DESIGN_MSG_MAX_BYTES = 256 * 1024;
const _DESIGN_HISTORY_MAX_BYTES = 1024 * 1024;

// Deterministic localStorage key _designRunKey() derives from state.repoRoot.
const runKey = "asicode.design.run_session_id." +
  btoa(unescape(encodeURIComponent("/tmp/repo"))).slice(0, 16).replace(/[/+=]/g, "_");

function makeFetch(cancelOutcome) {
  return async (url, opts) => {
    calls.push({ url: String(url), method: (opts && opts.method) || "GET" });
    if (typeof url === "string" && url.startsWith("/agent/cancel/")) {
      if (cancelOutcome === "throws") throw new Error("net down");
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (typeof url === "string" && url.startsWith("/design/chat")) {
      return { ok: true, status: 200, json: async () => ({ session_id: "sess-new" }) };
    }
    throw new Error("unexpected fetch: " + url);
  };
}

function makeEnv(fetch) {
  const ta = { value: "second message" };
  const getEl = (id) => (id === "prompt-input" ? ta : null);
  _designChat.attachedImages = [];
  _designChat.history = history;
  return {
    fns: {
      _applySlashExpansionToPrompt: async () => "none",
      _designCancelTypewriter,
      _designRenderImagePreviews,
      _designAppendUserMsg,
      _designShowTypingIndicator,
      _designShowRunStartError,
      _designTrimHistory,
      _fetchTimeoutSignal,
      _getProviderApiKey,
      _getAgentSettings,
      addTL,
      _designUserScrolledUp,
      _designChat,
      state,
      _DESIGN_MSG_MAX_BYTES,
      _DESIGN_HISTORY_MAX_BYTES,
      currentRepoRoot: undefined,
      document: { getElementById: getEl },
      fetch,
      EventSource: FakeEventSource,
      _designSetupSSEHandlers,
    },
  };
}

// NOTE: _lsGet and _designRunKey are NOT module-scope identifiers — their REAL
// function text is declared inside the new Function body (hoisted), so they
// are not passed as parameters; _lsGet reads the global localStorage stub and
// _designRunKey derives the key from the state param + global btoa/unescape.
function loadSend(fns) {
  return new Function(
    "_applySlashExpansionToPrompt", "_designCancelTypewriter", "_designRenderImagePreviews",
    "_designAppendUserMsg", "_designShowTypingIndicator", "_designShowRunStartError",
    "_designTrimHistory",
    "_fetchTimeoutSignal", "_getProviderApiKey", "_getAgentSettings",
    "addTL", "_designUserScrolledUp", "_designChat", "state", "_DESIGN_MSG_MAX_BYTES",
    "_DESIGN_HISTORY_MAX_BYTES", "currentRepoRoot",
    "document", "fetch", "EventSource", "_designSetupSSEHandlers",
    `${lsGetFnText}; ${runKeyFnText}; ${sendFnText}; return _designSendMessage;`
  )(fns._applySlashExpansionToPrompt, fns._designCancelTypewriter, fns._designRenderImagePreviews,
    fns._designAppendUserMsg, fns._designShowTypingIndicator, fns._designShowRunStartError,
    fns._designTrimHistory,
    fns._fetchTimeoutSignal, fns._getProviderApiKey, fns._getAgentSettings,
    fns.addTL, fns._designUserScrolledUp, fns._designChat, fns.state, fns._DESIGN_MSG_MAX_BYTES,
    fns._DESIGN_HISTORY_MAX_BYTES, fns.currentRepoRoot,
    fns.document, fns.fetch, fns.EventSource, fns._designSetupSSEHandlers);
}

function reset() {
  lastError = null;
  history = [];
  appended = [];
  calls.length = 0;
  lastSseUrl = null;
  sseClosed = false;
  for (const k of Object.keys(lsStore)) delete lsStore[k];
  _designChat.history = history;
  _designChat.attachedImages = [];
  _designChat.sessionId = null;
  _designChat.sse = null;
}

function cancelCalls() {
  return calls.filter((c) => c.url.startsWith("/agent/cancel/"));
}

function flush() {
  return new Promise((r) => setTimeout(r, 0));
}

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gates ──
check("send path reads the persisted previous run id",
  sendFnText.includes("_lsGet(_designRunKey())"), "prev-run read missing");
check("send path fires POST /agent/cancel for the previous run",
  sendFnText.includes('/agent/cancel/" + encodeURIComponent(_prevRunId)'), "cancel POST missing");
check("cancel POST is fire-and-forget (swallows errors)",
  sendFnText.includes(".catch(() => {})"), "unhandled rejection risk");
const closeIdx = sendFnText.indexOf("_designChat.sse = null;");
const cancelIdx = sendFnText.indexOf("_lsGet(_designRunKey())");
check("cancel fires right after the SSE close (before uploads/run start)",
  cancelIdx > closeIdx && cancelIdx >= 0, "ordering changed");
check("_designRunKey derives the per-repo run key",
  runKeyFnText.includes("asicode.design.run_session_id"), "run key namespace changed");

(async () => {
  let env, send;

  // ── previous run id present + active SSE: cancel POST fires, SSE closed, new run starts ──
  reset();
  lsStore[runKey] = "run-old";
  _designChat.sse = new FakeEventSource("old-sse");
  env = makeEnv(makeFetch("ok"));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("prev+active: cancel POST fires for the previous run id",
    cancelCalls().length === 1 && cancelCalls()[0].url === "/agent/cancel/run-old",
    JSON.stringify(calls));
  check("prev+active: cancel is the FIRST request (before the new run start)",
    calls.length >= 1 && calls[0].url === "/agent/cancel/run-old", JSON.stringify(calls));
  check("prev+active: cancel uses POST",
    cancelCalls()[0].method === "POST", "method=" + cancelCalls()[0].method);
  check("prev+active: old SSE is closed",
    sseClosed, "old sse not closed");
  check("prev+active: new SSE attached in its place",
    _designChat.sse !== null && _designChat.sse.url === "/agent/attach/sess-new",
    JSON.stringify(_designChat.sse && _designChat.sse.url));
  check("prev+active: new run still starts",
    lastSseUrl !== null && /\/agent\/attach\/sess-new/.test(lastSseUrl), lastSseUrl || "no sse");

  // ── no previous run id (first message): no cancel POST, normal flow ──
  reset();
  env = makeEnv(makeFetch("ok"));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("first-message: no cancel POST fired", cancelCalls().length === 0, JSON.stringify(calls));
  check("first-message: run start proceeds normally", lastSseUrl !== null, "send aborted");

  // ── stale id without an active SSE (reload case): cancel still fires ──
  reset();
  lsStore[runKey] = "run-stale";
  env = makeEnv(makeFetch("ok"));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("reload-case: stale run id still cancelled server-side",
    cancelCalls().length === 1 && cancelCalls()[0].url === "/agent/cancel/run-stale",
    JSON.stringify(calls));
  check("reload-case: new run starts", lastSseUrl !== null, "send aborted");

  // ── cancel POST network failure: swallowed, send continues ──
  reset();
  lsStore[runKey] = "run-old";
  env = makeEnv(makeFetch("throws"));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("cancel-fail: send not aborted by the cancel fetch", lastSseUrl !== null, "send aborted");
  check("cancel-fail: no error surfaced for the background cancel",
    lastError === null, lastError || "unexpected error");
})().then(() => {
  console.log(`P18-1 design supersede-cancel gate: ${passed} checks PASS`);
}).catch((e) => {
  console.error(`P18-1 design supersede-cancel gate FAILED: ${e.message}`);
  process.exit(1);
});
