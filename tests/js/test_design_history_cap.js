#!/usr/bin/env node
/**
 * Regression harness — P15-14: the design-chat send path bounds the legacy
 * history payload BEFORE any UI mutation.
 *
 * Gap: _designSendMessage forwarded the 20-turn history slice to
 * POST /design/chat with no client-side bound — the server cap exists
 * (_DESIGN_HISTORY_MAX_BYTES = 1 MiB, P15-1) but is only reachable after the
 * history push / typing indicator / SSE start already happened. A single turn
 * with a huge assistant dump can push the slice past 1 MiB.
 *
 * Fix under test: _designSendMessage rejects a history slice over
 * _DESIGN_HISTORY_MAX_BYTES (1 MiB, mirroring the server) with an addTL
 * error, before history push / typing indicator / SSE start.
 *
 * The REAL function text is sliced out of design-chat.js (brace-balanced) and
 * executed against stub globals (no test framework, no browser).
 * Run: node tests/js/test_design_history_cap.js
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

const fnText = sliceFunction(src, "async function _designSendMessage()");

// ── stub environment ──
let lastError = null;
let history = [];
let appended = [];
let cancelled = 0;
let typingShown = 0;
let runStartErrors = 0;
let fetchCalls = 0;

function addTL(level, msg) { if (level === "error") lastError = msg; }
function _designCancelTypewriter() { cancelled++; }
function _designTrimHistory() {}  // no-op stub — P18-2 trim is covered by test_design_history_trim.js
function _designRenderImagePreviews() {}
function _designAppendUserMsg(m, imgs) { appended.push({ m, imgs }); }
function _designShowTypingIndicator() { typingShown++; }
function _designShowRunStartError() { runStartErrors++; }
function _fetchTimeoutSignal() { return undefined; }
function _getProviderApiKey() { return ""; }
function _getAgentSettings() { return { design_max_turns: 2 }; }

// Bomb stubs: any call means the guard failed to return early.
function bomb(name) {
  return function () { throw new Error(`should not be reached for oversize history: ${name}`); };
}

const _designChat = { attachedImages: [], history, sessionId: null, sse: null };
const state = { repoRoot: "/tmp/repo" };
let _designUserScrolledUp = true;

// Must mirror the source constant; the source gate below pins the real value.
const _DESIGN_MSG_MAX_BYTES = 256 * 1024;
const _DESIGN_HISTORY_MAX_BYTES = 1024 * 1024;

function makeEnv(taValue, opts) {
  opts = opts || {};
  const ta = { value: taValue };
  const getEl = (id) => {
    if (id === "prompt-input") return ta;
    return opts.getEl ? opts.getEl(id) : null;
  };
  return {
    ta,
    fns: {
      _applySlashExpansionToPrompt: async () => "none",
      _designCancelTypewriter: opts.bomb ? bomb("_designCancelTypewriter") : _designCancelTypewriter,
      _designRenderImagePreviews: opts.bomb ? bomb("_designRenderImagePreviews") : _designRenderImagePreviews,
      _designAppendUserMsg: opts.bomb ? bomb("_designAppendUserMsg") : _designAppendUserMsg,
      _designShowTypingIndicator: opts.bomb ? bomb("_designShowTypingIndicator") : _designShowTypingIndicator,
      _designShowRunStartError: opts.bomb ? bomb("_designShowRunStartError") : _designShowRunStartError,
      _designTrimHistory: opts.bomb ? bomb("_designTrimHistory") : _designTrimHistory,
      _fetchTimeoutSignal,
      _getProviderApiKey,
      _getAgentSettings,
      addTL,
      _designUserScrolledUp,
      _designChat,
      state,
      _DESIGN_MSG_MAX_BYTES,
      _DESIGN_HISTORY_MAX_BYTES,
      _lsGet: () => null,
      _designRunKey: () => "",
      document: { getElementById: getEl },
      fetch: async () => { fetchCalls++; throw new Error("net"); },
    },
  };
}

function loadSend(fns) {
  return new Function(
    "_applySlashExpansionToPrompt", "_designCancelTypewriter", "_designRenderImagePreviews",
    "_designAppendUserMsg", "_designShowTypingIndicator", "_designShowRunStartError",
    "_designTrimHistory",
    "_fetchTimeoutSignal", "_getProviderApiKey", "_getAgentSettings",
    "addTL", "_designUserScrolledUp", "_designChat", "state", "_DESIGN_MSG_MAX_BYTES",
    "_DESIGN_HISTORY_MAX_BYTES", "_lsGet", "_designRunKey", "document", "fetch",
    `${fnText}; return _designSendMessage;`
  )(fns._applySlashExpansionToPrompt, fns._designCancelTypewriter, fns._designRenderImagePreviews,
    fns._designAppendUserMsg, fns._designShowTypingIndicator, fns._designShowRunStartError,
    fns._designTrimHistory,
    fns._fetchTimeoutSignal, fns._getProviderApiKey, fns._getAgentSettings,
    fns.addTL, fns._designUserScrolledUp, fns._designChat, fns.state, fns._DESIGN_MSG_MAX_BYTES,
    fns._DESIGN_HISTORY_MAX_BYTES, fns._lsGet, fns._designRunKey, fns.document, fns.fetch);
}

function reset(hist) {
  lastError = null;
  history = hist || [];
  appended = [];
  cancelled = 0;
  typingShown = 0;
  runStartErrors = 0;
  fetchCalls = 0;
  _designChat.history = history;
  _designChat.attachedImages = [];
  _designChat.sessionId = null;
  _designChat.sse = null;
}

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gates ──
const constLine = src.split("\n").find((l) => l.includes("_DESIGN_HISTORY_MAX_BYTES ="));
check("history cap constant declared in design-chat.js", !!constLine, "constant missing");
check("history cap is 1 MiB (mirrors server _DESIGN_HISTORY_MAX_BYTES)",
  !!constLine && constLine.includes("1024 * 1024"), constLine || "");
check("history guard runs before history push",
  fnText.indexOf("encode(_histJson).length") < fnText.indexOf("_designChat.history.push"),
  "guard must precede the first UI mutation");
check("history guard runs before any fetch",
  fnText.indexOf("encode(_histJson).length") < fnText.indexOf('fetch("/design/chat"'),
  "guard must precede the SSE start");

// ── behavior: oversize history rejection (nothing downstream may run) ──
// The send slice is history.slice(-21, -1): with 2 turns only the FIRST is
// sent, so a huge first turn makes the payload oversize.
reset([{ role: "user", content: "x".repeat(_DESIGN_HISTORY_MAX_BYTES + 128) }, { role: "ai", content: "short" }]);
let env = makeEnv("hi", { bomb: true });
let send = loadSend(env.fns);
send().then(() => {
  check("oversize history surfaces an addTL error naming the limit",
    typeof lastError === "string" && /1 MiB limit/.test(lastError), lastError || "no error");
  check("oversize history never pushes a new turn", history.length === 2, `history=${history.length}`);
  check("oversize history never appends a user bubble", appended.length === 0, `appended=${appended.length}`);
  check("oversize history never starts a typing indicator", typingShown === 0, `typingShown=${typingShown}`);
  check("oversize history never cancels a typewriter", cancelled === 0, `cancelled=${cancelled}`);
  check("oversize history never fires the SSE start", fetchCalls === 0, `fetchCalls=${fetchCalls}`);

  // ── boundary: in-limit history proceeds through the normal send path ──
  reset([{ role: "user", content: "y".repeat(512 * 1024) }, { role: "ai", content: "short" }]);
  env = makeEnv("hi", { bomb: false });
  send = loadSend(env.fns);
  return send();
}).then(() => {
  check("in-limit history passes the guard (history push happened)", history.length === 3, `history=${history.length}`);
  check("in-limit history appends a user bubble", appended.length === 1, `appended=${appended.length}`);
  check("in-limit history starts the typing indicator", typingShown === 1, `typingShown=${typingShown}`);
  check("in-limit history did not error client-side", lastError === null, lastError || "unexpected error");
  check("in-limit history did reach the SSE start (fetch attempt)", fetchCalls === 1, `fetchCalls=${fetchCalls}`);

  // ── server-side sessions skip the legacy history check entirely ──
  reset([{ role: "user", content: "z".repeat(_DESIGN_HISTORY_MAX_BYTES + 128) }, { role: "ai", content: "short" }]);
  _designChat.sessionId = "sess-123";
  env = makeEnv("hi", { bomb: false });
  send = loadSend(env.fns);
  return send();
}).then(() => {
  check("server-side session bypasses the legacy history guard (send proceeds)",
    fetchCalls === 1, `fetchCalls=${fetchCalls}`);
  check("server-side session send did not error", lastError === null, lastError || "unexpected error");
  console.log(`P15-14 design history cap gate: ${passed} checks PASS`);
}).catch((e) => {
  console.error(`P15-14 design history cap gate FAILED: ${e.message}`);
  process.exit(1);
});
