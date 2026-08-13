#!/usr/bin/env node
/**
 * Regression harness — P15-6: the design-chat send path bounds the message
 * size BEFORE any UI mutation.
 *
 * Bug: _designSendMessage forwarded the prompt-bar text verbatim to
 * POST /design/chat — a multi-MB paste caused a full round trip and only then
 * a server 413 (_DESIGN_TEXT_MAX_BYTES = 256 KiB). The image attach path got
 * its client-side pre-check in P15-2; the message path had none.
 *
 * Fix under test: _designSendMessage rejects messages over
 * _DESIGN_MSG_MAX_BYTES (256 KiB, mirroring the server) with an addTL error,
 * before history push / typing indicator / SSE start — nothing mutates.
 *
 * The REAL function text is sliced out of design-chat.js (brace-balanced) and
 * executed against stub globals (no test framework, no browser).
 * Run: node tests/js/test_design_message_cap.js
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
let lastInfo = null; // P20-2: info-level toasts (pending-attachment notice)
let history = [];
let appended = [];
let cancelled = 0;
let typingShown = 0;
let runStartErrors = 0;
let fetchCalls = 0;

function addTL(level, msg) { if (level === "error") lastError = msg; if (level === "info") lastInfo = msg; }
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
  return function () { throw new Error(`should not be reached for oversize message: ${name}`); };
}

const _designChat = { attachedImages: [], history, sessionId: null, sse: null };
const state = { repoRoot: "/tmp/repo" };
let _designUserScrolledUp = true;

// Must mirror the source constant; the source gate below pins the real value.
const _DESIGN_MSG_MAX_BYTES = 256 * 1024;

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
      _designUploadImages: async (imgs) => ({ ids: imgs.map((i) => "id-" + (i.name || "x")), failed: 0 }),
      addTL,
      _designUserScrolledUp,
      _designChat,
      state,
      _DESIGN_MSG_MAX_BYTES,
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
    "_designUploadImages",
    "addTL", "_designUserScrolledUp", "_designChat", "state", "_DESIGN_MSG_MAX_BYTES",
    "_lsGet", "_designRunKey", "document", "fetch",
    `${fnText}; return _designSendMessage;`
  )(fns._applySlashExpansionToPrompt, fns._designCancelTypewriter, fns._designRenderImagePreviews,
    fns._designAppendUserMsg, fns._designShowTypingIndicator, fns._designShowRunStartError,
    fns._designTrimHistory,
    fns._fetchTimeoutSignal, fns._getProviderApiKey, fns._getAgentSettings,
    fns._designUploadImages,
    fns.addTL, fns._designUserScrolledUp, fns._designChat, fns.state, fns._DESIGN_MSG_MAX_BYTES,
    fns._lsGet, fns._designRunKey, fns.document, fns.fetch);
}

function reset() {
  lastError = null;
  lastInfo = null;
  history = [];
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
const constLine = src.split("\n").find((l) => l.includes("_DESIGN_MSG_MAX_BYTES ="));
check("message cap constant declared in design-chat.js", !!constLine, "constant missing");
check("message cap is 256 KiB (mirrors server _DESIGN_TEXT_MAX_BYTES)",
  !!constLine && constLine.includes("256 * 1024"), constLine || "");
check("size guard runs before history push",
  fnText.indexOf("encode(msg).length") < fnText.indexOf("_designChat.history.push"),
  "guard must precede the first UI mutation");
check("size guard runs before any fetch",
  fnText.indexOf("encode(msg).length") < fnText.indexOf('fetch("/design/chat"'),
  "guard must precede the SSE start");

// ── behavior: oversize rejection (nothing downstream may run) ──
reset();
const big = "x".repeat(_DESIGN_MSG_MAX_BYTES + 1);
let env = makeEnv(big, { bomb: true });
let send = loadSend(env.fns);
send().then(() => {
  check("oversize message surfaces an addTL error naming the limit",
    typeof lastError === "string" && /256 KiB/.test(lastError), lastError || "no error");
  check("oversize message never touches history", history.length === 0, `history=${history.length}`);
  check("oversize message never appends a user bubble", appended.length === 0, `appended=${appended.length}`);
  check("oversize message never starts a typing indicator", typingShown === 0, `typingShown=${typingShown}`);
  check("oversize message never cancels a typewriter", cancelled === 0, `cancelled=${cancelled}`);
  check("oversize message never fires the SSE start", fetchCalls === 0, `fetchCalls=${fetchCalls}`);

  // ── boundary: exactly the cap proceeds through the normal send path ──
  reset();
  const exact = "y".repeat(_DESIGN_MSG_MAX_BYTES);
  env = makeEnv(exact, { bomb: false });
  send = loadSend(env.fns);
  return send();
}).then(() => {
  check("exact-cap message passes the guard (history push happened)", history.length === 1, `history=${history.length}`);
  check("exact-cap message appends a user bubble", appended.length === 1, `appended=${appended.length}`);
  check("exact-cap message starts the typing indicator", typingShown === 1, `typingShown=${typingShown}`);
  check("exact-cap message did not error client-side", lastError === null, lastError || "unexpected error");
  check("exact-cap message did reach the SSE start (fetch attempt)", fetchCalls === 1, `fetchCalls=${fetchCalls}`);
  check("prompt bar was cleared after send", env.ta.value === "", `value=${JSON.stringify(env.ta.value)}`);

  // ── small message passes untouched ──
  reset();
  env = makeEnv("fix the login bug", { bomb: false });
  send = loadSend(env.fns);
  return send();
}).then(() => {
  check("small message proceeds (history push)", history.length === 1, `history=${history.length}`);
  check("small message did not error", lastError === null, lastError || "unexpected error");

  // ── P20-2: pending-only attachments + empty text must NOT send ──
  reset();
  _designChat.attachedImages = [{ token: 1, bytes: 100, dataUrl: null }]; // FileReader still reading
  env = makeEnv("", { bomb: false });
  send = loadSend(env.fns);
  return send();
}).then(() => {
  check("pending-only: empty text + pending placeholder does NOT send",
    fetchCalls === 0 && history.length === 0 && appended.length === 0,
    `fetchCalls=${fetchCalls} history=${history.length}`);
  check("pending-only: user gets an info toast", /이미지를 읽는 중/.test(lastInfo || ""), lastInfo || "no info");

  // P20-2: a READY image + empty text still sends (with the image)
  reset();
  _designChat.attachedImages = [{ token: 2, bytes: 100, dataUrl: "data:image/png;base64,AA", mediaType: "image/png", name: "x.png" }];
  env = makeEnv("", { bomb: false });
  send = loadSend(env.fns);
  return send();
}).then(() => {
  check("ready image + empty text sends (history push)", history.length === 1, `history=${history.length}`);
  check("ready image + empty text reaches the run start", fetchCalls >= 1, `fetchCalls=${fetchCalls}`);
  console.log(`P15-6 design message cap gate: ${passed} checks PASS`);
}).catch((e) => {
  console.error(`P15-6 design message cap gate FAILED: ${e.message}`);
  process.exit(1);
});
