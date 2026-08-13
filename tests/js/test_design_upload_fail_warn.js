#!/usr/bin/env node
/**
 * Regression harness — P16-1: design-chat image upload failures are reported
 * instead of being silently dropped.
 *
 * Bug: _designSendMessage aggregated uploads with
 *   Promise.all(...).then(ids => _buildAndFireSSE(ids.filter(Boolean)))
 * — any 413/429/network/30s-timeout collapsed to null and was silently
 * filtered, so the message went out without the image while the chat showed a
 * preview: the LLM never saw an image the user believes was attached. Zero
 * user feedback.
 *
 * Fix under test: _designUploadImages(imagesToUpload) → { ids, failed }
 * (REAL function text sliced from design-chat.js) counts failures and issues
 * an addTL error; the send path continues with the successful ids only.
 *
 * The REAL function text of BOTH _designUploadImages and _designSendMessage is
 * sliced out (brace-balanced) and executed against stub globals.
 * Run: node tests/js/test_design_upload_fail_warn.js
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
const upFnText = sliceFunction(src, "async function _designUploadImages(imagesToUpload)");

// ── stub environment ──
let lastError = null;
let history = [];
let appended = [];
let fetchCalls = 0;
let lastSseUrl = null;
let lastPostBody = null;   // body of POST /design/chat — where image_ids actually travel

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
  constructor(url) { this.url = url; }
  close() {}
}
function _designSetupSSEHandlers(sse) { lastSseUrl = sse.url; }

const _designChat = { attachedImages: [], history, sessionId: null, sse: null };
const state = { repoRoot: "/tmp/repo" };
let _designUserScrolledUp = true;

// Must mirror the source constant; the send path checks it before anything else.
const _DESIGN_MSG_MAX_BYTES = 256 * 1024;

function img(id, name) {
  return { dataUrl: `data:image/png;base64,${id}`, mediaType: "image/png", name: name || `${id}.png` };
}

// fetch router: upload outcomes per image; the run-start POST always succeeds
// and captures its body (the query params travel in the POST body — the SSE
// attach URL carries only the session id).
function makeFetch(uploadOutcomes) {
  let idx = 0;
  return async (url, opts) => {
    fetchCalls++;
    if (typeof url === "string" && url.startsWith("/design/chat/upload-image")) {
      const outcome = uploadOutcomes[Math.min(idx++, uploadOutcomes.length - 1)];
      if (outcome.throws) throw new Error("net down");
      return { ok: !!outcome.ok, status: outcome.ok ? 200 : 413, json: async () => outcome.json || {} };
    }
    if (typeof url === "string" && url.startsWith("/design/chat")) {
      try { lastPostBody = JSON.parse((opts && opts.body) || "{}"); } catch (_) { lastPostBody = {}; }
      return { ok: true, status: 200, json: async () => ({ session_id: "sess-1" }) };
    }
    throw new Error("unexpected fetch: " + url);
  };
}

function makeEnv(taValue, images, fetch) {
  const ta = { value: taValue };
  const getEl = (id) => (id === "prompt-input" ? ta : null);
  _designChat.attachedImages = images.slice();
  return {
    ta,
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
      _lsGet: () => null,
      _designRunKey: () => "",
      document: { getElementById: getEl },
      fetch,
      EventSource: FakeEventSource,
      _designSetupSSEHandlers,
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
    "_lsGet", "_designRunKey", "document", "fetch", "EventSource", "_designSetupSSEHandlers",
    `${upFnText}; ${sendFnText}; return _designSendMessage;`
  )(fns._applySlashExpansionToPrompt, fns._designCancelTypewriter, fns._designRenderImagePreviews,
    fns._designAppendUserMsg, fns._designShowTypingIndicator, fns._designShowRunStartError,
    fns._designTrimHistory,
    fns._fetchTimeoutSignal, fns._getProviderApiKey, fns._getAgentSettings,
    fns.addTL, fns._designUserScrolledUp, fns._designChat, fns.state, fns._DESIGN_MSG_MAX_BYTES,
    fns._lsGet, fns._designRunKey, fns.document, fns.fetch, fns.EventSource, fns._designSetupSSEHandlers);
}

function reset() {
  lastError = null;
  history = [];
  appended = [];
  fetchCalls = 0;
  lastSseUrl = null;
  lastPostBody = null;
  _designChat.history = history;
  _designChat.attachedImages = [];
  _designChat.sessionId = null;
  _designChat.sse = null;
}

function postedImageIds() {
  return (lastPostBody && lastPostBody.image_ids) || null;
}

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gates ──
check("_designUploadImages helper exists in design-chat.js",
  src.includes("async function _designUploadImages"), "helper missing");
check("helper issues an addTL error on failure",
  upFnText.includes("addTL(") && upFnText.includes("failed to upload"), "warning missing");
check("helper returns the { ids, failed } shape", upFnText.includes("return { ids, failed }"), "shape changed");
const upStart = sendFnText.indexOf("_designUploadImages(imagesToUpload)");
const upEnd = sendFnText.indexOf("_buildAndFireSSE(ids)", upStart);
check("send path routes uploads through the helper", upStart >= 0, "helper call missing");
check("send path no longer filter(Boolean)s upload results (silent drop gone)",
  upEnd > upStart && !sendFnText.slice(upStart, upEnd).includes(".filter(Boolean)"),
  "silent-drop pattern still present");
// P18-3: the `target_files` query param was a dead contract — the server only
// logged it and never pre-loaded content; the "Target file:" line rides in the
// message text itself. The send path must not re-introduce the wire usage
// (the explanatory comment may mention the identifier).
check("P18-3: dead target_files param removed from the send path",
  !sendFnText.includes('qs.set("target_files"'), "target_files still sent");

// Uploads are fire-and-forget inside _designSendMessage (async continuation),
// so after `await send()` the SSE may not be attached yet. All stubs resolve
// from already-resolved promises, so the whole upload → SSE chain completes
// during the microtask drain — one macrotask flush makes it deterministic.
function flush() {
  return new Promise((r) => setTimeout(r, 0));
}

(async () => {
  let env, send;

  // ── all uploads succeed: no warning, all ids attached in order ──
  reset();
  env = makeEnv("look at this", [img("AA"), img("BB"), img("CC")], makeFetch([
    { ok: true, json: { image_id: "id-AA" } },
    { ok: true, json: { image_id: "id-BB" } },
    { ok: true, json: { image_id: "id-CC" } },
  ]));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("all-ok: no warning is issued", lastError === null, lastError || "unexpected error");
  check("all-ok: run start fired (SSE attached)", lastSseUrl !== null, "send aborted");
  check("all-ok: POST /design/chat carries all ids in original order",
    postedImageIds() === "id-AA,id-BB,id-CC", JSON.stringify(lastPostBody));
  check("all-ok: user bubble still rendered with the original images",
    appended.length === 1 && appended[0].imgs.length === 3, `appended=${appended.length}`);
  check("all-ok: exactly 3 uploads + 1 run start attempted",
    fetchCalls === 4, `fetchCalls=${fetchCalls}`);
  check("P18-3: SSE URL carries no target_files param even with target files present",
    !(lastSseUrl || "").includes("target_files"), lastSseUrl || "no url");

  // ── partial failure (1 of 3): warning + SSE carries only the successes ──
  reset();
  env = makeEnv("look at this", [img("AA"), img("BB"), img("CC")], makeFetch([
    { ok: true, json: { image_id: "id-AA" } },
    { ok: false },
    { ok: true, json: { image_id: "id-CC" } },
  ]));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("partial: warning names the failed count",
    typeof lastError === "string" && /1 image failed/.test(lastError), lastError || "no error");
  check("partial: warning explains the message still went out",
    /message sent without them/.test(lastError || ""), lastError || "no error");
  check("partial: POST carries only the successful ids",
    postedImageIds() === "id-AA,id-CC", JSON.stringify(lastPostBody));
  check("partial: failed upload did not abort the send",
    lastSseUrl !== null, "send aborted");

  // ── total failure (413 + network throw + missing image_id): text-only send ──
  reset();
  env = makeEnv("plain text", [img("AA"), img("BB"), img("CC")], makeFetch([
    { ok: false },
    { throws: true },
    { ok: true, json: {} },
  ]));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("all-fail: warning counts all three failures",
    typeof lastError === "string" && /3 images failed/.test(lastError), lastError || "no error");
  check("all-fail: SSE still fires (text-only send)", lastSseUrl !== null, "send aborted");
  check("all-fail: POST carries no image_ids param", postedImageIds() === null, JSON.stringify(lastPostBody));

  // ── no images: unchanged path, no uploads, no warning ──
  reset();
  env = makeEnv("plain text", [], makeFetch([]));
  send = loadSend(env.fns);
  await send();
  await flush();
  check("no-images: no upload fetch attempted", fetchCalls === 1, `fetchCalls=${fetchCalls}`);
  check("no-images: no warning", lastError === null, lastError || "unexpected error");
  check("no-images: SSE fired normally", lastSseUrl !== null, "send aborted");
})().then(() => {
  console.log(`P16-1 design upload failure warning gate: ${passed} checks PASS`);
}).catch((e) => {
  console.error(`P16-1 design upload failure warning gate FAILED: ${e.message}`);
  process.exit(1);
});
