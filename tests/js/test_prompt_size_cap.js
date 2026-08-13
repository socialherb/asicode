#!/usr/bin/env node
/**
 * Regression harness — P15-7: the main prompt bar bounds the prompt size
 * client-side, mirroring the server caps (256 KiB on /agent/run request_text,
 * /agent/message, and /edit/run prompt).
 *
 * Fix under test: _promptWithinLimit() rejects text over _PROMPT_MAX_BYTES
 * with an addTL error, and the three send entry points (_smartRunAfterSlash,
 * runOnly, _launchAgent) each call it BEFORE any downstream work.
 *
 * The REAL function text is sliced out of ui.js (brace-balanced) and executed
 * against stub globals (no test framework, no browser).
 * Run: node tests/js/test_prompt_size_cap.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "ui.js"), "utf8");

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

const helperText = sliceFunction(src, "function _promptWithinLimit(text)");
const smartText = sliceFunction(src, "function _smartRunAfterSlash()");
const runOnlyText = sliceFunction(src, "async function runOnly()");
const launchText = sliceFunction(src, "async function _launchAgent(extraMaxTurns = 0");

// ── stub environment ──
let lastError = null;
function addTL(level, msg) { if (level === "error") lastError = msg; }

// Must mirror the source constant; the source gate below pins the real value.
const _PROMPT_MAX_BYTES = 256 * 1024;

const runHelper = new Function("addTL", "_PROMPT_MAX_BYTES", `${helperText}; return _promptWithinLimit;`);
const _promptWithinLimit = runHelper(addTL, _PROMPT_MAX_BYTES);

function bomb(name) {
  return function () { throw new Error(`should not be reached for oversize prompt: ${name}`); };
}

function makeSmartEnv(taValue) {
  const elStub = (id) => (id === "prompt-input" ? { value: taValue } : null);
  return {
    el: elStub,
    fns: {
      _promptWithinLimit,
      addTL,
      document: { getElementById: () => null },
      _designSendMessage: bomb("_designSendMessage"),
      _sendMidTaskMessage: bomb("_sendMidTaskMessage"),
      runOnly: bomb("runOnly"),
      _launchAgent: bomb("_launchAgent"),
      normalizeRouteMode: bomb("normalizeRouteMode"),
      _isAgentRunning: bomb("_isAgentRunning"),
      _promptSuggestsAgent: bomb("_promptSuggestsAgent"),
    },
  };
}

function loadSmart(env) {
  return new Function(
    "el", "_promptWithinLimit", "addTL", "document",
    "_designSendMessage", "_sendMidTaskMessage", "runOnly", "_launchAgent",
    "normalizeRouteMode", "_isAgentRunning", "_promptSuggestsAgent",
    `${smartText}; return _smartRunAfterSlash;`
  )(env.el, env.fns._promptWithinLimit, env.fns.addTL, env.fns.document,
    env.fns._designSendMessage, env.fns._sendMidTaskMessage, env.fns.runOnly, env.fns._launchAgent,
    env.fns.normalizeRouteMode, env.fns._isAgentRunning, env.fns._promptSuggestsAgent);
}

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gates ──
const constLine = src.split("\n").find((l) => l.includes("_PROMPT_MAX_BYTES ="));
check("prompt cap constant declared in ui.js", !!constLine, "constant missing");
check("prompt cap is 256 KiB (mirrors server caps)",
  !!constLine && constLine.includes("256 * 1024"), constLine || "");
check("guard wired into _smartRunAfterSlash", smartText.includes("_promptWithinLimit(req)"), "missing");
check("guard wired into _launchAgent", launchText.includes("_promptWithinLimit(req)"), "missing");
check("guard wired into runOnly", runOnlyText.includes("_promptWithinLimit(text)"), "missing");
check("smartRun guard precedes the mid-task injection",
  smartText.indexOf("_promptWithinLimit(req)") < smartText.indexOf("_sendMidTaskMessage(req)"),
  "guard must run before any branch");
check("launchAgent guard precedes the run start",
  launchText.indexOf("_promptWithinLimit(req)") < launchText.indexOf("agentRunStream("),
  "guard must precede the run start");
check("runOnly guard precedes the send",
  runOnlyText.indexOf("_promptWithinLimit(text)") < runOnlyText.indexOf('apiPost("/edit/run"'),
  "guard must precede the fetch");

// ── helper behavior ──
check("oversize helper returns false", _promptWithinLimit("x".repeat(_PROMPT_MAX_BYTES + 1)) === false, "");
check("oversize helper surfaces an addTL error naming the limit",
  typeof lastError === "string" && /256 KiB/.test(lastError), lastError || "no error");
lastError = null;
check("exact-cap helper returns true", _promptWithinLimit("y".repeat(_PROMPT_MAX_BYTES)) === true, "");
check("exact-cap helper does not error", lastError === null, lastError || "unexpected error");
check("small helper returns true", _promptWithinLimit("fix the login bug") === true, "");
check("empty helper returns true (0 bytes)", _promptWithinLimit("") === true, "");
check("multibyte counts UTF-8 bytes (cap/3 + 1 Korean chars > cap)",
  _promptWithinLimit("가".repeat(Math.floor(_PROMPT_MAX_BYTES / 3) + 1)) === false,
  "3-byte chars must count as 3 bytes");

// ── entry-point guard: oversize prompt blocks before any branch ──
lastError = null;
const env = makeSmartEnv("x".repeat(_PROMPT_MAX_BYTES + 1));
const smart = loadSmart(env);
smart();
check("smartRun blocks oversize prompt (no branch ran)", true, "bomb stub would have thrown");
check("smartRun surfaces an addTL error naming the limit",
  typeof lastError === "string" && /256 KiB/.test(lastError), lastError || "no error");

// ── entry-point guard: within-limit prompt proceeds to routing ──
lastError = null;
const okEnv = makeSmartEnv("fix the login bug");
const okFns = Object.assign({}, okEnv.fns, {
  _isAgentRunning: () => false,
  normalizeRouteMode: () => "auto",
  _promptSuggestsAgent: () => false,
  runOnly: () => { okFns.runOnlyCalled = true; },
});
const okSmart = new Function(
  "el", "_promptWithinLimit", "addTL", "document",
  "_designSendMessage", "_sendMidTaskMessage", "runOnly", "_launchAgent",
  "normalizeRouteMode", "_isAgentRunning", "_promptSuggestsAgent",
  `${smartText}; return _smartRunAfterSlash;`
)(okEnv.el, okFns._promptWithinLimit, okFns.addTL, okFns.document,
  okFns._designSendMessage, okFns._sendMidTaskMessage, okFns.runOnly, okFns._launchAgent,
  okFns.normalizeRouteMode, okFns._isAgentRunning, okFns._promptSuggestsAgent);
okSmart();
check("smartRun passes within-limit prompt through to routing", okFns.runOnlyCalled === true, "runOnly not reached");
check("smartRun does not error for within-limit prompt", lastError === null, lastError || "unexpected error");

console.log(`P15-7 prompt size cap gate: ${passed} checks PASS`);
