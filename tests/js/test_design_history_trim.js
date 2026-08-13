#!/usr/bin/env node
/**
 * Regression harness — P18-2: _designChat.history must be trimmed to a bounded
 * tail (_DESIGN_HISTORY_MAX_TURNS = 60) at every push site + restore, so long
 * design-chat conversations cannot grow the browser-side array unboundedly
 * (server already persists + compresses its own copy; the local array is only
 * consumed by the legacy seed path via slice(-21, -1)).
 *
 * The REAL _designTrimHistory function text is sliced out of design-chat.js
 * (brace-balanced) and executed against a stub _designChat — no framework.
 * Run: node tests/js/test_design_history_trim.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "design-chat.js"), "utf8");

function sliceFunction(srcText, anchor) {
  const start = srcText.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found`);
  const open = srcText.indexOf("{", start);
  assert.ok(open >= 0, `no body for ${anchor}`);
  let depth = 0;
  let i = open;
  for (; i < srcText.length; i++) {
    if (srcText[i] === "{") depth++;
    else if (srcText[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < srcText.length, `unbalanced braces while extracting ${anchor}`);
  return srcText.slice(start, i + 1);
}

const trimFnText = sliceFunction(src, "function _designTrimHistory()");
const trimFn = new Function(
  "_designChat",
  "_DESIGN_HISTORY_MAX_TURNS",
  trimFnText + "\nreturn _designTrimHistory;"
);

let checks = 0;
function check(name, cond) {
  checks++;
  assert.ok(cond, `FAIL: ${name}`);
}

// ── behavioral checks: trim keeps the LAST 60 turns ──
function freshChat(nTurns) {
  const history = [];
  for (let i = 0; i < nTurns; i++) history.push({ role: "user", content: `turn-${i}` });
  return { history };
}

{
  const _designChat = freshChat(61);
  const trim = trimFn(_designChat, 60);
  trim();
  check("61 turns -> 60 kept", _designChat.history.length === 60);
  check("oldest dropped (turn-0 gone)", _designChat.history[0].content === "turn-1");
  check("newest kept (turn-60 present)", _designChat.history[59].content === "turn-60");
}

{
  const _designChat = freshChat(120);
  const trim = trimFn(_designChat, 60);
  trim();
  check("120 turns -> 60 kept", _designChat.history.length === 60);
  check("oldest of tail is turn-60", _designChat.history[0].content === "turn-60");
  check("newest is turn-119", _designChat.history[59].content === "turn-119");
}

{
  const _designChat = freshChat(60);
  const trim = trimFn(_designChat, 60);
  trim();
  check("exactly 60 turns untouched", _designChat.history.length === 60 && _designChat.history[0].content === "turn-0");
}

{
  const _designChat = freshChat(10);
  const trim = trimFn(_designChat, 60);
  trim();
  check("10 turns untouched", _designChat.history.length === 10);
}

{
  // No-op on empty history (restore of a fresh session)
  const _designChat = freshChat(0);
  const trim = trimFn(_designChat, 60);
  trim();
  check("empty history untouched", _designChat.history.length === 0);
}

// ── source gates: every push site + restore trims ──
{
  // constant exists with the documented bound
  const constMatch = src.match(/const _DESIGN_HISTORY_MAX_TURNS\s*=\s*(\d+)/);
  check("constant defined", !!constMatch);
  check("constant is 60", constMatch && constMatch[1] === "60");
  check("comment references P18-2", src.includes("P18-2 — the local history array"));

  // helper is defined once and named exactly
  const defCount = (src.match(/function _designTrimHistory\(\)/g) || []).length;
  check("helper defined exactly once", defCount === 1);

  // 4 call sites: restore loop, user push, streamed-ai push, typing-ai push
  const callSites = (src.match(/_designTrimHistory\(\);/g) || []).length;
  check("4 trim call sites (restore + 3 pushes)", callSites === 4);

  // every call site sits right after a history.push
  let idx = -1;
  let allAdjacent = true;
  for (let n = 0; n < callSites; n++) {
    idx = src.indexOf("_designTrimHistory();", idx + 1);
    const before = src.slice(Math.max(0, idx - 200), idx);
    if (!/history\.push/.test(before) && !/for \(const turn of data\.turns\)/.test(src.slice(Math.max(0, idx - 400), idx))) {
      // restore call site follows the for-loop's closing brace, not a push line directly
      const block = src.slice(Math.max(0, idx - 400), idx);
      if (!/history\.push/.test(block)) allAdjacent = false;
    }
  }
  check("call sites adjacent to history pushes (within 400 chars)", allAdjacent);

  // seed path still slices the recent ~20 turns — trim bound must stay above that
  check("seed slice still present", src.includes("_designChat.history.slice(-21, -1)"));
}

console.log(`test_design_history_trim.js: ${checks} checks PASS`);
