#!/usr/bin/env node
/**
 * Regression harness — P15-10: the agent-panel checkpoint card's _sendAnswer()
 * bounds the free-text answer client-side, mirroring the server's
 * /agent/user_response answer cap (256 KiB, agent_control.py P15-8).
 *
 * The free-text .agent-cp-input is the unbounded path (yes/no and
 * multiple-choice buttons are fixed short values); before this fix a multi-MB
 * paste round-tripped to the server before the 413.
 *
 * Fix under test: _sendAnswer() rejects answer over _AGENT_ANSWER_MAX_BYTES
 * with an addTL error BEFORE any card mutation (buttons stay enabled).
 *
 * The REAL function text is sliced out of agent-panel.js (brace-balanced) and
 * executed against stub globals (no test framework, no browser).
 * Run: node tests/js/test_agent_answer_cap.js
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

const fnText = sliceFunction(src, "async function _sendAnswer(answer)");

// ── stub environment ──
let lastError = null;
let cardMutations = 0;
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
const _AGENT_ANSWER_MAX_BYTES = 256 * 1024;  // must mirror the source constant

function makeCard({ bomb = false } = {}) {
  // Any card mutation after the gate must be visible; with bomb=true the
  // querySelectorAll call itself throws so the harness proves the gate
  // returned before it.
  return {
    querySelectorAll() {
      if (bomb) throw new Error("card touched for oversize answer");
      cardMutations++;
      return [];
    },
    querySelector: () => null,
    classList: { add() {}, remove() {} },
    dataset: {},
  };
}

function loadSend(card, overrides) {
  return new Function(
    "card", "addTL", "_AGENT_ANSWER_MAX_BYTES", "TextEncoder",
    "clearInterval", "_fetchTimeoutSignal", "fetch",
    `${fnText}; return _sendAnswer;`
  )(card, overrides.addTL || addTL, overrides.maxBytes || _AGENT_ANSWER_MAX_BYTES,
    TextEncoder, overrides.clearInterval || (() => {}),
    overrides._fetchTimeoutSignal || (() => null),
    overrides.fetch || (() => Promise.resolve({ ok: true })));
}

(async () => {
  // ── oversize: reject before any card mutation ──
  lastError = null; cardMutations = 0;
  const big = "x".repeat(_AGENT_ANSWER_MAX_BYTES + 1);
  await loadSend(makeCard({ bomb: true }), {}).call(null, big);
  check("oversize answer surfaces an addTL error naming the limit",
    typeof lastError === "string" && /256 KiB/.test(lastError), lastError || "no error");
  check("oversize answer never touches the card (buttons stay enabled)",
    cardMutations === 0, `cardMutations=${cardMutations}`);

  // ── boundary: exactly the cap proceeds to card mutation ──
  lastError = null; cardMutations = 0;
  const exact = "y".repeat(_AGENT_ANSWER_MAX_BYTES);
  await loadSend(makeCard({}), {}).call(null, exact);
  check("exact-cap answer passes the guard (no client error)", lastError === null, lastError || "unexpected error");
  check("exact-cap answer proceeds to card mutation", cardMutations >= 1, `cardMutations=${cardMutations}`);

  // ── small fixed button values (yes/no / options) pass untouched ──
  lastError = null; cardMutations = 0;
  await loadSend(makeCard({}), {}).call(null, "yes");
  check("short answer proceeds", cardMutations >= 1, `cardMutations=${cardMutations}`);
  check("short answer did not error", lastError === null, lastError || "unexpected error");

  // ── source gate: constant pinned + gate sits before the disable loop ──
  const srcHasConst = /const _AGENT_ANSWER_MAX_BYTES\s*=\s*256 \* 1024;/.test(src);
  check("source gate: _AGENT_ANSWER_MAX_BYTES constant exists in agent-panel.js", srcHasConst);
  const gateInsideFn = /async function _sendAnswer\(answer\) \{[\s\S]*?_AGENT_ANSWER_MAX_BYTES[\s\S]*?Disable all buttons/.test(src);
  check("source gate: the size check sits before the disable-all-buttons block", gateInsideFn);

  console.log(`P15-10 agent answer cap gate: ${passed} checks PASS`);
})().catch((e) => {
  console.error(`P15-10 agent answer cap gate FAILED: ${e.message}`);
  process.exit(1);
});
