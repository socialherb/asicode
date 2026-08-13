#!/usr/bin/env node
/**
 * Regression harness — P6-1: stale approval countdown interval.
 *
 * Bug: _agentStartApprovalCountdown captures a deadline and starts a 1s
 * setInterval. When the chat is cleared (timeline.innerHTML = ""), the
 * approval card is removed from the DOM but the interval keeps running.
 * When its deadline fires, _agentUpdateLastApprovalCard("timeout") resolves
 * WHATEVER approval card is currently last — identity loss, the same bug
 * class as the P3-2 performanceStream race.
 *
 * Fix under test: the interval callback dies silently on the first tick
 * after its card has been disconnected from the DOM.
 *
 * The REAL function text is sliced out of agent-panel.js and executed
 * against a stub DOM + fake timers (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_approval_countdown.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

// ── Slice the REAL function out of agent-panel.js (brace-balanced) ──
const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");
const fnStart = src.indexOf("function _agentStartApprovalCountdown(card) {");
assert.ok(fnStart >= 0, "function _agentStartApprovalCountdown not found in agent-panel.js");
let depth = 0;
let i = src.indexOf("{", fnStart);
for (; i < src.length; i++) {
  if (src[i] === "{") depth++;
  else if (src[i] === "}") {
    depth--;
    if (depth === 0) break;
  }
}
assert.ok(i < src.length, "unbalanced braces while extracting _agentStartApprovalCountdown");
const fnText = src.slice(fnStart, i + 1);

// ── Stub environment: fake timers, fake clock, spy for the resolver ──
const intervals = new Map(); // timerId -> callback (insertion order)
let nextTimerId = 0;
let now = 1_700_000_000_000;

globalThis.setInterval = (cb) => {
  const id = ++nextTimerId;
  intervals.set(id, cb);
  return id;
};
globalThis.clearInterval = (id) => {
  intervals.delete(id);
};
globalThis.Date = class extends Date {
  static now() { return now; }
};

const resolveCalls = [];
globalThis._agentResolveApprovalCard = (state, requestId, cardEl) => {
  resolveCalls.push({ state, requestId, cardEl });
};

// Compile the real function in global scope so its free variables
// (setInterval/clearInterval/Date/_agentUpdateLastApprovalCard) resolve
// against the stubs above.
const startCountdown = new Function(fnText + "\nreturn _agentStartApprovalCountdown;")();

function makeCard({ connected = true, deadlineMs = null, classes = [] } = {}) {
  const countdown = { textContent: "" };
  const card = {
    isConnected: connected,
    dataset: { deadline: String(deadlineMs ?? Date.now() + 60_000) },
    _classes: new Set(classes),
    _countdown: countdown,
    classList: {
      contains: (c) => card._classes.has(c),
    },
    querySelector: (sel) => (sel === ".agent-approval-countdown" ? countdown : null),
  };
  return card;
}

function tickAll() {
  for (const cb of [...intervals.values()]) cb();
}

let passed = 0;
function check(name, fn) {
  intervals.clear();
  resolveCalls.length = 0;
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── Scenario 1: normal countdown ticks down, resolves exactly once ──
check("normal countdown ticks down and resolves once at zero", () => {
  const card = makeCard({ deadlineMs: now + 2_500 });
  startCountdown(card);
  assert.equal(intervals.size, 1);
  tickAll();
  assert.equal(card._countdown.textContent, "3s");
  now += 3_000;
  tickAll();
  assert.equal(card._countdown.textContent, "0s");
  assert.equal(resolveCalls.length, 1);
  assert.equal(resolveCalls[0].state, "timeout");
  assert.equal(intervals.size, 0, "interval cleared after timeout");
});

// ── Scenario 2: already-resolved card starts no interval ──
check("resolved card does not start an interval", () => {
  const card = makeCard({ classes: ["agent-approval-approved"] });
  startCountdown(card);
  assert.equal(intervals.size, 0);
});

// ── Scenario 3: card resolved mid-flight stops the interval ──
check("resolved mid-flight stops the interval without timeout", () => {
  const card = makeCard({ deadlineMs: now + 60_000 });
  startCountdown(card);
  card._classes.add("agent-approval-rejected");
  tickAll();
  assert.equal(intervals.size, 0);
  assert.equal(resolveCalls.length, 0);
});

// ── Scenario 4 (REGRESSION): stale card removed from DOM dies silently ──
check("stale card (removed from DOM) dies on first tick without resolving", () => {
  const card = makeCard({ connected: false, deadlineMs: now + 1_000 });
  startCountdown(card);
  assert.equal(intervals.size, 1, "interval starts (isConnected is not checked at start time)");
  tickAll();
  assert.equal(intervals.size, 0, "stale interval must clear itself");
  assert.equal(resolveCalls.length, 0, "removed card must never resolve an approval card");
});

// ── Scenario 5 (REGRESSION, full reproduction): chat clear + new card ──
// Card A is appended and starts its countdown; the chat is cleared (A is
// disconnected); card B is appended and starts its own countdown. When A's
// deadline arrives, A's stale interval must NOT resolve B — the bug did.
check("chat-clear identity: stale card A never resolves card B", () => {
  const cardA = makeCard({ deadlineMs: now + 2_000 });
  startCountdown(cardA);       // interval A
  assert.equal(intervals.size, 1);
  cardA.isConnected = false;   // chat clear removes A from the DOM
  const cardB = makeCard({ deadlineMs: now + 60_000 });
  startCountdown(cardB);       // interval B
  assert.equal(intervals.size, 2);
  now += 3_000;                // A's deadline passes
  tickAll();
  assert.equal(resolveCalls.length, 0, "A's deadline must not resolve B");
  assert.equal(intervals.size, 1, "only B's interval remains");
  now += 60_000;               // B's deadline arrives
  tickAll();
  assert.equal(resolveCalls.length, 1, "B resolves exactly once");
  assert.equal(resolveCalls[0].state, "timeout");
  assert.equal(intervals.size, 0);
});

// ── Scenario 6: card without countdown element is a no-op ──
check("card without countdown element is a no-op", () => {
  const card = {
    isConnected: true,
    dataset: { deadline: "0" },
    classList: { contains: () => false },
    querySelector: () => null,
  };
  startCountdown(card);
  assert.equal(intervals.size, 0);
});

console.log(`\n${passed}/6 harness checks PASSED`);
