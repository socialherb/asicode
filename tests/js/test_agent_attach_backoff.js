#!/usr/bin/env node
/**
 * Regression harness — P8-1: stale attach backoff identity guard.
 *
 * Bug: the SSE attach stream's onerror schedules a retry with backoff
 * (setTimeout(() => _agentAttachStream(sessionId, lastEventId), delay)).
 * The retry has NO identity guard: if the user starts a NEW run while the
 * backoff is pending, the fired retry closes the NEW stream (via the guard
 * inside _agentAttachStream) and attaches to the DEAD session — freezing
 * the new run's UI. Same bug class as the P3-2 performanceStream race and
 * the P6-1 stale approval countdown.
 *
 * Fix under test: the backoff callback reconnects ONLY when no newer stream
 * has been created in the meantime (i.e. `_agentSSESource` is still null —
 * onerror itself nulls it, so a non-null value proves a newer stream).
 *
 * The REAL onerror block is sliced out of agent-panel.js and executed
 * against a stub environment (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_attach_backoff.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

// ── Slice the REAL source.onerror block out of agent-panel.js (brace-balanced) ──
const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");
const marker = "source.onerror = () => {";
// Three onerror handlers exist in agent-panel.js (L3130 agentRunStream,
// L3825 _agentAttachStream, L4390 image loader). Target the ATTACH one by
// anchoring on its unique preceding comment:
const anchorComment = "// Connection-level error (network down, proxy closed, etc.)";
const anchor = src.indexOf(anchorComment);
assert.ok(anchor >= 0, "attach onerror comment not found in agent-panel.js");
const fnStart = src.indexOf(marker, anchor);
assert.ok(fnStart >= 0, "source.onerror block not found in agent-panel.js");
let depth = 0;
let i = src.indexOf("{", fnStart);
for (; i < src.length; i++) {
  if (src[i] === "{") depth++;
  else if (src[i] === "}") {
    depth--;
    if (depth === 0) break;
  }
}
assert.ok(i < src.length, "unbalanced braces while extracting source.onerror");
const onerrorText = src.slice(fnStart, i + 1);

// ── Stub environment (globalThis: the new Function below runs in GLOBAL scope) ──
const timers = []; // {delay, cb} — fake scheduler
const attachCalls = [];

globalThis.setTimeout = (cb, delay) => {
  timers.push({ delay, cb });
  return timers.length;
};

// Free variables referenced by the onerror block (match agent-panel.js):
globalThis.source = { close() {} };       // the errored EventSource instance
globalThis._agentSSESource = null;        // current live stream (null after onerror)
globalThis._attachRetryAttempt = 0;
globalThis._ATTACH_BACKOFF_MS = [1000, 3000, 8000];
globalThis.sessionId = "sess-1";
globalThis.lastEventId = "evt-9";
globalThis._agentAttachStream = (sid, lid) => attachCalls.push({ sid, lid });

// Compile the real block in global scope so its free variables resolve
// against the stubs above.
const runOnerror = new Function(
  `${onerrorText}\nreturn source.onerror;`
)();

let passed = 0;
function check(name, fn) {
  timers.length = 0;
  attachCalls.length = 0;
  globalThis._agentSSESource = null;
  globalThis._attachRetryAttempt = 0;
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── Scenario 1: normal backoff — no newer stream → reconnects ──
check("backoff reconnects when no newer stream started", () => {
  globalThis._agentSSESource = globalThis.source; // errored stream is current
  runOnerror(); // onerror nulls _agentSSESource internally
  assert.strictEqual(globalThis._agentSSESource, null, "onerror must null the stream");
  assert.strictEqual(timers.length, 1, "one backoff scheduled");
  timers[0].cb(); // fire the backoff
  assert.strictEqual(attachCalls.length, 1, "must reconnect once");
  assert.deepStrictEqual(attachCalls[0], { sid: "sess-1", lid: "evt-9" });
});

// ── Scenario 2: THE BUG — newer stream started while backoff pending → must NOT reconnect ──
check("stale backoff does not hijack a newer stream", () => {
  globalThis._agentSSESource = globalThis.source;
  runOnerror();
  assert.strictEqual(timers.length, 1, "one backoff scheduled");
  // User starts a NEW run while the backoff is pending:
  globalThis._agentSSESource = { close() {} }; // brand-new stream B
  timers[0].cb(); // stale backoff fires
  assert.strictEqual(attachCalls.length, 0,
    "stale backoff must NOT attach to the dead session (it would close stream B)");
  assert.ok(globalThis._agentSSESource, "the newer stream must remain untouched");
});

// ── Scenario 3: backoff budget exhausted → resets, no timer ──
check("exhausted backoff resets attempt counter without scheduling", () => {
  globalThis._agentSSESource = globalThis.source;
  globalThis._attachRetryAttempt = globalThis._ATTACH_BACKOFF_MS.length;
  runOnerror();
  assert.strictEqual(globalThis._attachRetryAttempt, 0, "attempt counter reset");
  assert.strictEqual(timers.length, 0, "no backoff scheduled");
});

// ── Scenario 4: backoff progression across successive errors ──
check("successive errors use increasing backoff delays", () => {
  globalThis._agentSSESource = globalThis.source;
  runOnerror();
  runOnerror();
  assert.strictEqual(timers.length, 2);
  assert.strictEqual(timers[0].delay, 1000);
  assert.strictEqual(timers[1].delay, 3000);
  // Firing the FIRST (older) backoff while the stream is null again:
  // reconnect is legitimate (no newer stream exists).
  timers[0].cb();
  assert.strictEqual(attachCalls.length, 1, "reconnect after null stream is valid");
});

console.log(`\n${passed}/4 checks passed (P8-1 attach backoff identity guard)`);
process.exit(passed === 4 ? 0 : 1);
