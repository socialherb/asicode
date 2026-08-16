#!/usr/bin/env node
/**
 * Regression harness — P12-2/P12-3: the `queued` worker-saturation chip and
 * the `events_dropped` loss banner in the agent panel.
 *
 * P12-2: the server's worker-saturation hint was dead on arrival twice over:
 * the qsize() probe read 0 for the first waiting run (off-by-one), and even
 * when emitted the `queued` event had NO client listener — the EventSource
 * delivered it to nobody and the user stared at keepalive silence. Now the
 * handlers map (the ONLY dispatch path; addEventListener is overridden to a
 * no-op after EventSequencer registration) renders a "워커 대기 중 (앞에 N건)"
 * chip removed when the run actually starts (session_start / agent_working).
 *
 * P12-3: `events_dropped` and done.queue_stats.dropped_events were emitted by
 * the server but never read — the slow-consumer loss banner is now shown and
 * updated in place.
 *
 * The REAL handler/helper text is sliced out of agent-panel.js (brace-
 * balanced) and executed against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_worker_queue_hint.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

// ── Slice helpers: brace-balanced from `function NAME(` to its closing } ──
function sliceFunction(anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found in agent-panel.js`);
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

// ── Slice map entries: `    name: (data) => { ... }` to the closing } ──
function sliceMapEntry(anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `map entry ${anchor} not found in agent-panel.js`);
  const open = src.indexOf("{", start);
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

const helpers = [
  sliceFunction("function _agentRemoveQueuedChip("),
  sliceFunction("function _agentRenderQueued("),
  sliceFunction("function _agentShowDroppedBanner("),
].join("\n");

const queuedEntry = sliceMapEntry("    queued: (data) => {");
const droppedEntry = sliceMapEntry("    events_dropped: (data) => {");
const sessionStartEntry = sliceMapEntry("    session_start: (data) => {");
const agentWorkingEntry = sliceMapEntry("    agent_working: (data) => {");
const doneEntry = sliceMapEntry("    done: (data) => {");

// Source gates: the removal/banner wiring must live in the real handlers.
assert.ok(sessionStartEntry.includes("_agentRemoveQueuedChip"), "session_start must drop the queued chip");
assert.ok(agentWorkingEntry.includes("_agentRemoveQueuedChip"), "agent_working must drop the queued chip");
assert.ok(doneEntry.includes("_agentRemoveQueuedChip"), "done must drop the queued chip");
assert.ok(doneEntry.includes("_agentShowDroppedBanner"), "done must surface queue_stats.dropped_events");
assert.ok(doneEntry.includes("queue_stats"), "done must read queue_stats from the payload");

// ── Stub environment ─────────────────────────────────────────────────────
function makeEl(className) {
  const el = {
    className: className || "",
    textContent: "",
    title: "",
    children: [],
    _parent: null,
    appendChild(c) { c._parent = this; this.children.push(c); },
    remove() {
      if (this._parent) {
        const i = this._parent.children.indexOf(this);
        if (i >= 0) this._parent.children.splice(i, 1);
      }
    },
    querySelector(sel) {
      return this.children.find((c) => (c.className || "").split(" ").includes(sel.slice(1))) || null;
    },
  };
  return el;
}

const timeline = makeEl();
globalThis.document = { createElement: () => makeEl() };
globalThis._agentGetTimeline = () => timeline;
let scrolls = 0;
globalThis._agentScrollBottom = () => { scrolls++; };
// Stub of the sibling working-indicator renderer (not under test here).
globalThis._agentRenderWorking = (tl) => {
  const ind = makeEl("agent-processing-indicator");
  ind.textContent = "⏳ 처리 중";
  tl.appendChild(ind);
};
// Stub of the OUTPUT-tab raw-log writer (not under test here) — the
// session_start handler under test also records the session id into the
// raw output log (round-29 audit #1 OUTPUT tab wiring).
globalThis._agentOutputLogEvent = () => {};
globalThis.sessionStorage = {
  _s: {},
  setItem(k, v) { this._s[k] = v; },
  removeItem(k) { delete this._s[k]; },
  getItem(k) { return k in this._s ? this._s[k] : null; },
};
globalThis.source = { closed: false, close() { this.closed = true; } };
globalThis.ctx = { cancelBtn: { disabled: false } };

const factory = new Function(
  `${helpers}\nreturn ({ ${queuedEntry}, ${droppedEntry}, ${sessionStartEntry}, ${agentWorkingEntry}, ${doneEntry} });`
);
const handlers = factory();
const q = (sel) => timeline.querySelector(sel);
const qAll = (sel) => timeline.children.filter((c) => (c.className || "").split(" ").includes(sel.slice(1)));

// ── queued chip ──────────────────────────────────────────────────────────
handlers.queued({ pending_ahead: 3 });
const chip = q(".agent-queued-chip");
assert.ok(chip, "queued must render a waiting chip");
assert.ok(chip.textContent.includes("3"), `chip must show the pending count, got: ${chip.textContent}`);
assert.ok(scrolls >= 1, "queued must scroll the timeline");

handlers.queued({ pending_ahead: 5 });
assert.strictEqual(qAll(".agent-queued-chip").length, 1, "a second queued event must update, not duplicate, the chip");
assert.ok(q(".agent-queued-chip").textContent.includes("5"), "chip count must update in place");

// P13-3: `blocked_on_gate` (runs parked on approval/ask_user gates) is
// appended to the chip so the user knows answering cards may free a worker.
handlers.queued({ pending_ahead: 2, blocked_on_gate: 2 });
assert.ok(q(".agent-queued-chip").textContent.includes("승인/질문 대기 2건"),
  `chip must show the gate-blocked count, got: ${q(".agent-queued-chip").textContent}`);
// Legacy payload without the field must not fabricate a gate note.
handlers.queued({ pending_ahead: 1 });
assert.ok(!q(".agent-queued-chip").textContent.includes("승인/질문"),
  "chip without blocked_on_gate must render no gate note");

// agent_working (first real work) removes the chip and shows the indicator.
handlers.agent_working({ reason: "context_compressed" });
assert.strictEqual(q(".agent-queued-chip"), null, "agent_working must remove the queued chip");
assert.ok(q(".agent-processing-indicator"), "agent_working must still render the indicator");

// session_start also removes a stale chip (reconnect edge).
handlers.queued({ pending_ahead: 1 });
handlers.session_start({ session_id: "s1" });
assert.strictEqual(q(".agent-queued-chip"), null, "session_start must remove the queued chip");
assert.strictEqual(globalThis.sessionStorage._s["asr_agent_session_id"], "s1");

// ── events_dropped banner ────────────────────────────────────────────────
handlers.events_dropped({ count: 3 });
const banner = q(".agent-events-dropped");
assert.ok(banner, "events_dropped must render a loss banner");
assert.ok(banner.textContent.includes("3"), `banner must show the count, got: ${banner.textContent}`);

handlers.events_dropped({ count: 7 });
assert.strictEqual(qAll(".agent-events-dropped").length, 1, "a higher count must update, not duplicate, the banner");
assert.ok(q(".agent-events-dropped").textContent.includes("7"), "banner count must update in place");

handlers.events_dropped({ count: 0 });
assert.strictEqual(qAll(".agent-events-dropped").length, 1, "count 0 must be a no-op");

// ── done: final dropped count + teardown ─────────────────────────────────
handlers.queued({ pending_ahead: 2 });
globalThis._agentSSESource = { marker: 1 };
handlers.done({ queue_stats: { dropped_events: 5 } });
assert.strictEqual(q(".agent-queued-chip"), null, "done must remove the queued chip");
assert.ok(q(".agent-events-dropped").textContent.includes("5"), "done must surface the final dropped count");
assert.ok(globalThis.source.closed, "done must close the EventSource");
assert.strictEqual(globalThis._agentSSESource, null, "done must clear _agentSSESource");
assert.strictEqual(globalThis.ctx.cancelBtn.disabled, true, "done must disable the cancel button");
assert.strictEqual(globalThis.sessionStorage._s["asr_agent_session_id"], undefined, "done must clear session storage");

// done with zero drops: no banner created by the final payload alone.
q(".agent-events-dropped").remove();
globalThis._agentSSESource = { marker: 2 };
handlers.done({ queue_stats: { dropped_events: 0 } });
assert.strictEqual(q(".agent-events-dropped"), null, "zero dropped events must not fabricate a banner");

console.log("PASS: queued chip + events_dropped banner (P12-2/P12-3)");
