#!/usr/bin/env node
/**
 * Regression harness — round 32-2: seven emitted-but-never-shown events
 * (batch_start / batch_complete / content / goal_reminder /
 * orchestrator_error / orchestrator_warning / server_retry) now have LIVE
 * handlers-map entries. Before this round they were silently dropped:
 *
 *   - "content" existed ONLY as a post-wire addEventListener twin — dead
 *     after the EventSequencer override — so streamed answers never
 *     rendered inline (and the dead twin would have spammed one summary
 *     message per chunk: token_callback delivers per-chunk DELTAS).
 *   - server 5xx retries were invisible while rate_limit_retry and
 *     connection_retry WERE shown — an inconsistent trio from the same
 *     _handle_retry_error helper.
 *   - orchestrator batch progress and diagnostics (dependency cycles,
 *     deadlock, file conflicts) never reached the user.
 *
 * Gates:
 *   R1 presence — the seven handlers live in the LIVE handlers-map region
 *      ("// Define event handlers" … "// Create EventSequencer").
 *   R2 dead twins — no post-wire addEventListener("content") twin remains;
 *      the live complete handler drops the streaming bubble.
 *   R3 payload contract — handlers read the REAL backend payload shapes
 *      (orchestrator.py batch/diagnostics, agent_loop.py _handle_retry_error,
 *      agent_turn_pipeline.py goal_reminder, tool_registry.py content).
 *   R4 behavior — content chunks accumulate into ONE bubble (O(1) text-node
 *      appends, 40KB cap, sub-agent agent_id streams filtered out); the
 *      other six render into the timeline; complete removes the bubble.
 *
 * Run: node tests/js/test_agent_r1_display.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

const NAMES = [
  "content", "server_retry", "batch_start", "batch_complete",
  "orchestrator_warning", "orchestrator_error", "goal_reminder",
];

// ── Slice helpers (brace-balanced) ────────────────────────────────────────
function sliceMapEntry(anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `map entry ${anchor} not found in agent-panel.js`);
  const open = src.indexOf("{", start);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") { depth--; if (depth === 0) break; }
  }
  assert.ok(i < src.length, `unbalanced braces while extracting ${anchor}`);
  return src.slice(start, i + 1);
}

// ── R1: the seven handlers must live in the LIVE map region ───────────────
const regionStart = src.indexOf("// Define event handlers");
const regionEnd = src.indexOf("// Create EventSequencer");
assert.ok(regionStart >= 0 && regionEnd > regionStart, "handlers-map region markers must exist");
const entries = {};
for (const name of NAMES) {
  const anchor = `    ${name}: (data) => {`;
  const i = src.indexOf(anchor);
  assert.ok(i >= 0, `${name} handler must exist in agent-panel.js (round 32-2)`);
  assert.ok(i > regionStart && i < regionEnd, `${name} must live in the live handlers map (not a dead listener)`);
  entries[name] = sliceMapEntry(anchor);
}

// ── R2: dead twins / duplicate render paths ───────────────────────────────
assert.ok(!/addEventListener\(\s*"content"/.test(src),
  "no post-wire addEventListener(\"content\") twin — it is dead after the EventSequencer override");
assert.ok(!entries.content.includes("_agentSetSummary"),
  "live content handler must NOT call _agentSetSummary per chunk — token_callback delivers per-chunk DELTAS; a per-chunk summary append spams one message per token");
const completeEntry = sliceMapEntry("    complete: (data) => {");
assert.ok(completeEntry.includes("_contentStreamEl.remove()") && completeEntry.includes("_contentStreamEl = null"),
  "complete must drop the streaming content bubble — final_message render is authoritative (no double display)");
assert.ok(src.includes("let _contentStreamEl = null;"),
  "streaming bubble state must be closure-scoped per wiring (fresh bubble per run)");

// ── R3: payload contracts (real backend shapes) ───────────────────────────
// tool_registry._token_cb → {"text": chunk}; sub-agents tagged agent_id.
assert.ok(entries.content.includes("data.text") && entries.content.includes("agent_id"),
  "content reads {text} and filters sub-agent streams (agent_id)");
// agent_loop._handle_retry_error → {attempt, max_retries, delay, message}
for (const f of ["attempt", "max_retries", "delay"]) {
  assert.ok(entries.server_retry.includes(`data.${f}`), `server_retry reads data.${f} (_handle_retry_error payload)`);
}
// orchestrator.batch_start → {batch_id, task_count, task_ids, priority}
assert.ok(entries.batch_start.includes("task_count") && entries.batch_start.includes("priority"),
  "batch_start reads task_count/priority (orchestrator payload)");
// orchestrator.batch_complete → {batch_id, success_count, total_count}
assert.ok(entries.batch_complete.includes("success_count") && entries.batch_complete.includes("total_count"),
  "batch_complete reads success_count/total_count (orchestrator payload)");
// orchestrator._cb warnings/errors → {type, message, ...}
for (const [n, e] of [["orchestrator_warning", entries.orchestrator_warning], ["orchestrator_error", entries.orchestrator_error]]) {
  assert.ok(e.includes("data.message") && e.includes("data.type"), `${n} reads message/type (orchestrator diagnostics payload)`);
}
// agent_turn_pipeline → {turn, reads_without_edit, reminder_count}
for (const f of ["reads_without_edit", "reminder_count", "turn"]) {
  assert.ok(entries.goal_reminder.includes(`data.${f}`), `goal_reminder reads data.${f} (pipeline payload)`);
}

// ── Stub DOM ──────────────────────────────────────────────────────────────
function makeEl(tag) {
  const el = {
    tag, className: "", textContent: "", title: "", isConnected: true, children: [],
    appendChild(c) {
      el.children.push(c);
      if (c && typeof c.text === "string") el.textContent += c.text;
      else if (c && typeof c.textContent === "string") el.textContent += c.textContent;
      return c;
    },
    remove() { el.isConnected = false; },
    querySelector() { return null; },
    classList: { add() {}, remove() {}, toggle() {} },
  };
  return el;
}
const timeline = {
  id: "agent-timeline", children: [],
  appendChild(c) { timeline.children.push(c); return c; },
  querySelector() { return null; },
};
globalThis.document = { createElement: makeEl, createTextNode: (t) => ({ text: String(t) }) };
globalThis._agentGetTimeline = () => timeline;
globalThis._agentScrollBottom = () => {};
globalThis._contentStreamEl = null;
// complete-handler deps
const statusCalls = [];
globalThis._agentSetStatus = (...a) => statusCalls.push(a);
globalThis._agentSetSummary = () => {};
globalThis._agentOutputLogEvent = () => {};
globalThis._pipelineDeveloperStarted = false;
globalThis._pipelineSetStage = () => {};
globalThis._pipelineDone = () => {};
globalThis._setUIMode = () => {};
globalThis._agentShowContinueBar = () => {};
globalThis._tlSave = () => {};

const factory = new Function("ctx", `return ({ ${NAMES.map((n) => entries[n]).join(",")}, ${completeEntry} });`);
const handlers = factory({ multiAgent: false, cancelBtn: null, onComplete: null });

// ── R4a: content — per-chunk deltas accumulate into ONE bubble ────────────
const bubblesBefore = timeline.children.length;
handlers.content({ text: "Hello" });
handlers.content({ text: " world" });
assert.strictEqual(timeline.children.length, bubblesBefore + 1, "two chunks → ONE bubble (not one per chunk)");
const bubble = timeline.children[timeline.children.length - 1];
assert.strictEqual(bubble.textContent, "Hello world", "chunks accumulate in order");
assert.ok(bubble.className.includes("agent-chat-msg--text-reply"), "streams as a text_reply bubble");
assert.ok(globalThis._contentStreamEl === bubble, "streaming state tracks the live bubble");

// Sub-agent token streams stay out of the main timeline (agent_id tagged).
handlers.content({ text: "sub", agent_id: "t3" });
assert.strictEqual(bubble.textContent, "Hello world", "sub-agent chunk filtered out of main bubble");
assert.strictEqual(timeline.children.length, bubblesBefore + 1, "sub-agent chunk creates no bubble");

// Degenerate payloads are no-ops, not crashes.
handlers.content({});
handlers.content({ text: "" });
handlers.content(null);
assert.strictEqual(timeline.children.length, bubblesBefore + 1, "empty/null payloads are no-ops");

// 40KB cap mirrors design-tool-live.
globalThis._contentStreamEl = null; // force a fresh bubble for the cap case
handlers.content({ text: "x".repeat(41000) });
assert.ok(globalThis._contentStreamEl.textContent.startsWith("...(older output truncated)"),
  "buffer beyond 40KB is trimmed with a truncation marker");
assert.ok(globalThis._contentStreamEl.textContent.length <= 28 + 40000, "capped at ~40KB");

// A detached bubble (timeline cleared between runs) is re-created, not resurrected.
globalThis._contentStreamEl.isConnected = false;
handlers.content({ text: "fresh" });
assert.notStrictEqual(globalThis._contentStreamEl, timeline.children[timeline.children.length - 2],
  "detached bubble is replaced by a new one");

// ── R4b: complete removes the streaming bubble ────────────────────────────
globalThis._contentStreamEl = makeEl("div");
const streamed = globalThis._contentStreamEl;
handlers.complete({ status: "success", agent_id: "main", final_message: "The answer" });
assert.strictEqual(streamed.isConnected, false, "complete removes the streamed bubble");
assert.strictEqual(globalThis._contentStreamEl, null, "streaming state reset on complete");

// ── R4c: the six status events render into the timeline ───────────────────
const t0 = timeline.children.length;
handlers.server_retry({ attempt: 2, max_retries: 5, delay: 3 });
handlers.batch_start({ batch_id: "b1", task_count: 3, task_ids: ["a", "b", "c"], priority: 1 });
handlers.batch_complete({ batch_id: "b1", success_count: 2, total_count: 3 });
handlers.orchestrator_warning({ type: "dependency_cycle", cycles: [["a"]], message: "Found 1 dependency cycle(s)." });
handlers.orchestrator_error({ type: "execution_deadlock", cycles: [["x"]], message: "Execution deadlock detected." });
handlers.goal_reminder({ turn: 7, reads_without_edit: 6, reminder_count: 2 });
assert.strictEqual(timeline.children.length, t0 + 6, "all six render one element each");

const srv = timeline.children[t0];
assert.ok(srv.textContent.includes("서버 오류(5xx)") && srv.textContent.includes("3") && srv.textContent.includes("2/5"),
  `server_retry renders delay and attempt/max: ${srv.textContent}`);
const bs = timeline.children[t0 + 1];
assert.ok(bs.textContent.includes("3") && bs.textContent.includes("P1"), `batch_start renders task count + priority: ${bs.textContent}`);
assert.strictEqual(bs.title, "a, b, c", "batch_start title lists task ids");
const bc = timeline.children[t0 + 2];
assert.ok(bc.textContent.includes("2/3"), `batch_complete renders success/total: ${bc.textContent}`);
const ow = timeline.children[t0 + 3];
assert.ok(ow.textContent.includes("경고") && ow.textContent.includes("dependency cycle"), `orchestrator_warning renders message: ${ow.textContent}`);
const oe = timeline.children[t0 + 4];
assert.ok(oe.textContent.includes("오류") && oe.textContent.includes("deadlock"), `orchestrator_error renders message: ${oe.textContent}`);
const gr = timeline.children[t0 + 5];
assert.ok(gr.textContent.includes("6") && gr.textContent.includes("#2"), `goal_reminder renders reads + count: ${gr.textContent}`);
assert.strictEqual(gr.title, "Turn 7", "goal_reminder title carries the turn number");

console.log("OK — round 32-2: 7 미표시 emit UI 표시 (live 7 handlers, payload contract locked, content accumulation + complete removal verified)");
