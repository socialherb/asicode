#!/usr/bin/env node
/**
 * Regression harness — round-31 #A: the orchestrator emits subagent_error /
 * subagent_retry / subagent_waiting_ipc but the LIVE handlers map (the only
 * dispatch path — source.addEventListener is overridden to a no-op after
 * EventSequencer registration) had no entries for them. A crashed subagent
 * left its taskboard item and inline section header spinning "running" until
 * the whole run ended, and the error message / traceback were never shown.
 *
 * Gates:
 *   R1 presence — the three handlers live inside the live handlers-map region
 *      ("// Define event handlers" … "// Create EventSequencer").
 *   R2 wiring   — no dead post-wire addEventListener("subagent_*") twins
 *      (they would be silently ignored), CSS classes referenced by the
 *      handlers exist in ui.css, server text rendered via textContent only.
 *   R3 behavior — the start/waiting/error/retry/complete state machine against
 *      a stub DOM: header class/icon transitions, in-place heartbeat updates
 *      (no unbounded timeline growth), taskboard statuses, traceback clip.
 *
 * Run: node tests/js/test_agent_subagent_error_handlers.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const cssPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.css");
const src = fs.readFileSync(srcPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");

// ── Slice helpers (brace-balanced) ────────────────────────────────────────
function sliceFunction(anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found in agent-panel.js`);
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

// ── R1: the new handlers must live in the LIVE map region ────────────────
const regionStart = src.indexOf("// Define event handlers");
const regionEnd = src.indexOf("// Create EventSequencer");
assert.ok(regionStart >= 0 && regionEnd > regionStart, "handlers-map region markers must exist");
for (const anchor of [
  "    subagent_error: (data) => {",
  "    subagent_retry: (data) => {",
  "    subagent_waiting_ipc: (data) => {",
]) {
  const i = src.indexOf(anchor);
  assert.ok(i >= 0, `${anchor.trim()} must exist`);
  assert.ok(i > regionStart && i < regionEnd, `${anchor.trim()} must live in the live handlers map (not a dead listener)`);
}

const startEntry = sliceMapEntry("    subagent_start: (data) => {");
const completeEntry = sliceMapEntry("    subagent_complete: (data) => {");
const errorEntry = sliceMapEntry("    subagent_error: (data) => {");
const retryEntry = sliceMapEntry("    subagent_retry: (data) => {");
const waitingEntry = sliceMapEntry("    subagent_waiting_ipc: (data) => {");
const helpers = sliceFunction("function _escHtml(");

// ── R2: wiring gates ─────────────────────────────────────────────────────
assert.ok(!/addEventListener\(\s*"subagent_(error|retry|waiting_ipc)"/.test(src),
  "no dead post-wire addEventListener twins — they are silently ignored after EventSequencer registration");
assert.ok(completeEntry.includes(".agent-section-wait"),
  "subagent_complete must drop the IPC heartbeat span (the wait is over)");
assert.ok(!/innerHTML\s*=/.test(errorEntry) && !/innerHTML\s*=/.test(retryEntry),
  "server-sourced text must render via textContent only — no innerHTML assignment (XSS-safe)");
for (const cls of [".agent-section-wait", ".agent-subagent-error", ".agent-subagent-error-msg", ".agent-subagent-retry"]) {
  assert.ok(css.includes(cls), `ui.css must style ${cls} (referenced-but-unstyled class)`);
}

// ── Stub DOM ─────────────────────────────────────────────────────────────
function makeEl(tag) {
  const el = {
    tag: tag || "div",
    className: "",
    textContent: "",
    title: "",
    dataset: {},
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
      const m = /^\.([A-Za-z0-9_-]+)$/.exec(sel);
      if (!m) return null;
      return this.children.find((c) => (c.className || "").split(/\s+/).includes(m[1])) || null;
    },
  };
  const classes = () => new Set((el.className || "").split(/\s+/).filter(Boolean));
  el.classList = {
    add: (...cs) => { const s = classes(); cs.forEach((c) => s.add(c)); el.className = [...s].join(" "); },
    remove: (...cs) => { const s = classes(); cs.forEach((c) => s.delete(c)); el.className = [...s].join(" "); },
    contains: (c) => classes().has(c),
  };
  let inner = "";
  Object.defineProperty(el, "innerHTML", {
    get: () => inner,
    set: (v) => {
      inner = String(v);
      el.children = [];
      // subagent_start writes the icon span via innerHTML — synthesize the
      // child so icon swaps in complete/error/retry are observable.
      if (inner.includes("agent-section-icon")) {
        const icon = makeEl("span");
        icon.className = "agent-section-icon";
        icon.textContent = "⟳";
        el.appendChild(icon);
      }
    },
  });
  return el;
}

const timeline = makeEl("div");
timeline.querySelector = (sel) => {
  let m = /^\.([A-Za-z0-9_-]+)\[data-agent-id="([^"]*)"\]$/.exec(sel);
  if (m) {
    return timeline.children.find((c) =>
      (c.className || "").split(/\s+/).includes(m[1]) && c.dataset && c.dataset.agentId === m[2]) || null;
  }
  m = /^\.([A-Za-z0-9_-]+)$/.exec(sel);
  if (m) return timeline.children.find((c) => (c.className || "").split(/\s+/).includes(m[1])) || null;
  return null;
};
globalThis.document = {
  createElement: (t) => makeEl(t),
  getElementById: (id) => (id === "agent-timeline" ? timeline : null),
};
globalThis.state = { agentSubtasks: [], agentRequestText: "root request", agentMaxTurns: 20 };
let boardRenders = 0;
globalThis._renderAgentTaskBoard = () => { boardRenders++; };
let scrolls = 0;
globalThis._agentScrollBottom = () => { scrolls++; };
const statusCalls = [];
globalThis._agentSetStatus = (...a) => { statusCalls.push(a); };
globalThis._setUIMode = () => {};
globalThis._agentShowContinueBar = () => {};

const factory = new Function(
  `${helpers}\nreturn ({ ${startEntry}, ${completeEntry}, ${errorEntry}, ${retryEntry}, ${waitingEntry} });`
);
const handlers = factory();
const findByClass = (cls) => timeline.children.filter((c) => (c.className || "").split(/\s+/).includes(cls));

// ── R3a: start → waiting heartbeat (in place) → complete ─────────────────
handlers.subagent_start({ task_id: "t1", title: "웹 패널 수정 <x>" });
let hdr = timeline.querySelector('.agent-section-header[data-agent-id="t1"]');
assert.ok(hdr, "start must create an inline section header");
assert.ok(hdr.classList.contains("running"), "header starts running");
assert.strictEqual(globalThis.state.agentSubtasks[0].status, "running", "taskboard starts running");

handlers.subagent_waiting_ipc({ agent_id: "t1", elapsed_s: 0.0 });
const wait1 = hdr.querySelector(".agent-section-wait");
assert.ok(wait1, "waiting_ipc must attach a heartbeat span");
handlers.subagent_waiting_ipc({ agent_id: "t1", elapsed_s: 12.4, turn: 3, last_tool: "edit_text" });
assert.strictEqual(hdr.children.filter((c) => (c.className || "").split(/\s+/).includes("agent-section-wait")).length, 1,
  "repeated heartbeat must update in place, never append (unbounded growth)");
assert.ok(wait1.textContent.includes("12"), `heartbeat shows seconds: ${wait1.textContent}`);
assert.ok(wait1.textContent.includes("턴 3") && wait1.textContent.includes("edit_text"),
  `heartbeat shows turn + last_tool: ${wait1.textContent}`);
handlers.subagent_waiting_ipc({ agent_id: "nobody", elapsed_s: 1 }); // no header: defensive no-op
assert.strictEqual(statusCalls.length, 0, "waiting_ipc without a header must not raise");

handlers.subagent_complete({ task_id: "t1", status: "success" });
assert.ok(hdr.classList.contains("completed") && !hdr.classList.contains("running"), "complete marks the header completed");
assert.strictEqual(hdr.querySelector(".agent-section-wait"), null, "complete must drop the heartbeat span");
assert.strictEqual(hdr.querySelector(".agent-section-icon").textContent, "✓", "icon flips to ✓");
assert.strictEqual(globalThis.state.agentSubtasks[0].status, "success", "taskboard flips to success");

// ── R3b: the headline bug — crash path (complete never fires) ────────────
handlers.subagent_start({ task_id: "t2", title: "리팩터" });
const hdr2 = timeline.querySelector('.agent-section-header[data-agent-id="t2"]');
handlers.subagent_waiting_ipc({ agent_id: "t2", elapsed_s: 5.5, turn: 1, last_tool: "bash" });
handlers.subagent_error({
  task_id: "t2",
  error: "boom <script>alert(1)</script>",
  traceback: "Traceback (most recent call last):\n  File \"o.py\", line 1, in <module>\nboom",
});
assert.ok(hdr2.classList.contains("failed") && !hdr2.classList.contains("running"),
  "error must fail the header (was stuck spinning before this fix)");
assert.strictEqual(hdr2.querySelector(".agent-section-wait"), null, "error must drop the heartbeat span");
assert.strictEqual(hdr2.querySelector(".agent-section-icon").textContent, "✗", "icon flips to ✗");
const errCards = findByClass("agent-subagent-error");
assert.strictEqual(errCards.length, 1, "exactly one crash card rendered");
assert.ok(errCards[0].children.some((c) => c.textContent.includes("Traceback")),
  "the traceback — previously shown NOWHERE — must be visible");
assert.ok(errCards[0].children.some((c) => c.textContent.includes("t2")), "card names the failing agent");
const t2 = globalThis.state.agentSubtasks.find((t) => t.agentId === "t2");
assert.strictEqual(t2.status, "error", "taskboard item must leave running → error");

// ── R3c: error without a prior start (defensive) — and the XSS fallback:
// when NO traceback is supplied the raw error string IS the body, and must
// round-trip as text, never executed markup.
const cardsBefore = findByClass("agent-subagent-error").length;
handlers.subagent_error({ task_id: "ghost", error: "no header case <script>alert(1)</script>" });
assert.strictEqual(findByClass("agent-subagent-error").length, cardsBefore + 1,
  "crash card still renders when no header exists");
assert.ok(findByClass("agent-subagent-error").some((c) => c.children.some((k) => k.textContent.includes("<script>alert(1)</script>"))),
  "raw server error text round-trips as textContent (XSS-safe), not executed markup");
assert.strictEqual(statusCalls.length, 0, "no-header error must not raise");

// ── R3d: review retry resurrects the header + shows feedback ─────────────
handlers.subagent_start({ task_id: "t3", title: "서브3" });
const hdr3 = timeline.querySelector('.agent-section-header[data-agent-id="t3"]');
handlers.subagent_error({ task_id: "t3", error: "first attempt failed" });
assert.ok(hdr3.classList.contains("failed"), "precondition: t3 failed");
handlers.subagent_retry({
  task_id: "t3", retry: 1, max_retries: 3,
  feedback: "edit_text 결과를 검증하지 않았음 <img onerror=1>",
  reverted_files: ["a.py", "b.py"],
});
assert.ok(hdr3.classList.contains("running") && !hdr3.classList.contains("failed"),
  "retry must return the header to running (a failed-looking header would lie)");
assert.strictEqual(hdr3.querySelector(".agent-section-icon").textContent, "⟳", "retry icon spins again");
const chips = findByClass("agent-subagent-retry");
assert.strictEqual(chips.length, 1, "exactly one retry chip per retry event");
assert.ok(chips[0].textContent.includes("1/3"), `chip shows attempt count: ${chips[0].textContent}`);
assert.ok(chips[0].textContent.includes("2"), "chip shows reverted file count");
assert.ok((chips[0].title || "").includes("edit_text"), "full review feedback available on hover");
const t3 = globalThis.state.agentSubtasks.find((t) => t.agentId === "t3");
assert.strictEqual(t3.status, "running", "taskboard back to running during retry");

// ── R3e: long traceback clipped ───────────────────────────────────────────
handlers.subagent_error({ task_id: "t3", error: "e", traceback: "X".repeat(2000) });
const lastCard = findByClass("agent-subagent-error").pop();
const msgEl = lastCard.children.find((c) => (c.className || "").split(/\s+/).includes("agent-subagent-error-msg"));
assert.strictEqual(msgEl.textContent.length, 1201, "traceback clipped to 1200 chars + ellipsis");
assert.ok(msgEl.textContent.endsWith("…"), "clip is visible to the user");

assert.ok(scrolls > 0, "new timeline entries scroll into view");
assert.strictEqual(statusCalls.length, 0, "no handler catch fired — all paths clean");

console.log("PASS: subagent error/retry/waiting_ipc handlers (round-31 #A)");
