#!/usr/bin/env node
/**
 * Regression harness — round 32: the orchestrator emits review_start /
 * review_approved / review_rejected / review_skipped for subagent review,
 * but the live handlers map listened on review_gate_start/pending/passed/
 * rejected — relics of a removed backend flow whose payloads (turn, summary,
 * applied_patches) no longer exist anywhere. Every subagent review verdict
 * was silently dropped: the UI never left the "reviewing" state per card and
 * the reviewer's feedback was never shown.
 *
 * Gates:
 *   R1 presence — the four handlers live inside the LIVE handlers-map region
 *      ("// Define event handlers" … "// Create EventSequencer").
 *   R2 convergence — no review_gate_* string anywhere in agent-panel.js (map
 *      keys AND dead post-wire addEventListener twins), no dead review_start
 *      twin, and the handlers read the orchestrator's real payload contract
 *      (title/task_id, feedback/note, feedback, reason) — not the relic
 *      fields (turn/summary/applied_patches).
 *   R3 behavior — invoking the handlers with the exact payloads orchestrator
 *      emits produces the card kinds/status transitions the UI promised.
 *
 * Run: node tests/js/test_agent_review_gate_convergence.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

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

// ── R1: the converged handlers must live in the LIVE map region ──────────
const regionStart = src.indexOf("// Define event handlers");
const regionEnd = src.indexOf("// Create EventSequencer");
assert.ok(regionStart >= 0 && regionEnd > regionStart, "handlers-map region markers must exist");
for (const anchor of [
  "    review_start: (data) => {",
  "    review_approved: (data) => {",
  "    review_rejected: (data) => {",
  "    review_skipped: (data) => {",
]) {
  const i = src.indexOf(anchor);
  assert.ok(i >= 0, `${anchor.trim()} must exist (round-32 convergence)`);
  assert.ok(i > regionStart && i < regionEnd, `${anchor.trim()} must live in the live handlers map (not a dead listener)`);
}

// ── R2: convergence gates ─────────────────────────────────────────────────
// Code-form gate: a map key (`review_gate_x: (data) =>`) or a string literal
// (`addEventListener("review_gate_x"`) is a violation; prose comments that
// document the round-32 removal legitimately mention the old prefix.
assert.ok(!/review_gate_[a-z_]+\s*:/.test(src),
  "no review_gate_* map keys may remain — the orchestrator only emits review_* (round 32)");
assert.ok(!/["']review_gate_/.test(src),
  "no review_gate_* string literals (dead addEventListener twins) may remain — the orchestrator only emits review_* (round 32)");
assert.ok(!/addEventListener\(\s*"review_start"/.test(src),
  "no dead post-wire review_start twin — it is silently ignored after EventSequencer registration and collides with the converged subagent-review name");

const startEntry = sliceMapEntry("    review_start: (data) => {");
const approvedEntry = sliceMapEntry("    review_approved: (data) => {");
const rejectedEntry = sliceMapEntry("    review_rejected: (data) => {");
const skippedEntry = sliceMapEntry("    review_skipped: (data) => {");

// The handlers must read the orchestrator payload contract (orchestrator.py
// _review_subagent): start={task_id,title} approved={feedback?,note?}
// rejected={feedback} skipped={reason} — NOT the relic fields.
assert.ok(startEntry.includes("data?.title") && startEntry.includes("data?.task_id"),
  "review_start must render the subtask title/task_id (orchestrator payload)");
assert.ok(approvedEntry.includes("data?.feedback") && approvedEntry.includes("data?.note"),
  "review_approved must fall back feedback → note → LGTM (both orchestrator emit paths)");
assert.ok(rejectedEntry.includes("data?.feedback"),
  "review_rejected must render the reviewer feedback (orchestrator payload)");
assert.ok(skippedEntry.includes("data?.reason"),
  "review_skipped must render the skip reason (orchestrator payload)");
for (const [name, entry] of [["start", startEntry], ["approved", approvedEntry], ["rejected", rejectedEntry], ["skipped", skippedEntry]]) {
  assert.ok(!/data\?\.(turn|summary|applied_patches)/.test(entry),
    `review_${name} must not read relic payload fields (turn/summary/applied_patches) — they are never emitted`);
}

// ── Stub DOM ──────────────────────────────────────────────────────────────
const cards = [];
const statusCalls = [];
const uiModes = [];
globalThis._agentGetTimeline = () => ({ id: "agent-timeline" });
globalThis._agentAddReviewCard = (kind, title, detail, tl) => { cards.push({ kind, title, detail, tl }); };
globalThis._agentSetStatus = (...a) => { statusCalls.push(a); };
globalThis._setUIMode = (...a) => { uiModes.push(a); };

const factory = new Function(
  "ctx",
  `return ({ ${startEntry}, ${approvedEntry}, ${rejectedEntry}, ${skippedEntry} });`
);
const handlers = factory({ multiAgent: false });
assert.strictEqual(cards.length, 0, "no cards before any event");

// ── R3a: review_start — {task_id, title} (orchestrator.py L3740) ─────────
handlers.review_start({ task_id: "t9", title: "Fix XSS escape" });
assert.strictEqual(cards[0].kind, "running", "start → running card");
assert.ok(cards[0].detail.includes("Fix XSS escape"), `start renders subtask title: ${cards[0].detail}`);
assert.strictEqual(tl(cards[0]), "agent-timeline", "card attached to the live timeline");
assert.deepStrictEqual(uiModes[0] && uiModes[0][0], "review_patch", "start enters review_patch UI mode");
assert.deepStrictEqual(statusCalls[0], ["reviewing"], "start → status reviewing");

// ── R3b: review_approved normal path — feedback empty → LGTM (L3808) ─────
handlers.review_approved({ task_id: "t9", approved: true, feedback: "", target_symbol: "" });
assert.strictEqual(cards[1].kind, "pass", "approved → pass card");
assert.strictEqual(cards[1].detail, "LGTM", "empty feedback falls back to LGTM");
assert.deepStrictEqual(statusCalls[1], ["running"], "approved → status running (verdict arrived)");

// ── R3c: review_approved parse-fail path — {note} (L3793) ─────────────────
handlers.review_approved({ task_id: "t9", note: "parse_failed_assumed_ok" });
assert.strictEqual(cards[2].detail, "parse_failed_assumed_ok", "note is shown when feedback is absent");

// ── R3d: review_rejected — {feedback} (L3771/L3808) ───────────────────────
handlers.review_rejected({ task_id: "t9", approved: false, feedback: "No changes detected (git diff is empty).", target_symbol: "" });
assert.strictEqual(cards[3].kind, "fixed", "rejected → fixed card (rework expected)");
assert.ok(cards[3].detail.includes("No changes detected"), `rejected renders reviewer feedback: ${cards[3].detail}`);
assert.deepStrictEqual(uiModes[uiModes.length - 1][0], "review_patch", "rejected re-enters review_patch UI mode");
assert.deepStrictEqual(statusCalls[statusCalls.length - 1], ["running"], "rejected → status running");

// ── R3e: review_skipped — {reason} (L3733) ────────────────────────────────
handlers.review_skipped({ task_id: "t9", reason: "no_assigned_files_parallel" });
assert.strictEqual(cards[4].kind, "pass", "skipped → pass card (unverifiable, not blocking)");
assert.ok(cards[4].detail.includes("no_assigned_files_parallel"), `skipped renders reason: ${cards[4].detail}`);
assert.deepStrictEqual(statusCalls[statusCalls.length - 1], ["running"], "skipped → status running");

function tl(card) { return card.tl && card.tl.id; }

console.log("OK — review_* convergence gates (4 handlers live, payload contract locked, 5 behavioral cases)");
