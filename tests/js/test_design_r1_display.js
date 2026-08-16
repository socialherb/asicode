#!/usr/bin/env node
/**
 * Regression harness — round 32-4: five emitted-but-never-shown design-loop
 * events (design_thinking_start / design_thinking_stop / design_thinking /
 * design_llm_call / design_plan_gate) now have LIVE listeners in
 * design-chat.js. Before this round they were silently dropped:
 *
 *   - The webapp keyed its "thinking" state off design_reasoning only —
 *     turns without a reasoning stream showed NOTHING between tool calls,
 *     while the CLI had a per-LLM-call ticker fed by these same events.
 *   - design_thinking_stop is emitted in _respond_impl's finally covering
 *     ALL exit paths (final-answer early return, LLMClientError,
 *     AgentCancelled, max-iterations tail) — the only reliable signal to
 *     close an open reasoning bubble's ●●● state.
 *   - Interim assistant statements, plan-gate nudges and per-call cache
 *     diagnostics never reached the user.
 *
 * Gates:
 *   R1 presence — the five listeners exist in design-chat.js (live code,
 *      not comments). design-chat.js has no EventSequencer override —
 *      direct sse.addEventListener IS the live path.
 *   R2 payload contract — listeners read the REAL backend payload shapes
 *      (design_chat_loop.py stream_callback calls) and the new CSS classes
 *      have consumers (no orphan rules).
 *   R3 behavior — ticker show, reasoning-bubble close (incl. already-closed
 *      idempotence), interim bubble render + empty no-op, plan-gate chip +
 *      degenerate payload safety, per-call meta line formatting.
 *
 * Run: node tests/js/test_design_r1_display.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "design-chat.js");
const src = fs.readFileSync(srcPath, "utf8");
const cssPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.css");
const css = fs.readFileSync(cssPath, "utf8");

const NAMES = [
  "design_thinking_start", "design_thinking_stop", "design_thinking",
  "design_llm_call", "design_plan_gate",
];

// ── Slice helper: extract a listener's arrow-function body (brace-balanced) ─
function sliceListenerBody(name) {
  const anchor = `sse.addEventListener("${name}"`;
  const idx = src.indexOf(anchor);
  assert.ok(idx >= 0, `listener ${name} must exist in design-chat.js (round 32-4)`);
  const lineStart = src.lastIndexOf("\n", idx) + 1;
  assert.ok(!src.slice(lineStart, idx).includes("//"),
    `${name} listener must be live code, not a comment`);
  const arrow = src.indexOf("=>", idx);
  assert.ok(arrow > 0 && arrow < idx + anchor.length + 80, `${name} listener must be an arrow function`);
  const open = src.indexOf("{", arrow);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") { depth--; if (depth === 0) break; }
  }
  assert.ok(i < src.length, `unbalanced braces while extracting ${name}`);
  return src.slice(open + 1, i);
}

const bodies = {};
for (const n of NAMES) bodies[n] = sliceListenerBody(n);

// ── R2: payload contracts (real design_chat_loop.py shapes) ───────────────
// design_chat_loop: stream_callback("design_thinking", {content, elapsed})
assert.ok(bodies.design_thinking.includes("d.content") && bodies.design_thinking.includes("d.elapsed"),
  "design_thinking reads {content, elapsed} (design_chat_loop payload)");
// stream_callback("design_plan_gate", {open_items, nudge, max_nudges})
for (const f of ["d.open_items", "d.nudge", "d.max_nudges"]) {
  assert.ok(bodies.design_plan_gate.includes(f), `design_plan_gate reads ${f} (plan-gate payload)`);
}
// stream_callback("design_llm_call", {prompt_tokens, completion_tokens,
//  cache_read/creation_tokens, cache_hit_ratio, provider, tool_call_count})
for (const f of ["d.prompt_tokens", "d.completion_tokens", "d.cache_hit_ratio", "d.provider", "d.tool_call_count"]) {
  assert.ok(bodies.design_llm_call.includes(f), `design_llm_call reads ${f} (per-call diagnostics payload)`);
}
// start/stop pair: {} payloads — ticker + reasoning-state close
assert.ok(bodies.design_thinking_start.includes("_designShowTypingIndicator"),
  "thinking_start drives the typing indicator (server-confirmed LLM call)");
assert.ok(bodies.design_thinking_stop.includes("_reasoningBubble") && bodies.design_thinking_stop.includes("design-reasoning-label"),
  "thinking_stop closes the open reasoning bubble state (finally covers all exit paths)");

// ── R2b: new CSS classes have consumers (no orphan rules) ─────────────────
for (const cls of ["design-plan-gate-chip", "design-llm-call-meta"]) {
  assert.ok(css.includes(`.${cls}`), `${cls} rule must exist in ui.css`);
  assert.ok(src.includes(cls), `${cls} must be consumed by design-chat.js (no orphan CSS)`);
}

// ── Stub DOM ──────────────────────────────────────────────────────────────
function makeEl(tag) {
  const el = {
    tag, className: "", textContent: "", title: "", style: {}, children: [],
    appendChild(c) {
      el.children.push(c);
      if (c && typeof c.text === "string") el.textContent += c.text;
      else if (c && typeof c.textContent === "string") el.textContent += c.textContent;
      return c;
    },
    remove() {},
    closest() { return null; },
    querySelector() { return null; },
    classList: { add() {}, remove() {}, toggle() {} },
  };
  return el;
}
const history = { children: [], appendChild(c) { history.children.push(c); return c; } };
const calls = { show: 0, hide: 0 };
const deps = {
  _designShowTypingIndicator: () => { calls.show++; },
  _designHideTypingIndicator: () => { calls.hide++; },
  _designScrollBottom: () => {},
  document: {
    getElementById: (id) => (id === "design-chat-history" ? history : null),
    createElement: makeEl,
  },
};
function fire(name, payload, reasoningBubble) {
  const fn = new Function("e", "d", "_reasoningBubble",
    `"use strict"; const { _designShowTypingIndicator, _designHideTypingIndicator, _designScrollBottom, document } = d; ${bodies[name]}`);
  fn(payload === undefined ? {} : { data: JSON.stringify(payload) }, deps, reasoningBubble === undefined ? null : reasoningBubble);
}

// ── R3a: thinking_start drives the ticker ─────────────────────────────────
const show0 = calls.show;
fire("design_thinking_start");
assert.strictEqual(calls.show, show0 + 1, "thinking_start shows the typing indicator");

// ── R3b: thinking_stop closes the reasoning bubble state, idempotently ────
const label = { innerHTML: "" };
const wrapperEl = { querySelector: (sel) => (sel.includes("design-reasoning-label") ? label : null) };
const rBubble = { closest: () => wrapperEl };
fire("design_thinking_stop", undefined, rBubble);
assert.ok(String(label.innerHTML).includes("✓"), `reasoning label flipped to ✓ Thought: ${label.innerHTML}`);
fire("design_thinking_stop", undefined, null); // already closed → no crash

// ── R3c: design_thinking renders one interim bubble; empty is a no-op ─────
const n0 = history.children.length;
fire("design_thinking", { content: "파일 구조를 먼저 확인하겠습니다.", elapsed: 12.34 });
assert.strictEqual(history.children.length, n0 + 1, "one interim bubble per statement");
const interim = history.children[history.children.length - 1];
assert.ok(interim.className.includes("design-msg--interim"), "rendered as an interim AI message");
assert.ok(interim.textContent.includes("중간 응답"), "meta marks it as interim (distinct from final answer)");
assert.ok(interim.textContent.includes("12.3s"), "meta carries elapsed seconds");
assert.ok(interim.textContent.includes("파일 구조"), "statement text rendered");
fire("design_thinking", { content: "   " });
fire("design_thinking", {});
assert.strictEqual(history.children.length, n0 + 1, "blank/empty content is a no-op");

// ── R3d: design_plan_gate chip with hover list; degenerate safety ─────────
fire("design_plan_gate", { open_items: ["API 스펙 정리", "라우터 구현"], nudge: 1, max_nudges: 3 });
const chip = history.children[history.children.length - 1];
assert.ok(chip.className.includes("design-plan-gate-chip"), "rendered as a plan-gate chip");
assert.ok(chip.textContent.includes("2개") && chip.textContent.includes("1/3"),
  `chip renders open count + nudge progress: ${chip.textContent}`);
assert.strictEqual(chip.title, "API 스펙 정리\n라우터 구현", "chip title lists open items");
fire("design_plan_gate", {});
assert.ok(history.children[history.children.length - 1].textContent.includes("0개"),
  "degenerate payload still renders safely");

// ── R3e: design_llm_call compact diagnostics line ─────────────────────────
fire("design_llm_call", {
  prompt_tokens: 1234, completion_tokens: 567, cache_hit_ratio: 0.85,
  provider: "anthropic", tool_call_count: 2,
});
const line = history.children[history.children.length - 1];
assert.ok(line.className.includes("design-llm-call-meta"), "rendered as a dim meta line");
assert.ok(line.textContent.includes("anthropic"), `provider shown: ${line.textContent}`);
assert.ok(line.textContent.includes("1,234→567"), `token counts shown: ${line.textContent}`);
assert.ok(line.textContent.includes("85%"), `cache efficiency shown: ${line.textContent}`);
assert.ok(line.textContent.includes("tools 2"), `tool fan-out shown: ${line.textContent}`);
fire("design_llm_call", {});
const line2 = history.children[history.children.length - 1];
assert.ok(line2.textContent.includes("llm") && line2.textContent.includes("0→0"),
  `degenerate payload falls back safely: ${line2.textContent}`);

console.log("OK — round 32-4: design 5종 미표시 emit UI 표시 (live listeners, payload contract locked, ticker/interim/chip/meta behavior verified)");
