#!/usr/bin/env node
/**
 * Regression harness — P7-1 + P7-2: reasoning streaming O(n²) append and
 * agent-panel scroll hijacking.
 *
 * P7-1 bug: _agentAppendReasoning (agent-panel.js) and the design_reasoning
 * handler (design-chat.js) accumulated streamed reasoning with
 * `textContent += chunk`. Each assignment reserializes the ENTIRE buffer and
 * rebuilds every child node — O(n²) over a long reasoning stream — with no
 * size cap (unbounded DOM growth; 200KB reasoning streams are realistic).
 *
 * P7-2 bug: agent-panel.js assigned `tl.scrollTop = tl.scrollHeight`
 * unconditionally at 44 sites. Every streamed event yanked the viewport to
 * the bottom even while the user was reading earlier history. design-chat.js
 * guards the same pattern with a near-bottom check; agent-panel had none.
 *
 * Fix under test:
 *   - append(document.createTextNode(chunk)) — O(1) per chunk; child text
 *     nodes accumulate (no reserialization), with a ~40KB cap mirroring the
 *     existing design-tool-live pattern (design-chat.js:1091-1096).
 *   - _agentScrollBottom(el): scrolls only when the user was already within
 *     40px of the bottom. All scroll sites route through it.
 *
 * The REAL function texts are sliced out of agent-panel.js / design-chat.js
 * and executed against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_reasoning_scroll.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const AGENT_SRC = fs.readFileSync(
  path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js"), "utf8");
const DESIGN_SRC = fs.readFileSync(
  path.join(__dirname, "..", "..", "webapp", "ui", "static", "design-chat.js"), "utf8");

// ── Source-level gates ──
// P7-2: no BARE `x.scrollTop = x.scrollHeight;` assignment may exist in
// agent-panel.js — every scroll must go through the near-bottom-guarded
// _agentScrollBottom. (The guard's own line is an `if (atBottom)` statement,
// so it never matches the bare-assignment shape.)
{
  const unguarded = [];
  AGENT_SRC.split("\n").forEach((l, i) => {
    if (/^\s*[A-Za-z_$][\w$]*\.scrollTop\s*=\s*[A-Za-z_$][\w$]*\.scrollHeight\s*;?\s*$/.test(l)) {
      unguarded.push(`${i + 1}: ${l.trim()}`);
    }
  });
  assert.deepStrictEqual(
    unguarded,
    [],
    `unguarded scrollTop assignments remain in agent-panel.js:\n  ${unguarded.join("\n  ")}`);
}
// P7-1: the reasoning append paths must not use `textContent +=` anymore
// (design-chat.js:1093 `live.textContent += chunk` is the CAPPED reference
// pattern, intentionally left in place — it is not a reasoning path).
assert.ok(!AGENT_SRC.includes("body.textContent +="),
  "agent-panel.js reasoning append still uses textContent +=");
assert.ok(!DESIGN_SRC.includes("_reasoningBubble.textContent +="),
  "design-chat.js reasoning append still uses textContent +=");

// ── Slice a REAL function out of source (brace-balanced) ──
function sliceFn(src, startMarker) {
  const start = src.indexOf(startMarker);
  assert.ok(start >= 0, `slice start marker not found: ${startMarker}`);
  let depth = 0;
  let i = src.indexOf("{", start);
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces while slicing: ${startMarker}`);
  return src.slice(start, i + 1);
}

const agentScrollBottom = eval(`(${sliceFn(AGENT_SRC, "function _agentScrollBottom(el) {")})`);
const appendReasoning = eval(`(${sliceFn(AGENT_SRC, 'function _agentAppendReasoning(text, agentId = "main") {')})`);

// ── Stub DOM ──
function makeTextNode(data) {
  return { nodeType: 3, data: String(data) };
}

class FakeEl {
  constructor() {
    this.childNodes = [];
    this.style = {};
    this.dataset = {};
    this.className = "";
    this.innerHTML = "";
    this.scrollHeight = 1000;
    this.clientHeight = 600;
    this._scrollTop = 0;
  }
  appendChild(c) { this.childNodes.push(c); c.parentNode = this; return c; }
  setAttribute() {}
  addEventListener(evt, cb) { (this._listeners || (this._listeners = {}))[evt] = cb; }
  remove() {}
  closest() { return null; }
  querySelector() { return null; }
  // Browser-like clamp: max scrollTop = scrollHeight - clientHeight
  get scrollTop() { return this._scrollTop; }
  set scrollTop(v) {
    this._scrollTop = Math.max(0, Math.min(Number(v) || 0, this.scrollHeight - this.clientHeight));
  }
  get textContent() {
    return this.childNodes.map((c) => (c.nodeType === 3 ? c.data : "")).join("");
  }
  set textContent(v) {
    const s = String(v);
    this.childNodes = [makeTextNode(s)];
  }
}

const history = new FakeEl(); // #design-chat-history
let cardEl = null;  // .agent-reasoning-card[data-agent-id=...]
let bodyEl = null;  // .agent-reasoning-body

globalThis.document = {
  getElementById(id) {
    if (id === "design-chat-history") return history;
    return null;
  },
  createElement(tag) {
    const el = new FakeEl();
    if (tag === "div" && !cardEl) {
      cardEl = el;
      el.querySelector = (sel) => (sel === ".agent-reasoning-body" ? bodyEl : null);
    } else if (tag === "pre") {
      bodyEl = el;
    }
    return el;
  },
  createTextNode(s) { return makeTextNode(s); },
};

// P7-4: _agentAppendReasoning reads this module-level constant for the
// default collapsed state of a NEW reasoning card.
globalThis._AGENT_REASONING_DEFAULT_COLLAPSED = false;

// ── Part A: _agentScrollBottom guard semantics ──
{
  // near-bottom (remaining 10 < 40) → scrolls to bottom (clamped to 400)
  const near = new FakeEl();
  near.scrollTop = 390;
  agentScrollBottom(near);
  assert.strictEqual(near.scrollTop, 400, "near-bottom should scroll to bottom");

  // scrolled up (remaining 300 >= 40) → must NOT move  ← the P7-2 regression
  const up = new FakeEl();
  up.scrollTop = 100;
  agentScrollBottom(up);
  assert.strictEqual(up.scrollTop, 100, "scrolled-up viewport must not be hijacked");

  // boundary: exactly 40px remaining → no scroll; 39px → scroll
  const b40 = new FakeEl();
  b40.scrollTop = 360; // remaining = 1000 - 360 - 600 = 40
  agentScrollBottom(b40);
  assert.strictEqual(b40.scrollTop, 360, "exactly 40px remaining should NOT scroll");
  const b39 = new FakeEl();
  b39.scrollTop = 361; // remaining = 39
  agentScrollBottom(b39);
  assert.strictEqual(b39.scrollTop, 400, "39px remaining should scroll");

  // null / undefined → no throw
  agentScrollBottom(null);
  agentScrollBottom(undefined);

  console.log("PASS Part A: _agentScrollBottom guard (5 checks)");
}

// ── Part B: _agentAppendReasoning (agent-panel.js) ──
{
  cardEl = null;
  bodyEl = null;
  const tl = new FakeEl();
  tl.scrollTop = 100; // user scrolled up: remaining 300 >= 40
  tl.querySelector = (sel) => {
    if (sel === ".agent-timeline-placeholder") return null;
    if (sel.startsWith(".agent-reasoning-card")) return cardEl;
    return null;
  };
  globalThis._agentGetTimeline = () => tl;
  globalThis._agentScrollBottom = agentScrollBottom;

  // first chunk: creates the card and appends one text node
  appendReasoning("abc", "main");
  assert.ok(cardEl, "card must be created on first call");
  assert.strictEqual(bodyEl.textContent, "abc");
  assert.strictEqual(bodyEl.childNodes.length, 1);
  assert.strictEqual(tl.scrollTop, 100, "scrolled-up timeline must not be hijacked on first chunk");

  // second chunk: accumulates WITHOUT reserialization  ← the P7-1 regression
  appendReasoning("def", "main");
  assert.strictEqual(bodyEl.textContent, "abcdef");
  assert.strictEqual(bodyEl.childNodes.length, 2,
    "chunks must append as separate text nodes (O(1), no reserialization)");
  assert.strictEqual(tl.scrollTop, 100, "scrolled-up timeline must not be hijacked on streamed chunks");

  // near-bottom → scrolls
  tl.scrollTop = 390;
  appendReasoning("ghi", "main");
  assert.strictEqual(tl.scrollTop, 400, "near-bottom timeline should scroll");

  // cap at ~40KB (mirrors design-chat.js live cap)
  appendReasoning("x".repeat(50000), "main");
  const t = bodyEl.textContent;
  assert.ok(t.startsWith("...(older output truncated)\n"), "cap marker missing");
  assert.ok(t.length <= 40028, `buffer not capped: length ${t.length}`);
  assert.ok(t.endsWith("x".repeat(40000)), "cap must keep the newest 40000 chars");

  // append after truncation still works
  appendReasoning("tail", "main");
  assert.ok(bodyEl.textContent.endsWith("tail"), "append after truncation must work");

  // empty text → early return (no append, no scroll)
  const before = bodyEl.textContent.length;
  appendReasoning("", "main");
  assert.strictEqual(bodyEl.textContent.length, before, "empty text must be ignored");

  console.log("PASS Part B: _agentAppendReasoning (8 checks)");
}

// ── Part C: design_reasoning handler (design-chat.js) ──
{
  const hStart = DESIGN_SRC.indexOf('sse.addEventListener("design_reasoning",');
  assert.ok(hStart >= 0, "design_reasoning handler not found in design-chat.js");
  let depth = 0;
  let i = DESIGN_SRC.indexOf("{", hStart);
  for (; i < DESIGN_SRC.length; i++) {
    if (DESIGN_SRC[i] === "{") depth++;
    else if (DESIGN_SRC[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < DESIGN_SRC.length, "unbalanced braces while slicing design_reasoning handler");
  // Brace balance ends at the arrow-function's closing `}` — re-add the
  // addEventListener call's own `);` that follows it in the source.
  const handlerText = DESIGN_SRC.slice(hStart, i + 1) + ");";

  let captured = null;
  let hideCalls = 0;
  let scrollCalls = 0;
  globalThis.sse = {
    addEventListener(evt, cb) { if (evt === "design_reasoning") captured = cb; },
  };
  globalThis._designHideTypingIndicator = () => { hideCalls++; };
  globalThis._designScrollBottom = () => { scrollCalls++; };
  globalThis._reasoningBubble = null;

  eval(handlerText);
  assert.ok(captured, "design_reasoning callback not captured");

  // empty content → early return before any side effect
  captured({ data: JSON.stringify({ content: "" }) });
  assert.strictEqual(hideCalls, 0, "empty content must return early");

  // first chunk: creates a new reasoning bubble
  captured({ data: JSON.stringify({ content: "hello" }) });
  assert.ok(globalThis._reasoningBubble, "first chunk must create a reasoning bubble");
  assert.strictEqual(globalThis._reasoningBubble.textContent, "hello");
  assert.strictEqual(history.childNodes.length, 1, "wrapper must be appended to history");
  assert.strictEqual(scrollCalls, 1, "first-chunk path must auto-scroll");

  // streaming append: O(1) node append (no reserialization)  ← the P7-1 regression
  captured({ data: JSON.stringify({ content: " world", append: true }) });
  assert.strictEqual(globalThis._reasoningBubble.textContent, "hello world");
  assert.strictEqual(globalThis._reasoningBubble.childNodes.length, 2,
    "chunks must append as separate text nodes (O(1), no reserialization)");
  assert.strictEqual(scrollCalls, 2, "streaming path must auto-scroll per chunk");

  // cap at ~40KB
  captured({ data: JSON.stringify({ content: "y".repeat(50000), append: true }) });
  const rt = globalThis._reasoningBubble.textContent;
  assert.ok(rt.startsWith("...(older output truncated)\n"), "cap marker missing (design-chat)");
  assert.ok(rt.length <= 40028, `buffer not capped (design-chat): length ${rt.length}`);
  assert.ok(rt.endsWith("y".repeat(40000)), "cap must keep the newest 40000 chars (design-chat)");

  console.log("PASS Part C: design_reasoning handler (7 checks)");
}

console.log("\nAll P7-1/P7-2 regression checks passed ✔");
