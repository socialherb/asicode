#!/usr/bin/env node
/**
 * Regression harness — P7-3 + P7-4.
 *
 * P7-3 bug (typewriterEffect, ui.js): every tick did `el.textContent +=
 * chunk` — reserializing the ENTIRE buffer and rebuilding every child node
 * (O(n²) over a long run) — with an unconditional
 * `scrollParent.scrollTop = scrollParent.scrollHeight` (viewport hijack while
 * the user reads earlier history) and no generation token: a stale rAF chain
 * kept typing (and scroll-fighting) after the user sent a new message,
 * cancelled, or cleared the timeline. Multi-KB thinking text also animated
 * for 10+ seconds (3 chars/tick at ~16ms).
 *
 * Fix under test:
 *   - >500 chars renders immediately (cap)
 *   - append(document.createTextNode(chunk)) — O(1) per tick
 *   - _agentScrollBottom guard — scrolls only when near the bottom
 *   - _typewriterSeq generation token (P3-3 pattern); bumped by
 *     _launchAgent / agentRunStreamCancel / _agentResetForNewRun (source gate)
 *
 * P7-4 (reasoning card, agent-panel.js): the reasoning card accumulated
 * across turns with no way to collapse it. Fix under test: clicking the
 * card title toggles the body; new cards honor
 * _AGENT_REASONING_DEFAULT_COLLAPSED.
 *
 * The REAL function texts are sliced out of ui.js / agent-panel.js and
 * executed against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_typewriter_collapse.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const UI_SRC = fs.readFileSync(
  path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js"), "utf8");
const AGENT_SRC = fs.readFileSync(
  path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js"), "utf8");

// ── Source-level gates ──
// P7-3: no `textContent +=` may remain inside typewriterEffect.
{
  const fnStart = UI_SRC.indexOf("function typewriterEffect(el, text, speed = 12) {");
  assert.ok(fnStart >= 0, "typewriterEffect not found in ui.js");
  const fnText = UI_SRC.slice(fnStart, UI_SRC.indexOf("\nfunction ", fnStart + 10));
  assert.ok(!fnText.includes("el.textContent +="),
    "typewriterEffect still uses textContent += (O(n²) reserialization)");
  assert.ok(fnText.includes("_typewriterSeq"), "typewriterEffect lost its generation token");
  assert.ok(fnText.includes("_agentScrollBottom"), "typewriterEffect lost its near-bottom scroll guard");
  assert.ok(fnText.includes("isConnected"), "typewriterEffect lost its detached-element guard");
  assert.ok(fnText.includes("> 500"), "typewriterEffect lost its long-text immediate-render cap");
}
// P7-3 cancel wiring: new run / user cancel / timeline clear must bump the token.
for (const [src, fnName] of [[UI_SRC, "function _launchAgent("],
                             [AGENT_SRC, "function agentRunStreamCancel() {"],
                             [AGENT_SRC, "function _agentResetForNewRun(multiAgent) {"]]) {
  const i = src.indexOf(fnName);
  assert.ok(i >= 0, `cancel-wiring anchor not found: ${fnName}`);
  const head = src.slice(i, i + 400);
  assert.ok(head.includes("_typewriterSeq++"),
    `cancel wiring missing: ${fnName} must bump _typewriterSeq`);
}
// P7-4: toggle present in the card-creation path.
{
  const fnStart = AGENT_SRC.indexOf('function _agentAppendReasoning(text, agentId = "main") {');
  assert.ok(fnStart >= 0, "_agentAppendReasoning not found in agent-panel.js");
  const fnText = AGENT_SRC.slice(fnStart, AGENT_SRC.indexOf("\nfunction ", fnStart + 10));
  assert.ok(fnText.includes('addEventListener("click"'), "reasoning card lost its collapse toggle");
  assert.ok(fnText.includes("_AGENT_REASONING_DEFAULT_COLLAPSED"),
    "reasoning card lost its default-collapsed option");
  assert.ok(AGENT_SRC.includes("const _AGENT_REASONING_DEFAULT_COLLAPSED = false;"),
    "_AGENT_REASONING_DEFAULT_COLLAPSED constant missing");
}

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

const typewriterEffect = eval(`(${sliceFn(UI_SRC, "function typewriterEffect(el, text, speed = 12) {")})`);
const agentScrollBottom = eval(`(${sliceFn(AGENT_SRC, "function _agentScrollBottom(el) {")})`);
const appendReasoning = eval(`(${sliceFn(AGENT_SRC, 'function _agentAppendReasoning(text, agentId = "main") {')})`);

// The token lives at module scope in ui.js — provide the runtime cell the
// sliced function reads as a free variable.
globalThis._typewriterSeq = 0;
globalThis._agentScrollBottom = agentScrollBottom;
globalThis._AGENT_REASONING_DEFAULT_COLLAPSED = false;

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
    this.isConnected = false;
    this._attrs = {};
    this._listeners = {};
    this.scrollHeight = 1000;
    this.clientHeight = 600;
    this._scrollTop = 0;
    this.classList = {
      _set: new Set(),
      add(...cs) { cs.forEach((c) => this._set.add(c)); },
      remove(...cs) { cs.forEach((c) => this._set.delete(c)); },
      contains(c) { return this._set.has(c); },
    };
  }
  appendChild(c) { this.childNodes.push(c); c.parentNode = this; c.isConnected = true; return c; }
  setAttribute(k, v) { this._attrs[k] = String(v); }
  addEventListener(evt, cb) { (this._listeners[evt] || (this._listeners[evt] = [])).push(cb); }
  remove() {
    this.isConnected = false;
    if (this.parentNode) {
      const i = this.parentNode.childNodes.indexOf(this);
      if (i >= 0) this.parentNode.childNodes.splice(i, 1);
      this.parentNode = null;
    }
  }
  // Real matching for `.class` and `.class[data-x="y"]` — the production code
  // re-finds its own children this way (`card.querySelector(".agent-reasoning-body")`),
  // so a stub that always returns null silently skips the append path.
  _matches(sel) {
    const m = /^\.([\w-]+)(?:\[data-([\w-]+)="([^"]*)"\])?$/.exec(String(sel || ""));
    if (!m) return false;
    if (!String(this.className).split(/\s+/).includes(m[1])) return false;
    if (m[2] && this._attrs["data-" + m[2]] !== m[3]) return false;
    return true;
  }
  querySelector(sel) {
    for (const c of this.childNodes) {
      if (c.nodeType === 3) continue;
      if (c._matches(sel)) return c;
      const deep = c.querySelector(sel);
      if (deep) return deep;
    }
    return null;
  }
  querySelectorAll() { return []; }
  closest(sel) {
    if (sel === ".agent-timeline") return this._timeline || null;
    return null; // .agent-chat-msg → no parent bubble in these tests
  }
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
    // DOM "string replace all": an empty string clears the children and adds
    // NO text node (a stub that appends an empty node inflates childNodes).
    this.childNodes = s === "" ? [] : [makeTextNode(s)];
  }
}

// Manual rAF queue so we can advance the animation tick by tick.
const rafQueue = [];
globalThis.requestAnimationFrame = (cb) => { rafQueue.push(cb); };
function drain(n = Infinity) {
  let c = 0;
  while (rafQueue.length && c < n) { rafQueue.shift()(); c++; }
}
// The queue is shared across cases and a case that stops mid-animation leaves
// its terminal tick behind — without this, the next case's drain(1) would
// spend its tick on the PREVIOUS run's leftover.
function resetRaf() { rafQueue.length = 0; }
const flush = () => new Promise((r) => setImmediate(r));

// document stub — createTextNode is used by typewriterEffect (Part A);
// createElement is specialized for the reasoning card (Part B).
globalThis.document = {
  getElementById() { return null; },
  createElement() { return new FakeEl(); },
  createTextNode(s) { return makeTextNode(s); },
};

// ── Part A: typewriterEffect (P7-3) ──
(async () => {
  // A1: short text types char-by-char, resolves, cleans the typing class
  {
    const el = new FakeEl();
    el.isConnected = true;
    el.classList.add("agent-chat-msg");  // classList is a stub; add marker for check
    const timeline = new FakeEl();
    timeline.scrollTop = 390;  // near bottom: remaining 10 < 40
    el._timeline = timeline;
    let done = false;
    typewriterEffect(el, "hello", 10).then(() => { done = true; });
    assert.strictEqual(el.textContent, "", "animation must start empty");
    assert.strictEqual(done, false, "must not resolve before typing finishes");
    drain(1);
    assert.strictEqual(el.textContent, "h", "first tick must reveal 1 char");
    drain(4);
    assert.strictEqual(el.textContent, "hello", "full text after 5 ticks");
    // The tick that writes the last char still schedules one more; finish()
    // (= resolve) runs in that terminal tick — N chars need N+1 ticks.
    drain(1);
    await flush();
    assert.strictEqual(done, true, "must resolve after the last tick");
    // O(1) append proof: each char is its own text node (no reserialization)
    assert.strictEqual(el.childNodes.length, 5,
      "chars must append as separate text nodes, not reserialize the buffer");
    assert.strictEqual(timeline.scrollTop, 400, "near-bottom timeline must be scrolled");
  }

  // A2: long text (>500) renders immediately — no animation ticks at all
  {
    resetRaf();
    const el = new FakeEl();
    el.isConnected = true;
    const long = "x".repeat(501);
    let done = false;
    const q0 = rafQueue.length;
    typewriterEffect(el, long, 10).then(() => { done = true; });
    await flush();
    assert.strictEqual(done, true, "long text must resolve immediately");
    assert.strictEqual(el.textContent, long, "long text must render in full at once");
    assert.strictEqual(rafQueue.length, q0, "long text must not schedule any ticks");
  }

  // A3: stale run stops writing when a newer run starts  ← P3-3 regression
  {
    resetRaf();
    const el1 = new FakeEl();
    el1.isConnected = true;
    const el2 = new FakeEl();
    el2.isConnected = true;
    let done1 = false;
    typewriterEffect(el1, "aaa", 10).then(() => { done1 = true; });
    drain(1);
    assert.strictEqual(el1.textContent, "a", "first run typed 1 char");
    typewriterEffect(el2, "b", 10);  // supersedes run 1
    drain(10);                        // any leftover ticks of run 1
    assert.strictEqual(el1.textContent, "a",
      "stale run must not keep writing after a newer run started");
    await flush();
    assert.strictEqual(done1, true, "stale run must still settle its promise");
    assert.strictEqual(el2.textContent, "b", "new run types normally");
  }

  // A4: detached element stops the chain (chat cleared mid-animation)
  {
    resetRaf();
    const el = new FakeEl();
    el.isConnected = true;
    let done = false;
    typewriterEffect(el, "hello", 10).then(() => { done = true; });
    drain(2);
    assert.strictEqual(el.textContent, "he");
    el.remove();  // element removed from the document
    drain(10);
    assert.strictEqual(el.textContent, "he", "detached element must not keep receiving text");
    await flush();
    assert.strictEqual(done, true, "detached run must settle its promise");
  }

  // A5: scrolled-up timeline is NOT hijacked
  {
    resetRaf();
    const el = new FakeEl();
    el.isConnected = true;
    const timeline = new FakeEl();
    timeline.scrollTop = 100;  // scrolled up: remaining 300 >= 40
    el._timeline = timeline;
    typewriterEffect(el, "hello", 10);
    drain(5);
    assert.strictEqual(timeline.scrollTop, 100,
      "scrolled-up viewport must not be hijacked while typing");
  }

  // A6: empty text resolves immediately
  {
    resetRaf();
    const el = new FakeEl();
    el.isConnected = true;
    let done = false;
    typewriterEffect(el, "", 10).then(() => { done = true; });
    drain(1);  // empty text takes the normal path: first tick finds i === len
    await flush();
    assert.strictEqual(done, true, "empty text must resolve");
    assert.strictEqual(el.textContent, "", "empty text must not write anything");
  }

  console.log("PASS Part A: typewriterEffect (6 checks)");

  // ── Part B: reasoning card collapse toggle (P7-4) ──
  let cardEl = null;
  let titleEl = null;
  let bodyEl = null;

  globalThis.document = {
    getElementById() { return null; },
    createElement(tag) {
      const el = new FakeEl();
      if (tag === "pre") {
        bodyEl = el;
      } else if (!cardEl) {
        cardEl = el;  // first div = the card
      } else {
        titleEl = el;  // second div = the title
      }
      return el;
    },
    createTextNode(s) { return makeTextNode(s); },
  };

  // No querySelector override: the card is looked up through the real matcher,
  // so "does the production selector actually find its own card?" stays tested.
  function freshTimeline() {
    const tl = new FakeEl();
    tl.className = "agent-timeline";
    tl.scrollTop = 100;  // user scrolled up: remaining 300 >= 40
    globalThis._agentGetTimeline = () => tl;
    return tl;
  }

  // B1: default state — expanded, chevron ▾
  {
    cardEl = null; titleEl = null; bodyEl = null;
    globalThis._AGENT_REASONING_DEFAULT_COLLAPSED = false;
    const tl = freshTimeline();
    appendReasoning("abc", "main");
    assert.ok(cardEl, "card must be created on first append");
    assert.ok(titleEl, "title element must be created");
    assert.strictEqual(titleEl.textContent, "🧠 Reasoning ▾", "expanded title must show ▾");
    assert.notStrictEqual(bodyEl.style.display, "none", "body must be visible by default");
    assert.strictEqual(bodyEl.textContent, "abc");
    assert.strictEqual(tl.scrollTop, 100, "scrolled-up timeline must not be hijacked");
  }

  // B2: click collapses, click again expands
  {
    titleEl._listeners.click[0]();
    assert.strictEqual(bodyEl.style.display, "none", "first click must collapse the body");
    assert.strictEqual(titleEl.textContent, "🧠 Reasoning ▸", "collapsed title must show ▸");
    titleEl._listeners.click[0]();
    assert.strictEqual(bodyEl.style.display, "", "second click must expand the body");
    assert.strictEqual(titleEl.textContent, "🧠 Reasoning ▾", "expanded title must show ▾");
  }

  // B3: appends keep flowing while collapsed (user can reopen and see all)
  {
    titleEl._listeners.click[0]();  // collapse again
    assert.strictEqual(bodyEl.style.display, "none");
    appendReasoning("def", "main");
    assert.strictEqual(bodyEl.textContent, "abcdef", "append must work while collapsed");
    assert.strictEqual(bodyEl.style.display, "none", "append must not auto-expand");
    titleEl._listeners.click[0]();
    assert.strictEqual(bodyEl.textContent, "abcdef", "reopening must show accumulated text");
  }

  // B4: default-collapsed option → new card starts collapsed
  {
    globalThis._AGENT_REASONING_DEFAULT_COLLAPSED = true;
    cardEl = null; titleEl = null; bodyEl = null;
    freshTimeline();
    appendReasoning("xyz", "main");
    assert.strictEqual(bodyEl.style.display, "none", "default-collapsed card must start collapsed");
    assert.strictEqual(titleEl.textContent, "🧠 Reasoning ▸", "collapsed title must show ▸");
    globalThis._AGENT_REASONING_DEFAULT_COLLAPSED = false;
  }

  console.log("PASS Part B: reasoning card collapse toggle (4 checks)");
})().catch((err) => {
  console.error("HARNESS FAILURE:", err.message);
  process.exit(1);
});
