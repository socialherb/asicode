#!/usr/bin/env node
/**
 * Regression harness — P11-3: plain Enter in the main prompt sends (Shift+Enter
 * = newline), IME-guarded, and design mode keeps its own Enter handling.
 *
 * Previously the main prompt-input only had Alt+Enter (smartRun) — Enter just
 * inserted a newline, while design mode had Enter=send with an IME guard in
 * design-chat.js. The two modes disagreed. Now the ui.js keydown handler:
 *   1. IME guard: isComposing / keyCode 229 → ignore (CJK composition must
 *      never fire a ghost run)
 *   2. Alt+Enter → smartRun (unchanged, works in both modes)
 *   3. plain Enter (no shift) → smartRun, UNLESS the design conversation layer
 *      is active — there the design-chat handler already dispatched
 *      (_designSendMessage), so running smartRun too would send the message
 *      twice.
 *
 * The REAL handler text is sliced out of ui.js (brace-balanced) and executed
 * against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_prompt_enter.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const src = fs.readFileSync(srcPath, "utf8");

// ── Slice the REAL keydown registration out of ui.js (brace-balanced) ──
const anchor = 'el("prompt-input")?.addEventListener("keydown"';
const start = src.indexOf(anchor);
assert.ok(start >= 0, `anchor ${anchor} not found in ui.js`);
let depth = 0;
let i = src.indexOf("{", start);
assert.ok(i >= 0, "no handler body found");
for (; i < src.length; i++) {
  if (src[i] === "{") depth++;
  else if (src[i] === "}") {
    depth--;
    if (depth === 0) break;
  }
}
assert.ok(i < src.length, "unbalanced braces while extracting keydown handler");
// The handler's closing brace ends the arrow body; the registration statement
// still needs its trailing `)` and `;` to be valid expression syntax.
const end = src.indexOf(";", i);
assert.ok(end >= 0, "statement terminator not found after handler");
const handlerText = src.slice(start, end + 1);

// ── Stub environment (global scope — new Function compiles there) ──
const listeners = {};
let designMode = false;
let smartRunCalls = 0;

globalThis.el = () => ({
  addEventListener: (name, fn) => { listeners[name] = fn; },
});
globalThis.document = {
  getElementById: (id) => (id === "conv-layer-toggle" ? { checked: designMode } : null),
};
globalThis.smartRun = () => { smartRunCalls++; };
// Slash-command menu state (ui.js): closed by default — the handler's menu
// branch reads `_slashMenuItems.length`, which would be a ReferenceError
// without this binding (the handler is compiled in global scope).
globalThis._slashMenuItems = [];
globalThis._slashMoveSel = () => {};
globalThis._slashAcceptSel = () => {};
globalThis._slashCloseMenu = () => {};

new Function(handlerText)();  // registers the keydown handler against the stubs

const kd = listeners.keydown;
assert.ok(typeof kd === "function", "keydown handler not registered");

// ── Checks ──
let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`  ok - ${name}`);
}

function fire(overrides) {
  const e = {
    isComposing: false,
    keyCode: 0,
    key: "Enter",
    altKey: false,
    shiftKey: false,
    _prevented: false,
    preventDefault() { this._prevented = true; },
    ...overrides,
  };
  kd(e);
  return e;
}

function reset() {
  smartRunCalls = 0;
  designMode = false;
}

check("plain Enter sends in main mode (preventDefault + smartRun)", () => {
  reset();
  const e = fire({});
  assert.strictEqual(e._prevented, true);
  assert.strictEqual(smartRunCalls, 1);
});

check("Shift+Enter is a newline (no send, no preventDefault)", () => {
  reset();
  const e = fire({ shiftKey: true });
  assert.strictEqual(e._prevented, false);
  assert.strictEqual(smartRunCalls, 0);
});

check("IME composing never fires a run", () => {
  reset();
  const e = fire({ isComposing: true });
  assert.strictEqual(smartRunCalls, 0);
});

check("keyCode 229 (legacy IME) never fires a run", () => {
  reset();
  const e = fire({ keyCode: 229 });
  assert.strictEqual(smartRunCalls, 0);
});

check("Alt+Enter still sends in main mode", () => {
  reset();
  const e = fire({ altKey: true });
  assert.strictEqual(e._prevented, true);
  assert.strictEqual(smartRunCalls, 1);
});

check("design mode: plain Enter does NOT smartRun (design-chat owns it)", () => {
  reset();
  designMode = true;
  const e = fire({});
  assert.strictEqual(smartRunCalls, 0, "double send: ui.js + design-chat.js both dispatched");
});

check("design mode: Alt+Enter still routes through smartRun", () => {
  reset();
  designMode = true;
  const e = fire({ altKey: true });
  assert.strictEqual(smartRunCalls, 1);
});

check("non-Enter keys are untouched", () => {
  reset();
  const e = fire({ key: "a" });
  assert.strictEqual(e._prevented, false);
  assert.strictEqual(smartRunCalls, 0);
});

console.log(`\nPASS — ${passed} checks`);
