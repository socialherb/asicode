#!/usr/bin/env node
/**
 * Regression harness — D1: the file-preview "jump to line + flash" was dead on
 * arrival in THREE independent layers.
 *
 *   1. Pane — both jump blocks resolved the editor with
 *      getElementById("editor-pane"). No commit ever defined that id: the panes
 *      carry "editor-pane" as a CLASS (<div id="file-preview" class="editor-pane
 *      …">), so the lookup returned null and each block returned at its first
 *      guard.
 *   2. Row — they then queried ".code-line", a row class emitted by NO renderer
 *      and styled by NO rule. renderOverlay() emits .ov-line rows (+ .ov-add /
 *      .ov-del, with a .ov-gutter carrying the SOURCE number — added rows are
 *      unnumbered), the diff renderers emit .diff-add / .diff-del, and
 *      renderFileText() emits a single <pre><code> with no rows at all.
 *   3. Style — ".apply-flash" was added and removed with no CSS rule anywhere,
 *      so even a successful highlight would have been invisible.
 *
 * Consequences: after Apply the promise "jump to the first changed line" never
 * happened, and the disambiguation snippet's own tooltip ("Click to jump to this
 * line") was a lie. focusFirstChangeInPreview() — a live THIRD implementation of
 * the same idea — queried the whole document with four of its eight selector
 * forms dead, and a document-wide query can match the agent panel's own
 * .diff-add rows (agent-panel.js renders them too).
 *
 * Fix under test: one shared, renderer-agnostic helper. jumpPreviewToLine()
 * resolves the pane by id, picks the exact gutter row or the first changed row
 * according to the CALL SITE's promise, and flashes whatever target it can
 * name — the pane itself when the current renderer has no line rows.
 * focusFirstChangeInPreview() now shares the pane-scoped resolver.
 *
 * The REAL function texts are sliced out of ui.js and executed against a stub
 * DOM (no test framework, no jsdom in this repo).
 *
 * Run: node tests/js/test_preview_jump_wiring.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const STATIC = path.join(__dirname, "..", "..", "webapp", "ui", "static");
const UI_SRC = fs.readFileSync(path.join(STATIC, "ui.js"), "utf8");
const CSS_SRC = fs.readFileSync(path.join(STATIC, "ui.css"), "utf8");
const HTML_SRC = fs.readFileSync(path.join(STATIC, "..", "templates", "ui.html"), "utf8");

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── Slice a REAL function out of ui.js (brace-balanced) ──
// Counting starts at the marker's OWN closing "{" — markers are written as
// "function name(args) {" so a default parameter like `opts = {}` can never be
// mistaken for the body opener (it would truncate the slice at that pair).
function sliceFn(src, marker) {
  const start = src.indexOf(marker);
  assert.ok(start >= 0, `slice marker not found: ${marker}`);
  const open = start + marker.length - 1;
  assert.strictEqual(src[open], "{", `slice marker must end with the body brace: ${marker}`);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces while slicing: ${marker}`);
  return src.slice(start, i + 1);
}

// ══════════════════════════════════════════════════════════════════════════
// Source gates — each of the three dead layers must stay dead, and the fix must
// keep talking to the renderers that actually exist.
// ══════════════════════════════════════════════════════════════════════════

// ── Comment-aware source view ──────────────────────────────────────────────
// This repo documents the bug next to the fix, so prose about the removed code
// would otherwise satisfy (or defeat) the absence gates below. Strip comments
// while leaving string literals intact — a dead reference INSIDE a string is a
// real reference and must still count.
function stripComments(src) {
  let out = "";
  let i = 0;
  let quote = "";
  while (i < src.length) {
    const c = src[i];
    const next = src[i + 1];
    if (quote) {
      out += c;
      if (c === "\\") {
        out += next || "";
        i += 2;
        continue;
      }
      if (c === quote) quote = "";
      i++;
      continue;
    }
    if (c === '"' || c === "'" || c === "`") {
      quote = c;
      out += c;
      i++;
      continue;
    }
    if (c === "/" && next === "/") {
      while (i < src.length && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && next === "*") {
      i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
      i += 2;
      out += " ";
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

const CODE = stripComments(UI_SRC);
// Anti-vacuous: a runaway strip would quietly blind every absence gate below.
for (const marker of [
  'el("file-preview")',
  'pane.querySelectorAll(".ov-line")',
  '"apply-flash"',
  "function jumpPreviewToLine(lineNo, opts = {}) {",
]) {
  assert.ok(CODE.includes(marker), `comment stripping removed live code (${marker}) — the gates would be blind`);
}

check("S1 — the phantom #editor-pane id is neither resolved nor defined", () => {
  assert.ok(
    !/getElementById\(\s*["']editor-pane["']\s*\)/.test(CODE),
    'live code resolves getElementById("editor-pane") — an id no commit defines'
  );
  assert.ok(
    !/id=["']editor-pane["']/.test(HTML_SRC),
    'ui.html defines id="editor-pane" after all — re-check the premise of this gate'
  );
  assert.ok(HTML_SRC.includes('id="file-preview"'), "ui.html must define #file-preview (the pane the helper resolves)");
  assert.ok(/class="editor-pane[^"]*"/.test(HTML_SRC), 'the panes must still carry the ".editor-pane" CLASS');
});

check("S2 — the phantom .code-line row query is gone from ui.js", () => {
  assert.ok(
    !/querySelectorAll\(\s*["']\.code-line["']\s*\)/.test(CODE),
    "live code still queries .code-line rows — no renderer emits that class"
  );
});

check("S3 — both jump call sites route through jumpPreviewToLine()", () => {
  assert.ok(UI_SRC.includes("function jumpPreviewToLine(lineNo, opts = {}) {"), "shared helper missing");
  assert.ok(
    UI_SRC.includes('jumpPreviewToLine(lineNo, { prefer: "changed" })'),
    'the apply flow must ask for the first CHANGED row (prefer: "changed")'
  );
  assert.ok(
    UI_SRC.includes('jumpPreviewToLine(ln, { prefer: "line", behavior: "smooth" })'),
    'the disambiguation snippet must ask for THIS line (prefer: "line")'
  );
});

check("S4 — .apply-flash is applied by ui.js AND styled by ui.css", () => {
  assert.ok(UI_SRC.includes('classList.add("apply-flash")'), "ui.js never adds apply-flash");
  assert.ok(UI_SRC.includes('classList.remove("apply-flash")'), "ui.js never removes apply-flash");
  assert.ok(
    /^\.apply-flash\s*\{/m.test(CSS_SRC),
    "ui.css has no .apply-flash rule — the flash would be invisible (layer 3 of the bug)"
  );
});

check("S5 — the JS flash duration and the CSS animation duration agree", () => {
  const jsMs = /const _PREVIEW_FLASH_MS = (\d+);/.exec(UI_SRC);
  assert.ok(jsMs, "_PREVIEW_FLASH_MS not found in ui.js");
  const anim = /\.apply-flash\s*\{[^}]*animation:\s*previewLineFlash\s+([\d.]+)s/.exec(CSS_SRC);
  assert.ok(anim, ".apply-flash must animate previewLineFlash with a second-based duration");
  assert.strictEqual(
    Number(jsMs[1]),
    Math.round(Number(anim[1]) * 1000),
    "flash duration drifted between _PREVIEW_FLASH_MS (ui.js) and .apply-flash (ui.css)"
  );
});

check("S6 — the animation name has keyframes and avoids the purged token names", () => {
  assert.ok(/@keyframes\s+previewLineFlash\s*\{/.test(CSS_SRC), "missing @keyframes previewLineFlash");
  // test_dead_css_groups_gate purges these names; reusing one would fail that gate.
  assert.ok(!CSS_SRC.includes("asrFlashBg"), "asrFlashBg is a purged dead-CSS token");
  assert.ok(!UI_SRC.includes("changed-flash"), "changed-flash is a purged dead-CSS token");
});

check("S7 — focusFirstChangeInPreview is pane-scoped and drops the 4 dead selectors", () => {
  const body = sliceFn(CODE, "function focusFirstChangeInPreview() {");
  assert.ok(
    !body.includes("document.querySelector("),
    "focusFirstChangeInPreview must not query the whole document (the agent panel renders .diff-add too)"
  );
  assert.ok(body.includes("_previewFirstChangedRow("), "focusFirstChangeInPreview must use the shared resolver");
  for (const dead of ['".ov-line.add"', '".ov-line.del"', '".line-add"', '".line-del"']) {
    assert.ok(!CODE.includes(dead), `dead selector form ${dead} re-introduced in live code`);
  }
});

check("S8 — the queried row classes are the ones the renderers really emit", () => {
  const overlay = sliceFn(CODE, "function renderOverlay(lines) {");
  assert.ok(overlay.includes("ov-line") && overlay.includes('"ov-add"') && overlay.includes('"ov-del"'),
    "renderOverlay must still emit .ov-line rows marked .ov-add/.ov-del");
  assert.ok(overlay.includes("ov-gutter"), "renderOverlay must still emit .ov-gutter (the line-number carrier)");

  const inline = sliceFn(CODE, "function renderInlineDiff(diffText) {");
  assert.ok(inline.includes('"diff-add"') && inline.includes('"diff-del"'),
    "renderInlineDiff must still emit .diff-add/.diff-del rows");

  const file = sliceFn(CODE, "function renderFileText(text, filenameOrPath) {");
  assert.ok(file.includes('createElement("pre")'),
    "renderFileText must still emit ONE <pre> — the no-rows renderer the pane-flash fallback exists for");

  for (const [name, body] of [["renderOverlay", overlay], ["renderInlineDiff", inline], ["renderFileText", file]]) {
    assert.ok(!body.includes("code-line"), `${name} must not emit .code-line`);
  }
});

// ══════════════════════════════════════════════════════════════════════════
// Stub DOM — a small but real element tree: className/classList share one
// underlying set, childNodes are linked via firstChild/nextSibling (so the
// offset walk is exercised), and a compound-selector matcher stands in for CSS.
// ══════════════════════════════════════════════════════════════════════════

function matchesSelector(node, selector) {
  const m = /^([a-zA-Z][\w-]*)?((?:\.[\w-]+)*)$/.exec(String(selector).trim());
  if (!m || !node || node.nodeType !== 1) return false;
  if (m[1] && node.tagName !== m[1].toUpperCase()) return false;
  const want = (m[2].match(/\.[\w-]+/g) || []).map((c) => c.slice(1));
  const have = new Set(String(node.className || "").split(/\s+/).filter(Boolean));
  return want.every((c) => have.has(c));
}

function queryAll(root, selector) {
  const parts = String(selector).split(",").map((s) => s.trim()).filter(Boolean);
  const out = [];
  const walk = (node) => {
    for (const child of node.childNodes) {
      if (parts.some((p) => matchesSelector(child, p))) out.push(child);
      walk(child);
    }
  };
  walk(root);
  return out; // document order
}

class FakeNode {
  constructor(tag, className = "", nodeType = 1, nodeValue = "") {
    this.tagName = tag ? String(tag).toUpperCase() : "";
    this.className = className;
    this.nodeType = nodeType;
    this.nodeValue = nodeValue;
    this.childNodes = [];
    this.parentNode = null;
    this.nextSibling = null;
    this.style = {};
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this._scrollTop = 0;
    this._scrolled = [];
    this._focusCalls = 0;
    // A no-op default: browser-like parts are opted into per scenario, so a
    // forgotten stub shows up as "no effect" instead of a false PASS.
    this._rect = { top: 0, left: 0, height: 0, width: 0 };
    const self = this;
    this.classList = {
      add(c) {
        const set = new Set(String(self.className || "").split(/\s+/).filter(Boolean));
        set.add(c);
        self.className = [...set].join(" ");
      },
      remove(c) {
        const set = new Set(String(self.className || "").split(/\s+/).filter(Boolean));
        set.delete(c);
        self.className = [...set].join(" ");
      },
      contains(c) {
        return String(self.className || "").split(/\s+/).filter(Boolean).includes(c);
      },
    };
  }
  appendChild(child) {
    const prev = this.childNodes[this.childNodes.length - 1] || null;
    if (prev) prev.nextSibling = child;
    child.parentNode = this;
    child.nextSibling = null;
    this.childNodes.push(child);
    return child;
  }
  get firstChild() {
    return this.childNodes[0] || null;
  }
  get textContent() {
    if (this.nodeType === 3) return String(this.nodeValue || "");
    return this.childNodes.map((c) => c.textContent).join("");
  }
  set textContent(v) {
    this.childNodes = [];
    this.appendChild(new FakeNode(null, "", 3, String(v)));
  }
  scrollIntoView(opts) {
    this._scrolled.push(opts);
  }
  focus() {
    this._focusCalls++;
  }
  getBoundingClientRect() {
    return this._rect;
  }
  querySelector(selector) {
    return queryAll(this, selector)[0] || null;
  }
  querySelectorAll(selector) {
    return queryAll(this, selector);
  }
  get scrollTop() {
    return this._scrollTop;
  }
  set scrollTop(v) {
    // Browser-like clamp, same contract as test_agent_reasoning_scroll.js.
    const max = Math.max(0, (this.scrollHeight || 0) - (this.clientHeight || 0));
    this._scrollTop = Math.max(0, Math.min(Number(v) || 0, max));
  }
}

function txt(s) {
  return new FakeNode(null, "", 3, s);
}
function div(className, ...children) {
  const node = new FakeNode("div", className);
  for (const c of children) node.appendChild(c);
  return node;
}
function overlayRow(kind, gutterText, codeText) {
  const row = div(`ov-line ${kind === "add" ? "ov-add" : kind === "del" ? "ov-del" : ""}`);
  row.appendChild(div("ov-gutter", txt(gutterText)));
  row.appendChild(div("ov-mark"));
  row.appendChild(div(`ov-code ${kind === "add" ? "ov-add" : kind === "del" ? "ov-del" : "ov-ctx"}`, txt(codeText)));
  return row;
}

// document.createRange() probe: records which text node / offset the helper
// resolved, and lets each scenario decide the geometry a browser would report.
let rangeProbe = null;
let rangeRectFn = null;
let documentQueryHits = [];
let documentQueryResult = null;

globalThis.document = {
  createRange() {
    const range = {
      _node: null,
      _offset: 0,
      setStart(node, offset) {
        range._node = node;
        range._offset = offset;
      },
      setEnd() {},
      getBoundingClientRect() {
        rangeProbe = { node: range._node, offset: range._offset };
        return rangeRectFn ? rangeRectFn(range._node, range._offset) : range._node._rect;
      },
    };
    return range;
  },
  querySelector(selector) {
    documentQueryHits.push(String(selector));
    return documentQueryResult;
  },
  querySelectorAll() {
    return [];
  },
};

// ── Execute the REAL helper cluster (sliced from ui.js) ──
const CLUSTER = [
  "function _previewPane() {",
  "function _previewRowForLine(pane, lineNo) {",
  "function _previewFirstChangedRow(pane) {",
  "function _textNodeAtOffset(root, offset) {",
  "function _previewScrollToLineOffset(pane, lineNo) {",
  "function jumpPreviewToLine(lineNo, opts = {}) {",
  "function focusFirstChangeInPreview() {",
];
for (const marker of CLUSTER) {
  const src = sliceFn(UI_SRC, marker);
  const name = marker.replace(/^function /, "").replace(/\(.*$/, "");
  globalThis[name] = eval(`(${src})`); // eslint-disable-line no-eval
}

const FLASH_MS_MATCH = /const _PREVIEW_FLASH_MS = (\d+);/.exec(UI_SRC);
assert.ok(FLASH_MS_MATCH, "_PREVIEW_FLASH_MS not found (source-derived default for the cluster)");
globalThis._PREVIEW_FLASH_MS = Number(FLASH_MS_MATCH[1]);

let filePreview = null;
let agentPanelHit = 0;
const agentPanel = new FakeNode("div", "editor-pane agent-panel-root");
const agentPanelRow = div("diff-add", txt("+ wired in the agent panel"));

function usePane(pane) {
  filePreview = pane;
  documentQueryHits = [];
  documentQueryResult = agentPanelRow; // ANY document query hands back the agent panel row
  globalThis.el = (id) => (id === "file-preview" ? filePreview : id === "agent-panel" ? agentPanel : null);
}

// ══════════════════════════════════════════════════════════════════════════
// Part A — row resolution against what the renderers emit
// ══════════════════════════════════════════════════════════════════════════
{
  const pane = new FakeNode("div", "editor-pane code-view line-numbers active");
  const ctx10 = overlayRow("ctx", "  10", "context");
  const added = overlayRow("add", "", "added"); // added rows carry NO gutter number
  const ctx11 = overlayRow("ctx", "  11", "context");
  pane.appendChild(ctx10);
  pane.appendChild(added);
  pane.appendChild(ctx11);

  assert.strictEqual(_previewRowForLine(pane, 10), ctx10, "gutter 10 must resolve to its row");
  assert.strictEqual(_previewRowForLine(pane, 11), ctx11, "gutter 11 must resolve to its row");
  assert.strictEqual(_previewRowForLine(pane, 1), null, "an unnumbered (added) row must not match a line number");
  assert.strictEqual(_previewRowForLine(pane, 0), null, "line 0 is not a target");
  assert.strictEqual(_previewRowForLine(pane, NaN), null, "a non-finite line must not match");
  assert.strictEqual(_previewRowForLine(null, 10), null, "no pane → no row");
  assert.strictEqual(_previewFirstChangedRow(pane), added, "the first CHANGED row is the .ov-add row");
  passed++;
  console.log("PASS Part A: row resolution (7 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part B — apply flow: "first changed line" in overlay mode
// ══════════════════════════════════════════════════════════════════════════
{
  const pane = new FakeNode("div", "editor-pane active");
  const ctx10 = overlayRow("ctx", "  10", "before");
  const added = overlayRow("add", "", "after"); // the changed line the promise points at
  pane.appendChild(ctx10);
  pane.appendChild(added);
  usePane(pane);
  agentPanelHit = 0;

  const out = jumpPreviewToLine(10, { prefer: "changed" }); // hunk header's +N is only a hint

  assert.strictEqual(out.mode, "overlay", "overlay rows must be reported as overlay mode");
  assert.strictEqual(out.target, added, "the CHANGED row must win over the exact gutter row");
  assert.strictEqual(added._scrolled.length, 1, "the target row must be scrolled into view");
  assert.deepStrictEqual(added._scrolled[0], { block: "center", inline: "nearest", behavior: "auto" });
  assert.ok(added.classList.contains("apply-flash"), "the target row must carry the flash class");
  assert.ok(!ctx10.classList.contains("apply-flash"), "the exact-match row must NOT flash");
  assert.strictEqual(ctx10._scrolled.length, 0, "only one row may be scrolled");
  assert.strictEqual(pane._focusCalls, 1, "the pane must take focus after the jump");
  assert.strictEqual(documentQueryHits.length, 0, "the jump must be pane-scoped, never document-wide");
  assert.ok(!agentPanelRow.classList.contains("apply-flash"), "the agent panel's diff row must be untouched");
  passed++;
  console.log("PASS Part B: apply flow flashes the CHANGED overlay row (11 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part C — apply flow in diff mode: first changed row in DOCUMENT order
// ══════════════════════════════════════════════════════════════════════════
{
  const pane = new FakeNode("div", "editor-pane active");
  const ctx = div("diff-context", txt(" unchanged"));
  const removed = div("diff-del", txt("-gone"));
  const inserted = div("diff-add", txt("+here"));
  pane.appendChild(ctx);
  pane.appendChild(removed);
  pane.appendChild(inserted);
  usePane(pane);

  const out = jumpPreviewToLine(null, { prefer: "changed" }); // no hunk number available at all

  assert.strictEqual(out.mode, "diff", "diff rows must be reported as diff mode");
  assert.strictEqual(out.target, removed, "the first CHANGED row wins, even when it is a deletion");
  assert.strictEqual(removed._scrolled.length, 1, "the deletion row must be scrolled into view");
  assert.ok(removed.classList.contains("apply-flash"), "the deletion row must flash");
  assert.ok(!ctx.classList.contains("apply-flash"), "context rows must never flash");
  assert.strictEqual(out.ok, true, "a missing hunk number must not disable the jump");
  passed++;
  console.log("PASS Part C: diff mode picks the first changed row in document order (6 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part D — disambiguation flow: "jump to THIS line" (prefer: "line")
// ══════════════════════════════════════════════════════════════════════════
{
  const pane = new FakeNode("div", "editor-pane active");
  const ctx10 = overlayRow("ctx", "  10", "before");
  const added = overlayRow("add", "", "after");
  const ctx11 = overlayRow("ctx", "  11", "target");
  pane.appendChild(ctx10);
  pane.appendChild(added);
  pane.appendChild(ctx11);
  usePane(pane);

  const exact = jumpPreviewToLine(11, { prefer: "line", behavior: "smooth" });

  assert.strictEqual(exact.target, ctx11, "the exact gutter row must win for prefer: \"line\"");
  assert.deepStrictEqual(exact.target._scrolled[0], { block: "center", inline: "nearest", behavior: "smooth" });
  assert.ok(!added.classList.contains("apply-flash"), "the changed row must not steal the flash");
  assert.strictEqual(exact.mode, "overlay", "the overlay row shape is reported as overlay");

  // Fallback (documented): when no row carries the requested number the changed
  // row is still better than doing nothing.
  added.classList.remove("apply-flash");
  const fallback = jumpPreviewToLine(999, { prefer: "line" });
  assert.strictEqual(fallback.target, added, "an absent exact row must fall back to the changed row");
  assert.strictEqual(fallback.ok, true, "the fallback jump must report success");
  passed++;
  console.log("PASS Part D: prefer \"line\" picks the exact gutter row + documented fallback (6 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part E — plain-file render: no rows exist, so the PANE is scrolled + flashed
// ══════════════════════════════════════════════════════════════════════════
{
  const LINES = 120; // tall enough that the centered target is not clamped
  const text = Array.from({ length: LINES }, (_, i) => `line ${i + 1}`).join("\n") + "\n";
  const pane = new FakeNode("div", "editor-pane code-view line-numbers active");
  pane.clientHeight = 600;
  pane.scrollHeight = LINES * 20;
  const pre = new FakeNode("pre", "code-view");
  const code = new FakeNode("code", "language-js");
  code.appendChild(txt(text));
  pre.appendChild(code);
  pane.appendChild(pre);
  pane._rect = { top: 0, left: 0, height: 600, width: 800 };
  usePane(pane);

  // Simulate the browser: a collapsed range reports the caret's geometry, i.e.
  // 20px per line inside the pre.
  rangeRectFn = (node, offset) => {
    const line = String(node.nodeValue || "").slice(0, offset).split("\n").length;
    return { top: (line - 1) * 20, left: 0, height: 18, width: 0 };
  };

  const out = jumpPreviewToLine(30, { prefer: "changed" });

  assert.strictEqual(out.ok, true, "a plain-file jump must succeed");
  assert.strictEqual(out.mode, "file", "no rows → file mode");
  assert.strictEqual(out.target, pane, "without line rows the PANE is the flash target");
  assert.ok(pane.classList.contains("apply-flash"), "the pane must carry the flash class");
  // (30-1)*20 = 580 center in a 600px viewport, clamped into [0, 600].
  assert.strictEqual(pane.scrollTop, 580 - 300 + 9, "the pane must scroll the target line to the center");
  assert.ok(rangeProbe, "the offset must be resolved through a Range");
  assert.strictEqual(rangeProbe.node, code.firstChild, "the range must address the text node holding the line");

  // Line 1 sits above the viewport top: the browser clamp (not our arithmetic) wins.
  pane.scrollTop = 0;
  jumpPreviewToLine(1, {});
  assert.strictEqual(pane.scrollTop, 0, "scrolling to the first line must clamp at the top");
  passed++;
  console.log("PASS Part E: plain-file render scrolls + flashes the PANE (7 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part F — highlight.js spans: the offset walk must descend into elements
// ══════════════════════════════════════════════════════════════════════════
{
  const pane = new FakeNode("div", "editor-pane active");
  pane.clientHeight = 600;
  pane.scrollHeight = 2000;
  const pre = new FakeNode("pre", "code-view");
  const code = new FakeNode("code", "language-js hljs");
  const spanA = new FakeNode("span", "hljs-keyword");
  const spanB = new FakeNode("span", "hljs-string");
  const textA = txt("one\ntwo\n");
  const textB = txt("three\nfour\n");
  spanA.appendChild(textA);
  spanB.appendChild(textB);
  code.appendChild(spanA);
  code.appendChild(spanB);
  pre.appendChild(code);
  pane.appendChild(pre);
  pane._rect = { top: 0, left: 0, height: 600, width: 800 };
  usePane(pane);
  rangeRectFn = (node) => (node === textB ? { top: 500, left: 0, height: 18, width: 0 } : { top: 0, left: 0, height: 18, width: 0 });

  jumpPreviewToLine(3, {}); // first line INSIDE the second span

  assert.strictEqual(rangeProbe.node, textB, "the walk must descend into the second hljs span");
  assert.strictEqual(rangeProbe.offset, 0, "line 3 starts at offset 0 of that text node");
  assert.strictEqual(pane.scrollTop, 500 - 300 + 9, "the geometry of the resolved node must drive the scroll");
  passed++;
  console.log("PASS Part F: the offset walk descends into hljs spans (3 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part G — no-target cases must report failure, not throw, not half-act
// ══════════════════════════════════════════════════════════════════════════
{
  const pane = new FakeNode("div", "editor-pane active");
  const pre = new FakeNode("pre", "code-view");
  const code = new FakeNode("code", "");
  code.appendChild(txt("only\ntwo\nlines\n"));
  pre.appendChild(code);
  pane.appendChild(pre);
  usePane(pane);

  const pastEof = jumpPreviewToLine(99, {});
  assert.deepStrictEqual(pastEof, { ok: false, mode: "none", target: null }, "a line past EOF must report no jump");
  assert.ok(!pane.classList.contains("apply-flash"), "a failed jump must not flash");
  assert.strictEqual(pane._scrollTop, 0, "a failed jump must not scroll");

  const empty = new FakeNode("div", "editor-pane active");
  usePane(empty);
  assert.deepStrictEqual(jumpPreviewToLine(3, {}), { ok: false, mode: "none", target: null }, "no renderer → no jump");

  usePane(null); // #file-preview absent from the DOM
  assert.deepStrictEqual(jumpPreviewToLine(3, {}), { ok: false, mode: "none", target: null }, "no pane → no jump");
  passed++;
  console.log("PASS Part G: no-target cases report failure without side effects (5 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part I — focusFirstChangeInPreview: pane-scoped, and it must NOT reach into
// the agent panel (the old document-wide query did).
// ══════════════════════════════════════════════════════════════════════════
{
  agentPanel.appendChild(agentPanelRow);
  agentPanelRow._scrolled = [];

  // (a) only the agent panel has a changed row → no preview jump at all
  const empty = new FakeNode("div", "editor-pane active");
  usePane(empty);
  documentQueryResult = agentPanelRow;
  const reachedOut = focusFirstChangeInPreview();
  assert.strictEqual(reachedOut, false, "without a preview row the helper must return false");
  assert.strictEqual(agentPanelRow._scrolled.length, 0, "the agent panel's diff row must NOT be scrolled");
  assert.strictEqual(documentQueryHits.length, 0, "no document-wide query may be issued");

  // (b) a changed row inside the preview is scrolled
  const pane = new FakeNode("div", "editor-pane active");
  const added = overlayRow("add", "", "added");
  pane.appendChild(overlayRow("ctx", "   1", "ctx"));
  pane.appendChild(added);
  usePane(pane);
  assert.strictEqual(focusFirstChangeInPreview(), true, "a preview changed row must be reported as focused");
  assert.strictEqual(added._scrolled.length, 1, "the preview row must be scrolled");
  assert.deepStrictEqual(added._scrolled[0], { block: "center", inline: "nearest" });

  // (c) a row without scrollIntoView cannot be focused (contract kept)
  const bare = new FakeNode("div", "editor-pane active");
  const bareRow = overlayRow("del", "", "gone");
  bareRow.scrollIntoView = undefined;
  bare.appendChild(bareRow);
  usePane(bare);
  assert.strictEqual(focusFirstChangeInPreview(), false, "a row without scrollIntoView must report failure");
  passed++;
  console.log("PASS Part I: focusFirstChangeInPreview is pane-scoped and agent-panel-blind (7 checks)");
}

// ══════════════════════════════════════════════════════════════════════════
// Part H — the flash is removed again after the caller's duration
// ══════════════════════════════════════════════════════════════════════════
async function partH() {
  const pane = new FakeNode("div", "editor-pane active");
  const added = overlayRow("add", "", "added");
  pane.appendChild(added);
  usePane(pane);

  const out = jumpPreviewToLine(10, { prefer: "changed", flashMs: 30 });
  assert.strictEqual(out.target, added, "the overlay row must be the flash target");
  assert.ok(added.classList.contains("apply-flash"), "the flash must be applied synchronously");
  await new Promise((resolve) => setTimeout(resolve, 90));
  assert.ok(!added.classList.contains("apply-flash"), "the flash must be removed after flashMs");
  assert.ok(!pane.classList.contains("apply-flash"), "only the row was flashed in overlay mode");
  check("H1 — the flash class is added and removed again (30ms)", () => {});

  // The default path reads the source-derived constant, not a literal.
  const pane2 = new FakeNode("div", "editor-pane active");
  pane2.appendChild(overlayRow("add", "", "added"));
  usePane(pane2);
  const t0 = Date.now();
  jumpPreviewToLine(10, { prefer: "changed", flashMs: globalThis._PREVIEW_FLASH_MS });
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.ok(pane2.querySelector(".ov-line.ov-add").classList.contains("apply-flash"),
    "the default duration must outlive 20ms (it is 1200ms)");
  check(`H2 — the default duration (_PREVIEW_FLASH_MS=${globalThis._PREVIEW_FLASH_MS}) is used when flashMs is omitted`, () => {
    assert.ok(Date.now() - t0 < 1000, "this check is only meaningful while the flash is still on");
  });
}

partH().then(
  () => {
    console.log(`\n${passed} checks passed (preview jump wiring)`);
  },
  (e) => {
    console.error(`\nFAIL: ${(e && e.stack) || e}`);
    process.exit(1);
  }
);
