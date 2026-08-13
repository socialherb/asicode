#!/usr/bin/env node
/**
 * Regression harness — P7-5: agent timeline collapse option.
 *
 * Continuous chat mode grows #agent-timeline unbounded across runs
 * (intentional design), so the feature adds a header (label + entry count +
 * collapse toggle) whose state persists in localStorage. Collapsed state is
 * a class on the container; CSS hides every child except the header and the
 * live request echo. A MutationObserver keeps the entry count fresh while
 * streaming.
 *
 * The REAL function text is sliced out of agent-panel.js and executed
 * against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_timeline_collapse.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const cssPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.css");
const src = fs.readFileSync(srcPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");

// ── Slice REAL functions/constants out of agent-panel.js (brace-balanced) ──
function sliceFunction(name) {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found in agent-panel.js`);
  let depth = 0;
  let i = src.indexOf("{", start);
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces extracting ${name}`);
  return src.slice(start, i + 1);
}

const consts = [...src.matchAll(/const (_AGENT_TIMELINE_[A-Z_]+ = [^;]+);/g)]
  .map((m) => m[1])
  .join("\n");
assert.ok(consts.includes("asicode.agent.timeline_collapsed"), "timeline collapse key constant missing");

const fnNames = [
  "_lsGet",   // P9-2: timeline persistence now routes through the try-wrapped helpers
  "_lsSet",
  "_lsRemove",
  "_agentUpdateTimelineMeta",
  "_agentTimelineSetCollapsed",
  "_agentEnsureTimelineHeader",
  "_agentInitTimelineCollapse",
];
const fnText = fnNames.map(sliceFunction).join("\n");

// ── Stub DOM / storage / observer ─────────────────────────────────────────
const registry = new Map(); // id -> element (mirrors document.getElementById)

class FakeEl {
  constructor(tag, id) {
    this.tag = tag;
    this.id = id || "";
    this.className = "";
    this.textContent = "";
    this._children = [];
    this._classes = new Set();
    this._listeners = {};
    this._attrs = {};
  }
  appendChild(el) {
    this._children.push(el);
    if (el.id) registry.set(el.id, el);
    return el;
  }
  prepend(el) {
    this._children.unshift(el);
    if (el.id) registry.set(el.id, el);
    return el;
  }
  remove() {
    if (this.id) registry.delete(this.id);
  }
  classList = {
    contains: (c) => this._classes.has(c),
    add: (c) => this._classes.add(c),
    remove: (c) => this._classes.delete(c),
    toggle: (c, force) => {
      const want = force === undefined ? !this._classes.has(c) : !!force;
      if (want) this._classes.add(c);
      else this._classes.delete(c);
      return want;
    },
  };
  addEventListener(type, fn) { this._listeners[type] = fn; }
  setAttribute(k, v) { this._attrs[k] = String(v); }
  querySelectorAll(sel) {
    const tokens = sel.split(",").map((t) => t.trim().replace(/^\./, ""));
    const out = [];
    const walk = (el) => {
      const mine = el.className.split(/\s+/).some((c) => tokens.includes(c));
      if (mine) out.push(el);
      for (const ch of el._children) walk(ch);
    };
    for (const ch of this._children) walk(ch);
    return out;
  }
}

const documentStub = {
  getElementById: (id) => registry.get(id) || null,
  createElement: (tag) => new FakeEl(tag),
};
globalThis.document = documentStub;

const storage = new Map();
globalThis.localStorage = {
  getItem: (k) => (storage.has(k) ? storage.get(k) : null),
  setItem: (k, v) => storage.set(k, String(v)),
  removeItem: (k) => storage.delete(k),
};

let observerCb = null;
let observerEl = null;
globalThis.MutationObserver = class {
  constructor(cb) { observerCb = cb; }
  observe(el, opts) { observerEl = el; }
};

// Boot-time stale-key sweep (P9-5) — no-op stub; storage fixture is empty.
globalThis._tlSweepStale = () => {};

// Compile the real functions in global scope so free variables resolve
// against the stubs.
const api = new Function(
  consts + "\n" + fnText + "\nreturn {" + fnNames.join(", ") + "};"
)();

// ── Fixtures ──────────────────────────────────────────────────────────────
function makeTimeline() {
  registry.clear();
  storage.clear();
  observerCb = null;
  observerEl = null;
  const tl = new FakeEl("div", "agent-timeline");
  const echo = new FakeEl("div", "agent-request-echo");
  echo.className = "agent-request-echo";
  tl.appendChild(echo);
  registry.set("agent-timeline", tl);
  return tl;
}
function makeEntry(className) {
  const el = new FakeEl("div");
  el.className = className;
  return el;
}
function headerBtn() { return registry.get("agent-timeline-toggle"); }
function headerCount() { return registry.get("agent-timeline-count"); }

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── Scenario 1: init applies persisted collapsed state ──
check("init with persisted collapsed=1 injects header, collapses, labels Expand", () => {
  const tl = makeTimeline();
  storage.set("asicode.agent.timeline_collapsed", "1");
  api._agentInitTimelineCollapse();
  assert.ok(tl._children[0].id === "agent-timeline-header", "header must be first child");
  assert.ok(tl.classList.contains("agent-timeline--collapsed"), "collapsed class applied");
  assert.equal(headerBtn().textContent, "▸ Expand");
  assert.equal(headerBtn()._attrs["aria-expanded"], "false");
  assert.equal(headerCount().textContent, "");
});

// ── Scenario 2: init with no persistence stays expanded ──
check("init without persistence stays expanded with Collapse label", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  assert.ok(!tl.classList.contains("agent-timeline--collapsed"));
  assert.equal(headerBtn().textContent, "▾ Collapse");
  assert.equal(headerBtn()._attrs["aria-expanded"], "true");
});

// ── Scenario 3: click toggles collapsed on + persists ──
check("toggle click collapses and persists", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  headerBtn()._listeners.click();
  assert.ok(tl.classList.contains("agent-timeline--collapsed"), "class added after click");
  assert.equal(headerBtn().textContent, "▸ Expand");
  assert.equal(storage.get("asicode.agent.timeline_collapsed"), "1");
  // and back
  headerBtn()._listeners.click();
  assert.ok(!tl.classList.contains("agent-timeline--collapsed"), "class removed after second click");
  assert.equal(headerBtn().textContent, "▾ Collapse");
  assert.equal(storage.get("asicode.agent.timeline_collapsed"), "0");
});

// ── Scenario 4: entry count excludes header/echo/placeholder ──
check("entry count counts only chat msgs, tool cards, reasoning cards", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  const placeholder = new FakeEl("div");
  placeholder.className = "agent-timeline-placeholder";
  tl.appendChild(placeholder);
  tl.appendChild(makeEntry("agent-chat-msg"));
  tl.appendChild(makeEntry("agent-chat-msg agent-chat-msg--result"));
  tl.appendChild(makeEntry("agent-chat-msg agent-chat-msg--thinking"));
  tl.appendChild(makeEntry("agent-tool-card"));
  tl.appendChild(makeEntry("agent-tool-card"));
  tl.appendChild(makeEntry("agent-reasoning-card"));
  api._agentUpdateTimelineMeta();
  assert.equal(headerCount().textContent, "6 entries");
});

// ── Scenario 5: header re-injected after a clear wipes the timeline ──
check("clear wipe re-injects working header", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  // Simulate _agentClearPanel's timeline.innerHTML = "" (children + header gone)
  tl._children = [];
  registry.delete("agent-timeline-header");
  registry.delete("agent-timeline-count");
  registry.delete("agent-timeline-toggle");
  api._agentEnsureTimelineHeader();
  assert.ok(tl._children[0].id === "agent-timeline-header", "header re-created as first child");
  assert.ok(headerBtn(), "toggle button re-created");
  headerBtn()._listeners.click();
  assert.ok(tl.classList.contains("agent-timeline--collapsed"), "re-injected toggle works");
});

// ── Scenario 6: MutationObserver keeps count fresh while streaming ──
check("observer updates count on childList mutation", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  assert.ok(observerEl === tl, "observer attached to timeline");
  tl.appendChild(makeEntry("agent-tool-card"));
  observerCb();
  assert.equal(headerCount().textContent, "1 entries");
  tl.appendChild(makeEntry("agent-chat-msg"));
  observerCb();
  assert.equal(headerCount().textContent, "2 entries");
});

// ── Source gates ──────────────────────────────────────────────────────────
check("css gate: collapsed selector hides non-header children, keeps echo", () => {
  assert.ok(
    css.includes(".agent-timeline--collapsed > :not(.agent-timeline-header):not(.agent-request-echo)"),
    "collapse CSS rule missing"
  );
  assert.ok(css.includes(".agent-timeline-toggle"), "toggle button CSS missing");
});

check("source gate: clear panel re-injects header after wipe", () => {
  const start = src.indexOf("function _agentClearPanel(");
  const end = src.indexOf("\n}\n", start);
  const body = src.slice(start, end);
  const wipe = body.indexOf('innerHTML = ""');
  const reinject = body.indexOf("_agentEnsureTimelineHeader()");
  assert.ok(wipe >= 0 && reinject > wipe, "clear panel must re-inject header after wiping");
});

check("source gate: init wired on DOM ready", () => {
  assert.ok(
    src.includes('document.addEventListener("DOMContentLoaded", _agentInitTimelineCollapse)'),
    "init not wired"
  );
});

console.log(`\nAll ${passed} checks passed.`);
