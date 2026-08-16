#!/usr/bin/env node
/**
 * Regression harness — G1: cap the LIVE #agent-timeline DOM at _TL_MAX_ENTRIES.
 *
 * The persistence layer (_tlSave) already sliced serialized history to
 * _TL_MAX_ENTRIES (300), but the live DOM grew unbounded across runs in a
 * session (see _agentResetForNewRun: timeline is intentionally NOT cleared
 * between runs). A long session with hundreds of tool/reasoning cards thus
 * accumulated DOM nodes + _agentListenerCleanup registry entries indefinitely,
 * diverging from what a reload restores (only 300) and leaking memory.
 *
 * Fix: _tlPruneLiveTimeline() removes the oldest entries (preserving the
 * timeline header, vacuuming orphaned leading dividers) and releases the
 * listener-registry entries of the pruned cards. It is wired into the single
 * MutationObserver in _agentInitTimelineCollapse so every append site is
 * covered without per-site hooks.
 *
 * The REAL function text is sliced out of agent-panel.js and executed against
 * a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_timeline_prune.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

// ── Slice REAL functions out of agent-panel.js (brace-balanced) ──
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

// Extract the consts the pruner depends on, verbatim from source.
function extractConst(name) {
  const re = new RegExp(`(const ${name} = [^;]+);`);
  const m = src.match(re);
  assert.ok(m, `const ${name} not found in agent-panel.js`);
  return m[1];
}

// Pull the module-level registry declaration so the compiled functions share
// one array instance (mirrors how the real module works).
const registryDecl = src.match(/const _agentListenerCleanup = (\[[^\]]*\]);/);
assert.ok(registryDecl, "_agentListenerCleanup declaration not found");

const consts = [
  extractConst("_AGENT_TIMELINE_ENTRY_SELECTOR"),
  extractConst("_TL_MAX_ENTRIES"),
  `const _agentListenerCleanup = ${registryDecl[1]}`,
].join("\n");

const fnNames = [
  "_agentAddListener",
  "_agentCleanupListeners",
  "_tlReleaseListeners",
  "_agentUpdateTimelineMeta",
  "_agentEnsureTimelineHeader",
  "_agentInitTimelineCollapse",
  "_tlPruneLiveTimeline",
  "_lsGet",
  "_lsSet",
  "_lsRemove",
];
const fnText = fnNames.map(sliceFunction).join("\n");

// ── Stub DOM / storage / observer ─────────────────────────────────────────
const registry = new Map(); // id -> element (mirrors document.getElementById)

class FakeEl {
  constructor(tag, id) {
    this.tag = tag;
    this.id = id || "";
    this._cls = "";
    this.textContent = "";
    this._children = [];
    this._listeners = {};
    this._attrs = {};
    this.parentNode = null;
  }
  get className() { return this._cls; }       // DOM-accurate: className & classList stay in sync
  set className(v) { this._cls = String(v || ""); }
  appendChild(el) {
    el.parentNode = this;
    this._children.push(el);
    if (el.id) registry.set(el.id, el);
    return el;
  }
  prepend(el) {
    el.parentNode = this;
    this._children.unshift(el);
    if (el.id) registry.set(el.id, el);
    return el;
  }
  remove() {
    if (this.parentNode) {
      const i = this.parentNode._children.indexOf(this);
      if (i >= 0) this.parentNode._children.splice(i, 1);
      this.parentNode = null;
    }
    if (this.id) registry.delete(this.id);
  }
  removeChild(el) { el.remove(); return el; }
  contains(target) {
    if (target === this) return true;
    const walk = (el) => {
      for (const ch of el._children) {
        if (ch === target || walk(ch)) return true;
      }
      return false;
    };
    return walk(this);
  }
  get children() { return this._children; }
  classList = {
    contains: (c) => this._cls.split(/\s+/).includes(c),
    add: (c) => { const s = new Set(this._cls.split(/\s+/).filter(Boolean)); s.add(c); this._cls = [...s].join(" "); },
    remove: (c) => { const s = new Set(this._cls.split(/\s+/).filter(Boolean)); s.delete(c); this._cls = [...s].join(" "); },
    toggle: (c, force) => {
      const s = new Set(this._cls.split(/\s+/).filter(Boolean));
      const want = force === undefined ? !s.has(c) : !!force;
      if (want) s.add(c); else s.delete(c);
      this._cls = [...s].join(" ");
      return want;
    },
  };
  addEventListener(type, fn) { this._listeners[type] = fn; }
  removeEventListener() {}
  setAttribute(k, v) { this._attrs[k] = String(v); }
  querySelector(sel) {
    const tokens = sel.replace(/^[.#]/, "").split(/[.#]/).map((t) => t.trim());
    const walk = (el) => {
      const mine = tokens.every((t) => el._classes.has(t) || el.id === t);
      if (mine) return el;
      for (const ch of el._children) { const f = walk(ch); if (f) return f; }
      return null;
    };
    for (const ch of this._children) { const f = walk(ch); if (f) return f; }
    return null;
  }
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
  observe(el) { observerEl = el; }
};
globalThis.state = { repoRoot: "/repo" };

// Boot-time stale-key sweep — real impl pulls in storage-prefix consts we don't
// need here; stub as a no-op (mirrors the collapse-test harness).
globalThis._tlSweepStale = () => {};
// F1 (pending tool card orphan): prune now also drops pruned cards from the
// _pendingToolCards map via this helper. Not compiled in this harness — stub
// so _tlPruneLiveTimeline keeps working (slice-harness free-variable rule).
globalThis._pendingToolCardsRemoveCard = () => {};

// Compile the real functions in global scope so free variables resolve
// against the stubs. Return handles to the ones we test directly.
const api = new Function(
  consts + "\n" + fnText + "\nreturn {" +
  "_agentListenerCleanup, _agentAddListener, _agentCleanupListeners, " +
  "_tlReleaseListeners, _tlPruneLiveTimeline, _agentInitTimelineCollapse, " +
  "_agentUpdateTimelineMeta, _AGENT_TIMELINE_ENTRY_SELECTOR, _TL_MAX_ENTRIES };"
)();

// ── Fixtures ──────────────────────────────────────────────────────────────
function makeTimeline() {
  registry.clear();
  storage.clear();
  observerCb = null;
  observerEl = null;
  api._agentListenerCleanup.length = 0;
  const tl = new FakeEl("div", "agent-timeline");
  registry.set("agent-timeline", tl);
  return tl;
}
function makeEntry(className, label) {
  const el = new FakeEl("div");
  el.className = className;
  el.dataset = { label };
  return el;
}
function entryCount(tl) {
  return tl.querySelectorAll(api._AGENT_TIMELINE_ENTRY_SELECTOR).length;
}

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── Scenario 1: prune caps over-cap timeline to exactly _TL_MAX_ENTRIES ──
check("prune trims over-cap timeline to exactly the cap, keeping newest", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();   // injects header as first child
  const cap = api._TL_MAX_ENTRIES;    // 300
  for (let i = 0; i < cap + 5; i++) tl.appendChild(makeEntry("agent-tool-card", `c${i}`));
  assert.equal(entryCount(tl), cap + 5, "fixture: 305 entries");
  api._tlPruneLiveTimeline();
  assert.equal(entryCount(tl), cap, "pruned to exactly cap");
  // Oldest 5 (c0..c4) removed; newest (c304) survives and is last.
  const survivors = tl.querySelectorAll(api._AGENT_TIMELINE_ENTRY_SELECTOR);
  assert.equal(survivors[survivors.length - 1].dataset.label, `c${cap + 4}`, "newest preserved");
  assert.equal(survivors[0].dataset.label, "c5", "oldest survivor is c5");
  // Header preserved as first child.
  assert.equal(tl._children[0].id, "agent-timeline-header", "header preserved at top");
});

// ── Scenario 2: prune is a no-op at/below cap ──
check("prune is a no-op when entry count ≤ cap", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  for (let i = 0; i < 100; i++) tl.appendChild(makeEntry("agent-chat-msg", `m${i}`));
  const before = entryCount(tl);
  api._tlPruneLiveTimeline();
  assert.equal(entryCount(tl), before, "unchanged when below cap");
  assert.equal(entryCount(tl), 100);
});

// ── Scenario 3: prune vacuums orphaned leading dividers, keeps inter-survivor ──
check("prune removes leading/orphaned dividers but preserves inter-survivor dividers", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  const cap = api._TL_MAX_ENTRIES;
  const half = cap / 2;            // 150
  // Layout (after header): [leadDiv, 4 doomed entries with a mid divider,
  //   survDivA (leads survivor group A), 150 survivors, INTER-survivor
  //   divider, 150 more survivors]. removeCount = 4.
  const leadDivider = makeEntry("agent-session-divider", "d-top");
  tl.appendChild(leadDivider);
  for (let i = 0; i < 2; i++) tl.appendChild(makeEntry("agent-tool-card", `doomA${i}`));
  const midDivider = makeEntry("agent-session-divider", "d-mid-doomed");
  tl.appendChild(midDivider);
  for (let i = 0; i < 2; i++) tl.appendChild(makeEntry("agent-tool-card", `doomB${i}`));
  const survDivA = makeEntry("agent-session-divider", "d-lead-survivors");
  tl.appendChild(survDivA);
  for (let i = 0; i < half; i++) tl.appendChild(makeEntry("agent-tool-card", `survA${i}`));
  const interSurvivorDivider = makeEntry("agent-session-divider", "d-between-survivors");
  tl.appendChild(interSurvivorDivider);
  for (let i = 0; i < half; i++) tl.appendChild(makeEntry("agent-tool-card", `survB${i}`));
  assert.equal(entryCount(tl), cap + 4, "fixture: cap+4 entries");
  api._tlPruneLiveTimeline();
  assert.equal(entryCount(tl), cap, "pruned to cap");
  const kids = tl._children;
  assert.equal(kids[0].id, "agent-timeline-header", "header still first");
  assert.equal(kids[1].dataset.label, "survA0", "first surviving entry is now first child after header (clean start)");
  assert.notEqual(kids.indexOf(interSurvivorDivider), -1, "inter-survivor divider RETAINED (separates two live groups)");
  assert.equal(kids.indexOf(leadDivider), -1, "top divider (above pruned) vacuumed");
  assert.equal(kids.indexOf(midDivider), -1, "mid divider (between doomed) vacuumed");
  assert.equal(kids.indexOf(survDivA), -1, "survivor-group-A divider vacuumed (orphaned once preceding session pruned)");
});

// ── Scenario 4: prune releases listener-registry entries of pruned cards ──
check("prune releases listener-registry entries for doomed cards, keeps survivors", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  const cap = api._TL_MAX_ENTRIES;
  // doomed card with a tracked listener on a descendant header
  const doomedCard = makeEntry("agent-tool-card", "doomed");
  const doomedHdr = new FakeEl("div"); doomedHdr.className = "agent-card-header";
  doomedCard.appendChild(doomedHdr);
  tl.appendChild(doomedCard);
  api._agentAddListener(doomedHdr, "click", () => {});
  // survivor card with a tracked listener
  for (let i = 0; i < cap; i++) tl.appendChild(makeEntry("agent-tool-card", `s${i}`));
  const survCard = makeEntry("agent-tool-card", "surv-last");
  const survHdr = new FakeEl("div"); survHdr.className = "agent-card-header";
  survCard.appendChild(survHdr);
  tl.appendChild(survCard);               // total cap+2
  api._agentAddListener(survHdr, "click", () => {});
  assert.equal(api._agentListenerCleanup.length, 2, "two tracked listeners before prune");
  api._tlPruneLiveTimeline();
  assert.equal(entryCount(tl), cap, "pruned to cap");
  assert.equal(api._agentListenerCleanup.length, 1, "only survivor listener retained");
  assert.strictEqual(api._agentListenerCleanup[0].el, survHdr, "retained listener is the survivor's");
});

// ── Scenario 5: _tlReleaseListeners directly drops matching + contained els ──
check("_tlReleaseListeners drops entries for node and its descendants only", () => {
  makeTimeline();
  api._agentListenerCleanup.length = 0;
  const card = new FakeEl("div");
  const hdr = new FakeEl("div"); card.appendChild(hdr);
  const other = new FakeEl("div");
  api._agentAddListener(hdr, "click", () => {});
  api._agentAddListener(other, "click", () => {});
  assert.equal(api._agentListenerCleanup.length, 2);
  api._tlReleaseListeners(card);   // hdr is inside card → dropped; other kept
  assert.equal(api._agentListenerCleanup.length, 1);
  assert.strictEqual(api._agentListenerCleanup[0].el, other);
});

// ── Scenario 6: MutationObserver wires prune (hooked, prune-first) ──
check("observer callback prunes over-cap timeline before updating meta", () => {
  const tl = makeTimeline();
  api._agentInitTimelineCollapse();
  assert.ok(observerEl === tl, "observer attached to timeline");
  const cap = api._TL_MAX_ENTRIES;
  for (let i = 0; i < cap + 3; i++) tl.appendChild(makeEntry("agent-tool-card", `x${i}`));
  assert.ok(typeof observerCb === "function", "observer callback captured");
  observerCb();   // simulate the microtask the browser would fire
  assert.equal(entryCount(tl), cap, "observer-driven prune trims to cap");
  const countEl = registry.get("agent-timeline-count");
  assert.equal(countEl.textContent, `${cap} entries`, "meta reflects post-prune count");
});

// ── Source gates ──────────────────────────────────────────────────────────
check("source gate: prune references the SSOT cap and entry selector", () => {
  const body = sliceFunction("_tlPruneLiveTimeline");
  assert.ok(body.includes("_TL_MAX_ENTRIES"), "prune must cap against _TL_MAX_ENTRIES");
  assert.ok(body.includes("_AGENT_TIMELINE_ENTRY_SELECTOR"), "prune must use the canonical entry selector");
  assert.ok(body.includes("_tlReleaseListeners"), "prune must release listeners on removal");
});

check("source gate: observer calls _tlPruneLiveTimeline before _agentUpdateTimelineMeta", () => {
  const m = src.match(/new MutationObserver\(\(\) => \{([\s\S]*?)\}\)\.observe\(timeline/);
  assert.ok(m, "MutationObserver callback bound to timeline not found");
  const obsBody = m[1];
  const pruneIdx = obsBody.indexOf("_tlPruneLiveTimeline()");
  const metaIdx = obsBody.indexOf("_agentUpdateTimelineMeta()");
  assert.ok(pruneIdx >= 0, "observer must call _tlPruneLiveTimeline");
  assert.ok(metaIdx >= 0, "observer must call _agentUpdateTimelineMeta");
  assert.ok(pruneIdx < metaIdx, "prune must run before meta (count reflects survivors)");
});

check("source gate: _tlReleaseListeners filters by node identity/containment", () => {
  const body = sliceFunction("_tlReleaseListeners");
  assert.ok(body.includes("node.contains(e.el)"), "must release contained-element listeners");
  assert.ok(body.includes("removeEventListener"), "must detach the listener");
});

console.log(`\nAll ${passed} checks passed.`);
