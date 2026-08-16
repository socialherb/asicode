#!/usr/bin/env node
/**
 * Regression harness — P11-1: repo switching must not destroy timelines.
 *
 * The old repo-switch path (ui.js set-repo-btn handler + pickRepoRoot) did:
 *   state.repoRoot = <NEW>;        // state flips FIRST
 *   _agentClearPanel();            // purge removes _tlKey(<NEW>)  ← destroys
 *   _tlLoad(state.repoRoot);       // reads the just-deleted key → always null
 * so the incoming repo's saved history was permanently deleted and the
 * restore feature never fired. The outgoing repo was never saved either.
 *
 * The fix: _tlSave() (outgoing) → state.repoRoot = next → _agentClearPanel
 * ({ purgeStorage: false }) (DOM-only) → _tlLoad(). _agentClearPanel gained
 * an opts param: default purge=true (clear button semantics preserved),
 * purgeStorage:false skips the localStorage delete.
 *
 * The REAL function text is sliced out of agent-panel.js / ui.js and executed
 * against a stub DOM + stub storage (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_timeline_roundtrip.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const apPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const uiPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const apSrc = fs.readFileSync(apPath, "utf8");
const uiSrc = fs.readFileSync(uiPath, "utf8");

// ── Slice REAL functions/constants out of the sources (brace-balanced) ──
function sliceFunction(name) {
  const start = apSrc.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found in agent-panel.js`);
  let depth = 0;
  let i = apSrc.indexOf("{", start);
  for (; i < apSrc.length; i++) {
    if (apSrc[i] === "{") depth++;
    else if (apSrc[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < apSrc.length, `unbalanced braces extracting ${name}`);
  return apSrc.slice(start, i + 1);
}

function sliceUiStatement(needle) {
  const start = uiSrc.indexOf(needle);
  assert.ok(start >= 0, `statement not found: ${needle}`);
  let depth = 0;
  let i = uiSrc.indexOf("{", start);
  for (; i < uiSrc.length; i++) {
    if (uiSrc[i] === "{") depth++;
    else if (uiSrc[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < uiSrc.length, `unbalanced braces extracting ${needle}`);
  const end = uiSrc.indexOf(");", i);
  assert.ok(end > i && end - i < 5, `expected '});' after handler body (${needle})`);
  return uiSrc.slice(start, end + 2);
}

const consts = [...apSrc.matchAll(/const (_TL_[A-Z_]+)\s*=\s*([^;]+);/g)]
  .map((m) => `const ${m[1]} = ${m[2]};`)
  .join("\n");
assert.ok(consts.includes("asicode.timeline.v2"), "_TL_STORAGE_KEY_PREFIX constant missing");

const fnNames = [
  "_lsGet",   // P9-2: timeline persistence routes through the try-wrapped helpers
  "_lsSet",
  "_lsRemove",
  "_lsKeys",  // P9-5: stale-key sweep iterates storage keys
  "_escHtml",
  "_renderMd",
  "_tlKey",
  "_tlSerialize",
  "_tlSave",
  "_tlLoad",
  "_tlRestore",
  "_tlSweepStale",
  "_agentClearPanel",
];
const fnText = fnNames.map(sliceFunction).join("\n");
const uiSlices = [
  sliceUiStatement('el("set-repo-btn")?.addEventListener("click", () => {'),
  uiSrc.slice(uiSrc.indexOf("async function pickRepoRoot() {"), (() => {
    const start = uiSrc.indexOf("async function pickRepoRoot() {");
    let depth = 0;
    let i = uiSrc.indexOf("{", start);
    for (; i < uiSrc.length; i++) {
      if (uiSrc[i] === "{") depth++;
      else if (uiSrc[i] === "}") {
        depth--;
        if (depth === 0) break;
      }
    }
    return i + 1;
  })()),
].join("\n");

// ── Stub DOM / storage ─────────────────────────────────────────────────────
const registry = new Map(); // id -> element (mirrors document.getElementById)

class FakeEl {
  constructor(tag, id) {
    this.tag = tag;
    this.id = id || "";
    this.textContent = "";
    this._children = [];
    this._classes = new Set();
    this._listeners = {};
    this._attrs = {};
    this._innerHTML = "";
  }
  get className() { return [...this._classes].join(" "); }
  set className(v) { this._classes = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get children() { return this._children; }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(v) {
    if (v === "") this._children = []; // clearing innerHTML drops children
    this._innerHTML = String(v);
  }
  appendChild(el) { this._children.push(el); if (el.id) registry.set(el.id, el); return el; }
  prepend(el) { this._children.unshift(el); if (el.id) registry.set(el.id, el); return el; }
  remove() { if (this.id) registry.delete(this.id); }
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
  querySelector(sel) {
    const token = sel.replace(/^\./, "");
    if (this._classes.has(token)) return this;
    for (const ch of this._children) {
      const r = ch.querySelector ? ch.querySelector(sel) : null;
      if (r) return r;
    }
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
  querySelector: () => null,
};
globalThis.document = documentStub;

const storage = new Map();
globalThis.localStorage = {
  getItem: (k) => (storage.has(k) ? storage.get(k) : null),
  setItem: (k, v) => { storage.set(k, String(v)); },
  removeItem: (k) => { storage.delete(k); },
  get length() { return storage.size; },
  key: (i) => [...storage.keys()][i] ?? null,
};

// Global app state + UI shell (created BEFORE compilation so the set-repo-btn
// listener registration inside the compiled statement finds its elements).
globalThis.state = { repoRoot: "" };
const input = new FakeEl("input", "repo-root");
const btn = new FakeEl("button", "set-repo-btn");
registry.set("repo-root", input);
registry.set("set-repo-btn", btn);

// Free-variable stubs referenced by the compiled functions.
globalThis.el = (id) => registry.get(id) || null;
globalThis.addTL = () => {};
globalThis.refreshFileTree = () => {};
globalThis.refreshGitPanelsSafe = () => {};
globalThis.apiPost = async () => ({ ok: true, repo_root: "/repo/B" });
globalThis._designClearChat = () => { designClears++; };
globalThis._designRestoreHistory = () => {};
globalThis.finalizeProgressRows = () => {};
globalThis._agentCleanupListeners = () => {};
globalThis._agentResetTabs = () => {};
globalThis._agentEnsureTimelineHeader = () => {};
globalThis._agentScrollBottom = () => {};
globalThis._pipelineReset = () => {};
globalThis._agentSetStatus = () => {};
// F1 (pending tool card orphan): _agentClearPanel now clears the pending map.
// The real declaration lives at module scope in agent-panel.js and is not
// compiled into this harness — provide an isolated Map (slice-harness
// free-variable rule).
globalThis._pendingToolCards = new Map();

let designClears = 0;

// Compile the real functions + ui.js handler statements in one global scope.
const api = new Function(
  consts + "\n" + fnText + "\n" + uiSlices + "\nreturn {" + fnNames.concat(["pickRepoRoot"]).join(", ") + "};"
)();

// ── Fixtures ──────────────────────────────────────────────────────────────
function makeTimeline() {
  storage.clear();
  designClears = 0;
  state.repoRoot = "";
  const tl = new FakeEl("div", "agent-timeline");
  registry.set("agent-timeline", tl);
  return tl;
}
function userMsg(text) {
  const el = new FakeEl("div");
  el.className = "agent-chat-msg agent-chat-msg--user";
  el.textContent = text;
  return el;
}
function toolCard(toolname) {
  const card = new FakeEl("div");
  card.className = "agent-tool-card ok";
  const hdr = new FakeEl("div");
  hdr.className = "agent-card-header";
  const tn = new FakeEl("code");
  tn.className = "agent-card-toolname";
  tn.textContent = toolname;
  hdr.appendChild(tn);
  card.appendChild(hdr);
  return card;
}
function savedPayload(repoRoot, texts) {
  return JSON.stringify({ v: 2, ts: Date.now(), repoRoot,
    entries: texts.map((t) => ({ type: "user", text: t })) });
}
function clickRepoSwitch(value) {
  input.value = value;
  btn._listeners.click();
}

let passed = 0;
async function check(name, fn) { await fn(); passed++; console.log(`PASS: ${name}`); }

(async () => {
  // ── Scenario 1: A→B switch saves outgoing, preserves incoming, restores B ──
  await check("A→B switch: outgoing saved, incoming key preserved, B restored", () => {
    const tl = makeTimeline();
    state.repoRoot = "/repo/A";
    tl.appendChild(userMsg("hello A"));
    tl.appendChild(toolCard("read_file"));
    const aKey = api._tlKey("/repo/A");
    const bKey = api._tlKey("/repo/B");
    storage.set(bKey, savedPayload("/repo/B", ["B-history"]));

    clickRepoSwitch("/repo/B");

    assert.ok(storage.has(aKey), "outgoing A timeline must be saved on switch");
    const savedA = JSON.parse(storage.get(aKey));
    assert.equal(savedA.entries.length, 2, "A's live DOM entries persisted");
    assert.equal(savedA.entries[1].type, "tool");
    assert.equal(savedA.entries[1].tool, "read_file");
    assert.ok(storage.has(bKey), "incoming B key must NOT be purged");
    assert.equal(state.repoRoot, "/repo/B", "state flipped to new repo");
    const restored = tl._children.some(
      (c) => c.className.includes("agent-chat-msg--user") && c.textContent === "B-history");
    assert.ok(restored, "B's history restored into timeline");
    const banner = tl._children.some((c) => c.className.includes("agent-session-restore-banner"));
    assert.ok(banner, "restore banner present");
  });

  // ── Scenario 2: B→A roundtrip (continuation of scenario 1's state) ──
  await check("B→A roundtrip: both keys survive, A restored", () => {
    const tl = registry.get("agent-timeline");
    const aKey = api._tlKey("/repo/A");
    const bKey = api._tlKey("/repo/B");

    clickRepoSwitch("/repo/A");

    assert.ok(storage.has(aKey), "A key survives the roundtrip");
    assert.ok(storage.has(bKey), "B key saved on exit (overwritten with B's DOM)");
    const restored = tl._children.some(
      (c) => c.className.includes("agent-chat-msg--user") && c.textContent === "hello A");
    assert.ok(restored, "A's history restored on return");
    assert.equal(state.repoRoot, "/repo/A");
  });

  // ── Scenario 3: same-repo click is a no-op for storage/DOM ──
  await check("same-repo click: no save, no purge, no restore", () => {
    const tl = makeTimeline();
    state.repoRoot = "/repo/A";
    tl.appendChild(userMsg("hello A"));
    const aKey = api._tlKey("/repo/A");
    storage.set(aKey, savedPayload("/repo/A", ["SAVED-A"]));

    clickRepoSwitch("/repo/A");

    assert.equal(state.repoRoot, "/repo/A");
    const kept = JSON.parse(storage.get(aKey));
    assert.equal(kept.entries[0].text, "SAVED-A", "stored payload untouched");
    assert.ok(!tl._children.some((c) => c.textContent === "SAVED-A"), "no restore on same-repo click");
    assert.ok(!tl._children.some((c) => c.className.includes("agent-session-restore-banner")),
      "no restore banner on same-repo click");
  });

  // ── Scenario 4/5: _agentClearPanel purge semantics ──
  await check("_agentClearPanel() default purges the current repo key", () => {
    const tl = makeTimeline();
    state.repoRoot = "/repo/A";
    const aKey = api._tlKey("/repo/A");
    storage.set(aKey, "x");

    api._agentClearPanel();

    assert.ok(!storage.has(aKey), "default purge removes the key (clear-button semantics)");
  });

  await check("_agentClearPanel({purgeStorage:false}) keeps the key", () => {
    const tl = makeTimeline();
    state.repoRoot = "/repo/A";
    const aKey = api._tlKey("/repo/A");
    storage.set(aKey, "x");

    api._agentClearPanel({ purgeStorage: false });

    assert.ok(storage.has(aKey), "purgeStorage:false preserves the key (repo-switch path)");
  });

  // ── Scenario 6: pickRepoRoot (async picker path) ──
  await check("pickRepoRoot switch: same save/preserve/restore + design chat cleared", async () => {
    const tl = makeTimeline();
    state.repoRoot = "/repo/A";
    tl.appendChild(userMsg("hello A"));
    const aKey = api._tlKey("/repo/A");
    const bKey = api._tlKey("/repo/B");
    storage.set(bKey, savedPayload("/repo/B", ["B-picked"]));

    await api.pickRepoRoot();

    assert.ok(storage.has(aKey), "outgoing A saved via picker");
    assert.ok(storage.has(bKey), "incoming B preserved via picker");
    assert.equal(designClears, 1, "design chat cleared on repo switch");
    assert.ok(tl._children.some((c) => c.textContent === "B-picked"), "B restored via picker");
    assert.equal(state.repoRoot, "/repo/B");
  });

  // ── Scenario 7: source gates ──
  await check("source gate: purgeStorage:false in both handlers, save→clear→load order, pagehide", () => {
    const ui = fs.readFileSync(uiPath, "utf8");
    const ap = fs.readFileSync(apPath, "utf8");
    assert.equal(ui.split("_agentClearPanel({ purgeStorage: false })").length - 1, 2,
      "exactly the two repo-switch call sites pass purgeStorage:false");
    assert.ok(ui.includes('addEventListener("pagehide"'), "P11-2 pagehide listener present");
    assert.ok(ap.includes("function _agentClearPanel(opts)"), "opts parameter added");
    assert.ok(ap.includes("opts.purgeStorage !== false"), "default purge=true preserved");
    for (const needle of ['el("set-repo-btn")?.addEventListener("click"', "async function pickRepoRoot()"]) {
      const seg = ui.slice(ui.indexOf(needle), ui.indexOf(needle) + 1200);
      const s = seg.indexOf("_tlSave();");
      const a = seg.indexOf("_agentClearPanel({ purgeStorage: false });");
      const l = seg.indexOf("_tlLoad(state.repoRoot);");
      assert.ok(s >= 0 && a >= 0 && l >= 0, `handler contains save/clear/load: ${needle}`);
      assert.ok(s < a && a < l, `ordering _tlSave → _agentClearPanel → _tlLoad: ${needle}`);
    }
  });

  console.log(`\n${passed} checks passed`);
})().catch((e) => {
  console.error("FAIL:", e && e.message ? e.message : e);
  process.exit(1);
});
