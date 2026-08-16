#!/usr/bin/env node
/**
 * Regression harness — F1/F2: pending tool card lifecycle (preview → update).
 *
 * F1 — _pendingToolCards orphan bug: the Map ("T{turn}:{tool}" → card queue)
 * was NEVER cleared. A previewed-but-unexecuted card from a cancelled run
 * (AgentCancelled before tool_call, connection loss, etc.) stayed in the map;
 * the next run reuses the same turn numbers, so its tool_call resolved the
 * PREVIOUS run's DOM card via FIFO shift and left the current card spinning
 * "⚡ 실행 중..." forever.
 *
 * F2 — _tlSerialize rendered pending cards as ok:false, so a reload restored
 * a never-executed tool call as a FAILED one ("안 한 것 ≠ 실패").
 *
 * The REAL function text is sliced out of agent-panel.js and executed against
 * a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_pending_tool_cards.js
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

function extractConst(name) {
  const re = new RegExp(`(const ${name} = [^;]+);`);
  const m = src.match(re);
  assert.ok(m, `const ${name} not found in agent-panel.js`);
  return m[1];
}

// Module-level declarations so all compiled functions share ONE instance
// (mirrors how the real module works).
const pendingDecl = src.match(/const _pendingToolCards = new Map\(\);/);
assert.ok(pendingDecl, "_pendingToolCards declaration not found");
const registryDecl = src.match(/const _agentListenerCleanup = (\[[^\]]*\]);/);
assert.ok(registryDecl, "_agentListenerCleanup declaration not found");

const consts = [
  extractConst("_AGENT_TIMELINE_ENTRY_SELECTOR"),
  extractConst("_TL_MAX_ENTRIES"),
  pendingDecl[0],
  `const _agentListenerCleanup = ${registryDecl[1]}`,
].join("\n");

const fnNames = [
  "_agentAddListener",
  "_tlReleaseListeners",
  "_agentAddPendingToolCard",
  "_agentUpdatePendingToolCard",
  "_agentFinalizePendingToolCards",   // F1 (new)
  "_pendingToolCardsRemoveCard",      // F1 (new)
  "_tlPruneLiveTimeline",
  "_tlSerialize",
  "_tlRestore",
];
const fnText = fnNames.map(sliceFunction).join("\n");

// ── Stub DOM ─────────────────────────────────────────────────────────────
const registry = new Map(); // id -> element (mirrors document.getElementById)

class FakeEl {
  constructor(tag, id) {
    this.tag = tag || "div";
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
    const token = sel.replace(/^\./, "");
    const walk = (el) => {
      const mine = el.className.split(/\s+/).includes(token);
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
  querySelector: () => null,
};
globalThis.document = documentStub;

// Free-variable stubs referenced by the compiled functions.
globalThis._escHtml = (s) => String(s == null ? "" : s);
globalThis._renderMd = (s) => String(s == null ? "" : s);
globalThis._agentToolNarration = () => "";
globalThis._agentFormatResultSummary = () => "";
globalThis._toolCategory = () => "read";
globalThis._agentScrollBottom = () => {};
globalThis._pipelineRunStart = 0;

// Compile the real functions in one scope; return the handles we test.
const api = new Function(
  consts + "\n" + fnText + "\nreturn {" +
  "_agentListenerCleanup, _agentAddPendingToolCard, _agentUpdatePendingToolCard, " +
  "_agentFinalizePendingToolCards, _pendingToolCardsRemoveCard, _pendingToolCards, " +
  "_tlPruneLiveTimeline, _tlSerialize, _tlRestore, _TL_MAX_ENTRIES };"
)();

// ── Fixtures ──────────────────────────────────────────────────────────────
function makeTimeline() {
  registry.clear();
  api._agentListenerCleanup.length = 0;
  api._pendingToolCards.clear();
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
function toolCardEl(extraClass) {
  const card = new FakeEl("div");
  card.className = `agent-tool-card ${extraClass}`;
  return card;
}

let passed = 0;
function check(name, fn) { fn(); passed++; console.log(`PASS: ${name}`); }

// ── Scenario 1: F1 cross-run orphan (the user-visible bug) ──
check("F1: cancelled-run orphan card is finalized ⊘, never resolved by next run", () => {
  const tl = makeTimeline();
  // Run A: preview queued, then the run is cancelled BEFORE tool_call.
  const cardA = api._agentAddPendingToolCard(2, "read_file", {}, tl, "main");
  api._agentFinalizePendingToolCards();            // run-A terminal event
  assert.ok(!cardA.classList.contains("pending"), "orphan must lose pending");
  assert.ok(cardA.classList.contains("skipped"), "orphan must be marked skipped (not fail)");
  assert.equal(api._pendingToolCards.size, 0, "map must be emptied at run end");

  // Run B: same turn/tool numbers (turns restart at 1).
  const cardB = api._agentAddPendingToolCard(2, "read_file", {}, tl, "main");
  const updated = api._agentUpdatePendingToolCard(2, "read_file", { ok: true }, "main");
  assert.equal(updated, true, "run B's update must find run B's card");
  assert.ok(cardB.classList.contains("ok") && !cardB.classList.contains("pending"),
    "run B card must resolve to ok");
  assert.ok(!cardA.classList.contains("ok"), "run A orphan must NOT receive run B's result");
  assert.equal(api._pendingToolCards.size, 0, "queue consumed");
});

// ── Scenario 2: finalize marks ⊘ 미실행 + empties FIFO queues ──
check("F1: finalize marks all queued cards ⊘ and empties the map", () => {
  const tl = makeTimeline();
  const c1 = api._agentAddPendingToolCard(1, "grep", {}, tl, "main");
  const c2 = api._agentAddPendingToolCard(1, "grep", {}, tl, "main");  // parallel same-tool
  for (const c of [c1, c2]) {
    const st = new FakeEl("span");
    st.className = "agent-card-status";
    c.appendChild(st);
  }
  api._agentFinalizePendingToolCards();
  for (const c of [c1, c2]) {
    assert.ok(c.classList.contains("skipped") && !c.classList.contains("pending"), "card skipped");
    assert.equal(c.querySelector(".agent-card-status").textContent, "⊘", "status icon ⊘");
  }
  assert.equal(api._pendingToolCards.size, 0, "map emptied");
  api._agentFinalizePendingToolCards();   // idempotent no-op
  assert.equal(api._pendingToolCards.size, 0, "second finalize is a no-op");
});

// ── Scenario 3: prune (G1) drops pending cards from map + DOM ──
check("F1: _tlPruneLiveTimeline removes pending card from map and DOM", () => {
  const tl = makeTimeline();
  const pendingCard = api._agentAddPendingToolCard(1, "grep", {}, tl, "main");
  assert.equal(api._pendingToolCards.size, 1, "pending card tracked");
  for (let i = 0; i < api._TL_MAX_ENTRIES + 2; i++) tl.appendChild(makeEntry("agent-chat-msg", `m${i}`));
  api._tlPruneLiveTimeline();
  assert.ok(!tl._children.includes(pendingCard), "pruned pending card removed from DOM");
  assert.equal(api._pendingToolCards.size, 0,
    "pending queue must not outlive its DOM card (no detached resolve)");
});

// ── Scenario 4: F2 serialize — third state ──
check("F2: serialize distinguishes ok / fail / pending / skipped", () => {
  const tl = makeTimeline();
  tl.appendChild(toolCardEl("ok"));
  tl.appendChild(toolCardEl("fail"));
  tl.appendChild(toolCardEl("pending"));
  tl.appendChild(toolCardEl("skipped"));
  const entries = api._tlSerialize().filter((e) => e.type === "tool");
  assert.equal(entries.length, 4);
  assert.deepEqual(entries.map((e) => e.status), ["ok", "fail", "pending", "skipped"],
    "pending/skipped must NOT collapse into fail (안 한 것 ≠ 실패)");
});

// ── Scenario 5: F2 restore — neutral render for pending/skipped, legacy ok compat ──
check("F2: restore renders pending/skipped as neutral ⊘, legacy ok:false stays fail", () => {
  const tl = makeTimeline();
  api._tlRestore([
    { type: "tool", tool: "a", status: "pending", narration: "", result: "" },
    { type: "tool", tool: "b", status: "skipped", narration: "", result: "" },
    { type: "tool", tool: "c", status: "ok", narration: "", result: "" },
    { type: "tool", tool: "d", ok: false, narration: "", result: "" },   // legacy payload
    { type: "tool", tool: "e", ok: true, narration: "", result: "" },    // legacy payload
  ]);
  const cards = tl._children.filter((c) => c.className.includes("agent-tool-card"));
  assert.equal(cards.length, 5);
  const cls = (c) => c.className;
  assert.ok(cls(cards[0]).includes("skipped") && !cls(cards[0]).includes("fail"),
    "pending entry restores neutral, not fail");
  assert.ok(cls(cards[1]).includes("skipped") && !cls(cards[1]).includes("fail"),
    "skipped entry restores neutral");
  assert.ok(cls(cards[2]).includes("ok"), "ok entry restores ok");
  assert.ok(cls(cards[3]).includes("fail") && !cls(cards[3]).includes("ok"),
    "legacy ok:false still restores fail");
  assert.ok(cls(cards[4]).includes("ok"), "legacy ok:true still restores ok");
});

// ── Scenario 6: _pendingToolCardsRemoveCard direct ──
check("F1: _pendingToolCardsRemoveCard drops a card from its queue", () => {
  makeTimeline();
  const tl = registry.get("agent-timeline");
  const c1 = api._agentAddPendingToolCard(1, "grep", {}, tl, "main");
  const c2 = api._agentAddPendingToolCard(1, "grep", {}, tl, "main");
  api._pendingToolCardsRemoveCard(c1);
  assert.equal(api._pendingToolCards.size, 1, "queue keeps survivor");
  api._pendingToolCardsRemoveCard(c1);   // already gone — no-op
  assert.equal(api._pendingToolCards.size, 1);
  api._pendingToolCardsRemoveCard(c2);
  assert.equal(api._pendingToolCards.size, 0, "empty queue entry deleted");
});

// ── Scenario 7: F3 agent-scoped keys — parallel lanes must not collide ──
check("F3: sub-agent card resolves only via its own agent_id", () => {
  const tl = makeTimeline();
  const subCard = api._agentAddPendingToolCard(2, "read_file", {}, tl, "sub_1");
  const mainCard = api._agentAddPendingToolCard(2, "read_file", {}, tl, "main");
  // The sub-agent's result arrives — must NOT resolve the main lane's card.
  const updatedSub = api._agentUpdatePendingToolCard(2, "read_file", { ok: true }, "sub_1");
  assert.equal(updatedSub, true, "sub-agent update finds its own card");
  assert.ok(subCard.classList.contains("ok") && !subCard.classList.contains("pending"),
    "sub-agent card resolved");
  assert.ok(!mainCard.classList.contains("ok") && mainCard.classList.contains("pending"),
    "main lane card untouched by sub-agent result");
  // Now the main lane's result resolves the main card.
  const updatedMain = api._agentUpdatePendingToolCard(2, "read_file", { ok: true }, "main");
  assert.equal(updatedMain, true, "main update finds its own card");
  assert.ok(mainCard.classList.contains("ok"), "main card resolved by main result");
  assert.equal(api._pendingToolCards.size, 0, "all queues consumed");
});

check("F3: finalize scoped per agent (sub-agent completion only kills its cards)", () => {
  const tl = makeTimeline();
  const subCard = api._agentAddPendingToolCard(1, "grep", {}, tl, "sub_2");
  const mainCard = api._agentAddPendingToolCard(3, "write_plan", {}, tl, "main");
  api._agentFinalizePendingToolCards("sub_2");   // sub-agent completes; main still runs
  assert.ok(subCard.classList.contains("skipped"), "sub-agent card finalized");
  assert.ok(mainCard.classList.contains("pending"), "main card still pending");
  assert.equal(api._pendingToolCards.size, 1, "main queue survives");
  api._agentFinalizePendingToolCards("main");    // main completes
  assert.ok(mainCard.classList.contains("skipped"), "main card finalized at its own terminal");
  assert.equal(api._pendingToolCards.size, 0, "map empty");
});

check("F3: cards carry data-agent-id + agent badge for sub-agents", () => {
  const tl = makeTimeline();
  const mainCard = api._agentAddPendingToolCard(1, "grep", {}, tl, "main");
  const subCard = api._agentAddPendingToolCard(1, "grep", {}, tl, "sub_4");
  assert.ok(!mainCard._attrs["data-agent-id"], "main card has no agent attr");
  assert.equal(subCard._attrs["data-agent-id"], "sub_4", "sub-agent card attributed");
  // Source gate: the card template renders an agent badge when not main.
  const addFn = src.slice(src.indexOf("function _agentAddPendingToolCard("),
    src.indexOf("function _agentUpdatePendingToolCard("));
  assert.ok(addFn.includes('agentKey !== "main"'), "badge gated on non-main agent");
  assert.ok(addFn.includes('"agent-card-agent"'), "badge class present in template");
});

// ── Source gates ──────────────────────────────────────────────────────────
function bodyOf(fnName) {
  const start = src.indexOf(`function ${fnName}(`);
  assert.ok(start >= 0, `function ${fnName} not found`);
  let depth = 0;
  let i = src.indexOf("{", start);
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces extracting ${fnName}`);
  return src.slice(start, i + 1);
}
function handlerBody(eventName) {
  const mark = `source.addEventListener("${eventName}"`;
  const start = src.indexOf(mark);
  assert.ok(start >= 0, `handler not found: ${eventName}`);
  const end = src.indexOf("\n  });", start);
  assert.ok(end > start, `handler close not found: ${eventName}`);
  return src.slice(start, end);
}

check("source gate: run boundaries clear the pending map", () => {
  assert.ok(bodyOf("_agentResetForNewRun").includes("_pendingToolCards.clear()"),
    "_agentResetForNewRun must clear pending cards (run boundary)");
  assert.ok(bodyOf("_agentClearPanel").includes("_pendingToolCards.clear()"),
    "_agentClearPanel must clear pending cards (full wipe)");
});

check("source gate: prune drops pending cards before DOM removal", () => {
  const prune = bodyOf("_tlPruneLiveTimeline");
  assert.ok(prune.includes("_pendingToolCardsRemoveCard"),
    "prune must drop the card from pending queues");
  const doomedLoop = prune.slice(prune.indexOf("for (const node of doomed)"));
  assert.ok(doomedLoop.indexOf("_pendingToolCardsRemoveCard") < doomedLoop.indexOf("_tlReleaseListeners"),
    "pending-drop must run before listener release/removal");
});

check("source gate: every run-terminal path finalizes pending cards", () => {
  assert.ok(handlerBody("complete").includes("_agentFinalizePendingToolCards"),
    "complete handler must finalize");
  assert.ok(handlerBody("cancelled").includes("_agentFinalizePendingToolCards"),
    "cancelled handler must finalize");
  assert.ok(handlerBody("error").includes("_agentFinalizePendingToolCards"),
    "error handler must finalize");
  assert.ok(handlerBody("done").includes("_agentFinalizePendingToolCards"),
    "done handler must finalize");
  const onerrorStart = src.indexOf("source.onerror = () => {");
  assert.ok(onerrorStart >= 0, "source.onerror not found");
  const onerrorEnd = src.indexOf("\n  };", onerrorStart);
  const onerrorBody = src.slice(onerrorStart, onerrorEnd);
  assert.ok(onerrorBody.includes("_agentFinalizePendingToolCards"),
    "source.onerror must finalize (connection lost)");
  assert.ok(bodyOf("agentRunStreamCancel").includes("_agentFinalizePendingToolCards"),
    "agentRunStreamCancel must finalize (user cancel)");
});

check("source gate: finalize marks ⊘ skipped, distinct from fail", () => {
  const body = bodyOf("_agentFinalizePendingToolCards");
  assert.ok(body.includes('card.classList.remove("pending")'), "must clear pending class");
  assert.ok(body.includes('card.classList.add("skipped")'), "must add skipped class");
  assert.ok(body.includes("⊘"), "must show ⊘ (never-executed) marker");
  assert.ok(body.includes("_pendingToolCards.delete(key)"), "must drop map entries");
});

check("source gate: serialize/restore carry the third state", () => {
  const ser = bodyOf("_tlSerialize");
  assert.ok(ser.includes('node.classList.contains("pending")'), "serialize must detect pending");
  assert.ok(ser.includes('node.classList.contains("skipped")'), "serialize must detect skipped");
  assert.ok(ser.includes("status"), "serialize must emit a status field");
  const res = bodyOf("_tlRestore");
  assert.ok(res.includes('"skipped"') && res.includes("neutral"),
    "restore must render pending/skipped as neutral, not fail");
  assert.ok(res.includes("entry.status"), "restore must read the status field");
});

console.log(`\nAll ${passed} checks passed.`);
