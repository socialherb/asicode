#!/usr/bin/env node
/**
 * Regression harness — P9-1: approval timeout resolves the card by identity,
 * not by position.
 *
 * Bug: _agentUpdateLastApprovalCard("timeout") picked `.agent-approval-card`
 * with `.at(-1)` — the LAST card in the timeline. With parallel subagent
 * gates (ThreadPoolExecutor in orchestrator), two approval cards coexist;
 * subagent A's timeout stripped the Approve/Reject buttons off still-pending
 * card B (and vice versa). Checkpoint (user-input) cards share the
 * `.agent-approval-card` class, so an approval timeout could also resolve a
 * pending user-input gate and remove its input UI.
 *
 * Fix under test: _agentResolveApprovalCard(state, requestId, cardEl)
 *  1. exact cardEl (countdown expiry — caller holds the card)
 *  2. dataset.approvalRequestId lookup (server ships request_id on
 *     approval_timeout events, agent_stream.py)
 *  3. fallback: last UNRESOLVED non-checkpoint card (legacy events)
 *
 * The REAL function text is sliced out of agent-panel.js (brace-balanced)
 * and executed against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_approval_identity.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

// ── Slice the REAL functions out of agent-panel.js (brace-balanced) ──
const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

function sliceFn(name) {
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
  assert.ok(i < src.length, `unbalanced braces while extracting ${name}`);
  return src.slice(start, i + 1);
}

const fnText = sliceFn("_agentResolveApprovalCard")
  + "\n" + sliceFn("_agentApplyApprovalState")
  + "\n" + sliceFn("_agentUpdateLastApprovalCard");

// ── Stub environment ──
const timelineStub = { _cards: [] };
timelineStub.querySelectorAll = (sel) => {
  if (sel === ".agent-approval-card:not(.agent-checkpoint-card)") {
    return timelineStub._cards.filter(c => !c._classes.has("agent-checkpoint-card"));
  }
  return [];
};

globalThis._agentGetTimeline = () => timelineStub;
globalThis._agentScrollBottom = () => {};

function makeCard({ reqId = "", checkpoint = false, resolved = false, connected = true } = {}) {
  const actions = { removed: false, remove() { this.removed = true; } };
  const card = {
    isConnected: connected,
    dataset: { approvalRequestId: reqId },
    _classes: new Set(),
    classList: {
      add: (c) => card._classes.add(c),
      contains: (c) => card._classes.has(c),
    },
    querySelector: (sel) => {
      if (sel === ".agent-approval-icon") return { textContent: "" };
      if (sel === ".agent-approval-countdown") return { textContent: "" };
      if (sel === ".agent-approval-actions") return actions;
      return null;
    },
    closest: (sel) => (sel === "#agent-timeline" ? timelineStub : null),
    _actions: actions,
  };
  if (checkpoint) card._classes.add("agent-checkpoint-card");
  if (resolved) card._classes.add("agent-approval-timeout");
  return card;
}

// Compile the real functions in global scope so free variables
// (_agentGetTimeline, _agentScrollBottom) resolve against the stubs.
const resolveCard = new Function(fnText + "\nreturn _agentResolveApprovalCard;")();

// ── Helpers ──
function resetTimeline(cards) {
  timelineStub._cards = cards;
  for (const c of cards) {
    c._classes.delete("agent-approval-timeout");
    c._actions.removed = false;
  }
}
const isResolved = (c) => c._classes.has("agent-approval-timeout");

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`  ok - ${name}`);
}

// ── Part A: request_id identity ──
console.log("Part A: request_id identity (parallel subagent gates)");
{
  // Card A (req "A") and card B (req "B") coexist; A's timeout arrives.
  const cardA = makeCard({ reqId: "A" });
  const cardB = makeCard({ reqId: "B" });
  resetTimeline([cardA, cardB]);

  resolveCard("timeout", "A");

  check("A resolved by request_id", () => assert.ok(isResolved(cardA)));
  check("B untouched by A's timeout", () => {
    assert.ok(!isResolved(cardB), "B must not be resolved");
    assert.ok(!cardB._actions.removed, "B's Approve/Reject buttons must survive");
  });
  check("A's buttons removed", () => assert.ok(cardA._actions.removed));
}
{
  // Timeout for a request_id with NO matching card -> no-op (nothing stripped).
  const cardB = makeCard({ reqId: "B" });
  resetTimeline([cardB]);
  resolveCard("timeout", "nonexistent");
  check("unknown request_id resolves nothing", () => {
    assert.ok(!isResolved(cardB) && !cardB._actions.removed);
  });
}
{
  // Legacy event WITHOUT request_id falls back to last unresolved card.
  const cardA = makeCard({ reqId: "A", resolved: true });
  const cardB = makeCard({ reqId: "B" });
  resetTimeline([cardA, cardB]);
  resolveCard("timeout");
  check("legacy fallback picks last unresolved", () => {
    assert.ok(!isResolved(cardA), "already-resolved card must be skipped");
    assert.ok(isResolved(cardB));
  });
}

// ── Part B: checkpoint isolation ──
console.log("Part B: checkpoint (user-input) cards are never resolved by approval timeout");
{
  const cp = makeCard({ reqId: "cp-1", checkpoint: true });
  resetTimeline([cp]);
  resolveCard("timeout", "cp-1");
  check("checkpoint-only timeline: no-op", () => {
    assert.ok(!isResolved(cp), "checkpoint card must not be marked timeout");
    assert.ok(!cp._actions.removed);
  });
}
{
  // Approval card + checkpoint card: legacy fallback must skip the checkpoint.
  const cardA = makeCard({ reqId: "A" });
  const cp = makeCard({ reqId: "cp-1", checkpoint: true });
  resetTimeline([cardA, cp]);
  resolveCard("timeout");
  check("fallback skips checkpoint card", () => {
    assert.ok(isResolved(cardA), "approval card resolved");
    assert.ok(!isResolved(cp), "checkpoint card untouched");
    assert.ok(!cp._actions.removed);
  });
}

// ── Part C: exact-card passthrough (countdown expiry path) ──
console.log("Part C: countdown expiry passes the exact card");
{
  const cardA = makeCard({ reqId: "A" });
  const cardB = makeCard({ reqId: "B" });
  resetTimeline([cardA, cardB]);
  // Countdown on B fires with its own request_id + card handle.
  resolveCard("timeout", "B", cardB);
  check("cardEl wins over position", () => {
    assert.ok(isResolved(cardB));
    assert.ok(!isResolved(cardA), "A must not be resolved");
  });
}
{
  // Disconnected cardEl is ignored -> falls back to request_id lookup.
  const cardA = makeCard({ reqId: "A" });
  const ghost = makeCard({ reqId: "GHOST", connected: false });
  resetTimeline([cardA]);
  resolveCard("timeout", "GHOST", ghost);
  check("disconnected cardEl ignored", () => {
    assert.ok(!isResolved(cardA), "no match, nothing resolved");
  });
}

// ── Part D: legacy wrapper ──
console.log("Part D: _agentUpdateLastApprovalCard wrapper");
{
  const legacy = new Function(fnText + "\nreturn _agentUpdateLastApprovalCard;")();
  const cardB = makeCard({ reqId: "B" });
  resetTimeline([cardB]);
  legacy("timeout", "main");
  check("wrapper routes to fallback resolver", () => assert.ok(isResolved(cardB)));
}

// ── Part E: source gates (regression prevention) ──
console.log("Part E: source gates");
{
  // No positional ".at(-1)" resolution left anywhere in the file.
  const posPattern = /querySelectorAll\("\.agent-approval-card"\)\]\.at\(-1\)/;
  check("no positional .at(-1) resolution remains", () => {
    assert.ok(!posPattern.test(src), "positional last-card resolution still present");
  });

  // The 3 call sites must pass identity (request_id / card handle).
  check("SSE handler passes data.request_id", () =>
    assert.ok(src.includes('_agentResolveApprovalCard("timeout", data.request_id)'),
      "agent-panel.js approval_timeout handler lost request_id"));
  check("attach handler passes d.request_id", () =>
    assert.ok(src.includes('_agentResolveApprovalCard("timeout", d.request_id)'),
      "attach approval_timeout listener lost request_id"));
  check("countdown passes card + request_id", () =>
    assert.ok(src.includes('_agentResolveApprovalCard("timeout", card.dataset.approvalRequestId, card)'),
      "countdown expiry lost exact-card passthrough"));

  // Only the wrapper definition remains; its body routes to the resolver.
  const legacyCalls = [...src.matchAll(/(?<!function )_agentUpdateLastApprovalCard\(/g)];
  check("legacy wrapper has no external callers", () => {
    assert.strictEqual(legacyCalls.length, 0,
      `expected 0 external calls, found ${legacyCalls.length}`);
  });
  check("legacy wrapper routes to fallback resolver", () => {
    assert.ok(src.includes('function _agentUpdateLastApprovalCard(state, agentId = "main")'),
      "wrapper definition missing");
    assert.ok(src.includes("_agentResolveApprovalCard(state);"),
      "wrapper no longer routes to _agentResolveApprovalCard");
  });

  // P9-4: dead analyze_first_aborted handler removed.
  check("analyze_first_aborted dead handler removed", () => {
    assert.ok(!src.includes("analyze_first_aborted"), "dead handler still present");
  });

  // P9-5: timeline byte cap + stale sweep present.
  check("_TL_MAX_BYTES constant present", () =>
    assert.ok(src.includes("_TL_MAX_BYTES"), "_TL_MAX_BYTES missing"));
  check("_tlSweepStale present + boot-wired", () => {
    assert.ok(src.includes("function _tlSweepStale()"), "_tlSweepStale missing");
    assert.ok(src.includes("_tlSweepStale(); // boot-time cleanup"),
      "_tlSweepStale not wired into _agentInitTimelineCollapse");
  });

  // P9-6: _agentGetTimeline's no-op parameter removed everywhere.
  check("_agentGetTimeline has no parameters", () =>
    assert.ok(src.includes("function _agentGetTimeline()"),
      "no-op agentId parameter still present"));
  check("_agentGetTimeline call sites pass no argument", () => {
    const withArgs = [...src.matchAll(/_agentGetTimeline\(([^)]*)\)/g)]
      .filter(m => m[1].trim() !== "");
    assert.deepStrictEqual(withArgs.map(m => m[0]), [],
      "call sites still pass an argument to _agentGetTimeline");
  });
}

console.log(`\nPASS — ${passed} checks`);
