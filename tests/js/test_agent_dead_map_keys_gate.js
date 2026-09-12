#!/usr/bin/env node
/**
 * Regression harness — round 32-3 (C-round): 18 dead consumers removed.
 *
 * The SSE emit↔consume gate (tests/unit/test_sse_emit_consume_gate.py) found
 * 16 map keys in agent-panel.js and 2 listeners in design-chat.js whose events
 * NO webapp emitter ever produces: a removed local-assistant/delegation flow
 * (7), a removed spec-resolver/graph-enrichment/tool-filter emit (4), a
 * removed self-review flow (2), checkpoint notify/timeout relics (2), and
 * CLI-only budget/complexity warnings (2) — agent-panel.js is webapp UI, so
 * CLI emits never reach it. design_chunk was a noop placeholder and
 * design_tool_stream a relic of a removed streaming-tool flow.
 *
 * Removed with them: 4 dead post-wire addEventListener twins, the orphaned
 * _agentUpdateLastReviewCard helper, and 59 lines of orphaned CSS
 * (.spec-resolver-card family, .agent-graph-enrichment-chip).
 *
 * Gates:
 *   R1 removal      — the 18 names are absent from every consumer surface
 *                     (agent-panel.js, design-chat.js, ui.js, ui-actions.js,
 *                     ui.html): neither map keys, addEventListener twins,
 *                     nor string literals (catches re-introduction).
 *   R2 orphans      — orphaned CSS tokens and the orphaned helper are absent
 *                     from ui.css AND consumer surfaces (silent no-op
 *                     styling bug prevention).
 *   R3 anti-vacuous — the LIVE things a purge like this could sweep up must
 *                     still exist: fail_loop_detected (the surviving consumer
 *                     of .agent-tool-filtered-chip), the review-card family
 *                     (_agentAddReviewCard + converged review_* handlers),
 *                     checkpoint approval_timeout (kept — it IS emitted),
 *                     ≥40 live map keys, design_token (the real stream) and
 *                     design_typing (emitted noop).
 *
 * R1/R2 match by TOKEN SHAPE, never by bare substring: a purged name counts as
 * absent only when it is absent as a WHOLE token, because a live name that
 * merely CONTAINS it (measured 2026-09-12: the live handler key
 * `context_budget_warning` contains the purged `budget_warning`) otherwise reads
 * as a re-introduction — a false alarm, which is indistinguishable from a real
 * regression and trains people to weaken the gate. See `hasToken` below; both
 * directions of every rule it implements are pinned by
 * tests/js/test_agent_dead_map_gate_boundaries.js.
 *
 * Run: node tests/js/test_agent_dead_map_keys_gate.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const staticDir = path.join(__dirname, "..", "..", "webapp", "ui", "static");
const surfaces = {
  "agent-panel.js": fs.readFileSync(path.join(staticDir, "agent-panel.js"), "utf8"),
  "design-chat.js": fs.readFileSync(path.join(staticDir, "design-chat.js"), "utf8"),
  "ui.js": fs.readFileSync(path.join(staticDir, "ui.js"), "utf8"),
  "ui-actions.js": fs.readFileSync(path.join(staticDir, "ui-actions.js"), "utf8"),
};
const html = fs.readFileSync(path.join(staticDir, "..", "templates", "ui.html"), "utf8");
const css = fs.readFileSync(path.join(staticDir, "ui.css"), "utf8");

const AGENT_KEYS = [
  "local_assistant_start", "local_assistant_plan", "local_assistant_fallback",
  "local_assistant_error", "local_assistant_complete",
  "local_delegation_start", "local_delegation_complete",
  "graph_enrichment_applied", "spec_resolver_complete", "tool_filtered",
  "review_tool_call", "review_complete",
  "user_input_received", "user_input_timeout",
  "budget_warning", "small_model_complexity_warning",
];
const DESIGN_KEYS = ["design_chunk", "design_tool_stream"];

// ── Token matching ─────────────────────────────────────────────────────────
// A purge gate written as `!src.includes(name)` is a SUBSTRING test, so a LIVE
// name that merely CONTAINS a purged one reads as its re-introduction. Measured
// 2026-09-12: the live handler key `context_budget_warning` (agent-panel.js,
// added after this suite last ran in CI) contains the purged `budget_warning`,
// so the gate failed on a tree whose purge was fully intact. Every token
// therefore declares the SHAPE it is looked for in:
//
//   identifier — a whole JS identifier: a handlers-map key, an SSE event name, a
//                helper name. Present only when NEITHER neighbour is an
//                identifier character ([A-Za-z0-9_$]), so `context_budget_warning`
//                no longer counts as `budget_warning`, while `budget_warning:`,
//                `"budget_warning"`, `(budget_warning)` and `x.budget_warning`
//                still do.
//   class      — a CSS family PREFIX: `.spec-resolver-card` and
//                `.spec-resolver-header` ARE references to the purged
//                `.spec-resolver` family, so a trailing `-suffix` deliberately
//                stays part of the match (right side unbounded). The token must
//                still BEGIN an identifier run, so a preceding `-` is a boundary
//                only when the separator run it crosses does not itself continue
//                an identifier: `--spec-resolver-card` (a custom property) is a
//                hit, `my-spec-resolver-widget` (a class with its own head) is
//                not.
//
// Both rules are character predicates evaluated per occurrence — no token ever
// reaches a regex compiler, so a token carrying regex metacharacters cannot
// change the rule it is judged by.
const IDENT_CHAR = /[A-Za-z0-9_$]/;
const IDENT_RUN_CHAR = /[A-Za-z0-9_]/;

function hasToken(src, token, kind) {
  for (let i = src.indexOf(token); i !== -1; i = src.indexOf(token, i + 1)) {
    if (kind === "class") {
      let j = i;
      while (j > 0 && src[j - 1] === "-") j -= 1; // cross the family separator run
      if (j === 0 || !IDENT_RUN_CHAR.test(src[j - 1])) return true;
    } else {
      const before = src[i - 1];
      const after = src[i + token.length];
      if ((before === undefined || !IDENT_CHAR.test(before)) &&
          (after === undefined || !IDENT_CHAR.test(after))) {
        return true;
      }
    }
  }
  return false;
}

// ── R1: removal — absent from every consumer surface ──────────────────────
for (const [file, src] of Object.entries(surfaces)) {
  for (const name of AGENT_KEYS) {
    assert.ok(!hasToken(src, name, "identifier"),
      `${file} must not reference dead key "${name}"`);
  }
  if (file === "design-chat.js") {
    for (const name of DESIGN_KEYS) {
      assert.ok(!hasToken(src, name, "identifier"),
        `design-chat.js must not reference dead key "${name}"`);
    }
  }
}
for (const name of [...AGENT_KEYS, ...DESIGN_KEYS]) {
  assert.ok(!hasToken(html, name, "identifier"),
    `ui.html must not reference dead key "${name}"`);
}

// ── R2: orphaned CSS + helper absent everywhere ────────────────────────────
const ORPHAN_CSS = ["spec-resolver", "spec-file", "spec-symbols", "agent-graph-enrichment"];
for (const token of ORPHAN_CSS) {
  assert.ok(!hasToken(css, token, "class"),
    `ui.css must not define orphaned rule ".${token}"`);
  for (const [file, src] of Object.entries(surfaces)) {
    assert.ok(!hasToken(src, token, "class"),
      `${file} must not reference removed class "${token}"`);
  }
}
for (const [file, src] of Object.entries(surfaces)) {
  assert.ok(!hasToken(src, "_agentUpdateLastReviewCard", "identifier"),
    `${file} must not reference the orphaned helper _agentUpdateLastReviewCard`);
}

// ── R3: anti-vacuous — live mechanisms survived the purge ─────────────────
const ap = surfaces["agent-panel.js"];
const regionStart = ap.indexOf("// Define event handlers");
const regionEnd = ap.indexOf("// Create EventSequencer");
assert.ok(regionStart >= 0 && regionEnd > regionStart, "live handlers-map region markers must exist");
const mapRegion = ap.slice(regionStart, regionEnd);

// fail_loop_detected survives and still consumes .agent-tool-filtered-chip
assert.ok(mapRegion.includes("fail_loop_detected:"), "fail_loop_detected map entry must survive");
assert.ok(mapRegion.includes("agent-tool-filtered-chip"), "fail_loop_detected must keep .agent-tool-filtered-chip");
assert.ok(css.includes(".agent-tool-filtered-chip"), ".agent-tool-filtered-chip must stay in ui.css");

// review-card family: helper + all four converged handlers
assert.ok(ap.includes("function _agentAddReviewCard("), "_agentAddReviewCard must survive");
for (const h of ["review_start:", "review_approved:", "review_rejected:", "review_skipped:"]) {
  assert.ok(mapRegion.includes(h), `converged handler ${h} must stay in the live map`);
}

// approval_timeout IS emitted — its handler must survive both surfaces
assert.ok(mapRegion.includes("approval_timeout:"), "approval_timeout map entry must survive");
assert.ok(ap.includes('source.addEventListener("approval_timeout"'),
  "approval_timeout post-wire listener must survive");

// the live map is still substantial (48 consumers measured at round 32-3)
const keyCount = (mapRegion.match(/^    [a-z_0-9]+: \(data\) => \{$/gm) || []).length;
assert.ok(keyCount >= 40, `live map must keep >=40 keys, found ${keyCount}`);

// design: the real token stream and the emitted typing noop stay
const dc = surfaces["design-chat.js"];
assert.ok(dc.includes('sse.addEventListener("design_token"'), "design_token listener must survive");
assert.ok(dc.includes('sse.addEventListener("design_typing"'), "design_typing noop must survive");

console.log(`OK — dead-map-key gate: 18 names absent from 5 surfaces, ` +
  `${ORPHAN_CSS.length} orphan CSS families + orphan helper gone, ` +
  `live mechanisms intact (${keyCount} map keys)`);
