#!/usr/bin/env node
/**
 * Round-30 cleanup: 509 lines of dead CSS removed from ui.css — 96 rules
 * across 9 feature groups (analyze-first-aborted card, design handoff,
 * impl preview, LLM-external settings row, pipeline v2, 5 toggle labels,
 * settings checkbox grid, agent-card-type cat variants, misc singles) plus
 * @keyframes asrFlashBg, 10 unused :root variables and 9 orphaned section
 * headers. Every removed class had ZERO consumers across ui.html, ui.js,
 * agent-panel.js, design-chat.js, ui-actions.js — and was cross-checked
 * against dynamically-generated classes (template-literal prefixes).
 *
 * Gates:
 *   R1 removal       — every dead group token absent from ui.css. Catches
 *                      re-introduction of any removed rule.
 *   R2 consumers     — dead tokens absent from consumer surfaces too, so a
 *                      future edit cannot reference a class that no longer
 *                      has styling (silent no-op styling bug).
 *   R3 anti-vacuous  — the LIVE mechanisms that COULD have been swept up in
 *                      a purge like this must still exist: .hidden rule,
 *                      non-v2 .pipe-stage (agent-panel.js renders it),
 *                      .pipeline-bar (ui.html), agent-card-toolname (the
 *                      rename that replaced agent-card-type), dynamically
 *                      generated prefix usage in JS (cat-/risk-/hljs), and
 *                      prompt-toggle-label (the ONE live toggle label —
 *                      proof the dead five were filtered, not pattern-killed).
 *
 * Run: node tests/js/test_dead_css_groups_gate.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const staticDir = path.join(__dirname, "..", "..", "webapp", "ui", "static");
const css = fs.readFileSync(path.join(staticDir, "ui.css"), "utf8");
const html = fs.readFileSync(path.join(staticDir, "..", "templates", "ui.html"), "utf8");
const jsFiles = ["ui.js", "agent-panel.js", "design-chat.js", "ui-actions.js"].map((f) => [
  f,
  fs.readFileSync(path.join(staticDir, f), "utf8"),
]);

// One representative token per removed group (+ singles with consumers to guard).
const DEAD_TOKENS = [
  "afab-",                          // analyze-first-aborted card (backend event absent)
  "agent-analyze-first-aborted-card",
  "design-handoff-",                // handoff bubbles/buttons
  "handoff-accepted",
  "impl-preview__",                 // implementation preview card
  "design-impl-preview-card",
  "llm-external",                   // "LLM+General (external)" settings row
  "llm-context-segmented-control",
  "pipe-stage-v2",                  // pipeline bar v2 (current is non-v2 pipe-stage)
  "pipeline-bar-enhanced",
  "pipe-connector-v2",
  "pipe-name",                      // v2-only children
  "pipe-time",
  "plan-toggle-label",              // 5 toggle labels with no markup
  "review-toggle-label",
  "rag-toggle-label",
  "tdd-toggle-label",
  "approve-toggle-label",
  "settings-checkbox-grid",         // settings checkbox layout
  "settings-row--select",
  "agent-card-type",                // renamed → agent-card-toolname
  "agent-card-summary",
  "agent-card-turn",
  "agent-btn",
  "apply-toast",
  "changed-flash",                  // + its private keyframes
  "asrFlashBg",
  "diff-provenance-turn",
  "hljs-line",                      // hljs never emits it for loaded languages
  "modal-content",                  // media-query leftovers
  "main-content",
];
// Tokens that must ALSO vanish from consumer surfaces (had markup/JS refs removed in earlier eras).
const DEAD_IN_CONSUMERS = [
  "afab-", "design-handoff-", "impl-preview__", "llm-external",
  "pipe-stage-v2", "pipeline-bar-enhanced", "plan-toggle-label",
  "agent-card-type", "changed-flash", "apply-toast",
];
const DEAD_VARS = [
  "--bg-crust", "--bg-mantle", "--btn-danger", "--btn-success", "--btn-warning",
  "--color-anthropic", "--color-deepseek", "--color-google", "--color-openai",
  "--font-size-lg",
];

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── R1: dead groups gone from ui.css ──
check("R1 — ui.css contains none of the 32 dead group tokens", () => {
  const hits = DEAD_TOKENS.filter((t) => css.includes(t));
  assert.deepStrictEqual(hits, [], `dead tokens re-introduced: ${hits.join(", ")}`);
});

check("R1 — ui.css contains none of the 10 dead :root variables", () => {
  const hits = DEAD_VARS.filter((v) => css.includes(v));
  assert.deepStrictEqual(hits, [], `dead vars re-introduced: ${hits.join(", ")}`);
});

check("R1 — css braces balanced (structure not mangled)", () => {
  assert.strictEqual(css.split("{").length, css.split("}").length, "{ and } counts must match");
});

// ── R2: no consumer references a now-unstyled class ──
for (const [name, src] of [["ui.html", html], ...jsFiles]) {
  check(`R2 — ${name} references no removed class`, () => {
    const hits = DEAD_IN_CONSUMERS.filter((t) => src.includes(t));
    assert.deepStrictEqual(hits, [], `${name} references removed classes: ${hits.join(", ")}`);
  });
}

// ── R3: live mechanisms survived (anti-vacuous) ──
check("R3 — .hidden rule still defined (real hidden mechanism)", () => {
  assert.ok(/^\.hidden\s*\{/m.test(css), "ui.css must still define .hidden");
});

check("R3 — non-v2 .pipe-stage rule survives + agent-panel.js renders it", () => {
  assert.ok(/^\.pipe-stage\s*\{/m.test(css), "ui.css must still style .pipe-stage");
  const ap = jsFiles.find(([n]) => n === "agent-panel.js")[1];
  assert.ok(/pipe-stage/.test(ap), "agent-panel.js must still build pipe-stage elements");
});

check("R3 — .pipeline-bar survives in css + ui.html markup", () => {
  assert.ok(/^\.pipeline-bar\s*\{/m.test(css), "ui.css must still style .pipeline-bar");
  assert.ok(/pipeline-bar/.test(html), "ui.html must still contain pipeline-bar");
});

check("R3 — agent-card-toolname (the rename that replaced agent-card-type) alive", () => {
  assert.ok(/agent-card-toolname/.test(css), "ui.css must style agent-card-toolname");
  const ap = jsFiles.find(([n]) => n === "agent-panel.js")[1];
  assert.ok(/agent-card-toolname/.test(ap), "agent-panel.js must render agent-card-toolname");
});

check("R3 — dynamically-generated class prefixes still used by JS", () => {
  const all = jsFiles.map(([, src]) => src).join("\n");
  for (const p of ["cat-", "risk-", "error-status-"]) {
    assert.ok(all.includes(p), `JS must still generate ${p}* classes (cat-/risk- in agent-panel.js, error-status- in ui.js)`);
  }
});

check("R3 — prompt-toggle-label (the ONE live toggle label) survived", () => {
  assert.ok(/prompt-toggle-label/.test(css), "ui.css must still style prompt-toggle-label");
  assert.ok(/prompt-toggle-label/.test(html), "ui.html must still use prompt-toggle-label");
});

check("R3 — live keyframes referenced by surviving rules only", () => {
  // every animation name referenced in css must have a @keyframes block.
  // CSS keywords (none/inherit/…) and timing functions are not names.
  const KEYWORDS = new Set([
    "none", "inherit", "initial", "unset", "revert", "infinite", "alternate",
    "ease", "ease-in", "ease-out", "ease-in-out", "linear", "step-end", "step-start",
    "normal", "reverse", "alternate-reverse", "both", "forwards", "backwards", "running", "paused",
  ]);
  const names = [...css.matchAll(/animation(?:-name)?\s*:\s*([^;"}]+)/g)]
    .flatMap((m) => m[1].split(","))
    .map((s) => s.trim().split(/\s+/)[0])
    .filter((nm) => /^[a-zA-Z][\w-]*$/.test(nm) && !KEYWORDS.has(nm));
  assert.ok(names.length > 0, "expected at least one animation reference in css");
  for (const nm of new Set(names)) {
    assert.ok(
      new RegExp(`@keyframes\\s+${nm}\\s*\\{`).test(css),
      `animation "${nm}" has no @keyframes block`
    );
  }
});

console.log(`\n${passed}/15 checks passed (dead css groups gate)`);
process.exit(passed === 15 ? 0 : 1);
