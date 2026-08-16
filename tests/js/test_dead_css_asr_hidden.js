#!/usr/bin/env node
/**
 * Round-29 cleanup: `.asr-hidden { display: none !important; }` was the only
 * `asr-*` class in ui.css and had ZERO consumers — no reference in ui.html,
 * ui.js, agent-panel.js or design-chat.js. A dead utility class that looks
 * load-bearing (who toggles it? nothing) invites confusion with the REAL
 * hidden mechanism: `.hidden` (ui.css) + classList add/remove("hidden").
 *
 * Fix under test: the rule (and its now-empty section header
 * "Agent UI dynamic mode helpers") is deleted from ui.css.
 *
 * Gates:
 *   R1 removal      — "asr-hidden" absent from ui.css AND every consumer
 *                     surface (html/js) — catches re-introduction and any
 *                     consumer left behind by the removal.
 *   R2 anti-vacuous — the LIVE `.hidden` rule must still exist in ui.css and
 *                     must still be toggled from JS. Without this, someone
 *                     could "clean up" the real hidden class too and the
 *                     removal gate would still pass vacuously.
 *
 * Run: node tests/js/test_dead_css_asr_hidden.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const staticDir = path.join(__dirname, "..", "..", "webapp", "ui", "static");
const css = fs.readFileSync(path.join(staticDir, "ui.css"), "utf8");
const html = fs.readFileSync(path.join(staticDir, "..", "templates", "ui.html"), "utf8");
const jsFiles = ["ui.js", "agent-panel.js", "design-chat.js"].map((f) => [
  f,
  fs.readFileSync(path.join(staticDir, f), "utf8"),
]);

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ── R1: the dead class is gone everywhere ──
check("R1 — ui.css no longer defines .asr-hidden", () => {
  assert.ok(!css.includes("asr-hidden"), "ui.css must not contain asr-hidden");
});

check("R1 — ui.html has no asr-hidden usage", () => {
  assert.ok(!html.includes("asr-hidden"), "ui.html must not reference asr-hidden");
});

for (const [name, src] of jsFiles) {
  check(`R1 — ${name} has no asr-hidden usage`, () => {
    assert.ok(!src.includes("asr-hidden"), `${name} must not reference asr-hidden`);
  });
}

// ── R2: the live `.hidden` mechanism survived (anti-vacuous) ──
check("R2 — live .hidden rule still defined in ui.css", () => {
  assert.ok(
    /^\.hidden\s*\{/m.test(css),
    "ui.css must still define .hidden — the REAL hidden mechanism"
  );
});

check("R2 — JS still toggles .hidden (classList)", () => {
  const uses = jsFiles.filter(([, src]) => /classList\.(add|remove|toggle|contains)\(\s*["']hidden["']/.test(src));
  assert.ok(uses.length > 0, "at least one JS file must toggle the hidden class");
});

console.log(`\n${passed}/7 checks passed (dead css .asr-hidden removal)`);
process.exit(passed === 7 ? 0 : 1);
