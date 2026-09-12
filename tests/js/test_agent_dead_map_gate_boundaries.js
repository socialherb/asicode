#!/usr/bin/env node
/**
 * RED/GREEN proof harness for the dead-map-key gate's token matcher
 * (tests/js/test_agent_dead_map_keys_gate.js).
 *
 * Why this file exists: that gate was written with `!src.includes(name)`, a bare
 * SUBSTRING test, so a LIVE name containing a purged one read as a
 * re-introduction. Measured 2026-09-12: the live handler key
 * `context_budget_warning` contains the purged `budget_warning`, so the gate —
 * and with it the whole JS suite — failed on a tree whose purge was intact.
 * The fix introduces token SHAPES (`identifier` / `class`). A matcher change is
 * exactly the kind of change that can silently turn a purge gate VACUOUS, so
 * this harness runs the REAL gate file, unmodified, over a sandbox copy of the
 * real surfaces and pins both directions of every rule:
 *
 *   GREEN — no false alarm (the gate must PASS):
 *     · the untouched tree, i.e. the live `context_budget_warning` must not read
 *       as the purged `budget_warning`
 *     · `prefix_budget_warning`            identifier continuation on the left
 *     · `my-spec-resolver-widget`          a class with its own head
 *   RED — removal still caught (the gate must FAIL, with its own message):
 *     · `budget_warning: (data) => {`      handlers-map key
 *     · `addEventListener("budget_warning"` SSE event name
 *     · `"local_assistant_start"`          string literal in another surface
 *     · `"design_chunk"`                   design-chat-only key
 *     · `.spec-resolver-card {`            CSS rule
 *     · `--spec-resolver-card:`            CSS custom property
 *     · `querySelector(".agent-graph-enrichment")`  class token in a JS surface
 *     · `_agentUpdateLastReviewCard`       orphaned helper
 *     · `data-probe="tool_filtered"`       the ui.html surface
 *
 * The sandbox reproduces the path shape the gate resolves (`__dirname/../../`),
 * so the gate copy sees the sandbox files and nothing else. Probe children get a
 * SANITIZED env on purpose: the surrounding suite runs every test under
 * coverage_preload.js with COV_OUT pointing at this test's evidence file, and an
 * inherited COV_OUT would make each probe child overwrite that file (the
 * coverage gate would then report net-new uncovered keys).
 *
 * Run: node tests/js/test_agent_dead_map_gate_boundaries.js
 */
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const assert = require("assert");
const { spawnSync } = require("child_process");

const GATE_REL = path.join("tests", "js", "test_agent_dead_map_keys_gate.js");
const STATIC_REL = path.join("webapp", "ui", "static");
const HTML_REL = path.join("webapp", "ui", "templates", "ui.html");
const SURFACES = ["agent-panel.js", "design-chat.js", "ui.js", "ui-actions.js", "ui.css"];

const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), "dead-map-gate-"));
const pristine = new Map(); // sandbox-relative path -> original text

function seed(rel) {
  const dest = path.join(sandbox, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const text = fs.readFileSync(path.join(__dirname, "..", "..", rel), "utf8");
  fs.writeFileSync(dest, text);
  pristine.set(rel, text);
}

function restoreAll() {
  for (const [rel, text] of pristine) fs.writeFileSync(path.join(sandbox, rel), text);
}

function runGate() {
  const res = spawnSync(process.execPath, [path.join(sandbox, GATE_REL)], {
    cwd: sandbox,
    encoding: "utf8",
    env: { ...process.env, NODE_OPTIONS: "", COV_OUT: "" },
    timeout: 60000,
  });
  return { status: res.status, output: `${res.stdout || ""}${res.stderr || ""}` };
}

let checks = 0;
let green = 0;
let red = 0;

/** `expect` is "pass", or the exact message the gate must print for that token. */
function probe(label, { file = null, snippet = "" } = {}, expect) {
  restoreAll();
  if (file) fs.appendFileSync(path.join(sandbox, file), snippet);
  const { status, output } = runGate();
  // The child's own AssertionError line is the signal that matters ("... must not
  // reference dead key ..."); fall back to the child's last lines when it failed
  // some other way (an uncaught throw, a syntax error in the sandbox copy).
  const cause = (output.match(/AssertionError \[ERR_ASSERTION\]: .*/) || [""])[0];
  const tail = cause || output.trim().split("\n").slice(-6).join("\n");
  if (expect === "pass") {
    assert.strictEqual(status, 0,
      `${label}\nexpected: the gate PASSES (no false alarm). Either the live-name\n` +
      `false alarm is back, or a purged token was genuinely re-introduced — the\n` +
      `gate's own output above names which.\n${tail}`);
    green++;
  } else {
    assert.notStrictEqual(status, 0,
      `${label}\nexpected: the gate FAILS (this token was re-introduced)`);
    assert.ok(output.includes(expect),
      `${label}\nexpected the gate's own message for that token (${expect}), got:\n${tail}`);
    red++;
  }
  checks++;
}

const js = (name) => path.join(STATIC_REL, name);

try {
  seed(GATE_REL);
  seed(HTML_REL);
  for (const s of SURFACES) seed(js(s));

  // ── GREEN: live names that merely CONTAIN a purged token are not removals ──
  probe("untouched tree — live key context_budget_warning must not read as budget_warning", {}, "pass");
  probe("prefix_budget_warning — identifier continuation on the LEFT is not the bare key",
    { file: js("ui.js"), snippet: "\n// probe: prefix_budget_warning\n" }, "pass");
  probe("my-spec-resolver-widget — a class whose own head differs is not the purged family",
    { file: js("ui.js"), snippet: "\nconst probe = \"my-spec-resolver-widget\";\n" }, "pass");

  // ── RED: every shape a purged token can come back in is still caught ──────
  probe("handlers-map key", { file: js("agent-panel.js"), snippet: "\nbudget_warning: (data) => { return data; },\n" },
    "agent-panel.js must not reference dead key \"budget_warning\"");
  probe("SSE event name", { file: js("agent-panel.js"), snippet: "\nsource.addEventListener(\"budget_warning\", () => {});\n" },
    "agent-panel.js must not reference dead key \"budget_warning\"");
  probe("string literal in another surface", { file: js("ui.js"), snippet: "\nconst dead = \"local_assistant_start\";\n" },
    "ui.js must not reference dead key \"local_assistant_start\"");
  probe("design-chat-only key", { file: js("design-chat.js"), snippet: "\nsse.addEventListener(\"design_chunk\", () => {});\n" },
    "design-chat.js must not reference dead key \"design_chunk\"");
  probe("CSS rule of an orphaned family", { file: js("ui.css"), snippet: "\n.spec-resolver-card { color: red; }\n" },
    "ui.css must not define orphaned rule \".spec-resolver\"");
  probe("CSS custom property of an orphaned family", { file: js("ui.css"), snippet: "\n:root { --spec-resolver-card: 1; }\n" },
    "ui.css must not define orphaned rule \".spec-resolver\"");
  probe("orphaned class token in a JS surface", { file: js("ui.js"), snippet: "\ndocument.querySelector(\".agent-graph-enrichment\");\n" },
    "ui.js must not reference removed class \"agent-graph-enrichment\"");
  probe("orphaned helper", { file: js("agent-panel.js"), snippet: "\nfunction _agentUpdateLastReviewCard() { return 1; }\n" },
    "agent-panel.js must not reference the orphaned helper _agentUpdateLastReviewCard");
  probe("ui.html surface", { file: HTML_REL, snippet: "\n<div data-probe=\"tool_filtered\"></div>\n" },
    "ui.html must not reference dead key \"tool_filtered\"");
} finally {
  fs.rmSync(sandbox, { recursive: true, force: true });
}

console.log(`OK — dead-map-key gate boundaries: ${checks} checks (${green} no-false-alarm, ` +
  `${red} re-introduction); matcher pinned in both directions`);
