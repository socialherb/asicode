#!/usr/bin/env node
/**
 * Regression harness — P9-1: agent taskboard LLM data is HTML-escaped.
 *
 * Bug: _renderAgentTaskBoard interpolates subtask ids, dependency lists and
 * assigned-file names straight into innerHTML. These values originate from
 * the SSE "orchestrator_plan" event — LLM-generated data — while the file's
 * own convention (header comment + _renderMd) treats LLM/tool text as
 * untrusted and escapes it. An LLM (or prompt-injection) can emit
 * `"><img src=x onerror=...>` and own the page.
 *
 * Fix under test: all four interpolation sites in the task-dep / task-files
 * templates now run through _escHtml.
 *
 * The REAL function + _escHtml are sliced out of agent-panel.js and executed
 * against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_taskboard_escape.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

// ── Slice helpers + the real function out of agent-panel.js (brace-balanced) ──
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

const escHtmlText = sliceFunction("_escHtml");
const renderText = sliceFunction("_renderAgentTaskBoard");

// ── Stub environment ──
const hostStub = { style: {}, innerHTML: "" };
globalThis.document = {
  getElementById: (id) => (id === "agent-taskboard" ? hostStub : null),
};

const XSS_ID = `"><img src=x onerror=alert(1)>`;
const XSS_DEP = `dep<script>alert(2)</script>`;
const XSS_FILE = `src/evil"><svg onload=alert(3)>.py`;

const render = new Function(
  `${escHtmlText}\n${renderText}\nreturn _renderAgentTaskBoard;`
)();

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

function freshHost() {
  hostStub.innerHTML = "";
  hostStub.style.display = "";
}

// ── Scenario 1: XSS payload in id + dependencies ──
check("task-dep escapes subtask id and dependency list", () => {
  freshHost();
  globalThis.state = {
    agentSubtasks: [
      { id: XSS_ID, status: "running", dependencies: [XSS_DEP, "ok-dep"] },
    ],
  };
  render();
  const html = hostStub.innerHTML;
  assert.ok(!html.includes(`<img src=x`), "raw <img must not survive");
  assert.ok(!html.includes(`<script>alert(2)`), "raw <script> must not survive");
  assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt;"), "id must be escaped");
  assert.ok(html.includes("dep&lt;script&gt;alert(2)&lt;/script&gt;"), "deps must be escaped");
  assert.ok(html.includes("ok-dep"), "safe dep untouched");
});

// ── Scenario 2: XSS payload in assigned file names ──
check("file badges escape basename", () => {
  freshHost();
  globalThis.state = {
    agentSubtasks: [
      { id: "safe-id", status: "running", assigned_files: [XSS_FILE] },
    ],
  };
  render();
  const html = hostStub.innerHTML;
  assert.ok(!html.includes(`<svg onload=`), "raw <svg onload> must not survive");
  assert.ok(html.includes("evil&quot;&gt;&lt;svg onload=alert(3)&gt;.py"), "basename must be escaped");
});

// ── Scenario 3: independent task (no deps) escapes id too ──
check("independent task-dep escapes id", () => {
  freshHost();
  // hasDeps needs at least one task WITH dependencies for the dep block to render;
  // the second (independent) task then exercises the "(독립)" branch.
  globalThis.state = {
    agentSubtasks: [
      { id: "dep-task", status: "running", dependencies: ["x"] },
      { id: XSS_ID, status: "success" },
    ],
  };
  render();
  const html = hostStub.innerHTML;
  assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt; (독립)"), "independent line escaped");
});

// ── Scenario 4: task-files label escapes id ──
check("task-files label escapes id", () => {
  freshHost();
  globalThis.state = {
    agentSubtasks: [{ id: XSS_ID, status: "success", assigned_files: ["a.py"] }],
  };
  render();
  const html = hostStub.innerHTML;
  assert.ok(!html.includes(XSS_ID), "raw id must not survive");
  assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt;: <span"), "label escaped");
});

// ── Scenario 5: empty subtasks hides the board ──
check("no subtasks hides board without touching innerHTML", () => {
  hostStub.style.display = "flex";
  globalThis.state = { agentSubtasks: [] };
  render();
  assert.strictEqual(hostStub.style.display, "none");
});

// ── Scenario 6: normal data renders unchanged ──
check("normal data renders unchanged", () => {
  freshHost();
  globalThis.state = {
    agentSubtasks: [
      { id: "agent-1", status: "success", dependencies: ["agent-0"], assigned_files: ["src/a.py", "src/b.py"] },
      { id: "agent-0", status: "success" },
    ],
  };
  render();
  const html = hostStub.innerHTML;
  assert.ok(html.includes("2/2"), "progress stat rendered");
  assert.ok(html.includes("agent-1 ← agent-0"), "dep line intact");
  assert.ok(html.includes("a.py") && html.includes("b.py"), "file badges intact");
  assert.ok(!html.includes("&amp;"), "no spurious escaping of plain data");
});

// ── SOURCE GATE: templates must keep _escHtml on all 4 sites ──
(function sourceGate() {
  const mustHave = [
    'class="task-dep">${_escHtml(t.id)} ← ${_escHtml(deps.join(", "))}',
    'class="task-dep">${_escHtml(t.id)} (독립)',
    'class="file-badge">${_escHtml(f.split("/").pop())}',
    'class="task-files">${_escHtml(t.id)}:',
  ];
  for (const pattern of mustHave) {
    assert.ok(src.includes(pattern), `source gate: missing escaped template: ${pattern.slice(0, 40)}...`);
  }
  console.log("PASS: source gate — all 4 taskboard templates escape via _escHtml");
  passed++;
})();

console.log(`\n${passed}/7 checks passed (P9-1 taskboard escape)`);
process.exit(passed === 7 ? 0 : 1);
