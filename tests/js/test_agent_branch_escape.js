#!/usr/bin/env node
/**
 * Regression harness — P9-3: git branch name is HTML-escaped in the status bar.
 *
 * Bug: renderGitStatus interpolated data.branch straight into
 * branchEl.innerHTML. Git ref rules forbid `~^:?*[\` and whitespace but
 * ALLOW `<`, `>` and `"` — so `git checkout -b '"><img src=x onerror=...>'`
 * plants a stored XSS in the status bar that re-fires on every refresh.
 *
 * Fix under test: escapeHtml(data.branch || "unknown").
 *
 * The REAL renderGitStatus + escapeHtml are sliced out of ui.js and executed
 * against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_agent_branch_escape.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const uiPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const src = fs.readFileSync(uiPath, "utf8");

// ── Slice the real functions out of ui.js (brace-balanced) ──
function sliceFunction(name) {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found in ui.js`);
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

const escapeHtmlText = sliceFunction("escapeHtml");
const renderText = sliceFunction("renderGitStatus");

// ── Stub environment ──
const preStub = { textContent: "" };
const branchStub = { innerHTML: "" };
globalThis.el = (id) =>
  id === "git-status" ? preStub : id === "status-branch" ? branchStub : null;
globalThis.state = {};
globalThis.ensureSessionDigest = () => {};
globalThis.currentRepoRoot = () => "";

const render = new Function(
  `${escapeHtmlText}\n${renderText}\nreturn renderGitStatus;`
)();

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

const XSS_BRANCH = `"><img src=x onerror=alert(1)>`;

check("XSS branch name is escaped in status bar", () => {
  branchStub.innerHTML = "";
  render({ ok: true, branch: XSS_BRANCH, clean: true, porcelain: [] });
  assert.ok(!branchStub.innerHTML.includes(`<img src=x`), "raw branch must not survive");
  assert.ok(branchStub.innerHTML.includes("&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"), "branch escaped");
});

check("missing branch falls back to unknown (no crash)", () => {
  branchStub.innerHTML = "";
  render({ ok: true, clean: true, porcelain: [] });
  assert.ok(branchStub.innerHTML.includes("unknown"));
});

check("plain branch name renders unchanged", () => {
  branchStub.innerHTML = "";
  render({ ok: true, branch: "feat/p9", clean: false, porcelain: [" M ui.js"] });
  assert.ok(branchStub.innerHTML.includes("feat/p9"));
  assert.ok(!branchStub.innerHTML.includes("&lt;"), "no spurious escaping");
});

check("unavailable status renders placeholder", () => {
  preStub.textContent = "";
  render({ ok: false });
  assert.strictEqual(preStub.textContent, "(git status unavailable)");
});

// ── SOURCE GATE: branch interpolation must stay escaped ──
check("source gate — branch innerHTML uses escapeHtml", () => {
  assert.ok(src.includes(`escapeHtml(data.branch || "unknown")`), "branch site must use escapeHtml");
});

console.log(`\n${passed}/5 checks passed (P9-3 branch escape)`);
process.exit(passed === 5 ? 0 : 1);
