#!/usr/bin/env node
/**
 * Round-28 audit #4: the status-bar sync indicator (ui.html #status-sync)
 * was a static stub — a rotate icon with zero JS references, no text, no
 * updates. Its sibling #status-branch is fed by renderGitStatus() from
 * GET /ui/api/git/status ({branch, clean, porcelain}).
 *
 * Fix under test: renderGitStatus() also renders the sync indicator from
 * the SAME payload — working-tree sync state:
 *   clean  -> "Synced"          (class sync-clean)
 *   dirty  -> "N changes"       (class sync-dirty, title w/ count)
 *   error  -> empty text        (class sync-unknown, title unavailable)
 * and the rotate icon doubles as a manual refresh trigger
 * (click -> refreshGitPanelsSafe).
 *
 * The REAL renderGitStatus + escapeHtml are sliced out of ui.js and executed
 * against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_status_sync_wiring.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const uiPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const cssPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.css");
const htmlPath = path.join(__dirname, "..", "..", "webapp", "ui", "templates", "ui.html");
const src = fs.readFileSync(uiPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");
const html = fs.readFileSync(htmlPath, "utf8");

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
const syncStub = { className: "", title: "", textContent: "" };
globalThis.el = (id) =>
  id === "git-status" ? preStub
  : id === "status-branch" ? branchStub
  : id === "status-sync" ? syncStub
  : null;
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

check("ui.html has a status-sync element", () => {
  assert.ok(html.includes('id="status-sync"'), "status-sync span must exist in ui.html");
});

check("clean tree renders 'Synced' with sync-clean class", () => {
  syncStub.className = ""; syncStub.title = ""; syncStub.textContent = "";
  render({ ok: true, branch: "main", clean: true, porcelain: [] });
  assert.strictEqual(syncStub.textContent, "Synced");
  assert.ok(syncStub.className.includes("sync-clean"), `got class: ${syncStub.className}`);
  assert.ok(syncStub.title.includes("clean"), `got title: ${syncStub.title}`);
});

check("dirty tree renders 'N changes' with sync-dirty class", () => {
  syncStub.className = ""; syncStub.title = ""; syncStub.textContent = "";
  render({ ok: true, branch: "main", clean: false, porcelain: [" M ui.js", "?? tests/js/"] });
  assert.strictEqual(syncStub.textContent, "2 changes");
  assert.ok(syncStub.className.includes("sync-dirty"), `got class: ${syncStub.className}`);
  assert.ok(syncStub.title.includes("2 uncommitted"), `got title: ${syncStub.title}`);
});

check("single dirty line renders '1 change' (singular)", () => {
  syncStub.className = ""; syncStub.title = ""; syncStub.textContent = "";
  render({ ok: true, branch: "main", clean: false, porcelain: [" M ui.js"] });
  assert.strictEqual(syncStub.textContent, "1 change");
});

check("unavailable status degrades to sync-unknown, no crash", () => {
  syncStub.className = ""; syncStub.title = ""; syncStub.textContent = "";
  render({ ok: false });
  assert.strictEqual(syncStub.textContent, "");
  assert.ok(syncStub.className.includes("sync-unknown"), `got class: ${syncStub.className}`);
  assert.ok(syncStub.title.includes("unavailable"), `got title: ${syncStub.title}`);
});

check("dirty count ignores blank porcelain lines", () => {
  syncStub.textContent = "";
  render({ ok: true, branch: "main", clean: false, porcelain: [" M ui.js", "   ", ""] });
  assert.strictEqual(syncStub.textContent, "1 change");
});

// ── SOURCE GATES: wiring must stay in place ──
check("source gate — click listener refreshes git panels", () => {
  assert.ok(
    src.includes(`el("status-sync")?.addEventListener("click", refreshGitPanelsSafe)`),
    "status-sync must be clickable to refresh"
  );
});

check("source gate — sync state classes exist in ui.css", () => {
  assert.ok(css.includes("sync-clean"), "ui.css must define .sync-clean");
  assert.ok(css.includes("sync-dirty"), "ui.css must define .sync-dirty");
  assert.ok(css.includes("sync-unknown"), "ui.css must define .sync-unknown");
});

console.log(`\n${passed}/8 checks passed (status-sync wiring)`);
process.exit(passed === 8 ? 0 : 1);
