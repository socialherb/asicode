#!/usr/bin/env node
/**
 * Regression harness — 27차 감사 #1: "Load" button in the Commit Log panel
 * was completely dead (no listener, no handler, no matching id).
 *
 * Bug class: 3-way mismatch —
 *   1. ui.js:2647 registered a listener on el("git-commit-refresh-btn") which
 *      does NOT exist in ui.html (the real button is id="commit-refresh-btn")
 *      → el() returned null → optional chaining silently skipped the
 *      registration.
 *   2. The registered handler refreshGitCommitInfoSafe is defined NOWHERE in
 *      ui.js → even a matching id would have thrown ReferenceError on click.
 *   3. The real #commit-refresh-btn element had ZERO listeners → clicking
 *      "Load" did nothing at all.
 *
 * Fix under test: commit-refresh-btn is wired to refreshGitPanelsSafe (the
 * existing, defined function that refreshes both status and log), and the
 * stale git-commit-refresh-btn / refreshGitCommitInfoSafe references are gone.
 *
 * Run: node tests/js/test_git_commit_refresh_wiring.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const JS_PATH = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const HTML_PATH = path.join(__dirname, "..", "..", "webapp", "ui", "templates", "ui.html");
const uiSrc = fs.readFileSync(JS_PATH, "utf8");
const htmlSrc = fs.readFileSync(HTML_PATH, "utf8");

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// 1. The Load button id actually exists in the HTML.
check("commit-refresh-btn exists in ui.html", () => {
  assert.ok(
    htmlSrc.includes('id="commit-refresh-btn"'),
    'ui.html has no id="commit-refresh-btn"'
  );
});

// 2. ui.js wires that id to a click listener (this is the line that was dead).
check("commit-refresh-btn wired in ui.js", () => {
  assert.ok(
    uiSrc.includes('el("commit-refresh-btn")?.addEventListener'),
    'ui.js does not register a listener on commit-refresh-btn'
  );
});

// 3. The wired handler is actually defined in ui.js (no dangling reference).
check("wired handler is defined in ui.js", () => {
  const m = uiSrc.match(
    /el\("commit-refresh-btn"\)\?\.addEventListener\("click",\s*(\w+)/
  );
  assert.ok(m, "cannot parse commit-refresh-btn listener registration");
  const handler = m[1];
  const defined =
    new RegExp(`(?:^|\\n)async function ${handler}\\b`).test(uiSrc) ||
    new RegExp(`(?:^|\\n)function ${handler}\\b`).test(uiSrc) ||
    uiSrc.includes(`const ${handler} =`);
  assert.ok(defined, `handler ${handler} is not defined in ui.js`);
});

// 4. The stale wiring is gone (old non-existent id + undefined function).
check("stale git-commit-refresh-btn / refreshGitCommitInfoSafe references gone", () => {
  assert.ok(
    !uiSrc.includes("git-commit-refresh-btn"),
    "stale git-commit-refresh-btn reference remains"
  );
  assert.ok(
    !uiSrc.includes("refreshGitCommitInfoSafe"),
    "stale refreshGitCommitInfoSafe reference remains"
  );
});

// 5. The wired handler actually refreshes the commit log (dead-handler guard:
//    wiring a no-op would leave the button just as dead).
check("wired handler refreshes the commit log", () => {
  const name = "refreshGitPanelsSafe";
  const start = uiSrc.indexOf(`async function ${name}(`);
  assert.ok(start >= 0, `async function ${name} not found`);
  let depth = 0;
  let i = uiSrc.indexOf("{", start);
  let end = -1;
  for (; i < uiSrc.length; i++) {
    if (uiSrc[i] === "{") depth++;
    else if (uiSrc[i] === "}") {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  assert.ok(end > 0, "unbalanced braces extracting refreshGitPanelsSafe");
  const body = uiSrc.slice(start, end + 1);
  assert.ok(
    body.includes("/ui/api/git/log"),
    "refreshGitPanelsSafe does not call the git log endpoint"
  );
  assert.ok(
    body.includes("renderGitLog"),
    "refreshGitPanelsSafe does not render the commit log"
  );
});

console.log(`\n${passed} checks passed`);
