#!/usr/bin/env node
/**
 * Regression harness — 27차 감사 #2: the /ui/api/git/commit endpoint had
 * ZERO consumers (dead contract, archived P11-4) even though git_commit was
 * hardened in round 26 (silent add -A fallback removal cf4acc65, staging-
 * composition guard 5ef39e29, dash-prefix filter removal 8d382ac6). This
 * test locks in the UI commit wiring: a commit-message input + Commit button
 * in the Git panel, wired to doGitCommit(), which POSTs to /ui/api/git/commit
 * with the Changes-panel porcelain as explicit touched_files (so the server
 * staging guard is active) and refreshes the git panels afterwards.
 *
 * Run: node tests/js/test_git_commit_wiring.js
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

// --- helpers -------------------------------------------------------------

// Extract a top-level function body from source (brace-matched).
function extractFunction(src, name) {
  const re = new RegExp(`(?:^|\\n)(?:async )?function ${name}\\(`);
  const start = src.search(re);
  if (start < 0) return null;
  let depth = 0;
  let i = src.indexOf("{", start);
  if (i < 0) return null;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  return null;
}

// --- checks --------------------------------------------------------------

// 1-2. The commit UI exists in the HTML (input + button).
check("commit-message-input exists in ui.html", () => {
  assert.ok(
    htmlSrc.includes('id="commit-message-input"'),
    'ui.html has no id="commit-message-input"'
  );
});

check("commit-btn exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="commit-btn"'), 'ui.html has no id="commit-btn"');
});

// 3. ui.js wires the button to a click listener.
check("commit-btn wired to a click listener in ui.js", () => {
  assert.ok(
    uiSrc.includes('el("commit-btn")?.addEventListener("click"'),
    "ui.js does not register a click listener on commit-btn"
  );
});

// 4. The wired handler is actually defined (no dangling reference).
check("wired click handler is defined in ui.js", () => {
  const m = uiSrc.match(/el\("commit-btn"\)\?\.addEventListener\("click",\s*(\w+)/);
  assert.ok(m, "cannot parse commit-btn click listener registration");
  const handler = m[1];
  const defined = extractFunction(uiSrc, handler) !== null || uiSrc.includes(`const ${handler} =`);
  assert.ok(defined, `handler ${handler} is not defined in ui.js`);
});

// 5. The handler actually POSTs to the commit endpoint (dead-wiring guard).
check("doGitCommit POSTs to /ui/api/git/commit", () => {
  const body = extractFunction(uiSrc, "doGitCommit");
  assert.ok(body, "doGitCommit not found in ui.js");
  assert.ok(
    body.includes('apiPost("/ui/api/git/commit"'),
    "doGitCommit does not call apiPost(/ui/api/git/commit)"
  );
});

// 6. Success path refreshes the git panels (dead-end guard).
check("doGitCommit refreshes git panels after committing", () => {
  const body = extractFunction(uiSrc, "doGitCommit");
  assert.ok(body, "doGitCommit not found in ui.js");
  assert.ok(
    body.includes("refreshGitPanelsSafe"),
    "doGitCommit never refreshes the git panels"
  );
});

// 7. The commit sends the Changes-panel porcelain as explicit touched_files
//    so the server-side staging-composition guard stays active.
check("doGitCommit sends touched_files from porcelainPaths", () => {
  const body = extractFunction(uiSrc, "doGitCommit");
  assert.ok(body, "doGitCommit not found in ui.js");
  assert.ok(
    body.includes("touched_files"),
    "doGitCommit does not send touched_files"
  );
  assert.ok(
    body.includes("porcelainPaths"),
    "doGitCommit does not derive touched_files from the status porcelain"
  );
});

// 8. porcelain parsing is correct for the porcelain=v1 shapes (executed, not
//    just source-inspected): plain modified, untracked, rename (take the
//    destination), C-quoted paths, and too-short lines are skipped.
check("porcelainPaths parses all porcelain=v1 shapes", () => {
  const fnSrc = extractFunction(uiSrc, "porcelainPaths");
  assert.ok(fnSrc, "porcelainPaths not found in ui.js");
  const unescapeSrc = extractFunction(uiSrc, "unescapeGitPath");
  assert.ok(unescapeSrc, "unescapeGitPath not found in ui.js");
  const porcelainPaths = eval(`(${fnSrc})`); // eslint-disable-line no-eval
  const unescapeGitPath = eval(`(${unescapeSrc})`); // eslint-disable-line no-eval
  assert.deepStrictEqual(
    porcelainPaths([
      " M src/a.py",
      "A  src/new.py",
      "?? untracked.txt",
      "R  old.py -> new.py",
      ' M "a b.txt"',
      "D  gone.txt",
      "!! ignored",
      "x",
      "",
    ]),
    ["src/a.py", "src/new.py", "untracked.txt", "new.py", "a b.txt", "gone.txt", "ignored"]
  );
  // Untracked directory collapses to the dir itself (git emits "dir/");
  // a trailing-slash entry is passed through and the server guards it.
  assert.deepStrictEqual(porcelainPaths(["?? vendor/"]), ["vendor/"]);
  assert.deepStrictEqual(porcelainPaths(null), []);
  assert.deepStrictEqual(porcelainPaths([]), []);
});

// 9. Enter in the message input triggers the commit.
check("Enter in commit-message-input triggers doGitCommit", () => {
  assert.ok(
    uiSrc.includes('el("commit-message-input")?.addEventListener("keydown"'),
    "commit-message-input has no keydown listener"
  );
  const m = uiSrc.match(/el\("commit-message-input"\)\?\.addEventListener\("keydown",[\s\S]{0,300}?doGitCommit\(\)/);
  assert.ok(m, "keydown handler does not call doGitCommit");
});

// 10. Empty message is blocked client-side before hitting the server.
check("doGitCommit guards against an empty commit message", () => {
  const body = extractFunction(uiSrc, "doGitCommit");
  assert.ok(body, "doGitCommit not found in ui.js");
  assert.ok(
    /message\s*=\s*input\.value\.trim\(\)/.test(body) && body.includes("if (!message)"),
    "doGitCommit does not trim + guard the commit message"
  );
});

console.log(`\n${passed} checks passed`);
