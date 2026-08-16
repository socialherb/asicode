#!/usr/bin/env node
/**
 * Regression harness — 28차 감사 #2: the design-spec banner was a DEAD
 * contract. ui.html declares it (display:none), ui.css styles it, and BOTH
 * existing JS references only HIDE it (agent-panel.js clear-panel,
 * ui-actions.js dismiss-banner). Nothing ever fills design-spec-banner-text
 * or shows the banner — the "설계 명세" banner was permanently invisible.
 *
 * Contract (from the ui.html comment): "shown when transitioning from design
 * chat" — leaving design mode (_applyDesignMode(false)) must pin a compact
 * digest of the design conversation (turn count + latest user request).
 *
 * Run: node tests/js/test_design_spec_banner_wiring.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const HTML_PATH = path.join(__dirname, "..", "..", "webapp", "ui", "templates", "ui.html");
const CSS_PATH = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.css");
const DESIGN_JS = path.join(__dirname, "..", "..", "webapp", "ui", "static", "design-chat.js");
const AGENT_JS = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const ACTIONS_JS = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui-actions.js");

const htmlSrc = fs.readFileSync(HTML_PATH, "utf8");
const cssSrc = fs.readFileSync(CSS_PATH, "utf8");
const designSrc = fs.readFileSync(DESIGN_JS, "utf8");
const agentSrc = fs.readFileSync(AGENT_JS, "utf8");
const actionsSrc = fs.readFileSync(ACTIONS_JS, "utf8");

let passed = 0;
function check(name, fn) { fn(); passed++; console.log(`PASS: ${name}`); }

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

// ── DOM anchors (these existed all along — the bug is the missing show path) ──
check("banner element exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="design-spec-banner"'), 'ui.html has no id="design-spec-banner"');
});

check("banner text span exists in ui.html", () => {
  assert.ok(htmlSrc.includes('id="design-spec-banner-text"'), 'ui.html has no id="design-spec-banner-text"');
});

check("close button exists with dismiss-banner data-action", () => {
  assert.ok(htmlSrc.includes('data-action="dismiss-banner"'), "close button data-action missing");
});

check("banner styles exist in ui.css", () => {
  assert.ok(cssSrc.includes("#design-spec-banner"), "ui.css has no #design-spec-banner rule");
});

// ── Existing hide paths must keep working (guards) ──
check("dismiss-banner handler still hides the banner", () => {
  assert.ok(actionsSrc.includes("dismiss-banner"), "ui-actions.js lost dismiss-banner");
  assert.ok(actionsSrc.includes("design-spec-banner"), "ui-actions.js lost banner reference");
});

check("clear-panel still hides the banner", () => {
  assert.ok(agentSrc.includes("design-spec-banner"), "agent-panel.js lost banner reference");
});

// ── Source gates: the missing show path must now exist ──
check("_designShowSpecBanner exists in design-chat.js", () => {
  assert.ok(designSrc.includes("function _designShowSpecBanner("), "_designShowSpecBanner not found — show path is still dead");
});

check("_buildDesignSpecDigest exists in design-chat.js", () => {
  assert.ok(designSrc.includes("function _buildDesignSpecDigest("), "_buildDesignSpecDigest not found");
});

check("_applyDesignMode calls _designShowSpecBanner when leaving design mode", () => {
  const fn = extractFunction(designSrc, "_applyDesignMode");
  assert.ok(fn, "_applyDesignMode not extractable");
  // Must be on the leave-design-mode path (the !enabled block).
  const leaveIdx = fn.indexOf("if (!enabled)");
  assert.ok(leaveIdx >= 0, "_applyDesignMode has no !enabled block");
  const leaveBlock = fn.slice(leaveIdx);
  assert.ok(leaveBlock.includes("_designShowSpecBanner()"), "_applyDesignMode never shows the banner on the leave path");
});

// ── Execution gates: slice the REAL functions and run them on stubs ──
function buildDesignApi() {
  const digest = extractFunction(designSrc, "_buildDesignSpecDigest");
  const show = extractFunction(designSrc, "_designShowSpecBanner");
  assert.ok(digest, "could not extract _buildDesignSpecDigest");
  assert.ok(show, "could not extract _designShowSpecBanner");
  const blob =
    "const _designChat = { history: [] };\n" + digest + "\n" + show +
    "\nreturn { _designChat, _buildDesignSpecDigest, _designShowSpecBanner };\n";
  return new Function(blob)();
}

function fakeDom(map) {
  globalThis.document = {
    getElementById: (id) => (Object.prototype.hasOwnProperty.call(map, id) ? map[id] : null),
  };
}

check("digest is null for empty history", () => {
  const api = buildDesignApi();
  assert.strictEqual(api._buildDesignSpecDigest([]), null);
  assert.strictEqual(api._buildDesignSpecDigest([{ role: "ai", content: "hi" }]), null,
    "ai-only history has no user request to pin");
});

check("digest includes turn count and the LAST user request", () => {
  const api = buildDesignApi();
  const history = [
    { role: "user", content: "첫 요청" },
    { role: "ai", content: "응답1" },
    { role: "user", content: "최종 요청" },
    { role: "ai", content: "응답2" },
  ];
  const d = api._buildDesignSpecDigest(history);
  assert.ok(d.includes("4"), `digest should mention turn count: ${d}`);
  assert.ok(d.includes("최종 요청"), `digest should pin the latest request: ${d}`);
  assert.ok(!d.includes("첫 요청"), "digest must not pin an older request");
});

check("digest collapses whitespace and truncates long requests", () => {
  const api = buildDesignApi();
  const long = "x".repeat(300);
  const d = api._buildDesignSpecDigest([{ role: "user", content: `a\n\n b ${long}` }]);
  assert.ok(!d.includes("\n"), "digest must be single-line");
  assert.ok(d.length < 200, `digest must be truncated, got ${d.length} chars`);
  assert.ok(d.includes("…"), "truncation marker missing");
});

check("showSpec displays the banner with the digest text", () => {
  const api = buildDesignApi();
  const banner = { style: { display: "none" } };
  const text = { textContent: "" };
  fakeDom({ "design-spec-banner": banner, "design-spec-banner-text": text });
  api._designChat.history = [{ role: "user", content: "요청" }, { role: "ai", content: "답" }];
  api._designShowSpecBanner();
  assert.strictEqual(banner.style.display, "", "banner must be shown after leaving design mode");
  assert.ok(text.textContent.includes("요청"), "banner text must carry the design digest");
});

check("showSpec hides the banner when there is no design context", () => {
  const api = buildDesignApi();
  const banner = { style: { display: "" } };
  const text = { textContent: "stale" };
  fakeDom({ "design-spec-banner": banner, "design-spec-banner-text": text });
  api._designChat.history = [];
  api._designShowSpecBanner();
  assert.strictEqual(banner.style.display, "none", "empty design history must hide the banner");
  assert.strictEqual(text.textContent, "", "stale digest text must be cleared");
});

check("missing banner elements are a harmless no-op", () => {
  const api = buildDesignApi();
  fakeDom({});
  api._designChat.history = [{ role: "user", content: "요청" }];
  assert.doesNotThrow(() => api._designShowSpecBanner());
});

console.log(`\nAll ${passed} checks passed.`);
