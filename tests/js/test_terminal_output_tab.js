#!/usr/bin/env node
/**
 * Round-29 audit #1: the OUTPUT terminal tab (ui.html #terminal-tab-output)
 * was dead UI — zero JS references, no click handler, clicking did nothing.
 *
 * Fix under test:
 *   1. ui.js initTerminalTabs() — switches between the STATUS body (#timeline)
 *      and the OUTPUT body (#terminal-output) via .active / .hidden classes.
 *   2. agent-panel.js _agentOutputClear / _agentOutputAppend /
 *      _agentOutputLogEvent — the OUTPUT body renders the RAW agent-run
 *      transcript as plain-text rows. Rows are appended via textContent ONLY
 *      (untrusted LLM/tool text must never become innerHTML), and the log is
 *      bounded by _AGENT_OUTPUT_MAX_CHARS (oldest content dropped at the cap).
 *   3. The LIVE dispatch path (the EventSequencer handlers map — the
 *      addEventListener blocks in agentRunStream are overridden to no-ops by
 *      the sequencer) records tool / reasoning / session / complete /
 *      cancelled / error events, and a fresh run clears the log.
 *
 * The REAL functions are sliced out of ui.js / agent-panel.js and executed
 * against a stub DOM (no test framework, no browser).
 *
 * Run: node tests/js/test_terminal_output_tab.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const uiPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js");
const panelPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const cssPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.css");
const htmlPath = path.join(__dirname, "..", "..", "webapp", "ui", "templates", "ui.html");
const uiSrc = fs.readFileSync(uiPath, "utf8");
const panelSrc = fs.readFileSync(panelPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");
const html = fs.readFileSync(htmlPath, "utf8");

// ── Slice real functions out of a source file (brace-balanced) ──
function sliceFunction(src, name) {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found`);
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

// ── Minimal DOM stub: textContent-backed node (setter clears children) ──
function makeNode() {
  const n = {
    _text: "",
    children: [],
    className: "",
    classes: new Set(),
    scrollHeight: 200,
    scrollTop: 200,
    clientHeight: 100,
    classList: {
      toggle(name, force) {
        if (force === undefined) {
          if (n.classes.has(name)) n.classes.delete(name); else n.classes.add(name);
        } else if (force) n.classes.add(name); else n.classes.delete(name);
      },
      add() {}, remove() {},
    },
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); this.children = []; },
    appendChild(c) { this.children.push(c); this._text += c.textContent; },
  };
  return n;
}

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`PASS: ${name}`);
}

// ═══════════════════ 1. HTML structure ═══════════════════

check("ui.html has STATUS + OUTPUT terminal tabs and both bodies", () => {
  assert.ok(html.includes('id="terminal-tab-status"'), "STATUS tab id missing");
  assert.ok(html.includes('id="terminal-tab-output"'), "OUTPUT tab id missing");
  assert.ok(html.includes('id="timeline"'), "timeline body missing");
  assert.ok(html.includes('id="terminal-output"'), "terminal-output body missing");
});

check("ui.html OUTPUT body starts hidden", () => {
  const m = html.match(/id="terminal-output"[^>]*/);
  assert.ok(m && m[0].includes("hidden"), "terminal-output must start with hidden class");
});

check("ui.css styles .terminal-output-line (pre-wrap for raw text)", () => {
  assert.ok(css.includes(".terminal-output-line"), "missing .terminal-output-line rule");
});

// ═══════════════════ 2. ui.js tab switching ═══════════════════

const initTerminalTabsText = sliceFunction(uiSrc, "initTerminalTabs");

check("ui.js defines initTerminalTabs() and wires it in init", () => {
  assert.ok(initTerminalTabsText.includes("function initTerminalTabs()"), "definition missing");
  assert.ok(uiSrc.includes("initTerminalTabs();"), "init wiring call missing");
});

check("ui.js initTerminalTabs switches bodies + active class both ways", () => {
  const calls = [];
  const mkTab = () => {
    const tab = { _active: false, listeners: {} };
    tab.classList = {
      toggle(name, force) {
        if (name === "active") tab._active = force === undefined ? !tab._active : !!force;
        calls.push([tab, name, force]);
      },
    };
    tab.addEventListener = (ev, fn) => { tab.listeners[ev] = fn; };
    return tab;
  };
  const statusTab = mkTab();
  const outputTab = mkTab();
  const timeline = makeNode();
  const outputBody = makeNode();
  globalThis.el = (id) =>
    id === "terminal-tab-status" ? statusTab
    : id === "terminal-tab-output" ? outputTab
    : id === "timeline" ? timeline
    : id === "terminal-output" ? outputBody
    : null;

  const init = new Function(`${initTerminalTabsText}\nreturn initTerminalTabs;`)();
  init();

  // Click OUTPUT
  outputTab.listeners.click();
  assert.ok(outputTab._active, "output tab should become active");
  assert.ok(!statusTab._active, "status tab should lose active");
  assert.ok(timeline.classes.has("hidden"), "timeline should be hidden");
  assert.ok(!outputBody.classes.has("hidden"), "output body should be visible");
  // Click STATUS
  statusTab.listeners.click();
  assert.ok(statusTab._active, "status tab should be active again");
  assert.ok(!outputTab._active, "output tab should lose active");
  assert.ok(!timeline.classes.has("hidden"), "timeline should be visible");
  assert.ok(outputBody.classes.has("hidden"), "output body should be hidden");
});

check("ui.js initTerminalTabs is a safe no-op when elements are missing", () => {
  globalThis.el = () => null;
  const init = new Function(`${initTerminalTabsText}\nreturn initTerminalTabs;`)();
  assert.doesNotThrow(() => init(), "missing elements must not crash");
});

// ═══════════════════ 3. agent-panel.js raw output log ═══════════════════

const scrollBottomText = sliceFunction(panelSrc, "_agentScrollBottom");
const clearText = sliceFunction(panelSrc, "_agentOutputClear");
const appendText = sliceFunction(panelSrc, "_agentOutputAppend");
const logEventText = sliceFunction(panelSrc, "_agentOutputLogEvent");
const maxCharsMatch = panelSrc.match(/const _AGENT_OUTPUT_MAX_CHARS = [^;]+;/);
assert.ok(maxCharsMatch, "_AGENT_OUTPUT_MAX_CHARS declaration not found");
const maxCharsText = maxCharsMatch[0];
const MAX = 256 * 1024;

check("agent-panel.js defines the OUTPUT log API", () => {
  assert.ok(panelSrc.includes("function _agentOutputClear()"), "clear missing");
  assert.ok(panelSrc.includes("function _agentOutputAppend(line)"), "append missing");
  assert.ok(panelSrc.includes("function _agentOutputLogEvent(kind, detail)"), "logEvent missing");
  assert.ok(panelSrc.includes("_AGENT_OUTPUT_MAX_CHARS"), "cap constant missing");
});

function makeLogEnv() {
  const body = makeNode();
  globalThis.document = { createElement: () => makeNode(), getElementById: () => body };
  const make = new Function(
    `${maxCharsText}\n${scrollBottomText}\n${clearText}\n${appendText}\n${logEventText}\n` +
    `return { clear: _agentOutputClear, append: _agentOutputAppend, log: _agentOutputLogEvent };`
  )();
  return { body, ...make };
}

check("append writes a plain-text row (XSS: raw HTML stays text)", () => {
  const { body, append } = makeLogEnv();
  append("<img src=x onerror=alert(1)>");
  assert.strictEqual(body.children.length, 1, "one row appended");
  assert.strictEqual(body.children[0].className, "terminal-output-line");
  assert.strictEqual(body.children[0].textContent, "<img src=x onerror=alert(1)>");
  assert.strictEqual(body.textContent, "<img src=x onerror=alert(1)>");
});

check("clear empties the log", () => {
  const { body, append, clear } = makeLogEnv();
  append("line-1");
  append("line-2");
  clear();
  assert.strictEqual(body.textContent, "");
  assert.strictEqual(body.children.length, 0);
});

check("append is a safe no-op when the OUTPUT body is absent", () => {
  globalThis.document = { createElement: () => makeNode(), getElementById: () => null };
  const make = new Function(
    `${maxCharsText}\n${scrollBottomText}\n${clearText}\n${appendText}\n${logEventText}\n` +
    `return { clear: _agentOutputClear, append: _agentOutputAppend };`
  )();
  assert.doesNotThrow(() => { make.append("x"); make.clear(); });
});

check("log is bounded: oldest content drops at _AGENT_OUTPUT_MAX_CHARS", () => {
  const { body, append } = makeLogEnv();
  append("HEAD" + "A".repeat(MAX / 2));
  append("B".repeat(MAX / 2) + "TAIL");
  // After the cap: textContent sliced to keep the TAIL of existing content,
  // then the new row appended — total must not exceed the cap.
  assert.ok(body.textContent.length <= MAX, `log exceeded cap: ${body.textContent.length}`);
  assert.ok(!body.textContent.includes("HEAD"), "oldest content must be dropped");
  assert.ok(body.textContent.includes("TAIL"), "newest content must survive");
});

check("logEvent formats '[HH:MM:SS] [kind] detail' and clips detail at 4000", () => {
  const { body, log } = makeLogEnv();
  log("tool", "x".repeat(5000));
  assert.ok(/^\[\d{2}:\d{2}:\d{2}\] \[tool\] /.test(body.textContent), "timestamp+kind prefix");
  assert.ok(body.textContent.includes("x".repeat(4000)), "first 4000 chars kept");
  assert.ok(!body.textContent.includes("x".repeat(4001)), "detail must be clipped");
  log("tool", null);
  log("tool", undefined);
  assert.ok(body.textContent.includes("[tool] "), "null/undefined detail degrades to empty");
});

// ═══════════════════ 4. LIVE dispatch path (EventSequencer map) ═══════════════════

check("OUTPUT log hooks live in the EventSequencer handlers map (live path)", () => {
  const start = panelSrc.indexOf("// Define event handlers for EventSequencer");
  const end = panelSrc.indexOf("// Create EventSequencer");
  assert.ok(start >= 0 && end > start, "handlers map region not found");
  const map = panelSrc.slice(start, end);
  for (const hook of ["_agentOutputLogEvent(\"tool\"", "_agentOutputLogEvent(\"reasoning\"",
                      "_agentOutputLogEvent(\"session\"", "_agentOutputLogEvent(\"complete\"",
                      "_agentOutputLogEvent(\"cancelled\"", "_agentOutputLogEvent(\"error\""]) {
    assert.ok(map.includes(hook), `hook missing from live handlers map: ${hook}`);
  }
  // The run-start hook lives in agentRunStream (new run → fresh log).
  const runStart = panelSrc.indexOf("async function agentRunStream(params)");
  assert.ok(runStart >= 0, "agentRunStream not found");
  const runBody = panelSrc.slice(runStart);
  assert.ok(runBody.includes("_agentOutputClear();"), "run start must clear the log");
  assert.ok(runBody.includes("_agentOutputLogEvent(\"run\""), "run start must log the request");
});

console.log(`\nALL ${passed} CHECKS PASSED`);
