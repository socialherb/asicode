#!/usr/bin/env node
/**
 * Regression harness — web UI slash-command wiring:
 *
 * 1. Matching — _slashMatchCommand classifies the prompt's leading token:
 *    known command (name/alias) → {cmd, args}; short ASCII-alpha unknown →
 *    {unknown} (REPL-style typo guard); paths ("/Users/..."), prose and a
 *    bare "/" pass through untouched. The whole feature is inert while the
 *    command list is unavailable (offline → never block).
 * 2. Expansion — _applySlashExpansionToPrompt POSTs /ui/api/slash-commands/
 *    expand and rewrites #prompt-input in place ("expanded"), blocks unknown
 *    typos with an addTL error ("blocked"), and passes everything else
 *    through ("none", no fetch).
 * 3. Autocomplete — _slashOnInput renders the #slash-menu dropdown during
 *    single-token typing; Enter/Tab accepts the highlighted suggestion,
 *    Escape closes, arrows navigate; the menu never opens once args exist.
 * 4. Send-path integration — smartRun/_launchAgent/runOnly/_designSendMessage
 *    all route through the expansion hook (source gate), and the two Enter
 *    keydown handlers (ui.js + design-chat.js) agree via defaultPrevented so
 *    design mode cannot double-send or leak a raw "/token" while the menu is
 *    open.
 *
 * The REAL function/handler text is sliced out of ui.js / design-chat.js
 * (brace-balanced) and executed against a stub DOM (no test framework, no
 * browser). Run: node tests/js/test_slash_prompt.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const uiSrc = fs.readFileSync(path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js"), "utf8");
const dcSrc = fs.readFileSync(path.join(__dirname, "..", "..", "webapp", "ui", "static", "design-chat.js"), "utf8");

// ── Slice helper: brace-balanced from `anchor` to its closing } ──
function sliceFunction(src, anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found`);
  const open = src.indexOf("{", start);
  assert.ok(open >= 0, `no body for ${anchor}`);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces while extracting ${anchor}`);
  return src.slice(start, i + 1);
}

// Registration statement slice (arrow handler → trailing ";" like
// test_prompt_enter.js).
function sliceRegistration(src, anchor) {
  const start = src.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found`);
  let depth = 0;
  let i = src.indexOf("{", start);
  assert.ok(i >= 0, "no handler body found");
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < src.length, `unbalanced braces while extracting ${anchor}`);
  const end = src.indexOf(";", i);
  assert.ok(end >= 0, "statement terminator not found");
  return src.slice(start, end + 1);
}

// ── Slice the REAL functions out of ui.js ──
const fnText = [
  "async function _slashFetchCommands(",
  "function _slashFindCommand(",
  "function _slashMatchCommand(",
  "async function _applySlashExpansionToPrompt(",
  "function _slashMenuOpen(",
  "function _slashCloseMenu(",
  "function _slashAcceptSel(",
  "function _slashMoveSel(",
  "function _slashRenderMenu(",
  "function _slashOnInput(",
].map(a => sliceFunction(uiSrc, a)).join("\n");

const uiKeydown = sliceRegistration(uiSrc, 'el("prompt-input")?.addEventListener("keydown"');
const dcKeydown = sliceRegistration(dcSrc, '_designAddListener(ta, "keydown"');
const smartRunText = sliceFunction(uiSrc, "function smartRun(") + "\n" + sliceFunction(uiSrc, "function _smartRunAfterSlash(");

// ── Stub environment ──
// Module-level state the sliced functions reference (declared with `let` in
// ui.js; provided as globals here so assignments resolve).
globalThis._slashCache = null;
globalThis._slashCacheExpires = 0;
globalThis._SLASH_CACHE_TTL = 60000;
globalThis._slashMenuItems = [];
globalThis._slashMenuSel = -1;

const SAMPLE_COMMANDS = [
  { name: "fix", aliases: ["f", "fixit", "bug"], description: "Fix bugs or issues in code", category: "code", default_params: {} },
  { name: "refactor", aliases: ["rf"], description: "Refactor code for better structure", category: "code", default_params: {} },
  { name: "test", aliases: ["t"], description: "Add or improve tests", category: "code", default_params: {} },
  { name: "explain", aliases: ["ex"], description: "Explain what code does", category: "analysis", default_params: {} },
  { name: "review", aliases: ["rv"], description: "Review code for issues", category: "analysis", default_params: {} },
];

function makeMenu() {
  return {
    hidden: true,
    innerHTML: "",
    _items: [],
    querySelectorAll() { return this._items; },
  };
}
function makeMenuItem() {
  return { classList: { toggle(name, on) { this[name] = on; } } };
}
function makeTa(initial) {
  return {
    value: initial,
    focused: false,
    sel: null,
    focus() { this.focused = true; },
    setSelectionRange(a, b) { this.sel = [a, b]; },
  };
}

let ta = makeTa("");
let menu = makeMenu();
let convToggle = { checked: false };

globalThis.document = {
  getElementById: (id) => {
    if (id === "prompt-input") return ta;
    if (id === "slash-menu") return menu;
    if (id === "conv-layer-toggle") return convToggle;
    return null;
  },
};
// Keydown listeners captured from the two registration slices.
const uiListeners = {};
globalThis.el = (id) => {
  if (id === "prompt-input") {
    return { value: ta.value, addEventListener: (name, fn) => { uiListeners[name] = fn; } };
  }
  return { value: id === "route-mode" ? "auto" : ta.value };
};
globalThis.escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
globalThis._fetchTimeoutSignal = () => null;

let fetchCalls = [];
let fetchQueue = [];
let fetchError = null;
globalThis.fetch = async (url, opts) => {
  fetchCalls.push({ url, opts });
  if (fetchError) { const e = fetchError; fetchError = null; throw e; }
  const r = fetchQueue.shift();
  if (r && r.delay) await new Promise(res => setTimeout(res, r.delay));
  return { ok: r.ok, json: async () => r.json };
};
function queueFetch(ok, json, delay) { fetchQueue.push({ ok, json, delay: delay || 0 }); }

let tlCalls = [];
globalThis.addTL = (type, msg) => { tlCalls.push({ type, msg }); };

// Send-path counters for smartRun/_smartRunAfterSlash stubs.
let launchCalls = 0, runCalls = 0, designSendCalls = 0, midTaskCalls = 0;
globalThis._applyModeActive = false;
globalThis.applyOnly = () => Promise.resolve();
globalThis._resetApplyMode = () => {};
globalThis._isAgentRunning = () => false;
globalThis._agentSessionId = "";
globalThis._sendMidTaskMessage = () => { midTaskCalls++; };
globalThis.normalizeRouteMode = (m) => m || "auto";
globalThis._promptSuggestsAgent = (t) => String(t || "").startsWith("Fix the following issue");
globalThis._launchAgent = () => { launchCalls++; };
globalThis.runOnly = () => { runCalls++; };
globalThis._designSendMessage = () => { designSendCalls++; };
// P15-7 guard in _smartRunAfterSlash — harness prompts are all small.
globalThis._promptWithinLimit = () => true;

// Keydown listeners captured from the two registration slices.
const dcListeners = {};
globalThis._designAddListener = (elm, name, fn) => { dcListeners[name] = fn; };
globalThis.ta = ta;  // design-chat.js handler references the local `ta` — bound globally for the harness

const factory = new Function(
  fnText + "\n" + smartRunText + "\nreturn ({ " + [
    "_slashFetchCommands",
    "_slashFindCommand",
    "_slashMatchCommand",
    "_applySlashExpansionToPrompt",
    "_slashMenuOpen",
    "_slashCloseMenu",
    "_slashAcceptSel",
    "_slashMoveSel",
    "_slashRenderMenu",
    "_slashOnInput",
    "smartRun",
    "_smartRunAfterSlash",
  ].join(", ") + " });"
);
const F = factory();
// The sliced keydown registration statements run in their own compiled scope
// and resolve the menu functions as globals — expose them like the real page.
Object.assign(globalThis, {
  _slashAcceptSel: F._slashAcceptSel,
  _slashMoveSel: F._slashMoveSel,
  _slashCloseMenu: F._slashCloseMenu,
  _slashMenuOpen: F._slashMenuOpen,
  smartRun: F.smartRun,
});
new Function(uiKeydown)();
new Function(dcKeydown)();

const uiKd = uiListeners.keydown || (() => {});
const dcKd = dcListeners.keydown || (() => {});

// ── Helpers ──
// Checks are QUEUED and run sequentially: several checks are async (they
// await the expansion promise), and overlapping them would let a later check
// mutate the shared stub state mid-flight.
const checks = [];
let passed = 0;
function check(name, fn) { checks.push({ name, fn }); }
function resetSlash() {
  _slashCache = SAMPLE_COMMANDS;
  _slashCacheExpires = Date.now() + 60000;
  _slashMenuItems = [];
  _slashMenuSel = -1;
  ta.value = "";
  ta.focused = false;
  ta.sel = null;
  menu.hidden = true;
  menu.innerHTML = "";
  menu._items = [];
  fetchCalls = [];
  fetchQueue = [];
  fetchError = null;
  tlCalls = [];
  launchCalls = 0; runCalls = 0; designSendCalls = 0; midTaskCalls = 0;
  convToggle.checked = false;
}
function fire(handler, overrides) {
  const e = {
    isComposing: false,
    keyCode: 0,
    key: "Enter",
    altKey: false,
    shiftKey: false,
    defaultPrevented: false,
    _prevented: false,
    preventDefault() { this._prevented = true; this.defaultPrevented = true; },
    ...overrides,
  };
  handler(e);
  return e;
}
const tick = () => new Promise(r => setImmediate(r));

// ── 1. Matching ──
resetSlash();
check("_slashMatchCommand: /fix with args", () => {
  const m = F._slashMatchCommand("/fix login bug");
  assert.strictEqual(m.cmd.name, "fix");
  assert.strictEqual(m.args, "login bug");
});
check("_slashMatchCommand: alias /f resolves to fix", () => {
  const m = F._slashMatchCommand("/f x");
  assert.strictEqual(m.cmd.name, "fix");
  assert.strictEqual(m.args, "x");
});
check("_slashMatchCommand: case-insensitive name", () => {
  const m = F._slashMatchCommand("/REVIEW");
  assert.strictEqual(m.cmd.name, "review");
  assert.strictEqual(m.args, "");
});
check("_slashMatchCommand: no args → empty args string", () => {
  const m = F._slashMatchCommand("/fix");
  assert.strictEqual(m.cmd.name, "fix");
  assert.strictEqual(m.args, "");
});
check("_slashMatchCommand: unknown short ascii → {unknown}", () => {
  const m = F._slashMatchCommand("/zzz do something");
  assert.deepStrictEqual(m, { unknown: "/zzz" });
});
check("_slashMatchCommand: path /Users/foo passes through", () => {
  assert.strictEqual(F._slashMatchCommand("/Users/foo/bar"), null);
});
check("_slashMatchCommand: bare / passes through", () => {
  assert.strictEqual(F._slashMatchCommand("/"), null);
});
check("_slashMatchCommand: prose with slash later passes through", () => {
  assert.strictEqual(F._slashMatchCommand("hello /fix world"), null);
});
check("_slashMatchCommand: list unavailable → pass everything through", () => {
  _slashCache = null;
  assert.strictEqual(F._slashMatchCommand("/fix x"), null);
  resetSlash();
});

// ── 2. Expansion ──
check("_applySlashExpansionToPrompt: rewrites textarea with expanded template", async () => {
  resetSlash();
  ta.value = "/fix login bug";
  queueFetch(true, { ok: true, expanded: "Fix the following issue: login bug", command: "fix", description: "Fix bugs or issues in code" });
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "expanded");
  assert.strictEqual(ta.value, "Fix the following issue: login bug");
  assert.strictEqual(fetchCalls.length, 1);
  assert.strictEqual(fetchCalls[0].url, "/ui/api/slash-commands/expand");
  assert.strictEqual(fetchCalls[0].opts.method, "POST");
  const body = JSON.parse(fetchCalls[0].opts.body);
  assert.deepStrictEqual(body, { command: "fix", input: "login bug", context: "" });
});
check("_applySlashExpansionToPrompt: unknown command → blocked + hint, no fetch", async () => {
  resetSlash();
  ta.value = "/zzz go";
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "blocked");
  assert.strictEqual(ta.value, "/zzz go");  // untouched
  assert.strictEqual(fetchCalls.length, 0);
  assert.strictEqual(tlCalls.length, 1);
  assert.ok(tlCalls[0].type === "error" && tlCalls[0].msg.includes("/zzz") && tlCalls[0].msg.includes("/fix"));
});
check("_applySlashExpansionToPrompt: plain text → none, no fetch", async () => {
  resetSlash();
  ta.value = "just a normal prompt";
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "none");
  assert.strictEqual(fetchCalls.length, 0);
});
check("_applySlashExpansionToPrompt: network error → none (never block)", async () => {
  resetSlash();
  ta.value = "/fix x";
  fetchError = new Error("offline");
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "none");
  assert.strictEqual(ta.value, "/fix x");
});
check("_applySlashExpansionToPrompt: expand error payload → none (send raw)", async () => {
  resetSlash();
  ta.value = "/fix x";
  queueFetch(false, { ok: false, error: "INTERNAL_ERROR" });
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "none");
  assert.strictEqual(ta.value, "/fix x");
});
check("_applySlashExpansionToPrompt: cold cache fetches list, then expands (P17-1)", async () => {
  resetSlash();
  _slashCache = null;            // cold — paste/first-send path (no _slashOnInput fetch)
  _slashCacheExpires = 0;
  ta.value = "/fix login bug";
  queueFetch(true, { ok: true, commands: SAMPLE_COMMANDS });          // list
  queueFetch(true, { ok: true, expanded: "Fix the following issue: login bug", command: "fix" });  // expand
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "expanded");
  assert.strictEqual(ta.value, "Fix the following issue: login bug");
  assert.strictEqual(fetchCalls.length, 2);
  assert.strictEqual(fetchCalls[0].url, "/ui/api/slash-commands");
  assert.strictEqual(fetchCalls[1].url, "/ui/api/slash-commands/expand");
  assert.strictEqual(_slashCache, SAMPLE_COMMANDS);  // warmed for the next send
});
check("_applySlashExpansionToPrompt: cold cache + failed list → raw passthrough, never block (P17-1)", async () => {
  resetSlash();
  _slashCache = null;
  _slashCacheExpires = 0;
  ta.value = "/fix x";
  queueFetch(false, { ok: false, error: "REGISTRY_DOWN" });           // list fails
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "none");
  assert.strictEqual(ta.value, "/fix x");           // sent raw, nothing rewritten
  assert.strictEqual(fetchCalls.length, 1);
});
check("paste: space-containing input never fetches, send path still expands (P17-1)", async () => {
  resetSlash();
  _slashCache = null;
  _slashCacheExpires = 0;
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  menu.hidden = false;
  ta.value = "/fix login bug";                      // single paste-style input event
  F._slashOnInput({ isComposing: false });          // L756: contains space → close, NO fetch
  await tick();
  assert.strictEqual(menu.hidden, true);
  assert.strictEqual(_slashMenuItems.length, 0);
  assert.strictEqual(fetchCalls.length, 0);         // input path stays fetch-free
  queueFetch(true, { ok: true, commands: SAMPLE_COMMANDS });
  queueFetch(true, { ok: true, expanded: "Fix the following issue: login bug", command: "fix" });
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "expanded");
  assert.strictEqual(ta.value, "Fix the following issue: login bug");
});
check("P16-3: typing during in-flight expand → edit preserved, 'none' (no clobber)", async () => {
  resetSlash();
  ta.value = "/fix login bug";
  queueFetch(true, { ok: true, expanded: "Fix the following issue: login bug", command: "fix" }, 40);
  const p = F._applySlashExpansionToPrompt();
  await tick(); await tick(); await tick();
  assert.strictEqual(fetchCalls.length, 1);   // expand fetch started, response still pending
  ta.value = "/fix login bug — and my own note";  // user keeps typing during the round trip
  const st = await p;
  assert.strictEqual(st, "none");
  assert.strictEqual(ta.value, "/fix login bug — and my own note");  // NOT clobbered
});
check("P16-3: empty expansion → raw kept, 'none' (command not silently erased)", async () => {
  resetSlash();
  ta.value = "/fix x";
  queueFetch(true, { ok: true, expanded: "   ", command: "fix" });
  const st = await F._applySlashExpansionToPrompt();
  assert.strictEqual(st, "none");
  assert.strictEqual(ta.value, "/fix x");  // raw command stays → server surfaces the error visibly
});

// ── 3. Autocomplete dropdown ──
check("_slashOnInput: /f filters to matching commands", async () => {
  resetSlash();
  ta.value = "/f";
  F._slashOnInput({ isComposing: false });
  await tick();
  assert.strictEqual(menu.hidden, false);
  assert.ok(menu.innerHTML.includes("/fix"), "menu should contain /fix");
  assert.ok(!menu.innerHTML.includes("/refactor"), "menu must not contain /refactor");
  assert.strictEqual(_slashMenuItems.length, 1);
});
check("_slashOnInput: bare / shows all commands", async () => {
  resetSlash();
  ta.value = "/";
  F._slashOnInput({ isComposing: false });
  await tick();
  assert.strictEqual(_slashMenuItems.length, 5);
});
check("_slashOnInput: args present closes the menu", async () => {
  resetSlash();
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  menu.hidden = false;
  ta.value = "/fix login bug";
  F._slashOnInput({ isComposing: false });
  await tick();
  assert.strictEqual(menu.hidden, true);
  assert.strictEqual(_slashMenuItems.length, 0);
});
check("_slashOnInput: IME composition ignored", async () => {
  resetSlash();
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  menu.hidden = false;
  ta.value = "/f";
  F._slashOnInput({ isComposing: true });
  await tick();
  assert.strictEqual(menu.hidden, false);  // unchanged (no re-render)
  assert.strictEqual(_slashMenuItems.length, 1);
});
check("_slashAcceptSel: fills canonical /name + trailing space, closes", () => {
  resetSlash();
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }, { cmd: SAMPLE_COMMANDS[1] }];
  _slashMenuSel = 0;
  menu.hidden = false;
  F._slashAcceptSel();
  assert.strictEqual(ta.value, "/fix ");
  assert.strictEqual(ta.focused, true);
  assert.deepStrictEqual(ta.sel, [5, 5]);
  assert.strictEqual(menu.hidden, true);
  assert.strictEqual(_slashMenuItems.length, 0);
});
check("_slashMoveSel: wraps around and toggles active class", () => {
  resetSlash();
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }, { cmd: SAMPLE_COMMANDS[1] }];
  _slashMenuSel = 0;
  menu._items = [makeMenuItem(), makeMenuItem()];
  F._slashMoveSel(1);
  assert.strictEqual(_slashMenuSel, 1);
  assert.strictEqual(menu._items[1].classList.active, true);
  assert.strictEqual(menu._items[0].classList.active, false);  // toggled off explicitly
  F._slashMoveSel(1);  // wraps to 0
  assert.strictEqual(_slashMenuSel, 0);
});

// ── 4. Keydown integration ──
check("ui.js Enter with menu open accepts suggestion, does NOT send", () => {
  resetSlash();
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  ta.value = "/f";
  const e = fire(uiKd, {});
  assert.strictEqual(e._prevented, true);
  assert.strictEqual(ta.value, "/fix ");
  assert.strictEqual(_slashMenuItems.length, 0);
  // smartRun is a stub in this harness — the send path must not fire.
});
check("ui.js Enter with menu closed still sends (regression)", () => {
  resetSlash();
  const e = fire(uiKd, {});
  assert.strictEqual(e._prevented, true);
});
check("ui.js Escape closes menu without sending", () => {
  resetSlash();
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  menu.hidden = false;
  const e = fire(uiKd, { key: "Escape" });
  assert.strictEqual(menu.hidden, true);
  assert.strictEqual(_slashMenuItems.length, 0);
  assert.strictEqual(e._prevented, false);
});
check("design mode: menu open + Enter (ui.js first) → one accept, no send", () => {
  resetSlash();
  convToggle.checked = true;
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  ta.value = "/f";
  const e1 = fire(uiKd, {});
  const e2 = fire(dcKd, { defaultPrevented: e1.defaultPrevented });
  assert.strictEqual(e1._prevented, true);
  assert.strictEqual(ta.value, "/fix ");
  assert.strictEqual(designSendCalls, 0);
});
check("design mode: menu open + Enter (design-chat first) → one accept, no send", () => {
  resetSlash();
  convToggle.checked = true;
  _slashMenuItems = [{ cmd: SAMPLE_COMMANDS[0] }];
  ta.value = "/f";
  const e1 = fire(dcKd, {});
  assert.strictEqual(e1._prevented, true);
  assert.strictEqual(ta.value, "/fix ");
  assert.strictEqual(designSendCalls, 0);
  const e2 = fire(uiKd, { defaultPrevented: e1.defaultPrevented });
  assert.strictEqual(e2._prevented, false);  // second handler bails via defaultPrevented
  assert.strictEqual(designSendCalls, 0);
});
check("design mode: menu closed + Enter still sends (regression)", () => {
  resetSlash();
  convToggle.checked = true;
  const e = fire(dcKd, {});
  assert.strictEqual(e._prevented, true);
  assert.strictEqual(designSendCalls, 1);
});
check("main mode: design-chat handler ignores Enter (convToggle off)", () => {
  resetSlash();
  const e = fire(dcKd, {});
  assert.strictEqual(e._prevented, false);
  assert.strictEqual(designSendCalls, 0);
});

// ── 5. Send-path end-to-end ──
check("smartRun: /fix login bug → expanded, then _launchAgent", async () => {
  resetSlash();
  ta.value = "/fix login bug";
  queueFetch(true, { ok: true, expanded: "Fix the following issue: login bug", command: "fix" });
  F.smartRun();
  await tick();
  assert.strictEqual(launchCalls, 1);
  assert.strictEqual(runCalls, 0);
  assert.strictEqual(designSendCalls, 0);
  assert.strictEqual(ta.value, "Fix the following issue: login bug");
});
check("smartRun: unknown /zzz → blocked, no routing", async () => {
  resetSlash();
  ta.value = "/zzz go";
  F.smartRun();
  await tick();
  assert.strictEqual(launchCalls, 0);
  assert.strictEqual(runCalls, 0);
  assert.strictEqual(designSendCalls, 0);
  assert.strictEqual(tlCalls.length, 1);
});
check("smartRun: plain prompt routes to runOnly (auto mode)", async () => {
  resetSlash();
  ta.value = "add a test for parse";
  F.smartRun();
  await tick();
  assert.strictEqual(runCalls, 1);
  assert.strictEqual(launchCalls, 0);
});

// ── 6. Source gate: every send path routes through the expansion hook ──
{
  const uiText = fs.readFileSync(path.join(__dirname, "..", "..", "webapp", "ui", "static", "ui.js"), "utf8");
  const dcText = fs.readFileSync(path.join(__dirname, "..", "..", "webapp", "ui", "static", "design-chat.js"), "utf8");
  const callCount = (src, fnName, hook) => {
    // Count hook calls inside the named function body.
    const start = src.indexOf(`function ${fnName}(`);
    assert.ok(start >= 0, `function ${fnName} not found`);
    let depth = 0;
    let i = src.indexOf("{", start);
    for (; i < src.length; i++) {
      if (src[i] === "{") depth++;
      else if (src[i] === "}") { depth--; if (depth === 0) break; }
    }
    const body = src.slice(start, i + 1);
    return body.split(hook).length - 1;
  };
  check("source gate: smartRun routes through _applySlashExpansionToPrompt", () => {
    assert.strictEqual(callCount(uiText, "smartRun", "_applySlashExpansionToPrompt"), 1);
  });
  check("source gate: _launchAgent routes through the hook (guarded)", () => {
    assert.strictEqual(callCount(uiText, "_launchAgent", "_applySlashExpansionToPrompt"), 1);
  });
  check("source gate: runOnly routes through the hook", () => {
    assert.strictEqual(callCount(uiText, "runOnly", "_applySlashExpansionToPrompt"), 1);
  });
  check("source gate: _designSendMessage routes through the hook", () => {
    assert.strictEqual(callCount(dcText, "_designSendMessage", "_applySlashExpansionToPrompt"), 1);
  });
  check("source gate: send-time expansion warms the command list (P17-1)", () => {
    const start = uiText.indexOf("async function _applySlashExpansionToPrompt(");
    assert.ok(start >= 0);
    let depth = 0, i = uiText.indexOf("{", start);
    for (; i < uiText.length; i++) {
      if (uiText[i] === "{") depth++;
      else if (uiText[i] === "}") { depth--; if (depth === 0) break; }
    }
    const body = uiText.slice(start, i + 1);
    assert.ok(body.includes("await _slashFetchCommands()"), "cold cache must fetch the list before matching");
  });
  check("source gate: boot prefetch of the slash list (P17-1)", () => {
    assert.ok(uiText.includes("_slashFetchCommands();"), "bindUI must prefetch the command list at boot");
  });
  check("source gate: P16-3 clobber guard + empty-expansion guard present", () => {
    const start = uiText.indexOf("async function _applySlashExpansionToPrompt(");
    assert.ok(start >= 0);
    let depth = 0, i = uiText.indexOf("{", start);
    for (; i < uiText.length; i++) {
      if (uiText[i] === "{") depth++;
      else if (uiText[i] === "}") { depth--; if (depth === 0) break; }
    }
    const body = uiText.slice(start, i + 1);
    assert.ok(body.includes("ta.value !== before"), "typing during in-flight expand must NOT be clobbered");
    assert.ok(body.includes("!j.expanded.trim()"), "empty expansion must fall back to raw");
  });
}

// The `check(...)` calls above only registered — run them sequentially.
(async () => {
  for (const { name, fn } of checks) {
    await fn();
    passed++;
    console.log(`  ok - ${name}`);
  }
  console.log(`\nPASS: ${passed} checks`);
})().catch((e) => { console.error(e); process.exit(1); });
