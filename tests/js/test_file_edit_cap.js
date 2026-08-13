#!/usr/bin/env node
/**
 * Regression harness — P18-1: truncated file preview must never be saved over
 * the original (silent data loss of everything past the 200 KiB GET cap).
 *
 * Bug: openFile() never read the server's `truncated` flag, so a 600 KB file
 * opened in the preview was fully editable; _saveEditedFile() POSTed the
 * truncated text back and the server overwrote the whole file — the last
 * 400 KB disappeared with a cheerful "Saved" toast.
 *
 * Fix under test (REAL function text sliced from ui.js):
 *   - openFile() records state._fileTruncated / state._fileSize from the GET
 *     response (reset at the start of every open).
 *   - _enterEditMode() refuses to enter edit mode for a truncated preview.
 *   - _saveEditedFile() sends the optimistic-concurrency token base_size and
 *     restores the file's trailing newline (innerText drops it).
 * Run: node tests/js/test_file_edit_cap.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "ui.js"), "utf8");

function sliceFunction(s, anchor) {
  const start = s.indexOf(anchor);
  assert.ok(start >= 0, `anchor ${anchor} not found`);
  const open = s.indexOf("{", start);
  assert.ok(open >= 0, `no body for ${anchor}`);
  let depth = 0;
  let i = open;
  for (; i < s.length; i++) {
    if (s[i] === "{") depth++;
    else if (s[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  assert.ok(i < s.length, `unbalanced braces while extracting ${anchor}`);
  return s.slice(start, i + 1);
}

const openFnText = sliceFunction(src, "async function openFile(rel, clickedRowElOrOpts = null)");
const enterFnText = sliceFunction(src, "function _enterEditMode()");
const saveFnText = sliceFunction(src, "async function _saveEditedFile()");

// ── stub environment ──
let tl = [];               // addTL captures
let lastApiUrl = null;
let lastSaveBody = null;
let saveCalls = 0;
let saveOkResponse = { ok: true, rel: "x", bytes: 5 };
let rendered = null;

function addTL(level, msg) { tl.push({ level, msg }); }
function currentRepoRoot() { return "/tmp/repo"; }
function pushRecent() {}
function setPreviewTitle() {}
function ensurePromptTargetFileVisible() {}
function highlightTreeFile() {}
function finalizeProgressRows() {}
function renderFileText(text, name) { rendered = { text, name }; }
function diffLooksLikeTouchesFile() { return false; }
function _fetchTimeoutSignal() { return undefined; }

function makeState() {
  return {
    selectedRel: null,
    selectedTreeRow: null,
    lastRunResult: null,
    previewMode: null,
    _editMode: false,
    _editEl: null,
    _currentFileText: "",
    _fileTruncated: false,
    _fileSize: null,
  };
}

function makePre() {
  return {
    contentEditable: undefined,
    spellcheck: undefined,
    style: {},
    _listeners: {},
    setAttribute() {},
    addEventListener(type, fn) { this._listeners[type] = fn; },
    removeEventListener(type) { delete this._listeners[type]; },
  };
}

let saveBtnShown = null;
function makeEnv(state, pre, fileData, innerText) {
  const saveBtn = { style: { display: "none" } };
  saveBtnShown = null;
  const getEl = (id) => {
    if (id === "file-preview") return { focus() {}, querySelector: (sel) => (sel === "pre.code-view" ? pre : null) };
    if (id === "save-file-btn") return saveBtn;
    if (id === "tab-preview") return { click() {} };
    return null;
  };
  const state2 = state || makeState();
  state2._editEl = { innerText: innerText !== undefined ? innerText : "line1", focus() {} };
  return {
    state: state2,
    el: getEl,
    apiGet: async (url) => { lastApiUrl = url; return fileData; },
    fetch: async (url, opts) => {
      if (typeof url === "string" && url.startsWith("/ui/api/repo/file")) {
        saveCalls++;
        lastSaveBody = JSON.parse((opts && opts.body) || "{}");
        return { ok: true, status: 200, json: async () => saveOkResponse };
      }
      throw new Error("unexpected fetch: " + url);
    },
    window: { getSelection: () => ({ rangeCount: 0 }) },
    Element: class {},
    saveBtn,
  };
}

function load(fns) {
  return new Function(
    "currentRepoRoot", "addTL", "pushRecent", "setPreviewTitle", "ensurePromptTargetFileVisible",
    "highlightTreeFile", "finalizeProgressRows", "renderFileText", "diffLooksLikeTouchesFile",
    "_fetchTimeoutSignal", "apiGet", "fetch", "el", "state", "window", "Element",
    fns.open + "\n" + fns.enter + "\n" + fns.save + "\n" + "return { openFile, _enterEditMode, _saveEditedFile };"
  );
}

function loadAll(state, pre, fileData, innerText) {
  const env = makeEnv(state, pre, fileData, innerText);
  const loaded = load({ open: openFnText, enter: enterFnText, save: saveFnText })(
    currentRepoRoot, addTL, pushRecent, setPreviewTitle,
    ensurePromptTargetFileVisible, highlightTreeFile, finalizeProgressRows, renderFileText,
    diffLooksLikeTouchesFile, _fetchTimeoutSignal, env.apiGet, env.fetch, env.el, env.state, env.window, env.Element);
  return { loaded, state: env.state, getSaveDisplay: () => env.saveBtn.style.display };
}

let failures = 0;
function check(cond, msg) {
  if (!cond) { failures++; console.error("FAIL: " + msg); }
}

// ── 1. truncated preview → edit mode blocked ──
{
  const pre = makePre();
  const { loaded, state, getSaveDisplay } = loadAll(null, pre, { ok: true, text: "abc", size: 600000, truncated: true });
  (async () => {
    await loaded.openFile("big.py");
    check(state._fileTruncated === true, "truncated preview sets state._fileTruncated");
    check(state._fileSize === 600000, "truncated preview records size");
    loaded._enterEditMode();
    check(pre.contentEditable !== "true", "edit mode must NOT be entered for truncated preview");
    check(state._editMode !== true, "state._editMode must stay false");
    const warn = tl.find(t => t.level === "warn" && /truncated/i.test(t.msg));
    check(!!warn, "truncated refusal surfaces an addTL warning");
    check(getSaveDisplay() === "none", "save button must not be shown for truncated preview");
  })().then(runNext, e => { console.error("FAIL(1) threw:", e); process.exit(1); });
}

let step = 1;
function runNext() {
  step++;
  if (step === 2) test2();
  else if (step === 3) test3();
  else if (step === 4) test4();
  else if (step === 5) test5();
  else if (step === 6) test6();
  else {
    console.log(failures === 0 ? "ALL PASS" : `${failures} FAILURES`);
    process.exit(failures === 0 ? 0 : 1);
  }
}

// ── 2. full preview → edit mode works, file size + truncated reset per open ──
function test2() {
  const pre = makePre();
  const { loaded, state, getSaveDisplay } = loadAll(null, pre, { ok: true, text: "hello", size: 5, truncated: false });
  (async () => {
    await loaded.openFile("ok.py");
    check(state._fileTruncated === false, "full preview clears truncated flag");
    check(state._fileSize === 5, "full preview records size");
    loaded._enterEditMode();
    check(pre.contentEditable === "true", "edit mode entered for full preview");
    check(state._editMode === true, "state._editMode true");
    check(getSaveDisplay() === "", "save button shown in edit mode");
  })().then(runNext, e => { console.error("FAIL(2) threw:", e); process.exit(1); });
}

// ── 3. save sends base_size + restores trailing newline ──
function test3() {
  const state = makeState();
  state.selectedRel = "ok.py";
  state._currentFileText = "line1\n";      // file ends with newline
  state._fileSize = 6;
  const pre = makePre();
  const { loaded } = loadAll(state, pre, { ok: true, text: "line1\n", size: 6, truncated: false }, "line1");
  (async () => {
    await loaded._saveEditedFile();
    check(lastSaveBody !== null, "save POST issued");
    check(lastSaveBody.base_size === 6, "base_size sent from preview size");
    check(lastSaveBody.content === "line1\n", "trailing newline restored in saved content");
    check(state._fileSize === 5, "fileSize updated from server bytes");
    const ok = tl.find(t => t.level === "success" && /Saved/.test(t.msg));
    check(!!ok, "success toast shown");
  })().then(runNext, e => { console.error("FAIL(3) threw:", e); process.exit(1); });
}

// ── 4. no trailing newline in original → content untouched ──
function test4() {
  const state = makeState();
  state.selectedRel = "x.py";
  state._currentFileText = "line1";        // no trailing newline
  state._fileSize = 5;
  const pre = makePre();
  const { loaded } = loadAll(state, pre, null, "line1");
  (async () => {
    await loaded._saveEditedFile();
    check(lastSaveBody.content === "line1", "content unchanged when original lacks trailing newline");
    check(lastSaveBody.base_size === 5, "base_size still sent");
  })().then(runNext, e => { console.error("FAIL(4) threw:", e); process.exit(1); });
}

// ── 5. no size known → base_size omitted ──
function test5() {
  const state = makeState();
  state.selectedRel = "y.py";
  state._currentFileText = "";
  state._fileSize = null;
  const pre = makePre();
  const { loaded } = loadAll(state, pre, null, "z");
  (async () => {
    await loaded._saveEditedFile();
    check(lastSaveBody.base_size === undefined, "base_size omitted when size unknown");
  })().then(runNext, e => { console.error("FAIL(5) threw:", e); process.exit(1); });
}

// ── 6. source gates ──
function test6() {
  check(openFnText.includes("state._fileTruncated = !!data.truncated"), "openFile records truncated flag");
  check(openFnText.includes("state._fileSize = (typeof data.size"), "openFile records file size");
  check(enterFnText.includes("state._fileTruncated"), "_enterEditMode guards on truncated flag");
  check(saveFnText.includes("base_size"), "_saveEditedFile sends base_size");
  check(saveFnText.includes('content += "\\n"'), "_saveEditedFile restores trailing newline");
  runNext();
}

console.log(`${failures} failures so far (async) — waiting for completions...`);
