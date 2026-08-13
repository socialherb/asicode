#!/usr/bin/env node
/**
 * Regression harness — P15-2: the design-chat image attach path checks file
 * size BEFORE readAsDataURL.
 *
 * Bug: _designAttachFile read ANY image File into memory via readAsDataURL
 * (base64, ~4/3x the file size) with no size check — a 500 MB screenshot
 * paste/drop caused a multi-hundred-MB memory spike and only then a server
 * 413. The server cap (_DESIGN_IMAGE_MAX_BYTES = 10 MiB) exists but is
 * reachable only after the full body has been read.
 *
 * Fix under test: _designAttachFile rejects files > _DESIGN_ATTACH_MAX_BYTES
 * (10 MiB, mirroring the server) with an addTL error, before any FileReader
 * work; attachment state and preview rendering are untouched.
 *
 * The REAL function text is sliced out of design-chat.js (brace-balanced) and
 * executed against stub globals (no test framework, no browser).
 * Run: node tests/js/test_design_image_attach_cap.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "design-chat.js"), "utf8");

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

const fnText = sliceFunction(src, "function _designAttachFile(file)");

// ── stub environment ──
let attached = [];
let lastError = null;
let readerStarted = null;

const _designChat = { attachedImages: attached, attachedTotalBytes: 0 };
function _designRenderImagePreviews() {}
function addTL(level, msg) { if (level === "error") lastError = msg; }

class FakeFileReader {
  readAsDataURL(file) {
    readerStarted = file;
    this.onload({ target: { result: "data:image/png;base64,AAAA" } });
  }
}

// Must mirror the source constant; the source gate below pins the real value.
const _DESIGN_ATTACH_MAX_BYTES = 10 * 1024 * 1024;
// P19-2 mirrors: count + total-budget caps (source gates below pin the values).
const _DESIGN_ATTACH_MAX_COUNT = 8;
const _DESIGN_ATTACH_TOTAL_MAX_BYTES = 20 * 1024 * 1024;

const run = new Function(
  "_designChat", "_designRenderImagePreviews", "addTL", "FileReader", "_DESIGN_ATTACH_MAX_BYTES",
  "_DESIGN_ATTACH_MAX_COUNT", "_DESIGN_ATTACH_TOTAL_MAX_BYTES", "_designAttachSeq",
  `${fnText}; return _designAttachFile;`
);
const _designAttachFile = run(
  _designChat, _designRenderImagePreviews, addTL, FakeFileReader, _DESIGN_ATTACH_MAX_BYTES,
  _DESIGN_ATTACH_MAX_COUNT, _DESIGN_ATTACH_TOTAL_MAX_BYTES, 0
);

function reset() {
  attached = [];
  lastError = null;
  readerStarted = null;
  _designChat.attachedImages = attached;
  _designChat.attachedTotalBytes = 0;
}

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gates ──
check("size guard appears before readAsDataURL in the real function",
  fnText.indexOf("file.size >") < fnText.indexOf("reader.readAsDataURL(file)"),
  "guard must run before any FileReader work");
const constLine = src.split("\n").find((l) => l.includes("_DESIGN_ATTACH_MAX_BYTES ="));
check("attach cap constant declared in design-chat.js", !!constLine, "constant missing");
check("attach cap is 10 MiB (mirrors server _DESIGN_IMAGE_MAX_BYTES)",
  !!constLine && constLine.includes("10 * 1024 * 1024"), constLine || "");
const countLine = src.split("\n").find((l) => l.includes("_DESIGN_ATTACH_MAX_COUNT ="));
check("P19-2: attach count cap constant declared", !!countLine, "constant missing");
check("P19-2: attach count cap is 8", !!countLine && countLine.includes("= 8"), countLine || "");
const totalLine = src.split("\n").find((l) => l.includes("_DESIGN_ATTACH_TOTAL_MAX_BYTES ="));
check("P19-2: attach total-bytes cap constant declared", !!totalLine, "constant missing");
check("P19-2: attach total cap is 20 MiB",
  !!totalLine && totalLine.includes("20 * 1024 * 1024"), totalLine || "");
check("P19-2: count guard runs before readAsDataURL",
  fnText.indexOf("_DESIGN_ATTACH_MAX_COUNT") < fnText.indexOf("reader.readAsDataURL(file)"),
  "count guard must run before any FileReader work");
check("P19-2: total guard runs before readAsDataURL",
  fnText.indexOf("_DESIGN_ATTACH_TOTAL_MAX_BYTES") < fnText.indexOf("reader.readAsDataURL(file)"),
  "total guard must run before any FileReader work");

// ── behavior: oversize rejection ──
reset();
_designAttachFile({ name: "huge.png", type: "image/png", size: _DESIGN_ATTACH_MAX_BYTES + 1 });
check("oversize file is not attached", attached.length === 0, `attached=${attached.length}`);
check("oversize file never reaches readAsDataURL", readerStarted === null, String(readerStarted));
check("oversize file surfaces an addTL error naming the limit",
  typeof lastError === "string" && /10 MiB/.test(lastError), lastError || "no error");
check("oversize error names the file", lastError.includes("huge.png"), lastError || "no error");

// ── boundary: exactly the cap passes ──
reset();
_designAttachFile({ name: "exact.png", type: "image/png", size: _DESIGN_ATTACH_MAX_BYTES });
check("exact-cap file is accepted", readerStarted !== null && readerStarted.name === "exact.png", "readAsDataURL not called");
check("exact-cap file renders a preview slot",
  attached.length === 1 && attached[0] && attached[0].dataUrl === "data:image/png;base64,AAAA",
  `attached=${JSON.stringify(attached)}`);
check("exact-cap file does not error", lastError === null, lastError || "unexpected error");

// ── small file passes untouched ──
reset();
_designAttachFile({ name: "small.jpg", type: "image/jpeg", size: 4096 });
check("small file is accepted", readerStarted !== null && readerStarted.name === "small.jpg", "readAsDataURL not called");
check("small file preserves mediaType/name",
  attached[0].mediaType === "image/jpeg" && attached[0].name === "small.jpg",
  `attached=${JSON.stringify(attached[0])}`);
check("small file does not error", lastError === null, lastError || "unexpected error");

// ── non-image still ignored (pre-existing contract) ──
reset();
_designAttachFile({ name: "notes.txt", type: "text/plain", size: 10 });
check("non-image file is still ignored", attached.length === 0 && readerStarted === null, "");

console.log(`P15-2 design image attach cap gate: ${passed} checks PASS`);
