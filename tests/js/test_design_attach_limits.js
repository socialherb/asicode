#!/usr/bin/env node
/**
 * Regression harness — P19-2: design-chat attachments are bounded by COUNT
 * (8) and TOTAL BYTES (20 MiB) before any FileReader work, removal releases
 * the budget share, pending placeholder slots never render/upload, and image
 * uploads run with bounded concurrency (2) while preserving attachment order.
 *
 * Bug: _designAttachFile only checked the per-file 10 MiB cap — a 50-image
 * paste/drop base64-inflated up to ~500 MiB into the tab before a byte was
 * sent (server store is capped at 100 MiB total, but the client was
 * unbounded), and _designUploadImages fired EVERY upload simultaneously via
 * Promise.all (N concurrent connections + server upload work).
 *
 * The REAL function text of _designAttachFile, _designRenderImagePreviews and
 * _designUploadImages is sliced out of design-chat.js (brace-balanced) and
 * executed against stub globals.
 * Run: node tests/js/test_design_attach_limits.js
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

const attachFnText = sliceFunction(src, "function _designAttachFile(file)");
const renderFnText = sliceFunction(src, "function _designRenderImagePreviews()");
const upFnText = sliceFunction(src, "async function _designUploadImages(imagesToUpload)");
const sendFnText = sliceFunction(src, "async function _designSendMessage()");

// ── stub environment ──
let lastError = null;
let attached = [];
let readerStarted = null;
let readerDeferred = null;   // set when a deferred FileReader is used
const readers = [];          // created FileReader stubs (for deferred onload)

const _designChat = { attachedImages: attached, attachedTotalBytes: 0 };
function addTL(level, msg) { if (level === "error") lastError = msg; }

const _DESIGN_ATTACH_MAX_BYTES = 10 * 1024 * 1024;
const _DESIGN_ATTACH_MAX_COUNT = 8;
const _DESIGN_ATTACH_TOTAL_MAX_BYTES = 20 * 1024 * 1024;

class FakeFileReader {
  readAsDataURL(file) {
    readerStarted = file;
    readers.push(this);
    if (readerDeferred) return; // test drives onload manually
    this.onload({ target: { result: "data:image/png;base64,AAAA" } });
  }
}

// DOM stub for the real render function (thumbs + remove buttons)
const created = [];
const bar = { style: {}, innerHTML: "", appendChild() {} };
const documentStub = {
  getElementById: (id) => (id === "design-image-preview" ? bar : null),
  createElement: () => {
    const el = {
      className: "", style: {}, textContent: "", title: "", src: "", alt: "",
      onclick: null, appendChild() {},
    };
    created.push(el);
    return el;
  },
};

function makeRender() {
  return new Function(
    "document", "_designChat",
    `${renderFnText}; return _designRenderImagePreviews;`
  )(documentStub, _designChat);
}
const _designRenderImagePreviews = makeRender();

function makeAttach() {
  return new Function(
    "_designChat", "_designRenderImagePreviews", "addTL", "FileReader",
    "_DESIGN_ATTACH_MAX_BYTES", "_DESIGN_ATTACH_MAX_COUNT", "_DESIGN_ATTACH_TOTAL_MAX_BYTES",
    "_designAttachSeq",
    `${attachFnText}; return _designAttachFile;`
  )(
    _designChat, _designRenderImagePreviews, addTL, FakeFileReader,
    _DESIGN_ATTACH_MAX_BYTES, _DESIGN_ATTACH_MAX_COUNT, _DESIGN_ATTACH_TOTAL_MAX_BYTES, 0
  );
}
const _designAttachFile = makeAttach();

function reset() {
  attached = [];
  lastError = null;
  readerStarted = null;
  readerDeferred = null;
  readers.length = 0;
  created.length = 0;
  _designChat.attachedImages = attached;
  _designChat.attachedTotalBytes = 0;
}

function img(name, size) {
  return { name, type: "image/png", size };
}

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gates ──
check("attach count guard precedes readAsDataURL",
  attachFnText.indexOf("_DESIGN_ATTACH_MAX_COUNT") < attachFnText.indexOf("reader.readAsDataURL(file)"),
  "count guard must run before FileReader work");
check("attach total guard precedes readAsDataURL",
  attachFnText.indexOf("_DESIGN_ATTACH_TOTAL_MAX_BYTES") < attachFnText.indexOf("reader.readAsDataURL(file)"),
  "total guard must run before FileReader work");
check("send path filters pending placeholder slots (dataUrl required)",
  sendFnText.includes("filter(img => img && img.dataUrl)"), "placeholder filter missing");
check("send path releases the total budget after splice",
  sendFnText.includes("attachedTotalBytes = 0"), "budget reset missing");
check("render skips pending placeholder slots",
  renderFnText.includes("!img.dataUrl"), "render still renders placeholders");
check("remove handler releases the budget share",
  renderFnText.includes("attachedTotalBytes") && renderFnText.includes("removed.bytes"),
  "remove decrement missing");
check("upload helper bounds concurrency to 2",
  upFnText.includes("_UPLOAD_CONCURRENCY = 2"), "concurrency cap missing");
check("upload helper preserves original order via index slots",
  upFnText.includes("const results = new Array(imagesToUpload.length)"), "order-preserving pool missing");
check("P20-2: attach onload re-locates its slot by token (not index)",
  attachFnText.includes("findIndex(s => s && s.token === token)"), "token lookup missing in onload");
check("P20-2: remove button re-locates by token",
  renderFnText.includes("findIndex(s => s && s.token === img.token)"), "token lookup missing in remove");
check("P20-3: FileReader onerror handler exists", attachFnText.includes("reader.onerror"), "onerror missing");
check("P20-3: FileReader onabort handler exists", attachFnText.includes("reader.onabort"), "onabort missing");
check("P20-2: send gate counts only ready (dataUrl) images",
  sendFnText.includes("_readyImages"), "ready-only send gate missing");

// ── count cap ──
reset();
for (let i = 0; i < 8; i++) _designAttachFile(img(`n${i}.png`, 1024));
check("count: 8 attachments accepted", attached.length === 8, `attached=${attached.length}`);
check("count: total budget tracks all 8", _designChat.attachedTotalBytes === 8 * 1024, String(_designChat.attachedTotalBytes));
_designAttachFile(img("ninth.png", 1024));
check("count: 9th attachment rejected", attached.length === 8, `attached=${attached.length}`);
check("count: rejection names the limit", /Maximum 8 images/.test(lastError || ""), lastError || "no error");
check("count: rejected file never reaches FileReader",
  readerStarted === null || readerStarted.name !== "ninth.png", String(readerStarted && readerStarted.name));

// ── total-bytes cap (and exact boundary) ──
reset();
_designAttachFile(img("a.png", 9 * 1024 * 1024));
_designAttachFile(img("b.png", 9 * 1024 * 1024));
_designAttachFile(img("c.png", 3 * 1024 * 1024));
check("total: 9MiB + 9MiB + 3MiB rejected", attached.length === 2, `attached=${attached.length}`);
check("total: rejection names the limit", /20 MiB limit/.test(lastError || ""), lastError || "no error");
reset();
_designAttachFile(img("a.png", 9 * 1024 * 1024));
_designAttachFile(img("b.png", 9 * 1024 * 1024));
_designAttachFile(img("c.png", 2 * 1024 * 1024));
check("total: 9MiB + 9MiB + 2MiB accepted (exact boundary)", attached.length === 3, `attached=${attached.length}`);
check("total: budget is exactly 20MiB", _designChat.attachedTotalBytes === 20 * 1024 * 1024, String(_designChat.attachedTotalBytes));

// ── removal releases the budget share (via the real render's remove button) ──
reset();
_designAttachFile(img("x.png", 10 * 1024 * 1024));
_designAttachFile(img("y.png", 10 * 1024 * 1024));
created.length = 0; // renders accumulate; snapshot only the explicit render below
_designRenderImagePreviews();
// per attached image the render creates: [item, img, removeBtn] — y's button is index 5
check("remove: both thumbs rendered", created.length === 6, `created=${created.length}`);
created[5].onclick(); // remove y
check("remove: list shrinks to 1", attached.length === 1, `attached=${attached.length}`);
check("remove: budget releases y's share", _designChat.attachedTotalBytes === 10 * 1024 * 1024, String(_designChat.attachedTotalBytes));
_designAttachFile(img("z.png", 10 * 1024 * 1024));
check("remove: freed budget is reusable", attached.length === 2, `attached=${attached.length}`);

// ── pending placeholder slots: never rendered, never uploaded ──
reset();
readerDeferred = true;
_designAttachFile(img("pending.png", 2048));
check("placeholder: slot holds bytes + null dataUrl",
  attached.length === 1 && attached[0].bytes === 2048 && attached[0].dataUrl === null,
  JSON.stringify(attached));
check("placeholder: budget counted at attach time", _designChat.attachedTotalBytes === 2048, String(_designChat.attachedTotalBytes));
_designRenderImagePreviews();
check("placeholder: render creates NO thumb for pending slot", created.length === 0, `created=${created.length}`);
readers[0].onload({ target: { result: "data:image/png;base64,FULL" } });
check("placeholder: onload replaces the slot with the loaded object",
  attached[0].dataUrl === "data:image/png;base64,FULL" && attached[0].bytes === 2048,
  JSON.stringify(attached[0]));
created.length = 0; // onload already rendered; snapshot the explicit render below
_designRenderImagePreviews();
check("placeholder: loaded slot now renders", created.length === 3, `created=${created.length}`);

// ── P20-2: token-based slots — splice interactions ──
reset();
readerDeferred = true;
_designAttachFile(img("A.png", 2048));
_designAttachFile(img("B.png", 2048));
_designAttachFile(img("C.png", 2048));
check("token: three pending slots carry distinct tokens",
  attached[0].token !== attached[1].token && attached[1].token !== attached[2].token,
  JSON.stringify(attached.map(s => s.token)));
readers[0].onload({ target: { result: "data:image/png;base64,A" } });
created.length = 0;
_designRenderImagePreviews();
created[2].onclick(); // remove A while B and C are still being read
check("token: removing A shifts the array", attached.length === 2, `attached=${attached.length}`);
check("token: budget released for A", _designChat.attachedTotalBytes === 4096, String(_designChat.attachedTotalBytes));
readers[1].onload({ target: { result: "data:image/png;base64,B" } });
readers[2].onload({ target: { result: "data:image/png;base64,C" } });
check("token: late onloads land on THEIR OWN slots after splice",
  attached.length === 2
    && attached[0].dataUrl === "data:image/png;base64,B"
    && attached[1].dataUrl === "data:image/png;base64,C",
  JSON.stringify(attached));
check("token: no placeholder survives the shift", attached.every(s => s && s.dataUrl), JSON.stringify(attached));

// P20-2: send mid-read — splice(0) + budget reset, then a late onload must not
// resurrect a slot (the old code wrote into the cleared array, creating a
// sparse hole that silently rode along with the NEXT message).
reset();
readerDeferred = true;
_designAttachFile(img("X.png", 2048));
const sent = _designChat.attachedImages.splice(0); // what _designSendMessage does
_designChat.attachedTotalBytes = 0;
check("send-mid-read: snapshot taken before splice", sent.length === 1 && sent[0].dataUrl === null, JSON.stringify(sent));
readers[0].onload({ target: { result: "data:image/png;base64,LATE" } });
check("send-mid-read: late onload does NOT resurrect a slot",
  _designChat.attachedImages.length === 0, `attached=${_designChat.attachedImages.length}`);
check("send-mid-read: budget stays zero", _designChat.attachedTotalBytes === 0, String(_designChat.attachedTotalBytes));

// ── P20-3: FileReader failure / abort releases the slot + budget ──
reset();
readerDeferred = true;
_designAttachFile(img("broken.png", 4096));
check("onerror: slot reserved while reading", attached.length === 1 && _designChat.attachedTotalBytes === 4096, "");
readers[0].onerror();
check("onerror: failed slot is removed", attached.length === 0, `attached=${attached.length}`);
check("onerror: budget released", _designChat.attachedTotalBytes === 0, String(_designChat.attachedTotalBytes));
check("onerror: user gets an error toast naming the file",
  /Could not read "broken.png"/.test(lastError || ""), lastError || "no error");
_designAttachFile(img("after.png", 4096));
check("onerror: freed count slot is reusable", attached.length === 1, `attached=${attached.length}`);

reset();
readerDeferred = true;
_designAttachFile(img("aborted.png", 4096));
readers[0].onabort();
check("onabort: slot removed and budget released",
  attached.length === 0 && _designChat.attachedTotalBytes === 0,
  `attached=${attached.length} total=${_designChat.attachedTotalBytes}`);

// ── upload concurrency: at most 2 in flight, order preserved, failures counted ──
function makeUploadEnv(okByBodyName) {
  let inFlight = 0;
  let maxInFlight = 0;
  const makeFetch = (url, opts) => {
    const name = (JSON.parse(opts && opts.body || "{}").name) || "?";
    inFlight++;
    maxInFlight = Math.max(maxInFlight, inFlight);
    return new Promise((resolve) => setTimeout(() => {
      inFlight--;
      if (okByBodyName && !okByBodyName(name)) return resolve({ ok: false, status: 413, json: async () => ({}) });
      resolve({ ok: true, status: 200, json: async () => ({ image_id: `id-${name}` }) });
    }, 5));
  };
  return { makeFetch, maxInFlight: () => maxInFlight };
}

(async () => {
  // all 4 succeed, order preserved
  const env = makeUploadEnv(null);
  const up = new Function("fetch", "_fetchTimeoutSignal", "addTL", `${upFnText}; return _designUploadImages;`)(
    env.makeFetch, () => undefined, addTL
  );
  const images = [img("A.png", 1), img("B.png", 1), img("C.png", 1), img("D.png", 1)].map(f => ({
    dataUrl: "data:image/png;base64,AA", mediaType: f.type, name: f.name,
  }));
  const res = await up(images);
  check("concurrency: never more than 2 in flight", env.maxInFlight() <= 2, `max=${env.maxInFlight()}`);
  check("concurrency: all 4 succeed", res.ids.length === 4 && res.failed === 0, JSON.stringify(res));
  check("concurrency: original order preserved", res.ids.join(",") === "id-A.png,id-B.png,id-C.png,id-D.png", res.ids.join(","));

  // 1 of 4 fails → ids drop it, failure counted
  lastError = null;
  const env2 = makeUploadEnv((name) => name !== "C.png");
  const up2 = new Function("fetch", "_fetchTimeoutSignal", "addTL", `${upFnText}; return _designUploadImages;`)(
    env2.makeFetch, () => undefined, addTL
  );
  const res2 = await up2(images);
  check("concurrency: failed upload dropped from ids", res2.ids.join(",") === "id-A.png,id-B.png,id-D.png", res2.ids.join(","));
  check("concurrency: failure counted", res2.failed === 1, JSON.stringify(res2));
  check("concurrency: warning names the failed count", /1 image failed/.test(lastError || ""), lastError || "no error");
  check("concurrency: max in flight still bounded", env2.maxInFlight() <= 2, `max=${env2.maxInFlight()}`);
})().then(() => {
  console.log(`P19-2 design attach limits gate: ${passed} checks PASS`);
}).catch((e) => {
  console.error(`P19-2 design attach limits gate FAILED: ${e.message}`);
  process.exit(1);
});
