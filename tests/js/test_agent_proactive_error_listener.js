#!/usr/bin/env node
/**
 * Regression harness — round 32-6: the server failure frame of the
 * /agent/proactive/stream endpoint (stats.py `_err_gen`) is now consumed
 * and DISPLAYED by the proactive src listener instead of being silently
 * dropped.
 *
 * Background — stats.py hosts TWO SSE endpoints: the performance stream
 * and /agent/proactive/stream. The latter answers with ONE `event: error`
 * frame carrying the startup-failure reason ("ProactiveRunner
 * unavailable" / "ProactiveRunner init failed" / "PushManager
 * unavailable") whenever the backend cannot start. File-level gate
 * attribution matched that emit against ui.js's native
 * stream.addEventListener("error") — a coincidental name collision that
 * let the R1 gap pass silently (32차-6: endpoint-slice attribution + a
 * real consumer on the proactive src).
 *
 * Gates:
 *   R1 presence — `src.addEventListener("error", ...)` lives among the
 *      proactive src listeners, before src.onerror (the connect block).
 *   R2 guard — `typeof e.data !== "string"` returns early: native
 *      lifecycle Events (connection error → 10s reconnect) carry no data
 *      and stay with src.onerror — the two paths cannot double-fire.
 *   R3 payload contract — stats.py emits `data: {"message": msg}`; the
 *      listener JSON-parses; _onError reads data.error || data.message
 *      and renders error status dot + toast (ttl 8s).
 *   R4 behavior — server frame → _onError(parsed); data-less Event →
 *      no-op; malformed JSON → safe (empty object); _onError renders.
 *
 * Run: node tests/js/test_agent_proactive_error_listener.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const srcPath = path.join(__dirname, "..", "..", "webapp", "ui", "static", "agent-panel.js");
const src = fs.readFileSync(srcPath, "utf8");

// ── R1: the listener lives in the proactive connect block ────────────────
const connAnchor = 'src.addEventListener("proactive_connected"';
const errAnchor = 'src.addEventListener("error", (e) => {';
const onerrorAnchor = "src.onerror = () => {";
const iConn = src.indexOf(connAnchor);
const iErr = src.indexOf(errAnchor);
const iOnErr = src.indexOf(onerrorAnchor);
assert.ok(iConn >= 0, "proactive connect block must exist");
assert.ok(iErr >= 0, 'src.addEventListener("error") listener must exist (round 32-6)');
assert.ok(iErr > iConn && iErr < iOnErr,
  "error listener must live inside the proactive connect block (before src.onerror)");

// ── extract the listener arrow fn (brace-balanced) ────────────────────────
const arrowOpen = src.indexOf("(e) => {", iErr);
assert.ok(arrowOpen > iErr && arrowOpen < iOnErr, "listener must be an arrow fn");
let depth = 0;
let i = arrowOpen;
for (; i < src.length; i++) {
  if (src[i] === "{") depth++;
  else if (src[i] === "}") { depth--; if (depth === 0) break; }
}
assert.ok(i < src.length && src.slice(i, i + 3) === "});", "listener must end with });");
const arrowSrc = src.slice(arrowOpen, i + 1);

// ── R2: the native-lifecycle guard ────────────────────────────────────────
assert.ok(arrowSrc.includes('typeof e.data !== "string"'),
  "data-less native Events must return early (stay with src.onerror — no double-fire)");
assert.ok(!/src\.onerror/.test(arrowSrc),
  "listener must not touch src.onerror (reconnect ownership stays native)");

// ── R3: payload contract (stats.py _err_gen) ──────────────────────────────
assert.ok(arrowSrc.includes("JSON.parse(e.data)"), "server frame data must be JSON-parsed");
const onErrorDef = "function _onError(data) {";
const iDef = src.indexOf(onErrorDef);
assert.ok(iDef >= 0, "_onError must exist");
const iOpen = src.indexOf("{", iDef);
depth = 0;
i = iOpen;
for (; i < src.length; i++) {
  if (src[i] === "{") depth++;
  else if (src[i] === "}") { depth--; if (depth === 0) break; }
}
assert.ok(i < src.length, "unbalanced _onError body");
const onErrorSrc = src.slice(iDef, i + 1);
assert.ok(onErrorSrc.includes("data.error || data.message"),
  "_onError must read data.error || data.message (err_gen frame shape)");
assert.ok(onErrorSrc.includes('_setStatusDot("error")'), "_onError must set the error status dot");
assert.ok(onErrorSrc.includes("_makeToast("), "_onError must show a toast");

// ── R4a: behavior — server frame → _onError(parsed); native → no-op ──────
const calls = [];
const handlerFactory = new Function("_onError", `return (${arrowSrc});`);
const handler = handlerFactory((d) => calls.push(d));

handler({ data: '{"message": "ProactiveRunner init failed"}' });
assert.strictEqual(calls.length, 1, "server frame must reach _onError");
assert.deepStrictEqual(calls[0], { message: "ProactiveRunner init failed" },
  "frame data must arrive JSON-parsed");

handler({});                       // native lifecycle Event — no data
handler({ data: undefined });      // same
assert.strictEqual(calls.length, 1, "data-less Events must NOT reach _onError (native → onerror)");

handler({ data: "not-json" });     // malformed frame
assert.strictEqual(calls.length, 2, "malformed frame must still be reported (safe fallback)");
assert.deepStrictEqual(calls[1], {}, "malformed JSON → empty object (no crash)");

// ── R4b: behavior — _onError renders error dot + toast ────────────────────
const dots = [], toasts = [], timeouts = [];
const onErrorFactory = new Function(
  "_setStatusDot", "_makeToast", "setTimeout",
  `return (${onErrorSrc});`
);
const onError = onErrorFactory(
  (s) => dots.push(s),
  (t, tag, msg, opts) => toasts.push({ t, tag, msg, opts }),
  (fn, ms) => { timeouts.push({ fn, ms }); return 1; }
);
onError({ message: "PushManager unavailable" });
assert.deepStrictEqual(dots, ["error"], "status dot must go error");
assert.strictEqual(toasts.length, 1, "one toast");
assert.strictEqual(toasts[0].t, "error");
assert.strictEqual(toasts[0].msg, "PushManager unavailable", "toast shows the frame reason");
assert.strictEqual(toasts[0].opts.ttl, 8000);
assert.strictEqual(timeouts.length, 1, "dot reset must be scheduled");
assert.strictEqual(timeouts[0].ms, 3000);

onError({ error: "boom", message: "m" });
assert.strictEqual(toasts[1].msg, "boom", "data.error takes precedence over message");

onError({});
assert.strictEqual(toasts[2].msg, "Proactive 오류", "fallback label when payload is empty");

console.log("OK — proactive stream error frame is consumed + displayed (round 32-6)");
