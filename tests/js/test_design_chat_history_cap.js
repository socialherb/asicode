#!/usr/bin/env node
/**
 * Regression harness — P11-2: the design-chat SSE history param is capped.
 *
 * Bug: when a new SSE was fired without a server session, the client re-sent
 * the ENTIRE conversation in the `history` query param (slice(0, -1)).  The
 * server consumes `history` only to seed an EMPTY session (migration), so
 * everything older than the last compression window (15 turns) is dead weight
 * — and unbounded it grows the SSE URL without limit (multi-MB at long
 * sessions, eventually 431 from the server/proxy request-line cap).
 *
 * Fix under test: slice(-21, -1) — at most the last 20 turns.
 *
 * Run: node tests/js/test_design_chat_history_cap.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const src = fs.readFileSync(path.join(root, "webapp", "ui", "static", "design-chat.js"), "utf8");

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── source gate: the history param must use the capped slice ──
const histLine = src.split("\n").find((l) => l.includes('"history"') && l.includes("slice("));
check("history send line exists", !!histLine, "no line sets the history query param");
check("history uses capped slice(-21, -1)", !!histLine && histLine.includes("slice(-21, -1)"), histLine || "");
check("no unbounded slice(0, -1) remains in the file", !src.includes("slice(0, -1)"), "unbounded history resend still present");

// ── semantics of the pinned slice expression ──
// The source gate pins the exact expression; this block pins what that
// expression does, so a change to either side fails loudly.
const longHistory = Array.from({ length: 30 }, (_, i) => `t${i}`);
const capped = longHistory.slice(-21, -1);
check("keeps the last 20 turns", capped.length === 20, `length=${capped.length}`);
check("excludes the newest turn (current message)", capped[capped.length - 1] === "t28", capped[capped.length - 1] || "");
check("drops turns older than the window", capped[0] === "t9", capped[0] || "");  // 30-21=9

const shortHistory = Array.from({ length: 5 }, (_, i) => `s${i}`);
const shortCapped = shortHistory.slice(-21, -1);
check("short history passes through intact (minus current message)", shortCapped.length === 4 && shortCapped[0] === "s0", `length=${shortCapped.length}`);

console.log(`P11-2 design-chat history cap gate: ${passed} checks PASS`);
