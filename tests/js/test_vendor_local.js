#!/usr/bin/env node
/**
 * Regression harness — P10-1: Chart.js is vendored locally, not loaded from CDN.
 *
 * Bug: ui.html loaded Chart.js from https://cdn.jsdelivr.net — the only
 * external script in an otherwise fully-local vendor/ tree. Offline / firewall
 * deployments silently lost the performance dashboard, and the CSP had to
 * allow an external origin for a single script.
 *
 * Fix under test: vendor/chart.umd.min.js (pinned 4.4.9) + local <script src>,
 * CSP script-src tightened to 'self' only.
 *
 * Run: node tests/js/test_vendor_local.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.join(__dirname, "..", "..");
const uiHtml = fs.readFileSync(path.join(root, "webapp", "ui", "templates", "ui.html"), "utf8");
const mainPy = fs.readFileSync(path.join(root, "webapp", "main.py"), "utf8");

let passed = 0;
function check(name, cond, detail) {
  assert.ok(cond, `${name} — ${detail || ""}`);
  passed++;
}

// ── ui.html: chart script must come from the local vendor tree ──
const chartScript = uiHtml.match(/<script[^>]*src="([^"]*chart[^"]*)"[^>]*>/);
check("chart script tag present", !!chartScript, (uiHtml.match(/<script[^>]*src="[^"]*"/g) || []).join(" | ") || "no script tags");
check("chart src is local vendor", !!chartScript && chartScript[1].startsWith("/static/vendor/"), chartScript && chartScript[1]);
check("chart src points at chart.umd.min.js", !!chartScript && /chart\.umd\.min\.js$/.test(chartScript[1]), chartScript && chartScript[1]);

// ── no external script origins left anywhere in the page ──
check("no cdn.jsdelivr.net in ui.html", !uiHtml.includes("cdn.jsdelivr.net"), "ui.html still references jsDelivr");
check("no https:// script src in ui.html", !/src="https?:/.test(uiHtml), "ui.html has an external script src");

// ── CSP: script-src must be 'self' only ──
check("no cdn.jsdelivr.net in main.py CSP", !mainPy.includes("cdn.jsdelivr.net"), "CSP still allows jsDelivr");
const scriptSrc = mainPy.match(/"script-src ([^"]+)"/);
check("script-src exists", !!scriptSrc, (mainPy.match(/"script-src[^"]*"/) || ["<none>"])[0]);
check("script-src is self-only", !!scriptSrc && scriptSrc[1] === "'self'", scriptSrc && scriptSrc[1]);

// ── vendor file exists and is the pinned build ──
const vendorPath = path.join(root, "webapp", "ui", "static", "vendor", "chart.umd.min.js");
const vendorStat = fs.statSync(vendorPath);
check("vendor chart.umd.min.js exists", vendorStat.isFile(), vendorPath);
check("vendor chart file is non-trivial", vendorStat.size > 100_000, `size=${vendorStat.size}`);
const vendorHead = fs.readFileSync(vendorPath, "utf8").slice(0, 2000);
check("vendor file is pinned chart.js 4.4.9", vendorHead.includes("Chart.js v4.4.9"), "version marker missing in vendor file");

console.log(`P10-1 vendor-local gate: ${passed} checks PASS`);
