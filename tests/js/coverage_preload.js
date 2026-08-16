#!/usr/bin/env node
/**
 * Coverage preload — execution-evidence capture for the JS coverage gate.
 *
 * tests/js tests extract code from webapp/ui/static/*.js as STRINGS and
 * compile it via `new Function(...)`, so V8 native coverage
 * (NODE_V8_COVERAGE / --experimental-test-coverage) attributes the executed
 * code to empty-url eval scripts and reports ZERO coverage for the real
 * files (empirically verified 2026-08-16: a handler executed through
 * `new Function` never appears under the source file's URL). This preload
 * closes that gap with a different mechanism:
 *
 *   - global Function is replaced by a proxy whose construct/apply trap
 *     captures the compiled body (the last constructor argument) and wraps
 *     the returned function in a call-tracking proxy.
 *   - Wrapped OBJECT return values hand out call-tracking wrappers for
 *     function-valued properties, so the factory pattern
 *     (`new Function("ctx", "return ({ name: (data) => {...} });")` followed
 *     by `handlers.name(...)`) records the factory body on every call.
 *   - Direct-call patterns (`new Function("e", "d", body)` then `fn(...)`)
 *     record the body on the call itself.
 *
 * Every actual CALL appends the owning body string to an evidence registry
 * (deduplicated with a call count), flushed to $COV_OUT as JSON on exit.
 * run_coverage_gate.js decides coverage by checking whether recorded bodies
 * contain the exact source slices of the live handlers-map entries / SSE
 * listener bodies — i.e. the coverage criterion is "a test compiled the
 * real code and called it".
 *
 * Loaded via: node --require tests/js/coverage_preload.js tests/js/test_*.js
 */
"use strict";

const fs = require("fs");

const registry = new Map(); // body -> call count
let flushed = false;

function record(body) {
  if (typeof body !== "string" || body.length === 0) return;
  registry.set(body, (registry.get(body) || 0) + 1);
}

// Wrap a value so that the value itself (when a function) and every
// function-valued property of an object record a call under `ownerBody`
// (the Function body that produced this value).
function wrapValue(value, ownerBody) {
  if (typeof value === "function") {
    return new Proxy(value, {
      apply(target, thisArg, args) {
        record(ownerBody);
        return Reflect.apply(target, thisArg, args);
      },
    });
  }
  if (value !== null && typeof value === "object") {
    return new Proxy(value, {
      get(target, key, receiver) {
        const v = Reflect.get(target, key, receiver);
        return typeof v === "function" ? wrapValue(v, ownerBody) : v;
      },
    });
  }
  return value;
}

const RealFunction = global.Function;
const WrappedFunction = new Proxy(RealFunction, {
  construct(target, args, newTarget) {
    const fn = Reflect.construct(target, args, newTarget);
    const body = String(args[args.length - 1] || "");
    return wrapValue(fn, body);
  },
  apply(target, thisArg, args) {
    const fn = Reflect.apply(target, thisArg, args);
    const body = String(args[args.length - 1] || "");
    return wrapValue(fn, body);
  },
});
global.Function = WrappedFunction;

function flush() {
  if (flushed) return;
  flushed = true;
  const out = process.env.COV_OUT;
  if (!out) return;
  const payload = {
    records: [...registry.entries()].map(([body, count]) => ({ body, count })),
  };
  try {
    fs.writeFileSync(out, JSON.stringify(payload));
  } catch (e) {
    process.stderr.write(`[coverage_preload] flush failed: ${e.message}\n`);
  }
}
process.on("exit", flush);
