# Typed Failure Classifier — migration design based on tree-sitter error nodes

> Long-term item. Design for migrating `vm/failure_classifier.py`
> (+ `ts_vm/repair/failure_classifier.py`) — the top-priority target for
> CLAUDE.md's design insight ("keyword/regex → AST/graph/typed policy migration").

## 1. Current state and problems

Current pipeline (`vm/vm.py`):

```
verifier (compile/javac/kotlinc/go build)
    → VerifyError(message, line, column, code)
    → BaseFailureClassifier._classify_single(errors[0])
        1) error_code_map   (only a partial set, e.g. Python E0602/F821)
        2) keyword_map      (lowercase substring matching)
        3) regex_patterns   (message regex)
    → FailureType enum
    → RepairPlanner → RepairRegistry[FailureType] → strategy execution
```

Problems:

| # | Problem | Evidence |
|---|------|------|
| P1 | **Locale-fragile**: javac follows the JDK locale and can print output in Korean ("오류:"). The `(error\|warning)` regex in `_parse_javac_output` and the entire keyword_map are disabled by this → errors parse as 0 even though `returncode!=0`, leaving errors=[] | `verifier.py:169`, `failure_classifier.py:125-137` |
| P2 | **Compiler version/wording fragile**: if message wording changes, it silently downgrades to UNKNOWN with no failure signal | all 3 keyword/regex layers |
| P3 | **Inaccurate symbol extraction**: `\w+`-based, so it truncates or fails on qualifiers (`pkg.Foo`), generics, Kotlin backtick identifiers, and Unicode identifiers | `_PY_EXTRACT_SYMBOL`, etc. |
| P4 | **Only the first error is classified**: `classify()` only looks at `errors[0]`. Misclassifies when the root cause is the 2nd error (cascading errors) | `failure_classifier.py:39-42` |
| P5 | **Duplicate FailureType enums**: nearly identical enums exist in both vm/ and ts_vm/ (differing only in PROPERTY_NOT_EXIST, MISSING_VARIABLE, UNUSED_IMPORT) | comparing the two files |
| P6 | **Doesn't use the most reliable signal**: the *failed code itself* is available, yet only the compiler's *message string* gets parsed | overall design |

## 2. Core principle

**"Message parsing is a last resort."** Three layers, in descending order of confidence:

1. **Layer A — code structure (tree-sitter)**: parse the failed code directly and
   determine syntax errors *structurally* from ERROR/MISSING nodes. Locale/compiler-
   independent; all 8 grammars already exist as `pyproject.toml` dependencies.
2. **Layer B — machine-readable diagnostic codes**: tree-sitter cannot see semantic
   errors (type/import/undefined-symbol). Instead, obtain the compiler's **stable
   machine-readable code** and map it through a typed table.
3. **Layer C — message fallback**: keep the existing keyword/regex logic but shrink
   it, and instrument fallback usage to measure convergence.

Honest limitation: tree-sitter error nodes only cover the **syntax-error axis**. The
real benefit of this migration is (a) structuring SYNTAX_ERROR determination,
(b) structuring symbol extraction, and (c) producing a typed FixHint to hand to
repair; de-regexing semantic-error classification is Layer B's (diagnostic codes) job.

## 3. Data model (typed output)

Expand the single `FailureType` enum return into a `Classification` dataclass:

```python
# vm/classification.py (new, shared by vm/ts_vm)

class EvidenceSource(str, Enum):
    TREE_SITTER = "tree_sitter"      # Layer A
    ERROR_CODE = "error_code"        # Layer B
    MESSAGE_FALLBACK = "message"     # Layer C
    NONE = "none"                    # UNKNOWN

@dataclass(frozen=True)
class FixHint:
    """Structured hint passed to the repair strategy (optional)."""
    kind: str                  # "insert_token" | "remove_import" | "rename" ...
    token: Optional[str]       # e.g. ";"  (the MISSING node's expected token)
    line: Optional[int]
    column: Optional[int]

@dataclass(frozen=True)
class Classification:
    type: FailureType
    source: EvidenceSource
    symbol: Optional[str] = None      # absorbs extract_symbol (single pass)
    fix_hint: Optional[FixHint] = None
    error_index: int = 0              # which VerifyError this was classified from
```

- Keep the `classify()` signature but add a new `classify_typed()` → existing
  consumers (`repair_planner.py:61`) migrate incrementally to `classify_typed().type`.
- `extract_symbol()` gets absorbed into `Classification.symbol` (removes a re-run of the regex).
- Promote vm/ts_vm's FailureType to a shared module, unified as the union of both
  (including PROPERTY_NOT_EXIST + MISSING_VARIABLE + UNUSED_IMPORT).

## 4. Layer A — tree-sitter syntax classifier

Add a utility to `languages/tree_sitter_utils.py` extending the existing `has_error()` (bool):

```python
@dataclass(frozen=True)
class SyntaxErrorNode:
    kind: str            # "ERROR" | "MISSING"
    missing_token: str   # expected token for a MISSING node (e.g. ";", ")")
    line: int            # 0-based → consumer converts to 1-based
    column: int
    context_snippet: str # surrounding source (for repair prompts/strategies)

def find_error_nodes(content: str, language: str) -> Optional[list[SyntaxErrorNode]]:
    """Collect all ERROR/MISSING nodes. Returns None if tree-sitter isn't installed (fallback signal)."""
```

- Implementation mirrors `has_error()`'s iterative DFS (safe recursion limit) —
  collects instead of early-returning.
- For a MISSING node, `node.type` *is* the expected token, so `FixHint(kind="insert_token",
  token=node.type, line=..., column=...)` comes for free. This is a superset of the
  information the current `py_repair_syntax_error`-style code re-derives from the message.
- Classification priority rules:
  - **tree-sitter ERROR/MISSING present ⇒ SYNTAX_ERROR (high confidence)**. The message is not consulted.
  - **Tree is clean + compiler reports an error ⇒ guaranteed not a syntax error** → defer to Layer B/C.
    This rule alone lets us delete roughly half of the syntax-related entries in the
    current keyword_map ("expected ';'", "unclosed", "expecting", "invalid syntax",
    "unexpected indent", …).
- Fallback: skip Layer A if `is_available()==False` or `find_error_nodes()==None`
  (same guard pattern already used elsewhere in the repo).
- Caveat (risk R1): tree-sitter grammars can be more lenient/strict than the
  compiler (Python soft keywords, newest syntax). So "tree clean ⇒ not a syntax
  error" is kept as a **weak guarantee that Layer B can override when the compiler
  gives an explicit syntax code**.

## 5. Layer B — diagnostic code normalization (requires verifier upgrades first)

For the classifier to become typed, the verifier must carry machine-readable codes:

| Language | Current | Migration | What we gain |
|------|------|------|---------|
| Python | text-parses `pyright <file>`, code fixed as "PYRIGHT" | `pyright --outputjson` | a `rule` field (reportUndefinedVariable, reportMissingImports…) → direct code-map lookup, regex parsing removed |
| Java | locale-dependent `javac` text | `javac -XDrawDiagnostics` | locale-independent stable keys like `compiler.err.cant.resolve.location`, `compiler.err.expected` → fixes P1 at the root |
| Kotlin | kotlinc text | kotlinc is **locale-independent** (always emits English, empirically verified) and offers no stable code → **keep the message fallback**. `-J-Duser.language=en` is a no-op but kept for defensive consistency | P1 doesn't apply (no locale fragility) |
| Go | `go build` text | no stable code available (the message is de facto stable) → keep the fallback, pin `LANG=C` | status quo (Go messages are fixed in English, so risk is low) |
| TS | tsc TS-codes (ts_vm) | already best practice (`_TSC_CODE_MAP`) | reference implementation for Layer B |

- `error_code_map` remains a per-language **data table** (dict) — this is a typed
  mapping, not a regex, so it doesn't violate the design insight. ts_vm's `_TSC_CODE_MAP` is the model.
- Immediate low-cost win (can land ahead of the rest of the design): just pinning
  javac's locale removes P1's "0 errors parsed" silent bug.
- **kotlinc empirical result** (2026-07-05): all three of `LANG=C`, `LANG=ko_KR.UTF-8`,
  and `-J-Duser.language=en` produced identical English output
  (`Bad.kt:1:22: error: unresolved reference 'undefinedSymbol'.`).
  kotlinc is not localized, so P1's locale fragility does not apply to Kotlin.
  The `-J` flag is harmless but a no-op — kept for defensive consistency.

## 6. Layer C — message fallback + instrumentation

- Keep the existing keyword/regex logic, but shrink its surface by removing the
  syntax-related entries (now absorbed by Layer A).
- Expose fallback usage via `Classification.source`, and log a counter on each
  classification: `logger.info("classify: %s via %s", type, source)`. Wiring this
  into the existing adaptive/weight_learning infrastructure gives a "fallback rate"
  telemetry signal to quantitatively measure migration convergence.
- Once the fallback rate converges to 0 for a given language, delete that
  language's Layer C entries.

## 7. Symbol extraction migration (P3)

Message regex (`\w+`) → **position-based tree lookup**:

```
Compiler gives us line/column (VerifyError.line/column already exists)
    → tree.root_node.descendant_for_point_range((line-1, col-1), ...)
    → identifier/type_identifier node (or nearest identifier ancestor/sibling)
    → node.text = the exact symbol
```

- Correct for qualifiers, generics, backtick identifiers, and Unicode identifiers
  alike. Non-ASCII identifier symbols are also handled safely (an internationalization requirement).
- Only errors missing line/column fall back to the existing message regex (recorded in source).

## 8. Migration phases

| Phase | Content | Verification |
|-------|------|------|
| 0 | Unify FailureType into a shared module + introduce `Classification`/`classify_typed()` (behavior unchanged, still delegates to the existing regex) | **Golden corpus snapshot**: collect real compiler output as fixtures across the language × failure-type matrix, pin current classification results as golden |
| 1 | Pin verifier locale + `pyright --outputjson` + `javac -XDrawDiagnostics` + expand the code map (Layer B) | golden diff — only UNKNOWN reductions allowed |
| 2 | Add `find_error_nodes()` + apply Layer A priority, remove syntax keyword/regex entries | golden diff + FixHint snapshot |
| 3 | Position-based symbol extraction + absorb extract_symbol | symbol-extraction accuracy fixtures (including qualifier/backtick/Unicode cases) |
| 4 | Replace the ts_vm classifier with the shared implementation, wire up fallback-rate instrumentation | existing ts_vm tests + fallback-rate logs |

Each phase is an independent commit/independent test. The golden corpus is the
regression safety net (repo principle: subsystem tests come first —
memory/run-scope-tests-before-commit).

## 9. Risks

- **R1 grammar–compiler mismatch**: mitigated by the priority rule in §4 — designed
  so an explicit syntax code from Layer B can override the tree-sitter determination.
- **R2 tree-sitter not installed**: since it's an optional dependency group, the
  `is_available()` fallback is mandatory. Layer B/C alone must perform at least as
  well as current behavior without Layer A (why Phase 1 precedes Phase 2).
- **R3 imprecise ERROR node position**: tree-sitter's error recovery is heuristic,
  so an ERROR node's position may not match the actual cause → FixHint is used only
  as a "hint," and the repair strategy applies it after verification (re-parse).
  The VM loop already re-verifies, so this safety net already exists.
- **R4 whether to keep classify(errors[0]) (P4)**: introduce a root-cause-priority
  rule in Phase 2 — `SYNTAX_ERROR > MISSING_IMPORT > other` — over the classify_all
  results (cascading semantic errors are noise once a syntax error is present).
