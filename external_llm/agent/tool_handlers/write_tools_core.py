"""Shared module-level helpers for the write tool handlers (P2-2 split).

Split out of ``write_tools.py``. Contains the repo file index, indentation /
fragment-duplication / block-introducer guards, and the LLM JSON repair helper
shared by the write tool mixins. This module imports nothing from the sibling
``write_tools_*`` modules — mixins depend on it, never the other way around.
"""

from __future__ import annotations

import logging
import re

from ...common.indent_utils import (
    _file_indent_unit_from_logical,
    detect_indent_char,
    reindent_to_match,
)

# ── tree-sitter (optional) — single source of truth for block extents ────
# tree_sitter_utils imports only stdlib and guards the native `tree_sitter`
# library internally (_HAS_TREE_SITTER + per-call None checks), so importing
# it can never fail — the old `except Exception` stub block was dead.  `_HAS_TS`
# stays a patchable module flag so tests can force the fallback strategies
# (same pattern as symbol_search / code_structure_utils).
from ...languages.base import (
    _iter_brace_tokens,
    find_brace_block_end,
)
from ...languages.tree_sitter_utils import (
    _LANG_MODULE_MAP as _TS_LANG_MODULE_MAP,
)
from ...languages.tree_sitter_utils import (
    find_all_symbols as _ts_find_all_symbols,
)
from ...languages.tree_sitter_utils import (
    is_language_available as _ts_language_available,
)

_HAS_TS = True

logger = logging.getLogger(__name__)


def _find_block_end_line(
    content: str,
    lang_str: str,
    anchor_lineno: int,
    lines: list[str],
) -> int | None:
    """Return the inclusive 0-indexed END line of the block whose header sits
    at ``anchor_lineno``, or ``None`` if the anchor is NOT a block header.

    Called by anchor_edit ``insert_after`` to keep the new construct a SIBLING
    of the block instead of nesting it inside the body — the classic
    "insert_after on a ``def``/``{`` line lands inside the body" bug.

    The strategy is language-agnostic and tree-sitter-first, so installing a
    grammar (e.g. ``tree_sitter_kotlin``) enables the correction for that
    language with **no code change**:

      1. tree-sitter ``find_all_symbols`` → the symbol whose ``start_line``
         equals the anchor → its ``end_line`` (authoritative; covers
         def/class/method across Python/TS/JS/Go/Java/Rust/C/C++/...).
         This runs BEFORE the per-language header heuristics because those
         mis-classify common header shapes — Allman braces (``void foo()`` /
         ``{`` on the next line), multi-line signatures (``void foo(`` /
         ``int x) {``) and ``def foo():  # comment`` all read as "not a
         header" to the cheap filters, which silently nested insert_after
         into the body (the old order).
      2. brace fallback for brace languages without an installed grammar
         (Kotlin/PHP/Swift/Scala/...): delegates to the literal/comment-aware
         ``languages.base`` brace scanner (the SSOT every C-family provider
         uses) — the old ``count('{') - count('}')`` loop miscounted braces
         inside string/char/comment literals (e.g. Kotlin ``val s = "}"``)
         and truncated the block early.
      3. indent fallback for Python without a grammar (rare — the python
         grammar ships with the package).
    """
    if anchor_lineno < 0 or anchor_lineno >= len(lines):
        return None
    anchor_line = lines[anchor_lineno].rstrip("\n\r")
    stripped = anchor_line.strip()
    if not stripped:
        return None
    anchor_indent = len(anchor_line) - len(anchor_line.lstrip())

    # ── Strategy 1: tree-sitter (authoritative end_line) ─────────────────
    if _HAS_TS and lang_str in _TS_LANG_MODULE_MAP and _ts_language_available(lang_str):
        # find_all_symbols degrades to [] internally when tree-sitter or the
        # grammar is unavailable (parse/query paths are guarded inside
        # tree_sitter_utils) — the call itself never raises, so no wrapper is
        # needed here; an unexpected error now surfaces instead of silently
        # falling through to the heuristic strategies.
        syms = _ts_find_all_symbols(content, lang_str)
        a1 = anchor_lineno + 1
        best_end = None
        for _name, _kind, start, end in syms:
            # start == a1 ⇔ the symbol begins ON the anchor line (1-based).
            # end > a1 excludes one-line symbols whose span is the header
            # itself; a decorated function anchored on its decorator line is
            # intentionally included — the whole decorated def is the block.
            if start == a1 and end > a1 and (best_end is None or end > best_end):
                best_end = end
        if best_end is not None:
            return min(best_end, len(lines)) - 1  # 0-indexed inclusive

    # ── Grammar-less fallbacks, per language family ──────────────────────
    is_py = lang_str == "python"
    is_brace = lang_str in (
        "typescript",
        "javascript",
        "go",
        "java",
        "rust",
        "c",
        "cpp",
        "c_sharp",
        "kotlin",
        "php",
        "swift",
        "scala",
    )
    if not is_py and not is_brace:
        return None

    if is_brace:
        # ── Strategy 2: literal/comment-aware brace scan (SSOT) ──────────
        # Delegate to languages.base._iter_brace_tokens + find_brace_block_end
        # (the scanner every C-family provider delegates to). Only an opening
        # brace ON the anchor line makes it a block header; a brace opener on
        # a later line means the anchor is plain code (the old
        # ``depth <= 0 → None`` contract). A block that closes on the header
        # line itself (``if (x) { f(); }``) or is unterminated (SSOT's
        # conservative start-line fallback) is likewise not a header to
        # insert past → None.
        line_start = sum(len(_l) for _l in lines[:anchor_lineno])
        line_end = line_start + len(anchor_line)
        first_open = None
        for _ch, _idx in _iter_brace_tokens(content, line_start):
            if _idx >= line_end:
                break
            if _ch == "{":
                first_open = _idx
                break
        if first_open is None:
            return None
        end_0based = find_brace_block_end(content, first_open) - 1
        if end_0based <= anchor_lineno:
            return None
        return min(end_0based, len(lines) - 1)

    # ── Strategy 3: Python indent (grammar unavailable) ──────────────────
    is_block_header = bool(
        re.match(
            r"^(async\s+def\s|def\s|class\s|if\s|for\s|while\s|with\s|"
            r"try\s*:|elif\s|else\s*:|except\s|finally\s*:|match\s)",
            stripped,
        )
    ) and stripped.endswith(":")
    if not is_block_header:
        return None
    for i in range(anchor_lineno + 1, len(lines)):
        ln = lines[i]
        if not ln.strip():
            continue
        if (len(ln) - len(ln.lstrip())) <= anchor_indent:
            return i - 1
    return len(lines) - 1


# ── Re-indent helper for replace_all + fallback ──────────────────────────


def _reindent_to_match(new_string: str, matched_text: str, file_unit: int | None = None) -> str:
    """Reindent *new_string* to match *matched_text*'s base indentation.

    Delegates to the canonical ``indent_utils.reindent_to_match`` for ALL cases
    (space-only AND tab/space-mixed).  The canonical reindenter is depth-ratio
    aware, preserves bracket-continuation alignment, and content-maps unchanged
    lines to the file's exact indentation — so it never collapses an ``if`` body
    to the same column as the ``if`` (the classic "expected an indented block"
    SyntaxError that the old flat char-count delta produced).

    The previous implementation kept a naive ``_delta = len(match_lead) -
    len(orig_lead)`` path for the common space-only case and only fell back to
    the canonical reindenter when a tab was present.  That flat delta shifts every
    non-empty line by the same number of columns, ignoring relative nesting: a
    block whose first line sits one level shallower than its body got its body
    dedented along with the header, yielding invalid Python.  JSONL failure
    analysis showed ``syntax_invalid_after_edit`` was the single most frequent
    write-tool failure class, and every edit_text instance traced back to this
    path.  Unifying on the canonical reindenter removes the failure mode for
    both indent-char styles.

    Empty lines are left untouched by the canonical reindenter.
    """
    return reindent_to_match(new_string, matched_text, file_unit=file_unit)


def _detect_file_unit(content: str) -> int | None:
    """Per-level indent width (chars) of the *destination file* content.

    Gives :func:`reindent_to_match` a file-wide unit hint so a flat, single-
    level match site in a 2-space file no longer inherits the LLM snippet's
    (possibly 4-space) unit and over-indents.  Returns ``None`` when
    undetectable (empty/garbled content); the canonical reindenter then keeps
    its historic fallback.
    """
    if not content:
        return None
    # Route through the Python-tokenizer path (_file_indent_unit_from_logical)
    # so multi-line strings/docstrings don't poison the GCD toward 1 — a
    # bracket- or paren-heavy docstring makes the language-agnostic
    # ``indent_unit`` mis-detect the file's per-level width, which inflates
    # downstream indent ratios and triggers indent explosion on edit.  The
    # tokenizer path treats string interiors as a single logical line; for
    # non-Python content it transparently falls back to ``indent_unit``.
    # Tokenizer failures (TokenError / IndentationError / SyntaxError) are
    # handled inside indent_utils, so no wrapper is needed here.
    return _file_indent_unit_from_logical(content, detect_indent_char(content.split("\n"))) or None


def _leading_indent_width(text: str) -> int:
    """Leading-whitespace column count of the first non-blank line of *text*.

    Used to surface the *actual* indent at the edit site in edit_text metadata
    (``matched_indent``), so the LLM can self-verify it matched the file's
    indentation — the same metric ``read_file``'s ``│N│`` gutter reports. Empty
    / whitespace-only text returns 0. Tabs count as width 1 (consistent with
    ``min_indent`` in common/indent_utils).
    """
    for ln in text.splitlines():
        if ln.strip():
            return len(ln) - len(ln.lstrip())
    return 0


# ── Fragment-duplication pre-guard for anchor_edit insert modes ──────────────
#
# When the LLM is asked to INSERT new code via anchor_edit (insert_before /
# insert_after), a common failure mode is that code_snippet accidentally
# COPIES existing code around the anchor (a "fragment duplication") instead
# of providing only the new lines. The inserted duplicate is then re-indented
# and lands as a dangling block that only fails the POST-write syntax check
# with an opaque message — forcing 2-3 retry cycles.
#
# This helper detects such duplication BEFORE the file is touched by comparing
# the snippet's non-trivial lines against a window of the file around the
# insertion point. Returns a diagnostic dict when duplication is likely, else
# None. ``replace_line`` / ``delete`` are exempt (they legitimately overlap
# existing code), so the caller gates this on edit_mode.

# Lines that carry no structural identity and must be excluded from BOTH the
# numerator (matched lines) and denominator (total content lines), so a
# snippet reusing only ``return`` / ``}`` / blank lines is not false-positived.
_FRAGMENT_DUP_TRIVIAL = frozenset(
    {
        "",
        "{",
        "}",
        "(",
        ")",
        "[",
        "]",
        "pass",
        "return",
        "continue",
        "break",
        "...",
        "else",
        "try",
        "finally",
        "end",
    }
)
# Minimum non-trivial content lines in the snippet before duplication is even
# judged — below this the snippet is too small to carry structural identity.
_FRAGMENT_DUP_MIN_LINES = 3
# Overlap ratio (matched non-trivial lines / snippet non-trivial lines) at or
# above which duplication is reported.
_FRAGMENT_DUP_RATIO_THRESHOLD = 0.5
# Half-window size around insert_idx scanned for existing code to compare.
_FRAGMENT_DUP_WINDOW = 12


def _detect_fragment_duplication(file_lines, insert_idx, snippet):
    """Detect whether ``snippet`` duplicates existing code around ``insert_idx``.

    ``file_lines`` is the list of lines (with trailing newlines) of the file
    BEFORE the insert. ``insert_idx`` is the 0-based index at which the
    snippet would be inserted. ``snippet`` is the raw code_snippet string.

    Compares each non-trivial line of ``snippet`` (stripped, trailing comment
    ignored) against the file lines in ``[insert_idx - WINDOW, insert_idx +
    WINDOW]``. Returns a dict ``{"ratio": float, "content_lines": int,
    "dup_lines": str}`` when ``content_lines >= MIN_LINES`` and
    ``ratio >= THRESHOLD``; otherwise returns ``None``.

    Best-effort pre-guard: returns ``None`` when no duplication is detected and
    never blocks a legitimate insert (the caller treats the result as a
    diagnostic hint, not a gate).  The computation is pure string ops on the
    caller-provided lines, so no exception handling is needed — an unexpected
    error is a caller bug and should surface.
    """
    # Normalise snippet into non-trivial content lines.
    snip_stripped = []
    for raw in snippet.splitlines():
        s = raw.strip()
        if not s:
            continue
        # strip trailing inline comment for comparison
        if "#" in s:
            s_code = s.split("#", 1)[0].rstrip()
            if not s_code:
                continue
            s = s_code
        if s in _FRAGMENT_DUP_TRIVIAL:
            continue
        snip_stripped.append(s)
    if len(snip_stripped) < _FRAGMENT_DUP_MIN_LINES:
        return None

    # Build the set of existing non-trivial lines in the window.
    lo = max(0, insert_idx - _FRAGMENT_DUP_WINDOW)
    hi = min(len(file_lines), insert_idx + _FRAGMENT_DUP_WINDOW)
    existing = set()
    for i in range(lo, hi):
        s = file_lines[i].strip() if i < len(file_lines) else ""
        if not s:
            continue
        if "#" in s:
            s_code = s.split("#", 1)[0].rstrip()
            if not s_code:
                continue
            s = s_code
        if s in _FRAGMENT_DUP_TRIVIAL:
            continue
        existing.add(s)

    # De-duplicate snippet lines before counting: a snippet that repeats the
    # same content line N times would otherwise inflate both numerator and
    # denominator, but the denominator (unique lines) more accurately reflects
    # "how much of this snippet is new material". dict.fromkeys preserves
    # first-seen order (Python 3.7+) and removes exact duplicates.
    snip_unique = list(dict.fromkeys(snip_stripped))
    matched = [s for s in snip_unique if s in existing]
    ratio = len(matched) / len(snip_unique)
    if ratio >= _FRAGMENT_DUP_RATIO_THRESHOLD:
        return {
            "ratio": ratio,
            "content_lines": len(snip_unique),
            "matched": len(matched),
            "dup_lines": "\n".join(matched[:8]),
        }
    return None


# ── Enclosing-scope detection for anchor_edit structural feedback ──────────
_PY_BLOCK_HEADERS = ("def ", "async def ", "class ")


def _detect_enclosing_scope(file_lines, anchor_lineno):
    """Best-effort structural context around ``anchor_lineno``.

    Returns a dict::

        {
            "innermost": ("function"|"class"|None, name|None, indent|None),
            "top_level": ("function"|"class"|None, name|None, indent|None),
            "anchor_indent": int,
        }

    ``innermost`` is the nearest def/class header at/above the anchor line
    (scanning upward). ``top_level`` is the nearest header at indent 0 — i.e.
    the module-level construct the anchor lives in. Robustness: out-of-range
    anchors degrade to the default dict (all-None entries). Used to surface
    "inserted inside scope X" feedback in anchor_edit metadata so the LLM can
    self-verify the structural correctness of an insert without a separate
    read_file round-trip.
    """
    out = {
        "innermost": (None, None, None),
        "top_level": (None, None, None),
        "anchor_indent": 0,
    }
    if anchor_lineno < 0 or anchor_lineno >= len(file_lines):
        return out
    anchor_text = file_lines[anchor_lineno]
    out["anchor_indent"] = len(anchor_text) - len(anchor_text.lstrip())
    innermost = None
    top_level = None
    for up in range(anchor_lineno, -1, -1):
        line = file_lines[up]
        stripped = line.strip()
        if not stripped:
            continue
        for hdr in _PY_BLOCK_HEADERS:
            if stripped.startswith(hdr):
                indent = len(line) - len(line.lstrip())
                kind = "class" if hdr == "class " else "function"
                name = stripped[len(hdr) :].split("(", 1)[0].split(":", 1)[0].strip()
                if innermost is None:
                    innermost = (kind, name, indent)
                if indent == 0 and top_level is None:
                    top_level = (kind, name, indent)
                break
        if top_level is not None and innermost is not None:
            break
    if innermost is not None:
        out["innermost"] = innermost
    if top_level is not None:
        out["top_level"] = top_level
    return out


def _check_block_introducer_nesting(new_content, insert_start_line, insert_end_line):
    """AST backstop: verify an inserted def/class neither landed nested in a
    function body NOR swallowed pre-existing trailing code into its own body.

    The text-based indent-correction above (block-introducer re-anchoring)
    covers the common cases, but it is still a heuristic over raw lines — it
    can miss snippet shapes it wasn't taught to recognize. This is the
    structural gate: parse the ACTUAL new file with the ``ast`` module and
    check, for every top-level def/class introduced within the inserted
    range (lines ``[insert_start_line, insert_end_line)``, 0-based
    half-open):

    1. Nested-in-function — is it a lexical child of a PRE-EXISTING
       FunctionDef / AsyncFunctionDef? Landing inside someone else's
       function is essentially never the intent for a snippet that itself
       introduces a new def/class — the intent is always sibling/module (or
       class-body) level, never "define a new nested helper inside an
       unrelated function" via anchor_edit. NOTE: an inner helper defined
       inside a function that is ALSO part of the same insertion is a
       legitimate closure (the whole construct moved together), so the
       enclosing function must itself be OUTSIDE the inserted range to count
       as a violation.
    2. Swallowed-trailing-code — does its body extend PAST the inserted
       range? Re-anchoring the new construct to a shallower indent (fix #1)
       can leave pre-existing sibling statements that followed the anchor
       dangling at their original (deeper) indent with nothing to close the
       new construct's block first — so they silently become part of ITS
       body instead of remaining where they were. This is syntactically
       valid Python (so the separate syntax gate below cannot catch it) but
       it silently makes original code unreachable/misplaced.

    Returns an error string describing the first violation found, or
    ``None`` when neither problem is found (including on parse failure —
    the separate syntax-validation gate already handles unparseable output).
    """
    import ast as _ast

    try:
        tree = _ast.parse(new_content)
    except SyntaxError as e:
        logger.debug("parse failed in block-introducer nesting check: %s", e)
        return None

    _violations = []

    def _walk(node, func_stack):
        # func_stack entries: (name, was_introduced_in_range) for each enclosing
        # FUNCTION (ClassDef is deliberately excluded — a method is not "nested").
        for child in _ast.iter_child_nodes(node):
            _is_func = isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef))
            _is_def = _is_func or isinstance(child, _ast.ClassDef)
            _introduced = False
            if _is_def:
                # Access .lineno only on def/class nodes — not every child has it
                # (e.g. `arguments`), and a stray AttributeError is swallowed by
                # the broad except below, silently masking real violations.
                _lineno0 = getattr(child, "lineno", 0) - 1
                _introduced = insert_start_line <= _lineno0 < insert_end_line
                if _introduced:
                    # Only flag nested-in-function for a PRE-EXISTING enclosing
                    # function. An inner helper defined inside a function that is
                    # ALSO part of this same insertion is a legitimate closure —
                    # the whole construct moved together, so nothing "landed" in
                    # someone else's body. Walk outward to the nearest enclosing
                    # function that was NOT introduced by this edit.
                    _enclosing = None
                    for _ename, _e_introduced in reversed(func_stack):
                        if not _e_introduced:
                            _enclosing = _ename
                            break
                    if _enclosing is not None:
                        _violations.append(("nested_in_function", getattr(child, "name", ""), _enclosing))
                    _end_lineno = getattr(child, "end_lineno", None)
                    if _end_lineno is not None and _end_lineno > insert_end_line:
                        _violations.append(("swallowed_trailing_code", getattr(child, "name", ""), None))
            if _is_func:
                _walk(child, [*func_stack, (getattr(child, "name", ""), _introduced)])
            else:
                _walk(child, func_stack)

    # The walk only touches attributes guaranteed on a valid AST (guarded by
    # isinstance checks + getattr above), so an unexpected error here is a real
    # bug and must propagate (fail-fast) — the old `except Exception` silently
    # masked violations.
    _walk(tree, [])

    if not _violations:
        return None
    _kind, _name, _enclosing = _violations[0]
    if _kind == "nested_in_function":
        return (
            f"structural nesting violation: inserted 'def/class {_name}' landed "
            f"inside function '{_enclosing}()' body. This is almost always "
            f"unintended for anchor_edit inserts — re-check the anchor line, or "
            f"use apply_patch for top-level insertions."
        )
    return (
        f"structural nesting violation: inserted 'def/class {_name}' swallowed "
        f"pre-existing code that followed the anchor into its own body — that "
        f"code is now unreachable/misplaced instead of remaining a sibling "
        f"statement. Split the insertion so nothing follows the anchor inside "
        f"the same block, or use apply_patch for top-level insertions."
    )


# ── Truncation diagnostics: identify the op/path that was being written ──


def _extract_truncated_op_path(raw: str) -> str | None:
    """Best-effort: identify the op/path being written when a ``write_plan``
    tool_call was truncated mid-stream.

    The truncation-detection branch in ``_tool_write_plan`` only fires when the
    raw arguments JSON has unbalanced braces (cut off mid-value), so the payload
    cannot be parsed as JSON. This scanner walks the raw text char-by-char while
    tracking double-quoted-string state, so a literal ``"path":`` appearing
    *inside* a ``content``/``before``/``after`` string value is never mistaken
    for a real key. It records the most recent complete ``op`` and the ``path``
    that followed it, then returns a short diagnostic hint such as
    ``"op=create_file path=src/foo.py"`` — or ``None`` when nothing identifiable
    precedes the cut point.

    Purely diagnostic: it only flavours the error message; no write decision
    depends on it. Caller is the truncation branch of ``_tool_write_plan``.
    """
    n = len(raw)
    i = 0
    pending_key: str | None = None  # a key literal just followed by ':'
    current_op: str | None = None  # last seen "op" value (string)
    last_snapshot: tuple | None = None  # (op, path) of the most recent path
    orphan_op: str | None = None  # an op with no path emitted after it
    while i < n:
        ch = raw[i]
        if ch != '"':
            # A non-string token sitting in a value slot means the value is
            # structural (object/array/number/bool) — clear the pending key so
            # the next string literal is not mis-attributed as its value.
            # ':' and whitespace are tolerated (they separate key from value).
            if pending_key is not None and ch not in " \t\r\n:":
                pending_key = None
            i += 1
            continue
        # Read one string literal starting at the opening quote (index i).
        j = i + 1
        esc = False
        while j < n:
            c = raw[j]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                break
            j += 1
        closed = j < n
        value = raw[i + 1 : j]
        if pending_key is not None:
            # This literal is the value for ``pending_key``.
            if pending_key == "op":
                if closed:
                    current_op = value
                    orphan_op = value
            elif pending_key == "path":
                if closed:
                    last_snapshot = (current_op, value)
                    orphan_op = None
                else:
                    # Truncated mid-path value — still useful; mark as partial.
                    last_snapshot = (current_op, value + "…")
                    orphan_op = None
            pending_key = None
        else:
            # Candidate key: only treat as a key when followed by ':'.
            k = (j + 1) if closed else j
            while k < n and raw[k] in " \t\r\n":
                k += 1
            if k < n and raw[k] == ":":
                pending_key = value
        if not closed:
            # Ran off the end inside this literal — nothing after can parse.
            break
        i = j + 1

    # Prefer an op whose path was never emitted (the most recent op, truncated
    # before its path key); otherwise report the last (op, path) pair seen.
    if orphan_op is not None:
        return f"op={orphan_op}"
    if last_snapshot is not None:
        _op, _path = last_snapshot
        if _op and _path:
            return f"op={_op} path={_path}"
        if _path:
            return f"path={_path}"
        if _op:
            return f"op={_op}"
    return None


# ── JSON repair for LLM-generated plan JSON ──────────────────────────────


def _repair_plan_json(text: str) -> str:
    """Repair common JSON issues in LLM-generated plan strings.

    Fixes:
    - Markdown fence extraction (```json ... ```)
    - Single quotes → double quotes (outside strings)
    - Trailing commas in objects/arrays
    - Unquoted keys
    - Extra/missing brackets
    - Unescaped newlines inside string values (LLM multi-line before/after)
    """
    # 1. Markdown fence
    _m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if _m:
        text = _m.group(1).strip()

    # 2. Escape literal newlines inside string values before any other processing.
    #    JSON does not allow raw newlines in strings — LLMs often include them in
    #    multi-line before/after blocks.  Track string state character by character.
    #
    #    BOTH quote styles must be tracked: this step runs *before* the single→
    #    double conversion in step 3, so a single-quoted multi-line value
    #    (``'before': 'line1\nline2'`` — common LLM output) would otherwise keep
    #    its raw newline and fail json.loads with "Invalid control character".
    #    The active quote char is remembered so a foreign quote inside a string
    #    (e.g. the ``'`` in ``"don't"``) does not toggle string state.
    _escaped: list[str] = []
    _quote: str | None = None  # active opening quote char, or None
    _escape = False
    for ch in text:
        if _escape:
            _escaped.append(ch)
            _escape = False
            continue
        if ch == "\\":
            _escaped.append(ch)
            _escape = True
            continue
        if _quote is None and ch in ("'", '"'):
            _quote = ch
            _escaped.append(ch)
            continue
        if _quote is not None and ch == _quote:
            _quote = None
            _escaped.append(ch)
            continue
        if _quote is not None and ch == "\n":
            _escaped.append("\\n")
            continue
        if _quote is not None and ch == "\r":
            _escaped.append("\\r")  # CRLF: pair the \n escape so loads() succeeds
            continue
        _escaped.append(ch)
    text = "".join(_escaped)

    # 3. Single quotes to double quotes for key-value patterns
    #    ('key': 'value' → "key": "value")
    #    Only convert single-quotes that are NOT inside double-quoted strings.
    #    This prevents breaking strings like {"msg": "don't panic"}.
    #
    #    Two-state tracking (_in_dq + _in_sq) is REQUIRED: a single-quote that
    #    OPENED a converted string must be matched by its CLOSING single-quote.
    #    Reusing _in_dq for both (the old approach) made the closing ' look like
    #    "already inside a DQ string" so it was emitted unchanged — corrupting
    #    e.g. {'key': 'value'} into {"key': 'value'} (unterminated string).
    _result: list[str] = []
    _in_dq = False  # inside a literal double-quoted string
    _in_sq = False  # inside a single-quote-converted string; closes on next '
    _escape = False
    for ch in text:
        if _escape:
            _result.append(ch)
            _escape = False
            continue
        if ch == "\\":
            _result.append(ch)
            _escape = True
            continue
        if _in_sq and ch == "'":
            # closing single-quote of a converted string → emit '"'
            _in_sq = False
            _result.append('"')
            continue
        if ch == '"':
            if _in_sq:
                # A literal " inside a single-quoted string being converted to DQ
                # must be escaped, otherwise json.loads() sees a premature string
                # end. Toggling _in_dq here (the old behavior) corrupted
                # {'k': 'say "hi"'} into {"k": "say "hi""} (JSONDecodeError).
                _result.append('\\"')
            else:
                _in_dq = not _in_dq
                _result.append(ch)
            continue
        if not _in_dq and not _in_sq and ch == "'":
            # opening single-quote outside any string → start converted string
            _in_sq = True
            _result.append('"')
            continue
        _result.append(ch)
    text = "".join(_result)

    # 4/5. Trailing commas + unquoted keys — applied OUTSIDE string values only.
    #    These rewrites must never touch the inside of a (double-quoted) string:
    #    plan content routinely embeds code like ``[1, 2, ]`` or ``{foo: 1}``,
    #    and rewriting it would corrupt the very content being written to disk.
    _segments: list[tuple[bool, str]] = []  # (is_string, segment_text)
    _seg: list[str] = []
    _in_str = False
    _escape = False
    for ch in text:
        if _in_str and _escape:
            _seg.append(ch)
            _escape = False
            continue
        if _in_str and ch == "\\":
            _seg.append(ch)
            _escape = True
            continue
        if ch == '"':
            if _in_str:
                _seg.append(ch)
                _segments.append((True, "".join(_seg)))
                _seg = []
                _in_str = False
            else:
                _segments.append((False, "".join(_seg)))
                _seg = [ch]
                _in_str = True
            continue
        _seg.append(ch)
    _segments.append((_in_str, "".join(_seg)))

    def _fix_code_segment(seg: str) -> str:
        # Trailing commas before closing brackets
        seg = re.sub(r",\s*([}\]])", r"\1", seg)
        # Unquoted keys (key: → "key":) at start of object / after comma
        return re.sub(r"([\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', seg)

    return "".join(seg if is_str else _fix_code_segment(seg) for is_str, seg in _segments)


# ── repo file index for "File not found" path suggestions ────────────────
# `edit_text`/`anchor_edit` already emit close-match hints when the anchor
# text or symbol isn't found, but a missing *file* returned a bare
# "File not found" with no corrective info — the #1 recurring write-tool
# failure signal in this repo. The index below lets `_suggest_missing_paths`
# offer "Did you mean: a/b/foo.py?" hints, mirroring the existing behaviour.
#
# P5-2: the cache machinery moved DOWN to ``common.repo_files`` (SSOT) so the
# editor-lane runtime gate / planner / executor scans share ONE per-repo file
# listing cache with the write tools instead of each running its own full
# ``os.walk`` (~185 ms measured). The names below are re-exports of the same
# objects, so every existing importer (tool_registry, read_tools, the barrel)
# keeps working unchanged.
from ...common.repo_files import (  # noqa: E402, F401
    _FILE_INDEX_CACHE,
    _FILE_INDEX_GEN,
    _FILE_INDEX_SKIP_DIRS,
    _FILE_INDEX_TTL,
    canonical_repo_key,
    invalidate_repo_file_index,
)
from ...common.repo_files import (  # noqa: E402
    cached_repo_file_list as _repo_file_index,  # noqa: F401 — re-exported via .write_tools
)
from ...common.repo_files import (  # noqa: E402, V104 — re-exported via .write_tools
    git_list_repo_files as _git_list_tracked_files,  # noqa: F401 — re-exported via .write_tools
)
