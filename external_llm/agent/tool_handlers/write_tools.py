"""Write tool handlers for ToolRegistry."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ...languages import LanguageId
from ...common.atomic_io import atomic_write_text
from .._shared_utils import compile_quiet, extract_files_from_patch

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)

# ── tree-sitter (optional) — single source of truth for block extents ────
try:
    from ...languages.tree_sitter_utils import (
        _LANG_MODULE_MAP as _TS_LANG_MODULE_MAP,
    )
    from ...languages.tree_sitter_utils import (  # type: ignore
        find_all_symbols as _ts_find_all_symbols,
    )
    from ...languages.tree_sitter_utils import (
        get_available_languages as _ts_available_languages,
    )
    _HAS_TS = True
except Exception:  # pragma: no cover - tree-sitter is optional
    _HAS_TS = False
    _ts_find_all_symbols = None  # type: ignore
    _ts_available_languages = lambda: set()  # noqa: E731
    _TS_LANG_MODULE_MAP = {}  # type: ignore


def _find_block_end_line(
    content: str, lang_str: str, anchor_lineno: int, lines: list[str],
) -> Optional[int]:
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
      2. brace-balance fallback for brace languages without an installed
         grammar (Kotlin/PHP/Swift/Scala/...).
      3. indent fallback for Python without a grammar (rare — the python
         grammar ships with the package).
    """
    if anchor_lineno < 0 or anchor_lineno >= len(lines):
        return None
    anchor_line = lines[anchor_lineno].rstrip('\n\r')
    stripped = anchor_line.strip()
    if not stripped:
        return None
    anchor_indent = len(anchor_line) - len(anchor_line.lstrip())

    # ── Cheap header detection per language family ───────────────────
    is_py = lang_str == "python"
    is_brace = lang_str in (
        "typescript", "javascript", "go", "java", "rust", "c", "cpp",
        "c_sharp", "kotlin", "php", "swift", "scala",
    )
    is_block_header = False
    if is_py:
        is_block_header = bool(re.match(
            r'^(async\s+def\s|def\s|class\s|if\s|for\s|while\s|with\s|'
            r'try\s*:|elif\s|else\s*:|except\s|finally\s*:|match\s)',
            stripped,
        )) and stripped.endswith(':')
    elif is_brace:
        is_block_header = anchor_line.count('{') > anchor_line.count('}')
    else:
        return None
    if not is_block_header:
        return None

    # ── Strategy 1: tree-sitter (authoritative end_line) ─────────────
    if (
        _HAS_TS
        and lang_str in _TS_LANG_MODULE_MAP
        and lang_str in _ts_available_languages()
    ):
        try:
            syms = _ts_find_all_symbols(content, lang_str)
            a1 = anchor_lineno + 1
            best_end = None
            for _name, _kind, start, end in syms:
                if start == a1 and end >= a1:
                    if best_end is None or end > best_end:
                        best_end = end
            if best_end is not None:
                return min(best_end, len(lines)) - 1  # 0-indexed inclusive
        except Exception:  # pragma: no cover - non-critical, fall through
            pass

    # ── Strategy 2: brace balance (brace langs w/o grammar) ──────────
    if is_brace:
        depth = anchor_line.count('{') - anchor_line.count('}')
        if depth <= 0:
            return None
        for i in range(anchor_lineno + 1, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if depth <= 0:
                return i
        return len(lines) - 1

    # ── Strategy 3: Python indent (grammar unavailable) ──────────────
    if is_py:
        for i in range(anchor_lineno + 1, len(lines)):
            ln = lines[i]
            if not ln.strip():
                continue
            if (len(ln) - len(ln.lstrip())) <= anchor_indent:
                return i - 1
        return len(lines) - 1

    return None


# ── Re-indent helper for replace_all + fallback ──────────────────────────

def _reindent_to_match(new_string: str, matched_text: str, file_unit: Optional[int] = None) -> str:
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
    try:
        from ...common.indent_utils import reindent_to_match as _canon_reindent
        return _canon_reindent(new_string, matched_text, file_unit=file_unit)
    except Exception:
        logger.debug(
            "_reindent_to_match canonical reindenter failed (%s); "
            "returning new_string unchanged", exc_info=True,
        )
        return new_string


def _detect_file_unit(content: str) -> Optional[int]:
    """Per-level indent width (chars) of the *destination file* content.

    Gives :func:`reindent_to_match` a file-wide unit hint so a flat, single-
    level match site in a 2-space file no longer inherits the LLM snippet's
    (possibly 4-space) unit and over-indents.  Returns ``None`` when
    undetectable (empty/garbled content); the canonical reindenter then keeps
    its historic fallback.
    """
    if not content:
        return None
    try:
        # Route through the Python-tokenizer path (_file_indent_unit_from_logical)
        # so multi-line strings/docstrings don't poison the GCD toward 1 — a
        # bracket- or paren-heavy docstring makes the language-agnostic
        # ``indent_unit`` mis-detect the file's per-level width, which inflates
        # downstream indent ratios and triggers indent explosion on edit.  The
        # tokenizer path treats string interiors as a single logical line; for
        # non-Python content it transparently falls back to ``indent_unit``.
        from ...common.indent_utils import (
            detect_indent_char,
            _file_indent_unit_from_logical,
        )
        return _file_indent_unit_from_logical(
            content, detect_indent_char(content.split("\n"))
        ) or None
    except Exception:
        return None


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
_FRAGMENT_DUP_TRIVIAL = frozenset({
    "", "{", "}", "(", ")", "[", "]", "pass", "return", "continue", "break",
    "...", "else", "try", "finally", "end",
})
# Minimum non-trivial content lines in the snippet before duplication is even
# judged — below this the snippet is too small to carry structural identity.
_FRAGMENT_DUP_MIN_LINES = 3
# Overlap ratio (matched non-trivial lines / snippet non-trivial lines) at or
# above which duplication is reported.
_FRAGMENT_DUP_RATIO_THRESHOLD = 0.5
# Half-window size around insert_idx scanned for existing code to compare.
_FRAGMENT_DUP_WINDOW = 12

# ── Precompiled regexes used on the hot apply_patch path ─────────────────────
# Module-level so each call reuses the compiled pattern instead of recompiling.
_PATCH_PATH_PREFIX_RE = re.compile(r"^(?:a/|b/)")  # strip git diff a// b/ prefixes
_HUNK_HEADER_RE = re.compile(r'^@@ -([0-9]+),([0-9]+) \+([0-9]+),([0-9]+) @@')


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

    Robustness: any error degrades to ``None`` (never blocks a legitimate
    insert) — the pre-guard is a best-effort optimization, not a safety gate.
    """
    try:
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
    except Exception:
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
    the module-level construct the anchor lives in. Robustness: degrades to
    None entries; never raises. Used to surface "inserted inside scope X"
    feedback in anchor_edit metadata so the LLM can self-verify the structural
    correctness of an insert without a separate read_file round-trip.
    """
    out = {
        "innermost": (None, None, None),
        "top_level": (None, None, None),
        "anchor_indent": 0,
    }
    try:
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
                    name = stripped[len(hdr):].split("(", 1)[0].split(":", 1)[0].strip()
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
    except Exception:
        pass
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

    1. Nested-in-function — is it a lexical child of a FunctionDef /
       AsyncFunctionDef? Landing inside an unrelated function is essentially
       never the intent for a snippet that itself introduces a new def/class
       — the intent is always sibling/module (or class-body) level, never
       "define a new nested helper inside someone else's function" via
       anchor_edit.
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
    except SyntaxError:
        return None

    _violations = []

    def _walk(node, func_stack):
        for child in _ast.iter_child_nodes(node):
            _is_def = isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
            if _is_def:
                _lineno0 = child.lineno - 1
                _introduced = insert_start_line <= _lineno0 < insert_end_line
                if _introduced:
                    if func_stack:
                        _violations.append(("nested_in_function", child.name, func_stack[-1]))
                    _end_lineno = getattr(child, "end_lineno", None)
                    if _end_lineno is not None and _end_lineno > insert_end_line:
                        _violations.append(("swallowed_trailing_code", child.name, None))
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                _walk(child, func_stack + [child.name])
            else:
                _walk(child, func_stack)

    try:
        _walk(tree, [])
    except Exception:
        return None

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
    _m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
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
    _quote: Optional[str] = None  # active opening quote char, or None
    _escape = False
    for ch in text:
        if _escape:
            _escaped.append(ch)
            _escape = False
            continue
        if ch == '\\':
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
        if _quote is not None and ch == '\n':
            _escaped.append('\\n')
            continue
        if _quote is not None and ch == '\r':
            _escaped.append('\\r')  # CRLF: pair the \n escape so loads() succeeds
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
        if ch == '\\':
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
        if _in_str and ch == '\\':
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
        seg = re.sub(r',\s*([}\]])', r'\1', seg)
        # Unquoted keys (key: → "key":) at start of object / after comma
        seg = re.sub(r'([\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', seg)
        return seg

    text = "".join(
        seg if is_str else _fix_code_segment(seg)
        for is_str, seg in _segments
    )

    return text

try:
    from ...patch_engine import PatchContext, PatchEngine
except ImportError:
    PatchEngine = None  # type: ignore
    PatchContext = None  # type: ignore


class WriteToolsMixin:
    """Mixin providing write tool implementations for ToolRegistry."""

    # ── Plan normalizer class variables ─────────────────────────────── #
    _ACTION_TO_OP: dict[str, str] = {
        "insert": "insert_after",
        "append": "insert_after",
        "prepend": "insert_before",
        "add": "insert_after",
        "replace": "replace_file",
        "edit": "edit_blocks",
        "modify": "edit_blocks",
        "update": "edit_blocks",
        "patch": "edit_blocks",
        "change": "edit_blocks",
        "create": "create_file",
        "write": "create_file",
        "new": "create_file",
        "overwrite": "replace_file",
        "rewrite": "replace_file",
        "delete_content": "edit_blocks",
    }

    _WRITE_PLAN_OP_TYPES = frozenset({
        "create_file",
        "replace_file",
        "edit_blocks",
        "insert_after",
        "insert_before",
        "insert_after_line",
    })

    _OP_TYPE_ALIASES: dict[str, str] = {
        "createfile": "create_file",
        "replacefile": "replace_file",
        "replace": "replace_file",
        "editblocks": "edit_blocks",
        "insertafter": "insert_after",
        "insertbefore": "insert_before",
    }

    # NOTE: short/lowercase tokens that can legitimately appear in real code
    # (e.g. "..." Python ellipsis, "old_code"/"new_code" as actual identifiers,
    # "old text"/"new text"/"new code" inside docstrings) are excluded to avoid
    # rejecting valid edits.
    _PLACEHOLDER_BEFORE = frozenset({
        "OLD TEXT", "ORIGINAL CODE", "EXISTING CODE", "CURRENT CODE",
        "YOUR CODE HERE", "REPLACE THIS", "...\n...", "old code",
        "existing code", "current content", "put old code here",
    })
    _PLACEHOLDER_AFTER = frozenset({
        "NEW TEXT", "UPDATED CODE", "NEW CODE", "REPLACEMENT CODE",
        "YOUR NEW CODE HERE", "updated content",
        "put new code here",
    })

    # ── Path normalizer helpers ──────────────────────────────────────── #

    def _normalize_op_path(self, raw_path: str, repairs: list[str]) -> str:
        if not raw_path:
            return raw_path
        p = raw_path.replace("\\", "/")
        repo = str(self.repo_root).rstrip("/") + "/"
        if p.startswith(repo):
            rel = p[len(repo):]
            repairs.append(f"abs path stripped repo prefix→{rel!r}")
            return rel
        if p.startswith("/"):
            rel = p.lstrip("/")
            repairs.append(f"abs path→rel: {rel!r}")
            return rel
        return raw_path

    def _normalize_plan_op(self, op: dict, repairs: list[str]) -> dict:
        """Normalize a single plan op dict for small-model compat."""
        op = dict(op)

        if "path" in op:
            op["path"] = self._normalize_op_path(str(op["path"]), repairs)

        if "action" in op and "op" not in op and "type" not in op:
            action = str(op.pop("action")).lower().strip()
            mapped = self._ACTION_TO_OP.get(action, action)
            op["op"] = mapped
            repairs.append(f"action:{action!r}→op:{mapped!r}")

        op_type = str(op.get("op") or op.get("type") or "").lower().replace("-", "_")

        if op_type in ("insert_after", "insert_before"):
            if "anchor" not in op and ("start_line" in op or "line" in op):
                path = str(op.get("path") or "")
                line_no = int(op.get("start_line") or op.get("line") or 1)
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        file_lines = fp.read_text(encoding="utf-8").splitlines()
                        idx = max(0, line_no - 1)
                        if idx < len(file_lines):
                            op["anchor"] = file_lines[idx]
                            repairs.append(f"line {line_no}→anchor:{op['anchor']!r}")
                except Exception:
                    pass
                op.pop("start_line", None)
                op.pop("end_line", None)
                op.pop("line", None)

            if "content" in op and "lines" not in op:
                content = op.pop("content")
                if isinstance(content, str):
                    op["lines"] = content.splitlines() or [""]
                elif isinstance(content, list):
                    op["lines"] = content
                repairs.append("content→lines")

        if op_type in ("insert_after", "insert_before") and "lines" in op:
            if isinstance(op["lines"], str):
                op["lines"] = op["lines"].splitlines() or [op["lines"]]
                repairs.append("lines string→list")

        if op_type == "insert_after_line":
            if "content" in op and "lines" not in op:
                content = op.pop("content")
                if isinstance(content, str):
                    op["lines"] = content.splitlines() or [""]
                elif isinstance(content, list):
                    op["lines"] = content
                repairs.append("insert_after_line: content→lines")
            if "line" not in op and "start_line" in op:
                op["line"] = int(op.pop("start_line"))
                repairs.append("insert_after_line: start_line→line")
            if isinstance(op.get("lines"), str):
                op["lines"] = op["lines"].splitlines() or [op["lines"]]
                repairs.append("insert_after_line: lines string→list")

        if op_type == "edit_blocks" and "before" in op and "blocks" not in op and "edits" not in op:
            op["blocks"] = [{"before": op.pop("before"), "after": op.pop("after", "")}]
            repairs.append("before/after→blocks")

        if op_type == "edit_blocks":
            # 1. blocks dict→list
            raw_blocks = op.get("blocks") or op.get("edits")
            if isinstance(raw_blocks, dict):
                op["blocks"] = [raw_blocks]
                op.pop("edits", None)
                repairs.append("blocks dict→list")

            # 2. blocks alias normalization + line→before
            _BEFORE_ALIASES = ("old", "original", "from", "search",
                               "replace_this", "find", "source", "existing")
            _AFTER_ALIASES  = ("new", "new_content", "replacement", "to", "with",
                               "replace_with", "substitute", "target", "updated")
            blocks = op.get("blocks") or op.get("edits") or []
            if isinstance(blocks, list) and blocks:
                new_blocks = []
                for blk in blocks:
                    if not isinstance(blk, dict):
                        new_blocks.append(blk)
                        continue
                    blk = dict(blk)
                    if not blk.get("before"):
                        for alias in _BEFORE_ALIASES:
                            if alias in blk:
                                blk["before"] = blk.pop(alias)
                                repairs.append(f"{alias}→before")
                                break
                    if blk.get("after") is None:
                        for alias in _AFTER_ALIASES:
                            if alias in blk:
                                blk["after"] = blk.pop(alias)
                                repairs.append(f"{alias}→after")
                                break
                    # Block-level start_line/end_line → before (file read)
                    if not blk.get("before") and ("start_line" in blk or "line" in blk):
                        _path = str(op.get("path") or "")
                        _start = int(blk.get("start_line") or blk.get("line") or 1)
                        _end = int(blk.get("end_line") or _start)
                        try:
                            _fp = Path(self.repo_root) / _path
                            if _fp.exists():
                                _file_lines = _fp.read_text(encoding="utf-8").splitlines()
                                blk["before"] = "\n".join(_file_lines[_start - 1:_end])
                                blk.pop("start_line", None)
                                blk.pop("end_line", None)
                                blk.pop("line", None)
                                repairs.append(f"blk line {_start}-{_end}→before")
                        except Exception:
                            pass
                    new_blocks.append(blk)
                op["blocks"] = new_blocks
                op.pop("edits", None)

            # 3. strip line-number prefixes
            _ln_re = re.compile(r"^\d+:\s?")
            blocks = op.get("blocks") or []
            if isinstance(blocks, list) and blocks:
                new_blocks = []
                any_stripped = False
                for blk in blocks:
                    if not isinstance(blk, dict):
                        new_blocks.append(blk)
                        continue
                    blk = dict(blk)
                    for field in ("before", "after"):
                        val = blk.get(field)
                        if isinstance(val, str):
                            lines_in = val.splitlines()
                            lines_out = [_ln_re.sub("", ln) for ln in lines_in]
                            if lines_out != lines_in:
                                blk[field] = "\n".join(lines_out)
                                any_stripped = True
                    new_blocks.append(blk)
                if any_stripped:
                    op["blocks"] = new_blocks
                    repairs.append("stripped line-number prefixes from blocks")

            # 4. before+indent enrichment
            blocks = op.get("blocks") or []
            path = str(op.get("path") or "")
            if isinstance(blocks, list) and blocks and path:
                fp = Path(self.repo_root) / path
                try:
                    if fp.exists():
                        file_lines = fp.read_text(encoding="utf-8").splitlines()
                        any_enriched = False
                        new_blocks = []
                        for blk in blocks:
                            if not isinstance(blk, dict):
                                new_blocks.append(blk)
                                continue
                            before = blk.get("before", "")
                            if before and isinstance(before, str):
                                before_lines = before.splitlines()
                                if (
                                    len(before_lines) == 1
                                    and before[:1] not in (" ", "\t")
                                ):
                                    stripped = before.strip()
                                    matches = [
                                        (i, ln)
                                        for i, ln in enumerate(file_lines)
                                        if ln.strip() == stripped
                                    ]
                                    if len(matches) == 1:
                                        blk = dict(blk)
                                        blk["before"] = matches[0][1]
                                        repairs.append(
                                            f"before+indent (unique): {blk['before']!r:.60}"
                                        )
                                        any_enriched = True
                                    # multi-match: do NOT silently prepend context — let LLM
                                    # provide a more specific before block via error feedback
                            new_blocks.append(blk)
                        if any_enriched:
                            op["blocks"] = new_blocks
                except Exception:
                    pass  # non-critical enrichment — never block patch application

        if op_type == "edit_blocks" and "blocks" not in op and "edits" not in op and "before" not in op:
            if "start_line" in op or "end_line" in op:
                path = str(op.get("path") or "")
                start = int(op.get("start_line") or 1)
                end = int(op.get("end_line") or start)
                content = op.get("content", "")
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        file_lines = fp.read_text(encoding="utf-8").splitlines()
                        before_lines = file_lines[start - 1:end]
                        before_text = "\n".join(before_lines)
                        after_text = content if isinstance(content, str) else "\n".join(content)
                        op["blocks"] = [{"before": before_text, "after": after_text}]
                        op.pop("start_line", None)
                        op.pop("end_line", None)
                        op.pop("content", None)
                        repairs.append(f"line_range {start}-{end}→edit_blocks")
                except Exception:
                    pass  # non-critical line range → edit_blocks conversion

        # NOTE: empty edit_blocks (all blocks missing 'before') are left as-is so that
        # validation below rejects them with a clear error message — do NOT silently
        # convert to insert_before with an auto-chosen anchor, as that distorts LLM intent.

        return op

    def _detect_placeholder_op(self, op: dict[str, Any]) -> Optional[str]:
        if not isinstance(op, dict):
            return None
        op_type = str(op.get("op") or op.get("type") or "").lower().replace("-", "_")
        if op_type != "edit_blocks":
            return None
        for blk in (op.get("blocks") or []):
            if not isinstance(blk, dict):
                continue
            before = str(blk.get("before") or "").strip()
            after = str(blk.get("after") or "").strip()
            if before in self._PLACEHOLDER_BEFORE:
                return (
                    f"'before' contains a placeholder value ({before!r}). "
                    "You must read the target file first and copy the EXACT text "
                    "you want to replace into 'before'."
                )
            if after in self._PLACEHOLDER_AFTER:
                return (
                    f"'after' contains a placeholder value ({after!r}). "
                    "Replace 'after' with the actual new content you want."
                )
        return None

    def _enrich_plan_error(self, plan: Any, error_str: str) -> str:
        if not isinstance(plan, dict):
            return ""
        ops = plan.get("ops") or plan.get("operations") or []
        if not isinstance(ops, list) or not ops:
            return ""

        hints: list[str] = []

        for op in ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("op") or op.get("type") or "").lower().replace("-", "_")
            path = str(op.get("path") or "")

            if op_type == "edit_blocks" and "missing" in error_str.lower() and "before" in error_str.lower():
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        file_content = fp.read_text(encoding="utf-8")
                        preview = file_content[:1500]
                        if len(file_content) > 1500:
                            preview += "\n... (truncated)"
                        hints.append(
                            f"HINT: For edit_blocks on '{path}', the 'before' field must contain "
                            f"exact text from the file. Current file content:\n```\n{preview}\n```\n"
                            f"Copy the exact lines you want to replace into 'before', and put "
                            f"the replacement text in 'after'.\n"
                            f"To INSERT a new line without replacing anything, use "
                            f"op='insert_before' or op='insert_after' with an 'anchor' line and "
                            f"a 'lines' list instead."
                        )
                except Exception:
                    pass  # non-critical: error message building must not block

            if op_type == "edit_blocks" and "not found" in error_str.lower():
                try:
                    import difflib
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        file_lines = fp.read_text(encoding="utf-8").splitlines()
                        before_text = ""
                        for blk in (op.get("blocks") or []):
                            if isinstance(blk, dict) and blk.get("before"):
                                before_text = str(blk["before"])
                                break
                        before_first = before_text.splitlines()[0].strip() if before_text else ""

                        ctx_hint = ""
                        if before_first:
                            close = difflib.get_close_matches(before_first, file_lines, n=1, cutoff=0.5)
                            if close:
                                idx = next((i for i, _item_ in enumerate(file_lines) if _item_ == close[0]), -1)
                                if idx >= 0:
                                    start = max(0, idx - 2)
                                    end = min(len(file_lines), idx + 8)
                                    ctx = "\n".join(
                                        f"{start+j+1:4d}: {file_lines[start+j]}"
                                        for j in range(end - start)
                                    )
                                    ctx_hint = (
                                        f"\nClosest match found near line {idx+1}:\n```\n{ctx}\n```\n"
                                        f"Copy the EXACT text from this block into 'before'."
                                    )

                        if ctx_hint:
                            hints.append(
                                f"HINT: 'before' text not found in '{path}'. "
                                f"Your text did not match the file.{ctx_hint}"
                            )
                        else:
                            preview = "\n".join(
                                f"{i+1:4d}: {_item_}" for i, _item_ in enumerate(file_lines[:60])
                            )
                            hints.append(
                                f"HINT: 'before' text not found in '{path}'. "
                                f"Use find_symbol to locate the target, then bash (cat) at the "
                                f"returned line. First 60 lines:\n```\n{preview}\n```"
                            )
                except Exception:
                    pass  # non-critical: error message building must not block

            if op_type == "create_file" and "already exists" in error_str.lower():
                hints.append(
                    f"HINT: '{path}' already exists. Use op='replace_file' to overwrite it, "
                    f"or op='edit_blocks' to make partial changes."
                )

            if op_type in ("insert_after", "insert_before") and "anchor" in error_str.lower():
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        first_lines = fp.read_text(encoding="utf-8").splitlines()[:10]
                        preview = "\n".join(first_lines)
                        hints.append(
                            f"HINT: For {op_type} on '{path}', 'anchor' must be an exact line from "
                            f"the file. First 10 lines:\n```\n{preview}\n```"
                        )
                except Exception:
                    pass  # non-critical: error message building must not block

        return "\n".join(hints)

    def _looks_like_unified_diff(self, text: str) -> bool:
        t = str(text or '')
        if not t.strip():
            return False
        has_header = any(s in t for s in ('diff --git ', '--- a/', '+++ b/')) or t.lstrip().startswith('--- ')
        has_hunk = ('@@ ' in t)
                # Allow hunk-only patches (starting with @@, no header) — git apply handles them
        return bool(has_hunk and (has_header or t.lstrip().startswith('@@')))

    def _write_staged_files_directly(
        self, staged: dict[str, str], picked_files: list[str],
    ) -> "ToolResult":
        """Apply a compiled plan by writing each file's final content directly.

        plan_compiler already computed the exact post-edit content of every
        touched file; this writes it to disk without going through `git apply`,
        sidestepping that path's failure modes (context fuzz, trailing-newline
        mismatch, untracked/gitignored paths, missing --3way blob).

        Safety mirrors the git-apply path: snapshot existing files, write, run
        py_compile on touched .py files, and roll everything back (restore
        snapshots, delete created files) if any syntax error is introduced.
        Files whose content is unchanged are skipped (not touched, not counted).
        """
        import os as _os
        repo = str(self._effective_repo_root)
        snapshots: dict[str, str] = {}   # abs_path -> original content (existing files)
        created: list[str] = []          # abs_paths that did not exist before
        written: list[str] = []          # rel_paths actually written (changed)

        def _rollback() -> None:
            for _ap, _orig in snapshots.items():
                try:
                    atomic_write_text(_ap, _orig)
                except OSError as _re:
                    logger.debug("write_plan rollback restore failed for %s: %s", _ap, _re)
            for _ap in created:
                try:
                    _os.remove(_ap)
                except OSError as _re:
                    logger.debug("write_plan rollback remove failed for %s: %s", _ap, _re)

        try:
            for rel in (picked_files or list(staged.keys())):
                if rel not in staged:
                    continue
                new_content = staged[rel]
                ap = _os.path.join(repo, rel)
                if _os.path.isfile(ap):
                    try:
                        with open(ap, encoding="utf-8", errors="replace") as _fh:
                            cur = _fh.read()
                    except OSError as _re:
                        # The file exists but we cannot read it (permission denied,
                        # race, etc.). We therefore cannot snapshot it, so
                        # overwriting would break the rollback contract (a later
                        # syntax error could not restore it). Abort; the outer
                        # handler rolls back everything written so far.
                        raise OSError(
                            f"cannot read existing file {rel} for snapshot: {_re}"
                        ) from _re
                    if cur == new_content:
                        continue  # unchanged — don't touch
                    snapshots[ap] = cur
                else:
                    _parent = _os.path.dirname(ap)
                    if _parent and not _os.path.isdir(_parent):
                        _os.makedirs(_parent, exist_ok=True)
                    created.append(ap)
                atomic_write_text(ap, new_content)
                written.append(rel)
        except Exception as exc:
            _rollback()
            return self._make_result(
                ok=False, content="",
                error=f"Direct write failed: {type(exc).__name__}: {exc}",
                metadata={"rolled_back": True},
            )

        # Post-write syntax validation for .py (mirrors git-apply rollback gate).
        _syntax_errors: list[str] = []
        for rel in written:
            if LanguageId.from_path(rel) is LanguageId.PYTHON:
                try:
                    compile_quiet(staged[rel], rel, "exec")
                except SyntaxError as _se:
                    _syntax_errors.append(f"{rel}: {_se}")
                except Exception as _exc:
                    logger.debug("write_plan direct-write compile() non-SyntaxError for %s: %s", rel, _exc)
        if _syntax_errors:
            _rollback()
            return self._make_result(
                ok=False, content="",
                error=f"Plan introduced syntax errors (rolled back): {'; '.join(_syntax_errors)}",
                metadata={"syntax_errors": _syntax_errors, "rolled_back": True},
            )

        if written:
            try:
                self._invalidate_cache_after_write(written)
            except Exception as _exc:
                logger.debug("write_plan cache invalidation failed: %s", _exc)
        return self._make_result(
            ok=True,
            content=f"Wrote {len(written)} file(s) directly",
            metadata={"touched_files": written},
        )

    # ── Main write tools ─────────────────────────────────────────────── #

    def _tool_write_plan(self, args: dict[str, Any]) -> "ToolResult":
        if "__raw_arguments" in args and "plan" not in args:
            import json as _json
            _raw = args["__raw_arguments"]
            if isinstance(_raw, str):
                try:
                    _parsed = _json.loads(_raw)
                    if isinstance(_parsed, dict):
                        args = _parsed
                        logger.info("write_plan: recovered args from __raw_arguments")
                except (ValueError, _json.JSONDecodeError):
                    _repaired = _repair_plan_json(_raw)
                    _start = -1
                    try:
                        _parsed = _json.loads(_repaired)
                        if isinstance(_parsed, dict):
                            args = _parsed
                            logger.info("write_plan: recovered args from __raw_arguments (repaired)")
                    except (ValueError, _json.JSONDecodeError):
                        _start = _raw.find('{')
                    _end = _raw.rfind('}')
                    if _start != -1 and _end > _start:
                        _sub = _raw[_start:_end + 1]
                        try:
                            _parsed = _json.loads(_sub)
                            if isinstance(_parsed, dict):
                                args = _parsed
                                logger.info("write_plan: recovered args from __raw_arguments (substring)")
                        except (ValueError, _json.JSONDecodeError):
                            # Try repair on substring too
                            _sub_repaired = _repair_plan_json(_sub)
                            try:
                                _parsed = _json.loads(_sub_repaired)
                                if isinstance(_parsed, dict):
                                    args = _parsed
                                    logger.info("write_plan: recovered args from __raw_arguments (substring repaired)")
                            except (ValueError, _json.JSONDecodeError):
                                pass

        plan = args.get("plan")
        if not plan:
            ops = args.get("ops")
            if ops is None:
                ops = args.get("operations")
            if ops is None:
                _raw_hint = ""
                _raw = args.get("__raw_arguments", "")
                if isinstance(_raw, str) and len(_raw) > 10:
                    # Detect truncation: unclosed braces indicate the tool_call
                    # arguments were cut off mid-stream (common with large content fields).
                    _trimmed = _raw.strip()
                    _open_br = _trimmed.count("{")
                    _close_br = _trimmed.count("}")
                    if _open_br > _close_br:
                        return self._make_result(
                            ok=False, content="", error=(
                                f"write_plan: tool_call arguments were truncated "
                                f"({_open_br - _close_br} unclosed braces, {len(_trimmed)} chars). "
                                f"For large file creation/edits, use bash (python3/cat) to write "
                                f"the file directly, then use write_plan to update other files."
                            )
                        )
                    _raw_hint = f" (raw args: {_raw[:120]})"
                return self._make_result(
                    ok=False, content="", error=(
                        f"plan is required{_raw_hint}. Correct format:\n"
                        'write_plan({"plan": {"kind": "ASICODE_PLAN_V1", "ops": ['
                        '{"op": "create_file", "path": "path/to/file.py", "content": "file content here"}'
                        "]}})\n"
                        "For new files use op='create_file'. For patches use op='patch' with unified diff."
                    )
                )
            plan = {"kind": "ASICODE_PLAN_V1", "ops": ops}

        # Normalize string/JSON/list plan to dict (LLM may pass raw JSON string, markdown block, or bare list)
        if isinstance(plan, str):
            stripped = plan.strip()
            md_m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', stripped)
            if md_m:
                stripped = md_m.group(1).strip()
            try:
                plan = json.loads(stripped)
            except json.JSONDecodeError:
                # Try repair before giving up
                _repaired = _repair_plan_json(stripped)
                try:
                    plan = json.loads(_repaired)
                    logger.info("write_plan: repaired JSON via _repair_plan_json")
                except json.JSONDecodeError:
                    # Include first 200 chars of input for debugging
                    _sample = stripped[:200].replace("\n", "\\n")
                    return self._make_result(ok=False, content="", error=(
                        f"plan must be a valid JSON object. Received: {_sample}"
                    ))
        if isinstance(plan, list):
            plan = {"kind": "ASICODE_PLAN_V1", "ops": plan}

        # Guard BEFORE any plan.get(...) call below: json.loads above can yield a
        # non-dict scalar (int/float/bool/str/None from inputs like {"plan": "42"}
        # or {"plan": true}), and plan.get(...) on such a value raises
        # AttributeError — surfacing as a generic dispatch error instead of the
        # intended "plan must be a JSON object" rejection.
        if not isinstance(plan, dict):
            return self._make_result(ok=False, content="", error="plan must be a JSON object")

        # Normalize ops for plan_compiler compatibility (path, field aliases, etc.)
        repairs: list[str] = []
        ops = plan.get("ops") or plan.get("operations") or []
        for i, op in enumerate(ops):
            if isinstance(op, dict):
                ops[i] = self._normalize_plan_op(op, repairs)
        if repairs:
            logger.info("write_plan normalized ops: %s", "; ".join(repairs))

        kind = plan.get("kind") or plan.get("version")
        if not kind or str(kind).strip() != "ASICODE_PLAN_V1":
            return self._make_result(ok=False, content="", error="plan must have 'kind' or 'version' field set to 'ASICODE_PLAN_V1'")

        ops = plan.get("ops") or plan.get("operations")
        if not ops:
            return self._make_result(ok=False, content="", error="plan must have non-empty 'ops' or 'operations' list")
        if not isinstance(ops, list):
            return self._make_result(ok=False, content="", error="'ops' or 'operations' must be a list")
        if len(ops) == 0:
            return self._make_result(ok=False, content="", error="'ops' or 'operations' list must not be empty")

        for op_idx, op in enumerate(ops or []):
            # ── Phase 1: op must be a dict ───────────────────────────────
            if not isinstance(op, dict):
                return self._make_result(
                    ok=False, content="", error=(
                        f"write_plan rejected: ops[{op_idx}] is not a JSON object "
                        f"(type={type(op).__name__!r}).\n"
                        "ACTION: Each op must be a JSON object with 'op', 'path', "
                        "and type-specific fields."
                    ),
                )

            # ── Phase 2: Placeholder content detection ──────────────────
            ph_err = self._detect_placeholder_op(op)
            if ph_err:
                return self._make_result(
                    ok=False, content="", error=(
                        f"write_plan rejected: {ph_err}\n"
                        "ACTION: Use bash (cat) on the target file first, then use the actual "
                        "text from the file in 'before', and your desired replacement in 'after'."
                    ),
                )

            # ── Phase 3: Path validation ────────────────────────────────
            path_val = op.get("path")
            if path_val is None or str(path_val).strip() == "":
                op_type_info = str(op.get("op") or op.get("type") or "(unknown)")
                return self._make_result(
                    ok=False, content="", error=(
                        f"write_plan rejected: ops[{op_idx}] (type={op_type_info!r}) has missing or empty 'path'.\n"
                        "ACTION: Every op must include a 'path' field with the relative file path "
                        "(e.g. 'external_llm/agent/example.py'). Add the correct path and retry write_plan."
                    ),
                )
            if ".." in str(path_val).split("/"):
                return self._make_result(
                    ok=False, content="", error=(
                        f"write_plan rejected: ops[{op_idx}] has path traversal ('..') in "
                        f"path={path_val!r}.\n"
                        "ACTION: Use a relative path within the repository, without '..' segments."
                    ),
                )

            # ── Phase 4: Op type validation ─────────────────────────────
            raw_op_type = str(op.get("op") or op.get("type") or "").strip()
            if not raw_op_type:
                return self._make_result(
                    ok=False, content="", error=(
                        f"write_plan rejected: ops[{op_idx}] is missing 'op' or 'type' field.\n"
                        f"ACTION: Add an 'op' field. Supported types: "
                        f"{', '.join(sorted(self._WRITE_PLAN_OP_TYPES))}."
                    ),
                )
            # Normalize the same way plan_compiler._norm_op_type does
            op_type = raw_op_type.lower().replace("-", "_").replace(" ", "_")
            op_type = "".join(ch for ch in op_type if (ch.isalnum() or ch == "_"))
            op_type = self._OP_TYPE_ALIASES.get(op_type, op_type)
            if op_type not in self._WRITE_PLAN_OP_TYPES:
                return self._make_result(
                    ok=False, content="", error=(
                        f"write_plan rejected: ops[{op_idx}] has unsupported op type "
                        f"{raw_op_type!r} (normalized={op_type!r}).\n"
                        f"ACTION: Use one of the supported types: "
                        f"{', '.join(sorted(self._WRITE_PLAN_OP_TYPES))}."
                    ),
                )

            # ── Phase 5: Per-op required field validation ───────────────
            if op_type in ("create_file", "replace_file"):
                if "content" not in op or op.get("content") is None:
                    return self._make_result(
                        ok=False, content="", error=(
                            f"write_plan rejected: ops[{op_idx}] ({op_type}) is missing "
                            f"'content' field.\n"
                            "ACTION: Add a 'content' field with the full file content."
                        ),
                    )

            if op_type == "edit_blocks":
                edits = op.get("edits") or op.get("blocks")
                if not isinstance(edits, list) or not edits:
                    return self._make_result(
                        ok=False, content="", error=(
                            f"write_plan rejected: ops[{op_idx}] (edit_blocks) is missing "
                            f"non-empty 'edits' or 'blocks' list.\n"
                            "ACTION: Add an 'edits' list with 'before'/'after' pairs."
                        ),
                    )

            if op_type in ("insert_after", "insert_before"):
                if not op.get("anchor"):
                    return self._make_result(
                        ok=False, content="", error=(
                            f"write_plan rejected: ops[{op_idx}] ({op_type}) is missing "
                            f"'anchor' field.\n"
                            "ACTION: Add an 'anchor' field with an exact line from the target "
                            "file. Then add 'lines' with the text to insert."
                        ),
                    )
                lines = op.get("lines")
                if not isinstance(lines, list) or not lines:
                    return self._make_result(
                        ok=False, content="", error=(
                            f"write_plan rejected: ops[{op_idx}] ({op_type}) has missing or "
                            f"non-list 'lines' field.\n"
                            "ACTION: Add a 'lines' list with the text to insert."
                        ),
                    )

            if op_type == "insert_after_line":
                op_line = op.get("line")
                if not isinstance(op_line, int) or op_line < 1:
                    return self._make_result(
                        ok=False, content="", error=(
                            f"write_plan rejected: ops[{op_idx}] (insert_after_line) is missing "
                            f"or invalid 'line' field. Must be a positive integer.\n"
                            "ACTION: Add a 'line' field with the 1-based line number."
                        ),
                    )
                lines = op.get("lines")
                if not isinstance(lines, list) or not lines:
                    return self._make_result(
                        ok=False, content="", error=(
                            f"write_plan rejected: ops[{op_idx}] (insert_after_line) has missing or "
                            f"non-list 'lines' field.\n"
                            "ACTION: Add a 'lines' list with the text to insert."
                        ),
                    )

        try:
            from plan_compiler import compile_plan_to_unified_diff
        except ImportError:
            return self._make_result(ok=False, content="", error="plan_compiler not available")

        def _compile_and_apply(p: dict[str, Any]) -> "ToolResult":
            try:
                result = compile_plan_to_unified_diff(
                    repo_root=str(self._effective_repo_root),
                    plan=p,
                    allow_empty=False,
                )
            except Exception as exc:
                return self._make_result(
                    ok=False, content="",
                    error=f"Plan compilation failed: {type(exc).__name__}: {exc}",
                )
            patch = result.diff_patch or ""
            warnings = result.warnings or []

            if not patch.strip():
                return self._make_result(ok=False, content="", error="Plan compiled to empty patch")

            # ── Apply by writing compiler-staged content directly (no git apply) ──
            # The compiler already produced the exact final content of every touched
            # file (result.staged). Writing it directly avoids the git-apply failure
            # modes the diff round-trip introduced — context fuzz, trailing-newline
            # mismatch, untracked/gitignored paths, missing --3way blob — none of
            # which reflect a real problem with the computed result. The diff is
            # still kept for the no-op gate above, line-count display, and the
            # applied_patches record. Snapshot + py_compile + rollback preserve the
            # same safety the git-apply path provided.
            apply_result = self._write_staged_files_directly(
                result.staged, result.picked_files,
            )
            if not apply_result.ok:
                _err_content = f"Plan compiled successfully but apply failed: {apply_result.error}"
                _err_metadata = {"patch": patch, "warnings": warnings, "files": result.picked_files}
                if apply_result.metadata:
                    for _k, _v in apply_result.metadata.items():
                        _err_metadata.setdefault(_k, _v)
                return self._make_result(
                    ok=False, content=_err_content,
                    error=apply_result.error, metadata=_err_metadata,
                )

            # Line counts come from the diff (display only).
            added_lines = removed_lines = 0
            try:
                for line in patch.split("\n"):
                    if line.startswith("+++") or line.startswith("---"):
                        continue
                    if line.startswith("+"):
                        added_lines += 1
                    elif line.startswith("-"):
                        removed_lines += 1
            except (AttributeError, TypeError):
                pass

            touched_files = (apply_result.metadata or {}).get("touched_files") or result.picked_files or []
            display_file = touched_files[0] if touched_files else "unknown"
            if added_lines > 0 or removed_lines > 0:
                line_info = f" (+{added_lines} lines, -{removed_lines} lines)"
            else:
                line_info = ""

            # Record the diff so agent_loop detects a real edit happened.
            try:
                if patch:
                    self._applied_patches.append(str(patch))
            except (AttributeError, TypeError):
                pass

            content_parts = [f"Plan applied. Touched: {display_file}{line_info}"]
            if warnings:
                content_parts.append("Warnings: " + "; ".join(warnings))
            return self._make_result(
                ok=True,
                content="\n".join(content_parts),
                metadata={
                    "patch": patch,
                    "warnings": warnings,
                    "files": result.picked_files,
                    "touched_files": touched_files,
                },
            )

        try:
            compile_result = _compile_and_apply(plan)
            if compile_result.ok:
                return compile_result

            first_error = compile_result.error or ""
            error_msg = first_error
            enriched = self._enrich_plan_error(plan, first_error)
            if enriched:
                error_msg = f"{error_msg}\n\n{enriched}"

            compile_result.error = error_msg
            return compile_result

        except Exception as e:
            error_msg = f"Plan compilation failed: {type(e).__name__}: {e}"
            enriched = self._enrich_plan_error(plan, str(e))
            if enriched:
                error_msg = f"{error_msg}\n\n{enriched}"
            return self._make_result(ok=False, content="", error=error_msg)

    # ── apply_patch → modify_symbol auto-fallback ──────────────────────────
    # When PatchEngine fails (e.g. context mismatch on a freshly-edited/
    # untracked file where `git apply --3way` has no pre-image blob to merge
    # against), inspect the unified diff: if it touches a SINGLE Python file
    # and replaces exactly ONE top-level def/class symbol, route it to
    # modify_symbol, which is AST-based and needs no git blob. This eliminates
    # the manual LLM retry loop for the most common single-symbol patch failure.

    _FALLBACK_PATCH_HUNK_RE = re.compile(
        r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*$'
    )

    def _parse_unified_diff_files(self, patch_text: str) -> list[dict[str, Any]]:
        """Split a unified diff into per-file hunks.

        Returns list of dicts: {file, hunks: [{old_start, old_count, new_start,
        new_count, lines: [(kind, text), ...]}]} where kind is '+', '-', or ' '.
        Conservative: returns [] on any structural ambiguity (binary patches,
        rename/copy headers, no recognizable hunks).
        """
        if not patch_text:
            return []
        # Reject rename/copy/mode-only patches — out of scope for symbol edit.
        if re.search(r'^(rename from|rename to|copy from|copy to|similarity index|new file mode|deleted file mode|old mode|new mode)\b', patch_text, re.MULTILINE):
            return []

        files: list[dict[str, Any]] = []
        cur_file: Optional[dict[str, Any]] = None
        cur_hunk: Optional[dict[str, Any]] = None

        for raw_line in patch_text.splitlines():
            if raw_line.startswith('diff --git '):
                # New file boundary. Defer file path to +++ header.
                cur_file = {"file": None, "hunks": []}
                files.append(cur_file)
                cur_hunk = None
                continue
            m_path = re.match(r'^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$', raw_line)
            if m_path and raw_line != '+++ /dev/null':
                if cur_file is None:
                    cur_file = {"file": None, "hunks": []}
                    files.append(cur_file)
                cur_file["file"] = m_path.group(1).strip()
                cur_hunk = None
                continue
            m_hunk = self._FALLBACK_PATCH_HUNK_RE.match(raw_line)
            if m_hunk:
                if cur_file is None:
                    cur_file = {"file": None, "hunks": []}
                    files.append(cur_file)
                cur_hunk = {
                    "old_start": int(m_hunk.group(1)),
                    "old_count": int(m_hunk.group(2)) if m_hunk.group(2) else 1,
                    "new_start": int(m_hunk.group(3)),
                    "new_count": int(m_hunk.group(4)) if m_hunk.group(4) else 1,
                    "lines": [],
                }
                cur_file["hunks"].append(cur_hunk)
                continue
            if cur_hunk is not None and (raw_line.startswith('+') or raw_line.startswith('-') or raw_line.startswith(' ')):
                # Context/added/removed line. Skip bare '---'/'+++' file headers.
                if raw_line.startswith('---') or raw_line.startswith('+++'):
                    continue
                cur_hunk["lines"].append((raw_line[0], raw_line[1:]))
                continue
            # Lines outside any hunk (e.g. 'Index:', diff metadata) — ignore.

        # Drop files with no resolved path or no hunks.
        files = [f for f in files if f["file"] and f["hunks"]]
        return files

    def _extract_new_file_target(self, patch_text: str, path_hint: Optional[str]) -> Optional[dict[str, Any]]:
        """Detect a new-file unified diff and extract its full content.

        A creation patch has no pre-image (``--- /dev/null`` and/or a
        ``new file mode`` header) and every body line is an addition, so the
        '+' lines ARE the complete file content — no context matching and no
        git blob are needed. Such a patch can be routed to create_file safely.

        Conservative: requires exactly ONE created file, a /dev/null or
        new-file-mode signal, and a body of pure '+' additions (any '-' or
        non-marker context line disqualifies it).

        Returns None (not a clean creation) or {file_path, content}.
        """
        if not patch_text:
            return None
        lines = patch_text.splitlines()
        is_creation_signal = any(
            _item_.strip() == "--- /dev/null" or _item_.startswith("new file mode")
            for _item_ in lines
        )
        if not is_creation_signal:
            return None

        # Resolve the created path from the +++ header (ignore /dev/null).
        new_path: Optional[str] = None
        plus_headers = 0
        for raw in lines:
            m = re.match(r'^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$', raw)
            if m and raw != '+++ /dev/null':
                plus_headers += 1
                new_path = m.group(1).strip()
        if plus_headers != 1:
            return None  # zero or multiple files — out of scope
        file_path = new_path or path_hint
        if not file_path:
            return None

        # Collect body: must be a pure-addition hunk set.
        content_lines: list[str] = []
        in_hunk = False
        for raw in lines:
            if raw.startswith('+++') or raw.startswith('---'):
                continue
            if self._FALLBACK_PATCH_HUNK_RE.match(raw):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if raw.startswith('\\'):
                # "\ No newline at end of file" marker — note and skip.
                continue
            if raw.startswith('+'):
                content_lines.append(raw[1:])
                continue
            if raw.strip() == "":
                # Blank line inside a creation hunk is an unprefixed empty
                # context line in some diff dialects — treat as empty content.
                content_lines.append("")
                continue
            # Any '-' (deletion) or ' ' context line means this is NOT a pure
            # creation; bail out so we don't fabricate content.
            return None

        if not content_lines:
            return None
        content = "\n".join(content_lines)
        # Preserve trailing newline unless a "no newline" marker was present.
        no_newline = any(_item_.startswith('\\') for _item_ in lines)
        if not no_newline:
            content += "\n"
        return {"file_path": file_path, "content": content}

    def _analyze_patch_symbol_change(self, patch_text: str) -> Optional[dict[str, Any]]:
        """Apply a failed patch to the on-disk file in memory and diff symbols.

        Shared core for the single- and multi-symbol fallbacks. Requires exactly
        ONE supported file whose hunks anchor cleanly against the current disk
        content (blob-free — this survives a missing git pre-image blob). Because
        the new source is built by splicing hunks into the real disk source,
        untouched lines are always preserved (no truncation/data-loss).

        Python files are located via the stdlib ``ast`` (PythonAstLocator); other
        languages via the tree-sitter ``SyntaxProvider`` registry
        (ProviderLocator), so the fallback is multi-language.

        Returns None (ineligible) or a dict:
          {file_path, abs_path, language, is_python, old_lines, new_lines,
           new_src, old_by_name, new_by_name, changed}
        where *_by_name map a top-level symbol's qualname to its source text and
        `changed` is the set of qualnames whose source text differs.
        """
        try:
            files = self._parse_unified_diff_files(patch_text)
        except Exception:
            return None
        if len(files) != 1:
            return None
        f = files[0]
        file_path = f["file"]

        # ── Resolve a locator: Python via ast, others via tree-sitter provider ──
        is_python = file_path.endswith(".py")
        language = "python"
        locator: Any
        if is_python:
            from external_llm.agent.symbol_locator import PythonAstLocator
            locator = PythonAstLocator()
        else:
            try:
                from external_llm.agent.symbol_locator import ProviderLocator
                from external_llm.languages.models import LanguageId
                from external_llm.languages.registry import LanguageRegistry
            except Exception:
                return None
            lang_id = LanguageId.from_path(file_path)
            if lang_id == LanguageId.UNKNOWN:
                return None  # unsupported language — surface original error
            language = lang_id.value
            provider = LanguageRegistry.instance().get(file_path)
            if provider is None:
                return None
            locator = ProviderLocator(provider)

        try:
            sec = self._secure_path(file_path)
            if sec is None:
                return None
            abs_path = str(sec)
            if not os.path.isfile(abs_path):
                return None
            with open(abs_path, encoding="utf-8") as fh:
                existing = fh.read()
        except Exception:
            return None

        old_lines = existing.split("\n")
        new_lines = self._apply_hunks_in_memory(old_lines, f["hunks"])
        if new_lines is None:
            return None  # a hunk couldn't be anchored — bail (no fabrication)
        new_src = "\n".join(new_lines)

        old_spans = [s for s in locator.locate(existing) if s.top_level]
        new_spans = [s for s in locator.locate(new_src) if s.top_level]
        if not new_spans and not old_spans:
            return None  # unparseable / no symbols on both sides → not safe

        def _src(lines: list[str], span) -> str:
            return "\n".join(lines[span.start_line - 1: span.end_line])

        old_by_name = {s.qualname: _src(old_lines, s) for s in old_spans}
        new_by_name = {s.qualname: _src(new_lines, s) for s in new_spans}
        changed = {
            name for name in (set(old_by_name) | set(new_by_name))
            if old_by_name.get(name) != new_by_name.get(name)
        }
        return {
            "file_path": file_path, "abs_path": abs_path,
            "language": language, "is_python": is_python,
            "old_lines": old_lines, "new_lines": new_lines, "new_src": new_src,
            "old_by_name": old_by_name, "new_by_name": new_by_name,
            "changed": changed,
        }

    def _extract_modify_symbol_target(self, patch_text: str, path_hint: Optional[str]) -> Optional[dict[str, Any]]:
        """Analyze a failed patch to see if it can route to modify_symbol.

        Eligible when the patch changes EXACTLY ONE top-level symbol that already
        exists in the file (a modify, not an add). See
        ``_analyze_patch_symbol_change`` for the shared, data-loss-free core.

        Returns None (ineligible) or a dict: {file_path, symbol, code, reason}
        """
        info = self._analyze_patch_symbol_change(patch_text)
        if info is None:
            return None
        if not info["is_python"]:
            return None  # modify_symbol is Python-AST-only; others use rewrite
        changed = info["changed"]
        if len(changed) != 1:
            return None  # zero or multiple symbols changed (multi-symbol → None)
        symbol = next(iter(changed))
        # Must be a MODIFY (symbol exists in both): an add/remove is out of scope.
        if symbol not in info["old_by_name"] or symbol not in info["new_by_name"]:
            return None

        code = info["new_by_name"][symbol]
        if not code or not code.strip():
            return None
        return {
            "file_path": info["file_path"],
            "symbol": symbol,
            "code": code,
            "reason": "single_python_symbol",
        }

    def _extract_multi_symbol_rewrite(self, patch_text: str, path_hint: Optional[str]) -> Optional[dict[str, Any]]:
        """See if a failed patch can be applied as a multi-symbol rewrite.

        Eligible when the patch changes TWO OR MORE top-level symbols (the
        single-symbol case is handled by modify_symbol). The complete new file
        was already reconstructed by ``_analyze_patch_symbol_change`` via
        content-anchored hunk application, so the only remaining safety bar is
        that it parses. Application is an atomic whole-file write (all-or-nothing
        by construction — no partial per-symbol state).

        Returns None (ineligible) or a dict:
          {file_path, abs_path, new_src, symbols}
        """
        info = self._analyze_patch_symbol_change(patch_text)
        if info is None:
            return None
        changed = info["changed"]
        # Python single-symbol edits are handled by modify_symbol (nicer diffs);
        # for other languages there is no modify_symbol, so the rewrite path owns
        # single edits too.
        min_changed = 2 if info["is_python"] else 1
        if len(changed) < min_changed:
            return None
        # The new file must be syntactically valid before we write it wholesale.
        if info["is_python"]:
            try:
                import ast as _ast
                _ast.parse(info["new_src"])
            except SyntaxError:
                return None
        else:
            try:
                from external_llm.languages.tree_sitter_utils import has_error
                err = has_error(info["new_src"], info["language"])
            except Exception:
                err = None
            # True → syntax error; None → tree-sitter unavailable (can't verify).
            # Bail in both cases; only a clean parse (False) is safe to write.
            if err is not False:
                return None
        return {
            "file_path": info["file_path"],
            "abs_path": info["abs_path"],
            "new_src": info["new_src"],
            "symbols": sorted(changed),
        }

    def _find_block(self, lines: list[str], block: list[str], hint: int = 0) -> Optional[int]:
        """Locate a contiguous ``block`` within ``lines``; return its start index.

        Tries exact match, then trailing-whitespace-insensitive, then fully
        whitespace-stripped — to tolerate the kind of drift that makes git apply
        fail. When several positions match, picks the one nearest ``hint`` (the
        hunk's expected location) to disambiguate repeated blocks. Returns None
        if the block cannot be found (caller must treat as ineligible).
        """
        if not block:
            return max(0, min(hint, len(lines)))
        n = len(block)
        if n > len(lines):
            return None
        for key in (lambda s: s, lambda s: s.rstrip(), lambda s: s.strip()):
            keyed = [key(b) for b in block]
            matches = [
                i for i in range(0, len(lines) - n + 1)
                if [key(lines[j]) for j in range(i, i + n)] == keyed
            ]
            if matches:
                return min(matches, key=lambda i: abs(i - hint))
        return None

    def _apply_hunks_in_memory(self, file_lines: list[str], hunks: list[dict[str, Any]]) -> Optional[list[str]]:
        """Apply unified-diff hunks to ``file_lines`` by content matching.

        Pure-Python, blob-free: each hunk's old-side block (context + deleted
        lines) is located in the evolving file by content (not line number, so
        it survives line drift) and spliced with the new-side block (context +
        added lines). Hunks are assumed ascending; a running delta keeps the
        search hint aligned as earlier splices shift line numbers.

        Returns the new file lines, or None if any hunk cannot be anchored.
        """
        result = list(file_lines)
        delta = 0
        for hunk in hunks:
            old_block = [t for k, t in hunk["lines"] if k in (" ", "-")]
            new_block = [t for k, t in hunk["lines"] if k in (" ", "+")]
            hint = hunk["old_start"] - 1 + delta
            idx = self._find_block(result, old_block, hint=hint)
            if idx is None:
                return None
            result = result[:idx] + new_block + result[idx + len(old_block):]
            delta += len(new_block) - len(old_block)
        return result

    def _try_apply_patch_create_file_fallback(
        self, patch_text: str, path_hint: Optional[str], original_error: str,
        start_time: float,
    ) -> "Optional[ToolResult]":
        """Route a failed new-file patch to create_file.

        Returns None when the patch is NOT a clean creation (so the caller can
        try the modify_symbol path). Otherwise returns a ToolResult: ok on a
        successful create, or the ORIGINAL error enriched with the create
        failure (e.g. the file already exists, which means it wasn't really a
        creation and the LLM should retry as a modify).
        """
        nf = self._extract_new_file_target(patch_text, path_hint)
        if nf is None:
            return None
        import time as _time
        try:
            result = self._tool_create_file({
                "path": nf["file_path"],
                "content": nf["content"],
            })
        except Exception as e:
            logger.warning("apply_patch create_file fallback raised: %s", e, exc_info=True)
            return self._make_result(
                ok=False, content="", error=original_error,
                execution_time=_time.monotonic() - start_time,
                metadata={"auto_fallback_attempted": "create_file",
                          "auto_fallback_exception": f"{type(e).__name__}: {e}"},
            )
        if result.ok:
            logger.info(
                "apply_patch auto-fallback to create_file succeeded: %s",
                nf["file_path"],
            )
            _meta = dict(result.metadata) if result.metadata else {}
            _meta.update({
                "auto_fallback_attempted": "create_file",
                "auto_fallback_reason": "new_file_patch",
                "file_path": nf["file_path"],
            })
            return self._make_result(
                ok=True,
                content=(
                    f"Patch applied via create_file fallback (new file {nf['file_path']}).\n"
                    f"Original apply_patch error: {original_error}"
                ),
                execution_time=_time.monotonic() - start_time,
                metadata=_meta,
            )
        logger.info(
            "apply_patch auto-fallback to create_file failed: %s", result.error,
        )
        return self._make_result(
            ok=False, content="",
            error=(
                f"{original_error}\n\n"
                f"[auto-fallback create_file also failed for {nf['file_path']}: {result.error}]"
            ),
            execution_time=_time.monotonic() - start_time,
            metadata={"auto_fallback_attempted": "create_file",
                      "auto_fallback_failed": True,
                      "auto_fallback_error": str(result.error)[:2000]},
        )

    def _try_apply_patch_multi_symbol_fallback(
        self, patch_text: str, path_hint: Optional[str], original_error: str,
        start_time: float,
    ) -> "Optional[ToolResult]":
        """Apply a multi-symbol patch as an atomic whole-file rewrite.

        Returns None when the patch is NOT a clean multi-symbol change (so the
        caller falls through to the single-symbol modify_symbol path). Otherwise
        writes the fully-reconstructed file (all-or-nothing) and returns the
        ToolResult.
        """
        ms = self._extract_multi_symbol_rewrite(patch_text, path_hint)
        if ms is None:
            return None
        import time as _time
        try:
            result = self._tool_create_file({
                "path": ms["file_path"],
                "content": ms["new_src"],
                "overwrite": True,
            })
        except Exception as e:
            logger.warning("apply_patch multi-symbol fallback raised: %s", e, exc_info=True)
            return self._make_result(
                ok=False, content="", error=original_error,
                execution_time=_time.monotonic() - start_time,
                metadata={"auto_fallback_attempted": "multi_symbol_rewrite",
                          "auto_fallback_exception": f"{type(e).__name__}: {e}"},
            )
        if result.ok:
            logger.info(
                "apply_patch auto-fallback to multi_symbol_rewrite succeeded: %s (%s)",
                ms["file_path"], ", ".join(ms["symbols"]),
            )
            _meta = dict(result.metadata) if result.metadata else {}
            _meta.update({
                "auto_fallback_attempted": "multi_symbol_rewrite",
                "auto_fallback_reason": "multi_symbol_patch",
                "file_path": ms["file_path"],
                "symbols": ms["symbols"],
            })
            _syn = self._run_syntax_check_for_file(ms["abs_path"])
            if not _syn.get("skipped"):
                _meta["syntax_check"] = _syn
            return self._make_result(
                ok=True,
                content=(
                    f"Patch applied via multi-symbol rewrite "
                    f"(symbols {', '.join(ms['symbols'])} in {ms['file_path']}).\n"
                    f"Original apply_patch error: {original_error}"
                ),
                execution_time=_time.monotonic() - start_time,
                metadata=_meta,
            )
        return self._make_result(
            ok=False, content="",
            error=(
                f"{original_error}\n\n"
                f"[auto-fallback multi-symbol rewrite also failed for "
                f"{ms['file_path']}: {result.error}]"
            ),
            execution_time=_time.monotonic() - start_time,
            metadata={"auto_fallback_attempted": "multi_symbol_rewrite",
                      "auto_fallback_failed": True,
                      "auto_fallback_error": str(result.error)[:2000]},
        )

    def _try_apply_patch_modify_symbol_fallback(
        self, patch_text: str, path_hint: Optional[str], original_error: str,
        start_time: float,
    ) -> "ToolResult":
        """Attempt modify_symbol as a fallback for a failed unified-diff patch.

        Returns a ToolResult. On success, metadata.auto_fallback_attempted marks
        the path taken. On ineligibility or failure, returns the ORIGINAL error
        so the caller (LLM) sees why the patch failed.
        """
        # ── New-file patch → create_file (no symbol to modify) ──
        nf = self._try_apply_patch_create_file_fallback(
            patch_text, path_hint, original_error, start_time,
        )
        if nf is not None:
            return nf

        # ── Multi-symbol patch → atomic whole-file rewrite ──
        ms = self._try_apply_patch_multi_symbol_fallback(
            patch_text, path_hint, original_error, start_time,
        )
        if ms is not None:
            return ms

        target = self._extract_modify_symbol_target(patch_text, path_hint)
        if target is None:
            # Not eligible — return original error with a skip marker.
            return self._make_result(
                ok=False, content="",
                error=original_error,
                metadata={"auto_fallback_attempted": None,
                          "auto_fallback_skipped_reason": "not_single_python_symbol"},
            )

        try:
            from external_llm.agent.symbol_modify_tool import modify_symbol as _do_modify
            sec = self._secure_path(target["file_path"])
            if sec is None:
                return self._make_result(ok=False, content="", error=f"Path traversal blocked: {target['file_path']}")
            abs_path = str(sec)
            success, diff_or_error, _new_content = _do_modify(
                abs_path, target["symbol"], target["code"],
                repo_root=str(self._effective_repo_root),
            )
            import time as _time
            execution_time = _time.monotonic() - start_time
            if success:
                rel_path = os.path.relpath(abs_path, str(self._effective_repo_root))
                _meta = {
                    "file_path": rel_path,
                    "symbol": target["symbol"],
                    "auto_fallback_attempted": "modify_symbol",
                    "auto_fallback_reason": target["reason"],
                    "diff_preview": diff_or_error[:25000] if diff_or_error else "",
                    "changed": True,
                }
                _syn = self._run_syntax_check_for_file(abs_path)
                if not _syn.get("skipped"):
                    _meta["syntax_check"] = _syn
                logger.info(
                    "apply_patch auto-fallback to modify_symbol succeeded: %s@%s",
                    rel_path, target["symbol"],
                )
                return self._make_result(
                    ok=True,
                    content=(
                        f"Patch applied via modify_symbol fallback (symbol '{target['symbol']}' in {rel_path}).\n"
                        f"Original apply_patch error: {original_error}\n"
                        f"Diff:\n{diff_or_error}"
                    ),
                    execution_time=execution_time,
                    metadata=_meta,
                )
            else:
                logger.info(
                    "apply_patch auto-fallback to modify_symbol failed: %s",
                    diff_or_error,
                )
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"{original_error}\n\n"
                        f"[auto-fallback modify_symbol also failed for "
                        f"{target['file_path']}@{target['symbol']}: {diff_or_error}]"
                    ),
                    execution_time=execution_time,
                    metadata={
                        "auto_fallback_attempted": "modify_symbol",
                        "auto_fallback_failed": True,
                        "auto_fallback_error": str(diff_or_error)[:2000],
                    },
                )
        except Exception as e:
            logger.warning("apply_patch auto-fallback raised: %s", e, exc_info=True)
            import time as _time
            return self._make_result(
                ok=False, content="",
                error=original_error,
                execution_time=_time.monotonic() - start_time,
                metadata={
                    "auto_fallback_attempted": "modify_symbol",
                    "auto_fallback_exception": f"{type(e).__name__}: {e}",
                },
            )

    def _tool_apply_patch(self, args: dict[str, Any]) -> "ToolResult":
        import time as _time
        start_time = _time.monotonic()
        args = self._recover_args_from_raw(args, ("patch", "path"))
        patch_text = args.get("patch", "")
        if not patch_text.strip():
            execution_time = _time.monotonic() - start_time
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(ok=False, content="", error=f"patch is empty{_raw_hint}", execution_time=execution_time)
        path = args.get("path")

        if isinstance(path, str):
            repo_root_str = str(self._effective_repo_root)
            if path.startswith(repo_root_str):
                path = path[len(repo_root_str):].lstrip("/")
            if path.startswith("/"):
                path = path.lstrip("/")
        if path is not None and not path.strip():
            path = None

        # ── Hard guard (MAIN entry point): reject BEFORE PatchEngine / diff_apply mutates
        # the working tree. The main path (_tool_apply_patch → PatchEngine.apply_patch →
        # diff_apply with skip_3way=True) reverts a DIRTY file to HEAD ("git checkout --")
        # while returning ok=False — silently DELETING uncommitted edits (e.g. just made
        # via edit_text / modify_symbol / anchor_edit this session). The fallback guard in
        # _apply_patch_text does NOT protect this path: PatchEngine is reached first.
        # Detect BEFORE any apply so the working tree is never mutated; post-apply
        # detection is meaningless (the revert itself makes the file match HEAD again).
        try:
            _mp_touched = extract_files_from_patch(patch_text)
        except Exception:
            _mp_touched = []
        if isinstance(path, str) and path and path not in _mp_touched:
            _mp_touched.append(path)
        # Opt D: refuse only files THIS SESSION already edited via a text-editing
        # tool (edit_text / modify_symbol / edit_ast / anchor_edit), tracked in
        # _text_edited_files. Those edits live in the working tree, but apply_patch /
        # diff_apply reconstructs hunk context from HEAD and, on a freshly-edited
        # target (skip_3way=True), _rollback() reverts the file to HEAD — silently
        # deleting the session edit. Unlike the prior git-dirty check, this does NOT
        # block user pre-existing uncommitted edits or unrelated dirty files in a
        # multi-file patch (less friction); it targets the agent's own consecutive
        # edits precisely.
        _mp_session = [p for p in _mp_touched if self._norm_repo_rel(p) in self._text_edited_files]
        if _mp_session:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    "apply_patch refused: target file(s) were already edited this session "
                    "via edit_text / modify_symbol / edit_ast / anchor_edit — those edits "
                    "live in the working tree but apply_patch would revert to HEAD on "
                    "conflict (silently losing them): "
                    + ", ".join(_mp_session)
                    + ". Continue editing these files with the same text-editing tool instead."
                ),
                execution_time=_time.monotonic() - start_time,
                metadata={
                    "refused_dirty_files": _mp_session,
                    "reason": "session_text_edit_overwrite_risk",
                },
            )

        try:
            from ...patch_engine import PatchContext, PatchEngine
            engine = PatchEngine(self._effective_repo_root)

            if not self._looks_like_unified_diff(patch_text):
                if path is None:
                    execution_time = _time.monotonic() - start_time
                    _raw_hint = ""
                    _raw = args.get("__raw_arguments", "")
                    if isinstance(_raw, str) and len(_raw) > 10:
                        _raw_hint = f" (raw args: {_raw[:120]})"
                    return self._make_result(
                        ok=False,
                        content="",
                        error=f"Non-diff patch input requires 'path' so PatchEngine can synthesize and apply it{_raw_hint}",
                        execution_time=execution_time,
                        metadata={
                            "reason": "missing_path_for_non_diff_input",
                            "source": "agent_apply_patch",
                        },
                    )

                patch_result = engine.synthesize_and_apply(patch_text, path, output_mode="auto")
                execution_time = _time.monotonic() - start_time

                if patch_result.success:
                    try:
                        patch_record = None
                        if patch_result.metadata and patch_result.metadata.get("patch"):
                            patch_record = patch_result.metadata.get("patch")
                        else:
                            patch_record = patch_text
                        if patch_record:
                            self._applied_patches.append(str(patch_record))
                    except (AttributeError, TypeError):
                        pass

                    _meta = dict(patch_result.metadata) if patch_result.metadata else {}
                    _syn = self._run_syntax_check_for_file(path)
                    if not _syn.get("skipped"):
                        _meta["syntax_check"] = _syn
                    return self._make_result(
                        ok=True,
                        content=patch_result.patch_applied or "Patch applied successfully",
                        execution_time=execution_time,
                        metadata=_meta
                    )

                logger.warning(
                    "PatchEngine synthesize_and_apply failed for non-diff input; "
                    "falling back to legacy apply path. error=%s",
                    patch_result.error,
                )
                legacy_result = self._apply_patch_text(patch_text, path_hint=path)
                legacy_result.metadata.setdefault("fallback_from_patch_engine", True)
                legacy_result.metadata.setdefault("patch_engine_error", patch_result.error)
                if legacy_result.execution_time < 1e-9:
                    legacy_result.execution_time = execution_time
                return legacy_result

            else:
                context = PatchContext(
                    original_request=None,
                    file_content=None,
                    llm_output=None,
                    output_mode="auto",
                    metadata={"source": "agent_apply_patch"}
                )
                patch_result = engine.apply_patch(patch_text, target_file=path, context=context)
                execution_time = _time.monotonic() - start_time

                if patch_result.success:
                    try:
                        patch_record = None
                        if patch_result.metadata and patch_result.metadata.get("patch"):
                            patch_record = patch_result.metadata.get("patch")
                        else:
                            patch_record = patch_text
                        if patch_record:
                            self._applied_patches.append(str(patch_record))
                    except (AttributeError, TypeError):
                        pass

                    _meta2 = dict(patch_result.metadata) if patch_result.metadata else {}
                    _check_path = path or (
                        patch_result.metadata.get("file") if patch_result.metadata else None
                    )
                    if _check_path:
                        _syn2 = self._run_syntax_check_for_file(_check_path)
                        if not _syn2.get("skipped"):
                            _meta2["syntax_check"] = _syn2
                    return self._make_result(
                        ok=True,
                        content=patch_result.patch_applied or "Patch applied successfully",
                        execution_time=execution_time,
                        metadata=_meta2
                    )
                else:
                    # ── Auto-fallback: try modify_symbol for single-symbol patches ──
                    # PatchEngine exhausted its repair ladder (plain apply, --3way,
                    # tolerant, re-anchor, AST/symbol repair). For untracked or
                    # freshly-edited files, `git apply --3way` cannot find the
                    # pre-image blob, so AST-based modify_symbol is the only path
                    # that works. Route single-Python-symbol patches there before
                    # surfacing the failure to the LLM.
                    _fb = self._try_apply_patch_modify_symbol_fallback(
                        patch_text, path,
                        patch_result.error or "Patch application failed",
                        start_time,
                    )
                    if _fb.ok:
                        try:
                            self._applied_patches.append(str(patch_text))
                        except (AttributeError, TypeError):
                            pass
                        return _fb
                    # Fallback ineligible or failed — return its enriched result
                    # (preserves original error + skip/attempt metadata).
                    _fb.metadata = {**(patch_result.metadata or {}), **_fb.metadata}
                    _fb.metadata.setdefault("failure_class", "patch_apply_failed")
                    return _fb
        except ImportError as e:
            logger.warning(f"PatchEngine not available, falling back to legacy apply: {e}")
            result = self._apply_patch_text(patch_text, path_hint=path)
            import time as _time2
            if result.execution_time < 1e-9:
                result.execution_time = _time2.time() - start_time
            return result
        except Exception as e:
            logger.exception("Unexpected error in patch engine")
            import time as _time3
            execution_time = _time3.time() - start_time
            return self._make_result(
                ok=False,
                content="",
                error=f"Patch engine error: {type(e).__name__}: {e}",
                execution_time=execution_time,
                metadata={"error_type": "patch_engine_exception"}
            )

    def _resolve_edit_anchor(
        self, modified: str, anchor: str, line: Optional[int] = None,
    ) -> tuple[int, str, float]:
        """Find anchor in modified text with exact-match fallback strategies.

        Returns (position, actual_anchor_text, match_ratio).
        match_ratio is always 0.0 (exact/line-hint/first-line matches only;
        fuzzy matching is disabled to avoid false-positive anchor resolution).
        Raises ValueError with close-match suggestions on complete failure.
        """
        import difflib

        _lines = modified.splitlines(True)

        # 1. Exact match — but the contract requires a UNIQUE anchor. Silently
        #    taking the first of several matches is the tool's worst failure mode
        #    (it edits an unintended, often mid-line, location). So:
        #      • exactly one match  → use it
        #      • multiple matches   → disambiguate via the line hint, else fail loud
        _count = modified.count(anchor)
        if _count == 1:
            return modified.find(anchor), anchor, 0.0
        if _count > 1:
            if line is not None and 1 <= line <= len(_lines):
                _target_byte = sum(len(_lines[i]) for i in range(line - 1))
                _positions: list[int] = []
                _p = modified.find(anchor)
                while _p != -1:
                    _positions.append(_p)
                    _p = modified.find(anchor, _p + 1)
                _best = min(_positions, key=lambda q: abs(q - _target_byte))
                logger.info(
                    "edit_file: anchor not unique (%d matches) — line hint %d → byte %d",
                    _count, line, _best,
                )
                return _best, anchor, 0.0
            raise ValueError(
                f"anchor not unique: found {_count} occurrences of {anchor[:60]!r}. "
                "Include 2-3 surrounding lines to make it unique, or pass a 'line' hint."
            )

        # 2. Line hint → use actual content at that line
        if line is not None and 1 <= line <= len(_lines):
            _line_content = _lines[line - 1].rstrip('\n\r')
            if _line_content:
                # Compute exact byte offset from line number (handles duplicates correctly)
                _byte_pos = sum(len(_lines[i]) for i in range(line - 1))
                if modified[_byte_pos:_byte_pos+len(_line_content)] == _line_content:
                    logger.info(
                        "edit_file: line hint %d → exact byte offset", line,
                    )
                    return _byte_pos, _line_content, 0.0

        # 3. First anchor line → strip-match in search window
        _first_line = anchor.split('\n')[0].strip()
        if _first_line:
            _search_start = 0
            _search_end = len(_lines)
            if line is not None:
                _search_start = max(0, line - 1 - 5)
                _search_end = min(len(_lines), line + 5)

            # Single-line anchor: enforce the same uniqueness contract as the
            # exact-match path. Silently taking the first of several stripped
            # matches is exactly the wrong-location failure mode the exact
            # path fails loud on — the fallback must not reintroduce it.
            if '\n' not in anchor:
                _strip_matches = [
                    _smi for _smi in range(_search_start, _search_end)
                    if _lines[_smi].strip() == _first_line
                ]
                if not _strip_matches:
                    pass  # fall through to the error path below
                else:
                    if len(_strip_matches) == 1:
                        _li = _strip_matches[0]
                    elif line is not None:
                        _li = min(_strip_matches, key=lambda q: abs(q - (line - 1)))
                        logger.info(
                            "edit_file: %d stripped matches — line hint %d → line %d",
                            len(_strip_matches), line, _li + 1,
                        )
                    else:
                        raise ValueError(
                            f"anchor matches {len(_strip_matches)} lines when ignoring "
                            f"indentation ({_first_line[:60]!r}). Include 2-3 surrounding "
                            "lines to make it unique, or pass a 'line' hint."
                        )
                    _raw = _lines[_li].rstrip('\n\r')
                    _li_byte_pos = sum(len(_lines[i]) for i in range(_li))
                    logger.info("edit_file: first-line match → line %d", _li + 1)
                    return _li_byte_pos, _raw, 0.0

            for _li in range(_search_start, _search_end):
                if _lines[_li].strip() == _first_line:
                    _raw = _lines[_li].rstrip('\n\r')
                    pos = modified.find(_raw)
                    if pos == -1:
                        continue
                    # Multi-line anchor: reconstruct from actual file content
                    _anchor_line_count = anchor.count('\n') + 1
                    if _anchor_line_count > 1 and _li + _anchor_line_count <= len(_lines):
                        _actual = "".join(_lines[_li:_li + _anchor_line_count]).rstrip('\n\r')
                        if modified.count(_actual) == 1:
                            _actual_pos = modified.find(_actual)
                            if _actual_pos != -1:
                                logger.info(
                                    "edit_file: first-line match → line %d (reconstructed)",
                                    _li + 1,
                                )
                                return _actual_pos, _actual, 0.0
                        # Progressive fallback: try shorter anchor suffix (2+ lines).
                        # Uniqueness must be verified for each variant before returning;
                        # otherwise a common prefix could match the wrong location.
                        if _anchor_line_count > 2:
                            for _try_lines in range(_anchor_line_count - 1, 1, -1):
                                _partial = "".join(_lines[_li:_li + _try_lines]).rstrip('\n\r')
                                if _partial and modified.count(_partial) == 1:
                                    _actual_pos = modified.find(_partial)
                                    if _actual_pos != -1:
                                        logger.info(
                                            "edit_file: first-line match → progressive %d/%d lines at line %d",
                                            _try_lines, _anchor_line_count, _li + 1,
                                        )
                                        return _actual_pos, _partial, 0.0
                        # Multi-line reconstruction failed (duplicate/out-of-bounds/progressive)
                        # → continue searching; do NOT fall through to single-line return
                        continue
                    # Use the byte offset of THIS matched line, not modified.find(_raw):
                    # _raw can occur earlier (as a substring or a duplicate line), and
                    # find() would resolve to that wrong location. Mirrors the multi-line
                    # path's _li_byte_pos fix above.
                    _li_byte_pos = sum(len(_lines[i]) for i in range(_li))
                    logger.info("edit_file: first-line match → line %d", _li + 1)
                    return _li_byte_pos, _raw, 0.0

        # (step 4 — fuzzy match disabled: caused false-positive anchor resolution
        #  leading to syntax errors when applied at the wrong location)

        # 5. Build helpful error
        _suggestions = ""
        if _first_line:
            _close = difflib.get_close_matches(
                _first_line, [ln.strip() for ln in modified.splitlines()], n=3, cutoff=0.4,
            )
            if _close:
                _suggestions = f". Did you mean: {_close}"

        raise ValueError(f"anchor text not found: {anchor[:80]!r}{_suggestions}")

    def _tool_edit_file(self, args: dict[str, Any]) -> "ToolResult":
        """Edit a single file using anchor-based text operations.

        No diff/patch syntax needed.  Supports:
          replace       -- find *anchor* text and replace it with *content*
          insert_after  -- insert *content* after *anchor*
          insert_before -- insert *content* before *anchor*

        Operations are applied **sequentially** in order.  If one fails the
        whole call is rolled back and an error is returned.
        """
        import time as _time
        start_time = _time.monotonic()
        args = self._recover_args_from_raw(args, ("path",))
        file_path = (args.get("path") or "").strip()
        ops = args.get("operations") or args.get("ops") or []
        # If operations is empty but __raw_arguments has serialized JSON, try to recover
        if not ops and "__raw_arguments" in args:
            _raw_ops = self._extract_ops_from_raw(args["__raw_arguments"])
            if _raw_ops:
                ops = _raw_ops
        if not file_path:
            # If __raw_arguments is present, the JSON was likely truncated during streaming
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(
                ok=False,
                error=f"path is required{_raw_hint}",
                execution_time=0,
            )
        if not ops:
            return self._make_result(ok=False, error=f"operations list cannot be empty for {file_path}", execution_time=0)

        _norm = Path(self.repo_root) / file_path
        if not _norm.exists():
            return self._make_result(
                ok=False, error=f"File not found: {file_path}", execution_time=0
            )

        try:
            original = _norm.read_text(encoding="utf-8")
        except Exception as e:
            return self._make_result(
                ok=False, error=f"Failed to read {file_path}: {e}", execution_time=0
            )

        # Detect file line ending style so insert operations preserve it
        _file_newline = '\r\n' if '\r\n' in original else '\n'
        modified = original
        _op_type_counts: dict[str, int] = {}
        _edit_warnings: list[str] = []
        for i, op in enumerate(ops):
            op_type = op.get("type", "replace")
            anchor = op.get("anchor", "")
            content = op.get("content", "")
            if not anchor:
                return self._make_result(
                    ok=False, error=f"Operation {i}: 'anchor' is required"
                )

            try:
                _pos, _actual_anchor, _fuzzy_ratio = self._resolve_edit_anchor(
                    modified, anchor, op.get("line"),
                )
            except ValueError as e:
                return self._make_result(
                    ok=False,
                    error=f"Operation {i}: {e}",
                )

            if _fuzzy_ratio > 0.0:
                _warn = f"op {i}: anchor fuzzy-matched at ratio={_fuzzy_ratio:.2f} (anchor={anchor[:60]!r}, matched={_actual_anchor[:60]!r})"
                logger.warning("edit_file %s", _warn)
                _edit_warnings.append(_warn)



            if op_type == "replace":
                _op_type_counts["replace"] = _op_type_counts.get("replace", 0) + 1
                # Defensive: warn if content contains the anchor (likely copy-paste error)
                if anchor in content:
                    _warn = f"op {i}: content contains anchor text — possible content/anchor inversion (anchor_len={len(anchor)}, content_len={len(content)})"
                    logger.warning("edit_file %s", _warn)
                    _edit_warnings.append(_warn)
                # Heuristic: warn if content >> anchor suggests whole-file intent
                if len(content) > len(anchor) * 20 and len(content) > 500:
                    _warn_content_anchor_ratio = (
                        f"op {i}: content ({len(content)} chars) is much larger than "
                        f"anchor ({len(anchor)} chars). Did you mean to replace the whole file? "
                        f"If so, use write_plan's replace_file or create_file(overwrite=true) instead."
                    )
                    logger.warning("edit_file %s", _warn_content_anchor_ratio)
                    _edit_warnings.append(_warn_content_anchor_ratio)
                modified = modified[:_pos] + content + modified[_pos + len(_actual_anchor):]

            elif op_type == "insert_after":
                _op_type_counts["insert_after"] = _op_type_counts.get("insert_after", 0) + 1
                _eol = modified.find('\n', _pos + len(_actual_anchor))

                # ── Block-end auto-correction (mirrors anchor_edit) ────────
                # If the anchor line is a block header (def/class/if/... in
                # Python, a '{'-opening line in brace languages), the EOL
                # computed above points at the END of the header line — so the
                # insert would nest the content INSIDE the body. Find the
                # block's real end (language-agnostic via _find_block_end_line)
                # and move the insertion point past it so the content becomes a
                # sibling. Installing a grammar enables this for that language
                # with no code change.
                if _eol != -1:
                    _ef_anchor_lineno = modified[:_pos].count('\n')
                    _ef_lines = modified.splitlines(True)
                    _ef_block_end = _find_block_end_line(
                        modified,
                        LanguageId.from_path(str(_norm)).value,
                        _ef_anchor_lineno, _ef_lines,
                    )
                    if _ef_block_end is not None and _ef_block_end > _ef_anchor_lineno:
                        _ef_new_eol = (
                            sum(len(_l) for _l in _ef_lines[:_ef_block_end + 1]) - 1
                        )
                        if 0 <= _ef_new_eol < len(modified) and modified[_ef_new_eol] == '\n':
                            _eol = _ef_new_eol
                            logger.info(
                                "edit_file op %d (insert_after): anchor L%d is a "
                                "%d-line block header — inserting after block end "
                                "L%d instead of into the body",
                                i, _ef_anchor_lineno + 1,
                                _ef_block_end - _ef_anchor_lineno + 1,
                                _ef_block_end + 1,
                            )

                if _eol == -1:
                    # Anchor line is the last line of a file with no trailing
                    # newline — terminate it first, otherwise the slice below
                    # glues the inserted content onto the anchor line.
                    modified += _file_newline
                    _eol = len(modified) - 1  # index of the '\n' just added
                # Idempotency check: skip if content already exists after the anchor.
                # Second clause covers the EOF case (no trailing newline) — exact
                # match only, since a permissive startswith would false-positive on
                # any prefix (e.g. content="x = 1" vs after_text="x = 123\n...").
                _after_text = modified[_eol+1:]
                _normalized_content = content.rstrip('\r\n') + _file_newline
                _already_present = (
                    _after_text.startswith(_normalized_content)
                    or _after_text == content.rstrip('\r\n')
                )
                if _already_present:
                    logger.info(
                        "edit_file op %d (insert_after): content already present after anchor — skipping (idempotent)", i,
                    )
                else:
                    modified = modified[:_eol+1] + _normalized_content + modified[_eol+1:]

            elif op_type == "insert_before":
                _op_type_counts["insert_before"] = _op_type_counts.get("insert_before", 0) + 1
                # Idempotency check: skip if content (normalized) already exists before anchor.
                # Checks the text immediately preceding _pos (after the last newline before anchor).
                _before_text = modified[:_pos]
                _candidate_end = _before_text.rfind('\n', 0, _pos)
                if _candidate_end == -1:
                    _candidate = _before_text[:_pos]
                else:
                    _candidate = _before_text[_candidate_end + 1:_pos]
                _candidate_norm = _candidate.rstrip('\r\n')
                _content_norm = content.rstrip('\r\n')
                _already_present = _candidate_norm == _content_norm
                # Also check multi-line: content might span multiple lines before anchor
                if not _already_present:
                    _content_with_newline = _content_norm + _file_newline
                    _already_present = _before_text.endswith(_content_with_newline)
                if _already_present:
                    logger.info(
                        "edit_file op %d (insert_before): content already present before anchor -- skipping (idempotent)", i,
                    )
                else:
                    modified = modified[:_pos] + content.rstrip('\r\n') + _file_newline + modified[_pos:]

            else:
                return self._make_result(
                    ok=False,
                    error=f"Operation {i}: unknown type '{op_type}' (expected replace, insert_after, insert_before)",
                )

        try:
            _norm.write_text(modified, encoding="utf-8")
        except Exception as e:
            return self._make_result(
                ok=False, error=f"Failed to write {file_path}: {e}"
            )

        _orig_lines = original.splitlines()
        _mod_lines = modified.splitlines()
        _delta = len(_mod_lines) - len(_orig_lines)
        # Compute separate add/remove counts via a simple line-diff
        _added_lines = _removed_lines = 0
        try:
            import difflib
            for line in difflib.unified_diff(_orig_lines, _mod_lines, n=0):
                if line.startswith('+++') or line.startswith('---'):
                    continue
                if line.startswith('+'):
                    _added_lines += 1
                elif line.startswith('-'):
                    _removed_lines += 1
        except Exception:
            pass
        # Build op-type breakdown for the summary
        _op_type_breakdown = " + ".join(
            f"{_op_type_counts[k]} {k}"
            for k in ("replace", "insert_after", "insert_before")
            if _op_type_counts.get(k, 0)
        )
        _line_detail = f" (+{_added_lines}, -{_removed_lines})" if _added_lines or _removed_lines else ""
        # Structural validation: warn if replace op type resulted in zero removed lines
        _replace_count = _op_type_counts.get("replace", 0)
        if _replace_count > 0 and _added_lines > 0 and _removed_lines == 0:
            _warn = (
                f"{_replace_count} replace op(s) resulted in +{_added_lines}, -0 — "
                "replace structurally always removes the anchor text. "
                "This suggests the intended operation was insert_after or insert_before, not replace."
            )
            logger.warning("edit_file op-type mismatch: %s", _warn)
            _edit_warnings.append(_warn)
        _meta: dict[str, Any] = {}
        if _edit_warnings:
            _meta["edit_warnings"] = _edit_warnings
        _syn = self._run_syntax_check_for_file(file_path)
        if not _syn.get("skipped"):
            _meta["syntax_check"] = _syn
            if not _syn.get("ok"):
                # Syntax error detected — rollback file to original and return error.
                # This prevents the LLM from operating on a broken file and avoids
                # downstream verify_after_write failures that cause repeated 100K+
                # token edit_file retry loops (observed: asi.py 8088 lines).
                _norm.write_text(original, encoding="utf-8")
                _error_details = "; ".join(
                    f"line {e.get('line')}:{e.get('col')} \u2014 {e.get('message', '').strip()}"
                    for e in (_syn.get("errors") or [])
                )
                _exec = _time.monotonic() - start_time
                _meta["rollback_reason"] = "syntax_error"
                return self._make_result(
                    ok=False,
                    error=(
                        f"Syntax error after editing {file_path}. Rolled back to original.\n"
                        f"Errors: {_error_details}\n"
                    ),
                    metadata=_meta,
                    execution_time=_exec,
                )
        _exec = _time.monotonic() - start_time
        # Track applied patch so agent_loop can detect successful writes
        try:
            self._applied_patches.append(
                f"edit_file:{file_path}:{_op_type_breakdown}:{_added_lines:+}/{-_removed_lines:-}"
            )
        except (AttributeError, TypeError):
            pass
        return self._make_result(
            ok=True,
            content=f"File updated: {file_path} ({_op_type_breakdown} →{_line_detail}) [{_exec:.1f}s]",
            metadata=_meta,
            execution_time=_exec,
        )

    @staticmethod
    def _raw_repr(text: str, max_lines: int = 3) -> str:
        """Show raw character representation of a text snippet.

        Makes invisible differences (trailing whitespace, unusual Unicode,
        tabs vs spaces, CRLF vs LF) immediately visible.
        Returns an empty string if text is empty.
        """
        if not text:
            return ""
        lines = text.splitlines(keepends=True)
        if len(lines) <= max_lines:
            target = text
        else:
            target = "".join(lines[:max_lines]) + f"\n... ({len(lines)} total lines)"
        raw = repr(target)
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1]
        return f"Raw old_string (repr): {raw}\n"

    def _near_match_hint(
        self, content: str, old_string: str, max_window_lines: int = 200
    ) -> str:
        """Best-effort 'did you mean' hint for a failed exact-match edit.

        When old_string does not match verbatim (the dominant cause being
        leading-whitespace / indentation drift in LLM-reconstructed code),
        locate the file region most similar to old_string and return:
          1. a line-numbered, copyable snippet of the real text, and
          2. a unified diff (your old_string → actual file) so the model
             sees exactly which characters differ instead of blindly
             re-guessing or falling back to fragile shell here-docs.

        Returns "" if no plausible candidate is found or on any error —
        the caller appends the result to the failure message, so a blank
        hint degrades gracefully to the original behaviour.
        """
        try:
            import difflib
            file_lines = content.splitlines()
            old_lines = old_string.splitlines()
            if not file_lines or not old_lines:
                return ""
            window = len(old_lines)
            # Anchor on the first non-blank line of old_string to build a
            # cheap candidate set, then score full windows around each
            # candidate by similarity. Skip the window scan for pathologically
            # large old_strings — the anchor block alone is still useful.
            old_first = next((_item_.strip() for _item_ in old_lines if _item_.strip()), "")
            if not old_first:
                return ""
            stripped = [_item_.strip() for _item_ in file_lines]
            close = difflib.get_close_matches(old_first, stripped, n=5, cutoff=0.4)
            if not close:
                return ""
            close_set = set(close)
            cand_idxs = [i for i, _item_ in enumerate(stripped) if _item_ in close_set]

            old_blob = "\n".join(old_lines)
            best_ratio, best_start = 0.0, -1
            if window <= max_window_lines:
                _sm = difflib.SequenceMatcher()
                _sm.set_seq2(old_blob)
                for ci in cand_idxs:
                    for start in range(max(0, ci - window + 1), ci + 1):
                        region = file_lines[start:start + window]
                        if not region:
                            continue
                        _sm.set_seq1("\n".join(region))
                        r = _sm.ratio()
                        if r > best_ratio:
                            best_ratio, best_start = r, start
            if best_start < 0:
                # Window scan skipped or inconclusive — anchor on the single
                # best candidate line so the model at least gets a location.
                best_start = cand_idxs[0]
                best_ratio = difflib.SequenceMatcher(
                    None, old_first, stripped[best_start]
                ).ratio()

            ctx_start = max(0, best_start - 2)
            ctx_end = min(len(file_lines), best_start + window + 2)
            numbered = "\n".join(
                f"{ctx_start + j + 1:5d}| {file_lines[ctx_start + j]}"
                for j in range(ctx_end - ctx_start)
            )
            region = file_lines[best_start:best_start + window]
            diff = "\n".join(difflib.unified_diff(
                old_lines, region,
                fromfile="your old_string", tofile="actual file", lineterm="",
            ))
            # Cap diff size so a wildly-wrong old_string can't bloat the result.
            if len(diff) > 2000:
                diff = diff[:2000] + "\n… (diff truncated)"

            def _decor_norm(s: str) -> str:
                """Strip markdown decoration chars (` * _) for content comparison.

                Equality after normalization means the ONLY difference is
                decoration characters — whitespace and surrounding text are
                identical. Used to distinguish a markdown-decoration drift
                (``x`` vs `x`, **b** vs *b*) from a real content mismatch, so
                the hint can name the cause precisely instead of leaving the
                model to eyeball a 97%-similar diff.
                """
                return s.replace("`", "").replace("*", "").replace("_", "")

            ws_note = ""
            decor_note = ""
            # High similarity + exact-match failure ⇒ the difference is subtle.
            # Classify the first differing line into one of two causes so the
            # hint names it unambiguously:
            #   - whitespace/indent drift (ol.strip() == rl.strip())
            #   - markdown decoration drift (decor-normalized equal, content differs)
            # Whitespace is checked first (it is the dominant failure mode); a
            # decoration note never fires when whitespace is the cause, and vice
            # versa — the two causes are mutually exclusive by construction
            # (decoration chars are non-whitespace, so a whitespace-only diff
            # normalizes away under .strip() while a decoration diff does not).
            if best_ratio >= 0.88:
                for ol, rl in zip(old_lines, region, strict=False):
                    if ol == rl:
                        continue
                    if ol.strip() == rl.strip():
                        def _vis(s):
                            lead = s[:len(s) - len(s.lstrip())]
                            return lead.replace(" ", "·").replace("\t", "⇥") + s.lstrip()
                        ws_note = (
                            "\nWhitespace differs (· = space, ⇥ = tab):\n"
                            f"  yours: {_vis(ol)!r}\n"
                            f"  file : {_vis(rl)!r}"
                        )
                        break
                    if _decor_norm(ol) == _decor_norm(rl):
                        decor_note = (
                            "\nMarkdown decoration differs (backticks `` ` ``/asterisks "
                            "`*`/underscores `_`). The surrounding text is identical — "
                            "only the decoration characters differ. Copy the EXACT "
                            "decoration from the file (shown in the diff/numbered block)."
                        )
                        break
            return (
                f"\nClosest match (~{best_ratio:.0%} similar) near line {best_start + 1}:\n"
                f"```\n{numbered}\n```\n"
                f"Diff (your old_string vs file):\n```diff\n{diff}\n```"
                f"{ws_note}{decor_note}\n"
                "Copy the EXACT text — including indentation — from the numbered "
                "block above into old_string."
            )
        except Exception:
            return ""

    def _ast_fail_hint(
        self, source: str, ops: list[dict[str, Any]], symbol: str
    ) -> str:
        """'Did you mean' hint for a failed edit_ast call.

        edit_ast fails in two distinct ways the model can't self-diagnose
        from the bare ``no match found`` string:

          1. Symbol resolution — ops like add_guard target a function/method
             by name; if ``symbol`` doesn't resolve, suggest the closest
             defined names (the add_guard failure seen in the wild).
          2. Text search — replace_expr.old / delete_stmt.pattern look for
             existing text; reuse _near_match_hint to surface the real span.

        Returns "" on no candidate or any error — appended to the failure
        message so a blank hint preserves the original behaviour.
        """
        try:
            import ast as _ast
            import difflib
            parts: list[str] = []

            # 1. Symbol resolution suggestions.
            sym = (symbol or "").strip()
            if sym:
                try:
                    tree = _ast.parse(source)
                except SyntaxError:
                    tree = None
                if tree is not None:
                    defined: set[str] = set()
                    qualified: set[str] = set()

                    class _V(_ast.NodeVisitor):
                        def __init__(self):
                            self.stack: list[str] = []

                        def _record(self, name: str):
                            defined.add(name)
                            if self.stack:
                                qualified.add(".".join([*self.stack, name]))

                        def visit_ClassDef(self, node):
                            self._record(node.name)
                            self.stack.append(node.name)
                            self.generic_visit(node)
                            self.stack.pop()

                        def visit_FunctionDef(self, node):
                            self._record(node.name)
                            self.stack.append(node.name)
                            self.generic_visit(node)
                            self.stack.pop()

                        visit_AsyncFunctionDef = visit_FunctionDef

                    _V().visit(tree)
                    bare = sym.split(".")[-1]
                    if bare not in defined and sym not in qualified:
                        pool = sorted(qualified | defined)
                        close = difflib.get_close_matches(bare, pool, n=5, cutoff=0.4)
                        if close:
                            parts.append(
                                f"symbol {sym!r} not found in this file. "
                                f"Did you mean: {close}?"
                            )
                        else:
                            _avail = sorted(defined)[:20]
                            if _avail:
                                parts.append(
                                    f"symbol {sym!r} not found. Defined here: {_avail}"
                                )

            # 2. Text-search op suggestions — only when the searched text is
            #    genuinely absent verbatim (a reliable proxy for "this op is
            #    the one that failed on a mismatch", avoiding misleading hints
            #    for ops that failed for other reasons).
            for op in ops:
                if not isinstance(op, dict):
                    continue
                t = (op.get("type") or "").strip()
                search = ""
                if t == "replace_expr":
                    search = op.get("old") or ""
                elif t == "delete_stmt":
                    search = op.get("pattern") or ""
                if search and search not in source:
                    h = self._near_match_hint(source, search)
                    if h:
                        parts.append(f"[{t}] {h}")

            return ("\n" + "\n".join(parts)) if parts else ""
        except Exception:
            return ""

    def _resolve_with_fallback(
        self, content: str, old_string: str
    ) -> "tuple[str, int, Optional[list], Optional[list]]":
        """Resolve old_string via exact → whitespace-tolerant → unicode-tolerant matching.

        Returns (resolved_old_string, count, fallback_matches, orig_split).

        - resolved_old_string: exact text from file to use in content.replace()/content.index()
        - count: number of occurrences found after all fallback attempts
        - fallback_matches: None if exact match succeeded; [(line_idx, text), ...] for fallback matches
        - orig_split: content.splitlines(keepends=True) from fallback path (None for exact match)
        """
        count = content.count(old_string)
        if count > 0:
            return old_string, count, None, None

        # ── Fallback matching setup ───────────────────────────────────
        _orig_split = content.splitlines(keepends=True)
        _norm_content_lines = [_item_.rstrip() for _item_ in _orig_split]
        _norm_old_lines = [_item_.rstrip() for _item_ in old_string.splitlines()]

        # ── Unicode decoration map ──
        _UNI_DECORATIVE = str.maketrans({
            '─': '-', '━': '-',  # box-drawing horizontal
            '—': '-', '–': '-',  # em-dash, en-dash
            '│': '|', '┃': '|',  # box-drawing vertical
            '┌': '|', '┐': '|',  # box-drawing corners
            '└': '|', '┘': '|',  # box-drawing corners
        })

        # ── Whitespace-tolerant fallback ──────────────────────────────
        if _norm_old_lines:
            _ws_matches: list[tuple[int, str]] = []
            for _s_idx in range(
                len(_norm_content_lines) - len(_norm_old_lines) + 1
            ):
                if _norm_content_lines[_s_idx:_s_idx + len(_norm_old_lines)] == _norm_old_lines:
                    _recon = "".join(_orig_split[_s_idx:_s_idx + len(_norm_old_lines)])
                    # Honor caller's trailing-newline intent
                    if not old_string.endswith(("\n", "\r")):
                        _recon = _recon.rstrip("\r\n")
                    _ws_matches.append((_s_idx, _recon))
            if _ws_matches:
                count = len(_ws_matches)
                resolved = _ws_matches[0][1] if count == 1 else old_string
                return resolved, count, _ws_matches, _orig_split

        # ── Indent-tolerant fallback ──────────────────────────────────
        # Normalize both leading AND trailing whitespace — catches
        # indentation differences (reindent, tab↔space, wrong indent level).
        # An empty/whitespace old_string is rejected by _apply_one_edit_text
        # before reaching here; without that guard, an all-empty
        # _indent_norm_old would spuriously match every line position.
        _indent_norm_content = [_item_.strip() for _item_ in _orig_split]
        _indent_norm_old = [_item_.strip() for _item_ in old_string.splitlines()]
        _indent_matches: list[tuple[int, str]] = []
        for _s_idx in range(
            len(_indent_norm_content) - len(_indent_norm_old) + 1
        ):
            if _indent_norm_content[_s_idx:_s_idx + len(_indent_norm_old)] == _indent_norm_old:
                _recon = "".join(_orig_split[_s_idx:_s_idx + len(_indent_norm_old)])
                if not old_string.endswith(("\n", "\r")):
                    _recon = _recon.rstrip("\r\n")
                _indent_matches.append((_s_idx, _recon))
        if _indent_matches:
            count = len(_indent_matches)
            resolved = _indent_matches[0][1] if count == 1 else old_string
            return resolved, count, _indent_matches, _orig_split

        # ── Unicode-tolerant fallback ─────────────────────────────────
        import unicodedata
        def _unorm(s):
            s = unicodedata.normalize("NFC", s)
            s = s.translate(_UNI_DECORATIVE)
            return s
        _uni_norm_lines = [_unorm(_item_.rstrip()) for _item_ in _orig_split]
        _uni_old_lines = [_unorm(_item_.rstrip()) for _item_ in old_string.splitlines()]
        if _uni_old_lines:
            _uni_matches: list[tuple[int, str]] = []
            for _s_idx in range(
                len(_uni_norm_lines) - len(_uni_old_lines) + 1
            ):
                if _uni_norm_lines[_s_idx:_s_idx + len(_uni_old_lines)] == _uni_old_lines:
                    _recon = "".join(_orig_split[_s_idx:_s_idx + len(_uni_old_lines)])
                    if not old_string.endswith(("\n", "\r")):
                        _recon = _recon.rstrip("\r\n")
                    _uni_matches.append((_s_idx, _recon))
            if _uni_matches:
                count = len(_uni_matches)
                resolved = _uni_matches[0][1] if count == 1 else old_string
                return resolved, count, _uni_matches, _orig_split

        # ── No match found ──
        return old_string, 0, None, _orig_split

    @staticmethod
    def _edited_line_regions(
        original: str,
        modified: str,
        lineno_1based: int,
        context: int = 1,
    ):
        """Determine whether ``lineno_1based`` falls inside an edited region.

        Used by the edit_text syntax gate to give a scope-aware diagnosis:
        Python reports an INDENTATION error on the line where the parser
        *notices* the inconsistency, which is often several lines below the
        line whose indentation was actually wrong. Without knowing whether
        the reported line was touched by the edit, the LLM cannot tell its
        own indentation mistake from a cascade caused elsewhere.

        Compares ``original`` (pre-edit content) against ``modified``
        (post-edit content) with :class:`difflib.SequenceMatcher` and returns
        a tuple ``(in_edited_region, changed_regions)`` where
        ``changed_regions`` is a list of ``(start_1based, end_1based)``
        inclusive line ranges (in ``modified``) that differ from ``original``.
        A line counts as "edited" if it is within ``context`` lines of any
        differing block (default 1) — this matches how indentation errors
        typically surface one line off.

        Robustness: any difflib/line-split failure degrades gracefully to
        ``(True, [])`` so the gate still refuses (safe default) without
        crashing the tool call.
        """
        try:
            import difflib

            _orig_lines = original.splitlines(keepends=True)
            _mod_lines = modified.splitlines(keepends=True)
            _sm = difflib.SequenceMatcher(a=_orig_lines, b=_mod_lines, autojunk=False)
            regions = []
            for _tag, _i1, _i2, _j1, _j2 in _sm.get_opcodes():
                if _tag == "equal":
                    continue
                if _j1 >= _j2:
                    # pure deletion — no lines in modified. Anchor the region at
                    # the position the deletion happened so a cascade landing
                    # there is still flagged.
                    _start = max(1, _j1)
                    _end = _start
                else:
                    _start = max(1, _j1 + 1)
                    _end = min(len(_mod_lines), _j2)
                regions.append((_start, _end))
            if not regions:
                return True, []  # couldn't find a change but content differs — be safe
            in_region = any(
                max(1, s - context) <= lineno_1based <= e + context
                for (s, e) in regions
            )
            return in_region, regions
        except Exception:
            return True, []  # safe default: assume in-region so gate still refuses

    @staticmethod
    def _indentation_hint(content: str, lineno_1based: int, msg: str) -> str:
        """Build a concrete indentation suggestion for a SyntaxError.

        Python's indentation messages are generic ("unexpected indent",
        "unindent does not match", "expected an indented block"). The LLM
        often retries 2-3 times guessing the right column because the message
        never states *how many* spaces are needed. This helper inspects the
        surrounding lines of ``content`` (the post-edit source) and produces a
        specific hint, e.g.::

            Indentation: line 122 has 8 leading spaces; nearby statements use 4.
            Reduce this line to 4 spaces.

        Returns ``""`` when no confident suggestion can be derived (the caller
        then omits the hint). All arithmetic is defensive against tabs,
        blank lines, and boundary indices.
        """
        try:
            lines = content.split("\n")
            if not (1 <= lineno_1based <= len(lines)):
                return ""
            err_idx = lineno_1based - 1
            err_raw = lines[err_idx]
            err_lead = len(err_raw) - len(err_raw.lstrip(" "))

            def _lead_of(idx):
                if idx < 0 or idx >= len(lines):
                    return None
                _l = lines[idx]
                stripped = _l.lstrip(" ")
                if not stripped or stripped.startswith("#"):
                    return None
                return len(_l) - len(stripped)

            prev_leads = []
            for j in range(err_idx - 1, -1, -1):
                lv = _lead_of(j)
                if lv is not None:
                    prev_leads.append((j + 1, lv))
                    if len(prev_leads) >= 4:
                        break
            next_leads = []
            for j in range(err_idx + 1, len(lines)):
                lv = _lead_of(j)
                if lv is not None:
                    next_leads.append((j + 1, lv))
                    if len(next_leads) >= 3:
                        break

            m_lower = (msg or "").lower()
            from collections import Counter

            if "unexpected indent" in m_lower:
                candidates = [lv for (_ln, lv) in (prev_leads + next_leads) if lv < err_lead]
                if not candidates:
                    return ""
                target = Counter(candidates).most_common(1)[0][0]
                return (
                    f"Indentation: line {lineno_1based} has {err_lead} leading "
                    f"spaces; nearby statements use {target}. Reduce this line "
                    f"to {target} spaces."
                )

            if "unindent does not match" in m_lower or "unexpected unindent" in m_lower:
                outer = sorted({lv for (_ln, lv) in prev_leads if lv < err_lead})
                if not outer:
                    return ""
                levels = ", ".join(f"{x} spaces" for x in outer)
                return (
                    f"Indentation: line {lineno_1based} dedents to {err_lead} "
                    f"spaces, but no enclosing block uses that width. Valid "
                    f"outer indentation level(s): {levels}."
                )

            if "expected an indented block" in m_lower:
                opener_idx = None
                for j in range(err_idx - 1, -1, -1):
                    _l = lines[j].rstrip()
                    if _l and not _l.startswith("#") and _l.endswith(":"):
                        opener_idx = j
                        break
                if opener_idx is None:
                    return ""
                opener_lead = _lead_of(opener_idx) or 0
                target = opener_lead + 4
                return (
                    f"Indentation: line {opener_idx + 1} ends with ':' and opens "
                    f"a block, so line {lineno_1based} (and the rest of its body) "
                    f"must be indented deeper — use {target} spaces (opener is "
                    f"at {opener_lead})."
                )

            return ""
        except Exception:
            return ""

    @staticmethod
    def _structural_imbalance_hint(msg: str) -> str:
        """Turn a structural SyntaxError message into an actionable hint.

        Python reports structural imbalances ("expected 'except' or 'finally'",
        "'(' was never closed", ...) on the line where the parser GIVES UP, not
        the line that opened the unbalanced construct. The generic diagnosis
        above (indentation/cascade) rarely names WHICH structural token got
        dropped, so the LLM has to re-read the file to find it. This helper maps
        the exact error substring ast.parse already raised to a concise hint
        naming the missing element. No heuristic guessing is performed, so false
        positives are impossible — the hint only fires when Python itself
        pinpointed the missing token.

        Returns ``""`` when ``msg`` does not match a known structural pattern;
        the caller then omits the hint and falls back to indentation/cascade.
        """
        m_lower = (msg or "").lower()
        if (
            "expected 'except' or 'finally'" in m_lower
            or "has no 'except' or 'finally'" in m_lower
        ):
            return (
                "Structural imbalance: new_string opens a `try:` block but is "
                "missing its matching `except`/`finally` clause — a truncated "
                "replacement block is the usual cause. Include the complete "
                "try/except/finally structure in new_string."
            )
        if "'(' was never closed" in m_lower:
            return (
                "Structural imbalance: an opening `(` is never closed — "
                "new_string likely dropped the closing parenthesis. Balance "
                "all parentheses in new_string."
            )
        if "'[' was never closed" in m_lower:
            return (
                "Structural imbalance: an opening `[` is never closed — "
                "new_string likely dropped the closing bracket."
            )
        if "'{' was never closed" in m_lower:
            return (
                "Structural imbalance: an opening `{` is never closed — "
                "new_string likely dropped the closing brace."
            )
        if "unexpected eof while parsing" in m_lower:
            return (
                "Structural imbalance: the source ends while a bracket or block "
                "is still open — new_string likely truncated the closing token(s)."
            )
        return ""

    def _apply_scoped_replacement(self, content, file_path, old_string, new_string, scope):
        """Scope-restricted old_string->new_string replacement.

        Uniqueness is measured WITHIN the (start, end) 1-based inclusive line
        range only. Occurrences OUTSIDE the range are ignored. The replacement
        is a POSITION-BASED splice (not content.replace(...,1)) so the correct
        in-scope occurrence is replaced even when an identical block exists
        earlier out-of-scope — and critically, this holds for BOTH the exact
        and the fallback (whitespace/indent/unicode-tolerant) matching paths.

        Bug history: the fallback path previously stored ``_char_pos = None``
        and fell back to ``content.replace(resolved, ..., 1)``. But when the
        fallback matched 2+ sites, ``resolved == old_string`` (the caller's
        non-matching original), so that replace was a NO-OP yet returned
        ``{"ok": True}`` — a silent success with an unmodified file. The fix
        stores the precise (char_pos, recon) from the fallback's line index so
        the splice is always position-based, mirroring the replace_all path.
        """
        scope_start, scope_end = scope
        if not old_string.strip():
            return {
                "ok": False,
                "error": "old_string is empty or whitespace only; cannot perform a meaningful replacement.",
                "metadata": {},
            }

        _resolved, total_count, fallback_matches, _orig_split = self._resolve_with_fallback(content, old_string)

        in_scope_matches = []
        if fallback_matches is not None:
            # Build per-line char offsets so we can splice the EXACT in-scope
            # site. _orig_split uses keepends=True, so offset[i] is the char
            # position of line i (0-based). Mirrors the replace_all path.
            _offset_by_line = [0]
            for _l in _orig_split:
                _offset_by_line.append(_offset_by_line[-1] + len(_l))
            for (_m_idx, _m_recon) in fallback_matches:
                _start_line = _m_idx + 1
                _n = len(_m_recon.splitlines()) or 1
                if scope_start <= _start_line <= scope_end:
                    _pos = _offset_by_line[_m_idx]
                    in_scope_matches.append((_start_line, _start_line + _n - 1, _pos, _m_recon))
        else:
            _search = 0
            _sl = len(old_string)
            while True:
                _pos = content.find(old_string, _search)
                if _pos < 0:
                    break
                _start_line = content[:_pos].count("\n") + 1
                if scope_start <= _start_line <= scope_end:
                    in_scope_matches.append((_start_line, _start_line + old_string.count("\n"), _pos, old_string))
                _search = _pos + _sl

        in_scope_count = len(in_scope_matches)

        if in_scope_count == 0:
            if total_count == 0:
                _hint = self._near_match_hint(content, old_string)
                _raw = self._raw_repr(old_string)
                return {
                    "ok": False,
                    "error": (f"old_string not found in {file_path}\n{_raw}{_hint}\nTo fix this, re-read the file and include 2-3 lines of surrounding context as old_string."),
                    "metadata": {"matched": False, "near_match": bool(_hint), "failure_class": "search_string_mismatch"},
                }
            return {
                "ok": False,
                "error": (f"old_string not found WITHIN scope L{scope_start}-{scope_end} in {file_path}, but {total_count} occurrence(s) exist OUTSIDE the scope. Adjust scope_start_line/scope_end_line to cover the intended occurrence."),
                "metadata": {"matched": False, "in_scope_count": 0, "out_of_scope_count": total_count, "scope": list(scope), "failure_class": "search_string_mismatch"},
            }

        if in_scope_count > 1:
            return {
                "ok": False,
                "error": (f"Found {in_scope_count} occurrences of old_string WITHIN scope L{scope_start}-{scope_end} in {file_path}. Narrow the scope or make old_string more unique (include 2-3 lines of context)."),
                "metadata": {"matched": False, "in_scope_count": in_scope_count, "scope": list(scope), "failure_class": "search_string_mismatch"},
            }

        _ms, _me, _char_pos, _recon = in_scope_matches[0]
        # Position-based splice using the char offset we recorded. For the
        # fallback path, reindent new_string to the match's actual indentation
        # (same as replace_all); for the exact path _recon == old_string so
        # _reindent_to_match is a no-op.
        _reindented_new = _reindent_to_match(new_string, _recon, file_unit=_detect_file_unit(content))
        new_content = content[:_char_pos] + _reindented_new + content[_char_pos + len(_recon):]
        return {
            "ok": True,
            "new_content": new_content,
            "occurrences": 1,
            "high_count_warning": "",
            "match_line": _ms,
            "match_indent": _leading_indent_width(_recon),
            "reindent_applied": (_reindented_new != new_string),
        }


    def _apply_one_edit_text(
        self,
        content: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
        scope=None,
    ) -> dict[str, Any]:
        """Apply ONE old_string→new_string replacement to ``content``.

        Pure transformation — no disk I/O, no encoding, no syntax gate.
        Used by both the single-edit path and the batch (``edits``) path so
        both share identical matching/reindent/disambiguation behaviour.

        Returns a dict:
          * failure: ``{"ok": False, "error": str, "metadata": dict}``
          * success: ``{"ok": True, "new_content": str, "occurrences": int,
                        "high_count_warning": str}``

        ``scope`` — when not None, a ``(start, end)`` 1-based inclusive line
        range. Matching & uniqueness are restricted to that range: occurrences
        OUTSIDE it are ignored. This is a position-based splice (not
        ``content.replace(..., 1)``) so the correct in-scope occurrence is
        replaced even when an earlier identical block exists out-of-scope.
        Mutually exclusive with ``replace_all`` (validated by the caller).
        """
        _MAX_REPLACE_ALL_MATCHES = 20

        # ── Scoped replacement: restrict matching to a line range ──
        # Early-return branch: when scope is set we delegate to a dedicated
        # helper. The scope=None path below is COMPLETELY UNTOUCHED, so no
        # existing caller can regress.
        if scope is not None:
            return self._apply_scoped_replacement(
                content, file_path, old_string, new_string, scope
            )


        # No minimum-length gate: uniqueness is measured by occurrence count
        # below (count==0 → not found; count>1 → not unique; count==1 → safe).
        # Length is a poor proxy for uniqueness ("if __name__" is 13 chars but
        # appears dozens of times; a rare constant can be 4 chars but unique).
        # The ONLY hard rejection here is an empty/whitespace old_string:
        # content.count("") returns len(content)+1, so it would pass the exact
        # match path and content.replace("", x, 1) would prepend x nonsensically.
        # NOTE: the single-edit path also guards this in _tool_edit_text, but
        # the batch path does NOT pre-check each edit's old_string, so this is
        # the authoritative guard for batch edits.
        if not old_string.strip():
            return {
                "ok": False,
                "error": (
                    "old_string is empty or whitespace only; cannot perform a "
                    "meaningful replacement."
                ),
                "metadata": {},
            }

        # ── Resolve old_string via fallback matching ──
        _orig_old_string = old_string
        _orig_new_string = new_string
        old_string, count, fallback_matches, _orig_split = self._resolve_with_fallback(content, old_string)

        # ── Re-indent new_string when fallback resolved with diff indent ──
        _reindent_applied = False
        if count > 0 and old_string != _orig_old_string:
            new_string = _reindent_to_match(new_string, old_string, file_unit=_detect_file_unit(content))
            _reindent_applied = (new_string != _orig_new_string)

        _high_count_warning = ""
        if not replace_all:
            if count == 0:
                _hint = self._near_match_hint(content, old_string)
                _raw = self._raw_repr(old_string)
                if not _hint and _raw:
                    _extra = (
                        "\nNo close match found in the file. The old_string you sent "
                        "may be completely different from the actual file content, or "
                        "the file has changed since you last read it.\n"
                    )
                else:
                    _extra = ""
                return {
                    "ok": False,
                    "error": (
                        f"old_string not found in {file_path}\n{_raw}{_hint}{_extra}\n"
                        "To fix this, re-read the file and include 2-3 lines of "
                        "surrounding context as old_string (not just the line you "
                        "want to change)."
                    ),
                    "metadata": {"matched": False, "near_match": bool(_hint), "failure_class": "search_string_mismatch"},
                }
            if count > 1:
                # Build disambiguation context for each occurrence
                if fallback_matches is not None:
                    _contexts = []
                    _n_lines = len(old_string.splitlines())
                    for _mi, (_m_idx, _m_recon) in enumerate(fallback_matches):
                        _start_line = max(0, _m_idx - 2)
                        _end_line = min(len(_orig_split), _m_idx + _n_lines + 2)
                        _snippet = "".join(_orig_split[_start_line:_end_line]).rstrip()
                        _contexts.append(
                            f"  [match {_mi + 1}] around line {_m_idx + 1}:\n{_snippet}"
                        )
                else:
                    lines = content.splitlines(keepends=True)
                    _contexts = []
                    _search_start = 0
                    for _occ_idx in range(count):
                        _pos = content.index(old_string, _search_start)
                        _line_no = content[:_pos].count("\n")
                        _start_line = max(0, _line_no - 2)
                        _end_line = min(len(lines), _line_no + 3)
                        _snippet = "".join(lines[_start_line:_end_line]).rstrip()
                        _contexts.append(f"  [match {_occ_idx + 1}] around line {_line_no + 1}:\n{_snippet}")
                        _search_start = _pos + len(old_string)
                return {
                    "ok": False,
                    "error": (
                        f"Found {count} occurrences of old_string in {file_path}. "
                        f"Make old_string more unique (include 2-3 lines of surrounding context).\n"
                        + "\n---\n".join(_contexts)
                    ),
                    "metadata": {"occurrences": count, "matched": False, "failure_class": "search_string_mismatch"},
                }
            if fallback_matches is not None:
                # Position-based splice: the fallback matched a specific LINE, but
                # content.replace(resolved, ..., 1) replaces the first SUBSTRING
                # occurrence, which may sit inside an earlier longer line that the
                # line-based fallback did NOT match — a silent wrong-site edit that
                # still parses as valid Python (so the syntax gate can't catch it).
                # Mirror _apply_scoped_replacement: splice at the matched line's char
                # offset. (_reindent_to_match is idempotent, so re-applying it here to
                # the already-reindented new_string from line 3430 is a no-op.)
                _offset_by_line = [0]
                for _l in _orig_split:
                    _offset_by_line.append(_offset_by_line[-1] + len(_l))
                _m_idx, _m_recon = fallback_matches[0]
                _first_pos = _offset_by_line[_m_idx]
                _reindented_new = _reindent_to_match(new_string, _m_recon, file_unit=_detect_file_unit(content))
                new_content = content[:_first_pos] + _reindented_new + content[_first_pos + len(_m_recon):]
                _match_line = _m_idx + 1
                _match_indent = _leading_indent_width(_m_recon)
            else:
                new_content = content.replace(old_string, new_string, 1)
                #── Capture edit location for Enot/inside metadata ──
                _first_pos = content.find(old_string)
                _match_line = content[:_first_pos].count("\n") + 1 if _first_pos >= 0 else 0
                _match_indent = _leading_indent_width(old_string)
            _occurrences_replaced = 1
        else:
            if count > _MAX_REPLACE_ALL_MATCHES:
                _high_count_warning = (
                    f" ⚠️ old_string matched {count} times "
                    f"(max recommended: {_MAX_REPLACE_ALL_MATCHES}). "
                    f"Use a more specific old_string if this was unintentional."
                )
                logger.warning(
                    "%s [REPLACE_ALL_HIGH_COUNT] in %s",
                    _high_count_warning, file_path,
                )
            if count == 0:
                _hint = self._near_match_hint(content, old_string)
                _raw = self._raw_repr(old_string)
                return {
                    "ok": False,
                    "error": (
                        f"old_string not found in {file_path}\n{_raw}{_hint}\n"
                        "To fix this, re-read the file and include 2-3 lines of "
                        "surrounding context as old_string (not just the line you "
                        "want to change)."
                    ),
                    "metadata": {"matched": False, "near_match": bool(_hint), "failure_class": "search_string_mismatch"},
                }
            # ── replace_all: when the fallback matched (count == 1 OR > 1),
            #    old_string may not actually exist in the file verbatim — for
            #    count == 1 it was reassigned to the reconstructed line text, which
            #    can appear as a substring of an earlier longer line that the
            #    line-based fallback did NOT match, so content.replace() would edit
            #    the wrong site (same wrong-site bug as the single-edit path). Use
            #    fallback_matches line indices + _orig_split offsets for a precise
            #    position-based splice in ALL fallback cases. ──
            if fallback_matches is not None:
                _repl = content
                _offset_by_line = [0]
                for _l in _orig_split:
                    _offset_by_line.append(_offset_by_line[-1] + len(_l))
                for _m_idx, _m_recon in reversed(fallback_matches):
                    _pos = _offset_by_line[_m_idx]
                    # Reindent new_string to match this particular match's indent
                    _reindented_new = _reindent_to_match(new_string, _m_recon, file_unit=_detect_file_unit(content))
                    _repl = _repl[:_pos] + _reindented_new + _repl[_pos + len(_m_recon):]
                new_content = _repl
                # First match location for metadata
                _match_line = (fallback_matches[0][0] + 1) if fallback_matches else 0
                _match_indent = _leading_indent_width(fallback_matches[0][1]) if fallback_matches else 0
            else:
                new_content = content.replace(old_string, new_string)
                _first_pos = content.find(old_string)
                _match_line = content[:_first_pos].count("\n") + 1 if _first_pos >= 0 else 0
                _match_indent = _leading_indent_width(old_string)
            _occurrences_replaced = count

        return {
            "ok": True,
            "new_content": new_content,
            "occurrences": _occurrences_replaced,
            "high_count_warning": _high_count_warning,
            "match_line": _match_line,
            "match_indent": _match_indent,
            "reindent_applied": _reindent_applied,
        }

    def _tool_edit_text(self, args: dict[str, Any]) -> "ToolResult":
        """Edit a file by replacing exact strings — mirrors Claude Code's Edit tool.

        Pure string replacement with two safety nets, but NO disk rollback:

        * **Blocking syntax gate** — for Python, ``compile()`` runs in-memory on
          the post-edit content *before* any byte touches disk; if the original
          file parsed and the edit would break parsing, the write is refused
          (file left untouched). This catches the classic indent-mismatch case.
        * **Non-blocking semantic diagnostics** — after a successful write,
          pyright/tsc/go diagnostics (type/undefined-name/import issues) are
          collected and surfaced as ``metadata.syntax_check`` for LLM
          self-healing, mirroring apply_patch/edit_file/modify_symbol/anchor_edit.

        Because there is no rollback (unlike apply_patch/edit_file), the syntax
        gate is the only write-time safety net — semantic findings are advisory.

        Two modes (mutually exclusive):

        1. **Single** (default): replace one ``old_string``→``new_string``,
           with optional ``replace_all``.

        2. **Batch (MultiEdit)**: pass ``edits`` — a list of objects
           ``[{"old_string": ..., "new_string": ..., "replace_all"?: false}, ...]``.
           Edits apply in order; each later edit sees the result of earlier ones.
           The whole batch is **atomic**: if any edit fails to match, the file is
           left untouched and the failing edit's index + error is returned with
           ``partial_failure`` metadata. The file is written exactly once.
           Use batch mode to make several unrelated substitutions in a single
           tool call instead of N round-trips.
        """
        import time as _time
        start_time = _time.monotonic()
        # --- Fix ①: Recover args from truncated streaming JSON ---
        args = self._recover_args_from_raw(args, ("file_path", "old_string", "new_string", "edits"))
        file_path = (args.get("file_path") or "").strip()

        if not file_path:
            return self._make_result(ok=False, error="file_path is required", execution_time=0)

        # ── Determine mode: batch (edits) vs single ──
        raw_edits = args.get("edits")
        is_batch = isinstance(raw_edits, list) and len(raw_edits) > 0
        # Reject ambiguous mixed-mode calls up front.
        if is_batch and (args.get("old_string") or args.get("new_string")):
            return self._make_result(
                ok=False,
                error=(
                    "Cannot mix 'edits' (batch mode) with 'old_string'/'new_string' "
                    "(single mode). Use one mode or the other."
                ),
                execution_time=0,
            )

        # ── scope_start_line / scope_end_line: range-restricted matching ──
        # When provided, uniqueness is measured WITHIN the [start, end] line
        # range only — occurrences outside it are ignored. This lets edit_text
        # target one of several identical blocks. Validation rules:
        #   * both must be provided together (one without the other is rejected)
        #   * start <= end
        #   * mutually exclusive with replace_all
        # The scope applies per-edit in batch mode (each edit may carry its own).
        def _parse_scope(d, allow_replace_all_field=True):
            """Extract & validate a (start, end) 1-based scope tuple or None.

            Returns (scope_tuple_or_None, error_str_or_None). On error the
            caller must abort with the message.
            """
            _ssl = d.get("scope_start_line")
            _sel = d.get("scope_end_line")
            _has_start = _ssl is not None
            _has_end = _sel is not None
            if _has_start != _has_end:
                return None, (
                    "scope_start_line and scope_end_line must be provided "
                    "together (got only one of them)."
                )
            if not _has_start:
                return None, None
            try:
                _s = int(_ssl)
                _e = int(_sel)
            except (TypeError, ValueError):
                return None, (
                    f"scope_start_line/scope_end_line must be integers "
                    f"(got start={_ssl!r}, end={_sel!r})."
                )
            if _s > _e:
                return None, (
                    f"scope_start_line ({_s}) must be <= scope_end_line ({_e})."
                )
            if _s < 1:
                return None, "scope_start_line must be >= 1."
            if allow_replace_all_field and d.get("replace_all"):
                return None, (
                    "scope_start_line/scope_end_line cannot be combined with "
                    "replace_all (scope targets a single occurrence; replace_all "
                    "targets all)."
                )
            return (_s, _e), None

        if is_batch:
            edits = []
            for i, e in enumerate(raw_edits):
                if not isinstance(e, dict):
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}] must be an object with old_string/new_string",
                        execution_time=0,
                    )
                _old = e.get("old_string")
                _new = e.get("new_string")
                if _old is None:
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}] is missing old_string",
                        execution_time=0,
                    )
                if _new is None:
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}] is missing new_string",
                        execution_time=0,
                    )
                _e_scope, _e_scope_err = _parse_scope(e)
                if _e_scope_err is not None:
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}]: {_e_scope_err}",
                        execution_time=0,
                    )
                edits.append({
                    "old_string": _old,
                    "new_string": _new,
                    "replace_all": bool(e.get("replace_all", False)),
                    "scope": _e_scope,
                })
        else:
            old_string = args.get("old_string") or ""
            new_string = args.get("new_string") or ""
            replace_all = args.get("replace_all", False)
            if not old_string:
                return self._make_result(ok=False, error="old_string is required", execution_time=0)
            _single_scope, _single_scope_err = _parse_scope(args)
            if _single_scope_err is not None:
                return self._make_result(ok=False, error=_single_scope_err, execution_time=0)
            edits = [{"old_string": old_string, "new_string": new_string, "replace_all": replace_all, "scope": _single_scope}]

        _norm = Path(file_path)
        if not _norm.is_absolute():
            _norm = Path(self.repo_root) / file_path

        if not _norm.exists():
            return self._make_result(ok=False, error=f"File not found: {_norm}", execution_time=0)

        # Strict UTF-8 first, then latin-1. latin-1 decodes ANY byte sequence
        # losslessly (1:1 byte↔char), so untouched regions round-trip exactly
        # when written back with the SAME encoding. The previous
        # errors="replace" fallback baked U+FFFD over every undecodable byte
        # and then rewrote the whole file as UTF-8 — silently corrupting
        # regions far from the edit.
        content = None
        _read_encoding = "utf-8"
        for _enc in ("utf-8", "latin-1"):
            try:
                content = _norm.read_text(encoding=_enc)
                _read_encoding = _enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if content is None:
            return self._make_result(
                ok=False, error=f"Failed to read {file_path}: unsupported encoding", execution_time=0
            )

        # ── Apply all edits in-memory. Atomic: a failing edit aborts ALL ──
        _cur_content = content
        _total_occurrences = 0
        _high_count_warnings = []
        #Capture edit-site location from the FIRST edit (for single-mode Enot/inside
        # metadata: matched_line / matched_indent / reindent_applied). In batch
        # mode only the first edit's location is surfaced — batch callers get
        # per-edit detail via the diff, not metadata.
        _first_match_line = 0
        _first_match_indent = 0
        _first_reindent_applied = False
        for i, e in enumerate(edits):
            _res = self._apply_one_edit_text(
                _cur_content, file_path, e["old_string"], e["new_string"], e["replace_all"],
                scope=e.get("scope"),
            )
            if not _res["ok"]:
                # Single mode: return the raw error verbatim (preserves the
                # exact message existing tests/callers depend on). Batch mode:
                # annotate with the failing edit's index.
                if is_batch:
                    _error = (
                        f"edit_text refused (file NOT modified): edit #{i + 1} "
                        f"(edits[{i}]) failed to match — no edits were applied "
                        f"(atomic batch).\n" + _res["error"]
                    )
                else:
                    _error = _res["error"]
                _meta = dict(_res.get("metadata", {}))
                if is_batch:
                    _meta["failed_edit_index"] = i
                    _meta["applied_edits"] = i  # edits before this one were computed in-memory only
                return self._make_result(
                    ok=False,
                    error=_error,
                    metadata=_meta,
                    execution_time=_time.monotonic() - start_time,
                )
            _cur_content = _res["new_content"]
            _total_occurrences += _res["occurrences"]
            if _res["high_count_warning"]:
                _high_count_warnings.append(_res["high_count_warning"])
            if i == 0:
                _first_match_line = _res.get("match_line", 0)
                _first_match_indent = _res.get("match_indent", 0)
                _first_reindent_applied = _res.get("reindent_applied", False)
        new_content = _cur_content

        # ── Syntax gate: refuse to write a .py edit that would BREAK parsing ──
        # edit_text does pure string replacement plus an indent-tolerant
        # fallback; a reindent that lands content at the wrong column (tab/space
        # mismatch, non-uniform old_string indent) silently corrupts the file —
        # the one write tool with no rollback.  Catch it here, in memory, BEFORE
        # touching disk.  Only gate when the ORIGINAL file parsed, so we never
        # block an edit that is fixing a pre-existing syntax error.
        if LanguageId.from_path(file_path) is LanguageId.PYTHON:
            _orig_parses = True
            try:
                compile_quiet(content, file_path, "exec")
            except SyntaxError:
                _orig_parses = False
            except Exception:
                _orig_parses = True  # non-SyntaxError → don't block on it
            if _orig_parses:
                try:
                    compile_quiet(new_content, file_path, "exec")
                except SyntaxError as _se:
                    # ── Scope-aware diagnosis ──
                    # Python reports an INDENTATION/structure error on the line
                    # where the parser NOTICES it, not necessarily the line whose
                    # indentation is wrong. We compare pre/post-edit content to
                    # tell whether ``_se.lineno`` was actually touched by this
                    # edit, so the LLM gets an actionable diagnosis instead of
                    # guessing whether it broke its own block or a cascade from
                    # elsewhere surfaced lines away.
                    _err_line = _se.lineno or 0
                    _in_edited, _regions = self._edited_line_regions(
                        content, new_content, _err_line
                    )
                    _region_str = (
                        ", ".join(f"L{s}-{e}" for (s, e) in _regions[:6])
                        or "unknown"
                    )
                    if _in_edited:
                        _diagnosis = (
                            "The error line is INSIDE the block you just edited "
                            f"(edited regions: {_region_str}). This is almost always "
                            "an indentation mistake in new_string — copy the exact "
                            "indentation (including comment lines) from the file. "
                            "Note Python may report the error one or two lines "
                            "BELOW the actually-misindented line."
                        )
                    else:
                        _diagnosis = (
                            "The error line was NOT directly edited (edited regions: "
                            f"{_region_str}), so this is likely a CASCADE: new_string "
                            "changed a block's structure (indentation, dedent, or an "
                            "unbalanced bracket/colon) whose effect the parser only "
                            "notices here. Check that new_string preserves the "
                            "surrounding block's indentation and that you didn't "
                            "accidentally drop a line or close a block early."
                        )
                    # ── Structural-imbalance hint ──
                    # ast.parse pinpoints the missing structural token
                    # ("expected 'except' or 'finally'", "'(' was never
                    # closed", ...). Surface it up-front so the LLM knows WHICH
                    # token got truncated — no file re-read required. Reuses the
                    # exact error Python raised, so no false positives.
                    _structure_hint = self._structural_imbalance_hint(_se.msg or "")
                    if _structure_hint:
                        _diagnosis = _structure_hint + " " + _diagnosis
                    # ── Concrete indentation hint ──
                    # Generic messages ("unexpected indent") never state the
                    # column count, so the LLM retries guessing. Compute the
                    # actual expected width from neighbouring lines.
                    _indent_hint = self._indentation_hint(
                        new_content, _err_line, _se.msg or ""
                    )
                    if _indent_hint:
                        _diagnosis += " " + _indent_hint
                    return self._make_result(
                        ok=False,
                        error=(
                            f"edit_text refused (file NOT modified): the replacement would "
                            f"introduce a Python syntax error in {file_path}: "
                            f"{_se.msg} at line {_se.lineno}.\n"
                            + _diagnosis
                        ),
                        metadata={
                            "syntax_error": str(_se),
                            "syntax_error_line": _err_line,
                            "error_in_edited_region": _in_edited,
                            "edited_regions": _regions,
                            "indentation_hint": _indent_hint,
                            "written": False,
                            "matched": True,
                            "failure_class": "syntax_invalid_after_edit",
                        },
                        execution_time=_time.monotonic() - start_time,
                    )

        # ── Language-neutral syntax gate (non-Python) ──────────────────────
        # edit_text is excluded from dispatch's snapshot+verify+rollback cycle
        # (tool_registry.py:1264/1271) because it has no rollback path, and the
        # Python ``compile()`` gate above only covers .py. For every OTHER
        # language we run the SAME provider.validate_syntax the dispatch path
        # uses — in memory, BEFORE writing — so a broken new_string never reaches
        # disk. The gate mirrors dispatch exactly: only GENUINE syntax errors
        # (FailureType.SYNTAX_ERROR) are refused; soft-fail errors that may
        # resolve cross-file (Go "undefined:", Java "cannot find symbol" →
        # UNKNOWN_SYMBOL) are KEPT, so edit_text is neither stricter nor looser
        # than apply_patch/edit_file for the same file+edit. Skip when the
        # ORIGINAL already failed parsing (we never block an edit fixing a
        # pre-existing error), matching the Python branch above.
        _et_lang = LanguageId.from_path(file_path)
        if _et_lang is not LanguageId.PYTHON and _et_lang is not LanguageId.UNKNOWN:
            try:
                from ...languages import LanguageRegistry as _LR_ET
                _et_provider = _LR_ET.instance().get(file_path)
            except Exception:
                _et_provider = None
            if (
                _et_provider is not None
                and _et_provider.capabilities().has_syntax_validator
            ):
                try:
                    _et_orig_ok = _et_provider.validate_syntax(
                        file_path, content
                    ).ok
                except Exception:
                    _et_orig_ok = True  # validator crash → don't block the edit
                if _et_orig_ok:
                    try:
                        _et_new_val = _et_provider.validate_syntax(
                            file_path, new_content
                        )
                    except Exception:
                        _et_new_val = None
                    if _et_new_val is not None and not _et_new_val.ok:
                        _et_errs = _et_new_val.errors or []
                        if _et_errs:
                            _e0 = _et_errs[0]
                            _et_detail = (
                                f"{_e0.file}:{_e0.line}:{_e0.col}: {_e0.message}"
                            )
                            for _e in _et_errs[1:3]:
                                _et_detail += (
                                    f"; L{_e.line}:{_e.col} {_e.message}"
                                )
                            if len(_et_errs) > 3:
                                _et_detail += (
                                    f" (+{len(_et_errs) - 3} more syntax errors)"
                                )
                        else:
                            _et_detail = f"syntax error in {file_path}"
                        # Mirror dispatch soft-fail: keep cross-file-resolvable
                        # errors; refuse only genuine syntax errors.
                        if not self._should_soft_fail_verify(
                            _et_detail, {file_path: content}
                        ):
                            return self._make_result(
                                ok=False,
                                error=(
                                    f"edit_text refused (file NOT modified): the "
                                    f"replacement would introduce a syntax error "
                                    f"in {file_path}: {_et_detail}"
                                ),
                                metadata={
                                    "syntax_error": _et_detail,
                                    "written": False,
                                    "matched": True,
                                    "failure_class": "syntax_invalid_after_edit",
                                },
                                execution_time=_time.monotonic() - start_time,
                            )
                        # soft-fail → fall through and write (dispatch keeps these)
        # Write back with the encoding the file was read with — re-encoding a
        # latin-1 file as UTF-8 would alter every non-ASCII byte. Encode BEFORE
        # opening the file: write_text() truncates first and encodes during the
        # write, so an encode failure there would leave the file EMPTY.
        try:
            _encoded = new_content.encode(_read_encoding)
        except UnicodeEncodeError as e:
            return self._make_result(
                ok=False,
                error=(
                    f"Failed to write {file_path}: new_string contains characters "
                    f"not representable in the file's encoding ({_read_encoding}): {e}"
                ),
                execution_time=0,
            )
        try:
            _norm.write_bytes(_encoded)
        except Exception as e:
            return self._make_result(ok=False, error=f"Failed to write {file_path}: {e}", execution_time=0)

        _added = len(new_content) - len(content)
        _exec = _time.monotonic() - start_time
        # Track applied patch so agent_loop can detect successful writes
        try:
            self._applied_patches.append(
                f"edit_text:{file_path}:replace:{is_batch}"
            )
        except (AttributeError, TypeError):
            pass
        self._record_text_edit(file_path)
        _enc_detail = f" [enc: {_read_encoding}]" if _read_encoding != "utf-8" else ""
        _high_warn = "".join(_high_count_warnings)
        if is_batch:
            _content_msg = (
                f"Edited {file_path} ({len(edits)} edits, "
                f"{_total_occurrences} occurrence"
                f"{'s' if _total_occurrences != 1 else ''} replaced, "
                f"{_added:+d} chars{_enc_detail}){_high_warn} [{_exec:.1f}s]"
            )
        else:
            # Preserve the exact single-edit success wording.
            _count_detail = (
                f"{_total_occurrences} occurrence"
                f"{'s' if _total_occurrences > 1 else ''}"
            )
            _content_msg = (
                f"Edited {file_path} (replaced {_count_detail}, "
                f"{_added:+d} chars{_enc_detail}){_high_warn} [{_exec:.1f}s]"
            )
        # Semantic feedback (non-blocking): pyright/tsc/go diagnostics for
        # type/undefined-name/import issues. Mirrors apply_patch/edit_file/
        # modify_symbol/anchor_edit. The blocking syntax gate above already
        # refused syntactically-broken Python edits; this catches semantic
        # problems (undefined names, type mismatches) and surfaces them for
        # LLM self-healing. _norm is on disk at this point.
        _meta: dict[str, Any] = {}
        if _high_count_warnings:
            _meta["high_count_warnings"] = _high_count_warnings
        #── Enot/inside: surface the edit site's actual indentation ──
        # In single mode, matched_line/matched_indent let the LLM verify it hit
        # the intended location at the intended depth — paired with read_file's
        # │N│ gutter, this closes the indent-guessing loop. reindent_applied
        # warns that new_string's indentation was auto-corrected to old_string's
        # base indent (the LLM's original new_string indent did not match the
        # file). Batch mode omits per-edit detail (see comment in the loop).
        if not is_batch:
            if _first_match_line:
                _meta["matched_line"] = _first_match_line
            _meta["matched_indent"] = _first_match_indent
            if _first_reindent_applied:
                _meta["reindent_applied"] = True
        _syn = self._run_syntax_check_for_file(str(_norm))
        if not _syn.get("skipped"):
            _meta["syntax_check"] = _syn
        return self._make_result(
            ok=True,
            content=_content_msg,
            metadata=_meta,
            execution_time=_exec,
        )

    def _tool_create_file(self, args: dict[str, Any]) -> "ToolResult":
        """Create a new file with the given content.

        Creates parent directories automatically if they don't exist.
        Fails if the file already exists (use write_plan for overwrites).
        """
        import time as _time
        start_time = _time.monotonic()
        args = self._recover_args_from_raw(args, ("path",))
        file_path = (args.get("path") or "").strip()
        content = args.get("content", "")
        description = args.get("description", "")
        overwrite = args.get("overwrite", False)

        if not file_path:
            # If __raw_arguments is present, the JSON was likely truncated during streaming
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(ok=False, error=f"path is required{_raw_hint}", execution_time=0)

        _norm = Path(self.repo_root) / file_path

        if _norm.exists() and not overwrite:
            return self._make_result(
                ok=False,
                error=f"File already exists: {file_path} (use overwrite=True to replace)",
                execution_time=0,
            )

        try:
            _norm.parent.mkdir(parents=True, exist_ok=True)
            _norm.write_text(content, encoding="utf-8")
        except Exception as e:
            return self._make_result(
                ok=False, error=f"Failed to create {file_path}: {e}"
            )

        _exec = _time.monotonic() - start_time
        _desc = f" ({description})" if description else ""
        _size = len(content)
        return self._make_result(
            ok=True,
            content=f"Created: {file_path}{_desc} ({_size} chars) [{_exec:.1f}s]",
            execution_time=_exec,
        )

    def _extract_ops_from_raw(self, raw: str) -> list[dict[str, Any]]:
        """Try to extract ``operations`` list from truncated raw JSON string.

        Stream truncation can cut the JSON before the outer ``}``, leaving a
        complete ``"operations": [...]`` inside the partial string.  Extract
        the array via bracket matching instead of full JSON parsing.
        """
        import json as _json
        _m = re.search(r'"operations"\s*:\s*(\[)', raw)
        if not _m:
            return []
        _start = _m.start(1)
        _depth = 0
        _end = -1
        for _i, _c in enumerate(raw[_start:], start=_start):
            if _c == '[':
                _depth += 1
            elif _c == ']':
                _depth -= 1
                if _depth == 0:
                    _end = _i + 1
                    break
        if _end == -1:
            return []  # truncated inside the array — cannot recover
        try:
            _parsed = _json.loads(raw[_start:_end])
            if isinstance(_parsed, list):
                return _parsed
        except _json.JSONDecodeError:
            pass
        return []

    def _recover_args_from_raw(
        self,
        args: dict[str, Any],
        required_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        """Try to recover args from __raw_arguments when content or required keys are missing.

        Several provider paths preserve the raw tool call arguments as
        __raw_arguments when JSON parsing fails (stream truncation, model
        error).  This method attempts to re-parse that raw string and
        substitute the result when required keys are absent.

        Additionally, ``content`` is recovered from truncated JSON even when
        ``required_keys`` are already present — content is the most common
        truncation victim (it is the last parameter in many schemas and can be
        very large).
        """
        if "__raw_arguments" not in args:
            return args
        raw = args["__raw_arguments"]
        if not isinstance(raw, str):
            return args

        # ── Full json.loads (handles complete JSON) ──
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                logger.info("recovered args from __raw_arguments for %s", required_keys)
                return parsed
        except json.JSONDecodeError:
            pass

        # ── Regex fallback: extract simple string keys from truncated JSON ──
        # Stream truncation can produce incomplete JSON like:
        #   {"path": "example.py", "operations": [{"type": "replace", ...
        # json.loads can't parse it, but simple string values ARE present.
        try:
            _result = dict(args)  # preserve __raw_arguments
            for _key in required_keys:
                if _key not in _result or not _result.get(_key):
                    # Escape-aware value pattern: ([^"\\]|\\.)* steps over \" and \\
                    # instead of truncating at the first escaped quote.
                    _m = re.search(
                        r'"' + re.escape(_key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', raw
                    )
                    if _m is not None:
                        # JSON-unescape the captured value — the raw text contains
                        # literal \n / \" / \\ sequences. Writing them through
                        # un-decoded would corrupt file content (e.g. a recovered
                        # new_string with literal backslash-n baked into the file).
                        try:
                            _result[_key] = json.loads('"' + _m.group(1) + '"')
                        except (ValueError, json.JSONDecodeError):
                            continue  # leave key missing → tool returns a clean error
            if all(_result.get(k) for k in required_keys):
                logger.info(
                    "recovered args from __raw_arguments (regex fallback) for %s",
                    required_keys,
                )
                return _result
        except Exception:
            pass

        # ── JSON repair: close unterminated strings/objects ──
        # Truncated JSON like {"path": "example.py", "content": "line1\nline2" can't
        # be parsed by json.loads because the content string and/or object
        # are unterminated.  Try adding closing quotes/braces to fix.
        _repaired = self._try_repair_truncated_json(raw)
        if _repaired is not None:
            _result = dict(args)
            _result.update(_repaired)
            _result.pop("__raw_arguments", None)
            logger.info("recovered args from __raw_arguments (JSON repair) for %s", required_keys)
            return _result

        return args

    @staticmethod
    def _try_repair_truncated_json(raw: str) -> Optional[dict[str, Any]]:
        """Attempt to repair and parse a truncated JSON string.

        Streaming truncation can cut off the end of a JSON object, e.g.:
          {"path": "example.py", "content": "line1\nline2
        This tries adding ``\"}`` (close string + object) or just ``}`` to
        produce valid JSON.
        """
        if not raw.startswith("{"):
            return None
        _open_b = raw.count("{")
        _close_b = raw.count("}")
        if _open_b <= _close_b:
            return None  # braces balanced or closed more than opened — not a truncation case

        # Only ever close the OBJECT — never close an unterminated string.
        # If the raw ends mid-string, the last value was cut off by the
        # truncation; appending ``"}`` would make the partial value parse as
        # if it were complete, and a tool would then silently write half a
        # file (create_file/write_plan content is the most common victim).
        # Refusing the repair lets the caller surface a clean "truncated
        # arguments" error so the LLM can retry instead.
        try:
            _parsed = json.loads(raw + "}")
            if isinstance(_parsed, dict):
                return _parsed
        except json.JSONDecodeError:
            pass
        return None

    def _run_syntax_check_for_file(self, rel_or_abs_path: str) -> dict:
        """Run post-apply syntax validation for *rel_or_abs_path* if a provider exists.

        Always returns a dict.  ``skipped=True`` means no provider is registered
        for this file type — callers should omit the result from metadata rather
        than treating it as an error.

        When syntax is OK and the provider supports semantic validation
        (``has_semantic_validator``), an additional ``semantic_diagnostics``
        field is populated with type/undefined-name/import diagnostics collected
        by running the backing tool (pyright/tsc/go build) against the real
        project. These are **non-blocking** — surfaced for LLM self-healing.
        """
        try:
            import os

            from ...languages.registry import LanguageRegistry

            abs_path = (
                rel_or_abs_path
                if os.path.isabs(rel_or_abs_path)
                else os.path.join(str(self.repo_root), rel_or_abs_path)
            )
            provider = LanguageRegistry.instance().get(abs_path)
            if provider is None:
                return {"ok": True, "skipped": True, "reason": "no_provider"}

            try:
                with open(abs_path, encoding="utf-8") as fh:
                    content = fh.read()
            except OSError:
                return {"ok": True, "skipped": True, "reason": "file_read_error"}

            result = provider.validate_syntax(abs_path, content)
            out = {
                "ok": result.ok,
                "language": result.language.value if result.language else None,
                "errors": [
                    {"line": e.line, "col": e.col, "message": e.message}
                    for e in (result.errors or [])
                ],
            }
            # Only run semantic check on syntactically-valid files to avoid
            # cascading-error noise from the backing tool.
            if result.ok and provider.capabilities().has_semantic_validator:
                try:
                    sem = provider.validate_semantics(abs_path)
                    out["semantic_diagnostics"] = [
                        {
                            "file_path": abs_path,
                            "line": e.line, "col": e.col,
                            "message": e.message,
                            "severity": getattr(e, "severity", "error"),
                            "code": getattr(e, "code", ""),
                        }
                        for e in (sem.errors or [])
                    ]
                except Exception as sem_exc:
                    logger.debug("Semantic check failed for %s: %s", abs_path, sem_exc)
                    out["semantic_diagnostics"] = []
            return out
        except Exception as exc:
            logger.debug("Post-apply syntax check failed: %s", exc)
            return {"ok": True, "skipped": True, "reason": "exception"}


    def _norm_repo_rel(self, p: str) -> str:
        """Normalize a path (absolute or relative) to repo-root-relative form."""
        if not p:
            return ""
        rr = str(getattr(self, "_effective_repo_root", None) or getattr(self, "repo_root", ""))
        if rr and p.startswith(rr):
            p = p[len(rr):]
        return p.lstrip("/")

    def _record_text_edit(self, file_path: str) -> None:
        """Record that a text-editing tool wrote ``file_path`` this session.

        Tracked so apply_patch can refuse to clobber a working-tree edit it cannot
        safely merge: apply_patch / diff_apply reconstructs hunk context from HEAD,
        and on a freshly-edited target PatchEngine uses skip_3way=True whose
        _rollback() reverts the working tree to HEAD — silently deleting the edit.
        See the session-edit guards in _tool_apply_patch and _apply_patch_text.
        """
        try:
            rel = self._norm_repo_rel(file_path)
            if rel:
                self._text_edited_files.add(rel)
        except Exception:
            pass

    def _apply_patch_text(self, patch_text: str, path_hint: Optional[str] = None) -> "ToolResult":
        """Shared internal method to apply a patch via git apply chain."""
        import time as _time
        start_time = _time.monotonic()

        # ── Hard guard: capture uncommitted-changes state BEFORE any apply ──
        # `git apply` (and diff_apply, which wraps it) reconstructs hunk context
        # from the HEAD blob, so a patch whose context matches HEAD can silently
        # overwrite pre-existing working-tree edits on a DIRTY file (e.g. one
        # just edited via edit_text / modify_symbol / anchor_edit this session).
        # We detect this BEFORE applying — post-apply detection is meaningless
        # (the patch itself makes the file differ from HEAD) — and REJECT the
        # patch outright so the working tree is never mutated. The caller must
        # continue editing such files with edit_text / modify_symbol.
        try:
            _pre_touched_files = extract_files_from_patch(patch_text)
        except Exception:
            _pre_touched_files = []
        # Opt D session-edit guard (mirrors the main entry guard in _tool_apply_patch).
        _pre_session = [p for p in _pre_touched_files if self._norm_repo_rel(p) in self._text_edited_files]
        if _pre_session:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    "apply_patch refused: target file(s) were already edited this session "
                    "via edit_text / modify_symbol / edit_ast / anchor_edit — those edits "
                    "live in the working tree but apply_patch would revert to HEAD on "
                    "conflict (silently losing them): "
                    + ", ".join(_pre_session)
                    + ". Continue editing these files with the same text-editing tool instead."
                ),
                execution_time=_time.monotonic() - start_time,
                metadata={
                    "refused_dirty_files": _pre_session,
                    "reason": "session_text_edit_overwrite_risk",
                },
            )

        try:
            from diff_apply import apply_patch
        except ImportError:
            apply_patch = None
        if apply_patch is not None:
            ok, msg, _reason, details = apply_patch(self._effective_repo_root, patch_text, file_path_hint=path_hint)
            execution_time = _time.monotonic() - start_time
            if ok:
                # Track applied patch so agent_loop can detect successful writes
                # (applied_patches non-empty = "real edit happened", avoids false-success nudge)
                try:
                    self._applied_patches.append(str(patch_text))
                except (AttributeError, TypeError):
                    pass
                return self._make_result(ok=True, content=msg, execution_time=execution_time, metadata=details)
            else:
                return self._make_result(ok=False, content="", error=msg, execution_time=execution_time, metadata=details)

        try:
            from services.patch_helpers import normalize_patch_text
        except ImportError:
            def normalize_patch_text(x: str) -> str:
                return x or ""

        try:
            from diff_apply import _clean_diff, extract_touched_files_from_diff
        except ImportError:
            _clean_diff = None
            extract_touched_files_from_diff = None

        patch_norm = normalize_patch_text(patch_text)

        if "diff --git a/.." in patch_norm or "diff --git b/.." in patch_norm:
            return self._make_result(ok=False, content="", error="Unsafe path in patch (path traversal detected)")

        patch_clean = patch_norm
        if _clean_diff is not None:
            try:
                patch_clean = _clean_diff(patch_norm, self.repo_root, file_path_hint=path_hint)
            except Exception as e:
                return self._make_result(ok=False, content="", error=f"Patch cleanup failed: {e}")

        # Private (mode 0o600), unpredictably-named temp file for the patch.
        # A fixed /tmp path created with default umask leaks source-code diffs
        # to other users on shared/multi-user systems. mkstemp creates the file
        # atomically with restrictive perms; subsequent open(patch_file, "w")
        # calls truncate and rewrite while preserving the 0o600 mode (open never
        # relaxes permissions of an existing file).
        try:
            _fd, patch_file = tempfile.mkstemp(suffix=".patch", prefix="asicode.")
            os.close(_fd)
        except OSError as _e:
            return self._make_result(ok=False, content="", error=f"Failed to create temp patch file: {_e}")
        try:

            if not patch_clean.strip():
                try:
                    synthesized = None
                    if PatchEngine is not None and path_hint:
                        try:
                            engine = PatchEngine(self._effective_repo_root)
                            synthesized = engine._salvage_small_model_output(patch_text, path_hint)
                        except Exception as e:
                            logger.debug("PatchEngine salvage failed: %s", e)
                            synthesized = None
                    if synthesized and synthesized.strip():
                        try:
                            with open(patch_file, "w", encoding="utf-8") as fh:
                                fh.write(synthesized)
                        except OSError as e:
                            return self._make_result(ok=False, content="", error=f"Failed to write synthesized patch file: {e}")
                        # Validate synthesized patch with git apply --check BEFORE accepting it
                        check = subprocess.run(
                            ["git", "apply", "--check", patch_file],
                            cwd=self._effective_repo_root,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if check.returncode == 0:
                            patch_clean = synthesized
                            logger.info("Recovered empty normalized patch via small-model synthesizer")
                        else:
                            return self._make_result(
                                ok=False,
                                content="",
                                error="empty diff after cleaning (salvage failed git apply --check)",
                                metadata={
                                    "check_stderr": (check.stderr or "").strip(),
                                },
                            )
                    else:
                        return self._make_result(ok=False, content="", error="empty diff after cleaning")
                except Exception as e:
                    logger.debug("Pre-check synthesizer failed: %s", e)
                    return self._make_result(ok=False, content="", error="empty diff after cleaning")
            patch_sha256 = hashlib.sha256(patch_clean.encode("utf-8")).hexdigest()[:16]
            patch_len = len(patch_clean)
            try:
                with open(patch_file, "w", encoding="utf-8") as fh:
                    fh.write(patch_clean)
            except OSError as e:
                return self._make_result(ok=False, content="", error=f"Failed to write patch file: {e}")

            _head_lines = (patch_clean.lstrip().splitlines()[:8] if patch_clean else [])
            _has_git_header = ("diff --git " in patch_clean)
            _looks_like_ab_paths = any(s.startswith("--- a/") for s in _head_lines) and any(s.startswith("+++ b/") for s in _head_lines)
            _needs_p1 = (not _has_git_header) and _looks_like_ab_paths

            _apply_base = ["git", "apply"]
            if _needs_p1:
                _apply_base.append("-p1")

            use_ignore_ws = False
            try:
                check = subprocess.run(
                    [*_apply_base, "--check", patch_file],
                    cwd=self._effective_repo_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if check.returncode != 0:
                    check_ws = subprocess.run(
                        [*_apply_base, "--check", "--ignore-whitespace", patch_file],
                        cwd=self._effective_repo_root,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if check_ws.returncode == 0:
                        use_ignore_ws = True
                        logger.info("Patch check passed with --ignore-whitespace; will apply with flag")
                    else:
                        # --check failed even with --ignore-whitespace. Last resort:
                        # small-model diff synthesizer. If it can't produce a patch
                        # that passes --check, fail HERE. (A previous regression put
                        # this return inside the synthesizer's `except` block, so a
                        # normally-failing check fell through to the real `git apply`;
                        # it also NameError'd when the ws-check had passed.)
                        failure_analysis = self._analyze_patch_failure(patch_clean, check.stderr or "")
                        _salvaged = False
                        try:
                            synthesized = None
                            if PatchEngine is not None and path_hint:
                                try:
                                    engine = PatchEngine(self._effective_repo_root)
                                    synthesized = engine._salvage_small_model_output(patch_text, path_hint)
                                except Exception as e:
                                    logger.debug("PatchEngine salvage failed: %s", e)
                                    synthesized = None
                            if synthesized:
                                logger.info("Patch synthesizer activated for small-model diff repair")
                                with open(patch_file, "w", encoding="utf-8") as fh:
                                    fh.write(synthesized)
                                retry = subprocess.run(
                                    [*_apply_base, "--check", patch_file],
                                    cwd=self._effective_repo_root,
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                )
                                if retry.returncode == 0:
                                    patch_clean = synthesized
                                    _salvaged = True
                                    logger.info("Synthesized diff accepted by git apply")
                                else:
                                    logger.debug("Synthesized diff still invalid: %s", (retry.stderr or retry.stdout or "").strip())
                        except Exception as e:
                            logger.debug("Small-model diff synthesizer failed: %s", e)

                        if not _salvaged:
                            return self._make_result(
                                ok=False,
                                content="",
                                error=failure_analysis.get("error_message", "git apply --check failed"),
                                metadata={
                                    "patch_file": patch_file,
                                    "patch_sha256": patch_sha256,
                                    "patch_len": patch_len,
                                    "check_stderr": (check.stderr or "").strip(),
                                    "failure_analysis": failure_analysis,
                                },
                            )
            except subprocess.TimeoutExpired:
                return self._make_result(ok=False, content="", error="git apply --check timeout after 30 seconds",
                                         metadata={"patch_file": patch_file, "timeout": True})
            except Exception as e:
                return self._make_result(ok=False, content="", error=f"git apply --check error: {e}",
                                         metadata={"patch_file": patch_file})

            # ── Pre-apply snapshot for rollback ────────────────────────────
            # Extract file paths from diff BEFORE apply, snapshot their content.
            import os as _os_snap

            _pre_touched: list[str] = extract_files_from_patch(patch_text)


            _pre_apply_snapshot: dict[str, str] = {}
            for _tf_snap in _pre_touched:
                _abs_snap = _os_snap.path.join(self._effective_repo_root, _tf_snap)
                if _os_snap.path.isfile(_abs_snap):
                    try:
                        with open(_abs_snap, encoding="utf-8", errors="replace") as _fsnap:
                            _pre_apply_snapshot[_abs_snap] = _fsnap.read()
                    except OSError:
                        pass

            # Hard guard: capture uncommitted-changes state BEFORE apply (post-apply
            # detection is meaningless — the patch itself makes files differ from
            # HEAD) and REJECT so the working tree is never mutated. Mirrors the
            # diff_apply main path.
            # Opt D session-edit guard (mirrors the main entry guard in _tool_apply_patch).
            # NB: this pure `git apply` path has no _rollback, so it is safe even for dirty
            # files — but we still refuse session-edited targets to give a clear "use
            # edit_text instead" message rather than a confusing context-mismatch failure.
            _pre_session = [p for p in _pre_touched if self._norm_repo_rel(p) in self._text_edited_files]
            if _pre_session:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        "apply_patch refused: target file(s) were already edited this session "
                        "via edit_text / modify_symbol / edit_ast / anchor_edit — those edits "
                        "live in the working tree but apply_patch would revert to HEAD on "
                        "conflict (silently losing them): "
                        + ", ".join(_pre_session)
                        + ". Continue editing these files with the same text-editing tool instead."
                    ),
                    execution_time=_time.monotonic() - start_time,
                    metadata={
                        "refused_dirty_files": _pre_session,
                        "reason": "session_text_edit_overwrite_risk",
                    },
                )

            apply_cmd = list(_apply_base)
            if use_ignore_ws:
                apply_cmd.append("--ignore-whitespace")
            apply_cmd.append(patch_file)
            try:
                apply_proc = subprocess.run(
                    apply_cmd,
                    cwd=self._effective_repo_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if apply_proc.returncode != 0:
                    failure_analysis = self._analyze_patch_failure(patch_clean, apply_proc.stderr or "")
                    return self._make_result(
                        ok=False,
                        content="",
                        error=failure_analysis.get("error_message", "git apply failed"),
                        metadata={
                            "patch_file": patch_file,
                            "patch_sha256": patch_sha256,
                            "patch_len": patch_len,
                            "apply_stderr": (apply_proc.stderr or "").strip(),
                            "failure_analysis": failure_analysis,
                        },
                    )
            except subprocess.TimeoutExpired:
                return self._make_result(ok=False, content="", error="git apply timeout after 30 seconds",
                                         metadata={"patch_file": patch_file, "timeout": True})
            except Exception as e:
                return self._make_result(ok=False, content="", error=f"git apply error: {e}",
                                         metadata={"patch_file": patch_file})

            ratio_warning = self._check_patch_content_ratio(patch_clean)

            touched: list[str] = []
            if extract_touched_files_from_diff is not None:
                try:
                    touched = extract_touched_files_from_diff(patch_clean)
                except (ValueError, TypeError, AttributeError):
                    pass

            if not touched:
                try:
                    for line in patch_clean.splitlines():
                        if line.startswith("diff --git "):
                            parts = line.split()
                            if len(parts) >= 4:
                                a_path = _PATCH_PATH_PREFIX_RE.sub("", parts[2])
                                b_path = _PATCH_PATH_PREFIX_RE.sub("", parts[3])
                                rel = (b_path or a_path).strip().replace("\\", "/").lstrip("/")
                                if rel and rel not in touched:
                                    touched.append(rel)
                        elif line.startswith("+++ b/"):
                            rel = line[6:].strip().replace("\\", "/").lstrip("/")
                            if rel and rel not in touched:
                                touched.append(rel)
                        elif line.startswith("--- a/"):
                            rel = line[6:].strip().replace("\\", "/").lstrip("/")
                            if rel and rel not in touched:
                                touched.append(rel)
                except (AttributeError, TypeError):
                    pass

            # ── Post-apply syntax validation + snapshot-based rollback ────────
            _syntax_errors: list[str] = []
            for _tf_chk in (touched or _pre_touched):
                _abs_chk = _os_snap.path.join(self._effective_repo_root, _tf_chk)
                if LanguageId.from_path(_abs_chk) is LanguageId.PYTHON and _os_snap.path.isfile(_abs_chk):
                    try:
                        with open(_abs_chk, encoding="utf-8", errors="replace") as _fchk:
                            _post_content = _fchk.read()
                        compile_quiet(_post_content, _tf_chk, "exec")
                    except SyntaxError as _se:
                        _syntax_errors.append(f"{_tf_chk}: {_se}")
                    except Exception as _exc:
                        logger.debug("Post-apply compile() check raised non-SyntaxError: %s", _exc)

            if _syntax_errors:
                # Rollback using pre-apply snapshot (git-independent)
                for _snap_path, _snap_content in _pre_apply_snapshot.items():
                    try:
                        with open(_snap_path, "w", encoding="utf-8") as _froll:
                            _froll.write(_snap_content)
                    except Exception as _roll_exc:
                        logger.debug("Rollback write failed for %s: %s", _snap_path, _roll_exc)
                logger.warning(
                    "Patch introduced syntax errors — rolled back from snapshot: %s",
                    _syntax_errors,
                )
                return self._make_result(
                    ok=False,
                    content="",
                    error=f"Patch introduced syntax errors (rolled back): {'; '.join(_syntax_errors)}",
                    metadata={"syntax_errors": _syntax_errors, "rolled_back": True},
                )

            self._applied_patches.append(patch_clean)
            if touched:
                self._invalidate_cache_after_write(touched)
            content_msg = f"Patch applied successfully. Touched files: {', '.join(touched) or 'unknown'}"
            if ratio_warning:
                content_msg += f"\n{ratio_warning}"
            _fallback_meta = {"touched_files": touched, "patch": patch_clean,
                              "content_ratio_warning": ratio_warning}
            return self._make_result(
                ok=True,
                content=content_msg,
                metadata=_fallback_meta,
            )
        finally:
            try:
                os.unlink(patch_file)
            except OSError:
                pass

    def _analyze_patch_failure(self, patch_text: str, git_error: str) -> dict[str, Any]:
        import re
        try:
            from ..failure_context import analyze_failure
            failure_ctx = analyze_failure(
                stage="git_apply_check",
                raw_text=git_error,
                repo_root=self.repo_root
            )
        except ImportError:
            failure_ctx = None

        file_path = None
        hunks = []

        lines = patch_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('--- a/'):
                file_path = line[6:].strip()
                if file_path == '/dev/null':
                    file_path = None
                if i + 1 < len(lines) and lines[i + 1].startswith('+++ b/'):
                    new_file_path = lines[i + 1][6:].strip()
                    if new_file_path != '/dev/null':
                        file_path = new_file_path
                break
            elif line.startswith('+++ b/'):
                file_path = line[6:].strip()
                if file_path == '/dev/null':
                    file_path = None
                break
            elif line.startswith('diff --git a/'):
                parts = line.split()
                if len(parts) >= 4:
                    b_path = parts[3]
                    if b_path.startswith('b/'):
                        file_path = b_path[2:]
                    else:
                        file_path = b_path
                    break

        current_hunk = None

        for line in lines:
            hunk_match = _HUNK_HEADER_RE.match(line)
            if hunk_match:
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = {
                    'old_start': int(hunk_match.group(1)),
                    'old_lines': int(hunk_match.group(2)),
                    'new_start': int(hunk_match.group(3)),
                    'new_lines': int(hunk_match.group(4)),
                    'context_lines': [],
                    'original_lines': []
                }
            elif current_hunk:
                if line.startswith(' '):
                    content = line[1:]
                    current_hunk['context_lines'].append(content)
                    current_hunk['original_lines'].append(('context', content))
                elif line.startswith('-'):
                    content = line[1:]
                    current_hunk['original_lines'].append(('remove', content))
                elif line.startswith('+'):
                    content = line[1:]
                    current_hunk['original_lines'].append(('add', content))

        if current_hunk:
            hunks.append(current_hunk)

        error_lower = git_error.lower()
        reason = "unknown"
        hint = ""
        conflicting_lines = []

        if "already applied" in error_lower or "already exists" in error_lower or "file exists" in error_lower:
            reason = "already_applied"
            hint = "Patch appears to be already applied to the file."
        elif "corrupt patch" in error_lower or "patch format" in error_lower or "unrecognized input" in error_lower:
            reason = "offset_error"
            hint = "Patch format is corrupt or malformed."
        elif "does not apply" in error_lower or "patch failed" in error_lower or "hunk failed" in error_lower:
            reason = "context_mismatch"
            hint = "Patch context does not match the current file content."

            line_match = re.search(r'at line\s+(\d+)', git_error)
            if not line_match:
                line_match = re.search(r'line\s+(\d+)', git_error)
            if not line_match:
                line_match = re.search(r'hunk\s+(\d+)', git_error)

            if line_match:
                conflicting_line = int(line_match.group(1))
                conflicting_lines.append(conflicting_line)

                if file_path:
                    try:
                        file_full_path = Path(self.repo_root) / file_path
                        if file_full_path.exists():
                            with open(file_full_path, encoding='utf-8') as f:
                                file_content = f.read()
                                file_lines_list = file_content.split('\n')

                                if 0 <= conflicting_line - 1 < len(file_lines_list):
                                    actual_line = file_lines_list[conflicting_line - 1]

                                    expected_line = None
                                    for hunk in hunks:
                                        if hunk['old_start'] <= conflicting_line <= hunk['old_start'] + hunk['old_lines']:
                                            offset = conflicting_line - hunk['old_start']
                                            context_counter = 0
                                            for line_type, content in hunk['original_lines']:
                                                if line_type == 'context':
                                                    if context_counter == offset:
                                                        expected_line = content
                                                        break
                                                    context_counter += 1
                                                elif line_type == 'remove':
                                                    context_counter += 1
                                            break

                                    if expected_line:
                                        if len(expected_line) > 100:
                                            expected_line = expected_line[:97] + '...'
                                        if len(actual_line) > 100:
                                            actual_line = actual_line[:97] + '...'
                                        hint = f"Context mismatch at line ~{conflicting_line}. Expected: '{expected_line}' but found: '{actual_line}'"
                                    else:
                                        start = max(0, conflicting_line - 3)
                                        end = min(len(file_lines_list), conflicting_line + 2)
                                        ctx = '\n'.join(f"{i+1}: {file_lines_list[i]}" for i in range(start, end))
                                        hint = f"Patch failed at line {conflicting_line}. File context around line:\n{ctx}"
                    except Exception as e:
                        logger.debug(f"Failed to read file {file_path} for patch analysis: {e}")
        elif "no such file" in error_lower or "cannot stat" in error_lower:
            reason = "file_not_found"
            hint = f"Target file not found: {file_path or 'unknown'}"

        file_context_snippet: Optional[str] = None
        if reason == "context_mismatch" and file_path and hunks:
            try:
                file_full_path = Path(self.repo_root) / file_path
                if file_full_path.exists():
                    file_lines_list = file_full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    hunk_start = hunks[0]["old_start"]
                    ctx_start = max(0, hunk_start - 5)
                    ctx_end = min(len(file_lines_list), hunk_start + hunks[0].get("old_lines", 10) + 5)
                    ctx_lines = [
                        f"{ctx_start + j + 1:4d}: {file_lines_list[ctx_start + j]}"
                        for j in range(ctx_end - ctx_start)
                    ]
                    file_context_snippet = "\n".join(ctx_lines)
            except (IndexError, TypeError):
                pass

        parts = [f"Patch failed ({reason}): {hint or git_error.strip()[:200]}"]

        if reason in ["context_mismatch", "offset_error", "unknown"]:
            parts.append("\n**Patch guidance**:")
            parts.append("- Provide exact context lines from the file (copy-paste, don't paraphrase)")
            parts.append("- For simple changes, use format: `-old line\\n+new line`")
            parts.append("\n**Example of correct unified diff format**:")
            parts.append("```diff")
            parts.append("diff --git a/path/to/file.py b/path/to/file.py")
            parts.append("--- a/path/to/file.py")
            parts.append("+++ b/path/to/file.py")
            parts.append("@@ -10,7 +10,7 @@")
            parts.append(" def old_function():")
            parts.append("     print('old')")
            parts.append("-    return 1")
            parts.append("+    return 2")
            parts.append("```")

            if "```" in patch_text and "diff --git" not in patch_text:
                parts.append("\n**Detected issue**: Your patch contains markdown code fences but not proper diff format.")
                parts.append("**Try this instead**: Remove the ``` markers and use unified diff format above.")

            if "before:" in patch_text.lower() or "after:" in patch_text.lower():
                parts.append("\n**Detected issue**: Your patch uses 'before:/after:' notation.")
                parts.append("**Try this instead**: Convert to unified diff format with exact context lines.")

        if file_context_snippet:
            parts.append(
                f"\n**Actual file content at patch location** (copy exact text for retry):\n"
                f"```\n{file_context_snippet}\n```"
            )
        elif reason == "file_not_found":
            parts = [
                f"Patch failed: {hint}. "
                f"For new files use the 'create_file' tool, or start the patch with "
                f"'--- /dev/null' to indicate new file creation."
            ]

        if not patch_text.strip().startswith("diff --git") and not patch_text.strip().startswith("---"):
            parts.append("\n**Patch format issue**: Your patch doesn't start with standard diff headers.")
            parts.append("**Try this**: Start with `diff --git a/file/path b/file/path` or `--- a/file/path`")

        error_message = "\n".join(parts)

        result = {
            "reason": reason,
            "hint": hint,
            "conflicting_lines": conflicting_lines,
            "error_message": error_message,
            "file_path": file_path,
            "hunk_count": len(hunks),
            "file_context_snippet": file_context_snippet,
        }

        if failure_ctx:
            result["failure_context"] = {
                "stage": failure_ctx.stage,
                "type": failure_ctx.type,
                "message": failure_ctx.message,
                "tags": failure_ctx.tags,
                "fingerprint": failure_ctx.fingerprint,
            }

        return result

    def _check_patch_content_ratio(self, patch_text: str) -> Optional[str]:
        warnings_out: list[str] = []
        current_file: Optional[str] = None
        removals: dict[str, int] = {}
        additions: dict[str, int] = {}

        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                    removals[current_file] = 0
                    additions[current_file] = 0
            elif current_file:
                if line.startswith("-") and not line.startswith("---"):
                    removals[current_file] = removals.get(current_file, 0) + 1
                elif line.startswith("+") and not line.startswith("+++"):
                    additions[current_file] = additions.get(current_file, 0) + 1

        for fpath, removed in removals.items():
            added = additions.get(fpath, 0)
            if removed < 10:
                continue
            try:
                abs_fp = Path(self.repo_root) / fpath
                if not abs_fp.is_file():
                    continue
                total = len(abs_fp.read_text(encoding="utf-8", errors="replace").splitlines())
                if total < 20:
                    continue
                ratio = removed / total
                if ratio > 0.7 and added < removed * 0.3:
                    warnings_out.append(
                        f"CONTENT LOSS WARNING: {fpath} — removing {removed}/{total} lines "
                        f"({ratio:.0%}) but only adding {added}. "
                        "Verify this is intentional (not an accidental wipe)."
                    )
            except (ValueError, AttributeError):
                pass

        return "\n".join(warnings_out) if warnings_out else None

    def _tool_modify_symbol(self, args: dict[str, Any]) -> "ToolResult":
        """Modify a symbol in a file deterministically — no LLM call.

        Fallback chain: AST precise (Python) → surgical edit (any language) → text replacement.
        Each fallback is deterministic — no Developer LLM calls.
        """
        from external_llm.agent.symbol_modify_tool import modify_symbol as _do_modify

        args = self._recover_args_from_raw(args, ("file_path", "symbol", "code"))
        file_path = str(args.get("file_path", "")).strip()
        symbol = str(args.get("symbol", "")).strip()
        code = str(args.get("code", "")).strip("\n")
        dry_run = args.get("dry_run", False)

        if not file_path:
            return self._make_result(ok=False, content="", error="'file_path' is required")
        if not symbol:
            return self._make_result(ok=False, content="", error="'symbol' is required")
        if not code:
            return self._make_result(ok=False, content="", error="'code' is required")

        sec = self._secure_path(file_path)
        if sec is None:
            return self._make_result(ok=False, content="", error=f"Path traversal blocked: {file_path}")
        abs_path = str(sec)
        if not os.path.isfile(abs_path):
                return self._make_result(ok=False, content="", error=f"File not found: {file_path}")

        rel_path = os.path.relpath(abs_path, self.repo_root)

        # dry_run snapshot: _do_modify writes the file on every success path,
        # so a preview REQUIRES a snapshot to restore from. Refuse the dry run
        # if the snapshot cannot be taken — otherwise the "preview" would
        # silently mutate the file.
        original_source: Optional[str] = None
        if dry_run:
            try:
                with open(abs_path, encoding='utf-8') as f:
                    original_source = f.read()
            except Exception as e:
                return self._make_result(
                    ok=False, content="",
                    error=f"[DRY RUN] cannot snapshot {rel_path} for preview: {e}",
                )

        success, diff_or_error, new_content = _do_modify(abs_path, symbol, code, repo_root=self.repo_root)

        if dry_run:
            # Restore the pre-edit content — _do_modify already wrote the file.
            if success and original_source is not None:
                try:
                    atomic_write_text(abs_path, original_source)
                except Exception as e:
                    return self._make_result(
                        ok=False, content="",
                        error=(
                            f"[DRY RUN] modify succeeded but restoring {rel_path} failed: {e} "
                            f"— the file HAS BEEN MODIFIED on disk"
                        ),
                        metadata={"file_path": rel_path, "symbol": symbol, "dry_run": True,
                                  "restore_failed": True},
                    )
            if success:
                return self._make_result(
                    ok=True,
                    content=(
                        f"[DRY RUN] modify_symbol preview for {rel_path}@{symbol}\n"
                        f"Diff:\n{diff_or_error}"
                    ),
                    metadata={
                        "file_path": rel_path,
                        "symbol": symbol,
                        "dry_run": True,
                        "diff_preview": diff_or_error[:25000] if diff_or_error else "",
                    }
                )
            else:
                preview = f"[DRY RUN] Preview for {rel_path}@{symbol} (apply skipped: {diff_or_error})\n"
                preview += f"New code:\n{code}"
                return self._make_result(
                    ok=True,
                    content=preview,
                    metadata={"file_path": rel_path, "symbol": symbol, "dry_run": True, "preview_only": True}
                )

        if success:
            self._record_text_edit(rel_path)
            try:
                self._applied_patches.append(f"modify_symbol:{rel_path}:{symbol}")
            except (AttributeError, TypeError):
                pass
            _meta = {
                "file_path": rel_path,
                "symbol": symbol,
                "diff_preview": diff_or_error[:25000] if diff_or_error else "",
                "changed": True,
            }
            #── Enot/inside: surface the replaced symbol's definition indent ──
            # new_content is the post-edit file in memory (3rd tuple element of
            # _do_modify). Locate the symbol's def/decorator start line and
            # report its leading-whitespace column count, mirroring read_file's
            # │N│ gutter. This lets the LLM verify the replacement landed at the
            # intended nesting depth (esp. for body-only mode, where the LLM
            # must guess the body indent). Best-effort: any failure is swallowed.
            if new_content:
                try:
                    from external_llm.agent.symbol_modify_tool import _find_symbol_line_range as _find_range
                    _rng = _find_range(new_content, symbol, rel_path)
                    if _rng is not None:
                        _nc_lines = new_content.splitlines()
                        _def_idx = _rng[0]
                        if 0 <= _def_idx < len(_nc_lines):
                            _meta["symbol_def_line"] = _def_idx + 1
                            _meta["symbol_def_indent"] = _leading_indent_width(_nc_lines[_def_idx])
                except Exception:
                    pass
            # Semantic feedback (non-blocking): pyright/tsc/go diagnostics for
            # type/undefined-name/import issues. Mirrors apply_patch/edit_file.
            _syn = self._run_syntax_check_for_file(abs_path)
            if not _syn.get("skipped"):
                _meta["syntax_check"] = _syn
            return self._make_result(
                ok=True,
                content=(
                    f"Modified symbol '{symbol}' in {rel_path}\n"
                    f"Diff:\n{diff_or_error}"
                ),
                metadata=_meta,
            )
        else:
            return self._make_result(
                ok=False, content="",
                error=f"modify_symbol failed for {rel_path}@{symbol}: {diff_or_error}"
            )

    def _tool_edit_ast(self, args: dict[str, Any]) -> "ToolResult":
        """Apply typed AST operations to a Python file. Deterministic — no LLM call.

        Each op has a 'type' and type-specific parameters.
        Supported ops: replace_expr, add_import, add_guard, delete_stmt,
        add_class_field, remove_import_name, list_append, list_remove.

        Unlike apply_patch, AST operations are whitespace-agnostic and survive
        context drift — they operate on the AST, not on raw text lines.
        """
        args = self._recover_args_from_raw(args, ("file_path",))
        file_path = str(args.get("file_path", "")).strip()
        ops_raw = args.get("ops")
        dry_run = args.get("dry_run", False)

        if not file_path:
            # If __raw_arguments is present, the JSON was likely truncated during streaming
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(
                ok=False, content="",
                error=f"'file_path' is required{_raw_hint}"
            )
        if not ops_raw:
            return self._make_result(
                ok=False, content="",
                error="'ops' is required"
            )

        if not isinstance(ops_raw, list) or not ops_raw:
            return self._make_result(
                ok=False, content="",
                error="'ops' must be a non-empty list of operation dicts"
            )

        # Resolve file path
        sec = self._secure_path(file_path)
        if sec is None:
            return self._make_result(ok=False, content="", error=f"Path blocked (outside repo): {file_path}")
        abs_path = str(sec)
        if not os.path.isfile(abs_path):
            return self._make_result(ok=False, content="", error=f"File not found: {file_path}")

        # Normalize to relative for output
        rel_path = os.path.relpath(abs_path, self.repo_root)
        file_path = rel_path

        # Read the file — strict UTF-8 first, then latin-1 (lossless 1:1 byte
        # round-trip when written back with the same encoding). The previous
        # errors="replace" fallback baked U+FFFD into the whole file before
        # rewriting it as UTF-8.
        source = None
        _read_encoding = "utf-8"
        for _enc in ("utf-8", "latin-1"):
            try:
                with open(abs_path, encoding=_enc) as f:
                    source = f.read()
                _read_encoding = _enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except OSError:
                return self._make_result(
                    ok=False, content="", error=f"Failed to read {file_path}: OSError"
                )
        if source is None:
            return self._make_result(
                ok=False, content="", error=f"Failed to read {file_path}: unsupported encoding"
            )

        # Check language — Python only
        if LanguageId.from_path(file_path) is not LanguageId.PYTHON:
            return self._make_result(
                ok=False, content="",
                error=f"AST edit is only supported for Python files (not {file_path})"
            )

        try:
            # Parse AST and validate syntax before applying
            import ast as _ast
            _ast.parse(source, filename=file_path)
        except SyntaxError as e:
            return self._make_result(
                ok=False, content="",
                error=f"Syntax error in {file_path}: {e}"
            )

        # Apply AST operations
        from .ast_op_executor import ASTOpExecutor

        executor = ASTOpExecutor()
        symbol = str(args.get("symbol", "")).strip()

        # Normalize LLM-friendly field names to ASTOpExecutor's internal parameter names
        _FIELD_ALIASES: dict[str, dict[str, str]] = {
            "add_import": {"import_name": "import", "import_stmt": "import"},
            "replace_expr": {"target": "old", "old_expr": "old", "old_text": "old", "new_expr": "new", "new_text": "new"},
            "add_guard": {"guard": "statement", "condition": "statement", "guard_stmt": "statement"},
            "delete_stmt": {"text_pattern": "pattern", "pattern_text": "pattern", "match": "pattern"},
            # NB: never alias the reserved op-discriminator key "type" here — it
            # would steal the op's own 'type' field. Use "annotation" for field_type.
            "add_class_field": {
                "class": "class_name", "cls": "class_name",
                "field": "field_name", "name": "field_name", "attr": "field_name",
                "annotation": "field_type",
                "default": "field_default", "value": "field_default",
            },
            "remove_import_name": {"import_name": "name", "symbol": "name"},
            "list_append": {"list": "list_name", "target": "list_name"},
            "list_remove": {"list": "list_name", "target": "list_name"},
        }

        ops_normalized: list[dict] = []
        for op in ops_raw:
            if isinstance(op, dict):
                normalized = dict(op)
                type_ = normalized.get("type", "")
                if not type_:
                    type_ = normalized.pop("op", None) or normalized.pop("action", "") or ""
                normalized["type"] = type_
                # Apply field name aliases for this op type
                aliases = _FIELD_ALIASES.get(type_, {})
                for alias, canonical in aliases.items():
                    if alias in normalized and canonical not in normalized:
                        normalized[canonical] = normalized.pop(alias)
                ops_normalized.append(normalized)

        result = executor.apply(source, ops_normalized, symbol=symbol)

        if not result.success:
            failed_str = "; ".join(result.ops_failed) if result.ops_failed else "unknown"
            _hint = self._ast_fail_hint(source, ops_normalized, symbol)
            error_msg = (
                f"AST edit failed in {file_path}@{symbol or '(module)'}: "
                f"{failed_str}{_hint}"
            )
            return self._make_result(
                ok=False, content="", error=error_msg,
                metadata={"near_match": bool(_hint)},
            )

        if not result.changed:
            return self._make_result(
                ok=True,
                content=f"AST edit: no changes needed (all {result.ops_applied} ops were idempotent)",
                metadata={
                    "file_path": file_path,
                    "ops_applied": result.ops_applied,
                    "changed": False,
                }
            )

        new_source = result.new_source

        # Final semantic validation with compile()
        try:
            compile_quiet(new_source, file_path, "exec")
        except SyntaxError as e:
            return self._make_result(
                ok=False, content="",
                error=f"AST edit produced invalid syntax in {file_path}: {e}"
            )

        # Generate diff for preview
        import difflib
        diff_lines = list(difflib.unified_diff(
            source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))
        diff_text = "".join(diff_lines)

        if dry_run:
            return self._make_result(
                ok=True,
                content=(
                    f"[DRY RUN] AST edit preview for {file_path}\n"
                    f"Ops: {len(ops_normalized)} ({result.ops_applied} applied)\n"
                    f"Diff ({len(diff_lines)} lines):\n"
                    f"{diff_text}"
                ),
                metadata={
                    "file_path": file_path,
                    "ops_applied": result.ops_applied,
                    "ops_total": len(ops_normalized),
                    "diff_preview": diff_text[:25000],
                    "changed": True,
                    "dry_run": True,
                }
            )

        # Write the file — same encoding it was read with (see read fallback
        # above). Encode before opening so an encode failure can't truncate.
        try:
            _encoded_source = new_source.encode(_read_encoding)
            with open(abs_path, "wb") as f:
                f.write(_encoded_source)
        except (OSError, UnicodeEncodeError) as e:
            return self._make_result(
                ok=False, content="",
                error=f"Failed to write {file_path}: {e}"
            )

        self._record_text_edit(file_path)
        return self._make_result(
            ok=True,
            content=(
                f"AST edit applied to {file_path}\n"
                f"Ops: {result.ops_applied}/{len(ops_normalized)} applied, "
                f"{len(result.ops_failed)} failed\n"
                f"Diff ({len(diff_lines)} lines):\n"
                f"{diff_text}"
            ),
            metadata={
                "file_path": file_path,
                "ops_applied": result.ops_applied,
                "ops_failed": result.ops_failed if result.ops_failed else [],
                "diff_preview": diff_text[:25000],
                "changed": True,
                "symbol": symbol,
            }
        )

    # ═══════════════════════════════════════════════════════════════════════
    # anchor_edit — pattern-based sub-symbol insertion/deletion
    # ═══════════════════════════════════════════════════════════════════════

    def _tool_anchor_edit(self, args: dict[str, Any]) -> "ToolResult":
        """Pattern-based file editing for precise sub-symbol insertion/deletion.

        Uses anchor_pattern (substring-first, regex-fallback) to locate the
        target line.  Supports occurrence selection, context-before/after
        disambiguation, and fuzzy fallback.  Deterministic — no LLM call;
        the calling LLM provides code_snippet directly.
        """
        import time as _time
        start_time = _time.monotonic()

        args = self._recover_args_from_raw(args, ("file_path",))
        file_path = (args.get("file_path") or "").strip()
        anchor_pattern = (args.get("anchor_pattern") or "").strip()
        edit_mode = (args.get("edit_mode") or "insert_before").strip()
        code_snippet = (args.get("code_snippet") or "").strip()
        occurrence = args.get("occurrence", -1)
        context_before = (args.get("context_before") or "").strip() or None
        context_after = (args.get("context_after") or "").strip() or None
        # anchor_ast_lineno: caller-supplied 1-indexed line that bypasses string
        # search (mirrors editor path's 2a strategy). Optional — when present,
        # anchor_pattern becomes a readability hint and is not searched.
        anchor_ast_lineno = args.get("anchor_ast_lineno")

        # ── Validate required fields ──────────────────────────────────────
        if not file_path:
            return self._make_result(
                ok=False, content="", error="'file_path' is required", execution_time=0,
            )
        # anchor_pattern OR anchor_ast_lineno — exactly one locating strategy required.
        if not anchor_pattern and anchor_ast_lineno is None:
            return self._make_result(
                ok=False, content="",
                error="'anchor_pattern' or 'anchor_ast_lineno' is required (one of them)",
                execution_time=0,
            )
        if edit_mode not in ("insert_before", "insert_after", "replace_line", "delete"):
            return self._make_result(
                ok=False, content="",
                error=f"Invalid edit_mode: {edit_mode!r} (expected insert_before, insert_after, replace_line, or delete)",
                execution_time=0,
            )
        if edit_mode != "delete" and not code_snippet:
            return self._make_result(
                ok=False, content="",
                error=f"'code_snippet' is required for edit_mode={edit_mode!r}",
                execution_time=0,
            )

        # ── Resolve file path ──────────────────────────────────────────────
        _norm = Path(self.repo_root) / file_path
        if not _norm.exists():
            return self._make_result(
                ok=False, content="", error=f"File not found: {file_path}", execution_time=0,
            )

        try:
            original = _norm.read_text(encoding="utf-8")
        except Exception as e:
            return self._make_result(
                ok=False, content="", error=f"Failed to read {file_path}: {e}", execution_time=0,
            )

        lines = original.splitlines(True)
        lang_id = LanguageId.from_path(str(_norm))

        # ── Anchor matching — import shared helpers ────────────────────────
        from external_llm.agent.anchor_shared import (
            _find_anchor_line,
            _fuzzy_find_anchor_line,
            _inherit_anchor_indent_if_bare,
            _match_anchor,
            resolve_multiline_anchor,
        )
        from external_llm.common.indent_utils import detect_indent_char, indent_unit
        # Destination file's chars-per-level — so a tab/4-space snippet rebased
        # into this file maps each level to the file's real unit, not hardcoded 4.
        _wt_dest_unit = indent_unit(original, detect_indent_char(lines))
        from external_llm.agent.operation_models import FailureClass as _FC

        # ── anchor_ast_lineno: direct line bypasses string search ──────────
        # Mirrors the editor path's 2a strategy (symbol_handlers_anchor.py:452).
        # When the caller supplies an exact 1-indexed line (e.g. right after a
        # read_file/read_symbol), skip the string/regex/fuzzy search entirely —
        # eliminates anchor_miss and anchor_not_unique failures. Falls back to
        # the string path if the number is out of range (stale after a prior edit).
        _ast_anchor = None  # 0-indexed anchor line, or None
        if isinstance(anchor_ast_lineno, int) and anchor_ast_lineno > 0:
            _candidate = anchor_ast_lineno - 1  # 1-indexed → 0-indexed
            if 0 <= _candidate < len(lines):
                _ast_anchor = _candidate
                logger.info(
                    "[ANCHOR_AST] tool anchor_edit using caller-supplied line %d "
                    "(pattern=%r bypassed)",
                    anchor_ast_lineno, (anchor_pattern or "")[:40],
                )
            else:
                logger.warning(
                    "[ANCHOR_AST] tool anchor_edit line %d out of range "
                    "(file=%d lines) — falling back to string search",
                    anchor_ast_lineno, len(lines),
                )

        # ── Multiline anchor: resolve to a block range instead of rejecting ─
        # A '\n'-containing anchor_pattern (the recurring failure mode where
        # the LLM concatenates several lines) is now *resolved*: the first
        # non-empty line locates the anchor, and every subsequent non-empty
        # line must strip-match the corresponding file line. This turns the
        # previous fail-fast rejection into a fail-tolerant auto-resolve,
        # eliminating the read→retry debug loop. The resolved inclusive range
        # is consumed below by delete / insert / replace per their semantics.
        # Skipped when _ast_anchor is set (lineno is authoritative).
        _multiline_range = None  # (anchor, end) when resolved; else single-line
        if _ast_anchor is None and "\n" in (anchor_pattern or ""):
            _ml = resolve_multiline_anchor(
                lines, anchor_pattern, occurrence,
                ctx_before=context_before, ctx_after=context_after,
            )
            if not _ml["ok"]:
                return self._make_result(
                    ok=False, content="", error=_ml["error"],
                    metadata={
                        "file_path": file_path,
                        "failure_class": _ml.get("failure_class") or _FC.ANCHOR_MULTILINE_PATTERN.value,
                    },
                )
            _multiline_range = (_ml["anchor"], _ml["end"])

        # ── Delete mode: deterministic, no code_snippet needed ─────────────
        if edit_mode == "delete":
            if _multiline_range is not None:
                # Multiline path: range already resolved + verified by
                # resolve_multiline_anchor() above. No fuzzy, no re-verify.
                _del_anchor, _del_end_inclusive = _multiline_range
                _del_count = _del_end_inclusive - _del_anchor + 1
                _del_search_pat = anchor_pattern.split("\n", 1)[0]
                _del_fuzzy_match = False
            else:
                _del_lines_raw = anchor_pattern.split("\n") if anchor_pattern else []
                _del_pat_lines = [pl for pl in _del_lines_raw if pl.strip()]
                if _ast_anchor is not None:
                    # AST lineno path — line is authoritative; pattern (if any)
                    # only supplies the line count for multi-line delete. The
                    # verify-all-pattern-lines block below still runs for count>1.
                    _del_search_pat = (
                        _del_pat_lines[0] if _del_pat_lines else lines[_ast_anchor].strip()
                    )
                    _del_count = len(_del_pat_lines) if _del_pat_lines else 1
                    _del_anchor = _ast_anchor
                    _del_fuzzy_match = False
                else:
                    if not _del_pat_lines:
                        return self._make_result(
                            ok=False, content="",
                            error="anchor_edit(delete): empty anchor pattern",
                        )
                    _del_search_pat = _del_pat_lines[0]
                    _del_count = len(_del_pat_lines)

                    _del_anchor = _find_anchor_line(
                        lines, _del_search_pat, occurrence,
                        ctx_before=context_before, ctx_after=context_after,
                    )

                # Fuzzy fallback — conservative (margin gate, no indent gate for delete)
                _del_fuzzy_match = False
                if _del_anchor is None:
                    _fz_lineno, _fz_score = _fuzzy_find_anchor_line(
                        lines, _del_search_pat, snippet_lines=None, edit_mode="delete",
                    )
                    if _fz_lineno is not None:
                        _del_anchor = _fz_lineno
                        _del_fuzzy_match = True

                _del_fuzzy_match = _del_fuzzy_match if _del_anchor is not None else False

                if _del_anchor is None:
                    return self._make_result(
                        ok=False, content="",
                        error=(
                            f"anchor_edit(delete): pattern {_del_search_pat!r} not found "
                            f"in {file_path} (searched {len(lines)} lines)"
                        ),
                    )

                # ── Verify ALL pattern lines match before deleting ──────────
                _del_mismatch = False
                for _pi in range(1, _del_count):
                    _file_lineno = _del_anchor + _pi
                    if _file_lineno >= len(lines):
                        # Pattern extends beyond file end — only an issue if pattern line is non-empty
                        if _del_pat_lines[_pi].strip():
                            _del_mismatch = True
                        break
                    _pat_stripped = _del_pat_lines[_pi].strip()
                    _file_stripped = lines[_file_lineno].strip()
                    if _pat_stripped and _pat_stripped not in _file_stripped:
                        _del_mismatch = True
                        break
                if _del_mismatch:
                    _del_end_mismatch = min(_del_anchor + _del_count, len(lines))
                    _actual_lines = "".join(lines[_del_anchor:_del_end_mismatch])[:500]
                    return self._make_result(
                        ok=False, content="",
                        error=(
                            f"anchor_edit(delete): pattern line {_pi + 1} mismatch after anchor "
                            f"at line {_del_anchor + 1} in {file_path}. "
                            f"The remaining {_del_count - 1} pattern line(s) do not match file content. "
                            f"Read the file and provide the exact text to delete."
                        ),
                        metadata={
                            "file_path": file_path,
                            "failure_class": "delete_mismatch",
                        },
                    )
                _del_end_inclusive = min(_del_anchor + _del_count - 1, len(lines) - 1)

            _del_end = _del_end_inclusive + 1
            _deleted_text = "".join(lines[_del_anchor:_del_end])
            lines = lines[:_del_anchor] + lines[_del_end:]
            new_content = "".join(lines)

            if new_content == original:
                return self._make_result(
                  ok=True,
                  content="The content is already as requested — nothing to delete (already_equal)",
                  error="",
                )

            # Syntax validation
            from ...languages.syntax_validator import SyntaxValidator
            _sv = SyntaxValidator.validate_syntax(new_content, lang_id)
            if not _sv.ok:
                _sv_err_msg = _sv.errors[0].message if _sv.errors else "unknown"
                return self._make_result(
                    ok=False, content="",
                    error=f"anchor_edit(delete) produced invalid syntax: {_sv_err_msg}",
                    metadata={"file_path": file_path, "failure_class": "syntax_invalid_after_edit"},
                )

            _norm.write_text(new_content, encoding="utf-8")
            self._record_text_edit(file_path)
            _exec_time = _time.monotonic() - start_time

            _anchor_meta = {
                "file_path": file_path,
                "mode": "delete",
                "deleted_lines": _del_count,
                "anchor_line": _del_anchor + 1,
                "deleted_text": _deleted_text[:2000],
                "execution_time": _exec_time,
                "fuzzy_match": _del_fuzzy_match,
            }
            _syn = self._run_syntax_check_for_file(str(_norm))
            if not _syn.get("skipped"):
                _anchor_meta["syntax_check"] = _syn
            logger.info(
                "anchor_edit(delete): removed %d lines (L%d-L%d) matching %r from %s",
                _del_count, _del_anchor + 1, _del_end, _del_search_pat[:60], file_path,
            )
            return self._make_result(
                ok=True,
                content=f"Deleted {_del_count} line(s) from {file_path} (lines {_del_anchor + 1}-{_del_end})",
                metadata=_anchor_meta,
            )

        # ── Find anchor for insert/replace modes ──────────────────────────
        # When the pattern was multiline, the inclusive range was already
        # resolved + verified above (_multiline_range). Use the block's END for
        # insert_after (insert past the block) and its START for insert_before
        # / replace_line semantics — see the edit-mode branches below.
        _fuzzy_match = False  # set True only via fuzzy path below (multiline: always False)
        _anchor_end = None  # inclusive end of the matched block (0-indexed)
        if _ast_anchor is not None:
            # AST lineno path — authoritative line; bypass string/regex/fuzzy.
            anchor_lineno = _ast_anchor
        elif _multiline_range is not None:
            anchor_lineno, _anchor_end = _multiline_range
        else:
            anchor_lineno = _find_anchor_line(
                lines, anchor_pattern, occurrence,
                ctx_before=context_before, ctx_after=context_after,
            )

            # ── Too-many-matches guard (mirrors editor path's ANCHOR_MAX_MATCHES) ──
            # When occurrence=-1 (default) and no context hints, _find_anchor_line
            # silently picks the LAST match even if the pattern matches many lines,
            # leading to wrong-target edits (e.g. inserting inside the wrong
            # try/except block). Fail loudly instead so the caller supplies
            # `occurrence` or `context_before`/`context_after` to disambiguate.
            # This matches edit_text's "old_string must be UNIQUE" contract.
            if (
                anchor_lineno is not None
                and occurrence in (-1, None)
                and not context_before
                and not context_after
            ):
                _match_count = sum(1 for _item_ in lines if _match_anchor(anchor_pattern, _item_))
                if _match_count > 1:
                    return self._make_result(
                        ok=False, content="",
                        error=(
                            f"anchor_pattern {anchor_pattern!r} matched {_match_count} "
                            f"times in {file_path}. The default occurrence=-1 (last match) "
                            f"is ambiguous with multiple matches. Specify `occurrence` "
                            f"(1=first, 2=second, ...) or context_before/context_after to "
                            f"disambiguate."
                        ),
                        metadata={
                            "file_path": file_path,
                            "failure_class": "anchor_not_unique",
                            "match_count": _match_count,
                        },
                    )

            # Fuzzy fallback — conservative (margin gate + indent compatibility gate)
            if anchor_lineno is None:
                _fz_lineno, _fz_score = _fuzzy_find_anchor_line(
                    lines, anchor_pattern,
                    snippet_lines=code_snippet.splitlines() if code_snippet else None,
                    edit_mode=edit_mode,
                )
                if _fz_lineno is not None:
                    _fuzzy_match = True
                    anchor_lineno = _fz_lineno

        if anchor_lineno is None:
            return self._make_result(
                ok=False, content="",
                error=(
                    f"anchor_pattern {anchor_pattern!r} not found in {file_path} "
                    f"(searched {len(lines)} lines) — read the file first and use exact text"
                ),
                metadata={"file_path": file_path, "failure_class": "anchor_miss"},
            )

        # ── Compute anchor indent ──────────────────────────────────────────
        anchor_line_text = lines[anchor_lineno].rstrip('\n\r')
        anchor_indent = len(lines[anchor_lineno]) - len(lines[anchor_lineno].lstrip())
        # Track indent correction for structural feedback metadata. Populated
        # by the Python block-introducer correction below (insert/replace path);
        # stays None for delete mode or non-Python files.
        _indent_correction_info = None

        # ── Collection-literal indentation fix ─────────────────────────────
        if (
            edit_mode == "insert_before"
            and anchor_line_text.strip() in ('}', '};', '},', '})', '});')
        ):
            _entry_indent = None
            _brace_depth = 0
            for _bi in range(anchor_lineno - 1, max(anchor_lineno - 200, -1), -1):
                _bl_stripped = lines[_bi].strip()
                for _ch in lines[_bi]:
                    if _ch == '}':
                        _brace_depth += 1
                    elif _ch == '{':
                        _brace_depth -= 1
                if _brace_depth < 0:
                    break
                if (
                    _bl_stripped
                    and not _bl_stripped.startswith(('//', '#', '/*', '*'))
                    and _bl_stripped not in ('{', '}', '};', '},', '})', '});')
                ):
                    _detected = len(lines[_bi]) - len(lines[_bi].lstrip())
                    if _detected > anchor_indent:
                        _entry_indent = _detected
                        break
            if _entry_indent is not None:
                anchor_indent = _entry_indent

        # ── Apply the edit ─────────────────────────────────────────────────
        new_code = code_snippet
        orig_content = original
        # (insert_start_line, insert_end_line) of a block-introducer snippet,
        # populated only for insert_before/insert_after on Python — feeds the
        # post-insert AST nesting gate below.
        _introducer_insert_range = None

        if edit_mode == "replace_line":
            # Multiline anchor: replace the WHOLE matched block [anchor, end]
            # with the snippet. Single-line path (the common case) retains the
            # bracket-balance guard below.
            if _anchor_end is not None:
                _old_lines = lines[anchor_lineno:_anchor_end + 1]
                _replace_block = new_code.splitlines(True)
                if _replace_block:
                    _adj_block = _inherit_anchor_indent_if_bare(
                        _replace_block, anchor_line_text,
                        _wt_dest_unit,
                    )
                    _block_text = "".join(_adj_block)
                    if not _block_text.endswith("\n"):
                        _block_text += "\n"
                else:
                    _block_text = "\n"
                lines[anchor_lineno:_anchor_end + 1] = [_block_text]
            else:
                _old_line = lines[anchor_lineno]
                # Indent bare snippet to anchor depth — strip() at L3125 already
                # removed all leading whitespace, so snippet is always "bare".
                _replace_lines = new_code.splitlines(True)
                if _replace_lines:
                    _adj_lines = _inherit_anchor_indent_if_bare(
                        _replace_lines, anchor_line_text,
                        _wt_dest_unit,
                    )
                    _new_line = "".join(_adj_lines)
                    if not _new_line.endswith("\n"):
                        _new_line += "\n"
                else:
                    _new_line = "\n"

                # Bracket-balance guard (single-line replace only)
                def _bracket_delta(_s: str) -> int:
                    """Count bracket delta ignoring string/comment content."""
                    _delta = 0
                    _in_str = None
                    _in_triple = False
                    _j = 0
                    while _j < len(_s):
                        _ch = _s[_j]
                        if _in_str is not None:
                            if _in_triple and _s[_j:_j+3] == _in_str * 3:
                                _in_str = None
                                _in_triple = False
                                _j += 3
                                continue
                            elif not _in_triple and _ch == _in_str and (_j == 0 or _s[_j-1] != '\\'):
                                _in_str = None
                        else:
                            if _ch in ('"', "'", '`'):
                                if _j + 2 < len(_s) and _s[_j:_j+3] == _ch * 3:
                                    _in_str = _ch
                                    _in_triple = True
                                    _j += 3
                                    continue
                                _in_str = _ch
                            elif _ch == '#':
                                break  # rest of line is comment
                            elif _ch == '{':
                                _delta += 1
                            elif _ch == '}':
                                _delta -= 1
                            elif _ch == '(':
                                _delta += 1
                            elif _ch == ')':
                                _delta -= 1
                            elif _ch == '[':
                                _delta += 1
                            elif _ch == ']':
                                _delta -= 1
                        _j += 1
                    return _delta

                _old_delta = _bracket_delta(_old_line)
                _new_delta = _bracket_delta(_new_line)

                if _old_delta != _new_delta:
                    # Guard: snippet starts with '}' → continuation fragment
                    if _new_line.strip().startswith("}"):
                        return self._make_result(
                            ok=False, content="",
                            error=(
                                f"anchor_edit(replace_line): snippet starts with '}}' "
                                f"at {file_path}:{anchor_lineno + 1} — continuation fragment "
                                f"cannot replace a top-level construct."
                            ),
                            metadata={"file_path": file_path, "failure_class": "structural_gate_violation"},
                        )

                    # Attempt bracket-balance expansion
                    _needed_balance = _new_delta
                    _scan_balance = _old_delta
                    _close_line = None
                    _in_str = None
                    _in_triple = False
                    for _scan_i in range(anchor_lineno + 1, min(len(lines), anchor_lineno + 500)):
                        _sl = lines[_scan_i]
                        _j = 0
                        while _j < len(_sl):
                            _ch = _sl[_j]
                            if _in_str is not None:
                                if _in_triple and _sl[_j:_j+3] == _in_str * 3:
                                    _in_str = None
                                    _in_triple = False
                                    _j += 3
                                    continue
                                elif not _in_triple and _ch == _in_str and (_j == 0 or _sl[_j-1] != '\\'):
                                    _in_str = None
                            else:
                                if _ch in ('"', "'", '`'):
                                    if _j + 2 < len(_sl) and _sl[_j:_j+3] == _ch * 3:
                                        _in_str = _ch
                                        _in_triple = True
                                        _j += 3
                                        continue
                                    _in_str = _ch
                                elif _ch == '{':
                                    _scan_balance += 1
                                elif _ch == '}':
                                    _scan_balance -= 1
                                elif _ch == '(':
                                    _scan_balance += 1
                                elif _ch == ')':
                                    _scan_balance -= 1
                                elif _ch == '[':
                                    _scan_balance += 1
                                elif _ch == ']':
                                    _scan_balance -= 1
                            _j += 1
                        if _scan_balance == _needed_balance:
                            _close_line = _scan_i
                            break

                    if _close_line is not None:
                        logger.warning(
                            "anchor_edit(replace_line): bracket delta mismatch — "
                            "expanding replace from 1 line to %d lines",
                            _close_line - anchor_lineno + 1,
                        )
                        lines[anchor_lineno] = _new_line
                        del lines[anchor_lineno + 1:_close_line + 1]
                    else:
                        return self._make_result(
                            ok=False, content="",
                            error=(
                                f"anchor_edit(replace_line): bracket imbalance "
                                f"(old={_old_delta:+d}, new={_new_delta:+d}) at "
                                f"{file_path}:{anchor_lineno + 1} — cannot safely replace"
                            ),
                            metadata={"file_path": file_path, "failure_class": "structural_gate_violation"},
                        )
                else:
                    lines[anchor_lineno] = _new_line

        else:
            # insert_before or insert_after
            if edit_mode == "insert_before":
                insert_idx = anchor_lineno
            else:  # insert_after
                # Multiline anchor: insert past END of matched block (semantics A).
                # def/class multi-line signature skip is single-line-anchor only —
                # a multiline block is already fully resolved so no skip is needed.
                if _anchor_end is not None:
                    insert_idx = _anchor_end + 1
                else:
                    insert_idx = anchor_lineno + 1

                    # ── Block-end auto-correction ────────────────────────────
                    # If the anchor line is a block HEADER (def/class/if/... in
                    # Python, a '{'-opening line in brace languages), inserting
                    # at lineno+1 lands INSIDE the body — the classic
                    # "insert_after on a def/{ line nests the snippet" bug.
                    # Find the block's real END (language-agnostic, tree-sitter
                    # first with brace/indent fallbacks) and insert past it so
                    # the new construct becomes a sibling. Installing a grammar
                    # enables the correction for that language with no code
                    # change. Replaces the old def-skip logic that only scanned
                    # to the signature colon (which still landed in the body).
                    _block_end = _find_block_end_line(
                        original, lang_id.value, anchor_lineno, lines,
                    )
                    if _block_end is not None and _block_end > anchor_lineno:
                        logger.info(
                            "anchor_edit(insert_after): anchor L%d is a %d-line "
                            "block header — inserting after block end L%d "
                            "instead of into the body",
                            anchor_lineno + 1,
                            _block_end - anchor_lineno + 1,
                            _block_end + 1,
                        )
                        insert_idx = _block_end + 1
                        # anchor_indent already reflects the header's own indent,
                        # so the new construct is placed as a sibling at the same
                        # depth. (The old def-skip path bumped indent to the BODY
                        # level, which is what caused the nesting bug.)

            # ── Python block-introducer indent correction ────────────────
            # When the snippet STARTS a new def/class (a block introducer) but
            # the anchor matched a *body* line (deeper than its enclosing
            # header), blindly inheriting anchor_indent would land the new
            # block as a nested function/class — a silent structural bug.
            # Re-derive anchor_indent from the nearest enclosing def/class
            # header so the new block is inserted as a sibling instead.
            # (Indentation has no structural meaning in brace-languages, so
            # this correction is Python-only — see system prompt rule 7.)
            if lang_id is LanguageId.PYTHON:
                # Detect block-introducer using the snippet's MINIMUM-indent
                # line (not the first non-empty line). When the LLM prepends a
                # fragment of existing code (e.g. "    finally:\n
                # rec.cleanup()\n") before the new block, the first non-empty
                # line is the fragment — not the introducer — so the old
                # "_snip_first_line" check silently failed to correct, landing
                # the new def/class at the anchor's deep indent. Using the
                # min-indent line reliably finds the new top-level construct.
                _snip_lines_for_intro = new_code.splitlines()
                _snip_min_indent = None
                for _ln in _snip_lines_for_intro:
                    if _ln.strip():
                        _ind = len(_ln) - len(_ln.lstrip())
                        if _snip_min_indent is None or _ind < _snip_min_indent:
                            _snip_min_indent = _ind
                # Check ALL min-indent lines, not just the first one found. A
                # comment banner, module-level constant, or decorator at the
                # same min-indent as the def/class it introduces would
                # otherwise be picked as "the" introducer line, fail the
                # startswith check, and silently skip this correction —
                # exactly the gap that let a cache-section snippet (banner +
                # constants + defs) inherit a nested anchor's indent.
                _snip_is_block_introducer = any(
                    _item_.strip() and (len(_item_) - len(_item_.lstrip())) == _snip_min_indent
                    and _item_.lstrip().startswith(("def ", "async def ", "class ", "@"))
                    for _item_ in _snip_lines_for_intro
                ) if _snip_min_indent is not None else False
                # Only correct when the anchor is NOT itself a def/class header:
                # a header anchor already sits at the right sibling level, and
                # lifting it further (e.g. first-method-of-class) would corrupt
                # the enclosing class body by de-indenting to the class level.
                _anchor_is_header = lines[anchor_lineno].strip().startswith(
                    ("def ", "async def ", "class ")
                )
                if _snip_is_block_introducer and not _anchor_is_header:
                    for _up in range(anchor_lineno - 1, -1, -1):
                        _up_text = lines[_up]
                        _up_stripped = _up_text.strip()
                        if _up_stripped.startswith(("def ", "async def ", "class ")):
                            _up_indent = len(_up_text) - len(_up_text.lstrip())
                            if _up_indent < anchor_indent:
                                _prev = anchor_indent
                                anchor_indent = _up_indent
                                logger.debug(
                                    "anchor_edit(indent-correct): snippet is a block "
                                    "introducer; re-anchored indent %d→%d (nearest enclosing header)",
                                    _prev, _up_indent,
                                )
                                _indent_correction_info = {
                                    "snippet_base_indent": _snip_min_indent if _snip_min_indent is not None else 0,
                                    "original_anchor_indent": _prev,
                                    "corrected_anchor_indent": _up_indent,
                                    "reason": "block_introducer_at_nested_anchor",
                                }
                            break

            # ── Fragment-duplication pre-guard (insert_before/insert_after) ──
            # If code_snippet duplicates existing code around the anchor, the
            # insert would land a dangling block that only fails the POST-write
            # syntax check with an opaque message. Detect it HERE (before
            # indentation normalization touches the snippet) so we can reject
            # with an actionable, file-preserving error. replace_line/delete
            # are exempt — they legitimately overlap existing code.
            _dup = _detect_fragment_duplication(lines, insert_idx, new_code)
            if _dup is not None:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"anchor_edit({edit_mode}): code_snippet duplicates "
                        f"{_dup['matched']}/{_dup['content_lines']} non-trivial "
                        f"lines already present near {file_path}:L{anchor_lineno + 1} "
                        f"(ratio {_dup['ratio']:.2f}). code_snippet must contain "
                        f"ONLY the new lines to insert — not a copy of the anchor "
                        f"or its surrounding context. Duplicated lines:\n"
                        f"{_dup['dup_lines']}"
                    ),
                    metadata={
                        "file_path": file_path,
                        "failure_class": "fragment_duplication",
                        "anchor_line": anchor_lineno + 1,
                        "mode": edit_mode,
                        "dup_ratio": _dup["ratio"],
                        "dup_content_lines": _dup["content_lines"],
                    },
                )

            # ── Indentation normalization ───────────────────────────────
            _llm_lines = new_code.splitlines()
            _min_indent = None
            for _ln in _llm_lines:
                if _ln.strip():
                    _ind = len(_ln) - len(_ln.lstrip())
                    if _min_indent is None or _ind < _min_indent:
                        _min_indent = _ind
            if _min_indent is None:
                _min_indent = 0
            indented_lines = []
            for ln in _llm_lines:
                if ln.strip():
                    current_indent = len(ln) - len(ln.lstrip())
                    rel_indent = current_indent - _min_indent
                    if rel_indent < 0:
                        rel_indent = 0
                    indented_lines.append(" " * (anchor_indent + rel_indent) + ln.lstrip())
                else:
                    indented_lines.append("")
            indented_code = "\n".join(indented_lines) + "\n"

            # ── Newline guard: last-line insert on non-\n-terminated files ──
            if insert_idx > 0 and insert_idx == len(lines) and not lines[-1].endswith('\n'):
                lines[-1] += '\n'

            lines.insert(insert_idx, indented_code)

            if lang_id is LanguageId.PYTHON and _snip_is_block_introducer:
                _introducer_insert_range = (insert_idx, insert_idx + len(indented_lines))

        new_content = "".join(lines)

        # ── Already-equal guard: no-op success ──────────────────────────────
        if new_content == orig_content:
            return self._make_result(
                ok=True,
                content="The content is already as requested — no change needed (already_equal)",
                error="",
                metadata={"file_path": file_path, "failure_class": "already_equal"},
            )

        # ── Structural gate: block-introducer nested inside a function ──────
        # Defense-in-depth AST check behind the text-based indent-correction
        # above. If the snippet introduces a def/class and it still ended up
        # lexically inside a FunctionDef/AsyncFunctionDef body, reject before
        # writing — a silent nesting bug that the syntax gate below cannot
        # catch (nested defs/constants are syntactically valid Python).
        if _introducer_insert_range is not None:
            _nest_err = _check_block_introducer_nesting(
                new_content, _introducer_insert_range[0], _introducer_insert_range[1]
            )
            if _nest_err is not None:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"anchor_edit({edit_mode}): {_nest_err} "
                        f"file={file_path}, anchor_line={anchor_lineno + 1}"
                    ),
                    metadata={
                        "file_path": file_path,
                        "failure_class": "structural_gate_violation",
                        "anchor_line": anchor_lineno + 1,
                        "mode": edit_mode,
                    },
                )

        # ── Syntax validation + write ──────────────────────────────────────
        from ...languages.syntax_validator import SyntaxValidator
        _sv = SyntaxValidator.validate_syntax(new_content, lang_id)
        if not _sv.ok:
            _sv_err_msg = _sv.errors[0].message if _sv.errors else "unknown"
            _sv_err_line = getattr(_sv.errors[0], "line", None) if _sv.errors else None
            # Build an actionable hint: show the region around the error and
            # the anchor context so the LLM can see WHY the edit broke syntax.
            # The most common cause of "invalid syntax" in insert mode is the
            # snippet accidentally including a copy of existing code (a
            # "fragment duplication"), which then gets re-indented to the
            # anchor level and produces a duplicate/dangling block.
            _hint_parts = [f"anchor_edit introduced syntax error (file unchanged): {_sv_err_msg}"]
            _hint_parts.append(f"file={file_path}, anchor_line={anchor_lineno + 1}")
            if _sv_err_line:
                _hint_parts.append(f"syntax_error_at_line={_sv_err_line}")
            if edit_mode in ("insert_before", "insert_after"):
                _hint_parts.append(
                    "Likely cause: code_snippet accidentally includes a copy of "
                    "existing code around the anchor (fragment duplication). The "
                    "snippet should contain ONLY the new code to insert, not the "
                    "anchor line or its surrounding context. Re-read the file, "
                    "then provide only the new lines in code_snippet."
                )
            _hint_parts.append(
                "If inserting a top-level construct (def/class) at file scope, "
                "prefer apply_patch (which uses exact line ranges) over anchor_edit."
            )
            return self._make_result(
                ok=False, content="",
                error=" ".join(_hint_parts),
                metadata={
                    "file_path": file_path,
                    "failure_class": "syntax_invalid_after_edit",
                    "anchor_line": anchor_lineno + 1,
                    "syntax_error_line": _sv_err_line,
                    "mode": edit_mode,
                },
            )

        _norm.write_text(new_content, encoding="utf-8")
        self._record_text_edit(file_path)
        _exec_time = _time.monotonic() - start_time

        _orig_lines = orig_content.splitlines()
        _mod_lines = new_content.splitlines()
        _delta = len(_mod_lines) - len(_orig_lines)

        _anchor_meta = {
            "file_path": file_path,
            "mode": edit_mode,
            "anchor_line": anchor_lineno + 1,
            "line_delta": _delta,
            "execution_time": _exec_time,
            "fuzzy_match": _fuzzy_match,
        }
        if _anchor_end is not None:
            _anchor_meta["anchor_end"] = _anchor_end + 1
            _anchor_meta["multiline_anchor"] = True
        # ── Structural feedback (Python only) ────────────────────────────────
        # Expose the indent the snippet was inserted at, the enclosing scope it
        # landed in, and (when applicable) whether anchor_indent was corrected
        # to lift a block-introducer snippet to its proper sibling level. This
        # lets the LLM self-verify the structural correctness of an insert
        # without a separate read_file round-trip (see _detect_enclosing_scope).
        if lang_id is LanguageId.PYTHON:
            try:
                _scope = _detect_enclosing_scope(orig_content.splitlines(), anchor_lineno)
                _tl = _scope.get("top_level")
                _il = _scope.get("innermost")
                if _tl and _tl[0] is not None:
                    _anchor_meta["enclosing_scope"] = {
                        "kind": _tl[0],
                        "name": _tl[1],
                        "indent": _tl[2],
                    }
                    if _il and _il[0] is not None and _il[1] != _tl[1]:
                        _anchor_meta["enclosing_scope"]["innermost"] = {
                            "kind": _il[0], "name": _il[1], "indent": _il[2],
                        }
                _anchor_meta["inserted_at_indent"] = (
                    _indent_correction_info["corrected_anchor_indent"]
                    if _indent_correction_info
                    else _scope.get("anchor_indent", anchor_indent)
                )
            except Exception:
                pass
            if _indent_correction_info is not None:
                _anchor_meta["indent_correction"] = _indent_correction_info
        # Semantic feedback (non-blocking): pyright/tsc/go diagnostics for
        # type/undefined-name/import issues. Mirrors apply_patch/edit_file.
        _syn = self._run_syntax_check_for_file(str(_norm))
        if not _syn.get("skipped"):
            _anchor_meta["syntax_check"] = _syn

        _line_desc = (
            f"lines {anchor_lineno + 1}-{_anchor_end + 1}"
            if _anchor_end is not None else f"line {anchor_lineno + 1}"
        )
        logger.info(
            "anchor_edit: %s in %s at %s (mode=%s)",
            edit_mode, file_path, _line_desc, edit_mode,
        )

        return self._make_result(
            ok=True,
            content=(
                f"anchor_edit ({'[fuzzy] ' if _fuzzy_match else ''}{edit_mode}) applied to {file_path} "
                f"at {_line_desc} (delta: {_delta:+d} lines)"
                + (" ⚠️ fuzzy match — verify result with read_file" if _fuzzy_match else "")
            ),
            metadata=_anchor_meta,
        )

    anchor_edit = _tool_anchor_edit  # alias for direct dispatch
