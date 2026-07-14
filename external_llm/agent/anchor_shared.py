"""Shared anchor-matching helpers used by both REPL and editor paths.

Extracted from editor/operation_handlers/symbol_handlers_anchor.py so that
the REPL (anchor_edit tool) can use these functions without pulling in the
entire editor/ subtree.  The editor path re-imports from here too, keeping
a single source of truth.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from external_llm.common.indent_utils import reindent_to_anchor

logger = logging.getLogger(__name__)


def _match_anchor(pattern: str, line: str) -> bool:
    """Return True if *pattern* matches *line*.
    Priority: literal substring (backward compatible) → regex re.search (supplement).
    """
    if not pattern or not line:
        return False
    # 1. Literal substring — 100% backward compatible (safe for foo(arg), return[x], etc.)
    if pattern in line:
        return True
    # 2. Regex fallback — enables disambiguation like ^}$, id="btn-raid"
    try:
        return re.search(pattern, line) is not None
    except re.error:
        return False


def _find_anchor_line(
    lines: "list[str]",
    pattern: str,
    occurrence: int = -1,
    ctx_before: "Optional[str]" = None,
    ctx_after: "Optional[str]" = None,
) -> "Optional[int]":
    """Find 0-indexed line matching *pattern* with occurrence semantics.

    occurrence > 0 → nth occurrence (1-indexed)
    occurrence == -1 → **LAST** occurrence (canonical default)

    ctx_before / ctx_after: if set, the preceding / following line must
    also pass ``_match_anchor`` (anchor disambiguation).

    Returns 0-indexed line index, or *None* if not found.
    """
    _last_match = None
    _count = 0
    for _i, _line in enumerate(lines):
        if not _match_anchor(pattern, _line):
            continue
        if ctx_before is not None and _i > 0:
            if not _match_anchor(ctx_before, lines[_i - 1]):
                continue
        if ctx_after is not None:
            if _i >= len(lines) - 1:
                continue  # last line: no following line to verify ctx_after
            if not _match_anchor(ctx_after, lines[_i + 1]):
                continue
        _count += 1
        _last_match = _i  # track last match for occurrence-overflow fallback
        if occurrence == -1:
            if _count == 2:  # warn only once, on second match
                logger.warning(
                    "[ANCHOR_NON_UNIQUE] pattern %r matched %d times "
                    "(occurrence=-1→last match). Specify anchor_occurrence, "
                    "ctx_before, or ctx_after for deterministic resolution.",
                    pattern, _count,
                )
        elif _count == occurrence:
            return _i
    # ── Occurrence-overflow fallback ────────────────────────────────────
    if occurrence > 0 and _last_match is not None:
        logger.warning(
            "[ANCHOR_OCCURRENCE_FALLBACK] pattern %r: requested occurrence=%d "
            "but only %d match(es) found — falling back to last match "
            "(line %d, 0-indexed).",
            pattern[:120], occurrence, _count, _last_match,
        )
        return _last_match
    if occurrence == -1 and _last_match is not None:
        return _last_match
    return None


def resolve_multiline_anchor(
    lines: "list[str]",
    pattern: str,
    occurrence: int = -1,
    ctx_before: "Optional[str]" = None,
    ctx_after: "Optional[str]" = None,
) -> "dict":
    """Resolve a ``\\n``-containing anchor_pattern to an inclusive line range.

    Enables ``insert_before`` / ``insert_after`` / ``replace_line`` to accept a
    multi-line block as the anchor (the recurring failure mode where the LLM
    concatenates several lines in ``anchor_pattern``). The FIRST non-empty
    pattern line locates the anchor via :func:`_find_anchor_line` (with
    occurrence + context disambiguation), and every subsequent non-empty
    pattern line must strip-match the corresponding file line. On success the
    inclusive ``[anchor, end]`` range (0-indexed) is returned.

    No fuzzy fallback: a block whose first line is not found exactly is
    rejected rather than guessed, because guessing a multi-line range is far
    more dangerous than guessing a single line.

    Returns a dict::

        ok=True  → {"ok": True,  "anchor": int, "end": int, "count": int}
        ok=False → {"ok": False, "error": str, "failure_class": str}

    Failure classes:
      - ``anchor_multiline_pattern`` : empty pattern (no non-empty lines)
      - ``anchor_miss``              : first line not found exactly
      - ``multiline_mismatch``       : a later pattern line ≠ file line / EOF
    """
    _pat_lines = [pl for pl in pattern.split("\n") if pl.strip()]
    if not _pat_lines:
        return {
            "ok": False,
            "error": "multiline anchor pattern is empty (no non-empty lines)",
            "failure_class": "anchor_multiline_pattern",
        }
    _search_pat = _pat_lines[0]
    _count = len(_pat_lines)

    _anchor = _find_anchor_line(
        lines, _search_pat, occurrence,
        ctx_before=ctx_before, ctx_after=ctx_after,
    )
    if _anchor is None:
        return {
            "ok": False,
            "error": (
                f"multiline anchor: first line {_search_pat!r} not found "
                f"in file (searched {len(lines)} lines). Provide the exact "
                f"first line of the target block, or use occurrence/context "
                f"to disambiguate."
            ),
            "failure_class": "anchor_miss",
        }

    # Verify ALL subsequent pattern lines match before accepting the range.
    for _pi in range(1, _count):
        _file_lineno = _anchor + _pi
        if _file_lineno >= len(lines):
            # Pattern extends past EOF — only a problem if that pattern line
            # is non-empty (trailing blank pattern lines are tolerated).
            if _pat_lines[_pi].strip():
                return {
                    "ok": False,
                    "error": (
                        f"multiline anchor: pattern has {_count} lines but the "
                        f"file ends at line {len(lines)} (pattern line "
                        f"{_pi + 1} extends past end of file). Re-read the "
                        f"file and provide the exact block."
                    ),
                    "failure_class": "multiline_mismatch",
                }
            break
        _pat_stripped = _pat_lines[_pi].strip()
        _file_stripped = lines[_file_lineno].strip()
        if _pat_stripped and _pat_stripped not in _file_stripped:
            return {
                "ok": False,
                "error": (
                    f"multiline anchor: pattern line {_pi + 1} "
                    f"{_pat_stripped!r} does not match file line "
                    f"{_file_lineno + 1} {_file_stripped!r}. The block "
                    f"starting at line {_anchor + 1} does not fully match "
                    f"the pattern. Read the file and provide the exact block."
                ),
                "failure_class": "multiline_mismatch",
            }

    _end = min(_anchor + _count - 1, len(lines) - 1)
    logger.info(
        "[ANCHOR_MULTILINE_RESOLVED] %d-line pattern matched file lines %d-%d "
        "(0-indexed %d-%d)",
        _count, _anchor + 1, _end + 1, _anchor, _end,
    )
    return {"ok": True, "anchor": _anchor, "end": _end, "count": _count}


def _fuzzy_find_anchor_line(
    lines: "list[str]",
    pattern: str,
    snippet_lines: "Optional[list[str]]" = None,
    edit_mode: "Optional[str]" = None,
    threshold: float = 0.75,
) -> "tuple[Optional[int], float]":
    """Conservative fuzzy fallback for anchor resolution.

    Token-overlap (Jaccard) similarity against every line, with structural
    gates to avoid the false-positive failure mode where a 0-indent snippet
    (e.g. a new top-level class) gets anchored inside an indented method body.

    Returns ``(lineno, score)`` where lineno is 0-indexed, or ``(None, 0.0)``
    when no candidate survives the gates.

    Gates (all must pass):
      1. Pattern must have >2 whitespace tokens (else too ambiguous).
      2. Best candidate score >= ``threshold`` (default 0.75).
      3. **No tie**: best score must exceed runner-up by >= 0.1 (margin gate).
      4. **Indent compatibility**: when *snippet_lines* is provided, the best
         candidate's indentation must be compatible with the snippet's base
         indentation. Specifically: a snippet whose base indent is strictly
         less than the candidate's indent is rejected (a root-level construct
         must not anchor into a deeper-indented context). This is the gate
         that prevents the class-inside-method corruption seen historically.
    """
    _pat_tokens_raw = pattern.split()
    if len(_pat_tokens_raw) <= 2:
        return None, 0.0  # too ambiguous for fuzzy

    _pat_tokens = set(_pat_tokens_raw)
    # Collect ALL candidates with their scores (need runner-up for margin gate)
    _candidates: "list[tuple[float, int]]" = []  # (score, 0-indexed lineno)
    for _i, _line in enumerate(lines):
        _line_tokens = set(_line.split())
        if not _line_tokens:
            continue
        _overlap = len(_pat_tokens & _line_tokens) / len(_pat_tokens)
        if _overlap > 0:
            _candidates.append((_overlap, _i))

    if not _candidates:
        return None, 0.0

    _candidates.sort(reverse=True)
    _best_score, _best_lineno = _candidates[0]

    # Gate 1: absolute threshold
    if _best_score < threshold:
        return None, 0.0

    # Gate 2: margin (best must beat runner-up decisively, else ambiguous)
    if len(_candidates) >= 2:
        _runner_up_score = _candidates[1][0]
        if _best_score - _runner_up_score < 0.1:
            logger.warning(
                "[ANCHOR_FUZZY_AMBIGUOUS] pattern %r: best score %.2f at line %d "
                "vs runner-up %.2f at line %d — margin too small, refusing fuzzy "
                "match (require exact anchor or ctx_before/after).",
                pattern[:80], _best_score, _best_lineno + 1,
                _runner_up_score, _candidates[1][1] + 1,
            )
            return None, 0.0

    # Gate 3: indent compatibility
    if snippet_lines:
        _snip_base_indent = None
        for _sl in snippet_lines:
            _ss = _sl.lstrip()
            if _ss:
                _sw = _sl[: len(_sl) - len(_ss)]
                if _snip_base_indent is None or len(_sw) < _snip_base_indent:
                    _snip_base_indent = len(_sw)
        if _snip_base_indent is not None:
            _anchor_line = lines[_best_lineno]
            _anchor_indent = len(_anchor_line) - len(_anchor_line.lstrip())
            # A root-level snippet (indent 0) anchoring into an indented
            # context is almost always a false positive.
            if _snip_base_indent < _anchor_indent:
                logger.warning(
                    "[ANCHOR_FUZZY_INDENT_MISMATCH] pattern %r: snippet base indent "
                    "%d < candidate line indent %d (line %d) — refusing fuzzy match "
                    "to avoid nesting a shallower construct in a deeper context.",
                    pattern[:80], _snip_base_indent, _anchor_indent, _best_lineno + 1,
                )
                return None, 0.0

    logger.warning(
        "[ANCHOR_FUZZY_MATCH] pattern %r not found exactly — fuzzy match at "
        "line %d (score=%.2f, mode=%s). ⚠️ verify with read_file.",
        pattern[:80], _best_lineno + 1, _best_score, edit_mode,
    )
    return _best_lineno, _best_score


def _inherit_anchor_indent_if_bare(
    insert_lines: "list[str]",
    anchor_line: str,
    dest_unit: Optional[int] = None,
) -> "list[str]":
    """Indent a replace_line snippet to its anchor *only when it is bare*.

    ``replace_line`` treats the planner's snippet as authoritative, including
    any indentation it specifies (e.g. an explicit 8→6 space fix), which must
    not be undone. But the common case is a bare statement with no leading
    indent (``peerMaps.delete(ws.room);``); writing it verbatim lands it
    flush-left inside a nested block. Inherit the anchor's indent only when the
    snippet carries none of its own; otherwise return it unchanged.
    """
    _snip_base = None
    for _sl in insert_lines:
        _ss = _sl.lstrip()
        if _ss:
            _sw = _sl[:len(_sl) - len(_ss)]
            if _snip_base is None or len(_sw) < len(_snip_base):
                _snip_base = _sw
    _anchor_lead = anchor_line[:len(anchor_line) - len(anchor_line.lstrip())]
    if _anchor_lead and not _snip_base:
        return reindent_to_anchor(insert_lines, anchor_line, dest_unit)
    return insert_lines
