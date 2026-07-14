"""
Shared JSON repair utilities for LLM output post-processing.

Extracted from planner_helpers_candidate.py and agent_loop.py to eliminate
code duplication (both files had near-identical implementations).

Usage::

    from .json_repair import repair_json_brackets, repair_truncated_json, try_parse_json

    text = llm_response_content
    obj = try_parse_json(text)
"""
from __future__ import annotations

import json
from typing import Any

from .llm_output_utils import strip_markdown_fences
def repair_json_brackets(text: str) -> str:
    """
    Repair common JSON bracket/brace imbalances.

    Handles:
    - Extra closing brackets: ``]]}}`` where ``}`` was expected
    - Extra closing braces: ``}]`` where ``]`` was expected
    - Unclosed arrays/objects at end of string
    - Unterminated string literals (truncated mid-field)
    - Truncated ``operations`` array with incomplete last object

    Strategy: scan character by character tracking a stack of opening brackets.
    Skip any closing bracket that doesn't match the top of the stack.
    Append any unclosed brackets at the end.
    If a string literal is unterminated at end of text, close it first.
    As last resort, detect truncation in the operations array and recover
    only the complete operation objects.
    """
    result: list[str] = []
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            result.append(ch)
            continue
        # --- outside string ---
        if ch in ("{", "["):
            stack.append(ch)
            result.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
                result.append(ch)
            # else: skip unmatched `}`
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                result.append(ch)
            # else: skip unmatched `]`
        else:
            result.append(ch)

    # ── Unterminated string repair ────────────────────────────────────
    # If the last character was inside a string literal, close it.
    # This handles truncation like: "code_snippet": "def foo(
    # where the LLM stopped before the closing quote.
    if in_string and not escape_next:
        result.append('"')

    # ── Close any remaining unclosed openers ───────────────────────────
    for opener in reversed(stack):
        result.append("}" if opener == "{" else "]")

    return "".join(result)


def repair_truncated_json(text: str) -> str | None:
    """
    Repair truncated JSON by recovering the last complete operation object
    in the ``operations`` array.

    This is a second-line defense when the LLM output ends mid-field
    (e.g. ``"code_snippet": "def foo():"`` with no closing quote or brace).
    The strategy finds the last complete ``{...}`` block inside the
    ``operations`` array, truncates to just after it, and closes the outer
    JSON structure.

    Returns the repaired string if recovery succeeded, ``None`` if the
    text does not look like truncated JSON with an operations array.
    """
    _t = text.strip()
    if not _t.startswith("{"):
        return None

    # Only attempt recovery if we can locate the operations array.
    _ops_marker = '"operations"'
    _ops_start = _t.find(_ops_marker)
    if _ops_start == -1:
        return None

    # Scan forward from _ops_start to find the '[' that opens the operations array.
    _array_start = _t.find("[", _ops_start)
    if _array_start == -1:
        return None

    # Now scan character by character from _array_start to find the
    # last complete operation object.  We track bracket depth and
    # string state, recording the position just after each complete
    # top-level object in the operations array.
    _depth = 0
    _in_str = False
    _esc = False
    _last_complete_end = -1  # position of last '}' at depth 1 (inside array)
    _in_ops_array = False

    for i in range(_array_start, len(_t)):
        ch = _t[i]
        if _esc:
            _esc = False
            continue
        if ch == "\\" and _in_str:
            _esc = True
            continue
        if ch == '"':
            _in_str = not _in_str
            continue
        if _in_str:
            continue
        if ch == "[":
            if _depth == 0:
                _in_ops_array = True
            _depth += 1
        elif ch == "]":
            _depth -= 1
            if _in_ops_array and _depth == 0:
                _in_ops_array = False
        elif ch == "{":
            _depth += 1
        elif ch == "}":
            _depth -= 1
            # A complete object at operations-array depth (1 inside array).
            if _depth == 1 and _in_ops_array:
                _last_complete_end = i

    if _last_complete_end == -1:
        # No complete operation object found.
        # If inside an operations array, return empty operations array
        # rather than giving up entirely.
        if _in_ops_array:
            return _t[:_array_start + 1] + "]}"
        return None

    # Only repair if the JSON was genuinely truncated — i.e. the
    # operations array is NOT already properly closed after the last
    # complete operation object.  Check if there's a closing ``]``
    # within 3 chars after _last_complete_end (allowing for whitespace).
    _tail = _t[_last_complete_end + 1:].lstrip()
    if _tail and _tail[0] == "]":
        # Already properly closed — no truncation.
        return None

    # Truncate to just after the last complete operation object
    # and close the operations array + outer object.
    repaired = _t[:_last_complete_end + 1] + "]}"

    # Validate: the result is at least structurally balanced.
    if not repaired.startswith("{"):
        return None

    return repaired


def _isolate_outermost_json(text: str) -> str:
    """Return the outermost balanced ``{...}``/``[...]`` substring.

    LLM output frequently wraps JSON in prose ("Here is the result:\\n{...}").
    This scans from the first ``{``/``[`` tracking bracket balance (string-aware)
    and returns the slice up to the matching close. If no opener is found or the
    scan never rebalances, the original text is returned unchanged so the
    downstream 3-tier repair ladder can still attempt recovery.

    Idempotent on already-clean JSON: a single object/array is returned as-is.
    """
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return text

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unbalanced (truncated) — return original so repair_truncated_json can try.
    return text


def _normalize_llm_json(text: str) -> str:
    """Canonical pre-processing for LLM-emitted JSON text.

    1. Strip markdown code fences (```` ```json ... ``` ````) — delegated to the
       shared :func:`strip_markdown_fences` to avoid re-implementing fence parsing.
    2. Isolate the outermost JSON object/array to drop surrounding prose.

    Both steps are idempotent and no-ops on already-clean JSON.
    """
    text = strip_markdown_fences(text)
    return _isolate_outermost_json(text)


def try_parse_json(text: str) -> Any | None:
    """
    Parse JSON from LLM output with fence/prose normalization + 3-tier repair.

    Normalization (applied before any parse attempt):
    - Strip markdown code fences (```` ```json ... ``` ````) via :func:`strip_markdown_fences`.
    - Isolate the outermost ``{...}`` / ``[...]`` to drop leading/trailing prose
      (e.g. ``"Here is the result:\\n{...}"``) using a bracket-balance scan.

    Then the 3-tier repair ladder:
    1. Direct ``json.loads``
    2. Bracket/brace repair via :func:`repair_json_brackets`
    3. Truncated JSON recovery via :func:`repair_truncated_json`

    Returns the parsed object on success, ``None`` on failure.
    """
    text = _normalize_llm_json(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt bracket repair (extra/missing brackets common in small LLM output)
    repaired = repair_json_brackets(text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Attempt truncated JSON recovery (last complete operation object)
    truncated = repair_truncated_json(text)
    if truncated is not None and truncated != text:
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    return None
