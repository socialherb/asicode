"""
Unified indentation utilities — single source of truth.

All indent-character detection, minimum-indent calculation, indent-unit
detection, block reindenting, and indent-char normalization live here.
Previously these were duplicated across 6+ locations in patch_synth,
plan_compiler, repair_apply, symbol_handlers_anchor, and tool_registry.
"""
from __future__ import annotations

import io
import logging
import re
import tokenize as _tok
from collections import defaultdict
from math import gcd
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Core primitives (level 1 — no dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def detect_indent_char(lines: list[str]) -> str:
    """Detect indent character ('\t' or ' ') from a list of file lines.

    Returns '\t' if any non-empty line starts with a tab, otherwise ' '.
    """
    for s in lines:
        stripped = s.lstrip()
        if stripped and s != stripped:
            return "\t" if "\t" in s[:len(s) - len(stripped)] else " "
    return " "


def min_indent(lines: list[str]) -> int:
    """Minimum leading whitespace count across non-empty lines.

    Uses lstrip() (all whitespace) — tab-indented files are handled
    correctly (unlike the old lstrip(" ") which ignored tabs).
    """
    vals: list[int] = []
    for s in lines:
        if (s or "").strip() == "":
            continue
        vals.append(len(s) - len(s.lstrip()))
    return min(vals) if vals else 0


def _continuation_rows(text: str, lines: Optional[list[str]] = None) -> set[int]:
    """1-based rows of physical lines that *continue* a previous logical line.

    A continuation line is one that begins while a ``(`` or ``[`` is still open
    (its leading whitespace is *alignment*, e.g. ``b)`` lined up under the open
    paren), or that follows an explicit backslash line-continuation.

    Deliberately **language-agnostic**: only ``()`` and ``[]`` are tracked, not
    ``{}`` — in C/JS/Java ``{`` opens a *block* whose inner lines carry real
    nesting depth, whereas a Python dict/set literal spanning lines is rare.
    Tracking ``{`` here (as Python's tokenizer does) would swallow whole JS
    function bodies as a single logical line and break indent-unit detection.

    Strings (``'`` ``"`` `````), line comments (``#``, ``//``) and ``/* */``
    block comments are skipped so brackets inside them are not counted.  Strings
    are treated as single-line (state resets at EOL) to avoid runaway
    over-marking on an unterminated quote in a snippet.

    *lines*: a pre-split list may be passed to skip a redundant ``split`` when
    the caller already split the text (e.g. :func:`reindent_to_match`).
    """
    rows: set[int] = set()
    depth = 0
    in_str: Optional[str] = None
    in_block_comment = False
    backslash_cont = False
    phys = lines if lines is not None else text.split("\n")
    for idx, line in enumerate(phys, start=1):
        if depth > 0 or backslash_cont or in_block_comment:
            rows.add(idx)
        backslash_cont = False
        i, n = 0, len(line)
        escaped = False
        while i < n:
            c = line[i]
            if in_block_comment:
                if c == "*" and i + 1 < n and line[i + 1] == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if in_str is not None:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == in_str:
                    in_str = None
                i += 1
                continue
            if c == "\\" and i == n - 1:
                backslash_cont = True
            elif c == "#":
                break
            elif c == "/" and i + 1 < n and line[i + 1] == "/":
                break
            elif c == "/" and i + 1 < n and line[i + 1] == "*":
                in_block_comment = True
                i += 2
                continue
            elif c in ("'", '"', "`"):
                in_str = c
            elif c in "([":
                depth += 1
            elif c in ")]":
                if depth > 0:
                    depth -= 1
            i += 1
        in_str = None  # strings don't span physical lines (snippet-safe)
    return rows


def indent_unit(text: str, char: str, cont_rows: Optional[set[int]] = None) -> int:
    """GCD of leading-run widths for lines indented *purely* with ``char``.

    Gives the file's chars-per-level (e.g. 4 for 4-space, 1 for tab).
    Falls back to a sensible default when no pure-``char`` indented line exists.

    Bracket-continuation lines (e.g. ``b)`` aligned under an open ``(``) are
    excluded: their alignment column has nothing to do with the file's nesting
    unit, and including them collapses the GCD to 1 — which inflates downstream
    indent ratios and causes indent explosion.

    *cont_rows*: a precomputed ``_continuation_rows(text)`` set may be passed to
    avoid a redundant scan when the caller already has one (e.g.
    ``reindent_to_match``).  When ``None`` it is computed here.
    """
    if cont_rows is None:
        cont_rows = _continuation_rows(text)
    u = 0
    for row, ln in enumerate(text.splitlines(), start=1):
        if not ln.strip():
            continue
        if row in cont_rows:
            continue
        st = ln.lstrip(" \t")
        lead = ln[:len(ln) - len(st)]
        if lead and set(lead) == {char}:
            u = gcd(u, len(lead))
    return u or (1 if char == "\t" else 4)


# ══════════════════════════════════════════════════════════════════════════════
# Python-only logical-line analysis (level 1b — tokenizer-based, NOT language-
# agnostic). Distinct from _continuation_rows/indent_unit above, which are
# language-agnostic char-scanners shared across Python/C/JS/Java. These use
# Python's tokenizer for exact owner-mapping of logical vs continuation rows,
# which the modify_symbol (Python-only) path needs for precise depth-remap.
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_logical_lines(snippet: str) -> Optional[tuple[dict, set]]:
    """Tokenize ``snippet`` to classify each physical row.

    Returns ``(owner, logical_rows)`` where:
      * ``owner`` maps each 1-based physical row to the row of the logical line
        it belongs to. A row whose owner is itself *starts* a logical line; any
        other row is a continuation — either inside an open bracket (alignment /
        hanging indent) or an interior line of a multi-line string.
      * ``logical_rows`` is the set of rows that start a logical line.

    Returns ``None`` when the snippet cannot be tokenized (the caller then falls
    back to char-preserving behavior). The snippet must already be dedented to
    column 0 so it tokenizes as module-level code.

    NOTE: Python-only (uses the ``tokenize`` module). The language-agnostic
    ``_continuation_rows`` above is the right tool for C/JS/Java files; this
    richer owner-mapping is only needed by the Python ``modify_symbol`` path.
    """
    owner: dict = {}
    logical_rows: set = set()
    cur: Optional[int] = None
    logical_start = True
    try:
        for t in _tok.generate_tokens(io.StringIO(snippet).readline):
            tt = t.type
            if tt == _tok.NEWLINE:
                logical_start = True
                continue
            if tt in (_tok.NL, _tok.INDENT, _tok.DEDENT, _tok.COMMENT,
                      _tok.ENCODING, _tok.ENDMARKER):
                continue
            sr, er = t.start[0], t.end[0]
            if logical_start:
                cur = sr
                logical_rows.add(sr)
                logical_start = False
            owner.setdefault(sr, cur)
            # Multi-line token (e.g. triple-quoted string): interior rows belong
            # to the same logical line and shift uniformly with it.
            for r in range(sr + 1, er + 1):
                owner.setdefault(r, cur)
    except (_tok.TokenError, IndentationError, SyntaxError):
        return None
    return owner, logical_rows


def _file_indent_unit_from_logical(source: str, file_char: str) -> int:
    """Compute the file's indent unit from logical-statement lines only.

    Uses ``_analyze_logical_lines`` to exclude alignment continuations and
    multi-line string interiors that can poison the GCD toward 1 (e.g. 2-space
    docstring bullets or column-aligned params at 17 spaces in a 4-space file).

    Falls back to the language-agnostic ``indent_unit`` when the source cannot
    be tokenized (non-Python or unparseable snippets).
    """
    analysis = _analyze_logical_lines(source)
    if analysis is None:
        return indent_unit(source, file_char)
    _, logical_rows = analysis
    lines = source.splitlines()
    logical_indents = [
        len(lines[r - 1]) - len(lines[r - 1].lstrip())
        for r in logical_rows
        if r - 1 < len(lines) and lines[r - 1].strip()
    ]
    if not logical_indents:
        return indent_unit(source, file_char)
    pu = 0
    for ind in logical_indents:
        if ind > 0:
            pu = gcd(pu, ind)
    return pu or (1 if file_char == "\t" else 4)


# ══════════════════════════════════════════════════════════════════════════════
# Block reindenting (level 2 — uses min_indent)
# ══════════════════════════════════════════════════════════════════════════════

def shift_block(
    lines: list[str],
    before_min: int,
    after_min: int,
    indent_char: str = " ",
) -> list[str]:
    """Shift indentation of *lines* to match the file's indent level.

    Strips ALL leading whitespace then regenerates indentation from scratch,
    avoiding previous bugs where ``s[k] == " "`` ignored tabs.

    Uses ratio-based scaling when ``after_min > 0`` (handles depth changes),
    and additive fallback when ``after_min == 0`` (LLM block flush-left →
    inherit file's base indent).
    """
    if not lines or before_min == after_min:
        return list(lines)

    # When the LLM replacement block is flush-left (after_min==0),
    # ratio-based scaling produces identity (ratio=1.0) and loses the
    # file's base indentation.  Use additive fallback instead.
    if after_min == 0:
        indent_ratio: Optional[float] = None
    else:
        indent_ratio = before_min / after_min

    fixed: list[str] = []
    for s in lines:
        if (s or "").strip() == "":
            fixed.append(s)
        else:
            stripped = s.lstrip()
            leading = len(s) - len(stripped)
            if indent_ratio is None:
                new_indent = max(0, leading + before_min)
            else:
                new_indent = max(0, round(leading * indent_ratio))
            fixed.append(indent_char * new_indent + stripped)
    return fixed


def reindent_text(text: str, target_indent: int, indent_char: str = "") -> Optional[str]:
    """Re-indent multiline text so its minimum indent becomes *target_indent*.

    Preserves relative indentation; empty lines unchanged.
    Returns None when the text has no non-empty lines.

    *indent_char*: indent character to use.  ``""`` (default) auto-detects
    from the input: tabs if any non-empty line starts with a tab, else spaces.
    """
    ls = text.split("\n")
    non_empty = [ln for ln in ls if ln.strip()]
    if not non_empty:
        return None
    content_base = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
    diff = target_indent - content_base
    if diff == 0:
        return text
    # Auto-detect indent char if not specified
    if not indent_char:
        indent_char = "\t" if any(ln[0] == "\t" for ln in non_empty) else " "
    adjusted = []
    for ln in ls:
        if ln.strip():
            indent = len(ln) - len(ln.lstrip())
            adjusted.append(indent_char * (indent + diff) + ln.lstrip())
        else:
            adjusted.append(ln)
    return "\n".join(adjusted)


# ══════════════════════════════════════════════════════════════════════════════
# Content-aware reindent (level 3 — plan_compiler style)
# ══════════════════════════════════════════════════════════════════════════════

def _indent_of(line: str) -> str:
    """Leading whitespace of *line*, or '' if line is blank."""
    return line[:len(line) - len(line.lstrip())] if line.strip() else ""


def _first_logical_indent(text: str) -> Optional[str]:
    """Leading whitespace of the first *logical* line in ``text``.

    Uses Python's tokenizer to distinguish logical lines (terminated by
    NEWLINE) from continuation lines (inside brackets — terminated by NL).
    This avoids the bug where a bracket-aligned continuation line at the
    top of ``text`` (e.g. an LLM snippet starting with ``    arg):`` on
    column 27) is mistaken for the base indent level, collapsing the ratio.

    Returns ``None`` when tokenization fails (e.g. syntax error in snippet);
    the caller falls back to the first non-empty line.
    """
    src = text if text.endswith("\n") else text + "\n"
    first_logical_row: Optional[int] = None
    try:
        logical_start = True
        for t in _tok.generate_tokens(io.StringIO(src).readline):
            tt = t.type
            if tt == _tok.NEWLINE:
                logical_start = True
                continue
            if tt in (_tok.NL, _tok.INDENT, _tok.DEDENT,
                      _tok.COMMENT, _tok.ENCODING, _tok.ENDMARKER):
                continue
            if logical_start:
                first_logical_row = t.start[0]
                break
    except (_tok.TokenError, IndentationError, SyntaxError):
        return None
    if first_logical_row is None:
        return None
    lines = text.split("\n")
    if first_logical_row <= len(lines):
        return _indent_of(lines[first_logical_row - 1])
    return None


def _match_site_unit(
    actual_before: str,
    actual_base: str,
    after_unit: int,
    file_unit: Optional[int] = None,
    before_lines: Optional[list[str]] = None,
) -> int:
    """Reliable per-level indent width of the match site, in its own chars.

    ``indent_unit`` returns the gcd of *absolute* indent depths, so for a site
    that shows only one logical level it returns that level's full width (e.g. a
    body all at 8 spaces → 8), not the file's true per-level unit — which is
    simply *undetectable* from a single level.  Dividing a ratio by that bogus
    "unit" re-explodes the indent (8/4 = 2× deeper).  So:

    * a tab site is always 1 char/level;
    * a space site's unit is trusted only when it shows >1 *logical* depth
      (continuation rows excluded, matching ``indent_unit``'s own view);
    * a flat space site's unit is *undetectable* here — prefer the caller-supplied
      ``file_unit`` (detected from the whole file) over the snippet's
      ``after_unit``: a 2-space file with a single-level match site would
      otherwise inherit the snippet's 4-space unit and over-indent.  Falls back
      to ``after_unit`` (the long-standing additive-path behaviour) when the
      caller did not supply a file-wide unit.
    """
    if "\t" in actual_base:
        return 1
    phys = before_lines if before_lines is not None else actual_before.split("\n")
    cont = _continuation_rows(actual_before, phys)
    depths = {
        len(_indent_of(line))
        for i, line in enumerate(phys, start=1)
        if line.strip() and i not in cont
    }
    if len(depths) > 1:
        # Reuse the cont set computed above (line 372) instead of letting
        # ``indent_unit`` re-scan ``actual_before`` for continuation rows.
        return indent_unit(actual_before, actual_base[0] if actual_base else " ", cont) or file_unit or after_unit or 1
    # Flat site (single depth): unit undetectable from the site alone.
    return file_unit or after_unit or 1


def reindent_to_match(after: str, actual_before: str, file_unit: Optional[int] = None) -> str:
    """Align indentation of ``after`` to match ``actual_before``'s file-level indent.

    When fuzzy match is used (whitespace-normalized match) in edit_blocks,
    the ``after`` text from the LLM may use different indentation than the
    actual file. This function adjusts each line:

    * If a corresponding line exists in ``actual_before`` (same index),
      that line's exact indentation is used.
    * For extra lines in ``after`` beyond ``actual_before``'s length,
      the base indentation difference is applied as a shift.
    * Empty/whitespace-only lines are preserved as-is.

    *file_unit* is the destination file's chars-per-level (detect via
    ``indent_unit(file_content, detect_indent_char(file_content.split('\\n')))``).
    When supplied it overrides the flat-site fallback in :func:`_match_site_unit`
    so a single-level match site in a 2-space file no longer inherits the
    snippet's (possibly 4-space) unit and over-indents.  ``None`` (default)
    preserves the historic behaviour (assume the snippet and file share a unit).
    """
    after_lines = after.split("\n")
    before_lines = actual_before.split("\n")

    if not after_lines or not before_lines:
        return after

    # Continuation rows of ``after`` are needed both for unit detection and for
    # the per-line shift below — compute once and reuse to avoid re-scanning.
    after_cont_rows = _continuation_rows(after, after_lines)

    # Compute depth-based indent ratio instead of flat character-count shift.
    after_base = _first_logical_indent(after) or _indent_of(next((line for line in after_lines if line.strip()), ""))
    actual_base = _indent_of(next((line for line in before_lines if line.strip()), ""))
    if after_base and actual_base:
        # Detect indent unit sizes (chars-per-level) to distinguish
        # "different unit" scale (e.g. 4→2 space) from "different depth"
        # shift (e.g. both 4-space, but file is 2 levels deeper).
        # Using raw char-count ratio for the latter causes indent explosion.
        after_unit = indent_unit(after, after_base[0], after_cont_rows)
        actual_unit = _match_site_unit(actual_before, actual_base, after_unit, file_unit, before_lines)
        indent_ratio = (actual_unit / after_unit) if after_unit else 1.0
        use_scaled_additive = True
    elif after_base and not actual_base:
        # LLM code has a base indent but the file match site is at column 0
        # (e.g. editing a top-level statement where the LLM's output was
        # emitted with its own base indentation — common in multi-line edits).
        # Use scaled-additive with actual_base="" (len=0) to strip the LLM's
        # base indent while preserving relative nesting within the block.
        indent_ratio = 1.0
        use_scaled_additive = True
    elif actual_base and not after_base:
        # The file match site is indented but ``after``'s first logical line is
        # at column 0 (the LLM emitted the whole block flush-left). Shift every
        # line to the file's base while preserving ``after``'s relative nesting.
        # Convert ``after``'s nesting *levels* into the file's indent char rather
        # than adding raw leading chars.  Scaling by levels stops the space→tab
        # explosion where 4 spaces of depth were added as 4 *tabs* (the
        # ``fix/reindent-indent-explosion`` bug class, previously missed on this
        # flush-left additive path).  ``after_base`` is "" (first logical line at
        # column 0) so the scaled-additive new-line formula strips nothing and
        # shifts each line to the file's base.  ``_match_site_unit`` supplies the
        # file's reliable per-level width (1 for tabs; the gcd unit for a nested
        # site; ``after_unit`` for a flat site whose unit is undetectable).
        after_unit = indent_unit(after, detect_indent_char(after_lines), after_cont_rows)
        file_per_level = _match_site_unit(actual_before, actual_base, after_unit, file_unit, before_lines)
        indent_ratio = (file_per_level / after_unit) if after_unit else 1.0
        use_scaled_additive = True
    else:
        indent_ratio = 1.0
        use_scaled_additive = False
    actual_indent_char = "\t" if actual_base and "\t" in actual_base else " "
    # When ``after``'s first logical line is at column 0, ``after_base`` is empty
    # and carries no char info — detect the char from the body instead so a
    # tab/space mismatch is still recognised by ``cross_char``.
    after_indent_char = after_base[0] if after_base else detect_indent_char(after_lines)
    cross_char = (after_indent_char != actual_indent_char)

    # Build content→indent map from before_lines. A stripped line that appears at
    # *different* indentations in the match site is ambiguous — content-mapping
    # it would silently flatten control flow: e.g. a ``return`` both inside an
    # ``if`` (depth 8) and at the block's base (depth 4) shares one key, so the
    # last-seen indent wins and the in-block copy escapes to column 4. The result
    # still parses, so the syntax gate cannot catch it (silent corruption).
    # Exclude such ambiguous keys so their lines fall through to the depth-remap
    # path below, which preserves ``after``'s relative nesting against the base.
    # Unambiguous lines (a single indent) keep the exact-match fast path.
    _indents_by_content: dict[str, set[str]] = defaultdict(set)
    for bl in before_lines:
        bs = bl.strip()
        if bs:
            _indents_by_content[bs].add(_indent_of(bl))
    before_indent: dict[str, str] = {}
    for bs, indents in _indents_by_content.items():
        if len(indents) == 1:
            before_indent[bs] = next(iter(indents))

    # Bracket-continuation lines carry *alignment* (e.g. ``b)`` lined up under
    # an open ``(``), not nesting depth.  Ratio-scaling them in the scaled path
    # explodes their column; instead shift each by the same delta applied to the
    # logical line it continues (tracked in ``last_delta``).  Only relevant for
    # the scaled path — the additive/identity paths already shift every line by
    # a constant, so continuation alignment is preserved there for free.
    result = []
    last_delta = 0
    for row, line in enumerate(after_lines, start=1):
        stripped = line.lstrip()
        if not stripped:
            result.append(line)
            continue
        leading = len(line) - len(stripped)
        # Content-map exact match — use the file's own indent string verbatim.
        # This MUST run before the continuation guard: a bracket-continuation
        # line whose stripped content exists in the match site (e.g. a closing
        # ``b)`` at a mixed tab+space alignment) should use that exact indent,
        # not a delta-based approximation that quantises away mixed indent.
        if stripped in before_indent:
            new_lead = before_indent[stripped]
            result.append(new_lead + stripped)
            # Domain of ``last_delta`` MUST match the continuation consumer
            # (below), which selects raw vs scaled units by ``cross_char`` —
            # NOT by ``use_scaled_additive``.  Gating here on
            # ``use_scaled_additive`` alone left a same-char/different-unit
            # block (cross_char=False, use_scaled_additive=True) computing
            # last_delta in file-char units while the consumer read it as raw
            # chars, collapsing the continuation to a column shallower than its
            # own opener.  Mirror the new-line producer's gate exactly.
            if use_scaled_additive and cross_char:
                # Normalise *leading* into file-char units so subtraction with
                # ``len(new_lead)`` (also file chars) is in a single unit.
                # Without this, e.g. tabs are subtracted from spaces, delta
                # goes negative and the next continuation line explodes.
                last_delta = len(new_lead) - round(leading * indent_ratio)
            else:
                last_delta = len(new_lead) - leading
            continue
        if use_scaled_additive and row in after_cont_rows:
            # Continuation line inside open brackets — preserve its offset from
            # the logical line above by applying that line's shift verbatim.
            if cross_char:
                new_indent = max(0, round(leading * indent_ratio) + last_delta)
            else:
                new_indent = max(0, leading + last_delta)
            result.append(actual_indent_char * new_indent + stripped)
            continue
        # New/changed line with no content match in ``before``. Re-indent it
        # to the file's level. This MUST run for column-0 lines too: an LLM
        # commonly emits the whole ``after`` block flush-left while the match
        # site is nested, so a new logical line at column 0 (e.g. a multi-line
        # call) has to shift up with the rest. Keeping such lines verbatim
        # (the old ``else`` branch) left them at column 0 next to re-indented
        # siblings — an "unexpected indent" SyntaxError.
        if use_scaled_additive:
            # Scale only the depth relative to after_base, then shift
            # to the file's base.  This correctly handles two distinct
            # cases that the old multiplicative ratio conflated:
            #   A. Different indent unit (2sp→4sp): scale is ≠1.0
            #   B. Different base depth (same unit, nested context):
            #      scale=1.0, additive shift preserves nesting levels.
            relative = max(0, leading - len(after_base))
            new_indent = max(0, len(actual_base) + round(relative * indent_ratio))
        else:
            new_indent = max(0, int(leading * indent_ratio))
        result.append(actual_indent_char * new_indent + stripped)
        if use_scaled_additive and cross_char:
            last_delta = new_indent - round(leading * indent_ratio)
        else:
            last_delta = new_indent - leading

    return "\n".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# Indent-char normalization (level 3 — repair_apply style)
# ══════════════════════════════════════════════════════════════════════════════

def _indent_char_counts(text: str) -> tuple[int, int]:
    """Return (tab-led, space-led) counts across non-blank lines."""
    tabs = spaces = 0
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if ln[0] == "\t":
            tabs += 1
        elif ln[0] == " ":
            spaces += 1
    return tabs, spaces


def normalize_indent_char_to_file(new_content: str, old_content: str) -> str:
    """Re-express *new_content*'s leading whitespace in the file's indent char.

    Surgical/diff strategies feed an LLM-rewritten ``new_content`` into a unified
    diff that is applied verbatim.  LLMs habitually emit space indentation even
    when the file uses tabs (or vice-versa), producing mixed indentation that
    may corrupt the file's style.  This converts only the lines that are indented
    with the *wrong* character — depth is preserved via per-style indent-unit
    detection, and unchanged lines are left untouched.
    """
    o_tabs, o_spaces = _indent_char_counts(old_content)
    if o_tabs == 0 and o_spaces == 0:
        return new_content  # file has no indentation to learn from (e.g. create_file)
    file_char = "\t" if o_tabs >= o_spaces else " "
    other_char = " " if file_char == "\t" else "\t"

    n_tabs, n_spaces = _indent_char_counts(new_content)
    wrong = n_spaces if file_char == "\t" else n_tabs
    if wrong == 0:
        return new_content

    new_cont_rows = _continuation_rows(new_content)
    file_unit = 1 if file_char == "\t" else indent_unit(old_content, " ")
    wrong_unit = indent_unit(new_content, other_char, new_cont_rows)
    old_line_set = set(old_content.splitlines())

    out: list[str] = []
    converted = 0
    for row, ln in enumerate(new_content.splitlines(keepends=True), start=1):
        if ln.endswith("\r\n"):
            nl, body = "\r\n", ln[:-2]
        elif ln.endswith("\n"):
            nl, body = "\n", ln[:-1]
        else:
            nl, body = "", ln
        st = body.lstrip(" \t")
        if not st or body in old_line_set:
            out.append(ln)
            continue
        # Bracket-continuation lines carry *alignment*, not nesting depth.
        # Re-quantizing their leading width into ``depth * file_unit`` units (as
        # the conversion below does) would shift the alignment column and break
        # the visual layout — leave them untouched.
        if row in new_cont_rows:
            out.append(ln)
            continue
        lead = body[:len(body) - len(st)]
        if lead and set(lead) == {other_char}:
            depth = round(len(lead) / wrong_unit) if wrong_unit else 0
            out.append(file_char * (depth * file_unit) + st + nl)
            converted += 1
        else:
            out.append(ln)
    if converted:
        logger.info(
            "[INDENT_NORM] normalized %d new line(s) to file indent char %r (unit=%d)",
            converted, file_char, file_unit,
        )
    return "".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# Anchor-based reindent (level 3 — symbol_handlers_anchor style)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Level-based rebase core (shared by reindent_to_anchor & reindent_block)
# ══════════════════════════════════════════════════════════════════════════════
# These two reindenters share ONE algorithm: min-indent → GCD unit → per-line
# level → re-emit each level as one unit of the destination's indent char.  The
# level→char mapping (space_unit) lives in a single place (_resolve_space_unit)
# so both cannot drift apart — the B2 bug class was exactly this duplication
# (tab→space mapping hardcoded to 4 in one path, detected-unit in the other).
#
# NOT consolidated here (deliberately — distinct algorithms):
#   * shift_block      — multiplicative ratio scaling (before_min/after_min),
#                        strips ALL leading whitespace; not level-based.
#   * reindent_text    — additive char-count shift (target - content_base);
#                        preserves relative indent by raw columns, no levels.
#   * reindent_to_match — content-map (verbatim file indent by stripped text)
#                        + scaled-additive leaf + bracket-continuation delta;
#                        far richer than a flat rebase.
# Forcing these into the level-based core would either change their behaviour
# or make the "core" branchier than the current separate functions.


def _block_levels(lines: list[str]) -> Optional[tuple[int, int, str]]:
    """Level analysis of an indented code block.

    Returns ``(min_count, unit, block_char)`` or ``None`` when no non-empty
    line exists.

    * ``min_count`` — leading-whitespace column of the least-indented line.
    * ``unit`` — GCD of the *relative* offsets (chars-per-level); divides every
      offset exactly, so ``level = round(rel / unit)`` round-trips losslessly
      for same-style blocks.
    * ``block_char`` — ``"\\t"`` if any non-empty line has a leading tab,
      else ``" "``.

    Single-level blocks (``unit == 0`` after the GCD) default to 1 (tab) or 4
    (space) — the historic fallback.  This is the shared "level-based rebase"
    core; keeping it in one place guarantees :func:`reindent_block` and
    :func:`reindent_to_anchor` always agree on the level→char mapping.
    """
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return None
    counts = [len(ln) - len(ln.lstrip()) for ln in nonblank]
    min_count = min(counts)
    block_char = "\t" if any(
        "\t" in ln[:len(ln) - len(ln.lstrip())] for ln in nonblank
    ) else " "
    unit = 0
    for c in counts:
        rel = c - min_count
        if rel > 0:
            unit = gcd(unit, rel)
    if unit == 0:
        unit = 1 if block_char == "\t" else 4
    return min_count, unit, block_char


def _resolve_space_unit(
    dest_unit: Optional[int], unit: int, block_char: str
) -> int:
    """Per-level width (chars) for a *space* destination.

    Prefer the caller-supplied file unit (``dest_unit``) so a tab/4-space block
    dropped into a 2-space file maps 1 level → 2 spaces (not the hardcoded 4).
    Fall back to the block's own unit (space source) or 4 (tab source) — the
    legacy default preserved for callers that don't hold the destination file
    content.
    """
    return dest_unit if dest_unit else (unit if block_char == " " else 4)


def reindent_to_anchor(
    insert_lines: list[str],
    anchor_line: str,
    dest_unit: Optional[int] = None,
) -> list[str]:
    """Re-indent *insert_lines* to match the leading whitespace of *anchor_line*.

    Computes the **minimum** leading whitespace across non-empty insert lines
    (the "base indent"), then replaces it with the anchor line's indent.
    Relative indentation within the snippet is preserved.

    *dest_unit* is the destination file's chars-per-level (e.g. 4 for a 4-space
    file, 2 for a 2-space file).  When provided, each nesting level is emitted as
    ``dest_unit`` spaces — so a tab- or 4-space-indented snippet dropped into a
    2-space file maps 1 level → 2 spaces, not the hardcoded 4.  Detect it via
    ``indent_unit(file_content, detect_indent_char(file_content))`` and pass it
    from any caller that holds the file content.  When ``None`` (default), the
    snippet's own unit is used for a space source and 4 for a tab source
    (legacy behaviour).

    The level→char math is shared with :func:`reindent_block` via
    :func:`_block_levels` / :func:`_resolve_space_unit`.

    Examples
    --------
    insert_lines = ["pc.createOffer()\\n", "  .then(...)\\n"]
    anchor_line  = "        if localStream:\\n"
    Returns      = ["        pc.createOffer()\\n", "          .then(...)\\n"]
    """
    anchor_indent = re.match(r'^[ \t]*', anchor_line).group(0)

    # NOTE: anchor_indent may be "" (column 0).  That is valid — it means
    # "rebase to 0 indent".  We continue processing below; the loop already
    # handles anchor_indent == "" correctly (all output lines get 0 indent).

    analysis = _block_levels(insert_lines)
    if analysis is None:
        return insert_lines  # all lines are empty / whitespace-only
    min_indent_count, unit, block_char = analysis

    # Destination indent character (tab if the anchor line itself uses tabs).
    indent_unit_char = "\t" if "\t" in anchor_indent else " "
    space_unit = _resolve_space_unit(dest_unit, unit, block_char)

    result = []
    for ln in insert_lines:
        stripped = ln.lstrip()
        if stripped:
            rel = max(0, (len(ln) - len(stripped)) - min_indent_count)
            level = round(rel / unit) if rel > 0 else 0
            if indent_unit_char == "\t":
                result.append(anchor_indent + "\t" * level + stripped)
            else:
                result.append(anchor_indent + " " * (level * space_unit) + stripped)
        else:
            # Preserve empty / whitespace-only lines as-is (but ensure \n)
            if not ln.endswith("\n"):
                result.append(ln + "\n")
            else:
                result.append(ln)
    return result


def reindent_block(text: str, base_indent: str, dest_unit: Optional[int] = None) -> str:
    """Re-base a block so its least-indented line sits at *base_indent*,
    preserving relative nesting **levels** across differing indent styles.

    Unlike a raw char-count rebase, this infers the block's own indent unit
    (e.g. 4-space or 1-tab) via the GCD of its relative offsets, measures each
    line's depth in *levels*, and re-emits each level as one unit of
    *base_indent*'s character.  This prevents the over-/under-indentation that
    a char-count rebase produces when the block and the destination use
    different indent characters — e.g. a 4-space-indented snippet inserted
    into a tab file (which a naïve rebase turns into 4 tabs per level).

    *dest_unit* is the destination file's chars-per-level (detect via
    ``indent_unit(file_content, detect_indent_char(file_content))``).  When
    provided, each level is emitted as ``dest_unit`` spaces in a space
    destination — so a tab block into a 2-space file maps 1 level → 2 spaces,
    not the hardcoded 4.  When ``None``, the block's own unit (space source) or
    4 (tab source) is used.

    The level→char math is shared with :func:`reindent_to_anchor` via
    :func:`_block_levels` / :func:`_resolve_space_unit`, so the two functions
    can never disagree on level→char mapping.

    Same-style blocks (space→space, tab→tab) are reproduced exactly, because
    the GCD unit divides every relative offset, so ``level * unit`` round-trips
    losslessly.  Empty lines become bare newlines; every line is \\n-terminated.
    """
    lines = text.splitlines()
    analysis = _block_levels(lines)
    if analysis is None:
        return "".join((ln if ln.endswith("\n") else ln + "\n") for ln in lines)

    min_count, unit, block_char = analysis

    # Destination character; spaces-per-level only matters for a space dest.
    base_char = "\t" if "\t" in base_indent else " "
    space_unit = _resolve_space_unit(dest_unit, unit, block_char)

    out: list[str] = []
    for ln in lines:
        stripped = ln.lstrip()
        if not stripped:
            out.append("\n")
            continue
        rel = (len(ln) - len(stripped)) - min_count
        level = round(rel / unit) if rel > 0 else 0
        if base_char == "\t":
            new_indent = base_indent + "\t" * level
        else:
            new_indent = base_indent + " " * (level * space_unit)
        out.append(new_indent + stripped + "\n")
    return "".join(out)
