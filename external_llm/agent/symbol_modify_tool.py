"""
Standalone modify_symbol tool for Main Agent.

Pure AST-based symbol replacement — no LLM calls, no class hierarchy dependencies.
Provides reliable symbol-level modifications with automatic fallback chain:

  1. AST precise (Python): decorator-aware, body-only vs full-block split
  2. Surgical edit (any language): search/replace within symbol range
  3. Line-range patch (any language): fallback by line numbers

Designed to be used as a tool handler in WriteToolsMixin, but also usable
standalone for testing or internal programmatic use.
"""

from __future__ import annotations

import ast
import difflib
import logging
import os
import re
import shutil
import subprocess
from typing import Optional

from external_llm.common.atomic_io import atomic_write_text
from external_llm.common.indent_utils import (
    _analyze_logical_lines,
    _file_indent_unit_from_logical,
    detect_indent_char,
    indent_unit,
    min_indent,
    normalize_indent_char_to_file,
    reindent_block,
)

from ..languages import LanguageId, LanguageRegistry
from ..languages.base import _find_closing_brace, net_brace_count
from ._shared_utils import compile_quiet
from .repair_helpers import _strip_redundant_dataclass_decorator, _strip_redundant_inline_imports

# Brace-delimited languages with no inline compiler gate of their own in
# _post_edit_syntax_ok (PYTHON uses compile(), JS/TS use `node --check`, GO uses
# `gofmt -e`). These fall through to the literal-aware brace-balance gate so
# a symbol-range scan that left an orphan `}` (or dropped a brace) is rejected
# before write instead of corrupting the file.
_BRACE_LANGUAGES_NO_COMPILER = frozenset({
    LanguageId.KOTLIN, LanguageId.RUST, LanguageId.C, LanguageId.CPP,
    LanguageId.JAVA, LanguageId.SCALA, LanguageId.SWIFT, LanguageId.CSHARP,
})

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_DEFINITION_PREFIXES = (
    "def ", "async def ", "class ",
    "function ", "async function ", "export function ", "export async function ",
    "export class ", "export default function ", "export default class ",
    "export interface ", "export type ", "interface ", "type ",
    "abstract class ", "export abstract class ",
    "func ", "func(",
    "public ", "private ", "protected ",
    "static ", "final ", "synchronized ",
    "enum ",
    "fun ", "suspend fun ", "override fun ", "internal fun ",
    "data class ", "sealed class ", "open class ", "inner class ",
    "object ", "companion object ",
)


def _first_significant_line(text: str, skip_decorators: bool = True) -> str:
    """First non-blank, non-comment line of ``text`` (single SSOT comment policy).

    Skips blank lines, ``#`` line comments, ``//`` line comments, ``/* */``
    block comments (possibly multi-line), and — unless ``skip_decorators`` is
    False — ``@`` decorators. Returns the stripped first significant line, or
    ``""`` if none.

    Shared by :func:`_looks_like_full_symbol_block` and the surgical-edit
    full-block reclassification so both apply ONE comment-skipping policy. The
    prior inline filters skipped only ``#``/``//``/``@`` and missed ``/* */``
    block comments: a replacement block prefixed with a javadoc ``/** */`` had
    its first significant line read as the comment text, so the def-line match
    against the (comment-free) symbol range failed, the block was misrouted to
    the body-only path, and the def line was duplicated — brace-balanced output
    the net-brace verify gate cannot catch (silent corruption).
    """
    in_block = False
    for raw in text.splitlines():
        s = raw.strip()
        if in_block:
            idx = s.find("*/")
            if idx != -1:
                in_block = False
                tail = s[idx + 2:].strip()
                if tail and not tail.startswith(("#", "//")):
                    return tail
            continue
        if not s:
            continue
        if skip_decorators and s.startswith("@"):
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        if s.startswith("/*"):
            idx = s.find("*/")
            if idx == -1:
                in_block = True
                continue
            tail = s[idx + 2:].strip()
            if tail and not tail.startswith(("#", "//")):
                return tail
            continue
        return s
    return ""


def _looks_like_full_symbol_block(text: str) -> bool:
    """Return True if text appears to be a full function/class definition block."""
    first = _first_significant_line(text)
    return any(first.startswith(p) for p in _DEFINITION_PREFIXES)


def _realign_dedented_leading_lines(code: str) -> str:
    """Realign leading decorator/comment lines that sit shallower than the def line.

    Models often emit a full replacement block whose FIRST line (typically a
    decorator) lost its leading whitespace while the remaining lines kept the
    original depth. Every strategy below anchors the block's indentation on its
    first line, so the artifact shifts the def one level deeper than its
    decorator (IndentationError: unexpected indent) or splices a column-0
    decorator into a class body. Grammar requires decorators and comments
    directly above a def/class to sit at the def's depth — pull any shallower
    one up to it. Deeper leading lines (multi-line decorator continuations)
    are left untouched.
    """
    lines = code.splitlines()
    def_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.strip().startswith(("def ", "async def ", "class "))),
        -1,
    )
    if def_idx <= 0:
        return code  # no leading lines, or not a Python definition block
    def_line = lines[def_idx]
    def_indent = len(def_line) - len(def_line.lstrip())
    prefix = def_line[:def_indent]
    changed = False
    for i in range(def_idx):
        ln = lines[i]
        if ln.strip() and (len(ln) - len(ln.lstrip())) < def_indent:
            lines[i] = prefix + ln.lstrip()
            changed = True
    if not changed:
        return code
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def _trailing_foreign_stmt(code: str) -> Optional[str]:
    """Detect a def/class/decorator row PAST the symbol's own block.

    Diagnostic-only: consulted when composing the final syntax-blocked error,
    never to gate a write. A full replacement block must contain exactly one
    definition at its base indent — the symbol's own. Models sometimes append
    content past the symbol boundary (real case: the NEXT method's signature
    opening line, sent to express "add a blank line between methods"); the
    splice then leaves that statement dangling next to the original and every
    strategy fails compile with a generic error the model misdiagnoses as a
    re-indentation failure. Returns the first offending row (stripped,
    truncated) so the error can name it, or None.

    Only rows at an indent <= the symbol def row's own indent count: nested
    helpers/inner classes sit deeper and are legitimate. A docstring line
    starting with 'def ' at a shallow column could false-positive, but that
    merely rewords an error already being returned.
    """
    lines = code.splitlines()
    def_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.strip().startswith(("def ", "async def ", "class "))),
        -1,
    )
    if def_idx < 0:
        return None
    def_line = lines[def_idx]
    def_indent = len(def_line) - len(def_line.lstrip())
    for ln in lines[def_idx + 1:]:
        stripped = ln.strip()
        if not stripped:
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent <= def_indent and stripped.startswith(
            ("def ", "async def ", "class ", "@")
        ):
            return stripped[:80]
    return None


def _reindent_relative(
    body_lines: list[str],
    anchor_indent: int,
    base_prefix: str,
    model_char: str,
    model_unit: int,
    file_char: str,
    file_unit: int,
) -> list[str]:
    """Re-emit ``body_lines`` against the file's indentation style.

    Logical-statement lines are re-expressed at the file's indent unit and char,
    prefixed by ``base_prefix``: a line's depth (its indent relative to the
    block's least-indented logical line, in the *model's* unit) is rendered in
    the *file's* unit. This is what normalizes a 2-space block into a 4-space
    file, and a tab block into a space file.

    Continuation lines — those inside an open bracket (alignment / hanging
    indent) or interior lines of a multi-line string — are NOT depth-remapped.
    They are shifted by exactly the same delta as the logical line that owns
    them, preserving their relative column. This is critical: ``model_unit`` is a
    GCD of leading-run widths, and alignment continuations (e.g. a line aligned
    to an open paren at column 27) collapse that GCD toward 1. Feeding such lines
    through the depth remap multiplied every line's indent (``file_unit /
    model_unit`` ≈ ×4), producing catastrophic over-indent. Excluding them from
    both the unit detection and the remap fixes that while keeping normalization.

    Falls back to a same-char-preserving shift when the block cannot be
    tokenized (e.g. a body-only fragment that is not valid on its own).

    Returns lines WITHOUT trailing newlines; the caller joins them.
    """
    # Dedent to column 0 so the fragment tokenizes as module-level code; row
    # numbers are preserved 1:1 with body_lines.
    dedented = "\n".join(
        (bl[anchor_indent:] if len(bl) - len(bl.lstrip()) >= anchor_indent else bl.lstrip())
        if bl.strip() else ""
        for bl in body_lines
    )
    analysis = _analyze_logical_lines(dedented) if anchor_indent >= 0 else None

    # Recompute the model's unit and anchor from logical-statement lines only,
    # so alignment continuations cannot poison the GCD / minimum.
    if analysis is not None:
        owner, logical_rows = analysis
        logical_indents = [
            len(body_lines[r - 1]) - len(body_lines[r - 1].lstrip())
            for r in logical_rows
            if r - 1 < len(body_lines) and body_lines[r - 1].strip()
        ]
        if logical_indents:
            anchor_indent = min(logical_indents)
            from math import gcd as _gcd
            mu = 0
            for ind in logical_indents:
                rel = ind - anchor_indent
                if rel > 0:
                    mu = _gcd(mu, rel)
            model_unit = mu or model_unit
    else:
        owner, logical_rows = {}, set()

    out: list[str] = []
    delta_by_owner: dict = {}
    for idx, bl in enumerate(body_lines):
        row = idx + 1
        stripped = bl.lstrip()
        if not stripped:
            # Blank / whitespace-only line — emit empty, never trailing spaces.
            out.append("")
            continue
        cur_indent = len(bl) - len(stripped)
        is_logical = (analysis is None) or (owner.get(row, row) == row)
        # Owned-continuation preservation only makes sense within one indent
        # character. Across a tab↔space conversion, a column-aligned offset has
        # no faithful translation, so fall back to depth-remapping such lines too
        # (keeps them bounded instead of emitting a column-count run of tabs).
        if not is_logical and model_char != file_char:
            is_logical = True
        if is_logical:
            rel = max(0, cur_indent - anchor_indent)
            if model_char == file_char and (analysis is None or model_unit == file_unit):
                # Same char: exact shift. model_unit is only trusted for a
                # normalizing remap when recomputed from logical lines (analysis
                # succeeded) — an untrusted GCD can be poisoned to 1 and explode.
                extra = rel
            else:
                depth = rel / model_unit if model_unit else 0
                extra = round(depth * file_unit)
            line = base_prefix + file_char * extra + stripped
            # delta in characters, so owned continuations shift identically.
            new_count = len(line) - len(stripped)
            delta_by_owner[row] = new_count - cur_indent
            out.append(line)
        else:
            owner_row = owner.get(row, row)
            delta = delta_by_owner.get(owner_row, len(base_prefix) - anchor_indent)
            new_count = max(0, cur_indent + delta)
            out.append(file_char * new_count + stripped)
    return out


def _mode_logical_indent(lines: list[str]) -> int:
    """Return the most common leading-whitespace width among non-blank lines.

    Ties are broken toward the SHALLOWEST indent (min), which is the safe
    choice for drift detection: an ambiguous block is assumed to sit at the
    shallower of the competing depths so the correction never over-shifts.

    Unlike ``_min_logical_indent``, this is robust to a single outlier line
    sitting shallower than the body (e.g. a docstring whose first line the
    model anchored at the target depth while the rest drifted one level deep).
    The minimum would read the outlier's depth and report "no drift", leaving
    the majority of lines over-indented; the mode reports the majority's true
    depth and lets the correction fire. Used by ``_correct_indent_drift``.
    """
    from collections import Counter
    widths = [
        len(ln) - len(ln.lstrip())
        for ln in lines
        if ln.strip()
    ]
    if not widths:
        return 0
    counts = Counter(widths)
    top = max(counts.values())
    return min(w for w, c in counts.items() if c == top)


def _block_parses_after_dedent(lines: list[str]) -> bool:
    """Return True if ``lines`` parse as valid Python after dedenting to col 0.

    Used by ``_correct_indent_drift`` to tell a **nested** body (parses cleanly
    even though its mode-indent is deep) apart from a **shallow outlier
    masking a drifted majority** (does not parse, because the outlier sits at
    the target depth while the rest of the body is one level deep — an
    inconsistent indent).

    The block is dedented by its own minimum indent so that relative structure
    is preserved; ``ast.parse`` then succeeds exactly when the indent profile
    is internally consistent.
    """
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return True
    common = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
    dedented = [
        (ln[common:] if len(ln) - len(ln.lstrip()) >= common else ln.lstrip())
        if ln.strip() else ""
        for ln in lines
    ]
    try:
        ast.parse("\n".join(dedented))
        return True
    except SyntaxError:
        return False


def _correct_indent_drift(
    reindented: list[str],
    body_indent: str,
    symbol: str,
) -> list[str]:
    """Validate ``_reindent_relative`` output and correct a one-level drift.

    In body-only mode the re-emitted body's logical lines MUST sit at
    ``len(body_indent)`` (the symbol's original base indent). A result that is
    one full ``file_unit`` deeper indicates the re-anchor under-shot — the model
    body was passed in already-indented-by-one-level and the relative remap
    preserved that extra level rather than collapsing it. This performs a
    delta shift to restore the contract.

    Metric selection (min-primary, mode-fallback):

    The PRIMARY diagnostic is the MINIMUM logical indent. It is the true base
    of the body and correctly leaves **nested** blocks alone — a body whose
    majority of lines sit inside a nested ``def``/``for``/``if`` has a deep
    *mode* but a shallow *min*, and ``min == target`` means no correction is
    needed. Using the mode there (the old behaviour) flattened the nested body
    into a parse error.

    The MODE is consulted only as a FALLBACK, and only when the block does NOT
    parse after dedenting — i.e. when a shallow outlier (a docstring's first
    physical line, or a dedented ``return`` realigned by
    ``_reindent_relative``) sits at the target depth while the rest of the
    body drifted one level deep. In that shape ``min == target`` would report
    "no drift" yet the block is unparseable, so the mode (the majority's true
    depth) drives the correction and the outlier is left untouched (it is
    already at the target). A block that parses cleanly with ``min == target``
    is never shifted, so nested functions, alignment continuations and
    legitimately-dedented statements are preserved.

    Only fires when the offset is a POSITIVE multiple of the body indent width
    (a downward correction). Negative / non-multiple offsets are left alone to
    avoid masking genuine alignment; those are already guarded by the downstream
    ``compile()`` check. The corrective shift is applied ONLY to lines deeper
    than ``target`` — a line already at or above the target keeps its indent,
    so a shallow outlier is never pushed further up into a parse error.
    """
    target = len(body_indent)
    if target == 0:
        return reindented  # module-level body: nothing to validate against

    actual = min_indent(reindented)
    delta = actual - target
    if delta <= 0:
        # min is at or above the target. If the block is already parseable
        # (the common case: nested function, uniform-correct, or alignment
        # continuation), there is nothing to fix — shifting would corrupt a
        # legitimate deeper-nested body. Only when the block does NOT parse
        # (a shallow outlier masking a drifted majority) do we fall back to
        # the mode to recover the majority's true depth.
        if _block_parses_after_dedent(reindented):
            return reindented
        mode = _mode_logical_indent(reindented)
        delta = mode - target
        if delta <= 0:
            return reindented
        logger.info(
            "modify_symbol body-only indent drift (parse-fail fallback) for %s: "
                "min_indent=%d mode_indent=%d expected=%d (delta=%d); "
                "correcting over-indented lines",
            symbol, actual, mode, target, delta,
        )
    else:
        logger.warning(
            "modify_symbol body-only indent drift for %s: min_indent=%d expected=%d "
                "(delta=%d); applying corrective shift to over-indented lines",
            symbol, actual, target, delta,
        )
    corrected = []
    for ln in reindented:
        if not ln.strip():
            corrected.append(ln)  # preserve blank lines exactly
            continue
        cur = len(ln) - len(ln.lstrip())
        if cur <= target:
            # A line already at or above the target (e.g. a docstring or a
            # legitimately-dedented statement) is left untouched — shifting it
            # up by `delta` would under-indent it into a parse error.
            corrected.append(ln)
            continue
        # Strip exactly `delta` leading chars (a uniform de-indent). delta <= cur
        # always holds because cur > target and delta is anchored to target.
        cut = min(delta, cur)
        corrected.append(ln[cut:])
    return corrected


def _correct_full_block_body_drift(
    block_lines: list[str],
    file_char: str,
    def_indent: int,
    file_unit: int,
    symbol: str,
) -> list[str]:
    """Correct a one-level BODY drift in a full-block replacement.

    Full-block counterpart of :func:`_correct_indent_drift`. The relative
    remap (:func:`_reindent_relative`) preserves the model's indent PROFILE,
    so when the model emits its body one full ``file_unit`` deeper than its
    own def line, the remap faithfully reproduces that extra level in the
    file. Python parses a uniformly-over-indented body as valid, so the
    downstream ``compile()`` check does NOT catch it — a silent body drift
    reported to the caller as success.

    Unlike body-only mode, the full block contains the def/class statement
    itself (re-anchored to ``def_indent``) plus any multi-line signature
    continuation lines. Those are OWNED by the def statement and must keep
    their re-anchored position; only the BODY (everything past the def
    statement and its signature) is validated and corrected. The split uses
    :func:`_analyze_logical_lines` to group signature continuation rows with
    the def row, which keeps multi-line signatures and decorators intact.

    The correction delegates to :func:`_correct_indent_drift` on the body
    slice only, so the metric selection (min-primary / mode-fallback), the
    positive-multiple gate, and the per-line "deeper than target only" shift
    are identical to body-only mode — the two paths stay symmetric.

    ``def_indent`` is the def line's column (``target.col_offset``); the
    body's expected base indent is one ``file_unit`` deeper
    (``def_indent + file_unit``), matching how the file indents a body. A
    module-level def (``def_indent == 0``) still works: body_indent becomes
    ``file_unit`` (e.g. 4), and a body the model over-indented is corrected.
    """
    if not block_lines or file_unit <= 0:
        return block_lines

    body_indent = file_char * (def_indent + file_unit)
    target = len(body_indent)

    # Locate the def/class statement row, then split the block into
    # header (def row + its signature/decorator continuation rows) and body.
    # _analyze_logical_lines groups bracket/hanging continuation rows under
    # the statement that opens them, so a multi-line signature is owned by
    # the def row and stays in the header.
    analysis = _analyze_logical_lines("\n".join(block_lines))
    if analysis is None:
        # Block didn't tokenize (shouldn't happen for a full def block, but
        # be defensive): fall back to a simple row split — the first line
        # whose content starts with def/class is the header anchor.
        def_row = next(
            (i + 1 for i, ln in enumerate(block_lines)
             if ln.lstrip().startswith(("def ", "async def ", "class "))),
            1,
        )
        header_end = def_row
    else:
        owner, logical_rows = analysis
        def_row = next(
            (r for r in sorted(logical_rows)
             if 1 <= r <= len(block_lines)
             and block_lines[r - 1].lstrip().startswith(("def ", "async def ", "class "))),
            None,
        )
        if def_row is None:
            return block_lines  # no def/class statement — nothing to anchor on
        # Header = the def row and every row owned by it (signature
        # continuations), plus any rows before it (decorators/comments).
        header_end = max(
            r for r in range(1, len(block_lines) + 1)
            if r <= def_row or owner.get(r, r) == def_row
        )

    header = block_lines[:header_end]
    body = block_lines[header_end:]
    if not body:
        return block_lines  # nothing to validate

    corrected_body = _correct_indent_drift(body, body_indent, symbol)
    if corrected_body == body:
        return block_lines  # no drift detected — leave untouched

    logger.info(
        "modify_symbol full-block body indent drift for %s: def_indent=%d "
            "expected body_indent=%d; applying corrective shift to over-indented body lines",
        symbol, def_indent, target,
    )
    return header + corrected_body


def _create_unified_diff(file_path: str, old_content: str, new_content: str) -> str:
    """Create a unified diff string compatible with git apply.

    Normalizes indent chars before diffing to prevent LLM-emitted
    space-indented code from corrupting tab-indented files (Go, Makefile, etc.).
    """
    if old_content and new_content:
        new_content = normalize_indent_char_to_file(new_content, old_content)

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm='',
    )
    result = '\n'.join(diff)
    if result:
        result += '\n'
    return result


def _find_symbol_ast_node(
    source: str,
    symbol: str,
    *,
    tree: Optional[ast.AST] = None,
) -> Optional[ast.AST]:
    """Find the best matching AST node for a symbol name.

    Supports 'ClassName.method_name' notation for locating methods.

    When *tree* is provided, it is reused as-is (the caller is responsible for
    having parsed it). Otherwise the source is parsed here. This lets a caller
    that already parsed the source (e.g. :func:`_apply_ast_precise`) avoid a
    redundant second ``ast.parse`` of the same text.
    """
    class_name: Optional[str] = None
    method_name = symbol
    if "." in symbol:
        parts = symbol.split(".", 1)
        class_name, method_name = parts[0], parts[1]

    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

    candidates: list = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if class_name:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        candidates.append(item)
        else:
            if node.name == method_name:
                candidates.append(node)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    top_level = [n for n in candidates if n.col_offset == 0]
    if top_level:
        return top_level[0]
    return candidates[0]


def _apply_ast_precise(
    source: str,
    file_path: str,
    symbol: str,
    code: str,
) -> tuple[Optional[str], str]:
    """Apply AST-precise symbol body replacement.

    Returns (diff_or_None, mode_name).
    """
    new_body = code
    if not new_body or not new_body.strip():
        return None, "skipped_no_new_body"

    if source:
        new_body = _strip_redundant_inline_imports(new_body, source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, "skipped_ast_error"

    target = _find_symbol_ast_node(source, symbol, tree=tree)
    if target is None:
        return None, "skipped_symbol_not_found"

    end_lineno = getattr(target, "end_lineno", None)
    if end_lineno is None:
        return None, "skipped_no_end_lineno"

    lines = source.splitlines(keepends=True)

    deco_list = getattr(target, "decorator_list", [])
    deco_start_line = deco_list[0].lineno if deco_list else target.lineno

    is_full_block = _looks_like_full_symbol_block(new_body)

    if is_full_block:
        start_idx = deco_start_line - 1
        end_idx = end_lineno

        if start_idx < 0 or end_idx > len(lines):
            return None, "skipped_invalid_range"

        original_indent = target.col_offset
        block_lines = new_body.splitlines()
        new_first_line = next((nl for nl in block_lines if nl.strip()), "")
        new_indent = len(new_first_line) - len(new_first_line.lstrip()) if new_first_line else 0

        # Re-anchor the model's block to the symbol's true indentation and
        # normalize its indent unit/char to the file's. The def line itself
        # (relative depth 0) lands at original_indent; body/continuation lines
        # keep their logical depth but in the file's indent style.
        file_char = detect_indent_char(lines)
        file_unit = _file_indent_unit_from_logical(source, file_char)
        model_char = detect_indent_char(block_lines)
        model_unit = indent_unit(new_body, model_char)
        if new_indent != original_indent or model_char != file_char or model_unit != file_unit:
            logger.info(
                "Reindent %s: orig=%d new=%d, model unit/char=%d/%r -> file %d/%r",
                symbol, original_indent, new_indent, model_unit, model_char,
                file_unit, file_char,
            )
            corrected = _reindent_relative(
                block_lines, new_indent, file_char * original_indent,
                model_char, model_unit, file_char, file_unit,
            )
            new_body = "\n".join(corrected)
            block_lines = corrected

        # Defense-1 (full-block parity with body-only below): the relative
        # remap preserves the model's RELATIVE indent profile, so a body the
        # model emitted one full file_unit deeper than its own def line lands
        # one level too deep in the file. Python parses a uniformly-over-
        # indented body fine, so the downstream compile() check does NOT catch
        # it — a silent one-level body drift reported as success (the very
        # regression observed in design-chat logs). Symmetric with the
        # body-only branch, validate the BODY lines (everything past the
        # def/class statement and its multi-line signature) against the
        # symbol's original body indent and correct a one-level drift. The
        # def line + signature continuation lines are owned by the def
        # statement and are excluded so the re-anchor on original_indent holds.
        block_lines = _correct_full_block_body_drift(
            block_lines, file_char, original_indent, file_unit, symbol,
        )
        new_body = "\n".join(block_lines)

        new_body_text = new_body if new_body.endswith("\n") else new_body + "\n"
        new_lines = [*lines[:start_idx], new_body_text, *lines[end_idx:]]
        precision_mode = "python_full_block"
    else:
        # NOTE: _strip_redundant_dataclass_decorator belongs ONLY in the
        # body-only path. In the full-block path above, the model's block —
        # decorators included — REPLACES the original symbol region, so
        # stripping @dataclass from new_body would silently delete it (the
        # original decorator lines are overwritten, not preserved as a header).
        # It only matters here, where header_lines keeps the original decorator
        # and a misclassified full block in new_body would otherwise duplicate
        # it as @dataclass\n@dataclass\nclass X:.
        new_body = _strip_redundant_dataclass_decorator(new_body, source)
        # Defense-2: detect a Python full-block misclassification. The model
        # intended a full replacement (it sent a def/class line) but the prefix
        # heuristic in _looks_like_full_symbol_block did not fire — e.g. a
        # variant like 'def\tfoo' (tab instead of space) or a dedented column-0
        # def that the prefix check rejects. Splicing such a block as body-only
        # would inject a nested def at the symbol's base indent (a nested
        # function, silently shadowing the real one). We cannot safely re-route
        # here, so emit a warning so the failure is observable. The keyword test
        # below is deliberately independent of _DEFINITION_PREFIXES so it catches
        # exactly the variants that slipped past the prefix heuristic.
        _first_stmt = next(
            (ln.strip() for ln in new_body.splitlines()
             if ln.strip() and not ln.strip().startswith(("#", "@"))),
            "",
        )
        # 'def'/'class' followed by a non-identifier boundary (space, tab, paren)
        # identifies a real Python def/class statement regardless of which
        # boundary char the prefix heuristic demanded.
        if re.match(r"(async\s+def|def|class)(?![A-Za-z0-9_])", _first_stmt):
            logger.warning(
                "modify_symbol mode misclassification risk for %s: body-only path "
                    "taken but model code starts with a def/class statement (%r) "
                    "— _looks_like_full_symbol_block did not match the prefix",
                symbol, _first_stmt[:60],
            )
        body_stmts = getattr(target, "body", [])
        if not body_stmts:
            return None, "skipped_empty_body"

        body_start_line = body_stmts[0].lineno

        body_indent = "    "
        for bl in lines[body_start_line - 1: end_lineno]:
            stripped_bl = bl.rstrip("\n\r")
            if stripped_bl.strip():
                body_indent = stripped_bl[: len(stripped_bl) - len(stripped_bl.lstrip())]
                break

        # Re-emit the model's body at the symbol body's base indent, mapping the
        # model's indent unit/char onto the file's (no-op when they match, so
        # consistent 4-space bodies are untouched).
        model_lines = new_body.splitlines()
        anchor = min(
            (len(_item_) - len(_item_.lstrip()) for _item_ in model_lines if _item_.strip()),
            default=0,
        )
        file_char = "\t" if "\t" in body_indent else detect_indent_char(lines)
        file_unit = _file_indent_unit_from_logical(source, file_char)
        model_char = detect_indent_char(model_lines)
        model_unit = indent_unit(new_body, model_char)
        reindented = _reindent_relative(
            model_lines, anchor, body_indent,
            model_char, model_unit, file_char, file_unit,
        )
        # Defense-1: validate the re-anchor contract. The least-indented
        # logical line MUST sit at len(body_indent); a deeper result signals a
        # one-level drift the relative remap failed to collapse (the original
        # over-indent regression). Correct it before splicing.
        reindented = _correct_indent_drift(reindented, body_indent, symbol)
        new_body_lines = [bl + "\n" for bl in reindented]

        if not new_body_lines:
            return None, "skipped_empty_new_body"

        header_start_idx = deco_start_line - 1
        header_end_idx = body_start_line - 1
        header_lines = lines[header_start_idx:header_end_idx]

        end_idx = end_lineno
        if header_start_idx < 0 or end_idx > len(lines):
            return None, "skipped_invalid_range"

        if not header_lines:
            logger.warning(
                "body_only: header_lines empty for %s (deco_start=%d body_start=%d) - skipping",
                symbol, deco_start_line, body_start_line,
            )
            return None, "skipped_empty_header"

        new_lines = lines[:header_start_idx] + header_lines + new_body_lines + lines[end_idx:]
        precision_mode = "python_body_only"

    new_content = "".join(new_lines)

    try:
        compile_quiet(new_content, file_path, "exec")
    except SyntaxError as e:
        logger.warning("AST precise produced invalid syntax for %s: %s", symbol, e)
        return None, "skipped_compile_error"

    if new_content == source:
        return None, "no_change"

    diff = _create_unified_diff(file_path, source, new_content)
    return diff, precision_mode


def _apply_surgical_edit(
    source: str,
    file_path: str,
    symbol: str,
    code: str,
    sym_start_line: int,
    sym_end_line: int,
) -> Optional[str]:
    """Apply a surgical search/replace edit within the symbol's line range."""
    lines = source.splitlines(keepends=True)
    sym_text = "".join(lines[sym_start_line:sym_end_line])

    # `_looks_like_full_symbol_block` recognises Python/JS/modifier-prefixed
    # defs but misses bare C-family return-type signatures (``int foo()``,
    # ``void bar()``, ``char* baz()``). For those, reclassify the replacement as
    # a full block when its first significant line re-states the symbol's own
    # signature. Without this, Allman-style C blocks were misrouted to the
    # body-only path and the full code spliced into the body slot — duplicating
    # the def line and silently corrupting the file (the inserted block is
    # brace-balanced, so the net-brace verify gate cannot catch it). This match
    # is exact, so body-only code (which by definition omits the def line) is
    # never misclassified — no regression risk for the Python body-only path.
    full_block = _looks_like_full_symbol_block(code)
    if not full_block:
        # SSOT comment policy: _first_significant_line skips #/// AND /* */
        # block comments, so a javadoc-prefixed replacement still matches the
        # (comment-free) symbol signature and is correctly classified as a
        # full block — preventing the def-line duplication described above.
        sym_def_line = _first_significant_line(sym_text)
        code_def_line = _first_significant_line(code)
        if sym_def_line and code_def_line == sym_def_line:
            full_block = True

    if full_block:
        search_text = sym_text
    else:
        # Locate the first BODY line: skip the def/class statement header
        # INCLUDING any multi-line signature (a parenthesised parameter list
        # that spans rows). The old logic only skipped the first non-blank/
        # non-decorator line, so for ``def foo(\n    a: int,\n) -> None:`` it
        # picked the ``a: int,`` parameter row as body_start and built a
        # search_text spanning the parameters too — corrupting the signature
        # on a body-only edit. Track bracket depth so the body starts only
        # after the header line whose trailing ``:`` closes all brackets.
        body_start = sym_start_line
        paren_depth = 0
        header_seen = False
        for i in range(sym_start_line, sym_end_line):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("@") or stripped.startswith("#"):
                continue
            # Count brackets outside of this row's stripped content (brackets
            # rarely appear inside comments on a signature continuation row).
            paren_depth += stripped.count("(") - stripped.count(")")
            paren_depth += stripped.count("[") - stripped.count("]")
            paren_depth += stripped.count("{") - stripped.count("}")
            header_seen = True
            if paren_depth <= 0 and stripped.endswith(":") and header_seen:
                # This is the closing header row (e.g. ') -> None:'); the body
                # starts on the next non-blank line.
                for j in range(i + 1, sym_end_line):
                    if lines[j].strip():
                        body_start = j
                        break
                break
            if paren_depth <= 0 and not stripped.endswith(":"):
                # Single-line header already closed and this row IS the header;
                # body is the next non-blank line.
                for j in range(i + 1, sym_end_line):
                    if lines[j].strip():
                        body_start = j
                        break
                break
        search_text = "".join(lines[body_start:sym_end_line])

    replace_text = code if code.endswith("\n") else code + "\n"

    # Strategy 1: exact match (with indentation normalization)
    # Re-base replace indentation to search_text's base indent, preserving the
    # block's relative nesting LEVELS in the destination's indent character
    # (handles tab/space mismatch without over-indenting — see reindent_block).
    _si_base_indent = ""
    for _sil in search_text.splitlines():
        if _sil.strip():
            _si_base_indent = _sil[:len(_sil) - len(_sil.lstrip())]
            break
    # Use the LOGICAL file indent unit (docstring/continuation lines excluded) —
    # NOT the raw indent_unit(). Raw GCD across the whole source is poisoned by
    # docstring continuation lines (e.g. "      a → b" inside a 4-space file gives
    # gcd(4,6)=2), which would make reindent_block() compress the model's correct
    # 4-space block to 2-space and corrupt the file. Parity with the AST-precise
    # path (see _file_indent_unit_from_logical usage above). Falls back to raw
    # indent_unit() only when the file can't be tokenized (non-Python / broken).
    _si_dest_unit = _file_indent_unit_from_logical(source, detect_indent_char(source.splitlines()))
    _si_normalized_replace = reindent_block(replace_text, _si_base_indent, _si_dest_unit)

    match_pos = sym_text.find(search_text)
    if match_pos >= 0:
        new_sym_text = sym_text[:match_pos] + _si_normalized_replace + sym_text[match_pos + len(search_text):]
        new_content = "".join(lines[:sym_start_line]) + new_sym_text + "".join(lines[sym_end_line:])
        if new_content != source:
            return _create_unified_diff(file_path, source, new_content)
        return None

    # Strategy 2: fuzzy match with uniqueness guard
    search_stripped = [ln.strip() for ln in search_text.splitlines() if ln.strip()]
    replace_stripped_lines = replace_text.splitlines(keepends=True)
    if search_stripped:
        fuzzy_candidates = []
        for si in range(sym_start_line, min(sym_end_line, len(lines))):
            matched = True
            file_idx = si
            for s_ln in search_stripped:
                while file_idx < sym_end_line and file_idx < len(lines):
                    if lines[file_idx].strip():
                        break
                    file_idx += 1
                if file_idx >= sym_end_line or file_idx >= len(lines):
                    matched = False
                    break
                if lines[file_idx].strip() != s_ln:
                    matched = False
                    break
                file_idx += 1
            if matched:
                fuzzy_candidates.append(si)

        if len(fuzzy_candidates) == 1:
            best_start = fuzzy_candidates[0]
            actual_end = best_start
            s_idx = 0
            while s_idx < len(search_stripped) and actual_end < len(lines):
                if lines[actual_end].strip():
                    s_idx += 1
                actual_end += 1

            orig_indent = 0
            for ln in lines[best_start:actual_end]:
                if ln.strip():
                    orig_indent = len(ln) - len(ln.lstrip())
                    break

            adjusted_replace = []
            # Measure base indent of replacement (first non-blank line)
            _fi_repl_indent = 0
            for _fi_rl in replace_stripped_lines:
                if _fi_rl.strip():
                    _fi_repl_indent = len(_fi_rl) - len(_fi_rl.lstrip())
                    break
            file_char = detect_indent_char(lines)
            for rl in replace_stripped_lines:
                stripped = rl.lstrip()
                if stripped:
                    cur_indent = len(rl) - len(stripped)
                    relative = cur_indent - _fi_repl_indent
                    adjusted_indent = max(0, orig_indent + relative)
                    adjusted_replace.append(file_char * adjusted_indent + stripped)
                else:
                    adjusted_replace.append(rl)

            if adjusted_replace and not adjusted_replace[-1].endswith("\n"):
                adjusted_replace[-1] += "\n"

            new_content = (
                "".join(lines[:best_start])
                + "".join(adjusted_replace)
                + "".join(lines[actual_end:])
            )
            if new_content != source:
                return _create_unified_diff(file_path, source, new_content)
    return None


def _find_symbol_def_line(
    lines: list[str], bare: str, file_path: str
) -> Optional[int]:
    """Find the 0-indexed line where symbol ``bare`` is defined.

    Uses the language provider's typed symbol patterns first — the single
    source of truth already maintained per language (handles visibility
    modifiers, ``override``/``suspend``/``open``/``data`` keywords,
    annotations, and other prefixes that a naive leading-keyword check
    misses). Falls back to the legacy prefix list only for languages without
    a registered provider.
    """
    # 1. Provider patterns (typed policy) — single source of truth.
    provider = LanguageRegistry.instance().get(file_path)
    if provider is not None:
        name_re = re.escape(bare)
        for sp in provider.get_symbol_patterns(kind="any"):
            try:
                rx = re.compile(sp.regex.replace("{name}", name_re))
            except re.error:
                continue
            for i, line in enumerate(lines):
                # Provider regexes anchor on the declaration keyword
                # (``fun``, ``func``, ``class``, ``def`` …), so call sites
                # do not match — safe to take the first hit.
                if rx.search(line):
                    return i

    # 2. Legacy prefix fallback (languages with no registered provider).
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped.startswith(f"{p}{bare}") or stripped.startswith(f"{p} {bare}")
               for p in ("async def ", "async function ", "def ", "class ",
                         "func ", "function ", "fun ")):
            return i
    return None


def _find_symbol_range_via_treesitter(
    source: str, symbol: str, file_path: str
) -> Optional[tuple[int, int]]:
    """Locate a symbol's line range via the tree-sitter AST.

    Returns 0-indexed ``(start, exclusive_end)`` to match
    ``_find_symbol_line_range``'s contract, or ``None`` when tree-sitter is
    unavailable, the grammar is not installed, or the symbol is not found.

    Uses the same ``find_all_symbols`` extractor as the cross-file index and the
    file-outline path — the AST is the single source of truth for BOTH the start
    and the end line of a construct, so no brace-balancing heuristic is needed
    (those miscount on strings/comments that contain braces). Installing a
    grammar (e.g. ``tree_sitter_kotlin``) automatically enables this path with
    no code change here.
    """
    try:
        from ..languages.tree_sitter_utils import (
            _LANG_MODULE_MAP as _TS_LANG_MODULE_MAP,
        )
        from ..languages.tree_sitter_utils import (
            find_all_symbols as _ts_find_all_symbols,
        )
        from ..languages.tree_sitter_utils import (
            get_available_languages as _ts_available_languages,
        )
    except ImportError:
        return None
    lang_id = LanguageId.from_path(file_path).value
    if lang_id not in _TS_LANG_MODULE_MAP or lang_id == "python":
        return None
    if lang_id not in _ts_available_languages():
        return None  # grammar not installed → caller falls back to regex
    try:
        syms = _ts_find_all_symbols(source, lang_id)
    except Exception:
        return None
    bare = symbol.split(".")[-1]
    for name, _kind, start_line, end_line in syms:
        if name == bare:
            # find_all_symbols yields 1-indexed inclusive lines; convert to
            # 0-indexed start / exclusive end (Python slice boundary).
            return (start_line - 1, end_line)
    return None


def _find_symbol_line_range(source: str, symbol: str, file_path: str) -> Optional[tuple[int, int]]:
    """Find the line range (start_line, end_line) for a symbol.

    Returns 0-indexed (start, exclusive_end).
    """
    if LanguageId.from_path(file_path) is LanguageId.PYTHON:
        node = _find_symbol_ast_node(source, symbol)
        if node is not None:
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is not None:
                deco_list = getattr(node, "decorator_list", [])
                start = deco_list[0].lineno - 1 if deco_list else node.lineno - 1
                return (start, end_lineno)

    # Non-Python: prefer the tree-sitter AST for an accurate (start, end) taken
    # straight from the parse — no brace-balancing. Falls back to the typed
    # provider regex + brace/indent heuristic when tree-sitter is unavailable
    # (e.g. Kotlin grammar not yet installed) or the symbol isn't in the AST.
    ts_range = _find_symbol_range_via_treesitter(source, symbol, file_path)
    if ts_range is not None:
        return ts_range

    # Non-Python: locate the definition line via the typed provider patterns,
    # then compute its extent by brace balance / indentation.
    lines = source.splitlines()
    bare = symbol.split(".")[-1]
    i = _find_symbol_def_line(lines, bare, file_path)
    if i is None:
        return None

    stripped = lines[i].strip()
    sym_indent = len(lines[i]) - len(stripped)
    # Line index carrying the opening brace: the def line itself (K&R style),
    # or — for Allman/BSD style — the line after the signature.
    brace_open_line = i
    has_brace = "{" in stripped
    if not has_brace:
        # Allman style (C/C++/C#/Java/…): the '{' sits on the line after the
        # signature. Detect it so brace-balancing yields an accurate range.
        # Without this, the indentation heuristic below sees the '{' line at
        # the signature's own indent and returns a body-less range, which on
        # edit leaves an orphan body block (or, with the prior off-by-one, an
        # empty (i, i) range that duplicated the symbol instead of replacing).
        j = i + 1
        in_block = False
        while j < len(lines):
            s = lines[j].strip()
            if in_block:
                if "*/" in s:
                    in_block = False
                j += 1
                continue
            if s.startswith("/*"):
                # /* ... */ block comment (javadoc etc.) may sit between an
                # Allman signature and its '{'; skip it — only enter multi-line
                # state when the close marker isn't on this same line.
                if "*/" not in s:
                    in_block = True
                j += 1
                continue
            if s and not s.startswith(("#", "//")):
                break
            j += 1
        if j < len(lines) and lines[j].lstrip().startswith("{"):
            has_brace = True
            brace_open_line = j
    if has_brace:
        # Brace-delimited language: compute the extent via the literal/comment-
        # aware brace scanner SSOT (languages.base.find_brace_block_end, shared
        # with all C-family providers), NOT a naive per-line
        # `count('{') - count('}')`. The naive counter miscounts braces inside
        # string/char/comment literals — e.g. Kotlin `val close = "close }"`
        # or `// trailing brace }` — and reaches depth 0 one line early,
        # dropping the real closing brace from the range and leaving an orphan
        # `}` after a surgical edit (172 syntax errors in the reported case).
        # Byte offset of brace_open_line's start within `source`, robust to
        # CRLF/CR: splitlines() strips the terminator, so a naive
        # `len(line) + 1` per preceding line drifts one char per CRLF/CR line
        # and can land the scan point inside an earlier symbol's body.
        _ke_lines = source.splitlines(keepends=True)
        line_start = sum(len(_ke_lines[k]) for k in range(brace_open_line))
        # find_brace_block_end scans from line_start via the literal-aware SSOT,
        # so the first REAL '{' it balances is the body opener — a default-arg
        # literal like fmt = "{}" no longer hijacks the scan point (the prior
        # `line_start + find('{')` could land mid-string, truncate the range,
        # and orphan the body on edit). Returns a 1-based inclusive line;
        # conservative fallback is the start line when no match is found.
        close_off = _find_closing_brace(source, line_start)
        if close_off == -1:
            # No matching ``}`` (genuinely unbalanced/malformed input). Return
            # the def line alone: the brace-balance gate in _post_edit_syntax_ok
            # then rejects the edit, which is the correct fail-closed behaviour
            # for malformed code (the prior trailing-skip walked the body via
            # indentation here, but on an unbalanced file any range is unreliable).
            return (i, i + 1)
        # Symbol boundary is EXACTLY the closing-brace line — no trailing-skip.
        # The brace scanner is literal-aware and returns the precise matching ``}``,
        # so anything past it is either trailing whitespace or the NEXT sibling's
        # leading doc comment. The old trailing-skip absorbed the next sibling's
        # comment into this symbol's range, so editing/deleting this symbol would
        # silently delete the sibling's doc (pre-existing latent corruption,
        # fallback-path only since the tree-sitter AST path returns a precise node
        # range that excludes attached comments). Returning the exact close-brace
        # line yields the precise symbol range with no absorption.
        close_line_1based = source[:close_off].count("\n") + 1
        return (i, close_line_1based)
    # No opening brace anywhere → indentation-based extent. (Python is handled
    # earlier by the AST path; this covers the rare genuinely brace-less case.)
    # Examine the candidate boundary line directly (lines[end]) and break
    # WITHOUT decrementing: end is an exclusive slice boundary. The prior
    # `lines[end-1]` + `end -= 1` form re-examined the def line itself on the
    # first iteration (it sits at exactly sym_indent) and returned (i, i) — an
    # empty range that on edit duplicated the symbol instead of replacing it.
    end = i + 1
    while end < len(lines):
        cs = lines[end].strip()
        if cs and not cs.startswith(("@", "#", "//")):
            nindent = len(lines[end]) - len(cs)
            if nindent <= sym_indent:
                break
        end += 1
    return (i, end)


def _apply_diff_to_source(source: str, diff: str) -> str:
    """Apply a unified diff to source string and return the result."""
    lines = source.splitlines(keepends=True)
    result_lines = list(lines)

    diff_lines = diff.splitlines()
    hunk_start = -1
    hunk_old_lines = []
    hunk_new_lines = []
    in_hunk = False

    for line in diff_lines:
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("\\ "):
            continue
        if line.startswith("@@"):
            if in_hunk and hunk_start >= 0:
                _apply_hunk(result_lines, hunk_start, hunk_old_lines, hunk_new_lines)

            parts = line.split(" ")
            if len(parts) >= 2:
                new_range = parts[2]
                if "," in new_range:
                    hunk_start = int(new_range[1:].split(",")[0]) - 1
                else:
                    hunk_start = int(new_range[1:]) - 1
            hunk_old_lines = []
            hunk_new_lines = []
            in_hunk = True
        elif in_hunk:
            if line.startswith("-"):
                hunk_old_lines.append(line[1:])
            elif line.startswith("+"):
                hunk_new_lines.append(line[1:])
            else:
                # Context line: unified diff prefixes these with a single space.
                # That marker MUST be stripped — otherwise every unchanged line in
                # the hunk accrues a leading space, corrupting indentation and
                # producing "unindent does not match any outer indentation level".
                ctx = line[1:] if line.startswith(" ") else line
                hunk_old_lines.append(ctx)
                hunk_new_lines.append(ctx)

    if in_hunk and hunk_start >= 0:
        _apply_hunk(result_lines, hunk_start, hunk_old_lines, hunk_new_lines)

    return "".join(result_lines)


def _apply_hunk(
    result_lines: list[str],
    hunk_start: int,
    old_lines: list[str],
    new_lines: list[str],
) -> None:
    """Replace old_lines with new_lines starting at hunk_start."""
    end = min(hunk_start + len(old_lines), len(result_lines))
    del result_lines[hunk_start:end]
    for i, nl in enumerate(new_lines):
        result_lines.insert(hunk_start + i, nl if nl.endswith("\n") else nl + "\n")


def _post_edit_syntax_ok(
    content: str, path: str, source: str = "", _source_net: Optional[int] = None
) -> bool:
    """Whether ``content`` is safe to write for ``path``.

    For PYTHON files, uses ``compile()`` to verify the candidate before it ever
    touches disk. Only ``_apply_ast_precise`` had this guard; the surgical and
    text fallbacks would otherwise write re-indented/spliced code that breaks
    Python syntax, then get caught by the post-write verifier and trigger a
    noisy ROLLBACK. Validating here turns that into a clean fall-through.

    For JS/TS/JSX/TSX files, uses ``node --check`` (if available) to catch syntax
    errors before write. For GO files, uses ``gofmt -e`` (if available).

    For brace languages without an inline compiler, verifies literal-aware brace
    balance. When ``source`` (the pre-edit content) is supplied, the check is
    RELATIVE — only edits that CHANGE the net brace count are rejected. The
    absolute ``net == 0`` form false-rejected any edit to a file with a
    pre-existing imbalance (a scanner limitation on brace-bearing raw/template
    strings, or genuinely broken code the user is mid-fixing). The corruption
    this gate exists to catch — an orphan ``}`` left by a bad symbol-range scan
    — always shifts the balance, so the delta is the precise signal.

    All other languages are passed through unchanged (they rely on post-write
    rollback or manual detection).
    """
    lid = LanguageId.from_path(path)
    if lid is LanguageId.PYTHON:
        try:
            compile_quiet(content, path, "exec")
            return True
        except SyntaxError:
            return False
    if lid in (LanguageId.JAVASCRIPT, LanguageId.TYPESCRIPT):
        node_path = shutil.which("node")
        if node_path:
            try:
                # node --check defaults to CJS for stdin; ESM files with
                # import/export need --input-type=module.  Try both modes
                # so we accept both CJS and ESM without false rejections.
                for extra_args in (["--input-type=module"], []):
                    r = subprocess.run(
                        [node_path, "--check", "--no-warnings", *extra_args],
                        input=content.encode("utf-8"),
                        capture_output=True,
                        timeout=10,
                    )
                    if r.returncode == 0:
                        return True
                return False
            except (subprocess.TimeoutExpired, OSError):
                return True  # fall through on infra failure
    if lid is LanguageId.GO:
        gofmt_path = shutil.which("gofmt")
        if gofmt_path:
            try:
                r = subprocess.run(
                    [gofmt_path, "-e"],
                    input=content.encode("utf-8"),
                    capture_output=True,
                    timeout=10,
                )
                return r.returncode == 0 and not r.stderr
            except (subprocess.TimeoutExpired, OSError):
                return True  # fall through on infra failure
    # Brace-delimited languages without an inline compiler (Kotlin/Rust/C/C++/
    # Java/Scala/Swift/C#): verify literal-aware brace balance so a symbol-range
    # scan that left an orphan `}` (or dropped a brace) is rejected before write
    # instead of corrupting the file. A real compiler is stronger, but this
    # catches the known corruption class in net terms and needs no toolchain.
    if lid in _BRACE_LANGUAGES_NO_COMPILER:
        new_net = net_brace_count(content)
        if source:
            # Relative delta: reject only edits that shift the net brace count.
            # See docstring — catches the orphan-brace corruption class while no
            # longer false-rejecting edits to files with a pre-existing imbalance.
            # ``_source_net`` (precomputed once by the caller for brace languages)
            # avoids re-scanning the same pre-edit content on each fallback tier.
            src_net = _source_net if _source_net is not None else net_brace_count(source)
            if new_net != src_net:
                return False
        elif new_net != 0:
            return False
    return True


# ── Public API ─────────────────────────────────────────────────────────────

def modify_symbol(
    file_path: str,
    symbol: str,
    code: str,
    repo_root: str = "",
) -> tuple[bool, str, str]:
    """Modify a symbol in a file deterministically.

    Main entry point. Fallback chain:
      1. AST precise (Python only)
      2. Surgical search/replace (any language)
      3. Simple text replacement (any language)

    Args:
        file_path: Relative or absolute path to the file.
        symbol: Symbol name (supports 'ClassName.method_name').
        code: New code for the symbol (full block or body-only).
        repo_root: Repository root for resolving relative paths.

    Returns:
        (success, diff_or_error, new_content_or_empty)
    """
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(repo_root, file_path)
    if not os.path.isfile(abs_path):
        return False, f"File not found: {file_path}", ""

    rel_path = os.path.relpath(abs_path, repo_root) if repo_root else file_path

    try:
        with open(abs_path, encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return False, f"Failed to read {file_path}: {e}", ""

    # ── Empty code guard (all strategies) ──
    if not code or not code.strip():
        return False, (
            f"Empty replacement code for symbol '{symbol}' in {rel_path} — "
            "cannot replace a symbol with nothing"
        ), ""

    # Normalize the dedented-first-line artifact once, before any strategy:
    # all three anchor the block's indentation on its first line.
    if _looks_like_full_symbol_block(code):
        code = _realign_dedented_leading_lines(code)

    # Set when a strategy produced output we refused to write because it would
    # have broken Python syntax — used to return an actionable error at the end.
    syntax_blocked = False

    # Pre-edit net brace count, computed ONCE for brace languages so the three
    # fallback tiers don't each re-scan the same source (net_brace_count is O(n)).
    source_net: Optional[int] = None
    if LanguageId.from_path(rel_path) in _BRACE_LANGUAGES_NO_COMPILER:
        source_net = net_brace_count(source)

    # ── Try 1: AST precise (Python only) ──
    if LanguageId.from_path(rel_path) is LanguageId.PYTHON:
        diff, mode = _apply_ast_precise(source, rel_path, symbol, code)
        if diff is not None:
            try:
                new_content = _apply_diff_to_source(source, diff)
            except Exception as e:
                return False, f"Diff apply failed after AST precise: {e}", ""
            if not _post_edit_syntax_ok(new_content, rel_path, source, _source_net=source_net):
                logger.info(
                    "AST precise diff reapplication produced invalid Python for %s "
                    "- trying surgical fallback", symbol,
                )
                syntax_blocked = True
            else:
                atomic_write_text(abs_path, new_content)
                return True, diff, new_content

        if mode == "skipped_compile_error":
            # AST precise assembled the block but the spliced file failed
            # compile — same fail-closed class as the post-apply checks below.
            syntax_blocked = True
        if mode not in ("skipped_no_new_body",) and not syntax_blocked:
            logger.info("AST precise failed (%s) for %s - trying surgical fallback", mode, symbol)

    # ── Try 2: Surgical edit (any language) ──
    sym_range = _find_symbol_line_range(source, symbol, rel_path)
    if sym_range is not None:
        sym_start, sym_end = sym_range
        diff = _apply_surgical_edit(source, rel_path, symbol, code, sym_start, sym_end)
        if diff is not None:
            try:
                new_content = _apply_diff_to_source(source, diff)
                if not _post_edit_syntax_ok(new_content, rel_path, source, _source_net=source_net):
                    logger.info(
                        "Surgical edit produced invalid Python for %s - trying text fallback",
                        symbol,
                    )
                    syntax_blocked = True
                else:
                    atomic_write_text(abs_path, new_content)
                    return True, diff, new_content
            except Exception as e:
                return False, f"Write failed after surgical edit: {e}", ""

    # ── Try 3: Text replacement fallback ──
    # NOTE: body-only mode is intentionally skipped here. Try3 replaces the
    # entire symbol range (def line + body) with the replacement text. If code
    # is body-only, this would strip the def/class line. Try2 already handles
    # body-only correctly; if it failed, Try3 cannot improve on it.
    if sym_range is not None and _looks_like_full_symbol_block(code):
        sym_start, sym_end = sym_range
        lines = source.splitlines(keepends=True)
        new_text = code if code.endswith("\n") else code + "\n"
        new_content = "".join([*lines[:sym_start], new_text, *lines[sym_end:]])
        if new_content != source:
            if not _post_edit_syntax_ok(new_content, rel_path, source, _source_net=source_net):
                logger.info("Text replacement produced invalid Python for %s", symbol)
                syntax_blocked = True
            else:
                try:
                    atomic_write_text(abs_path, new_content)
                    diff = _create_unified_diff(rel_path, source, new_content)
                    return True, diff, new_content
                except Exception as e:
                    return False, f"Write failed after text replacement: {e}", ""

    if syntax_blocked:
        foreign = (
            _trailing_foreign_stmt(code)
            if _looks_like_full_symbol_block(code) else None
        )
        if foreign:
            return False, (
                f"modify_symbol could not produce syntactically valid code for '{symbol}': "
                f"the replacement block extends past the symbol boundary — it contains "
                f"'{foreign}' at the symbol's own indent level after the '{symbol}' block. "
                f"Trim the block to the '{symbol}' definition only and retry; edits outside "
                "the symbol (e.g. blank lines between methods) belong to apply_patch."
            ), ""
        return False, (
            f"modify_symbol could not produce syntactically valid code for '{symbol}' "
            "(re-indentation/splice would break Python syntax). Use apply_patch instead."
        ), ""
    return False, "All strategies failed - could not locate or replace symbol", ""
