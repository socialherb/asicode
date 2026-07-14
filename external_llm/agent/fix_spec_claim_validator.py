"""Deterministic AST-based validator for FixSpec structural claims.

Validates that FixSpec claims about code structure (orphaned blocks,
duplicate methods, dead code locations) match the actual AST before
they influence execution.  No LLM calls — purely AST-driven.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .ast_cache import parse_cached_optional as _parse_ast

logger = logging.getLogger(__name__)

# ── Patterns that signal a structural-delete claim ──────────────────────────

# "Remove the orphaned code block at lines 7204-7222"
# "Delete the duplicate method body"
# "delete the orphaned code block", "remove duplicate method body"
_REMOVE_VERBS = ('delete', 'remove', 'eliminate', 'drop')
_REMOVE_NOUNS = ('block', 'method', 'function', 'body', 'line', 'lines', 'definition', 'section')


def _has_remove_pattern(text: str) -> bool:
    """Check if text contains a remove/delete claim pattern (keyword-based, no regex)."""
    lower = text.lower()
    return any(v in lower for v in _REMOVE_VERBS) and any(n in lower for n in _REMOVE_NOUNS)


def _has_orphan(text: str) -> bool:
    """Check if text mentions 'orphan' or 'orphaned'."""
    return 'orphan' in text.lower()


def _extract_line_range(text: str) -> Optional[tuple[int, int]]:
    """Extract (start, end) line numbers from text like 'lines 7204-7222'."""
    idx = text.lower().find('line')
    if idx < 0:
        return None
    # Scan for digit sequences after 'line(s)'
    rest = text[idx + 4:]
    digits_found: list[int] = []
    i = 0
    while i < len(rest) and len(digits_found) < 2:
        if rest[i].isdigit():
            start = i
            while i < len(rest) and rest[i].isdigit():
                i += 1
            digits_found.append(int(rest[start:i]))
        else:
            i += 1
    if len(digits_found) == 2:
        return (digits_found[0], digits_found[1])
    if len(digits_found) == 1:
        return (digits_found[0], digits_found[0])
    return None


def _extract_dotted_name(text: str) -> Optional[str]:
    """Find dotted name like ``foo.bar.baz`` in text (char scanner, no regex)."""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isalpha() or ch == '_':
            start = i
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] in '_.'):
                i += 1
            name = text[start:i]
            if '.' in name:
                parts = name.split('.')
                if all(p and (p[0].isalpha() or p[0] == '_') for p in parts):
                    return name
        else:
            i += 1
    return None


def _extract_method_ref(text: str) -> Optional[str]:
    """Extract symbol name after 'method' or 'function' keyword."""
    lower = text.lower()
    for kw in ('method', 'function'):
        idx = lower.find(kw)
        if idx < 0:
            continue
        rest = text[idx + len(kw):].lstrip()
        if rest.startswith('`'):
            rest = rest[1:]
        name_chars: list[str] = []
        for ch in rest:
            if ch.isalnum() or ch == '_':
                name_chars.append(ch)
            else:
                break
        if name_chars:
            return ''.join(name_chars)
    return None


def _has_file_like_path(text: str) -> bool:
    """Check if text contains a file-like path (word + '.' + 1-4 char extension)."""
    for i in range(len(text)):
        if text[i] == '.' and 0 < i < len(text) - 1:
            # Check extension: 1-4 alpha chars
            ext_start = i + 1
            ext_end = ext_start
            while ext_end < len(text) and text[ext_end].isalpha():
                ext_end += 1
            ext_len = ext_end - ext_start
            if 1 <= ext_len <= 4:
                # Check the part before dot has at least one word char
                j = i - 1
                while j >= 0 and (text[j].isalnum() or text[j] in '/._-'):
                    j -= 1
                if i - j > 1:  # at least one char before dot
                    return True
    return False


@dataclass
class ClaimValidationResult:
    """Result of validating a single FixSpec structural claim."""

    claim_text: str          # The original claim text from FixSpec
    symbol: str = ""         # Symbol name if extracted
    file_path: str = ""      # File path if referenced
    line_range: tuple[int, int] = (0, 0)  # (start, end) if line numbers found

    is_valid: bool = True              # Claim matches actual AST structure
    hallucinated: bool = False          # Flagged as likely hallucination
    confidence: float = 1.0            # 0.0 = certain hallucination, 1.0 = certain valid
    reason: str = ""                   # Why the claim was validated or rejected


@dataclass
class ValidatorSummary:
    """Aggregated result of FixSpec claim validation."""

    validated_targets: list[dict] = field(default_factory=list)
    suppressed_targets: list[dict] = field(default_factory=list)
    hallucinated_claims: list[ClaimValidationResult] = field(default_factory=list)
    valid_claims: list[ClaimValidationResult] = field(default_factory=list)
    narrative_warnings: list[str] = field(default_factory=list)
    has_hallucination: bool = False


# ── AST helpers ────────────────────────────────────────────────────────────


def _get_method_ranges(tree: ast.Module) -> list[tuple[str, int, int, int]]:
    """Return [(name, start_line, end_line, is_class_method), ...] from AST.

    Uses 1-based lines matching Python AST lineno.  Includes nested classes.
    """
    # Single-pass O(N): precompute the identity of every FunctionDef that is a
    # *direct* child of a ClassDef body (i.e. a method). The previous
    # implementation re-walked the entire tree once per function node (O(N²)),
    # which is quadratic on large files and runs per-claim.
    method_ids: set[int] = set()
    for cls_node in ast.walk(tree):
        if isinstance(cls_node, ast.ClassDef):
            for item in cls_node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_ids.add(id(item))

    ranges: list[tuple[str, int, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _name = node.name
            _start = node.lineno
            _end = getattr(node, "end_lineno", _start) or _start
            ranges.append((_name, _start, _end, 1 if id(node) in method_ids else 0))
    return ranges


def _get_class_ranges(tree: ast.Module, file_path: str) -> list[tuple[str, int, int]]:
    """Return [(class_name, start_line, end_line), ...]."""
    ranges: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _end = getattr(node, "end_lineno", node.lineno) or node.lineno
            ranges.append((node.name, node.lineno, _end))
    return ranges


def _read_file_lines(file_path: str) -> Optional[list[str]]:
    """Read file and return lines, or None on failure."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except (FileNotFoundError, OSError):
        return None


# ── Claim parsing ──────────────────────────────────────────────────────────


def _extract_symbol_name(text: str, known_targets: list[dict]) -> str:
    """Try to extract a symbol name from claim text."""
    # Check if any known target symbol is mentioned
    for t in known_targets:
        sym = t.get("symbol", "")
        if sym and sym in text:
            return sym
    # Fallback: look for dotted name pattern
    name = _extract_dotted_name(text)
    if name:
        return name
    # Look for simple name near "method" or "function"
    name = _extract_method_ref(text)
    if name:
        return name
    return ""


def _extract_file_path(text: str, known_targets: list[dict]) -> str:
    """Try to extract a file path from claim text."""
    for t in known_targets:
        fp = t.get("file", "")
        if fp and fp in text:
            return fp
    if _has_file_like_path(text):
        # Re-scan to find the actual match; for simplicity return first word before dot
        for i in range(len(text)):
            if text[i] == '.' and 0 < i < len(text) - 1:
                j = i - 1
                while j >= 0 and (text[j].isalnum() or text[j] in '/._-'):
                    j -= 1
                ext_end = i + 1
                while ext_end < len(text) and text[ext_end].isalpha():
                    ext_end += 1
                if 1 <= ext_end - (i + 1) <= 4:
                    return text[j + 1:ext_end]
    return ""


# ── Core validators ────────────────────────────────────────────────────────


def _validate_orphaned_block_claim(
    claim: str,
    file_path: str,
    line_range: tuple[int, int],
    repo_root: str,
) -> ClaimValidationResult:
    """Validate that an 'orphaned block at lines X-Y' claim matches AST reality.

    An orphaned block claim is valid only if the claimed line range falls
    *outside* any method/class definition in the file.  If the range is
    inside a valid method body, the claim is hallucinated.
    """
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(repo_root, file_path)

    lines = _read_file_lines(abs_path)
    if lines is None:
        return ClaimValidationResult(
            claim_text=claim,
            file_path=file_path,
            line_range=line_range,
            is_valid=False,
            hallucinated=False,  # can't verify — don't flag as hallucination
            confidence=0.0,
            reason=f"File not found: {abs_path}",
        )

    tree = _parse_ast("".join(lines))
    if tree is None:
        return ClaimValidationResult(
            claim_text=claim,
            file_path=file_path,
            line_range=line_range,
            is_valid=False,
            hallucinated=False,
            confidence=0.0,
            reason=f"AST parse failed for {abs_path}",
        )

    start, end = line_range
    if start < 1 or end > len(lines):
        return ClaimValidationResult(
            claim_text=claim,
            file_path=file_path,
            line_range=line_range,
            is_valid=False,
            hallucinated=True,
            confidence=0.95,
            reason=f"Line range ({start}-{end}) exceeds file length ({len(lines)} lines)",
        )

    # Collect all method/class line ranges
    method_ranges = _get_method_ranges(tree)
    class_ranges = _get_class_ranges(tree, abs_path)

    # Check: does the claimed range fall inside a valid method?
    inside_method = False
    containing_method = ""
    for mname, mstart, mend, _ in method_ranges:
        # Standard range-overlap: [start, end] intersects [mstart, mend].
        # The previous check (``mstart <= start <= mend or ...) only flagged a
        # claim whose START line was inside the method — a block starting before
        # the method and extending into it slipped through unflagged, contradicting
        # the docstring contract ("range falls outside any method").
        if start <= mend and end >= mstart:
            inside_method = True
            containing_method = mname
            break

    if inside_method:
        return ClaimValidationResult(
            claim_text=claim,
            file_path=file_path,
            line_range=line_range,
            is_valid=False,
            hallucinated=True,
            confidence=0.95,
            reason=(
                f"Claimed orphaned block at lines {start}-{end} is actually "
                f"inside method '{containing_method}' (lines {mstart}-{mend})"
            ),
        )

    # Check: does the claimed range fall inside a class?
    inside_class = False
    containing_class = ""
    for cname, cstart, cend in class_ranges:
        # Standard range-overlap (see method check above for rationale).
        if start <= cend and end >= cstart:
            inside_class = True
            containing_class = cname
            break

    if inside_class:
        return ClaimValidationResult(
            claim_text=claim,
            file_path=file_path,
            line_range=line_range,
            is_valid=False,
            hallucinated=True,
            confidence=0.90,
            reason=(
                f"Claimed orphaned block at lines {start}-{end} is inside "
                f"class '{containing_class}' (lines {cstart}-{cend})"
            ),
        )

    # Range is outside any method/class — plausible orphaned block
    return ClaimValidationResult(
        claim_text=claim,
        file_path=file_path,
        line_range=line_range,
        is_valid=True,
        hallucinated=False,
        confidence=0.85,
        reason=f"Lines {start}-{end} are outside any method/class — plausible orphaned code",
    )


def _validate_delete_target(
    target: dict,
    repo_root: str,
) -> Optional[ClaimValidationResult]:
    """Validate a FixSpec target whose 'fix' text claims delete/remove.

    If the target names a symbol, verify that the symbol actually exists
    at the claimed location (the *absence* of a symbol is *not* proof of
    a hallucination — the claim might be that it *should* be deleted).  We
    only flag cases where the structural claim is clearly impossible.
    """
    symbol = target.get("symbol", "")
    file_path = target.get("file", "")
    fix = target.get("fix", "")

    if not file_path or not symbol:
        return None

    abs_path = file_path if os.path.isabs(file_path) else os.path.join(repo_root, file_path)
    lines = _read_file_lines(abs_path)
    if lines is None:
        return None  # Can't verify — don't flag

    # Extract line range from fix text
    line_range = _extract_line_range(fix)

    if line_range:
        return _validate_orphaned_block_claim(fix, file_path, line_range, repo_root)

    # No line numbers but delete claim — check if symbol actually exists
    # This is a weak check: non-existence doesn't prove hallucination
    # (maybe it genuinely doesn't exist and that's the point of deletion).
    # Only flag if there's a structural contradiction.
    return None


# ── Public API ─────────────────────────────────────────────────────────────


def validate_fix_spec_claims(
    fix_spec: dict,
    repo_root: str = "",
) -> ValidatorSummary:
    """Validate all structural claims in a FixSpec against actual AST.

    Scans FixSpec targets and narrative text for delete/remove/orphan claims,
    cross-references with file AST, and returns validation results.

    Args:
        fix_spec: The FixSpec dict from LLM output.
        repo_root: Repository root path for resolving relative file paths.

    Returns:
        ValidatorSummary with validated/suppressed targets and hallucination flags.
    """
    if not fix_spec or not isinstance(fix_spec, dict):
        return ValidatorSummary()

    summary = ValidatorSummary()
    targets = fix_spec.get("targets", []) or []
    primary_issue = fix_spec.get("primary_issue", "") or ""
    narrative = fix_spec.get("analysis_narrative", "") or ""

    # Track which targets were flagged as hallucinated
    suppressed_indices: set[int] = set()

    # ── Step 1: Validate each target with delete/remove/orphan fix text ──
    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            summary.validated_targets.append(target)
            continue

        fix = target.get("fix", "") or ""
        symbol = target.get("symbol", "") or ""

        # Check if this is a structural-delete claim
        has_remove = _has_remove_pattern(fix)
        has_orphan = _has_orphan(fix)

        if has_remove or has_orphan:
            result = _validate_delete_target(target, repo_root)
            if result and result.hallucinated:
                suppressed_indices.add(idx)
                summary.hallucinated_claims.append(result)
                summary.has_hallucination = True
                logger.warning(
                    "[FIX_SPEC_HALLUCINATION] target '%s' hallucinated: %s",
                    symbol or "(unknown)", result.reason,
                )
                continue

            if result:
                summary.valid_claims.append(result)

        summary.validated_targets.append(target)

    # ── Step 2: Scan narrative/primary_issue for orphaned-block claims ──
    # Claims that match a remove/orphan pattern but lack enough location
    # info to validate (no line range, or no resolvable file path) are
    # recorded as narrative_warnings instead of being silently dropped —
    # surfacing "detected but unverifiable" structural claims so they are
    # not lost (the validator's own ValidatorSummary.narrative_warnings
    # field was previously declared but never populated).
    for _source, claim_text in (
        ("primary_issue", primary_issue),
        ("analysis_narrative", narrative),
    ):
        if not claim_text:
            continue
        if not _has_orphan(claim_text) and not _has_remove_pattern(claim_text):
            continue

        line_range = _extract_line_range(claim_text)
        if not line_range:
            summary.narrative_warnings.append(
                f"{_source}: structural-remove claim has no resolvable line range"
            )
            continue

        file_path = _extract_file_path(claim_text, targets)
        if not file_path:
            # Resolve via the symbol mentioned in the claim: prefer the known
            # target whose symbol appears in the claim text. The previous code
            # extracted this symbol but discarded it, blindly taking targets[0]
            # — wrong file when the claim references a later target's symbol.
            sym = _extract_symbol_name(claim_text, targets)
            if sym:
                for t in targets:
                    if t.get("symbol", "") == sym and t.get("file", ""):
                        file_path = t.get("file", "")
                        break
            # Last resort: fall back to the first target's file, if any.
            if not file_path and targets:
                file_path = targets[0].get("file", "")

        if not file_path:
            summary.narrative_warnings.append(
                f"{_source}: structural-remove claim has no resolvable file path"
            )
            continue

        result = _validate_orphaned_block_claim(
            claim_text, file_path, line_range, repo_root,
        )
        if result.hallucinated:
            summary.hallucinated_claims.append(result)
            summary.has_hallucination = True
            logger.warning(
                "[FIX_SPEC_HALLUCINATION] narrative claim hallucinated: %s",
                result.reason,
            )
        else:
            summary.valid_claims.append(result)

    # ── Step 3: Build suppressed targets list ──
    summary.suppressed_targets = [
        targets[i] for i in suppressed_indices
    ]
    summary.validated_targets = [
        t for i, t in enumerate(targets) if i not in suppressed_indices
    ]

    if summary.has_hallucination:
        logger.info(
            "[FIX_SPEC_HALLUCINATION] suppressed %d target(s), "
            "found %d hallucinated claim(s)",
            len(summary.suppressed_targets),
            len(summary.hallucinated_claims),
        )
    else:
        logger.debug(
            "[FIX_SPEC_HALLUCINATION] no hallucinated claims detected "
            "(%d targets validated)",
            len(targets),
        )

    return summary


# ── UnboundLocal detection for post-edit code ──────────────────────────────


def _walk_skip_comprehension_targets(node: ast.AST):
    """Walk all descendants, skipping comprehension loop variable targets.

    Comprehension variables (ListComp/SetComp/DictComp/GeneratorExp) do not
    leak into the enclosing scope in Python 3, so treating them as
    assignments would produce false-positive UnboundLocal warnings.
    """
    if isinstance(node, ast.comprehension):
        # Walk iter and ifs, but NOT target — scope-local to comprehension
        yield from _walk_skip_comprehension_targets(node.iter)
        for if_clause in node.ifs:
            yield from _walk_skip_comprehension_targets(if_clause)
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_skip_comprehension_targets(child)


def _collect_assigned_names(stmts: list[ast.stmt]) -> set[str]:
    """Return set of names that are assigned targets in a statement list."""
    names: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in _walk_skip_comprehension_targets(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                # Track attribute assignments like self.x = v
                pass  # skip attributes — only track simple local vars
    return names


def _collect_referenced_names(stmts: list[ast.stmt]) -> set[str]:
    """Return set of names that are read (Load context) in a statement list."""
    names: set[str] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    return names


def _get_pre_if_assignments(if_node: ast.If, tree: ast.Module) -> set[str]:
    """Return names that are assigned BEFORE this if/elif/else chain in the same scope.

    A variable initialized before the conditional is not at risk of UnboundLocalError
    even if only some branches reassign it.
    Includes function parameters (always defined) and assignments from enclosing
    scopes like `try`/`except` blocks.
    """
    pre_assigned: set[str] = set()
    for parent in ast.walk(tree):
        if hasattr(parent, "body") and isinstance(parent.body, list):
            if if_node in parent.body:
                idx = parent.body.index(if_node)
                before_stmts = parent.body[:idx]
                pre_assigned |= _collect_assigned_names(before_stmts)

                # Collect function parameters from the enclosing function
                _func_args = (
                    getattr(parent, "args", None)
                    if isinstance(parent, ast.AsyncFunctionDef)
                    or isinstance(parent, ast.FunctionDef)
                    else None
                )
                if _func_args:
                    for _arg in _func_args.args + _func_args.kwonlyargs:
                        pre_assigned.add(_arg.arg)
                    if _func_args.vararg:
                        pre_assigned.add(_func_args.vararg.arg)
                    if _func_args.kwarg:
                        pre_assigned.add(_func_args.kwarg.arg)
                break
    return pre_assigned


def _check_if_chain_unbound_locals(if_node: ast.If, tree: ast.Module) -> list[tuple[str, int]]:
    """Check an if/elif/else chain for variables read after but not defined in else.

    Returns list of (var_name, lineno) for variables at risk of UnboundLocalError.
    """
    # Collect branches: body + each elif body
    branch_bodies: list[list[ast.stmt]] = [if_node.body]
    current = if_node
    while isinstance(current.orelse, list) and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
        current = current.orelse[0]
        branch_bodies.append(current.body)

    else_body = current.orelse if isinstance(current.orelse, list) else []
    if not else_body:
        return []  # No else branch — variable might not be defined, but Python doesn't guarantee it

    # Collect names assigned in each branch
    branch_assigned = [_collect_assigned_names(b) for b in branch_bodies]
    else_assigned = _collect_assigned_names(else_body)

    # Names assigned in ALL non-else branches but NOT in else
    # (set.intersection of all branch bodies, minus else_assigned)
    if not branch_assigned:
        return []
    common_assigned: set[str] = branch_assigned[0]
    for ba in branch_assigned[1:]:
        common_assigned = common_assigned & ba
    common_assigned = common_assigned - else_assigned

    # Exclude names already initialized before the if/elif/else chain
    # (e.g. _x = None before conditional — no UnboundLocal risk)
    common_assigned -= _get_pre_if_assignments(if_node, tree)

    if not common_assigned:
        return []

    # ── Post-conditional usage check (false positive guard) ────────────────
    # Variables assigned inside if/elif branches but NOT in else are only
    # at risk if they're actually READ after the if/elif/else chain.
    # Common false positive: for-loop variables and helper variables defined
    # inside `if new_ops:` that are never read outside that branch — they
    # are scoped entirely within the if block and won't trigger
    # UnboundLocalError even if the else branch doesn't define them.
    _parent = None
    for _node in ast.walk(tree):
        if hasattr(_node, 'body') and isinstance(_node.body, list):
            if if_node in _node.body:
                _parent = _node
                break
    if _parent is not None:
        _idx = _parent.body.index(if_node)
        _after_stmts = _parent.body[_idx + 1:]
        if _after_stmts:
            # Collect all Name nodes referenced in post-conditional code
            _post_names: set[str] = set()
            for _stmt in _after_stmts:
                for _n in ast.walk(_stmt):
                    if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Load):
                        _post_names.add(_n.id)
            # Filter: only report variables that are ACTUALLY read after the chain
            common_assigned = {v for v in common_assigned if v in _post_names}

    return [(v, if_node.lineno) for v in sorted(common_assigned)]


def check_edit_unbound_locals(source: str, file_path: str = "") -> list[str]:
    """Check modified source for potential UnboundLocalError in if/elif/else chains.

    Returns human-readable warnings for each risky pattern found.

    Typical false-positive-free case detected:
      if cond:
          _x = val1
      elif cond2:
          _x = val2
      else:
          pass                    # ← _x NOT assigned here
      use(_x)                     # ← UnboundLocalError if else branch taken
    """
    tree = _parse_ast(source)
    if tree is None:
        return []

    warnings: list[str] = []
    _if_chain_nodes: set[int] = set()   # id()s of nodes covered by a chain walk

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and id(node) not in _if_chain_nodes:
            # Mark all nodes in this if/elif/else chain as covered
            _chain_ids = {id(node)}
            _cur = node
            while isinstance(_cur.orelse, list) and len(_cur.orelse) == 1 and isinstance(_cur.orelse[0], ast.If):
                _cur = _cur.orelse[0]
                _chain_ids.add(id(_cur))
            _if_chain_nodes |= _chain_ids

            results = _check_if_chain_unbound_locals(node, tree)
            for var_name, lineno in results:

                # Verify the variable is actually used AFTER the if chain
                # (not just defined within it)
                parent_body = None
                for parent in ast.walk(tree):
                    if hasattr(parent, "body") and isinstance(parent.body, list):
                        if node in parent.body:
                            parent_body = parent.body
                            break

                if parent_body is not None:
                    node_idx = parent_body.index(node)
                    after_stmts = parent_body[node_idx + 1:]
                    if after_stmts:
                        used_after = var_name in _collect_referenced_names(after_stmts)
                    else:
                        used_after = False
                else:
                    used_after = True  # conservative

                if used_after:
                    warnings.append(
                        f"{file_path}:{lineno} — variable '{var_name}' is assigned in "
                        f"some but not all if/elif/else branches and read after the "
                        f"conditional — risk of UnboundLocalError"
                    )

    return warnings
