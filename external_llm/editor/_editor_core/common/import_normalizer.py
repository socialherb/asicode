"""import_normalizer.py — Deterministic typing import normalizer.

Problem: LLM-generated code has unstable 'from typing import ...' lines.
  - Sometimes over-generates (Any, Union added but unused)
  - Sometimes under-generates (Dict used but not imported → F821)
  - Both trigger cs-repair / replan cycles unnecessarily

Solution: After each successful edit, scan the file with AST and
  rewrite the 'from typing import ...' block to match actual symbol usage.
  Fully deterministic — no LLM involvement.

Scope intentionally limited to `typing` module only:
  - typing symbols have exactly one source (typing module)
  - usage is 100% AST-detectable
  - safe to strip/rewrite without graph lookup

Cross-module project imports (from external_llm.agent.foo import Bar) are
handled by the existing _strip_hallucinated_imports + F821 repair chain.
"""
from __future__ import annotations

import os
from typing import Optional

from external_llm.common.atomic_io import atomic_write_text
from external_llm.languages import LanguageId

# ── F821-protected imports cache ──────────────────────────────────────
# When F821 auto-repair inserts a typing import (e.g. `from typing import
_f821_protected: dict[str, set[str]] = {}




def _collect_f821_protected_from_source(source: str) -> set[str]:
    """Extract F821-protected typing symbol names from ``# f821-protected`` comments.

    Scans each ``from typing import ...`` line for a trailing
    ``# f821-protected`` or ``# f821-protected:<name>`` comment marker.
    Returns the set of import names on matching lines.

    Survives process restarts because markers are written to the file
    itself by ``mark_f821_protected()``, not held in memory.
    """
    _protected: set[str] = set()
    for _line in source.splitlines():
        _line_stripped = _line.strip()
        if not _line_stripped.startswith('from typing import '):
            continue
        # Check if the line has a f821-protected marker in a comment
        _comment_start = _line_stripped.find('#')
        if _comment_start == -1:
            continue
        _comment = _line_stripped[_comment_start + 1:].strip()
        if not (_comment == 'f821-protected' or _comment.startswith('f821-protected:')):
            continue
        # Extract names from the import statement (before the comment)
        _import_part = _line_stripped[:_comment_start].strip()
        if not _import_part.startswith('from typing import '):
            continue
        _rest = _import_part[len('from typing import '):]
        for _name in _rest.split(','):
            _name = _name.strip()
            if _name:
                _protected.add(_name)
    return _protected


def mark_f821_protected(file_path: str, name: str) -> None:
    """Mark a typing import name as F821-verified for a given file.

    Writes a ``# f821-protected`` comment marker to the existing typing import
    line in the file, so the protection survives process restarts.
    In-memory cache is also updated for same-process reads.

    Callers (F821 auto-repair) should invoke this after successfully
    inserting a ``from typing import <name>`` line so the import
    normalizer preserves it.
    """
    _abs = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)
    _f821_protected.setdefault(_abs, set()).add(name)

    # ── Persist to file as comment marker (survives process restart) ──
    try:
        with open(_abs, encoding='utf-8') as _fh:
            _lines = _fh.readlines()
    except OSError:
        return

    _modified = False
    for _i, _line in enumerate(_lines):
        if _line.strip().startswith('from typing import ') and name in _line:
            if '# f821-protected' not in _line:
                _stripped = _line.rstrip('\n').rstrip()
                _lines[_i] = _stripped + '  # f821-protected\n'
                _modified = True
            break

    if _modified:
        try:
            atomic_write_text(_abs, "".join(_lines))
        except OSError as _exc:
            logger.warning(
                "mark_f821_protected: failed to persist marker to %s: %s "
                "(in-memory cache updated; marker may not survive restart)",
                file_path, _exc,
            )


import ast
import logging

logger = logging.getLogger(__name__)

# ── Known typing symbols ─────────────────────────────────────────────────────
# Covers Python 3.8+ stdlib typing. Deliberately conservative — only symbols
# that appear in `from typing import X` style imports.

_TYPING_SYMBOLS: set[str] = {
    # Generics
    "Any", "Callable", "ClassVar", "Dict", "Final", "FrozenSet",
    "Generator", "Generic", "Iterable", "Iterator", "List", "Literal",
    "Mapping", "MutableMapping", "MutableSequence", "Optional", "Protocol",
    "Sequence", "Set", "Tuple", "Type", "TypeVar", "Union",
    # Utilities
    "Annotated", "TypedDict", "NamedTuple", "cast", "overload",
    "runtime_checkable", "TYPE_CHECKING", "get_type_hints",
    # Python 3.10+
    "ParamSpec", "Concatenate", "TypeAlias", "Never",
}


def collect_typing_usage(source: str) -> set[str]:
    """Return typing symbols actually used in source, excluding import lines.

    Handles:
    - Direct name references (ast.Name nodes): `x: Dict[str, int]`
    - typing.X attribute references: `typing.Optional[str]`
    - String annotations in annotation-only positions:
        `def foo(x: "Dict[str, Any]") -> "Optional[str]"`
        `x: "List[int]" = []`

    String constants are scanned ONLY when they appear in annotation context
    (function argument annotations, return annotations, variable annotations).
    This prevents docstrings from triggering false-positive detections, e.g.:
        '''Returns Optional value''' → does NOT mean typing.Optional is needed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    # ── Pass 1: Collect line numbers of string constants that are annotations.
    # We mark these by their node id (using id() of the ast.Constant) so we
    # can skip all other string constants in Pass 2.
    annotation_node_ids: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Argument annotations
            all_args = (
                node.args.args
                + node.args.posonlyargs
                + node.args.kwonlyargs
            )
            for arg in all_args:
                if arg.annotation and isinstance(arg.annotation, ast.Constant):
                    annotation_node_ids.add(id(arg.annotation))
            if node.args.vararg and node.args.vararg.annotation:
                if isinstance(node.args.vararg.annotation, ast.Constant):
                    annotation_node_ids.add(id(node.args.vararg.annotation))
            if node.args.kwarg and node.args.kwarg.annotation:
                if isinstance(node.args.kwarg.annotation, ast.Constant):
                    annotation_node_ids.add(id(node.args.kwarg.annotation))
            # Return annotation
            if node.returns and isinstance(node.returns, ast.Constant):
                annotation_node_ids.add(id(node.returns))
        elif isinstance(node, ast.AnnAssign):
            # Variable annotation: `x: "List[int]" = []`
            if isinstance(node.annotation, ast.Constant):
                annotation_node_ids.add(id(node.annotation))

    # ── Pass 2: Collect typing symbol usages ─────────────────────────────
    used: set[str] = set()

    for node in ast.walk(tree):
        # Skip import lines themselves
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        # Direct Name usage: List, Dict, Optional, ...
        if isinstance(node, ast.Name):
            if node.id in _TYPING_SYMBOLS:
                used.add(node.id)

        # Attribute: typing.List, typing.Optional (less common but valid)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "typing":
                if node.attr in _TYPING_SYMBOLS:
                    used.add(node.attr)

        # String annotations — ONLY if in annotation context (not docstrings).
        # e.g. `def foo(x: "Dict[str, Any]") -> "Optional[str]":` is valid.
        # e.g. `"""Returns Optional value"""` is NOT an annotation.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in annotation_node_ids:
                for sym in _TYPING_SYMBOLS:
                    sv = node.value
                    idx = sv.find(sym)
                    while idx != -1:
                        before = idx == 0 or not (sv[idx-1].isalnum() or sv[idx-1] == '_')
                        after = idx + len(sym) >= len(sv) or not (sv[idx+len(sym)].isalnum() or sv[idx+len(sym)] == '_')
                        if before and after:
                            used.add(sym)
                            break
                        idx = sv.find(sym, idx + 1)

    return used


def normalize_typing_imports(file_path: str) -> bool:
    """Rewrite 'from typing import ...' in file to match actual usage.

    Idempotent — if imports already match usage, file is unchanged.
    Returns True if the file was modified.

    Only modifies files where:
    - At least one existing 'from typing import ...' line is present, OR
    - Typing symbols are used but no import exists (pre-empts F821)
    """
    if LanguageId.from_path(file_path) is not LanguageId.PYTHON:
        return False

    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError:
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    # ── Step 1: Find existing 'from typing import ...' nodes ─────────────
    typing_nodes: list = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "typing"
            and node.names  # not "from typing import *"
            and not any(a.name == "*" for a in node.names)
        ):
            typing_nodes.append(node)

    existing_names: set[str] = set()
    for node in typing_nodes:
        for alias in node.names:
            existing_names.add(alias.asname or alias.name)

    # ── Step 1b: Collect names already imported from non-typing sources ──
    # If a symbol like `Mapping` is already imported via `from collections
    already_imported_non_typing: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module != "typing":
            for alias in node.names:
                already_imported_non_typing.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                already_imported_non_typing.add(alias.asname or alias.name.split(".")[0])

    # ── Step 2: Collect actual usage, minus already-covered symbols ───────
    # Symbols provided by a non-typing import must NOT be added to the
    # `from typing import ...` line — they are already in scope.
    needed: set[str] = collect_typing_usage(source) - already_imported_non_typing

    # ── Step 2b: Preserve F821-protected imports ────────────────────────
    # F821 auto-repair may have inserted typing imports that the AST pass
    # cannot detect (e.g. symbols used only in deferred string annotations).
    # Those names are verified by ruff at runtime — do NOT strip them.
    # Protection markers are written as ``# f821-protected`` comments on the
    # import line itself, surviving process restarts.
    _protected = _collect_f821_protected_from_source(source)
    if _protected:
        needed |= _protected

    # ── Step 3: Skip if already correct ──────────────────────────────────
    if needed == existing_names:
        return False

    if not needed and not existing_names:
        return False

    logger.info(
        "normalize_typing_imports %s: existing=%s needed=%s",
        file_path,
        sorted(existing_names), sorted(needed),
    )

    # ── Step 4: Rewrite source ───────────────────────────────────────────
    lines = source.splitlines(keepends=True)

    # Collect line indices of all existing typing import lines (0-indexed).
    # A node may span multiple lines for parenthesized imports.
    typing_line_indices: set[int] = set()
    for node in typing_nodes:
        start = node.lineno - 1          # 0-indexed
        end = getattr(node, "end_lineno", node.lineno) - 1
        for i in range(start, end + 1):
            typing_line_indices.add(i)

    # Build new import line (empty string if no symbols needed)
    # Preserve ``# f821-protected`` marker when any protected name survives.
    _has_protected = bool(_collect_f821_protected_from_source(source) & needed)
    new_import_line = (
        "from typing import " + ", ".join(sorted(needed))
        + ("  # f821-protected" if _has_protected else "") + "\n"
        if needed else ""
    )

    # Replace: insert new line at first typing import position,
    # then blank out the rest.
    first_idx: Optional[int] = (
        min(typing_line_indices) if typing_line_indices else None
    )

    new_lines = list(lines)
    replaced = False

    if first_idx is not None:
        for idx in sorted(typing_line_indices):
            if idx == first_idx and not replaced and new_import_line:
                new_lines[idx] = new_import_line
                replaced = True
            else:
                new_lines[idx] = ""   # blank out extra/old lines

    # No existing import but symbols needed → find first non-docstring import
    # line and insert before it.
    if not replaced and new_import_line:
        insert_at = _find_first_import_line(tree, lines)
        new_lines.insert(insert_at, new_import_line)

    new_source = "".join(new_lines)

    # Validate before writing
    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        logger.warning(
            "normalize_typing_imports: parse error after rewrite of %s: %s — skipping",
            file_path, exc,
        )
        return False

    if new_source == source:
        return False

    try:
        atomic_write_text(file_path, new_source)
        logger.info(
            "normalize_typing_imports: updated %s (%s → %s)",
            file_path, sorted(existing_names), sorted(needed),
        )
        return True
    except OSError as exc:
        logger.warning("normalize_typing_imports: write failed for %s: %s", file_path, exc)
        return False


def _find_first_import_line(tree: ast.Module, lines: list) -> int:
    """Return 0-indexed line number of first import statement, or 0."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return node.lineno - 1
    return 0
