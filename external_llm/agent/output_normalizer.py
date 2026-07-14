"""P11: Output Normalizer — deterministic cleanup for LLM-generated code.

Only semantic-neutral changes: import dedup, whitespace normalization.
Runs right before contract_verification (duplicate_definitions_check) final state
validation to eliminate verifier noise.

Design principles:
- deterministic cleanup only (never modify logic)
- import dedup + whitespace normalization
- syntax-validate; keep original on failure
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ..languages import LanguageId
from ..common.atomic_io import atomic_write_text

# ── Task drift detection ───────────────────────────────────────────────────
# Detects when the agent's plan operations drift from the original spec.
# Runs after plan creation, before execution.


@dataclass
class DriftReport:
    """Report of task drift between spec and plan operations."""
    has_drift: bool = False
    untargeted_files: list[str] = field(default_factory=list)

    drifted_kinds: list[str] = field(default_factory=list)
    severity: str = "none"  # "none", "low", "medium", "high"
    summary: str = ""


def detect_task_drift(
    spec_target_files: Optional[list[str]],
    spec_target_symbols: Optional[list[str]],
    plan_operations: list,
    request_type: str = "",
    reference_files: Optional[list[str]] = None,
    reference_symbols: Optional[list[str]] = None,
) -> DriftReport:
    """Detect task drift between spec targets and plan operations.

    Checks:
    1. Untargeted files: operations touch files not in spec.target_files
       (lenient: allow if the file is in new_files or is a test file)
    2. Drifted kinds: operation kinds inconsistent with request_type
       (e.g., DELETE_SYMBOL_RANGE on a "fix" request)

    Note: target_files without ops are not considered drift —
    the Planner may judge "no change needed" and skip op generation.
    Symbol-level drift validation is handled by candidate_ranker's spec_alignment quality gate.

    Returns a DriftReport with severity classification.
    """
    report = DriftReport()

    if not plan_operations:
        report.has_drift = False
        report.severity = "none"
        report.summary = "no operations to check"
        return report

    # Normalize all paths to normpath for reliable comparison.
    # Prevents false-positive drift when spec has absolute paths
    # but ops have relative paths (or vice versa).
    def _norm(p: str) -> str:
        return os.path.normpath(p)

    target_files = {_norm(tf) for tf in (spec_target_files or [])}

    # Collect operation targets
    ops_files: set = set()
    ops_kinds: set = set()
    for op in plan_operations:
        _path = getattr(op, "path", None) or ""
        _norm_path = _norm(_path) if _path else ""
        _kind = getattr(op, "kind", None)
        if _norm_path:
            ops_files.add(_norm_path)
        if _kind is not None:
            _kind_str = _kind.value if hasattr(_kind, "value") else str(_kind)
            ops_kinds.add(_kind_str)

    # 1. Untargeted files (ops touch files not in spec)
    # Allow READ_SYMBOL ops as they are analysis, not modification
    _WRITE_KINDS = {
        "MODIFY_SYMBOL", "INSERT_AFTER_SYMBOL", "INSERT_AFTER_LINE",
        "DELETE_SYMBOL_RANGE",
        "ANCHOR_EDIT", "INSERT_IMPORT", "REMOVE_IMPORT", "REMOVE_IMPORT_NAME", "CREATE_FILE",
        "OVERWRITE_FILE", "REPLACE_FILE", "DELETE_FILE", "SUMMARIZE_ANALYSIS",
    }
    _untargeted_write_files: set = set()
    for op in plan_operations:
        _kind_str = getattr(op, "kind", None)
        _kind_str = _kind_str.value if hasattr(_kind_str, "value") else str(_kind_str)
        if _kind_str not in _WRITE_KINDS:
            continue
        _path = getattr(op, "path", None) or ""
        _norm_path = _norm(_path) if _path else ""
        if _norm_path and _norm_path not in target_files:
            _untargeted_write_files.add(_norm_path)

    if _untargeted_write_files:
        report.untargeted_files = sorted(_untargeted_write_files)

    # 2. (removed) Untargeted symbols — symbol-level drift detection is
    # handled by candidate_ranker's _compute_spec_alignment + quality gate.
    # TASK_DRIFT now only checks file-level and kind-level drift.

    # 3. Drifted kinds: detect DELETE operations on non-delete requests
    _DELETE_KINDS = {"DELETE_SYMBOL_RANGE", "DELETE_FILE"}
    _has_delete = bool(ops_kinds & _DELETE_KINDS)
    _is_edit_request = request_type in ("modify", "edit", "fix", "refactor", "", "unknown")
    if _has_delete and _is_edit_request:
        _delete_kinds_found = sorted(ops_kinds & _DELETE_KINDS)
        report.drifted_kinds = _delete_kinds_found

    # ── Severity classification ────────────────────────────────────────────
    _severity_score = 0
    if report.untargeted_files:
        _severity_score += len(report.untargeted_files) * 2
    if report.drifted_kinds:
        _severity_score += 3
    if _severity_score >= 5:
        report.severity = "high"
    elif _severity_score >= 3:
        report.severity = "medium"
    elif _severity_score >= 1:
        report.severity = "low"
    else:
        report.severity = "none"

    report.has_drift = _severity_score > 0

    # ── Summary ────────────────────────────────────────────────────────────
    _parts = []
    if report.untargeted_files:
        _parts.append(f"untargeted_files={report.untargeted_files}")
    if report.drifted_kinds:
        _parts.append(f"drifted_kinds={report.drifted_kinds}")

    report.summary = "; ".join(_parts) if _parts else "no drift detected"

    return report

logger = logging.getLogger(__name__)


def dedup_imports(source: str) -> str:
    """Remove duplicate import lines while preserving order.

    Three-phase cleanup:
    1. Remove exact-match duplicate lines.
    2. Merge same-module from-imports (e.g. ``from X import Y`` + ``from X import Z``
       → ``from X import Y, Z``), preserving position of the first occurrence.
    3. Deduplicate names within a single import line (e.g. ``import os, os``).

    Uses ast.parse() to identify top-level import boundaries accurately,
    avoiding fragility of string-based multi-line import detection.
    """
    lines = source.splitlines(keepends=True)

    # Primary path: use AST to find top-level import nodes and their
    # line ranges.  This correctly handles multi-line imports and avoids
    # the fragile string-based opener detection used by the fallback.
    try:
        tree = ast.parse(source)
        top_imports: list[tuple[int, int, str]] = []
        for _node in tree.body:
            if isinstance(_node, (ast.Import, ast.ImportFrom)):
                start = _node.lineno - 1
                end = _node.end_lineno or (start + 1)
                text = "".join(lines[start:end]).strip()
                top_imports.append((start, end, text))

        if top_imports:
            seen: set = set()
            skip_lines: set = set()
            for start, end, text in top_imports:
                if text in seen:
                    for i in range(start, end):
                        skip_lines.add(i)
                else:
                    seen.add(text)

            # Phase 2: Merge same-module from-imports.
            # After exact-match dedup, group remaining ImportFrom nodes by
            # module and merge their name lists into a single import statement.
            # Position is preserved at the first occurrence's line.
            try:
                _deduped_source = "".join(
                    line for i, line in enumerate(lines) if i not in skip_lines
                )
                _merge_tree = ast.parse(_deduped_source)
                _merge_lines = _deduped_source.splitlines(keepends=True)

                # Collect all ImportFrom nodes grouped by module.
                # Also collect Import nodes with multiple aliases for name dedup.
                _from_nodes: dict[tuple[str, int], list] = {}  # (module, level) → [(start, end, names, level)]
                _import_merge: dict[int, list] = {}  # lineno → [alias names]
                for _node in _merge_tree.body:
                    if isinstance(_node, ast.ImportFrom):
                        _mod = _node.module or ""
                        _level = getattr(_node, "level", 0) or 0
                        _mod_key = (_mod, _level)
                        if _mod_key not in _from_nodes:
                            _from_nodes[_mod_key] = []
                        _start = _node.lineno - 1
                        _end = _node.end_lineno or (_start + 1)
                        _names = [a.name for a in _node.names]
                        _from_nodes[_mod_key].append((_start, _end, _names, _level))
                    elif isinstance(_node, ast.Import):
                        if len(_node.names) > 1:
                            # import X, Y, X → dedup names within single line
                            _all_names = [a.name for a in _node.names]
                            if len(_all_names) != len(set(_all_names)):
                                _import_merge[_node.lineno - 1] = _all_names

                _merge_skip: set = set()
                _replacements: dict[int, str] = {}  # target_lineno → merged_line

                for (_mod, _merge_level), _entries in _from_nodes.items():
                    if len(_entries) <= 1:
                        continue
                    # Merge all names from all entries of this module
                    _all_names: list = []
                    _first_start = _entries[0][0]
                    for _entry in _entries:
                        _all_names.extend(_entry[2])  # _names at index 2
                        _entry_start = _entry[0]
                        _entry_end = _entry[1]
                        if _entry_start != _first_start:
                            for _i in range(_entry_start, _entry_end):
                                _merge_skip.add(_i)
                    # Deduplicate names while preserving order
                    _seen_names: set = set()
                    _unique_names: list = []
                    for _n in _all_names:
                        if _n not in _seen_names:
                            _seen_names.add(_n)
                            _unique_names.append(_n)
                    # Sort for deterministic output (underscore last)
                    _unique_names.sort(key=lambda n: (n.startswith("_"), n))
                    # AST stores module as "constants" for both "constants" and ".constants";
                    # the level encodes dot depth: 0=absolute, 1=., 2=.., etc.
                    # _merge_level comes from the dict key (tuple[module, level]).
                    if _merge_level > 0:
                        _mod_prefix = "." * _merge_level + _mod
                    else:
                        _mod_prefix = _mod
                    _merged_line = f"from {_mod_prefix} import {', '.join(_unique_names)}"
                    _replacements[_first_start] = _merged_line

                # Dedup names within single import X, Y lines
                for _lineno, _names in _import_merge.items():
                    _seen_imp: set = set()
                    _unique_imp: list = []
                    for _n in _names:
                        if _n not in _seen_imp:
                            _seen_imp.add(_n)
                            _unique_imp.append(_n)
                    if len(_unique_imp) < len(_names):
                        _replacements[_lineno] = f"import {', '.join(_unique_imp)}"

                if _merge_skip or _replacements:
                    _result_lines: list = []
                    for _i, _line in enumerate(_merge_lines):
                        if _i in _merge_skip:
                            continue
                        if _i in _replacements:
                            _result_lines.append(_replacements[_i] + _line[len(_line.rstrip('\n')):])
                        else:
                            _result_lines.append(_line)
                    result = "".join(_result_lines)
                else:
                    result = _deduped_source
            except SyntaxError:
                # If re-parse fails, fall back to exact-match dedup only
                result = _deduped_source

            return result
    except SyntaxError:
        pass

    # Fallback: string-based heuristic for partial / invalid files
    seen: set = set()
    out: list = []

    for line in lines:
        stripped = line.strip()

        if (stripped.startswith("import ") or stripped.startswith("from ")) \
                and line[:1] not in (' ', '\t'):
            if not stripped.endswith("("):
                if stripped in seen:
                    continue
                seen.add(stripped)

        out.append(line)

    result = "".join(out)
    return result


def normalize_whitespace(source: str) -> str:
    """Clean up excessive blank lines and trailing whitespace.

    Rules:
    - Max 2 consecutive blank lines
    - Remove trailing whitespace from each line
    - Ensure file ends with exactly one newline
    """
    lines = source.splitlines()
    out: list = []
    blank_run = 0

    for line in lines:
        cleaned = line.rstrip()
        if cleaned == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(cleaned)

    # Remove trailing blank lines, then add exactly one newline
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out) + "\n" if out else ""


def normalize_python_source(source: str) -> str:
    """Apply all normalizations to a Python source file.

    Order matters: dedup imports first, then whitespace.
    """
    result = source
    result = dedup_imports(result)
    result = normalize_whitespace(result)
    return result


def normalize_file(
    file_path: str,
    repo_root: str = "",
) -> tuple[bool, Optional[str]]:
    """Normalize a single Python file.

    Returns (changed, error_or_None).
    If normalization introduces a syntax error, the original is restored.
    """
    abs_path = os.path.join(repo_root, file_path) if not os.path.isabs(file_path) else file_path
    if LanguageId.from_path(abs_path) is not LanguageId.PYTHON or not os.path.isfile(abs_path):
        return False, None

    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            before = f.read()
    except Exception as e:
        return False, f"read error: {e}"

    after = normalize_python_source(before)

    if after == before:
        return False, None  # no changes needed

    # Safety: verify syntax is preserved
    try:
        ast.parse(after)
    except SyntaxError as e:
        logger.warning(
            "P11: normalization would break syntax in %s: %s — skipping",
            file_path, e,
        )
        return False, f"syntax error after normalization: {e}"

    # Write normalized content
    try:
        atomic_write_text(abs_path, after)
    except Exception as e:
        return False, f"write error: {e}"

    return True, None


def normalize_modified_files(
    modified_files: list[str],
    repo_root: str = "",
) -> list[str]:
    """Normalize all modified Python files.

    Returns list of files that were actually changed.
    """
    normalized: list = []

    for rel_path in modified_files:
        if LanguageId.from_path(rel_path) is not LanguageId.PYTHON:
            continue

        changed, error = normalize_file(rel_path, repo_root)
        if changed:
            logger.info("P11: normalized %s (import dedup + whitespace)", rel_path)
            normalized.append(rel_path)
        elif error:
            logger.debug("P11: skipped %s: %s", rel_path, error)

    return normalized
