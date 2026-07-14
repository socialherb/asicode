"""
Repair engine mixin for OperationExecutor.

Extracted from operation_executor.py — provides all self-repair methods as a
mixin class so OperationExecutor can inherit them without bloating the main file.

All methods use `self` which resolves to the OperationExecutor instance via MRO.
"""
from __future__ import annotations

import ast as _ast
import logging

# import re — replaced with native string ops (6 patterns); local imports in methods only
from dataclasses import dataclass
from typing import Optional

from .operation_models import (
    EditInstruction,
    EditInstructionKind,
    FailureClass,
)

logger = logging.getLogger(__name__)


_SYMBOL_LEVEL_EDIT_KINDS = frozenset({
    EditInstructionKind.REPLACE_SYMBOL_BODY,
    EditInstructionKind.INSERT_AFTER_SYMBOL,
    EditInstructionKind.PATCH_SYMBOL,
    EditInstructionKind.SURGICAL_EDIT,
    EditInstructionKind.AST_OP,
})
_EXISTING_SYMBOL_BODY_EDIT_KINDS = frozenset({
    EditInstructionKind.REPLACE_SYMBOL_BODY,
    EditInstructionKind.SURGICAL_EDIT,
    EditInstructionKind.AST_OP,
    EditInstructionKind.AST_DIRECT_BODY,
})


# Repair handlers that change `kind` must go through `_select_next_repr_for_repair`
# to stay inside the selector's allowed/forbidden set.
_REPR_AST_OP             = "ast_op"
_REPR_AST_DIRECT_BODY    = "ast_direct_body"
_REPR_SURGICAL_EDIT      = "surgical_edit"
_REPR_REPLACE_SYM_BODY   = "replace_symbol_body"
_REPR_FORCE_FULL_REWRITE = "force_full_rewrite"

_INSTRUCTION_KIND_TO_REPR: dict[EditInstructionKind, str] = {
    EditInstructionKind.AST_OP:              _REPR_AST_OP,
    EditInstructionKind.AST_DIRECT_BODY:    _REPR_AST_DIRECT_BODY,
    EditInstructionKind.SURGICAL_EDIT:       _REPR_SURGICAL_EDIT,
    EditInstructionKind.REPLACE_SYMBOL_BODY: _REPR_REPLACE_SYM_BODY,
}

_REPR_TO_INSTRUCTION_KIND: dict[str, EditInstructionKind] = {
    _REPR_AST_OP:              EditInstructionKind.AST_OP,
    _REPR_AST_DIRECT_BODY:    EditInstructionKind.AST_DIRECT_BODY,
    _REPR_SURGICAL_EDIT:       EditInstructionKind.SURGICAL_EDIT,
    _REPR_REPLACE_SYM_BODY:    EditInstructionKind.REPLACE_SYMBOL_BODY,
    _REPR_FORCE_FULL_REWRITE:  EditInstructionKind.REPLACE_SYMBOL_BODY,
}


@dataclass(frozen=True)
class SelectorRetryContext:
    """Snapshot of the executor's representation selector decision."""

    current_representation: str
    selector_first_choice: str
    forbidden_representations: frozenset
    fallback_representations: tuple
    semantic_change_family: str
    control_path: str

    def has_selector(self) -> bool:
        return self.control_path == "selector_native" and bool(self.selector_first_choice)


def _extract_selector_retry_context(instruction: EditInstruction) -> SelectorRetryContext:
    meta = instruction.metadata or {}
    psd = meta.get("patch_strategy_decision") or {}
    rep_sel = psd.get("representation_selection") or {}

    current = _INSTRUCTION_KIND_TO_REPR.get(instruction.kind, "")
    first = (meta.get("selector_first_choice") or "").strip()
    forbidden = tuple(
        meta.get("forbidden_representations")
        or rep_sel.get("forbidden_representations")
        or ()
    )
    fallback = tuple(
        rep_sel.get("fallback_representations") or ()
    )
    family = (meta.get("semantic_change_family")
              or rep_sel.get("semantic_change_family") or "").strip()
    control_path = psd.get("control_path") or ""
    return SelectorRetryContext(
        current_representation=current,
        selector_first_choice=first,
        forbidden_representations=frozenset(forbidden),
        fallback_representations=fallback,
        semantic_change_family=family,
        control_path=str(control_path),
    )


# Per-failure-class representation preference (best-first).
_FAILURE_CLASS_REPRESENTATION_PREFERENCE: dict[str, tuple[str, ...]] = {
    "no_diff_generated":             (_REPR_AST_DIRECT_BODY, _REPR_REPLACE_SYM_BODY, _REPR_FORCE_FULL_REWRITE),
    "search_not_found":              (_REPR_AST_DIRECT_BODY, _REPR_REPLACE_SYM_BODY, _REPR_FORCE_FULL_REWRITE),
    "placement_violation":           (_REPR_AST_OP, _REPR_AST_DIRECT_BODY, _REPR_SURGICAL_EDIT),
    "semantic_verification_failed":  (_REPR_AST_DIRECT_BODY, _REPR_SURGICAL_EDIT),
    "symbol_missing_after_edit":     (_REPR_AST_DIRECT_BODY, _REPR_REPLACE_SYM_BODY),
    "signature_changed":             (_REPR_REPLACE_SYM_BODY, _REPR_AST_DIRECT_BODY),
    "":                              (_REPR_AST_DIRECT_BODY, _REPR_SURGICAL_EDIT, _REPR_REPLACE_SYM_BODY),
}


def _select_next_repr_for_repair(
    ctx: SelectorRetryContext,
    failure_class: str,
    *,
    exclude_current: bool = True,
    instruction_kind_compatible_only: bool = False,
) -> Optional[str]:
    """Return the next allowed representation to escalate to, or None."""
    pref = _FAILURE_CLASS_REPRESENTATION_PREFERENCE.get(
        failure_class, _FAILURE_CLASS_REPRESENTATION_PREFERENCE[""]
    )

    if ctx.has_selector() and ctx.fallback_representations:
        allowed = set(ctx.fallback_representations) | {ctx.selector_first_choice}
        candidates = [r for r in pref if r in allowed]
        for r in ctx.fallback_representations:
            if r not in candidates:
                candidates.append(r)
    else:
        candidates = list(pref)

    for rep in candidates:
        if exclude_current and rep == ctx.current_representation:
            continue
        if rep in ctx.forbidden_representations:
            continue
        if instruction_kind_compatible_only and rep not in _REPR_TO_INSTRUCTION_KIND:
            continue
        return rep
    return None





# semantic_primary_issue → op-tier failure_class
_SEMANTIC_PRIMARY_ISSUE_MAP: dict[str, str] = {
    "ast_parse_failed": "syntax_invalid_after_edit",
    "symbol_removed": "symbol_missing_after_edit",
    "signature_changed": "signature_changed",
    "class_shape_changed": "signature_changed",
    "placement_violation": "placement_violation",
    "insertion_scope_warning": "placement_violation",
    "symbol_kind_changed": "signature_changed",
    "receiver_scope_mismatch": "placement_violation",
    "dead_code_inserted": "dead_code_introduced",
    "partial_implementation": "no_effect",
    "incomplete_return_path": "signature_changed",
}

_SEMANTIC_ERROR_PREFIX_MAP: dict[str, str] = {
    "ast_parse_failed": "syntax_invalid_after_edit",
    "symbol_removed": "symbol_missing_after_edit",
    "signature_changed": "signature_changed",
}


# Fallback for string-only apply errors; prefer exception classes when present.
_APPLY_ERROR_PREFIX_MAP: dict[str, str] = {
    "unable to find context": "anchor_miss",
    "no context found": "anchor_miss",
    "context not found": "anchor_miss",
    "while searching for": "anchor_miss",
    "patch does not apply": "anchor_miss",
    "anchor not found": "anchor_miss",
    "no match found": "anchor_miss",
    "permission denied": "write_error",
    "operation not permitted": "write_error",
    "access is denied": "write_error",
    "read-only file system": "write_error",
    "no such file": "file_not_found",
    "does not exist": "file_not_found",
    "cannot find the file": "file_not_found",
    "timed out": "timeout",
    "timeout expired": "timeout",
    "deadline exceeded": "timeout",
}


def _classify_apply_error(tool_error: Optional[str]) -> str:
    """Map tool error text to a normalized failure class via prefix matching."""
    if not tool_error:
        return FailureClass.PATCH_APPLY_FAILED
    _text = tool_error.lower()
    for prefix, mapped in sorted(_APPLY_ERROR_PREFIX_MAP.items(), key=lambda x: -len(x[0])):
        # All prefixes are multi-word phrases long enough for substring ``in``.
        # Word-boundary regex \b is unnecessary — false positives are
        # extremely unlikely in real error/status messages.
        if prefix in _text:
            return mapped
    return FailureClass.PATCH_APPLY_FAILED


def _classify_post_write_exception(exc: BaseException) -> str:
    """Map a Python exception raised during post-write checks to a failure class."""
    if isinstance(exc, SyntaxError):
        return FailureClass.SYNTAX_ERROR_AFTER_PATCH
    if isinstance(exc, FileNotFoundError):
        return FailureClass.FILE_NOT_FOUND
    if isinstance(exc, PermissionError):
        return FailureClass.WRITE_ERROR
    if isinstance(exc, (UnicodeDecodeError, UnicodeEncodeError)):
        return FailureClass.WRITE_ERROR
    if isinstance(exc, TimeoutError):
        return FailureClass.TIMEOUT
    if isinstance(exc, (IsADirectoryError, NotADirectoryError, OSError)):
        return FailureClass.WRITE_ERROR
    return FailureClass.UNKNOWN


# Acceptance check name → failure class.
_ACCEPTANCE_CHECK_FAILURE_MAP: dict[str, str] = {
    "symbol must remain present": "symbol_missing_after_edit",
    "file must remain syntactically valid": "syntax_invalid_after_edit",
    "inserted snippet must appear in file": "snippet_missing_after_edit",
    "signature must be preserved": "signature_changed",
    "imports must remain stable": "imports_changed_unexpectedly",
    "all edited files must remain syntactically valid": "syntax_invalid_after_edit",
    "all required symbols must remain present": "symbol_missing_after_edit",
}

# precision_mode → failure_class for the no-diff path.
_PM_NO_DIFF_FAILURE_MAP: dict[str, str] = {
    "insert_after_symbol": FailureClass.INSERT_POSITION_UNKNOWN,
    "insert_after_symbol_grep_fallback": FailureClass.INSERT_POSITION_UNKNOWN,
}


def _classify_semantic_failure(
    semantic_primary_issue: Optional[str],
    errors: list[str],
) -> str:
    """Return an op-tier failure_class for a failed semantic verification."""
    if semantic_primary_issue:
        return _SEMANTIC_PRIMARY_ISSUE_MAP.get(
            semantic_primary_issue, "semantic_verification_failed"
        )
    for err in errors:
        for key, mapped in _SEMANTIC_ERROR_PREFIX_MAP.items():
            if err.startswith(key):
                return mapped
    return "semantic_verification_failed"


def _strip_redundant_inline_imports(new_body: str, file_source: str) -> str:
    """Remove indented imports from new_body when the module already exists at module level."""
    try:
        tree = _ast.parse(file_source)
    except SyntaxError:
        return new_body

    module_imports: set = set()
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                module_imports.add(alias.name)
                if alias.asname:
                    module_imports.add(alias.asname)
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                module_imports.add(node.module)
            for alias in (node.names or []):
                module_imports.add(alias.name)
                if alias.asname:
                    module_imports.add(alias.asname)

    if not module_imports:
        return new_body

    try:
        _tree = _ast.parse(new_body)
        _redundant_lines = set()
        _body_lines = new_body.split("\n")
        for _node in _ast.walk(_tree):
            # Cover the FULL source span of the import statement, not just its
            # first line. A parenthesized multi-line import spans lineno..end_lineno;
            # removing only lineno (the `from … import (` opener) leaves the
            # continuation lines (`write_task,`, `read_result,`, `)`) as orphans
            # that produce an "unexpected indent" SyntaxError — which in turn
            # defeats the AST-precise path and forces a surgical fallback.
            # Module and other non-statement nodes have no lineno/end_lineno → skip.
            _node_last = getattr(_node, "end_lineno", None)
            if _node_last is None:
                continue
            _node_span = set(range(_node.lineno, _node_last + 1))
            if isinstance(_node, _ast.Import):
                for _alias in _node.names:
                    if _alias.name in module_imports or (_alias.asname and _alias.asname in module_imports):
                        if _alias.asname and _alias.asname not in module_imports:
                            _al_refs = [_ln for i, _ln in enumerate(_body_lines, start=1) if i not in _node_span and _alias.asname in _ln]
                            if _al_refs:
                                continue
                        _redundant_lines |= _node_span
                        break
            elif isinstance(_node, _ast.ImportFrom):
                if _node.module and _node.module in module_imports:
                    _keep_from = False
                    for _alias in (_node.names or []):
                        if _alias.asname and _alias.asname not in module_imports:
                            _al_refs = [_ln for i, _ln in enumerate(_body_lines, start=1) if i not in _node_span and _alias.asname in _ln]
                            if _al_refs:
                                _keep_from = True
                                break
                    if not _keep_from:
                        _redundant_lines |= _node_span
                else:
                    for _alias in (_node.names or []):
                        if _alias.name in module_imports or (_alias.asname and _alias.asname in module_imports):
                            if _alias.asname and _alias.asname not in module_imports:
                                _al_refs = [_ln for i, _ln in enumerate(_body_lines, start=1) if i not in _node_span and _alias.asname in _ln]
                                if _al_refs:
                                    continue
                            _redundant_lines |= _node_span
                            break
        if _redundant_lines:
            _cleaned = [_l for _i, _l in enumerate(_body_lines, start=1) if _i not in _redundant_lines]
            _candidate = "\n".join(_cleaned)
            # Guard: never return a stripped result that itself fails to parse —
            # a broken strip (orphaned lines / dangling continuation) is worse
            # than leaving the redundant import in place. Bail out to the original.
            try:
                _ast.parse(_candidate)
            except SyntaxError:
                logger.info(
                    "[STRIP_INLINE_IMPORT] skipped removal: would produce invalid syntax "
                    "(removed %d line(s)); keeping original inline import(s)",
                    len(_redundant_lines),
                )
                return new_body
            logger.info("[STRIP_INLINE_IMPORT] removed %d redundant inline import line(s)", len(_redundant_lines))
            return _candidate
        return new_body
    except SyntaxError:
        return new_body


def _strip_redundant_dataclass_decorator(new_body: str, file_source: str) -> str:
    """Remove ``@dataclass`` line from *new_body* when the class in *file_source*
    already carries ``@dataclass``.

    Prevents the common LLM hallucination where ``@dataclass`` is added to a
    class that already has it, producing ``@dataclass\n@dataclass\nclass X:``.
    Parallels ``_strip_redundant_inline_imports`` in approach.
    """
    try:
        _tree = _ast.parse(file_source)
    except SyntaxError:
        return new_body

    # ── 1. Collect class names that already have @dataclass in file_source ──
    _file_dataclass_classes: set = set()
    for _node in _ast.iter_child_nodes(_tree):
        if isinstance(_node, _ast.ClassDef):
            for _dec in _node.decorator_list:
                if isinstance(_dec, _ast.Name) and _dec.id == "dataclass":
                    _file_dataclass_classes.add(_node.name)
                    break
                if isinstance(_dec, _ast.Call) and isinstance(_dec.func, _ast.Name) and _dec.func.id == "dataclass":
                    _file_dataclass_classes.add(_node.name)
                    break

    if not _file_dataclass_classes:
        return new_body

    # ── 2. Check new_body for the same pattern ────────────────────────────
    try:
        _new_tree = _ast.parse(new_body)
    except SyntaxError:
        return new_body

    _body_lines = new_body.split("\n")
    _lines_to_remove: set = set()

    for _node in _ast.walk(_new_tree):
        if not isinstance(_node, _ast.ClassDef):
            continue
        if _node.name not in _file_dataclass_classes:
            continue

        # Check if this class in new_body has @dataclass decorator
        _has_dataclass = False
        for _dec in _node.decorator_list:
            if isinstance(_dec, (_ast.Name, _ast.Attribute)):
                if _dec.id == "dataclass":
                    _has_dataclass = True
                    break
            if isinstance(_dec, _ast.Call):
                _fn = _dec.func
                if isinstance(_fn, _ast.Name) and _fn.id == "dataclass":
                    _has_dataclass = True
                    break

        if not _has_dataclass:
            continue

        # Remove the bare @dataclass line(s) that precede this class in new_body
        for _dec in _node.decorator_list:
            if isinstance(_dec, _ast.Name) and _dec.id == "dataclass":
                _lines_to_remove.add(_dec.lineno)
            elif isinstance(_dec, _ast.Call):
                _fn = _dec.func
                if isinstance(_fn, _ast.Name) and _fn.id == "dataclass":
                    _lines_to_remove.add(_dec.lineno)

    if _lines_to_remove:
        _cleaned = [_l for _i, _l in enumerate(_body_lines, start=1) if _i not in _lines_to_remove]
        _removed = len(_lines_to_remove)
        logger.info("[STRIP_DATACLASS_DECO] removed %d redundant @dataclass decorator line(s)", _removed)
        return "\n".join(_cleaned)

    return new_body


def _restore_missing_docstring(
    original_content: str,
    new_content: str,
    target_symbol: str,
) -> Optional[str]:
    """Restore docstring from original into new content if the LLM dropped it."""
    try:
        _o_tree = _ast.parse(original_content)
        _n_tree = _ast.parse(new_content)
    except SyntaxError:
        return None

    _bare = target_symbol.split('.')[-1]
    _parent = target_symbol.split('.')[-2] if '.' in target_symbol else None

    def _find_docstring_node(tree, bare, parent):
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == bare:
                if parent:
                    for cls in (n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef) and n.name == parent):
                        if any(n is node for n in _ast.walk(cls)):
                            return node
                else:
                    return node
        return None

    _o_node = _find_docstring_node(_o_tree, _bare, _parent)
    _n_node = _find_docstring_node(_n_tree, _bare, _parent)
    if _o_node is None or _n_node is None:
        return None
    if _ast.get_docstring(_n_node) is not None:
        return None
    _orig_doc = _ast.get_docstring(_o_node)
    if not _orig_doc:
        return None

    _first = _n_node.body[0]
    _indent = ' ' * _first.col_offset
    _n_lines = new_content.split('\n')
    _n_lines.insert(_first.lineno - 1, f'{_indent}"""{_orig_doc}"""')
    return '\n'.join(_n_lines)

