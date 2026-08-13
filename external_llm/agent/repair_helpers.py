"""
Self-repair helpers.

Originally a mixin extracted from the PLANNER lane's operation_executor.py.
That executor is gone; what survives is the module-level helper set, whose only
live consumer is symbol_modify_tool (``_strip_redundant_dataclass_decorator`` /
``_strip_redundant_inline_imports``). The remaining ``self``-taking methods are
kept together with them rather than split across files.
"""
from __future__ import annotations

import ast as _ast
import logging

# import re — replaced with native string ops (6 patterns); local imports in methods only

logger = logging.getLogger(__name__)


def _strip_redundant_inline_imports(new_body: str, file_source: str, _src_tree=None) -> str:
    """Remove indented imports from new_body when the module already exists at module level."""
    try:
        tree = _src_tree if _src_tree is not None else _ast.parse(file_source)
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
    except SyntaxError:
        return new_body
    else:
        return new_body


def _strip_redundant_dataclass_decorator(new_body: str, file_source: str, _src_tree=None) -> str:
    """Remove ``@dataclass`` line from *new_body* when the class in *file_source*
    already carries ``@dataclass``.

    Prevents the common LLM hallucination where ``@dataclass`` is added to a
    class that already has it, producing ``@dataclass\n@dataclass\nclass X:``.
    Parallels ``_strip_redundant_inline_imports`` in approach.
    """
    try:
        _tree = _src_tree if _src_tree is not None else _ast.parse(file_source)
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
            if isinstance(_dec, (_ast.Name, _ast.Attribute)) and _dec.id == "dataclass":
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

