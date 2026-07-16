# code_integrity.py — Language-neutral code integrity verification.
#
# Extracted from TSVMBridge (ts_vm_bridge.py) to share verification
# logic across all language VMs.  These functions are pure (no self)
# and operate at the AST/tree-sitter level.

from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.agent.operation_models import Operation

logger = logging.getLogger(__name__)


# ── Verification: target symbol survival ──────────────────────────────


def verify_target_survived(
    op: Operation, code: str, abs_path: str,
    pre_module: Any = None, pre_code: str | None = None,
) -> bool:
    """Verify target symbol still exists after VM execution.

    The LLM may generate code that unintentionally destroys the target
    function/field.  This check prevents silent data loss.

    When pre_module is provided and the target is a class, additionally
    verifies that ALL known methods and fields from the pre-execution IR
    still exist in the new code.

    Returns True if:
      - Edit kind is additive (field/call/logging) — safe by nature
      - Target is a class AND all its known members still exist
      - Symbol still exists in the result code
    Returns False if the symbol was destroyed.
    """
    _edit_kind = (op.metadata or {}).get("edit_kind", "").lower()
    _SMALL_ADDITIVE_KINDS = frozenset({
        "add_call", "add_validation", "add_field", "add_logging",
        "set_value", "local_patch", "guard_add",
        "delegate_common_logic", "helper_extraction",
    })
    if _edit_kind in _SMALL_ADDITIVE_KINDS:
        return True

    _bare = op.symbol.split(".")[-1] if op.symbol else ""
    if not _bare:
        return True

    # ── Verify target itself exists ──
    try:
        from external_llm.agent._shared_utils import _is_real_ts_symbol
        if not _is_real_ts_symbol(_bare, code, file_path=abs_path):
            return False
    except ImportError:
        if _bare not in code:
            return False
    except Exception:
        if _bare not in code:
            return False

    # ── Class member integrity check ──
    if pre_module is not None and op.symbol and "." not in op.symbol:
        _cls = None
        for cls in getattr(pre_module, "classes", []):
            if getattr(cls, "name", "") == op.symbol:
                _cls = cls
                break
        if _cls is not None:
            for method in getattr(_cls, "methods", []):
                _method_name = getattr(method, "name", "")
                if _method_name:
                    try:
                        if not _is_real_ts_symbol(_method_name, code, file_path=abs_path):
                            logger.warning(
                                "[TS_VM] target class %r lost method %r "
                                "after LLM generation — downgrading to failure",
                                op.symbol, _method_name,
                            )
                            return False
                    except Exception:
                        if _method_name not in code:
                            logger.warning(
                                "[TS_VM] target class %r lost method %r "
                                "(substring fallback) — downgrading to failure",
                                op.symbol, _method_name,
                            )
                            return False
                    if pre_code is not None:
                        _meta = getattr(method, "meta", None)
                        if _meta is not None:
                            from external_llm.languages.tree_sitter_utils import (
                                count_method_statements,
                                grammar_key_for_path,
                            )
                            _fp = getattr(pre_module, "file_path", "") or ""
                            _lang = grammar_key_for_path(_fp) or "typescript"

                            _old_method_source = pre_code[_meta.start_byte:_meta.end_byte]
                            _old_stmt_count = count_method_statements(
                                _old_method_source, _method_name, _lang,
                            )
                            _new_stmt_count = count_method_statements(
                                code, _method_name, _lang,
                            )

                            if (_old_stmt_count is not None
                                    and _new_stmt_count is not None
                                    and _old_stmt_count > 0
                                    and _new_stmt_count == 0):
                                logger.warning(
                                    "[TS_VM] target class %r method %r body "
                                    "is empty stub (%d -> 0 AST statements) — "
                                    "downgrading to failure",
                                    op.symbol, _method_name, _old_stmt_count,
                                )
                                return False
            for prop in getattr(_cls, "properties", []):
                _prop_name = getattr(prop, "name", "")
                if _prop_name:
                    try:
                        if not _is_real_ts_symbol(_prop_name, code, file_path=abs_path):
                            logger.warning(
                                "[TS_VM] target class %r lost field %r "
                                "after LLM generation — downgrading to failure",
                                op.symbol, _prop_name,
                            )
                            return False
                    except Exception:
                        if _prop_name not in code:
                            logger.warning(
                                "[TS_VM] target class %r lost field %r "
                                "(substring fallback) — downgrading to failure",
                                op.symbol, _prop_name,
                            )
                            return False

        # ── Structural integrity checks (Patterns 3, 4, 5) ──
        if op.symbol and "." not in op.symbol:
            if not verify_no_duplicate_in_class(
                code, op.symbol, abs_path,
            ):
                return False
            if not verify_no_hallucinated_calls(
                code, pre_module, op.symbol,
            ):
                return False

        if not verify_scope_integrity(op, code, pre_module, pre_code):
            return False

    return True


# ── Verification: find last top-level definition ───────────────────────


def verify_last_top_level_def(code: str) -> Optional[str]:
    """Find the last top-level function/class/const definition in code.

    Returns the name of the last definition, or None if none found.
    Used as fallback anchor when a requested symbol is not in the file.
    """
    try:
        from external_llm.languages.tree_sitter_utils import find_all_symbols
        _symbols = find_all_symbols(code, "typescript")
        if _symbols:
            return _symbols[-1][0]
    except Exception:
        pass
    return None


# ── Verification: no duplicate class members ───────────────────────────


def verify_no_duplicate_in_class(
    code: str, class_name: str, file_path: str = "",
) -> bool:
    """Verify a class has no duplicate method/field definitions.

    Uses tree-sitter AST to count unique member names vs total member
    definitions.  Returns True if no duplicates found, False if duplicates
    detected or if tree-sitter is unavailable (non-blocking).
    """
    if not class_name:
        return True
    try:
        from external_llm.languages.tree_sitter_utils import (
            count_class_members,
            get_class_member_names,
            grammar_key_for_path,
        )
        _lang = grammar_key_for_path(file_path) or "typescript"
        _unique = get_class_member_names(code, class_name, _lang)
        _total = count_class_members(code, class_name, _lang)
        if _unique is not None and _total is not None:
            _unique_methods, _unique_fields = _unique
            _total_methods, _total_fields = _total
            if _total_methods > len(_unique_methods):
                logger.warning(
                    "[TS_VM] class %r has %d method definitions but "
                    "only %d unique names — duplicate methods detected",
                    class_name, _total_methods, len(_unique_methods),
                )
                return False
            if _total_fields > len(_unique_fields):
                logger.warning(
                    "[TS_VM] class %r has %d field definitions but "
                    "only %d unique names — duplicate fields detected",
                    class_name, _total_fields, len(_unique_fields),
                )
                return False
    except Exception:
        pass
    return True


# ── Verification: no hallucinated this-references ──────────────────────


def verify_no_hallucinated_calls(
    code: str, module: Any, class_name: str,
) -> bool:
    """Verify no ``this.xxx()`` calls reference undefined members.

    Extracts all ``this.xxx`` member access expressions from the code
    and cross-references each name against the IR class's known methods
    and fields.  Returns False if any this-reference is not found in IR.
    """
    if module is None or not class_name:
        return True

    _known_members: set = set()
    for cls in getattr(module, "classes", []):
        if getattr(cls, "name", "") == class_name:
            for method in getattr(cls, "methods", []):
                _name = getattr(method, "name", "")
                if _name:
                    _known_members.add(_name)
            for prop in getattr(cls, "properties", []):
                _name = getattr(prop, "name", "")
                if _name:
                    _known_members.add(_name)
            break

    if not _known_members:
        return True

    _fp = getattr(module, "file_path", "") or ""
    try:
        from external_llm.languages.tree_sitter_utils import (
            grammar_key_for_path,
            parse_to_tree,
        )
        _lang = grammar_key_for_path(_fp) or "typescript"
        _tree = parse_to_tree(code, _lang)
        if _tree is not None:
            def _collect_members(node):
                if node.type in ("method_definition", "field_definition",
                                 "public_field_definition"):
                    _name_node = node.child_by_field_name("name")
                    if _name_node is not None:
                        _known_members.add(
                            _name_node.text.decode("utf-8"))
                for child in node.named_children:
                    _collect_members(child)
            _collect_members(_tree.root_node)
    except Exception:
        pass

    try:
        from external_llm.languages.tree_sitter_utils import (
            extract_this_references,
        )
        _refs = extract_this_references(code, _lang)
        if not _refs:
            return True

        _hallucinated = [r for r in _refs if r not in _known_members]
        if _hallucinated:
            logger.warning(
                "[TS_VM] class %r has hallucinated this-references "
                "not in IR: %s — known members: %s",
                class_name, _hallucinated, sorted(_known_members),
            )
            return False
    except Exception:
        pass
    return True


# ── Verification: scope integrity ──────────────────────────────────────


def verify_scope_integrity(
    op: Operation, code: str, module: Any, pre_code: str,
) -> bool:
    """Verify the edit was placed in the correct scope.

    Checks:
    - For ``add_field``: new code is inside a class body, not a function body.
    - For ``add_call``/``add_validation``/``add_logging``: new code is
      inside a function/method body.

    Uses tree-sitter to find the AST node at the byte offset where
    code first differs from pre_code, then checks the enclosing scope.
    """
    _edit_kind = (op.metadata or {}).get("edit_kind", "").lower()
    if not _edit_kind:
        return True

    _CLASS_SCOPE_KINDS = frozenset({"add_field"})
    _FUNCTION_SCOPE_KINDS = frozenset({
        "add_call", "add_validation", "add_logging",
        "set_value", "local_patch", "guard_add",
    })

    if _edit_kind not in _CLASS_SCOPE_KINDS | _FUNCTION_SCOPE_KINDS:
        return True

    if not pre_code or code == pre_code:
        return True

    _insert_offset = 0
    for i, (a, b) in enumerate(zip(pre_code, code, strict=False)):
        if a != b:
            _insert_offset = i
            break
    else:
        _insert_offset = len(pre_code)

    try:
        from external_llm.languages.tree_sitter_utils import (
            parse_to_tree,
        )
        _fp = getattr(module, "file_path", "") or ""
        try:
            from external_llm.languages.tree_sitter_utils import grammar_key_for_path
            _lang = grammar_key_for_path(_fp) or "typescript"
        except ImportError:
            _lang = "typescript"

        _tree = parse_to_tree(code, _lang)
        if _tree is None:
            return True

        def _find_node_at_offset(node, offset):
            if node.start_byte <= offset <= node.end_byte:
                for child in node.children:
                    result = _find_node_at_offset(child, offset)
                    if result is not None:
                        return result
                return node
            return None

        _insert_node = _find_node_at_offset(
            _tree.root_node, _insert_offset,
        )
        if _insert_node is None:
            return True

        _scope_type = None
        _parent = _insert_node
        while _parent is not None:
            if _parent.type == "class_body":
                _scope_type = "class"
                break
            if _parent.type == "statement_block":
                _scope_type = "function"
                break
            _parent = getattr(_parent, "parent", None)

        if _scope_type is None:
            return True

        if _edit_kind in _CLASS_SCOPE_KINDS and _scope_type != "class":
            logger.warning(
                "[TS_VM] scope mismatch: edit_kind=%r expected class body "
                "but insertion is inside %s scope",
                _edit_kind, _scope_type,
            )
            return False

        if _edit_kind in _FUNCTION_SCOPE_KINDS and _scope_type != "function":
            logger.warning(
                "[TS_VM] scope mismatch: edit_kind=%r expected function body "
                "but insertion is inside %s scope",
                _edit_kind, _scope_type,
            )
            return False

    except Exception:
        pass
    return True
