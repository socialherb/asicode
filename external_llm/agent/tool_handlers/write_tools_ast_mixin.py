"""AST write tool handlers (P2-2 split).

``WriteToolsAstMixin`` — edit_ast. Split out of ``WriteToolsMixin`` in
``write_tools.py``; recombined there via ``class WriteToolsMixin(...,
WriteToolsAstMixin)``.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ...common.atomic_io import atomic_write_bytes
from ...common.text_reading import read_text_with_encoding_fallback
from ...languages import LanguageId
from .._shared_utils import compile_quiet

if TYPE_CHECKING:
    from ..tool_registry import ToolResult


class WriteToolsAstMixin:
    """AST edit handler: edit_ast."""

    def _tool_edit_ast(self, args: dict[str, Any]) -> "ToolResult":
        """Apply typed AST operations to a Python file. Deterministic — no LLM call.

        Each op has a 'type' and type-specific parameters.
        Supported ops: replace_expr, add_import, add_guard, delete_stmt,
        add_class_field, remove_import_name, list_append, list_remove.

        Unlike apply_patch, AST operations are whitespace-agnostic and survive
        context drift — they operate on the AST, not on raw text lines.
        """
        args = self._recover_args_from_raw(args, ("file_path",))
        file_path = str(args.get("file_path", "")).strip()
        ops_raw = args.get("ops")
        dry_run = args.get("dry_run", False)

        if not file_path:
            # If __raw_arguments is present, the JSON was likely truncated during streaming
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(
                ok=False, content="",
                error=f"'file_path' is required{_raw_hint}"
            )
        if not ops_raw:
            return self._make_result(
                ok=False, content="",
                error="'ops' is required"
            )

        if not isinstance(ops_raw, list) or not ops_raw:
            return self._make_result(
                ok=False, content="",
                error="'ops' must be a non-empty list of operation dicts"
            )

        # Resolve file path
        sec = self._secure_path(file_path, confine=True)
        if sec is None:
            return self._make_result(ok=False, content="", error=f"Path blocked (outside repo): {file_path}")
        abs_path = str(sec)
        if not os.path.isfile(abs_path):
            return self._make_result(ok=False, content="", error=f"File not found: {file_path}{self._suggest_missing_paths(file_path)}")

        # Normalize to relative for output
        rel_path = os.path.relpath(abs_path, self.repo_root)
        file_path = rel_path

        # F1 cross-process edit-lease guard.
        _lease_refused = self._refuse_foreign_leased([abs_path])
        if _lease_refused is not None:
            return _lease_refused

        # Read the file — strict UTF-8 first, then latin-1 (lossless 1:1 byte
        # round-trip when written back with the same encoding). The previous
        # errors="replace" fallback baked U+FFFD into the whole file before
        # rewriting it as UTF-8. Shared helper — the edit mixin must read with
        # the same chain (write-safety gate parity).
        try:
            source, _read_encoding = read_text_with_encoding_fallback(abs_path)
        except OSError:
            return self._make_result(
                ok=False, content="", error=f"Failed to read {file_path}: OSError"
            )
        if source is None:
            return self._make_result(
                ok=False, content="", error=f"Failed to read {file_path}: unsupported encoding"
            )

        # Check language — Python only
        if LanguageId.from_path(file_path) is not LanguageId.PYTHON:
            return self._make_result(
                ok=False, content="",
                error=f"AST edit is only supported for Python files (not {file_path})"
            )

        try:
            # Parse AST and validate syntax before applying
            import ast as _ast
            _ast.parse(source, filename=file_path)
        except SyntaxError as e:
            return self._make_result(
                ok=False, content="",
                error=f"Syntax error in {file_path}: {e}"
            )

        # Apply AST operations
        from .ast_op_executor import ASTOpExecutor

        executor = ASTOpExecutor()
        symbol = str(args.get("symbol", "")).strip()

        # Normalize LLM-friendly field names to ASTOpExecutor's internal parameter names
        _FIELD_ALIASES: dict[str, dict[str, str]] = {
            "add_import": {"import_name": "import", "import_stmt": "import"},
            "replace_expr": {"target": "old", "old_expr": "old", "old_text": "old", "new_expr": "new", "new_text": "new"},
            "add_guard": {"guard": "statement", "condition": "statement", "guard_stmt": "statement"},
            "delete_stmt": {"text_pattern": "pattern", "pattern_text": "pattern", "match": "pattern"},
            # NB: never alias the reserved op-discriminator key "type" here — it
            # would steal the op's own 'type' field. Use "annotation" for field_type.
            "add_class_field": {
                "class": "class_name", "cls": "class_name",
                "field": "field_name", "name": "field_name", "attr": "field_name",
                "annotation": "field_type",
                "default": "field_default", "value": "field_default",
            },
            "remove_import_name": {"import_name": "name", "symbol": "name"},
            "list_append": {"list": "list_name", "target": "list_name"},
            "list_remove": {"list": "list_name", "target": "list_name"},
        }

        ops_normalized: list[dict] = []
        for op in ops_raw:
            if isinstance(op, dict):
                normalized = dict(op)
                type_ = normalized.get("type", "")
                if not type_:
                    type_ = normalized.pop("op", None) or normalized.pop("action", "") or ""
                normalized["type"] = type_
                # Apply field name aliases for this op type
                aliases = _FIELD_ALIASES.get(type_, {})
                for alias, canonical in aliases.items():
                    if alias in normalized and canonical not in normalized:
                        normalized[canonical] = normalized.pop(alias)
                ops_normalized.append(normalized)

        result = executor.apply(source, ops_normalized, symbol=symbol)

        if not result.success:
            failed_str = "; ".join(result.ops_failed) if result.ops_failed else "unknown"
            _hint = self._ast_fail_hint(source, ops_normalized, symbol)
            error_msg = (
                f"AST edit failed in {file_path}@{symbol or '(module)'}: "
                f"{failed_str}{_hint}"
            )
            return self._make_result(
                ok=False, content="", error=error_msg,
                metadata={"near_match": bool(_hint)},
            )

        if not result.changed:
            return self._make_result(
                ok=True,
                content=f"AST edit: no changes needed (all {result.ops_applied} ops were idempotent)",
                metadata={
                    "file_path": file_path,
                    "ops_applied": result.ops_applied,
                    "changed": False,
                }
            )

        new_source = result.new_source

        # Final semantic validation with compile()
        try:
            compile_quiet(new_source, file_path, "exec")
        except SyntaxError as e:
            return self._make_result(
                ok=False, content="",
                error=f"AST edit produced invalid syntax in {file_path}: {e}"
            )

        # Generate diff for preview
        import difflib
        diff_lines = list(difflib.unified_diff(
            source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))
        diff_text = "".join(diff_lines)

        if dry_run:
            return self._make_result(
                ok=True,
                content=(
                    f"[DRY RUN] AST edit preview for {file_path}\n"
                    f"Ops: {len(ops_normalized)} ({result.ops_applied} applied)\n"
                    f"Diff ({len(diff_lines)} lines):\n"
                    f"{diff_text}"
                ),
                metadata={
                    "file_path": file_path,
                    "ops_applied": result.ops_applied,
                    "ops_total": len(ops_normalized),
                    "diff_preview": diff_text[:25000],
                    "changed": True,
                    "dry_run": True,
                }
            )

        # Write the file — same encoding it was read with (see read fallback
        # above). Encode BEFORE any file I/O: the atomic bytes writer opens its
        # temp file only after encoding succeeds, so an encode failure never
        # touches the target (a plain open("wb") would truncate first). The
        # atomic funnel (atomic_write_bytes -> invalidate_for_written_path)
        # keeps cached consumers fresh, same as every other write tool.
        try:
            _encoded_source = new_source.encode(_read_encoding)
            atomic_write_bytes(abs_path, _encoded_source)
        except (OSError, UnicodeEncodeError) as e:
            return self._make_result(
                ok=False, content="",
                error=f"Failed to write {file_path}: {e}"
            )

        self._record_text_edit(file_path)
        return self._make_result(
            ok=True,
            content=(
                f"AST edit applied to {file_path}\n"
                f"Ops: {result.ops_applied}/{len(ops_normalized)} applied, "
                f"{len(result.ops_failed)} failed\n"
                f"Diff ({len(diff_lines)} lines):\n"
                f"{diff_text}"
            ),
            metadata={
                "file_path": file_path,
                "ops_applied": result.ops_applied,
                "ops_failed": result.ops_failed if result.ops_failed else [],
                "diff_preview": diff_text[:25000],
                "changed": True,
                "symbol": symbol,
            }
        )
