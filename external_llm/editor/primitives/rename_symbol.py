"""rename_symbol.py — Generic RENAME_SYMBOL primitive.

Language-agnostic version that works on CodeContext.
Uses tree-sitter reference queries for usage detection.
"""
from __future__ import annotations

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult


def rename_symbol(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Rename a symbol across all usage sites.

    Payload:
        old_name: str — current symbol name
        new_name: str — desired symbol name
    """
    old_name = op.payload.get("old_name", "")
    new_name = op.payload.get("new_name", "")

    if not old_name or not new_name:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="RENAME_SYMBOL: missing 'old_name' or 'new_name'",
        )

    if old_name == new_name:
        return PrimitiveResult(
            success=True, code=ctx.code,
            message="RENAME_SYMBOL: old and new names are identical, no-op",
        )

    # Collect all byte ranges to replace
    ranges: list[tuple[int, int]] = []

    # 1. Use tree-sitter reference queries if available (includes definitions)
    from external_llm.languages.tree_sitter_utils import query_captures

    lang_str = ctx.language.value
    from external_llm.languages.tree_sitter_utils import _REFERENCE_QUERIES

    ref_q = _REFERENCE_QUERIES.get(lang_str)
    if ref_q:
        refs = query_captures(ctx.code, lang_str, ref_q)
        for ref in refs:
            if ref.text == old_name:
                # Verify word boundary (not substring of another name)
                before = ref.start_byte - 1
                after = ref.end_byte
                good = True
                if before >= 0:
                    b_char = ctx._code_bytes[before:before+1].decode("utf-8", errors="replace")
                    if b_char.isalnum() or b_char == "_":
                        good = False
                if after < len(ctx._code_bytes):
                    a_char = ctx._code_bytes[after:after+1].decode("utf-8", errors="replace")
                    if a_char.isalnum() or a_char == "_":
                        good = False
                if good:
                    ranges.append((ref.start_byte, ref.end_byte))

    if not ranges:
        # Fallback: find-replace all occurrences with word boundary check
        code = ctx.code
        idx = code.find(old_name)
        while idx != -1:
            after_idx = idx + len(old_name)
            before = code[idx - 1] if idx > 0 else " "
            after = code[after_idx] if after_idx < len(code) else " "
            if not before.isalnum() and before != "_" and not after.isalnum() and after != "_":
                ranges.append((idx, after_idx))
            idx = code.find(old_name, idx + 1)

    if not ranges:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message=f"RENAME_SYMBOL: no occurrences of '{old_name}' found",
        )

    # Deduplicate and apply back-to-front
    ranges = sorted(set(ranges), key=lambda r: r[0], reverse=True)
    code = ctx.code
    for start, end in ranges:
        code = code[:start] + new_name + code[end:]

    count = len(ranges)
    return PrimitiveResult(
        success=True, code=code,
        message=f"Renamed '{old_name}' → '{new_name}' ({count} sites)",
    )
