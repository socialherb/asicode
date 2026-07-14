"""delete_node.py — Generic DELETE_NODE primitive.

Language-agnostic version that works on CodeContext.
"""
from __future__ import annotations

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult


def delete_node(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Delete a named symbol (function, class, variable).

    Payload:
        name: str — symbol name to delete
    """
    name = op.payload.get("name", "")
    if not name:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="DELETE_NODE: missing 'name'",
        )

    sym = ctx.get_symbol(name)
    if sym is None:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message=f"DELETE_NODE: '{name}' not found",
        )

    end = sym.end_byte
    # Include trailing newlines
    while end < len(ctx.code) and ctx.code[end] in ("\n", "\r"):
        end += 1

    new_code = ctx.delete_range(sym.start_byte, end)
    return PrimitiveResult(
        success=True, code=new_code,
        message=f"Deleted {sym.kind} '{name}'",
        affected_range=(sym.start_byte, end),
    )
