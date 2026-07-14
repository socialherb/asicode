"""insert_statement.py — Generic INSERT_STATEMENT primitive.

Language-agnostic version that works on CodeContext.
"""
from __future__ import annotations

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult


def insert_statement(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Insert a statement at a position relative to an anchor symbol.

    Payload:
        statement: str — the statement to insert
        anchor: str — symbol name to anchor to
        position: str — "before" | "after" | "start" | "end"
    """
    statement = op.payload.get("statement", "").rstrip()
    anchor = op.payload.get("anchor", "")
    position = op.payload.get("position", "end")

    if not statement:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="INSERT_STATEMENT: missing 'statement'",
        )
    if not anchor:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="INSERT_STATEMENT: missing 'anchor'",
        )

    sym = ctx.get_symbol(anchor)

    if position in ("start", "end") and sym and sym.body_start_byte is not None:
        # Insert at start/end of function body
        indent = ctx.symbol_indent(anchor)
        body_indent = indent + "  "
        indented = body_indent + statement

        if position == "start":
            insert_at = sym.body_start_byte
            new_code = ctx.replace_range(insert_at, insert_at, "\n" + indented)
        else:  # end
            insert_at = sym.body_end_byte
            new_code = ctx.replace_range(insert_at, insert_at, indented + "\n")

        return PrimitiveResult(
            success=True, code=new_code,
            message=f"Inserted statement at {position} of '{anchor}'",
        )

    if sym is None:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message=f"INSERT_STATEMENT: symbol '{anchor}' not found",
        )

    if position == "before":
        insert_at = sym.start_byte
        new_code = ctx.replace_range(insert_at, insert_at, statement + "\n")
    elif position == "after":
        insert_at = sym.end_byte
        while insert_at < len(ctx._code_bytes) and ctx._code_bytes[insert_at] in (ord("\n"), ord("\r")):
            insert_at += 1
        new_code = ctx.replace_range(insert_at, insert_at, "\n" + statement + "\n")
    else:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message=f"INSERT_STATEMENT: unknown position '{position}'",
        )

    return PrimitiveResult(
        success=True, code=new_code,
        message=f"Inserted statement {position} '{anchor}'",
    )
