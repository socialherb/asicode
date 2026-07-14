"""insert_import.py — Generic INSERT_IMPORT / REMOVE_IMPORT primitives.

Language-agnostic versions that work on CodeContext.
"""
from __future__ import annotations

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult


def insert_import(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Insert an import statement at the top of the file.

    Payload:
        statement: str — full import statement
    """
    statement = op.payload.get("statement", "").strip()
    if not statement:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="INSERT_IMPORT: missing 'statement' in payload",
        )

    # Deduplicate
    source = _extract_source(statement)
    if source:
        for imp in ctx.get_imports():
            if imp.source == source:
                return PrimitiveResult(
                    success=True, code=ctx.code,
                    message=f"INSERT_IMPORT: import from '{source}' already exists, skipped",
                )

    insert_at = ctx.get_import_insertion_point()

    if insert_at == 0:
        new_code = statement + "\n" + ctx.code
    else:
        new_code = ctx.insert_at(insert_at, statement + "\n")

    return PrimitiveResult(
        success=True, code=new_code,
        message=f"Inserted import: {statement}",
        affected_range=(insert_at, insert_at + len(statement)),
    )


def remove_import(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Remove an import by source module path.

    Payload:
        source: str — module path to remove
    """
    source = op.payload.get("source", "")
    if not source:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="REMOVE_IMPORT: missing 'source' in payload",
        )

    for imp in ctx.get_imports():
        if imp.source == source:
            end = imp.end_byte
            while end < len(ctx._code_bytes) and ctx._code_bytes[end] in (ord("\n"), ord("\r")):
                end += 1
            new_code = ctx.delete_range(imp.start_byte, end)
            return PrimitiveResult(
                success=True, code=new_code,
                message=f"Removed import from '{source}'",
                affected_range=(imp.start_byte, end),
            )

    return PrimitiveResult(
        success=False, code=ctx.code,
        message=f"REMOVE_IMPORT: import from '{source}' not found",
    )


def _extract_source(statement: str) -> str:
    """Extract module source from an import statement string."""
    for q in ("'", '"'):
        if f"from {q}" in statement:
            start = statement.index(f"from {q}") + len(f"from {q}")
            end = statement.index(q, start)
            return statement[start:end]
        if f"from{q}" in statement:
            start = statement.index(f"from{q}") + len(f"from{q}")
            end = statement.index(q, start)
            return statement[start:end]
    return ""
