"""update_call.py — Generic UPDATE_CALL primitive.

Language-agnostic version that works on CodeContext.
"""
from __future__ import annotations

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult


def update_call(op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
    """Update call expressions by callee name.

    Payload:
        callee: str — current callee name to match
        new_callee: str (optional) — new callee name
        new_args: str (optional) — new arguments string (without parens)
        scope: str (optional) — limit to calls within this function
    """
    callee = op.payload.get("callee", "")
    new_callee = op.payload.get("new_callee")
    new_args = op.payload.get("new_args")
    scope = op.payload.get("scope")

    if not callee:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="UPDATE_CALL: missing 'callee' in payload",
        )

    if new_callee is None and new_args is None:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message="UPDATE_CALL: need 'new_callee' or 'new_args'",
        )

    # Find matching call sites
    call_sites = ctx.get_call_sites(callee)
    if scope:
        call_sites = [cs for cs in call_sites if cs.caller == scope]

    if not call_sites:
        return PrimitiveResult(
            success=False, code=ctx.code,
            message=f"UPDATE_CALL: no calls to '{callee}' found",
        )

    # Sort back-to-front for safe replacement
    call_sites.sort(key=lambda cs: cs.start_byte, reverse=True)

    code = ctx.code
    count = 0

    for cs in call_sites:
        call_text = code[cs.start_byte:cs.end_byte]

        if new_callee is not None and new_args is not None:
            new_call = f"{new_callee}({new_args})"
            code = code[:cs.start_byte] + new_call + code[cs.end_byte:]
            count += 1
        elif new_callee is not None:
            new_call = call_text.replace(callee, new_callee, 1)
            code = code[:cs.start_byte] + new_call + code[cs.end_byte:]
            count += 1
        elif new_args is not None:
            paren_open = call_text.find("(")
            paren_close = call_text.rfind(")")
            if paren_open != -1 and paren_close != -1:
                new_call = call_text[:paren_open + 1] + new_args + call_text[paren_close:]
                code = code[:cs.start_byte] + new_call + code[cs.end_byte:]
                count += 1

    return PrimitiveResult(
        success=True, code=code,
        message=f"Updated {count} call(s) to '{callee}'",
    )
