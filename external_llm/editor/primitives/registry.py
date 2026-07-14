"""registry.py — Generic primitive handler registry.

Maps PrimitiveKind → handler function.
Each handler: (op: PrimitiveOp, ctx: CodeContext) → PrimitiveResult
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import PrimitiveKind, PrimitiveOp, PrimitiveResult

logger = logging.getLogger(__name__)

# Handler type: (PrimitiveOp, CodeContext) → PrimitiveResult
PrimitiveHandler = Callable[[PrimitiveOp, CodeContext], PrimitiveResult]


class PrimitiveRegistry:
    """Registry of primitive operation handlers."""

    def __init__(self):
        self._handlers: dict[PrimitiveKind, PrimitiveHandler] = {}

    def register(self, kind: PrimitiveKind, handler: PrimitiveHandler) -> None:
        self._handlers[kind] = handler

    def execute(self, op: PrimitiveOp, ctx: CodeContext) -> PrimitiveResult:
        handler = self._handlers.get(op.kind)
        if handler is None:
            return PrimitiveResult(
                success=False,
                code=ctx.code,
                message=f"Unknown primitive: {op.kind.value}",
            )
        try:
            return handler(op, ctx)
        except Exception as e:
            logger.exception("Primitive %s failed", op.kind.value)
            return PrimitiveResult(
                success=False,
                code=ctx.code,
                message=f"Primitive {op.kind.value} failed: {e}",
            )

    @property
    def registered_kinds(self) -> list:
        return list(self._handlers.keys())
