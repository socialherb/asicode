"""models.py — TS/JS primitive models (re-exports from language-agnostic core).

Re-exports all model classes from external_llm.editor.primitives.models so that
existing ts_vm imports continue to work while the actual primitive logic
is shared across all languages.
"""
from __future__ import annotations

# ── Re-export from language-agnostic core ─────────────────────────────
# All primitive model classes are now defined in external_llm.editor.primitives.models.
# The ts_vm re-exports them so existing imports continue to work while
# the actual primitive logic is shared across all languages.
from external_llm.editor.primitives.models import (  # noqa: F401
    CallSite,
    ImportInfo,
    PrimitiveKind,
    PrimitiveOp,
    PrimitivePlan,
    PrimitiveResult,
    SymbolDef,
)
