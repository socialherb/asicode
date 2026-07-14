"""models.py — Language-agnostic primitive operation models.

Shared data types for the generic primitive execution core.
These replace TS-specific TSModule/IRNodeMeta dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PrimitiveKind(Enum):
    """Supported primitive operations."""

    # ── core (language-agnostic) ─────────────────────────────────────
    REPLACE_FUNCTION_BODY = "REPLACE_FUNCTION_BODY"
    INSERT_IMPORT = "INSERT_IMPORT"
    REMOVE_IMPORT = "REMOVE_IMPORT"
    RENAME_SYMBOL = "RENAME_SYMBOL"
    UPDATE_CALL = "UPDATE_CALL"
    INSERT_STATEMENT = "INSERT_STATEMENT"
    DELETE_NODE = "DELETE_NODE"


@dataclass
class PrimitiveOp:
    """A single deterministic code modification."""
    kind: PrimitiveKind
    payload: dict[str, Any] = field(default_factory=dict)
    target_id: Optional[str] = None


@dataclass
class PrimitiveResult:
    """Result of executing a single primitive."""
    success: bool
    code: str  # resulting code after operation
    message: str = ""
    affected_range: Optional[tuple] = None  # (start_byte, end_byte) of change


@dataclass
class PrimitivePlan:
    """An ordered list of primitives to execute."""
    ops: list[PrimitiveOp] = field(default_factory=list)
    description: str = ""


@dataclass
class SymbolDef:
    """A symbol definition found in source code.

    Attributes:
        name: Symbol name.
        kind: Kind string ("function", "class", "interface", "variable", etc.).
        start_byte: Byte offset of the definition start.
        end_byte: Byte offset of the definition end.
        body_start_byte: Byte offset of the body opening ({), or None.
        body_end_byte: Byte offset of the body closing (}), or None.
    """
    name: str
    kind: str
    start_byte: int
    end_byte: int
    body_start_byte: Optional[int] = None
    body_end_byte: Optional[int] = None


@dataclass
class ImportInfo:
    """An import statement found in source code."""
    source: str
    start_byte: int
    end_byte: int
    statement: str = ""


@dataclass
class CallSite:
    """A call expression found in source code."""
    callee: str
    start_byte: int
    end_byte: int
    caller: str = ""
    is_method_call: bool = False
    receiver: str = ""
