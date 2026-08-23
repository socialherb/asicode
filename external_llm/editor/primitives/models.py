"""models.py — Language-agnostic primitive operation models.

Shared data types for the primitive repair pipeline (PrimitiveKind/PrimitiveOp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PrimitiveKind(Enum):
    """Supported primitive operations."""

    # ── core (language-agnostic) ─────────────────────────────────────
    INSERT_IMPORT = "INSERT_IMPORT"
    INSERT_STATEMENT = "INSERT_STATEMENT"


@dataclass
class PrimitiveOp:
    """A single deterministic code modification."""

    kind: PrimitiveKind
    payload: dict[str, Any] = field(default_factory=dict)
