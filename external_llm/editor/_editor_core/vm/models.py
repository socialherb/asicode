"""models.py — Execution VM error types (shared)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerifyError:
    """A single verification error."""

    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None  # error code (e.g. TS2304, E0602)

