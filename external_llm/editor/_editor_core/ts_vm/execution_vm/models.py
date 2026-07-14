"""models.py — Execution VM result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerifyError:
    """A single verification error."""

    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None  # error code (e.g. TS2304)


@dataclass
class VMResult:
    """Result of a full VM execution cycle (apply → verify → repair → rollback)."""

    success: bool
    code: str  # final code (repaired, or rolled-back on failure)
    message: str = ""
    repair_attempts: int = 0
    verify_errors: list[VerifyError] = field(default_factory=list)
    rolled_back: bool = False
