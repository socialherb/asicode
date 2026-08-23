"""models.py — Execution VM error types (shared)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifyError:
    """A single verification error."""

    message: str
    line: int | None = None
    column: int | None = None
    code: str | None = None  # error code (e.g. TS2304, E0602)
