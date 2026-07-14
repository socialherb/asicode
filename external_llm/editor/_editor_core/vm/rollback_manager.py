"""rollback_manager.py - Snapshot-based rollback for safe execution.
Every mutation is recorded so the VM can revert to any prior state
if verification fails or repair is exhausted.
"""
from __future__ import annotations


class RollbackManager:
    """Maintains a history of code snapshots for safe rollback."""

    def __init__(self, original_code: str):
        self._original = original_code
        self._history: list[str] = [original_code]

    def push(self, code: str) -> None:
        """Record a new code snapshot after a successful mutation."""
        self._history.append(code)

    def rollback(self) -> str:
        """Revert to the original (pre-mutation) code."""
        return self._original

    def rollback_one(self) -> str:
        """Revert to the previous snapshot (undo last mutation)."""
        if len(self._history) > 1:
            self._history.pop()
        return self._history[-1]

    def last(self) -> str:
        """Return the most recent code snapshot."""
        return self._history[-1]

    @property
    def original(self) -> str:
        return self._original

    @property
    def depth(self) -> int:
        """Number of snapshots (including original)."""
        return len(self._history)
