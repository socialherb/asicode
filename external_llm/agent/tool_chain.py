"""
Tool Chain Manager and Scoped Tool Filter for asicode Agent (simplified).

Previously contained ToolChainManager with dependency graph and regex-based
tool suggestions. Those were removed — LLMs are better at choosing the next
tool autonomously. Only ScopedToolFilter remains.
"""
from __future__ import annotations

from typing import Optional


class ScopedToolFilter:
    """File access filter for scoped delegation.

    Restricts which files can be written/read during delegated execution.
    Completed ops' files are read-only to prevent corruption.
    """

    def __init__(
        self,
        allowed_write: Optional[set[str]] = None,
        readonly_files: Optional[set[str]] = None,
    ):
        self._allowed_write = allowed_write  # None = no restriction
        self._readonly = readonly_files or set()

    def can_write(self, path: str) -> bool:
        """Check if writing to path is allowed."""
        if self._allowed_write is None:
            return path not in self._readonly
        return path in self._allowed_write and path not in self._readonly

    def can_read(self, path: str) -> bool:
        """Check if reading path is allowed. Always True (no read restriction)."""
        return True
