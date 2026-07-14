"""
Safe metadata access helper — single source of truth.

_safe_metadata — read-only safe access; returns {} when metadata is missing or non-dict.
"""
from __future__ import annotations

from typing import Any


def _safe_metadata(obj: Any) -> dict:
    metadata = getattr(obj, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}

