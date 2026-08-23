"""classification.py — Typed failure classification for the VM subsystem.

This module defines the unified FailureType enum and Classification dataclass
for structured failure analysis. Replaces the duplicate enums in
vm/failure_classifier.py.

Design: docs/design/typed_failure_classifier.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureType(str, Enum):
    """Actionable failure categories (language-agnostic).

    Unified enum covering vm languages (Python/Java/Kotlin/Go).
    """

    MISSING_IMPORT = "missing_import"
    UNKNOWN_SYMBOL = "unknown_symbol"
    TYPE_MISMATCH = "type_mismatch"
    ARGUMENT_MISMATCH = "argument_mismatch"
    MISSING_RETURN = "missing_return"
    SYNTAX_ERROR = "syntax_error"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"
    MISSING_VARIABLE = "missing_variable"  # Python: NameError
    UNUSED_IMPORT = "unused_import"  # Go: imported and not used
    UNKNOWN = "unknown"


class EvidenceSource(str, Enum):
    """Where the classification evidence came from (for telemetry and debugging)."""

    TREE_SITTER = "tree_sitter"  # Layer A: structural (ERROR/MISSING nodes)
    ERROR_CODE = "error_code"  # Layer B: compiler diagnostic code (TS2304, pyright rule, etc.)
    MESSAGE_FALLBACK = "message"  # Layer C: keyword/regex on error message
    NONE = "none"  # UNKNOWN — no evidence


@dataclass(frozen=True)
class FixHint:
    """Structured hint for repair strategies (optional).

    Generated from tree-sitter MISSING nodes or compiler diagnostics.
    Example: MISSING ";" → FixHint(kind="insert_token", token=";", line=10, column=5)
    """

    kind: str  # "insert_token" | "remove_import" | "rename" | ...
    token: str | None = None  # Expected token (e.g. ";", ")")
    line: int | None = None  # 1-based line number
    column: int | None = None  # 1-based column number


@dataclass(frozen=True)
class Classification:
    """Typed classification result.

    Replaces bare FailureType return. Includes evidence source, extracted symbol,
    and optional fix hint. The extract_symbol() regex pass is absorbed here.
    """

    type: FailureType
    source: EvidenceSource
    symbol: str | None = None  # Extracted symbol (e.g. missing variable name)
    fix_hint: FixHint | None = None  # Structural hint for repair
    error_index: int = 0  # Which VerifyError triggered this (0 = first)
