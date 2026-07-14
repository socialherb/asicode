"""classification.py — Typed failure classification (shared by vm and ts_vm).

This module defines the unified FailureType enum and Classification dataclass
for structured failure analysis. Replaces the duplicate enums in
vm/failure_classifier.py and ts_vm/repair/failure_classifier.py.

Design: docs/design/typed_failure_classifier.md
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureType(str, Enum):
    """Actionable failure categories (language-agnostic).

    Unified enum covering both vm (Python/Java/Kotlin/Go) and ts_vm (TS/JS).
    """

    MISSING_IMPORT = "missing_import"
    UNKNOWN_SYMBOL = "unknown_symbol"
    TYPE_MISMATCH = "type_mismatch"
    ARGUMENT_MISMATCH = "argument_mismatch"
    MISSING_RETURN = "missing_return"
    SYNTAX_ERROR = "syntax_error"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"
    PROPERTY_NOT_EXIST = "property_not_exist"      # TS: property does not exist on type
    MISSING_VARIABLE = "missing_variable"           # Python: NameError
    UNUSED_IMPORT = "unused_import"                 # Go: imported and not used
    UNKNOWN = "unknown"


class EvidenceSource(str, Enum):
    """Where the classification evidence came from (for telemetry and debugging)."""

    TREE_SITTER = "tree_sitter"       # Layer A: structural (ERROR/MISSING nodes)
    ERROR_CODE = "error_code"         # Layer B: compiler diagnostic code (TS2304, pyright rule, etc.)
    MESSAGE_FALLBACK = "message"      # Layer C: keyword/regex on error message
    NONE = "none"                     # UNKNOWN — no evidence


@dataclass(frozen=True)
class FixHint:
    """Structured hint for repair strategies (optional).

    Generated from tree-sitter MISSING nodes or compiler diagnostics.
    Example: MISSING ";" → FixHint(kind="insert_token", token=";", line=10, column=5)
    """

    kind: str                  # "insert_token" | "remove_import" | "rename" | ...
    token: Optional[str] = None       # Expected token (e.g. ";", ")")
    line: Optional[int] = None        # 1-based line number
    column: Optional[int] = None      # 1-based column number


@dataclass(frozen=True)
class Classification:
    """Typed classification result.

    Replaces bare FailureType return. Includes evidence source, extracted symbol,
    and optional fix hint. The extract_symbol() regex pass is absorbed here.
    """

    type: FailureType
    source: EvidenceSource
    symbol: Optional[str] = None          # Extracted symbol (e.g. missing variable name)
    fix_hint: Optional[FixHint] = None    # Structural hint for repair
    error_index: int = 0                  # Which VerifyError triggered this (0 = first)
