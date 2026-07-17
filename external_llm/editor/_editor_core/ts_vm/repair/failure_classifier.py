"""failure_classifier.py — Classify verification errors into actionable types.

Maps raw error messages (from tsc, parse, eslint) to FailureType enums
so the repair planner can dispatch to the right strategy.
"""
from __future__ import annotations

import re
from typing import Optional

from external_llm.editor._editor_core.ts_vm.execution_vm.models import VerifyError
from external_llm.editor._editor_core.vm.classification import FailureType
from external_llm.editor._editor_core.vm.failure_classifier import BaseFailureClassifier

__all__ = ["FailureType", "TSFailureClassifier"]


# TS error code → FailureType
_TSC_CODE_MAP = {
    "TS2304": FailureType.UNKNOWN_SYMBOL,      # Cannot find name 'X'
    "TS2305": FailureType.UNKNOWN_SYMBOL,       # Module has no exported member
    "TS2307": FailureType.MISSING_IMPORT,       # Cannot find module
    "TS2322": FailureType.TYPE_MISMATCH,        # Type X not assignable to Y
    "TS2345": FailureType.ARGUMENT_MISMATCH,    # Argument type not assignable
    "TS2554": FailureType.ARGUMENT_MISMATCH,    # Expected N args, got M
    "TS2355": FailureType.MISSING_RETURN,        # Function must return a value
    "TS2300": FailureType.DUPLICATE_IDENTIFIER,  # Duplicate identifier
    "TS2339": FailureType.PROPERTY_NOT_EXIST,    # Property does not exist on type
}

# ── Keyword-based classification (no regex) ────────────────────────────
# Long enough phrases — substring ``in`` suffices (no word-boundary needed).
_TS_KEYWORD_MAP = [
    ("cannot find module", FailureType.MISSING_IMPORT),
    ("is not assignable to", FailureType.TYPE_MISMATCH),
    ("must return a value", FailureType.MISSING_RETURN),
    ("duplicate identifier", FailureType.DUPLICATE_IDENTIFIER),
    ("parse error", FailureType.SYNTAX_ERROR),
    ("expected ';'", FailureType.SYNTAX_ERROR),
]

# ── Regex patterns (kept where truly needed) ────────────────────────────
# Capture groups (\w+) and \d+ make regex unavoidable here.
_TS_REGEX_PATTERNS = [
    (re.compile(r"cannot find name ['\"]?(\w+)", re.IGNORECASE), FailureType.UNKNOWN_SYMBOL),
    (re.compile(r"expected \d+ arguments?.+got \d+", re.IGNORECASE), FailureType.ARGUMENT_MISMATCH),
    (re.compile(r"argument.+not assignable", re.IGNORECASE), FailureType.ARGUMENT_MISMATCH),
    (re.compile(r"property.+does not exist", re.IGNORECASE), FailureType.PROPERTY_NOT_EXIST),
    (re.compile(r"expected\s+['\"]?[;})\\]'\"]", re.IGNORECASE), FailureType.SYNTAX_ERROR),
]

# ── Pre-compiled regexes for symbol extraction ─────────────────────────
_RE_CANNOT_FIND_NAME = re.compile(r"[Cc]annot find name ['\"](\w+)['\"]")
_RE_NO_EXPORTED_MEMBER = re.compile(r"no exported member ['\"](\w+)['\"]")
_RE_EXPECTED_ARGS = re.compile(r"[Ee]xpected (\d+) arguments?")


class TSFailureClassifier(BaseFailureClassifier):
    """Classifies TypeScript/JavaScript verification errors into FailureType.
    
    Inherits from BaseFailureClassifier to share Layer A (tree-sitter) and
    Layer B/C (error code + keyword/regex) classification logic.
    """

    # TS-specific error code map (Layer B)
    error_code_map = _TSC_CODE_MAP
    
    # TS-specific keyword and regex patterns (Layer C)
    keyword_map = _TS_KEYWORD_MAP
    regex_patterns = _TS_REGEX_PATTERNS

    def extract_symbol(self, error: VerifyError) -> Optional[str]:
        """Extract the missing symbol name from an error message."""
        msg = error.message
        m = _RE_CANNOT_FIND_NAME.search(msg)
        if m:
            return m.group(1)
        m = _RE_NO_EXPORTED_MEMBER.search(msg)
        if m:
            return m.group(1)
        return None

    def extract_expected_args(self, error: VerifyError) -> Optional[int]:
        """Extract expected argument count from error."""
        m = _RE_EXPECTED_ARGS.search(error.message)
        return int(m.group(1)) if m else None
