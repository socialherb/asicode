"""failure_classifier.py — Classify verification errors into actionable types.

Maps raw error messages (from compile/javac/kotlinc/go) to FailureType enums
so the repair planner can dispatch to the right strategy.
"""
from __future__ import annotations

import re
from typing import Optional

from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor._editor_core.vm.classification import (
    Classification, EvidenceSource, FailureType, FixHint
)

__all__ = ["FailureType", "BaseFailureClassifier", "create_failure_classifier"]


class BaseFailureClassifier:
    """Base classifier that subclasses override with language-specific patterns."""

    # Subclasses define: error_code_map, keyword_map, regex_patterns, extract_symbol_re

    error_code_map: dict = {}
    keyword_map: list = []
    regex_patterns: list = []

    def classify(self, errors: list[VerifyError]) -> FailureType:
        if not errors:
            return FailureType.UNKNOWN
        return self._classify_single(errors[0])

    def classify_typed(
        self,
        errors: list[VerifyError],
        code: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Classification:
        """Typed classification returning Classification with evidence source.

        Layer A (tree-sitter): If code and language are provided, check for
        ERROR/MISSING nodes first. If found, return SYNTAX_ERROR with high confidence.

        Layer B/C: Fall back to error code map and keyword/regex matching.
        """
        if not errors:
            return Classification(type=FailureType.UNKNOWN, source=EvidenceSource.NONE)

        # Layer A: tree-sitter structural check (highest confidence for syntax errors)
        if code is not None and language is not None:
            try:
                from external_llm.languages.tree_sitter_utils import find_error_nodes
                error_nodes = find_error_nodes(code, language)
                if error_nodes:
                    # Found syntax errors — classify as SYNTAX_ERROR
                    node = error_nodes[0]  # Use first error node
                    fix_hint = None
                    if node.kind == "MISSING" and node.missing_token:
                        # MISSING node provides expected token (e.g. ";", ")")
                        fix_hint = FixHint(
                            kind="insert_token",
                            token=node.missing_token,
                            line=node.line + 1,  # 0-based → 1-based
                            column=node.column + 1,
                        )
                    return Classification(
                        type=FailureType.SYNTAX_ERROR,
                        source=EvidenceSource.TREE_SITTER,
                        fix_hint=fix_hint,
                        error_index=0,
                    )
            except Exception:
                # tree-sitter unavailable or failed — fall back to Layer B/C
                pass

        return self._classify_single_typed(errors[0], error_index=0, code=code, language=language)

    def classify_all(self, errors: list[VerifyError]) -> list[FailureType]:
        return [self._classify_single(e) for e in errors]

    def _classify_single(self, error: VerifyError) -> FailureType:
        if error.code and error.code in self.error_code_map:
            return self.error_code_map[error.code]

        msg = error.message.lower()
        for keyword, ftype in self.keyword_map:
            if keyword in msg:
                return ftype

        for pattern, ftype in self.regex_patterns:
            if pattern.search(msg):
                return ftype

        return FailureType.UNKNOWN

    def _classify_single_typed(
        self, error: VerifyError, error_index: int = 0,
        code: Optional[str] = None, language: Optional[str] = None,
    ) -> Classification:
        """Classify a single error with evidence source tracking.

        Phase 3: Uses position-based symbol extraction when code and language
        are provided, falling back to regex-based extraction otherwise.
        """
        # Layer B: error code map (typed mapping, not regex)
        if error.code and error.code in self.error_code_map:
            ftype = self.error_code_map[error.code]
            symbol = self._extract_symbol_structured(error, code, language)
            return Classification(
                type=ftype,
                source=EvidenceSource.ERROR_CODE,
                symbol=symbol,
                error_index=error_index,
            )

        msg = error.message.lower()
        # Layer C: keyword matching (fast path, no regex)
        for keyword, ftype in self.keyword_map:
            if keyword in msg:
                symbol = self._extract_symbol_structured(error, code, language)
                return Classification(
                    type=ftype,
                    source=EvidenceSource.MESSAGE_FALLBACK,
                    symbol=symbol,
                    error_index=error_index,
                )

        # Layer C: regex matching (capture groups, \d+, etc.)
        for pattern, ftype in self.regex_patterns:
            if pattern.search(msg):
                symbol = self._extract_symbol_structured(error, code, language)
                return Classification(
                    type=ftype,
                    source=EvidenceSource.MESSAGE_FALLBACK,
                    symbol=symbol,
                    error_index=error_index,
                )

        return Classification(type=FailureType.UNKNOWN, source=EvidenceSource.NONE, error_index=error_index)

    def _extract_symbol_structured(
        self, error: VerifyError, code: Optional[str], language: Optional[str],
    ) -> Optional[str]:
        """Extract symbol using position-based tree-sitter lookup (Phase 3).

        Falls back to regex-based extract_symbol() if tree-sitter unavailable
        or error lacks position information.
        """
        # Try position-based extraction first
        if code and language and error.line and error.column:
            try:
                from external_llm.languages.tree_sitter_utils import extract_symbol_at_position
                symbol = extract_symbol_at_position(code, language, error.line, error.column)
                if symbol:
                    return symbol
            except Exception:
                pass

        # Fallback to regex-based extraction
        return self.extract_symbol(error)

    def extract_symbol(self, error: VerifyError) -> Optional[str]:
        return None


# ── Python ────────────────────────────────────────────────────────────

# Python specific: compile() errors use different codes
_PY_ERROR_CODE_MAP = {
    # compile() / ast.parse error codes
    "E0602": FailureType.MISSING_VARIABLE,  # undefined variable
    "E9999": FailureType.SYNTAX_ERROR,       # SyntaxError
    # flake8 codes
    "F821": FailureType.MISSING_VARIABLE,    # undefined name
    "F811": FailureType.DUPLICATE_IDENTIFIER, # redefined function
    # pyright rule codes (from --outputjson)
    "reportUndefinedVariable": FailureType.MISSING_VARIABLE,
    "reportMissingImports": FailureType.MISSING_IMPORT,
    "reportInvalidSyntax": FailureType.SYNTAX_ERROR,
    "reportDuplicateImport": FailureType.DUPLICATE_IDENTIFIER,
    "reportMissingReturnType": FailureType.MISSING_RETURN,
    "reportGeneralTypeIssues": FailureType.TYPE_MISMATCH,
    "reportOptionalMemberAccess": FailureType.TYPE_MISMATCH,
    "reportCallIssue": FailureType.ARGUMENT_MISMATCH,
}

_PY_KEYWORD_MAP = [
    ("is not defined", FailureType.MISSING_VARIABLE),
    ("nameerror", FailureType.MISSING_VARIABLE),
    ("undefined name", FailureType.MISSING_VARIABLE),
    ("undefined variable", FailureType.MISSING_VARIABLE),
    ("cannot import", FailureType.MISSING_IMPORT),
    ("no module named", FailureType.MISSING_IMPORT),
    ("import-error", FailureType.MISSING_IMPORT),
    ("syntaxerror", FailureType.SYNTAX_ERROR),
    ("invalid syntax", FailureType.SYNTAX_ERROR),
    ("expected ':'", FailureType.SYNTAX_ERROR),
    ("unexpected indent", FailureType.SYNTAX_ERROR),
    ("unindent does not match", FailureType.SYNTAX_ERROR),
    ("duplicate argument", FailureType.DUPLICATE_IDENTIFIER),
    ("missing return", FailureType.MISSING_RETURN),
]

_PY_REGEX_PATTERNS = [
    (re.compile(r"name\s+['\"]?(\w+)['\"]?\s+is not defined", re.IGNORECASE), FailureType.MISSING_VARIABLE),
    (re.compile(r"module\s+['\"]?(\w+)['\"]?\s+not found", re.IGNORECASE), FailureType.MISSING_IMPORT),
    (re.compile(r"no\s+module\s+named\s+['\"]?(\w+)", re.IGNORECASE), FailureType.MISSING_IMPORT),
    (re.compile(r"takes\s+\d+\s+positional\s+arguments?\s+but\s+\d+", re.IGNORECASE), FailureType.ARGUMENT_MISMATCH),
    (re.compile(r"missing\s+\d+\s+required\s+positional\s+argument", re.IGNORECASE), FailureType.ARGUMENT_MISMATCH),
]

_PY_EXTRACT_SYMBOL = re.compile(r"name\s+['\"]?(\w+)['\"]?\s+is not defined", re.IGNORECASE)


class PythonFailureClassifier(BaseFailureClassifier):
    error_code_map = _PY_ERROR_CODE_MAP
    keyword_map = _PY_KEYWORD_MAP
    regex_patterns = _PY_REGEX_PATTERNS

    def extract_symbol(self, error: VerifyError) -> Optional[str]:
        m = _PY_EXTRACT_SYMBOL.search(error.message)
        if m:
            return m.group(1)
        # Try import error patterns
        m = re.search(r"cannot import\s+['\"]?(\w+)", error.message, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"module\s+['\"]?(\w+)['\"]?", error.message, re.IGNORECASE)
        if m:
            return m.group(1)
        return None


# ── Java ──────────────────────────────────────────────────────────────

# Java error codes from -XDrawDiagnostics (stable, locale-independent)
_JAVA_ERROR_CODE_MAP = {
    "compiler.err.cant.resolve.location": FailureType.UNKNOWN_SYMBOL,
    "compiler.err.cant.resolve": FailureType.UNKNOWN_SYMBOL,
    "compiler.err.doesnt.exist": FailureType.MISSING_IMPORT,
    "compiler.err.expected": FailureType.SYNTAX_ERROR,
    "compiler.err.unclosed.str.lit": FailureType.SYNTAX_ERROR,
    "compiler.err.unclosed.char.lit": FailureType.SYNTAX_ERROR,
    "compiler.err.duplicate.class": FailureType.DUPLICATE_IDENTIFIER,
    "compiler.err.missing.ret.stmt": FailureType.MISSING_RETURN,
    "compiler.err.prob.found.req": FailureType.TYPE_MISMATCH,
    "compiler.err.bad.op.types": FailureType.TYPE_MISMATCH,
    "compiler.err.not.stmt": FailureType.SYNTAX_ERROR,
}

_JAVA_KEYWORD_MAP = [
    ("cannot find symbol", FailureType.UNKNOWN_SYMBOL),
    ("package does not exist", FailureType.MISSING_IMPORT),
    ("cannot find", FailureType.UNKNOWN_SYMBOL),
    ("expected ';'", FailureType.SYNTAX_ERROR),
    ("unclosed", FailureType.SYNTAX_ERROR),
    ("duplicate class", FailureType.DUPLICATE_IDENTIFIER),
    ("missing return", FailureType.MISSING_RETURN),
    ("missing return statement", FailureType.MISSING_RETURN),
    ("incompatible types", FailureType.TYPE_MISMATCH),
    ("bad operand types", FailureType.TYPE_MISMATCH),
    ("not a statement", FailureType.SYNTAX_ERROR),
]

_JAVA_REGEX_PATTERNS = [
    (re.compile(r"cannot find symbol\s*\n?\s*symbol:\s+(.+)$", re.MULTILINE), FailureType.UNKNOWN_SYMBOL),
    (re.compile(r"';'\s+expected", re.IGNORECASE), FailureType.SYNTAX_ERROR),
    (re.compile(r"unclosed\s+(string|char)\s+literal", re.IGNORECASE), FailureType.SYNTAX_ERROR),
    (re.compile(r"class\s+\w+\s+is\s+public", re.IGNORECASE), FailureType.SYNTAX_ERROR),
]

_JAVA_EXTRACT_SYMBOL = re.compile(r"symbol:\s+(variable|method|class)\s+(\w+)", re.MULTILINE)


class JavaFailureClassifier(BaseFailureClassifier):
    error_code_map = _JAVA_ERROR_CODE_MAP
    keyword_map = _JAVA_KEYWORD_MAP
    regex_patterns = _JAVA_REGEX_PATTERNS

    def extract_symbol(self, error: VerifyError) -> Optional[str]:
        m = _JAVA_EXTRACT_SYMBOL.search(error.message)
        if m:
            return m.group(2)
        m = re.search(r"cannot find symbol[\s\S]*?symbol:[\s\S]*?(\w+)\s*$", error.message, re.MULTILINE)
        if m:
            return m.group(1)
        return None


# ── Kotlin ────────────────────────────────────────────────────────────

_KOTLIN_KEYWORD_MAP = [
    ("unresolved reference", FailureType.UNKNOWN_SYMBOL),
    ("unresolved", FailureType.UNKNOWN_SYMBOL),
    ("expecting", FailureType.SYNTAX_ERROR),
    ("expecting ';'", FailureType.SYNTAX_ERROR),
    ("expecting a", FailureType.SYNTAX_ERROR),
    ("no return", FailureType.MISSING_RETURN),
    ("a return is required", FailureType.MISSING_RETURN),
    ("type mismatch", FailureType.TYPE_MISMATCH),
    ("inferred type", FailureType.TYPE_MISMATCH),
    ("duplicate", FailureType.DUPLICATE_IDENTIFIER),
    ("conflicting overloads", FailureType.DUPLICATE_IDENTIFIER),
]

_KOTLIN_REGEX_PATTERNS = [
    (re.compile(r"unresolved reference:\s+(\w+)", re.IGNORECASE), FailureType.UNKNOWN_SYMBOL),
    (re.compile(r"expecting\s+['\"]?(;|}|\))['\"]?", re.IGNORECASE), FailureType.SYNTAX_ERROR),
]

_KOTLIN_EXTRACT_SYMBOL = re.compile(r"unresolved reference:\s+(\w+)", re.IGNORECASE)


class KotlinFailureClassifier(BaseFailureClassifier):
    error_code_map = {}
    keyword_map = _KOTLIN_KEYWORD_MAP
    regex_patterns = _KOTLIN_REGEX_PATTERNS

    def extract_symbol(self, error: VerifyError) -> Optional[str]:
        m = _KOTLIN_EXTRACT_SYMBOL.search(error.message)
        if m:
            return m.group(1)
        return None


# ── Go ────────────────────────────────────────────────────────────────

_GO_KEYWORD_MAP = [
    ("undefined", FailureType.UNKNOWN_SYMBOL),
    ("undeclared name", FailureType.UNKNOWN_SYMBOL),
    ("unused import", FailureType.UNUSED_IMPORT),
    ("imported and not used", FailureType.UNUSED_IMPORT),
    ("syntax error", FailureType.SYNTAX_ERROR),
    ("expected ';'", FailureType.SYNTAX_ERROR),
    ("expected '{'", FailureType.SYNTAX_ERROR),
    ("expected operand", FailureType.SYNTAX_ERROR),
    ("missing return", FailureType.MISSING_RETURN),
    ("too many arguments", FailureType.ARGUMENT_MISMATCH),
    ("not enough arguments", FailureType.ARGUMENT_MISMATCH),
    ("type mismatch", FailureType.TYPE_MISMATCH),
    ("cannot use", FailureType.TYPE_MISMATCH),
    ("redeclared", FailureType.DUPLICATE_IDENTIFIER),
    ("assignment mismatch", FailureType.ARGUMENT_MISMATCH),
]

_GO_REGEX_PATTERNS = [
    (re.compile(r"undefined:\s+(\w+)"), FailureType.UNKNOWN_SYMBOL),
    (re.compile(r"undeclared name:\s+(\w+)"), FailureType.UNKNOWN_SYMBOL),
    (re.compile(r"imported and not used:\s+['\"]?(\S+)['\"]?"), FailureType.UNUSED_IMPORT),
    (re.compile(r"expected\s+['\"]?(;|}|\))['\"]?"), FailureType.SYNTAX_ERROR),
]

_GO_EXTRACT_SYMBOL = re.compile(r"(?:undefined|undeclared name):\s+(\w+)")


class GoFailureClassifier(BaseFailureClassifier):
    error_code_map = {}
    keyword_map = _GO_KEYWORD_MAP
    regex_patterns = _GO_REGEX_PATTERNS

    def extract_symbol(self, error: VerifyError) -> Optional[str]:
        m = _GO_EXTRACT_SYMBOL.search(error.message)
        if m:
            return m.group(1)
        m = re.search(r"imported and not used:\s+['\"]?(\S+)", error.message)
        if m:
            return m.group(1)
        return None


# ── Factory ───────────────────────────────────────────────────────────

def create_failure_classifier(language: str) -> BaseFailureClassifier:
    """Factory: create the appropriate classifier for *language*."""
    _CLASSIFIERS = {
        "python": PythonFailureClassifier,
        "java": JavaFailureClassifier,
        "kotlin": KotlinFailureClassifier,
        "go": GoFailureClassifier,
        "typescript": None,
        "javascript": None,
    }
    cls = _CLASSIFIERS.get(language)
    if cls is None:
        raise ValueError(f"No failure classifier for language: {language}")
    return cls()
