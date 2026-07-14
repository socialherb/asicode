"""PHP syntax provider — regex-based symbol detection.

Symbol search only; syntax validation is non-blocking (returns ok=True)
since no bundled PHP toolchain is assumed. Symbols are served via the
provider index (_nonpy_index_for), removing the need for the legacy
hardcoded rg fallback.
"""
from __future__ import annotations

from typing import Optional

from .base import SyntaxProvider, tree_sitter_syntax_fallback
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxValidationResult,
)

_CAPABILITIES = LanguageCapabilities(
    has_syntax_validator=True,
    has_symbol_search=True,
)


class PhpSyntaxProvider(SyntaxProvider):
    """PHP language support (regex symbol detection)."""

    def language_id(self) -> LanguageId:
        return LanguageId.PHP

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        # No bundled PHP toolchain — fall back to the tree-sitter syntax check,
        # which catches structural errors (this is a hard gate on write paths).
        return tree_sitter_syntax_fallback(content, LanguageId.PHP)

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "method", "any"):
            # Plain function and class method (optional visibility modifiers).
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"\bfunction\s+{name}\s*\(",
                description="PHP function declaration",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"\b(?:final\s+|abstract\s+)?class\s+{name}\b",
                description="PHP class declaration",
            ))
            patterns.append(SymbolPattern(
                kind="interface",
                regex=r"\binterface\s+{name}\b",
                description="PHP interface declaration",
            ))
            patterns.append(SymbolPattern(
                kind="trait",
                regex=r"\btrait\s+{name}\b",
                description="PHP trait declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.php"]

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return None

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        return None

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        return None

    def get_definition_keywords(self) -> list[str]:
        return []
