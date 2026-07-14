"""Rust syntax provider — regex-based symbol detection.

Symbol search only; syntax validation is non-blocking (returns ok=True)
since no bundled Rust toolchain is assumed. Symbols are served via the
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


class RustSyntaxProvider(SyntaxProvider):
    """Rust language support (regex symbol detection)."""

    def language_id(self) -> LanguageId:
        return LanguageId.RUST

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        # No bundled Rust toolchain — fall back to the tree-sitter syntax check,
        # which catches structural errors (this is a hard gate on write paths).
        return tree_sitter_syntax_fallback(content, LanguageId.RUST)

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "method", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"\bfn\s+{name}\s*[<(]",
                description="Rust function declaration",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="struct",
                regex=r"\bstruct\s+{name}\s*[<{]",
                description="Rust struct declaration",
            ))
            patterns.append(SymbolPattern(
                kind="trait",
                regex=r"\btrait\s+{name}\s*[<{]",
                description="Rust trait declaration",
            ))
            patterns.append(SymbolPattern(
                kind="enum",
                regex=r"\benum\s+{name}\s*[<{]",
                description="Rust enum declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.rs"]

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
