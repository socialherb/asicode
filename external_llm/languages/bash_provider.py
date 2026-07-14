"""Bash syntax provider — regex + tree-sitter symbol detection.

Symbol search only; syntax validation is non-blocking (returns ok=True)
since no bundled Bash toolchain (shellcheck) is assumed.  Symbols are
served via the provider index (_nonpy_index_for), which prefers the
tree-sitter AST (find_all_symbols — catches ``function_definition``
nodes including the C-style ``function name { ... }`` form) and falls
back to the regex patterns here when the grammar is not installed.

Unlike CSS, bash keeps a regex fallback because bash function names are
plain ``[A-Za-z_][\\w]*`` identifiers — no leading-dash rg shell-arg
trap exists, so the fallback is safe and keeps symbol search working
without the tree-sitter binding.
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


class BashSyntaxProvider(SyntaxProvider):
    """Bash/POSIX shell language support (regex + AST symbol detection)."""

    def language_id(self) -> LanguageId:
        return LanguageId.BASH

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        # No bundled bash toolchain — fall back to the tree-sitter syntax check,
        # which catches structural errors (this is a hard gate on write paths).
        return tree_sitter_syntax_fallback(content, LanguageId.BASH)

    # ── Symbol patterns ───────────────────────────────────────────────────
    #
    # These are a fallback for the tree-sitter AST path.  _nonpy_index_for
    # prefers find_all_symbols (which catches ``function_definition`` nodes,
    # including the C-style ``function name { ... }`` form), and only uses
    # these regex patterns when the tree_sitter_bash grammar is not
    # installed.  Because bash function names are plain identifiers, the
    # patterns have no rg shell-arg hazards.

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            # POSIX form: name() { ... }   (the { may sit on the next line)
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"\b{name}\s*\(\)\s*(?:\{|\n|$)",
                description="Bash POSIX function definition (name())",
            ))
            # C-style/Bash keyword form: function name { ... }
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"\bfunction\s+{name}\s*(?:\{|\n|$)",
                description="Bash keyword function definition (function name)",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.sh", "*.bash"]

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
