"""Lua syntax provider — tree-sitter AST symbol detection.

Symbol search + structural syntax validation.  No bundled Lua toolchain is
assumed, so syntax validation falls back to the tree-sitter error-node check
(``tree_sitter_syntax_fallback``), which is a hard gate on write paths.

Symbols are served via the provider index (``_nonpy_index_for``).  When the
``tree_sitter_lua`` grammar is installed (the normal case — it ships in the
core ``tree-sitter-language-pack`` dependency), ``_index_via_treesitter``
extracts ``function``/``function_definition`` nodes directly from the AST,
including the dotted ``function M.foo()`` method form.  The regex patterns
here are a fallback for the grammar-missing edge case only.

LUA was previously "half-wired": ``LanguageId.LUA``, ``_EXT_MAP``, the grammar
queries and ``comment_syntax`` all existed, but no provider was registered, so
``_nonpy_index_for`` (which iterates registered providers) never reached
``.lua`` files — ``find_symbol``/``modify_symbol`` returned empty with no
signal.  Registering this provider closes that silent-empty-results gap.
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


class LuaSyntaxProvider(SyntaxProvider):
    """Lua language support (AST symbol detection + tree-sitter syntax gate)."""

    def language_id(self) -> LanguageId:
        return LanguageId.LUA

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        # No bundled Lua toolchain — fall back to the tree-sitter syntax check,
        # which catches structural errors (this is a hard gate on write paths).
        return tree_sitter_syntax_fallback(content, LanguageId.LUA)

    # ── Symbol patterns ───────────────────────────────────────────────────
    #
    # Fallback for the tree-sitter AST path only.  _nonpy_index_for prefers
    # find_all_symbols (which catches ``function_definition`` nodes, including
    # the ``function M.foo()`` dotted method form) and only uses these regex
    # patterns when the tree_sitter_lua grammar is not installed.  Lua
    # identifiers are plain ``\w`` so there is no rg shell-arg hazard; the
    # broader ``[\w.]+`` capture preserves the dotted method name.

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "method", "any"):
            # function name(...) / function M.name(...) / function Obj:method(...)
            # The ``:`` colon form is Lua OOP (implicit self); ``[\w.:]+`` captures the
            # full qualified name so both dotted (M.foo) and colon (Account:withdraw)
            # methods index under their complete name rather than being truncated at
            # the separator.
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"\bfunction\s+{name}\s*\(",
                description="Lua function definition (function name(...))",
                name_capture=r"[\w.:]+",
            ))
            # local name = function(...)  — anonymous function bound to a local
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"\blocal\s+{name}\s*=\s*function\b",
                description="Lua local function expression (local name = function)",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        # Must stay in sync with _EXT_MAP / _LANGUAGE_EXTENSION_GROUPS /
        # _EXT_TO_GRAMMAR_KEY (all four list .lua → lua).  A glob dropped here
        # silently drops those files from the symbol index — see
        # TestGrammarMapConsistency.test_provider_globs_cover_ext_map.
        return ["*.lua"]

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
