"""Scala syntax provider — tree-sitter AST symbol detection.

Symbol search + structural syntax validation.  No bundled Scala toolchain
(scalac) is assumed, so syntax validation falls back to the tree-sitter
error-node check (``tree_sitter_syntax_fallback``), which is a hard gate on
write paths.

Symbols are served via the provider index (``_nonpy_index_for``).  When the
``tree_sitter_scala`` grammar is installed (the normal case — it ships in the
core ``tree-sitter-language-pack`` dependency), ``_index_via_treesitter``
extracts ``object_definition``/``class_definition``/``function_definition``
nodes directly from the AST.  The regex patterns here are a fallback for the
grammar-missing edge case only.

SCALA was previously "half-wired": ``LanguageId.SCALA``, ``_EXT_MAP``, the
grammar queries and ``comment_syntax`` all existed, but no provider was
registered, so ``_nonpy_index_for`` never reached ``.scala``/``.sc`` files —
``find_symbol``/``modify_symbol`` returned empty with no signal.  Registering
this provider closes that silent-empty-results gap.
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


class ScalaSyntaxProvider(SyntaxProvider):
    """Scala language support (AST symbol detection + tree-sitter syntax gate)."""

    def language_id(self) -> LanguageId:
        return LanguageId.SCALA

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        # No bundled Scala toolchain — fall back to the tree-sitter syntax
        # check, which catches structural errors (this is a hard gate on write
        # paths).
        return tree_sitter_syntax_fallback(content, LanguageId.SCALA)

    # ── Symbol patterns ───────────────────────────────────────────────────
    #
    # Fallback for the tree-sitter AST path only.  _nonpy_index_for prefers
    # find_all_symbols (which catches def/class/object/trait definitions)
    # and only uses these regex patterns when the tree_sitter_scala grammar is
    # not installed.  Scala identifiers are plain ``\w`` so there is no rg
    # shell-arg hazard.  ``class`` also matches inside ``case class``, so no
    # separate pattern is needed for that form.

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "method", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                # ``def`` is Scala's ONLY method keyword, so a word boundary after the
                # name is sufficient and also catches parameterless idioms the previous
                # ``[\[(]``-requiring form silently dropped: ``def size = xs.length``,
                # ``def name: Int = ...``, ``def generic[T](...)`` (the ``[`` case still
                # matches). ``def`` never appears outside a definition, so there is no
                # over-match risk beyond the usual regex-in-comment limitation.
                regex=r"\bdef\s+{name}\b",
                description="Scala method definition (def name, incl. parameterless)",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"\b(?:class|object|trait|enum)\s+{name}\b",
                description="Scala class/object/trait/enum declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        # Must stay in sync with _EXT_MAP / _LANGUAGE_EXTENSION_GROUPS /
        # _EXT_TO_GRAMMAR_KEY (all four list .scala/.sc → scala).  A glob
        # dropped here silently drops those files from the symbol index — see
        # TestGrammarMapConsistency.test_provider_globs_cover_ext_map.
        return ["*.scala", "*.sc"]

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
