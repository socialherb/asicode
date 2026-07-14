"""
SyntaxValidator — language-agnostic syntax and symbol analysis facade.

Dispatches to ``LanguageRegistry`` providers (``SyntaxProvider`` subclasses)
for each supported language.  Python uses ``PythonSyntaxProvider`` (stdlib
``ast``), TS/JS/Go/Java/Kotlin use their respective providers (tree-sitter
or language-specific tools).

Previously used ``if lang == LanguageId.PYTHON`` branching — replaced with
provider dispatch so that adding a new language only requires registering
a ``SyntaxProvider`` subclass.

progressive TS/JS/Go/Java/Kotlin support.

Usage::

    from ..languages.syntax_validator import SyntaxValidator

    result = SyntaxValidator.validate_syntax(content, lang)
    rng = SyntaxValidator.find_symbol_range(content, "MyClass", lang)
    syms = SyntaxValidator.find_symbols(content, lang)
    body = SyntaxValidator.extract_symbol_body(code, name, lang)
"""
from __future__ import annotations

from typing import Any, Optional

from . import tree_sitter_utils as ts_utils
from .models import LanguageId, SyntaxError_, SyntaxValidationResult


def _get_provider(lang: LanguageId):
    """Return the ``SyntaxProvider`` for *lang*, or ``None`` if unsupported."""
    from .registry import LanguageRegistry

    return LanguageRegistry.instance().get_by_lang(lang)


class SyntaxValidator:
    """Unified syntax/symbol analysis — dispatches to per-language providers."""

    # ── Syntax validation ──────────────────────────────────────────────

    @staticmethod
    def validate_syntax(content: str, lang: LanguageId) -> SyntaxValidationResult:
        """Check *content* for syntax errors via the language provider."""
        provider = _get_provider(lang)
        if provider is not None:
            return provider.validate_syntax("", content)
        # Fallback: tree-sitter has_error
        has_err = ts_utils.has_error(content, lang.value)
        if has_err is None:
            return SyntaxValidationResult(ok=True, language=lang)
        if has_err:
            return SyntaxValidationResult(
                ok=False,
                errors=[SyntaxError_(file="", line=0, col=0, message="syntax error (tree-sitter)")],
                language=lang,
            )
        return SyntaxValidationResult(ok=True, language=lang)

    # ── Symbol range detection ─────────────────────────────────────────

    @staticmethod
    def find_symbol_range(
        content: str, symbol_name: str, lang: LanguageId,
    ) -> Optional[tuple[int, int]]:
        """Return ``(start_line, end_line)`` (1-indexed) for *symbol_name*.

        Python → ``ast.parse`` + walk.
        Others → ``tree_sitter_utils.find_symbol_range``.

        Returns None if not found or parser unavailable.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.find_symbol_range(content, symbol_name)
        return ts_utils.find_symbol_range(content, symbol_name, lang.value)

    # ── All symbols enumeration ────────────────────────────────────────

    @staticmethod
    def find_symbols(
        content: str, lang: LanguageId,
    ) -> list[tuple[str, str, int, int]]:
        """Enumerate all top-level symbols.

        Returns ``[(name, kind, start_line, end_line), ...]`` or empty list.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.find_symbols(content)
        return ts_utils.find_all_symbols(content, lang.value)

    # ── Symbol body extraction ─────────────────────────────────────────

    @staticmethod
    def extract_symbol_body(
        code: str, symbol_name: str, lang: LanguageId,
    ) -> Optional[tuple[int, int]]:
        """Return ``(body_start_line, body_end_line)`` for a function/method.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.extract_symbol_body(code, symbol_name)
        return ts_utils.extract_symbol_body(code, symbol_name, lang.value)

    # ── Dead code detection ────────────────────────────────────────────

    @staticmethod
    def is_dead_code_introduced(orig: str, new: str, lang: LanguageId) -> bool:
        """Check if *new* introduces dead code compared to *orig*.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.is_dead_code_introduced(orig, new)
        # Fallback: conservative — validate syntax of new code
        return not SyntaxValidator.validate_syntax(new, lang).ok

    # ── File-based symbol lookup ──────────────────────────────────────

    @staticmethod
    def find_symbol_in_file(
        file_path: str, symbol_name: str, content: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Read *file_path* and locate *symbol_name*, returning definition info."""

        lang = LanguageId.from_path(file_path)
        if lang == LanguageId.UNKNOWN:
            return None

        if content is None:
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                return None

        provider = _get_provider(lang)
        if provider is not None:
            result = provider.find_symbol_in_file(file_path, symbol_name, content)
            if result is not None:
                start_line, end_line = result
                # Determine kind from all-symbols enumeration
                kind: str = "symbol"
                for name, k, sl, _ in provider.find_symbols(content):
                    if name == symbol_name and sl == start_line:
                        kind = k
                        break
                return {
                    "file": file_path,
                    "line": start_line,
                    "end_line": end_line,
                    "kind": kind,
                    "name": symbol_name,
                }
            return None
        # Fallback: tree-sitter direct lookup
        return SyntaxValidator._ts_find_symbol_in_file(file_path, symbol_name, content, lang)

    @staticmethod
    def _ts_find_symbol_in_file(
        file_path: str, symbol_name: str, source: str, lang: LanguageId,
    ) -> Optional[dict[str, Any]]:
        """Tree-sitter based symbol lookup for non-Python files."""
        lang_str = lang.value
        rng = ts_utils.find_symbol_range(source, symbol_name, lang_str)
        if rng is None:
            return None
        start_line, end_line = rng
        # Determine kind from all-symbols enumeration (more reliable)
        kind: str = "symbol"
        for name, k, sl, _ in ts_utils.find_all_symbols(source, lang_str):
            if name == symbol_name and sl == start_line:
                kind = k
                break
        return {
            "file": file_path,
            "line": start_line,
            "end_line": end_line,
            "kind": kind,
            "name": symbol_name,
        }
