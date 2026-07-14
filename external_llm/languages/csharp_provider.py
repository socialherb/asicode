"""C# syntax provider — regex-based symbol detection.

Symbol search only; syntax validation is non-blocking (returns ok=True)
since no bundled .NET toolchain is assumed. Symbols are served via the
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

# Method modifiers that may precede the return type + name. A modifier is
# required (matches the legacy _find_in_other_langs behavior — a bare
# "void DoStuff()" with no modifier was never matched). Class-ish modifiers
# are reused for class/interface/struct/enum/record prefixes.
_METHOD_MODS = (
    r"public|private|protected|internal|static|async|virtual|override|"
    r"abstract|sealed|partial|extern|new|unsafe|readonly"
)
_TYPE_MODS = r"public|private|protected|internal|static|abstract|sealed|partial"


class CSharpSyntaxProvider(SyntaxProvider):
    """C# language support (regex symbol detection)."""

    def language_id(self) -> LanguageId:
        return LanguageId.CSHARP

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        # No bundled C# toolchain — fall back to the tree-sitter syntax check,
        # which catches structural errors (this is a hard gate on write paths).
        return tree_sitter_syntax_fallback(content, LanguageId.CSHARP)

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "method", "any"):
            # <modifier(s)> <return-type> Name(  — return type may be generic
            # (e.g. Task<User>) or a dotted qualified name (ns.IFace.Method).
            patterns.append(SymbolPattern(
                kind="method",
                regex=rf"(?:{_METHOD_MODS})\s+(?:\w+\.)*\w+(?:<[^>]+>)?\s+{{name}}\s*\(",
                description="C# method declaration",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=rf"(?:{_TYPE_MODS}\s+)*class\s+{{name}}\b",
                description="C# class declaration",
            ))
            patterns.append(SymbolPattern(
                kind="interface",
                regex=rf"(?:{_TYPE_MODS}\s+)*interface\s+{{name}}\b",
                description="C# interface declaration",
            ))
            patterns.append(SymbolPattern(
                kind="struct",
                regex=rf"(?:{_TYPE_MODS}\s+)*struct\s+{{name}}\b",
                description="C# struct declaration",
            ))
            patterns.append(SymbolPattern(
                kind="enum",
                regex=rf"(?:{_TYPE_MODS}\s+)*enum\s+{{name}}\b",
                description="C# enum declaration",
            ))
            patterns.append(SymbolPattern(
                kind="record",
                regex=rf"(?:{_TYPE_MODS}\s+)*record\s+{{name}}\b",
                description="C# record declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.cs"]

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
