"""CSS syntax provider — brace-balance and comment validation."""
from __future__ import annotations

import logging
from typing import Optional

from .base import SyntaxProvider
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)

logger = logging.getLogger(__name__)

_CAPABILITIES = LanguageCapabilities(
    has_syntax_validator=True,
)


class CssSyntaxProvider(SyntaxProvider):
    """CSS/SCSS/Less syntax provider.

    Validates by tracking:
    - Brace balance (unclosed or extra closing braces)
    - Unclosed block comments (/* ... */)
    - Unclosed string literals (' or ")

    This is intentionally lightweight — it catches the most common edit-induced
    errors (mismatched braces, broken comments) without requiring an external
    CSS parser.  Gracefully returns ok=True on unexpected inputs.
    """

    def language_id(self) -> LanguageId:
        return LanguageId.CSS

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        errors: list[SyntaxError_] = []
        try:
            errors = self._check(file_path, content)
        except Exception as e:
            logger.debug("CSS syntax check failed unexpectedly: %s", e)
            # Non-blocking: unknown errors don't fail the edit
            return SyntaxValidationResult(ok=True, language=LanguageId.CSS)

        return SyntaxValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            language=LanguageId.CSS,
        )

    def _check(self, file_path: str, content: str) -> list[SyntaxError_]:
        errors: list[SyntaxError_] = []
        depth = 0           # brace nesting depth
        in_string: Optional[str] = None    # '"' or "'"
        in_comment = False  # inside /* ... */
        line = 1
        i = 0
        n = len(content)

        while i < n:
            ch = content[i]

            # Track line numbers
            if ch == "\n":
                line += 1

            # Inside block comment
            if in_comment:
                if content[i:i+2] == "*/":
                    in_comment = False
                    i += 2
                    continue
                i += 1
                continue

            # Inside string literal
            if in_string:
                if ch == "\\" and i + 1 < n:
                    i += 2  # skip escaped char
                    continue
                if ch == in_string:
                    in_string = None
                i += 1
                continue

            # Start of block comment
            if content[i:i+2] == "/*":
                in_comment = True
                i += 2
                continue

            # Line comment (SCSS/Less support //)
            if content[i:i+2] == "//":
                while i < n and content[i] != "\n":
                    i += 1
                continue

            # String start
            if ch in ('"', "'"):
                in_string = ch
                i += 1
                continue

            # Brace tracking
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append(SyntaxError_(
                        file=file_path,
                        line=line,
                        col=0,
                        message="Unexpected '}' — no matching '{'",
                    ))
                    depth = 0  # reset to continue checking

            i += 1

        # End-of-file checks
        if in_comment:
            errors.append(SyntaxError_(
                file=file_path, line=line, col=0,
                message="Unclosed block comment (/* ... */)",
            ))
        if in_string:
            errors.append(SyntaxError_(
                file=file_path, line=line, col=0,
                message=f"Unclosed string literal ({in_string!r})",
            ))
        if depth > 0:
            errors.append(SyntaxError_(
                file=file_path, line=line, col=0,
                message=f"Unclosed block — {depth} unmatched " + "'{'"
            ))

        return errors

    # ── Symbol patterns ──────────────────────────────────────────────────
    #
    # CSS symbols are extracted directly from the tree-sitter AST (see
    # tree_sitter_utils.find_all_symbols: class_selector→class_name,
    # id_selector→id_name, declaration with a "--"-prefixed property_name).
    # That AST path is the single source of truth for CSS, so this provider
    # contributes no regex patterns: _nonpy_index_for routes CSS through
    # find_all_symbols when the tree_sitter_css binding is installed, and the
    # "--name" leading-dash rg shell-arg trap never arises because no pattern
    # is ever handed to rg.

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        return []

    def get_file_globs(self) -> list[str]:
        return ["*.css", "*.scss", "*.less"]

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
