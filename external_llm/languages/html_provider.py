"""HTML syntax provider — stdlib html.parser based validation."""
from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Optional

# HTMLParseError was removed in Python 3.14; fall back to Exception for older versions.
try:
    from html.parser import HTMLParseError  # type: ignore[attr-defined]
except ImportError:
    HTMLParseError = Exception  # type: ignore[misc,assignment]

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

# Void elements that don't need a closing tag (HTML5 spec)
_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


class _TagTracker(HTMLParser):
    """Tracks open tags and collects parse errors."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._stack: list[tuple[str, int]] = []  # (tag, lineno)
        self.errors: list[tuple[int, str]] = []  # (lineno, message)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag not in _VOID_ELEMENTS:
            self._stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_ELEMENTS:
            return
        # Pop matching open tag (allow mismatches in the stack for leniency)
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                self._stack.pop(i)
                return
        # Closing tag with no matching open tag
        line = self.getpos()[0]
        self.errors.append((line, f"Unexpected closing tag </{tag}>"))

    def unclosed_tags(self) -> list[tuple[int, str]]:
        return [(lineno, f"Unclosed tag <{tag}>") for tag, lineno in self._stack]


class HtmlSyntaxProvider(SyntaxProvider):
    """HTML syntax provider using Python's stdlib html.parser.

    Validates:
    - Hard parse errors (HTMLParseError) — malformed attribute syntax, etc.
    - Unclosed non-void HTML tags
    - Unexpected closing tags

    Gracefully returns ok=True on unexpected internal errors so it is never
    a blocking failure when the parser itself cannot handle edge cases.
    """

    def language_id(self) -> LanguageId:
        return LanguageId.HTML

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        errors: list[SyntaxError_] = []
        try:
            tracker = _TagTracker()
            tracker.feed(content)

            # Collect tag-mismatch errors
            for lineno, msg in tracker.errors:
                errors.append(SyntaxError_(file=file_path, line=lineno, col=0, message=msg))
            for lineno, msg in tracker.unclosed_tags():
                errors.append(SyntaxError_(file=file_path, line=lineno, col=0, message=msg))

        except HTMLParseError as e:
            errors.append(SyntaxError_(
                file=file_path,
                line=getattr(e, "lineno", 0) or 0,
                col=getattr(e, "offset", 0) or 0,
                message=str(e.msg) if hasattr(e, "msg") else str(e),
            ))
        except Exception as e:
            logger.debug("HTML syntax check failed unexpectedly: %s", e)
            # Non-blocking fallback
            return SyntaxValidationResult(ok=True, language=LanguageId.HTML)

        return SyntaxValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            language=LanguageId.HTML,
        )

    # ── Unused abstract methods ───────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        return []

    def get_file_globs(self) -> list[str]:
        return ["*.html", "*.htm"]

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
