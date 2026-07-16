"""JSON syntax provider — validates JSON/JSONC files via stdlib json module."""
from __future__ import annotations

import json
from typing import Optional

from .base import SyntaxProvider
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)

_CAPABILITIES = LanguageCapabilities(
    has_syntax_validator=True,
)


class JsonSyntaxProvider(SyntaxProvider):
    """JSON language support backed by the stdlib ``json`` module.

    Supports .json and .jsonc files.  JSONC (JSON with Comments) is validated
    by stripping line-comments before parsing — this handles the most common
    case (tsconfig.json, .vscode/settings.json, etc.) without requiring an
    external dependency.
    """

    def language_id(self) -> LanguageId:
        return LanguageId.JSON

    def capabilities(self) -> LanguageCapabilities:
        return _CAPABILITIES

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate JSON by attempting json.loads().

        For .jsonc files, strips ``//`` line comments before parsing so that
        standard JSON-with-comments files (tsconfig, vscode settings) pass.
        """
        parse_content = content
        if file_path.endswith(".jsonc"):
            parse_content = self._strip_line_comments(content)

        try:
            json.loads(parse_content)
            return SyntaxValidationResult(ok=True, language=LanguageId.JSON)
        except json.JSONDecodeError as e:
            return SyntaxValidationResult(
                ok=False,
                errors=[SyntaxError_(
                    file=file_path,
                    line=e.lineno,
                    col=e.colno,
                    message=e.msg,
                )],
                language=LanguageId.JSON,
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _strip_line_comments(content: str) -> str:
        """Strip // line comments outside of strings."""
        lines = []
        for line in content.splitlines():
            stripped = JsonSyntaxProvider._strip_comment_from_line(line)
            lines.append(stripped)
        return "\n".join(lines)

    @staticmethod
    def _strip_comment_from_line(line: str) -> str:
        in_string = False
        escape = False
        for i, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string and line[i:i+2] == "//":
                return line[:i].rstrip()
        return line

    # ── Unused abstract methods (JSON has no symbols/lint/tests) ──────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        return []

    def get_file_globs(self) -> list[str]:
        return ["*.json", "*.jsonc"]

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
