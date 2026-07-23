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
        """Strip ``//`` line comments and ``/* */`` block comments outside strings.

        Block comments may span multiple lines, so block-comment state is carried
        across lines (JSONC, e.g. tsconfig.json / VS Code settings, supports both).
        """
        lines = []
        in_block_comment = False
        for line in content.splitlines():
            stripped, in_block_comment = JsonSyntaxProvider._strip_comment_from_line(
                line, in_block_comment
            )
            lines.append(stripped)
        return "\n".join(lines)

    @staticmethod
    def _strip_comment_from_line(
        line: str, in_block_comment: bool = False
    ) -> tuple[str, bool]:
        """Strip ``//`` and ``/* */`` comments outside of strings.

        Returns ``(stripped_line, new_in_block_comment_state)``. The caller threads
        the returned state into the next line so multi-line block comments are
        removed correctly.
        """
        out: list[str] = []
        in_string = False
        escape = False
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            pair = line[i:i + 2]
            if in_block_comment:
                # Inside a block comment: only the closer ends it.
                if pair == "*/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if escape:
                escape = False
                out.append(ch)
                i += 1
                continue
            if ch == "\\" and in_string:
                escape = True
                out.append(ch)
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                i += 1
                continue
            if not in_string:
                if pair == "//":
                    break  # rest of line is a line comment
                if pair == "/*":
                    in_block_comment = True
                    i += 2
                    continue
            out.append(ch)
            i += 1
        return "".join(out).rstrip(), in_block_comment

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
