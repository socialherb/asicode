"""
JavaScript / JSX syntax provider.

Inherits symbol patterns, brace counting, and test runner from
TypeScriptSyntaxProvider.  Overrides validation (ESLint-only, no tsc)
and file globs.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess

from .base import (
    SyntaxProvider,
    _replace_last_cmd_path,
    _tempfile_for_content,
    tree_sitter_syntax_fallback,
)
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)
from .typescript_provider import TypeScriptSyntaxProvider

logger = logging.getLogger(__name__)


def _make_capabilities() -> LanguageCapabilities:
    from .tree_sitter_utils import is_available

    return LanguageCapabilities(
        has_ast_parser=False,
        has_syntax_validator=True,
        has_semantic_validator=True,
        has_linter=True,
        has_test_runner=True,
        has_symbol_search=True,
        has_tree_sitter=is_available(),
        supports_modify_symbol=True,
        supports_insert_after_symbol=True,
    )


class JavaScriptSyntaxProvider(SyntaxProvider):
    """JavaScript language support.

    Shares symbol patterns and brace counting with TypeScript but uses
    Node.js ``--check`` for syntax validation instead of ``tsc``.
    """

    # Reuse the TS provider for shared logic (symbol finder, brace counting)
    _ts = TypeScriptSyntaxProvider()

    _caps: LanguageCapabilities | None = None

    def language_id(self) -> LanguageId:
        return LanguageId.JAVASCRIPT

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate JavaScript via ``node --check`` on *content* (written to temp file).

        Falls back to ``ok=True`` when node is not available.
        """
        _suffix = os.path.splitext(file_path)[1] or ".js"
        _tmp_path, _cleanup = _tempfile_for_content(content, _suffix)
        if not _tmp_path:
            return SyntaxValidationResult(ok=True, language=LanguageId.JAVASCRIPT)
        _cmd = _replace_last_cmd_path(
            ["node", "--check", file_path],
            file_path,
            _tmp_path,
        )
        try:
            try:
                proc = subprocess.run(
                    _cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=os.path.dirname(_tmp_path) or ".",
                    check=False,
                )
            except FileNotFoundError:
                logger.debug("node not installed; falling back to tree-sitter")
                return tree_sitter_syntax_fallback(content, LanguageId.JAVASCRIPT, file_path)
            except subprocess.TimeoutExpired:
                return tree_sitter_syntax_fallback(content, LanguageId.JAVASCRIPT, file_path)
            except Exception:
                return tree_sitter_syntax_fallback(content, LanguageId.JAVASCRIPT, file_path)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVASCRIPT)

            # Parse node --check error output
            # Format (modern node, stderr multi-line):
            #   /path/file.js:LINE
            #   offending_line
            #   ^
            #   SyntaxError: message
            # Or (older node, stderr single-line):
            #   /path/file.js:LINE:COL  SyntaxError: message
            errors: list[SyntaxError_] = []
            _detected_line = 0
            _detected_msg = ""
            _stderr_lines = (proc.stderr or "").splitlines()
            for i, line in enumerate(_stderr_lines):
                # 1) Look for the SyntaxError message line (primary — multi-line format)
                _syn_err_m = re.search(r"SyntaxError:\s*(.*)", line, re.IGNORECASE)
                if _syn_err_m:
                    _detected_msg = _syn_err_m.group(1).strip() or line.strip()
                    # Find the nearest preceding file:line marker for the line number
                    for j in range(i - 1, -1, -1):
                        _loc_m = re.search(r":(\d+)\b", _stderr_lines[j])
                        if _loc_m:
                            _detected_line = int(_loc_m.group(1))
                            break
                    break
                # 2) Fallback: file:line marker (single-line format:
                #    /path/file.js:LINE  SyntaxError: message)
                m = re.search(r":(\d+)\b\s+.*SyntaxError", line, re.IGNORECASE)
                if m:
                    _detected_line = int(m.group(1))
                    # Strip file:line:col prefix from the same line
                    _stripped = re.sub(
                        r"^.*?:\d+(?::\d+)?\s*",
                        "",
                        line.strip(),
                        count=1,
                    ).strip()
                    _detected_msg = _stripped or line.strip()
                    break
            if _detected_msg:
                errors.append(
                    SyntaxError_(
                        file=file_path,
                        line=_detected_line,
                        col=0,
                        message=_detected_msg,
                    )
                )

            if not errors and proc.returncode != 0:
                # Couldn't parse but node failed — report generic error
                errors.append(
                    SyntaxError_(
                        file=file_path,
                        line=0,
                        col=0,
                        message=(proc.stderr or "syntax error").strip()[:200],
                    )
                )

            return SyntaxValidationResult(
                ok=len(errors) == 0,
                errors=errors,
                language=LanguageId.JAVASCRIPT,
            )
        finally:
            _cleanup()

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``tsc --noEmit`` on the **on-disk** JS file via the shared
        project-mode check (delegates to ``TypeScriptSyntaxProvider``).

        ``--checkJs`` opts the JS file into tsc's semantic analysis (undefined
        names TS2304, missing imports TS2307, type mismatches TS2xxx) which
        ``node --check`` (the syntax validator) cannot catch.

        JS projects configure tsc via ``jsconfig.json``; TS projects use
        ``tsconfig.json`` (whose ``allowJs``/``checkJs`` may also cover ``.js``
        files). Either config is accepted — see :meth:`_config_at_root` for the
        selection logic. Without any config the check is skipped to avoid tsc's
        environment/config noise.
        """
        return self.validate_semantics_batch([file_path])[file_path]

    def validate_semantics_batch(
        self,
        file_paths: list[str],
    ) -> dict[str, SyntaxValidationResult]:
        """Semantic-check *file_paths* with one tsc run per (project, config).

        Same batching as the TS provider — see
        :meth:`TypeScriptSyntaxProvider.validate_semantics_batch` — with the
        extra split on which config a root carries: a JS project may use
        ``jsconfig.json`` or ``tsconfig.json``, and the temp config can only
        extend one of them, so the two cannot share a run.
        """
        return self._ts._batch_by_root(
            file_paths,
            language=LanguageId.JAVASCRIPT,
            config_markers=("jsconfig.json", "tsconfig.json"),
            config_for=self._config_at_root,
            allow_js=True,
        )

    @staticmethod
    def _config_at_root(project_root: str) -> str | None:
        """Which config filename *project_root* carries, preferring jsconfig."""
        for name in ("jsconfig.json", "tsconfig.json"):
            if os.path.isfile(os.path.join(project_root, name)):
                return name
        return None

    # ── Symbol patterns (JS subset of TS — no interface/type) ─────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(
                SymbolPattern(
                    kind="function",
                    regex=r"(?:export\s+)?(?:async\s+)?function\s*\*?\s*{name}\s*\(",
                    description="JS function declaration",
                )
            )
            patterns.append(
                SymbolPattern(
                    kind="function",
                    regex=r"(?:export\s+)?(?:const|let|var)\s+{name}\s*=\s*(?:async\s*)?\(",
                    description="JS arrow / function expression",
                )
            )
        if kind in ("class", "any"):
            patterns.append(
                SymbolPattern(
                    kind="class",
                    regex=r"(?:export\s+)?class\s+{name}\s*(?:extends|\{)",
                    description="JS class declaration",
                )
            )
        # JS has no interface/type keywords
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.js", "*.jsx", "*.mjs", "*.cjs"]

    # ── Lint / test commands (same as TS) ─────────────────────────────────

    def get_lint_command(self, file_path: str) -> list[str] | None:
        return ["npx", "eslint", "--format=json", file_path]

    def get_test_directory(self, repo_root: str) -> str | None:
        return self._ts.get_test_directory(repo_root)

    def get_test_command(self, repo_root: str, test_args: list[str] | None = None) -> list[str] | None:
        return self._ts.get_test_command(repo_root, test_args)

    # ── Symbol finder (delegate to TS brace counting) ─────────────────────

    def find_symbol_in_file(self, file_path: str, symbol_name: str, content: str) -> tuple[int, int] | None:
        """Find symbol using tree-sitter (precise) or regex + brace counting (fallback)."""
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "javascript")
            if result:
                return result

        return self._find_symbol_regex(symbol_name, content, js_lexing=True)

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Regex fallback for structural queries ─────────────────────────────
    # JS shares TS infrastructure via TypeScriptSyntaxProvider static methods.

    def _find_top_level_definitions_regex(
        self,
        content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: delegate to TS provider (same patterns)."""
        return self._ts._find_top_level_definitions_regex(content)

    def _find_class_methods_regex(
        self,
        content: str,
        class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: delegate to TS provider (same patterns)."""
        return self._ts._find_class_methods_regex(content, class_name)

    # ── Structural query methods (tree-sitter → regex fallback) ────────────

    def find_top_level_definitions(
        self,
        content: str,
    ) -> list[tuple[str, str, int, int]]:
        from .tree_sitter_utils import find_all_symbols, is_available

        result = find_all_symbols(content, "javascript") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self,
        content: str,
        class_name: str,
    ) -> list[tuple[str, int, int]]:
        from .tree_sitter_utils import extract_class_methods, is_available

        result = extract_class_methods(content, class_name, "javascript") if is_available() else None
        if result:
            return result
        return self._find_class_methods_regex(content, class_name)

    def find_symbol_body_range(
        self,
        content: str,
        symbol_name: str,
    ) -> tuple[int, int] | None:
        from .tree_sitter_utils import extract_symbol_body, is_available

        result = extract_symbol_body(content, symbol_name, "javascript") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name, js_lexing=True)

    def get_definition_keywords(self) -> list[str]:
        return [
            "function ",
            "async function ",
            "class ",
            "const ",
            "let ",
            "var ",
            "export function ",
            "export async function ",
            "export class ",
            "export const ",
            "export default function ",
        ]
