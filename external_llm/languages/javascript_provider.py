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
from typing import Optional

from .base import SyntaxProvider, _replace_last_cmd_path, _tempfile_for_content, detect_project_root
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

    _caps: Optional[LanguageCapabilities] = None

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
            file_path, _tmp_path,
        )
        try:
            try:
                proc = subprocess.run(
                    _cmd,
                    capture_output=True, text=True, timeout=10,
                    cwd=os.path.dirname(_tmp_path) or ".",
                )
            except FileNotFoundError:
                logger.debug("node not installed; skipping JS validation")
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVASCRIPT)
            except subprocess.TimeoutExpired:
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVASCRIPT)
            except Exception:
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVASCRIPT)

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
                        r"^.*?:\d+(?::\d+)?\s*", "", line.strip(), count=1,
                    ).strip()
                    _detected_msg = _stripped or line.strip()
                    break
            if _detected_msg:
                errors.append(SyntaxError_(
                    file=file_path,
                    line=_detected_line,
                    col=0,
                    message=_detected_msg,
                ))

            if not errors and proc.returncode != 0:
                # Couldn't parse but node failed — report generic error
                errors.append(SyntaxError_(
                    file=file_path, line=0, col=0,
                    message=(proc.stderr or "syntax error").strip()[:200],
                ))

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
        files). Either config is accepted — see :meth:`_resolve_js_config` for
        the selection logic. Without any config the check is skipped to avoid
        tsc's environment/config noise.
        """
        config_filename = self._resolve_js_config(file_path)
        if config_filename is None:
            return SyntaxValidationResult(ok=True, language=LanguageId.JAVASCRIPT)
        return self._ts._run_tsc_semantic(
            file_path,
            language=LanguageId.JAVASCRIPT,
            config_markers=("jsconfig.json", "tsconfig.json"),
            config_filename=config_filename,
            allow_js=True,
        )

    def _resolve_js_config(self, file_path: str) -> Optional[str]:
        """Return the config filename (``jsconfig.json`` or ``tsconfig.json``)
        nearest to *file_path*.

        Prefers ``jsconfig.json`` (JS-native) but accepts ``tsconfig.json``
        when a JS project reuses the TS config. Returns ``None`` if neither is
        found in any ancestor — callers skip the check in that case.
        """
        project_root = detect_project_root(
            file_path, markers=("jsconfig.json", "tsconfig.json"),
        )
        for name in ("jsconfig.json", "tsconfig.json"):
            if os.path.isfile(os.path.join(project_root, name)):
                return name
        return None

    # ── Symbol patterns (JS subset of TS — no interface/type) ─────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:export\s+)?(?:async\s+)?function\s+{name}\s*\(",
                description="JS function declaration",
            ))
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:export\s+)?(?:const|let|var)\s+{name}\s*=\s*(?:async\s*)?\(",
                description="JS arrow / function expression",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"(?:export\s+)?class\s+{name}\s*(?:extends|\{)",
                description="JS class declaration",
            ))
        # JS has no interface/type keywords
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.js", "*.jsx", "*.mjs", "*.cjs"]

    # ── Lint / test commands (same as TS) ─────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return ["npx", "eslint", "--format=json", file_path]

    def get_test_directory(self, repo_root: str) -> Optional[str]:
        return self._ts.get_test_directory(repo_root)

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        return self._ts.get_test_command(repo_root, test_args)

    # ── Symbol finder (delegate to TS brace counting) ─────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Find symbol using tree-sitter (precise) or regex + brace counting (fallback)."""
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "javascript")
            if result:
                return result

        return self._find_symbol_regex(file_path, symbol_name, content)

    def _find_symbol_regex(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Fallback: regex + brace counting (same heuristic as TS)."""
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                start_offset = m.start()
                start_line = content[:start_offset].count("\n") + 1
                end_line = TypeScriptSyntaxProvider._find_block_end(content, start_offset)
                return (start_line, end_line)
        return None

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Regex fallback for structural queries ─────────────────────────────
    # JS shares TS infrastructure via TypeScriptSyntaxProvider static methods.

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: delegate to TS provider (same patterns)."""
        return self._ts._find_top_level_definitions_regex(content)

    def _find_class_methods_regex(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: delegate to TS provider (same patterns)."""
        return self._ts._find_class_methods_regex(content, class_name)

    def _find_symbol_body_range_regex(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Regex fallback: delegate to TS provider (same patterns)."""
        return self._ts._find_symbol_body_range_regex(content, symbol_name)

    # ── Structural query methods (tree-sitter → regex fallback) ────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        from .tree_sitter_utils import find_all_symbols, is_available
        result = find_all_symbols(content, "javascript") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        from .tree_sitter_utils import extract_class_methods, is_available
        result = extract_class_methods(content, class_name, "javascript") if is_available() else None
        if result:
            return result
        return self._find_class_methods_regex(content, class_name)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, "javascript") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

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
