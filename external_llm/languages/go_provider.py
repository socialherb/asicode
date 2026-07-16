"""
Go syntax provider.

Uses ``go build`` for validation and regex-based symbol detection.
Gracefully degrades when Go toolchain is not installed.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Optional

from .base import (
    SyntaxProvider,
    _compile_env,
    _filter_genuine_syntax_errors,
    _replace_last_cmd_path,
    _tempfile_for_content,
    detect_project_root,
    find_brace_block_end,
    tree_sitter_syntax_fallback,
)
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)

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


# go build error: file.go:10:5: expected ';', found 'EOF'
_GO_ERROR_RE = re.compile(
    r"^(.+?):(\d+):(\d+):\s+(.+)$"
)


class GoSyntaxProvider(SyntaxProvider):
    """Go language support (regex + tree-sitter symbols, go build validation)."""

    _caps: Optional[LanguageCapabilities] = None

    def language_id(self) -> LanguageId:
        return LanguageId.GO

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate Go source via ``go build`` on *content* (written to temp file).

        Falls back to ``ok=True`` when go is not available.

        Resolution safety: an isolated temp file is NOT part of the file's real
        Go module, so ``go build`` would emit module/import-resolution failures
        ("no required module provides package", "go.mod file not found") for any
        non-stdlib import — and those match the ``file:line:col:`` error shape,
        so without filtering they wrongly roll back valid edits. We therefore
        run from the module root (``go.mod``) when present so imports resolve
        via the module graph (command-line-arguments build mode), and drop any
        residual resolution errors via :func:`_filter_genuine_syntax_errors`.
        Only genuine syntax errors gate the edit; the on-disk
        :meth:`validate_semantics` pass re-checks with full package context.
        """
        _tmp_path, _cleanup = _tempfile_for_content(content, ".go")
        if not _tmp_path:
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)
        _cmd = _replace_last_cmd_path(
            ["go", "build", "-o", os.devnull, file_path],
            file_path, _tmp_path,
        )
        # Run from the Go MODULE root when available so imports resolve via the
        # module's dependency graph (command-line-arguments build mode). Without
        # module context `go build` emits resolution failures for every
        # non-stdlib import; those are filtered out below regardless of cwd.
        _module_root = detect_project_root(file_path, markers=("go.mod",))
        _cwd = (
            _module_root
            if os.path.isfile(os.path.join(_module_root, "go.mod"))
            else (os.path.dirname(_tmp_path) or ".")
        )
        try:
            try:
                proc = subprocess.run(
                    _cmd,
                    capture_output=True, text=True, timeout=30,
                    cwd=_cwd,
                    env=_compile_env(),
                )
            except FileNotFoundError:
                logger.debug("go not installed; falling back to tree-sitter")
                return tree_sitter_syntax_fallback(content, LanguageId.GO, file_path)
            except subprocess.TimeoutExpired:
                logger.debug("go build timed out for %s", file_path)
                return SyntaxValidationResult(ok=True, language=LanguageId.GO)
            except Exception as e:
                logger.debug("go build error: %s", e)
                return SyntaxValidationResult(ok=True, language=LanguageId.GO)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.GO)

            errors: list[SyntaxError_] = []
            raw_lines = (proc.stdout + proc.stderr).splitlines()
            i = 0
            while i < len(raw_lines):
                line = raw_lines[i]
                m = _GO_ERROR_RE.match(line)
                if m:
                    msg = m.group(4)
                    # Capture multi-line detail (e.g. "have (...)\n    want (...)")
                    # that follows "not enough arguments" / "too many arguments" errors
                    if "not enough arguments" in msg or "too many arguments" in msg:
                        j = i + 1
                        while j < len(raw_lines) and (
                            raw_lines[j].startswith("\t") or raw_lines[j].startswith("    ")
                        ):
                            msg += "\n" + raw_lines[j]
                            j += 1
                        i = j - 1  # -1 because i will be incremented after loop
                    errors.append(SyntaxError_(
                        file=m.group(1),
                        line=int(m.group(2)),
                        col=int(m.group(3)),
                        message=msg,
                    ))
                i += 1
            # Drop module/import-resolution failures (no syntax error in the
            # proposed content); the isolated temp file lacks the module graph
            # the real file lives in.
            errors = _filter_genuine_syntax_errors(errors, LanguageId.GO)
            if not errors:
                return SyntaxValidationResult(ok=True, language=LanguageId.GO)

            return SyntaxValidationResult(ok=False, errors=errors, language=LanguageId.GO)
        finally:
            _cleanup()

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``go build`` on the **on-disk** file's package directory.

        Go is compiled per-package, so unlike pyright/tsc this runs against the
        directory containing *file_path* (from the project root detected via
        ``go.mod``). This catches real semantic errors the config-blind single-
        file syntax check misses: undefined names, wrong-arity calls, type
        mismatches, and missing imports.

        Design choices:
        - Skips if there is no ``go.mod`` (no module → no stable import graph).
        - Runs ``go build ./<pkg-dir>`` (or ``.``) from the module root so all
          sibling files in the package are compiled together.
        - Only reports diagnostics whose file path matches *file_path* (a
          package build surfaces errors from sibling files too).
        - Errors make ``ok=False``; build warnings are surfaced.
        """
        if not file_path or not os.path.exists(file_path):
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)

        module_root = detect_project_root(file_path, markers=("go.mod",))
        if not os.path.isfile(os.path.join(module_root, "go.mod")):
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)

        # Package dir relative to module root
        pkg_dir_abs = os.path.dirname(os.path.abspath(file_path))
        try:
            pkg_rel = os.path.relpath(pkg_dir_abs, module_root)
        except ValueError:
            pkg_rel = "."
        pkg_target = "./" + pkg_rel if pkg_rel != "." else "."

        cmd = ["go", "build", pkg_target]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=120,
                cwd=module_root,
                env=_compile_env(),
            )
        except FileNotFoundError:
            logger.debug("go not installed; skipping semantic validation")
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)
        except subprocess.TimeoutExpired:
            logger.debug("go build timed out for %s; skipping", file_path)
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)
        except Exception as e:
            logger.debug("go build semantic check failed: %s", e)
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)

        if proc.returncode == 0:
            return SyntaxValidationResult(ok=True, language=LanguageId.GO)

        # Parse: ./pkg/file.go:10:5: undefined: foo
        target_norm = os.path.normpath(os.path.abspath(file_path))
        errors: list[SyntaxError_] = []
        has_error = False
        for line in (proc.stdout + proc.stderr).splitlines():
            m = _GO_ERROR_RE.match(line)
            if not m:
                continue
            _file, _line, _col, _msg = m.groups()
            # Only report the file we asked about (package build surfaces siblings)
            # removeprefix (Python 3.9+) is safe: lstrip("./") strips ANY
            # leading '/' or '.' characters, e.g. "..hidden" → "hidden".
            _normalized = _file.removeprefix("./") if _file.startswith("./") else _file
            _candidate = os.path.normpath(os.path.abspath(os.path.join(module_root, _normalized)))
            if _file and os.path.normpath(_file) != target_norm and _candidate != target_norm:
                continue
            errors.append(SyntaxError_(
                file=file_path,
                line=int(_line), col=int(_col),
                message=_msg,
                severity="error",
                code="",
            ))
            has_error = True
        return SyntaxValidationResult(
            ok=not has_error,
            errors=errors,
            language=LanguageId.GO,
        )

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"func\s+{name}\s*\(",
                description="Go function declaration",
            ))
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"func\s+\([^)]+\)\s+{name}\s*\(",
                description="Go method declaration (receiver)",
            ))
        if kind in ("type", "class", "any"):
            patterns.append(SymbolPattern(
                kind="type",
                regex=r"type\s+{name}\s+struct\s*\{",
                description="Go struct type",
            ))
            patterns.append(SymbolPattern(
                kind="interface",
                regex=r"type\s+{name}\s+interface\s*\{",
                description="Go interface type",
            ))
        if kind in ("variable", "constant", "any"):
            patterns.append(SymbolPattern(
                kind="variable",
                regex=r"var\s+{name}\b",
                description="Go var declaration",
            ))
            patterns.append(SymbolPattern(
                kind="constant",
                regex=r"const\s+{name}\b",
                description="Go const declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.go"]

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return ["golangci-lint", "run", file_path]

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        return ["go", "test", "./..."] + (test_args or [])

    # ── Symbol finder (tree-sitter → regex fallback) ──────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Find symbol using tree-sitter (precise) or regex + brace counting (fallback)."""
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "go")
            if result:
                return result

        return self._find_symbol_regex(file_path, symbol_name, content)

    def _find_symbol_regex(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Fallback: regex + brace counting (or line-based for var/const)."""
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                start_offset = m.start()
                start_line = content[:start_offset].count("\n") + 1
                if sp.kind in ("variable", "constant"):
                    # var/const declarations are usually single-line
                    end_pos = content.find("\n", m.end())
                    end_line = (content[:end_pos].count("\n") + 1) if end_pos != -1 else start_line
                else:
                    end_line = self._find_block_end(content, start_offset)
                return (start_line, end_line)
        return None

    @staticmethod
    def _find_block_end(content: str, offset: int) -> int:
        """Heuristic: find the matching closing brace from *offset*.

        Delegates to the shared :func:`find_brace_block_end` (C-family SSOT)
        which skips string/char/template literals and ``//`` / ``/* */``
        comments so braces inside them do not corrupt the depth counter.
        """
        return find_brace_block_end(content, offset)

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Regex fallback for structural queries ─────────────────────────────

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: find all top-level Go definitions via pattern + brace counting."""
        results: list[tuple[str, str, int, int]] = []
        for m in re.finditer(r'^func\s+(\w+)\s*\(', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "function", start_line, end_line))
        for m in re.finditer(r'^func\s+\([^)]*\)\s+(\w+)\s*\(', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "method", start_line, end_line))
        for m in re.finditer(r'^type\s+(\w+)\s+(struct|interface)\s*\{', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), m.group(2), start_line, end_line))
        for m in re.finditer(r'^var\s+(\w+)\b', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            # var declarations are single-line or grouped with parens
            end_pos = content.find("\n", m.end())
            if end_pos == -1:
                end_pos = len(content)
            end_line = content[:end_pos].count("\n") + 1
            results.append((m.group(1), "variable", start_line, end_line))
        for m in re.finditer(r'^const\s+(\w+)\b', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_pos = content.find("\n", m.end())
            if end_pos == -1:
                end_pos = len(content)
            end_line = content[:end_pos].count("\n") + 1
            results.append((m.group(1), "constant", start_line, end_line))
        return results

    def _find_class_methods_regex(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: find methods of a Go struct via receiver matching."""
        results: list[tuple[str, int, int]] = []
        esc = re.escape(class_name)
        for m in re.finditer(
            r'^func\s+\([^)]*?\b' + esc + r'\b[^)]*?\)\s+(\w+)\s*\(',
            content, re.MULTILINE,
        ):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), start_line, end_line))
        return results

    def _find_symbol_body_range_regex(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Regex fallback: find function body via first { after definition."""
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                body_start = content.find("{", m.end())
                if body_start == -1:
                    continue
                body_start_line = content[:body_start].count("\n") + 1
                body_end_line = self._find_block_end(content, body_start)
                return (body_start_line, body_end_line)
        return None

    # ── Structural query methods (tree-sitter → regex fallback) ────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        from .tree_sitter_utils import find_all_symbols, is_available
        result = find_all_symbols(content, "go") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        from .tree_sitter_utils import extract_class_methods, is_available
        result = extract_class_methods(content, class_name, "go") if is_available() else None
        if result:
            return result
        return self._find_class_methods_regex(content, class_name)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, "go") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

    def get_definition_keywords(self) -> list[str]:
        return ["func ", "type "]
