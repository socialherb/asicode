"""
Kotlin syntax provider.

Uses ``kotlinc`` for validation and regex-based symbol detection.
Gracefully degrades when Kotlin toolchain is not installed.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import Optional

from .base import (
    SyntaxProvider,
    _compile_env,
    _filter_genuine_syntax_errors,
    _replace_last_cmd_path,
    _tempfile_for_content,
    detect_project_root,
    find_brace_block_end,
    find_brace_block_end_offset,
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
        has_linter=False,
        has_test_runner=True,
        has_symbol_search=True,
        has_tree_sitter=is_available(),
        supports_modify_symbol=True,
        supports_insert_after_symbol=True,
    )


# kotlinc error: file.kt:10:5: error: expecting member declaration
_KOTLINC_ERROR_RE = re.compile(
    r"^(.+?):(\d+):(\d+):\s+error:\s+(.+)$"
)


class KotlinSyntaxProvider(SyntaxProvider):
    """Kotlin language support (regex + tree-sitter symbols, kotlinc validation)."""

    _caps: Optional[LanguageCapabilities] = None

    def language_id(self) -> LanguageId:
        return LanguageId.KOTLIN

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate Kotlin source via ``kotlinc`` on *content* (written to temp file).

        Falls back to ``ok=True`` when kotlinc is not available.
        """
        _suffix = os.path.splitext(file_path)[1] or ".kt"
        _tmp_path, _cleanup = _tempfile_for_content(content, _suffix)
        if not _tmp_path:
            return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)
        # kotlinc is locale-independent (always outputs English diagnostics),
        # so no locale flags needed. -J-Duser.language=en is a no-op but kept
        # for defensive consistency with verifier.py.
        #
        # kotlinc has no "-fsyntax-only": it must write .class output to a real
        # directory. ``-script`` only works for ``.kts`` files — for a ``.kt``
        # file kotlinc aborts with "unrecognized script type" (rc=1, no
        # file:line: diagnostic), which the error regex silently discards,
        # turning this gate fail-open (every source, even a genuine syntax
        # error, returned ok=True). Mirror Java and hand kotlinc a throwaway
        # TemporaryDirectory via ``-d``.
        _out_dir = tempfile.TemporaryDirectory()
        _cmd = _replace_last_cmd_path(
            ["kotlinc", "-J-Duser.language=en", "-d", _out_dir.name, file_path],
            file_path, _tmp_path,
        )
        try:
            try:
                proc = subprocess.run(
                    _cmd,
                    capture_output=True, text=True, timeout=30,
                    cwd=os.path.dirname(_tmp_path) or ".",
                    env=_compile_env(),
                )
            except FileNotFoundError:
                logger.debug("kotlinc not installed; falling back to tree-sitter")
                return tree_sitter_syntax_fallback(content, LanguageId.KOTLIN, file_path)
            except subprocess.TimeoutExpired:
                logger.warning("kotlinc timed out for %s", file_path)
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)
            except Exception as e:
                logger.debug("kotlinc error: %s", e)
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

            errors: list[SyntaxError_] = []
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _KOTLINC_ERROR_RE.match(line)
                if m:
                    errors.append(SyntaxError_(
                        file=m.group(1),
                        line=int(m.group(2)),
                        col=int(m.group(3)),
                        message=m.group(4),
                    ))
            # Drop resolution/semantic failures (the isolated temp file has no
            # classpath, so any non-JDK import fails to resolve). Only genuine
            # syntax errors gate the edit; validate_semantics re-checks from the
            # project root with full context after the write.
            #
            # Kotlin-specific disambiguation: kotlinc emits "unresolved reference"
            # IDENTICALLY for a failed import AND a genuine local typo (``total +=
            # valeu``) — unlike Java's distinct "does not exist" vs "cannot find
            # symbol", there is no phrase-level signal. The only reliable proof
            # that an import failed in this classpath-less compile is an
            # unresolved reference reported ON AN IMPORT LINE (kotlinc flags the
            # unresolved package segment there). When that happens every
            # co-occurring unresolved reference is cascade noise; otherwise a
            # bare unresolved reference is a real typo that MUST gate.
            _import_lines = frozenset(
                i + 1 for i, ln in enumerate(content.splitlines())
                if ln.lstrip().startswith("import ")
            )
            _has_import_failure = any(
                "unresolved reference" in e.message.lower() and e.line in _import_lines
                for e in errors
            )
            errors = _filter_genuine_syntax_errors(
                errors, LanguageId.KOTLIN,
                has_resolution_context=_has_import_failure,
            )
            if not errors:
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

            return SyntaxValidationResult(ok=False, errors=errors, language=LanguageId.KOTLIN)
        finally:
            _cleanup()
            _out_dir.cleanup()

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``kotlinc`` on the **on-disk** file to catch semantic errors
        (unresolved references, type mismatches, missing imports).

        Unlike :meth:`validate_syntax` (which compiles an isolated temp file in
        ``-script`` mode), this compiles the real file from the project root so
        the backing tool can resolve imports / types against the surrounding
        project. This catches errors such as unresolved references that
        pure-syntax validation misses.

        Design choices:
        - Skips if no Kotlin/Gradle project marker is found (``build.gradle.kts``
          / ``pom.xml`` / ``build.gradle`` / ``settings.gradle.kts``) — no stable
          project root.
        - Uses ``-d <TemporaryDirectory>`` so compiled output is sandboxed and
          auto-cleaned, never polluting the source tree.
        - Skips (``ok=True``) when kotlinc is missing/timed out (non-blocking).
        """
        if not file_path or not os.path.exists(file_path):
            return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

        project_root = detect_project_root(
            file_path,
            markers=("build.gradle.kts", "pom.xml", "build.gradle", "settings.gradle.kts"),
        )
        _markers = ("build.gradle.kts", "pom.xml", "build.gradle", "settings.gradle.kts")
        if not any(
            os.path.isfile(os.path.join(project_root, m)) for m in _markers
        ):
            return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

        target_norm = os.path.normpath(os.path.abspath(file_path))
        out_dir = tempfile.TemporaryDirectory()
        try:
            # kotlinc is locale-independent (always outputs English diagnostics),
            # so no locale flags needed. -J-Duser.language=en is a no-op but kept
            # for defensive consistency with verifier.py.
            cmd = [
                "kotlinc",
                "-J-Duser.language=en",
                "-d", out_dir.name,
                file_path,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=120,
                    cwd=project_root,
                    env=_compile_env(),
                )
            except FileNotFoundError:
                logger.debug("kotlinc not installed; skipping semantic validation")
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)
            except subprocess.TimeoutExpired:
                logger.debug("kotlinc timed out for %s; skipping", file_path)
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)
            except Exception as e:
                logger.debug("kotlinc semantic check failed: %s", e)
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.KOTLIN)

            errors: list[SyntaxError_] = []
            has_error = False
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _KOTLINC_ERROR_RE.match(line)
                if not m:
                    continue
                _file, _line, _col, _msg = m.group(1), m.group(2), m.group(3), m.group(4)
                # Only report the file we asked about
                if _file and os.path.normpath(os.path.abspath(_file)) != target_norm:
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
                language=LanguageId.KOTLIN,
            )
        finally:
            out_dir.cleanup()

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:(?:public|private|protected|internal|override)\s+)*fun\s+{name}\s*[\(<]",
                description="Kotlin function declaration",
            ))
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"fun\s+\w+\.{name}\s*\(",
                description="Kotlin extension function",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"(?:(?:data|sealed|abstract|open|inner)\s+)*class\s+{name}\s*(?:\(|<|:|\{)",
                description="Kotlin class declaration",
            ))
        if kind in ("interface", "any"):
            patterns.append(SymbolPattern(
                kind="interface",
                regex=r"interface\s+{name}\s*(?:<|:|\{)",
                description="Kotlin interface declaration",
            ))
        if kind in ("type", "any"):
            patterns.append(SymbolPattern(
                kind="type",
                regex=r"object\s+{name}\s*(?::|\{)",
                description="Kotlin object declaration",
            ))
        if kind in ("enum", "any"):
            patterns.append(SymbolPattern(
                kind="enum",
                regex=r"enum\s+class\s+{name}\s*(?:\(|\{)",
                description="Kotlin enum class",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.kt", "*.kts"]

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return ["ktlint", file_path]

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        """Auto-detect Gradle."""
        if os.path.isfile(os.path.join(repo_root, "build.gradle.kts")) or \
           os.path.isfile(os.path.join(repo_root, "build.gradle")):
            return ["./gradlew", "test"] + (test_args or [])
        return ["./gradlew", "test"] + (test_args or [])

    # ── Symbol finder (tree-sitter → regex fallback) ──────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "kotlin")
            if result:
                return result

        return self._find_symbol_regex(file_path, symbol_name, content)

    def _find_symbol_regex(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                start_offset = m.start()
                start_line = content[:start_offset].count("\n") + 1
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
        """Regex fallback: find all top-level Kotlin definitions via pattern + brace counting."""
        results: list[tuple[str, str, int, int]] = []
        # Top-level functions: fun Name()  (not indented)
        for m in re.finditer(r'^fun\s+(\w+)\s*[\(<]', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "function", start_line, end_line))
        # Classes: (modifiers) class Name
        for m in re.finditer(
            r'^(?:(?:data|sealed|abstract|open|inner|public|private|protected|internal)\s+)*'
            r'class\s+(\w+)', content, re.MULTILINE,
        ):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "class", start_line, end_line))
        # Interfaces
        for m in re.finditer(r'^interface\s+(\w+)', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "interface", start_line, end_line))
        # Objects: object Name
        for m in re.finditer(r'^object\s+(\w+)\s*(?::|\{|$)', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "object", start_line, end_line))
        # Enum classes
        for m in re.finditer(r'^enum\s+class\s+(\w+)', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "enum", start_line, end_line))
        return results

    def _find_class_methods_regex(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: find methods inside a Kotlin class body."""
        results: list[tuple[str, int, int]] = []
        esc = re.escape(class_name)
        # Find class definition: class Name ... {
        pat = (
            r'(?:(?:data|sealed|abstract|open|inner|public|private|protected|internal)\s+)*'
            r'class\s+' + esc + r'\s*(?:\(|<|:|\{)'
        )
        for cm in re.finditer(pat, content):
            class_body_start = content.find("{", cm.start())
            if class_body_start == -1:
                continue
            class_end = self._find_block_end_offset(content, class_body_start)
            class_body = content[class_body_start:class_end]
            # Scan for method definitions inside class body
            for mm in re.finditer(
                r'(?:(?:public|private|protected|internal|override|suspend)\s+)*'
                r'fun\s+(\w+)\s*[\(<]',
                class_body,
            ):
                method_start = class_body_start + mm.start()
                method_line = content[:method_start].count("\n") + 1
                method_end = self._find_block_end(content, method_start)
                results.append((mm.group(1), method_line, method_end))
        return results

    @staticmethod
    def _find_block_end_offset(content: str, offset: int) -> int:
        """Offset (exclusive) of matching ``}`` for the class-body range.

        Delegates to :func:`base.find_brace_block_end_offset` (the shared SSOT) so
        braces inside string/char/template literals or comments cannot corrupt the
        depth counter. See base.py for the full contract.
        """
        return find_brace_block_end_offset(content, offset)

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
        result = find_all_symbols(content, "kotlin") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        from .tree_sitter_utils import extract_class_methods, is_available
        result = extract_class_methods(content, class_name, "kotlin") if is_available() else None
        if result:
            return result
        return self._find_class_methods_regex(content, class_name)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, "kotlin") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

    def get_definition_keywords(self) -> list[str]:
        return [
            "fun ", "class ", "interface ", "object ",
            "data class ", "sealed class ", "enum class ",
            "abstract class ", "open class ",
            "private fun ", "public fun ", "internal fun ",
            "override fun ", "suspend fun ",
        ]
