"""
Java syntax provider.

Uses ``javac`` for validation and regex-based symbol detection.
Gracefully degrades when Java toolchain is not installed.
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


# javac error: file.java:10: error: ';' expected
_JAVAC_ERROR_RE = re.compile(
    r"^(.+?):(\d+):\s+error:\s+(.+)$"
)

# package com.example.foo;  (optional; must precede type declarations)
_JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;")


def _java_source_root(file_path: str) -> str:
    """Derive the Java source root for *file_path* from its package declaration.

    ``javac -sourcepath`` must point at the directory containing the package
    hierarchy, **not** the project root — otherwise cross-file references to
    siblings in the same source tree fail to resolve. For a file declaring
    ``package com.example;`` at ``D/com/example/Main.java`` the source root is
    ``D``; for the default package (no declaration) it is the file's own
    directory. Falls back to the file's directory if the declared package does
    not match the on-disk layout.
    """
    file_dir = os.path.dirname(os.path.abspath(file_path)) or "."
    pkg: Optional[str] = None
    try:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            for _line in range(50):  # package decl precedes any type decl
                line = fh.readline()
                if not line:
                    break
                m = _JAVA_PACKAGE_RE.match(line)
                if m:
                    pkg = m.group(1)
                    break
    except OSError:
        return file_dir

    if not pkg:
        return file_dir  # default package → file's own directory

    pkg_rel = pkg.replace(".", os.sep)  # com/example
    dir_norm = os.path.normpath(file_dir).replace(os.sep, "/")
    pkg_norm = pkg_rel.replace(os.sep, "/")
    if dir_norm == pkg_norm or dir_norm.endswith("/" + pkg_norm):
        root = os.path.normpath(file_dir)
        for _ in range(pkg.count(".") + 1):
            root = os.path.dirname(root)
        return root or "."
    return file_dir  # layout mismatch → safe fallback


class JavaSyntaxProvider(SyntaxProvider):
    """Java language support (regex + tree-sitter symbols, javac validation)."""

    _caps: Optional[LanguageCapabilities] = None

    def language_id(self) -> LanguageId:
        return LanguageId.JAVA

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate Java source via ``javac`` on *content* (written to temp file).

        Falls back to ``ok=True`` when javac is not available.
        """
        _suffix = os.path.splitext(file_path)[1] or ".java"
        _tmp_path, _cleanup = _tempfile_for_content(content, _suffix)
        if not _tmp_path:
            return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)
        # javac has no "-fsyntax-only": it must write .class output to a real
        # directory. ``-d /dev/null`` makes javac abort with "not a directory"
        # BEFORE compiling anything (rc=2, no file:line: diagnostic) — which the
        # error regex silently discards, turning this gate fail-open (every
        # source, even a genuine syntax error, returned ok=True). Mirror
        # validate_semantics and hand javac a throwaway TemporaryDirectory.
        _out_dir = tempfile.TemporaryDirectory()
        _cmd = _replace_last_cmd_path(
            ["javac", "-d", _out_dir.name, file_path],
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
                logger.debug("javac not installed; falling back to tree-sitter")
                return tree_sitter_syntax_fallback(content, LanguageId.JAVA, file_path)
            except subprocess.TimeoutExpired:
                logger.warning("javac timed out for %s", file_path)
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)
            except Exception as e:
                logger.debug("javac error: %s", e)
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

            errors: list[SyntaxError_] = []
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _JAVAC_ERROR_RE.match(line)
                if m:
                    errors.append(SyntaxError_(
                        file=m.group(1),
                        line=int(m.group(2)),
                        col=0,
                        message=m.group(3),
                    ))
            # Drop resolution/semantic failures (the isolated temp file has no
            # -sourcepath/-classpath, so any non-JDK import fails to resolve).
            # Only genuine syntax errors gate the edit; validate_semantics
            # re-checks with the project sourcepath after the write.
            errors = _filter_genuine_syntax_errors(errors, LanguageId.JAVA)
            if not errors:
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

            return SyntaxValidationResult(ok=False, errors=errors, language=LanguageId.JAVA)
        finally:
            _cleanup()
            _out_dir.cleanup()

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``javac -sourcepath`` on the **on-disk** file to catch semantic
        errors (undefined symbols, type mismatches, missing imports).

        Unlike :meth:`validate_syntax` (which compiles an isolated temp file),
        this runs from the project root with ``-sourcepath <root>`` so cross-file
        references within the same source tree resolve correctly — e.g. ``Main``
        calling ``Helper.greet()`` in a sibling file.

        javac compiles dependencies on demand, so a build may surface errors from
        sibling files too; we filter to report only diagnostics whose file path
        matches *file_path* (same approach as the Go provider).

        Design choices:
        - Skips if no Java project marker is found (``pom.xml`` / ``build.gradle``
          / ``build.gradle.kts`` / ``settings.gradle``) — no stable source root.
        - Uses ``-d <TemporaryDirectory>`` so ``.class`` output is sandboxed and
          auto-cleaned, never polluting the source tree.
        - Skips (``ok=True``) when javac is missing/timed out (non-blocking).
        """
        if not file_path or not os.path.exists(file_path):
            return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

        project_root = detect_project_root(
            file_path,
            markers=("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"),
        )
        # detect_project_root falls back to the file's own dir; only proceed when
        # a real project marker is present.
        if not any(
            os.path.isfile(os.path.join(project_root, m))
            for m in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle")
        ):
            return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

        target_norm = os.path.normpath(os.path.abspath(file_path))
        # -sourcepath must be the source root (top of the package hierarchy),
        # NOT the project root — otherwise cross-file references to siblings
        # fail to resolve. Derived deterministically from the package decl.
        source_root = _java_source_root(file_path)
        out_dir = tempfile.TemporaryDirectory()
        try:
            cmd = [
                "javac",
                "-sourcepath", source_root,
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
                logger.debug("javac not installed; skipping semantic validation")
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)
            except subprocess.TimeoutExpired:
                logger.debug("javac timed out for %s; skipping", file_path)
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)
            except Exception as e:
                logger.debug("javac semantic check failed: %s", e)
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.JAVA)

            errors: list[SyntaxError_] = []
            has_error = False
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _JAVAC_ERROR_RE.match(line)
                if not m:
                    continue
                _file, _line, _msg = m.group(1), m.group(2), m.group(3)
                # Only report the file we asked about (javac compiles deps too)
                if _file and os.path.normpath(os.path.abspath(_file)) != target_norm:
                    continue
                errors.append(SyntaxError_(
                    file=file_path,
                    line=int(_line), col=0,
                    message=_msg,
                    severity="error",
                    code="",
                ))
                has_error = True
            return SyntaxValidationResult(
                ok=not has_error,
                errors=errors,
                language=LanguageId.JAVA,
            )
        finally:
            out_dir.cleanup()
    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+{name}\s*(?:extends|implements|<|\{)",
                description="Java class declaration",
            ))
        if kind in ("interface", "any"):
            patterns.append(SymbolPattern(
                kind="interface",
                regex=r"(?:public\s+|private\s+|protected\s+)?interface\s+{name}\s*(?:extends|<|\{)",
                description="Java interface declaration",
            ))
        if kind in ("enum", "any"):
            patterns.append(SymbolPattern(
                kind="enum",
                regex=r"(?:public\s+|private\s+|protected\s+)?enum\s+{name}\s*(?:implements|\{)",
                description="Java enum declaration",
            ))
        if kind in ("function", "method", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?[\w<>\[\]]+\s+{name}\s*\(",
                description="Java method declaration",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.java"]

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return None  # checkstyle optional — no default lint

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        """Auto-detect Maven or Gradle."""
        if os.path.isfile(os.path.join(repo_root, "pom.xml")):
            return ["mvn", "test"] + (test_args or [])
        if os.path.isfile(os.path.join(repo_root, "build.gradle")) or \
           os.path.isfile(os.path.join(repo_root, "build.gradle.kts")):
            return ["./gradlew", "test"] + (test_args or [])
        # Fallback: assume Maven
        return ["mvn", "test"] + (test_args or [])

    # ── Symbol finder (tree-sitter → regex fallback) ──────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Find symbol using tree-sitter (precise) or regex + brace counting (fallback)."""
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "java")
            if result:
                return result

        return self._find_symbol_regex(file_path, symbol_name, content)

    def _find_symbol_regex(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Fallback: regex + brace counting."""
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

    _JAVA_CLASS_PAT = re.compile(
        r'(?:(?:public|private|protected|abstract|final|static|sealed|non-sealed)\s+)*'
        r'(?:class|interface|enum|record|@interface)\s+(\w+)'
    )

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: find all top-level Java class/interface/enum/record definitions."""
        results: list[tuple[str, str, int, int]] = []
        for m in self._JAVA_CLASS_PAT.finditer(content):
            # Skip inner classes (indented relative to 0)
            line_start = content.rfind("\n", 0, m.start()) + 1 if m.start() > 0 else 0
            if content[line_start:m.start()].strip():
                continue  # has leading content on line — not top-level
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            kind = m.group(0).split()[-2] if len(m.group(0).split()) >= 2 else "class"
            results.append((m.group(1), kind, start_line, end_line))
        return results

    def _find_class_methods_regex(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: find methods inside a Java class body."""
        results: list[tuple[str, int, int]] = []
        # Find the class definition
        esc = re.escape(class_name)
        pat = r'(?:public|private|protected|static|final|synchronized|\s)*\s*(?:class|interface)\s+' + esc + r'\s*(?:extends|implements|<|\{|[^{]+?\{)'
        for cm in re.finditer(pat, content):
            class_body_start = content.find("{", cm.start())
            if class_body_start == -1:
                continue
            # Find matching closing brace for class
            class_body_end_offset = self._find_block_end_offset(content, class_body_start)
            class_body = content[class_body_start:class_body_end_offset]
            # Scan for methods inside class body
            for mm in re.finditer(
                r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?'
                r'(?:<[^>]+>\s+)?[\w<>\[\],\s]+\s+(\w+)\s*\(',
                class_body,
            ):
                method_start = class_body_start + mm.start()
                method_line = content[:method_start].count("\n") + 1
                method_end_line = self._find_block_end(content, method_start)
                results.append((mm.group(1), method_line, method_end_line))
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
        result = find_all_symbols(content, "java") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        from .tree_sitter_utils import extract_class_methods, is_available
        result = extract_class_methods(content, class_name, "java") if is_available() else None
        if result:
            return result
        return self._find_class_methods_regex(content, class_name)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, "java") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

    def get_definition_keywords(self) -> list[str]:
        return [
            "public ", "private ", "protected ",
            "class ", "interface ", "enum ",
            "abstract class ", "final class ",
            "static ",
        ]
