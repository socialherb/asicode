"""
Python syntax provider — wraps existing AST-based logic.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import replace
from typing import Optional

from .base import SyntaxProvider, detect_project_root
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)

logger = logging.getLogger(__name__)


def _tree_sitter_available() -> bool:
    """Dynamically check if tree-sitter is available for Python."""
    try:
        from .tree_sitter_utils import get_parser, is_available

        return is_available() and get_parser("python") is not None
    except Exception:
        return False



_CAPABILITIES = LanguageCapabilities(
    has_ast_parser=True,
    has_syntax_validator=True,
    has_semantic_validator=True,
    has_linter=True,
    has_test_runner=True,
    has_symbol_search=True,
    supports_modify_symbol=True,
    supports_insert_after_symbol=True,
)


class PythonSyntaxProvider(SyntaxProvider):
    """Python language support backed by the stdlib ``ast`` module."""

    def language_id(self) -> LanguageId:
        return LanguageId.PYTHON

    def capabilities(self) -> LanguageCapabilities:
        caps = _CAPABILITIES
        # Dynamically reflect parser availability —
        # use replace() to avoid mutating the module-level object.
        if _tree_sitter_available():
            caps = replace(caps, has_tree_sitter=True)
        return caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate Python source via ``ast.parse`` + ``compile``."""
        errors: list[SyntaxError_] = []
        # 1. AST parse
        try:
            ast.parse(content, filename=file_path)
        except SyntaxError as e:
            errors.append(SyntaxError_(
                file=file_path,
                line=e.lineno or 0,
                col=e.offset or 0,
                message=f"Syntax error: {e.msg}",
            ))
            return SyntaxValidationResult(ok=False, errors=errors, language=LanguageId.PYTHON)

        # 2. compile() — stricter, catches some issues AST doesn't
        try:
            compile(content, file_path, "exec")
        except SyntaxError as e:
            errors.append(SyntaxError_(
                file=file_path,
                line=e.lineno or 0,
                col=e.offset or 0,
                message=f"Compile error: {e.msg}",
            ))
        except ValueError as e:
            errors.append(SyntaxError_(
                file=file_path, line=0, col=0,
                message=f"Compile error: {e}",
            ))
        except Exception:
            # Non-syntax errors (e.g. memory) are not validation failures
            pass

        return SyntaxValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            language=LanguageId.PYTHON,
        )

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``pyright --outputjson`` on the **on-disk** file.

        Unlike :meth:`validate_syntax`, this resolves imports/types against the
        surrounding project (cwd = detected project root), so it catches
        undefined names, missing imports, and type errors that pure AST parsing
        misses.

        Design choices:
        - Operates on the real file (no temp copy) so import resolution works.
        - Skips entirely if pyright is not installed (``ok=True``).
        - Skips if the file cannot be read (deferred to syntax check).
        - Only diagnostics whose ``file`` matches ``file_path`` are reported,
          to avoid noise from other files in a multi-file pyright run.
        - Errors make ``ok=False``; warnings/info are reported but kept as-is.
        """
        import json
        import subprocess

        # Relative/non-existent path → defer to syntax check
        if not file_path or not os.path.exists(file_path):
            return SyntaxValidationResult(ok=True, language=LanguageId.PYTHON)

        project_root = detect_project_root(file_path)
        cmd = ["pyright", "--outputjson", file_path]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=120,  # large projects can take a while on cold start
                cwd=project_root,
            )
        except FileNotFoundError:
            logger.debug("pyright not found; skipping semantic validation")
            return SyntaxValidationResult(ok=True, language=LanguageId.PYTHON)
        except subprocess.TimeoutExpired:
            logger.debug("pyright timed out for %s; skipping", file_path)
            return SyntaxValidationResult(ok=True, language=LanguageId.PYTHON)
        except Exception as e:
            logger.debug("pyright semantic check failed: %s", e)
            return SyntaxValidationResult(ok=True, language=LanguageId.PYTHON)

        # Parse JSON output
        try:
            payload = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            # pyright crashed / non-JSON output → skip
            return SyntaxValidationResult(ok=True, language=LanguageId.PYTHON)

        diags = payload.get("generalDiagnostics", []) or []
        # Normalize target path for matching (pyright reports absolute paths).
        target_norm = os.path.normpath(os.path.abspath(file_path))
        errors: list[SyntaxError_] = []
        has_error = False
        for d in diags:
            try:
                sev = (d.get("severity") or "error").lower()
                rng = d.get("range") or {}
                start = rng.get("start") or {}
                d_file = d.get("file") or ""
                # Only report diagnostics for the file we asked about
                if d_file and os.path.normpath(d_file) != target_norm:
                    continue
                errors.append(SyntaxError_(
                    file=file_path,
                    line=(start.get("line") or 0) + 1,  # pyright is 0-indexed
                    col=(start.get("character") or 0) + 1,
                    message=d.get("message", "").strip(),
                    severity=sev,
                    code=d.get("rule") or "",
                ))
                if sev == "error":
                    has_error = True
            except Exception:
                continue
        return SyntaxValidationResult(
            ok=not has_error,
            errors=errors,
            language=LanguageId.PYTHON,
        )

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:async\s+)?def\s+{name}\s*\(",
                description="Python function/method definition",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"class\s+{name}\s*[:\(]",
                description="Python class definition",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.py"]

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> Optional[list[str]]:
        return ["ruff", "check", "--output-format=json", file_path]

    def get_test_command(
        self, repo_root: str, test_args: Optional[list[str]] = None
    ) -> Optional[list[str]]:
        return ["python", "-m", "pytest", "-q"] + (test_args or [])

    # ── Symbol finder (tree-sitter → LibCST → stdlib ast) ─────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> Optional[tuple[int, int]]:
        """Return ``(start_line, end_line)`` for *symbol_name*.

        Priority: tree-sitter → LibCST → stdlib ast.
        Supports qualified names (``ClassName.method``).
        Lines are 1-indexed.
        """

        # Priority 1: tree-sitter (precise range, multi-language)
        if _tree_sitter_available():
            try:
                from .tree_sitter_utils import find_symbol_range

                result = find_symbol_range(content, symbol_name, "python")
                if result is not None:
                    return result
            except Exception:
                pass

        # Priority 2: LibCST (precise end_lineno, decorator-aware)
        try:
            from .libcst_utils import find_symbol_range as _lc_range

            result = _lc_range(content, symbol_name)
            if result is not None:
                return result
        except Exception:
            pass

        # Priority 3: stdlib ast (fallback)
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            return None

        parts = symbol_name.split(".")
        if len(parts) >= 2:
            class_name = parts[-2]
            method_name = parts[-1]
            for cls_node in ast.walk(tree):
                if isinstance(cls_node, ast.ClassDef) and cls_node.name == class_name:
                    for child in cls_node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == method_name:
                                end = getattr(child, "end_lineno", None)
                                if end is None:
                                    return None
                                return (child.lineno, end)
                    break
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol_name:
                    end = getattr(node, "end_lineno", None)
                    if end is None:
                        return None
                    return (node.lineno, end)
        return None

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Structural query methods (ast.parse-based) ─────────────────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Return ``[(name, kind, start_line, end_line), ...]`` for Python.

        Uses ast.parse for precise range info.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        results: list[tuple[str, str, int, int]] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", None)
                if end is not None:
                    results.append((node.name, "function", node.lineno, end))
            elif isinstance(node, ast.ClassDef):
                end = getattr(node, "end_lineno", None)
                if end is not None:
                    results.append((node.name, "class", node.lineno, end))
        return results

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Return ``[(method_name, start_line, end_line), ...]`` for a Python class.

        Uses ast.parse to scan class body for FunctionDef/AsyncFunctionDef nodes.
        """
        return self.find_all_class_methods(content).get(class_name, [])

    def find_all_class_methods(
        self, content: str,
    ) -> dict[str, list[tuple[str, int, int]]]:
        """Return ``{class_name: [(method_name, start_line, end_line), ...]}``.

        Parses the source exactly once and extracts methods for every class.
        This avoids the O(C) re-parses that the per-class ``find_class_methods``
        would otherwise trigger (ast.parse is the dominant cost on large files).
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {}

        result: dict[str, list[tuple[str, int, int]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods: list[tuple[str, int, int]] = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        end = getattr(item, "end_lineno", None)
                        if end is not None:
                            methods.append((item.name, item.lineno, end))
                if methods:
                    result[node.name] = methods
        return result

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Return ``(body_start_line, body_end_line)`` for a Python function/method.

        Body = the indented block after the ``def`` signature (``:`` line).
        Uses ast.parse to find the function node and compute body range.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        parts = symbol_name.split(".")
        if len(parts) >= 2:
            # Qualified name: ClassName.method
            class_name = parts[-2]
            method_name = parts[-1]
            for cls_node in ast.walk(tree):
                if isinstance(cls_node, ast.ClassDef) and cls_node.name == class_name:
                    for child in cls_node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == method_name:
                                body_start = child.lineno + 1
                                end = getattr(child, "end_lineno", None)
                                if end is None:
                                    return None
                                return (body_start, end)
                    break
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == symbol_name:
                    body_start = node.lineno + 1  # First line after def
                    end = getattr(node, "end_lineno", None)
                    if end is None:
                        return None
                    return (body_start, end)
        return None


    def get_definition_keywords(self) -> list[str]:
        return ["def ", "async def ", "class "]
