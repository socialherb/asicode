"""
Python syntax provider — wraps existing AST-based logic.
"""
from __future__ import annotations

import ast
import contextlib
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
    """Dynamically check if tree-sitter is available for Python.

    Delegates to :func:`~external_llm.languages.tree_sitter_utils.is_language_available`,
    which is memoised in the module's ``_LANG_CACHE`` (positive AND negative) and
    cleared by ``invalidate_caches()`` — the same invalidation contract the rest of
    the module uses. So a grammar pip-installed mid-process is picked up without a
    restart, and repeated checks are a cache hit instead of building a per-thread
    parser (the old ``get_parser`` probe had that side effect).
    """
    from .tree_sitter_utils import is_available, is_language_available

    return is_available() and is_language_available("python")



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

        # 2. compile() — stricter, catches some issues AST doesn't. Non-syntax
        # errors (e.g. memory) are not validation failures.
        with contextlib.suppress(ValueError, RecursionError, MemoryError):  # non-syntax compile failures
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

        Checking several files at once goes through
        :meth:`validate_semantics_batch`, which shares one pyright process
        across them.
        """
        return self.validate_semantics_batch([file_path])[file_path]

    def validate_semantics_batch(
        self, file_paths: list[str],
    ) -> dict[str, SyntaxValidationResult]:
        """Semantic-check *file_paths* with one pyright process per project root.

        pyright takes any number of files on the command line and returns every
        diagnostic in a single ``generalDiagnostics`` array tagged with its
        file, so N files cost roughly one cold start instead of N (measured
        over 4 files: 2.167 s sequential vs 0.391 s batched).

        Grouped by :func:`detect_project_root` because that is pyright's cwd,
        and it decides which ``pyproject.toml`` / ``pyrightconfig.json`` and
        which virtualenv apply — a monorepo can resolve several roots, and
        merging those into one invocation would type-check files against the
        wrong config.
        """
        out: dict[str, SyntaxValidationResult] = {}
        groups: dict[str, list[str]] = {}
        for p in file_paths:
            # Relative/non-existent path → defer to syntax check. A fresh
            # result per file, never one shared instance: the dataclass is
            # mutable and carries a list, so sharing would let one consumer's
            # append surface on every other skipped file.
            if not p or not os.path.exists(p):
                out[p] = SyntaxValidationResult.unchecked(
                    LanguageId.PYTHON, "the file is not on disk",
                )
                continue
            groups.setdefault(detect_project_root(p), []).append(p)
        for project_root, paths in groups.items():
            out.update(self._run_pyright(project_root, paths))
        return out

    def _run_pyright(
        self, project_root: str, file_paths: list[str],
    ) -> dict[str, SyntaxValidationResult]:
        """One pyright invocation over *file_paths*, split back out per file.

        Every input path is present in the result; a run that fails for any
        reason degrades to ``checked=False`` for all of them — advisory, never
        blocking, and never reported as a clean verdict it did not reach.
        """
        import json
        import subprocess

        def _skip(reason: str) -> dict[str, SyntaxValidationResult]:
            return {
                p: SyntaxValidationResult.unchecked(LanguageId.PYTHON, reason)
                for p in file_paths
            }

        cmd = ["pyright", "--outputjson", *file_paths]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                # Advisory only — non-blocking, surfaced for LLM self-healing.
                # Scales with the batch so a large turn is not cut off at the
                # single-file budget; the base 30 s dominates for small batches
                # because startup, not per-file analysis, is the bulk of a run.
                timeout=30 + 5 * len(file_paths),
                cwd=project_root,
                check=False,
            )
        except FileNotFoundError:
            logger.debug("pyright not found; skipping semantic validation")
            return _skip("pyright is not installed")
        except subprocess.TimeoutExpired:
            logger.debug("pyright timed out for %s; skipping", file_paths)
            return _skip("pyright timed out")
        except Exception as e:
            logger.debug("pyright semantic check failed: %s", e)
            return _skip("pyright could not be run")

        # Parse JSON output
        try:
            payload = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            # pyright crashed / non-JSON output → skip
            return _skip("pyright produced no readable output")

        diags = payload.get("generalDiagnostics", []) or []
        # pyright reports absolute paths; index the batch by the same
        # normalisation so each diagnostic lands on the file that asked for it.
        by_norm = {os.path.normpath(os.path.abspath(p)): p for p in file_paths}
        collected: dict[str, list[SyntaxError_]] = {p: [] for p in file_paths}
        failed: set[str] = set()
        for d in diags:
            with contextlib.suppress(AttributeError, TypeError):  # malformed diagnostic shape
                sev = (d.get("severity") or "error").lower()
                rng = d.get("range") or {}
                start = rng.get("start") or {}
                d_file = d.get("file") or ""
                # A batched run also reports files we did not ask about (and,
                # for a single-file run, other files pyright pulled in) — drop
                # anything outside the batch rather than misattributing it.
                if d_file:
                    owner = by_norm.get(os.path.normpath(d_file))
                elif len(file_paths) == 1:
                    # File-less diagnostic (config/environment). The old
                    # single-file path attributed these to the target, so keep
                    # that; with several files there is no honest owner, and
                    # copying it onto all of them would invent errors.
                    owner = file_paths[0]
                else:
                    owner = None
                if owner is None:
                    continue
                collected[owner].append(SyntaxError_(
                    file=owner,
                    line=(start.get("line") or 0) + 1,  # pyright is 0-indexed
                    col=(start.get("character") or 0) + 1,
                    message=d.get("message", "").strip(),
                    severity=sev,
                    code=d.get("rule") or "",
                ))
                if sev == "error":
                    failed.add(owner)
        return {
            p: SyntaxValidationResult(
                ok=p not in failed,
                errors=collected[p],
                language=LanguageId.PYTHON,
            )
            for p in file_paths
        }

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
        # .pyi type stubs declare the same symbols as .py implementations and
        # are first-class in _EXT_MAP / the Python callability family
        # (see test_python_family_includes_type_stubs) — index them too.
        return ["*.py", "*.pyi"]

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
            except Exception as _exc:  # parser fallback chain
                logger.debug("python_provider: tree-sitter find failed (%s) — falling to LibCST", _exc)
            else:
                if result is not None:
                    return result

        # Priority 2: LibCST (precise end_lineno, decorator-aware)
        try:
            from .libcst_utils import find_symbol_range as _lc_range

            result = _lc_range(content, symbol_name)
        except Exception as _exc:  # parser fallback chain
            logger.debug("python_provider: LibCST find failed (%s) — falling to ast", _exc)
        else:
            if result is not None:
                return result

        # Priority 3: stdlib ast (fallback)
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            logger.debug("ast.parse failed in find_symbol_in_file: %s", e)
            return None

        parts = symbol_name.split(".")
        if len(parts) >= 2:
            class_name = parts[-2]
            method_name = parts[-1]
            for cls_node in ast.walk(tree):
                if isinstance(cls_node, ast.ClassDef) and cls_node.name == class_name:
                    for child in cls_node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                            end = getattr(child, "end_lineno", None)
                            if end is None:
                                return None
                            return (child.lineno, end)
                    break
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_name:
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
        except SyntaxError as e:
            logger.debug("ast.parse failed in find_symbol_body_range: %s", e)
            return None

        parts = symbol_name.split(".")
        if len(parts) >= 2:
            # Qualified name: ClassName.method
            class_name = parts[-2]
            method_name = parts[-1]
            for cls_node in ast.walk(tree):
                if isinstance(cls_node, ast.ClassDef) and cls_node.name == class_name:
                    for child in cls_node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                            body_start = child.lineno + 1
                            end = getattr(child, "end_lineno", None)
                            if end is None:
                                return None
                            return (body_start, end)
                    break
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
                body_start = node.lineno + 1  # First line after def
                end = getattr(node, "end_lineno", None)
                if end is None:
                    return None
                return (body_start, end)
        return None


    def get_definition_keywords(self) -> list[str]:
        return ["def ", "async def ", "class "]
