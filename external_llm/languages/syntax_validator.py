"""
SyntaxValidator — language-agnostic syntax and symbol analysis facade.

Dispatches to ``LanguageRegistry`` providers (``SyntaxProvider`` subclasses)
for each supported language.  Python uses ``PythonSyntaxProvider`` (stdlib
``ast``), TS/JS/Go/Java/Kotlin use their respective providers (tree-sitter
or language-specific tools).

Previously used ``if lang == LanguageId.PYTHON`` branching — replaced with
provider dispatch so that adding a new language only requires registering
a ``SyntaxProvider`` subclass.

progressive TS/JS/Go/Java/Kotlin support.

Usage::

    from ..languages.syntax_validator import SyntaxValidator

    result = SyntaxValidator.validate_syntax(content, lang)
    rng = SyntaxValidator.find_symbol_range(content, "MyClass", lang)
    syms = SyntaxValidator.find_symbols(content, lang)
    body = SyntaxValidator.extract_symbol_body(code, name, lang)
"""
from __future__ import annotations

import ast
from typing import Any, Optional

from . import tree_sitter_utils as ts_utils
from .models import LanguageId, SyntaxError_, SyntaxValidationResult


def _get_provider(lang: LanguageId):
    """Return the ``SyntaxProvider`` for *lang*, or ``None`` if unsupported."""
    from .registry import LanguageRegistry

    return LanguageRegistry.instance().get_by_lang(lang)


class SyntaxValidator:
    """Unified syntax/symbol analysis — dispatches to per-language providers."""

    # ── Syntax validation ──────────────────────────────────────────────

    @staticmethod
    def validate_syntax(content: str, lang: LanguageId) -> SyntaxValidationResult:
        """Check *content* for syntax errors via the language provider."""
        provider = _get_provider(lang)
        if provider is not None:
            return provider.validate_syntax("", content)
        # Fallback: tree-sitter has_error
        has_err = ts_utils.has_error(content, lang.value)
        if has_err is None:
            return SyntaxValidationResult(ok=True, language=lang)
        if has_err:
            return SyntaxValidationResult(
                ok=False,
                errors=[SyntaxError_(file="", line=0, col=0, message="syntax error (tree-sitter)")],
                language=lang,
            )
        return SyntaxValidationResult(ok=True, language=lang)

    @staticmethod
    def _validate_python_ast(content: str) -> SyntaxValidationResult:
        """Validate Python via ast.parse."""
        try:
            ast.parse(content)
            return SyntaxValidationResult(ok=True, language=LanguageId.PYTHON)
        except SyntaxError as e:
            err = SyntaxError_(
                file="",
                line=getattr(e, "lineno", 0) or 0,
                col=getattr(e, "offset", 0) or 0,
                message=e.msg or str(e),
            )
            return SyntaxValidationResult(
                ok=False, errors=[err], language=LanguageId.PYTHON,
            )

    # ── Symbol range detection ─────────────────────────────────────────

    @staticmethod
    def find_symbol_range(
        content: str, symbol_name: str, lang: LanguageId,
    ) -> Optional[tuple[int, int]]:
        """Return ``(start_line, end_line)`` (1-indexed) for *symbol_name*.

        Python → ``ast.parse`` + walk.
        Others → ``tree_sitter_utils.find_symbol_range``.

        Returns None if not found or parser unavailable.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.find_symbol_range(content, symbol_name)
        return ts_utils.find_symbol_range(content, symbol_name, lang.value)

    @staticmethod
    def _python_find_symbol_range(
        content: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Python AST walk to find symbol line range."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        bare = symbol_name.split(".")[-1]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == bare:
                    end = getattr(node, "end_lineno", None)
                    if end is None:
                        end = node.lineno
                    return (node.lineno, end)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == bare:
                        end = max(
                            (getattr(t, "end_lineno", None) or t.lineno)
                            for t in node.targets
                        ) if node.targets else node.lineno
                        return (node.lineno, end)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == bare:
                    end = getattr(node, "end_lineno", None) or node.lineno
                    return (node.lineno, end)
        return None

    # ── All symbols enumeration ────────────────────────────────────────

    @staticmethod
    def find_symbols(
        content: str, lang: LanguageId,
    ) -> list[tuple[str, str, int, int]]:
        """Enumerate all top-level symbols.

        Returns ``[(name, kind, start_line, end_line), ...]`` or empty list.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.find_symbols(content)
        return ts_utils.find_all_symbols(content, lang.value)

    @staticmethod
    def _python_find_symbols(
        content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Python AST walk to enumerate top-level symbols."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        results: list[tuple[str, str, int, int]] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", None) or node.lineno
                results.append((node.name, "function", node.lineno, end))
            elif isinstance(node, ast.ClassDef):
                end = getattr(node, "end_lineno", None) or node.lineno
                results.append((node.name, "class", node.lineno, end))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        results.append((target.id, "variable", node.lineno, node.lineno))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                results.append((node.target.id, "variable", node.lineno, node.lineno))
        return results

    # ── Symbol body extraction ─────────────────────────────────────────

    @staticmethod
    def extract_symbol_body(
        code: str, symbol_name: str, lang: LanguageId,
    ) -> Optional[tuple[int, int]]:
        """Return ``(body_start_line, body_end_line)`` for a function/method.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.extract_symbol_body(code, symbol_name)
        return ts_utils.extract_symbol_body(code, symbol_name, lang.value)

    @staticmethod
    def _python_extract_symbol_body(
        code: str, symbol_name: str,
    ) -> Optional[tuple[int, int]]:
        """Extract function/method body range using Python AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        bare = symbol_name.split(".")[-1]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == bare and node.body:
                    first_stmt = node.body[0]
                    last_stmt = node.body[-1]
                    end = getattr(last_stmt, "end_lineno", None) or last_stmt.lineno
                    return (first_stmt.lineno, end)
        return None

    # ── Dead code detection ────────────────────────────────────────────

    @staticmethod
    def is_dead_code_introduced(orig: str, new: str, lang: LanguageId) -> bool:
        """Check if *new* introduces dead code compared to *orig*.
        """
        provider = _get_provider(lang)
        if provider is not None:
            return provider.is_dead_code_introduced(orig, new)
        # Fallback: conservative — validate syntax of new code
        return not SyntaxValidator.validate_syntax(new, lang).ok

    @staticmethod
    def _is_dead_code_python(orig: str, new: str) -> bool:
        """Python AST dead code detection — compare reachable statement count."""
        try:
            orig_tree = ast.parse(orig)
            new_tree = ast.parse(new)
        except SyntaxError:
            return False

        def _count_reachable(node: ast.AST) -> int:
            """Count reachable statements in a function body."""
            count = 0
            for stmt in ast.walk(node):
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue  # Don't descend into nested scopes
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    count += 1
                    break  # Statements after return/raise are dead
                if isinstance(stmt, ast.stmt):
                    count += 1
            return count

        # Compare reachable statements in first function of each tree
        def _first_func(tree: ast.AST) -> Optional[ast.AST]:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node
            return None

        orig_func = _first_func(orig_tree)
        new_func = _first_func(new_tree)
        if orig_func is None or new_func is None:
            return False

        return _count_reachable(new_func) < _count_reachable(orig_func)

    # ── File-based symbol lookup ──────────────────────────────────────

    @staticmethod
    def find_symbol_in_file(
        file_path: str, symbol_name: str, content: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Read *file_path* and locate *symbol_name*, returning definition info."""

        lang = LanguageId.from_path(file_path)
        if lang == LanguageId.UNKNOWN:
            return None

        if content is None:
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                return None

        provider = _get_provider(lang)
        if provider is not None:
            result = provider.find_symbol_in_file(file_path, symbol_name, content)
            if result is not None:
                start_line, end_line = result
                # Determine kind from all-symbols enumeration
                kind: str = "symbol"
                for name, k, sl, _ in provider.find_symbols(content):
                    if name == symbol_name and sl == start_line:
                        kind = k
                        break
                return {
                    "file": file_path,
                    "line": start_line,
                    "end_line": end_line,
                    "kind": kind,
                    "name": symbol_name,
                }
            return None
        # Fallback: tree-sitter direct lookup
        return SyntaxValidator._ts_find_symbol_in_file(file_path, symbol_name, content, lang)

    @staticmethod
    def _python_find_symbol_in_file(
        file_path: str, symbol_name: str, source: str,
    ) -> Optional[dict[str, Any]]:
        """Python AST-based symbol lookup (replaces _ast_find_symbol_in_file)."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        bare = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == bare:
                    return {
                        "file": file_path,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", None),
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "name": bare,
                    }
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == bare:
                        return {
                            "file": file_path,
                            "line": node.lineno,
                            "end_line": max(
                                (getattr(t, "end_lineno", None) or t.lineno)
                                for t in node.targets
                            ) if node.targets else node.lineno,
                            "kind": "variable",
                            "name": bare,
                        }
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == bare:
                    return {
                        "file": file_path,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", None) or node.lineno,
                        "kind": "variable",
                        "name": bare,
                    }
        return None

    @staticmethod
    def _ts_find_symbol_in_file(
        file_path: str, symbol_name: str, source: str, lang: LanguageId,
    ) -> Optional[dict[str, Any]]:
        """Tree-sitter based symbol lookup for non-Python files."""
        lang_str = lang.value
        rng = ts_utils.find_symbol_range(source, symbol_name, lang_str)
        if rng is None:
            return None
        start_line, end_line = rng
        # Determine kind from all-symbols enumeration (more reliable)
        kind: str = "symbol"
        for name, k, sl, _ in ts_utils.find_all_symbols(source, lang_str):
            if name == symbol_name and sl == start_line:
                kind = k
                break
        return {
            "file": file_path,
            "line": start_line,
            "end_line": end_line,
            "kind": kind,
            "name": symbol_name,
        }
