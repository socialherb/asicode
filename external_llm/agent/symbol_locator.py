"""Symbol location abstraction for the apply_patch → symbol-edit fallback.

The fallback needs to know *where* a named symbol lives in a source file so it
can rebuild that symbol's complete text after a patch fails to apply via git
(e.g. untracked / freshly-edited files where ``git apply --3way`` lacks the
pre-image blob).

This module defines a Python-`ast` symbol locator and a tree-sitter-backed
locator (``ProviderLocator``) for multi-language support. The locate()
interface is duck-typed — call sites use any locator interchangeably without
touching the in-memory hunk-apply engine or the routing logic that consume
spans.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SymbolSpan:
    """A located symbol.

    Attributes:
        name: Bare symbol name (e.g. ``foo`` or ``bar`` for a method).
        qualname: Qualified name usable by modify_symbol — ``foo`` for a
            top-level symbol, ``Foo.bar`` for a method of class ``Foo``.
        start_line: 1-based first line of the symbol (its def/class line, or the
            first decorator if decorated).
        end_line: 1-based last line of the symbol (inclusive).
        kind: ``"function"`` | ``"class"`` | ``"method"``.
        top_level: True when the symbol sits at module level (indent 0).
    """

    name: str
    qualname: str
    start_line: int
    end_line: int
    kind: str
    top_level: bool


class PythonAstLocator:
    """Symbol locator backed by the Python ``ast`` module.

    Reports top-level functions/classes and one level of class methods. Spans
    include leading decorators so a decorated symbol is rebuilt in full.
    """

    def locate(self, source: str) -> list[SymbolSpan]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        spans: list[SymbolSpan] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append(self._span(node, node.name, "function", True))
            elif isinstance(node, ast.ClassDef):
                spans.append(self._span(node, node.name, "class", True))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        spans.append(
                            self._span(
                                item, f"{node.name}.{item.name}", "method", False,
                                bare=item.name,
                            )
                        )
        return spans

    @staticmethod
    def _span(node: ast.AST, qualname: str, kind: str, top_level: bool,
              bare: Optional[str] = None) -> SymbolSpan:
        start = node.lineno  # type: ignore[attr-defined]
        for deco in getattr(node, "decorator_list", []):
            start = min(start, deco.lineno)
        end = getattr(node, "end_lineno", None) or node.lineno  # type: ignore[attr-defined]
        name = bare if bare is not None else getattr(node, "name", qualname)
        return SymbolSpan(
            name=name, qualname=qualname,
            start_line=start, end_line=end, kind=kind, top_level=top_level,
        )


class ProviderLocator:
    """Symbol locator backed by a language ``SyntaxProvider`` (tree-sitter).

    Wraps the existing multi-language symbol infrastructure
    (``find_top_level_definitions`` → ``[(name, kind, start, end)]``) so the
    fallback works for non-Python files. Reports top-level symbols only — that
    is sufficient for the whole-file rewrite path (editing a method changes its
    enclosing top-level symbol's text, which the rewrite captures wholesale).
    """

    def __init__(self, provider: object):
        self._provider = provider

    def locate(self, source: str) -> list[SymbolSpan]:
        try:
            defs = self._provider.find_top_level_definitions(source)  # type: ignore[attr-defined]
        except Exception:
            return []
        spans: list[SymbolSpan] = []
        for item in defs or []:
            try:
                name, kind, start, end = item
            except (ValueError, TypeError):
                continue
            spans.append(SymbolSpan(
                name=name, qualname=name,
                start_line=int(start), end_line=int(end),
                kind=str(kind), top_level=True,
            ))
        return spans
