"""Language backend abstraction — two-tier architecture.

PythonBackend:  uses existing Python AST logic (ast_engine.py)
TreeSitterBackend: uses tree-sitter for all non-Python languages
                    (TS/JS/Go/Java/Kotlin/...)

LanguageBackend.for_path(file_path) -> appropriate backend.
Adding a new language requires only:
  1. Adding its query to _SYMBOL_QUERIES in tree_sitter_utils.py
  2. (Optionally) a LanguageId enum entry
No code change in this module or in executor_schedule_handlers.py.
"""

from __future__ import annotations

import ast
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

from ..languages import LanguageId
from ..languages.tree_sitter_utils import (
    _get_language as _ts_get_language,
)
from ..languages.tree_sitter_utils import (
    find_anchor_node as _ts_find_anchor_node,
)
from ..languages.tree_sitter_utils import (
    find_symbol_range as _ts_find_symbol_range,
)
from ..languages.tree_sitter_utils import (
    has_error as _ts_has_error,
)
from .ast_engine import (
    delete_unused_import_from_python_source as _py_delete_unused_import,
)
from .ast_engine import (
    find_symbol_line_range_for_delete as _py_find_symbol_line_range_for_delete,
)
from .ast_engine import (
    py_expand_to_branch_body_end as _py_expand_to_branch_body_end,
)

logger = logging.getLogger(__name__)


class LanguageBackend(ABC):
    """Abstract base for language-specific operation support.

    Subclasses implement find_symbol_range() and optionally other
    language-specific operations.  Callers use LanguageBackend.for_path()
    to get the appropriate backend — no if/elif branching needed.
    """

    @abstractmethod
    def find_symbol_range(self, src: str, symbol: str) -> Optional[tuple[int, int]]:
        """Resolve (start_line, end_line) of *symbol* in *src* (1-indexed).

        Returns None if the symbol cannot be found or the source is unparseable.
        """
        ...

    def validate_syntax(self, src: str) -> bool:
        """Check if *src* is syntactically valid.

        Returns True if valid (or validator unavailable).
        Override in subclasses with parser access.
        """
        return True

    def resolve_anchor(self, src: str, line: int) -> Optional[dict]:
        """Resolve *line* (1-indexed) to an AST node.

        Returns node metadata dict (start_line, end_line, text, node_type),
        or None if resolution is not available or the line cannot be resolved.

        Base default: returns None (not available).  Subclasses with parser
        access should override to enable structural anchor resolution.
        """
        _ = src, line
        return None

    def supports_resolution(self) -> bool:
        """Return True if this backend can resolve anchors to AST nodes.

        Subclasses with parser access may override to report False for
        languages whose grammar is not installed (e.g. HTML/CSS mapped to
        TreeSitterBackend by for_path() but lacking a tree-sitter parser).

        Callers should check this before hard-rejecting unresolved anchors:
        if False, fall back to regex-based line resolution instead.
        """
        return True

    def delete_import_name(
        self, src: str, symbol: str, start_line: int, end_line: int, line_text: str = "",
    ) -> Optional[str]:
        """Delete *symbol* from an import at *start_line*.

        Unlike line-range delete (which removes the entire line), this
        removes only the named specifier — keeping other imports on the
        same line intact.

        Returns modified *src*, or None if:
        - The language does not support import-aware deletion, or
        - The target line is not an import, or
        - The symbol is the last specifier (falls through to line-range).
        Default: return None (no import-aware deletion available).
        """
        return None

    def expand_to_branch_body_end(self, src: str, start_line: int) -> Optional[int]:
        """Expand *start_line* to the end of the branch body at that line.

        Used when ``end_line == start_line`` to find the full range of
        an ``if/while/for`` body that must be removed to avoid orphaned
        indented code.

        Returns expanded end_line (1-indexed), or None if not applicable.
        Default: return None.
        """
        return None

    def find_nearest_symbol(
        self, src: str, target_line: int,
    ) -> Optional[tuple[str, int, int]]:
        """Find (name, start_line, end_line) of symbol nearest to *target_line*
        (1-indexed).  Walks all top-level function/class definitions and
        returns the one closest to *target_line* by absolute line distance.

        A symbol that *contains* target_line (start ≤ target ≤ end) is
        always preferred (distance 0).  Returns None if no symbol found or
        parser unavailable.

        Default: return None (not available).
        """
        return None

    def find_duplicate_definitions(self, src: str) -> list[str]:
        """Return list of top-level symbol names defined more than once in
        *src*.  Returns empty list if no duplicates or parser unavailable.

        Used to detect hallucinated duplicate definitions after an insert
        or anchor edit.  Default: return [] (not available).
        """
        return []

    def decorator_expand_start(self, lines: list[str], start_line: int) -> int:
        """Expand *start_line* backward to include contiguous decorator lines.

        Walks upward from ``start_line - 1`` collecting ``@...`` lines.
        Returns the expanded start_line (≤ original).
        Default: no expansion (returns start_line unchanged).
        """
        return start_line

    @staticmethod
    def for_path(file_path: str) -> "LanguageBackend":
        """Factory: return the appropriate backend for *file_path*.

        Python -> PythonBackend (AST-based)
        Everything else (TS/JS/Go/Java/Kotlin/...) -> TreeSitterBackend
        Unknown -> TreeSitterBackend (graceful degradation)
        """
        _lid = LanguageId.from_path(file_path)
        if _lid is LanguageId.PYTHON:
            return PythonBackend()
        if _lid is LanguageId.UNKNOWN:
            logger.info(
                "Unknown language for %s — falling back to TreeSitterBackend",
                file_path,
            )
        return TreeSitterBackend(_lid)


class TreeSitterBackend(LanguageBackend):
    """Tree-sitter backend: TS/JS/Go/Java/Kotlin + any future language.

    Uses tree-sitter query-based symbol resolution.
    Adding a new language requires only a _SYMBOL_QUERIES entry —
    no code change in this class.
    """

    def __init__(self, lid: LanguageId) -> None:
        self._lang_str = lid.value  # e.g. "typescript", "go", "java"

    def find_symbol_range(self, src: str, symbol: str) -> Optional[tuple[int, int]]:
        """Resolve symbol range via tree-sitter.

        Returns None if tree-sitter is unavailable, the symbol is not found,
        or the source cannot be parsed."""
        try:
            return _ts_find_symbol_range(src, symbol, self._lang_str)
        except Exception:
            logger.exception(
                "Tree-sitter symbol range lookup failed for lang=%s symbol=%s",
                self._lang_str, symbol,
            )
            return None

    def validate_syntax(self, src: str) -> bool:
        """Validate via tree-sitter has_error()."""
        result = _ts_has_error(src, self._lang_str)
        if result is None:
            return True  # tree-sitter unavailable → be permissive
        return not result

    def resolve_anchor(self, src: str, line: int) -> Optional[dict]:
        """Resolve *line* (1-indexed) to an AST node via tree-sitter.

        Returns node metadata dict (start_line, end_line, start_byte,
        end_byte, text, node_type), or None if tree-sitter is unavailable
        or the line cannot be resolved to a named AST node.

        This is the core anchor-resolution primitive for structural editing.
        Instead of regex-matching text, we find the real AST node at *line*.
        """
        return _ts_find_anchor_node(src, self._lang_str, line)

    def supports_resolution(self) -> bool:
        """Return True if tree-sitter has a parser installed for this language.

        Languages like HTML, CSS, JSON are mapped to TreeSitterBackend by
        for_path() but tree-sitter may not have a grammar for them.  Returns
        False for unsupported languages so callers fall back to regex-based
        resolution instead of hard-rejecting valid anchors.
        """
        return _ts_get_language(self._lang_str) is not None

    def delete_import_name(
        self, src: str, symbol: str, start_line: int, end_line: int, line_text: str = "",
    ) -> Optional[str]:
        """Delete *symbol* from a TS/JS import line.

        Handles ``import { X, Y, Z } from 'mod'`` — removes just *symbol*
        and its trailing comma, keeping other specifiers.

        For Go/Java/Kotlin (single import per line) returns None so the
        caller falls through to line-range delete.
        """
        if self._lang_str not in ("typescript", "javascript"):
            return None  # single-import-per-line languages → line-range is fine

        if not line_text:
            lines = src.splitlines(keepends=True)
            if 1 <= start_line <= len(lines):
                line_text = lines[start_line - 1]
            else:
                return None

        stripped = line_text.strip()
        # Only handle named-import syntax: import { ... } from ...
        # Default imports, namespace imports, side-effect imports → line-range
        if not stripped.startswith("import {") or "}" not in stripped:
            return None

        escaped = re.escape(symbol)
        # Comma before symbol (non-first): `, X`
        result = re.sub(
            r"[ \t]*,[ \t]*\b" + escaped + r"\b[ \t]*",
            "", line_text, count=1,
        )
        if result != line_text:
            # Clean up: `{ , Y}` → `{ Y}`, `{ Y, }` → `{ Y}`
            result = re.sub(r",\s*,", ",", result)
            result = re.sub(r",\s*}", "}", result)
            result = re.sub(r"\{\s*,", "{", result)
            lines = src.splitlines(keepends=True)
            lines[start_line - 1] = result
            return "".join(lines)

        # Symbol with trailing comma (first specifier): `X,`
        result = re.sub(
            r"\b" + escaped + r"\b[ \t]*,[ \t]*",
            "", line_text, count=1,
        )
        if result != line_text:
            result = re.sub(r",\s*,", ",", result)
            result = re.sub(r",\s*}", "}", result)
            result = re.sub(r"\{\s*,", "{", result)
            lines = src.splitlines(keepends=True)
            lines[start_line - 1] = result
            return "".join(lines)

        # Last specifier (no comma): `{ X }` — remove whole line
        return None  # line-range will handle it

    def expand_to_branch_body_end(self, src: str, start_line: int) -> Optional[int]:
        """Expand to branch body end via tree-sitter (language-aware).

        Supports TS/JS/Go/Java/Kotlin if/try/for/while statements.
        Returns None for languages without a block-body grammar.
        """
        return _ts_expand_to_branch_body_end(src, start_line, self._lang_str)

    def find_nearest_symbol(
        self, src: str, target_line: int,
    ) -> Optional[tuple[str, int, int]]:
        """Find nearest top-level symbol via tree-sitter tree walk.
        Only considers direct children of the ``program``/``module`` root
        — nested symbols (methods, inner classes) are excluded.
        """
        try:
            from ..languages.tree_sitter_utils import (
                _SYMBOL_NODE_TYPES as _TS_NODE_TYPES,
            )
            from ..languages.tree_sitter_utils import (
                _extract_name as _ts_extract_name,
            )
            from ..languages.tree_sitter_utils import (
                get_parser as _ts_get_parser,
            )
            parser = _ts_get_parser(self._lang_str)
            if parser is None:
                return None
            tree = parser.parse(src.encode("utf-8"))
            root = tree.root_node
            best_name: Optional[str] = None
            best_start = best_end = 0
            best_dist = float("inf")
            for child in root.children:
                if child.type not in _TS_NODE_TYPES:
                    continue
                name = _ts_extract_name(child)
                if not name:
                    continue
                _start = child.start_point[0] + 1  # 0→1-indexed
                _end = child.end_point[0] + 1
                if _start <= target_line <= _end:
                    _dist = 0
                elif target_line < _start:
                    _dist = _start - target_line
                else:
                    _dist = target_line - _end
                if (_dist < best_dist or
                        (_dist == best_dist and
                         (_end - _start) < (best_end - best_start))):
                    best_dist = _dist
                    best_name, best_start, best_end = name, _start, _end
            if best_name:
                return (best_name, best_start, best_end)
            return None
        except Exception:
            logger.exception(
                "find_nearest_symbol tree-sitter failed for lang=%s",
                self._lang_str,
            )
            return None

    def find_duplicate_definitions(self, src: str) -> list[str]:
        """Find duplicate top-level definitions via tree-sitter.
        Only counts direct children of the ``program``/``module`` root
        node — nested methods inside classes are excluded.
        """
        try:
            from ..languages.tree_sitter_utils import (
                _SYMBOL_NODE_TYPES as _TS_NODE_TYPES,
            )
            from ..languages.tree_sitter_utils import (
                _extract_name as _ts_extract_name,
            )
            from ..languages.tree_sitter_utils import (
                get_parser as _ts_get_parser,
            )
            parser = _ts_get_parser(self._lang_str)
            if parser is None:
                return []
            tree = parser.parse(src.encode("utf-8"))
            root = tree.root_node
            top_names: list[str] = []
            for child in root.children:
                if child.type in _TS_NODE_TYPES:
                    name = _ts_extract_name(child)
                    if name:
                        top_names.append(name)
            dupes = [name for name, count in Counter(top_names).items() if count > 1]
            return dupes
        except Exception:
            return []

    def decorator_expand_start(self, lines: list[str], start_line: int) -> int:
        """Expand backward for TS/JS decorators (starts with @)."""
        idx = start_line - 2
        if idx >= len(lines):
            idx = len(lines) - 1
        while idx >= 0:
            stripped = lines[idx].strip()
            if stripped.startswith("@"):
                idx -= 1
            else:
                break
        return idx + 2  # 1-indexed, first decorator or original


def _ts_expand_to_branch_body_end(src: str, start_line: int, lang: str) -> Optional[int]:
    """Tree-sitter: return last line of branch body at *start_line* for *lang*.

    Supports Python, TS/JS, Go, Java, Kotlin if/while/try/for statements.
    Returns None if not found or tree-sitter unavailable.
    """
    from ..languages.tree_sitter_utils import (
        get_parser,
        parse_to_tree,
    )

    parser = get_parser(lang)
    if parser is None:
        return None
    tree = parse_to_tree(src, lang)
    if tree is None:
        return None
    root = tree.root_node
    target = start_line - 1  # 1→0-indexed

    # Block-producing node types across languages
    _BLOCK_NODES = {
        "if_statement", "while_statement", "for_statement", "for_in_statement",
        "try_statement", "try_block", "catch_block", "finally_block",
        "else_clause", "elif_clause",
        # TS/JS
        "switch_statement", # Go
        "select_statement",
    }
    _BODY_FIELD_NAMES = {"body", "consequence", "alternative", "block", "statements"}

    def _find_node(n):
        if n is None or n.start_point[0] > target or n.end_point[0] < target:
            return None
        if n.start_point[0] == target and n.type in _BLOCK_NODES:
            return n
        for c in (n.children or []):
            r = _find_node(c)
            if r is not None:
                return r
        return None

    node = _find_node(root)
    if node is None:
        return None

    # Find the body child node and compute its end line
    def _body_end(n) -> Optional[int]:
        for c in (n.children or []):
            if c.type in _BODY_FIELD_NAMES:
                stmts = [
                    ch for ch in (c.children or [])
                    if ch.type not in ("comment", "newline", ";")
                ]
                if not stmts:
                    return c.start_point[0] + 1
                return max(s.end_point[0] for s in stmts) + 1
        return None

    # For if_statement: check which branch starts at target
    if node.type == "if_statement":
        # Primary if branch
        cond = node.child_by_field_name("condition")
        if cond and cond.start_point[0] == target:
            result = _body_end(node)
            if result is not None:
                return result
        # elif/else branches (language-specific node names)
        for c in (node.children or []):
            if c.type in ("elif_clause", "else_clause"):
                if c.start_point[0] == target:
                    result = _body_end(c)
                    if result is not None:
                        return result

    # For try/for/while: just find body
    result = _body_end(node)
    if result is not None:
        return result

    return None


class PythonBackend(LanguageBackend):
    """Python backend: uses Python AST (with tree-sitter fallback internally).

    Handles functions, classes, imports, constants, and dotted names (class.method).
    """

    def find_symbol_range(self, src: str, symbol: str) -> Optional[tuple[int, int]]:
        """Resolve symbol range via Python AST + tree-sitter fallback."""
        return _py_find_symbol_line_range_for_delete(src, symbol)

    def validate_syntax(self, src: str) -> bool:
        """Validate via ast.parse()."""
        try:
            ast.parse(src)
            return True
        except SyntaxError:
            return False

    def resolve_anchor(self, src: str, line: int) -> Optional[dict]:
        """Resolve *line* (1-indexed) to an AST node via Python ``ast``.

        Returns a dict with ``start_line``, ``end_line``, ``text``,
        ``node_type``, or ``None`` if the source cannot be parsed or the
        line does not fall inside any parseable statement.
        """
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None

        lines = src.splitlines()
        best: Optional[tuple[int, int, str, str]] = None
        # (start_line, end_line, text, node_type)

        def _walk(node: ast.AST) -> None:
            """Walk *node* recursively, tracking the innermost match."""
            nonlocal best
            _start = getattr(node, "lineno", None)
            _end = getattr(node, "end_lineno", None)
            if _start is not None and _end is not None:
                if _start <= line <= _end:
                    _text = ""
                    if 1 <= _start <= len(lines):
                        _text = lines[_start - 1].strip()
                    _type = type(node).__name__
                    # Keep the tightest (innermost) match: smallest range
                    if best is None or (_end - _start) < (best[1] - best[0]):
                        best = (_start, _end, _text, _type)
            # Recurse into children
            for _field in node._fields:
                _child = getattr(node, _field, None)
                if isinstance(_child, ast.AST):
                    _walk(_child)
                elif isinstance(_child, list):
                    for _item in _child:
                        if isinstance(_item, ast.AST):
                            _walk(_item)

        _walk(tree)
        if best is None:
            return None
        return {
            "start_line": best[0], "end_line": best[1],
            "text": best[2], "node_type": best[3],
        }

    def delete_import_name(
        self, src: str, symbol: str, start_line: int, end_line: int, line_text: str = "",
    ) -> Optional[str]:
        """Delete *symbol* from a Python import line via AST logic."""
        return _py_delete_unused_import(src, symbol, start_line)

    def expand_to_branch_body_end(self, src: str, start_line: int) -> Optional[int]:
        """Expand to branch body end via Python AST + tree-sitter fallback."""
        return _py_expand_to_branch_body_end(src, start_line)

    def find_nearest_symbol(
        self, src: str, target_line: int,
    ) -> Optional[tuple[str, int, int]]:
        """Find nearest top-level symbol via Python AST walk.
        Only considers direct children of the ``Module`` body —
        nested symbols (methods, inner classes) are excluded.
        """
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        best_name: Optional[str] = None
        best_start = best_end = 0
        best_dist = float("inf")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _start = node.lineno
                _end = getattr(node, "end_lineno", _start)
                if _start <= target_line <= _end:
                    _dist = 0
                elif target_line < _start:
                    _dist = _start - target_line
                else:
                    _dist = target_line - _end
                if (_dist < best_dist or
                        (_dist == best_dist and
                         (_end - _start) < (best_end - best_start))):
                    best_dist = _dist
                    best_name, best_start, best_end = node.name, _start, _end
        if best_name:
            return (best_name, best_start, best_end)
        return None

    def find_duplicate_definitions(self, src: str) -> list[str]:
        """Find duplicate top-level definitions via Python AST."""
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return []
        names = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        return [name for name, count in Counter(names).items() if count > 1]

    def decorator_expand_start(self, lines: list[str], start_line: int) -> int:
        """Expand backward for Python decorators."""
        idx = start_line - 2
        if idx >= len(lines):
            idx = len(lines) - 1
        while idx >= 0:
            stripped = lines[idx].strip()
            if stripped.startswith("@"):
                idx -= 1
            else:
                break
        return idx + 2  # 1-indexed, first decorator or original
