"""
Repository-level symbol graph for asicode P1 architecture.

Implements a global symbol graph capturing:
- Symbol definitions (functions, classes)
- Call relationships (who calls whom)
- Import relationships (module dependencies)
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .models import ImportEdge, SymbolNode

_logger = logging.getLogger(__name__)
from ..languages import LanguageId, LanguageRegistry


# Backward compat re-export: RepositoryGraph uses a simplified CallEdge internally.
# The canonical CallEdge (from models.py) is available via graph_facade.
@dataclass
class CallEdge:
    """Edge representing a function/method call (repository_graph internal format)."""
    caller: str  # symbol name of caller
    callee: str  # symbol name of callee
    file_path: str  # file where call occurs
    line: int  # line number of call
    call_args: list[str] = field(default_factory=list)
    """Literal positional argument values at the call site.
    e.g. get_user(1) → ["1"], fetch("admin") → ['"admin"'].
    Enables object identity: distinguishes get_user(1) from get_user(2).
    Only constant (non-expression) args are captured.
    """
    is_mutating: bool = False
    """True when heuristics indicate this call has write/side-effect semantics.
    e.g. db.save(user), session.commit(), cache.set(k, v).
    Used to boost UPDATE_CALLERS propagation for data-structure change requests.
    """

# Re-export canonical models for consumers that import from this module
from .models import ImportEdge, SymbolNode  # noqa: F811


class RepositoryGraph:
    """Repository-wide symbol graph."""

    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.symbols: dict[str, SymbolNode] = {}
        self.call_edges: list[CallEdge] = []
        self.import_edges: list[ImportEdge] = []
        self.file_symbols: dict[str, list[str]] = defaultdict(list)
        self._symbol_locations: dict[tuple[str, str], str] = {}  # (name, file) -> unique_id
        # Build diagnostics — exposed for post-build inspection and log analysis.
        # Reset at the start of each build() call.
        self.build_exception_types: dict[str, int] = {}
        """Exception type name → count mapping from the last build, keyed by
        ``{ExceptionType}: {language or 'python'}`` so callers can distinguish
        Python parsing failures from tree-sitter or I/O errors."""

    # Directories to skip during repository scan
    _SKIP_DIRS = {
        "__pycache__", "node_modules", "venv", ".venv", "env",
        ".tox", "dist", "build", ".eggs", ".mypy_cache", ".pytest_cache",
        "worktrees",
    }

    def build(self) -> None:
        """Scan the repository and populate the graph."""
        self.build_exception_types = {}
        for root, dirs, files in os.walk(self.repo_root):
            # Skip hidden, venv, and vendor directories
            dirs[:] = [
                d for d in dirs
                if not d.startswith('.')
                and d not in self._SKIP_DIRS
                and not d.startswith('venv')
                and 'site-packages' not in d
            ]
            for file in files:
                file_path = os.path.join(root, file)
                lang = LanguageId.from_path(file_path)
                if lang == LanguageId.UNKNOWN:
                    continue
                try:
                    if lang == LanguageId.PYTHON:
                        self._process_file(file_path)
                    elif LanguageRegistry.instance().supports_structured_ops(file_path):
                        self._process_file_ripgrep(file_path)
                except Exception as _exc:
                    _rel = os.path.relpath(file_path, self.repo_root)
                    _etype = type(_exc).__name__
                    _tag = f"{_etype}: {'python' if lang == LanguageId.PYTHON else lang.value}"
                    self.build_exception_types[_tag] = self.build_exception_types.get(_tag, 0) + 1
                    _logger.debug(
                        "Graph build skipped %s: %s — %s",
                        _rel, _etype, _exc,
                    )
                    continue

    def _process_file(self, file_path: str) -> None:
        """Parse a single Python file and extract symbols, calls, imports."""
        with open(file_path, encoding='utf-8') as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                return

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            return

        relative_path = os.path.relpath(file_path, self.repo_root)
        visitor = GraphVisitor(relative_path, self.repo_root)
        visitor.visit(tree)

        # Add symbols discovered by visitor
        for symbol in visitor.symbols:
            unique_id = f"{relative_path}:{symbol.qualname}"
            self.symbols[unique_id] = symbol
            self.file_symbols[relative_path].append(unique_id)
            self._symbol_locations[(symbol.qualname, relative_path)] = unique_id

        # Add call edges
        for call in visitor.calls:
            self.call_edges.append(call)

        # Add import edges
        for imp in visitor.imports:
            self.import_edges.append(imp)

        # Refine end_line using tree-sitter when available (more precise than
        # stdlib ast for decorated definitions and deeply nested structures)
        try:
            from ..languages.tree_sitter_utils import (
                find_all_symbols as _ts_find,
            )
            from ..languages.tree_sitter_utils import (
                is_available as _ts_avail,
            )
            if _ts_avail():
                ts_symbols = _ts_find(content, "python")
                if ts_symbols:
                    # Build lookup: name -> (end_line) from tree-sitter
                    ts_end_map: dict[str, int] = {}
                    for sym_name, _kind, _start, end_line in ts_symbols:
                        existing = ts_end_map.get(sym_name)
                        if existing is None or end_line > existing:
                            ts_end_map[sym_name] = end_line
                    # Update end_line for matching symbols if tree-sitter
                    # gives a more precise (larger) range
                    for unique_id, symbol in self.symbols.items():
                        if symbol.file_path == relative_path:
                            ts_end = ts_end_map.get(symbol.name)
                            if ts_end is not None and ts_end > symbol.end_line:
                                symbol.end_line = ts_end
        except Exception:
            _logger.debug("_process_file tree-sitter refinement failed for %s", relative_path)

    def _process_file_ripgrep(self, file_path: str) -> None:
        """Extract symbols from a non-Python file using provider patterns + regex.

        When tree-sitter is available, also extracts call and import edges
        for multi-language call-graph and dependency tracking.
        """

        provider = LanguageRegistry.instance().get(file_path)
        if provider is None:
            return

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            return

        relative_path = os.path.relpath(file_path, self.repo_root)
        lang_value = provider.language_id().value

        # Try tree-sitter first for precise end_line + call/import edges.
        # Gracefully fall through to regex fallback on any failure (missing
        # grammar, unsupported language, parsing error) so the build is never
        # aborted by a single file.
        _ts_ok = False
        try:
            from ..languages.tree_sitter_utils import (
                extract_calls as _ts_extract_calls,
            )
            from ..languages.tree_sitter_utils import (
                extract_imports as _ts_extract_imports,
            )
            from ..languages.tree_sitter_utils import (
                find_all_symbols as _ts_find_all,
            )
            from ..languages.tree_sitter_utils import (
                is_available as _ts_available,
            )
            if _ts_available():
                ts_symbols = _ts_find_all(content, lang_value)
                if ts_symbols:
                    # Track symbol line ranges for caller attribution
                    _sym_ranges: list[tuple[str, int, int]] = []
                    for sym_name, kind, start_line, end_line in ts_symbols:
                        qualname = sym_name
                        unique_id = f"{relative_path}:{qualname}"
                        if unique_id in self.symbols:
                            continue
                        node = SymbolNode(
                            name=sym_name,
                            qualname=qualname,
                            module=relative_path,
                            file_path=relative_path,
                            kind=kind,
                            start_line=start_line,
                            end_line=end_line,
                            language=lang_value,
                        )
                        self.symbols[unique_id] = node
                        self.file_symbols[relative_path].append(unique_id)
                        self._symbol_locations[(qualname, relative_path)] = unique_id
                        # Only track "function"-kind symbols as potential callers
                        if kind == "function":
                            _sym_ranges.append((sym_name, start_line, end_line))

                    # Extract call edges
                    calls = _ts_extract_calls(content, lang_value)
                    for callee_name, call_line in calls:
                        caller_name = self._find_enclosing_symbol(
                            call_line, _sym_ranges
                        )
                        self.call_edges.append(CallEdge(
                            caller=caller_name,
                            callee=callee_name,
                            file_path=relative_path,
                            line=call_line,
                        ))

                    # Extract import edges
                    imports = _ts_extract_imports(content, lang_value)
                    for module_path, _import_line in imports:
                        self.import_edges.append(ImportEdge(
                            importer=relative_path,
                            imported=module_path,
                            import_type="import",
                        ))

                    _ts_ok = True
        except Exception:
            _logger.debug(
                "_process_file_ripgrep tree-sitter extraction failed for %s "
                "(lang=%s) — falling through to regex fallback",
                relative_path, lang_value,
            )

        if _ts_ok:
            return  # tree-sitter found symbols, skip regex

        # Fallback: regex-based extraction (end_line approximate)
        for sp in provider.get_symbol_patterns("any"):
            # Replace {name} with a capture group to find all definitions
            pat = sp.regex.replace(r"{name}", r"(\w+)")
            for m in re.finditer(pat, content, re.MULTILINE):
                sym_name = m.group(1)
                lineno = content[:m.start()].count("\n") + 1
                qualname = sym_name
                unique_id = f"{relative_path}:{qualname}"
                if unique_id in self.symbols:
                    continue
                node = SymbolNode(
                    name=sym_name,
                    qualname=qualname,
                    module=relative_path,  # use file path as module for non-Python
                    file_path=relative_path,
                    kind=sp.kind,
                    start_line=lineno,
                    end_line=lineno,  # approximate
                    language=lang_value,
                )
                self.symbols[unique_id] = node
                self.file_symbols[relative_path].append(unique_id)
                self._symbol_locations[(qualname, relative_path)] = unique_id

        # Regex-based import extraction fallback: when tree-sitter was unavailable
        # or failed, extract at least basic import edges so get_importers() has
        # some data for cross-file dead-code analysis and dependency tracking.
        _import_regexes: list[tuple[str, str]] = []
        if lang_value == "javascript" or lang_value == "typescript":
            _import_regexes = [
                # import ... from 'module'
                (r"""import\s+(?:\{[^}]*\}|\*\s+as\s+\w+|\w+(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))?)\s+from\s+['"]([^'"]+)['"]""", "js_import"),
                # require('module')
                (r"""(?:require|import)\s*\(\s*['"]([^'"]+)['"]\s*\)""", "js_require"),
                # import 'module' (side-effect import)
                (r"""import\s+['"]([^'"]+)['"]""", "js_side_effect"),
                # re-export ... from 'module'
                (r"""export\s+(?:\{[^}]*\}|\*\s+from)\s+from\s+['"]([^'"]+)['"]""", "js_re_export"),
            ]
        elif lang_value == "go":
            _import_regexes = [
                # import "module"
                (r"""import\s+['"]([^'"]+)['"]""", "go_import"),
                # import alias "module"
                (r"""import\s+\w+\s+['"]([^'"]+)['"]""", "go_alias_import"),
            ]
        elif lang_value in ("java", "kotlin"):
            _import_regexes = [
                # import package.Class;
                (r"""import\s+(?:static\s+)?([a-zA-Z_][\w.]*(?:\.[A-Z][\w]*)*)\s*;""", "java_import"),
            ]

        for _pat, _itype in _import_regexes:
            for _m in re.finditer(_pat, content, re.MULTILINE):
                _module_path = _m.group(1)
                # Normalize: strip leading/trailing quotes and whitespace
                _module_path = _module_path.strip("\"'")
                if not _module_path:
                    continue
                # Deduplicate by (importer, imported) pair
                _dup = False
                for _existing in self.import_edges:
                    if _existing.importer == relative_path and _existing.imported == _module_path:
                        _dup = True
                        break
                if not _dup:
                    self.import_edges.append(ImportEdge(
                        importer=relative_path,
                        imported=_module_path,
                        import_type=_itype,
                    ))

    @staticmethod
    def _find_enclosing_symbol(
        line: int, sym_ranges: list[tuple[str, int, int]],
    ) -> str:
        """Return the name of the symbol whose range contains *line*, or empty str."""
        best: Optional[tuple[str, int]] = None
        for sym_name, start, end in sym_ranges:
            if start <= line <= end:
                span = end - start
                if best is None or span < best[1]:
                    best = (sym_name, span)
        return best[0] if best else ""

    def get_symbol(
        self,
        name: str,
        file_path: Optional[str] = None,
        prefer_files: Optional[list[str]] = None,
    ) -> Optional[SymbolNode]:
        """Retrieve a symbol by name or qualname, optionally scoped to a file.

        A dotted ``name`` (e.g. ``MyClass.helper``) is matched against
        ``symbol.qualname``; a bare ``name`` is matched against ``symbol.name``.
        In BOTH cases, when multiple symbols match, the disambiguation cascade
        is identical:

          1. exact ``file_path`` match,
          2. suffix match (short path → full path, e.g. ``test_runner.py``
             → ``external_llm/agent/test_runner.py``),
          3. ``prefer_files`` scoring,
          4. first candidate.

        When ``file_path`` is provided and NO candidate resides in that file,
        ``None`` is returned (strict scoping) — callers that want lenient
        resolution retry without ``file_path`` (see spec_graph_enricher).

        This symmetry matters because two files commonly define the same
        qualname (e.g. ``MyClass.helper`` in a/v1.py and b/v2.py, or test
        stubs mirroring production classes). Previously the qualname branch
        ignored ``file_path``/``prefer_files`` and returned whichever symbol
        happened to be first in dict iteration order — a silent wrong-file
        result that bypassed callers' file-scoped-then-unscoped fallbacks.

        prefer_files: when provided and multiple symbols match by name,
            prefer one whose file_path is in this list (disambiguation).
        """
        # Matching predicate: qualname (dotted name) vs bare name.
        is_qualname = '.' in name
        candidates: list[SymbolNode] = [
            s for s in self.symbols.values()
            if (s.qualname == name if is_qualname else s.name == name)
        ]
        if not candidates:
            return None

        # file_path scope is honored uniformly for qualname AND bare names,
        # applied before any single-match short-circuit (matches the original
        # bare-name semantics so callers passing a file get strict scoping).
        if file_path:
            # Exact match first
            for symbol in candidates:
                if symbol.file_path == file_path:
                    return symbol
            # Suffix match: allow short names like "test_runner.py" to match
            # "external_llm/agent/test_runner.py"
            for symbol in candidates:
                if symbol.file_path and (
                    symbol.file_path.endswith("/" + file_path) or
                    symbol.file_path.endswith(os.sep + file_path)
                ):
                    return symbol
            # file_path requested but no candidate resides in that file.
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Multiple matches, no file_path — disambiguate with prefer_files
        if prefer_files:
            _pf_set = set(prefer_files)
            _pf_basenames = {os.path.basename(f) for f in prefer_files}
            _pf_dirs = {os.path.dirname(f) for f in prefer_files if f}
            _test_patterns = ('/test', '_test', '/tests/', 'test_', '/fixtures/')

            def _score(s: SymbolNode) -> float:
                sc = 0.0
                fp = s.file_path or ""
                if fp in _pf_set:
                    sc += 4.0
                elif os.path.basename(fp) in _pf_basenames:
                    sc += 3.0
                if os.path.dirname(fp) in _pf_dirs:
                    sc += 2.0
                if any(tp in fp.lower() for tp in _test_patterns):
                    sc -= 2.0
                return sc

            candidates.sort(key=_score, reverse=True)

        return candidates[0]

    def _edges_by_symbol_field(
        self, field: str, symbol_name: str
    ) -> list[CallEdge]:
        """Return call edges where *field* (``'callee'`` or ``'caller'``)
        matches *symbol_name*.

        Matching strategy:
        1. exact match first (avoids over-matching symbols that share the
           same method name across different classes).
        2. fallback to method-name suffix match.

        Dedup by (caller, callee, file_path, line).
        """
        exact = [edge for edge in self.call_edges if getattr(edge, field) == symbol_name]
        if exact:
            return exact

        parts = symbol_name.split(".")
        method = parts[-1] if parts else symbol_name

        result: list[CallEdge] = []
        seen: set[tuple[str, str, str, int]] = set()

        for edge in self.call_edges:
            val = getattr(edge, field)
            if not val:
                continue
            if val.endswith(f".{method}") or val == method:
                key = (edge.caller, edge.callee, edge.file_path, edge.line)
                if key not in seen:
                    seen.add(key)
                    result.append(edge)

        return result

    def get_callers(self, symbol_name: str) -> list[CallEdge]:
        """Return all call edges where the given symbol is the callee.

        Matching strategy:
        1. exact match first
        2. if no exact match exists, fallback to method-name suffix match
        """
        return self._edges_by_symbol_field("callee", symbol_name)

    def get_callees(self, symbol_name: str) -> list[CallEdge]:
        """Return all call edges where the given symbol is the caller.

        Matching strategy mirrors get_callers():
        1. Exact match first (fastest, avoids false positives).
        2. Suffix match fallback: ``execute_plan_canonical`` matches
           ``OperationExecutor.execute_plan_canonical``.

        The suffix fallback is necessary because _get_current_symbol()
        returns the qualname (e.g. ``OperationExecutor.execute_plan_canonical``)
        but callers of get_callees() typically use the bare method name.
        """
        return self._edges_by_symbol_field("caller", symbol_name)

    def get_file_dependencies(self, file_path: str) -> list[ImportEdge]:
        """Return all import edges where the given file is the importer."""
        return [edge for edge in self.import_edges if edge.importer == file_path]

    def get_importers(self, file_path: str) -> list[str]:
        """Return file paths that import the given file (reverse dependency lookup).

        ``file_path`` is a relative path like ``external_llm/agent/foo.py``.
        It is converted to a dotted module prefix (``external_llm.agent.foo``),
        then all ImportEdge entries whose ``imported`` starts with that prefix
        are collected and their ``importer`` file paths returned (deduped).
        """
        if not file_path:
            return []
        if LanguageId.from_path(file_path) is not LanguageId.PYTHON:
            # Non-Python (TS/JS/...) import edges store module paths, not
            # dotted names: imported="../string_utils" from
            # "__tests__/x.test.ts".  Resolve relative to the importer's
            # directory and match against the extensionless candidate path.
            _cand_noext = os.path.splitext(file_path)[0]
            _importers: list[str] = []
            _seen: set = set()
            for edge in self.import_edges:
                _imp = edge.imported or ""
                if not _imp or edge.importer in _seen:
                    continue
                if _imp.startswith("."):
                    _resolved = os.path.normpath(
                        os.path.join(os.path.dirname(edge.importer or ""), _imp)
                    )
                else:
                    _resolved = os.path.normpath(_imp)
                if _resolved in (_cand_noext, file_path):
                    _seen.add(edge.importer)
                    _importers.append(edge.importer)
            return _importers
        # Convert "a/b/c.py" → "a.b.c"
        _module_prefix = file_path.replace("/", ".").replace("\\", ".")[:-3]
        # basename fallback: graph builder uses relative import without absolute path
        # Handle stored as "module_name.Symbol" form.
        # e.g. imported="operation_models.X" (from .operation_models import X)
        _module_basename = _module_prefix.rsplit(".", 1)[-1]  # "operation_models"
        _basename_differs = _module_basename != _module_prefix
        # Collect unique importer file paths
        _importers: list[str] = []
        _seen: set = set()
        for edge in self.import_edges:
            _imp = edge.imported or ""
            # Match exact module or a sub-name within it (e.g. "a.b.c.Foo")
            if _imp == _module_prefix or _imp.startswith(_module_prefix + "."):
                if edge.importer not in _seen:
                    _seen.add(edge.importer)
                    _importers.append(edge.importer)
            elif _basename_differs and (
                _imp == _module_basename or _imp.startswith(_module_basename + ".")
            ):
                # Match relative import form ("operation_models.X") —
                # only valid when importer is in the same package directory as the candidate file
                import os as _os_gi
                _cand_dir = _os_gi.path.dirname(file_path)
                _imp_dir = _os_gi.path.dirname(edge.importer or "")
                if _cand_dir == _imp_dir and edge.importer not in _seen:
                    _seen.add(edge.importer)
                    _importers.append(edge.importer)
        return _importers

    def get_symbols_in_file(self, file_path: str) -> list[SymbolNode]:
        """Return all symbols defined in the given file."""
        symbol_ids = self.file_symbols.get(file_path, [])
        return [self.symbols[sid] for sid in symbol_ids if sid in self.symbols]

    def remove_file(self, rel_path: str) -> None:
        """Remove all symbols, call edges, and import edges for a file."""
        # Remove symbols
        symbol_ids = self.file_symbols.pop(rel_path, [])
        for sid in symbol_ids:
            self.symbols.pop(sid, None)

        # Remove from _symbol_locations
        keys_to_remove = [k for k in self._symbol_locations if k[1] == rel_path]
        for k in keys_to_remove:
            del self._symbol_locations[k]

        # Remove call edges from this file
        self.call_edges = [e for e in self.call_edges if e.file_path != rel_path]

        # Remove import edges from this file
        self.import_edges = [e for e in self.import_edges if e.importer != rel_path]

    def reparse_file(self, abs_path: str) -> None:
        """Re-parse a single file: remove old data, then re-process.

        Args:
            abs_path: Absolute path to the file.
        """
        rel_path = os.path.relpath(abs_path, self.repo_root)
        self.remove_file(rel_path)
        try:
            self._process_file(abs_path)
        except (SyntaxError, UnicodeDecodeError):
            pass  # File has errors — just leave it removed


# ── Mutating-call detection ─────────────────────────────────────────────────
# A call is "mutating" (has write/side-effect semantics) when:
# 1. The return value is discarded (call used as statement — ast.Expr parent node).
# 2. The method name is a known Python data-model mutator (__setitem__, etc.).
# 3. The callee is a conventional mutating verb with a state-store receiver.
#
# Signals 1+2 are structurally derived from the AST and language spec.
# Signal 3 is a heuristic fallback for conventions not captured by 1+2.

# Python data-model mutating methods (language spec — exact, not heuristic).
_MUTATING_DUNDER = frozenset({
    "__setitem__", "__delitem__", "__iadd__", "__isub__", "__imul__",
    "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__",
    "__ilshift__", "__irshift__", "__iand__", "__ixor__", "__ior__",
    "__setattr__", "__delattr__", "__set_name__",
})


def _is_mutating_call(node: ast.Call, callee: str, parent_is_expr: bool = False) -> bool:
    """Return True when structural analysis suggests write/side-effect semantics.

    Priority:
      1. Parent is ast.Expr (return value discarded) — strongest signal.
      2. Method is a Python data-model mutator (exact match).
      3. Method name + receiver suggests conventional state-mutation pattern.
    """
    # Signal 1: return value discarded → almost certainly mutating
    if parent_is_expr:
        return True

    bare = callee.split(".")[-1].lower()

    # Signal 2: Python data-model mutating methods (exact match, language spec)
    if bare in _MUTATING_DUNDER:
        return True

    # Signal 3: conventional naming patterns — receiver.method(...) where
    # method is a mutating verb and receiver is a stateful object.
    # This is a heuristic fallback, not a structural guarantee.
    if isinstance(node.func, ast.Attribute):
        method = node.func.attr.lower()
        _mutating_methods = {
            "append", "extend", "insert", "pop", "remove", "clear",
            "add", "discard", "update", "difference_update",
            "symmetric_difference_update", "intersection_update",
        }
        if method in _mutating_methods:
            return True

    return False


class GraphVisitor(ast.NodeVisitor):
    """AST visitor that extracts symbols, calls, and imports.

    Tracks parent nodes via _parent_stack so call-site analysis can determine
    whether a call's return value is used or discarded.
    """

    def __init__(self, file_path: str, repo_root: str):
        self.file_path = file_path
        self.repo_root = repo_root
        self.symbols: list[SymbolNode] = []
        self.calls: list[CallEdge] = []
        self.imports: list[ImportEdge] = []
        self.current_class: Optional[str] = None
        self._in_function: int = 0  # nesting depth — guards module-level constant detection
        self._parent_stack: list[ast.AST] = []
        # Scope-qualname stack: carries function AND class context (innermost
        # last) so nested symbols get unique, fully-qualified qualnames like
        # ``deco_a.wrapper`` instead of bare names. Without this, two sibling
        # scopes defining a function of the same bare name (e.g. two
        # ``def wrapper`` closures in different decorators) collide on
        # ``unique_id = f"{path}:{qualname}"`` — the first definition is
        # silently overwritten in RepositoryGraph.symbols and file_symbols
        # accumulates duplicate entries. Mirrors Python's own __qualname__
        # nesting semantics.
        self._scope_stack: list[str] = []
        # Compute module name from file path relative to repo root
        self.module = self._path_to_module(file_path)

    def visit(self, node: ast.AST) -> None:
        """Override visit to track parent nodes."""
        self._parent_stack.append(node)
        super().visit(node)
        self._parent_stack.pop()

    def _is_call_used_as_stmt(self, call_node: ast.Call) -> bool:
        """Check if a Call node's parent is an ast.Expr (return value discarded)."""
        if len(self._parent_stack) < 2:
            return False
        parent = self._parent_stack[-2]
        return isinstance(parent, ast.Expr)

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to Python module name."""
        # Remove repo_root prefix and .py extension, replace / with .
        rel_path = os.path.relpath(file_path, self.repo_root)
        if LanguageId.from_path(rel_path) is LanguageId.PYTHON:
            rel_path = rel_path[:-3]
        # Handle __init__ files
        if rel_path.endswith('/__init__'):
            rel_path = rel_path[:-9]
        elif rel_path.endswith('__init__'):
            rel_path = rel_path[:-9]
        # Replace path separators
        module = rel_path.replace('/', '.')
        # Remove leading dots
        if module.startswith('.'):
            module = module[1:]
        return module

    def _compute_signature_hash(self, node: ast.FunctionDef) -> Optional[str]:
        """Compute a hash of the function signature."""
        try:
            parts = [node.name]
            # Add positional arguments
            for arg in node.args.args:
                parts.append(arg.arg)
            # Add vararg
            if node.args.vararg:
                parts.append('*' + node.args.vararg.arg)
            # Add kwonlyargs
            for arg in node.args.kwonlyargs:
                parts.append(arg.arg)
            # Add kwarg
            if node.args.kwarg:
                parts.append('**' + node.args.kwarg.arg)
            signature = ','.join(parts)
            # Compute SHA1 hash (hex digest)
            return hashlib.sha1(signature.encode(), usedforsecurity=False).hexdigest()[:8]  # first 8 chars
        except Exception:
            return None

    def _extract_signature(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract full function signature text with type annotations."""
        try:
            params = []
            args = node.args

            # positional args (includes self)
            defaults_offset = len(args.args) - len(args.defaults)
            for i, arg in enumerate(args.args):
                p = arg.arg
                if arg.annotation:
                    try:
                        p += f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        pass
                # defaults
                default_idx = i - defaults_offset
                if default_idx >= 0 and default_idx < len(args.defaults):
                    try:
                        p += f" = {ast.unparse(args.defaults[default_idx])}"
                    except Exception:
                        pass
                params.append(p)

            # *args
            if args.vararg:
                p = f"*{args.vararg.arg}"
                if args.vararg.annotation:
                    try:
                        p += f": {ast.unparse(args.vararg.annotation)}"
                    except Exception:
                        pass
                params.append(p)
            elif args.kwonlyargs:
                params.append("*")

            # keyword-only args
            for i, arg in enumerate(args.kwonlyargs):
                p = arg.arg
                if arg.annotation:
                    try:
                        p += f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        pass
                if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                    try:
                        p += f" = {ast.unparse(args.kw_defaults[i])}"
                    except Exception:
                        pass
                params.append(p)

            # **kwargs
            if args.kwarg:
                p = f"**{args.kwarg.arg}"
                if args.kwarg.annotation:
                    try:
                        p += f": {ast.unparse(args.kwarg.annotation)}"
                    except Exception:
                        pass
                params.append(p)

            ret = ""
            if node.returns:
                try:
                    ret = f" -> {ast.unparse(node.returns)}"
                except Exception:
                    pass

            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{prefix} {node.name}({', '.join(params)}){ret}"
        except Exception:
            return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definition."""
        # Qualified name encodes the full scope path (functions AND classes)
        # so nested functions get unique qualnames — e.g. a ``wrapper``
        # closure inside ``deco_a`` resolves to ``deco_a.wrapper`` rather
        # than the bare ``wrapper`` that collides with every other sibling
        # scope's ``wrapper``. ``current_class`` is retained as a defensive
        # fallback for any code path that sets class context without pushing
        # onto the scope stack.
        if self._scope_stack:
            qualname = f"{self._scope_stack[-1]}.{node.name}"
        elif self.current_class:
            qualname = f"{self.current_class}.{node.name}"
        else:
            qualname = node.name

        # Compute signature hash for functions/methods
        signature_hash = self._compute_signature_hash(node)

        symbol = SymbolNode(
            name=node.name,
            qualname=qualname,
            module=self.module,
            file_path=self.file_path,
            kind="function" if not self.current_class else "method",
            start_line=node.lineno,
            end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            signature_hash=signature_hash,
            docstring=ast.get_docstring(node),
            signature=self._extract_signature(node),
        )
        self.symbols.append(symbol)

        # Visit child nodes to capture calls inside this function
        self._scope_stack.append(qualname)
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definition."""
        # Qualify local classes (defined inside a function scope) so two
        # functions each defining ``class Foo`` don't collide on qualname —
        # mirrors the nested-function fix.
        if self._scope_stack:
            class_qualname = f"{self._scope_stack[-1]}.{node.name}"
        else:
            class_qualname = node.name

        # Extract base class names
        bases: list[str] = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)

        symbol = SymbolNode(
            name=node.name,
            qualname=class_qualname,
            module=self.module,
            file_path=self.file_path,
            kind="class",
            start_line=node.lineno,
            end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            signature_hash=None,
            docstring=ast.get_docstring(node),
            bases=bases if bases else None,
        )
        self.symbols.append(symbol)

        # Visit methods inside class
        previous_class = self.current_class
        self.current_class = class_qualname
        self._scope_stack.append(class_qualname)
        self.generic_visit(node)
        self._scope_stack.pop()
        self.current_class = previous_class

    def _resolve_call_name(self, node: ast.AST) -> Optional[str]:
        """Resolve a call expression to a normalised string name.

        Normalisation rule: strip ``self.`` and ``cls.`` prefixes so that
        ``self._schedule_operations()`` resolves to ``_schedule_operations``
        rather than ``self._schedule_operations``.  This keeps callee names
        consistent with the bare method names stored in SymbolNode.qualname,
        which is what ``get_callees`` / ``get_callers`` query against.

        Without this, ``get_callees('execute_plan_canonical')`` would never
        match edges whose callee was stored as ``self.execute_plan_canonical``.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._resolve_call_name(node.value)
            if value is None:
                return None
            # Strip instance/class receiver — keep only the method name.
            # "self.foo" → "foo", "cls.bar" → "bar"
            if value in ("self", "cls"):
                return node.attr
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Call):
            # Chained calls like obj.method()() — resolve the function part
            return self._resolve_call_name(node.func)
        else:
            return None

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function calls with object-identity and side-effect annotations."""
        callee = self._resolve_call_name(node.func)
        if callee is None:
            # Unsupported call expression
            self.generic_visit(node)
            return

        caller = self._get_current_symbol(line=node.lineno)
        if caller:
            # Object identity: capture literal positional arg values.
            # get_user(1) → ["1"], fetch("admin") → ['"admin"'].
            # Only ast.Constant nodes — expressions like f(x+1) are skipped.
            call_args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    call_args.append(repr(arg.value))

            # Side-effect semantics: AST-structural mutating call detection.
            is_mutating = _is_mutating_call(node, callee, parent_is_expr=self._is_call_used_as_stmt(node))

            edge = CallEdge(
                caller=caller,
                callee=callee,
                file_path=self.file_path,
                line=node.lineno,
                call_args=call_args,
                is_mutating=is_mutating,
            )
            self.calls.append(edge)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statements."""
        for alias in node.names:
            edge = ImportEdge(
                importer=self.file_path,
                imported=alias.name,
                import_type="import",
                alias=alias.asname
            )
            self.imports.append(edge)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from ... import statements."""
        module = node.module or ""
        for alias in node.names:
            imported = f"{module}.{alias.name}" if module else alias.name
            edge = ImportEdge(
                importer=self.file_path,
                imported=imported,
                import_type="from",
                alias=alias.asname
            )
            self.imports.append(edge)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Index module-level variable/constant assignments (e.g. WRITE_OP_KINDS = frozenset(...))."""
        if self._in_function > 0 or self.current_class:
            self.generic_visit(node)
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                symbol = SymbolNode(
                    name=target.id,
                    qualname=target.id,
                    module=self.module,
                    file_path=self.file_path,
                    kind="constant",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature_hash=None,
                    docstring=None,
                )
                self.symbols.append(symbol)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Index module-level annotated assignments (e.g. TIMEOUT: int = 30)."""
        if self._in_function > 0 or self.current_class:
            self.generic_visit(node)
            return
        if isinstance(node.target, ast.Name):
            symbol = SymbolNode(
                name=node.target.id,
                qualname=node.target.id,
                module=self.module,
                file_path=self.file_path,
                kind="constant",
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature_hash=None,
                docstring=None,
            )
            self.symbols.append(symbol)
        self.generic_visit(node)

    def _get_current_symbol(self, line: Optional[int] = None) -> Optional[str]:
        """Get the qualname of the function/method that contains *line*.

        When *line* is provided, uses line-range matching against all function
        and method symbols to correctly handle nested functions. Returns the
        innermost function whose range [start_line, end_line] contains *line*.

        This fixes the nested-function scope bug where code after a nested
        function definition (but still inside the outer function) was incorrectly
        attributed to the nested function's qualname.

        When *line* is None, falls back to returning the most recently added
        function/method symbol (the original behavior for callers that don't
        have a line number context).
        """
        if line is not None:
            # Line-range matching: find the innermost function/method
            # whose scope contains the given line number.
            best: Optional[tuple[str, int]] = None  # (qualname, span)
            for symbol in self.symbols:
                if symbol.kind not in ("function", "method"):
                    continue
                if symbol.start_line <= line <= symbol.end_line:
                    span = symbol.end_line - symbol.start_line
                    if best is None or span < best[1]:
                        best = (symbol.qualname, span)
            if best is not None:
                return best[0]
            # No enclosing function found — module-level code
            return None

        # Fallback: most recently added symbol (original behavior)
        if self.current_class and self.symbols:
            for symbol in reversed(self.symbols):
                if symbol.kind in ("function", "method"):
                    return symbol.qualname
        elif self.symbols:
            for symbol in reversed(self.symbols):
                if symbol.kind == "function":
                    return symbol.qualname
        return None
