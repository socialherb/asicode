"""
Call Graph Index for asicode Agent.

Builds a repo-wide call graph from Python AST, enabling:
  - forward edges:  caller -> callee
  - reverse edges:  callee <- caller
  - cross-file callee resolution (when definition is found in repo)

MVP scope: Python only.
"""
from __future__ import annotations

import ast
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from external_llm.graph.models import CallEdge

from ._shared_utils import (
    _walk_py_files as _shared_walk_py_files,
)
from ._shared_utils import (
    _walk_ts_js_files as _shared_walk_ts_js_files,
)
from .config.thresholds import config as _cfg

logger = logging.getLogger(__name__)

# Max files to AST-scan before stopping (avoids very large repos). Passed into
# the shared walkers (._shared_utils._walk_*_files) which apply the cap inside
# the walk loop. The shared walkers also cache per-root (TTL 30s) and share
# that cache with symbol_search, so call-graph builds no longer re-rglob.
_MAX_PY_FILES = _cfg.counts.SYMBOL_MAX_PY_FILES
_MAX_TS_FILES = _cfg.counts.SYMBOL_MAX_TS_FILES


def _walk_py_files(root: Path) -> list[Path]:
    """Walk repo returning .py files (shared, cached implementation)."""
    return _shared_walk_py_files(root, _MAX_PY_FILES)


def _walk_ts_js_files(root: Path) -> list[Path]:
    """Walk repo returning TS/JS files (shared, cached implementation)."""
    return _shared_walk_ts_js_files(root, _MAX_TS_FILES)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallGraphNode:
    symbol: str          # "foo" or "ClassName.method"
    file: str            # relative to repo root
    line: int
    kind: str            # function | async_function | method


# CallEdge is imported from external_llm.graph.models (canonical definition)



# ─────────────────────────────────────────────────────────────────────────────
# Indexer
# ─────────────────────────────────────────────────────────────────────────────

class CallGraphIndexer:
    """Repo-wide call graph index for Python files.

    Usage:
        idx = CallGraphIndexer("/path/to/repo")
        result = idx.get_related_symbols("MyClass.my_method")
    """

    def __init__(self, repo_root: str):
        self._root = Path(repo_root).resolve()
        # symbol -> first definition node
        self._nodes: dict[str, CallGraphNode] = {}
        # caller_symbol -> edges out
        self._forward: dict[str, list[CallEdge]] = {}
        # callee_symbol -> edges in
        self._reverse: dict[str, list[CallEdge]] = {}
        self._built = False
        # Guards _nodes/_forward/_reverse. This indexer is reference-shared
        # across parallel sub-agents (clone._call_graph = self._call_graph in
        # ToolRegistry), so EVERY mutator (build/invalidate) and reader
        # (get_callees/get_callers/_lookup_edges) must hold it. Without it a
        # concurrent invalidate()'s dict.clear() races a reader's dict iteration
        # -> "RuntimeError: dictionary changed size during iteration", or worse
        # a silently corrupted index. RLock so build() can call _resolve_callees()
        # (which reads the dicts) reentrantly without deadlock.
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Walk repo and build index. Safe to call multiple times (rebuilds).

        Holds _lock for the whole rebuild so concurrent readers (which also
        hold _lock) observe a consistent index rather than a half-built one.
        RLock makes the nested _resolve_callees() dict reads reentrant-safe and
        the build()->_resolve_callees()/_index_file() calls non-deadlocking.
        """
        with self._lock:
            self._nodes = {}
            self._forward = {}
            self._reverse = {}

            # ── Python files (existing) ──
            py_files = _walk_py_files(self._root)
            for py_file in py_files:
                try:
                    self._index_file(py_file)
                except SyntaxError:
                    pass  # skip unparseable files silently
                except Exception as e:
                    logger.debug(f"call_graph: skip {py_file}: {e}")

            # ── TS/JS files (Phase 4: opt-in) ──
            _ts_count = 0
            try:
                from config import MULTILANG_CALLGRAPH as _ML_CG
            except Exception:
                _ML_CG = False  # non-critical — never block execution
            if _ML_CG:
                ts_files = _walk_ts_js_files(self._root)
                for ts_file in ts_files:
                    try:
                        self._index_ts_file(ts_file)
                        _ts_count += 1
                    except Exception as e:
                        logger.debug("call_graph: skip TS %s: %s", ts_file, e)

            self._resolve_callees()
            self._built = True
            logger.debug(
                "call_graph: indexed %d symbols from %d py + %d ts files",
                len(self._nodes), len(py_files), _ts_count,
            )

    def invalidate(self) -> None:
        """Mark index as stale; it will be rebuilt on next access."""
        with self._lock:
            self._built = False
            self._nodes.clear()
            self._forward.clear()
            self._reverse.clear()

    def _lookup_edges(
        self,
        index: dict[str, list[CallEdge]],
        file_attr: str,
        symbol: str,
        file_path: Optional[str] = None,
    ) -> list[CallEdge]:
        """Resolve edges from *index* with exact-then-suffix matching.

        Matching strategy:
        1. Exact key lookup (fastest).
        2. Suffix fallback: ``execute_plan_canonical`` matches the index key
           ``OperationExecutor.execute_plan_canonical``.  Needed because
           _collect_calls stores callers under the qualified name
           (ClassName.method) but callers typically pass the bare method name.

        ``file_attr`` selects which CallEdge attribute
        (``'caller_file'`` / ``'callee_file'``) the optional *file_path*
        filter applies to.
        """
        edges = index.get(symbol, [])
        if not edges:
            bare = symbol.split(".")[-1]
            seen_keys: set = set()
            for key, key_edges in index.items():
                if (key.endswith(f".{bare}") or key == bare) and key not in seen_keys:
                    seen_keys.add(key)
                    edges = edges + key_edges
        if file_path and edges:
            matching = [e for e in edges if getattr(e, file_attr) == file_path]
            if matching:
                return matching
        return edges

    def get_callees(
        self, symbol: str, file_path: Optional[str] = None
    ) -> list[CallEdge]:
        """Return edges where symbol is the caller.

        Suffix fallback: ``execute_plan_canonical`` matches the index key
        ``OperationExecutor.execute_plan_canonical``.
        """
        self._ensure_built()
        with self._lock:
            return self._lookup_edges(self._forward, "caller_file", symbol, file_path)

    def get_callers(
        self, symbol: str, file_path: Optional[str] = None
    ) -> list[CallEdge]:
        """Return edges where symbol is the callee.

        Same suffix-fallback logic as get_callees(): ``_schedule_operations``
        matches the index key ``OperationExecutor._schedule_operations``.
        """
        self._ensure_built()
        with self._lock:
            return self._lookup_edges(self._reverse, "callee_file", symbol, file_path)

    def get_related_symbols(
        self,
        symbol: str,
        file_path: Optional[str] = None,
        depth: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return a structured summary for the given symbol.

        The whole body runs under _lock so the multi-step read (get_callees →
        get_callers → _nodes.get → scoring) observes one consistent index
        snapshot even if a concurrent invalidate() fires mid-way. The nested
        get_callees()/get_callers() calls re-acquire the RLock (reentrant).
        """
        self._ensure_built()
        with self._lock:
            callees = self.get_callees(symbol, file_path)
            callers = self.get_callers(symbol, file_path)

            # Optional depth-2 expansion (callees of callees)
            extra_callees: list[CallEdge] = []
            if depth > 1:
                for e in callees[:5]:
                    extra_callees.extend(self.get_callees(e.callee_symbol)[:5])

            # Build next_read_candidates: file-dedup, score-sorted, max 5
            file_scores: dict[str, float] = {}
            file_reasons: dict[str, str] = {}

            node = self._nodes.get(symbol)
            if node:
                _upd(file_scores, file_reasons, node.file, 1.0, "definition")

            for e in callees:
                if e.callee_file:
                    _upd(file_scores, file_reasons, e.callee_file,
                         e.confidence * 0.95, "direct callee")

            for e in callers:
                if e.caller_file:
                    _upd(file_scores, file_reasons, e.caller_file,
                         e.confidence * 0.70, "caller")

            for e in extra_callees:
                if e.callee_file:
                    _upd(file_scores, file_reasons, e.callee_file,
                         e.confidence * 0.50, "transitive callee")

            candidates = sorted(
                [
                    {"path": f, "reason": file_reasons[f], "score": round(s, 3)}
                    for f, s in file_scores.items()
                ],
                key=lambda x: -x["score"],
            )[:5]

            related_syms = sorted(
                set(
                    [e.callee_symbol for e in callees]
                    + [e.caller_symbol for e in callers]
                )
            )[:limit]

            return {
                "symbol": symbol,
                "node": (
                    {"file": node.file, "line": node.line, "kind": node.kind}
                    if node
                    else None
                ),
                "callees": [
                    {
                        "symbol": e.callee_symbol,
                        "display": e.callee_display,
                        "file": e.callee_file,
                        "line": e.callee_line,
                        "confidence": round(e.confidence, 2),
                    }
                    for e in callees[:limit]
                ],
                "callers": [
                    {
                        "symbol": e.caller_symbol,
                        "file": e.caller_file,
                        "line": e.caller_line,
                        "confidence": round(e.confidence, 2),
                    }
                    for e in callers[:limit]
                ],
                "related_symbols": related_syms,
                "next_read_candidates": candidates,
            }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_built(self) -> None:
        # Double-checked locking. Fast path: lock-free read of _built (a bool,
        # atomic in CPython). A True means the index is fully built (build()
        # sets _built=True last, under the lock); a stale True observed right
        # after invalidate() merely yields an empty result for that one call and
        # the next call rebuilds — never a torn-dict iteration.
        if self._built:
            return
        with self._lock:
            if not self._built:
                self.build()

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._root))
        except ValueError:
            return str(path)

    def _index_file(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
        rel = self._rel(path)

        # Map node id -> enclosing class name (for method naming)
        class_names: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_names[id(child)] = node.name

        self._collect_defs(tree, rel, class_names)
        self._collect_calls(tree, rel, class_names)

    def _collect_defs(
        self,
        tree: ast.AST,
        rel: str,
        class_names: dict[int, str],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_name = class_names.get(id(node))
                if class_name:
                    symbol = f"{class_name}.{node.name}"
                    kind = "method"
                else:
                    symbol = node.name
                    kind = (
                        "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    )
                if symbol not in self._nodes:  # first definition wins
                    self._nodes[symbol] = CallGraphNode(
                        symbol=symbol, file=rel, line=node.lineno, kind=kind
                    )

    def _collect_calls(
        self,
        tree: ast.AST,
        rel: str,
        class_names: dict[int, str],
    ) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            class_name = class_names.get(id(node))
            caller_sym = (
                f"{class_name}.{node.name}" if class_name else node.name
            )
            seen: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                edge = self._parse_call(
                    child, caller_sym, rel, node.lineno, class_name
                )
                if edge and edge.callee_display not in seen:
                    seen.add(edge.callee_display)
                    self._forward.setdefault(caller_sym, []).append(edge)
                    self._reverse.setdefault(edge.callee_symbol, []).append(edge)

    def _parse_call(
        self,
        call: ast.Call,
        caller_sym: str,
        caller_file: str,
        caller_line: int,
        class_name: Optional[str],
    ) -> Optional[CallEdge]:
        func = call.func
        if isinstance(func, ast.Name):
            # foo()
            return CallEdge(
                caller_symbol=caller_sym,
                caller_file=caller_file,
                caller_line=caller_line,
                callee_symbol=func.id,
                callee_display=func.id,
                confidence=0.9,
            )
        if isinstance(func, ast.Attribute):
            # Reconstruct dotted name from attribute chain
            parts: list[str] = []
            n: ast.expr = func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if not isinstance(n, ast.Name):
                return None
            parts.append(n.id)
            dotted = ".".join(reversed(parts))
            root_name = parts[-1]   # outermost name (e.g. "self", "obj")
            attr = parts[0]         # the actual method/function name
            if root_name == "self" and class_name:
                # self.method() -> ClassName.method (high confidence)
                return CallEdge(
                    caller_symbol=caller_sym,
                    caller_file=caller_file,
                    caller_line=caller_line,
                    callee_symbol=f"{class_name}.{attr}",
                    callee_display=dotted,
                    confidence=0.85,
                )
            # obj.method() or module.func() -> lower confidence
            return CallEdge(
                caller_symbol=caller_sym,
                caller_file=caller_file,
                caller_line=caller_line,
                callee_symbol=attr,
                callee_display=dotted,
                confidence=0.5,
            )
        return None

    def _index_ts_file(self, path: Path) -> None:
        """Index a TS/JS file using TSSemanticTracer for call graph edges."""
        from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

        from ..languages.models import LanguageId

        content = path.read_text(encoding="utf-8", errors="replace")
        rel = self._rel(path)
        lang_id = LanguageId.from_path(str(path))
        lang_str = "typescript" if lang_id == LanguageId.TYPESCRIPT else "javascript"

        tracer = TSSemanticTracer(language=lang_str)
        module = tracer.analyze_core(content, str(path))

        # Register function definitions
        for fn in module.functions:
            if fn.name and fn.name not in self._nodes:
                self._nodes[fn.name] = CallGraphNode(
                    symbol=fn.name, file=rel,
                    line=fn.meta.start_line if fn.meta else fn.start_line,
                    kind="async_function" if fn.is_async else "function",
                )

        # Register class methods
        for cls in module.classes:
            for method in cls.methods:
                symbol = f"{cls.name}.{method.name}"
                if symbol not in self._nodes:
                    self._nodes[symbol] = CallGraphNode(
                        symbol=symbol, file=rel,
                        line=method.meta.start_line if method.meta else method.start_line,
                        kind="method",
                    )

        # Register call edges from TSModule.call_sites
        for cs in module.call_sites:
            if not cs.caller or not cs.callee:
                continue
            callee_display = (
                f"{cs.receiver}.{cs.callee}" if cs.receiver else cs.callee
            )
            edge = CallEdge(
                caller_symbol=cs.caller,
                caller_file=rel,
                caller_line=cs.line,
                callee_symbol=cs.callee,
                callee_display=callee_display,
                confidence=0.8 if cs.is_method_call else 0.9,
            )
            self._forward.setdefault(cs.caller, []).append(edge)
            self._reverse.setdefault(cs.callee, []).append(edge)

    def _resolve_callees(self) -> None:
        """Fill callee_file / callee_line using the collected node index."""
        for edges in self._forward.values():
            for edge in edges:
                node = self._nodes.get(edge.callee_symbol)
                if node:
                    edge.callee_file = node.file
                    edge.callee_line = node.line


# ─────────────────────────────────────────────────────────────────────────────
# Internal utility
# ─────────────────────────────────────────────────────────────────────────────

def _upd(
    scores: dict[str, float],
    reasons: dict[str, str],
    path: str,
    score: float,
    reason: str,
) -> None:
    if path not in scores or scores[path] < score:
        scores[path] = score
        reasons[path] = reason
