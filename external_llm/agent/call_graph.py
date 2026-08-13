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
import contextlib
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from external_llm.graph.models import CallEdge
from external_llm.graph.root_cache import RootCache
from external_llm.graph.structural_cache import default_cache_path as _rg_default_cache_path
from external_llm.graph.structural_cache import load as _load_rg_snapshot

from ..analysis import parse_cache
from ..common.walk_policy import (
    _path_is_walk_admissible,
    _walk_dir_sort_key,
    _walk_should_skip_dir,
)
from ._shared_utils import (
    _PY_EXTENSIONS,
    _TS_JS_EXTENSIONS,
)
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
# Per-FILE size gates. The file-count caps above bound how MANY files are
# parsed and said nothing about how big one is — see the thresholds for the
# measurements.
_MAX_PY_BYTES = _cfg.lines.CALLGRAPH_PY_MAX_BYTES
_MAX_TS_BYTES = _cfg.lines.CALLGRAPH_TS_MAX_BYTES


# ─────────────────────────────────────────────────────────────────────────────
# Per-file cache tiers (P2, 2026-08-12; CGI's own disk snapshot REMOVED P3
# Stage 3, 2026-08-12 — RepositoryGraph's snapshot is the single SSOT):
#
#   build() → _process_file_cached:  in-process _file_cache → RG snapshot
#   (.cache/structural_graph_v1.json, converted via _rg_payload_to_cgi) →
#   fresh parse.  The RG tier turns a cold-process first build after a gate /
#   graph_builder run into stat + JSON lookups; the process-wide _file_cache
#   serves warm rebuilds in the same process.  Both tiers store JSON-ready
#   payloads — edges are reconstructed fresh on injection (_inject_file_data)
#   so the graph's in-place callee resolution (_resolve_edges) can never
#   pollute the cache.
# ─────────────────────────────────────────────────────────────────────────────

_FILE_CACHE_MAX_ENTRIES = 2048
"""Admission cap for the in-process per-file cache (mirrors
repository_graph._EXTRACT_CACHE_MAX_ENTRIES): root-partitioned fair
admission (RootCache) — each active root keeps its guaranteed share; refused
files are parsed and injected but not cached."""

_file_cache = RootCache(_FILE_CACHE_MAX_ENTRIES)
"""Process-wide per-file extraction cache: (root, abs_path) →
(mtime_ns, size, JSON-ready payload).  Root-partitioned fair admission
(2026-08-12): in multi-repo (webapp) processes a repo arriving after the
cache is full no longer starves — it claims slots from the most-over-quota
hoarder; single-root behavior is unchanged.  Survives across indexer
instances and rebuilds — stamp-keyed, exactly like
RepositoryGraph._extract_cache."""

_file_cache_gc_deficit = 0


def _too_big_to_index(path: Path, max_bytes: int) -> bool:
    """True when one file is too large to read + parse in this process.

    build() is the last unbounded parse loop in the agent: it reads and
    ``ast.parse``s every walked ``.py``, and TSSemanticTracer-parses every
    walked ``.ts``/``.js``, with no size check anywhere in the chain (the
    shared walkers filter on extension only). One 3.7 MB generated module was
    enough to cost 10.12 s and 762 MB peak RSS, and the entry points are the
    shipping ``analyze_impact`` / ``trace_call_path`` tools.

    Degradation is partial, never fatal: the file's symbols are absent from the
    graph, so callers see the same thing an unparseable file already produces
    (``build()`` catches SyntaxError per file and keeps going). Callers that
    need symbols from an oversized file still have find_symbol and the rg
    outline path.

    A stat failure is NOT treated as oversized — an unreadable file falls
    through to the existing per-file ``except`` in build(), which is where that
    case has always been handled.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= max_bytes:
        return False
    logger.debug(
        "call_graph: skipping %s (%d bytes > %d)", path, size, max_bytes,
    )
    return True


def _gc_file_cache() -> None:
    """Dead-entry sweep for _file_cache (mirror of repository_graph._gc_names_cache).

    Sweeps only entries whose source file no longer exists — live files stay
    warm across rebuilds.  Rate-limited by a deficit counter so each sweep
    defers the next by one cap worth of insertions (total stat work stays
    ~O(N) per build).
    """
    global _file_cache_gc_deficit
    if _file_cache_gc_deficit > 0:
        _file_cache_gc_deficit -= 1
        return
    _file_cache.sweep_dead()
    _file_cache_gc_deficit = _FILE_CACHE_MAX_ENTRIES


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

    def __init__(
        self,
        repo_root: str,
        cancel_event: Optional[threading.Event] = None,
        config: Any = None,
    ):
        self._root = Path(repo_root).resolve()
        # Cooperative cancel: when set (ESC / Ctrl-C in the agent loop), the
        # repo-wide ast.parse loop in build() bails out between files.  We hold
        # a *config* reference (NOT the event value) and read
        # ``config.cancel_event`` FRESH at build() time via _get_cancel_event —
        # mirroring the call-time fresh read vulture uses in analysis_tools.
        # This is required by the design-chat REPL, which sets
        # ``config.cancel_event`` PER TURN (asi.py) AFTER the ToolRegistry — and
        # thus this indexer — was constructed with cancel_event=None; a capture-
        # at-construction value would freeze None forever and leave ESC inert on
        # the exact interactive path it is meant to protect.  An explicit
        # ``cancel_event`` arg (tests, direct callers without a config) still
        # takes precedence.  Cloned ToolRegistries share this indexer AND the
        # parent's config (clone.config = self.config, clone._call_graph =
        # self._call_graph), so one event reaches all holders without per-clone
        # wiring.
        self._cancel_event = cancel_event
        self._config = config
        # symbol -> first definition node
        self._nodes: dict[str, CallGraphNode] = {}
        # caller_symbol -> edges out
        self._forward: dict[str, list[CallEdge]] = {}
        # callee_symbol -> edges in
        self._reverse: dict[str, list[CallEdge]] = {}
        # bare-name -> [qualified keys sharing that last segment] — the
        # suffix-fallback index for _lookup_edges (M1). Built once at the end
        # of build() (after _resolve_callees so the key universe is final);
        # cleared on every rebuild/cancel/invalidate alongside the dicts so a
        # stale bare index can never outlive its _forward/_reverse.
        self._bare_index: dict[str, list[str]] = {}
        # Per-file contribution tracking for invalidate_files() incremental
        # updates:
        #   _file_nodes: rel -> [symbols whose node OWNERSHIP belongs to this
        #     file] (first-definition-wins — the owning file recorded the node)
        #   _file_edges: rel -> [edges collected from this file] (identity-
        #     based removal, so equal-but-distinct edges from other files are
        #     never touched)
        #   _def_sources: symbol -> {rel: (line, kind)} for EVERY defining
        #     file — lets remove-file reassign node ownership to the next
        #     defining file (walk order ⇒ _owner_rank min) without re-parsing
        #   _file_defs: rel -> [symbols DEFINED here, owned or not] — the
        #     cleanup mirror for _def_sources.  Without it a file that only
        #     shadowed a symbol (never owned it) left a ghost (rel → stale
        #     line) in _def_sources when rewritten/deleted, and setdefault in
        #     _register_node preserved that stale line on re-registration —
        #     deleting the owner then reassigned the node to a line that no
        #     longer exists.
        self._file_nodes: dict[str, list[str]] = {}
        self._file_edges: dict[str, list[CallEdge]] = {}
        self._def_sources: dict[str, dict[str, tuple[int, str]]] = {}
        self._file_defs: dict[str, list[str]] = {}
        # rel-dir -> pre-order index in build()'s walk order (root "" -> 0).
        # Ownership comparisons (_owner_rank) must mirror the walk exactly —
        # lexicographic min(srcs) diverges from it because os.walk visits
        # root-level files BEFORE descending into subdirectories and
        # deprioritizes tests/fixtures dirs via _walk_dir_sort_key.
        # Rebuilt in build() and invalidate_files() (dir-only walk, no I/O).
        self._dir_ranks: dict[str, int] = {"": 0}
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
        # P3 Stage 2/3: RepositoryGraph's snapshot (.cache/structural_graph_v1.json)
        # as the SSOT — the ONLY disk tier (CGI's own snapshot was removed,
        # P3 Stage 3 2026-08-12).  When it exists and a file's manifest stamp
        # matches, the build serves RG's per-file extraction CONVERTED to CGI
        # payload (_rg_payload_to_cgi) instead of CGI's own parse — the release
        # gate and the agent then share one extraction.  Lazy-loaded per build,
        # mtime-guarded, fail-open to a fresh parse (the RG snapshot is written
        # by RG builds with collect_imported_names=True — graph_builder does
        # this on every agent-side build).
        self._rg_cache_path: Path = _rg_default_cache_path(self._root)
        self._rg_cache: Optional[dict] = None
        self._rg_cache_mtime_ns = 0
        # Per-build accounting (mirrors RepositoryGraph.cache_stats):
        #   total   = walked py files with a payload
        #   hit     = served from a cache tier (in-process or RG snapshot)
        #   changed = computed AND admitted (cap-overflow computes are neither
        #             cached nor counted — they are transient; the RG snapshot
        #             converges them on the next RG build)
        self.cache_stats = {"hit": 0, "total": 0, "changed": 0}
        self._py_stamps: list[tuple[str, Path, os.stat_result]] = []
        self._fresh_parsed: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def _get_cancel_event(self) -> Optional[threading.Event]:
        """Return the live cooperative-cancel event.

        Reads ``config.cancel_event`` FRESH (call-time, not construction-time)
        so a per-turn mutation of ``config.cancel_event`` — as the design-chat
        REPL performs each turn — is honored even though this indexer was
        constructed before the mutation landed.  An explicit ``cancel_event``
        passed to ``__init__`` (tests / direct callers without a config) takes
        precedence and is returned as-is.  Returns None when neither is set
        (non-interactive CLI, out-of-process callers) → checkpoints become
        inert no-ops, matching the pre-cancel behavior.
        """
        if self._cancel_event is not None:
            return self._cancel_event
        return getattr(self._config, "cancel_event", None)

    def build(self) -> None:
        """Walk repo and build index. Safe to call multiple times (rebuilds).

        Holds _lock for the whole rebuild so concurrent readers (which also
        hold _lock) observe a consistent index rather than a half-built one.
        RLock makes the nested _resolve_callees() dict reads reentrant-safe and
        the build()->_resolve_callees()/_process_file_cached() calls non-deadlocking.
        """
        with self._lock:
            self._nodes = {}
            self._forward = {}
            self._reverse = {}
            self._bare_index = {}
            self._file_nodes = {}
            self._file_edges = {}
            self._def_sources = {}
            self._file_defs = {}
            self._dir_ranks = self._compute_dir_ranks()
            # Per-build cache accounting (the process-wide _file_cache and the
            # on-disk snapshot survive across builds — stamp-keyed, so stale
            # entries self-heal on mismatch).
            self.cache_stats = {"hit": 0, "total": 0, "changed": 0, "rg_served": 0, "rg_self_healed": 0}
            self._py_stamps = []
            self._fresh_parsed = []

            # ── Python files (existing) ──
            py_files = _walk_py_files(self._root)
            # Size the shared parse cache to the working set BEFORE the first
            # parse (P1, 2026-08-11): the structural scanners and
            # RepositoryGraph.extract_file share this cache, so sizing it here
            # lets a cold agent start reuse whichever consumer warmed it first
            # (and lets this build reuse entries it warms). Monotonic/no-op when
            # already sized; the dominant parse cost of a first query (several
            # seconds on large repos) is then paid at most once per file per
            # stat-version across all consumers.
            parse_cache.ensure_capacity(len(py_files))
            for _fi, py_file in enumerate(py_files):
                # Cooperative cancel: the ast.parse loop is the dominant cost of a
                # first graph query (several seconds on large repos).  Bail out
                # between files.  _built is left False (set only at the end of
                # build()) and the dicts are already empty (reset above), so no
                # partially-built index is ever observed by readers (which hold
                # the same _lock and only run after build() returns).
                # Fresh read each iteration so a mid-build ESC (design-chat sets
                # config.cancel_event on the live config object) is honored.
                _ce = self._get_cancel_event()
                if _ce is not None and _ce.is_set():
                    logger.debug("call_graph: build cancelled at %d/%d py files", _fi, len(py_files))
                    # Discard the partial index — _inject_file_data writes
                    # directly into self._nodes/_forward/_reverse, so a
                    # mid-build cancel would otherwise leave a torn index
                    # visible to the in-flight query (which then runs
                    # _lookup_edges on a partial _forward).
                    self._nodes = {}
                    self._forward = {}
                    self._reverse = {}
                    self._bare_index = {}
                    self._file_nodes = {}
                    self._file_edges = {}
                    self._def_sources = {}
                    self._file_defs = {}
                    # Release the RG snapshot payload before the early return
                    # (P2, 2026-08-12): a cancelled build must not leak the
                    # whole 42MB JSON's in-memory form (~209MB) for the
                    # indexer's lifetime.
                    self._rg_cache = None
                    self._rg_cache_mtime_ns = 0
                    return
                try:
                    self._process_file_cached(py_file)
                except SyntaxError:
                    logger.debug("call_graph: skip unparseable %s", py_file, exc_info=True)
                except Exception as e:
                    logger.debug("call_graph: skip %s: %s", py_file, e)

            # ── TS/JS files (Phase 4: opt-in) ──
            _ts_count = 0
            if self._multilang_callgraph_enabled():
                ts_files = _walk_ts_js_files(self._root)
                for ts_file in ts_files:
                    _ce = self._get_cancel_event()
                    if _ce is not None and _ce.is_set():
                        logger.debug("call_graph: build cancelled during TS indexing")
                        self._nodes = {}
                        self._forward = {}
                        self._reverse = {}
                        self._bare_index = {}
                        self._file_nodes = {}
                        self._file_edges = {}
                        self._def_sources = {}
                        self._file_defs = {}
                        # Release the RG snapshot payload (P2, 2026-08-12) —
                        # same leak guard as the py-loop ESC above.
                        self._rg_cache = None
                        self._rg_cache_mtime_ns = 0
                        return
                    try:
                        self._index_ts_file(ts_file)
                        _ts_count += 1
                    except Exception as e:
                        logger.debug("call_graph: skip TS %s: %s", ts_file, e)

            self._resolve_callees()
            self._rebuild_bare_index()
            self.cache_stats["total"] = len(self._py_stamps)
            self.cache_stats["changed"] = len(self._fresh_parsed)
            self._built = True
            logger.debug(
                "call_graph: indexed %d symbols from %d py + %d ts files "
                "(cache %d hit / %d changed / %d rg-served)",
                len(self._nodes), len(self._py_stamps), _ts_count,
                self.cache_stats["hit"], self.cache_stats["changed"],
                self.cache_stats["rg_served"],
            )
            # P1 (2026-08-12): an agent session that only queries CGI-routed
            # methods never builds RG, so in a gate-less repo the SSOT snapshot
            # never exists and every fresh process pays a full cold parse.  If
            # this build served ZERO files from the RG tier and the snapshot
            # file is absent, trigger one GraphBuilder build to create it.
            self._maybe_self_heal_rg_snapshot()
            # Release the RG snapshot payload: it holds the whole JSON while
            # the graph lives, and nothing reads it between builds (P3 Stage 2).
            # The payload re-loads lazily on the first disk-tier miss.  The two
            # ESC early-returns above release it too (P2, 2026-08-12) — a
            # cancelled build must not leak ~209MB for the indexer's lifetime.
            self._rg_cache = None
            self._rg_cache_mtime_ns = 0

    def invalidate(self) -> None:
        """Mark index as stale; it will be rebuilt on next access."""
        with self._lock:
            self._built = False
            self._nodes.clear()
            self._forward.clear()
            self._reverse.clear()
            self._bare_index.clear()
            self._file_nodes.clear()
            self._file_edges.clear()
            self._def_sources.clear()
            self._file_defs.clear()

    def invalidate_files(self, changed_paths: list[str]) -> None:
        """Incrementally re-index only the changed files instead of a full rebuild.

        Mirror of ``RepositoryGraphFacade.invalidate_files``: each path is
        repo-relative (a leading ``/`` is tolerated); files that no longer
        exist are removed entirely.  Node ownership follows the same
        first-definition-wins rule as :meth:`build` (sorted walk order ⇒ the
        lexicographically first defining file); if the owning file's
        definition disappears, the node is reassigned to the next defining
        file deterministically — no silent node loss, no full re-parse.

        Runs under ``_lock`` so concurrent readers (which also hold it) never
        observe a torn index.  A no-op when the index was never built (or was
        fully invalidated) — the next access performs a complete build()
        anyway.

        The three repo-wide passes are incremental (A4, 2026-08-12): dir
        ranks are recomputed only when a changed path sits in a never-seen
        directory, callee resolution only touches edges whose callee's node
        may have moved (symbols defined by a changed file, plus the fresh
        edges of re-indexed files), and the bare-name suffix index is updated
        by key delta instead of being rebuilt.  A single-file edit therefore
        costs O(changed file + edges touching its symbols) instead of
        O(repo).
        """
        with self._lock:
            if not self._built:
                return
            rels = [str(raw).strip().lstrip("/") for raw in changed_paths]
            # Re-apply build()'s walk admission — dir pruning AND the basename
            # suffix policy — so the incremental path indexes exactly what a
            # fresh build()'s os.walk would yield (B1, 2026-08-11; F1,
            # 2026-08-12).  Re-applying only the dir half made *.min.js
            # bundles appear in the graph on touch (write-history dependence).
            rels = [r for r in rels if r and _path_is_walk_admissible(r)]
            # Fresh dir ranks only when a changed path lives in a directory we
            # have never seen.  Rank order of known dirs never changes on file
            # edits, and _owner_rank falls back to "ranks last" for unknown
            # dirs, so skipping the dir-only walk for edits inside known dirs
            # is exactly equivalent to recomputing (missing dirs rank last
            # either way; only relative order of known dirs matters).
            if any(os.path.dirname(r) not in self._dir_ranks for r in rels):
                self._dir_ranks = self._compute_dir_ranks()
            bare_keys_before = set(self._forward) | set(self._reverse)
            affected_syms: set[str] = set()
            reindexed_rels: list[str] = []
            for raw in rels:
                rel = raw
                affected_syms |= self._remove_file_contributions(rel)
                abs_path = os.path.join(self._root, rel)
                if not os.path.isfile(abs_path):
                    continue
                # Route exactly as build() does: py files via _index_file,
                # ts/js only when the opt-in flag is on, and nothing else —
                # json/md/go/... are never indexed by build(), so parsing them
                # here would inject symbols a fresh build lacks (P1,
                # 2026-08-12; the old `else: _index_file` branch ast.parsed
                # any non-ts/js file — json is often valid Python syntax).
                try:
                    if abs_path.endswith(_TS_JS_EXTENSIONS):
                        if not self._multilang_callgraph_enabled():
                            continue
                        self._index_ts_file(Path(abs_path))
                    elif not abs_path.endswith(_PY_EXTENSIONS):
                        continue
                    else:
                        self._index_file(Path(abs_path))
                except SyntaxError:
                    logger.debug(
                        "call_graph: incremental skip unparseable %s", abs_path, exc_info=True,
                    )
                except Exception as e:
                    logger.debug("call_graph: incremental skip %s: %s", abs_path, e)
                reindexed_rels.append(rel)
            # Incremental callee resolution: only edges whose callee node may
            # have changed (symbols defined by a changed file — their node may
            # have moved, vanished or been re-registered at a new line) and
            # the fresh edges of re-indexed files (CallEdge starts with
            # callee_file/callee_line=None).  Everything else is untouched.
            for rel in reindexed_rels:
                affected_syms.update(self._file_defs.get(rel, ()))
            todo: list[CallEdge] = []
            seen: set[int] = set()
            for sym in affected_syms:
                for edge in self._reverse.get(sym, ()):
                    if id(edge) not in seen:
                        seen.add(id(edge))
                        todo.append(edge)
            for rel in reindexed_rels:
                for edge in self._file_edges.get(rel, ()):
                    if id(edge) not in seen:
                        seen.add(id(edge))
                        todo.append(edge)
            self._resolve_edges(todo)
            # Incremental bare-name suffix index: drop keys that vanished,
            # add keys that appeared (typical edit: a handful each way).
            bare_keys_after = set(self._forward) | set(self._reverse)
            self._update_bare_index(bare_keys_before, bare_keys_after)

    def _remove_file_contributions(self, rel: str) -> set[str]:
        """Drop every node and edge this file contributed, leaving the rest intact.

        Returns the set of symbols this file DEFINED (their ``_def_sources``
        entries and possibly their nodes changed, so edges pointing at them
        must be re-resolved by the caller — A4, 2026-08-12).
        """
        # Every symbol DEFINED here — owned or merely shadowed — loses its
        # _def_sources entry.  The owned-only cleanup used to leave ghosts:
        # rewriting a file to drop a definition kept (rel → stale line) in
        # _def_sources, and setdefault in _register_node then preserved the
        # stale line on re-registration; deleting the actual owner reassigned
        # the node to a line that no longer exists in the file.
        defs = set(self._file_defs.pop(rel, []))
        for sym in defs:
            srcs = self._def_sources.get(sym)
            if srcs is None:
                continue
            srcs.pop(rel, None)
            if not srcs:
                del self._def_sources[sym]
        for sym in self._file_nodes.pop(rel, []):
            node = self._nodes.get(sym)
            if node is not None and node.file == rel:
                # This file owned the node — drop it and reassign to the next
                # defining file.  The rank-based min is the exact full-rebuild
                # winner (first definition wins): _owner_rank mirrors build()'s
                # walk order (root files before subdirs, source tiers before
                # tests), which lexicographic min(srcs) diverges from.
                # (srcs may already be gone — the _file_defs cleanup above
                # deleted an emptied _def_sources entry — in which case the
                # node is dropped with no successor.)
                self._nodes.pop(sym, None)
                srcs = self._def_sources.get(sym)
                if srcs:
                    nxt = min(srcs, key=self._owner_rank)
                    ln, kd = srcs[nxt]
                    self._nodes[sym] = CallGraphNode(
                        symbol=sym, file=nxt, line=ln, kind=kd,
                    )
                    # Ownership moved — track it so a later removal of *nxt*
                    # still finds the node (see the _file_nodes pop above).
                    self._file_nodes.setdefault(nxt, []).append(sym)
        for edge in self._file_edges.pop(rel, []):
            # Identity-based removal: an equal-but-distinct CallEdge from
            # another file must survive (dataclass __eq__ would match it).
            fwd = self._forward.get(edge.caller_symbol)
            if fwd:
                filtered = [e for e in fwd if e is not edge]
                if filtered:
                    self._forward[edge.caller_symbol] = filtered
                else:
                    del self._forward[edge.caller_symbol]
            rev = self._reverse.get(edge.callee_symbol)
            if rev:
                filtered = [e for e in rev if e is not edge]
                if filtered:
                    self._reverse[edge.callee_symbol] = filtered
                else:
                    del self._reverse[edge.callee_symbol]
        return defs

    def _rebuild_bare_index(self) -> None:
        """(Re)build the bare-name suffix index from the current key universe.

        One pass over the final keys (post _resolve_callees / re-index),
        turning _lookup_edges' suffix fallback from a full index scan (O(n)
        per miss) into a dict lookup (O(k) on the matching keys).
        """
        self._bare_index = {}
        seen_keys: set[str] = set()
        for key in (*self._forward, *self._reverse):
            if key in seen_keys:
                # symbol is both a caller and a callee — register once, or
                # suffix-fallback lookups would double the returned edges
                continue
            seen_keys.add(key)
            self._bare_index.setdefault(key.split(".")[-1], []).append(key)

    def _update_bare_index(self, before: set[str], after: set[str]) -> None:
        """Apply a key delta to the bare-name suffix index (A4, 2026-08-12).

        ``before``/``after`` are the ``_forward | _reverse`` key universes at
        the start and end of an incremental update.  Vanished keys are
        dropped from their bare-name bucket (empty buckets are removed), new
        keys are appended — O(changed) instead of a full rebuild over every
        key.  Semantics are identical to ``_rebuild_bare_index``: a key
        appearing in both directions is still registered once, and bucket
        order (insertion order of the full rebuild) is preserved for keys
        that did not change.
        """
        for key in before - after:
            bucket = self._bare_index.get(key.split(".")[-1])
            if bucket is None:
                continue
            with contextlib.suppress(ValueError):
                bucket.remove(key)
            if not bucket:
                del self._bare_index[key.split(".")[-1]]
        for key in after - before:
            self._bare_index.setdefault(key.split(".")[-1], []).append(key)

    def _multilang_callgraph_enabled(self) -> bool:
        """Fresh read of the opt-in TS/JS indexing flag (mirrors build())."""
        try:
            from config import MULTILANG_CALLGRAPH as _ML_CG
        except Exception as e:
            logger.debug("call_graph: MULTILANG_CALLGRAPH unreadable — TS indexing off: %s", e)
            return False  # fail-open: TS indexing is opt-in, never a hard dep
        return bool(_ML_CG)

    def _compute_dir_ranks(self) -> dict[str, int]:
        """Map every rel dir to its pre-order index in build()'s walk order.

        Mirrors the descent of ``_walk_py_files`` exactly: os.walk's traversal
        IS a pre-order DFS, so pruning + sorting ``dirnames`` in place (the
        same predicate and key the shared walkers use) and enumerating
        ``(dirpath, dirname)`` pairs as they are yielded assigns the very
        indices the real walk visits. Dir-only — no file stat/read — so it is
        cheap enough to run at the start of every invalidate_files().
        """
        ranks: dict[str, int] = {"": 0}
        idx = 1
        for dirpath, dirnames, _filenames in os.walk(self._root):
            dirnames[:] = sorted(
                (d for d in dirnames if not _walk_should_skip_dir(d)),
                key=_walk_dir_sort_key,
            )
            for d in dirnames:
                ranks[os.path.relpath(os.path.join(dirpath, d), self._root)] = idx
                idx += 1
        return ranks

    def _owner_rank(self, rel: str) -> tuple[int, str]:
        """Sort key placing *rel* exactly where build()'s walk visits it.

        Files within a directory are visited in filename order after that
        directory's own pre-order position, so ``(dir_rank, filename)`` is the
        full order. A directory absent from ``_dir_ranks`` (created between
        ranks computation and use) ranks last — the safe direction: it can
        only lose ownership it never had, and the next build/invalidate
        recomputes ranks.
        """
        _d, _f = os.path.split(rel)
        return (self._dir_ranks.get(_d, 1 << 30), _f)

    def _register_node(self, symbol: str, rel: str, line: int, kind: str) -> None:
        """Record one definition site and claim node ownership when this is the first.

        First-definition-wins mirrors the walk order of :meth:`build`
        (tracked in ``_file_nodes``); every defining file is tracked in
        ``_def_sources`` so :meth:`invalidate_files` can reassign ownership
        when the current owner's definition disappears.

        Ownership is rank-based, not once-only: if an earlier-visited file
        re-defines a symbol whose node a later file took over (e.g. the owner
        was edited to drop the definition, ownership moved, and the original
        file is now edited back), the earlier file reclaims the node — exactly
        what a full rebuild would produce. ``_owner_rank`` is the single
        source of truth shared with ``_remove_file_contributions``'s
        reassignment, so incremental ownership can never diverge from build().
        """
        srcs = self._def_sources.setdefault(symbol, {})
        srcs.setdefault(rel, (line, kind))
        self._file_defs.setdefault(rel, []).append(symbol)
        node = self._nodes.get(symbol)
        if node is None or self._owner_rank(rel) < self._owner_rank(node.file):
            if node is not None:
                # Reclaim from the previous owner — drop its ownership record
                # so a later removal of that file does not find a ghost node.
                _prev = self._file_nodes.get(node.file)
                if _prev is not None:
                    try:
                        _prev.remove(symbol)
                    except ValueError:
                        # Invariant violation: the old owner's record should
                        # contain the symbol it owns.  Not worth failing the
                        # index over, but the ghost would corrupt a later
                        # removal, so make it visible.
                        logger.debug(
                            "call_graph: ownership record for %s missing in %s",
                            symbol, node.file,
                        )
                    if not _prev:
                        del self._file_nodes[node.file]
            self._nodes[symbol] = CallGraphNode(
                symbol=symbol, file=rel, line=line, kind=kind,
            )
            self._file_nodes.setdefault(rel, []).append(symbol)

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
        2. Suffix fallback (O(1) via the build-time bare-name index):
           ``execute_plan_canonical`` matches the index key
           ``OperationExecutor.execute_plan_canonical``.  Needed because
           _collect_calls stores callers under the qualified name
           (ClassName.method) but callers typically pass the bare method name.

        ``file_attr`` selects which CallEdge attribute
        (``'caller_file'`` / ``'callee_file'``) the optional *file_path*
        filter applies to.
        """
        edges = index.get(symbol, [])
        if not edges:
            # Bare-name index (M1) — keys whose last segment equals the bare
            # symbol ("a" vs "X.a"). Built once in build(), so the suffix
            # fallback no longer scans every index key on each miss.
            bare = symbol.rsplit(".", maxsplit=1)[-1]
            for key in self._bare_index.get(bare, ()):
                edges = edges + index.get(key, [])
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
        """Parse one Python file and inject it into the graph (compute path).

        Used by :meth:`invalidate_files` (and tests); :meth:`build` routes
        through :meth:`_process_file_cached` so unchanged files skip the
        parse entirely.  Extraction and injection are factored into
        :meth:`_extract_file` / :meth:`_inject_file_data`, so the cached path
        replays the same pure payload bit-for-bit.
        """
        extracted = self._extract_file(path)
        if extracted is None:
            return
        rel, payload = extracted
        self._inject_file_data(rel, payload)

    def _process_file_cached(self, path: Path) -> None:
        """Process one Python file, serving unchanged files from cache tiers (P2).

        Mirror of ``RepositoryGraph._process_file_cached``: unchanged (same
        ``mtime_ns`` + ``size``) files are injected from the process-wide
        ``_file_cache`` without re-parsing; a first build in a fresh process
        additionally serves unchanged files from RepositoryGraph's snapshot
        (``_rg_file_data``) before parsing.  The cached payload IS the pure
        per-file extraction, so injection is bit-for-bit identical to a cold
        parse (same ownership, same edge order) and the sorted walk order in
        :meth:`build` keeps the whole graph identical too.

        The walk stamp (``_py_stamps``) is recorded for every successfully
        processed file; served files count into ``cache_stats["hit"]`` and
        re-parsed ADMITTED files land in ``_fresh_parsed`` (the "changed"
        stat).  Cap-overflow computes are transient — neither cached nor
        counted (the RG snapshot is the convergence mechanism, P3 Stage 3).
        """
        try:
            st = os.stat(path)
        except OSError:
            logger.debug("call_graph: _process_file_cached: cannot stat %s", path)
            return
        rel = self._rel(path)
        key = (os.path.normpath(str(self._root)), os.path.abspath(str(path)))
        cached = _file_cache.get(key)
        if cached is not None:
            mtime_ns, size, payload = cached
            if mtime_ns == st.st_mtime_ns and size == st.st_size:
                self._py_stamps.append((rel, path, st))
                self.cache_stats["hit"] += 1
                self._inject_file_data(rel, payload)
                return
            _file_cache.pop(key, None)
        # P3 Stage 2: RepositoryGraph's snapshot is the SSOT — serve it FIRST,
        # before CGI's own disk tier.  RG's extraction carries the CGI-
        # convention fields (_rg_payload_to_cgi converts), so a fresh RG build
        # (release gate) makes the agent build parse-free.  Falls back to
        # CGI's own tiers when the RG snapshot is absent/stale/partial.
        payload = self._rg_file_data(rel, st)
        served_from_cache = payload is not None
        if served_from_cache:
            self.cache_stats["rg_served"] += 1
        if payload is None:
            extracted = self._extract_file(path)
            if extracted is None:
                return
            rel, payload = extracted
        # Store BOTH paths (compute and RG-snapshot-served) into the
        # in-process tier, admission-gated.  Served files count into ``hit``;
        # only computed ADMITTED files land in ``_fresh_parsed`` (the
        # "changed" stat).
        if len(_file_cache) >= _FILE_CACHE_MAX_ENTRIES:
            _gc_file_cache()
        admitted = _file_cache.admit(key, (st.st_mtime_ns, st.st_size, payload))
        self._py_stamps.append((rel, path, st))
        if served_from_cache:
            self.cache_stats["hit"] += 1
        elif admitted:
            self._fresh_parsed.append(rel)
        self._inject_file_data(rel, payload)

    def _load_rg_disk_snapshot(self) -> None:
        """Load RepositoryGraph's snapshot (once per JSON rewrite).

        Atomic rewrites mean a new mtime ⇒ reload; any problem (missing, corrupt, version
        mismatch) fails open to None and per-file lookups fall through to a
        fresh parse.  RG's snapshot is py-only by construction, so TS/JS
        files never reach this tier.

        A failed load pins the marker to the CURRENT mtime (not 0): the
        per-file RG tier (``_rg_file_data``) calls this once per file, and
        with a corrupt or version-mismatched snapshot the old 0-marker
        re-read + re-parsed the whole JSON for EVERY file (on asicode ~39MB
        — the same 300s+ hang F9 fixed on the RG side, 2026-08-12).
        Pinning still reloads a REWRITTEN snapshot (new mtime), and
        ``build()`` resets the marker to 0 after each build, so the next
        build retries a still-broken file once.
        """
        path = self._rg_cache_path
        try:
            st = os.stat(path)
        except OSError:
            self._rg_cache = None
            self._rg_cache_mtime_ns = 0
            return
        if st.st_mtime_ns == self._rg_cache_mtime_ns:
            return
        self._rg_cache = _load_rg_snapshot(path)
        self._rg_cache_mtime_ns = st.st_mtime_ns
    def _maybe_self_heal_rg_snapshot(self) -> None:
            """P1 (2026-08-12): create RG's snapshot when this build couldn't use it.

            The RG snapshot is the SSOT disk tier (P3 Stage 2/3).  It is written
            by ``RepositoryGraph`` builds with ``collect_imported_names=True``
            (release gate / ``GraphBuilder.build_repo_graph``), but an agent
            session that only queries CGI-routed methods (``get_callers`` /
            ``get_callees`` / ``get_related_symbols``) never builds RG — in a
            gate-less repo the snapshot never exists and EVERY fresh process pays
            a full cold parse (measured 0.87s -> 4.08s on asicode, 818 py files).

            When this build served ZERO files from the RG tier and the snapshot
            file does not exist, trigger one ``GraphBuilder`` build to create it
            (fail-open: any error leaves the CGI index as-is; the snapshot
            rewrite is hint-gated, so a warm/served build pays ~0 for it).  A
            present-but-stale snapshot is NOT healed here — tree edits are the
            normal incremental path and the next RG build refreshes it.
            """
            if self.cache_stats.get("rg_served", 0) > 0:
                return
            if os.path.exists(self._rg_cache_path):
                return  # exists but stale (tree changed) — RG build refreshes it
            # Lazy import: graph_builder -> repository_graph, and call_graph is
            # imported by tool_registry which graph_builder must not depend on.
            from external_llm.graph.graph_builder import GraphBuilder

            logger.warning(
                "call_graph: RG snapshot missing at %s — self-healing with one "
                "GraphBuilder build (subsequent cold builds serve from it)",
                self._rg_cache_path,
            )
            try:
                GraphBuilder(str(self._root)).build_repo_graph()
                self.cache_stats["rg_self_healed"] = 1
            except Exception as exc:  # fail-open: keep the already-built CGI index
                logger.debug("call_graph: RG snapshot self-heal failed: %s", exc)

    def _rg_file_data(self, rel: str, st: os.stat_result) -> Optional[dict]:
        """One file's CGI-converted payload from the RG snapshot, or None.

        Serves only files whose manifest stamp matches the CURRENT stat — the
        same staleness contract as the per-file disk tier, so a snapshot
        written from any earlier tree state is safe.  The RG payload is
        converted to CGI shape (``_rg_payload_to_cgi``) — a pure function, so
        serving is bit-for-bit identical to a cold parse of the same file.
        """
        self._load_rg_disk_snapshot()
        cache = self._rg_cache
        if cache is None:
            return None
        manifest = cache.get("manifest") or {}
        if manifest.get(rel) != [st.st_mtime_ns, st.st_size]:
            return None
        payload = (cache.get("files") or {}).get(rel)
        if payload is None or not isinstance(payload, dict):
            return None
        if not isinstance(payload.get("symbols"), list) or not isinstance(payload.get("calls"), list):
            logger.debug("call_graph: RG snapshot payload for %s unusable; falling back", rel)
            return None
        return _rg_payload_to_cgi(payload)

    def _extract_file(self, path: Path) -> Optional[tuple[str, dict]]:
        """Pure per-file extraction: parse → JSON-ready payload (P2).

        Returns ``(rel, payload)`` or None on the size gate / parse failure
        (mirrors build()'s silent-skip contract).  The payload holds ``defs``
        (``[symbol, lineno, kind]`` rows in BFS order) and ``calls`` (asdict
        CallEdge rows) — exactly the data the cache tiers store, so a
        cache-served build injects bit-for-bit what a fresh parse produced.
        """
        # Gate BEFORE the read: ast.parse holds ~155x the source size in
        # transient memory, so the read and the parse have to be refused
        # together. See _too_big_to_index.
        if _too_big_to_index(path, _MAX_PY_BYTES):
            return None
        # Route through the shared parse cache (P1, 2026-08-11): the structural
        # scanners and RepositoryGraph.extract_file parse the SAME files in the
        # same turn, and this worker walks the tree read-only (class_names is a
        # side table keyed by id(node); no AST mutation) — so one cached parse
        # serves every consumer. parse_ast decodes utf-8/replace (as this worker
        # already did) and returns None on read failure or SyntaxError (logged
        # at debug), matching build()'s silent-skip contract.
        tree = parse_cache.parse_ast(str(path))
        if tree is None:
            return None
        rel = self._rel(path)

        # Single BFS pass instead of three separate ast.walk()s (A2,
        # 2026-08-12): ast.walk is breadth-first, so a ClassDef is always
        # yielded BEFORE its body children — class_names is therefore fully
        # populated by the time the functions it names are reached.  Merging
        # the class-name pass, the def pass and the call pass cut warm-build
        # time ~42% (ast.walk was ~54% of build() under cProfile).  The order
        # of func_nodes is exactly the old walk order (BFS), so
        # first-definition-wins and call insertion order are unchanged.
        class_names: dict[int, str] = {}
        func_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_names[id(child)] = node.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_nodes.append(node)

        defs = self._collect_defs(func_nodes, class_names)
        calls = self._collect_calls(func_nodes, rel, class_names)
        return rel, {"defs": defs, "calls": [asdict(c) for c in calls]}

    def _inject_file_data(self, rel: str, payload: dict) -> None:
        """Replay one file's extraction into the graph (P2).

        Cache-serve == cold parse: the same ``_register_node`` ownership
        rules and the same edge insertion order as the historical
        collect-and-mutate path.  Edges are reconstructed FRESH from the
        JSON-ready dicts here — the graph's callee-resolution pass
        (``_resolve_edges``) mutates edge objects in place, so injecting
        cached instances would let a resolved build pollute the cache tiers,
        which hold the same dicts.
        """
        for symbol, line, kind in payload["defs"]:
            self._register_node(symbol, rel, line, kind)
        for d in payload["calls"]:
            edge = CallEdge(**d)
            self._forward.setdefault(edge.caller_symbol, []).append(edge)
            self._reverse.setdefault(edge.callee_symbol, []).append(edge)
            self._file_edges.setdefault(rel, []).append(edge)

    def _collect_defs(
        self,
        func_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef],
        class_names: dict[int, str],
    ) -> list[list]:
        """Extract one file's defs as ``[symbol, lineno, kind]`` rows (P2).

        Pure extraction (no graph mutation): the caller injects via
        :meth:`_inject_file_data`, so cache-served builds replay exactly what
        a fresh parse would have registered.  Order is the BFS walk order of
        *func_nodes* — the same order ``_register_node`` was historically
        called in, so ``_file_defs``/``_file_nodes`` stay identical.
        """
        defs: list[list] = []
        for node in func_nodes:
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
            defs.append([symbol, node.lineno, kind])
        return defs

    def _collect_calls(
        self,
        func_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef],
        rel: str,
        class_names: dict[int, str],
    ) -> list[CallEdge]:
        """Extract one file's call edges (P2).

        Pure extraction (no graph mutation) — the caller injects via
        :meth:`_inject_file_data`.  Dedup contract unchanged: per caller
        function, one edge per distinct ``callee_display``, in call order.
        """
        calls: list[CallEdge] = []
        for node in func_nodes:
            class_name = class_names.get(id(node))
            caller_sym = (
                f"{class_name}.{node.name}" if class_name else node.name
            )
            seen: set[str] = set()
            for child in _iter_calls(node):
                edge = self._parse_call(
                    child, caller_sym, rel, child.lineno, class_name
                )
                if edge and edge.callee_display not in seen:
                    seen.add(edge.callee_display)
                    calls.append(edge)
        return calls

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
        # Same gate as the Python path, ahead of the read for the same reason:
        # a minified dist bundle is read AND tree-sitter-parsed in full below.
        if _too_big_to_index(path, _MAX_TS_BYTES):
            return
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
            if fn.name:
                self._register_node(
                    fn.name, rel,
                    fn.meta.start_line if fn.meta else fn.start_line,
                    "async_function" if fn.is_async else "function",
                )

        # Register class methods
        for cls in module.classes:
            for method in cls.methods:
                symbol = f"{cls.name}.{method.name}"
                self._register_node(
                    symbol, rel,
                    method.meta.start_line if method.meta else method.start_line,
                    "method",
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
            self._file_edges.setdefault(rel, []).append(edge)

    def _resolve_callees(self) -> None:
        """Fill callee_file / callee_line using the collected node index."""
        self._resolve_edges(
            [e for edges in self._forward.values() for e in edges]
        )

    def _resolve_edges(self, edges: list[CallEdge]) -> None:
        """Resolve callee_file / callee_line for a subset of edges (A4).

        Full rebuilds call this over every forward edge; incremental updates
        pass only the edges whose callee node may have changed.
        """
        for edge in edges:
            node = self._nodes.get(edge.callee_symbol)
            if node:
                edge.callee_file = node.file
                edge.callee_line = node.line
            else:
                # Clear stale pointers: after an incremental update the
                # callee may have vanished (its defining file deleted or
                # rewritten).  Leaving the previously-resolved values in
                # place reports a definition that no longer exists —
                # fresh builds leave them None, so the two diverged.
                edge.callee_file = None
                edge.callee_line = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal utility
# ─────────────────────────────────────────────────────────────────────────────

def _rg_payload_to_cgi(payload: dict) -> dict:
    """Convert one RepositoryGraph per-file payload to a CGI payload (P3 Stage 2).

    RG's snapshot is the SSOT: it carries CGI-convention fields (``cgi_symbol``,
    ``caller_symbol``, ``caller_def_line``, ``is_async``, ``ast_depth``) so the
    conversion reproduces :meth:`CallGraphIndexer._extract_file` bit-for-bit —
    verified over every py file in the repo (818/818 files, exact defs AND
    call order parity, 2026-08-12):

    * **defs** — function/method symbols only, sorted ``(ast_depth, start_line)``
      = ``ast.walk`` BFS order (CGI collects func_nodes via ast.walk, so ALL
      depth-k defs precede any depth-(k+1) def; if/for nesting is invisible to
      qualnames, hence RG records the real AST depth).  Kind: direct class
      method → ``"method"`` (cgi_symbol != name), else async/function.
    * **calls** — edges with ``resolution == "fallback"`` are RG's legacy
      fallback for unsupported call forms (chained ``obj.m()()`` etc.); CGI
      emits NO edge for them, so they are dropped.  (The old pre-P2
      ``confidence == 1.0`` marker clause was removed, P2 2026-08-12: it
      matched zero edges in practice AND the schema version folds CallEdge's
      field signatures, so a pre-``resolution`` snapshot cannot load anyway —
      while the field default of 1.0 would silently swallow any future edge
      that omits it.)  Dedup is per FUNCTION NODE: key = (full qualname,
      caller_def_line) — same-qualname redefinitions (if/else branch functions)
      must not merge.  RG's call list is already in CGI's LIFO traversal order
      (GraphVisitor.generic_visit mirrors _iter_calls), so first-seen per
      distinct callee_display == CGI's keep.  Groups are emitted in defs (BFS)
      order.  ``call_args``/``is_mutating`` are carried over from RG (richer
      than CGI's always-empty; consumers get the same fields RG computed).
    """
    symbols = [
        s for s in payload.get("symbols", [])
        if s.get("kind") in ("function", "method")
    ]
    # ast.walk BFS order == (ast_depth, start_line); ast_depth is stored by RG.
    symbols.sort(key=lambda s: (s.get("ast_depth", 0), s.get("start_line", 0)))
    defs: list[list] = []
    fn_order: dict[tuple[str, int], int] = {}
    for i, s in enumerate(symbols):
        sym = s.get("cgi_symbol") or s.get("name")
        kind = (
            "method"
            if sym != s.get("name")
            else ("async_function" if s.get("is_async") else "function")
        )
        defs.append([sym, s.get("start_line"), kind])
        fn_order[(s.get("qualname"), s.get("start_line"))] = i

    # Group by function NODE (full qualname + def line): CGI dedups per
    # func_node, and same-qualname redefinitions must stay separate.
    groups: dict[tuple[str, int], list[dict]] = {}
    for c in payload.get("calls", []):
        if c.get("resolution") == "fallback":
            continue  # RG legacy fallback for unsupported forms — CGI has none
        q = c.get("caller") or c.get("caller_symbol")
        d = c.get("caller_def_line") or 0
        groups.setdefault((q, d), []).append(c)

    kept: list[dict] = []
    for (_q, _d), clist in sorted(
        groups.items(), key=lambda kv: fn_order.get((kv[0][0], kv[0][1]), 1 << 30)
    ):
        seen: set[str] = set()
        for c in clist:
            if c.get("callee_display") in seen:
                continue
            seen.add(c["callee_display"])
            kept.append(c)

    calls = [
        {
            "caller_symbol": c.get("caller_symbol") or c.get("caller"),
            "caller_file": c["file_path"],
            "caller_line": c["line"],
            "callee_symbol": c.get("callee_symbol"),
            "callee_display": c.get("callee_display"),
            "callee_file": None,
            "callee_line": None,
            "confidence": c.get("confidence", 1.0),
            "edge_kind": "calls",
            "call_args": c.get("call_args", []),
            "is_mutating": c.get("is_mutating", False),
        }
        for c in kept
    ]
    return {"defs": defs, "calls": calls}


def _iter_calls(func_node: ast.AST) -> Iterator[ast.Call]:
    """Yield Call nodes in *func_node*'s subtree, pruning nested function bodies.

    Nested ``def``/``async def`` statements are indexed as their own callers by
    ``CallGraphIndexer._collect_calls``, so calls inside them must not be
    attributed to the enclosing function (B1: nested-function misattribution).
    """
    todo: list[ast.AST] = [func_node]
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Call):
            yield node
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not func_node
        ):
            continue  # nested function — its body belongs to a separate caller
        todo.extend(ast.iter_child_nodes(node))


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
