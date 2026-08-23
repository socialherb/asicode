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
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.languages.base import build_line_index, line_at_offset

from ..analysis import parse_cache
from ..common.walk_policy import (
    _WALK_SKIP_FILE_SUFFIXES,
    _path_is_walk_admissible,
    _walk_should_skip_dir,
)
from ..languages import LanguageId, LanguageRegistry
from .models import ImportEdge, SymbolNode
from .root_cache import RootCache
from .structural_cache import (
    data_from_json,
    data_to_json,
    default_cache_path,
)
from .structural_cache import (
    load as _load_structural_cache,
)
from .structural_cache import (
    save as _save_structural_cache,
)

_logger = logging.getLogger(__name__)


# Process-wide per-file extraction cache for build().  Keyed by absolute path;
# value is ``(mtime_ns, size, extract_file payload)``.  build() re-parses ONLY
# files whose (mtime_ns, size) changed since the last extraction and re-injects
# the cached payload for everything else — a second build in the same process
# (rebuild after facade.invalidate(), another RepositoryGraph over the same
# root, the gate + agent sharing one process) drops from a full re-parse to a
# stat per file.  Same staleness contract as the structural gate's
# ``.cache/structural_graph_v1.json`` manifest (mtime_ns + size): a rewritten
# file that keeps an identical timestamp is the cache's only failure mode, and
# it is identical to the gate's.
#
# FIRST-BUILD DISK TIER (2026-08-11): before parsing, _process_file_cached
# also consults the structural gate's on-disk JSON
# (``.cache/structural_graph_v1.json`` — see structural_cache.py) when the
# gate ran before this process: a fresh process then reuses the gate's
# per-file extraction instead of re-parsing the whole repo (~10.9s → ~1.5s
# measured on asicode itself).  WRITE is shared (2026-08-11, pipeline
# integration): ``build(collect_imported_names=True)`` — the structural
# scanner gate's mode — additionally computes the per-file imported-name
# sets and rewrites the cache with the COMPLETE payload (files + manifest +
# imported_names), making RepositoryGraph the single producer of the JSON.
# The plain ``build()`` stays read-only and never computes names, so the app
# pays nothing it does not consume.  Entries are validated against the
# CURRENT stat before reuse, so a cache from any earlier tree state is safe.
#
# Cached payloads are SHARED across graph instances: they must be treated as
# immutable after caching (extract_file's tree-sitter end_line refinement is
# the only post-parse mutation and happens before the payload is stored).
#
# Bounded by _EXTRACT_CACHE_MAX_ENTRIES via ADMISSION CONTROL (2026-08-12):
# when full, the dead-file sweep frees slots for deleted sources and the
# INSERT site in _process_file_cached admits a new file only if a slot freed
# up — otherwise the new file is re-parsed and injected but NOT cached.  A
# repo with more files than the cap therefore keeps a STABLE first ``cap``
# entries (hit rate cap/N on every rebuild) instead of the whole-cache clear
# that previously thrashed to 0% hits each rebuild: the walk revisits every
# file each build, so a clear-and-refill cycle re-parses everything.  The
# cache is a speed optimization, never a correctness input.
#
# KEYED BY (repo_root, abs path), not abs path alone: the payload carries
# root-relative fields (SymbolNode.file_path / .module, CallEdge.file_path),
# so serving a nested root (repo vs repo/pkg) another root's extraction
# would contaminate those fields with the wrong-relative values — and hand
# back the very same SymbolNode objects (aliasing) for graphs that must
# disagree on them.
# RootCache (2026-08-12): root-partitioned fair admission — in multi-repo
# (webapp) processes each active root keeps ~cap/max_roots slots instead of
# starving once earlier repos filled the cap; single-root behavior is
# unchanged (quota == cap).
_EXTRACT_CACHE_MAX_ENTRIES = 2048
_extract_cache = RootCache(_EXTRACT_CACHE_MAX_ENTRIES)
# Dead-file sweep rate-limit for both caches now lives on the RootCache
# instance (``_gc_deficit``, see sweep_dead) under the cache's own lock
# (C1, 2026-08-12) — the module globals this replaced were read-modify-write
# races across webapp sessions.
# Process-wide memo of persisted snapshot manifest lengths, keyed by cache
# path (A5, 2026-08-12): the on-disk snapshot is a process-wide artifact
# (like _extract_cache), so a SECOND RepositoryGraph over the same repo in
# the same process must not re-load the multi-MB JSON just because its own
# counter started at 0.  Keyed by cache path so different repos (e.g. test
# tmpdirs) never share a memo.
_disk_manifest_lens: dict[Path, int] = {}
# Process-wide imported-name memo (A5, 2026-08-12): keyed like
# _extract_cache — (repo_root, abs path) tuples with the same
# (mtime_ns, size) staleness contract, bounded by _EXTRACT_CACHE_MAX_ENTRIES.
# Instance-local meant every facade rebuild (invalidate_files -> fresh
# RepositoryGraph) re-loaded the whole disk JSON just to serve names — the
# agent's edit loop pays that on every write.  C1 (2026-08-12): the cap is
# enforced at BOTH insert sites via _gc_names_cache (dead-entry sweep +
# refusal when still full) — never by clearing, which thrashed hits to 0%
# on every cap-sized build.  RootCache (2026-08-12): root-partitioned
# admission like _extract_cache.
_names_cache = RootCache(_EXTRACT_CACHE_MAX_ENTRIES)

# Per-file Python size gate, imported from the single source of truth
# (agent/config/thresholds.py — "never redefine these values in-place"): CGI
# skips giant generated .py files, and extract_file applies the same gate so
# RG doesn't parse them into the SHARED parse_cache where CGI never reuses
# them (F5, 2026-08-12).  Imported, not mirrored — a parallel hardcode drifted
# apart silently breaks CGI/RG parity on giant files.
_MAX_PY_BYTES = _cfg.lines.CALLGRAPH_PY_MAX_BYTES


def _extract_cache_key(root: str, file_path: str) -> tuple[str, str]:
    """Cache key: (normalized repo_root, abs path)."""
    return (os.path.normpath(root), os.path.abspath(file_path))


def _gc_extract_cache() -> None:
    """Reclaim slots from deleted sources so a full cache can admit live files.

    Admission control (2026-08-12): the cap is enforced by the INSERT site in
    :meth:`_process_file_cached` refusing new entries when full — never by
    clearing the whole cache (that thrashed to 0% hits every rebuild on a
    repo larger than the cap, because the walk revisits every file each
    build).  This sweep only frees entries whose source no longer exists,
    making room for new live files to be admitted.

    Rate-limited: a full sweep is O(cap) stat calls.  A sweep defers the next
    one by ``cap`` calls, keeping total sweep work ~O(N)/build instead of
    O(N*cap).  Deletions happen between builds, not mid-walk, so one sweep
    per cap-worth of rejections reclaims promptly without a per-insert storm.
    The sweep itself is delegated to :meth:`RootCache.sweep_dead`
    (root-partitioned, 2026-08-12): keys are ``(root, abs path)`` tuples, so
    the stat target is ``key[1]``.  The rate-limit deficit counter lives on
    the cache instance (``_gc_deficit``) under the cache's lock (C1,
    2026-08-12).
    """
    _extract_cache.sweep_dead(_EXTRACT_CACHE_MAX_ENTRIES)


def _gc_names_cache() -> None:
    """Reclaim slots from deleted sources so a full cache can admit live files.

    Mirror of :func:`_gc_extract_cache` (C1, 2026-08-12): ``_names_cache`` is
    a speed optimization only (names are always recomputable), but a full
    ``clear()`` at the cap thrashed hits to 0% the same way the extraction
    cache did — every cap-sized rebuild re-populated from zero and re-read
    the disk JSON each time.  Same admission contract: the cap is enforced
    at the INSERT sites (refuse new entries when still full after sweeping),
    never by clearing.  Keys are ``(repo_root, abs path)`` tuples like
    _extract_cache's (2026-08-12), so the sweep stats ``key[1]``
    (:meth:`RootCache.sweep_dead`).

    Rate-limited identically: a full sweep is O(cap) stat calls; a sweep
    defers the next one by ``cap`` calls, keeping total sweep work
    ~O(N)/build.  Deficit counter on the cache instance (``_gc_deficit``),
    under the cache's lock (C1, 2026-08-12).
    """
    _names_cache.sweep_dead(_EXTRACT_CACHE_MAX_ENTRIES)


def _dedupe_importers(entries: list[tuple[str, int]]) -> list[str]:
    """Merge reverse-import-index bucket entries into the legacy importer list.

    The legacy ``get_importers`` scanned ``import_edges`` in order, keeping
    each importer at its FIRST occurrence (edge order).  Bucket entries carry
    the edge's ordinal so a query can re-merge its match branches in exactly
    that order.  Sorts in place — callers pass freshly-built lists.
    """
    if not entries:
        return []
    entries.sort(key=lambda t: t[1])
    importers: list[str] = []
    seen: set = set()
    for importer, _ord in entries:
        if importer not in seen:
            seen.add(importer)
            importers.append(importer)
    return importers


def path_to_module(file_path: str, repo_root: str | None = None) -> str:
    """Convert a file path to a dotted Python module name — SINGLE SOURCE.

    The one place that turns a repo path into the module name used in import
    edges: ``pkg/mod.py`` → ``pkg.mod`` and ``pkg/__init__.py`` → ``pkg``
    (a package's module name is the package, matching what
    ``from pkg import X`` edges store — ``pkg.X``).  :class:`GraphVisitor`
    (forward edges) and :meth:`RepositoryGraph.get_importers` (reverse
    lookups) both derive their prefixes here so the two always agree (B1,
    2026-08-11 — previously ``get_importers`` computed ``pkg.__init__`` for
    ``pkg/__init__.py``, a prefix no edge stores, so importers of package
    re-exports were silently never found).

    *file_path* is repo-RELATIVE (GraphVisitor passes the relpath
    ``extract_file`` already computed) — ``os.path.relpath`` must NOT be run
    on it unconditionally: a relative path is resolved against the CWD, so a
    build from a cwd != repo_root produced module names like
    "....Users.x.y.pkg.m" (and, mixed with disk-cache payloads written from
    the repo root, a per-file module convention that disagreed within one
    graph).  Only absolute inputs are relativized against *repo_root*;
    relative ones are used as-is.
    """
    rel_path = file_path
    if os.path.isabs(rel_path):
        rel_path = os.path.relpath(rel_path, repo_root) if repo_root else rel_path
    if LanguageId.from_path(rel_path) is LanguageId.PYTHON:
        rel_path = rel_path[:-3]
    # Handle __init__ files
    if rel_path.endswith(("/__init__", "__init__")):
        rel_path = rel_path[:-9]
    # Replace path separators
    module = rel_path.replace("/", ".")
    # Remove leading dots
    if module.startswith("."):
        module = module[1:]
    return module


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
    # P3 Stage 1 (2026-08-12): canonical attribution fields mirroring
    # CallGraphIndexer._parse_call, so RG's per-file payload can eventually
    # serve CGI (single extraction / single snapshot).  ``callee`` above stays
    # the legacy bare-name field the analysis scanners query; these fields
    # carry the CGI-convention attribution (qualified self. callee_symbol,
    # dotted callee_display, call-form confidence).  Additive defaults keep
    # pre-P3 snapshots loadable: CallEdge(**d) without these keys still works.
    callee_symbol: str | None = None  # canonical callee (CGI convention)
    callee_display: str = ""  # dotted call-site form, e.g. "self.foo"
    confidence: float = 1.0  # call-form heuristic (0.9 / 0.85 / 0.5; 0.2 = unsupported form)
    # P2 (2026-08-12): explicit marker for unsupported call forms (chained
    # ``obj.m()()``, dynamic receivers) — CGI emits no edge for them, and the
    # SSOT conversion drops them by this field, NEVER by confidence value.
    resolution: str = ""  # "fallback" = unsupported form (confidence 0.2)
    # P3 Stage 2: canonical CALLER attribution (CGI convention).  ``caller``
    # above stays the legacy full-scope qualname (e.g. "Outer.inner" for a
    # nested function) the analysis scanners query; this field carries the
    # CGI-convention caller — direct class method "Class.method", bare name
    # otherwise — so RG's snapshot can serve CallGraphIndexer's forward edges
    # without re-deriving scope rules.  Additive default keeps pre-P3 snapshots
    # loadable.
    caller_symbol: str | None = None  # canonical caller (CGI convention)
    # P3 Stage 2: definition line of the caller function — disambiguates
    # same-qualname redefinitions (e.g. two ``ui_root`` functions in if/else
    # branches both resolve to qualname ``mount_ui.ui_root``; CGI dedups per
    # function NODE, so the SSOT conversion needs the def line to split them).
    caller_def_line: int | None = None


class _SymbolRangeIndex:
    """Sorted interval index for O(log n) non-Python caller attribution.

    ``_process_file_ripgrep`` previously scanned every function range for each
    call edge (O(calls x functions)); in a large TS/JS file with many top-level
    functions that linear scan dominated the extraction.  Ranges are sorted by
    start line ONCE per file; a query bisects to the ranges starting at or
    before the call line, then walks backwards bounded by a prefix max of end
    lines that terminates as soon as no earlier range can contain the line.
    Each query is O(log n + nesting depth) and answers exactly the old
    question: the name of the smallest-span function containing the line
    (innermost function wins).  Ties (identical spans — only possible for
    duplicate ranges, unreachable for real function definitions where
    containment is strict) resolve to the last-listed range.
    """

    __slots__ = ("_max_end_prefix", "_ranges", "_starts")

    def __init__(self, ranges: list[tuple[str, int, int]]) -> None:
        # Stable sort: ranges sharing a start line keep extraction order.
        self._ranges = sorted(ranges, key=lambda r: r[1])
        self._starts = [r[1] for r in self._ranges]
        prefix: list[int] = []
        best_end = -1
        for _name, _start, end in self._ranges:
            best_end = max(best_end, end)
            prefix.append(best_end)
        self._max_end_prefix = prefix

    def find(self, line: int) -> str:
        """Return the name of the smallest-span range containing *line*, or ""."""
        i = bisect_right(self._starts, line)
        best_name = ""
        best_span = -1
        for j in range(i - 1, -1, -1):
            if self._max_end_prefix[j] < line:
                break  # no range in 0..j reaches line -> none can contain it
            name, start, end = self._ranges[j]
            if end >= line:
                span = end - start
                if best_span < 0 or span < best_span:
                    best_name = name
                    best_span = span
        return best_name


class RepositoryGraph:
    """Repository-wide symbol graph."""

    def __init__(self, repo_root: str, cache_path: str | Path | None = None):
        self.repo_root = os.path.abspath(repo_root)
        self.symbols: dict[str, SymbolNode] = {}
        self.call_edges: list[CallEdge] = []
        self.import_edges: list[ImportEdge] = []
        self.file_symbols: dict[str, list[str]] = defaultdict(list)
        self._symbol_locations: dict[tuple[str, str], str] = {}  # (name, file) -> unique_id
        # Reverse call-edge indexes — lazily built by _ensure_call_index(),
        # dropped by _invalidate_call_index() on any call_edges mutation.
        #   _call_index:    field ("caller" | "callee") -> exact value -> edges
        #   _segment_index: field -> last dotted segment of value -> edges
        self._call_index: dict[str, dict[str, list[CallEdge]]] | None = None
        self._segment_index: dict[str, dict[str, list[CallEdge]]] | None = None
        # Reverse import-edge index — lazily built by _ensure_import_index(),
        # dropped by _invalidate_import_index() on any import_edges mutation.
        #   _import_index:  {"prefix" | "basename" | "resolved"} -> key ->
        #                   [(importer, edge ordinal), ...] — see
        #                   _ensure_import_index for the bucket semantics.
        self._import_index: dict | None = None
        # Build diagnostics — exposed for post-build inspection and log analysis.
        # Reset at the start of each build() call.
        self.build_exception_types: dict[str, int] = {}
        """Exception type name → count mapping from the last build, keyed by
        ``{ExceptionType}: {language or 'python'}`` so callers can distinguish
        Python parsing failures from tree-sitter or I/O errors."""
        # Structural-gate disk cache (read-only warm tier for the first build
        # in a fresh process; write tier under build(collect_imported_names=True)).
        # Loaded lazily by _load_disk_cache_snapshot() at the start of each
        # build(); reloaded only when the JSON file changed since the last load.
        # *cache_path* overrides the default ``{repo_root}/.cache/...`` location
        # (the structural-scanner gate passes its patched test seam here).
        self._cache_path: Path = default_cache_path(self.repo_root) if cache_path is None else Path(cache_path)
        self._disk_cache: dict | None = None
        self._disk_cache_mtime_ns: int = 0
        # Pipeline-integration state — meaningful only under
        # build(collect_imported_names=True) (the gate's mode); see build().
        self.imported_names: set[str] = set()
        """Every name imported or module-attr-read across the repo's .py files
        (the structural gate's cross-file dead-code suppression set)."""
        self.cache_stats: dict = {"hit": 0, "total": 0, "changed": 0, "parsed_uncapped": 0}
        """Per-build cache reuse counters — ``hit`` = files served from a cache
        tier without re-parsing, ``changed`` = files re-parsed (Python AND
        non-Python; py includes cap-overflow re-parses — the merge-preserving
        snapshot persists those too, P0 2026-08-12), ``parsed_uncapped`` = PY
        files re-parsed beyond the in-process admission cap (cap-pressure
        canary: non-zero means the repo outgrew ``_EXTRACT_CACHE_MAX_ENTRIES``),
        ``total`` = walked source files (Python + non-Python).  Invariant:
        ``hit + changed <= total`` (P1, 2026-08-11)."""
        self._py_stamps: list[tuple[str, str, os.stat_result]] = []
        """(rel, abs path, stat) for every walked .py file — build-tracked."""
        self._fresh_parsed: list[str] = []
        """rel paths of PY files re-parsed AND admitted this build.  Drives
        cache_stats["changed"] and the cache-rewrite hint (P0, 2026-08-11).
        Cap-overflow re-parses live in ``_fresh_parsed_uncapped`` — the
        merge-preserving snapshot persists them too, so they count into
        "changed" and fire the hint as well (P0, 2026-08-12).  Non-Python
        re-parses live in ``_fresh_parsed_nonpy`` — they never reach the
        snapshot, so they must not fire the rewrite hint (P1, 2026-08-11)."""
        self._fresh_parsed_uncapped: list[str] = []
        """rel paths of PY files re-parsed this build beyond the admission cap
        (refused by the in-process cache).  The merge-preserving on-disk
        snapshot persists their payloads anyway (via ``_pending_snapshot``), so
        they count into cache_stats["changed"] and fire the rewrite hint like
        admitted re-parses (P0, 2026-08-12).  Non-empty ⇒ the repo outgrew
        ``_EXTRACT_CACHE_MAX_ENTRIES``; the build logs a one-shot WARNING."""
        self._pending_snapshot: dict[str, tuple[int, int, dict]] = {}
        """rel → (mtime_ns, size, extraction payload) for PY files freshly
        parsed BEYOND the admission cap this build.  Build-local and transient:
        the in-process cache refuses them (memory cap), but the merge-preserving
        snapshot rewrite needs their payloads to persist coverage so the next
        build serves them from disk instead of re-parsing (P0, 2026-08-12).
        Re-initialized at the top of every :meth:`build`."""
        self._fresh_parsed_nonpy: list[str] = []
        """rel paths of non-Python files re-parsed AND admitted this build —
        counted into cache_stats["changed"] but NOT into _fresh_parsed (the
        persisted snapshot only ever contains .py files, so a non-Python
        re-parse must not fire the cache-rewrite hint; P1, 2026-08-11)."""
        self._nonpy_stamps: list[tuple[str, str, os.stat_result]] = []
        """(rel, abs path, stat) for every walked non-Python file — build-tracked
        like ``_py_stamps``; feeds cache_stats["total"] so the hit/change
        counters stay coherent across languages (P1, 2026-08-11)."""
        self._rel_names: dict[str, set] = {}
        """rel → non-empty imported names (only files with names)."""

    # Walk admission is SHARED with the agent walkers (B2', 2026-08-11):
    # build()'s os.walk and _file_is_walk_admissible use
    # ``_walk_should_skip_dir`` / ``_WALK_SKIP_FILE_SUFFIXES`` from
    # external_llm/common/walk_policy — the single source of truth the
    # CGI / symbol_search / vulture / RAG walkers use too.  The former private
    # _SKIP_DIRS / _SKIP_FILE_SUFFIXES were a second copy that drifted
    # (vendor/ was RG-only, .egg-info/ shared-only, .min.js RG-only); keep
    # any future policy change in walk_policy.py, never here (F5).

    def build(self, collect_imported_names: bool = False) -> None:
        """Scan the repository and populate the graph.

        Idempotent: every call starts from an EMPTY graph (``symbols``,
        ``call_edges``, ``import_edges``, ``file_symbols``,
        ``_symbol_locations`` are reset), so re-calling ``build()`` on the
        same instance never duplicates edges and drops state of files deleted
        since the previous build (P1, 2026-08-11).  Only the process-wide
        extraction caches survive across builds, by design.

        With *collect_imported_names* (the structural-scanner gate's mode) the
        build ALSO unions every file's cross-file import names into
        :attr:`imported_names` and — when at least one file had to be
        re-parsed, or the walked file count no longer matches the persisted
        manifest (delete/rename-only changes re-parse nothing) — rewrites the
        on-disk cache
        (``.cache/structural_graph_v1.json``) with the COMPLETE payload
        (files + manifest + imported_names), so the next run in any process
        reuses this build instead of re-parsing.  The plain mode stays
        read-only and never computes names: the app pays nothing it does not
        consume.  Either way the walk stamps (``_py_stamps`` →
        :attr:`py_files`) are recorded in BOTH modes — the structural tool
        unions the uncapped py list into its cross-file-ref input even when
        it did not request imported names (2026-08-11).  Either way the walk
        order is the single injection order, so
        cache-served builds are bit-for-bit identical to cold builds by
        construction.

        The walk is deliberately UNLIMITED — no file cap.  The structural
        gate's *scanner* file lists truncate at ``SCAN_FILE_CAP`` (4000,
        external_llm/analysis/scan_walk.py — the single scan-walk source
        shared with the agent tool), so in a repo above that size
        this graph is strictly MORE complete than what the scanners run on;
        the direction of completeness is intentional and documented there.
        In ``collect_imported_names`` mode the name pass also sizes the
        shared parse cache to the uncapped walked py count before parsing
        (parse_cache module docstring, P2 2026-08-11).
        """
        self.build_exception_types = {}
        # Every build starts from an EMPTY graph (P1, 2026-08-11): a re-call
        # must not double-append call/import edges (or file_symbols ids — the
        # same accumulation pattern), and must not keep symbols of files
        # deleted since the previous build.  Re-calling build() is therefore
        # idempotent: N calls produce exactly the graph of one call.  The
        # process-wide extraction caches (_extract_cache/_names_cache) are
        # deliberately NOT reset — they are (mtime_ns, size)-keyed and serve
        # every build.
        self.symbols = {}
        self.call_edges = []
        self.import_edges = []
        self.file_symbols = defaultdict(list)
        self._symbol_locations = {}
        # The lazily-built reverse indexes reference the previous build's edge
        # objects — drop them so the next query rebuilds from the fresh lists.
        self._invalidate_call_index()
        self._invalidate_import_index()
        # Pipeline-integration state (see __init__ for the fields).
        self.imported_names = set()
        self.cache_stats = {"hit": 0, "total": 0, "changed": 0, "parsed_uncapped": 0}
        self._py_stamps = []
        self._nonpy_stamps = []
        self._fresh_parsed = []
        self._fresh_parsed_nonpy = []
        self._fresh_parsed_uncapped = []
        self._pending_snapshot = {}
        self._rel_names = {}
        # The structural gate's on-disk snapshot loads LAZILY (P0, 2026-08-11):
        # only the per-file disk tier (_disk_file_data / _imported_names_for)
        # and _save_cache_snapshot's payload reuse force a load.  A warm build
        # whose in-process caches cover every file never touches the JSON — on
        # asicode that was a 25MB load + ~170MB of transient allocations per
        # rebuild, for data nothing would read (0.205s → 0.023s).
        for root, dirs, files in os.walk(self.repo_root):
            # Skip hidden, venv, and vendor directories
            dirs[:] = sorted(d for d in dirs if not _walk_should_skip_dir(d))
            for file in sorted(files):
                file_path = os.path.join(root, file)
                if file.endswith(_WALK_SKIP_FILE_SUFFIXES):
                    continue
                lang = LanguageId.from_path(file_path)
                if lang == LanguageId.UNKNOWN:
                    continue
                try:
                    if lang == LanguageId.PYTHON:
                        self._process_file_cached(file_path, track=collect_imported_names)
                    elif LanguageRegistry.instance().supports_structured_ops(file_path):
                        self._process_file_ripgrep(file_path, track=collect_imported_names)
                except Exception as _exc:
                    _rel = os.path.relpath(file_path, self.repo_root)
                    _etype = type(_exc).__name__
                    _tag = f"{_etype}: {'python' if lang == LanguageId.PYTHON else lang.value}"
                    self.build_exception_types[_tag] = self.build_exception_types.get(_tag, 0) + 1
                    _logger.debug(
                        "Graph build skipped %s: %s — %s",
                        _rel,
                        _etype,
                        _exc,
                    )
                    continue
        if collect_imported_names:
            # Size the shared parse cache to the walked py set BEFORE the name
            # pass: the walk is UNCAPPED (unlike the scanner lists' SCAN_FILE_CAP)
            # and the pass parses every walked py file through the cache, so a
            # default-sized cache would thrash on any repo bigger than it.  The
            # same process's cross-file-ref pass re-parses importers right after
            # and finds the cache already sized (ensure_capacity is monotonic —
            # see parse_cache module docstring, P2 2026-08-11).
            parse_cache.ensure_capacity(len(self._py_stamps))
            for rel, path, st in self._py_stamps:
                names = self._imported_names_for(rel, path, st)
                self.imported_names |= names
                # Record EVERY file (empty sets included): the saved section
                # must be self-describing, or a warm run re-computes files
                # whose names are simply empty (they look "unknown").
                self._rel_names[rel] = names
            # Maybe-rewrite the on-disk cache when (a) an admitted file was
            # re-parsed, (a2) a cap-overflow file was re-parsed (merge-preserving
            # snapshot persists those too), or (b) the walked file count differs
            # from the persisted manifest — a delete-only/rename-only change
            # re-parses NOTHING but leaves dead entries in the JSON (which would
            # otherwise grow forever), and a failed/missing disk load (manifest
            # empty) self-heals here.  This is only a "possibly changed" HINT:
            # _save_cache_snapshot does the authoritative check — it skips the
            # atomic rewrite when the freshly-built manifest is identical to
            # the persisted one (P0, 2026-08-11).  The manifest LENGTH memo
            # (not the multi-MB JSON) plus an existence probe decide the hint:
            # a warm no-change build fires nothing and never loads the JSON.
            if (
                self._fresh_parsed
                or self._fresh_parsed_uncapped
                or len(self._py_stamps) != _disk_manifest_lens.get(self._cache_path, 0)
                or not os.path.exists(self._cache_path)
            ):
                self._save_cache_snapshot()
        self.cache_stats["total"] = len(self._py_stamps) + len(self._nonpy_stamps)
        self.cache_stats["changed"] = (
            len(self._fresh_parsed) + len(self._fresh_parsed_uncapped) + len(self._fresh_parsed_nonpy)
        )
        self.cache_stats["parsed_uncapped"] = len(self._fresh_parsed_uncapped)
        if self._fresh_parsed_uncapped:
            _logger.warning(
                "RepositoryGraph build: %d of %d walked .py files re-parsed beyond the "
                "%d-entry in-process extraction cap (root %s at %d/%d entries; "
                "merge-preserving snapshot persists them; raise "
                "_EXTRACT_CACHE_MAX_ENTRIES to cache them in-process too)",
                len(self._fresh_parsed_uncapped),
                len(self._py_stamps),
                _EXTRACT_CACHE_MAX_ENTRIES,
                self.repo_root,
                _extract_cache.count(self.repo_root),
                _extract_cache.quota(),
            )
        # Release the on-disk snapshot payload: it holds the whole gate JSON
        # (measured ~32% of resident memory on asicode — 24MB file, 33.5MB
        # retained) while the graph lives, and nothing reads it between
        # builds.  The manifest LENGTH memo survives (a single int): it lets
        # the next build's rewrite hint and _save_cache_snapshot's fast-skip
        # decide without re-loading the JSON (P0, 2026-08-11).  The payload
        # re-loads lazily on the first disk-tier miss.
        self._disk_cache = None
        self._disk_cache_mtime_ns = 0

    def _process_file(self, file_path: str) -> None:
        """Parse a single Python file and add its symbols/edges to the graph.

        Thin wrapper: extraction (parse + tree-sitter end-line refinement) is
        factored into :meth:`extract_file` so the structural-scanner gate can
        cache the per-file result and re-inject it on a later incremental
        build via :meth:`_inject_file_data`.
        """
        data = self.extract_file(file_path)
        if data is None:
            return
        self._inject_file_data(os.path.relpath(file_path, self.repo_root), data)

    def _load_disk_cache_snapshot(self) -> None:
        """Load the structural gate's on-disk cache (once per JSON rewrite).

        The gate rewrites ``.cache/structural_graph_v1.json`` atomically
        (tmp + os.replace), so a new mtime means a new snapshot — reload only
        then.  Any problem (missing file, corrupt JSON, version mismatch)
        fails open to None and per-file lookups fall through to
        :meth:`extract_file`.

        A failed load pins the marker to the CURRENT mtime (not 0): the
        per-file disk tier (``_disk_file_data`` / ``_imported_names_for``)
        calls this once per file, and with a corrupt or version-mismatched
        snapshot the old 0-marker re-read + re-parsed the whole 25MB JSON
        for EVERY file (~0.2s x 2 x N on asicode — the version bump from
        new dataclass fields turned the gate into a 300s+ hang, 2026-08-12).
        Pinning still reloads a REWRITTEN snapshot (new mtime — the gate
        rewrites at build end), and ``build()`` resets the marker to 0 after
        each build, so the next build retries a still-broken file once.
        """
        path = self._cache_path
        try:
            st = os.stat(path)
        except OSError:
            self._disk_cache = None
            self._disk_cache_mtime_ns = 0
            _disk_manifest_lens[self._cache_path] = 0
            return
        if st.st_mtime_ns == self._disk_cache_mtime_ns:
            return
        self._disk_cache = _load_structural_cache(path)
        self._disk_cache_mtime_ns = st.st_mtime_ns
        if self._disk_cache is not None:
            _disk_manifest_lens[self._cache_path] = len(self._disk_cache.get("manifest") or {})
        else:
            # Load failed (corrupt/version-mismatch): drop the stale memo so
            # build()'s rewrite hint and _save_cache_snapshot's fast-skip do
            # not compare against an old length (minor 3, 2026-08-12).
            _disk_manifest_lens[self._cache_path] = 0

    def _ensure_disk_cache(self) -> None:
        """Lazily load the on-disk snapshot on first use within a build (P0).

        ``build()`` no longer pre-loads the snapshot: a warm process whose
        in-process ``_extract_cache``/``_names_cache`` cover every file never
        touches the JSON (a 25MB load + ~170MB of transient allocations on
        asicode).  Only the per-file disk tier (:meth:`_disk_file_data`,
        :meth:`_imported_names_for`) and :meth:`_save_cache_snapshot` (payload
        reuse + authoritative manifest compare) call this.  The mtime guard
        inside :meth:`_load_disk_cache_snapshot` makes repeat calls cheap.
        """
        if self._disk_cache is None:
            self._load_disk_cache_snapshot()

    def _disk_file_data(self, rel: str, st: os.stat_result) -> dict | None:
        """One file's extraction from the gate's disk cache, or None.

        Serves only files whose manifest stamp (mtime_ns + size) matches the
        CURRENT stat — the same staleness contract the gate itself uses, so a
        cache written from any earlier tree state is safe.  A payload that
        fails dataclass reconstruction falls back to None: the file is then
        re-parsed, never skipped.
        """
        self._ensure_disk_cache()
        cache = self._disk_cache
        if cache is None:
            return None
        manifest = cache.get("manifest") or {}
        if manifest.get(rel) != [st.st_mtime_ns, st.st_size]:
            return None
        payload = (cache.get("files") or {}).get(rel)
        if payload is None:
            return None
        try:
            return data_from_json(payload)
        except (TypeError, ValueError, KeyError) as exc:
            _logger.debug("disk cache payload for %s unusable (%s); re-parsing", rel, exc)
            return None

    def _process_file_cached(self, file_path: str, track: bool = False, use_disk_tier: bool = True) -> None:
        """Process one Python file, serving unchanged files from the process cache.

        Unchanged (same ``mtime_ns`` + ``size``) files are injected from
        ``_extract_cache`` without re-parsing; only changed/new files go
        through :meth:`extract_file`.  A first build in a fresh process
        additionally serves unchanged files from the structural gate's
        on-disk cache (``_disk_file_data``) before parsing.  The cached
        payload IS the pure extraction result, so injection is bit-for-bit
        identical to a cold parse (same unique_ids, same edge objects) and
        the sorted walk order in :meth:`build` keeps call_edges/import_edges
        order identical too.

        The walk stamp (``_py_stamps`` → :attr:`py_files`) is recorded for
        EVERY caller: plain ``build()`` needs it too, because the structural
        tool unions the uncapped py list into its cross-file-ref input
        (2026-08-11).  *track* (build's pipeline-integration mode) gates only
        the bookkeeping counters: files served from either cache tier count
        into ``cache_stats["hit"]``, re-parsed ADMITTED files land in
        ``_fresh_parsed`` and cap-overflow re-parses in
        ``_fresh_parsed_uncapped`` (both count as "changed" and fire the
        cache-rewrite hint — the merge-preserving snapshot persists both,
        P0 2026-08-12).  Untracked callers (``reparse_file``) therefore still
        maintain the stamp, but touch none of the counters.

        *use_disk_tier* disables the on-disk snapshot tier (P0-1, 2026-08-12).
        The incremental path (:meth:`reparse_files`) passes False: a just-
        reparsed file is definitionally stamp-mismatched against the disk
        manifest, so loading the whole snapshot JSON for it is pure waste
        (~0.93s stall + ~209MB resident on asicode) — parse the file instead
        (~5ms).  An ALREADY-loaded ``_disk_cache`` is still consulted (the
        ``self._disk_cache is not None`` arm): skipping it would drop a
        cost-free hit once the snapshot is in memory.
        """
        try:
            st = os.stat(file_path)
        except OSError:
            _logger.debug("_process_file_cached: cannot stat %s", file_path)
            return
        rel = os.path.relpath(file_path, self.repo_root)
        key = _extract_cache_key(self.repo_root, file_path)
        cached = _extract_cache.get(key)
        if cached is not None:
            mtime_ns, size, data = cached
            if mtime_ns == st.st_mtime_ns and size == st.st_size:
                self._py_stamps.append((rel, file_path, st))
                if track:
                    self.cache_stats["hit"] += 1
                self._inject_file_data(rel, data)
                return
            _extract_cache.pop(key, None)
        # First-build disk tier (structural gate JSON, read-only).
        data = self._disk_file_data(rel, st) if (use_disk_tier or self._disk_cache is not None) else None
        served_from_cache = data is not None
        if data is None:
            data = self.extract_file(file_path)
            if data is None:
                return
        self._py_stamps.append((rel, file_path, st))
        if len(_extract_cache) >= _EXTRACT_CACHE_MAX_ENTRIES:
            _gc_extract_cache()
        admitted = _extract_cache.admit(key, (st.st_mtime_ns, st.st_size, data))
        if not admitted and track:
            # Refused at the cap: keep the payload build-locally so the
            # merge-preserving snapshot rewrite persists this file's coverage
            # (the next build serves it from disk; P0, 2026-08-12).
            self._pending_snapshot[rel] = (st.st_mtime_ns, st.st_size, data)
        if track:
            if served_from_cache:
                self.cache_stats["hit"] += 1
            elif admitted:
                # Count as "changed" files whose content the snapshot will
                # actually hold.  Cap-overflow re-parses are persisted too
                # (merge-preserving rewrite), so they count as well and land
                # in _fresh_parsed_uncapped for the cap-pressure warning.
                self._fresh_parsed.append(rel)
            else:
                self._fresh_parsed_uncapped.append(rel)
        self._inject_file_data(rel, data)

    def _imported_names_for(self, rel: str, abs_path: str, st: os.stat_result) -> set:
        """One file's cross-file import names: in-process → disk → compute.

        Mirrors :meth:`_process_file_cached`'s tier order with the same
        staleness contract (``mtime_ns`` + ``size`` against the CURRENT
        stat).  The disk tier reads the cache's ``imported_names`` section
        only when its manifest stamp matches, so names written from any
        earlier tree state are never reused; a missing section (older
        format) or an unusable value falls through to compute.  Per-file
        skip contract: any compute failure yields an empty set, logged
        (the gate's historical ``except Exception → set()``).
        """
        key = _extract_cache_key(self.repo_root, abs_path)
        cached = _names_cache.get(key)
        if cached is not None:
            mtime_ns, size, names = cached
            if mtime_ns == st.st_mtime_ns and size == st.st_size:
                return names
            _names_cache.pop(key, None)
        self._ensure_disk_cache()
        disk = self._disk_cache
        if disk is not None:
            try:
                manifest = disk.get("manifest") or {}
                if manifest.get(rel) == [st.st_mtime_ns, st.st_size]:
                    names = (disk.get("imported_names") or {}).get(rel)
                    if names is not None:
                        names_set = set(names)
                        if len(_names_cache) >= _EXTRACT_CACHE_MAX_ENTRIES:
                            _gc_names_cache()
                        _names_cache.admit(key, (st.st_mtime_ns, st.st_size, names_set))
                        return names_set
            except TypeError:
                _logger.debug("disk imported_names for %s unusable; recomputing", rel)
        try:
            from external_llm.analysis.cross_file_refs import (
                extract_imported_names_for_file,
            )

            names_set = set(extract_imported_names_for_file(abs_path))
        except Exception as exc:
            _logger.debug("imported names for %s failed (%s); treating as empty", rel, exc)
            names_set = set()
        if len(_names_cache) >= _EXTRACT_CACHE_MAX_ENTRIES:
            _gc_names_cache()
        _names_cache.admit(key, (st.st_mtime_ns, st.st_size, names_set))
        return names_set

    def _save_cache_snapshot(self) -> None:
        """Persist the COMPLETE cache (files + manifest + imported_names).

        Called from :meth:`build` with ``collect_imported_names=True`` as a
        "possibly changed" hint (re-parse OR file-count drift).  Authoritative:
        when the freshly-built manifest is identical to the persisted one the
        rewrite is SKIPPED — re-serializing a byte-identical payload every
        build would be pure waste (P0, 2026-08-11).

        Merge-preserving (P0, 2026-08-12): the manifest is NOT rebuilt from the
        in-process cache alone.  Files served or parsed this build are
        refreshed from ``_extract_cache`` / ``_pending_snapshot``; files the
        walk visited but that are in neither (unchanged, beyond the admission
        cap or evicted) are carried over VERBATIM from the loaded disk payload
        when their stamp still matches — so coverage converges to the full
        walked set instead of the cap, and beyond-cap files stop re-parsing on
        every build.  Rels missing from the walk are dropped (delete pruning).
        Payloads are the verbatim disk payloads where still stamp-valid
        (bit-for-bit identity with what the next reader will inject), else
        re-serialized fresh extractions.  Best-effort — a failed write never
        fails the build (the next run rebuilds; fail-open).

        P0 fast-skip: when nothing re-parsed (admitted OR cap-overflow) and the
        walked py count matches the persisted manifest length, the snapshot
        cannot have changed — return WITHOUT loading the multi-MB JSON.  Any
        re-parse, count drift or missing JSON falls through to the
        authoritative compare below (which then loads the JSON for payload
        reuse).
        """
        if (
            self._disk_cache is None
            and not self._fresh_parsed
            and not self._fresh_parsed_uncapped
            and len(self._py_stamps) == _disk_manifest_lens.get(self._cache_path, 0)
            and os.path.exists(self._cache_path)
        ):
            return
        self._ensure_disk_cache()
        manifest: dict[str, list[int]] = {}
        files: dict[str, dict] = {}
        names: dict[str, list[str]] = {}
        disk = self._disk_cache or {}
        disk_manifest = disk.get("manifest") or {}
        disk_files = disk.get("files") or {}
        disk_names = disk.get("imported_names") or {}
        for rel, path, st in self._py_stamps:
            stamp = [st.st_mtime_ns, st.st_size]
            entry = _extract_cache.get(_extract_cache_key(self.repo_root, path))
            if entry is not None:
                # Admitted (or cache-validated) this build — refresh the entry.
                mtime_ns, size, data = entry
                payload = None
                if disk_manifest.get(rel) == [mtime_ns, size]:
                    payload = disk_files.get(rel)
                files[rel] = payload if payload is not None else data_to_json(data)
                manifest[rel] = stamp
                if rel in self._rel_names:
                    names[rel] = sorted(self._rel_names[rel])
                continue
            pending = self._pending_snapshot.get(rel)
            if pending is not None and pending[0] == st.st_mtime_ns and pending[1] == st.st_size:
                # Freshly parsed beyond the admission cap — persist it so the
                # next build serves it from disk (P0, 2026-08-12).
                files[rel] = data_to_json(pending[2])
                manifest[rel] = stamp
                if rel in self._rel_names:
                    names[rel] = sorted(self._rel_names[rel])
                continue
            if disk_manifest.get(rel) == stamp and rel in disk_files:
                # Unchanged but not in the in-process cache (beyond the cap or
                # evicted) — carry the disk payload over verbatim instead of
                # dropping the coverage (P0, 2026-08-12).
                files[rel] = disk_files[rel]
                manifest[rel] = stamp
                if rel in disk_names:
                    names[rel] = disk_names[rel]
                continue
            # Unreachable for walked rels (each is either cached, pending, or
            # stamp-valid on disk); defensive: record NOTHING — manifest must
            # mirror files so the snapshot stays self-describing.
        # Skip the atomic rewrite when the freshly-built manifest is identical
        # to the persisted one.  On a capped repo the merge-preserving carryover
        # makes the manifest converge to the full walked set, so this equality
        # is the steady state there — only real re-parses reach this point and
        # the (byte-identical) rewrite is skipped.  An equal manifest (same
        # keys, same stamps) implies identical files/names too — payloads and
        # imported-name sets are stamp/content-derived — so the skip is safe
        # (P0, 2026-08-11).
        if manifest == disk_manifest:
            return
        try:
            _save_structural_cache(self._cache_path, manifest, files, names)
            _disk_manifest_lens[self._cache_path] = len(manifest)
        except Exception as exc:
            _logger.debug("structural cache save failed (%s): %s", self._cache_path, exc)

    def extract_file(self, file_path: str) -> dict | None:
        """Parse ONE Python file → ``{"symbols": [...], "calls": [...], "imports": [...]}``.

        Pure extraction with no graph mutation: the caller decides where the
        result goes (straight into a build, or into a persistent cache for a
        later incremental build).  ``symbols`` are already end-line-refined
        via tree-sitter, exactly matching what ``_process_file`` used to
        produce — injecting the result reproduces a full ``build()``
        bit-for-bit.  Returns ``None`` on any parse failure (mirrors
        ``_process_file``'s silent-skip contract).
        """
        # Same per-file size gate as CallGraphIndexer._index_file
        # (CALLGRAPH_PY_MAX_BYTES = 1 MiB): CGI deliberately skips giant
        # generated .py files, and a symmetric gate here keeps RG from parsing
        # them into the SHARED parse_cache where CGI never reuses them — pure
        # memory loss (F5, 2026-08-12).  The value is imported from the agent
        # thresholds (single source of truth), not mirrored.
        try:
            if os.path.getsize(file_path) > _MAX_PY_BYTES:
                return None
        except OSError:
            _logger.debug("extract_file: cannot stat %s", file_path, exc_info=True)
            return None
        # Shared parse cache (P1, 2026-08-11): this parse is byte-identical to
        # CallGraphIndexer._index_file's and to every structural scanner's over
        # the same file, and GraphVisitor walks read-only (no AST mutation), so
        # one cached parse serves every consumer in the same turn. read_and_parse
        # resolves ONE stat key (mtime_ns, size) for both values, so the source
        # string the tree was parsed from is guaranteed to be the same version
        # of the file (calling read_source + parse_ast separately could observe
        # two different versions if the file changed in between). Decode policy
        # is utf-8/replace (matching _index_file) rather than strict — non-UTF-8
        # .py decodes lossily instead of silently skipping, unifying the two
        # Python workers.
        content, tree = parse_cache.read_and_parse(file_path)
        if tree is None or content is None:
            return None

        relative_path = os.path.relpath(file_path, self.repo_root)
        visitor = GraphVisitor(relative_path, self.repo_root)
        visitor.visit(tree)

        # Refine end_line using tree-sitter when available (the historical
        # _process_file refinement filtered symbols by file_path — applying it
        # to THIS file's symbols only is identical).
        symbols = visitor.symbols
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
                    ts_end_map: dict[str, int] = {}
                    for sym_name, _kind, _start, end_line in ts_symbols:
                        existing = ts_end_map.get(sym_name)
                        if existing is None or end_line > existing:
                            ts_end_map[sym_name] = end_line
                    for symbol in symbols:
                        ts_end = ts_end_map.get(symbol.name)
                        if ts_end is not None and ts_end > symbol.end_line:
                            symbol.end_line = ts_end
        except Exception:
            _logger.debug("extract_file tree-sitter refinement failed for %s", relative_path)

        return {"symbols": symbols, "calls": visitor.calls, "imports": visitor.imports}

    def _inject_file_data(self, relative_path: str, data: dict) -> None:
        """Merge one file's extraction result into the graph.

        The exact inverse of what ``_process_file`` used to do inline, so a
        graph assembled from cached ``extract_file`` results is identical to a
        full ``build()`` (same unique_ids, same edge objects).
        """
        for symbol in data["symbols"]:
            unique_id = f"{relative_path}:{symbol.qualname}"
            self.symbols[unique_id] = symbol
            self.file_symbols[relative_path].append(unique_id)
            self._symbol_locations[(symbol.qualname, relative_path)] = unique_id

        # Add call edges
        for call in data["calls"]:
            self.call_edges.append(call)

        # Add import edges
        for imp in data["imports"]:
            self.import_edges.append(imp)

    def _extract_non_python(self, file_path: str, content: str) -> dict | None:
        """Pure extraction of one non-Python file → ``{"symbols", "calls", "imports"}``.

        The lang-agnostic analogue of :meth:`extract_file`: parse-only, no graph
        mutation, so the result can be cached in ``_extract_cache`` (same key, same
        admission control, same staleness contract as Python) and later injected via
        :meth:`_inject_file_data` (P2, 2026-08-11).  Returns ``None`` when the file
        has no language provider (silently skipped upstream).

        Semantics are bit-for-bit identical to the pre-P2 inline write path:

        * tree-sitter is tried first for precise end_line + call/import edges, and
          gracefully falls through to regex on any failure (missing grammar,
          unsupported language, parse error) so one bad file never aborts a build;
        * intra-file dedup keeps the first symbol per ``qualname`` (the old
          ``unique_id in self.symbols`` check, narrowed to this file); and
        * regex import edges dedup by ``imported`` module path within this file.

        The caller-attribution step (tree-sitter call edges) is performed here at
        extraction time via :class:`_SymbolRangeIndex`, so a cached payload already
        carries the resolved ``caller`` names — no re-attribution on injection.
        """

        provider = LanguageRegistry.instance().get(file_path)
        if provider is None:
            return None

        relative_path = os.path.relpath(file_path, self.repo_root)
        lang_value = provider.language_id().value

        symbols: list[SymbolNode] = []
        calls: list[CallEdge] = []
        imports: list[ImportEdge] = []
        seen_qualnames: set[str] = set()

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
            from ..languages.tree_sitter_utils import (
                parse_to_tree as _ts_parse,
            )

            if _ts_available():
                # Parse ONCE and share the tree across all four extractions
                # (P5, 2026-08-11): parse_to_tree's memo is bypassed for
                # sources above _MAX_CACHED_SOURCE_CHARS, so large files used
                # to be parsed up to 4x per cold extraction.
                tree = _ts_parse(content, lang_value)
                ts_symbols = _ts_find_all(content, lang_value, tree=tree)
                if ts_symbols:
                    # Track symbol line ranges for caller attribution
                    _sym_ranges: list[tuple[str, int, int]] = []
                    for sym_name, kind, start_line, end_line in ts_symbols:
                        qualname = sym_name
                        if qualname not in seen_qualnames:
                            seen_qualnames.add(qualname)
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
                            symbols.append(node)
                        # Track EVERY function's range for caller attribution,
                        # dedup'd or not (P3, 2026-08-11): a later same-named
                        # function (e.g. a second ``render`` method) must still
                        # capture the calls inside ITS span — its range is
                        # tighter than the first duplicate's, so min-span
                        # resolution picks it.  The symbol itself stays dedup'd
                        # (first qualname wins, legacy contract).
                        if kind == "function":
                            _sym_ranges.append((sym_name, start_line, end_line))

                    # Extract call edges — attribute each call to the smallest
                    # enclosing function through a sorted interval index (P1):
                    # O(log n + nesting depth) per call instead of a linear
                    # scan over every function range in the file.
                    ts_calls = _ts_extract_calls(content, lang_value, tree=tree)
                    if ts_calls:
                        _sym_index = _SymbolRangeIndex(_sym_ranges)
                        for callee_name, call_line in ts_calls:
                            caller_name = _sym_index.find(call_line)
                            calls.append(
                                CallEdge(
                                    caller=caller_name,
                                    callee=callee_name,
                                    file_path=relative_path,
                                    line=call_line,
                                )
                            )

                    # Extract import edges
                    ts_imports = _ts_extract_imports(content, lang_value, tree=tree)
                    for module_path, _import_line in ts_imports:
                        imports.append(
                            ImportEdge(
                                importer=relative_path,
                                imported=module_path,
                                import_type="import",
                            )
                        )

                    _ts_ok = True
        except Exception:
            _logger.debug(
                "_extract_non_python tree-sitter extraction failed for %s "
                "(lang=%s) — falling through to regex fallback",
                relative_path,
                lang_value,
            )

        if _ts_ok:
            return {"symbols": symbols, "calls": calls, "imports": imports}

        # Fallback: regex-based extraction (end_line approximate)
        nl = build_line_index(content)
        for sp in provider.get_symbol_patterns("any"):
            # Replace {name} with a capture group to find all definitions
            pat = sp.regex.replace(r"{name}", r"(\w+)")
            for m in re.finditer(pat, content, re.MULTILINE):
                sym_name = m.group(1)
                lineno = line_at_offset(nl, m.start())
                qualname = sym_name
                if qualname in seen_qualnames:
                    continue
                seen_qualnames.add(qualname)
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
                symbols.append(node)

        # Regex-based import extraction fallback: when tree-sitter was unavailable
        # or failed, extract at least basic import edges so get_importers() has
        # some data for cross-file dead-code analysis and dependency tracking.
        _import_regexes: list[tuple[str, str]] = []
        if lang_value in {"javascript", "typescript"}:
            _import_regexes = [
                # import ... from 'module'
                (
                    r"""import\s+(?:\{[^}]*\}|\*\s+as\s+\w+|\w+(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))?)\s+from\s+['"]([^'"]+)['"]""",
                    "js_import",
                ),
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

        seen_imports: set[str] = set()
        for _pat, _itype in _import_regexes:
            for _m in re.finditer(_pat, content, re.MULTILINE):
                _module_path = _m.group(1)
                # Normalize: strip leading/trailing quotes and whitespace
                _module_path = _module_path.strip("\"'")
                if not _module_path:
                    continue
                # Deduplicate by (importer, imported) pair within this file
                if _module_path in seen_imports:
                    continue
                seen_imports.add(_module_path)
                imports.append(
                    ImportEdge(
                        importer=relative_path,
                        imported=_module_path,
                        import_type=_itype,
                    )
                )

        return {"symbols": symbols, "calls": calls, "imports": imports}

    def _process_file_ripgrep(self, file_path: str, track: bool = False) -> None:
        """Extract symbols from a non-Python file, serving unchanged files from cache.

        Mirrors :meth:`_process_file_cached`'s two-tier order (in-process
        ``_extract_cache`` → parse via :meth:`_extract_non_python`) with the SAME
        staleness contract (``mtime_ns`` + ``size`` against the CURRENT stat) and
        the SAME admission control, so a non-Python file parsed once is injected
        bit-for-bit identically on every rebuild without re-reading or re-parsing
        (P2, 2026-08-11).  The cache is language-agnostic: the payload is a plain
        ``{"symbols", "calls", "imports"}`` dict identical in shape to Python's
        :meth:`extract_file` result, so :meth:`_inject_file_data` consumes both.
        Cap-overflow re-parses follow the same contract as the Python path — never
        persisted, never counted as "changed".

        Ordering note (P2, 2026-08-11): stat → cache lookup → READ ONLY ON MISS,
        like the Python path — a cache-hit build never reads the file bytes from
        disk (on asicode that was 1.25MB of non-Python source per warm rebuild;
        multi-hundred-MB on TS/JS-heavy repos).

        *track* mirrors the Python path's bookkeeping so ``build()`` reports uniform
        cache_stats regardless of language (a non-Python file served from cache
        counts as a "hit"; an ADMITTED re-parse counts as "changed").  The walk
        stamp (``_nonpy_stamps``) is recorded for EVERY caller, like ``_py_stamps``
        — it feeds ``cache_stats["total"]`` (P1, 2026-08-11).  Non-build callers
        (``reparse_file``) pass ``track=False`` to touch no counters.
        """
        try:
            st = os.stat(file_path)
        except OSError:
            _logger.debug("_process_file_ripgrep: cannot stat %s", file_path)
            return
        rel = os.path.relpath(file_path, self.repo_root)
        key = _extract_cache_key(self.repo_root, file_path)
        cached = _extract_cache.get(key)
        if cached is not None:
            mtime_ns, size, data = cached
            if mtime_ns == st.st_mtime_ns and size == st.st_size:
                self._nonpy_stamps.append((rel, file_path, st))
                if track:
                    self.cache_stats["hit"] += 1
                self._inject_file_data(rel, data)
                return
            _extract_cache.pop(key, None)

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            _logger.debug("_process_file_ripgrep: cannot read %s", file_path)
            return

        data = self._extract_non_python(file_path, content)
        if data is None:
            return
        self._nonpy_stamps.append((rel, file_path, st))
        if len(_extract_cache) >= _EXTRACT_CACHE_MAX_ENTRIES:
            _gc_extract_cache()
        admitted = _extract_cache.admit(key, (st.st_mtime_ns, st.st_size, data))
        if track and admitted:
            # Count as "changed" ONLY files admitted to the cache (mirroring
            # the Python path's P0 contract): a cap-overflow file is re-parsed
            # every build but never persisted, so counting it would skew
            # cache_stats["changed"] (P1, 2026-08-11).  Non-Python re-parses
            # land in _fresh_parsed_nonpy — the snapshot is py-only, so they
            # must not fire the cache-rewrite hint.
            self._fresh_parsed_nonpy.append(rel)
        self._inject_file_data(rel, data)

    def get_symbol(
        self,
        name: str,
        file_path: str | None = None,
        prefer_files: list[str] | None = None,
    ) -> SymbolNode | None:
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
        resolution retry without ``file_path``.

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
        is_qualname = "." in name
        candidates: list[SymbolNode] = [
            s for s in self.symbols.values() if (s.qualname == name if is_qualname else s.name == name)
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
                    symbol.file_path.endswith("/" + file_path) or symbol.file_path.endswith(os.sep + file_path)
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
            _test_patterns = ("/test", "_test", "/tests/", "test_", "/fixtures/")

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

    def _invalidate_call_index(self) -> None:
        """Drop the lazily-built reverse indexes after any call_edges mutation."""
        self._call_index = None
        self._segment_index = None

    def _ensure_call_index(self) -> None:
        """Build reverse indexes over ``self.call_edges`` once (lazy).

        Two dicts per field (``'caller'`` / ``'callee'``):
        - exact index: full field value -> edges
        - segment index: last dotted segment of the field value -> edges

        The segment index serves the suffix-fallback query: the legacy
        predicate ``val.endswith(f".{method}") or val == method`` is exactly
        equivalent to "last dotted segment of val == method", so a query for a
        bare method name needs only the one bucket, not a full scan.
        """
        if self._call_index is not None:
            return
        call_index: dict[str, dict[str, list[CallEdge]]] = {"caller": {}, "callee": {}}
        segment_index: dict[str, dict[str, list[CallEdge]]] = {"caller": {}, "callee": {}}
        for edge in self.call_edges:
            for fname in ("caller", "callee"):
                val = getattr(edge, fname)
                # Exact index keeps empty values — the legacy exact path matched
                # them for a "" query.  Segment index skips empties (legacy
                # suffix path did too).
                call_index[fname].setdefault(val, []).append(edge)
                if not val:
                    continue
                segment_index[fname].setdefault(val.rsplit(".", 1)[-1], []).append(edge)
        self._call_index = call_index
        self._segment_index = segment_index

    def _invalidate_import_index(self) -> None:
        """Drop the lazily-built reverse import index after an import_edges mutation."""
        self._import_index = None

    def _ensure_import_index(self) -> None:
        """Build reverse indexes over ``self.import_edges`` once (lazy).

        Three dicts, each key → ``[(importer, edge ordinal), ...]`` in edge
        order — the ordinal lets a query re-merge branches in the exact order
        the legacy linear scan produced:

        - ``prefix``: every dotted ancestor prefix of each ``imported`` value.
          The Python match is ``imported == prefix or imported.startswith(
          prefix + ".")``, so one dict lookup returns exactly the matching
          edges — at the cost of one bucket entry per ancestor (measured
          ~2.5x edges on asicode, dotted depth ≤ 6).
        - ``basename``: the first dotted segment of each ``imported`` value —
          serves the relative-form fallback (``from .operation_models import
          X`` stores ``operation_models.X``), whose match condition is
          ``imported == basename or imported.startswith(basename + ".")``,
          exactly equivalent to "first segment == basename".  The legacy
          same-directory constraint depends on the query and is applied at
          query time.
        - ``resolved``: normpath-resolved ``imported`` (non-dot imports are
          root-relative, dot imports resolve against the importer's
          directory) — serves the non-Python path-like match
          (``resolved in (splitext(candidate)[0], candidate)``).  Built over
          ALL edges because the legacy non-Python scan examined every edge.
        """
        if self._import_index is not None:
            return
        prefix: dict[str, list[tuple[str, int]]] = {}
        basename: dict[str, list[tuple[str, int]]] = {}
        resolved: dict[str, list[tuple[str, int]]] = {}
        for ord_, edge in enumerate(self.import_edges):
            imp = edge.imported or ""
            if not imp:
                # Legacy skipped empty imported values in every branch.
                continue
            entry = (edge.importer, ord_)
            parts = imp.split(".")
            for i in range(1, len(parts) + 1):
                prefix.setdefault(".".join(parts[:i]), []).append(entry)
            basename.setdefault(parts[0], []).append(entry)
            if imp.startswith("."):
                _r = os.path.normpath(os.path.join(os.path.dirname(edge.importer or ""), imp))
            else:
                _r = os.path.normpath(imp)
            resolved.setdefault(_r, []).append(entry)
        self._import_index = {"prefix": prefix, "basename": basename, "resolved": resolved}

    def _edges_by_symbol_field(self, field: str, symbol_name: str) -> list[CallEdge]:
        """Return call edges where *field* (``'callee'`` or ``'caller'``)
        matches *symbol_name*.

        Matching strategy:
        1. exact match first (avoids over-matching symbols that share the
           same method name across different classes).
        2. fallback to method-name suffix match.

        Dedup by (caller, callee, file_path, line).
        """
        self._ensure_call_index()
        assert self._call_index is not None
        assert self._segment_index is not None
        exact = self._call_index[field].get(symbol_name)
        if exact:
            # Copy — callers must be able to mutate the returned list without
            # corrupting the index (legacy path returned a fresh list each call).
            return list(exact)

        parts = symbol_name.split(".")
        method = parts[-1] if parts else symbol_name

        edges = self._segment_index[field].get(method)
        if not edges:
            return []
        result: list[CallEdge] = []
        seen: set[tuple[str, str, str, int]] = set()

        for edge in edges:
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
        It is converted to a dotted module prefix (``external_llm.agent.foo``)
        via the canonical :func:`path_to_module` — a package init maps to the
        package itself (``pkg/__init__.py`` → ``pkg``), so importers of
        re-exports defined in it are found — then all ImportEdge entries
        whose ``imported`` starts with that prefix are collected and their
        ``importer`` file paths returned (deduped).

        Backed by :meth:`_ensure_import_index` (a lazy reverse index, dropped
        on any ``import_edges`` mutation): a query is one dict lookup per
        match branch instead of a full ``import_edges`` scan, with the legacy
        semantics preserved exactly — edge order, first-wins importer dedup,
        and the same-directory constraint on relative-form matches.
        """
        if not file_path:
            return []
        self._ensure_import_index()
        assert self._import_index is not None
        if LanguageId.from_path(file_path) is not LanguageId.PYTHON:
            # Non-Python (TS/JS/...) import edges store module paths, not
            # dotted names: imported="../string_utils" from
            # "__tests__/x.test.ts".  Resolve relative to the importer's
            # directory and match against the extensionless candidate path.
            _cand_noext = os.path.splitext(file_path)[0]
            _resolved = self._import_index["resolved"]
            _entries = list(_resolved.get(_cand_noext, ()))
            if file_path != _cand_noext:
                _entries += list(_resolved.get(file_path, ()))
            return _dedupe_importers(_entries)
        # Convert "a/b/c.py" → "a.b.c" via the canonical path→module helper —
        # "pkg/__init__.py" → "pkg" (NOT "pkg.__init__"), matching how the
        # graph stores `from pkg import X` edges (B1, 2026-08-11).
        _module_prefix = path_to_module(file_path, self.repo_root)
        # basename fallback: graph builder uses relative import without absolute path
        # Handle stored as "module_name.Symbol" form.
        # e.g. imported="operation_models.X" (from .operation_models import X)
        _module_basename = _module_prefix.rsplit(".", 1)[-1]  # "operation_models"
        _basename_differs = _module_basename != _module_prefix
        _idx = self._import_index
        _entries = list(_idx["prefix"].get(_module_prefix, ()))
        if _basename_differs:
            # Relative-form matches are only valid when the importer shares
            # the candidate's directory — the graph stores them without the
            # package prefix ("operation_models.X"), so a same-named module
            # in another package must not match.
            _cand_dir = os.path.dirname(file_path)
            _entries += [
                entry
                for entry in _idx["basename"].get(_module_basename, ())
                if os.path.dirname(entry[0] or "") == _cand_dir
            ]
        return _dedupe_importers(_entries)

    def get_symbols_in_file(self, file_path: str) -> list[SymbolNode]:
        """Return all symbols defined in the given file."""
        symbol_ids = self.file_symbols.get(file_path, [])
        return [self.symbols[sid] for sid in symbol_ids if sid in self.symbols]

    @property
    def py_files(self) -> list[str]:
        """Repo-relative paths of every walked .py file — UNCAPPED.

        Populated by EVERY :meth:`build` — plain mode and
        ``collect_imported_names=True`` alike: the structural tool unions
        this list into its cross-file-ref input even when it did not request
        imported names (2026-08-11).  Kept live by :meth:`remove_file` /
        :meth:`reparse_file` (the facade's incremental path), so long-lived
        processes never serve a stale walk.
        ``build()`` walks without a file cap (see build()), unlike the
        structural scanners' ``SCAN_FILE_CAP`` (external_llm/analysis/
        scan_walk.py), so this list is strictly more complete than any capped
        scan list.  The structural gate unions it into its cross-file-ref
        computation (scripts/check_structural_scanners.py) so references that
        live only in files beyond the cap still suppress dead-code candidates
        — closing the truncation soundness gap (2026-08-11).
        """
        return [rel for rel, _path, _st in self._py_stamps]

    def remove_file(self, rel_path: str) -> None:
        """Remove all symbols, call edges, and import edges for a file.

        Single-file entry point; delegates to the batched :meth:`_remove_files`
        (P3, 2026-08-11). ``remove_file`` was historically called once per
        changed path, and each call rebuilt ``call_edges``/``import_edges``/
        ``_symbol_locations`` (tens of thousands of entries on a real repo) from
        scratch — O(N paths x M edges). Callers with a set of paths should use
        :meth:`_remove_files` directly.
        """
        self._remove_files({rel_path})

    def _remove_files(self, rel_paths: set[str]) -> None:
        """Remove all symbols, call edges, and import edges for a SET of files.

        One pass over each large list (``call_edges``, ``import_edges``,
        ``_symbol_locations``, the walk stamps) for the whole set, plus a single
        index invalidation — vs ``len(rel_paths)`` passes when :meth:`remove_file`
        is called per path (P3, 2026-08-11). Membership-tested against a set, so
        the per-list filter is O(M) regardless of how many paths are removed.
        """
        if not rel_paths:
            return
        # Drop the walk stamps too: reparse_file/remove_file must keep
        # py_files (and the non-Python walk set) live in long-lived processes
        # (the facade's incremental path), or an edited file lingers — and
        # reparse would duplicate it — until the next full build.
        self._py_stamps = [t for t in self._py_stamps if t[0] not in rel_paths]
        self._nonpy_stamps = [t for t in self._nonpy_stamps if t[0] not in rel_paths]
        # Remove symbols (per-file file_symbols bucket; symbols dict keyed by id)
        for rel_path in rel_paths:
            for sid in self.file_symbols.pop(rel_path, []):
                self.symbols.pop(sid, None)
        # Remove _symbol_locations entries whose file is in the set
        self._symbol_locations = {k: v for k, v in self._symbol_locations.items() if k[1] not in rel_paths}
        # Remove call edges from these files (one pass, set membership)
        self.call_edges = [e for e in self.call_edges if e.file_path not in rel_paths]
        # Remove import edges from these files (one pass, set membership)
        self.import_edges = [e for e in self.import_edges if e.importer not in rel_paths]
        # The reverse indexes reference the removed edges — drop them ONCE.
        self._invalidate_call_index()
        self._invalidate_import_index()

    def _file_is_walk_admissible(self, abs_path: str) -> bool:
        """True iff ``build()``'s os.walk would have visited this file (B1).

        Delegates to the shared
        :func:`~external_llm.common.walk_policy._path_is_walk_admissible`
        predicate — directory pruning (hidden / ``venv*`` / ``site-packages`` /
        ``.egg-info`` / ``vendor``) AND the basename suffix policy
        (:data:`~external_llm.common.walk_policy._WALK_SKIP_FILE_SUFFIXES`) —
        so this admission cannot drift from the agent walker family.

        Language routing (:data:`LanguageId` / ``supports_structured_ops``) is
        intentionally NOT checked here: :meth:`reparse_file` routes by language
        exactly as ``build()`` does, so it is not part of this admission.
        """
        return _path_is_walk_admissible(os.path.relpath(abs_path, self.repo_root))

    def reparse_file(self, abs_path: str) -> None:
        """Re-parse a single file: remove old data, then re-process.

        Single-file entry point; delegates to the batched
        :meth:`reparse_files` (P3, 2026-08-11).

        Args:
            abs_path: Absolute path to the file.
        """
        self.reparse_files([abs_path])

    def reparse_files(self, abs_paths: list[str]) -> None:
        """Re-parse a SET of files: remove old data once, then re-process each.

        Batched companion to :meth:`reparse_file`: a single :meth:`_remove_files`
        pass over all targets, then one re-process per admissible survivor.
        Per-file ``remove_file``/``_process_file_*`` would rebuild the edge lists
        for every path (O(N paths x M edges)) and re-warm the extraction cache
        entry one path at a time — the batch drops that to O(M) + O(N) parses
        (P3, 2026-08-11).

        Goes through the language-appropriate cached processor
        (:meth:`_process_file_cached` for Python, :meth:`_process_file_ripgrep`
        otherwise) so the process-wide extraction cache is refreshed too — a
        subsequent full ``build()`` serves these files from cache instead of
        re-parsing them (P2, 2026-08-11: non-Python files now also invalidate
        and repopulate the cache on incremental reparse).

        Every target is removed first (drops stale injections and the state of
        files that moved *into* a walk-pruned directory); a path the ``build()``
        walk would prune is then NOT re-processed, so an incremental reparse and
        a full ``build()`` converge to the same graph (B1, 2026-08-11).
        """
        rels = {os.path.relpath(p, self.repo_root): p for p in abs_paths}
        self._remove_files(set(rels))
        for abs_path in rels.values():
            if not self._file_is_walk_admissible(abs_path):
                continue
            lang = LanguageId.from_path(abs_path)
            if lang == LanguageId.PYTHON:
                # use_disk_tier=False: a just-reparsed file can never match the
                # disk manifest stamp, so don't load the whole snapshot JSON
                # for it (P0-1, 2026-08-12 — ~0.93s + ~209MB saved per write).
                self._process_file_cached(abs_path, use_disk_tier=False)
            elif LanguageRegistry.instance().supports_structured_ops(abs_path):
                self._process_file_ripgrep(abs_path)


# ── Mutating-call detection ─────────────────────────────────────────────────
# A call is "mutating" (has write/side-effect semantics) when:
# 1. The return value is discarded (call used as statement — ast.Expr parent node).
# 2. The method name is a known Python data-model mutator (__setitem__, etc.).
# 3. The callee is a conventional mutating verb with a state-store receiver.
#
# Signals 1+2 are structurally derived from the AST and language spec.
# Signal 3 is a heuristic fallback for conventions not captured by 1+2.

# Python data-model mutating methods (language spec — exact, not heuristic).
_MUTATING_DUNDER = frozenset(
    {
        "__setitem__",
        "__delitem__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__ilshift__",
        "__irshift__",
        "__iand__",
        "__ixor__",
        "__ior__",
        "__setattr__",
        "__delattr__",
        "__set_name__",
    }
)


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

    bare = callee.rsplit(".", maxsplit=1)[-1].lower()

    # Signal 2: Python data-model mutating methods (exact match, language spec)
    if bare in _MUTATING_DUNDER:
        return True

    # Signal 3: conventional naming patterns — receiver.method(...) where
    # method is a mutating verb and receiver is a stateful object.
    # This is a heuristic fallback, not a structural guarantee.
    if isinstance(node.func, ast.Attribute):
        method = node.func.attr.lower()
        _mutating_methods = {
            "append",
            "extend",
            "insert",
            "pop",
            "remove",
            "clear",
            "add",
            "discard",
            "update",
            "difference_update",
            "symmetric_difference_update",
            "intersection_update",
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
        self.current_class: str | None = None
        self._in_function: int = 0  # nesting depth — guards module-level constant detection
        self._parent_stack: list[ast.AST] = []
        # P3 Stage 2: real AST nesting depth of the node currently being
        # visited (module=0, its body=1, ...).  Recorded on SymbolNode as
        # ast_depth so the CGI SSOT conversion can sort defs in ast.walk BFS
        # order — qualname dotted-depth cannot see if/for nesting levels.
        self._ast_depth: int = 0
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
        # Parallel kind map for _scope_stack entries: True = class scope,
        # False = function scope.  Lets visit_Call decide whether the CURRENT
        # function is a DIRECT method of an enclosing class — the condition
        # under which CGI qualifies ``self.foo()`` → ``ClassName.foo`` (a
        # nested function inside a method is NOT a direct method; P3 Stage 1).
        self._scope_is_class: list[bool] = []
        # P3 Stage 2: qualnames of functions whose decorator_list is being
        # visited — decorator calls sit OUTSIDE the function's line range, so
        # _get_current_symbol(line) misses them; this stack supplies the caller.
        self._decorator_stack: list[str] = []
        # P3 Stage 2: definition lines parallel to _scope_stack — lets
        # visit_Call record caller_def_line to disambiguate same-qualname
        # function redefinitions (if/else branches) for the CGI SSOT split.
        self._scope_def_line: list[int] = []
        # Compute module name from file path relative to repo root
        self.module = self._path_to_module(file_path)

    def visit(self, node: ast.AST) -> None:
        """Override visit to track parent nodes and AST depth."""
        self._parent_stack.append(node)
        self._ast_depth += 1
        try:
            super().visit(node)
        finally:
            self._ast_depth -= 1
            self._parent_stack.pop()

    def generic_visit(self, node: ast.AST) -> None:
        """Visit children in REVERSE order (P3 Stage 2).

        ``CallGraphIndexer._iter_calls`` walks each function's subtree with a
        LIFO stack: children are extended in field order and popped from the
        END, so the LAST field's children are visited first (a parent call
        still precedes its own children).  ast's default ``generic_visit``
        visits forward — RG's per-file call list would carry a different
        order than CGI's, and the SSOT conversion (RG snapshot → CGI graph)
        must be bit-identical.  ``reversed(iter_child_nodes(node))`` mirrors
        the LIFO pop order exactly (same child sequence, same nesting).
        """
        for child in reversed(list(ast.iter_child_nodes(node))):
            self.visit(child)

    def _is_call_used_as_stmt(self, call_node: ast.Call) -> bool:
        """Check if a Call node's parent is an ast.Expr (return value discarded)."""
        if len(self._parent_stack) < 2:
            return False
        parent = self._parent_stack[-2]
        return isinstance(parent, ast.Expr)

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to Python module name.

        Thin delegate to the module-level :func:`path_to_module` — the single
        source of the path→module convention (see its docstring for the
        CWD-relativization warnings).  Kept as a method so callers of this
        visitor keep a stable private API.
        """
        return path_to_module(file_path, self.repo_root)

    def _compute_signature_hash(self, node: ast.FunctionDef) -> str | None:
        """Compute a hash of the function signature.

        Pure arg-name arithmetic over a parser-produced AST — cannot raise.
        B2: previously wrapped in ``suppress(Exception)`` which silently
        degraded to ``None`` (missing change detection / duplicate-symbol
        quality loss) on any bug.  Fail-fast now.

        RG-B1: positional-only args (``def f(a, /, b)``) were dropped —
        ``def f(b)`` and ``def f(a, /, b)`` collided to the same hash, so a
        posonly arg added/removed was invisible to change detection.  The
        ``/`` boundary marker is now encoded so a posonly↔regular conversion
        (an API-breaking change) also changes the hash.
        """
        parts = [node.name]
        # positional-only args (def f(a, /, b))
        parts.extend(arg.arg for arg in node.args.posonlyargs)
        if node.args.posonlyargs:
            parts.append("/")  # encode boundary so posonly↔regular changes the hash
        # Add positional-or-keyword arguments
        parts.extend(arg.arg for arg in node.args.args)
        # Add vararg
        if node.args.vararg:
            parts.append("*" + node.args.vararg.arg)
        # Add kwonlyargs
        parts.extend(arg.arg for arg in node.args.kwonlyargs)
        # Add kwarg
        if node.args.kwarg:
            parts.append("**" + node.args.kwarg.arg)
        signature = ",".join(parts)
        # Compute SHA1 hash (hex digest)
        return hashlib.sha1(signature.encode(), usedforsecurity=False).hexdigest()[:8]  # first 8 chars

    def _extract_signature(self, node: ast.FunctionDef) -> str | None:
        """Extract full function signature text with type annotations.

        Operates only on parser-produced AST — ``ast.unparse`` cannot fail on
        a valid tree.  B1: the 8 ``suppress(Exception)`` wrappers were dead
        defense; a failure here is a bug and must fail loudly instead of
        silently indexing the symbol WITHOUT a signature (search/read output
        degradation).

        RG-B1: positional-only args were dropped — ``def f(a, /, b)``
        rendered as ``def f(b)`` (losing the ``a`` parameter entirely),
        corrupting the LLM-facing signature.  Defaults span BOTH posonlyargs
        and regular args (``node.args.defaults`` covers the trailing
        positional params of the combined list), so the offset is computed
        over the merged list.
        """
        params = []
        args = node.args

        # positional-only + positional-or-keyword args (def f(a, /, b, c=1)).
        # ``defaults`` covers the TRAILING positional params across BOTH
        # groups, so the offset must be over the combined list — otherwise a
        # posonlyarg default misaligns with the wrong regular arg.
        pos_params = list(args.posonlyargs) + list(args.args)
        defaults_offset = len(pos_params) - len(args.defaults)
        n_posonly = len(args.posonlyargs)
        for i, arg in enumerate(pos_params):
            p = arg.arg
            if arg.annotation:
                p += f": {ast.unparse(arg.annotation)}"
            # defaults
            default_idx = i - defaults_offset
            if 0 <= default_idx < len(args.defaults):
                p += f" = {ast.unparse(args.defaults[default_idx])}"
            params.append(p)
        # bare '/' separator after positional-only args — always rendered
        # when posonlyargs exist; coexists with *args (def f(a, /, *args, b)).
        if n_posonly:
            params.insert(n_posonly, "/")

        # *args
        if args.vararg:
            p = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                p += f": {ast.unparse(args.vararg.annotation)}"
            params.append(p)
        elif args.kwonlyargs:
            params.append("*")

        # keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            p = arg.arg
            if arg.annotation:
                p += f": {ast.unparse(arg.annotation)}"
            kd = args.kw_defaults[i] if i < len(args.kw_defaults) else None
            if kd is not None:
                p += f" = {ast.unparse(kd)}"
            params.append(p)

        # **kwargs
        if args.kwarg:
            p = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                p += f": {ast.unparse(args.kwarg.annotation)}"
            params.append(p)

        ret = ""
        if node.returns:
            ret = f" -> {ast.unparse(node.returns)}"

        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(params)}){ret}"

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

        # P3 Stage 2: CGI-convention defs symbol — direct class method
        # "Class.method" (parent scope is a class), bare name otherwise.
        # RG's full-scope qualname (e.g. "Outer.inner") differs from CGI's
        # (e.g. "inner"); the snapshot must carry the CGI form so
        # CallGraphIndexer can reconstruct defs without re-deriving scopes.
        if self._scope_stack and self._scope_is_class[-1]:
            _parent_bare = self._scope_stack[-1].rsplit(".", 1)[-1]
            cgi_symbol = f"{_parent_bare}.{node.name}"
        else:
            cgi_symbol = node.name
        is_async = isinstance(node, ast.AsyncFunctionDef)

        symbol = SymbolNode(
            name=node.name,
            qualname=qualname,
            module=self.module,
            file_path=self.file_path,
            kind="function" if not self.current_class else "method",
            start_line=node.lineno,
            end_line=node.end_lineno if node.end_lineno is not None else node.lineno,
            signature_hash=signature_hash,
            docstring=ast.get_docstring(node),
            signature=self._extract_signature(node),
            is_async=is_async,
            cgi_symbol=cgi_symbol,
            ast_depth=self._ast_depth - 1,  # depth of the DEFINING scope
        )
        self.symbols.append(symbol)

        # Visit child nodes to capture calls inside this function.
        # P3 Stage 2: decorator calls are attributed to the decorated function
        # (CGI parity — CallGraphIndexer._iter_calls traverses the whole
        # function subtree INCLUDING decorator_list).  The line-based
        # _get_current_symbol would drop them (their lines sit OUTSIDE the
        # function's [start,end] range), so visit them explicitly under the
        # pushed scope, in LIFO order — reversed() mirrors CGI's todo.pop()
        # traversal, which yields the LAST decorator first.  The decorator
        # list is detached during generic_visit so its calls are not
        # double-visited, and restored afterwards (visit() is synchronous).
        self._scope_stack.append(qualname)
        self._scope_is_class.append(False)
        self._scope_def_line.append(node.lineno)
        self._in_function += 1
        if node.decorator_list:
            _decorators = node.decorator_list
            node.decorator_list = []
            self._decorator_stack.append(qualname)
            try:
                for _dec in reversed(_decorators):
                    self.visit(_dec)
            finally:
                self._decorator_stack.pop()
            self.generic_visit(node)
            # Restore AFTER generic_visit: the reversed child walk would
            # otherwise re-visit the (detached) decorators and double-collect
            # their calls.
            node.decorator_list = _decorators
        else:
            self.generic_visit(node)
        self._in_function -= 1
        self._scope_stack.pop()
        self._scope_is_class.pop()
        self._scope_def_line.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definition."""
        # Qualify local classes (defined inside a function scope) so two
        # functions each defining ``class Foo`` don't collide on qualname —
        # mirrors the nested-function fix.
        class_qualname = f"{self._scope_stack[-1]}.{node.name}" if self._scope_stack else node.name

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
            end_line=node.end_lineno if node.end_lineno is not None else node.lineno,
            signature_hash=None,
            docstring=ast.get_docstring(node),
            bases=bases if bases else None,
        )
        self.symbols.append(symbol)

        # Visit methods inside class
        previous_class = self.current_class
        self.current_class = class_qualname
        self._scope_stack.append(class_qualname)
        self._scope_is_class.append(True)
        self._scope_def_line.append(node.lineno)
        self.generic_visit(node)
        self._scope_stack.pop()
        self._scope_is_class.pop()
        self._scope_def_line.pop()
        self.current_class = previous_class

    def _resolve_call_name(self, node: ast.AST) -> str | None:
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
        if isinstance(node, ast.Attribute):
            value = self._resolve_call_name(node.value)
            if value is None:
                return None
            # Strip instance/class receiver — keep only the method name.
            # "self.foo" → "foo", "cls.bar" → "bar"
            if value in ("self", "cls"):
                return node.attr
            return f"{value}.{node.attr}"
        if isinstance(node, ast.Call):
            # Chained calls like obj.method()() — resolve the function part
            return self._resolve_call_name(node.func)
        return None

    def _canonical_call_attrs(self, func: ast.expr, legacy_callee: str) -> tuple[str, str, float, str]:
        """CGI-convention call attribution (P3 Stage 1).

        Mirrors ``CallGraphIndexer._parse_call`` so RG's per-file payload is
        consumable by CGI (the SSOT-merge prerequisite):

        * ``foo()``                      → ("foo", "foo", 0.9, "")
        * ``self.foo()`` inside a DIRECT method of class C
                                         → ("C.foo", "self.foo", 0.85, "")
        * ``obj.foo()`` / ``mod.foo()``  → ("foo", "obj.foo", 0.5, "")
        * unsupported forms             → (legacy_callee, legacy_callee, 0.2, "fallback")

        ``cls.`` receivers deliberately fall to the generic-attribute branch:
        CGI special-cases ONLY ``self``, and bit-for-bit cache parity requires
        matching that exactly.  Unsupported forms (chained ``obj.m()()``,
        dynamic receivers) fall back to the legacy attribution — the edge is
        kept for RG's legacy consumers with LOW confidence (0.2) and an
        explicit ``resolution="fallback"`` marker, so it never outranks a
        resolved call and the SSOT conversion can drop it by marker, not by
        confidence value (P2, 2026-08-12).

        Stage 2 precondition (the remaining SSOT-merge gap): the bare
        immediate class NAME is used to qualify ``self.`` calls — NOT the full
        qualname — because CGI's ``class_names`` maps a method to its enclosing
        ``ClassDef.name`` (``node.name``), so a nested ``Outer.Inner`` qualifies
        as ``"Inner.method"``.  The two workers must agree bit-for-bit; this
        method derives the bare name from the scope-stack qualname to match.
        """
        if isinstance(func, ast.Name):
            return func.id, func.id, 0.9, ""
        if isinstance(func, ast.Attribute):
            # Reconstruct dotted name from attribute chain
            parts: list[str] = []
            n: ast.expr = func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if not isinstance(n, ast.Name):
                return legacy_callee, legacy_callee, 0.2, "fallback"
            parts.append(n.id)
            dotted = ".".join(reversed(parts))
            root_name = parts[-1]  # outermost name (e.g. "self", "obj")
            attr = parts[0]  # the actual method/function name
            # Direct-method receiver: the entry directly below the current
            # function on the scope stack is its parent scope — qualify only
            # when that parent is a class (scope_is_class[-2]).  Use the bare
            # immediate class NAME (last qualname component), matching CGI's
            # class_names[node.name] — a nested Outer.Inner qualifies as Inner.
            if root_name == "self" and len(self._scope_is_class) >= 2 and self._scope_is_class[-2]:
                immediate_class = self._scope_stack[-2].rsplit(".", 1)[-1]
                return f"{immediate_class}.{attr}", dotted, 0.85, ""
            return attr, dotted, 0.5, ""
        return legacy_callee, legacy_callee, 0.2, "fallback"

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function calls with object-identity and side-effect annotations."""
        callee = self._resolve_call_name(node.func)
        if callee is None:
            # Unsupported call expression
            self.generic_visit(node)
            return

        # Decorator call: attribute to the decorated function (CGI parity;
        # P3 Stage 2).  The line-based lookup would mis-attribute it to an
        # enclosing function whose range covers the decorator line.
        caller = self._decorator_stack[-1] if self._decorator_stack else self._get_current_symbol(line=node.lineno)
        if caller:
            # Object identity: capture literal positional arg values.
            # get_user(1) → ["1"], fetch("admin") → ['"admin"'].
            # Only ast.Constant nodes — expressions like f(x+1) are skipped.
            call_args = [repr(arg.value) for arg in node.args if isinstance(arg, ast.Constant)]

            # Side-effect semantics: AST-structural mutating call detection.
            is_mutating = _is_mutating_call(node, callee, parent_is_expr=self._is_call_used_as_stmt(node))

            # P3 Stage 1: canonical (CGI-convention) attribution alongside the
            # legacy bare ``callee`` — see _canonical_call_attrs.
            callee_symbol, callee_display, confidence, resolution = self._canonical_call_attrs(node.func, callee)

            # P3 Stage 2: CGI-convention CALLER — direct class method
            # "Class.method" (the current function's parent scope is a class),
            # bare name otherwise.  Mirrors CallGraphIndexer._collect_calls'
            # caller_sym so RG's snapshot can serve CGI's forward edges.
            # Calls in a CLASS BODY (e.g. ``chromium = _FakeLauncher()``) are
            # NOT their own scope: CGI attributes them to the enclosing
            # function (its subtree walk includes the class), so the innermost
            # FUNCTION scope wins — skip class scopes when looking for the
            # caller.
            _i = len(self._scope_stack) - 1
            while _i >= 0 and self._scope_is_class[_i]:
                _i -= 1
            if _i >= 0:
                _fn_qname = self._scope_stack[_i]
                _caller_def_line = self._scope_def_line[_i]
                if _i >= 1 and self._scope_is_class[_i - 1]:
                    _parent_bare = self._scope_stack[_i - 1].rsplit(".", 1)[-1]
                    caller_symbol = f"{_parent_bare}.{_fn_qname.rsplit('.', 1)[-1]}"
                else:
                    caller_symbol = _fn_qname.rsplit(".", 1)[-1]
            else:
                caller_symbol = caller
                _caller_def_line = None

            edge = CallEdge(
                caller=caller,
                callee=callee,
                file_path=self.file_path,
                line=node.lineno,
                call_args=call_args,
                is_mutating=is_mutating,
                callee_symbol=callee_symbol,
                callee_display=callee_display,
                confidence=confidence,
                resolution=resolution,
                caller_symbol=caller_symbol,
                caller_def_line=_caller_def_line,
            )
            self.calls.append(edge)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statements."""
        for alias in node.names:
            edge = ImportEdge(importer=self.file_path, imported=alias.name, import_type="import", alias=alias.asname)
            self.imports.append(edge)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from ... import statements."""
        module = node.module or ""
        for alias in node.names:
            imported = f"{module}.{alias.name}" if module else alias.name
            edge = ImportEdge(importer=self.file_path, imported=imported, import_type="from", alias=alias.asname)
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

    def _get_current_symbol(self, line: int | None = None) -> str | None:
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
            best: tuple[str, int] | None = None  # (qualname, span)
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
