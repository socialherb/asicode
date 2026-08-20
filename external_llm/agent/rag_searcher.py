"""
BM25-based code relevance searcher for asicode Agent (RAG context injection).

No external dependencies — BM25 implemented from scratch.
Indexes Python/JS/TS/Go/Rust/Java/… files; handles CamelCase and snake_case.

Public API
----------
RAGSearcher(repo_root)
  .find_relevant_files(query, top_k, *, file_glob)  -> List[SearchResult]
  .invalidate_files(changed_paths)  # incremental index update after edits
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Deterministic, source-prioritized descent order shared with the symbol /
# call-graph walkers — keeps the RAG corpus reproducible and source-first.
from ..common.walk_policy import _WALK_SKIP_FILE_SUFFIXES, _walk_dir_sort_key, _walk_should_skip_dir

# Language extension SSOT — keeps the RAG corpus in lock-step with the rest of
# the language layer (see the "6 SSOT dimensions" invariant in test_tree_sitter
# _utils.py).  Importing _EXT_MAP here closes the last open drift the invariant
# documents: _INDEXED_EXTS was a hardcoded subset that silently dropped
# half-wired languages (.lua/.scala/.css/.html/.json/.pyi/.mjs/.cc/.kts/…).
from ..languages.models import _EXT_MAP
from .bm25 import bm25_idf_pairs, bm25_score_pairs
from .cancel_scope import effective_cancel
from .config.thresholds import config as _cfg
from .performance_metrics import get_global_collector
from .rag_configs import CodeTokenizer
from .vector_cache import HAS_FAISS, HAS_NUMPY, HAS_SENTENCE_TRANSFORMERS, VectorCacheManager

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Single source of truth: every extension the language layer recognises
# (_EXT_MAP) plus doc/config extras that carry no language semantics but are
# still worth searching (README, pyproject/Cargo manifests, CI configs).
# Derived at import time so adding a language to _EXT_MAP makes it
# RAG-indexable automatically — no manual sync, no silent omission.
_INDEXED_EXTS = set(_EXT_MAP) | {".md", ".toml", ".yaml", ".yml"}

# RAG-only extra exclusions layered on the shared walk policy (below).
_RAG_EXTRA_SKIP_DIRS = frozenset({"migrations"})


def _rag_should_skip_dir(d: str) -> bool:
    """Shared walk-admission predicate (B2' parity) + RAG-only ``migrations``.

    The RAG walker is the 4th repo walker: it must prune exactly what
    CGI/symbol_search/vulture/RepositoryGraph prune (hidden / venv* / vendor /
    site-packages / *.egg-info via ``_walk_should_skip_dir``), or the corpus
    drifts from the graph universes — vendored bundles or a venv
    site-packages subtree can even starve real source under the file cap
    (F-RAG-2, 2026-08-12).  ``migrations`` is a deliberate RAG-only
    exclusion (DB migrations carry no searchable symbols).
    """
    return _walk_should_skip_dir(d) or d in _RAG_EXTRA_SKIP_DIRS

_MAX_FILES = _cfg.counts.RAG_MAX_FILES
_MAX_FILE_CHARS = _cfg.lines.RAG_FILE_CHARS

# (BM25 tuning constants ``_K1``/``_B`` and the ``_bm25_score`` core moved to
# ``agent/bm25.py`` — the single source shared with insights_manager,
# design_chat_loop, symbol_search and read_tools. P9-1.)

# (Stopwords moved to ``CodeTokenizer`` in ``rag_configs.py`` — removed from
# this module.)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    file: str           # relative path from repo root
    score: float
    snippet: str        # most relevant excerpt (~120 chars)
    line: int = 0       # approximate line of best match


# ── Shared per-repo index ─────────────────────────────────────────────────────
# PERF-4: every RAGSearcher instance of this process searching the SAME
# repo_root shares one index (arrays, corpus stats, generation counter, lock
# and fingerprint map).  A fresh ToolRegistry / session therefore reuses the
# existing build after a cheap walk+stat reconciliation (_ensure_index) instead
# of re-reading and re-tokenizing the whole corpus — the multi-second first-call
# cost is paid once per process, not once per instance.  Bounded LRU (cap 8,
# matching the cached_repo_file_list pattern): cold repos are evicted and their
# next instance falls back to a full build.


class _SharedIndex:
    """Process-wide BM25 index state for one repo_root.

    The generation counter is SHARED so a mutation by ANY instance invalidates
    every instance's per-instance search cache (see the generation checks in
    ``find_relevant_files``).
    """

    __slots__ = (
        "avgdl",
        "built",
        "df",
        "doc_lengths",
        "doc_texts",
        "doc_token_counts",
        # rel_path -> (st_mtime_ns, st_size) for every walked candidate file,
        # captured at build/invalidate time.  The reconciliation diff in
        # _ensure_index compares a cheap re-walk against this map to decide
        # which files must be re-read (O(changed) instead of O(corpus)).
        "fingerprints",
        "index_generation",
        "index_truncated",
        "lock",
        "n_docs",
        "rel_path_to_idx",
        "rel_paths",
        "total_doc_len",
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.built = False
        self.rel_paths: list[str] = []
        self.doc_token_counts: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.doc_texts: list[str] = []
        self.rel_path_to_idx: dict[str, int] = {}
        self.df: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.total_doc_len: int = 0
        self.index_generation: int = 0
        self.index_truncated: bool = False
        self.fingerprints: dict[str, tuple[int, int]] = {}


_SHARED_INDEXES: OrderedDict[str, _SharedIndex] = OrderedDict()
_SHARED_INDEXES_LOCK = threading.Lock()
_SHARED_INDEXES_MAX = 8  # bound (LRU eviction) — match cached_repo_file_list cap


def _get_shared_index(repo_root: str) -> _SharedIndex:
    """Return the process-wide shared index for ``repo_root`` (LRU-capped)."""
    with _SHARED_INDEXES_LOCK:
        si = _SHARED_INDEXES.get(repo_root)
        if si is not None:
            _SHARED_INDEXES.move_to_end(repo_root)
            return si
        si = _SharedIndex()
        _SHARED_INDEXES[repo_root] = si
        if len(_SHARED_INDEXES) > _SHARED_INDEXES_MAX:
            _SHARED_INDEXES.popitem(last=False)
        return si


# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Module-level singleton — replaces ad-hoc ``_split_camel`` + ``_tokenize``
# regex functions and ``_STOP`` frozenset.  Handles CamelCase, snake_case,
# and stop-word filtering consistently with the rest of the codebase.
_TOKENIZER = CodeTokenizer()


# ── Snippet extraction ────────────────────────────────────────────────────────

def _extract_snippet(text: str, query_tokens: list[str], window: int = 120) -> tuple[str, int]:
    """Return (snippet, 1-indexed line) for the best-matching line.

    Optimization: skip lines that cannot contain a query token, so the expensive
    ``_TOKENIZER.tokenize()`` runs only on candidate lines (~188 ms for a 10k-line
    file otherwise; a top_k=10 search over this repo spent ~970 ms here).

    The prefilter tests against the LOWERCASED line, because every token
    ``tokenize()`` emits is a lowercased contiguous substring of the raw text:
    it lowercases, then sub-splits on CamelCase/underscore boundaries, and
    min_token_len/stop-words only ever DROP tokens (``rag_configs.tokenize``).
    Case-folding is therefore load-bearing, not cosmetic — a case-sensitive
    check misses ``write`` in ``WriteToolHandler`` and silently returns a worse
    line (measured: 14.3% of lookups diverged).

    ``best_score`` starts at 0, not -1, for the same reason: the unfiltered loop
    always evaluated line 0 and so fell back to line 1 for a zero-hit document.
    With a prefilter line 0 may be skipped, so seeding 0 preserves that default.
    Together these make the output byte-identical to the unfiltered scan
    (verified: 0 divergences over 110 files x 20 queries).
    """
    lines = text.splitlines()
    if not lines:
        return "", 1
    q_set = set(query_tokens)
    best_line, best_score = 0, 0
    for i, line in enumerate(lines):
        # Prefilter: no query token as a substring => tokenize() cannot match either.
        lowered = line.lower()
        if not any(qt in lowered for qt in q_set):
            continue
        hit = sum(1 for t in _TOKENIZER.tokenize(line) if t in q_set)
        if hit > best_score:
            best_score, best_line = hit, i
    snippet = lines[best_line].strip()[:window]
    return snippet, best_line + 1


# ── Main class ────────────────────────────────────────────────────────────────

class RAGSearcher:
    """
    Lightweight BM25 code searcher.

    Index is built lazily on first search and cached in memory.
    Call invalidate_files() to update incrementally after edits.

    All instances of this process searching the same repo_root SHARE one index
    (see ``_SharedIndex``).  A fresh instance reuses the shared build after a
    cheap walk+stat reconciliation against the fingerprint map
    (``_ensure_index``), so the full corpus re-read only happens once per
    process — or when files actually changed.  Externally-modified files (edits
    outside this process's write funnel) are picked up by the next instance's
    reconciliation.
    """

    def __init__(
        self,
        repo_root: str,
        vector_cache_enabled: bool = True,
        cancel_event: Optional[threading.Event] = None,
        config: Any = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        # Shared per-repo index (see _SharedIndex): every instance of this
        # process searching the SAME repo_root shares one BM25 index, one lock,
        # one generation counter and the fingerprint map.  A fresh ToolRegistry
        # / session therefore reuses the existing build after a cheap walk+stat
        # reconciliation (_ensure_index) instead of re-reading and
        # re-tokenizing the whole corpus — the multi-second first-call cost is
        # paid once per process, not once per instance.
        self._s = _get_shared_index(str(self.repo_root))
        # The index lock is SHARED: array mutations from any instance serialize
        # against every instance's readers (subagent clones share by reference
        # via ToolRegistry.clone_for_subagent).
        self._index_lock = self._s.lock
        # True once this instance has either built the index or verified the
        # shared index against disk.  Read lock-free in the _ensure_index fast
        # path (benign race — worst case a redundant reconciliation).
        self._reconciled = False
        # Cooperative cancel: hold config (NOT the event value) and read
        # config.cancel_event FRESH in _build_index via _get_cancel_event.
        # The design-chat REPL mutates config.cancel_event PER TURN (asi.py)
        # AFTER this searcher is constructed with cancel_event=None; a captured
        # value would freeze None and leave ESC inert during the multi-second
        # first find_relevant_files build — the exact interactive path ESC must
        # protect. An explicit cancel_event arg (tests / direct callers) wins.
        self._cancel_event = cancel_event
        self._config = config
        self.vector_cache_enabled = vector_cache_enabled
        self.vector_cache_manager = None
        if vector_cache_enabled:
            # vector_cache owns its optional-dependency degradation (HAS_* flags);
            # the module itself always imports, so the old try/except ImportError
            # fallback was dead code.
            if HAS_SENTENCE_TRANSFORMERS and HAS_NUMPY and HAS_FAISS:
                self.vector_cache_manager = VectorCacheManager(str(self.repo_root / ".asicode" / "vector_cache"))
            else:
                logger.warning("Vector cache dependencies not fully installed, disabling")
                self.vector_cache_enabled = False
                self.vector_cache_manager = None
        # Search cache: per-INSTANCE (query results are instance-specific —
        # vector-cache availability differs), bounded LRU + TTL, thread-safe.
        # Entries store the index generation at write time; a hit is served
        # only when that generation still matches, so a mutation by ANY
        # instance (which bumps the SHARED generation) cannot leave a stale
        # entry alive for the 5-min TTL in a sibling instance's cache.
        self._search_cache: "OrderedDict[str, tuple[float, int, list[SearchResult]]]" = OrderedDict()
        self._search_cache_lock = threading.Lock()
        self._search_cache_max = 256  # bound (LRU eviction) — match ToolResultCache

    # ── shared-index property delegation ──────────────────────────────────────
    # All corpus data lives on the process-wide ``_SharedIndex`` (``self._s``).
    # These properties keep the historical ``self._<attr>`` access pattern
    # working unchanged while making the state shared across instances.

    @property
    def _built(self) -> bool:
        return self._s.built

    @_built.setter
    def _built(self, value: bool) -> None:
        self._s.built = value

    @property
    def _rel_paths(self) -> list[str]:
        return self._s.rel_paths

    @_rel_paths.setter
    def _rel_paths(self, value: list[str]) -> None:
        self._s.rel_paths = value

    @property
    def _doc_token_counts(self) -> list[dict[str, int]]:
        return self._s.doc_token_counts

    @_doc_token_counts.setter
    def _doc_token_counts(self, value: list[dict[str, int]]) -> None:
        self._s.doc_token_counts = value

    @property
    def _doc_lengths(self) -> list[int]:
        return self._s.doc_lengths

    @_doc_lengths.setter
    def _doc_lengths(self, value: list[int]) -> None:
        self._s.doc_lengths = value

    @property
    def _doc_texts(self) -> list[str]:
        return self._s.doc_texts

    @_doc_texts.setter
    def _doc_texts(self, value: list[str]) -> None:
        self._s.doc_texts = value

    @property
    def _rel_path_to_idx(self) -> dict[str, int]:
        return self._s.rel_path_to_idx

    @_rel_path_to_idx.setter
    def _rel_path_to_idx(self, value: dict[str, int]) -> None:
        self._s.rel_path_to_idx = value

    @property
    def _df(self) -> dict[str, int]:
        return self._s.df

    @_df.setter
    def _df(self, value: dict[str, int]) -> None:
        self._s.df = value

    @property
    def _avgdl(self) -> float:
        return self._s.avgdl

    @_avgdl.setter
    def _avgdl(self, value: float) -> None:
        self._s.avgdl = value

    @property
    def _n_docs(self) -> int:
        return self._s.n_docs

    @_n_docs.setter
    def _n_docs(self, value: int) -> None:
        self._s.n_docs = value

    @property
    def _total_doc_len(self) -> int:
        return self._s.total_doc_len

    @_total_doc_len.setter
    def _total_doc_len(self, value: int) -> None:
        self._s.total_doc_len = value

    @property
    def _index_generation(self) -> int:
        return self._s.index_generation

    @_index_generation.setter
    def _index_generation(self, value: int) -> None:
        self._s.index_generation = value

    @property
    def _index_truncated(self) -> bool:
        return self._s.index_truncated

    @_index_truncated.setter
    def _index_truncated(self, value: bool) -> None:
        self._s.index_truncated = value

    # ── public API ────────────────────────────────────────────────────────────

    def find_relevant_files(
        self,
        query: str,
        top_k: int = 5,
        *,
        file_glob: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Return top_k files most relevant to query using hybrid BM25 + vector search.
        Optionally filter by file_glob pattern (e.g., '*.py').
        """
        if not query.strip():
            return []
        self._ensure_index()
        if self._n_docs == 0:
            return []

        # Check cache (thread-safe bounded LRU; matches ToolResultCache pattern)
        cache_key = self._make_cache_key(query, top_k, file_glob)
        now = time.monotonic()
        _cached = None
        with self._search_cache_lock:
            if cache_key in self._search_cache:
                timestamp, entry_gen, results = self._search_cache[cache_key]
                if now - timestamp < 300 and entry_gen == self._index_generation:
                    self._search_cache.move_to_end(cache_key)  # refresh LRU position
                    _cached = results
                else:
                    del self._search_cache[cache_key]  # expired, or stale (index changed since store)
        if _cached is not None:
            # Cache hit. Return a shallow copy so a caller mutating the returned
            # list (sort/append/clear/slice-assign) cannot poison the shared cache
            # entry for the 5-min TTL — the cached list object must stay private.
            get_global_collector().record_rag_cache(True)
            return list(_cached)
        # Cache miss
        get_global_collector().record_rag_cache(False)

        # Capture the index generation BEFORE reading the index. If an
        # invalidate_files mutation lands between this read and the cache write
        # below, the generation will differ and we discard the (possibly stale)
        # result instead of poisoning the cache for the 5-min TTL.
        _gen = self._index_generation

        # Start timing for cache miss
        search_start = time.monotonic()

        # Step 1: Get BM25 results
        bm25_results = self._bm25_search(query, top_k * 2, file_glob)  # Get more for merging

        # Step 2: Get vector cache results if enabled
        vector_results = []
        if self.vector_cache_enabled and self.vector_cache_manager is not None:
            vector_results = self._vector_search(query, top_k * 2, file_glob)

        # Step 3: Merge and rank results
        results = self._merge_results(bm25_results, vector_results, top_k)

        # Store in cache (bounded LRU eviction) — but only if the index has not
        # been invalidated since we read it, lest we cache a stale result. The
        # generation comparison MUST run under _search_cache_lock so it is atomic
        # w.r.t. invalidate_files' cache.clear(): the invalidator bumps the
        # generation BEFORE acquiring this lock to clear, so holding the lock
        # across compare+write leaves only two outcomes — we see the bumped
        # generation and skip, or we write and the subsequent clear removes it.
        # (Comparing outside the lock would leave a window between the passing
        # compare and lock acquisition in which the invalidator bumps+clears and
        # the searcher then records a stale result that survives the 5-min TTL.)
        with self._search_cache_lock:
            if _gen == self._index_generation:
                self._search_cache[cache_key] = (time.monotonic(), self._index_generation, results)
                if len(self._search_cache) >= self._search_cache_max:
                    self._search_cache.popitem(last=False)

        # Record search time
        search_elapsed_ms = (time.monotonic() - search_start) * 1000
        get_global_collector().record_rag_search(search_elapsed_ms)

        return results

    def invalidate_files(self, changed_paths: list[str]) -> None:
        """Incrementally update index for changed/new/deleted files only.

        Updates only affected docs while keeping the rest of the index intact.
        Also refreshes the shared fingerprint map so a later instance's
        reconciliation does not re-read files already reflected here.

        Thread-safety / critical-section discipline: split into two phases so a
        bulk invalidation (branch switch / large patch touching dozens of files)
        never blocks parallel subagents' searches on disk + tokenize work.

          * **Phase 1 (outside ``_index_lock``)** — stat + read + tokenize each
            changed file (``_prepare_files``).  This is pure filesystem + CPU
            work and is by far the expensive part; running it under the lock
            would stall every concurrent ``_bm25_search`` / ``_vector_search``
            for the whole duration.  The result is staged in a
            ``norm_path -> (text, tokens)`` map keyed only by path (no array
            index), so phase 2 can re-resolve the index under the lock.
          * **Phase 2 (inside ``_index_lock``)** — only the parallel-array
            mutations (``_apply_changes_locked``): locate each existing entry
            (``list.index`` races with a concurrent ``_remove_doc_at``
            otherwise), subtract/add df contributions, and append/replace/
            remove. This instance is shared across in-process parallel
            subagents (``ToolRegistry.clone_for_subagent`` shares it by
            reference), so a subagent's write-success callback invoking
            invalidate_files while a sibling searches would otherwise corrupt
            the arrays (IndexError, or worse, a silent path↔document
            misalignment as ``_remove_doc_at``'s ``pop`` shifts indices).

        The read/reflection split is safe because phase 2 bumps
        ``_index_generation`` under the lock: an in-flight searcher that
        already read the PRE-mutation index discards its result at the
        cache-write site rather than re-caching stale data. Vector-cache I/O
        (embedding computation) is also deferred outside the lock. The search
        cache is cleared AFTER the mutation completes (and outside the index
        lock).

        Args:
            changed_paths: List of relative file paths that were modified.
        """
        # Phase 1 (outside the lock): stat + read + tokenize each changed file.
        # Files that no longer exist, are not indexable, are unreadable, or
        # yield no tokens are simply absent from `prepared` — phase 2 then
        # treats them as removals (if previously indexed) or no-ops (if never
        # indexed), matching the previous in-lock semantics exactly.
        prepared, fp_map = self._prepare_files(changed_paths)

        # Phase 2 (inside the lock): apply only the array mutations.
        with self._index_lock:
            if not self._built:
                # Index not built yet, nothing to incrementally update.
                return
            vc_updates = self._apply_changes_locked(changed_paths, prepared, fp_map)

        # Clear search cache AFTER the index mutation completes (and outside the
        # index lock). This alone is NOT sufficient: a searcher that already read
        # the PRE-mutation index (and is now past the lock, in the merge/write
        # phase) would re-cache its stale result AFTER this clear. The
        # generation check at the searcher's cache-write site closes that window
        # — such a searcher sees the bumped generation and discards its write.
        with self._search_cache_lock:
            self._search_cache.clear()

        # Flush deferred vector-cache updates outside the index lock (embedding
        # I/O must not block concurrent searches).  ONE batched encode() pass
        # instead of N per-file calls — 1.6x-2.2x faster on large updates,
        # mirroring the cold-start flush in _build_index.
        if vc_updates and self.vector_cache_enabled and self.vector_cache_manager is not None:
            try:
                self.vector_cache_manager.add_documents(vc_updates)
            except Exception as e:
                logger.debug("RAG vector-cache update failed: %s", e)

        logger.debug("RAG incremental update: %d files, index now %d docs", len(changed_paths), self._n_docs)

    def _prepare_files(self, paths: list[str]) -> tuple[dict[str, tuple[str, list[str]]], dict[str, tuple[int, int]]]:
        """Phase 1 of an incremental update (call OUTSIDE ``_index_lock``).

        For every candidate path: stat it, and if it exists / is indexable /
        yields tokens, read + tokenize it.  Returns:
          * ``prepared``: norm_path -> (text, tokens) for files that were read
            and tokenized successfully (absent for deleted / unindexable /
            tokenless files — the caller treats their absence as removal).
          * ``fp_map``: norm_path -> (st_mtime_ns, st_size) for every existing
            indexable candidate (tokenless ones included — their fingerprint is
            stored so reconciliation does not re-read them every time).
        """
        prepared: dict[str, tuple[str, list[str]]] = {}
        fp_map: dict[str, tuple[int, int]] = {}
        for rel_path in paths:
            norm_path = rel_path.strip().lstrip("/")
            abs_path = self.repo_root / norm_path

            file_exists = abs_path.is_file()
            is_indexable = (
                file_exists
                and abs_path.suffix.lower() in _INDEXED_EXTS
                and not Path(norm_path).name.endswith(_WALK_SKIP_FILE_SUFFIXES)
                and not any(
                    _rag_should_skip_dir(part)
                    for part in Path(norm_path).parts[:-1]
                )
            )
            if not is_indexable:
                continue
            try:
                st = abs_path.stat()
            except OSError:
                logger.debug("RAG prepare: stat failed for %s", norm_path)
                continue
            fp_map[norm_path] = (st.st_mtime_ns, st.st_size)
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_FILE_CHARS:
                    text = text[:_MAX_FILE_CHARS].rsplit("\n", 1)[0]
                path_text = norm_path.replace("/", " ").replace("\\", " ").replace(".", " ")
                tokens = _TOKENIZER.tokenize(text + " " + path_text)
                if tokens:
                    prepared[norm_path] = (text, tokens)
            except Exception:
                logger.debug("RAG prepare: read/tokenize failed for %s", norm_path)
                # non-critical — never block execution; treated as unindexable
        return prepared, fp_map

    def _apply_changes_locked(
        self,
        changed_paths: list[str],
        prepared: dict[str, tuple[str, list[str]]],
        fp_map: dict[str, tuple[int, int]],
    ) -> list[tuple[str, str]]:
        """Phase 2 of an incremental update. Caller MUST hold ``_index_lock``.

        Applies only the parallel-array mutations for ``changed_paths`` using
        the pre-tokenized ``prepared`` map (locating each existing entry via
        the ``_rel_path_to_idx`` mirror — rebuilt at the end of every prior
        mutating call, so it matches the arrays at entry; O(1) per lookup
        instead of ``list.index``'s O(corpus) string scan under the lock),
        subtracts/adds df contributions, and refreshes the shared fingerprint
        map.  Pops are deferred to the end and applied in descending index
        order so the snapshot stays valid for the whole loop.  Returns the
        deferred vector-cache updates for the caller to flush OUTSIDE the lock
        (embedding I/O must not block concurrent searches).
        """
        vc_updates: list[tuple[str, str]] = []
        removals: list[int] = []
        _idx_of = self._rel_path_to_idx  # immutable snapshot; valid at entry
        for rel_path in changed_paths:
            norm_path = rel_path.strip().lstrip("/")
            prep = prepared.get(norm_path)

            # Check if this file is in our index (resolved under the lock —
            # list.index races with a concurrent _remove_doc_at otherwise).
            try:
                existing_idx = self._rel_paths.index(norm_path)
            except ValueError:
                existing_idx = -1

            if existing_idx >= 0:
                # File was in index — remove old contribution from df.
                old_tc = self._doc_token_counts[existing_idx]
                for token in set(old_tc):
                    if token in self._df:
                        self._df[token] -= 1
                        if self._df[token] <= 0:
                            del self._df[token]

                if prep is not None:
                    # UPDATE: replace in-place with the pre-tokenized text.
                    text, tokens = prep
                    tc: dict[str, int] = {}
                    for t in tokens:
                        tc[t] = tc.get(t, 0) + 1

                    old_len = self._doc_lengths[existing_idx]
                    self._doc_token_counts[existing_idx] = tc
                    self._doc_lengths[existing_idx] = len(tokens)
                    self._doc_texts[existing_idx] = text
                    # avgdl via running total (O(1)) instead of
                    # per-file sum(self._doc_lengths) (O(n)).
                    self._total_doc_len += len(tokens) - old_len

                    for t in set(tc):
                        self._df[t] = self._df.get(t, 0) + 1

                    self._avgdl = self._total_doc_len / max(self._n_docs, 1)

                    vc_updates.append((norm_path, text))
                else:
                    # File deleted / no longer indexable / no tokens — remove.
                    removals.append(existing_idx)

            elif prep is not None and self._n_docs - len(removals) < _MAX_FILES:
                # NEW file — append to index.
                text, tokens = prep
                tc = {}
                for t in tokens:
                    tc[t] = tc.get(t, 0) + 1

                self._rel_paths.append(norm_path)
                self._doc_token_counts.append(tc)
                self._doc_lengths.append(len(tokens))
                self._doc_texts.append(text)
                self._n_docs += 1
                self._total_doc_len += len(tokens)

                for t in set(tc):
                    self._df[t] = self._df.get(t, 0) + 1

                self._avgdl = self._total_doc_len / max(self._n_docs, 1)

                vc_updates.append((norm_path, text))

            # Fingerprint maintenance: store the stat captured in phase 1 for
            # existing candidates, drop it for deleted/unindexable ones — a
            # later instance's reconciliation must not re-read files that are
            # already reflected here.
            if norm_path in fp_map:
                self._s.fingerprints[norm_path] = fp_map[norm_path]
            else:
                self._s.fingerprints.pop(norm_path, None)

        # Deferred pops: apply in DESCENDING index order so each document is
        # still at its original index when popped — a lower-index removal has
        # not shifted it yet, and higher-index removals do not affect it.
        for idx in sorted(removals, reverse=True):
            self._remove_doc_at(idx)

        # Bump the generation so an in-flight searcher that already read the
        # pre-mutation index discards its (now-stale) result rather than
        # re-caching it after the clear below. Rebuild the path→idx mirror to
        # match the mutated arrays (rebuilt wholesale — list.pop shifts every
        # later index, so incremental maintenance within the loop is unsafe).
        self._index_generation += 1
        self._rel_path_to_idx = {
            _p: _i for _i, _p in enumerate(self._rel_paths)
        }
        return vc_updates

    def _fingerprint_diff_locked(self) -> tuple[list[str], list[str], list[str], bool]:
        """Diff the corpus against the shared fingerprint map.

        Caller MUST hold ``_index_lock`` (the walk mutates the shared
        ``index_truncated`` flag).  Walk + stat costs ~1-4% of a full build and
        replaces the full re-read + re-tokenize for unchanged files.  Returns
        (added, changed, deleted, truncated) where:
          * added    — walked candidates with no stored fingerprint
          * changed  — walked candidates whose (mtime_ns, size) differs
          * deleted  — previously fingerprinted files no longer in the walk
          * truncated— the walk hit _MAX_FILES (the caller must fall back to a
            full rebuild: under a cap the walk itself is incomplete, so an
            incremental diff could drift from a fresh build)
        """
        walked = self._walk_files()
        current: dict[str, tuple[int, int]] = {}
        for fpath in walked:
            try:
                st = fpath.stat()
            except OSError:
                logger.debug("RAG reconcile: stat failed for %s", fpath)
                continue  # vanished between walk and stat — shows up as deleted
            rel = str(fpath.relative_to(self.repo_root))
            current[rel] = (st.st_mtime_ns, st.st_size)
        old = self._s.fingerprints
        added = sorted(k for k in current if k not in old)
        changed = sorted(k for k in old if k in current and old[k] != current[k])
        deleted = sorted(k for k in old if k not in current)
        return added, changed, deleted, self._index_truncated

    def _remove_doc_at(self, idx: int) -> None:
        """Remove document at given index from all parallel arrays.

        Caller MUST hold ``self._index_lock`` — this private helper is invoked
        only from ``invalidate_files`` (under the lock) and does not acquire the
        lock itself to avoid non-reentrant ``threading.Lock`` deadlock.
        """
        self._total_doc_len -= self._doc_lengths[idx]
        self._rel_paths.pop(idx)
        self._doc_token_counts.pop(idx)
        self._doc_lengths.pop(idx)
        self._doc_texts.pop(idx)
        self._n_docs -= 1
        # Recalculate avgdl from the running total (O(1)).
        self._avgdl = self._total_doc_len / max(self._n_docs, 1)

    # ── hybrid search methods ────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int, file_glob: Optional[str] = None) -> list[SearchResult]:
        """BM25-only search returning SearchResult objects."""
        q_tokens = _TOKENIZER.tokenize(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, int]] = []
        # Score all docs and snapshot the winners under the index lock so this
        # traversal cannot race with invalidate_files / _remove_doc_at on the
        # shared parallel arrays (the instance is shared across parallel
        # subagents). Snippet extraction is CPU-bound and touches no shared
        # state, so it runs after releasing the lock using immutable snapshots.
        with self._index_lock:
            n_docs = self._n_docs
            avgdl = self._avgdl
            df = self._df
            rel_paths = self._rel_paths
            doc_tcs = self._doc_token_counts
            doc_lens = self._doc_lengths
            doc_texts = self._doc_texts
            # IDF is query-only (df/n_docs are loop-invariant here) — compute
            # it once per query instead of per doc x token, shortening the
            # lock-held traversal (P9-1; bit-identical scores, sealed by
            # tests/unit/agent/test_bm25_core.py).
            q_pairs = bm25_idf_pairs(q_tokens, df, n_docs)
            for i, rel in enumerate(rel_paths):
                if file_glob and not _match_glob(rel, file_glob):
                    continue
                s = bm25_score_pairs(q_pairs, doc_tcs[i], doc_lens[i], avgdl)
                if s > 0:
                    scored.append((s, i))
            scored.sort(reverse=True)
            winners = [
                (s, rel_paths[idx], doc_texts[idx])
                for s, idx in scored[:top_k]
            ]

        results: list[SearchResult] = []
        for s, path, text in winners:
            snippet, line = _extract_snippet(text, q_tokens)
            results.append(SearchResult(
                file=path,
                score=round(s, 3),
                snippet=snippet,
                line=line,
            ))
        return results

    def _vector_search(self, query: str, top_k: int, file_glob: Optional[str] = None) -> list[SearchResult]:
        """Vector cache search returning SearchResult objects."""
        if not self.vector_cache_enabled or self.vector_cache_manager is None:
            return []

        raw_results = self.vector_cache_manager.search(query, top_k)
        # Record vector cache hit/miss
        if raw_results:
            get_global_collector().record_vector_cache(True)
        else:
            get_global_collector().record_vector_cache(False)

        results: list[SearchResult] = []
        q_tokens = _TOKENIZER.tokenize(query)

        for item in raw_results:
            file_path = item["file_path"]

            # Apply file glob filter
            if file_glob and not _match_glob(file_path, file_glob):
                continue

            # Snapshot the doc text under the index lock (serializes against
            # invalidate_files on the shared arrays). The vector search itself
            # touches no index state and runs lock-free; snippet extraction is
            # CPU-bound and runs outside the lock on an immutable string.
            doc_text = ""
            with self._index_lock:
                idx = self._rel_path_to_idx.get(file_path)
                if idx is not None:
                    try:
                        doc_text = self._doc_texts[idx]
                    except IndexError:
                        doc_text = ""

            if doc_text:
                snippet, line = _extract_snippet(doc_text, q_tokens)
            else:
                # Fallback: use query-relevant snippet from raw content
                snippet, line = _extract_snippet(item.get("content", ""), q_tokens)

            # Convert vector score (0-1) to compatible range with BM25
            # BM25 scores are typically 0-15+, so we scale vector scores
            vector_score = item["score"] * 10.0  # Scale to 0-10 range

            results.append(SearchResult(
                file=file_path,
                score=round(vector_score, 3),
                snippet=snippet,
                line=line,
            ))

        return results

    def _merge_results(self, bm25_results: list[SearchResult], vector_results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Merge and deduplicate BM25 and vector search results using Reciprocal Rank Fusion."""
        all_files = {r.file for r in bm25_results} | {r.file for r in vector_results}

        # Reciprocal Rank Fusion — no score normalization needed, just ranks
        RRF_K = 60.0

        def _rrf_score(file: str, rank_list: list[SearchResult]) -> float:
            for rank, r in enumerate(rank_list):
                if r.file == file:
                    return 1.0 / (RRF_K + rank)
            return 0.0

        scored_files: list[tuple[float, str]] = []
        for file in all_files:
            rrf = _rrf_score(file, bm25_results) + _rrf_score(file, vector_results)
            scored_files.append((rrf, file))

        # Sort by RRF score
        scored_files.sort(reverse=True)

        # Build final results with snippets from whichever source has better snippet
        final_results: list[SearchResult] = []
        for score, file in scored_files[:top_k]:
            # Prefer BM25 result for snippet (has line info)
            bm25_result = next((r for r in bm25_results if r.file == file), None)
            vector_result = next((r for r in vector_results if r.file == file), None)

            if bm25_result:
                final_results.append(SearchResult(
                    file=file,
                    score=round(score, 4),
                    snippet=bm25_result.snippet,
                    line=bm25_result.line,
                ))
            elif vector_result:
                final_results.append(SearchResult(
                    file=file,
                    score=round(score, 4),
                    snippet=vector_result.snippet,
                    line=vector_result.line,
                ))

        return final_results

    # ── internal ──────────────────────────────────────────────────────────────

    def _make_cache_key(self, query: str, top_k: int, file_glob: Optional[str]) -> str:
        """Generate cache key for search parameters."""
        key_data = f"{query}:{top_k}:{file_glob if file_glob else ''}"
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def _ensure_index(self) -> None:
        # Fast path: the shared index is built AND this instance already
        # reconciled it against disk — no lock, no walk.  (Benign race: a
        # concurrent mutation only makes the data fresher.)
        if self._built and self._reconciled:
            return
        with self._index_lock:
            if not self._built:
                t0 = time.monotonic()
                completed = self._build_index()
                if not completed:
                    # Cancelled mid-build: leave _built False so the next query
                    # retries.  _build_index accumulates into *local* lists/dicts
                    # and only commits to self._* at the very end, so instance
                    # state is pristine on cancel (no half-populated arrays to
                    # reset).  (Side note: vector_cache_manager.add_document is a
                    # per-file side-effect inside the loop and may be partially
                    # written on cancel; it is incremental/idempotent and
                    # decoupled from the BM25 path, so this is safe.)
                    return
                elapsed = time.monotonic() - t0
                logger.debug("RAG index built: %d docs in %.2fs", self._n_docs, elapsed)
                if self._index_truncated:
                    logger.warning(
                        "RAG index for %s TRUNCATED at %d files (cap %d) — files beyond "
                        "the cap are INVISIBLE to find_relevant_files. Raise "
                        "RAG_MAX_FILES if the repo is larger.",
                        self.repo_root, self._n_docs, _MAX_FILES,
                    )
                self._built = True
                self._reconciled = True
                return

            # The shared index was built by an earlier instance of this
            # process.  Reconcile it against disk: walk + stat (cheap), diff
            # against the stored fingerprints, then re-read ONLY the differing
            # files (O(changed)) — a new ToolRegistry / session on the same
            # repo no longer pays the full re-read + re-tokenize cost.
            added, changed, deleted, truncated = self._fingerprint_diff_locked()
            if truncated:
                # Cap-mode repo: the walk itself is incomplete, so an
                # incremental diff could drift from a fresh build (files
                # entering/leaving the cap window).  Fall back to the full
                # rebuild — the rare, already-expensive case.
                t0 = time.monotonic()
                completed = self._build_index()
                if not completed:
                    return
                logger.debug(
                    "RAG index rebuilt (cap-mode): %d docs in %.2fs",
                    self._n_docs, time.monotonic() - t0,
                )
                self._built = True
                self._reconciled = True
                return

        # Outside the lock: read + tokenize only the differing files.  This is
        # pure filesystem + CPU work (the expensive part) and must not run
        # under the shared index lock, mirroring invalidate_files' phase split.
        if added or changed:
            prepared, fp_map = self._prepare_files(added + changed)
        else:
            prepared, fp_map = {}, {}
        if added or changed or deleted:
            with self._index_lock:
                vc_updates = self._apply_changes_locked(added + changed + deleted, prepared, fp_map)
            with self._search_cache_lock:
                self._search_cache.clear()
            # Flush deferred vector-cache updates outside the index lock
            # (embedding I/O must not block concurrent searches).  ONE batched
            # encode() pass instead of N per-file calls (see _build_index).
            if vc_updates and self.vector_cache_enabled and self.vector_cache_manager is not None:
                try:
                    self.vector_cache_manager.add_documents(vc_updates)
                except Exception as e:
                    logger.debug("RAG vector-cache update failed: %s", e)
        self._reconciled = True

    def _get_cancel_event(self) -> Optional[threading.Event]:
        """Return the live cooperative-cancel event.

        Reads ``config.cancel_event`` FRESH (call-time, not construction-time)
        so a per-turn mutation of ``config.cancel_event`` — as the design-chat
        REPL performs each turn — is honored even though this searcher was
        constructed before the mutation landed.  An explicit ``cancel_event``
        passed to ``__init__`` (tests / direct callers without a config) takes
        precedence and is returned as-is.  Returns None when neither is set
        (non-interactive CLI, out-of-process callers) → checkpoints become
        inert no-ops.

        Also merges the innermost per-call cancel scope (MCP ``wait_for``
        timeout, aborted ``dispatch_parallel`` batch) via :func:`effective_cancel`,
        so an index build abandoned by ITS caller bails at the next checkpoint
        with its instance state pristine — ``_build_index`` runs on the calling
        thread, so the thread-local scope is visible here.  The composite
        duck-types ``is_set()``, the only method the checkpoints consume.
        """
        if self._cancel_event is not None:
            return self._cancel_event
        return effective_cancel(getattr(self._config, "cancel_event", None))

    def _build_index(self) -> bool:
        """Build the BM25 index. Returns True if completed, False if cancelled.

        Accumulates into *local* lists/dicts and only commits to ``self._*`` at
        the very end, so a mid-build cancel leaves instance state pristine —
        ``_built`` stays False and the next query re-runs this method from
        scratch.  Vector-cache additions are staged during the loop and flushed
        in a single batched ``add_documents`` call only on success; on cancel
        the staged list is discarded (next rebuild re-adds everything).  Hold
        ``_index_lock`` for the whole build (the caller does) so no reader
        races the final commit.

        Also captures a (st_mtime_ns, st_size) fingerprint for every walked
        candidate file and commits it with the index: a later instance's
        reconciliation diffs a cheap re-walk against this map instead of
        re-reading the whole corpus.
        """
        rel_paths: list[str] = []
        doc_tcs: list[dict[str, int]] = []
        doc_lens: list[int] = []
        doc_texts: list[str] = []
        df: dict[str, int] = {}
        total_len = 0
        fingerprints: dict[str, tuple[int, int]] = {}
        # Staged vector-cache additions — flushed as one batched encode() after
        # the loop (see add_documents). Left empty on cancel, so nothing is
        # written to the vector cache for a partial build.
        vc_updates: list[tuple[str, str]] = []

        for fpath in self._walk_files():
            # Cooperative cancel: the per-file read+tokenize loop is the
            # dominant cost of a first find_relevant_files call (seconds on
            # large repos).  Bail out between files; _ensure_index keeps
            # _built False so the partial arrays never become visible.
            _ce = self._get_cancel_event()
            if _ce is not None and _ce.is_set():
                logger.debug("RAG index build cancelled")
                return False
            try:
                # Capture the fingerprint BEFORE reading: it records the exact
                # bytes this build consumed (reconciliation diffs against it).
                st = fpath.stat()
                rel = str(fpath.relative_to(self.repo_root))
                fingerprints[rel] = (st.st_mtime_ns, st.st_size)
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_FILE_CHARS:
                    text = text[:_MAX_FILE_CHARS].rsplit("\n", 1)[0]
                # Augment with path tokens (filename + parent dirs carry signal)
                path_text = rel.replace("/", " ").replace("\\", " ").replace(".", " ")
                tokens = _TOKENIZER.tokenize(text + " " + path_text)
                if not tokens:
                    continue
                tc: dict[str, int] = {}
                for t in tokens:
                    tc[t] = tc.get(t, 0) + 1
                rel_paths.append(rel)
                doc_tcs.append(tc)
                doc_lens.append(len(tokens))
                doc_texts.append(text)
                total_len += len(tokens)
                for t in set(tc):
                    df[t] = df.get(t, 0) + 1

                # Stage vector-cache additions for a single batched flush after
                # the loop (one encode() over all docs — 1.6x-2.2x faster than
                # per-file add_document on a cold-start build). On cancel the
                # staged list is discarded; the next successful rebuild re-adds
                # everything, so no data is lost in steady state.
                if self.vector_cache_enabled and self.vector_cache_manager is not None:
                    vc_updates.append((rel, text))
            except (AttributeError, TypeError, OSError):
                logger.debug("rag add_document failed for file", exc_info=True)
                continue

        n = len(rel_paths)
        self._rel_paths = rel_paths
        self._doc_token_counts = doc_tcs
        self._doc_lengths = doc_lens
        self._doc_texts = doc_texts
        self._df = df
        self._n_docs = n
        self._total_doc_len = total_len
        self._avgdl = total_len / max(n, 1)
        self._rel_path_to_idx = {p: i for i, p in enumerate(rel_paths)}
        self._s.fingerprints = fingerprints

        # Flush staged vector-cache additions in one batched encode pass.
        if vc_updates and self.vector_cache_enabled and self.vector_cache_manager is not None:
            try:
                self.vector_cache_manager.add_documents(vc_updates)
            except Exception as e:
                logger.debug("Failed to batch-add %s documents to vector cache: %s", len(vc_updates), e)
        return True

    def _walk_files(self) -> list[Path]:
        results: list[Path] = []
        # Use os.walk with directory pruning instead of rglob("*") to avoid
        # descending into hidden/vendor subtrees (node_modules, .git, etc.).
        # rglob visits every entry then filters, which is 70x+ slower on
        # repos with large vendor trees.
        self._index_truncated = False
        for root, dirs, files in os.walk(self.repo_root):
            # Prune in-place, then SORT so the descent is deterministic across
            # machines and source-prioritized (real code before tests/fixtures),
            # mirroring ._shared_utils._walk_repo_files. Without this, os.walk's
            # filesystem-enumeration order can starve entire subtrees (e.g. on
            # this very repo the first 600 hits were all tests/ and
            # external_llm/ got ZERO coverage).
            dirs[:] = sorted(
                [d for d in dirs if not _rag_should_skip_dir(d)],
                key=_walk_dir_sort_key,
            )
            files.sort()
            for fname in files:
                if fname.endswith(_WALK_SKIP_FILE_SUFFIXES):
                    continue
                p = Path(root) / fname
                if p.suffix.lower() not in _INDEXED_EXTS:
                    continue
                results.append(p)
                if len(results) >= _MAX_FILES:
                    self._index_truncated = True
                    return results
        return results

    @property
    def index_truncated(self) -> bool:
        """True if the corpus walk hit the file cap (index is incomplete)."""
        return self._index_truncated


# ── helpers ───────────────────────────────────────────────────────────────────

def _match_glob(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)
