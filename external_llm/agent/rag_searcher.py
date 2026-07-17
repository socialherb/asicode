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
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

from .config.thresholds import config as _cfg
from .performance_metrics import get_global_collector
from .rag_configs import CodeTokenizer

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_INDEXED_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".rb", ".php",
    ".cs", ".swift", ".kt", ".cpp", ".c", ".h",
    ".md", ".toml", ".yaml", ".yml",
}

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "env", ".tox", "dist", "build",
    ".eggs", "migrations", "worktrees",
}

_MAX_FILES = _cfg.counts.RAG_MAX_FILES
_MAX_FILE_CHARS = _cfg.lines.RAG_FILE_CHARS

# BM25 tuning
_K1 = 1.5
_B = 0.75

# (Stopwords moved to ``CodeTokenizer`` in ``rag_configs.py`` — removed from
# this module.)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    file: str           # relative path from repo root
    score: float
    snippet: str        # most relevant excerpt (~120 chars)
    line: int = 0       # approximate line of best match


# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Module-level singleton — replaces ad-hoc ``_split_camel`` + ``_tokenize``
# regex functions and ``_STOP`` frozenset.  Handles CamelCase, snake_case,
# and stop-word filtering consistently with the rest of the codebase.
_TOKENIZER = CodeTokenizer()


# ── BM25 core ─────────────────────────────────────────────────────────────────

def _bm25_score(
    query_tokens: list[str],
    doc_token_counts: dict[str, int],
    doc_len: int,
    df: dict[str, int],
    n_docs: int,
    avgdl: float,
) -> float:
    if doc_len == 0 or avgdl == 0:
        return 0.0
    score = 0.0
    for qt in query_tokens:
        tf = doc_token_counts.get(qt, 0)
        if tf == 0:
            continue
        idf = math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
        tf_norm = tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * doc_len / avgdl))
        score += idf * tf_norm
    return score


# ── Snippet extraction ────────────────────────────────────────────────────────

def _extract_snippet(text: str, query_tokens: list[str], window: int = 120) -> tuple[str, int]:
    """Return (snippet, 1-indexed line) for the best-matching line."""
    lines = text.splitlines()
    if not lines:
        return "", 1
    q_set = set(query_tokens)
    best_line, best_score = 0, -1
    for i, line in enumerate(lines):
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
    """

    def __init__(
        self,
        repo_root: str,
        vector_cache_enabled: bool = True,
        cancel_event: Optional[threading.Event] = None,
        config: Any = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self._built = False
        # Cooperative cancel: hold config (NOT the event value) and read
        # config.cancel_event FRESH in _build_index via _get_cancel_event.
        # The design-chat REPL mutates config.cancel_event PER TURN (asi.py)
        # AFTER this searcher is constructed with cancel_event=None; a captured
        # value would freeze None and leave ESC inert during the multi-second
        # first find_relevant_files build — the exact interactive path ESC must
        # protect. An explicit cancel_event arg (tests / direct callers) wins.
        self._cancel_event = cancel_event
        self._config = config
        self._index_lock = threading.Lock()
        self.vector_cache_enabled = vector_cache_enabled
        self.vector_cache_manager = None
        if vector_cache_enabled:
            try:
                from .vector_cache import HAS_FAISS, HAS_NUMPY, HAS_SENTENCE_TRANSFORMERS, VectorCacheManager
                if HAS_SENTENCE_TRANSFORMERS and HAS_NUMPY and HAS_FAISS:
                    self.vector_cache_manager = VectorCacheManager(".asicode/vector_cache")
                else:
                    logger.warning("Vector cache dependencies not fully installed, disabling")
                    self.vector_cache_enabled = False
                    self.vector_cache_manager = None
            except ImportError as e:
                logger.warning(f"Vector cache import failed: {e}, disabling")
                self.vector_cache_enabled = False
                self.vector_cache_manager = None
        # Per-document data
        self._rel_paths: list[str] = []
        self._doc_token_counts: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._doc_texts: list[str] = []
        # {rel_path: idx} mirror of ``_rel_paths`` → O(1) lookup in the read
        # hot-path (_vector_search resolves each result's doc text by path).
        # Rebuilt in _build_index and at the end of each invalidate_files batch
        # (under _index_lock, alongside the parallel arrays). Maintaining it
        # incrementally during a removal batch is unsafe (``list.pop`` shifts
        # every later index), so it is rebuilt wholesale once per batch instead.
        self._rel_path_to_idx: dict[str, int] = {}
        # Corpus-level stats
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        # Running total of doc lengths → O(1) avgdl. Maintained under
        # ``_index_lock`` alongside the parallel arrays; replaces the per-file
        # ``sum(self._doc_lengths)`` O(n) recompute in invalidate_files.
        self._total_doc_len: int = 0
        # Monotonic generation counter, bumped under ``_index_lock`` on every
        # invalidate_files mutation. A searcher captures it at cache-miss start
        # and, before writing its result to _search_cache, checks it is
        # unchanged — if an invalidation raced in between (the searcher read the
        # OLD index but finishes after the cache was cleared), the write is
        # discarded so a stale result can never be re-cached for the 5-min TTL.
        self._index_generation: int = 0
        # Search cache: key -> (timestamp, results). Bounded LRU + TTL,
        # thread-safe (matches ToolResultCache / run_scoped_graph_cache pattern).
        self._search_cache: "OrderedDict[str, tuple[float, list[SearchResult]]]" = OrderedDict()
        self._search_cache_lock = threading.Lock()
        self._search_cache_max = 256  # bound (LRU eviction) — match ToolResultCache

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
                timestamp, results = self._search_cache[cache_key]
                if now - timestamp < 300:  # 5 minute TTL
                    self._search_cache.move_to_end(cache_key)  # refresh LRU position
                    _cached = results
                else:
                    del self._search_cache[cache_key]  # expired
        if _cached is not None:
            # Cache hit
            get_global_collector().record_rag_cache(True)
            return _cached
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
                self._search_cache[cache_key] = (time.monotonic(), results)
                if len(self._search_cache) >= self._search_cache_max:
                    self._search_cache.popitem(last=False)

        # Record search time
        search_elapsed_ms = (time.monotonic() - search_start) * 1000
        get_global_collector().record_rag_search(search_elapsed_ms)

        return results

    def invalidate_files(self, changed_paths: list[str]) -> None:
        """Incrementally update index for changed/new/deleted files only.

        Updates only affected docs while keeping the rest of the index intact.

        Thread-safety / critical-section discipline: split into two phases so a
        bulk invalidation (branch switch / large patch touching dozens of files)
        never blocks parallel subagents' searches on disk + tokenize work.

          * **Phase 1 (outside ``_index_lock``)** — read each changed file and
            tokenize it. This is pure filesystem + CPU work and is by far the
            expensive part; running it under the lock would stall every
            concurrent ``_bm25_search`` / ``_vector_search`` for the whole
            duration. The result is staged in a ``norm_path -> (text, tokens)``
            map keyed only by path (no array index), so phase 2 can re-resolve
            the index under the lock.
          * **Phase 2 (inside ``_index_lock``)** — only the parallel-array
            mutations: locate each existing entry (``list.index`` races with a
            concurrent ``_remove_doc_at`` otherwise), subtract/add df
            contributions, and append/replace/remove. This instance is shared
            across in-process parallel subagents
            (``ToolRegistry.clone_for_subagent`` shares it by reference), so a
            subagent's write-success callback invoking invalidate_files while a
            sibling searches would otherwise corrupt the arrays (IndexError, or
            worse, a silent path↔document misalignment as ``_remove_doc_at``'s
            ``pop`` shifts indices).

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
        # Phase 1 (outside the lock): read + tokenize each changed file. Files
        # that no longer exist, are not indexable, are unreadable, or yield no
        # tokens are simply absent from `prepared` — phase 2 then treats them as
        # removals (if previously indexed) or no-ops (if never indexed),
        # matching the previous in-lock semantics exactly.
        prepared: dict[str, tuple[str, list[str]]] = {}
        for rel_path in changed_paths:
            norm_path = rel_path.strip().lstrip("/")
            abs_path = self.repo_root / norm_path

            file_exists = abs_path.is_file()
            is_indexable = (
                file_exists
                and abs_path.suffix.lower() in _INDEXED_EXTS
                and not any(
                    part.startswith(".") or part in _SKIP_DIRS
                    for part in Path(norm_path).parts
                )
            )
            if not is_indexable:
                continue
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_FILE_CHARS:
                    text = text[:_MAX_FILE_CHARS].rsplit("\n", 1)[0]
                path_text = norm_path.replace("/", " ").replace("\\", " ").replace(".", " ")
                tokens = _TOKENIZER.tokenize(text + " " + path_text)
                if tokens:
                    prepared[norm_path] = (text, tokens)
            except Exception:
                pass  # non-critical — never block execution; treated as unindexable

        # Defer vector-cache updates until after releasing the index lock.
        vc_updates: list[tuple[str, str]] = []
        # Phase 2 (inside the lock): apply only the array mutations.
        with self._index_lock:
            if not self._built:
                # Index not built yet, nothing to incrementally update.
                return

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
                        self._remove_doc_at(existing_idx)

                elif prep is not None and self._n_docs < _MAX_FILES:
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

            n_after = self._n_docs
            # Bump the generation so an in-flight searcher that already read the
            # pre-mutation index discards its (now-stale) result rather than
            # re-caching it after the clear below. Rebuild the path→idx mirror to
            # match the mutated arrays (rebuilt wholesale — list.pop shifts every
            # later index, so incremental maintenance within the loop is unsafe).
            self._index_generation += 1
            self._rel_path_to_idx = {
                _p: _i for _i, _p in enumerate(self._rel_paths)
            }

        # Clear search cache AFTER the index mutation completes (and outside the
        # index lock). This alone is NOT sufficient: a searcher that already read
        # the PRE-mutation index (and is now past the lock, in the merge/write
        # phase) would re-cache its stale result AFTER this clear. The
        # generation check at the searcher's cache-write site closes that window
        # — such a searcher sees the bumped generation and discards its write.
        with self._search_cache_lock:
            self._search_cache.clear()

        # Flush deferred vector-cache updates outside the index lock (embedding
        # I/O must not block concurrent searches).
        if self.vector_cache_enabled and self.vector_cache_manager is not None:
            for vc_path, vc_text in vc_updates:
                try:
                    self.vector_cache_manager.add_document(vc_path, vc_text)
                except Exception:
                    pass  # non-critical — never block execution

        logger.debug("RAG incremental update: %d files, index now %d docs", len(changed_paths), n_after)

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
            for i, rel in enumerate(rel_paths):
                if file_glob and not _match_glob(rel, file_glob):
                    continue
                s = _bm25_score(
                    q_tokens,
                    doc_tcs[i],
                    doc_lens[i],
                    df,
                    n_docs,
                    avgdl,
                )
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
        with self._index_lock:
            if self._built:
                return
            t0 = time.monotonic()
            completed = self._build_index()
            if not completed:
                # Cancelled mid-build: leave _built False so the next query
                # retries.  _build_index accumulates into *local* lists/dicts
                # and only commits to self._* at the very end, so instance state
                # is pristine on cancel (no half-populated arrays to reset).
                # (Side note: vector_cache_manager.add_document is a per-file
                # side-effect inside the loop and may be partially written on
                # cancel; it is incremental/idempotent and decoupled from the
                # BM25 path, so this is safe.)
                return
            elapsed = time.monotonic() - t0
            logger.debug("RAG index built: %d docs in %.2fs", self._n_docs, elapsed)
            self._built = True

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
        """
        if self._cancel_event is not None:
            return self._cancel_event
        return getattr(self._config, "cancel_event", None)

    def _build_index(self) -> bool:
        """Build the BM25 index. Returns True if completed, False if cancelled.

        Accumulates into *local* lists/dicts and only commits to ``self._*`` at
        the very end, so a mid-build cancel leaves instance state pristine —
        ``_built`` stays False and the next query re-runs this method from
        scratch.  ``vector_cache_manager.add_document`` is the one side-effect
        inside the loop; on cancel it may be partially written, but it is
        incremental/idempotent and decoupled from the BM25 path.  Hold
        ``_index_lock`` for the whole build (the caller does) so no reader
        races the final commit.
        """
        rel_paths: list[str] = []
        doc_tcs: list[dict[str, int]] = []
        doc_lens: list[int] = []
        doc_texts: list[str] = []
        df: dict[str, int] = {}
        total_len = 0

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
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_FILE_CHARS:
                    text = text[:_MAX_FILE_CHARS].rsplit("\n", 1)[0]
                rel = str(fpath.relative_to(self.repo_root))
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

                # Add to vector cache if enabled
                if self.vector_cache_enabled and self.vector_cache_manager is not None:
                    try:
                        self.vector_cache_manager.add_document(rel, text)
                    except Exception as e:
                        logger.debug(f"Failed to add document {rel} to vector cache: {e}")
            except (AttributeError, TypeError):
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
        return True

    def _walk_files(self) -> list[Path]:
        results: list[Path] = []
        # Use os.walk with directory pruning instead of rglob("*") to avoid
        # descending into hidden/vendor subtrees (node_modules, .git, etc.).
        # rglob visits every entry then filters, which is 70x+ slower on
        # repos with large vendor trees.
        for root, dirs, files in os.walk(self.repo_root):
            # Prune in-place: skip hidden dirs and known vendor/noise dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in _SKIP_DIRS
            ]
            for fname in files:
                p = Path(root) / fname
                if p.suffix.lower() not in _INDEXED_EXTS:
                    continue
                results.append(p)
                if len(results) >= _MAX_FILES:
                    return results
        return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _match_glob(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)
