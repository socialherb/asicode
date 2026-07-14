"""Regression tests for ``RAGSearcher`` shared-instance index race.

Covers a defect fixed in this change: ``RAGSearcher`` is shared **by reference**
across in-process parallel subagents (``ToolRegistry.clone_for_subagent`` /
``clone_with_filter`` both assign ``clone._rag_searcher = self._rag_searcher``).
A subagent's write-success callback (``_invalidate_cache_after_write`` →
``invalidate_files``) mutates the five parallel arrays
(``_rel_paths`` / ``_doc_token_counts`` / ``_doc_lengths`` / ``_doc_texts`` /
``_df``) — including ``_remove_doc_at``'s ``pop(idx)`` — while a *sibling*
subagent's ``find_relevant_files`` → ``_bm25_search`` traverses the very same
arrays.

``self._index_lock`` existed but was acquired **only** in ``_ensure_index``
(the initial build). Every other accessor was lock-free, so under concurrency:

  * **Crash:** ``_bm25_search`` captured ``idx`` values during its scoring loop
    that were invalidated by a concurrent ``_remove_doc_at`` ``pop`` by the time
    it indexed ``self._doc_texts[idx]`` / ``self._rel_paths[idx]`` →
    ``IndexError('list index out of range')`` (reproduced in ~6s with a 4-search
    + 2-invalidate stress).
  * **Silent corruption (worse):** ``pop(idx)`` shifts every later index down by
    one, so an in-flight reader's ``i``/``idx`` silently re-pairs ``_rel_paths``
    with a *different* document's token counts / text → a wrong file returned as
    "relevant". Non-deterministic pollution of parallel orchestration results.

The fix extends ``_index_lock`` to the whole
``invalidate_files`` mutation, the ``_bm25_search`` traversal (winning docs are
snapshotted under the lock, snippet extraction runs lock-free on immutable
strings), and the array-lookup portion of ``_vector_search``. ``_remove_doc_at``
remains a caller-must-hold-lock private helper (invoked only from
``invalidate_files``) to avoid non-reentrant ``threading.Lock`` deadlock.

These tests assert (1) no crash under the same mixed search/invalidate stress,
and (2) the parallel arrays stay mutually consistent — including the new
running-total ``_total_doc_len`` invariant and per-doc path↔text alignment — so
the silent-corruption mode cannot recur.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from external_llm.agent.rag_searcher import RAGSearcher

_N_DOCS = 30


def _seed_repo(root: Path, n: int = _N_DOCS) -> None:
    """Seed ``n`` indexable files, each carrying a path-stable unique marker.

    The marker ``unique_token_doc{k}`` is rewritten on every mutation so it
    lets us verify path↔text alignment after concurrent invalidation regardless
    of how many times a file was updated/deleted/recreated.
    """
    for k in range(n):
        _write_doc(root, k, payload=0)


def _write_doc(root: Path, k: int, payload: int) -> None:
    (root / f"doc{k}.py").write_text(
        f"# unique_token_doc{k}\n"
        f"def function_{k}(x, y):\n"
        f"    return x + y + {payload}\n"
    )


def _build_searcher(root: Path) -> RAGSearcher:
    s = RAGSearcher(str(root), vector_cache_enabled=False)
    # Force an initial build so invalidate_files takes the incremental path.
    assert s.find_relevant_files("function", top_k=5), "seed repo must be searchable"
    return s


def test_concurrent_search_and_invalidate_no_crash(tmp_path: Path) -> None:
    """4 searchers + 2 writers (update/delete/add via invalidate_files) on the
    shared searcher must complete without raising. Pre-fix this raised
    IndexError within seconds."""
    _seed_repo(tmp_path)
    searcher = _build_searcher(tmp_path)

    errors: list[BaseException] = []
    stop = threading.Event()

    def search_loop() -> None:
        i = 0
        while not stop.is_set() and i < 500:
            try:
                k = i % _N_DOCS
                res = searcher.find_relevant_files(f"function_{k}", top_k=(i % 4) + 1)
                for r in res:
                    if not r.file.startswith("doc"):
                        raise AssertionError(f"unexpected result file: {r.file}")
            except BaseException as e:  # noqa: BLE001 — surface any failure
                errors.append(e)
                stop.set()
                return
            i += 1

    def write_loop() -> None:
        i = 0
        while not stop.is_set() and i < 300:
            try:
                k = i % _N_DOCS
                target = tmp_path / f"doc{k}.py"
                if i % 3 == 0:
                    try:
                        target.unlink()  # exercise _remove_doc_at (delete path)
                    except FileNotFoundError:
                        pass  # sibling writer may have unlinked first
                else:
                    _write_doc(tmp_path, k, payload=i)  # exercise UPDATE / NEW path
                searcher.invalidate_files([f"doc{k}.py"])
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()
                return
            i += 1

    threads = [threading.Thread(target=search_loop) for _ in range(4)]
    threads += [threading.Thread(target=write_loop) for _ in range(2)]
    # Minimize the GIL switch interval so threads interleave at (nearly) every
    # bytecode boundary. On lock-less code this deterministically surfaces the
    # array-mutation race (6/6 reproduction of IndexError within seconds); under
    # the index lock it stays error-free regardless. Restored in finally so
    # other tests sharing the process are unaffected.
    prev_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "thread did not finish within timeout"
    finally:
        sys.setswitchinterval(prev_interval)

    assert not errors, f"concurrent access raised: {[type(e).__name__ for e in errors]}"


def test_index_arrays_consistent_under_concurrent_invalidate(tmp_path: Path) -> None:
    """Silent-corruption guard. After concurrent readers + a mutator that
    repeatedly deletes/rewrites docs, the five parallel arrays and derived
    stats must remain mutually consistent, and each indexed doc's text must
    still contain the unique marker matching its path (no path↔doc misalignment
    from index-shifting pops)."""
    _seed_repo(tmp_path)
    searcher = _build_searcher(tmp_path)

    done = threading.Event()

    def mutator() -> None:
        i = 0
        while not done.is_set() and i < 400:
            k = i % _N_DOCS
            target = tmp_path / f"doc{k}.py"
            if i % 4 == 0:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass  # concurrent reader/writer may have removed it first
            else:
                _write_doc(tmp_path, k, payload=i)
            searcher.invalidate_files([f"doc{k}.py"])
            i += 1

    def read_loop() -> None:
        for _ in range(5):
            for k in range(_N_DOCS):
                searcher.find_relevant_files(f"unique_token_doc{k}", top_k=3)

    readers = [threading.Thread(target=read_loop) for _ in range(4)]
    mut = threading.Thread(target=mutator)
    for t in readers:
        t.start()
    mut.start()
    for t in readers:
        t.join(timeout=30)
        assert not t.is_alive()
    done.set()
    mut.join(timeout=30)
    assert not mut.is_alive()

    # Quiesce: a final search guarantees the incremental updates are applied.
    searcher.find_relevant_files("function", top_k=5)

    with searcher._index_lock:
        n = searcher._n_docs
        assert len(searcher._rel_paths) == n, "rel_paths length drift"
        assert len(searcher._doc_token_counts) == n, "doc_token_counts length drift"
        assert len(searcher._doc_lengths) == n, "doc_lengths length drift"
        assert len(searcher._doc_texts) == n, "doc_texts length drift"
        # Running-total invariant (replaces per-file sum(); must stay exact).
        assert searcher._total_doc_len == sum(searcher._doc_lengths), (
            f"total_doc_len={searcher._total_doc_len} != "
            f"sum={sum(searcher._doc_lengths)}"
        )
        expected_avgdl = searcher._total_doc_len / max(n, 1)
        assert abs(searcher._avgdl - expected_avgdl) < 1e-9, (
            f"avgdl={searcher._avgdl} != expected {expected_avgdl}"
        )
        # path↔text alignment: docN.py's text must contain unique_token_docN.
        for idx, path in enumerate(searcher._rel_paths):
            assert path.endswith(".py"), f"unexpected path: {path}"
            marker = "unique_token_" + path[: -len(".py")]
            assert marker in searcher._doc_texts[idx], (
                f"path↔doc misalignment at idx {idx}: path={path!r} "
                f"text lacks {marker!r}"
            )


def test_remove_doc_at_requires_caller_lock(tmp_path: Path) -> None:
    """``_remove_doc_at`` is documented as caller-must-hold-lock; it must not
    acquire the lock itself (would deadlock under the non-reentrant Lock when
    called from within invalidate_files). Verify the running-total delta is
    applied correctly when invoked under the lock."""
    _seed_repo(tmp_path)
    searcher = _build_searcher(tmp_path)

    with searcher._index_lock:
        before = searcher._n_docs
        total_before = searcher._total_doc_len
        removed_len = searcher._doc_lengths[0]
        searcher._remove_doc_at(0)

    assert searcher._n_docs == before - 1
    assert searcher._total_doc_len == total_before - removed_len
    assert abs(
        searcher._avgdl - (searcher._total_doc_len / max(searcher._n_docs, 1))
    ) < 1e-9


def test_stale_result_not_recached_after_invalidate_race(tmp_path: Path) -> None:
    """A searcher that read the PRE-mutation index must NOT re-cache its stale
    result after a concurrent ``invalidate_files`` clears the cache.

    Reproduces the stale-cache window deterministically: the searcher reads the
    old index, then (paused) an invalidator mutates + clears the cache, then the
    searcher reaches its cache-write step. Without the generation guard the
    searcher's already-computed stale result is written back and served for the
    5-min TTL; with the guard the generation mismatch discards the write.
    """
    import threading

    _seed_repo(tmp_path, n=5)
    # Make doc0 uniquely findable, then build the index.
    (tmp_path / "doc0.py").write_text("alpha_token_xyz marker\n")
    searcher = _build_searcher(tmp_path)

    query = "alpha_token_xyz"
    cache_key = searcher._make_cache_key(query, 5, None)
    # Drain the warm cache so the next search is a genuine miss.
    with searcher._search_cache_lock:
        searcher._search_cache.clear()

    search_read_done = threading.Event()
    allow_search_finish = threading.Event()
    real_bm25 = searcher._bm25_search

    def _pausing_bm25(q, top_k, file_glob=None):
        res = real_bm25(q, top_k, file_glob)  # reads the (old) index under lock
        search_read_done.set()                 # searcher has read the old index
        allow_search_finish.wait(timeout=5.0)  # block until invalidation lands
        return res

    searcher._bm25_search = _pausing_bm25

    def _search():
        searcher.find_relevant_files(query)

    t = threading.Thread(target=_search)
    t.start()
    assert search_read_done.wait(timeout=5.0), "searcher did not reach the read step"

    # Mutate doc0 so the searcher's already-computed result is now stale, then
    # invalidate (bumps generation under the lock + clears the cache).
    (tmp_path / "doc0.py").write_text("beta_token_zzz completely_different\n")
    searcher.invalidate_files(["doc0.py"])

    allow_search_finish.set()  # release the searcher to its cache-write step
    t.join(timeout=5.0)
    assert not t.is_alive(), "searcher thread hung"

    with searcher._search_cache_lock:
        assert cache_key not in searcher._search_cache, (
            "stale pre-mutation result was re-cached after the clear "
            "(generation guard missing)"
        )


def test_generation_compare_holds_cache_lock(tmp_path: Path) -> None:
    """The generation re-check at the cache-WRITE site must run UNDER
    ``_search_cache_lock`` (not before it), else a micro-window lets an
    invalidator bump+clear between the passing compare and lock acquisition,
    after which the searcher records a stale result that survives the 5-min TTL.

    Deterministic injection: wrap ``_search_cache_lock`` so the cache-WRITE
    acquire (the 2nd acquire in ``find_relevant_files``; the 1st is the cache
    read, which is a miss here) bumps ``_index_generation`` and clears the cache
    via the REAL lock immediately before acquiring it -- i.e. exactly in the
    compare->acquire gap of the vulnerable layout. With the guard correctly
    inside the lock the subsequent compare sees the bumped generation and skips.
    """
    _seed_repo(tmp_path, n=5)
    (tmp_path / "doc0.py").write_text("alpha_token_xyz marker\n")
    searcher = _build_searcher(tmp_path)

    query = "alpha_token_xyz"
    cache_key = searcher._make_cache_key(query, 5, None)
    with searcher._search_cache_lock:        # drain -> the 1st acquire (read) is a miss
        searcher._search_cache.clear()

    real_lock = searcher._search_cache_lock
    fired = []

    class _InjectingLock:
        def __init__(self):
            self._n = 0

        def __enter__(self):
            self._n += 1
            # 1 = cache-read (miss), 2 = cache-WRITE (the vulnerable site).
            if self._n == 2:
                # Bump generation + clear through the REAL lock (bypassing this
                # wrapper to avoid re-entrancy) -- exactly the compare->acquire
                # gap under the OLD layout.
                searcher._index_generation += 1
                with real_lock:
                    searcher._search_cache.clear()
                fired.append(True)
            real_lock.__enter__()

        def __exit__(self, *a):
            real_lock.__exit__(*a)

    searcher._search_cache_lock = _InjectingLock()
    try:
        searcher.find_relevant_files(query)
    finally:
        searcher._search_cache_lock = real_lock

    assert fired, "injection did not fire at the cache-write acquire"
    with real_lock:
        assert cache_key not in searcher._search_cache, (
            "stale result recorded AFTER generation bump+clear - the generation "
            "compare is not atomic w.r.t. the cache clear (must run inside "
            "_search_cache_lock)"
        )


def test_path_to_index_map_stays_consistent(tmp_path: Path) -> None:
    """The ``_rel_path_to_idx`` mirror must match ``_rel_paths`` after builds and
    after add/remove/update invalidations, so the O(1) _vector_search lookup and
    the invalidate_files lookup never mis-resolve a path to the wrong doc."""
    _seed_repo(tmp_path, n=6)
    searcher = _build_searcher(tmp_path)

    def _check():
        assert searcher._rel_path_to_idx == {
            p: i for i, p in enumerate(searcher._rel_paths)
        }

    _check()
    # Remove doc2.
    (tmp_path / "doc2.py").unlink()
    searcher.invalidate_files(["doc2.py"])
    _check()
    assert searcher._rel_path_to_idx.get("doc2.py") is None
    # Add a new doc.
    (tmp_path / "doc_new.py").write_text("def fresh_func():\n    pass\n")
    searcher.invalidate_files(["doc_new.py"])
    _check()
    assert searcher._rel_path_to_idx.get("doc_new.py") is not None
    # Update doc0.
    (tmp_path / "doc0.py").write_text("def rewritten():\n    return 42\n")
    searcher.invalidate_files(["doc0.py"])
    _check()
