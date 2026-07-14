"""Regression tests for ``CallGraphIndexer`` shared-instance index race.

Mirrors the ``RAGSearcher`` race fixed earlier. ``CallGraphIndexer`` is shared
**by reference** across in-process parallel subagents
(``ToolRegistry.clone_for_subagent`` / ``clone_with_filter`` both assign
``clone._call_graph = self._call_graph``). A subagent's write-success callback
(``_invalidate_cache_after_write`` -> ``CallGraphIndexer.invalidate``) clears the
index dicts (``_nodes`` / ``_forward`` / ``_reverse``) while a *sibling*
subagent's analysis tool (``analyze_change_impact`` / ``query_dependency_graph``
/ ``find_references``) traverses the very same dicts.

The ``RepositoryGraphFacade._rebuild_lock`` does **not** protect this path:
``get_callers``/``get_callees`` delegate straight to the indexer when it is set,
and the write callback calls ``invalidate()`` on the indexer directly — both
bypass the facade lock. With no lock on the indexer itself, under concurrency:

  * **Crash:** ``_lookup_edges`` suffix-fallback iterates ``index.items()`` while
    a concurrent ``invalidate()`` calls ``dict.clear()`` ->
    ``RuntimeError('dictionary changed size during iteration')`` (reproduced with
    a 4-reader + 2-invalidator stress under ``setswitchinterval(1e-6)``).
  * **Build/build race (silent):** two readers both observe ``_built=False`` and
    rebuild concurrently, interleaving dict writes -> lost updates / partial index.

The fix adds a single ``threading.RLock`` guarding ``build``/``invalidate``/
``_ensure_built`` (double-checked) and every reader (``get_callees``/
``get_callers``/``_lookup_edges``), with ``get_related_symbols`` holding the lock
across its whole multi-step read. RLock makes the nested ``_resolve_callees`` and
``get_callees``/``get_callers`` calls reentrant.

These tests assert (1) no crash under the mixed reader/invalidator stress, and
(2) the index rebuilds to a consistent, non-empty state afterwards.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from external_llm.agent.call_graph import CallGraphIndexer

_N_MODS = 25


def _seed_repo(root: Path, n: int = _N_MODS) -> None:
    """Seed ``n`` modules, each a class with inter-calling methods.

    Methods are stored under their *qualified* name (``Klass{k}.method_{k}``)
    while readers query the *bare* name (``method_{k}``) — this forces
    ``_lookup_edges`` into its suffix-fallback ``index.items()`` iteration, the
    exact path a concurrent ``invalidate().clear()`` corrupts.
    """
    for k in range(n):
        (root / f"mod{k}.py").write_text(
            f"class Klass{k}:\n"
            f"    def method_{k}(self):\n"
            f"        self.helper_{k}()\n"
            f"        return self.method_{k}_b()\n"
            f"\n"
            f"    def method_{k}_b(self):\n"
            f"        return self.helper_{k}()\n"
            f"\n"
            f"    def helper_{k}(self):\n"
            f"        return {k}\n"
        )


def _build_indexer(root: Path) -> CallGraphIndexer:
    idx = CallGraphIndexer(str(root))
    # Force an initial build so invalidate() takes the tear-down path.
    assert idx.get_callers("helper_0"), "seed repo must be indexable"
    return idx


def test_concurrent_read_and_invalidate_no_crash(tmp_path: Path) -> None:
    """4 readers (bare-name suffix-fallback) + 2 invalidators on the shared
    indexer must complete without raising. Pre-fix this raised
    ``RuntimeError('dictionary changed size during iteration')`` within seconds."""
    _seed_repo(tmp_path)
    idx = _build_indexer(tmp_path)

    errors: list[BaseException] = []
    stop = threading.Event()

    def read_loop() -> None:
        i = 0
        while not stop.is_set() and i < 500:
            try:
                k = i % _N_MODS
                # Bare names -> suffix-fallback iterates the shared dicts.
                idx.get_callees(f"method_{k}")
                idx.get_callers(f"helper_{k}")
                idx.get_related_symbols(f"method_{k}_b")
            except BaseException as e:  # noqa: BLE001 — surface any failure
                errors.append(e)
                stop.set()
                return
            i += 1

    def invalidate_loop() -> None:
        i = 0
        while not stop.is_set() and i < 300:
            try:
                idx.invalidate()  # clears _nodes/_forward/_reverse
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()
                return
            i += 1

    threads = [threading.Thread(target=read_loop) for _ in range(4)]
    threads += [threading.Thread(target=invalidate_loop) for _ in range(2)]
    # Minimize the GIL switch interval so threads interleave at (nearly) every
    # bytecode boundary. On lock-less code this deterministically surfaces the
    # dict-clear-vs-iteration race; under the lock it stays error-free
    # regardless. Restored in finally so other tests sharing the process are
    # unaffected.
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

    assert not errors, (
        f"concurrent access raised: {[type(e).__name__ for e in errors]}"
    )


def test_concurrent_build_no_lost_updates(tmp_path: Path) -> None:
    """Build/build race guard. Several threads racing the first ``_ensure_built``
    on a freshly-constructed (un-built) indexer must converge on a single,
    complete build rather than interleaving partial writes. After the race, the
    index must contain every seeded symbol and the known edges must resolve."""
    _seed_repo(tmp_path)
    idx = CallGraphIndexer(str(tmp_path))  # NOT pre-built

    def first_read() -> None:
        idx.get_related_symbols("method_0")

    prev_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=first_read) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()
    finally:
        sys.setswitchinterval(prev_interval)

    # Convergence: the index is built exactly once and is complete.
    with idx._lock:
        assert idx._built, "index not built after concurrent first-access"
        for k in range(_N_MODS):
            assert f"Klass{k}.method_{k}" in idx._nodes, (
                f"missing qualified node Klass{k}.method_{k} (lost update)"
            )
            assert f"Klass{k}.helper_{k}" in idx._nodes, (
                f"missing qualified node Klass{k}.helper_{k} (lost update)"
            )
    # Known edges resolve under the public API.
    callees = idx.get_callees("method_0")  # suffix-fallback
    callee_syms = {e.callee_symbol for e in callees}
    assert "Klass0.helper_0" in callee_syms, callees
    callers = idx.get_callers("helper_0")  # suffix-fallback
    caller_syms = {e.caller_symbol for e in callers}
    assert "Klass0.method_0" in caller_syms, callers


def test_invalidate_then_read_rebuilds_consistently(tmp_path: Path) -> None:
    """After invalidate(), the next read rebuilds and returns the full graph —
    not an empty/stale result (guards the ``_built`` flag tear-down ordering)."""
    _seed_repo(tmp_path)
    idx = _build_indexer(tmp_path)

    idx.invalidate()
    # Immediately after invalidation, a read must trigger a rebuild and return
    # real edges, not an empty list from cleared dicts.
    callees = idx.get_callees("method_0")
    assert callees, "post-invalidate read returned empty (rebuild did not fire)"
    assert {e.callee_symbol for e in callees}, callees
