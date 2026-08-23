"""Regression tests for ``CallGraphIndexer`` shared-instance index race.

Mirrors the ``RAGSearcher`` race fixed earlier. ``CallGraphIndexer`` is shared
**by reference** across in-process parallel subagents
(``ToolRegistry.clone_for_subagent`` assigns
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
            except BaseException as e:  # surface any failure
                errors.append(e)
                stop.set()
                return
            i += 1

    def invalidate_loop() -> None:
        i = 0
        while not stop.is_set() and i < 300:
            try:
                idx.invalidate()  # clears _nodes/_forward/_reverse
            except BaseException as e:
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

    assert not errors, f"concurrent access raised: {[type(e).__name__ for e in errors]}"


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
            assert f"Klass{k}.method_{k}" in idx._nodes, f"missing qualified node Klass{k}.method_{k} (lost update)"
            assert f"Klass{k}.helper_{k}" in idx._nodes, f"missing qualified node Klass{k}.helper_{k} (lost update)"
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


# ── Cooperative cancel regressions ───────────────────────────────────────────
# ``CallGraphIndexer`` is constructed by ``ToolRegistry`` with the agent's
# ``cancel_event``. The repo-wide ``ast.parse`` loop in ``build()`` (seconds on
# large repos — the lazy first-call cost of every graph query /
# ``analyze_change_impact``) must bail out promptly on ESC / Ctrl-C and leave NO
# partially-built index visible to the in-flight query (which would otherwise
# run ``_lookup_edges`` on a torn ``_forward``).


def test_build_with_pre_set_cancel_event_no_partial_index(tmp_path: Path) -> None:
    """A pre-set cancel_event short-circuits build() at the first checkpoint."""
    _seed_repo(tmp_path, n=10)
    ev = threading.Event()
    ev.set()
    idx = CallGraphIndexer(str(tmp_path), cancel_event=ev)
    idx.build()
    assert idx._built is False
    assert len(idx._nodes) == 0
    assert len(idx._forward) == 0
    assert len(idx._reverse) == 0


def test_build_mid_cancel_discards_partial_index(tmp_path: Path) -> None:
    """Cancelling mid-build (after some files were indexed into the dicts) must
    discard the partial index so the in-flight query never sees a torn ``_forward``.

    Deterministic via a ``_process_file_cached`` wrapper that sets the event
    after the 3rd file is indexed — guaranteeing the dicts are non-empty at
    the checkpoint.  (P2: build() routes per-file work through
    ``_process_file_cached``, the cache-tier entry point — ``_index_file`` is
    now only the compute half used by ``invalidate_files``.)
    """
    _seed_repo(tmp_path, n=10)
    ev = threading.Event()
    idx = CallGraphIndexer(str(tmp_path), cancel_event=ev)
    orig = idx._process_file_cached
    count = [0]

    def _count_then_cancel(path):
        orig(path)
        count[0] += 1
        if count[0] >= 3:
            ev.set()

    idx._process_file_cached = _count_then_cancel
    idx.build()
    assert count[0] >= 3, "wrapper must index >=3 files before tripping cancel"
    assert idx._built is False
    assert len(idx._nodes) == 0, "partial index must be discarded on cancel"
    assert len(idx._forward) == 0
    assert len(idx._reverse) == 0
    # Retry after clearing cancel: restore the real _process_file_cached (the
    # wrapper would otherwise keep re-tripping cancel since count[0] is
    # already >=3), then full-build succeeds (no stuck/half state).
    idx._process_file_cached = orig
    ev.clear()
    idx.build()
    assert idx._built is True
    assert len(idx._nodes) > 0


def test_build_without_cancel_event_unchanged(tmp_path: Path) -> None:
    """cancel_event=None (direct API callers, tests) builds exactly as before."""
    _seed_repo(tmp_path, n=10)
    idx = CallGraphIndexer(str(tmp_path))  # cancel_event defaults to None
    idx.build()
    assert idx._built is True
    assert len(idx._nodes) > 0


# ── design-chat per-turn mutation regressions ────────────────────────────────
# REGRESSION for a capture-at-construction bug: ``CallGraphIndexer`` is built by
# ``ToolRegistry`` from ``config``, and the design-chat REPL (asi.py) sets
# ``config.cancel_event`` PER TURN *after* the registry — and thus the indexer —
# was constructed with ``cancel_event=None``. A frozen construction-time value
# would leave ESC inert on every ``analyze_change_impact`` / ``query_dependency_
# graph`` first-call build (the exact interactive path ESC must protect). The
# indexer must read ``config.cancel_event`` FRESH at ``build()`` time — the same
# call-time fresh read vulture uses in analysis_tools. ``config=`` (not
# ``cancel_event=``) is the ToolRegistry wiring.


def test_build_honors_per_turn_config_mutation(tmp_path: Path) -> None:
    """ESC pressed via a config.cancel_event set AFTER construction is honored."""
    import types

    _seed_repo(tmp_path, n=10)
    cfg = types.SimpleNamespace(cancel_event=None)  # asi.py:6703 state at construction
    idx = CallGraphIndexer(str(tmp_path), config=cfg)  # ToolRegistry wiring (config=)
    assert idx._cancel_event is None  # construction-time value frozen
    assert idx._get_cancel_event() is None  # no event yet

    # Per-turn mutation (asi.py:8536): a live event lands on config AFTER build.
    ev = threading.Event()
    cfg.cancel_event = ev
    ev.set()  # ESC pressed
    assert idx._cancel_event is None  # STILL frozen None ...
    assert idx._get_cancel_event() is ev  # ... but fresh read sees live event
    idx.build()
    assert idx._built is False  # fresh read → cancel honored
    assert len(idx._nodes) == 0 and len(idx._forward) == 0

    # Clear ESC → same indexer now builds (fresh read reflects the clear).
    ev.clear()
    idx.build()
    assert idx._built is True
    assert len(idx._nodes) > 0


def test_build_with_config_no_event_builds_normally(tmp_path: Path) -> None:
    """config passed but cancel_event stays None → builds exactly as before."""
    import types

    _seed_repo(tmp_path, n=10)
    cfg = types.SimpleNamespace(cancel_event=None)
    idx = CallGraphIndexer(str(tmp_path), config=cfg)
    idx.build()
    assert idx._built is True
    assert len(idx._nodes) > 0


def test_cancelled_build_releases_rg_snapshot(tmp_path: Path, monkeypatch) -> None:
    """P2 (2026-08-12): an ESC-cancelled build must release the RG snapshot.

    The RG snapshot payload (~209MB in memory for the 42MB JSON) was only
    released on the normal build exit; the two ESC early-returns leaked it
    for the indexer's lifetime.  Deterministic via a ``_process_file_cached``
    wrapper that sets the event after the first file — the second file's
    checkpoint then cancels with the payload already loaded.
    """
    from external_llm.graph.repository_graph import RepositoryGraph

    repo = tmp_path
    for rel, src in (
        ("a.py", "def fa():\n    pass\n"),
        ("b.py", "def fb():\n    pass\n"),
    ):
        (repo / rel).write_text(src, encoding="utf-8")
    RepositoryGraph(str(repo)).build(collect_imported_names=True)  # write snapshot

    ev = threading.Event()
    idx = CallGraphIndexer(str(repo), cancel_event=ev)
    orig = CallGraphIndexer._process_file_cached

    def wrapper(self, path):
        orig(self, path)
        ev.set()  # cancel at the NEXT checkpoint, after the tier loaded

    monkeypatch.setattr(CallGraphIndexer, "_process_file_cached", wrapper)
    idx.build()
    assert not idx._built
    assert idx._rg_cache is None, "cancelled build must release the RG snapshot payload"
    assert idx._rg_cache_mtime_ns == 0
