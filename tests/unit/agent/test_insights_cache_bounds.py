"""Regression tests for bounded insights caches + WeakValueDictionary lock registry.

Guards the 6th instance of the per-repo module-level leak class fixed across
this repo (``_searcher_cache`` / ``_RECORD_COUNT_CACHE`` / ``_GRAPH_FACADE_CACHE``
/ ``_ACTIVE_ENGINES`` / ``_runners``). Three module-level dicts in
``insights_manager`` grew without bound under a long-lived webapp visiting many
repos:

  * ``_INSIGHTS_THREAD_LOCKS``  — per-repo RLock registry (heaviest; fixed via
    ``weakref.WeakValueDictionary``, the pattern proven in
    ``orchestrator._file_locks``).
  * ``_ARCHIVE_WRITE_VERSIONS`` — per-path monotonic write-version counter
    (lockstep-evicted with ``_ARCHIVE_PARSED_CACHE`` / ``_ARCHIVE_ANALYZED_CACHE``
    via ``_archive_capped_put``).
  * ``_ACTIVE_WRITE_VERSIONS``  — same, for the active insights file.

Mutation guards: if ``_INSIGHTS_THREAD_LOCKS`` is reverted to a plain ``dict``,
the GC-after-release assertion fails (entry survives). If lockstep eviction in
``_archive_capped_put`` is removed, the version-dict-bounded assertion fails.
"""
from __future__ import annotations

import gc
import os
import threading
import time

import pytest

from external_llm.agent.insights_manager import (
    _ACTIVE_CONTENT_CACHE,
    _ACTIVE_WRITE_VERSIONS,
    _ARCHIVE_CACHE_MAX_ENTRIES,
    _ARCHIVE_PARSED_CACHE,
    _ARCHIVE_WRITE_VERSIONS,
    _INSIGHTS_THREAD_LOCKS,
    _active_invalidate,
    _parsed_archive_cached,
    append_entries_to_archive,
    atomic_write_text,
    insights_archive_path,
    insights_path,
    insights_write_lock,
    load_active_insights_cached,
    parse_insights,
)


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Isolation: clear all module-level caches before AND after each test."""
    for d in (
        _ARCHIVE_PARSED_CACHE,
        _ARCHIVE_WRITE_VERSIONS,
        _ACTIVE_CONTENT_CACHE,
        _ACTIVE_WRITE_VERSIONS,
    ):
        d.clear()
    yield
    for d in (
        _ARCHIVE_PARSED_CACHE,
        _ARCHIVE_WRITE_VERSIONS,
        _ACTIVE_CONTENT_CACHE,
        _ACTIVE_WRITE_VERSIONS,
    ):
        d.clear()


def _make_repo(tmp_path, name: str) -> str:
    repo = str(tmp_path / name)
    os.makedirs(os.path.join(repo, ".asicode"))
    return repo


def _archive_entry(text: str):
    preamble, entries = parse_insights(f"### [bug] 2025-01-15 10:00\n{text}\n\n")
    assert entries, "fixture content must parse to at least one entry"
    return entries


# ── _INSIGHTS_THREAD_LOCKS: WeakValueDictionary ──────────────────────────────


class TestInsightsLockRegistry:
    """The per-repo RLock registry must be a WeakValueDictionary so idle locks
    are GC'd — a plain dict leaks one RLock per visited repo forever."""

    def test_lock_entry_gone_after_release(self, tmp_path):
        """Core regression: after exiting ``insights_write_lock``, the
        WeakValueDictionary entry is GC'd. Reverting to a plain ``dict`` makes
        this assertion FAIL (entry survives → unbounded leak)."""
        repo = _make_repo(tmp_path, "repo_a")
        key = os.path.abspath(repo)

        assert key not in _INSIGHTS_THREAD_LOCKS  # cold

        with insights_write_lock(repo):
            assert key in _INSIGHTS_THREAD_LOCKS  # alive while in use

        gc.collect()
        assert key not in _INSIGHTS_THREAD_LOCKS, (
            "lock leaked: WeakValueDictionary entry survived after release "
            "(was _INSIGHTS_THREAD_LOCKS reverted to a plain dict?)"
        )

    def test_reentry_within_context_reuses_same_lock(self, tmp_path):
        """Re-entrant nesting (``enforce_budget_by_demotion`` →
        ``append_entries_to_archive``) must acquire the SAME RLock, not create a
        second one. The outer ``with`` frame holds the strong ref that keeps the
        weak entry alive for the nested call."""
        repo = _make_repo(tmp_path, "repo_b")
        key = os.path.abspath(repo)
        captured = {}

        with insights_write_lock(repo):
            outer = _INSIGHTS_THREAD_LOCKS.get(key)
            captured["outer"] = outer
            with insights_write_lock(repo):  # re-entrant (same thread)
                inner = _INSIGHTS_THREAD_LOCKS.get(key)
                captured["inner"] = inner

        assert captured["outer"] is captured["inner"], (
            "re-entrant call created a DIFFERENT lock — nesting would deadlock "
            "or fail to serialize"
        )

    def test_concurrent_threads_for_same_repo_serialize(self, tmp_path):
        """Two threads entering ``insights_write_lock`` for the same repo must
        share the SAME RLock and therefore serialize. If the registry returned
        distinct locks per thread, both would enter concurrently."""
        repo = _make_repo(tmp_path, "repo_c")
        log: list[str] = []
        log_lock = threading.Lock()

        def worker(name: str):
            with insights_write_lock(repo):
                with log_lock:
                    log.append(f"{name}-enter")
                time.sleep(0.08)  # hold the lock to force overlap
                with log_lock:
                    log.append(f"{name}-exit")

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Serialized: one worker fully completes before the other enters.
        # Concurrent (broken) would interleave: A-enter, B-enter, …
        assert log in (
            ["A-enter", "A-exit", "B-enter", "B-exit"],
            ["B-enter", "B-exit", "A-enter", "A-exit"],
        ), f"threads were NOT serialized (shared RLock missing?): {log}"

    def test_distinct_repos_get_distinct_locks(self, tmp_path):
        """Different repos must get different locks so they do NOT serialize
        against each other (independent repos)."""
        repo_a = _make_repo(tmp_path, "da")
        repo_b = _make_repo(tmp_path, "db")
        captured = {}
        with insights_write_lock(repo_a):
            captured["a"] = _INSIGHTS_THREAD_LOCKS.get(os.path.abspath(repo_a))
        # repo_a's lock is GC'd after release; repo_b gets its own
        gc.collect()
        with insights_write_lock(repo_b):
            captured["b"] = _INSIGHTS_THREAD_LOCKS.get(os.path.abspath(repo_b))

        assert captured["a"] is not None
        assert captured["b"] is not None
        assert captured["a"] is not captured["b"]


# ── _ARCHIVE_WRITE_VERSIONS: lockstep eviction ───────────────────────────────


class TestArchiveVersionLockstep:
    """The archive write-version dict must be bounded in LOCKSTEP with the
    content caches: ``_archive_capped_put`` pops the version entry for an evicted
    path, preventing a stale ``version==0`` reset from matching a surviving
    cache entry."""

    def test_version_dict_bounded_with_content_cache(self, tmp_path):
        """After visiting >cap repos, BOTH the parsed-cache and the version dict
        stay ≤ cap. Removing lockstep eviction makes the version-dict assertion
        FAIL (it grows to N entries)."""
        cap = _ARCHIVE_CACHE_MAX_ENTRIES  # 8
        n = cap + 3  # 11 repos

        for i in range(n):
            repo = _make_repo(tmp_path, f"r{i}")
            # Write (bumps version via _archive_invalidate) then read (populates
            # content cache via _archive_capped_put → lockstep eviction).
            append_entries_to_archive(repo, _archive_entry(f"entry {i}"))
            _parsed_archive_cached(repo)

        assert len(_ARCHIVE_PARSED_CACHE) <= cap, "content cache unbounded"
        assert len(_ARCHIVE_WRITE_VERSIONS) <= cap, (
            "_ARCHIVE_WRITE_VERSIONS grew unbounded — lockstep eviction in "
            "_archive_capped_put is missing"
        )

    def test_evicted_repo_version_is_gone(self, tmp_path):
        """The oldest repo's version entry is popped when its content-cache
        entry is FIFO-evicted."""
        cap = _ARCHIVE_CACHE_MAX_ENTRIES
        repos = []
        for i in range(cap + 2):
            repo = _make_repo(tmp_path, f"e{i}")
            repos.append(repo)
            append_entries_to_archive(repo, _archive_entry(f"e{i}"))
            _parsed_archive_cached(repo)

        # First repos were FIFO-evicted; their version entries must be gone.
        evicted_path = insights_archive_path(repos[0])
        assert evicted_path not in _ARCHIVE_WRITE_VERSIONS, (
            "evicted repo's version entry survived — lockstep pop missing"
        )
        # The most-recent repo's version must still be present.
        live_path = insights_archive_path(repos[-1])
        assert live_path in _ARCHIVE_WRITE_VERSIONS

    def test_reaccess_after_eviction_no_stale_hit(self, tmp_path):
        """After eviction + re-access, the version resets to 0 and the content
        is re-read fresh (no stale hit). This is the safe worst-case of lockstep:
        a false miss, never a stale hit."""
        repo = _make_repo(tmp_path, "stale")
        append_entries_to_archive(repo, _archive_entry("original"))
        entries1 = _parsed_archive_cached(repo)
        assert len(entries1) == 1

        # Simulate eviction of just this repo (fill cache with others).
        for i in range(_ARCHIVE_CACHE_MAX_ENTRIES):
            other = _make_repo(tmp_path, f"other{i}")
            append_entries_to_archive(other, _archive_entry(f"o{i}"))
            _parsed_archive_cached(other)

        path = insights_archive_path(repo)
        assert path not in _ARCHIVE_WRITE_VERSIONS  # evicted in lockstep

        # Re-access: version is now 0 (reset), content re-read fresh.
        entries2 = _parsed_archive_cached(repo)
        assert len(entries2) == 1
        # Version dict got re-populated only if a WRITE happens; a pure read
        # leaves version absent (.get(path, 0) == 0). No stale hit possible.
        assert _ARCHIVE_WRITE_VERSIONS.get(path, 0) == 0


# ── _ACTIVE_WRITE_VERSIONS: lockstep eviction ────────────────────────────────


class TestActiveVersionLockstep:
    """Mirror of the archive test for the active insights file content cache."""

    def test_active_version_dict_bounded_with_content_cache(self, tmp_path):
        """After visiting >cap repos, BOTH the active content cache and the
        active version dict stay ≤ cap."""
        cap = _ARCHIVE_CACHE_MAX_ENTRIES  # shared cap
        n = cap + 3

        for i in range(n):
            repo = _make_repo(tmp_path, f"a{i}")
            atomic_write_text(insights_path(repo), f"### [bug]\nactive {i}\n\n")
            _active_invalidate(repo)  # bumps version
            load_active_insights_cached(repo)  # populates content cache

        assert len(_ACTIVE_CONTENT_CACHE) <= cap, "active content cache unbounded"
        assert len(_ACTIVE_WRITE_VERSIONS) <= cap, (
            "_ACTIVE_WRITE_VERSIONS grew unbounded — lockstep eviction missing"
        )
