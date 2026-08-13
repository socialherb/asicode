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
    _ARCHIVE_ANALYZED_CACHE,
    _ARCHIVE_CACHE_MAX_ENTRIES,
    _ARCHIVE_PARSED_CACHE,
    _ARCHIVE_WRITE_VERSIONS,
    _INSIGHTS_THREAD_LOCKS,
    _active_invalidate,
    _archive_analyzed_cached,
    _archive_capped_put,
    _archive_invalidate,
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
        _ARCHIVE_ANALYZED_CACHE,
        _ARCHIVE_WRITE_VERSIONS,
        _ACTIVE_CONTENT_CACHE,
        _ACTIVE_WRITE_VERSIONS,
    ):
        d.clear()
    yield
    for d in (
        _ARCHIVE_PARSED_CACHE,
        _ARCHIVE_ANALYZED_CACHE,
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
    _preamble, entries = parse_insights(f"### [bug] 2025-01-15 10:00\n{text}\n\n")
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

    def test_sibling_cache_keeps_version_after_eviction(self, tmp_path):
        """P1: a path evicted from ONE cache keeps its write version while it
        still lives in the sibling cache, so the sibling does not false-miss
        (full re-parse + re-tokenize) on its next request."""
        class _Tok:
            def tokenize(self, text):
                return text.split()

        tok = _Tok()
        cap = _ARCHIVE_CACHE_MAX_ENTRIES
        keep = _make_repo(tmp_path, "keep")
        append_entries_to_archive(keep, _archive_entry("keep"))
        _parsed_archive_cached(keep)          # PARSED: [keep]
        _archive_analyzed_cached(keep, tok)   # ANALYZED: [keep]
        path = insights_archive_path(keep)
        assert path in _ARCHIVE_WRITE_VERSIONS

        # Fill PARSED past cap: keep is the oldest entry → FIFO-evicted from
        # PARSED only. ANALYZED still holds it, so the version must survive.
        for i in range(cap):
            other = _make_repo(tmp_path, f"s{i}")
            append_entries_to_archive(other, _archive_entry(f"s{i}"))
            _parsed_archive_cached(other)

        assert path not in _ARCHIVE_PARSED_CACHE
        assert path in _ARCHIVE_ANALYZED_CACHE
        assert path in _ARCHIVE_WRITE_VERSIONS  # kept for the sibling

        # Re-access via the analyzed path hits — no re-parse + re-tokenize.
        entries, toksets, _df, _avgdl = _archive_analyzed_cached(keep, tok)
        assert len(entries) == 1
        assert len(toksets) == 1


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


# ── C1: _archive_capped_put must be a true LRU (pop-before-reinsert) ──────────


class TestArchiveCappedPutLRU:
    """C1 regression: re-putting an existing key must refresh its dict position.

    Without the ``cache.pop(key, None)`` before the re-insert, an ACTIVE repo
    (whose archive changes every turn -> re-put every turn) keeps its ORIGINAL
    position at the front of the dict and reads as the oldest entry — so it is
    the first eviction candidate once cap+1 repos are visited, forcing a full
    archive re-read + re-parse every turn in multi-repo sessions (the demo:
    interleaved re-puts of A yield ``BCDEFGHI`` instead of ``CDEFGHAI``)."""

    def test_reput_refreshes_position(self):
        """Unit-level demo (C1): a re-put key must land at the BACK of the
        dict. Without pop-before-reinsert, re-putting ``A`` leaves it at the
        FRONT, so the cap+1-th insert evicts ``A`` (``BCDEFGHI``) instead of
        the true LRU ``K0`` (``CDEFGHAI``)."""
        cap = _ARCHIVE_CACHE_MAX_ENTRIES
        cache: dict = {}
        _archive_capped_put(cache, "A", "A")
        for i in range(cap - 1):
            _archive_capped_put(cache, f"K{i}", f"K{i}")
        _archive_capped_put(cache, "A", "A-refreshed")  # re-put → must move to back
        _archive_capped_put(cache, "K7", "K7")  # over cap → evict exactly one

        assert cache.get("A") == "A-refreshed", (
            "re-put key was evicted — insertion order not refreshed (C1)"
        )
        assert "K0" not in cache, "true LRU survived — eviction is FIFO (C1)"
        assert len(cache) == cap

    def test_external_edit_reinsert_keeps_active_repo(self, tmp_path):
        """Integration path where the C1 bug actually bites: the active repo's
        archive is edited OUTSIDE the write funnel (no ``_archive_invalidate``
        — a hand edit), so the signature (mtime/size) miss re-reads and
        re-PUTS over the still-present key. With the bug, that re-put keeps A
        at the FRONT, so the 9th repo's insert evicts the active repo itself."""
        repo_a = _make_repo(tmp_path, "A")
        append_entries_to_archive(repo_a, _archive_entry("a-v1"))
        _parsed_archive_cached(repo_a)  # A joins first → oldest position
        others = []
        for name in "BCDEFGH":
            repo = _make_repo(tmp_path, name)
            others.append(repo)
            append_entries_to_archive(repo, _archive_entry(f"{name}-v1"))
            _parsed_archive_cached(repo)
        # 8 entries: A,B,C,D,E,F,G,H — cache is full.

        # A's archive changes outside the funnel → signature miss → re-put
        # over the still-present key (no pop beforehand).
        atomic_write_text(
            insights_archive_path(repo_a),
            "### [bug] 2025-01-15 10:00\nexternally changed\n\n",
        )
        assert _parsed_archive_cached(repo_a)[0].body.strip() == "externally changed"

        # 9th repo arrives → exactly one eviction: the true LRU (B), not A.
        repo_i = _make_repo(tmp_path, "I")
        append_entries_to_archive(repo_i, _archive_entry("i-v1"))
        _parsed_archive_cached(repo_i)

        assert insights_archive_path(repo_a) in _ARCHIVE_PARSED_CACHE, (
            "active repo evicted despite fresh re-read — re-put must refresh "
            "insertion order (C1)"
        )
        assert insights_archive_path(others[0]) not in _ARCHIVE_PARSED_CACHE, (
            "true LRU survived — eviction uses stale dict positions (C1)"
        )
        assert len(_ARCHIVE_PARSED_CACHE) == _ARCHIVE_CACHE_MAX_ENTRIES


# ── C2: cache keys canonicalized via canonical_repo_key ───────────────────────


class TestCanonicalArchiveCacheKeys:
    """C2 regression: spelling variants of one repo (macOS ``/var`` vs
    ``/private/var``, symlinked aliases) must share ONE cache slot and ONE
    write-version counter — otherwise the same repo occupies two of the 8
    slots and an invalidator bumps a version the reader never checks."""

    def test_symlink_alias_share_slot_and_version(self, tmp_path):
        repo = _make_repo(tmp_path, "canon")
        alias = repo + "_alias"
        try:
            os.symlink(repo, alias)
        except OSError:
            pytest.skip("symlink not available on this platform")

        try:
            append_entries_to_archive(repo, _archive_entry("v1"))
            assert len(_parsed_archive_cached(repo)) == 1
            assert len(_ARCHIVE_PARSED_CACHE) == 1

            # Reading via the alias must hit the SAME slot…
            assert len(_parsed_archive_cached(alias)) == 1
            assert len(_ARCHIVE_PARSED_CACHE) == 1, (
                "alias spelling occupied a second cache slot — "
                "canonical_repo_key missing in insights_archive_path (C2)"
            )
            assert insights_archive_path(repo) == insights_archive_path(alias)
            # One shared counter: the alias read sees version 1 (bumped by the
            # single write), not 0 (a split counter would read 0).
            assert _ARCHIVE_WRITE_VERSIONS[insights_archive_path(repo)] == 1
        finally:
            os.unlink(alias)

    def test_invalidate_via_alias_hits_same_counter(self, tmp_path):
        repo = _make_repo(tmp_path, "inv")
        alias = repo + "_alias"
        try:
            os.symlink(repo, alias)
        except OSError:
            pytest.skip("symlink not available on this platform")

        try:
            append_entries_to_archive(repo, _archive_entry("v1"))
            _parsed_archive_cached(repo)
            key = insights_archive_path(repo)
            before = _ARCHIVE_WRITE_VERSIONS[key]

            _archive_invalidate(alias)  # invalidate via the OTHER spelling

            assert _ARCHIVE_WRITE_VERSIONS[key] == before + 1, (
                "invalidator bumped a version key the reader never checks — "
                "spelling split (C2)"
            )
            assert key not in _ARCHIVE_PARSED_CACHE, "cache not dropped on invalidate"
        finally:
            os.unlink(alias)


class TestCanonicalActiveCacheKeys:
    """C2 mirror for the ACTIVE insights file cache family."""

    def test_active_symlink_alias_share_slot(self, tmp_path):
        repo = _make_repo(tmp_path, "act_canon")
        alias = repo + "_alias"
        try:
            os.symlink(repo, alias)
        except OSError:
            pytest.skip("symlink not available on this platform")

        try:
            content = "### [bug]\nactive\n\n"
            atomic_write_text(insights_path(repo), content)
            _active_invalidate(repo)
            assert load_active_insights_cached(repo) == content
            assert load_active_insights_cached(alias) == content
            assert len(_ACTIVE_CONTENT_CACHE) == 1, (
                "active cache split across spellings — canonical_repo_key "
                "missing in insights_path (C2)"
            )
            assert insights_path(repo) == insights_path(alias)
        finally:
            os.unlink(alias)
