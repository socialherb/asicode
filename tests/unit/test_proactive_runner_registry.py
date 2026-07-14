"""Regression tests for the bounded LRU ProactiveRunner registry.

Guards against the thread+memory leak fixed in this commit: ``_runners`` was an
unbounded plain dict — each distinct repo_root touching the webapp got a
ProactiveRunner with a drain daemon thread (``while self._running``) plus
TriggerEngine schedule timers, none of which were ever reclaimed (``stop()`` was
defined but never called; no pop/clear/eviction existed).

This is the same leak class as _GRAPH_FACADE_CACHE / _ACTIVE_ENGINES
(commit ffdb7c7f) except it ALSO leaks a thread per entry.

Mutation guard: if _runners is reverted to a plain dict, the OrderedDict
assertion fails (plain dict has no move_to_end → LRU silently disabled).
"""
from __future__ import annotations

from collections import OrderedDict

import external_llm.editor.agent.autonomous.proactive_runner as pr_mod
from external_llm.editor.agent.autonomous.proactive_runner import (
    _runners,
    _runners_lock,
    get_or_create_runner,
)


# ── Test config stub ─────────────────────────────────────────────────────────
# The real config is a frozen dataclass singleton; we swap the module-level
# ``_cfg`` reference (read at call time inside get_or_create_runner) for a stub
# with a small cap so the test exercises eviction without spawning 9 threads.
class _StubCounts:
    def __init__(self, cap: int) -> None:
        self.AUTONOMOUS_RUNNER_MAX = cap


class _StubCfg:
    def __init__(self, cap: int) -> None:
        self.counts = _StubCounts(cap)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestProactiveRunnerRegistryLRU:
    """get_or_create_runner must bound _runners via LRU and stop() evicted runners."""

    def setup_method(self):
        # Snapshot for restoration; clear to start from a known state.
        with _runners_lock:
            self._saved = dict(_runners)
            _runners.clear()

    def teardown_method(self):
        # Stop ALL live runners so no drain daemon thread leaks across tests.
        with _runners_lock:
            runners = list(_runners.values())
            _runners.clear()
        for r in runners:
            try:
                r.stop()
            except Exception:
                pass
        # Restore the pre-test registry (without re-starting saved runners —
        # they are external test-env state we must not mutate further).
        with _runners_lock:
            _runners.clear()
            _runners.update(self._saved)

    def test_overflow_evicts_oldest_and_bounds_registry(self, monkeypatch):
        """Creating cap+1 distinct repos keeps size at cap; oldest evicted."""
        cap = 3
        monkeypatch.setattr(pr_mod, "_cfg", _StubCfg(cap))
        for i in range(cap + 1):
            get_or_create_runner(f"/repo-{i}")
        assert len(_runners) == cap, (
            f"_runners must be bounded at {cap}, got {len(_runners)} "
            "(unbounded leak regression)"
        )
        assert "/repo-0" not in _runners, "oldest entry must be evicted"
        assert f"/repo-{cap}" in _runners, "newest entry must be retained"

    def test_get_promotes_to_most_recently_used(self, monkeypatch):
        """Touching the oldest entry protects it from the next eviction."""
        cap = 3
        monkeypatch.setattr(pr_mod, "_cfg", _StubCfg(cap))
        for i in range(cap):
            get_or_create_runner(f"/repo-{i}")
        # Touch the oldest → it becomes most-recently-used.
        get_or_create_runner("/repo-0")
        # Add one more → should evict what is now the oldest (/repo-1).
        get_or_create_runner("/repo-extra")
        assert "/repo-0" in _runners, "touched entry must survive eviction"
        assert "/repo-1" not in _runners, "untouched oldest must be evicted"
        assert len(_runners) == cap

    def test_evicted_runner_is_stopped_and_drain_thread_terminates(self, monkeypatch):
        """CRITICAL: the evicted runner must be stop()'d — _running cleared and
        its drain daemon thread must terminate. Without the stop() call, the
        thread leaks for the process lifetime (the bug being fixed here)."""
        cap = 2
        monkeypatch.setattr(pr_mod, "_cfg", _StubCfg(cap))
        get_or_create_runner("/repo-0")
        get_or_create_runner("/repo-1")
        assert len(_runners) == cap
        # Capture the LRU victim before overflow.
        with _runners_lock:
            victim = _runners["/repo-0"]
        drain_thread = victim._drain_thread
        assert victim._running is True, "precondition: victim is running"
        assert drain_thread is not None and drain_thread.is_alive(), (
            "precondition: victim has a live drain thread"
        )
        # Overflow → evicts /repo-0 (LRU) and stop()'s it (outside the lock).
        get_or_create_runner("/repo-2")
        assert "/repo-0" not in _runners, "victim must be evicted from registry"
        assert victim._running is False, (
            "evicted runner must be stop()'d — _running flag must be cleared "
            "(drain thread would otherwise loop forever)"
        )
        # The drain thread must actually terminate (within DRAIN_INTERVAL sleep).
        if drain_thread is not None:
            drain_thread.join(timeout=5.0)
            assert not drain_thread.is_alive(), (
                "evicted runner's drain daemon thread must terminate after stop(); "
                "a live thread here is a thread leak"
            )
        assert len(_runners) == cap

    def test_re_access_via_get_promotes_existing_entry(self, monkeypatch):
        """Re-calling get_or_create_runner on an existing repo must move it to
        the MRU end (move_to_end) without creating a new runner."""
        cap = 2
        monkeypatch.setattr(pr_mod, "_cfg", _StubCfg(cap))
        r0 = get_or_create_runner("/repo-0")
        r1 = get_or_create_runner("/repo-1")
        # Re-access /repo-0 → it should become most-recently-used.
        r0_again = get_or_create_runner("/repo-0")
        assert r0_again is r0, "existing runner must be returned (no new instance)"
        # Now overflow with /repo-2 → /repo-1 (now oldest) should be evicted.
        get_or_create_runner("/repo-2")
        assert "/repo-0" in _runners, "re-accessed entry must survive (was promoted)"
        assert "/repo-1" not in _runners, "stale oldest must be evicted"

    def test_registry_is_ordered_dict(self):
        """Mutation guard: _runners must remain an OrderedDict (plain dict has
        no move_to_end and silently disables LRU promotion/eviction)."""
        assert isinstance(_runners, OrderedDict), (
            "_runners must be OrderedDict for LRU semantics; a plain dict has no "
            "move_to_end and silently disables LRU promotion/eviction"
        )
