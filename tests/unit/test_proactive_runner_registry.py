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

import contextlib
from collections import OrderedDict

import external_llm.editor.agent.autonomous.proactive_runner as pr_mod
from external_llm.editor.agent.autonomous.proactive_runner import (
    _runners,
    _runners_lock,
    get_or_create_runner,
    make_stream_callback_interceptor,
    update_runner_features,
    update_runner_model_tier,
)
from external_llm.editor.agent.autonomous.trigger_policy import TriggerPolicy


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
            with contextlib.suppress(Exception):
                r.stop()
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
        get_or_create_runner("/repo-1")  # occupies slot 2; result not needed
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


# ── RED→GREEN: uncovered branches ────────────────────────────────────────────


def _cleanup_repo(repo_root: str) -> None:
    """Stop and remove a runner created by a test."""
    with _runners_lock:
        r = _runners.pop(repo_root, None)
    if r is not None:
        with contextlib.suppress(Exception):
            r.stop()


def test_get_or_create_updates_llm_invoke_fn():
    """Re-access with a new llm_invoke_fn updates the existing runner (L100)."""
    def fn1(**kw):
        return {"status": "first"}

    def fn2(**kw):
        return {"status": "second"}

    get_or_create_runner("/llm-repo", llm_invoke_fn=fn1)
    try:
        r = get_or_create_runner("/llm-repo", llm_invoke_fn=fn2)
        assert r.llm_invoke_fn is fn2
    finally:
        _cleanup_repo("/llm-repo")


def test_evicted_runner_stop_failure_logged(monkeypatch, caplog):
    """A stop() failure while evicting is logged, not fatal (L113-114)."""
    cap = 2
    monkeypatch.setattr(pr_mod, "_cfg", _StubCfg(cap))
    get_or_create_runner("/repo-0")
    get_or_create_runner("/repo-1")
    with _runners_lock:
        victim = _runners["/repo-0"]
    # Stop failure is injected on the victim itself, not by swapping its
    # engine: the drain daemon thread can touch _engine at any time, so a
    # half-broken engine object makes the failure a race (flake under load).
    # Patching stop() keeps the eviction path intact — evicted.stop() →
    # exception → logged, not fatal — with a deterministic failure.
    def _boom_stop():
        raise RuntimeError("engine stop boom")

    monkeypatch.setattr(victim, "stop", _boom_stop)
    try:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="external_llm.editor.agent.autonomous.proactive_runner"
        ):
            get_or_create_runner("/repo-2")  # evicts /repo-0 → stop() raises
        assert any("Failed to stop evicted" in r.message for r in caplog.records)
        assert "/repo-0" not in _runners
    finally:
        # The engine was never touched; stop() is still patched here, so the
        # failure is suppressed and the monkeypatch is undone by pytest.
        with contextlib.suppress(Exception):
            victim.stop()
        _cleanup_repo("/repo-1")
        _cleanup_repo("/repo-2")


def test_update_runner_model_tier_updates_existing_runner(monkeypatch):
    """Live model-tier update reaches the runner's policy (L120-124)."""
    get_or_create_runner("/tier-repo")
    try:
        update_runner_model_tier("/tier-repo", "strong")
        with _runners_lock:
            assert _runners["/tier-repo"].policy.model_tier == "strong"
        # Unknown repo → silent no-op.
        update_runner_model_tier("/missing-repo", "strong")
    finally:
        _cleanup_repo("/tier-repo")


def test_update_runner_features_all_and_csv():
    """Feature updates: 'all'/empty → every feature; CSV → filtered set;
    unknown repo → no-op (L135-145)."""
    get_or_create_runner("/feat-repo")
    try:
        update_runner_features("/feat-repo", "all")
        with _runners_lock:
            r = _runners["/feat-repo"]
        assert r.policy.enabled_features == set(TriggerPolicy._ALL_FEATURES)
        update_runner_features("/feat-repo", "file_review, bogus_feature")
        assert r.policy.enabled_features == {"file_review"}
        update_runner_features("/feat-repo", "")
        assert r.policy.enabled_features == set(TriggerPolicy._ALL_FEATURES)
        # Unknown repo → early return.
        update_runner_features("/missing-repo", "file_review")
    finally:
        _cleanup_repo("/feat-repo")


def test_stream_callback_interceptor_forwards_and_routes():
    """The interceptor forwards every event unchanged and routes
    fail_loop/complete/error to the engine; callback failures are swallowed
    (L168-187)."""
    forwarded: list = []
    notified: list = []

    class _FakeEngine:
        def notify_agent_event(self, event_name, data):
            notified.append((event_name, data))

    get_or_create_runner("/cb-repo")
    try:
        with _runners_lock:
            _runners["/cb-repo"]._engine = _FakeEngine()
        cb = make_stream_callback_interceptor(
            "/cb-repo", lambda ev, d: forwarded.append((ev, d))
        )
        cb("complete", {"x": 1})
        assert forwarded == [("complete", {"x": 1})]
        assert notified == [("complete", {"x": 1})]
        # original_cb failure is swallowed (L174-175).
        cb2 = make_stream_callback_interceptor(
            "/cb-repo", lambda ev, d: (_ for _ in ()).throw(RuntimeError("cb boom"))
        )
        cb2("complete", {"x": 2})  # must not raise
        # Unknown repo → forwarded only; no engine lookup result (L179-181).
        cb3 = make_stream_callback_interceptor("/nope", None)
        cb3("fail_loop_detected", {"x": 3})  # must not raise
        # Engine notify failure is swallowed (L184-185).
        class _BoomEngine:
            def notify_agent_event(self, *a):
                raise RuntimeError("engine boom")

        with _runners_lock:
            _runners["/cb-repo"]._engine = _BoomEngine()
        cb4 = make_stream_callback_interceptor("/cb-repo", None)
        cb4("complete", {"x": 4})  # must not raise
    finally:
        _cleanup_repo("/cb-repo")


def test_start_stop_idempotent_guards():
    """start() on a running runner and stop() on a stopped runner are no-ops
    (L238/L254)."""
    r = get_or_create_runner("/idem-repo")
    try:
        r.start()  # already running → early return
        r.stop()
        r.stop()  # already stopped → early return
    finally:
        _cleanup_repo("/idem-repo")


def test_notify_test_result_forwards_to_engine():
    """notify_test_result delegates to the engine (L267)."""
    seen: list = []

    class _FakeEngine:
        def notify_test_result(self, ok, details):
            seen.append((ok, details))

    r = get_or_create_runner("/ntr-repo")
    try:
        r._engine = _FakeEngine()
        r.notify_test_result(True, {"d": 1})
        assert seen == [(True, {"d": 1})]
    finally:
        _cleanup_repo("/ntr-repo")


def test_on_trigger_routes_ignore_escalate_and_enqueue():
    """_on_trigger maps policy decisions: IGNORE → nothing, ESCALATE →
    immediate broadcast, anything else → task queue (L273-292)."""
    from external_llm.editor.agent.autonomous.trigger_engine import (
        TriggerEvent,
        TriggerKind,
    )
    from external_llm.editor.agent.autonomous.trigger_policy import (
        ActionDecision,
        ActionKind,
    )

    r = get_or_create_runner("/trig-repo")
    r.stop()  # no drain thread → deterministic queue asserts
    try:
        event = TriggerEvent(
            kind=TriggerKind.AGENT_COMPLETED,
            repo_root="/trig-repo",
            source_file="a.py",
            metadata={"m": 1},
        )

        class _FakePush:
            def __init__(self):
                self.broadcasts: list = []

            def broadcast(self, *a):
                self.broadcasts.append(a)

        fake_push = _FakePush()
        r.push = fake_push

        class _FakePolicy:
            def __init__(self, decision):
                self._decision = decision
                self.enabled_features = set()

            model_tier = "small"

            def evaluate(self, ev):
                return self._decision

        # IGNORE → return immediately, nothing queued/broadcast.
        r.policy = _FakePolicy(ActionDecision(kind=ActionKind.IGNORE))
        r._on_trigger(event)
        assert fake_push.broadcasts == []
        assert r._queue.qsize() == 0
        # ESCALATE → immediate broadcast, not queued (L281-290).
        r.policy = _FakePolicy(
            ActionDecision(kind=ActionKind.ESCALATE, message="urgent", priority=1)
        )
        r._on_trigger(event)
        assert fake_push.broadcasts and fake_push.broadcasts[0][0] == "proactive_escalation"
        assert fake_push.broadcasts[0][1]["priority"] == 1
        assert r._queue.qsize() == 0
        # SUGGEST → enqueued (L292).
        r.policy = _FakePolicy(ActionDecision(kind=ActionKind.SUGGEST, prompt="analyze"))
        r._on_trigger(event)
        assert r._queue.qsize() == 1
    finally:
        _cleanup_repo("/trig-repo")


def test_execute_task_suggest_and_llm_paths():
    """SUGGEST/AUTO_FIX execution broadcasts start/done around the LLM call;
    the no-LLM stub, llm success and llm failure are all covered
    (L337-362, L377-393)."""
    from external_llm.editor.agent.autonomous.task_queue import AutonomousTask
    from external_llm.editor.agent.autonomous.trigger_engine import (
        TriggerEvent,
        TriggerKind,
    )
    from external_llm.editor.agent.autonomous.trigger_policy import (
        ActionDecision,
        ActionKind,
    )

    r = get_or_create_runner("/exec-repo")
    r.stop()  # deterministic: no drain thread consuming the queue
    try:

        class _FakePush:
            def __init__(self):
                self.broadcasts: list = []

            def broadcast(self, *a):
                self.broadcasts.append(a)

        fake_push = _FakePush()
        r.push = fake_push
        event = TriggerEvent(
            kind=TriggerKind.FILE_MODIFIED, repo_root="/exec-repo", source_file="f.py"
        )
        task = AutonomousTask(
            priority=2,
            created_at=0.0,
            event=event,
            action=ActionDecision(kind=ActionKind.SUGGEST, message="m", prompt="p"),
            task_id="t1",
        )
        # No llm_invoke_fn → stub result (L377-383).
        r._execute_task(task)
        assert fake_push.broadcasts[0][0] == "proactive_fix_started"
        assert fake_push.broadcasts[1][0] == "proactive_fix_done"
        assert fake_push.broadcasts[1][1]["result"]["status"] == "no_model"
        # llm_invoke_fn success (L385-390).
        r.llm_invoke_fn = lambda **kw: {"status": "ok", "text": kw["request_text"]}
        task2 = AutonomousTask(
            priority=2, created_at=0.0, event=event,
            action=ActionDecision(kind=ActionKind.AUTO_FIX, message="m", prompt="p2"),
            task_id="t2",
        )
        r._execute_task(task2)
        assert fake_push.broadcasts[-1][1]["result"]["status"] == "ok"
        # llm_invoke_fn failure → error result (L391-393).
        def _boom_llm(**kw):
            raise RuntimeError("llm down")

        r.llm_invoke_fn = _boom_llm
        task3 = AutonomousTask(
            priority=2, created_at=0.0, event=event,
            action=ActionDecision(kind=ActionKind.SUGGEST, message="m", prompt="p3"),
            task_id="t3",
        )
        r._execute_task(task3)
        assert fake_push.broadcasts[-1][1]["result"]["status"] == "error"
        # Broadcast failure inside execution → warning + proactive_error
        # (L360-362).
        class _FlakyPush:
            def __init__(self):
                self.calls: list = []

            def broadcast(self, *a):
                self.calls.append(a)
                if len(self.calls) == 1:
                    raise RuntimeError("push boom")

        r.push = _FlakyPush()
        task4 = AutonomousTask(
            priority=2, created_at=0.0, event=event,
            action=ActionDecision(kind=ActionKind.NOTIFY, message="n"),
            task_id="t4",
        )
        r._execute_task(task4)
        assert r.push.calls[1][0] == "proactive_error"
    finally:
        _cleanup_repo("/exec-repo")
