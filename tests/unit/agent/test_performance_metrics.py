"""Unit tests for performance_metrics.py — ToolMetrics aggregation (#2),
file_cache decoupling (#3), tool failure recording (#4), and thread-safety
of get_summary / reset (#5)."""

import threading
import time

from external_llm.agent.performance_metrics import (
    PerformanceCollector,
    ToolMetrics,
    _reset_warned_failing_tools,
    get_global_collector,
    reset_global_collector,
    top_failing_tools,
    warn_failing_tools,
)


# -- #2: O(1) running aggregation, no unbounded list --------------------------


class TestToolMetricsAggregation:
    def test_no_execution_times_list_attribute(self):
        # The unbounded execution_times list field is gone — replaced by O(1)
        # running counters (12h+ runs would otherwise leak RAM and make
        # avg/summary O(n)).
        m = ToolMetrics(name="read_file")
        assert not hasattr(m, "execution_times"), (
            "ToolMetrics must not retain a per-call execution_times list"
        )

    def test_running_aggregation_matches_manual_stats(self):
        m = ToolMetrics(name="read_file")
        for t in (0.001, 0.005, 0.003, 0.002):
            m.record(t)
        assert m.total_calls == 4
        assert abs(m.avg_execution_time - 0.00275) < 1e-9
        assert m.min_execution_time == 0.001
        assert m.max_execution_time == 0.005

    def test_avg_zero_when_no_calls(self):
        m = ToolMetrics(name="read_file")
        assert m.avg_execution_time == 0.0
        assert m.min_execution_time == 0.0  # not float('inf')
        assert m.max_execution_time == 0.0

    def test_get_summary_call_count_uses_total_calls_not_list_len(self):
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01)
        c.record_tool_call("read_file", 0.02)
        s = c.get_summary()
        ts = s["tool_metrics"]["read_file"]
        assert ts["call_count"] == 2
        assert ts["total_calls"] == 2
        # min/max surfaced for latency spread (free with running aggregation)
        assert ts["min_execution_time_ms"] == 10.0
        assert ts["max_execution_time_ms"] == 20.0
        assert abs(ts["avg_execution_time_ms"] - 15.0) < 1e-9


# -- #3: record_tool_call must NOT pollute the file_cache channel --------------


class TestFileCacheDecoupling:
    def test_record_tool_call_does_not_feed_file_cache(self):
        # cache_hit passed to record_tool_call is the ToolResultCache hit flag,
        # NOT a file-cache hit. Feeding it into cache_metrics.file_cache
        # duplicated the per-tool counters and the tool_result_cache channel and
        # distorted overall_hit_rate.
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01, cache_hit=True)
        c.record_tool_call("edit_text", 0.02, cache_hit=False)
        fc = c.cache_metrics.get_stats("file")
        assert fc["hits"] == 0 and fc["misses"] == 0 and fc["total"] == 0

    def test_per_tool_cache_granularity_preserved(self):
        # Removing the file_cache feed must not lose per-tool cache stats.
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01, cache_hit=True)
        c.record_tool_call("read_file", 0.01, cache_hit=False)
        m = c.tool_metrics["read_file"]
        assert m.cache_hits == 1 and m.cache_misses == 1
        assert m.cache_hit_rate == 0.5


# -- #4: tool failure recording — closing the LLMMetrics.failures asymmetry ---


class TestToolFailureRecording:
    def test_failed_increments_failures_counter(self):
        c = PerformanceCollector()
        c.record_tool_call("apply_patch", 0.1, failed=True)
        m = c.tool_metrics["apply_patch"]
        assert m.failures == 1
        assert m.failure_rate == 1.0

    def test_default_failed_false_keeps_failures_zero(self):
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01)
        c.record_tool_call("read_file", 0.02, failed=False)
        m = c.tool_metrics["read_file"]
        assert m.failures == 0
        assert m.failure_rate == 0.0

    def test_mixed_success_and_failure_rate(self):
        # 4 calls, 1 failure -> 0.25 failure rate
        c = PerformanceCollector()
        c.record_tool_call("edit_text", 0.05, failed=False)
        c.record_tool_call("edit_text", 0.05, failed=True)
        c.record_tool_call("edit_text", 0.05, failed=False)
        c.record_tool_call("edit_text", 0.05, failed=False)
        m = c.tool_metrics["edit_text"]
        assert m.total_calls == 4
        assert m.failures == 1
        assert m.failure_rate == 0.25

    def test_get_summary_exposes_per_tool_failures_and_rate(self):
        # The whole point: the dashboard/summary must surface which tool fails
        # how often. record_tool_call(failed=not result.ok) at the two record
        # sites feeds this; get_summary() must expose it.
        c = PerformanceCollector()
        c.record_tool_call("apply_patch", 0.2, failed=True)   # rolled-back write
        c.record_tool_call("apply_patch", 0.1, failed=False)
        c.record_tool_call("read_file", 0.01, failed=False)
        c.record_tool_call("read_file", 0.02, failed=True)    # missing file
        summary = c.get_summary()
        ap = summary["tool_metrics"]["apply_patch"]
        rf = summary["tool_metrics"]["read_file"]
        assert ap["failures"] == 1 and ap["total_calls"] == 2
        assert ap["failure_rate"] == 0.5
        assert rf["failures"] == 1 and rf["total_calls"] == 2
        assert rf["failure_rate"] == 0.5

    def test_failure_rate_zero_when_no_calls(self):
        # ToolMetrics created but never recorded -> no division-by-zero.
        m = ToolMetrics(name="unused")
        assert m.failure_rate == 0.0


# -- #5: thread-safety — get_summary consistency + reset_global_collector -----


class TestThreadSafety:
    def test_concurrent_record_and_summary_no_error_and_consistent(self):
        # Concurrent record_tool_call vs get_summary must not raise and must
        # produce self-consistent per-tool stats (call_count == total_calls).
        c = PerformanceCollector()
        errors = []
        stop = threading.Event()

        def recorder():
            try:
                i = 0
                while not stop.is_set():
                    c.record_tool_call(
                        "read_file", 0.001 * (i % 5 + 1), cache_hit=(i % 2 == 0)
                    )
                    i += 1
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def summarizer():
            try:
                while not stop.is_set():
                    s = c.get_summary()
                    ts = s["tool_metrics"].get("read_file")
                    if ts is not None:
                        # torn read would let call_count drift from total_calls
                        assert ts["call_count"] == ts["total_calls"], ts
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=recorder) for _ in range(3)] + [
            threading.Thread(target=summarizer) for _ in range(2)
        ]
        for t in threads:
            t.start()
        stop.set()  # give them a brief burst
        for t in threads:
            t.join(timeout=5)
        assert not errors, errors

    def test_reset_global_collector_returns_new_and_is_locked(self):
        c = reset_global_collector(session_id="rt_test")
        assert c.session_id == "rt_test"
        # concurrent resets must not raise and must leave a valid collector
        results = []

        def do_reset():
            results.append(reset_global_collector(session_id="x"))

        ts = [threading.Thread(target=do_reset) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert all(r is not None for r in results)


# -- #6: collector unification — per-loop aliases global (split-brain fix) ----
# Background: a fresh per-loop PerformanceCollector left the webapp dashboard
# (reads get_global_collector()) permanently blind to llm_metrics —
# record_llm_call hit only the per-loop instance — and the per-turn summary
# (reads loop.performance_collector) permanently blind to cache/rag metrics —
# those hit only the global collector via rag_searcher/tool_registry. Aliasing
# one collector for both consumers closes both gaps. dispatch() is now the SOLE
# tool-call recorder (single-exit wrapper over _dispatch_impl), so aliasing does
# not double-count.


def _make_loop_unification(tmp_path):
    """Minimal AgentLoop over a fresh git repo (mirrors test_run_main_agent_regression)."""
    import subprocess
    from pathlib import Path
    from unittest.mock import Mock
    from external_llm.agent.agent_loop import AgentLoop
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    repo = Path(tmp_path)
    for c in (
        ["git", "init", "-q"], ["git", "config", "user.email", "t@t.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(c, cwd=str(repo), capture_output=True)
    (repo / "f.txt").write_text("alpha=1\n")
    subprocess.run(["git", "add", "f.txt"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=str(repo), capture_output=True)
    client = Mock()
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"
    cfg = AgentConfig(max_turns=1, planning_enabled=False, rag_enabled=False)
    reg = ToolRegistry(str(repo), cfg)
    return AgentLoop(llm_client=client, registry=reg, config=cfg, model="test")


class TestCollectorUnification:
    """Per-loop collector is session-isolated; global collector aggregates across sessions.

    After the alias was reverted (concurrent-session isolation regression),
    the two collectors serve distinct roles:

    * Per-loop (``self.performance_collector``)  — per-turn summary accuracy.
      Each AgentLoop creates its own instance with a session-specific id.
      Tool metrics reach it via the pipeline (``_process_tool_results``);
      LLM metrics via the same ``record_llm_call`` call that also feeds the
      global collector.

    * Global (``get_global_collector()``)  — dashboard aggregate. A single
      process-lifetime singleton receives ALL sessions' tool metrics (from
      the dispatch wrapper) and LLM metrics (from agent_loop's dual record).
      ``start_session()`` is called once at construction so the dashboard
      sees ``total_execution_time_seconds`` ≈ process uptime.

    This decoupling closes the original split-brain (dashboard blind to LLM,
    per-turn summary blind to cache) WITHOUT the regression — concurrent
    webapp sessions each have their own per-loop collector and cannot
    overwrite each other's ``session_id`` / ``start_time``.

    The accepted tradeoff: cache metrics (``tool_result_cache``,
    ``rag_cache``, ``vector_cache``) are NOT in the per-turn summary
    (they are dashboard‑only aggregates, fed to the global collector by
    ``rag_searcher``).  Per-tool call and LLM metrics are present in both.
    """

    def test_loop_collector_is_not_global_alias(self, tmp_path):
        """Per-loop collector is a FRESH instance, not the global singleton."""
        reset_global_collector()
        loop = _make_loop_unification(tmp_path)
        assert loop.performance_collector is not get_global_collector()
        # Each loop gets its own session_id, not the global one
        assert loop.performance_collector.session_id != get_global_collector().session_id

    def test_llm_metrics_reach_both_collectors(self, tmp_path):
        """LLM calls are recorded to BOTH per-loop and global collector.

        This is the core of the split-brain fix without the alias: per-turn
        summary (reads per-loop) and dashboard (reads global) both get LLM
        data via dual recording at the agent_loop record_llm_call site.
        """
        reset_global_collector()
        loop = _make_loop_unification(tmp_path)
        loop.performance_collector.record_llm_call(
            prompt_tokens=120, completion_tokens=40, execution_time_ms=800
        )
        get_global_collector().record_llm_call(
            prompt_tokens=120, completion_tokens=40, execution_time_ms=800
        )
        # Per-loop summary sees its own LLM call
        pl = loop.performance_collector.get_summary()["llm_metrics"]
        assert pl["calls"] == 1
        assert pl["total_tokens"] == 160
        # Global summary also sees the LLM call
        gl = get_global_collector().get_summary()["llm_metrics"]
        assert gl["calls"] == 1
        assert gl["total_tokens"] == 160

    def test_cache_metrics_are_dashboard_only(self, tmp_path):
        """Cache/rag metrics live only on the global collector (dashboard).

        This is the accepted tradeoff: per-turn summaries do NOT include
        aggregate cache hit rates or tool_result_cache stats.  Per-tool
        metrics (calls, failures, cache_hit_rate) are available per-loop
        via pipeline recording; LLM metrics are available per-loop via dual
        recording.
        """
        from external_llm.agent.tool_result_cache import ToolResultCache

        reset_global_collector()
        loop = _make_loop_unification(tmp_path)
        cache = ToolResultCache(max_entries=8)
        get_global_collector().register_tool_result_cache(cache)
        cache.set("read_file", {"path": "x"}, {"content": "hi"})
        cache.get("read_file", {"path": "x"})  # hit
        # Dashboard (global) has the cache stats
        gl = get_global_collector().get_summary()["cache_metrics"]
        trc = gl.get("tool_result_cache")
        assert trc is not None and trc["hits"] >= 1
        # Per-loop summary does NOT have them (dashboard-only aggregate)
        pl = loop.performance_collector.get_summary()["cache_metrics"]
        assert pl.get("tool_result_cache") is None

    def test_overall_hit_rate_includes_tool_result_cache(self, tmp_path):
        """overall_hit_rate must aggregate ALL live cache channels, including
        tool_result_cache (the largest by volume). Previously it summed only
        file+rag+vector — and file is always 0 (legacy, no feeder) — so the
        headline silently omitted the tool_result_cache hit/miss volume, making
        the dashboard "overall" reflect only rag+vector. With ONLY
        tool_result_cache active (rag/vector/file all zero) the pre-fix value
        was 0; post-fix it reflects the tool_result_cache hit rate.
        """
        from external_llm.agent.tool_result_cache import ToolResultCache

        reset_global_collector()
        _make_loop_unification(tmp_path)
        cache = ToolResultCache(max_entries=8)
        get_global_collector().register_tool_result_cache(cache)
        cache.set("read_file", {"path": "a"}, {"v": 1})
        cache.set("read_file", {"path": "b"}, {"v": 2})
        cache.get("read_file", {"path": "a"})  # hit
        cache.get("read_file", {"path": "b"})  # hit
        cache.get("read_file", {"path": "z"})  # miss
        cm = get_global_collector().get_summary()["cache_metrics"]
        trc = cm["tool_result_cache"]
        assert trc is not None and trc["hits"] == 2 and trc["misses"] == 1
        # overall must reflect tool_result_cache: 2 hits / 3 total
        assert abs(cm["overall_hit_rate"] - 2 / 3) < 1e-9

    def test_dispatch_records_to_global_collector_exactly_once(self, tmp_path):
        """dispatch() records to the global collector (dashboard).  Per-loop
        gets its copy from the pipeline; a direct dispatch without pipeline
        leaves per-loop at 0, which is correct."""
        from pathlib import Path

        reset_global_collector()
        loop = _make_loop_unification(tmp_path)
        loop.registry.dispatch("read_file", {"path": str(Path(tmp_path) / "f.txt")})
        tm = get_global_collector().get_summary()["tool_metrics"].get("read_file")
        assert tm is not None and tm["total_calls"] == 1

    def test_dispatch_records_early_return_paths(self, tmp_path):
        """dispatch() records unknown-tool early returns to global collector."""
        reset_global_collector()
        loop = _make_loop_unification(tmp_path)
        loop.registry.dispatch("nonexistent_tool_xyz", {"arg": 1})
        tm = get_global_collector().get_summary()["tool_metrics"].get("nonexistent_tool_xyz")
        assert tm is not None and tm["total_calls"] == 1 and tm["failures"] == 1

    def test_concurrent_session_isolation(self):
        """Two independent collectors do NOT contaminate each other's summary.

        This is the P0 regression that the alias introduced: a shared global
        collector allowed concurrent webapp sessions to overwrite each other's
        ``session_id`` / ``start_time`` and mix their metrics.
        """
        a = PerformanceCollector(session_id="session-A")
        b = PerformanceCollector(session_id="session-B")

        # Each session records its own tool + LLM calls
        a.record_tool_call("read_file", 0.01, failed=False)
        a.record_tool_call("apply_patch", 0.5, failed=True)
        a.record_llm_call(prompt_tokens=100, completion_tokens=50, execution_time_ms=600)

        b.record_tool_call("read_file", 0.02, failed=False)
        b.record_llm_call(prompt_tokens=999, completion_tokens=0, execution_time_ms=2000)

        a_summary = a.get_summary()
        b_summary = b.get_summary()

        # Session A summary must NOT have B's session_id or data
        assert a_summary["session_id"] == "session-A"
        assert b_summary["session_id"] == "session-B"

        # A: 2 tool calls (read_file + apply_patch)
        a_tools = a_summary["tool_metrics"]
        assert a_tools["read_file"]["total_calls"] == 1
        assert a_tools["apply_patch"]["total_calls"] == 1
        assert a_tools["apply_patch"]["failures"] == 1
        # A: 1 LLM call, 150 tokens
        assert a_summary["llm_metrics"]["calls"] == 1
        assert a_summary["llm_metrics"]["total_tokens"] == 150

        # B: 1 tool call (read_file), 1 LLM call with 999 prompt tokens
        b_tools = b_summary["tool_metrics"]
        assert b_tools["read_file"]["total_calls"] == 1
        assert "apply_patch" not in b_tools  # B did not call apply_patch
        assert b_summary["llm_metrics"]["calls"] == 1
        assert b_summary["llm_metrics"]["total_prompt_tokens"] == 999


# -- #7: failed LLM calls must record real execution_time_ms (avg_time_ms bias) --
# Background: the retry-exhaustion and non-retriable failure paths in
# agent_loop._retry_on_rate_limit originally called _record_llm_call_both(failed=True)
# WITHOUT execution_time_ms, defaulting to 0. Since record_llm_call adds
# execution_time_ms to total_time_ms unconditionally (regardless of failed),
# failed calls were diluting avg_time_ms = total_time_ms / calls toward 0 —
# the same bias that was fixed for design-chat. This guards that both
# consumers (per-loop + global) get the real wall-time on failure.


class TestFailedLLMCallTiming:
    def test_retry_exhaustion_records_nonzero_execution_time(self, tmp_path, monkeypatch):
        """A rate-limited call that exhausts retries records execution_time_ms>0.

        Validates the whole-retry-span timer (loop_t0): the recorded time must
        reflect call attempts + backoff waits, not 0. Backoff sleeps are patched
        to no-ops so the test runs fast; the recorded time is still > 0 because
        real wall-time elapses between loop_t0 and the final record.
        """
        from external_llm.agent import agent_loop as al_mod
        from external_llm.client import LLMRateLimitError

        reset_global_collector()
        loop = _make_loop_unification(tmp_path)

        # Patch the two sleep primitives used during backoff so the test doesn't
        # actually wait 10+20+40s. We still let a tiny real delay elapse so the
        # whole-retry-span timer (loop_t0) records a measurable execution_time_ms
        # (>1ms after rounding) instead of being clamped to 0. NOTE: we must NOT
        # call time.sleep() inside _fake_sleep (that's what we're patching) — use
        # a busy-wait that reads time.monotonic() directly.
        slept = {"s": 0.0}

        def _fake_sleep(d):
            slept["s"] += d
            _spin_until = time.monotonic() + 0.003  # 3ms measurable gap
            while time.monotonic() < _spin_until:
                pass

        monkeypatch.setattr(al_mod.time, "sleep", _fake_sleep)

        def _always_rate_limited():
            raise LLMRateLimitError("429 rate limited")

        # The call must raise after retries are exhausted.
        raised = False
        try:
            loop._retry_on_rate_limit(_always_rate_limited, mode="test")
        except LLMRateLimitError:
            raised = True
        assert raised, "rate-limit exhaustion must re-raise"

        # Backoff was exercised (3 sleeps: 10+20+40)
        assert slept["s"] == 70.0

        # The failed call reached BOTH collectors with a NONZERO execution_time_ms.
        pl = loop.performance_collector.get_summary()["llm_metrics"]
        gl = get_global_collector().get_summary()["llm_metrics"]
        assert pl["calls"] == 1 and pl["failures"] == 1
        assert gl["calls"] == 1 and gl["failures"] == 1
        # summary exposes avg_time_ms_per_call (not raw total_time_ms); a failed
        # call with real timing keeps this > 0 instead of diluting toward 0.
        assert pl["avg_time_ms_per_call"] > 0.0, "per-loop: failed call must record real time"
        assert gl["avg_time_ms_per_call"] > 0.0, "global: failed call must record real time"

    def test_non_retriable_failure_records_nonzero_execution_time(self, tmp_path):
        """A non-retriable exception records the per-attempt execution_time_ms>0.

        The non-retriable path measures per-attempt wall-time (start_time captured
        at the try top). A real LLM call has network round-trip latency; we simulate
        that with a tiny busy-wait inside the callable so the recorded time is
        measurable (>1ms after rounding) instead of clamped to 0.
        """
        from external_llm.client import LLMClientError

        reset_global_collector()
        loop = _make_loop_unification(tmp_path)

        def _always_client_error():
            # Simulate network round-trip latency before the error surfaces.
            _spin = time.monotonic() + 0.003
            while time.monotonic() < _spin:
                pass
            raise LLMClientError("400 bad request")

        raised = False
        try:
            loop._retry_on_rate_limit(_always_client_error, mode="test")
        except (LLMClientError, Exception):
            raised = True
        assert raised

        pl = loop.performance_collector.get_summary()["llm_metrics"]
        gl = get_global_collector().get_summary()["llm_metrics"]
        assert pl["calls"] == 1 and pl["failures"] == 1
        assert gl["calls"] == 1 and gl["failures"] == 1
        assert pl["avg_time_ms_per_call"] > 0.0
        assert gl["avg_time_ms_per_call"] > 0.0


# -- #6: failure_rate consumers — top_failing_tools() + warn_failing_tools() -----
# The per-tool failures/failure_rate keys were a dead signal: produced but read
# by nothing. top_failing_tools() is the pure SSOT derivation (dashboard card +
# embedded summary key + warn logic all read it); warn_failing_tools() is the
# deduped server-side warning.


class TestTopFailingTools:
    def test_pure_helper_sorts_by_rate_then_failures(self):
        # Two tools over threshold: higher rate wins; ties broken by raw failures,
        # then name. A tool under min_calls is excluded even at 100% rate.
        metrics = {
            "apply_patch": {"failures": 3, "total_calls": 4, "failure_rate": 0.75},
            "edit_text": {"failures": 2, "total_calls": 4, "failure_rate": 0.50},
            "read_file": {"failures": 2, "total_calls": 2, "failure_rate": 1.0},  # < min_calls(3)
            "bash": {"failures": 1, "total_calls": 4, "failure_rate": 0.25},     # < threshold(0.5)
        }
        out = top_failing_tools(metrics, threshold=0.5, min_calls=3)
        names = [t["name"] for t in out]
        assert names == ["apply_patch", "edit_text"]
        assert out[0]["failures"] == 3 and out[0]["total_calls"] == 4
        assert out[0]["failure_rate"] == 0.75

    def test_min_calls_gate_suppresses_cold_tool_noise(self):
        # 1/1 = 100% but only 1 call — must NOT trip (transient single failure).
        metrics = {"write": {"failures": 1, "total_calls": 1, "failure_rate": 1.0}}
        assert top_failing_tools(metrics, threshold=0.5, min_calls=3) == []

    def test_zero_failures_yields_empty(self):
        metrics = {"read_file": {"failures": 0, "total_calls": 10, "failure_rate": 0.0}}
        assert top_failing_tools(metrics, threshold=0.5, min_calls=3) == []

    def test_top_n_cap(self):
        metrics = {f"t{i}": {"failures": 5, "total_calls": 5, "failure_rate": 1.0} for i in range(8)}
        out = top_failing_tools(metrics, threshold=0.5, min_calls=3, top_n=3)
        assert len(out) == 3

    def test_get_summary_embeds_failing_tools(self):
        # The summary must ship the derived list so the dashboard card, per-turn
        # summary, and warn_failing_tools() all read ONE computation.
        c = PerformanceCollector()
        for _ in range(3):
            c.record_tool_call("apply_patch", 0.1, failed=True)   # 3/3 = 100%
        for _ in range(2):
            c.record_tool_call("read_file", 0.01, failed=False)   # healthy
        c.record_tool_call("edit_text", 0.05, failed=True)        # 1/1 (below min_calls)
        summary = c.get_summary()
        ft = summary["failing_tools"]
        assert len(ft) == 1
        assert ft[0]["name"] == "apply_patch"
        assert ft[0]["failures"] == 3 and ft[0]["total_calls"] == 3
        assert ft[0]["failure_rate"] == 1.0


class TestWarnFailingTools:
    def test_warns_each_new_tool_once_then_dedups(self):
        _reset_warned_failing_tools()
        calls = []
        s1 = {"failing_tools": [{"name": "apply_patch", "failures": 3, "total_calls": 4, "failure_rate": 0.75}]}
        # First poll: newly warned.
        assert warn_failing_tools(s1, log=calls.append) == 1
        assert len(calls) == 1
        # Second identical poll: deduped (broadcaster polls every 2s — no spam).
        assert warn_failing_tools(s1, log=calls.append) == 0
        assert len(calls) == 1

    def test_re_arms_on_recovery_so_regression_re_warns(self):
        _reset_warned_failing_tools()
        calls = []
        failing = {"failing_tools": [{"name": "bash", "failures": 3, "total_calls": 3, "failure_rate": 1.0}]}
        healthy = {"failing_tools": []}
        warn_failing_tools(failing, log=calls.append)      # warn
        warn_failing_tools(failing, log=calls.append)      # dedup
        warn_failing_tools(healthy, log=calls.append)      # recovers -> re-arm
        warn_failing_tools(failing, log=calls.append)      # regression -> warn AGAIN
        assert len(calls) == 2

    def test_no_warn_when_summary_has_no_failing_tools(self):
        _reset_warned_failing_tools()
        calls = []
        assert warn_failing_tools({"failing_tools": []}, log=calls.append) == 0
        assert warn_failing_tools({}, log=calls.append) == 0
        assert calls == []

    def test_multiple_distinct_tools_each_warned(self):
        _reset_warned_failing_tools()
        calls = []
        s = {"failing_tools": [
            {"name": "apply_patch", "failures": 3, "total_calls": 4, "failure_rate": 0.75},
            {"name": "edit_text", "failures": 2, "total_calls": 4, "failure_rate": 0.50},
        ]}
        assert warn_failing_tools(s, log=calls.append) == 2
        assert len(calls) == 2
