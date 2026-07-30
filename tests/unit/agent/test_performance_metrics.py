"""Unit tests for performance_metrics.py — ToolMetrics aggregation (#2),
rag/vector cache-channel decoupling (#3), tool failure recording (#4), and
thread-safety of get_summary / reset (#5)."""

import threading
import time
from unittest.mock import patch

import pytest

from external_llm.agent.performance_metrics import (
    LLMMetrics,
    PerformanceCollector,
    ToolMetrics,
    _reset_warned_failing_llm,
    _reset_warned_failing_tools,
    _reset_warned_slow_llm,
    _reset_warned_slow_tools,
    get_global_collector,
    reset_global_collector,
    top_failing_llm,
    top_failing_tools,
    top_slow_llm,
    top_slow_tools,
    warn_failing_llm,
    warn_failing_tools,
    warn_slow_llm,
    warn_slow_tools,
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


# -- #3: record_tool_call must NOT pollute the rag/vector cache channels ---------


class TestFileCacheDecoupling:
    def test_record_tool_call_does_not_feed_cache_channels(self):
        # cache_hit passed to record_tool_call is the ToolResultCache hit flag,
        # NOT a rag/vector cache hit. Feeding it into cache_metrics would
        # duplicate the per-tool counters and the tool_result_cache channel and
        # distort overall_hit_rate.
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01, cache_hit=True)
        c.record_tool_call("edit_text", 0.02, cache_hit=False)
        rag = c.cache_metrics.get_stats("rag")
        assert rag["hits"] == 0 and rag["misses"] == 0 and rag["total"] == 0
        vec = c.cache_metrics.get_stats("vector")
        assert vec["hits"] == 0 and vec["misses"] == 0 and vec["total"] == 0

    def test_per_tool_cache_granularity_preserved(self):
        # Removing the file_cache feed must not lose per-tool cache stats.
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01, cache_hit=True)
        c.record_tool_call("read_file", 0.01, cache_hit=False)
        m = c.tool_metrics["read_file"]
        assert m.cache_hits == 1 and m.cache_misses == 1
        assert m.cache_hit_rate == 0.5


# -- cache_hit 3-state: None (non-cacheable) must NOT pollute per-tool rate -----
# Regression: record_tool_call() used to treat every non-hit as a miss, so write
# /serial tools (never probed by the cache) accumulated cache_misses and read a
# structurally-faked 0% hit rate. The fix makes cache_hit Optional[bool]=None the
# "not applicable" signal — None contributes NEITHER a hit nor a miss.


class TestCacheOutcomeThreeState:
    def test_none_does_not_record_hit_or_miss(self):
        c = PerformanceCollector()
        c.record_tool_call("edit_text", 0.01, cache_hit=None)
        m = c.tool_metrics["edit_text"]
        assert m.cache_hits == 0 and m.cache_misses == 0
        # The ToolMetrics.cache_hit_rate property now returns None (not 0.0) for a
        # zero denominator, matching the SHIPPED summary semantics (get_summary
        # emits None). None = "not applicable"; 0.0 is reserved for a REAL 0% (1
        # miss, 0 hits) so a future reader cannot mistake "never probed" for
        # "always misses".
        assert m.cache_hit_rate is None

    def test_default_is_none_so_uncalled_cache_does_not_pollute(self):
        # The default is None (safe): a caller that forgets to classify
        # cacheability cannot reintroduce the contamination.
        c = PerformanceCollector()
        c.record_tool_call("edit_text", 0.01)  # no cache_hit arg
        m = c.tool_metrics["edit_text"]
        assert m.cache_hits == 0 and m.cache_misses == 0

    def test_non_cacheable_tool_keeps_zero_denominator_alongside_cacheable(self):
        # A realistic mix: write tool (None) + read tool (True/False). The write
        # tool must read 0/0 while the read tool records normally — proving the
        # write tool is no longer dragged into the per-tool cache stats.
        c = PerformanceCollector()
        c.record_tool_call("edit_text", 0.05, cache_hit=None)   # non-cacheable
        c.record_tool_call("edit_text", 0.05, cache_hit=None)
        c.record_tool_call("read_file", 0.01, cache_hit=True)   # cacheable hit
        c.record_tool_call("read_file", 0.01, cache_hit=False)  # cacheable miss
        et = c.tool_metrics["edit_text"]
        rf = c.tool_metrics["read_file"]
        assert et.cache_hits == 0 and et.cache_misses == 0
        assert rf.cache_hits == 1 and rf.cache_misses == 1
        assert rf.cache_hit_rate == 0.5

    def test_summary_emits_zero_cache_for_non_cacheable_tool(self):
        c = PerformanceCollector()
        c.record_tool_call("apply_patch", 0.1, cache_hit=None, failed=False)
        s = c.get_summary()
        ap = s["tool_metrics"]["apply_patch"]
        assert ap["cache_hits"] == 0 and ap["cache_misses"] == 0
        # Summary dict ships None (not 0.0) for a never-probed tool: 0.0 would be
        # read as "0% hit rate" and mistaken for "always misses". None = N/A.
        assert ap["cache_hit_rate"] is None

    def test_summary_emits_none_for_zero_denominator_distinct_from_real_zero(self):
        # A cacheable tool with a real 0% (1 miss, 0 hits) must ship 0.0; a
        # never-probed tool (0/0) must ship None. The two are NOT the same signal.
        c = PerformanceCollector()
        c.record_tool_call("read_file", 0.01, cache_hit=False)   # real miss → 0.0
        c.record_tool_call("apply_patch", 0.1, cache_hit=None)    # never probed → None
        s = c.get_summary()["tool_metrics"]
        assert s["read_file"]["cache_hit_rate"] == 0.0
        assert s["apply_patch"]["cache_hit_rate"] is None


class TestIsResultCacheableProbeGuard:
    """``is_result_cacheable`` must mirror _dispatch_impl's probe guard EXACTLY:
    it returns True only when BOTH (a) the tool is read-only AND (b) the cache is
    actually live (``_tool_result_cache is not None``). Previously it checked only
    the set membership, so with the cache disabled
    (``tool_result_cache_enabled=False`` → ``_tool_result_cache`` is None) every
    read-only tool was classified "cacheable" → the recorders counted each call as
    a miss → a fake structural 0% per-tool cache_hit_rate, the exact contamination
    the 3-state ``cache_hit`` recording was built to prevent.
    """

    def _registry(self, tmp_path, *, cache_enabled):
        import subprocess
        from pathlib import Path
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        repo = Path(tmp_path)
        for c in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.com"],
                  ["git", "config", "user.name", "t"]):
            subprocess.run(c, cwd=str(repo), capture_output=True)
        (repo / "f.txt").write_text("x\n")
        subprocess.run(["git", "add", "f.txt"], cwd=str(repo), capture_output=True)
        subprocess.run(["git", "commit", "-qm", "b"], cwd=str(repo), capture_output=True)
        cfg = AgentConfig(
            max_turns=1, planning_enabled=False, rag_enabled=False,
            tool_result_cache_enabled=cache_enabled,
        )
        return ToolRegistry(str(repo), cfg)

    def test_read_only_tool_cacheable_when_cache_live(self, tmp_path):
        reg = self._registry(tmp_path, cache_enabled=True)
        assert reg._tool_result_cache is not None
        assert reg.is_result_cacheable("read_file") is True
        assert reg.is_result_cacheable("grep") is True

    def test_read_only_tool_not_cacheable_when_cache_disabled(self, tmp_path):
        # THE FIX: cache disabled → read-only tool is never probed, so it must NOT
        # be classified cacheable (else recorders emit a fake miss → 0%).
        reg = self._registry(tmp_path, cache_enabled=False)
        assert reg._tool_result_cache is None
        assert reg.is_result_cacheable("read_file") is False
        assert reg.is_result_cacheable("grep") is False

    def test_write_and_serial_tools_never_cacheable(self, tmp_path):
        reg = self._registry(tmp_path, cache_enabled=True)
        for t in ("edit_text", "apply_patch", "modify_symbol", "ask_user", "job"):
            assert reg.is_result_cacheable(t) is False


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


# -- #7: recent_failure_rate — sliding-window live health (the warn GATE) -------
# The cumulative failure_rate (failures/total_calls) is diluted toward 0 over a
# long run: a tool that succeeded 1000× then fails its next 20 calls reads ~2%
# cumulative and NEVER trips the 0.50 health gate. recent_failure_rate tracks the
# last N calls (bounded deque) so a freshly-broken tool trips within ~N calls.


class TestRecentFailureRate:
    def test_recent_rate_zero_when_no_calls(self):
        m = ToolMetrics(name="unused")
        assert m.recent_calls == 0
        assert m.recent_failures == 0
        assert m.recent_failure_rate == 0.0  # no division-by-zero

    def test_recent_rate_tracks_current_health_not_diluted(self):
        # THE core scenario: 1000 successes then 20 failures. Cumulative rate is
        # ~2% (would NEVER trip a 0.50 gate); recent rate over a 30-window is
        # 20/30 ≈ 0.67 (trips). The warn/card must fire off the RECENT value.
        c = PerformanceCollector()
        for _ in range(1000):
            c.record_tool_call("read_file", 0.01, failed=False)
        for _ in range(20):
            c.record_tool_call("read_file", 0.01, failed=True)
        s = c.get_summary()["tool_metrics"]["read_file"]
        assert s["failures"] == 20 and s["total_calls"] == 1020
        assert abs(s["failure_rate"] - 20 / 1020) < 1e-9  # cumulative, diluted
        assert s["recent_calls"] == 30          # capped at the window
        assert s["recent_failures"] == 20       # all recent calls failed
        assert abs(s["recent_failure_rate"] - 20 / 30) < 1e-9
        # The gate fires (recent ≥ 0.50) where cumulative never would.
        ft = c.get_summary()["failing_tools"]
        assert len(ft) == 1 and ft[0]["name"] == "read_file"
        assert ft[0]["recent_failure_rate"] >= 0.50

    def test_window_evicts_old_samples(self):
        # Fill the window past its cap: old outcomes drop, recent rate reflects
        # only the last N calls. A tool that failed early then fully recovered
        # must read recent_failure_rate == 0.0 (re-armed), even though cumulative
        # failures are still nonzero.
        c = PerformanceCollector()
        for _ in range(40):  # > window(30) of failures
            c.record_tool_call("bash", 0.01, failed=True)
        for _ in range(35):  # > window(30) of recoveries — pushes failures out
            c.record_tool_call("bash", 0.01, failed=False)
        s = c.get_summary()["tool_metrics"]["bash"]
        assert s["failures"] == 40          # cumulative survives
        assert s["recent_calls"] == 30      # capped
        assert s["recent_failures"] == 0    # all evicted
        assert s["recent_failure_rate"] == 0.0
        assert c.get_summary()["failing_tools"] == []  # recovered → not flagged

    def test_partial_recovery_lowers_recent_rate(self):
        # 15 fail + 15 success in the window (N=30): recent = 0.5, right at the
        # threshold. Cumulative over a longer history would be far lower.
        c = PerformanceCollector()
        for _ in range(15):
            c.record_tool_call("grep", 0.01, failed=True)
        for _ in range(15):
            c.record_tool_call("grep", 0.01, failed=False)
        s = c.get_summary()["tool_metrics"]["grep"]
        assert s["recent_calls"] == 30
        assert abs(s["recent_failure_rate"] - 0.5) < 1e-9

    def test_bounded_memory_does_not_grow_with_calls(self):
        # The deque maxlen caps per-tool memory regardless of call count — the
        # 12h+ unbounded-list RAM leak (insight) cannot recur here.
        c = PerformanceCollector()
        for _ in range(100000):
            c.record_tool_call("read_file", 0.001, failed=False)
        m = c.tool_metrics["read_file"]
        assert len(m._recent_outcomes) == _window_cap()  # capped, not 100000


def _window_cap():
    from external_llm.agent.config.thresholds import config as _cfg
    return _cfg.scores.TOOL_FAILURE_RATE_WINDOW
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

    def test_gate_uses_recent_not_cumulative(self):
        # THE contract change: the gate is recent_failure_rate, NOT cumulative.
        # - "recovered": huge cumulative failures but recent window all-success →
        #   recent_failure_rate 0.0 → must NOT be flagged, even though cumulative
        #   failure_rate is sky-high.
        # - "just_broke": tiny cumulative rate (diluted) but recent window all-fail
        #   → recent_failure_rate 1.0 → MUST be flagged.
        metrics = {
            "recovered": {
                "failures": 500, "total_calls": 1000, "failure_rate": 0.5,
                "recent_calls": 30, "recent_failures": 0, "recent_failure_rate": 0.0,
            },
            "just_broke": {
                "failures": 20, "total_calls": 1020, "failure_rate": 0.0196,
                "recent_calls": 30, "recent_failures": 20, "recent_failure_rate": 0.667,
            },
        }
        out = top_failing_tools(metrics, threshold=0.5, min_calls=3)
        names = [t["name"] for t in out]
        assert names == ["just_broke"]          # recovered excluded despite cum 0.5
        assert out[0]["recent_failure_rate"] == 0.667
        # The entry still carries the cumulative rate for display context.
        assert out[0]["failure_rate"] == 0.0196

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
        assert ft[0]["failure_rate"] == 1.0          # cumulative (display)
        assert ft[0]["recent_failure_rate"] == 1.0   # the gate field is shipped
        assert ft[0]["recent_calls"] == 3 and ft[0]["recent_failures"] == 3


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

    def test_warn_message_reports_recent_rate_and_window(self):
        # The warn text must surface the RECENT rate (the live signal that tripped
        # the gate) and the windowed count, not the diluted lifetime average.
        _reset_warned_failing_tools()
        calls = []
        s = {"failing_tools": [{
            "name": "read_file", "failures": 20, "total_calls": 1020,
            "failure_rate": 0.0196,
            "recent_calls": 30, "recent_failures": 20, "recent_failure_rate": 0.667,
        }]}
        warn_failing_tools(s, log=calls.append)
        assert len(calls) == 1
        msg = calls[0]
        assert "recent" in msg
        assert "read_file" in msg
        assert "20/30" in msg          # windowed count, not 20/1020
        assert "67%" in msg            # recent rate, not 2%
# ── LLM-side symmetry: LLMMetrics carries the SAME recent_failure_rate live-health
#    signal as ToolMetrics, plus top_failing_llm() / warn_failing_llm() mirrors.
#    A provider that just started rate-limiting / 5xx-ing must trip the gate within
#    ~N calls regardless of how many successes preceded it (the cumulative rate is
#    diluted toward 0 over a long autonomous run) — exactly the tool-side contract.
class TestRecentFailureRateLLM:
    def test_recent_rate_zero_when_no_calls(self):
        m = LLMMetrics()
        assert m.recent_calls == 0
        assert m.recent_failures == 0
        assert m.recent_failure_rate == 0.0  # no division-by-zero

    def test_recent_rate_tracks_current_health_not_diluted(self):
        # THE core scenario (mirror of the tool side): 1000 successes then 20 failures.
        # Cumulative ≈ 2% (never trips 0.50); recent over the 30-window ≈ 0.67 (trips).
        c = PerformanceCollector()
        for _ in range(1000):
            c.record_llm_call(provider="ollama", prompt_tokens=10, failed=False)
        for _ in range(20):
            c.record_llm_call(provider="ollama", prompt_tokens=10, failed=True)
        s = c.get_summary()["llm_metrics"]
        assert s["calls"] == 1020 and s["failures"] == 20
        assert abs(s["failure_rate"] - 20 / 1020) < 1e-9   # cumulative, diluted
        assert s["recent_calls"] == 30
        assert s["recent_failures"] == 20
        assert abs(s["recent_failure_rate"] - 20 / 30) < 1e-9
        fl = c.get_summary()["failing_llm"]
        assert len(fl) == 1 and fl[0]["name"] == "ollama"
        assert fl[0]["recent_failure_rate"] >= 0.50

    def test_window_evicts_old_samples(self):
        c = PerformanceCollector()
        for _ in range(40):
            c.record_llm_call(failed=True)
        for _ in range(35):  # pushes the early failures out of the window
            c.record_llm_call(failed=False)
        s = c.get_summary()["llm_metrics"]
        assert s["failures"] == 40          # cumulative survives
        assert s["recent_calls"] == 30      # capped
        assert s["recent_failures"] == 0    # all evicted
        assert s["recent_failure_rate"] == 0.0
        assert c.get_summary()["failing_llm"] == []  # recovered → not flagged

    def test_healthy_provider_not_flagged(self):
        c = PerformanceCollector()
        for _ in range(50):
            c.record_llm_call(failed=False)
        assert c.get_summary()["failing_llm"] == []

    def test_cold_provider_below_min_calls_not_flagged(self):
        # A single failure on a 1-call stream (1/1 = 100%) must not permanently flag
        # the provider — the min_calls floor applies, exactly as for tools.
        c = PerformanceCollector()
        c.record_llm_call(failed=True)
        assert c.get_summary()["failing_llm"] == []


class TestTopFailingLLM:
    def _summary(self, **kw):
        base = {"calls": 0, "failures": 0, "failure_rate": 0.0,
                "recent_calls": 0, "recent_failures": 0, "recent_failure_rate": 0.0}
        base.update(kw)
        # top_failing_llm now iterates a PER-PROVIDER dict (summary['llm_providers']),
        # so the fixture wraps one provider's metrics under its provider key.
        return {"ollama": base}

    def test_trips_when_recent_rate_at_threshold(self):
        s = self._summary(calls=10, failures=6, failure_rate=0.6,
                          recent_calls=10, recent_failures=6, recent_failure_rate=0.6)
        out = top_failing_llm(s, threshold=0.50, min_calls=3)
        assert len(out) == 1 and out[0]["name"] == "ollama"
        assert out[0]["recent_failure_rate"] == 0.6

    def test_below_threshold_returns_empty(self):
        s = self._summary(calls=10, failures=2, failure_rate=0.2,
                          recent_calls=10, recent_failures=2, recent_failure_rate=0.2)
        assert top_failing_llm(s, threshold=0.50, min_calls=3) == []

    def test_below_min_calls_returns_empty(self):
        s = self._summary(calls=2, failures=2, failure_rate=1.0,
                          recent_calls=2, recent_failures=2, recent_failure_rate=1.0)
        assert top_failing_llm(s, threshold=0.50, min_calls=3) == []

    def test_no_failures_returns_empty(self):
        s = self._summary(calls=10, failures=0,
                          recent_calls=10, recent_failures=0, recent_failure_rate=0.0)
        assert top_failing_llm(s, threshold=0.50, min_calls=3) == []

    def test_falls_back_to_cumulative_when_recent_absent(self):
        # Older summary snapshot lacking the recent_* fields: the gate falls back to
        # the cumulative rate so this stays callable against a minimal dict.
        s = {"ollama": {"calls": 10, "failures": 8}}
        out = top_failing_llm(s, threshold=0.50, min_calls=3)
        assert len(out) == 1 and abs(out[0]["recent_failure_rate"] - 0.8) < 1e-9


    def test_only_failing_provider_flagged_among_many(self):
        # THE dilution guard (pure-function level): a healthy + a failing provider in
        # one breakdown — only the failing one is returned, sorted first by
        # recent_failure_rate desc. A single aggregate stream would have merged them.
        s = {
            "ollama": {"calls": 490, "failures": 0, "failure_rate": 0.0,
                       "recent_calls": 30, "recent_failures": 0, "recent_failure_rate": 0.0},
            "zai": {"calls": 20, "failures": 20, "failure_rate": 1.0,
                    "recent_calls": 20, "recent_failures": 20, "recent_failure_rate": 1.0},
        }
        out = top_failing_llm(s, threshold=0.50, min_calls=3)
        assert [e["name"] for e in out] == ["zai"]
        assert out[0]["recent_failure_rate"] == 1.0


class TestLLMPerProviderIsolation:
    """THE bug: a failing fallback provider must trip the warn gate even when a
    healthy primary's traffic dominates the aggregate. With one shared deque the
    fallback's failures were diluted below the threshold; per-provider isolation
    keeps each provider's recent window independent (symmetric with per-tool).
    """

    def test_failing_fallback_not_diluted_by_healthy_primary(self):
        c = PerformanceCollector()
        # Fallback provider fails 20 times in a row (100%).
        for _ in range(20):
            c.record_llm_call(provider="zai", failed=True)
        # Then the healthy primary floods in. In a SINGLE shared deque this would
        # push the 20 zai failures out of the maxlen=30 window (recent_rate → 0,
        # never tripping the gate) — the exact dilution blind spot. Per-provider
        # isolation keeps zai's OWN window at 100%.
        for _ in range(40):
            c.record_llm_call(provider="ollama", failed=False)
        s = c.get_summary()
        # THE gate: zai is flagged on its OWN 100% recent rate. A single shared
        # stream would have read a diluted aggregate and missed it entirely.
        fl = s["failing_llm"]
        names = [e["name"] for e in fl]
        assert "zai" in names
        zai = next(e for e in fl if e["name"] == "zai")
        assert zai["recent_failure_rate"] == 1.0
        assert "ollama" not in names  # healthy primary not flagged
        # Cross-check the dilution: the AGGREGATE recent window is dominated by
        # ollama (zai's 20 failures over a 50-sample cross-provider window = 0.4,
        # BELOW the 0.5 threshold) — proving the gate reads per-provider, not the
        # diluted aggregate. A single-stream gate would gate on this and miss zai.
        agg = s["llm_metrics"]
        assert agg["calls"] == 60 and agg["failures"] == 20
        assert agg["recent_failure_rate"] == 0.4
        assert agg["recent_failure_rate"] < 0.5

    def test_per_provider_breakdown_shipped(self):
        c = PerformanceCollector()
        c.record_llm_call(provider="ollama", prompt_tokens=100, failed=False)
        c.record_llm_call(provider="openai", prompt_tokens=50, failed=False)
        prov = c.get_summary()["llm_providers"]
        assert set(prov.keys()) == {"ollama", "openai"}
        assert prov["ollama"]["calls"] == 1 and prov["ollama"]["total_tokens"] == 100
        assert prov["openai"]["calls"] == 1 and prov["openai"]["total_tokens"] == 50
        # Aggregate sums both.
        assert c.get_summary()["llm_metrics"]["total_tokens"] == 150

    def test_provider_normalized_to_lowercase(self):
        c = PerformanceCollector()
        c.record_llm_call(provider="Ollama", failed=True)
        c.record_llm_call(provider="OLLAMA", failed=True)
        prov = c.get_summary()["llm_providers"]
        assert list(prov.keys()) == ["ollama"]
        assert prov["ollama"]["calls"] == 2

    def test_unknown_provider_default(self):
        c = PerformanceCollector()
        c.record_llm_call(failed=False)  # no provider arg
        prov = c.get_summary()["llm_providers"]
        assert list(prov.keys()) == ["unknown"]


class TestWarnFailingLLM:
    def _s(self, rate=0.667, recent_failures=20, recent_calls=30, calls=1020, failures=20):
        return {"failing_llm": [{
            "name": "llm", "calls": calls, "failures": failures,
            "failure_rate": failures / calls,
            "recent_calls": recent_calls, "recent_failures": recent_failures,
            "recent_failure_rate": rate,
        }]}

    def test_warns_once_then_dedups(self):
        _reset_warned_failing_llm()
        calls = []
        assert warn_failing_llm(self._s(), log=calls.append) == 1
        assert len(calls) == 1
        assert warn_failing_llm(self._s(), log=calls.append) == 0  # deduped
        assert len(calls) == 1

    def test_re_arms_on_recovery(self):
        _reset_warned_failing_llm()
        calls = []
        failing = self._s()
        healthy = {"failing_llm": []}
        warn_failing_llm(failing, log=calls.append)     # warn
        warn_failing_llm(failing, log=calls.append)     # dedup
        warn_failing_llm(healthy, log=calls.append)     # recover -> re-arm
        warn_failing_llm(failing, log=calls.append)     # regression -> warn AGAIN
        assert len(calls) == 2

    def test_no_warn_when_healthy(self):
        _reset_warned_failing_llm()
        calls = []
        assert warn_failing_llm({"failing_llm": []}, log=calls.append) == 0
        assert warn_failing_llm({}, log=calls.append) == 0
        assert calls == []

    def test_warn_message_reports_recent_rate(self):
        _reset_warned_failing_llm()
        calls = []
        warn_failing_llm(self._s(rate=0.667, recent_failures=20, recent_calls=30),
                         log=calls.append)
        assert len(calls) == 1
        msg = calls[0]
        assert "LLM provider" in msg
        assert "20/30" in msg            # windowed count
        assert "67%" in msg              # recent rate, not the diluted cumulative %

    def test_llm_dedup_independent_of_tool_dedup(self):
        # Tool and LLM warn state are SEPARATE sets — tripping/re-arming one must not
        # affect the other. A tool regression and an LLM provider regression are
        # distinct operator signals and must not share re-arm bookkeeping.
        _reset_warned_failing_tools()
        _reset_warned_failing_llm()
        tool_calls, llm_calls = [], []
        tool_s = {"failing_tools": [{"name": "bash", "failures": 3, "total_calls": 3,
                                     "failure_rate": 1.0}]}
        assert warn_failing_tools(tool_s, log=tool_calls.append) == 1
        assert warn_failing_llm(self._s(), log=llm_calls.append) == 1
        assert len(tool_calls) == 1 and len(llm_calls) == 1


class TestLatencyPercentile:
    """p50/p95 over the bounded RECENT-latency window (ToolMetrics).

    A sliding window — NOT a uniform reservoir — so the tail tracks CURRENT
    latency and is not diluted toward early fast calls on a long run (the same
    history-dilution reasoning as recent_failure_rate). Constant memory (deque
    maxlen = LATENCY_SAMPLE_WINDOW).
    """

    def test_percentile_empty_is_zero(self):
        m = ToolMetrics(name="t")
        assert m.percentile(50) == 0.0
        assert m.percentile(95) == 0.0

    def test_percentile_single_sample(self):
        m = ToolMetrics(name="t")
        m.record(12.5)
        assert m.percentile(50) == 12.5
        assert m.percentile(95) == 12.5

    def test_median_odd_count(self):
        m = ToolMetrics(name="t")
        for v in (10.0, 20.0, 30.0, 40.0, 50.0):
            m.record(v)
        # median of 5 sorted = 30.0
        assert m.percentile(50) == 30.0

    def test_median_even_count_interpolates(self):
        m = ToolMetrics(name="t")
        for v in (10.0, 20.0, 30.0, 40.0):
            m.record(v)
        # 4 samples, p50 → k=(4-1)*0.5=1.5 → between idx1(20) and idx2(30) → 25.0
        assert m.percentile(50) == 25.0

    def test_p95_picks_tail(self):
        m = ToolMetrics(name="t")
        # 100 fast calls at 5ms then the tail is unchanged; p95 ~ 5ms
        for _ in range(100):
            m.record(5.0)
        assert m.percentile(95) == 5.0
        # Append a clear TAIL MAJORITY (≥5% of the window) of slow outliers; p95
        # must climb into the slow region. (Linear-interpolation percentile: a
        # handful of outliers below the 95th rank interpolate back toward the
        # bulk, so use a real tail share to exercise the high-rank read.)
        for _ in range(20):
            m.record(1000.0)
        assert m.percentile(95) == 1000.0

    def test_window_evicts_to_reflect_recent_not_lifetime(self):
        # The whole point of a sliding window over a uniform reservoir: after a
        # long run of slow calls, the recent window must reflect CURRENT slowness
        # even if the lifetime was mostly fast. Fill past the cap with fast, then
        # flood with slow; p95 must climb (a uniform lifetime sample would stay
        # diluted near the fast bulk).
        from external_llm.agent.config.thresholds import config as _cfg
        cap = _cfg.scores.LATENCY_SAMPLE_WINDOW
        m = ToolMetrics(name="t")
        for _ in range(cap):
            m.record(1.0)          # fill the window with fast calls
        # Evict every fast sample with slow ones
        for _ in range(cap):
            m.record(1000.0)
        assert m.percentile(95) >= 1000.0   # window is now all-slow
        # Constant memory: never more than cap samples retained
        assert len(m._latency_samples) == cap


class TestLLMLatencyPercentile:
    """p50/p95 per provider + the aggregate (merged) tail in get_summary()."""

    def test_per_provider_p50_p95_via_summary(self):
        c = PerformanceCollector()
        # provider "ollama": a tight distribution
        for v in (10, 10, 10, 10, 200):
            c.record_llm_call(provider="ollama", execution_time_ms=float(v))
        s = c.get_summary()
        prov = s["llm_providers"]["ollama"]
        assert prov["p50_ms"] == 10.0          # median
        # Linear-interpolation p95 over [10,10,10,10,200]: k=4*0.95=3.8 → between
        # idx3(10) and idx4(200) = 10 + 190*0.8 = 162. Strictly above the median,
        # reflecting the tail without equalling the raw max.
        assert prov["p50_ms"] < prov["p95_ms"] < 200.0
        assert prov["p95_ms"] == pytest.approx(162.0)



    def test_no_llm_calls_yields_no_aggregate_latency(self):
        c = PerformanceCollector()
        s = c.get_summary()
        assert "p50_ms" not in s["llm_metrics"]
        assert "p95_ms" not in s["llm_metrics"]
        assert s["llm_providers"] == {}


class TestLatencyInToolSummary:
    def test_tool_summary_ships_p50_p95(self):
        c = PerformanceCollector()
        # Inputs in SECONDS — the real caller feeds result.execution_time =
        # time.monotonic() - start_time (seconds), NOT ms. get_summary converts
        # the stored-seconds window to ms at emit (matching avg/min/max keys).
        for v in (0.010, 0.020, 0.030, 0.040, 0.100):
            c.record_tool_call("grep", execution_time=v)
        s = c.get_summary()
        entry = s["tool_metrics"]["grep"]
        assert entry["p50_ms"] == 30.0         # median of 5 → 0.030s * 1000
        # Linear-interpolation p95: k=4*0.95=3.8 → between idx3(0.040) and
        # idx4(0.100) = 0.040 + 0.060*0.8 = 0.088s → 88ms.
        assert entry["p95_ms"] == pytest.approx(88.0)

    def test_tool_p_units_match_avg_min_max(self):
        # Regression for the 1000x unit bug: p50_ms/p95_ms MUST share the ms
        # unit of avg/min/max_execution_time_ms. A tool averaging 0.030s must
        # report p50_ms≈30 (not 0.030) and stay ≥ min_execution_time_ms.
        c = PerformanceCollector()
        for v in (0.020, 0.030, 0.040):
            c.record_tool_call("grep", execution_time=v)
        s = c.get_summary()
        e = s["tool_metrics"]["grep"]
        assert e["min_execution_time_ms"] == pytest.approx(20.0)
        assert e["max_execution_time_ms"] == pytest.approx(40.0)
        assert e["avg_execution_time_ms"] == pytest.approx(30.0)
        # Before the fix this read 0.030 (seconds leaked as "ms") < min (20ms) —
        # an impossible ordering that proves the bug. Now consistent.
        assert e["p50_ms"] == pytest.approx(30.0)
        assert e["min_execution_time_ms"] <= e["p50_ms"] <= e["max_execution_time_ms"]


# -- #1: latency window is SUCCESS-ONLY (failed calls are not latency samples) --
# A failed call's wall time is NOT a completion-latency sample. Mixing it in
# distorts p95 BOTH ways: a fast-fail (429/auth, ~0ms) drags p95 DOWN and hides
# degradation; a slow-fail (timeout, huge ms) spikes p95 UP and conflates "slow"
# with "failing". The failure dimension already lives on recent_failure_rate /
# failures; the latency dimension stays a clean completion-latency read. Each
# test below asserts the value the gate produces AND is constructed so that
# REMOVING the gate (appending failed-call time too) would flip the assertion —
# a built-in perturbation proof the gate is actually doing the work.


class TestLatencySuccessOnly:
    """Tool + LLM latency windows exclude failed-call wall times."""

    def test_failed_call_excluded_from_latency_window(self):
        # A slow-fail (timeout) must NOT enter the latency window. With the gate
        # the window is [5,5,5] → p95=5.0; without it the 30000 would
        # interpolate p95 up toward it (false "slow" signal).
        m = ToolMetrics(name="t")
        m.record(5.0)
        m.record(5.0)
        m.record(5.0)
        m.record(30000.0, failed=True)   # slow-fail — excluded
        assert m.percentile(95) == 5.0
        assert len(m._latency_samples) == 3

    def test_record_tool_call_failed_excludes_latency(self):
        # Same via the public record_tool_call API (the real caller path).
        c = PerformanceCollector()
        c.record_tool_call("grep", execution_time=0.005, failed=False)
        c.record_tool_call("grep", execution_time=0.005, failed=False)
        c.record_tool_call("grep", execution_time=30.0, failed=True)   # timeout
        s = c.get_summary()
        e = s["tool_metrics"]["grep"]
        # Only the two 5ms successes are in the window → p95 = 5ms.
        assert e["p95_ms"] == pytest.approx(5.0)
        # Cumulative aggregates STILL see every call (max = the 30s timeout — a
        # genuine worst-case wall time), proving only the latency window is gated.
        assert e["max_execution_time_ms"] == pytest.approx(30000.0)
        assert e["total_calls"] == 3

    def test_fast_fail_majority_does_not_flatten_p95(self):
        # A tool that mostly fast-fails (429, ~0ms) but whose rare successes are
        # slow. The fast-fails must NOT flood the window and drag p95 down.
        m = ToolMetrics(name="t")
        for _ in range(24):
            m.record(0.1, failed=True)   # 24 fast-fails — excluded
        m.record(200.0)                  # the ONE real completion
        # window is [200] → p95=200.0. Without the gate the 24 fast-fails push
        # this single real completion below the 95th rank (24/25 = 96% are
        # 0.1ms fails) and p95 would read ~0.1 — hiding that real calls take
        # 200ms (false "fast/healthy" signal).
        assert m.percentile(95) == 200.0
        assert len(m._latency_samples) == 1

    def test_fully_failing_tool_yields_zero_percentiles(self):
        # The acknowledged tradeoff: a fully-failing tool has NO success samples,
        # so p50/p95 read 0.0 alongside a 100% failure_rate — honestly "dead, not
        # slow" rather than presenting timeout wall-times as "latency".
        c = PerformanceCollector()
        for _ in range(5):
            c.record_tool_call("bash", execution_time=30.0, failed=True)
        s = c.get_summary()
        e = s["tool_metrics"]["bash"]
        assert e["p50_ms"] == 0.0
        assert e["p95_ms"] == 0.0
        assert e["failure_rate"] == 1.0
        assert e["recent_failure_rate"] == 1.0

    def test_llm_failed_call_excluded_from_latency(self):
        c = PerformanceCollector()
        c.record_llm_call(provider="ollama", execution_time_ms=10.0, failed=False)
        c.record_llm_call(provider="ollama", execution_time_ms=2000.0, failed=True)  # slow-fail
        s = c.get_summary()
        prov = s["llm_providers"]["ollama"]
        # Only the 10ms success is in the window → p50=p95=10.0. Without the
        # gate the 2000ms slow-fail would spike p95 toward it.
        assert prov["p50_ms"] == 10.0
        assert prov["p95_ms"] == 10.0
        # Cumulative avg STILL includes the failed call's time (honest), proving
        # only the latency window is gated.
        assert prov["avg_time_ms_per_call"] == pytest.approx(1005.0)




class TestTopSlowTools:
    """Tests for the pure derivation top_slow_tools()."""

    def test_p95_above_threshold(self):
        metrics = {
            "apply_patch": {"p50_ms": 200.0, "p95_ms": 5000.0, "call_count": 10, "p95_n": 10},
            "edit_text": {"p50_ms": 100.0, "p95_ms": 3000.0, "call_count": 20, "p95_n": 20},
            "read_file": {"p50_ms": 50.0, "p95_ms": 100.0, "call_count": 50, "p95_n": 50},
        }
        out = top_slow_tools(metrics, threshold_ms=4000.0)
        names = [t["name"] for t in out]
        assert names == ["apply_patch"]

    def test_sorted_by_p95_desc(self):
        metrics = {
            "bash": {"p50_ms": 1000.0, "p95_ms": 25000.0, "call_count": 5, "p95_n": 5},
            "web_search": {"p50_ms": 2000.0, "p95_ms": 15000.0, "call_count": 10, "p95_n": 10},
            "apply_patch": {"p50_ms": 100.0, "p95_ms": 6000.0, "call_count": 30, "p95_n": 30},
        }
        out = top_slow_tools(metrics, threshold_ms=5000.0)
        names = [t["name"] for t in out]
        assert names == ["bash", "web_search", "apply_patch"]

    def test_top_n_cap(self):
        metrics = {f"t{i}": {"p50_ms": 100.0, "p95_ms": 10000.0, "call_count": 5, "p95_n": 5} for i in range(8)}
        out = top_slow_tools(metrics, threshold_ms=5000.0, top_n=3)
        assert len(out) == 3

    def test_all_below_threshold_yields_empty(self):
        metrics = {
            "bash": {"p50_ms": 100.0, "p95_ms": 50.0, "call_count": 5, "p95_n": 5},
        }
        assert top_slow_tools(metrics, threshold_ms=5000.0) == []

    def test_missing_p95_is_skipped(self):
        metrics = {
            "bash": {"p50_ms": 100.0, "call_count": 5},  # no p95
        }
        assert top_slow_tools(metrics, threshold_ms=5000.0) == []

    def test_non_dict_value_skipped(self):
        metrics = {"bash": "not a dict"}
        assert top_slow_tools(metrics, threshold_ms=5000.0) == []

    def test_insufficient_samples_skipped(self):
        """A tool with fewer than min_samples successful latency calls is skipped,
        even if its p95 exceeds the threshold."""
        metrics = {
            "bash": {"p50_ms": 1000.0, "p95_ms": 25000.0, "call_count": 5, "p95_n": 1},
            "web_search": {"p50_ms": 2000.0, "p95_ms": 15000.0, "call_count": 10, "p95_n": 10},
        }
        out = top_slow_tools(metrics, threshold_ms=5000.0, min_samples=5)
        names = [t["name"] for t in out]
        assert names == ["web_search"]  # bash has p95_n=1 < 5


class TestTopSlowLLM:
    """Tests for the pure derivation top_slow_llm()."""

    def test_p95_above_threshold(self):
        providers = {
            "openai": {"p50_ms": 5000.0, "p95_ms": 35000.0, "calls": 20, "p95_n": 20},
            "claude": {"p50_ms": 3000.0, "p95_ms": 25000.0, "calls": 30, "p95_n": 30},
        }
        out = top_slow_llm(providers, threshold_ms=30000.0)
        names = [p["name"] for p in out]
        assert names == ["openai"]

    def test_all_healthy_yields_empty(self):
        providers = {
            "openai": {"p50_ms": 2000.0, "p95_ms": 5000.0, "calls": 20, "p95_n": 20},
        }
        assert top_slow_llm(providers, threshold_ms=30000.0) == []

    def test_none_or_empty_providers(self):
        assert top_slow_llm(None, threshold_ms=30000.0) == []
        assert top_slow_llm({}, threshold_ms=30000.0) == []

    def test_insufficient_samples_skipped(self):
        """A provider with fewer than min_samples latency samples is skipped."""
        providers = {
            "openai": {"p50_ms": 5000.0, "p95_ms": 35000.0, "calls": 20, "p95_n": 1},
            "claude": {"p50_ms": 3000.0, "p95_ms": 35000.0, "calls": 30, "p95_n": 30},
        }
        out = top_slow_llm(providers, threshold_ms=30000.0, min_samples=5)
        names = [p["name"] for p in out]
        assert names == ["claude"]  # openai has p95_n=1 < 5


class TestWarnSlowTools:
    """Tests for the deduped log gate warn_slow_tools()."""

    def test_warns_new_tool_then_dedups(self):
        _reset_warned_slow_tools()
        calls = []
        log = lambda msg: calls.append(msg)  # noqa: E731

        s1 = {"slow_tools": [{"name": "bash", "p50_ms": 1000.0, "p95_ms": 25000.0}]}
        n = warn_slow_tools(s1, log=log)
        assert n == 1
        assert len(calls) == 1
        assert "bash" in calls[0]
        assert "25000ms" in calls[0]

        # Second poll: same tool still slow → deduped (no new warn).
        n = warn_slow_tools(s1, log=log)
        assert n == 0
        assert len(calls) == 1  # still 1 — dedup worked

    def test_re_arm_on_recovery(self):
        _reset_warned_slow_tools()
        calls = []
        log = lambda msg: calls.append(msg)  # noqa: E731

        s1 = {"slow_tools": [{"name": "bash", "p50_ms": 1000.0, "p95_ms": 25000.0}]}
        n = warn_slow_tools(s1, log=log)
        assert n == 1

        # Tool recovers (drops out of list) → re-armed.
        s2 = {"slow_tools": []}
        n = warn_slow_tools(s2, log=log)
        assert n == 0

        # Same tool slow again → re-warned.
        n = warn_slow_tools(s1, log=log)
        assert n == 1
        assert len(calls) == 2

    def test_no_warn_when_summary_has_no_slow_tools(self):
        _reset_warned_slow_tools()
        calls = []
        log = lambda msg: calls.append(msg)  # noqa: E731
        assert warn_slow_tools({}, log=log) == 0
        assert warn_slow_tools({"slow_tools": []}, log=log) == 0
        assert len(calls) == 0


class TestWarnSlowLLM:
    """Tests for the deduped log gate warn_slow_llm()."""

    def test_warns_new_provider_then_dedups(self):
        _reset_warned_slow_llm()
        calls = []
        log = lambda msg: calls.append(msg)  # noqa: E731

        s1 = {"slow_llm": [{"name": "openai", "p50_ms": 5000.0, "p95_ms": 35000.0}]}
        n = warn_slow_llm(s1, log=log)
        assert n == 1
        assert len(calls) == 1
        assert "openai" in calls[0]
        assert "35000ms" in calls[0]

        # Second poll: deduped.
        n = warn_slow_llm(s1, log=log)
        assert n == 0
        assert len(calls) == 1

    def test_re_arm_on_recovery(self):
        _reset_warned_slow_llm()
        calls = []
        log = lambda msg: calls.append(msg)  # noqa: E731

        s1 = {"slow_llm": [{"name": "openai", "p50_ms": 5000.0, "p95_ms": 35000.0}]}
        warn_slow_llm(s1, log=log)

        # Provider recovers.
        warn_slow_llm({"slow_llm": []}, log=log)

        # Same provider slow again → re-warned.
        n = warn_slow_llm(s1, log=log)
        assert n == 1
        assert len(calls) == 2

    def test_no_warn_when_empty(self):
        _reset_warned_slow_llm()
        calls = []
        log = lambda msg: calls.append(msg)  # noqa: E731
        assert warn_slow_llm({}, log=log) == 0
        assert warn_slow_llm({"slow_llm": []}, log=log) == 0
        assert len(calls) == 0


class TestSlowToolsInSummary:
    """Tests that get_summary() ships slow_tools / slow_llm keys."""

    def test_summary_embeds_empty_slow_tools(self):
        c = PerformanceCollector()
        summary = c.get_summary()
        assert "slow_tools" in summary
        assert summary["slow_tools"] == []

    def test_summary_embeds_empty_slow_llm(self):
        c = PerformanceCollector()
        summary = c.get_summary()
        assert "slow_llm" in summary
        assert summary["slow_llm"] == []

    def test_slow_tools_triggers_on_high_p95(self):
        # A tool with p95 above the configured threshold should appear in slow_tools.
        c = PerformanceCollector()
        for _ in range(10):
            c.record_tool_call("slow_tool", 10.0, failed=False)  # 10000ms
        for _ in range(10):
            c.record_tool_call("fast_tool", 0.01, failed=False)  # 10ms
        summary = c.get_summary()
        slow = summary["slow_tools"]
        names = [t["name"] for t in slow]
        assert "slow_tool" in names
        assert "fast_tool" not in names

    def test_slow_llm_triggers_on_high_p95(self):
        c = PerformanceCollector()
        for _ in range(10):
            c.record_llm_call("slow_provider", 0, 0, execution_time_ms=50000.0, failed=False)
        for _ in range(10):
            c.record_llm_call("fast_provider", 0, 0, execution_time_ms=1000.0, failed=False)
        summary = c.get_summary()
        slow = summary["slow_llm"]
        names = [p["name"] for p in slow]
        assert "slow_provider" in names
        assert "fast_provider" not in names

    def test_slow_tools_min_samples_wired(self):
        """The LATENCY_P95_MIN_SAMPLES config knob is wired through get_summary().
        A tool with 10 slow successful calls is excluded from slow_tools when
        LATENCY_P95_MIN_SAMPLES is raised to 15 (greater than available samples)."""
        from dataclasses import replace
        from external_llm.agent.config.thresholds import config as _cfg
        import external_llm.agent.performance_metrics as _pm
        c = PerformanceCollector()
        for _ in range(10):
            c.record_tool_call("moderate_tool", 10.0, failed=False)  # 10000ms p95
        # Without patch: 10 samples >= 5 (default) → appears in slow_tools
        summary_before = c.get_summary()
        assert "moderate_tool" in [t["name"] for t in summary_before["slow_tools"]]

        # With patch: replace whole config with modified min_samples=15
        new_scores = replace(_cfg.scores, LATENCY_P95_MIN_SAMPLES=15)
        new_config = replace(_cfg, scores=new_scores)
        with patch.object(_pm, "_threshold_config", new_config):
            summary_after = c.get_summary()
        assert summary_after["slow_tools"] == []

    def test_slow_llm_min_samples_wired(self):
        """Same wiring verification for the LLM path."""
        from dataclasses import replace
        from external_llm.agent.config.thresholds import config as _cfg
        import external_llm.agent.performance_metrics as _pm
        c = PerformanceCollector()
        for _ in range(10):
            c.record_llm_call("moderate_provider", 0, 0, execution_time_ms=50000.0, failed=False)
        # Without patch: appears in slow_llm
        summary_before = c.get_summary()
        assert "moderate_provider" in [p["name"] for p in summary_before["slow_llm"]]

        # With patch: min_samples=15 > 10 → excluded
        new_scores = replace(_cfg.scores, LATENCY_P95_MIN_SAMPLES=15)
        new_config = replace(_cfg, scores=new_scores)
        with patch.object(_pm, "_threshold_config", new_config):
            summary_after = c.get_summary()
        assert summary_after["slow_llm"] == []
