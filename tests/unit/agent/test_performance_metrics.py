"""Unit tests for performance_metrics.py — ToolMetrics aggregation (#2),
file_cache decoupling (#3), and thread-safety of get_summary / reset (#5)."""

import threading

from external_llm.agent.performance_metrics import (
    PerformanceCollector,
    ToolMetrics,
    reset_global_collector,
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
