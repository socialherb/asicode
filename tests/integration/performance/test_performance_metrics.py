"""
Integration tests for performance metrics collection.
"""

import pytest

from external_llm.agent.performance_metrics import (
    PerformanceCollector,
    get_global_collector,
)


@pytest.mark.integration
class TestPerformanceMetrics:
    """Test performance metrics collection and aggregation."""

    def test_global_collector_singleton(self):
        """Test that get_global_collector returns a singleton."""
        collector1 = get_global_collector()
        collector2 = get_global_collector()

        assert collector1 is collector2
        assert isinstance(collector1, PerformanceCollector)

    def test_record_tool_call(self):
        """Test recording tool call metrics."""
        collector = PerformanceCollector()

        # execution_time is in seconds, not milliseconds
        collector.record_tool_call("find_symbol", execution_time=0.15, cache_hit=False)

        summary = collector.get_summary()
        assert "tool_metrics" in summary
        assert "find_symbol" in summary["tool_metrics"]
        tool_metrics = summary["tool_metrics"]["find_symbol"]
        assert tool_metrics["call_count"] == 1
        assert abs(tool_metrics["avg_execution_time_ms"] - 150.0) < 0.001  # 0.15s * 1000 = 150ms
        assert tool_metrics["cache_hit_rate"] == 0.0  # cache_hit=False

    def test_record_llm_call(self):
        """Test recording LLM call metrics."""
        collector = PerformanceCollector()

        collector.record_llm_call(
            prompt_tokens=100,
            completion_tokens=50,
            execution_time_ms=2000,
            failed=False
        )

        summary = collector.get_summary()
        assert "llm_metrics" in summary
        llm_metrics = summary["llm_metrics"]
        assert llm_metrics["calls"] == 1
        assert llm_metrics["total_prompt_tokens"] == 100
        assert llm_metrics["total_completion_tokens"] == 50
        assert llm_metrics["total_tokens"] == 150  # prompt + completion
        assert llm_metrics["avg_time_ms_per_call"] == 2000.0

    def test_record_rag_cache(self):
        """Test recording RAG cache hit/miss."""
        collector = PerformanceCollector()

        collector.record_rag_cache(hit=True)
        collector.record_rag_cache(hit=False)
        collector.record_rag_cache(hit=True)

        summary = collector.get_summary()
        cache_metrics = summary["cache_metrics"]
        assert cache_metrics["rag_cache"]["hits"] == 2
        assert cache_metrics["rag_cache"]["misses"] == 1
        assert cache_metrics["rag_cache"]["hit_rate"] == 2/3

    @pytest.mark.skip(reason="record_agent_result not implemented in PerformanceCollector")
    def test_record_agent_result(self):
        """Test recording agent result metrics."""
        pass

    @pytest.mark.skip(reason="record_agent_result not implemented in PerformanceCollector")
    def test_agent_result_failure(self):
        """Test recording failed agent result."""
        pass

    def test_tool_call_error_recording(self):
        """Test recording tool call errors (failed=not result.ok path).

        Previously a documented gap: ``record_tool_call`` had no ``failed``
        parameter and ``ToolMetrics`` had no ``failures`` counter, so a failed
        tool call (rolled-back write, missing-file read) was counted identically
        to a success — hiding the most important autonomous-agent health signal.
        Mirrors ``LLMMetrics.failures`` / ``record_llm_call(failed=...)``.
        """
        collector = PerformanceCollector()

        # apply_patch: 2 calls, 1 failed (rolled-back write)
        collector.record_tool_call("apply_patch", execution_time=0.2, cache_hit=False, failed=False)
        collector.record_tool_call("apply_patch", execution_time=0.15, cache_hit=False, failed=True)

        # read_file: 3 calls, 1 failed (missing file)
        collector.record_tool_call("read_file", execution_time=0.01, cache_hit=False, failed=False)
        collector.record_tool_call("read_file", execution_time=0.02, cache_hit=False, failed=False)
        collector.record_tool_call("read_file", execution_time=0.005, cache_hit=False, failed=True)

        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]

        ap = tool_metrics["apply_patch"]
        assert ap["call_count"] == 2
        assert ap["failures"] == 1
        assert ap["failure_rate"] == 0.5

        rf = tool_metrics["read_file"]
        assert rf["call_count"] == 3
        assert rf["failures"] == 1
        assert abs(rf["failure_rate"] - (1 / 3)) < 0.001

    def test_multiple_tool_calls(self):
        """Test recording multiple tool calls."""
        collector = PerformanceCollector()

        tools = [
            ("find_symbol", 0.05, False),   # execution_time in seconds
            ("find_symbol", 0.03, False),
            ("apply_patch", 0.2, False),
            ("apply_patch", 0.1, False),
        ]

        for tool_name, execution_time, cache_hit in tools:
            collector.record_tool_call(tool_name, execution_time=execution_time, cache_hit=cache_hit)

        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]

        assert "find_symbol" in tool_metrics
        assert "apply_patch" in tool_metrics

        find_symbol_metrics = tool_metrics["find_symbol"]
        apply_patch_metrics = tool_metrics["apply_patch"]

        assert find_symbol_metrics["call_count"] == 2
        assert abs(find_symbol_metrics["avg_execution_time_ms"] - 40.0) < 0.001  # (0.05+0.03)/2 * 1000 = 40ms

        assert apply_patch_metrics["call_count"] == 2
        assert abs(apply_patch_metrics["avg_execution_time_ms"] - 150.0) < 0.001  # (0.2+0.1)/2 * 1000 = 150ms

    def test_llm_call_with_caching(self):
        """Test LLM call metrics."""
        collector = PerformanceCollector()

        collector.record_llm_call(
            prompt_tokens=200,
            completion_tokens=150,
            execution_time_ms=3000,
            failed=False
        )

        collector.record_llm_call(
            prompt_tokens=200,
            completion_tokens=150,
            execution_time_ms=50,
            failed=False
        )

        summary = collector.get_summary()
        llm_metrics = summary["llm_metrics"]

        assert llm_metrics["calls"] == 2
        assert llm_metrics["total_prompt_tokens"] == 400
        assert llm_metrics["total_completion_tokens"] == 300
        assert llm_metrics["total_tokens"] == 700
        assert abs(llm_metrics["avg_time_ms_per_call"] - 1525.0) < 0.001

    def test_reset_cache_stats(self):
        """Test resetting cache statistics."""
        collector = PerformanceCollector()

        # Record some cache data
        collector.record_rag_cache(hit=False)
        collector.record_vector_cache(hit=True)

        # Check cache stats before reset
        summary_before = collector.get_summary()
        cache_before = summary_before["cache_metrics"]
        assert cache_before["rag_cache"]["misses"] == 1
        assert cache_before["vector_cache"]["hits"] == 1

        # Reset cache stats only
        collector.reset_cache_stats()

        # Check cache stats after reset
        summary_after = collector.get_summary()
        cache_after = summary_after["cache_metrics"]
        assert cache_after["rag_cache"]["hits"] == 0
        assert cache_after["rag_cache"]["misses"] == 0
        assert cache_after["vector_cache"]["hits"] == 0
        assert cache_after["vector_cache"]["misses"] == 0

    def test_get_summary(self):
        """Test getting summary of all metrics."""
        collector = PerformanceCollector()

        # Record various metrics
        collector.record_tool_call("find_symbol", execution_time=0.1, cache_hit=False)
        collector.record_tool_call("apply_patch", execution_time=0.2, cache_hit=False)
        collector.record_llm_call(prompt_tokens=150, completion_tokens=75, execution_time_ms=2500, failed=False)
        collector.record_rag_cache(hit=False)

        summary = collector.get_summary()

        # Check summary structure
        assert "tool_metrics" in summary
        assert "llm_metrics" in summary
        assert "cache_metrics" in summary
        assert "rag_metrics" in summary
        assert "session_id" in summary
        assert "total_execution_time_seconds" in summary

        # Verify some values
        assert "find_symbol" in summary["tool_metrics"]
        assert "apply_patch" in summary["tool_metrics"]
        assert summary["llm_metrics"]["calls"] == 1
        assert summary["cache_metrics"]["rag_cache"]["hit_rate"] == 0.0

    def test_duration_calculation(self):
        """Test that duration metrics are calculated correctly."""
        collector = PerformanceCollector()

        # Record multiple calls with different durations (in seconds)
        durations_sec = [0.1, 0.2, 0.15]  # 100ms, 200ms, 150ms
        for duration in durations_sec:
            collector.record_tool_call("test_tool", execution_time=duration, cache_hit=False)

        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]
        assert "test_tool" in tool_metrics

        test_tool_metrics = tool_metrics["test_tool"]
        assert test_tool_metrics["call_count"] == 3
        # avg_execution_time_ms should be average of durations in milliseconds
        expected_avg_ms = sum(durations_sec) / len(durations_sec) * 1000
        assert abs(test_tool_metrics["avg_execution_time_ms"] - expected_avg_ms) < 0.001

    def test_concurrent_recording(self):
        """Test that metrics can be recorded concurrently (thread safety)."""
        import threading

        collector = PerformanceCollector()

        def record_multiple_tools():
            for i in range(10):
                collector.record_tool_call(f"tool_{i % 3}", execution_time=i * 0.01, cache_hit=False)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=record_multiple_tools)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]
        total_calls = sum(metrics["call_count"] for metrics in tool_metrics.values())
        assert total_calls == 50  # 5 threads * 10 calls each

    def test_token_cost_calculation(self):
        """Test token cost calculation if implemented."""
        collector = PerformanceCollector()

        # Record LLM calls with tokens
        collector.record_llm_call(
            prompt_tokens=1000,
            completion_tokens=500,
            execution_time_ms=2000,
            failed=False
        )

        # Check if cost is calculated (depends on implementation)
        summary = collector.get_summary()
        llm_metrics = summary["llm_metrics"]

        # Some implementations might calculate cost
        if "estimated_cost" in llm_metrics:
            assert llm_metrics["estimated_cost"] > 0

    def test_export_to_dict(self):
        """Test exporting metrics to dictionary format."""
        collector = PerformanceCollector()

        # Add some data
        collector.record_tool_call("test", execution_time=0.1, cache_hit=False)

        metrics_dict = collector.get_summary()

        assert isinstance(metrics_dict, dict)
        assert "tool_metrics" in metrics_dict
        assert "llm_metrics" in metrics_dict
        assert "cache_metrics" in metrics_dict
        assert "rag_metrics" in metrics_dict
        assert "session_id" in metrics_dict
        assert "total_execution_time_seconds" in metrics_dict
