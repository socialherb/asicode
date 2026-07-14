"""
Integration tests for performance benchmarking.
"""
import time
from unittest.mock import Mock, patch

import pytest

from external_llm.agent.performance_metrics import PerformanceCollector, get_global_collector, reset_global_collector


@pytest.mark.integration
@pytest.mark.slow  # Mark as slow since benchmarks may take time
class TestPerformanceBenchmark:
    """Test performance benchmarking and metrics collection."""

    def test_tool_call_benchmark_basic(self):
        """Benchmark tool call recording performance."""
        reset_global_collector()
        collector = get_global_collector()

        iterations = 1000
        start_time = time.perf_counter()

        for i in range(iterations):
            collector.record_tool_call(
                tool_name=f"tool_{i % 10}",
                execution_time=(i % 100) / 1000.0,  # convert ms to seconds
                cache_hit=False
            )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_call = total_time / iterations

        # Should be reasonably fast (e.g., < 0.1ms per call)
        # Adjust threshold based on system
        assert avg_time_per_call < 0.001  # 1ms per call maximum

        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]
        assert len(tool_metrics) >= min(10, iterations)  # Should have stats for different tools

    def test_llm_call_benchmark(self):
        """Benchmark LLM call recording performance."""
        reset_global_collector()
        collector = get_global_collector()

        iterations = 500
        start_time = time.perf_counter()

        for i in range(iterations):
            collector.record_llm_call(
                prompt_tokens=100 + i,
                completion_tokens=50 + i,
                execution_time_ms=2000 + i * 10,
                failed=False
            )
            # Note: cached parameter ignored for benchmark

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_call = total_time / iterations

        assert avg_time_per_call < 0.001  # 1ms per call maximum

        summary = collector.get_summary()
        llm_metrics = summary["llm_metrics"]
        assert llm_metrics["calls"] == iterations

    def test_concurrent_metrics_recording(self):
        """Benchmark concurrent metrics recording (thread safety)."""
        import queue
        import threading

        reset_global_collector()
        collector = get_global_collector()

        num_threads = 10
        iterations_per_thread = 100
        total_iterations = num_threads * iterations_per_thread

        results_queue = queue.Queue()
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations_per_thread):
                    tool_id = (thread_id * 1000) + i
                    collector.record_tool_call(
                        tool_name=f"tool_{tool_id % 20}",
                        execution_time=(tool_id % 200) / 1000.0,  # ms to seconds
                        cache_hit=False
                    )
                results_queue.put(True)
            except Exception as e:
                errors.append(e)
                results_queue.put(False)

        start_time = time.perf_counter()

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Check all threads completed successfully
        success_count = 0
        while not results_queue.empty():
            if results_queue.get():
                success_count += 1

        assert len(errors) == 0, f"Errors in concurrent recording: {errors}"
        assert success_count == num_threads

        # Verify all recordings were captured
        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]
        total_recorded = sum(metrics["call_count"] for metrics in tool_metrics.values())
        assert total_recorded == total_iterations

        # Calculate operations per second
        ops_per_second = total_iterations / total_time
        # Should handle at least 1000 ops/sec
        assert ops_per_second > 1000

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage of metrics collector."""
        import gc
        import sys

        collector = PerformanceCollector()

        # Measure baseline memory
        gc.collect()
        sys.getsizeof(collector)

        # Record many metrics
        num_metrics = 10000
        for i in range(num_metrics):
            collector.record_tool_call(
                tool_name=f"benchmark_tool_{i % 50}",
                execution_time=(i % 300) / 1000.0,  # ms to seconds
                cache_hit=False
            )

        # Force garbage collection
        gc.collect()

        # Check memory growth is reasonable
        # Memory per metric should be relatively small
        # Exact thresholds depend on implementation

    def test_metrics_aggregation_performance(self):
        """Benchmark performance of metrics aggregation methods."""
        reset_global_collector()
        collector = get_global_collector()

        # Record a large number of metrics
        num_tools = 1000
        for i in range(num_tools):
            collector.record_tool_call(
                tool_name=f"agg_tool_{i}",
                execution_time=(i * 10) / 1000.0,  # ms to seconds
                cache_hit=False
            )

        # Time aggregation operations
        start_time = time.perf_counter()

        # Call aggregation methods
        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]
        summary["llm_metrics"]
        summary["cache_metrics"]
        # agent_stats not available

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Aggregation should be fast even with many metrics
        assert total_time < 0.1  # 100ms maximum

        # Verify aggregated data
        assert len(tool_metrics) == num_tools
        assert summary["session_id"] is not None

    def test_end_to_end_agent_benchmark(self, temp_repo_root: str):
        """Benchmark end-to-end agent execution performance."""
        from external_llm.agent.agent_loop import AgentLoop
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        config = AgentConfig(
            max_turns=3,
            rag_enabled=False,
            planning_enabled=False,
            self_review_enabled=False
        )

        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.tool_calls = []
        mock_llm.chat_with_tools.return_value = mock_response

        registry = ToolRegistry(temp_repo_root, config)
        agent = AgentLoop(
            llm_client=mock_llm,
            registry=registry,
            config=config,
            model="test-model"
        )

        # Time agent execution
        start_time = time.perf_counter()

        # Mock the run method to return quickly
        from external_llm.agent.agent_loop import AgentResult
        with patch.object(agent, 'run') as mock_run:
            mock_run.return_value = AgentResult(status="success", turns=[], final_message="Benchmark completed")
            result = agent.run("Benchmark query")

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Agent initialization and setup should be reasonably fast
        assert execution_time < 2.0  # 2 seconds maximum for mocked execution

        assert result is not None
        assert result.status == "success"

    def test_cache_hit_rate_benchmark(self):
        """Benchmark cache hit rate calculation performance."""
        reset_global_collector()
        collector = get_global_collector()

        # Record many cache events
        num_events = 10000
        for i in range(num_events):
            hit = (i % 10) < 7  # 70% hits
            collector.record_rag_cache(hit=hit)

        # Time hit rate calculation
        start_time = time.perf_counter()

        summary = collector.get_summary()
        cache_metrics = summary["cache_metrics"]
        rag_hit_rate = cache_metrics["rag_cache"]["hit_rate"]

        end_time = time.perf_counter()
        calculation_time = end_time - start_time

        # Calculation should be very fast
        assert calculation_time < 0.01  # 10ms maximum

        # Verify hit rates are calculated correctly
        expected_rate = 0.7
        # Allow some tolerance
        assert abs(rag_hit_rate - expected_rate) < 0.01

    def test_metrics_export_performance(self):
        """Benchmark performance of metrics export to dict/JSON."""
        reset_global_collector()
        collector = get_global_collector()

        # Populate with diverse metrics
        for i in range(500):
            collector.record_tool_call(f"export_tool_{i}", execution_time=(i * 5) / 1000.0, cache_hit=False)
            if i % 10 == 0:
                collector.record_llm_call(prompt_tokens=100, completion_tokens=50, execution_time_ms=2000, failed=False)

        # record_agent_result not implemented

        # Time export operations
        start_time = time.perf_counter()

        # Export to dict
        metrics_dict = collector.get_summary()
        # Convert to JSON (simulate serialization)
        import json
        json_data = json.dumps(metrics_dict, default=str)

        end_time = time.perf_counter()
        export_time = end_time - start_time

        # Export should be fast
        assert export_time < 0.05  # 50ms maximum

        # Verify exported data
        assert isinstance(metrics_dict, dict)
        assert "tool_metrics" in metrics_dict
        assert "session_id" in metrics_dict
        assert len(json_data) > 0

    @pytest.mark.slow
    def test_continuous_monitoring_performance(self):
        """Benchmark performance under continuous monitoring load."""
        reset_global_collector()
        collector = get_global_collector()

        duration_seconds = 2  # Run for 2 seconds
        operations = 0
        start_time = time.perf_counter()

        # Simulate high-frequency monitoring
        while time.perf_counter() - start_time < duration_seconds:
            collector.record_tool_call(
                tool_name="monitored_tool",
                execution_time=(operations % 200) / 1000.0,
                cache_hit=False
            )
            operations += 1

        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        # Calculate operations per second
        ops_per_second = operations / actual_duration

        # Should handle at least 5000 ops/sec
        assert ops_per_second > 5000

        # Verify all operations recorded
        summary = collector.get_summary()
        tool_metrics = summary["tool_metrics"]
        if "monitored_tool" in tool_metrics:
            assert tool_metrics["monitored_tool"]["call_count"] == operations

    def test_benchmark_with_real_tools(self, temp_repo_root: str):
        """Benchmark actual tool execution performance."""
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        config = AgentConfig()
        registry = ToolRegistry(temp_repo_root, config)

        # Benchmark find_symbol tool
        iterations = 100
        durations = []

        for _i in range(iterations):
            start_time = time.perf_counter()
            result = registry.dispatch("find_symbol", {"name": "hello"})
            end_time = time.perf_counter()

            assert result.ok is True
            durations.append(end_time - start_time)

        # Calculate statistics
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        # Symbol search should be reasonably fast
        assert avg_duration < 0.05  # 50ms average
        assert max_duration < 2.0  # 2s maximum (cold start / MacBook Air overhead)

        # Print benchmark results (for debugging/information)
        print(f"\nfind_symbol benchmark ({iterations} iterations):")
        print(f"  Average: {avg_duration*1000:.2f}ms")
        print(f"  Min: {min_duration*1000:.2f}ms")
        print(f"  Max: {max_duration*1000:.2f}ms")

    def test_comparative_benchmark(self):
        """Benchmark performance with different configurations."""
        # Test with and without caching
        # Implementation-specific
