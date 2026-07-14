"""
Async Tool Executor for asicode Agent

Parallel tool execution with dependency-aware scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .tool_registry import ToolResult

logger = logging.getLogger(__name__)


class AsyncToolExecutor:
    """Manages parallel execution of independent tools."""

    def __init__(self, tool_registry, max_workers: int = 5):
        self.registry = tool_registry
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.dependency_graph = tool_registry.dependency_graph if hasattr(tool_registry, 'dependency_graph') else None

    async def execute_parallel(self, tool_calls: list[dict]) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel, respecting dependencies.

        Args:
            tool_calls: List of dicts with 'tool' and 'args' keys.

        Returns:
            List of ToolResult objects in the same order as input.
        """
        if not tool_calls:
            return []

        # Group tools by dependency layers
        execution_groups = self._group_by_dependencies(tool_calls)

        results: list[Optional[ToolResult]] = [None] * len(tool_calls)

        # Execute groups sequentially, tools within groups in parallel
        for group in execution_groups:
            # Create tasks for all tools in this group
            tasks = []
            group_indices = []
            for idx in group:
                tool_call = tool_calls[idx]
                tasks.append(self._execute_single(tool_call))
                group_indices.append(idx)

            # Run all tasks in parallel
            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results
            for idx, result in zip(group_indices, group_results, strict=False):
                if isinstance(result, BaseException):
                    logger.error(f"Tool execution failed: {result}")
                    results[idx] = ToolResult(
                        ok=False,
                        content=f"Parallel execution error: {result}",
                        error=str(result),
                        execution_time=0.0
                    )
                else:
                    results[idx] = result

        # Guard against unexpected None slots (should not happen, but
        # prevents silent length mismatch if a dependency grouping misses
        # an index).
        for i, r in enumerate(results):
            if r is None:
                logger.error("Tool at index %d produced no result — filling with error ToolResult", i)
                results[i] = ToolResult(
                    ok=False,
                    content="no result",
                    error="execution index not covered by grouping",
                    execution_time=0.0,
                )
        return results

    def _group_by_dependencies(self, tool_calls: list[dict]) -> list[list[int]]:
        """
        Group tool calls by dependency layers using topological sorting.

        Returns:
            List of groups where each group contains indices of tools that can run in parallel.
        """
        if not self.dependency_graph:
            # No dependency graph, all tools can run in parallel
            return [list(range(len(tool_calls)))]

        # Build adjacency matrix for this specific set of tools
        n = len(tool_calls)
        tool_names = [tc['tool'] for tc in tool_calls]

        # Check dependencies between tools
        depends = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._has_dependency(tool_names[i], tool_names[j]):
                    depends[i][j] = True

        # Topological sorting to find layers
        in_degree = [0] * n
        for i in range(n):
            for j in range(n):
                if depends[i][j]:  # i depends on j → j is a prerequisite for i
                    in_degree[i] += 1

        groups = []
        remaining = set(range(n))

        while remaining:
            # Find nodes with no incoming dependencies from remaining nodes
            layer = []
            for i in remaining:
                if in_degree[i] == 0:
                    layer.append(i)

            if not layer:
                # Cyclic dependency, break by putting all remaining in one layer
                layer = list(remaining)

            groups.append(layer)
            for i in layer:
                remaining.remove(i)
                # Decrease in-degree of nodes that depend on i
                for j in range(n):
                    if depends[j][i]:  # j depends on i → i was a prereq for j
                        in_degree[j] -= 1

        return groups

    def _has_dependency(self, tool1: str, tool2: str) -> bool:
        """Check if tool1 depends on tool2."""
        if not self.dependency_graph:
            return False
        # Use dependency graph to check if there's a path from tool2 to tool1
        return self.dependency_graph.has_path(tool2, tool1)

    async def _execute_single(self, tool_call: dict) -> ToolResult:
        """Execute a single tool call in thread pool."""
        tool_name = tool_call['tool']
        args = tool_call['args']

        # We are inside an async function, so a loop is always running.
        # get_running_loop() is the correct API; get_event_loop() is deprecated
        # since Python 3.10 and raises RuntimeError in 3.12+ under a running loop.
        loop = asyncio.get_running_loop()
        try:
            start_time = time.monotonic()
            # Run sync dispatch in thread pool
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.registry.dispatch(tool_name, args)
            )
            execution_time = time.monotonic() - start_time
            try:
                result.execution_time = execution_time
            except AttributeError:
                pass
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                ok=False,
                content=f"Parallel execution error: {e}",
                error=str(e),
                execution_time=0.0
            )

    def shutdown(self):
        """Shutdown thread pool executor."""
        self.executor.shutdown(wait=False)
