"""
Performance metrics collection for asicode agent.

Collects execution times, LLM call statistics, tool usage patterns,
cache hit rates, and other performance metrics for profiling and optimization.

Thread-safe cache hit rate tracking with comprehensive metrics collection.
"""
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Any, Optional


class CacheHitRateMetrics:
    """
    Thread-safe cache hit rate metrics tracker.

    Tracks hits and misses for different cache types and provides
    methods to calculate hit rates and retrieve statistics.
    """

    def __init__(self):
        self._file_hits = 0
        self._file_misses = 0
        self._rag_hits = 0
        self._rag_misses = 0
        self._vector_hits = 0
        self._vector_misses = 0
        self._lock = threading.Lock()

    def record_rag_cache(self, hit: bool):
        """Record a RAG cache hit or miss"""
        with self._lock:
            if hit:
                self._rag_hits += 1
            else:
                self._rag_misses += 1

    def record_vector_cache(self, hit: bool):
        """Record a vector cache hit or miss"""
        with self._lock:
            if hit:
                self._vector_hits += 1
            else:
                self._vector_misses += 1

    def get_stats(self, cache_type: str) -> dict[str, Any]:
        """Get comprehensive statistics for a cache type"""
        with self._lock:
            if cache_type == "file":
                hits, misses = self._file_hits, self._file_misses
            elif cache_type == "rag":
                hits, misses = self._rag_hits, self._rag_misses
            elif cache_type == "vector":
                hits, misses = self._vector_hits, self._vector_misses
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")

            total = hits + misses
            hit_rate = hits / total if total > 0 else 0

            return {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hit_rate,
                "hit_rate_percentage": hit_rate * 100
            }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all cache types"""
        return {
            "file": self.get_stats("file"),
            "rag": self.get_stats("rag"),
            "vector": self.get_stats("vector")
        }

    def reset(self):
        """Reset all counters"""
        with self._lock:
            self._file_hits = 0
            self._file_misses = 0
            self._rag_hits = 0
            self._rag_misses = 0
            self._vector_hits = 0
            self._vector_misses = 0


@dataclass
class ToolMetrics:
    """Tool execution metrics.

    Uses O(1) running aggregation (sum/count/min/max) instead of retaining
    every ``execution_time`` sample in a list. A 12h+ autonomous run can issue
    hundreds of thousands of tool calls; the old ``execution_times: list[float]``
    grew unbounded in RAM and made ``avg_execution_time`` / ``get_summary()``
    O(n) on every access (sum() over the whole list). The running counters are
    constant memory and constant time per record and per summary.

    min/max come along for free and surface latency spread in the summary; a
    full distribution (percentiles) would need a bounded reservoir sample — add
    one only if/when the summary needs percentiles.
    """
    name: str
    total_calls: int = 0
    _time_sum: float = 0.0
    _time_min: float = float("inf")
    _time_max: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def record(self, execution_time: float) -> None:
        """Update running aggregates with one sample.

        Caller is expected to hold the collector's guard lock (the same one that
        guards ``tool_metrics`` dict mutation), so this is NOT itself locked.
        """
        self.total_calls += 1
        self._time_sum += execution_time
        if execution_time < self._time_min:
            self._time_min = execution_time
        if execution_time > self._time_max:
            self._time_max = execution_time

    @property
    def avg_execution_time(self) -> float:
        return self._time_sum / self.total_calls if self.total_calls else 0.0

    @property
    def min_execution_time(self) -> float:
        return self._time_min if self.total_calls else 0.0

    @property
    def max_execution_time(self) -> float:
        return self._time_max if self.total_calls else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


@dataclass
class LLMMetrics:
    """LLM call metrics"""
    calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_time_ms: float = 0
    failures: int = 0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.calls if self.calls > 0 else 0


class PerformanceCollector:
    """Performance metrics collector for asicode agent"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Thread-safety guard: design_chat_loop dispatches multiple read tools
        # in parallel (shared_pool) which all call record_tool_call / record_llm_call
        # concurrently. Without this lock, the tool_metrics dict can be mutated
        # while iterated/read by get_summary(), and integer counters race
        # (lost updates). Mirrors the locking pattern of CacheHitRateMetrics.
        self._lock = threading.Lock()

        # Tool metrics
        self.tool_metrics: dict[str, ToolMetrics] = {}

        # LLM metrics
        self.llm_metrics = LLMMetrics()

        # Cache hit rate metrics
        self.cache_metrics = CacheHitRateMetrics()

        # RAG metrics
        self.rag_searches: int = 0
        self.rag_search_time_ms: float = 0

        # Optional live references to registered ToolResultCache(s), wired up via
        # register_tool_result_cache() so their hit/miss/size stats surface in
        # get_summary() — each cache tracks its own stats internally (get_stats(),
        # independently locked); this just points at them. A WeakSet holds ALL
        # caches registered across a session (parent registry + per-clone caches)
        # so get_summary() aggregates across every live cache: a short-lived clone
        # cache no longer masks the parent's hit-rate (last-registered-wins was
        # the pre-WeakSet behavior). weakref auto-drops a cache once the
        # registry/clone holding it is collected, so dead clones don't leak.
        self._tool_result_cache_refs: "weakref.WeakSet[Any]" = weakref.WeakSet()

    def register_tool_result_cache(self, cache: Optional[Any]) -> None:
        """Wire up a ToolResultCache instance so its stats appear in get_summary().

        Each clone (clone_for_subagent / clone_with_filter) gets its own isolated
        cache and registers it here; get_summary() aggregates stats across ALL
        live registered caches. Held by WeakSet so a collected clone's cache
        vanishes automatically — no leak, and a short-lived clone no longer masks
        the parent registry's hit-rate.

        Thread-safe: a clone can be created (and thus register here) on a worker
        thread while get_summary() iterates the WeakSet on another. CPython's
        WeakSet is not documented as thread-safe — its _IterationGuard only
        defers weakref callbacks *within the iterating thread*, so a concurrent
        .add() from another thread can raise "Set changed size during iteration".
        Guard .add() with self._lock (get_summary() snapshots under the same
        lock) to close that window, mirroring the tool_metrics discipline.
        """
        if cache is not None:
            with self._lock:
                self._tool_result_cache_refs.add(cache)

    def start_session(self):
        """Start performance measurement session"""
        self.start_time = time.monotonic()

    def end_session(self):
        """End performance measurement session"""
        self.end_time = time.monotonic()

    def record_tool_call(self, tool_name: str, execution_time: float, cache_hit: bool = False):
        """Record a tool call with execution time.

        Thread-safe: concurrent read tools in design_chat_loop's parallel batch
        all record through here, so the tool_metrics dict mutation and the
        per-ToolMetrics running-counter updates must be guarded by ``self._lock``.

        ``cache_hit`` is the **ToolResultCache** hit flag (set by dispatch on a
        cache HIT; ``result.metadata["cache_hit"]``). It is NOT a file-cache hit.
        Per-tool granularity is recorded in ``tool_metrics[tool].cache_hits /
        cache_misses`` below; the authoritative aggregate lives in the
        ``tool_result_cache`` summary channel (WeakSet over registered caches).

        It is deliberately NOT forwarded to any file-cache channel: doing so
        mislabeled every tool call (including non-cacheable write tools, which
        always counted as file-cache misses) as file-cache activity, duplicating
        the per-tool counters above AND the dedicated ``tool_result_cache``
        channel, and distorting ``overall_hit_rate``. ``record_file_cache`` was
        removed entirely; the ``file`` cache type remains in :meth:`get_stats` /
        :meth:`get_summary` for backward-compat (always zeros — no real file
        cache feeds it).
        """
        with self._lock:
            if tool_name not in self.tool_metrics:
                self.tool_metrics[tool_name] = ToolMetrics(name=tool_name)

            metrics = self.tool_metrics[tool_name]
            metrics.record(execution_time)
            if cache_hit:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1

    def record_rag_cache(self, hit: bool):
        """Record RAG cache hit or miss"""
        self.cache_metrics.record_rag_cache(hit)

    def record_vector_cache(self, hit: bool):
        """Record vector cache hit or miss"""
        self.cache_metrics.record_vector_cache(hit)

    def record_llm_call(self, prompt_tokens: int = 0, completion_tokens: int = 0,
                       execution_time_ms: float = 0, failed: bool = False):
        """Record an LLM call

        Thread-safe: streaming token callbacks and parallel tool threads can
        interleave LLM calls.
        """
        with self._lock:
            self.llm_metrics.calls += 1
            self.llm_metrics.total_prompt_tokens += prompt_tokens
            self.llm_metrics.total_completion_tokens += completion_tokens
            self.llm_metrics.total_time_ms += execution_time_ms

            if failed:
                self.llm_metrics.failures += 1

    def record_rag_search(self, search_time_ms: float):
        """Record a RAG search operation"""
        with self._lock:
            self.rag_searches += 1
            self.rag_search_time_ms += search_time_ms

    def reset_cache_stats(self):
        """Reset cache statistics only"""
        self.cache_metrics.reset()

    def get_cache_stats(self, cache_type: str) -> dict[str, Any]:
        """Get detailed cache statistics for specific cache type"""
        return self.cache_metrics.get_stats(cache_type)

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary"""
        if self.start_time and self.end_time:
            total_execution_time = self.end_time - self.start_time
        elif self.start_time:
            total_execution_time = time.monotonic() - self.start_time
        else:
            total_execution_time = 0

        # Calculate tool metrics summary.
        # Build the WHOLE tool_summary under self._lock (not just snapshot
        # items()) so the per-ToolMetrics scalar reads — running sum/count/
        # min/max, cache hits/misses — are consistent with record_tool_call()'s
        # mutations. The old code read metrics.avg_execution_time /
        # metrics.cache_hit_rate (properties over mutable fields) OUTSIDE the
        # lock, a torn read (statistics-only distortion, no crash). The
        # computation here is pure scalar arithmetic bounded by the number of
        # DISTINCT tools (small), not the number of calls, so holding the lock
        # for the loop is cheap — and it still avoids the heavier per-cache
        # get_stats() calls (those run outside, below).
        with self._lock:
            tool_summary = {}
            for _name, _m in self.tool_metrics.items():
                _calls = _m.total_calls
                _cm_total = _m.cache_hits + _m.cache_misses
                tool_summary[_name] = {
                    'call_count': _calls,
                    'avg_execution_time_ms': (_m._time_sum / _calls * 1000.0) if _calls else 0.0,
                    'min_execution_time_ms': (_m._time_min * 1000.0) if _calls else 0.0,
                    'max_execution_time_ms': (_m._time_max * 1000.0) if _calls else 0.0,
                    'cache_hit_rate': (_m.cache_hits / _cm_total) if _cm_total > 0 else 0.0,
                    'cache_hits': _m.cache_hits,
                    'cache_misses': _m.cache_misses,
                    'total_calls': _calls,
                }
            llm_calls = self.llm_metrics.calls
            llm_prompt = self.llm_metrics.total_prompt_tokens
            llm_completion = self.llm_metrics.total_completion_tokens
            llm_failures = self.llm_metrics.failures
            llm_avg_ms = self.llm_metrics.avg_time_ms
            rag_searches = self.rag_searches
            rag_time_ms = self.rag_search_time_ms

        # Get cache metrics from CacheHitRateMetrics (independently locked)
        cache_stats = self.cache_metrics.get_all_stats()

        # ToolResultCache instances (registered separately; each get_stats() is
        # independently locked). Aggregate across ALL live registered caches —
        # the parent registry's plus any still-live clones — so a short-lived
        # clone cache does not mask the parent's hit-rate. WeakSet drops dead
        # caches automatically. None when no cache is registered (e.g. cache
        # disabled). ``instances`` reports how many caches were aggregated.
        #
        # Snapshot the WeakSet to a list under self._lock so register_tool_result_cache()'s
        # .add() (which takes the same lock) cannot mutate the set mid-iteration and
        # raise "Set changed size during iteration" when a worker thread registers a
        # clone concurrently. list() now holds strong refs, so the per-cache get_stats()
        # calls (each independently locked) run safely WITHOUT holding self._lock —
        # minimizing lock hold time (mirrors the tool_metrics snapshot pattern above).
        with self._lock:
            _registered_caches = list(self._tool_result_cache_refs)
        tool_result_cache_stats = None
        for _cache in _registered_caches:
            try:
                _s = _cache.get_stats()
            except Exception:
                continue
            if not _s:
                continue
            if tool_result_cache_stats is None:
                tool_result_cache_stats = {
                    "hits": 0, "misses": 0, "hit_rate": 0.0,
                    "size": 0, "max_entries": 0, "instances": 0,
                }
            tool_result_cache_stats["hits"] += _s.get("hits", 0)
            tool_result_cache_stats["misses"] += _s.get("misses", 0)
            tool_result_cache_stats["size"] += _s.get("size", 0)
            tool_result_cache_stats["max_entries"] += _s.get("max_entries", 0)
            tool_result_cache_stats["instances"] += 1
        if tool_result_cache_stats is not None:
            _trc_total = (
                tool_result_cache_stats["hits"] + tool_result_cache_stats["misses"]
            )
            tool_result_cache_stats["hit_rate"] = (
                tool_result_cache_stats["hits"] / _trc_total if _trc_total > 0 else 0.0
            )

        # Calculate averages
        avg_rag_search_time = rag_time_ms / rag_searches if rag_searches > 0 else 0

        return {
            'session_id': self.session_id,
            'total_execution_time_seconds': total_execution_time,
            'start_time': self.start_time,
            'end_time': self.end_time,

            'llm_metrics': {
                'calls': llm_calls,
                'total_prompt_tokens': llm_prompt,
                'total_completion_tokens': llm_completion,
                'total_tokens': llm_prompt + llm_completion,
                'avg_time_ms_per_call': llm_avg_ms,
                'failures': llm_failures
            },

            'tool_metrics': tool_summary,

            'cache_metrics': {
                'file_cache': cache_stats['file'],
                'rag_cache': cache_stats['rag'],
                'vector_cache': cache_stats['vector'],
                'tool_result_cache': tool_result_cache_stats,
                'overall_hit_rate': (
                    (cache_stats['file']['hits'] + cache_stats['rag']['hits'] + cache_stats['vector']['hits']) /
                    (cache_stats['file']['total'] + cache_stats['rag']['total'] + cache_stats['vector']['total'])
                                        if (cache_stats['file']['total'] + cache_stats['rag']['total'] + cache_stats['vector']['total']) > 0 else 0
                )
            },

            'rag_metrics': {
                'searches': rag_searches,
                'total_search_time_ms': rag_time_ms,
                'avg_search_time_ms': avg_rag_search_time
            }
        }


# Global collector for easy access
_global_collector: Optional[PerformanceCollector] = None
_global_collector_lock = threading.Lock()


def get_global_collector() -> PerformanceCollector:
    """Get or create global performance collector (thread-safe DCL)"""
    global _global_collector
    if _global_collector is None:
        with _global_collector_lock:
            if _global_collector is None:
                _global_collector = PerformanceCollector()
    return _global_collector


def reset_global_collector(session_id: Optional[str] = None) -> PerformanceCollector:
    """Reset global performance collector.

    Takes ``_global_collector_lock`` so a concurrent ``get_global_collector()``
    (DCL under the same lock) never observes a half-published replacement, and
    two concurrent resets don't race (previously the replacement was unguarded,
    asymmetric with the getter). Returns the new collector.
    """
    global _global_collector
    with _global_collector_lock:
        _global_collector = PerformanceCollector(session_id)
    return _global_collector
