"""
Performance metrics collection for asicode agent.

Collects execution times, LLM call statistics, tool usage patterns,
cache hit rates, and other performance metrics for profiling and optimization.

Thread-safe cache hit rate tracking with comprehensive metrics collection.
"""
import bisect
import logging
import threading
import time
import uuid
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from .config.thresholds import config as _threshold_config

logger = logging.getLogger(__name__)


def _percentiles(samples: "deque", pcts: tuple[float, ...]) -> tuple[float, ...]:
    """Linear-interpolated percentiles from a SINGLE sort of ``samples``.

    Pure: snapshots the deque and sorts a copy (does not mutate it). Returns one
    value per requested percentile, in input order.

    NOTE (delta cache): ``get_summary()`` no longer goes through this function —
    per-entity reads come from ``ToolMetrics.percentiles`` /
    ``LLMMetrics.percentiles``, which compute from the incrementally-maintained
    ``_latency_sorted`` mirror (O(1) per percentile, no sort). This sort-based
    path remains the REFERENCE implementation: it backs the ``percentile``
    single-value API and the unit tests, and the delta cache is parity-tested
    against it.
    """
    return _percentiles_from_sorted(sorted(samples), pcts)


def _percentiles_from_sorted(s: list, pcts: tuple[float, ...]) -> tuple[float, ...]:
    """Linear-interpolated percentiles over an ALREADY-SORTED sample list.

    Shared interpolation math for BOTH percentile paths — ``_percentiles``
    (sort + delegate) and the delta-cache reads (``ToolMetrics.percentiles`` /
    ``LLMMetrics.percentiles`` on the ``_latency_sorted`` mirror) — so the
    cached read and the reference sort path cannot drift.

    Linear interpolation between the two closest ranks (matches numpy's default
    'linear' method); for a single sample every pct returns that sample. 0.0 for
    every requested pct when the window is empty so the summary reads "no data"
    honestly rather than raising.
    """
    n = len(s)
    if n == 0:
        return tuple(0.0 for _ in pcts)
    if n == 1:
        _only = float(s[0])
        return tuple(_only for _ in pcts)
    out = []
    for pct in pcts:
        k = (n - 1) * (pct / 100.0)
        lo = int(k)
        hi = lo + 1 if lo + 1 < n else lo
        frac = k - lo
        out.append(float(s[lo] + (s[hi] - s[lo]) * frac))
    return tuple(out)


def _append_latency(samples: "deque", sorted_samples: list, value: float) -> None:
    """Append one SUCCESS latency sample to the bounded sliding window, keeping the
    sorted mirror (``sorted_samples``) in lockstep with the FIFO deque.

    This is the delta-cache update: O(K) worst case (list insert/pop shift,
    K ≤ LATENCY_SAMPLE_WINDOW) at RECORD time, so ``get_summary()`` percentile
    reads are O(1) per entity instead of O(K log K) per poll. ``get_summary()``
    is polled every ~2s by the SSE broadcaster across every tool AND provider;
    the sort-per-read cost is what pushed test_metrics_export_performance over
    its 50ms budget under full-suite CPU contention (the regression this
    replaces).

    The deque is the FIFO authority (eviction order); the sorted list mirrors
    its MULTISET (duplicates retained). Eviction: when the deque is at maxlen
    the leftmost (oldest) value is about to be dropped — remove one occurrence
    of it from the mirror via ``bisect_left`` (any equal occurrence is
    multiset-equivalent, so the percentile result is unchanged). Caller holds
    the collector's guard lock (same contract as ``record`` / ``percentile``).
    """
    _cap = samples.maxlen
    _evicted = samples[0] if _cap and len(samples) == _cap else None
    samples.append(value)
    bisect.insort(sorted_samples, value)
    if _evicted is not None:
        sorted_samples.pop(bisect.bisect_left(sorted_samples, _evicted))


class CacheHitRateMetrics:
    """
    Thread-safe cache hit rate metrics tracker.

    Tracks hits and misses for different cache types and provides
    methods to calculate hit rates and retrieve statistics.
    """

    def __init__(self):
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
            if cache_type == "rag":
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
            "rag": self.get_stats("rag"),
            "vector": self.get_stats("vector")
        }

    def reset(self):
        """Reset all counters"""
        with self._lock:
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
    failures: int = 0
    # Bounded ring buffer of recent call outcomes (1=failure, 0=success), capped at
    # TOOL_FAILURE_RATE_WINDOW (config). Unlike the cumulative ``failures`` counter
    # (denominator = total_calls, diluted toward 0 over a long run), this gives a
    # LIVE health read: recent_failure_rate = sum / len over the last N calls. A
    # previously-trusted tool that just started failing trips the warn gate within
    # ~N calls instead of never (1000 successes + 20 failures reads ~2% cumulative
    # but ~67% over a 30-window). Constant memory per tool regardless of run length.
    _recent_outcomes: deque = field(
        default_factory=lambda: deque(maxlen=_threshold_config.scores.TOOL_FAILURE_RATE_WINDOW)
    )
    # Bounded RECENT-latency window (deque maxlen = LATENCY_SAMPLE_WINDOW). The
    # O(1) avg/min/max running aggregates above are constant memory but only a
    # single point estimate; a degrading tool shows up FIRST in the tail (p95)
    # while the avg stays flat under a mass of fast cache hits — the same
    # history-dilution blind spot that motivated ``_recent_outcomes``. A bounded
    # SLIDING WINDOW (not a uniform reservoir) is used so the tail is not diluted
    # back toward early fast calls on a 12h+ run: recent p95 tracks CURRENT
    # degradation. percentile() below reads it. SUCCESS-ONLY:
    # record(failed=True) skips the append — a failed call's wall time is not a
    # latency sample (fast-fail drags p95 down, slow-fail spikes it; failures
    # live on the recent_failure_rate axis). Constant memory per tool.
    _latency_samples: deque = field(
        default_factory=lambda: deque(maxlen=_threshold_config.scores.LATENCY_SAMPLE_WINDOW)
    )
    # Sorted mirror of ``_latency_samples`` (same multiset, kept in lockstep by
    # ``_append_latency`` at RECORD time) — the p50/p95 DELTA CACHE. Percentile
    # reads (``percentile`` / ``percentiles``, i.e. every get_summary() poll)
    # are O(1) index math on this list instead of an O(K log K) sort per read.
    # Constant memory: length == len(_latency_samples) ≤ LATENCY_SAMPLE_WINDOW.
    _latency_sorted: list = field(default_factory=list)

    @property
    def latency_samples_count(self) -> int:
        """How many SUCCESSFUL calls contributed to the current latency window.
        Used as ``p95_n`` in the summary for the min-samples floor in
        ``top_slow_tools`` — a tool with fewer than ``LATENCY_P95_MIN_SAMPLES``
        samples is not yet meaningful for p95-based health gating.
        """
        return len(self._latency_samples)

    # NOTE on the ``failures`` counter: ``record_tool_call`` is invoked at two
    # sites (ToolRegistry.execute_tool dispatch + the agent turn pipeline), both
    # of which previously ignored ``result.ok`` — a failed tool call (e.g. a
    # rolled-back write, a read of a missing file) was counted identically to a
    # success. ``LLMMetrics`` has carried ``failures`` since inception; the tool
    # side was an asymmetry that hid the single most important health signal for
    # an autonomous agent (which tool fails how often). ``failure_rate`` below
    # and the per-tool ``failures``/``failure_rate`` keys in get_summary() close
    # that gap. A failure is ``not result.ok`` at the record site.

    def record(self, execution_time: float, failed: bool = False) -> None:
        """Update running aggregates with one sample.

        Caller is expected to hold the collector's guard lock (the same one that
        guards ``tool_metrics`` dict mutation), so this is NOT itself locked.

        The CUMULATIVE aggregates (total_calls / _time_sum / _time_min /
        _time_max) update for EVERY call regardless of ``failed`` — ``_time_max``
        as "slowest wall time ever observed" is a genuine worst-case read even
        when that call failed (e.g. a 30s timeout). The RECENT-latency window
        (``_latency_samples``, read by ``percentile``) feeds the p50/p95 SPEED
        signal, so it is SUCCESS-ONLY: a failed call's wall time is not a
        completion-latency sample. A fast-fail (429/auth) would drag p95 DOWN
        and hide degradation; a slow-fail (timeout) would spike p95 UP and
        conflate "slow" with "failing". The failure DIMENSION is already
        captured by ``recent_failure_rate`` / ``failures``; the latency
        dimension stays a clean completion-latency read. Symmetric with
        ``record_llm_call``'s ``failed`` gating of its own latency window.
        """
        self.total_calls += 1
        self._time_sum += execution_time
        self._time_min = min(self._time_min, execution_time)
        self._time_max = max(self._time_max, execution_time)
        if not failed:
            _append_latency(self._latency_samples, self._latency_sorted, execution_time)

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
    def cache_hit_rate(self) -> Optional[float]:
        # Mirrors the SHIPPED summary semantics (get_summary emits None for a zero
        # denominator): None = "not applicable" (write/serial tool, or cache never
        # probed), NOT a fake 0%. A real 0% (1 miss, 0 hits) is still 0.0. Returning
        # None keeps this accessor honest — a future reader cannot mistake "never
        # probed" for "always misses", the exact contamination the 3-state
        # cache_hit recording removed at a different entry point.
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else None

    @property
    def failure_rate(self) -> float:
        # CUMULATIVE failure rate (failures / total_calls) — kept for display/context
        # ("how bad overall"). It does NOT gate warn_failing_tools / the dashboard
        # card: a 12h+ run dilutes it so badly that a currently-broken tool reads
        # healthy. The GATE is recent_failure_rate below. Both are shipped in the
        # per-tool summary so the UI can show cumulative rate while acting on recent.
        return self.failures / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def recent_calls(self) -> int:
        # Samples currently in the window = min(total_calls, N) once warm.
        return len(self._recent_outcomes)

    @property
    def recent_failures(self) -> int:
        return sum(self._recent_outcomes)

    @property
    def recent_failure_rate(self) -> float:
        # THE live-health gate. Not diluted by historical success: a sustained
        # failure burst climbs to ~1.0 within N calls regardless of how many
        # successes preceded it. 0.0 when the window is empty (no calls yet).
        n = len(self._recent_outcomes)
        return sum(self._recent_outcomes) / n if n > 0 else 0.0

    def percentile(self, pct: float) -> float:
        """Linear-interpolated percentile over the RECENT-latency window.

        50 → median, 95 → p95 tail. Reads the ``_latency_sorted`` mirror (a
        bounded sliding window, NOT a uniform reservoir — see the field comment)
        so the tail reflects CURRENT behavior, not a lifetime estimate diluted
        toward early fast calls on a long run. 0.0 when no samples yet. O(1) —
        the sort work moved to RECORD time (``_append_latency`` delta cache),
        so every get_summary() poll is cheap. Caller is expected to hold the
        collector's guard lock (same contract as ``record()`` /
        ``recent_failure_rate``) since this reads shared state without its own
        lock.
        """
        return _percentiles_from_sorted(self._latency_sorted, (pct,))[0]

    def percentiles(self, pcts: tuple[float, ...]) -> tuple[float, ...]:
        """Multi-percentile read over the RECENT-latency window.

        Same semantics as ``percentile`` (raw stored units — tool latency is
        stored in SECONDS; ``get_summary`` converts to ms at emit, matching the
        avg/min/max keys). Computes every requested pct from the
        ``_latency_sorted`` delta-cache mirror — O(len(pcts)) with O(1) per pct,
        no sort. Used by ``get_summary()`` to read p50+p95 together; same lock
        contract as ``percentile`` (caller holds the collector's guard lock).
        """
        return _percentiles_from_sorted(self._latency_sorted, pcts)


@dataclass
class LLMMetrics:
    """LLM call metrics for ONE provider.

    A ``PerformanceCollector`` holds ``llm_metrics: dict[str, LLMMetrics]`` keyed by
    provider name (``LLMClient.get_provider_name()``, lower-cased) — symmetric with
    ``tool_metrics: dict[str, ToolMetrics]``. The previous SINGLE aggregate stream
    mixed every provider's outcomes into one bounded deque: a 100%-failing fallback
    provider was diluted against a healthy primary's traffic within the maxlen window
    and read a low recent_failure_rate — never tripping the warn gate (the exact
    blind spot per-tool isolation already closed for tools). Per-provider isolation
    keeps each provider's recent window independent. Constant memory per provider
    regardless of run length.
    """
    provider: str = ""
    calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_time_ms: float = 0
    failures: int = 0
    # Symmetric with ToolMetrics._recent_outcomes: a bounded ring buffer of recent
    # call outcomes (1=failure, 0=success). The cumulative ``failures``/``calls`` rate
    # is diluted toward 0 over a long autonomous run, so a provider that just started
    # rate-limiting / 5xx-ing reads healthy — exactly the blind spot that motivated
    # the tool-side window. recent_failure_rate tracks CURRENT health and is the gate
    # for warn_failing_llm / the summary's failing_llm entry. Reuses the SAME window
    # config (TOOL_FAILURE_RATE_WINDOW); a separate LLM knob is easy to add if tuning
    # ever needs to diverge. Constant memory regardless of run length.
    _recent_outcomes: deque = field(
        default_factory=lambda: deque(maxlen=_threshold_config.scores.TOOL_FAILURE_RATE_WINDOW)
    )
    # Bounded RECENT-latency window — symmetric with ToolMetrics._latency_samples
    # (same rationale: avg_time_ms is diluted flat under fast calls while a
    # degrading provider surfaces FIRST in the p95 tail). Sliding window, not a
    # uniform reservoir, so the tail tracks CURRENT latency. percentile() reads
    # it. SUCCESS-ONLY: record_llm_call(failed=True) skips the append (mirrors
    # ToolMetrics — failed wall time is not a completion-latency sample).
    # Constant memory per provider regardless of run length.
    _latency_samples: deque = field(
        default_factory=lambda: deque(maxlen=_threshold_config.scores.LATENCY_SAMPLE_WINDOW)
    )
    # Sorted mirror of ``_latency_samples`` — symmetric with ToolMetrics's
    # p50/p95 delta cache (``_append_latency`` keeps it in lockstep at record
    # time; percentile reads are O(1) per provider per get_summary() poll).
    _latency_sorted: list = field(default_factory=list)

    @property
    def latency_samples_count(self) -> int:
        """How many SUCCESSFUL calls contributed to the current latency window.
        Symmetric with ``ToolMetrics.latency_samples_count`` — used as ``p95_n``
        in the summary for the min-samples floor in ``top_slow_llm``.
        """
        return len(self._latency_samples)

    @property
    def failure_rate(self) -> float:
        # Cumulative (failures / calls) — display/context only, mirrors ToolMetrics.
        return self.failures / self.calls if self.calls > 0 else 0.0

    @property
    def recent_calls(self) -> int:
        return len(self._recent_outcomes)

    @property
    def recent_failures(self) -> int:
        return sum(self._recent_outcomes)

    @property
    def recent_failure_rate(self) -> float:
        # THE live-health gate (mirrors ToolMetrics.recent_failure_rate). Not diluted
        # by historical success: a sustained provider failure climbs toward 1.0 within
        # N calls regardless of how many successes preceded it. 0.0 when empty.
        n = len(self._recent_outcomes)
        return sum(self._recent_outcomes) / n if n > 0 else 0.0

    def percentile(self, pct: float) -> float:
        """Linear-interpolated percentile over the RECENT-latency window.

        Mirrors ``ToolMetrics.percentile``: 50 → median, 95 → p95 tail; reads the
        ``_latency_sorted`` delta-cache mirror so the tail reflects CURRENT
        provider latency, not a lifetime estimate diluted toward early fast
        calls. 0.0 when no calls yet. O(1). Caller holds the collector's guard
        lock (same contract as ``record_llm_call``).
        """
        return _percentiles_from_sorted(self._latency_sorted, (pct,))[0]

    def percentiles(self, pcts: tuple[float, ...]) -> tuple[float, ...]:
        """Multi-percentile read over the RECENT-latency window.

        Same semantics as ``percentile`` (raw stored units — LLM latency is
        already in MILLISECONDS, so ``get_summary`` emits these as-is for the
        p50_ms/p95_ms keys). Computes every requested pct from the
        ``_latency_sorted`` delta-cache mirror — O(len(pcts)) with O(1) per pct,
        no sort. Used by ``get_summary()`` to read p50+p95 together; same lock
        contract as ``percentile`` (caller holds the collector's guard lock).
        """
        return _percentiles_from_sorted(self._latency_sorted, pcts)


@dataclass
class AgentResultMetrics:
    """Turn-outcome metrics for agent runs.

    Mirrors ``ToolMetrics``/``LLMMetrics``: cumulative counters plus a bounded
    recent-outcomes window (``TOOL_FAILURE_RATE_WINDOW``), so both the
    cumulative and the live-health reads come from one source. Constant memory.

    This is the ONLY channel that can surface **truncation storms**: when a
    turn's LLM response exhausts its max_tokens budget after retries
    (``finish_reason`` in ``("length", "truncated")`` 3x — agent_loop clears
    the tool calls and continues with the partial text), the LLM call itself
    "succeeded", so ``record_llm_call`` never sees ``failed=True`` and
    ``llm_metrics`` reads healthy. ``truncated_turns`` counts those events so
    a run being silently cut off shows up in the metrics instead of reading
    as normal traffic.
    """
    turns: int = 0
    failed_turns: int = 0
    truncated_turns: int = 0
    total_time_ms: float = 0.0
    # Bounded recent-outcome window, symmetric with ToolMetrics._recent_outcomes
    # — same live-health rationale (the cumulative failed_turns/turns rate is
    # diluted over a long run). 3-value encoding: a truncated turn must NOT read
    # as success in the live window (that bug made truncation storms invisible
    # to recent_failure_rate while cumulative truncation_rate climbed).
    _OUTCOME_SUCCESS = 0
    _OUTCOME_FAILED = 1
    _OUTCOME_TRUNCATED = 2
    _recent_outcomes: deque = field(
        default_factory=lambda: deque(maxlen=_threshold_config.scores.TOOL_FAILURE_RATE_WINDOW)
    )

    @property
    def failure_rate(self) -> float:
        return self.failed_turns / self.turns if self.turns > 0 else 0.0

    @property
    def truncation_rate(self) -> float:
        return self.truncated_turns / self.turns if self.turns > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.turns if self.turns > 0 else 0.0

    @property
    def recent_turns(self) -> int:
        return len(self._recent_outcomes)

    @property
    def recent_failed_turns(self) -> int:
        return sum(1 for o in self._recent_outcomes if o == self._OUTCOME_FAILED)

    @property
    def recent_failure_rate(self) -> float:
        n = len(self._recent_outcomes)
        return self.recent_failed_turns / n if n > 0 else 0.0

    @property
    def recent_truncated_turns(self) -> int:
        return sum(1 for o in self._recent_outcomes if o == self._OUTCOME_TRUNCATED)

    @property
    def recent_truncation_rate(self) -> float:
        n = len(self._recent_outcomes)
        return self.recent_truncated_turns / n if n > 0 else 0.0


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

        # LLM metrics — PER-PROVIDER dict, symmetric with tool_metrics. Keyed by
        # provider name (lower-cased) so a failing fallback provider is not diluted
        # by a healthy primary's traffic within one shared deque. record_llm_call()
        # buckets on this key; get_summary() builds both a per-provider breakdown
        # (the gate input) and an aggregate headline (backward-compat display).
        self.llm_metrics: dict[str, LLMMetrics] = {}

        # Cache hit rate metrics
        self.cache_metrics = CacheHitRateMetrics()

        # RAG metrics
        self.rag_searches: int = 0
        self.rag_search_time_ms: float = 0

        # Turn-level agent result metrics — the ONLY channel that can surface
        # truncation storms (see record_agent_result / AgentResultMetrics).
        self.agent_result_metrics = AgentResultMetrics()

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

        Each clone (clone_for_subagent) gets its own isolated
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

    def record_tool_call(
        self,
        tool_name: str,
        execution_time: float,
        cache_hit: Optional[bool] = None,
        failed: bool = False,
    ):
        """Record a tool call with execution time.

        Thread-safe: concurrent read tools in design_chat_loop's parallel batch
        all record through here, so the tool_metrics dict mutation and the
        per-ToolMetrics running-counter updates must be guarded by ``self._lock``.

        ``failed`` is ``not result.ok`` at the call site — a rolled-back write,
        a missing-file read, an unknown tool, etc. It feeds the per-tool
        ``failures`` counter (and ``failure_rate`` in :meth:`get_summary`),
        closing the asymmetry with :meth:`record_llm_call`'s ``failed`` param /
        ``LLMMetrics.failures``. Defaults to ``False`` for backward compat.

        ``cache_hit`` is a **3-state** signal (the ToolResultCache outcome):

        * ``True``  — cacheable tool, cache HIT (``result.metadata["cache_hit"]``).
        * ``False`` — cacheable tool, cache MISS (probed the cache, no entry).
        * ``None``  — **not cacheable** (write / serial tools are never probed,
          see ``ToolRegistry._READ_ONLY_TOOLS``). Recording these as misses would
          structurally pin the per-tool ``cache_hit_rate`` at 0% (denominator =
          ``total_calls``), making every write tool look like "cache always
          misses" when in fact caching does not apply. ``None`` is the default so
          any future caller that forgets to classify cacheability cannot
          reintroduce the contamination — only an explicit ``True``/``False``
          counts toward ``cache_hits``/``cache_misses``.

        Per-tool granularity is recorded in ``tool_metrics[tool].cache_hits /
        cache_misses`` below; the authoritative aggregate lives in the
        ``tool_result_cache`` summary channel (WeakSet over registered caches).

        It is deliberately NOT forwarded to the ``rag``/``vector`` cache
        channels: doing so would mislabel every tool call (including
        non-cacheable write tools) as rag/vector activity, duplicating the
        per-tool counters above AND the dedicated ``tool_result_cache``
        channel, and distorting ``overall_hit_rate``.
        """
        with self._lock:
            if tool_name not in self.tool_metrics:
                self.tool_metrics[tool_name] = ToolMetrics(name=tool_name)

            metrics = self.tool_metrics[tool_name]
            metrics.record(execution_time, failed=failed)
            # 3-state: None (non-cacheable) contributes NEITHER hit nor miss, so
            # write/serial tools keep cache_hits == cache_misses == 0 and their
            # per-tool cache_hit_rate reads 0.0 with a zero denominator (honestly
            # "not applicable") instead of a structurally-faked 0%.
            if cache_hit is not None:
                if cache_hit:
                    metrics.cache_hits += 1
                else:
                    metrics.cache_misses += 1
            if failed:
                # Mirrors record_llm_call()'s ``if failed: failures += 1``.
                metrics.failures += 1
            # Push the outcome (1=failure, 0=success) into the bounded recent
            # window. This is the LIVE-health feeder: the cumulative ``failures``
            # counter above is diluted by history, but the windowed deque lets
            # recent_failure_rate reflect CURRENT health (the warn gate). deque
            # maxlen caps memory; oldest sample is dropped once the window is full.
            metrics._recent_outcomes.append(1 if failed else 0)

    def record_rag_cache(self, hit: bool):
        """Record RAG cache hit or miss"""
        self.cache_metrics.record_rag_cache(hit)

    def record_vector_cache(self, hit: bool):
        """Record vector cache hit or miss"""
        self.cache_metrics.record_vector_cache(hit)

    def record_llm_call(self, provider: str = "", prompt_tokens: int = 0, completion_tokens: int = 0,
                       execution_time_ms: float = 0, failed: bool = False):
        """Record an LLM call.

        ``provider`` is the LLM client's provider name (``get_provider_name()``);
        calls are bucketed PER-PROVIDER so a failing fallback provider is not diluted
        by a healthy primary's traffic within one shared deque (the bug a single
        aggregate stream had) — symmetric with ``record_tool_call(tool_name=...)``
        bucketing by tool name. Normalized to lower-case; empty/unknown falls back to
        ``"unknown"``.

        Thread-safe: streaming token callbacks and parallel tool threads can
        interleave LLM calls.
        """
        _prov = (provider or "").strip().lower() or "unknown"
        with self._lock:
            m = self.llm_metrics.get(_prov)
            if m is None:
                m = LLMMetrics(provider=_prov)
                self.llm_metrics[_prov] = m
            m.calls += 1
            m.total_prompt_tokens += prompt_tokens
            m.total_completion_tokens += completion_tokens
            m.total_time_ms += execution_time_ms
            m._recent_outcomes.append(1 if failed else 0)
            # Latency window is SUCCESS-ONLY (mirrors ToolMetrics.record's
            # ``failed`` gating): a failed call's wall time is not a
            # completion-latency sample — fast-fail hides degradation, slow-fail
            # conflates "slow" with "failing". The failure dimension is tracked
            # by _recent_outcomes / failures above. total_time_ms (cumulative,
            # feeds avg_time_ms_per_call) still records every call's time so a
            # fully-failing provider keeps avg_time_ms_per_call honest.
            if not failed:
                _append_latency(m._latency_samples, m._latency_sorted, execution_time_ms)

            if failed:
                m.failures += 1

    def record_agent_result(self, failed: bool = False, execution_time_ms: float = 0.0,
                            truncated: bool = False) -> None:
        """Record the outcome of an agent turn.

        The turn-level result channel. Unlike :meth:`record_llm_call` /
        :meth:`record_tool_call` (which see every call, successful or not),
        this is an OUTCOME-EVENT channel: production wires it at the points
        where a turn ends abnormally or is visibly degraded —

        * ``truncated=True`` — the turn's LLM response exhausted its retry
          budget (``finish_reason`` ``length``/``truncated`` 3x; agent_loop
          cleared the tool calls and continued with partial text). The LLM
          call itself succeeded, so no other channel ever sees this — without
          this flag a truncation storm reads as perfectly healthy traffic.
        * ``failed=True`` — the turn terminated with an LLM error
          (connection / rate-limit / auth / server-unavailable / quota).

        ``turns`` counts recorded outcome events (successful turns are not
        recorded by default), so ``failure_rate`` / ``truncation_rate`` read
        as "share of recorded outcomes" — the recent window mirrors the
        tool/LLM live-health pattern, with a 3-value encoding so truncated
        turns are counted as degraded (not success) in the live window.

        Thread-safe (guarded by ``self._lock``, same contract as
        :meth:`record_tool_call`).
        """
        with self._lock:
            _arm = self.agent_result_metrics
            _arm.turns += 1
            _arm.total_time_ms += execution_time_ms
            if failed:
                _arm.failed_turns += 1
            if truncated:
                _arm.truncated_turns += 1
            # 3-value window encoding — truncated takes precedence over failed
            # (production wiring sites are mutually exclusive: truncation fires
            # only when the LLM call itself succeeded).
            _outcome = (
                AgentResultMetrics._OUTCOME_TRUNCATED if truncated
                else AgentResultMetrics._OUTCOME_FAILED if failed
                else AgentResultMetrics._OUTCOME_SUCCESS
            )
            _arm._recent_outcomes.append(_outcome)

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
        # computation here is O(1) per entity — p50/p95 come from the per-tool /
        # per-provider ``_latency_sorted`` delta-cache mirrors (updated at RECORD
        # time by ``_append_latency``; max LATENCY_SAMPLE_WINDOW=128 elements),
        # NOT from a sort per poll. Still bounded by the number of DISTINCT
        # entities (small), and avoids a lock-release storm under concurrent
        # tool calls — holding the lock for the loop is cheaper than per-cache
        # get_stats() calls (those run outside, below).
        with self._lock:
            tool_summary = {}
            for _name, _m in self.tool_metrics.items():
                _calls = _m.total_calls
                _cm_total = _m.cache_hits + _m.cache_misses
                # O(1) delta-cache read (seconds internally, → converted to ms
                # at emit below, matching avg/min/max keys).
                _t_p50, _t_p95 = _m.percentiles((50, 95))
                tool_summary[_name] = {
                    'call_count': _calls,
                    'avg_execution_time_ms': (_m._time_sum / _calls * 1000.0) if _calls else 0.0,
                    'min_execution_time_ms': (_m._time_min * 1000.0) if _calls else 0.0,
                    'max_execution_time_ms': (_m._time_max * 1000.0) if _calls else 0.0,
                    # None (not 0.0) when a tool was never probed for cache
                    # (write/serial tools, or cache disabled) — 0.0 would read as
                    # "0% hit rate" and be mistaken for "always misses". None
                    # signals N/A honestly. No card reads this (per-tool detail is
                    # raw-JSON only); future UI should render null as "—"/"N/A".
                    'cache_hit_rate': (_m.cache_hits / _cm_total) if _cm_total > 0 else None,
                    'cache_hits': _m.cache_hits,
                    'cache_misses': _m.cache_misses,
                    'failures': _m.failures,
                    # Cumulative rate — DISPLAY only ("how bad overall"). Diluted
                    # by history on a long run; NOT the warn gate.
                    'failure_rate': (_m.failures / _calls) if _calls > 0 else 0.0,
                    # THE live-health gate. recent_calls = samples in the window
                    # (= min(total_calls, N)); recent_failure_rate = recent_failures
                    # / recent_calls. A currently-broken tool trips the warn gate
                    # off these, regardless of how many successes preceded it.
                    'recent_calls': _m.recent_calls,
                    'recent_failures': _m.recent_failures,
                    'recent_failure_rate': _m.recent_failure_rate,
                    # Latency distribution over the RECENT window — p50 (median)
                    # and p95 (tail). The avg_execution_time above is diluted flat
                    # by fast cache hits; a degrading tool surfaces FIRST in p95.
                    # Bounded sliding window → reflects CURRENT latency. Stored in
                    # SECONDS internally (matching _time_sum/_time_min/_time_max, all
                    # fed by result.execution_time = time.monotonic() - start_time);
                    # converted to ms here so p50_ms/p95_ms share the unit of
                    # avg/min/max_execution_time_ms. SUCCESS-only window (failed
                    # calls excluded): a fully-failing tool has no samples →
                    # p50/p95 0.0 alongside a high failure_rate, reading "dead,
                    # not slow".
                    'p50_ms': _t_p50 * 1000.0,
                    'p95_ms': _t_p95 * 1000.0,
                    # How many successful calls contributed to the latency window
                    # (the denominator for p95). Used by top_slow_tools() for its
                    # min-samples floor — a single slow call does not trip the gate.
                    'p95_n': _m.latency_samples_count,
                    'total_calls': _calls,
                }
            # LLM metrics are PER-PROVIDER (dict[str, LLMMetrics], symmetric with
            # tool_metrics). Build the per-provider breakdown (the GATE input for
            # top_failing_llm) AND the AGGREGATE scalars for the headline llm_metrics
            # dict (backward-compat display). The aggregate recent_failure_rate is the
            # diluted cross-provider number (honest as an overall read); the per-
            # provider recent_failure_rate is what the warn gate reads — a failing
            # fallback provider is no longer masked by a healthy primary's traffic
            # within one shared deque. All reads under self._lock for consistency with
            # record_llm_call()'s per-provider dict + deque mutations.
            llm_provider_summary: dict[str, dict[str, Any]] = {}
            _agg_calls = _agg_pt = _agg_ct = 0
            _agg_time = 0.0
            _agg_failures = 0
            _agg_rc = _agg_rf = 0

            for _prov, _lm in self.llm_metrics.items():
                _lc = _lm.calls
                _agg_calls += _lc
                _agg_pt += _lm.total_prompt_tokens
                _agg_ct += _lm.total_completion_tokens
                _agg_time += _lm.total_time_ms
                _agg_failures += _lm.failures
                _rc = _lm.recent_calls
                _rf = _lm.recent_failures
                _agg_rc += _rc
                _agg_rf += _rf

                # O(1) delta-cache read. LLM latency is stored in MILLISECONDS
                # (record_llm_call receives execution_time_ms), so these emit
                # as-is for p50_ms/p95_ms — no unit conversion.
                _lp_p50, _lp_p95 = _lm.percentiles((50, 95))
                llm_provider_summary[_prov] = {
                    'calls': _lc,
                    'total_prompt_tokens': _lm.total_prompt_tokens,
                    'total_completion_tokens': _lm.total_completion_tokens,
                    'total_tokens': _lm.total_prompt_tokens + _lm.total_completion_tokens,
                    'avg_time_ms_per_call': (_lm.total_time_ms / _lc) if _lc else 0.0,
                    'failures': _lm.failures,
                    'failure_rate': (_lm.failures / _lc) if _lc > 0 else 0.0,
                    'recent_calls': _rc,
                    'recent_failures': _rf,
                    'recent_failure_rate': (_rf / _rc) if _rc > 0 else 0.0,
                    # Per-provider recent-latency distribution (see ToolMetrics p50/p95).
                    'p50_ms': _lp_p50,
                    'p95_ms': _lp_p95,
                    # Latency sample count for the min-samples floor
                    # (symmetric with tool p95_n).
                    'p95_n': _lm.latency_samples_count,
                }
            llm_calls = _agg_calls
            llm_prompt = _agg_pt
            llm_completion = _agg_ct
            llm_failures = _agg_failures
            llm_avg_ms = (_agg_time / _agg_calls) if _agg_calls else 0.0
            llm_failure_rate = (_agg_failures / _agg_calls) if _agg_calls > 0 else 0.0
            llm_recent_calls = _agg_rc
            llm_recent_failures = _agg_rf
            llm_recent_failure_rate = (_agg_rf / _agg_rc) if _agg_rc > 0 else 0.0

            rag_searches = self.rag_searches
            rag_time_ms = self.rag_search_time_ms

            # Turn-level agent result channel — the ONLY place truncation
            # storms surface (see record_agent_result). Snapshot under the same
            # lock as the tool/llm channels for consistency.
            _arm = self.agent_result_metrics
            agent_result_summary = {
                'turns': _arm.turns,
                'failed_turns': _arm.failed_turns,
                'truncated_turns': _arm.truncated_turns,
                'failure_rate': _arm.failure_rate,
                'truncation_rate': _arm.truncation_rate,
                'avg_time_ms': _arm.avg_time_ms,
                'total_time_ms': _arm.total_time_ms,
                'recent_turns': _arm.recent_turns,
                'recent_failed_turns': _arm.recent_failed_turns,
                'recent_failure_rate': _arm.recent_failure_rate,
                'recent_truncated_turns': _arm.recent_truncated_turns,
                'recent_truncation_rate': _arm.recent_truncation_rate,
            }

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
                logger.debug("cache stats failed", exc_info=True)
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

        # Top failing tools — derived (pure) from the already-built tool_summary so
        # the dashboard card, the per-turn summary, and warn_failing_tools() all
        # read ONE computation (no second filtering site to drift from). Gates on
        # BOTH failure_rate and min_calls (config) so a single transient failure
        # on a cold tool (1/1 = 100%) does not trip it.
        failing_tools = top_failing_tools(
            tool_summary,
            threshold=_threshold_config.scores.TOOL_FAILURE_RATE_WARN,
            min_calls=_threshold_config.scores.TOOL_FAILURE_WARN_MIN_CALLS,
        )

        # Overall cache hit-rate across ALL live cache channels. tool_result_cache is
        # the largest by volume (every cacheable tool call flows through it), so it
        # MUST be in the sum — excluding it made the headline reflect only rag+vector.
        # tool_result_cache_stats may be None when no cache is registered; treat as a
        # zero contribution.
        _trc = tool_result_cache_stats
        _trc_hits = _trc["hits"] if _trc else 0
        _trc_total = (_trc["hits"] + _trc["misses"]) if _trc else 0
        _all_hits = (
            cache_stats['rag']['hits']
            + cache_stats['vector']['hits'] + _trc_hits
        )
        _all_total = (
            cache_stats['rag']['total']
            + cache_stats['vector']['total'] + _trc_total
        )
        overall_hit_rate = _all_hits / _all_total if _all_total > 0 else 0

        # AGGREGATE LLM headline: sums across all
        # providers. Its recent_failure_rate is the diluted cross-provider number
        # (honest as an overall read) — NOT what the gate uses. The gate reads each
        # provider's OWN window via top_failing_llm(llm_provider_summary, ...) below.
        llm_summary = {
            'calls': llm_calls,
            'total_prompt_tokens': llm_prompt,
            'total_completion_tokens': llm_completion,
            'total_tokens': llm_prompt + llm_completion,
            'avg_time_ms_per_call': llm_avg_ms,
            'failures': llm_failures,
            # Cumulative rate — display/context ("how bad overall"), diluted by history.
            'failure_rate': llm_failure_rate,
            # THE live-health gate fields (mirror the per-tool recent_* keys).
            'recent_calls': llm_recent_calls,
            'recent_failures': llm_recent_failures,
            'recent_failure_rate': llm_recent_failure_rate,
        }
        # LLM health gate — symmetric with failing_tools but iterates the PER-
        # PROVIDER breakdown so a failing fallback provider is not diluted by a
        # healthy primary. Returns one entry per provider tripping its own
        # recent_failure_rate + min_calls gate. Empty when every provider is healthy
        # / cold. The dashboard card, per-turn summary, and warn_failing_llm() all
        # read this one derivation.
        failing_llm = top_failing_llm(
            llm_provider_summary,
            threshold=_threshold_config.scores.TOOL_FAILURE_RATE_WARN,
            min_calls=_threshold_config.scores.TOOL_FAILURE_WARN_MIN_CALLS,
        )

        # Top slow tools — derived (pure) from the already-built tool_summary, gated
        # on p95_ms ≥ TOOL_LATENCY_P95_WARN_MS. The dashboard card and warn_slow_tools()
        # both read this one derivation (not a second independent filter).
        slow_tools = top_slow_tools(
            tool_summary,
            threshold_ms=_threshold_config.scores.TOOL_LATENCY_P95_WARN_MS,
            min_samples=_threshold_config.scores.LATENCY_P95_MIN_SAMPLES,
        )

        # Top slow LLM providers — symmetric with slow_tools, gated on p95_ms ≥
        # LLM_LATENCY_P95_WARN_MS. Dashboard + warn_slow_llm() read this.
        slow_llm = top_slow_llm(
            llm_provider_summary,
            threshold_ms=_threshold_config.scores.LLM_LATENCY_P95_WARN_MS,
            min_samples=_threshold_config.scores.LATENCY_P95_MIN_SAMPLES,
        )

        return {
            'session_id': self.session_id,
            'total_execution_time_seconds': total_execution_time,
            'start_time': self.start_time,
            'end_time': self.end_time,

            'llm_metrics': llm_summary,

            # Per-provider LLM breakdown — the gate input for top_failing_llm().
            # One entry per provider (keyed by lower-cased provider name) carrying
            # the same shape as the aggregate llm_metrics above. Empty when no LLM
            # calls yet. Additive to the headline; no existing consumer breaks.
            'llm_providers': llm_provider_summary,

            'tool_metrics': tool_summary,

            # Sorted list of {name, failures, total_calls, failure_rate} for tools
            # exceeding the configured health gates — empty when nothing is failing.
            'failing_tools': failing_tools,

            # One entry per provider whose recent_failure_rate trips the gate —
            # empty when all providers are healthy / cold. Per-provider isolation
            # means a failing fallback is no longer diluted by a healthy primary.
            'failing_llm': failing_llm,

            # Sorted list of {name, p50_ms, p95_ms, call_count} for tools whose
            # recent p95_ms exceeds the configured latency threshold — empty when
            # all tools are healthy. The dashboard "Slow Tools" card reads this.
            'slow_tools': slow_tools,

            # One entry per provider whose recent p95_ms exceeds the configured
            # latency threshold — empty when all providers are healthy / cold.
            'slow_llm': slow_llm,

            'cache_metrics': {
                'rag_cache': cache_stats['rag'],
                'vector_cache': cache_stats['vector'],
                'tool_result_cache': tool_result_cache_stats,
                'overall_hit_rate': overall_hit_rate,
                # Aggregate hits/total behind overall_hit_rate — shipped so the UI
                # derives BOTH the rate AND the Hits/Misses detail from ONE source
                # (this backend formula) instead of mirroring it client-side and
                # silently drifting the next time a cache channel is added.
                'overall_hits': _all_hits,
                'overall_total': _all_total,
            },

            'rag_metrics': {
                'searches': rag_searches,
                'total_search_time_ms': rag_time_ms,
                'avg_search_time_ms': avg_rag_search_time
            },

            # Turn-level agent outcome channel — see record_agent_result. All
            # zeros until the first outcome event is recorded.
            'agent_result': agent_result_summary,
        }


# ── failure_rate consumers ─────────────────────────────────────────────────
# ``ToolMetrics.failure_rate`` (and the per-tool ``failures``/``failure_rate``
# keys in get_summary()) were a dead signal: produced but never read by any
# decision logic or UI. The two helpers below make it observable —
# ``top_failing_tools`` is the pure derivation (SSOT for the dashboard card AND
# the embedded ``failing_tools`` summary key); ``warn_failing_tools`` is the
# deduped server-side warning the SSE broadcaster emits so operators see a
# degraded tool without opening the dashboard.


def _top_by_metric(
    entries, *, derive, sort_key, top_n=None
) -> list[dict]:
    """Shared skeleton of the top_* pure derivations: derive → sort → cap.

    *entries*: iterable of ``(name, metrics_dict)`` pairs (``.items()`` of a
    summary section). *derive*: ``(name, m) -> dict | None`` — returns the
    output entry when *m* trips its gate, else ``None``. *sort_key*:
    ``(entry) -> tuple`` for the deterministic sort (gate rate first, then raw
    count, then name). *top_n*: result cap (``None`` = unbounded, used by
    ``top_failing_llm`` where the provider space is naturally bounded).

    The ``isinstance(m, dict)`` guard lives here for every derivation: a
    malformed summary entry must be skipped, not crash the SSE poll — this
    also closes the parity gap where the llm variants used to lack the guard
    their tool siblings had.
    """
    out: list[dict] = []
    for name, m in entries:
        if not isinstance(m, dict):
            continue
        entry = derive(name, m)
        if entry is not None:
            out.append(entry)
    out.sort(key=sort_key)
    return out[:top_n] if top_n is not None else out


def _derive_failing(
    name, m, threshold: float, min_calls: int, total_key: str
) -> dict | None:
    """Gate + build for the failing-* derivations.

    *total_key* is the cumulative-count key — ``"total_calls"`` (tool
    summary) or ``"calls"`` (provider summary) — used both for the
    ``min_calls`` floor and as the output key.
    """
    calls = m.get(total_key, 0)
    if calls < min_calls:
        return None
    failures = m.get("failures", 0)
    if failures <= 0:
        return None
    # THE gate: recent_failure_rate over the sliding window. Fall back to the
    # cumulative rate only when the summary predates the recent field (older
    # snapshot) so this stays callable against a minimal dict.
    recent_rate = m.get("recent_failure_rate")
    if recent_rate is None:
        recent_rate = failures / calls if calls else 0.0
    if recent_rate < threshold:
        return None
    return {
        "name": name,
        "failures": failures,
        total_key: calls,
        # Cumulative rate — kept for display context ("how bad overall").
        "failure_rate": m.get("failure_rate", (failures / calls if calls else 0.0)),
        "recent_calls": m.get("recent_calls", 0),
        "recent_failures": m.get("recent_failures", 0),
        # The live rate that tripped the gate.
        "recent_failure_rate": recent_rate,
    }


def _derive_slow(
    name, m, threshold_ms: float, min_samples: int, count_key: str
) -> dict | None:
    """Gate + build for the slow-* derivations.

    *count_key* is the sample-count output key — ``"call_count"`` (tool
    summary) or ``"calls"`` (provider summary).
    """
    p95 = m.get("p95_ms")
    if p95 is None or p95 < threshold_ms:
        return None
    # Enforce min-samples floor: too few successful latency samples make p95
    # unreliable — a single slow first call should not trip the gate.
    # Symmetric with the failing derivations' min_calls guard.
    if m.get("p95_n", 0) < min_samples:
        return None
    return {
        "name": name,
        "p50_ms": m.get("p50_ms", 0.0),
        "p95_ms": p95,
        count_key: m.get(count_key, 0),
    }


def top_failing_tools(
    tool_metrics: dict,
    *,
    threshold: float,
    min_calls: int,
    top_n: int = 5,
) -> list[dict]:
    """Return the tools whose RECENT failure rate ≥ ``threshold`` AND
    ``total_calls`` ≥ ``min_calls``, sorted by recent_failure_rate desc then
    failures desc (deterministic tie-break), capped at ``top_n``.

    The gate is ``recent_failure_rate`` (the sliding-window live-health signal),
    NOT the cumulative ``failure_rate``. A long autonomous run dilutes the
    cumulative rate so badly that a tool currently failing every call reads
    healthy (1000 successes + 20 failures ≈ 2% cumulative); the recent window
    tracks CURRENT health so a freshly-broken tool trips within ~N calls.

    Pure: takes the ``tool_metrics`` dict shape produced by ``get_summary()``
    (values carry ``failures``/``total_calls``/``failure_rate``/
    ``recent_calls``/``recent_failures``/``recent_failure_rate``) and returns a
    fresh list of ``{name, failures, total_calls, failure_rate, recent_calls,
    recent_failures, recent_failure_rate}``. No collector instance is required,
    so this is callable from tests, the per-turn summary, and the self-improve
    orchestrator without holding the collector lock.

    The ``min_calls`` gate is load-bearing: applied to the CUMULATIVE
    ``total_calls`` (a floor against one-shot noise — a single transient failure
    on a cold tool must not permanently flag it). Since ``recent_calls`` =
    ``min(total_calls, N)``, ``total_calls ≥ min_calls`` also guarantees
    ``recent_calls ≥ min_calls`` samples, so no separate recent-sample knob is
    needed. ``threshold`` is the health gate (config.scores.TOOL_FAILURE_RATE_WARN).
    """
    return _top_by_metric(
        tool_metrics.items(),
        derive=lambda name, m: _derive_failing(name, m, threshold, min_calls, "total_calls"),
        sort_key=lambda t: (-t["recent_failure_rate"], -t["failures"], t["name"]),
        top_n=top_n,
    )


def top_failing_llm(
    provider_summary: dict,
    *,
    threshold: float,
    min_calls: int,
) -> list[dict]:
    """Return a list of LLM PROVIDERS whose RECENT failure rate trips the health
    gate, else ``[]``.

    Iterates the per-provider breakdown (``summary['llm_providers']``) — symmetric
    with ``top_failing_tools`` iterating the per-tool summary. Each provider is gated
    on its OWN ``recent_failure_rate`` + ``min_calls``, so a failing fallback provider
    is NOT diluted by a healthy primary's traffic within one shared deque (the exact
    blind spot a single aggregate stream had). Reuses the SAME threshold/window config
    the tool side uses.

    Pure: takes the ``llm_providers`` dict shape produced by ``get_summary()`` (each
    entry carries ``calls``/``failures``/``failure_rate``/``recent_calls``/
    ``recent_failures``/``recent_failure_rate``) and returns a list of
    ``{name, calls, failures, failure_rate, recent_calls, recent_failures,
    recent_failure_rate}`` for the providers that trip the gate, sorted by
    recent_failure_rate desc (most-degraded provider first; ties by raw failure count,
    then name for determinism). No collector instance required — callable from tests /
    per-turn summary / self-improve.
    """
    # No top_n: the provider space is naturally bounded, so every degraded
    # provider surfaces (the tool side caps at top_n for the dashboard card).
    return _top_by_metric(
        (provider_summary or {}).items(),
        derive=lambda name, m: _derive_failing(name, m, threshold, min_calls, "calls"),
        sort_key=lambda t: (-t["recent_failure_rate"], -t["failures"], t["name"]),
    )


# ── p95 latency consumers ───────────────────────────────────────────────────
# ``p50_ms``/``p95_ms`` per tool and per LLM provider are shipped in every
# get_summary() call but had no consumer (raw JSON only) — the latency distribution
# feature was inert. The two helpers below close the loop: ``top_slow_tools`` and
# ``top_slow_llm`` are the pure derivations (SSOT for the summary key AND the
# dashboard card); ``warn_slow_tools`` / ``warn_slow_llm`` are the deduped log gate
# (single-consumer contract, symmetric with ``warn_failing_tools``).


def top_slow_tools(
    tool_metrics: dict,
    *,
    threshold_ms: float,
    top_n: int = 3,
    min_samples: int = 5,
) -> list[dict]:
    """Return the tools whose recent p95_ms ≥ ``threshold_ms``, sorted descending
    by p95_ms, capped at ``top_n``.

    Pure: takes the ``tool_metrics`` dict shape produced by ``get_summary()``
    (values carry ``p50_ms``/``p95_ms``/``p95_n`` alongside the failure/cache
    fields) and returns a fresh list of ``{name, p50_ms, p95_ms, call_count}``.
    No collector instance required — callable from tests / per-turn summary /
    dashboard card.

    The ``threshold_ms`` gate is load-bearing: a tool with p95 below this is
    healthy. The ``min_samples`` floor prevents a single slow call (cold start,
    first-of-run network hit) from tripping a false "degraded" warning —
    symmetric with ``top_failing_tools()``\\'s ``min_calls`` for the failure axis.
    Default threshold is ``TOOL_LATENCY_P95_WARN_MS`` (5s). ``top_n`` limits the
    list to the worst offenders (dashboard card shows top 3).
    """
    return _top_by_metric(
        tool_metrics.items(),
        derive=lambda name, m: _derive_slow(name, m, threshold_ms, min_samples, "call_count"),
        sort_key=lambda t: (-t["p95_ms"], -t["call_count"], t["name"]),
        top_n=top_n,
    )


def top_slow_llm(
    provider_summary: dict,
    *,
    threshold_ms: float,
    top_n: int = 3,
    min_samples: int = 5,
) -> list[dict]:
    """Return the LLM providers whose recent p95_ms ≥ ``threshold_ms``, sorted
    descending by p95_ms, capped at ``top_n``.

    Iterates the per-provider breakdown (``summary['llm_providers']``) — symmetric
    with ``top_slow_tools`` iterating the per-tool summary. Pure: takes the
    ``llm_providers`` dict shape produced by ``get_summary()`` and returns a list
    of ``{name, p50_ms, p95_ms, calls}``. No collector instance required.

    The ``min_samples`` floor is symmetric with ``top_slow_tools`` — prevents a
    single slow call from tripping a false "degraded" warning. Default threshold
    is ``LLM_LATENCY_P95_WARN_MS`` (30s).
    """
    return _top_by_metric(
        (provider_summary or {}).items(),
        derive=lambda name, m: _derive_slow(name, m, threshold_ms, min_samples, "calls"),
        sort_key=lambda t: (-t["p95_ms"], -t["calls"], t["name"]),
        top_n=top_n,
    )


# ── deduped warn gates ───────────────────────────────────────────────────────
# Shared machinery for warn_failing_tools/llm and warn_slow_tools/llm. The SSE
# broadcaster polls get_summary() every 2s; without dedup the same degraded
# tool/provider would log every tick. Each gate owns ONE _WarnDedup instance, so
# a tool regression, an LLM regression, a slow tool and a slow provider are four
# independent operator signals (separate re-arm bookkeeping).

class _WarnDedup:
    """Dedup + re-arm machinery shared by the warn_* log gates.

    Re-arm semantics: ``warn()`` rebuilds the warned set from the CURRENT entry
    set each call, so an entry that recovers (leaves the set) becomes warnable
    again — a later regression re-warns rather than being silently suppressed.
    The lock keeps the read-modify-write atomic for concurrent broadcasters
    (there is only one, but the lock keeps the contract honest).
    """

    __slots__ = ("lock", "warned")

    def __init__(self) -> None:
        self.warned: set[str] = set()
        self.lock = threading.Lock()

    def reset(self) -> None:
        """Clear the warn-dedup set (test-only: gives each test a clean slate)."""
        with self.lock:
            self.warned.clear()

    def warn(
        self,
        entries: list[dict],
        formatter: Callable[[dict], str],
        log: Callable[[str], None],
    ) -> int:
        """Log once per named entry in ``entries`` that is new this streak.

        Returns the count of newly-warned entries (0 when nothing new).
        ``formatter(entry)`` is called ONLY for newly-warned entries and must
        return a single pre-formatted ``str`` (NOT printf-style ``*args``);
        ``log`` receives that ``str``, so any ``(str) -> None`` callable works
        (``logger.warning``, ``list.append``, …).
        """
        current = {t.get("name") for t in entries if t.get("name")}
        with self.lock:
            newly = current - self.warned
            # Rebuild from current: entries no longer present drop out (re-arm);
            # entries still present stay (suppressed). The whole dedup in one line.
            self.warned.clear()
            self.warned.update(current)
        for t in entries:
            if t.get("name") in newly:
                log(formatter(t))
        return len(newly)


# One gate per operator signal — a tool failure, an LLM failure, a slow tool and
# a slow LLM provider are distinct signals and must not share re-arm bookkeeping.
# Single-consumer contract: the SSE broadcaster (webapp/routes/stats.py) is the
# SOLE caller of the warn_* functions; logic consumers read the pure top_* /
# summary derivations directly.
_warned_failing_tools = _WarnDedup()
_warned_failing_llm = _WarnDedup()
_warned_slow_tools = _WarnDedup()
_warned_slow_llm = _WarnDedup()


def _reset_warned_failing_tools() -> None:
    """Clear the warn-dedup set (test-only: gives each test a clean slate)."""
    _warned_failing_tools.reset()


def _reset_warned_failing_llm() -> None:
    """Clear the warn-dedup set (test-only: gives each test a clean slate)."""
    _warned_failing_llm.reset()


def _reset_warned_slow_tools() -> None:
    """Clear the slow-tool warn-dedup set (test-only)."""
    _warned_slow_tools.reset()


def _reset_warned_slow_llm() -> None:
    """Clear the slow-llm warn-dedup set (test-only)."""
    _warned_slow_llm.reset()


def _fmt_failing(entry_kind: str, total_key: str, t: dict) -> str:
    """Format a failing entry: the RECENT rate (what tripped the gate) plus
    the windowed count, so the operator sees CURRENT health, not a diluted
    lifetime average. Falls back to the cumulative rate/total for an older
    summary snapshot lacking the recent fields.

    *entry_kind* is the display label (``"tool"`` / ``"LLM provider"``) and
    *total_key* the cumulative-count key (``"total_calls"`` for the tool
    summary, ``"calls"`` for the provider summary).
    """
    rate = t.get("recent_failure_rate")
    if rate is None:
        rate = t.get("failure_rate", 0.0)
        rfail = t.get("failures", 0)
        rtot = t.get(total_key, 0)
    else:
        rfail = t.get("recent_failures", t.get("failures", 0))
        rtot = t.get("recent_calls", t.get(total_key, 0))
    return (
        f"{entry_kind} '{t.get('name')}' recent failure_rate "
        f"{rate * 100.0:.0f}% ({rfail}/{rtot} recent calls) exceeds health threshold"
    )


def _fmt_failing_tool(t: dict) -> str:
    """Format a failing-tool entry (see :func:`_fmt_failing`)."""
    return _fmt_failing("tool", "total_calls", t)


def warn_failing_tools(summary: dict, *, log=logger.warning) -> int:
    """Emit a deduped ``warning`` log for each tool in
    ``summary['failing_tools']`` not yet warned this failing streak.

    Returns the count of newly-warned tools (0 when nothing new). Dedup +
    re-arm machinery is shared (see ``_WarnDedup``). Single-consumer contract:
    intended for the SSE broadcaster ONLY; logic consumers (self-improve
    orchestrator, per-turn summary, tests) must read the PURE
    ``top_failing_tools()`` derivation (or ``summary['failing_tools']``) directly —
    those carry no state and are safe to call from anywhere.
    """
    return _warned_failing_tools.warn(
        summary.get("failing_tools") or [], _fmt_failing_tool, log
    )


def _fmt_failing_llm(t: dict) -> str:
    """Format a failing-LLM entry (see :func:`_fmt_failing`)."""
    return _fmt_failing("LLM provider", "calls", t)


def warn_failing_llm(summary: dict, *, log=logger.warning) -> int:
    """Emit a deduped ``warning`` log when an LLM provider enters
    ``summary['failing_llm']`` (its recent_failure_rate tripped the gate).

    Symmetric with ``warn_failing_tools``: same shared dedup + re-arm machinery
    (see ``_WarnDedup``), a SEPARATE dedup gate, and a provider-appropriate
    message. Returns the count of newly-warned providers (0 or more,
    per-provider). Single-consumer: intended for the SSE broadcaster ONLY; logic
    consumers read ``summary['failing_llm']`` directly.
    """
    return _warned_failing_llm.warn(
        summary.get("failing_llm") or [], _fmt_failing_llm, log
    )


def _fmt_slow_tool(t: dict) -> str:
    """Format a slow-tool entry (p95/p50 latency pair)."""
    return (
        f"tool '{t.get('name')}' p95 latency {t.get('p95_ms', 0.0):.0f}ms "
        f"(p50={t.get('p50_ms', 0.0):.0f}ms) exceeds slow-tool threshold"
    )


def warn_slow_tools(summary: dict, *, log=logger.warning) -> int:
    """Emit a deduped ``warning`` log for each tool in
    ``summary['slow_tools']`` not yet warned this slow streak.

    Same dedup + re-arm semantics as ``warn_failing_tools`` (shared machinery,
    see ``_WarnDedup``): a tool is warned ONCE per slow streak; recovery
    (dropping out of the list) re-arms it so a later regression re-warns.
    Single-consumer contract (SSE broadcaster ONLY). Returns the count of
    newly-warned tools.
    """
    return _warned_slow_tools.warn(
        summary.get("slow_tools") or [], _fmt_slow_tool, log
    )


def _fmt_slow_llm(t: dict) -> str:
    """Format a slow-LLM entry (p95/p50 latency pair)."""
    return (
        f"LLM provider '{t.get('name')}' p95 latency "
        f"{t.get('p95_ms', 0.0):.0f}ms (p50={t.get('p50_ms', 0.0):.0f}ms) "
        "exceeds slow-provider threshold"
    )


def warn_slow_llm(summary: dict, *, log=logger.warning) -> int:
    """Emit a deduped ``warning`` log for each LLM provider in
    ``summary['slow_llm']`` not yet warned this slow streak.

    Symmetric with ``warn_slow_tools``: same shared dedup + re-arm machinery
    (see ``_WarnDedup``), a SEPARATE dedup gate, and a provider-appropriate
    message. Single-consumer contract (SSE broadcaster ONLY). Returns the count
    of newly-warned providers.
    """
    return _warned_slow_llm.warn(
        summary.get("slow_llm") or [], _fmt_slow_llm, log
    )


# Global collector for easy access
_global_collector: Optional[PerformanceCollector] = None
_global_collector_lock = threading.Lock()


def get_global_collector() -> PerformanceCollector:
    """Get or create global performance collector (thread-safe DCL).

    The global collector is a process-lifetime singleton used for dashboard
    aggregation (stats.py SSE broadcaster). It auto-starts an uptime timer
    so the dashboard sees ``total_execution_time_seconds`` ≈ process uptime
    even after per-loop collectors were restored for session isolation.
    """
    global _global_collector
    if _global_collector is None:
        with _global_collector_lock:
            if _global_collector is None:
                _global_collector = PerformanceCollector()
                _global_collector.start_session()
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
