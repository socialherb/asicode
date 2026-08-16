
"""Centralized threshold/limit/constant configuration.

All hardcoded numeric thresholds across the codebase are defined here as the
single source of truth. Import via `from .config.thresholds import config`
(inside `external_llm.agent.*`) or `from external_llm.agent.config.thresholds
import config` (everywhere else). Never redefine these values in-place.

Categories:
    tokens   — LLM max_tokens per call site (output budget)
    lines    — content/char/byte truncation budgets
    counts   — iteration/sample/file count caps
    scores   — confidence/similarity/score gates

Some domain policy modules keep their own constants (`weight_learning.py`)
because the values are tightly coupled to that module's algorithm. They remain
defined in-place by design — they are policy, not magic numbers.
"""

import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean env var (1/true/yes/on vs 0/false/no/off); fallback to default."""
    v = (os.getenv(name, "") or "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Parse an int env var; fallback to default on empty/invalid/out-of-range.

    ``minimum`` is the smallest accepted value. Anything below it is treated as
    misconfiguration and yields *default* rather than being clamped — a typo
    should not silently become a boundary value. ``minimum=1`` keeps the
    original positive-only contract for existing callers; pass ``minimum=0``
    where zero is a meaningful setting (e.g. a "never" cap).
    """
    try:
        v = int((os.getenv(name, "") or "").strip() or str(default))
    except Exception:
        return default
    else:
        return v if v >= minimum else default


@dataclass(frozen=True)
class TokenLimits:
    """LLM `max_tokens` per call site. Names encode the call path so each
    site's intent is explicit. Repair / retry sites are sized larger than the
    first attempt — retrying with the same budget that just failed reproduces
    the failure (verified anti-pattern: instruction/plan repair loops in
    `llm_execution.py` previously fed v4-flash 2048 tokens twice in a row)."""

    INSTRUCTION_JSON: int = 4096
    INSTRUCTION_REPAIR: int = 8192
    PLAN_JSON: int = 8192
    PLAN_REPAIR: int = 16384
    INTENT_CLASSIFY: int = 4096
    # 8192: reasoner models bill reasoning tokens against max_tokens, so a 4096
    # budget is exhausted by reasoning alone and every resolve call truncates
    # (observed 5/5 self_eval runs: "truncated ... retrying with 8192").
    INTENT_RESOLVER_DEFAULT: int = 8192
    SERVICE_DEFAULT: int = 4096
    SERVICE_REPAIR: int = 8192
    SUBAGENT_SHORT: int = 2000
    LOCAL_ASSISTANT_SHORT: int = 512
    LOCAL_MODEL_CONTEXT_CHARS: int = 4000
    INTELLIGENT_SERVICE_DEFAULT: int = 4096
    AGENT_STREAM: int = 4096
    ANTHROPIC_DEFAULT: int = 65536

    AGENT_TOOL_CALL: int = 32768  # agent_loop.py _llm_call_with_tools default max_tokens

    CONTEXT_HARD_CAP_SAFETY_MARGIN: int = 1024  # agent_loop.py / design_chat_loop.py: pre-flight
                                                 # message-budget margin subtracted from the model's
                                                 # context_limit for the structural-collapse check.
                                                 # Prevents HTTP 400 "max context length exceeded"
                                                 # errors from API providers (DeepSeek 1M, etc.).

    BASH_OUTPUT_MAX_CHARS: int = 60_000   # git_tools.py: max chars returned by `bash` tool output
                                           # Prevents sudden token surges from large stdout/stderr.
                                           # Sized for the WORST-case density, not prose: token-dense
                                           # ASCII (timestamps, hashes, JSON, base64, `ls -la` listings)
                                           # tokenizes at ~2 chars/token, so 60K chars ≈ 30K tokens
                                           # worst-case (≈20K for prose at 3 chars/token). The old 100K
                                           # cap assumed 3 chars/token universally and let dense output
                                           # hit ~47K tokens — 1.5x the intended budget. (< 10% of 1M)

@dataclass(frozen=True)
class LineLimits:
    """File read caps — soft limits to avoid OOM on huge files."""

    DESIGN_TURN_MAX_CHARS: int = 100000
    RAG_FILE_CHARS: int = 200_000
    UI_FULL_MAX_LINES: int = 1000

    # ── read_file output budget ──────────────────────────────────────────
    # Largest file read_file returns in full when the model passes no line
    # range. 200 -> 800: the median module in this repo is 281 lines, so the
    # old cap made the MEDIAN file cost two round-trips (59% of files here
    # exceeded 200; 21% exceed 800). The model had to pick a range with no
    # idea what was where, so it usually guessed twice. Over the cap the tool
    # now returns the file outline instead of a bare line count, so one extra
    # call is enough and it is an informed one.
    READ_FILE_FULL_LINES: int = 800
    # Hard ceiling on emitted content, applied to EVERY read_file path —
    # including an explicit start_line/end_line, which previously had no cap
    # at all: `end_line=999999` on a 6.4K-line module returned ~388K chars
    # (~130K tokens), enough to blow the whole context window in one call.
    # Sized like BASH_OUTPUT_MAX_CHARS for worst-case token density (~2
    # chars/token for dense ASCII => ~30K tokens), not for prose.
    READ_FILE_MAX_CHARS: int = 60_000
    # Per-LINE clamp for search output (grep). The char budget below bounds how
    # many lines are emitted but said nothing about how WIDE one is, so a
    # single match inside a minified bundle, a .map, or one-line JSON returned
    # the whole line: measured 34,000,257 chars from one match against a
    # 60,000-char cap — 566x, straight into the conversation history. 2,000 is
    # generous for real source (the longest line in this repo is under 400) and
    # is what rg's own --max-columns is given, so the clamp happens in the
    # child for the rg path and only on retention for the grep fallback.
    SEARCH_MAX_LINE_CHARS: int = 2_000
    # Symbols listed in the over-cap outline before truncating. Enough to
    # cover a large module's public surface without the outline itself
    # becoming the thing that costs a round-trip.
    READ_FILE_OUTLINE_MAX_SYMBOLS: int = 60

    # ── call-graph indexing caps ─────────────────────────────────────────
    # Largest source file CallGraphIndexer will parse in process. build() had
    # NO size gate of any kind while every sibling path grew one (P26-4 for
    # the batch tree-sitter walker, _too_big_to_parse_inproc for the per-file
    # symbol-search entry points), and it is reachable from the shipping
    # analyze_impact / trace_call_path tools.
    #
    # Measured: one 3.7 MB generated .py in an otherwise empty repo cost
    # build() 10.12 s and 762 MB peak RSS. ast.parse alone is ~155x the source
    # size in transient memory, consistently across sizes:
    #     0.34 MB -> 0.07 s /  56 MB      1.48 MB -> 0.28 s / 232 MB
    #     0.98 MB -> 0.19 s / 160 MB      3.72 MB -> 0.85 s / 570 MB
    # so the CAP is what bounds the peak, not the average file.
    #
    # 1 MiB is ~3x the largest first-party file in this repo (repl_impl.py at
    # 358 KB) and far above any hand-written module (CPython's _pydecimal.py is
    # 230 KB, SQLAlchemy's largest ~400 KB). What it excludes is the generated
    # class — *_pb2.py, generated API clients, bundled migrations — where the
    # call-graph value is lowest and the size is unbounded. The vendored-code
    # case does not arise: the walker already skips .venv / site-packages.
    CALLGRAPH_PY_MAX_BYTES: int = 1 << 20
    # Same gate for the TS/JS tree-sitter path. Deliberately the SAME number as
    # symbol_search's _NONPY_INPROC_MAX_BYTES rather than a third threshold —
    # that is the trade P26-4 already made for tree-sitter, and a minified
    # bundle is the shape both are defending against.
    CALLGRAPH_TS_MAX_BYTES: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ScoreThresholds:
    """Confidence/similarity/score gates."""

    # ── Semantic intent fallback (embedding cosine) ──────────────────────────
    # semantic_intent.py matchers score a query by *mean* cosine to each label's
    # example set and pick the top label only if it clears MIN and beats the
    # runner-up label by MARGIN. Mean+margin (not argmax over individual rows)
    # is what makes this robust: a constant anisotropy offset — large for some
    # multilingual models, and the reason raw cosines run high — cancels in the
    # margin, so the same MARGIN separates intents across models. MIN is a light
    # floor against matching unrelated text; MARGIN does the real work and is set
    # to reject the worst observed false positive (additive/refactor phrasings
    # that share imperative surface form with removal) on both the multilingual
    # and English fallback models.
    SEMANTIC_INTENT_MIN: float = 0.10
    SEMANTIC_INTENT_MARGIN: float = 0.08


    # ── Tool health (failure_rate consumer) ─────────────────────────────────
    # A tool whose failure_rate ≥ this trips a ``warn_failing_tools`` health
    # warning and appears in the dashboard "Top Failing Tools" card. 0.5 = a tool
    # that fails at least half its calls is clearly degraded.
    TOOL_FAILURE_RATE_WARN: float = 0.50
    # Minimum total_calls before a tool is eligible for the warn/card. Without a
    # floor, a single transient failure on a tool called once (1/1 = 100%) would
    # permanently flag it. 3 calls is the smallest sample where a ≥50% rate means
    # ≥2 failures — a real signal, not one-shot noise.
    TOOL_FAILURE_WARN_MIN_CALLS: int = 3
    # Sliding window (call count) for ``recent_failure_rate`` — the LIVE-health
    # signal that gates ``warn_failing_tools`` and the "Top Failing Tools" card
    # (the cumulative failure_rate is kept for display but no longer gates). A
    # 12h+ run dilutes the cumulative rate so badly that a tool failing its last
    # 20 calls after 1000 successes reads ~2% and never trips the warn gate; the
    # recent window tracks CURRENT health instead. With window=30 and threshold
    # 0.50, a previously-healthy tool trips after ~15 consecutive failures, while
    # a ≤10-failure transient burst stays under 50% (10/30) and is ignored. The
    # window is a bounded per-tool deque (maxlen), so memory is constant
    # regardless of run length. min_calls(TOOL_FAILURE_WARN_MIN_CALLS) on the
    # CUMULATIVE total still applies as a floor: since recent = min(total, N),
    # total≥min_calls ⇒ recent≥min_calls samples, so the floor also guarantees a
    # minimum recent sample count (no separate knob needed).
    TOOL_FAILURE_RATE_WINDOW: int = 30

    # ── Latency distribution (p50 / p95) ───────────────────────────────────
    # Bounded RECENT-latency window (call count) kept per tool / per LLM
    # provider so get_summary() can ship p50_ms / p95_ms alongside the existing
    # O(1) avg. On a 12h+ run the avg is diluted flat by a mass of fast cache
    # hits while a degrading tool/provider shows up FIRST in the tail (p95) —
    # the same "history dilution" blind spot that motivated recent_failure_rate.
    # A bounded sliding window (NOT a uniform reservoir) is chosen precisely so
    # a uniform lifetime sample does not dilute the tail back toward early fast
    # calls; recent p95 tracks CURRENT degradation. Constant memory per tool /
    # provider (deque maxlen) regardless of run length; percentile() is
    # O(K log K) with K ≤ this cap, called at most once per tool/provider per
    # get_summary() (every ~2s). 128 ≈ a stable tail estimate that still fits a
    # tight memory budget across ~50 tools + N providers.
    LATENCY_SAMPLE_WINDOW: int = 128
    # p95 latency warn threshold (milliseconds). A tool whose recent p95 exceeds this
    # trips a ``warn_slow_tools`` health warning and appears in the dashboard "Slow
    # Tools" card. 5000ms = 5 seconds: any tool whose tail latency surpasses 5s is
    # clearly degraded (tool execution times are typically sub-second for code-serving
    # tools like read_file/grep; multi-second tools like web_search/bash have a higher
    # natural baseline, but a p95 above 5s on ANY tool warrants operator attention).
    TOOL_LATENCY_P95_WARN_MS: float = 5000.0
    # Same semantics for LLM providers. LLM calls are inherently slower (network I/O),
    # so the threshold is higher. 30000ms = 30 seconds: a provider whose p95 tail
    # exceeds 30s is degrading (typical LLM calls span 2-15s depending on model size
    # and output length; 30s p95 means a significant fraction of calls are stalling).
    LLM_LATENCY_P95_WARN_MS: float = 30000.0
    # Minimum number of latency samples needed before a tool/provider's p95 is
    # considered meaningful for slow-health gates (top_slow_tools / top_slow_llm).
    # Prevents a single slow call (cold start, first-of-run network hit) from
    # tripping a false "degraded" warning. Matches the design intent of
    # TOOL_FAILURE_WARN_MIN_CALLS (the failure gate's equivalent floor).
    LATENCY_P95_MIN_SAMPLES: int = 5


@dataclass(frozen=True)
class CountLimits:
    """Hardcoded upper bounds on iteration / sample / fan-out counts."""

    AGENT_NO_TOOL_NUDGE_MAX: int = 3
    AGENT_NO_PROGRESS_THRESHOLD: int = 5
    AGENT_FAIL_LOOP_LARGE: int = 3
    SYMBOL_MAX_PY_FILES: int = 3000
    SYMBOL_MAX_TS_FILES: int = 1500
    RAG_MAX_FILES: int = 3000
    PUSH_CLIENT_QUEUE_SIZE: int = 200
    PROACTIVE_DRAIN_INTERVAL_S: float = 1.0
    # Defense-in-depth cap on the autonomous task queue. Normal operation never
    # approaches this — policy cooldowns (_FILE_COOLDOWN, _KIND_COOLDOWN,
    # _AUTO_FIX_PER_HOUR) bound the enqueue rate. The cap is the last line of
    # defense if those policies are bypassed or misconfigured.
    AUTONOMOUS_TASK_QUEUE_MAX: int = 256
    # Cap on the per-repo ProactiveRunner registry. Each runner owns a drain
    # daemon thread + TriggerEngine schedule timers; an unbounded registry leaks
    # threads (not just memory) in long-lived multi-repo webapp processes.
    # Evicted runners are stop()'d (drain thread + engine timers torn down) on
    # overflow. See proactive_runner.get_or_create_runner.
    AUTONOMOUS_RUNNER_MAX: int = 8
    VULTURE_HUB_IMPORTER_THRESHOLD: int = 5  # arbitrary — no empirical basis yet; revisit after shadow log data accumulates


    # Scanner max_per_file defaults — prevents silent truncation from hiding issues.
    # Values are conservative (5-10) to avoid overwhelming callers with noise, but
    # each scanner logs a warning when the cap is hit so the caller can detect
    # incomplete results and widen the limit or re-scan with narrower scope.
    SCANNER_DEAD_BLOCK_MAX: int = 5
    SCANNER_PUBLIC_DEAD_BLOCK_MAX: int = 5
    SCANNER_VULTURE_MAX: int = 10
    SCANNER_VULTURE_MIN_CONFIDENCE: int = 60  # 0-100 raw Vulture confidence floor
    SCANNER_DUP_DEF_MAX: int = 10
    SCANNER_UNUSED_IMPORT_MAX: int = 10
    SCANNER_CONTAINER_REACH_MAX: int = 5
    SCANNER_CONTRADICTORY_MAX: int = 10
    SCANNER_CONTRADICTORY_DUP_DISTANCE: int = 100

    # ── Symbol Search / Tool Loop ────────────────────────────────────────
    SEARCH_RESULTS_CAP: int = 30             # symbol_search.py max results before early break
    AGENT_TOOL_RETRY_LIMIT: int = 5          # agent_loop.py per-tool cumulative exhaustion warning
    AGENT_MAX_TURNS_DEFAULT: int = 500         # tool_registry + agent_stream + asi + subagent (ipc/orchestrator/worker) fallbacks
    AGENT_MAX_TURNS_WEBAPP_MAX: int = 200      # webapp /agent/run ceiling — deliberately tighter than the CLI default (API-credit protection; see body_params.body_int P14-3 note)
    DESIGN_CHAT_MAX_TOOL_ITERATIONS: int = 500  # design_chat_loop.py + design_chat.py
    DESIGN_CHAT_LLM_MAX_RETRIES: int = 2        # design_chat_loop.py outer retries on transient LLM errors (on top of the client's own)


@dataclass(frozen=True)
class CompressionConfig:
    """Context compression tuning thresholds for SessionCompressionContext."""

    MIN_RECENT_TURNS_KEEP: int = 4    # Always keep the most recent N turns as original text
    COMPRESS_BATCH_MIN: int = 11      # Compress when this many new turns accumulate beyond recent_keep
    # In /general chat mode the periodic turn-count compression (above) is disabled:
    # turns accumulate verbatim so the stable prefix — and its prompt cache — survives
    # across many turns. Compression (summarize) fires only once the LIVE context window
    # reaches this occupancy fraction, preempting the lossy hard-cap front-trim backstop.
    # Kept comfortably below 1.0 so the summarize path always wins over the overflow trim.
    GENERAL_MODE_COMPRESS_OCCUPANCY: float = 0.80
    # Minimum compressible turns required even on the force path (occupancy-gated
    # /general compression). Without this, when the recent window itself is large
    # enough to keep occupancy ≥ GENERAL_MODE_COMPRESS_OCCUPANCY after a compress,
    # every subsequent turn would trigger an LLM summarize call for a single turn
    # (the compress-lock blocks concurrency, not re-firing). 3 is small enough to
    # fire well before the hard-cap front-trim, but large enough to avoid per-turn
    # summarize thrash.
    FORCE_COMPRESS_MIN_TURNS: int = 3


@dataclass(frozen=True)
class DisplayConfig:
    """CLI progress display (progress/diff) configuration.

    Values use these defaults as the single source of truth, but runtime toggling
    and tuning can be overridden via environment variables (evaluated once at process
    start). When env vars are unset, the defaults below are used.
    """

    # Whether to auto-display the full file diff ("changes" block) after successful
    # execution. Default off — use /diff when needed.
    #enable: ASICODE_RUN_DIFF=1 (or on/true/yes)
    RUN_DIFF: bool = field(
        default_factory=lambda: _env_flag("ASICODE_RUN_DIFF", False)
    )
    # Whether to generate a one-line "next task" suggestion via a helper model after
    # turn end, shown as ghost text in the empty prompt. Adds one LLM call per turn.
    #disable: ASICODE_NEXT_SUGGEST=0 (or off/false/no)
    NEXT_SUGGEST: bool = field(
        default_factory=lambda: _env_flag("ASICODE_NEXT_SUGGEST", True)
    )


@dataclass(frozen=True)
class ThresholdConfig:
    tokens: TokenLimits = field(default_factory=TokenLimits)
    lines: LineLimits = field(default_factory=LineLimits)
    scores: ScoreThresholds = field(default_factory=ScoreThresholds)
    counts: CountLimits = field(default_factory=CountLimits)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


# Single global instance — all consumers import this.
config = ThresholdConfig()
