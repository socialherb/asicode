
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

Some domain policy modules keep their own constants (`termination_policy.py`,
`execution_policy.py`, `learned_policy.py`, `task_quality.py`,
`alignment_scorer.py`, `self_planning_policy.py`, `weight_learning.py`) because
the values are tightly coupled to that module's algorithm. They remain
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


def _env_int(name: str, default: int) -> int:
    """Parse a positive-int env var; fallback to default on empty/invalid/non-positive."""
    try:
        v = int((os.getenv(name, "") or "").strip() or str(default))
        return v if v > 0 else default
    except Exception:
        return default


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
    INTENT_RESOLVER_DEFAULT: int = 4096
    SERVICE_DEFAULT: int = 4096
    SERVICE_REPAIR: int = 8192
    PAIR_EVALUATOR_MAX_TOKENS: int = 4096
    PLANNER_AGENT_DEFAULT: int = 16384
    PLANNER_SUMMARY: int = 4096
    PLANNER_SUMMARY_REPAIR: int = 8192
    SUBAGENT_SHORT: int = 2000
    LOCAL_ASSISTANT_DEFAULT: int = 4096
    LOCAL_ASSISTANT_SHORT: int = 512
    LOCAL_MODEL_CONTEXT_CHARS: int = 4000
    INTELLIGENT_SERVICE_DEFAULT: int = 4096
    META_STRATEGY: int = 4096
    EXPLORATION: int = 4096
    AGENT_STREAM: int = 4096
    ANTHROPIC_DEFAULT: int = 65536

    PLANNER_AGENT_INPUT_SCALE_THRESHOLD: int = 10000  # planner_agent.py: input token estimate → scaled output switch
    PLANNER_AGENT_SCALED_OUTPUT: int = 65536           # planner_agent.py: output max_tokens when input is large
                                                      # Increased from 32768 because the old cap made retry
                                                      # doubling useless (capped at same value), and modern
                                                      # LLMs support 64K+ output tokens.

    AGENT_TOOL_CALL: int = 32768  # agent_loop.py _llm_call_with_tools default max_tokens

    CONTEXT_HARD_CAP_SAFETY_MARGIN: int = 1024  # agent_loop.py: pre-flight input token guard
                                                 # subtracted from model's context_limit before
                                                 # triggering _preemptive_trim. Prevents
                                                 # HTTP 400 "max context length exceeded" errors
                                                 # from API providers (DeepSeek 1M, etc.).

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

    PRIOR_REF_LINES_PER_FILE: int = 200   # per-file cap for prior-read reference context
    PRIOR_REF_TOTAL_LINES: int = 500      # total cap across all prior-read files
    PAIR_EVALUATOR_SYM_MIN_LINES: int = 80
    PAIR_EVALUATOR_SYM_FACTOR: float = 3.0
    PAIR_EVALUATOR_SYM_ABSOLUTE_MAX: int = 5000
    DESIGN_TURN_MAX_CHARS: int = 100000
    SYM_BUDGET_CHARS: int = 6_000
    TOTAL_BUDGET_CHARS: int = 50_000
    RAG_FILE_CHARS: int = 200_000
    STRATEGY_LIGHT_LINES: int = 150
    STRATEGY_LIGHT_BYTES: int = 8_000
    STRATEGY_LIGHT_TOKENS: int = 2_000
    STRATEGY_CHARS_PER_TOKEN: float = 3.5
    UI_FULL_MAX_LINES: int = 1000

    # Budget for RTP-lite lightweight file preview.
    # 8K → 24K: enough for small-to-medium projects (2-4 files ~20K chars)
    # without losing the "lite" character versus full RTP (30K default).
    # Prevents mid-size files (e.g. 8599-char client.js) from being dropped.
    PLANNER_RTP_LITE_CHARS: int = 24_000

    # Budget for replan anchor-miss file content injection.
    # Same rationale as RTP-lite: anchor hallucination recovery needs
    # enough file content for the LLM to pick a real anchor.
    PLANNER_REPLAN_ANCHOR_CHARS: int = 24_000
    SUMMARIZE_ANALYSIS_CHARS: int = 500_000
    SUMMARIZE_STRUCTURAL_CHARS: int = 80_000


@dataclass(frozen=True)
class ScoreThresholds:
    """Confidence/similarity/score gates."""

    AUTO_CORRECT: float = 0.65
    HINT_ONLY: float = 0.40
    SIM_WEIGHT_JACCARD: float = 0.45
    SIM_WEIGHT_PREFIX: float = 0.20
    SIM_WEIGHT_EDIT: float = 0.35
    STRICT_TOP1: float = 0.30
    STRICT_MARGIN: float = 0.20
    STRICT_TARGETED_TOP1: float = 0.35
    STRICT_TARGETED_MARGIN: float = 0.25
    GUIDED_TOP1: float = 0.15
    MIN_EDGE_CONFIDENCE: float = 0.50
    MIN_REPAIR_CONFIDENCE: float = 0.65
    PHASE_SKIP_COVERAGE_HIGH: float = 0.95
    PHASE_SKIP_COVERAGE_MED: float = 0.85
    UNKNOWN_SYMBOL_ABORT_RATIO: float = 0.50
    PLANNER_MIN_GROUNDING_CONF: float = 0.30
    EXPLORE_FIRST_THRESHOLD: float = 0.85

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

    # ── Phase / Execution ────────────────────────────────────────────────
    PHASE_RECONSTRUCTION_SKIP_COVERAGE: float = 0.95  # phase_orchestrator.py reconstruction skip gate
    REPAIR_IMPROVEMENT_RATIO_LOW: float = 0.30   # repair_engine.py low-improvement hint
    REPAIR_IMPROVEMENT_RATIO_HIGH: float = 0.70  # repair_engine.py high-improvement hint

    # ── Rename / Overlap Detection ───────────────────────────────────────
    RENAME_SIMILARITY_RATIO: float = 0.50   # executor_signal_utils.py rename char-ratio gate
    RENAME_BEST_SCORE_THRESHOLD: float = 1.0  # executor_signal_utils.py rename score gate

    # ── Distillation / Learning ────────────────────────────────────────────
    DISTILL_THRESHOLD_SPARSE: float = 0.90   # context_utils.py: sample_count < 10
    DISTILL_THRESHOLD_MODERATE: float = 0.82 # context_utils.py: sample_count < 30
    DISTILL_THRESHOLD_CONFIDENT: float = 0.75# context_utils.py: sample_count >= 30
    DISTILL_SPARSE_LIMIT: int = 10           # context_utils.py: sparse data bound
    DISTILL_MODERATE_LIMIT: int = 30         # context_utils.py: moderate data bound


@dataclass(frozen=True)
class WeightConfig:
    """Hand-tuned scoring weights across planner modules.

    These weights are heuristics — not empirically calibrated. Each weight is
    annotated with its usage site and design rationale. When adjusting, prefer
    small deltas (<=0.05) and log the change rationale in the commit message.
    """

    # ── planner_policy_adapter.py: _synthesize_weights() ─────────────────
    # final = learned_policy*LEARNED_POLICY + strategy_policy*STRATEGY_POLICY
    #         + reward_ema*REWARD_EMA + execution_bias*EXECUTION_BIAS
    LEARNED_POLICY: float = 0.30       # learned Q-policy (strongest signal)
    STRATEGY_POLICY: float = 0.20      # strategy policy score
    REWARD_EMA: float = 0.20           # exponential-moving-average reward
    EXECUTION_BIAS: float = 0.15       # execution history bias

    # ── multi_strategy_planner.py: plan_score formula ────────────────────
    # plan_score = strategy*MULTI_STRATEGY + contract_rate*MULTI_CONTRACT
    #              - complexity*MULTI_COMPLEXITY - graph_impact*MULTI_IMPACT
    MULTI_STRATEGY: float = 0.65       # strategy simulator (primary signal)
    MULTI_CONTRACT: float = 0.25       # contract-preserving intent quality
    MULTI_COMPLEXITY: float = 0.10     # complexity penalty per excess op
    MULTI_IMPACT: float = 0.10         # graph blast-radius penalty
    MULTI_COMPLEXITY_BASELINE: int = 6   # ops beyond this count as excess
    MULTI_COMPLEXITY_PENALTY_PER_OP: float = 0.01

    # ── planner_agent.py: memory-aware strategy ordering ─────────────────
    # preference = success_rate
    #              - PREF_ROLLBACK_PENALTY*rollback_rate
    #              - PREF_REPAIR_PENALTY*repair_rate
    PREF_ROLLBACK_PENALTY: float = 0.30
    PREF_REPAIR_PENALTY: float = 0.20
    PREF_EXPLORATION_UPLIFT: float = 0.10  # added when selected_count < threshold
    PREF_EXPLORE_THRESHOLD: int = 3
    PREF_WEAK_DEPRIORITIZE: float = 0.20
    PREF_WEAK_MIN_RUNS: int = 5
    PREF_WEAK_SUCCESS_THRESHOLD: float = 0.25
    PREF_WEAK_ROLLBACK_THRESHOLD: float = 0.50
    PREF_GRAPH_SYMBOL_BOOST: float = 0.15
    PREF_GRAPH_SYMBOL_PENALTY: float = 0.10
    PREF_EARLY_SHUFFLE_THRESHOLD: int = 5
    PREF_NEUTRAL_SCORE: float = 0.0



@dataclass(frozen=True)
class CountLimits:
    """Hardcoded upper bounds on iteration / sample / fan-out counts."""

    REF_FILES_PRELOAD: int = 10
    DPB_SAMPLE_SYMBOLS: int = 6
    MAX_BRIDGE_LINES: int = 20
    LOCALIZED_EDIT_LINES: int = 50
    PLANNER_MAX_BODIES: int = 5
    PLANNER_MAX_STEPS: int = 1
    PLANNER_LARGE_SYMBOL_LINES: int = 80
    SEMANTIC_REFINER_EXPAND: int = 2
    SEMANTIC_VERIFIER_MAX_CALLERS: int = 10
    SEMANTIC_VERIFIER_MAX_ISSUES: int = 8
    AGENT_NO_TOOL_NUDGE_MAX: int = 3
    AGENT_NO_PROGRESS_THRESHOLD: int = 5
    AGENT_FAIL_LOOP_LARGE: int = 3
    SYMBOL_MAX_PY_FILES: int = 600
    SYMBOL_MAX_TS_FILES: int = 300
    RAG_MAX_FILES: int = 600
    AST_CACHE_MAX: int = 16
    ROUTING_POLICY_CACHE_TTL_S: float = 300.0
    RUN_STORE_TOP_K: int = 3
    RUN_STORE_MAX_DYNAMIC: int = 10
    PHASE_MAX_STORE_LOAD_BYTES: int = 20 * 1024 * 1024
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
    AGENT_CTX_BUDGET_TIME_S: float = 3.0
    PAIR_EVALUATOR_MAX_PAIRS: int = 8
    VULTURE_HUB_IMPORTER_THRESHOLD: int = 5  # arbitrary — no empirical basis yet; revisit after shadow log data accumulates

    # Callee / caller source injection budgets in operation_executor.py.
    # Lines injected into developer LLM context per modify/insert call.
    # Keep generous — truncating callee bodies mid-way gives the LLM worse
    # information than omitting them entirely and costs extra repair rounds.
    CALLEE_SOURCE_LINE_BUDGET: int = 2000   # total across all injected callees
    CALLEE_SOURCE_MAX_COUNT: int = 8        # top-N callees to consider
    CALLER_SOURCE_LINE_BUDGET: int = 300    # total across all injected callers
    CALLER_SOURCE_MAX_COUNT: int = 4        # top-N callers to consider

    # Repair engine hint budget — adaptive: max(min_lines, len(lines) * factor)
    REPAIR_SUMMARY_MIN_LINES: int = 12
    REPAIR_SUMMARY_ADAPTIVE_FACTOR: float = 0.5

    # Planner context limits — adaptive: max(PLANNER_MAX_SYMBOLS_MIN,
    #   min(PLANNER_MAX_SYMBOLS_TOTAL, len(file_paths) * PLANNER_SYMBOLS_PER_FILE))
    PLANNER_MAX_SYMBOLS_MIN: int = 20
    PLANNER_SYMBOLS_PER_FILE: int = 15
    PLANNER_MAX_SYMBOLS_TOTAL: int = 200

    # Scanner max_per_file defaults — prevents silent truncation from hiding issues.
    # Values are conservative (5-10) to avoid overwhelming callers with noise, but
    # each scanner logs a warning when the cap is hit so the caller can detect
    # incomplete results and widen the limit or re-scan with narrower scope.
    SCANNER_DEAD_BLOCK_MAX: int = 5
    SCANNER_PUBLIC_DEAD_BLOCK_MAX: int = 5
    SCANNER_VULTURE_MAX: int = 10
    SCANNER_VULTURE_MIN_CONFIDENCE: int = 60  # 0–100 raw Vulture confidence floor
    SCANNER_DUP_DEF_MAX: int = 10
    SCANNER_UNUSED_IMPORT_MAX: int = 10
    SCANNER_CONTAINER_REACH_MAX: int = 5
    SCANNER_CONTRADICTORY_MAX: int = 10
    SCANNER_CONTRADICTORY_DUP_DISTANCE: int = 100

    # ── Symbol Search / Tool Loop ────────────────────────────────────────
    SEARCH_RESULTS_CAP: int = 30             # symbol_search.py max results before early break
    AGENT_TOOL_RETRY_LIMIT: int = 5          # agent_loop.py per-tool cumulative exhaustion warning
    AGENT_MAX_TURNS_DEFAULT: int = 500         # tool_registry + agent_stream + asi
    DESIGN_CHAT_MAX_TOOL_ITERATIONS: int = 500  # design_chat_loop.py + design_chat.py
    DESIGN_CHAT_LLM_MAX_RETRIES: int = 2        # design_chat_loop.py outer retries on transient LLM errors (on top of the client's own)

    # ── Plan / Contract ──────────────────────────────────────────────────
    PLAN_CREATE_OPS_MIN_FOR_WIRING: int = 2  # contract_driven_planning.py missing-route wiring check
    PLAN_OPS_MIN_FOR_REORDER: int = 2        # contract_driven_planning.py topo sort gate (len > 1)
    MAX_CONTRACT_LINES_SKIP: int = 3         # contract_driven_planning.py short contract skip

    # ── Runtime Gate ─────────────────────────────────────────────────────
    RUNTIME_MIN_TOTAL_FOR_CHECK: int = 1     # runtime_gate.py check skip threshold (total <= 1)
    CONTEXT_BUCKET_MULTI_FILE_COUNT: int = 3 # repair_engine.py multi-file vs single-file bucket
    PHASE_INCOMPLETE_GAP_COUNT: int = 2      # phase_orchestrator.py min missing primitives to run F.3/G/G.1

    # ── Repair / PDG / Design ────────────────────────────────────────────
    PDG_INCLUDE_MUTATION_LINES: int = 300    # pdg_lite.py mutation inclusion gate
    PDG_VERY_LARGE_FUNC_LINES: int = 800     # pdg_lite.py depth cap gate

    # ── Execution / Replan ────────────────────────────────────────────────
    EXECUTION_MAX_REPLAN_COUNT: int = 2
    EXECUTION_MAX_REPLAN_OP_RATIO: float = 2.0
    EXECUTION_MEDIUM_FUNC_SURGICAL_EDIT: int = 40
    EXECUTION_MAX_DELEGATION_COUNT: int = 1
    EXECUTION_MAX_ALIGNMENT_RETRIES: int = 2


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

    # Whether to show change diffs inline for each successful op during execution.
    #disable: ASICODE_INLINE_OP_DIFF=0 (or off/false/no)
    #   Enable: ASICODE_INLINE_OP_DIFF=1  (default True)
    INLINE_OP_DIFF: bool = field(
        default_factory=lambda: _env_flag("ASICODE_INLINE_OP_DIFF", True)
    )
    # Max lines per op for inline diff display (excess truncated to "… N more lines").
    #adjust: ASICODE_INLINE_OP_DIFF_MAX_LINES=80
    INLINE_OP_DIFF_MAX_LINES: int = field(
        default_factory=lambda: _env_int("ASICODE_INLINE_OP_DIFF_MAX_LINES", 40)
    )
    # Maximum char length of diff string delivered via operation_complete event patch_preview.
    #adjust: ASICODE_INLINE_OP_DIFF_MAX_CHARS=8000
    INLINE_OP_DIFF_MAX_CHARS: int = field(
        default_factory=lambda: _env_int("ASICODE_INLINE_OP_DIFF_MAX_CHARS", 4000)
    )
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
    weights: WeightConfig = field(default_factory=WeightConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


# Single global instance — all consumers import this.
config = ThresholdConfig()
