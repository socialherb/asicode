"""
TriggerPolicy — rule-based decision engine.

Maps TriggerEvent → ActionDecision based on:
  - Event kind and severity
  - Model tier (none / small / strong)
  - Per-file and per-kind cooldowns (dedup / rate limit)
  - Global AUTO_FIX rate limit (per hour)

Model tier scaling:
  "none"   → NOTIFY + ESCALATE only     (0 LLM calls)
  "small"  → + SUGGEST                  (bounded prompts, small context)
  "strong" → + AUTO_FIX                 (full PLANNER lane)

AUTO_FIX is automatically downgraded to SUGGEST when model_tier < required tier.
SUGGEST is automatically downgraded to NOTIFY when model_tier == "none".
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from external_llm.editor.agent.autonomous.trigger_engine import TriggerEvent, TriggerKind

# Module-level mapping: TriggerKind → feature name (used by TriggerPolicy._evaluate)
_KIND_TO_FEATURE: dict = {
    TriggerKind.FILE_MODIFIED: "file_review",
    TriggerKind.TEST_FAILED: "test_analysis",
    TriggerKind.TEST_RECOVERED: "test_recovery",
    TriggerKind.AGENT_STALL: "agent_stall",
    TriggerKind.AGENT_COMPLETED: "agent_status",
    TriggerKind.AGENT_FAILED: "agent_status",
    TriggerKind.IMPORT_ERROR: "import_repair",
    TriggerKind.INTEGRATION_MISSING: "integration_repair",
}


class ActionKind(Enum):
    IGNORE = "ignore"
    NOTIFY = "notify"  # Template message, no LLM
    SUGGEST = "suggest"  # 1-2 sentence analysis, small LLM sufficient
    AUTO_FIX = "auto_fix"  # Full PLANNER lane — model quality determines result quality
    ESCALATE = "escalate"  # Immediate user notification + optional approval


@dataclass
class ActionDecision:
    kind: ActionKind
    message: str = ""  # Human-readable message (NOTIFY / ESCALATE)
    prompt: str = ""  # LLM request text (SUGGEST / AUTO_FIX)
    priority: int = 2  # 0=critical  1=high  2=normal
    requires_model_tier: str = "none"  # "none" | "small" | "strong"


class TriggerPolicy:
    """Thread-safe rule-based policy: TriggerEvent → ActionDecision."""

    # Per-file cooldown: same file won't trigger again within this window
    _FILE_COOLDOWN = 30.0

    # All valid feature names (for validation / default)
    _ALL_FEATURES: frozenset = frozenset(
        {
            "file_review",
            "test_analysis",
            "test_recovery",
            "agent_stall",
            "agent_status",
            "import_repair",
            "integration_repair",
        }
    )

    # Per-kind cooldown (seconds between same-kind events)
    _KIND_COOLDOWN: ClassVar[dict[str, float]] = {
        "file_modified": 15.0,
        "test_failed": 10.0,
        "test_recovered": 10.0,
        "agent_stall": 5.0,
        "agent_completed": 2.0,
        "agent_failed": 5.0,
        "import_error": 5.0,
        "integration_missing": 30.0,
        "schedule": 0.0,  # scheduler manages its own interval
    }

    # AUTO_FIX rate limit: max N per hour
    # Allowed model tiers — anything else fails CLOSED to "none" (P0-1)
    _MODEL_TIERS: ClassVar[frozenset] = frozenset({"none", "small", "strong"})

    # Longest cooldown window across file/kind gates — entries older than this
    # can never gate a decision again and are pruned at stamp time (P0-5)
    _MAX_COOLDOWN: ClassVar[float] = max(_FILE_COOLDOWN, max(_KIND_COOLDOWN.values()))

    _AUTO_FIX_PER_HOUR = 10

    def __init__(self, model_tier: str = "small", enabled_features: set | None = None):
        self._lock = threading.Lock()
        # model_tier is a normalized property: unknown tiers fail CLOSED to
        # "none" (P0-1) — the setter runs under _lock, so create it first.
        self.model_tier = model_tier
        # Per-feature on/off — default all enabled
        self.enabled_features: set = set(enabled_features) if enabled_features is not None else set(self._ALL_FEATURES)
        self._file_last: dict[tuple[str, str], float] = {}  # (source_file, kind) → last emit
        self._kind_last: dict[str, float] = {}  # kind → last emit
        self._auto_fix_ts: list[float] = []  # recent AUTO_FIX decision times

    @property
    def model_tier(self) -> str:
        """Normalized model tier — "none" | "small" | "strong" (fail-closed)."""
        return self._model_tier

    @model_tier.setter
    def model_tier(self, value: str) -> None:
        tier = str(value).strip().lower() if value else "none"
        normalized = tier if tier in self._MODEL_TIERS else "none"
        with self._lock:
            self._model_tier = normalized

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, event: TriggerEvent) -> ActionDecision:
        """Evaluate event and return action. Thread-safe."""
        with self._lock:
            return self._evaluate(event)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate(self, event: TriggerEvent) -> ActionDecision:
        now = time.monotonic()
        kind_str = event.kind.value

        # Feature gate — check if this kind's feature is enabled
        _feature = _KIND_TO_FEATURE.get(event.kind)
        if _feature and _feature not in self.enabled_features:
            return ActionDecision(kind=ActionKind.IGNORE)

        # Per-file cooldown — keyed by (source_file, kind) so a save followed by
        # an import error / test failure (the primary workflow) is NOT swallowed
        # by the earlier FILE_MODIFIED stamp (P0-3). `None` sentinel: monotonic's
        # origin is boot time, so a 0 sentinel would drop the first event on a
        # freshly booted host (P0-2).
        if event.source_file:
            _file_last = self._file_last.get((event.source_file, kind_str))
            if _file_last is not None and now - _file_last < self._FILE_COOLDOWN:
                return ActionDecision(kind=ActionKind.IGNORE)

        # Per-kind cooldown
        cooldown = self._KIND_COOLDOWN.get(kind_str, 10.0)
        _kind_last = self._kind_last.get(kind_str)
        if _kind_last is not None and now - _kind_last < cooldown:
            return ActionDecision(kind=ActionKind.IGNORE)

        # Route
        decision = self._route(event)

        # Downgrade AUTO_FIX if model tier insufficient
        if decision.kind == ActionKind.AUTO_FIX:
            if self.model_tier == "none":
                decision = ActionDecision(
                    kind=ActionKind.NOTIFY,
                    message=f"[Auto-fix available] {decision.message} (no model configured, notify only)",
                    priority=decision.priority,
                )
            elif self.model_tier == "small" and decision.requires_model_tier == "strong":
                decision = ActionDecision(
                    kind=ActionKind.SUGGEST,
                    message=decision.message,
                    prompt=decision.prompt,
                    priority=decision.priority,
                    requires_model_tier="small",
                )

        # Downgrade SUGGEST if no model
        elif decision.kind == ActionKind.SUGGEST and self.model_tier == "none":
            decision = ActionDecision(
                kind=ActionKind.NOTIFY,
                message=decision.message or f"[Proactive] {kind_str}",
                priority=decision.priority,
            )

        # AUTO_FIX rate limit
        if decision.kind == ActionKind.AUTO_FIX:
            self._auto_fix_ts = [t for t in self._auto_fix_ts if now - t < 3600]
            if len(self._auto_fix_ts) >= self._AUTO_FIX_PER_HOUR:
                # Rate-limited NOTIFY still consumes cooldowns (P0-4): without
                # the stamp every post-cap event re-enters this branch and emits
                # another NOTIFY — a per-event notification storm.
                self._stamp_cooldowns(event, kind_str, now)
                return ActionDecision(
                    kind=ActionKind.NOTIFY,
                    message="[Rate limited] AUTO_FIX per-hour cap exceeded. Will retry next hour.",
                    priority=2,
                )
            self._auto_fix_ts.append(now)

        # Update last-trigger timestamps AFTER all checks (rate-limited events
        # now DO consume cooldowns — see above) and prune entries that can no
        # longer gate any decision (P0-5: bounded memory for daemon lifetime).
        self._stamp_cooldowns(event, kind_str, now)

        return decision

    def _stamp_cooldowns(self, event: TriggerEvent, kind_str: str, now: float) -> None:
        """Record emission time and prune entries older than the longest window.

        Called for every emitted decision — including the rate-limited NOTIFY —
        so post-cap events are throttled to one NOTIFY per cooldown window, and
        the two timestamp dicts stay bounded (keys are file paths / kinds and
        the daemon lives for the whole session).
        """
        if event.source_file:
            self._file_last[(event.source_file, kind_str)] = now
        self._kind_last[kind_str] = now
        cutoff = now - self._MAX_COOLDOWN
        self._file_last = {k: v for k, v in self._file_last.items() if v >= cutoff}
        self._kind_last = {k: v for k, v in self._kind_last.items() if v >= cutoff}

    def _route(self, event: TriggerEvent) -> ActionDecision:
        k = event.kind

        # ── FILE_MODIFIED ─────────────────────────────────────────────────────
        if k == TriggerKind.FILE_MODIFIED:
            path = event.source_file or ""
            fname = path.rsplit("/", 1)[-1]
            return ActionDecision(
                kind=ActionKind.SUGGEST,
                message=f"`{fname}` was modified.",
                prompt=(
                    f"File `{path}` was just saved.\n"
                    "Quickly scan for immediate issues (import errors, obvious logic bugs, "
                    "integration gaps with existing system) and report in 1-2 sentences. "
                    "If nothing is wrong, reply 'No issues.'"
                ),
                priority=2,
                requires_model_tier="small",
            )

        # ── TEST_FAILED ───────────────────────────────────────────────────────
        if k == TriggerKind.TEST_FAILED:
            failing = event.metadata.get("failing_tests", [])
            first_tb = event.metadata.get("first_traceback", "")
            summary = event.metadata.get("summary_line", "")
            return ActionDecision(
                kind=ActionKind.SUGGEST,
                message=f"{len(failing)} test(s) failed: {', '.join(str(t) for t in failing[:3])}",
                prompt=(
                    f"pytest failure:\n"
                    f"Summary: {summary}\n"
                    f"Failed tests: {failing}\n\n"
                    f"Traceback:\n{first_tb[:1500]}\n\n"
                    "Summarize the cause and fix direction in 2-3 sentences."
                ),
                priority=1,
                requires_model_tier="small",
            )

        # ── TEST_RECOVERED ────────────────────────────────────────────────────
        if k == TriggerKind.TEST_RECOVERED:
            return ActionDecision(
                kind=ActionKind.NOTIFY,
                message="Tests are passing again.",
                priority=2,
            )

        # ── AGENT_STALL ───────────────────────────────────────────────────────
        if k == TriggerKind.AGENT_STALL:
            tool = event.metadata.get("tool", "unknown")
            streak = event.metadata.get("streak", 0)
            turn = event.metadata.get("turn", 0)
            return ActionDecision(
                kind=ActionKind.ESCALATE,
                message=(
                    f"Agent failed `{tool}` {streak} consecutive times at turn {turn}.\n"
                    "Try a different strategy or abort?"
                ),
                priority=0,
            )

        # ── AGENT_FAILED ──────────────────────────────────────────────────────
        if k == TriggerKind.AGENT_FAILED:
            reasons = event.metadata.get("blocking_reasons", [])
            error = event.metadata.get("error", "")
            msg = ", ".join(str(r) for r in reasons[:2]) if reasons else str(error)[:100]
            return ActionDecision(
                kind=ActionKind.NOTIFY,
                message=f"Agent execution failed: {msg}",
                priority=1,
            )

        # ── AGENT_COMPLETED ───────────────────────────────────────────────────
        if k == TriggerKind.AGENT_COMPLETED:
            turns = event.metadata.get("turns", 0)
            status = event.metadata.get("status", "success")
            return ActionDecision(
                kind=ActionKind.NOTIFY,
                message=f"Agent completed ({status}, {turns} turns)",
                priority=2,
            )

        # ── IMPORT_ERROR ──────────────────────────────────────────────────────
        if k == TriggerKind.IMPORT_ERROR:
            path = event.source_file or ""
            err = event.metadata.get("error", "")
            return ActionDecision(
                kind=ActionKind.SUGGEST,
                message=f"`{path.rsplit('/', 1)[-1]}` import error: {str(err)[:80]}",
                prompt=(
                    f"Import/syntax error in file `{path}`:\n{err}\n\n"
                    "Identify the cause and fix it with minimal changes."
                ),
                priority=1,
                requires_model_tier="small",
            )

        # ── INTEGRATION_MISSING ───────────────────────────────────────────────
        if k == TriggerKind.INTEGRATION_MISSING:
            missing = event.metadata.get("missing_imports", [])
            return ActionDecision(
                kind=ActionKind.AUTO_FIX,
                message=f"Integration missing detected: {missing}",
                prompt=(
                    f"The following files were created but not connected to the existing system: {missing}\n"
                    "Find appropriate entry points (dispatch, gate_check, run_tool, etc.) "
                    "and add import and call code."
                ),
                priority=1,
                requires_model_tier="strong",
            )

        # ── SCHEDULE ──────────────────────────────────────────────────────────
        if k == TriggerKind.SCHEDULE:
            label = event.metadata.get("label", "scheduled")
            return ActionDecision(
                kind=ActionKind.AUTO_FIX,
                message=f"[{label}] Scheduled task execution",
                prompt=f"Scheduled task '{label}': check repository state and fix obvious issues.",
                priority=2,
                requires_model_tier="strong",
            )

        return ActionDecision(kind=ActionKind.IGNORE)
