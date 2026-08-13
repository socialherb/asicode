"""
Agent Loop Types — all dataclasses used by agent_loop.py and its mixins.

Extracted from agent_loop.py to resolve import ordering and avoid 9000-line monolith.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ── Write-tool name SSOT ────────────────────────────────────────────────────
# Single source of truth for the set of tool names that modify files. Both
# ``ToolRegistry._WRITE_TOOLS`` (file-locking, failure-logging, cache
# invalidation) and ``TurnContext.write_tools`` (reads_since_last_edit reset,
# write_tool_used detection, test-impact invalidation) derive from this.
# Adding a write tool here keeps all six mechanisms in lockstep; forgetting
# one lets edits via it silently bypass them. This module is a leaf
# (stdlib-only imports), so ``tool_registry`` can import it without a cycle.
WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "apply_patch", "write_plan", "edit_ast", "edit_file",
    "edit_text", "modify_symbol", "anchor_edit",
})


class AgentCancelled(Exception):  # noqa: N818 — Cancelled-suffix convention (asyncio.CancelledError parity)
    """Exception raised when agent execution is cancelled."""


@dataclass
class AgentTurn:
    turn_num: int
    tool_name: Optional[str]
    tool_args: dict[str, Any]
    tool_result: Any  # ToolResult (lazy import to avoid circular deps)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    status: str  # "success", "max_turns", "error", "cancelled", "clarification_needed", "text_reply"
    turns: list[AgentTurn] = field(default_factory=list)
    final_message: str = ""
    applied_patches: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FinalAnswerOutcome:
    """Result of _handle_final_answer_turn().

    If nudge_message is set → append it to messages, update nudge_count, and continue the turn loop.
    Otherwise → return result directly.
    """
    result: Optional[AgentResult] = None
    nudge_message: Optional[Any] = None
    nudge_count: int = 0


@dataclass
class _ToolTurnOutcome:
    """Result of _execute_and_process_tool_calls() for one turn."""
    new_messages: list
    prepared_calls: list
    write_tool_used: bool
    any_tool_called: bool
    fail_streak: dict
    reads_since_last_edit: int
    plan_current_index: int
    early_return: Optional[AgentResult] = None
    should_continue: bool = False
    phase_rule_messages: list = field(default_factory=list)
    noop_confirmed: bool = False


@dataclass
class _TurnPrepResult:
    """Result of _prepare_turn_messages()."""
    messages: list
    budget_warned: bool
    goal_reminder_injected: int
    search_first_hint_done: bool
    reads_since_last_edit: int


@dataclass
class _PostToolResult:
    """Result of _process_post_tool_turn()."""
    messages: list
    tdd_fail_count: int
    tdd_total_runs: int
    tdd_total_pass: int
    early_return: Optional[AgentResult] = None


@dataclass
class _ResultsProcessingOutcome:
    """Result of _process_tool_results()."""
    new_messages: list
    write_tool_used: bool
    reads_since_last_edit: int
    noop_confirmed: bool
    fail_streak: dict
    early_return: Optional[AgentResult] = None


@dataclass
class _PreparedCallsResult:
    """Result of _build_and_filter_prepared_calls()."""
    prepared_calls: list
    phase_rule_messages: list
    plan_current_index: int
    should_continue: bool = False


@dataclass
class TurnContext:
    """Consolidated params and mutable state for the LLM turn loop.

    Replaces 14+ individual parameters across _run_llm_loop and its
    5 sub-methods. Fixed config is set once on creation; mutable
    state is updated throughout the loop.
    """
    # Fixed config (from run())
    request: str
    context: str
    route: Any
    git_state: Any
    session_id: str
    is_local_model: bool
    has_native_tools: bool
    read_only_request: bool
    known_target_file: str
    target_keywords: list[str]
    tier: Any
    plan: Optional[dict[str, Any]]
    plan_subtasks: list[dict[str, Any]]

    # Mutable loop state
    turn_num: int = 0
    turns: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    ephemeral_pending: list = field(default_factory=list)
    tdd_fail_count: int = 0
    tdd_total_runs: int = 0
    tdd_total_pass: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    last_call_prompt_tokens: int = 0
    last_call_completion_tokens: int = 0
    provider_name: str = ""
    model_name: str = ""
    write_tool_used: bool = False
    any_tool_called: bool = False
    budget_warned: bool = False
    fail_streak: dict = field(default_factory=dict)
    # Per-run session identity for failure-recall dedup (failure_pattern_store).
    # Generated once per run via new_session_key() — NOT str(id(fail_streak)),
    # which reuses freed object addresses and collapses distinct runs onto one
    # key, silencing [RECALL] for the process lifetime.
    recall_session_key: str = ""
    noop_confirmed: bool = False
    # True when IntentResolver never produced a classification (LLM failure,
    # unparseable response, no IntentResult attached). ``read_only_request``
    # still defaults to False in that case so a legitimate edit is never
    # blocked — but the "write intent finished with no patch" gate must not
    # fire on a premise nothing established. See intent_is_undetermined().
    intent_undetermined: bool = False
    no_tool_nudge_count: int = 0
    search_first_hint_done: bool = False
    reads_since_last_edit: int = 0
    goal_reminder_injected: int = 0
    rollback_performed: bool = False
    rollback_result: Any = None
    plan_current_index: int = 0
    # Drives reads_since_last_edit reset (GOAL REMINDER guard), write_tool_used
    # detection (read-only early-finish guard), and write-time test-impact index
    # invalidation. Derived from the ``WRITE_TOOL_NAMES`` SSOT in this module so
    # it never drifts from ``ToolRegistry._WRITE_TOOLS``; a fresh copy per
    # instance keeps turn contexts independently mutable.
    write_tools: set = field(default_factory=lambda: set(WRITE_TOOL_NAMES))
