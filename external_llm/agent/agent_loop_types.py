"""
Agent Loop Types — all dataclasses used by agent_loop.py and its mixins.

Extracted from agent_loop.py to resolve import ordering and avoid 9000-line monolith.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


class AgentCancelled(Exception):
    """Exception raised when agent execution is cancelled."""
    pass


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
class _PlannerLaneOutcome:
    """Result of _run_planner_lane().

    result=non-None  → caller returns it directly.
    result=None      → PLANNER fell through; use fallback_context if set.
    """
    result: Optional[AgentResult] = None
    fallback_context: Optional[str] = None


@dataclass
class _EscalationOutcome:
    """Result of _handle_analyze_first_escalation()."""
    exec_result: dict
    op_plan: Any
    spec: Any
    completed: int
    failed: int
    exec_status: str
    exec_detail: str
    early_outcome: Optional[_PlannerLaneOutcome] = None


@dataclass
class _SpecResolutionResult:
    """Spec resolution result consumed by _build_and_execute_plan().

    Built in PlannerPipelineMixin._run_planner_lane from the prebuilt spec
    (Design Chat analysis). fit_verdict.action == CLARIFY → caller handles
    early return. Escalation state fields are forwarded to
    _build_and_execute_plan().
    """
    spec: Optional[Any] = None
    fit_verdict: Optional[Any] = None
    grounding_summary: Optional[Any] = None
    is_read_only_intent: bool = False
    llm_hints: dict[str, Any] = field(default_factory=dict)
    pending_guidance: Optional[dict[str, Any]] = None
    prev_spec_fingerprint: Optional[frozenset] = None
    escalation_attempt: int = 0


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
    noop_confirmed: bool = False
    no_tool_nudge_count: int = 0
    search_first_hint_done: bool = False
    reads_since_last_edit: int = 0
    goal_reminder_injected: int = 0
    rollback_performed: bool = False
    rollback_result: Any = None
    plan_current_index: int = 0
    # NOTE: keep in sync with ToolRegistry._WRITE_TOOLS (tool_registry.py:854).
    #       This default drives reads_since_last_edit reset (GOAL REMINDER guard),
    #       write_tool_used detection (read-only early-finish guard), and
    #       write-time test-impact index invalidation.  Missing a tool here means
    #       edits via that tool silently bypass all three mechanisms.
    write_tools: set = field(default_factory=lambda: {
        "apply_patch", "write_plan", "edit_ast",
        "edit_file", "edit_text", "modify_symbol", "anchor_edit",
    })
