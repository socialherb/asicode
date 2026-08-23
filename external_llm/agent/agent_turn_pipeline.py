"""
Turn pipeline mixin for AgentLoop.

Handles the LLM turn loop: message preparation, tool execution,
result processing, max-turns handling, cancellation, and errors.

Extracted from agent_loop.py to keep that file manageable.
AgentLoop inherits TurnPipelineMixin, so all methods have full access to
self.config, self.registry, etc.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from external_llm.agent._response_utils import extract_llm_reasoning
from external_llm.agent.message_shapes import (
    _is_anthropic_tool_result as _is_anthropic_shape,  # backward compat alias
)
from external_llm.agent.message_shapes import (
    _is_gemini_tool_result as _is_gemini_shape,
)
from external_llm.agent.message_shapes import (
    is_tool_call,
    is_tool_result,
)

from ..client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from ..output_parser import parse_tool_args
from ._shared_utils import (
    cache_hit_pct,
    coerce_token_count,
    context_message_cap,
    estimate_cache_adjusted_cost,
    estimate_cost,
    estimate_tokens_from_msgs,
    extract_files_from_patch,
    make_tool_signature,
    render_file_diagnostics_block,
)
from .agent_loop_types import (
    AgentCancelled,
    AgentResult,
    AgentTurn,
    TurnContext,
    _FinalAnswerOutcome,
    _PostToolResult,
    _PreparedCallsResult,
    _ResultsProcessingOutcome,
    _ToolTurnOutcome,
    _TurnPrepResult,
)
from .config.thresholds import _env_flag, config
from .context_budget import _resolve_context_limit
from .performance_metrics import get_global_collector
from .tool_registry import ToolResult

if TYPE_CHECKING:
    from external_llm.agent.performance_metrics import PerformanceCollector
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.client import LLMClient

logger = logging.getLogger(__name__)

# Config-derived constants for turn loop logic.
_NO_TOOL_NUDGE_MAX: int = config.counts.AGENT_NO_TOOL_NUDGE_MAX
_NO_PROGRESS_THRESHOLD: int = config.counts.AGENT_NO_PROGRESS_THRESHOLD
_TOOL_RETRY_LIMIT: int = config.counts.AGENT_TOOL_RETRY_LIMIT

# Stream-display content budgets (UI callback only — the LLM context budget
# lives elsewhere).  Module-level SSOT: the inline sets at the two call sites
# had already drifted once (write_plan appeared in one, not the other), and
# the tool loop would re-create them on every tool call.  `job` joins the
# verbose tier with `bash`: both surface command output, so both get the same
# 6000-char display budget instead of the 2000-char default.
_STREAM_LARGE_TOOLS = frozenset({"apply_patch", "write_plan"})
_STREAM_VERBOSE_TOOLS = frozenset({"run_tests", "run_lint", "bash", "job"})


def _write_touched_test_file(tool_name: str, tool_args: dict) -> bool:
    """Check if a write tool wrote to a test file.

    Examines *tool_args* to determine whether the write touched at least
    one path that ``is_test_file`` considers a test file.  Used to
    conditionally invalidate the test-impact index cache so that newly
    created test files are visible to subsequent scoped-verification
    selections without invalidating the cache on every edit.

    Handles four argument layouts:

    * **Direct path** — ``tool_args["path"]`` (apply_patch/edit_file) OR
      ``tool_args["file_path"]`` (edit_text/modify_symbol/edit_ast/anchor_edit)
      is set and names a test file.
    * **``apply_patch``** — the ``"patch"`` argument is parsed with
      ``extract_files_from_patch`` to recover target paths, because the
      ``"path"`` argument is optional.
    * **``write_plan``** — paths are extracted from the plan's
      ``ops`` / ``operations`` list.  The ``"plan"`` argument may be a
      ``dict``, a JSON-encoded string, or a bare ``list``
      (mirroring the normalisation in ``write_tools.py``).
    """
    from .test_impact_selector import is_test_file

    # 1. Direct path argument. edit_file/apply_patch use "path"; the AST/symbol
    #    tools (edit_text, modify_symbol, edit_ast, anchor_edit) use "file_path".
    _wp = tool_args.get("path") or tool_args.get("file_path") or ""
    if _wp and is_test_file(str(_wp)):
        return True

    # 2. apply_patch: path is optional — extract from patch text.
    if tool_name == "apply_patch":
        _patch = tool_args.get("patch", "")
        if _patch:
            for _f in extract_files_from_patch(_patch):
                if is_test_file(_f):
                    return True

    # 3. write_plan: extract paths from plan operations.
    if tool_name == "write_plan":
        _write_ops: list = []
        _plan = tool_args.get("plan")
        if isinstance(_plan, dict):
            _write_ops = _plan.get("ops") or _plan.get("operations") or []
        elif isinstance(_plan, str):
            # JSON string — mirror write_tools.py normalisation.
            try:
                _parsed = json.loads(_plan)
                if isinstance(_parsed, dict):
                    _write_ops = _parsed.get("ops") or _parsed.get("operations") or []
                elif isinstance(_parsed, list):
                    _write_ops = _parsed
            except (json.JSONDecodeError, TypeError):
                logger.debug(
                    "<module>::_write_touched_test_file:0 suppressed (json.JSONDecodeError, TypeError)", exc_info=True
                )
        elif isinstance(_plan, list):
            # Bare list — write_tools.py wraps as {"ops": plan}.
            _write_ops = _plan
        else:
            # Fallback: top-level ops / operations.
            _write_ops = tool_args.get("ops") or tool_args.get("operations") or []

        if isinstance(_write_ops, list):
            for _op in _write_ops:
                if isinstance(_op, dict) and is_test_file(str(_op.get("path", ""))):
                    return True

    return False


class TurnPipelineMixin:
    """Mixin providing the LLM turn loop for AgentLoop.

    Requires the host class to expose:
      - self.config       (AgentConfig)
      - self.registry     (ToolRegistry)
      - self.llm_client   (LLMClient)
      - self.performance_collector
      - self._failure_classifier
      - self._tool_retry_counter
      - self._patch_fail_count
      - AgentLoop methods: _cb, _build_initial_messages, _build_continuation_messages, _llm_call_with_tools,
          (cost helpers replaced by direct _shared_utils.estimate_cost / estimate_cache_adjusted_cost)
          _is_trivial_edit_request,
        _save_session_log,
        _strip_thinking_text, _append_native_tool_messages,
        _record_tool_success, _record_tool_failure, _auto_repair_apply_patch_args,
        _rollback_patches, _try_readonly_early_finish, _build_tool_result_message,
        _trim_context
      - PhaseManagerMixin: _run_self_review, _auto_test_and_inject,
        _build_tool_hint, _build_phase_state_message, _advance_phase_after_success
      - ContextManagerMixin: _trajectory_compress
    """

    # Patch engine for tolerant patch mode (lazy import)
    _patch_engine = None

    # ── Host-class attributes (provided by AgentLoop, not set here) ──
    # Class-level annotations give pyright the host contract WITHOUT runtime
    # assignment: AgentLoop.__init__ owns the real values, so these are pure
    # typing scaffolding (no setattr, no __getattr__). Mirrors the docstring
    # contract above; keep the two in sync.
    config: AgentConfig
    registry: ToolRegistry
    llm_client: LLMClient
    performance_collector: PerformanceCollector
    _failure_classifier: Any
    _tool_retry_counter: Any
    _patch_fail_count: int
    # Host methods (defined in AgentLoop / PhaseManagerMixin /
    # ContextManagerMixin / FastPathMixin). Only the *existence* matters to
    # this mixin; exact signatures live at the definitions.
    _cb: Any
    _build_initial_messages: Any
    _build_continuation_messages: Any
    _llm_call_with_tools: Any
    _is_trivial_edit_request: Any
    _save_session_log: Any
    _strip_thinking_text: Any
    _append_native_tool_messages: Any
    _record_tool_success: Any
    _record_tool_failure: Any
    _auto_repair_apply_patch_args: Any
    _rollback_patches: Any
    _try_readonly_early_finish: Any
    _build_tool_result_message: Any
    _trim_context: Any
    _run_self_review: Any
    _auto_test_and_inject: Any
    _build_tool_hint: Any
    _build_phase_state_message: Any
    _advance_phase_after_success: Any
    _trajectory_compress: Any

    # ------------------------------------------------------------------
    # Turn loop entry point
    # ------------------------------------------------------------------

    def _run_llm_loop(self, ctx: TurnContext) -> AgentResult:
        """LLM tool-loop: build initial messages, run turn loop, handle cancellation/errors."""
        ctx.plan_current_index = 0

        # ── Design chat continuation: use preserved messages instead of fresh build ──
        _continuation = getattr(self, "_continuation_data", None)
        _is_planner = (
            getattr(ctx.route, "lane", None) is not None and str(getattr(ctx.route, "lane", "")).upper() == "PLANNER"
        )
        if _continuation and not _is_planner:
            ctx.messages = self._build_continuation_messages(
                _continuation,
                ctx.request,
            )
            logger.info(
                "Using continuation messages: conversation=%d turns",
                len(_continuation.get("conversation") or []),
            )
        else:
            ctx.messages = self._build_initial_messages(ctx.request, ctx.context, tier=ctx.tier)

        ctx.tdd_fail_count = 0
        ctx.tdd_total_runs = 0
        ctx.tdd_total_pass = 0

        ctx.total_prompt_tokens = 0
        ctx.total_completion_tokens = 0
        ctx.total_cache_read_tokens = 0
        ctx.total_cache_creation_tokens = 0
        ctx.last_call_prompt_tokens = 0
        ctx.last_call_completion_tokens = 0
        ctx.provider_name = self.llm_client.get_provider_name().lower()
        ctx.model_name = self.config.model_name
        ctx.base_url = getattr(self.llm_client, "base_url", "") or ""

        ctx.write_tool_used = False
        ctx.rollback_performed = False
        ctx.rollback_result = None
        ctx.budget_warned = False
        ctx.fail_streak = {}
        # Per-run recall session identity: fresh key each run so [RECALL] dedup
        # re-arms (id()-based keys reused freed addresses and silenced recall
        # for the whole process lifetime — see failure_pattern_store note).
        from .failure_pattern_store import new_session_key

        ctx.recall_session_key = new_session_key()
        ctx.no_tool_nudge_count = 0
        ctx.any_tool_called = False
        ctx.noop_confirmed = False

        ctx.search_first_hint_done = False
        ctx.reads_since_last_edit = 0
        ctx.goal_reminder_injected = 0

        try:
            ctx.turn_num = 0
            while True:
                ctx.turn_num += 1

                # Enforce max_turns cap.
                if ctx.turn_num > self.config.max_turns:
                    return self._handle_max_turns_reached(ctx)

                # Start coalescing semantic checks for THIS turn; they run once
                # per written file at turn end, against the final content.
                self.registry.begin_semantic_turn()

                _prep = self._prepare_turn_messages(ctx)
                # NOTE: do NOT assign _prep.messages back to ctx.messages.
                # _prep.messages is the *send-time* transcript = ctx.messages +
                # ctx.ephemeral_pending hints.  Re-assigning it would permanently
                # fold the hints into history (they are re-added every turn), so
                # [AGENT STATE]/[GOAL REMINDER] would accumulate without bound.
                # ctx.messages stays pure history; _prep.messages is used only for
                # the outgoing LLM call below.
                ctx.budget_warned = _prep.budget_warned
                ctx.goal_reminder_injected = _prep.goal_reminder_injected
                ctx.search_first_hint_done = _prep.search_first_hint_done
                ctx.reads_since_last_edit = _prep.reads_since_last_edit

                logger.info("Agent turn %d (native_tools=%s)", ctx.turn_num, ctx.has_native_tools)
                self._cb(
                    "turn_start",
                    {
                        "turn": ctx.turn_num,
                        "native_tools": ctx.has_native_tools,
                        "provider": ctx.provider_name,
                        "model": getattr(self.config, "model_name", "") or "",
                    },
                )

                # Streaming token callback: forwards text_delta chunks to the
                # frontend as they arrive, so the final summary streams
                # incrementally instead of appearing all at once after a
                # blocking non-streaming call.
                _token_cb = self.config.make_token_callback()

                _llm_call_start = time.monotonic()
                try:
                    response = self._llm_call_with_tools(
                        _prep.messages,
                        token_callback=_token_cb,
                    )
                except (
                    LLMConnectionError,
                    LLMRateLimitError,
                    LLMServerUnavailableError,
                    LLMQuotaExceededError,
                    LLMAuthenticationError,
                ) as e:
                    if isinstance(e, LLMConnectionError):
                        error_type = "connection"
                        error_message = f"LLM connection error: {e}"
                    elif isinstance(e, LLMServerUnavailableError):
                        error_type = "server_unavailable"
                        error_message = f"LLM server unavailable: {e}"
                    elif isinstance(e, LLMQuotaExceededError):
                        error_type = "quota_exceeded"
                        error_message = f"LLM quota exceeded: {e}"
                    elif isinstance(e, LLMAuthenticationError):
                        error_type = "auth"
                        error_message = f"LLM authentication error: {e}"
                    else:
                        error_type = "rate_limit"
                        error_message = f"LLM rate limit error: {e}"
                    logger.exception("LLM error on turn %d", ctx.turn_num)
                    self._cb(
                        "error",
                        {
                            "message": error_message,
                            "error_type": error_type,
                            "turn": ctx.turn_num,
                        },
                    )
                    # Turn-level outcome channel: this turn terminated with an
                    # LLM error. The per-provider llm_metrics.failures already
                    # counts the call; the agent_result channel records the
                    # TURN-level outcome (same event, different granularity).
                    self.performance_collector.record_agent_result(failed=True)
                    get_global_collector().record_agent_result(failed=True)
                    self.performance_collector.end_session()
                    performance_summary = self.performance_collector.get_summary()

                    return AgentResult(
                        status="error",
                        turns=ctx.turns,
                        error=error_message,
                        applied_patches=list(self.registry.applied_patches),
                        metadata={
                            "performance": performance_summary,
                        },
                    )

                def _rget(_key: str, _default=None, _response=response):
                    # B023: bind the per-iteration response at def time. All
                    # call sites are synchronous within this iteration today,
                    # but def-time binding keeps the closure correct if it is
                    # ever deferred past the next `response =` reassignment.
                    if isinstance(_response, dict):
                        return _response.get(_key, _default)
                    return getattr(_response, _key, _default)

                # ── Token tracking ──
                _pt = _rget("prompt_tokens", 0)
                if _pt is None:
                    _pt = _rget("tokens_used", 0)  # fallback: total when split unavailable
                _ct = _rget("completion_tokens", 0)
                # Coerce None -> 0 AFTER the fallback decision: the fallback
                # must keep `is None` semantics (a real 0 is a valid value and
                # must not trigger the tokens_used fallback), while None fields
                # (Optional[int] = None on provider responses that carry no
                # usage report) must not TypeError the accumulation below.
                # coerce_token_count additionally guards non-int truthy values
                # (Mock auto-attributes, JSON-decoded strings from gateway
                # shims) that would crash the whole turn loop in +=.
                _pt = coerce_token_count(_pt)
                _ct = coerce_token_count(_ct)
                _crt = coerce_token_count(_rget("cache_read_input_tokens", 0))
                _cct = coerce_token_count(_rget("cache_creation_input_tokens", 0))
                ctx.total_prompt_tokens += _pt
                ctx.total_completion_tokens += _ct
                ctx.total_cache_read_tokens += _crt
                ctx.total_cache_creation_tokens += _cct
                ctx.last_call_prompt_tokens = _pt
                ctx.last_call_completion_tokens = _ct
                _llm_elapsed_ms = int((time.monotonic() - _llm_call_start) * 1000)
                _finish_reason = _rget("finish_reason", "")
                _raw_tool_calls = _rget("tool_calls", [])
                # Same crash class as the token fields: a Mock auto-attribute
                # has no len() and would TypeError here.
                _tool_calls_count = len(_raw_tool_calls) if isinstance(_raw_tool_calls, (list, tuple)) else 0
                if _pt or _ct:
                    _turn_cost = estimate_cost(ctx.provider_name, _pt, _ct, model=ctx.model_name)
                    _total_cost = estimate_cost(
                        ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name
                    )
                    _turn_actual_cost = estimate_cache_adjusted_cost(
                        ctx.provider_name, _pt, _ct, _crt, _cct, model=ctx.model_name, base_url=ctx.base_url
                    )
                    _total_actual_cost = estimate_cache_adjusted_cost(
                        ctx.provider_name,
                        ctx.total_prompt_tokens,
                        ctx.total_completion_tokens,
                        ctx.total_cache_read_tokens,
                        ctx.total_cache_creation_tokens,
                        model=ctx.model_name,
                        base_url=ctx.base_url,
                    )
                    self._cb(
                        "token_usage",
                        {
                            "turn": ctx.turn_num,
                            "prompt_tokens": _pt,
                            "completion_tokens": _ct,
                            "cache_read_tokens": _crt,
                            "cache_creation_tokens": _cct,
                            "total_prompt_tokens": ctx.total_prompt_tokens,
                            "total_completion_tokens": ctx.total_completion_tokens,
                            "total_cache_read_tokens": ctx.total_cache_read_tokens,
                            "total_cache_creation_tokens": ctx.total_cache_creation_tokens,
                            "turn_cost_usd": round(_turn_cost, 6),
                            "total_cost_usd": round(_total_cost, 6),
                            "turn_actual_cost_usd": round(_turn_actual_cost, 6),
                            "total_actual_cost_usd": round(_total_actual_cost, 6),
                            "provider": ctx.provider_name,
                            "llm_elapsed_ms": _llm_elapsed_ms,
                            "finish_reason": _finish_reason,
                            "tool_calls_count": _tool_calls_count,
                        },
                    )

                # Normalize to a real list: tuples from gateway shims must not
                # leak into _execute_and_process_tool_calls' list-typed param.
                tool_calls = list(_raw_tool_calls) if isinstance(_raw_tool_calls, (list, tuple)) else []
                content = _rget("content", "")

                if content and content.strip():
                    self._cb(
                        "agent_thinking",
                        {
                            "turn": ctx.turn_num,
                            "content": content.strip()[:2000],
                            "agent_id": self.config.agent_id,
                            "has_tool_calls": bool(tool_calls),
                        },
                    )

                # Completion detection: finish_reason=stop/end_turn with tool_calls
                if tool_calls and _finish_reason in ("stop", "end_turn"):
                    logger.info(
                        "finish_reason=%s with tool_calls → treating as completion",
                        _finish_reason,
                    )
                    tool_calls = []

                # No tool calls → final answer
                if not tool_calls:
                    _fa = self._handle_final_answer_turn(
                        ctx,
                        final_msg=self._effective_final_content(response),
                    )
                    if _fa.nudge_message is not None:
                        ctx.no_tool_nudge_count = _fa.nudge_count
                        ctx.messages.append(_fa.nudge_message)
                        # Reset any_tool_called so post-nudge text-only
                        # responses hit the text_reply path (the no-tool-call text_reply outcome)
                        # instead of false-success re-nudge death loop.
                        ctx.any_tool_called = False
                        continue
                    if _fa.result is None:
                        # Unreachable by construction (_handle_final_answer_turn
                        # always sets result when nudge_message is None), but a
                        # defensive bar keeps _run_llm_loop's AgentResult return
                        # type honest for type checkers.
                        raise RuntimeError("final-answer outcome carried no result")
                    return _fa.result

                # Execute tool calls
                _tool_out = self._execute_and_process_tool_calls(
                    ctx,
                    tool_calls=tool_calls,
                )
                if _tool_out.early_return is not None:
                    return _tool_out.early_return
                if _tool_out.should_continue:
                    if _tool_out.phase_rule_messages:
                        ctx.messages = ctx.messages + _tool_out.phase_rule_messages
                    continue
                new_messages = _tool_out.new_messages
                ctx.write_tool_used = _tool_out.write_tool_used
                ctx.any_tool_called = _tool_out.any_tool_called
                ctx.fail_streak = _tool_out.fail_streak
                ctx.reads_since_last_edit = _tool_out.reads_since_last_edit
                ctx.plan_current_index = _tool_out.plan_current_index
                if _tool_out.noop_confirmed:
                    ctx.noop_confirmed = True

                _post = self._process_post_tool_turn(
                    ctx,
                    response=response,
                    new_messages=new_messages,
                )
                if _post.early_return is not None:
                    return _post.early_return
                ctx.messages = _post.messages
                ctx.tdd_fail_count = _post.tdd_fail_count
                ctx.tdd_total_runs = _post.tdd_total_runs
                ctx.tdd_total_pass = _post.tdd_total_pass

        except AgentCancelled:
            return self._handle_loop_cancellation(turns=ctx.turns, git_state=ctx.git_state)

        except Exception as e:
            return self._handle_loop_error(
                error=e,
                turns=ctx.turns,
                git_state=ctx.git_state,
                rollback_performed=ctx.rollback_performed,
                rollback_result=ctx.rollback_result,
            )
        finally:
            # The turn body can be abandoned mid-flight (uncaught exception /
            # cancellation) after begin_semantic_turn() opened a turn but before
            # the normal drain at turn end (_settle_deferred_semantics). Without
            # this, the registry is left believing a turn is active and every
            # subsequent out-of-turn dispatch would defer — and silently drop —
            # its diagnostics into a queue nothing drains. No-op on the normal
            # path: drain already cleared both fields.
            self.registry.end_semantic_turn()

    @staticmethod
    def _effective_final_content(response) -> str:
        """Return the final-message content, falling back to ``reasoning_content``.

        GLM-5.2 (thinking ON) / DeepSeek Reasoner may emit the final answer in
        ``reasoning_content`` with an empty ``content`` field. Without this
        fallback the closing summary after tool work is silently swallowed and
        the agent returns to the prompt with no final message — tools ran and
        succeeded but the user sees nothing.

        Mirrors the reasoning_content→content fallback that already exists on
        EVERY termination path of DesignChatLoop (design_chat_loop.py) and in
        the OpenAI streaming reconstruction. The agent turn pipeline has five
        termination/early-finish paths that extract ``content``; all must apply
        this fallback for parity (multi-path fallback parity principle).
        """
        if isinstance(response, dict):
            content = response.get("content") or ""
            raw_obj = response.get("raw")
        else:
            content = getattr(response, "content", "") or ""
            raw_obj = getattr(response, "raw", None)
        if isinstance(content, str) and content.strip():
            return content
        raw_resp = getattr(raw_obj, "raw_response", None) if raw_obj is not None else None
        if isinstance(raw_resp, dict):
            try:
                rc = extract_llm_reasoning(raw_resp, strip=True)
                if rc:
                    return rc
            except (AttributeError, TypeError, IndexError):
                logger.debug(
                    "<module>::TurnPipelineMixin::_effective_final_content:0 suppressed (AttributeError, TypeError, IndexError)",
                    exc_info=True,
                )
        return content if isinstance(content, str) else ""

    # ------------------------------------------------------------------
    # Max turns handler
    # ------------------------------------------------------------------

    def _handle_max_turns_reached(self, ctx: TurnContext) -> AgentResult:
        """Handle max_turns exhaustion: attempt one final no-tool LLM call, return result."""
        logger.warning("Agent reached max_turns=%d", self.config.max_turns)

        # Streaming token callback (same as main turn loop)
        _token_cb = self.config.make_token_callback()

        try:
            response = self._llm_call_with_tools(
                ctx.messages,
                token_callback=_token_cb,
            )

            def _rget(key: str, default=None):
                getter = getattr(response, "get", None)
                if callable(getter):
                    try:
                        return getter(key, default)
                    except (AttributeError, TypeError, KeyError):
                        logger.debug(
                            "<module>::TurnPipelineMixin::_handle_max_turns_reached::_rget:0 suppressed (AttributeError, TypeError, KeyError)",
                            exc_info=True,
                        )
                return getattr(response, key, default)

            _pt = _rget("prompt_tokens", 0)
            if _pt is None:
                _pt = _rget("tokens_used", 0)  # fallback: total when split unavailable
            _pt = coerce_token_count(_pt)
            _ct = coerce_token_count(_rget("completion_tokens", 0))
            # Defense-depth parity with main turn loop (_run_llm_loop): accumulate
            # cache tokens here too — the max_turns final call consumes cache
            # budget and must be reflected in the per-bucket counters.
            # Fallback semantics match the main loop's documented contract: a
            # REAL 0 is a valid split value and must not trigger the fallback
            # (only an explicit None — provider omitted the split — may).
            _crt = coerce_token_count(_rget("cache_read_input_tokens", 0))
            _cct = coerce_token_count(_rget("cache_creation_input_tokens", 0))
            ctx.total_prompt_tokens += _pt
            ctx.total_completion_tokens += _ct
            ctx.total_cache_read_tokens += _crt
            ctx.total_cache_creation_tokens += _cct
            ctx.last_call_prompt_tokens = _pt
            ctx.last_call_completion_tokens = _ct

            final_tool_calls = _rget("tool_calls", []) or []

            if final_tool_calls:
                ctx.messages.append(
                    LLMMessage(
                        role="user",
                        content="[WRAP UP] Turn limit reached. Do NOT call any more tools. "
                        "Summarize what was accomplished. Do NOT continue working — the session is ending.",
                    )
                )
                response = self._llm_call_with_tools(
                    ctx.messages,
                    token_callback=_token_cb,
                )

                # NOTE: _rget closes over `response` *by reference*, so after the
                # reassignment above it already reads the new response — a second
                # identical closure (_rget2) was redundant and has been removed.
                _pt = _rget("prompt_tokens", 0)
                if _pt is None:
                    _pt = _rget("tokens_used", 0)  # fallback: total when split unavailable
                _pt = coerce_token_count(_pt)
                _ct = coerce_token_count(_rget("completion_tokens", 0))
                # Defense-depth parity with main turn loop (_run_llm_loop): accumulate
                # cache tokens for the wrap-up retry call too.
                _crt = coerce_token_count(_rget("cache_read_input_tokens", 0))
                _cct = coerce_token_count(_rget("cache_creation_input_tokens", 0))
                ctx.total_prompt_tokens += _pt
                ctx.total_completion_tokens += _ct
                ctx.total_cache_read_tokens += _crt
                ctx.total_cache_creation_tokens += _cct
                ctx.last_call_prompt_tokens = _pt
                ctx.last_call_completion_tokens = _ct

                final_tool_calls = _rget("tool_calls", []) or []
                if final_tool_calls:
                    raise RuntimeError("model still requesting tools after wrap-up")
                final_msg = self._effective_final_content(response)
            else:
                final_msg = self._effective_final_content(response)

            if (not ctx.read_only_request) and (not self.registry.applied_patches):
                raise RuntimeError("write-intent request reached final completion path without any applied patches")

            review_summary: str | None = None
            should_do_review = self.config.self_review_enabled and self.registry.applied_patches
            if should_do_review and self._is_trivial_edit_request(ctx.request):
                should_do_review = False
            if should_do_review:
                review_summary = self._run_self_review()
                if review_summary and " lgtm " not in f" {review_summary.lower()} ":
                    final_msg += f"\n\n---\n**[Self-Review]** {review_summary}"

            self.performance_collector.end_session()
            performance_summary = self.performance_collector.get_summary()

            _final_result = AgentResult(
                status="success",
                turns=ctx.turns,
                final_message=final_msg,
                applied_patches=list(self.registry.applied_patches),
                metadata={
                    "turns_used": self.config.max_turns,
                    "plan": ctx.plan,
                    "tdd": {
                        "runs": ctx.tdd_total_runs,
                        "pass": ctx.tdd_total_pass,
                        "fail": ctx.tdd_fail_count,
                    },
                    "self_review": {
                        "enabled": self.config.self_review_enabled,
                        "summary": review_summary,
                        "issues_found": bool(review_summary and " lgtm " not in f" {review_summary.lower()} "),
                    },
                    "tokens": _token_metadata(ctx),
                    "performance": performance_summary,
                },
            )
            self._save_session_log(
                ctx.session_id,
                ctx.request,
                _final_result,
                ctx.total_prompt_tokens,
                ctx.total_completion_tokens,
                ctx.total_cache_read_tokens,
                ctx.total_cache_creation_tokens,
            )

        except Exception as e:
            logger.debug("Final LLM call after max_turns failed: %s", e)
            if "without any applied patches" in str(e):
                logger.warning("Blocking false success at max_turns for write-intent request")
        else:
            return _final_result

        _max_result = AgentResult(
            status="max_turns",
            turns=ctx.turns,
            final_message=f"Reached maximum turns ({self.config.max_turns})",
            applied_patches=list(self.registry.applied_patches),
            metadata={
                "turns_used": self.config.max_turns,
                "git_state": ctx.git_state,
                "tdd": {
                    "runs": ctx.tdd_total_runs,
                    "pass": ctx.tdd_total_pass,
                    "fail": ctx.tdd_fail_count,
                },
                "tokens": _token_metadata(ctx),
            },
        )
        self._save_session_log(
            ctx.session_id,
            ctx.request,
            _max_result,
            ctx.total_prompt_tokens,
            ctx.total_completion_tokens,
            ctx.total_cache_read_tokens,
            ctx.total_cache_creation_tokens,
        )
        return _max_result

    # ------------------------------------------------------------------
    # Final answer handler
    # ------------------------------------------------------------------

    def _handle_final_answer_turn(
        self,
        ctx: TurnContext,
        final_msg: str,
    ) -> _FinalAnswerOutcome:

        logger.info("Agent finished after %d turns", ctx.turn_num - 1)

        # The block below asserts "the user asked for an edit and none happened".
        # ``read_only_request`` alone does not establish that premise: every
        # IntentResolver failure path also yields read_only_request=False (the
        # deliberate never-block-an-edit default). Running the gate on an
        # unresolved intent turned answered QUESTIONS into status="error" and
        # injected `bash('cat > path << EOF')` instructions into read-only
        # conversations. Write PERMISSION is untouched — only the expectation.
        if ctx.intent_undetermined and not self.registry.applied_patches:
            logger.info(
                "Skipping write-intent false-success gate — intent resolution "
                "never classified this request (no evidence an edit was asked for)"
            )
        elif (not ctx.read_only_request) and (not self.registry.applied_patches):
            if ctx.noop_confirmed and final_msg:
                logger.info("No-op task confirmed — returning success with no patches")
                self.performance_collector.end_session()
                performance_summary = self.performance_collector.get_summary()
                _noop_result = AgentResult(
                    status="success",
                    turns=ctx.turns,
                    final_message=final_msg,
                    applied_patches=[],
                    metadata={
                        "turns_used": ctx.turn_num - 1,
                        "noop": True,
                        "performance": performance_summary,
                        "tokens": _token_metadata(ctx),
                    },
                )
                self._save_session_log(
                    ctx.session_id,
                    ctx.request,
                    _noop_result,
                    ctx.total_prompt_tokens,
                    ctx.total_completion_tokens,
                )
                return _FinalAnswerOutcome(result=_noop_result)

            if not ctx.any_tool_called and final_msg:
                logger.info("No tools called, text reply detected — returning text_reply status")
                self.performance_collector.end_session()
                _text_reply_result = AgentResult(
                    status="text_reply",
                    turns=ctx.turns,
                    final_message=final_msg,
                    applied_patches=[],
                    metadata={
                        "turns_used": ctx.turn_num - 1,
                        "tokens": _token_metadata(ctx),
                    },
                )
                self._save_session_log(
                    ctx.session_id,
                    ctx.request,
                    _text_reply_result,
                    ctx.total_prompt_tokens,
                    ctx.total_completion_tokens,
                )
                return _FinalAnswerOutcome(result=_text_reply_result)

            logger.warning("Blocking false success: write-intent request finished with no applied patches")

            if ctx.no_tool_nudge_count < _NO_TOOL_NUDGE_MAX and ctx.turn_num < self.config.max_turns:
                ctx.no_tool_nudge_count += 1
                # Only the write wording is reachable here: the enclosing branch
                # already requires ``not ctx.read_only_request``. A read-only
                # variant used to sit behind `if ctx.read_only_request:` inside
                # this same block — dead since it contradicted its own guard.
                nudge_content = (
                    "[ACTION REQUIRED] You described the change but applied NO patch.\n"
                    "You MUST output a tool call. Do NOT write code as plain text.\n\n"
                    "To CREATE a new file:\n"
                    "  write_plan with a create_file op, or bash(\"cat > path << 'EOF'\\n...content...\\nEOF\")\n\n"
                    "To MODIFY an existing file:\n"
                    "  1. read_file to see current content\n"
                    "  2. apply_patch with unified diff\n\n"
                    f"Task: {ctx.request[:2000]}"
                )
                nudge_msg = LLMMessage(role="user", content=nudge_content)
                self._cb(
                    "tool_nudge",
                    {"turn": ctx.turn_num, "nudge_count": ctx.no_tool_nudge_count, "agent_id": self.config.agent_id},
                )
                logger.info("Re-nudging small model (nudge %d/%d)", ctx.no_tool_nudge_count, _NO_TOOL_NUDGE_MAX)
                return _FinalAnswerOutcome(nudge_message=nudge_msg, nudge_count=ctx.no_tool_nudge_count)

            self.performance_collector.end_session()
            performance_summary = self.performance_collector.get_summary()

            _false_success_result = AgentResult(
                status="error",
                turns=ctx.turns,
                final_message=(final_msg or "Model finished without calling apply_patch, and no patch was applied."),
                applied_patches=list(self.registry.applied_patches),
                error="write_intent_finished_without_patch",
                metadata={
                    "turns_used": ctx.turn_num - 1,
                    "plan": ctx.plan,
                    "tdd": {
                        "runs": ctx.tdd_total_runs,
                        "pass": ctx.tdd_total_pass,
                        "fail": ctx.tdd_fail_count,
                    },
                    "self_review": {
                        "enabled": self.config.self_review_enabled,
                        "summary": None,
                        "issues_found": False,
                    },
                    "tokens": _token_metadata(ctx),
                    "performance": performance_summary,
                    "false_success_blocked": True,
                    "nudge_count": ctx.no_tool_nudge_count,
                },
            )
            self._save_session_log(
                ctx.session_id,
                ctx.request,
                _false_success_result,
                ctx.total_prompt_tokens,
                ctx.total_completion_tokens,
            )
            return _FinalAnswerOutcome(result=_false_success_result)

        review_summary: str | None = None

        should_do_review = self.config.self_review_enabled and self.registry.applied_patches
        if should_do_review and self._is_trivial_edit_request(ctx.request):
            logger.info("Skipping self-review phase for trivial request")
            should_do_review = False

        if should_do_review:
            review_summary = self._run_self_review()
            if review_summary and " lgtm " not in f" {review_summary.lower()} ":
                final_msg = final_msg + f"\n\n---\n**[Self-Review]** {review_summary}"

        # LLM re-invocation for future-tense detection removed — once a patch has been applied,
        # the task is complete even if the sentence is in future tense. Unnecessary LLM calls
        # only waste tokens.

        self.performance_collector.end_session()
        performance_summary = self.performance_collector.get_summary()

        _final_result = AgentResult(
            status="success",
            turns=ctx.turns,
            final_message=final_msg,
            applied_patches=list(self.registry.applied_patches),
            metadata={
                "turns_used": ctx.turn_num - 1,
                "plan": ctx.plan,
                "tdd": {
                    "runs": ctx.tdd_total_runs,
                    "pass": ctx.tdd_total_pass,
                    "fail": ctx.tdd_fail_count,
                },
                "self_review": {
                    "enabled": self.config.self_review_enabled,
                    "summary": review_summary,
                    "issues_found": bool(review_summary and " lgtm " not in f" {review_summary.lower()} "),
                },
                "tokens": _token_metadata(ctx),
                "performance": performance_summary,
            },
        )
        self._save_session_log(
            ctx.session_id,
            ctx.request,
            _final_result,
            ctx.total_prompt_tokens,
            ctx.total_completion_tokens,
        )
        return _FinalAnswerOutcome(result=_final_result)

    # ------------------------------------------------------------------
    # Post-tool turn processing
    # ------------------------------------------------------------------

    def _process_post_tool_turn(
        self,
        ctx: TurnContext,
        response: Any,
        new_messages: list,
    ) -> _PostToolResult:
        """Process results after tool execution: sanitize, update messages, auto-observe, TDD."""
        try:
            if hasattr(response, "content") and isinstance(response.content, str):
                response.content = self._strip_thinking_text(response.content)
        except (AttributeError, TypeError):
            logger.debug(
                "<module>::TurnPipelineMixin::_process_post_tool_turn:0 suppressed (AttributeError, TypeError)",
                exc_info=True,
            )

        ctx.messages = self._append_native_tool_messages(ctx.messages, response, new_messages)

        _patch_tools = {"apply_patch", "write_plan"}
        patch_ok_this_turn = any(
            t.tool_name in _patch_tools and t.tool_result.ok for t in ctx.turns if t.turn_num == ctx.turn_num
        )

        # Auto-observation after successful patch.
        # Scope `git diff` to files touched THIS turn only. A bare `git diff`
        # dumps the entire working tree and re-surfaces earlier turns' changes
        # every turn, ballooning ctx.messages with redundant content. Both
        # apply_patch and write_plan record affected paths in their ToolResult
        # metadata under "touched_files" (write_plan via diff_apply details).
        if patch_ok_this_turn and self.config.auto_observation_enabled and not self.config.auto_test_on_patch:
            _obs_paths: list[str] = []
            for _t in ctx.turns:
                if _t.turn_num != ctx.turn_num or _t.tool_name not in _patch_tools:
                    continue
                _tr = _t.tool_result
                if not getattr(_tr, "ok", False):
                    continue
                _meta = getattr(_tr, "metadata", None) or {}
                _obs_paths.extend(_meta.get("touched_files") or _meta.get("files") or [])
            _obs_paths = list(dict.fromkeys(_obs_paths))  # de-dup, preserve order
            obs_content = ""
            if _obs_paths:
                _sp = __import__("subprocess")
                try:
                    _diff_proc = _sp.run(
                        ["git", "diff", "--", *_obs_paths],
                        cwd=self.registry.repo_root,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    obs_content = (_diff_proc.stdout or "").strip()
                except Exception:
                    obs_content = ""
            if obs_content:
                ctx.messages.append(LLMMessage(role="user", content=f"[auto_observation]\n{obs_content}"))
                self._cb("auto_observation", {"turn": ctx.turn_num, "diff": obs_content})

        # Early finish after successful patch (no TDD, no self-review)
        if patch_ok_this_turn and (not self.config.auto_test_on_patch) and (not self.config.self_review_enabled):
            self.performance_collector.end_session()
            performance_summary = self.performance_collector.get_summary()

            _llm_last_msg = self._effective_final_content(response).strip() if response else ""
            _final_result = AgentResult(
                status="success",
                turns=ctx.turns,
                final_message=_llm_last_msg or "Task completed. Changes applied.",
                applied_patches=list(self.registry.applied_patches),
                metadata={
                    "turns_used": ctx.turn_num,
                    "plan": ctx.plan,
                    "tdd": {
                        "runs": ctx.tdd_total_runs,
                        "pass": ctx.tdd_total_pass,
                        "fail": ctx.tdd_fail_count,
                    },
                    "self_review": {
                        "enabled": self.config.self_review_enabled,
                        "summary": None,
                        "issues_found": False,
                    },
                    "tokens": _token_metadata(ctx),
                    "performance": performance_summary,
                    "early_finish": {
                        "enabled": True,
                        "reason": "patch_ok_this_turn_and_no_tdd_and_no_self_review",
                    },
                },
            )
            self._save_session_log(
                ctx.session_id,
                ctx.request,
                _final_result,
                ctx.total_prompt_tokens,
                ctx.total_completion_tokens,
            )
            return _PostToolResult(
                messages=ctx.messages,
                tdd_fail_count=ctx.tdd_fail_count,
                tdd_total_runs=ctx.tdd_total_runs,
                tdd_total_pass=ctx.tdd_total_pass,
                early_return=_final_result,
            )

        # TDD auto-test
        if self.config.auto_test_on_patch and patch_ok_this_turn:
            ctx.tdd_total_runs += 1
            ctx.messages, ctx.tdd_fail_count = self._auto_test_and_inject(
                ctx.messages, ctx.turn_num, ctx.tdd_fail_count
            )
            if ctx.tdd_fail_count == 0:
                ctx.tdd_total_pass += 1
                if not self.config.self_review_enabled:
                    self.performance_collector.end_session()
                    performance_summary = self.performance_collector.get_summary()

                    _llm_last_msg2 = self._effective_final_content(response).strip() if response else ""
                    _final_result = AgentResult(
                        status="success",
                        turns=ctx.turns,
                        final_message=_llm_last_msg2 or "All tests passed. Changes applied.",
                        applied_patches=list(self.registry.applied_patches),
                        metadata={
                            "turns_used": ctx.turn_num,
                            "plan": ctx.plan,
                            "tdd": {
                                "runs": ctx.tdd_total_runs,
                                "pass": ctx.tdd_total_pass,
                                "fail": ctx.tdd_fail_count,
                            },
                            "self_review": {
                                "enabled": self.config.self_review_enabled,
                                "summary": None,
                                "issues_found": False,
                            },
                            "tokens": _token_metadata(ctx),
                            "performance": performance_summary,
                        },
                    )
                    self._save_session_log(
                        ctx.session_id,
                        ctx.request,
                        _final_result,
                        ctx.total_prompt_tokens,
                        ctx.total_completion_tokens,
                    )
                    return _PostToolResult(
                        messages=ctx.messages,
                        tdd_fail_count=ctx.tdd_fail_count,
                        tdd_total_runs=ctx.tdd_total_runs,
                        tdd_total_pass=ctx.tdd_total_pass,
                        early_return=_final_result,
                    )

        return _PostToolResult(
            messages=ctx.messages,
            tdd_fail_count=ctx.tdd_fail_count,
            tdd_total_runs=ctx.tdd_total_runs,
            tdd_total_pass=ctx.tdd_total_pass,
        )

    # ------------------------------------------------------------------
    # Turn message preparation
    # ------------------------------------------------------------------

    def _prepare_turn_messages(self, ctx: TurnContext) -> _TurnPrepResult:
        """Prepare message list for one LLM turn.

        Injects hints and guidance via ctx.ephemeral_pending (merged at return),
        trims context.  May raise AgentCancelled if cancel_event is set.
        Returns updated messages and loop-state flags.
        """
        # Ephemeral hint/guidance messages are no longer injected into
        # ctx.messages — they accumulate in ctx.ephemeral_pending and are
        # merged into the outgoing message list at the return point.
        # No pruning is needed; ctx.messages remains stable across turns.
        ctx.ephemeral_pending.clear()
        ctx.messages = _evict_for_loop(
            ctx.messages,
            model=getattr(ctx, "model_name", "") or "",
            # Must match the variant sent to the API (agent_loop passes
            # lang_filter=repo_language) or the budget accounts for masked
            # python-only tools the request never carries — get_tool_names'
            # docstring declares this agreement an explicit contract.
            tool_schemas=(
                self.registry.get_tool_schemas(lang_filter=self.registry.repo_language) if self.registry else None
            ),
            base_url=getattr(ctx, "base_url", None),
        )

        if ctx.turn_num == 1 and not ctx.search_first_hint_done and ctx.target_keywords and not ctx.known_target_file:
            _sf_targets = ", ".join(f'"{k}"' for k in ctx.target_keywords[:2])
            ctx.ephemeral_pending.append(
                LLMMessage(
                    role="system",
                    content=(
                        f"[TOOL HINT] Text change task detected. Target: {_sf_targets}.\n"
                        "Priority order to locate the target:\n"
                        "  1. find_symbol — fastest if the target is a function/class/symbol name.\n"
                        "  2. bash (grep -rn) — use if the target is arbitrary text, not a named symbol.\n"
                        "  3. read_symbol or bash (cat) — ONLY after you know the exact file and line from step 1 or 2.\n"
                        "Do NOT browse files randomly without locating the target first."
                    ),
                )
            )
            ctx.search_first_hint_done = True

        if ctx.reads_since_last_edit >= _NO_PROGRESS_THRESHOLD and ctx.goal_reminder_injected < 3:
            _rf_count = ctx.reads_since_last_edit
            _reminder_text = (
                f"[GOAL REMINDER] You have called {_rf_count} exploration tools "
                "without making any edits.\n"
                f"Original task: {ctx.request[:2000]}\n"
                "Action required: either apply the edit now with apply_patch, "
                "or use find_symbol/bash (grep) to locate the exact target first.\n"
                "Stop reading files that don't contain the target text."
            )
            if ctx.goal_reminder_injected >= 1:
                _reminder_text += (
                    "\n\nIf you have enough context, apply the edit NOW with apply_patch. Do not read more files."
                )
            ctx.ephemeral_pending.append(
                LLMMessage(
                    role="user",
                    content=_reminder_text,
                )
            )
            ctx.goal_reminder_injected += 1
            ctx.reads_since_last_edit = 0
            logger.info(
                "Goal reminder injected (reminder #%d, was %d reads without edit)",
                ctx.goal_reminder_injected,
                _rf_count,
            )
            self._cb(
                "goal_reminder",
                {
                    "turn": ctx.turn_num,
                    "reads_without_edit": _rf_count,
                    "reminder_count": ctx.goal_reminder_injected,
                },
            )
        try:
            tool_hint = self._build_tool_hint()
            if tool_hint:
                ctx.ephemeral_pending.append(LLMMessage(role="system", content=tool_hint))
        except (AttributeError, TypeError):
            logger.debug(
                "<module>::TurnPipelineMixin::_prepare_turn_messages:0 suppressed (AttributeError, TypeError)",
                exc_info=True,
            )

        # Planner progress hint
        if ctx.plan_subtasks and ctx.plan_current_index < len(ctx.plan_subtasks):
            try:
                task = ctx.plan_subtasks[ctx.plan_current_index]
                hint = (
                    "[PLAN PROGRESS]\n"
                    f"Current subtask ({ctx.plan_current_index + 1}/{len(ctx.plan_subtasks)}): "
                    f"{task.get('title', '')}\n"
                    f"Target files: {', '.join(task.get('files') or [])}"
                )
                ctx.ephemeral_pending.append(LLMMessage(role="user", content=hint))
            except (AttributeError, TypeError):
                logger.debug(
                    "<module>::TurnPipelineMixin::_prepare_turn_messages:1 suppressed (AttributeError, TypeError)",
                    exc_info=True,
                )

        if ctx.read_only_request or ctx.is_local_model:
            try:
                ctx.ephemeral_pending.append(
                    LLMMessage(
                        role="system",
                        content=self._build_phase_state_message(ctx.read_only_request),
                    )
                )
            except (AttributeError, TypeError):
                logger.debug(
                    "<module>::TurnPipelineMixin::_prepare_turn_messages:2 suppressed (AttributeError, TypeError)",
                    exc_info=True,
                )

        if ctx.known_target_file and not ctx.read_only_request and ctx.turn_num == 1:
            try:
                ctx.ephemeral_pending.append(
                    LLMMessage(
                        role="system",
                        content=(
                            "[TARGET FILE STRATEGY]\n"
                            f"Known target file: {ctx.known_target_file}\n"
                            "This is a write task with a known target — no need to search. "
                            "Read the file first before modifying: read_symbol or bash (cat) on that file. "
                            "After reading, proceed to apply_patch."
                        ),
                    )
                )
            except (AttributeError, TypeError):
                logger.debug(
                    "<module>::TurnPipelineMixin::_prepare_turn_messages:3 suppressed (AttributeError, TypeError)",
                    exc_info=True,
                )

        try:
            if ctx.turn_num > 2:
                traj = self._trajectory_compress(ctx.turns)
                if traj:
                    ctx.ephemeral_pending.append(LLMMessage(role="user", content=traj))
        except (AttributeError, TypeError):
            logger.debug(
                "<module>::TurnPipelineMixin::_prepare_turn_messages:4 suppressed (AttributeError, TypeError)",
                exc_info=True,
            )

        if self.config.cancel_event and self.config.cancel_event.is_set():
            raise AgentCancelled("cancelled by user")

        if self.config.message_queue is not None:
            import queue as _queue_mod

            while True:
                try:
                    mid_msg = self.config.message_queue.get_nowait()
                    ctx.messages.append(
                        LLMMessage(
                            role="user",
                            content=f"[USER INTERRUPT] {mid_msg}",
                        )
                    )
                    self._cb(
                        "user_message_received",
                        {
                            "message": mid_msg,
                            "turn": ctx.turn_num,
                        },
                    )
                    logger.info("Mid-task user message injected at turn %d: %s", ctx.turn_num, mid_msg[:80])
                except _queue_mod.Empty:
                    logger.debug(
                        "<module>::TurnPipelineMixin::_prepare_turn_messages:5 suppressed _queue_mod.Empty",
                        exc_info=True,
                    )
                    break

        ctx.messages = self._trim_context(ctx.messages)

        _max_turns = getattr(self.config, "max_turns", 0)
        if _max_turns > 0 and not ctx.budget_warned and ctx.turn_num >= _max_turns - 4:
            budget_warned_msg = (
                f"[BUDGET WARNING] You have approximately {_max_turns - ctx.turn_num + 1} turns remaining. "
                "Focus on completing edits and preparing a final summary. "
                "Avoid starting new explorations or initiating additional changes."
            )
            ctx.ephemeral_pending.append(LLMMessage(role="user", content=budget_warned_msg))
            ctx.budget_warned = True

        return _TurnPrepResult(
            messages=ctx.messages + ctx.ephemeral_pending,
            budget_warned=ctx.budget_warned,
            goal_reminder_injected=ctx.goal_reminder_injected,
            search_first_hint_done=ctx.search_first_hint_done,
            reads_since_last_edit=ctx.reads_since_last_edit,
        )

    # ------------------------------------------------------------------
    # Build and filter prepared calls
    # ------------------------------------------------------------------

    def _build_and_filter_prepared_calls(
        self,
        tool_calls: list,
        turns: list,
        plan_subtasks: list,
        plan_current_index: int,
        read_only_request: bool,
        turn_num: int,
    ) -> _PreparedCallsResult:
        """Build prepared_calls from raw tool_calls, filter, emit previews.

        May raise AgentCancelled if cancel_event is set.
        Returns _PreparedCallsResult. If should_continue is True, caller should
        return a should_continue _ToolTurnOutcome immediately.
        """
        if plan_subtasks:
            try:
                last_tool = turns[-1].tool_name if turns else None
                if last_tool in {"apply_patch", "write_plan"}:
                    plan_current_index = min(plan_current_index + 1, len(plan_subtasks))
            except (TypeError, AttributeError):
                logger.debug(
                    "<module>::TurnPipelineMixin::_build_and_filter_prepared_calls:0 suppressed (TypeError, AttributeError)",
                    exc_info=True,
                )

        prepared_calls = []
        _unknown_tool_notices: list[str] = []

        # Pre-bind so the except path below cannot leak an unbound name: the
        # masking notices read _repo_lang after the try/except, and a guard
        # failure must degrade to "no language filter" (None), never NameError.
        _repo_lang = None
        _known_tools: frozenset[str] = frozenset()
        _masked_tools: frozenset[str] = frozenset()
        try:
            _repo_lang = getattr(self.registry, "repo_language", None)
            if hasattr(self.registry, "get_tool_names"):
                _known_tools = frozenset(str(t) for t in self.registry.get_tool_names(lang_filter=_repo_lang))
                # Tools hidden by language masking (non-Python repo). Distinguishing
                # them from genuinely-unknown tools lets us emit a precise notice
                # instead of the generic read-only-mode message.
                _all_names = frozenset(str(t) for t in self.registry.get_tool_names())
                _masked_tools = _all_names - _known_tools if _repo_lang is not None else frozenset()
            else:
                _known_tools = frozenset(
                    str(t.get("name")) for t in (self.registry.get_tool_schemas() or []) if t.get("name")
                )
                _masked_tools = frozenset()
        except (AttributeError, TypeError, KeyError):
            _known_tools = frozenset()
            _masked_tools = frozenset()

        for call in tool_calls:
            if not isinstance(call, dict):
                logger.warning("Skipping non-dict tool call: %r", call)
                continue

            tool_name: str = ""
            tool_args: dict[str, Any] = {}

            if call.get("name"):
                tool_name = str(call.get("name") or "").strip()
                tool_args = call.get("args") or {}

            elif call.get("tool"):
                tool_name = str(call.get("tool") or "").strip()
                tool_args = call.get("args") or {}

            elif isinstance(call.get("function"), dict):
                fn = call.get("function") or {}
                tool_name = str(fn.get("name") or "").strip()
                raw_args = fn.get("arguments")

                if isinstance(raw_args, dict):
                    tool_args = raw_args
                elif isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                        # Only a dict is usable as tool args; a list/scalar JSON
                        # payload is dropped here (same as the old inline path).
                        tool_args = parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        # Salvage via the shared contract in output_parser —
                        # single source of truth for the find/rfind JSON recovery.
                        # (The inline copy previously drifted; its extra
                        # "__parse_error" key had zero consumers.)
                        tool_args = parse_tool_args(raw_args)
                else:
                    tool_args = {}

            if not isinstance(tool_args, dict):
                tool_args = {}

            if not tool_name:
                logger.warning("Skipping tool call with empty name: %r", call)
                continue

            if _masked_tools and tool_name in _masked_tools:
                logger.warning(
                    "Skipping Python-only tool call '%s' (repo is %s)",
                    tool_name,
                    getattr(_repo_lang, "value", _repo_lang),
                )
                _unknown_tool_notices.append(
                    f"Tool `{tool_name}` is a Python-only tool and is not available "
                    f"in this {getattr(_repo_lang, 'value', _repo_lang)} repository. "
                    "Do not call it again. Use a language-native approach or a "
                    "different available tool instead."
                )
                continue

            if _known_tools and tool_name not in _known_tools:
                logger.warning("Skipping unknown tool call '%s': %r", tool_name, call)
                _mode = "read-only" if read_only_request else "current"
                _unknown_tool_notices.append(
                    f"Tool `{tool_name}` is not available in {_mode} mode. "
                    "Do not call write/edit tools. "
                    "Only read_symbol, find_symbol, bash (cat/grep), or similar read-only tools are allowed. "
                    "Answer based on what you have already found."
                )
                continue

            call_id = str(call.get("id") or f"call_{turn_num}_{tool_name}")
            prepared_calls.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "call_id": call_id,
                    "original_call": call,
                }
            )

            logger.info("Tool call (pending): %s(%s)", tool_name, list(tool_args.keys()))

        phase_rule_messages: list[LLMMessage] = [
            LLMMessage(role="user", content=f"[PHASE RULE] {n}") for n in _unknown_tool_notices
        ]
        # The phase/read-only pass through _filter_prepared_calls used to sit
        # here. It never removed a call and never produced a notice, so the
        # block that consumed its output was dead as well; see the note where
        # that method was defined (agent_phase_manager) for why it is gone
        # rather than implemented. Unknown / language-masked tools are still
        # filtered above, and those notices still become [PHASE RULE] messages.

        if not prepared_calls:
            return _PreparedCallsResult(
                prepared_calls=prepared_calls,
                phase_rule_messages=phase_rule_messages,
                plan_current_index=plan_current_index,
                should_continue=True,
            )

        if self.config.stream_callback:
            for _pc in prepared_calls:
                try:
                    self._cb(
                        "tool_call_preview",
                        {
                            "turn": turn_num,
                            "tool": _pc["tool"],
                            "args": self.registry.normalize_args_for_display(_pc["args"]),
                            "agent_id": self.config.agent_id,
                        },
                    )
                except (AttributeError, TypeError):
                    logger.debug(
                        "<module>::TurnPipelineMixin::_build_and_filter_prepared_calls:4 suppressed (AttributeError, TypeError)",
                        exc_info=True,
                    )

        if self.config.cancel_event and self.config.cancel_event.is_set():
            raise AgentCancelled("cancelled by user before tool execution")

        return _PreparedCallsResult(
            prepared_calls=prepared_calls,
            phase_rule_messages=phase_rule_messages,
            plan_current_index=plan_current_index,
        )

    # ------------------------------------------------------------------
    # Process tool results
    # ------------------------------------------------------------------

    def _process_tool_results(
        self,
        results: list,
        prepared_calls: list,
        new_messages: list,
        write_tool_used: bool,
        reads_since_last_edit: int,
        fail_streak: dict,
        fail_streak_threshold: int,
        session_key: str,
        write_tools: set,
        read_only_request: bool,
        request: str,
        session_id: str,
        git_state: Any,
        turn_num: int,
        turns: list,
    ) -> _ResultsProcessingOutcome:
        """Process tool call results: track writes, early-finish, chaining, fail-loop, SSE emit."""
        _noop_confirmed = False

        for i, pc in enumerate(prepared_calls):
            tool_name = pc["tool"]
            tool_args = pc["args"]
            if tool_name in write_tools:
                write_tool_used = True
            call_id = pc["call_id"]
            result = results[i]

            # Record tool-call metric to the per-loop collector for accurate
            # per-turn summary isolation. The global collector (dashboard) is
            # fed by the dispatch wrapper in tool_registry.py — separate sinks
            # so concurrent sessions do not contaminate each other's summary.
            #
            # 3-state cache outcome (mirrors tool_registry.dispatch): only
            # cacheable (read-only) tools are probed, so a write/serial tool
            # emits ``None`` and contributes NEITHER a hit nor a miss — otherwise
            # its per-tool cache_hit_rate would be structurally faked to 0%.
            # Cacheability is the registry's SSOT (is_result_cacheable), not
            # ``write_tools`` (which omits serial-only tools like ask_user/job).
            if result.metadata and result.metadata.get("cache_hit"):
                _cache_outcome: bool | None = True
            elif self.registry and self.registry.is_result_cacheable(tool_name):
                _cache_outcome = False
            else:
                _cache_outcome = None
            self.performance_collector.record_tool_call(
                tool_name,
                result.execution_time,
                cache_hit=_cache_outcome,
                failed=not result.ok,
            )

            # Read-only early finish detection
            if read_only_request and not write_tool_used:
                early_result = self._try_readonly_early_finish(tool_name, result, request, read_only_request)
                if early_result is not None:
                    if self.config.stream_callback:
                        content_limit = (
                            8000
                            if tool_name in _STREAM_LARGE_TOOLS
                            else (6000 if tool_name in _STREAM_VERBOSE_TOOLS else 2000)
                        )
                        try:
                            self._cb(
                                "tool_call",
                                {
                                    "turn": turn_num,
                                    "tool": tool_name,
                                    "args": self.registry.normalize_args_for_display(tool_args),
                                    "result": {
                                        "ok": result.ok,
                                        "content": result.content[:content_limit],
                                        "error": result.error,
                                    },
                                    "agent_id": self.config.agent_id,
                                },
                            )
                        except Exception:
                            logger.debug(
                                "<module>::TurnPipelineMixin::_process_tool_results:0 suppressed Exception",
                                exc_info=True,
                            )
                    # `turns` is ctx.turns, which the caller already appended
                    # this tool call to before invoking _process_tool_results.
                    # Building another AgentTurn here would double-count it.
                    early_result.turns = list(turns)
                    early_result.metadata.update(
                        {
                            "turns_used": turn_num,
                            "readonly_early_finish": True,
                            "deterministic_tool": tool_name,
                        }
                    )
                    self.performance_collector.end_session()
                    performance_summary = self.performance_collector.get_summary()
                    early_result.metadata["performance"] = performance_summary
                    early_result.metadata["session_id"] = session_id
                    early_result.metadata["git_state"] = git_state
                    return _ResultsProcessingOutcome(
                        new_messages=new_messages,
                        write_tool_used=write_tool_used,
                        reads_since_last_edit=reads_since_last_edit,
                        noop_confirmed=_noop_confirmed,
                        fail_streak=fail_streak,
                        early_return=early_result,
                    )

            # Tool-call metrics are recorded at two sites, but to DIFFERENT
            # collectors, so there is no double-counting for any single
            # consumer:
            #   1) This pipeline loop → self.performance_collector (per-loop,
            #      session-isolated, feeds the per-turn summary).
            #   2) The dispatch wrapper (tool_registry.py) → global collector,
            #      feeds the webapp dashboard.
            # The single-choke-point principle applies per sink. Concurrent
            # sessions each have their own per-loop collector, so no
            # contamination.

            _loop_key = make_tool_signature(tool_name, tool_args)
            # Settle any recall hint fired on a PRIOR tool result this run: if a
            # [RECALL] nudge fired on the previous failure, did the LLM recover
            # on this call?  Must run before recall_on_failure arms a new marker
            # below so the two never tangle.
            try:
                from .failure_pattern_store import record_recall_outcome

                record_recall_outcome(ok=result.ok, session_key=session_key)
            except Exception:  # recall bookkeeping must not break the pipeline
                logger.debug("<module>::TurnPipelineMixin::_process_tool_results:1 suppressed Exception", exc_info=True)
            if not result.ok:
                classification = self._failure_classifier.classify(tool_name, result)
                try:
                    result.metadata = result.metadata or {}
                    result.metadata["failure_classification"] = {
                        "action": classification.action,
                        "reason": classification.reason,
                    }
                except (AttributeError, TypeError):
                    logger.debug(
                        "<module>::TurnPipelineMixin::_process_tool_results:2 suppressed (AttributeError, TypeError)",
                        exc_info=True,
                    )
                # ── Per-repo persistent failure recall ─────────────────────────
                # recall_on_failure is the single shared hook (also used by the
                # CLI design-chat loop): classify → in-session dedup → record →
                # recall.  Centralising it guarantees both surfaces record with the
                # same FailureClassifier vocabulary, so the repo-local store (shared
                # across surfaces) stays consistent.  In-session dedup replaces the
                # old fail_streak==0 gate — the run-local strategy warning below
                # still keys off fail_streak independently.
                try:
                    from .failure_pattern_store import recall_on_failure

                    _recall_hint = recall_on_failure(
                        tool_name,
                        tool_args,
                        result,
                        getattr(self.registry, "repo_root", "") or "",
                        # session_key is a fresh per-run key (new_session_key()
                        # at _run_llm_loop entry) → a new run re-records &
                        # re-hints (matches the old fail_streak==0 gate that
                        # reset each run).
                        session_key=session_key,
                    )
                    if _recall_hint:
                        new_messages.append(LLMMessage(role="user", content=_recall_hint))
                except Exception:  # recall must never break the pipeline
                    logger.debug(
                        "<module>::TurnPipelineMixin::_process_tool_results:3 suppressed Exception", exc_info=True
                    )
                self._tool_retry_counter[tool_name] += 1
                if self._tool_retry_counter[tool_name] >= _TOOL_RETRY_LIMIT:
                    _exhaust_warn = LLMMessage(
                        role="user",
                        content=(
                            f"[STRATEGY WARNING] `{tool_name}` has failed "
                            f"{self._tool_retry_counter[tool_name]} times in a row. "
                            f"Your current approach is not working. Stop trying variations "
                            f"of this tool and switch to a completely different strategy, "
                            f"or provide your final assessment as plain text."
                        ),
                    )
                    new_messages.append(_exhaust_warn)
                    self._cb(
                        "fail_loop_detected",
                        {
                            "turn": turn_num,
                            "tool": tool_name,
                            "streak": self._tool_retry_counter[tool_name],
                            "signal": "tool_exhaustion",
                        },
                    )
                    self._tool_retry_counter[tool_name] = 0
                fail_streak[_loop_key] = fail_streak.get(_loop_key, 0) + 1
                if fail_streak[_loop_key] == fail_streak_threshold:
                    if tool_name == "write_plan":
                        _recovery = (
                            "Do NOT call write_plan with the same arguments again. "
                            "Instead: (1) use find_symbol to locate the target, "
                            "(2) use read_file with start_line/end_line to see the exact text "
                            "— not bash cat/sed, which hide the │N│ leading-whitespace count "
                            "that 'before' must match, "
                            "(3) copy that exact text into 'before' and call write_plan again."
                        )
                    elif tool_name == "apply_patch":
                        _recovery = "Switch to write_plan with edit_blocks instead of apply_patch."
                    else:
                        _recovery = "Try a different tool or a different approach."
                    strategy_warn = LLMMessage(
                        role="user",
                        content=(
                            f"[STRATEGY WARNING] `{tool_name}` failed "
                            f"{fail_streak[_loop_key]} times in a row. "
                            f"STOP retrying. {_recovery}"
                        ),
                    )
                    new_messages.append(strategy_warn)
                    self._cb(
                        "fail_loop_detected",
                        {
                            "turn": turn_num,
                            "tool": tool_name,
                            "streak": fail_streak[_loop_key],
                        },
                    )
            else:
                fail_streak.pop(_loop_key, None)
                self._tool_retry_counter[tool_name] = 0

            self._advance_phase_after_success(
                tool_name,
                tool_args,
                result,
            )

            if tool_name in {"write_plan", "apply_patch"} and not result.ok:
                _err_lower = (result.error or "").lower()
                if (
                    "no-op" in _err_lower
                    or "no change" in _err_lower
                    or "empty diff" in _err_lower
                    or "empty patch" in _err_lower
                    or "compiled to empty" in _err_lower
                ):
                    _noop_confirmed = True
                    logger.info("No-op confirmed via %s empty/no-change error", tool_name)

            if self.config.stream_callback:
                content_limit = (
                    8000 if tool_name in _STREAM_LARGE_TOOLS else (6000 if tool_name in _STREAM_VERBOSE_TOOLS else 2000)
                )
                try:
                    self._cb(
                        "tool_call",
                        {
                            "turn": turn_num,
                            "tool": tool_name,
                            "args": self.registry.normalize_args_for_display(tool_args),
                            "result": {
                                "ok": result.ok,
                                "content": result.content[:content_limit],
                                "error": result.error,
                            },
                            "agent_id": self.config.agent_id,
                        },
                    )
                except (AttributeError, TypeError):
                    logger.debug(
                        "<module>::TurnPipelineMixin::_process_tool_results:4 suppressed (AttributeError, TypeError)",
                        exc_info=True,
                    )

            new_messages.append(self._build_tool_result_message(call_id, tool_name, result, tool_args))

            # Read-only exploration tools. Counting these toward
            # reads_since_last_edit lets the GOAL REMINDER fire when the agent
            # loops on reads (incl. read_symbol/read_file) without editing.
            _exploration_tools = {
                "bash",
                "find_symbol",
                "find_references",
                "find_relevant_files",
                "read_file",
                "read_symbol",
                "get_file_outline",
                "get_project_info",
                "grep",
                "glob",
            }
            if result.ok:
                if tool_name in _exploration_tools:
                    reads_since_last_edit += 1
                if tool_name in write_tools:
                    # Reset the read counter on ANY successful write — not just
                    # apply_patch/write_plan. edit_text/edit_file/modify_symbol/
                    # anchor_edit/edit_ast are equally "an edit happened", so the
                    # GOAL REMINDER must not misfire while the agent is actively
                    # editing via those tools.
                    reads_since_last_edit = 0
                    # Conditionally invalidate the test-impact index cache (600 s TTL).
                    # Only invalidate when the write touched a test file — otherwise
                    # every edit in a busy session kills the cache, forcing repeated
                    # full-dir walks for no benefit.
                    # The predicate _write_touched_test_file handles four argument
                    # layouts: direct "path"/"file_path", apply_patch "patch" text
                    # extraction, and write_plan "plan" normalisation (dict / JSON
                    # string / list). edit_text/modify_symbol/edit_ast/anchor_edit
                    # carry their target under "file_path".
                    try:
                        if getattr(self.config, "scoped_verification", False):
                            from .test_impact_selector import invalidate_index

                            if _write_touched_test_file(tool_name, tool_args):
                                _rr = getattr(self.registry, "repo_root", None)
                                if _rr:
                                    invalidate_index(_rr)
                    except Exception:  # must never break the pipeline
                        logger.debug(
                            "<module>::TurnPipelineMixin::_process_tool_results:5 suppressed Exception", exc_info=True
                        )

        return _ResultsProcessingOutcome(
            new_messages=new_messages,
            write_tool_used=write_tool_used,
            reads_since_last_edit=reads_since_last_edit,
            noop_confirmed=_noop_confirmed,
            fail_streak=fail_streak,
        )

    def _settle_deferred_semantics(self, new_messages: list) -> None:
        """Run this turn's coalesced semantic checks and inject the results.

        ``_run_syntax_check_for_file`` deferred each written file rather than
        spawning a toolchain process per write (see
        ``ToolRegistry.begin_semantic_turn``). Now that the turn's writes have
        all landed, one check per file runs against the final content and its
        diagnostics are written into the tool-result message of that file's LAST
        write — the one whose edit produced the state being reported.

        The message content is a JSON payload (``_build_tool_result_message``),
        so the diagnostics are injected by re-serialising it rather than by
        appending text, which would corrupt the payload.

        Earlier messages for the same file keep their ``semantic_deferred`` flag
        and gain no ``semantic_diagnostics`` key: an empty list there would be
        rendered as "checked, clean" by ``_append_semantic_diagnostics``, which
        is the exact miscue this design exists to avoid.

        Advisory throughout — any failure leaves the messages untouched.
        """
        try:
            diags = self.registry.drain_pending_semantic_checks()
        except Exception as exc:  # never break the turn over a lint
            # Logged, not silent: if this ever starts failing, the whole
            # advisory channel goes dark and the agent stops hearing about the
            # errors its own edits introduce. That must be discoverable.
            logger.warning("Deferred semantic checks could not be drained: %s", exc)
            return
        if not diags:
            return
        # Every deferring message is visited, not just enough of them to fill
        # each path once: the earlier writes still carry the internal
        # ``semantic_deferred_path`` key, and stopping early would ship it to
        # the model.
        _filled: set = set()
        for _msg in reversed(new_messages):
            _raw = getattr(_msg, "content", None)
            if not isinstance(_raw, str) or "semantic_deferred" not in _raw:
                continue
            try:
                _payload = json.loads(_raw)
                _syn = (_payload.get("metadata") or {}).get("syntax_check")
                if not isinstance(_syn, dict):
                    continue
                _path = _syn.pop("semantic_deferred_path", None)
                if _path is None or _path not in diags:
                    continue
                if _path in _filled:
                    # An earlier write to the same file: the reported state
                    # belongs to the later one, so leave this message unfilled.
                    _msg.content = json.dumps(_payload, ensure_ascii=False)
                    continue
                _syn.pop("semantic_deferred", None)
                _outcome = diags[_path]
                if not _outcome.checked:
                    # Nothing examined the file — say so instead of writing an
                    # empty diagnostics list, which reads as a clean check.
                    # Still marked filled: the model has been told the truth
                    # about this file, and the earlier writes must not be
                    # re-answered with the same notice.
                    _syn["semantic_check_skipped"] = _outcome.skip_reason
                    _filled.add(_path)
                    _msg.content = json.dumps(_payload, ensure_ascii=False)
                    continue
                _syn["semantic_diagnostics"] = _outcome.diagnostics
                # Render the SAME <file_diagnostics> guidance block the inline
                # path produces, so a coalesced check is as salient to the model
                # as an inline one — not buried only in the JSON metadata. The
                # inline path appended it during _build_tool_result_message,
                # which ran BEFORE this settle, so it must be re-created here.
                _block = render_file_diagnostics_block(_outcome.diagnostics)
                if _block:
                    _payload["content"] = (_payload.get("content") or "") + _block
                _filled.add(_path)
                _msg.content = json.dumps(_payload, ensure_ascii=False)
            except Exception as exc:  # a single bad payload is not fatal
                logger.debug("Could not settle semantics into a tool message: %s", exc)
                continue

    # ------------------------------------------------------------------
    # Execute and process tool calls
    # ------------------------------------------------------------------

    def _execute_and_process_tool_calls(
        self,
        ctx: TurnContext,
        tool_calls: list,
    ) -> _ToolTurnOutcome:
        """Prepare, execute, and process tool calls for one LLM turn.

        Returns _ToolTurnOutcome. If early_return is set, caller should return it immediately.
        If should_continue is set, caller should update messages with phase_rule_messages and continue.
        """
        _noop_confirmed = False
        new_messages: list[LLMMessage] = []

        plan_current_index = ctx.plan_current_index
        any_tool_called = ctx.any_tool_called
        write_tool_used = ctx.write_tool_used
        reads_since_last_edit = ctx.reads_since_last_edit
        fail_streak = ctx.fail_streak
        fail_streak_threshold = config.counts.AGENT_FAIL_LOOP_LARGE

        _pcr = self._build_and_filter_prepared_calls(
            tool_calls=tool_calls,
            turns=ctx.turns,
            plan_subtasks=ctx.plan_subtasks,
            plan_current_index=plan_current_index,
            read_only_request=ctx.read_only_request,
            turn_num=ctx.turn_num,
        )
        if _pcr.should_continue:
            return _ToolTurnOutcome(
                new_messages=new_messages,
                prepared_calls=_pcr.prepared_calls,
                write_tool_used=write_tool_used,
                any_tool_called=any_tool_called,
                fail_streak=fail_streak,
                reads_since_last_edit=reads_since_last_edit,
                plan_current_index=_pcr.plan_current_index,
                should_continue=True,
                phase_rule_messages=_pcr.phase_rule_messages,
            )
        prepared_calls = _pcr.prepared_calls
        plan_current_index = _pcr.plan_current_index

        # Parallel execution if enabled
        if (
            hasattr(self.config, "parallel_tool_execution_enabled")
            and self.config.parallel_tool_execution_enabled
            and len(prepared_calls) > 1
        ):
            parallel_calls = [{"tool": pc["tool"], "args": pc["args"]} for pc in prepared_calls]
            try:
                results = self.registry.dispatch_parallel(parallel_calls)
            except StopIteration:
                logger.exception("Tool dispatch_parallel StopIteration (mock side_effect exhausted)")
                results = [ToolResult(ok=False, content="", error="StopIteration", metadata={}) for _ in parallel_calls]
            # Parity with the serial branch: a multi-call turn still counts as
            # "tools called" for the false-success gate, must feed the
            # adaptive-routing success/failure memory channels, and failed
            # apply_patch results climb the same auto-repair / tolerant
            # edit_blocks recovery ladder (previously skipped entirely when a
            # turn carried 2+ tool calls).
            any_tool_called = True
            for _i, _pc in enumerate(prepared_calls):
                _r = results[_i]
                try:
                    if _r and getattr(_r, "ok", False):
                        self._record_tool_success(_pc["tool"], _pc["args"])
                    else:
                        self._record_tool_failure(_pc["tool"], _pc["args"])
                except (AttributeError, TypeError):
                    logger.debug(
                        "<module>::TurnPipelineMixin::_execute_and_process_tool_calls:2 suppressed (AttributeError, TypeError)",
                        exc_info=True,
                    )
                results[_i] = self._post_dispatch_patch_recovery(_pc["tool"], _pc["args"], _r)
            _log_parallel_write_failures(results, parallel_calls, self, session_key=ctx.recall_session_key)
        else:
            results = []
            for pc in prepared_calls:
                tool_name = pc["tool"]
                tool_args = pc["args"]
                try:
                    result = self.registry.dispatch(tool_name, tool_args)
                    any_tool_called = True

                    try:
                        if result and getattr(result, "ok", False):
                            self._record_tool_success(tool_name, tool_args)
                        else:
                            self._record_tool_failure(tool_name, tool_args)
                        # Persist write-tool outcomes to JSONL for analysis and
                        # settle/fire the suggestion-hit tracker. Called on EVERY
                        # write-tool result: the record helper no-ops on success,
                        # but the settle must run on success too so a suggestion
                        # fired by an earlier failure settles as helped/ignored.
                        if tool_name in self.registry._WRITE_TOOLS:
                            try:
                                from .tool_failure_log import (
                                    record_write_tool_failure_from_tr,
                                )

                                record_write_tool_failure_from_tr(
                                    tool=tool_name,
                                    tr=result,
                                    args=tool_args,
                                    model=getattr(self.config, "model", None),
                                    repo_root=getattr(self.registry, "repo_root", None),
                                    session_key=ctx.recall_session_key,
                                )
                            except Exception:
                                logger.debug("tool_failure_log: record failed", exc_info=True)
                    except (AttributeError, TypeError):
                        logger.debug(
                            "<module>::TurnPipelineMixin::_execute_and_process_tool_calls:2 suppressed (AttributeError, TypeError)",
                            exc_info=True,
                        )
                except StopIteration:
                    logger.exception(
                        "Tool dispatch StopIteration (mock side_effect exhausted): %s",
                        tool_name,
                    )
                    result = ToolResult(ok=False, content="", error="StopIteration", metadata={})

                # Auto-repair + tolerant edit_blocks fallback — the SAME ladder
                # the parallel branch runs, so a failed apply_patch recovers
                # identically whether it was batched with other tools or not.
                result = self._post_dispatch_patch_recovery(tool_name, tool_args, result)
                results.append(result)

        # Record one AgentTurn per executed tool call. Both the parallel and
        # serial branches above converge here with `results` aligned to
        # `prepared_calls`, and `result` is final (post auto-repair / retry /
        # edit_blocks fallback) — so this is the single point that sees every
        # tool call exactly once.
        #
        # Until now AgentTurn was built ONLY in the read-only early-finish
        # branch below, so AgentResult.turns came back empty from every normal
        # MAIN_AGENT run however many tools ran (11 dispatched -> turns == []),
        # while metadata["turns_used"] reported the real count. Consumers that
        # read .turns — intelligent_service's turn_summary and its
        # "turns_used": len(turns) — therefore always saw nothing.
        for _pc, _res in zip(prepared_calls, results, strict=False):
            ctx.turns.append(
                AgentTurn(
                    turn_num=ctx.turn_num,
                    tool_name=_pc["tool"],
                    tool_args=_pc["args"],
                    tool_result=_res,
                )
            )

        _rpr = self._process_tool_results(
            results=results,
            prepared_calls=prepared_calls,
            new_messages=new_messages,
            write_tool_used=write_tool_used,
            reads_since_last_edit=reads_since_last_edit,
            fail_streak=fail_streak,
            fail_streak_threshold=fail_streak_threshold,
            session_key=ctx.recall_session_key,
            write_tools=ctx.write_tools,
            read_only_request=ctx.read_only_request,
            request=ctx.request,
            session_id=ctx.session_id,
            git_state=ctx.git_state,
            turn_num=ctx.turn_num,
            turns=ctx.turns,
        )
        # Every write of this turn has landed, so the files are now in their
        # final state: run the coalesced semantic checks and fill the results
        # into the tool-result messages that deferred them. Done here rather
        # than inside _process_tool_results because it must observe the LAST
        # write, and that is only known once the loop over results is over.
        self._settle_deferred_semantics(_rpr.new_messages)
        if _rpr.early_return is not None:
            return _ToolTurnOutcome(
                new_messages=_rpr.new_messages,
                prepared_calls=prepared_calls,
                write_tool_used=_rpr.write_tool_used,
                any_tool_called=any_tool_called,
                fail_streak=_rpr.fail_streak,
                reads_since_last_edit=_rpr.reads_since_last_edit,
                plan_current_index=plan_current_index,
                early_return=_rpr.early_return,
            )
        new_messages = _rpr.new_messages
        write_tool_used = _rpr.write_tool_used
        reads_since_last_edit = _rpr.reads_since_last_edit
        _noop_confirmed = _rpr.noop_confirmed
        fail_streak = _rpr.fail_streak

        return _ToolTurnOutcome(
            new_messages=new_messages,
            prepared_calls=prepared_calls,
            write_tool_used=write_tool_used,
            any_tool_called=any_tool_called,
            fail_streak=fail_streak,
            reads_since_last_edit=reads_since_last_edit,
            plan_current_index=plan_current_index,
            noop_confirmed=_noop_confirmed,
        )

    # ------------------------------------------------------------------
    def _post_dispatch_patch_recovery(self, tool_name: str, tool_args: dict, result) -> Any:
        """Apply the apply_patch recovery ladder to a dispatched result.

        Two stages, both apply_patch-only:
          1. auto-repair (max 1 retry) via ``_auto_repair_apply_patch_args``;
          2. tolerant-patch mode: count consecutive failures and, at the
             threshold, auto-convert the patch to edit_blocks via PatchEngine
             and dispatch it as write_plan.

        Shared by the serial and parallel dispatch branches in
        ``_execute_and_process_tool_calls`` so a failed apply_patch gets the
        same recovery whether it ran alone or was batched with other tools in
        one turn. Returns the (possibly replaced) final result.
        """
        if tool_name != "apply_patch":
            return result
        if not result.ok:
            new_args = self._auto_repair_apply_patch_args(tool_args)
            if new_args:
                logger.debug("Auto-repair for apply_patch: attempting repair")
                # Capture the failure cause BEFORE retry_result replaces
                # `result`; on success result.error becomes None and the
                # original error would otherwise be lost from metadata.
                _orig_error = result.error
                retry_result = self.registry.dispatch(tool_name, new_args)
                if retry_result.ok:
                    result = retry_result
                    result.metadata["auto_repair"] = {
                        "attempted": True,
                        "kind": "patch_format_fix",
                        "original_error": _orig_error,
                        "success": True,
                    }
                else:
                    result.metadata["auto_repair"] = {
                        "attempted": True,
                        "kind": "patch_format_fix",
                        "original_error": _orig_error,
                        "success": False,
                        "retry_error": retry_result.error,
                    }
        if result.ok:
            self._patch_fail_count = 0
        else:
            self._patch_fail_count += 1
            max_failures = getattr(self.config, "tolerant_patch_max_failures", 2)
            if getattr(self.config, "tolerant_patch_mode", False) and self._patch_fail_count >= max_failures:
                patch_text = tool_args.get("patch", "")
                path_hint = tool_args.get("path")
                eb_result = None
                # patch_engine is first-party — import cannot fail.
                from ..patch_engine import PatchEngine

                if PatchEngine is not None:
                    try:
                        engine = PatchEngine(self.registry.repo_root)
                        converted = engine.convert_patch_to_edit_blocks(patch_text, path_hint)
                        if converted:
                            plan = {
                                "kind": "ASICODE_PLAN_V1",
                                "ops": [
                                    {"op": "edit_blocks", "path": converted["file_path"], "blocks": converted["blocks"]}
                                ],
                            }
                            plan_str = json.dumps(plan, ensure_ascii=False)
                            eb_result = self.registry.dispatch("write_plan", {"plan": plan_str})
                            if eb_result.ok:
                                eb_result.metadata["auto_converted_from_patch"] = True
                                eb_result.metadata["edit_blocks_count"] = len(converted["blocks"])
                                eb_result.content = (
                                    f"Patch auto-converted to edit_blocks and applied successfully "
                                    f"({len(converted['blocks'])} block(s) in {converted['file_path']}).\n"
                                    + (eb_result.content or "")
                                )
                    except Exception as e:
                        logger.debug("PatchEngine convert_patch_to_edit_blocks failed: %s", e)
                        eb_result = None
                if eb_result is not None:
                    if eb_result.ok:
                        logger.info(
                            "edit_blocks auto-conversion succeeded after %d patch failures",
                            self._patch_fail_count,
                        )
                        self._patch_fail_count = 0
                        result = eb_result
                    else:
                        logger.debug("edit_blocks auto-conversion also failed: %s", eb_result.error)
                        result.metadata["edit_blocks_fallback_error"] = eb_result.error
        return result

    # ------------------------------------------------------------------
    # Loop cancellation handler
    # ------------------------------------------------------------------
    # Loop cancellation handler
    # ------------------------------------------------------------------

    def _handle_loop_cancellation(
        self,
        turns: list,
        git_state: Any,
    ) -> AgentResult:
        """Handle AgentCancelled: rollback patches and return cancelled result."""
        logger.info("Agent execution cancelled")
        self._cb("cancelled", {"message": "Agent execution cancelled by user"})

        rollback_performed = False
        rollback_result = None

        if self.registry.applied_patches:
            logger.info("Attempting to rollback %d applied patches", len(self.registry.applied_patches))
            rollback_result = self._rollback_patches(self.registry.applied_patches)
            rollback_performed = True

            if rollback_result["success"]:
                logger.info(
                    "Successfully rolled back %d/%d patches", rollback_result["rolled_back"], rollback_result["total"]
                )
                # Clear applied_patches after successful rollback so DIFF_VERIFY
                # does not falsely warn that git diff is empty.
                self.registry.applied_patches.clear()
            else:
                logger.warning(
                    "Partial or failed rollback: %d/%d patches rolled back",
                    rollback_result["rolled_back"],
                    rollback_result["total"],
                )
                for i, result in enumerate(rollback_result.get("results", [])):
                    if not result.get("success"):
                        logger.error(
                            "Rollback failed for patch %d: %s",
                            result.get("patch_index", i),
                            result.get("message", "unknown error"),
                        )
                # Partial rollback: keep applied_patches so DIFF_VERIFY and
                # downstream callers know some changes are still in the working tree.
        else:
            logger.info("No patches to rollback")
            rollback_result = {"success": True, "message": "No patches to rollback", "rolled_back": 0}

        rollback_msg, rollback_meta = _summarize_rollback(rollback_result if rollback_performed else None)

        self.performance_collector.end_session()
        performance_summary = self.performance_collector.get_summary()

        return AgentResult(
            status="cancelled",
            turns=turns,
            final_message=f"Agent execution cancelled. {rollback_msg}",
            applied_patches=list(self.registry.applied_patches),
            metadata={
                "turns_used": len(turns),
                "git_state": git_state,
                "rollback": rollback_meta,
                "performance": performance_summary,
            },
        )

    # ------------------------------------------------------------------
    # Loop error handler
    # ------------------------------------------------------------------

    def _handle_loop_error(
        self,
        error: Exception,
        turns: list,
        git_state: Any,
        rollback_performed: bool,
        rollback_result: Any,
    ) -> AgentResult:
        """Handle unexpected Exception: rollback patches and return error result."""
        logger.exception("Unexpected error in agent loop")

        # Turn-level outcome channel: this loop dies with status="error".
        # The typed LLM-error path in the turn body records failed=True for
        # the 5 known client errors; anything escaping to here (other
        # LLMClientError subclasses, unexpected exceptions) must not vanish
        # from failure_rate / the recent-outcome window.
        self.performance_collector.record_agent_result(failed=True)
        get_global_collector().record_agent_result(failed=True)

        if isinstance(error, LLMConnectionError):
            error_type = "connection"
        elif isinstance(error, LLMRateLimitError):
            error_type = "rate_limit"
        elif isinstance(error, LLMServerUnavailableError):
            error_type = "server_unavailable"
        else:
            error_type = "api"

        self._cb("error", {"message": f"Unexpected error in agent loop: {error}", "error_type": error_type})

        if self.registry.applied_patches:
            logger.info("Attempting to rollback %d applied patches due to error", len(self.registry.applied_patches))
            rollback_result = self._rollback_patches(self.registry.applied_patches)
            rollback_performed = True

            if rollback_result["success"]:
                logger.info(
                    "Successfully rolled back %d/%d patches", rollback_result["rolled_back"], rollback_result["total"]
                )
                # Clear applied_patches after successful rollback so DIFF_VERIFY
                # does not falsely warn that git diff is empty.
                self.registry.applied_patches.clear()
            else:
                logger.warning(
                    "Partial or failed rollback: %d/%d patches rolled back",
                    rollback_result["rolled_back"],
                    rollback_result["total"],
                )

        self.performance_collector.end_session()
        performance_summary = self.performance_collector.get_summary()

        rollback_msg, rollback_meta = _summarize_rollback(rollback_result if rollback_performed else None)

        return AgentResult(
            status="error",
            turns=turns,
            error=f"Unexpected error: {error}. {rollback_msg}" if rollback_performed else f"Unexpected error: {error}",
            applied_patches=list(self.registry.applied_patches),
            metadata={
                "turns_used": len(turns),
                "git_state": git_state,
                "rollback": rollback_meta,
                "performance": performance_summary,
            },
        )


# ------------------------------------------------------------------
# Module-level helper: evict consumed tool results
# ------------------------------------------------------------------


def _summarize_rollback(rollback_result: Any) -> tuple:
    """Build a human-readable message + structured metadata from a rollback result.

    Surfaces the ``needs_manual_rollback`` signal produced by
    ``AgentLoop._rollback_patches`` when a shared-tree conflict prevents the
    automatic reverse-apply (so the operator/LLM learns a targeted manual revert
    is required instead of seeing only a generic "partially failed" message).

    Returns ``(rollback_msg, rollback_meta)`` where ``rollback_meta`` is the dict
    intended for ``AgentResult.metadata["rollback"]`` (top-level
    ``needs_manual_rollback`` + ``affected_files`` are promoted out of the
    per-patch ``results`` for easy inspection).
    """
    if not rollback_result:
        return (
            "No patches needed rollback.",
            {"performed": False, "result": None, "needs_manual_rollback": False, "affected_files": []},
        )

    results_list = rollback_result.get("results", []) if isinstance(rollback_result, dict) else []
    needs_manual = [r for r in results_list if isinstance(r, dict) and r.get("needs_manual_rollback")]
    affected = []
    seen = set()
    for r in needs_manual:
        for f in r.get("affected_files") or []:
            if f not in seen:
                seen.add(f)
                affected.append(f)

    if rollback_result.get("success"):
        msg = "All applied patches were successfully rolled back."
    else:
        rolled = rollback_result.get("rolled_back", 0)
        total = rollback_result.get("total", 0)
        msg = f"Rollback partially failed: {rolled}/{total} patches rolled back."
        if needs_manual:
            files_str = ", ".join(affected) if affected else "(unparseable)"
            msg += (
                f" Automatic rollback was aborted for {len(needs_manual)} patch(es) "
                f"to protect concurrent edits on shared file(s): {files_str}. "
                f"Manual targeted rollback required."
            )

    meta = {
        "performed": True,
        "result": rollback_result,
        "needs_manual_rollback": bool(needs_manual),
        "affected_files": affected,
    }
    return (msg, meta)


def _log_parallel_write_failures(results, parallel_calls, pipeline, session_key=""):
    """Persist write-tool outcomes produced by ``dispatch_parallel``.

    ``dispatch_parallel`` bypasses the serial loop where the per-call failure
    logging lives, so parallel write-tool failures would otherwise escape the
    JSONL forensic log. This walks the parallel ``results`` and records every
    write tool (the record helper no-ops on success). The suggestion-hit
    settle must also run on success, hence the unconditional walk. Best-effort
    — never raises.
    """
    try:
        from .tool_failure_log import record_write_tool_failure_from_tr

        write_tools = pipeline.registry._WRITE_TOOLS
        for idx, result in enumerate(results):
            tool_name = parallel_calls[idx]["tool"] if idx < len(parallel_calls) else ""
            if tool_name not in write_tools:
                continue
            record_write_tool_failure_from_tr(
                tool=tool_name,
                tr=result,
                args=parallel_calls[idx].get("args"),
                model=getattr(pipeline.config, "model", None),
                repo_root=getattr(pipeline.registry, "repo_root", None),
                session_key=session_key,
            )
    except Exception:
        logger.debug("tool_failure_log: parallel record failed", exc_info=True)


# Marker prefix stamped onto stubbed tool_result content. Doubles as the
# idempotency guard so already-evicted results are not re-processed.
_EVICTED_MARKER = "[EVICTED TOOL OUTPUT"
# Tool-result content at or below this length is left verbatim: stubbing it
# would grow context (the stub itself is ~80 chars) for negligible savings.
_EVICT_MIN_CONTENT_LEN = 200
# ── Eviction trigger: context OCCUPANCY, not turn-count / cost ───────────────
# ``_EVICTION_KEEP_RECENT``: most-recent tool_results always kept verbatim —
# the live working set. This is the QUALITY floor and is model-independent.
#
# Design rationale (why occupancy, not a cost-tuned turn-count hysteresis):
# eviction's ONLY unconditional benefit is bounding context so the window never
# overflows. Its cost effect is marginal, provider-dependent, and often NEGATIVE
# (each firing invalidates the cached prefix → a one-time cache-WRITE), so tuning
# a firing *cadence* to a per-model cost model (a) needs an always-current price
# table = maintenance burden, and (b) makes eviction fire PROACTIVELY, minting
# self-inflicted cache-miss spikes even when nowhere near the window. We removed
# that machinery. Eviction now fires ONLY when the estimated prompt approaches
# the model's effective cap — i.e. only when we'd otherwise hit the wall, where
# a rewrite is unavoidable anyway. Below the trigger, every tool result is kept
# verbatim and the prefix cache stays warm (cheap reads, no spikes).
#
# ``_EVICTION_OCCUPANCY_TRIGGER``: fire once estimated tokens exceed this
# fraction of ``context_message_cap`` (the SAME accounting the pre-flight
# guards use). Sitting below the cap means the gentle stub-based bound here
# preempts the provider's context-length 400 (preemptive_trim removed — an
# over-cap turn now surfaces as a provider error), which would invalidate the
# ENTIRE prefix. 0.75 leaves headroom for the current turn to grow.
#
# Prefix-stability invariant (CRITICAL, independent of the trigger mechanism).
# For the prefix cache to stay warm between turns, eviction must be the ONLY
# thing that rewrites the early prefix. It holds today because the design-chat
# loop (a) removed per-iteration re-injection (design_chat_loop.py) and
# (b) injects turn-volatile L3 promoted-insights at a LATE position (`load_promoted_insights`),
# keeping the cached system/insights prefix byte-stable across turns. If a
# future change re-introduces a per-turn-mutated banner/timestamp/state into the
# system prompt or early messages, EVERY turn becomes a full cache miss and the
# warm-cache assumption collapses silently. Treat the LATE-position injection
# rule as the load-bearing defence, not a style preference.
_EVICTION_KEEP_RECENT = 6
_EVICTION_OCCUPANCY_TRIGGER = 0.75
# ── Eviction master switch ───────────────────────────────────────────────────
# Occupancy-gated eviction is DISABLED by default (``False``). It was found to
# mint self-inflicted cache-miss spikes inside the design-chat tool-loop: once
# occupancy crosses ``_EVICTION_OCCUPANCY_TRIGGER``, stubbing older tool_results
# rewrites the cached prefix and the whole tail is re-billed as a cache-WRITE —
# the very cost the gate was meant to save. With the gate off the ONLY context
# bound is the provider's own context limit (``_apply_context_hard_cap`` only
# raises on structural collapse), so a routine loop never pays a prefix rewrite.
#
# The occupancy gate, keep_recent floor and ``_evict_consumed_tool_results``
# primitive are all kept intact and unit-tested, so re-enabling eviction for a
# small-window model that routinely overflows is an env-var flip of this flag
# (ASICODE_TOOL_RESULT_EVICTION=1) — no code change or redeploy required.
_EVICTION_ENABLED = _env_flag("ASICODE_TOOL_RESULT_EVICTION", False)


def _evict_for_loop(messages, model: str = "", tool_schemas=None, base_url: str | None = None):
    """Occupancy-gated tool-result eviction for an in-flight agent tool-loop.

    Single source of truth for the trigger: callers pass only the model
    *identity* (``model``, ``base_url`` — which Ollama server) and the
    ``tool_schemas`` sent alongside the prompt — never a raw threshold — so the
    firing decision lives in exactly ONE place.
    Every production tool-loop (MAIN_AGENT pipeline AND design-chat loop) runs
    the SAME logic, so they can never drift apart.

    DISABLED by default (``_EVICTION_ENABLED is False``): returns ``messages``
    unchanged for every model, so no prefix is ever rewritten by the gentle stub
    bound — only the provider's context limit (via the 400 → overflow-override
    backstop) bounds the window. When enabled, it fires
    ``_evict_consumed_tool_results`` (stub every tool result beyond the
    most-recent ``_EVICTION_KEEP_RECENT``) ONLY when the estimated prompt exceeds
    ``_EVICTION_OCCUPANCY_TRIGGER x context_message_cap(model)``. Below that it
    returns ``messages`` unchanged, so the prefix cache stays warm and no
    proactive cache-miss is minted. Unknown model → 1M default cap → effectively
    off until the prompt is genuinely huge, leaving the hard-cap front-trim as
    the backstop. If occupancy cannot be estimated it skips (never a hard fail).
    """
    if not messages:
        return messages
    if not _EVICTION_ENABLED:
        # Occupancy-gated eviction disabled (see _EVICTION_ENABLED); only the
        # hard-cap front-trim remains as the window-overflow backstop.
        return messages
    try:
        limit = _resolve_context_limit(model or "", base_url=base_url)
        cap = context_message_cap(limit, config.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN, tool_schemas)
        est = estimate_tokens_from_msgs(messages)
    except Exception:
        logger.debug("eviction occupancy estimate failed; skipping", exc_info=True)
        return messages
    if est <= cap * _EVICTION_OCCUPANCY_TRIGGER:
        return messages
    return _evict_consumed_tool_results(messages, keep_recent=_EVICTION_KEEP_RECENT)


def _stub_tool_result(m, name_map: dict | None = None):
    """Return a COPY of tool_result message ``m`` with its content replaced by a
    compact eviction stub, or ``m`` itself when it is already stubbed / too small
    to be worth stubbing.

    Copy-on-write via ``dataclasses.replace``: ``ctx.messages`` entries share
    references with event payloads, continuation-message builders, and run
    records. Mutating ``m.content`` in place would retroactively rewrite a past
    snapshot of the conversation that another consumer still holds. The caller
    reassigns ``ctx.messages`` to the returned list, so only the live
    conversation sees the stub; recorded history keeps the original object.

    Measures BOTH ``content`` (plain str) and ``raw_content`` (provider-native
    blocks — Anthropic ``content[]`` / Gemini ``parts[]``): a multi-part
    tool_result carries its payload in ``raw_content`` with ``content == ""``,
    so ignoring ``raw_content`` leaves eviction a silent no-op for those
    providers.
    """
    content = getattr(m, "content", "")
    raw_content = getattr(m, "raw_content", None)

    size = len(content) if isinstance(content, str) else 0
    if raw_content:
        try:
            size += len(json.dumps(raw_content, ensure_ascii=False))
        except (TypeError, ValueError):
            logger.debug("<module>::_stub_tool_result:0 suppressed (TypeError, ValueError)", exc_info=True)

    if _is_stubbed_tool_result(m) or size <= _EVICT_MIN_CONTENT_LEN:
        return m

    name = getattr(m, "name", "") or "tool"
    tid = getattr(m, "tool_call_id", "") or ""
    tid_suffix = f" ({tid})" if tid else ""
    stub = _eviction_stub(f"{name}{tid_suffix}", size)
    # Copy-on-write: replace returns a NEW dataclass instance, leaving the
    # original message object (and any shared reference to it) intact.
    # ── Anthropic format: per-block stub using name_map (tool_use_id → name) ─
    if _is_anthropic_shape(m):
        return _stub_anthropic_tool_result(m, stub, name_map)
    # ── Gemini format: per-part stub (functionResponse carries its own name) ─
    if _is_gemini_shape(m):
        return _stub_gemini_tool_result(m, stub, name_map)
    # ── Standard format: replace content, clear raw_content ─────────────────
    return dataclasses.replace(m, content=stub, raw_content=None)


def _cache_hit_ratio(ctx=None, *, cache_read_tokens=0, prompt_tokens=0, cache_creation_tokens=0, provider="") -> float:
    """Compute cache hit ratio (0..1), honoring provider token accounting.

    For separate-accounting providers (Anthropic/z.ai) ``prompt_tokens``
    *excludes* cache tokens, so the denominator must include ``cache_read``
    AND ``cache_creation`` to reflect true context size. ``cache_hit_pct``
    (shared) encodes exactly that semantics; we reuse it and convert % → ratio.

    Can be called with a ``ctx`` object (has ``total_*`` fields +
    ``provider_name``) OR with explicit kwargs.
    """
    if ctx is not None:
        cache_read_tokens = ctx.total_cache_read_tokens
        prompt_tokens = ctx.total_prompt_tokens
        cache_creation_tokens = getattr(ctx, "total_cache_creation_tokens", 0) or 0
        provider = ctx.provider_name or ""
    return round(
        cache_hit_pct(provider, prompt_tokens, cache_read_tokens, cache_creation_tokens) / 100.0,
        4,
    )


def _token_metadata(ctx) -> dict:
    """Build ``AgentResult.metadata["tokens"]`` — single source for every exit path.

    Keeps the field set uniform across all statuses so consumers (session log,
    dashboards) never see a path-dependent subset: ``cache_hit_ratio``,
    ``last_call_*`` and ``provider`` are always present, on every status.
    """
    return {
        "prompt": ctx.total_prompt_tokens,
        "completion": ctx.total_completion_tokens,
        "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
        "cost_usd": round(
            estimate_cost(
                ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name
            ),
            6,
        ),
        "cache_adjusted_cost_usd": round(
            estimate_cache_adjusted_cost(
                ctx.provider_name,
                ctx.total_prompt_tokens,
                ctx.total_completion_tokens,
                ctx.total_cache_read_tokens,
                ctx.total_cache_creation_tokens,
                model=ctx.model_name,
                base_url=ctx.base_url,
            ),
            6,
        ),
        "cache_read_tokens": ctx.total_cache_read_tokens,
        "cache_creation_tokens": ctx.total_cache_creation_tokens,
        "cache_hit_ratio": _cache_hit_ratio(ctx),
        "last_call_prompt": ctx.last_call_prompt_tokens,
        "last_call_completion": ctx.last_call_completion_tokens,
        "provider": ctx.provider_name,
    }


def _is_stubbed_tool_result(m) -> bool:
    """Check if ``m`` is an already-evicted (stubbed) tool result.

    OpenAI/standard: ``content.startswith(_EVICTED_MARKER)``.
    Anthropic/native: at least one ``tool_result`` block in ``raw_content``
    whose inner ``content`` starts with ``_EVICTED_MARKER``.
    """
    content = getattr(m, "content", "")
    if isinstance(content, str) and content.startswith(_EVICTED_MARKER):
        return True
    raw_content = getattr(m, "raw_content", None)
    if isinstance(raw_content, list):
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            # Anthropic: {"type": "tool_result", "content": "[EVICTED..."}
            if block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and inner.startswith(_EVICTED_MARKER):
                    return True
            # Gemini: {"functionResponse": {"response": {"content": "[EVICTED..."}}}
            if "functionResponse" in block:
                fr = block["functionResponse"]
                inner = fr.get("response", {}) if isinstance(fr, dict) else {}
                inner = inner.get("content", "") if isinstance(inner, dict) else ""
                if isinstance(inner, str) and inner.startswith(_EVICTED_MARKER):
                    return True
    return False


def _eviction_stub(label: str, size: int) -> str:
    """Build the eviction stub text shared by every provider format."""
    return f"{_EVICTED_MARKER}: {label} — {size} chars evicted to save context; re-read if still needed.]"


def _payload_size(content) -> int:
    """Size of a tool-result payload (str, or JSON-serializable dict/list).

    Shared by the provider-specific block stubbing helpers: both measure the
    *payload of one block* (str length, or ``json.dumps`` of structured
    content) so the eviction stub names a real per-block size.
    """
    if isinstance(content, str):
        return len(content)
    if content is not None:
        return len(json.dumps(content, ensure_ascii=False))
    return 0


def _stub_tool_result_blocks(m, stub: str, replace_block) -> Any:
    """Shared scaffold for provider-native tool-result stubbing.

    Returns a **copy** of ``m`` with every ``raw_content`` block mapped
    through *replace_block* (a callable that returns the stubbed replacement
    for a tool-result block, or the block unchanged otherwise), preserving the
    block structure so the provider does not reject the request.  When
    ``raw_content`` is not a list, the whole message content is replaced by
    *stub* — the standard-format fallback.
    """
    raw_content = getattr(m, "raw_content", None)
    if not isinstance(raw_content, list):
        return dataclasses.replace(m, content=stub, raw_content=None)
    stubbed_raw = [replace_block(b) if isinstance(b, dict) else b for b in raw_content]
    return dataclasses.replace(m, content="", raw_content=stubbed_raw)


def _stub_anthropic_tool_result(m, stub: str, name_map: dict | None = None):
    """Stub an anthropic-format tool result message (``role="user"`` with
    ``raw_content`` containing ``tool_result`` blocks).

    Per-block stubbing (BUG-2 fix): Anthropic batches parallel tool calls'
    results into ONE ``role="user"`` message with N ``tool_result`` blocks.
    The message-level ``name``/``tool_call_id`` attributes are empty for this
    shape, so a single caller-provided *stub* would (a) carry the wrong tool
    name (``"tool"``) for every block and (b) claim the *aggregated* size for
    each block. We instead build a per-block stub from that block's own
    ``tool_use_id`` → name (recovered from the preceding assistant ``tool_use``
    block via *name_map*) and its own content size, so the "re-read" hint names
    the correct tool.  Text blocks (e.g. strategy warnings folded into the
    same message) are left intact.
    """

    def replace_block(block: dict) -> dict:
        if block.get("type") != "tool_result":
            return block
        tid = block.get("tool_use_id", "")
        bname = name_map.get(tid, "") if name_map and tid else ""
        # Size of THIS block's payload only (not the aggregated message size).
        inner = block.get("content", "")
        bsize = _payload_size(inner)
        label = f"{bname} ({tid})" if bname and tid else (bname or tid or "tool")
        return {**block, "content": _eviction_stub(label, bsize)}

    return _stub_tool_result_blocks(m, stub, replace_block)


def _stub_gemini_tool_result(m, stub: str, name_map: dict | None = None):
    """Stub a Gemini-format tool result message (``role="user"`` with
    ``raw_content`` containing ``functionResponse`` parts).

    Gemini ``functionResponse`` parts carry their OWN ``name`` (unlike
    Anthropic, whose ``tool_result`` blocks only carry an opaque
    ``tool_use_id``), so the tool name is read directly from the part.  *name_map*
    is accepted for signature symmetry with the Anthropic handler but is not
    required.
    """

    def replace_block(block: dict) -> dict:
        if "functionResponse" not in block:
            return block
        fr = block["functionResponse"] or {}
        gname = fr.get("name", "") or "tool"
        # Size of THIS part's payload only.  NOTE: when the response content is
        # not a string, the WHOLE response object is measured (not just the
        # content field) — the response dict is the serializable unit here.
        resp = fr.get("response", {})
        rcontent = resp.get("content", "") if isinstance(resp, dict) else ""
        psize = (
            len(rcontent) if isinstance(rcontent, str) else (len(json.dumps(resp, ensure_ascii=False)) if resp else 0)
        )
        return {
            **block,
            "functionResponse": {
                **fr,
                "response": {"content": _eviction_stub(gname, psize)},
            },
        }

    return _stub_tool_result_blocks(m, stub, replace_block)


def _build_tool_name_map(messages) -> dict:
    """Build a ``{tool_use_id: name}`` map from assistant tool-call messages.

    Standard (``role="assistant"`` + ``tool_calls``) and Anthropic-native
    (``tool_use`` blocks) formats are scanned, so per-block eviction stubs can
    recover the tool name for those result shapes. Gemini is intentionally NOT
    mapped: its ``functionResponse`` parts carry their own name, and
    ``functionCall`` parts have no id to key by (a self-referencing name->name
    entry would only pollute the id->name map). Returns an empty dict when
    nothing matches (the stub then falls back to ``"tool"`` / the part's own
    name).
    """
    out: dict = {}
    for m in messages:
        if not is_tool_call(m):
            continue
        # Standard: tool_calls is a list of {"id": ..., "function": {"name": ...}}
        for tc in getattr(m, "tool_calls", None) or []:
            if isinstance(tc, dict):
                tid = tc.get("id")
                tname = (tc.get("function") or {}).get("name")
                if tid and tname:
                    out[tid] = tname
        # Anthropic-native: {"type": "tool_use", "id": ..., "name": ...}
        raw = getattr(m, "raw_content", None)
        if isinstance(raw, list):
            for b in raw:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tid = b.get("id")
                    tname = b.get("name")
                    if tid and tname:
                        out[tid] = tname
    return out


def _evict_consumed_tool_results(messages, keep_recent: int = 6, batch_evict_threshold: int = 0):
    """Stub the *content* of older tool_result messages to bound context size.

    The assistant tool_call <-> tool_result pairing must stay intact: removing
    either side yields an orphaned-tool_call / orphaned-tool_result HTTP 400 on
    OpenAI and Anthropic (exactly what ``repair_tool_message_sequence`` exists
    to repair). So instead of dropping tool_result messages, this replaces the
    ``content`` of every tool_result beyond the most recent ``keep_recent`` with
    a compact stub. The message shell (role / tool_call_id / name) is preserved
    so the pairing holds; the model can re-read the source if it still needs
    the data. This is symmetric to how the design-chat lane converts finished
    turns to a digest.

    Why a budget over *all* results: by construction every tool_result is
    referenced by its preceding assistant tool_call, so a "referenced ==
    preserve forever" rule (the previous implementation) was a silent no-op in
    normal conversations and never bounded anything. The ``keep_recent`` budget
    now applies unconditionally.

    Hysteresis (``batch_evict_threshold``): when > 0, eviction is delayed until
    ``keep_recent + batch_evict_threshold`` *pending* (non-stubbed) tool
    results have accumulated.  Once a result is stubbed it no longer counts,
    so the counter resets after each batch — giving a true ``N``-turn cadence
    (every ``batch_evict_threshold`` turns) instead of a one-shot delay.
    Default 0 preserves the old per-turn behaviour.

    Copy-on-write (see :func:`_stub_tool_result`): stubbing produces a NEW
    message object via ``dataclasses.replace`` rather than mutating the original
    in place, so a result shared with a past event payload / run record is never
    retroactively rewritten. Idempotent: messages already carrying
    ``_EVICTED_MARKER`` are returned unchanged.  The hysteresis check
    (*pending* count) excludes stubbed results, so the counter truly
    resets after each batch. Results
    whose total payload (``content`` + ``raw_content``) is at or below
    ``_EVICT_MIN_CONTENT_LEN`` are kept verbatim — stubbing them would *grow*
    context, since the stub itself is ~80 chars. Both the plain ``content``
    string and provider-native ``raw_content`` blocks are measured, so eviction
    is not silently defeated for multi-part (Anthropic/Gemini) tool results.
    """
    # Hysteresis: skip eviction if not enough *pending* tool results have
    # accumulated beyond keep_recent.  "Pending" means non-stubbed — once a
    # result has been evicted (stubbed) it no longer counts, so the counter
    # resets after each batch.  This gives a true N-turn cadence (every
    # ``batch_evict_threshold`` turns) instead of a one-shot delay.
    if batch_evict_threshold > 0:
        pending_count = sum(1 for m in messages if is_tool_result(m) and not _is_stubbed_tool_result(m))
        if pending_count < keep_recent + batch_evict_threshold:
            return messages

    seen = 0
    stubbed = 0
    result = []
    # Build tool_use_id → name map from ALL preceding assistant tool_call
    # messages. Anthropic ``tool_result`` blocks carry only an opaque
    # ``tool_use_id`` (no name), so per-block stubbing needs this map to name
    # the correct tool. Standard (role="tool") and Gemini (functionResponse
    # carries its own name) do not need it but accept it harmlessly.
    #
    # NOTE (counting caveat, BUG-2): ``keep_recent`` / ``pending_count`` count
    # MESSAGE objects. Standard format is one result == one message, but
    # Anthropic (and Gemini) batch N parallel results into ONE user message,
    # so a batch counts as 1. The break-even cost model (eviction costs ~1.25x a cached read) is calibrated
    # in result-units, so on Anthropic/Gemini parallel calls eviction fires one
    # batch later than the model assumes. Impact is small (the first eviction
    # dominates the cost) and bounded, hence tolerated.
    name_map = _build_tool_name_map(messages)
    for m in reversed(messages):
        if is_tool_result(m):
            if seen < keep_recent:
                # Inside the recent window — keep verbatim.
                result.append(m)
                seen += 1
                continue
            # Beyond the recent window — reclaim context by stubbing content.
            # ``is not m`` counts only results NEWLY rewritten this call —
            # _stub_tool_result is idempotent (returns m unchanged when already
            # stubbed or too small), so this is the accurate count of the
            # cache-rewrite work done this invocation.
            _stubbed_m = _stub_tool_result(m, name_map)
            if _stubbed_m is not m:
                stubbed += 1
            result.append(_stubbed_m)
        else:
            result.append(m)

    result.reverse()
    if stubbed:
        # Observability: the one-time prefix rewrite this eviction causes shows
        # up as a cache_creation spike on the NEXT LLM call. This log lets an
        # operator correlate that spike with the eviction event — it is the
        # "1.25x rewrite" the break-even cost model prices. Fires at
        # most every ``batch_evict_threshold`` turns (the hysteresis gate above).
        logger.debug(
            "evict_tool_results: stubbed %d new tool result(s) (keep_recent=%d, batch_evict_threshold=%d)",
            stubbed,
            keep_recent,
            batch_evict_threshold,
        )
    return result
