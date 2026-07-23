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
from typing import Any, Optional

from ..client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from ._shared_utils import (
    cache_hit_pct,
    context_message_cap,
    estimate_cache_adjusted_cost,
    estimate_cost,
    estimate_tokens_from_msgs,
    extract_files_from_patch,
    make_tool_signature,
)
from .context_budget import _resolve_context_limit
from external_llm.agent.message_shapes import (
    is_tool_result,
    is_tool_call,
    _is_anthropic_tool_result as _is_anthropic_shape,  # backward compat alias
    _is_gemini_tool_result as _is_gemini_shape,
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
from .config.thresholds import config, _env_flag
from .tool_registry import ToolResult

logger = logging.getLogger(__name__)

# Config-derived constants for turn loop logic.
_NO_TOOL_NUDGE_MAX: int = config.counts.AGENT_NO_TOOL_NUDGE_MAX
_NO_PROGRESS_THRESHOLD: int = config.counts.AGENT_NO_PROGRESS_THRESHOLD
_TOOL_RETRY_LIMIT: int = config.counts.AGENT_TOOL_RETRY_LIMIT


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
                pass
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
        _build_tool_hint, _build_phase_state_message, _advance_phase_after_success,
        _filter_prepared_calls
      - ContextManagerMixin: _trajectory_compress
    """

    # Patch engine for tolerant patch mode (lazy import)
    _patch_engine = None

    # ------------------------------------------------------------------
    # Turn loop entry point
    # ------------------------------------------------------------------

    def _run_llm_loop(self, ctx: TurnContext) -> "AgentResult":
        """LLM tool-loop: build initial messages, run turn loop, handle cancellation/errors."""
        ctx.plan_current_index = 0

        # ── Design chat continuation: use preserved messages instead of fresh build ──
        _continuation = getattr(self, '_continuation_data', None)
        _is_planner = (
            getattr(ctx.route, 'lane', None) is not None
            and str(getattr(ctx.route, 'lane', '')).upper() == 'PLANNER'
        )
        if _continuation and not _is_planner:
            ctx.messages = self._build_continuation_messages(
                _continuation, ctx.request, ctx.has_native_tools,
            )
            logger.info(
                "Using continuation messages: conversation=%d turns",
                len(_continuation.get("conversation") or []),
            )
        else:
            ctx.messages = self._build_initial_messages(ctx.request, ctx.context, ctx.has_native_tools, tier=ctx.tier)

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
        ctx.base_url = getattr(self.llm_client, 'base_url', '') or ''

        ctx.write_tool_used = False
        ctx.rollback_performed = False
        ctx.rollback_result = None
        ctx.budget_warned = False
        ctx.fail_streak = {}
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
                self._cb("turn_start", {
                    "turn": ctx.turn_num,
                    "native_tools": ctx.has_native_tools,
                    "provider": ctx.provider_name,
                    "model": getattr(self.config, "model_name", "") or "",
                })

                # Streaming token callback: forwards text_delta chunks to the
                # frontend as they arrive, so the final summary streams
                # incrementally instead of appearing all at once after a
                # blocking non-streaming call.
                _token_cb = self.config.make_token_callback()

                _llm_call_start = time.monotonic()
                try:
                    response = self._llm_call_with_tools(
                            _prep.messages,
                            read_only_request=ctx.read_only_request,
                            token_callback=_token_cb,
                        )
                except (LLMConnectionError, LLMRateLimitError, LLMServerUnavailableError, LLMQuotaExceededError, LLMAuthenticationError) as e:
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
                    self._cb("error", {
                        "message": error_message,
                        "error_type": error_type,
                        "turn": ctx.turn_num,
                    })
                    self.performance_collector.end_session()
                    performance_summary = self.performance_collector.get_summary()

                    return AgentResult(
                        status="error",
                        turns=ctx.turns,
                        error=error_message,
                        applied_patches=self.registry.applied_patches,
                        metadata={
                            "performance": performance_summary,
                        },
                    )
                def _rget(_key: str, _default=None):
                    if isinstance(response, dict):
                        return response.get(_key, _default)
                    return getattr(response, _key, _default)

                # ── Token tracking ──
                _pt = _rget("prompt_tokens", 0)
                if _pt is None:
                    _pt = _rget("tokens_used", 0)  # fallback: total when split unavailable
                _ct = _rget("completion_tokens", 0)
                _crt = _rget("cache_read_input_tokens", 0) or 0
                _cct = _rget("cache_creation_input_tokens", 0) or 0
                ctx.total_prompt_tokens += _pt
                ctx.total_completion_tokens += _ct
                ctx.total_cache_read_tokens += _crt
                ctx.total_cache_creation_tokens += _cct
                ctx.last_call_prompt_tokens = _pt
                ctx.last_call_completion_tokens = _ct
                _llm_elapsed_ms = int((time.monotonic() - _llm_call_start) * 1000)
                _finish_reason = _rget("finish_reason", "")
                _tool_calls_count = len(_rget("tool_calls", []))
                if _pt or _ct:
                    _turn_cost = estimate_cost(ctx.provider_name, _pt, _ct, model=ctx.model_name)
                    _total_cost = estimate_cost(
                        ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name)
                    _turn_actual_cost = estimate_cache_adjusted_cost(
                        ctx.provider_name, _pt, _ct, _crt, _cct, model=ctx.model_name, base_url=ctx.base_url)
                    _total_actual_cost = estimate_cache_adjusted_cost(
                        ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                        ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url)
                    self._cb("token_usage", {
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
                    })

                tool_calls = _rget("tool_calls", None) or []
                content = _rget("content", "")

                if content and content.strip():
                    self._cb("agent_thinking", {
                        "turn": ctx.turn_num,
                        "content": content.strip()[:2000],
                        "agent_id": self.config.agent_id,
                        "has_tool_calls": bool(tool_calls),
                    })

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
                        # responses hit the text_reply path (line 585)
                        # instead of false-success re-nudge death loop.
                        ctx.any_tool_called = False
                        continue
                    return _fa.result

                # Execute tool calls
                _tool_out = self._execute_and_process_tool_calls(
                    ctx,
                    tool_calls=tool_calls,
                    content=content,
                    response=response,
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
                    prepared_calls=_tool_out.prepared_calls,
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
                choices = raw_resp.get("choices") or []
                msg_obj = (choices[0].get("message", {}) if choices else {}) or {}
                rc = msg_obj.get("reasoning_content", "") or ""
                if isinstance(rc, str) and rc.strip():
                    return rc.strip()
            except (AttributeError, TypeError, IndexError):
                pass
        return content if isinstance(content, str) else ""

    # ------------------------------------------------------------------
    # Max turns handler
    # ------------------------------------------------------------------

    def _handle_max_turns_reached(self, ctx: TurnContext) -> "AgentResult":
        """Handle max_turns exhaustion: attempt one final no-tool LLM call, return result."""
        logger.warning("Agent reached max_turns=%d", self.config.max_turns)

        # Streaming token callback (same as main turn loop)
        _token_cb = self.config.make_token_callback()

        try:
            response = self._llm_call_with_tools(
                    ctx.messages,
                    read_only_request=ctx.read_only_request,
                    token_callback=_token_cb,
                )

            def _rget(key: str, default=None):
                getter = getattr(response, "get", None)
                if callable(getter):
                    try:
                        return getter(key, default)
                    except (AttributeError, TypeError, KeyError):
                        pass
                return getattr(response, key, default)

            _pt = _rget("prompt_tokens", 0) or 0
            if not _pt:
                _pt = _rget("tokens_used", 0) or 0  # fallback: total when split unavailable
            _ct = _rget("completion_tokens", 0) or 0
            # Defense-depth parity with main turn loop (L228-239): accumulate
            # cache tokens here too — the max_turns final call consumes cache
            # budget and must be reflected in the per-bucket counters.
            _crt = _rget("cache_read_input_tokens", 0) or 0
            _cct = _rget("cache_creation_input_tokens", 0) or 0
            ctx.total_prompt_tokens += _pt
            ctx.total_completion_tokens += _ct
            ctx.total_cache_read_tokens += _crt
            ctx.total_cache_creation_tokens += _cct
            ctx.last_call_prompt_tokens = _pt
            ctx.last_call_completion_tokens = _ct

            final_tool_calls = _rget("tool_calls", []) or []

            if final_tool_calls:
                ctx.messages.append(LLMMessage(
                    role="user",
                    content="[WRAP UP] Turn limit reached. Do NOT call any more tools. "
                            "Summarize what was accomplished. Do NOT continue working — the session is ending."
                ))
                response = self._llm_call_with_tools(
                    ctx.messages,
                    read_only_request=ctx.read_only_request,
                    token_callback=_token_cb,
                )

                # NOTE: _rget closes over `response` *by reference*, so after the
                # reassignment above it already reads the new response — a second
                # identical closure (_rget2) was redundant and has been removed.
                _pt = _rget("prompt_tokens", 0) or 0
                if not _pt:
                    _pt = _rget("tokens_used", 0) or 0  # fallback: total when split unavailable
                _ct = _rget("completion_tokens", 0) or 0
                # Defense-depth parity with main turn loop (L228-239): accumulate
                # cache tokens for the wrap-up retry call too.
                _crt = _rget("cache_read_input_tokens", 0) or 0
                _cct = _rget("cache_creation_input_tokens", 0) or 0
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

            review_summary: Optional[str] = None
            should_do_review = self.config.self_review_enabled and self.registry.applied_patches
            if should_do_review and self._is_trivial_edit_request(ctx.request):
                should_do_review = False
            if should_do_review:
                review_summary = self._run_self_review(ctx.messages, ctx.has_native_tools)
                if review_summary and ' lgtm ' not in f' {review_summary.lower()} ':
                    final_msg += f"\n\n---\n**[Self-Review]** {review_summary}"

            self.performance_collector.end_session()
            performance_summary = self.performance_collector.get_summary()

            _final_result = AgentResult(
                status="success",
                turns=ctx.turns,
                final_message=final_msg,
                applied_patches=self.registry.applied_patches,
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
                        "issues_found": bool(
                            review_summary and ' lgtm ' not in f' {review_summary.lower()} '
                        ),
                    },
                    "tokens": {
                        "prompt": ctx.total_prompt_tokens,
                        "completion": ctx.total_completion_tokens,
                        "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                        "cost_usd": round(
                            estimate_cost(
                                ctx.provider_name,
                                ctx.total_prompt_tokens,
                                ctx.total_completion_tokens, model=ctx.model_name),
                            6,
                        ),
                        "cache_adjusted_cost_usd": round(
                            estimate_cache_adjusted_cost(
                                ctx.provider_name,
                                ctx.total_prompt_tokens,
                                ctx.total_completion_tokens,
                                ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url),
                            6,
                        ),
                        "cache_read_tokens": ctx.total_cache_read_tokens,
                        "cache_creation_tokens": ctx.total_cache_creation_tokens,
                        "cache_hit_ratio": _cache_hit_ratio(ctx),
                        "last_call_prompt": ctx.last_call_prompt_tokens,
                        "last_call_completion": ctx.last_call_completion_tokens,
                        "provider": ctx.provider_name,
                    },
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
            return _final_result

        except Exception as e:
            logger.debug("Final LLM call after max_turns failed: %s", e)
            if "without any applied patches" in str(e):
                logger.warning("Blocking false success at max_turns for write-intent request")

        _max_result = AgentResult(
            status="max_turns",
            turns=ctx.turns,
            final_message=f"Reached maximum turns ({self.config.max_turns})",
            applied_patches=self.registry.applied_patches,
            metadata={
                "turns_used": self.config.max_turns,
                "git_state": ctx.git_state,
                "tdd": {
                    "runs": ctx.tdd_total_runs,
                    "pass": ctx.tdd_total_pass,
                    "fail": ctx.tdd_fail_count,
                },
                "tokens": {
                    "prompt": ctx.total_prompt_tokens,
                    "completion": ctx.total_completion_tokens,
                    "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                    "cost_usd": round(estimate_cost(
                        ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                    "cache_adjusted_cost_usd": round(
                        estimate_cache_adjusted_cost(
                            ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                            ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                    ),
                    "cache_read_tokens": ctx.total_cache_read_tokens,
                    "cache_creation_tokens": ctx.total_cache_creation_tokens,
                    "cache_hit_ratio": _cache_hit_ratio(ctx),
                },
            },
        )
        self._save_session_log(
            ctx.session_id, ctx.request, _max_result,
            ctx.total_prompt_tokens, ctx.total_completion_tokens,
            ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens,
        )
        return _max_result

    # ------------------------------------------------------------------
    # Final answer handler
    # ------------------------------------------------------------------

    def _handle_final_answer_turn(
        self,
        ctx: TurnContext,
        final_msg: str,
    ) -> "_FinalAnswerOutcome":

        logger.info("Agent finished after %d turns", ctx.turn_num - 1)

        if (not ctx.read_only_request) and (not self.registry.applied_patches):
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
                        "tokens": {
                            "prompt": ctx.total_prompt_tokens,
                            "completion": ctx.total_completion_tokens,
                            "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                            "cost_usd": round(estimate_cost(
                                ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                            "cache_adjusted_cost_usd": round(
                                estimate_cache_adjusted_cost(
                                    ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                                    ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                            ),
                            "cache_read_tokens": ctx.total_cache_read_tokens,
                            "cache_creation_tokens": ctx.total_cache_creation_tokens,
                        },
                    },
                )
                self._save_session_log(
                    ctx.session_id, ctx.request, _noop_result,
                    ctx.total_prompt_tokens, ctx.total_completion_tokens,
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
                        "tokens": {
                            "prompt": ctx.total_prompt_tokens,
                            "completion": ctx.total_completion_tokens,
                            "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                            "cost_usd": round(estimate_cost(
                                ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                            "cache_adjusted_cost_usd": round(
                                estimate_cache_adjusted_cost(
                                    ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                                    ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                            ),
                            "cache_read_tokens": ctx.total_cache_read_tokens,
                            "cache_creation_tokens": ctx.total_cache_creation_tokens,
                            "provider": ctx.provider_name,
                        },
                    },
                )
                self._save_session_log(
                    ctx.session_id, ctx.request, _text_reply_result,
                    ctx.total_prompt_tokens, ctx.total_completion_tokens,
                )
                return _FinalAnswerOutcome(result=_text_reply_result)

            logger.warning("Blocking false success: write-intent request finished with no applied patches")

            if ctx.no_tool_nudge_count < _NO_TOOL_NUDGE_MAX and ctx.turn_num < self.config.max_turns:
                ctx.no_tool_nudge_count += 1
                if ctx.read_only_request:
                    nudge_content = (
                        "[ACTION REQUIRED] You explained the task but did NOT call any tool.\n"
                        f"Original task: {ctx.request}\n\n"
                        "You MUST output a JSON tool call. Do NOT write text — output the call directly.\n\n"
                        "This is a READ-ONLY request.\n"
                        "Use only read/search tools.\n"
                        "Do NOT call apply_patch.\n"
                        "Once the answer is confirmed, finish with the final summary in the user's language."
                    )
                else:
                    nudge_content = (
                        "[ACTION REQUIRED] You explained the task but did NOT call any tool.\n"
                        "You MUST output a tool call. Do NOT write code as plain text.\n\n"
                        "To CREATE a new file:\n"
                        "  bash('cat > path << EOF\\n...content...\\nEOF') or bash('tee path << EOF\\n...\\nEOF')\n\n"
                        "To MODIFY an existing file:\n"
                        "  1. bash cat/pygmentize to see current content\n"
                        "  2. apply_patch with unified diff\n\n"
                        f"Task: {ctx.request[:2000]}"
                    )
                nudge_msg = LLMMessage(role="user", content=nudge_content)
                self._cb("tool_nudge", {"turn": ctx.turn_num, "nudge_count": ctx.no_tool_nudge_count,
                                        "agent_id": self.config.agent_id})
                logger.info("Re-nudging small model (nudge %d/%d)", ctx.no_tool_nudge_count, _NO_TOOL_NUDGE_MAX)
                return _FinalAnswerOutcome(nudge_message=nudge_msg, nudge_count=ctx.no_tool_nudge_count)

            self.performance_collector.end_session()
            performance_summary = self.performance_collector.get_summary()

            _false_success_result = AgentResult(
                status="error",
                turns=ctx.turns,
                final_message=(
                    final_msg
                    or "Model finished without calling apply_patch, and no patch was applied."
                ),
                applied_patches=self.registry.applied_patches,
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
                    "tokens": {
                        "prompt": ctx.total_prompt_tokens,
                        "completion": ctx.total_completion_tokens,
                        "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                        "cost_usd": round(estimate_cost(
                            ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                        "cache_adjusted_cost_usd": round(
                            estimate_cache_adjusted_cost(
                                ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                                ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                        ),
                        "cache_read_tokens": ctx.total_cache_read_tokens,
                        "cache_creation_tokens": ctx.total_cache_creation_tokens,
                        "last_call_prompt": ctx.last_call_prompt_tokens,
                        "last_call_completion": ctx.last_call_completion_tokens,
                        "provider": ctx.provider_name,
                    },
                    "performance": performance_summary,
                    "false_success_blocked": True,
                    "nudge_count": ctx.no_tool_nudge_count,
                },
            )
            self._save_session_log(
                ctx.session_id, ctx.request, _false_success_result,
                ctx.total_prompt_tokens, ctx.total_completion_tokens,
            )
            return _FinalAnswerOutcome(result=_false_success_result)

        review_summary: Optional[str] = None

        should_do_review = self.config.self_review_enabled and self.registry.applied_patches
        if should_do_review:
            if self._is_trivial_edit_request(ctx.request):
                logger.info("Skipping self-review phase for trivial request")
                should_do_review = False

        if should_do_review:
            review_summary = self._run_self_review(ctx.messages, ctx.has_native_tools)
            if review_summary and ' lgtm ' not in f' {review_summary.lower()} ':
                final_msg = (
                    final_msg
                    + f"\n\n---\n**[Self-Review]** {review_summary}"
                )

        # LLM re-invocation for future-tense detection removed — once a patch has been applied,
        # the task is complete even if the sentence is in future tense. Unnecessary LLM calls
        # only waste tokens.

        self.performance_collector.end_session()
        performance_summary = self.performance_collector.get_summary()

        _final_result = AgentResult(
            status="success",
            turns=ctx.turns,
            final_message=final_msg,
            applied_patches=self.registry.applied_patches,
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
                    "issues_found": bool(
                        review_summary
                        and ' lgtm ' not in f' {review_summary.lower()} '
                    ),
                },
                "tokens": {
                    "prompt": ctx.total_prompt_tokens,
                    "completion": ctx.total_completion_tokens,
                    "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                    "cost_usd": round(estimate_cost(
                        ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                    "cache_adjusted_cost_usd": round(
                        estimate_cache_adjusted_cost(
                            ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                            ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                    ),
                    "cache_read_tokens": ctx.total_cache_read_tokens,
                    "cache_creation_tokens": ctx.total_cache_creation_tokens,
                    "provider": ctx.provider_name,
                },
                "performance": performance_summary,
            },
        )
        self._save_session_log(
            ctx.session_id, ctx.request, _final_result,
            ctx.total_prompt_tokens, ctx.total_completion_tokens,
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
        prepared_calls: list,
    ) -> "_PostToolResult":
        """Process results after tool execution: sanitize, update messages, auto-observe, TDD."""
        try:
            if hasattr(response, "content") and isinstance(response.content, str):
                response.content = self._strip_thinking_text(response.content)
        except (AttributeError, TypeError):
            pass

        ctx.messages = self._append_native_tool_messages(ctx.messages, response, new_messages)

        _PATCH_TOOLS = {"apply_patch", "write_plan"}
        patch_ok_this_turn = any(
            t.tool_name in _PATCH_TOOLS and t.tool_result.ok
            for t in ctx.turns
            if t.turn_num == ctx.turn_num
        )

        # Auto-observation after successful patch.
        # Scope `git diff` to files touched THIS turn only. A bare `git diff`
        # dumps the entire working tree and re-surfaces earlier turns' changes
        # every turn, ballooning ctx.messages with redundant content. Both
        # apply_patch and write_plan record affected paths in their ToolResult
        # metadata under "touched_files" (write_plan via diff_apply details).
        if (
            patch_ok_this_turn
            and self.config.auto_observation_enabled
            and not self.config.auto_test_on_patch
        ):
            _obs_paths: list[str] = []
            for _t in ctx.turns:
                if _t.turn_num != ctx.turn_num or _t.tool_name not in _PATCH_TOOLS:
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
                        capture_output=True, text=True, timeout=10,
                    )
                    obs_content = (_diff_proc.stdout or "").strip()
                except Exception:
                    obs_content = ""
            if obs_content:
                ctx.messages.append(
                    LLMMessage(role="user", content=f"[auto_observation]\n{obs_content}")
                )
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
                applied_patches=self.registry.applied_patches,
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
                    "tokens": {
                        "prompt": ctx.total_prompt_tokens,
                        "completion": ctx.total_completion_tokens,
                        "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                        "cost_usd": round(estimate_cost(
                            ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                        "cache_adjusted_cost_usd": round(
                            estimate_cache_adjusted_cost(
                                ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                                ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                        ),
                        "cache_read_tokens": ctx.total_cache_read_tokens,
                        "cache_creation_tokens": ctx.total_cache_creation_tokens,
                        "last_call_prompt": ctx.last_call_prompt_tokens,
                        "last_call_completion": ctx.last_call_completion_tokens,
                        "provider": ctx.provider_name,
                    },
                    "performance": performance_summary,
                    "early_finish": {
                        "enabled": True,
                        "reason": "patch_ok_this_turn_and_no_tdd_and_no_self_review",
                    },
                },
            )
            self._save_session_log(
                ctx.session_id, ctx.request, _final_result,
                ctx.total_prompt_tokens, ctx.total_completion_tokens,
            )
            return _PostToolResult(messages=ctx.messages, tdd_fail_count=ctx.tdd_fail_count, tdd_total_runs=ctx.tdd_total_runs, tdd_total_pass=ctx.tdd_total_pass, early_return=_final_result)

        # TDD auto-test
        if self.config.auto_test_on_patch:
            patch_succeeded_this_turn = any(
                t.tool_name in _PATCH_TOOLS and t.tool_result.ok
                for t in ctx.turns
                if t.turn_num == ctx.turn_num
            )
            if patch_succeeded_this_turn:
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
                            applied_patches=self.registry.applied_patches,
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
                                "tokens": {
                                    "prompt": ctx.total_prompt_tokens,
                                    "completion": ctx.total_completion_tokens,
                                    "total": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                                    "cost_usd": round(estimate_cost(
                                        ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens, model=ctx.model_name), 6),
                                    "cache_adjusted_cost_usd": round(
                                        estimate_cache_adjusted_cost(
                                            ctx.provider_name, ctx.total_prompt_tokens, ctx.total_completion_tokens,
                                            ctx.total_cache_read_tokens, ctx.total_cache_creation_tokens, model=ctx.model_name, base_url=ctx.base_url), 6,
                                    ),
                                    "cache_read_tokens": ctx.total_cache_read_tokens,
                                    "cache_creation_tokens": ctx.total_cache_creation_tokens,
                                    "cache_hit_ratio": _cache_hit_ratio(ctx),
                                    "provider": ctx.provider_name,
                                },
                                "performance": performance_summary,
                            },
                        )
                        self._save_session_log(
                            ctx.session_id, ctx.request, _final_result,
                            ctx.total_prompt_tokens, ctx.total_completion_tokens,
                        )
                        return _PostToolResult(messages=ctx.messages, tdd_fail_count=ctx.tdd_fail_count, tdd_total_runs=ctx.tdd_total_runs, tdd_total_pass=ctx.tdd_total_pass, early_return=_final_result)

        return _PostToolResult(messages=ctx.messages, tdd_fail_count=ctx.tdd_fail_count, tdd_total_runs=ctx.tdd_total_runs, tdd_total_pass=ctx.tdd_total_pass)

    # ------------------------------------------------------------------
    # Turn message preparation
    # ------------------------------------------------------------------

    def _prepare_turn_messages(self, ctx: TurnContext) -> "_TurnPrepResult":
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
            tool_schemas=self.registry.get_tool_schemas() if self.registry else None,
        )

        if (
            ctx.turn_num == 1
            and not ctx.search_first_hint_done
            and ctx.target_keywords
            and not ctx.known_target_file
        ):
            _sf_targets = ", ".join(f'"{k}"' for k in ctx.target_keywords[:2])
            ctx.ephemeral_pending.append(LLMMessage(
                role="system",
                content=(
                    f"[TOOL HINT] Text change task detected. Target: {_sf_targets}.\n"
                    "Priority order to locate the target:\n"
                    "  1. find_symbol — fastest if the target is a function/class/symbol name.\n"
                    "  2. bash (grep -rn) — use if the target is arbitrary text, not a named symbol.\n"
                    "  3. read_symbol or bash (cat) — ONLY after you know the exact file and line from step 1 or 2.\n"
                    "Do NOT browse files randomly without locating the target first."
                ),
            ))
            ctx.search_first_hint_done = True

        if (
            ctx.reads_since_last_edit >= _NO_PROGRESS_THRESHOLD
            and ctx.goal_reminder_injected < 3
        ):
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
                    "\n\nIf you have enough context, apply the edit NOW with apply_patch. "
                    "Do not read more files."
                )
            ctx.ephemeral_pending.append(LLMMessage(
                role="user",
                content=_reminder_text,
            ))
            ctx.goal_reminder_injected += 1
            ctx.reads_since_last_edit = 0
            logger.info(
                "Goal reminder injected (reminder #%d, was %d reads without edit)",
                ctx.goal_reminder_injected, _rf_count,
            )
            self._cb("goal_reminder", {
                "turn": ctx.turn_num,
                "reads_without_edit": _rf_count,
                "reminder_count": ctx.goal_reminder_injected,
            })
        try:
            tool_hint = self._build_tool_hint()
            if tool_hint:
                ctx.ephemeral_pending.append(
                    LLMMessage(role="system", content=tool_hint)
                )
        except (AttributeError, TypeError):
            pass

        # Planner progress hint
        if ctx.plan_subtasks and ctx.plan_current_index < len(ctx.plan_subtasks):
            try:
                task = ctx.plan_subtasks[ctx.plan_current_index]
                hint = (
                    "[PLAN PROGRESS]\n"
                    f"Current subtask ({ctx.plan_current_index+1}/{len(ctx.plan_subtasks)}): "
                    f"{task.get('title','')}\n"
                    f"Target files: {', '.join(task.get('files') or [])}"
                )
                ctx.ephemeral_pending.append(
                    LLMMessage(role="user", content=hint)
                )
            except (AttributeError, TypeError):
                pass

        if ctx.read_only_request or ctx.is_local_model:
            try:
                ctx.ephemeral_pending.append(
                    LLMMessage(
                        role="system",
                        content=self._build_phase_state_message(ctx.read_only_request),
                    )
                )
            except (AttributeError, TypeError):
                pass

        if (
            ctx.known_target_file
            and not ctx.read_only_request
            and ctx.turn_num == 1
        ):
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
                pass

        try:
            if ctx.turn_num > 2:
                traj = self._trajectory_compress(ctx.turns)
                if traj:
                    ctx.ephemeral_pending.append(
                        LLMMessage(role="user", content=traj)
                    )
        except (AttributeError, TypeError):
            pass

        if self.config.cancel_event and self.config.cancel_event.is_set():
            raise AgentCancelled("cancelled by user")

        if self.config.message_queue is not None:
            import queue as _queue_mod
            while True:
                try:
                    mid_msg = self.config.message_queue.get_nowait()
                    ctx.messages.append(LLMMessage(
                        role="user",
                        content=f"[USER INTERRUPT] {mid_msg}",
                    ))
                    self._cb("user_message_received", {
                        "message": mid_msg,
                        "turn": ctx.turn_num,
                    })
                    logger.info("Mid-task user message injected at turn %d: %s", ctx.turn_num, mid_msg[:80])
                except _queue_mod.Empty:
                    break

        ctx.messages = self._trim_context(ctx.messages)

        _max_turns = getattr(self.config, 'max_turns', 0)
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
    ) -> "_PreparedCallsResult":
        """Build prepared_calls from raw tool_calls, filter, emit previews.

        May raise AgentCancelled if cancel_event is set.
        Returns _PreparedCallsResult. If should_continue is True, caller should
        return a should_continue _ToolTurnOutcome immediately.
        """
        if plan_subtasks:
            try:
                last_tool = turns[-1].tool_name if turns else None
                if last_tool in {"apply_patch", "write_plan"}:
                    plan_current_index = min(
                        plan_current_index + 1,
                        len(plan_subtasks)
                    )
            except (TypeError, AttributeError):
                pass

        prepared_calls = []
        _unknown_tool_notices: list[str] = []

        try:
            _repo_lang = getattr(self.registry, "repo_language", None)
            if hasattr(self.registry, "get_tool_names"):
                _known_tools = self.registry.get_tool_names(lang_filter=_repo_lang)
                # Tools hidden by language masking (non-Python repo). Distinguishing
                # them from genuinely-unknown tools lets us emit a precise notice
                # instead of the generic read-only-mode message.
                _masked_tools = (
                    self.registry.get_tool_names() - _known_tools
                    if _repo_lang is not None else frozenset()
                )
            else:
                _known_tools = {
                    t.get("name")
                    for t in (self.registry.get_tool_schemas() or [])
                    if t.get("name")
                }
                _masked_tools = frozenset()
        except (AttributeError, TypeError, KeyError):
            _known_tools = set()
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
                        tool_args = json.loads(raw_args)
                    except Exception:
                        s = raw_args.strip()
                        try:
                            _item_ = s.find("{")
                            r = s.rfind("}")
                            if _item_ != -1 and r != -1 and r > _item_:
                                mid = s[_item_ : r + 1]
                                obj2 = json.loads(mid)
                                tool_args = obj2 if isinstance(obj2, dict) else {"__raw_arguments": s}
                            else:
                                tool_args = {"__raw_arguments": s}
                        except Exception as e2:
                            tool_args = {"__raw_arguments": s, "__parse_error": str(e2)}
                else:
                    tool_args = {}

            if not isinstance(tool_args, dict):
                tool_args = {}

            if not tool_name:
                logger.warning("Skipping tool call with empty name: %r", call)
                continue

            if _masked_tools and tool_name in _masked_tools:
                logger.warning("Skipping Python-only tool call '%s' (repo is %s)",
                               tool_name, getattr(_repo_lang, "value", _repo_lang))
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
            prepared_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "call_id": call_id,
                "original_call": call,
            })

            logger.info("Tool call (pending): %s(%s)", tool_name, list(tool_args.keys()))

        phase_rule_messages: list[LLMMessage] = [
            LLMMessage(role="user", content=f"[PHASE RULE] {n}")
            for n in _unknown_tool_notices
        ]
        _calls_before_filter = len(prepared_calls)
        prepared_calls, phase_notices = self._filter_prepared_calls(
            prepared_calls,
            read_only_request=read_only_request,
        )
        if len(prepared_calls) < _calls_before_filter:
            logger.info(
                "Tool call filter: %d/%d blocked (guards/phase)",
                _calls_before_filter - len(prepared_calls),
                _calls_before_filter,
            )
        for notice in phase_notices:
            phase_rule_messages.append(
                LLMMessage(role="user", content=f"[PHASE RULE] {notice}")
            )
            if self.config.stream_callback:
                try:
                    self.config.stream_callback("tool_filtered", {
                        "turn": turn_num,
                        "notice": notice,
                    })
                except (AttributeError, TypeError):
                    pass

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
                    self._cb("tool_call_preview", {
                        "turn": turn_num,
                        "tool": _pc["tool"],
                        "args": self.registry.normalize_args_for_display(_pc["args"]),
                    })
                except (AttributeError, TypeError):
                    pass

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
        any_tool_called: bool,
        write_tools: set,
        read_only_request: bool,
        is_local_model: bool,
        target_keywords: list[str],
        request: str,
        session_id: str,
        git_state: Any,
        turn_num: int,
        plan_current_index: int,
        turns: list,
    ) -> "_ResultsProcessingOutcome":
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
            _cache_hit = result.metadata.get("cache_hit", False) if result.metadata else False
            self.performance_collector.record_tool_call(
                tool_name, result.execution_time,
                cache_hit=_cache_hit,
                failed=not result.ok,
            )

            # Read-only early finish detection
            if read_only_request and not write_tool_used:
                early_result = self._try_readonly_early_finish(
                    tool_name, result, request, read_only_request
                )
                if early_result is not None:
                    if self.config.stream_callback:
                        _LARGE_TOOLS = {"apply_patch", "write_plan"}
                        _VERBOSE_TOOLS = {"run_tests", "run_lint", "bash"}
                        content_limit = 8000 if tool_name in _LARGE_TOOLS else (6000 if tool_name in _VERBOSE_TOOLS else 2000)
                        try:
                            self._cb("tool_call", {
                                "turn": turn_num,
                                "tool": tool_name,
                                "args": self.registry.normalize_args_for_display(tool_args),
                                "result": {
                                    "ok": result.ok,
                                    "content": result.content[:content_limit],
                                    "error": result.error,
                                },
                            })
                        except Exception:
                            pass
                    current_turn = AgentTurn(
                        turn_num=turn_num,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=result,
                    )
                    early_result.turns = [*turns, current_turn]
                    early_result.metadata.update({
                        "turns_used": turn_num,
                        "readonly_early_finish": True,
                        "deterministic_tool": tool_name,
                    })
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
                record_recall_outcome(ok=result.ok, session_key=str(id(fail_streak)))
            except Exception:  # noqa: BLE001 — recall bookkeeping must not break the pipeline
                pass
            if not result.ok:
                classification = self._failure_classifier.classify(
                    tool_name,
                    tool_args,
                    result
                )
                try:
                    result.metadata = result.metadata or {}
                    result.metadata["failure_classification"] = {
                        "action": classification.action,
                        "reason": classification.reason
                    }
                except (AttributeError, TypeError):
                    pass
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
                        tool_name, tool_args, result,
                        getattr(self.registry, "repo_root", "") or "",
                        # fail_streak is a fresh dict per run (ctx.fail_streak={}
                        # at _run_llm_loop entry) → id() is a stable per-run key,
                        # so a new run re-records & re-hints (matches the old
                        # fail_streak==0 gate that reset each run).
                        session_key=str(id(fail_streak)),
                    )
                    if _recall_hint:
                        new_messages.append(
                            LLMMessage(role="user", content=_recall_hint)
                        )
                except Exception:  # noqa: BLE001 — recall must never break the pipeline
                    pass
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
                    self._cb("fail_loop_detected", {
                        "turn": turn_num,
                        "tool": tool_name,
                        "streak": self._tool_retry_counter[tool_name],
                        "signal": "tool_exhaustion",
                    })
                    self._tool_retry_counter[tool_name] = 0
                fail_streak[_loop_key] = fail_streak.get(_loop_key, 0) + 1
                if fail_streak[_loop_key] == fail_streak_threshold:
                    if tool_name == "write_plan":
                        _recovery = (
                            "Do NOT call write_plan with the same arguments again. "
                            "Instead: (1) use find_symbol to locate the target, "
                            "(2) use bash with cat or sed to see the exact text, "
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
                    self._cb("fail_loop_detected", {
                        "turn": turn_num,
                        "tool": tool_name,
                        "streak": fail_streak[_loop_key],
                    })
            else:
                fail_streak.pop(_loop_key, None)
                self._tool_retry_counter[tool_name] = 0

            self._advance_phase_after_success(
                tool_name,
                tool_args,
                result,
                read_only_request=read_only_request,
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
                _LARGE_TOOLS = {"apply_patch", "write_plan"}
                _VERBOSE_TOOLS = {"run_tests", "run_lint", "bash"}
                content_limit = 8000 if tool_name in _LARGE_TOOLS else (6000 if tool_name in _VERBOSE_TOOLS else 2000)
                try:
                    self._cb("tool_call", {
                        "turn": turn_num,
                        "tool": tool_name,
                        "args": self.registry.normalize_args_for_display(tool_args),
                        "result": {
                            "ok": result.ok,
                            "content": result.content[:content_limit],
                            "error": result.error,
                        },
                    })
                except (AttributeError, TypeError):
                    pass

            new_messages.append(
                self._build_tool_result_message(call_id, tool_name, result, tool_args)
            )

            # Read-only exploration tools. Counting these toward
            # reads_since_last_edit lets the GOAL REMINDER fire when the agent
            # loops on reads (incl. read_symbol/read_file) without editing.
            _EXPLORATION_TOOLS = {
                "bash", "find_symbol", "find_references",
                "find_relevant_files", "read_file", "read_symbol",
                "get_file_outline", "get_project_info", "grep",
            }
            if result.ok:
                if tool_name in _EXPLORATION_TOOLS:
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
                    except Exception:  # noqa: BLE001 — must never break the pipeline
                        pass

        return _ResultsProcessingOutcome(
            new_messages=new_messages,
            write_tool_used=write_tool_used,
            reads_since_last_edit=reads_since_last_edit,
            noop_confirmed=_noop_confirmed,
            fail_streak=fail_streak,
        )

    # ------------------------------------------------------------------
    # Execute and process tool calls
    # ------------------------------------------------------------------

    def _execute_and_process_tool_calls(
        self,
        ctx: TurnContext,
        tool_calls: list,
        content: str,
        response: Any,
    ) -> "_ToolTurnOutcome":
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
        if (hasattr(self.config, 'parallel_tool_execution_enabled') and
            self.config.parallel_tool_execution_enabled and
            len(prepared_calls) > 1):
            parallel_calls = [{"tool": pc["tool"], "args": pc["args"]} for pc in prepared_calls]
            try:
                results = self.registry.dispatch_parallel(parallel_calls)
            except StopIteration:
                logger.error("Tool dispatch_parallel StopIteration (mock side_effect exhausted)")
                results = [
                    ToolResult(ok=False, content="", error="StopIteration", metadata={})
                    for _ in parallel_calls
                ]
            _log_parallel_write_failures(results, parallel_calls, self)
        else:
            results = []
            for pc in prepared_calls:
                tool_name = pc["tool"]
                tool_args = pc["args"]
                already_retried = False
                try:
                    result = self.registry.dispatch(tool_name, tool_args)
                    any_tool_called = True

                    try:
                        if result and getattr(result, "ok", False):
                            self._record_tool_success(tool_name, tool_args)
                        else:
                            self._record_tool_failure(tool_name, tool_args)
                            # Persist write-tool failures to JSONL for analysis.
                            if tool_name in self.registry._WRITE_TOOLS:
                                try:
                                    from .tool_failure_log import (
                                        record_write_tool_failure_from_tr,
                                    )
                                    record_write_tool_failure_from_tr(
                                        tool=tool_name, tr=result, args=tool_args,
                                        model=getattr(self.config, "model", None),
                                        repo_root=getattr(self.registry, "repo_root", None),
                                    )
                                except Exception:
                                    logger.debug(
                                        "tool_failure_log: record failed", exc_info=True
                                    )
                    except (AttributeError, TypeError):
                        pass
                except StopIteration:
                    logger.error(
                        "Tool dispatch StopIteration (mock side_effect exhausted): %s",
                        tool_name,
                    )
                    result = ToolResult(ok=False, content="", error="StopIteration", metadata={})

                # Auto-repair for apply_patch failures (max 1 retry)
                if tool_name == "apply_patch" and not result.ok and not already_retried:
                    new_args = self._auto_repair_apply_patch_args(tool_args, result)
                    if new_args:
                        logger.debug("Auto-repair for apply_patch: attempting repair")
                        # Capture the failure cause BEFORE retry_result replaces
                        # `result`; on success result.error becomes None and the
                        # original error would otherwise be lost from metadata.
                        _orig_error = result.error
                        retry_result = self.registry.dispatch(tool_name, new_args)
                        already_retried = True
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

                # Tolerant patch: track failures & try edit_blocks auto-conversion
                if tool_name == "apply_patch":
                    if result.ok:
                        self._patch_fail_count = 0
                    else:
                        self._patch_fail_count += 1
                        max_failures = getattr(self.config, "tolerant_patch_max_failures", 2)
                        if (
                            getattr(self.config, "tolerant_patch_mode", False)
                            and self._patch_fail_count >= max_failures
                        ):
                            patch_text = tool_args.get("patch", "")
                            path_hint = tool_args.get("path")
                            eb_result = None
                            try:
                                from ..patch_engine import PatchEngine
                            except ImportError:
                                PatchEngine = None
                            if PatchEngine is not None:
                                try:
                                    engine = PatchEngine(self.registry.repo_root)
                                    converted = engine.convert_patch_to_edit_blocks(patch_text, path_hint)
                                    if converted:
                                        plan = {
                                            "kind": "ASICODE_PLAN_V1",
                                            "ops": [{"op": "edit_blocks", "path": converted["file_path"], "blocks": converted["blocks"]}],
                                        }
                                        plan_str = json.dumps(plan, ensure_ascii=False)
                                        eb_result = self.registry.dispatch("write_plan", {"plan": plan_str})
                                        if eb_result.ok:
                                            eb_result.metadata["auto_converted_from_patch"] = True
                                            eb_result.metadata["edit_blocks_count"] = len(converted["blocks"])
                                            eb_result.content = (
                                                f"Patch auto-converted to edit_blocks and applied successfully "
                                                f"({len(converted['blocks'])} block(s) in {converted['file_path']}).\n" + (eb_result.content or "")
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
                                    logger.debug(
                                        "edit_blocks auto-conversion also failed: %s", eb_result.error
                                    )
                                    result.metadata["edit_blocks_fallback_error"] = eb_result.error
                results.append(result)

        _rpr = self._process_tool_results(
            results=results,
            prepared_calls=prepared_calls,
            new_messages=new_messages,
            write_tool_used=write_tool_used,
            reads_since_last_edit=reads_since_last_edit,
            fail_streak=fail_streak,
            fail_streak_threshold=fail_streak_threshold,
            any_tool_called=any_tool_called,
            write_tools=ctx.write_tools,
            read_only_request=ctx.read_only_request,
            is_local_model=ctx.is_local_model,
            target_keywords=ctx.target_keywords,
            request=ctx.request,
            session_id=ctx.session_id,
            git_state=ctx.git_state,
            turn_num=ctx.turn_num,
            plan_current_index=plan_current_index,
            turns=ctx.turns,
        )
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
    # Loop cancellation handler
    # ------------------------------------------------------------------

    def _handle_loop_cancellation(
        self,
        turns: list,
        git_state: Any,
    ) -> "AgentResult":
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
                logger.info("Successfully rolled back %d/%d patches",
                           rollback_result["rolled_back"], rollback_result["total"])
                # Clear applied_patches after successful rollback so DIFF_VERIFY
                # does not falsely warn that git diff is empty.
                self.registry.applied_patches.clear()
            else:
                logger.warning("Partial or failed rollback: %d/%d patches rolled back",
                              rollback_result["rolled_back"], rollback_result["total"])
                for i, result in enumerate(rollback_result.get("results", [])):
                    if not result.get("success"):
                        logger.error("Rollback failed for patch %d: %s",
                                    result.get("patch_index", i), result.get("message", "unknown error"))
                # Partial rollback: keep applied_patches so DIFF_VERIFY and
                # downstream callers know some changes are still in the working tree.
        else:
            logger.info("No patches to rollback")
            rollback_result = {"success": True, "message": "No patches to rollback", "rolled_back": 0}

        rollback_msg, rollback_meta = _summarize_rollback(
            rollback_result if rollback_performed else None
        )

        self.performance_collector.end_session()
        performance_summary = self.performance_collector.get_summary()

        return AgentResult(
            status="cancelled",
            turns=turns,
            final_message=f"Agent execution cancelled. {rollback_msg}",
            applied_patches=self.registry.applied_patches,
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
    ) -> "AgentResult":
        """Handle unexpected Exception: rollback patches and return error result."""
        logger.exception("Unexpected error in agent loop")

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
                logger.info("Successfully rolled back %d/%d patches",
                           rollback_result["rolled_back"], rollback_result["total"])
                # Clear applied_patches after successful rollback so DIFF_VERIFY
                # does not falsely warn that git diff is empty.
                self.registry.applied_patches.clear()
            else:
                logger.warning("Partial or failed rollback: %d/%d patches rolled back",
                              rollback_result["rolled_back"], rollback_result["total"])

        self.performance_collector.end_session()
        performance_summary = self.performance_collector.get_summary()

        rollback_msg, rollback_meta = _summarize_rollback(
            rollback_result if rollback_performed else None
        )

        return AgentResult(
            status="error",
            turns=turns,
            error=f"Unexpected error: {error}. {rollback_msg}" if rollback_performed else f"Unexpected error: {error}",
            applied_patches=self.registry.applied_patches,
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
            {"performed": False, "result": None,
             "needs_manual_rollback": False, "affected_files": []},
        )

    results_list = rollback_result.get("results", []) if isinstance(rollback_result, dict) else []
    needs_manual = [
        r for r in results_list
        if isinstance(r, dict) and r.get("needs_manual_rollback")
    ]
    affected = []
    seen = set()
    for r in needs_manual:
        for f in (r.get("affected_files") or []):
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
def _log_parallel_write_failures(results, parallel_calls, pipeline):
    """Persist write-tool failures produced by ``dispatch_parallel``.

    ``dispatch_parallel`` bypasses the serial loop where the per-call failure
    logging lives, so parallel write-tool failures would otherwise escape the
    JSONL forensic log. This walks the parallel ``results`` and records every
    failed write tool. Best-effort — never raises.
    """
    try:
        from .tool_failure_log import record_write_tool_failure_from_tr
        write_tools = pipeline.registry._WRITE_TOOLS
        for idx, result in enumerate(results):
            tool_name = parallel_calls[idx]["tool"] if idx < len(parallel_calls) else ""
            if tool_name not in write_tools:
                continue
            if result and not getattr(result, "ok", True):
                record_write_tool_failure_from_tr(
                    tool=tool_name, tr=result,
                    args=parallel_calls[idx].get("args"),
                    model=getattr(pipeline.config, "model", None),
                    repo_root=getattr(pipeline.registry, "repo_root", None),
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
# fraction of ``context_message_cap`` (the SAME accounting the hard-cap
# front-trim uses). Sitting below the cap means the gentle stub-based bound here
# preempts the cruder count-based front-trim (design_chat_loop ~L141), which
# would invalidate the ENTIRE prefix. 0.75 leaves headroom for the current turn
# to grow before the hard cap becomes necessary.
#
# Prefix-stability invariant (CRITICAL, independent of the trigger mechanism).
# For the prefix cache to stay warm between turns, eviction must be the ONLY
# thing that rewrites the early prefix. It holds today because the design-chat
# loop (a) removed per-iteration re-injection (design_chat_loop.py ~L1098) and
# (b) injects turn-volatile L3 promoted-insights at a LATE position (~L661),
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
# bound is the hard-cap front-trim (``_apply_context_hard_cap``), which fires far
# later (only on a genuinely oversized prompt) and acts purely as the HTTP-400
# backstop — so a routine loop never pays a prefix rewrite.
#
# The occupancy gate, keep_recent floor and ``_evict_consumed_tool_results``
# primitive are all kept intact and unit-tested, so re-enabling eviction for a
# small-window model that routinely overflows is an env-var flip of this flag
# (ASICODE_TOOL_RESULT_EVICTION=1) — no code change or redeploy required.
_EVICTION_ENABLED = _env_flag("ASICODE_TOOL_RESULT_EVICTION", False)


def _evict_for_loop(messages, model: str = "", tool_schemas=None):
    """Occupancy-gated tool-result eviction for an in-flight agent tool-loop.

    Single source of truth for the trigger: callers pass only the model
    *identity* (``model``) and the ``tool_schemas`` sent alongside the prompt —
    never a raw threshold — so the firing decision lives in exactly ONE place.
    Every production tool-loop (MAIN_AGENT pipeline AND design-chat loop) runs
    the SAME logic, so they can never drift apart.

    DISABLED by default (``_EVICTION_ENABLED is False``): returns ``messages``
    unchanged for every model, so no prefix is ever rewritten by the gentle stub
    bound — only the hard-cap front-trim (``_apply_context_hard_cap``) bounds the
    window, as an overflow-only HTTP-400 backstop. When enabled, it fires
    ``_evict_consumed_tool_results`` (stub every tool result beyond the
    most-recent ``_EVICTION_KEEP_RECENT``) ONLY when the estimated prompt exceeds
    ``_EVICTION_OCCUPANCY_TRIGGER × context_message_cap(model)``. Below that it
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
        limit = _resolve_context_limit(model or "")
        cap = context_message_cap(
            limit, config.tokens.CONTEXT_HARD_CAP_SAFETY_MARGIN, tool_schemas
        )
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
            pass

    if _is_stubbed_tool_result(m) or size <= _EVICT_MIN_CONTENT_LEN:
        return m

    name = getattr(m, "name", "") or "tool"
    tid = getattr(m, "tool_call_id", "") or ""
    tid_suffix = f" ({tid})" if tid else ""
    stub = (
        f"{_EVICTED_MARKER}: {name}{tid_suffix} — {size} chars "
        f"evicted to save context; re-read if still needed.]"
    )
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


def _cache_hit_ratio(ctx=None, *, cache_read_tokens=0, prompt_tokens=0,
                     cache_creation_tokens=0, provider="") -> float:
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
                inner = (block["functionResponse"] or {}).get("response", {})
                inner = inner.get("content", "") if isinstance(inner, dict) else ""
                if isinstance(inner, str) and inner.startswith(_EVICTED_MARKER):
                    return True
    return False


def _stub_anthropic_tool_result(m, stub: str, name_map: dict | None = None):
    """Stub an anthropic-format tool result message (``role="user"`` with
    ``raw_content`` containing ``tool_result`` blocks).

    Returns a **copy** of ``m`` with every ``tool_result`` block's inner
    ``content`` replaced by a per-block stub, preserving the ``tool_use_id``
    pairing so the provider does not reject the request.  Text blocks (e.g.
    strategy warnings folded into the same message) are left intact.

    Per-block stubbing (BUG-2 fix): Anthropic batches parallel tool calls'
    results into ONE ``role="user"`` message with N ``tool_result`` blocks.
    The message-level ``name``/``tool_call_id`` attributes are empty for this
    shape, so a single caller-provided *stub* would (a) carry the wrong tool
    name (``"tool"``) for every block and (b) claim the *aggregated* size for
    each block. We instead build a per-block stub from that block's own
    ``tool_use_id`` → name (recovered from the preceding assistant ``tool_use``
    block via *name_map*) and its own content size, so the "re-read" hint names
    the correct tool.
    """
    raw_content = getattr(m, "raw_content", None)
    if not isinstance(raw_content, list):
        return dataclasses.replace(m, content=stub, raw_content=None)
    stubbed_raw = []
    for block in raw_content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tid = block.get("tool_use_id", "")
            bname = ""
            if name_map and tid:
                bname = name_map.get(tid, "")
            # Size of THIS block's payload only (not the aggregated message size).
            inner = block.get("content", "")
            bsize = len(inner) if isinstance(inner, str) else len(
                json.dumps(inner, ensure_ascii=False)
            ) if inner is not None else 0
            label = f"{bname} ({tid})" if bname and tid else (bname or tid or "tool")
            bstub = (
                f"{_EVICTED_MARKER}: {label} — {bsize} chars "
                f"evicted to save context; re-read if still needed.]"
            )
            stubbed_raw.append({**block, "content": bstub})
        else:
            stubbed_raw.append(block)
    return dataclasses.replace(m, content="", raw_content=stubbed_raw)


def _stub_gemini_tool_result(m, stub: str, name_map: dict | None = None):
    """Stub a Gemini-format tool result message (``role="user"`` with
    ``raw_content`` containing ``functionResponse`` parts).

    Returns a **copy** of ``m`` with every ``functionResponse`` part's inner
    ``response`` content replaced by a per-part stub, preserving the part
    structure so the Gemini API does not reject the request.  Text parts in the
    same message are left intact.

    Gemini ``functionResponse`` parts carry their OWN ``name`` (unlike
    Anthropic, whose ``tool_result`` blocks only carry an opaque
    ``tool_use_id``), so the tool name is read directly from the part.  *name_map*
    is accepted for signature symmetry with the Anthropic handler but is not
    required.
    """
    raw_content = getattr(m, "raw_content", None)
    if not isinstance(raw_content, list):
        return dataclasses.replace(m, content=stub, raw_content=None)
    stubbed_raw = []
    for block in raw_content:
        if isinstance(block, dict) and "functionResponse" in block:
            fr = block["functionResponse"] or {}
            gname = fr.get("name", "") or "tool"
            # Size of THIS part's payload only.
            resp = fr.get("response", {})
            rcontent = resp.get("content", "") if isinstance(resp, dict) else ""
            psize = len(rcontent) if isinstance(rcontent, str) else (
                len(json.dumps(resp, ensure_ascii=False)) if resp else 0
            )
            pstub = (
                f"{_EVICTED_MARKER}: {gname} — {psize} chars "
                f"evicted to save context; re-read if still needed.]"
            )
            stubbed_raw.append({
                **block,
                "functionResponse": {
                    **fr,
                    "response": {"content": pstub},
                },
            })
        else:
            stubbed_raw.append(block)
    return dataclasses.replace(m, content="", raw_content=stubbed_raw)


def _build_tool_name_map(messages) -> dict:
    """Build a ``{tool_use_id: name}`` map from assistant tool-call messages.

    Standard (``role="assistant"`` + ``tool_calls``), Anthropic-native
    (``tool_use`` blocks), and Gemini-native (``functionCall`` parts) formats
    are all scanned, so per-block eviction stubs can recover the tool name for
    ANY provider's parallel-result batch. Returns an empty dict when nothing
    matches (the stub then falls back to ``"tool"`` / the part's own name).
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
                # Gemini-native: {"functionCall": {"name": ...}}
                # (Gemini functionResponse carries its OWN name, but map the
                #  functionCall name too for completeness / future pairing.)
                if isinstance(b, dict) and "functionCall" in b:
                    tname = (b["functionCall"] or {}).get("name")
                    if tname:
                        out[tname] = tname
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
        pending_count = sum(
            1 for m in messages
            if is_tool_result(m)
            and not _is_stubbed_tool_result(m)
        )
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
    # so a batch counts as 1. The break-even cost model (~L2028) is calibrated
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
        # "1.25x rewrite" the break-even cost model (~L2028) prices. Fires at
        # most every ``batch_evict_threshold`` turns (the hysteresis gate above).
        logger.debug(
            "evict_tool_results: stubbed %d new tool result(s) "
            "(keep_recent=%d, batch_evict_threshold=%d)",
            stubbed, keep_recent, batch_evict_threshold,
        )
    return result
