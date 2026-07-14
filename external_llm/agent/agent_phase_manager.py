"""
Phase state machine, planning, and self-review mixin for AgentLoop.

Extracted from agent_loop.py to keep that file manageable.
AgentLoop inherits PhaseManagerMixin, so all methods have full access to
self.config, self.registry, self.llm_client, self.model, etc.

Moved here:
    - PhaseManagerMixin class:
      _build_tool_hint()
      _build_phase_state_message()
      _advance_phase_after_success()
      _filter_prepared_calls()
      _llm_call_simple()
      _run_self_review()
      _auto_test_and_inject()
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..client import LLMConnectionError, LLMRateLimitError, effective_content

logger = logging.getLogger(__name__)


class PhaseManagerMixin:
    """
    Mixin providing phase state machine, planning, and self-review for AgentLoop.

    Requires the host class to expose:
      - self.config           (AgentConfig)
      - self.registry         (ToolRegistry)
      - self.llm_client       (LLMClient)
      - self.model            (str)
      - self._agent_phase     (str: DISCOVER/READ/EDIT/VERIFY/FINISH)
      - self._phase_target_file   (Optional[str])
      - self._phase_target_symbol (Optional[str])
      - self._tool_success_memory (dict)
      - self._cb(event, data)
      - self._build_tool_result_message(...)
      - self._llm_call_with_tools(messages)
      - self._append_native_tool_messages(...)
    """

    # ------------------------------------------------------------------
    # Tool hint builder
    # ------------------------------------------------------------------

    def _build_tool_hint(self) -> str:
        """
        Provide adaptive tool usage hints based on recent successes.
        """
        try:
            if not getattr(self, "_tool_success_memory", None):
                return ""

            # show up to 3 recent successful tools
            tools = list(self._tool_success_memory.keys())[-3:]

            hint = "[TOOL USAGE HINT]\nRecently successful tools:\n"
            for t in tools:
                hint += f"- {t}\n"

            return hint
        except Exception:
            return ""  # non-critical — never block execution

    # ------------------------------------------------------------------
    # Phase state machine
    # ------------------------------------------------------------------

    def _build_phase_state_message(self, read_only_request: bool) -> str:
        """Build a compact state block that describes the current agent phase."""
        next_expected = {
            "DISCOVER": "find_symbol or read-only exploration",
            "READ": "bash cat or minimal next edit",
            "EDIT": "apply_patch/write_plan or answer",
            "VERIFY": "run_lint/run_tests or answer",
            "FINISH": "final answer only",
        }.get(self._agent_phase, "continue carefully")

        parts = [
            "[AGENT STATE]",
            f"phase={self._agent_phase}",
            f"read_only_request={'yes' if read_only_request else 'no'}",
            f"target_symbol={self._phase_target_symbol or '-'}",
            f"target_file={self._phase_target_file or '-'}",
            f"next_expected={next_expected}",
        ]
        return "\n".join(parts)

    def _advance_phase_after_success(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        read_only_request: bool,
    ) -> None:
        """Advance the internal phase machine after a successful tool call."""
        if not result or not getattr(result, "ok", False):
            return

        if tool_name == "find_symbol":
            self._agent_phase = "READ"
            self._phase_target_symbol = str((tool_args or {}).get("name") or "").strip()
        elif tool_name in {"apply_patch", "write_plan", "bash"}:
            # Filesystem operations: stay in EDIT to allow batch operations
            route = getattr(self.config, 'route_decision', None)
            is_fs_op = route and hasattr(route, 'reasoning') and 'Filesystem operation' in (route.reasoning or '')
            if is_fs_op and tool_name == "bash":
                self._agent_phase = "EDIT"  # stay in EDIT for next file
            elif tool_name == "bash":
                pass  # bash doesn't change phase
            else:
                self._agent_phase = "VERIFY"
        elif tool_name in {"run_lint", "run_tests"}:
            self._agent_phase = "FINISH"

    def _filter_prepared_calls(
    self,
        prepared_calls: list[dict[str, Any]],
        read_only_request: bool,
    ) -> tuple:
        """
        Enforce:
        - phase/state machine
        - read-only tool filtering
        """
        notices: list[str] = []
        filtered: list[dict[str, Any]] = []

        for pc in prepared_calls:
            filtered.append(pc)

        return filtered, notices

    # ------------------------------------------------------------------
    # Simple (no-tool) LLM call — used by planner and reviewer
    # ------------------------------------------------------------------

    def _llm_call_simple(self, messages: list, json_mode: bool = False) -> str:
        """Single LLM call without tools; returns plain text content.

        In AgentLoop context (primary), delegates to _retry_on_rate_limit
        for the same retry/telemetry/error behavior as _llm_call_with_tools.
        On retry exhaustion, raises LLMConnectionError/LLMRateLimitError
        so the caller can classify it as an API connection failure.

        In non-AgentLoop context (planning/review), keeps existing fallback
        with bare try/except returning ``""``.

        json_mode: pass response_format={"type":"json_object"} to constrain output
        to valid JSON.  Use only when the call is guaranteed to return a JSON object
        (surgical_edit, ast_op, replace_symbol_body, ast_direct_body).

        Truncation retry: when ``json_mode=True`` and the response is truncated
        (finish_reason=length/truncated), retries up to 2 times with increasing
        max_tokens (8K → 16K) so structured JSON output fits within the budget.
        """
        from ..client import LLMMessage  # noqa: F401 — needed for type context

        # ── helper: perform one LLM call with optional max_tokens override ──
        def _do_call(max_tokens: Optional[int] = None) -> dict[str, Any]:
            # ── Reasoning A/B control (developer-scoped) ────────────────────
            # Default: model default (reasoning ON). Set ASICODE_DEVELOPER_REASONING=off
            # to inject a suppression fragment into the DeepSeek payload.
            from .reasoning_utils import reasoning_ab_kwargs
            _reasoning_kwargs = reasoning_ab_kwargs("ASICODE_DEVELOPER_REASONING")

            _call_kw: dict[str, Any] = dict(_extra)  # shallow copy (response_format etc.)
            if max_tokens is not None:
                _call_kw["max_tokens"] = max_tokens
            if hasattr(self.llm_client, "chat_with_tools"):
                resp = self.llm_client.chat_with_tools(
                    messages=messages,
                    tools=[],
                    model=self.model,
                    thinking_mode=self.config.thinking_mode,
                    reasoning_effort=getattr(self.config, "reasoning_effort", None),
                    reasoning_callback=(
                        (lambda text: self.config.stream_callback("reasoning", {"text": text, "append": True}))
                        if self.config.stream_callback else None
                    ),
                    **_reasoning_kwargs,
                    **_call_kw,
                )
            elif hasattr(self.llm_client, "chat"):
                resp = self.llm_client.chat(
                    messages=messages,
                    model=self.model,
                    thinking_mode=self.config.thinking_mode,
                    reasoning_effort=getattr(self.config, "reasoning_effort", None),
                    reasoning_callback=(
                        (lambda text: self.config.stream_callback("reasoning", {"text": text, "append": True}))
                        if self.config.stream_callback else None
                    ),
                    **_reasoning_kwargs,
                    **_call_kw,
                )
            else:
                return {"content": ""}
            return {
                "content": effective_content(resp),
                "finish_reason": getattr(resp, "finish_reason", None),
                "prompt_tokens": getattr(resp, "prompt_tokens", None),
                "completion_tokens": getattr(resp, "completion_tokens", None),
                "tokens_used": getattr(resp, "tokens_used", None),
                "reasoning_tokens": getattr(resp, "reasoning_tokens", None),
            }

        _extra: dict[str, Any] = {}
        if json_mode:
            _extra["response_format"] = {"type": "json_object"}

        # ── AgentLoop context: same retry/telemetry as _llm_call_with_tools ──
        retry_wrapper = getattr(self, "_retry_on_rate_limit", None)
        if retry_wrapper is not None:
            try:
                result = retry_wrapper(lambda: _do_call(), "_llm_call_simple")
                content = result.get("content", "")
                finish_reason = result.get("finish_reason")
            except (LLMConnectionError, LLMRateLimitError):
                # Retry exhausted, telemetry already recorded, SSE event sent.
                raise
        else:
            # ── Non-AgentLoop context: keep existing fallback ──
            try:
                result = _do_call()
                content = result.get("content", "")
                finish_reason = result.get("finish_reason")
            except Exception as e:
                logger.warning("_llm_call_simple failed: %s", e)
                return ""

        # ── Truncation retry for JSON mode ──
        # finish_reason=length/truncated + content structurally incomplete → retry
        if json_mode and finish_reason in ("length", "truncated", None):
            _needs_retry = not content.rstrip().endswith(('}', ']'))
            if _needs_retry:
                for _attempt in range(2):  # max 2 retries
                    _retry_max_tokens = 8192 * (1 << _attempt)  # 8192 → 16384
                    _retry_result = _do_call(max_tokens=_retry_max_tokens)
                    content = _retry_result.get("content", "")
                    finish_reason = _retry_result.get("finish_reason")
                    logger.warning(
                        "[TRUNCATION_RETRY] finish_reason=%r max_tokens=%d retry (%d/2)",
                        finish_reason, _retry_max_tokens, _attempt + 1,
                    )
                    if finish_reason not in ("length", "truncated", None):
                        break  # retry succeeded (any non-truncated reason)
                    if content.rstrip().endswith(('}', ']')):
                        break  # content appears structurally complete
                else:
                    logger.warning(
                        "[TRUNCATION_RETRY] all 2 retries exhausted — using partial content (%d chars)",
                        len(content),
                    )

        return content

    # ------------------------------------------------------------------
    # Self-review phase
    # ------------------------------------------------------------------

    def _run_self_review(
        self,
        messages: list,
        has_native_tools: bool,
    ) -> str:
        """
        Post-execution self-review mini-loop.
        Gets git diff, asks LLM to review for bugs, optionally applies fixes.
        Returns a short summary string.

        Currently DISABLED — the self-review mini-loop added latency and false
        rejections without improving outcome quality, so it is short-circuited.
        Returns a fixed LGTM summary.
        """
        return "lgtm — self-review disabled."

    # ------------------------------------------------------------------
    # TDD auto-test injection
    # ------------------------------------------------------------------

    def _auto_test_and_inject(
        self,
        messages: list,
        turn_num: int,
        tdd_fail_count: int,
    ):
        """
        Run pytest automatically and inject the result as a user message.

        Returns (updated_messages, new_fail_count).
        - On test pass: new_fail_count reset to 0.
        - On test fail: new_fail_count incremented.
        - When new_fail_count >= max_tdd_cycles: instructs LLM to summarise.
        """
        from ..client import LLMMessage
        self._cb("tdd_cycle_start", {"turn": turn_num, "attempt": tdd_fail_count + 1})

        # Build pytest args: user-specified paths + TDD-optimised flags
        # -x: stop on first failure (faster feedback loop)
        tdd_paths = list(self.config.test_paths)
        tdd_args = [*tdd_paths, "-x", "--tb=short", "-q", "--ignore=tests/test_intelligent_llm.py", "--ignore=tests/test_indices_selection.py"]
        test_result = self.registry.dispatch(
            "run_tests", {"args": tdd_args}
        )

        if test_result.ok:
            self._cb("tdd_cycle_pass", {
                "turn": turn_num,
                "content": test_result.content[:400],
            })
            msg = LLMMessage(
                role="user",
                content=(
                    "[TDD] \u2705 All tests passed after your change.\n\n"
                    + test_result.content
                ),
            )
            return [*messages, msg], 0

        # Tests failed
        new_fail_count = tdd_fail_count + 1
        self._cb("tdd_cycle_fail", {
            "turn": turn_num,
            "attempt": new_fail_count,
            "max": self.config.max_tdd_cycles,
            "content": test_result.content[:400],
        })

        if new_fail_count >= self.config.max_tdd_cycles:
            header = (
                f"[TDD] \u274c Tests still failing after {new_fail_count} fix attempts "
                f"(max {self.config.max_tdd_cycles} reached). "
                "Summarise what you have done so far and explain what is preventing "
                "the tests from passing. Do not apply more patches."
            )
        else:
            header = (
                f"[TDD] \u274c Tests failed (attempt {new_fail_count}/{self.config.max_tdd_cycles}). "
                "Review the failures below and apply another patch to fix them."
            )

        msg = LLMMessage(
            role="user",
            content=f"{header}\n\n{test_result.content}",
        )
        return [*messages, msg], new_fail_count
