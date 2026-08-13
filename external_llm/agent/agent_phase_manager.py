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
      _run_self_review()
      _auto_test_and_inject()
"""
from __future__ import annotations

import os
from typing import Any

from .tool_handlers.shell_policy import is_verification_command


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
      - self._tool_success_memory (dict[str, tuple[str, int]]: key → (tool_name, count))
      - self._tool_fail_memory    (dict[str, tuple[str, int]]: key → (tool_name, count))
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
        Provide adaptive tool usage hints based on recent successes and failures.

        Shows up to 3 most-recently-touched successes and up to 3 most-recently-
        touched failures. Touched = inserted or re-inserted: _remember_tool
        moves re-touched keys to the end (true LRU), so these are the tools used
        most recently, not the ones first seen earliest. Values are (tool_name,
        count); the dict keys are sha256 digests (see make_tool_signature) and
        must never be surfaced to the model.
        """
        try:
            success = getattr(self, "_tool_success_memory", None) or {}
            fail = getattr(self, "_tool_fail_memory", None) or {}
            if not success and not fail:
                return ""

            hint = "[TOOL USAGE HINT]\n"
            if success:
                hint += "Recently successful tools:\n"
                for _key, val in list(success.items())[-3:]:
                    if not isinstance(val, tuple):
                        continue  # legacy scalar value (no name recorded) — skip silently
                    name, count = val
                    hint += f"- {name} (x{count})\n"
            if fail:
                hint += "Recently failed tools:\n"
                for _key, val in list(fail.items())[-3:]:
                    if not isinstance(val, tuple):
                        continue  # legacy scalar value (no name recorded) — skip silently
                    name, count = val
                    hint += f"- {name} (x{count})\n"
        except (ValueError, TypeError, KeyError):  # malformed tool-state entries
            return ""  # non-critical — never block execution
        else:
            return hint

    # ------------------------------------------------------------------
    # Phase state machine
    # ------------------------------------------------------------------

    def _build_phase_state_message(self, read_only_request: bool) -> str:
        """Build a compact state block that describes the current agent phase."""
        # Every entry must name something the model can actually DO: this block
        # is injected as a system message, so an impossible instruction is spent
        # context that also invites a rejected call. Two were wrong —
        # "run_lint/run_tests" name tools with no schema (get_tool_names()
        # validation rejects them), and "bash cat" steers off read_file, whose
        # │N│ indent gutter is what prevents the old_string mismatches this
        # phase exists to avoid.
        next_expected = {
            "DISCOVER": "find_symbol or read-only exploration",
            "READ": "read_file (start_line/end_line) or minimal next edit",
            "EDIT": "apply_patch/write_plan or answer",
            "VERIFY": "bash running the tests/linter, or answer",
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
                # VERIFY -> FINISH, the transition that used to key off the
                # run_lint / run_tests TOOLS. Those were removed from the
                # schemas ("bash equivalents") but left as the only route into
                # FINISH, so FINISH had been unreachable: the model has no
                # schema for either, and get_tool_names() rejects them if it
                # emits one. Verification is a bash command now, so that is
                # what advances the phase — but only from VERIFY, because bash
                # is the general-purpose tool and must not reshuffle the machine
                # from every other state (it deliberately did nothing before).
                if self._agent_phase == "VERIFY" and is_verification_command(
                    str((tool_args or {}).get("command") or "")
                ):
                    self._agent_phase = "FINISH"
                # otherwise bash does not change phase
            else:
                self._agent_phase = "VERIFY"

    # NOTE: _filter_prepared_calls used to live here. Its docstring claimed to
    # "Enforce: phase/state machine, read-only tool filtering" and its body
    # copied the list and returned it with an empty notice list — a pass-through
    # called on every turn. Because the notices were always empty, three things
    # hanging off it were dead too: the "[PHASE RULE]" messages built from them,
    # the ``stream_callback("tool_filtered", ...)`` event, and the
    # "Tool call filter: N/M blocked (guards/phase)" log (the count could never
    # drop). All removed with it.
    #
    # Removed rather than implemented: read_only_request is a SOFT signal by
    # design, not a permission boundary — get_tool_schemas documents that it
    # "no longer filters write tools", and a7fa46b5 records why (every
    # IntentResolver failure path yields read_only_request=False, so blocking on
    # it would kill legitimate edits whenever intent resolution failed). Genuine
    # filtering of unknown / language-masked tools still happens in
    # agent_turn_pipeline._build_and_filter_prepared_calls and is untouched.
    # The phase machine is advisory only: it shapes the [AGENT STATE] hint and
    # nothing else.


    # ------------------------------------------------------------------
    # Self-review phase
    # ------------------------------------------------------------------

    def _run_self_review(self) -> str:
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
        # Legacy --ignore flags from an older repo layout. pytest silently
        # tolerates nonexistent --ignore paths, so in a repo that no longer
        # ships these files the flags were dead weight on every TDD run — pass
        # them only when the target exists (keeps the historical behavior for
        # repos that still have them).
        _repo_root = getattr(self.registry, "repo_root", None)
        _legacy_ignores = [
            _p for _p in ("tests/test_intelligent_llm.py", "tests/test_indices_selection.py")
            if _repo_root and os.path.exists(os.path.join(_repo_root, _p))
        ]
        tdd_args = [
            *tdd_paths, "-x", "--tb=short", "-q",
            *[f"--ignore={_p}" for _p in _legacy_ignores],
        ]
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
