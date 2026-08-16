"""E3: sub-agent worker max_turns resolution follows the SSOT.

The worker's per-task budget chain (task.json → --max-turns → SSOT) used to
end in a hardcoded magic 12 while every other entry point used
``AGENT_MAX_TURNS_DEFAULT`` (500). ``_resolve_subagent_max_turns`` now ends
the chain at the SSOT; these tests pin the full resolution matrix.
"""
import argparse

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.agent.subagent_ipc import SubagentTask
from external_llm.repl.repl_impl import _resolve_subagent_max_turns


def _task(**kw) -> SubagentTask:
    return SubagentTask(task_id="t1", title="x", description="d", **kw)


class TestResolveSubagentMaxTurns:
    def test_task_budget_wins_over_args(self):
        args = argparse.Namespace(max_turns=3)
        assert _resolve_subagent_max_turns(_task(max_turns=7), args) == 7

    def test_args_budget_used_when_task_field_is_none(self):
        # Programmatic task without a budget (None): --max-turns wins.
        args = argparse.Namespace(max_turns=3)
        assert _resolve_subagent_max_turns(_task(max_turns=None), args) == 3

    def test_ssot_default_wins_over_args_for_real_task_json(self):
        # A task deserialized from task.json always carries the materialized
        # SSOT default (500) — the dataclass default, not None, so it wins
        # over a manual --max-turns (matches the historical chain order).
        args = argparse.Namespace(max_turns=3)
        assert _resolve_subagent_max_turns(_task(), args) == 500

    def test_ssot_fallback_when_both_absent(self):
        # Non-CLI launcher: Namespace without a max_turns attribute.
        assert (
            _resolve_subagent_max_turns(_task(), argparse.Namespace())
            == _cfg.counts.AGENT_MAX_TURNS_DEFAULT
        )
        assert _cfg.counts.AGENT_MAX_TURNS_DEFAULT == 500

    def test_zero_falls_through_like_unset(self):
        # Historical `or` chain semantics: 0 means "not set".
        assert (
            _resolve_subagent_max_turns(_task(max_turns=0), argparse.Namespace(max_turns=0))
            == _cfg.counts.AGENT_MAX_TURNS_DEFAULT
        )
