"""RED→GREEN: PhaseManagerMixin — tool hint/phase 전이/self-review/TDD 주입.

Mixin은 AgentLoop에 상속된다. 여기서는 최소 표면(self.config, self.registry,
self._cb, self._tool_*_memory, self._agent_phase)을 갖춘 호스트 스텁으로
직접 호출해 각 브랜치를 고정한다.
"""

from __future__ import annotations

from types import SimpleNamespace

from external_llm.agent.agent_phase_manager import PhaseManagerMixin


class _Host(PhaseManagerMixin):
    """PhaseManagerMixin이 요구하는 최소 표면을 갖춘 호스트 스텁."""

    def __init__(
        self,
        *,
        success=None,
        fail=None,
        phase="DISCOVER",
        route_decision=None,
        test_paths=(),
        max_tdd_cycles=2,
        repo_root=None,
        dispatch_result=None,
    ):
        self._tool_success_memory = success or {}
        self._tool_fail_memory = fail or {}
        self._agent_phase = phase
        self._phase_target_symbol = None
        self._phase_target_file = None
        self.config = SimpleNamespace(
            route_decision=route_decision,
            test_paths=list(test_paths),
            max_tdd_cycles=max_tdd_cycles,
        )
        self.registry = SimpleNamespace(
            repo_root=repo_root,
            dispatch=lambda tool, args: dispatch_result or SimpleNamespace(ok=True, content="all passed"),
        )
        self.events: list[tuple[str, dict]] = []

    def _cb(self, event: str, data: dict) -> None:
        self.events.append((event, data))


# ── _build_tool_hint ────────────────────────────────────────────────────


class TestBuildToolHint:
    def test_empty_memories_returns_empty(self):
        assert _Host()._build_tool_hint() == ""

    def test_success_and_fail_rendered(self):
        h = _Host(success={"k1": ("find_symbol", 3)}, fail={"k2": ("apply_patch", 1)})
        hint = h._build_tool_hint()
        assert "[TOOL USAGE HINT]" in hint
        assert "Recently successful tools:" in hint
        assert "- find_symbol (x3)" in hint
        assert "Recently failed tools:" in hint
        assert "- apply_patch (x1)" in hint

    def test_legacy_scalar_value_skipped(self):
        h = _Host(success={"k1": ("bash", 2)}, fail={"k2": "legacy-scalar"})
        hint = h._build_tool_hint()
        assert "bash (x2)" in hint
        assert "legacy-scalar" not in hint

    def test_malformed_tuple_arity_returns_empty(self):
        # name, count = val 언패킹이 ValueError를 던져 except 경로로.
        h = _Host(success={"k1": ("a", 1, 2)})
        assert h._build_tool_hint() == ""

    def test_only_three_most_recent_shown(self):
        mem = {f"k{i}": (f"tool{i}", i) for i in range(5)}
        hint = _Host(success=mem)._build_tool_hint()
        assert "tool0" not in hint
        assert "tool2" in hint and "tool4" in hint


# ── _advance_phase_after_success ────────────────────────────────────────


class TestAdvancePhase:
    def test_no_result_noop(self):
        h = _Host(phase="READ")
        h._advance_phase_after_success("find_symbol", {}, None)
        assert h._agent_phase == "READ"

    def test_failed_result_noop(self):
        h = _Host(phase="READ")
        h._advance_phase_after_success("bash", {}, SimpleNamespace(ok=False))
        assert h._agent_phase == "READ"

    def test_find_symbol_advances_to_read(self):
        h = _Host(phase="DISCOVER")
        h._advance_phase_after_success("find_symbol", {"name": "Foo"}, SimpleNamespace(ok=True))
        assert h._agent_phase == "READ"
        assert h._phase_target_symbol == "Foo"

    def test_find_symbol_strips_blank_name(self):
        h = _Host(phase="DISCOVER")
        h._advance_phase_after_success("find_symbol", {}, SimpleNamespace(ok=True))
        assert h._phase_target_symbol == ""

    def test_apply_patch_advances_to_verify(self):
        h = _Host(phase="EDIT")
        h._advance_phase_after_success("apply_patch", {}, SimpleNamespace(ok=True))
        assert h._agent_phase == "VERIFY"

    def test_write_plan_advances_to_verify(self):
        h = _Host(phase="EDIT")
        h._advance_phase_after_success("write_plan", {}, SimpleNamespace(ok=True))
        assert h._agent_phase == "VERIFY"

    def test_bash_with_filesystem_route_stays_edit(self):
        route = SimpleNamespace(reasoning="Filesystem operation: apply_patch")
        h = _Host(phase="VERIFY", route_decision=route)
        h._advance_phase_after_success("bash", {"command": "ls"}, SimpleNamespace(ok=True))
        assert h._agent_phase == "EDIT"

    def test_bash_verify_with_verification_command_finishes(self):
        h = _Host(phase="VERIFY")
        h._advance_phase_after_success("bash", {"command": "pytest tests/x.py"}, SimpleNamespace(ok=True))
        assert h._agent_phase == "FINISH"

    def test_bash_verify_without_verification_command_stays(self):
        h = _Host(phase="VERIFY")
        h._advance_phase_after_success("bash", {"command": "echo hi"}, SimpleNamespace(ok=True))
        assert h._agent_phase == "VERIFY"

    def test_bash_from_edit_does_not_reshuffle(self):
        h = _Host(phase="EDIT")
        h._advance_phase_after_success("bash", {"command": "pytest"}, SimpleNamespace(ok=True))
        assert h._agent_phase == "EDIT"


# ── _run_self_review ────────────────────────────────────────────────────


def test_self_review_short_circuits_lgtm():
    assert _Host()._run_self_review() == "lgtm — self-review disabled."


# ── _auto_test_and_inject ───────────────────────────────────────────────


class TestAutoTestAndInject:
    @staticmethod
    def _msgs():
        from external_llm.client import LLMMessage

        return [LLMMessage(role="user", content="orig")]

    def test_pass_resets_fail_count_and_injects_pass_message(self):
        h = _Host()
        msgs, new_count = h._auto_test_and_inject(self._msgs(), turn_num=3, tdd_fail_count=1)
        assert new_count == 0
        assert len(msgs) == 2
        assert "All tests passed" in msgs[-1].content
        assert ("tdd_cycle_start", {"turn": 3, "attempt": 2}) in h.events
        assert h.events[-1][0] == "tdd_cycle_pass"

    def test_fail_below_max_keeps_retry_instruction(self):
        h = _Host(max_tdd_cycles=2, dispatch_result=SimpleNamespace(ok=False, content="FAILURES"))
        msgs, new_count = h._auto_test_and_inject(self._msgs(), turn_num=1, tdd_fail_count=0)
        assert new_count == 1
        assert "Tests failed (attempt 1/2)" in msgs[-1].content
        assert "apply another patch" in msgs[-1].content
        assert h.events[-1][0] == "tdd_cycle_fail"
        assert h.events[-1][1]["max"] == 2

    def test_fail_at_max_asks_for_summary(self):
        h = _Host(max_tdd_cycles=2, dispatch_result=SimpleNamespace(ok=False, content="STILL FAILING"))
        msgs, new_count = h._auto_test_and_inject(self._msgs(), turn_num=2, tdd_fail_count=1)
        assert new_count == 2
        assert "max 2 reached" in msgs[-1].content
        assert "Do not apply more patches" in msgs[-1].content

    def test_legacy_ignore_flags_passed_only_when_target_exists(self, tmp_path):
        target = tmp_path / "tests" / "test_intelligent_llm.py"
        target.parent.mkdir(parents=True)
        target.write_text("x")
        seen = {}

        def fake_dispatch(tool, args):
            seen["args"] = args
            return SimpleNamespace(ok=True, content="ok")

        h = _Host(repo_root=str(tmp_path), test_paths=["tests/unit"])
        h.registry.dispatch = fake_dispatch
        h._auto_test_and_inject(self._msgs(), turn_num=1, tdd_fail_count=0)
        tdd_args = seen["args"]["args"]
        assert tdd_args[0] == "tests/unit"
        assert "-x" in tdd_args
        assert any(a.startswith("--ignore=tests/test_intelligent_llm.py") for a in tdd_args)

    def test_no_repo_root_skips_legacy_ignores(self):
        seen = {}

        def fake_dispatch(tool, args):
            seen["args"] = args
            return SimpleNamespace(ok=True, content="ok")

        h = _Host(test_paths=["t"])
        h.registry.dispatch = fake_dispatch
        h._auto_test_and_inject(self._msgs(), turn_num=1, tdd_fail_count=0)
        assert not any(a.startswith("--ignore=") for a in seen["args"]["args"])


class TestBuildPhaseStateMessage:
    def test_discover_phase_next_expected(self):
        h = _Host(phase="DISCOVER")
        out = h._build_phase_state_message(read_only_request=True)
        assert "phase=DISCOVER" in out
        assert "read_only_request=yes" in out
        assert "find_symbol or read-only exploration" in out
        assert "target_symbol=-" in out and "target_file=-" in out

    def test_edit_phase_with_targets(self):
        h = _Host(phase="EDIT")
        h._phase_target_symbol = "Foo"
        h._phase_target_file = "a.py"
        out = h._build_phase_state_message(read_only_request=False)
        assert "phase=EDIT" in out
        assert "read_only_request=no" in out
        assert "target_symbol=Foo" in out and "target_file=a.py" in out
        assert "apply_patch/write_plan or answer" in out

    def test_unknown_phase_falls_back(self):
        h = _Host(phase="???")
        out = h._build_phase_state_message(read_only_request=False)
        assert "continue carefully" in out


class TestBuildToolHintSuccessScalar:
    def test_legacy_scalar_in_success_dict_skipped(self):
        h = _Host(success={"k1": "scalar-no-name"})
        hint = h._build_tool_hint()
        assert hint == "[TOOL USAGE HINT]\nRecently successful tools:\n"
