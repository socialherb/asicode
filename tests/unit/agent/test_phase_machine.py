"""The advisory phase machine must reach FINISH and advise possible actions.

`_advance_phase_after_success` drives a DISCOVER→READ→EDIT→VERIFY→FINISH state
whose only externally visible effect is the `[AGENT STATE]` block injected as a
system message. Two things had rotted:

* the sole transition into FINISH keyed off the `run_lint` / `run_tests` TOOLS,
  which were removed from `AGENT_TOOL_SCHEMAS` ("bash equivalents; kept as
  internal dispatch only"). The model has no schema for either and
  `get_tool_names()` validation rejects them, so FINISH became unreachable;
* `next_expected` advertised those same two tools for VERIFY, and `bash cat`
  for READ — the reading method that omits read_file's `│N│` indent gutter.

The machine is advisory: nothing is blocked on the phase. These tests therefore
assert on reachability and on the advice text, which is all it produces.
"""
from __future__ import annotations

import pytest

from external_llm.agent.agent_phase_manager import PhaseManagerMixin
from external_llm.agent.tool_schemas import AGENT_TOOL_NAMES


class _Result:
    ok = True


class _Phase(PhaseManagerMixin):
    """Minimal host exposing only what the phase methods touch."""

    def __init__(self, phase="DISCOVER"):
        self._agent_phase = phase
        self._phase_target_symbol = ""
        self._phase_target_file = ""
        self.config = type("C", (), {"route_decision": None})()

    def advance(self, tool, **args):
        self._advance_phase_after_success(tool, args, _Result(), False)
        return self._agent_phase


def test_the_full_path_reaches_finish():
    """The regression: this ended at VERIFY and could go no further."""
    p = _Phase()
    assert p.advance("find_symbol", name="foo") == "READ"
    assert p.advance("apply_patch", patch="…") == "VERIFY"
    assert p.advance("bash", command="pytest tests/unit -q") == "FINISH"


@pytest.mark.parametrize(
    "command",
    ["pytest tests/unit", "python3 -m pytest -q", "ruff check .",
     "npm run test", "go test ./...", "cargo clippy", "make lint",
     "cd build\npytest"],
)
def test_any_verification_runner_advances_verify(command):
    assert _Phase("VERIFY").advance("bash", command=command) == "FINISH"


@pytest.mark.parametrize(
    "command",
    ["ls -la", "cat notes.md", "pip install pytest", "grep -rn pytest .",
     "git commit -m 'add pytest'"],
)
def test_ordinary_bash_does_not_advance_verify(command):
    """`bash` is the general-purpose tool; only an actual verification run
    means the edit was checked."""
    assert _Phase("VERIFY").advance("bash", command=command) == "VERIFY"


@pytest.mark.parametrize("phase", ["DISCOVER", "READ", "EDIT"])
def test_bash_still_does_not_reshuffle_other_phases(phase):
    """Pre-existing behaviour: bash was deliberately phase-neutral. Only the
    VERIFY exit is new, so every other state must be untouched — including by
    a command that happens to run tests."""
    assert _Phase(phase).advance("bash", command="pytest tests/unit") == phase


def test_no_advice_names_a_tool_the_model_was_not_given():
    """The defect that made FINISH unreachable also leaked into the prompt."""
    for phase in ("DISCOVER", "READ", "EDIT", "VERIFY", "FINISH"):
        block = _Phase(phase)._build_phase_state_message(read_only_request=True)
        for dead in ("run_lint", "run_tests"):
            assert dead not in block, f"{phase} advises unavailable tool {dead}"
        for named in ("find_symbol", "apply_patch", "write_plan", "read_file", "bash"):
            if named in block:
                assert named in AGENT_TOOL_NAMES, f"{phase} names unavailable {named}"


def test_read_phase_points_at_read_file_not_bash_cat():
    block = _Phase("READ")._build_phase_state_message(read_only_request=True)
    assert "read_file" in block
    assert "bash cat" not in block


def test_the_phase_filter_stub_is_gone():
    """It claimed to enforce the phase machine and read-only filtering, and was
    a pass-through. Kept as a test because "add it back as a no-op" is the
    tempting shape when a caller wants a hook."""
    assert not hasattr(PhaseManagerMixin, "_filter_prepared_calls")
