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

import types

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
        self._advance_phase_after_success(tool, args, _Result())
        return self._agent_phase


def test_the_full_path_reaches_finish():
    """The regression: this ended at VERIFY and could go no further."""
    p = _Phase()
    assert p.advance("find_symbol", name="foo") == "READ"
    assert p.advance("apply_patch", patch="…") == "VERIFY"
    assert p.advance("bash", command="pytest tests/unit -q") == "FINISH"


@pytest.mark.parametrize(
    "command",
    [
        "pytest tests/unit",
        "python3 -m pytest -q",
        "ruff check .",
        "npm run test",
        "go test ./...",
        "cargo clippy",
        "make lint",
        "cd build\npytest",
    ],
)
def test_any_verification_runner_advances_verify(command):
    assert _Phase("VERIFY").advance("bash", command=command) == "FINISH"


@pytest.mark.parametrize(
    "command",
    ["ls -la", "cat notes.md", "pip install pytest", "grep -rn pytest .", "git commit -m 'add pytest'"],
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


def test_tdd_legacy_ignores_only_when_files_exist(tmp_path):
    """The two legacy --ignore flags (an older repo layout) must not ride on
    every TDD run in a repo that no longer ships those files — pass them only
    when the target exists, so repos that still have them keep the behavior."""
    captured: dict = {}

    class _Reg:
        repo_root = str(tmp_path)

        def dispatch(self, tool, args):
            captured["args"] = args["args"]
            return type("R", (), {"ok": True, "content": "1 passed"})

    class _Phase(PhaseManagerMixin):
        def __init__(self):
            self.config = type("C", (), {"test_paths": [], "max_tdd_cycles": 3})()
            self.registry = _Reg()

        def _cb(self, *a, **k):
            pass

    ph = _Phase()
    ph._auto_test_and_inject([], 0, 0)
    args = captured["args"]
    assert args[:3] == ["-x", "--tb=short", "-q"]
    assert "--ignore=tests/test_intelligent_llm.py" not in args
    assert "--ignore=tests/test_indices_selection.py" not in args

    # When a legacy file exists again, its ignore flag comes back.
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_intelligent_llm.py").write_text("", encoding="utf-8")
    ph._auto_test_and_inject([], 0, 0)
    args = captured["args"]
    assert "--ignore=tests/test_intelligent_llm.py" in args
    assert "--ignore=tests/test_indices_selection.py" not in args


# ---------------------------------------------------------------------------
# Tool usage hint (_build_tool_hint)
# ---------------------------------------------------------------------------
# The hint is the only consumer of _tool_success_memory and _tool_fail_memory.
# Their keys are sha256 digests (make_tool_signature) that the model cannot
# read, so the hint must print the (tool_name, count) values — and never the
# keys. These tests pin that producer→consumer contract.


def test_tool_hint_empty_when_no_memory_recorded():
    p = _Phase()
    assert p._build_tool_hint() == ""


def test_tool_hint_prints_tool_names_not_sha256_digests():
    p = _Phase()
    p._tool_success_memory = {
        "a" * 64: ("read_file", 5),
        "b" * 64: ("apply_patch", 2),
        "c" * 64: ("bash", 9),
    }
    hint = p._build_tool_hint()
    assert "read_file (x5)" in hint
    assert "apply_patch (x2)" in hint
    assert "bash (x9)" in hint
    # the keys are 64-char sha256 digests — the model must never see them
    for digest in ("a" * 64, "b" * 64, "c" * 64):
        assert digest not in hint


def test_tool_hint_shows_only_the_three_most_recent_entries():
    p = _Phase()
    p._tool_success_memory = {f"{i:064x}": (f"tool_{i}", i) for i in range(4)}
    hint = p._build_tool_hint()
    assert "tool_0" not in hint
    for i in (1, 2, 3):
        assert f"tool_{i} (x{i})" in hint


def test_tool_hint_skips_legacy_scalar_values_silently():
    """A value without a recorded name must never be printed raw."""
    p = _Phase()
    p._tool_success_memory = {"a" * 64: 3, "b" * 64: ("grep", 1)}
    hint = p._build_tool_hint()
    assert "grep (x1)" in hint
    assert ("a" * 64) not in hint
    assert "3" not in hint


def test_record_tool_success_carries_tool_name_into_hint():
    """Producer→consumer e2e: what _record_tool_success writes, the hint reads.

    Host stands in for AgentLoop with only the attributes the producer touches.
    """
    from external_llm.agent._shared_utils import make_tool_signature
    from external_llm.agent.agent_loop import AgentLoop

    host = type("H", (), {})()
    host._tool_success_memory = {}
    host._tool_fail_memory = {}
    host._tool_key = AgentLoop._tool_key
    host._record_tool_usage = types.MethodType(AgentLoop._record_tool_usage, host)
    host._shared_run_store = type("S", (), {"record_tool_usage": staticmethod(lambda *a, **k: None)})()

    AgentLoop._record_tool_success(host, "read_file", {"path": "a.py"})
    AgentLoop._record_tool_success(host, "read_file", {"path": "a.py"})
    AgentLoop._record_tool_success(host, "bash", {"command": "pytest -q"})

    key = make_tool_signature("read_file", {"path": "a.py"})
    assert host._tool_success_memory[key] == ("read_file", 2)

    hint = PhaseManagerMixin._build_tool_hint(host)
    assert "read_file (x2)" in hint
    assert "bash (x1)" in hint
    assert key not in hint


def _make_tool_memory_host():
    """Minimal AgentLoop stand-in with only the attributes the record methods touch."""
    from external_llm.agent.agent_loop import AgentLoop

    host = type("H", (), {})()
    host._tool_success_memory = {}
    host._tool_fail_memory = {}
    host._tool_key = AgentLoop._tool_key
    host._record_tool_usage = types.MethodType(AgentLoop._record_tool_usage, host)
    host._shared_run_store = type("S", (), {"record_tool_usage": staticmethod(lambda *a, **k: None)})()
    return host


def test_tool_success_memory_evicts_oldest_over_cap(monkeypatch):
    """F1: success memory is bounded — oldest inserts evicted, newest survive."""
    import external_llm.agent.agent_loop as al
    from external_llm.agent._shared_utils import make_tool_signature

    monkeypatch.setattr(al, "_TOOL_MEMORY_MAX_ENTRIES", 4)
    host = _make_tool_memory_host()
    for name in ["a", "b", "c", "d", "e", "f"]:
        al.AgentLoop._record_tool_success(host, name, {"i": name})

    assert len(host._tool_success_memory) == 4
    # oldest two inserts (a, b) evicted, newest (f) kept
    assert make_tool_signature("a", {"i": "a"}) not in host._tool_success_memory
    assert make_tool_signature("b", {"i": "b"}) not in host._tool_success_memory
    assert make_tool_signature("f", {"i": "f"}) in host._tool_success_memory


def test_tool_fail_memory_bounded_too(monkeypatch):
    """F1: fail memory (write-only) is bounded the same way."""
    import external_llm.agent.agent_loop as al
    from external_llm.agent._shared_utils import make_tool_signature

    monkeypatch.setattr(al, "_TOOL_MEMORY_MAX_ENTRIES", 3)
    host = _make_tool_memory_host()
    for name in ["a", "b", "c", "d"]:
        al.AgentLoop._record_tool_failure(host, name, {"i": name})

    assert len(host._tool_fail_memory) == 3
    assert make_tool_signature("a", {"i": "a"}) not in host._tool_fail_memory
    assert make_tool_signature("d", {"i": "d"}) in host._tool_fail_memory


def test_eviction_keeps_hint_recent_and_success_clears_fail(monkeypatch):
    """F1 + existing contract: after eviction the hint still shows the last 3
    inserts, and a success on a previously-failed key still clears the fail entry."""
    import external_llm.agent.agent_loop as al
    from external_llm.agent._shared_utils import make_tool_signature
    from external_llm.agent.agent_phase_manager import PhaseManagerMixin

    monkeypatch.setattr(al, "_TOOL_MEMORY_MAX_ENTRIES", 4)
    host = _make_tool_memory_host()
    al.AgentLoop._record_tool_failure(host, "bash", {"command": "x"})
    for name in ["a", "b", "c", "d", "e"]:
        al.AgentLoop._record_tool_success(host, name, {"i": name})
    al.AgentLoop._record_tool_success(host, "bash", {"command": "x"})

    # success on the failed key clears it from fail memory
    assert make_tool_signature("bash", {"command": "x"}) not in host._tool_fail_memory
    # memory stayed bounded
    assert len(host._tool_success_memory) == 4

    hint = PhaseManagerMixin._build_tool_hint(host)
    assert "e (x1)" in hint
    assert "bash (x1)" in hint
    assert "c (x1)" not in hint  # evicted from view along with older entries
    assert "a (x1)" not in hint
    # no raw sha256 key ever surfaces
    assert make_tool_signature("bash", {"command": "x"}) not in hint


# ---------------------------------------------------------------------------
# R5: tool memory is a true LRU — re-touching re-ranks (move-to-end)
# ---------------------------------------------------------------------------


def test_remember_tool_primitive_moves_existing_key_to_end():
    """R5 primitive contract: re-insert = delete + append."""
    from external_llm.agent.agent_loop import _remember_tool

    mem = {}
    _remember_tool(mem, "k1", 1)
    _remember_tool(mem, "k2", 2)
    _remember_tool(mem, "k3", 3)
    _remember_tool(mem, "k1", 1)  # re-touch the oldest
    assert list(mem) == ["k2", "k3", "k1"]
    assert mem["k1"] == 1


def test_tool_hint_reuses_moves_oldest_back_into_window():
    """R5: a re-used tool re-enters the hint window even after newer one-offs.

    Regression: re-inserts used to keep the key's FIRST-insertion position, so
    a tool used every turn could be pushed out of the last-3 hint window by
    tools that only ever ran once after it.
    """
    import external_llm.agent.agent_loop as al
    from external_llm.agent._shared_utils import make_tool_signature

    host = _make_tool_memory_host()
    for name in ["a", "b", "c", "d"]:
        al.AgentLoop._record_tool_success(host, name, {"i": name})
    al.AgentLoop._record_tool_success(host, "a", {"i": "a"})  # re-touch oldest

    order = list(host._tool_success_memory)
    ka = make_tool_signature("a", {"i": "a"})
    kb = make_tool_signature("b", {"i": "b"})
    assert order[-1] == ka  # re-ranked most recent
    assert order.index(ka) > order.index(kb)

    hint = PhaseManagerMixin._build_tool_hint(host)
    # last-3 window now shows a (re-touched) instead of b — the LRU fix
    assert "a (x2)" in hint
    assert "b (x1)" not in hint
    # hint order follows recency: a (re-touched) renders last of the three
    assert hint.index("c (x1)") < hint.index("d (x1)") < hint.index("a (x2)")


def test_remember_tool_eviction_spares_recently_touched(monkeypatch):
    """R5: eviction drops the least-recently-touched, not the first-inserted."""
    import external_llm.agent.agent_loop as al
    from external_llm.agent._shared_utils import make_tool_signature

    monkeypatch.setattr(al, "_TOOL_MEMORY_MAX_ENTRIES", 4)
    host = _make_tool_memory_host()
    for name in ["a", "b", "c", "d"]:
        al.AgentLoop._record_tool_success(host, name, {"i": name})
    al.AgentLoop._record_tool_success(host, "a", {"i": "a"})  # re-touch oldest
    al.AgentLoop._record_tool_success(host, "e", {"i": "e"})  # over cap → evict

    assert len(host._tool_success_memory) == 4
    # b is the least-recently-touched now; a was re-ranked and survives
    assert make_tool_signature("b", {"i": "b"}) not in host._tool_success_memory
    assert make_tool_signature("a", {"i": "a"}) in host._tool_success_memory


# ---------------------------------------------------------------------------
# F2: fail memory feeds the hint too
# ---------------------------------------------------------------------------


def test_tool_hint_includes_fail_section():
    p = _Phase()
    p._tool_success_memory = {"s" * 64: ("read_file", 5)}
    p._tool_fail_memory = {"f" * 64: ("apply_patch", 2), "g" * 64: ("bash", 3)}
    hint = p._build_tool_hint()
    assert "Recently successful tools:" in hint
    assert "read_file (x5)" in hint
    assert "Recently failed tools:" in hint
    assert "apply_patch (x2)" in hint
    assert "bash (x3)" in hint
    for digest in ("s" * 64, "f" * 64, "g" * 64):
        assert digest not in hint


def test_tool_hint_fail_only_when_success_empty():
    """F2 regression: success-empty must not blank the hint when fails exist."""
    p = _Phase()
    p._tool_success_memory = {}
    p._tool_fail_memory = {"a" * 64: ("bash", 2)}
    hint = p._build_tool_hint()
    assert hint != ""
    assert "Recently successful tools:" not in hint
    assert "Recently failed tools:" in hint
    assert "bash (x2)" in hint


def test_tool_hint_skips_legacy_scalar_in_fail_memory():
    """A fail value without a recorded name must never be printed raw."""
    p = _Phase()
    p._tool_success_memory = {}
    p._tool_fail_memory = {"a" * 64: True, "b" * 64: ("grep", 1)}
    hint = p._build_tool_hint()
    assert "grep (x1)" in hint
    assert ("a" * 64) not in hint
    assert "True" not in hint


def test_tool_hint_fail_section_shows_three_most_recent():
    p = _Phase()
    p._tool_success_memory = {}
    p._tool_fail_memory = {f"{i:064x}": (f"bad_{i}", i) for i in range(4)}
    hint = p._build_tool_hint()
    assert "bad_0" not in hint
    for i in (1, 2, 3):
        assert f"bad_{i} (x{i})" in hint


def test_record_tool_failure_carries_tool_name_into_hint():
    """Producer→consumer e2e for the fail channel: what _record_tool_failure
    writes, the hint reads — names only, never the sha256 keys."""
    from external_llm.agent._shared_utils import make_tool_signature
    from external_llm.agent.agent_loop import AgentLoop

    host = _make_tool_memory_host()
    AgentLoop._record_tool_failure(host, "apply_patch", {"patch": "x"})
    AgentLoop._record_tool_failure(host, "apply_patch", {"patch": "x"})
    AgentLoop._record_tool_failure(host, "bash", {"command": "bad"})

    key = make_tool_signature("apply_patch", {"patch": "x"})
    assert host._tool_fail_memory[key] == ("apply_patch", 2)

    hint = PhaseManagerMixin._build_tool_hint(host)
    assert "Recently failed tools:" in hint
    assert "apply_patch (x2)" in hint
    assert "bash (x1)" in hint
    assert key not in hint


def test_record_tool_usage_routes_by_ok_and_clears_fail_on_success():
    """R1: the shared core routes by ok — success clears the fail entry (F2
    recovery), and the adaptive learning channel receives the exact ok flag."""
    from external_llm.agent._shared_utils import make_tool_signature
    from external_llm.agent.agent_loop import AgentLoop

    seen = []
    host = _make_tool_memory_host()
    host._shared_run_store = type("S", (), {"record_tool_usage": staticmethod(lambda *a, **k: seen.append(a))})()

    key = make_tool_signature("apply_patch", {"patch": "x"})
    AgentLoop._record_tool_usage(host, "apply_patch", {"patch": "x"}, False)
    AgentLoop._record_tool_usage(host, "apply_patch", {"patch": "x"}, True)

    assert key not in host._tool_fail_memory  # success dropped the fail entry
    assert host._tool_success_memory[key] == ("apply_patch", 1)
    assert seen == [("MAIN_AGENT", "apply_patch", False, ""), ("MAIN_AGENT", "apply_patch", True, "")]


def test_record_tool_usage_failure_leaves_success_memory_untouched():
    """R1: a failure must not create or drop success entries — the two channels
    only interact in one direction (success clears failure)."""
    from external_llm.agent._shared_utils import make_tool_signature
    from external_llm.agent.agent_loop import AgentLoop

    host = _make_tool_memory_host()
    key = make_tool_signature("bash", {"command": "x"})
    AgentLoop._record_tool_usage(host, "bash", {"command": "x"}, False)

    assert host._tool_fail_memory[key] == ("bash", 1)
    assert key not in host._tool_success_memory
