"""Guard: no write-only planner client/model may live on AgentConfig again.

``planner_llm_client`` / ``planner_model`` carried the separate LLM client that
PlannerAgent used. PlannerAgent went with the PLANNER lane on 2026-08-03, after
which every reference in the tree was an assignment or a guard around one — the
webapp spent a service construction per request filling a field with zero
readers, and a caller who set ``planner_model`` got a run on a different model
with no indication anything had been ignored.

Two things are pinned here:

1. the fields stay off AgentConfig, so the write-only plumbing cannot come back
   by accident; and
2. the property that made them removable — nothing on the agent path consumes
   such a field — is re-measured rather than assumed, by running a real
   AgentLoop with a tripwire object in an extra config attribute.

The tripwire has a NEGATIVE CONTROL, because a silent instrument and a silent
code path are indistinguishable: the same probe is placed on ``helper_enabled``,
which ``AgentLoop.__init__`` demonstrably reads, and must fire there.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.task_router import Lane
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.client import LLMResponse, ToolCallRequest, ToolCallResponse


def test_agent_config_has_no_planner_client_or_model():
    config = AgentConfig()
    assert not hasattr(config, "planner_llm_client"), (
        "planner_llm_client is back on AgentConfig. It had zero readers; if a "
        "planner client is genuinely needed again, wire a consumer in the same "
        "change — a field only ever assigned is indistinguishable from a "
        "setting the user's request silently ignored."
    )
    assert not hasattr(config, "planner_model"), (
        "planner_model is back on AgentConfig — see above. Note /design/chat's "
        "own planner_model parameter is unrelated and still selects a model."
    )


class _Tripwire:
    """Records any attribute access / truthiness test / call against itself."""

    def __init__(self, label: str, log: list) -> None:
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_log", log)

    def _record(self, what: str):
        object.__getattribute__(self, "_log").append(what)

    def __getattr__(self, name):
        label = object.__getattribute__(self, "_label")
        self._record(f"{label}.{name}")
        return _Tripwire(f"{label}.{name}", object.__getattribute__(self, "_log"))

    def __bool__(self):
        self._record(f"bool({object.__getattribute__(self, '_label')})")
        return True

    def __call__(self, *a, **k):
        self._record(f"call({object.__getattribute__(self, '_label')})")


class _StubClient:
    """Scripted tool calls, then a final answer. No network, no API key."""

    def __init__(self, script):
        self.script = script
        self.i = 0

    def get_provider_name(self):
        return "openai"

    def chat_with_tools(self, messages, tools, model="", **kw):
        if self.i < len(self.script):
            calls = self.script[self.i]
            self.i += 1
            return ToolCallResponse(
                content="", model="stub", provider="openai", tokens_used=150,
                finish_reason="tool_calls", raw_response=None,
                tool_calls=calls, is_final=False,
                prompt_tokens=100, completion_tokens=50,
            )
        return ToolCallResponse(
            content="Done.", model="stub", provider="openai", tokens_used=150,
            finish_reason="stop", raw_response=None, tool_calls=[], is_final=True,
            prompt_tokens=100, completion_tokens=50,
        )

    def chat(self, messages, model="", **kw):
        return LLMResponse(content="ok", model="stub", provider="openai",
                           tokens_used=10, finish_reason="stop", raw_response=None)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _run(repo, config):
    script = [
        [ToolCallRequest(call_id="c1", name="read_file",
                         args={"path": "app.py", "start_line": 1, "end_line": 5})],
        [ToolCallRequest(call_id="c2", name="edit_text",
                         args={"file_path": "app.py", "old_string": "x = 1",
                               "new_string": "x = 2"})],
    ]
    registry = ToolRegistry(str(repo), config)
    loop = AgentLoop(llm_client=_StubClient(script), registry=registry,
                     config=config, model="stub-model")
    loop.config.route_decision = SimpleNamespace(
        lane=Lane.MAIN_AGENT, confidence=0.9, task_kind="general",
        reasoning="", complexity=None, target_specificity_score=0.5,
    )
    return loop.run("tripwire probe")


def test_a_planner_client_on_config_is_never_consumed(repo):
    """The measurement that justified the removal, kept runnable."""
    accesses: list = []
    config = AgentConfig(max_turns=6, rag_enabled=False)
    config.planner_llm_client = _Tripwire("planner_llm_client", accesses)
    config.planner_model = _Tripwire("planner_model", accesses)

    result = _run(repo, config)

    assert getattr(result, "status", "") == "success", "probe run must complete"
    assert (repo / "app.py").read_text(encoding="utf-8").strip() == "x = 2", (
        "the run must actually reach the write — a run that dies early touches "
        "nothing and would pass this test for the wrong reason"
    )
    assert accesses == [], f"something consumed a planner field: {accesses}"


def test_the_tripwire_fires_for_a_field_that_is_read(repo):
    """Negative control for the test above.

    AgentLoop.__init__ reads config.helper_enabled and truthiness-tests it. If
    this records nothing, the probe is broken and the assertion above is vacuous.
    """
    accesses: list = []
    config = AgentConfig(max_turns=6, rag_enabled=False)
    config.helper_enabled = _Tripwire("helper_enabled", accesses)

    _run(repo, config)

    assert accesses, "tripwire never fired on a field known to be read"
    assert any("helper_enabled" in a for a in accesses)

