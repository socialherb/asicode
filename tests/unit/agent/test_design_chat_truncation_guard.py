"""Truncation guard in the design-chat tool loop.

A response cut off at max_tokens (finish_reason=length/truncated) may carry a
PARTIAL tool call whose arguments were truncated mid-JSON. Executing it would
dispatch stale/partial args (e.g. a bash command cut mid-way) — mirroring
agent_loop's contract, the design-chat loop clears such tool calls (keeping the
text content) and records the truncation on the turn-level outcome channel
(record_agent_result), the only place a truncation storm surfaces in the
metrics.
"""

import subprocess
from unittest import mock

import pytest

from external_llm.agent import design_chat_loop as dcl_mod
from external_llm.agent.config.thresholds import config as _cfg
from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.client import LLMMessage, ToolCallRequest, ToolCallResponse


class _StubClient:
    """Scripted stub: returns a fixed ToolCallResponse per call."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    @staticmethod
    def get_provider_name() -> str:
        return "stub"

    def chat_with_tools(self, messages, tools, model, **kw):
        self.calls += 1
        return self._response

    def chat(self, messages, model, **kw):
        raise AssertionError("plain chat must not be used in this test")


class _ScriptedClient:
    """Scripted stub: returns responses in order (last one repeats), recording
    the max_tokens budget each call received."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.max_tokens_seen = []

    @staticmethod
    def get_provider_name() -> str:
        return "stub"

    def chat_with_tools(self, messages, tools, model, **kw):
        self.calls += 1
        self.max_tokens_seen.append(kw.get("max_tokens"))
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]

    def chat(self, messages, model, **kw):
        raise AssertionError("plain chat must not be used in this test")


_BASE_BUDGET = _cfg.tokens.AGENT_TOOL_CALL * 2


def _tool_response(content: str, tool_calls, finish_reason: str):
    return ToolCallResponse(
        content=content,
        model="m",
        provider="stub",
        tokens_used=1,
        finish_reason=finish_reason,
        raw_response=None,
        tool_calls=tool_calls,
    )


@pytest.fixture
def _repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("hello\n")
    return str(tmp_path)


def _make_loop(client, _repo):
    reg = ToolRegistry(_repo, AgentConfig())
    loop = DesignChatLoop(client, reg, "stub-model")
    # Never actually execute tools — any dispatch call is a test failure signal.
    reg.dispatch = mock.Mock(return_value="")
    return loop, reg


# ── finish_reason=length with a partial tool call ────────────────────────────


def test_truncated_tool_calls_cleared_and_not_dispatched(_repo):
    """After the 3-attempt budget ladder (agent_loop parity) exhausts, the
    partial tool call is cleared (never dispatched), while the text content is
    preserved so the turn loop can continue naturally."""
    client = _StubClient(
        _tool_response(
            "partial text",
            [ToolCallRequest(call_id="c1", name="read_file", args={"path": "README.md"})],
            finish_reason="length",
        )
    )
    loop, reg = _make_loop(client, _repo)
    res = loop.respond([LLMMessage(role="user", content="do stuff")])
    assert reg.dispatch.call_count == 0
    assert res.content == "partial text"  # text preserved
    assert client.calls == 3  # 3-attempt budget ladder exhausts, then the guard clears


def test_finish_reason_truncated_value_also_guarded(_repo):
    client = _StubClient(
        _tool_response(
            "partial",
            [ToolCallRequest(call_id="c1", name="read_file", args={"path": "README.md"})],
            finish_reason="truncated",
        )
    )
    loop, reg = _make_loop(client, _repo)
    loop.respond([LLMMessage(role="user", content="do stuff")])
    assert reg.dispatch.call_count == 0


# ── Outcome-channel recording (truncation storm visibility) ─────────────────


def test_truncation_recorded_on_outcome_channel(_repo, monkeypatch):
    """finish_reason=length records record_agent_result(truncated=True): the
    LLM call itself succeeded, so record_llm_call(failed=True) never fires and
    without this channel a truncation storm is invisible in the metrics."""
    fake = mock.Mock()
    fake.record_llm_call = mock.Mock()
    fake.record_agent_result = mock.Mock()
    monkeypatch.setattr(dcl_mod, "get_global_collector", lambda: fake)
    client = _StubClient(
        _tool_response(
            "partial",
            [ToolCallRequest(call_id="c1", name="read_file", args={"path": "README.md"})],
            finish_reason="length",
        )
    )
    loop, _reg = _make_loop(client, _repo)
    loop.respond([LLMMessage(role="user", content="do stuff")])
    fake.record_agent_result.assert_called_once_with(truncated=True)


def test_normal_stop_not_recorded_as_truncation(_repo, monkeypatch):
    """A clean finish_reason=stop turn must NOT trip the truncation channel."""
    fake = mock.Mock()
    fake.record_llm_call = mock.Mock()
    fake.record_agent_result = mock.Mock()
    monkeypatch.setattr(dcl_mod, "get_global_collector", lambda: fake)
    client = _StubClient(
        _tool_response(
            "final answer",
            [],
            finish_reason="stop",
        )
    )
    loop, _reg = _make_loop(client, _repo)
    res = loop.respond([LLMMessage(role="user", content="do stuff")])
    assert res.content == "final answer"
    fake.record_agent_result.assert_not_called()


# ── Truncation max_tokens ladder (agent_loop parity) ────────────────────────


def test_truncation_ladder_doubles_budget_and_succeeds(_repo):
    """finish_reason=length retries with a doubled max_tokens budget (agent_loop
    parity); on success the turn proceeds normally — the truncated partial tool
    call is never dispatched."""
    client = _ScriptedClient(
        [
            _tool_response(
                "partial",
                [ToolCallRequest(call_id="c1", name="read_file", args={"path": "README.md"})],
                finish_reason="length",
            ),
            _tool_response("final answer", [], finish_reason="stop"),
        ]
    )
    loop, reg = _make_loop(client, _repo)
    res = loop.respond([LLMMessage(role="user", content="do stuff")])
    assert client.calls == 2
    assert client.max_tokens_seen == [_BASE_BUDGET, _BASE_BUDGET * 2]
    assert res.content == "final answer"
    assert reg.dispatch.call_count == 0  # truncated call never dispatched


def test_truncation_ladder_two_retries_then_succeeds(_repo):
    """Both truncation signals ("length" and "truncated") climb the same ladder:
    base -> 2x -> 4x, then a clean response ends the turn."""
    client = _ScriptedClient(
        [
            _tool_response("p1", [], finish_reason="length"),
            _tool_response("p2", [], finish_reason="truncated"),
            _tool_response("final", [], finish_reason="stop"),
        ]
    )
    loop, _reg = _make_loop(client, _repo)
    res = loop.respond([LLMMessage(role="user", content="do stuff")])
    assert client.calls == 3
    assert client.max_tokens_seen == [_BASE_BUDGET, _BASE_BUDGET * 2, _BASE_BUDGET * 4]
    assert res.content == "final"
