"""Regression guard: DeepSeek v4 ``thinking_mode=False`` must send ``thinking:{type:disabled}``.

On the OpenCode Go / OpenRouter endpoints (routed through ``OpenAIClient``),
DeepSeek v4 is a reasoning model. The OLD code translated
``thinking_mode=False`` into ``reasoning_effort="low"``, which only *dials down*
reasoning (measured 882→742 tokens) — it does NOT disable it. Measured on
OpenCode Go::

    reasoning_effort="low"            → 742 reasoning tokens, 8.4s
    thinking:{"type":"disabled"}       →   0 reasoning tokens, 1.8s

The native ``thinking`` parameter is the ONLY way to produce zero reasoning
tokens; both OpenCode Go and native DeepSeek honor it (per the docstring on
``_is_reasoning_model``). The fix routes DeepSeek v4 to the native ``thinking``
parameter (matching ``DeepSeekClient`` exactly) and keeps OpenAI o-series /
gpt-5 on the ``reasoning_effort`` dial (which lack the ``thinking`` parameter).

This pins three layers:
  1. ``_is_deepseek_v4`` classifier (prefix-stripping).
  2. ``_apply_thinking_mode`` payload mutation (unit).
  3. The full ``chat()`` / ``chat_with_tools()`` payload (integration).
"""
from __future__ import annotations

import json
from typing import ClassVar

import pytest

from external_llm.client import LLMMessage
from external_llm.openai_client import (
    OpenAIClient,
    _apply_thinking_mode,
    _is_deepseek_v4,
)

# ── 1. classifier ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek-v4-flash", True),
        ("deepseek-v4-pro", True),
        ("deepseek-v4", True),
        # Provider/route prefixes must be stripped.
        ("deepseek/deepseek-v4-flash", True),
        ("openrouter/deepseek/deepseek-v4-pro", True),
        # Negative cases — these must NOT match.
        ("deepseek-chat", False),
        ("deepseek-reasoner", False),  # legacy reasoner, not v4
        ("glm-5.2", False),
        ("o3", False),
        ("gpt-5", False),
        ("", False),
    ],
)
def test_is_deepseek_v4(model: str, expected: bool) -> None:
    assert _is_deepseek_v4(model) is expected


# ── 2. payload mutation (unit) ─────────────────────────────────────────────

def test_apply_thinking_deepseek_off_sends_disabled_no_effort() -> None:
    """DeepSeek v4 OFF = native thinking:disabled + NO reasoning_effort.

    reasoning_effort='low' only dials reasoning down (882→742 tok); it must NOT
    be sent alongside (or instead of) the native disable.
    """
    p: dict = {}
    _apply_thinking_mode(p, "deepseek-v4-flash", False, None, is_reasoning=True)
    assert p == {"thinking": {"type": "disabled"}}


def test_apply_thinking_deepseek_on_sends_enabled() -> None:
    p: dict = {}
    _apply_thinking_mode(p, "deepseek/deepseek-v4-pro", True, None, is_reasoning=True)
    assert p == {"thinking": {"type": "enabled"}}


def test_apply_thinking_deepseek_on_with_effort_override() -> None:
    p: dict = {}
    _apply_thinking_mode(p, "deepseek-v4-flash", True, "high", is_reasoning=True)
    assert p == {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}


def test_apply_thinking_openai_o3_keeps_reasoning_effort() -> None:
    """OpenAI o-series lacks the `thinking` param — stays on reasoning_effort."""
    p: dict = {}
    _apply_thinking_mode(p, "o3", False, None, is_reasoning=True)
    assert p == {"reasoning_effort": "low"}
    assert "thinking" not in p


def test_apply_thinking_gpt5_off_uses_minimal() -> None:
    p: dict = {}
    _apply_thinking_mode(p, "gpt-5", False, None, is_reasoning=True)
    assert p == {"reasoning_effort": "minimal"}


def test_apply_thinking_none_is_noop() -> None:
    """No thinking_mode toggle → nothing added (default provider behavior)."""
    p: dict = {}
    _apply_thinking_mode(p, "deepseek-v4-flash", None, "high", is_reasoning=True)
    assert p == {}


# ── 3. full path: chat() / chat_with_tools() payload ───────────────────────

class _FakeResp:
    status_code: int = 200
    headers: ClassVar[dict] = {}

    def json(self):
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    @property
    def text(self):
        return json.dumps(self.json())


def _capture_client(monkeypatch):
    """OpenAIClient whose session POST records each payload body."""
    import external_llm.openai_client as oc
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    c = OpenAIClient(api_key="test")
    c.base_url = "https://opencode.ai/v1"
    captured: list[dict] = []

    class _S:
        pass

    c._session = _S()
    c._session.post = lambda *a, **k: (captured.append(k.get("json")) or _FakeResp())
    return c, captured


def test_chat_deepseek_off_sends_thinking_disabled(monkeypatch):
    """THE BUG: chat() with DeepSeek v4 + thinking_mode=False must send the
    native thinking:disabled (0 reasoning tokens), not reasoning_effort='low'."""
    c, cap = _capture_client(monkeypatch)
    c.chat([LLMMessage(role="user", content="hi")],
           model="deepseek/deepseek-v4-flash", thinking_mode=False, max_tokens=1000)
    p = cap[-1]
    assert p.get("thinking") == {"type": "disabled"}
    assert "reasoning_effort" not in p, (
        "reasoning_effort='low' only dials reasoning down (882→742 tok); the native "
        "thinking:disabled is the only way to reach 0 reasoning tokens"
    )


def test_chat_deepseek_on_sends_thinking_enabled(monkeypatch):
    c, cap = _capture_client(monkeypatch)
    c.chat([LLMMessage(role="user", content="hi")],
           model="deepseek/deepseek-v4-flash", thinking_mode=True, max_tokens=1000)
    assert cap[-1].get("thinking") == {"type": "enabled"}


def test_chat_o3_off_keeps_reasoning_effort(monkeypatch):
    """OpenAI o3 has no `thinking` param — must stay on reasoning_effort."""
    c, cap = _capture_client(monkeypatch)
    c.chat([LLMMessage(role="user", content="hi")],
           model="o3", thinking_mode=False, max_tokens=1000)
    p = cap[-1]
    assert p.get("reasoning_effort") == "low"
    assert "thinking" not in p


def test_chat_with_tools_deepseek_off_sends_thinking_disabled(monkeypatch):
    """chat_with_tools() parity with chat() — same dispatch helper."""
    c, cap = _capture_client(monkeypatch)
    c.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                      model="deepseek/deepseek-v4-flash", thinking_mode=False, max_tokens=1000)
    p = cap[-1]
    assert p.get("thinking") == {"type": "disabled"}
    assert "reasoning_effort" not in p
