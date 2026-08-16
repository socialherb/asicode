"""AnthropicClient.chat_with_tools must honor dict-form messages consistently.

The message loop explicitly supports BOTH LLMMessage objects and plain dicts
for role/content (``_dict = isinstance(msg, dict)``), but the tool-specific
fields (tool_call_id / tool_calls / raw_content / images) were read via
getattr() only — so a dict-form message silently lost its tool metadata:

* a dict assistant message with ``tool_calls`` fell through to the plain-text
  branch (no tool_use blocks) while the following dict tool result still
  became a tool_result block → orphaned tool_result → Anthropic HTTP 400
  "unexpected tool_use_id found in tool_result blocks";
* a dict tool message lost ``tool_call_id`` (sent as "").

openai_client.py fails fast on dicts (direct attribute access); providers.py
assumes objects. Only anthropic_client half-supports the dict form — these
tests pin the consistent behavior: dict messages keep their tool metadata.
"""
from __future__ import annotations

import json

from external_llm.anthropic_client import AnthropicClient
from external_llm.client import LLMMessage


class _Capture:
    """Fake session.post that records the JSON payload of every call."""

    def __init__(self, response):
        self.response = response
        self.payloads: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.payloads.append(kwargs.get("json"))
        return self.response


class _FakeJsonResponse:
    def __init__(self, data):
        self.data = data
        self.status_code = 200
        self.headers: dict = {}

    def json(self):
        return self.data


_OK = _FakeJsonResponse({
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
})

_TOOLS = [{"name": "grep", "description": "search", "parameters": {"type": "object", "properties": {}}}]


def _client_with_capture(monkeypatch) -> tuple[AnthropicClient, _Capture]:
    client = AnthropicClient(api_key="test")
    cap = _Capture(_OK)
    monkeypatch.setattr(client._session, "post", cap)
    return client, cap


def _api_msgs(cap: _Capture) -> list[dict]:
    assert cap.payloads, "session.post was never called"
    return cap.payloads[0]["messages"]


def test_dict_assistant_tool_calls_become_tool_use_blocks(monkeypatch):
    client, cap = _client_with_capture(monkeypatch)
    client.chat_with_tools(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "tc_1", "function": {"name": "grep", "arguments": json.dumps({"q": "x"})}}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "match found"},
        ],
        _TOOLS,
    )
    msgs = _api_msgs(cap)
    # assistant turn must carry a tool_use block with the parsed input
    assistant = msgs[1]
    blocks = assistant["content"]
    tool_use = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
    assert tool_use, f"dict assistant tool_calls were dropped: {assistant!r}"
    assert tool_use[0]["id"] == "tc_1"
    assert tool_use[0]["name"] == "grep"
    assert tool_use[0]["input"] == {"q": "x"}


def test_dict_tool_result_keeps_tool_use_id(monkeypatch):
    client, cap = _client_with_capture(monkeypatch)
    client.chat_with_tools(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "tc_42", "function": {"name": "grep", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "tc_42", "content": "ok"},
        ],
        _TOOLS,
    )
    msgs = _api_msgs(cap)
    tool_turn = msgs[2]
    assert tool_turn["role"] == "user"
    block = tool_turn["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tc_42", "dict tool_call_id silently became ''"


def test_dict_raw_content_preferred(monkeypatch):
    client, cap = _client_with_capture(monkeypatch)
    raw = [{"type": "text", "text": "native"}]
    client.chat_with_tools([{"role": "user", "content": "plain", "raw_content": raw}], _TOOLS)
    msgs = _api_msgs(cap)
    # _mark_last_message_for_caching may append cache_control to the last
    # block; compare ignoring that marker only.
    got = msgs[0]["content"]
    assert [{k: v for k, v in b.items() if k != "cache_control"} for b in got] == raw


def test_dict_images_become_image_blocks(monkeypatch):
    client, cap = _client_with_capture(monkeypatch)
    client.chat_with_tools(
        [{
            "role": "user", "content": "look",
            "images": [{"media_type": "image/png", "data": "aGk="}],
        }],
        _TOOLS,
    )
    msgs = _api_msgs(cap)
    blocks = msgs[0]["content"]
    img = [b for b in blocks if b.get("type") == "image"]
    assert img, f"dict images were dropped: {blocks!r}"
    assert img[0]["source"]["data"] == "aGk="
    assert next(b for b in blocks if b.get("type") == "text")["text"] == "look"


def test_object_messages_still_work(monkeypatch):
    """LLMMessage objects (the production path) must keep the exact prior behavior."""
    client, cap = _client_with_capture(monkeypatch)
    client.chat_with_tools(
        [
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="",
                       tool_calls=[{"id": "tc_9", "function": {"name": "grep", "arguments": "{}"}}]),
            LLMMessage(role="tool", content="ok", tool_call_id="tc_9"),
        ],
        _TOOLS,
    )
    msgs = _api_msgs(cap)
    assert next(b for b in msgs[1]["content"] if b.get("type") == "tool_use")["id"] == "tc_9"
    assert msgs[2]["content"][0]["tool_use_id"] == "tc_9"


def test_object_raw_content_and_images_still_work(monkeypatch):
    client, cap = _client_with_capture(monkeypatch)
    raw = [{"type": "text", "text": "native"}]
    client.chat_with_tools([LLMMessage(role="user", content="c", raw_content=raw)], _TOOLS)
    got = _api_msgs(cap)[0]["content"]
    assert [{k: v for k, v in b.items() if k != "cache_control"} for b in got] == raw

    client2, cap2 = _client_with_capture(monkeypatch)
    client2.chat_with_tools(
        [LLMMessage(role="user", content="c", images=[{"media_type": "image/png", "data": "aGk="}])],
        _TOOLS,
    )
    blocks = _api_msgs(cap2)[0]["content"]
    assert [b for b in blocks if b.get("type") == "image"]


# ── Null / partial usage → None-safe token accounting ──────────────────────
# The plain chat() path once used usage.get(key, 0), which TypeErrors on
# `"input_tokens": null` (explicit-null usage blocks), while the tools path
# already used the `or 0` idiom. These pin that the plain chat path is
# None-safe too and keeps tokens_used an int (downstream accounting treats
# int 0 and None differently).

_NULL_USAGE = {
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": None, "output_tokens": None},
}

_MISSING_USAGE = {
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
}


def _chat_client_with_response(monkeypatch, data):
    client = AnthropicClient(api_key="test")
    cap = _Capture(_FakeJsonResponse(data))
    monkeypatch.setattr(client._session, "post", cap)
    return client


def test_chat_null_usage_tokens_never_typeerror(monkeypatch):
    """Plain chat() with explicit-null usage must not crash (was TypeError)."""
    client = _chat_client_with_response(monkeypatch, _NULL_USAGE)
    resp = client.chat([LLMMessage(role="user", content="hi")])
    assert resp.tokens_used == 0
    assert resp.content == "done"


def test_chat_missing_usage_keeps_zero_tokens(monkeypatch):
    """No usage block at all stays 0 (prior behavior preserved)."""
    client = _chat_client_with_response(monkeypatch, _MISSING_USAGE)
    assert client.chat([LLMMessage(role="user", content="hi")]).tokens_used == 0


def test_chat_positive_usage_still_summed(monkeypatch):
    """Normal usage keeps summing input+output."""
    ok = dict(_NULL_USAGE)
    ok["usage"] = {"input_tokens": 5, "output_tokens": 3}
    client = _chat_client_with_response(monkeypatch, ok)
    assert client.chat([LLMMessage(role="user", content="hi")]).tokens_used == 8
