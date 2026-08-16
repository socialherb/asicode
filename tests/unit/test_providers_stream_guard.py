"""B2-follow-up: stream-iteration failures at the provider call sites must
surface as typed LLM errors (``LLMAPIError``), never as raw exceptions that
kill the whole turn.

The shared framing layer (``iter_sse_data_events``) never raises for
malformed input, but (a) a malformed-but-JSON-valid event can still trip a
consumer loop (Gemini's ``candidates[0]`` on ``{"candidates": 42}`` used to
escape as a raw ``TypeError``) and (b) any non-requests failure inside
``response.iter_bytes()`` used to escape raw from every provider loop.  Each
loop now runs through ``guard_sse_iteration``, which converts everything
except typed LLM errors and requests exceptions into ``LLMAPIError``.
"""
from __future__ import annotations

import pytest

from external_llm.client import LLMAPIError
from external_llm.providers import DeepSeekClient, GoogleClient


class _FakeStreamResponse:
    """requests.Response stand-in: HTTP 200 + raw SSE bytes + no-op close."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.status_code = 200
        self.headers = {}

    @property
    def text(self):
        return ""

    def iter_bytes(self):
        return iter(self._chunks)

    def close(self):
        pass


class _BoomStreamResponse:
    """HTTP 200 stand-in whose iter_bytes() yields one healthy event, then
    raises a plain (non-requests) exception — e.g. a proxy aborting mid-frame
    in a way requests does not classify."""

    def __init__(self):
        self.status_code = 200
        self.headers = {}
        self.text = ""

    def iter_bytes(self):
        yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n'
        raise RuntimeError("socket went away weirdly")

    def close(self):
        pass


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def post(self, *args, **kwargs):
        return self._response


def test_gemini_malformed_event_is_typed_error_not_crash():
    # RED: pre-guard this raised raw TypeError out of the loop.
    client = GoogleClient(api_key="test-key")
    client._session = _FakeSession(
        _FakeStreamResponse([b'data: {"candidates": 42}\n'])
    )
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat_with_tools(
            [], [], model="gemini-2.5-flash", token_callback=lambda _c: None
        )


def test_deepseek_stream_plain_iteration_failure_is_typed():
    # RED: pre-guard a non-requests failure inside iter_bytes() escaped raw
    # from _chat_streaming (its except clauses only know requests types).
    client = DeepSeekClient(api_key="test-key")
    client._session = _FakeSession(_BoomStreamResponse())
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client._chat_streaming(
            url="http://deepseek.test/v1/chat/completions",
            headers={},
            payload={"messages": [], "model": "deepseek-chat"},
            model="deepseek-chat",
            token_callback=lambda _c: None,
        )


def test_gemini_malformed_event_reports_diagnostic(caplog):
    # The converted error must carry a traceback at ERROR level so the
    # failure stays diagnosable even though the turn survives.
    client = GoogleClient(api_key="test-key")
    client._session = _FakeSession(
        _FakeStreamResponse([b'data: {"candidates": 42}\n'])
    )
    with pytest.raises(LLMAPIError):
        client.chat_with_tools(
            [], [], model="gemini-2.5-flash", token_callback=lambda _c: None
        )
    assert any("SSE stream iteration failed" in r.getMessage()
               for r in caplog.records)
