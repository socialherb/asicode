"""Unit tests for ``client.iter_sse_data_events`` — the shared SSE framing
extracted from the OpenAI/DeepSeek/Gemini/Anthropic streaming clients.

Pins the framing contract so future per-client stream loops can rely on it:
``data:``-only dispatch, chunk-split line assembly (P28-1: the parser
consumes ``iter_bytes()`` and splits on ``\n`` itself — ``iter_lines()``
buffered a whole line before yielding), a per-line size cap that aborts the
stream instead of buffering an oversized frame, ``[DONE]`` as a skippable
sentinel (NOT a terminator), silent malformed-JSON skip, and order-preserving
yields.  A live ``_FakeStreamResponse`` never reaches this code path — every
caller passes a real streaming response — so a plain iterable stand-in is
sufficient.
"""
from __future__ import annotations

import json

from external_llm.client import _SSE_MAX_LINE_BYTES, iter_sse_data_events


class _FakeChunks:
    """Minimal iter_bytes() stand-in yielding raw byte chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def iter_bytes(self):
        return iter(self._chunks)


def _collect(*chunks: bytes) -> list:
    return list(iter_sse_data_events(_FakeChunks(chunks)))


def test_parses_data_frames_in_order():
    events = _collect(
        b'data: {"a": 1}\n',
        b'data: {"b": 2}\n',
        b'data: {"c": 3}\n',
    )
    assert events == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_line_spanning_chunks_is_assembled():
    # P28-1: a line split across chunk boundaries must be reassembled — the
    # old iter_lines() hid this; the manual splitter must not.
    events = _collect(
        b'data: {"a"',
        b': 1}\ndata: {"b": 2}',
        b"\n",
    )
    assert events == [{"a": 1}, {"b": 2}]


def test_crlf_line_endings():
    events = _collect(b'data: {"a": 1}\r\n', b'data: {"b": 2}\r\n')
    assert events == [{"a": 1}, {"b": 2}]


def test_skips_keepalive_and_blank_lines():
    # OpenAI sends blank keep-alive lines inside the stream.
    events = _collect(
        b"\n",
        b'data: {"a": 1}\n',
        b"\n",
        b'data: {"b": 2}\n',
    )
    assert events == [{"a": 1}, {"b": 2}]


def test_done_sentinel_is_skipped_not_terminating():
    # [DONE] must not stop iteration: a few providers may send trailing
    # frames after it, and a break here would silently drop them.
    events = _collect(
        b'data: {"a": 1}\n',
        b"data: [DONE]\n",
        b'data: {"b": 2}\n',
    )
    assert events == [{"a": 1}, {"b": 2}]


def test_skips_malformed_json_without_aborting():
    events = _collect(
        b'data: {"ok": 1}\n',
        b"data: not-json{{{\n",
        b'data: {"ok": 2}\n',
        b"data: 42\n",  # valid JSON but not a dict — still yielded
        b'data: {"ok": 3}\n',
    )
    assert events == [{"ok": 1}, {"ok": 2}, 42, {"ok": 3}]


def test_skips_non_data_frames():
    # event:/id:/retry: frames are part of the SSE spec and must be ignored.
    events = _collect(
        b"event: message\n",
        b'data: {"a": 1}\n',
        b"id: 7\n",
        b'data: {"b": 2}\n',
        b"retry: 1000\n",
    )
    assert events == [{"a": 1}, {"b": 2}]


def test_empty_data_frame_is_skipped():
    events = _collect(
        b"data:\n",
        b"data: \n",
        b'data: {"a": 1}\n',
    )
    assert events == [{"a": 1}]


def test_tail_without_trailing_newline():
    # iter_lines() yielded the final unterminated line; the manual splitter
    # must too.
    events = _collect(b'data: {"a": 1}')
    assert events == [{"a": 1}]


def test_produces_parsed_dicts_json_loads_equivalent():
    payload = '{"choices": [{"delta": {"content": "hi"}}]}'
    events = _collect(f"data: {payload}\n".encode())
    assert events == [json.loads(payload)]


def test_oversized_line_aborts_stream(caplog):
    # P28-1: a single data: frame larger than the cap must abort the stream
    # (with a warning) instead of buffering it — and nothing after it may be
    # parsed.
    big = b"data: " + b"A" * (_SSE_MAX_LINE_BYTES + 1)
    events = _collect(big + b"\n", b'data: {"after": 1}\n')
    assert events == []
    assert any("aborting stream" in r.message for r in caplog.records)


def test_oversized_line_spanning_many_chunks_aborts(caplog):
    # P28-1: the no-newline remainder check — a line arriving in many small
    # chunks must abort once the accumulated buffer exceeds the cap, NOT
    # buffer the whole line first (that would re-create the unbounded-memory
    # bug at chunk granularity).
    n = _SSE_MAX_LINE_BYTES // 64 + 2
    events = list(
        iter_sse_data_events(_FakeChunks([b"data: x"] + [b"y" * 64] * n))
    )
    assert events == []
    assert any("aborting stream" in r.message for r in caplog.records)
