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
import logging

import pytest
import requests

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


@pytest.mark.slow
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


@pytest.mark.slow
def test_many_small_lines_in_one_large_chunk_all_yielded():
    # B1: the per-line cap was applied to the WHOLE remaining buffer, so a
    # healthy stream of small lines delivered as one chunk larger than the cap
    # aborted after the very first event (the check fired on the leftover
    # multi-line buffer, not on an oversized line).  Every line here is well
    # under the cap; all must be yielded.
    line = b'data: {"i": 0}\n'
    n = _SSE_MAX_LINE_BYTES // len(line) + 2
    events = _collect(line * n)
    assert len(events) == n


def test_many_small_chunks_totaling_over_cap_all_yielded():
    # B1 companion: the stream TOTAL may exceed the cap — only an individual
    # line (or unterminated partial line) is bounded.  Many small chunks each
    # well under the cap must pass through untouched.
    line = b'data: {"i": 0}\n'
    per_chunk = 8
    chunks = [line * per_chunk] * (_SSE_MAX_LINE_BYTES // (len(line) * per_chunk) + 2)
    events = _collect(*chunks)
    assert len(events) == per_chunk * len(chunks)


def test_truncated_multibyte_tail_is_skipped_not_raised():
    # B2: EOF cutting a multi-byte UTF-8 character in half (tail path) used to
    # raise UnicodeDecodeError out of the framing loop — a turn-crashing bug
    # for Korean/Japanese/CJK responses.  The line is undecodable, so it must
    # be treated like any other malformed frame: skipped, never raised.
    line = b'data: {"x": "' + "한".encode()[:2]  # cut mid-character
    assert _collect(line) == []


def test_invalid_utf8_in_complete_line_never_raises(caplog):
    # B2: a complete line containing non-UTF-8 bytes (proxy error pages in
    # latin-1, mojibake) used to raise out of the loop.  The undecodable byte
    # becomes U+FFFD (lossy, logged at debug); if the JSON structure survives
    # the event still flows and the rest of the stream keeps going.  The
    # contract pinned here is: never raise, never truncate the stream.
    caplog.set_level(logging.DEBUG, logger="external_llm.client")
    events = _collect(b'data: {"x": "a\xffb"}\n', b'data: {"ok": 1}\n')
    assert events == [{"x": "a\ufffdb"}, {"ok": 1}]
    assert any("undecodable bytes" in r.message for r in caplog.records)


def test_invalid_utf8_split_across_chunks_never_raises():
    # B2: the same undecodable line arriving split across two chunks must not
    # raise either (chunk boundaries are arbitrary; the splitter may assemble
    # a line that never formed a valid character sequence).
    head, tail = b'data: {"x": "a\xff', b'b"}\n'
    events = _collect(head, tail)
    assert events == [{"x": "a\ufffdb"}]


# --- B2-follow-up: guard_sse_iteration (call-site defense) -----------------
# The framing/parse layer never raises for malformed input, but a transport
# quirk or a provider bug must not escape the consuming loop as a raw
# exception and kill the whole turn.  The guard converts everything except
# typed LLM errors and requests exceptions into LLMAPIError; requests
# exceptions keep their meaning so call sites can still map them to
# retryable/fatal outcomes.


class _BoomChunks(_FakeChunks):
    """iter_bytes() that yields the given chunks, then raises *exc*."""

    def __init__(self, chunks, exc):
        super().__init__(chunks)
        self._exc = exc

    def iter_bytes(self):
        yield from self._chunks
        raise self._exc


def test_guard_passthrough_normal_stream():
    from external_llm.client import guard_sse_iteration
    events = list(guard_sse_iteration(iter_sse_data_events(
        _FakeChunks([b'data: {"a": 1}\n', b'data: {"b": 2}\n']))))
    assert events == [{"a": 1}, {"b": 2}]


def test_guard_converts_plain_iteration_exception_to_llm_api_error():
    from external_llm.client import LLMAPIError, guard_sse_iteration
    gen = guard_sse_iteration(iter_sse_data_events(
        _BoomChunks([b'data: {"a": 1}\n'], RuntimeError("proxy exploded"))))
    assert next(gen) == {"a": 1}  # events delivered before the failure survive
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed: proxy exploded"):
        next(gen)


def test_guard_preserves_requests_exceptions():
    # Transport failures keep their type: call sites map ConnectionError /
    # ChunkedEncodingError to retryable outcomes, so the guard must not
    # reclassify them.
    from external_llm.client import guard_sse_iteration
    gen = guard_sse_iteration(iter_sse_data_events(
        _BoomChunks([b'data: {"a": 1}\n'],
                    requests.exceptions.ChunkedEncodingError("aborted"))))
    assert next(gen) == {"a": 1}
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        next(gen)


def test_guard_preserves_typed_llm_errors():
    # LLMCancelled (and friends) must propagate untouched — cancellation
    # semantics depend on it.
    from external_llm.client import LLMCancelled, guard_sse_iteration
    gen = guard_sse_iteration(iter_sse_data_events(
        _BoomChunks([b'data: {"a": 1}\n'], LLMCancelled())))
    assert next(gen) == {"a": 1}
    with pytest.raises(LLMCancelled):
        next(gen)


def test_guard_does_not_absorb_consumer_side_exception():
    # Python generator semantics: a raise in the consumer's for-body
    # propagates up the consumer's stack — it never enters the suspended
    # generator, so the guard cannot (and must not) convert it.  This is why
    # every call site pairs the guard with its own ``except Exception``
    # clause funneling body failures through raise_sse_iteration_failure.
    from external_llm.client import guard_sse_iteration
    with pytest.raises(TypeError):
        for _ev in guard_sse_iteration(iter_sse_data_events(
                _FakeChunks([b'data: {"candidates": 42}\n']))):
            _ = _ev["candidates"][0]  # TypeError propagates untouched
