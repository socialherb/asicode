"""Regression tests for AgentConfig.make_token_callback gate.

Verifies that:
1. No callback when stream_callback is None.
2. No callback when consume_content_events is False (CLI mode).
3. Callback is returned and gated when both conditions met.
4. Callback does NOT emit events for None text (reset sentinel).
5. Callback emits correct event shape for real text.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from external_llm.agent.tool_registry import AgentConfig


# ══════════════════════════════════════════════════════════════════════════════
# 1. No callback when stream_callback is None
# ══════════════════════════════════════════════════════════════════════════════

def test_no_callback_when_no_stream():
    cfg = AgentConfig(stream_callback=None, consume_content_events=True)
    assert cfg.make_token_callback() is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. No callback when consume_content_events is False (CLI mode)
# ══════════════════════════════════════════════════════════════════════════════

def test_no_callback_when_consume_disabled():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=False)
    assert cfg.make_token_callback() is None
    mock_stream.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Callback returned when both conditions met
# ══════════════════════════════════════════════════════════════════════════════

def test_callback_returned_when_gated_on():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    assert cb is not None
    assert callable(cb)


# ══════════════════════════════════════════════════════════════════════════════
# 4. No event emitted for None text (reset sentinel)
# ══════════════════════════════════════════════════════════════════════════════

def test_none_text_does_not_emit():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    cb(None)  # reset sentinel
    mock_stream.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Real text emits correct event shape
# ══════════════════════════════════════════════════════════════════════════════

def test_real_text_emits_content_event():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    cb("hello world")
    mock_stream.assert_called_once_with("content", {"text": "hello world"})


# ══════════════════════════════════════════════════════════════════════════════
# 6. Multiple tokens accumulate correctly
# ══════════════════════════════════════════════════════════════════════════════

def test_multiple_tokens_emit_sequentially():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    cb("hello")
    cb(" ")
    cb("world")
    assert mock_stream.call_count == 3
    mock_stream.assert_any_call("content", {"text": "hello"})
    mock_stream.assert_any_call("content", {"text": " "})
    mock_stream.assert_any_call("content", {"text": "world"})


# ══════════════════════════════════════════════════════════════════════════════
# 7. Empty string IS emitted (valid text, not a reset sentinel)
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_string_emitted_but_not_none():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    cb("")  # empty string is valid text
    mock_stream.assert_called_once_with("content", {"text": ""})


# ══════════════════════════════════════════════════════════════════════════════
# 8. Stream-reset: None text → no event, then real text → event
# ══════════════════════════════════════════════════════════════════════════════

def test_reset_then_emit_pattern():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    cb(None)   # reset sentinel — should not emit
    cb("new")  # first real token — should emit
    mock_stream.assert_called_once_with("content", {"text": "new"})


# ══════════════════════════════════════════════════════════════════════════════
# 9. Gate parity: stream_callback=True + consume=False → None
#    (mirrors CLI: _ProgressPrinter has no "content" handler)
# ══════════════════════════════════════════════════════════════════════════════

def test_cli_mode_gate():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=False)
    cb = cfg.make_token_callback()
    assert cb is None
    # Confirm no calls leaked
    mock_stream.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 10. Webapp mode: stream_callback=True + consume=True → callback works
# ══════════════════════════════════════════════════════════════════════════════

def test_webapp_mode_gate():
    mock_stream = MagicMock()
    cfg = AgentConfig(stream_callback=mock_stream, consume_content_events=True)
    cb = cfg.make_token_callback()
    assert cb is not None
    cb("token")
    mock_stream.assert_called_once_with("content", {"text": "token"})