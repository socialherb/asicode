"""Tests for ``_parse_text_tool_calls`` — the text-mode (non-native tool-calling)
JSON tool-call extractor used for some Ollama/local models.

The load-bearing invariant is the **fence-boundary** rule: a tool call written
OUTSIDE a code fence is only recovered by the stage-3 free-text fallback, which
runs SOLELY when no fenced call was found. So a free-text JSON *example* is
never misexecuted alongside a real fenced call — a text-mode model that writes
'예시: {"name": "edit_text", ...}' followed by a real fenced call must run only
the fenced one.
"""
from external_llm.agent.design_chat_loop import _parse_text_tool_calls


def _names(calls):
    return [c["name"] for c in calls]


# ── Stage 1: whole-content JSON ───────────────────────────────────────────────

def test_whole_content_json_object():
    r = _parse_text_tool_calls('{"name":"read_file","arguments":{"path":"a.py"}}')
    assert _names(r) == ["read_file"]


def test_whole_content_json_list():
    r = _parse_text_tool_calls('[{"name":"read_file","arguments":{}},{"name":"a","arguments":{}}]')
    assert _names(r) == ["read_file", "a"]


def test_empty_and_blank_returns_empty():
    assert _parse_text_tool_calls("") == []
    assert _parse_text_tool_calls("   \n\n  ") == []


# ── Stage 2: fenced blocks (odd-index segments only) ──────────────────────────

def test_fenced_json_block_parsed():
    content = '```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```'
    assert _names(_parse_text_tool_calls(content)) == ["read_file"]


def test_fenced_block_without_language_tag():
    content = '```\n{"name":"read_file","arguments":{"path":"a.py"}}\n```'
    assert _names(_parse_text_tool_calls(content)) == ["read_file"]


def test_multiple_fenced_calls_all_parsed():
    content = (
        '```\n{"name":"a","arguments":{}}\n```\n'
        '```\n{"name":"b","arguments":{}}\n```'
    )
    assert _names(_parse_text_tool_calls(content)) == ["a", "b"]


def test_unbalanced_opening_fence_still_parsed():
    # A single opening fence (no closer) makes the trailing segment odd-indexed.
    content = 'call:\n```json\n{"name":"read_file","arguments":{"path":"a.py"}}'
    assert _names(_parse_text_tool_calls(content)) == ["read_file"]


# ── Fence-boundary guard (the regression this round fixes) ────────────────────

def test_free_text_example_suppressed_when_fenced_call_present():
    """KEY: a JSON-shaped example in free text must NOT run when a real fenced
    call also exists. Previously stage-2 scanned ALL split segments (including
    the even-index free-text one), so the example was misexecuted alongside the
    real call."""
    content = (
        '예시: {"name": "edit_text", "arguments": {"path": "x.py"}} 처럼 쓰세요\n'
        '```\n{"name":"read_file","arguments":{"path":"real.py"}}\n```'
    )
    assert _names(_parse_text_tool_calls(content)) == ["read_file"]


def test_free_text_example_before_and_after_fenced_not_run():
    content = (
        'before: {"name":"edit_text","arguments":{}}\n'
        '```\n{"name":"read_file","arguments":{}}\n```\n'
        'after: {"name":"delete","arguments":{}}'
    )
    assert _names(_parse_text_tool_calls(content)) == ["read_file"]


# ── Stage 3: free-text fallback (only when no fenced call found) ──────────────

def test_free_text_only_call_recovered_via_fallback():
    # No fence at all → stage-3 scans free text and recovers the genuine call.
    content = 'please run {"name":"read_file","arguments":{"path":"a.py"}} now'
    assert _names(_parse_text_tool_calls(content)) == ["read_file"]


def test_free_text_lookalike_alone_still_parsed_by_fallback():
    """Documented behavior: with no fenced call, the stage-3 fallback DOES parse
    a free-text JSON object. The fix narrows stage-2 to fenced segments only; it
    does not remove the free-text fallback (some models never fence their
    calls)."""
    content = '예시: {"name": "edit_text", "arguments": {"path": "x.py"}}'
    assert _names(_parse_text_tool_calls(content)) == ["edit_text"]


def test_non_tool_json_object_not_normalized():
    # A JSON object that isn't a recognized tool-call shape must not produce a
    # bogus tool call (no name/tool/tool_name/function key).
    content = '```\n{"key": "value", "nested": {"a": 1}}\n```'
    assert _parse_text_tool_calls(content) == []


def test_fenced_non_json_code_block_ignored():
    content = '```python\nx = {"not": "a tool"}\nprint(x)\n```'
    assert _parse_text_tool_calls(content) == []
