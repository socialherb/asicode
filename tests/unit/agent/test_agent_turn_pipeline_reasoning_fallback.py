"""Agent turn pipeline reasoning_content fallback parity.

GLM-5.2 (thinking ON) / DeepSeek Reasoner may emit the final answer in
``reasoning_content`` with an empty ``content`` field. DesignChatLoop already
falls back to reasoning_content on EVERY termination path; the agent turn
pipeline (code/orchestrator mode) had the SAME five extraction sites with NO
fallback, so the closing summary after tool work was silently swallowed and
the REPL returned to the prompt with no final message.

This pins ``TurnPipelineMixin._effective_final_content`` — the single helper
now used by all five termination/early-finish paths.
"""
from __future__ import annotations

from types import SimpleNamespace

from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin


def _resp_dict(*, content="", reasoning="", has_raw=True):
    """Build the dict shape returned by AgentLoop._llm_call_with_tools / n()."""
    raw = SimpleNamespace(raw_response={
        "choices": [{"message": {"reasoning_content": reasoning}}] if reasoning or content is not None else [],
    }) if has_raw else None
    return {"content": content, "raw": raw}


def test_content_present_returns_content():
    r = _resp_dict(content="done", reasoning="ignored")
    assert TurnPipelineMixin._effective_final_content(r) == "done"


def test_empty_content_falls_back_to_reasoning():
    reasoning = "## Done: commit abc123 applied"
    r = _resp_dict(content="", reasoning=reasoning)
    assert TurnPipelineMixin._effective_final_content(r) == reasoning


def test_whitespace_content_falls_back_to_reasoning():
    reasoning = "  summary body  "
    r = _resp_dict(content="   \n  \t  ", reasoning=reasoning)
    # reasoning is stripped on fallback
    assert TurnPipelineMixin._effective_final_content(r) == reasoning.strip()


def test_empty_content_no_reasoning_returns_empty():
    r = _resp_dict(content="", reasoning="")
    assert TurnPipelineMixin._effective_final_content(r) == ""


def test_object_response_uses_getattr_path():
    """When response is an object (not dict), read .content / .raw via getattr."""
    raw = SimpleNamespace(raw_response={
        "choices": [{"message": {"reasoning_content": "from-object"}}],
    })
    resp = SimpleNamespace(content="", raw=raw)
    assert TurnPipelineMixin._effective_final_content(resp) == "from-object"


def test_missing_raw_response_is_safe():
    r = {"content": "", "raw": SimpleNamespace(raw_response=None)}
    assert TurnPipelineMixin._effective_final_content(r) == ""


def test_empty_choices_list_is_safe():
    r = {"content": "", "raw": SimpleNamespace(raw_response={"choices": []})}
    assert TurnPipelineMixin._effective_final_content(r) == ""


def test_raw_response_not_dict_is_safe():
    r = {"content": "", "raw": SimpleNamespace(raw_response="not-a-dict")}
    assert TurnPipelineMixin._effective_final_content(r) == ""


def test_none_response_is_safe():
    assert TurnPipelineMixin._effective_final_content(None) == ""


def test_reasoning_not_string_is_safe():
    r = {
        "content": "",
        "raw": SimpleNamespace(raw_response={
            "choices": [{"message": {"reasoning_content": None}}],
        }),
    }
    assert TurnPipelineMixin._effective_final_content(r) == ""


def test_all_five_sites_use_helper():
    """Source-contract guard: every termination/early-finish content extraction
    must route through the fallback helper, not read content directly."""
    import re
    from pathlib import Path

    src = Path("external_llm/agent/agent_turn_pipeline.py").read_text()
    # The helper must be referenced exactly 5 times (5 termination paths).
    assert src.count("_effective_final_content(response)") >= 5
    # No remaining direct content extraction at the final-answer sites.
    # `_rget("content"` may still exist for the agent_thinking callback (L273),
    # but must NOT appear inside final_msg / _llm_last_msg assignments.
    bad = re.findall(r"final_msg\s*=\s*_rget\(\"content\"|_llm_last_msg\w*\s*=\s*\(.*?\.get\(\"content\"", src)
    assert not bad, f"termination paths still extract content directly: {bad}"
