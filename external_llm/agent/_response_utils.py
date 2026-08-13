"""Shared utilities for extracting data from LLM API response dicts."""

from __future__ import annotations

import contextlib
from typing import Any

# finish_reason values that signal a TRUNCATED response: the max_tokens budget
# was exhausted ("length"), or the streaming client detected a silently dropped
# final delta ("truncated"). Single source shared by agent_loop, design_chat_loop
# and repl_impl (insights compact) truncation handling — never inline these
# literals at call sites.
_TRUNCATION_REASONS: tuple[str, ...] = ("length", "truncated")


def extract_llm_reasoning(
    response: dict[str, Any], *, default: str = "", strip: bool = False
) -> str:
    """Extract LLM reasoning_content from a standard OpenAI-format response dict.

    DeepSeek Reasoner / GLM-5.2 (thinking ON) models may place the analysis in
    ``choices[0].message.reasoning_content`` while ``content`` stays empty.

    Args:
        response: Raw LLM API response dict.
        default: Fallback string when reasoning is empty or missing (default: "").
        strip: If True, whitespace-strip the result before returning.

    Returns:
        The reasoning string, or *default* if missing/empty.
    """
    try:
        msg = response["choices"][0]["message"]
        reasoning = msg.get("reasoning_content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return default
    text = str(reasoning) if reasoning else default
    return text.strip() if strip else text


def replace_tool_calls(resp: Any, calls: list) -> Any:
    """Replace ``tool_calls`` on a response dict or object, return the response.

    Single source for the dict/object dual-path tool_calls mutation shared by
    agent_loop (truncation recovery) and design_chat_loop (truncation guard).
    A response object without a settable ``tool_calls`` attribute — or a dict
    subclass that rejects the key — is left untouched rather than raising.
    """
    if isinstance(resp, dict):
        resp["tool_calls"] = calls
    else:
        with contextlib.suppress(AttributeError, TypeError):
            resp.tool_calls = calls
    return resp
