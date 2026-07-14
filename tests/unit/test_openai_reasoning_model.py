"""Tests for ``external_llm.openai_client._is_reasoning_model``.

The classification decides whether the client sends ``max_tokens`` (shared by
reasoning + content on some providers) or ``max_completion_tokens`` (reasoning
gets its own budget). Misclassifying DeepSeek v4 as non-reasoning caused
``/insights compact`` to return empty content with ``finish_reason=length``
on the OpenCode Go endpoint — reasoning tokens ate the whole ``max_tokens``
budget and content got nothing.
"""
from __future__ import annotations

import pytest

from external_llm.openai_client import _is_reasoning_model


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # OpenAI o-series + gpt-5
        ("o1", True),
        ("o1-mini", True),
        ("o3-mini", True),
        ("o3", True),
        ("o4-mini", True),
        ("o-1", True),
        ("gpt-5", True),
        ("gpt-5-turbo", True),
        # DeepSeek reasoner
        ("deepseek-reasoner", True),
        # DeepSeek v4 — the regression this test pins. v4-flash/pro emit
        # ``reasoning_tokens`` and share ``max_tokens`` between reasoning + content
        # on OpenCode Go / OpenRouter, so they MUST be classified as reasoning.
        ("deepseek-v4-flash", True),
        ("deepseek-v4-pro", True),
        ("deepseek-v4", True),
        # Provider/route-prefixed ids must still match after stripping the prefix.
        ("deepseek/deepseek-v4-flash", True),
        ("openrouter/deepseek/deepseek-v4-flash", True),
        ("openrouter/deepseek/deepseek-v4-pro", True),
        # Non-reasoning
        ("gpt-4o", False),
        ("gpt-4", False),
        ("gpt-3.5-turbo", False),
        ("deepseek-chat", False),
        ("deepseek/deepseek-chat", False),
        ("claude-3-5-sonnet", False),
        ("", False),
    ],
)
def test_is_reasoning_model(model: str, expected: bool) -> None:
    assert _is_reasoning_model(model) == expected


def test_is_reasoning_model_none_safe() -> None:
    """A None model must not crash the classifier (callers pass ``svc.model or ""``)."""
    # The function does ``model.strip()`` — guard the caller contract by ensuring
    # an empty string (the realistic None-substitute) classifies as non-reasoning.
    assert _is_reasoning_model("") is False
