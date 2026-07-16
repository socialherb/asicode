"""Shared utilities for extracting data from LLM API response dicts."""

from __future__ import annotations

from typing import Any


def extract_llm_content(response: dict[str, Any], *, default: str = "") -> str:
    """Extract LLM response content from a standard OpenAI-format response dict.

    Handles the standard openai/python format:
        {"choices": [{"message": {"content": "..."}}]}

    Args:
        response: Raw LLM API response dict.
        default: Fallback string when content is empty or missing (default: "").

    Returns:
        The content string, or *default* if missing/empty.
    """
    try:
        msg = response["choices"][0]["message"]
        content = msg.get("content")
        return str(content) if content else default
    except (KeyError, IndexError, TypeError):
        return default
