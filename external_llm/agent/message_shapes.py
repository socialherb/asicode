"""Provider-agnostic message shape detection helpers.

Single source of truth for identifying tool-call and tool-result
messages across all supported provider formats:

- Standard (OpenAI / DeepSeek / Ollama): ``role="tool"`` results and
  ``tool_calls`` on the assistant.
- Anthropic-native: ``role="user"``/``"assistant"`` with ``raw_content``
  ``tool_result`` / ``tool_use`` blocks.
- Gemini-native: ``role="user"`` ``functionResponse`` parts and
  ``role="assistant"`` ``functionCall`` parts.
"""

from __future__ import annotations


def is_tool_result(m) -> bool:
    """Detect a tool-result message in any provider format.

    Standard: ``role="tool"`` with ``content`` as the serialised result.
    Anthropic-native: ``role="user"`` with ``raw_content`` containing at
    least one ``{"type": "tool_result", ...}`` block.
    Gemini-native: ``role="user"`` with ``raw_content`` containing at least
    one ``{"functionResponse": ...}`` part.
    """
    # Standard format (OpenAI, DeepSeek, Ollama, etc.)
    if getattr(m, "role", "") == "tool":
        return True
    # Provider-native formats
    return _is_anthropic_tool_result(m) or _is_gemini_tool_result(m)


def _is_anthropic_tool_result(m) -> bool:
    """True if ``m`` is a ``role="user"`` msg with ``tool_result`` blocks."""
    return _has_raw_blocks(m, "user", "tool_result")


def is_tool_call(m) -> bool:
    """Detect a tool-calling assistant message in any provider format."""
    if getattr(m, "role", "") != "assistant":
        return False
    return (
        bool(getattr(m, "tool_calls", None))
        or _is_anthropic_tool_call(m)
        or _is_gemini_tool_call(m)
    )


def _is_anthropic_tool_call(m) -> bool:
    """True if ``m`` is a ``role="assistant"`` msg with ``tool_use`` blocks."""
    return _has_raw_blocks(m, "assistant", "tool_use")


def _is_gemini_tool_result(m) -> bool:
    """True if ``m`` is a ``role="user"`` msg with ``functionResponse`` parts."""
    return _has_raw_key(m, "user", "functionResponse")


def _is_gemini_tool_call(m) -> bool:
    """True if ``m`` is a ``role="assistant"`` msg with ``functionCall`` parts."""
    return _has_raw_key(m, "assistant", "functionCall")


def _has_raw_blocks(m, role: str, block_type: str) -> bool:
    """True if *m* has ``role`` == *role* and at least one
    ``{"type": block_type, ...}`` dict in ``raw_content``."""
    if getattr(m, "role", "") != role:
        return False
    raw = getattr(m, "raw_content", None)
    return isinstance(raw, list) and any(
        isinstance(b, dict) and b.get("type") == block_type
        for b in raw
    )


def _has_raw_key(m, role: str, key: str) -> bool:
    """True if *m* has ``role`` == *role* and at least one dict in
    ``raw_content`` whose top-level contains *key*.

    Used for Gemini parts (``functionResponse`` / ``functionCall``), which —
    unlike Anthropic blocks — are keyed by name rather than by a ``"type"``
    field, so :func:`_has_raw_blocks` cannot match them.
    """
    if getattr(m, "role", "") != role:
        return False
    raw = getattr(m, "raw_content", None)
    return isinstance(raw, list) and any(
        isinstance(b, dict) and key in b for b in raw
    )
