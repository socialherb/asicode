"""Shared utilities for cleaning LLM output — strips markdown fences, normalises JSON,
and handles common formatting quirks from LLM responses.

Consolidates ~10 near-duplicate implementations scattered across the codebase
that each did ``if raw.startswith('```'): raw.split('```', 2)[1]; ... .rstrip('`')``
with minor local variations.
"""

from __future__ import annotations


def strip_markdown_fences(text: str, strip_json_prefix: bool = True) -> str:
    """Remove markdown code fences (```) from LLM output.

    Handles:
    - `` ```json ... ``` `` (inline)
    - `` ``` ... ``` `` with language tag
    - Line-based fences where `` ``` `` sits on its own line
    - Trailing ````` markers (e.g. dangling backticks)

    Args:
        text: Raw LLM output potentially wrapped in fences.
        strip_json_prefix: If True, removes leading ``json`` or ``JSON``
            after the opening fence.

    Returns:
        Cleaned text with fences removed.
    """
    text_stripped = text.strip()
    if not text_stripped.startswith("```"):
        return text
    text = text_stripped

    # ── Phase 1: split on the opening fence ──
    # `` ```json\n{...}\n``` `` → ``{...}\n``` ``
    parts = text.split("```", 2)
    if len(parts) >= 2:
        text = parts[1]
        # Language tag right after ```
        if strip_json_prefix and text.startswith(("json\n", "JSON\n", "json", "JSON")):
            # Strip "json" language tag but keep the newline/content
            idx = 4
            if len(text) > idx and text[idx] == "\n":
                idx += 1
            text = text[idx:]
        text = text.strip()

    # ── Phase 2: remove trailing backticks from partial stripping ──
    # Note: text.endswith("```") and line-based backtick removal are
    # structurally unreachable here — parts[1] from split("```", 2)
    # can never contain "```" since "```" is the delimiter.
    return text.rstrip("`").strip()
