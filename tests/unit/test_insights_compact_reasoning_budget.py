"""Regression guard for the ``/insights compact`` reasoning-budget bug.

``_compact_insights_interactive`` (asi.py) compacts the design-insights
file via a single LLM call. On reasoning-capable models (DeepSeek v4 on the
OpenCode Go / OpenRouter endpoints) the ``max_tokens`` budget is SHARED
between reasoning tokens and content tokens — so a budget sized only for the
rewritten file (input + 2k slack) gets eaten entirely by the reasoning trace
and content comes back empty (``finish_reason=length``), surfacing as the
opaque ``✗ compaction failed (LLM call error)`` message.

The fix has two parts, both pinned here:
  1. ``_is_reasoning_model`` (external_llm/openai_client.py) classifies
     ``deepseek-v4-*`` as reasoning so the client sends
     ``max_completion_tokens`` (reasoning gets its own budget).
  2. The compact path sizes ``max_tokens`` to hold BOTH the reasoning trace
     AND the full rewritten file when the model is a reasoner.

Source-contract tests (inspect.getsource) — importing asi has heavy
import-time side effects, so we parse the source text instead.
"""
from __future__ import annotations

import re

import pytest


def _get_compact_insights_source() -> str:
    """Extract ``_compact_insights_interactive`` source from asi.py."""
    src_path = "asi.py"
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^    def _compact_insights_interactive\(\) -> bool:", ln):
            start = i
            break
    if start is None:
        pytest.skip("_compact_insights_interactive not found in asi.py")
    body = [lines[start]]
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("\t"):
            if re.match(r"^    def ", ln) or re.match(r"^def ", ln) or re.match(r"^    [a-zA-Z_]", ln):
                break
        body.append(ln)
    return "".join(body)


class TestInsightsCompactReasoningBudget:
    def test_reasoning_model_check_imported(self):
        """The compact path must consult ``_is_reasoning_model`` to decide whether
        to expand the budget — without this check the budget stays sized for
        non-reasoning models and reasoning tokens eat it whole."""
        src = _get_compact_insights_source()
        assert "_is_reasoning_model" in src, (
            "compact path must import _is_reasoning_model to detect reasoning-capable "
            "models (DeepSeek v4 on OpenCode Go shares max_tokens between reasoning + content)"
        )

    def test_reasoning_budget_expansion_present(self):
        """When the model is a reasoner, the budget MUST be expanded beyond the
        default ``input + 2k`` to hold the reasoning trace too.  On OpenCode
        (which lacks ``max_completion_tokens``), reasoning traces for a 6 KB
        compact task can be 5K-15K tokens; the model generates MORE reasoning
        when given more room, so a generous fixed floor (32k) is more reliable
        than a multiplier that under-sizes for large files."""
        src = _get_compact_insights_source()
        # The expansion must be gated on the reasoning-model check.
        assert "_ci_is_reasoning" in src, (
            "reasoning budget expansion must be gated on _is_reasoning_model"
        )
        # Must set a floor large enough for reasoning traces on OpenCode
        # (which shares the max_tokens budget between reasoning + content).
        assert re.search(r"32000|_ci_in_tokens\s*\*\s*[4-9]", src), (
            "reasoning budget must have a generous floor for shared-budget "
            "providers like OpenCode (reasoning trace + content must both fit)"
        )

    def test_thinking_mode_disabled(self):
        """Compaction is deterministic curation, not reasoning — the call must
        still pass ``thinking_mode=False`` so providers that honor it (native
        DeepSeek) skip reasoning entirely and the whole budget goes to content."""
        src = _get_compact_insights_source()
        assert "thinking_mode=False" in src, (
            "compact call must pass thinking_mode=False (deterministic curation, "
            "not a reasoning task)"
        )
