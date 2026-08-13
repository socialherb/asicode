"""B1/F2: truncation finish_reason handling must be single-sourced.

B1 — ``_compact_insights_interactive`` (repl_impl.py) treated ``finish_reason=
"truncated"`` as SUCCESS: the retry loop broke out (``!= "length"``) and the
final sanity gate (``== "length"``) did not refuse, so a silently-truncated
rewrite was written straight into the insights single-source file. "truncated"
is the provider-level silent-truncation signal (agent_loop contract) and must
drive the exact same recovery/refusal chain as "length".

F2 — the ``("length", "truncated")`` literal was inlined at 5 sites across 3
files (agent_loop x2, design_chat_loop x1, repl_impl x2). All consumers must
reference the single ``_TRUNCATION_REASONS`` constant instead.

Source-contract tests (file parsing, not import) — importing repl_impl/asi has
heavy import-time side effects, mirroring test_insights_compact_reasoning_budget.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from external_llm.agent._response_utils import _TRUNCATION_REASONS

_REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def _compact_source() -> str:
    """Extract ``_compact_insights_interactive`` source from repl_impl.py."""
    lines = _read("external_llm/repl/repl_impl.py").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^    def _compact_insights_interactive\(\) -> bool:", ln):
            start = i
            break
    if start is None:
        pytest.fail("_compact_insights_interactive not found in repl_impl.py — update this test or restore the symbol; silent skip would mask the regression")
    body = [lines[start]]
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if (
            ln.strip()
            and not ln.startswith("        ")
            and not ln.startswith("\t")
            and (re.match(r"^    def ", ln) or re.match(r"^def ", ln) or re.match(r"^    [a-zA-Z_]", ln))
        ):
            break
        body.append(ln)
    return "\n".join(body)


class TestTruncationReasonsConstant:
    """F2: the single source holds both truncation signals."""

    def test_constant_holds_the_two_truncation_reasons(self):
        assert _TRUNCATION_REASONS == ("length", "truncated")

    def test_constant_defined_in_response_utils(self):
        src = _read("external_llm/agent/_response_utils.py")
        assert '_TRUNCATION_REASONS: tuple[str, ...] = ("length", "truncated")' in src

    def test_agent_loop_references_constant_not_literal(self):
        src = _read("external_llm/agent/agent_loop.py")
        assert "_TRUNCATION_REASONS" in src
        assert 'in ("length", "truncated")' not in src

    def test_design_chat_loop_references_constant_not_literal(self):
        src = _read("external_llm/agent/design_chat_loop.py")
        assert "_TRUNCATION_REASONS" in src
        assert 'in ("length", "truncated")' not in src

    def test_intent_resolver_references_constant_not_literal(self):
        # F2 covers intent_resolver too: the truncation detection gate
        # (resolve() retry ladder) must use the shared constant, and no bare
        # `== "length"` comparison may survive anywhere in the module.
        src = _read("external_llm/agent/intent_resolver.py")
        assert "_TRUNCATION_REASONS" in src
        assert '== "length"' not in src
        assert 'in ("length", "truncated")' not in src


class TestInsightsCompactTruncationGates:
    """B1: "truncated" must retry and refuse exactly like "length"."""

    def test_compact_retry_gate_uses_shared_constant(self):
        src = _compact_source()
        # The retry loop must NOT treat "truncated" as success: break only when
        # finish_reason is outside the shared truncation-reason set.
        assert "_TRUNCATION_REASONS" in src, (
            "compact path must reference the shared truncation-reasons constant"
        )
        assert "if _ci_finish_reason not in _TRUNCATION_REASONS:" in src, (
            "retry gate must break only on non-truncation finish_reason — "
            "finish_reason=truncated must retry with doubled budget like length"
        )
        # No bare literal comparison may remain in the compact function.
        assert '!= "length"' not in src

    def test_compact_final_gate_refuses_truncated(self):
        src = _compact_source()
        # The final sanity gate (before overwriting the insights SSOT file) must
        # refuse BOTH truncation signals — a truncated rewrite must never be
        # written to the single source of truth even when content is non-empty.
        assert "if _ci_finish_reason in _TRUNCATION_REASONS:" in src, (
            "final gate must refuse finish_reason=truncated — a silently "
            "truncated rewrite must not overwrite the insights file"
        )
        assert '== "length"' not in src
