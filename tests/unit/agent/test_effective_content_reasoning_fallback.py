"""Reasoning-content fallback parity for the shared ``effective_content`` helper.

GLM-5.2 (thinking ON) / DeepSeek Reasoner intermittently emit the final answer
in ``reasoning_content`` while leaving ``content`` empty. Two LLM-consuming
subsystems read ``response.content`` as a final/user-facing message WITHOUT the
fallback that DesignChatLoop (4 paths) and AgentTurnPipeline (5 paths) already
have:

  * ``ContextManager.compress_old_turns`` — the conversation summary. When
    ``content`` is empty the summary is left stale while ``compressed_up_to``
    still advances, so the compressed turns are silently LOST (no summary, no
    verbatim path back).
  * ``IntentResolver._resolve_with_llm`` — the intent JSON. When ``content`` is
    empty the parser finds no JSON and collapses to the heuristic fallback,
    degrading routing/grounding.

Both now route through ``external_llm.client.effective_content`` — the single
canonical extractor for the LLMResponse shape.
"""
from __future__ import annotations

from types import SimpleNamespace

from external_llm.agent.context_manager import SessionCompressionContext
from external_llm.agent.intent_models import IntentResolutionConfig
from external_llm.agent.intent_resolver import IntentResolver
from external_llm.client import LLMResponse, effective_content


# ── pure-function tests for the canonical extractor ─────────────────────────


def _resp(*, content="", reasoning="", raw_response=None):
    return LLMResponse(
        content=content,
        model="glm-5.2",
        provider="zai",
        raw_response=raw_response if raw_response is not None else {
            "choices": [{"message": {"reasoning_content": reasoning}}] if reasoning else [],
        },
    )


def test_effective_content_present_returns_content():
    r = _resp(content="visible answer", reasoning="ignored chain-of-thought")
    assert effective_content(r) == "visible answer"


def test_effective_content_empty_falls_back_to_reasoning():
    r = _resp(content="", reasoning="## Summary: auth bug fix complete")
    assert effective_content(r) == "## Summary: auth bug fix complete"


def test_effective_content_whitespace_falls_back_to_reasoning():
    r = _resp(content="   \n\t  ", reasoning="  recovered body  ")
    # reasoning is stripped on the fallback path
    assert effective_content(r) == "recovered body"


def test_effective_content_empty_no_reasoning_returns_empty():
    r = _resp(content="", reasoning="")
    assert effective_content(r) == ""


def test_effective_content_object_via_getattr():
    """A plain object (not LLMResponse) with .content / .raw_response works too."""
    obj = SimpleNamespace(
        content="",
        raw_response={"choices": [{"message": {"reasoning_content": "from-object"}}]},
    )
    assert effective_content(obj) == "from-object"


def test_effective_content_malformed_raw_response_is_safe():
    """raw_response shapes that don't match choices[0].message must not raise."""
    assert effective_content(_resp(content="", raw_response={})) == ""
    assert effective_content(_resp(content="", raw_response={"choices": []})) == ""
    assert effective_content(_resp(content="", raw_response={"choices": [{}]})) == ""
    # reasoning_content present but blank
    assert effective_content(
        _resp(content="", raw_response={"choices": [{"message": {"reasoning_content": "   "}}]})
    ) == ""


# ── context_manager integration: summary recovered, turns not lost ──────────


class _FakeClient:
    """Minimal llm_client whose .chat() always returns one canned response."""

    def __init__(self, response):
        self._response = response

    def chat(self, **kwargs):
        return self._response


def _make_session(turns):
    return SimpleNamespace(
        turns=turns,
        compressed_up_to=0,
        compressed_summary="",
        session_id="test-session",
        archived_count=0,
    )


def test_compress_recovers_summary_from_reasoning_content():
    """The reported bug: summary lands in reasoning_content (empty content).

    Without the fallback, ``new_summary`` is empty so ``compressed_summary`` is
    left stale — yet ``compressed_up_to`` still advances below, silently losing
    the compressed turns. With the fix the summary is recovered.
    """
    response = _resp(content="", reasoning="## Summary: auth bug fix complete")
    cm = SessionCompressionContext("/tmp/nonexistent-repo")
    turns = [
        {"role": "user", "content": "fix the auth bug"},
        {"role": "assistant", "content": "done, fixed it"},
        {"role": "user", "content": "recent1"},        # recent window (kept)
        {"role": "assistant", "content": "recent2"},
    ]
    session = _make_session(turns)
    cm.compress_old_turns(session, _FakeClient(response), "glm-5.2", recent_keep=2)

    assert session.compressed_summary == "## Summary: auth bug fix complete"
    # and the pointer advanced (turns were folded into the recovered summary)
    assert session.compressed_up_to == 2


def test_compress_still_uses_content_when_present():
    """Behavior-preserving: non-empty content is preferred over reasoning."""
    response = _resp(content="plain summary", reasoning="would-be-fallback")
    cm = SessionCompressionContext("/tmp/nonexistent-repo")
    turns = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    session = _make_session(turns)
    cm.compress_old_turns(session, _FakeClient(response), "glm-5.2", recent_keep=2)
    assert session.compressed_summary == "plain summary"


# ── intent_resolver integration: intent JSON recovered ──────────────────────


_INTENT_JSON = (
    '{"intent_type":"feature","lane_hint":"planner","scope_hint":"single_file",'
    '"modify_symbols":["UserModel"],"new_symbols":'
    '[{"name":"validate","kind":"method","parent":"UserModel"}],'
    '"reference_symbols":[],"search_terms":["UserModel","validate"],'
    '"confidence":0.9,"normalized_query":"Add validate method to UserModel"}'
)


def test_intent_resolved_from_reasoning_content():
    """GLM-5.2 emits the intent JSON in reasoning_content with empty content.

    Without the fallback, ``_parse_llm_response`` finds no JSON and IntentResolver
    collapses to the heuristic fallback (intent_type='unknown', confidence=0.1).
    """
    response = _resp(content="", reasoning=_INTENT_JSON)
    cfg = IntentResolutionConfig(
        llm_client=_FakeClient(response), model="glm-5.2", enable_cache=False,
    )
    resolver = IntentResolver(cfg)
    result = resolver.resolve("Add validate() method to UserModel")

    # Recovered from reasoning_content → real parse, not heuristic fallback.
    assert result.intent_type == "feature"
    assert "UserModel" in result.modify_symbols
    assert result.confidence == 0.9
    assert any(ns.get("name") == "validate" for ns in result.new_symbols)


def test_intent_still_uses_content_when_present():
    """Behavior-preserving: content preferred over reasoning."""
    response = _resp(content=_INTENT_JSON, reasoning="thinking traces")
    cfg = IntentResolutionConfig(
        llm_client=_FakeClient(response), model="glm-5.2", enable_cache=False,
    )
    resolver = IntentResolver(cfg)
    result = resolver.resolve("Add validate() method to UserModel")
    assert result.intent_type == "feature"
    assert "UserModel" in result.modify_symbols
