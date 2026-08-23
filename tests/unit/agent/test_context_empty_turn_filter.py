"""Empty-content turn guard in build_context_messages (single funnel).

A turn can be persisted with empty content (a design-chat turn that ended with
an empty LLM response before the funnel guard, or an ESC-interrupted turn).
Blank turns must not reach the provider — Anthropic rejects empty text blocks.

The guard **replaces** conversation turns rather than dropping them. Dropping an
empty assistant turn leaves the user turns on either side adjacent, and
Anthropic rejects non-alternating roles just as hard as it rejects the empty
block (``anthropic_client`` builds its payload verbatim — it never merges
consecutive same-role messages), so dropping would only trade one 400 for
another. Both properties are pinned below: no blank content AND no consecutive
same-role conversation turns.
"""

from external_llm.agent.context_manager import (
    _EMPTY_TURN_PLACEHOLDER,
    SessionCompressionContext,
)

_CONVO_ROLES = ("user", "assistant")


class _FakeSession:
    def __init__(self, turns):
        self.turns = turns
        self.compressed_up_to = 0
        self.compressed_summary = ""
        self.session_id = "test"


def _build(session):
    ctx = SessionCompressionContext("/tmp/nonexistent_repo_for_test")
    return ctx.build_context_messages(session, skip_core_prompt=True, mode="code")


def _consecutive_same_role(msgs):
    roles = [m["role"] for m in msgs]
    return [(roles[i], i) for i in range(1, len(roles)) if roles[i] == roles[i - 1] and roles[i] in _CONVO_ROLES]


def test_no_blank_content_reaches_the_provider():
    session = _FakeSession(
        [
            {"role": "user", "content": "hello", "model": ""},
            {"role": "assistant", "content": "", "model": ""},  # empty
            {"role": "assistant", "content": "   ", "model": ""},  # whitespace-only
            {"role": "assistant", "content": "real answer", "model": ""},
            {"role": "user", "content": "next", "model": ""},
        ]
    )
    msgs = _build(session)
    assert all((m.get("content") or "").strip() for m in msgs)


def test_empty_assistant_turn_keeps_its_alternation_slot():
    """The regression this guard exists to avoid: an empty assistant turn
    between two user turns must NOT collapse into two adjacent user turns."""
    session = _FakeSession(
        [
            {"role": "user", "content": "first question", "model": ""},
            {"role": "assistant", "content": "", "model": ""},
            {"role": "user", "content": "second question", "model": ""},
        ]
    )
    msgs = _build(session)

    assert _consecutive_same_role(msgs) == [], (
        "dropping the empty turn left consecutive same-role messages — "
        "Anthropic rejects that exactly as it rejects an empty content block"
    )
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["content"] == _EMPTY_TURN_PLACEHOLDER


def test_real_content_is_untouched():
    session = _FakeSession(
        [
            {"role": "user", "content": "hello", "model": ""},
            {"role": "assistant", "content": "reply", "model": ""},
        ]
    )
    msgs = _build(session)
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == ["reply"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert "(turn 1) hello" in user_msgs[0]["content"]
    assert _EMPTY_TURN_PLACEHOLDER not in user_msgs[0]["content"]


def test_all_empty_session_yields_no_blank_blocks():
    session = _FakeSession(
        [
            {"role": "user", "content": "", "model": ""},
            {"role": "assistant", "content": "", "model": ""},
        ]
    )
    msgs = _build(session)
    assert all((m.get("content") or "").strip() for m in msgs)
    assert _consecutive_same_role(msgs) == []


def test_placeholder_is_not_injected_into_scaffolding():
    """Only conversation turns get the placeholder; system scaffolding that is
    somehow blank is dropped, since it holds no alternation slot."""
    session = _FakeSession(
        [
            {"role": "user", "content": "hello", "model": ""},
            {"role": "assistant", "content": "reply", "model": ""},
        ]
    )
    msgs = _build(session)
    for m in msgs:
        if m["role"] not in _CONVO_ROLES:
            assert _EMPTY_TURN_PLACEHOLDER not in (m.get("content") or "")
