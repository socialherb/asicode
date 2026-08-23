"""The write-intent false-success gate must not fire on an unresolved intent.

``_handle_final_answer_turn`` blocks a run that finishes with no applied patch,
on the premise that the user asked for an edit. That premise comes from
``read_only_request``, which every IntentResolver failure path also sets to
False — deliberately, so a legitimate edit is never blocked (see
request_intent_classifier's module docstring).

The consequence was that when intent resolution failed (LLM unreachable,
unparseable JSON — a shape z.ai/GLM reasoning_content already produces), a pure
QUESTION was treated as a write request: the run burned nudge round-trips that
injected `bash('cat > path << EOF')` instructions into a read-only conversation,
and a correct answer could be reported as status="error".

``intent_undetermined`` separates permission (still fails open toward editing)
from expectation (the gate). These tests pin both directions.
"""

from __future__ import annotations

from unittest import mock

from external_llm.agent.agent_loop_types import TurnContext
from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin


def _make_ctx(*, read_only: bool, undetermined: bool, any_tool_called: bool = True) -> TurnContext:
    ctx = TurnContext.__new__(TurnContext)
    ctx.read_only_request = read_only
    ctx.intent_undetermined = undetermined
    ctx.any_tool_called = any_tool_called
    ctx.noop_confirmed = False
    ctx.no_tool_nudge_count = 0
    ctx.turn_num = 2
    ctx.turns = []
    ctx.messages = []
    ctx.request = "greet 함수가 뭐 하는 건지 설명해줘"
    ctx.context = None
    ctx.session_id = "test-session"
    ctx.route = None
    ctx.plan = None
    ctx.tier = 1
    ctx.git_state = None
    ctx.has_native_tools = False
    ctx.provider_name = "anthropic"
    ctx.model_name = "claude-test"
    ctx.base_url = ""
    ctx.tdd_fail_count = 0
    ctx.tdd_total_runs = 0
    ctx.tdd_total_pass = 0
    ctx.total_prompt_tokens = 100
    ctx.total_completion_tokens = 20
    ctx.total_cache_read_tokens = 0
    ctx.total_cache_creation_tokens = 0
    return ctx


def _make_loop(applied_patches=None):
    loop = TurnPipelineMixin.__new__(TurnPipelineMixin)
    loop.config = mock.MagicMock()
    loop.config.max_turns = 5
    loop.config.self_review_enabled = False
    loop.config.agent_id = "test"
    loop.config.stream_callback = None
    loop.registry = mock.MagicMock()
    loop.registry.applied_patches = applied_patches if applied_patches is not None else []
    loop.performance_collector = mock.MagicMock()
    loop.performance_collector.get_summary.return_value = {}
    loop._save_session_log = mock.MagicMock()
    loop._cb = mock.MagicMock()
    loop._is_trivial_edit_request = lambda req: True
    return loop


ANSWER = "`greet()` prints a greeting for the given name."


def test_undetermined_intent_answer_is_not_a_false_success():
    """Intent never classified + tools used + no patch → plain answer, no gate."""
    loop = _make_loop()
    ctx = _make_ctx(read_only=False, undetermined=True)

    outcome = loop._handle_final_answer_turn(ctx, ANSWER)

    assert outcome.nudge_message is None, "a question must not be nudged to write files"
    assert outcome.result is not None
    assert outcome.result.status == "success"
    assert outcome.result.final_message == ANSWER
    assert getattr(outcome.result, "error", None) != "write_intent_finished_without_patch"


def test_undetermined_intent_does_not_disable_the_gate_once_a_patch_lands():
    """The suppression is scoped to the no-patch case; it must not leak."""
    loop = _make_loop(applied_patches=[{"file": "a.py"}])
    ctx = _make_ctx(read_only=False, undetermined=True)

    outcome = loop._handle_final_answer_turn(ctx, "patched it")

    assert outcome.result is not None
    assert outcome.result.status == "success"
    assert outcome.result.applied_patches == [{"file": "a.py"}]


def test_classified_edit_intent_with_no_patch_is_still_gated():
    """The regression guard: a REAL edit request that applied nothing still
    gets the write nudge. Suppressing that would defeat the gate entirely."""
    loop = _make_loop()
    ctx = _make_ctx(read_only=False, undetermined=False)

    outcome = loop._handle_final_answer_turn(ctx, "I would change foo() like this…")

    assert outcome.nudge_message is not None, "classified edit intent must still be nudged"
    assert "apply_patch" in outcome.nudge_message.content
    assert outcome.result is None


def test_classified_edit_intent_errors_after_nudges_exhausted():
    loop = _make_loop()
    ctx = _make_ctx(read_only=False, undetermined=False)
    ctx.no_tool_nudge_count = 99  # past _NO_TOOL_NUDGE_MAX

    outcome = loop._handle_final_answer_turn(ctx, "I would change foo() like this…")

    assert outcome.result is not None
    assert outcome.result.status == "error"
    assert outcome.result.error == "write_intent_finished_without_patch"


def test_read_only_request_never_reaches_the_gate():
    loop = _make_loop()
    ctx = _make_ctx(read_only=True, undetermined=False)

    outcome = loop._handle_final_answer_turn(ctx, ANSWER)

    assert outcome.nudge_message is None
    assert outcome.result is not None
    assert outcome.result.status == "success"


def test_nudge_text_matches_its_own_guard():
    """The nudge is only reachable under `not read_only_request`, so it must not
    claim to handle the read-only case (a dead `if ctx.read_only_request:` branch
    used to sit inside this block), and it must describe what actually happened:
    tools may well have been called — no PATCH was applied."""
    loop = _make_loop()
    ctx = _make_ctx(read_only=False, undetermined=False, any_tool_called=True)

    outcome = loop._handle_final_answer_turn(ctx, "described the change only")

    text = outcome.nudge_message.content
    assert "READ-ONLY" not in text
    assert "did NOT call any tool" not in text
    assert "applied NO patch" in text
