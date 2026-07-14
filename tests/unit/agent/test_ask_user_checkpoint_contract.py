"""Contract tests for the user_checkpoint callback ↔ _tool_ask_user boundary.

These lock in the return-shape contract that the agent-lane checkpoint callback
(``_build_user_checkpoint_callback`` in webapp/routes/agent_stream.py) must
satisfy for ``_tool_ask_user`` (external_llm/agent/tool_handlers/agent_tools.py)
to interpret it correctly:

* answered -> ``{"status": "answered", "answer": <user answer>}``
* timeout  -> ``{"status": "timeout"}`` (NO answer key -> tool ``default`` applies)
* cancelled-> ``{"status": "cancelled"}`` (NO answer key -> tool ``default`` applies)

Regression guard: returning an empty-string ``answer`` key on timeout/cancel
would shadow the tool's own ``default`` (config note: "초과 시 default로 자율 진행").
"""
import pytest


def _ask(registry, args):
    return registry._tool_ask_user(dict(args))


@pytest.mark.parametrize(
    "callback_return, expected_status, expected_answer",
    [
        # User answered -> answer carried through verbatim.
        ({"status": "answered", "answer": "use-postgres"}, "answered", "use-postgres"),
        # Timeout -> NO answer key: tool must fall back to its own `default`,
        # NOT to an empty string.
        ({"status": "timeout"}, "timeout", "proceed-anyway"),
        # Cancelled -> NO answer key: tool default applies.
        ({"status": "cancelled"}, "cancelled", "proceed-anyway"),
    ],
)
def test_ask_user_callback_return_shapes(
    tool_registry, agent_config, callback_return, expected_status, expected_answer
):
    """_tool_ask_user must honor status and fall back to default when the
    callback omits the answer key (timeout/cancel)."""
    agent_config.user_checkpoint_enabled = True
    agent_config.user_checkpoint_callback = lambda _qd: dict(callback_return)
    agent_config._user_checkpoint_count = 0

    result = _ask(tool_registry, {"question": "Which DB?", "default": "proceed-anyway"})

    assert result.ok
    assert result.metadata["status"] == expected_status, result.metadata
    assert result.metadata["answer"] == expected_answer, result.metadata


def test_ask_user_empty_answer_key_shadows_default_regression(
    tool_registry, agent_config
):
    """Explicit regression: an empty-string answer key (the old buggy shape)
    WOULD shadow the tool default. This documents WHY the callback must omit the
    key on timeout/cancel rather than return ``{"answer": ""}``."""
    agent_config.user_checkpoint_enabled = True
    # Old buggy shape: empty-string answer key present.
    agent_config.user_checkpoint_callback = lambda _qd: {"status": "timeout", "answer": ""}
    agent_config._user_checkpoint_count = 0

    result = _ask(tool_registry, {"question": "Which DB?", "default": "proceed-anyway"})

    # This is the WRONG behavior the old shape produced — empty string wins over
    # the tool default. The callback must therefore omit the key (covered above).
    assert result.metadata["answer"] == "", "documents the shadowing the fix avoids"


def test_ask_user_callback_note_surfaces_in_content(tool_registry, agent_config):
    """_tool_ask_user reads response["note"] and appends it to the tool content,
    so free-text user context reaches the LLM.

    The checkpoint callbacks (_build_user_checkpoint_callback /
    _design_checkpoint_cb) forward this key when the client attaches a note;
    this test locks in the consumer side — that the tool actually surfaces it.
    """
    agent_config.user_checkpoint_enabled = True
    agent_config.user_checkpoint_callback = lambda _qd: {
        "status": "answered", "answer": "yes", "note": "only if tests pass",
    }
    agent_config._user_checkpoint_count = 0

    result = _ask(tool_registry, {"question": "Proceed?", "default": "no"})

    assert result.ok
    assert result.metadata["status"] == "answered"
    assert result.metadata["answer"] == "yes"
    assert "only if tests pass" in result.content, (
        "note must surface in tool content so the LLM sees user context"
    )
