"""P6-1: REPL dispatch / chat-turn return-code contract tests (behavioral).

``_dispatch_command`` (asi.py:7552) and ``_run_chat_turn`` (asi.py:8504) are
nested closures inside ``_run_repl_impl`` — they cannot be imported or called
directly.  This suite obtains *live references* by running ``_run_repl_impl``
under a line tracer with ``_init_repl_engine`` faked: once the main loop has
bound its per-turn free variables (``_was_auto_input``,
``_current_user_images``), the two closures are captured into a registry and
execution aborts with a sentinel before any real input is processed.

Unlike the source-contract guards in ``test_helper_command.py``
(``inspect.getsource`` substring checks), these are **behavioral** tests:
they execute the real dispatch/turn code and assert the actual returned
``(action, user_input)`` / action-code values.

Pinned contracts (from the P1-2 continue/break refactor):

  _dispatch_command(user_input) -> (action, user_input)
      "break"    — end the session (main loop ``break``)
      "continue" — handled; main loop ``continue``
      "chat"     — not a command; proceed to the chat turn
  _run_chat_turn(user_input) -> action
      "break"    — end the session
      "continue" — handled; main loop ``continue``
      "ok"       — turn completed; fall through to the next iteration

The ``"ok"`` path and the orchestrator/ESC paths of ``_run_chat_turn`` need a
live design-chat worker (LLM service), so they are deliberately out of scope
for this unit suite.
"""
from __future__ import annotations

import argparse
import sys
import threading
from types import SimpleNamespace

import pytest

import asi
from external_llm.repl import repl_impl


class _ClosuresCapturedError(Exception):
    """Sentinel raised from the line tracer to abort ``_run_repl_impl``."""


class _CapturedRepl:
    """Live references to the REPL closures captured from ``_run_repl_impl``."""

    def __init__(self, dispatch, turn):
        self.dispatch = dispatch
        self.turn = turn


class _StubSession:
    chat_mode = "code"


class _StubSessionMgr:
    """Minimal session-manager stand-in.

    ``fail_on`` names the method whose first call must raise ``exc``; all
    other methods record and return benign values.
    """

    def __init__(self, fail_on: str | None = None, exc: Exception | None = None):
        self._fail_on = fail_on
        self._exc = exc
        self._failed = False
        self.add_turn_calls: list[tuple] = []

    def _maybe_fail(self, method: str) -> None:
        if not self._failed and method == self._fail_on:
            self._failed = True
            raise self._exc

    def add_turn(self, *args, **kwargs):
        self._maybe_fail("add_turn")
        self.add_turn_calls.append((args, kwargs))

    def get_or_create(self, session_id):
        self._maybe_fail("get_or_create")
        return _StubSession()

    def build_context_messages(self, ds, **kwargs):
        self._maybe_fail("build_context_messages")
        return []


class _ScriptedInput:
    """Stand-in for ``asi._prompt_input``: yields queued values, then EOFError."""

    def __init__(self, values):
        self._values = list(values)

    def __call__(self, *args, **kwargs):
        if not self._values:
            raise EOFError
        return self._values.pop(0)


def _raise_eoferror(*args, **kwargs):
    raise EOFError


def _noop_print(*args, **kwargs):
    pass


def _make_args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(repo=str(tmp_path), verbose=False)


def _make_fake_init(session_mgr=None):
    """Replacement for ``asi._init_repl_engine`` — no LLM service / models."""
    svc = SimpleNamespace(
        model="test-model",
        provider="test",
        llm_service=SimpleNamespace(
            client=None,
            model="test-model",
            provider="test",
            thinking_mode=None,
            reasoning_effort=None,
        ),
    )
    design_config = SimpleNamespace(
        thinking_mode=None,
        reasoning_effort=None,
        model_name="test-model",
        cancel_event=None,
    )

    def fake_init(args, repo_root):
        return {
            "svc": svc,
            "design_config": design_config,
            "design_registry": None,
            "session_mgr": session_mgr if session_mgr is not None else _StubSessionMgr(),
            "session_id": "test-session",
            "pending_notifications": [],
            "notifications_lock": threading.Lock(),
            "provider_str": "test",
            "model_str": "test-model",
        }

    return fake_init


def _capture_closures(monkeypatch, tmp_path, session_mgr=None) -> _CapturedRepl:
    """Run ``_run_repl_impl`` far enough to bind the closures, then abort.

    The loop's first iteration is entered with a scripted empty prompt so the
    per-turn free variables (``_was_auto_input``, ``_current_user_images``)
    are bound; the tracer then captures ``_dispatch_command`` /
    ``_run_chat_turn`` and raises ``_ClosuresCaptured`` before any real input
    is processed.
    """
    captured: dict[str, object] = {}
    prev_tracer = sys.gettrace()

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_name == "_run_repl_impl":
            f_locals = frame.f_locals
            if (
                "_dispatch_command" in f_locals
                and "_run_chat_turn" in f_locals
                and "_was_auto_input" in f_locals
                and "_current_user_images" in f_locals
            ):
                captured["dispatch"] = f_locals["_dispatch_command"]
                captured["turn"] = f_locals["_run_chat_turn"]
                raise _ClosuresCapturedError()
        return tracer

    monkeypatch.setattr(repl_impl, "_init_repl_engine", _make_fake_init(session_mgr))
    monkeypatch.setattr(repl_impl, "_prompt_input", _ScriptedInput([""]))
    monkeypatch.setattr(repl_impl, "_print", _noop_print)
    monkeypatch.setattr("builtins.input", _raise_eoferror)
    sys.settrace(tracer)
    try:
        with pytest.raises(_ClosuresCapturedError):
            asi._run_repl_impl(_make_args(tmp_path))
    finally:
        sys.settrace(prev_tracer)
    assert set(captured) == {"dispatch", "turn"}
    return _CapturedRepl(captured["dispatch"], captured["turn"])


@pytest.mark.unit
class TestDispatchCommandContract:
    """Per-command ``(action, user_input)`` contract of ``_dispatch_command``."""

    @pytest.fixture
    def repl(self, monkeypatch, tmp_path):
        return _capture_closures(monkeypatch, tmp_path)

    @pytest.mark.parametrize("raw", ["exit", "quit", ":q", "/quit", "/exit"])
    def test_session_end_commands_return_break(self, repl, raw):
        assert repl.dispatch(raw) == ("break", raw)

    @pytest.mark.parametrize(
        "raw", ["/help", "/copy", "/think", "/auto off", "/unknownxyz"]
    )
    def test_utility_commands_return_continue(self, repl, raw):
        assert repl.dispatch(raw) == ("continue", raw)

    def test_plain_text_routes_to_chat(self, repl):
        assert repl.dispatch("hello world") == ("chat", "hello world")

    def test_code_mode_switch_without_message_continues(self, repl):
        assert repl.dispatch("/code") == ("continue", "/code")

    def test_code_mode_switch_with_message_routes_to_chat(self, repl):
        assert repl.dispatch("/code hello") == ("chat", "hello")

    def test_general_mode_switch_without_message_continues(self, repl):
        assert repl.dispatch("/general") == ("continue", "")

    def test_orchestrate_with_inline_task_routes_to_chat(self, repl):
        assert repl.dispatch("/orchestrate do X") == ("chat", "do X")


@pytest.mark.unit
class TestRunChatTurnContract:
    """``_run_chat_turn`` error-path action codes (design-chat phase)."""

    def test_keyboard_interrupt_returns_break(self, monkeypatch, tmp_path):
        mgr = _StubSessionMgr(fail_on="add_turn", exc=KeyboardInterrupt())
        repl = _capture_closures(monkeypatch, tmp_path, session_mgr=mgr)
        assert repl.turn("hello") == "break"

    def test_design_chat_error_returns_continue_and_records_error_turn(
        self, monkeypatch, tmp_path
    ):
        mgr = _StubSessionMgr(
            fail_on="build_context_messages", exc=RuntimeError("boom")
        )
        repl = _capture_closures(monkeypatch, tmp_path, session_mgr=mgr)
        assert repl.turn("hello") == "continue"
        # user turn recorded first, then the error turn by the recovery handler
        assert [c[0][1] for c in mgr.add_turn_calls] == ["user", "assistant"]


@pytest.mark.unit
class TestMainLoopWiring:
    """The main loop consumes the action codes: break exits, continue re-prompts."""

    def _run_loop(self, monkeypatch, tmp_path, script):
        monkeypatch.setattr(repl_impl, "_init_repl_engine", _make_fake_init())
        monkeypatch.setattr(repl_impl, "_prompt_input", _ScriptedInput(script))
        monkeypatch.setattr(repl_impl, "_check_clipboard_image", lambda: [])
        monkeypatch.setattr(repl_impl, "_print", _noop_print)
        monkeypatch.setattr("builtins.input", _raise_eoferror)
        return asi._run_repl_impl(_make_args(tmp_path))

    def test_exit_breaks_the_loop(self, monkeypatch, tmp_path):
        assert self._run_loop(monkeypatch, tmp_path, ["exit"]) is None

    def test_empty_input_continues_then_exit_breaks(self, monkeypatch, tmp_path):
        assert self._run_loop(monkeypatch, tmp_path, ["", "exit"]) is None

    def test_eof_breaks_the_loop(self, monkeypatch, tmp_path):
        assert self._run_loop(monkeypatch, tmp_path, []) is None
