"""RED->GREEN pty-driver coverage for repl_impl.py layer 1 (terminal-bound).

Layer-1 functions identified in the 52% gap analysis:

  _run_esc_watcher    (33 miss)  -- termios raw-mode ESC watchdog thread
  _cli_checkpoint_cb  (111 miss) -- stdin checkpoint question callback
  _collect_input      (183 miss) -- prompt_toolkit input session

Approach: in-process pty (tests/unit/pty_driver.py). ``sys.stdin``/``sys.stdout``
are redirected to a pty slave; a drain thread reads the master (a terminal-
emulator surrogate that also answers cursor-position requests). Module state
(``_prompt_session``, auto-continue globals, esc-watcher pause) is snapshotted
and restored per test. Source-free: repl_impl.py is not modified.

Test-pattern rules (hard-won):

* Never pre-send input to a prompt_toolkit prompt: the pty line discipline is
  still canonical at that point and translates ``\\r`` -> ``\\n``, which ptk
  parses as Ctrl+J (newline in multiline mode) instead of Enter. Wait for the
  rendered prompt (raw mode is then active) and send afterwards.
* pytest's capture machinery re-wraps ``sys.stdout`` after fixture setup, so
  each test must call ``tty.activate()`` to re-apply the pty redirection.
* Python 3.12+ threads do not inherit the caller's contextvars, so every
  worker thread resolves prompt_toolkit's shared *default* AppSession, whose
  input/output are cached on first access -- reset them per test.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import select as select_mod
import sys
import termios
import threading
import time
from typing import ClassVar

import pytest

import asi
from external_llm.repl import repl_impl
from tests.unit.pty_driver import PtySession

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX pty/termios required")

pytest.importorskip("prompt_toolkit")


# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------


def _reset_app_session() -> None:
    """Clear prompt_toolkit's shared default AppSession input/output caches.

    Python 3.12+ ``threading.Thread`` does not inherit the caller's
    contextvars (``sys.flags.thread_inherit_context``), so worker threads
    always resolve the ContextVar *default* AppSession -- whose
    ``input``/``output`` are lazily cached on first access. Without this
    reset, the second prompt in a process would reuse the first test's
    (already closed) pty and die with EIO.
    """
    import contextvars

    from prompt_toolkit.application.current import _current_app_session

    sess = contextvars.Context().run(_current_app_session.get)
    sess._input = None  # ptk internal cache reset
    sess._output = None


_REPL_GLOBALS: ClassVar[tuple[str, ...]] = (
    "_prompt_session", "_input_underline", "_prompt_history_path",
    "_ctrlc_armed", "_next_prompt_suggestion", "_next_suggestion_gen",
    "_auto_continue_state", "_auto_submit_gen", "_auto_countdown_active",
    "_last_input_was_auto", "_completer_provider", "_completer_model",
)


@pytest.fixture(autouse=True)
def repl_state():
    """Snapshot/restore repl_impl module globals touched by interactive paths."""
    saved = {}
    for name in _REPL_GLOBALS:
        value = getattr(repl_impl, name)
        if name == "_auto_continue_state":
            value = dict(value)
        saved[name] = value
    repl_impl._prompt_session = None  # fresh session per test (input bound at construction)
    repl_impl._esc_watcher_pause.clear()
    _reset_app_session()
    yield
    for name, value in saved.items():
        if name == "_auto_continue_state":
            value = dict(value)
        setattr(repl_impl, name, value)
    repl_impl._esc_watcher_pause.clear()


@pytest.fixture
def tty(monkeypatch, repl_state):
    harness = PtySession(monkeypatch)
    yield harness
    harness.close()


@pytest.fixture
def tty_stdin_only(monkeypatch, repl_state):
    harness = PtySession(monkeypatch, redirect_stdout=False)
    yield harness
    harness.close()


def _start_prompt(tty, prompt: str, bottom_toolbar: bool = False):
    """Run ``_collect_input`` in a thread; block until the prompt rendered.

    Returns ``(thread, result_box)``. ``result_box`` gets ``{"value": ...}``
    or ``{"exc": ...}``. The rendered prompt implies ptk's raw mode is
    active, so subsequent ``tty.send`` bytes reach ptk untranslated.
    """
    tty.activate()
    _reset_app_session()
    result: dict = {}

    def _call() -> None:
        try:
            result["value"] = repl_impl._collect_input(prompt, bottom_toolbar)
        except BaseException as exc:
            result["exc"] = exc

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    tty.wait_for(prompt.rstrip().encode("utf-8"), timeout=12)
    return t, result


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeSpinner:
    """Duck-typed _ProgressPrinter for _cli_checkpoint_cb's stop/restart calls."""

    def __init__(self, fail_stop: bool = False, fail_start: bool = False) -> None:
        self.fail_stop = fail_stop
        self.fail_start = fail_start
        self.stopped = 0
        self.started = 0

    def _stop_spinner(self) -> None:
        if self.fail_stop:
            raise RuntimeError("stop boom")
        self.stopped += 1

    def _start_spinner(self, _msg: str) -> None:
        if self.fail_start:
            raise RuntimeError("start boom")
        self.started += 1


class _FakeOutConsole:
    """Duck-typed asi._out_console for the Rich markdown branch."""

    class _File:
        def reset_bol(self) -> None:
            pass

    file = _File()

    def __init__(self) -> None:
        self.printed = []

    def print(self, obj) -> None:
        self.printed.append(obj)


# ---------------------------------------------------------------------------
# _run_esc_watcher (33 miss)
# ---------------------------------------------------------------------------


class TestRunEscWatcher:
    def test_non_tty_returns_immediately(self, monkeypatch, repl_state):
        cancel = threading.Event()
        stop = threading.Event()
        with open(os.devnull) as f:
            monkeypatch.setattr(sys, "stdin", f)
            repl_impl._run_esc_watcher(cancel, stop)
        assert not cancel.is_set()

    def test_stop_event_exits_loop(self, tty):
        cancel = threading.Event()
        stop = threading.Event()
        t = threading.Thread(
            target=repl_impl._run_esc_watcher, args=(cancel, stop), daemon=True)
        t.start()
        time.sleep(0.5)  # let it enter the select(0.3) loop at least once
        stop.set()
        t.join(timeout=3)
        assert not t.is_alive()
        assert not cancel.is_set()

    def test_esc_sets_cancel_and_restores_termios(self, tty):
        import select as _sel

        tty.activate()
        before = termios.tcgetattr(tty.slave_fd)
        cancel = threading.Event()
        stop = threading.Event()
        t = threading.Thread(
            target=repl_impl._run_esc_watcher, args=(cancel, stop), daemon=True)
        t.start()
        time.sleep(0.2)  # raw mode is set at thread start
        tty.send(b"\x1b")
        assert cancel.wait(timeout=3)
        stop.set()
        t.join(timeout=3)
        assert not t.is_alive()
        # The kernel leaves PENDIN set when input is pending at the raw->
        # canonical transition; consuming the byte clears it.
        r, _, _ = _sel.select([tty.slave_fd], [], [], 0.2)
        if r:
            os.read(tty.slave_fd, 4096)
        assert termios.tcgetattr(tty.slave_fd) == before

    def test_pause_blocks_esc_until_cleared(self, tty):
        tty.activate()
        cancel = threading.Event()
        stop = threading.Event()
        t = threading.Thread(
            target=repl_impl._run_esc_watcher, args=(cancel, stop), daemon=True)
        t.start()
        time.sleep(0.2)
        repl_impl._esc_watcher_pause.set()
        try:
            # wait out one select(0.3) cycle so the watcher notices the pause
            # and enters its sleep(0.1) loop before the ESC arrives
            time.sleep(0.5)
            tty.send(b"\x1b")
            time.sleep(0.5)
            assert not cancel.is_set()
            repl_impl._esc_watcher_pause.clear()
            assert cancel.wait(timeout=3)
        finally:
            repl_impl._esc_watcher_pause.clear()
            stop.set()
            t.join(timeout=3)

    def test_cancel_preset_exits_immediately(self, tty):
        tty.activate()
        cancel = threading.Event()
        cancel.set()
        stop = threading.Event()
        t = threading.Thread(
            target=repl_impl._run_esc_watcher, args=(cancel, stop), daemon=True)
        t.start()
        t.join(timeout=3)
        assert not t.is_alive()

    def test_termios_get_failure_logs_debug(self, tty, monkeypatch, caplog):
        tty.activate()

        def _boom(*_a, **_k):
            raise OSError("tty gone")

        monkeypatch.setattr(termios, "tcgetattr", _boom)
        cancel = threading.Event()
        stop = threading.Event()
        with caplog.at_level(logging.DEBUG):
            repl_impl._run_esc_watcher(cancel, stop)
        assert not cancel.is_set()
        assert "esc watcher failed" in caplog.text

    def test_termios_restore_failure_logs_debug(self, tty, monkeypatch, caplog):
        tty.activate()
        real = termios.tcsetattr
        calls = {"n": 0}

        def _failing_restore(fd, when, attrs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("restore boom")
            real(fd, when, attrs)

        monkeypatch.setattr(termios, "tcsetattr", _failing_restore)
        cancel = threading.Event()
        stop = threading.Event()
        with caplog.at_level(logging.DEBUG):
            t = threading.Thread(
                target=repl_impl._run_esc_watcher, args=(cancel, stop),
                daemon=True)
            t.start()
            time.sleep(0.2)
            stop.set()
            t.join(timeout=3)
            assert not t.is_alive()
            assert "esc watcher: terminal restore failed" in caplog.text


# ---------------------------------------------------------------------------
# _cli_checkpoint_cb (111 miss)
# ---------------------------------------------------------------------------


class TestCliCheckpointCb:
    def test_numeric_option_answer(self, tty_stdin_only, capsys):
        tty_stdin_only.send(b"2\r")
        out = repl_impl._cli_checkpoint_cb({
            "question": "Which file?",
            "options": ["a.py", "b.py"],
            "default": "1",
            "timeout": 30,
        })
        assert out == {"status": "answered", "answer": "b.py"}
        text = capsys.readouterr().out
        assert "User Checkpoint" in text
        assert "Options:" in text
        assert "[Default: 1] (auto-applied in 30s)" in text
        assert "→ Answer: b.py" in text

    def test_free_text_answer_with_options(self, tty_stdin_only, capsys):
        tty_stdin_only.send(b"just do it\r")
        out = repl_impl._cli_checkpoint_cb({
            "question": "q", "options": ["a", "b"], "default": "1", "timeout": 30})
        assert out == {"status": "answered", "answer": "just do it"}
        assert "(no option" not in capsys.readouterr().out

    def test_out_of_range_option_passthrough(self, tty_stdin_only, capsys):
        tty_stdin_only.send(b"9\r")
        out = repl_impl._cli_checkpoint_cb({
            "question": "q", "options": ["a", "b"], "default": "1", "timeout": 30})
        assert out["status"] == "answered"
        assert out["answer"] == "9"
        assert "(no option #9" in capsys.readouterr().out

    def test_empty_line_keeps_default(self, tty_stdin_only):
        tty_stdin_only.send(b"\r")
        out = repl_impl._cli_checkpoint_cb(
            {"question": "q", "default": "yes", "timeout": 30})
        assert out == {"status": "answered", "answer": "yes"}

    def test_timeout_applies_default_with_milestones(self, tty_stdin_only,
                                                     monkeypatch, capsys):
        clock = {"n": 0}
        t0 = 1000.0

        def _fake_monotonic() -> float:
            clock["n"] += 1
            return t0 + 20 * (clock["n"] - 1)

        def _no_input(rlist, wlist, xlist, timeout=None):
            return ([], [], [])

        monkeypatch.setattr(time, "monotonic", _fake_monotonic)
        monkeypatch.setattr(select_mod, "select", _no_input)
        out = repl_impl._cli_checkpoint_cb(
            {"question": "q", "default": "yes", "timeout": 65})
        assert out == {"status": "timeout", "answer": "yes"}
        text = capsys.readouterr().out
        # each loop iteration consumes two monotonic calls (deadline check +
        # remaining), so with a 20s/call step the milestones fire at 25s and 1s
        assert "auto-applies in 25s" in text
        assert "auto-applies in 1s" in text

    def test_invalid_timeout_falls_back_to_120(self, tty_stdin_only, capsys):
        tty_stdin_only.send(b"1\r")
        out = repl_impl._cli_checkpoint_cb(
            {"question": "q", "default": "d", "timeout": "abc"})
        assert out == {"status": "answered", "answer": "1"}
        assert "(auto-applied in 120s)" in capsys.readouterr().out

    def test_spinner_stopped_and_restarted(self, tty_stdin_only, monkeypatch):
        sp = _FakeSpinner()
        monkeypatch.setattr(repl_impl, "_active_spinner_printer", sp)
        tty_stdin_only.send(b"x\r")
        out = repl_impl._cli_checkpoint_cb({"question": "q"})
        assert out["answer"] == "x"
        assert sp.stopped == 1 and sp.started == 1

    def test_spinner_stop_and_start_failures_logged(self, tty_stdin_only,
                                                    monkeypatch, caplog):
        sp = _FakeSpinner(fail_stop=True, fail_start=True)
        monkeypatch.setattr(repl_impl, "_active_spinner_printer", sp)
        tty_stdin_only.send(b"x\r")
        with caplog.at_level(logging.DEBUG):
            out = repl_impl._cli_checkpoint_cb({"question": "q"})
        assert out["answer"] == "x"
        assert "checkpoint: spinner stop failed" in caplog.text
        assert "checkpoint: spinner restart failed" in caplog.text

    def test_termios_save_failure_continues(self, tty_stdin_only, monkeypatch,
                                            caplog):
        def _boom(*_a, **_k):
            raise OSError("no termios")

        monkeypatch.setattr(termios, "tcgetattr", _boom)
        tty_stdin_only.send(b"free\r")
        with caplog.at_level(logging.DEBUG):
            out = repl_impl._cli_checkpoint_cb(
                {"question": "q", "options": ["a"], "timeout": 10})
        assert out["status"] == "answered"
        assert out["answer"] == "free"
        assert "checkpoint: terminal state save failed" in caplog.text

    def test_termios_restore_failure_logged(self, tty_stdin_only, monkeypatch,
                                            caplog):
        real = termios.tcsetattr
        calls = {"n": 0}

        def _failing(fd, when, attrs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("restore boom")
            real(fd, when, attrs)

        monkeypatch.setattr(termios, "tcsetattr", _failing)
        tty_stdin_only.send(b"x\r")
        with caplog.at_level(logging.DEBUG):
            out = repl_impl._cli_checkpoint_cb({"question": "q"})
        assert out["answer"] == "x"
        assert "checkpoint: terminal state restore failed" in caplog.text

    def test_rich_markdown_branch(self, tty_stdin_only, monkeypatch, capsys):
        fake = _FakeOutConsole()
        monkeypatch.setattr(asi, "_out_console", fake)
        tty_stdin_only.send(b"1\r")
        out = repl_impl._cli_checkpoint_cb({
            "question": "**bold** question", "options": ["a"], "default": "1"})
        assert out["answer"] == "a"
        assert len(fake.printed) == 1
        assert "User Checkpoint" in capsys.readouterr().out

    def test_plain_text_question_when_rich_disabled(self, tty_stdin_only,
                                                    monkeypatch, capsys):
        # A previous test may have created asi._out_console; force the plain
        # wrap_cjk render branch instead of the Rich markdown one.
        monkeypatch.setattr(repl_impl, "_RICH", False)
        tty_stdin_only.send(b"x\r")
        out = repl_impl._cli_checkpoint_cb(
            {"question": "a long question line that wraps\n\nblank gap",
             "timeout": 10})
        assert out == {"status": "answered", "answer": "x"}
        text = capsys.readouterr().out
        assert "a long question line that wraps" in text
        assert "\n" in text
        assert "User Checkpoint" in text


# ---------------------------------------------------------------------------
# _collect_input (183 miss)
# ---------------------------------------------------------------------------


class TestCollectInput:
    def test_basic_input_with_arrow_prompt(self, tty):
        t, result = _start_prompt(tty, "\u276f ")  # real REPL prompt glyph
        tty.send(b"hello\r")
        t.join(timeout=12)
        assert result == {"value": "hello"}

    def test_ctrlc_auxiliary_prompt_raises(self, tty):
        t, result = _start_prompt(tty, "y/N ")
        tty.send(b"\x03")
        t.join(timeout=12)
        assert isinstance(result["exc"], KeyboardInterrupt)

    def test_ctrlc_main_arm_hint_then_second_raises(self, tty):
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"\x03")
        tty.wait_for(b"press Ctrl+C again to exit", timeout=12)
        tty.send(b"\x03")
        t.join(timeout=12)
        assert isinstance(result["exc"], KeyboardInterrupt)

    def test_ctrlc_with_text_clears_buffer(self, tty):
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"abc\x03def\r")
        t.join(timeout=12)
        assert result == {"value": "def"}

    def test_ctrl_d_raises_eof(self, tty):
        t, result = _start_prompt(tty, "x> ")
        tty.send(b"\x04")
        t.join(timeout=12)
        assert isinstance(result["exc"], EOFError)

    def test_meta_enter_inserts_newline(self, tty):
        t, result = _start_prompt(tty, "x> ")
        tty.send(b"a\x1b\rb\r")
        t.join(timeout=12)
        assert result == {"value": "a\nb"}

    def test_ctrl_j_inserts_newline(self, tty):
        t, result = _start_prompt(tty, "x> ")
        tty.send(b"c\x0ad\r")
        t.join(timeout=12)
        assert result == {"value": "c\nd"}

    def test_enter_submits_auto_suggestion(self, tty):
        repl_impl._auto_countdown_active = True
        repl_impl._next_prompt_suggestion = "next: fix tests"
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"\r")
        t.join(timeout=12)
        assert result == {"value": "next: fix tests"}
        assert repl_impl._last_input_was_auto is True
        assert repl_impl._auto_countdown_active is False

    def test_typing_cancels_auto_countdown(self, tty):
        repl_impl._auto_countdown_active = True
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"a\r")
        t.join(timeout=12)
        assert result == {"value": "a"}
        assert repl_impl._auto_countdown_active is False

    def test_esc_cancels_auto_countdown(self, tty):
        repl_impl._auto_countdown_active = True
        repl_impl._next_prompt_suggestion = "ghost"
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"\x1b")
        tty.wait_for(b"auto-continue: cancelled", timeout=12)
        tty.send(b"x\r")
        t.join(timeout=12)
        assert result == {"value": "x"}
        assert repl_impl._auto_countdown_active is False

    def test_seed_suggestion_arms_auto_continue(self, tty, monkeypatch):
        monkeypatch.setattr(repl_impl, "_AUTO_CONTINUE_DELAY", 3600.0)
        repl_impl._auto_continue_state.update({"on": True, "cap": 5, "depth": 0})
        repl_impl._next_prompt_suggestion = "ghost task"
        t, result = _start_prompt(tty, "x> ", True)
        tty.wait_for(b"auto-continue 1/5 in 3600s", timeout=12)
        assert repl_impl._auto_countdown_active is True
        tty.send(b"\r")
        t.join(timeout=12)
        assert result == {"value": "ghost task"}
        assert repl_impl._last_input_was_auto is True

    def test_history_file_persistence(self, tty, tmp_path):
        hist = tmp_path / "asr" / "cli_history"
        repl_impl._prompt_history_path = str(hist)
        t, result = _start_prompt(tty, "x> ")
        tty.send(b"cmd one\r")
        t.join(timeout=12)
        assert result == {"value": "cmd one"}
        assert hist.exists()
        assert "cmd one" in hist.read_text(encoding="utf-8")

    def test_history_fallback_on_bad_path(self, tty, tmp_path, caplog):
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        repl_impl._prompt_history_path = str(blocker / "cli_history")
        with caplog.at_level(logging.DEBUG):
            t, result = _start_prompt(tty, "x> ")
            tty.send(b"ok\r")
            t.join(timeout=12)
            assert result == {"value": "ok"}
        assert "cli history persistence failed" in caplog.text

    def test_no_color_style(self, tty, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        t, result = _start_prompt(tty, "x> ")
        tty.send(b"plain\r")
        t.join(timeout=12)
        assert result == {"value": "plain"}

    def test_running_event_loop_thread_path(self, tty):
        tty.activate()
        _reset_app_session()
        result: dict = {}

        def _run() -> None:
            async def _main():
                return repl_impl._collect_input("x> ")

            loop = asyncio.new_event_loop()
            try:
                result["value"] = loop.run_until_complete(_main())
            except BaseException as exc:
                result["exc"] = exc
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        tty.wait_for(b"x>", timeout=12)
        tty.send(b"threaded\r")
        t.join(timeout=15)
        assert result == {"value": "threaded"}

    def test_running_event_loop_thread_path_eof(self, tty):
        tty.activate()
        _reset_app_session()
        result: dict = {}

        def _run() -> None:
            async def _main():
                repl_impl._collect_input("x> ")

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_main())
            except BaseException as exc:
                result["exc"] = exc
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        tty.wait_for(b"x>", timeout=12)
        tty.send(b"\x04")
        t.join(timeout=15)
        assert isinstance(result["exc"], EOFError)

    def test_stdin_without_fileno_dummy_input(self, tty, monkeypatch):
        tty.activate()
        monkeypatch.setattr(sys, "stdin", io.StringIO("x"))
        with pytest.raises(EOFError):
            repl_impl._collect_input("\u276f ")

    def test_pipe_stdin_readline(self, monkeypatch, repl_state):
        r, w = os.pipe()
        monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r"))
        try:
            os.write(w, b"hello\n")
            assert repl_impl._collect_input("x> ") == "hello"
        finally:
            os.close(w)

    def test_pipe_stdin_eof(self, monkeypatch, repl_state):
        r, w = os.pipe()
        monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r"))
        os.close(w)
        with pytest.raises(EOFError):
            repl_impl._collect_input("x> ")

    def test_input_fallback_when_ptk_unavailable(self, monkeypatch, repl_state):
        # _load_prompt_toolkit() False -> builtin input() path
        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: False)
        monkeypatch.setattr(sys, "stdin", io.StringIO("fallback line\n"))
        assert repl_impl._collect_input("x> ") == "fallback line"

    def test_input_fallback_eof(self, monkeypatch, repl_state):
        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: False)
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        with pytest.raises(EOFError):
            repl_impl._collect_input("x> ")

    def test_input_fallback_keyboard_interrupt(self, monkeypatch, repl_state):
        class _KickingStdin:
            def readline(self, *a, **k):
                raise KeyboardInterrupt

        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: False)
        monkeypatch.setattr(sys, "stdin", _KickingStdin())
        with pytest.raises(KeyboardInterrupt):
            repl_impl._collect_input("x> ")

    def test_auto_suggest_empty_buffer_returns_suggestion(self, tty):
        repl_impl._next_prompt_suggestion = "sugg"
        t, result = _start_prompt(tty, "x> ", True)
        # type then delete -> buffer becomes empty again, auto-suggest
        # recalculates on the text change with the ghost suggestion present
        tty.send(b"a\x7f\r")
        t.join(timeout=12)
        assert result == {"value": ""}

    def test_auto_suggest_empty_buffer_without_suggestion(self, tty):
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"a\x7f\r")
        t.join(timeout=12)
        assert result == {"value": ""}

    def test_sigwinch_debounced_resize(self, tty, monkeypatch):
        """Drive a real SIGWINCH through the ptk loop's debounced on_resize.

        ptk only registers its WINCH handler when the app runs in the main
        thread, so this test runs _collect_input directly in the main thread
        and feeds input from a helper thread.
        """
        import signal as signal_mod

        tty.activate()
        _reset_app_session()
        result: dict = {}

        def _feed() -> None:
            time.sleep(1.0)  # prompt rendered, loop registered WINCH handler
            os.kill(os.getpid(), signal_mod.SIGWINCH)
            time.sleep(0.25)  # second WINCH inside the 0.5s debounce window
            os.kill(os.getpid(), signal_mod.SIGWINCH)  # cancels the first
            time.sleep(1.0)  # debounce window (0.5s) elapses
            tty.send(b"x\r")

        threading.Thread(target=_feed, daemon=True).start()
        try:
            result["value"] = repl_impl._collect_input("x> ")
        except BaseException as exc:
            result["exc"] = exc
        assert result == {"value": "x"}

    def test_truncation_over_50000_chars(self, tty):
        """50k+ input truncation, delivered as a real terminal paste.

        Huge inputs arrive as ONE bracketed paste (``ESC[200~ .. ESC[201~``),
        which ptk inserts as a single buffer operation. The pre-P11-2 form fed
        50k bytes as raw per-key characters — a physically-untypable path that
        forced ptk's full-line re-render per key (quadratic: 50k raw = 2.27x the
        time of 25k) and made this the longest test in the suite (8.4s solo,
        13.8s in-suite, two CI flake incidents) while asserting nothing extra:
        the truncation check runs on the prompt()'s returned text either way.
        Bracketed paste: 0.56s, same value + same truncation notice.
        """
        t, result = _start_prompt(tty, "x> ")
        feed_done = threading.Event()

        def _feed() -> None:
            try:
                # Single paste of 50001 chars (1 over the limit), then Enter.
                # Chunked sends are fine: the vt100 parser accumulates paste
                # content until the ESC[201~ terminator, so kernel-queue
                # backpressure cannot split the paste semantics.
                tty.send(b"\x1b[200~")
                for i in range(0, 50001, 8192):
                    tty.send(b"a" * min(8192, 50001 - i))
                tty.send(b"\x1b[201~")
                tty.send(b"\r")
            finally:
                feed_done.set()

        threading.Thread(target=_feed, daemon=True).start()
        t.join(timeout=90)
        feed_done.wait(timeout=5)  # all bytes flushed → no fd-reuse leak into next test
        assert result["value"] == "a" * 50000
        assert b"input truncated to 50000 chars" in tty.dump()

    def test_underline_rule_after_submit(self, tty):
        asi._ensure_out_console_imported()
        t, result = _start_prompt(tty, "x> ", True)
        tty.send(b"task\r")
        t.join(timeout=12)
        assert result == {"value": "task"}
        assert b"\xe2\x94\x80" in tty.dump()  # "---" rule emitted after submit


class TestPtyDriverDumpDrainsPendingBytes:
    """dump() must pump the master fd itself, not just snapshot the buffer.

    v0.2.24 CI flake class: test_truncation_over_50000_chars failed only on
    the 2-core runner with the value assertion passing but the just-printed
    truncation notice missing from the dump — the background drain thread
    only reads *eventually*, and under xdist load it lagged behind the app's
    final flushed writes, so the bare buffer snapshot predates them.
    Deterministic repro: stop the drain thread first, then write, then dump.
    """

    def test_dump_pumps_bytes_the_starved_drain_thread_has_not_read(self, tty):
        tty.activate()  # re-bind sys.stdout to the pty slave (pytest rewraps it)
        tty._stop.set()
        tty._drain.join(timeout=1.0)  # simulate a starved/stalled drain thread
        sys.stdout.write("drain-proof-trailer\n")
        sys.stdout.flush()  # bytes now sit unread in the kernel master queue
        assert b"drain-proof-trailer" in tty.dump()
