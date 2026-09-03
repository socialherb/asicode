"""Ctrl+C graceful-exit contract for the design-chat worker thread.

The REPL runs the design-chat tool loop on a daemon worker thread. Ctrl+C used
to return "break" without signaling or joining it: a worker caught mid-flight
(LLM / tool / RAG-embedding call creating multiprocessing semaphores via
joblib/loky) was killed at interpreter shutdown with the semaphores still
registered with resource_tracker → "resource_tracker: leaked semaphore
objects" warning after exit.

``_graceful_join_design_chat`` is the module-level helper the KeyboardInterrupt
handler calls: it sets the worker's cancel_event (the worker unwinds at its
next checkpoint via AgentCancelled) and joins with a bounded timeout so Ctrl+C
stays snappy. A second Ctrl+C during the wait skips it.
"""

from __future__ import annotations

import pathlib
import threading
import time

from external_llm.repl import repl_impl
from external_llm.repl.repl_impl import _graceful_join_design_chat


def _spawn_worker(respond_to_cancel: bool, hold: float = 30.0):
    """Daemon worker thread that waits on a cancel event (or ignores it)."""
    cancel = threading.Event()

    def _run():
        if respond_to_cancel:
            cancel.wait(timeout=hold)
        else:
            time.sleep(hold)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, cancel


def test_none_thread_is_noop():
    assert _graceful_join_design_chat(None, None) is True


def test_already_dead_thread_returns_true():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    assert _graceful_join_design_chat(t, None) is True


def test_cancel_signal_makes_worker_exit_quickly():
    t, cancel = _spawn_worker(respond_to_cancel=True)
    t0 = time.monotonic()
    ok = _graceful_join_design_chat(t, cancel, timeout=3.0)
    assert ok is True
    assert not t.is_alive()
    assert cancel.is_set()  # cancel signal was delivered
    assert time.monotonic() - t0 < 3.0


def test_uncooperative_worker_times_out_without_raising():
    t, cancel = _spawn_worker(respond_to_cancel=False, hold=30.0)
    t0 = time.monotonic()
    ok = _graceful_join_design_chat(t, cancel, timeout=0.2)
    assert ok is False
    assert t.is_alive()
    assert cancel.is_set()  # cancel signal is still delivered on timeout
    assert time.monotonic() - t0 >= 0.2


def test_second_ctrl_c_skips_wait(monkeypatch):
    t, cancel = _spawn_worker(respond_to_cancel=False, hold=30.0)

    def _boom(timeout=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(t, "join", _boom)
    t0 = time.monotonic()
    ok = _graceful_join_design_chat(t, cancel, timeout=3.0)
    assert ok is False
    assert cancel.is_set()
    assert time.monotonic() - t0 < 3.0


def test_keyboard_interrupt_handler_calls_graceful_join():
    """Source-level wiring contract: the design-chat KeyboardInterrupt handler
    (nested closure — not directly invocable without a live LLM service) must
    delegate to ``_handle_design_chat_keyboard_interrupt`` (which signals +
    joins the worker before exiting). Precedent: test_helper_command.py
    source-contract guards."""
    src = pathlib.Path(repl_impl.__file__).read_text(encoding="utf-8")
    # 1) The extracted handler itself joins the worker via the graceful helper.
    handler_src = src[
        src.index("def _handle_design_chat_keyboard_interrupt") : src.index("def _advance_auto_continue_state")
    ]
    assert "_graceful_join_design_chat(thread, cancel_event)" in handler_src
    # 2) The REPL KeyboardInterrupt handler delegates to the extracted helper.
    call = "return _handle_design_chat_keyboard_interrupt("
    assert call in src
    idx = src.index(call)
    window = src[max(0, idx - 800) : idx]
    assert "except KeyboardInterrupt:" in window


# ── Extracted REPL-loop helpers (source-level refactor: P5) ────────────────


def test_handle_design_chat_kb_joins_and_breaks(monkeypatch):
    """Ctrl+C handler joins the worker via the graceful helper, prints the
    session summary, and returns "break" — no pty required."""
    joined: list = []
    monkeypatch.setattr(
        repl_impl,
        "_graceful_join_design_chat",
        lambda thread, cancel: joined.append((thread, cancel)) or True,
    )
    printed: list = []
    summarized: list = []
    t = threading.Thread(target=lambda: None)  # dummy, never started
    cancel = threading.Event()
    code = repl_impl._handle_design_chat_keyboard_interrupt(
        t,
        cancel,
        lambda text, style: printed.append(text),
        lambda: summarized.append(1),
    )
    assert code == "break"
    assert joined == [(t, cancel)]  # thread + cancel passed through
    assert printed == ["", "session ended."]
    assert summarized == [1]


def test_handle_design_chat_kb_none_thread_is_safe():
    """Handler tolerates a missing worker (e.g. Ctrl+C before thread start)."""
    code = repl_impl._handle_design_chat_keyboard_interrupt(None, None, lambda text, style: None, lambda: None)
    assert code == "break"


def test_advance_auto_continue_auto_input_increments():
    state = {"on": True, "cap": 5, "depth": 2}
    printed: list = []
    repl_impl._advance_auto_continue_state(state, True, lambda text, style: printed.append((text, style)), "muted")
    assert state["depth"] == 3
    assert printed == [("  🔁 auto-continue step 3/5", "muted")]


def test_advance_auto_continue_auto_input_no_style():
    """muted_style=None (test caller) skips the progress print."""
    state = {"on": True, "cap": 5, "depth": 2}
    printed: list = []
    repl_impl._advance_auto_continue_state(state, True, lambda text, style: printed.append(text))
    assert state["depth"] == 3
    assert printed == []


def test_advance_auto_continue_manual_input_resets():
    state = {"on": True, "cap": 5, "depth": 4}
    printed: list = []
    repl_impl._advance_auto_continue_state(state, False, lambda text, style: printed.append(text), "muted")
    assert state["depth"] == 0
    assert printed == []


def test_advance_auto_continue_cap_is_honored():
    """Depth keeps increasing past cap (cap is enforced at fire time, not
    bookkeeping time) — regression guard for the extracted helper."""
    state = {"on": True, "cap": 5, "depth": 5}
    repl_impl._advance_auto_continue_state(state, True, lambda text, style: None, None)
    assert state["depth"] == 6
