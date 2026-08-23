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
    signal + join the worker before exiting. Precedent: test_helper_command.py
    source-contract guards."""
    src = pathlib.Path(repl_impl.__file__).read_text(encoding="utf-8")
    trio = (
        '            _print("", "")\n'
        "            _print_session_summary(_session_tokens, _session_t0)\n"
        '            _print("session ended.", "")\n'
        '            return "break"'
    )
    assert trio in src
    idx = src.index(trio)
    window = src[max(0, idx - 800) : idx]
    assert "except KeyboardInterrupt:" in window
    assert "_graceful_join_design_chat(_dc_thread, _dc_cancel)" in window
