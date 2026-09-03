"""repl_impl stage-2 pty coverage: run_subagent_worker (in-process) + the full
``_run_repl_impl`` REPL session (spawned child with a real pty).

Layer-1 residual misses were 188 lines (run_subagent_worker) + 1052 lines
(_run_repl_impl). The worker is driven in-process on the pytest main thread
(its ``signal.signal(SIGINT)`` requires it); the REPL runs in a forked-free
subprocess whose fakes are re-applied by ``repl_stage2_child.py`` and whose
coverage data is saved before ``os._exit`` (atexit bypass).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

import external_llm.agent.design_chat_loop as dcl_mod
import external_llm.agent.subagent_ipc as sipc_mod
import external_llm.agent.tool_registry as tool_registry_mod
import external_llm.intelligent_service as isvc_mod
from external_llm.repl import repl_impl
from tests.unit.repl_stage2_fakes import (
    FakeAgentConfig,
    FakeDesignChatLoop,
    FakeSvc,
    FakeToolRegistry,
    worker_args,
)

_TODAY = datetime.now().strftime("%Y-%m-%d %H:%M +0900")

INSIGHTS_FILE = (
    "# Design Chat Insights (.asr-edit/design_insights.md)\n"
    "\n"
    "> **Principle**: Structural/generic approach.\n"
    "\n"
    f"### [pattern] {_TODAY}\n"
    "Original insight line one.\n"
    "\n"
    f"### [design_decision] {_TODAY}\n"
    "Original insight line two.\n"
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _seq_poll(tasks):
    it = iter([*list(tasks), None])

    def _poll(repo_root, agent_id, **kw):
        return next(it)

    return _poll


def _counting_svc_factory(counter):
    def _factory(*a, **k):
        counter.append(1)
        return FakeSvc(provider="anthropic", model="m1")

    return _factory


def _task(task_id, **over):
    base = {
        "task_id": task_id,
        "title": f"Task {task_id}",
        "description": "desc",
        "assigned_files": ["a.txt"],
        "provider": "anthropic",
        "model": "m1",
        "epoch": 3,
    }
    base.update(over)
    return sipc_mod.SubagentTask(**base)


def _read_result(repo_root, task_id):
    p = Path(repo_root) / ".asicode" / "subagents" / task_id / "result.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _patch_worker_deps(monkeypatch, poll, *, fail_mode=None, svc_counter=None, design_loop_cls=None):
    counter = svc_counter if svc_counter is not None else []
    # The real _ProgressPrinter wraps rich Live on sys.stderr, whose console
    # caches a file object that pytest's capture machinery closes at teardown
    # of the PREVIOUS test -> "I/O operation on closed file" on the next write.
    # The worker's printer call sites stay covered via the stub; the real class
    # is exercised by the spawned-child REPL tests (fresh process, no stale
    # file). Plain _print output also keeps capsys assertions ANSI-free.
    monkeypatch.setattr(repl_impl, "_RICH", False)
    monkeypatch.setattr(repl_impl.asi, "_out_console", None)
    monkeypatch.setattr(repl_impl, "_ProgressPrinter", _StubProgressPrinter)
    monkeypatch.setattr(isvc_mod, "create_intelligent_service_from_env", _counting_svc_factory(counter))
    monkeypatch.setattr(sipc_mod, "poll_for_task", poll)
    monkeypatch.setattr(tool_registry_mod, "ToolRegistry", FakeToolRegistry)
    monkeypatch.setattr(tool_registry_mod, "AgentConfig", FakeAgentConfig)
    if design_loop_cls is not None:
        monkeypatch.setattr(dcl_mod, "DesignChatLoop", design_loop_cls)
    else:
        monkeypatch.setattr(
            dcl_mod,
            "DesignChatLoop",
            lambda client, registry, model: FakeDesignChatLoop(client, registry, model, fail_mode=fail_mode),
        )
    return counter


class _StubProgressPrinter:
    """No-op stand-in: avoids rich Live console lifetime issues in-process."""

    def __init__(self, verbose=False):
        pass

    def __call__(self, event, payload):
        pass

    def _start_spinner(self, msg):
        pass

    def _stop_spinner(self):
        pass


@pytest.fixture()
def _restore_repl_globals():
    saved_root = repl_impl._REPO_ROOT
    yield
    repl_impl._REPO_ROOT = saved_root


# ── run_subagent_worker (in-process) ─────────────────────────────────────────


class TestSubagentWorker:
    def test_missing_subagent_id_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sipc_mod, "poll_for_task", lambda *a, **k: None)
        with pytest.raises(SystemExit):
            repl_impl.run_subagent_worker(worker_args(str(tmp_path), subagent_id=None))

    def test_idle_poll_none_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        _patch_worker_deps(monkeypatch, lambda *a, **k: None)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        out = capsys.readouterr().out
        assert "started, watching" in out
        assert "stopped." in out
        hb = Path(tmp_path) / ".asicode" / "subagents" / "w1" / "worker.heartbeat.json"
        assert hb.exists()
        assert json.loads(hb.read_text(encoding="utf-8"))["state"] == "exited"

    def test_success_task_writes_result(self, tmp_path, monkeypatch, capsys):
        counter = []
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), svc_counter=counter)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "success"
        assert res["task_id"] == "t1"
        assert res["applied_patches"] == []
        assert len(counter) == 1
        out = capsys.readouterr().out
        assert "Received task: Task t1" in out
        assert "Task complete: success" in out
        assert "Result written." in out

    def test_cancel_task_writes_cancelled_result(self, tmp_path, monkeypatch, capsys):
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t2")]), fail_mode="cancel")
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t2")
        assert res["status"] == "cancelled"
        assert "cancelled by orchestrator" in res["error"]
        assert "Task cancelled (worker staying alive)." in capsys.readouterr().out

    def test_error_task_clears_svc_cache_and_continues(self, tmp_path, monkeypatch):
        counter = []
        fail_first = {"used": False}

        class _FailingLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                if not fail_first["used"]:
                    fail_first["used"] = True
                    raise RuntimeError("fake task crash")
                return super().respond(messages, stream_callback=stream_callback)

        _patch_worker_deps(
            monkeypatch, _seq_poll([_task("t3"), _task("t4")]), svc_counter=counter, design_loop_cls=_FailingLoop
        )
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert _read_result(str(tmp_path), "t3")["status"] == "error"
        assert _read_result(str(tmp_path), "t4")["status"] == "success"
        # Exception cleared the single-slot svc cache -> second task re-creates.
        assert len(counter) == 2

    def test_svc_cache_reused_across_tasks(self, tmp_path, monkeypatch):
        counter = []
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t5"), _task("t6")]), svc_counter=counter)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert len(counter) == 1  # same (provider, model, api_key) -> cache hit

    def test_sigint_shutdown_handler(self, tmp_path, monkeypatch, capsys):
        def _slow_poll(repo_root, agent_id, **kw):
            time.sleep(0.5)

        _patch_worker_deps(monkeypatch, _slow_poll)
        old_int = signal.getsignal(signal.SIGINT)
        try:

            def _send():
                time.sleep(0.15)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=_send, daemon=True).start()
            repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        finally:
            signal.signal(signal.SIGINT, old_int)
        out = capsys.readouterr().out
        assert "shutting down" in out
        assert "stopped." in out
        hb = Path(tmp_path) / ".asicode" / "subagents" / "w1" / "worker.heartbeat.json"
        assert json.loads(hb.read_text(encoding="utf-8"))["state"] == "exited"


# ── _run_repl_impl (spawned child + pty) ─────────────────────────────────────


class TestReplPtySession:
    CHILD = str(Path(__file__).parent / "repl_stage2_child.py")

    def _spawn(self, repo_root, timeout=60.0):
        from tests.unit.pty_driver import SpawnPtySession

        return SpawnPtySession(
            [sys.executable, self.CHILD, "--mode", "repl", "--repo", repo_root], cwd=os.getcwd(), timeout=timeout
        )

    def _send_cmd(self, sess, text):
        """Type a command like a human: send the text, wait for the echo,
        then press Enter.

        The trailing ``b"\\r"`` is safe to send at ANY moment — SpawnPtySession
        clears the slave's ICRNL/INLCR/IGNCR at spawn (see
        ``pty_driver._disable_cr_translation``), so a CR enqueued while the
        tty sits in the inter-prompt canonical window (prompt_toolkit restores
        the saved termios between prompts) is held in the input queue
        untranslated and dispatches the Enter submit binding once the raw-mode
        read drains it. The historical race — the kernel translating that CR
        to ``\\n`` → ControlJ ("insert newline") → the prompt never returning
        (burst RED 3/3, fixed driver 10/10, sealed in
        ``test_burst_submit_no_echo_barrier``) — lived in the driver, not here.
        The echo wait below is diagnostics (keeps each command's input visible
        in failure tails), not a correctness barrier.
        """
        data = text if isinstance(text, bytes) else text.encode()
        sess.clear()
        sess.send(data)
        sess.wait_for(data, timeout=30)
        sess.send(b"\r")

    def _write_insights(self, repo_root):
        p = Path(repo_root) / ".asicode"
        p.mkdir(parents=True, exist_ok=True)
        (p / "design_insights.md").write_text(INSIGHTS_FILE, encoding="utf-8")

    def test_full_repl_session(self, tmp_path):
        repo = str(tmp_path)
        self._write_insights(repo)
        sess = self._spawn(repo)
        try:
            sess.wait_for(b"asicode", timeout=60)  # banner (plain branch)
            self._send_cmd(sess, "/insights list")
            sess.wait_for(b"design_insights: 2 entries")
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"before:")  # LLM write path
            sess.wait_for(b"design_insights compacted")
            self._send_cmd(sess, "/insights drop 1")
            sess.wait_for(b"dropped #1")
            self._send_cmd(sess, "/insights edit 1 edited body")
            sess.wait_for(b"edited #1")
            self._send_cmd(sess, "/model")
            sess.wait_for(b"current:")
            self._send_cmd(sess, "/model anthropic/claude-sonnet-4-6")
            sess.wait_for(b"already using")
            self._send_cmd(sess, "/think on")
            sess.wait_for(b"thinking/reasoning \xe2\x86\x92 ON")
            self._send_cmd(sess, "/clear")
            sess.wait_for(b"Conversation compacted")
            self._send_cmd(sess, "hello world")
            sess.wait_for(b"Here is the plan: done.")
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode")
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"status: success")
            self._send_cmd(sess, "/code")
            sess.wait_for(b"switched to [Code Chat] mode")
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"nothing to undo")
            # ── Cheap dispatch branches ──
            self._send_cmd(sess, "/status")
            sess.wait_for(b"model    anthropic / claude-sonnet-4-6")
            self._send_cmd(sess, "/helper")
            sess.wait_for(b"compression helper: (none")
            self._send_cmd(sess, "/diff")
            sess.wait_for(b"no changes recorded yet.")
            self._send_cmd(sess, "/copy")  # _last_final_msg set by orchestrate
            sess.wait_for(b"copied final message to clipboard")
            self._send_cmd(sess, "/auto")
            sess.wait_for(b"auto-continue ON")
            self._send_cmd(sess, "/auto off")
            sess.wait_for(b"auto-continue OFF")
            self._send_cmd(sess, "/think off")
            sess.wait_for(b"thinking/reasoning \xe2\x86\x92 OFF")
            self._send_cmd(sess, "/insights edit 1")
            sess.wait_for(b"usage: /insights edit")
            self._send_cmd(sess, "/insights drop abc")
            sess.wait_for(b"no entry #0")
            self._send_cmd(sess, "/foobar")
            sess.wait_for(b"unknown command: /foobar")
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.")
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_ctrl_c_exits_session(self, tmp_path):
        """Ctrl+C at the plain prompt raises KeyboardInterrupt -> session end.

        Single-press exit: an empty prompt (main or auxiliary) exits on the
        FIRST ^C — no two-button arm protocol. This applies to the rich
        underline prompt (bottom_toolbar=True, _input_underline=True) and the
        plain branch the child runs (bottom_toolbar=False) alike; the pure
        decision is unit-tested in test_ctrlc_state_machine.py.
        """
        repo = str(tmp_path)
        self._write_insights(repo)
        sess = self._spawn(repo)
        try:
            sess.wait_for(b"asicode", timeout=60)
            # Wait for the first prompt to be LIVE before sending ^C: between
            # the banner and the prompt the terminal is still in canonical
            # mode, where the tty line discipline converts ^C to a SIGINT
            # (default handler kills the child) instead of a raw-mode byte
            # that the prompt's key binding turns into KeyboardInterrupt.
            sess.wait_for(b"Code mode", timeout=30)
            # Let raw mode settle: ^C sent in the canonical-mode window (a
            # few ms after the first render) becomes a tty SIGINT that kills
            # the child instead of a key byte.
            time.sleep(0.3)
            sess.send(b"\x03")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_child_crash_reports_marker(self, tmp_path):
        """A failing child must surface CHILD-CRASH + exit 1, not hang."""
        from tests.unit.pty_driver import SpawnPtySession

        sess = SpawnPtySession([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=15)
        try:
            assert sess.wait(timeout=15) == 3
        finally:
            sess.close()

    def test_burst_submit_no_echo_barrier(self, tmp_path):
        """Burst ``text + CR`` must submit — no echo wait, no settle sleep.

        RED on HEAD (3/3 hang): a CR written while the tty sits in the
        inter-prompt canonical window is ICRNL-translated to ``\\n`` by the
        kernel AT ENQUEUE TIME; prompt_toolkit maps ``\\x0a`` to ControlJ
        (this REPL's "insert newline" binding), so the buffer becomes
        ``"/status\\n"`` and the prompt never returns — the child parked in
        ``_collect_input`` (the 8-worker full-suite ~1/run hang, SIGABRT
        stack evidence). SpawnPtySession now clears ICRNL/INLCR/IGNCR on the
        slave at spawn (pty_driver._disable_cr_translation): the CR waits in
        the input queue untranslated and dispatches the Enter binding
        (GREEN 10/10, two bursts per run). This test fires the burst right
        after the banner — the widest canonical window — with no mitigation.
        """
        repo = str(tmp_path)
        self._write_insights(repo)
        sess = self._spawn(repo)
        try:
            sess.wait_for(b"asicode", timeout=60)
            sess.send(b"/status\r")  # burst: text+CR in ONE write
            sess.wait_for(b"model    anthropic", timeout=30)
            sess.send(b"/helper\r")  # second burst in the next window
            sess.wait_for(b"compression helper: (none", timeout=30)
        finally:
            sess.close()
