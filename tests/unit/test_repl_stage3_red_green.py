"""repl_impl stage-3 RED→GREEN: run_subagent_worker residual exception branches
(in-process) + the interactive-command REPL branches (spawned child + pty).

Stage-2 left ~709 lines missing. This file targets the residual branches:

* run_subagent_worker (L6445-6979): idle-heartbeat failures, cancel-sentinel
  watcher, heartbeat-writer failures, git-diff failure, hb-stream callback
  branches, partition failure, write-result failure retry, spinner stop
  failures, exited-heartbeat failure, max_turns/error status mapping.
* _dispatch_command interactive branches (helper/dev_models/insights
  prune/verify/failure-patterns/undo/model-api-key/think-effort/auto-cap) and
  the startup sweep/_init branches — driven through a real pty child.

The pty child re-applies the same fakes as repl_stage2_child.py but lives in
its OWN file (repl_stage3_child.py) so the parallel session's in-flight edits
to the stage-2 files are never touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

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

_TASK_BASE = {
    "task_id": "t1",
    "title": "Task t1",
    "description": "desc",
    "assigned_files": ["a.txt"],
    "provider": "anthropic",
    "model": "m1",
    "epoch": 3,
}


def _task(task_id, **over):
    base = dict(_TASK_BASE, task_id=task_id, title=f"Task {task_id}")
    base.update(over)
    return sipc_mod.SubagentTask(**base)


def _seq_poll(tasks):
    it = iter([*list(tasks), None])

    def _poll(repo_root, agent_id, **kw):
        return next(it)

    return _poll


def _read_result(repo_root, task_id):
    p = Path(repo_root) / ".asicode" / "subagents" / task_id / "result.json"
    return json.loads(p.read_text(encoding="utf-8"))


class _StubProgressPrinter:
    """Default no-op printer (same contract as stage-2's stub)."""

    def __init__(self, verbose=False):
        self._stopped = False

    def __call__(self, event, payload):
        pass

    def _start_spinner(self, msg):
        pass

    def _stop_spinner(self):
        self._stopped = True


class _RaisingPrinter:
    """Printer whose _stop_spinner raises — drives the stop-failure branches."""

    def __init__(self, verbose=False):
        pass

    def __call__(self, event, payload):
        pass

    def _start_spinner(self, msg):
        pass

    def _stop_spinner(self):
        raise RuntimeError("fake spinner stop crash")


def _patch_worker_deps(monkeypatch, poll, *, fail_mode=None, svc_counter=None, design_loop_cls=None, printer_cls=None):
    counter = svc_counter if svc_counter is not None else []

    def _factory(*a, **k):
        counter.append(1)
        return FakeSvc(provider="anthropic", model="m1")

    monkeypatch.setattr(repl_impl, "_RICH", False)
    monkeypatch.setattr(repl_impl.asi, "_out_console", None)
    monkeypatch.setattr(repl_impl, "_ProgressPrinter", printer_cls or _StubProgressPrinter)
    monkeypatch.setattr(isvc_mod, "create_intelligent_service_from_env", _factory)
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


# ── run_subagent_worker residual branches ────────────────────────────────────


class TestWorkerResidual:
    def test_initial_idle_hb_failure(self, tmp_path, monkeypatch, capsys):
        """write_worker_idle_heartbeat raising must not kill the worker."""
        _patch_worker_deps(monkeypatch, lambda *a, **k: None)
        monkeypatch.setattr(
            sipc_mod, "write_worker_idle_heartbeat", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
        )
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert "started, watching" in capsys.readouterr().out

    def test_idle_hb_writer_exception_swallowed(self, tmp_path, monkeypatch):
        """The daemon writer's try/except (L6546) swallows a failing write."""
        calls = {"n": 0}

        def _flaky_hb(*a, **k):
            calls["n"] += 1
            raise OSError("hb disk full")

        _patch_worker_deps(monkeypatch, _seq_poll([]))
        monkeypatch.setattr(sipc_mod, "_IDLE_HEARTBEAT_INTERVAL_S", 0.02)
        monkeypatch.setattr(sipc_mod, "write_worker_idle_heartbeat", _flaky_hb)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert calls["n"] >= 1

    def test_cancel_sentinel_fires_watcher(self, tmp_path, monkeypatch, capsys):
        """cancel.json present at task start -> watcher prints + unlinks + sets
        the task-scope cancel_event, then the task completes normally."""
        cancel_path = Path(tmp_path) / ".asicode" / "subagents" / "w1" / "cancel.json"
        cancel_path.parent.mkdir(parents=True, exist_ok=True)
        cancel_path.write_text("{}", encoding="utf-8")
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "success"
        out = capsys.readouterr().out
        assert "cancel signal received" in out
        assert not cancel_path.exists()  # fire-once unlink

    def test_cancel_sentinel_unlink_failure(self, tmp_path, monkeypatch, capsys):
        """unlink failure inside the watcher is logged, not fatal."""
        cancel_path = Path(tmp_path) / ".asicode" / "subagents" / "w1" / "cancel.json"
        cancel_path.parent.mkdir(parents=True, exist_ok=True)
        cancel_path.write_text("{}", encoding="utf-8")

        def _fail_unlink(p):
            raise OSError("busy")

        monkeypatch.setattr(os, "unlink", _fail_unlink)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        try:
            repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        finally:
            monkeypatch.undo()
        assert "cancel signal received" in capsys.readouterr().out

    def test_cancel_watcher_poll_exception(self, tmp_path, monkeypatch):
        """check_cancel_sentinel raising inside the watcher is swallowed."""

        def _boom(repo_root, agent_id):
            raise OSError("sentinel read race")

        monkeypatch.setattr(sipc_mod, "check_cancel_sentinel", _boom)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert _read_result(str(tmp_path), "t1")["status"] == "success"

    def test_initial_task_heartbeat_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sipc_mod, "write_heartbeat", lambda *a, **k: (_ for _ in ()).throw(OSError("io")))
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert _read_result(str(tmp_path), "t1")["status"] == "success"

    def test_task_heartbeat_writer_failure(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def _flaky_hb(*a, **k):
            calls["n"] += 1
            raise OSError("hb io")

        monkeypatch.setattr(sipc_mod, "HEARTBEAT_INTERVAL_S", 0.02)
        monkeypatch.setattr(sipc_mod, "write_heartbeat", _flaky_hb)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert calls["n"] >= 2  # initial write + at least one periodic

    def test_git_diff_failure(self, tmp_path, monkeypatch):
        def _boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd=["git"], timeout=10)

        monkeypatch.setattr(subprocess, "run", _boom)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "success"
        assert res["diff"] == ""

    def test_hb_stream_cb_tool_event_and_printer_failure(self, tmp_path, monkeypatch):
        """design_tool_call running events update _hb_state; a raising printer
        callback is swallowed (L6784)."""
        events = []

        class _LoopWithEvents(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                if stream_callback is not None:
                    stream_callback("design_llm_call", None)
                    stream_callback("design_tool_call", {"status": "running", "tool": "edit_text"})
                events.append(1)
                return super().respond(messages)

        class _RaiseOnCallPrinter(_StubProgressPrinter):
            def __call__(self, event, payload):
                raise RuntimeError("fake printer crash")

        _patch_worker_deps(
            monkeypatch, _seq_poll([_task("t1")]), design_loop_cls=_LoopWithEvents, printer_cls=_RaiseOnCallPrinter
        )
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert events == [1]
        assert _read_result(str(tmp_path), "t1")["status"] == "success"

    def test_max_turns_status_mapping(self, tmp_path, monkeypatch):
        class _MaxTurnsLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                return DesignChatResult(
                    content="out of turns",
                    tool_calls_made=[],
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    last_call_prompt_tokens=0,
                    last_call_completion_tokens=0,
                    last_call_cache_read_tokens=0,
                    last_call_cache_creation_tokens=0,
                    provider="anthropic",
                    is_error=False,
                    error_type=None,
                    hit_max_iterations=True,
                    total_llm_calls=5,
                )

        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), design_loop_cls=_MaxTurnsLoop)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert _read_result(str(tmp_path), "t1")["status"] == "max_turns"

    def test_error_result_status_mapping(self, tmp_path, monkeypatch):
        class _ErrorResultLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                return DesignChatResult(
                    content="fake error result",
                    tool_calls_made=[],
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    last_call_prompt_tokens=0,
                    last_call_completion_tokens=0,
                    last_call_cache_read_tokens=0,
                    last_call_cache_creation_tokens=0,
                    provider="anthropic",
                    is_error=True,
                    error_type="general",
                    hit_max_iterations=False,
                    total_llm_calls=0,
                )

        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), design_loop_cls=_ErrorResultLoop)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "error"

    def test_svc_none_raises_and_writes_error_result(self, tmp_path, monkeypatch):
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        monkeypatch.setattr(isvc_mod, "create_intelligent_service_from_env", lambda *a, **k: None)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "error"
        assert "failed to initialize LLM service" in res["error"]

    def test_partition_failure_in_error_path(self, tmp_path, monkeypatch):
        class _CrashLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                raise RuntimeError("fake crash")

        def _boom(repo_root, assigned_files):
            raise OSError("git status race")

        monkeypatch.setattr(sipc_mod, "partition_changed_files", _boom)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), design_loop_cls=_CrashLoop)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "error"
        assert res["applied_patches"] == []

    def test_partition_failure_in_cancel_path(self, tmp_path, monkeypatch):
        def _boom(repo_root, assigned_files):
            raise OSError("git status race")

        monkeypatch.setattr(sipc_mod, "partition_changed_files", _boom)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), fail_mode="cancel")
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "cancelled"
        assert res["applied_patches"] == []

    def test_write_result_failure_minimal_retry(self, tmp_path, monkeypatch, capsys):
        """write_result raising once -> minimal error result retry succeeds."""
        calls = {"n": 0}
        real_write = sipc_mod.write_result

        def _flaky_write(repo_root, result):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("result write disk full")
            return real_write(repo_root, result)

        monkeypatch.setattr(sipc_mod, "write_result", _flaky_write)
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "error"
        assert "result write failed" in res["error"]
        assert "Minimal error result written." in capsys.readouterr().out

    def test_write_result_total_failure(self, tmp_path, monkeypatch):
        """Both write attempts failing must NOT kill the worker (the
        'always writes a result' contract degrades to a logged timeout)."""
        monkeypatch.setattr(sipc_mod, "write_result", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))

    def test_spinner_stop_failure_cancel(self, tmp_path, monkeypatch):
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), fail_mode="cancel", printer_cls=_RaisingPrinter)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert _read_result(str(tmp_path), "t1")["status"] == "cancelled"

    def test_spinner_stop_failure_error(self, tmp_path, monkeypatch):
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), fail_mode="error", printer_cls=_RaisingPrinter)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert _read_result(str(tmp_path), "t1")["status"] == "error"

    def test_exited_heartbeat_failure(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sipc_mod, "write_worker_exited_heartbeat", lambda *a, **k: (_ for _ in ()).throw(OSError("io"))
        )
        _patch_worker_deps(monkeypatch, lambda *a, **k: None)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert "stopped." in capsys.readouterr().out

    def test_agent_cancelled_writes_partial_changes(self, tmp_path, monkeypatch):
        """AgentCancelled path records partition_changed_files in the result."""
        _patch_worker_deps(monkeypatch, _seq_poll([_task("t1")]), fail_mode="cancel")
        monkeypatch.setattr(sipc_mod, "partition_changed_files", lambda repo_root, assigned: (["a.txt"], []))
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        res = _read_result(str(tmp_path), "t1")
        assert res["status"] == "cancelled"
        assert res["applied_patches"] == ["a.txt"]


# ── spawned-child REPL sessions (real pty) ──────────────────────────────────

_CHILD = str(Path(__file__).parent / "repl_stage3_child.py")


class TestSpawnedReplStage3:
    CHILD = _CHILD

    def _spawn(self, repo_root, timeout=90.0, *extra):
        from tests.unit.pty_driver import SpawnPtySession

        return SpawnPtySession(
            [sys.executable, self.CHILD, "--repo", repo_root, *extra], cwd=os.getcwd(), timeout=timeout
        )

    def _send_cmd(self, sess, text, echo_wait=None):
        """Type a command like a human: send text, wait for prompt_toolkit to
        render the echo, then press Enter (see stage-2 _send_cmd rationale)."""
        data = text if isinstance(text, bytes) else text.encode()
        sess.clear()
        sess.send(data)
        sess.wait_for(echo_wait or data, timeout=30)
        time.sleep(0.15)
        sess.send(b"\r")

    def _wait_prompt(self, sess):
        sess.wait_for(b"asicode", timeout=60)
        sess.wait_for(b"Code mode", timeout=30)
        time.sleep(0.3)

    def test_session_a_model_helper_think_auto(self, tmp_path):
        """Sweep-raise + helper set/off + dev_N slots + /model list + /think
        effort + /auto cap/error + insights no-file/unknown + prune usage."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--sweep-raise", "--ollama-models", "--no-insights")
        try:
            self._wait_prompt(sess)
            # /helper: set -> show -> off -> show-none
            self._send_cmd(sess, "/helper anthropic/claude-sonnet-4-6")
            sess.wait_for(b"compression helper set", timeout=30)
            self._send_cmd(sess, "/helper")
            sess.wait_for(b"compression helper: anthropic / claude-sonnet-4-6", timeout=30)
            self._send_cmd(sess, "/helper off")
            sess.wait_for(b"compression helper cleared", timeout=30)
            self._send_cmd(sess, "/helper")
            sess.wait_for(b"compression helper: (none", timeout=30)
            # /model dev_N slots
            self._send_cmd(sess, "/model dev_1 qwen2.5-coder:3b")
            sess.wait_for(b"dev_1 set: anthropic / qwen2.5-coder:3b", timeout=30)
            self._send_cmd(sess, "/model dev_1")
            sess.wait_for(b"dev_1: anthropic / qwen2.5-coder:3b", timeout=30)
            self._send_cmd(sess, "/model dev_2")
            sess.wait_for(b"(not set", timeout=30)
            self._send_cmd(sess, "/model dev_abc")
            sess.wait_for(b"invalid slot", timeout=30)
            self._send_cmd(sess, "/model dev_1 off")
            sess.wait_for(b"dev_1 cleared", timeout=30)
            self._send_cmd(sess, "/model dev_1 qwen2.5-coder:3b")
            sess.wait_for(b"dev_1 set", timeout=30)
            # /model (no args): current + dev slots + known models + ollama
            self._send_cmd(sess, "/model")
            sess.wait_for(b"current: anthropic / claude-sonnet-4-6", timeout=30)
            sess.wait_for(b"sub-agent slots", timeout=30)
            sess.wait_for(b"dev_1: anthropic / qwen2.5-coder:3b", timeout=30)
            sess.wait_for(b"ollama (local):", timeout=30)
            sess.wait_for(b"qwen2.5-coder:3b", timeout=30)
            # /think effort + auto
            self._send_cmd(sess, "/think high")
            sess.wait_for(b"thinking/reasoning \xe2\x86\x92 ON (effort=high)", timeout=30)
            self._send_cmd(sess, "/think auto")
            sess.wait_for(b"thinking/reasoning \xe2\x86\x92 auto", timeout=30)
            # /auto cap + error
            self._send_cmd(sess, "/auto 3")
            sess.wait_for(b"auto-continue ON", timeout=30)
            self._send_cmd(sess, "/auto abc")
            sess.wait_for(b"usage: /auto [N | on | off]", timeout=30)
            self._send_cmd(sess, "/auto off")
            sess.wait_for(b"auto-continue OFF", timeout=30)
            # insights: no-file + unknown + prune usage
            self._send_cmd(sess, "/insights")
            sess.wait_for(b"no design_insights file yet.", timeout=30)
            self._send_cmd(sess, "/insights badcmd")
            sess.wait_for(b"unknown subcommand 'badcmd'", timeout=30)
            self._send_cmd(sess, "/insights prune abc")
            sess.wait_for(b"usage: /insights prune <days>", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_b_failure_patterns_compact_verify(self, tmp_path):
        """Sweep-ok + failure-patterns (data/clear/prune/drop/badcmd) +
        /insights list-with-archive + compact success + verify success."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--sweep-ok", "--fp-data", "--archive-data")
        try:
            self._wait_prompt(sess)
            # /insights list: entries + archive count
            self._send_cmd(sess, "/insights list")
            sess.wait_for(b"design_insights: 2 entries", timeout=30)
            sess.wait_for(b"archive: 1 demoted entries", timeout=30)
            # failure-patterns: show (non-empty)
            self._send_cmd(sess, "/failure-patterns")
            sess.wait_for(b"failure-pattern store: 3 patterns", timeout=30)
            sess.wait_for(b"syntax error", timeout=30)
            # prune with valid threshold -> 1 pattern (k2) below 0.5
            self._send_cmd(sess, "/failure-patterns prune 0.5")
            sess.wait_for(b"pruned 1 pattern", timeout=30)
            # prune with bad threshold -> usage
            self._send_cmd(sess, "/failure-patterns prune abc")
            sess.wait_for(b"usage: /failure-patterns prune [threshold]", timeout=30)
            # drop by substring: multi-match (k1 reason + k3 reason both contain 'a')
            self._send_cmd(sess, "/failure-patterns drop a")
            sess.wait_for(b"2 patterns match 'a'", timeout=30)
            # drop by index (k1)
            self._send_cmd(sess, "/failure-patterns drop 1")
            sess.wait_for(b"dropped [edit_text] syntax error", timeout=30)
            # drop by index (k3)
            self._send_cmd(sess, "/failure-patterns drop 1")
            sess.wait_for(b"dropped [edit_text] race condition", timeout=30)
            # drop by index: out of range
            self._send_cmd(sess, "/failure-patterns drop 1")
            sess.wait_for(b"no pattern #1", timeout=30)
            # drop by substring: no match
            self._send_cmd(sess, "/failure-patterns drop nosuchthing")
            sess.wait_for(b"no pattern matching 'nosuchthing'", timeout=30)
            # prune on empty store -> nothing below threshold
            self._send_cmd(sess, "/failure-patterns prune 0.5")
            sess.wait_for(b"no patterns below threshold 0.5", timeout=30)
            # clear with y
            self._send_cmd(sess, "/failure-patterns clear")
            sess.wait_for(b"Clear the failure-pattern store", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for(b"failure-pattern store cleared.", timeout=30)
            # empty store display
            self._send_cmd(sess, "/failure-patterns")
            sess.wait_for(b"failure-pattern store: empty", timeout=30)
            # unknown subcommand
            self._send_cmd(sess, "/failure-patterns bogus")
            sess.wait_for(b"unknown subcommand 'bogus'", timeout=30)
            # compact success (LLM rewrites -> before/after; tty spinner path)
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"design_insights compacted", timeout=30)
            # verify success
            self._send_cmd(sess, "/insights verify")
            sess.wait_for(b"design_insights verified", timeout=30)
            # archive list
            self._send_cmd(sess, "/insights archive list")
            sess.wait_for(b"archived", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_c_compact_fail_nudge(self, tmp_path):
        """Compact LLM crash -> failure notice; session-end nudge fires and
        answering 'n' keeps the session flowing."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--compact-fail", "--helper-client-fail", "--nudge")
        try:
            self._wait_prompt(sess)
            # helper client creation failure
            self._send_cmd(sess, "/helper anthropic/claude-sonnet-4-6")
            sess.wait_for(b"failed to create helper client", timeout=30)
            # compact LLM crash -> failure notice (no partial write)
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"fake compact LLM crash", timeout=30)
            # exit -> nudge prompt (file > 6000 bytes) -> answer n
            self._send_cmd(sess, "exit")
            sess.wait_for(b"compact now? (y/N)", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_d_model_api_key_undo_baseline(self, tmp_path):
        """/model provider switch with inline API key (entered / cancelled /
        typo warning), baseline /undo with y, orchestrate with dev slot,
        insights prune old entries (y + no-old)."""
        repo = str(tmp_path)
        sess = self._spawn(
            repo,
            150,
            "--no-api-key",
            "--sweep-ok",
            "--git-repo",
            "--suggest-kick-fail",
            "--undo-baseline",
            "--helper-persisted",
            "--old-entries",
        )
        try:
            self._wait_prompt(sess)
            # /model (no args): helper line (persisted) shown
            self._send_cmd(sess, "/model")
            sess.wait_for(b"helper:  anthropic / claude-sonnet-4-6", timeout=30)
            # provider switch -> API key prompt -> enter a key
            self._send_cmd(sess, "/model openai/gpt-4o")
            sess.wait_for(b"OPENAI_API_KEY not set in environment.", timeout=30)
            self._send_cmd(sess, "sk-test-abc")
            sess.wait_for(b"using inline API key for openai", timeout=30)
            sess.wait_for(b"model switched: anthropic / claude-sonnet-4-6 \xe2\x86\x92 openai / gpt-4o", timeout=30)
            # switch back -> key prompt -> empty Enter -> cancelled + rollback
            self._send_cmd(sess, "/model anthropic/claude-sonnet-4-6")
            sess.wait_for(b"enter API key (or press Enter to cancel)", timeout=30)
            self._send_cmd(sess, "\r")
            sess.wait_for(b"cancelled \xe2\x80\x94 no API key provided.", timeout=30)
            # unknown model -> typo warning (key already in env -> switch ok)
            self._send_cmd(sess, "/model openai/unknown-model")
            sess.wait_for(b"model 'unknown-model' is not in the known list for openai", timeout=30)
            sess.wait_for(b"model switched: openai / gpt-4o \xe2\x86\x92 openai / unknown-model", timeout=30)
            # dev slot for the orchestrate subagent-model mapping
            self._send_cmd(sess, "/model dev_1 qwen2.5-coder:3b")
            sess.wait_for(b"dev_1 set", timeout=30)
            # chat turn: suggestion kick fails silently; change summary shown
            self._send_cmd(sess, "hello world")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            sess.wait_for(b"/diff full diff", timeout=30)
            # baseline undo: list + y -> reverted
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"revert 1 file(s)? (y/N)", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for("✓ reverted a.txt".encode(), timeout=30)
            # orchestrate turn with dev slot populated
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"status: success", timeout=30)
            self._send_cmd(sess, "/code")
            sess.wait_for(b"switched to [Code Chat] mode", timeout=30)
            # insights prune: old entries listed -> y -> pruned
            self._send_cmd(sess, "/insights prune 1")
            sess.wait_for(b"2 entries older than 1 days:", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for("✓ pruned 2 entries older than 1 days (2→0).".encode(), timeout=30)
            # second prune: nothing older
            self._send_cmd(sess, "/insights prune 1")
            sess.wait_for(b"no entries older than 1 days.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_e_checkpoint_undo_empty_verify(self, tmp_path):
        """Checkpoint undo path (y), empty-response turn, OCR/clipboard-image
        turn, verify on an empty file, notify-deferred message drain."""
        repo = str(tmp_path)
        sess = self._spawn(
            repo,
            150,
            "--undo-cp",
            "--empty-response",
            "--sweep-raise",
            "--empty-insights",
            "--clipboard-image",
            "--ocr-text",
            "--notify",
        )
        try:
            self._wait_prompt(sess)
            # clipboard-image + OCR turn -> empty-response notice
            self._send_cmd(sess, "hello")
            sess.wait_for(b"design chat returned an empty response", timeout=30)
            # checkpoint undo: turn recorded cp1 + a.txt -> list -> y
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"\xc2\xb7 a.txt", timeout=30)
            sess.wait_for(b"revert 1 file(s)? (y/N)", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for("✓ reverted 1 file(s)".encode(), timeout=30)
            # verify on empty file -> nothing to verify
            self._send_cmd(sess, "/insights verify")
            sess.wait_for(b"nothing to verify.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_f_verify_fail(self, tmp_path):
        """/insights verify with an error result -> failed notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--verify-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights verify")
            sess.wait_for(b"verify failed (tool loop error)", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_g_auth_retry_auto_error_stop(self, tmp_path):
        """Auth-error turn -> API-key retry prompt -> new key -> retried turn;
        auto-continue on an error turn stops the loop with a notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--auth-error-turn")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/auto on")
            sess.wait_for(b"auto-continue ON", timeout=30)
            # auth-error turn: retry prompt accepts a key, retry succeeds
            self._send_cmd(sess, "hello world")
            sess.wait_for(b"API key is expired or invalid.", timeout=30)
            self._send_cmd(sess, "sk-new-key")
            sess.wait_for(b"auto-continue: turn ended with an error \xe2\x80\x94 stopped", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_h_compact_noop_over_budget(self, tmp_path):
        """Over-budget /insights list + compact-noop demotion path + drop
        usage/abc/out-of-range + failure-patterns drop usage + clear-cancel +
        copy failure."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 150, "--insights-big", "--compact-mode=noop", "--fp-data", "--copy-fail", "--sweep-ok")
        try:
            self._wait_prompt(sess)
            # over-budget list line
            self._send_cmd(sess, "/insights list")
            sess.wait_for(b"over budget (6,000 bytes)", timeout=30)
            sess.wait_for(b"design_insights: 2 entries", timeout=30)
            # compact-noop (prompt-wrapped reply is not a true noop) ->
            # post-write over-budget backstop demotion message
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"design_insights compacted", timeout=30)
            sess.wait_for(b"demoted", timeout=30)
            # drop usage / abc / out-of-range
            self._send_cmd(sess, "/insights drop")
            sess.wait_for(b"usage: /insights drop <n>", timeout=30)
            self._send_cmd(sess, "/insights drop abc")
            sess.wait_for(b"no entry #0", timeout=30)
            self._send_cmd(sess, "/insights drop 99")
            sess.wait_for(b"no entry #99", timeout=30)
            # failure-patterns drop usage + clear cancel
            self._send_cmd(sess, "/failure-patterns drop")
            sess.wait_for(b"usage: /failure-patterns drop <n>", timeout=30)
            self._send_cmd(sess, "/failure-patterns clear")
            sess.wait_for(b"Clear the failure-pattern store", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"clear cancelled.", timeout=30)
            # copy failure
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/copy")
            sess.wait_for(b"clipboard copy failed", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_i_preamble_arch_fail(self, tmp_path):
        """Preamble-only insights list + archive-count failure branch."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--preamble-only", "--arch-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights list")
            sess.wait_for(b"design_insights: 0 entries", timeout=30)
            sess.wait_for(b"(preamble only", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_j_prune_cancel_and_bak_fail(self, tmp_path):
        """insights prune: cancel (n) + backup-failure (bak dir) + compact
        with finish_reason (fr) debug path."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--old-entries", "--bak-dir", "--compact-mode=fr")
        try:
            self._wait_prompt(sess)
            # prune -> cancel with n
            self._send_cmd(sess, "/insights prune 1")
            sess.wait_for(b"2 entries older than 1 days:", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"prune cancelled. File unchanged.", timeout=30)
            # prune -> y, backup fails (bak is a directory) but prune proceeds
            self._send_cmd(sess, "/insights prune 1")
            sess.wait_for(b"2 entries older than 1 days:", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for("\u2713 pruned 2 entries older than 1 days".encode(), timeout=30)
            # compact with finish_reason=stop -> success path
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"design_insights compacted", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_k_undo_nothing_nudge_yes(self, tmp_path):
        """Checkpoint undo with empty change list -> nothing-to-undo; session
        end nudge answered y -> compact runs."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--undo-cp-empty", "--nudge", "--compact-mode=noop")
        try:
            self._wait_prompt(sess)
            # a turn records the checkpoint, then /undo finds no changed files
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"nothing to undo", timeout=30)
            # exit -> nudge -> y -> compact runs
            self._send_cmd(sess, "exit")
            sess.wait_for(b"compact now? (y/N)", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_h2_compact_true_noop_over_budget(self, tmp_path):
        """True noop compact (echo messages) on an over-budget file -> the
        enforce-budget-by-demotion branch; under-budget -> already-compact."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--insights-big", "--compact-echo", "--compact-mode=noop")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"already compact", timeout=30)
            sess.wait_for(b"demoted", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_h3_compact_true_noop_under_budget(self, tmp_path):
        """True noop compact on a normal file -> already-compact-no-changes."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--compact-echo", "--compact-mode=noop")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(b"already compact", timeout=30)
            sess.wait_for(b"no changes", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def _run_compact_session(self, tmp_path, *extra, expect, nudge=False):
        """Shared driver: /insights compact then assert the expected output."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, *extra)
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights compact")
            sess.wait_for(expect, timeout=30)
            self._send_cmd(sess, "exit")
            if nudge:
                sess.wait_for(b"compact now? (y/N)", timeout=30)
                self._send_cmd(sess, "n")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_l_compact_truncated_retry(self, tmp_path):
        """finish_reason=length on attempt 1 -> doubled-budget retry -> success
        with the finish_reason debug log."""
        self._run_compact_session(tmp_path, "--compact-mode=truncated", expect=b"design_insights compacted")

    def test_session_m_compact_truncated_final_reasoning(self, tmp_path):
        """Reasoning model raises max_tokens floor; truncated after retry ->
        refused message."""
        self._run_compact_session(
            tmp_path,
            "--compact-mode=truncated-final",
            "--reasoning-model",
            expect=b"compaction refused: LLM response truncated",
        )

    def test_session_n_compact_empty_content(self, tmp_path):
        """Empty LLM content without exception -> diagnostic message."""
        self._run_compact_session(tmp_path, "--compact-mode=empty", expect=b"LLM returned empty content")

    def test_session_o_compact_drops_all(self, tmp_path):
        """Compactor dropping every entry -> refused message."""
        self._run_compact_session(tmp_path, "--compact-mode=drops-all", expect=b"all entries were dropped")

    def test_session_p_compact_tokens_accounting(self, tmp_path):
        """Successful compact with token counters -> accounting debug path."""
        self._run_compact_session(tmp_path, "--compact-mode=tokens", expect=b"design_insights compacted")

    def test_session_q_compact_keyboard_interrupt(self, tmp_path):
        """Ctrl+C inside the compact LLM call -> cancel notice."""
        self._run_compact_session(tmp_path, "--compact-mode=kb", expect=b"insights compact cancelled")

    def test_session_r_compact_notice_failure(self, tmp_path):
        """_compress_failure_notice raising -> last-resort inline notice."""
        self._run_compact_session(tmp_path, "--compact-fail", "--notice-fail", expect=b"fake compact LLM crash")

    def test_session_s_compact_bak_failure(self, tmp_path):
        """Backup copy failing (bak is a directory) is logged, not fatal."""
        self._run_compact_session(tmp_path, "--compact-mode=fr", "--bak-dir", expect=b"design_insights compacted")

    def test_session_t_save_insight_auto_compact(self, tmp_path):
        """A turn recording save_insight on an over-budget file fires the
        post-turn auto-compact."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--save-insight", "--insights-big", "--compact-mode=noop", "--compact-echo")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"auto-compacting", timeout=30)
            sess.wait_for(b"already compact", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_u_general_compress_notify(self, tmp_path):
        """/general-mode occupancy gate fires background compress; the notify
        callback queues a deferred message that drains at the next prompt."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--ctx-low", "--notify")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/general")
            sess.wait_for(b"switched to [General Chat] mode", timeout=30)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            sess.wait_for(b"background compress complete", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_v_dc_crash_verbose(self, tmp_path):
        """Design-chat worker crash -> error print + verbose traceback."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--dc-crash", "--verbose")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"design chat error: fake design chat crash", timeout=30)
            sess.wait_for(b"Traceback (most recent call last)", timeout=30)
            time.sleep(0.8)  # let the prompt re-render before typing
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_w_verify_crash(self, tmp_path):
        """/insights verify with a crashing loop -> failure notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--verify-crash")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights verify")
            sess.wait_for(b"verify failed (tool loop error)", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_x_nudge_failure(self, tmp_path):
        """should_nudge raising at session end is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--nudge-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_y_orchestrator_error_verbose(self, tmp_path):
        """OrchestratorAgent raising -> error print + verbose traceback."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--orch-fail", "--verbose")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"orchestrator error: fake orchestrator crash", timeout=30)
            sess.wait_for(b"Traceback (most recent call last)", timeout=30)
            time.sleep(0.8)  # let the prompt re-render before typing
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_z_orchestrator_no_result(self, tmp_path):
        """OrchestratorAgent returning None -> no-result turn close."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--orch-none")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"[Orchestrator] >", timeout=30)  # silent no-result close
            time.sleep(0.8)  # let the prompt re-render before typing
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_aa_clear_dsm_crash(self, tmp_path):
        """/clear with a crashing session manager -> logged, banner still."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--dsm-crash")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/clear")
            sess.wait_for(b"asicode", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ab_next_suggest_off_mode_already(self, tmp_path):
        """/auto on with NEXT_SUGGEST disabled -> warning; repeated mode
        switches -> already-in-mode notices."""
        repo = str(tmp_path)
        from tests.unit.pty_driver import SpawnPtySession

        sess = SpawnPtySession(
            [sys.executable, self.CHILD, "--repo", repo], cwd=os.getcwd(), timeout=90, env={"ASICODE_NEXT_SUGGEST": "0"}
        )
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/auto on")
            sess.wait_for(b"next-step suggestion is disabled in config", timeout=30)
            self._send_cmd(sess, "/general")
            sess.wait_for(b"switched to [General Chat] mode", timeout=30)
            self._send_cmd(sess, "/general")
            sess.wait_for(b"already in [General Chat] mode", timeout=30)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"already in [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ac_resolve_none(self, tmp_path):
        """/model dev_N / model / helper with an unresolvable name -> silent
        continue."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--resolve-none")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/model dev_1 qwen2.5-coder:3b")
            sess.wait_for(b"Code mode", timeout=30)
            self._send_cmd(sess, "/model gpt-4o")
            sess.wait_for(b"Code mode", timeout=30)
            self._send_cmd(sess, "/helper gpt-4o")
            sess.wait_for(b"Code mode", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ad_llm_client_failure(self, tmp_path):
        """/model client creation raising -> error print + rollback."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--llm-client-fail", "--api-key-env")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/model openai/gpt-4o")
            sess.wait_for(b"failed to create LLM client: fake client crash", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ae_context_limit_failure(self, tmp_path):
        """/model with a crashing context-limit resolver -> silent continue."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--ctx-fail", "--api-key-env")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/model openai/gpt-4o")
            sess.wait_for(b"model switched", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_af_undo_checkpoint_fail(self, tmp_path):
        """/undo checkpoint revert failing -> failure notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--undo-cp-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"revert 1 file(s)? (y/N)", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for(b"revert failed", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ag_undo_baseline_partial_fail(self, tmp_path):
        """/undo baseline with a failing file -> per-file failure line."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--undo-base-fail", "--git-repo")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"revert 1 file(s)? (y/N)", timeout=30)
            self._send_cmd(sess, "y")
            sess.wait_for(b"failed to revert a.txt", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ah_fp_race(self, tmp_path):
        """/failure-patterns drop <substr> whose match disappears -> race msg."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--fp-data", "--fp-race")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/failure-patterns drop a")
            sess.wait_for(b"disappeared before drop", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ai_fp_store_error(self, tmp_path):
        """/failure-patterns with a crashing store -> store error print."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--fp-error")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/failure-patterns")
            sess.wait_for(b"failure-pattern store error: store read crash", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_aj_unknown_provider(self, tmp_path):
        """/model with an unknown provider -> unknown-provider hint + key
        prompt; empty key -> cancelled."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--no-api-key")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/model unknownprov/m1")
            sess.wait_for(b"unknown provider 'unknownprov'", timeout=30)
            self._send_cmd(sess, "\r")
            sess.wait_for(b"no API key provided", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ak_dev_slot_tip(self, tmp_path):
        """/model dev_2 set without dev_1 -> fallback tip line."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90)
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/model dev_2 qwen2.5-coder:3b")
            sess.wait_for(b"dev_2 set", timeout=30)
            sess.wait_for(b"(tip: unconfigured slots fall back", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_al_clear_margin_reset(self, tmp_path):
        """/clear with a fake margin stderr -> reset_bol path."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--margin-stderr")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/clear")
            sess.wait_for(b"asicode", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ba_init_none(self, tmp_path):
        """_init_repl_engine returning None -> immediate clean exit."""
        repo = str(tmp_path)
        from tests.unit.pty_driver import SpawnPtySession

        sess = SpawnPtySession([sys.executable, self.CHILD, "--repo", repo, "--init-none"], cwd=os.getcwd(), timeout=30)
        try:
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bb_compact_nothing(self, tmp_path):
        """Compact on an empty insights file -> nothing-to-compact."""
        self._run_compact_session(tmp_path, "--empty-insights", expect=b"nothing to compact.")

    def test_session_bc_helper_lazy_create_fail(self, tmp_path):
        """Persisted helper + client creation failure -> HELPER_CREATION_FAILED
        fallback to the main model."""
        self._run_compact_session(
            tmp_path,
            "--helper-persisted",
            "--helper-client-fail",
            "--compact-mode=fr",
            expect=b"design_insights compacted",
        )

    def test_session_bd_helper_lazy_create_ok(self, tmp_path):
        """Persisted helper + successful lazy client creation."""
        self._run_compact_session(
            tmp_path, "--helper-persisted", "--compact-mode=fr", expect=b"design_insights compacted"
        )

    def test_session_be_no_result(self, tmp_path):
        """Design loop returning None -> no-result notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--no-result")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"design chat returned no result", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bf_cache_tokens_group(self, tmp_path):
        """Turn with cache tokens -> cache-hit % group in the status line."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--cache-tokens")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"cache ", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bg_ocr_failure(self, tmp_path):
        """Clipboard image + OCR crash -> swallowed enrichment failure."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--clipboard-image", "--ocr-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bh_turn_summary_fail(self, tmp_path):
        """Change-summary raising at turn end is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--print-summary-fail", "--git-repo")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bi_cp_summary_fail(self, tmp_path):
        """Checkpoint summary raising at turn end is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--cp-summary-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bj_orch_keyboard_interrupt(self, tmp_path):
        """Orchestrator run KeyboardInterrupt -> cancelled notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--orch-kb")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"cancelled.", timeout=30)
            time.sleep(0.8)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bk_orch_turn_persist_fail(self, tmp_path):
        """Orchestrator turn persist failure is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--dsm-turn-crash")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"status: success", timeout=30)
            time.sleep(0.8)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bl_orch_no_result_close_fail(self, tmp_path):
        """Orchestrator no-result turn-close failure is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--orch-none", "--dsm-turn-crash")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"[Orchestrator] >", timeout=30)
            time.sleep(0.8)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bm_undo_cancel(self, tmp_path):
        """/undo answered n -> cancelled (checkpoint and baseline paths)."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--undo-cp")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"revert 1 file(s)? (y/N)", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"cancelled.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bn_insights_edit_errors(self, tmp_path):
        """/insights edit with a non-numeric index or out-of-range index."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90)
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights edit abc body")
            sess.wait_for(b"no entry #0", timeout=30)
            self._send_cmd(sess, "/insights edit 99 body")
            sess.wait_for(b"no entry #99", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bo_save_insight_under_budget(self, tmp_path):
        """save_insight turn on a normal file -> auto-compact gate skips."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--save-insight")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bp_save_insight_compact_fail(self, tmp_path):
        """save_insight auto-compact LLM failure is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--save-insight", "--insights-big", "--compact-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"auto-compacting", timeout=30)
            sess.wait_for(b"fake compact LLM crash", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"compact now? (y/N)", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bq_save_insight_stats_fail(self, tmp_path):
        """save_insight auto-compact stats failure is swallowed."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--save-insight", "--stats-fail")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_br_compact_ebd_fail(self, tmp_path):
        """Noop over-budget compact with demotion crashing -> fallback bytes
        and the no-reduction warning."""
        self._run_compact_session(
            tmp_path,
            "--insights-big",
            "--compact-echo",
            "--compact-mode=noop",
            "--ebd-fail",
            expect=b"no reduction possible",
            nudge=True,
        )

    def test_session_bs_compact_ebd_zero(self, tmp_path):
        """Noop over-budget compact with zero demotions -> warn message."""
        self._run_compact_session(
            tmp_path,
            "--insights-big",
            "--compact-echo",
            "--compact-mode=noop",
            "--ebd-zero",
            expect=b"no reduction possible",
            nudge=True,
        )

    def test_session_bt_auth_retry_ok(self, tmp_path):
        """Auth-error turn -> retry with a fresh key succeeds -> key commit."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--auth-retry-ok")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"API key is expired or invalid.", timeout=30)
            self._send_cmd(sess, "sk-fresh")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bu_auth_retry_crash(self, tmp_path):
        """Auth-error turn -> retry raising -> retry-failed result."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--auth-retry-crash")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"API key is expired or invalid.", timeout=30)
            self._send_cmd(sess, "sk-fresh")
            sess.wait_for(b"retry failed: fake retry crash", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bv_nudge_plain(self, tmp_path):
        """Session-end nudge without an /insights marker -> plain print."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--nudge-plain")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"plain accumulation notice", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_bw_partial_loss_guard(self, tmp_path):
        """Compact dropping >50% of entries on a normal file -> warning."""
        self._run_compact_session(
            tmp_path, "--entries-3", "--compact-mode=partial", expect=b"entries dropped", nudge=True
        )

    def test_session_ca_rich_turn_render(self, tmp_path):
        """Rich console enabled -> the final-answer render path (try branch)."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--rich")
        try:
            sess.wait_for(b"asicode", timeout=60)
            time.sleep(1.0)  # rich renders its own prompt chrome
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_cb_rich_orch_render(self, tmp_path):
        """Rich console enabled -> the orchestrator result render path."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--rich")
        try:
            sess.wait_for(b"asicode", timeout=60)
            time.sleep(1.0)  # rich renders its own prompt chrome
            self._send_cmd(sess, "/orchestrate")
            sess.wait_for(b"switched to [Orchestrator] mode", timeout=30)
            self._send_cmd(sess, "refactor the parser")
            sess.wait_for(b"status: success", timeout=30)
            sess.wait_for(b"Refactored the parser module.", timeout=30)
            time.sleep(0.8)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_cc_verify_keyboard_interrupt(self, tmp_path):
        """/insights verify with KeyboardInterrupt -> cancelled notice."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 90, "--verify-kb")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "/insights verify")
            sess.wait_for(b"verify cancelled", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_cd_compact_backstop_ebd_fail(self, tmp_path):
        """Post-write backstop demotion crashing -> swallowed + message."""
        self._run_compact_session(
            tmp_path, "--compact-mode=big", "--ebd-fail", expect=b"no demotion possible", nudge=True
        )

    def test_session_ce_compact_backstop_ebd_zero(self, tmp_path):
        """Post-write backstop demoting nothing -> honest warning."""
        self._run_compact_session(tmp_path, "--compact-mode=big", "--ebd-zero", expect=b"no demotion", nudge=True)

    def test_session_cf_undo_baseline_cancel(self, tmp_path):
        """/undo baseline answered n -> cancelled."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--undo-baseline", "--git-repo")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"revert 1 file(s)? (y/N)", timeout=30)
            self._send_cmd(sess, "n")
            sess.wait_for(b"cancelled.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_cg_input_eof_branches(self, tmp_path):
        """builtins.input raising EOFError -> nudge/undo/api-key EOF branches."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--input-eof", "--undo-cp", "--nudge", "--no-api-key")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            # /undo confirmation prompt hits EOF -> cancelled
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"cancelled.", timeout=30)
            # /model API-key prompt (_collect_input) -> Ctrl+C -> cancelled
            self._send_cmd(sess, "/model openai/gpt-4o")
            sess.wait_for(b"API key:", timeout=30)
            time.sleep(0.3)
            sess.send(b"\x03")
            sess.wait_for(b"cancelled \xe2\x80\x94 no API key provided.", timeout=30)
            # exit -> nudge prompt hits EOF -> no compact, session ends
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()

    def test_session_ch_input_eof_baseline_undo(self, tmp_path):
        """/undo baseline confirmation prompt hits EOF -> cancelled."""
        repo = str(tmp_path)
        sess = self._spawn(repo, 120, "--input-eof", "--undo-baseline", "--git-repo")
        try:
            self._wait_prompt(sess)
            self._send_cmd(sess, "hello")
            sess.wait_for(b"Here is the plan: done.", timeout=30)
            self._send_cmd(sess, "/undo")
            sess.wait_for(b"cancelled.", timeout=30)
            self._send_cmd(sess, "exit")
            sess.wait_for(b"session ended.", timeout=30)
            assert sess.wait(timeout=30) == 0
        finally:
            sess.close()


# ── worker idle-heartbeat success path (in-process) ──────────────────────────


class TestWorkerIdleHbSuccess:
    def _slow_none_poll(self, reps=4, gap=0.05):
        """Return a poll that idles long enough for the 0.02s daemon loop to
        run several times before returning None."""

        def _poll(repo_root, agent_id, **kw):
            for _ in range(reps):
                time.sleep(gap)
                yield

        # generator-based: run the loop body reps times, then None
        it = iter(_poll(None, None))

        def _p(repo_root, agent_id, **kw):
            for _ in range(reps):
                next(it)

        return _p

    def test_idle_hb_writer_success(self, tmp_path, monkeypatch):
        """The periodic idle-heartbeat write succeeds at least once."""
        calls = {"n": 0}
        real_hb = sipc_mod.write_worker_idle_heartbeat

        def _hb(*a, **k):
            calls["n"] += 1
            return real_hb(*a, **k)

        _patch_worker_deps(monkeypatch, self._slow_none_poll())
        monkeypatch.setattr(sipc_mod, "_IDLE_HEARTBEAT_INTERVAL_S", 0.02)
        monkeypatch.setattr(sipc_mod, "write_worker_idle_heartbeat", _hb)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert calls["n"] >= 2  # initial + periodic

    def test_idle_hb_flaky_then_ok(self, tmp_path, monkeypatch):
        """A flaky idle-heartbeat writer covers BOTH the success and the
        exception branch of the daemon loop."""
        calls = {"n": 0}
        real_hb = sipc_mod.write_worker_idle_heartbeat

        def _flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise OSError("hb hiccup")
            return real_hb(*a, **k)

        _patch_worker_deps(monkeypatch, self._slow_none_poll())
        monkeypatch.setattr(sipc_mod, "_IDLE_HEARTBEAT_INTERVAL_S", 0.02)
        monkeypatch.setattr(sipc_mod, "write_worker_idle_heartbeat", _flaky)
        repl_impl.run_subagent_worker(worker_args(str(tmp_path)))
        assert calls["n"] >= 3


from pathlib import Path


def test_session_cj_run_diff_render(tmp_path):
    """ASICODE_RUN_DIFF=1 -> the per-turn /diff full-diff render path."""
    from tests.unit.pty_driver import SpawnPtySession

    repo = str(tmp_path)
    sess = SpawnPtySession(
        [sys.executable, _CHILD, "--repo", repo, "--undo-baseline", "--git-repo"],
        cwd=os.getcwd(),
        timeout=120,
        env={"ASICODE_RUN_DIFF": "1"},
    )
    try:
        sess.wait_for(b"asicode", timeout=60)
        sess.wait_for(b"Code mode", timeout=30)
        time.sleep(0.3)

        def send_cmd(text):
            sess.clear()
            sess.send(text.encode())
            sess.wait_for(text.encode(), timeout=30)
            time.sleep(0.15)
            sess.send(b"\r")

        send_cmd("hello")
        sess.wait_for(b"Here is the plan: done.", timeout=30)
        # ASICODE_RUN_DIFF=1 -> the _render_run_diff branch (patched no-op)
        time.sleep(0.5)
        send_cmd("exit")
        sess.wait_for(b"session ended.", timeout=30)
        assert sess.wait(timeout=30) == 0
    finally:
        sess.close()
