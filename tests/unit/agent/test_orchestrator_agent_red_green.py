"""RED→GREEN coverage for OrchestratorAgent-level paths in orchestrator.py.

SCOPE NOTE (parallel-session coordination): a sibling session owns the
"round 1 pure-logic" file (tests/unit/test_orchestrator_red_green.py —
module-level helpers, FileLockManager, snapshot capture/restore, symbol
hints). This file covers the AGENT-level clusters only: continue_subagent,
the IPC wrapper paths (heartbeat on_poll / timeout abandon / review-retry
revert policy / cancelled grace loop), worker launch helpers, _run_subagent
in-process edges, _run_tool_loop session inheritance + status mapping,
drain/shutdown, _check_bg_subagent / background jobs, git-backed revert &
synthetic-diff helpers, and the _OrchestratorBackedRegistry facade.
"""

import os
import subprocess as real_subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import external_llm.agent.orchestrator as orch_mod
from external_llm.agent.orchestrator import (
    OrchestratorAgent,
    OrchestratorConfig,
    SubTaskSpec,
    _OrchestratorBackedRegistry,
)


def _mk_orch(tmp_path, callback=None, **cfg):
    """OrchestratorAgent against a Mock registry rooted at tmp_path."""
    registry = Mock()
    registry.repo_root = str(tmp_path)
    return OrchestratorAgent(
        Mock(),
        registry,
        OrchestratorConfig(**cfg),
        callback=callback,
    )


def _sub(task_id="dev_1", files=None, desc="do the thing"):
    return SubTaskSpec(
        task_id=task_id,
        title="T",
        description=desc,
        assigned_files=list(files or []),
        dependencies=[],
    )


def _git_repo(tmp_path, *extra_files):
    """Init a real git repo; commit tracked.py plus any extra files."""

    def git(*args):
        real_subprocess.run(
            ["git", *args],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            timeout=30,
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (tmp_path / "tracked.py").write_text("ORIGINAL\n")
    for f in extra_files:
        (tmp_path / f).write_text("BASE\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    return git


# ══════════════════════════════════════════════════════════════════════
# A. continue_subagent — previously fully uncovered (13 miss)
# ══════════════════════════════════════════════════════════════════════


class TestContinueSubagent:
    def test_cancelled_before_start(self, tmp_path):
        ce = threading.Event()
        ce.set()
        orch = _mk_orch(tmp_path, cancel_event=ce)
        res = orch.continue_subagent("dev_1", "keep going")
        assert res.status == "cancelled"
        assert "cancelled" in res.summary.lower()

    def test_wraps_prior_context_and_maps_result(self, tmp_path):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path)
        seen = {}

        def fake_run(subtask, extra_turns=0, task_text=None, original_request=""):
            seen["desc"] = subtask.description
            seen["extra"] = extra_turns
            return AgentResult(
                status="success",
                final_message="done",
                turns=[SimpleNamespace()],
                applied_patches=[],
            )

        orch._run_subagent = fake_run
        res = orch.continue_subagent("dev_2", "fix more", prior_context="EARLIER\nstuff")
        assert "[CONTINUE FROM PREVIOUS SESSION]" in seen["desc"]
        assert seen["desc"].endswith("fix more")
        assert seen["extra"] == 5
        assert res.status == "success"
        assert res.summary == "done"
        assert res.total_turns == 1
        assert res.metadata == {"continued": True, "agent_id": "dev_2"}

    def test_bare_request_without_prior_context(self, tmp_path):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path)
        seen = {}
        orch._run_subagent = lambda st, extra_turns=0, task_text=None, original_request="": (
            seen.setdefault("d", st.description),
            AgentResult(status="max_turns", final_message="", turns=[], applied_patches=[]),
        )[1]
        res = orch.continue_subagent("dev_1", "plain")
        assert seen["d"] == "plain"
        assert res.status == "max_turns"
        assert res.summary == "Sub-agent continuation completed"


# ══════════════════════════════════════════════════════════════════════
# B. IPC wrapper edges in _run_subagent_ipc
# ══════════════════════════════════════════════════════════════════════


def _ipc_orch(tmp_path, callback=None, **cfg):
    return _mk_orch(
        tmp_path,
        callback=callback,
        subagent_mode="ipc",
        subagent_models={"1": ("prov", "model-x", "key")},
        **cfg,
    )


def _fake_ipc_result(status="success", turns=1):
    from external_llm.agent.subagent_ipc import SubagentResult

    return SubagentResult(task_id="dev_1", status=status, final_message="ok", turns=turns)


def _patch_ipc_mocks(monkeypatch, wait_result=None):
    import external_llm.agent.subagent_ipc as ipc

    monkeypatch.setattr(ipc, "clear_result", lambda *a, **k: None)
    monkeypatch.setattr(ipc, "write_task", lambda *a, **k: None)
    monkeypatch.setattr(ipc, "wait_for_result", wait_result or (lambda *a, **k: _fake_ipc_result()))


def _patch_ipc_fs(monkeypatch):
    monkeypatch.setattr(orch_mod, "_capture_assigned_snapshots", lambda *a, **k: {})
    monkeypatch.setattr(orch_mod, "asr_subagent_argv", lambda rr: ["asi", "--subagent"])


class TestRunSubagentIpc:
    def test_heartbeat_on_poll_emits_turn_and_tool(self, tmp_path, monkeypatch):
        import external_llm.agent.subagent_ipc as ipc

        events = []
        orch = _ipc_orch(tmp_path, callback=lambda e, d: events.append((e, d)))

        def fake_wait(*a, **k):
            on_poll = k.get("on_poll")
            if on_poll:
                monkeypatch.setattr(
                    ipc,
                    "read_heartbeat_state",
                    lambda rr, aid: {"turn": 3, "last_tool": "read_file"},
                )
                on_poll(1.5, "dev_1")
            return _fake_ipc_result()

        _patch_ipc_mocks(monkeypatch, wait_result=fake_wait)
        _patch_ipc_fs(monkeypatch)

        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success"
        waits = [d for e, d in events if e == "subagent_waiting_ipc" and d.get("turn")]
        assert waits and waits[0]["turn"] == 3 and waits[0]["last_tool"] == "read_file"

    def test_heartbeat_read_failure_still_emits(self, tmp_path, monkeypatch):
        import external_llm.agent.subagent_ipc as ipc

        events = []
        orch = _ipc_orch(tmp_path, callback=lambda e, d: events.append((e, d)))

        def fake_wait(*a, **k):
            on_poll = k.get("on_poll")
            if on_poll:

                def bad(rr, aid):
                    raise ValueError("bad json")

                monkeypatch.setattr(ipc, "read_heartbeat_state", bad)
                on_poll(2.0, "dev_1")
            return _fake_ipc_result()

        _patch_ipc_mocks(monkeypatch, wait_result=fake_wait)
        _patch_ipc_fs(monkeypatch)
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success"
        assert any(e == "subagent_waiting_ipc" for e, _ in events)

    def test_timeout_abandons_and_errors(self, tmp_path, monkeypatch):
        orch = _ipc_orch(tmp_path)
        _patch_ipc_mocks(monkeypatch, wait_result=lambda *a, **k: None)
        _patch_ipc_fs(monkeypatch)
        abandoned = {}
        orch._abandon_ipc_worker = lambda rr, wid, snaps, task_id=None, grace_s=None: (
            abandoned.setdefault("wid", wid) or False
        )
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "error"
        assert "timed out or was cancelled" in res.final_message
        assert abandoned["wid"] == "dev_1"

    def test_timeout_reusable_worker_returns_to_pool(self, tmp_path, monkeypatch):
        orch = _ipc_orch(tmp_path)
        _patch_ipc_mocks(monkeypatch, wait_result=lambda *a, **k: None)
        _patch_ipc_fs(monkeypatch)
        orch._abandon_ipc_worker = lambda *a, **k: True
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "error"
        assert "dev_1" in orch._reusable_worker_ids

    def test_provider_model_appended_to_launch_command(self, tmp_path, monkeypatch):
        orch = _ipc_orch(tmp_path, auto_spawn_worker=True)
        _patch_ipc_mocks(monkeypatch)
        _patch_ipc_fs(monkeypatch)
        orch._spawn_ipc_worker_background = lambda *a, **k: True
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success"
        cmd = orch._subagent_ipc_commands["dev_1"]
        assert "--provider prov" in cmd and "--model model-x" in cmd
        assert "dev_1" in orch._ipc_worker_ids

    def test_worker_reuse_dispatches_to_worker_dir(self, tmp_path, monkeypatch):
        orch = _ipc_orch(tmp_path)
        writes = []
        import external_llm.agent.subagent_ipc as ipc

        monkeypatch.setattr(ipc, "clear_result", lambda *a, **k: None)
        monkeypatch.setattr(
            ipc,
            "write_task",
            lambda rr, task, worker_id=None: writes.append(worker_id),
        )
        monkeypatch.setattr(ipc, "wait_for_result", lambda *a, **k: _fake_ipc_result())
        _patch_ipc_fs(monkeypatch)
        orch._claim_reusable_worker = lambda rr: "worker-9"
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success"
        assert writes == ["worker-9"]
        assert "worker-9" in orch._ipc_worker_ids

    def test_review_retry_reverts_strays_under_revert_policy(self, tmp_path, monkeypatch):
        events = []
        orch = _ipc_orch(
            tmp_path,
            callback=lambda e, d: events.append(e),
            review_enabled=True,
            review_max_retries=1,
            scope_violation_policy="revert",
        )
        (tmp_path / "a.py").write_text("PRE\n")
        calls = {"n": 0}

        def wait(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                r = _fake_ipc_result()
                r.unassigned_changes = [{"file": "stray.py"}]
                return r
            return None  # retry wait times out

        _patch_ipc_mocks(monkeypatch, wait_result=wait)
        monkeypatch.setattr(orch_mod, "asr_subagent_argv", lambda rr: ["asi", "--subagent"])
        orch._review_subagent_result = lambda **k: (False, "fix it")
        orch._detect_genuine_violations = lambda rr, files, raw_unassigned=None: [{"file": "stray.py"}]
        orch._revert_unassigned_changes = lambda rr, u: ["stray.py"]
        orch._abandon_ipc_worker = lambda *a, **k: False
        orch._cached_git_diff = Mock(return_value="")
        res = orch._run_subagent_ipc(_sub(files=["a.py"]))
        assert res.status == "error"
        assert "review-retry" in res.final_message
        assert "subagent_retry" in events

    def test_cancelled_result_exit_confirmed_not_reused(self, tmp_path, monkeypatch):
        import external_llm.agent.subagent_ipc as ipc

        orch = _ipc_orch(tmp_path)
        _patch_ipc_mocks(monkeypatch, wait_result=lambda *a, **k: _fake_ipc_result(status="cancelled"))
        _patch_ipc_fs(monkeypatch)
        monkeypatch.setattr(ipc, "read_worker_idle_heartbeat_state", lambda rr, wid: "exited")
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "cancelled"
        assert "dev_1" not in orch._reusable_worker_ids

    def test_cancelled_result_idle_read_failure_breaks_grace(self, tmp_path, monkeypatch):
        import external_llm.agent.subagent_ipc as ipc

        orch = _ipc_orch(tmp_path)
        _patch_ipc_mocks(monkeypatch, wait_result=lambda *a, **k: _fake_ipc_result(status="cancelled"))
        _patch_ipc_fs(monkeypatch)

        def bad(rr, wid):
            raise OSError("gone")

        monkeypatch.setattr(ipc, "read_worker_idle_heartbeat_state", bad)
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "cancelled"
        # read failed → break → treated as NOT exited → returned to pool
        assert "dev_1" in orch._reusable_worker_ids

    def test_success_returns_worker_to_pool(self, tmp_path, monkeypatch):
        orch = _ipc_orch(tmp_path)
        _patch_ipc_mocks(monkeypatch)
        _patch_ipc_fs(monkeypatch)
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success"
        assert "dev_1" in orch._reusable_worker_ids


# ══════════════════════════════════════════════════════════════════════
# C. worker launch helpers
# ══════════════════════════════════════════════════════════════════════


class TestWorkerLaunchers:
    def test_terminal_macos_no_command(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._launch_ipc_worker_terminal_macos("dev_1", "dev_1") is False

    def test_terminal_macos_popen_failure(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        orch._subagent_ipc_commands["dev_1"] = "cd /x && asi --subagent"

        def bad_popen(*a, **k):
            raise OSError("no osascript")

        monkeypatch.setattr(orch_mod.subprocess, "Popen", bad_popen)
        assert orch._launch_ipc_worker_terminal_macos("dev_1", "dev_1") is False

    def test_terminal_macos_success_escapes_applescript(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        orch._subagent_ipc_commands["dev_1"] = 'cd "/a b" && echo "hi\\n"'
        opened = {}
        monkeypatch.setattr(
            orch_mod.subprocess,
            "Popen",
            lambda argv, **k: opened.setdefault("argv", argv),
        )
        assert orch._launch_ipc_worker_terminal_macos("dev_1", "dev_1") is True
        osa = opened["argv"][2]
        assert 'do script "' in osa

    def test_spawn_background_registers_proc(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        argv_seen = {}
        proc = MagicMock(pid=4242)

        def fake_popen(argv, **k):
            argv_seen["argv"] = list(argv)
            return proc

        monkeypatch.setattr(orch_mod.subprocess, "Popen", fake_popen)
        assert orch._spawn_ipc_worker_background(str(tmp_path), "w1", "p", "m") is True
        assert orch._ipc_worker_procs["w1"] is proc
        assert argv_seen["argv"][argv_seen["argv"].index("--provider") + 1] == "p"
        assert argv_seen["argv"][argv_seen["argv"].index("--model") + 1] == "m"

    def test_spawn_background_popen_failure(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)

        def bad_popen(argv, **k):
            raise FileNotFoundError("no asi")

        monkeypatch.setattr(orch_mod.subprocess, "Popen", bad_popen)
        assert orch._spawn_ipc_worker_background(str(tmp_path), "w1", "", "") is False

    def test_spawn_background_rotates_oversized_log(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        d = tmp_path / ".asicode" / "subagents" / "w1"
        d.mkdir(parents=True)
        (d / "worker.log").write_bytes(b"x" * 32)
        monkeypatch.setattr(orch_mod, "_WORKER_LOG_ROTATE_BYTES", 10)
        monkeypatch.setattr(
            orch_mod.subprocess,
            "Popen",
            lambda argv, **k: MagicMock(pid=1),
        )
        assert orch._spawn_ipc_worker_background(str(tmp_path), "w1", "", "") is True
        assert (d / "worker.log.old").exists()


# ══════════════════════════════════════════════════════════════════════
# D. _run_subagent in-process edges
# ══════════════════════════════════════════════════════════════════════


class TestRunSubagentInProcess:
    def _patch_loop(self, monkeypatch, results, captured):
        import external_llm.agent.agent_loop as al
        from external_llm.agent.agent_loop_types import AgentResult

        class FakeLoop:
            def __init__(self, **kw):
                pass

            def run(self, task_text):
                captured.append(task_text)
                return (
                    results.pop(0)
                    if results
                    else AgentResult(status="success", final_message="ok", turns=[], applied_patches=[])
                )

        monkeypatch.setattr(al, "AgentLoop", FakeLoop)

    def test_dedicated_client_created_and_goal_wrapped(self, tmp_path, monkeypatch):
        import external_llm.client as client_mod

        orch = _mk_orch(tmp_path, subagent_models={"1": ("ollama", "qwen", "k")})
        made = {}
        monkeypatch.setattr(
            client_mod,
            "create_llm_client",
            lambda **kw: made.setdefault("kw", kw) or Mock(),
        )
        captured = []
        self._patch_loop(monkeypatch, [], captured)
        orch._cached_git_diff = Mock(return_value="")
        res = orch._run_subagent(_sub(), original_request="BUILD THE FEATURE")
        assert res.status == "success"
        assert made["kw"]["provider"] == "ollama"
        assert "[Original request goal]" in captured[0]
        assert "BUILD THE FEATURE" in captured[0]

    def test_dedicated_client_failure_falls_back(self, tmp_path, monkeypatch):
        import external_llm.client as client_mod

        orch = _mk_orch(tmp_path, subagent_models={"1": ("prov", "m", "k")})

        def bad_client(**kw):
            raise RuntimeError("bad key")

        monkeypatch.setattr(client_mod, "create_llm_client", bad_client)
        captured = []
        self._patch_loop(monkeypatch, [], captured)
        orch._cached_git_diff = Mock(return_value="")
        res = orch._run_subagent(_sub())
        assert res.status == "success"  # ran on the orchestrator client

    def test_review_approved_breaks_loop(self, tmp_path, monkeypatch):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path, review_enabled=True, review_max_retries=3)
        captured = []
        self._patch_loop(
            monkeypatch,
            [AgentResult(status="success", final_message="v1", turns=[], applied_patches=[])],
            captured,
        )
        orch._review_subagent_result = lambda **k: (True, "")
        orch._cached_git_diff = Mock(return_value="")
        res = orch._run_subagent(_sub())
        assert res.status == "success"
        assert len(captured) == 1  # no retry

    def test_review_rejected_retries_with_scope_revert(self, tmp_path, monkeypatch):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(
            tmp_path,
            review_enabled=True,
            review_max_retries=1,
            scope_violation_policy="revert",
        )
        captured = []
        self._patch_loop(
            monkeypatch,
            [
                AgentResult(status="success", final_message="v1", turns=[], applied_patches=[]),
                AgentResult(status="success", final_message="v2", turns=[], applied_patches=[]),
            ],
            captured,
        )
        orch._review_subagent_result = lambda **k: (False, "worse")
        orch._detect_genuine_violations = lambda rr, files: [{"file": "s.py"}]
        orch._revert_unassigned_changes = lambda rr, u: ["s.py"]
        orch._cached_git_diff = Mock(return_value="")
        res = orch._run_subagent(_sub(files=["a.py"]))
        assert res.status == "success" and res.final_message == "v2"
        assert len(captured) == 2
        assert "REVIEW FEEDBACK" in captured[1]

    def test_model_context_scope_entered_with_run_store(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path, subagent_models={"1": ("p", "m", "k")})
        import external_llm.client as client_mod

        monkeypatch.setattr(client_mod, "create_llm_client", lambda **kw: Mock())
        rs = Mock()
        ctx = MagicMock()
        rs.model_context_scope.return_value = ctx
        orch._run_store = rs
        captured = []
        self._patch_loop(monkeypatch, [], captured)
        orch._cached_git_diff = Mock(return_value="")
        orch._run_subagent(_sub())
        assert ctx.__enter__.called and ctx.__exit__.called


# ══════════════════════════════════════════════════════════════════════
# E. _run_tool_loop — session inheritance, cancel, status mapping
# ══════════════════════════════════════════════════════════════════════


class TestRunToolLoop:
    def _patch_dcl(self, monkeypatch, respond):
        import external_llm.agent.design_chat_loop as dcl

        instances = []

        class FakeLoop:
            def __init__(self, *a, **k):
                self.msgs = None
                instances.append(self)

            def respond(self, msgs, **k):
                self.msgs = list(msgs)
                return respond

        monkeypatch.setattr(dcl, "DesignChatLoop", FakeLoop)
        return instances

    def _registry(self, tmp_path, flm=None):
        reg = Mock()
        reg.repo_root = str(tmp_path)
        reg.config = SimpleNamespace(file_lock_manager=flm)
        return reg

    def test_bare_request_when_no_session(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        orch._registry_proto = self._registry(tmp_path, flm=None)
        insts = self._patch_dcl(monkeypatch, SimpleNamespace(content="answer", is_error=False))
        res = orch._run_tool_loop("DO WORK")
        assert res.status == "success"  # direct answer, no subagents
        assert res.summary == "answer"
        roles_contents = [(m.role, m.content) for m in insts[0].msgs]
        assert ("user", "DO WORK") in roles_contents
        # file_lock_manager was None on the wrapped config → injected
        assert orch._registry_proto.config.file_lock_manager is orch._file_lock_mgr

    def test_session_context_inherited(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        orch._session_id = "s1"
        sm = Mock()
        sm.get_or_create.return_value = object()
        sm.build_context_messages.return_value = [
            {"role": "system", "content": "──"},  # empty divider → filtered
            {"role": "user", "content": "PRIOR TURN"},
        ]
        orch.orch_config = OrchestratorConfig(session_mgr=sm)
        insts = self._patch_dcl(monkeypatch, SimpleNamespace(content="ok", is_error=False))
        res = orch._run_tool_loop("TASK")
        assert res.status == "success"
        contents = [m.content for m in insts[0].msgs]
        assert "PRIOR TURN" in contents
        assert "TASK" not in contents  # not appended again

    def test_session_context_failure_falls_back(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        orch._session_id = "s1"
        sm = Mock()
        sm.get_or_create.side_effect = RuntimeError("no session dir")
        orch.orch_config = OrchestratorConfig(session_mgr=sm)
        insts = self._patch_dcl(monkeypatch, SimpleNamespace(content="fallback", is_error=False))
        res = orch._run_tool_loop("TASK")
        assert res.status == "success"
        contents = [m.content for m in insts[0].msgs]
        assert "TASK" in contents  # bare-request fallback

    def test_cancelled_returns_partial_synthesis(self, tmp_path, monkeypatch):
        import external_llm.agent.design_chat_loop as dcl
        from external_llm.agent.agent_loop_types import AgentCancelled

        class CancellingLoop:
            def __init__(self, *a, **k):
                pass

            def respond(self, msgs, **k):
                raise AgentCancelled()

        monkeypatch.setattr(dcl, "DesignChatLoop", CancellingLoop)
        orch = _mk_orch(tmp_path)
        res = orch._run_tool_loop("TASK")
        assert res.status == "cancelled"
        assert res.metadata["cancelled"] is True

    def test_partial_when_subagents_all_failed(self, tmp_path, monkeypatch):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path)
        self._patch_dcl(monkeypatch, None)  # dc_result None
        # _run_tool_loop resets _bg_results at entry; results arrive via the
        # drain in the finally block — simulate a failed sub-agent there.
        err = AgentResult(status="error", final_message="x", turns=[], applied_patches=[])
        orch._drain_background_subagents = lambda **k: orch._bg_results.append(err)
        res = orch._run_tool_loop("TASK")
        assert res.status == "partial"

    def test_error_when_nothing_produced(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        self._patch_dcl(monkeypatch, None)
        res = orch._run_tool_loop("TASK")
        assert res.status == "error"


# ══════════════════════════════════════════════════════════════════════
# F. drain / shutdown / bg bookkeeping
# ══════════════════════════════════════════════════════════════════════


class TestDrainAndShutdown:
    def test_drain_no_agents_returns(self, tmp_path):
        orch = _mk_orch(tmp_path)
        orch._drain_background_subagents()  # no-op, no hang

    def test_drain_cancel_event_returns_immediately(self, tmp_path):
        ce = threading.Event()
        ce.set()
        orch = _mk_orch(tmp_path, cancel_event=ce)
        orch._bg_subagents["a"] = {"future": Mock(), "result": None, "status": "running"}
        orch._drain_background_subagents(per_agent_timeout=10)
        orch._bg_subagents["a"]["future"].result.assert_not_called()

    def test_drain_polls_until_resolved(self, tmp_path):
        orch = _mk_orch(tmp_path)
        calls = []

        def check(aid, timeout_s=0.0):
            calls.append(aid)
            return ("success", object())

        orch._check_bg_subagent = check
        orch._bg_subagents["a"] = {"future": Mock(), "result": None, "status": "running"}
        orch._bg_subagents["b"] = {"future": Mock(), "result": None, "status": "running"}
        orch._drain_background_subagents(per_agent_timeout=4.0)
        assert set(calls) == {"a", "b"}

    def test_shutdown_none_executor(self, tmp_path):
        orch = _mk_orch(tmp_path)
        orch._shutdown_bg_executor()  # no-op

    def test_shutdown_handles_error(self, tmp_path):
        orch = _mk_orch(tmp_path)
        ex = Mock()
        ex.shutdown.side_effect = RuntimeError("pool broken")
        orch._bg_executor = ex
        orch._shutdown_bg_executor()
        ex.shutdown.assert_called_once_with(wait=False)
        assert orch._bg_executor is None


class TestCheckBgSubagent:
    def test_unknown_agent(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._check_bg_subagent("ghost") == ("unknown", None)

    def test_cached_result_short_circuit(self, tmp_path):
        orch = _mk_orch(tmp_path)
        sentinel = object()
        orch._bg_subagents["a"] = {"future": Mock(), "result": sentinel, "status": "success"}
        st, res = orch._check_bg_subagent("a")
        assert st == "success" and res is sentinel
        assert orch._bg_results == []  # cached → not re-appended

    def test_timeout_returns_running(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        orch = _mk_orch(tmp_path)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(time.sleep, 0.4)
            orch._bg_subagents["slow"] = {"future": fut, "result": None, "status": "running"}
            st, res = orch._check_bg_subagent("slow", timeout_s=0.05)
            assert st == "running" and res is None
        finally:
            pool.shutdown(wait=True)

    def test_future_exception_wraps_error(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        orch = _mk_orch(tmp_path)
        pool = ThreadPoolExecutor(max_workers=1)
        try:

            def boom():
                raise RuntimeError("kaboom")

            fut = pool.submit(boom)
            orch._bg_subagents["bad"] = {"future": fut, "result": None, "status": "running"}
            st, res = orch._check_bg_subagent("bad", timeout_s=2.0)
            assert st == "error"
            assert "kaboom" in res.final_message
            assert res in orch._bg_results
        finally:
            pool.shutdown(wait=True)


class TestRunSubagentBackground:
    def test_success_registers_and_resolves(self, tmp_path):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path)
        ok = AgentResult(status="success", final_message="fine", turns=[], applied_patches=[])
        orch._run_subagent = Mock(return_value=ok)
        aid = orch._run_subagent_background(_sub(task_id="dev_7"))
        assert aid == "dev_7"
        st, res = orch._check_bg_subagent("dev_7", timeout_s=5.0)
        assert st == "success" and res is ok
        entry = orch._bg_subagents["dev_7"]
        assert entry["subtask"].task_id == "dev_7"

    def test_job_exception_wraps_crash_result(self, tmp_path):
        orch = _mk_orch(tmp_path)
        orch._run_subagent = Mock(side_effect=RuntimeError("worker died"))
        orch._run_subagent_background(_sub(task_id="dev_8"))
        st, res = orch._check_bg_subagent("dev_8", timeout_s=5.0)
        assert st == "error"
        assert "Sub-agent crashed: worker died" in res.final_message


# ══════════════════════════════════════════════════════════════════════
# G. git-backed revert & synthetic-diff helpers (real repos)
# ══════════════════════════════════════════════════════════════════════


class TestRevertUnassignedChanges:
    def test_tracked_batched_checkout(self, tmp_path):
        _git_repo(tmp_path, "other.py")
        (tmp_path / "tracked.py").write_text("EDITED\n")
        (tmp_path / "other.py").write_text("EDITED2\n")
        orch = _mk_orch(tmp_path)
        out = orch._revert_unassigned_changes(
            str(tmp_path),
            [
                {"file": "tracked.py"},
                {"file": "other.py"},
            ],
        )
        assert sorted(out) == ["other.py", "tracked.py"]
        assert (tmp_path / "tracked.py").read_text() == "ORIGINAL\n"
        assert (tmp_path / "other.py").read_text() == "BASE\n"

    def test_untracked_unlinked(self, tmp_path):
        _git_repo(tmp_path)
        (tmp_path / "stray.txt").write_text("junk\n")
        orch = _mk_orch(tmp_path)
        out = orch._revert_unassigned_changes(str(tmp_path), [{"file": "stray.txt"}])
        assert out == ["stray.txt"]
        assert not (tmp_path / "stray.txt").exists()

    def test_infra_and_empty_entries_skipped(self, tmp_path):
        _git_repo(tmp_path)
        orch = _mk_orch(tmp_path)
        assert (
            orch._revert_unassigned_changes(
                str(tmp_path),
                [
                    {"file": ".asicode/subagents/x/task.json"},
                    {"file": ""},
                    None,
                ],
            )
            == []
        )

    def test_directory_entry_rmtree(self, tmp_path):
        _git_repo(tmp_path)
        d = tmp_path / "junkdir"
        d.mkdir()
        (d / "f.txt").write_text("x")
        orch = _mk_orch(tmp_path)
        out = orch._revert_unassigned_changes(str(tmp_path), [{"file": "junkdir"}])
        assert out == ["junkdir"]
        assert not d.exists()

    def test_non_git_repo_falls_back_to_unlink(self, tmp_path):
        # No git at all → batched ls-files fails (rc!=0) → per-file fallback →
        # per-file ls-files fails → unlink path.
        (tmp_path / "plain.txt").write_text("x")
        orch = _mk_orch(tmp_path)
        out = orch._revert_unassigned_changes(str(tmp_path), [{"file": "plain.txt"}])
        assert out == ["plain.txt"]
        assert not (tmp_path / "plain.txt").exists()

    def test_batched_checkout_failure_falls_back_per_file(self, tmp_path, monkeypatch):
        _git_repo(tmp_path, "second.py")
        (tmp_path / "tracked.py").write_text("E1\n")
        (tmp_path / "second.py").write_text("E2\n")
        real_run = real_subprocess.run

        def failing_run(cmd, *a, **k):
            # Fail only the BATCHED checkout (multiple pathspecs after --).
            if cmd[:4] == ["git", "checkout", "HEAD", "--"] and len(cmd) > 5:
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"batch fail")
            return real_run(cmd, *a, **k)

        monkeypatch.setattr(real_subprocess, "run", failing_run)
        orch = _mk_orch(tmp_path)
        out = orch._revert_unassigned_changes(
            str(tmp_path),
            [
                {"file": "tracked.py"},
                {"file": "second.py"},
            ],
        )
        assert sorted(out) == ["second.py", "tracked.py"]
        assert (tmp_path / "tracked.py").read_text() == "ORIGINAL\n"

    def test_lsfiles_raise_takes_fallback(self, tmp_path, monkeypatch):
        _git_repo(tmp_path)
        (tmp_path / "u.txt").write_text("x")
        real_run = real_subprocess.run

        def raising_run(cmd, *a, **k):
            if "ls-files" in cmd and "-z" in cmd:
                raise real_subprocess.TimeoutExpired(cmd, 1)
            return real_run(cmd, *a, **k)

        monkeypatch.setattr(real_subprocess, "run", raising_run)
        orch = _mk_orch(tmp_path)
        out = orch._revert_unassigned_changes(str(tmp_path), [{"file": "u.txt"}])
        assert out == ["u.txt"]
        assert not (tmp_path / "u.txt").exists()


class TestSynthesizeUntrackedDiff:
    def test_untracked_file_gets_synthetic_diff(self, tmp_path):
        _git_repo(tmp_path)
        (tmp_path / "newmod.py").write_text("def a():\n    pass\n")
        out = OrchestratorAgent._synthesize_untracked_diff(str(tmp_path), ["newmod.py"])
        assert "--- /dev/null" in out
        assert "+++ b/newmod.py" in out
        assert "@@ -0,0 +1,2 @@" in out
        assert "+def a():" in out

    def test_char_limit_stops_early_with_marker(self, tmp_path):
        _git_repo(tmp_path)
        (tmp_path / "n1.py").write_text("x = 1\n" * 40)
        (tmp_path / "n2.py").write_text("y = 2\n")
        out = OrchestratorAgent._synthesize_untracked_diff(
            str(tmp_path),
            ["n1.py", "n2.py"],
            char_limit=20,
        )
        assert "further untracked files omitted" in out
        assert "+++ b/n2.py" not in out

    def test_tracked_only_returns_empty(self, tmp_path):
        _git_repo(tmp_path)
        assert (
            OrchestratorAgent._synthesize_untracked_diff(
                str(tmp_path),
                ["tracked.py"],
            )
            == ""
        )

    def test_git_failure_returns_empty(self, tmp_path):
        assert (
            OrchestratorAgent._synthesize_untracked_diff(
                str(tmp_path / "nope"),
                ["a.py"],
            )
            == ""
        )

    def test_directory_assignment_prefix_match(self, tmp_path):
        _git_repo(tmp_path)
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "m.py").write_text("z = 0\n")
        out = OrchestratorAgent._synthesize_untracked_diff(str(tmp_path), ["pkg"])
        assert "+++ b/pkg/m.py" in out


class TestGitHelpers:
    def test_get_git_diff_failure_returns_empty(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._get_git_diff(str(tmp_path / "norepo"), ["a.py"]) == ""

    def test_git_status_changed_paths_edges(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._git_status_changed_paths(None) == []
        assert orch._git_status_changed_paths("") == []
        # non-git dir → CalledProcessError → []
        assert orch._git_status_changed_paths(str(tmp_path)) == []

    def test_patch_files_have_wt_changes_edges(self, tmp_path):
        assert OrchestratorAgent._patch_files_have_wt_changes(None, ["a"]) is False
        assert OrchestratorAgent._patch_files_have_wt_changes(str(tmp_path), []) is False
        # non-git → False
        assert OrchestratorAgent._patch_files_have_wt_changes(str(tmp_path), ["a"]) is False

    def test_snapshot_dirty_path_set(self, tmp_path):
        from external_llm.agent.orchestrator import _snapshot_dirty_path_set

        assert _snapshot_dirty_path_set(str(tmp_path)) == set()
        _git_repo(tmp_path)
        (tmp_path / "dirty.py").write_text("d\n")
        assert "dirty.py" in _snapshot_dirty_path_set(str(tmp_path))


# ══════════════════════════════════════════════════════════════════════
# H. _OrchestratorBackedRegistry facade
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorBackedRegistry:
    def _facade(self, tmp_path):
        orch = _mk_orch(tmp_path)
        base = Mock()
        base.get_tool_schemas.return_value = [
            {"name": "read_file"},
            {"name": "spawn_subagent"},
        ]
        obr = _OrchestratorBackedRegistry(base, orch)
        return obr, base, orch

    def test_setattr_private_goes_to_self_others_to_base(self, tmp_path):
        obr, base, orch = self._facade(tmp_path)
        obr.marker = "x"
        assert base.marker == "x"
        object.__setattr__(obr, "_obr_base", base)  # private branch
        assert obr._obr_orch is orch

    def test_getattr_delegates_to_base(self, tmp_path):
        obr, base, _ = self._facade(tmp_path)
        base.repo_root = "/somewhere"
        assert obr.repo_root == "/somewhere"

    def test_schemas_native_first_with_dedup(self, tmp_path):
        obr, _base, _orch = self._facade(tmp_path)
        schemas = obr.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names.count("spawn_subagent") == 1
        assert "read_file" in names
        assert names[0] in ("spawn_subagent", "poll_subagent", "list_subagents")

    def test_dispatch_native_ok_and_errors(self, tmp_path):
        obr, _base, orch = self._facade(tmp_path)
        orch._dispatch_native_tool = lambda name, args: "NATIVE_OUT"
        tr = obr.dispatch("spawn_subagent", {})
        assert tr.ok is True and tr.content == "NATIVE_OUT"

        def boom(name, args):
            raise orch_mod._NativeToolError("bad id")

        orch._dispatch_native_tool = boom
        tr = obr.dispatch("poll_subagent", {})
        assert tr.ok is False and tr.error == "bad id"

        def crash(name, args):
            raise RuntimeError("unexpected")

        orch._dispatch_native_tool = crash
        tr = obr.dispatch("list_subagents", {})
        assert tr.ok is False and "unexpected" in tr.error

    def test_dispatch_passthrough(self, tmp_path):
        obr, base, _ = self._facade(tmp_path)
        obr.dispatch("read_file", {"path": "x"})
        base.dispatch.assert_called_once_with("read_file", {"path": "x"})


# ══════════════════════════════════════════════════════════════════════
# I. misc agent-level helpers
# ══════════════════════════════════════════════════════════════════════


class TestMiscAgentHelpers:
    def test_gc_removes_stale_keeps_fresh(self, tmp_path, monkeypatch):
        base = tmp_path / ".asicode" / "subagents"
        stale = base / "old"
        stale.mkdir(parents=True)
        fresh = base / "new"
        fresh.mkdir()
        old = time.time() - 10 * 86400
        os.utime(stale, (old, old))
        real_getmtime = os.path.getmtime

        def flaky_mtime(p):
            if str(p).endswith("flaky"):
                raise OSError("no mtime")
            return real_getmtime(p)

        monkeypatch.setattr(os.path, "getmtime", flaky_mtime)
        (base / "flaky").mkdir()
        OrchestratorAgent._gc_subagent_artifacts(str(tmp_path))
        assert not stale.exists()
        assert fresh.exists()
        assert (base / "flaky").exists()

    def test_gc_missing_base_noop(self, tmp_path):
        OrchestratorAgent._gc_subagent_artifacts(str(tmp_path / "nothing"))

    def test_tool_list_subagents(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._tool_list_subagents() == "No sub-agents spawned yet."
        st = _sub(task_id="dev_3")
        orch._bg_subagents["dev_3"] = {
            "subtask": st,
            "status": "running",
            "future": None,
            "result": None,
        }
        orch._bg_subagents["dev_4"] = {
            "subtask": None,
            "status": "done",
            "future": None,
            "result": None,
        }
        out = orch._tool_list_subagents()
        assert "- dev_3 [running]: T" in out
        assert "- dev_4 [done]: " in out

    def test_extract_subagent_summary_patch_shapes(self, tmp_path):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path)
        st = _sub(task_id="dev_1")
        res = AgentResult(
            status="success",
            final_message="did stuff",
            turns=[],
            applied_patches=[
                {"file": "a.py"},
                {"file_path": "b.py"},
                SimpleNamespace(file_path="c.py"),
                "not-a-patch",
                {"file": "a.py"},  # dedup
                {"file": "d.py"},
            ],
        )
        s = orch._extract_subagent_summary(st, res)
        assert "[dev_1: T → completed" in s
        # patches[:5] cap: a/b/c (indices 0-4 after shapes) are in, the 7th
        # entry (d.py) is dropped by the cap.
        assert "a.py" in s and "b.py" in s and "c.py" in s
        assert "d.py" not in s
        assert s.count("a.py") == 1
        # non-list patches → no Files note
        res2 = AgentResult(status="error", final_message="x", turns=[], applied_patches="nope")
        s2 = orch._extract_subagent_summary(st, res2)
        assert "Files:" not in s2 and "failed" in s2


# ══════════════════════════════════════════════════════════════════════
# J. run() entry + decomposition-mode edges (batch 2)
# ══════════════════════════════════════════════════════════════════════


class TestRunEntry:
    def test_run_cancelled_before_decomposition(self, tmp_path):
        ce = threading.Event()
        ce.set()
        orch = _mk_orch(tmp_path, cancel_event=ce)
        res = orch.run("do things")
        assert res.status == "cancelled"
        assert "cancelled" in res.summary

    def test_run_decompose_failure_returns_error(self, tmp_path):
        orch = _mk_orch(tmp_path)
        orch._decompose_task = lambda req: []
        res = orch.run("do things")
        assert res.status == "error"
        assert "decomposition failed" in res.summary.lower()

    def test_run_subagent_delegates_to_ipc_mode(self, tmp_path):
        orch = _mk_orch(tmp_path, subagent_mode="ipc")
        sentinel = object()
        orch._run_subagent_ipc = lambda *a, **k: sentinel
        assert orch._run_subagent(_sub()) is sentinel

    def test_decompose_unparseable_json_returns_empty(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        monkeypatch.setattr(orch_mod, "simple_llm_call", lambda *a, **k: "NOT JSON")
        assert orch._decompose_task("split this") == []


class TestDependencyAware:
    def test_unbreakable_cycle_falls_back_and_forces_execution(self, tmp_path, monkeypatch):
        events = []
        orch = _mk_orch(tmp_path, callback=lambda e, d: events.append((e, d)))
        # Force the "cycle detected" branch, then make breaking fail, so the
        # fallback-to-original-order + no-ready/deadlock/force path all run.
        monkeypatch.setattr(
            orch,
            "_detect_cycles_kahn",
            lambda tm: ([], [["dev_1", "dev_2"]]),
        )
        monkeypatch.setattr(orch, "_break_cycles", lambda subs, cyc: None)
        from external_llm.agent.agent_loop_types import AgentResult

        ok = AgentResult(status="success", final_message="ok", turns=[], applied_patches=[])
        orch._run_subagent = Mock(return_value=ok)
        subs = [
            _sub(task_id="dev_1", files=["a.py"]),
            _sub(task_id="dev_2", files=["b.py"]),
        ]
        subs[0].dependencies = ["dev_2"]
        subs[1].dependencies = ["dev_1"]
        results = orch._run_dependency_aware(subs)
        assert all(r is ok for r in results)
        types = [d.get("type") for e, d in events]
        assert "dependency_cycle_fallback" in types
        assert "execution_deadlock" in types

    def test_parallel_batch_empty(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._run_parallel_batch([], {}, {}) == []


class TestParallelBatchCancelDrain:
    def _mk_with_cancel(self, tmp_path):
        ce = threading.Event()
        ce.set()
        return _mk_orch(tmp_path, cancel_event=ce, max_subagents=1)

    def test_cancel_drain_mixes_raising_and_cancelled(self, tmp_path):
        orch = self._mk_with_cancel(tmp_path)

        def slow_boom(st, *a, **k):
            time.sleep(0.3)
            raise RuntimeError("worker boom")

        orch._run_subagent = slow_boom
        batch = [_sub(task_id="dev_1"), _sub(task_id="dev_2")]
        results = orch._run_parallel_batch(batch, {}, {s.task_id: s for s in batch})
        statuses = sorted(r.status for r in results)
        # t0 raised inside the cancel-drain; t1 was cancelled before start
        # (queued behind max_workers=1) OR ran — both must surface, never None.
        assert None not in results
        assert statuses in (["cancelled", "error"], ["error", "success"])

    def test_cancel_drain_still_pending_gets_cancelled(self, tmp_path):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = self._mk_with_cancel(tmp_path)

        def slow(st, *a, **k):
            time.sleep(3.0)
            return AgentResult(status="success", final_message="late", turns=[], applied_patches=[])

        orch._run_subagent = slow
        batch = [_sub(task_id="dev_1"), _sub(task_id="dev_2")]
        results = orch._run_parallel_batch(batch, {}, {s.task_id: s for s in batch})
        # The queued one is cancelled; the still-running one is reported as
        # cancelled by the still_pending loop within the 2s window.
        assert None not in results
        assert any(r.status == "cancelled" for r in results)

    def test_normal_drain_wraps_cancellederror_future(self, tmp_path):
        import concurrent.futures as cf

        orch = _mk_orch(tmp_path, max_subagents=2)

        def raiser(st, *a, **k):
            raise cf.CancelledError()

        orch._run_subagent = raiser
        batch = [_sub(task_id="dev_1"), _sub(task_id="dev_2")]
        results = orch._run_parallel_batch(batch, {}, {s.task_id: s for s in batch})
        assert all(r.status == "cancelled" for r in results)
        assert all("cancelled" in (r.final_message or "").lower() for r in results)


class TestSynthesizeFromSubtasks:
    def test_all_result_shapes_rendered(self, tmp_path):
        from external_llm.agent.agent_loop_types import AgentResult

        orch = _mk_orch(tmp_path)
        pairs = [
            (_sub(task_id="dev_1"), None),
            (_sub(task_id="dev_2"), AgentResult(status="success", final_message="done", turns=[], applied_patches=[])),
            (
                _sub(task_id="dev_3"),
                AgentResult(status="max_turns", final_message="partial", turns=[], applied_patches=[]),
            ),
            (_sub(task_id="dev_4"), AgentResult(status="error", final_message="bad", turns=[], applied_patches=[])),
        ]
        out = orch._synthesize_from_subtasks(pairs)
        assert "[dev_1] T: no result" in out
        assert "✅ [dev_2]" in out and "⚠️ [dev_3]" in out and "❌ [dev_4]" in out
        assert orch._synthesize_from_subtasks([]) == "Multi-agent task completed."


class TestToolSpawnSubagent:
    def test_missing_description_raises_native_error(self, tmp_path):
        orch = _mk_orch(tmp_path)
        with pytest.raises(orch_mod._NativeToolError):
            orch._tool_spawn_subagent({}, "req")

    def test_string_files_and_bad_priority_coerced(self, tmp_path):
        orch = _mk_orch(tmp_path)
        orch._run_subagent_background = Mock()
        out = orch._tool_spawn_subagent(
            {"task_description": "fix the parser bug properly", "assigned_files": "src/a.py", "priority": "high"},
            "original request",
        )
        assert "src/a.py" in out
        spawned = orch._run_subagent_background.call_args[0][0]
        assert spawned.assigned_files == ["src/a.py"]
        assert spawned.priority == 1
        assert "src/a.py" in orch._global_assigned_paths

    def test_file_conflict_with_running_agent_warned(self, tmp_path):
        events = []
        orch = _mk_orch(tmp_path, callback=lambda e, d: events.append((e, d)))
        orch._run_subagent_background = Mock()
        # A still-RUNNING agent holding src/shared.py …
        orch._bg_subagents["dev_1"] = {
            "subtask": _sub(task_id="dev_1", files=["src/shared.py"]),
            "future": None,
            "result": None,
            "status": "running",
        }
        # … and a FINISHED one (result set) that must be skipped.
        orch._bg_subagents["dev_2"] = {
            "subtask": _sub(task_id="dev_2", files=["src/shared.py"]),
            "future": None,
            "result": object(),
            "status": "success",
        }
        out = orch._tool_spawn_subagent(
            {"task_description": "touch shared", "assigned_files": ["src/shared.py"]},
            "req",
        )
        warns = [d for e, d in events if e == "orchestrator_warning" and d.get("type") == "tool_loop_file_conflict"]
        assert warns and warns[0]["conflicting_agents"] == ["dev_1"]
        assert "File overlap with running sub-agent(s) dev_1" in out

    def test_queued_note_when_future_waiting_for_slot(self, tmp_path):
        orch = _mk_orch(tmp_path)
        orch._run_subagent_background = Mock()
        orch._future_is_queued = staticmethod(lambda entry: True)
        out = orch._tool_spawn_subagent({"task_description": "x"}, "req")
        assert "Queued (waiting for a free worker slot)" in out


class TestFormatPollPatches:
    def test_mixed_shapes_extract_names(self):
        out = OrchestratorAgent._format_poll_patches(
            [
                {"file": "a.py"},
                {"file_path": "b.py"},
                SimpleNamespace(file_path="c.py"),
                "raw diff text",
                {"file": "a.py"},
            ]
        )
        assert "Applied patches (3): a.py, b.py, c.py" in out

    def test_unextractable_falls_back_to_count(self):
        out = OrchestratorAgent._format_poll_patches(["raw1", "raw2", {}])
        assert out == "\nApplied patches: 3"

    def test_cap_shows_more_tail(self):
        patches = [{"file": f"f{i}.py"} for i in range(10)]
        out = OrchestratorAgent._format_poll_patches(patches)
        assert "(+2 more)" in out


class TestCycleHelpers:
    def test_detect_skips_unknown_dependencies(self, tmp_path):
        orch = _mk_orch(tmp_path)
        st = _sub(task_id="dev_1")
        st.dependencies = ["ghost"]
        order, cycles = orch._detect_cycles_kahn({"dev_1": st})
        assert order == ["dev_1"] and cycles == []

    def test_break_cycles_none_when_no_cycles(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._break_cycles([_sub()], []) is None

    def test_break_cycles_ignores_single_node_cycle(self, tmp_path):
        orch = _mk_orch(tmp_path)
        out = orch._break_cycles([_sub(task_id="dev_1")], [["dev_1"]])
        assert out is not None and out[0].task_id == "dev_1"

    def test_break_cycles_unbreakable_returns_none(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        monkeypatch.setattr(
            orch,
            "_detect_cycles_kahn",
            lambda tm: ([], [["dev_1", "dev_2"]]),
        )
        subs = [_sub(task_id="dev_1"), _sub(task_id="dev_2")]
        subs[0].dependencies = ["dev_2"]
        subs[1].dependencies = ["dev_1"]
        assert orch._break_cycles(subs, [["dev_1", "dev_2"]]) is None

    def test_find_current_cycles_reports_cycle(self, tmp_path):
        orch = _mk_orch(tmp_path)
        subs = [_sub(task_id="dev_1"), _sub(task_id="dev_2")]
        subs[0].dependencies = ["dev_2"]
        subs[1].dependencies = ["dev_1"]
        cycles = orch._find_current_cycles({"dev_1", "dev_2"}, {s.task_id: s for s in subs})
        assert cycles and sorted(cycles[0]) == ["dev_1", "dev_2"]

    def test_cb_swallows_handler_exception(self, tmp_path):
        def evil(event, data):
            raise RuntimeError("handler bug")

        orch = _mk_orch(tmp_path, callback=evil)
        orch._cb("any_event", {"x": 1})  # must not raise


class TestClaimAndCleanup:
    def test_cleanup_terminate_failure_logged(self, tmp_path):
        orch = _mk_orch(tmp_path, subagent_mode="ipc")
        proc = MagicMock()
        proc.poll.side_effect = RuntimeError("poll broken")
        orch._ipc_worker_procs["w1"] = proc
        orch._cleanup_ipc_workers()  # must not raise
        assert "w1" not in orch._ipc_worker_procs

    def test_claim_heartbeat_read_failure_still_claims(self, tmp_path, monkeypatch):
        import external_llm.agent.subagent_ipc as ipc

        orch = _mk_orch(tmp_path)
        orch._reusable_worker_ids.add("w1")

        def bad_state(rr, wid):
            raise OSError("heartbeat unreadable")

        monkeypatch.setattr(ipc, "read_worker_idle_heartbeat_state", bad_state)
        assert orch._claim_reusable_worker(str(tmp_path)) == "w1"


class TestAbandonEdges:
    def test_cancel_sentinel_failure_warns_and_fails(self, tmp_path, monkeypatch):
        import external_llm.agent.subagent_ipc as ipc

        orch = _mk_orch(tmp_path, cancel_event=threading.Event())  # soft timeout

        def bad_sentinel(rr, wid):
            raise OSError("cannot write")

        monkeypatch.setattr(ipc, "write_cancel_sentinel", bad_sentinel)
        reusable = orch._abandon_ipc_worker(str(tmp_path), "w1", grace_s=0.3)
        assert reusable is False

    def test_hard_cancel_proc_exits_within_grace_not_terminated(self, tmp_path):
        ce = threading.Event()
        ce.set()
        orch = _mk_orch(tmp_path, cancel_event=ce)
        proc = MagicMock()
        proc.wait.return_value = 0  # exits immediately
        orch._ipc_worker_procs["w1"] = proc
        reusable = orch._abandon_ipc_worker(str(tmp_path), "w1", grace_s=1.0)
        proc.terminate.assert_not_called()
        assert reusable is False


class TestSpawnBackgroundPlatforms:
    def test_win32_creationflags_applied(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        seen = {}
        monkeypatch.setattr(orch_mod.sys, "platform", "win32")

        def fake_popen(argv, **k):
            seen["k"] = k
            return MagicMock(pid=1)

        monkeypatch.setattr(orch_mod.subprocess, "Popen", fake_popen)
        assert orch._spawn_ipc_worker_background(str(tmp_path), "w1", "", "") is True
        assert "creationflags" in seen["k"]
        assert "start_new_session" not in seen["k"]

    def test_log_handle_close_failure_logged(self, tmp_path, monkeypatch):
        orch = _mk_orch(tmp_path)
        import builtins

        real_open = builtins.open

        class FragileFile:
            def __init__(self, *a, **k):
                pass

            def close(self):
                raise OSError("close failed")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fragile_open(p, mode="r", *a, **k):
            if str(p).endswith("worker.log") and "b" in mode:
                return FragileFile()
            return real_open(p, mode, *a, **k)

        monkeypatch.setattr(builtins, "open", fragile_open)
        monkeypatch.setattr(
            orch_mod.subprocess,
            "Popen",
            lambda argv, **k: MagicMock(pid=1),
        )
        assert orch._spawn_ipc_worker_background(str(tmp_path), "w1", "", "") is True


class _RigidResult:
    """An AgentResult stand-in that rejects attribute attachment."""

    def __init__(self, status="success", final_message="m", turns=None, applied_patches=None, error=None):
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "final_message", final_message)
        object.__setattr__(self, "turns", turns or [])
        object.__setattr__(self, "applied_patches", applied_patches or [])

    def __setattr__(self, name, value):
        raise AttributeError("rigid")


class TestAttachFailures:
    def test_diff_verdict_attach_failure_returns_verdict(self, tmp_path):
        orch = _mk_orch(tmp_path)  # parallel=True → attribution possible
        result = _RigidResult(applied_patches=[{"file": "a.py"}])
        verdict = orch._compute_diff_verdict(
            agent_id="dev_1",
            result=result,
            repo_root=str(tmp_path),
            diff_cache={},
        )
        assert isinstance(verdict, str) and verdict

    def test_ipc_unassigned_attach_failure_logged(self, tmp_path, monkeypatch):
        import external_llm.agent.agent_loop as al

        orch = _ipc_orch(tmp_path)
        monkeypatch.setattr(al, "AgentResult", _RigidResult)
        _patch_ipc_mocks(monkeypatch)
        _patch_ipc_fs(monkeypatch)
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success"  # attach failure swallowed

    @pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Terminal.app auto-launch branch is macOS-only (orchestrator.py:2207)",
    )
    def test_ipc_auto_launch_terminal_darwin_branch(self, tmp_path, monkeypatch):
        # Host IS darwin → the Terminal.app branch is taken when enabled.
        orch = _ipc_orch(tmp_path, auto_launch_terminal=True)
        _patch_ipc_mocks(monkeypatch)
        _patch_ipc_fs(monkeypatch)
        launched = []
        orch._launch_ipc_worker_terminal_macos = lambda aid, wid: launched.append(wid) or True
        res = orch._run_subagent_ipc(_sub())
        assert res.status == "success" and launched == ["dev_1"]


class TestLocateSymbol:
    def test_unreadable_file_skipped(self, tmp_path):
        orch = _mk_orch(tmp_path)
        assert orch._locate_symbol("foo", ["missing.py"], str(tmp_path)) == ""

    def test_found_symbol_produces_hint(self, tmp_path):
        orch = _mk_orch(tmp_path)
        (tmp_path / "mod.py").write_text("import os\n\n\ndef foo():\n    return 1\n")
        hint = orch._locate_symbol("foo", ["mod.py"], str(tmp_path))
        assert "Found `foo` in mod.py at line 4" in hint
        assert "Use bash (cat -n)" in hint
