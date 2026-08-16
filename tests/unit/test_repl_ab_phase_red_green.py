"""RED→GREEN coverage for repl_impl.py interactive-surface layer (A+B phase).

Targets the Layer-2/Layer-3 functions identified in the 52% gap analysis:

  Layer 3 (pure mock):  _kick_next_prompt_suggestion, _run_orchestrate_single_shot,
                        _prompt_input, _init_session_state
  Layer 2 (mockable):   _init_repl_engine, run_once, _finalize_pending_design_chat

All tests are source-free (repl_impl.py is not modified) — every branch is
reached through mocks of LLM services / terminal I/O / engine construction.
"""

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import external_llm.agent.orchestrator as orchestrator_mod
import external_llm.agent.tool_registry as tool_registry_mod
import external_llm.design_session as design_session_mod
import external_llm.intelligent_service as intelligent_service_mod
from external_llm.agent import insights_manager
from external_llm.repl import repl_impl

# ── shared fakes ──────────────────────────────────────────────────────────────


class _FakeToolRegistry:
    instances: ClassVar[list] = []

    def __init__(self, repo_root, config):
        self.repo_root = repo_root
        self.config = config
        _FakeToolRegistry.instances.append(self)


class _FakeAgentConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeDSM:
    instances: ClassVar[list] = []

    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.sessions = {}
        _FakeDSM.instances.append(self)

    def get_or_create(self, sid):
        return self.sessions.setdefault(sid, {})


class _FakeSessionMgr:
    def __init__(self):
        self.turns = []

    def add_turn(self, session_id, role, note, model=None, digest=None, tool_results=None):
        self.turns.append({
            "session_id": session_id, "role": role, "note": note,
            "model": model, "digest": digest, "tool_results": tool_results,
        })


def _write_config(repo_root, data):
    cfg = Path(repo_root) / ".asicode" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(data), encoding="utf-8")
    return cfg


# ── _kick_next_prompt_suggestion ──────────────────────────────────────────────


class _SyncThread:
    """Runs the worker synchronously inside start() — no real thread."""

    def __init__(self, target=None, args=(), kwargs=None, **kw):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class _BumpBeforeStartThread(_SyncThread):
    """Bumps the suggestion gen before running — covers the first bail check."""

    def start(self):
        repl_impl._next_suggestion_gen += 1
        super().start()


class _FakeLLM:
    def __init__(self, content="improve the tests", on_chat=None, emit_warning=False):
        self.content = content
        self.on_chat = on_chat
        self.emit_warning = emit_warning
        self.calls = []

    def chat(self, messages=None, **kw):
        self.calls.append((messages or [], kw))
        if self.emit_warning:
            logging.getLogger("external_llm").warning("background noise")
        if self.on_chat:
            self.on_chat()
        return SimpleNamespace(content=self.content)


class TestKickNextPromptSuggestion:
    @pytest.fixture(autouse=True)
    def _auto_state(self):
        saved = dict(repl_impl._auto_continue_state)
        saved_gen = repl_impl._next_suggestion_gen
        yield
        repl_impl._auto_continue_state.update(saved)
        repl_impl._next_suggestion_gen = saved_gen

    def test_valid_delivery(self, monkeypatch):
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        delivered = []
        monkeypatch.setattr(repl_impl, "_deliver_next_suggestion",
                            lambda text, gen: delivered.append((text, gen)))
        llm = _FakeLLM(content="improve the tests", emit_warning=True)
        gen = repl_impl._next_suggestion_gen
        repl_impl._kick_next_prompt_suggestion(
            llm, "model-x", "refactor the parser", "final answer", "work digest")
        assert delivered == [("improve the tests", gen)]
        msgs, kw = llm.calls[0]
        assert kw["model"] == "model-x" and kw["temperature"] == 0.3
        assert msgs[0].role == "system" and msgs[0].content == repl_impl._NEXT_SUGGEST_SYSTEM
        body = msgs[1].content
        assert "[user request]" in body and "[assistant final answer]" in body
        assert "[work log]" in body

    def test_empty_final_message_and_digest(self, monkeypatch):
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        delivered = []
        monkeypatch.setattr(repl_impl, "_deliver_next_suggestion",
                            lambda text, gen: delivered.append((text, gen)))
        llm = _FakeLLM(content="run the linter")
        repl_impl._kick_next_prompt_suggestion(llm, "m", "fix the bug", "", "")
        assert len(delivered) == 1
        body = llm.calls[0][0][1].content
        assert "[assistant final answer]" not in body and "[work log]" not in body

    def test_auto_none_stops_loop(self, monkeypatch):
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        repl_impl._auto_continue_state.update({"on": True, "depth": 2, "cap": 5})
        noted = []
        monkeypatch.setattr(repl_impl, "_notify_above_prompt",
                            lambda text, color: noted.append((text, color)))
        llm = _FakeLLM(content="NONE")
        repl_impl._kick_next_prompt_suggestion(
            llm, "model-x", "refactor the parser", "", "", auto_mode=True)
        assert len(noted) == 1
        assert "stopped" in noted[0][0] and "after 2 auto step(s)" in noted[0][0]
        assert llm.calls[0][0][0].content == repl_impl._AUTO_NEXT_SUGGEST_SYSTEM

    def test_auto_none_with_auto_off_no_notify(self, monkeypatch):
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        repl_impl._auto_continue_state.update({"on": False, "depth": 0, "cap": 5})
        noted = []
        monkeypatch.setattr(repl_impl, "_notify_above_prompt",
                            lambda text, color: noted.append((text, color)))
        repl_impl._kick_next_prompt_suggestion(
            _FakeLLM(content="NONE"), "m", "req", "", "", auto_mode=True)
        assert noted == []

    def test_gen_mismatch_inside_chat_discards(self, monkeypatch):
        # The mid-worker gen check happens BEFORE llm_client.chat is invoked
        # (right after the lazy import). Swap the client module for a proxy that
        # bumps the gen on import — the L2427 "user started a new turn" bail
        # must fire before any LLM call.
        import sys
        import types as _types

        real_client = sys.modules["external_llm.client"]

        class _BumpingProxy(_types.ModuleType):
            def __getattr__(self, name):
                repl_impl._next_suggestion_gen += 1
                return getattr(real_client, name)

        monkeypatch.setitem(sys.modules, "external_llm.client",
                            _BumpingProxy("external_llm.client"))
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        delivered = []
        monkeypatch.setattr(repl_impl, "_deliver_next_suggestion",
                            lambda t, g: delivered.append((t, g)))
        llm = _FakeLLM(content="improve the tests")
        repl_impl._kick_next_prompt_suggestion(llm, "m", "refactor the parser", "f", "d")
        assert llm.calls == [] and delivered == []

    def test_first_check_bail_no_llm_call(self, monkeypatch):
        llm = _FakeLLM()
        monkeypatch.setattr(threading, "Thread", _BumpBeforeStartThread)
        repl_impl._kick_next_prompt_suggestion(llm, "m", "req", "f", "d")
        assert llm.calls == []

    def test_llm_exception_swallowed(self, monkeypatch):
        class _Boom:
            def chat(self, **kw):
                raise RuntimeError("llm down")

        monkeypatch.setattr(threading, "Thread", _SyncThread)
        delivered = []
        monkeypatch.setattr(repl_impl, "_deliver_next_suggestion",
                            lambda t, g: delivered.append(1))
        repl_impl._kick_next_prompt_suggestion(_Boom(), "m", "req", "f", "d")  # must not raise
        assert delivered == []


# ── _prompt_input ─────────────────────────────────────────────────────────────


class TestPromptInput:
    def _collector(self, monkeypatch):
        collected = []
        monkeypatch.setattr(
            repl_impl, "_collect_input",
            lambda prompt, bottom_toolbar=False: collected.append((prompt, bottom_toolbar)) or "in",
        )
        return collected

    def test_fallback_plain_path(self, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        collected = self._collector(monkeypatch)
        out = repl_impl._prompt_input("code", "claude / sonnet - thinking ON")
        assert out == "in"
        assert collected and collected[0][0].endswith("> ") and collected[0][1] is False
        captured = capsys.readouterr().out
        assert "claude / sonnet - thinking ON" in captured
        assert "Code mode" in captured and "/help for commands" in captured

    def test_general_mode_tag(self, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        collected = self._collector(monkeypatch)
        repl_impl._prompt_input("general")
        assert collected[0][0].startswith("[General] ")
        assert "General mode" in capsys.readouterr().out

    def test_orchestrator_mode_tag(self, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        collected = self._collector(monkeypatch)
        repl_impl._prompt_input("orchestrator")
        assert collected[0][0].startswith("[Orchestrator] ")
        assert "Orchestrator mode" in capsys.readouterr().out

    def test_rich_path(self, monkeypatch):
        calls = {"rule": [], "print": []}
        console = SimpleNamespace(
            rule=lambda *a, **k: calls["rule"].append((a, k)),
            print=lambda *a, **k: calls["print"].append((a, k)),
        )
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        collected = self._collector(monkeypatch)
        out = repl_impl._prompt_input("code", "status-line")
        assert out == "in"
        assert len(calls["rule"]) == 1
        assert collected[0][1] is True
        texts = ["".join(str(x) for x in a) for a, _ in calls["print"]]
        assert any("status-line" in t for t in texts)
        assert any("/help for commands" in t for t in texts)

    def test_rich_no_status(self, monkeypatch):
        calls = {"print": []}
        console = SimpleNamespace(
            rule=lambda *a, **k: None,
            print=lambda *a, **k: calls["print"].append((a, k)),
        )
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        self._collector(monkeypatch)
        repl_impl._prompt_input("code", "")
        texts = ["".join(str(x) for x in a) for a, _ in calls["print"]]
        assert all("status-line" not in t for t in texts)


# ── _init_session_state ───────────────────────────────────────────────────────


class TestInitSessionState:
    @staticmethod
    def _svc(with_llm=True):
        if not with_llm:
            return SimpleNamespace()
        return SimpleNamespace(
            llm_service=SimpleNamespace(thinking_mode="keep", reasoning_effort="keep"))

    def test_no_config_defaults(self, tmp_path):
        dc = SimpleNamespace(thinking_mode="x", reasoning_effort="y")
        svc = self._svc()
        state = repl_impl._init_session_state(str(tmp_path), svc, dc)
        assert state["thinking_state"] is None
        assert state["reasoning_effort"] is None
        assert state["current_chat_mode"] == "code"
        assert state["dev_models"] == {}
        assert state["helper_provider_str"] == ""
        assert state["pending_dc"] is None
        assert state["thinking_state_path"] == str(Path(tmp_path) / ".asicode" / "config.json")
        assert dc.thinking_mode is None and dc.reasoning_effort is None
        assert svc.llm_service.thinking_mode is None

    def test_config_values_loaded(self, tmp_path):
        _write_config(tmp_path, {
            "thinking_state": True,
            "reasoning_effort": "high",
            "helper_provider": "openai",
            "helper_model": "gpt-x",
            "chat_mode": "orchestrator",
            "dev_models": {
                "1": ["p1", "m1"], "2": ("p2", "m2", "extra"),
                "bad": "x", "3": ["only-one"],
            },
        })
        dc = SimpleNamespace(thinking_mode=None, reasoning_effort=None)
        svc = self._svc()
        state = repl_impl._init_session_state(str(tmp_path), svc, dc)
        assert state["thinking_state"] is True
        assert state["reasoning_effort"] == "high"
        assert state["helper_provider_str"] == "openai"
        assert state["helper_model_str"] == "gpt-x"
        assert state["current_chat_mode"] == "orchestrator"
        assert state["dev_models"] == {"1": ("p1", "m1"), "2": ("p2", "m2")}
        assert dc.thinking_mode is True and dc.reasoning_effort == "high"
        assert svc.llm_service.thinking_mode is True
        assert svc.llm_service.reasoning_effort == "high"

    def test_malformed_config_defaults(self, tmp_path):
        cfg = Path(tmp_path) / ".asicode" / "config.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{not json", encoding="utf-8")
        state = repl_impl._init_session_state(str(tmp_path), self._svc(), SimpleNamespace())
        assert state["thinking_state"] is None

    def test_non_dict_dev_models_ignored(self, tmp_path):
        _write_config(tmp_path, {"dev_models": "nope"})
        state = repl_impl._init_session_state(str(tmp_path), self._svc(), SimpleNamespace())
        assert state["dev_models"] == {}

    def test_terminal_branch_uses_term_cfg(self, tmp_path, monkeypatch):
        shared = Path(tmp_path) / ".asicode" / "config.json"
        shared.parent.mkdir(parents=True)
        shared.write_text(json.dumps({"thinking_state": False, "reasoning_effort": "max"}),
                          encoding="utf-8")
        term = Path(tmp_path) / ".asicode" / "terminals" / "ttys007.json"
        monkeypatch.setattr(repl_impl, "_terminal_config_path", lambda root: str(term))
        state = repl_impl._init_session_state(str(tmp_path), self._svc(), SimpleNamespace())
        assert state["thinking_state_path"] == str(term)
        assert state["thinking_state"] is False
        assert state["reasoning_effort"] == "max"

    def test_svc_without_llm_service_skips(self, tmp_path):
        dc = SimpleNamespace(thinking_mode=None, reasoning_effort=None)
        state = repl_impl._init_session_state(str(tmp_path), self._svc(with_llm=False), dc)
        assert state["thinking_state"] is None


# ── _init_repl_engine ─────────────────────────────────────────────────────────


class TestInitReplEngine:
    _DEFAULT_SVC = object()

    def _patches(self, monkeypatch, *, svc=_DEFAULT_SVC, nudge=(False, "")):
        if svc is self._DEFAULT_SVC:
            svc = SimpleNamespace(model="test-m", provider="test-p",
                                  llm_service=SimpleNamespace())
        monkeypatch.setattr(
            intelligent_service_mod, "create_intelligent_service_from_env",
            lambda *a, **k: svc)
        monkeypatch.setattr(insights_manager, "compute_stats", lambda root: {"count": 1})
        monkeypatch.setattr(insights_manager, "should_nudge", lambda stats: nudge)
        seen = []
        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: True)
        monkeypatch.setattr(repl_impl, "_print_banner", lambda root: None)
        monkeypatch.setattr(repl_impl, "_print_dep_status", lambda root, **k: None)
        monkeypatch.setattr(repl_impl, "_kick_embedding_model_warmup", lambda: None)
        monkeypatch.setattr(repl_impl, "_print", lambda text, color: seen.append((text, color)))
        monkeypatch.setattr(tool_registry_mod, "ToolRegistry", _FakeToolRegistry)
        monkeypatch.setattr(tool_registry_mod, "AgentConfig", _FakeAgentConfig)
        monkeypatch.setattr(design_session_mod, "DesignSessionManager", _FakeDSM)
        return seen

    @staticmethod
    def _args(**over):
        base = {"provider": "test-p", "model": "test-m", "api_key": None, "no_deps_check": True}
        base.update(over)
        return SimpleNamespace(**base)

    def test_full_init(self, monkeypatch, tmp_path):
        svc = SimpleNamespace(model="test-m", provider="test-p", llm_service=SimpleNamespace())
        self._patches(monkeypatch, svc=svc)
        out = repl_impl._init_repl_engine(self._args(), str(tmp_path))
        assert out is not None
        assert out["svc"] is svc
        assert out["provider_str"] == "test-p" and out["model_str"] == "test-m"
        assert out["session_id"].startswith("cli-")
        assert out["pending_notifications"] == []
        assert str(tmp_path) == repl_impl._REPO_ROOT
        assert repl_impl._prompt_history_path == str(Path(tmp_path) / ".asicode" / "cli_history")
        assert _FakeToolRegistry.instances[-1].repo_root == str(tmp_path)
        assert isinstance(_FakeDSM.instances[-1], _FakeDSM)

    def test_insights_nudge_split(self, monkeypatch, tmp_path):
        seen = self._patches(monkeypatch, nudge=(True, "Tier 2 message /insights archive list"))
        repl_impl._init_repl_engine(self._args(), str(tmp_path))
        texts = [t for t, _ in seen]
        assert "Tier 2 message." in texts
        assert any(t.startswith(" /insights") for t in texts)

    def test_insights_nudge_plain(self, monkeypatch, tmp_path):
        seen = self._patches(monkeypatch, nudge=(True, "just a nudge message"))
        repl_impl._init_repl_engine(self._args(), str(tmp_path))
        assert "just a nudge message" in [t for t, _ in seen]

    def test_insights_nudge_exception_swallowed(self, monkeypatch, tmp_path):
        self._patches(monkeypatch)

        def _boom(root):
            raise RuntimeError("stats fail")

        monkeypatch.setattr(insights_manager, "compute_stats", _boom)
        out = repl_impl._init_repl_engine(self._args(), str(tmp_path))
        assert out is not None

    def test_svc_none_returns_none(self, monkeypatch, tmp_path):
        seen = self._patches(monkeypatch, svc=None)
        assert repl_impl._init_repl_engine(self._args(), str(tmp_path)) is None
        assert any("failed to initialize LLM service." in t for t, _ in seen)

    def test_ptk_missing_install_yes_restarts(self, monkeypatch, tmp_path):
        self._patches(monkeypatch)
        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: False)
        monkeypatch.setattr(repl_impl, "_collect_input", lambda prompt, **k: "y")
        monkeypatch.setattr(repl_impl, "_pip_install", lambda pkgs, **k: True)

        def _restart():
            raise SystemExit(0)

        monkeypatch.setattr(repl_impl, "_restart_cli", _restart)
        with pytest.raises(SystemExit):
            repl_impl._init_repl_engine(self._args(), str(tmp_path))

    def test_ptk_missing_install_no_continues(self, monkeypatch, tmp_path):
        self._patches(monkeypatch)
        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: False)
        monkeypatch.setattr(repl_impl, "_collect_input", lambda prompt, **k: "n")
        out = repl_impl._init_repl_engine(self._args(), str(tmp_path))
        assert out is not None

    def test_ptk_missing_eof_continues(self, monkeypatch, tmp_path):
        self._patches(monkeypatch)
        monkeypatch.setattr(repl_impl, "_load_prompt_toolkit", lambda: False)

        def _eof(prompt, **k):
            raise EOFError

        monkeypatch.setattr(repl_impl, "_collect_input", _eof)
        out = repl_impl._init_repl_engine(self._args(), str(tmp_path))
        assert out is not None


# ── _run_orchestrate_single_shot ──────────────────────────────────────────────


class _FakeOrchAgent:
    instances: ClassVar[list] = []

    def __init__(self, **kw):
        self.kw = kw
        _FakeOrchAgent.instances.append(self)

    def run(self, prompt):
        return f"RESULT:{prompt}"


class _FakeOrchConfig:
    instances: ClassVar[list] = []

    def __init__(self, **kw):
        self.kw = kw
        _FakeOrchConfig.instances.append(self)


class TestRunOrchestrateSingleShot:
    def _setup(self, monkeypatch, *, raises=False):
        if raises:
            monkeypatch.setattr(
                intelligent_service_mod, "create_intelligent_service_from_env",
                lambda *a, **k: None)
        else:
            svc = SimpleNamespace(model="orch-m", llm_service=SimpleNamespace(client="CLIENT"))
            monkeypatch.setattr(
                intelligent_service_mod, "create_intelligent_service_from_env",
                lambda *a, **k: svc)
        monkeypatch.setattr(orchestrator_mod, "OrchestratorAgent", _FakeOrchAgent)
        monkeypatch.setattr(orchestrator_mod, "OrchestratorConfig", _FakeOrchConfig)
        monkeypatch.setattr(tool_registry_mod, "ToolRegistry", _FakeToolRegistry)
        monkeypatch.setattr(tool_registry_mod, "AgentConfig", _FakeAgentConfig)

    def test_builds_and_runs(self, monkeypatch):
        self._setup(monkeypatch)
        args = SimpleNamespace(provider="p", model="m", api_key=None,
                               thinking_mode=True, reasoning_effort="high")
        out = repl_impl._run_orchestrate_single_shot(
            args, "/repo", "do it", lambda e, p=None: None, threading.Event())
        assert out == "RESULT:do it"
        agent = _FakeOrchAgent.instances[-1]
        assert agent.kw["llm_client"] == "CLIENT" and agent.kw["model"] == "orch-m"
        assert agent.kw["callback"] is not None
        cfg = _FakeOrchConfig.instances[-1].kw
        assert cfg["subagent_mode"] == "ipc" and cfg["auto_spawn_worker"] is True
        assert cfg["thinking_mode"] is True and cfg["reasoning_effort"] == "high"

    def test_no_thinking_args_defaults(self, monkeypatch):
        self._setup(monkeypatch)
        args = SimpleNamespace(provider="p", model="m", api_key=None)
        out = repl_impl._run_orchestrate_single_shot(args, "/repo", "p", None, threading.Event())
        assert out == "RESULT:p"
        cfg = _FakeOrchConfig.instances[-1].kw
        assert cfg["thinking_mode"] is None and cfg["reasoning_effort"] is None

    def test_svc_none_raises(self, monkeypatch):
        self._setup(monkeypatch, raises=True)
        args = SimpleNamespace(provider="p", model="m", api_key=None)
        with pytest.raises(RuntimeError, match="orchestration"):
            repl_impl._run_orchestrate_single_shot(args, "/repo", "p", None, threading.Event())


# ── run_once ──────────────────────────────────────────────────────────────────


class TestRunOnce:
    _NO_RESULT = object()

    @staticmethod
    def _mk_result(status="success", final_message="ok", error="",
                   applied_patches=None, turns=0, metadata=None):
        return SimpleNamespace(
            status=status, final_message=final_message, error=error,
            applied_patches=applied_patches or [], turns=turns, metadata=metadata or {})

    @staticmethod
    def _args(**over):
        base = {
            "repo": "", "verbose": False, "json_stream": False, "json": False, "orchestrate": False,
            "provider": "p", "model": "m", "api_key": None, "max_turns": 5,
            "thinking_mode": None, "reasoning_effort": None, "scoped_verification": True,
        }
        base.update(over)
        return SimpleNamespace(**base)

    def _setup(self, monkeypatch, *, result=_NO_RESULT, engine_raises=None,
               orch=None, orch_raises=None):
        box = {"shown": [], "stream": [], "json": [], "err": [], "printed": [], "orch": []}
        monkeypatch.setattr(repl_impl, "_resolve_repo_root", lambda repo: "/tmp/fake-repo")
        monkeypatch.setattr(repl_impl, "_git_baseline", lambda root: "BASE")
        monkeypatch.setattr(
            repl_impl, "_show_result",
            lambda result, elapsed, repo_root=None, baseline=None:
            box["shown"].append((result, repo_root, baseline)))
        monkeypatch.setattr(
            repl_impl, "_json_stream_emit",
            lambda event, payload=None, **kw: box["stream"].append((event, payload)))
        monkeypatch.setattr(
            repl_impl, "_build_json_output",
            lambda result, elapsed: box["json"].append((result, elapsed)))
        monkeypatch.setattr(
            repl_impl, "_json_error_output",
            lambda status, error, duration_ms=0: box["err"].append((status, error, duration_ms)))
        monkeypatch.setattr(repl_impl, "_print",
                            lambda text, color: box["printed"].append(text))
        if orch is not None:
            monkeypatch.setattr(repl_impl, "_run_orchestrate_single_shot", orch)
            monkeypatch.setattr(repl_impl, "_orchestrator_result_to_agent_like", lambda r: r)
        if orch_raises is not None:
            def _boom(*a, **k):
                raise orch_raises

            monkeypatch.setattr(repl_impl, "_run_orchestrate_single_shot", _boom)
        if engine_raises is not None:
            def _raises(*a, **k):
                raise engine_raises

            monkeypatch.setattr(repl_impl, "_run_with_cancel", _raises)
        else:
            monkeypatch.setattr(
                repl_impl, "_run_with_cancel",
                lambda *a, **k: (None if result is self._NO_RESULT else result))
        monkeypatch.setattr(repl_impl, "_build_engine", lambda config: SimpleNamespace())
        return box

    @staticmethod
    def _run(args):
        old = signal.getsignal(signal.SIGINT)
        try:
            return repl_impl.run_once(args, "test prompt")
        finally:
            signal.signal(signal.SIGINT, old)

    def test_success_plain(self, monkeypatch):
        box = self._setup(monkeypatch, result=self._mk_result())
        rc = self._run(self._args())
        assert rc == 0
        assert len(box["shown"]) == 1
        assert box["shown"][0][1] == "/tmp/fake-repo" and box["shown"][0][2] == "BASE"

    def test_already_satisfied_returns_0(self, monkeypatch):
        self._setup(monkeypatch, result=self._mk_result(status="already_satisfied"))
        assert self._run(self._args()) == 0

    def test_clarification_returns_2(self, monkeypatch):
        box = self._setup(monkeypatch, result=self._mk_result(
            status="clarification_needed", final_message="which file?"))
        rc = self._run(self._args())
        assert rc == 2
        assert any("which file?" in t for t in box["printed"])
        assert any("Use --prompt" in t for t in box["printed"])

    def test_cancelled_status_returns_130(self, monkeypatch):
        self._setup(monkeypatch, result=self._mk_result(status="cancelled"))
        assert self._run(self._args()) == 130

    def test_none_result_returns_130(self, monkeypatch):
        box = self._setup(monkeypatch, result=None)
        assert self._run(self._args()) == 130
        assert any("cancelled." in t for t in box["printed"])

    def test_engine_runtime_error_plain(self, monkeypatch):
        box = self._setup(monkeypatch, engine_raises=RuntimeError("no svc"))
        assert self._run(self._args()) == 1
        assert any("error: no svc" in t for t in box["printed"])

    def test_engine_unexpected_error_verbose_traceback(self, monkeypatch):
        box = self._setup(monkeypatch, engine_raises=ValueError("bad"))
        assert self._run(self._args(verbose=True)) == 1
        assert any("unexpected error: bad" in t for t in box["printed"])

    def test_json_stream_success(self, monkeypatch):
        box = self._setup(monkeypatch, result=self._mk_result())
        rc = self._run(self._args(json_stream=True))
        assert rc == 0
        assert box["stream"] and box["stream"][0][0] == "result"
        assert box["stream"][0][1]["status"] == "success"
        assert box["stream"][0][1]["output"] == "ok"

    def test_json_stream_error(self, monkeypatch):
        box = self._setup(monkeypatch, engine_raises=RuntimeError("boom"))
        rc = self._run(self._args(json_stream=True))
        assert rc == 1
        assert box["stream"][0][0] == "error"
        assert box["stream"][0][1]["error"] == "boom"

    def test_json_stream_unexpected_error(self, monkeypatch):
        box = self._setup(monkeypatch, engine_raises=ValueError("bad"))
        assert self._run(self._args(json_stream=True)) == 1
        assert box["stream"][0][0] == "error"
        assert box["stream"][0][1]["status"] == "unexpected_error"

    def test_json_stream_stream_cb_emits_events(self, monkeypatch):
        captured = {}

        def _fake_build(config):
            captured["config"] = config
            return SimpleNamespace()

        box = self._setup(monkeypatch, result=self._mk_result())
        monkeypatch.setattr(repl_impl, "_build_engine", _fake_build)
        assert self._run(self._args(json_stream=True)) == 0
        # The engine wiring passes run_once's NDJSON closure as stream_cb — invoke it.
        cb = captured["config"].stream_cb
        cb("turn", {"status": "running"})
        assert ("turn", {"status": "running"}) in box["stream"]

    def test_sigint_first_cancels_second_exits(self, monkeypatch):
        """Both _sigint_handler branches: 1st SIGINT sets cancel, 2nd force-exits 130."""
        box = self._setup(monkeypatch, result=None)

        def _blocking(loop, request, context, cancel_event, **kw):
            time.sleep(0.4)

        monkeypatch.setattr(repl_impl, "_run_with_cancel", _blocking)

        def _sender():
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.15)
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=_sender, daemon=True).start()
        old = signal.getsignal(signal.SIGINT)
        try:
            with pytest.raises(SystemExit) as ei:
                repl_impl.run_once(self._args(), "test prompt")
            assert ei.value.code == 130
        finally:
            signal.signal(signal.SIGINT, old)
        assert any("cancelling" in t for t in box["printed"])
        assert any("forcing exit" in t for t in box["printed"])

    def test_json_stream_cancelled(self, monkeypatch):
        box = self._setup(monkeypatch, result=None)
        assert self._run(self._args(json_stream=True)) == 130
        assert box["stream"][0][0] == "cancelled"

    def test_json_blob_success(self, monkeypatch):
        box = self._setup(monkeypatch, result=self._mk_result())
        rc = self._run(self._args(json=True))
        assert rc == 0
        assert len(box["json"]) == 1

    def test_json_blob_error(self, monkeypatch):
        box = self._setup(monkeypatch, engine_raises=RuntimeError("boom"))
        assert self._run(self._args(json=True)) == 1
        assert box["err"] and box["err"][0][0] == "error" and box["err"][0][1] == "boom"

    def test_orchestrate_path(self, monkeypatch):
        orch_calls = []

        def _orch(args, repo_root, prompt, cb, cancel):
            orch_calls.append((repo_root, prompt, cb, cancel))
            return self._mk_result(status="success")

        box = self._setup(monkeypatch, orch=_orch)
        rc = self._run(self._args(orchestrate=True))
        assert rc == 0
        assert orch_calls[0][0] == "/tmp/fake-repo" and orch_calls[0][1] == "test prompt"
        assert box["shown"][0][2] == "BASE"

    def test_orchestrate_error(self, monkeypatch):
        box = self._setup(monkeypatch, orch_raises=RuntimeError("orch failed"))
        assert self._run(self._args(orchestrate=True)) == 1
        assert any("orch failed" in t for t in box["printed"])


# ── _finalize_pending_design_chat ─────────────────────────────────────────────


class TestFinalizePendingDesignChat:
    @staticmethod
    def _live_short_thread(delay=0.05):
        t = threading.Thread(target=lambda: time.sleep(delay), daemon=True)
        t.start()
        return t

    def test_fast_path_completed_content(self):
        box = {"result": SimpleNamespace(content="final answer text",
                                         tool_results=[SimpleNamespace()])}
        pending = {"thread": threading.Thread(), "box": box,
                   "design_config": SimpleNamespace(cancel_event=object())}
        mgr = _FakeSessionMgr()
        repl_impl._finalize_pending_design_chat(pending, mgr, "sess-1", "model-x")
        assert mgr.turns[0]["session_id"] == "sess-1" and mgr.turns[0]["model"] == "model-x"
        assert "completed in the background" in mgr.turns[0]["note"]
        assert mgr.turns[0]["tool_results"] is None
        assert pending["design_config"].cancel_event is None

    def test_timeout_note(self, monkeypatch):
        t = threading.Thread(target=lambda: time.sleep(30), daemon=True)
        t.start()
        pending = {"thread": t, "box": {}, "design_config": None}
        clock = {"calls": 0, "base": time.monotonic()}

        def _monotonic():
            clock["calls"] += 1
            return clock["base"] + (61.0 if clock["calls"] >= 2 else 0.0)

        monkeypatch.setattr(repl_impl.time, "monotonic", _monotonic)
        mgr = _FakeSessionMgr()
        repl_impl._finalize_pending_design_chat(pending, mgr, "s", "m")
        assert "interrupted after 60s timeout" in mgr.turns[0]["note"]
        assert mgr.turns[0]["digest"] == ""

    def test_esc_interrupt_preserves_tool_results(self):
        partial = SimpleNamespace(content="partial answer",
                                  tool_results=[SimpleNamespace(), SimpleNamespace()])
        pending = {"thread": threading.Thread(), "box": {"error": SimpleNamespace(partial_result=partial)}}
        mgr = _FakeSessionMgr()
        repl_impl._finalize_pending_design_chat(pending, mgr, "s", "m")
        turn = mgr.turns[0]
        assert "Interrupted during tool loop" in turn["note"]
        assert "Partial response at interruption" in turn["note"]
        assert len(turn["tool_results"]) == 2

    def test_no_result_no_error_plain_interrupt(self):
        pending = {"thread": threading.Thread(), "box": {}}
        mgr = _FakeSessionMgr()
        repl_impl._finalize_pending_design_chat(pending, mgr, "s", "m")
        assert mgr.turns[0]["note"] and mgr.turns[0]["tool_results"] is None

    def test_alive_thread_waits_then_completes(self):
        t = self._live_short_thread(0.05)
        box = {"result": SimpleNamespace(content="done in bg")}
        pending = {"thread": t, "box": box, "design_config": None}
        mgr = _FakeSessionMgr()
        repl_impl._finalize_pending_design_chat(pending, mgr, "s", "m")
        assert "completed in the background" in mgr.turns[0]["note"]
