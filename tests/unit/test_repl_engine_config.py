"""Regression tests for the EngineConfig refactor of ``_build_engine`` (P2-3).

Covers two contracts:
1. ``EngineConfig`` field defaults mirror the old keyword-only parameters
   (svc/route_decision/thinking_mode/reasoning_effort/scoped_verification).
2. ``_build_engine`` wires every ``EngineConfig`` field into the constructed
   AgentConfig/AgentLoop, reuses a passed-in svc/route_decision, and only
   creates fresh ones (LLM service + TaskRouter) when they are absent.
"""

import threading
from types import SimpleNamespace

import pytest

from external_llm.repl import repl_impl
from external_llm.repl.repl_impl import EngineConfig


def _fake_svc(model="m1"):
    return SimpleNamespace(model=model, llm_service=SimpleNamespace(client=object()))


def _config(**overrides):
    kwargs = {
        "repo_root": "/tmp/repo",
        "request_text": "do the thing",
        "provider": "openai",
        "model": "gpt-x",
        "api_key": "k",
        "max_turns": 7,
        "stream_cb": lambda *a, **kw: None,
        "cancel_event": threading.Event(),
    }
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


class TestEngineConfigDefaults:
    def test_optional_fields_default_to_old_keyword_parameter_values(self):
        cfg = _config()
        assert cfg.svc is None
        assert cfg.route_decision is None
        assert cfg.thinking_mode is None
        assert cfg.reasoning_effort is None
        assert cfg.scoped_verification is True

    def test_required_fields_mirror_old_positional_parameters(self):
        evt = threading.Event()
        cb = lambda *a, **kw: None  # noqa: E731 — no-op stream callback stub
        cfg = EngineConfig("/r", "q", "p", "m", "k", 3, cb, evt)
        assert (cfg.repo_root, cfg.request_text, cfg.provider, cfg.model) == ("/r", "q", "p", "m")
        assert (cfg.api_key, cfg.max_turns, cfg.stream_cb, cfg.cancel_event) == ("k", 3, cb, evt)


class TestBuildEngineWiring:
    """_build_engine must translate EngineConfig -> AgentConfig/AgentLoop."""

    def _patch_engine_parts(self, monkeypatch, captured):
        import external_llm.agent.agent_loop as agent_loop_mod
        import external_llm.agent.tool_registry as tool_registry_mod
        import external_llm.intelligent_service as intelligent_service_mod

        def _capture_registry(root, agent_config):
            captured["root"] = root
            captured["agent_config"] = agent_config
            return object()

        def _capture_loop(**kw):
            captured["loop_kwargs"] = kw
            return object()

        monkeypatch.setattr(tool_registry_mod, "ToolRegistry", _capture_registry)
        monkeypatch.setattr(agent_loop_mod, "AgentLoop", _capture_loop)
        monkeypatch.setattr(
            intelligent_service_mod,
            "create_intelligent_service_from_env",
            lambda *a, **kw: pytest.fail("svc must not be recreated when EngineConfig.svc is set"),
        )

    def test_reuses_svc_and_route_decision_and_wires_all_fields(self, monkeypatch):
        captured = {}
        self._patch_engine_parts(monkeypatch, captured)

        svc = _fake_svc()
        rd = SimpleNamespace(intent_result=object())
        evt = threading.Event()
        cb = lambda *a, **kw: None  # noqa: E731 — no-op stream callback stub
        cfg = _config(
            svc=svc,
            route_decision=rd,
            cancel_event=evt,
            stream_cb=cb,
            thinking_mode=True,
            reasoning_effort="high",
            scoped_verification=False,
        )

        loop = repl_impl._build_engine(cfg)

        # svc/route_decision reuse: no fresh service was created (pytest.fail spy),
        # and the same route_decision object flowed into AgentConfig.
        ac = captured["agent_config"]
        assert captured["root"] == "/tmp/repo"
        assert ac.route_decision is rd
        assert ac.intent_result is rd.intent_result
        assert ac.max_turns == 7
        assert ac.stream_callback is cb
        assert ac.cancel_event is evt
        assert ac.scoped_verification is False
        assert ac.thinking_mode is True
        assert ac.reasoning_effort == "high"
        # Loop receives the reused svc's client/model.
        assert captured["loop_kwargs"]["llm_client"] is svc.llm_service.client
        assert captured["loop_kwargs"]["model"] == "m1"
        assert loop is not None

    def test_creates_svc_and_routes_when_absent(self, monkeypatch):
        import external_llm.agent.task_router as task_router_mod

        captured = {}
        self._patch_engine_parts(monkeypatch, captured)

        svc = _fake_svc()
        created = {}
        monkeypatch.setattr(
            task_router_mod,
            "TaskRouter",
            lambda **kw: SimpleNamespace(route=lambda request_text, repo_root=None: rd),
        )

        def _create(provider, model, api_key):
            created["args"] = (provider, model, api_key)
            return svc

        import external_llm.intelligent_service as intelligent_service_mod

        monkeypatch.setattr(intelligent_service_mod, "create_intelligent_service_from_env", _create)
        rd = SimpleNamespace(intent_result=object())

        repl_impl._build_engine(_config(svc=None, route_decision=None))

        assert created["args"] == ("openai", "gpt-x", "k")
        ac = captured["agent_config"]
        assert ac.route_decision is rd
        assert ac.intent_result is rd.intent_result

    def test_missing_service_raises_with_provider_model_in_message(self, monkeypatch):
        import external_llm.agent.agent_loop as agent_loop_mod
        import external_llm.agent.tool_registry as tool_registry_mod
        import external_llm.intelligent_service as intelligent_service_mod

        monkeypatch.setattr(intelligent_service_mod, "create_intelligent_service_from_env", lambda *a, **kw: None)
        monkeypatch.setattr(tool_registry_mod, "ToolRegistry", lambda *a, **kw: None)
        monkeypatch.setattr(agent_loop_mod, "AgentLoop", lambda **kw: None)

        with pytest.raises(RuntimeError, match=r"--provider openai.*--model gpt-x"):
            repl_impl._build_engine(_config())
