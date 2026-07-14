"""Regression guard: run() must handle the MAIN_AGENT lane.

Background: commit 333751c3 deleted run()'s MAIN_AGENT branch, the fallthrough
``_run_llm_loop`` call, and the ``return result`` statement. Because task_router
always routes to MAIN_AGENT (PLANNER permanently disabled), run() silently
returned ``None`` for every request, breaking intelligent_service, orchestrator,
and local_assistant callers. This went undetected because no test exercised
run() end-to-end. These tests prevent that class of regression.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.agent_loop_types import AgentResult
from external_llm.agent.task_router import Lane
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def _run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def _make_loop(tmp_path) -> AgentLoop:
    repo = Path(tmp_path)
    _run(["git", "init", "-q"], cwd=str(repo))
    _run(["git", "config", "user.email", "t@t.com"], cwd=str(repo))
    _run(["git", "config", "user.name", "t"], cwd=str(repo))
    (repo / "f.txt").write_text("alpha=1\n")
    _run(["git", "add", "f.txt"], cwd=str(repo))
    _run(["git", "commit", "-qm", "base"], cwd=str(repo))
    client = Mock()
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"
    cfg = AgentConfig(max_turns=1, planning_enabled=False, rag_enabled=False)
    reg = ToolRegistry(str(repo), cfg)
    return AgentLoop(llm_client=client, registry=reg, config=cfg, model="test")


def test_run_main_agent_invokes_llm_loop(tmp_path):
    """run() must route MAIN_AGENT to _run_llm_loop and return a real AgentResult."""
    loop = _make_loop(tmp_path)
    loop.config.route_decision = SimpleNamespace(
        lane=Lane.MAIN_AGENT, confidence=0.9, task_kind="general",
        reasoning="", complexity=None, target_specificity_score=0.5,
    )
    called = []
    loop._run_llm_loop = lambda ctx: (
        called.append(ctx),
        AgentResult(status="success", final_message="ok", turns=[],
                    applied_patches=[], metadata={}),
    )[-1]

    result = loop.run("change alpha to 2")

    # The core regression assertion: run() returns a real result, not None.
    assert isinstance(result, AgentResult), f"REGRESSION: run() returned {result!r}"
    assert result.status == "success"
    assert len(called) == 1, f"_run_llm_loop called {len(called)} times"
    ctx = called[0]
    assert ctx.request == "change alpha to 2"
    assert ctx.route.lane == Lane.MAIN_AGENT


def test_run_route_none_returns_fallback_result(tmp_path):
    """run() must never return None — unknown routes yield a partial_success result."""
    loop = _make_loop(tmp_path)
    loop.config.route_decision = None
    result = loop.run("test request")
    assert isinstance(result, AgentResult)
    assert result.metadata.get("unhandled_lane") is True
