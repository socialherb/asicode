"""RED→GREEN: IntelligentLLMService multi-file + agent paths + env factory.

Covers _handle_multi_file (LLM/rule-based planner, force_* overrides, per-op
retry/fallback, operation-result metadata), _handle_agent_mode (AgentLoop
delegation, stream callback, exception mapping), _adapt_agent_result (success /
max_turns / error statuses), _build_agent_context, and
create_intelligent_service_from_env (env resolution, provider validation).
"""

from __future__ import annotations

from pathlib import Path

import external_llm.intelligent_service as isi_mod
from external_llm.agent.agent_loop_types import AgentResult, AgentTurn
from external_llm.agent.tool_registry import ToolResult
from external_llm.intelligent_service import IntelligentLLMService
from external_llm.multi_planner import ExecutionPlan, FileOperation
from external_llm.output_modes import OutputMode
from external_llm.project_analyzer import ProjectStructure
from external_llm.smart_analyzer import RequestAnalysis

# ── shared fakes ─────────────────────────────────────────────────────────────


class _FakeLLM:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.client = object()
        self.model = "fake-model"

    def generate_patch(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return {"success": True, "patch": "--- a/x\n+++ b/x\n"}


def _make_service(llm=None, provider="mock") -> IntelligentLLMService:
    svc = object.__new__(IntelligentLLMService)
    svc.llm_service = llm or _FakeLLM()
    svc.provider = provider
    svc.model = getattr(svc.llm_service, "model", "mock-model")
    return svc


def _analysis(**kw) -> RequestAnalysis:
    defaults = {
        "original_request": "req",
        "intent": "create_feature",
        "feature_name": "login",
        "suggested_files": ["a.py"],
        "tech_stack": ["flask"],
        "confidence": 0.9,
        "needs_planning": True,
    }
    defaults.update(kw)
    return RequestAnalysis(**defaults)


def _structure(**kw) -> ProjectStructure:
    defaults = {
        "framework": "Flask",
        "frameworks": ["Flask"],
        "project_types": ["web"],
        "directories": {"app": ["app/"]},
        "naming_style": "snake_case",
        "common_imports": ["os"],
        "entry_points": ["main.py"],
        "test_dir": "tests",
        "example_files": {"route": "app/routes.py"},
    }
    defaults.update(kw)
    return ProjectStructure(**defaults)


def _plan(*ops: FileOperation, complexity: str = "moderate") -> ExecutionPlan:
    return ExecutionPlan(
        original_request="build login",
        operations=list(ops),
        strategy="sequential",
        complexity=complexity,
        success_criteria=["tests pass"],
    )


def _op(file_path, operation="create", description="d", instructions="i") -> FileOperation:
    return FileOperation(file_path=file_path, operation=operation, description=description, instructions=instructions)


# ── _handle_multi_file: planner selection ───────────────────────────────────


def test_multi_file_llm_planner_used_when_client_present(tmp_path, monkeypatch):
    svc = _make_service()
    captured = {}

    class _FakeLLMPlanner:
        def __init__(self, **kw):
            captured.update(kw)

        def create_plan(self, req):
            return _plan(_op("a.py"))

    monkeypatch.setattr(isi_mod, "LLMEnhancedMultiFilePlanner", _FakeLLMPlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=True)
    assert result["success"] is True
    assert captured["llm_client"] is svc.llm_service.client
    assert captured["llm_model"] == "fake-model"
    assert captured["temperature"] == 0.0


def test_multi_file_rule_planner_without_client(tmp_path, monkeypatch):
    llm = _FakeLLM()
    del llm.client  # rule-based branch: no client attr
    svc = _make_service(llm)
    captured = {}

    class _FakePlanner:
        def __init__(self, repo_root):
            captured["root"] = repo_root

        def create_plan(self, req):
            return _plan(_op("a.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=False)
    assert result["success"] is True
    assert str(tmp_path) in captured["root"]


def test_multi_file_force_overrides_logged_and_applied(tmp_path, monkeypatch):
    svc = _make_service(provider="google")
    llm = svc.llm_service

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("a.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        force_output_mode=OutputMode.PLAN_JSON,
        force_context_variant="super",
    )
    assert result["success"] is True
    assert llm.calls[0]["context_variant"] == "super"
    assert llm.calls[0]["output_mode"] == "auto"  # PLAN_JSON → auto


def test_multi_file_force_files_filters_operations(tmp_path, monkeypatch):
    svc = _make_service()
    llm = svc.llm_service

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("keep.py"), _op("drop.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        force_files=["/keep.py"],
    )
    assert result["success"] is True
    assert [c["target_file"] for c in llm.calls] == ["keep.py"]


def test_multi_file_force_files_no_match_runs_full_plan(tmp_path, monkeypatch):
    svc = _make_service()
    llm = svc.llm_service

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("keep.py"), _op("drop.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        force_files=["z.py", ""],
    )
    assert result["success"] is True
    assert [c["target_file"] for c in llm.calls] == ["keep.py", "drop.py"]


def test_multi_file_force_files_filter_exception_falls_back(tmp_path, monkeypatch):
    svc = _make_service()
    llm = svc.llm_service

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("a.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    # force_files=[5] → (5 or "").strip() raises AttributeError inside the
    # try → caught by the except → warning logged → full plan runs.
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        force_files=[5],
    )
    assert result["success"] is True
    assert [c["target_file"] for c in llm.calls] == ["a.py"]


# ── _handle_multi_file: operation execution ─────────────────────────────────


def test_multi_file_create_success_and_metadata(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "P1"}, {"success": True, "patch": "P2"})
    svc = _make_service(llm)

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("a.py"), _op("b.py", operation="modify"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=False)
    assert result["success"] is True
    assert result["mode"] == "multi_file"
    assert "P1" in result["patch"] and "P2" in result["patch"]
    assert len(result["operations"]) == 2
    first = result["operations"][0]
    assert first["file"] == "a.py"
    assert first["success"] is True
    assert first["output_mode_used"] == "full_file"  # create op → FULL_FILE
    assert first["retry_count"] == 1
    assert result["plan"]["complexity"] == "moderate"
    assert "✅ Successful: 2" in result["explanation"]


def test_multi_file_create_failure_falls_back_to_placeholder(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": False, "error": "invalid_diff: x"})
    svc = _make_service(llm)

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("new.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=False)
    assert result["success"] is True  # fallback placeholder counts as success
    assert result["operations"][0]["fallback_used"] is True
    assert (tmp_path / "new.py").exists()
    assert "✅ Successful: 1" in result["explanation"]


def test_multi_file_modify_failure_and_all_success_false(tmp_path, monkeypatch):
    llm = _FakeLLM(
        {"success": False, "error": "empty_patch: none"},
        {"success": False, "error": "empty_patch: none again"},
    )
    svc = _make_service(llm)
    target = tmp_path / "app.py"
    target.write_text("KEEP\n", encoding="utf-8")

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("app.py", operation="modify"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=False)
    assert result["success"] is False
    op = result["operations"][0]
    assert op["success"] is False
    assert op["same_failure_repeat"] is True
    assert op["error_feedback_included"] is True
    assert target.read_text(encoding="utf-8") == "KEEP\n"
    assert "❌ Failed: 1" in result["explanation"]
    assert "❌ app.py (modify)" in result["explanation"]


def test_multi_file_delete_operation(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "D"})
    svc = _make_service(llm)

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("gone.py", operation="delete"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=False)
    assert result["success"] is True
    assert result["operations"][0]["operation"] == "delete"


def test_multi_file_force_diff_mode_retry_chain(tmp_path, monkeypatch):
    llm = _FakeLLM(
        {"success": False, "error": "git_apply_check_failed: ctx"},
        {"success": True, "patch": "OK"},
    )
    svc = _make_service(llm)

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("app.py", operation="modify"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        force_output_mode=OutputMode.UNIFIED_DIFF,
    )
    assert result["success"] is True
    assert result["operations"][0]["output_mode_used"] == "full_file"
    assert len(llm.calls) == 2


def test_multi_file_force_asicode_mode_uses_auto_string(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "A"})
    svc = _make_service(llm)

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("a.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        force_output_mode=OutputMode.ASICODE_BLOCK,
    )
    assert result["success"] is True
    assert llm.calls[0]["output_mode"] == "auto"


def test_multi_file_progress_callback_direct_call(tmp_path, monkeypatch):
    svc = _make_service()
    llm = svc.llm_service

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("a.py"), _op("b.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    events = []
    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        progress_callback=lambda *a: events.append(a),
    )
    assert result["success"] is True
    exec_events = [e for e in events if e[0] == "executing_operation"]
    assert len(exec_events) == 2
    assert events[0][0] == "planning"
    assert llm.calls


def test_multi_file_create_op_on_existing_file_fallback_empty(tmp_path, monkeypatch):
    """create op whose target ALREADY exists → default-file guard returns ""
    → fallback_generation_failed (success False, no clobber)."""
    llm = _FakeLLM({"success": False, "error": "invalid_diff: x"})
    svc = _make_service(llm)
    target = tmp_path / "existing.py"
    target.write_text("KEEP\n", encoding="utf-8")

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("existing.py", operation="create"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc._handle_multi_file(tmp_path, "req", _analysis(), _structure(), 0.0, llm_planning=False)
    assert result["success"] is False
    op = result["operations"][0]
    assert op["success"] is False
    assert op["fallback_used"] is False
    assert op["fallback_reason"] == "fallback_generation_failed"
    assert target.read_text(encoding="utf-8") == "KEEP\n"


# ── _build_project_context_summary + _build_enhanced_context are in helpers file ──


# ── agent mode ───────────────────────────────────────────────────────────────


class _FakeToolRegistry:
    def __init__(self, repo_root, config):
        self.config = config


class _FakeAgentLoop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = kwargs["config"]

    def run(self, user_request, context):
        self.user_request = user_request
        self.context = context
        # Exercise the stream callback once with valid data.
        if self.config.stream_callback:
            self.config.stream_callback("agent_tool_call", {"turn": 1, "tool": "bash", "result": {"ok": True}})
        return self._result

    def _result(self):
        return AgentResult(status="success", final_message="done", applied_patches=["P1"])


def _turn(ok=True) -> AgentTurn:
    return AgentTurn(turn_num=1, tool_name="bash", tool_args={}, tool_result=ToolResult(ok=ok, content="out"))


def test_handle_agent_mode_success(monkeypatch):
    monkeypatch.setattr("external_llm.agent.agent_loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("external_llm.agent.tool_registry.ToolRegistry", _FakeToolRegistry)
    svc = _make_service()
    svc._agent_max_turns = 30
    events = []

    class _Loop:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]

        def run(self, user_request, context):
            self.config.stream_callback("agent_tool_call", {"turn": 2, "tool": "apply_patch", "result": {"ok": False}})
            return AgentResult(
                status="success",
                final_message="all done",
                applied_patches=["P1", "P2"],
                turns=[_turn()],
                metadata={"extra": 1},
            )

    monkeypatch.setattr("external_llm.agent.agent_loop.AgentLoop", _Loop)
    result = svc._handle_agent_mode(
        Path("/tmp"),
        "make it",
        _analysis(),
        _structure(),
        progress_callback=lambda *a: events.append(a),
    )
    assert result["success"] is True
    assert result["mode"] == "agent"
    assert result["patch"] == "P1\nP2"
    assert result["agent"]["turns_used"] == 1
    assert result["agent"]["max_turns"] == 30
    assert result["agent"]["extra"] == 1
    assert result["analysis"]["intent"] == "create_feature"
    # stream callback events: agent_start + tool call
    assert events[0][0] == "agent_start"
    assert any(e[0] == "agent_tool_call" for e in events)


def test_handle_agent_mode_stream_callback_errors_swallowed(monkeypatch):
    svc = _make_service()
    svc._agent_max_turns = 30
    events = []

    class _Loop:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]

        def run(self, user_request, context):
            cb = self.config.stream_callback
            cb("agent_tool_call", None)  # data None → AttributeError inside cb
            return AgentResult(status="error", error="boom", turns=[])

    monkeypatch.setattr("external_llm.agent.agent_loop.AgentLoop", _Loop)
    monkeypatch.setattr("external_llm.agent.tool_registry.ToolRegistry", _FakeToolRegistry)
    result = svc._handle_agent_mode(
        Path("/tmp"),
        "r",
        _analysis(),
        _structure(),
        progress_callback=lambda *a: events.append(a),
    )
    assert result["success"] is False
    assert result["agent"]["status"] == "error"
    assert result["error"] == "boom"
    # The broken tool-call event raises inside the cb BEFORE progress_callback;
    # it must be swallowed (debug-log only) and never crash the agent run.
    assert [e[0] for e in events] == ["agent_start"]


def test_handle_agent_mode_loop_exception_maps_to_error(monkeypatch):
    class _BoomLoop:
        def __init__(self, **kwargs):
            pass

        def run(self, user_request, context):
            raise RuntimeError("loop died")

    monkeypatch.setattr("external_llm.agent.agent_loop.AgentLoop", _BoomLoop)
    monkeypatch.setattr("external_llm.agent.tool_registry.ToolRegistry", _FakeToolRegistry)
    svc = _make_service()
    svc._agent_max_turns = 30
    result = svc._handle_agent_mode(Path("/tmp"), "r", _analysis(), _structure())
    assert result["success"] is False
    assert result["mode"] == "agent"
    assert result["agent"]["status"] == "error"
    assert "agent_loop_error" in result["error"]


def test_handle_agent_mode_no_progress_callback():
    svc = _make_service()

    class _Loop:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]

        def run(self, user_request, context):
            # no progress_callback → _stream_cb early-return branch
            self.config.stream_callback("agent_tool_call", {"turn": 1, "tool": "bash", "result": {"ok": True}})
            return AgentResult(status="success", final_message="ok", applied_patches=[])

    pytest = __import__("pytest")
    mp = pytest.MonkeyPatch()
    mp.setattr("external_llm.agent.agent_loop.AgentLoop", _Loop)
    mp.setattr("external_llm.agent.tool_registry.ToolRegistry", _FakeToolRegistry)
    try:
        result = svc._handle_agent_mode(Path("/tmp"), "r", _analysis(), _structure(), progress_callback=None)
        assert result["success"] is True
    finally:
        mp.undo()


def test_handle_request_agent_mode_with_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("external_llm.agent.agent_loop.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("external_llm.agent.tool_registry.ToolRegistry", _FakeToolRegistry)
    svc = _make_service()

    class _Ana:
        def analyze(self, req):
            return _analysis()

    class _Proj:
        def analyze(self):
            return _structure()

    monkeypatch.setattr(isi_mod, "SmartRequestAnalyzer", lambda root: _Ana())
    monkeypatch.setattr(isi_mod, "ProjectAnalyzer", lambda root: _Proj())

    class _Loop:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]

        def run(self, user_request, context):
            assert "Repository root:" in context
            return AgentResult(status="success", final_message="ok", applied_patches=["X"])

    monkeypatch.setattr("external_llm.agent.agent_loop.AgentLoop", _Loop)
    result = svc.handle_request(
        repo_root=str(tmp_path),
        user_request="r",
        mode="agent",
        system_prompt="ignored-in-agent",
    )
    assert result["success"] is True
    assert result["mode"] == "agent"


# ── _adapt_agent_result ─────────────────────────────────────────────────────


def test_adapt_agent_result_success():
    svc = _make_service()
    svc._agent_max_turns = 20
    res = AgentResult(
        status="success",
        final_message="done",
        applied_patches=["A", "B"],
        turns=[_turn(True), _turn(False)],
        metadata={"k": "v"},
    )
    out = svc._adapt_agent_result(res, _analysis())
    assert out["success"] is True
    assert out["patch"] == "A\nB"
    assert out["agent"]["turn_summary"][0]["ok"] is True
    assert out["agent"]["turn_summary"][1]["ok"] is False
    assert out["agent"]["k"] == "v"


def test_adapt_agent_result_max_turns_with_patches():
    svc = _make_service()
    svc._agent_max_turns = 20
    res = AgentResult(status="max_turns", applied_patches=["A"], turns=[_turn()])
    out = svc._adapt_agent_result(res, _analysis())
    assert out["success"] is True
    assert out["error"] == "agent_max_turns"


def test_adapt_agent_result_max_turns_without_patches():
    svc = _make_service()
    svc._agent_max_turns = 20
    res = AgentResult(status="max_turns", applied_patches=[], turns=[])
    out = svc._adapt_agent_result(res, _analysis())
    assert out["success"] is False
    assert "reached max turns" in out["explanation"]


def test_adapt_agent_result_error_status():
    svc = _make_service()
    svc._agent_max_turns = 20
    res = AgentResult(status="error", error="exploded", turns=[])
    out = svc._adapt_agent_result(res, _analysis())
    assert out["success"] is False
    assert out["error"] == "exploded"


def test_adapt_agent_result_empty_final_message():
    svc = _make_service()
    svc._agent_max_turns = 20
    res = AgentResult(status="success", applied_patches=[], turns=[])
    out = svc._adapt_agent_result(res, _analysis())
    assert out["success"] is False  # no patches AND no final message
    assert "Agent finished with status: success" in out["explanation"]


# ── _build_agent_context ────────────────────────────────────────────────────


def test_build_agent_context_all_fields():
    svc = _make_service()
    ctx = svc._build_agent_context(Path("/repo"), _analysis(feature_name="login", tech_stack=["flask"]), _structure())
    assert "Frameworks: Flask" in ctx
    assert "Project types: web" in ctx
    assert "Entry points: main.py" in ctx
    assert "Test directory: tests" in ctx
    assert "Request intent: create_feature" in ctx
    assert "Feature: login" in ctx
    assert "Suggested files: a.py" in ctx
    assert "Tech stack: flask" in ctx
    assert "Repository root: /repo" in ctx


def test_build_agent_context_framework_fallback():
    svc = _make_service()
    ctx = svc._build_agent_context(
        Path("/repo"),
        _analysis(feature_name=None, tech_stack=[]),
        _structure(frameworks=[], framework="Django", entry_points=[], test_dir=None, project_types=[]),
    )
    assert "Framework: Django" in ctx
    assert "Project types:" not in ctx
    assert "Request intent: create_feature" in ctx


# ── create_intelligent_service_from_env ─────────────────────────────────────


class _FakeService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model") or "env-model"

    def raise_me(self):
        raise RuntimeError("ctor failed")


def test_env_factory_no_provider(monkeypatch):
    monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
    assert isi_mod.create_intelligent_service_from_env() is None


def test_env_factory_external_prefix_and_key(monkeypatch):
    monkeypatch.setattr(isi_mod, "IntelligentLLMService", _FakeService)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "external_deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    svc = isi_mod.create_intelligent_service_from_env()
    assert svc.kwargs["provider"] == "deepseek"
    assert svc.kwargs["api_key"] == "sk-env"


def test_env_factory_unknown_provider(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "wat")
    assert isi_mod.create_intelligent_service_from_env() is None


def test_env_factory_missing_key_non_ollama(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isi_mod.create_intelligent_service_from_env() is None


def test_env_factory_ollama_no_key_ok(monkeypatch):
    monkeypatch.setattr(isi_mod, "IntelligentLLMService", _FakeService)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    svc = isi_mod.create_intelligent_service_from_env()
    assert svc.kwargs["api_key"] == ""


def test_env_factory_model_prefix_stripped(monkeypatch):
    monkeypatch.setattr(isi_mod, "IntelligentLLMService", _FakeService)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EXTERNAL_LLM_MODEL", "external_openai:gpt-4")
    svc = isi_mod.create_intelligent_service_from_env()
    assert svc.kwargs["model"] == "gpt-4"


def test_env_factory_model_plain(monkeypatch):
    monkeypatch.setattr(isi_mod, "IntelligentLLMService", _FakeService)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EXTERNAL_LLM_MODEL", "gpt-4")
    svc = isi_mod.create_intelligent_service_from_env()
    assert svc.kwargs["model"] == "gpt-4"


def test_env_factory_constructor_exception_returns_none(monkeypatch):
    def _raising(**kw):
        raise RuntimeError("ctor failed")

    monkeypatch.setattr(isi_mod, "IntelligentLLMService", _raising)
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isi_mod.create_intelligent_service_from_env() is None


def test_multi_file_raising_progress_callback_does_not_abort(tmp_path, monkeypatch):
    """Contract: a failing progress callback must not break the multi-file run
    (mirrors _emit_progress's 'never fail the run' guarantee)."""
    llm = _FakeLLM({"success": True, "patch": "P1"}, {"success": True, "patch": "P2"})
    svc = _make_service(llm)

    class _FakePlanner:
        def __init__(self, *a, **kw):
            pass

        def create_plan(self, req):
            return _plan(_op("a.py"), _op("b.py"))

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)

    def boom(*a):
        raise RuntimeError("ui broke")

    result = svc._handle_multi_file(
        tmp_path,
        "req",
        _analysis(),
        _structure(),
        0.0,
        llm_planning=False,
        progress_callback=boom,
    )
    assert result["success"] is True
    assert len(llm.calls) == 2
