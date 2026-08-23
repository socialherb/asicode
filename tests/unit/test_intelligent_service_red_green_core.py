"""RED→GREEN: IntelligentLLMService core paths (construction, handle_request,
output-mode policy, single-file retry/fallback).

Covers lines of intelligent_service.py that the pre-existing suite did not
exercise: __init__ timeout policy, _emit_progress, handle_request mode routing,
_determine_output_mode heuristics, _output_mode_to_string, and the single-file
retry loop (diff→full_file fallback, error feedback, same-failure repeat,
create-fallback / modify-protection).
"""

from __future__ import annotations

from pathlib import Path

import external_llm.intelligent_service as isi_mod
from external_llm.client import DEFAULT_LLM_TIMEOUT, OLLAMA_LLM_TIMEOUT
from external_llm.intelligent_service import IntelligentLLMService
from external_llm.output_modes import OutputMode
from external_llm.project_analyzer import ProjectStructure
from external_llm.smart_analyzer import RequestAnalysis

# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeLLM:
    """Minimal ExternalLLMService stand-in: scripted generate_patch responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.client = object()
        self.model = "fake-model"

    def generate_patch(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return {"success": True, "patch": "--- a/x.py\n+++ b/x.py\n"}


class _FakeAnalyzer:
    def __init__(self, analysis):
        self._analysis = analysis

    def analyze(self, user_request):
        return self._analysis


class _FakeProjectAnalyzer:
    def __init__(self, structure):
        self._structure = structure

    def analyze(self):
        return self._structure


def _make_service(llm=None, provider="mock") -> IntelligentLLMService:
    svc = object.__new__(IntelligentLLMService)
    svc.llm_service = llm or _FakeLLM()
    svc.provider = provider
    svc.model = getattr(svc.llm_service, "model", "mock-model")
    return svc


def _patch_analyzers(monkeypatch, analysis: RequestAnalysis, structure: ProjectStructure):
    monkeypatch.setattr(isi_mod, "SmartRequestAnalyzer", lambda root: _FakeAnalyzer(analysis))
    monkeypatch.setattr(isi_mod, "ProjectAnalyzer", lambda root: _FakeProjectAnalyzer(structure))


def _structure(**kw) -> ProjectStructure:
    defaults = {
        "framework": "Flask",
        "frameworks": [],
        "languages": ["Python"],
        "primary_language": "Python",
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


def _analysis(**kw) -> RequestAnalysis:
    defaults = {
        "original_request": "req",
        "intent": "general",
        "feature_name": None,
        "suggested_files": [],
        "file_operations": {},
        "tech_stack": [],
        "enhanced_request": "",
        "confidence": 0.5,
        "needs_planning": False,
    }
    defaults.update(kw)
    return RequestAnalysis(**defaults)


# ── __init__ ────────────────────────────────────────────────────────────────


class _FakeExternalLLM:
    """Captures constructor kwargs; exposes .model/.client like the real class."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model") or "resolved-model"
        self.client = object()


def test_init_ollama_extends_default_timeout(monkeypatch):
    monkeypatch.setattr(isi_mod, "ExternalLLMService", _FakeExternalLLM)
    svc = IntelligentLLMService("ollama", "key", timeout=DEFAULT_LLM_TIMEOUT)
    assert svc.llm_service.kwargs["timeout"] == OLLAMA_LLM_TIMEOUT
    assert svc.provider == "ollama"


def test_init_ollama_keeps_explicit_timeout(monkeypatch):
    monkeypatch.setattr(isi_mod, "ExternalLLMService", _FakeExternalLLM)
    svc = IntelligentLLMService("ollama", "key", timeout=123)
    assert svc.llm_service.kwargs["timeout"] == 123


def test_init_model_falls_back_to_service_model(monkeypatch):
    monkeypatch.setattr(isi_mod, "ExternalLLMService", _FakeExternalLLM)
    svc = IntelligentLLMService("openai", "key")
    assert svc.model == "resolved-model"


def test_init_model_override(monkeypatch):
    monkeypatch.setattr(isi_mod, "ExternalLLMService", _FakeExternalLLM)
    svc = IntelligentLLMService("openai", "key", model="gpt-x")
    assert svc.model == "gpt-x"


# ── _emit_progress ──────────────────────────────────────────────────────────


def test_emit_progress_no_callback():
    svc = _make_service()
    assert svc._emit_progress(None, "p", "m", 1, 2) is None


def test_emit_progress_calls_callback():
    svc = _make_service()
    seen = []
    svc._emit_progress(lambda *a: seen.append(a), "p", "m", 1, 2)
    assert seen == [("p", "m", 1, 2)]


def test_emit_progress_callback_exception_swallowed():
    svc = _make_service()

    def boom(*a):
        raise RuntimeError("cb broke")

    assert svc._emit_progress(boom, "p", "m") is None  # must not raise


# ── handle_request: mode routing ────────────────────────────────────────────


def test_handle_request_auto_planning_routes_multi(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "--- a/a.py\n+++ b/a.py\n"})
    svc = _make_service(llm)
    _patch_analyzers(
        monkeypatch,
        _analysis(needs_planning=True, intent="create_feature"),
        _structure(),
    )
    captured = {}

    class _FakeLLMPlanner:
        def __init__(self, **kw):
            captured.update(kw)

        def create_plan(self, req):
            from external_llm.multi_planner import ExecutionPlan, FileOperation

            captured["req"] = req
            return ExecutionPlan(
                original_request=req,
                operations=[FileOperation(file_path="a.py", operation="create", description="d", instructions="i")],
                complexity="moderate",
            )

    monkeypatch.setattr(isi_mod, "LLMEnhancedMultiFilePlanner", _FakeLLMPlanner)
    result = svc.handle_request(repo_root=str(tmp_path), user_request="make login")
    assert result["mode"] == "multi_file"
    assert result["success"] is True
    assert "a.py" in result["patch"]
    assert captured["req"] == "make login"
    assert llm.calls[0]["target_file"] == "a.py"


def test_handle_request_auto_simple_routes_single(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"})
    svc = _make_service(llm)
    _patch_analyzers(
        monkeypatch,
        _analysis(needs_planning=False, suggested_files=["app.py"]),
        _structure(),
    )
    result = svc.handle_request(repo_root=str(tmp_path), user_request="fix bug")
    assert result["mode"] == "single_file"
    assert result["success"] is True
    assert llm.calls[0]["target_file"] == "app.py"


def test_handle_request_mode_single(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "P"})
    svc = _make_service(llm)
    _patch_analyzers(monkeypatch, _analysis(suggested_files=["x.py"]), _structure())
    result = svc.handle_request(repo_root=str(tmp_path), user_request="r", mode="single")
    assert result["mode"] == "single_file"
    assert result["success"] is True


def test_handle_request_mode_llm_plan(tmp_path, monkeypatch):
    svc = _make_service()
    _patch_analyzers(monkeypatch, _analysis(), _structure())
    captured = {}

    class _FakeLLMPlanner:
        def __init__(self, **kw):
            captured.update(kw)

        def create_plan(self, req):
            from external_llm.multi_planner import ExecutionPlan

            return ExecutionPlan(original_request=req)

    monkeypatch.setattr(isi_mod, "LLMEnhancedMultiFilePlanner", _FakeLLMPlanner)
    result = svc.handle_request(repo_root=str(tmp_path), user_request="r", mode="llm_plan")
    assert result["mode"] == "multi_file"
    assert captured["llm_client"] is svc.llm_service.client


def test_handle_request_mode_unknown_falls_back_to_multi(tmp_path, monkeypatch):
    svc = _make_service()
    _patch_analyzers(monkeypatch, _analysis(), _structure())
    captured = {}

    class _FakePlanner:
        def __init__(self, repo_root):
            captured["root"] = repo_root

        def create_plan(self, req):
            from external_llm.multi_planner import ExecutionPlan

            return ExecutionPlan(original_request=req)

    monkeypatch.setattr(isi_mod, "MultiFilePlanner", _FakePlanner)
    result = svc.handle_request(repo_root=str(tmp_path), user_request="r", mode="whatever")
    assert result["mode"] == "multi_file"


def test_handle_request_exception_returns_error_dict(tmp_path, monkeypatch):
    svc = _make_service()

    def boom(root):
        raise RuntimeError("analyzer exploded")

    monkeypatch.setattr(isi_mod, "SmartRequestAnalyzer", boom)
    result = svc.handle_request(repo_root=str(tmp_path), user_request="r")
    assert result["success"] is False
    assert result["mode"] == "error"
    assert "analyzer exploded" in result["error"]


def test_handle_request_emits_progress(tmp_path, monkeypatch):
    svc = _make_service()
    _patch_analyzers(monkeypatch, _analysis(suggested_files=["x.py"]), _structure())
    events = []
    svc.handle_request(
        repo_root=str(tmp_path),
        user_request="r",
        mode="single",
        progress_callback=lambda *a: events.append(a),
    )
    assert events[0][0] == "analyzing_request"
    assert events[-1][0] == "generating_patch"


# ── _determine_output_mode ──────────────────────────────────────────────────


def test_output_mode_new_file_default_full_file(tmp_path):
    svc = _make_service()
    mode, cv = svc._determine_output_mode(tmp_path, "new.py", operation="create")
    assert mode == OutputMode.FULL_FILE
    assert cv == "v7"


def test_output_mode_new_file_ollama_diff(tmp_path):
    svc = _make_service(provider="ollama")
    mode, _ = svc._determine_output_mode(tmp_path, "new.py", operation="create")
    assert mode == OutputMode.UNIFIED_DIFF


def test_output_mode_auto_detect_missing_file(tmp_path):
    svc = _make_service()
    mode, _ = svc._determine_output_mode(tmp_path, "missing.py")
    assert mode == OutputMode.FULL_FILE  # not-exists treated as create


def test_output_mode_existing_large_hint_gemini_diff(tmp_path):
    target = tmp_path / "big.py"
    target.write_text("x = 1\n", encoding="utf-8")
    svc = _make_service(provider="google")
    mode, _ = svc._determine_output_mode(tmp_path, "big.py", change_size_hint="large")
    assert mode == OutputMode.UNIFIED_DIFF


def test_output_mode_existing_large_hint_default_full(tmp_path):
    target = tmp_path / "big.py"
    target.write_text("x = 1\n", encoding="utf-8")
    svc = _make_service(provider="deepseek")
    mode, _ = svc._determine_output_mode(tmp_path, "big.py", change_size_hint="rewrite")
    assert mode == OutputMode.FULL_FILE


def test_output_mode_large_file_with_medium_hint_full(tmp_path):
    target = tmp_path / "huge.py"
    target.write_text("\n".join(f"x{i} = {i}" for i in range(600)), encoding="utf-8")
    svc = _make_service()
    mode, _ = svc._determine_output_mode(tmp_path, "huge.py", change_size_hint="medium")
    assert mode == OutputMode.FULL_FILE


def test_output_mode_large_file_no_hint_diff(tmp_path):
    target = tmp_path / "huge.py"
    target.write_text("\n".join(f"x{i} = {i}" for i in range(600)), encoding="utf-8")
    svc = _make_service()
    mode, _ = svc._determine_output_mode(tmp_path, "huge.py")
    assert mode == OutputMode.UNIFIED_DIFF


def test_output_mode_unreadable_file_warns_and_defaults_diff(tmp_path, monkeypatch):
    target = tmp_path / "weird.py"
    target.write_text("x = 1\n", encoding="utf-8")
    svc = _make_service()

    def broken_open(*a, **kw):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "boom")

    monkeypatch.setattr("builtins.open", broken_open)
    mode, _ = svc._determine_output_mode(tmp_path, "weird.py")
    assert mode == OutputMode.UNIFIED_DIFF


def test_output_mode_small_existing_diff(tmp_path):
    target = tmp_path / "small.py"
    target.write_text("x = 1\n", encoding="utf-8")
    svc = _make_service()
    mode, _ = svc._determine_output_mode(tmp_path, "small.py", change_size_hint="small")
    assert mode == OutputMode.UNIFIED_DIFF


# ── _output_mode_to_string ──────────────────────────────────────────────────


def test_output_mode_to_string_all_branches():
    svc = _make_service()
    assert svc._output_mode_to_string(OutputMode.UNIFIED_DIFF) == "diff"
    assert svc._output_mode_to_string(OutputMode.FULL_FILE) == "full_file"
    assert svc._output_mode_to_string(OutputMode.ASICODE_BLOCK) == "auto"
    assert svc._output_mode_to_string(OutputMode.TARGETED_BLOCK) == "auto"
    assert svc._output_mode_to_string(OutputMode.PLAN_JSON) == "auto"
    assert svc._output_mode_to_string(OutputMode.NEEDS_DISAMBIGUATION) == "diff"


# ── _handle_single_file: target resolution ──────────────────────────────────


def test_single_file_target_from_suggested(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": True, "patch": "P"})
    svc = _make_service(llm)
    _patch_analyzers(monkeypatch, _analysis(suggested_files=["sug.py"]), _structure())
    result = svc.handle_request(repo_root=str(tmp_path), user_request="r", mode="single", target_file=None)
    assert result["success"] is True
    assert llm.calls[0]["target_file"] == "sug.py"


def test_single_file_no_target_no_suggested_errors(tmp_path, monkeypatch):
    svc = _make_service()
    _patch_analyzers(monkeypatch, _analysis(suggested_files=[]), _structure())
    result = svc.handle_request(repo_root=str(tmp_path), user_request="r", mode="single")
    assert result["success"] is False
    assert "No target file specified" in result["error"]


# ── _handle_single_file: retry loop ─────────────────────────────────────────


def test_single_file_diff_then_full_file_retry_with_feedback(tmp_path):
    llm = _FakeLLM(
        {"success": False, "error": "git_apply_check_failed: context mismatch"},
        {"success": True, "patch": "--- a/app.py\n+++ b/app.py\n"},
    )
    svc = _make_service(llm)
    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    result = svc._handle_single_file(
        tmp_path,
        "fix it",
        _analysis(suggested_files=["app.py"]),
        _structure(),
        "app.py",
        0.0,
    )
    assert result["success"] is True
    assert result["output_mode_used"] == "full_file"
    assert result["error_feedback_included"] is True
    assert "ERROR FEEDBACK" in llm.calls[1]["user_request"]
    assert "git apply check failed" in llm.calls[1]["user_request"]


def test_single_file_same_failure_repeat_stops_early(tmp_path):
    llm = _FakeLLM(
        {"success": False, "error": "empty_patch: nothing"},
        {"success": False, "error": "empty_patch: nothing again"},
    )
    svc = _make_service(llm)
    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    result = svc._handle_single_file(
        tmp_path,
        "fix",
        _analysis(suggested_files=["app.py"]),
        _structure(),
        "app.py",
        0.0,
    )
    assert result["success"] is False
    assert result["same_failure_repeat"] is True
    assert result["failure_reason"] == "empty_patch"
    assert len(llm.calls) == 2  # second failure repeated → no third attempt


def test_single_file_create_fallback_placeholder(tmp_path):
    llm = _FakeLLM({"success": False, "error": "invalid_diff: nope"})
    svc = _make_service(llm)
    result = svc._handle_single_file(
        tmp_path,
        "create a module",
        _analysis(suggested_files=["new_mod.py"]),
        _structure(),
        "new_mod.py",
        0.0,
    )
    assert result["success"] is True
    assert result["fallback_used"] is True
    assert result["retry_count"] == 1
    assert (tmp_path / "new_mod.py").exists()


def test_single_file_modify_failure_no_fallback(tmp_path):
    llm = _FakeLLM(
        {"success": False, "error": "invalid_diff: nope"},
        {"success": False, "error": "invalid_diff: nope again"},
    )
    svc = _make_service(llm)
    target = tmp_path / "app.py"
    target.write_text("KEEP\n", encoding="utf-8")
    result = svc._handle_single_file(
        tmp_path,
        "fix",
        _analysis(suggested_files=["app.py"]),
        _structure(),
        "app.py",
        0.0,
    )
    assert result["success"] is False
    assert result["fallback_used"] is False
    assert result["same_failure_repeat"] is True
    assert target.read_text(encoding="utf-8") == "KEEP\n"


def test_single_file_create_fallback_write_failure(tmp_path, monkeypatch):
    llm = _FakeLLM({"success": False, "error": "boom"})
    svc = _make_service(llm)

    def broken_write(self, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", broken_write)
    result = svc._handle_single_file(
        tmp_path,
        "create",
        _analysis(suggested_files=["x.py"]),
        _structure(),
        "x.py",
        0.0,
    )
    assert result["success"] is False
    assert result["fallback_reason"] == "fallback_generation_failed"


def test_single_file_success_metadata(tmp_path):
    llm = _FakeLLM({"success": True, "patch": "--- a/app.py\n+++ b/app.py\n"})
    svc = _make_service(llm)
    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    result = svc._handle_single_file(
        tmp_path,
        "fix",
        _analysis(suggested_files=["app.py"]),
        _structure(),
        "app.py",
        0.0,
    )
    assert result["success"] is True
    assert result["output_mode_used"] == "unified_diff"
    assert result["retry_count"] == 1
    assert result["fallback_used"] is False
    assert result["mode"] == "single_file"
    assert result["target_file"] == "app.py"
    assert result["analysis"]["intent"] == "general"
