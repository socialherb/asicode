"""RED→GREEN: IntelligentLLMService helper methods (full branch coverage).

Covers the context builders (_build_project_context_summary,
_build_enhanced_context, _build_operation_request), patch/explanation helpers
(_combine_patches, _create_default_file_patch, _generate_multi_file_explanation,
_estimate_change_size), failure analysis (_analyze_failure_patterns,
_build_error_feedback, _extract_failure_reason), and the dict converters
(_analysis_to_dict, _plan_to_dict).
"""

from __future__ import annotations

from pathlib import Path

from external_llm.intelligent_service import IntelligentLLMService
from external_llm.multi_planner import ExecutionPlan, FileOperation
from external_llm.output_modes import OutputMode
from external_llm.project_analyzer import ProjectStructure
from external_llm.smart_analyzer import RequestAnalysis


def _make_service() -> IntelligentLLMService:
    svc = object.__new__(IntelligentLLMService)
    svc.llm_service = type("L", (), {"client": object(), "model": "m"})()
    svc.provider = "mock"
    svc.model = "m"
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
        "directories": {"app": ["app/"], "other": ["misc/"]},
        "naming_style": "snake_case",
        "common_imports": ["os"],
        "entry_points": ["main.py"],
        "test_dir": "tests",
        "example_files": {"route": "app/routes.py", "model": "app/models.py", "x": "x.py", "y": "y.py"},
    }
    defaults.update(kw)
    return ProjectStructure(**defaults)


def _op(file_path, operation="create", description="d", instructions="", **kw) -> FileOperation:
    return FileOperation(
        file_path=file_path, operation=operation, description=description, instructions=instructions, **kw
    )


# ── _build_project_context_summary ──────────────────────────────────────────


def test_project_context_summary_all_sections():
    svc = _make_service()
    out = svc._build_project_context_summary(_structure())
    assert "- **Frameworks**: Flask" in out
    assert "- **Project Type**: web" in out
    assert "- **Directory Structure**:" in out
    assert "app: app/" in out
    assert "misc" not in out  # 'other' purpose excluded
    assert "- **Naming Convention**: snake_case" in out
    assert "- **Common Imports**: os" in out
    assert "- **Example Files**:" in out
    assert "app/routes.py" in out
    assert "y.py" not in out  # only first 3 examples


def test_project_context_summary_framework_fallback_and_empty():
    svc = _make_service()
    out = svc._build_project_context_summary(
        _structure(
            frameworks=[], framework="Django", directories={}, example_files={}, common_imports=[], project_types=[]
        )
    )
    assert "- **Framework**: Django" in out
    out2 = svc._build_project_context_summary(
        _structure(
            frameworks=[],
            framework=None,
            directories={},
            example_files={},
            common_imports=[],
            project_types=[],
            naming_style="",
        )
    )
    assert out2 == "No project context available."


# ── _build_enhanced_context ─────────────────────────────────────────────────


def test_enhanced_context_full_file_mode():
    svc = _make_service()
    out = svc._build_enhanced_context(
        "add login",
        _analysis(),
        _structure(),
        "app.py",
        output_mode=OutputMode.FULL_FILE,
        operation="create",
    )
    assert "**User Request**: add login" in out
    assert "**Feature**: login" in out
    assert "**Frameworks**: Flask" in out
    assert "**Naming Convention**: snake_case" in out
    assert "**Reference Examples**" in out
    assert "**Output Format (FULL_FILE Mode)**" in out
    assert "FILE: app.py" in out
    assert "- Operation: create" in out


def test_enhanced_context_unified_diff_mode():
    svc = _make_service()
    out = svc._build_enhanced_context(
        "fix",
        _analysis(feature_name=None),
        _structure(frameworks=[], framework="Django"),
        "app.py",
        output_mode=OutputMode.UNIFIED_DIFF,
        operation="modify",
    )
    assert "**Output Format (UNIFIED_DIFF Mode)**" in out
    assert "--- a/app.py" in out
    assert "Follow Django conventions" in out  # framework fallback


def test_enhanced_context_other_mode():
    svc = _make_service()
    out = svc._build_enhanced_context(
        "r",
        _analysis(),
        _structure(example_files={}),
        "app.py",
        output_mode=OutputMode.PLAN_JSON,
    )
    assert "**Output Format**: Follow the appropriate output format" in out


# ── _build_operation_request ────────────────────────────────────────────────


def test_operation_request_with_instructions():
    svc = _make_service()
    out = svc._build_operation_request(_op("a.py", instructions="DO THIS"), _structure(), "goal")
    assert "**Overall Goal**: goal" in out
    assert "**Current Task**: d" in out
    assert "**File**: `a.py`" in out
    assert "**Specific Instructions**:" in out
    assert "DO THIS" in out


def test_operation_request_create_css_editor_and_plain():
    svc = _make_service()
    out = svc._build_operation_request(_op("style.css", description="add line number column"), _structure(), "goal")
    assert "Monospace font for code" in out
    out2 = svc._build_operation_request(_op("theme.css", description="theme"), _structure(), "goal")
    assert "appropriate styles" in out2


def test_operation_request_create_js_editor_and_plain():
    svc = _make_service()
    out = svc._build_operation_request(_op("editor.js", description="line number sync"), _structure(), "goal")
    assert "generate line numbers" in out
    out2 = svc._build_operation_request(_op("util.js", description="utils"), _structure(), "goal")
    assert "proper functions and event handlers" in out2


def test_operation_request_create_html_editor_and_plain():
    svc = _make_service()
    out = svc._build_operation_request(_op("templates/editor.html", description="editor page"), _structure(), "goal")
    assert "contenteditable" in out
    out2 = svc._build_operation_request(_op("index.html", description="home"), _structure(), "goal")
    assert "proper structure, elements" in out2


def test_operation_request_create_py_service_test_plain():
    svc = _make_service()
    out = svc._build_operation_request(_op("svc.py", description="add service endpoint"), _structure(), "goal")
    assert "Route/endpoint definitions" in out
    out2 = svc._build_operation_request(_op("test_editor.py", description="tests"), _structure(), "goal")
    assert "Test cases for line number functionality" in out2
    out3 = svc._build_operation_request(_op("mod.py", description="module"), _structure(), "goal")
    assert "proper imports, functions, classes" in out3


def test_operation_request_create_other_ext():
    svc = _make_service()
    out = svc._build_operation_request(_op("data.json", description="data"), _structure(), "goal")
    assert "Create a new file at data.json for goal..." in out


def test_operation_request_main_py_route_and_enhancement():
    svc = _make_service()
    out = svc._build_operation_request(
        _op("main.py", operation="modify", description="add route for editor", instructions=""), _structure(), "goal"
    )
    assert "Import for editor service router" in out
    assert "**Important for main.py modifications**" in out


def test_operation_request_main_py_enhancement_no_duplication():
    svc = _make_service()
    out = svc._build_operation_request(
        _op(
            "main.py",
            operation="modify",
            description="add route for editor",
            instructions="has **Important for main.py modifications** already",
        ),
        _structure(),
        "goal",
    )
    assert out.count("**Important for main.py modifications**") == 1


def test_operation_request_generic_modify():
    svc = _make_service()
    out = svc._build_operation_request(
        _op("app.py", operation="modify", description="add feature", instructions=""), _structure(), "goal"
    )
    assert "Modify the file app.py to add feature" in out


def test_operation_request_dependencies_template_and_conventions():
    svc = _make_service()
    out = svc._build_operation_request(_op("b.py", dependencies=["a.py"], template_file="tpl.py"), _structure(), "goal")
    assert "**Dependencies** (already created):" in out
    assert "- `a.py`" in out
    assert "**Template Reference**: `tpl.py`" in out
    assert "**Follow Flask best practices**" in out
    assert "**Project Type**: web" in out


def test_operation_request_framework_fallback():
    svc = _make_service()
    out = svc._build_operation_request(_op("b.py"), _structure(frameworks=[], framework="Django"), "goal")
    assert "**Framework**: Django" in out
    assert "**Follow Django best practices**" in out


def test_operation_request_output_modes_all():
    svc = _make_service()
    for mode, needle in [
        (OutputMode.UNIFIED_DIFF, "UNIFIED_DIFF ONLY"),
        (OutputMode.FULL_FILE, "FULL_FILE MODE"),
        (OutputMode.ASICODE_BLOCK, "ASICODE_BEGIN / BEFORE / AFTER / ASICODE_END"),
        (OutputMode.TARGETED_BLOCK, "FUNCTION: <name> + INSERT_AFTER"),
        (OutputMode.PLAN_JSON, "JSON plan with operations array"),
        (OutputMode.NEEDS_DISAMBIGUATION, "NEEDS_DISAMBIGUATION with clarification questions"),
    ]:
        out = svc._build_operation_request(_op("x.py"), _structure(), "goal", output_mode=mode)
        assert needle in out, mode


# ── _combine_patches ────────────────────────────────────────────────────────


def test_combine_patches_only_successful():
    svc = _make_service()
    out = svc._combine_patches(
        [
            {"success": True, "patch": "A"},
            {"success": False, "patch": "B"},
            {"success": True, "patch": ""},
            {"success": True, "patch": "C"},
        ]
    )
    assert out == "A\n\nC"


def test_combine_patches_empty():
    svc = _make_service()
    assert svc._combine_patches([]) == ""


# ── _create_default_file_patch ──────────────────────────────────────────────


def test_default_patch_refuses_existing(tmp_path):
    svc = _make_service()
    target = tmp_path / "exists.py"
    target.write_text("KEEP", encoding="utf-8")
    assert svc._create_default_file_patch(tmp_path, _op("exists.py")) == ""
    assert target.read_text(encoding="utf-8") == "KEEP"


def test_default_patch_js(tmp_path):
    svc = _make_service()
    patch = svc._create_default_file_patch(tmp_path, _op("app.js"))
    assert "function updateLineNumbers" in patch
    assert "new file mode 100644" in patch
    assert (tmp_path / "app.js").read_text(encoding="utf-8").startswith("// JavaScript")


def test_default_patch_css(tmp_path):
    svc = _make_service()
    patch = svc._create_default_file_patch(tmp_path, _op("style.css"))
    assert ".line-numbers" in patch
    assert (tmp_path / "style.css").exists()


def test_default_patch_html(tmp_path):
    svc = _make_service()
    patch = svc._create_default_file_patch(tmp_path, _op("templates/editor.html"))
    assert "<!DOCTYPE html>" in patch
    assert (tmp_path / "templates" / "editor.html").exists()


def test_default_patch_py_test_and_plain(tmp_path):
    svc = _make_service()
    patch = svc._create_default_file_patch(tmp_path, _op("test_editor.py"))
    assert "def test_editor_basic()" in patch
    patch2 = svc._create_default_file_patch(tmp_path, _op("mod.py"))
    assert "# File: mod.py" in patch2
    assert "Please implement" in patch2


def test_default_patch_other_ext(tmp_path):
    svc = _make_service()
    patch = svc._create_default_file_patch(tmp_path, _op("data.json"))
    assert "# File: data.json" in patch
    assert "placeholder file" in patch


def test_default_patch_write_error_returns_empty(tmp_path, monkeypatch):
    svc = _make_service()

    def broken_write(self, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", broken_write)
    assert svc._create_default_file_patch(tmp_path, _op("x.py")) == ""


# ── _generate_multi_file_explanation ────────────────────────────────────────


def test_generate_multi_file_explanation_with_failures():
    svc = _make_service()
    plan = ExecutionPlan(original_request="r", complexity="complex")
    out = svc._generate_multi_file_explanation(
        plan,
        [
            {"success": True, "file": "a.py", "operation": "create"},
            {"success": False, "file": "b.py", "operation": "modify"},
        ],
    )
    assert "Executed multi-file plan for: r" in out
    assert "Complexity: complex" in out
    assert "✅ Successful: 1" in out
    assert "❌ Failed: 1" in out
    assert "✅ a.py (create)" in out
    assert "❌ b.py (modify)" in out


def test_generate_multi_file_explanation_all_success():
    svc = _make_service()
    plan = ExecutionPlan(original_request="r")
    out = svc._generate_multi_file_explanation(plan, [{"success": True, "file": "a.py", "operation": "create"}])
    assert "✅ Successful: 1" in out
    assert "❌ Failed" not in out


# ── _estimate_change_size ───────────────────────────────────────────────────


def test_estimate_change_size():
    svc = _make_service()
    assert svc._estimate_change_size("create") == "medium"
    assert svc._estimate_change_size("modify") == "small"
    assert svc._estimate_change_size("delete") is None


# ── _analyze_failure_patterns ───────────────────────────────────────────────


def test_analyze_failure_patterns_success_noop():
    svc = _make_service()
    assert svc._analyze_failure_patterns({"success": True}) is None


def test_analyze_failure_patterns_empty_patch_diff_recommendation():
    svc = _make_service()
    svc._analyze_failure_patterns(
        {
            "success": False,
            "error": "empty_patch: none",
            "output_mode_used": "unified_diff",
            "target_file": "app.py",
        }
    )


def test_analyze_failure_patterns_git_apply_and_unknown():
    svc = _make_service()
    svc._analyze_failure_patterns(
        {
            "success": False,
            "error": "git_apply_check_failed: ctx",
            "output_mode": "full_file",
            "target_file": "noext",
        }
    )
    svc._analyze_failure_patterns(
        {
            "success": False,
            "error": "something else",
            "target_file": "unknown",
        }
    )


# ── _build_error_feedback ───────────────────────────────────────────────────


def test_error_feedback_error_codes_and_existing_file(tmp_path):
    svc = _make_service()
    target = tmp_path / "mod.py"
    target.write_text("\n".join(f"x{i} = {i}" for i in range(5)), encoding="utf-8")
    out = svc._build_error_feedback("git_apply_check_failed: boom", target, tmp_path)
    assert "git apply check failed" in out
    assert "x1 = 1" in out
    out2 = svc._build_error_feedback("empty_patch: e", target, tmp_path)
    assert "empty patch" in out2
    out3 = svc._build_error_feedback("missing_hunks: h", target, tmp_path)
    assert "missing hunks" in out3
    out4 = svc._build_error_feedback("invalid_diff: i", target, tmp_path)
    assert "invalid diff format" in out4


def test_error_feedback_missing_file_and_outside_root(tmp_path):
    svc = _make_service()
    out = svc._build_error_feedback("weird error", tmp_path / "absent.py", tmp_path)
    assert "does not exist" in out
    # file outside repo root → rel_path falls back to absolute path
    outside = Path("/tmp/outside.py")
    out2 = svc._build_error_feedback("e", outside, tmp_path)
    assert "outside.py" in out2


def test_error_feedback_read_exception(tmp_path, monkeypatch):
    svc = _make_service()
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")

    def broken_open(*a, **kw):
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "open", broken_open)
    out = svc._build_error_feedback("e", target, tmp_path)
    assert "- Failed file:" in out
    assert "- Error: e" in out


# ── _extract_failure_reason ─────────────────────────────────────────────────


def test_extract_failure_reason_all_codes():
    svc = _make_service()
    assert svc._extract_failure_reason("") == "unknown"
    assert svc._extract_failure_reason("empty_patch detected") == "empty_patch"
    assert svc._extract_failure_reason("git_apply_check_failed") == "git_apply_failed"
    assert svc._extract_failure_reason("missing_hunks") == "missing_hunks"
    assert svc._extract_failure_reason("header-only diff") == "header_only"
    assert svc._extract_failure_reason("header only diff") == "header_only"
    assert svc._extract_failure_reason("invalid_diff format") == "invalid_diff"
    assert svc._extract_failure_reason("no diff found") == "no_diff"
    assert svc._extract_failure_reason("no_diff") == "no_diff"
    assert svc._extract_failure_reason("inconsistent hunk line counts") == "inconsistent_hunk_lines"
    assert svc._extract_failure_reason("custom thing") == "custom thing"


# ── dict converters ─────────────────────────────────────────────────────────


def test_analysis_to_dict():
    svc = _make_service()
    out = svc._analysis_to_dict(_analysis())
    assert out["intent"] == "create_feature"
    assert out["feature_name"] == "login"
    assert out["suggested_files"] == ["a.py"]
    assert out["confidence"] == 0.9
    assert out["needs_planning"] is True


def test_plan_to_dict():
    svc = _make_service()
    plan = ExecutionPlan(
        original_request="r",
        complexity="complex",
        strategy="parallel",
        success_criteria=["ok"],
        operations=[_op("a.py", dependencies=["b.py"])],
    )
    out = svc._plan_to_dict(plan)
    assert out["complexity"] == "complex"
    assert out["strategy"] == "parallel"
    assert out["operations"][0]["file"] == "a.py"
    assert out["operations"][0]["dependencies"] == ["b.py"]
    assert out["success_criteria"] == ["ok"]
