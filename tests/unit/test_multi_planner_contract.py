"""Contract tests for multi_planner: create_plan routing, generic feature
planning, suggested-file operations, topological order, complexity assessment,
project-context summarization, and LLM plan parsing branches.

Complements test_multi_planner.py (null-coercion regressions) by covering the
rule-based planner surface with fake analyzers so no filesystem scan is needed.
"""
from external_llm.multi_planner import (
    FileOperation,
    LLMEnhancedMultiFilePlanner,
    MultiFilePlanner,
)
from external_llm.project_analyzer import ProjectStructure
from external_llm.smart_analyzer import RequestAnalysis


class _FakeAnalyzer:
    """Stand-in for SmartRequestAnalyzer: returns a fixed RequestAnalysis."""

    def __init__(self, analysis):
        self._analysis = analysis
        self.last_request = None

    def analyze(self, user_request):
        self.last_request = user_request
        return self._analysis


class _FakeProjectAnalyzer:
    """Stand-in for ProjectAnalyzer: returns a fixed ProjectStructure."""

    def __init__(self, structure):
        self._structure = structure

    def analyze(self):
        return self._structure


def _planner(analysis, structure=None, cls=MultiFilePlanner):
    """Build a planner with fake analyzers (no filesystem scan)."""
    p = cls.__new__(cls)
    p.analyzer = _FakeAnalyzer(analysis)
    p.project_analyzer = _FakeProjectAnalyzer(structure or ProjectStructure())
    return p


# ---------------------------------------------------------------------------
# __init__ / create_plan routing
# ---------------------------------------------------------------------------


def test_init_creates_real_analyzers(tmp_path):
    """__init__ wires repo_root and real analyzers (cheap, no scan)."""
    p = MultiFilePlanner(str(tmp_path))
    assert p.repo_root == tmp_path
    assert p.analyzer is not None
    assert p.project_analyzer is not None


def test_create_plan_simple_with_suggested_files():
    analysis = RequestAnalysis(
        original_request="fix typo in main.py",
        intent="fix_bug",
        suggested_files=["main.py"],
        file_operations={"main.py": "modify"},
        needs_planning=False,
    )
    plan = _planner(analysis).create_plan("fix typo in main.py")
    assert plan.strategy == "sequential"
    assert plan.complexity == "simple"
    assert len(plan.operations) == 1
    op = plan.operations[0]
    assert op.file_path == "main.py"
    assert op.operation == "modify"
    assert op.priority == 0
    assert "fix_bug" in op.description


def test_create_plan_simple_no_suggested_files():
    analysis = RequestAnalysis(
        original_request="hello", intent="general", needs_planning=False
    )
    plan = _planner(analysis).create_plan("hello")
    assert plan.operations == []
    assert plan.complexity == "simple"


def test_create_plan_complex_generic_feature():
    """needs_planning with no suggested files -> generic feature plan."""
    analysis = RequestAnalysis(
        original_request="add login feature",
        intent="create_feature",
        feature_name="auth",
        needs_planning=True,
    )
    plan = _planner(analysis).create_plan("add login feature")
    assert len(plan.operations) == 1
    assert plan.operations[0].file_path == "auth.py"
    assert plan.operations[0].operation == "create"
    assert plan.strategy == "sequential"
    assert plan.complexity == "simple"  # single op
    assert len(plan.success_criteria) == 3


def test_create_plan_complex_suggested_files():
    """needs_planning with suggested files -> suggested-file operations."""
    analysis = RequestAnalysis(
        original_request="build landing page",
        intent="create_feature",
        feature_name="landing",
        suggested_files=["index.html", "style.css", "app.js"],
        needs_planning=True,
    )
    plan = _planner(analysis).create_plan("build landing page")
    files = [op.file_path for op in plan.operations]
    assert files == ["index.html", "app.js", "style.css"]
    # HTML priority 0 first, then JS (1), then CSS (2)
    priorities = [op.priority for op in plan.operations]
    assert priorities == [0, 1, 2]
    # CSS/JS depend on the HTML file
    css = next(op for op in plan.operations if op.file_path == "style.css")
    assert css.dependencies == ["index.html"]
    js = next(op for op in plan.operations if op.file_path == "app.js")
    assert js.dependencies == ["index.html"]
    assert plan.complexity == "moderate"  # 3 ops


# ---------------------------------------------------------------------------
# _plan_generic_feature
# ---------------------------------------------------------------------------


def test_plan_generic_feature_with_test_dir():
    structure = ProjectStructure(test_dir="tests")
    ops = _planner(
        RequestAnalysis(original_request="r"), structure
    )._plan_generic_feature("widget", structure)
    assert [o.file_path for o in ops] == ["widget.py", "tests/test_widget.py"]
    assert ops[1].dependencies == ["widget.py"]
    assert ops[1].priority == 1


def test_plan_generic_feature_no_test_dir():
    structure = ProjectStructure(test_dir=None)
    ops = _planner(
        RequestAnalysis(original_request="r"), structure
    )._plan_generic_feature("widget", structure)
    assert [o.file_path for o in ops] == ["widget.py"]


# ---------------------------------------------------------------------------
# _suggested_files_to_operations
# ---------------------------------------------------------------------------


def test_suggested_files_operations_feature_and_intent_descriptions():
    analysis = RequestAnalysis(
        original_request="req",
        intent="refactor",
        feature_name="",  # falsy -> intent-based description
        suggested_files=["a.py"],
        file_operations={"a.py": "modify"},
        needs_planning=True,
    )
    ops = _planner(analysis)._suggested_files_to_operations(analysis)
    assert len(ops) == 1
    assert ops[0].operation == "modify"
    assert "refactor" in ops[0].description
    assert "req" in ops[0].instructions


def test_suggested_files_feature_name_description():
    analysis = RequestAnalysis(
        original_request="req",
        intent="create_feature",
        feature_name="payments",
        suggested_files=["pay.py"],
        needs_planning=True,
    )
    ops = _planner(analysis)._suggested_files_to_operations(analysis)
    assert "payments" in ops[0].description


def test_suggested_files_plain_py_no_priority_no_deps():
    analysis = RequestAnalysis(
        original_request="req",
        intent="create_feature",
        feature_name=None,
        suggested_files=["util.py"],
        needs_planning=True,
    )
    ops = _planner(analysis)._suggested_files_to_operations(analysis)
    assert ops[0].priority == 0
    assert ops[0].dependencies == []


# ---------------------------------------------------------------------------
# _order_operations / _assess_complexity
# ---------------------------------------------------------------------------


def test_order_operations_empty():
    assert _planner(RequestAnalysis(original_request="r"))._order_operations([]) == []


def test_order_operations_dependency_respected():
    ops = [
        FileOperation(
            file_path="b.py", operation="create", description="", dependencies=["a.py"], priority=0
        ),
        FileOperation(file_path="a.py", operation="create", description="", priority=0),
    ]
    ordered = _planner(RequestAnalysis(original_request="r"))._order_operations(ops)
    assert [o.file_path for o in ordered] == ["a.py", "b.py"]


def test_order_operations_circular_dependency_breaks():
    ops = [
        FileOperation(file_path="a.py", operation="create", description="", dependencies=["b.py"]),
        FileOperation(file_path="b.py", operation="create", description="", dependencies=["a.py"]),
    ]
    ordered = _planner(RequestAnalysis(original_request="r"))._order_operations(ops)
    # circular -> remaining appended in order; both present
    assert {o.file_path for o in ordered} == {"a.py", "b.py"}


def test_assess_complexity_boundaries():
    p = _planner(RequestAnalysis(original_request="r"))
    assert p._assess_complexity([]) == "simple"
    assert (
        p._assess_complexity(
            [FileOperation(file_path="a", operation="create", description="")]
        )
        == "simple"
    )
    two = [
        FileOperation(file_path="a", operation="create", description=""),
        FileOperation(file_path="b", operation="create", description=""),
    ]
    assert p._assess_complexity(two) == "moderate"
    three = [*two, FileOperation(file_path="c", operation="create", description="")]
    assert p._assess_complexity(three) == "moderate"
    four = [*three, FileOperation(file_path="d", operation="create", description="")]
    assert p._assess_complexity(four) == "complex"


def test_define_success_criteria():
    ops = [FileOperation(file_path="a", operation="create", description="")]
    criteria = _planner(RequestAnalysis(original_request="r"))._define_success_criteria(ops)
    assert len(criteria) == 3
    assert "1 files" in criteria[0]


# ---------------------------------------------------------------------------
# LLMEnhancedMultiFilePlanner
# ---------------------------------------------------------------------------


def test_llm_enhanced_init(tmp_path):
    client = object()
    p = LLMEnhancedMultiFilePlanner(
        str(tmp_path), llm_client=client, llm_model="m", temperature=0.5
    )
    assert p.llm_client is client
    assert p.llm_model == "m"
    assert p.temperature == 0.5


class _StubClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def chat(self, messages, model, temperature, max_tokens):
        self.calls.append((model, temperature, max_tokens))
        if self._exc is not None:
            raise self._exc
        return self._response


class _SimpleResponse:
    content = None

    def __init__(self, text):
        self.content = text
        self.raw_response = {"choices": [{"message": {"content": text}}]}


_LLM_PLAN = """```yaml
plan:
  strategy: sequential
  complexity: moderate
  operations:
    - file_path: x.py
      operation: create
      description: Create x
```"""


def test_llm_create_plan_success():
    analysis = RequestAnalysis(
        original_request="add feature", intent="create_feature", needs_planning=True
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = _StubClient(response=_SimpleResponse(_LLM_PLAN))
    p.llm_model = "m"
    p.temperature = 0.0
    plan = p.create_plan("add feature")
    assert plan is not None
    assert [op.file_path for op in plan.operations] == ["x.py"]
    assert plan.complexity == "simple"  # reassessed from 1 op
    assert p.llm_client.calls and p.llm_client.calls[0][0] == "m"


def test_llm_create_plan_falls_back_when_client_none():
    analysis = RequestAnalysis(
        original_request="add feature", intent="create_feature", needs_planning=True
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = None
    plan = p.create_plan("add feature")
    # rule-based fallback produced generic feature module
    assert plan is not None
    assert plan.operations[0].file_path == "feature.py"


def test_llm_create_plan_falls_back_when_llm_returns_none():
    analysis = RequestAnalysis(
        original_request="add feature", intent="create_feature", needs_planning=True
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = _StubClient(response=_SimpleResponse("no plan here"))
    p.llm_model = "m"
    p.temperature = 0.0
    plan = p.create_plan("add feature")
    assert plan is not None and plan.operations[0].file_path == "feature.py"


def test_llm_create_plan_falls_back_on_exception():
    analysis = RequestAnalysis(
        original_request="add feature", intent="create_feature", needs_planning=True
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = _StubClient(exc=RuntimeError("boom"))
    p.llm_model = "m"
    p.temperature = 0.0
    plan = p.create_plan("add feature")
    assert plan is not None and plan.operations[0].file_path == "feature.py"


def test_llm_create_plan_skips_llm_when_not_needed():
    analysis = RequestAnalysis(
        original_request="hi", intent="general", needs_planning=False
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = _StubClient(response=_SimpleResponse(_LLM_PLAN))
    plan = p.create_plan("hi")
    assert p.llm_client.calls == []  # LLM never consulted
    assert plan.operations == []


def test_create_llm_based_plan_no_client():
    p = _planner(RequestAnalysis(original_request="r"), cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = None
    assert p._create_llm_based_plan("r", None, None) is None


def test_create_llm_based_plan_exception_returns_none():
    analysis = RequestAnalysis(original_request="r", needs_planning=True)
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = _StubClient(exc=RuntimeError("boom"))
    p.llm_model = "m"
    p.temperature = 0.0
    assert p._create_llm_based_plan("r", analysis, ProjectStructure()) is None


# ---------------------------------------------------------------------------
# _build_project_context_summary / _build_llm_planning_prompt
# ---------------------------------------------------------------------------


def test_build_project_context_summary_full():
    structure = ProjectStructure(
        framework="flask",
        frameworks=["flask", "sqlalchemy"],
        project_types=["web"],
        directories={"models": ["models/"], "other": ["misc/"]},
        naming_style="snake_case",
        common_imports=["flask", "os"],
        example_files={"model": "models/base.py", "view": "views/main.py"},
    )
    p = _planner(RequestAnalysis(original_request="r"), cls=LLMEnhancedMultiFilePlanner)
    summary = p._build_project_context_summary(structure)
    assert "flask, sqlalchemy" in summary
    assert "web" in summary
    assert "models" in summary
    assert "misc" not in summary  # purpose == 'other' excluded
    assert "snake_case" in summary
    assert "flask, os" in summary
    assert "models/base.py" in summary


def test_build_project_context_summary_framework_fallback():
    structure = ProjectStructure(framework="django", frameworks=[])
    p = _planner(RequestAnalysis(original_request="r"), cls=LLMEnhancedMultiFilePlanner)
    summary = p._build_project_context_summary(structure)
    assert "Framework**" in summary and "django" in summary


def test_build_project_context_summary_empty():
    p = _planner(RequestAnalysis(original_request="r"), cls=LLMEnhancedMultiFilePlanner)
    # naming_style default is "snake_case" (truthy) — pass None for a truly empty structure
    summary = p._build_project_context_summary(ProjectStructure(naming_style=None))
    assert summary == "No project context available."


def test_build_llm_planning_prompt_includes_context():
    structure = ProjectStructure(framework="fastapi")
    analysis = RequestAnalysis(
        original_request="req",
        intent="create_feature",
        feature_name="api",
        tech_stack=["python"],
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    prompt = p._build_llm_planning_prompt("req", analysis, structure, "CONTEXT")
    assert "CONTEXT" in prompt
    assert "create_feature" in prompt
    assert "api" in prompt
    assert "python" in prompt
    assert "fastapi" in prompt


# ---------------------------------------------------------------------------
# _parse_llm_plan_response extra branches
# ---------------------------------------------------------------------------


def test_parse_plan_without_yaml_backticks():
    """No ``` fences -> locate 'plan:' marker and parse from there."""
    text = (
        "Here is my plan:\n"
        "plan:\n"
        "  operations:\n"
        "    - file_path: nf.py\n"
        "      operation: create\n"
    )
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is not None
    assert plan.operations[0].file_path == "nf.py"


def test_parse_direct_operations_dict_without_plan_key():
    """Response shaped as {'operations': [...]} with no 'plan' key."""
    text = "```yaml\noperations:\n  - file_path: d.py\n    operation: create\n```"
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is not None
    assert plan.operations[0].file_path == "d.py"


def test_parse_skips_bad_yaml_then_accepts_good():
    """First block invalid YAML -> continue; second block valid -> plan."""
    text = (
        "```yaml\nplan: [unclosed\n```\n"
        "```yaml\nplan:\n  operations:\n    - file_path: ok.py\n      operation: create\n```"
    )
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is not None
    assert plan.operations[0].file_path == "ok.py"


def test_parse_plan_without_operations_returns_empty_plan():
    """plan exists but lacks operations -> empty ExecutionPlan (not None)."""
    text = "```yaml\nplan:\n  strategy: sequential\n```"
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is not None
    assert plan.operations == []


def test_parse_skips_operation_entries_with_non_dict_data():
    """op_data that is not a dict (e.g. null) -> exception -> continue -> None."""
    text = "```yaml\nplan:\n  operations:\n    - null\n```"
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is None


def test_parse_generic_error_returns_none(monkeypatch):
    """yaml.safe_load raising non-YAML exception -> outer except -> None."""
    import sys

    class _BoomYaml:
        safe_load = staticmethod(lambda text: (_ for _ in ()).throw(RuntimeError("load failed")))

    monkeypatch.setitem(sys.modules, "yaml", _BoomYaml)
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(_LLM_PLAN, "req")
    assert plan is None


def test_parse_phases_uses_file_fallback_key():
    """New format: op with 'file' key (no 'file_path') and template_file."""
    text = """```yaml
plan:
  phases:
    - phase: 1
      operations:
        - file: legacy.py
          operation: modify
          description: Legacy
          template_file: tpl.py
```"""
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is not None
    op = plan.operations[0]
    assert op.file_path == "legacy.py"
    assert op.template_file == "tpl.py"


def test_llm_create_plan_falls_back_when_context_build_raises(monkeypatch):
    """create_plan's own except (L378-379): exception escaping
    _create_llm_based_plan's inner try (context build runs before it)."""
    analysis = RequestAnalysis(
        original_request="add feature", intent="create_feature", needs_planning=True
    )
    p = _planner(analysis, cls=LLMEnhancedMultiFilePlanner)
    p.llm_client = _StubClient(response=_SimpleResponse(_LLM_PLAN))
    p.llm_model = "m"
    p.temperature = 0.0

    def _boom(structure):
        raise RuntimeError("context build failed")

    monkeypatch.setattr(p, "_build_project_context_summary", _boom)
    plan = p.create_plan("add feature")
    # fallback to rule-based generic feature plan
    assert plan is not None and plan.operations[0].file_path == "feature.py"


def test_parse_skips_non_plan_dict_without_operations():
    """parsed dict without 'plan' and without 'operations' -> continue -> None."""
    text = "```yaml\njust: a string\n```"
    plan = _planner(
        RequestAnalysis(original_request="req"), cls=LLMEnhancedMultiFilePlanner
    )._parse_llm_plan_response(text, "req")
    assert plan is None
