"""Regression tests for multi_planner YAML null-collection handling.

LLMs frequently emit bare YAML keys like ``dependencies:`` or ``success_criteria:``
which parse to ``None``. ``dict.get(k, default)`` returns ``None`` (not the
default) when the key *exists* with a null value, so a single null field used to
crash ``_order_operations`` (``for dep in None`` -> TypeError) and silently
discard an otherwise-valid LLM plan.
"""
from external_llm.multi_planner import (
    FileOperation,
    LLMEnhancedMultiFilePlanner,
)


def _planner():
    """Build an LLMEnhanced planner without the heavy __init__ analyzers.

    We unit-test the pure parse/order logic, which does not touch the analyzer
    or project_analyzer attributes.
    """
    return LLMEnhancedMultiFilePlanner.__new__(LLMEnhancedMultiFilePlanner)


_LLM_NULL_DEPS = """```yaml
plan:
  strategy: sequential
  complexity: moderate
  phases:
    - phase: 1
      operations:
        - file_path: a.py
          operation: create
          dependencies:
        - file_path: b.py
          operation: create
          dependencies: null
  success_criteria:
  warnings: null
```"""


def test_parse_null_dependencies_coerced_to_list():
    """`dependencies: null` must coerce to [] instead of propagating None."""
    plan = _planner()._parse_llm_plan_response(_LLM_NULL_DEPS, "req")
    assert plan is not None, "valid plan was discarded due to a null field"
    assert all(isinstance(op.dependencies, list) for op in plan.operations)
    # null top-level collections also coerced, not None
    assert plan.success_criteria == []
    assert plan.warnings == []


def test_order_operations_handles_null_dependencies_without_crash():
    """_order_operations must not raise on a None dependencies value.

    Defense-in-depth: even a FileOperation constructed elsewhere with
    dependencies=None must not crash the topological sort.
    """
    p = _planner()
    plan = p._parse_llm_plan_response(_LLM_NULL_DEPS, "req")
    ordered = p._order_operations(plan.operations)
    assert [o.file_path for o in ordered] == ["a.py", "b.py"]

    # Direct guard: stray None dependencies
    stray = [FileOperation(file_path="x.py", operation="create",
                           description="", dependencies=None)]
    ordered2 = p._order_operations(stray)
    assert len(ordered2) == 1 and ordered2[0].file_path == "x.py"


class _ReasoningOnlyResponse:
    """GLM-5.2 (thinking ON) / DeepSeek Reasoner drift shape: the final answer
    lands in ``reasoning_content`` while top-level ``content`` is empty."""

    def __init__(self, plan_text: str):
        self.content = ""
        self.raw_response = {
            "choices": [{"message": {"reasoning_content": plan_text}}]
        }


class _StubClient:
    def __init__(self, response):
        self._response = response
        self.last_model = "unset"

    def chat(self, messages, model, temperature, max_tokens):
        self.last_model = model
        return self._response


def test_create_llm_based_plan_recovers_from_reasoning_content(monkeypatch):
    """Plan must parse when the answer arrives only in reasoning_content.

    Regression for the effective_content() fallback (commit ac6c138d): reading
    ``response.content`` directly returned '' here, parse failed, and the
    planner silently downgraded to rule-based.
    """
    p = _planner()
    p.llm_client = _StubClient(_ReasoningOnlyResponse(_LLM_NULL_DEPS))
    p.llm_model = "glm-5.2"
    p.temperature = 0.0
    monkeypatch.setattr(p, "_build_project_context_summary", lambda s: "")
    monkeypatch.setattr(p, "_build_llm_planning_prompt", lambda *a: "prompt")

    plan = p._create_llm_based_plan("req", analysis=None, structure=None)
    assert plan is not None, "reasoning_content-only response was discarded"
    assert [op.file_path for op in plan.operations] == ["a.py", "b.py"]
    # No stale hardcoded fallback: the configured model must reach the client.
    assert p.llm_client.last_model == "glm-5.2"


def test_parse_old_format_null_dependencies():
    """The non-phases (old-format) branch must also coerce null dependencies."""
    llm = """```yaml
plan:
  operations:
    - file_path: c.py
      operation: create
      dependencies:
```"""
    plan = _planner()._parse_llm_plan_response(llm, "req")
    assert plan is not None
    assert all(isinstance(op.dependencies, list) for op in plan.operations)


def test_parse_returns_none_when_pyyaml_missing(monkeypatch):
    """PyYAML is an optional [config] extra; its absence must degrade to
    None (rule-based fallback), never propagate an ImportError.

    Regression: ``import yaml`` previously sat *outside* the try block, making
    the ``except ImportError`` handler dead code and relying on the caller's
    broad ``except Exception`` as the real safety net.
    """
    import sys

    # sys.modules[name] = None forces ``import name`` to raise ImportError,
    # simulating "PyYAML not installed" without touching the environment.
    monkeypatch.setitem(sys.modules, "yaml", None)
    plan = _planner()._parse_llm_plan_response(_LLM_NULL_DEPS, "req")
    assert plan is None, "missing PyYAML must degrade to None, not raise"
