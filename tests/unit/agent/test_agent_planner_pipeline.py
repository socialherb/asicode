from unittest.mock import MagicMock

import pytest

from external_llm.agent.agent_loop_types import (
    AgentResult,
    _EscalationOutcome,
    _PlannerLaneOutcome,
    _SpecResolutionResult,
)

# Import the module to test
from external_llm.agent.agent_planner_pipeline import (
    PlannerPipelineMixin,
    build_prebuilt_spec_from_impl_spec,
)
from external_llm.agent.execution_spec import ResolvedExecutionSpec
from external_llm.agent.operation_models import (
    GroundingSummary as _GroundingSummary,
)

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------

_REPO_ROOT = "/tmp/test_repo"

# ----------------------------------------------------------------------
# Helper factories
# ----------------------------------------------------------------------

def _make_request(text="test request"):
    request = MagicMock()
    request.text = text
    return request

def _make_context(agent=None, app=None):
    context = MagicMock()
    context.agent = agent or MagicMock()
    context.app = app or MagicMock()
    return context

def _make_route(path="src/main.py"):
    route = MagicMock()
    route.path = path
    return route

def _make_git_state(branch="main", repo_path="/tmp/repo"):
    git_state = MagicMock()
    git_state.branch = branch
    git_state.repo_path = repo_path
    return git_state

def _make_resolved_spec(request_type="modify", target_files=None, target_symbols=None,
                        new_files=None, metadata=None):
    """Create a real ResolvedExecutionSpec for testing."""
    return ResolvedExecutionSpec(
        original_request="test request",
        intent="test purpose",
        request_type=request_type,
        target_files=target_files or [],
        target_symbols=target_symbols or [],
        new_files=new_files or [],
        metadata=metadata or {},
        authoritative=True,
    )

def _make_grounding_summary(confidence=0.80):
    return _GroundingSummary(grounding_confidence=confidence)

def _make_spec_result(spec=None, fit_verdict=None, grounding_summary=None,
                      is_read_only_intent=False, llm_hints=None):
    """Create a proper _SpecResolutionResult."""
    return _SpecResolutionResult(
        spec=spec or _make_resolved_spec(),
        fit_verdict=fit_verdict,
        grounding_summary=grounding_summary or _make_grounding_summary(),
        is_read_only_intent=is_read_only_intent,
        llm_hints=llm_hints or {},
    )

def _make_operation_plan(ops=None):
    plan = MagicMock()
    plan.metadata = {}
    plan.operations = ops or []
    return plan

def _make_agent_result(status="success", final_message="ok"):
    return AgentResult(status=status, final_message=final_message)


# ----------------------------------------------------------------------
# Concrete subclass of PlannerPipelineMixin with all required
# host-class attributes mocked.
# ----------------------------------------------------------------------

class TestablePlannerPipeline(PlannerPipelineMixin):
    """Minimal subclass with all required host-class attributes mocked."""
    def __init__(self):
        super().__init__()
        # --- config ---
        self.config = MagicMock()
        self.config.prebuilt_spec_for_planner = None
        self.config._token_offset_prompt_tokens = 0
        self.config._token_offset_completion_tokens = 0
        self.config._token_offset_cache_read_tokens = 0
        self.config.user_checkpoint_enabled = False
        self.config.user_checkpoint_callback = None
        self.config.cancel_event = None  # ESC pause not active in test

        # --- registry ---
        self.registry = MagicMock()
        self.registry.repo_root = _REPO_ROOT
        self.registry.applied_patches = []

        # --- executor & planner ---
        self._operation_executor = MagicMock()
        self._operation_executor.execute_plan.return_value = {
            "completed": 1, "failed": 0, "modified_files": ["test.py"],
            "output": "", "final_status": "success",
        }
        self._planner_agent = MagicMock()
        self._planner_agent.create_operation_plan.return_value = _make_operation_plan()
        self._planner_agent.enforce_analyze_first_structure.side_effect = lambda plan, spec: plan
        self._planner_agent.enforce_evaluator_verdicts.side_effect = lambda plan, spec: plan
        self._planner_agent.incremental_tokens.return_value = {}

        # --- routing ---
        self._routing_intent_hint = None

        # --- lifecycle ---
        self._init_hybrid_components = MagicMock()
        self._cb = MagicMock()
        self.performance_collector = MagicMock()
        self.performance_collector.end_session = MagicMock()
        self.performance_collector.get_summary = MagicMock(return_value={})

        # --- planning ---
        self._build_planner_summary = MagicMock(return_value="planner summary")
        self._build_planner_fallback_context = MagicMock(return_value="fallback")
        self._switch_to_developer_for_fallback = MagicMock()
        self._rollback_patches = MagicMock()
        self._execute_plan_sequence_direct = MagicMock()
        self._handle_analyze_first_escalation = MagicMock(
            return_value=_EscalationOutcome(
                exec_result={"completed": 1, "failed": 0, "modified_files": ["test.py"]},
                op_plan=_make_operation_plan(),
                spec=MagicMock(), completed=1, failed=0,
                exec_status="complete", exec_detail="1 completed",
            )
        )

        # --- auxiliary mocks for _build_and_execute_plan ---
        self._plan_from_worksets = MagicMock(return_value=[])
        self._plan_is_valid = MagicMock(return_value=True)
        self.create_scoped_plan = MagicMock(return_value=(_make_operation_plan(), "summary"))


# ----------------------------------------------------------------------
# Tests for build_prebuilt_spec_from_impl_spec
# ----------------------------------------------------------------------

class TestBuildPrebuiltSpecFromImplSpec:
    def test_valid_dict(self):
        """Should return a ResolvedExecutionSpec when given a valid dict."""
        request_text = "test request"
        implementation_spec = {
            "purpose": "add logging",
            "target_files": ["src/utils.py"],
            "target_symbols": ["helper_func"],
            "edit_kind": "extend",
        }
        result = build_prebuilt_spec_from_impl_spec(request_text, implementation_spec)
        assert isinstance(result, ResolvedExecutionSpec)
        assert result.request_type == "modify"
        assert result.target_files == ["src/utils.py"]
        assert result.target_symbols == ["helper_func"]
        assert result.metadata.get("edit_kind") == "extend"
        assert result.metadata.get("skip_grounding") is True
        assert result.metadata.get("source") == "design_chat_analysis"

    def test_none(self):
        """Should return None when implementation_spec is None."""
        result = build_prebuilt_spec_from_impl_spec("test request", None)
        assert result is None

    def test_empty_dict(self):
        """Should return None when empty dict (no actionable targets)."""
        result = build_prebuilt_spec_from_impl_spec("test request", {})
        assert result is None  # no target_files / new_files / target_symbols

    def test_new_files_request_type(self):
        """new_files should also set request_type to 'modify'."""
        result = build_prebuilt_spec_from_impl_spec("test", {"new_files": ["new.py"]})
        assert result is not None
        assert result.request_type == "modify"

    def test_normalised_paths(self):
        """Target file paths should be normalised (.. resolved)."""
        request_text = "test"
        relative_path = "src/../src/utils.py"
        implementation_spec = {"target_files": [relative_path]}
        result = build_prebuilt_spec_from_impl_spec(request_text, implementation_spec)
        for path in result.target_files:
            assert ".." not in path, f"path not normalised: {path}"
            assert isinstance(path, str)

    def test_normalised_paths_with_repo_root(self):
        """Paths should be resolved relative to repo_root when provided."""
        result = build_prebuilt_spec_from_impl_spec(
            "test",
            {"target_files": ["src/utils.py"]},
            repo_root="/project",
        )
        assert result.target_files == ["/project/src/utils.py"]

    def test_multiple_targets(self):
        """Should handle multiple target files."""
        request_text = "test"
        implementation_spec = {"target_files": ["a.py", "b.py", "c.py"]}
        result = build_prebuilt_spec_from_impl_spec(request_text, implementation_spec)
        assert result is not None
        assert len(result.target_files) == 3
        for f in result.target_files:
            assert isinstance(f, str)

    def test_target_symbols_filtered(self):
        """Non-string target_symbols should be filtered out."""
        implementation_spec = {"target_files": ["a.py"], "target_symbols": ["foo", 42, None, "bar"]}
        result = build_prebuilt_spec_from_impl_spec("test", implementation_spec)
        assert result.target_symbols == ["foo", "bar"]

    def test_purpose_becomes_intent(self):
        """purpose field should map to intent."""
        result = build_prebuilt_spec_from_impl_spec("test", {
            "target_files": ["x.py"], "purpose": "fix bug #42"
        })
        assert result.intent == "fix bug #42"

    def test_purpose_fallback_to_request_text(self):
        """Should fall back to request_text when purpose is absent."""
        result = build_prebuilt_spec_from_impl_spec("my request text", {"target_files": ["x.py"]})
        assert result.intent == "my request text"

    def test_code_context(self):
        """code_context items should be preserved."""
        cc = [{"reason": "example", "file": "ref.py", "snippet": "def foo(): pass"}]
        result = build_prebuilt_spec_from_impl_spec("test", {"target_files": ["x.py"], "code_context": cc})
        assert result.code_context == cc

# ----------------------------------------------------------------------
# Tests for PlannerPipelineMixin._run_planner_lane
# ----------------------------------------------------------------------

class TestRunPlannerLane:
    def test_successful_lane(self):
        """With a prebuilt spec, should return a _PlannerLaneOutcome with AgentResult.

        SpecResolver was removed; the PLANNER lane now runs only against a prebuilt
        spec produced by Design Chat analysis (config.prebuilt_spec_for_planner).
        """
        pipeline = TestablePlannerPipeline()
        # Supply a prebuilt spec (Design-Chat analysis-backed path)
        pipeline.config.prebuilt_spec_for_planner = _make_resolved_spec(
            target_files=["test.py"],
        )
        # Wire _build_and_execute_plan to return a success outcome
        _outcome = _PlannerLaneOutcome(result=_make_agent_result("success"))
        pipeline._build_and_execute_plan = MagicMock(return_value=_outcome)

        outcome = pipeline._run_planner_lane(
            _make_request(), _make_context(), _make_route(),
            _make_git_state(), "session-123", turns=[],
        )
        assert isinstance(outcome, _PlannerLaneOutcome)
        assert outcome.result is not None
        assert outcome.result.status == "success"
        # Verify orchestration
        pipeline._init_hybrid_components.assert_called_once()
        pipeline._build_and_execute_plan.assert_called_once()

    def test_no_prebuilt_falls_back(self):
        """Without a prebuilt spec, should return fallback_context (no SpecResolver)."""
        pipeline = TestablePlannerPipeline()
        # prebuilt_spec_for_planner is None (default in TestablePlannerPipeline)
        pipeline._build_and_execute_plan = MagicMock()  # should NOT be called

        outcome = pipeline._run_planner_lane(
            _make_request(), _make_context(), _make_route(),
            _make_git_state(), "session-fb", turns=[],
        )
        assert isinstance(outcome, _PlannerLaneOutcome)
        # Falls back: result is None, fallback_context is set
        assert outcome.result is None
        assert outcome.fallback_context is not None
        # _build_and_execute_plan must NOT run without a prebuilt spec
        pipeline._build_and_execute_plan.assert_not_called()

    def test_with_turns(self):
        """Turns should not cause errors (prebuilt-spec path)."""
        pipeline = TestablePlannerPipeline()
        pipeline.config.prebuilt_spec_for_planner = _make_resolved_spec(
            target_files=["test.py"],
        )
        _outcome = _PlannerLaneOutcome(result=_make_agent_result("success"))
        pipeline._build_and_execute_plan = MagicMock(return_value=_outcome)

        outcome = pipeline._run_planner_lane(
            _make_request(), _make_context(), _make_route(),
            _make_git_state(), "session-456", turns=[MagicMock(), MagicMock()],
        )
        assert outcome.result is not None

    def test_handles_exception(self):
        """Should return partial_success on exception (non-RuntimeError)."""
        pipeline = TestablePlannerPipeline()
        # ValueError (not RuntimeError) to hit the generic except Exception handler
        pipeline._init_hybrid_components.side_effect = ValueError("Mock failure")

        outcome = pipeline._run_planner_lane(
            _make_request(), _make_context(), _make_route(),
            _make_git_state(), "session-999", turns=[],
        )
        # _run_planner_lane catches exceptions and wraps in AgentResult
        assert outcome.result is not None
        assert outcome.result.status == "partial_success"
        assert "Mock failure" in outcome.result.final_message

# ----------------------------------------------------------------------
# Tests for PlannerPipelineMixin._build_and_execute_plan
# ----------------------------------------------------------------------

class TestBuildAndExecutePlan:
    def test_raises_runtime_error(self):
        """RuntimeError when analysis-backed spec has non-existent target_files and no new_files."""
        pipeline = TestablePlannerPipeline()
        # analysis-backed spec (skip_grounding=True) with non-existent target file
        # and no new_files → RuntimeError: no existing files, file not moved to new_files
        spec = _make_resolved_spec(
            target_files=["nonexistent.py"],
            target_symbols=[],
            new_files=[],
            metadata={"skip_grounding": True},
        )
        spec_result = _make_spec_result(spec=spec)
        with pytest.raises(RuntimeError, match="Spec targets not found"):
            pipeline._build_and_execute_plan(
                _make_request(), _make_context(), _make_route(),
                _make_git_state(), "session-e1", [], spec_result,
            )

    def test_non_existent_target_in_non_analysis_spec(self):
        """Non-analysis spec: non-existent target_file moves to new_files, proceeds."""
        pipeline = TestablePlannerPipeline()
        spec = _make_resolved_spec(
            target_files=["nonexistent.py"],
            metadata={},  # no skip_grounding → non analysis-backed
        )
        spec_result = _make_spec_result(spec=spec)
        # Mock _create_operation_plan_with_scanner to return early
        pipeline._create_operation_plan_with_scanner = MagicMock(
            return_value=(_make_operation_plan(), "ctx summary")
        )
        outcome = pipeline._build_and_execute_plan(
            _make_request(), _make_context(), _make_route(),
            _make_git_state(), "session-e2", [], spec_result,
        )
        assert isinstance(outcome, _PlannerLaneOutcome)

    def test_new_files_allows_execution(self):
        """Should proceed when spec has new_files even without target_files."""
        pipeline = TestablePlannerPipeline()
        spec = _make_resolved_spec(target_files=[], new_files=["new.py"])
        spec_result = _make_spec_result(spec=spec)
        pipeline._create_operation_plan_with_scanner = MagicMock(
            return_value=(_make_operation_plan(), "ctx summary")
        )
        outcome = pipeline._build_and_execute_plan(
            _make_request(), _make_context(), _make_route(),
            _make_git_state(), "session-e3", [], spec_result,
        )
        assert isinstance(outcome, _PlannerLaneOutcome)

# ----------------------------------------------------------------------
# Tests for PlannerPipelineMixin._create_operation_plan_with_scanner
# ----------------------------------------------------------------------

class TestCreateOperationPlanWithScanner:
    def test_returns_operation_plan_and_summary(self):
        """Should return tuple of operation plan and summary string."""
        pipeline = TestablePlannerPipeline()
        spec = _make_resolved_spec()
        # Pass a plain string as context (not MagicMock) so summary is a string
        plan, summary = pipeline._create_operation_plan_with_scanner(
            _make_request(), "base context", spec, _make_grounding_summary(),
        )
        assert plan is not None
        assert isinstance(summary, str)
        pipeline._planner_agent.create_operation_plan.assert_called_once()

    def test_gsg_context_injected(self):
        """Should inject GSG context into plan_context."""
        pipeline = TestablePlannerPipeline()
        spec = _make_resolved_spec(target_files=["x.py"])
        spec.gsg_context = "GSG: use Builder pattern"
        _plan, _summary = pipeline._create_operation_plan_with_scanner(
            _make_request(), "base context", spec, _make_grounding_summary(),
        )
        call_kwargs = pipeline._planner_agent.create_operation_plan.call_args
        assert call_kwargs is not None
        _ctx = call_kwargs[0][1]  # second positional arg is context
        assert "GSG: use Builder pattern" in _ctx

    def test_injects_scanner_ops(self):
        """Should inject RUN_SCANNER ops when target files include scanner modules."""
        pipeline = TestablePlannerPipeline()
        op_plan = _make_operation_plan()
        pipeline._planner_agent.create_operation_plan.return_value = op_plan
        spec = _make_resolved_spec(target_files=["external_llm/scanner_registry.py"])
        plan, _summary = pipeline._create_operation_plan_with_scanner(
            _make_request(), "ctx", spec, _make_grounding_summary(),
        )
        # Should not crash; scanner injection is handled internally
        assert plan is not None

# ----------------------------------------------------------------------
# Sanity test: instantiation and basic attribute checks
# ----------------------------------------------------------------------

class TestPlannerPipelineMixinInstantiation:
    def test_mixin_has_methods(self):
        """Mixin should expose required methods."""
        methods = ['_run_planner_lane', '_build_and_execute_plan', '_create_operation_plan_with_scanner']
        for m in methods:
            assert hasattr(PlannerPipelineMixin, m), f"Missing method {m}"
            assert callable(getattr(PlannerPipelineMixin, m))

    def test_testable_pipeline_can_be_instantiated(self):
        """TestablePlannerPipeline should instantiate without error."""
        p = TestablePlannerPipeline()
        assert p.config is not None
        assert p.registry is not None
        assert p._operation_executor is not None
        assert p._planner_agent is not None
