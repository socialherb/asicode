"""Tests for task_router — DeterministicClassifier routing logic."""

from unittest.mock import MagicMock, patch

import pytest

from external_llm.agent.enums import Complexity, Scope
from external_llm.agent.task_router import (
    DeterministicClassifier,
    Lane,
    RouteDecision,
    RouteFeatures,
    TaskKind,
    TaskRouter,
)


class TestDeterministicClassifier:
    """Tests for DeterministicClassifier — core routing logic."""

    def setup_method(self):
        self.clf = DeterministicClassifier()

    def _make_features(self, **overrides) -> RouteFeatures:
        """Build a RouteFeatures with sensible defaults and per-test overrides."""
        f = RouteFeatures()
        f.request = "test request"
        f.request_lower = "test request"
        f.word_count = 2
        f.has_edit_intent = True
        f.has_read_intent = False
        f.has_explicit_file = False
        f.has_explicit_symbol = False
        f.has_specific_change_object = False
        f.is_multi_file = False
        f.is_project_wide = False
        f.all_targets_non_structured = False
        f.target_specificity_score = 0.5
        f.complexity = Complexity.LOW
        f.scope = Scope.SINGLE_FILE
        f.task_kind = TaskKind.SINGLE_FILE_EDIT
        f.mentioned_files = []
        f.symbol_count = 0
        f.file_count = 0
        f.has_conflicting_intent = False
        f.has_cross_file_signal = False
        f.has_propagation_signal = False
        f.looks_trivial_edit = False
        f.requests_test_work = False
        f.requests_refactor = False
        f.requests_style_change = False
        f.requests_ui_change = False
        f.requests_filesystem_op = False
        f.requests_boilerplate = False
        f.has_anchor_or_exact_target = False
        f.has_question_form = False
        f.has_explain_intent = False
        f.has_locate_intent = False
        f.readonly_kind = None
        for k, v in overrides.items():
            setattr(f, k, v)
        return f

    # ── classify / decide_flow ─────────────────────────────────────────

    def test_classify_planner_default(self):
        """Default route should be MAIN_AGENT (PLANNER lane disabled by Tier 3 consolidation)."""
        clf = DeterministicClassifier()
        f = self._make_features(has_edit_intent=True)
        decision = clf.decide_flow(f)
        assert decision.lane == Lane.MAIN_AGENT
        assert decision.requires_planner is False

    def test_classify_main_agent_non_structured(self):
        """Non-structured targets route to MAIN_AGENT."""
        clf = DeterministicClassifier()
        f = self._make_features(all_targets_non_structured=True)
        decision = clf.decide_flow(f)
        assert decision.lane == Lane.MAIN_AGENT

    def test_classify_main_agent_low_specificity(self):
        """Ambiguous edit with low specificity routes to MAIN_AGENT."""
        clf = DeterministicClassifier()
        f = self._make_features(
            target_specificity_score=0.2,
            has_specific_change_object=False,
            has_read_intent=False,
        )
        decision = clf.decide_flow(f)
        assert decision.lane == Lane.MAIN_AGENT

    # ── _compute_confidence ────────────────────────────────────────────

    def test_compute_confidence_default(self):
        assert DeterministicClassifier._compute_confidence(
            self._make_features()
        ) == 0.70

    def test_compute_confidence_with_file(self):
        conf = DeterministicClassifier._compute_confidence(
            self._make_features(has_explicit_file=True)
        )
        assert conf == pytest.approx(0.78)

    def test_compute_confidence_all_penalties(self):
        conf = DeterministicClassifier._compute_confidence(
            self._make_features(
                is_multi_file=True, is_project_wide=True,
                has_conflicting_intent=True, all_targets_non_structured=True,
            )
        )
        assert conf == 0.55  # clamped to min 0.55

    # ── _build_main_agent_reason ───────────────────────────────────────

    def test_build_main_agent_reason_default(self):
        reason = DeterministicClassifier._build_main_agent_reason(
            self._make_features(has_edit_intent=False, has_specific_change_object=True)
        )
        assert reason == "MAIN_AGENT default path"

    def test_build_main_agent_reason_non_structured(self):
        reason = DeterministicClassifier._build_main_agent_reason(
            self._make_features(
                all_targets_non_structured=True, has_edit_intent=True,
            )
        )
        assert "non-structured targets" in reason
        assert "edit request" in reason

    def test_build_main_agent_reason_ui_change(self):
        reason = DeterministicClassifier._build_main_agent_reason(
            self._make_features(requests_ui_change=True)
        )
        assert "UI/style change" in reason

    def test_build_main_agent_reason_filesystem(self):
        reason = DeterministicClassifier._build_main_agent_reason(
            self._make_features(requests_filesystem_op=True)
        )
        assert "filesystem operation" in reason

    # ── _classify_task_meta ────────────────────────────────────────────

    def test_classify_task_meta_micro_edit(self):
        tk, _cx, _sc = self.clf._classify_task_meta(
            self._make_features(looks_trivial_edit=True)
        )
        assert tk == TaskKind.MICRO_EDIT

    def test_classify_task_meta_style_fix(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features(requests_style_change=True)
        )
        assert tk == TaskKind.STYLE_FIX

    def test_classify_task_meta_refactor(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features(requests_refactor=True)
        )
        assert tk == TaskKind.REFACTOR

    def test_classify_task_meta_test_write(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features(requests_test_work=True)
        )
        assert tk == TaskKind.TEST_WRITE

    def test_classify_task_meta_boilerplate(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features(requests_boilerplate=True)
        )
        assert tk == TaskKind.BOILERPLATE

    def test_classify_task_meta_exploration(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features(
                has_edit_intent=False, has_read_intent=True,
            )
        )
        assert tk == TaskKind.EXPLORATION

    def test_classify_task_meta_multi_file(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features(is_multi_file=True)
        )
        assert tk == TaskKind.MULTI_FILE_FEATURE

    def test_classify_task_meta_single_file(self):
        tk, _, _ = self.clf._classify_task_meta(
            self._make_features()
        )
        assert tk == TaskKind.SINGLE_FILE_EDIT

    def test_classify_complexity_high(self):
        _, cx, _ = self.clf._classify_task_meta(
            self._make_features(file_count=5)
        )
        assert cx == Complexity.HIGH

    def test_classify_complexity_medium(self):
        _, cx, _ = self.clf._classify_task_meta(
            self._make_features(is_multi_file=True)
        )
        assert cx == Complexity.MEDIUM

    def test_classify_complexity_vague_refactor_is_high(self):
        """Refactor request with NO files named but substantial length (>=30
        words) is underspecified/riskier → HIGH.

        Regression: this escalation branch was previously shadowed by the
        ``word_count > 25`` (MEDIUM) guard and was therefore unreachable —
        such requests silently became MEDIUM instead of HIGH.
        """
        _, cx, _ = self.clf._classify_task_meta(
            self._make_features(requests_refactor=True, file_count=0, word_count=40)
        )
        assert cx == Complexity.HIGH

    def test_classify_complexity_refactor_with_files_not_escalated(self):
        """A refactor request that NAMES files (file_count >= 1) does NOT
        trigger the vague-refactor escalation — it follows the generic ladders.
        Guards against over-triggering the new first branch."""
        # file_count=1, word_count=40: vague-refactor branch skipped (needs 0 files);
        # not >=3 files, not >60 words, not multifile, not >=2 files, but
        # word_count > 25 → MEDIUM.
        _, cx, _ = self.clf._classify_task_meta(
            self._make_features(requests_refactor=True, file_count=1, word_count=40)
        )
        assert cx == Complexity.MEDIUM

    def test_classify_scope_project_wide(self):
        _, _, sc = self.clf._classify_task_meta(
            self._make_features(is_project_wide=True)
        )
        assert sc == Scope.PROJECT_WIDE

    def test_classify_scope_single_file(self):
        _, _, sc = self.clf._classify_task_meta(
            self._make_features(is_multi_file=False)
        )
        assert sc == Scope.SINGLE_FILE

    # ── _build_main_agent_decision ─────────────────────────────────────

    def test_build_main_agent_decision(self):
        clf = DeterministicClassifier()
        f = self._make_features(
            task_kind=TaskKind.SINGLE_FILE_EDIT,
            complexity=Complexity.LOW,
            scope=Scope.SINGLE_FILE,
        )
        d = clf._build_main_agent_decision(f)
        assert d.lane == Lane.MAIN_AGENT
        assert d.planning_enabled is False
        assert d.requires_planner is False

    # ── extract_features (structural extraction only) ─────────────────

    def test_extract_features_basic(self):
        clf = DeterministicClassifier()
        f = clf.extract_features("Fix the bug in utils.py")
        assert f.word_count == 5
        assert "utils.py" in f.mentioned_files
        assert f.file_count >= 1
        assert f.has_explicit_file is True
        assert f.has_edit_intent is True

    def test_extract_features_question(self):
        clf = DeterministicClassifier()
        f = clf.extract_features("How does this work?")
        assert f.has_question_form is True

    def test_extract_features_css_selector(self):
        clf = DeterministicClassifier()
        f = clf.extract_features("Change .header color to red")
        assert f.has_anchor_or_exact_target is True

    def test_extract_features_no_intent_fallback(self):
        """Without IntentResult, has_edit_intent defaults to True."""
        clf = DeterministicClassifier()
        f = clf.extract_features("What is this?", intent_result=None)
        assert f.has_edit_intent is True
        assert f.has_read_intent is False


class TestTaskRouter:
    """Tests for TaskRouter — integration + lane defaults."""

    @patch("external_llm.agent.task_router.create_intent_resolver")
    def test_apply_lane_defaults_planner(self, mock_resolver):
        mock_resolver.return_value = MagicMock()
        router = TaskRouter()
        d = RouteDecision(
            task_kind=TaskKind.SINGLE_FILE_EDIT,
            complexity=Complexity.LOW,
            scope=Scope.SINGLE_FILE,
            lane=Lane.PLANNER,
            confidence=0.7,
            reasoning="test",
        )
        d = router._apply_lane_defaults(d)
        assert d.planning_enabled is True
        assert d.self_review_enabled is True
        assert d.rag_enabled is True

    @patch("external_llm.agent.task_router.create_intent_resolver")
    def test_apply_lane_defaults_main_agent(self, mock_resolver):
        mock_resolver.return_value = MagicMock()
        router = TaskRouter()
        d = RouteDecision(
            task_kind=TaskKind.SINGLE_FILE_EDIT,
            complexity=Complexity.LOW,
            scope=Scope.SINGLE_FILE,
            lane=Lane.MAIN_AGENT,
            confidence=0.7,
            reasoning="test",
        )
        d = router._apply_lane_defaults(d)
        assert d.planning_enabled is False
        assert d.rag_enabled is False

    @patch("external_llm.agent.task_router.create_intent_resolver")
    def test_apply_lane_defaults_preserves_explicit(self, mock_resolver):
        """Explicitly set fields should NOT be overwritten by defaults."""
        mock_resolver.return_value = MagicMock()
        router = TaskRouter()
        d = RouteDecision(
            task_kind=TaskKind.SINGLE_FILE_EDIT,
            complexity=Complexity.LOW,
            scope=Scope.SINGLE_FILE,
            lane=Lane.PLANNER,
            confidence=0.7,
            reasoning="test",
            planning_enabled=False,  # explicitly False
        )
        d = router._apply_lane_defaults(d)
        assert d.planning_enabled is False  # preserved

    @patch("external_llm.agent.task_router.create_intent_resolver")
    def test_route_without_llm_client(self, mock_resolver):
        """TaskRouter can be created without llm_client."""
        mock_resolver.return_value = MagicMock()
        router = TaskRouter()
        assert router._deterministic is not None
