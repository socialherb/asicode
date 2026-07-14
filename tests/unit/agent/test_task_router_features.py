"""Golden routing test set for MAIN_AGENT routing.

All requests go to MAIN_AGENT (PLANNER lane disabled by Tier 3 consolidation).
The LLM tool loop handles all tasks directly. No keyword-based gate routing.
"""
import pytest

from external_llm.agent.task_router import (
    DeterministicClassifier,
    Lane,
    Scope,
    TaskRouter,
)


@pytest.fixture
def dc():
    return DeterministicClassifier()


# ── All requests → PLANNER ──────────────────────────────────────────────

class TestUnifiedPlannerRouting:
    """Verify DeterministicClassifier routing for each request type.

    Note: IntentResult (LLM-derived) is NOT provided here, so routing falls back
    to structural signals. Requests that are ambiguous, non-structured,
    filesystem-op, or read-only route to MAIN_AGENT by design.
    """

    @pytest.mark.parametrize("req,expected_lane", [
        # Bug fix requests
        ("버그 수정해줘", Lane.MAIN_AGENT),  # ambiguous, no file/symbol target
        ("fix the authentication bug", Lane.MAIN_AGENT),  # ambiguous
        ("login 함수에 JWT 검증 추가해줘", Lane.MAIN_AGENT),  # no structured file target
        # SWE-bench style issue reports
        ("Modeling separability_matrix does not compute separability correctly", Lane.MAIN_AGENT),  # symbol detected
        ("Set default FILE_UPLOAD_PERMISSION to 0o644", Lane.MAIN_AGENT),  # config-style, ambiguous
        ("HttpResponse doesnt handle memoryview objects", Lane.MAIN_AGENT),  # symbol detected
        ("delete() on instances of models without any dependencies doesnt clear PKs", Lane.MAIN_AGENT),  # ambiguous
        # Trivial edits (formerly FAST_PATH) — has explicit file → MAIN_AGENT
        ("main.py에 TODO 주석 한 줄 추가", Lane.MAIN_AGENT),
        ("foo.py에 import os 추가", Lane.MAIN_AGENT),
        # CSS/HTML edits — non-structured targets → MAIN_AGENT
        ("styles.css에서 color 값만 #fff로 변경", Lane.MAIN_AGENT),
        ("change the button color to blue", Lane.MAIN_AGENT),
        # Filesystem ops
        ("파일을 tests/로 이동해줘", Lane.MAIN_AGENT),  # no explicit structured target
        ("move auth.py to utils/", Lane.MAIN_AGENT),  # file mentioned
        # Read/explain requests
        ("explain how the auth middleware works", Lane.MAIN_AGENT),  # ambiguous, no target
        ("이 코드 설명해줘", Lane.MAIN_AGENT),  # ambiguous
        ("what does this function do?", Lane.MAIN_AGENT),  # question form → read intent
        ("라우팅 구조 설명해줘", Lane.MAIN_AGENT),  # ambiguous Korean
        # Mixed intent
        ("이 코드 분석해서 버그 고쳐줘", Lane.MAIN_AGENT),  # ambiguous, no target
        # Ambiguous (formerly CLARIFY)
        ("버그 잡아줘", Lane.MAIN_AGENT),
        ("fix this", Lane.MAIN_AGENT),
        # Refactoring
        ("라우팅 구조를 단순화해줘", Lane.MAIN_AGENT),  # ambiguous, no explicit target
        ("agent_loop.py의 종료 조건 리팩토링해줘", Lane.MAIN_AGENT),  # file mentioned
        # Complex
        ("함수명 바꾸고 호출부 전부 업데이트해줘", Lane.MAIN_AGENT),  # no explicit target
    ])
    def test_all_requests_go_to_planner(self, dc, req, expected_lane):
        result = dc.classify(req)
        assert result.lane == expected_lane, (
            f'"{req}" → {result.lane.value} (expected {expected_lane.value})'
        )


# ── Feature extraction tests (still useful for SpecResolver hints) ──────

class TestFeatureExtraction:
    """Verify feature extraction produces correct structural signals.
    Features are still extracted for SpecResolver hints, not for routing."""

    @pytest.fixture(autouse=True)
    def setup_intent(self, monkeypatch):
        """Provide IntentResult for propagation/cross-file signal tests.
        These signals are LLM-derived, not regex-derived."""
        from unittest.mock import MagicMock

        self.intent = MagicMock()
        self.intent.intent_type = "edit"
        self.intent.confidence = 0.8
        self.intent.target_symbols = []
        self.intent.target_files = []
        self.intent.is_filesystem_op = False
        self.intent.is_ui_change = False
        self.intent.is_style_fix = False
        self.intent.is_test_write = False
        self.intent.lane_hint = None
        self.intent.scope_hint = None
        self.intent.complexity_hint = None
        def _reset_intent():
            self.intent.scope_hint = None
        self._reset_intent = _reset_intent
    """Verify feature extraction produces correct structural signals.
    Features are still extracted for SpecResolver hints, not for routing."""

    def test_file_extraction(self, dc):
        f = dc.extract_features("main.py와 utils.py 수정해줘")
        assert f.has_explicit_file
        assert f.file_count == 2
        assert f.is_multi_file

    def test_symbol_extraction(self, dc):
        f = dc.extract_features("UserService 클래스 수정해줘")
        assert f.has_explicit_symbol
        assert f.symbol_count > 0

    def test_korean_symbol_detection(self, dc):
        # Korean-only symbol names are not detected by the current symbol pattern
        f = dc.extract_features("함수명 바꿔줘")
        assert not f.has_explicit_symbol  # intent: Korean-only symbol names don't match the pattern

    def test_propagation_signal(self, dc):
        # Without an IntentResult, propagation_signal is not detected (LLM-derived)
        f = dc.extract_features("함수명 바꾸고 호출부 전부 업데이트해줘")
        assert not f.has_propagation_signal
        assert not f.is_multi_file

        # Activated when IntentResult passes scope_hint=PROJECT_WIDE
        self.intent.scope_hint = Scope.PROJECT_WIDE
        f2 = dc.extract_features("함수명 바꾸고 호출부 전부 업데이트해줘", intent_result=self.intent)
        assert f2.has_propagation_signal
        assert f2.is_multi_file

    def test_cross_file_signal(self, dc):
        # Without an IntentResult, cross_file_signal is not detected (LLM-derived)
        f = dc.extract_features("모든 곳에서 이 함수를 변경해줘")
        assert not f.has_cross_file_signal

        # Activated when IntentResult passes scope_hint=MULTI_FILE
        self.intent.scope_hint = Scope.MULTI_FILE
        f2 = dc.extract_features("모든 곳에서 이 함수를 변경해줘", intent_result=self.intent)
        assert f2.has_cross_file_signal

    def test_specificity_score(self, dc):
        f1 = dc.extract_features("main.py에 있는 login 함수 수정해줘")
        f2 = dc.extract_features("수정해줘")
        assert f1.target_specificity_score > f2.target_specificity_score


# ── TaskRouter integration ────────────────────────────────────────────────

class TestTaskRouterIntegration:
    """Verify TaskRouter.route() always returns MAIN_AGENT (PLANNER disabled)."""

    @pytest.fixture(autouse=True)
    def patch_intent_resolver(self, monkeypatch):
        """Patch create_intent_resolver to avoid needing a real model/LLM."""
        from unittest.mock import MagicMock
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = MagicMock(
            intent_kind="edit", confidence=0.8, search_terms=[]
        )
        monkeypatch.setattr(
            "external_llm.agent.task_router.create_intent_resolver",
            lambda **kwargs: mock_resolver,
        )

    def test_route_returns_main_agent(self):
        router = TaskRouter()
        decision = router.route("login 함수에 JWT 추가해줘")
        assert decision.lane == Lane.MAIN_AGENT

    def test_trivial_edit_goes_to_main_agent(self):
        router = TaskRouter()
        decision = router.route("main.py에 주석 한 줄 추가")
        assert decision.lane == Lane.MAIN_AGENT

    def test_read_request_goes_to_main_agent(self):
        router = TaskRouter()
        decision = router.route("explain how auth works")
        assert decision.lane == Lane.MAIN_AGENT

    def test_ambiguous_request_goes_to_main_agent(self):
        router = TaskRouter()
        decision = router.route("fix this")
        assert decision.lane == Lane.MAIN_AGENT

    def test_swe_bench_style_goes_to_main_agent(self):
        router = TaskRouter()
        decision = router.route(
            "Fix the following bug:\n\n"
            "UsernameValidator allows trailing newline in usernames\n"
            "Description\n"
            "ASCIIUsernameValidator and UnicodeUsernameValidator use the regex..."
        )
        assert decision.lane == Lane.MAIN_AGENT
