"""Unit tests for request_intent_classifier.py — 100% branch coverage."""
import pytest

from external_llm.agent.request_intent_classifier import (
    intent_is_undetermined,
    is_non_edit_intent,
    normalize_routing_label,
    routing_intent_from_intent_result,
)

# ── normalize_routing_label ──────────────────────────────────────────────────

class TestNormalizeRoutingLabel:
    """Cover all branches of normalize_routing_label."""

    def test_already_canonical(self):
        assert normalize_routing_label("read_only") == "read_only"

    def test_hyphen_to_underscore(self):
        assert normalize_routing_label("read-only") == "read_only"

    def test_no_space(self):
        assert normalize_routing_label("readonly") == "read_only"

    def test_spaces(self):
        assert normalize_routing_label("read only") == "read_only"

    def test_camelcase(self):
        assert normalize_routing_label("ReadOnly") == "read_only"

    def test_explore_and_edit_canonical(self):
        assert normalize_routing_label("explore_and_edit") == "explore_and_edit"

    def test_explore_and_edit_variation(self):
        assert normalize_routing_label("explore-and-edit") == "explore_and_edit"

    def test_explore_and_edit_no_sep(self):
        assert normalize_routing_label("exploreandedit") == "explore_and_edit"

    def test_planner_label_maps_to_explore_and_edit(self):
        """IntentResolver lane_hint 'planner' is an edit intent."""
        assert normalize_routing_label("planner") == "explore_and_edit"

    def test_main_agent_label_maps_to_explore_and_edit(self):
        """IntentResolver lane_hint 'main_agent' is an edit intent."""
        assert normalize_routing_label("main_agent") == "explore_and_edit"

    def test_bugfix_label_maps_to_explore_and_edit(self):
        """IntentResolver intent_type 'bugfix' is an edit intent."""
        assert normalize_routing_label("bugfix") == "explore_and_edit"

    def test_feature_label_maps_to_explore_and_edit(self):
        assert normalize_routing_label("feature") == "explore_and_edit"

    def test_refactor_label_maps_to_explore_and_edit(self):
        assert normalize_routing_label("refactor") == "explore_and_edit"

    def test_exploration_label_maps_to_explore_and_edit(self):
        assert normalize_routing_label("exploration") == "explore_and_edit"

    def test_modify_label_maps_to_explore_and_edit(self):
        assert normalize_routing_label("modify") == "explore_and_edit"

    def test_extend_label_maps_to_explore_and_edit(self):
        assert normalize_routing_label("extend") == "explore_and_edit"

    def test_create_label_maps_to_explore_and_edit(self):
        assert normalize_routing_label("create") == "explore_and_edit"

    def test_question_label_is_recognized_no_drift(self, caplog):
        """'question' is a valid intent_type; must NOT trigger LABEL_DRIFT."""
        import logging
        caplog.set_level(logging.WARNING)
        result = normalize_routing_label("question")
        assert result == "question"
        assert "LABEL_DRIFT" not in caplog.text

    def test_unknown_sentinel_is_recognized_no_drift(self, caplog):
        """'unknown' is the IntentResult default / classification-failure sentinel
        (intent_models.py:41; emitted by every IntentResolver failure path). It is
        an internal first-class sentinel, NOT LLM-output drift, and must NOT trigger
        LABEL_DRIFT. Maps to explore_and_edit (fail-open: never block a legit edit)."""
        import logging
        caplog.set_level(logging.WARNING)
        result = normalize_routing_label("unknown")
        assert result == "explore_and_edit"
        assert "LABEL_DRIFT" not in caplog.text

    def test_unrecognized_label_passthrough(self, caplog):
        """Unrecognized label is logged as drift warning and returned as-is."""
        import logging
        caplog.set_level(logging.WARNING)
        result = normalize_routing_label("unknown_label")
        assert result == "unknown_label"
        assert "LABEL_DRIFT" in caplog.text

    def test_unrecognized_still_lowercased_but_returned(self, caplog):
        """Lowercased version not in dict either → passed through."""
        import logging
        caplog.set_level(logging.WARNING)
        result = normalize_routing_label("Some_Unknown")
        assert result == "Some_Unknown"  # returned as-is (original case preserved)
        assert "LABEL_DRIFT" in caplog.text


# ── is_non_edit_intent ───────────────────────────────────────────────────────

class TestIsNonEditIntent:
    def test_read_only_is_non_edit(self):
        assert is_non_edit_intent("read_only") is True

    def test_explore_and_edit_is_edit(self):
        assert is_non_edit_intent("explore_and_edit") is False

    def test_clarification_needed_is_non_edit(self):
        assert is_non_edit_intent("clarification_needed") is True


# ── routing_intent_from_intent_result ────────────────────────────────────────

class FakeIntentResult:
    def __init__(self, lane_hint="", intent_type=""):
        self.lane_hint = lane_hint
        self.intent_type = intent_type


class TestRoutingIntentFromIntentResult:
    """Cover all branches of routing_intent_from_intent_result."""

    def test_none_intent_result(self):
        assert routing_intent_from_intent_result(None) == "explore_and_edit"

    def test_lane_hint_read_only(self):
        result = FakeIntentResult(lane_hint="read_only")
        assert routing_intent_from_intent_result(result) == "read_only"

    def test_intent_type_question(self):
        result = FakeIntentResult(intent_type="question")
        assert routing_intent_from_intent_result(result) == "read_only"

    def test_lane_hint_clarify(self):
        result = FakeIntentResult(lane_hint="clarify")
        assert routing_intent_from_intent_result(result) == "explore_and_edit"

    def test_default_explore_and_edit(self):
        result = FakeIntentResult(lane_hint="", intent_type="")
        assert routing_intent_from_intent_result(result) == "explore_and_edit"

    def test_empty_attrs_fallback_to_explore(self):
        result = FakeIntentResult()
        assert routing_intent_from_intent_result(result) == "explore_and_edit"

    def test_normalized_variation_lane_hint(self):
        """lane_hint='read-only' (hyphen) normalizes to 'read_only'."""
        result = FakeIntentResult(lane_hint="read-only")
        assert routing_intent_from_intent_result(result) == "read_only"

    def test_normalized_variation_intent_type(self):
        """intent_type='question' (lowercase) is recognized as read_only."""
        result = FakeIntentResult(intent_type="question")
        assert routing_intent_from_intent_result(result) == "read_only"

    def test_classification_failure_routes_to_explore_and_edit(self, caplog):
        """End-to-end contract: a real IntentResolver failure path produces
        IntentResult(intent_type='unknown', lane_hint='planner') — see
        intent_resolver.py:796/811/825 and intent_models.py:41. This must route to
        explore_and_edit (fail-open: never block a legitimate edit) and must NOT
        emit LABEL_DRIFT, since 'unknown' is a first-class internal sentinel."""
        import logging
        caplog.set_level(logging.WARNING)
        result = FakeIntentResult(lane_hint="planner", intent_type="unknown")
        assert routing_intent_from_intent_result(result) == "explore_and_edit"
        assert "LABEL_DRIFT" not in caplog.text


# ── intent_is_undetermined ───────────────────────────────────────────────────

class TestIntentIsUndetermined:
    """The permission default (explore_and_edit) must stay separable from the
    question 'did classification actually happen?'."""

    def test_none_is_undetermined(self):
        """No IntentResult attached at all — nothing was classified."""
        assert intent_is_undetermined(None) is True

    @pytest.mark.parametrize("source", [
        "minimal_fallback",   # LLM call raised / no client
        "llm_parse_failed",   # response was not parseable JSON
        "empty_request",      # nothing to classify
    ])
    def test_resolver_failure_sources_are_undetermined(self, source):
        result = FakeIntentResult(lane_hint="planner", intent_type="unknown")
        result.metadata = {"source": source}
        assert intent_is_undetermined(result) is True
        # …while routing still fails OPEN so a legitimate edit is never blocked.
        assert routing_intent_from_intent_result(result) == "explore_and_edit"

    def test_real_classification_is_determined(self):
        result = FakeIntentResult(lane_hint="planner", intent_type="bugfix")
        result.metadata = {"source": "llm"}
        assert intent_is_undetermined(result) is False

    def test_real_unknown_classification_is_determined(self):
        """intent_type='unknown' from a SUCCESSFUL resolve is not a failure —
        the sentinel alone cannot distinguish the two, the source can."""
        result = FakeIntentResult(lane_hint="planner", intent_type="unknown")
        result.metadata = {"source": "llm"}
        assert intent_is_undetermined(result) is False

    def test_missing_metadata_is_determined(self):
        """No metadata attribute → no evidence of failure, treat as classified."""
        assert intent_is_undetermined(FakeIntentResult(intent_type="feature")) is False

    def test_matches_real_resolver_failure_shape(self):
        """Contract test against the real IntentResolver, not a fake: whatever
        it emits when its LLM call raises must be recognised as undetermined."""
        from external_llm.agent.intent_resolver import (
            IntentResolutionConfig,
            IntentResolver,
        )

        class _Boom:
            def chat(self, **kw):
                raise RuntimeError("intent LLM unreachable")

        resolver = IntentResolver(IntentResolutionConfig(llm_client=_Boom(), model="x"))
        result = resolver.resolve("이 함수 뭐 하는 건지 설명해줘")
        assert intent_is_undetermined(result) is True
