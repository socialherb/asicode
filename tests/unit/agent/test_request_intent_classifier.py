"""Unit tests for request_intent_classifier.py — 100% branch coverage."""
from external_llm.agent.request_intent_classifier import (
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
