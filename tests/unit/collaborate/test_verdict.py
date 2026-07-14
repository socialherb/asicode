"""
Tests for CollaborationVerdict.
"""
from __future__ import annotations

from external_llm.repl.collaborate import CollaborationVerdict


class TestCollaborationVerdict:
    """Verify the verdict dataclass and its utilities."""

    def test_default_creation(self):
        v = CollaborationVerdict()
        assert v.status == "needs_review"
        assert v.summary == ""
        assert v.confidence == 0.5
        assert v.suggestions == []
        assert v.plan is None

    def test_is_success(self):
        assert CollaborationVerdict(status="success").is_success()
        assert not CollaborationVerdict(status="failure").is_success()

    def test_is_failure(self):
        assert CollaborationVerdict(status="failure").is_failure()
        assert not CollaborationVerdict(status="success").is_failure()

    def test_needs_review(self):
        assert CollaborationVerdict(status="needs_review").needs_review()
        assert not CollaborationVerdict(status="success").needs_review()

    def test_to_dict(self):
        v = CollaborationVerdict(
            status="success",
            summary="All good",
            details="Everything passed.",
            confidence=0.95,
            suggestions=["Add tests"],
            plan={"steps": ["refactor"]},
        )
        d = v.to_dict()
        assert d["status"] == "success"
        assert d["summary"] == "All good"
        assert d["confidence"] == 0.95
        assert d["suggestions"] == ["Add tests"]
        assert d["plan"] == {"steps": ["refactor"]}

    def test_from_result_message_dict(self):
        data = {
            "status": "success",
            "summary": "Analysis complete",
            "details": "No issues found.",
            "confidence": 0.9,
            "suggestions": ["Deploy"],
        }
        v = CollaborationVerdict.from_result_message(data)
        assert v.status == "success"
        assert v.summary == "Analysis complete"
        assert v.confidence == 0.9

    def test_from_result_message_empty(self):
        v = CollaborationVerdict.from_result_message({})
        assert v.status == "needs_review"
        assert v.confidence == 0.5

    def test_from_result_message_confidence_str_normalized(self):
        # The structured_candidate path skips output_format validation, so
        # confidence can arrive as a string. Values like '0.8' must be coerced to float.
        v = CollaborationVerdict.from_result_message({"confidence": "0.8"})
        assert v.confidence == 0.8

    def test_from_result_message_confidence_invalid_falls_back(self):
        v = CollaborationVerdict.from_result_message({"confidence": "high"})
        assert v.confidence == 0.5

    def test_from_result_message_confidence_clamped(self):
        v = CollaborationVerdict.from_result_message({"confidence": 1.5})
        assert v.confidence == 1.0
        v = CollaborationVerdict.from_result_message({"confidence": -0.3})
        assert v.confidence == 0.0

    def test_from_result_message_unknown_status_normalized(self):
        v = CollaborationVerdict.from_result_message({"status": "weird"})
        assert v.status == "needs_review"

    def test_from_result_message_suggestions_coerced_to_list(self):
        # If suggestions arrives as a non-list, coerce to a single-element list with a str element
        v = CollaborationVerdict.from_result_message({"suggestions": "fix it"})
        assert v.suggestions == ["fix it"]
        v = CollaborationVerdict.from_result_message({"suggestions": [1, 2]})
        assert v.suggestions == ["1", "2"]

    def test_from_result_message_summary_details_coerced_to_str(self):
        v = CollaborationVerdict.from_result_message({"summary": 42, "details": None})
        assert v.summary == "42"
        assert v.details == "None"

    def test_post_init_normalizes_confidence_on_direct_construction(self):
        # __post_init__ protects **every** construction path, not just
        # from_result_message. The literal-float sites in claude_session.py were
        # already safe, but paths like orchestrator's
        # format_verdict_for_session(v.confidence:.0%), which format the dataclass
        # object directly without going through from_result_message, are still
        # safe because __post_init__ normalizes them.
        assert CollaborationVerdict(confidence="0.8").confidence == 0.8
        assert CollaborationVerdict(confidence="high").confidence == 0.5
        assert CollaborationVerdict(confidence=1.5).confidence == 1.0
        assert CollaborationVerdict(confidence=-0.3).confidence == 0.0
        assert CollaborationVerdict(confidence=None).confidence == 0.5
        # A float that is already valid should be left unchanged
        assert CollaborationVerdict(confidence=0.42).confidence == 0.42

    def test_output_format_schema(self):
        schema = CollaborationVerdict.output_format_schema()
        assert schema["type"] == "object"
        assert "status" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "details" in schema["properties"]
        assert schema["required"] == ["status", "summary", "details"]

        status_prop = schema["properties"]["status"]
        assert "success" in status_prop["enum"]
        assert "failure" in status_prop["enum"]
        assert "needs_review" in status_prop["enum"]
        assert "insufficient_info" in status_prop["enum"]
