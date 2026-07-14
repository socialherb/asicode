"""Tests for edit_localization.request_analyzer — code identifier extraction only.

Action/role detection has been removed; only code_identifiers extraction remains.
"""

from external_llm.edit_localization.request_analyzer import (
    analyze_request,
)


class TestAnalyzeRequest:
    """Test request semantic analysis — code identifier extraction only."""

    def test_code_identifiers_backtick(self):
        sem = analyze_request("Change `is_async` to use a different check")
        assert "is_async" in sem.code_identifiers

    def test_code_identifiers_dotted(self):
        sem = analyze_request("Update `node.kind` to include new types")
        assert "node.kind" in sem.code_identifiers

    def test_code_identifiers_snake_case_bare(self):
        sem = analyze_request("rename `process_data` to `handle_data`")
        assert "process_data" in sem.code_identifiers
        assert "handle_data" in sem.code_identifiers

    def test_code_identifiers_mixed(self):
        sem = analyze_request("fix the bug in calculate_total")
        assert "calculate_total" in sem.code_identifiers

    def test_korean_with_code_identifier(self):
        sem = analyze_request("please add the end_line field")
        assert "end_line" in sem.code_identifiers

    def test_no_code_identifiers(self):
        sem = analyze_request("show me this code")
        assert not sem.code_identifiers

    def test_actions_always_empty(self):
        """Action detection has been removed — actions is always empty."""
        sem = analyze_request("please unify kind")
        assert not sem.actions

    def test_target_roles_always_empty(self):
        """Role detection has been removed — target_roles is always empty."""
        sem = analyze_request("please separate out the async flag")
        assert not sem.target_roles

    def test_natural_language_no_false_identifiers(self):
        """Natural language words should not be extracted as code identifiers."""
        sem = analyze_request("please refactor this function")
        assert "function" not in sem.code_identifiers

    def test_single_quoted_identifier(self):
        sem = analyze_request("Change 'is_async' to a different check")
        assert "is_async" in sem.code_identifiers
