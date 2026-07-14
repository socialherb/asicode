"""
Tests for intent_classifier.py — request intent analysis.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from external_llm.agent.execution_mode_classifier import (
    _LLM_MODE_ALIASES,
    ExecuteMode,
    _analyze_intent_with_keywords,
    _analyze_intent_with_llm_if_available,
    _has_number_after_keyword,
    analyze_request_for_optimal_mode,
    validate_instruction_target_file,
)

# ── ExecuteMode ───────────────────────────────────────────────────────────

class TestExecuteMode:
    """Tests for ExecuteMode enum with fuzzy matching.

    Note: _missing_ handles case, hyphen/underscore normalization, and
    leading/trailing whitespace.  Compound aliases like "planjson" → "plan_json"
    are handled only in _LLM_MODE_ALIASES, not in _missing_.
    """

    def test_exact_match(self):
        assert ExecuteMode("normal") == ExecuteMode.NORMAL

    def test_case_insensitive(self):
        assert ExecuteMode("NORMAL") == ExecuteMode.NORMAL
        assert ExecuteMode("Strict_JSON") == ExecuteMode.STRICT_JSON

    def test_hyphen_to_underscore(self):
        assert ExecuteMode("strict-json") == ExecuteMode.STRICT_JSON
        assert ExecuteMode("plan-json") == ExecuteMode.PLAN_JSON

    def test_spaces_to_underscore(self):
        assert ExecuteMode("plan json") == ExecuteMode.PLAN_JSON
        assert ExecuteMode("strict json") == ExecuteMode.STRICT_JSON

    def test_leading_trailing_whitespace(self):
        assert ExecuteMode("  normal  ") == ExecuteMode.NORMAL

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            ExecuteMode("unknown_mode")

    def test_non_string_value_raises(self):
        with pytest.raises(ValueError):
            ExecuteMode(42)

    @pytest.mark.parametrize("value,expected", [
        ("intelligent", ExecuteMode.INTELLIGENT),
        ("INTELLIGENT", ExecuteMode.INTELLIGENT),
        ("Intelligent", ExecuteMode.INTELLIGENT),
        ("LEGACY", ExecuteMode.LEGACY),
        ("plan_json", ExecuteMode.PLAN_JSON),
        ("plan json", ExecuteMode.PLAN_JSON),
        ("intelligent", ExecuteMode.INTELLIGENT),
    ])
    def test_various_modes(self, value, expected):
        assert ExecuteMode(value) == expected


# ── _LLM_MODE_ALIASES ────────────────────────────────────────────────────

class TestLLMModeAliases:
    """Test that all aliases map to correct ExecuteMode."""

    def test_all_aliases_are_valid(self):
        for alias, mode in _LLM_MODE_ALIASES.items():
            assert isinstance(alias, str)
            assert isinstance(mode, ExecuteMode)

    def test_all_modes_represented(self):
        modes_in_aliases = set(_LLM_MODE_ALIASES.values())
        all_modes = set(ExecuteMode)
        for m in all_modes:
            assert m in modes_in_aliases, f"{m} missing from aliases"
            # The canonical .value string MUST be an exact-lookup key: the primary
            # classification path (_LLM_MODE_ALIASES.get(cleaned)) matches on the
            # lowercased LLM output, so an LLM emitting exactly the mode's value
            # (e.g. "strict_json") must resolve. A mode reachable only via a
            # non-canonical alias key would silently fail exact lookup and rely on
            # the fuzzy \b{alias}\b fallback — which also can't match a key it
            # doesn't have. This guards against alias-key/value drift when a new
            # mode is added (e.g. typo'd key "strct_json" with correct value).
            assert m.value in _LLM_MODE_ALIASES, (
                f"{m}.value={m.value!r} is not a key in _LLM_MODE_ALIASES; "
                f"the exact-lookup path would miss the canonical LLM output"
            )


# ── _has_number_after_keyword ─────────────────────────────────────────────

class TestHasNumberAfterKeyword:
    """Tests for _has_number_after_keyword."""

    @pytest.mark.parametrize("text,keywords,expected", [
        # Standard cases
        ("fix line 42", ("line",), True),
        ("go to line42", ("line",), True),
        ("modify line  42 now", ("line",), True),
        ("no line reference", ("line",), False),
        # Keyword embedded in word but followed by a number → matches
        ("online 42", ("line",), True),
        ("multi line", ("line",), False),
        # Korean keywords
        ("라인 42에서 수정", ("라인",), True),
        ("줄 5로 이동", ("줄",), True),
        # Edge cases
        ("no keyword at all", ("line", "라인", "줄"), False),
        ("", ("line",), False),
        ("line", ("line",), False),
        ("line   ", ("line",), False),
        ("line42extra", ("line",), True),
        ("prefix_line 42", ("line",), True),
    ])
    def test_various_inputs(self, text, keywords, expected):
        assert _has_number_after_keyword(text, keywords) == expected


# ── _analyze_intent_with_keywords ─────────────────────────────────────────

class TestAnalyzeIntentWithKeywords:
    """Tests for keyword-based intent analysis fallback."""

    def test_legacy_explicit(self):
        assert _analyze_intent_with_keywords("use legacy mode", None) == "legacy"

    def test_legacy_case_insensitive(self):
        assert _analyze_intent_with_keywords("LEGACY please", None) == "legacy"

    def test_line_number_strict_json(self):
        assert _analyze_intent_with_keywords("change line 42", None) == "strict_json"

    @pytest.mark.slow
    def test_line_number_with_hangul(self):
        assert _analyze_intent_with_keywords("라인 42 수정", None) == "strict_json"
        assert _analyze_intent_with_keywords("줄 5 변경", None) == "strict_json"

    def test_legacy_beats_line_number(self):
        """'legacy' keyword takes priority over line number."""
        assert _analyze_intent_with_keywords("legacy mode on line 42", None) == "legacy"

    def test_default_normal(self):
        assert _analyze_intent_with_keywords("fix the bug in foo()", None) == "normal"

    def test_empty_prompt(self):
        assert _analyze_intent_with_keywords("", None) == "normal"


# ── validate_instruction_target_file ──────────────────────────────────────

class TestValidateInstructionTargetFile:
    """Tests for validate_instruction_target_file."""

    def test_matching_target(self):
        validate_instruction_target_file({"target_file": "src/main.py"}, "src/main.py")

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="target_file mismatch"):
            validate_instruction_target_file({"target_file": "src/other.py"}, "src/main.py")

    def test_no_instruction_target(self):
        validate_instruction_target_file({"mode": "edit"}, "src/main.py")

    def test_no_expected_target(self):
        validate_instruction_target_file({"target_file": "src/main.py"}, "")

    def test_both_empty(self):
        validate_instruction_target_file({}, "")

    def test_instruction_target_none(self):
        validate_instruction_target_file({"target_file": None}, "src/main.py")

    def test_whitespace_expected(self):
        validate_instruction_target_file({"target_file": "src/main.py"}, "  ")


# ── _analyze_intent_with_llm_if_available ─────────────────────────────────

class TestAnalyzeIntentWithLLM:
    """Tests for LLM-based intent analysis with mocked external service.

    Since create_intelligent_service_from_env is imported inside the function
    body, we patch at the external module path.
    """

    def _make_mock_service(self, content_value: str, mock_create: MagicMock) -> MagicMock:
        """Set up a standard mock service chain and return the service mock."""
        mock_service = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = content_value
        mock_client.chat.return_value = mock_response
        mock_client.timeout = 120
        mock_service.llm_service.client = mock_client
        mock_service.model = "gpt-4"
        mock_create.return_value = mock_service
        return mock_service

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env", return_value=None)
    def test_no_service_returns_none(self, mock_create):
        result = _analyze_intent_with_llm_if_available("test prompt", None)
        assert result is None
        mock_create.assert_called_once()

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_successful_analysis(self, mock_create):
        self._make_mock_service("strict_json", mock_create)
        result = _analyze_intent_with_llm_if_available("add a comment on line 42", "foo.py")
        assert result == ExecuteMode.STRICT_JSON

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_llm_returns_normal(self, mock_create):
        self._make_mock_service("normal", mock_create)
        result = _analyze_intent_with_llm_if_available("fix a typo", None)
        assert result == ExecuteMode.NORMAL

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_llm_returns_intelligent(self, mock_create):
        self._make_mock_service("intelligent", mock_create)
        result = _analyze_intent_with_llm_if_available("add login feature", None)
        assert result == ExecuteMode.INTELLIGENT

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_fuzzy_alias_in_response(self, mock_create):
        """LLM response contains alias text with punctuation."""
        self._make_mock_service("I think plan-json is best.", mock_create)
        result = _analyze_intent_with_llm_if_available("create a module", None)
        assert result == ExecuteMode.PLAN_JSON

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_word_boundary_matches_genuine_alias(self, mock_create):
        """Word-boundary match still routes when alias appears in commentary."""
        self._make_mock_service("the optimal mode is normal here", mock_create)
        result = _analyze_intent_with_llm_if_available("fix a typo", None)
        assert result == ExecuteMode.NORMAL

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_word_boundary_rejects_embedded_substring(self, mock_create):
        """Substring 'normal' inside 'abnormal' must NOT route to NORMAL.

        Regression for the pre-fix substring fallback (alias in response_text)
        which over-matched inflected/embedded forms.
        """
        self._make_mock_service("this request is abnormal", mock_create)
        result = _analyze_intent_with_llm_if_available("weird request", None)
        assert result is None

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_unknown_response_returns_none(self, mock_create):
        self._make_mock_service("something completely different", mock_create)
        result = _analyze_intent_with_llm_if_available("test", None)
        assert result is None

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_exception_falls_through(self, mock_create):
        mock_create.side_effect = RuntimeError("service creation failed")
        result = _analyze_intent_with_llm_if_available("test", None)
        assert result is None

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_timeout_restored_after_use(self, mock_create):
        mock_service = self._make_mock_service("normal", mock_create)
        _analyze_intent_with_llm_if_available("test", None)
        assert mock_service.llm_service.client.timeout == 120  # restored

    @patch("external_llm.intelligent_service.create_intelligent_service_from_env")
    def test_empty_response_content(self, mock_create):
        self._make_mock_service("", mock_create)
        result = _analyze_intent_with_llm_if_available("test", None)
        assert result is None


# ── analyze_request_for_optimal_mode ──────────────────────────────────────

class TestAnalyzeRequestForOptimalMode:
    """Tests for top-level analyze_request_for_optimal_mode."""

    @patch("external_llm.agent.execution_mode_classifier._analyze_intent_with_llm_if_available", return_value=ExecuteMode.STRICT_JSON)
    def test_llm_takes_priority(self, mock_llm):
        result = analyze_request_for_optimal_mode("test", "foo.py")
        assert result == "strict_json"
        mock_llm.assert_called_once()

    @patch("external_llm.agent.execution_mode_classifier._analyze_intent_with_llm_if_available", return_value=None)
    @patch("external_llm.agent.execution_mode_classifier._analyze_intent_with_keywords", return_value="normal")
    def test_falls_back_to_keywords(self, mock_kw, mock_llm):
        result = analyze_request_for_optimal_mode("fix bug", None)
        assert result == "normal"
        mock_kw.assert_called_once_with("fix bug", None)
