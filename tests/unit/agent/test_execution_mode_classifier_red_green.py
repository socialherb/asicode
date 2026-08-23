"""RED→GREEN: execution_mode_classifier — 모드 분류 전 경로 고정.

ExecuteMode._missing_ 퍼지 매칭, 키워드 분석, LLM 의존 경로(서비스
부재/성공/퍼지/예외), semantic backstop, target_file 검증을 커버한다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar  # f821-protected

import pytest

import external_llm.agent.execution_mode_classifier as emc
from external_llm.agent.execution_mode_classifier import (
    _LINE_REFERENCE_KEYWORDS,
    ExecuteMode,
    _analyze_intent_with_keywords,
    _get_mode_matcher,
    _has_digit,
    _has_number_after_keyword,
    analyze_request_for_optimal_mode,
    validate_instruction_target_file,
)

# ── ExecuteMode._missing_ ───────────────────────────────────────────────


class TestExecuteModeMissing:
    def test_case_and_separator_normalized(self):
        assert ExecuteMode("Strict_JSON") is ExecuteMode.STRICT_JSON
        assert ExecuteMode("strict-json") is ExecuteMode.STRICT_JSON
        assert ExecuteMode("PLAN JSON") is ExecuteMode.PLAN_JSON

    def test_unknown_string_raises_value_error(self):
        with pytest.raises(ValueError, match="bogus"):
            ExecuteMode("bogus")

    def test_non_string_value_raises_value_error(self):
        with pytest.raises(ValueError, match="42"):
            ExecuteMode(42)


# ── 헬퍼 ────────────────────────────────────────────────────────────────


class TestHasDigit:
    def test_digit_present(self):
        assert _has_digit("line 42")
        assert _has_digit("3rd")

    def test_no_digit(self):
        assert not _has_digit("fix the bug")
        assert not _has_digit("")


class TestHasNumberAfterKeyword:
    def test_space_form(self):
        assert _has_number_after_keyword("edit line 42", _LINE_REFERENCE_KEYWORDS)

    def test_glued_form(self):
        assert _has_number_after_keyword("modify row10 now", _LINE_REFERENCE_KEYWORDS)

    def test_korean_keyword(self):
        assert _has_number_after_keyword("줄 42에 주석", _LINE_REFERENCE_KEYWORDS)

    def test_embedded_keyword_does_not_shadow_later_reference(self):
        # "delineate" 안의 "line"은 후속 "line 42"를 가리지 않는다
        assert _has_number_after_keyword("delineate then line 42", _LINE_REFERENCE_KEYWORDS)

    def test_embedded_keyword_alone_is_false(self):
        assert not _has_number_after_keyword("delineate the thing", _LINE_REFERENCE_KEYWORDS)

    def test_number_not_directly_after_keyword_is_false(self):
        assert not _has_number_after_keyword("line at 42", _LINE_REFERENCE_KEYWORDS)

    def test_multiple_occurrences_scanned(self):
        # 첫 "line" 뒤는 숫자가 아니지만 두 번째 "line 42"에서 발견
        assert _has_number_after_keyword("line at 5 then line 42", _LINE_REFERENCE_KEYWORDS)


# ── _get_mode_matcher 싱글턴 ────────────────────────────────────────────


class _StubMatcherClass:
    instances: ClassVar[list] = []

    def __init__(self, examples, threshold, margin, name):
        self.examples, self.threshold, self.margin, self.name = examples, threshold, margin, name
        _StubMatcherClass.instances.append(self)

    def matches(self, text, label):
        return False


@pytest.fixture(autouse=True)
def _reset_matcher_global():
    emc._mode_matcher = None
    _StubMatcherClass.instances = []
    yield
    emc._mode_matcher = None


class TestGetModeMatcher:
    def test_builds_once_and_caches(self, monkeypatch):
        monkeypatch.setattr("external_llm.agent.semantic_intent.SemanticIntentMatcher", _StubMatcherClass)
        first = _get_mode_matcher()
        second = _get_mode_matcher()
        assert first is second
        assert len(_StubMatcherClass.instances) == 1
        assert _StubMatcherClass.instances[0].name == "mode-line-edit"


# ── 키워드 분석 ─────────────────────────────────────────────────────────


class TestKeywordAnalysis:
    def test_explicit_legacy(self):
        assert _analyze_intent_with_keywords("please use legacy format") == "legacy"

    def test_legacy_word_boundary_rejects_identifier(self):
        assert _analyze_intent_with_keywords("fix legacy_parser.py") == "normal"

    def test_line_number_routes_strict_json(self):
        assert _analyze_intent_with_keywords("add a comment on line 42") == "strict_json"
        assert _analyze_intent_with_keywords("줄 7 수정") == "strict_json"

    def test_semantic_backstop_with_digit(self, monkeypatch):
        monkeypatch.setattr(emc, "_get_mode_matcher", lambda: SimpleNamespace(matches=lambda t, label: True))
        assert _analyze_intent_with_keywords("update the 3rd widget") == "strict_json"

    def test_semantic_backstop_rejects_without_digit(self, monkeypatch):
        monkeypatch.setattr(emc, "_get_mode_matcher", lambda: SimpleNamespace(matches=lambda t, label: True))
        assert _analyze_intent_with_keywords("update the widget") == "normal"

    def test_semantic_backstop_no_match_falls_to_normal(self, monkeypatch):
        monkeypatch.setattr(emc, "_get_mode_matcher", lambda: SimpleNamespace(matches=lambda t, label: False))
        assert _analyze_intent_with_keywords("update the 3rd widget") == "normal"

    def test_default_normal(self):
        assert _analyze_intent_with_keywords("fix the bug") == "normal"


# ── analyze_request_for_optimal_mode ────────────────────────────────────


class TestAnalyzeRequest:
    def test_llm_decision_wins(self, monkeypatch):
        monkeypatch.setattr(emc, "_analyze_intent_with_llm_if_available", lambda p, t: ExecuteMode.INTELLIGENT)
        assert analyze_request_for_optimal_mode("build a feature", None) == "intelligent"

    def test_llm_none_falls_back_to_keywords(self, monkeypatch):
        monkeypatch.setattr(emc, "_analyze_intent_with_llm_if_available", lambda p, t: None)
        assert analyze_request_for_optimal_mode("add a comment on line 42", None) == "strict_json"


# ── LLM 경로 ────────────────────────────────────────────────────────────


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.timeout = 120
        self.chat_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


def _fake_service(client, model="m"):
    return SimpleNamespace(llm_service=SimpleNamespace(client=client), model=model)


class TestLLMPath:
    @pytest.fixture(autouse=True)
    def _reset_locks(self):
        emc._client_timeout_locks.clear()
        yield
        emc._client_timeout_locks.clear()

    def test_service_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr("external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: None)
        assert emc._analyze_intent_with_llm_if_available("x", None) is None

    def test_service_creation_raises_returns_none(self, monkeypatch):
        def boom(a, b):
            raise RuntimeError("no api key")

        monkeypatch.setattr("external_llm.intelligent_service.create_intelligent_service_from_env", boom)
        assert emc._analyze_intent_with_llm_if_available("x", None) is None

    def test_alias_exact_match(self, monkeypatch):
        client = _FakeClient(response=SimpleNamespace(content="strict_json"))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        mode = emc._analyze_intent_with_llm_if_available("edit line 9", "a.py")
        assert mode is ExecuteMode.STRICT_JSON
        assert client.timeout == 120  # 타임아웃 복원
        assert client.chat_calls[0]["temperature"] == 0.0

    def test_punctuation_stripped_before_alias_lookup(self, monkeypatch):
        client = _FakeClient(response=SimpleNamespace(content="plan_json."))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        assert emc._analyze_intent_with_llm_if_available("x", None) is ExecuteMode.PLAN_JSON

    def test_fuzzy_alias_match_inside_prose(self, monkeypatch):
        client = _FakeClient(response=SimpleNamespace(content="I recommend the intelligent mode for this."))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        assert emc._analyze_intent_with_llm_if_available("x", None) is ExecuteMode.INTELLIGENT

    def test_fuzzy_word_boundary_rejects_embedded_alias(self, monkeypatch):
        client = _FakeClient(response=SimpleNamespace(content="this is abnormal output"))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        assert emc._analyze_intent_with_llm_if_available("x", None) is None

    def test_chat_exception_falls_back_to_none(self, monkeypatch):
        client = _FakeClient(exc=RuntimeError("upstream down"))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        assert emc._analyze_intent_with_llm_if_available("x", None) is None

    def test_braces_escaped_in_prompt(self, monkeypatch):
        client = _FakeClient(response=SimpleNamespace(content="normal"))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        mode = emc._analyze_intent_with_llm_if_available('add {key: "v"} to dict', None)
        assert mode is ExecuteMode.NORMAL
        sent = client.chat_calls[0]["messages"]
        assert '{{key: "v"}}' in sent[0].content  # .format() 안전 — 중괄호 이스케이프 확인

    def test_shared_client_lock_restores_timeout_after_exception(self, monkeypatch):
        client = _FakeClient(exc=RuntimeError("boom"))
        monkeypatch.setattr(
            "external_llm.intelligent_service.create_intelligent_service_from_env", lambda a, b: _fake_service(client)
        )
        emc._analyze_intent_with_llm_if_available("x", None)
        assert client.timeout == 120


# ── validate_instruction_target_file ────────────────────────────────────


class TestValidateTargetFile:
    def test_empty_expected_skips(self):
        validate_instruction_target_file({"target_file": "a"}, "")
        validate_instruction_target_file({"target_file": "a"}, None)

    def test_no_target_file_emitted_skips(self):
        validate_instruction_target_file({}, "a.py")
        validate_instruction_target_file(None, "a.py")
        validate_instruction_target_file({"target_file": ""}, "a.py")

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="target_file mismatch"):
            validate_instruction_target_file({"target_file": "b.py"}, "a.py")

    def test_match_ok(self):
        validate_instruction_target_file({"target_file": "a.py"}, "a.py")
