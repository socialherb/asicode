"""JsonSyntaxProvider 단위 테스트."""
from __future__ import annotations

import pytest

from external_llm.languages.json_provider import JsonSyntaxProvider
from external_llm.languages.models import LanguageId


@pytest.fixture
def provider():
    return JsonSyntaxProvider()


# ── language_id / capabilities ────────────────────────────────────────────────

def test_language_id(provider):
    assert provider.language_id() == LanguageId.JSON


def test_capabilities_has_syntax_validator(provider):
    assert provider.capabilities().has_syntax_validator is True


def test_capabilities_no_symbol_ops(provider):
    caps = provider.capabilities()
    assert not caps.supports_modify_symbol
    assert not caps.supports_insert_after_symbol


# ── 유효한 JSON ───────────────────────────────────────────────────────────────

def test_valid_simple_object(provider):
    result = provider.validate_syntax("test.json", '{"a": 1, "b": true}')
    assert result.ok is True
    assert result.errors == []


def test_valid_nested(provider):
    content = '{"compilerOptions": {"strict": true}, "include": ["src"]}'
    result = provider.validate_syntax("tsconfig.json", content)
    assert result.ok is True


def test_valid_empty_object(provider):
    result = provider.validate_syntax("empty.json", "{}")
    assert result.ok is True


def test_valid_array(provider):
    result = provider.validate_syntax("arr.json", "[1, 2, 3]")
    assert result.ok is True


# ── 유효하지 않은 JSON ────────────────────────────────────────────────────────

def test_invalid_trailing_comma(provider):
    result = provider.validate_syntax("bad.json", '{"a": 1,}')
    assert result.ok is False
    assert len(result.errors) == 1
    assert result.errors[0].file == "bad.json"


def test_invalid_unclosed_brace(provider):
    result = provider.validate_syntax("bad.json", '{"a": 1')
    assert result.ok is False


def test_invalid_reports_line_col(provider):
    result = provider.validate_syntax("bad.json", '{\n  "a": ,\n}')
    assert result.ok is False
    err = result.errors[0]
    assert err.line >= 1
    assert err.col >= 0


def test_language_in_result(provider):
    result = provider.validate_syntax("x.json", "{}")
    assert result.language == LanguageId.JSON


# ── JSONC (주석 포함 JSON) ────────────────────────────────────────────────────

def test_jsonc_line_comment_stripped(provider):
    content = '{\n  // 설명\n  "strict": true\n}'
    result = provider.validate_syntax("tsconfig.jsonc", content)
    assert result.ok is True


def test_jsonc_inline_comment_stripped(provider):
    content = '{"a": 1 // inline\n}'
    result = provider.validate_syntax("settings.jsonc", content)
    assert result.ok is True


def test_jsonc_comment_inside_string_not_stripped(provider):
    # // inside string literal should NOT be treated as comment
    content = '{"url": "http://example.com"}'
    result = provider.validate_syntax("x.jsonc", content)
    assert result.ok is True


def test_plain_json_comment_is_invalid(provider):
    # .json 파일에서는 주석을 strip하지 않으므로 파싱 오류
    content = '{"a": 1 // comment\n}'
    result = provider.validate_syntax("plain.json", content)
    assert result.ok is False


# ── 파일 글로브 ───────────────────────────────────────────────────────────────

def test_file_globs(provider):
    globs = provider.get_file_globs()
    assert "*.json" in globs
    assert "*.jsonc" in globs


# ── 미구현 메서드 안전 반환 ───────────────────────────────────────────────────

def test_get_symbol_patterns_returns_empty(provider):
    assert provider.get_symbol_patterns() == []


def test_get_lint_command_returns_none(provider):
    assert provider.get_lint_command("x.json") is None


def test_find_symbol_in_file_returns_none(provider):
    assert provider.find_symbol_in_file("x.json", "foo", "{}") is None
