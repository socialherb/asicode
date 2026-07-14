"""HtmlSyntaxProvider 단위 테스트."""
from __future__ import annotations

import pytest

from external_llm.languages.html_provider import HtmlSyntaxProvider
from external_llm.languages.models import LanguageId


@pytest.fixture
def provider():
    return HtmlSyntaxProvider()


# ── language_id / capabilities ────────────────────────────────────────────────

def test_language_id(provider):
    assert provider.language_id() == LanguageId.HTML


def test_capabilities_has_syntax_validator(provider):
    assert provider.capabilities().has_syntax_validator is True


# ── 유효한 HTML ───────────────────────────────────────────────────────────────

def test_valid_minimal(provider):
    result = provider.validate_syntax("index.html", "<html><body></body></html>")
    assert result.ok is True
    assert result.errors == []


def test_valid_full_document(provider):
    html = "<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>"
    result = provider.validate_syntax("page.html", html)
    assert result.ok is True


def test_valid_void_elements_no_close(provider):
    # br, img, input 등 void element은 닫는 태그 없어도 유효
    html = "<div><br><img src='x.png'><input type='text'></div>"
    result = provider.validate_syntax("form.html", html)
    assert result.ok is True


def test_valid_empty_string(provider):
    result = provider.validate_syntax("empty.html", "")
    assert result.ok is True


def test_valid_comment(provider):
    html = "<!-- 주석 --><div></div>"
    result = provider.validate_syntax("a.html", html)
    assert result.ok is True


def test_valid_all_void_elements(provider):
    html = "<area><base><col><embed><hr><link><meta><param><source><track><wbr>"
    result = provider.validate_syntax("voids.html", html)
    assert result.ok is True


# ── 무효 HTML: 태그 불균형 ────────────────────────────────────────────────────

def test_unclosed_div(provider):
    result = provider.validate_syntax("bad.html", "<div><p>text</p>")
    assert result.ok is False
    assert any("Unclosed" in e.message for e in result.errors)


def test_extra_closing_tag(provider):
    result = provider.validate_syntax("bad.html", "<div></div></div>")
    assert result.ok is False
    assert any("Unexpected" in e.message for e in result.errors)


def test_multiple_unclosed(provider):
    result = provider.validate_syntax("bad.html", "<html><body><div>")
    assert result.ok is False
    # html, body, div — 3개 모두 미닫힘
    assert len(result.errors) >= 1


def test_wrong_nesting(provider):
    # <div> 내부에 </span> — 매칭 안 되는 닫힘 태그
    result = provider.validate_syntax("bad.html", "<div></span></div>")
    assert result.ok is False


# ── 오류 위치 및 메타데이터 ───────────────────────────────────────────────────

def test_error_has_file_path(provider):
    result = provider.validate_syntax("my/page.html", "<div>")
    assert result.ok is False
    assert result.errors[0].file == "my/page.html"


def test_unclosed_tag_reports_open_line(provider):
    html = "\n\n<div>"
    result = provider.validate_syntax("a.html", html)
    assert result.ok is False
    # 태그가 열린 줄(3번째 줄)을 보고해야 함
    assert result.errors[0].line >= 3


def test_language_in_result(provider):
    result = provider.validate_syntax("x.html", "<p></p>")
    assert result.language == LanguageId.HTML


# ── .htm 확장자 ───────────────────────────────────────────────────────────────

def test_htm_extension_ok(provider):
    result = provider.validate_syntax("old.htm", "<html></html>")
    assert result.ok is True


# ── 파일 글로브 ───────────────────────────────────────────────────────────────

def test_file_globs(provider):
    globs = provider.get_file_globs()
    assert "*.html" in globs
    assert "*.htm" in globs


# ── 미구현 메서드 안전 반환 ───────────────────────────────────────────────────

def test_get_symbol_patterns_empty(provider):
    assert provider.get_symbol_patterns() == []


def test_get_lint_command_none(provider):
    assert provider.get_lint_command("x.html") is None


def test_find_symbol_in_file_none(provider):
    assert provider.find_symbol_in_file("x.html", "foo", "<div></div>") is None
