"""Unit tests for CssSyntaxProvider."""
from __future__ import annotations

import pytest

from external_llm.languages.css_provider import CssSyntaxProvider
from external_llm.languages.models import LanguageId


@pytest.fixture
def provider():
    return CssSyntaxProvider()


# ── language_id / capabilities ────────────────────────────────────────────────

def test_language_id(provider):
    assert provider.language_id() == LanguageId.CSS


def test_capabilities_has_syntax_validator(provider):
    assert provider.capabilities().has_syntax_validator is True


# ── Valid CSS ────────────────────────────────────────────────────────────────

def test_valid_simple_rule(provider):
    result = provider.validate_syntax("style.css", "body { color: red; }")
    assert result.ok is True
    assert result.errors == []


def test_valid_nested_scss(provider):
    css = ".parent {\n  color: blue;\n  .child { font-size: 12px; }\n}"
    result = provider.validate_syntax("style.scss", css)
    assert result.ok is True


def test_valid_empty(provider):
    result = provider.validate_syntax("empty.css", "")
    assert result.ok is True


def test_valid_block_comment(provider):
    css = "/* header */ body { margin: 0; }"
    result = provider.validate_syntax("a.css", css)
    assert result.ok is True


def test_valid_line_comment_scss(provider):
    scss = "// comment\nbody { color: red; }"
    result = provider.validate_syntax("a.scss", scss)
    assert result.ok is True


def test_valid_string_with_brace(provider):
    # even if the content property contains a brace, it's treated as a string literal
    css = 'div::before { content: "{"; }'
    result = provider.validate_syntax("a.css", css)
    assert result.ok is True


# ── Invalid CSS: brace errors ─────────────────────────────────────────────────

def test_unclosed_brace(provider):
    result = provider.validate_syntax("bad.css", "body { color: red;")
    assert result.ok is False
    assert any("Unclosed" in e.message for e in result.errors)


def test_extra_closing_brace(provider):
    result = provider.validate_syntax("bad.css", "body { color: red; }}")
    assert result.ok is False
    assert any("}" in e.message for e in result.errors)


def test_mismatched_braces(provider):
    result = provider.validate_syntax("bad.css", ".a { .b { color: red; }")
    assert result.ok is False


# ── Invalid CSS: comment/string errors ────────────────────────────────────────

def test_unclosed_block_comment(provider):
    result = provider.validate_syntax("bad.css", "/* open comment\nbody { color: red; }")
    assert result.ok is False
    assert any("comment" in e.message.lower() for e in result.errors)


def test_unclosed_string_double_quote(provider):
    result = provider.validate_syntax("bad.css", 'body { content: "open string; }')
    assert result.ok is False


def test_unclosed_string_single_quote(provider):
    result = provider.validate_syntax("bad.css", "body { content: 'open; }")
    assert result.ok is False


# ── Error location ─────────────────────────────────────────────────────────────

def test_error_has_file_path(provider):
    result = provider.validate_syntax("my/path.css", "}")
    assert result.ok is False
    assert result.errors[0].file == "my/path.css"


def test_error_has_line_number(provider):
    result = provider.validate_syntax("a.css", "\n\n}")
    assert result.ok is False
    assert result.errors[0].line >= 3


def test_language_in_result(provider):
    result = provider.validate_syntax("x.css", "body {}")
    assert result.language == LanguageId.CSS


# ── File globs ───────────────────────────────────────────────────────────────

def test_file_globs(provider):
    globs = provider.get_file_globs()
    assert "*.css" in globs
    assert "*.scss" in globs
    assert "*.less" in globs


# ── Symbol patterns (AST-sourced, provider contributes no regex) ───────────
#
# CSS symbols (class/id/custom-property) are extracted from the tree-sitter
# AST via tree_sitter_utils.find_all_symbols, not from provider regex
# patterns. The provider's get_symbol_patterns therefore returns an empty
# list — this is intentional and keeps CSS off the rg spawn path entirely.

def test_get_symbol_patterns_empty(provider):
    """CSS symbols come from the tree-sitter AST (find_all_symbols), so the
    provider contributes no regex patterns. _nonpy_index_for routes CSS
    through find_all_symbols when tree_sitter_css is installed."""
    assert provider.get_symbol_patterns() == []
    assert provider.get_symbol_patterns(kind="class") == []
    assert provider.get_symbol_patterns(kind="any") == []


def test_get_file_globs_unchanged(provider):
    """Even though symbols are AST-sourced, file globs still drive which files
    _index_via_treesitter scans for CSS."""
    globs = provider.get_file_globs()
    assert "*.css" in globs


def test_get_lint_command_none(provider):
    assert provider.get_lint_command("x.css") is None
