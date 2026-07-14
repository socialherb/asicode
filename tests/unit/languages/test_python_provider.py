"""Tests for PythonSyntaxProvider (external_llm/languages/python_provider.py)."""
import pytest

from external_llm.languages.models import LanguageId
from external_llm.languages.python_provider import PythonSyntaxProvider


@pytest.fixture
def provider():
    return PythonSyntaxProvider()


# ── language_id ───────────────────────────────────────────────────────────────

def test_language_id(provider):
    assert provider.language_id() == LanguageId.PYTHON


# ── capabilities ──────────────────────────────────────────────────────────────

def test_capabilities_python(provider):
    caps = provider.capabilities()
    assert caps.has_ast_parser is True
    assert caps.has_syntax_validator is True
    assert caps.has_linter is True
    assert caps.has_test_runner is True
    assert caps.has_symbol_search is True
    assert caps.supports_modify_symbol is True
    assert caps.supports_insert_after_symbol is True


# ── validate_syntax ───────────────────────────────────────────────────────────

class TestValidateSyntax:
    def test_valid_python(self, provider):
        result = provider.validate_syntax("foo.py", "x = 1\ndef f():\n    return x\n")
        assert result.ok is True
        assert result.errors == []

    def test_invalid_syntax(self, provider):
        result = provider.validate_syntax("foo.py", "def (broken:\n")
        assert result.ok is False
        assert len(result.errors) > 0
        assert result.errors[0].file == "foo.py"

    def test_empty_content_is_valid(self, provider):
        result = provider.validate_syntax("foo.py", "")
        assert result.ok is True

    def test_error_includes_line_number(self, provider):
        content = "x = 1\ndef (broken):\n    pass\n"
        result = provider.validate_syntax("foo.py", content)
        assert result.ok is False
        assert result.errors[0].line > 0

    def test_language_is_python(self, provider):
        result = provider.validate_syntax("foo.py", "x = 1\n")
        assert result.language == LanguageId.PYTHON


# ── get_symbol_patterns ───────────────────────────────────────────────────────

class TestGetSymbolPatterns:
    def test_any_returns_function_and_class(self, provider):
        patterns = provider.get_symbol_patterns("any")
        kinds = {p.kind for p in patterns}
        assert "function" in kinds
        assert "class" in kinds

    def test_function_only(self, provider):
        patterns = provider.get_symbol_patterns("function")
        assert all(p.kind == "function" for p in patterns)

    def test_class_only(self, provider):
        patterns = provider.get_symbol_patterns("class")
        assert all(p.kind == "class" for p in patterns)

    def test_patterns_have_regex(self, provider):
        for pattern in provider.get_symbol_patterns("any"):
            assert pattern.regex
            assert "{name}" in pattern.regex


# ── get_file_globs ────────────────────────────────────────────────────────────

def test_file_globs(provider):
    globs = provider.get_file_globs()
    assert "*.py" in globs


# ── get_lint_command ──────────────────────────────────────────────────────────

def test_lint_command(provider):
    cmd = provider.get_lint_command("foo.py")
    assert cmd is not None
    assert "ruff" in cmd
    assert "foo.py" in cmd


# ── get_test_command ──────────────────────────────────────────────────────────

def test_test_command(provider):
    cmd = provider.get_test_command("/repo")
    assert cmd is not None
    assert "pytest" in " ".join(cmd)


def test_test_command_with_args(provider):
    cmd = provider.get_test_command("/repo", ["tests/unit/"])
    assert "tests/unit/" in cmd


# ── find_symbol_in_file ───────────────────────────────────────────────────────

class TestFindSymbolInFile:
    def test_finds_function(self, provider):
        content = "def my_func():\n    return 1\n"
        result = provider.find_symbol_in_file("foo.py", "my_func", content)
        assert result is not None
        start, _end = result
        assert start == 1

    def test_finds_class(self, provider):
        content = "class MyClass:\n    pass\n"
        result = provider.find_symbol_in_file("foo.py", "MyClass", content)
        assert result is not None
        assert result[0] == 1

    def test_not_found_returns_none(self, provider):
        content = "def other_func():\n    pass\n"
        result = provider.find_symbol_in_file("foo.py", "missing_func", content)
        assert result is None

    def test_syntax_error_returns_none(self, provider):
        content = "def (broken:\n"
        result = provider.find_symbol_in_file("foo.py", "any", content)
        assert result is None

    def test_nested_function_found(self, provider):
        content = "def outer():\n    def inner():\n        pass\n"
        result = provider.find_symbol_in_file("foo.py", "inner", content)
        assert result is not None
        assert result[0] == 2

    def test_async_function_found(self, provider):
        content = "async def async_func():\n    pass\n"
        result = provider.find_symbol_in_file("foo.py", "async_func", content)
        assert result is not None


# ── get_definition_keywords ───────────────────────────────────────────────────

def test_definition_keywords(provider):
    keywords = provider.get_definition_keywords()
    assert "def " in keywords
    assert "class " in keywords
