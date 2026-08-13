"""Regression tests for the JavaScript syntax provider.

Covers the graceful-degrade contract: when ``node`` is not on ``$PATH``
the provider must fall back to tree-sitter syntax checking rather than
silently returning ``ok=True``.
"""
from unittest.mock import patch

import pytest

from external_llm.languages.javascript_provider import JavaScriptSyntaxProvider
from external_llm.languages.models import LanguageId


class TestNodeAbsentDegrade:
    """When node is not installed, tree-sitter provides a zero-toolchain syntax check."""

    @staticmethod
    def _tool_absent():
        return patch(
            "external_llm.languages.javascript_provider.subprocess.run",
            side_effect=FileNotFoundError("node not found"),
        )

    def test_valid_passes_tree_sitter_fallback(self):
        with self._tool_absent():
            r = JavaScriptSyntaxProvider().validate_syntax(
                "app.js", "const x = 1;"
            )
        assert r.ok is True
        assert r.language is LanguageId.JAVASCRIPT

    def test_syntax_error_caught_by_tree_sitter(self):
        with self._tool_absent():
            r = JavaScriptSyntaxProvider().validate_syntax(
                "app.js", "const x = ;"
            )
        assert r.ok is False
        assert r.language is LanguageId.JAVASCRIPT
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message


class TestJsRegistryWiring:
    def test_js_provider_registered(self):
        from external_llm.languages.registry import LanguageRegistry
        r = LanguageRegistry.instance()
        prov = r.get("app.js")
        assert prov.__class__.__name__ == "JavaScriptSyntaxProvider"
        assert prov.language_id() is LanguageId.JAVASCRIPT

    def test_capabilities_advertise_syntax(self):
        caps = JavaScriptSyntaxProvider().capabilities()
        assert caps.has_syntax_validator


class TestJsGeneratorSymbolPatterns:
    """Bug (P3): ``function*`` generator declarations were invisible to the JS
    symbol regexes (``function\\s+Name`` skipped the generator asterisk) —
    both in get_symbol_patterns (used by _find_symbol_regex and the ripgrep
    outline) and in the top-level-definitions fallback. JS delegates the
    latter to the TS provider, so the shared fix is covered from both sides.
    """

    @pytest.fixture
    def provider(self):
        return JavaScriptSyntaxProvider()

    def test_generator_function_symbol_pattern(self, provider):
        import re as _re
        pats = [p for p in provider.get_symbol_patterns("function")
                if "const" not in p.regex and "let" not in p.regex and "var" not in p.regex]
        assert pats, "expected the plain function-declaration pattern"
        for p in pats:
            rx = _re.compile(p.regex.format(name="genSeq"))
            assert rx.search("function* genSeq() {") is not None, p.regex
            assert rx.search("async function* genSeq() {") is not None, p.regex
            assert rx.search("export default function* genSeq() {") is not None, p.regex
            assert rx.search("function genSeq() {") is not None, p.regex

    def test_generator_top_level_definitions_regex(self, provider):
        out = provider._find_top_level_definitions_regex(
            "function* genSeq() {\n  yield 1;\n}\n\n"
            "async function* stream() {\n  yield 2;\n}\n"
        )
        names = [r[0] for r in out]
        assert "genSeq" in names, names
        assert "stream" in names, names

    def test_find_generator_via_regex_fallback(self, provider):
        """find_symbol_in_file with tree-sitter disabled must still locate a
        generator declaration through the regex fallback."""
        from unittest.mock import patch

        with patch("external_llm.languages.tree_sitter_utils.is_available", return_value=False):
            result = provider.find_symbol_in_file(
                "app.js", "genSeq", "function* genSeq() {\n  yield 1;\n}\n"
            )
        assert result is not None
        assert result[0] == 1
