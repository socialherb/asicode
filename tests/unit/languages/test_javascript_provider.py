"""Regression tests for the JavaScript syntax provider.

Covers the graceful-degrade contract: when ``node`` is not on ``$PATH``
the provider must fall back to tree-sitter syntax checking rather than
silently returning ``ok=True``.
"""
from unittest.mock import patch

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
