"""Regression tests for the Java syntax provider.

Covers the graceful-degrade contract: when ``javac`` is not on ``$PATH``
the provider must fall back to tree-sitter syntax checking rather than
silently returning ``ok=True``.
"""
from unittest.mock import patch

from external_llm.languages.java_provider import JavaSyntaxProvider
from external_llm.languages.models import LanguageId


class TestJavacAbsentDegrade:
    """When javac is not installed, tree-sitter provides a zero-toolchain syntax check."""

    @staticmethod
    def _tool_absent():
        return patch(
            "external_llm.languages.java_provider.subprocess.run",
            side_effect=FileNotFoundError("javac not found"),
        )

    def test_valid_passes_tree_sitter_fallback(self):
        with self._tool_absent():
            r = JavaSyntaxProvider().validate_syntax(
                "Foo.java", "class Foo {}"
            )
        assert r.ok is True
        assert r.language is LanguageId.JAVA

    def test_syntax_error_caught_by_tree_sitter(self):
        with self._tool_absent():
            r = JavaSyntaxProvider().validate_syntax(
                "Foo.java", "class Foo {"
            )
        assert r.ok is False
        assert r.language is LanguageId.JAVA
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message


class TestJavaRegistryWiring:
    def test_java_provider_registered(self):
        from external_llm.languages.registry import LanguageRegistry
        r = LanguageRegistry.instance()
        prov = r.get("Foo.java")
        assert prov.__class__.__name__ == "JavaSyntaxProvider"
        assert prov.language_id() is LanguageId.JAVA

    def test_capabilities_advertise_syntax(self):
        caps = JavaSyntaxProvider().capabilities()
        assert caps.has_syntax_validator
