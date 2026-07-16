"""Regression tests for the Kotlin syntax provider.

Covers the graceful-degrade contract: when ``kotlinc`` is not on ``$PATH``
the provider must fall back to tree-sitter syntax checking rather than
silently returning ``ok=True``.
"""
from unittest.mock import patch

from external_llm.languages.kotlin_provider import KotlinSyntaxProvider
from external_llm.languages.models import LanguageId


class TestKotlincAbsentDegrade:
    """When kotlinc is not installed, tree-sitter provides a zero-toolchain syntax check."""

    @staticmethod
    def _tool_absent():
        return patch(
            "external_llm.languages.kotlin_provider.subprocess.run",
            side_effect=FileNotFoundError("kotlinc not found"),
        )

    def test_valid_passes_tree_sitter_fallback(self):
        with self._tool_absent():
            r = KotlinSyntaxProvider().validate_syntax(
                "Main.kt", "fun main() {}"
            )
        assert r.ok is True
        assert r.language is LanguageId.KOTLIN

    def test_syntax_error_caught_by_tree_sitter(self):
        with self._tool_absent():
            r = KotlinSyntaxProvider().validate_syntax(
                "Main.kt", "fun main(}"
            )
        assert r.ok is False
        assert r.language is LanguageId.KOTLIN
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message


class TestKtRegistryWiring:
    def test_kt_provider_registered(self):
        from external_llm.languages.registry import LanguageRegistry
        r = LanguageRegistry.instance()
        prov = r.get("Main.kt")
        assert prov.__class__.__name__ == "KotlinSyntaxProvider"
        assert prov.language_id() is LanguageId.KOTLIN

    def test_capabilities_advertise_syntax(self):
        caps = KotlinSyntaxProvider().capabilities()
        assert caps.has_syntax_validator
