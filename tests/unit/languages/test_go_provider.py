"""Regression tests for the Go syntax provider.

Covers the graceful-degrade contract: when ``go`` is not on ``$PATH``
the provider must fall back to tree-sitter syntax checking rather than
silently returning ``ok=True``.
"""
from unittest.mock import patch

from external_llm.languages.go_provider import GoSyntaxProvider
from external_llm.languages.models import LanguageId


class TestGoToolAbsentDegrade:
    """When go is not installed, tree-sitter provides a zero-toolchain syntax check."""

    @staticmethod
    def _tool_absent():
        return patch(
            "external_llm.languages.go_provider.subprocess.run",
            side_effect=FileNotFoundError("go not found"),
        )

    def test_valid_passes_tree_sitter_fallback(self):
        with self._tool_absent():
            r = GoSyntaxProvider().validate_syntax(
                "main.go", "package main\nfunc main(){}\n"
            )
        assert r.ok is True
        assert r.language is LanguageId.GO

    def test_syntax_error_caught_by_tree_sitter(self):
        with self._tool_absent():
            r = GoSyntaxProvider().validate_syntax(
                "main.go", "package main\nfunc main(}\n"
            )
        assert r.ok is False
        assert r.language is LanguageId.GO
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message


class TestGoRegistryWiring:
    def test_go_provider_registered(self):
        from external_llm.languages.registry import LanguageRegistry
        r = LanguageRegistry.instance()
        prov = r.get("main.go")
        assert prov.__class__.__name__ == "GoSyntaxProvider"
        assert prov.language_id() is LanguageId.GO

    def test_capabilities_advertise_syntax(self):
        caps = GoSyntaxProvider().capabilities()
        assert caps.has_syntax_validator
