"""Regression tests for the TypeScript syntax provider.

Covers the graceful-degrade contract: when ``tsc`` is not on ``$PATH``
the provider must fall back to tree-sitter syntax checking rather than
silently returning ``ok=True``.
"""
from unittest.mock import patch

from external_llm.languages.models import LanguageId
from external_llm.languages.typescript_provider import TypeScriptSyntaxProvider


class TestTscAbsentDegrade:
    """When tsc is not installed, tree-sitter provides a zero-toolchain syntax check."""

    @staticmethod
    def _tool_absent():
        return patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=FileNotFoundError("tsc not found"),
        )

    def test_valid_passes_tree_sitter_fallback(self):
        with self._tool_absent():
            r = TypeScriptSyntaxProvider().validate_syntax(
                "app.ts", "const x: number = 1;"
            )
        assert r.ok is True
        assert r.language is LanguageId.TYPESCRIPT

    def test_syntax_error_caught_by_tree_sitter(self):
        with self._tool_absent():
            r = TypeScriptSyntaxProvider().validate_syntax(
                "app.ts", "const x: number = ;"
            )
        assert r.ok is False
        assert r.language is LanguageId.TYPESCRIPT
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message


class TestTsxFallback:
    """When ``tsc`` is absent, tree-sitter must correctly handle ``.tsx`` files.

    The plain "typescript" grammar does NOT understand JSX; the ``tsx`` grammar
    must be selected based on the ``file_path`` suffix to avoid false rollback
    of valid JSX-using ``.tsx`` files.
    """

    @staticmethod
    def _tool_absent():
        return patch(
            "external_llm.languages.typescript_provider.subprocess.run",
            side_effect=FileNotFoundError("tsc not found"),
        )

    def test_valid_tsx_with_jsx_passes(self):
        """Valid JSX in a .tsx file must not be rolled back."""
        with self._tool_absent():
            r = TypeScriptSyntaxProvider().validate_syntax(
                "component.tsx", "const A = () => <div>hello</div>;"
            )
        assert r.ok is True
        assert r.language is LanguageId.TYPESCRIPT

    def test_syntax_error_tsx_caught(self):
        """Genuine syntax error in a .tsx file must still be caught."""
        with self._tool_absent():
            r = TypeScriptSyntaxProvider().validate_syntax(
                "broken.tsx", "const A = () => <div>hello</div"
            )
        assert r.ok is False
        assert r.language is LanguageId.TYPESCRIPT
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message


class TestTsRegistryWiring:
    def test_ts_provider_registered(self):
        from external_llm.languages.registry import LanguageRegistry
        r = LanguageRegistry.instance()
        prov = r.get("app.ts")
        assert prov.__class__.__name__ == "TypeScriptSyntaxProvider"
        assert prov.language_id() is LanguageId.TYPESCRIPT

    def test_capabilities_advertise_syntax(self):
        caps = TypeScriptSyntaxProvider().capabilities()
        assert caps.has_syntax_validator
