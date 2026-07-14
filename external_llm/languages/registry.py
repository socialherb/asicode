"""
Language registry — singleton that maps file paths to providers.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .base import SyntaxProvider
from .models import LanguageId

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Singleton registry of language providers, keyed by ``LanguageId``."""

    _instance: Optional["LanguageRegistry"] = None

    def __init__(self) -> None:
        self._providers: dict[LanguageId, SyntaxProvider] = {}
        # Auto-register built-in providers
        from .bash_provider import BashSyntaxProvider
        from .csharp_provider import CSharpSyntaxProvider
        from .css_provider import CssSyntaxProvider
        from .go_provider import GoSyntaxProvider
        from .html_provider import HtmlSyntaxProvider
        from .java_provider import JavaSyntaxProvider
        from .javascript_provider import JavaScriptSyntaxProvider
        from .json_provider import JsonSyntaxProvider
        from .kotlin_provider import KotlinSyntaxProvider
        from .php_provider import PhpSyntaxProvider
        from .python_provider import PythonSyntaxProvider
        from .ruby_provider import RubySyntaxProvider

        # Regex-only providers (symbol search via the provider index; no
        # bundled toolchain assumed). These retire the legacy hardcoded rg
        # fallback (_find_in_other_langs): every non-Python language now has
        # a provider, so the index is the single source of truth.
        from .rust_provider import RustSyntaxProvider
        from .swift_provider import SwiftSyntaxProvider
        from .typescript_provider import TypeScriptSyntaxProvider
        self.register(PythonSyntaxProvider())
        self.register(TypeScriptSyntaxProvider())
        self.register(JavaScriptSyntaxProvider())
        self.register(GoSyntaxProvider())
        self.register(JavaSyntaxProvider())
        self.register(KotlinSyntaxProvider())
        self.register(JsonSyntaxProvider())
        self.register(CssSyntaxProvider())
        self.register(HtmlSyntaxProvider())
        self.register(RustSyntaxProvider())
        self.register(CSharpSyntaxProvider())
        self.register(RubySyntaxProvider())
        self.register(PhpSyntaxProvider())
        self.register(SwiftSyntaxProvider())
        self.register(BashSyntaxProvider())

    @classmethod
    def instance(cls) -> "LanguageRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for tests)."""
        cls._instance = None

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, provider: SyntaxProvider) -> None:
        lang = provider.language_id()
        self._providers[lang] = provider
        logger.debug("Registered language provider: %s", lang.value)

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, file_path: str) -> Optional[SyntaxProvider]:
        """Return the provider for *file_path*, or ``None`` if unsupported."""
        lang = LanguageId.from_path(file_path)
        return self._providers.get(lang)

    def get_by_lang(self, lang: LanguageId) -> Optional[SyntaxProvider]:
        """Return the provider for *lang*, or ``None`` if unsupported."""
        return self._providers.get(lang)

    def supports_structured_ops(self, file_path: str) -> bool:
        """Whether *file_path* can be edited with structured operations
        (modify_symbol, insert_after_symbol, etc.)."""
        provider = self.get(file_path)
        if provider is None:
            return False
        caps = provider.capabilities()
        return caps.supports_modify_symbol or caps.supports_insert_after_symbol

    def supports_syntax_validation(self, file_path: str) -> bool:
        provider = self.get(file_path)
        if provider is None:
            return False
        return provider.capabilities().has_syntax_validator

    # ── Aggregate helpers ─────────────────────────────────────────────────

    def get_file_pattern(self) -> str:
        """Return a single regex that matches file paths of any registered language.

        Example output: ``r"[\\w/.-]+\\.(?:py|ts|tsx)(?![a-zA-Z0-9_])"``.
        """
        all_exts: list[str] = []
        for provider in self._providers.values():
            for glob in provider.get_file_globs():
                # "*.py" → "py"
                ext = glob.lstrip("*.")
                if ext:
                    all_exts.append(re.escape(ext))
        if not all_exts:
            return r"[\w/.-]+\.py(?![a-zA-Z0-9_])"
        joined = "|".join(sorted(set(all_exts)))
        return rf"[\w/.-]+\.(?:{joined})(?![a-zA-Z0-9_])"

    def get_all_file_globs(self) -> list[str]:
        """Return combined file globs from all registered providers."""
        globs: list[str] = []
        for provider in self._providers.values():
            globs.extend(provider.get_file_globs())
        return globs
