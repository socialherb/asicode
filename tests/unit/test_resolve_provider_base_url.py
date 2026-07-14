"""Regression tests for provider-scoped base_url resolution.

Guards against the leak where a global ``EXTERNAL_LLM_BASE_URL`` configured for
one provider (e.g. opencode's host) was injected into EVERY provider's client,
pointing it at the wrong host (zai key → opencode endpoint → HTTP 401 →
misleading "Invalid API key" prompt) and disabling zai's endpoint-failover.
"""
import os

import pytest

from external_llm.client import resolve_provider_base_url


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every base_url-related env var so each test starts from scratch."""
    for k in (
        "EXTERNAL_LLM_BASE_URL", "EXTERNAL_LLM_PROVIDER",
        "ZAI_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL",
        "OPENCODE_BASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


class TestResolveProviderBaseUrl:
    def test_no_env_returns_none(self, clean_env):
        # No base_url env at all → client uses its DEFAULT_BASE_URL.
        assert resolve_provider_base_url("zai") is None

    def test_global_base_url_applies_only_to_matching_provider(self, clean_env):
        clean_env.setenv("EXTERNAL_LLM_PROVIDER", "openai")
        clean_env.setenv("EXTERNAL_LLM_BASE_URL", "https://opencode.ai/zen/go/v1")

        # The global base_url belongs to the openai/opencode provider.
        assert resolve_provider_base_url("openai") == "https://opencode.ai/zen/go/v1"
        # A DIFFERENT provider must NOT inherit the foreign host's base_url.
        assert resolve_provider_base_url("zai") is None
        assert resolve_provider_base_url("anthropic") is None

    def test_per_provider_override_wins(self, clean_env):
        clean_env.setenv("EXTERNAL_LLM_PROVIDER", "openai")
        clean_env.setenv("EXTERNAL_LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
        clean_env.setenv("ZAI_BASE_URL", "https://zai-proxy.internal/")

        assert resolve_provider_base_url("zai") == "https://zai-proxy.internal/"
        # other provider still uses the global (matching) base_url
        assert resolve_provider_base_url("openai") == "https://opencode.ai/zen/go/v1"

    def test_per_provider_override_without_global(self, clean_env):
        clean_env.setenv("ANTHROPIC_BASE_URL", "https://my-claude-proxy/")
        assert resolve_provider_base_url("anthropic") == "https://my-claude-proxy/"
        # zai unaffected
        assert resolve_provider_base_url("zai") is None

    def test_empty_provider_returns_none(self, clean_env):
        clean_env.setenv("EXTERNAL_LLM_BASE_URL", "https://x/")
        assert resolve_provider_base_url("") is None

    def test_case_insensitive_provider(self, clean_env):
        clean_env.setenv("EXTERNAL_LLM_PROVIDER", "zai")
        clean_env.setenv("EXTERNAL_LLM_BASE_URL", "https://z.global/")
        # mixed-case provider name should still match the global provider.
        assert resolve_provider_base_url("ZAI") == "https://z.global/"
        # and the per-provider override env var is upper-cased.
        clean_env.setenv("DEEPSEEK_BASE_URL", "https://ds.proxy/")
        assert resolve_provider_base_url("DeepSeek") == "https://ds.proxy/"


class TestServiceFactoryDoesNotLeak:
    """End-to-end: the intelligent-service factory must build a zai client whose
    base_url is None (so it uses z.ai's DEFAULT and so failover is allowed),
    even when a foreign EXTERNAL_LLM_BASE_URL is globally set."""

    def test_zai_client_base_url_is_none_under_foreign_global(self, monkeypatch):
        monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
        monkeypatch.setenv("EXTERNAL_LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
        monkeypatch.setenv("ZAI_API_KEY", "dummy-key")

        from external_llm.intelligent_service import create_intelligent_service_from_env
        svc = create_intelligent_service_from_env(provider="zai", model="glm-5.2")
        client = svc.llm_service.client if hasattr(svc, "llm_service") else svc.client
        assert type(client).__name__ == "ZAIAnthropicClient"
        # base_url MUST be None — otherwise the client points at the wrong host
        # AND zai's auth/connection endpoint-failover is disabled.
        assert client.base_url is None
        # The failover guard mirrors DesignChatLoop._flip_zai_endpoint:
        assert not client.base_url, "base_url must be falsy so endpoint failover is allowed"

    def test_openai_client_keeps_global_base_url(self, monkeypatch):
        monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openai")
        monkeypatch.setenv("EXTERNAL_LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

        from external_llm.intelligent_service import create_intelligent_service_from_env
        svc = create_intelligent_service_from_env(provider="openai", model="deepseek-chat")
        client = svc.llm_service.client if hasattr(svc, "llm_service") else svc.client
        # The matching provider keeps the configured global base_url.
        assert client.base_url == "https://opencode.ai/zen/go/v1"
