"""Unit tests for the OpenRouter provider integration.

Covers:
  - model_registry.detect_cloud_provider / _norm for OpenRouter slugs
    (both ``openrouter/<slug>`` and ``openrouter:<slug>`` forms).
  - create_llm_client factory routing to OpenRouterClient.
  - OpenRouterClient._strip_internal_prefix, _provider_preference,
    _build_headers, _log_cache.
  - Cost estimation: OpenRouter slugs get OpenRouter pricing (cheaper than
    native) while bare vendor slugs keep native pricing (no shadowing).
  - env-var service creation (create_service_from_env / intelligent variant).
"""
from __future__ import annotations

import pytest

from external_llm.client import create_llm_client
from external_llm.model_registry import _norm, detect_cloud_provider
from external_llm.openai_client import OpenAIClient, OpenRouterClient

# ── detect_cloud_provider / _norm ────────────────────────────────────────────

@pytest.mark.parametrize("model,expected", [
    # OpenRouter routing prefix → openrouter (both colon and slash forms)
    ("openrouter/deepseek/deepseek-v4-flash", "openrouter"),
    ("openrouter:deepseek/deepseek-v4-flash", "openrouter"),
    ("openrouter:anthropic/claude-sonnet-4-6", "openrouter"),
    # Bare vendor slug (no openrouter prefix) → native provider (BY DESIGN)
    ("deepseek/deepseek-v4-flash", "deepseek"),
    ("deepseek-v4-flash", "deepseek"),
    ("glm-5.2", "zai"),
    ("claude-sonnet-4-6", "anthropic"),
    # Unknown / ollama
    ("some-unknown-model", None),
])
def test_detect_cloud_provider(model, expected):
    assert detect_cloud_provider(model) == expected


@pytest.mark.parametrize("inp,expected", [
    ("openrouter:deepseek/deepseek-v4-flash", "openrouter/deepseek/deepseek-v4-flash"),
    ("openrouter/deepseek/deepseek-v4-flash", "openrouter/deepseek/deepseek-v4-flash"),
    ("ollama:gemma4:e2b", "gemma4:e2b"),
    ("deepseek-v4-flash", "deepseek-v4-flash"),
])
def test_norm_openrouter_forms(inp, expected):
    assert _norm(inp) == expected


# ── create_llm_client factory ────────────────────────────────────────────────

def test_factory_creates_openrouter_client():
    c = create_llm_client("openrouter", "sk-or-test")
    assert isinstance(c, OpenRouterClient)
    assert c.DEFAULT_BASE_URL == "https://openrouter.ai/api/v1"
    assert c.DEFAULT_MODEL == "deepseek/deepseek-v4-flash"
    assert c.get_provider_name() == "openrouter"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="openrouter"):
        create_llm_client("bogus", "k")


# ── OpenRouterClient internals ───────────────────────────────────────────────

@pytest.mark.parametrize("inp,expected", [
    ("openrouter/deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
    ("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
    ("anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
    ("", ""),
])
def test_strip_internal_prefix(inp, expected):
    assert OpenRouterClient._strip_internal_prefix(inp) == expected


def test_provider_preference_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepSeek")
    assert OpenRouterClient._provider_preference() == {"order": ["DeepSeek"]}

    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepSeek, Hyperbolic")
    assert OpenRouterClient._provider_preference() == {"order": ["DeepSeek", "Hyperbolic"]}

    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    assert OpenRouterClient._provider_preference() is None


@pytest.mark.parametrize("raw,expected", [
    ("deny", "deny"),
    ("DENY", "deny"),
    ("  deny  ", "deny"),
    ("allow", "allow"),
    ("Allow ", "allow"),
])
def test_provider_preference_data_collection(monkeypatch, raw, expected):
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", raw)
    assert OpenRouterClient._provider_preference() == {"data_collection": expected}


@pytest.mark.parametrize("raw", ["permissive", "", "0", "true", "no", "1"])
def test_provider_preference_data_collection_rejects_invalid(monkeypatch, raw):
    """Only the canonical 'allow'/'deny' (case/whitespace-insensitive) are accepted."""
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", raw)
    assert OpenRouterClient._provider_preference() is None


def test_provider_preference_combines_order_and_data_collection(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepSeek, Hyperbolic")
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "deny")
    pref = OpenRouterClient._provider_preference()
    assert pref == {"order": ["DeepSeek", "Hyperbolic"], "data_collection": "deny"}


def test_inject_provider_preference_does_not_override_caller(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepSeek")
    c = OpenRouterClient("sk-test")
    kw = {"provider": {"order": ["Custom"]}}
    c._inject_provider_preference(kw)
    assert kw["provider"] == {"order": ["Custom"]}

    kw2 = {}
    c._inject_provider_preference(kw2)
    assert kw2 == {"provider": {"order": ["DeepSeek"]}}


def test_build_headers_includes_attribution(monkeypatch):
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://asicode.dev")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "asicode")
    c = OpenRouterClient("sk-or-test")
    h = c._build_headers()
    assert h["Authorization"] == "Bearer sk-or-test"
    assert h["Content-Type"] == "application/json"
    assert h["HTTP-Referer"] == "https://asicode.dev"
    assert h["X-Title"] == "asicode"


def test_build_headers_without_attribution(monkeypatch):
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)
    c = OpenRouterClient("sk-or-test")
    h = c._build_headers()
    assert "HTTP-Referer" not in h
    assert "X-Title" not in h


def test_openai_client_build_headers_unchanged():
    """Refactor: OpenAIClient must still produce plain base headers."""
    c = OpenAIClient("sk-test")
    h = c._build_headers()
    assert h == {"Authorization": "Bearer sk-test", "Content-Type": "application/json"}


def test_log_cache_handles_various_inputs(caplog):
    OpenRouterClient._log_cache(None)
    OpenRouterClient._log_cache({})
    OpenRouterClient._log_cache({"usage": {}})
    OpenRouterClient._log_cache({
        "usage": {"prompt_tokens": 10000,
                  "prompt_tokens_details": {"cached_tokens": 8500}}
    })


# ── Cost estimation ──────────────────────────────────────────────────────────

def test_openrouter_slug_gets_openrouter_pricing():
    from external_llm.agent._shared_utils import _get_rates
    assert _get_rates("openrouter", "deepseek/deepseek-v4-flash") == (0.09, 0.18)


def test_native_deepseek_not_shadowed_by_openrouter():
    """Bare deepseek-v4-flash must keep native ($0.14/$0.28), not OpenRouter rate."""
    from external_llm.agent._shared_utils import _get_rates
    assert _get_rates("deepseek", "deepseek-v4-flash") == (0.14, 0.28)


def test_openrouter_cheaper_than_native():
    from external_llm.agent._shared_utils import estimate_cost
    or_cost = estimate_cost("openrouter", 1_000_000, 1_000_000, "deepseek/deepseek-v4-flash")
    native = estimate_cost("deepseek", 1_000_000, 1_000_000, "deepseek-v4-flash")
    assert or_cost < native
    assert abs(or_cost - 0.27) < 1e-9


# ── Service creation from env ────────────────────────────────────────────────

def test_create_service_from_env_openrouter(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("EXTERNAL_LLM_MODEL", "deepseek/deepseek-v4-flash")
    from external_llm.service import create_service_from_env
    svc = create_service_from_env()
    assert svc is not None
    assert svc.provider == "openrouter"
    assert svc.model == "deepseek/deepseek-v4-flash"
    assert isinstance(svc.client, OpenRouterClient)


def test_create_service_missing_key_returns_none(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from external_llm.service import create_service_from_env
    assert create_service_from_env() is None


def test_create_intelligent_service_explicit_key(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("EXTERNAL_LLM_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from external_llm.intelligent_service import create_intelligent_service_from_env
    svc = create_intelligent_service_from_env(api_key="sk-or-explicit")
    assert svc is not None
    assert svc.provider == "openrouter"
