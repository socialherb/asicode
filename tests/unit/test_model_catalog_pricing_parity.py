"""Cross-validation: the model catalog (SSOT for which IDs exist) must never
contain a model that silently prices at (0,0) or at a wrong provider fallback.

The cost tables in external_llm.agent._shared_utils are hand-maintained and
drift independently of external_llm.model_catalog. A newly-added catalog model
without a rate entry is silently priced at the provider fallback — or at
(0,0) when the provider itself has no fallback entry. These tests make that
drift a test failure instead of a silent cost-stats bug.

Providers with a distinct rate table from the native one (opencode) are
checked against their own table (``_OPENCODE_COST_PER_M``); their unpriced
ids must be in the explicit allowlist (``_OPENCODE_UNPRICED_ALLOWLIST``).
"""

from __future__ import annotations

import pytest

from external_llm.agent._shared_utils import (
    _COST_PER_M,
    _MODEL_CACHE_RATE,
    _MODEL_COST_PER_M,
    _OPENCODE_CACHE_RATE,
    _OPENCODE_COST_PER_M,
    _OPENCODE_UNPRICED_ALLOWLIST,
    _catalog_unpriced_models,
    _get_rates,
    _longest_prefix_match,
)
from external_llm.model_catalog import KNOWN_MODELS, LEGACY_MODELS

OPENCODE_TABLE = _OPENCODE_COST_PER_M
OPENCODE_UNPRICED = _OPENCODE_UNPRICED_ALLOWLIST


def _every_catalog_model() -> list[tuple[str, str]]:
    return [(p, m) for p, models in {**KNOWN_MODELS, **LEGACY_MODELS}.items() for m in models]


@pytest.mark.parametrize(
    "provider,model",
    list(_every_catalog_model()),
)
def test_every_catalog_model_has_rate_or_nonzero_provider_fallback(provider, model):
    """No catalog model may be silently priced at (0,0) — a model-specific
    entry, or a non-zero provider-level fallback, must cover it (or it must
    be an explicitly-allowlisted zero-rate model: unpriced ids on the opencode
    table)."""
    in_rate, out_rate = _get_rates(provider, model)
    free_by_design = provider == "opencode" and model in OPENCODE_UNPRICED
    assert in_rate or out_rate or free_by_design, (
        f"{provider}/{model} prices at (0,0) — add a {provider} rate entry or unpriced allowlist"
    )


def test_no_catalog_model_with_nonzero_fallback_is_reported():
    """Sanity: the unpriced-enumerator agrees with the direct query — every
    reported model has no model-specific rate, and every model with a rate is
    not reported."""
    for provider, model in _every_catalog_model():
        reported = model in _catalog_unpriced_models().get(provider, [])
        if provider == "opencode":
            has_rate = _longest_prefix_match(model.lower(), _OPENCODE_COST_PER_M) is not None
            has_decision = has_rate or model in OPENCODE_UNPRICED
        else:
            has_rate = _longest_prefix_match(model.lower(), _MODEL_COST_PER_M) is not None
            has_decision = has_rate
        assert reported == (not has_decision), f"{provider}/{model}: reported={reported}, has_rate={has_rate}"


def test_every_provider_in_catalog_has_cost_fallback():
    """Every catalog provider must appear in _COST_PER_M — otherwise every one
    of its models prices at (0,0)."""
    for provider in KNOWN_MODELS:
        assert provider in _COST_PER_M, (
            f"provider {provider!r} missing from _COST_PER_M; "
            f"all its models price at (0,0) (fallback={_COST_PER_M.get(provider)})"
        )


def test_provider_fallbacks_are_nonzero_for_nonfree_providers():
    """Provider fallbacks should be the safety net, not (0,0) — a deliberately
    zero-rated provider (ollama) is allowed and must be the only one."""
    for provider in KNOWN_MODELS:
        if provider in ("ollama",):
            continue
        in_rate, out_rate = _COST_PER_M[provider]
        assert in_rate or out_rate, f"{provider} fallback is (0,0) — every unknown model of this provider prices free"


def test_cache_rate_or_provider_discount_for_every_catalog_model():
    """Cache-aware cost must also not silently misprice: either a model-specific
    cache rate, or a provider-level cache discount, must exist."""
    for provider, model in _every_catalog_model():
        if provider == "opencode":
            cached = _longest_prefix_match(model.lower(), _OPENCODE_CACHE_RATE)
        else:
            cached = _longest_prefix_match(model.lower(), _MODEL_CACHE_RATE)
        if provider == "opencode" and model in OPENCODE_UNPRICED:
            continue  # unpriced models are exempt from cache-rate completeness
        discount = _COST_PER_M.get(provider, (0, 0))[0] * 0.1
        assert cached is not None or discount, (
            f"{provider}/{model} has no cache pricing (no model rate, no provider discount)"
        )


def test_opencode_catalog_models_priced_via_opencode_table_not_openai():
    """opencode is served by an OpenAI-protocol client, so naive cost
    accounting with provider="openai" would apply the OpenAI fallback
    (5.00/15.00) — wrong for every opencode model. The OpenCodeClient must
    report provider_name="opencode" so _get_rates consults the OpenCode table.
    Here we assert the table lookup itself is correct for each catalog id."""
    for model in KNOWN_MODELS.get("opencode", []):
        in_rate, _ = _get_rates("opencode", model)
        if model in OPENCODE_UNPRICED:
            # Unpriced by design: falls to the opencode provider floor (0.22/0.66).
            assert in_rate == _COST_PER_M["opencode"][0], model
        else:
            assert in_rate > 0.0, f"opencode/{model} unpriced outside allowlist"
        # Never the OpenAI fallback rate by accidental prefix match:
        assert in_rate != _COST_PER_M["openai"][0], f"opencode/{model} leaks OpenAI pricing"


def test_opencode_client_reports_opencode_provider():
    """The runtime cost-accounting path: create_llm_client('opencode') must
    yield a client whose provider_name is 'opencode' (not 'openai'), or cost
    estimates would silently use the OpenAI fallback."""
    from external_llm.client import create_llm_client

    client = create_llm_client("opencode", api_key="sk-test")
    assert client.get_provider_name() == "opencode"
    assert client.base_url == "https://opencode.ai/zen/go/v1"


def test_opencode_cache_rates_match_go_zen_sheet():
    """A handful of spot values from the official price sheet (verified
    2026-08-25 https://opencode.ai/docs/go/ and /docs/zen/), so a re-paste
    typo breaks here instead of silently mispricing."""
    assert _OPENCODE_COST_PER_M["grok-4.5"] == (2.00, 6.00)
    assert _OPENCODE_COST_PER_M["gpt-5.6-luna"] == (0.20, 1.20)
    assert _OPENCODE_COST_PER_M["kimi-k3"] == (3.00, 15.00)
    assert _OPENCODE_COST_PER_M["longcat-2.0"] == (0.30, 1.20)
    assert _OPENCODE_COST_PER_M["minimax-m3"] == (0.30, 1.20)
    assert _OPENCODE_COST_PER_M["qwen3.6-plus"] == (0.50, 3.00)
    assert _OPENCODE_COST_PER_M["glm-5.3"] == (1.40, 4.40)
    assert _OPENCODE_COST_PER_M["deepseek-v4-flash"] == (0.22, 0.66)
    assert _OPENCODE_CACHE_RATE["kimi-k2.6"] == 0.16
    assert _OPENCODE_CACHE_RATE["qwen3.7-plus"] == 0.04


def test_unpriced_report_names_exact_missing_models():
    """The exposed drift-report helper lists the exact provider/model pairs so
    a fix can be driven from the output. Pinned to the CURRENT known gaps —
    each entry is a catalog model with no model-specific rate (it still gets
    the provider fallback, so cost is not zero, just not model-accurate).
    Adding a NEW catalog model without a rate entry makes this fail, which is
    the gate: the model either gains a rate entry or the list is extended by
    a deliberate, reviewed change."""
    assert _catalog_unpriced_models() == {
        "anthropic": ["claude-opus-5", "claude-opus-4-6"],
        "google": [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro",
            "gemini-3-flash",
            "gemini-2.5-flash",
        ],
        "openai": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "o3"],
    }
