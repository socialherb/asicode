"""Regression: ``_cache_hit_ratio`` must use a provider-aware denominator.

For separate-accounting providers (Anthropic/zai) the reported
``prompt_tokens`` (= ``input_tokens``) EXCLUDES cache tokens, so the true
context size is ``prompt + cache_read + cache_creation``. The old local
formula divided by ``prompt_tokens`` only, yielding ratios far above 1.0 on
Anthropic cache-heavy turns. This pins the corrected behavior (delegating to
``cache_hit_pct``) and guards the subset-provider path (OpenAI/DeepSeek) and
the explicit-kwargs call form used by design_chat_loop.
"""
from types import SimpleNamespace

from external_llm.agent.agent_turn_pipeline import _cache_hit_ratio


def _ctx(provider, prompt, cache_read, cache_creation=0):
    return SimpleNamespace(
        provider_name=provider,
        total_prompt_tokens=prompt,
        total_cache_read_tokens=cache_read,
        total_cache_creation_tokens=cache_creation,
    )


def test_anthropic_denominator_includes_cache_creation_and_read():
    # prompt=100 excludes cache; real context = 100 + 900 (read) + 1000 (create)
    # -> ratio = 900 / 2000 = 0.45. Old buggy formula returned 900/100 = 9.0.
    r = _cache_hit_ratio(_ctx("anthropic", prompt=100, cache_read=900, cache_creation=1000))
    assert r == 0.45


def test_anthropic_without_creation_still_uses_read_in_denominator():
    # 900 / (100 + 900) = 0.9
    r = _cache_hit_ratio(_ctx("anthropic", prompt=100, cache_read=900, cache_creation=0))
    assert r == 0.9


def test_zai_is_treated_as_separate_accounting():
    # same as anthropic
    r = _cache_hit_ratio(_ctx("zai", prompt=50, cache_read=450, cache_creation=500))
    assert r == 0.45


def test_subset_provider_prompt_already_includes_cache():
    # OpenAI/DeepSeek: prompt already includes cached reads in denominator.
    # 400 / 1000 = 0.4
    r = _cache_hit_ratio(_ctx("openai", prompt=1000, cache_read=400, cache_creation=0))
    assert r == 0.4


def test_explicit_kwargs_form_carries_provider_and_creation():
    # design_chat_loop call form (no ctx): must still apply correct denominator.
    r = _cache_hit_ratio(
        cache_read_tokens=900,
        cache_creation_tokens=1000,
        prompt_tokens=100,
        provider="anthropic",
    )
    assert r == 0.45


def test_zero_cache_read_returns_zero():
    assert _cache_hit_ratio(_ctx("anthropic", prompt=100, cache_read=0, cache_creation=500)) == 0.0


def test_ratio_never_exceeds_one():
    # Guard against the old bug (ratio > 1) for any separate provider.
    for provider in ("anthropic", "zai"):
        r = _cache_hit_ratio(_ctx(provider, prompt=1, cache_read=9999, cache_creation=0))
        assert 0.0 < r <= 1.0
