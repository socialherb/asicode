import pytest

from external_llm.agent._shared_utils import (
    _longest_prefix_match,
    cache_cost_summary,
    cache_hit_pct,
    estimate_cache_adjusted_cost,
    estimate_cost,
    extract_files_from_patch,
    make_tool_signature,
    total_input_tokens,
)

# Rates are ($/1M input, $/1M output) — mirror _COST_PER_M in _shared_utils.
_RATES = {
    "anthropic": (3.00, 15.00),
    "deepseek": (0.27, 1.10),
    "google": (0.10, 0.40),
}


class TestEstimateCost:
    """Tests for estimate_cost (signature: provider, prompt_tok, completion_tok)."""

    @pytest.mark.parametrize("provider", ["anthropic", "deepseek", "google"])
    def test_rate_table(self, provider):
        in_rate, out_rate = _RATES[provider]
        cost = estimate_cost(provider, 2000, 1000)
        expected = (2000 * in_rate + 1000 * out_rate) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_unknown_provider_is_free(self):
        # Unknown provider falls back to (0.0, 0.0) — it does not raise.
        assert estimate_cost("unknown_provider", 100, 100) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("anthropic", 0, 0) == 0.0


class TestEstimateCacheAdjustedCost:
    """Tests for estimate_cache_adjusted_cost across both token-accounting models."""

    def test_deepseek_subset_applies_discount(self):
        # DeepSeek: cache_read ⊆ prompt → cached portion re-priced down.
        # Provider-level discount for "deepseek" (no model) is 0.26.
        in_rate, out_rate = _RATES["deepseek"]
        discount = 0.26
        cost = estimate_cache_adjusted_cost("deepseek", 12327, 3312, cache_read_tok=3968)
        raw = (12327 * in_rate + 3312 * out_rate) / 1_000_000
        savings = 3968 * in_rate * (1 - discount) / 1_000_000
        assert cost == pytest.approx(raw - savings, rel=1e-9)
        assert cost < raw

    def test_anthropic_separate_adds_cache_on_top(self):
        # Anthropic: prompt EXCLUDES cache → read added at the discounted rate.
        in_rate, _ = _RATES["anthropic"]
        cost = estimate_cache_adjusted_cost("anthropic", 15308, 749, cache_read_tok=23574)
        expected = (15308 * in_rate + 749 * 15.00 + 23574 * in_rate * 0.1) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_anthropic_cache_read_exceeding_prompt_stays_positive(self):
        # Regression: cache_read > prompt previously drove cost NEGATIVE.
        cost = estimate_cache_adjusted_cost("anthropic", 15308, 749, cache_read_tok=23574)
        assert cost > 0

    def test_anthropic_cache_creation_premium(self):
        # Cache writes cost a 25% premium over the base input rate.
        in_rate, _ = _RATES["anthropic"]
        cost = estimate_cache_adjusted_cost(
            "anthropic", 1000, 0, cache_read_tok=0, cache_creation_tok=10000
        )
        expected = (1000 * in_rate + 10000 * in_rate * 1.25) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_no_discount_provider_unchanged(self):
        base = estimate_cost("google", 1000, 500)
        adjusted = estimate_cache_adjusted_cost("google", 1000, 500, cache_read_tok=500)
        assert adjusted == pytest.approx(base, rel=1e-9)

    def test_cache_read_zero_equals_base(self):
        for provider in ("anthropic", "deepseek"):
            cost = estimate_cache_adjusted_cost(provider, 1000, 500, cache_read_tok=0)
            assert cost == pytest.approx(estimate_cost(provider, 1000, 500), rel=1e-9)


class TestCacheHitPct:
    """Hit-rate denominator must be provider-aware (never exceed 100%)."""

    def test_anthropic_uses_total_input_denominator(self):
        # Regression: 23574/15308 = 154% under the old (DeepSeek) formula.
        pct = cache_hit_pct("anthropic", 15308, 23574)
        assert pct == pytest.approx(23574 * 100.0 / (15308 + 23574), rel=1e-9)
        assert 0.0 <= pct <= 100.0

    def test_deepseek_uses_prompt_denominator(self):
        # DeepSeek prompt already includes the cached reads as a subset.
        assert cache_hit_pct("deepseek", 12327, 3968) == pytest.approx(
            3968 * 100.0 / 12327, rel=1e-9
        )

    def test_zai_uses_total_input_denominator(self):
        # Regression: ZAIAnthropicClient serves GLM over the Anthropic Messages
        # API, so input_tokens EXCLUDES cache_read (same usage shape as
        # Anthropic). Under the subset (DeepSeek) formula a small uncached
        # prompt + large cache read blows past 100% — the design-chat loop was
        # showing ⚡3241% (↑1.4K) and ⚡9363% (↑419). Separate denominator yields
        # a sane <100% figure.
        pct = cache_hit_pct("zai", 1400, 45374)
        assert pct == pytest.approx(45374 * 100.0 / (1400 + 45374), rel=1e-9)
        assert 0.0 <= pct <= 100.0
        # Repro of the ↑419 ⚡9363% case from the loop:
        assert cache_hit_pct("zai", 419, 39210) < 100.0

    def test_zero_cache_read(self):
        assert cache_hit_pct("anthropic", 1000, 0) == 0.0
        assert cache_hit_pct("zai", 1000, 0) == 0.0

    def test_total_input_tokens_includes_cache_creation(self):
        # Regression: Anthropic input_tokens EXCLUDES both cache_read AND
        # cache_creation. The true context size the model ingested is therefore
        # prompt + cache_read + cache_creation. Omitting cache_creation makes
        # the ↑ display drop on cache-WRITE turns (cold start / post-eviction
        # prefix re-write) even though the context actually grew.
        # separate accounting: creation is additive
        assert total_input_tokens("anthropic", 1500, 40000, 5000) == 46500
        assert total_input_tokens("zai", 1500, 40000, 5000) == 46500
        # default cache_creation_tok=0 → unchanged (backward compatible)
        assert total_input_tokens("zai", 1400, 45374) == 1400 + 45374
        # subset accounting: creation is irrelevant (always 0), total == prompt
        assert total_input_tokens("deepseek", 50000, 0, 0) == 50000

    def test_cache_hit_pct_denominator_includes_cache_creation(self):
        # cache-WRITE tokens are part of the context but NOT served from cache,
        # so they lower the ratio on cache-WRITE turns (e.g. post-eviction prefix
        # re-write), instead of being silently excluded from both numerator and
        # denominator.
        without_creation = cache_hit_pct("anthropic", 1500, 40000, 0)
        with_creation = cache_hit_pct("anthropic", 1500, 40000, 5000)
        assert with_creation < without_creation
        assert with_creation == pytest.approx(
            40000 * 100.0 / (1500 + 40000 + 5000), rel=1e-9
        )
        assert 0.0 <= with_creation <= 100.0


class TestCacheCostSummary:
    """cache_cost_summary returns (full_counterfactual, actual_billed, hit_pct)."""

    def test_anthropic_actual_below_full_for_reads(self):
        full, actual, hit = cache_cost_summary("anthropic", 15308, 749, 23574, 0)
        assert actual < full          # arrow reads as a real saving
        assert actual > 0             # not negative
        assert 0.0 <= hit <= 100.0    # not 154%

    def test_deepseek_matches_legacy_helpers(self):
        full, actual, _hit = cache_cost_summary("deepseek", 12327, 3312, 3968, 0)
        assert full == pytest.approx(estimate_cost("deepseek", 12327, 3312), rel=1e-9)
        assert actual == pytest.approx(
            estimate_cache_adjusted_cost("deepseek", 12327, 3312, 3968), rel=1e-9
        )


class TestDeepSeekModelCacheRate:
    """DeepSeek charges per-model cached-input rates (source: api-docs.deepseek.com).

    The v4 models have very low cache-hit rates (~2%/0.8%) while the deprecated
    chat/reasoner models are ~26%. Model-specific rates in ``_MODEL_CACHE_RATE``
    must take precedence over the provider-level ``_CACHE_DISCOUNT`` fallback.
    """

    @pytest.mark.parametrize("model,cache_rate", [
        ("deepseek-v4-flash",  0.0028),
        ("deepseek-v4-pro",    0.003625),
        ("deepseek-chat",      0.07),
        ("deepseek-reasoner",  0.14),
        ("deepseek-r1",        0.14),
    ])
    def test_model_specific_cache_rate_used(self, model, cache_rate):
        """Model-specific cached rate must be used instead of provider discount."""
        # DeepSeek uses subset accounting: prompt INCLUDES cache_read.
        prompt, cached, completion = 10000, 5000, 2000
        # Get the input rate for this model.
        in_rate = {
            "deepseek-v4-flash": 0.14, "deepseek-v4-pro": 0.435,
            "deepseek-chat": 0.27, "deepseek-reasoner": 0.55,
            "deepseek-r1": 0.55,
        }[model]
        out_rate = {
            "deepseek-v4-flash": 0.28, "deepseek-v4-pro": 0.87,
            "deepseek-chat": 1.10, "deepseek-reasoner": 2.19,
            "deepseek-r1": 2.19,
        }[model]
        actual = estimate_cache_adjusted_cost(
            "deepseek", prompt, completion, cached, model=model,
        )
        # Subset formula: raw - cached * (in_rate - cache_rate)
        raw = (prompt * in_rate + completion * out_rate) / 1_000_000
        expected = raw - cached * (in_rate - cache_rate) / 1_000_000
        assert actual == pytest.approx(expected, rel=1e-9)
        # Must be cheaper than no-cache cost.
        assert actual < estimate_cost("deepseek", prompt, completion, model=model)

    def test_v4_flash_cache_is_very_cheap(self):
        """v4-flash cache hit at $0.0028/1M is dramatically cheaper than full input."""
        actual = estimate_cache_adjusted_cost(
            "deepseek", 100_000, 5000, cache_read_tok=80_000, model="deepseek-v4-flash",
        )
        full = estimate_cost("deepseek", 100_000, 5000, model="deepseek-v4-flash")
        # With 80% cache hit at v4-flash's $0.0028/1M, savings should be substantial.
        assert actual < full * 0.5  # more than 50% cheaper

    def test_unknown_deepseek_uses_fallback(self):
        """Unknown DeepSeek model falls back to _CACHE_DISCOUNT['deepseek'] = 0.26."""
        from external_llm.agent._shared_utils import _get_cached_input_rate
        cached = _get_cached_input_rate("deepseek", 0.5, "deepseek-v5-flash")
        assert cached == pytest.approx(0.5 * 0.26, rel=1e-9)


class TestOpenRouterDeepSeekCache:
    """OpenRouter applies a flat 10% cache discount on its own input rate.

    Per OpenRouter docs (DEEPSEEK_CACHE_READ_MULTIPLIER = 0.1), cache reads are
    charged at 10% of the input rate — NOT the native provider's per-model rate.
    This means OpenRouter v4-flash cache = $0.09 * 0.1 = $0.009/1M, which is
    different from native v4-flash cache = $0.0028/1M.
    """

    def test_flat_10_percent_discount(self):
        """OpenRouter DeepSeek cache uses 10% of OpenRouter's input rate."""
        from external_llm.agent._shared_utils import _get_cached_input_rate
        # v4-flash: OpenRouter input = $0.09/1M
        cached = _get_cached_input_rate("openrouter", 0.09, "deepseek/deepseek-v4-flash")
        assert cached == pytest.approx(0.09 * 0.1, rel=1e-9)  # $0.009

    def test_v4_pro_flat_discount(self):
        """OpenRouter v4-pro also uses flat 10% (not native's 0.8%)."""
        from external_llm.agent._shared_utils import _get_cached_input_rate
        cached = _get_cached_input_rate("openrouter", 0.435, "deepseek/deepseek-v4-pro")
        assert cached == pytest.approx(0.435 * 0.1, rel=1e-9)  # $0.0435

    def test_unknown_openrouter_deepseek_uses_fallback(self):
        """Unknown OpenRouter model falls back to openrouter flat 10%."""
        from external_llm.agent._shared_utils import _get_cached_input_rate
        cached = _get_cached_input_rate("openrouter", 0.5, "deepseek/deepseek-v5-flash")
        assert cached == pytest.approx(0.5 * 0.1, rel=1e-9)

    def test_cost_differs_from_native(self):
        """OpenRouter cache cost differs from native — this is expected (different input rates)."""
        or_cost = estimate_cache_adjusted_cost(
            "openrouter", 100_000, 5000, 50_000, model="deepseek/deepseek-v4-flash",
        )
        native_cost = estimate_cache_adjusted_cost(
            "deepseek", 100_000, 5000, 50_000, model="deepseek-v4-flash",
        )
        # Both should be cheaper than no-cache cost.
        assert or_cost < estimate_cost("openrouter", 100_000, 5000, model="deepseek/deepseek-v4-flash")
        assert native_cost < estimate_cost("deepseek", 100_000, 5000, model="deepseek-v4-flash")
        # They differ because input rates differ (OpenRouter 35% cheaper on flash).
        assert or_cost != native_cost


class TestZaiModelCacheRate:
    """Z.AI charges a per-model cached-input rate (not a provider discount).

    Source: https://docs.z.ai/guides/overview/pricing (verified 2026-06)
        GLM-5.2: input $1.40, cached $0.26
        GLM-4.6: input $0.60,  cached $0.11
    Z.AI is served over the Anthropic Messages API (ZAIAnthropicClient), so its
    usage shape matches Anthropic: ``prompt_tok`` (input_tokens) EXCLUDES the
    separately-reported cached tokens. Per-model cached rates still apply, and
    Z.AI's cache storage is free so no creation premium applies.
    """

    @pytest.mark.parametrize("model,in_rate,cached_rate", [
        ("glm-5.2", 1.40, 0.26),
        ("glm-5",   1.00, 0.20),
        ("glm-4.6", 0.60, 0.11),
        ("glm-4.5", 0.60, 0.11),
    ])
    def test_separate_reprice_bit_exact(self, model, in_rate, cached_rate):
        # 12327 uncached input + 3968 cached (reported separately), 3312 output.
        # prompt_tok EXCLUDES cached (Anthropic usage shape); cached is billed at
        # its own rate on top of the full-priced prompt.
        # Pass PAYG base_url so the model-specific cached rate applies.
        prompt, cached, completion = 12327, 3968, 3312
        out_rate = {"glm-5.2": 4.40, "glm-5": 3.20, "glm-4.6": 2.20, "glm-4.5": 2.20}[model]
        actual = estimate_cache_adjusted_cost(
            "zai", prompt, completion, cached, model=model,
            base_url="https://api.z.ai/paas/v4/chat",
        )
        expected = (prompt * in_rate + cached * cached_rate
                    + completion * out_rate) / 1_000_000
        assert actual == pytest.approx(expected, rel=1e-9)
        # Cheaper than billing ALL input (prompt + cached) at the full rate.
        assert actual < estimate_cost("zai", prompt + cached, completion, model=model)

    def test_no_model_falls_back_to_full_price(self):
        # No model-specific cached rate AND no provider-level discount → cached
        # tokens billed at the full input rate, so adjusted equals the cost of
        # ALL input (prompt + cached) at full price (no savings).
        prompt, cached, completion = 12327, 3968, 3312
        adjusted = estimate_cache_adjusted_cost(
            "zai", prompt, completion, cached  # no model=
        )
        assert adjusted == pytest.approx(
            estimate_cost("zai", prompt + cached, completion), rel=1e-9
        )

    def test_summary_shows_savings_for_glm52(self):
        # Regression: before model-specific rates, actual == full (no savings).
        # Pass PAYG base_url so the model-specific cached rate applies.
        full, actual, hit = cache_cost_summary(
            "zai", 12327, 3312, 3968, 0, model="glm-5.2",
            base_url="https://api.z.ai/paas/v4/chat",
        )
        assert actual < full
        # Separate accounting: hit = cached / (input + cached).
        assert hit == pytest.approx(3968 * 100.0 / (12327 + 3968), rel=1e-9)
        assert 0.0 <= hit <= 100.0

    def test_longest_prefix_shadows_shorter_regardless_of_order(self):
        """Longest-prefix match must win even if a shorter prefix is listed first.

        Regression guard: a naive first-match scan is insertion-order dependent.
        If ``glm-5`` were listed before ``glm-5.2`` in the cost table, the
        specific rate ($0.26) would be shadowed by the generic ($0.20). The
        longest-prefix helper resolves ``glm-5.2`` to its own rate no matter
        the dict ordering.
        """
        # Reordered table: shorter prefix FIRST (worst case for naive scan).
        reversed_table = {"glm-5": 0.20, "glm-5.2": 0.26}
        assert _longest_prefix_match("glm-5.2", reversed_table) == 0.26
        # And the production table resolves both tiers correctly.
        from external_llm.agent._shared_utils import _MODEL_CACHE_RATE
        assert _longest_prefix_match("glm-5.2", _MODEL_CACHE_RATE) == 0.26
        assert _longest_prefix_match("glm-5", _MODEL_CACHE_RATE) == 0.20

    # ── Coding Plan (default, no base_url) ──────────────────────────────────

    def test_coding_plan_cached_at_full_price(self):
        """z.ai Coding Plan (no base_url): cached tokens billed at full in_rate."""
        prompt, cached, completion = 12327, 3968, 3312
        actual = estimate_cache_adjusted_cost(
            "zai", prompt, completion, cached, model="glm-5.2",
        )
        # No discount: cost = (prompt + cached) * in_rate + completion * out_rate
        expected = estimate_cost("zai", prompt + cached, completion, model="glm-5.2")
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_coding_plan_summary_shows_no_savings(self):
        """cache_cost_summary for Coding Plan: actual == full (no savings)."""
        full, actual, hit = cache_cost_summary(
            "zai", 12327, 3312, 3968, 0, model="glm-5.2",
        )
        assert actual == pytest.approx(full, rel=1e-9)
        assert 0.0 <= hit <= 100.0

    # ── Pay-as-you-go (explicit /paas/v4 base_url) ──────────────────────────

    _PAYG_URL = "https://api.z.ai/paas/v4/chat/completions"

    def test_payg_discount_applies(self):
        """z.ai PAYG endpoint: cached tokens at model-specific discount rate."""
        prompt, cached, completion = 12327, 3968, 3312
        actual = estimate_cache_adjusted_cost(
            "zai", prompt, completion, cached, model="glm-5.2",
            base_url=self._PAYG_URL,
        )
        # Should be cheaper than full price (discount applied).
        full_price = estimate_cost("zai", prompt + cached, completion, model="glm-5.2")
        assert actual < full_price

    def test_payg_summary_shows_savings(self):
        """cache_cost_summary for PAYG: actual < full (savings)."""
        full, actual, hit = cache_cost_summary(
            "zai", 12327, 3312, 3968, 0, model="glm-5.2",
            base_url=self._PAYG_URL,
        )
        assert actual < full
        assert 0.0 <= hit <= 100.0

    # ── URL detection variants ──────────────────────────────────────────────

    def test_is_zai_payg_url_variants(self):
        from external_llm.agent._shared_utils import _is_zai_payg_url
        # Coding Plan endpoints
        assert not _is_zai_payg_url("")
        assert not _is_zai_payg_url("https://api.z.ai/anthropic/v1/messages")
        assert not _is_zai_payg_url("https://api.z.ai/coding/paas/v4/chat/completions")
        assert not _is_zai_payg_url("https://api.z.ai/PAAS/V4/anthropic/v1")
        # PAYG endpoint
        assert _is_zai_payg_url("https://api.z.ai/paas/v4/chat/completions")
        assert _is_zai_payg_url("https://api.z.ai/PAAS/V4/chat/completions")


class TestExtractFilesFromPatch:
    """Tests for extract_files_from_patch."""

    def test_plus_format(self):
        patch = "+++ b/src/main.py\n@@ -1 +1 @@\n"
        result = extract_files_from_patch(patch)
        assert result == ["src/main.py"]

    def test_diff_git_format(self):
        patch = "diff --git a/src/utils.py b/src/utils.py\n"
        result = extract_files_from_patch(patch)
        assert result == ["src/utils.py"]

    def test_both_formats_deduplication(self):
        patch = (
            "diff --git a/file_a.py b/file_a.py\n"
            "+++ b/file_a.py\n"
            "diff --git a/file_b.py b/file_b.py\n"
            "+++ b/file_b.py\n"
        )
        result = extract_files_from_patch(patch)
        assert result == ["file_a.py", "file_b.py"]

    def test_empty_input(self):
        result = extract_files_from_patch("")
        assert result == []

    def test_no_match(self):
        patch = "some random text without plus or diff"
        result = extract_files_from_patch(patch)
        assert result == []

    def test_case_sensitivity(self):
        patch = "+++ b/README.md\n"
        result = extract_files_from_patch(patch)
        assert result == ["README.md"]

    def test_multiple_files_plus(self):
        patch = (
            "+++ b/first.py\n"
            "+++ b/second.py\n"
            "--- a/third.py\n"
        )
        result = extract_files_from_patch(patch)
        assert result == ["first.py", "second.py"]


class TestMakeToolSignature:
    """Tests for make_tool_signature — stable tool-call keying.

    Regression guard for the `hash(json.dumps(...))` antipattern that caused
    (a) cross-process instability via PYTHONHASHSEED and (b) 64-bit collision
    risk between distinct arg dicts. The signature is consumed by
    AgentLoop._tool_key (success/failure memory) and the turn pipeline's
    fail_streak loop-detection key, so a collision would falsely trip a
    STRATEGY WARNING.
    """

    def test_deterministic_same_inputs(self):
        """Same (tool_name, args) must always produce the same signature."""
        args = {"path": "foo.py", "patch": "@@ -1 +1 @@\n-old\n+new\n"}
        assert make_tool_signature("apply_patch", args) == make_tool_signature("apply_patch", args)

    def test_dict_key_order_invariance(self):
        """Dict insertion order must not affect the signature."""
        args_a = {"path": "foo.py", "patch": "x", "mode": "replace"}
        args_b = {"mode": "replace", "patch": "x", "path": "foo.py"}
        assert make_tool_signature("apply_patch", args_a) == make_tool_signature("apply_patch", args_b)

    def test_different_tool_name_distinct(self):
        args = {"path": "foo.py"}
        assert make_tool_signature("apply_patch", args) != make_tool_signature("edit_text", args)

    def test_different_args_distinct(self):
        """Different arg contents must not collide (the old hash() bug)."""
        a = make_tool_signature("apply_patch", {"path": "foo.py", "patch": "@@ -1 +1 @@\n-old\n+new"})
        b = make_tool_signature("apply_patch", {"path": "foo.py", "patch": "@@ -1 +1 @@\n-old\n+DIFFERENT"})
        assert a != b

    def test_cross_process_stable(self, tmp_path):
        """Signature must be invariant across interpreter launches.

        This is the core regression for PYTHONHASHSEED: the old `hash()` of a
        str changed per-process, breaking persisted/cross-process memory.
        """
        import subprocess
        import sys

        code = (
            "from external_llm.agent._shared_utils import make_tool_signature;"
            "print(make_tool_signature('grep', {'pattern': 'x', 'path': 'y'}))"
        )
        out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        assert out1 == out2
        assert out1  # non-empty

    def test_returns_hex_digest(self):
        """Output should be a sha256 hex string (64 hex chars)."""
        sig = make_tool_signature("bash", {"command": "ls"})
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# ── P4: bounded module-level cache eviction ─────────────────────────────────


def test_capped_put_evicts_oldest_when_over_cap():
    """P4: _capped_put FIFO-evicts the oldest entry when the cache exceeds its
    cap, bounding memory in a long-lived REPL that visits many repos."""
    from external_llm.agent._shared_utils import _capped_put

    cache: dict = {}
    for i in range(12):
        _capped_put(cache, f"repo-{i}", [f"f{i}"], cap=8)
    # Cap held: only the 8 most-recent entries survive.
    assert len(cache) == 8
    # Oldest (repo-0..repo-3) evicted; newest (repo-4..repo-11, capped to 8) kept.
    assert "repo-0" not in cache and "repo-3" not in cache
    assert "repo-11" in cache and "repo-4" in cache


def test_capped_put_refreshes_existing_key_in_place():
    """Re-inserting an existing key updates its value (no eviction of it)."""
    from external_llm.agent._shared_utils import _capped_put

    cache: dict = {"a": 1, "b": 2}
    _capped_put(cache, "a", 99, cap=8)
    assert cache["a"] == 99
    assert len(cache) == 2


def test_archive_capped_put_bounds_archive_cache():
    """P4: the insights archive cache helper also bounds its entry count."""
    from external_llm.agent.insights_manager import _archive_capped_put, _ARCHIVE_CACHE_MAX_ENTRIES

    cache: dict = {}
    for i in range(_ARCHIVE_CACHE_MAX_ENTRIES + 5):
        _archive_capped_put(cache, f"p{i}", (i, i, i, []))
    assert len(cache) == _ARCHIVE_CACHE_MAX_ENTRIES


# ── Walk pruning predicate ────────────────────────────────────────────────
# Regression guard: _walk_py_files and _walk_ts_js_files previously used
# *divergent* predicates. The TS/JS walker carried a redundant ``node_modules``
# substring check (already in _WALK_SKIP_DIRS) while MISSING ``venv*`` and
# ``site-packages`` dirs — letting vendored JS/TS bundled inside a Python
# package pollute the index. Both walkers now share _walk_should_skip_dir, so
# these cases must be skipped regardless of which walker runs.

def test_walk_should_skip_dir_excludes_hidden_vendor_venv_and_site_packages():
    """_walk_should_skip_dir must skip hidden, vendor, venv*, site-packages, egg-info."""
    from external_llm.agent._shared_utils import _walk_should_skip_dir

    # NOTE: the venv* branch uses startswith("venv"), so it catches "venv",
    # "venv310", "venv-proj" but NOT "myvenv" (only contains "venv"). This is
    # the *original* _walk_py_files semantics, preserved verbatim — broadening
    # to `"venv" in d` would also catch benign dirs like "invention"/"prevention".
    skip = [
        ".git", ".hidden", ".venv", ".mypy_cache",              # hidden / dot-dirs
        "__pycache__", "node_modules", "env", "build", "dist",  # _WALK_SKIP_DIRS exact
        "venv", "venv310", "venv-proj",                         # venv* prefix (TS/JS regressed here)
        "site-packages", "lib.site-packages",                   # site-packages substring (TS/JS regressed here)
        "foo.egg-info", "pkg.egg-info",                         # *.egg-info
    ]
    for d in skip:
        assert _walk_should_skip_dir(d), f"expected {d!r} to be skipped"


def test_walk_should_skip_dir_keeps_normal_source_dirs():
    """_walk_should_skip_dir must NOT skip ordinary source directories."""
    from external_llm.agent._shared_utils import _walk_should_skip_dir

    keep = ["src", "tests", "external_llm", "agent", "lib", "pkg", "mypy", "utils"]
    for d in keep:
        assert not _walk_should_skip_dir(d), f"expected {d!r} to be kept"


# ── Walk cache memoization & isolation (consolidation regression guard) ──────
# _walk_py_files / _walk_ts_js_files are now thin wrappers over the shared
# _walk_repo_files engine, but they MUST keep *separate* per-extension caches:
# a .py walk must never satisfy a later .ts lookup (and vice versa). These lock
# in the design decision that justifies keeping two wrappers + two caches
# instead of collapsing into a single walker.

def test_walk_caches_are_isolated_per_extension_set(tmp_path):
    """A py walk populates _PY_WALK_CACHE only; a ts/js walk the ts cache only."""
    from external_llm.agent import _shared_utils as su

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.ts").write_text("const y = 2;\n")

    # Clear cross-test residue so we observe a fresh population.
    su._PY_WALK_CACHE.clear()
    su._TS_WALK_CACHE.clear()

    py_files = su._walk_py_files(tmp_path, max_files=100)
    ts_files = su._walk_ts_js_files(tmp_path, max_files=100)

    assert {p.name for p in py_files} == {"a.py"}
    assert {p.name for p in ts_files} == {"b.ts"}

    # Each walk populated ONLY its own cache (no cross-contamination).
    assert str(tmp_path) in su._PY_WALK_CACHE
    assert str(tmp_path) in su._TS_WALK_CACHE
    assert {p.name for p in su._PY_WALK_CACHE[str(tmp_path)][1]} == {"a.py"}
    assert {p.name for p in su._TS_WALK_CACHE[str(tmp_path)][1]} == {"b.ts"}

def test_walk_ts_js_consumes_TS_JS_EXTENSIONS_constant(tmp_path, monkeypatch):
    """_walk_ts_js_files must honor the _TS_JS_EXTENSIONS constant (single source
    of truth), NOT an inline literal. Guards against drift that left the declared
    constant dead while the walker re-defined the extensions inline (turn review)."""
    from external_llm.agent import _shared_utils as su

    (tmp_path / "a.ts").write_text("x")
    (tmp_path / "b.customext").write_text("y")
    su._TS_WALK_CACHE.clear()

    # Re-point the constant to a DIFFERENT extension set; the walker must follow.
    monkeypatch.setattr(su, "_TS_JS_EXTENSIONS", (".customext",))
    got = {p.name for p in su._walk_ts_js_files(tmp_path, max_files=10)}
    assert "b.customext" in got, "walker must read _TS_JS_EXTENSIONS at call time"
    assert "a.ts" not in got, "walker must not carry a stale inline extension list"


def test_walk_extension_constants_derived_from_family_groups():
    """_TS_JS_EXTENSIONS / _PY_EXTENSIONS must be derived from the
    _LANGUAGE_EXTENSION_GROUPS SSOT, not hardcoded literals. A hardcoded tuple
    drifted from the SSOT and silently dropped .mts/.cts/.mjs/.cjs/.pyi from
    find_symbol / call_graph (confirmed regression). This pins the derivation so
    re-hardcoding (or adding a family extension without the walker picking it up)
    fails loudly."""
    from external_llm.agent import _shared_utils as su
    from external_llm.languages.models import _LANGUAGE_EXTENSION_GROUPS

    ts_js_group = next(g for g in _LANGUAGE_EXTENSION_GROUPS if ".ts" in g)
    py_group = next(g for g in _LANGUAGE_EXTENSION_GROUPS if ".py" in g)
    assert set(su._TS_JS_EXTENSIONS) == set(ts_js_group)
    assert set(su._PY_EXTENSIONS) == set(py_group)
    # The drift class that motivated this: modern JS/TS + Python stub extensions.
    for ext in (".mts", ".cts", ".mjs", ".cjs"):
        assert ext in su._TS_JS_EXTENSIONS, f"{ext} must be in the JS/TS walker set"
    assert ".pyi" in su._PY_EXTENSIONS, ".pyi must be in the Python walker set"


def test_walk_ts_js_files_covers_modern_extensions(tmp_path):
    """_walk_ts_js_files must discover .mts/.cts/.mjs/.cjs files, not just the
    legacy .ts/.tsx/.js/.jsx quartet."""
    from external_llm.agent import _shared_utils as su

    for ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs"):
        (tmp_path / f"mod{ext}").write_text("x = 1\n")
    su._TS_WALK_CACHE.clear()
    got = {p.suffix for p in su._walk_ts_js_files(tmp_path, max_files=100)}
    assert got == {".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs"}


def test_walk_py_files_covers_type_stubs(tmp_path):
    """_walk_py_files must discover .pyi type-stub files, not just .py."""
    from external_llm.agent import _shared_utils as su

    (tmp_path / "real.py").write_text("x = 1\n")
    (tmp_path / "stub.pyi").write_text("y: int\n")
    su._PY_WALK_CACHE.clear()
    got = {p.name for p in su._walk_py_files(tmp_path, max_files=100)}
    assert got == {"real.py", "stub.pyi"}


def test_walk_repo_files_returns_cached_on_second_call(tmp_path):
    """Second call within TTL returns a copy of the cached result — same contents,
    distinct object (shallow copy prevents cache pollution from caller mutations)."""
    from external_llm.agent._shared_utils import _walk_repo_files

    (tmp_path / "one.py").write_text("x = 1\n")
    cache: dict = {}
    first = _walk_repo_files(tmp_path, 100, cache, lambda n: n.endswith(".py"))
    # Mutate the filesystem after the first walk; a TTL cache hit must ignore it.
    (tmp_path / "two.py").write_text("y = 2\n")
    second = _walk_repo_files(tmp_path, 100, cache, lambda n: n.endswith(".py"))
    assert second is not first  # distinct object → shallow copy, not reference
    assert {p.name for p in second} == {"one.py"}  # two.py invisible (cached)
    assert {p.name for p in first} == {"one.py"}  # original also unchanged


def test_walk_repo_files_truncated_cache_does_not_leak_to_larger_cap(tmp_path):
    """When a walk is truncated (hit max_files), a later caller with a larger
    cap must NOT receive the truncated list — it would silently miss files and
    cause spurious "unused symbol" / "not found" errors (the exact failure mode
    described in the design insight).  The cache-hit guard re-walks when the
    cached ``len(files) < max_files`` for a truncated entry."""
    from external_llm.agent._shared_utils import _walk_repo_files

    for i in range(5):
        (tmp_path / f"f{i}.py").write_text(f"x = {i}\n")
    cache: dict = {}

    # First walk with cap=3 → truncated, 3 files cached.
    small = _walk_repo_files(tmp_path, 3, cache, lambda n: n.endswith(".py"))
    assert len(small) == 3

    # Second walk with cap=5 → cache hit but truncated+insufficient → re-walk.
    big = _walk_repo_files(tmp_path, 5, cache, lambda n: n.endswith(".py"))
    assert len(big) == 5, f"Expected 5 files, got {len(big)} — truncated cache leaked"
    assert {p.name for p in big} == {f"f{i}.py" for i in range(5)}

    # Third walk with cap=3 → cache has 5 untruncated files → cache hit, but
    # the result is sliced to the caller's cap. Callers consume the list
    # directly without re-slicing, so an over-long result would cause
    # redundant work (e.g. 4000-cap vulture cache served to a 600-cap
    # symbol_search call would index 4000 files instead of 600).
    small2 = _walk_repo_files(tmp_path, 3, cache, lambda n: n.endswith(".py"))
    assert len(small2) == 3, "Cache hit must honor the caller's max_files cap"
    # Should have re-used the cache, not re-walked (verify by checking that
    # the result is a *different* object — copy, not reference).
    assert small2 is not big


def test_walk_repo_files_complete_cache_serves_smaller_cap_without_overshoot(tmp_path):
    """A COMPLETE (untruncated) walk cached under a large cap must slice down to
    a smaller caller's cap. Models the real cross-caller hazard: vulture scans
    with max_files=4000 and populates _PY_WALK_CACHE; symbol_search then asks for
    max_files=600 and must NOT receive 4000 files (callers use the list directly,
    so an over-long result triggers redundant symbol indexing)."""
    from external_llm.agent._shared_utils import _walk_repo_files

    for i in range(8):
        (tmp_path / f"f{i}.py").write_text(f"x = {i}\n")
    cache: dict = {}

    # First walk with a cap larger than the file count → complete, 8 files cached.
    big = _walk_repo_files(tmp_path, 4000, cache, lambda n: n.endswith(".py"))
    assert len(big) == 8

    # Second walk with a smaller cap → cache hit, but result sliced to the cap.
    small = _walk_repo_files(tmp_path, 3, cache, lambda n: n.endswith(".py"))
    assert len(small) == 3, (
        "Complete cache must be sliced to the caller's cap — got "
        f"{len(small)} files for a max_files=3 request"
    )
    assert small is not big  # still a distinct object (shallow copy after slice)


def test_walk_repo_files_miss_path_returns_copy_not_cached_object(tmp_path):
    """Regression: the cache-MISS paths previously returned the very list object
    stored in *cache* (only the HIT path returned a shallow copy). The FIRST
    caller to populate a cache entry therefore held the live cached object and
    could .append()/.sort() it, silently corrupting the cache for every later
    caller — exactly the pollution the HIT-path copy exists to prevent. Both
    MISS paths (truncated early-exit AND complete walk) must now return a copy."""
    from pathlib import Path

    from external_llm.agent._shared_utils import _walk_repo_files

    # ── Case A: truncated early-exit miss path (len(results) == max_files) ──
    for i in range(5):
        (tmp_path / f"a{i}.py").write_text("x = 1\n")
    cache: dict = {}
    first = _walk_repo_files(tmp_path, 3, cache, lambda n: n.endswith(".py"))
    assert len(first) == 3
    cached_obj = cache[str(tmp_path)][1]
    assert first is not cached_obj, (
        "Truncated miss path must return a COPY, not the cached list object"
    )
    # Mutate the returned copy; the cache entry must be unaffected.
    first.append(Path("/polluted.py"))
    assert Path("/polluted.py") not in cache[str(tmp_path)][1], (
        "Caller .append() on the returned list leaked into the cache"
    )

    # ── Case B: complete-walk miss path (walk finishes below max_files) ──
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("y = 1\n")
    cache2: dict = {}
    complete = _walk_repo_files(sub, 100, cache2, lambda n: n.endswith(".py"))
    assert len(complete) == 1
    cached_obj2 = cache2[str(sub)][1]
    assert complete is not cached_obj2, (
        "Complete-walk miss path must return a COPY, not the cached list object"
    )
    complete.append(Path("/polluted2.py"))
    assert Path("/polluted2.py") not in cache2[str(sub)][1]
