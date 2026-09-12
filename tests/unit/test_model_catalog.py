"""Contract tests for external_llm.model_catalog — the single source that
asi.py (/model display), the kp verify tool (validation) and the webapp
picker endpoint all read. Before it existed the three surfaces kept
hand-synced copies and drifted on every refresh."""

from external_llm.model_catalog import (
    KNOWN_MODELS,
    LEGACY_MODELS,
    MODEL_ALIASES,
    WEBAPP_MODEL_OVERRIDES,
    WEBAPP_PROVIDER_GROUPS,
    valid_models,
    webapp_external_model_groups,
)


def _webapp_values() -> list[str]:
    """Every option value the webapp picker offers, in group order."""
    return [o["value"] for g in webapp_external_model_groups() for o in g["options"]]


def test_valid_models_is_current_plus_legacy_in_order():
    # Derived from the tables, so a provider that drops its legacy tier
    # (deepseek, 2026-09-11) needs no edit here to keep the gate honest.
    for provider in set(KNOWN_MODELS) | set(LEGACY_MODELS):
        expected = KNOWN_MODELS.get(provider, []) + LEGACY_MODELS.get(provider, [])
        assert valid_models(provider) == expected, provider
    # A provider with no legacy tier degrades to the current list.
    for provider in ("openai", "deepseek"):
        assert provider not in LEGACY_MODELS, provider
        assert valid_models(provider) == KNOWN_MODELS[provider]
    assert valid_models("no-such-provider") == []


def test_no_id_is_both_current_and_legacy():
    for provider, legacy in LEGACY_MODELS.items():
        overlap = set(legacy) & set(KNOWN_MODELS.get(provider, []))
        assert not overlap, f"{provider}: {overlap} listed in both tiers"


def test_every_alias_target_is_a_known_model():
    """An alias pointing at a dropped ID would silently resolve users onto a
    model no surface recognizes."""
    all_ids = {m for models in KNOWN_MODELS.values() for m in models}
    all_ids |= {m for models in LEGACY_MODELS.values() for m in models}
    for alias, target in MODEL_ALIASES.items():
        assert target in all_ids, f"alias {alias!r} -> unknown id {target!r}"


def test_gateway_versioned_spelling_aliases_onto_the_vendor_id():
    """DeepSeek V4.1 Flash is served twice by the opencode Go gateway.

    The vendor id is ``deepseek-flash`` — the native API routes exactly that
    (``/models`` returns deepseek-flash + deepseek-v4-pro) — while the gateway
    also accepts the versioned spelling ``deepseek-v4.1-flash``. models.dev
    reports one model for both (family ``deepseek-flash``, 1M/384K, $0.15/$0.60,
    2026-09-10). The versioned spelling must therefore alias onto the vendor id
    rather than become a second catalog entry: a second id would star the same
    model twice in the picker and need its own context, pricing and capability
    decisions. The live-API gate in test_model_catalog_context_parity keeps the
    alias target honest — it fails if the gateway stops serving the short id.

    Known gap, pre-existing and shared with every alias-only spelling (e.g.
    ``hy3-preview``): the bare form ``/model deepseek-v4.1-flash`` is rejected by
    ``_model_candidates``, which scans the catalog only, so the alias conversion
    that runs later never sees it. ``/model opencode/deepseek-v4.1-flash`` does
    resolve and is reported as auto-corrected, and env-var / saved-session
    spellings resolve through the lookup tables (``bare_model_name``).
    """
    assert MODEL_ALIASES.get("deepseek-v4.1-flash") == "deepseek-flash", (
        "the gateway's versioned spelling must resolve to the vendor id"
    )
    # Alias resolution lands on the id the vendor's own /models serves, so the
    # corrected spelling is routable on the native route too — a target only a
    # gateway serves would leave native-route users with an unrouted id.
    assert "deepseek-flash" in KNOWN_MODELS["deepseek"]


def test_native_deepseek_tier_is_the_vendor_registry():
    """The native tier IS the vendor registry — nothing more, nothing less.

    One endpoint, one id list: ``api.deepseek.com/v1/models`` answers with exactly
    these two ids (verified live 2026-09-11; the live gate in
    test_model_catalog_context_parity re-reads it on every run). The tier used to
    hold two spellings that only the opencode gateway serves, so
    ``/model deepseek/deepseek-v4-flash`` switched the session to an endpoint that
    answers 404 for it.
    """
    assert KNOWN_MODELS["deepseek"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert "deepseek-v4-flash" not in KNOWN_MODELS["deepseek"]


def test_ids_the_vendor_stopped_routing_are_not_reintroduced():
    """``deepseek-chat`` / ``deepseek-reasoner`` left the catalog on 2026-09-11.

    Re-adding either one re-arms what this file fixes: the vendor registry answers
    404 "Model Not Found" for both — the same answer a bogus control id gets — and
    ``/chat/completions`` cannot prove them alive either, because a zero-balance key
    is answered 402 "Insufficient Balance" *before* the model is validated. A pinned
    session spelling one now fails resolution/validation loudly instead of 404ing at
    the API, and the picker no longer offers it.
    """
    for dead in ("deepseek-chat", "deepseek-reasoner"):
        assert dead not in valid_models("deepseek"), dead
        assert f"external_deepseek:{dead}" not in _webapp_values()


class TestWebappGroups:
    def test_value_encoding_and_slug_preservation(self):
        groups = webapp_external_model_groups()
        values = [o["value"] for g in groups for o in g["options"]]
        assert values, "picker would be empty"
        assert len(values) == len(set(values)), "duplicate option values"
        for v in values:
            assert v.startswith("external_") and ":" in v, v
        # OpenRouter models keep their vendor/model slug inside the value —
        # the webapp's generic prefix stripper relies on the colon split.
        assert "external_openrouter:deepseek/deepseek-v4-flash" in values

    def test_deepseek_offers_the_ids_the_native_api_routes(self):
        """The picker's DeepSeek options must be the tier its client can serve.

        The old override offered ``deepseek-chat``/``deepseek-reasoner`` — the ids
        the vendor API stopped routing — so no DeepSeek option in the webapp could
        work, while the live ``deepseek-v4-pro`` was withheld on purpose.
        """
        values = set(_webapp_values())
        for model in KNOWN_MODELS["deepseek"]:
            assert f"external_deepseek:{model}" in values
        for dead in ("deepseek-chat", "deepseek-reasoner"):
            assert f"external_deepseek:{dead}" not in values

    def test_legacy_ids_are_not_offered(self):
        """Legacy tier is validation-only; pickers show the current catalog."""
        values = set(_webapp_values())
        for legacy_google in LEGACY_MODELS["google"]:
            assert f"external_google:{legacy_google}" not in values


class TestWebappPickerIsTierOnly:
    """The webapp picker may shorten a tier, never extend it.

    ``WEBAPP_MODEL_OVERRIDES`` was free-form, which is how its DeepSeek entry came
    to advertise two ids the vendor registry does not route while withholding the
    two it does: every DeepSeek option in the webapp was dead and no other gate
    could see it, because an override value is not a catalog id — it appears in no
    parity gate, no live gate and no rate table.
    """

    def test_override_ids_are_all_in_the_known_tier(self):
        for provider, models in WEBAPP_MODEL_OVERRIDES.items():
            tier = KNOWN_MODELS.get(provider)
            assert tier is not None, f"override for unknown provider {provider!r}"
            stray = [m for m in models if m not in tier]
            assert not stray, f"{provider}: webapp override names non-tier ids {stray}"

    def test_each_picker_provider_offers_exactly_its_expected_list(self):
        values = _webapp_values()
        for group_label, providers in WEBAPP_PROVIDER_GROUPS:
            assert group_label, "group needs a label"
            for provider, provider_label in providers:
                expected = WEBAPP_MODEL_OVERRIDES.get(provider) or KNOWN_MODELS.get(provider, [])
                offered = [v for v in values if v.startswith(f"external_{provider}:")]
                assert offered == [f"external_{provider}:{m}" for m in expected], provider
                assert provider_label, f"{provider} needs a display label"
                assert expected, f"{provider} would offer an empty picker"
