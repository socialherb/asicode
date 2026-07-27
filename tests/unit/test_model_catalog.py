"""Contract tests for external_llm.model_catalog — the single source that
asi.py (/model display), the kp verify tool (validation) and the webapp
picker endpoint all read. Before it existed the three surfaces kept
hand-synced copies and drifted on every refresh."""
from external_llm.model_catalog import (
    KNOWN_MODELS,
    LEGACY_MODELS,
    MODEL_ALIASES,
    valid_models,
    webapp_external_model_groups,
)


def test_valid_models_is_current_plus_legacy_in_order():
    for provider in ("anthropic", "deepseek", "google"):
        assert valid_models(provider) == (
            KNOWN_MODELS[provider] + LEGACY_MODELS[provider]
        ), provider
    # A provider with no legacy tier degrades to the current list.
    assert valid_models("openai") == KNOWN_MODELS["openai"]
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

    def test_deepseek_offers_endpoint_ids_not_catalog_ids(self):
        """DeepSeek's API routes the fixed endpoint IDs; the picker must offer
        what the API actually serves, overriding the display catalog."""
        values = {
            o["value"]
            for g in webapp_external_model_groups() for o in g["options"]
        }
        assert "external_deepseek:deepseek-chat" in values
        assert "external_deepseek:deepseek-reasoner" in values
        assert "external_deepseek:deepseek-v4-pro" not in values

    def test_legacy_ids_are_not_offered(self):
        """Legacy tier is validation-only; pickers show the current catalog."""
        values = {
            o["value"]
            for g in webapp_external_model_groups() for o in g["options"]
        }
        for legacy_google in LEGACY_MODELS["google"]:
            assert f"external_google:{legacy_google}" not in values
