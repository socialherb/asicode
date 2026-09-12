"""Cross-validation: every catalog id reachable through the OpenAI-compatible
client family must DECLARE its capability vector.

``model_catalog.MODEL_CAPABILITIES`` is the source of truth for the dispatch
facts the OpenAI-compatible clients read (does the model spend part of
``max_tokens`` on a reasoning trace; which thinking control does it accept; does
it read image parts).  Name-prefix inference used to be that source of truth, and
it silently orphaned ``deepseek-flash``: the id of V4.1 Flash — the successor of
``deepseek-v4-flash`` — carries no ``v4``, so the model classified as
NON-reasoning. Its ``thinking_mode`` toggle became a no-op on the wire and
``/insights compact`` lost the 32k reasoning headroom its sibling got.

The IMAGE axis was orphaned the same way, from the other direction: the
``deepseek-v4`` family prefix matched ``deepseek-v4-flash-vision-exp``, so the one
id built to read images had its attachments replaced by OCR text — and because no
image part ever went out, the gateway could not return the 400 that would have
corrected the guess (2026-09-10 live probe: the id reads a red/blue test image
correctly while the same part 400s for ``deepseek-v4-flash``).

These tests make that class of drift a test failure instead of a silent default:
adding a catalog id to a covered provider without declaring it here fails, as
does leaving a declaration behind for an id the catalog no longer serves, as does
a prefix entry that would swallow a declared capability.
"""

from __future__ import annotations

import pytest

from external_llm.client import create_llm_client
from external_llm.model_catalog import (
    KNOWN_MODELS,
    LEGACY_MODELS,
    MODEL_CAPABILITIES,
    ThinkingControl,
)
from external_llm.model_registry import (
    TEXT_ONLY_MODEL_PREFIXES,
    bare_model_name,
    model_capabilities,
    text_only_model,
    vision_capable,
)
from external_llm.openai_client import (
    OpenAIClient,
    _is_deepseek_v4,
    _is_kimi_k3,
    _is_reasoning_model,
)

# Providers whose catalog ids are served by ``OpenAIClient`` or a subclass
# (OpenCodeClient / OpenRouterClient / ZAIClient) — i.e. the ids the classifiers
# behind this table actually see.  Pinned: a provider silently gaining or losing
# the OpenAI-compatible client changes which ids need a declaration, and that
# must be a deliberate edit here, not a surprise.
EXPECTED_OPENAI_PROTOCOL_PROVIDERS = {"openai", "opencode", "openrouter"}


def _openai_protocol_providers() -> set[str]:
    """Providers whose ``create_llm_client`` client is an ``OpenAIClient``.

    Derived from the factory rather than hand-listed, so it tracks the routing
    instead of a stale comment.
    """
    return {
        provider
        for provider in set(KNOWN_MODELS) | set(LEGACY_MODELS)
        if isinstance(create_llm_client(provider, api_key="sk-test"), OpenAIClient)
    }


def _catalog_entries() -> list[tuple[str, str, str]]:
    """``(provider, catalog id, bare id)`` for every entry, both tiers."""
    return [
        (provider, model, bare_model_name(model))
        for table in (KNOWN_MODELS, LEGACY_MODELS)
        for provider, models in table.items()
        for model in models
    ]


def test_openai_protocol_provider_set_is_what_this_contract_assumes():
    """The coverage rule below is only as good as its provider set."""
    assert _openai_protocol_providers() == EXPECTED_OPENAI_PROTOCOL_PROVIDERS


@pytest.mark.parametrize(
    "provider,model,bare",
    [entry for entry in _catalog_entries() if entry[0] in EXPECTED_OPENAI_PROTOCOL_PROVIDERS],
)
def test_every_openai_protocol_catalog_model_declares_capabilities(provider, model, bare):
    """No id may inherit the default — it must be decided.

    A new model either gains an entry (with the vector its behaviour requires)
    or an entry naming it explicitly as having no reasoning-specific dispatch.
    """
    assert bare in MODEL_CAPABILITIES, (
        f"{provider}/{model} (bare id {bare!r}) has no declared capability vector — "
        "add it to external_llm.model_catalog.MODEL_CAPABILITIES. Letting it inherit the "
        "default is how deepseek-flash became a non-reasoning model."
    )


def test_declared_ids_are_catalog_ids():
    """No stale or mistyped keys: a declaration for an id no surface serves is
    unreachable, and it hides the fact that the id's replacement is undeclared."""
    catalog_bare = {bare for _provider, _model, bare in _catalog_entries()}
    assert sorted(set(MODEL_CAPABILITIES) - catalog_bare) == []


def test_thinking_control_implies_a_reasoning_budget():
    """Table invariant: an id that accepts a thinking control necessarily spends
    part of ``max_tokens`` on a reasoning trace, so both flags travel together."""
    inconsistent = [
        bare
        for bare, caps in MODEL_CAPABILITIES.items()
        if caps.thinking is not ThinkingControl.NONE and not caps.reasoning
    ]
    assert inconsistent == []


@pytest.mark.parametrize("bare,caps", sorted(MODEL_CAPABILITIES.items()))
def test_classifiers_follow_the_declaration(bare, caps):
    """The table is not decorative: the classifiers read it before the prefixes,
    so a declaration is what the wire does."""
    assert _is_reasoning_model(bare) is caps.reasoning
    assert _is_deepseek_v4(bare) is (caps.thinking is ThinkingControl.THINKING)
    assert _is_kimi_k3(bare) is (caps.thinking is ThinkingControl.MAX_ONLY)


@pytest.mark.parametrize(
    "spelling",
    [
        "deepseek-flash",
        "opencode/deepseek-flash",
        "deepseek:deepseek-flash",
        "openrouter/deepseek/deepseek-flash",
        "deepseek/deepseek-flash",
        # The gateway's versioned spelling of the same model resolves through the
        # alias table to the vendor id, so it reads the same declaration.
        "deepseek-v4.1-flash",
    ],
)
def test_every_spelling_of_an_id_resolves_to_one_declaration(spelling):
    """A model arrives spelled however its route spells it; the lookup normalises
    (route + vendor prefixes, colon forms, aliases) before matching."""
    assert model_capabilities(spelling) == MODEL_CAPABILITIES["deepseek-flash"]


def test_deepseek_v4_family_shares_the_reasoning_vector_not_the_vision_axis():
    """The v4 line is one dispatch vector for reasoning — same budget, same native
    thinking control — while the IMAGE axis splits it: that split is precisely what
    a family prefix cannot express, and the reason ``deepseek-v4`` left
    ``TEXT_ONLY_MODEL_PREFIXES``.  Pinned per id from the 2026-09-10 live probes
    (documented on MODEL_CAPABILITIES): the ``-vision-exp`` id and V4.1 Flash read a
    red/blue test image, ``deepseek-v4-flash`` 400s on the part, and
    ``deepseek-v4-pro`` accepts then silently drops it (answers "NOIMAGE", so it has
    neither the image nor the OCR text)."""
    family = [
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash-vision-exp",
    ]
    for bare in family:
        caps = MODEL_CAPABILITIES[bare]
        assert (caps.reasoning, caps.thinking) == (True, ThinkingControl.THINKING), bare
    assert {bare: MODEL_CAPABILITIES[bare].vision for bare in family} == {
        "deepseek-flash": True,
        "deepseek-v4-flash-vision-exp": True,
        "deepseek-v4-flash": False,
        "deepseek-v4-pro": False,
    }


@pytest.mark.parametrize("bare,caps", sorted(MODEL_CAPABILITIES.items()))
def test_image_classifier_follows_the_declaration(bare, caps):
    """The fourth classifier obeys the same contract as the three reasoning ones:
    the declared vector is what goes on the wire, and ``text_only_model`` /
    ``vision_capable`` are exact inverses of it."""
    assert text_only_model(bare) is (not caps.vision)
    assert vision_capable(bare) is caps.vision


def test_image_classifier_still_falls_back_to_prefixes_for_undeclared_ids():
    """Declaration first, prefix as the fallback — an id outside the catalog is
    still classified by name (the native-endpoint DeepSeekClient ids), and an
    unknown id is assumed able to receive images (the client's strip-and-retry net
    absorbs a rejection)."""
    assert model_capabilities("deepseek-chat") is None
    assert text_only_model("deepseek-chat")
    assert text_only_model("openrouter/deepseek/deepseek-reasoner")
    assert vision_capable("some-unreleased-model")
    assert not text_only_model("deepseek-v4-flash-vision-exp")


def test_no_vision_declaration_is_swallowed_by_the_prefix_table():
    """A family prefix must never cover an id declared vision-capable.  This is the
    exact hole ``deepseek-v4`` left: it matched ``deepseek-v4-flash-vision-exp``, so
    the classifier pre-converted that id's attachments to OCR text — and since no
    image part ever went out, no 400 could ever correct the guess."""
    swallowed = [
        bare
        for bare, caps in MODEL_CAPABILITIES.items()
        if caps.vision and bare_model_name(bare).startswith(TEXT_ONLY_MODEL_PREFIXES)
    ]
    assert swallowed == []


def test_alias_spelling_reads_the_declaration_of_its_target():
    """Alias resolution runs before the lookup, so a deprecated spelling inherits
    the target id's vector (``deepseek-v4`` → ``deepseek-v4-pro``)."""
    assert model_capabilities("deepseek-v4") == MODEL_CAPABILITIES["deepseek-v4-pro"]


def test_undeclared_id_reports_none_not_a_default():
    """``None`` means "no decision recorded" — the signal callers use to fall back
    to prefix inference (variants, hand-typed ids), which is NOT the same thing as
    "declared as having no capabilities"."""
    assert model_capabilities("kimi-k3-0711") is None
    assert model_capabilities("some-unreleased-model") is None
    assert model_capabilities("") is None
