"""Parity gate: the model-keyed lookup tables must cover the catalog SSOT.

``external_llm.model_catalog`` exists because three surfaces kept hand-copying
model data and drifting; ``test_model_catalog_private_surfaces`` pins the kp
verify tool to it. Two more surfaces were never pinned, and both had drifted:

* ``context_budget._CONTEXT_LIMITS`` — six Claude ids were missing, including
  the two flagships ``KNOWN_MODELS["anthropic"]`` was actively offering
  (``claude-opus-5``, ``claude-fable-5``). A missing id takes the silent 1M
  fallback against a real 200K window, which leaves the pre-flight cap in
  ``agent_loop`` (the ONLY window guard — tool-result eviction is off by
  default) inert until ~988K estimated tokens, i.e. it never fires before the
  provider rejects the request.
* ``model_registry.CLOUD_PROVIDER_PREFIXES`` — ``"o3-"`` carried a trailing
  hyphen while the catalog offers bare ``o3``, so ``detect_cloud_provider``
  returned None and its caller's ``or "deepseek"`` sent an OpenAI model to
  DeepSeek.

The pre-existing ``test_known_models_in_table`` could not catch either: it
iterates ``_CONTEXT_LIMITS.items()``, so it only checks that *listed* entries
resolve, never that catalog models are listed. These tests run the other
direction — catalog first — which is the direction that fails silently.

Shipping-only imports (``external_llm.*``), so this rides into the public
snapshot: the contract has to hold in the wheel, not just the private tree.
"""

from __future__ import annotations

import pytest

from external_llm.agent.context_budget import (
    _CONTEXT_LIMITS,
    _DEFAULT_CONTEXT_LIMIT,
    _FALLBACK_IS_CORRECT,
    _resolve_base_context_limit,
    _resolve_context_limit,
)
from external_llm.model_catalog import KNOWN_MODELS, LEGACY_MODELS, MODEL_ALIASES
from external_llm.model_registry import bare_model_name, detect_cloud_provider

# ``detect_cloud_provider`` maps a model to the API that natively serves it, so
# only the native tiers can be asserted against their catalog key. "opencode"
# and "openrouter" are gateways that resell other vendors' models — glm-5.2
# served by opencode still detects as "zai", correctly.
_NATIVE_PROVIDER_TIERS = ("anthropic", "deepseek", "openai", "google", "zai")


def _catalog_entries() -> list[tuple[str, str]]:
    """Every (provider, model id) pair the catalog can hand to a lookup."""
    return [
        (provider, model)
        for source in (KNOWN_MODELS, LEGACY_MODELS)
        for provider, models in source.items()
        for model in models
    ]


@pytest.fixture(autouse=True)
def _reset_context_overrides():
    """Clear the process-global reactive overrides before each test.

    ``_resolve_context_limit`` consults ``_context_window_overrides`` first, and
    those dicts outlive a test file (tests/unit/agent has its own autouse copy
    of this; this file sits outside that directory and cannot inherit it).
    """
    from external_llm.agent.context_budget import (
        _context_window_overrides,
        _override_meta,
    )
    _context_window_overrides.clear()
    _override_meta.clear()
    yield


def test_every_catalog_model_has_a_decided_context_window():
    """No catalog model may reach the 1M fallback by omission.

    The fallback is silent and errs toward over-allocation, so absence from
    ``_CONTEXT_LIMITS`` cannot distinguish "1M is right" from "nobody added
    it". ``_FALLBACK_IS_CORRECT`` is where the former is recorded; anything in
    neither is the latter.
    """
    undecided = sorted({
        f"{model} (catalog: {provider})"
        for provider, model in _catalog_entries()
        if bare_model_name(model) not in _CONTEXT_LIMITS
        and bare_model_name(model) not in _FALLBACK_IS_CORRECT
    })
    assert not undecided, (
        "catalog models with no context-window decision — add each to "
        "_CONTEXT_LIMITS with its real window, or to _FALLBACK_IS_CORRECT if "
        "1M is genuinely right:\n  " + "\n  ".join(undecided)
    )


def test_anthropic_catalog_models_all_resolve_to_200k():
    """Regression pin for the drift this gate was written for.

    The table's own header says every modern Claude shares a 200K window; six
    ids nonetheless resolved to 1M. Asserted on the real entry point, not the
    table, so a lookup-path regression fails here too.
    """
    wrong = {
        model: _resolve_context_limit(model)
        for model in KNOWN_MODELS["anthropic"] + LEGACY_MODELS["anthropic"]
        if _resolve_context_limit(model) != 200_000
    }
    assert not wrong, f"Claude models not resolving to 200K: {wrong}"


def test_openrouter_slugs_resolve_like_their_bare_id():
    """A vendor-prefixed slug must not get a different window than the bare id.

    ``anthropic/claude-sonnet-5`` and ``claude-sonnet-5`` are one model; the
    exact-match lookup gave the first 1M and the second 200K.
    """
    for slug in KNOWN_MODELS["openrouter"]:
        bare = bare_model_name(slug)
        assert _resolve_base_context_limit(slug) == _resolve_base_context_limit(bare), (
            f"{slug} resolves differently from its bare id {bare}"
        )


def test_routing_prefixed_spellings_resolve_like_the_bare_id():
    """The colon routing forms the CLI and webapp emit must normalise too."""
    for spelling, bare in (
        ("anthropic:claude-opus-5", "claude-opus-5"),
        ("openrouter:anthropic/claude-sonnet-5", "claude-sonnet-5"),
        ("openrouter/anthropic/claude-sonnet-5", "claude-sonnet-5"),
        ("ANTHROPIC:Claude-Opus-5", "claude-opus-5"),
    ):
        assert _resolve_base_context_limit(spelling) == _resolve_base_context_limit(bare), (
            f"{spelling!r} did not resolve like {bare!r}"
        )


def test_deprecated_aliases_inherit_their_targets_window():
    """An alias is the same model under an old name, so it is the same window."""
    for alias, target in MODEL_ALIASES.items():
        assert _resolve_base_context_limit(alias) == _resolve_base_context_limit(target), (
            f"alias {alias!r} resolves differently from its target {target!r}"
        )


def test_native_provider_catalog_models_are_routable():
    """Every natively-served catalog model must route to the provider that lists it.

    ``detect_cloud_provider`` returning None is not inert: its one live caller
    (webapp/routes/agent_stream) reads None as "not a cloud model" and falls
    back to DeepSeek, so an unroutable id is a *misroute*, not a failure.
    """
    misrouted = {
        model: detect_cloud_provider(model)
        for provider, model in _catalog_entries()
        if provider in _NATIVE_PROVIDER_TIERS
        and detect_cloud_provider(model) != provider
    }
    assert not misrouted, f"catalog models routed to the wrong provider: {misrouted}"


def test_fallback_allowlist_has_no_stale_entries():
    """The allowlist may only name models the catalog actually offers.

    Without this it rots into a junk drawer: a removed model's entry would sit
    there forever, and a renamed one would look decided while its new id
    silently took the fallback.
    """
    catalog_bare = {bare_model_name(m) for _, m in _catalog_entries()}
    stale = sorted(_FALLBACK_IS_CORRECT - catalog_bare)
    assert not stale, f"_FALLBACK_IS_CORRECT names non-catalog models: {stale}"


def test_fallback_allowlist_is_disjoint_from_the_table():
    """A model is either given a window or declared to want the fallback.

    Both at once means the allowlist entry is dead and its comment is lying
    about why the model has the window it has.
    """
    overlap = sorted(_FALLBACK_IS_CORRECT & set(_CONTEXT_LIMITS))
    assert not overlap, f"models both listed and allowlisted: {overlap}"


def test_allowlisted_models_really_take_the_fallback():
    """Guards the allowlist's premise: these must actually resolve to 1M.

    If one starts resolving elsewhere (an entry added to _CONTEXT_LIMITS, a
    normalisation change), the allowlist entry is stale and the disjointness
    test above may not be what catches it.
    """
    for model in sorted(_FALLBACK_IS_CORRECT):
        assert _resolve_base_context_limit(model) == _DEFAULT_CONTEXT_LIMIT, (
            f"{model} is allowlisted as taking the 1M fallback but does not"
        )
