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

import json
import os
import urllib.error
import urllib.request

import pytest

from external_llm.agent.context_budget import (
    _CONTEXT_LIMITS,
    _DEFAULT_CONTEXT_LIMIT,
    _FALLBACK_IS_CORRECT,
    _resolve_base_context_limit,
    _resolve_context_limit,
)
from external_llm.model_catalog import KNOWN_MODELS, LEGACY_MODELS, MODEL_ALIASES, valid_models
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
    undecided = sorted(
        {
            f"{model} (catalog: {provider})"
            for provider, model in _catalog_entries()
            if bare_model_name(model) not in _CONTEXT_LIMITS and bare_model_name(model) not in _FALLBACK_IS_CORRECT
        }
    )
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
        if provider in _NATIVE_PROVIDER_TIERS and detect_cloud_provider(model) != provider
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


def test_opencode_catalog_is_up_to_date_with_live_api():
    """The opencode tier must track what the live API actually serves.

    The catalog is hand-curated from ``https://opencode.ai/zen/go/v1/models``;
    this pins the glue so a new model shipped upstream (or one removed) breaks
    the gate instead of silently starring a stale picker.

    Live model ids are fetched at test time (26 ids on 2026-08-14), so the
    gate self-updates as the API evolves.
    """
    # Any request shape works for this endpoint; a dummy auth is fine because
    # the models list is served before auth is enforced.
    req = urllib.request.Request(
        "https://opencode.ai/zen/go/v1/models",
        headers={
            "Authorization": "Bearer dummy-key",
            # The endpoint 403s the urllib default User-Agent; curl's is fine.
            "User-Agent": "curl/8.0",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.load(resp)
    live_ids = {m["id"] for m in payload["data"]}

    catalog = set(KNOWN_MODELS["opencode"])
    # hy3-preview is deliberately omitted (MODEL_ALIASES → hy3);
    # deepseek-v4.1-flash is the versioned gateway spelling of deepseek-flash
    # (MODEL_ALIASES → the vendor id the native API routes); omen-alpha is
    # LEGACY (dropped from the Go plan 2026-09-10, id still served by the API).
    # All three are expected to appear on the live API but not in KNOWN_MODELS.
    expected_missing_from_catalog = {"hy3-preview", "deepseek-v4.1-flash", "omen-alpha"}
    assert catalog <= live_ids, (
        f"opencode catalog names a model the live API no longer serves: {sorted(catalog - live_ids)}"
    )
    newly_served = live_ids - catalog - expected_missing_from_catalog
    assert not newly_served, (
        "live opencode API serves models the catalog is missing — add them to "
        "KNOWN_MODELS['opencode'] and give each a context-window decision "
        f"in context_budget.py: {sorted(newly_served)}"
    )


def test_openrouter_catalog_slugs_all_resolve_on_live_api():
    """The openrouter tier must only name slugs the live API actually serves.

    Mirror of ``test_opencode_catalog_is_up_to_date_with_live_api`` for the
    openrouter gateway. Three openrouter slugs had drifted off the live API
    (2026-08-14): ``zai/glm-5.2`` (vendor prefix is ``z-ai`` today),
    ``anthropic/claude-sonnet-4-6`` (live slug uses the dot spelling
    ``claude-sonnet-4.6``), and ``qwen/qwen3.6`` (replaced by the served
    ``qwen/qwen3.6-plus``). This pins the glue so the next upstream rename or
    removal breaks the gate instead of silently poisoning the picker.

    Unlike the opencode tier, openrouter is an open gateway serving hundreds
    of models, and our catalog is a curated subset — so only the catalog→live
    direction is gated (a stale slug is a hard 404). Live model ids are
    fetched at test time; OpenRouter's endpoints are public, no auth needed.
    """
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "curl/8.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.load(resp)
    live_slugs = {m["id"] for m in payload["data"]}

    catalog = set(KNOWN_MODELS["openrouter"])
    assert catalog <= live_slugs, (
        f"openrouter catalog names a slug the live API no longer serves: {sorted(catalog - live_slugs)}"
    )


def test_provider_default_models_have_a_context_decision():
    """Every client-class DEFAULT_MODEL must resolve without the 1M fallback warning.

    ``ExternalLLMService._PROVIDER_DEFAULT_MODELS["ollama"]`` is ``""`` — the
    effective ollama default lives in ``OllamaClient.DEFAULT_MODEL``
    (``qwen2.5-coder:3b``), which was **not** in ``_CONTEXT_LIMITS``,
    ``_FAMILY_PREFIX_LIMITS``, or ``_FALLBACK_IS_CORRECT``. It therefore resolved
    to the 1M fallback and emitted the unknown-model warning on every first use
    of the default provider (the server is unreachable / no num_ctx in the
    Modelfile → the dynamic /api/show query returns None). Worse, the pre-flight
    cap in agent_loop scales to the 1M figment while ``_num_ctx_for_model``
    actually serves 8K-128K — a 122x over-allocation that left the cap inert.

    The catalog-parity tests only see catalog models, and ollama has no catalog
    tier (it is a local serving layer, not a cloud vendor), so the default-model
    contract had to be pinned separately. This walks every client class in
    ``providers.py`` and asserts its ``DEFAULT_MODEL`` resolves without the
    unknown-model warning — i.e. it is either explicitly tabled
    (``_CONTEXT_LIMITS`` / ``_FAMILY_PREFIX_LIMITS``) or allowlisted as a real 1M
    model (``_FALLBACK_IS_CORRECT``).
    """
    import ast
    import inspect

    import external_llm.providers as providers_mod
    from external_llm.agent.context_budget import (
        _resolve_base_context_limit,
        _warned_unknown_models,
    )

    tree = ast.parse(inspect.getsource(providers_mod))
    default_models: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DEFAULT_MODEL" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    default_models.append(stmt.value.value)

    assert default_models, "no DEFAULT_MODEL constants found in providers.py (AST walk broken)"
    warned: list[tuple[str, str]] = []
    for model in default_models:
        _warned_unknown_models.clear()
        try:
            _resolve_base_context_limit(model)
        except Exception as exc:  # pragma: no cover - unexpected resolver failure
            warned.append((model, f"resolver raised: {exc!r}"))
            continue
        if model in _warned_unknown_models:
            warned.append((model, "reached the 1M unknown-model fallback"))
    assert not warned, (
        "provider DEFAULT_MODEL with no context-window decision — add each to "
        "_CONTEXT_LIMITS (real window), _FAMILY_PREFIX_LIMITS (family), or "
        "_FALLBACK_IS_CORRECT (verified 1M):\n  " + "\n  ".join(f"{m}: {why}" for m, why in warned)
    )


# (provider, client class name in external_llm.providers, registry listing path,
# API-key env var). Only DeepSeek is listed: it is the one native tier whose
# ambient key authenticates, so its registry can actually be read. The keys the
# anthropic/openai tiers would need answer 401 here — a probe of those would go
# red for an auth reason, not a catalog one, and a silently-skipped probe would
# be worse than this honest absence (both tiers stay covered offline by
# test_client_default_models_are_in_their_tier below).
_LIVE_NATIVE_REGISTRIES = [
    ("deepseek", "DeepSeekClient", "/models", "DEEPSEEK_API_KEY"),
]


@pytest.mark.parametrize("provider,client_name,path,key_env", _LIVE_NATIVE_REGISTRIES)
def test_native_tier_is_served_by_the_vendor_registry(provider, client_name, path, key_env):
    """A native tier asserts what ONE endpoint routes — the vendor's own.

    The gateway gates above cannot cover this: a gateway serving an id proves
    nothing about the vendor behind it. That is how ``deepseek-v4-flash`` and
    ``deepseek-v4-flash-vision-exp`` sat in the deepseek tier — one of them being
    ``DeepSeekClient.DEFAULT_MODEL``, i.e. the id every modelless call sends —
    while ``api.deepseek.com`` answered 404 "Model Not Found" for both.

    Needs ``DEEPSEEK_API_KEY``: the registry requires auth (unlike the gateway
    listings) and a chat probe cannot substitute for it — a zero-balance key gets
    402 "Insufficient Balance" *before* the model is validated, for a valid id and
    a bogus one alike.
    """
    key = (os.getenv(key_env) or "").strip()
    if not key:
        pytest.skip(f"{key_env} not set — the vendor registry requires auth")

    import external_llm.providers as providers_mod

    client_cls = getattr(providers_mod, client_name)
    base = client_cls.DEFAULT_BASE_URL.rstrip("/")

    def _probe(suffix: str):
        req = urllib.request.Request(
            f"{base}{suffix}",
            headers={"Authorization": f"Bearer {key}", "User-Agent": "curl/8.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, {}

    status, payload = _probe(path)
    assert status == 200, f"{provider}: registry unreachable (HTTP {status})"
    live_ids = {m["id"] for m in payload.get("data", [])}
    assert live_ids, f"{provider}: registry returned no ids — a subset check against an empty set passes vacuously"

    catalog = set(valid_models(provider))
    assert catalog <= live_ids, (
        f"catalog names ids the {provider} vendor does not serve: {sorted(catalog - live_ids)} "
        f"(registry serves {sorted(live_ids)})"
    )
    # Per-id probe: the listing could omit an id the endpoint still routes, and a
    # set difference would bury that disagreement.
    absent = sorted(m for m in catalog if _probe(f"{path}/{m}")[0] != 200)
    assert not absent, f"{provider}: catalog ids the vendor answers non-200 for: {absent}"
    # Anti-vacuity: the per-id probe must discriminate. A bogus id has to come back
    # non-200, or "every catalog id answers 200" would be an artifact of an endpoint
    # that answers 200 to anything.
    bogus_status, _ = _probe(f"{path}/asicode-bogus-probe-xyz")
    assert bogus_status != 200, f"{provider}: registry answered {bogus_status} for a bogus id"
    # The id a modelless call falls back to must be one of them.
    assert client_cls.DEFAULT_MODEL in live_ids, (
        f"{provider}: {client_name}.DEFAULT_MODEL={client_cls.DEFAULT_MODEL!r} is not served by the vendor"
    )


def test_client_default_models_are_in_their_tier():
    """Offline companion to the live gate: a ``DEFAULT_MODEL`` outside its own
    provider's tier is unusable by construction — no picker offers it and no other
    gate here sees it (which is how the DeepSeek client's default became an id the
    vendor does not route).

    ``test_provider_default_models_have_a_context_decision`` above pins the
    *window* of these constants; this pins their existence in the catalog.
    """
    import external_llm.providers as providers_mod
    from external_llm.anthropic_client import AnthropicClient, ZAIAnthropicClient
    from external_llm.openai_client import OpenAIClient, OpenRouterClient, ZAIClient

    clients = {
        "anthropic": AnthropicClient,
        "deepseek": providers_mod.DeepSeekClient,
        "google": providers_mod.GoogleClient,
        "openai": OpenAIClient,
        "openrouter": OpenRouterClient,
        "zai": ZAIAnthropicClient,
    }
    # opencode is served by the generic OpenAIClient (whose default belongs to the
    # openai tier), so it is the one catalog provider without a class of its own.
    assert set(clients) == set(KNOWN_MODELS) - {"opencode"}, (
        "a catalog provider gained/lost a client class — give its DEFAULT_MODEL a tier check here"
    )
    for provider, client_cls in clients.items():
        assert client_cls.DEFAULT_MODEL in KNOWN_MODELS[provider], (
            f"{provider}: {client_cls.__name__}.DEFAULT_MODEL={client_cls.DEFAULT_MODEL!r} "
            f"is not in KNOWN_MODELS[{provider!r}]"
        )
    # zai is served by two protocol clients; neither may carry an off-tier id.
    assert ZAIClient.DEFAULT_MODEL in KNOWN_MODELS["zai"]
