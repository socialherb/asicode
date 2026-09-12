"""Cross-validation: the image axis is declared per ROUTE, not only per id.

``model_catalog.MODEL_CAPABILITIES`` is keyed on bare ids and is therefore
route-invariant by construction.  That was enough while every multi-route id had
the same image behaviour on all of its routes; ``deepseek-flash`` (V4.1 Flash)
broke that — it reads an image through the opencode gateway, is advertised with
image input on OpenRouter, and the vendor's own API documents none for the ids it
serves.  One bare entry then has to answer for two wires, and whichever answer it
carries is wrong somewhere:

* ``vision=True`` puts image parts on a route that may answer 200 and then ignore
  them — the failure mode ``deepseek-v4-pro`` shows on opencode, where the answer
  comes back literally "NOIMAGE" so the model gets neither the image nor the OCR
  text it could have had, and no 400 arrives to learn from;
* ``vision=False`` pre-converts the attachment to OCR text on a route that reads
  images natively — a capability silently dropped, and no request ever goes out
  that could reveal the mistake, because the static decision runs before the wire.

So the declaration has a route dimension: ``model_catalog.ROUTE_VISION``, keyed by
``(route_key(base_url), bare id)``.  These tests make the split mechanical instead
of a convention — a route that serves a declared id has to declare for itself, a
bare ``vision=False`` has to be what every serving route agrees on, and a route key
has to name an id that route actually serves.  The live probes at the bottom
re-measure the two routes that can be reached with a chat request.
"""

from __future__ import annotations

import base64
import os
import struct
import uuid
import zlib

import pytest

from external_llm.client import create_llm_client
from external_llm.model_catalog import (
    KNOWN_MODELS,
    LEGACY_MODELS,
    MODEL_CAPABILITIES,
    ROUTE_HOSTS,
    ROUTE_VISION,
    Route,
    route_key,
)
from external_llm.model_registry import (
    bare_model_name,
    model_capabilities,
    text_only_model,
    vision_capable,
)

# Which route each catalog provider's client posts to.  Pinned: a provider gaining
# a different base URL changes which ids need a route-scoped declaration, and that
# must be a deliberate edit here rather than a surprise in the table.
EXPECTED_PROVIDER_ROUTES = {
    "anthropic": Route.ANTHROPIC,
    "deepseek": Route.DEEPSEEK,
    "google": Route.GOOGLE,
    "openai": Route.OPENAI,
    "opencode": Route.OPENCODE,
    "openrouter": Route.OPENROUTER,
    "zai": Route.ZAI,
}


def _catalog_entries() -> list[tuple[str, str, str]]:
    """``(provider, catalog id, bare id)`` for every entry, both tiers."""
    return [
        (provider, model, bare_model_name(model))
        for table in (KNOWN_MODELS, LEGACY_MODELS)
        for provider, models in table.items()
        for model in models
    ]


def _provider_route(provider: str) -> Route | None:
    """Route the provider's own client posts to, read from the factory.

    Derived from the instance's ``base_url`` (falling back to the class default)
    rather than from a hand-written table, so it tracks
    ``client.resolve_provider_base_url`` — ``opencode`` is served by the generic
    ``OpenAIClient`` whose DEFAULT_BASE_URL belongs to the openai tier.
    """
    client = create_llm_client(provider, api_key="sk-test")
    return route_key(getattr(client, "base_url", "") or getattr(client, "DEFAULT_BASE_URL", ""))


def _serving_routes() -> dict[str, set[Route]]:
    """bare id → the routes whose catalog offers it."""
    routes: dict[str, set[Route]] = {}
    for provider, _model, bare in _catalog_entries():
        route = _provider_route(provider)
        if route is None:  # pragma: no cover - guarded by a test below
            continue
        routes.setdefault(bare, set()).add(route)
    return routes


def _route_base(route: Route) -> str:
    """A base URL that resolves to *route*, taken from the real client defaults."""
    for provider in sorted(set(KNOWN_MODELS) | set(LEGACY_MODELS)):
        client = create_llm_client(provider, api_key="sk-test")
        base = str(getattr(client, "base_url", "") or getattr(client, "DEFAULT_BASE_URL", "") or "")
        if route_key(base) is route:
            return base
    raise AssertionError(f"no catalog provider posts to {route.value!r} — the route key is unreachable")


# ── route_key: the wire identity, read from the host ──────────────────────────


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://opencode.ai/zen/go/v1", Route.OPENCODE),
        ("http://opencode.ai:8080/zen/go/v1", Route.OPENCODE),
        ("OPENCODE.AI", Route.OPENCODE),
        ("https://user:pw@opencode.ai/v1", Route.OPENCODE),
        ("https://api.openai.com/v1", Route.OPENAI),
        ("https://api.deepseek.com/v1", Route.DEEPSEEK),
        ("https://api.deepseek.com", Route.DEEPSEEK),
        ("https://openrouter.ai/api/v1", Route.OPENROUTER),
        ("https://api.z.ai/api/coding/paas/v4", Route.ZAI),
        ("https://api.z.ai/api/anthropic/v1", Route.ZAI),
        ("https://generativelanguage.googleapis.com/v1beta", Route.GOOGLE),
        ("https://api.anthropic.com/v1", Route.ANTHROPIC),
    ],
)
def test_route_key_reads_the_host_not_the_path(base, expected):
    """Scheme, port, credentials and path are not part of the wire identity.

    All of them vary between a deployment and its test double while the endpoint
    the request reaches is the same one, so a route fact keyed on the full URL
    would miss its own route.
    """
    assert route_key(base) is expected


@pytest.mark.parametrize(
    "base",
    [
        "",
        "   ",
        None,
        "http://127.0.0.1:11434",
        "https://my.proxy.internal/v1",
        "https://opencode.ai.evil.test/v1",
    ],
)
def test_a_host_with_no_route_has_no_route_key(base):
    """None means "no route facts", not "the default route".

    A custom gateway genuinely has no key here, and the lookup that follows an
    unknown base has to fall back to the route-agnostic vector rather than to some
    other route's measurements.  The lookalike host is the interesting case: a
    substring match would hand ``opencode.ai.evil.test`` the gateway's
    declarations, so matching is on the host suffix boundary.
    """
    assert route_key(base) is None


def test_every_catalog_provider_posts_to_its_declared_route():
    """The route table is only as good as the URLs it has to resolve."""
    derived = {provider: _provider_route(provider) for provider in sorted(set(KNOWN_MODELS) | set(LEGACY_MODELS))}
    assert derived == EXPECTED_PROVIDER_ROUTES, (
        "a catalog provider changed which endpoint it posts to — update ROUTE_HOSTS "
        "and this map together, or the route-scoped declarations point at the wrong wire"
    )


def test_every_declared_route_host_is_reachable():
    """No decorative host entry: each route key must be one a client actually uses."""
    reachable = set(EXPECTED_PROVIDER_ROUTES.values())
    declared = {route for _suffix, route in ROUTE_HOSTS}
    assert declared <= reachable, f"ROUTE_HOSTS names routes no catalog client posts to: {declared - reachable}"


# ── The table's own consistency ───────────────────────────────────────────────


def test_route_declarations_name_a_route_that_serves_the_id():
    """Every route key must describe a real pair: route serves id.

    A key for an id the route does not serve is unreachable, and it hides the fact
    that the id's replacement on that route is undeclared — the same orphan check
    the capability table gets, one dimension further out.
    """
    serving = _serving_routes()
    orphans = sorted((route, bare) for route, bare in ROUTE_VISION if bare not in serving or route not in serving[bare])
    assert orphans == [], f"declared for a (route, id) pair nobody serves: {orphans}"


def test_every_route_serving_a_declared_id_declares_for_itself():
    """Route-specific evidence must not leak — or be silently dropped.

    The moment an id's image axis is known on one route, every other route serving
    it is a decision the table has to make: an undeclared route inherits the bare
    vector, which is exactly how ``deepseek-flash`` kept a vendor-route answer that
    only held behind a gateway.
    """
    serving = _serving_routes()
    declared_routes: dict[str, set[Route]] = {}
    for route, bare in ROUTE_VISION:
        declared_routes.setdefault(bare, set()).add(route)
    missing = sorted(
        (bare, sorted(r.value for r in serving[bare] - declared_routes.get(bare, set())))
        for bare in declared_routes
        if serving[bare] - declared_routes.get(bare, set())
    )
    assert missing == [], f"route-scoped ids with an undeclared route: {missing}"


def test_bare_text_only_requires_every_serving_route_to_agree_with_it():
    """``vision=False`` in the bare vector is a claim about EVERY route.

    The permissive direction is self-correcting — an image goes out, a 400 comes
    back, the runtime net degrades that route and remembers it.  The negative
    direction is not: the OCR conversion happens before the wire, so no request can
    ever reveal that the answer was route-specific.  A route that disagrees must
    therefore be declared in ROUTE_VISION, and the bare entry must stay permissive.
    """
    serving = _serving_routes()
    declared = {(route, bare): value for (route, bare), value in ROUTE_VISION.items()}
    disagreements = sorted(
        (bare, sorted(r.value for r in serving[bare]))
        for bare, caps in MODEL_CAPABILITIES.items()
        if caps.vision is False and any(declared.get((route, bare)) is not False for route in serving.get(bare, set()))
    )
    assert disagreements == [], (
        "bare vision=False is a route-specific claim for these ids — move it into "
        f"ROUTE_VISION and leave the vector permissive: {disagreements}"
    )


# ── The declarations drive the classifiers ────────────────────────────────────


@pytest.mark.parametrize(
    "route,bare,declared",
    sorted(((route, bare, value) for (route, bare), value in ROUTE_VISION.items()), key=lambda p: (p[0].value, p[1])),
)
def test_route_declaration_decides_the_classifiers_on_that_route(route, bare, declared):
    """The table is not decorative: the classifiers read it before the prefixes."""
    base = _route_base(route)
    assert vision_capable(bare, base) is declared
    assert text_only_model(bare, base) is (not declared)


@pytest.mark.parametrize(
    "route,bare,declared",
    sorted(((route, bare, value) for (route, bare), value in ROUTE_VISION.items()), key=lambda p: (p[0].value, p[1])),
)
def test_route_overlay_replaces_only_the_image_axis(route, bare, declared):
    """A route narrows one field: reasoning and thinking are facts about the id.

    ``deepseek-flash`` thinks natively (``thinking:{type:...}``) on the gateway AND
    at the vendor — only the image axis differs — so an overlay that swapped the
    whole vector would silently change the request budget on one route.
    """
    scoped = model_capabilities(bare, _route_base(route))
    assert scoped is not None
    assert scoped.vision is declared
    route_agnostic = MODEL_CAPABILITIES[bare]
    assert (scoped.reasoning, scoped.thinking) == (route_agnostic.reasoning, route_agnostic.thinking)


def test_the_route_overlay_is_live_not_decorative(monkeypatch):
    """The lookup consults the route table, and the route table decides.

    Today's data happens to agree across routes for every id (V4.1 Flash is
    multimodal both at the vendor and behind the gateway), so an agreeing table is
    exactly the state in which a route lookup could quietly stop being consulted —
    and the next divergence (the vendor re-routes ``deepseek-v4-pro`` to V4.1 Flash
    on 2026-09-14) would then be declared but ignored.  So this pins the mechanism,
    not just the values: with a divergent entry in place, the route answer follows
    the table while the bare vector stays put.
    """
    from external_llm import model_registry

    native = "https://api.deepseek.com/v1"
    assert vision_capable("deepseek-flash", native) is True  # vendor release notes
    assert vision_capable("deepseek-flash", "https://opencode.ai/zen/go/v1") is True  # measured

    divergent = dict(ROUTE_VISION)
    divergent[(Route.DEEPSEEK, "deepseek-flash")] = False
    monkeypatch.setattr(model_registry, "ROUTE_VISION", divergent)
    assert vision_capable("deepseek-flash", native) is False
    assert text_only_model("deepseek-flash", native) is True
    # The bare vector is untouched: a route entry narrows one route, it does not
    # rewrite what the id is.
    assert vision_capable("deepseek-flash") is True
    assert MODEL_CAPABILITIES["deepseek-flash"].vision is True


def test_unmeasured_route_keeps_the_route_agnostic_answer():
    """No key for that route (or none for the host) → the bare vector answers.

    A proxy, a new gateway, a picker with no wire identity: none of them may
    inherit another route's verdict, and the permissive default is what keeps the
    strip-and-retry net able to learn the route for real.
    """
    for base in ("", "https://my.proxy.internal/v1", None):
        assert vision_capable("deepseek-flash", base) is MODEL_CAPABILITIES["deepseek-flash"].vision
        assert vision_capable("deepseek-flash", base) is True
    # A route with no entry for THIS id (native serves no vision-exp) is the same
    # case: the id's own vector answers, not the route's other declarations.
    assert vision_capable("deepseek-v4-flash-vision-exp", "https://api.deepseek.com/v1") is True


@pytest.mark.parametrize(
    "spelling",
    [
        "deepseek-flash",
        "opencode/deepseek-flash",
        "deepseek:deepseek-flash",
        "openrouter/deepseek/deepseek-v4.1-flash",
        "deepseek-v4.1-flash",
    ],
)
def test_every_spelling_resolves_to_the_same_route_fact(spelling):
    """Route prefixes, colon forms and aliases all reduce before the lookup.

    The versioned gateway spelling is the one that bites: OpenRouter publishes
    ``deepseek/deepseek-v4.1-flash`` with image input, and the alias table already
    maps it onto ``deepseek-flash`` — a route fact keyed on the raw spelling would
    miss the model it describes.
    """
    assert vision_capable(spelling, "https://opencode.ai/zen/go/v1") is True


def test_content_builder_follows_the_route(monkeypatch):
    """End of the chain: the route-scoped answer is what the wire sees.

    ``_openai_content`` is the one builder the OpenAI-protocol routes share, so the
    image part appears or the OCR fold happens exactly where the route says — and
    ``_images_to_text`` is booby-trapped to prove which branch ran.
    """
    import external_llm.providers as providers_module
    from external_llm.client import LLMMessage
    from external_llm.openai_client import _openai_content

    images = [{"data": "AAAA", "media_type": "image/png", "ocr_text": "OCR-BODY"}]
    msg = LLMMessage(role="user", content="u", images=images)

    real_images_to_text = providers_module._images_to_text
    monkeypatch.setattr(providers_module, "_images_to_text", lambda _imgs: pytest.fail("unexpected OCR"))
    for base in ("https://opencode.ai/zen/go/v1", "https://api.deepseek.com/v1"):
        parts = _openai_content(msg, "deepseek-flash", base)
        assert isinstance(parts, list), base
        assert parts[0]["type"] == "image_url"

    # A text-only id on the very same routes: the declared answer has to reach the
    # wire as the OCR fold instead of a part.
    monkeypatch.setattr(providers_module, "_images_to_text", real_images_to_text)
    for base in ("https://opencode.ai/zen/go/v1", "https://api.deepseek.com/v1"):
        content = _openai_content(msg, "deepseek-v4-pro", base)
        assert isinstance(content, str), base
        assert content == "[Image 1 — OCR Extracted Text:\nOCR-BODY\n]\nu"


def test_runtime_learned_rejection_is_alias_aware(monkeypatch):
    """The learned set and the declared table must name the same model.

    ``_IMAGE_REJECTING_MODELS`` is keyed by base URL + bare id, so the id has to
    reach canonical form before it is written or read: a 400 observed for the
    gateway's ``deepseek-v4.1-flash`` is a fact about ``deepseek-flash``, which is
    what the alias table claims and what the static lookup already assumes.
    """
    from external_llm import openai_client as oc
    from external_llm.client import LLMMessage
    from external_llm.openai_client import _bare_model_name, _openai_content

    assert _bare_model_name("openrouter/deepseek/deepseek-v4.1-flash") == "deepseek-flash"

    learned = {(oc._norm_base("https://opencode.ai/zen/go/v1"), "deepseek-flash")}
    monkeypatch.setattr(oc, "_IMAGE_REJECTING_MODELS", learned)
    content = _openai_content(
        LLMMessage(role="user", content="u", images=[{"data": "AAAA", "media_type": "image/png", "ocr_text": "T"}]),
        "deepseek-v4.1-flash",
        "https://opencode.ai/zen/go/v1",
    )
    assert isinstance(content, str)


# ── Live probes: re-measure the route instead of trusting the comment ─────────
#
# Raw POSTs on purpose: the probe must put an image part on the wire whatever the
# declaration says, because that is the only way to falsify it.  Going through the
# client would let the declaration choose the body (OCR text for a ``False`` route)
# and the probe would then measure nothing at all.  The 2-colour image is order-
# discriminating for the same reason — a 1x1 or single-colour probe is read the
# same way by a model that ignores the attachment, which is how a vision probe
# passes vacuously right up until a user notices the answer ignored their image.

_PROBE_PROMPT = (
    "The attached image has two solid vertical halves. Reply with exactly two words: "
    "the LEFT half's colour followed by the RIGHT half's colour. If no image arrived, "
    "reply exactly NOIMAGE."
)

# ``deepseek-v4-flash`` is deliberately absent: that route is inconsistent by
# measurement (400 five runs in a row, then a correct read — see the note on
# ROUTE_VISION), so both a True and a False assertion for it would be flaky.  Its
# declaration stays False for the deterministic path OCR gives, and the honest
# record of that inconsistency is the comment, not a coin-flip test.
_LIVE_ROUTE_VISION_PROBES = [
    # (route, model, declared vision, credential env var)
    (Route.OPENCODE, "deepseek-flash", True, "OPENCODE_API_KEY"),
    (Route.OPENCODE, "deepseek-v4-flash-vision-exp", True, "OPENCODE_API_KEY"),
    (Route.OPENCODE, "deepseek-v4-pro", False, "OPENCODE_API_KEY"),
    # Native probes need a funded key: the vendor answers 402 before it resolves
    # the model, so they skip honestly until the account can pay for the call.
    (Route.DEEPSEEK, "deepseek-flash", True, "DEEPSEEK_API_KEY"),
    (Route.DEEPSEEK, "deepseek-v4-pro", False, "DEEPSEEK_API_KEY"),
]


def _two_color_png_base64() -> str:
    """64x32 PNG, left half red, right half blue; standard library only."""
    width, height = 64, 32
    rows = b"".join(
        b"\x00" + b"".join(b"\xff\x00\x00" if x < width // 2 else b"\x00\x00\xff" for x in range(width))
        for _ in range(height)
    )

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(rows, 9))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _probe_image_route(client, model: str) -> tuple[int, str]:
    """POST one image part to the client's route; return ``(status, answer)``."""
    base = str(client.base_url or client.DEFAULT_BASE_URL).rstrip("/")
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
        # Same gateway requirement the client applies by host detection; the probe
        # owns its headers because it talks to the wire directly.
        "x-opencode-session": uuid.uuid4().hex,
        "User-Agent": "asicode-route-probe/1.0",
    }
    payload = {
        "model": model,
        # Reasoning model: a small budget is spent entirely on the trace and the
        # answer comes back empty (32 tokens → content '', reasoning_tokens 32),
        # which would read as "did not see the image" on a route that did.
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_two_color_png_base64()}"}},
                    {"type": "text", "text": _PROBE_PROMPT},
                ],
            }
        ],
    }
    response = client._session.post(
        f"{base}/chat/completions",
        headers=headers,
        json=payload,
        timeout=getattr(client, "timeout", 60),
    )
    try:
        answer = ""
        if response.status_code < 400:
            data = response.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            # The colours can land in the reasoning trace instead of the answer
            # (the model decides them there), and a dropped image puts neither —
            # so the two fields together still discriminate.
            answer = f"{message.get('content') or ''}\n{message.get('reasoning_content') or ''}"
        return response.status_code, answer
    finally:
        response.close()


def _reads_both_colors(answer: str) -> bool:
    """True when the answer names red before blue — the only proof of a read."""
    lowered = (answer or "").lower()
    left, right = lowered.find("red"), lowered.find("blue")
    return left != -1 and right != -1 and left < right


@pytest.mark.parametrize("route,model,declared,key_env", _LIVE_ROUTE_VISION_PROBES)
def test_live_route_vision_matches_the_declaration(route, model, declared, key_env):
    """The declared value, re-measured on the route it claims to describe.

    A ``True`` has to read the two-colour probe correctly; a ``False`` has to fail
    to read it (rejected outright, or answered as if no image arrived) — both are
    reasons to pre-convert the attachment to OCR text.  A credential that cannot
    reach the model (401/402/429) is skipped rather than counted as evidence in
    either direction, since an unauthenticated route reads no image either way.
    """
    key = (os.getenv(key_env) or "").strip()
    if not key:
        pytest.skip(f"{key_env} not set — the route probe needs a credential")

    provider = next(p for p, r in EXPECTED_PROVIDER_ROUTES.items() if r is route)
    client = create_llm_client(provider, api_key=key)
    status, answer = _probe_image_route(client, model)

    if status in (401, 402, 429):
        pytest.skip(f"{route.value}: credential cannot exercise {model} (HTTP {status})")

    reads = _reads_both_colors(answer)
    if declared:
        assert status < 400, f"{route.value}/{model}: declared vision-capable but answered HTTP {status}"
        assert reads, f"{route.value}/{model}: declared vision-capable but did not read the probe image: {answer!r}"
    else:
        assert not reads, f"{route.value}/{model}: declared text-only but read the probe image: {answer!r}"
