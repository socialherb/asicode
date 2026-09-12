"""Single source of truth for the per-provider model catalog.

Three surfaces used to keep hand-synced copies of this data and drifted every
time one was updated (2026-07: one refresh landed 8 entries in asi.py, 1 in
the kp verify tool, 6 in the webapp picker):

- ``asi.py`` — /model command display, provider inference, alias resolution
- ``tools/kp_correctness_verify.py`` — pre-run model-ID validation
- ``webapp/ui/static/ui.js`` — model picker (now served by
  ``/ui/api/external-models`` in ``webapp/ui/ui_tools.py``)

Update THIS module when a provider ships/renames models; every surface
follows.

Two tiers, because display and validation want different sets:

- ``KNOWN_MODELS`` — the current, recommended IDs. What the CLI displays and
  the webapp offers. Keep it curated; stale entries here clutter every picker.
- ``LEGACY_MODELS`` — older IDs that still resolve at the provider API.
  Validation (kp verify) accepts these; display surfaces do not show them.
  Move an ID here (rather than deleting) when a provider deprecates it.

- ``MODEL_CAPABILITIES`` — per-id dispatch facts (does the model spend part of
  ``max_tokens`` on a reasoning trace; which thinking control does it accept).
  Read by the OpenAI-compatible client family; see the table for why this is
  declared per id rather than inferred from name prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ── Current, recommended model IDs per provider ────────────────────────────
KNOWN_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-fable-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5-20251001",
    ],
    "deepseek": [
        # Native tier = what the vendor registry serves, and nothing else.
        # Verified live 2026-09-11: ``GET https://api.deepseek.com/v1/models``
        # returns exactly these two ids, a per-id GET answers 200 for each, and
        # every other deepseek spelling answers 404 "Model Not Found" — the same
        # answer a bogus control id gets. The two spellings the opencode tier
        # serves (``deepseek-v4-flash``, ``deepseek-v4-flash-vision-exp``) are
        # therefore NOT native ids; they left this tier on 2026-09-11, when
        # ``/model deepseek/<id>`` still switched to this endpoint and 404'd.
        #
        # Recorded conflict, not a silent reversal: models.dev's ``deepseek``
        # provider still lists all four spellings (family ``deepseek-flash``,
        # released 2026-09-10). ``POST /chat/completions`` cannot arbitrate — a
        # zero-balance key gets 402 "Insufficient Balance" BEFORE the model is
        # validated, for a valid id and a bogus one alike. The vendor registry is
        # the only authority reachable and the source this catalog has always
        # cited, so it wins; the live gate in
        # tests/unit/test_model_catalog_context_parity.py re-reads it every run.
        #
        # DeepSeek V4.1 Flash (GA 2026-09-10) — 1M context, 384K max output;
        # off-peak 0.15/0.60, cache 0.003 (DeepSeek pricing notice, effective
        # 2026-09-10 04:00 UTC). One model, three spellings: this vendor id, the
        # gateway's ``deepseek-v4.1-flash`` (MODEL_ALIASES), and
        # ``DeepSeekClient.DEFAULT_MODEL``.
        "deepseek-flash",
        # DeepSeek V4-Pro — reasoning tier, 1M window, same off-peak sheet.
        "deepseek-v4-pro",
    ],
    "openai": [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    ],
    "google": [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro",
        "gemini-3-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
    "zai": [
        # GLM-5.3 (Z.ai, released 2026-08-14, 1M ctx) — native zai provider.
        # Matches the opencode tier and context_budget._CONTEXT_LIMITS.
        "glm-5.3",
        "glm-5.2",
        "glm-5.1",
        "glm-5-turbo",
        "glm-5",
        "glm-4.7",
        # GLM-5.3-Flash (released 2026-08-25; on the zai API since 2026-09-01)
        # — 1M window like glm-5.3 → _FALLBACK_IS_CORRECT.
        "glm-5.3-flash",
    ],
    "openrouter": [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "anthropic/claude-fable-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.6-flash",
        "google/gemini-2.5-pro",
        "moonshotai/kimi-k3",
        "minimax/minimax-m3",
        # OpenRouter uses the ``z-ai`` vendor prefix for Zhipu GLM models.
        "z-ai/glm-5.2",
        "qwen/qwen3.7-max",
        "qwen/qwen3.6-plus",
    ],
    "opencode": [
        # Curated list from https://opencode.ai/zen/go/v1/models.
        # Re-verified 2026-09-11 against the live API (37 ids returned). Two served
        # ids are omitted here because MODEL_ALIASES already carries them:
        # hy3-preview (→ hy3) and deepseek-v4.1-flash (→ deepseek-flash, the
        # vendor id — see the alias table for the evidence that it is one model).
        "glm-5.3-flash",
        "glm-5.3",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "hy4-preview",
        "hy3",
        "gpt-5.6-luna",
        "grok-4.6",
        "grok-4.5",
        # Meta Muse Spark 1.2/1.3 (released 2026-08-05/2026-09-01; on the
        # gateway since 2026-08-19/2026-09). "contributor" is the ~12x cheaper
        # variant whose data is used for training — a distinct routable id, not
        # an alias. The base muse-spark-1.2 was dropped by the gateway on
        # 2026-08-20; only the contributor variants are served now. 1M (2^20)
        # window for both; decision recorded in _FALLBACK_IS_CORRECT.
        "muse-spark-1.3-contributor",
        "muse-spark-1.2-contributor",
        # DeepSeek v4 Flash vision variant (served since 2026-08-21 alongside
        # deepseek-v4-flash; vision suffix = multimodal front-end over the
        # same v4-flash core). 1M window like deepseek-v4-flash →
        # _FALLBACK_IS_CORRECT (UNVERIFIED).
        "deepseek-v4-flash-vision-exp",
        # LongCat-2.0 (Meituan, open-sourced 2026-07-05; served by the opencode
        # gateway since 2026-08-24). 1.6T-param MoE, native 1M context (trained
        # on 1M-context data — longcat.chat/blog/longcat-2.0, github.com/
        # meituan-longcat/LongCat-2.0). Context decision in _FALLBACK_IS_CORRECT
        # (UNVERIFIED against an opencode-docs provider page; 1M fallback is the
        # correct window here).
        "longcat-2.0",
        # DeepSeek V4.1 Flash (2026-09-10 GA; on the opencode Go gateway since
        # 2026-09-09/10). /zen/go/v1/models serves it as "deepseek-flash"
        # (distinct from "deepseek-v4-flash"); the DeepSeek native API routes
        # the same id since 2026-09-10 too (verified live). 552B-param MoE,
        # native multimodal vision, 1M context, 384K max output. Go pricing:
        # $0.15/$0.60 off-peak (cached read $0.003); launch promo = 4x request
        # allowance, which the console renders with a "bonus" badge. → _OPENCODE
        # _COST_PER_M/_OPENCODE_CACHE_RATE + _FALLBACK_IS_CORRECT (1M window).
        "deepseek-flash",
    ],
}

# ── Older IDs that still resolve at the provider API ───────────────────────
# Accepted by validation (kp verify), hidden from display surfaces.
LEGACY_MODELS: dict[str, list[str]] = {
    "opencode": [
        # Omen Alpha (stealth coding model, 500K context, 128K max output,
        # reasoning + image input) — live on the Go gateway since ~2026-09-04,
        # REMOVED from the Go plan 2026-09-10 (opencode commit 859106e "remove
        # Omen Alpha listings"); the API still serves the id, so it keeps
        # resolving for already-pinned sessions, but it is no longer offered.
        "omen-alpha",
    ],
    "anthropic": [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    # deepseek — no legacy tier since 2026-09-11. The two ids that used to sit
    # here (``deepseek-chat`` / ``deepseek-reasoner``) are no longer routed by
    # api.deepseek.com: the registry lists exactly the two KNOWN ids above, and a
    # per-id probe answers 404 "Model Not Found" for both — the answer a bogus
    # control id gets. Keeping them would make validation accept an id the vendor
    # rejects, and they were the webapp picker's entire DeepSeek offering (see
    # WEBAPP_MODEL_OVERRIDES below): 0 of 2 options alive. A session still pinned
    # to one of them now fails resolution/validation loudly instead of 404ing at
    # the API.
    "google": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
    ],
    # Z.ai still serves the GLM-4.x line (verified live 2026-09-07); 128K
    # windows like glm-4.7. Kept out of the display tier (superseded by GLM-5).
    "zai": [
        "glm-4.6",
        "glm-4.5",
        "glm-4.5-air",
    ],
}

# ── Old/typo model names users might type → correct model names ────────────
MODEL_ALIASES: dict[str, str] = {
    # Anthropic: models that switched to dateless format
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    "claude-opus-4-20250514": "claude-opus-4-8",
    "claude-haiku-4-20250514": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250514": "claude-sonnet-4-6",
    "claude-opus-4-5-20250514": "claude-opus-4-8",
    # OpenCode Go: old model IDs → current model IDs
    "deepseek-v4": "deepseek-v4-pro",
    # DeepSeek V4.1 Flash is served under two spellings by the Go gateway: the
    # vendor id ``deepseek-flash`` — which is what the vendor's own API routes
    # (/models returns exactly deepseek-flash + deepseek-v4-pro) — and the
    # versioned gateway spelling ``deepseek-v4.1-flash``. models.dev reports ONE
    # model for both (family ``deepseek-flash``, 1M/384K, $0.15/$0.60, cached
    # read 0.003, released 2026-09-10), and a live probe on 2026-09-11 read a
    # red/black/blue test image correctly through both ids; the second call was
    # even billed as a prefix cache hit on the first call's bytes, i.e. one
    # upstream model behind two names. Aliasing keeps one catalog entry with one
    # set of context/pricing/capability decisions — as a second id the versioned
    # spelling would star the same model twice in the picker and need its own
    # window, rate and vector.
    "deepseek-v4.1-flash": "deepseek-flash",
    "kimi-k2": "kimi-k2.6",
    "mimo-m1": "mimo-v2.5",
    "qwq-32b": "qwen3.7-plus",
    # Tencent Hy3: hy3-preview → hy3 (GA transition); keep alias for back-compat.
    "hy3-preview": "hy3",
}


# ── Declared capability vector per model id ────────────────────────────────
# What the OpenAI-compatible client family puts on the wire for an id, and how
# it budgets tokens for it — NOT a claim about how much a model thinks.  Keyed
# on the BARE id (``model_registry.bare_model_name``), so every spelling —
# ``openrouter/deepseek/deepseek-v4-flash``, ``deepseek:deepseek-v4-flash``, the
# catalog id, a deprecated alias — resolves to one entry.
#
# Why declared instead of prefix-inferred: a prefix rule orphans any new id
# whose spelling breaks the family pattern.  ``deepseek-flash`` (DeepSeek V4.1
# Flash, GA 2026-09-10) carries no ``v4``, so the v4-flash successor classified
# as a NON-reasoning model: the thinking_mode toggle became a silent no-op
# (2026-09-10 live probe — the opencode gateway accepts
# ``thinking:{type:disabled}`` and drops reasoning tokens to zero) and
# ``/insights compact`` lost its 32k reasoning headroom, while its sibling
# ``deepseek-v4-flash`` got both.  Prefix rules therefore survive only as a
# FALLBACK for ids outside this catalog (variants such as ``kimi-k3-0711``, an
# id typed by hand).
#
# The vector also carries the IMAGE axis (``vision``): whether the client may put
# image_url parts on the wire or must pre-convert the attachment to OCR text.
# ``model_registry.text_only_model`` reads it through the same declaration-first
# rule, which closes the fourth orphan — the family prefix ``deepseek-v4`` matched
# ``deepseek-v4-flash-vision-exp``, the one id whose NAME says it reads images
# (2026-09-10 live probe: it reads a red/blue test image correctly, while the same
# part 400s for ``deepseek-v4-flash``).  A prefix rule cannot express "this member
# of the family is the exception", and the exception is exactly where the new
# capability lives.
#
# Coverage is a test contract, not a convention:
# tests/unit/test_model_catalog_capabilities_parity.py requires an entry for
# every id reachable through the OpenAI-compatible client family (openai /
# opencode / openrouter) — the native DeepSeek ids are covered through their
# opencode-tier spellings — so a new catalog id must make an explicit decision
# here rather than silently inheriting a default.  An id served only by a client
# that owns its thinking controls (AnthropicClient, GoogleClient) is out of scope;
# an OpenRouter slug for such a model (``anthropic/claude-sonnet-5``) is IN
# scope, because that route runs through this client family.
#
# A route-specific fact is NOT part of this table: the vector is looked up by
# bare id, which is route-invariant by construction.  ``ROUTE_VISION`` below
# carries the per-route image axis, and
# tests/unit/test_model_catalog_route_vision.py keeps the two consistent (a
# ``vision=False`` here must be what every route serving the id declares).


class ThinkingControl(str, Enum):
    """Which thinking-mode control an id accepts (NONE = send none)."""

    NONE = "none"
    # reasoning_effort dial — OpenAI o-series / gpt-5 (no ``thinking`` param).
    EFFORT = "effort"
    # Native ``thinking:{type:enabled|disabled}`` — DeepSeek v4 family; the only
    # control that reaches zero reasoning tokens (reasoning_effort="low" merely
    # dials reasoning down: 882→742 tokens measured on OpenCode Go).
    THINKING = "thinking"
    # Kimi K3: thinking is always on and cannot be disabled, and it 400s on any
    # reasoning_effort other than "max" — so "max" is all that is ever sent.
    MAX_ONLY = "max-only"


@dataclass(frozen=True)
class ModelCapabilities:
    """Declared dispatch vector for one model id (see ``MODEL_CAPABILITIES``).

    Attributes:
        reasoning: The model spends part of ``max_tokens`` on a reasoning trace.
            Callers must then protect the content budget: no ``temperature``
            (reasoning models reject a non-default one), ``max_completion_tokens``
            off the opencode route so the trace gets its own budget, and a 32k
            floor for ``/insights compact``.
        thinking: Which thinking-mode control reaches this id.
        vision: The id reads image parts, so the client may put them on the wire.
            ``True`` (the default) means "send the image and let the runtime
            strip-and-retry net absorb a rejection"; ``False`` is reserved for
            ids unable to use an image on EVERY route that serves them — either
            the route rejects the part outright or (worse) accepts and silently
            drops it, which costs the model the image AND the OCR text it could
            have had.  A fact that holds on one route only is declared per route
            in ``ROUTE_VISION`` and leaves this field permissive, because the
            runtime net can only correct the ``True`` direction: a static
            ``False`` never puts an image on the wire, so no 400 can arrive to
            revise it.
    """

    reasoning: bool = False
    thinking: ThinkingControl = ThinkingControl.NONE
    vision: bool = True


_NO_CONTROL = ModelCapabilities()
_EFFORT_DIAL = ModelCapabilities(reasoning=True, thinking=ThinkingControl.EFFORT)
_NATIVE_THINKING = ModelCapabilities(reasoning=True, thinking=ThinkingControl.THINKING)
_MAX_ONLY = ModelCapabilities(reasoning=True, thinking=ThinkingControl.MAX_ONLY)
# Same control, but the id cannot use an image: the client pre-converts the
# attachment to OCR text instead of spending a request on it.
_NATIVE_THINKING_TEXT_ONLY = ModelCapabilities(reasoning=True, thinking=ThinkingControl.THINKING, vision=False)

MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # DeepSeek v4 family — native thinking param.  deepseek-flash is V4.1 Flash,
    # the successor of deepseek-v4-flash behind an id that drops the ``v4``.
    #
    # Image axis, measured per id 2026-09-10 on the opencode Go route (a correct
    # reading of a red/blue test image is the only proof — a 200 alone is not,
    # because the part can be tolerated and dropped).  The per-route values live in
    # ROUTE_VISION below; the ``vision`` field here carries only what EVERY route
    # serving the id agrees on, so ``deepseek-flash`` — which reads an image on
    # opencode/openrouter and is text input at the vendor — stays permissive:
    #   flash-vision-exp  reads it → vision=True on both routes that serve it
    #   deepseek-flash    reads it on opencode/openrouter AND at the vendor (V4.1
    #                     Flash is native multimodal, changelog 2026-09-10) →
    #                     vision=True everywhere it is served; the gateway's
    #                     versioned spelling deepseek-v4.1-flash reads it too
    #                     (2026-09-11) and aliases onto this id
    #   deepseek-v4-flash 400s     → "Model only supports text input; received
    #                     unsupported content type 'image_url'" (5 runs at
    #                     max_tokens 128-4096).  It DID read the image in 7 of 8
    #                     runs inside one earlier window, so the route is
    #                     inconsistent; the deterministic OCR path is kept (see
    #                     TEXT_ONLY_MODEL_PREFIXES for the open question).
    #   deepseek-v4-pro   drops it → HTTP 200, but the answer is literally
    #                     "NOIMAGE" with the part attached.  Worse than a 400:
    #                     the model gets neither the image nor the OCR text.
    "deepseek-v4-flash": _NATIVE_THINKING_TEXT_ONLY,
    "deepseek-v4-pro": _NATIVE_THINKING_TEXT_ONLY,
    "deepseek-v4-flash-vision-exp": _NATIVE_THINKING,
    "deepseek-flash": _NATIVE_THINKING,
    # OpenAI o-series / GPT-5 — reasoning_effort dial.
    "o3": _EFFORT_DIAL,
    "o3-mini": _EFFORT_DIAL,
    "o4-mini": _EFFORT_DIAL,
    "gpt-5.6-sol": _EFFORT_DIAL,
    "gpt-5.6-terra": _EFFORT_DIAL,
    "gpt-5.6-luna": _EFFORT_DIAL,
    # Kimi K3 — always-on thinking, reasoning_effort="max" only.
    "kimi-k3": _MAX_ONLY,
    # ── No reasoning-specific dispatch ───────────────────────────────────────
    # Either the id does not reason on these routes, or it thinks by default and
    # asicode deliberately sends no control (ZAIClient forwards a caller's effort
    # override alone).  These entries are explicit decisions, not claims that the
    # model cannot reason.
    # opencode route
    "glm-5.3-flash": _NO_CONTROL,
    "glm-5.3": _NO_CONTROL,
    "glm-5.2": _NO_CONTROL,
    "glm-5.1": _NO_CONTROL,
    "glm-5": _NO_CONTROL,
    "kimi-k2.7-code": _NO_CONTROL,
    "kimi-k2.6": _NO_CONTROL,
    "kimi-k2.5": _NO_CONTROL,
    "mimo-v2.5-pro": _NO_CONTROL,
    "mimo-v2.5": _NO_CONTROL,
    "mimo-v2-pro": _NO_CONTROL,
    "mimo-v2-omni": _NO_CONTROL,
    "minimax-m3": _NO_CONTROL,
    "minimax-m2.7": _NO_CONTROL,
    "minimax-m2.5": _NO_CONTROL,
    "qwen3.8-max": _NO_CONTROL,
    "qwen3.8-flash": _NO_CONTROL,
    "qwen3.7-max": _NO_CONTROL,
    "qwen3.7-plus": _NO_CONTROL,
    "qwen3.6-plus": _NO_CONTROL,
    "qwen3.5-plus": _NO_CONTROL,
    "hy4-preview": _NO_CONTROL,
    "hy3": _NO_CONTROL,
    "grok-4.6": _NO_CONTROL,
    "grok-4.5": _NO_CONTROL,
    "muse-spark-1.3-contributor": _NO_CONTROL,
    "muse-spark-1.2-contributor": _NO_CONTROL,
    "longcat-2.0": _NO_CONTROL,
    # opencode legacy tier (still served, no longer offered)
    "omen-alpha": _NO_CONTROL,
    # openai route
    "gpt-4o": _NO_CONTROL,
    "gpt-4o-mini": _NO_CONTROL,
    # OpenRouter slugs whose bare id no other covered route serves.
    "claude-fable-5": _NO_CONTROL,
    "claude-sonnet-5": _NO_CONTROL,
    "claude-sonnet-4.6": _NO_CONTROL,
    "gemini-3.6-flash": _NO_CONTROL,
    "gemini-2.5-pro": _NO_CONTROL,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — the endpoint a client actually posts to
# ═══════════════════════════════════════════════════════════════════════════════


class Route(str, Enum):
    """A wire endpoint family, resolved from a base URL by :func:`route_key`.

    Derived from the URL's HOST rather than from the provider name, because the
    provider is only a local label for one of the routes: ``EXTERNAL_LLM_BASE_URL``
    can point the ``openai`` provider at the opencode gateway, so a fact that
    belongs to the wire has to follow the URL to stay true.

    Two base URLs that share a host share a key (zai's two protocols live on
    ``api.z.ai``) — a fact that needs that split gets its own key when someone
    declares it, not a cleverer matcher now.
    """

    OPENCODE = "opencode"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ZAI = "zai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# Host suffix → route. The host alone decides (scheme, port and path are ignored),
# so ``http://opencode.ai:8080/zen/go/v1`` and the production URL are one route.
ROUTE_HOSTS: tuple[tuple[str, Route], ...] = (
    ("opencode.ai", Route.OPENCODE),
    ("openrouter.ai", Route.OPENROUTER),
    ("api.deepseek.com", Route.DEEPSEEK),
    ("api.openai.com", Route.OPENAI),
    ("api.z.ai", Route.ZAI),
    ("api.anthropic.com", Route.ANTHROPIC),
    ("generativelanguage.googleapis.com", Route.GOOGLE),
)


def route_key(base: str) -> Route | None:
    """Route behind *base*, or ``None`` when the host matches no known route.

    ``None`` means "no route facts recorded": the caller falls back to the
    route-agnostic vector.  A custom gateway (proxy, self-hosted endpoint) really
    has no route key here, and guessing one from a path or a model name is how one
    route's measurements end up applied to another route's endpoint.
    """
    host = (base or "").strip().lower()
    if "//" in host:
        host = host.split("//", 1)[1]
    host = host.split("/", 1)[0].rsplit("@", 1)[-1]
    if ":" in host and not host.startswith("["):
        host = host.split(":", 1)[0]
    for suffix, route in ROUTE_HOSTS:
        if host == suffix or host.endswith("." + suffix):
            return route
    return None


# ── The image axis, per route ──────────────────────────────────────────────────
# ``ModelCapabilities.vision`` answers "may an image part go out for this id"
# WITHOUT knowing the route, so it may only carry what every serving route agrees
# on — a test enforces exactly that, and a route-specific fact belongs here
# instead.  A missing key means "not measured on that route", which falls through
# to the vector's ``vision`` (permissive, and corrected at runtime by the client's
# strip-and-retry net when the route answers 400).
#
# One id really does answer differently per route, which is the reason this table
# exists: ``deepseek-flash`` reads an image through the opencode gateway, is
# advertised with image input on OpenRouter, and the vendor's own API documents
# none.  Evidence, never carried over from one route to another:
#   opencode    measured 2026-09-10/11 through the Go gateway with a red/blue
#               probe image — a bare 200 proves nothing, since a part can be
#               tolerated and dropped (see the per-id notes on MODEL_CAPABILITIES).
#   openrouter  publishing metadata, read live 2026-09-11 from
#               https://openrouter.ai/api/v1/models → ``architecture.input_modalities``
#               per slug: ``deepseek/deepseek-v4.1-flash`` and
#               ``deepseek/deepseek-v4-flash-vision-exp`` carry ['text','image'];
#               ``deepseek/deepseek-v4-flash`` and ``deepseek/deepseek-v4-pro``
#               carry ['text'].  Slugs normalise to bare ids like every other
#               spelling, so the versioned slug lands on the ``deepseek-flash`` key.
#   deepseek    the vendor's own API, whose registry serves exactly ``deepseek-flash``
#               and ``deepseek-v4-pro`` (live 2026-09-11, see
#               tests/unit/test_model_catalog_context_parity.py).  Modality comes from
#               the vendor's release notes and from models.dev's per-id
#               ``modalities.input`` for the ``deepseek`` provider (2026-09-11):
#               V4.1 Flash IS native multimodal ("V4.1-Flash is now live on the
#               DeepSeek API with native multimodal support", changelog 2026-09-10,
#               input ['text','image']), while V4-Pro-0813 carries ['text'].  The
#               older "no image input" reading still quoted on TEXT_ONLY_MODEL_PREFIXES
#               described the RETIRED v4-flash generation.  No chat probe can confirm
#               either value here: a zero-balance key answers 402 "Insufficient
#               Balance" BEFORE the model is resolved, for a valid id and a bogus one
#               alike, so these entries stay documentation-based and the 400-recovery
#               net remains the correction path if a document goes stale.
ROUTE_VISION: dict[tuple[Route, str], bool] = {
    # opencode Go gateway — per-id measurements
    (Route.OPENCODE, "deepseek-flash"): True,
    (Route.OPENCODE, "deepseek-v4-flash-vision-exp"): True,
    (Route.OPENCODE, "deepseek-v4-flash"): False,
    (Route.OPENCODE, "deepseek-v4-pro"): False,
    # openrouter — publishing metadata, per slug.  Only the slugs this catalog
    # offers are declared: OpenRouter also publishes the versioned
    # ``deepseek/deepseek-v4.1-flash`` (which its metadata lists with image input
    # and our alias table maps onto ``deepseek-flash``), but that slug is not in
    # KNOWN_MODELS["openrouter"], so no reachable (route, id) pair exists for it
    # and the alias keeps it on the route-agnostic vector.
    (Route.OPENROUTER, "deepseek-v4-flash"): False,
    (Route.OPENROUTER, "deepseek-v4-pro"): False,
    # api.deepseek.com — vendor release notes + models.dev input modalities
    (Route.DEEPSEEK, "deepseek-flash"): True,
    # Dated, and deliberately so: the vendor announced that from 2026-09-14 (12:00
    # Beijing) until V4.1-Pro ships, every ``deepseek-v4-pro`` request is routed to
    # V4.1 Flash — which is multimodal.  This entry then describes a name that no
    # longer reaches V4-Pro and has to be revisited; the route split is what makes
    # that a one-line edit here instead of a change to the shared vector (which
    # would have dragged the opencode/openrouter entries along with it).
    (Route.DEEPSEEK, "deepseek-v4-pro"): False,
}


def valid_models(provider: str) -> list[str]:
    """Every ID that should pass validation for *provider*: current + legacy."""
    return list(KNOWN_MODELS.get(provider, [])) + list(LEGACY_MODELS.get(provider, []))


# ── Webapp picker composition ──────────────────────────────────────────────
# (group label, [(value-prefix provider, display label), ...]). Providers not
# listed here are CLI-only (opencode/zai direct, openai) until the webapp
# grows a client for them.
WEBAPP_PROVIDER_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "External",
        [
            ("deepseek", "DeepSeek API"),
            ("anthropic", "Anthropic"),
            ("google", "Google Gemini"),
        ],
    ),
    (
        "OpenRouter",
        [
            ("openrouter", "OpenRouter"),
        ],
    ),
]

# Providers whose webapp offering differs from KNOWN_MODELS — a SUBSET, never a
# superset. The webapp's client set (deepseek/anthropic/google/openrouter) is
# narrower than the CLI's, so a provider whose webapp route cannot serve its
# whole tier needs a shortened list here. An id OUTSIDE the tier is unchecked by
# definition — no parity test, no live gate and no rate table knows it — which is
# how ``deepseek-chat``/``deepseek-reasoner`` stayed on the picker long after the
# vendor stopped routing them: the webapp's whole DeepSeek offering was dead
# while the CLI offered two live ids. Empty as of 2026-09-11, so the deepseek tier
# (the native registry itself) is offered verbatim.
# Invariant: tests/unit/test_model_catalog.py::TestWebappOverrideIsASubsetOfTheTier
WEBAPP_MODEL_OVERRIDES: dict[str, list[str]] = {}


def webapp_external_model_groups() -> list[dict]:
    """Option groups for the webapp model picker.

    Values use the ``external_<provider>:<model>`` encoding that
    webapp/routes strips generically; OpenRouter models keep their
    ``vendor/model`` slug intact inside the value.
    """
    groups: list[dict] = []
    for group_label, providers in WEBAPP_PROVIDER_GROUPS:
        options: list[dict] = []
        for provider, provider_label in providers:
            models = WEBAPP_MODEL_OVERRIDES.get(provider) or KNOWN_MODELS.get(provider, [])
            options.extend(
                {
                    "value": f"external_{provider}:{m}",
                    "label": f"{provider_label} · {m}",
                }
                for m in models
            )
        groups.append({"label": group_label, "options": options})
    return groups


__all__ = [
    "KNOWN_MODELS",
    "LEGACY_MODELS",
    "MODEL_ALIASES",
    "MODEL_CAPABILITIES",
    "ROUTE_HOSTS",
    "ROUTE_VISION",
    "WEBAPP_MODEL_OVERRIDES",
    "WEBAPP_PROVIDER_GROUPS",
    "ModelCapabilities",
    "Route",
    "ThinkingControl",
    "route_key",
    "valid_models",
    "webapp_external_model_groups",
]
