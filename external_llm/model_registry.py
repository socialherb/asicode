"""
asicode Model Registry — Model Classification & Configuration

Sections:
  - OLLAMA_VISION_KEYWORDS — keyword substring match for vision capability.
  - OLLAMA_NUM_CTX_OVERRIDES — explicit num_ctx for non-size-parseable tags (":e2b").
  - MODEL_ANSWER_MAX_TOKENS — prefix-based answer generation limits for cloud models.
  - CLOUD_PROVIDER_PREFIXES — prefix-based provider detection for cloud APIs.

Matching rules:
  - No model-name-based tool-calling classification (runtime fallback handles it).
  - OLLAMA_VISION_KEYWORDS uses substring matching (keyword in name), not exact tags.
  - CLOUD_PROVIDER_PREFIXES and MODEL_ANSWER_MAX_TOKENS are checked in order;
    first match wins — put more specific entries before general ones.
  - Any table doing an EXACT lookup must reduce the id with bare_model_name()
    first; a model arrives spelled however its route spells it. Skipping that
    is what gave context_budget._CONTEXT_LIMITS two different windows for one
    Claude model depending on whether it was reached natively or via
    OpenRouter (see tests/unit/test_model_catalog_context_parity.py).
  - num_ctx resolution lives in providers.py (_num_ctx_for_model): priority 0
    reads the model's Modelfile via Ollama /api/show, priority 1 is
    OLLAMA_NUM_CTX_OVERRIDES, priority 2 is a flat 8192 floor (asicode's system
    prefix is ~5272 tokens, above Ollama's 4096 default). There is NO tag-name
    size parsing — the old "handles qwen3:8b" claim was never implemented.
"""

from __future__ import annotations

from typing import Optional

from external_llm.model_catalog import MODEL_ALIASES
from external_llm.ollama_api import query_ollama_capabilities

# ── Ollama: Vision-capable models ─────────────────────────────────────────────
# Keyword substring match — vision capability is identified by name keywords,
# not exact tags, since vision models have many version/quantization variants.
# Fast-path detection; runtime API detection (/api/show capabilities, cached)
# is used as a slow-path fallback for names the keywords miss (qwen2.5vl:7b
# carries no "-vl"/"_vl" separator — plain "vl" covers it, as do pixtral /
# internvl, which are vision-only model families).
OLLAMA_VISION_KEYWORDS: tuple[str, ...] = (
    "llava", "bakllava", "moondream", "minicpm-v", "vision", "-vl", "_vl",
    "vl", "pixtral", "internvl",
)


def _check_model_capability_cached(model_name: str, capability: str,
                                   base_url_hint: Optional[str] = None) -> bool:
    """Query model capability with in-memory caching.

    Returns False if the capability is unknown or the Ollama API is
    unavailable.  Delegates to :func:`external_llm.ollama_api.query_ollama_capabilities`,
    which owns the TTL/negative caching (5-min positive, 15-s negative) —
    the slow path is therefore safe to call on every generation.
    """
    caps = query_ollama_capabilities(model_name, base_url_hint)
    if caps is None:
        return False
    return capability in caps


# ── Ollama: Explicit num_ctx overrides ────────────────────────────────────────
# For models whose Modelfile lacks num_ctx. Most models set num_ctx via Modelfile
# (priority 0 in _num_ctx_for_model reads it from /api/show); the flat 8192 floor
# (priority 2) covers the rest. This override dict is the middle escape hatch for
# tags that need a value the Modelfile does not provide — NOT size parsing.
#
# Entries are intentionally empty.  Users who need custom num_ctx should set
# it via Modelfile: ``ollama run /set num_ctx X /save``.  asicode reads it
# from Ollama's /api/show at runtime (priority 0 in _num_ctx_for_model).
# Hardcoded overrides are removed because they presume a value without knowing
# the user's hardware or whether the user wants non-default behaviour.
#
# Format: exact lowercased model tag → num_ctx value
OLLAMA_NUM_CTX_OVERRIDES: dict[str, int] = {}

# ── Cloud models: answer generation max_tokens ────────────────────────────────
# NOTE: max_tokens was previously managed by MODEL_ANSWER_MAX_TOKENS (removed).
# Anthropic max_tokens fallback SSOT: agent/config/thresholds.py (ANTHROPIC_DEFAULT).
# For non-Anthropic providers, max_tokens=None → API uses model's own default.

# ── Cloud provider detection from model name ──────────────────────────────────
# Maps model name prefix → provider string used by create_intelligent_service.
# Checked in order — first match wins.
#
# OpenRouter note: OpenRouter slugs use the ``<vendor>/<model>`` form
# (e.g. ``deepseek/deepseek-v4-flash``). To route them through the OpenRouter
# gateway rather than the vendor's native API, prefix the slug with
# ``openrouter/`` (e.g. ``openrouter/deepseek/deepseek-v4-flash``). The
# ``openrouter/`` entry MUST be first — otherwise a bare
# ``deepseek/...`` slug would match the ``deepseek`` prefix below and be
# misrouted to the native DeepSeek client.
CLOUD_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("openrouter/", "openrouter"),
    ("claude",    "anthropic"),
    ("gpt-",      "openai"),
    # No trailing hyphen: OpenAI ships the bare reasoning-model ids alongside
    # their sized variants, and ``KNOWN_MODELS["openai"]`` offers ``o3``. With
    # ``"o3-"`` here that id matched nothing, and the one live caller
    # (webapp/routes/agent_stream.py) reads a None as "not a cloud model" and
    # falls back to ``"deepseek"`` — so asking for o3 dispatched to DeepSeek.
    # ``o3-mini``/``o4-mini`` still match, being prefixed by the bare id.
    ("o1",        "openai"),
    ("o3",        "openai"),
    ("o4",        "openai"),
    ("gemini",    "google"),
    ("deepseek",  "deepseek"),
    ("glm-",      "zai"),
)

# ── Cloud models: text-only (reject image input) ──────────────────────────────
# Bare-model-name prefix match (route prefixes like "openrouter/deepseek/…"
# stripped first). DeepSeek models (chat/reasoner/v4-flash/v4-pro) accept only
# string content — an image_url part draws HTTP 400 regardless of image size.
# Verified 2026-07-26 on three independent sources: (1) opencode Go rejected a
# 67-byte 1x1 PNG for BOTH v4-flash and v4-pro while the same route carried
# images fine for kimi-k3 (so it is the model, not the gateway); (2) OpenRouter
# metadata lists both v4 models as modality=text->text, input=['text'];
# (3) DeepSeek's official API docs describe no image/vision input. Only add
# entries verified text-only on EVERY route — per-route rejections belong to
# the runtime strip-and-retry net in openai_client, which keys on
# (base_url, model).
# DeepSeek models verified text-only on EVERY route.  Exclude prefixes that
# would falsely match vision-capable models (deepseek-vl, deepseek-vl2,
# deepseek-ocr).  Unlisted models are caught at runtime by the strip-and-retry
# net in openai_client.
TEXT_ONLY_MODEL_PREFIXES: tuple[str, ...] = (
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-v4",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Query API — import these functions instead of duplicating detection logic
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(model: str) -> str:
    """Lowercase and strip provider prefix (e.g. 'ollama:gemma4:e2b' → 'gemma4:e2b').

    Handles colon provider prefixes (``ollama:``, ``openai:``, ``openrouter:``,
    etc). For OpenRouter, BOTH the colon (``openrouter:<slug>``) and slash
    (``openrouter/<slug>``) forms are normalised to the slash form so that
    ``detect_cloud_provider`` can match the single ``openrouter/`` prefix. This
    is necessary because OpenRouter slugs themselves contain a slash
    (``deepseek/deepseek-v4-flash``) and must be preserved verbatim after the
    routing prefix is stripped.
    """
    name = (model or "").lower().strip()
    if ":" in name:
        first = name.split(":")[0]
        if first in ("ollama", "openai", "anthropic", "deepseek", "google", "zai"):
            name = name.split(":", 1)[1]
        elif first == "openrouter":
            # Normalise colon → slash so detect_cloud_provider matches a single
            # ``openrouter/`` prefix for both input forms.
            name = "openrouter/" + name.split(":", 1)[1]
    return name


def ollama_vision(model: str, base_url_hint: Optional[str] = None) -> bool:
    """Return True if this Ollama model supports image/vision input.

    Detection strategy (two-tier):
    1. Fast path — keyword substring match against OLLAMA_VISION_KEYWORDS.
    2. Slow path — runtime capability query via Ollama ``/api/show``
       (cached 5 min, negative-cached 15 s; ``base_url_hint`` matches the
       num_ctx query so a non-default Ollama server is consulted).
    """
    m = _norm(model)
    # Fast path: keyword detection
    if any(kw in m for kw in OLLAMA_VISION_KEYWORDS):
        return True
    # Slow path: runtime capability cache
    return _check_model_capability_cached(m, "vision", base_url_hint)


def ollama_supports_tools(model: str, base_url_hint: Optional[str] = None) -> Optional[bool]:
    """Return whether this Ollama model supports native tool calling.

    True/False when the server reported the ``tools`` capability (via the
    same cached ``/api/show`` query as :func:`ollama_vision`); None when the
    capability is unknown (server unreachable, non-tag name, older Ollama) —
    callers should treat None as "assume supported" (status quo).
    """
    m = _norm(model)
    caps = query_ollama_capabilities(m, base_url_hint)
    if caps is None:
        return None
    return "tools" in caps


def get_ollama_num_ctx(model: str) -> Optional[int]:
    """Return explicit num_ctx for a model, or None to use the size-based fallback.

    Only covers models whose tag doesn't encode a parseable size.
    Standard tags like 'qwen3:8b' should use the size regex fallback instead.
    """
    return OLLAMA_NUM_CTX_OVERRIDES.get(_norm(model))


# NOTE: get_answer_max_tokens() removed — see comment above for the new approach.


def detect_cloud_provider(model: str) -> Optional[str]:
    """Detect API provider from model name prefix. Returns None if unknown/Ollama."""
    m = _norm(model)
    for prefix, provider in CLOUD_PROVIDER_PREFIXES:
        if m.startswith(prefix):
            return provider
    return None


def bare_model_name(model: str) -> str:
    """Reduce any spelling of a model id to the bare ``model_catalog`` id.

    A model reaches a lookup table in whatever spelling its route uses:
    ``openrouter/anthropic/claude-sonnet-5``, ``anthropic:claude-opus-5``, a
    deprecated alias, or the bare id.  Every table in this repo that does an
    *exact* lookup has to normalise first, and until now only
    ``text_only_model`` did (with a local ``_norm(...).split("/")[-1]``).
    ``context_budget._CONTEXT_LIMITS`` did not, so the same Claude model
    resolved to 200K natively and to the 1M fallback through OpenRouter — one
    model, two context windows, depending on which route asked.  Hoisted here
    so both tables share the one normaliser rather than growing a second copy.

    Ollama tags (``gemma4:e2b``) carry no ``/`` and are returned unchanged.
    """
    bare = _norm(model).split("/")[-1]
    # Alias last: MODEL_ALIASES is keyed on bare ids, so it can only match once
    # the routing/vendor prefixes are gone.
    return MODEL_ALIASES.get(bare, bare)


def text_only_model(model: str) -> bool:
    """True if this cloud model is known to reject image (vision) input."""
    return bare_model_name(model).startswith(TEXT_ONLY_MODEL_PREFIXES)
