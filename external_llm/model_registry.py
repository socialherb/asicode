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
  - Size-based fallback for num_ctx lives in providers.py (_num_ctx_for_model)
    and handles standard tag names like "qwen3:8b". Only non-parseable tags
    (like ":e2b") need entries in OLLAMA_NUM_CTX_OVERRIDES.
"""

from __future__ import annotations

from typing import Optional

# ── Ollama: Vision-capable models ─────────────────────────────────────────────
# Keyword substring match — vision capability is identified by name keywords,
# not exact tags, since vision models have many version/quantization variants.
# Fast-path detection; runtime API detection is used as a slow-path fallback.
OLLAMA_VISION_KEYWORDS: tuple[str, ...] = (
    "llava", "bakllava", "moondream", "minicpm-v", "vision", "-vl", "_vl",
)

# Runtime model capability cache — populated lazily by _check_model_capability_cached.
_MODEL_CAPABILITY_CACHE: dict[str, dict[str, bool]] = {}


def _check_model_capability_cached(model_name: str, capability: str) -> bool:
    """Query model capability with in-memory caching.

    Returns False if the capability is unknown or the Ollama API is unavailable.
    The cache is populated by :func:`populate_model_capability_cache` which can
    be called at startup or on-demand.
    """
    cache_key = model_name.lower()
    if cache_key in _MODEL_CAPABILITY_CACHE:
        return _MODEL_CAPABILITY_CACHE[cache_key].get(capability, False)
    return False


# ── Ollama: Explicit num_ctx overrides ────────────────────────────────────────
# For models whose tag doesn't encode their size (e.g., ":e2b", ":e4b").
# Standard tags like "qwen3:8b" are handled by the size-based fallback in
# providers.py — no entry needed here.
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
# See anthropic_client.py (ANTHROPIC_DEFAULT=65536) for the Anthropic fallback.
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
    ("o1-",       "openai"),
    ("o3-",       "openai"),
    ("o4-",       "openai"),
    ("gemini",    "google"),
    ("deepseek",  "deepseek"),
    ("glm-",      "zai"),
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


def ollama_vision(model: str) -> bool:
    """Return True if this Ollama model supports image/vision input.

    Detection strategy (two-tier):
    1. Fast path — keyword substring match against OLLAMA_VISION_KEYWORDS.
    2. Slow path — check runtime capability cache (populated externally via
       :func:`populate_model_capability_cache` or ``ollama show``).
    """
    m = _norm(model)
    # Fast path: keyword detection
    if any(kw in m for kw in OLLAMA_VISION_KEYWORDS):
        return True
    # Slow path: runtime capability cache
    return _check_model_capability_cached(m, "vision")


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
