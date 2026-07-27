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
"""
from __future__ import annotations

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
        "deepseek-v4-flash",
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
        "glm-5.2",
        "glm-5.1",
        "glm-5-turbo",
        "glm-5",
        "glm-4.7",
    ],
    "openrouter": [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "anthropic/claude-fable-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-3.6-flash",
        "google/gemini-2.5-pro",
        "moonshotai/kimi-k3",
        "minimax/minimax-m3",
        "zai/glm-5.2",
        "qwen/qwen3.7-max",
        "qwen/qwen3.6",
    ],
    "opencode": [
        # Complete list from https://opencode.ai/zen/go/v1/models (20 models)
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
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "hy3",
    ],
}

# ── Older IDs that still resolve at the provider API ───────────────────────
# Accepted by validation (kp verify), hidden from display surfaces.
LEGACY_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    # DeepSeek's own API routes the fixed endpoint IDs, not the versioned
    # display IDs — the webapp picker offers exactly these two (see
    # WEBAPP_MODEL_OVERRIDES below).
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "google": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
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
    "kimi-k2": "kimi-k2.6",
    "mimo-m1": "mimo-v2.5",
    "qwq-32b": "qwen3.7-plus",
    # Tencent Hy3: hy3-preview → hy3 (GA transition); keep alias for back-compat.
    "hy3-preview": "hy3",
}


def valid_models(provider: str) -> list[str]:
    """Every ID that should pass validation for *provider*: current + legacy."""
    return list(KNOWN_MODELS.get(provider, [])) + list(LEGACY_MODELS.get(provider, []))


# ── Webapp picker composition ──────────────────────────────────────────────
# (group label, [(value-prefix provider, display label), ...]). Providers not
# listed here are CLI-only (opencode/zai direct, openai) until the webapp
# grows a client for them.
WEBAPP_PROVIDER_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("External", [
        ("deepseek", "DeepSeek API"),
        ("anthropic", "Anthropic"),
        ("google", "Google Gemini"),
    ]),
    ("OpenRouter", [
        ("openrouter", "OpenRouter"),
    ]),
]

# Providers whose webapp offering differs from KNOWN_MODELS. DeepSeek's API
# serves the fixed endpoint IDs (deepseek-chat/-reasoner), not the versioned
# catalog IDs, so the picker must offer what the API actually routes.
WEBAPP_MODEL_OVERRIDES: dict[str, list[str]] = {
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
}


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
            for m in models:
                options.append({
                    "value": f"external_{provider}:{m}",
                    "label": f"{provider_label} · {m}",
                })
        groups.append({"label": group_label, "options": options})
    return groups


__all__ = [
    "KNOWN_MODELS",
    "LEGACY_MODELS",
    "MODEL_ALIASES",
    "valid_models",
    "WEBAPP_PROVIDER_GROUPS",
    "WEBAPP_MODEL_OVERRIDES",
    "webapp_external_model_groups",
]
