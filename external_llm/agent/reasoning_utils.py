"""Reasoning A/B control utilities — shared by Planner and Developer LLM.

Provides a single function ``reasoning_ab_kwargs(env_var)`` that reads an
environment variable to decide whether to inject a reasoning-suppression
fragment into the LLM request payload.

Usage::

    _reasoning_kwargs = reasoning_ab_kwargs("ASICODE_PLANNER_REASONING")
    resp = client.chat(..., **_reasoning_kwargs)
"""

import json as _json
import logging
import os

logger = logging.getLogger(__name__)


def reasoning_ab_kwargs(env_var: str = "ASICODE_PLANNER_REASONING") -> dict:
    """Return extra request kwargs to control model reasoning (A/B knob).

    Controlled by env so an A/B arm can be flipped without code changes::

        ASICODE_PLANNER_REASONING     on (default) | off
        ASICODE_DEEPSEEK_NOTHINK_JSON  JSON fragment merged into the request when
                                   reasoning is off (the API field is
                                   deployment-specific). Default targets the
                                   common ``thinking.type=disabled``.

    Returns ``{}`` (no change) unless reasoning is explicitly turned off.  The
    returned keys are **not** in ``DeepSeekClient._NON_SERIALIZABLE_KEYS``, so
    they flow straight into the request payload.
    """
    _mode = os.getenv(env_var, "on").strip().lower()
    if _mode not in ("off", "0", "false", "no"):
        return {}

    # DeepSeek V4-Flash unifies thinking/non-thinking under one model.  The
    # effective control (confirmed via tools/probe_reasoning.py against the
    # live endpoint) is thinking.type=disabled — a ThinkingOptions struct.
    # The documented reasoning.enabled flag is silently ignored on this
    # deployment. Override via env if a deployment differs.
    _raw = os.getenv(
        "ASICODE_DEEPSEEK_NOTHINK_JSON",
        '{"thinking": {"type": "disabled"}}',
    )
    try:
        _frag = _json.loads(_raw)
        if not isinstance(_frag, dict):
            raise ValueError(f"{env_var} JSON must be a JSON object")
    except Exception as exc:
        logger.warning(
            "[%s] bad ASICODE_DEEPSEEK_NOTHINK_JSON (%r): %s — "
            "suppression skipped (reasoning stays ON)", env_var, _raw, exc,
        )
        return {}
    logger.info(
        "[%s] suppression ON — injecting %s "
        "(verify via reasoning→~0 tokens)", env_var, _frag,
    )
    return _frag
