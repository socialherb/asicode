"""Ollama API utilities — dynamically query model configuration from the Ollama server.

A single :func:`_query_ollama_show` performs ONE ``/api/show`` POST per
``(model, server)`` and caches the FULL response payload.  Both
:func:`query_ollama_num_ctx` (used by providers.py and context_budget.py) and
:func:`query_ollama_capabilities` (used by model_registry.py) are thin field
extractors over that one shared cached payload, so a cold start issues a single
POST instead of two (or three, when the agent tools path also runs).  This
module is what keeps providers.py (num_ctx enforcement) and context_budget.py
(context-limit resolution) in sync without duplicated heuristics.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import requests

from .common.cache_utils import _capped_put

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _ollama_base_url(override: Optional[str] = None) -> str:
    """Resolve the Ollama server URL: explicit override > OLLAMA_BASE_URL env > default."""
    if override:
        return override
    return os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)


# ── Unified /api/show cache ──────────────────────────────────────────────────
#
# A single cache holds the FULL /api/show JSON payload (or None for a
# definitive "no data" answer — 404 model-not-found, or a successful response
# that simply lacks the requested fields).  num_ctx and capabilities are BOTH
# fields of that one payload, so a single POST + cache entry serves both
# queries.  Before this consolidation each query maintained its own positive
# cache and issued its own POST, doubling (tripling when the tools path ran)
# the per-turn /api/show traffic on a cold cache or after the 5-min TTL.
#
# Cache key: ``(model_name.lower().strip(), base_url)``.  Lowercasing the
# model-name component is what lets callers that normalise differently
# converge on one entry — providers.py passes the raw model string while
# context_budget.py passes an already-lowercased name; without this they would
# cache-miss on every model whose tag has any uppercase letter.  num_ctx and
# capabilities are model properties (not name-spelling properties), so a
# case-insensitive key is correct.  The POST body still uses the ORIGINAL
# model name so an exact-match server sees the spelling the caller supplied.
#
# Entries are TTL-bounded (_SHOW_CACHE_TTL_SECONDS) so a Modelfile num_ctx /
# capability change + Ollama restart is eventually picked up instead of being
# served stale until process restart.  Each value is stored as a
# ``(payload, inserted_monotonic)`` tuple; time.monotonic() is used because it
# measures elapsed intervals immune to wall-clock adjustments.
_ollama_show_cache: dict = {}
_SHOW_CACHE_TTL_SECONDS = 300  # 5 minutes

# Negative cache: collapses per-request retry storms when the Ollama server is
# unreachable or slow.  _query_ollama_show is reached on every LLM generation
# (providers.py) and on every context-limit resolution
# (context_budget.py), so a dead or hung server would otherwise add a
# ``timeout=5`` HTTP POST to every single call — catastrophic in a long
# autonomous run.  A short TTL (_SHOW_NEGATIVE_CACHE_TTL_SECONDS) preserves the
# original "re-query after the server restarts" intent while preventing the
# storm.
#
# Separate from the positive cache because the absence of an entry means two
# different things: positive None = "server answered: model not found / no
# fields" (don't re-ask for 5 min); negative (no entry stored) = "server
# unreachable" (re-ask within seconds).  The value is the monotonic timestamp
# of the failure.
#
# Lock-free like the positive cache: the agent loop is single-threaded under the
# GIL, and a concurrent double-miss-then-double-write is idempotent (both store
# the same timestamp).
_ollama_show_negative_cache: dict = {}
_SHOW_NEGATIVE_CACHE_TTL_SECONDS = 15  # short: detect a restart promptly

# Bounded entry cap for BOTH caches.  Keys are (model_name, base_url); realistic
# configs sit well under this, but a pathological spread of base_url hints can
# no longer grow either dict without bound.  Generous vs the walk/git_sha
# cap=8 because payloads are tiny tuples and a too-small cap would evict live
# entries and force a re-query on every generation (cache churn defeating the
# 5-min TTL).  TTL-expired entries that linger under the cap are harmless: the
# read-side TTL check rejects them and FIFO evicts the oldest once the cap is
# hit.
_SHOW_CACHE_MAX_ENTRIES: int = 64


def _show_cache_key(model_name: str, base_url: str) -> tuple[str, str]:
    """Canonical cache key: case-insensitive model name + resolved base_url.

    Lowercasing the model name lets providers.py (raw model string) and
    context_budget.py (already-lowercased name) hit the SAME entry, so an
    uppercase tag no longer forces a duplicate POST.
    """
    return (model_name.lower().strip(), base_url)


def _query_ollama_show(model_name: str, base_url_hint: Optional[str] = None) -> Optional[dict]:
    """Query Ollama ``/api/show`` ONCE and cache the full response payload.

    Returns the parsed ``/api/show`` JSON dict on success; ``None`` for a
    definitive "no data" answer (404 model-not-found, or a successful response
    that simply lacks the requested fields — callers derive a definitive None
    for the 5-min TTL); ``None`` (via the shared negative cache) when the
    server is unreachable (re-queried within the short negative TTL).

    Both :func:`query_ollama_num_ctx` and :func:`query_ollama_capabilities`
    delegate here so a cold start issues ONE ``/api/show`` POST instead of two
    (or three, when the agent tools path also queries capabilities).
    """
    # Only native Ollama tags (colon-separated, no path separators like
    # OpenRouter's "qwen/qwen3:8b") are queryable.
    if ":" not in model_name or "/" in model_name:
        return None

    base_url = _ollama_base_url(base_url_hint)
    cache_key = _show_cache_key(model_name, base_url)

    # Return a fresh cached successful result if available.  Expired entries
    # (older than _SHOW_CACHE_TTL_SECONDS) fall through to a re-query so a
    # Modelfile/capability change is eventually reflected.
    cached = _ollama_show_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[1]) < _SHOW_CACHE_TTL_SECONDS:
        return cached[0]

    # Negative cache: a recent failure means skip the HTTP call entirely.
    # Short TTL preserves the "re-query after restart" intent while preventing
    # a dead/slow server from adding a per-request timeout to every call.
    neg_ts = _ollama_show_negative_cache.get(cache_key)
    if neg_ts is not None and (time.monotonic() - neg_ts) < _SHOW_NEGATIVE_CACHE_TTL_SECONDS:
        return None

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"model": model_name},
            timeout=5,
        )
        resp.raise_for_status()
        data: dict = resp.json()
        _capped_put(_ollama_show_cache, cache_key, (data, time.monotonic()), cap=_SHOW_CACHE_MAX_ENTRIES)
    except requests.ConnectionError:
        logger.debug("Ollama not reachable at %s", base_url)
        _capped_put(_ollama_show_negative_cache, cache_key, time.monotonic(), cap=_SHOW_CACHE_MAX_ENTRIES)
        return None
    except requests.Timeout:
        logger.debug("Ollama /api/show timed out for model %s", model_name)
        _capped_put(_ollama_show_negative_cache, cache_key, time.monotonic(), cap=_SHOW_CACHE_MAX_ENTRIES)
        return None
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # 404 "model not present on this server" is a STABLE, definitive
            # answer — the same kind of "data definitively unavailable" as a
            # successful response that lacks the requested field.  Cache None
            # in the POSITIVE cache (5-min TTL) so a misnamed model does not
            # re-POST every 15s for the whole 12h run (~2880 wasted calls) —
            # exactly the per-request storm the negative cache was added to
            # prevent.  Only genuinely transient HTTP errors (5xx / 429 / etc.)
            # use the negative cache.
            logger.debug("Model %s not found on Ollama server", model_name)
            _capped_put(_ollama_show_cache, cache_key, (None, time.monotonic()), cap=_SHOW_CACHE_MAX_ENTRIES)
        else:
            logger.debug("Ollama /api/show HTTP error for %s: %s", model_name, e)
            _capped_put(_ollama_show_negative_cache, cache_key, time.monotonic(), cap=_SHOW_CACHE_MAX_ENTRIES)
        return None
    except Exception as e:
        logger.debug("Ollama /api/show failed for %s: %s", model_name, e)
        _capped_put(_ollama_show_negative_cache, cache_key, time.monotonic(), cap=_SHOW_CACHE_MAX_ENTRIES)
        return None
    else:
        return data


def query_ollama_num_ctx(model_name: str, base_url_hint: Optional[str] = None) -> Optional[int]:
    """Query Ollama ``/api/show`` for the model's configured ``num_ctx``.

    Returns the configured ``num_ctx`` from the model's Modelfile, or ``None``
    if the model doesn't look like an Ollama model, if Ollama is unreachable,
    or if no explicit ``num_ctx`` is set in the Modelfile.

    Results are **cached per model name** (manual dict, keyed by the canonical
    ``(model_name, base_url)`` — see :func:`_show_cache_key`) with a bounded
    TTL, owned by :func:`_query_ollama_show`.  Successful responses populate a
    5-minute positive cache; connection/timeout/HTTP errors populate a
    short-TTL **negative cache** so a dead or slow server does not trigger a
    fresh ``timeout=5`` HTTP POST on every call.  This function is a thin field
    extractor over the single shared ``/api/show`` payload.

    Priority of extraction:
        1. ``parameters.num_ctx`` structured field (newer Ollama versions)
        2. ``modelfile`` text parsing for ``PARAMETER num_ctx <value>``
    """
    data = _query_ollama_show(model_name, base_url_hint)
    if data is None:
        return None
    # Priority 1: structured "parameters" dict (Ollama 0.5+)
    params = data.get("parameters")
    if isinstance(params, dict):
        num_ctx = params.get("num_ctx")
        if num_ctx is not None:
            return int(num_ctx)
    # Priority 2: parse modelfile text for PARAMETER num_ctx
    modelfile: str = data.get("modelfile", "")
    match = re.search(r'(?im)^PARAMETER\s+num_ctx\s+(\d+)\s*$', modelfile)
    if match:
        return int(match.group(1))
    return None


def query_ollama_capabilities(model_name: str, base_url_hint: Optional[str] = None) -> Optional[tuple[str, ...]]:
    """Query Ollama ``/api/show`` for the model's capability list (Ollama 0.6+).

    Returns a tuple of capability names (e.g. ``("completion", "tools",
    "vision")``) when the server reports them; ``None`` when the server is
    unreachable, the model is not an Ollama-native tag, the model is not on the
    server (404), or the response has no ``capabilities`` field (older Ollama —
    the field was added in 0.6.0).

    Caching mirrors :func:`query_ollama_num_ctx` — the two queries share the
    SAME positive + negative cache (owned by :func:`_query_ollama_show`), so if
    the num_ctx query just populated the payload, the capabilities query is a
    free dict lookup (and vice versa).  This is a thin field extractor over the
    single shared ``/api/show`` payload.
    """
    data = _query_ollama_show(model_name, base_url_hint)
    if data is None:
        return None
    caps = data.get("capabilities")
    if isinstance(caps, list) and caps:
        return tuple(str(c) for c in caps)
    return None
