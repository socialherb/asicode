"""Ollama API utilities — dynamically query model configuration from the Ollama server.

This module provides shared functions that both providers.py (num_ctx enforcement)
and context_budget.py (preemptive_trim guard) use, keeping them in sync without
duplicated heuristics.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _ollama_base_url(override: Optional[str] = None) -> str:
    """Resolve the Ollama server URL: explicit override > OLLAMA_BASE_URL env > default."""
    if override:
        return override
    return os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)


# Manual cache: stores **successful** results (including None when the
# server says no explicit num_ctx).  Unlike lru_cache, connection failures
# or timeouts do NOT poison the cache, so a retry after Ollama restarts
# will re-query the server.
#
# Entries are TTL-bounded (_NUM_CTX_CACHE_TTL_SECONDS) so a Modelfile
# num_ctx change + Ollama restart is eventually picked up instead of being
# served stale until process restart.  Each value is stored as a
# ``(num_ctx, inserted_monotonic)`` tuple; time.monotonic() is used because
# it measures elapsed intervals immune to wall-clock adjustments.
_num_ctx_cache: dict = {}
_NUM_CTX_CACHE_TTL_SECONDS = 300  # 5 minutes

def query_ollama_num_ctx(model_name: str, base_url_hint: Optional[str] = None) -> Optional[int]:
    """Query Ollama /api/show for the model's configured num_ctx.

    Returns the configured ``num_ctx`` from the model's Modelfile, or ``None``
    if the model doesn't look like an Ollama model, if Ollama is unreachable,
    or if no explicit ``num_ctx`` is set in the Modelfile.

    Results are **cached per model name** (manual dict, keyed by
    ``(model_name, base_url)``) with a bounded TTL.  Only successful
    responses are cached — connection failures or timeouts do NOT poison
    the cache, so a subsequent call after Ollama restarts will re-query the
    server; the TTL also ensures a Modelfile ``num_ctx`` change is picked up
    within ``_NUM_CTX_CACHE_TTL_SECONDS`` rather than being served stale
    until process restart.

    Priority of queries:
        1. ``parameters.num_ctx`` structured field (newer Ollama versions)
        2. ``modelfile`` text parsing for ``PARAMETER num_ctx <value>``
    """
    # Only try for models that look like Ollama native format (colon-separated tag,
    # no path separators like OpenRouter's "qwen/qwen3.6-27b-20260422")
    if ":" not in model_name or "/" in model_name:
        return None

    base_url = _ollama_base_url(base_url_hint)
    cache_key = (model_name, base_url)

    # Return a fresh cached successful result if available.  Expired entries
    # (older than _NUM_CTX_CACHE_TTL_SECONDS) fall through to a re-query so
    # a Modelfile num_ctx change is eventually reflected.
    cached = _num_ctx_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[1]) < _NUM_CTX_CACHE_TTL_SECONDS:
        return cached[0]

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"model": model_name},
            timeout=5,
        )
        resp.raise_for_status()
        data: dict = resp.json()

        # Priority 1: structured "parameters" dict (Ollama 0.5+)
        params = data.get("parameters")
        if isinstance(params, dict):
            num_ctx = params.get("num_ctx")
            if num_ctx is not None:
                result = int(num_ctx)
                _num_ctx_cache[cache_key] = (result, time.monotonic())
                return result

        # Priority 2: parse modelfile text for PARAMETER num_ctx
        modelfile: str = data.get("modelfile", "")
        match = re.search(r'(?im)^PARAMETER\s+num_ctx\s+(\d+)\s*$', modelfile)
        if match:
            result = int(match.group(1))
            _num_ctx_cache[cache_key] = (result, time.monotonic())
            return result

        # Server responded but no explicit num_ctx in either the structured
        # parameters dict or the modelfile text.  We cache None (TTL-bounded)
        # so subsequent calls don't re-query the server on every request — the
        # connection/timeout paths below intentionally skip the cache, allowing
        # a retry after Ollama restarts or the model is updated.
        _num_ctx_cache[cache_key] = (None, time.monotonic())
        return None

    except requests.ConnectionError:
        logger.debug("Ollama not reachable at %s", base_url)
        return None
    except requests.Timeout:
        logger.debug("Ollama /api/show timed out for model %s", model_name)
        return None
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.debug("Model %s not found on Ollama server", model_name)
        else:
            logger.debug("Ollama /api/show HTTP error for %s: %s", model_name, e)
        return None
    except Exception as e:
        logger.debug("Ollama /api/show failed for %s: %s", model_name, e)
        return None


