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

from .agent._shared_utils import _capped_put

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _ollama_base_url(override: Optional[str] = None) -> str:
    """Resolve the Ollama server URL: explicit override > OLLAMA_BASE_URL env > default."""
    if override:
        return override
    return os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)


# Manual cache: stores **successful** results (including None when the
# server says no explicit num_ctx).  Unlike lru_cache, connection failures
# or timeouts do NOT poison this cache, so a retry after Ollama restarts
# will re-query the server.
#
# Entries are TTL-bounded (_NUM_CTX_CACHE_TTL_SECONDS) so a Modelfile
# num_ctx change + Ollama restart is eventually picked up instead of being
# served stale until process restart.  Each value is stored as a
# ``(num_ctx, inserted_monotonic)`` tuple; time.monotonic() is used because
# it measures elapsed intervals immune to wall-clock adjustments.
_num_ctx_cache: dict = {}
_NUM_CTX_CACHE_TTL_SECONDS = 300  # 5 minutes

# Negative cache: collapses per-request retry storms when the Ollama server
# is unreachable or slow.  query_ollama_num_ctx is invoked on every LLM
# generation (providers.py) and on every preemptive_trim (context_budget.py),
# so a dead or hung server would otherwise add a ``timeout=5`` HTTP POST to
# every single call — catastrophic in a long autonomous run.  A short TTL
# (_NUM_CTX_NEGATIVE_CACHE_TTL_SECONDS) preserves the original "re-query
# after the server restarts" intent while preventing the storm.
#
# Separate from the positive cache because the two None results mean
# different things: positive None = "server answered: no num_ctx configured"
# (don't re-ask for 5 min); negative None = "server unreachable" (re-ask
# within seconds).  Each value is the monotonic timestamp of the failure.
#
# Lock-free like the positive cache and the rest of the module-level cache
# family (ast_cache, repo_files): the agent loop is single-threaded under the
# GIL, and a concurrent double-miss-then-double-write is idempotent (both
# store the same timestamp).
_num_ctx_negative_cache: dict = {}
_NUM_CTX_NEGATIVE_CACHE_TTL_SECONDS = 15  # short: detect a restart promptly

# Bounded entry cap for BOTH num_ctx caches (parity with the module-level cache
# family: ast_cache, repo_files, _shared_utils._PY_WALK_CACHE).  Keys are
# (model_name, base_url); realistic configs sit well under this, but a
# pathological spread of base_url hints can no longer grow either dict without
# bound.  Generous vs the walk/git_sha cap=8 because num_ctx values are tiny
# tuples and a too-small cap would evict live entries and force a re-query on
# every generation (cache churn defeating the 5-min TTL).  TTL-expired entries
# that linger under the cap are harmless: the read-side TTL check rejects them
# and FIFO evicts the oldest once the cap is hit.
_NUM_CTX_CACHE_MAX_ENTRIES: int = 64

def query_ollama_num_ctx(model_name: str, base_url_hint: Optional[str] = None) -> Optional[int]:
    """Query Ollama /api/show for the model's configured num_ctx.

    Returns the configured ``num_ctx`` from the model's Modelfile, or ``None``
    if the model doesn't look like an Ollama model, if Ollama is unreachable,
    or if no explicit ``num_ctx`` is set in the Modelfile.

    Results are **cached per model name** (manual dict, keyed by
    ``(model_name, base_url)``) with a bounded TTL.  Successful responses
    populate a 5-minute positive cache; connection/timeout/HTTP errors
    populate a short-TTL **negative cache** so a dead or slow server does
    not trigger a fresh ``timeout=5`` HTTP POST on every call (the negative
    TTL is short enough that a server restart is detected promptly).  The
    positive TTL ensures a Modelfile ``num_ctx`` change is picked up within
    ``_NUM_CTX_CACHE_TTL_SECONDS`` rather than being served stale.

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

    # Negative cache: a recent failure means skip the HTTP call entirely.
    # Short TTL preserves the "re-query after restart" intent while preventing
    # a dead/slow server from adding a per-request timeout to every call.
    neg_ts = _num_ctx_negative_cache.get(cache_key)
    if neg_ts is not None and (time.monotonic() - neg_ts) < _NUM_CTX_NEGATIVE_CACHE_TTL_SECONDS:
        return None

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
                _capped_put(_num_ctx_cache, cache_key, (result, time.monotonic()), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
                return result

        # Priority 2: parse modelfile text for PARAMETER num_ctx
        modelfile: str = data.get("modelfile", "")
        match = re.search(r'(?im)^PARAMETER\s+num_ctx\s+(\d+)\s*$', modelfile)
        if match:
            result = int(match.group(1))
            _capped_put(_num_ctx_cache, cache_key, (result, time.monotonic()), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
            return result

        # Server responded but no explicit num_ctx in either the structured
        # parameters dict or the modelfile text.  Cache None in the POSITIVE
        # cache (5-min TTL) — this is a definitive "no num_ctx" answer, distinct
        # from the unreachable-server negative cache below.
        _capped_put(_num_ctx_cache, cache_key, (None, time.monotonic()), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
        return None

    except requests.ConnectionError:
        logger.debug("Ollama not reachable at %s", base_url)
        _capped_put(_num_ctx_negative_cache, cache_key, time.monotonic(), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
        return None
    except requests.Timeout:
        logger.debug("Ollama /api/show timed out for model %s", model_name)
        _capped_put(_num_ctx_negative_cache, cache_key, time.monotonic(), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
        return None
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # 404 "model not present on this server" is a STABLE, definitive answer —
            # the same kind of "num_ctx is definitively unavailable" as the no-num_ctx
            # success path above (L143-148), not a transient reachability failure.
            # Cache it in the POSITIVE cache (5-min TTL) so a misnamed model does not
            # re-POST every 15s for the whole 12h run (~2880 wasted calls) — exactly
            # the per-request storm the negative cache was added to prevent. Only
            # genuinely transient HTTP errors (5xx / 429 / etc.) use the negative cache.
            logger.debug("Model %s not found on Ollama server", model_name)
            _capped_put(_num_ctx_cache, cache_key, (None, time.monotonic()), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
        else:
            logger.debug("Ollama /api/show HTTP error for %s: %s", model_name, e)
            _capped_put(_num_ctx_negative_cache, cache_key, time.monotonic(), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
        return None
    except Exception as e:
        logger.debug("Ollama /api/show failed for %s: %s", model_name, e)
        _capped_put(_num_ctx_negative_cache, cache_key, time.monotonic(), cap=_NUM_CTX_CACHE_MAX_ENTRIES)
        return None


