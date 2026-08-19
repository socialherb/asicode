"""Context Budget Manager - token-aware message fitting (budget check only, no truncation)."""
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import json
import logging
import os
import threading
import time
from typing import (
    TYPE_CHECKING,
    Optional,  # f821-protected
)

from external_llm.agent.config.thresholds import _env_int
from external_llm.agent.message_shapes import _is_anthropic_tool_result, is_tool_call, is_tool_result

if TYPE_CHECKING:
    pass
logger = logging.getLogger(__name__)

# Model-specific context window limits (tokens).
# Only exact match is used — no prefix matching, so every known model name that
# differs from its base model must have its own entry.  Only models whose context
# is *smaller* than _DEFAULT_CONTEXT_LIMIT need entries; models at or above 1M
# use the fallback without an explicit entry.
# Sources — API docs / official pricing pages / OpenRouter, verified 2026-07-04:
#   OpenAI:    https://developers.openai.com/api/docs/models
#   Anthropic: https://tygartmedia.com/claude-token-limit/
#   Google:    https://ai.google.dev/gemini-api/docs/long-context
#   DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing
#   GLM-5:     https://github.com/zai-org/GLM-5
#   OpenRouter:https://openrouter.ai/
#   Kimi:      https://platform.kimi.ai/docs/models
# DeepSeek v4-flash/v4-pro, GLM-5.2, Qwen3.7-max/plus, Qwen3.6-plus/3.5-plus,
# MiMo-v2.5-pro/v2.5/v2-pro, MiniMax-M3, kimi-k3, deepseek-chat, deepseek-reasoner
# are all 1M+ models — no explicit entry needed (use _DEFAULT_CONTEXT_LIMIT fallback).
_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4o":           128_000,
    "gpt-4o-mini":      128_000,
    "gpt-4o-2024-08-06": 128_000,
    "o3":               200_000,
    "o3-mini":          200_000,
    "o3-mini-high":     200_000,
    "o4-mini":          200_000,
    "o4-mini-high":     200_000,
    # Anthropic — all modern Claude models (Sonnet/Opus/Haiku generations 3-5)
    # share a 200K context window. Listed explicitly (no prefix matching) per the
    # table's design; new variants must be added here or they fall back to 1M.
    # That warning described a real defect for six of them, including the two
    # flagship ids the CLI was actively offering (claude-opus-5, claude-fable-5)
    # — see test_model_catalog_context_parity, which now fails when a catalog
    # model reaches this table without a decision.
    "claude-fable-5":            200_000,
    "claude-opus-5":             200_000,
    "claude-sonnet-5":           200_000,
    "claude-haiku-4-5":          200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # OpenRouter serves this model under the dot spelling ``claude-sonnet-4.6``
    # (same pinned snapshot as the hyphen form below); both resolve to 200K.
    "claude-sonnet-4.6":         200_000,
    "claude-sonnet-4-6":         200_000,
    "claude-sonnet-4-5":         200_000,
    "claude-opus-4-8":           200_000,
    "claude-opus-4-7":           200_000,
    "claude-opus-4-6":           200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-sonnet":           200_000,
    "claude-3-opus-20240229":    200_000,
    "claude-3-sonnet-20240229":  200_000,
    "claude-3-haiku-20240307":   200_000,
    # DeepSeek — original deepseek-r1 (64K context); deepseek-chat/reasoner are
    # deprecated aliases for deepseek-v4-flash thinking/non-thinking → 1M fallback.
    "deepseek-r1":       64_000,
    # Zhipu GLM (zai + opencode). glm-5.3 is the DEFAULT_MODEL and 1M is verified
    # (Z.ai docs: "Context Length: 1M" — generational leap from the 200K family).
    # Listed EXPLICITLY (not via _DEFAULT fallback) so the default model's window
    # cannot silently drift if _DEFAULT_CONTEXT_LIMIT changes. glm-5/5.1/5-turbo 200K; glm-4.7 128K.
    "glm-5.3":          1_000_000,
    "glm-5.2":          1_000_000,
    "glm-5.1":          200_000,
    "glm-5-turbo":      200_000,
    "glm-5":            200_000,
    "glm-4.7":          128_000,
    # Qwen3 (opencode provider) — 3.8-max/3.7-max/plus, 3.6-plus, 3.5-plus are 1M (fallback)
    # qwen3.6 is the base model at 262_144 (= 2^18 = binary 256K). Source: openrouter.ai.
    "qwen3.6":          262_144,
    # Xiaomi MiMo (opencode) — v2.5-pro/v2.5/v2-pro are 1M (fallback)
    # mimo-v2-omni has 256_000 (decimal 256K). Source: openrouter.ai.
    "mimo-v2-omni":     256_000,
    # Moonshot Kimi (opencode provider)
    # kimi-k3 is a 1M+ model (1,048,576 = 2^20) — uses _DEFAULT_CONTEXT_LIMIT fallback
    # (no explicit entry); variants like kimi-k3-0711/kimi-k3-turbo resolve uniformly to 1M.
    # kimi-k2.7-code uses binary 256K (262_144 = 2^18). kimi-k2.6/k2.5 use decimal 256K.
    # Source: platform.kimi.ai/docs/models.
    "kimi-k2.7-code":   262_144,
    "kimi-k2.6":        256_000,
    "kimi-k2.5":        256_000,
    # MiniMax (opencode provider) — M3 is 1M (fallback)
    # minimax-m2.7/m2.5: 205_000 tokens — per OpenRouter model specs (non-standard size).
    "minimax-m2.7":     205_000,
    "minimax-m2.5":     205_000,
    # Tencent Hy3 (opencode provider) — GA as "hy3"; keep conservative 128K.
    # hy3-preview is aliased to hy3 in _MODEL_ALIASES (asi.py); both resolve here.
    "hy3":              128_000,
    "hy3-preview":      128_000,
    # Grok 4.5 (opencode) — 500K context (xAI docs: a reduction from Grok 4.3's 1M).
    # MUST be explicit: the _DEFAULT_CONTEXT_LIMIT fallback is 1M, which would
    # over-allocate and risk HTTP errors on >500K-token requests.
    "grok-4.5":         500_000,
}


# Family-prefix fallback (verified windows) — catches ids whose EXACT name is
# not in _CONTEXT_LIMITS but whose family window is known: pinned-date ids
# (claude-opus-5-20260101), -fast/-mini variants (grok-4.5-fast), future
# generational ids (claude-fable-6). Without it they silently took the 1M
# fallback against a 200K/500K real window, leaving the pre-flight cap inert
# until the provider 400'd. Values MUST be at or below the family's real
# window (over-allocation risks HTTP errors; under-allocation only trims early).
_FAMILY_PREFIX_LIMITS: tuple[tuple[str, int], ...] = (
    ("claude-", 200_000),   # every claude-* entry above is 200K — shared by pinned dates
    ("grok-4.5", 500_000),  # grok-4.5 variants share the 500K window
    # Qwen2.5-Coder series (local Ollama: qwen2.5-coder:0.5b/1.5b/3b/7b/14b/32b).
    # The MODEL supports 128K tokens (Qwen2.5-Coder technical report); Ollama
    # serves a smaller num_ctx by default (4096) but _num_ctx_for_model raises it
    # to at least 8192. The dynamic /api/show query (priority 0) reads the user's
    # Modelfile value when one is set; this entry is the static fallback so a
    # server that is unreachable or lacks num_ctx uses 128K instead of the 1M
    # fallback (which is 122x the actual server window and made the pre-flight
    # cap inert). qwen2.5-coder is the OllamaClient.DEFAULT_MODEL.
    ("qwen2.5-coder", 128_000),
)


# Default context limit (fallback for unknown models).
_DEFAULT_CONTEXT_LIMIT = 1_000_000

# Catalog models for which reaching _DEFAULT_CONTEXT_LIMIT is a DECISION, not a
# miss. The fallback is silent and errs toward over-allocation, so "absent from
# _CONTEXT_LIMITS" cannot distinguish "1M is right" from "nobody added it" —
# which is exactly how six Claude ids sat at 1M against a real 200K window.
# test_model_catalog_context_parity requires every model_catalog id to appear in
# _CONTEXT_LIMITS or here, so adding a model forces the question to be answered.
_FALLBACK_IS_CORRECT: frozenset[str] = frozenset({
    # ── Verified 1M+ (sources in the _CONTEXT_LIMITS header above) ──────────
    "deepseek-v4-flash", "deepseek-v4-pro",
    "deepseek-chat", "deepseek-reasoner",   # aliases of v4-flash thinking/non-thinking
    "kimi-k3",                              # 1,048,576 = 2^20
    # Meta Muse Spark 1.2 (+contributor) — 1,048,576 = 2^20, carried over from
    # 1.1 unchanged. Sources: OpenRouter model page (meta/muse-spark-1.2), Meta
    # research blog 2026-08-05, eesel rate-card table (both variants 1M).
    # Fallback (1M decimal) slightly under-allocates vs 2^20 — safe direction.
    "muse-spark-1.2", "muse-spark-1.2-contributor",
    "minimax-m3",
    "mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus",
    # Gemini long-context line — 1M input across the 2.0/2.5 pro+flash tiers.
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite-001",

    # ── UNVERIFIED: window not confirmed against a provider source ──────────
    # These keep the 1M fallback, i.e. exactly the behaviour they had before
    # this gate existed — listing them changes nothing at runtime, it only
    # records that the number is unknown rather than agreed. If any is in fact
    # smaller, the symptom is the pre-flight cap in agent_loop staying inert
    # until the provider 400s, after which _record_context_overflow converges
    # on the real size. Move an entry into _CONTEXT_LIMITS once its window is
    # published.
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
    "gemini-3.1-pro", "gemini-3-flash",
    "qwen3.8-max",
})

# Models already warned about the 1M fallback (once per model per process).
_warned_unknown_models: set[str] = set()

# Runtime overrides: model → reduced context limit set after a context-length 400.
# Allows the reactive backstop to progressively reduce a misconfigured limit until
# the provider stops rejecting the prompt (see _record_context_overflow).
_context_window_overrides: dict[str, int] = {}

# ── Thread safety ──────────────────────────────────────────────────────────────
_override_lock = threading.RLock()  # RLock allows nested acquire by same thread
                                        # (_record_context_overflow locks then calls
                                        # _save_override_cache which also locks).

# ── TTL & reduction-capped overrides ───────────────────────────────────────────
# Overrides self-expire after _OVERRIDE_TTL_SECONDS of inactivity, preventing a
# single spurious 400 from permanently shrinking the window.  Additionally, each
# model has a _MAX_OVERRIDE_REDUCTIONS cap to avoid unbounded ratcheting.
# Parsed via _env_int, NOT a bare int(): these run at module import time, and
# this module is imported by agent_loop, so a bare int() turns any malformed
# value into an ImportError that kills every agent run. `export
# CONTEXT_OVERRIDE_TTL=` (empty — the common way to "unset" a var in a shell
# profile or a docker-compose `environment:` entry) was enough to do it, since
# int("") raises. Config parsing must degrade to the default, never abort the
# process. _env_int also rejects out-of-range values that silently disabled the
# mechanism they configure: TTL <= 0 expires every override the instant it is
# recorded, so the overflow-recovery cache stops working with no error at all.
# MAX_REDUCTIONS takes minimum=0 because 0 is a coherent setting there ("never
# step a model's window down"), unlike a zero-second TTL.
_OVERRIDE_TTL_SECONDS = _env_int("CONTEXT_OVERRIDE_TTL", 1800)              # 30 minutes
_MAX_OVERRIDE_REDUCTIONS = _env_int("CONTEXT_MAX_REDUCTIONS", 3, minimum=0)  # max step-downs per model

# Model → {ts: float, reductions: int, limit: int}
_override_meta: dict[str, dict] = {}

# ── On-disk cache ──────────────────────────────────────────────────────────────
# Persists overrides across restarts so a misconfigured 1M-fallback model doesn't
# hard-fail on every fresh process.  Best-effort: corrupted/missing file ignored.
# Overridable via ``ASICODE_CONTEXT_OVERRIDE_CACHE``, matching the convention
# ``ASICODE_RUNS_DIR`` and ``ASICODE_WRITE_TOOL_FAILURE_LOG`` already follow.
#
# An env var rather than a patchable attribute because the write that needed
# redirecting happens in an atexit flush, and 63 test files spawn Python
# subprocesses: a child importing this module gets a fresh copy of the constant
# and its own atexit handler, so no amount of in-process patching in the parent
# reaches it. Measured — a bare ``python -c "import
# external_llm.agent.context_budget"`` was enough to rewrite the real file to
# ``{}`` at exit. Env vars are inherited, so this covers the children too.
_OVERRIDE_CACHE_FILE = os.environ.get("ASICODE_CONTEXT_OVERRIDE_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "asicode", "context_override_cache.json",
)
_last_cache_save: float = 0
# True once THIS process has recorded an overflow of its own — i.e. its
# in-memory snapshot has diverged from the on-disk baseline loaded at import.
# Both the atexit force-flush and debounced saves check this, so a process that
# merely *imported* the module (a short-lived subagent worker, a test
# subprocess, a webapp that never hit a 400) never overwrites entries a
# concurrent process wrote to disk. Verified necessity: a bare
# ``import external_llm.agent.context_budget`` was enough to clobber the file.
_override_dirty: bool = False
_CACHE_SAVE_INTERVAL = 5.0  # seconds between disk writes (debounce)


def _read_override_cache() -> dict[str, dict]:
    """Read & validate the on-disk override cache → ``{model: entry}``.

    Best-effort: a missing or unreadable file yields ``{}``.  Each entry is
    validated in isolation (must be a dict containing ``"limit"``) so one
    corrupt entry does not discard the rest, and entries older than
    ``_OVERRIDE_TTL_SECONDS`` are dropped — they would be ignored on load
    anyway and must not be echoed back to disk by a merge-on-write flush.
    Shared by the import-time load and the save-path merge so the two never
    disagree about what counts as a valid entry.
    """
    try:
        if not os.path.exists(_OVERRIDE_CACHE_FILE):
            return {}
        with open(_OVERRIDE_CACHE_FILE, encoding="utf-8") as _f:
            _data = json.load(_f)
    except Exception:
        logger.debug("context_budget: override cache load failed", exc_info=True)  # best-effort (file-level: missing, corrupt JSON, IO error)
        return {}
    _now = time.time()
    _out: dict[str, dict] = {}
    for _model, _entry in _data.items():
        try:
            if not isinstance(_entry, dict) or "limit" not in _entry:
                continue
            if _now - _entry.get("ts", 0) < _OVERRIDE_TTL_SECONDS:
                _out[_model] = _entry
        except Exception:
            logger.debug("context_budget: skipping corrupt override entry for %r", _model)
            continue  # skip corrupt entry, keep processing rest
    return _out


def _ensure_override_cache_loaded() -> None:
    """Load persisted overrides from disk (best-effort, called once at module init).

    Populates ``_context_window_overrides`` / ``_override_meta`` from the
    validated snapshot returned by :func:`_read_override_cache`.  Does NOT mark
    the cache dirty — loading establishes the baseline against which the dirty
    guard measures divergence, so a process that only imports this module (and
    never records an overflow of its own) will not clobber entries a concurrent
    process wrote to disk.
    """
    for _model, _entry in _read_override_cache().items():
        _context_window_overrides[_model] = _entry["limit"]
        _override_meta[_model] = _entry


def _save_override_cache(force: bool = False) -> None:
    """Persist current overrides to disk (debounced, best-effort, atomic writes).

    Writes to a temp file then atomically renames to the target path so a
    concurrent reader or writer never sees a partial/corrupted JSON file.
    The temp file uses a pid+time_ns suffix so concurrent writers (e.g. atexit
    flush vs. worker thread, or two overlapping processes) do not collide.

    Two cross-process safety mechanisms (CLI + webapp + parallel subagents all
    share one cache file):

    1. **Dirty guard** — a process that only *loaded* the cache (never recorded
       an overflow of its own) returns immediately.  Without this, its atexit
       force-flush would overwrite the shared file with a snapshot taken at
       import time, silently deleting entries a concurrent process wrote in the
       meantime.  ``force`` bypasses the debounce but NOT this guard: a clean
       process flushes nothing.

    2. **Merge-on-write** — reconcile the in-memory snapshot with whatever a
       concurrent process wrote to disk, keeping the freshest entry per model
       (highest ``ts``) so neither writer loses data to last-writer-wins.

    Args:
        force: When True, skip the debounce interval check and write immediately.
               Used by ``atexit`` flush to prevent losing the last override write.
               The dirty guard still applies — a clean process flushes nothing.
    """
    global _last_cache_save, _override_dirty
    if not _override_dirty:
        return
    _now = time.monotonic()
    if not force and _now - _last_cache_save < _CACHE_SAVE_INTERVAL:
        return
    _tmp = None
    try:
        os.makedirs(os.path.dirname(_OVERRIDE_CACHE_FILE), exist_ok=True)
        _tmp = _OVERRIDE_CACHE_FILE + ".tmp." + str(os.getpid()) + "." + str(time.time_ns())
        # Snapshot + merge + debounce update under lock — prevents 'dictionary
        # changed size during iteration' when a concurrent writer mutates
        # _override_meta while we serialize it, and the debounce update inside
        # the lock prevents a race where two threads pass the debounce check
        # before either writes.
        with _override_lock:
            _merged = _read_override_cache()
            for _model, _entry in _override_meta.items():
                _disk = _merged.get(_model)
                if _disk is None or _entry.get("ts", 0) >= _disk.get("ts", 0):
                    _merged[_model] = _entry
            _snapshot = _merged
            _last_cache_save = time.monotonic()
            _override_dirty = False
        with open(_tmp, "w", encoding="utf-8") as _f:
            json.dump(_snapshot, _f, ensure_ascii=False)
        os.replace(_tmp, _OVERRIDE_CACHE_FILE)  # atomic on POSIX & Windows
    except Exception:
        # P5: Clean up tmp file on failure (best-effort) to prevent file leaks.
        # Re-mark dirty so a later flush retries the write rather than silently
        # dropping the in-memory override.
        _override_dirty = True
        with contextlib.suppress(OSError):  # tmp cleanup is best-effort
            if _tmp and os.path.exists(_tmp):
                os.unlink(_tmp)


# Load persisted overrides at module init.
_ensure_override_cache_loaded()

# Flush on-disk cache at process exit so the last override write is not lost
# when the debounce interval hasn't elapsed.  Registered after load so a
# load-time crash does not clobber an existing cache file.
atexit.register(lambda: _save_override_cache(force=True))


def _resolve_base_context_limit(model_name: str, base_url: Optional[str] = None) -> int:
    """Compute the configured context limit WITHOUT runtime overrides.

    Like ``_resolve_context_limit`` but skips ``_context_window_overrides`` so
    ``_record_context_overflow`` can compute the base value before reducing it.
    """
    model_lower = model_name.lower().strip()

    # 0. Dynamic query from Ollama API (Option B) — for native Ollama format only.
    #    Runs BEFORE the bare-name reduction and on the raw tag: an Ollama tag is
    #    the one spelling where the colon is part of the model id rather than a
    #    routing prefix, so normalising first would hand /api/show a name Ollama
    #    does not serve.
    if ":" in model_lower and "/" not in model_lower:
        from external_llm.ollama_api import query_ollama_num_ctx
        api_ctx = query_ollama_num_ctx(model_lower, base_url_hint=base_url)
        if api_ctx is not None:
            logger.debug("num_ctx=%d from Ollama API for model %s", api_ctx, model_lower)
            return api_ctx

    # 1. Exact match in _CONTEXT_LIMITS, on the BARE catalog id.
    #    The table is keyed on bare ids but a model arrives spelled however its
    #    route spells it, and the lookup was exact — so every OpenRouter slug
    #    (``anthropic/claude-sonnet-5``) missed the entry its own bare form
    #    (``claude-sonnet-5``) hits, and took the 1M fallback against a 200K
    #    window. Shares model_registry's normaliser rather than repeating it.
    from external_llm.model_registry import bare_model_name
    bare = bare_model_name(model_lower)
    if bare in _CONTEXT_LIMITS:
        return _CONTEXT_LIMITS[bare]

    # 1b. Family-prefix fallback (verified family window) — see
    #     _FAMILY_PREFIX_LIMITS: pinned-date ids and variants that exact match
    #     misses resolve to the family's verified window instead of the 1M
    #     over-allocation.
    for _prefix, _limit in _FAMILY_PREFIX_LIMITS:
        if bare.startswith(_prefix):
            return _limit

    # 2. Fallback — 1M default. Warn ONCE per unknown model so a silent
    #    over-allocation surfaces early (known 1M models in _FALLBACK_IS_CORRECT
    #    stay quiet; the warning is for models nobody has classified yet).
    if bare not in _FALLBACK_IS_CORRECT and bare not in _warned_unknown_models:
        _warned_unknown_models.add(bare)
        logger.warning(
            "No context-window entry for model %r — using %d fallback. "
            "If its real window is smaller, add it to _CONTEXT_LIMITS or "
            "_FAMILY_PREFIX_LIMITS in context_budget.py.",
            model_name, _DEFAULT_CONTEXT_LIMIT,
        )
    return _DEFAULT_CONTEXT_LIMIT


def _structural_window_floor() -> int:
    """The smallest context window at which a prompt can still fit.

    Largest tool-schema token count across all variants + the max output
    reserve (4096) + ``MIN_USABLE_MESSAGE_BUDGET``. ``_record_context_overflow``
    must never reduce below this: below it ``context_message_cap`` collapses to
    its 512 floor and the prompt 400s forever — the 25% reduction steps can't
    help because the schemas alone exceed the window.
    """
    from ._shared_utils import MIN_USABLE_MESSAGE_BUDGET, estimate_tokens_from_tool_schemas
    from .tool_schemas import TOOL_SCHEMA_VARIANTS
    _max_tool_tokens = max(
        estimate_tokens_from_tool_schemas(s) for s in TOOL_SCHEMA_VARIANTS.values()
    )
    return _max_tool_tokens + 4096 + MIN_USABLE_MESSAGE_BUDGET


def _record_context_overflow(model: str, estimated_prompt_tokens: int | None = None, base_url: Optional[str] = None) -> None:
    """Record a context-length overflow for ``model``, reducing its effective limit.

    Called when a provider returns HTTP 400 with a "context length exceeded" or
    equivalent message.  Reduces the limit by 25% (floor: the structural minimum
    — see ``_structural_window_floor``) so subsequent calls pre-trim more
    aggressively.  Repeated overflows progressively reduce until the provider
    stops rejecting the prompt.

    When ``estimated_prompt_tokens`` is provided, the new limit is clamped below
    that value so a single overflow can converge in one shot instead of requiring
    multiple turns of progressive reduction.

    Args:
        base_url: The Ollama server the failing request actually used. Threaded
            through to ``_resolve_base_context_limit`` so the /api/show lookup
            hits the SAME (model, server) cache entry as the request path —
            otherwise a separate POST goes to the *default* server, whose
            num_ctx may differ from the server that returned the 400.

    Thread-safe: the base limit is computed outside ``_override_lock`` (avoiding
    blocking concurrent ``_resolve_context_limit`` callers during any Ollama HTTP
    round-trip), then the dict update uses the lock for RMW safety.
    """
    global _override_dirty
    model_lower = model.lower().strip()

    # P3: Compute base limit OUTSIDE the lock — _resolve_base_context_limit may
    # issue an Ollama HTTP request (blocking ~5s), and holding _override_lock
    # during I/O would stall all other callers of _resolve_context_limit.
    base_limit = _resolve_base_context_limit(model_lower, base_url)

    with _override_lock:
        meta = _override_meta.get(model_lower, {})

        # ── P5: TTL-aware reduction cap ─────────────────────────────────────
        # If the meta entry has expired, treat it as a fresh overflow (reset
        # reductions counter).  Prevents an expired entry with reductions=3 from
        # permanently blocking further overrides for a persistently misconfigured
        # model.
        _ts = meta.get("ts")
        _now = time.time()
        if _ts is not None and (_now - _ts) > _OVERRIDE_TTL_SECONDS:
            logger.info(
                "Override meta TTL expired for %s — resetting reductions counter",
                model,
            )
            meta = {}  # treat as fresh
            _override_meta.pop(model_lower, None)
            # Also clear the override so the next call uses base_limit, not stale cap.
            _context_window_overrides.pop(model_lower, None)

        reductions = meta.get("reductions", 0)
        if reductions >= _MAX_OVERRIDE_REDUCTIONS:
            logger.warning(
                "Context overflow for %s — reached max override reductions (%d), "
                "cannot reduce further. If this persists, add it to _CONTEXT_LIMITS.",
                model, _MAX_OVERRIDE_REDUCTIONS,
            )
            return

        # Use any existing override as the starting point for progressive reduction.
        current = _context_window_overrides.get(model_lower) or base_limit
        # Never reduce below the structural minimum (tool schemas + output
        # reserve + min usable message budget): below it context_message_cap
        # collapses to its 512 floor and no prompt can ever fit — 25% steps
        # can't help because the schemas alone exceed the window.
        _floor = _structural_window_floor()
        reduced = max(_floor, current * 3 // 4)
        if estimated_prompt_tokens is not None:
            # Proportional headroom: the 400 error proves the estimator
            # *underestimated* the real prompt, so a flat -512 is insufficient.
            # Use 85% of the estimated size so the override actually fits within the
            # real (unknown) window — fast 1-shot convergence for typical errors.
            reduced = min(reduced, max(_floor, int(estimated_prompt_tokens * 0.85)))
        _context_window_overrides[model_lower] = reduced
        _override_meta[model_lower] = {
            "ts": time.time(),
            "reductions": reductions + 1,
            "limit": reduced,
        }
        # Mark the cache dirty: THIS process has now diverged from the on-disk
        # baseline, so the next save / atexit flush is a legitimate write (and
        # the merge-on-write reconciles it with any concurrent writer's entry).
        _override_dirty = True

        # ── Logging ──────────────────────────────────────────────────────────
        if base_limit == _DEFAULT_CONTEXT_LIMIT:
            logger.warning(
                "Context overflow for %s (base limit = 1M fallback) — "
                "the model may have a smaller actual context window. "
                "Consider adding it to _CONTEXT_LIMITS in context_budget.py. "
                "Reducing override: %d→%d",
                model, current, reduced,
            )
        else:
            logger.warning(
                "Context overflow for %s: reducing limit %d→%d",
                model, current, reduced,
            )

        _save_override_cache()


def _is_context_length_error(exc: Exception) -> bool:
    """Detect context-length exceeded errors (HTTP 400) from provider messages.

    Providers return distinct error text for oversized prompts:
    - OpenAI:   "maximum context length is X tokens, but you sent Y"
    - DeepSeek: "context length exceeded", "too large"
    - GLM/ZAI:  code 1305 "context window is too small"
    - Anthropic: "prompt is too long"
    Detecting by pattern rather than HTTP status alone avoids mis-classifying
    unrelated 400s (malformed payload, invalid image format, etc.).

    Also checks provider-specific error codes on structured exceptions (e.g.
    ``LLMRateLimitError.error_code``) when the text is ambiguous, to serve as a
    backstop for errors where the numeric code arrives without the full text.
    (GLM code 1305 doubles as "context window too small" AND "server overloaded";
    only treat it as context-length when context-related terms are also present.)
    """
    msg = str(exc).lower()

    # Narrow, provider-specific patterns (low false-positive risk).
    _narrow_patterns = (
        "context length", "context window", "reduce length",
        "reduce the length", "maximum context",
        "prompt length",
        # "too small" is intentionally absent — it's too broad (matches
        # "temperature too small", "image too small", etc.) and the only
        # real GLM case ("context window is too small") is already caught
        # by "context window" above plus the 1305 error-code backstop.
    )
    if any(p in msg for p in _narrow_patterns):
        return True

    # Provider-specific error code on structured exceptions (backstop).
    # GLM code 1305 = "context window is too small" — but also "server overloaded".
    # Only accept when the message also mentions context-related terms.
    _error_code = getattr(exc, "error_code", None)
    if _error_code is not None:
        try:
            _code = int(_error_code)
        except (TypeError, ValueError):
            _code = None
        if _code == 1305:
            _context_terms = ("context", "window", "too small", "length")
            if any(t in msg for t in _context_terms):
                return True

    # "too long" / "too large" — only count when a context-related term is nearby,
    # to avoid misclassifying image/payload size errors ("image too large").
    if "too long" in msg or "too large" in msg:
        _context_terms = ("context", "token", "prompt", "message")
        return any(t in msg for t in _context_terms)

    return False


def _resolve_context_limit(model_name: str, base_url: Optional[str] = None) -> int:
    """Return the context window limit for a given model name.

    Priority:
        -1. Runtime overrides (from _record_context_overflow) — checked first.
            Overrides self-expire after _OVERRIDE_TTL_SECONDS.
         0. Dynamic query from Ollama /api/show (Option B) — if the model has an
            explicit ``num_ctx`` set in its Modelfile, use it.  Only triggers for
            Ollama-native model tags (colon-delimited, no path separator).
         1. Exact match in ``_CONTEXT_LIMITS`` — all known variants must be
            listed explicitly.
         1a. Family-prefix fallback in ``_FAMILY_PREFIX_LIMITS`` (pinned date
            ids like ``claude-opus-5-20260101`` and ``-fast``/-mini variants) —
            conservative under-allocation, never over-allocation.
         2. 1M fallback for unknown models.

    Thread-safe (uses ``_override_lock``).
    """
    model_lower = model_name.lower().strip()

    # -1. Runtime overrides (set by context-length 400 reactive backstop)
    _needs_flush = False
    with _override_lock:
        if model_lower in _context_window_overrides:
            meta = _override_meta.get(model_lower)
            if meta and (time.time() - meta.get("ts", 0)) > _OVERRIDE_TTL_SECONDS:
                # TTL expired — clear override and fall through to base limit.
                del _context_window_overrides[model_lower]
                _override_meta.pop(model_lower, None)
                logger.info(
                    "Override TTL expired for %s — cleared, using base limit",
                    model_lower,
                )
                _needs_flush = True
            else:
                return _context_window_overrides[model_lower]
    if _needs_flush:
        _save_override_cache()

    # 0-2. Delegate base resolution (Ollama API query / _CONTEXT_LIMITS / 1M
    #      fallback) — single source of truth shared with _record_context_overflow.
    #      ``base_url`` is forwarded so the Ollama /api/show query hits the SAME
    #      (model, server) cache entry providers.py uses, avoiding a redundant
    #      POST when the explicit base_url differs from OLLAMA_BASE_URL env.
    return _resolve_base_context_limit(model_lower, base_url)

class ContextBudgetManager:
    """Manages token budget for LLM context windows.

    Provides:
    - Fast token estimation (no external dependencies)
    - Pre-flight budget check (no truncation — truncated info forces LLM to
      re-fetch, costing more tokens than it saves)
    """

    def __init__(self, model_name: str, reserve_for_output: int=4096,
                 tool_schemas: Optional[list] = None):
        self.model_name = model_name
        _adaptive_max = max(512, self.context_limit // 5)
        self.reserve_for_output = min(reserve_for_output, _adaptive_max)
        # Tool-schema tokens are deducted from the budget to match
        # context_message_cap accounting (used by the actual pre-trim guard).
        self._tool_schema_tokens = 0
        if tool_schemas:
            from external_llm.agent._shared_utils import estimate_tokens_from_tool_schemas
            self._tool_schema_tokens = estimate_tokens_from_tool_schemas(tool_schemas)
        logger.info(
            'ContextBudgetManager: model=%s limit=%d budget=%d (reserve=%d, tool_schemas=%d) '
            '(no truncation — sliding window handles context management)',
            model_name, self.context_limit, self.total_budget,
            self.reserve_for_output, self._tool_schema_tokens,
        )

    @property
    def context_limit(self) -> int:
        """Live context limit — re-resolves on every access so runtime overrides
        from ``_record_context_overflow`` (and their TTL expiry) are reflected
        immediately instead of a construction-time snapshot.

        The guard paths (agent_loop pre-flight guard, ``_resolve_context_limit``)
        all use the live value; the old snapshot made the __init__ log and
        ``fit_messages`` warnings report a cap that could be several 25%-steps
        stale after a 400-driven override — misleading when debugging with logs.
        """
        return _resolve_context_limit(self.model_name)

    @property
    def total_budget(self) -> int:
        """Live budget = live context limit - output reserve - tool-schema tokens."""
        return self.context_limit - self.reserve_for_output - self._tool_schema_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """CJK-aware token estimate via canonical _cjk_aware_tokens."""
        from external_llm.agent._shared_utils import _cjk_aware_tokens as _cat
        if not text:
            return 0
        return _cat(text)

    def estimate_messages_tokens(self, messages: list) -> int:
        """Estimate total tokens via canonical shared estimator.

        Delegates to ``estimate_tokens_from_msgs`` (the single canonical token
        estimator for message content across the guard path) so all consumers
        use the same counting logic (CJK-aware content + tool_calls JSON +
        native tool_use/tool_result blocks + images).
        """
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        return estimate_tokens_from_msgs(messages)

    def fit_messages(self, messages: list,
                     tool_schemas: Optional[list] = None) -> list:
        """Check message budget — no truncation.

        Truncating tool results or messages (head+tail) causes the LLM to lose
        intermediate context and re-issue the same tool calls, wasting more
        tokens than the truncation saves.  Phase 1/2/3 cascade truncation was
        removed for this reason.

        SlidingWindowContext in context_manager.py handles context management
        by summarising older messages rather than silently dropping content.

        Args:
            messages: List of LLMMessage objects.
            tool_schemas: Optional tool schemas to account for in the budget.
                          When provided, uses ``context_message_cap`` logic
                          (matching the pre-trim guard) instead of the
                          construction-time budget.

        Returns the **original** message list (never a copy). Callers may mutate
        freely. The list may exceed budget — the API model's own context window
        handles overflow gracefully.
        """
        est = self.estimate_messages_tokens(messages)
        if tool_schemas:
            # Reuse pre-computed tool-schema tokens to avoid redundant json.dumps
            # (computed once in __init__ with the session's tool schemas).
            _tool_tokens = self._tool_schema_tokens
            if not _tool_tokens:
                from external_llm.agent._shared_utils import estimate_tokens_from_tool_schemas
                _tool_tokens = estimate_tokens_from_tool_schemas(tool_schemas)
            from external_llm.agent._shared_utils import context_message_cap
            _cap = context_message_cap(self.context_limit, self.reserve_for_output,
                                       tool_tokens=_tool_tokens)
        else:
            _cap = self.total_budget
        if est > _cap:
            logger.info(
                'fit_messages: estimated %d tokens > cap %d (not truncating — '
                'sliding window handles context management)',
                est, _cap,
            )
        return messages

def repair_tool_message_sequence(messages: list) -> list:
    """Remove orphaned tool messages and assistant messages missing their tool responses.

    Enforces the invariant (OpenAI/DeepSeek/Anthropic):
    - Every assistant message with tool_calls/tool_use blocks must be immediately
      followed by tool messages that respond to ALL of its call IDs.
    - Tool messages must directly follow such an assistant message.

    Any group that violates these rules is dropped entirely, preserving the
    integrity of the surrounding history.
    """
    result: list = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if is_tool_result(msg) and not is_tool_call(msg):
            # NOTE: tool messages that legitimately follow an assistant with tool_calls
            # are consumed by the assistant handler below (which collects ALL consecutive
            # tool responses). Any tool message arriving here is an orphan.
            # Anthropic-native user message may mix text + tool_result blocks: drop only
            # the tool_result blocks and keep any text (strategy warnings, user input)
            # so it isn't lost along with the orphan result.  Standard role="tool"
            # messages are still dropped whole (they carry only tool payload).
            if _is_anthropic_tool_result(msg):
                _rc = getattr(msg, "raw_content", None)
                if isinstance(_rc, list):
                    _text_blocks = [
                        b for b in _rc
                        if isinstance(b, dict) and b.get("type") != "tool_result"
                    ]
                    if _text_blocks:
                        logger.info('repair_tool_message_sequence: orphan tool_result at idx=%d had text blocks — preserving text', i)
                        result.append(dataclasses.replace(msg, raw_content=_text_blocks, content="", tool_call_id=None, name=None))
                        i += 1
                        continue
            logger.warning('repair_tool_message_sequence: dropping orphaned tool result at idx=%d', i)
            i += 1
            continue
        if is_tool_call(msg):
            j = i + 1
            while j < len(messages) and is_tool_result(messages[j]):
                j += 1
            tool_msgs = messages[i + 1:j]
            if not tool_msgs:
                logger.warning('repair_tool_message_sequence: dropping assistant(tool_call) with no following tool messages at idx=%d', i)
                i = j
                continue
            # Validate tool_call_id matching: every assistant tool_call id
            # must have a corresponding tool message, and vice versa.
            # Mismatches cause HTTP 400 from OpenAI/DeepSeek.
            # Only validate when both sides have concrete IDs — some
            # providers (e.g. Ollama) don't use tool_call_id on tool msgs.
            _tool_calls = getattr(msg, "tool_calls", None) or []
            _expected_ids = {tc.get("id") for tc in _tool_calls if isinstance(tc, dict) and tc.get("id")}
            # Also check anthropic tool_use blocks in raw_content
            _raw = getattr(msg, "raw_content", None)
            if isinstance(_raw, list):
                _expected_ids |= {
                    b.get("id") for b in _raw
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
                }
            # Collect actual tool result IDs from both standard and anthropic formats.
            _actual_ids = set()
            for m in tool_msgs:
                _tid = getattr(m, "tool_call_id", None)
                if _tid:
                    _actual_ids.add(_tid)
                # Anthropic-native: tool_use_id lives inside raw_content blocks
                _mrc = getattr(m, "raw_content", None)
                if isinstance(_mrc, list):
                    _actual_ids |= {
                        b.get("tool_use_id") for b in _mrc
                        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")
                    }
            _expected_valid = bool(_expected_ids)
            _actual_valid = bool(_actual_ids)
            if _expected_valid and _actual_valid and _expected_ids != _actual_ids:
                logger.warning(
                    'repair_tool_message_sequence: tool_call_id mismatch at idx=%d '
                    '(expected=%s, actual=%s) — dropping group',
                    i, _expected_ids, _actual_ids,
                )
                i = j
                continue
            result.append(msg)
            result.extend(tool_msgs)
            i = j
            continue
        result.append(msg)
        i += 1
    return result
