"""Context Budget Manager – token-aware message fitting (budget check only, no truncation)."""
from __future__ import annotations

import atexit
import json
import logging
import dataclasses
import os
import threading
import time
from typing import TYPE_CHECKING

from external_llm.agent.message_shapes import is_tool_result, is_tool_call, _is_anthropic_tool_result
from typing import Optional  # f821-protected

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
    "o3-mini":          200_000,
    "o3-mini-high":     200_000,
    "o4-mini":          200_000,
    "o4-mini-high":     200_000,
    # Anthropic — all modern Claude models (Sonnet/Opus/Haiku generations 3–5)
    # share a 200K context window. Listed explicitly (no prefix matching) per the
    # table's design; new variants must be added here or they fall back to 1M.
    "claude-haiku-4-5":          200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-6":         200_000,
    "claude-sonnet-4-5":         200_000,
    "claude-sonnet-5":           200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-sonnet":           200_000,
    "claude-opus-4-8":           200_000,
    "claude-opus-4-7":           200_000,
    # DeepSeek — original deepseek-r1 (64K context); deepseek-chat/reasoner are
    # deprecated aliases for deepseek-v4-flash thinking/non-thinking → 1M fallback.
    "deepseek-r1":       64_000,
    # Zhipu GLM (zai + opencode). glm-5.2 is the DEFAULT_MODEL and 1M is verified
    # (Z.ai docs: "Context Length: 1M" — generational leap from the 200K family).
    # Listed EXPLICITLY (not via _DEFAULT fallback) so the default model's window
    # cannot silently drift if _DEFAULT_CONTEXT_LIMIT changes. glm-5/5.1/5-turbo 200K; glm-4.7 128K.
    "glm-5.2":          1_000_000,
    "glm-5.1":          200_000,
    "glm-5-turbo":      200_000,
    "glm-5":            200_000,
    "glm-4.7":          128_000,
    # Qwen3 (opencode provider) — 3.7-max/plus, 3.6-plus, 3.5-plus are 1M (fallback)
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
    # hy3-preview (opencode provider) — unverified, keep conservative 128K
    "hy3-preview":      128_000,
}


# Default context limit (fallback for unknown models).
_DEFAULT_CONTEXT_LIMIT = 1_000_000

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
_OVERRIDE_TTL_SECONDS = int(os.getenv("CONTEXT_OVERRIDE_TTL", "1800"))      # 30 minutes
_MAX_OVERRIDE_REDUCTIONS = int(os.getenv("CONTEXT_MAX_REDUCTIONS", "3"))   # max step-downs per model

# Model → {ts: float, reductions: int, limit: int}
_override_meta: dict[str, dict] = {}

# ── On-disk cache ──────────────────────────────────────────────────────────────
# Persists overrides across restarts so a misconfigured 1M-fallback model doesn't
# hard-fail on every fresh process.  Best-effort: corrupted/missing file ignored.
_OVERRIDE_CACHE_FILE = os.path.join(
    os.path.expanduser("~"), ".cache", "asicode", "context_override_cache.json",
)
_last_cache_save: float = 0
_CACHE_SAVE_INTERVAL = 5.0  # seconds between disk writes (debounce)


def _ensure_override_cache_loaded() -> None:
    """Load persisted overrides from disk (best-effort, called once at module init).

    Each entry is processed in its own try/except so a single corrupt entry
    (missing ``"limit"``, non-dict value, etc.) does **not** discard all
    remaining valid entries beyond it.
    """
    try:
        if not os.path.exists(_OVERRIDE_CACHE_FILE):
            return
        with open(_OVERRIDE_CACHE_FILE) as _f:
            _data = json.load(_f)
        _now = time.time()
        for _model, _entry in _data.items():
            try:
                if not isinstance(_entry, dict) or "limit" not in _entry:
                    continue
                _ts = _entry.get("ts", 0)
                if _now - _ts < _OVERRIDE_TTL_SECONDS:
                    _context_window_overrides[_model] = _entry["limit"]
                    _override_meta[_model] = _entry
            except Exception:
                continue  # skip corrupt entry, keep processing rest
    except Exception:
        pass  # best-effort (file-level: missing, corrupt JSON, IO error)


def _save_override_cache(force: bool = False) -> None:
    """Persist current overrides to disk (debounced, best-effort, atomic writes).

    Writes to a temp file then atomically renames to the target path so a
    concurrent reader or writer never sees a partial/corrupted JSON file.
    The temp file uses a UUID suffix so concurrent writers (e.g. atexit flush
    vs. worker thread) do not collide on the same path.

    Args:
        force: When True, skip the debounce interval check and write immediately.
               Used by ``atexit`` flush to prevent losing the last override write.
    """
    global _last_cache_save
    _now = time.time()
    if not force and _now - _last_cache_save < _CACHE_SAVE_INTERVAL:
        return
    _tmp = None
    try:
        os.makedirs(os.path.dirname(_OVERRIDE_CACHE_FILE), exist_ok=True)
        _tmp = _OVERRIDE_CACHE_FILE + ".tmp." + str(os.getpid()) + "." + str(time.time_ns())
        # Snapshot + debounce update under lock — prevents 'dictionary changed size
        # during iteration' when a concurrent writer mutates _override_meta while
        # the atexit flush handler serializes it.  Debounce update inside the lock
        # prevents a race window where two threads pass the debounce check before
        # either writes, then both write.
        with _override_lock:
            _snapshot = dict(_override_meta)
            _last_cache_save = time.time()
        with open(_tmp, "w", encoding="utf-8") as _f:
            json.dump(_snapshot, _f, ensure_ascii=False)
        os.replace(_tmp, _OVERRIDE_CACHE_FILE)  # atomic on POSIX & Windows
    except Exception:
        # P5: Clean up tmp file on failure (best-effort) to prevent file leaks.
        try:
            if _tmp and os.path.exists(_tmp):
                os.unlink(_tmp)
        except Exception:
            pass


# Load persisted overrides at module init.
_ensure_override_cache_loaded()

# Flush on-disk cache at process exit so the last override write is not lost
# when the debounce interval hasn't elapsed.  Registered after load so a
# load-time crash does not clobber an existing cache file.
atexit.register(lambda: _save_override_cache(force=True))


def _resolve_base_context_limit(model_name: str) -> int:
    """Compute the configured context limit WITHOUT runtime overrides.

    Like ``_resolve_context_limit`` but skips ``_context_window_overrides`` so
    ``_record_context_overflow`` can compute the base value before reducing it.
    """
    model_lower = model_name.lower().strip()

    # 0. Dynamic query from Ollama API (Option B) — for native Ollama format only
    if ":" in model_lower and "/" not in model_lower:
        from external_llm.ollama_api import query_ollama_num_ctx
        api_ctx = query_ollama_num_ctx(model_lower)
        if api_ctx is not None:
            return api_ctx

    # 1. Exact match in _CONTEXT_LIMITS
    if model_lower in _CONTEXT_LIMITS:
        return _CONTEXT_LIMITS[model_lower]

    # 2. Fallback
    return _DEFAULT_CONTEXT_LIMIT


def _record_context_overflow(model: str, estimated_prompt_tokens: int | None = None) -> None:
    """Record a context-length overflow for ``model``, reducing its effective limit.

    Called when a provider returns HTTP 400 with a "context length exceeded" or
    equivalent message.  Reduces the limit by 25% (floor 8K) so subsequent calls
    pre-trim more aggressively.  Repeated overflows progressively reduce until
    the provider stops rejecting the prompt.

    When ``estimated_prompt_tokens`` is provided, the new limit is clamped below
    that value so a single overflow can converge in one shot instead of requiring
    multiple turns of progressive reduction.

    Thread-safe: the base limit is computed outside ``_override_lock`` (avoiding
    blocking concurrent ``_resolve_context_limit`` callers during any Ollama HTTP
    round-trip), then the dict update uses the lock for RMW safety.
    """
    model_lower = model.lower().strip()

    # P3: Compute base limit OUTSIDE the lock — _resolve_base_context_limit may
    # issue an Ollama HTTP request (blocking ~5s), and holding _override_lock
    # during I/O would stall all other callers of _resolve_context_limit.
    base_limit = _resolve_base_context_limit(model_lower)

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
        reduced = max(8192, current * 3 // 4)
        if estimated_prompt_tokens is not None:
            # Proportional headroom: the 400 error proves the estimator
            # *underestimated* the real prompt, so a flat -512 is insufficient.
            # Use 85% of the estimated size so the override actually fits within the
            # real (unknown) window — fast 1-shot convergence for typical errors.
            reduced = min(reduced, max(8192, int(estimated_prompt_tokens * 0.85)))
        _context_window_overrides[model_lower] = reduced
        _override_meta[model_lower] = {
            "ts": time.time(),
            "reductions": reductions + 1,
            "limit": reduced,
        }

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


def _resolve_context_limit(model_name: str) -> int:
    """Return the context window limit for a given model name.

    Priority:
        -1. Runtime overrides (from _record_context_overflow) — checked first.
            Overrides self-expire after _OVERRIDE_TTL_SECONDS.
         0. Dynamic query from Ollama /api/show (Option B) — if the model has an
            explicit ``num_ctx`` set in its Modelfile, use it.  Only triggers for
            Ollama-native model tags (colon-delimited, no path separator).
         1. Exact match in ``_CONTEXT_LIMITS`` — all known variants must be listed
            explicitly; no prefix matching.
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

    # 0. Dynamic query from Ollama API (Option B) — for native Ollama format only
    if ":" in model_lower and "/" not in model_lower:
        from external_llm.ollama_api import query_ollama_num_ctx
        api_ctx = query_ollama_num_ctx(model_lower)
        if api_ctx is not None:
            logger.debug("num_ctx=%d from Ollama API for model %s", api_ctx, model_lower)
            return api_ctx

    # 1. Exact match in _CONTEXT_LIMITS (no prefix matching — every variant must
    #    be listed explicitly to avoid silent wrong answers for future models).
    if model_lower in _CONTEXT_LIMITS:
        return _CONTEXT_LIMITS[model_lower]

    # 2. Fallback — preemptive trim with 1M is advisory; the API enforces the real limit.
    return _DEFAULT_CONTEXT_LIMIT

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
        self.context_limit = _resolve_context_limit(model_name)
        _adaptive_max = max(512, self.context_limit // 5)
        self.reserve_for_output = min(reserve_for_output, _adaptive_max)
        # Tool-schema tokens are deducted from the budget to match
        # context_message_cap accounting (used by the actual pre-trim guard).
        self._tool_schema_tokens = 0
        if tool_schemas:
            from external_llm.agent._shared_utils import estimate_tokens_from_tool_schemas
            self._tool_schema_tokens = estimate_tokens_from_tool_schemas(tool_schemas)
        self.total_budget = self.context_limit - self.reserve_for_output - self._tool_schema_tokens
        logger.info(
            'ContextBudgetManager: model=%s limit=%d budget=%d (reserve=%d, tool_schemas=%d) '
            '(no truncation — sliding window handles context management)',
            model_name, self.context_limit, self.total_budget,
            self.reserve_for_output, self._tool_schema_tokens,
        )

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
