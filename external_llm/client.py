"""
External LLM client abstraction for asicode.

Supports multiple LLM providers:
- OpenAI (ChatGPT)
- Anthropic (Claude)
- Google (Gemini)
- DeepSeek

All clients return standardized response format for consistent processing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, NoReturn

import requests
from requests.adapters import HTTPAdapter

from external_llm.agent._response_utils import extract_llm_reasoning

logger = logging.getLogger(__name__)

# ── Timeout policy (single source of truth) ──────────────────────────────
# Reasoning models and long tool-result contexts frequently exceed the
# previous 120s ceiling, triggering spurious ReadTimeout retries. 180s gives
# headroom for long reasoning while still failing fast on truly dead servers.
DEFAULT_LLM_TIMEOUT = 180
# Local models (Ollama) need model-loading + warmup time → much larger budget.
OLLAMA_LLM_TIMEOUT = 600


@dataclass
class LLMMessage:
    """Standard message format for all LLM providers"""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    # Optional fields for tool-calling (OpenAI-compatible providers)
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # Provider-native content blocks (Anthropic content[], Gemini parts[]).
    # When set, providers use this directly instead of the plain `content` string.
    raw_content: list[dict[str, Any]] | None = None
    # DeepSeek Reasoner: chain-of-thought content that must be echoed back in multi-turn
    reasoning_content: str | None = None
    # Attached images (provider-agnostic). Each item: {"media_type": "image/png", "data": "<base64>"}
    # Each provider client converts these to its native format (Anthropic content blocks, Gemini inlineData).
    #
    # IN-MEMORY ONLY — these dicts must never be serialized to disk.  Two things
    # depend on that: ``providers._images_to_text`` caches its OCR output back
    # into each dict as ``ocr_text`` (so the token estimator can count it, see
    # ``_shared_utils._images_ocr_len``), and ``data`` already holds a full
    # base64 payload.  Persisting these would write both to disk on every turn.
    # If a persistence path is ever added, move the OCR cache out FIRST and make
    # ``_msg_token_fingerprint`` track wherever it moved to — otherwise the
    # estimator silently returns a stale pre-OCR under-count.
    images: list[dict[str, str]] | None = None


@dataclass
class LLMResponse:
    """
    Standardized LLM response across all providers

    Attributes:
        content: Raw response text from LLM
        model: Model used
        provider: Provider name (openai, anthropic, google, deepseek)
        tokens_used: Total tokens used (prompt + completion)
        finish_reason: Why generation stopped
        raw_response: Original API response for debugging
    """

    content: str
    model: str
    provider: str
    tokens_used: int | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    # Separated token counts + prompt-cache fields. Populated by providers that
    # expose usage detail on the plain chat() path too (e.g. DeepSeek), so that
    # non-tool callers (the planner) can account cache savings. ToolCallResponse
    # redeclares these for backward compat; the defaults match.
    prompt_tokens: int | None = None  # input tokens
    completion_tokens: int | None = None  # output tokens
    cache_read_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class ToolCallRequest:
    """Represents a single tool call requested by the LLM."""

    call_id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolCallResponse(LLMResponse):
    """
    LLM response that may include tool calls.

    Extends LLMResponse with tool_calls list.
    is_final=True means the LLM gave a final answer (no more tool calls needed).
    """

    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    is_final: bool = False
    # Separated token counts (input/output) for cost estimation.
    # tokens_used (from LLMResponse) = prompt_tokens + completion_tokens.
    prompt_tokens: int | None = None  # input tokens
    completion_tokens: int | None = None  # output tokens
    # Prompt caching fields (populated by providers that support them, e.g. Anthropic).
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    # Reasoning tokens (populated by DeepSeek — completion_tokens = reasoning + visible).
    reasoning_tokens: int | None = None


def effective_content(response) -> str:
    """Return the user-facing content of an LLM response, falling back to ``reasoning_content``.

    GLM-5.2 (thinking ON) / DeepSeek Reasoner intermittently emit the final
    answer in ``reasoning_content`` while leaving ``content`` empty. Without
    this fallback, every subsystem that reads ``response.content`` as a final or
    user-facing message silently swallows the result — summaries are not
    updated (turns get archived with no verbatim path back), intent resolutions
    collapse to the heuristic fallback, and closing answers vanish.

    Single canonical extractor for the LLMResponse shape (what
    ``llm_client.chat()`` returns). Mirrors the inline fallback already on
    EVERY termination path of DesignChatLoop and AgentTurnPipeline
    (multi-path fallback parity principle — insight 2026-07-05).
    """
    content = getattr(response, "content", "") or ""
    if isinstance(content, str) and content.strip():
        return content
    raw_resp = getattr(response, "raw_response", None)
    if isinstance(raw_resp, dict):
        with suppress(AttributeError, TypeError, IndexError):
            rc = extract_llm_reasoning(raw_resp, strip=True)
            if rc:
                return rc
    return content if isinstance(content, str) else ""


# Upper bound on a single Retry-After wait (seconds). Guards against absurdly
# large server values (e.g. far-future HTTP-dates) that would stall the agent.
RETRY_AFTER_MAX_WAIT = 60


def parse_retry_after(headers: Any) -> int | None:
    """Parse a ``Retry-After`` header value into seconds, or ``None``.

    Accepts any mapping with ``.get`` (e.g. ``requests.Response.headers``).
    Handles both integer-seconds and HTTP-date formats, clamped to
    ``[1, RETRY_AFTER_MAX_WAIT]``. Returns ``None`` when the header is missing,
    unparseable, or already in the past.
    """
    if headers is None:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    with suppress(ValueError):
        return min(RETRY_AFTER_MAX_WAIT, max(1, int(str(raw).strip())))
    with suppress(ValueError, TypeError, OSError):
        import time as _time
        from datetime import timezone
        from email.utils import parsedate_to_datetime

        retry_time = parsedate_to_datetime(str(raw).strip())
        if retry_time.tzinfo is None:
            # RFC 7231 HTTP-date is GMT; a timezone-less parse must not be read
            # as LOCAL time (KST +9h would make wait<=0 and silently drop the
            # server's Retry-After hint).
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        wait = int(retry_time.timestamp() - _time.time())
        if wait <= 0:
            return None
        return min(RETRY_AFTER_MAX_WAIT, max(1, wait))
    return None


# zai/GLM (and similar providers) report an EXHAUSTED ACCOUNT BALANCE / quota
# as HTTP 429 — a status the shared retry loop normally treats as a transient,
# retryable rate limit. But a depleted balance never recovers within a retry
# window: retrying only wastes time, and surfacing it as rate-limit (or, worse,
# auth) misleads the user into waiting or re-entering a perfectly valid key.
# These unambiguous billing signals mark the response as a hard quota/balance
# failure that must raise LLMQuotaExceededError, not retry. Single canonical
# source — consumed by BOTH the OpenAI-compatible client (ZAIClient) and the
# Anthropic-compatible client (ZAIAnthropicClient, the primary zai endpoint).
_BALANCE_QUOTA_CODES: frozenset[int] = frozenset({1113})  # zai "insufficient balance"
_BALANCE_QUOTA_PHRASES: tuple[str, ...] = (
    "insufficient balance",
    "no resource package",
    "please recharge",
    "out of credit",
    "payment required",
)


def is_balance_quota_signal(error_code: int | None, body_text: str = "") -> bool:
    """Return True when a 429 response is actually an exhausted balance/quota.

    Checks both the provider error ``code`` (e.g. zai 1113) and the lowercased
    response body for billing-specific phrases. Conservative — only matches
    unambiguous billing phrases, so genuine transient "rate quota" messages
    (GLM 1305 server overload, 1302 rate limit) stay rate-limit errors.

    Callers pass the already-parsed error code (each client extracts it from
    its own JSON envelope) plus the raw body text for the phrase fallback.
    """
    if error_code is not None and error_code in _BALANCE_QUOTA_CODES:
        return True
    return any(_p in (body_text or "").lower() for _p in _BALANCE_QUOTA_PHRASES)


# Per-SSE-line cap for the shared stream parser below.  A single `data:` line
# from a well-behaved LLM provider is a few KB (a handful of tokens per delta
# frame).  An unbounded line — from a broken or hostile upstream — buffers
# forever in iter_lines()/our splitter and then spikes CPU in json.loads
# (P28-1): the file-read bounds closed in P19-P25 had no network-read
# counterpart.  4 MiB is generous for any legitimate frame while keeping the
# worst-case buffer bounded.
_SSE_MAX_LINE_BYTES = 4 * 1024 * 1024


def _parse_sse_line(line: bytes) -> dict[str, Any] | None:
    """Parse one physical SSE line (no trailing ``\\n``) into an event dict.

    Returns ``None`` for keep-alive/blank lines, non-``data:`` frames
    (``event:``/``id:``/``retry:``), empty data, the ``[DONE]`` sentinel, and
    malformed JSON (logged at debug level).  Kept as a separate helper so the
    chunk-splitting loop stays a pure framing concern.
    """
    if line.endswith(b"\r"):
        line = line[:-1]
    if not line:
        return None
    try:
        line_str = line.decode("utf-8")
    except UnicodeDecodeError:
        # B2: a line that never formed valid UTF-8 (EOF cut mid-character,
        # proxy error pages in another encoding) must NOT raise out of the
        # framing loop — decode lossy and let the JSON check below decide.
        line_str = line.decode("utf-8", errors="replace")
        logger.debug("SSE line contained undecodable bytes (replaced): %.120s", line_str)
    if not line_str.startswith("data:"):
        return None
    data_str = line_str[5:].strip()
    if not data_str or data_str == "[DONE]":
        return None
    try:
        return json.loads(data_str)
    except Exception:
        logger.debug("Skipping malformed SSE data frame: %.120s", data_str)
        return None


def _iter_response_chunks(response: Any) -> Iterator[bytes]:
    """Yield raw byte chunks from a streaming HTTP response, transport-agnostic.

    Every LLM client builds its HTTP layer with ``requests.Session``
    (``LLMClient.__init__``), so the real streaming responses are
    ``requests.Response`` objects — whose byte-chunk API is ``iter_content()``,
    NOT ``iter_bytes()`` (that is httpx's).  P28-1 called ``iter_bytes()``
    unconditionally, which raised AttributeError on every production stream
    while the ``iter_bytes``-shaped test fakes stayed green.  Prefer
    ``iter_bytes`` when present (httpx-shaped transports/fakes) and fall back
    to ``iter_content`` (requests).

    ``iter_content(chunk_size=512)`` mirrors the historical ``iter_lines``
    transport behavior (requests' own ``ITER_CHUNK_SIZE``); ``chunk_size=None``
    would block until EOF on the urllib3 path, so it is deliberately NOT used.
    """
    if hasattr(response, "iter_bytes"):
        yield from response.iter_bytes()
        return
    yield from response.iter_content(chunk_size=512, decode_unicode=False)


def iter_sse_data_events(response: Any) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON events from an SSE ``data:`` stream.

    Shared SSE framing used by the OpenAI-, DeepSeek-, Gemini- and
    Anthropic-compatible clients: consumes raw byte chunks via
    ``_iter_response_chunks`` (requests ``iter_content`` / httpx ``iter_bytes``)
    and splits chunks on ``\n`` itself (``iter_lines()`` buffers a whole line
    before yielding — unbounded memory on a line that never terminates).
    A line exceeding ``_SSE_MAX_LINE_BYTES`` aborts the stream with a warning
    instead of buffering it.  Skips blank keep-alive lines, decodes bytes
    lines, ignores non-``data:`` frames and the ``[DONE]`` sentinel, and
    skips malformed JSON events (logged at debug level).  The caller retains
    ownership of ``response`` (including ``close()``); status handling and
    event-type dispatch stay in the per-client loop.

    Framing is linear in the stream size: a ``scanned`` watermark records how
    far ``find`` has already searched, so a long newline-free stretch (or a
    huge single chunk of small lines) is never re-scanned, and the consumed
    prefix is compacted once per chunk — the naive per-line re-slice
    (``buf = buf[idx + 1:]``) copied the whole remainder for every line
    (O(n^2) on a large single chunk).
    """
    buf = bytearray()
    line_start = 0  # start of the current (partial) line within buf
    scanned = 0  # watermark: buf[:scanned] contains no unprocessed ``\n``
    for chunk in _iter_response_chunks(response):
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            idx = buf.find(b"\n", scanned)
            if idx < 0:
                scanned = len(buf)  # everything scanned; no newline yet
                break
            line = bytes(buf[line_start:idx])
            line_start = idx + 1
            scanned = idx + 1
            if len(line) > _SSE_MAX_LINE_BYTES:
                logger.warning(
                    "SSE stream line exceeds %d bytes (%d) — aborting stream (runaway/oversized frame)",
                    _SSE_MAX_LINE_BYTES,
                    len(line),
                )
                return
            event = _parse_sse_line(line)
            if event is not None:
                yield event
        if line_start:
            # Consumed lines are compacted once per chunk (an offset scan
            # avoids re-copying the whole remainder for every line).
            del buf[:line_start]
            scanned -= line_start
            line_start = 0
        # No newline in the remainder: it is a (partial) line.  If it already
        # exceeds the cap there is no legitimate completion — abort now
        # instead of buffering the rest of it.
        if len(buf) > _SSE_MAX_LINE_BYTES:
            logger.warning(
                "SSE stream line exceeds %d bytes (%d) — aborting stream (runaway/oversized frame)",
                _SSE_MAX_LINE_BYTES,
                len(buf),
            )
            return
    # Tail after EOF without a trailing newline (iter_lines() also yields it).
    if line_start < len(buf):
        event = _parse_sse_line(bytes(buf[line_start:]))
        if event is not None:
            yield event


def raise_sse_iteration_failure(exc: Exception) -> NoReturn:
    """Convert an unexpected stream failure into a typed ``LLMAPIError``.

    Shared by ``guard_sse_iteration`` and the per-loop ``except Exception``
    clause at every provider call site, so the failure surface is uniform:
    a turn sees ``LLMAPIError`` (diagnosable via ``logger.exception``),
    never a raw exception escaping the streaming layer.
    """
    logger.exception("SSE stream iteration failed: %s", exc)
    raise LLMAPIError(f"SSE stream iteration failed: {exc}") from exc


def guard_sse_iteration(events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Defensive wrapper around a parsed SSE event stream.

    ``iter_sse_data_events`` never raises for malformed input, but a
    transport quirk (e.g. a non-requests failure inside
    ``_iter_response_chunks``) or a framing bug must not escape the
    consuming loop as a raw exception and kill the whole turn.  Typed
    ``LLMClientError`` s and ``requests`` exceptions keep their semantics —
    call sites map those to retryable/fatal outcomes — while anything else
    becomes an ``LLMAPIError`` with full diagnostics.

    Note: exceptions raised by the *consumer loop body* never enter this
    generator (Python semantics — a ``raise`` in the ``for`` body
    propagates up the consumer's stack, not into the suspended generator),
    so every call site pairs this wrapper with an ``except Exception``
    clause that funnels body failures through ``raise_sse_iteration_failure``
    as well.  Wrap the iterator at the call site::

        for ev in guard_sse_iteration(iter_sse_data_events(response)):
            ...
    """
    try:
        yield from events
    except (LLMClientError, requests.RequestException):
        raise
    except Exception as e:
        raise_sse_iteration_failure(e)


class LLMClientError(Exception):
    """Base exception for LLM client errors"""


class LLMConnectionError(LLMClientError):
    """Cannot connect to LLM API"""


class LLMAuthenticationError(LLMClientError):
    """Invalid API key or authentication failed"""


class LLMRateLimitError(LLMClientError):
    """Rate limit exceeded.

    ``retry_after`` carries the server's suggested wait (seconds, parsed from
    the ``Retry-After`` header) when available, so the retry loop can honor it
    instead of a fixed backoff. ``None`` means the server gave no hint.
    """

    def __init__(self, *args: object, retry_after: int | None = None, error_code: int | None = None) -> None:
        super().__init__(*args)
        # Clamp at construction so every consumer sees a bounded hint. The
        # header parser (parse_retry_after) already clamps to
        # RETRY_AFTER_MAX_WAIT; this guards direct constructions with absurd
        # values (e.g. retry_after=3600) that would stall a retry loop for an
        # hour. Floats are normalized to int (truncated); values below 1 clamp
        # to 1 (a present-but-tiny hint still means "wait a moment").
        # Non-numeric hints are left as-is; consumers reject them via isinstance.
        if isinstance(retry_after, (int, float)):
            retry_after = min(RETRY_AFTER_MAX_WAIT, max(1, int(retry_after)))
        self.retry_after = retry_after
        self.error_code = error_code


class LLMQuotaExceededError(LLMClientError):
    """API key has insufficient credits / quota exceeded (HTTP 402)"""


class LLMAPIError(LLMClientError):
    """API returned error response"""


class ContextWindowCollapseError(LLMAPIError):
    """Local pre-flight detection: the model's context window is too small for
    the serialised tool schemas + output reserve, so no message trimming can
    save the call.

    Raised locally (before the request) instead of letting the provider return
    an inevitable 400 "context length exceeded" — the retry path would shrink
    messages toward zero, never fit, and burn three attempts per turn.  Carries
    the structural cause (small window / oversized toolset) so the user-facing
    error mapper can show an actionable message.
    """


class LLMServerUnavailableError(LLMClientError):
    """Server is unavailable (503, timeout, connection failure) — abort, do not fall back."""


class LLMCancelled(LLMClientError):  # noqa: N818 — Cancelled-suffix convention (AgentCancelled parity)
    """A retry backoff wait was interrupted by the caller's cancel_event.

    Raised from client-layer retry sleeps (``_request_with_retry``) when the
    attached ``cancel_event`` is set mid-wait, so ESC / orchestrator
    cancellation stays responsive even while the client is inside its own
    internal retry backoff (up to ~36s of sleeps otherwise).  Agent loops
    convert this to their own ``AgentCancelled`` at the loop boundary; other
    callers must handle it like any other LLMClientError.
    """


def interruptible_sleep(seconds: float, cancel_event: Any | None = None) -> bool:
    """Sleep *seconds*, waking early (returning True) when *cancel_event* is set.

    Returns True when the wait was interrupted by cancellation — the caller
    should then abort the retry rather than proceed.  With no *cancel_event*
    (or one already set), behaves like ``time.sleep`` (returning False /
    True immediately, respectively).
    """
    if cancel_event is None:
        time.sleep(seconds)
        return False
    return bool(cancel_event.wait(timeout=seconds))


class LLMClient(ABC):
    """
    Abstract base class for LLM clients

    All external LLM providers must implement this interface
    """

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int = DEFAULT_LLM_TIMEOUT):
        """
        Initialize LLM client

        Args:
            api_key: API key for authentication
            base_url: Custom API endpoint (optional)
            timeout: Request timeout in seconds (DEFAULT_LLM_TIMEOUT by default)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        # Optional cancellation signal for retry backoff (see
        # interruptible_sleep).  Wired by agent loops (agent_loop /
        # design_chat_loop) from their config.cancel_event so ESC stays
        # responsive during client-internal retries; None = plain sleeps.
        self.cancel_event: Any | None = None
        # HTTP connection pooling: reuse TCP/TLS handshake to save 50-200ms per call
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat completion request to LLM

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with standardized format

        Raises:
            LLMConnectionError: Cannot connect to API
            LLMAuthenticationError: Invalid API key
            LLMRateLimitError: Rate limit exceeded
            LLMAPIError: Other API errors
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic')"""

    def chat_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]], model: str = "", **kwargs
    ) -> ToolCallResponse:
        """
        Chat with tool calling support.

        Default implementation calls chat() without tools and returns empty tool_calls.
        Override in providers that support native tool calling.

        Args:
            messages: Conversation messages
            tools: Tool schemas in OpenAI function-calling format
            model: Model identifier
            **kwargs: Provider-specific parameters

        Returns:
            ToolCallResponse with tool_calls list (may be empty for final answer)
        """
        response = self.chat(messages, model=model, **kwargs)
        return ToolCallResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            finish_reason=response.finish_reason,
            raw_response=response.raw_response,
            tool_calls=[],
            is_final=True,
        )

    def close(self) -> None:
        """Close the HTTP session, releasing connection pool resources."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
            logger.debug("LLMClient session closed for %s", self.get_provider_name())


def resolve_provider_base_url(provider: str) -> str | None:
    """Resolve the base URL override for ``provider`` in a provider-scoped way.

    Resolution order:

    1. Per-provider override: ``{PROVIDER}_BASE_URL`` (e.g. ``ZAI_BASE_URL``,
       ``OPENAI_BASE_URL``). Unambiguous — applies only to the named provider.
    2. The global ``EXTERNAL_LLM_BASE_URL`` — but ONLY when ``provider`` matches
       the globally-configured provider (``EXTERNAL_LLM_PROVIDER``). A global
       base_url belongs to one specific host; when a service is created for a
       DIFFERENT provider (e.g. a per-terminal ``/model`` switch to zai while
       the ``.env`` default is opencode), the foreign base_url must not leak in.
       It would point the client at the wrong host (the zai key sent to the
       opencode endpoint → HTTP 401 → misleading "Invalid API key" prompt) AND
       disable zai's auth/connection endpoint-failover, which treats a set
       base_url as a custom endpoint with no known sibling.
    3. ``None`` — the client falls back to its provider-specific DEFAULT_BASE_URL.

    This is the single canonical resolver used by every client-creation path so
    that provider switching never inherits a foreign host's base_url.
    """
    prov = (provider or "").strip().lower()
    if not prov:
        return None
    prov_override = (os.getenv(f"{prov.upper()}_BASE_URL", "") or "").strip() or None
    if prov_override:
        return prov_override
    global_prov = (os.getenv("EXTERNAL_LLM_PROVIDER", "") or "").strip().lower()
    if prov == global_prov:
        return (os.getenv("EXTERNAL_LLM_BASE_URL", "") or "").strip() or None
    return None


def create_llm_client(
    provider: str,
    api_key: str,
    base_url: str | None = None,
    timeout: int = DEFAULT_LLM_TIMEOUT,
) -> LLMClient:
    """
    Factory function to create LLM client

    Args:
        provider: Provider name (openai, anthropic, google, deepseek)
        api_key: API key
        base_url: Custom API endpoint (optional)
        timeout: Request timeout (DEFAULT_LLM_TIMEOUT by default)

    Returns:
        LLMClient instance

    Raises:
        ValueError: Unknown provider

    Example:
        >>> client = create_llm_client("openai", api_key="sk-...")
        >>> response = client.chat([LLMMessage(role="user", content="Fix this code")], model="gpt-4")
    """
    provider_lower = provider.lower()

    # Special handling for Ollama: use longer default timeout for model loading.
    # Compare against the cloud default (DEFAULT_LLM_TIMEOUT), not the magic
    # number 120, so explicit per-call overrides are still respected.
    if provider_lower == "ollama" and timeout == DEFAULT_LLM_TIMEOUT:
        timeout = OLLAMA_LLM_TIMEOUT
        logger.debug("Using extended timeout for Ollama: %ss", timeout)

    if provider_lower == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(api_key, base_url, timeout)

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(api_key, base_url, timeout)

    if provider_lower == "google":
        from .providers import GoogleClient

        return GoogleClient(api_key, base_url, timeout)

    if provider_lower == "deepseek":
        from .providers import DeepSeekClient

        return DeepSeekClient(api_key, base_url, timeout)

    if provider_lower == "ollama":
        from .providers import OllamaClient

        return OllamaClient(api_key, base_url, timeout)

    if provider_lower in ("zai",):
        from .anthropic_client import ZAIAnthropicClient

        return ZAIAnthropicClient(api_key, base_url, timeout)

    if provider_lower == "openrouter":
        from .openai_client import OpenRouterClient

        return OpenRouterClient(api_key, base_url, timeout)

    if provider_lower == "opencode":
        from .openai_client import OpenCodeClient

        if not base_url:
            base_url = "https://opencode.ai/zen/go/v1"  # default for OpenCode Go
        return OpenCodeClient(api_key, base_url, timeout)

    raise ValueError(
        f"Unknown LLM provider: {provider}. "
        f"Supported: openai, anthropic, google, deepseek, ollama, zai, openrouter, opencode"
    )
