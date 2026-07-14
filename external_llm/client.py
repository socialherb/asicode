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

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

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
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    # Provider-native content blocks (Anthropic content[], Gemini parts[]).
    # When set, providers use this directly instead of the plain `content` string.
    raw_content: Optional[list[dict[str, Any]]] = None
    # DeepSeek Reasoner: chain-of-thought content that must be echoed back in multi-turn
    reasoning_content: Optional[str] = None
    # Attached images (provider-agnostic). Each item: {"media_type": "image/png", "data": "<base64>"}
    # Each provider client converts these to its native format (Anthropic content blocks, Gemini inlineData).
    images: Optional[list[dict[str, str]]] = None


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
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[dict[str, Any]] = None
    # Separated token counts + prompt-cache fields. Populated by providers that
    # expose usage detail on the plain chat() path too (e.g. DeepSeek), so that
    # non-tool callers (the planner) can account cache savings. ToolCallResponse
    # redeclares these for backward compat; the defaults match.
    prompt_tokens: Optional[int] = None      # input tokens
    completion_tokens: Optional[int] = None  # output tokens
    cache_read_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


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
    prompt_tokens: Optional[int] = None     # input tokens
    completion_tokens: Optional[int] = None  # output tokens
    # Prompt caching fields (populated by providers that support them, e.g. Anthropic).
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    # Reasoning tokens (populated by DeepSeek — completion_tokens = reasoning + visible).
    reasoning_tokens: Optional[int] = None


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
        try:
            choices = raw_resp.get("choices") or []
            msg_obj = (choices[0].get("message", {}) if choices else {}) or {}
            rc = msg_obj.get("reasoning_content", "") or ""
            if isinstance(rc, str) and rc.strip():
                return rc.strip()
        except (AttributeError, TypeError, IndexError):
            pass
    return content if isinstance(content, str) else ""
# Upper bound on a single Retry-After wait (seconds). Guards against absurdly
# large server values (e.g. far-future HTTP-dates) that would stall the agent.
RETRY_AFTER_MAX_WAIT = 60


def parse_retry_after(headers: "Any") -> Optional[int]:
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
    try:
        return min(RETRY_AFTER_MAX_WAIT, max(1, int(str(raw).strip())))
    except ValueError:
        pass
    try:
        import time as _time
        from email.utils import parsedate_to_datetime
        retry_time = parsedate_to_datetime(str(raw).strip())
        wait = int(retry_time.timestamp() - _time.time())
        if wait <= 0:
            return None
        return min(RETRY_AFTER_MAX_WAIT, max(1, wait))
    except (ValueError, TypeError, OSError):
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


def is_balance_quota_signal(error_code: Optional[int], body_text: str = "") -> bool:
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
class LLMClientError(Exception):
    """Base exception for LLM client errors"""
    pass


class LLMConnectionError(LLMClientError):
    """Cannot connect to LLM API"""
    pass


class LLMAuthenticationError(LLMClientError):
    """Invalid API key or authentication failed"""
    pass


class LLMRateLimitError(LLMClientError):
    """Rate limit exceeded.

    ``retry_after`` carries the server's suggested wait (seconds, parsed from
    the ``Retry-After`` header) when available, so the retry loop can honor it
    instead of a fixed backoff. ``None`` means the server gave no hint.
    """

    def __init__(self, *args: object, retry_after: "Optional[int]" = None,
                 error_code: "Optional[int]" = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after
        self.error_code = error_code


class LLMQuotaExceededError(LLMClientError):
    """API key has insufficient credits / quota exceeded (HTTP 402)"""
    pass


class LLMAPIError(LLMClientError):
    """API returned error response"""
    pass


class LLMServerUnavailableError(LLMClientError):
    """Server is unavailable (503, timeout, connection failure) — abort, do not fall back."""
    pass


class LLMClient(ABC):
    """
    Abstract base class for LLM clients

    All external LLM providers must implement this interface
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: int = DEFAULT_LLM_TIMEOUT):
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
        max_tokens: Optional[int] = None,
        **kwargs
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
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic')"""
        pass

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        **kwargs
    ) -> "ToolCallResponse":
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
        session = getattr(self, '_session', None)
        if session is not None:
            session.close()
            logger.debug("LLMClient session closed for %s", self.get_provider_name())

    def simple_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "",
        **kwargs
    ) -> LLMResponse:
        """
        Simplified interface for single-turn prompts

        Args:
            user_prompt: User's prompt
            system_prompt: Optional system instructions
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            LLMResponse
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=user_prompt))

        return self.chat(messages, model=model, **kwargs)


def resolve_provider_base_url(provider: str) -> Optional[str]:
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
    base_url: Optional[str] = None,
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
        >>> response = client.simple_prompt("Fix this code", model="gpt-4")
    """
    provider_lower = provider.lower()

    # Special handling for Ollama: use longer default timeout for model loading.
    # Compare against the cloud default (DEFAULT_LLM_TIMEOUT), not the magic
    # number 120, so explicit per-call overrides are still respected.
    if provider_lower == "ollama" and timeout == DEFAULT_LLM_TIMEOUT:
        timeout = OLLAMA_LLM_TIMEOUT
        logger.debug(f"Using extended timeout for Ollama: {timeout}s")

    if provider_lower == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(api_key, base_url, timeout)

    elif provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(api_key, base_url, timeout)

    elif provider_lower == "google":
        from .providers import GoogleClient
        return GoogleClient(api_key, base_url, timeout)

    elif provider_lower == "deepseek":
        from .providers import DeepSeekClient
        return DeepSeekClient(api_key, base_url, timeout)

    elif provider_lower == "ollama":
        from .providers import OllamaClient
        return OllamaClient(api_key, base_url, timeout)

    elif provider_lower in ("zai",):
        from .anthropic_client import ZAIAnthropicClient
        return ZAIAnthropicClient(api_key, base_url, timeout)

    elif provider_lower == "openrouter":
        from .openai_client import OpenRouterClient
        return OpenRouterClient(api_key, base_url, timeout)

    elif provider_lower == "opencode":
        from .openai_client import OpenAIClient
        if not base_url:
            base_url = "https://opencode.ai/zen/go/v1"  # default for OpenCode Go
        return OpenAIClient(api_key, base_url, timeout)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: openai, anthropic, google, deepseek, ollama, zai, openrouter, opencode"
        )
