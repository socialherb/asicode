"""
OpenAI (ChatGPT) client for asicode
"""
from __future__ import annotations

import dataclasses
import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any, Optional

import requests

from .client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMClient,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponse,
    LLMServerUnavailableError,
    ToolCallRequest,
    ToolCallResponse,
    is_balance_quota_signal,
    parse_retry_after,
)
from .output_parser import parse_tool_args

logger = logging.getLogger(__name__)


def _openai_content(msg: LLMMessage):
    """Build OpenAI-compatible content: str if no images, list of parts if images attached."""
    images = getattr(msg, "images", None)
    if not images:
        return msg.content
    parts: list[dict[str, Any]] = []
    for img in images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"},
        })
    parts.append({"type": "text", "text": msg.content})
    return parts


def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (o-series or gpt-5+).

    Reasoning models: o1, o3, o4, gpt-5, deepseek-reasoner, deepseek-v4-*, etc.
    Non-reasoning: gpt-4o, gpt-4, gpt-3.5, deepseek-chat, etc.

    DeepSeek v4 (flash/pro) emits ``reasoning_tokens`` in ``completion_tokens_details``
    and shares the ``max_tokens`` budget between reasoning + content — so on providers
    that treat it as a reasoner (OpenCode Go, OpenRouter) the reasoning tokens eat the
    whole ``max_tokens`` budget and content comes back empty (finish_reason=length).
    Routing it through ``max_completion_tokens`` instead gives reasoning its own budget
    so content survives. The native DeepSeek endpoint honors ``thinking:{type:disabled}``
    and produces zero reasoning tokens regardless, so this classification is safe there too.
    """
    _m = model.strip().lower()
    # strip provider/route prefixes (e.g. "deepseek/deepseek-v4-flash",
    # "openrouter/deepseek/deepseek-v4-flash") down to the bare model name so the
    # v4-* check matches regardless of how the caller prefixed the model id.
    _bare = _m.split("/")[-1]
    return (
        _m.startswith("o-") or _m.startswith("o1") or _m.startswith("o3")
        or _m.startswith("o4") or "gpt-5" in _m or "reasoner" in _m
        or _bare.startswith("deepseek-v4")
        or "kimi-k3" in _m
    )


def _is_deepseek_v4(model: str) -> bool:
    """True for DeepSeek v4 models (flash/pro), which support the native ``thinking`` parameter.

    DeepSeek v4 is unique among reasoning models routed through ``OpenAIClient``:
    ``reasoning_effort="low"`` only *dials down* reasoning (measured 882→742 tokens
    on OpenCode Go) — it cannot disable it. The native ``thinking:{type:disabled}``
    parameter is the ONLY way to produce zero reasoning tokens (measured 0 on both
    OpenCode Go and native DeepSeek; see the docstring on ``_is_reasoning_model``).
    OpenAI o-series / gpt-5 lack this parameter, so they keep ``reasoning_effort``.

    Strips provider/route prefixes (``deepseek/``, ``openrouter/deepseek/``) so the
    bare model name is checked regardless of caller prefixing.
    """
    _bare = model.strip().lower().split("/")[-1]
    return _bare.startswith("deepseek-v4")


def _is_kimi_k3(model: str) -> bool:
    """True for Kimi K3, which always has thinking ON and only supports ``reasoning_effort="max"``.

    Kimi K3 is unique among reasoning models:
    * ``reasoning_effort`` currently supports only ``"max"`` — ``"medium"`` / ``"low"``
      cause a 400 error.
    * Thinking is always enabled and cannot be disabled.
    * The ``thinking`` parameter (used by K2.x) must NOT be sent.

    Strips provider/route prefixes so ``opencode/kimi-k3`` matches.
    """
    _bare = model.strip().lower().split("/")[-1]
    return "kimi-k3" in _bare  # substring to match variants (kimi-k3-0711, kimi-k3-turbo, etc.)


def _parse_error_code(response: requests.Response) -> Optional[int]:
    """Extract provider-specific error code (e.g., 1302, 1305) from JSON body.

    GLM API returns ``{"error":{"code":1305,"message":"..."}}`` — parse the
    ``code`` field and return it as an integer.  Returns ``None`` when the body
    is not valid JSON, lacks an ``error`` dict, or the code is absent.
    """
    try:
        obj = json.loads(response.text.strip())
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, (int, float)):
                return int(code)
            if isinstance(code, str) and code.strip().isdigit():
                return int(code.strip())
        return None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _short_error_reason(body: str, limit: int = 160) -> str:
    """Condense a provider error body into one tidy log-friendly line.

    Providers return a JSON envelope (``{"error":{"code":..,"message":..}}``)
    that, dumped raw into a WARNING, wraps mid-token and breaks the TUI's
    indentation.  Pull out ``code``/``message`` when present, collapse any
    whitespace to single spaces, and truncate — so retry warnings stay on one
    readable line.  Falls back to a flattened, trimmed copy of *body*.
    """
    raw = (body or "").strip()
    if not raw:
        return ""
    reason = ""
    start = raw.find("{")
    if start >= 0:
        try:
            obj = json.loads(raw[start:])
            err = obj.get("error") if isinstance(obj, dict) else None
            if isinstance(err, dict):
                msg = str(err.get("message") or "").strip()
                code = str(err.get("code") or "").strip()
                reason = f"{code} {msg}".strip() if code else msg
            elif isinstance(err, str):
                reason = err.strip()
        except (ValueError, TypeError):
            reason = ""
    if not reason:
        reason = raw
    reason = " ".join(reason.split())  # collapse newlines/runs of whitespace
    return reason if len(reason) <= limit else reason[: limit - 1] + "…"


def _is_balance_quota_response(response: requests.Response) -> bool:
    """Return True when *response* signals an exhausted balance/quota (not rate limit).

    Thin wrapper over the canonical ``is_balance_quota_signal`` (defined in
    ``client.py``) — the single source for balance/quota detection shared by
    the OpenAI-compatible client (this module) and the Anthropic-compatible
    client (the primary zai endpoint). Preserved as a local helper so the
    existing retry-loop call site and its tests stay stable.
    """
    return is_balance_quota_signal(_parse_error_code(response), response.text)


def _reasoning_effort_value(model: str, thinking_mode: bool) -> str:
    """reasoning_effort value per model family: 'minimal' is gpt-5-only; o-series floor is 'low'."""
    if thinking_mode:
        return "medium"
    return "minimal" if "gpt-5" in model.strip().lower() else "low"


def _apply_thinking_mode(
    payload: dict[str, Any],
    model: str,
    thinking_mode: Optional[bool],
    effort_override: Optional[str],
    *,
    is_reasoning: bool,
) -> None:
    """Mutate *payload* with the provider-appropriate thinking/reasoning controls.

    Two model families, dispatched by :func:`_is_deepseek_v4`:

    * **DeepSeek v4** (flash/pro) — supports the native ``thinking`` parameter,
      the ONLY way to truly suppress reasoning (0 tokens). ``reasoning_effort="low"``
      merely dials it down (882→742 tokens measured on OpenCode Go), so for
      ``thinking_mode=False`` we send ``thinking:{type:disabled}`` and NO
      ``reasoning_effort``. This matches :class:`DeepSeekClient` (native endpoint)
      exactly, removing the divergence where ``OpenAIClient`` (OpenCode Go /
      OpenRouter) wrongly used ``reasoning_effort`` for DeepSeek v4.

    * **Kimi K3** — always has thinking ON and only supports
      ``reasoning_effort="max"``; ``thinking`` parameter must NOT be sent.

    * **OpenAI o-series / gpt-5** — lack the ``thinking`` parameter, so they keep
      the ``reasoning_effort`` dial (``minimal`` for gpt-5, else ``low``).

    Shared by ``chat()`` and ``chat_with_tools()`` to keep their thinking-mode
    handling identical. No-op when ``thinking_mode is None`` (caller didn't request
    a toggle); a lone ``effort_override`` is left for the caller/subclass to handle.
    """
    if thinking_mode is None:
        return
    if _is_deepseek_v4(model):
        payload["thinking"] = {"type": "enabled" if thinking_mode else "disabled"}
        if thinking_mode and effort_override:
            payload["reasoning_effort"] = effort_override
    elif _is_kimi_k3(model):
        # Kimi K3: always thinking ON, only "max" supported, no "thinking" parameter.
        payload["reasoning_effort"] = "max"  # K3 only supports "max"; override is always ignored
    elif is_reasoning:
        payload["reasoning_effort"] = (
            effort_override if (thinking_mode and effort_override)
            else _reasoning_effort_value(model, thinking_mode)
        )
    elif effort_override:
        # Non-reasoning model but subclass explicitly requested reasoning_effort
        # (e.g. ZAIClient for GLM-5.2 which uses thinking.type but also supports
        # reasoning_effort as a separate top-level parameter).
        payload["reasoning_effort"] = effort_override


def _extract_cached_tokens(usage: Optional[dict[str, Any]]) -> Optional[int]:
    """Pull cached-prompt token count from OpenAI-style usage details.

    OpenAI, DeepSeek, Z.AI and OpenRouter all report the prompt-cache hit as
    ``usage.prompt_tokens_details.cached_tokens``. For every OpenAI-protocol
    provider it is a SUBSET of ``usage.prompt_tokens`` (included, not
    separate). Returns ``None`` when the field is absent so callers can tell
    "no cache reported" from "zero cached".
    """
    if not isinstance(usage, dict):
        return None
    ptd = usage.get("prompt_tokens_details")
    if ptd and isinstance(ptd, dict):
        ct = ptd.get("cached_tokens")
        if ct is not None:
            return int(ct)
    return None


class OpenAIClient(LLMClient):
    """
    OpenAI API client for ChatGPT models

    Supported models:
    - gpt-4-turbo-preview
    - gpt-4
    - gpt-3.5-turbo
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4-turbo-preview"
    # Minimum input tokens for prefix caching to take effect.
    # ZAI/GLM and OpenAI use automatic prefix matching (no explicit cache_control
    # breakpoints), with a 1024-token minimum for cache eligibility.
    min_input_tokens_for_cache = 1024

    def get_provider_name(self) -> str:
        return "openai"

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the API request.

        Subclasses can override this to add provider-specific headers
        (e.g. OpenRouter's ``HTTP-Referer`` / ``X-Title`` for app attribution).
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        tag: str = "chat",
        stream: bool = False,
    ) -> requests.Response:
        """POST ``payload`` with exponential backoff for transient errors.

        Unified retry path shared by ``chat()`` / ``chat_with_tools()`` /
        ``_chat_with_tools_streaming()`` — previously the same ~70-line block
        was copy-pasted three times (so any fix had to be applied in triplicate,
        and a backoff bug existed in all three). Returns the
        ``requests.Response`` for the first non-retryable status (success or a
        caller-handled 4xx); the caller inspects ``status_code``.

        Raises ``LLMRateLimitError`` / ``LLMServerUnavailableError`` when
        retries are exhausted.

        Backoff semantics: a ``Retry-After`` wait is honored literally, and it
        suppresses backoff for the *next* attempt only (we just slept).
        Previously ``_skipped_backoff`` was a sticky flag that, once set True
        by a 429-with-Retry-After, *permanently* disabled exponential backoff
        for all subsequent attempts — so a short Retry-After (e.g. 1s) followed
        by a 5xx would retry with zero backoff, hammering an already-strained
        server. The one-shot ``_skip_next_backoff`` below fixes that: after the
        suppressed attempt runs, normal exponential backoff resumes.
        """
        _max_retries = 3
        _skip_next_backoff = False  # one-shot: consumed by the next attempt
        t0 = time.monotonic()
        for _retry in range(_max_retries):
            if _retry > 0:
                if _skip_next_backoff:
                    _skip_next_backoff = False  # consume the one-shot skip
                else:
                    _delay = min(2 ** (_retry + 1), 20)  # 4s, 8s, 16s (capped at 20s)
                    _delay += random.uniform(0, _delay * 0.5)  # jitter: avoid retry storms
                    logger.info(
                        "API retry %d/%d after %.1fs delay (%s)",
                        _retry, _max_retries - 1, _delay, tag,
                    )
                    time.sleep(_delay)
            try:
                response = self._session.post(
                    url, headers=headers, json=payload,
                    timeout=self.timeout, stream=stream,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(
                    "API connection attempt %d/%d failed: %s (%s)",
                    _retry + 1, _max_retries, type(e).__name__, tag,
                )
                if _retry == _max_retries - 1:
                    _elapsed_ms = (time.monotonic() - t0) * 1000
                    if isinstance(e, requests.Timeout):
                        raise LLMServerUnavailableError(
                            f"API request timed out after {_max_retries} attempts "
                            f"({_elapsed_ms:.0f}ms total) ({tag})"
                        ) from e
                    raise LLMServerUnavailableError(
                        f"Cannot connect to API after {_max_retries} attempts "
                        f"({_elapsed_ms:.0f}ms total). Check internet connection. ({tag})"
                    ) from e
                continue
            # Check for retryable status codes inside the loop
            if response.status_code == 429:
                # Exhausted balance/quota is reported as 429 by some providers
                # (zai/GLM code 1113). It is NOT transient — retrying cannot fix
                # a depleted balance, so raise immediately as a non-retryable
                # quota error rather than burning the retry budget and surfacing
                # a misleading "rate limit / try again" (or auth) message.
                if _is_balance_quota_response(response):
                    raise LLMQuotaExceededError(
                        f"Account balance/quota exhausted (429): "
                        f"{_short_error_reason(response.text)}"
                    )
                error_body = _short_error_reason(response.text)
                if _retry < _max_retries - 1:
                    _retry_after = parse_retry_after(response.headers)
                    if _retry_after is not None:
                        logger.info(
                            "API rate limited (429), waiting %ds (Retry-After), retry %d/%d (%s)",
                            _retry_after, _retry + 1, _max_retries - 1, tag,
                        )
                        time.sleep(_retry_after)
                        _skip_next_backoff = True  # just slept — skip next attempt's backoff
                    else:
                        logger.info(
                            "API rate limited (429), retry %d/%d (%s)",
                            _retry + 1, _max_retries - 1, tag,
                        )
                    continue
                raise LLMRateLimitError(
                    f"API request rejected (429): {error_body}",
                    retry_after=parse_retry_after(response.headers),
                    error_code=_parse_error_code(response),
                )
            if response.status_code >= 500 and response.status_code != 501:
                error_body = _short_error_reason(response.text)
                if _retry < _max_retries - 1:
                    logger.info(
                        "API server error %d, retry %d/%d (%s)",
                        response.status_code, _retry + 1, _max_retries - 1, tag,
                    )
                    continue
                raise LLMServerUnavailableError(f"API returned HTTP {response.status_code}: {error_body}")
            break  # Success (non-retryable) — exit retry loop
        return response

    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request to OpenAI

        API Reference: https://platform.openai.com/docs/api-reference/chat
        """
        if not model:
            model = self.DEFAULT_MODEL

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = self._build_headers()

        # Convert to OpenAI format (with optional image support)
        api_messages = [
            {"role": msg.role, "content": _openai_content(msg)}
            for msg in messages
        ]

        is_reasoning = _is_reasoning_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        if not is_reasoning:
            payload["temperature"] = temperature

        if max_tokens:
            # Reasoning models on native OpenAI reject max_tokens and require
            # max_completion_tokens.  OpenCode (opencode.ai) does NOT support
            # max_completion_tokens — it is silently ignored, causing reasoning
            # to run unbounded and time out.  Detect OpenCode by base_url and
            # keep max_tokens (shared reasoning+content budget) for that endpoint.
            _base = (self.base_url or "").lower()
            _is_opencode = "opencode.ai" in _base
            _use_max_completion = is_reasoning and not _is_opencode
            payload["max_completion_tokens" if _use_max_completion else "max_tokens"] = max_tokens

        # Add any extra kwargs
        # reasoning_callback intentionally NOT consumed — closed-thinking
        # policy (OpenAI o-series + Anthropic extended-thinking) suppresses
        # reasoning content from the UI panel. See anthropic_client.py:344.
        kwargs.pop("reasoning_callback", None)
        thinking_mode = kwargs.pop("thinking_mode", None)
        _effort_override = kwargs.pop("reasoning_effort", None)
        _apply_thinking_mode(payload, model, thinking_mode, _effort_override, is_reasoning=is_reasoning)
        token_callback = kwargs.pop("token_callback", None)
        # cache_breakpoint_offset is Anthropic-specific (explicit cache_control
        # breakpoints). OpenAI-compatible providers (ZAI/GLM, DeepSeek, Ollama)
        # use automatic prefix matching and silently ignore this field, so strip
        # it to keep the payload clean and avoid future strict-validation 400s.
        kwargs.pop("cache_breakpoint_offset", None)
        payload.update(kwargs)

        # Streaming path: when token_callback is provided, use SSE streaming so
        # text content tokens are forwarded in real-time. This lets the final
        # summary (plain chat() call, no tools) stream incrementally instead of
        # blocking until the whole response is buffered — which would leave the
        # "thinking" ticker as the only visible UI for the entire call duration.
        # Mirrors chat_with_tools() L498-499 streaming gate.
        if token_callback is not None:
            return self._chat_streaming(url, headers, payload, model, token_callback)

        t0 = time.monotonic()

        response = self._request_with_retry(url, headers, payload, tag="chat")

        try:
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Handle specific error codes — only decode body on error paths
            # (avoid double-decoding on the success path where response.json()
            # follows anyway).
            if response.status_code == 401:
                error_body = response.text[:500]
                logger.error("OpenAI authentication failed (401): %s", error_body)
                raise LLMAuthenticationError(
                    f"Invalid API key (401): {error_body}"
                )

            if response.status_code == 402:
                error_body = response.text[:500]
                logger.error("Insufficient credits (402): %s", error_body)
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (402): {error_body}"
                )

            if response.status_code != 200:
                error_body = response.text[:500]
                logger.debug(
                    "OpenAI API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMAPIError(
                    f"API returned HTTP {response.status_code}: {error_body}"
                )

            data = response.json()

            # Extract content
            choices = data.get("choices", [])
            if not choices:
                logger.warning("OpenAI response has no choices")
                return LLMResponse(
                    content="",
                    model=model,
                    provider=self.get_provider_name(),
                    raw_response=data
                )

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason")

            # Extract token usage
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            cached_tokens = _extract_cached_tokens(usage)

            logger.info(
                "OpenAI %s: %.0fms, tok=%s, finish_reason=%s",
                model, elapsed_ms, tokens_used, finish_reason
            )

            return LLMResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=data,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_input_tokens=cached_tokens,
            )

        except (requests.ConnectionError, requests.Timeout):
            raise  # Already handled by retry loop above

        except requests.RequestException as e:
            logger.error("API request failed: %s", e)
            raise LLMAPIError(f"API request failed: {e}") from e

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        **kwargs
    ) -> ToolCallResponse:
        """
        Send chat completion request with tool calling support.

        Uses OpenAI function calling API.
        When token_callback is provided (via kwargs), streams text tokens in real-time.
        """
        if not model:
            model = self.DEFAULT_MODEL

        token_callback = kwargs.pop("token_callback", None)

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = self._build_headers()

        api_messages = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role, "content": _openai_content(m)}
            if m.role == "assistant" and getattr(m, "tool_calls", None):
                d["tool_calls"] = m.tool_calls
                if d.get("content") is None:
                    d["content"] = ""
            if m.role == "tool":
                if getattr(m, "tool_call_id", None):
                    d["tool_call_id"] = m.tool_call_id
                if getattr(m, "name", None):
                    d["name"] = m.name
            api_messages.append(d)

        # Convert tool schemas to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
            for t in tools
        ]

        is_reasoning = _is_reasoning_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "tools": openai_tools,
            "tool_choice": "auto",
        }
        # reasoning_callback intentionally NOT consumed — see chat() L312 note.
        kwargs.pop("reasoning_callback", None)
        thinking_mode = kwargs.pop("thinking_mode", None)
        _effort_override = kwargs.pop("reasoning_effort", None)
        _apply_thinking_mode(payload, model, thinking_mode, _effort_override, is_reasoning=is_reasoning)
        if is_reasoning:
            kwargs.pop("temperature", None)  # reasoning models reject temperature != 1
            _mt = kwargs.pop("max_tokens", None)
            if _mt:
                # OpenCode (opencode.ai) does NOT support max_completion_tokens — it is
                # silently ignored, causing reasoning to run unbounded and time out.
                # Mirror the per-provider dispatch in chat() to keep this consistent.
                _base = (self.base_url or "").lower()
                _is_opencode = "opencode.ai" in _base
                payload["max_completion_tokens" if not _is_opencode else "max_tokens"] = _mt
        # See chat(): strip Anthropic-only cache breakpoint marker.
        kwargs.pop("cache_breakpoint_offset", None)
        payload.update(kwargs)

        if token_callback is not None:
            return self._chat_with_tools_streaming(url, headers, payload, model, token_callback)

        t0 = time.monotonic()

        response = self._request_with_retry(url, headers, payload, tag="tools")

        try:
            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code == 401:
                error_body = response.text[:500]
                raise LLMAuthenticationError(f"Invalid API key (401): {error_body}")
            if response.status_code == 402:
                error_body = response.text[:500]
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (402): {error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(f"API returned HTTP {response.status_code}: {error_body}")

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ToolCallResponse(
                    content="", model=model, provider=self.get_provider_name(),
                    raw_response=data, tool_calls=[], is_final=True,
                )

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")

            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            cached_tokens = _extract_cached_tokens(usage)

            # Parse tool calls
            tool_calls: list[ToolCallRequest] = []
            raw_tool_calls = message.get("tool_calls") or []


            for tc in raw_tool_calls:

                func = tc.get("function", {})
                args = parse_tool_args(func.get("arguments", "{}"))
                tool_calls.append(ToolCallRequest(
                    call_id=tc.get("id", ""),
                    name=func.get("name", ""),
                    args=args,
                ))

            is_final = finish_reason == "stop" or not tool_calls
            logger.info(
                "OpenAI %s (tools): %.0fms, tok=%s, finish=%s (%d)",
                model, elapsed_ms, tokens_used, finish_reason, len(tool_calls),
            )
            return ToolCallResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=data,
                tool_calls=tool_calls,
                is_final=is_final,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_input_tokens=cached_tokens,
            )

        except (requests.ConnectionError, requests.Timeout):
            raise  # Already handled by retry loop above
        except requests.RequestException as e:
            raise LLMAPIError(f"API request failed: {e}") from e

    def _chat_streaming(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        token_callback,
    ) -> LLMResponse:
        """Streaming variant of chat() (no tools). Forwards text content tokens.

        Simplified version of _chat_with_tools_streaming: no tool_call deltas.
        Returns LLMResponse (not ToolCallResponse) to match chat()'s contract.
        """
        import json as _json

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}

        t0 = time.monotonic()

        response = self._request_with_retry(
            url, headers, stream_payload, tag="stream", stream=True,
        )

        # Status checks live INSIDE the try/finally so a non-200 streaming
        # response (or a failure while reading its error body via .text) is
        # always closed — mirroring the providers.py streaming pattern. A bare
        # raise before the try would leak the underlying connection.
        try:
            if response.status_code == 401:
                error_body = response.text[:500]
                raise LLMAuthenticationError(f"Invalid API key (401): {error_body}")
            if response.status_code == 402:
                error_body = response.text[:500]
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (402): {error_body}"
                )
            if response.status_code != 200:
                raise LLMAPIError(f"API returned HTTP {response.status_code}: {response.text[:500]}")

            text_content = ""
            finish_reason = None
            prompt_tokens = None
            completion_tokens = None
            cached_tokens = None
            reasoning_content = ""

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    ev = _json.loads(data_str)
                except Exception:
                    continue

                # Usage chunk (stream_options)
                if ev.get("usage"):
                    u = ev["usage"]
                    prompt_tokens = u.get("prompt_tokens") or prompt_tokens
                    completion_tokens = u.get("completion_tokens") or completion_tokens
                    _ct = _extract_cached_tokens(u)
                    if _ct is not None:
                        cached_tokens = _ct

                choices = ev.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason") or finish_reason

                # Text delta
                chunk = delta.get("content") or ""
                if chunk:
                    text_content += chunk
                    try:
                        token_callback(chunk)
                    except Exception:
                        pass

                # Reasoning content (DeepSeek Reasoner streams reasoning_content)
                _rc = delta.get("reasoning_content") or ""
                if _rc:
                    reasoning_content += _rc

        except requests.RequestException as e:
            raise LLMAPIError(f"OpenAI streaming request failed: {e}") from e
        finally:
            response.close()

        elapsed_ms = (time.monotonic() - t0) * 1000
        tokens_used = (prompt_tokens or 0) + (completion_tokens or 0) or None
        logger.info(
            "OpenAI %s (stream): %.0fms, tok=%s, finish=%s",
            model, elapsed_ms, tokens_used, finish_reason,
        )

        raw_response: dict[str, Any] = {
            "id": "stream",
            "model": model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text_content,
                    **({"reasoning_content": reasoning_content} if reasoning_content else {}),
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
            },
        }

        return LLMResponse(
            content=text_content,
            model=model,
            provider=self.get_provider_name(),
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            raw_response=raw_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
        )

    def _chat_with_tools_streaming(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        token_callback,
    ) -> ToolCallResponse:
        """Streaming variant for OpenAI tool calls.

        Forwards text content tokens via token_callback.
        Tool-call argument deltas are buffered silently.
        """
        import json as _json

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}

        t0 = time.monotonic()

        response = self._request_with_retry(
            url, headers, stream_payload, tag="stream", stream=True,
        )

        # Status checks inside try/finally (see _chat_streaming): guarantees the
        # streaming response is closed on a non-200 raise or an error-body read
        # failure, instead of leaking the connection.
        try:
            if response.status_code == 401:
                error_body = response.text[:500]
                raise LLMAuthenticationError(f"Invalid API key (401): {error_body}")
            if response.status_code == 402:
                error_body = response.text[:500]
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (402): {error_body}"
                )
            if response.status_code != 200:
                raise LLMAPIError(f"API returned HTTP {response.status_code}: {response.text[:500]}")

            text_content = ""
            finish_reason = None
            prompt_tokens = None
            completion_tokens = None
            cached_tokens = None

            # Accumulate tool calls by index
            _tool_acc: dict[int, dict] = {}

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    ev = _json.loads(data_str)
                except Exception:
                    continue

                # Usage chunk (stream_options)
                if ev.get("usage"):
                    u = ev["usage"]
                    prompt_tokens = u.get("prompt_tokens") or prompt_tokens
                    completion_tokens = u.get("completion_tokens") or completion_tokens
                    _ct = _extract_cached_tokens(u)
                    if _ct is not None:
                        cached_tokens = _ct

                choices = ev.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason") or finish_reason

                # Text delta
                chunk = delta.get("content") or ""
                if chunk:
                    text_content += chunk
                    try:
                        token_callback(chunk)
                    except Exception:
                        pass

                # Tool call deltas
                for tc_delta in (delta.get("tool_calls") or []):
                    idx = tc_delta.get("index", 0)
                    if idx not in _tool_acc:
                        _tool_acc[idx] = {"id": "", "name": "", "args_json": ""}
                    if tc_delta.get("id"):
                        _tool_acc[idx]["id"] = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        _tool_acc[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        _tool_acc[idx]["args_json"] += func["arguments"]

        except requests.RequestException as e:
            raise LLMAPIError(f"OpenAI streaming request failed: {e}") from e
        finally:
            response.close()

        # Build tool calls
        tool_calls: list[ToolCallRequest] = []
        for idx in sorted(_tool_acc):
            acc = _tool_acc[idx]
            try:
                args = _json.loads(acc["args_json"] or "{}")
            except Exception:
                args = {}
            tool_calls.append(ToolCallRequest(
                call_id=acc["id"],
                name=acc["name"],
                args=args,
            ))

        elapsed_ms = (time.monotonic() - t0) * 1000
        tokens_used = (prompt_tokens or 0) + (completion_tokens or 0) or None
        is_final = finish_reason == "stop" or not tool_calls
        logger.info(
            "OpenAI %s (tools): %.0fms, tok=%s, finish=%s (%d)",
            model, elapsed_ms, tokens_used, finish_reason, len(tool_calls),
        )
        return ToolCallResponse(
            content=text_content,
            model=model,
            provider=self.get_provider_name(),
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            raw_response={},
            tool_calls=tool_calls,
            is_final=is_final,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
                cache_read_input_tokens=cached_tokens,
        )


class ZAIClient(OpenAIClient):
    """
    Z.AI (GLM) API client — OpenAI-compatible protocol.

    NOTE: Coding Plan subscribers MUST use the coding-specific endpoint
    (https://api.z.ai/api/coding/paas/v4). The general API endpoint
    (https://api.z.ai/api/paas/v4) requires separate prepaid balance
    and is NOT covered by the Coding Plan subscription.

    Supported models:
    - glm-5.2
    - glm-5.1, glm-5, glm-5-turbo
    - glm-4.7, glm-4.6, glm-4.5, etc.

    Thinking/reasoning mode:
    GLM-5.2+ uses a different thinking format from OpenAI:
      thinking: {type: "enabled" | "disabled"}
      reasoning_effort: "max" | "xhigh" | "high" | "medium" | "low" | "minimal" | "none"
    Thinking is ENABLED by default on GLM-5.2+ — send thinking.type="disabled" to turn off.
    """
    DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
    DEFAULT_MODEL = "glm-5.2"

    def get_provider_name(self) -> str:
        return "zai"

    @staticmethod
    def _normalize_cache_accounting(resp):
        """Re-normalize OpenAI-protocol (subset) cache semantics to the
        SEPARATE-accounting shape ``_CACHE_TOKENS_SEPARATE`` expects for "zai".

        z.ai is normally served over the Anthropic protocol (whose
        ``input_tokens`` EXCLUDES cached reads), so "zai" is classified as
        separate-accounting. But THIS client speaks the OpenAI protocol, where
        ``prompt_tokens`` INCLUDES ``cached_tokens`` as a subset. Split them at
        the boundary so the downstream "zai" separate formulas
        (``cache_hit_pct`` / ``estimate_cache_adjusted_cost``) stay correct:
        ``prompt_tokens`` becomes the uncached-only count while
        ``cache_read_input_tokens`` (already populated by the parent from
        ``usage.prompt_tokens_details``) carries the cached subset.
        """
        cached = resp.cache_read_input_tokens
        if not cached:
            return resp
        prompt_tokens = resp.prompt_tokens or 0
        if cached <= prompt_tokens:
            resp = dataclasses.replace(resp, prompt_tokens=prompt_tokens - cached)
        _pct = (cached / prompt_tokens * 100) if prompt_tokens else 0
        logger.info(
            "Z.AI cache: %s cached tokens (%.0f%% of prompt — Coding Plan: no discount)",
            cached, _pct,
        )
        return resp

    @staticmethod
    def _apply_glm_reasoning_fallback(resp):
        """Recover a GLM-5.2 (thinking ON) final answer that arrived ONLY in
        ``reasoning_content`` with an empty ``content`` field.

        GLM-5.2 may place the entire answer in ``reasoning_content`` and leave
        ``content`` empty. Without recovery, every consumer reading
        ``resp.content`` directly (intent_resolver, agent_phase_manager,
        planner_plan_create, orchestrator, design_chat_loop) silently gets an
        empty string — the model's decision/answer is lost.

        Guarded against tool-call turns: when the model emits ``tool_calls``
        empty ``content`` is NORMAL (the tool calls ARE the response) and
        injecting reasoning would pollute the assistant message, so we return
        unchanged. ``LLMResponse`` from ``chat()`` has no ``tool_calls`` attr,
        so ``getattr`` returns ``None`` and never blocks the plain-chat path.

        Shared by ``chat()`` and ``chat_with_tools()`` — multi-path fallback
        parity (insight A36) via a single canonical helper (insight A38).
        """
        if getattr(resp, "tool_calls", None):
            return resp
        if resp.content or not resp.raw_response:
            return resp
        choices = resp.raw_response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {}) or {}
            rc = msg.get("reasoning_content")
            if rc:
                return dataclasses.replace(resp, content=rc)
        return resp

    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        # Pop z.ai-specific params before passing to parent
        thinking_mode = kwargs.pop("thinking_mode", None)
        _effort_override = kwargs.pop("reasoning_effort", None)

        if thinking_mode is not None:
            # Explicit thinking on/off
            kwargs["thinking"] = {"type": "enabled" if thinking_mode else "disabled"}
            if thinking_mode and _effort_override:
                kwargs["reasoning_effort"] = _effort_override
            # Re-inject thinking_mode so OpenAIClient.chat() can use it for
            # reasoning_effort logic.
            kwargs["thinking_mode"] = thinking_mode
        elif _effort_override:
            # GLM-5.2 thinks by default (thinking_mode=None).  When the caller
            # passes reasoning_effort without an explicit thinking_mode toggle,
            # pass the effort through so the parent can include it in the payload.
            kwargs["reasoning_effort"] = _effort_override

        resp = super().chat(messages, model, temperature, max_tokens, **kwargs)

        # Z.AI reports cache via the OpenAI protocol (cached ⊆ prompt_tokens).
        # The parent already parsed usage.prompt_tokens_details into
        # resp.cache_read_input_tokens; normalize to the separate-accounting
        # shape that provider "zai" expects.
        resp = self._normalize_cache_accounting(resp)

        # GLM-5.2 (thinking ON) may emit the final answer in reasoning_content
        # with an empty content field — recover it via the shared helper.
        resp = self._apply_glm_reasoning_fallback(resp)
        return resp

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        max_tokens: Optional[int] = None,
        token_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> ToolCallResponse:
        # Pop z.ai-specific params before passing to parent
        thinking_mode = kwargs.pop("thinking_mode", None)
        _effort_override = kwargs.pop("reasoning_effort", None)

        if thinking_mode is not None:
            kwargs["thinking"] = {"type": "enabled" if thinking_mode else "disabled"}
            if thinking_mode and _effort_override:
                kwargs["reasoning_effort"] = _effort_override
            # Re-inject thinking_mode so OpenAIClient.chat_with_tools() can use it
            # for reasoning_effort logic.
            kwargs["thinking_mode"] = thinking_mode
        elif _effort_override:
            # GLM-5.2 thinks by default (thinking_mode=None).  When the caller
            # passes reasoning_effort without an explicit thinking_mode toggle,
            # pass the effort through so the parent can include it in the payload.
            kwargs["reasoning_effort"] = _effort_override

        resp = super().chat_with_tools(
            messages, tools, model=model,
            max_tokens=max_tokens, token_callback=token_callback,
            **kwargs
        )

        # Normalize OpenAI subset cache → zai separate-accounting (see _normalize).
        resp = self._normalize_cache_accounting(resp)

        # GLM-5.2 (thinking ON) may emit the final answer in reasoning_content
        # with an empty content field — recover it here too (parity with chat()).
        # chat_with_tools feeds intent_resolver / agent_phase_manager /
        # planner_plan_create / orchestrator / design_chat, all of which read
        # .content directly; without this they silently get an empty string and
        # the model's decision is lost (multi-path fallback parity, insight A36).
        resp = self._apply_glm_reasoning_fallback(resp)

        return resp


class OpenRouterClient(OpenAIClient):
    """OpenRouter API client — OpenAI-compatible protocol.

    OpenRouter (https://openrouter.ai) is a unified gateway to hundreds of
    models from many providers (DeepSeek, Anthropic, OpenAI, Google, Z.AI, ...).
    Model slugs use the ``<vendor>/<model>`` form, e.g.::

        deepseek/deepseek-v4-flash
        anthropic/claude-sonnet-4-6
        zai/glm-5.2

    OpenRouter-specific features supported here:

    - **App attribution headers** — ``HTTP-Referer`` and ``X-Title``. Optional,
      but make the app discoverable on OpenRouter's rankings. Override via
      ``OPENROUTER_SITE_URL`` / ``OPENROUTER_APP_TITLE`` env vars.
    - **Provider pinning** — OpenRouter auto-routes to the cheapest provider,
      but this shatters the prompt cache between requests (cache is
      provider-local). Pinning a single upstream provider (e.g. ``DeepSeek``)
      raises the cache hit rate from ~50-70% to ~90%+. Set
      ``OPENROUTER_PROVIDER_ORDER`` (comma-separated) to enable it; it is
      injected as ``{"provider": {"order": [...]}}`` in the request body via
      the parent's ``payload.update(kwargs)``.

    Cost / cache accounting:
    OpenRouter reports ``usage.prompt_tokens_details.cached_tokens`` and, like
    DeepSeek/OpenAI, ``prompt_tokens`` INCLUDES the cached subset (not
    separate-accounting like Anthropic). We only emit a diagnostic log line —
    actual cost math uses the model-specific rates in ``_shared_utils``.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

    def get_provider_name(self) -> str:
        return "openrouter"

    @staticmethod
    def _strip_internal_prefix(model: str) -> str:
        """Strip the internal ``openrouter/`` routing prefix from a model slug.

        asicode accepts ``openrouter/<vendor>/<model>`` (e.g.
        ``openrouter/deepseek/deepseek-v4-flash``) purely to route the request
        through this client via ``detect_cloud_provider``. The OpenRouter API
        expects the bare ``<vendor>/<model>`` slug, so the leading
        ``openrouter/`` is stripped here. A bare slug (no internal prefix) is
        returned unchanged.
        """
        m = (model or "").strip()
        low = m.lower()
        if low.startswith("openrouter/"):
            return m[len("openrouter/"):]
        return m

    def _build_headers(self) -> dict[str, str]:
        import os
        headers = super()._build_headers()
        site_url = (os.getenv("OPENROUTER_SITE_URL", "") or "").strip()
        app_title = (os.getenv("OPENROUTER_APP_TITLE", "") or "").strip()
        # OpenRouter accepts both the canonical and legacy header names.
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_title:
            headers["X-Title"] = app_title
        return headers

    @staticmethod
    def _provider_preference() -> Optional[dict[str, Any]]:
        """Build the OpenRouter ``provider`` request field from env vars.

        Returns ``None`` when nothing is configured, leaving routing to
        OpenRouter's defaults. All knobs are opt-in (unset → omitted):

        - ``OPENROUTER_PROVIDER_ORDER``: comma-separated provider names (e.g.
          ``DeepSeek``). Pinning one provider keeps the prompt cache warm across
          requests (cache is provider-local; auto-routing scatters requests and
          breaks it). Raises the cache hit rate for
          ``deepseek/deepseek-v4-flash`` from ~50-70% to ~90%+.
        - ``OPENROUTER_DATA_COLLECTION``: ``deny`` refuses training-data
          collection by the upstream provider (privacy); ``allow`` permits it.
          Unset → omitted (OpenRouter default ``allow``). ``deny`` filters the
          provider pool to those that comply with the policy, which can reduce
          availability — hence it is opt-in.
        """
        import os
        pref: dict[str, Any] = {}

        order_raw = (os.getenv("OPENROUTER_PROVIDER_ORDER", "") or "").strip()
        if order_raw:
            order = [p.strip() for p in order_raw.split(",") if p.strip()]
            if order:
                pref["order"] = order

        dc_raw = (os.getenv("OPENROUTER_DATA_COLLECTION", "") or "").strip().lower()
        if dc_raw in ("allow", "deny"):
            pref["data_collection"] = dc_raw

        return pref or None

    def _inject_provider_preference(self, kwargs: dict[str, Any]) -> None:
        """Merge the provider preference into kwargs (forwarded to the payload).

        The parent ``OpenAIClient`` runs ``payload.update(kwargs)`` near the end
        of ``chat()`` / ``chat_with_tools()``, so placing ``provider`` here
        surfaces it as a top-level request field — exactly what OpenRouter
        expects. A caller-supplied ``provider`` kwarg wins over the env default.
        """
        pref = self._provider_preference()
        if pref is not None and "provider" not in kwargs:
            kwargs["provider"] = pref

    @staticmethod
    def _log_cache(raw_response: Optional[dict[str, Any]]) -> None:
        """Log the cached-token fraction from OpenRouter usage details."""
        if not raw_response:
            return
        usage = raw_response.get("usage", {})
        ptd = usage.get("prompt_tokens_details")
        if not (ptd and isinstance(ptd, dict)):
            return
        cached_tokens = ptd.get("cached_tokens")
        if cached_tokens is None:
            return
        prompt_tokens = usage.get("prompt_tokens", 0)
        cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
        logger.info(
            "OpenRouter cache: %s cached tokens (%.1f%% of prompt)",
            cached_tokens, cache_pct,
        )

    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        self._inject_provider_preference(kwargs)
        resp = super().chat(messages, self._strip_internal_prefix(model), temperature, max_tokens, **kwargs)
        self._log_cache(resp.raw_response)
        return resp

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        max_tokens: Optional[int] = None,
        token_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> ToolCallResponse:
        self._inject_provider_preference(kwargs)
        resp = super().chat_with_tools(
            messages, tools, model=self._strip_internal_prefix(model),
            max_tokens=max_tokens, token_callback=token_callback,
            **kwargs
        )
        self._log_cache(resp.raw_response)
        return resp
