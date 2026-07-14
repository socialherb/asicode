"""
Anthropic (Claude) client for asicode Test
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, Optional

import requests

from .agent.config.thresholds import config as _cfg
from .client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMClient,
    LLMConnectionError,
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

logger = logging.getLogger(__name__)

def _parse_glm_error_code(response: requests.Response) -> Optional[int]:
    """Extract GLM-specific error code (1302, 1305, 1210) from response body.

    Both the Anthropic and OpenAI endpoints at z.ai return errors in a JSON
    envelope like ``{"error":{"code":"1305","message":"..."}}``.  Returns the
    code as an integer, or ``None`` when the body is not parseable or lacks
    a code field.
    """
    try:
        obj = json.loads(response.text.strip())
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            code = err.get("code")
            # Handle both int and string-encoded int
            if isinstance(code, (int, float)):
                return int(code)
            if isinstance(code, str) and code.strip().isdigit():
                return int(code.strip())
        return None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _is_always_thinking_glm(model: str) -> bool:
    """Return True for GLM models where extended thinking is always on.

    GLM-5.2+ has thinking enabled by default. When not explicitly disabled
    via ``thinking.type = "disabled"``, the model always returns thinking content.
    """
    _m = (model or "").strip().lower()
    return any(
        _m.startswith(p)
        for p in ("glm-5.2", "glm-5.1", "glm-5", "glm-5-", "glm-5_turbo")
    )


def _inject_glm_thinking_kwargs(kwargs: dict, thinking_mode, effort_override, always_think: bool) -> None:
    """Inject thinking.type / effort kwargs for Z.AI GLM models.

    Called by ZAIAnthropicClient.chat() and chat_with_tools() to avoid
    duplicating the same if/elif chain in both methods.
    """
    if thinking_mode is False:
        kwargs["thinking"] = {"type": "disabled"}
    elif always_think:
        kwargs["thinking"] = {"type": "adaptive"}
        if effort_override:
            kwargs["effort"] = effort_override
    elif thinking_mode is True:
        kwargs["thinking"] = {"type": "enabled"}
        if effort_override:
            kwargs["effort"] = effort_override


def _is_always_thinking_anthropic(model: str) -> bool:
    """Return True for Anthropic models where extended thinking is always on.

    These models return a 400 error if ``thinking`` is omitted or set to
    ``disabled``:

    - claude-fable-5, claude-mythos-5
    - claude-opus-4-8, claude-opus-4-7
    """
    _m = (model or "").strip().lower()
    return any(
        _m.startswith(p)
        for p in (
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
        )
    )


class AnthropicClient(LLMClient):
    """
    Anthropic API client for Claude models

    Supported models:
    - claude-3-5-sonnet-20241022
    - claude-3-opus-20240229
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    API_VERSION = "2023-06-01"

    # Minimum system text length to apply prompt caching (avoid overhead for tiny prompts)
    _CACHE_MIN_CHARS = 500
    # Minimum size for an individual cached chunk; smaller chunks are merged
    # into the next block since Anthropic silently ignores markers below ~1024 tokens.
    _CACHE_CHUNK_MIN = 1000
    _provider_label = "Anthropic"

    def get_provider_name(self) -> str:
        return "anthropic"

    @staticmethod
    def _split_system_with_caching(system_text: str) -> list[dict[str, Any]]:
        """Split system prompt into cache-optimized chunks for Anthropic prompt caching.

        Uses ## section headers as natural split points. The general strategy:
        - Chunk 1: everything before the first ## section header — fully static intro
        - Chunk 2: the first ## section (between first and second ## header) — semi-stable
        - Chunk 3: everything from the second ## header onward — uncached (assumed dynamic)

        When fewer than two section headers exist, falls back to a 2-block split
        (chunk 1 cached, rest uncached) or a single uncached block for tiny prompts.

        Special case: when the prompt contains a "## Available Tools" section, that
        marker is used as the second split point instead, keeping the Tools section
        (semi-stable session-level definitions) in the cached zone.
        """
        if len(system_text) < AnthropicClient._CACHE_MIN_CHARS:
            return [{"type": "text", "text": system_text}]

        def _merge_small_cached(head, rest):
            """Merge head chunk into rest[0] if head is too small for caching."""
            if len(head["text"]) < AnthropicClient._CACHE_CHUNK_MIN and rest:
                merged = {"type": "text", "text": head["text"] + "\n\n" + rest[0]["text"]}
                # Preserve cache_control — prefer rest[0]'s marker, fall back to head's
                if "cache_control" in rest[0]:
                    merged["cache_control"] = rest[0]["cache_control"]
                elif "cache_control" in head:
                    merged["cache_control"] = head["cache_control"]
                rest[0] = merged
                return rest
            return [head, *rest]

        tools_marker = "## Available Tools"
        section_marker = "\n## "

        # ── Find the first section header ──────────────────────────────────
        tools_idx = system_text.find(tools_marker)

        if tools_idx != -1:
            # SPECIAL CASE: prompt has an Available Tools section.
            # Use it as the second split point (tools are semi-stable).
            # -> Chunk 1: before Available Tools (identity + core rules)
            # -> Chunk 2: Available Tools section (semi-stable)
            # -> Chunk 3: everything after (session state, intent, ...)
            chunk1 = system_text[:tools_idx].rstrip()
            rest = system_text[tools_idx + len(tools_marker):]
            next_hdr_idx = rest.find(section_marker)
            if next_hdr_idx != -1:
                chunk2_end = tools_idx + len(tools_marker) + next_hdr_idx
                chunk2 = system_text[tools_idx:chunk2_end].rstrip()
                chunk3 = system_text[chunk2_end:].rstrip()
                return _merge_small_cached(
                    {"type": "text", "text": chunk1, "cache_control": {"type": "ephemeral"}},
                    [
                        {"type": "text", "text": chunk2, "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": chunk3},
                    ],
                )
            chunk2 = system_text[tools_idx:].rstrip()
            return _merge_small_cached(
                {"type": "text", "text": chunk1, "cache_control": {"type": "ephemeral"}},
                [{"type": "text", "text": chunk2}],
            )

        # ── GENERAL CASE: no Available Tools section ──────────────────────
        # Find the first ## section header
        first_idx = -1
        if system_text.startswith("## "):
            first_idx = 0
        else:
            first_idx = system_text.find(section_marker)

        if first_idx == -1:
            return [{"type": "text", "text": system_text}]

        # search_from = position in system_text right after the ## marker
        # (3 for leading "## ", 4 for embedded "\n## ")
        header_marker_len = 3 if first_idx == 0 else len(section_marker)
        search_from = first_idx + header_marker_len

        # Find the second ## section header
        second_idx = system_text[search_from:].find(section_marker)

        # Chunk 1: everything before the first section header
        chunk1 = system_text[:first_idx].rstrip() if first_idx > 0 else ""

        if not chunk1:
            # Prompt starts with ## — no intro text before first header
            if second_idx != -1:
                # 2+ headers: first section cached, rest uncached
                chunk_a_end = search_from + second_idx
                chunk_a = system_text[first_idx:chunk_a_end].rstrip()
                chunk_b = system_text[chunk_a_end:].rstrip()
                return _merge_small_cached(
                    {"type": "text", "text": chunk_a, "cache_control": {"type": "ephemeral"}},
                    [{"type": "text", "text": chunk_b}],
                )
            return [{"type": "text", "text": system_text}]

        # Chunk 1 is non-empty — use as cached intro
        if second_idx != -1:
            # Two+ section headers → 3-block split
            chunk2_end = search_from + second_idx
            chunk2 = system_text[first_idx:chunk2_end].rstrip()
            chunk3 = system_text[chunk2_end:].rstrip()
            return _merge_small_cached(
                {"type": "text", "text": chunk1, "cache_control": {"type": "ephemeral"}},
                [
                    {"type": "text", "text": chunk2, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": chunk3},
                ],
            )

        # Only one section header → 2-block split
        chunk2 = system_text[first_idx:].rstrip()
        return _merge_small_cached(
            {"type": "text", "text": chunk1, "cache_control": {"type": "ephemeral"}},
            [{"type": "text", "text": chunk2}],
        )

    @staticmethod
    def _mark_last_message_for_caching(api_messages: list[dict[str, Any]], index: int = -1) -> None:
        """Place an ephemeral cache breakpoint on the content block of the message at ``index``.

        ``index`` defaults to -1 (last message). When the caller appends ephemeral
        tail messages (e.g. work-plan re-injection), pass ``index=-2`` so the
        breakpoint lands on the last *persistent* message. See design_chat_loop.py.

        Anthropic only caches up to an explicit ``cache_control`` breakpoint, and
        the code already marks the system prompt. Adding a *sliding* breakpoint on
        the final message lets each tool-loop turn cache the entire conversation
        prefix, so the next turn re-reads the accumulated history from cache
        instead of at full input price. (DeepSeek caches this automatically via
        server-side prefix matching; Anthropic requires the explicit breakpoint.)

        cache_control must sit on a content *block*, so a plain string content is
        promoted to a single text block. The local ``api_messages`` copy is
        mutated, never the caller's message store. Below Anthropic's minimum
        cacheable size the marker is simply ignored server-side, so this is
        always safe. Uses one breakpoint; combined with the ≤2 system breakpoints
        this stays within Anthropic's limit of 4.
        """
        if not api_messages:
            return
        target = api_messages[index]
        content = target.get("content")
        if isinstance(content, str):
            if not content:
                return  # empty content block is invalid — nothing to cache
            target["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            # Copy the last block (and rebuild the list) so the caller's original
            # message/content objects are never mutated.
            blk = dict(content[-1])
            blk["cache_control"] = {"type": "ephemeral"}
            target["content"] = [*content[:-1], blk]

    def _raise_for_429(self, response: requests.Response, error_code: Optional[int]) -> None:
        """Classify an HTTP 429 and raise the correct error.

        zai/GLM report an EXHAUSTED ACCOUNT BALANCE as HTTP 429 with code 1113
        (or an unambiguous billing phrase). That is a permanent billing failure
        — retrying cannot recharge the account, and surfacing it as a rate
        limit misleads the user into waiting or re-entering a valid key. So a
        balance/quota signal raises the non-retryable ``LLMQuotaExceededError``;
        every other 429 is a genuine transient rate limit.

        Single classification point for all four 429 sites in this client
        (chat / chat_with_tools and their streaming variants), mirroring the
        OpenAI-compatible client so both zai endpoints behave identically.
        """
        suffix = f" code={error_code}" if error_code else ""
        if is_balance_quota_signal(error_code, response.text):
            logger.error(
                f"{self._provider_label} account balance/quota exhausted (429){suffix}"
            )
            raise LLMQuotaExceededError(
                f"{self._provider_label} account balance/quota exhausted (429): "
                "recharge credits or switch providers."
            )
        logger.error(f"{self._provider_label} rate limit exceeded (429){suffix}")
        raise LLMRateLimitError(
            f"{self._provider_label} rate limit exceeded. Please try again later.",
            retry_after=parse_retry_after(response.headers),
            error_code=error_code,
        )
    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send message request to Anthropic

        API Reference: https://docs.anthropic.com/claude/reference/messages_post
        """
        if not model:
            model = self.DEFAULT_MODEL

        if not max_tokens:
            max_tokens = _cfg.tokens.ANTHROPIC_DEFAULT  # Anthropic requires max_tokens

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json"
        }

        # Anthropic requires system message separate
        system_content = None
        api_messages = []

        for msg in messages:
            _dict = isinstance(msg, dict)
            _role = msg.role if not _dict else msg.get("role", "")
            _content = msg.content if not _dict else msg.get("content", "")
            if _role == "system":
                if system_content is None:
                    system_content = _content
                else:
                    system_content = system_content + "\n\n" + _content
            else:
                api_messages.append({"role": _role, "content": _content})

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }

        if system_content:
            payload["system"] = self._split_system_with_caching(system_content)

        # Add any extra kwargs
        # NOTE: reasoning_callback is intentionally NOT consumed here. OpenAI
        # o-series and Anthropic extended-thinking both suppress reasoning
        # content from the UI reasoning panel (closed-model policy: the model's
        # internal chain-of-thought is not surfaced). Only providers.py clients
        # (DeepSeek / ZAI-GLM / Ollama) consume reasoning_callback. This mirrors
        # openai_client.py:313,472. Do NOT "fix" this by forwarding — it would
        # break parity between the two closed-thinking providers.
        kwargs.pop("reasoning_callback", None)
        _effort = kwargs.pop("reasoning_effort", None)
        thinking_mode = kwargs.pop("thinking_mode", None)
        _always_think = _is_always_thinking_anthropic(model)

        if _always_think:
            # Fable 5, Mythos 5, Opus 4.8, Opus 4.7: thinking always on — must include
            payload["thinking"] = {"type": "adaptive"}
            if _effort:
                payload["effort"] = _effort
            elif thinking_mode is not None:
                payload["effort"] = "high" if thinking_mode else "low"
        elif thinking_mode is True:
            payload["thinking"] = {"type": "adaptive"}
            if _effort:
                payload["effort"] = _effort
        else:
            if temperature is not None:
                payload["temperature"] = temperature
        token_callback = kwargs.pop("token_callback", None)
        payload.update(kwargs)

        # Streaming path: when token_callback is provided, use SSE streaming so
        # text_delta chunks are forwarded in real-time. This lets the final
        # summary (plain chat() call, no tools) stream incrementally instead of
        # blocking until the whole response is buffered — which would leave the
        # "thinking" ticker as the only visible UI for the entire call duration.
        # Mirrors chat_with_tools() L631-634 streaming gate. The parser is a
        # simplified version of _chat_with_tools_streaming (no tool_use blocks).
        if token_callback is not None:
            return self._chat_streaming(url, headers, payload, model, token_callback)

        t0 = time.monotonic()

        try:
            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            elapsed_ms = (time.monotonic() - t0) * 1000

            # Handle specific error codes
            if response.status_code == 401:
                logger.error(f"{self._provider_label} authentication failed (401)")
                raise LLMAuthenticationError(
                    f"Invalid {self._provider_label} API key. "
                    "Please check your ANTHROPIC_API_KEY environment variable."
                )

            if response.status_code == 429:
                self._raise_for_429(response, _parse_glm_error_code(response))

            if response.status_code >= 500 and response.status_code != 501:
                error_body = response.text[:500]
                logger.error(
                    f"{self._provider_label} API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMServerUnavailableError(
                    f"{self._provider_label} API returned HTTP {response.status_code}: {error_body}"
                )

            if response.status_code != 200:
                error_body = response.text[:500]
                logger.error(
                    f"{self._provider_label} API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMAPIError(
                    f"{self._provider_label} API returned HTTP {response.status_code}: {error_body}"
                )

            data = response.json()

            # Extract content
            content_blocks = data.get("content", [])
            if not content_blocks:
                logger.warning(f"{self._provider_label} response has no content blocks")
                return LLMResponse(
                    content="",
                    model=model,
                    provider=self.get_provider_name(),
                    raw_response=data
                )

            # Combine all text blocks
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")

            finish_reason = data.get("stop_reason")

            # Extract token usage
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            tokens_used = input_tokens + output_tokens

            logger.info(
                f"{self._provider_label} %s: %.0fms, tok=%d (in=%d, out=%d), finish_reason=%s",
                model, elapsed_ms, tokens_used, input_tokens, output_tokens, finish_reason
            )

            return LLMResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=data
            )

        except requests.ConnectionError as e:
            logger.error("Cannot connect to Anthropic API: %s", e)
            raise LLMConnectionError(
                "Cannot connect to Anthropic API. "
                "Please check your internet connection."
            ) from e

        except requests.Timeout as e:
            logger.error(f"{self._provider_label} request timed out after %ds", self.timeout)
            raise LLMConnectionError(
                f"{self._provider_label} request timed out after {self.timeout}s"
            ) from e

        except requests.RequestException as e:
            logger.error(f"{self._provider_label} request failed: %s", e)
            raise LLMAPIError(f"{self._provider_label} request failed: {e}") from e

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        **kwargs
    ) -> ToolCallResponse:
        """
        Send message request to Anthropic with tool use support.

        Uses Anthropic tool_use API.
        When token_callback is provided (via kwargs), streams text tokens in real-time.
        Tool-call inputs are buffered and not forwarded to the callback.
        """
        if not model:
            model = self.DEFAULT_MODEL

        max_tokens = kwargs.pop("max_tokens", _cfg.tokens.ANTHROPIC_DEFAULT)
        token_callback = kwargs.pop("token_callback", None)

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        # Separate system messages
        system_content = None
        api_messages = []
        for msg in messages:
            _dict = isinstance(msg, dict)
            _role = msg.role if not _dict else msg.get("role", "")
            _content = msg.content if not _dict else msg.get("content", "")
            if _role == "system":
                if system_content is None:
                    system_content = _content
                else:
                    system_content = system_content + "\n\n" + _content
            elif _role == "tool":
                # Single tool result: wrap in tool_result content block
                tool_use_id = getattr(msg, "tool_call_id", None) or ""
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": _content}],
                })
            elif _role == "assistant" and (tc_list := getattr(msg, "tool_calls", None)):
                # Prefer provider-native content blocks when available — these
                # include the `thinking` block (with signature) that Anthropic
                # extended-thinking requires to be echoed back on the next turn.
                # Without it, the API rejects with HTTP 400 on multi-turn.
                _raw_blocks = getattr(msg, "raw_content", None)
                if isinstance(_raw_blocks, list) and _raw_blocks:
                    api_messages.append({"role": "assistant", "content": _raw_blocks})
                    continue
                # Convert OpenAI-style tool_calls to Anthropic tool_use content blocks.
                # Without this, tool_calls are silently dropped, resulting in orphaned
                # tool_result blocks (no preceding tool_use). Anthropic API rejects with
                # HTTP 400: "unexpected tool_use_id found in tool_result blocks".
                content_blocks: list[dict[str, Any]] = []
                if _content:
                    content_blocks.append({"type": "text", "text": _content})
                for tc in tc_list:
                    tc_id = tc.get("id", "")
                    fn = tc.get("function", {})
                    try:
                        _input = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        _input = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": fn.get("name", ""),
                        "input": _input,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                raw_content = getattr(msg, "raw_content", None)
                images = getattr(msg, "images", None)
                if raw_content:
                    # Preserve native Anthropic content blocks (tool_use / tool_result)
                    api_messages.append({"role": _role, "content": raw_content})
                elif images:
                    # Multimodal: image blocks + text
                    content_blocks: list[dict[str, Any]] = []
                    for img in images:
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": img["media_type"],
                                "data": img["data"],
                            },
                        })
                    content_blocks.append({"type": "text", "text": _content})
                    api_messages.append({"role": _role, "content": content_blocks})
                else:
                    api_messages.append({"role": _role, "content": _content})

        # Convert tool schemas to Anthropic format
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        # Sliding cache breakpoint on the final message so each tool-loop turn
        # caches the growing conversation prefix (covers both the streaming and
        # non-streaming paths below, which share this payload).
        # When the caller appended ephemeral tail messages (e.g. work-plan
        # re-injection), cache_breakpoint_offset > 0 shifts the breakpoint
        # backward by that many messages, landing on the last persistent one.
        _cache_offset = kwargs.pop("cache_breakpoint_offset", 0)
        self._mark_last_message_for_caching(api_messages, -1 - _cache_offset)

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "tools": anthropic_tools,
        }
        if system_content:
            payload["system"] = self._split_system_with_caching(system_content)
        # reasoning_callback intentionally NOT consumed — see chat() L344 note.
        kwargs.pop("reasoning_callback", None)
        _effort = kwargs.pop("reasoning_effort", None)
        thinking_mode = kwargs.pop("thinking_mode", None)
        _always_think = _is_always_thinking_anthropic(model)

        if _always_think:
            # Fable 5, Mythos 5, Opus 4.8, Opus 4.7: thinking always on — must include
            payload["thinking"] = {"type": "adaptive"}
            kwargs.pop("temperature", None)
            if _effort:
                payload["effort"] = _effort
            elif thinking_mode is not None:
                payload["effort"] = "high" if thinking_mode else "low"
        elif thinking_mode is True:
            payload["thinking"] = {"type": "adaptive"}
            kwargs.pop("temperature", None)
            if _effort:
                payload["effort"] = _effort
        payload.update(kwargs)

        # Use streaming when token_callback is provided
        if token_callback is not None:
            return self._chat_with_tools_streaming(
                url, headers, payload, model, token_callback,
            )

        t0 = time.monotonic()
        try:
            response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code == 401:
                raise LLMAuthenticationError(f"Invalid {self._provider_label} API key.")
            if response.status_code == 429:
                self._raise_for_429(response, _parse_glm_error_code(response))
            if response.status_code >= 500 and response.status_code != 501:
                error_body = response.text[:500]
                raise LLMServerUnavailableError(
                    f"{self._provider_label} API returned HTTP {response.status_code}: "
                    f"{error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(f"{self._provider_label} API returned HTTP {response.status_code}: {error_body}")

            data = response.json()
            content_blocks = data.get("content", [])
            stop_reason = data.get("stop_reason")

            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens")
            completion_tokens = usage.get("output_tokens")
            tokens_used = (prompt_tokens or 0) + (completion_tokens or 0) or None
            cache_read_input_tokens = usage.get("cache_read_input_tokens")
            cache_creation_input_tokens = usage.get("cache_creation_input_tokens")

            # Parse content blocks
            text_content = ""
            tool_calls: list[ToolCallRequest] = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append(ToolCallRequest(
                        call_id=block.get("id", ""),
                        name=block.get("name", ""),
                        args=block.get("input", {}),
                    ))

            is_final = stop_reason == "end_turn" or not tool_calls
            logger.info(
                f"{self._provider_label} %s (tools): %.0fms, tok=%d, stop=%s (%d)",
                model, elapsed_ms, tokens_used, stop_reason, len(tool_calls),
            )
            return ToolCallResponse(
                content=text_content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=stop_reason,
                raw_response=data,
                tool_calls=tool_calls,
                is_final=is_final,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
            )

        except requests.ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Anthropic API.") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"{self._provider_label} request timed out after {self.timeout}s") from e
        except requests.RequestException as e:
            raise LLMAPIError(f"{self._provider_label} request failed: {e}") from e

    def _chat_streaming(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        token_callback,
    ) -> LLMResponse:
        """Streaming variant of chat() (no tools). Forwards text_delta tokens.

        Simplified version of _chat_with_tools_streaming: no tool_use blocks,
        no input_json_delta. Extended-thinking blocks are accumulated into
        raw_response so multi-turn echo still works. Returns LLMResponse (not
        ToolCallResponse) to match chat()'s contract.
        """
        import json as _json

        stream_payload = dict(payload)
        stream_payload["stream"] = True

        t0 = time.monotonic()
        try:
            response = self._session.post(
                url, headers=headers, json=stream_payload,
                timeout=self.timeout, stream=True,
            )
        except requests.ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Anthropic API.") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"{self._provider_label} request timed out after {self.timeout}s") from e

        # Status checks inside try/finally (matches the providers.py streaming
        # pattern): a non-200 streaming response, or a failure while draining
        # its error body via .text, must still close the connection.
        try:
            if response.status_code == 401:
                raise LLMAuthenticationError(f"Invalid {self._provider_label} API key.")
            if response.status_code == 429:
                self._raise_for_429(response, _parse_glm_error_code(response))
            if response.status_code >= 500 and response.status_code != 501:
                error_body = response.text[:500]
                raise LLMServerUnavailableError(
                    f"{self._provider_label} API returned HTTP {response.status_code}: "
                    f"{error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(f"{self._provider_label} API returned HTTP {response.status_code}: {error_body}")

            text_content = ""
            stop_reason = None
            prompt_tokens = None
            completion_tokens = None
            cache_read_input_tokens = None
            cache_creation_input_tokens = None

            _current_block_type: Optional[str] = None
            _is_streaming_text = False
            _thinking_blocks: list[dict[str, Any]] = []
            _current_thinking: Optional[dict[str, Any]] = None

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

                ev_type = ev.get("type", "")

                if ev_type == "content_block_start":
                    block = ev.get("content_block", {})
                    _current_block_type = block.get("type")
                    if _current_block_type == "text":
                        _is_streaming_text = True
                    elif _current_block_type == "thinking":
                        _current_thinking = {"thinking": "", "signature": ""}
                        _is_streaming_text = False
                    else:
                        _is_streaming_text = False

                elif ev_type == "content_block_delta":
                    delta = ev.get("delta", {})
                    delta_type = delta.get("type", "")
                    if delta_type == "text_delta":
                        chunk = delta.get("text", "")
                        text_content += chunk
                        if _is_streaming_text and chunk:
                            try:
                                token_callback(chunk)
                            except Exception:
                                pass
                    elif delta_type == "thinking_delta" and _current_thinking is not None:
                        _current_thinking["thinking"] += delta.get("thinking", "")
                    elif delta_type == "signature_delta" and _current_thinking is not None:
                        _current_thinking["signature"] += delta.get("signature", "")

                elif ev_type == "content_block_stop":
                    if _current_thinking is not None:
                        _thinking_blocks.append({
                            "type": "thinking",
                            "thinking": _current_thinking.get("thinking", ""),
                            "signature": _current_thinking.get("signature", ""),
                        })
                        _current_thinking = None
                    _is_streaming_text = False

                elif ev_type == "message_delta":
                    delta = ev.get("delta", {})
                    if delta.get("stop_reason"):
                        stop_reason = delta.get("stop_reason")
                    usage = ev.get("usage", {})
                    if usage:
                        if usage.get("input_tokens"):
                            prompt_tokens = usage.get("input_tokens")
                        if usage.get("output_tokens"):
                            completion_tokens = usage.get("output_tokens")
                        if usage.get("cache_read_input_tokens") is not None:
                            cache_read_input_tokens = usage.get("cache_read_input_tokens")
                        if usage.get("cache_creation_input_tokens") is not None:
                            cache_creation_input_tokens = usage.get("cache_creation_input_tokens")

                elif ev_type == "message_start":
                    msg = ev.get("message", {})
                    usage = msg.get("usage", {})
                    if usage:
                        if usage.get("input_tokens"):
                            prompt_tokens = usage.get("input_tokens")
                        if usage.get("cache_read_input_tokens") is not None:
                            cache_read_input_tokens = usage.get("cache_read_input_tokens")
                        if usage.get("cache_creation_input_tokens") is not None:
                            cache_creation_input_tokens = usage.get("cache_creation_input_tokens")

        except requests.RequestException as e:
            raise LLMAPIError(f"{self._provider_label} streaming request failed: {e}") from e
        finally:
            response.close()

        elapsed_ms = (time.monotonic() - t0) * 1000

        _pt = prompt_tokens or 0
        _ct = completion_tokens or 0
        tokens_used = _pt + _ct

        logger.info(
            f"{self._provider_label} %s (stream): %.0fms, tok=%d (in=%d, out=%d), finish_reason=%s",
            model, elapsed_ms, tokens_used, _pt, _ct, stop_reason
        )

        # Reconstruct raw_response so multi-turn echo preserves thinking blocks.
        content_blocks: list[dict[str, Any]] = []
        for tb in _thinking_blocks:
            content_blocks.append(dict(tb))
        if text_content:
            content_blocks.append({"type": "text", "text": text_content})
        raw_response = {
            "id": "stream",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": _pt,
                "output_tokens": _ct,
                **({"cache_read_input_tokens": cache_read_input_tokens} if cache_read_input_tokens is not None else {}),
                **({"cache_creation_input_tokens": cache_creation_input_tokens} if cache_creation_input_tokens is not None else {}),
            },
        }

        return LLMResponse(
            content=text_content,
            model=model,
            provider=self.get_provider_name(),
            tokens_used=tokens_used,
            finish_reason=stop_reason,
            raw_response=raw_response,
            prompt_tokens=_pt or None,
            completion_tokens=_ct or None,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    def _chat_with_tools_streaming(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        token_callback,
    ) -> ToolCallResponse:
        """Streaming variant: forwards text_delta tokens via token_callback.

        Tool-call inputs are buffered silently (not forwarded).
        Only the final text response (is_final=True) triggers the callback;
        intermediate text alongside tool calls is also forwarded.

        Extended-thinking blocks (thinking / signature deltas) are accumulated
        and passed through in raw_response so multi-turn echo works correctly.
        """
        import json as _json

        stream_payload = dict(payload)
        stream_payload["stream"] = True

        t0 = time.monotonic()
        try:
            response = self._session.post(
                url, headers=headers, json=stream_payload,
                timeout=self.timeout, stream=True,
            )
        except requests.ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Anthropic API.") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"{self._provider_label} request timed out after {self.timeout}s") from e

        # Status checks inside try/finally (see _chat_streaming above): ensures
        # the streaming response is closed on a non-200 raise.
        try:
            if response.status_code == 401:
                raise LLMAuthenticationError(f"Invalid {self._provider_label} API key.")
            if response.status_code == 429:
                self._raise_for_429(response, _parse_glm_error_code(response))
            if response.status_code >= 500 and response.status_code != 501:
                error_body = response.text[:500]
                raise LLMServerUnavailableError(
                    f"{self._provider_label} API returned HTTP {response.status_code}: "
                    f"{error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(f"{self._provider_label} API returned HTTP {response.status_code}: {error_body}")

            # Parse SSE stream
            text_content = ""
            tool_calls: list[ToolCallRequest] = []
            stop_reason = None
            prompt_tokens = None
            completion_tokens = None
            cache_read_input_tokens = None
            cache_creation_input_tokens = None

            # State for current content block
            _current_block_type: Optional[str] = None  # "text", "tool_use", or "thinking"
            _current_tool: Optional[dict[str, Any]] = None  # id/name/input_json accumulator
            _is_streaming_text = False  # True when we're in a text block with token_callback active

            # thinking block accumulator (for reconstructing raw_response with extended thinking)
            _thinking_blocks: list[dict[str, Any]] = []
            _current_thinking: Optional[dict[str, Any]] = None  # {thinking: str, signature: str}

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

                ev_type = ev.get("type", "")

                if ev_type == "content_block_start":
                    block = ev.get("content_block", {})
                    _current_block_type = block.get("type")
                    if _current_block_type == "tool_use":
                        _current_tool = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input_json": "",
                        }
                        _is_streaming_text = False
                    elif _current_block_type == "text":
                        _is_streaming_text = True
                    elif _current_block_type == "thinking":
                        _current_thinking = {"thinking": "", "signature": ""}
                        _is_streaming_text = False

                elif ev_type == "content_block_delta":
                    delta = ev.get("delta", {})
                    delta_type = delta.get("type", "")
                    if delta_type == "text_delta":
                        chunk = delta.get("text", "")
                        text_content += chunk
                        if _is_streaming_text and chunk:
                            try:
                                token_callback(chunk)
                            except Exception:
                                pass
                    elif delta_type == "input_json_delta" and _current_tool is not None:
                        _current_tool["input_json"] += delta.get("partial_json", "")
                    elif delta_type == "thinking_delta" and _current_thinking is not None:
                        _current_thinking["thinking"] += delta.get("thinking", "")
                    elif delta_type == "signature_delta" and _current_thinking is not None:
                        _current_thinking["signature"] += delta.get("signature", "")

                elif ev_type == "content_block_stop":
                    if _current_block_type == "tool_use" and _current_tool:
                        try:
                            args = _json.loads(_current_tool["input_json"] or "{}")
                        except Exception:
                            args = {}
                        tool_calls.append(ToolCallRequest(
                            call_id=_current_tool["id"],
                            name=_current_tool["name"],
                            args=args,
                        ))
                    elif _current_block_type == "thinking" and _current_thinking:
                        _thinking_blocks.append(dict(_current_thinking))
                    _current_block_type = None
                    _current_tool = None
                    _current_thinking = None
                    _is_streaming_text = False

                elif ev_type == "message_delta":
                    stop_reason = ev.get("delta", {}).get("stop_reason") or stop_reason
                    usage = ev.get("usage", {})
                    if usage.get("output_tokens"):
                        completion_tokens = usage["output_tokens"]

                elif ev_type == "message_start":
                    usage = ev.get("message", {}).get("usage", {})
                    prompt_tokens = usage.get("input_tokens")
                    cache_read_input_tokens = usage.get("cache_read_input_tokens")
                    cache_creation_input_tokens = usage.get("cache_creation_input_tokens")

        except requests.ConnectionError as e:
            raise LLMConnectionError(f"{self._provider_label} streaming connection lost: {e}") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"{self._provider_label} streaming timed out: {e}") from e
        except requests.RequestException as e:
            raise LLMAPIError(f"{self._provider_label} streaming request failed: {e}") from e
        finally:
            response.close()

        elapsed_ms = (time.monotonic() - t0) * 1000
        tokens_used = (prompt_tokens or 0) + (completion_tokens or 0) or None
        is_final = stop_reason == "end_turn" or not tool_calls

        # Reconstruct raw_response content array so extended-thinking blocks
        # are preserved for echo in subsequent multi-turn calls.
        _raw_content: list[dict[str, Any]] = []
        for tb in _thinking_blocks:
            _raw_content.append(
                {"type": "thinking", "thinking": tb["thinking"], "signature": tb["signature"]}
            )
        if text_content:
            _raw_content.append({"type": "text", "text": text_content})
        for tc in tool_calls:
            _raw_content.append(
                {"type": "tool_use", "id": tc.call_id, "name": tc.name, "input": tc.args}
            )

        logger.info(
            f"{self._provider_label} %s (tools): %.0fms, tok=%s, stop=%s (%d)",
            model, elapsed_ms, tokens_used, stop_reason, len(tool_calls),
        )
        return ToolCallResponse(
            content=text_content,
            model=model,
            provider=self.get_provider_name(),
            tokens_used=tokens_used,
            finish_reason=stop_reason,
            raw_response={"content": _raw_content} if _raw_content else {},
            tool_calls=tool_calls,
            is_final=is_final,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )


class ZAIAnthropicClient(AnthropicClient):
    """
    Z.AI (GLM) API client — Anthropic-compatible protocol.

    Uses z.ai's Anthropic-compatible endpoint (https://api.z.ai/api/anthropic/v1)
    instead of the OpenAI-compatible one (https://api.z.ai/api/coding/paas/v4).

    Supports the same GLM models as ZAIClient but via Anthropic Messages API format.
    This allows compatibility with tools that speak Anthropic protocol (e.g. Claude Code).

    Supported models:
    - glm-5.2
    - glm-5.1, glm-5, glm-5-turbo
    - glm-4.7, glm-4.6, glm-4.5, etc.

    Thinking/reasoning mode:
    GLM-5.2+ thinks by default. To disable: pass thinking_mode=False.
    """
    DEFAULT_BASE_URL = "https://api.z.ai/api/anthropic/v1"
    DEFAULT_MODEL = "glm-5.2"
    _provider_label = "Z.AI"

    def get_provider_name(self) -> str:
        return "zai"

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
        _always_think = _is_always_thinking_glm(model)

        _inject_glm_thinking_kwargs(kwargs, thinking_mode, _effort_override, _always_think)

        # When thinking is active, omit temperature (mutually exclusive).
        # Note: thinking={"type":"disabled"} still means thinking is OFF, so
        # temperature must be sent in that case.
        _thinking_on = "thinking" in kwargs and kwargs["thinking"].get("type") != "disabled"
        _temp_val = None if _thinking_on else temperature
        resp = super().chat(messages, model, temperature=_temp_val, max_tokens=max_tokens, **kwargs)

        # Z.AI context cache diagnostics (if the Anthropic endpoint returns cache info)
        if resp.raw_response:
            usage = resp.raw_response.get("usage", {})
            if isinstance(usage, dict):
                cached_tokens = usage.get("cache_read_input_tokens")
                if cached_tokens is not None:
                    prompt_tokens = usage.get("input_tokens", 0)
                    cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
                    logger.info(
                        "Z.AI cache: %s cached tokens (%.0f%% of prompt -- Coding Plan: no discount)",
                        cached_tokens, cache_pct,
                    )

        return resp

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        max_tokens: Optional[int] = None,
        token_callback: Optional[Callable] = None,
        **kwargs
    ) -> ToolCallResponse:
        # Pop z.ai-specific params before passing to parent
        thinking_mode = kwargs.pop("thinking_mode", None)
        _effort_override = kwargs.pop("reasoning_effort", None)
        _always_think = _is_always_thinking_glm(model)

        _inject_glm_thinking_kwargs(kwargs, thinking_mode, _effort_override, _always_think)

        # z.ai's Anthropic endpoint rejects a temperature with >2 decimal places
        # (HTTP 400, code 1210 "temperature parameter is illegal: limited to 2
        # decimal places") and treats thinking+temperature as mutually exclusive.
        # chat() already handles this; chat_with_tools must mirror it, otherwise a
        # high-precision temperature (e.g. random.uniform()) silently 400s every
        # tool-bearing call — which the design-chat loop then degrades to a
        # tool-LESS plain chat, manifesting as "the model talks but never calls a
        # tool." Drop temperature when thinking is on; otherwise quantize it.
        _thinking_on = "thinking" in kwargs and kwargs["thinking"].get("type") != "disabled"
        if _thinking_on:
            kwargs.pop("temperature", None)
        elif kwargs.get("temperature") is not None:
            kwargs["temperature"] = round(kwargs["temperature"], 2)

        resp = super().chat_with_tools(
            messages, tools, model=model,
            max_tokens=max_tokens, token_callback=token_callback,
            **kwargs
        )

        # Z.AI context cache diagnostics
        if resp.raw_response:
            usage = resp.raw_response.get("usage", {})
            if isinstance(usage, dict):
                cached_tokens = usage.get("cache_read_input_tokens")
                if cached_tokens is not None:
                    prompt_tokens = usage.get("input_tokens", 0)
                    cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
                    logger.info(
                        "Z.AI cache: %s cached tokens (%.0f%% of prompt -- Coding Plan: no discount)",
                        cached_tokens, cache_pct,
                    )

        return resp
