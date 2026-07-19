"""
Google (Gemini), DeepSeek, and Ollama clients for asicode
"""
from __future__ import annotations

import json as _json
import logging
import time
from typing import Any, Optional

import requests

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
    parse_retry_after,
)
from .output_parser import parse_tool_args

logger = logging.getLogger(__name__)
# ── Gemini finish_reason normalization ────────────────────────────────────
# Google's Gemini API returns UPPERCASE finishReason values (STOP, MAX_TOKENS, SAFETY, etc.)
# while all consumers (agent_loop, intent_resolver, agent_phase_manager) check lowercase
# OpenAI-style values (stop, length). Normalize at the provider boundary.

_GEMINI_FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "FINISH_REASON_UNSPECIFIED": "stop",
    "OTHER": "stop",
}

def _normalize_gemini_finish_reason(raw: Optional[str]) -> Optional[str]:
    """Map Gemini UPPERCASE finishReason to the lowercase OpenAI convention.

    All consumers (agent_loop, intent_resolver, agent_phase_manager) check for
    "length" to detect truncation. Without this normalization, Google's "MAX_TOKENS"
    silently bypasses the retry-on-truncation logic.
    """
    if raw is None:
        return None
    return _GEMINI_FINISH_REASON_MAP.get(str(raw).strip(), str(raw).strip().lower())


# ── Ollama vision helpers ─────────────────────────────────────────────────────

def _is_ollama_vision_model(model: str) -> bool:
    """Return True if the model name suggests multimodal (vision) support."""
    from external_llm.model_registry import ollama_vision
    return ollama_vision(model)


def _is_gpt_oss(model: str) -> bool:
    """Return True if the model is GPT-OSS (needs string think values, not boolean)."""
    return "gpt-oss" in (model or "").lower()


def _ollama_think_value(model: str, thinking_mode: Optional[bool],
                        reasoning_effort: Optional[str] = None) -> Any:
    """Compute the 'think' value for Ollama API.

    GPT-OSS requires string levels ('low'|'medium'|'high') instead of boolean.
    Other models use standard boolean.
    """
    if _is_gpt_oss(model):
        # GPT-OSS: think=False is ignored — map to "low" (minimal trace)
        if not thinking_mode:
            return "low"
        # Map reasoning_effort to GPT-OSS levels
        if reasoning_effort in ("high", "max"):
            return "high"
        if reasoning_effort == "low":
            return "low"
        return "medium"  # default when thinking is on
    # Standard models: boolean
    return bool(thinking_mode) if thinking_mode is not None else None


def _normalize_ollama_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge all ``system``-role messages into a single system message at index 0.

    Some Ollama chat templates (notably the Qwen3 family, e.g. ``bonsai27b``)
    enforce that a ``system`` message may ONLY appear as the very first message.
    A second system message — even a bare divider line — makes the Jinja
    template raise an exception and the whole request is rejected with HTTP 400::

        raise_exception('System message must be at the beginning.')

    The agent's context builder emits several consecutive ``system`` messages
    (core prompt, divider, repo root, project context, design insights, ...).
    Cloud providers tolerate multiple system messages, but strict local
    templates do not, so this normalisation is Ollama-specific.

    Collapses every ``system`` message into one at the front (content joined by
    newlines), preserving the relative order of all non-system messages. Safe
    no-op when there are 0 or 1 system messages.
    """
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    first_sys: Optional[dict[str, Any]] = None
    sys_count = 0
    for m in messages:
        if m.get("role") == "system":
            sys_count += 1
            content = m.get("content")
            if content:
                system_parts.append(content)
            if first_sys is None:
                first_sys = m
        else:
            rest.append(m)
    # No system at all, or exactly one already at index 0 → already valid.
    if sys_count <= 1 and (not messages or messages[0].get("role") == "system"):
        return messages
    if not system_parts:
        return messages
    merged: dict[str, Any] = {"role": "system", "content": "\n".join(system_parts)}
    # Preserve non-payload keys (e.g. 'images') from the first system message.
    if first_sys:
        for k, v in first_sys.items():
            if k not in ("role", "content"):
                merged[k] = v
    return [merged] + rest


def _is_gemini_3(model: str) -> bool:
    """Return True for Gemini 3 series models that use thinkingLevel (not thinkingBudget).

    Gemini 3+ uses ``thinkingConfig.thinkingLevel`` whereas 2.5 series uses
    ``thinkingConfig.thinkingBudget``.  The two cannot be mixed in one request.
    """
    _m = (model or "").strip().lower()
    return _m.startswith("gemini-3")


def _detect_image_ocr_lang(b64_data: str) -> tuple:
    """Detect OCR language from image data by trying language packs.

    Tries 'kor+eng' first (most common for this project), then 'eng' alone
    as fallback. Returns empty string if no text found with any language.
    """
    img_w = img_h = 0
    try:
        import base64 as _b64
        import io as _io

        import pytesseract as _tess
        from PIL import Image as _PILImage
        from pytesseract import Output as _Out

        img = _PILImage.open(_io.BytesIO(_b64.b64decode(b64_data)))
        img_w, img_h = img.size

        # Priority order: most likely first
        for lang in ("kor+eng", "eng", "chi_sim+eng", "jpn+eng"):
            data = _tess.image_to_data(img, lang=lang, output_type=_Out.DICT)
            texts = [
                (data["text"][i] or "").strip()
                for i in range(len(data["text"]))
                if int(data["conf"][i]) >= 20
            ]
            joined = " ".join(texts).strip()
            if joined:
                return lang, img_w, img_h, data
        return "eng", img_w, img_h, None
    except ImportError:
        return "eng", img_w, img_h, None
    except Exception:
        return "eng", img_w, img_h, None


def _try_ocr_base64(b64_data: str) -> str:
    """Extract text WITH positional labels from a base64-encoded image.

    Uses pytesseract image_to_data() to get per-word bounding boxes, then
    groups words into lines and annotates each line with a spatial label
        (top-left / top-center / top-right / middle-* / bottom-*).

    Example output:
        [Image OCR — Position Includes (1456×800px):
            [top-center] asicode Idle
            [top-right] Providers Settings
            [middle-left] DESIGN Chat — discuss design with AI
            [bottom-right] SEND RUN
        ]

    Returns empty string if pytesseract / Pillow is not installed or OCR fails.
    """
    if not b64_data:
        return ""

    _lang, img_w, img_h, data = _detect_image_ocr_lang(b64_data)
    if data is None or not data.get('text'):
        return ""

    # Collect words with sufficient confidence
    words = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = int(data["conf"][i])
        if not text or conf < 20:
            continue
        words.append({
            "text": text,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
        })

    if not words:
        return ""

    # Group words into lines: new line when y jumps by more than avg word height
    words.sort(key=lambda w: (w["y"], w["x"]))
    avg_word_h = sum(w["h"] for w in words) / len(words)
    line_thresh = max(8, avg_word_h * 0.6)

    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_y: float = words[0]["y"]
    for word in words:
        if abs(word["y"] - cur_y) <= line_thresh:
            cur.append(word)
            cur_y = sum(w["y"] for w in cur) / len(cur)  # rolling mean
        else:
            if cur:
                lines.append(sorted(cur, key=lambda w: w["x"]))
            cur = [word]
            cur_y = float(word["y"])
    if cur:
        lines.append(sorted(cur, key=lambda w: w["x"]))

    # Build output with spatial labels
    result = [f"[Image OCR — Position Includes ({img_w}×{img_h}px):"]
    for line_words in lines:
        line_text = " ".join(w["text"] for w in line_words)
        avg_cx = sum(w["x"] + w["w"] / 2 for w in line_words) / len(line_words)
        avg_ty = sum(w["y"] for w in line_words) / len(line_words)

        v = "top" if avg_ty < img_h * 0.30 else ("bottom" if avg_ty > img_h * 0.70 else "middle")
        h = "left" if avg_cx < img_w * 0.33 else ("right" if avg_cx > img_w * 0.67 else "center")
        result.append(f"  [{v}-{h}] {line_text}")

    result.append("]")
    return "\n".join(result)



def _images_to_text(images: list[dict[str, str]]) -> str:
    """Convert attached images to inline text for non-vision Ollama models.

    Tries OCR first; falls back to a placeholder marker so the LLM at least
    knows an image was attached.
    """
    parts = []
    for i, img in enumerate(images, 1):
        ocr_text = _try_ocr_base64(img.get("data", ""))
        if ocr_text:
            parts.append(f"[Image {i} — OCR Extracted Text:\n{ocr_text}\n]")
        else:
            media_type = img.get("media_type", "image")
            parts.append(
                f"[Image {i} Attached ({media_type}). "
                "Text extraction requires pytesseract+Pillow — OCR applied automatically when installed]"
            )
    return "\n".join(parts)


class GoogleClient(LLMClient):
    """
    Google Gemini API client

    Supported models:
    - gemini-2.5-flash (recommended)
    - gemini-2.5-pro
    - gemini-2.0-flash
    - gemini-2.0-flash-001
    - gemini-2.0-flash-lite-001
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def get_provider_name(self) -> str:
        return "google"

    @staticmethod
    def _build_gemini_thinking_config(
        model: str,
        thinking_mode: Optional[bool],
        reasoning_effort: Optional[str],
    ) -> dict[str, Any]:
        """Build Gemini thinkingConfig dict based on model version and thinking_mode.

        Gemini 3+ uses thinkingLevel (minimal/low/high).
        Gemini 2.5 uses thinkingBudget (-1 = dynamic, 0 = off, omit = on).
        """
        if _is_gemini_3(model):
            if thinking_mode is not None:
                if not thinking_mode:
                    _level = "minimal"
                elif reasoning_effort in ("high", "max"):
                    _level = "high"
                elif reasoning_effort == "low":
                    _level = "low"
                else:
                    _level = "high"
                return {"thinkingLevel": _level}
        else:
            if thinking_mode is True:
                return {"thinkingBudget": -1}
            elif thinking_mode is False and "pro" not in model.lower():
                return {"thinkingBudget": 0}
        return {}

    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send request to Google Gemini

        API Reference: https://ai.google.dev/api/rest
        """
        if not model:
            model = self.DEFAULT_MODEL

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={self.api_key}"

        headers = {"Content-Type": "application/json"}

        # Convert to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                # Gemini uses "user" and "model" roles
                role = "model" if msg.role == "assistant" else "user"
                images = getattr(msg, "images", None)
                if images:
                    parts: list[dict[str, Any]] = [{"text": msg.content or ""}]
                    for img in images:
                        parts.append({
                            "inlineData": {
                                "mimeType": img.get("media_type", "image/png"),
                                "data": img.get("data", ""),
                            }
                        })
                    contents.append({"role": role, "parts": parts})
                else:
                    contents.append({"role": role, "parts": [{"text": msg.content}]})

        # Consume non-serializable kwargs before building generationConfig
        kwargs.pop("reasoning_callback", None)
        kwargs.pop("token_callback", None)
        _reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_mode = kwargs.pop("thinking_mode", None)

        generation_config: dict[str, Any] = {
            "temperature": temperature,
        }
        thinking_cfg = self._build_gemini_thinking_config(model, thinking_mode, _reasoning_effort)
        if thinking_cfg:
            generation_config["thinkingConfig"] = thinking_cfg

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Merge remaining kwargs (e.g. safetySettings, top_p, etc.)
        payload.update(kwargs)

        t0 = time.monotonic()

        try:
            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code == 401 or response.status_code == 403:
                logger.error("Google API authentication failed (%d)", response.status_code)
                raise LLMAuthenticationError(
                    "Invalid Google API key. "
                    "Please check your GOOGLE_API_KEY environment variable."
                )

            if response.status_code == 429:
                logger.error("Google rate limit exceeded (429)")
                raise LLMRateLimitError(
                    "Google rate limit exceeded. Please try again later.",
                    retry_after=parse_retry_after(response.headers),
                )

            if response.status_code == 402:
                error_body = response.text[:500]
                logger.error("Google API quota exceeded (402): %s", error_body)
                raise LLMQuotaExceededError(
                    f"Google API quota exceeded (HTTP 402): {error_body}"
                )

            if response.status_code == 503 or (
                response.status_code >= 500 and response.status_code != 501
            ):
                error_body = response.text[:500]
                logger.error(
                    "Google API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMServerUnavailableError(
                    f"Google API returned HTTP {response.status_code}: {error_body}"
                )

            if response.status_code != 200:
                error_body = response.text[:500]
                logger.error(
                    "Google API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMAPIError(
                    f"Google API returned HTTP {response.status_code}: {error_body}"
                )

            data = response.json()

            # Extract content
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("Google response has no candidates")
                return LLMResponse(
                    content="",
                    model=model,
                    provider=self.get_provider_name(),
                    raw_response=data
                )

            candidate = candidates[0]
            content_data = candidate.get("content", {})
            parts = content_data.get("parts", [])

            content = ""
            for part in parts:
                content += part.get("text", "")

            finish_reason = _normalize_gemini_finish_reason(candidate.get("finishReason"))

            # Extract token usage
            usage_metadata = data.get("usageMetadata", {})
            tokens_used = usage_metadata.get("totalTokenCount")

            logger.info(
                "Google %s: %.0fms, tok=%s, finish_reason=%s",
                model, elapsed_ms, tokens_used, finish_reason
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
            logger.error("Cannot connect to Google API: %s", e)
            raise LLMConnectionError(
                "Cannot connect to Google API. "
                "Please check your internet connection."
            ) from e

        except requests.Timeout as e:
            logger.error("Google request timed out after %ds", self.timeout)
            raise LLMConnectionError(
                f"Google request timed out after {self.timeout}s"
            ) from e

        except requests.RequestException as e:
            logger.error("Google request failed: %s", e)
            raise LLMAPIError(f"Google request failed: {e}") from e

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        **kwargs
    ) -> ToolCallResponse:
        """
        Send request to Google Gemini with function calling support.
        When token_callback is provided, uses streamGenerateContent SSE endpoint.
        """
        if not model:
            model = self.DEFAULT_MODEL

        token_callback = kwargs.get("token_callback")

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        contents = []
        system_instruction = None
        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == "tool":
                # Single tool result: wrap in functionResponse part
                tool_name = getattr(msg, "name", "") or ""
                contents.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": tool_name, "response": {"content": msg.content}}}],
                })
            else:
                role = "model" if msg.role == "assistant" else "user"
                raw_content = getattr(msg, "raw_content", None)
                images = getattr(msg, "images", None)
                if raw_content:
                    # Preserve native Gemini parts (functionCall / functionResponse)
                    contents.append({"role": role, "parts": raw_content})
                elif images:
                    # Multimodal: inlineData parts + text
                    parts = []
                    for img in images:
                        parts.append({"inlineData": {"mimeType": img["media_type"], "data": img["data"]}})
                    parts.append({"text": msg.content})
                    contents.append({"role": role, "parts": parts})
                else:
                    contents.append({"role": role, "parts": [{"text": msg.content}]})

        # Convert to Gemini function declarations
        gemini_tools = []
        if tools:
            function_declarations = []
            for t in tools:
                params = t.get("parameters", {})
                function_declarations.append({
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                })
            gemini_tools = [{"functionDeclarations": function_declarations}]

        # Consume non-serializable kwargs
        kwargs.pop("reasoning_callback", None)
        kwargs.pop("token_callback", None)
        _reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_mode = kwargs.pop("thinking_mode", None)
        # Route generation params into generationConfig — Gemini rejects them top-level
        temperature = kwargs.pop("temperature", 0.0)
        max_tokens = kwargs.pop("max_tokens", None)

        generation_config: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens
        thinking_cfg = self._build_gemini_thinking_config(model, thinking_mode, _reasoning_effort)
        if thinking_cfg:
            generation_config["thinkingConfig"] = thinking_cfg

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if gemini_tools:
            payload["tools"] = gemini_tools
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Merge remaining kwargs (e.g. safetySettings, top_p, etc.)
        payload.update(kwargs)

        # Use streaming endpoint when token_callback is provided
        if token_callback:
            stream_url = f"{base_url.rstrip('/')}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
            return self._chat_with_tools_streaming_gemini(
                stream_url, headers, payload, model, token_callback,
            )

        t0 = time.monotonic()
        try:
            response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code in (401, 403):
                raise LLMAuthenticationError("Invalid Google API key.")
            if response.status_code == 429:
                raise LLMRateLimitError(
                    "Google rate limit exceeded.",
                    retry_after=parse_retry_after(response.headers),
                )
            if response.status_code == 402:
                error_body = response.text[:500]
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (HTTP 402): {error_body}"
                )
            if response.status_code >= 500:
                error_body = response.text[:500]
                raise LLMServerUnavailableError(
                    f"Google API server error (HTTP {response.status_code}): {error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(f"Google API returned HTTP {response.status_code}: {error_body}")

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ToolCallResponse(
                    content="", model=model, provider=self.get_provider_name(),
                    raw_response=data, tool_calls=[], is_final=True,
                )

            candidate = candidates[0]
            content_data = candidate.get("content", {})
            parts = content_data.get("parts", [])
            finish_reason = _normalize_gemini_finish_reason(candidate.get("finishReason"))

            usage = data.get("usageMetadata", {})
            tokens_used = usage.get("totalTokenCount")
            prompt_tokens = usage.get("promptTokenCount")
            completion_tokens = usage.get("candidatesTokenCount")

            text_content = ""
            tool_calls: list[ToolCallRequest] = []
            for part in parts:
                if "text" in part:
                    text_content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCallRequest(
                        call_id=f"gemini_{len(tool_calls)}",
                        name=fc.get("name", ""),
                        args=fc.get("args", {}),
                    ))

            is_final = not tool_calls
            logger.info(
                "Google %s (tools): %.0fms, tok=%s, finish=%s (%d)",
                model, elapsed_ms, tokens_used, finish_reason, len(tool_calls),
            )
            return ToolCallResponse(
                content=text_content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=data,
                tool_calls=tool_calls,
                is_final=is_final,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except requests.ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Google API.") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"Google request timed out after {self.timeout}s") from e
        except requests.RequestException as e:
            raise LLMAPIError(f"Google request failed: {e}") from e

    def _chat_with_tools_streaming_gemini(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        token_callback,
    ) -> ToolCallResponse:
        """Gemini streamGenerateContent SSE streaming.

        Text parts are forwarded via token_callback.
        Function call parts are buffered (they arrive as complete objects, not deltas).
        """
        import json as _json

        t0 = time.monotonic()
        try:
            response = self._session.post(
                url, headers=headers, json=payload,
                timeout=self.timeout, stream=True,
            )
        except requests.ConnectionError as e:
            raise LLMConnectionError("Cannot connect to Google API.") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"Google request timed out after {self.timeout}s") from e

        try:
            if response.status_code in (401, 403):
                raise LLMAuthenticationError("Invalid Google API key.")
            if response.status_code == 429:
                raise LLMRateLimitError(
                    "Google rate limit exceeded.",
                    retry_after=parse_retry_after(response.headers),
                )
            if response.status_code == 402:
                error_body = response.text[:500]
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (HTTP 402): {error_body}"
                )
            if response.status_code >= 500:
                raise LLMServerUnavailableError(
                    f"Google API server error (HTTP {response.status_code}): {response.text[:500]}"
                )
            if response.status_code != 200:
                raise LLMAPIError(f"Google API returned HTTP {response.status_code}: {response.text[:500]}")

            text_content = ""
            tool_calls: list[ToolCallRequest] = []
            finish_reason = None
            prompt_tokens = None
            completion_tokens = None
            tokens_used = None

            try:
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        ev = _json.loads(data_str)
                    except Exception:
                        continue

                    candidates = ev.get("candidates", [])
                    if candidates:
                        candidate = candidates[0]
                        finish_reason = _normalize_gemini_finish_reason(candidate.get("finishReason")) or finish_reason
                        parts = candidate.get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                chunk = part["text"]
                                text_content += chunk
                                if chunk:
                                    try:
                                        token_callback(chunk)
                                    except Exception:
                                        pass
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                tool_calls.append(ToolCallRequest(
                                    call_id=f"gemini_{len(tool_calls)}",
                                    name=fc.get("name", ""),
                                    args=fc.get("args", {}),
                                ))

                    usage = ev.get("usageMetadata", {})
                    if usage:
                        prompt_tokens = usage.get("promptTokenCount") or prompt_tokens
                        completion_tokens = usage.get("candidatesTokenCount") or completion_tokens
                        tokens_used = usage.get("totalTokenCount") or tokens_used

            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                raise LLMServerUnavailableError(
                    f"Google streaming request interrupted: {e}"
                ) from e
            except requests.RequestException as e:
                raise LLMAPIError(f"Google streaming request failed: {e}") from e

            elapsed_ms = (time.monotonic() - t0) * 1000
            # Gemini returns finishReason="STOP" even when tool_calls are present.
            # is_final must be driven by tool_calls only — same as the non-streaming path.
            is_final = not tool_calls
            logger.info(
                "Google %s (tools): %.0fms, tok=%s, finish=%s (%d)",
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
            )
        finally:
            response.close()


class DeepSeekClient(LLMClient):
    """
    DeepSeek API client

    DeepSeek uses OpenAI-compatible API
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-v4-flash"

    def get_provider_name(self) -> str:
        return "deepseek"

    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request to DeepSeek

        Uses OpenAI-compatible API
        """
        if not model:
            model = self.DEFAULT_MODEL

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Convert to DeepSeek format (same as OpenAI)
        is_reasoner = "reasoner" in (model or "").lower()
        api_messages = []
        for msg in messages:
            images = getattr(msg, "images", None)
            content = msg.content
            if images:
                content = _images_to_text(images) + ("\n" + content if content else "")
            d = {"role": msg.role, "content": content}
            # DeepSeek Reasoner: reasoning_content is REQUIRED on all assistant messages
            if msg.role == "assistant":
                rc = getattr(msg, "reasoning_content", None) or ""
                if rc or is_reasoner:
                    d["reasoning_content"] = rc
            api_messages.append(d)

        payload = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # thinking_mode → reasoning suppression (shared logic with reasoning_ab_kwargs)
        _thinking_mode = kwargs.pop("thinking_mode", None)
        _reasoning_effort = kwargs.pop("reasoning_effort", None)
        _NON_SERIALIZABLE_KEYS = {
            "reasoning_callback", "think", "token_callback",
            "cache_breakpoint_offset",  # Internal cache control key — DeepSeek uses automatic prefix caching, so payload serialization is forbidden (same as OpenAI client)
        }
        payload.update({k: v for k, v in kwargs.items() if k not in _NON_SERIALIZABLE_KEYS})
        if _thinking_mode is not None and not _thinking_mode:
            payload["thinking"] = {"type": "disabled"}
        elif _thinking_mode:
            payload["thinking"] = {"type": "enabled"}
            if _reasoning_effort:
                # DeepSeek v4: thinking depth ("high" default / "max")
                payload["reasoning_effort"] = _reasoning_effort

        # Streaming path: when the caller supplies a token_callback or
        # reasoning_callback, stream the response so they get incremental output
        # (progress display, early-abort). Total latency is unchanged — this only
        # surfaces tokens as they arrive. No-callback callers keep the blocking
        # JSON path below.
        _token_cb = kwargs.get("token_callback")
        _reasoning_cb = kwargs.get("reasoning_callback")
        if _token_cb or _reasoning_cb:
            return self._chat_streaming(
                url, headers, payload, model,
                token_callback=_token_cb, reasoning_callback=_reasoning_cb,
            )

        t0 = time.monotonic()

        try:
            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code in (401, 403):
                err_label = "authentication" if response.status_code == 401 else "forbidden"
                logger.error("DeepSeek %s failed (%d)", err_label, response.status_code)
                raise LLMAuthenticationError(
                    "Invalid DeepSeek API key. "
                    "Please check your DEEPSEEK_API_KEY environment variable."
                )

            if response.status_code == 402:
                error_body = response.text[:500]
                logger.error("DeepSeek quota exceeded (402): %s", error_body)
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (HTTP 402): {error_body}"
                )

            if response.status_code == 429:
                logger.error("DeepSeek rate limit exceeded (429)")
                raise LLMRateLimitError(
                    "DeepSeek rate limit exceeded. Please try again later.",
                    retry_after=parse_retry_after(response.headers),
                )

            if response.status_code == 503 or (
                response.status_code >= 500 and response.status_code != 501
            ):
                error_body = response.text[:500]
                logger.error(
                    "DeepSeek API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMServerUnavailableError(
                    f"DeepSeek API returned HTTP {response.status_code}: {error_body}"
                )

            if response.status_code != 200:
                error_body = response.text[:500]
                logger.error(
                    "DeepSeek API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMAPIError(
                    f"DeepSeek API returned HTTP {response.status_code}: {error_body}"
                )

            data = response.json()

            # Extract content (OpenAI format)
            choices = data.get("choices", [])
            if not choices:
                logger.warning("DeepSeek response has no choices")
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

            # Extract token usage — include prompt/completion split and the
            # server-side prefix-cache hit count so non-tool callers (the planner)
            # can account cache savings. DeepSeek reports the cached prompt tokens
            # as `prompt_cache_hit_tokens`; reasoning tokens live under
            # completion_tokens_details.reasoning_tokens.
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            _prompt_tokens = usage.get("prompt_tokens")
            _completion_tokens = usage.get("completion_tokens")
            _cache_read = usage.get("prompt_cache_hit_tokens")
            _reasoning_tokens = (
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            )

            logger.info(
                "DeepSeek %s: %.0fms, tok=%s, completion=%s, reasoning=%s, cache_read=%s, finish_reason=%s",
                model, elapsed_ms, tokens_used, _completion_tokens,
                _reasoning_tokens, _cache_read, finish_reason
            )

            return LLMResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=data,
                prompt_tokens=_prompt_tokens,
                completion_tokens=_completion_tokens,
                cache_read_input_tokens=_cache_read,
                reasoning_tokens=_reasoning_tokens,
            )

        except requests.ConnectionError as e:
            logger.error("Cannot connect to DeepSeek API: %s", e)
            raise LLMServerUnavailableError(
                "Cannot connect to DeepSeek API. "
                "Please check your internet connection."
            ) from e

        except requests.Timeout as e:
            logger.error("DeepSeek request timed out after %ds", self.timeout)
            raise LLMServerUnavailableError(
                f"DeepSeek request timed out after {self.timeout}s"
            ) from e

        except requests.RequestException as e:
            logger.error("DeepSeek request failed: %s", e)
            raise LLMAPIError(f"DeepSeek request failed: {e}") from e

    def _chat_streaming(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        token_callback=None,
        reasoning_callback=None,
    ) -> LLMResponse:
        """Streaming variant of chat() (no tools).

        Accumulates content/reasoning deltas, invoking the callbacks as tokens
        arrive, and returns the same LLMResponse shape as the blocking path so
        callers (the planner) parse identically. Total latency is unchanged.
        """
        payload = dict(payload)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        t0 = time.monotonic()
        try:
            response = self._session.post(
                url, headers=headers, json=payload, timeout=self.timeout, stream=True
            )

            if response.status_code in (401, 403):
                err_label = "authentication" if response.status_code == 401 else "forbidden"
                logger.error("DeepSeek %s failed (%d)", err_label, response.status_code)
                raise LLMAuthenticationError("Invalid DeepSeek API key.")
            if response.status_code == 402:
                error_body = response.text[:500]
                logger.error("DeepSeek quota exceeded (402): %s", error_body)
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (HTTP 402): {error_body}"
                )
            if response.status_code == 429:
                logger.error("DeepSeek rate limit exceeded (429)")
                raise LLMRateLimitError(
                    "DeepSeek rate limit exceeded. Please try again later.",
                    retry_after=parse_retry_after(response.headers),
                )
            if response.status_code == 503 or (
                response.status_code >= 500 and response.status_code != 501
            ):
                error_body = response.text[:500]
                logger.error("DeepSeek API error %d: %s", response.status_code, error_body)
                raise LLMServerUnavailableError(
                    f"DeepSeek API returned HTTP {response.status_code}: {error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                logger.error("DeepSeek API error %d: %s", response.status_code, error_body)
                raise LLMAPIError(
                    f"DeepSeek API returned HTTP {response.status_code}: {error_body}"
                )

            full_content = ""
            reasoning_parts: list[str] = []
            finish_reason = None
            usage: dict[str, Any] = {}

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                except Exception:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    continue

                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    full_content += delta["content"]
                    if token_callback:
                        try:
                            token_callback(delta["content"])
                        except Exception:
                            pass
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                    if reasoning_callback:
                        try:
                            reasoning_callback(delta["reasoning_content"])
                        except Exception:
                            pass

                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr
                if "usage" in chunk:
                    usage = chunk["usage"]

            elapsed_ms = (time.monotonic() - t0) * 1000

            tokens_used = usage.get("total_tokens")
            _prompt_tokens = usage.get("prompt_tokens")
            _completion_tokens = usage.get("completion_tokens")
            _cache_read = usage.get("prompt_cache_hit_tokens")
            _reasoning_tokens = (
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                if isinstance(usage, dict) else None
            )

            logger.info(
                "DeepSeek %s (stream): %.0fms, tok=%s, completion=%s, reasoning=%s, cache_read=%s, finish_reason=%s",
                model, elapsed_ms, tokens_used, _completion_tokens,
                _reasoning_tokens, _cache_read, finish_reason
            )

            # Reconstruct a non-streaming-style raw_response so reasoning_content
            # extraction in callers (planner _llm_call) works identically.
            reconstructed_message: dict[str, Any] = {"content": full_content}
            if reasoning_parts:
                reconstructed_message["reasoning_content"] = "".join(reasoning_parts)
            reconstructed_raw: dict[str, Any] = {
                "choices": [{"message": reconstructed_message, "finish_reason": finish_reason}],
                "usage": usage,
                "streamed": True,
            }

            return LLMResponse(
                content=full_content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=reconstructed_raw,
                prompt_tokens=_prompt_tokens,
                completion_tokens=_completion_tokens,
                cache_read_input_tokens=_cache_read,
                reasoning_tokens=_reasoning_tokens,
            )

        except requests.ConnectionError as e:
            logger.error("Cannot connect to DeepSeek API: %s", e)
            raise LLMServerUnavailableError(
                "Cannot connect to DeepSeek API. Please check your internet connection."
            ) from e
        except requests.Timeout as e:
            logger.error("DeepSeek stream timed out after %ds", self.timeout)
            raise LLMServerUnavailableError(
                f"DeepSeek request timed out after {self.timeout}s"
            ) from e
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error("DeepSeek stream interrupted: %s", e)
            raise LLMServerUnavailableError(
                f"DeepSeek stream interrupted: {e}"
            ) from e
        except requests.RequestException as e:
            logger.error("DeepSeek stream request failed: %s", e)
            raise LLMAPIError(f"DeepSeek request failed: {e}") from e
        finally:
            try:
                response.close()
            except NameError:
                pass

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        **kwargs
    ) -> ToolCallResponse:
        """
        Send chat completion request to DeepSeek with tool calling support.
        DeepSeek uses OpenAI-compatible API, so the format is identical.
        """
        if not model:
            model = self.DEFAULT_MODEL

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        is_reasoner = "reasoner" in (model or "").lower()

        api_messages = []
        for m in messages:
            images = getattr(m, "images", None)
            content = m.content
            if images:
                content = _images_to_text(images) + ("\n" + content if content else "")
            d = {"role": m.role, "content": content}
            if m.role == "assistant" and getattr(m, "tool_calls", None):
                d["tool_calls"] = m.tool_calls
                if d.get("content") is None:
                    d["content"] = ""
            # DeepSeek Reasoner: reasoning_content is REQUIRED on all assistant messages
            if m.role == "assistant":
                rc = getattr(m, "reasoning_content", None) or ""
                if rc or is_reasoner:
                    d["reasoning_content"] = rc
            if m.role == "tool":
                if getattr(m, "tool_call_id", None):
                    d["tool_call_id"] = m.tool_call_id
                if getattr(m, "name", None):
                    d["name"] = m.name
            api_messages.append(d)

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

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "tools": openai_tools,
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # thinking_mode → reasoning suppression (shared logic with reasoning_ab_kwargs)
        _thinking_mode = kwargs.pop("thinking_mode", None)
        _reasoning_effort = kwargs.pop("reasoning_effort", None)
        _NON_SERIALIZABLE_KEYS = {
            "reasoning_callback", "think", "token_callback",
            "cache_breakpoint_offset",  # Internal cache control key — DeepSeek uses automatic prefix caching, so payload serialization is forbidden (same as OpenAI client)
        }
        payload.update({k: v for k, v in kwargs.items() if k not in _NON_SERIALIZABLE_KEYS})
        if _thinking_mode is not None and not _thinking_mode:
            payload["thinking"] = {"type": "disabled"}
        elif _thinking_mode:
            payload["thinking"] = {"type": "enabled"}
            if _reasoning_effort:
                # DeepSeek v4: thinking depth ("high" default / "max")
                payload["reasoning_effort"] = _reasoning_effort

        t0 = time.monotonic()
        try:
            response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout, stream=True)

            if response.status_code == 401:
                raise LLMAuthenticationError("Invalid DeepSeek API key.")
            if response.status_code == 402:
                error_body = response.text[:500]
                raise LLMQuotaExceededError(
                    f"Insufficient credits or quota exceeded (HTTP 402): {error_body}"
                )
            if response.status_code == 429:
                raise LLMRateLimitError(
                    "DeepSeek rate limit exceeded.",
                    retry_after=parse_retry_after(response.headers),
                )
            if response.status_code == 403:
                error_body = response.text[:500]
                raise LLMAuthenticationError(
                    f"DeepSeek API access forbidden (HTTP 403): {error_body}"
                )
            if response.status_code >= 500:
                error_body = response.text[:500]
                raise LLMServerUnavailableError(
                    f"DeepSeek API server error (HTTP {response.status_code}): {error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(f"DeepSeek API returned HTTP {response.status_code}: {error_body}")

            # Stream processing
            full_content = ""
            full_tool_calls_raw: list[dict[str, Any]] = []
            finish_reason = None
            usage: dict[str, Any] = {}
            reasoning_content_parts: list[str] = []

            # Extract reasoning_callback if provided
            reasoning_callback = kwargs.get("reasoning_callback")

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Remove "data: " prefix
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                except Exception:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    # Usage may come in the final chunk (separate from choices)
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    continue

                delta = choices[0].get("delta", {})

                # Accumulate content
                if delta.get("content"):
                    full_content += delta["content"]
                    _token_cb = kwargs.get("token_callback")
                    if _token_cb:
                        try:
                            _token_cb(delta["content"])
                        except Exception:
                            pass

                # Accumulate reasoning_content (DeepSeek Reasoner)
                if delta.get("reasoning_content"):
                    reasoning_content_parts.append(delta["reasoning_content"])
                    if reasoning_callback:
                        try:
                            reasoning_callback(delta["reasoning_content"])
                        except Exception:
                            pass

                # Accumulate tool calls
                tc_deltas = delta.get("tool_calls") or []
                for tc_delta in tc_deltas:
                    idx = tc_delta.get("index", 0)
                    while len(full_tool_calls_raw) <= idx:
                        full_tool_calls_raw.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc_delta.get("id"):
                        full_tool_calls_raw[idx]["id"] = tc_delta["id"]
                    func_delta = tc_delta.get("function", {})
                    if func_delta.get("name"):
                        full_tool_calls_raw[idx]["function"]["name"] = func_delta["name"]
                    if func_delta.get("arguments"):
                        full_tool_calls_raw[idx]["function"]["arguments"] += func_delta["arguments"]

                # Capture finish_reason
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                # Usage in streaming may come in delta or in chunk
                if "usage" in chunk:
                    usage = chunk["usage"]

            # ── Response completeness validation ──────────────────────────────────
            # Streaming content can be silently truncated even when finish_reason='stop'
            # (last content chunk dropped but finish chunk received intact).
            # Detect structural truncation in JSON/text responses.
            if finish_reason == "stop" and full_content:
                _trimmed = full_content.strip()
                if _trimmed.startswith(("{", "[")):
                    # String-aware brace/bracket counting: ignore braces/brackets
                    # inside string literals to avoid false positives.
                    _open_cb = 0
                    _close_cb = 0
                    _open_sb = 0
                    _close_sb = 0
                    _in_str = False
                    _str_char = None
                    _esc = False
                    for _ch in _trimmed:
                        if _in_str:
                            if _esc:
                                _esc = False
                            elif _ch == "\\":
                                _esc = True
                            elif _ch == _str_char:
                                _in_str = False
                                _str_char = None
                            continue
                        if _ch in ('"', "'"):
                            _in_str = True
                            _str_char = _ch
                            continue
                        if _ch == "{":
                            _open_cb += 1
                        elif _ch == "}":
                            _close_cb += 1
                        elif _ch == "[":
                            _open_sb += 1
                        elif _ch == "]":
                            _close_sb += 1
                    if _trimmed.startswith("{") and _open_cb > _close_cb:
                        logger.warning(
                            "Response content appears truncated: %d unclosed braces "
                            "in %d chars (finish_reason='stop' is misleading)",
                            _open_cb - _close_cb, len(_trimmed),
                        )
                        finish_reason = "truncated"
                    elif _trimmed.startswith("[") and _open_sb > _close_sb:
                        logger.warning(
                            "Response content appears truncated: %d unclosed brackets "
                            "in %d chars (finish_reason='stop' is misleading)",
                            _open_sb - _close_sb, len(_trimmed),
                        )
                        finish_reason = "truncated"

            # ── Tool call arguments truncation detection ──────────────────────────
            # When finish_reason='tool_calls', the tool_call arguments may still be
            # truncated (API returned a partial tool call due to max_tokens or other
            # limits). Detect structural truncation in tool_call JSON arguments.
            if finish_reason == "tool_calls" and full_tool_calls_raw:
                _tc_truncated = False
                for _tc_idx, _tc in enumerate(full_tool_calls_raw):
                    _args = _tc.get("function", {}).get("arguments", "")
                    if not _args or not _args.strip():
                        continue
                    _trimmed = _args.strip()
                    # String-aware brace counting
                    _open_cb = 0
                    _close_cb = 0
                    _in_str = False
                    _str_char = None
                    _esc = False
                    for _ch in _trimmed:
                        if _in_str:
                            if _esc:
                                _esc = False
                            elif _ch == "\\":
                                _esc = True
                            elif _ch == _str_char:
                                _in_str = False
                                _str_char = None
                            continue
                        if _ch in ('"', "'"):
                            _in_str = True
                            _str_char = _ch
                            continue
                        if _ch == "{":
                            _open_cb += 1
                        elif _ch == "}":
                            _close_cb += 1
                    if _open_cb > _close_cb:
                        _tc_truncated = True
                        _tc_name = _tc.get("function", {}).get("name", f"tool_call[{_tc_idx}]")
                        logger.warning(
                            "Tool call '%s' arguments appear truncated: %d unclosed braces "
                            "in %d chars (finish_reason='tool_calls' may be misleading)",
                            _tc_name, _open_cb - _close_cb, len(_trimmed),
                        )
                if _tc_truncated:
                    finish_reason = "truncated"
                    # Clear malformed tool calls so the caller retries rather than
                    # attempting to execute partial arguments.
                    full_tool_calls_raw.clear()

            elapsed_ms = (time.monotonic() - t0) * 1000

            tokens_used = usage.get("total_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            reasoning_tokens = (
                usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage, dict) else None
            )
            cache_read_input_tokens = usage.get("prompt_cache_hit_tokens")

            tool_calls: list[ToolCallRequest] = []
            for tc in full_tool_calls_raw:
                func = tc.get("function", {})
                args = parse_tool_args(func.get("arguments", "{}"))
                tool_calls.append(ToolCallRequest(
                    call_id=tc.get("id", ""),
                    name=func.get("name", ""),
                    args=args,
                ))

            is_final = finish_reason == "stop" or not tool_calls
            logger.info(
                "DeepSeek %s (tools): %.0fms, tok=%s, cache_read=%s, finish=%s (%d)",
                model, elapsed_ms, tokens_used, cache_read_input_tokens, finish_reason, len(tool_calls),
            )

            # Reconstruct a non-streaming-style raw_response so that
            # agent_loop._append_native_tool_messages can extract
            # choices[0].message.tool_calls and reasoning_content.
            reconstructed_message: dict[str, Any] = {"content": full_content}
            if full_tool_calls_raw:
                reconstructed_message["tool_calls"] = full_tool_calls_raw
            if reasoning_content_parts:
                reconstructed_message["reasoning_content"] = "".join(reasoning_content_parts)
            reconstructed_raw: dict[str, Any] = {
                "choices": [{"message": reconstructed_message, "finish_reason": finish_reason}],
                "usage": usage,
                "streamed": True,
            }

            return ToolCallResponse(
                content=full_content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response=reconstructed_raw,
                tool_calls=tool_calls,
                is_final=is_final,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                reasoning_tokens=reasoning_tokens,
            )

        except requests.ConnectionError as e:
            raise LLMConnectionError("Cannot connect to DeepSeek API.") from e
        except requests.Timeout as e:
            raise LLMConnectionError(f"DeepSeek request timed out after {self.timeout}s") from e
        except requests.exceptions.ChunkedEncodingError as e:
            raise LLMServerUnavailableError(
                f"DeepSeek stream interrupted: {e}"
            ) from e
        except requests.RequestException as e:
            raise LLMAPIError(f"DeepSeek request failed: {e}") from e
        finally:
            try:
                response.close()
            except NameError:
                pass


class OllamaClient(LLMClient):
    """
    Ollama API client for local LLM models

    Supported models:
    - qwen2.5-coder:3b
    - codellama:7b
    - mistral:7b
    - llama3.2:3b
    - Any model available in local Ollama
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "qwen2.5-coder:3b"

    def __init__(self, api_key: str = "", base_url: Optional[str] = None, timeout: int = 300):
        """
        Initialize Ollama client with extended timeout for model loading.

        Args:
            api_key: Not used for Ollama (empty string)
            base_url: Custom Ollama server URL
            timeout: Request timeout in seconds (default 300 for large models)
        """
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)

    def get_provider_name(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: list[LLMMessage],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send request to Ollama /api/chat endpoint

        API Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
        """
        if not model:
            model = self.DEFAULT_MODEL

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/api/chat"

        # Convert to Ollama format
        is_vision = _is_ollama_vision_model(model)
        ollama_messages = []
        for msg in messages:
            m_dict: dict[str, Any] = {"role": msg.role, "content": msg.content}
            images = getattr(msg, "images", None)
            if images:
                if is_vision:
                    # Native Ollama vision: pass base64 data list directly
                    m_dict["images"] = [img["data"] for img in images]
                else:
                    # Non-vision model: prepend OCR text (or placeholder) to content
                    image_text = _images_to_text(images)
                    m_dict["content"] = image_text + ("\n" + msg.content if msg.content else "")
            ollama_messages.append(m_dict)

        # Collapse all system messages into one at index 0 — strict templates
        # (e.g. Qwen3 / bonsai27b) reject any system message that is not first.
        ollama_messages = _normalize_ollama_system_messages(ollama_messages)

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Auto-set num_ctx so Ollama does not truncate the system prompt. The
        # floor is 8192 for EVERY model: asicode's system prefix (core_prompt +
        # project.md + design_insights) is ~5272 tokens (measured via
        # _cjk_aware_tokens), which already OVERFLOWS Ollama's 4096 default and
        # would 400 ("exceeds context size") before any user content is added.
        # NOTE: earlier size-based tiers (e.g. 13B+ -> 4096, 8B-12B -> 6144) were
        # never implemented in code AND are not viable — both are below the 5272
        # token prefix, so applying them would make asicode unbootable.
        # KV cache grows linearly with num_ctx; roughly:
        #   KV_GB = num_ctx * num_layers * kv_heads * head_dim * 4 / 1e9
        # On memory-constrained hardware (8GB unified memory), users who need a
        # different value set num_ctx in the Modelfile — priority 0 in
        # _num_ctx_for_model reads it from /api/show at runtime.
        #
        if "num_ctx" not in kwargs:
            _num_ctx = self._num_ctx_for_model(model)
            if _num_ctx is not None:
                payload["options"]["num_ctx"] = _num_ctx
                logger.debug("Auto-set num_ctx=%d for model %s", _num_ctx, model)

        # UI Thinking Mode override — GPT-OSS needs string levels, others use boolean
        thinking_mode = kwargs.get("thinking_mode", None)
        reasoning_effort = kwargs.get("reasoning_effort", None)
        reasoning_callback = kwargs.get("reasoning_callback", None)

        # Thinking mode: explicit UI setting only (think=False is Ollama default)
        if thinking_mode is not None:
            _think_val = _ollama_think_value(model, thinking_mode, reasoning_effort)
            if _think_val is not None:
                payload["think"] = _think_val
                logger.debug("Thinking mode set by UI for model %s: think=%s", model, _think_val)

        token_callback = kwargs.get("token_callback")

        # Enable native Ollama streaming when Thinking Mode is ON or token_callback is provided
        use_reasoning_stream = bool(thinking_mode) and callable(reasoning_callback)
        use_stream = use_reasoning_stream or callable(token_callback)
        if use_stream:
            payload["stream"] = True

        # Add any additional kwargs to options (filter out non-Ollama keys)
        _SKIP_KEYS = {
            "model", "messages", "stream", "thinking_mode", "think",
            "reasoning_effort", "reasoning_callback", "token_callback", "timeout",
        }
        for key, value in kwargs.items():
            if key not in _SKIP_KEYS:
                payload["options"][key] = value

        t0 = time.monotonic()

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=use_stream,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code == 404:
                logger.error("Ollama model not found (404): %s", model)
                raise LLMAPIError(
                    f"Ollama model '{model}' not found. "
                    f"Make sure you've pulled the model with 'ollama pull {model}'"
                )

            if response.status_code == 401:
                logger.error("Ollama authentication failed (401)")
                raise LLMAuthenticationError(
                    "Ollama authentication failed. "
                    "Check if Ollama requires authentication."
                )

            if response.status_code == 429:
                retry_after = parse_retry_after(response.headers)
                raise LLMRateLimitError(
                    f"Ollama rate limited (429), retry after {retry_after}s"
                )
            if response.status_code >= 500:
                error_body = response.text[:500]
                logger.error(
                    "Ollama server error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMServerUnavailableError(
                    f"Ollama API returned HTTP {response.status_code}: {error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                logger.error(
                    "Ollama API error %d in %.0fms: %s",
                    response.status_code, elapsed_ms, error_body
                )
                raise LLMAPIError(
                    f"Ollama API returned HTTP {response.status_code}: {error_body}"
                )

            if use_stream:
                import json as _json

                content_parts: list[str] = []
                thinking_parts: list[str] = []
                finish_reason = None
                _stream_pt = 0
                _stream_ct = 0

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                    except Exception:
                        continue

                    if chunk.get("error"):
                        raise LLMAPIError(f"Ollama stream error: {chunk.get('error')}")

                    msg = chunk.get("message", {}) or {}
                    thinking_chunk = msg.get("thinking", "") or ""
                    content_chunk = msg.get("content", "") or ""

                    if thinking_chunk:
                        thinking_parts.append(thinking_chunk)
                        try:
                            reasoning_callback(thinking_chunk)
                        except Exception:
                            pass

                    if content_chunk:
                        content_parts.append(content_chunk)
                        if token_callback:
                            try:
                                token_callback(content_chunk)
                            except Exception:
                                pass

                    if chunk.get("done"):
                        finish_reason = chunk.get("done_reason")
                        _stream_pt = chunk.get("prompt_eval_count", 0) or 0
                        _stream_ct = chunk.get("eval_count", 0) or 0

                content = "".join(content_parts)
                thinking_text = "".join(thinking_parts)
                tokens_used = (_stream_pt + _stream_ct) or None

                logger.info(
                    "Ollama %s: %.0fms, finish_reason=%s, reasoning_chars=%d, tok=%s",
                    model, elapsed_ms, finish_reason, len(thinking_text), tokens_used
                )

                return LLMResponse(
                    content=content,
                    model=model,
                    provider=self.get_provider_name(),
                    tokens_used=tokens_used,
                    prompt_tokens=_stream_pt or None,
                    completion_tokens=_stream_ct or None,
                    finish_reason=finish_reason,
                    raw_response={
                        "streamed": True,
                        "thinking": thinking_text,
                        "prompt_eval_count": _stream_pt,
                        "eval_count": _stream_ct,
                    },
                )

            data = response.json()

            # Extract content
            message = data.get("message", {})
            content = message.get("content", "")
            finish_reason = data.get("done_reason")

            # Extract token usage from Ollama response fields
            _prompt_tokens = data.get("prompt_eval_count") or 0
            _completion_tokens = data.get("eval_count") or 0
            tokens_used = (_prompt_tokens + _completion_tokens) or None

            logger.info(
                "Ollama %s: %.0fms, finish_reason=%s, tok=%s",
                model, elapsed_ms, finish_reason, tokens_used
            )

            return LLMResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                prompt_tokens=_prompt_tokens or None,
                completion_tokens=_completion_tokens or None,
                finish_reason=finish_reason,
                raw_response=data
            )

        except requests.ConnectionError as e:
            logger.error("Cannot connect to Ollama at %s: %s", base_url, e)
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Is Ollama running? Try 'ollama serve'"
            ) from e

        except requests.Timeout as e:
            logger.error("Ollama request timed out after %ds", self.timeout)
            raise LLMConnectionError(
                f"Ollama request timed out after {self.timeout}s"
            ) from e

        except requests.RequestException as e:
            logger.error("Ollama request failed: %s", e)
            raise LLMAPIError(f"Ollama request failed: {e}") from e

    def _num_ctx_for_model(self, model: str) -> Optional[int]:
        """Return appropriate num_ctx for a given model name.

        Priority:
          0. Dynamic query from Ollama /api/show — if the model has an
             explicit ``num_ctx`` set in its Modelfile, use it.  This allows
             users to customize context size via ``ollama run /set num_ctx X /save``.
          1. Explicit registry override (OLLAMA_NUM_CTX_OVERRIDES) — for tags
             whose Modelfile lacks num_ctx and the 8192 floor is wrong.  Currently
             empty — users should set num_ctx via Modelfile for persistent values.
          2. Sensible fallback (8192) — Ollama's own default (4096) is too small
             for asicode's system prompt: the measured prefix (core_prompt +
             project.md + design_insights) is ~5272 tokens, so unknown models
             would otherwise 400 on "exceeds context size".  8192 is the floor
             for every model (NOT size-based — see the note in chat()).  Users
             wanting more (e.g. bonsai27b = 256K Qwen3, Q1_0 → 32768) or less
             should set num_ctx via Modelfile (priority 0).
        """
        # 0. Dynamic query from Ollama API (Option B)
        from external_llm.ollama_api import query_ollama_num_ctx
        base_url = self.base_url or self.DEFAULT_BASE_URL
        api_ctx = query_ollama_num_ctx(model, base_url_hint=base_url)
        if api_ctx is not None:
            logger.debug("num_ctx=%d from Ollama API for model %s", api_ctx, model)
            return api_ctx

        # 1. Explicit registry override
        from external_llm.model_registry import get_ollama_num_ctx
        override = get_ollama_num_ctx(model)
        if override is not None:
            return override

        # 2. Sensible fallback — Ollama's 4096 default is too small for asicode's
        #    system prompt.  8192 guarantees asicode boots for any unknown model;
        #    users can override via Modelfile (priority 0) for larger/smaller values.
        #    NOTE: this overrides the Ollama server's ``OLLAMA_CONTEXT_LENGTH`` env.
        #    Previously this branch returned None and let the server decide (so the
        #    env was respected), but that path silently 400'd on asicode's ~5272-token
        #    system prefix under Ollama's 4096 default.  Users relying on the env for
        #    a larger window should set num_ctx in the Modelfile (priority 0) instead.
        #    The architectural context_length from /api/show (e.g. qwen35.context_length
        #    =262144) is intentionally NOT used as a floor — it is the model's
        #    theoretical maximum, not a memory-safe value on constrained hardware
        #    (128K KV cache would OOM an 8GB box), and a correct per-device cap cannot
        #    be inferred without VRAM/slot knowledge.
        fallback = 8192
        logger.debug("num_ctx=%d fallback (no Modelfile/registry) for model %s", fallback, model)
        return fallback

    def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        model: str = "",
        **kwargs
    ) -> ToolCallResponse:
        """
        Send request to Ollama /api/chat with native tool calling support.

        Ollama's tool_calls format differs from OpenAI:
        - response tool_calls: {"function": {"name": "...", "arguments": {...}}} (no id; args is dict)
        - tool result messages: {"role": "tool", "content": "..."}
        - assistant messages with tool_calls need Ollama-specific serialization
        """
        if not model:
            model = self.DEFAULT_MODEL

        token_callback = kwargs.pop("token_callback", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning_callback", None)
        thinking_mode = kwargs.pop("thinking_mode", None)
        temperature = kwargs.pop("temperature", 0.0)
        max_tokens = kwargs.pop("max_tokens", None)

        base_url = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/api/chat"

        # Build Ollama message list, handling tool_calls and tool results
        ollama_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                # Tool result: Ollama only needs role + content (no tool_call_id)
                ollama_messages.append({"role": "tool", "content": msg.content or ""})
            elif msg.role == "assistant":
                m_dict: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                tc_list = getattr(msg, "tool_calls", None)
                if tc_list:
                    # Convert tool_calls to Ollama format.
                    # Handles two possible source formats:
                    #   A) OpenAI: {"type": "function", "function": {"name": "...", "arguments": "json_str_or_dict"}}
                    #   B) Ollama/normalized: {"function": {"name": "...", "arguments": {...}}}
                    #      or agent_loop normalized: {"id": "...", "name": "...", "args": {...}}
                    ollama_tcs = []
                    for tc in tc_list:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function", {})
                        if func:
                            # Format A or B: has "function" key
                            raw_args = func.get("arguments", {})
                            if isinstance(raw_args, str):
                                try:
                                    args = _json.loads(raw_args)
                                except Exception:
                                    args = {}
                            else:
                                args = raw_args if isinstance(raw_args, dict) else {}
                            name = func.get("name", "")
                        elif "name" in tc:
                            # agent_loop normalized: {"id": "...", "name": "...", "args": {...}}
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            if not isinstance(args, dict):
                                args = {}
                        else:
                            continue
                        ollama_tcs.append({"function": {"name": name, "arguments": args}})
                    if ollama_tcs:
                        m_dict["tool_calls"] = ollama_tcs
                ollama_messages.append(m_dict)
            else:
                # system / user
                ollama_messages.append({"role": msg.role, "content": msg.content or ""})

        # Collapse all system messages into one at index 0 — strict templates
        # (e.g. Qwen3 / bonsai27b) reject any system message that is not first.
        ollama_messages = _normalize_ollama_system_messages(ollama_messages)

        # Convert tools to Ollama format (same as OpenAI)
        ollama_tools = [
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "tools": ollama_tools,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Apply per-model context window sizing (same heuristics as chat())
        num_ctx = self._num_ctx_for_model(model)
        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx
            logger.debug("Auto-set num_ctx=%d for %s (chat_with_tools)", num_ctx, model)

        # Apply thinking/reasoning mode
        if thinking_mode is not None:
            _think_val = _ollama_think_value(model, thinking_mode, reasoning_effort)
            if _think_val is not None:
                payload["think"] = _think_val

        # Enable streaming only when token_callback is provided
        if callable(token_callback):
            payload["stream"] = True

        t0 = time.monotonic()

        try:
            response = self._session.post(
                url, json=payload, timeout=self.timeout,
                stream=bool(payload.get("stream")),
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            if response.status_code == 404:
                raise LLMAPIError(
                    f"Ollama model '{model}' not found. Pull it with 'ollama pull {model}'"
                )
            if response.status_code == 401:
                raise LLMAuthenticationError(
                    "Ollama authentication failed. "
                    "Check if Ollama requires authentication."
                )
            if response.status_code == 429:
                retry_after = parse_retry_after(response.headers)
                raise LLMRateLimitError(
                    f"Ollama rate limited (429), retry after {retry_after}s"
                )
            if response.status_code >= 500:
                error_body = response.text[:500]
                raise LLMServerUnavailableError(
                    f"Ollama API returned HTTP {response.status_code}: {error_body}"
                )
            if response.status_code != 200:
                error_body = response.text[:500]
                raise LLMAPIError(
                    f"Ollama API returned HTTP {response.status_code}: {error_body}"
                )

            # ── Streaming path (token_callback provided) ───────────────────────
            if payload.get("stream"):
                content_parts: list[str] = []
                finish_reason: Optional[str] = None
                _pt = _ct = 0
                raw_tool_calls_stream: list[dict] = []

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                    except Exception:
                        continue

                    if chunk.get("error"):
                        raise LLMAPIError(f"Ollama stream error: {chunk.get('error')}")

                    msg_chunk = chunk.get("message", {}) or {}
                    content_chunk = msg_chunk.get("content", "") or ""
                    if content_chunk:
                        content_parts.append(content_chunk)
                        try:
                            token_callback(content_chunk)
                        except Exception:
                            pass

                    # Tool calls may appear in the final chunk
                    if msg_chunk.get("tool_calls"):
                        raw_tool_calls_stream = msg_chunk["tool_calls"]

                    if chunk.get("done"):
                        finish_reason = chunk.get("done_reason")
                        _pt = chunk.get("prompt_eval_count", 0) or 0
                        _ct = chunk.get("eval_count", 0) or 0

                content = "".join(content_parts)
                raw_tool_calls = raw_tool_calls_stream
                tokens_used = (_pt + _ct) or None
                prompt_tokens, completion_tokens = _pt or None, _ct or None
            else:
                # ── Non-streaming path ─────────────────────────────────────────
                data = response.json()
                msg_obj = data.get("message", {}) or {}
                content = msg_obj.get("content") or ""
                finish_reason = data.get("done_reason")
                _pt = data.get("prompt_eval_count") or 0
                _ct = data.get("eval_count") or 0
                tokens_used = (_pt + _ct) or None
                prompt_tokens, completion_tokens = _pt or None, _ct or None
                raw_tool_calls = msg_obj.get("tool_calls") or []

            # Parse Ollama tool_calls → ToolCallRequest list
            tool_call_requests: list[ToolCallRequest] = []
            for i, tc in enumerate(raw_tool_calls):
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {"__raw_arguments": args}
                if not isinstance(args, dict):
                    args = {"__raw_arguments": str(args)}
                tool_call_requests.append(ToolCallRequest(
                    call_id=f"ollama_{i}_{func.get('name', 'tool')}",
                    name=func.get("name", ""),
                    args=args,
                ))

            # Ollama returns done_reason="stop" even when tool_calls are present.
            # is_final must be False whenever there are tool calls to execute.
            is_final = not tool_call_requests

            logger.info(
                "Ollama %s (tools): %.0fms, tok=%s, done_reason=%s (%d)",
                model, elapsed_ms, tokens_used, finish_reason, len(tool_call_requests),
            )

            return ToolCallResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                raw_response={"done_reason": finish_reason, "prompt_eval_count": _pt, "eval_count": _ct},
                tool_calls=tool_call_requests,
                is_final=is_final,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except requests.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {base_url}. Is Ollama running?"
            ) from e
        except requests.Timeout as e:
            raise LLMConnectionError(
                f"Ollama request timed out after {self.timeout}s"
            ) from e
        except requests.exceptions.ChunkedEncodingError as e:
            raise LLMServerUnavailableError(
                f"Ollama stream interrupted: {e}"
            ) from e
        except requests.RequestException as e:
            raise LLMAPIError(f"Ollama request failed: {e}") from e
        finally:
            try:
                response.close()
            except NameError:
                pass

