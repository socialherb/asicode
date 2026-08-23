"""Non-vision models must never receive image_url parts.

deepseek-v4-flash via OpenCode Go returns HTTP 400 ("Upstream request failed")
for ANY image_url part — verified 2026-07-26 with a 67-byte 1x1 PNG, the same
400 as a 1.5 MB screenshot — and the native DeepSeek API is likewise text-only.
Images therefore convert to OCR/placeholder text before the request for known
text-only models (model_registry.TEXT_ONLY_MODEL_PREFIXES), with a one-shot
strip-and-retry net in _request_with_retry for models not yet in the registry.
"""

from __future__ import annotations

import pytest

import external_llm.openai_client as oc
from external_llm.client import LLMAPIError, LLMMessage
from external_llm.model_registry import text_only_model
from external_llm.openai_client import (
    OpenAIClient,
    _openai_content,
    _strip_image_parts,
)

# 1x1 red PNG (67 bytes) — valid image data so the OCR path can actually run
TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _img_msg(text: str = "what is this?") -> LLMMessage:
    return LLMMessage(
        role="user",
        content=text,
        images=[{"media_type": "image/png", "data": TINY_PNG}],
    )


@pytest.fixture(autouse=True)
def _fresh_learned_set(monkeypatch):
    monkeypatch.setattr(oc, "_IMAGE_REJECTING_MODELS", set())


# ── Registry classification ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4-flash",
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash",
        "DeepSeek-V4-Flash",
    ],
)
def test_deepseek_family_is_text_only(model):
    assert text_only_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-5",
        "gemini-2.5-pro",
        "claude-sonnet-5",
        "kimi-k3",
        "deepseek-vl",
        "deepseek-vl2",
        "deepseek-ocr",
    ],
)
def test_other_models_not_classified_text_only(model):
    assert not text_only_model(model)


# ── Content building gate ────────────────────────────────────────────────────


def test_text_only_model_gets_text_fallback():
    content = _openai_content(_img_msg("설명해줘"), "deepseek-v4-flash")
    assert isinstance(content, str)
    assert "Image 1" in content
    assert content.endswith("설명해줘")


def test_vision_unknown_model_gets_image_parts():
    content = _openai_content(_img_msg(), "gpt-4o")
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "text", "text": "what is this?"}


def test_no_images_passes_through():
    assert _openai_content(LLMMessage(role="user", content="hi"), "deepseek-v4-flash") == "hi"


def test_runtime_learned_model_gets_text_fallback():
    oc._IMAGE_REJECTING_MODELS.add(("https://x.test/v1", "mystery-model"))
    content = _openai_content(_img_msg(), "some-route/mystery-model", "https://x.test/v1/")
    assert isinstance(content, str)


def test_learned_rejection_is_route_scoped():
    oc._IMAGE_REJECTING_MODELS.add(("https://x.test/v1", "mystery-model"))
    content = _openai_content(_img_msg(), "mystery-model", "https://other.test/v1")
    assert isinstance(content, list)  # other routes keep native image parts


# ── _strip_image_parts ───────────────────────────────────────────────────────


def test_strip_returns_none_without_images():
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert _strip_image_parts(payload) is None


def test_strip_replaces_image_parts_with_text():
    payload = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}},
                    {"type": "text", "text": "look"},
                ],
            },
        ],
    }
    stripped = _strip_image_parts(payload)
    assert stripped is not None
    assert stripped["messages"][0] == {"role": "system", "content": "sys"}
    user = stripped["messages"][1]["content"]
    assert isinstance(user, str)
    assert "look" in user
    # original payload untouched (retry must not mutate the caller's payload)
    assert isinstance(payload["messages"][1]["content"], list)


# ── 400 strip-and-retry net ──────────────────────────────────────────────────


class _Resp:
    def __init__(self, status: int):
        self.status_code = status
        self.text = '{"error":{"message":"Upstream request failed"}}'
        self.headers: dict[str, str] = {}

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 5},
        }

    def close(self):
        pass


def _capture_client(monkeypatch, responses: list[_Resp]):
    calls: list[dict] = []

    def _post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append(json)
        return responses[min(len(calls), len(responses)) - 1]

    client = OpenAIClient(api_key="test")
    monkeypatch.setattr(client._session, "post", _post)
    return client, calls


def test_400_with_images_strips_and_retries_once(monkeypatch):
    client, calls = _capture_client(monkeypatch, [_Resp(400), _Resp(200)])
    resp = client.chat([_img_msg()], model="mystery-model")
    assert resp.content == "ok"
    assert len(calls) == 2
    retry_contents = [m["content"] for m in calls[1]["messages"]]
    assert all(isinstance(c, str) for c in retry_contents)
    assert "image" in retry_contents[0].lower()
    learned_base = oc._norm_base(OpenAIClient.DEFAULT_BASE_URL)
    assert (learned_base, "mystery-model") in oc._IMAGE_REJECTING_MODELS


def test_400_without_images_raises_without_retry(monkeypatch):
    client, calls = _capture_client(monkeypatch, [_Resp(400)])
    with pytest.raises(LLMAPIError):
        client.chat([LLMMessage(role="user", content="hi")], model="mystery-model")
    assert len(calls) == 1
    assert not oc._IMAGE_REJECTING_MODELS


def test_400_with_images_retry_fails_does_not_poison_model(monkeypatch):
    """400 with images → stripped retry ALSO 400 → image was NOT the cause.
    The model must NOT be added to _IMAGE_REJECTING_MODELS."""
    client, calls = _capture_client(monkeypatch, [_Resp(400), _Resp(400)])
    with pytest.raises(LLMAPIError):
        client.chat([_img_msg()], model="mystery-model")
    assert len(calls) == 2  # original + stripped retry
    retry_contents = [m["content"] for m in calls[1]["messages"]]
    assert all(isinstance(c, str) for c in retry_contents)
    assert not oc._IMAGE_REJECTING_MODELS


def test_learned_model_skips_images_on_next_call(monkeypatch):
    oc._IMAGE_REJECTING_MODELS.add((oc._norm_base(OpenAIClient.DEFAULT_BASE_URL), "mystery-model"))
    client, calls = _capture_client(monkeypatch, [_Resp(200)])
    resp = client.chat([_img_msg()], model="mystery-model")
    assert resp.content == "ok"
    assert len(calls) == 1
    assert isinstance(calls[0]["messages"][0]["content"], str)
