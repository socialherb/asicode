"""Ollama image attachment parity: chat() and chat_with_tools() must convert
attached images identically — native base64 ``images`` for vision models, OCR/
placeholder text for non-vision models.

Regression for the ``chat_with_tools`` else-branch (system/user) which dropped
images entirely: only ``role`` + ``content`` were sent, so a local vision model
(llava) never saw the picture and a non-vision model never even got the OCR
placeholder. The agent loop always uses ``chat_with_tools`` for Ollama
(``_NATIVE_TOOL_PROVIDERS``), so ``chat`` alone was never enough.
"""
from __future__ import annotations

import pytest

from external_llm.client import LLMMessage
from external_llm.providers import OllamaClient


class _FakeResp:
    """Minimal requests.Response stand-in for the non-streaming path."""

    def __init__(self, payload=None):
        self.status_code = 200
        self._payload = payload or {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }

    @property
    def text(self):
        return ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return []

    def close(self):
        return None


class _FakeSession:
    """Captures outgoing payloads without touching the network."""

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None, stream=False):
        self.calls.append((url, json))
        return _FakeResp()


IMAGE = {"media_type": "image/png", "data": "AAAA"}


def _client(monkeypatch) -> OllamaClient:
    """OllamaClient with the network stubbed and num_ctx/capability queries inert."""
    monkeypatch.setattr(
        "external_llm.ollama_api.query_ollama_num_ctx", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities", lambda *a, **k: None
    )
    client = OllamaClient(base_url="http://ollama.test")
    client._session = _FakeSession()  # type: ignore[assignment]
    return client


def _user_messages(images) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content="be brief"),
        LLMMessage(role="user", content="what is in this image?", images=images),
    ]


# ── chat_with_tools: the path the agent loop actually uses ───────────────────


def test_chat_with_tools_vision_model_sends_native_images(monkeypatch):
    client = _client(monkeypatch)
    client.chat_with_tools(_user_messages([IMAGE]), tools=[], model="llava:13b")
    url, payload = client._session.calls[-1]  # type: ignore[attr-defined]
    assert url.endswith("/api/chat")
    user_msg = payload["messages"][1]
    assert user_msg["images"] == ["AAAA"]
    assert user_msg["content"] == "what is in this image?"  # unchanged


def test_chat_with_tools_non_vision_model_gets_ocr_text(monkeypatch):
    client = _client(monkeypatch)
    client.chat_with_tools(
        _user_messages([IMAGE]), tools=[], model="qwen2.5-coder:3b"
    )
    _, payload = client._session.calls[-1]  # type: ignore[attr-defined]
    user_msg = payload["messages"][1]
    assert "images" not in user_msg
    assert "[Image 1" in user_msg["content"]
    assert "what is in this image?" in user_msg["content"]


def test_chat_with_tools_system_message_keeps_images_out(monkeypatch):
    """Images only ride on the user turn — the system turn stays clean."""
    client = _client(monkeypatch)
    client.chat_with_tools(
        _user_messages([IMAGE]), tools=[], model="llava:13b"
    )
    _, payload = client._session.calls[-1]  # type: ignore[attr-defined]
    sys_msg = payload["messages"][0]
    assert sys_msg["role"] == "system"
    assert "images" not in sys_msg


# ── parity: chat() and chat_with_tools() convert identically ────────────────


def test_chat_and_chat_with_tools_image_parity_vision(monkeypatch):
    client = _client(monkeypatch)
    client.chat(_user_messages([IMAGE]), model="llava:13b")
    _, chat_payload = client._session.calls[-1]  # type: ignore[attr-defined]
    client.chat_with_tools(_user_messages([IMAGE]), tools=[], model="llava:13b")
    _, tools_payload = client._session.calls[-1]  # type: ignore[attr-defined]

    def _user(messages):
        return next(m for m in messages if m.get("role") == "user")

    assert _user(chat_payload["messages"]) == _user(tools_payload["messages"])
    assert _user(tools_payload["messages"])["images"] == ["AAAA"]


def test_chat_and_chat_with_tools_image_parity_non_vision(monkeypatch):
    client = _client(monkeypatch)
    client.chat(_user_messages([IMAGE]), model="qwen2.5-coder:3b")
    _, chat_payload = client._session.calls[-1]  # type: ignore[attr-defined]
    client.chat_with_tools(_user_messages([IMAGE]), tools=[], model="qwen2.5-coder:3b")
    _, tools_payload = client._session.calls[-1]  # type: ignore[attr-defined]

    def _user(messages):
        return next(m for m in messages if m.get("role") == "user")

    assert _user(chat_payload["messages"]) == _user(tools_payload["messages"])
    assert "[Image 1" in _user(tools_payload["messages"])["content"]


# ── vision keyword detection (fast path) ─────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("llava:13b", True),
        ("qwen2.5vl:7b", True),  # no -vl/_vl separator — plain "vl" keyword
        ("internvl2:8b", True),
        ("pixtral:12b", True),
        ("qwen2.5-coder:3b", False),
        ("deepseek-r1:7b", False),
    ],
)
def test_ollama_vision_keyword_detection(monkeypatch, model, expected):
    from external_llm.model_registry import ollama_vision

    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities", lambda *a, **k: None
    )
    assert ollama_vision(model) is expected


# ── capability slow path (wired runtime detection) ───────────────────────────


def test_ollama_vision_slow_path_uses_capabilities(monkeypatch):
    """Non-keyword model resolved via /api/show capabilities — the previously
    dead second tier (the capability cache was never populated)."""
    from external_llm.model_registry import ollama_vision

    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities",
        lambda *a, **k: ("completion", "tools", "vision"),
    )
    assert ollama_vision("gemma3:27b") is True


def test_ollama_vision_slow_path_unknown_is_false(monkeypatch):
    from external_llm.model_registry import ollama_vision

    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities", lambda *a, **k: None
    )
    assert ollama_vision("gemma3:27b") is False


def test_ollama_supports_tools_wired_to_capabilities(monkeypatch):
    from external_llm.model_registry import ollama_supports_tools

    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities",
        lambda *a, **k: ("completion", "tools", "vision"),
    )
    assert ollama_supports_tools("gemma3:27b") is True

    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities",
        lambda *a, **k: ("completion",),
    )
    assert ollama_supports_tools("gemma3:27b") is False

    monkeypatch.setattr(
        "external_llm.model_registry.query_ollama_capabilities", lambda *a, **k: None
    )
    assert ollama_supports_tools("gemma3:27b") is None
