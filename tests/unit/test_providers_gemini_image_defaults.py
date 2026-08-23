"""Regression for B1 — GoogleClient.chat_with_tools image inlineData defaults.

Before the fix, ``chat_with_tools`` was the *only* one of four sibling
image-handling sites in providers.py that read the image dict with direct
subscript (``img["media_type"]`` / ``img["data"]``). The other three —
``chat`` (L437) and two streaming spots — all used defensive ``.get(default)``.
A malformed image dict (e.g. one built outside ``image_utils`` that omits a
key) therefore raised ``KeyError`` on the tool path while the plain ``chat``
path degraded gracefully. B1 aligns the tool path to the same ``.get()``
defaults so both paths agree.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from external_llm.client import LLMMessage
from external_llm.providers import GoogleClient


def _make_client() -> GoogleClient:
    """A GoogleClient whose session always returns a minimal 200 text response."""
    client = GoogleClient(api_key="test-key")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
    client._session = MagicMock()
    client._session.post.return_value = fake_response
    return client


def test_chat_with_tools_image_defaults_when_keys_missing() -> None:
    """A malformed image dict (no media_type/data) must NOT raise KeyError.

    Regression target: before B1 this raised KeyError on the direct subscript.
    After B1 it degrades to mimeType="image/png", data="" — matching chat().
    """
    client = _make_client()
    msg = LLMMessage(role="user", content="describe", images=[{"ocr_text": "x"}])

    # Must not raise
    client.chat_with_tools(messages=[msg], tools=[], model="gemini-2.5-flash")

    # chat_with_tools builds parts as [inlineData, text] — inlineData at index 0
    sent = client._session.post.call_args.kwargs["json"]
    inline = sent["contents"][0]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert inline["data"] == ""
    assert sent["contents"][0]["parts"][1]["text"] == "describe"


def test_chat_with_tools_image_preserves_provided_keys() -> None:
    """A well-formed image dict passes its real values through (parity)."""
    client = _make_client()
    msg = LLMMessage(
        role="user",
        content="describe",
        images=[{"media_type": "image/jpeg", "data": "QUJDRA=="}],
    )
    client.chat_with_tools(messages=[msg], tools=[], model="gemini-2.5-flash")

    sent = client._session.post.call_args.kwargs["json"]
    inline = sent["contents"][0]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"
    assert inline["data"] == "QUJDRA=="


def test_chat_and_chat_with_tools_image_paths_agree_on_defaults() -> None:
    """Both image-handling paths produce identical defaulted inlineData.

    This is the cross-path contract B1 restores: a malformed dict no longer
    crashes one path (chat_with_tools) while succeeding on the other (chat).

    Note the two paths differ in *part ordering* (chat = [text, inlineData];
    chat_with_tools = [inlineData, text]) — a pre-existing structural
    difference outside B1's scope — so we compare the inlineData values, not
    the parts lists.
    """
    malformed = {"ocr_text": "y"}

    c1 = _make_client()
    c1.chat(
        messages=[LLMMessage(role="user", content="x", images=[dict(malformed)])],
        model="gemini-2.5-flash",
    )
    # chat(): parts = [text, inlineData] → inlineData at index 1
    chat_inline = c1._session.post.call_args.kwargs["json"]["contents"][0]["parts"][1]["inlineData"]

    c2 = _make_client()
    c2.chat_with_tools(
        messages=[LLMMessage(role="user", content="x", images=[dict(malformed)])],
        tools=[],
        model="gemini-2.5-flash",
    )
    # chat_with_tools(): parts = [inlineData, text] → inlineData at index 0
    tools_inline = c2._session.post.call_args.kwargs["json"]["contents"][0]["parts"][0]["inlineData"]

    assert chat_inline == tools_inline == {"mimeType": "image/png", "data": ""}
