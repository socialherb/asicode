"""Contract tests for images a TOOL produced (screenshots) reaching the model.

Until this transport existed, an image could only enter a conversation by being
typed/pasted by the USER: ``read_image`` — the one tool whose entire job is
looking at pixels — could only answer with OCR text. Computer use needs the
other direction.

Three properties are sealed here, because each one is a silent failure if it
regresses:

1. **The pixels never enter the tool-result JSON payload.** ``data`` is a full
   base64 blob; inside the payload text the model would be billed ~130k tokens
   for one screenshot (pixel geometry, not payload length, is the real cost —
   ``_shared_utils._IMAGE_BLOCK_TOKEN_ESTIMATE``) and any transcript that
   serialises a tool result would write pixels to disk.
2. **Each provider gets them in a form it accepts.** Anthropic and Gemini reject
   two user turns in a row, so the images are folded into the ONE user turn that
   already carries the tool results; the OpenAI family cannot put a content-parts
   list inside ``role="tool"``, so there they ride a trailing user message.
3. **A route that cannot take images is TOLD so.** Being served OCR as if that
   were the whole answer is how a model ends up describing an image it never
   saw.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.image_transport import (
    ATTACHMENT_NOTE_TAG,
    MAX_ATTACHED_IMAGES,
    attached_images_note,
    attachment_messages_for,
    pop_attached_images,
    read_attached_images,
    sanitize_attached_images,
    supersede_attachment_note,
    unviewable_images_note,
)
from external_llm.agent.tool_registry import ToolResult
from external_llm.client import LLMMessage
from external_llm.providers import GoogleClient

_VISION_ROUTE = "claude-sonnet-4-5"
_TEXT_ONLY_ROUTE = "deepseek-chat"
_IMAGE = {"media_type": "image/png", "data": "QUJDRA=="}
_CAPTION = "shot.png"


def _payload(**over) -> dict:
    meta = {"attach_images": [dict(_IMAGE, caption=_CAPTION)]}
    meta.update(over.pop("metadata", {}))
    return meta


def _meta(images=None) -> dict:
    if images is None:
        images = [dict(_IMAGE, caption=_CAPTION)]
    return {"attach_images": images, "width": 1280}


def _tool_result(**over) -> ToolResult:
    kw = {"ok": True, "content": "[Image OCR — shot.png]\nhello", "metadata": _meta()}
    kw.update(over)
    return ToolResult(**kw)


def _loop() -> AgentLoop:
    """Bare AgentLoop: ``_build_tool_result_message`` only needs its own helpers."""
    return AgentLoop.__new__(AgentLoop)


def _loop_for(provider: str) -> AgentLoop:
    """Bare AgentLoop whose provider name drives ``_append_native_tool_messages``."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.llm_client = SimpleNamespace(get_provider_name=lambda: provider)
    return loop


def _attachment(model: str = _VISION_ROUTE, provider: str = "") -> LLMMessage:
    msgs = attachment_messages_for("read_image", _meta(), model=model, base_url="", provider=provider)
    assert len(msgs) == 1
    return msgs[0]


# ══════════════════════════════════════════════════════════════════════════════
# 1. The payload never carries the pixels
# ══════════════════════════════════════════════════════════════════════════════


def test_tool_result_payload_strips_the_attachment_from_metadata():
    """The base64 must not exist anywhere in the JSON text the model reads."""
    msg = _loop()._build_tool_result_message("t1", "read_image", _tool_result())

    payload = json.loads(msg.content)
    assert "attach_images" not in payload["metadata"]
    assert _IMAGE["data"] not in msg.content
    # Everything else the tool declared survives — only the pixels leave.
    assert payload["metadata"]["width"] == 1280
    assert payload["ok"] is True
    # ...and the message itself is a plain tool result, carrying no pixels.
    assert msg.role == "tool"
    assert msg.images is None


def test_tool_result_payload_leaves_the_caller_metadata_intact():
    """``_build_tool_result_message`` strips a COPY: the caller still needs the
    declaration to build the attachment message (same turn, right after)."""
    result = _tool_result()
    _loop()._build_tool_result_message("t1", "read_image", result)

    assert len(result.metadata["attach_images"]) == 1
    assert read_attached_images(result.metadata)[0]["data"] == _IMAGE["data"]


def test_pop_attached_images_removes_the_declaration():
    meta = _meta()
    images = pop_attached_images(meta)

    assert [i["data"] for i in images] == [_IMAGE["data"]]
    assert "attach_images" not in meta
    assert images[0]["caption"] == _CAPTION


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sanitizer: a malformed declaration costs the image, never the turn
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not-a-list",
        {"data": "x"},
        42,
    ],
)
def test_sanitize_rejects_a_non_list_declaration(raw):
    assert sanitize_attached_images(raw) == []


def test_sanitize_drops_malformed_entries_and_unknown_keys():
    out = sanitize_attached_images(
        [
            7,
            {"media_type": "image/png"},  # no data
            {"data": ""},  # empty data
            {"data": "QQ==", "media_type": "image/jpeg", "junk": "dropped"},
        ]
    )

    assert out == [{"media_type": "image/jpeg", "data": "QQ=="}]


def test_sanitize_defaults_the_media_type():
    assert sanitize_attached_images([{"data": "QQ=="}])[0]["media_type"] == "image/png"


def test_sanitize_caps_the_image_count():
    out = sanitize_attached_images([{"data": f"data-{i}"} for i in range(MAX_ATTACHED_IMAGES + 3)])

    assert len(out) == MAX_ATTACHED_IMAGES


def test_read_attached_images_is_non_destructive():
    meta = _meta()
    read_attached_images(meta)

    assert "attach_images" in meta


def test_sanitize_tolerates_a_non_dict_metadata():
    assert read_attached_images(None) == []
    assert pop_attached_images("nope") == []


# ══════════════════════════════════════════════════════════════════════════════
# 3. The attachment message + the route gate
# ══════════════════════════════════════════════════════════════════════════════


def test_no_declaration_produces_no_message():
    assert attachment_messages_for("read_file", {"lines": 3}, model=_VISION_ROUTE, base_url="") == []


def test_vision_route_gets_the_pixels():
    msg = _attachment()

    assert msg.role == "user"
    assert msg.images == [dict(_IMAGE, caption=_CAPTION)]
    assert msg.content.startswith(ATTACHMENT_NOTE_TAG)
    assert _CAPTION in msg.content


def test_text_only_route_is_told_the_pixels_were_dropped():
    """Silence would let the model describe an image it never received."""
    (msg,) = attachment_messages_for("read_image", _meta(), model=_TEXT_ONLY_ROUTE, base_url="")

    assert msg.images is None
    assert _TEXT_ONLY_ROUTE in msg.content
    assert "NOT attached" in msg.content


def test_unviewable_note_agrees_with_the_count():
    assert "1 image produced" in unviewable_images_note("read_image", 1)
    assert "2 images produced" in unviewable_images_note("read_image", 2)
    assert "were NOT attached" in unviewable_images_note("read_image", 2)
    assert "was NOT attached" in unviewable_images_note("read_image", 1)


def test_attached_note_counts_and_lists_captions():
    note = attached_images_note("screenshot", [dict(_IMAGE, caption="a.png"), dict(_IMAGE, caption="b.png")])

    assert note.startswith(f"{ATTACHMENT_NOTE_TAG} 2 images attached by tool 'screenshot'")
    assert "  1. a.png" in note
    assert "  2. b.png" in note


# ══════════════════════════════════════════════════════════════════════════════
# 4. Per-provider fold — alternation-safe
# ══════════════════════════════════════════════════════════════════════════════


def _fold(provider: str, *, with_images: bool = True) -> list:
    """One assistant(tool_call) turn + its tool result (+ the attachment).

    The attachment is built through the same gate the pipeline uses, provider
    included — so this mirrors production rather than a hand-made message.
    """
    tool_msg = _loop()._build_tool_result_message("t1", "read_image", _tool_result())
    extra = attachment_messages_for("read_image", _meta(), model=_VISION_ROUTE, base_url="", provider=provider)
    results = [tool_msg, *extra] if with_images else [tool_msg]
    # A raw response carrying the tool_call, so the provider branches that
    # require assistant(tool_calls) -> tool adjacency actually emit the pair.
    raw = SimpleNamespace(
        raw_response={"choices": [{"message": {"tool_calls": [{"id": "t1", "function": {"name": "read_image"}}]}}]}
    )
    # Ollama is the one branch that reads the NORMALISED tool_calls instead of
    # the provider's raw response (it has no tool_call_id to match on).
    response: dict = {"tool_calls": [{"name": "read_image", "args": {}}]} if provider == "ollama" else {"raw": raw}
    return _loop_for(provider)._append_native_tool_messages([], response, results)


@pytest.mark.parametrize("provider", ["anthropic", "zai"])
def test_anthropic_folds_images_into_the_tool_result_turn(provider):
    """A second user turn is an HTTP 400 on Anthropic — the image must join the
    turn that already carries the tool_result blocks, and come AFTER them."""
    out = _fold(provider)

    assert [m.role for m in out] == ["assistant", "user"]
    last = out[-1]
    assert last.images == [dict(_IMAGE, caption=_CAPTION)]
    assert last.raw_content[0]["type"] == "tool_result"
    assert last.raw_content[0]["tool_use_id"] == "t1"
    # The note the transport wrote is folded as a text block (raw_content is
    # authoritative for text, so the fold must not rely on `content`).
    assert any(b.get("type") == "text" and b["text"].startswith(ATTACHMENT_NOTE_TAG) for b in last.raw_content)


def test_google_tool_turn_gains_no_extra_part():
    """Gemini requires the function-response part count to EQUAL the function-call
    part count, so nothing may be appended to that turn — not the image, and not
    the note either (the fold would turn the attachment message into a sibling
    text part, which is the same 400 by the same rule)."""
    out = _fold("google")

    assert [m.role for m in out] == ["assistant", "user"]
    last = out[-1]
    assert last.images is None
    assert len(last.raw_content) == 1
    assert "functionResponse" in last.raw_content[0]
    assert not any(b.get("type") == "text" or "text" in b for b in last.raw_content)


def test_google_is_excluded_at_the_source():
    """Excluded by the transport, not by the fold: that is what keeps the fold
    from minting a sibling part out of the attachment note."""
    assert (
        attachment_messages_for("read_image", _meta(), model="gemini-2.5-flash", base_url="", provider="google") == []
    )
    assert attachment_messages_for("read_image", _meta(), model="gemini-3-flash", base_url="", provider="GOOGLE ") == []


@pytest.mark.parametrize("provider", ["openai", "deepseek", "opencode", "ollama"])
def test_openai_family_appends_the_attachment_as_a_trailing_user_turn(provider):
    out = _fold(provider)

    assert [m.role for m in out] == ["assistant", "tool", "user"]
    assert out[-1].images == [dict(_IMAGE, caption=_CAPTION)]


def test_generic_fallback_keeps_the_pixels_on_its_single_user_turn():
    out = _fold("some-unknown-provider")

    assert [m.role for m in out] == ["assistant", "user"]
    assert out[-1].images == [dict(_IMAGE, caption=_CAPTION)]
    assert "[Image OCR" in out[-1].content


def test_fold_without_images_is_unchanged():
    """The overwhelmingly common case: no tool produced pixels, no extra turn."""
    out = _fold("anthropic", with_images=False)

    assert [m.role for m in out] == ["assistant", "user"]
    assert out[-1].images is None
    assert [b["type"] for b in out[-1].raw_content] == ["tool_result"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Wire level — the renderers actually emit the parts
# ══════════════════════════════════════════════════════════════════════════════


def test_anthropic_wire_payload_nests_the_image_inside_the_tool_result():
    """The documented form ("Example of tool result with images"): the image is
    a content block INSIDE the tool_result, not a sibling of it. Sibling
    placement is undocumented for images, and a screenshot that 400s on its
    first call costs the whole turn."""
    from external_llm.anthropic_client import AnthropicClient

    captured: dict = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {}}
        return resp

    client = AnthropicClient(api_key="test")
    client._session = MagicMock()
    client._session.post = _post

    client.chat_with_tools(messages=_fold("anthropic"), tools=[], model="claude-sonnet-4-5")

    blocks = captured["messages"][-1]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "t1"
    content = blocks[0]["content"]
    assert isinstance(content, list), "the string form must be widened to blocks"
    # The tool's own text stays first inside the result — it is the result.
    assert content[0]["type"] == "text"
    image = [b for b in content if b.get("type") == "image"]
    assert len(image) == 1
    assert image[0]["source"] == {"type": "base64", "media_type": "image/png", "data": _IMAGE["data"]}
    # No sibling part was invented on the turn itself.
    assert not any(b.get("type") == "image" for b in blocks)


def test_anthropic_image_lands_on_the_last_tool_result_of_the_turn():
    """Mapping an image back to the call that produced it is not recoverable at
    this layer; nesting under the last result is the deterministic choice, and
    the transport's note names the tool so nothing misleads the model."""
    from external_llm.anthropic_client import _ride_along_images

    blocks = [
        {"type": "tool_result", "tool_use_id": "a", "content": "{}"},
        {"type": "tool_result", "tool_use_id": "b", "content": "{}"},
    ]
    out = _ride_along_images(blocks, [dict(_IMAGE)], "user")

    assert [b["tool_use_id"] for b in out] == ["a", "b"]
    assert isinstance(out[0]["content"], str)
    assert isinstance(out[1]["content"], list)


def test_anthropic_merge_copies_the_result_it_touches():
    """Copy-on-write: the message's stored raw_content must not be rewritten."""
    from external_llm.anthropic_client import _ride_along_images

    original = {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}
    raw = [original]

    out = _ride_along_images(raw, [dict(_IMAGE)], "user")

    assert original["content"] == "{}"
    assert out[0] is not original
    assert raw == [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]


def test_anthropic_image_without_a_tool_result_is_still_rendered():
    """A plain image turn (no tool result to nest into) keeps the images as
    ordinary user-turn content."""
    from external_llm.anthropic_client import _ride_along_images

    out = _ride_along_images([{"type": "text", "text": "look"}], [dict(_IMAGE)], "user")

    assert out[0] == {"type": "text", "text": "look"}
    assert out[1]["type"] == "image"


def test_anthropic_wire_payload_still_ignores_content_when_raw_content_is_present():
    """Pinned contract (test_chat_with_tools_default_model_and_message_shaping):
    only IMAGES merge into a native turn — `content` stays a mirror that must
    not be sent twice."""
    from external_llm.anthropic_client import _ride_along_images

    raw = [{"type": "text", "text": "native block"}]
    assert _ride_along_images(raw, None, "user") is raw
    assert _ride_along_images(raw, [], "assistant") is raw
    # An assistant turn never gains image blocks, even with images attached.
    assert _ride_along_images(raw, [dict(_IMAGE)], "assistant") is raw


def test_gemini_wire_payload_keeps_one_part_per_function_call():
    """The invariant Gemini enforces (400 INVALID_ARGUMENT otherwise):
    "the number of function response parts should be equal to number of function
    call parts of the function call turn". The turn has one function call, so it
    must carry exactly one part."""
    client = GoogleClient(api_key="test-key")
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
    client._session = MagicMock()
    client._session.post.return_value = resp

    client.chat_with_tools(messages=_fold("google"), tools=[], model="gemini-2.5-flash")

    parts = client._session.post.call_args.kwargs["json"]["contents"][-1]["parts"]
    assert len(parts) == 1
    assert "functionResponse" in parts[0]


def test_openai_wire_content_carries_the_data_url():
    from external_llm.openai_client import _openai_content

    content = _openai_content(_attachment(), "gpt-4o", "")

    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == f"data:image/png;base64,{_IMAGE['data']}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. The claim line, and who may rewrite it
# ══════════════════════════════════════════════════════════════════════════════


def test_supersede_rewrites_only_the_tagged_line():
    note = attached_images_note("read_image", [dict(_IMAGE, caption=_CAPTION)])
    out = supersede_attachment_note(note, "[X] elided")

    assert out.startswith("[X] elided")
    # The caption line is not the claim and survives.
    assert f"  1. {_CAPTION}" in out


def test_supersede_appends_to_text_it_did_not_write():
    """Foreign text is never replaced — only appended to."""
    assert supersede_attachment_note("the user's own words", "[X]") == "the user's own words\n[X]"
    assert supersede_attachment_note("", "[X]") == "[X]"


def test_supersede_ignores_a_tag_that_is_not_at_the_start():
    assert supersede_attachment_note(f"prefix {ATTACHMENT_NOTE_TAG} suffix", "[X]") == (
        f"prefix {ATTACHMENT_NOTE_TAG} suffix\n[X]"
    )
