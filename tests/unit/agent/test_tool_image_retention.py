"""Contract tests for bounding the images a tool-loop keeps in context.

A computer-use loop re-sends its whole history every turn, so a screenshot per
step keeps every frame on the wire at ~1.6k tokens each AND leaves the model
guessing which frame is current. ``apply_image_retention`` keeps the most recent
DISTINCT images and elides the rest.

The properties sealed here are the ones whose regression is silent:

* the budget is GLOBAL across the conversation, not per message (a per-message
  reading of "keep 3" keeps 3 images per message and bounds nothing);
* an image whose bytes already appear in a newer message is elided even inside
  the budget — a loop that screenshots an unchanged screen must not spend the
  budget twice on one frame;
* the message's own claim is relabelled, wherever this message keeps its text,
  so the model is never told it can see pixels that were dropped;
* a message that would otherwise be left EMPTY gets the note as its text —
  Anthropic and Gemini reject an empty user turn outright;
* copy-on-write: the original message objects survive untouched, because a
  message may be shared with a past event payload or run record.
"""

from __future__ import annotations

import pytest

from external_llm.agent._shared_utils import estimate_tokens_from_msgs
from external_llm.agent.image_context_policy import (
    KEEP_RECENT_IMAGES,
    _payload_digest,
    apply_image_retention,
)
from external_llm.agent.image_transport import ATTACHMENT_NOTE_TAG, attached_images_note
from external_llm.client import LLMMessage


def _img(tag: str, media_type: str = "image/png") -> dict:
    return {"media_type": media_type, "data": f"payload-{tag}"}


def _attachment_msg(tag: str) -> LLMMessage:
    """Exactly what the transport emits for a tool image (OpenAI-family path)."""
    image = _img(tag)
    return LLMMessage(role="user", content=attached_images_note("screenshot", [image]), images=[image])


def _folded_msg(tag: str) -> LLMMessage:
    """What the Anthropic/Gemini fold emits: natives blocks + the note block."""
    image = _img(tag)
    return LLMMessage(
        role="user",
        content="",
        raw_content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "{}"},
            {"type": "text", "text": attached_images_note("screenshot", [image])},
        ],
        images=[image],
    )


def _images_of(msgs: list) -> list:
    return [len(m.images) for m in msgs if getattr(m, "images", None)]


def _payloads(images) -> list:
    """``(media_type, data)`` pairs — the fields that decide identity.

    Whole-dict comparison would also compare ``_payload_digest``, which the
    policy caches ON the image dict deliberately (the same in-memory discipline
    ``providers._images_to_text`` uses for ``ocr_text``), so a dict that has been
    through the pass is not equal to its pre-pass self by design.
    """
    return [(i.get("media_type"), i.get("data")) for i in images or []]


# ══════════════════════════════════════════════════════════════════════════════
# 1. The budget is global
# ══════════════════════════════════════════════════════════════════════════════


def test_budget_is_global_across_the_conversation():
    msgs = [_attachment_msg(f"frame{i}") for i in range(6)]

    out = apply_image_retention(msgs, keep_recent=3)

    assert _images_of(out) == [1, 1, 1]
    # The survivors are the three NEWEST, determined by position.
    assert [m for m in out if getattr(m, "images", None)] == [msgs[3], msgs[4], msgs[5]]


def test_default_budget_is_the_documented_constant():
    msgs = [_attachment_msg(f"frame{i}") for i in range(KEEP_RECENT_IMAGES + 4)]

    assert len(_images_of(apply_image_retention(msgs))) == KEEP_RECENT_IMAGES


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_zero_and_small_budgets(budget: int):
    msgs = [_attachment_msg(f"frame{i}") for i in range(4)]

    assert len(_images_of(apply_image_retention(msgs, keep_recent=budget))) == budget


def test_within_budget_is_a_noop_and_returns_the_same_list():
    msgs = [_attachment_msg("a"), _attachment_msg("b")]

    assert apply_image_retention(msgs, keep_recent=3) is msgs


def test_messages_without_images_are_untouched():
    keep = LLMMessage(role="user", content="hello")
    tool = LLMMessage(role="tool", content="{}")
    msgs = [keep, _attachment_msg("a"), tool, _attachment_msg("b"), _attachment_msg("c"), _attachment_msg("d")]

    out = apply_image_retention(msgs, keep_recent=1)

    assert out[0] is keep
    assert out[2] is tool


def test_empty_input():
    assert apply_image_retention([], keep_recent=3) == []


# ══════════════════════════════════════════════════════════════════════════════
# 2. Identity, not path: identical bytes collapse
# ══════════════════════════════════════════════════════════════════════════════


def test_identical_bytes_in_an_older_message_are_elided_inside_the_budget():
    """An unchanged screen re-screenshotted must not spend the budget twice."""
    msgs = [_attachment_msg("same"), _attachment_msg("a"), _attachment_msg("same")]

    out = apply_image_retention(msgs, keep_recent=3)

    # The newest occurrence survives; the older duplicate does not, even though
    # the budget would have allowed it.
    assert getattr(out[0], "images", None) is None
    assert _payloads(getattr(out[1], "images", None)) == [("image/png", "payload-a")]
    assert _payloads(getattr(out[2], "images", None)) == [("image/png", "payload-same")]


def test_distinct_bytes_are_never_collapsed():
    msgs = [_attachment_msg("a"), _attachment_msg("b")]

    assert len(_images_of(apply_image_retention(msgs, keep_recent=5))) == 2


def test_digest_is_cached_on_the_image_dict_and_survives_reuse():
    msgs = [_attachment_msg("x")]

    apply_image_retention(msgs, keep_recent=1)

    image = msgs[0].images[0]
    assert image.get("_payload_digest")
    # A second pass over the same dicts must reach the same verdict (the digest
    # is read back, not recomputed).
    assert apply_image_retention(msgs, keep_recent=1) is msgs


def test_repr_derived_digests_are_never_cached():
    """Only a digest derived from ``data`` may be cached.

    A repr-derived digest is a function of the whole dict, so caching it goes
    stale the moment the dict gains its payload — and two images that then look
    identical get one of them silently elided.
    """
    broken = {"media_type": "image/png"}
    first = _payload_digest(broken)

    assert "_payload_digest" not in broken

    broken["data"] = "payload-x"
    assert _payload_digest(broken) != first


def test_content_derived_digests_are_cached():
    image = _img("cache-me")
    first = _payload_digest(image)

    assert image["_payload_digest"] == first
    assert _payload_digest(image) == first


def test_partial_elision_note_agrees_grammatically():
    """A message that still shows one of its images must read as such."""
    one_left = LLMMessage(role="user", content="c", images=[_img("a"), _img("b")])
    none_left = LLMMessage(role="user", content="c", images=[_img("c"), _img("d")])

    kept_one = apply_image_retention([one_left], keep_recent=1)[0].content
    kept_none = apply_image_retention([none_left], keep_recent=0)[0].content

    assert "1 of 2 images elided" in kept_one
    assert "remains visible" in kept_one
    assert "no longer visible" in kept_none


def test_malformed_image_elements_are_elidable():
    msgs = [LLMMessage(role="user", content="c", images=[None]), LLMMessage(role="user", content="c", images=["x"])]

    out = apply_image_retention(msgs, keep_recent=1)

    assert getattr(out[0], "images", None) is None
    assert getattr(out[1], "images", None) == ["x"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. The claim is relabelled wherever this message keeps its text
# ══════════════════════════════════════════════════════════════════════════════


def test_claim_in_content_is_relabelled():
    msgs = [_attachment_msg("old"), _attachment_msg("new")]

    out = apply_image_retention(msgs, keep_recent=1)

    assert not out[0].content.startswith(ATTACHMENT_NOTE_TAG)
    assert "elided" in out[0].content
    # The survivor's claim is untouched.
    assert out[1].content.startswith(ATTACHMENT_NOTE_TAG)


def test_claim_in_a_raw_content_text_block_is_relabelled():
    """The Anthropic/Gemini carrier: `content` is ignored there, so the note is
    a block and THAT is what must stop claiming visibility."""
    msgs = [_folded_msg("old"), _folded_msg("new")]

    out = apply_image_retention(msgs, keep_recent=1)

    assert out[0].content == ""
    text_blocks = [b for b in out[0].raw_content if b.get("type") == "text"]
    assert len(text_blocks) == 1
    assert not text_blocks[0]["text"].startswith(ATTACHMENT_NOTE_TAG)
    assert "elided" in text_blocks[0]["text"]
    # The tool_result block is preserved — the assistant/tool pairing must hold.
    assert out[0].raw_content[0]["type"] == "tool_result"
    assert out[0].raw_content[0]["tool_use_id"] == "t1"
    # The survivor is untouched.
    assert any(b.get("text", "").startswith(ATTACHMENT_NOTE_TAG) for b in out[1].raw_content)


def test_foreign_text_is_appended_not_replaced():
    msgs = [
        LLMMessage(role="user", content="the user's own words", images=[_img("a")]),
        LLMMessage(role="user", content="", images=[_img("b")]),
    ]

    out = apply_image_retention(msgs, keep_recent=0)

    assert out[0].content.startswith("the user's own words")
    assert "elided" in out[0].content


def test_raw_content_without_a_claim_keeps_content_empty():
    """Nothing claimed visibility, and the remaining blocks keep the turn
    non-empty — so there is nothing to correct."""
    msgs = [
        LLMMessage(
            role="user",
            content="",
            raw_content=[{"functionResponse": {"name": "read_image", "response": {"content": "{}"}}}],
            images=[_img("a")],
        ),
        LLMMessage(role="user", content="c", images=[_img("b")]),
    ]

    out = apply_image_retention(msgs, keep_recent=1)

    assert out[0].content == ""
    assert out[0].raw_content == [{"functionResponse": {"name": "read_image", "response": {"content": "{}"}}}]


def test_an_image_only_message_never_becomes_empty():
    """Dropping the only images from a text-less turn would send an empty user
    message, which Anthropic and Gemini reject."""
    msgs = [
        LLMMessage(role="user", content="", images=[_img("a")]),
        LLMMessage(role="user", content="c", images=[_img("b")]),
    ]

    out = apply_image_retention(msgs, keep_recent=1)

    assert out[0].content
    assert "elided" in out[0].content


# ══════════════════════════════════════════════════════════════════════════════
# 4. Write discipline
# ══════════════════════════════════════════════════════════════════════════════


def test_copy_on_write_leaves_the_originals_intact():
    msgs = [_attachment_msg("old"), _attachment_msg("new")]

    out = apply_image_retention(msgs, keep_recent=1)

    assert out is not msgs
    assert out[0] is not msgs[0]
    # The ORIGINAL still shows its pixels and its original claim: a message may
    # be shared with a past event payload or run record.
    assert _payloads(msgs[0].images) == [("image/png", "payload-old")]
    assert msgs[0].content.startswith(ATTACHMENT_NOTE_TAG)
    # The survivor is the SAME object, not a copy.
    assert out[1] is msgs[1]


def test_idempotent():
    msgs = [_attachment_msg(f"f{i}") for i in range(4)]

    once = apply_image_retention(msgs, keep_recent=1)
    twice = apply_image_retention(once, keep_recent=1)

    assert twice is once


def test_dict_messages_are_supported():
    msgs = [
        {"role": "user", "content": "", "images": [_img("a")]},
        {"role": "user", "content": "c", "images": [_img("b")]},
    ]

    out = apply_image_retention(msgs, keep_recent=1)

    assert out[0]["images"] is None
    assert "elided" in out[0]["content"]
    assert out[0] is not msgs[0]
    assert _payloads(msgs[0]["images"]) == [("image/png", "payload-a")]


# ══════════════════════════════════════════════════════════════════════════════
# 5. The point of the pass: the estimate actually drops
# ══════════════════════════════════════════════════════════════════════════════


def test_elision_reduces_the_token_estimate():
    msgs = [_attachment_msg(f"f{i}") for i in range(5)]
    before = estimate_tokens_from_msgs(msgs)

    after = estimate_tokens_from_msgs(apply_image_retention(msgs, keep_recent=1))

    # Each image is billed at the provider cap, so eliding four frees roughly
    # 4x that. The four elision notes cost a little back, hence "> 3x".
    assert before - after > 3 * 1600
    # ...and the survivor is still billed — elision must not zero the estimate.
    assert after >= 1600
