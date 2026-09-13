"""Bound how many tool-produced images stay visible in a running conversation.

A computer-use / screenshot loop re-sends its whole history every turn. Twenty
steps of "look, click, look" is twenty frames of a screen that has since
changed — roughly 32k tokens of them, at the provider's per-image pixel-geometry
price (``_shared_utils._IMAGE_BLOCK_TOKEN_ESTIMATE``). The cost is the smaller
problem: the model cannot tell which frame is *current* and will answer about
the wrong one, which is a correctness bug that looks like a reasoning failure.

So: keep the most recent DISTINCT images, elide the rest in place.

Two rules, both structural rather than heuristic:

* **Recency budget** — at most ``KEEP_RECENT_IMAGES`` images survive, counted
  newest-first. This is the bound that makes the loop affordable.
* **Content identity** — an image whose bytes already appear in a NEWER message
  is elided regardless of the budget. A loop that screenshots an unchanged
  screen re-sends identical bytes; keeping both would spend the budget twice on
  one frame. Identity is the payload digest, not the file path — the path is
  re-used for changed content and distinct paths can hold identical bytes.

Elision replaces the message's image list with ``images=None`` (or with the
survivors) and relabels the transport's own attachment note so the text stops
claiming a visibility the model no longer has. Text the policy did not write is
never touched — the note is recognized by its tag, and anything else on the
message survives verbatim (see ``_relabel_claim``).

Write discipline: copy-on-write via ``dataclasses.replace``, exactly like
``agent_turn_pipeline._stub_tool_result`` — a message object may be shared with
a past event payload or run record, and retroactively rewriting one is the bug
that discipline exists to prevent. Idempotent: an elided message has no images
left to elide.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from typing import Any

from .config.thresholds import _env_int
from .image_transport import ATTACHMENT_NOTE_TAG, supersede_attachment_note

logger = logging.getLogger(__name__)


KEEP_RECENT_IMAGES: int = _env_int("ASICODE_IMAGE_CONTEXT_KEEP", 3, minimum=0)
"""How many of the most recent distinct images stay visible to the model.

Three is the smallest number that covers the actual need: the frame the agent
is acting on, plus enough continuity to notice "the screen changed since I
clicked". ``minimum=0`` is accepted because zero is a meaningful setting — a
text-only route where every image must be elided as soon as it has been read
once (the tool result's text form is what the model works from)."""


_DIGEST_KEY = "_payload_digest"
"""Cache key on the image dict holding its content digest.

Stored on the dict exactly like ``providers._images_to_text`` caches
``ocr_text`` there — the dict is IN-MEMORY ONLY (``client.LLMMessage.images``),
and digesting a 400 KB base64 payload on every LLM call would otherwise put a
hash of every screenshot on the hot path of every turn. The key is not read by
any provider client, by the token estimator, or by ``_msg_token_fingerprint``.
"""


def _elision_note(kept: int, dropped: int) -> str:
    """Text replacing the transport's attachment note after an elision.

    Count-aware: a message that still shows some of its images must not claim
    they are all gone, and vice versa. Either way the note says where the text
    form lives, because that is what remains readable in later turns.
    """
    plural = "s" if dropped != 1 else ""
    if kept:
        # Reachable only with total >= 2, so "images" is always correct here.
        return (
            f"[{dropped} of {kept + dropped} images elided to bound context — "
            f"{kept} more recent image{'s' if kept != 1 else ''} from this message "
            f"{'remain' if kept != 1 else 'remains'} visible; "
            "the text form in the tool result describes all of them.]"
        )
    return (
        f"[{dropped} image{plural} elided to bound context — no longer visible to you; "
        "the text form in the tool result above is what remains. "
        "Re-take the screenshot / re-read the file if you still need to look.]"
    )


def _images_of(message: Any) -> list:
    """The image list on an ``LLMMessage`` or on a plain wire dict, else ``[]``."""
    images = message.get("images") if isinstance(message, dict) else getattr(message, "images", None)
    return images if isinstance(images, list) and images else []


def _content_of(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    return content if isinstance(content, str) else ""


def _payload_digest(image: Any) -> str:
    """Digest identifying an image's BYTES (cached on the dict — see _DIGEST_KEY).

    A non-dict element has no payload to identify; it gets a digest of its repr
    so two identical malformed entries still collapse, while a repaired one
    (the realistic fix) does not collide with its broken form.

    Only a digest derived from ``data`` is cached. A repr-derived digest is a
    function of the whole dict, so caching it would go stale the moment the dict
    gained its ``data`` — and a stale digest would make two different images
    look identical, silently eliding one of them.
    """
    if not isinstance(image, dict):
        return "malformed:" + hashlib.sha1(repr(image).encode("utf-8", "replace")).hexdigest()
    cached = image.get(_DIGEST_KEY)
    if isinstance(cached, str) and cached:
        return cached
    data = image.get("data")
    digest = hashlib.sha1((data if isinstance(data, str) else repr(image)).encode("utf-8", "replace")).hexdigest()
    if isinstance(data, str):
        try:
            image[_DIGEST_KEY] = digest
        except Exception:  # pragma: no cover - a frozen/odd mapping is still elidable
            logger.debug("could not cache image digest", exc_info=True)
    return digest


def _raw_content_of(message: Any) -> Any:
    return message.get("raw_content") if isinstance(message, dict) else getattr(message, "raw_content", None)


def _relabel_claim(message: Any, note: str) -> tuple[str, Any]:
    """Point this message's own text at *note*; return ``(content, raw_content)``.

    The claim is the tagged line ``image_transport`` wrote, and it lives wherever
    the message keeps its text — two carriers, one rule:

    * ``content`` — the OpenAI-family path, where the attachment is its own user
      message and the note IS its content.
    * a ``raw_content`` text block — the Anthropic/Gemini path, where the fold
      pushed the note into the turn carrying the tool results and ``content``
      must stay empty (``raw_content`` is authoritative there, so writing to
      ``content`` would be invisible).

    Rewriting it is the point of this pass: a message that still claims "you can
    see the pixels directly" after they were dropped is how a model ends up
    describing an image it never saw.

    Returns ``(content, raw_content)``; ``raw_content`` is a NEW list with a NEW
    block when it was written to, so the caller's copy-on-write holds.
    """
    content = _content_of(message)
    if content:
        return supersede_attachment_note(content, note), _raw_content_of(message)
    raw = _raw_content_of(message)
    if isinstance(raw, list):
        for idx, block in enumerate(raw):
            text = block.get("text") if isinstance(block, dict) else None
            if not isinstance(text, str) or not text.startswith(ATTACHMENT_NOTE_TAG):
                continue
            new_blocks = list(raw)
            new_blocks[idx] = {**block, "text": supersede_attachment_note(text, note)}
            return content, new_blocks
    if isinstance(raw, list) and raw:
        # Nothing in this message claims visibility (no note block), and the
        # remaining blocks keep the turn non-empty on their own — so there is
        # nothing to correct.
        return content, raw
    # The message is images and nothing else. Dropping them would send an EMPTY
    # user turn, which Anthropic and Gemini reject outright, so the elision note
    # becomes the message's text rather than a cosmetic annotation.
    return note, raw


def _with_images(message: Any, kept: list, note: str) -> Any:
    """A COPY of *message* showing only *kept* images, with its claim relabelled."""
    content, raw = _relabel_claim(message, note)
    if isinstance(message, dict):
        out = dict(message)
        out["images"] = kept or None
        out["content"] = content
        out["raw_content"] = raw
        return out
    return dataclasses.replace(message, images=(kept or None), content=content, raw_content=raw)


def apply_image_retention(messages: list, *, keep_recent: int | None = None) -> list:
    """Elide images beyond the retention budget or duplicated by a newer message.

    Walks newest → oldest so "most recent" is decided by position, not by any
    timestamp the message does not carry. Returns *messages* itself when nothing
    changed (callers must not assume a new list) and a new list otherwise; the
    untouched messages in it are the SAME objects.

    Cheap on the common path: one attribute read per message, and a digest only
    for messages that actually carry images (then cached on the image dict).
    """
    budget = KEEP_RECENT_IMAGES if keep_recent is None else max(0, int(keep_recent))
    if not messages:
        return messages

    seen: set[str] = set()
    remaining = budget
    out: list = messages
    elided_total = 0

    for idx in range(len(messages) - 1, -1, -1):
        images = _images_of(messages[idx])
        if not images:
            continue
        kept: list = []
        for image in images:
            digest = _payload_digest(image)
            if digest in seen or remaining <= 0:
                continue
            seen.add(digest)
            remaining -= 1
            kept.append(image)
        if len(kept) == len(images):
            continue
        dropped = len(images) - len(kept)
        elided_total += dropped
        if out is messages:
            out = list(messages)
        out[idx] = _with_images(messages[idx], kept, _elision_note(len(kept), dropped))

    if out is not messages:
        logger.info(
            "image retention: elided %d image(s) beyond the %d most recent distinct ones",
            elided_total,
            budget,
        )
    return out
