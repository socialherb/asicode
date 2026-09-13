"""Transport for images a TOOL produced — screenshots, charts, rendered diagrams.

Until now an image could only enter a conversation by being typed/pasted by the
USER: ``LLMMessage.images`` is populated in the REPL/webapp and every provider
client already knows how to render it (Anthropic ``image`` blocks, Gemini
``inlineData``, OpenAI ``image_url``). ``read_image`` — the one tool whose whole
job is looking at pixels — downgraded them to OCR text, because there was no way
to hand a tool's pixels back to the model.

This module owns that path end to end:

    tool handler            -> ToolResult.metadata[ATTACH_IMAGES_KEY]
    _build_tool_result_message -> the key is STRIPPED from the JSON payload
    _process_tool_results   -> one synthetic user message (``images=`` attached)
    _append_native_tool_messages -> folded per provider (alternation-safe)
    provider client         -> native wire form

Why the pixels travel on a separate USER message
------------------------------------------------
``role="tool"`` payloads cannot carry them portably:

* OpenAI-compatible endpoints (and Ollama) require ``tool`` content to be a
  plain string — a content-parts list there is an HTTP 400.
* Anthropic *does* accept images alongside a tool result, but it enforces
  strict user/assistant alternation, so a *second* user turn after the tool
  results is a 400 as well. There the message is folded into the single user
  turn that already carries the results (``_append_native_tool_messages``) and
  the client nests the pixels inside the ``tool_result`` block — the documented
  form for a tool result with images.
* Gemini is excluded entirely (:data:`NO_TOOL_IMAGE_PROVIDERS`): its
  function-response turn must carry exactly one part per function call, so
  neither the image nor even a note about it may be appended there.

A user message satisfies the rest: OpenAI-family providers append it after the
tool block (valid), Anthropic folds it in (valid).

Why the key is popped, not merely read
-------------------------------------
``data`` is a full base64 payload. Left inside ``metadata`` it would be
serialised into the JSON text the model reads, where a 300 KB screenshot is
counted as ~130k tokens (see ``_shared_utils._IMAGE_BLOCK_TOKEN_ESTIMATE`` for
why pixel geometry, not payload length, is the real cost) and would be written
to disk by anything that persists a tool result. :func:`pop_attached_images`
removes it from the payload; the images live only on the message object, which
is already documented as IN-MEMORY ONLY (``client.LLMMessage.images``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


ATTACH_IMAGES_KEY = "attach_images"
"""``ToolResult.metadata`` key a tool writes to hand the model its pixels.

Shape — the same dict shape ``LLMMessage.images`` uses, so provider clients need
no translation: ``[{"media_type": "image/png", "data": "<base64>", "caption": ?}]``.
``caption`` is optional and is the only field that reaches the *text* of the
message, so a tool should put anything the model must be able to read later
(dimensions, the path it saved, which display) there.
"""

MAX_ATTACHED_IMAGES = 4
"""Per tool call. Every provider caps images per request (Anthropic: 100); the
real constraint is the context budget at ~1.6k tokens per image, so a tool that
returns more than a handful is mis-designed rather than merely expensive."""

NO_TOOL_IMAGE_PROVIDERS = frozenset({"google"})
"""Providers whose tool-result turn cannot carry an image, so none is offered.

Gemini requires the number of function-response parts to EQUAL the number of
function-call parts of that turn: "Please ensure that the number of function
response parts is equal to the number of function call parts of the function
call turn" (400 INVALID_ARGUMENT). An extra sibling part is therefore a hard
failure, not a degraded one — google's own gemini-cli files its sibling-part
fallback as a bug for exactly this reason (issue #16135), and for models
without multimodal function responses instructs clients to "not return sibling
parts for tool responses".

The supported alternative is nesting the image INSIDE the ``functionResponse``
part, which is Gemini-3-and-later only. Until this repo declares that axis
(``ModelCapabilities`` + the JS/Python parity gate, the project's normal route
for a capability), Gemini works from the tool result's text form — the same
content every other provider gets alongside its pixels.

Excluding the provider at the SOURCE rather than at the fold matters: the fold
would otherwise turn the attachment message into a sibling text part, which is
the same 400 by the same rule.
"""

MAX_IMAGE_B64_LEN = 5_000_000
"""Per image, in base64 characters (~3.7 MB of pixels). Sits deliberately under
Anthropic's 5 MB *binary* limit: the declared size is the base64 string, and a
payload that 400s would cost the whole turn, not just the image."""

ATTACHMENT_NOTE_TAG = "[TOOL IMAGE]"
"""Prefix of the note line :func:`attached_images_note` writes, and the ONLY
shape :func:`supersede_attachment_note` will rewrite.

A tag rather than matched prose: the retention policy must recognize exactly the
line this module produced, never a sentence a user or another tool happened to
write. Keeping the tag next to the function that emits it is what makes that
guarantee checkable — see the shape test in
``tests/unit/agent/test_tool_image_retention.py``."""

DEFAULT_MEDIA_TYPE = "image/png"

_MEDIA_TYPE_KEY = "media_type"
_DATA_KEY = "data"
_CAPTION_KEY = "caption"


def sanitize_attached_images(raw: Any) -> list[dict[str, str]]:
    """Validate and trim a declared attachment list. Never raises.

    A tool handler is arbitrary code, and a malformed declaration must cost the
    image — not the turn: an exception here would surface as a failed tool call
    whose cause is a screenshot nobody asked for. Elements that are not dicts,
    carry no ``data``, or exceed :data:`MAX_IMAGE_B64_LEN` are dropped with a
    warning; only the three known fields are copied through, so a handler cannot
    smuggle a stray key into the message object.
    """
    if not isinstance(raw, list):
        if raw is not None:
            logger.warning("%s must be a list, got %s — ignoring", ATTACH_IMAGES_KEY, type(raw).__name__)
        return []
    out: list[dict[str, str]] = []
    for entry in raw:
        if len(out) >= MAX_ATTACHED_IMAGES:
            logger.warning(
                "%s declared more than %d images — keeping the first %d",
                ATTACH_IMAGES_KEY,
                MAX_ATTACHED_IMAGES,
                MAX_ATTACHED_IMAGES,
            )
            break
        if not isinstance(entry, dict):
            logger.warning("%s entry is %s, not a dict — dropping", ATTACH_IMAGES_KEY, type(entry).__name__)
            continue
        data = entry.get(_DATA_KEY)
        if not isinstance(data, str) or not data:
            logger.warning("%s entry carries no %r string — dropping", ATTACH_IMAGES_KEY, _DATA_KEY)
            continue
        if len(data) > MAX_IMAGE_B64_LEN:
            logger.warning(
                "%s entry is %d base64 chars (limit %d) — dropping the image",
                ATTACH_IMAGES_KEY,
                len(data),
                MAX_IMAGE_B64_LEN,
            )
            continue
        media_type = entry.get(_MEDIA_TYPE_KEY) or DEFAULT_MEDIA_TYPE
        img: dict[str, str] = {
            _MEDIA_TYPE_KEY: str(media_type),
            _DATA_KEY: data,
        }
        caption = entry.get(_CAPTION_KEY)
        if caption:
            img[_CAPTION_KEY] = str(caption)
        out.append(img)
    return out


def read_attached_images(metadata: Any) -> list[dict[str, str]]:
    """Non-destructive read of a tool result's declared images."""
    if not isinstance(metadata, dict):
        return []
    return sanitize_attached_images(metadata.get(ATTACH_IMAGES_KEY))


def pop_attached_images(metadata: Any) -> list[dict[str, str]]:
    """Remove ``attach_images`` from *metadata* and return it sanitized.

    Used by ``_build_tool_result_message`` on its copy of the metadata so the
    base64 never reaches the serialised payload (module docstring: "Why the key
    is popped").
    """
    if not isinstance(metadata, dict):
        return []
    return sanitize_attached_images(metadata.pop(ATTACH_IMAGES_KEY, None))


def attached_images_note(tool_name: str, images: list[dict[str, str]]) -> str:
    """The text that travels beside the pixels.

    The model is told both that the image is *attached* (so it looks, rather
    than reasoning from OCR text as if that were all there is) and where the
    text form lives (so a later turn that has elided the image can still be
    answered from the tool result it read).

    The first line carries :data:`ATTACHMENT_NOTE_TAG` and is therefore the only
    line ``image_context_policy`` rewrites when it elides the pixels. Captions
    ride on the lines after it and stay true after an elision: they say which
    image is which, not that it is still visible.
    """
    n = len(images)
    lines = [
        f"{ATTACHMENT_NOTE_TAG} {n} image{'s' if n != 1 else ''} attached by tool '{tool_name}' — "
        "you can see the pixels directly in this message; the tool result above carries the text form."
    ]
    for i, img in enumerate(images, 1):
        caption = str(img.get(_CAPTION_KEY) or "").strip()
        if caption:
            lines.append(f"  {i}. {caption}")
    return "\n".join(lines)


def supersede_attachment_note(content: str, note: str) -> str:
    """*content* with this module's attachment note replaced by *note*.

    Owned by the producer of the text, not by the policy that needs to rewrite
    it: the same module defines the format and its parser, so the two cannot
    drift.

    Only the FIRST line is considered, and only when it starts with
    :data:`ATTACHMENT_NOTE_TAG` — the single line ``attached_images_note``
    writes and nothing else does. A message that was not built here (a user's
    own text, another tool's output) is not rewritten: the note is APPENDED, so
    an elision can never delete content the policy did not author. Anything
    after the tagged line — captions, a later annotation — survives verbatim.
    """
    text = content or ""
    lines = text.splitlines()
    if lines and lines[0].startswith(ATTACHMENT_NOTE_TAG):
        lines[0] = note
        return "\n".join(lines)
    return f"{text}\n{note}" if text else note


def unviewable_images_note(tool_name: str, count: int, model: str = "") -> str:
    """The text when a tool produced images this route cannot take.

    Silence would be the dangerous choice: the model asked for a screenshot,
    gets an OCR string, and has no way to know that the pixels were dropped
    rather than absent. It would then describe an image it never saw.
    """
    where = f"model '{model}'" if model else "this model"
    return (
        f"[{count} image{'s' if count != 1 else ''} produced by tool '{tool_name}' "
        f"{'were' if count != 1 else 'was'} NOT attached: {where} does not accept image input on this route. "
        "Work from the text form in the tool result above, and say what is unreadable rather than "
        "guessing at pixels you cannot see.]"
    )


def route_accepts_images(model: str, base_url: str | None = None) -> bool:
    """Whether an image part may go on the wire for *model* on this route.

    Delegates to ``model_registry.vision_capable`` — the declared capability
    vector, route-scoped, with the name-prefix table only as the fallback for
    ids the catalog does not list. Duplicating that decision here is what would
    eventually ship an image to a model that silently drops it (or 400s).

    Fail-CLOSED on any failure to resolve: an image the model cannot use costs a
    400 for the whole request, while a missing image costs nothing but the text
    form the tool already returned.
    """
    try:
        from ..model_registry import vision_capable

        return bool(vision_capable(model or "", base_url or ""))
    except Exception:  # pragma: no cover - defensive: never block a turn on the gate
        logger.debug("vision gate unavailable; not attaching tool images", exc_info=True)
        return False


def attachment_messages_for(
    tool_name: str,
    metadata: Any,
    model: str = "",
    base_url: str | None = None,
    provider: str = "",
) -> list[Any]:
    """Messages to append after a tool result so the model can see its images.

    Single entry point for the tool-result assembly path, so the gate, the note
    text and the message shape cannot drift apart. Returns ``[]`` when the tool
    declared nothing or the provider's tool-result turn cannot carry an image
    (:data:`NO_TOOL_IMAGE_PROVIDERS`), and exactly one synthetic ``role="user"``
    message otherwise — carrying the images, or the explanation of why it
    cannot.
    """
    images = read_attached_images(metadata)
    if not images:
        return []
    if str(provider or "").strip().lower() in NO_TOOL_IMAGE_PROVIDERS:
        logger.info(
            "tool %r produced %d image(s); provider %r cannot carry them on a tool-result turn "
            "— the text form in the tool result is what the model gets",
            tool_name,
            len(images),
            provider,
        )
        return []
    from ..client import LLMMessage

    if not route_accepts_images(model, base_url):
        return [
            LLMMessage(
                role="user",
                content=unviewable_images_note(tool_name, len(images), model),
            )
        ]
    return [
        LLMMessage(
            role="user",
            content=attached_images_note(tool_name, images),
            images=images,
        )
    ]
