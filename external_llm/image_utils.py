"""
Image utility function module.

Functions extracted from asi.py:
- _check_clipboard_image: Detect image data on macOS clipboard and return as base64
- _extract_images_from_input: Detect image file paths in user input, read as base64, and return text with paths removed
"""

from __future__ import annotations

import logging
from contextlib import suppress

logger = logging.getLogger(__name__)


# Clipboard image UTIs (Uniform Type Identifiers) probed in preference order.
# macOS pasteboards commonly offer several flavours of a copied image:
#   * public.png  — screenshots (Cmd+Ctrl+Shift+4 → clipboard)
#   * public.tiff — the system's native pasteboard image flavour, used when an
#                   image is copied from a browser or application
# Probing only public.png (as before) silently dropped the TIFF-only case, so
# images copied from apps were never attached. public.file-url is intentionally
# not probed here — drag/drop and explicit @path cover file references.
_CLIPBOARD_IMAGE_UTIS = (
    ("public.png", "image/png"),
    ("public.tiff", "image/tiff"),
)


def _build_clipboard_probe_script(utis):
    """Build the JXA script that probes clipboard image flavours in order.

    The script emits ``"<uti>\\t<base64>"`` for the first UTI whose pasteboard
    data is non-nil, or an empty string when no flavour is present.
    """
    js_utis = ",".join(f"'{uti}'" for uti, _ in utis)
    return (
        "ObjC.import('AppKit');\n"
        "var pb = $.NSPasteboard.generalPasteboard;\n"
        "var items = pb.pasteboardItems;\n"
        "if (items.js.length === 0) { ''; } else {\n"
        "  var it = items.js[0];\n"
        "  var types = it.types.js.map(function(t){ return t.js; });\n"
        f"  var utis = [{js_utis}];\n"
        "  var found = '';\n"
        "  for (var i = 0; i < utis.length; i++) {\n"
        "    if (types.indexOf(utis[i]) >= 0) {\n"
        "      var data = it.dataForType(utis[i]);\n"
        "      if (data && !data.isNil()) {\n"
        "        found = utis[i] + '\\t' + data.base64EncodedStringWithOptions(0).js;\n"
        "        break;\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  found;\n"
        "}\n"
    )


def _parse_clipboard_probe_result(raw):
    """Parse probe stdout into ``(media_type, base64)`` or ``None``.

    Returns None when the output is empty, lacks the ``<uti>\\t`` prefix, or
    names an unknown UTI.
    """
    if not raw:
        return None
    raw = raw.strip()
    if "\t" not in raw:
        return None
    uti, _, b64 = raw.partition("\t")
    media_type = dict(_CLIPBOARD_IMAGE_UTIS).get(uti)
    if media_type is None or not b64:
        return None
    return media_type, b64


def _check_clipboard_image() -> list[dict[str, str]]:
    """Detect image data on macOS clipboard and return as base64.

    Reads the clipboard directly when using Cmd+Ctrl+Shift+4 (clipboard
    screenshot) and converts to base64 data when entering a prompt. Probes the
    PNG and TIFF pasteboard flavours, preferring PNG.
    """
    import base64 as _b64
    import platform as _platform
    import subprocess as _sp

    if _platform.system() != "Darwin":
        return []

    with suppress(FileNotFoundError, _sp.TimeoutExpired, OSError):
        # Probe clipboard image flavours via JXA (JavaScript for Automation).
        # NB: ``-l JavaScript`` runs JXA — AppleScript syntax silently fails to
        # parse here, so the ObjC bridge must use the JS form (ObjC.import, $.).
        _result = _sp.run(
            ["osascript", "-l", "JavaScript", "-e", _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if _result.returncode != 0:
            return []
        _parsed = _parse_clipboard_probe_result(_result.stdout)
        if _parsed is None:
            return []
        _media_type, _b64_data = _parsed
        # minimum plausible image size — guards against stub fragments
        if len(_b64_data) > 100:
            try:
                _b64.b64decode(_b64_data, validate=True)
            except Exception:
                logger.debug("clipboard b64 decode failed", exc_info=True)
                return []
            return [{"media_type": _media_type, "data": _b64_data}]

    return []


# Raster images PIL can decode and the vision APIs accept. SVG is deliberately
# excluded — it is neither raster nor OCR-able, so encoding it as a base64
# image silently failed on every path: vision APIs reject image/svg+xml, and the
# non-vision/OCR path hit PIL.UnidentifiedImageError → an empty, useless
# "[Image Attached]" placeholder. The webapp upload route already rejects
# image/svg+xml (test_svg_media_type_rejected_by_whitelist), so the CLI drag
# path was an inconsistency. SVG is XML text, so it is inlined instead.
_IMAGE_EXTENSIONS = frozenset(
    (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    )
)

# Text-format attachments that are *not* raster images. They are read as UTF-8
# and inlined into the prompt so the model can read them directly — the same
# outcome read_file produces (read_tools._EXT_LANG_MAP maps svg→xml).
_TEXT_INLINE_EXTENSIONS = frozenset((".svg",))

# Cap beyond which an SVG (etc.) is left as a path for read_file (which streams)
# rather than inlined, so a multi-MB generated SVG can't flood the context.
_TEXT_INLINE_MAX_BYTES = 64 * 1024

# P21-2: raster attachments are read and base64-embedded into the prompt —
# an unbounded image meant a multi-hundred-MB PNG got base64-expanded
# (~1.33x) straight into the LLM context. Same policy as the design-chat
# attachment cap (P15-2, 10 MiB per image); oversized files are left as a
# path so read_file (which streams) can still reach them.
_IMAGE_MAX_BYTES = 10 * 1024 * 1024


def _read_text_inline(path) -> str | None:
    """Return a labelled text block for a small text file, or None.

    Returns None when the file is missing/unreadable, non-UTF-8, or larger than
    ``_TEXT_INLINE_MAX_BYTES``; the caller then leaves the path intact so
    ``read_file`` — which streams and already maps svg→xml — can reach it.
    """
    # P24-3: stat-first (P22-4 policy). The old order read_bytes() the whole
    # file and only then compared against the cap, so a multi-GB text
    # attachment was pulled fully into memory just to be rejected — the
    # raster branch of _classify_attachment already gates on stat() first.
    try:
        size = path.stat().st_size
    except OSError:
        logger.debug("stat of %s failed", path, exc_info=True)
        return None
    if size > _TEXT_INLINE_MAX_BYTES:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        logger.debug("inline text read of %s failed", path, exc_info=True)
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("inline text read of %s is not UTF-8", path, exc_info=True)
        return None
    return f"[Attached {path.name} ({len(raw)} bytes):\n{text}\n]"


def _classify_attachment(path):
    """Classify and load a candidate attachment path.

    Centralises the extension→strategy decision so the whitespace-split pass and
    the quoted-path pass stay in sync. Returns:

    * ``("image", {"media_type": ..., "data": b64})`` for a raster image;
    * ``("text", inline_block)`` for a small text-inlineable file (SVG);
    * ``None`` for an unknown extension or an unreadable file, in which case the
      caller keeps the path in the prompt text.
    """
    import base64 as _b64
    import mimetypes as _mimetypes

    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        try:
            if path.stat().st_size > _IMAGE_MAX_BYTES:
                # P21-2: refuse before read_bytes — the file is left as a
                # path for read_file (which streams) instead of base64-bombing
                # the prompt.
                logger.debug(
                    "attachment image %s too large (%d bytes > %d), keeping path in prompt",
                    path,
                    path.stat().st_size,
                    _IMAGE_MAX_BYTES,
                )
                return None
            raw = path.read_bytes()
        except (OSError, PermissionError, MemoryError):
            logger.debug("attachment %s unreadable, keeping path in prompt", path, exc_info=True)
            return None
        b64 = _b64.b64encode(raw).decode("ascii")
        mt = _mimetypes.guess_type(str(path))[0] or "image/png"
        return ("image", {"media_type": mt, "data": b64})
    if suffix in _TEXT_INLINE_EXTENSIONS:
        block = _read_text_inline(path)
        return ("text", block) if block is not None else None
    return None


def _extract_images_from_input(text: str) -> tuple[str, list[dict[str, str]]]:
    """Detect image file paths in user input, read as base64, and return text with paths removed.

    Detects image files dragged into the terminal (e.g. "path/to/img.png"),
    base64-encodes them, and removes the paths from text.
    Also handles shell-escaped spaces (`\\ `) pasted from macOS/iTerm2,
    and quotation-wrapped paths ("/path/with spaces/file.png").
    Also detects `data:image/...;base64,...` data URLs.

    SVG (and other ``_TEXT_INLINE_EXTENSIONS``) is NOT treated as an image: it
    is read as UTF-8 and inlined into the returned ``text`` so the model can read
    it directly. A raster image whose bytes can't be read — and a text attachment
    that is too large / unreadable — is left as a path in the text for read_file.
    """
    import re as _re
    from pathlib import Path as _Path

    # --- 1st pass: shell-escaped blank/space (\ ) processing ---
    _esc_marker = "\x00_IMG_ESC_SP_\x00"
    _escaped = text.replace("\\ ", _esc_marker)
    words = _escaped.split()
    cleaned_words = []
    images: list[dict[str, str]] = []
    inline_blocks: list[str] = []

    for w in words:
        w_restored = w.replace(_esc_marker, " ")
        p = _Path(w_restored.strip("'\""))
        kind = _classify_attachment(p)
        if kind is None:
            cleaned_words.append(w_restored)
            continue
        if kind[0] == "image":
            images.append(kind[1])
        else:  # "text"
            inline_blocks.append(kind[1])

    result = " ".join(cleaned_words).strip()

    # --- 2nd pass: quotation-wrapped paths (missed by 1st pass) ---
    if not images and not inline_blocks:
        # If starts with a quote, extract up to the closing quote and verify file
        _q = None
        if result.startswith('"'):
            _q = '"'
        elif result.startswith("'"):
            _q = "'"
        if _q:
            _end = result.find(_q, 1)
            if _end > 1:
                _candidate = result[1:_end]
                p = _Path(_candidate)
                kind = _classify_attachment(p)
                if kind is not None:
                    if kind[0] == "image":
                        images.append(kind[1])
                    else:  # "text"
                        inline_blocks.append(kind[1])
                    result = result[_end + 1 :].strip()

    # --- 3rd pass: data URL detection (base64-encoded images) ---
    # "data:image/png;base64,iVBOR..." form processing
    _data_url_re = _re.compile(
        r"data:image/(?P<fmt>\w+);base64,(?P<b64>[A-Za-z0-9+/=]+)",
    )
    _new_result_parts = []
    _last_end = 0
    for _m in _data_url_re.finditer(result):
        _fmt = _m.group("fmt")
        _b64_data = _m.group("b64")
        _mt = f"image/{_fmt}"
        if _fmt == "png":
            _mt = "image/png"
        elif _fmt in {"jpg", "jpeg"}:
            _mt = "image/jpeg"
        images.append({"media_type": _mt, "data": _b64_data})
        _new_result_parts.append(result[_last_end : _m.start()])
        _last_end = _m.end()
    _new_result_parts.append(result[_last_end:])
    result = "".join(_new_result_parts).strip()

    # Append any inlined text attachments (SVG, …) after the cleaned prompt text.
    if inline_blocks:
        result = "\n".join([result, *inline_blocks]).strip()

    return result.strip(), images
