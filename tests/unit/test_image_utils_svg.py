"""Tests for SVG / text-attachment handling in image_utils.

Locks the P0-2 fix: SVG is XML text, not a raster image, so it must NOT be
base64-encoded as ``image/svg+xml`` (vision APIs reject it and PIL cannot decode
it → a useless "[Image Attached]" placeholder). Instead it is inlined as UTF-8
text, while raster images, data-URLs, and the quoted-path pass keep working.
"""
from __future__ import annotations

import base64

from external_llm.image_utils import (
    _IMAGE_EXTENSIONS,
    _TEXT_INLINE_EXTENSIONS,
    _TEXT_INLINE_MAX_BYTES,
    _classify_attachment,
    _extract_images_from_input,
    _read_text_inline,
)

# ── regression guards on the extension sets ──────────────────────────────────

def test_svg_not_in_raster_set():
    """SVG must stay out of the raster set and in the text-inline set."""
    assert ".svg" not in _IMAGE_EXTENSIONS
    assert ".svg" in _TEXT_INLINE_EXTENSIONS
    # raster core still present
    assert ".png" in _IMAGE_EXTENSIONS and ".jpg" in _IMAGE_EXTENSIONS


# ── _read_text_inline unit tests ──────────────────────────────────────────────

def test_read_text_inline_missing_returns_none(tmp_path):
    assert _read_text_inline(tmp_path / "nope.svg") is None


def test_read_text_inline_non_utf8_returns_none(tmp_path):
    p = tmp_path / "bad.svg"
    p.write_bytes(b"\xff\xfe\x00\x01")  # not valid UTF-8
    assert _read_text_inline(p) is None


def test_read_text_inline_too_large_returns_none(tmp_path):
    p = tmp_path / "big.svg"
    p.write_text("<svg>" + "x" * (_TEXT_INLINE_MAX_BYTES + 10) + "</svg>", encoding="utf-8")
    assert _read_text_inline(p) is None


def test_read_text_inline_oversized_skips_read(tmp_path, monkeypatch):
    """P24-3: stat-first size gate (P22-4 policy).

    The inline-text reader used to read_bytes() the whole file and only then
    compare against the cap — a multi-GB text attachment was pulled fully
    into memory just to be rejected. Mirror the raster branch of
    _classify_attachment: stat() first, never open an oversized file.
    """
    import pathlib

    p = tmp_path / "huge.svg"
    with p.open("wb") as fh:
        fh.truncate(_TEXT_INLINE_MAX_BYTES + 1024)  # sparse file: no disk cost

    def _boom(*_a, **_k):
        raise AssertionError("read_bytes must not be called for oversized files")

    monkeypatch.setattr(pathlib.Path, "read_bytes", _boom)
    assert _read_text_inline(p) is None


def test_read_text_inline_small_returns_labelled_block(tmp_path):
    p = tmp_path / "icon.svg"
    body = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    p.write_text(body, encoding="utf-8")
    block = _read_text_inline(p)
    assert block is not None
    assert "icon.svg" in block
    assert str(len(body.encode("utf-8"))) in block
    assert body in block


# ── _classify_attachment unit tests ───────────────────────────────────────────

def test_classify_unknown_extension_is_none(tmp_path):
    p = tmp_path / "code.py"
    p.write_text("x = 1", encoding="utf-8")
    assert _classify_attachment(p) is None


def test_classify_svg_returns_text_kind(tmp_path):
    p = tmp_path / "icon.svg"
    p.write_text("<svg><rect/></svg>", encoding="utf-8")
    kind = _classify_attachment(p)
    assert kind is not None and kind[0] == "text"


def test_classify_raster_returns_image_kind(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    kind = _classify_attachment(p)
    assert kind is not None and kind[0] == "image"
    assert kind[1]["media_type"] == "image/png"


# ── _extract_images_from_input behaviour (the real boundary) ─────────────────

def test_svg_not_in_images_but_inlined(tmp_path):
    """The core fix: an SVG path is NOT returned as an image; its text is inlined."""
    svg = tmp_path / "diagram.svg"
    body = '<svg xmlns="http://www.w3.org/2000/svg"><text>hello</text></svg>'
    svg.write_text(body, encoding="utf-8")
    text, images = _extract_images_from_input(str(svg))
    assert images == []
    assert "diagram.svg" in text
    assert body in text  # the SVG markup is readable by the model


def test_raster_png_still_extracted_as_image(tmp_path):
    """Regression: a real raster file is still base64-encoded into images."""
    png = tmp_path / "shot.png"
    raw = b"\x89PNG\r\n\x1a\n" + b"pixel" * 8
    png.write_bytes(raw)
    text, images = _extract_images_from_input(str(png))
    assert len(images) == 1
    assert images[0]["media_type"] == "image/png"
    assert images[0]["data"] == base64.b64encode(raw).decode("ascii")
    assert str(png) not in text  # path removed from prompt text


def test_data_url_png_still_extracted():
    """Regression: pass-3 data-URL extraction is untouched by the SVG change."""
    b64 = "iVBORw0KGgoAAAANS="
    text, images = _extract_images_from_input(f"see data:image/png;base64,{b64} here")
    assert len(images) == 1
    assert images[0]["media_type"] == "image/png"
    assert images[0]["data"] == b64
    assert "data:image" not in text


def test_large_svg_left_as_path_for_read_file(tmp_path):
    """An SVG over the inline cap is left in text (read_file streams large files)."""
    svg = tmp_path / "huge.svg"
    svg.write_text("<svg>" + "x" * (_TEXT_INLINE_MAX_BYTES + 100) + "</svg>", encoding="utf-8")
    text, images = _extract_images_from_input(str(svg))
    assert images == []
    assert str(svg) in text  # path preserved so read_file can reach it


def test_nonexistent_svg_kept_as_path():
    """A missing .svg file is not swallowed; the path stays for the agent."""
    text, images = _extract_images_from_input("no_such_file.svg")
    assert images == []
    assert "no_such_file.svg" in text


def test_quoted_svg_path_with_space_inlined(tmp_path):
    """2nd-pass quoted-path handling also inlines SVG (path with a space)."""
    sub = tmp_path / "sub dir"
    sub.mkdir()
    svg = sub / "icon.svg"
    svg.write_text('<svg><rect width="10"/></svg>', encoding="utf-8")
    text, images = _extract_images_from_input(f'"{svg}"')
    assert images == []
    assert "icon.svg" in text
    assert "<rect" in text


def test_mixed_png_and_svg_in_one_prompt(tmp_path):
    """A raster image and an SVG in the same input split correctly per kind."""
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNGfake")
    svg = tmp_path / "b.svg"
    svg.write_text("<svg><rect/></svg>", encoding="utf-8")
    text, images = _extract_images_from_input(f"{png} {svg}")
    assert len(images) == 1
    assert images[0]["media_type"] == "image/png"
    assert str(png) not in text  # raster path removed
    assert "b.svg" in text and "<rect" in text  # svg inlined


# ── P21-2: raster attachment size cap ────────────────────────────────────────

def test_raster_over_cap_kept_as_path(tmp_path):
    """An image larger than _IMAGE_MAX_BYTES must not be base64-embedded."""
    from external_llm.image_utils import _IMAGE_MAX_BYTES
    p = tmp_path / "big.png"
    with open(p, "wb") as f:
        f.truncate(_IMAGE_MAX_BYTES + 1)  # sparse — stat-based guard
    assert _classify_attachment(p) is None


def test_raster_at_cap_embedded(tmp_path):
    """Exactly _IMAGE_MAX_BYTES is still embedded (<= policy)."""
    from external_llm.image_utils import _IMAGE_MAX_BYTES
    p = tmp_path / "cap.png"
    with open(p, "wb") as f:
        f.truncate(_IMAGE_MAX_BYTES)
    kind = _classify_attachment(p)
    assert kind is not None and kind[0] == "image"
    assert len(kind[1]["data"]) > 0


def test_raster_small_embedded(tmp_path):
    p = tmp_path / "tiny.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    kind = _classify_attachment(p)
    assert kind is not None and kind[0] == "image"
    assert kind[1]["media_type"] == "image/png"


def test_text_inline_cap_unchanged(tmp_path):
    """P21-2 must not disturb the existing 64 KiB SVG inline cap."""
    p = tmp_path / "big.svg"
    with open(p, "wb") as f:
        f.truncate(_TEXT_INLINE_MAX_BYTES + 1)
    assert _classify_attachment(p) is None
    small = tmp_path / "small.svg"
    small.write_text("<svg/>")
    kind = _classify_attachment(small)
    assert kind is not None and kind[0] == "text"
