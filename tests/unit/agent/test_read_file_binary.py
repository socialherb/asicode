"""Tests for read_file's binary / non-UTF-8 detection.

``read_file`` decoded every path with ``errors="replace"``, so a PNG or a
``.pyc`` came back as thousands of U+FFFD characters. That is worse than an
error: it reads as content, so the model reasons about it and the write tools
accept edits against the garbled view.

Pinned here:
  1. a binary file returns a refusal naming the file, not its bytes,
  2. the refusal routes to the right alternative (read_image for images,
     bash otherwise),
  3. BOM-declared UTF-16/32 is reported as an *encoding* problem with an
     iconv recipe — not lumped in with binary, whose advice would be wrong,
  4. the false-positive guards: a UTF-8 BOM, and high bytes from legacy
     cp949/latin-1 sources, are still text and still return content,
  5. ordinary text paths (including the empty file) are untouched,
  6. ``metadata`` is always a dict — a handler returning None crashes the
     result cache in ToolRegistry._dispatch_impl.
"""

from __future__ import annotations

import codecs

from external_llm.agent.tool_handlers.read_tools import (
    _BINARY_SNIFF_BYTES,
    _classify_binary,
)
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

# Real headers: PNG carries a NUL in its IHDR length, CPython bytecode does
# not until its header is past — so the two exercise different rules.
PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06"
ELF_HEADER = b"\x7fELF\x02\x01\x01\x00" + bytes(range(0x10)) * 4


def _reg(tmp_path):
    return ToolRegistry(str(tmp_path), AgentConfig())


def _read(tmp_path, name: str):
    return _reg(tmp_path).dispatch("read_file", {"path": name})


class TestBinaryRefusal:
    def test_png_returns_notice_not_replacement_chars(self, tmp_path):
        (tmp_path / "logo.png").write_bytes(PNG_HEADER + b"\x00\x01\x02" * 500)
        res = _read(tmp_path, "logo.png")
        assert res.ok
        assert res.metadata["binary"] is True
        assert "�" not in res.content
        assert "binary file" in res.content

    def test_image_notice_points_at_read_image(self, tmp_path):
        (tmp_path / "shot.png").write_bytes(PNG_HEADER)
        assert "read_image" in _read(tmp_path, "shot.png").content

    def test_non_image_notice_points_at_bash(self, tmp_path):
        (tmp_path / "a.out").write_bytes(ELF_HEADER)
        content = _read(tmp_path, "a.out").content
        assert "read_image" not in content
        assert "bash" in content

    def test_byte_size_is_reported(self, tmp_path):
        payload = PNG_HEADER + b"\x00" * 1234
        (tmp_path / "b.png").write_bytes(payload)
        res = _read(tmp_path, "b.png")
        assert res.metadata["byte_size"] == len(payload)
        assert f"{len(payload):,}" in res.content

    def test_control_byte_ratio_catches_nul_free_binary(self, tmp_path):
        """A header with no NUL is still caught by the control-byte share."""
        (tmp_path / "c.bin").write_bytes(bytes(range(0x01, 0x20)) * 400)
        assert _read(tmp_path, "c.bin").metadata.get("binary") is True


class TestEncodingRefusal:
    def test_utf16_le_reports_encoding_and_iconv(self, tmp_path):
        (tmp_path / "win.txt").write_bytes(codecs.BOM_UTF16_LE + "x = 1\n".encode("utf-16-le"))
        res = _read(tmp_path, "win.txt")
        assert res.ok
        assert res.metadata["reason"] == "UTF-16 LE"
        assert "UTF-16 LE" in res.content
        assert "iconv -f utf-16le" in res.content

    def test_utf32_bom_is_not_read_as_utf16(self, tmp_path):
        """UTF-32 LE starts with the UTF-16 LE BOM — order of the checks matters."""
        (tmp_path / "u32.txt").write_bytes(codecs.BOM_UTF32_LE + "a".encode("utf-32-le"))
        assert _read(tmp_path, "u32.txt").metadata["reason"] == "UTF-32 LE"

    def test_encoding_notice_does_not_claim_binary(self, tmp_path):
        """The binary advice (read_image / xxd) is wrong for mis-encoded text."""
        (tmp_path / "e.txt").write_bytes(codecs.BOM_UTF16_BE + "hi".encode("utf-16-be"))
        content = _read(tmp_path, "e.txt").content
        assert "binary file" not in content
        assert "read_image" not in content


class TestTextIsUnaffected:
    def test_plain_source_still_returns_content(self, tmp_path):
        (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        res = _read(tmp_path, "m.py")
        assert res.ok
        assert "def f():" in res.content
        assert not res.metadata.get("binary")

    def test_utf8_bom_is_text(self, tmp_path):
        (tmp_path / "bom.py").write_bytes(codecs.BOM_UTF8 + b"x = 1\n")
        res = _read(tmp_path, "bom.py")
        assert not res.metadata.get("binary")
        assert "x = 1" in res.content

    def test_legacy_high_bytes_are_text(self, tmp_path):
        """cp949/latin-1 sources decode to U+FFFD but are NOT binary."""
        (tmp_path / "k.py").write_bytes("# 한글 주석\nx = 1\n".encode("cp949"))
        res = _read(tmp_path, "k.py")
        assert not res.metadata.get("binary")
        assert "x = 1" in res.content

    def test_empty_file_is_not_binary(self, tmp_path):
        (tmp_path / "empty.py").write_text("", encoding="utf-8")
        res = _read(tmp_path, "empty.py")
        assert res.ok
        assert not res.metadata.get("binary")

    def test_metadata_is_always_a_dict(self, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "t.png").write_bytes(PNG_HEADER)
        assert isinstance(_read(tmp_path, "t.py").metadata, dict)
        assert isinstance(_read(tmp_path, "t.png").metadata, dict)


class TestClassifier:
    def test_text_prefix_returns_none(self):
        assert _classify_binary(b"def f():\n    return 1\n") is None

    def test_empty_returns_none(self):
        assert _classify_binary(b"") is None

    def test_tabs_crlf_and_ansi_escapes_are_text(self):
        """Captured terminal output is text: ESC, CR, TAB, FF must not trip it."""
        assert _classify_binary(b"\x1b[31mred\x1b[0m\r\n\tindented\x0c\n") is None

    def test_nul_beyond_the_sniff_window_is_missed_by_design(self):
        """Documents the heuristic's edge: git's rule reads a prefix, not the file.

        A file whose first 8 KiB is clean reads as text. Widening the window
        trades a hot-path read against a case that does not occur in source
        trees, so the limit is pinned rather than fixed.
        """
        assert _classify_binary(b"a" * _BINARY_SNIFF_BYTES) is None
