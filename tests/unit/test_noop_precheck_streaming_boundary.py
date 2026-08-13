"""P21-2: streaming needle search survives chunk-boundary multi-byte chars.

The no-op precheck streams the target in 64 KiB chunks. A per-chunk decode
turns a Hangul char straddling the chunk edge into U+FFFD twice, so a literal
needle that IS in the file was missed and the LLM asked to re-add it. The
incremental decoder keeps the char whole across chunks.
"""
from __future__ import annotations

from external_llm.service import ExternalLLMService


def test_needle_straddling_chunk_boundary_found(tmp_path):
    needle = "가나다라마바"  # 6 x 3 bytes — >= 6 chars so extraction keeps it
    # The 64 KiB boundary (byte 65536) falls INSIDE the needle: bytes
    # bytes 65534-65535 hold the first 2 bytes of 가 (chunk 1) and byte
    # 65536 its last byte (chunk 2) — a per-chunk decode breaks both halves.
    prefix = b"x" * 65534  # first 2 bytes of 가 land in chunk 1, the 3rd in chunk 2
    p = tmp_path / "b.txt"
    p.write_bytes(prefix + needle.encode() + b"\n")
    assert ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "b.txt", f'add "{needle}" to the file'
    )


def test_needle_fully_within_chunk_found(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"x" * 65500 + "가나다라마바".encode() + b"\n")
    assert ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "a.txt", 'add "가나다라마바" to the file'
    )


def test_needle_absent_returns_false(tmp_path):
    p = tmp_path / "c.txt"
    p.write_bytes(b"x" * 70000 + b"\n")
    assert not ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "c.txt", 'add "가나다라마바" to the file'
    )
