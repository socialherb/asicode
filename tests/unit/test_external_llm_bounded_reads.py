"""P21-1: external_llm/service.py target-file snippet reads are bounded.

Bug: three snippet paths read the WHOLE file —
  * _read_target_file_snippet_best_effort ("snippet" = whole file, inserted
    into the prompt as "TARGET FILE SNIPPET (authoritative)" with no cap)
  * _read_target_file_focused_snippet_best_effort (radius window, but the
    full file was read and split before slicing)
  * _noop_precheck_for_literal_add (whole file read just to search)
A multi-hundred-MB target spiked memory per read and blew the prompt —
the same class P19-1 fixed on the webapp side.

Fix under test:
- snippet: head-bounded (default 1 MiB, overridable max_bytes), explicit
  TRUNCATED marker, boundary-safe UTF-8 cut (no U+FFFD at the end)
- focused: same bounded read; a needle past the head yields "" (callers
  fall back to the bounded head_tail snippet — never a full read)
- needle search: streaming — a needle at the very END of a >1 MiB file is
  still found (a head-bounded read would miss it and cause a spurious re-add)
"""
from __future__ import annotations

from external_llm.service import _SNIPPET_READ_MAX_BYTES, ExternalLLMService


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_snippet_small_file_full_content(tmp_path):
    _write(tmp_path, "a.txt", b"line1\nline2\n")
    out = ExternalLLMService._read_target_file_snippet_best_effort(
        None, str(tmp_path), "a.txt"
    )
    assert out == "line1\nline2\n"


def test_snippet_huge_file_bounded_with_marker(tmp_path):
    _write(tmp_path, "big.txt", b"0123456789abcdef\n" * 100_000)  # 1.7 MiB
    out = ExternalLLMService._read_target_file_snippet_best_effort(
        None, str(tmp_path), "big.txt"
    )
    assert "TRUNCATED" in out
    assert out.count("TRUNCATED") == 1
    assert len(out) < _SNIPPET_READ_MAX_BYTES + 200


def test_snippet_custom_max_bytes(tmp_path):
    _write(tmp_path, "a.txt", b"x" * 10_000)
    out = ExternalLLMService._read_target_file_snippet_best_effort(
        None, str(tmp_path), "a.txt", max_bytes=100
    )
    assert "TRUNCATED" in out
    assert len(out) < 500


def test_snippet_utf8_boundary_no_replacement_char(tmp_path):
    # 1 MiB cut lands inside a 3-byte Hangul char (1 MiB % 3 == 1) — the cut
    # must drop the incomplete tail, never emit U+FFFD at the boundary.
    _write(tmp_path, "ko.txt", ("가" * 400_000).encode())  # 1.2 MiB
    out = ExternalLLMService._read_target_file_snippet_best_effort(
        None, str(tmp_path), "ko.txt"
    )
    assert "TRUNCATED" in out
    head = out[: out.index("TRUNCATED")]
    assert "\ufffd" not in head


def test_snippet_missing_file(tmp_path):
    out = ExternalLLMService._read_target_file_snippet_best_effort(
        None, str(tmp_path), "nope.txt"
    )
    assert out == ""


def test_focused_needle_in_head(tmp_path):
    lines = [f"line{i}" for i in range(100)] + ["def target_fn():", "    pass"]
    _write(tmp_path, "f.py", ("\n".join(lines)).encode())
    out = ExternalLLMService._read_target_file_focused_snippet_best_effort(
        None, str(tmp_path), "f.py", needles=["target_fn"], radius_lines=5
    )
    assert "target_fn" in out
    assert "line95" in out  # window covers the needle neighbourhood


def test_focused_needle_past_head_returns_empty(tmp_path):
    # needle lives beyond the 1 MiB head -> "" (callers fall back to the
    # bounded head_tail snippet — never a full read)
    _write(tmp_path, "big.py", b"0123456789abcdef\n" * 70_000 + b"def far_needle():\n    pass\n")
    out = ExternalLLMService._read_target_file_focused_snippet_best_effort(
        None, str(tmp_path), "big.py", needles=["far_needle"]
    )
    assert out == ""


def test_needle_search_finds_tail_needle_streaming(tmp_path):
    # needle at the very END of a >1 MiB file must be found (streaming
    # search — a head-bounded read would miss it)
    big = b"0123456789abcdef\n" * 70_000 + b"\nTHE_LITERAL_TOKEN\n"
    _write(tmp_path, "big.txt", big)
    assert ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "big.txt", 'add "THE_LITERAL_TOKEN" to the file'
    )


def test_needle_search_no_match(tmp_path):
    _write(tmp_path, "a.txt", b"hello world\n")
    assert not ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "a.txt", 'add "zzz_not_there" to the file'
    )


# ── P22-2 / P22-4: containment + stat-based size gate ────────────────────────

def test_snippet_rejects_escape_path(tmp_path):
    """P22-2: repo-relative resolution must not escape the repo (../SECRET.txt)."""
    secret = tmp_path.parent / "SECRET.txt"
    secret.write_text("TOP_SECRET_TOKEN=abc123\n")
    out = ExternalLLMService._read_target_file_snippet_best_effort(
        None, str(tmp_path), "../SECRET.txt"
    )
    assert out == ""


def test_focused_snippet_rejects_escape_path(tmp_path):
    out = ExternalLLMService._read_target_file_focused_snippet_best_effort(
        None, str(tmp_path), "../SECRET.txt", needles=["TOP_SECRET"]
    )
    assert out == ""


def test_noop_precheck_rejects_escape_path(tmp_path):
    secret = tmp_path.parent / "SECRET.txt"
    secret.write_text("TOP_SECRET_TOKEN=abc123\n")
    out = ExternalLLMService._noop_precheck_for_literal_add(
        str(tmp_path), "../SECRET.txt", 'add "TOP_SECRET_TOKEN=abc123"'
    )
    assert out is False


def test_file_retry_gate_stat_based(tmp_path):
    """P22-4: the FILE-retry size gate must not load the whole file to compare
    its size; P22-2: it must also refuse escape paths."""
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 61_000)  # > _MAX_FILE_RETRY_FILE_CHARS (60_000)
    small = tmp_path / "small.txt"
    small.write_bytes(b"y" * 100)
    svc = object.__new__(ExternalLLMService)
    assert svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "big.txt") is False
    assert svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "small.txt") is True
    assert svc._is_target_file_small_enough_for_file_retry(str(tmp_path), "../SECRET.txt") is False
