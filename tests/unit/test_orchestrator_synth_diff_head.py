"""P21-2: _read_synth_diff_head cuts on a UTF-8 boundary.

The synthetic-diff reviewer read 64 KiB and decoded with errors=replace, so a
multi-byte char straddling the window ended the review diff in U+FFFD (same
class as P20-5, last sibling). The trim drops the incomplete tail.
"""

from __future__ import annotations

import pytest

from external_llm.agent.orchestrator import _read_synth_diff_head


def test_cut_mid_hangul_no_replacement_char(tmp_path):
    # 120 KiB of 3-byte Hangul: the 64 KiB cut lands 1 byte into a char
    # (65536 % 3 == 1); the trim must drop the incomplete tail.
    p = tmp_path / "new.py"
    p.write_text("가" * 40_000, encoding="utf-8")
    text, truncated, size = _read_synth_diff_head(str(p))
    assert truncated is True
    assert size == 120_000
    assert text.endswith("가")
    assert "\ufffd" not in text


def test_small_file_full(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello\n")
    text, truncated, size = _read_synth_diff_head(str(p))
    assert (text, truncated, size) == ("hello\n", False, 6)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_synth_diff_head(str(tmp_path / "nope.txt"))
