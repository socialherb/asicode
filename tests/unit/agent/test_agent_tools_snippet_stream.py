"""``_read_local_snippet`` (delegate-to-helper local context) streaming tests.

The helper context asks for at most ~80 numbered lines; the previous
implementation read the whole file (``read_text`` + ``splitlines``) to answer
that. These tests pin the streaming contract: exact numbered window,
early-stop I/O (``stop_after_last``), whole-line retention cap,
splitlines-exact numbering on exotic separators, and out-of-range behavior.
"""

import io

from external_llm.agent.tool_handlers import agent_tools as at
from external_llm.agent.tool_handlers.read_tools import _stream_split_window


class _CountingReader:
    """BytesIO wrapper that records how many bytes ``read`` consumed."""

    def __init__(self, raw: bytes):
        self._buf = io.BytesIO(raw)
        self.bytes_read = 0

    def read(self, n: int = -1) -> bytes:
        data = self._buf.read(n)
        self.bytes_read += len(data)
        return data


def test_snippet_returns_exact_numbered_window(tmp_path):
    fp = tmp_path / "mod.py"
    fp.write_text("".join(f"line {i}\n" for i in range(1, 201)))
    out = at._read_local_snippet(fp, 90, 120)
    lines = out.splitlines()
    assert lines[0] == "90: line 90"
    assert lines[-1] == "120: line 120"
    assert len(lines) == 31
    numbered = [ln.split(": ", 1)[0] for ln in lines]
    assert numbered == [str(i) for i in range(90, 121)]


def test_snippet_out_of_range_returns_empty(tmp_path):
    fp = tmp_path / "short.py"
    fp.write_text("a\nb\nc\n")
    assert at._read_local_snippet(fp, 10, 20) == ""


def test_stream_split_window_stops_reading_after_last():
    raw = b"".join(b"line %d\n" % i for i in range(1, 100_001))
    fh = _CountingReader(raw)
    total, window = _stream_split_window(
        fh, b"", 1, 80, 1 << 20, stop_after_last=True
    )
    assert len(window) == 80
    # Chunk-granularity stop: the whole first 64 KiB chunk is counted before
    # the break fires, so total overshoots — the caller discards it.
    assert total >= 80
    # Window-only I/O: a few 64 KiB chunks, not the ~1 MB file.
    assert fh.bytes_read < len(raw) // 2

    fh2 = _CountingReader(raw)
    total2, _window2 = _stream_split_window(fh2, b"", 1, 80, 1 << 20)
    # Default (no flag): count to EOF, one full pass — contract unchanged.
    assert fh2.bytes_read == len(raw)
    assert total2 == 100_000


def test_snippet_retention_cap_drops_overflowing_whole_lines(tmp_path):
    fp = tmp_path / "wide.txt"
    fp.write_text("a" * 10_000 + "\n" + "b" * 10_000 + "\ntail\n")
    out = at._read_local_snippet(fp, 1, 80)
    assert "1: aaa" in out  # first 10 KB line fits under the 16 KiB budget
    assert "bbb" not in out  # second line would overflow -> dropped whole
    assert "tail" not in out


def test_snippet_numbering_matches_splitlines_on_exotic_separators(tmp_path):
    fp = tmp_path / "weird.txt"
    text = "a\x85b\x0bc\nc\rd\u2028e"
    fp.write_bytes(text.encode("utf-8"))
    expected = text.splitlines()
    out = at._read_local_snippet(fp, 1, 80)
    got = [ln.split(": ", 1)[1] for ln in out.splitlines()]
    assert got == expected
