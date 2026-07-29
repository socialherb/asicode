"""`read_file` must not materialise a file to answer questions about part of it.

Measured on a 108 MB log: +509 MB of peak RSS to produce a 116-character "too
long, use start_line" refusal — the bytes, the decoded string and a list of two
million line objects all live at once — and the same again for the ranged read
that refusal invites, so fixing only the refusal would have been half a fix.

The hard constraint is that the split stays ``str.splitlines()`` exactly.
read_file's line numbers are its own and anchor_edit's ``anchor_ast_lineno``
mode builds a matching array, so a streaming splitter that broke on ``\\n``
alone would silently renumber any file containing ``\\v \\f \\x1c \\x1d \\x1e
\\x85 \\u2028 \\u2029``. Hence PARITY, not plausibility: every case below is
answered twice — once through the bulk path, once through the streaming path —
and the two answers must be identical.
"""
from __future__ import annotations

import contextlib
import random
import sys
import tracemalloc

import pytest

import external_llm.agent.tool_handlers.read_tools as rt

# Every character str.splitlines() breaks on, plus multibyte and an ordinary
# space so lines are not all empty.
_BREAKS = ["\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]
_ALPHA = ["a", "b", " ", "\t", "가", "🎉"]


@contextlib.contextmanager
def _peak_alloc_mb(out: list):
    """Peak Python allocation during the block, in MB.

    tracemalloc rather than ``ru_maxrss``: the latter is a process-wide HIGH
    WATER MARK that never comes down, so a second measurement in the same
    interpreter reads ~0 growth no matter what it allocates. That is not
    hypothetical — it silently passed the ranged-read assertion below while the
    streaming path was disabled, because the test before it had already pushed
    the peak to 100 MB. Every allocation this measures (the bytes, the decoded
    str, the list of line objects) is a Python one, so tracemalloc sees all of
    it and sees only it.
    """
    tracemalloc.start()
    try:
        yield
        out.append(tracemalloc.get_traced_memory()[1] / (1024 * 1024))
    finally:
        tracemalloc.stop()


def _tricky_text(rng: random.Random, tokens: int) -> str:
    return "".join(
        rng.choice(_BREAKS) if rng.random() < 0.35 else rng.choice(_ALPHA)
        for _ in range(tokens)
    )


class TestSplitterParity:
    """The splitter itself, against the reference it has to reproduce."""

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64, 4096])
    def test_it_reproduces_splitlines_at_every_chunk_size(self, chunk, monkeypatch):
        import io

        monkeypatch.setattr(rt, "_STREAM_CHUNK", chunk)
        rng = random.Random(20260730)
        for _ in range(200):
            raw = _tricky_text(rng, 60).encode("utf-8")
            ref = raw.decode("utf-8", errors="replace").splitlines()
            total, window = rt._stream_split_window(io.BytesIO(raw), b"", 1, 1 << 40, 1 << 40)
            assert total == len(ref), f"line count drifted on {raw!r} at chunk={chunk}"
            assert window == ref, f"split drifted on {raw!r} at chunk={chunk}"

    @pytest.mark.parametrize(
        "raw",
        [b"\xff\xfe hi\n", b"a\xc3\n", b"\xe2\x80", b"x\xed\xa0\x80y\n"],
        ids=["bad-lead", "truncated-2byte", "truncated-3byte", "surrogate"],
    )
    @pytest.mark.parametrize("chunk", [1, 2, 3, 64])
    def test_invalid_utf8_degrades_identically(self, raw, chunk, monkeypatch):
        import io

        monkeypatch.setattr(rt, "_STREAM_CHUNK", chunk)
        ref = raw.decode("utf-8", errors="replace").splitlines()
        _total, window = rt._stream_split_window(io.BytesIO(raw), b"", 1, 1 << 40, 1 << 40)
        assert window == ref

    def test_a_crlf_split_across_reads_is_one_break(self, monkeypatch):
        """The only ambiguous boundary: `\\r` is a break until an `\\n` follows.

        Called out on its own because it is the case a plausible implementation
        gets wrong while every other test still passes.
        """
        import io

        monkeypatch.setattr(rt, "_STREAM_CHUNK", 1)
        raw = b"a\r\nb\r\nc"
        total, window = rt._stream_split_window(io.BytesIO(raw), b"", 1, 1 << 40, 1 << 40)
        assert window == ["a", "b", "c"]
        assert total == 3

    def test_the_count_survives_the_retention_cap(self):
        import io

        raw = ("x" * 40 + "\n").encode() * 5000
        total, window = rt._stream_split_window(io.BytesIO(raw), b"", 1, 1 << 40, 500)
        assert total == 5000, "counting must outlive retention"
        assert sum(map(len, window)) <= 500


@pytest.fixture
def streaming(monkeypatch):
    """Force every read through the streaming path."""
    monkeypatch.setattr(rt, "_STREAM_ABOVE_BYTES", 0)


def _write(root, name, text):
    with open(f"{root}/{name}", "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return name


class TestToolParity:
    """The whole tool, both paths, same answer."""

    @staticmethod
    def _both(reg, monkeypatch, args):
        # The tool result cache would serve the second call from the first,
        # so both dispatches would report the SAME path and the comparison
        # would prove nothing (it did: every parity assertion passed with a
        # `cache_hit: True` on the second result).
        reg._tool_result_cache = None
        monkeypatch.setattr(rt, "_STREAM_ABOVE_BYTES", 1 << 30)
        bulk = reg.dispatch("read_file", args)
        monkeypatch.setattr(rt, "_STREAM_ABOVE_BYTES", 0)
        streamed = reg.dispatch("read_file", args)
        return bulk, streamed

    @pytest.mark.parametrize(
        "args",
        [
            {},
            {"start_line": 3},
            {"start_line": 3, "end_line": 9},
            {"start_line": 1, "end_line": 1},
            {"end_line": 4},
            {"start_line": 900},              # past the end
            {"start_line": 9, "end_line": 2},  # inverted
            {"start_line": 1, "end_line": 0},  # malformed
        ],
        ids=lambda a: "-".join(f"{k}{v}" for k, v in a.items()) or "no-range",
    )
    def test_every_range_shape_agrees(self, tool_registry, temp_repo_root, monkeypatch, args):
        name = _write(
            temp_repo_root, "sample.py",
            "".join(f"def f{i}():\n    return {i}\n" for i in range(20)),
        )
        bulk, streamed = self._both(tool_registry, monkeypatch, {"path": name, **args})
        assert (bulk.ok, bulk.content, bulk.error) == (
            streamed.ok, streamed.content, streamed.error
        )
        assert bulk.metadata == streamed.metadata

    def test_a_file_of_exotic_line_breaks_agrees(self, tool_registry, temp_repo_root, monkeypatch):
        rng = random.Random(7)
        name = _write(temp_repo_root, "exotic.txt", _tricky_text(rng, 4000))
        for args in ({}, {"start_line": 5, "end_line": 40}, {"start_line": 2}):
            bulk, streamed = self._both(tool_registry, monkeypatch, {"path": name, **args})
            assert bulk.content == streamed.content, f"paths disagree for {args}"
            assert bulk.metadata == streamed.metadata

    def test_the_over_cap_refusal_agrees(self, tool_registry, temp_repo_root, monkeypatch):
        name = _write(
            temp_repo_root, "long.py",
            "".join(f"x = {i}\n" for i in range(rt._cfg.lines.READ_FILE_FULL_LINES + 50)),
        )
        bulk, streamed = self._both(tool_registry, monkeypatch, {"path": name})
        assert (bulk.metadata or {}).get("over_line_cap") is True
        assert bulk.content == streamed.content
        assert bulk.metadata == streamed.metadata

    def test_the_char_budget_cut_agrees(self, tool_registry, temp_repo_root, monkeypatch):
        """A range far past the output budget must truncate at the same line."""
        name = _write(
            temp_repo_root, "wide.txt",
            "".join(("y" * 200 + "\n") for _ in range(4000)),
        )
        bulk, streamed = self._both(
            tool_registry, monkeypatch, {"path": name, "start_line": 1, "end_line": 4000},
        )
        assert (bulk.metadata or {}).get("truncated") is True
        assert bulk.metadata == streamed.metadata, "resume_line drifted between paths"
        assert bulk.content == streamed.content

    def test_a_binary_file_is_still_refused_before_any_body_read(
        self, tool_registry, temp_repo_root, streaming,
    ):
        with open(f"{temp_repo_root}/blob.bin", "wb") as fh:
            fh.write(b"\x00\x01\x02" * 5000)
        result = tool_registry.dispatch("read_file", {"path": "blob.bin"})
        assert (result.metadata or {}).get("binary") is True


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="needs rusage")
class TestMemory:
    @staticmethod
    def _big(root):
        with open(f"{root}/big.log", "w", encoding="utf-8") as fh:
            fh.write("2026-07-30 INFO a log line with some text in it\n" * 700_000)
        return "big.log"

    def test_the_over_cap_refusal_does_not_materialise_the_file(
        self, tool_registry, temp_repo_root,
    ):
        name = self._big(temp_repo_root)
        peak: list = []
        with _peak_alloc_mb(peak):
            result = tool_registry.dispatch("read_file", {"path": name})
        assert (result.metadata or {}).get("over_line_cap") is True
        assert peak[0] < 20, (
            f"read_file allocated {peak[0]:.0f} MB to refuse a file — the whole "
            "file is being decoded and split just to count its lines"
        )

    def test_a_ranged_read_does_not_materialise_the_file(
        self, tool_registry, temp_repo_root,
    ):
        """The call the refusal tells the model to make next.

        Fixing only the refusal would have been half a fix: the model is told to
        come back with start_line, and that call paid the same 509 MB.
        """
        name = self._big(temp_repo_root)
        peak: list = []
        with _peak_alloc_mb(peak):
            result = tool_registry.dispatch(
                "read_file", {"path": name, "start_line": 100, "end_line": 120},
            )
        assert result.ok
        assert "2026-07-30 INFO" in result.content
        assert peak[0] < 20, f"a 21-line read allocated {peak[0]:.0f} MB"
