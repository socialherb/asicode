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

    def test_a_break_free_file_is_not_quadratic(self):
        """The carry must not be re-split once per chunk.

        Prepending the accumulated carry to each chunk and re-splitting the
        result is O(n^2) in the length of ONE line, and a file with almost no
        newlines is a real input (minified bundle, .map, one-line JSON). The
        `\\r`-boundary probe re-scanned the same carry a second time, so the
        cost came twice per chunk: measured 9.81 s inside 1,042 splitlines()
        calls on a 34 MB one-liner, ~1.1 GB of transient strings.

        Time against a calibration workload — not allocation, not an absolute
        threshold, and not a self-ratio. All three were tried and two of them
        cannot see this bug at all:

        * Allocation is blind to it. Each iteration's temporary dies before the
          next is built, so peak live memory is ~3x the file either way:
          tracemalloc peak measured 8.5 MB quadratic against 8.4 MB linear. An
          assertion on it is green on both and tests nothing.
        * A self-ratio (time at 4x the input / time at 1x) is too noisy. The
          small run is ~10 ms, inside the scheduler noise floor, and the linear
          build measured 12.1x growth against the quadratic build's 18.0x.

        So the budget is expressed in units of the machine's own speed: one
        decode plus one splitlines over the same bytes is the theoretical
        minimum this function can do. Measured at 16 MB: 3.4x that minimum with
        the fix, 402x with the carry re-split per chunk (0.13 s against 16.54 s)
        — a 118x separation, so 25x sits far from both edges.
        """
        import io
        import time

        raw = b"var a=1;" * (8 * 131_072)  # 8 MB, ZERO line breaks
        expected = raw.decode()

        def _best(fn) -> float:
            best = float("inf")
            for _ in range(3):
                started = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - started)
            return best

        # The clock must not include building the reader — a BytesIO(raw) is a
        # full copy of the input and swamped the measurement when it was inside.
        readers = [io.BytesIO(raw) for _ in range(3)]
        results: list = []

        def _run() -> None:
            results.append(rt._stream_split_window(readers.pop(), b"", 1, 1 << 40, 1 << 40))

        calibration = _best(lambda: raw.decode("utf-8", errors="replace").splitlines())
        measured = _best(_run)

        assert results[0] == (1, [expected]), "the split itself must still be right"
        ratio = measured / calibration if calibration else float("inf")
        assert ratio < 25.0, (
            f"the splitter cost {ratio:.0f}x one decode+split of the same bytes "
            f"({measured:.3f}s against {calibration:.3f}s) — the carry is being "
            "re-split on every chunk"
        )

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

    def test_a_line_wider_than_the_retention_window_agrees(
        self, tool_registry, temp_repo_root, monkeypatch,
    ):
        """One line wider than the streaming window — a minified bundle.

        The sibling below uses 200-char lines, so both paths stayed inside the
        window and the disagreement never showed: the bulk path cut the line to
        the budget and reported `partial_line` (_apply_char_budget's documented
        "emit a prefix and advance PAST it" branch), while the streaming path
        dropped it before the budget ever saw it and returned an EMPTY code
        block — 60,355 chars and full metadata against 91 chars and {}.
        """
        wide = "var a=1;" * 40_000  # 320k chars, well past _retain
        name = _write(temp_repo_root, "bundle.min.js", wide + "\n")
        bulk, streamed = self._both(tool_registry, monkeypatch, {"path": name})
        assert (bulk.metadata or {}).get("partial_line") == 1, "fixture no longer over-wide"
        assert bulk.content == streamed.content
        assert bulk.metadata == streamed.metadata
        assert "var a=1;" in streamed.content, "the streamed body must not be empty"

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
    def _big(root, ext="log"):
        name = f"big.{ext}"
        with open(f"{root}/{name}", "w", encoding="utf-8") as fh:
            fh.write("2026-07-30 INFO a log line with some text in it\n" * 700_000)
        return name

    # The extension is the whole point of the parametrisation, not incidental
    # coverage. This assertion was green for a year against `big.log` alone
    # while the branch it guards allocated 1.65 GB, because `.log` matches no
    # LanguageId: the over-cap refusal decorates itself with a file outline,
    # and for `.py`/`.js` that outline read and parsed the entire file
    # (tree_sitter.Parser.parse alone took 8.67 s on 32 MB). The fixture, not
    # the code, was what kept the number small.
    @pytest.mark.parametrize("ext", ["log", "py", "js"])
    def test_the_over_cap_refusal_does_not_materialise_the_file(
        self, tool_registry, temp_repo_root, ext,
    ):
        name = self._big(temp_repo_root, ext)
        peak: list = []
        with _peak_alloc_mb(peak):
            result = tool_registry.dispatch("read_file", {"path": name})
        assert (result.metadata or {}).get("over_line_cap") is True
        assert peak[0] < 20, (
            f"read_file allocated {peak[0]:.0f} MB to refuse a .{ext} file — the "
            "whole file is being decoded and split just to count its lines, or "
            "the outline that decorates the refusal is parsing all of it"
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
