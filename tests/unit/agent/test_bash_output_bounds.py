"""`bash` output must be bounded while it is read, not after.

The tool cuts its result to ``BASH_OUTPUT_MAX_CHARS`` (~130 KB), but
``communicate()`` had no way to hold less than everything, so the whole output
existed in memory first. Measured on ``cat`` of a 108 MB log:

    before   +360.6 MB peak RSS, 0.24 s  ->  ~130 KB of content
    after    +  0.4 MB peak RSS, 0.03 s  ->  the same content

This is the same defect the foreground search path was fixed for, on the tool
that can run anything — and ``BackgroundJobManager`` next door already solved it
for live job output with a tail cap.

Two things had to survive the bounding: the RENDERED result (head and tail are
both load-bearing — pytest's summary is at the end, the failing command at the
start) and the REPORTED size, which is no longer ``len(content)`` and would
otherwise describe a 108 MB run by the slice that was kept.
"""

from __future__ import annotations

import resource
import sys

import pytest

from external_llm.agent.tool_handlers.git_tools import (
    _BoundedCapture,
    _truncate_bash_output,
)


def _peak_mb() -> float:
    _raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return _raw / (1024 * 1024) if sys.platform == "darwin" else _raw / 1024


class TestBoundedCapture:
    def test_a_stream_under_the_cap_is_kept_verbatim(self):
        cap = _BoundedCapture(1000)
        for chunk in ("alpha\n", "beta\n", "gamma\n"):
            cap.feed(chunk)
        assert cap.text() == "alpha\nbeta\ngamma\n"
        assert cap.dropped == 0
        assert cap.total == len("alpha\nbeta\ngamma\n")

    @pytest.mark.parametrize("chunk", [7, 64, 4096], ids=lambda n: f"chunk{n}")
    def test_the_head_and_the_tail_both_survive(self, chunk):
        """However the caller happens to slice its reads."""
        body = "HEAD" + "x" * 5000 + "TAIL"
        cap = _BoundedCapture(10)
        for i in range(0, len(body), chunk):
            cap.feed(body[i : i + chunk])
        text = cap.text()
        assert text.startswith("HEAD")
        assert text.endswith("TAIL")
        assert cap.dropped > 4000, f"nothing was elided: {len(text)} chars kept"

    def test_the_true_size_is_counted_not_the_retained_size(self):
        cap = _BoundedCapture(100)
        for _ in range(1000):
            cap.feed("y" * 100)
        assert cap.total == 100_000
        assert len(cap.text()) < 1000
        assert cap.dropped == cap.total - (
            len(cap.text()) - len(f"\n... [{cap.dropped:,} chars dropped (middle)] ...\n")
        )

    def test_retention_covers_what_truncation_will_slice(self):
        """Each side keeps the FULL cap, so the rendered result is unchanged.

        ``_truncate_bash_output`` slices ``content[:cap//2]`` and
        ``content[-cap//2:]``. Retaining only ``cap//2`` per side would make
        those slices reach into the elision marker instead of into output.
        """
        cap_chars = 200
        body = "".join(f"{i:06d}\n" for i in range(10_000))
        cap = _BoundedCapture(cap_chars)
        cap.feed(body)
        rendered = _truncate_bash_output(cap.text(), cap_chars, true_len=cap.total)
        assert rendered.startswith(body[: cap_chars // 2])
        assert rendered.endswith(body[-(cap_chars // 2) :])


class TestTruncationReportsTheRealSize:
    def test_the_notice_names_what_the_command_produced(self):
        content = "h" * 500 + "t" * 500
        out = _truncate_bash_output(content, 200, true_len=108_000_000)
        assert "107999800" in out, (  # 108_000_000 - 200
            f"the notice describes the retained slice, not the run: {out!r}"
        )

    def test_it_falls_back_to_the_content_when_no_true_length_is_given(self):
        content = "z" * 1000
        out = _truncate_bash_output(content, 200)
        assert "800" in out

    def test_a_short_output_is_untouched(self):
        assert _truncate_bash_output("hi\n", 200, true_len=3) == "hi\n"


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="needs rusage")
def test_bash_does_not_materialise_a_huge_output(tool_registry, tmp_path):
    """The end-to-end property, measured the way the defect was found."""
    big = tmp_path / "big.log"
    with open(big, "w", encoding="utf-8") as fh:
        fh.write(("2026-07-30 INFO a log line with some text in it\n") * 400_000)

    before = _peak_mb()
    result = tool_registry.dispatch("bash", {"command": f"cat {big}", "timeout": 120})
    growth = _peak_mb() - before

    assert result.ok
    # ~19 MB of output. Materialising it costs the file size several times over
    # (bytes + decoded str); the bounded capture costs the cap.
    assert growth < 8, (
        f"bash grew peak RSS by {growth:.0f} MB for a ~130 KB answer — the "
        "output is being materialised before the budget applies"
    )
    assert "2026-07-30 INFO" in result.content
