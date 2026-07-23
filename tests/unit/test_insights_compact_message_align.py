"""Regression guard for compact-message vertical alignment.

``_compact_insights_interactive`` (asi.py) prints a multi-line summary after
compaction::

    before: N entries · B bytes
    after:  M entries · C bytes
    ✓ design_insights compacted (…) + K entries demoted … (not deleted; …)

These lines carry a literal 2-space indent. ``_MarginIO`` prepends a 4-space
left margin (``_CONSOLE_MARGIN``) to *every* line, so the first line of each
``_print`` lands at col 6 (4 + 2). But Rich's auto-wrap preserves the literal
indent only on the FIRST wrapped line — continuation lines start at col 4
(margin alone), leaving ``deleted`` / ``Active now`` dangling one column short
of ``before`` / ``after``.

The fix pre-wraps the long demotion/backstop messages with ``textwrap.fill``
using *identical* ``initial_indent`` / ``subsequent_indent`` so every wrapped
line keeps the literal 2-space indent and stays aligned at col 6.

These are source-contract tests (text parse) — they verify the wiring exists
without importing asi (which has heavy import-time side effects).
"""
import re
import textwrap

import pytest


def _get_compact_insights_source() -> str:
    """Extract ``_compact_insights_interactive`` source from asi.py.

    asi.py is a large script with import-time side effects; the function is a
    closure inside ``run_repl`` so we locate it by its ``def`` line and
    balance-indent the body.
    """
    src_path = "asi.py"
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^    def _compact_insights_interactive\(\) -> bool:", ln):
            start = i
            break
    if start is None:
        pytest.skip("_compact_insights_interactive not found in asi.py")
    body = [lines[start]]
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("\t"):
            if re.match(r"^    def ", ln) or re.match(r"^def ", ln) or re.match(r"^    [a-zA-Z_]", ln):
                break
        body.append(ln)
    return "".join(body)


class TestCompactMessageAlignment:
    def test_long_messages_pre_wrapped_with_paired_indent(self):
        """Both long messages (demotion + backstop) must be pre-wrapped with
        ``textwrap.fill`` and each call must pair an ``initial_indent`` with an
        equal ``subsequent_indent`` — otherwise Rich's auto-wrap drops the
        literal indent on continuation lines and they misalign under
        ``before:`` / ``after:``."""
        src = _get_compact_insights_source()
        assert "textwrap.fill" in src, (
            "long compact messages must be pre-wrapped via textwrap.fill so Rich's "
            "auto-wrap (which drops the literal indent on continuation lines) never "
            "runs on them"
        )
        n_init = src.count('initial_indent="  "')
        n_sub = src.count('subsequent_indent="  "')
        assert n_init == n_sub, (
            "each textwrap.fill must pair initial_indent with an equal "
            f"subsequent_indent (got {n_init} initial vs {n_sub} subsequent)"
        )
        assert n_init >= 2, (
            "both the demotion and the over-budget-backstop message must be "
            f"pre-wrapped (found only {n_init} textwrap.fill call(s) with indent)"
        )

    def test_margin_plus_indent_aligns_all_wrapped_lines(self):
        """Render the demotion message through the ``_MarginIO`` column model
        (margin 4 + literal indent 2) at several terminal widths and assert
        every wrapped line's text starts at the same column (col 6). This is
        the exact invariant the fix relies on."""
        msg = (
            "✓ design_insights compacted (19→16 entries) + 3 oldest entries "
            "demoted to design_insights_archive.md (not deleted; "
            "/insights archive list|restore). Active now 5,855 bytes ≤ 6,000 budget."
        )
        margin = 4  # _CONSOLE_MARGIN
        for cols in (60, 80, 100, 120):
            console_w = max(40, cols - margin * 2)
            filled = textwrap.fill(
                msg, width=console_w,
                initial_indent="  ", subsequent_indent="  ",
            )
            text_cols = {
                len((" " * margin + ln)) - len((" " * margin + ln).lstrip(" "))
                for ln in filled.split("\n")
            }
            assert text_cols == {6}, (
                f"cols={cols}: wrapped lines start at columns {text_cols}, "
                "expected all at col 6 (margin 4 + literal 2)"
            )
