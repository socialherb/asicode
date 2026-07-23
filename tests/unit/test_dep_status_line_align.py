"""Regression guard for dep-status line vertical alignment.

``_print_dep_status`` (asi.py) prints the tooling status line::

    tree-sitter: ON  eslint: ON  pyright: ON  ruff: ON  tsc: ON  vector: ON

It must sit one column in from the left so it vertically aligns with the
``/insights`` nudge continuation line printed above it in ``run_repl``::

    💡 design_insights: 16 items.
         /insights {list|verify|compact|drop <n>}
         tree-sitter: ON  eslint: ON  ...  vector: ON   <- aligns with /insights

``_MarginIO`` prepends ``_CONSOLE_MARGIN`` (4) to every line. The ``/insights``
continuation is emitted in ``run_repl`` as ``_print(" /insights" …)`` — exactly
1 literal leading space — landing at col 6 (margin 4 + 1). For the status line
to match, BOTH the Rich branch (``Text(" tree-sitter: ")``) and the non-Rich
fallback (``_print(" " + …)``) must carry exactly 1 literal leading space.

The cross-function invariant (test_status_line_matches_insights_nudge_column)
ties the two together: whatever literal lead ``run_repl`` uses for the
``/insights`` line, ``_print_dep_status`` must use the same lead for the
status line — otherwise they drift apart if either is edited in isolation.

These are source-contract tests (text parse) — they verify the wiring without
importing asi (which has heavy import-time side effects).
"""
import re

import pytest


def _read_asi() -> str:
    with open("asi.py", "r", encoding="utf-8") as f:
        return f.read()


def _get_print_dep_status_source() -> str:
    """Extract the module-level ``_print_dep_status`` body from asi.py."""
    lines = _read_asi().splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^def _print_dep_status\(", ln):
            start = i
            break
    if start is None:
        pytest.skip("_print_dep_status not found in asi.py")
    body = [lines[start]]
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        # stop at the next module-level construct (col-0 def/class/assignment)
        if ln[:1] and not ln.startswith((" ", "\t", "\n", "\r")):
            if re.match(r"^(def |class |[A-Za-z_][A-Za-z0-9_]*\s*[=:])", ln):
                break
        body.append(ln)
    return "".join(body)


def _rich_tree_sitter_lead(src: str) -> int:
    # capture only the leading spaces INSIDE Text(" ... tree-sitter: ")
    m = re.search(r'Text\("([ ]*)(tree-sitter: )"', src)
    assert m, 'Rich branch must build Text(" tree-sitter: ", ...)'
    return len(m.group(1))


def _plain_tree_sitter_lead(src: str) -> int:
    m = re.search(r'_print\("([ ]*)" \+ "  "\.join\(parts\)\)', src)
    assert m, 'non-Rich branch must call _print(" " + "  ".join(parts))'
    return len(m.group(1))


def _insights_nudge_lead(full_src: str) -> int:
    # run_repl splits the nudge and prints the continuation as _print(" /insights" + ...)
    m = re.search(r'_print\("([ ]*)/insights"', full_src)
    assert m, 'run_repl must print the nudge continuation via _print(" /insights" + ...)'
    return len(m.group(1))


class TestDepStatusLineAlignment:
    def test_rich_branch_has_one_leading_space(self):
        """Rich ``Text(" tree-sitter: ")`` must carry exactly 1 literal leading
        space → rendered at col 6 (margin 4 + 1), matching ``/insights``."""
        src = _get_print_dep_status_source()
        assert _rich_tree_sitter_lead(src) == 1

    def test_non_rich_branch_has_one_leading_space(self):
        """The non-Rich fallback ``_print(" " + "  ".join(parts))`` must use
        exactly 1 literal leading space, matching the Rich branch."""
        src = _get_print_dep_status_source()
        assert _plain_tree_sitter_lead(src) == 1

    def test_both_branches_share_the_same_lead(self):
        """Rich and non-Rich branches must agree on the literal leading-space
        count so the two code paths never diverge."""
        src = _get_print_dep_status_source()
        assert _rich_tree_sitter_lead(src) == _plain_tree_sitter_lead(src)

    def test_status_line_matches_insights_nudge_column(self):
        """The cross-function invariant: ``_print_dep_status``'s status line must
        start at the SAME column as the ``/insights`` nudge continuation line
        printed by ``run_repl``. Both leads are extracted from source so this
        catches drift if EITHER side is edited in isolation."""
        full = _read_asi()
        dep_src = _get_print_dep_status_source()
        margin = 4  # _CONSOLE_MARGIN
        # use the Rich branch lead as the status-line representative
        status_col = margin + _rich_tree_sitter_lead(dep_src)
        nudge_col = margin + _insights_nudge_lead(full)
        assert status_col == nudge_col, (
            f"dep-status line col {status_col} != /insights nudge col {nudge_col} "
            f"(margin {margin} + literal leads "
            f"{status_col - margin} vs {nudge_col - margin}); the two lines would "
            "no longer vertically align"
        )
