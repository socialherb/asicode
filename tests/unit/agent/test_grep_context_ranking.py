"""Regression tests for _tool_grep BM25 re-ranking of grep output.

Covers a correctness defect: when context>0 (-C N), grep/rg output is spatially
grouped (match lines, context lines, group separators) and the prior code
re-ranked EVERY line independently by BM25 score. That destroyed the grouping
— context lines detached from their match, line numbers shuffled out of order,
and separators floated to meaningless spots. The fix gates BM25 to context==0
(flat match-lines only).
"""
import pathlib
import re


def _line_numbers(output: str) -> list[int]:
    """Extract the lineno from each grep output line.

    Handles both single-file format (``lineno:content`` / ``lineno-content``)
    and multi-file format (``path:lineno:content`` / ``path-lineno-content``).
    The header line and ``--`` separators carry no ``<digits>:``/``-`` token and
    are skipped.
    """
    nums: list[int] = []
    for ln in output.splitlines()[1:]:  # skip the "rg: '...' in ..." header
        m = re.search(r"(\d+)[:\-]", ln)
        if m:
            nums.append(int(m.group(1)))
    return nums


def _svc_py(root: pathlib.Path) -> pathlib.Path:
    """A file with two 'connect_database' matches far apart (L3 and L18), with
    enough padding that context groups don't merge."""
    (root / "svc.py").write_text(
        "def setup(self):\n"
        "    x = 1\n"
        "    connect_database()\n"          # L3 — match
        "    return x\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "def pool():\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "    pass\n"
        "def more():\n"
        "    connect_database()\n"          # L18 — match
    )
    return root / "svc.py"


def test_grep_context_does_not_scramble_groups(tool_registry):
    """context>0 must preserve native grep group order: line numbers ascending
    within the file. Regression: BM25 re-ranked every line and scrambled the
    order to e.g. 3,18,5,4,2,17,16,1 (matches floated to top, context detached).
    """
    root = pathlib.Path(tool_registry.repo_root)
    _svc_py(root)
    res = tool_registry.dispatch(
        "grep", {"pattern": "connect_database", "context": 2, "path": "svc.py"}
    )
    assert res.ok, res.error
    nums = _line_numbers(res.content)
    assert nums, f"no line numbers parsed from output:\n{res.content}"
    assert nums == sorted(nums), (
        f"context>0 output scrambled (non-ascending line numbers): {nums}")
    # both matches present with their context
    assert 3 in nums and 18 in nums


def test_grep_context_match_keeps_neighbors(tool_registry):
    """context=1: a match line must be immediately followed by its trailing
    context line in native order, not a detached line from another match.
    """
    root = pathlib.Path(tool_registry.repo_root)
    _svc_py(root)
    res = tool_registry.dispatch(
        "grep", {"pattern": "connect_database", "context": 1, "path": "svc.py"}
    )
    assert res.ok, res.error
    out_lines = res.content.splitlines()[1:]  # skip header
    # first match line is L3; its trailing context (L4) must be the very next line
    match_idx = next(i for i, ln in enumerate(out_lines) if re.match(r"^3:", ln))
    assert re.match(r"^4[-:]", out_lines[match_idx + 1]), (
        f"trailing context L4 not adjacent to match L3: "
        f"{out_lines[match_idx:match_idx + 3]}")


def test_grep_flat_ranking_still_active(tool_registry):
    """Control: with context==0, BM25 re-ranking of flat match-lines must still
    run — a line whose content is richer in the query token ranks above a line
    with a single occurrence. Guards against the context-gate accidentally
    disabling ALL ranking.
    """
    root = pathlib.Path(tool_registry.repo_root)
    # query-neutral filenames so filename tokenization cannot affect the order
    (root / "alpha.py").write_text("# mytoken once\n")
    (root / "omega.py").write_text("# mytoken mytoken mytoken\n")
    res = tool_registry.dispatch(
        "grep", {"pattern": "mytoken", "context": 0, "path": "."}
    )
    assert res.ok, res.error
    assert "mytoken" in res.content
    # omega (3 occurrences) must rank above alpha (1 occurrence)
    assert res.content.index("omega.py") < res.content.index("alpha.py"), (
        f"BM25 ranking inactive: omega should rank above alpha.\n{res.content}")
