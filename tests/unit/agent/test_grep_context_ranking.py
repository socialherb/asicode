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


def test_grep_flat_no_ranking_when_results_fit(tool_registry):
    """grep output preserves native file/line order — ranking's only job is to
    select which results survive a cap, never to scramble the display order.

    What this actually guards is the ``_top.sort()`` in ``_tool_grep``: BM25
    selects the top N by score and that line puts the survivors back in native
    order. Skipping ranking entirely (the ``len(lines) > max_results`` guard)
    and running it on a set that fits are therefore INDISTINGUISHABLE in the
    output — verified by mutation: forcing the branch to run with 2 results
    changes nothing. So this is an order-contract test, not a branch test, and
    it fails exactly when ``_top.sort()`` is removed.
    """
    root = pathlib.Path(tool_registry.repo_root)
    # Order is asserted WITHIN one file, not across files. ripgrep walks
    # directories in parallel and its inter-file output order is nondeterministic
    # (there is no --sort here — adding one would force rg single-threaded for
    # every grep the agent runs). The cross-file spelling of this assertion
    # therefore flaked: measured 3 order flips in 400 runs under CPU load, often
    # enough to fail a full-suite run and never reproduce in isolation. Line
    # order inside one file is deterministic, and it targets the property more
    # directly anyway — the damage ranking would do is intra-file spatial
    # locality (lines 136, 112, 36 instead of 36, 112, 136).
    #
    # The bait: the LATER line scores higher under BM25 (three occurrences vs
    # one), so a lost _top.sort() pulls it to the front.
    (root / "alpha.py").write_text(
        "# mytoken once\n"
        "filler = 1\n"
        "# mytoken mytoken mytoken\n"
    )
    res = tool_registry.dispatch(
        "grep", {"pattern": "mytoken", "context": 0, "path": "."}
    )
    assert res.ok, res.error
    assert "mytoken" in res.content
    assert res.content.index("alpha.py:1:") < res.content.index("alpha.py:3:"), (
        f"native line order not preserved (line 1 must precede line 3) — "
        f"_top.sort() lost?\n{res.content}")


def test_grep_reports_true_match_count_when_ranking_runs(tool_registry):
    """The header count and the "refine your pattern" notice must describe the
    MATCH set, not the displayed set.

    Regression: the ranking branch selects the top ``max_results`` and discards
    the rest, and ``total``/``truncated`` were computed from ``lines`` *after*
    that selection. A pattern with 400 matches reported "(5 matches)" with no
    truncation notice — the agent concludes it has seen every hit and stops
    searching. Only stop-word patterns escaped (``tokenize("import") == []``
    skips ranking), so ordinary identifier searches were the broken case.
    """
    root = pathlib.Path(tool_registry.repo_root)
    for f in range(4):
        # 100 matches each => 400 total, far above the max_results=5 below.
        (root / f"bulk{f}.py").write_text("".join(f"mytoken_{i}\n" for i in range(100)))

    res = tool_registry.dispatch(
        "grep", {"pattern": "mytoken", "context": 0, "path": ".", "max_results": 5}
    )
    assert res.ok, res.error
    header = res.content.splitlines()[0]
    assert "(400 matches)" in header, (
        f"header must report the true match count, not the displayed count.\n{header}")
    assert "of 400 matches" in res.content, (
        f"truncation notice missing — agent cannot tell results were dropped.\n{res.content}")


def test_grep_ranked_survivors_are_in_file_line_order(tool_registry):
    """When ranking DOES run, the surviving results are still displayed in
    native file/line order — ranking selects, it does not reorder."""
    root = pathlib.Path(tool_registry.repo_root)
    (root / "bulk.py").write_text("".join(f"mytoken_{i}\n" for i in range(300)))

    res = tool_registry.dispatch(
        "grep", {"pattern": "mytoken", "context": 0, "path": ".", "max_results": 10}
    )
    assert res.ok, res.error
    nums = _line_numbers(res.content)
    assert nums, res.content
    assert nums == sorted(nums), (
        f"ranked survivors must be re-sorted into file/line order: {nums}")
