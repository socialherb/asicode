"""Regression tests for HybridOutputParser.

Focus (HP-B1): ``_parse_diff`` used ``matches[0]`` to select the fenced diff
block.  When a model splits a multi-file patch across several ``\\`\\`\\`diff``
fences (one file per fence), only the first file survived — every subsequent
file's changes were silently dropped while ``parse`` reported success.  The
patch then "applied" but never touched the other files.

These tests pin the multi-fence join contract plus the surrounding strict
validation / fallback behaviour so the join cannot silently weaken any gate.
"""
from external_llm.hybrid_parser import HybridOutputParser, ParseResult
from external_llm.output_modes import OutputMode


def _p() -> HybridOutputParser:
    return HybridOutputParser()


_FOO_DIFF = (
    "diff --git a/foo.py b/foo.py\n"
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1,3 +1,3 @@\n"
    " context\n"
    "-old\n"
    "+new\n"
)

_BAR_DIFF = (
    "diff --git a/bar.py b/bar.py\n"
    "--- a/bar.py\n"
    "+++ b/bar.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-x\n"
    "+y\n"
)


# ── HP-B1: multi-fence join ────────────────────────────────────────────────

def test_multi_fence_diff_preserves_all_files():
    """Each ```diff fence contributes a file; both must end up in the diff."""
    llm = f"```diff\n{_FOO_DIFF}```\n```diff\n{_BAR_DIFF}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success, r.error
    assert r.mode == OutputMode.UNIFIED_DIFF
    assert "foo.py" in r.diff
    assert "bar.py" in r.diff
    assert "+new" in r.diff and "+y" in r.diff


def test_multi_fence_diff_file_count_matches():
    """Joining must not merge or drop file boundaries."""
    llm = f"```diff\n{_FOO_DIFF}```\n```diff\n{_BAR_DIFF}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success
    assert r.diff.count("diff --git") == 2
    assert r.diff.count("@@ ") == 2


def test_single_fence_multi_file_still_works():
    """A multi-file diff inside ONE fence (the well-behaved case) is unaffected."""
    llm = f"```diff\n{_FOO_DIFF}{_BAR_DIFF}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success
    assert "foo.py" in r.diff and "bar.py" in r.diff


def test_three_fences_all_preserved():
    """Generalises the join beyond two fences."""
    baz = (
        "diff --git a/baz.py b/baz.py\n"
        "--- a/baz.py\n"
        "+++ b/baz.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-p\n"
        "+q\n"
    )
    llm = f"```diff\n{_FOO_DIFF}```\n```diff\n{_BAR_DIFF}```\n```diff\n{baz}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success
    assert r.diff.count("diff --git") == 3
    assert "foo.py" in r.diff and "bar.py" in r.diff and "baz.py" in r.diff


def test_single_fence_single_file_basic():
    """The common single-file case still parses."""
    llm = f"```diff\n{_FOO_DIFF}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success and "foo.py" in r.diff


def test_unfenced_raw_diff_still_parsed():
    """No fences at all → fall back to the stripped whole text (unchanged path)."""
    llm = _FOO_DIFF
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success
    assert "foo.py" in r.diff


# ── strict validation gates (must not be weakened by the join) ─────────────

def test_reject_diff_missing_hunk():
    """Prose that merely starts with 'diff --git' but has no @@ hunk → reject."""
    llm = "```diff\ndiff --git a/foo.py b/foo.py\nsome explanation, no hunk\n```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert not r.success


def test_reject_empty_input():
    r = _p().parse("", OutputMode.UNIFIED_DIFF)
    assert not r.success


def test_reject_no_diff_markers():
    r = _p().parse("just some prose, nothing diff-like here", OutputMode.UNIFIED_DIFF)
    assert not r.success


# ── NEEDS_DISAMBIGUATION short-circuit (parse-level, pre-dispatch) ─────────

def test_needs_disambiguation_short_circuit():
    llm = "NEEDS_DISAMBIGUATION: which file do you mean?\n```diff\n{_FOO_DIFF}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert r.success is True
    assert r.mode is None  # disambiguation sentinel


# ── fallback: expected mode fails → re-parsed as UNIFIED_DIFF with warning ─

def test_fallback_to_unified_diff_adds_warning():
    """When the expected mode (FULL_FILE) fails but the text is a real diff,
    parse() should recover as UNIFIED_DIFF and append a warning."""
    llm = f"```diff\n{_FOO_DIFF}```\n"
    r = _p().parse(llm, OutputMode.FULL_FILE)
    assert r.success
    assert r.mode == OutputMode.UNIFIED_DIFF
    assert any("unified_diff" in w for w in r.warnings)


# ── ParseResult is the right type / raw passthrough ────────────────────────

def test_parse_result_raw_output_preserved():
    llm = f"```diff\n{_FOO_DIFF}```\n"
    r = _p().parse(llm, OutputMode.UNIFIED_DIFF)
    assert isinstance(r, ParseResult)
    assert r.raw_output == llm


# ════════════════════════════════════════════════════════════════════════════
# Surrounding mode coverage: the other four parsers + the dispatch table.
# These were 0%-branch in the accuracy round; pinning them closes the module
# to ~100% branch coverage and guards against regressions in the dispatch.
# ════════════════════════════════════════════════════════════════════════════

# ── _parse_asicode ─────────────────────────────────────────────────────────

def test_asicode_valid_single_block():
    llm = (
        "ASICODE_BEGIN\n"
        "BEFORE\n"
        "old line\n"
        "AFTER\n"
        "new line\n"
        "ASICODE_END\n"
    )
    r = _p().parse(llm, OutputMode.ASICODE_BLOCK)
    assert r.success and r.mode == OutputMode.ASICODE_BLOCK
    assert r.blocks == [{"before": "old line", "after": "new line"}]


def test_asicode_multiple_blocks():
    llm = (
        "ASICODE_BEGIN\nBEFORE\na\nAFTER\nb\nASICODE_END\n"
        "ASICODE_BEGIN\nBEFORE\nc\nAFTER\nd\nASICODE_END\n"
    )
    r = _p().parse(llm, OutputMode.ASICODE_BLOCK)
    assert r.success and len(r.blocks) == 2
    assert r.blocks[0] == {"before": "a", "after": "b"}
    assert r.blocks[1] == {"before": "c", "after": "d"}


def test_asicode_no_blocks_rejected():
    r = _p().parse("nothing here", OutputMode.ASICODE_BLOCK)
    assert not r.success


def test_asicode_block_missing_before_after_rejected():
    """ASICODE_BEGIN/END present but no BEFORE/AFTER markers → reject."""
    llm = "ASICODE_BEGIN\nsome content\nASICODE_END\n"
    r = _p().parse(llm, OutputMode.ASICODE_BLOCK)
    assert not r.success


# ── _parse_targeted ────────────────────────────────────────────────────────

def test_targeted_valid():
    llm = (
        "FUNCTION: my_func\n"
        "INSERT_AFTER: def other():\n"
        "```python\n"
        "    pass\n"
        "```\n"
    )
    r = _p().parse(llm, OutputMode.TARGETED_BLOCK)
    assert r.success and r.mode == OutputMode.TARGETED_BLOCK
    assert r.code == "    pass"
    assert r.insert_point == "def other():"


def test_targeted_missing_function_rejected():
    llm = "INSERT_AFTER: x\n```python\npass\n```\n"
    r = _p().parse(llm, OutputMode.TARGETED_BLOCK)
    assert not r.success


def test_targeted_missing_insert_after_rejected():
    llm = "FUNCTION: f\n```python\npass\n```\n"
    r = _p().parse(llm, OutputMode.TARGETED_BLOCK)
    assert not r.success


def test_targeted_missing_code_block_rejected():
    llm = "FUNCTION: f\nINSERT_AFTER: x\n"
    r = _p().parse(llm, OutputMode.TARGETED_BLOCK)
    assert not r.success


# ── _parse_full_file ───────────────────────────────────────────────────────

def test_full_file_fenced():
    llm = "FILE: mod.py\n```python\nprint('hi')\n```\n"
    r = _p().parse(llm, OutputMode.FULL_FILE)
    assert r.success and r.mode == OutputMode.FULL_FILE
    assert r.file_path == "mod.py"
    assert r.content == "print('hi')\n"


def test_full_file_unfenced():
    llm = "FILE: mod.py\nprint('hi')\n"
    r = _p().parse(llm, OutputMode.FULL_FILE)
    assert r.success
    assert r.content == "print('hi')\n"


def test_full_file_no_marker_rejected():
    r = _p().parse("just prose here", OutputMode.FULL_FILE)
    assert not r.success


def test_full_file_empty_content_rejected():
    """FILE marker present but no code body → reject."""
    llm = "FILE: mod.py\n\n"
    r = _p().parse(llm, OutputMode.FULL_FILE)
    assert not r.success


# ── _parse_plan ────────────────────────────────────────────────────────────

def test_plan_valid():
    llm = "```json\n{\"operations\": [{\"op\": \"create\", \"path\": \"a.py\"}]}\n```\n"
    r = _p().parse(llm, OutputMode.PLAN_JSON)
    assert r.success and r.mode == OutputMode.PLAN_JSON
    assert r.plan == {"operations": [{"op": "create", "path": "a.py"}]}


def test_plan_no_json_block_rejected():
    r = _p().parse("no json here", OutputMode.PLAN_JSON)
    assert not r.success


def test_plan_invalid_json_rejected():
    llm = "```json\n{not valid json}\n```\n"
    r = _p().parse(llm, OutputMode.PLAN_JSON)
    assert not r.success


def test_plan_missing_operations_rejected():
    llm = "```json\n{\"steps\": []}\n```\n"
    r = _p().parse(llm, OutputMode.PLAN_JSON)
    assert not r.success


# ── dispatch table: total-failure path ─────────────────────────────────────

def test_total_parse_failure():
    """Expected mode rejects AND both fallback modes reject → generic failure."""
    r = _p().parse("totally unstructured prose with no markers at all",
                   OutputMode.TARGETED_BLOCK)
    assert not r.success
    assert r.raw_output == "totally unstructured prose with no markers at all"
