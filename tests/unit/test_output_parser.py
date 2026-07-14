"""Regression tests for EnhancedOutputParser diff extraction.

Focus: the trailing-newline phantom-context bug, where a valid unified diff that
ends with a newline (the norm) had its final ``split("\\n")`` artifact turned
into a phantom " " context line.  That overflowed the @@ header counts and made
``extract_diff`` discard an otherwise-applicable patch ("No diff found").
"""
from external_llm.output_parser import EnhancedOutputParser


def _p() -> EnhancedOutputParser:
    return EnhancedOutputParser()


def test_valid_diff_with_trailing_newline_not_dropped():
    """A fenced diff ending in a newline must survive extraction."""
    llm = (
        "```diff\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context\n"
        "-old\n"
        "+new\n"
        "```\n"
    )
    out = _p().extract_diff(llm)
    assert out, "valid diff was discarded"
    assert "+new" in out and "-old" in out


def test_insert_hunk_with_trailing_context_survives():
    llm = (
        "```diff\n"
        "--- a/g.py\n"
        "+++ b/g.py\n"
        "@@ -1,3 +1,4 @@\n"
        " a\n"
        " b\n"
        "+inserted\n"
        " c\n"
        "```\n"
    )
    out = _p().extract_diff(llm)
    assert out and "+inserted" in out


def test_fix_hunk_body_prefixes_no_phantom_context_line():
    """The trailing terminator token must not become a ' ' body line."""
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context\n"
        "-old\n"
        "+new\n"
    )
    fixed = _p()._fix_hunk_body_prefixes(diff)
    # No trailing bare-space phantom line.
    assert not fixed.endswith("+new\n \n"), fixed
    assert fixed.rstrip("\n").endswith("+new")


def test_hunk_counts_consistent_with_trailing_newline():
    p = _p()
    diff = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context\n"
        "-old\n"
        "+new\n"
    )
    assert p._hunks_have_consistent_line_counts(diff)


def test_excess_body_lines_still_rejected():
    """The fix must not weaken the real guard: too many body lines → reject."""
    p = _p()
    excess = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,1 +1,1 @@\n"
        " a\n"
        " b\n"
        " c\n"
    )
    assert not p._hunks_have_consistent_line_counts(excess)


def test_mid_hunk_empty_context_line_preserved():
    """An empty context line *inside* the hunk is real and must be counted."""
    p = _p()
    midblank = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a\n"
        "\n"
        " c\n"
    )
    assert p._hunks_have_consistent_line_counts(midblank)


def test_parse_file_blocks_unfenced_multi_block_not_merged():
    """Unfenced FILE: blocks must stay separate; DOTALL used to merge them.

    Regression: FILE_BLOCK_RE was compiled with DOTALL ('s'), so the unfenced
    code2 branch's '.*' crossed newlines and one repetition swallowed every
    following line up to the last newline — silently dropping all FILE blocks
    after the first (their content was misattributed to the first file).
    """
    llm = (
        "FILE: a.py\n"
        "print('a')\n"
        "FILE: b.py\n"
        "print('b')\n"
    )
    blocks = _p().parse_file_blocks(llm)
    assert len(blocks) == 2, f"expected 2 blocks, got {len(blocks)}: {blocks}"
    assert blocks[0]["path"] == "a.py" and blocks[0]["text"] == "print('a')\n"
    assert blocks[1]["path"] == "b.py" and blocks[1]["text"] == "print('b')\n"


def test_parse_file_blocks_fenced_still_extracted():
    """Fenced full-file blocks (code1 branch) must keep working after the fix."""
    llm = (
        "FILE: a.py\n"
        "```python\n"
        "print('a')\n"
        "```\n"
        "FILE: b.py\n"
        "```python\n"
        "print('b')\n"
        "```\n"
    )
    blocks = _p().parse_file_blocks(llm)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "print('a')\n"
    assert blocks[1]["text"] == "print('b')\n"
