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
    llm = "```diff\n--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n context\n-old\n+new\n```\n"
    out = _p().extract_diff(llm)
    assert out, "valid diff was discarded"
    assert "+new" in out and "-old" in out


def test_insert_hunk_with_trailing_context_survives():
    llm = "```diff\n--- a/g.py\n+++ b/g.py\n@@ -1,3 +1,4 @@\n a\n b\n+inserted\n c\n```\n"
    out = _p().extract_diff(llm)
    assert out and "+inserted" in out


def test_fix_hunk_body_prefixes_no_phantom_context_line():
    """The trailing terminator token must not become a ' ' body line."""
    diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n context\n-old\n+new\n"
    fixed = _p()._fix_hunk_body_prefixes(diff)
    # No trailing bare-space phantom line.
    assert not fixed.endswith("+new\n \n"), fixed
    assert fixed.rstrip("\n").endswith("+new")


def test_hunk_counts_consistent_with_trailing_newline():
    p = _p()
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n context\n-old\n+new\n"
    assert p._hunks_have_consistent_line_counts(diff)


def test_excess_body_lines_still_rejected():
    """The fix must not weaken the real guard: too many body lines → reject."""
    p = _p()
    excess = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n a\n b\n c\n"
    assert not p._hunks_have_consistent_line_counts(excess)


def test_mid_hunk_empty_context_line_preserved():
    """An empty context line *inside* the hunk is real and must be counted."""
    p = _p()
    midblank = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n\n c\n"
    assert p._hunks_have_consistent_line_counts(midblank)


def test_parse_file_blocks_unfenced_multi_block_not_merged():
    """Unfenced FILE: blocks must stay separate; DOTALL used to merge them.

    Regression: FILE_BLOCK_RE was compiled with DOTALL ('s'), so the unfenced
    code2 branch's '.*' crossed newlines and one repetition swallowed every
    following line up to the last newline — silently dropping all FILE blocks
    after the first (their content was misattributed to the first file).
    """
    llm = "FILE: a.py\nprint('a')\nFILE: b.py\nprint('b')\n"
    blocks = _p().parse_file_blocks(llm)
    assert len(blocks) == 2, f"expected 2 blocks, got {len(blocks)}: {blocks}"
    assert blocks[0]["path"] == "a.py" and blocks[0]["text"] == "print('a')\n"
    assert blocks[1]["path"] == "b.py" and blocks[1]["text"] == "print('b')\n"


def test_parse_file_blocks_fenced_still_extracted():
    """Fenced full-file blocks (code1 branch) must keep working after the fix."""
    llm = "FILE: a.py\n```python\nprint('a')\n```\nFILE: b.py\n```python\nprint('b')\n```\n"
    blocks = _p().parse_file_blocks(llm)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "print('a')\n"
    assert blocks[1]["text"] == "print('b')\n"


# ── hunk-aware ---/+++ header vs body-line disambiguation ──────────────────
# Deleting a line whose content starts with "-- " renders as "--- ..." inside
# the hunk body (likewise "+" additions of "++ " content render as "+++ ..."),
# which is indistinguishable from a file header by prefix alone.  Git resolves
# this positionally (headers appear only after the preceding hunk's counts are
# exhausted), so the parser must too — previously pass1 rewrote such lines into
# bogus "--- a/<content>" headers and validate_diff listed the content as a
# touched file.


def test_auto_correct_preserves_double_dash_removal_body_line():
    diff = (
        "diff --git a/f.txt b/f.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " -- keep\n"
        "--- gone\n"
        "+-- kept\n"
    )
    out = _p().extract_diff(diff)
    assert out, "valid diff was discarded"
    assert "--- a/gone" not in out, "body line was rewritten into a bogus header"
    assert "--- gone" in out
    ok, err = _p().validate_diff(out, target_file="f.txt")
    assert ok, f"validate_diff rejected a valid diff: {err}"


def test_auto_correct_preserves_double_plus_addition_body_line():
    diff = "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1,1 +1,2 @@\n old\n+++ new\n"
    out = _p().extract_diff(diff)
    assert out, "valid diff was discarded"
    assert "+++ b/new" not in out, "body line was rewritten into a bogus header"
    assert "+++ new" in out
    ok, err = _p().validate_diff(out, target_file="f.txt")
    assert ok, f"validate_diff rejected a valid diff: {err}"


def test_auto_correct_multi_file_header_after_completed_hunk_still_recognized():
    """A real '--- a/f2.txt' header after a completed hunk must still be
    treated as a header (not swallowed as a body line)."""
    diff = (
        "diff --git a/f1.txt b/f1.txt\n"
        "--- a/f1.txt\n"
        "+++ b/f1.txt\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
        "diff --git a/f2.txt b/f2.txt\n"
        "--- f2.txt\n"
        "+++ f2.txt\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    out = _p().extract_diff(diff)
    assert out.count("diff --git") == 2
    # missing a/ b/ prefixes on the SECOND file's headers still normalized
    assert "--- a/f2.txt" in out and "+++ b/f2.txt" in out
    ok, err = _p().validate_diff(out)
    assert ok, err
    ok, err = _p().validate_diff(out, target_file="f1.txt")
    assert not ok  # two touched files → target-scope rejection preserved


def test_auto_correct_header_normalization_outside_hunks_unchanged():
    """Headers outside hunks still get a/ b/ prefixes injected."""
    diff = "--- f.txt\n+++ f.txt\n@@ -1 +1 @@\n-x\n+y\n"
    out = _p().extract_diff(diff)
    assert "--- a/f.txt" in out and "+++ b/f.txt" in out
    ok, err = _p().validate_diff(out, target_file="f.txt")
    assert ok, err


# ── parse_tool_args salvage contract ────────────────────────────────────────


def test_parse_tool_args_dict_passthrough():
    from external_llm.output_parser import parse_tool_args

    d = {"path": "a.py", "content": "x"}
    assert parse_tool_args(d) is d


def test_parse_tool_args_none_and_empty():
    from external_llm.output_parser import parse_tool_args

    assert parse_tool_args(None) == {}
    assert parse_tool_args("") == {}
    assert parse_tool_args("   ") == {}


def test_parse_tool_args_valid_json_string():
    from external_llm.output_parser import parse_tool_args

    assert parse_tool_args('{"path": "a.py", "content": "x"}') == {
        "path": "a.py",
        "content": "x",
    }


def test_parse_tool_args_non_dict_json_keeps_raw():
    from external_llm.output_parser import parse_tool_args

    assert parse_tool_args("[1, 2, 3]") == {"__raw_arguments": "[1, 2, 3]"}


def test_parse_tool_args_malformed_salvage():
    from external_llm.output_parser import parse_tool_args

    # prose around a JSON object → first {...} region salvaged
    assert parse_tool_args('Here: {"path": "a.py"} thanks') == {"path": "a.py"}
    # no braces → raw fallback
    assert parse_tool_args("no json here") == {"__raw_arguments": "no json here"}
    # truncated object → raw fallback
    assert parse_tool_args('{"path": "a.py"') == {"__raw_arguments": '{"path": "a.py"'}
    # brace inside a string value does not confuse the rfind("}") salvage
    assert parse_tool_args('{"content": "line}"}') == {"content": "line}"}
