"""
Tests for common.py — shared utility functions.
"""
from __future__ import annotations

from common import (
    chunk_list,
    ensure_trailing_newline,
    norm_ws,
    normalize_rel_path_fast,
    safe_filename,
    unique_keep_order,
)


class TestUniqueKeepOrder:
    """Tests for unique_keep_order."""

    def test_basic_dedup(self):
        assert unique_keep_order([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]

    def test_preserves_order(self):
        assert unique_keep_order([3, 1, 2, 3, 2]) == [3, 1, 2]

    def test_no_duplicates(self):
        assert unique_keep_order(["a", "b", "c"]) == ["a", "b", "c"]

    def test_empty_list(self):
        assert unique_keep_order([]) == []

    def test_none_input(self):
        assert unique_keep_order(None) == []

    def test_removes_falsy_items(self):
        assert unique_keep_order([0, "", None, "a", 0]) == ["a"]

    def test_mixed_types(self):
        assert unique_keep_order([1, "1", 1, "1"]) == [1, "1"]


class TestSafeFilename:
    """Tests for safe_filename."""

    def test_normal_string(self):
        assert safe_filename("hello.txt") == "hello.txt"

    def test_replaces_invalid_chars(self):
        assert safe_filename("a/b:c*d?e\"f<g>h|i") == "a_b_c_d_e_f_g_h_i"

    def test_compresses_underscores(self):
        assert safe_filename("a___b__c") == "a_b_c"

    def test_strips_leading_trailing_spaces_and_dots(self):
        assert safe_filename("  file.txt  ") == "file.txt"
        assert safe_filename(".file.txt.") == "file.txt"
        assert safe_filename("  .  ") == "unnamed"

    def test_truncates_to_max_len(self):
        long_name = "a" * 200
        result = safe_filename(long_name, max_len=10)
        assert len(result) <= 10
        assert result.rstrip("_. ") == result  # no trailing junk

    def test_empty_result_returns_unnamed(self):
        # All-strip characters result in empty string → 'unnamed'
        assert safe_filename("...") == "unnamed"
        assert safe_filename("   ") == "unnamed"

    def test_backslash_replaced(self):
        assert safe_filename("a\\b") == "a_b"


class TestNormWs:
    """Tests for norm_ws."""

    def test_strips_whitespace(self):
        assert norm_ws("  hello  ") == "hello"

    def test_empty_string(self):
        assert norm_ws("") == ""

    def test_none_input(self):
        assert norm_ws(None) == ""

    def test_no_change_for_clean_string(self):
        assert norm_ws("hello") == "hello"


class TestEnsureTrailingNewline:
    """Tests for ensure_trailing_newline."""

    def test_adds_newline(self):
        assert ensure_trailing_newline("hello") == "hello\n"

    def test_preserves_existing_newline(self):
        assert ensure_trailing_newline("hello\n") == "hello\n"

    def test_normalizes_crlf(self):
        assert ensure_trailing_newline("hello\r\n") == "hello\n"

    def test_normalizes_cr(self):
        assert ensure_trailing_newline("hello\r") == "hello\n"

    def test_mixed_line_endings(self):
        result = ensure_trailing_newline("hello\r\nworld\r")
        assert result == "hello\nworld\n"

    def test_empty_string(self):
        assert ensure_trailing_newline("") == "\n"

    def test_none_input(self):
        assert ensure_trailing_newline(None) == "\n"


class TestChunkList:
    """Tests for chunk_list."""

    def test_exact_chunk(self):
        assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_remainder_chunk(self):
        assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        assert chunk_list([1, 2, 3], 5) == [[1, 2, 3]]

    def test_empty_list(self):
        assert chunk_list([], 3) == []

    def test_none_input(self):
        assert chunk_list(None, 3) == []

    def test_chunk_size_one(self):
        assert chunk_list([1, 2, 3], 1) == [[1], [2], [3]]

    def test_chunk_size_larger_than_list(self):
        assert chunk_list([1, 2], 10) == [[1, 2]]


class TestNormalizeRelPathFast:
    """Tests for normalize_rel_path_fast."""

    def test_strip_dot_slash(self):
        assert normalize_rel_path_fast("./foo/bar.py") == "foo/bar.py"

    def test_strip_whitespace(self):
        assert normalize_rel_path_fast("  foo/bar  ") == "foo/bar"

    def test_multiple_dot_slash(self):
        assert normalize_rel_path_fast("././foo.py") == "foo.py"

    def test_no_change(self):
        assert normalize_rel_path_fast("foo/bar.py") == "foo/bar.py"

    def test_empty_string(self):
        assert normalize_rel_path_fast("") == ""

    def test_none_input(self):
        assert normalize_rel_path_fast(None) == ""
