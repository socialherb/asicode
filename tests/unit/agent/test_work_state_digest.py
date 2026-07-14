"""Tests for work_state_digest — build_work_state_digest."""

from __future__ import annotations

from external_llm.agent.work_state_digest import (
    _arg_path,
    _arg_query,
    _dedup_capped,
    _first_line,
    build_work_state_digest,
)


class TestArgPath:
    """Tests for _arg_path helper."""

    def test_empty_args(self):
        assert _arg_path({}) == ""

    def test_path_key(self):
        assert _arg_path({"path": "foo.py"}) == "foo.py"

    def test_file_path_key(self):
        assert _arg_path({"file_path": "src/main.py"}) == "src/main.py"

    def test_target_file_key(self):
        assert _arg_path({"target_file": "bar.ts"}) == "bar.ts"

    def test_first_matching_key_wins(self):
        assert _arg_path({"path": "a.py", "file_path": "b.py"}) == "a.py"

    def test_empty_string_is_skipped(self):
        assert _arg_path({"path": "", "file_path": "real.py"}) == "real.py"

    def test_non_string_value_skipped(self):
        assert _arg_path({"path": ["a.py"]}) == ""

    def test_none_value_skipped(self):
        assert _arg_path({"path": None}) == ""

    def test_stripped_value(self):
        assert _arg_path({"path": "  spaced.py  "}) == "spaced.py"


class TestArgQuery:
    """Tests for _arg_query helper."""

    def test_empty_args(self):
        assert _arg_query({}) == ""

    def test_query_key(self):
        assert _arg_query({"query": "find foo"}) == "find foo"

    def test_pattern_key(self):
        assert _arg_query({"pattern": "def foo"}) == "def foo"

    def test_symbol_key(self):
        assert _arg_query({"symbol": "MyClass"}) == "MyClass"

    def test_url_key(self):
        assert _arg_query({"url": "https://example.com"}) == "https://example.com"

    def test_first_matching_key_wins(self):
        assert _arg_query({"pattern": "first", "query": "second"}) == "first"

    def test_truncation(self):
        q = "a" * 50
        result = _arg_query({"query": q})
        assert len(result) == 41  # 40 chars + ellipsis = 41
        assert result.endswith("…")

    def test_under_limit_no_truncation(self):
        q = "a" * 39
        result = _arg_query({"query": q})
        assert result == q
        assert not result.endswith("…")

    def test_exactly_at_limit_no_truncation(self):
        q = "a" * 40
        result = _arg_query({"query": q})
        assert result == q


class TestDedupCapped:
    """Tests for _dedup_capped helper."""

    def test_empty_list(self):
        assert _dedup_capped([], 10) == []

    def test_all_unique(self):
        assert _dedup_capped(["a", "b", "c"], 5) == ["a", "b", "c"]

    def test_duplicates_removed(self):
        assert _dedup_capped(["a", "b", "a", "c", "b"], 5) == ["a", "b", "c"]

    def test_order_preserved(self):
        assert _dedup_capped(["b", "a", "c", "a", "b"], 5) == ["b", "a", "c"]

    def test_capped_with_overflow_marker(self):
        items = ["a", "b", "c", "d", "e", "f"]
        result = _dedup_capped(items, 3)
        assert result == ["a", "b", "c", "(+3 more)"]

    def test_empty_strings_filtered(self):
        assert _dedup_capped(["a", "", "b", ""], 5) == ["a", "b"]

    def test_exactly_at_cap_no_marker(self):
        items = ["a", "b", "c"]
        assert _dedup_capped(items, 3) == ["a", "b", "c"]

    def test_zero_cap(self):
        assert _dedup_capped(["a", "b"], 0) == ["(+2 more)"]


class TestFirstLine:
    """Tests for _first_line helper."""

    def test_single_line(self):
        assert _first_line("hello world", 100) == "hello world"

    def test_multiline_text(self):
        assert _first_line("line1\nline2\nline3", 100) == "line1"

    def test_truncation(self):
        result = _first_line("a" * 50, 20)
        assert len(result) == 21  # 20 chars + ellipsis = 21
        assert result.endswith("…")

    def test_under_limit(self):
        assert _first_line("short", 100) == "short"

    def test_empty_text(self):
        assert _first_line("", 100) == ""

    def test_just_whitespace(self):
        assert _first_line("  \n  ", 100) == ""

    def test_none_coerced(self):
        assert _first_line(None, 10) == ""


class TestBuildWorkStateDigest:
    """Tests for build_work_state_digest(tool_results) -> str."""

    def test_empty_results(self):
        assert build_work_state_digest([]) == ""

    def test_none_results(self):
        assert build_work_state_digest(None) == ""

    def test_only_ignored_tools(self):
        """Ignored/UI tools produce no digest."""
        results = [
            {"tool": "ask_user", "args": {"question": "ok?"}, "ok": True},
            {"tool": "save_insight", "args": {}, "ok": True},
        ]
        assert build_work_state_digest(results) == ""

    def test_read_file(self):
        results = [{"tool": "read_file", "args": {"path": "main.py"}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "main.py" in digest
        assert "read" in digest

    def test_read_symbol(self):
        results = [
            {"tool": "read_symbol", "args": {"name": "MyClass", "file_path": "models.py"}, "ok": True}
        ]
        digest = build_work_state_digest(results)
        assert "models.py:MyClass" in digest

    def test_apply_patch(self):
        results = [{"tool": "apply_patch", "args": {"path": "fix.py"}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "fix.py (apply_patch)" in digest

    def test_write_with_failure_status(self):
        results = [{"tool": "apply_patch", "args": {"path": "fix.py"}, "ok": False, "content": "error"}]
        digest = build_work_state_digest(results)
        assert "fix.py (apply_patch FAILED)" in digest

    def test_bash_command(self):
        results = [{"tool": "bash", "args": {"command": "ls -la"}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "bash: ls -la" in digest
        assert "→ ok" in digest

    def test_bash_command_failed(self):
        results = [{"tool": "bash", "args": {"command": "rm -rf /"}, "ok": False}]
        digest = build_work_state_digest(results)
        assert "→ FAILED" in digest
        assert "failed" in digest  # also appears in failures section

    def test_search_tools(self):
        results = [
            {"tool": "grep", "args": {"pattern": "def main"}, "ok": True},
            {"tool": "find_symbol", "args": {"name": "Foo"}, "ok": True},
        ]
        digest = build_work_state_digest(results)
        assert "searched" in digest
        assert "grep def main" in digest
        assert "find_symbol Foo" in digest

    def test_query_dependency_graph(self):
        """query_dependency_graph with 'source' key — not in standard query keys → tool name only."""
        results = [{"tool": "query_dependency_graph", "args": {"source": "main.py"}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "searched" in digest
        assert "query_dependency_graph" in digest

    def test_unknown_tool_in_other_section(self):
        results = [{"tool": "some_custom_tool", "args": {}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "other tools" in digest
        assert "some_custom_tool" in digest

    def test_non_dict_entry_skipped(self):
        results = [42, {"tool": "bash", "args": {"command": "ls"}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "bash: ls" in digest

    def test_missing_tool_key(self):
        results = [{"args": {}, "ok": True}]
        assert build_work_state_digest(results) == ""

    def test_missing_args_key(self):
        results = [{"tool": "bash", "ok": True}]
        digest = build_work_state_digest(results)
        # missing args → treated as empty dict, command will be empty
        assert "bash" in digest

    def test_dedup_and_cap(self):
        """Multiple identical read_file entries → dedup to 1, no overflow marker."""
        results = [
            {"tool": "read_file", "args": {"path": "a.py"}, "ok": True},
        ] * 15  # 15 identical entries
        digest = build_work_state_digest(results)
        assert "read" in digest
        assert "a.py" in digest
        assert "(+14 more)" not in digest  # all identical → dedup to 1

    def test_many_unique_reads_triggers_cap(self):
        """More than MAX_ITEMS_PER_SECTION unique entries → cap with overflow marker."""
        results = [
            {"tool": "read_file", "args": {"path": f"file_{i}.py"}, "ok": True}
            for i in range(15)
        ]
        digest = build_work_state_digest(results)
        assert "(+5 more)" in digest  # 10 shown + "(+5 more)"

    def test_failure_section_with_excerpt(self):
        results = [
            {"tool": "bash", "args": {"command": "failing_command"}, "ok": False, "content": "error: permission denied"},
        ]
        digest = build_work_state_digest(results)
        assert "failed" in digest
        assert "error: permission denied" in digest

    def test_multi_section_digest(self):
        """Combination of reads, writes, commands, searches in one digest."""
        results = [
            {"tool": "read_file", "args": {"path": "a.py"}, "ok": True},
            {"tool": "apply_patch", "args": {"path": "b.py"}, "ok": True},
            {"tool": "bash", "args": {"command": "pytest"}, "ok": True},
            {"tool": "grep", "args": {"pattern": "TODO"}, "ok": True},
            {"tool": "save_insight", "args": {}, "ok": True},  # ignored
        ]
        digest = build_work_state_digest(results)
        assert "read" in digest and "modified" in digest
        assert "ran" in digest and "searched" in digest
        assert "other tools" not in digest
        assert "save_insight" not in digest

    def test_bash_newlines_replaced_with_spaces(self):
        results = [{"tool": "bash", "args": {"command": "echo hello\n echo world"}, "ok": True}]
        digest = build_work_state_digest(results)
        # newline is replaced with space (double space because \n echo → space + space)
        assert "\\n" not in digest
        assert "echo hello" in digest
        assert "echo world" in digest

    def test_bash_command_truncated(self):
        long_cmd = "echo " + "a" * 100
        results = [{"tool": "bash", "args": {"command": long_cmd}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "…" in digest

    def test_empty_command_in_bash(self):
        results = [{"tool": "bash", "args": {}, "ok": True}]
        digest = build_work_state_digest(results)
        assert "bash" in digest
