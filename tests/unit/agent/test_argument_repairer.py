"""Tests for ArgumentRepairer — file-argument alias normalization.

Regression coverage for the read_file `'path' is required` failure: the LLM
sends `file_path` (carried over from the file_path-named tools like read_symbol)
while read_file's schema names the argument `path`. The repairer must rename it
before dispatch. Both directions of the path/file_path mix-up are covered.
"""

from external_llm.agent.argument_repairer import ArgumentRepairer


def _repaired(tool, args):
    return ArgumentRepairer().repair(tool, args)


class TestPathFamilyAliases:
    def test_read_file_file_path_to_path(self):
        r = _repaired("read_file", {"file_path": "external_llm/agent/context_manager.py"})
        assert r.repaired
        assert r.repaired_args == {"path": "external_llm/agent/context_manager.py"}
        assert "file_path" not in r.repaired_args
        assert r.repairs_applied == ["file_path → path"]

    def test_read_file_filepath_to_path(self):
        r = _repaired("read_file", {"filepath": "a.py"})
        assert r.repaired_args == {"path": "a.py"}

    def test_read_file_target_file_to_path(self):
        r = _repaired("read_file", {"target_file": "a.py"})
        assert r.repaired_args == {"path": "a.py"}

    def test_edit_create_grep_run_lint_aliases(self):
        for tool in ("edit_file", "create_file", "grep", "run_lint"):
            r = _repaired(tool, {"file_path": "a.py"})
            assert r.repaired_args.get("path") == "a.py", tool
            assert "file_path" not in r.repaired_args, tool

    def test_canonical_path_present_wins(self):
        # When canonical `path` is already present, the alias must NOT clobber it.
        r = _repaired("read_file", {"path": "real.py", "file_path": "stray.py"})
        assert not r.repaired
        assert r.repaired_args == {"path": "real.py", "file_path": "stray.py"}

    def test_other_args_preserved(self):
        r = _repaired("read_file", {"file_path": "a.py", "start_line": 10, "end_line": 20})
        assert r.repaired_args == {"path": "a.py", "start_line": 10, "end_line": 20}


class TestFilePathFamilyAliases:
    def test_path_to_file_path_for_file_path_tools(self):
        for tool in (
            "modify_symbol", "edit_ast", "edit_text",
            "read_symbol", "analyze_change_impact",
        ):
            r = _repaired(tool, {"path": "a.py"})
            assert r.repaired_args.get("file_path") == "a.py", tool
            assert "path" not in r.repaired_args, tool


class TestCustomAliases:
    """Coverage for custom_aliases parameter in __init__."""

    def test_custom_alias_added(self):
        """Custom alias for an un-mapped tool."""
        r = ArgumentRepairer(custom_aliases={
            "new_tool": {"old_arg": "new_arg"},
        }).repair("new_tool", {"old_arg": "val"})
        assert r.repaired
        assert r.repaired_args == {"new_arg": "val"}
        assert "old_arg" not in r.repaired_args

    def test_custom_alias_overrides_default(self):
        """Custom alias for an existing tool merges with defaults."""
        r = ArgumentRepairer(custom_aliases={
            "read_file": {"filename": "path"},
        }).repair("read_file", {"filename": "test.py"})
        assert r.repaired
        assert r.repaired_args == {"path": "test.py"}
        assert "filename" not in r.repaired_args

    def test_custom_alias_does_not_break_defaults(self):
        """Default aliases still work when custom_aliases is provided."""
        r = ArgumentRepairer(custom_aliases={
            "read_file": {"filename": "path"},
        }).repair("read_file", {"file_path": "a.py"})
        assert r.repaired
        assert r.repaired_args == {"path": "a.py"}

    def test_custom_alias_empty_dict(self):
        """Empty custom_aliases preserves defaults unchanged."""
        r = ArgumentRepairer(custom_aliases={}).repair("read_file", {"file_path": "a.py"})
        assert r.repaired
        assert r.repaired_args == {"path": "a.py"}

    def test_none_custom_aliases(self):
        """None custom_aliases preserves defaults unchanged."""
        r = ArgumentRepairer(custom_aliases=None).repair("read_file", {"file_path": "a.py"})
        assert r.repaired
        assert r.repaired_args == {"path": "a.py"}


class TestUnaffected:
    def test_unknown_tool_unchanged(self):
        r = _repaired("bash", {"command": "ls"})
        assert not r.repaired
        assert r.repaired_args == {"command": "ls"}

    def test_apply_patch_aliases_still_work(self):
        r = _repaired("apply_patch", {"diff": "..."})
        assert r.repaired_args == {"patch": "..."}
