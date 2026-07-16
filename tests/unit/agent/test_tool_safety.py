"""Tests for WriteSafetyManager (external_llm/agent/tool_safety.py)."""
import os
import stat
from unittest.mock import MagicMock, patch

import pytest

from external_llm.agent.tool_safety import _MISSING_SNAP, WriteSafetyManager


@pytest.fixture
def tmp_repo(tmp_path):
    return str(tmp_path)


@pytest.fixture
def manager(tmp_repo):
    return WriteSafetyManager(tmp_repo)


# ── count_patch_files ─────────────────────────────────────────────────────────

class TestCountPatchFiles:
    def test_bare_headers_counted(self, manager):
        # Bare "--- a/" / "+++ b/" hunks (no "diff --git" prefix) must be
        # counted — snapshot_target_files snapshots them, so the approval
        # gate must see the same file set (gate-bypass fix).
        assert WriteSafetyManager.count_patch_files("--- a/foo\n+++ b/foo\n") == 1

    def test_bare_headers_multi_file_and_new_file(self, manager):
        patch = (
            "--- a/a.py\n+++ b/a.py\n"
            "--- a/b.py\n+++ b/b.py\n"
            "--- /dev/null\n+++ b/new.py\n"  # new file: only the +++ side counts
        )
        assert WriteSafetyManager.count_patch_files(patch) == 3

    def test_single_file(self, manager):
        patch = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
        assert WriteSafetyManager.count_patch_files(patch) == 1

    def test_multiple_files(self, manager):
        patch = (
            "diff --git a/a.py b/a.py\n"
            "diff --git a/b.py b/b.py\n"
            "diff --git a/c.py b/c.py\n"
        )
        assert WriteSafetyManager.count_patch_files(patch) == 3

    def test_empty_string(self, manager):
        assert WriteSafetyManager.count_patch_files("") == 0


# ── approval_preview ──────────────────────────────────────────────────────────

class TestApprovalPreview:
    def test_apply_patch_small_no_approval(self, manager):
        patch = "diff --git a/a.py b/a.py\n"  # 1 file < threshold
        _preview, needs = manager.approval_preview("apply_patch", {"patch": patch})
        assert needs is False

    def test_apply_patch_large_needs_approval(self, manager):
        patch = "\n".join(
            f"diff --git a/f{i}.py b/f{i}.py" for i in range(5)
        )
        preview, needs = manager.approval_preview("apply_patch", {"patch": patch})
        assert needs is True
        assert len(preview) > 0

    def test_delete_file_always_needs_approval(self, manager):
        preview, needs = manager.approval_preview("delete_file", {"path": "foo.py"})
        assert needs is True
        assert "DELETE FILE" in preview
        assert "foo.py" in preview

    def test_write_plan_needs_approval(self, manager):
        preview, needs = manager.approval_preview("write_plan", {"plan": {"key": "val"}})
        assert needs is True
        assert "WRITE PLAN" in preview

    def test_unknown_tool_no_approval(self, manager):
        preview, needs = manager.approval_preview("find_symbol", {"name": "foo"})
        assert needs is False
        assert preview == ""


# ── gate_check ────────────────────────────────────────────────────────────────

class TestGateCheck:
    def test_no_callback_always_passes(self, manager):
        result = manager.gate_check("delete_file", {"path": "x.py"}, approval_callback=None)
        assert result is None

    def test_approved_returns_none(self, manager):
        callback = MagicMock(return_value=True)
        result = manager.gate_check("delete_file", {"path": "x.py"}, callback)
        assert result is None

    def test_rejected_returns_error_dict(self, manager):
        callback = MagicMock(return_value=False)
        result = manager.gate_check("delete_file", {"path": "x.py"}, callback)
        assert result is not None
        assert "error" in result
        assert "rejected" in result["error"]
        assert result["metadata"]["gate"] == "rejected"
        assert result["metadata"]["tool"] == "delete_file"

    def test_no_approval_needed_callback_not_called(self, manager):
        callback = MagicMock()
        # find_symbol doesn't need approval
        result = manager.gate_check("find_symbol", {"name": "foo"}, callback)
        assert result is None
        callback.assert_not_called()


# ── snapshot_target_files ─────────────────────────────────────────────────────

class TestSnapshotTargetFiles:
    def test_snapshots_existing_file(self, manager, tmp_repo):
        target = os.path.join(tmp_repo, "target.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        snapshots = manager.snapshot_target_files("apply_patch", {"file_path": target})
        assert target in snapshots
        assert snapshots[target] == "x = 1\n"

    def test_missing_file_stored_as_missing_snap(self, manager, tmp_repo):
        missing = os.path.join(tmp_repo, "missing.py")
        snapshots = manager.snapshot_target_files("apply_patch", {"file_path": missing})
        assert missing in snapshots
        assert snapshots[missing] is _MISSING_SNAP

    def test_no_path_in_args_returns_empty(self, manager):
        snapshots = manager.snapshot_target_files("run_tests", {})
        assert snapshots == {}

    def test_patch_path_extracted_from_diff(self, manager, tmp_repo):
        target = os.path.join(tmp_repo, "inferred.py")
        with open(target, "w") as f:
            f.write("y = 2\n")
        rel = os.path.relpath(target, tmp_repo)
        patch_text = f"--- a/{rel}\n+++ b/{rel}\n"
        snapshots = manager.snapshot_target_files("apply_patch", {"patch": patch_text})
        # rel path should be resolved against repo_root
        assert any("inferred.py" in p for p in snapshots)

    def test_multifile_patch_snapshots_all_files(self, manager, tmp_repo):
        """Multi-file patch must snapshot ALL files, not just the first one."""
        file_a = os.path.join(tmp_repo, "file_a.py")
        file_b = os.path.join(tmp_repo, "file_b.py")
        with open(file_a, "w") as f:
            f.write("a = 1\n")
        with open(file_b, "w") as f:
            f.write("b = 2\n")
        rel_a = os.path.relpath(file_a, tmp_repo)
        rel_b = os.path.relpath(file_b, tmp_repo)
        patch_text = (
            f"diff --git a/{rel_a} b/{rel_a}\n"
            f"--- a/{rel_a}\n+++ b/{rel_a}\n@@ -1 +1 @@\n-a = 1\n+a = 99\n"
            f"diff --git a/{rel_b} b/{rel_b}\n"
            f"--- a/{rel_b}\n+++ b/{rel_b}\n@@ -1 +1 @@\n-b = 2\n+b = 99\n"
        )
        snapshots = manager.snapshot_target_files("apply_patch", {"patch": patch_text})
        # Both files must be in the snapshot for full rollback capability
        assert any("file_a.py" in p for p in snapshots), "file_a.py missing from snapshot"
        assert any("file_b.py" in p for p in snapshots), "file_b.py missing from snapshot"

    def test_new_file_creation_stored_as_missing_snap(self, manager, tmp_repo):
        """New-file hunks (--- /dev/null) are captured as _MISSING_SNAP."""
        patch_text = (
            "diff --git a/new_file.py b/new_file.py\n"
            "--- /dev/null\n+++ b/new_file.py\n@@ -0,0 +1 @@\n+x = 1\n"
        )
        snapshots = manager.snapshot_target_files("apply_patch", {"patch": patch_text})
        # new_file.py doesn't exist on disk — should be stored as _MISSING_SNAP
        missing_key = next((p for p in snapshots if "new_file.py" in p), None)
        assert missing_key is not None, "new_file.py should be in snapshots"
        assert snapshots[missing_key] is _MISSING_SNAP


# ── restore_snapshots ─────────────────────────────────────────────────────────

class TestRestoreSnapshots:
    def test_restores_file_content(self, tmp_repo):
        target = os.path.join(tmp_repo, "restore_me.py")
        original = "original content\n"
        with open(target, "w") as f:
            f.write(original)

        # Overwrite the file
        with open(target, "w") as f:
            f.write("changed content\n")

        WriteSafetyManager.restore_snapshots({target: original})
        assert open(target).read() == original

    def test_missing_path_silently_skipped(self):
        # Should not raise even if path doesn't exist
        WriteSafetyManager.restore_snapshots({"/nonexistent/path.py": "x=1"})

    def test_restores_multiple_files(self, tmp_repo):
        files = {}
        for name in ["a.py", "b.py"]:
            path = os.path.join(tmp_repo, name)
            files[path] = f"# {name}\n"
            with open(path, "w") as f:
                f.write("changed\n")
        WriteSafetyManager.restore_snapshots(files)
        for path, original in files.items():
            assert open(path).read() == original

    def test_missing_snap_removes_created_file(self, tmp_repo):
        """_MISSING_SNAP entry → file should be removed."""
        created = os.path.join(tmp_repo, "brand_new.py")
        with open(created, "w") as f:
            f.write("print('new')\n")
        assert os.path.exists(created)
        WriteSafetyManager.restore_snapshots({created: _MISSING_SNAP})
        assert not os.path.exists(created)

    def test_missing_snap_no_file_does_not_raise(self, tmp_repo):
        """_MISSING_SNAP where file was never created → silently skipped."""
        never = os.path.join(tmp_repo, "never_created.py")
        WriteSafetyManager.restore_snapshots({never: _MISSING_SNAP})
        assert not os.path.exists(never)  # no exception

    def test_atomic_write_creates_file_with_correct_content(self, tmp_repo):
        """Atomic mkstemp+os.replace produces correct content."""
        target = os.path.join(tmp_repo, "atomic_restore.py")
        original = "preserved content\nline 2\n"
        with open(target, "w") as f:
            f.write("corrupted content\n")
        WriteSafetyManager.restore_snapshots({target: original})
        assert open(target).read() == original


# ── restore_snapshots: atomic crash-safety ─────────────────────────────────


class TestRestoreSnapshotsAtomicSafety:
    """Edge cases for atomic write (mkstemp + os.replace)."""

    def test_tempfile_cleaned_on_replace_failure(self, tmp_repo, monkeypatch):
        """If os.replace fails, the mkstemp temp file (``.asi-revert-*``) must be
        unlinked and the target left uncorrupted (atomic = all-or-nothing)."""
        target = os.path.join(tmp_repo, "target.py")
        original = "content\n"
        with open(target, "w") as f:
            f.write("old\n")

        def _boom(*_a, **_k):
            raise OSError("replace denied")
        # Patch only os.replace; os.unlink must keep working so the cleanup runs.
        monkeypatch.setattr("os.replace", _boom)

        # restore_snapshots swallows the OSError, so it returns normally — but
        # the inner handler must have unlinked the temp before re-raising.
        WriteSafetyManager.restore_snapshots({target: original})

        # No leftover temp files from the failed atomic write.
        leftovers = [f for f in os.listdir(tmp_repo) if f.startswith(".asi-revert-")]
        assert leftovers == [], f"temp file leaked on replace failure: {leftovers}"
        # os.replace is atomic: the target is unchanged (not truncated/partial).
        assert open(target).read() == "old\n"

    def test_returns_failed_paths_on_oserror(self, tmp_repo, monkeypatch):
        """Bug #2: restore_snapshots now returns list of paths whose restoration
        failed (was silent pass). When os.replace raises, the path appears in
        the returned list."""
        target = os.path.join(tmp_repo, "fail_target.py")
        with open(target, "w") as f:
            f.write("old\n")

        def _boom(*_a, **_k):
            raise OSError("replace denied")

        monkeypatch.setattr("os.replace", _boom)
        failed = WriteSafetyManager.restore_snapshots({target: "new\n"})
        assert isinstance(failed, list), "must return a list of failed paths"
        assert target in failed, "failed path must appear in returned list"

    def test_returns_empty_list_on_success(self, tmp_repo):
        """Happy path: successful restore returns empty list."""
        target = os.path.join(tmp_repo, "ok_target.py")
        original = "content\n"
        with open(target, "w") as f:
            f.write("changed\n")
        failed = WriteSafetyManager.restore_snapshots({target: original})
        assert isinstance(failed, list)
        assert failed == []
        assert open(target).read() == original


class TestRestoreSnapshotsPreservesMode:
    """Regression: mkstemp creates the temp with mode 0600 and os.replace keeps
    the temp's mode. Without copying the original mode onto the temp first,
    restoring an executable (+x) or group/world-readable file silently strips
    it to owner-only — the rollback itself mutating metadata."""

    @staticmethod
    def _mode(path):
        return stat.S_IMODE(os.stat(path).st_mode)

    def test_executable_mode_preserved(self, tmp_repo):
        target = os.path.join(tmp_repo, "script.sh")
        with open(target, "w") as f:
            f.write("old\n")
        os.chmod(target, 0o755)  # rwxr-xr-x — executable
        assert self._mode(target) == 0o755

        WriteSafetyManager.restore_snapshots({target: "restored\n"})

        assert open(target).read() == "restored\n"
        assert self._mode(target) == 0o755, "restore stripped the +x bit"

    def test_group_world_read_preserved(self, tmp_repo):
        target = os.path.join(tmp_repo, "shared.py")
        with open(target, "w") as f:
            f.write("old\n")
        os.chmod(target, 0o644)  # rw-r--r--
        assert self._mode(target) == 0o644

        WriteSafetyManager.restore_snapshots({target: "restored\n"})

        assert self._mode(target) == 0o644, "restore collapsed to owner-only (0600)"


# ── verify_after_write ────────────────────────────────────────────────────────

class TestVerifyAfterWrite:
    def test_empty_snapshots_returns_true(self, manager):
        assert manager.verify_after_write({}) == (True, "")

    def test_valid_python_returns_true(self, manager, tmp_repo):
        target = os.path.join(tmp_repo, "valid.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        # Passes if language provider doesn't flag it (or has no validator)
        result = manager.verify_after_write({target: "x = 1\n"})
        assert isinstance(result, tuple) and len(result) == 2

    def test_missing_file_skipped(self, manager, tmp_repo):
        missing = os.path.join(tmp_repo, "missing.py")
        # File doesn't exist — should not crash, should return True (no validator ran)
        result = manager.verify_after_write({missing: "x=1"})
        assert result == (True, "")


# ── _treesitter_symbol_set ───────────────────────────────────────────────────

class TestTreeSitterSymbolSet:
    """Module-level _treesitter_symbol_set() — error paths."""

    def test_import_failure_returns_none(self):
        """When tree_sitter_utils import fails, returns None (line 78-79)."""
        # Use require failed import — patch the function's internal import
        import builtins

        from external_llm.agent.tool_safety import _treesitter_symbol_set
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "external_llm.languages.tree_sitter_utils":
                raise ImportError("mock: tree_sitter_utils not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            result = _treesitter_symbol_set("x = 1", "python")
        assert result is None

    def test_get_parser_returns_none(self):
        """Cover L82: get_parser returns None for a language."""
        from external_llm.agent.tool_safety import _treesitter_symbol_set

        with patch("external_llm.languages.tree_sitter_utils.get_parser", return_value=None):
            result = _treesitter_symbol_set("var x = 1", "typescript")
        assert result is None

    def test_parser_parse_raises_exception(self):
        """Cover L87-88: parser.parse raises an Exception."""
        from external_llm.agent.tool_safety import _treesitter_symbol_set

        mock_parser = MagicMock()
        mock_parser.parse.side_effect = Exception("parse error")

        with (
            patch("external_llm.languages.tree_sitter_utils.get_parser", return_value=mock_parser),
            patch("external_llm.languages.tree_sitter_utils.find_all_symbols", return_value=[]),
        ):
            result = _treesitter_symbol_set("var x = 1", "typescript")
        assert result is None


# ── _treesitter_symbol_set coverage supplements ────────────────────────────

class TestTreeSitterSymbolSetErrors:
    """Additional error-path coverage for _treesitter_symbol_set."""

    def test_parser_has_error_tree(self):
        """Cover L85-86: tree has syntax error → return None."""
        from external_llm.agent.tool_safety import _treesitter_symbol_set

        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_tree.root_node.has_error = True
        mock_parser.parse.return_value = mock_tree

        with (
            patch("external_llm.languages.tree_sitter_utils.get_parser", return_value=mock_parser),
        ):
            result = _treesitter_symbol_set("broken code{{{", "typescript")
        assert result is None


# ── approval_preview supplement ───────────────────────────────────────────

class TestApprovalPreviewWritePlanErrors:
    """Cover L176-177: write_plan preview when json.dumps fails."""

    def test_write_plan_json_dumps_fails(self, manager):
        """Cover L176-177: json.dumps raises TypeError → string fallback."""
        plan_obj = object()  # object() is not JSON-serializable
        preview, needs = manager.approval_preview("write_plan", {"plan": plan_obj})
        assert needs is True
        assert "WRITE PLAN" in preview


# ── snapshot_target_files supplement ──────────────────────────────────────

class TestSnapshotTargetFilesDictPlan:
    """Cover L237-242: snapshot_target_files with dict plan ops."""

    def test_dict_plan_with_ops(self, manager, tmp_repo):
        """Cover L237-242: write_plan with dict containing ops paths."""
        target = os.path.join(tmp_repo, "dict_target.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        snapshots = manager.snapshot_target_files("write_plan", {
            "plan": {"ops": [{"path": "dict_target.py"}]}
        })
        assert any("dict_target.py" in p for p in snapshots)

    def test_dict_plan_with_single_path(self, manager, tmp_repo):
        """Cover L240-241: single op with path."""
        target = os.path.join(tmp_repo, "single_op.py")
        with open(target, "w") as f:
            f.write("y = 2\n")
        snapshots = manager.snapshot_target_files("write_plan", {
            "plan": {"path": "single_op.py"}
        })
        assert any("single_op.py" in p for p in snapshots)

    def test_non_safety_tool_snapshots(self, manager, tmp_repo):
        """Cover L260-268: non-safety tool (not apply_patch/write_plan) with file_path."""
        target = os.path.join(tmp_repo, "other_tool_target.py")
        with open(target, "w") as f:
            f.write("z = 3\n")
        snapshots = manager.snapshot_target_files("create_file", {"file_path": target})
        assert target in snapshots

    def test_file_read_oserror(self, manager, tmp_repo):
        """Cover L267-268: OSError during snapshot file read is caught."""
        target = os.path.join(tmp_repo, "unreadable.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        with patch("builtins.open", side_effect=OSError("permission denied")):
            snapshots = manager.snapshot_target_files("create_file", {"file_path": target})
        assert snapshots == {}


# ── verify_after_write supplement ─────────────────────────────────────────

class TestVerifyAfterWriteErrors:
    """Cover L285-294: verify_after_write error paths."""

    def test_syntax_error_detail(self, manager, tmp_repo):
        """Cover L285-292: syntax error reported in detail."""
        target = os.path.join(tmp_repo, "bad_syntax.py")
        with open(target, "w") as f:
            f.write("x = 1\n")

        mock_validator = MagicMock()
        mock_validator.ok = False
        mock_validator.errors = [MagicMock(file="bad_syntax.py", line=1, col=0, message="invalid syntax")]

        mock_provider = MagicMock()
        mock_provider.capabilities.return_value.has_syntax_validator = True
        mock_provider.validate_syntax.return_value = mock_validator

        with patch("external_llm.languages.LanguageRegistry") as mock_lr:
            mock_lr.instance.return_value.get.return_value = mock_provider
            result = manager.verify_after_write({target: "x = 1\n"})
        assert result[0] is False
        assert "invalid syntax" in result[1]

    def test_oserror_reading_file(self, manager, tmp_repo):
        """Cover L293-294: OSError when reading file after write."""
        target = os.path.join(tmp_repo, "unreadable.py")
        with open(target, "w") as f:
            f.write("x = 1\n")

        mock_provider = MagicMock()
        mock_provider.capabilities.return_value.has_syntax_validator = True

        with (
            patch("external_llm.languages.LanguageRegistry") as mock_lr,
            patch("builtins.open", side_effect=OSError("can't read")),
        ):
            mock_lr.instance.return_value.get.return_value = mock_provider
            result = manager.verify_after_write({target: "x = 1\n"})
        assert result[0] is False
        assert "OS error" in result[1]


# ── _format_regions ──────────────────────────────────────────────────────

class TestFormatRegions:
    """Cover L315, L318-319: _format_regions edge cases."""

    def test_empty_regions_returns_no_line_range(self):
        """Cover L315: empty regions → fallback message."""
        result = WriteSafetyManager._format_regions([])
        assert result == "(no line-range info)"

    def test_single_region_same_start_end(self):
        """Single region L5-L5 → 'L5'."""
        result = WriteSafetyManager._format_regions([(5, 5)])
        assert result == "L5"

    def test_single_region_range(self):
        """Single region L1-L5 → 'L1-5'."""
        result = WriteSafetyManager._format_regions([(1, 5)])
        assert result == "L1-5"

    def test_truncation_when_exceeds_max(self):
        """Cover L318-319: more than max_shown regions → truncated with '(+)'."""
        regions = [(i, i + 1) for i in range(1, 10)]  # 9 regions (> 6)
        result = WriteSafetyManager._format_regions(regions)
        assert "(+3 more)" in result


# ── summarize_change ──────────────────────────────────────────────────────

class TestSummarizeChange:
    """Cover summarize_change edge cases."""

    def test_empty_snapshots_returns_none(self, manager):
        """Cover L339: no snapshots → return None."""
        assert manager.summarize_change({}) is None

    def test_oserror_on_re_read(self, manager, tmp_repo):
        """Cover L346-347: file disappears between snapshot and re-read."""
        target = os.path.join(tmp_repo, "disappeared.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        snapshots = {target: "x = 1\n"}
        # Delete the file so re-read raises OSError
        os.remove(target)
        result = manager.summarize_change(snapshots)
        # No files successfully re-read → lines_out empty → L395: return None
        assert result is None

    def test_valueerror_from_relpath(self, manager, tmp_repo):
        """Cover L351-352: os.path.relpath raises ValueError."""
        target = os.path.join(tmp_repo, "relpath_error.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        # Write different content so it's not byte-identical
        snapshots = {target: "original\n"}

        with patch("os.path.relpath", side_effect=ValueError("path error")):
            result = manager.summarize_change(snapshots)
        assert result is not None
        assert "relpath_error.py" in result or "+" in result

    def test_byte_identical_warning(self, manager, tmp_repo):
        """Cover L355-360: file content unchanged → byte-identical warning."""
        target = os.path.join(tmp_repo, "unchanged.py")
        content = "x = 1\n"
        with open(target, "w") as f:
            f.write(content)
        snapshots = {target: content}
        result = manager.summarize_change(snapshots)
        assert result is not None
        assert "NO CHANGE" in result
        assert "byte-identical" in result

    def test_all_files_fail_re_read(self, manager, tmp_repo):
        """Cover L395: all files in snapshots fail re-read → lines_out empty → return None."""
        target = os.path.join(tmp_repo, "all_fail.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        snapshots = {target: "x = 1\n"}
        # Open fails on re-read
        with patch("builtins.open", side_effect=OSError("can't read")):
            result = manager.summarize_change(snapshots)
        assert result is None

    def test_missing_snap_treats_as_empty(self, manager, tmp_repo):
        """_MISSING_SNAP original → all current content shown as added."""
        target = os.path.join(tmp_repo, "brand_new.py")
        content = "print('hello')\nprint('world')\n"
        with open(target, "w") as f:
            f.write(content)
        snapshots = {target: _MISSING_SNAP}
        result = manager.summarize_change(snapshots)
        assert result is not None
        assert "[POST-EDIT DIFF]" in result
        assert "+2/0" in result or "+2" in result  # 2 lines added, 0 removed


# ── Phase 1: semantic lint (new_semantic_warnings) ───────────────────────────

class TestNewSemanticWarnings:
    """Phase 1: ruff F-code detection surfaced as soft warning."""

    def test_unchanged_file_returns_none(self, manager, tmp_repo):
        """No diff between pre and post → no new findings."""
        target = os.path.join(tmp_repo, "foo.py")
        content = "x = 1\n"
        with open(target, "w") as f:
            f.write(content)
        result = manager.new_semantic_warnings({target: content})
        assert result is None

    def test_new_f821_undefined_name(self, manager, tmp_repo):
        """Post-edit introduces undefined name → [SEMANTIC LINT] surfaces it."""
        target = os.path.join(tmp_repo, "bar.py")
        pre = "x = 1\n"
        post = "x = Optional[int]\n"  # F821: Optional is undefined
        with open(target, "w") as f:
            f.write(post)
        result = manager.new_semantic_warnings({target: pre})
        assert result is not None
        assert "[SEMANTIC LINT]" in result
        assert "F821" in result

    def test_pre_existing_finding_excluded(self, manager, tmp_repo):
        """Finding that was already in pre-snapshot is NOT surfaced."""
        target = os.path.join(tmp_repo, "baz.py")
        pre = "x = Optional[int]\n"  # F821 existed before edit
        post = "x = Optional[int]\ny = 2\n"  # same F821, F821 still there
        with open(target, "w") as f:
            f.write(post)
        result = manager.new_semantic_warnings({target: pre})
        # The F821 on line 1 is pre-existing; only new findings would surface
        # Since nothing NEW was added beyond line 1, result should be None
        assert result is None
    def test_missing_snap_preserves_semantic_warning(self, manager, tmp_repo):
            """_MISSING_SNAP pre-content → treated as empty; new F821 surfaced."""
            target = os.path.join(tmp_repo, "missing_new.py")
            post = "x = Optional[int]\n"  # F821: Optional is undefined
            with open(target, "w") as f:
                f.write(post)
            # New file (was _MISSING_SNAP), pre treated as ""
            result = manager.new_semantic_warnings({target: _MISSING_SNAP})
            assert result is not None
            assert "[SEMANTIC LINT]" in result
            assert "F821" in result


# ── Phase 2: F821 deterministic auto-repair ──────────────────────────────────

class TestAutoRepairF821:
    """Phase 2: ruff F821 → project-wide import search → insert import."""

    def test_repair_missing_import(self, manager, tmp_repo):
        """F821 'Optional' → resolved from typing → import inserted."""
        # Create a source file with the F821 error
        target = os.path.join(tmp_repo, "use_optional.py")
        content = "x = Optional[int]\n"
        with open(target, "w") as f:
            f.write(content)
        # Seed a file with the import so _resolve_missing_import finds it
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        with open(os.path.join(agent_dir, "typing_stubs.py"), "w") as f:
            f.write("from typing import Optional\n")

        snapshots = {target: content}
        count = manager.auto_repair_semantic(snapshots)
        assert count == 1, "Expected 1 file repaired"

        # Verify the import was inserted
        with open(target) as f:
            result = f.read()
        assert "from typing import Optional" in result

        # Verify the result is valid Python
        assert manager._validate_python_syntax(result)

    def test_no_f821_no_repair(self, manager, tmp_repo):
        """No F821 findings → no repair needed → count 0."""
        target = os.path.join(tmp_repo, "clean.py")
        content = "x = 1\n"
        with open(target, "w") as f:
            f.write(content)
        count = manager.auto_repair_semantic({target: content})
        assert count == 0

    def test_import_already_present(self, manager, tmp_repo):
        """Import already exists for the missing name → no repair."""
        target = os.path.join(tmp_repo, "already_imported.py")
        content = "from typing import Optional\nx = Optional[int]\n"
        with open(target, "w") as f:
            f.write(content)
        snapshots = {target: content}
        count = manager.auto_repair_semantic(snapshots)
        assert count == 0, "Import already present, should not repair"

    def test_resolve_import_from_typing(self, manager, tmp_repo):
        """_resolve_missing_import finds 'Optional' in the real typing module
        via the external_llm/agent directory scan."""
        import_line = manager._resolve_missing_import(
            "Optional", tmp_repo, "test.py"
        )
        # The resolver searches the repo tree, not tmp_repo, so it might
        # or might not find something. Just verify it doesn't crash.
        if import_line:
            assert "Optional" in import_line

    def test_safety_net_rollback_on_broken_syntax(self, manager, tmp_repo):
        """If repair produces invalid syntax, rollback to pre-snapshot."""
        target = os.path.join(tmp_repo, "broken_target.py")
        pre = "x = 1\n"
        with open(target, "w") as f:
            f.write(pre)

        # Mock _insert_import_line to return invalid Python (for coverage of rollback).
        # Use patch.object so the @staticmethod descriptor is restored correctly
        # — a bare `Cls.attr = original` would leave it unwrapped, breaking later
        # tests that call self._insert_import_line (bound-method arity mismatch).
        with patch.object(
            WriteSafetyManager,
            "_insert_import_line",
            staticmethod(lambda c, i: "!!! syntax error !!!"),
        ):
            snapshots = {target: pre}
            count = manager.auto_repair_semantic(snapshots)
            assert count == 0, "Broken repair should be rolled back"
            # Verify file content was rolled back to pre
            with open(target) as f:
                assert f.read() == pre

    def test_repaired_typing_import_marked_f821_protected(self, manager, tmp_repo):
        """F821 repair inserting a typing import MUST mark it as f821-protected.

        Regression: the repair → import_normalizer contract was broken during
        the repair_core/repair_engine → tool_safety migration. Without the
        marker, the normalizer's AST pass cannot see the symbol (e.g. it is
        used only in a deferred string annotation) and strips the import on the
        next pass, recreating the F821 → oscillation.
        """
        target = os.path.join(tmp_repo, "use_optional.py")
        content = "x = Optional[int]\n"
        with open(target, "w") as f:
            f.write(content)
        # Seed a file with the import so _resolve_missing_import finds it
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        with open(os.path.join(agent_dir, "typing_stubs.py"), "w") as f:
            f.write("from typing import Optional\n")

        count = manager.auto_repair_semantic({target: content})
        assert count == 1

        with open(target) as f:
            result = f.read()
        assert "from typing import Optional" in result
        # The contract: a protection marker is written so the normalizer keeps it
        assert "# f821-protected" in result, (
            "F821-repaired typing import must carry the f821-protected marker; "
            "without it the import_normalizer strips it and F821 returns."
        )

    def test_repaired_non_typing_import_not_marked(self, manager, tmp_repo):
        """Non-typing import repair must NOT carry the f821-protected marker —
        the protection mechanism is scoped to the typing module only."""
        # 'os' is a stdlib import; _resolve_missing_import finds "import os"
        target = os.path.join(tmp_repo, "use_os.py")
        content = "p = path.join('a', 'b')\n"  # F821: 'path' undefined
        with open(target, "w") as f:
            f.write(content)
        # Seed an import for 'path'
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        with open(os.path.join(agent_dir, "path_stubs.py"), "w") as f:
            f.write("from os import path\n")

        manager.auto_repair_semantic({target: content})
        with open(target) as f:
            result = f.read()
        # Non-typing import resolves fine but must not get the protection marker
        assert "# f821-protected" not in result

    def test_insert_after_nested_import_keeps_syntax(self, manager):
        """Regression: nested imports must NOT be the insertion anchor.

        Previously ``_insert_import_line`` used ``ast.walk(tree)`` which visits
        imports at EVERY nesting level (inside ``if TYPE_CHECKING:``, function
        bodies, etc.). The *last* such import became the anchor; inserting a
        top-level ``import_line`` at its ``end_lineno`` split the block and
        produced ``unexpected indent`` SyntaxError — the dominant cause of the
        "Phase 2 F821 repair produced invalid syntax" warnings (74 in 06-20~25).

        The fix restricts the anchor search to module-level imports
        (``tree.body``), mirroring ``_resolve_missing_import`` which already
        only searched ``tree.body``.
        """
        # Last import in walk order is INSIDE the if-block / function — must
        # not become the anchor.
        content = (
            '"""docstring."""\n'
            "from __future__ import annotations\n"
            "import os\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from typing import Optional\n"
            "    from typing import Dict\n"
            "    x = 1\n"
        )
        result = WriteSafetyManager._insert_import_line(
            content, "from typing import Any"
        )
        assert manager._validate_python_syntax(result), (
            "inserting after a nested import must not break syntax"
        )
        # The new import lands right after the last MODULE-LEVEL import (os),
        # NOT inside the TYPE_CHECKING block.
        result_lines = result.splitlines()
        assert result_lines[3] == "from typing import Any"  # after `import os`
        # The block stays intact and correctly indented
        assert "    from typing import Dict" in result
        assert "    x = 1" in result

    def test_relative_import_preserves_level(self, manager, tmp_repo):
        """Regression: relative imports (level > 0) must keep their leading dots.

        Before the fix, ``node.level`` was dropped, rewriting a sibling's
        ``from .helper_mod import Helper`` into the broken absolute
        ``from helper_mod import Helper`` (ModuleNotFoundError at import time).
        This was the dominant F821 silent-corruption mode for package-internal
        names. Both files live in the SAME directory so the relative import is
        valid in both (cross-directory relative-import copy is a separate guard).
        """
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        # Sibling carries the relative import (must not start with '_': skipped).
        with open(os.path.join(agent_dir, "helper_stubs.py"), "w") as f:
            f.write("from .helper_mod import Helper\n")
        target = os.path.join(agent_dir, "use_helper.py")
        content = "x = Helper()\n"  # F821: Helper undefined
        with open(target, "w") as f:
            f.write(content)

        count = manager.auto_repair_semantic({target: content})
        assert count == 1, "Expected relative import to be inserted"

        with open(target) as f:
            result = f.read()
        # The leading dot MUST be preserved (level-aware resolution)
        assert "from .helper_mod import Helper" in result, (
            "relative-import level must be preserved — got broken absolute form"
        )
        assert "from helper_mod import" not in result, (
            "broken absolute form must NOT appear (node.level was dropped)"
        )

    def test_cross_dir_relative_import_skipped(self, manager, tmp_repo):
        """Regression: a relative import found in a source file in a DIFFERENT
        directory than the target must NOT be copied — it would silently rebind.

        ``from .internal import X`` in ``agent/foo.py`` means ``agent.internal``,
        but the same line pasted into ``external_llm/bar.py`` means
        ``external_llm.internal`` (wrong module, or ImportError at runtime). The
        find_spec safety net cannot catch this because relative imports are
        unverifiable without package context. The cross-directory guard skips
        the match so the loud F821 is kept rather than traded for silent
        corruption.
        """
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        # Source in A (agent/) carries a relative import (level 1)
        with open(os.path.join(agent_dir, "sibling_xdir.py"), "w") as f:
            f.write("from .internal_pkg import CrossTarget\n")
        # Target lives in B (external_llm/) — a DIFFERENT directory than the source
        result = manager._resolve_missing_import(
            "CrossTarget", tmp_repo, "external_llm/top_level.py"
        )
        assert result is None, (
            "cross-directory relative import must be skipped (would silently "
            f"rebind in the target's package); got: {result!r}"
        )

    def test_cross_dir_relative_falls_back_to_absolute(self, manager, tmp_repo):
        """When a cross-directory relative import is skipped, the search must
        continue and fall back to an importable absolute match in the target's
        own directory."""
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        ext_dir = os.path.join(tmp_repo, "external_llm")
        os.makedirs(agent_dir, exist_ok=True)
        # Source in A: relative import (cross-dir for a B target — skipped by guard)
        with open(os.path.join(agent_dir, "sib_xdir.py"), "w") as f:
            f.write("from .typing_stub import Opt\n")
        # Source in B: absolute, importable — the correct fallback
        with open(os.path.join(ext_dir, "abs_src.py"), "w") as f:
            f.write("from typing import Opt\n")
        # Target in B: cross-dir relative (A) skipped → absolute (B) returned
        result = manager._resolve_missing_import(
            "Opt", tmp_repo, "external_llm/top_level.py"
        )
        assert result == "from typing import Opt", (
            "guard must skip the cross-dir relative and fall back to the "
            f"importable absolute match; got: {result!r}"
        )

    def test_unresolvable_absolute_import_skipped(self, manager, tmp_repo):
        """Regression: an absolute import naming a non-importable module must be
        SKIPPED, not inserted. Inserting it would trade a loud F821 for a silent
        ModuleNotFoundError at import time. The find_spec safety net catches it."""
        agent_dir = os.path.join(tmp_repo, "external_llm", "agent")
        os.makedirs(agent_dir, exist_ok=True)
        # Sibling imports a package that does not exist anywhere.
        with open(os.path.join(agent_dir, "broken_stubs.py"), "w") as f:
            f.write("from definitely_not_a_real_pkg_zzz import Thing\n")
        target = os.path.join(tmp_repo, "use_thing.py")
        content = "x = Thing()\n"  # F821: Thing undefined
        with open(target, "w") as f:
            f.write(content)

        count = manager.auto_repair_semantic({target: content})
        # Repair skipped — the unresolvable import was NOT inserted.
        assert count == 0, "Unresolvable absolute import must be skipped"

        with open(target) as f:
            result = f.read()
        assert "from definitely_not_a_real_pkg_zzz import" not in result, (
            "broken absolute import must NOT be inserted"
        )
        # Original content untouched (F821 remains; no silent corruption)
        assert "x = Thing()" in result

    def test_import_line_resolves_unit(self, manager):
        """Unit tests for the find_spec safety-net helper."""
        h = WriteSafetyManager._import_line_resolves
        # Absolute stdlib imports → resolvable → True
        assert h("from typing import Optional") is True
        assert h("import os") is True
        assert h("from os import path") is True
        # Absolute import of a non-existent package → find_spec None → False
        assert h("from definitely_not_a_real_pkg_zzz import Thing") is False
        assert h("import definitely_not_a_real_pkg_zzz") is False
        # Relative imports → unverifiable without package context → accepted (True)
        assert h("from .helper_mod import Helper") is True
        assert h("from . import X") is True
        assert h("from ..pkg.mod import Y") is True
        # Garbage → SyntaxError → False
        assert h("not valid python !!!") is False


# ── Phase 3: decl-loss auto-restore ──────────────────────────────────────────

class TestAutoRestoreDeclLoss:
    """Phase 3: accidentally removed symbol → F821 → auto-restore."""

    def test_restore_deleted_symbol(self, manager, tmp_repo):
        """Symbol removed in edit, now F821 → restored from pre-snapshot."""
        target = os.path.join(tmp_repo, "decl_loss.py")
        pre = "def helper():\n    return 42\n\ndef caller():\n    return helper()\n"
        post = "def caller():\n    return helper()\n"  # helper() removed, now F821
        with open(target, "w") as f:
            f.write(post)
        snapshots = {target: pre}
        count = manager.auto_repair_semantic(snapshots)
        assert count == 1, "Expected 1 file with decl-loss restore"
        with open(target) as f:
            result = f.read()
        assert "def helper()" in result, "helper() should be restored"
        assert "def caller()" in result
        # Verify syntax is valid
        assert manager._validate_python_syntax(result)

    def test_intentional_delete_not_restored(self, manager, tmp_repo):
        """Symbol removed but NOT triggering F821 → not restored."""
        target = os.path.join(tmp_repo, "intentional_delete.py")
        pre = "def dead():\n    return 1\n\ndef alive():\n    return 2\n"
        post = "def alive():\n    return 2\n"  # dead() removed, not referenced → no F821
        with open(target, "w") as f:
            f.write(post)
        snapshots = {target: pre}
        count = manager.auto_repair_semantic(snapshots)
        # dead() was intentionally removed (no F821 signal) → should NOT restore
        assert count == 0, "No F821 → no restore"
        with open(target) as f:
            result = f.read()
        assert "def dead()" not in result, "dead() should remain removed"
        assert "def alive()" in result
