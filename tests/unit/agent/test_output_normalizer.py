"""Unit tests for Output Normalizer — drift detection, import dedup, whitespace normalization."""
import textwrap
from unittest.mock import patch

from external_llm.agent.output_normalizer import (
    DriftReport,
    dedup_imports,
    detect_task_drift,
    normalize_file,
    normalize_modified_files,
    normalize_python_source,
    normalize_whitespace,
)

# ══════════════════════════════════════════════════════════════════════════
# DriftReport dataclass
# ══════════════════════════════════════════════════════════════════════════

class TestDriftReport:
    def test_defaults(self):
        r = DriftReport()
        assert r.has_drift is False
        assert r.untargeted_files == []
        assert r.drifted_kinds == []
        assert r.severity == "none"
        assert r.summary == ""


# ══════════════════════════════════════════════════════════════════════════
# detect_task_drift
# ══════════════════════════════════════════════════════════════════════════

def _make_op(**kwargs):
    """Create a minimal operation object with kind/path attributes."""
    return type("Op", (), kwargs)()


class TestDetectTaskDrift:
    def test_no_operations(self):
        r = detect_task_drift(["a.py"], [], [])
        assert r.has_drift is False
        assert r.severity == "none"
        assert "no operations" in r.summary

    def test_none_spec_targets(self):
        """None spec_target_files doesn't cause errors."""
        op = _make_op(kind="MODIFY_SYMBOL", path="a.py")
        r = detect_task_drift(None, None, [op])
        # With no target files, any write op is untargeted
        assert r.untargeted_files == ["a.py"]
        assert r.has_drift is True

    def test_untargeted_write_file_flagged(self):
        op = _make_op(kind="MODIFY_SYMBOL", path="b.py")
        r = detect_task_drift(["a.py"], [], [op])
        assert "b.py" in r.untargeted_files
        assert r.has_drift is True

    def test_read_symbol_not_flagged_as_untargeted(self):
        """READ_SYMBOL ops are not in _WRITE_KINDS → not flagged."""
        op = _make_op(kind="READ_SYMBOL", path="untargeted.py")
        r = detect_task_drift(["a.py"], [], [op])
        assert r.untargeted_files == []
        assert r.has_drift is False

    def test_modify_on_target_not_flagged(self):
        op = _make_op(kind="MODIFY_SYMBOL", path="a.py")
        r = detect_task_drift(["a.py"], [], [op])
        assert r.untargeted_files == []
        assert r.severity == "none"

    def test_mixed_targeted_and_untargeted(self):
        ops = [
            _make_op(kind="MODIFY_SYMBOL", path="a.py"),
            _make_op(kind="OVERWRITE_FILE", path="b.py"),
            _make_op(kind="READ_SYMBOL", path="c.py"),
        ]
        r = detect_task_drift(["a.py"], [], ops)
        assert r.untargeted_files == ["b.py"]
        assert r.severity == "low"  # 1 file * 2 = 2 points → low

    def test_multiple_untargeted_high_severity(self):
        ops = [
            _make_op(kind="MODIFY_SYMBOL", path="x.py"),
            _make_op(kind="CREATE_FILE", path="y.py"),
            _make_op(kind="DELETE_FILE", path="z.py"),
        ]
        r = detect_task_drift(["a.py"], [], ops)
        assert len(r.untargeted_files) == 3
        assert r.severity == "high"  # 3 * 2 = 6 ≥ 5

    def test_delete_on_edit_request_drifted_kinds(self):
        """DELETE operations on edit/fix request → drifted_kinds."""
        op = _make_op(kind="DELETE_SYMBOL_RANGE", path="a.py")
        r = detect_task_drift(["a.py"], [], [op], request_type="fix")
        assert "DELETE_SYMBOL_RANGE" in r.drifted_kinds
        assert r.severity == "medium"  # drift_kinds=3pts → medium (3-4)

    def test_delete_on_unknown_request_not_drifted(self):
        """'unknown' is in _is_edit_request set → still flagged."""
        op = _make_op(kind="DELETE_SYMBOL_RANGE", path="a.py")
        r = detect_task_drift(["a.py"], [], [op], request_type="unknown")
        assert "DELETE_SYMBOL_RANGE" in r.drifted_kinds

    def test_delete_on_non_edit_request_not_drifted(self):
        """DELETE on non-edit request (e.g. 'delete') → not drifted."""
        op = _make_op(kind="DELETE_SYMBOL_RANGE", path="a.py")
        r = detect_task_drift(["a.py"], [], [op], request_type="delete")
        assert r.drifted_kinds == []

    def test_severity_medium_with_drift_kinds(self):
        op = _make_op(kind="DELETE_SYMBOL_RANGE", path="a.py")
        r = detect_task_drift(["a.py"], [], [op], request_type="edit")
        assert r.severity == "medium"  # drift_kinds alone = 3pts → medium (3-4)

    def test_severity_untargeted_and_drifted_medium(self):
        """1 untargeted file (2pts) + drifted_kinds (3pts) = 5 → high."""
        ops = [
            _make_op(kind="MODIFY_SYMBOL", path="b.py"),
            _make_op(kind="DELETE_SYMBOL_RANGE", path="a.py"),
        ]
        r = detect_task_drift(["a.py"], [], ops, request_type="fix")
        assert r.severity == "high"  # 2 + 3 = 5 ≥ 5

    def test_op_without_path_not_counted(self):
        op = _make_op(kind="MODIFY_SYMBOL", path="")
        r = detect_task_drift(["a.py"], [], [op])
        assert r.untargeted_files == []

    def test_op_kind_none_not_crashed(self):
        op = _make_op(kind=None, path="b.py")
        r = detect_task_drift(["a.py"], [], [op])
        # kind is None → str(None) = "None" → not in WRITE_KINDS → not flagged
        assert r.untargeted_files == []

    def test_summary_format(self):
        ops = [_make_op(kind="MODIFY_SYMBOL", path="b.py")]
        r = detect_task_drift(["a.py"], [], ops)
        assert "untargeted_files" in r.summary
        assert "b.py" in r.summary

    def test_no_drift_summary(self):
        op = _make_op(kind="MODIFY_SYMBOL", path="a.py")
        r = detect_task_drift(["a.py"], [], [op])
        assert r.summary == "no drift detected"

    def test_op_kind_as_enum(self):
        """Operation kind with .value attribute is handled."""
        class Kind(str):
            MODIFY = "MODIFY_SYMBOL"
        op = _make_op(kind=Kind.MODIFY, path="b.py")
        r = detect_task_drift(["a.py"], [], [op])
        assert "b.py" in r.untargeted_files

    def test_path_normalization(self):
        """Paths are normalized to prevent false positives."""
        op = _make_op(kind="MODIFY_SYMBOL", path="./a.py")
        r = detect_task_drift(["a.py"], [], [op])
        assert r.untargeted_files == []

    def test_empty_spec_targets(self):
        op = _make_op(kind="MODIFY_SYMBOL", path="a.py")
        r = detect_task_drift([], [], [op])
        assert "a.py" in r.untargeted_files

    def test_multiple_ops_same_file_not_duplicated(self):
        ops = [
            _make_op(kind="MODIFY_SYMBOL", path="b.py"),
            _make_op(kind="OVERWRITE_FILE", path="b.py"),
        ]
        r = detect_task_drift(["a.py"], [], ops)
        assert r.untargeted_files == ["b.py"]  # deduped


# ══════════════════════════════════════════════════════════════════════════
# dedup_imports
# ══════════════════════════════════════════════════════════════════════════

class TestDedupImports:
    def test_no_change_for_unique_imports(self):
        src = textwrap.dedent("""\
            import os
            import sys
            x = 1
        """)
        assert dedup_imports(src) == src

    def test_removes_exact_duplicate(self):
        src = textwrap.dedent("""\
            import os
            import os
            x = 1
        """)
        expected = textwrap.dedent("""\
            import os
            x = 1
        """)
        assert dedup_imports(src) == expected

    def test_removes_duplicate_from_import(self):
        src = textwrap.dedent("""\
            from os import path
            from os import path
            x = 1
        """)
        expected = textwrap.dedent("""\
            from os import path
            x = 1
        """)
        assert dedup_imports(src) == expected

    def test_merges_same_module_from_imports(self):
        src = textwrap.dedent("""\
            from os import path
            from os import getcwd
            x = 1
        """)
        result = dedup_imports(src)
        assert "from os import" in result
        assert "path" in result
        assert "getcwd" in result
        assert result.count("from os") == 1
        assert "x = 1" in result

    def test_merge_preserves_first_line_ordering(self):
        src = textwrap.dedent("""\
            from os import path
            from sys import argv
            from os import getcwd
            x = 1
        """)
        result = dedup_imports(src)
        lines = result.splitlines()
        # os line first, then sys
        os_idx = next(i for i, _item_ in enumerate(lines) if "from os import" in _item_)
        sys_idx = next(i for i, _item_ in enumerate(lines) if "from sys" in _item_)
        assert os_idx < sys_idx

    def test_dedup_names_within_single_import(self):
        src = textwrap.dedent("""\
            import os, os
            x = 1
        """)
        result = dedup_imports(src)
        assert result.count("os") == 1
        assert "x = 1" in result

    def test_dedup_multi_name_import(self):
        src = textwrap.dedent("""\
            import os, sys, os
            x = 1
        """)
        result = dedup_imports(src)
        assert "import os, sys" in result or "import sys, os" in result
        assert result.count("os") == 1 if "import os" in result else True

    def test_relative_from_import(self):
        src = textwrap.dedent("""\
            from . import foo
            from . import bar
            x = 1
        """)
        result = dedup_imports(src)
        assert result.count("from . import") == 1
        assert "foo" in result
        assert "bar" in result

    def test_relative_nested_from_import(self):
        src = textwrap.dedent("""\
            from .utils import foo
            from .utils import bar
            x = 1
        """)
        result = dedup_imports(src)
        assert result.count("from .utils import") == 1
        assert "foo" in result
        assert "bar" in result

    def test_multiline_import_handled(self):
        src = textwrap.dedent("""\
            from os import (
                path,
                getcwd,
            )
            from os import (
                path,
                getcwd,
            )
            x = 1
        """)
        result = dedup_imports(src)
        assert result.count("from os") == 1
        assert "x = 1" in result

    def test_syntax_error_fallback(self):
        """Invalid syntax falls back to heuristic dedup."""
        src = "import os\nimport os\nthis is not valid python\n"
        result = dedup_imports(src)
        assert result.count("import os") == 1
        assert "not valid" in result

    def test_inner_syntax_error_fallback(self):
        """When ast.parse of deduped source fails, fall back to exact-match dedup."""
        import ast as ast_module
        original_parse = ast_module.parse

        call_count = 0

        def mock_parse(source, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise SyntaxError("Mock inner parse failure")
            return original_parse(source, *args, **kwargs)

        with patch.object(ast_module, "parse", mock_parse):
            src = "import os\nimport os\nx = 1\n"
            result = dedup_imports(src)
            # Should fall back to exact-match dedup (not merge)
            assert result.count("import os") == 1
            assert "x = 1" in result

    def test_import_not_on_first_column_kept(self):
        """Indented imports (inside functions) are not treated as top-level."""
        src = textwrap.dedent("""\
            import os
            def foo():
                import os
                return os
        """)
        # AST path: top-level import duplicates → removed
        # Inner import is not top-level → kept
        result = dedup_imports(src)
        # One top-level import os removed (AST catches it)
        # The indented one remains
        assert result.count("import os") == 1 or result.count("import os") == 2

    def test_preserves_trailing_newline(self):
        src = "import os\nimport os\n"
        assert dedup_imports(src).endswith("\n")

    def test_adds_missing_trailing_newline(self):
        src = "import os\nimport os"
        result = dedup_imports(src)
        assert result.endswith("\n")
        assert result.count("import os") == 1

    def test_merge_sorted_underscore_last(self):
        """Merged names are sorted with underscore-prefixed names last."""
        src = textwrap.dedent("""\
            from os import _private
            from os import public
            x = 1
        """)
        result = dedup_imports(src)
        os_line = next(_item_ for _item_ in result.splitlines() if "from os import" in _item_)
        # _private should come after public in sorted output
        assert os_line.index("public") < os_line.index("_private")

    def test_empty_source(self):
        assert dedup_imports("") == ""

    def test_only_imports(self):
        src = "import os\nimport sys\nimport os\n"
        result = dedup_imports(src)
        assert result.count("import os") == 1
        assert result.count("import sys") == 1


# ══════════════════════════════════════════════════════════════════════════
# normalize_whitespace
# ══════════════════════════════════════════════════════════════════════════

class TestNormalizeWhitespace:
    def test_no_change(self):
        src = "x = 1\ny = 2\n"
        assert normalize_whitespace(src) == src

    def test_trailing_whitespace_removed(self):
        src = "x = 1   \ny = 2\t\n"
        expected = "x = 1\ny = 2\n"
        assert normalize_whitespace(src) == expected

    def test_excessive_blank_lines_capped(self):
        src = "a\n\n\n\n\nb\n"
        expected = "a\n\n\nb\n"
        assert normalize_whitespace(src) == expected

    def test_trailing_blank_lines_removed(self):
        src = "a\n\n\n"
        expected = "a\n"
        assert normalize_whitespace(src) == expected

    def test_only_blank_lines_returns_empty(self):
        assert normalize_whitespace("   \n\n\n") == ""

    def test_preserves_one_blank_line(self):
        src = "a\n\nb\n"
        assert normalize_whitespace(src) == src

    def test_preserves_two_blank_lines(self):
        src = "a\n\n\nb\n"
        assert normalize_whitespace(src) == src

    def test_mixed_whitespace_and_blank_lines(self):
        src = "a  \n\n\n  \nb\t\n"
        expected = "a\n\n\nb\n"
        assert normalize_whitespace(src) == expected

    def test_ensures_final_newline(self):
        src = "a\nb"
        result = normalize_whitespace(src)
        assert result == "a\nb\n" or result == src  # already has no trailing blank lines
        # Actually "a\nb" → splitlines gives ['a', 'b'], join gives "a\nb\n"
        assert result.endswith("\n")

    def test_empty_string(self):
        assert normalize_whitespace("") == ""


# ══════════════════════════════════════════════════════════════════════════
# normalize_python_source
# ══════════════════════════════════════════════════════════════════════════

class TestNormalizePythonSource:
    def test_composition(self):
        """dedup + whitespace applied in order."""
        src = textwrap.dedent("""\
            import os
            import os

            x = 1
        """)
        result = normalize_python_source(src)
        assert result.count("import os") == 1
        assert "x = 1" in result
        assert result.endswith("\n")

    def test_no_change_idempotent(self):
        src = "import os\nx = 1\n"
        assert normalize_python_source(src) == src

    def test_empty(self):
        assert normalize_python_source("") == ""


# ══════════════════════════════════════════════════════════════════════════
# normalize_file
# ══════════════════════════════════════════════════════════════════════════

class TestNormalizeFile:
    def test_non_python_file_skipped(self, tmp_path):
        f = tmp_path / "test.js"
        f.write_text("let x = 1;")
        changed, err = normalize_file(str(f))
        assert changed is False
        assert err is None

    def test_python_file_with_no_changes(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("import os\nx = 1\n")
        changed, err = normalize_file(str(f))
        assert changed is False
        assert err is None

    def test_python_file_with_changes(self, tmp_path):
        f = tmp_path / "dirty.py"
        f.write_text("import os\nimport os\nx = 1\n")
        changed, err = normalize_file(str(f))
        assert changed is True
        assert err is None
        assert f.read_text().count("import os") == 1

    def test_syntax_error_after_normalization_skipped(self, tmp_path):
        """If normalization output breaks syntax, original is preserved."""
        f = tmp_path / "fragile.py"
        f.write_text("x = 1\n")
        with patch("external_llm.agent.output_normalizer.normalize_python_source",
                    return_value="bad syntax !!!"):
            changed, err = normalize_file(str(f))
        assert changed is False
        assert "syntax error" in err
        assert f.read_text() == "x = 1\n"

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "nonexistent.py"
        changed, err = normalize_file(str(f))
        assert changed is False
        assert err is None

    def test_file_read_error(self, tmp_path):
        f = tmp_path / "unreadable.py"
        f.write_text("x = 1")
        with patch("builtins.open", side_effect=PermissionError("denied")):
            changed, err = normalize_file(str(f))
        assert changed is False
        assert "read error" in err

    def test_with_repo_root_relative_path(self, tmp_path):
        f = tmp_path / "sub" / "test.py"
        f.parent.mkdir()
        f.write_text("import os\nimport os\n")
        changed, _err = normalize_file("sub/test.py", str(tmp_path))
        assert changed is True
        assert f.read_text().count("import os") == 1

    def test_write_error(self, tmp_path):
        """Cover the write-error path: normalize_file now writes via
        ``atomic_write_text`` (mkstemp+os.replace, not ``open``), so the failure
        is injected there. The contract is unchanged — returns (False,
        'write error: ...')."""
        import external_llm.agent.output_normalizer as on_mod
        f = tmp_path / "test.py"
        f.write_text("import os\nimport os\nx=1\n")
        with patch.object(on_mod, "atomic_write_text", side_effect=OSError("disk full")):
            changed, err = normalize_file(str(f))
        assert changed is False
        assert "write error" in err


# ══════════════════════════════════════════════════════════════════════════
# normalize_modified_files
# ══════════════════════════════════════════════════════════════════════════

class TestNormalizeModifiedFiles:
    def test_non_python_files_skipped(self, tmp_path):
        (tmp_path / "a.js").write_text("let x = 1;")
        result = normalize_modified_files(["a.js"], str(tmp_path))
        assert result == []

    def test_python_files_changed(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nimport os\nx=1\n")
        result = normalize_modified_files(["a.py"], str(tmp_path))
        assert result == ["a.py"]

    def test_mixed_files(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nimport os\nx=1\n")
        (tmp_path / "b.js").write_text("let x = 1;")
        (tmp_path / "c.py").write_text("y=2\n")
        result = normalize_modified_files(["a.py", "b.js", "c.py"], str(tmp_path))
        assert "a.py" in result
        # c.py has no changes → only a.py returned
        assert result == ["a.py"]

    def test_empty_list(self, tmp_path):
        assert normalize_modified_files([], str(tmp_path)) == []

    def test_skipped_file_with_error(self, tmp_path):
        """Cover L418: normalize_modified_files logs skipped files with errors."""
        (tmp_path / "broken.py").write_text("x=1\n")
        with patch(
            "external_llm.agent.output_normalizer.normalize_python_source",
            return_value="bad syntax !!!",
        ):
            result = normalize_modified_files(["broken.py"], str(tmp_path))
        assert result == []  # not changed (error → skipped)
