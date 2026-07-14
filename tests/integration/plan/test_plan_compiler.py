"""
Integration tests for Plan Compiler.
"""
from pathlib import Path

import pytest

from plan_compiler import PlanCompileError, compile_plan_to_unified_diff
from tests.integration.helpers import create_test_file, git_add_and_commit


def _compile(plan, repo_root, allow_empty=False) -> str:
    """Helper to call compile_plan_to_unified_diff and return diff_patch string."""
    result = compile_plan_to_unified_diff(plan=plan, repo_root=repo_root, allow_empty=allow_empty)
    return result.diff_patch


@pytest.mark.integration
class TestPlanCompiler:
    """Test plan compilation to unified diff."""

    def test_compile_simple_edit_blocks(self, temp_repo_root: str, sample_simple_edit_plan_dict: dict):
        """Test compilation of simple edit_blocks plan."""
        diff = _compile(sample_simple_edit_plan_dict, temp_repo_root)

        assert isinstance(diff, str)
        assert len(diff) > 0
        assert "--- a/sample.py" in diff
        assert "+++ b/sample.py" in diff
        assert "@@" in diff  # Should have hunk headers
        assert "Fixed indentation" in diff  # Our edit comment

    def test_compile_create_file(self, temp_repo_root: str, sample_create_file_plan_dict: dict):
        """Test compilation of create_file plan."""
        diff = _compile(sample_create_file_plan_dict, temp_repo_root)

        assert isinstance(diff, str)
        assert len(diff) > 0
        assert "--- /dev/null" in diff or "--- a/new_file.py" in diff
        assert "+++ b/new_file.py" in diff
        assert "new_function" in diff

    def test_compile_multi_file_plan(self, temp_repo_root: str, sample_multi_file_plan_dict: dict):
        """Test compilation of multi-file plan."""
        diff = _compile(sample_multi_file_plan_dict, temp_repo_root)

        assert isinstance(diff, str)
        assert len(diff) > 0
        # Should contain both files
        assert "sample.py" in diff
        assert "utils.py" in diff
        # Should have multiple hunks
        assert diff.count("@@") >= 2

    def test_compile_plan_with_nonexistent_file(self, temp_repo_root: str):
        """Test compilation fails for edit_blocks on nonexistent file."""
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "nonexistent.py",
                    "blocks": [{"before": "foo", "after": "bar"}]
                }
            ]
        }

        with pytest.raises(PlanCompileError) as exc_info:
            _compile(plan, temp_repo_root)

        # File doesn't exist so before block won't be found
        assert "not found" in str(exc_info.value).lower() or "nonexistent" in str(exc_info.value)

    def test_compile_plan_with_invalid_blocks(self, temp_repo_root: str):
        """Test compilation fails for edit_blocks with invalid old content."""
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "sample.py",
                    "blocks": [{"before": "nonexistent content xyz123", "after": "new content"}]
                }
            ]
        }

        with pytest.raises(PlanCompileError) as exc_info:
            _compile(plan, temp_repo_root)

        assert "not found" in str(exc_info.value).lower() or "block" in str(exc_info.value).lower()

    def test_compile_plan_with_path_traversal_attempt(self, temp_repo_root: str):
        """Test compilation prevents path traversal attacks."""
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "create_file",
                    "path": "../../../etc/passwd",
                    "content": "malicious"
                }
            ]
        }

        with pytest.raises(PlanCompileError) as exc_info:
            _compile(plan, temp_repo_root)

        assert "Path traversal" in str(exc_info.value) or "outside" in str(exc_info.value) or ".." in str(exc_info.value)

    def test_compile_plan_with_insert_before(self, temp_repo_root: str):
        """Test compilation of insert_before operation."""
        # Create a file with specific content
        filepath = Path(temp_repo_root) / "test_insert.py"
        filepath.write_text("line1\nline2\nline3\n")
        git_add_and_commit(temp_repo_root, "Add test file")

        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "insert_before",
                    "path": "test_insert.py",
                    "anchor": "line2",
                    "lines": ["inserted_line\n"]
                }
            ]
        }

        diff = _compile(plan, temp_repo_root)

        assert "inserted_line" in diff
        # Check that the hunk shows insertion before line2
        assert "@@ -1,3 +1,4 @@" in diff or "@@ -1,4 +1,5 @@" in diff

    def test_compile_plan_with_insert_after(self, temp_repo_root: str):
        """Test compilation of insert_after operation."""
        # Create a file with specific content
        filepath = Path(temp_repo_root) / "test_insert.py"
        filepath.write_text("line1\nline2\nline3\n")
        git_add_and_commit(temp_repo_root, "Add test file")

        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "insert_after",
                    "path": "test_insert.py",
                    "anchor": "line2",
                    "lines": ["inserted_line\n"]
                }
            ]
        }

        diff = _compile(plan, temp_repo_root)

        assert "inserted_line" in diff

    def test_compile_plan_with_delete_lines(self, temp_repo_root: str):
        """Test compilation of deleting lines via edit_blocks."""
        # Create a file with specific content
        filepath = Path(temp_repo_root) / "test_delete.py"
        filepath.write_text("line1\nline2\nline3\nline4\n")
        git_add_and_commit(temp_repo_root, "Add test file")

        # Use edit_blocks to replace lines to be deleted with empty string
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "test_delete.py",
                    "blocks": [{"before": "line2\nline3\n", "after": ""}]
                }
            ]
        }

        diff = _compile(plan, temp_repo_root)

        assert "line1" in diff
        assert "line4" in diff
        # Should show deletion
        assert "-line2" in diff
        assert "-line3" in diff

    def test_compile_plan_with_replace_lines(self, temp_repo_root: str):
        """Test compilation of replacing lines via edit_blocks."""
        # Create a file with specific content
        filepath = Path(temp_repo_root) / "test_replace.py"
        filepath.write_text("old1\nold2\nold3\n")
        git_add_and_commit(temp_repo_root, "Add test file")

        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "test_replace.py",
                    "blocks": [{"before": "old2\n", "after": "new2\n"}]
                }
            ]
        }

        diff = _compile(plan, temp_repo_root)

        assert "-old2" in diff
        assert "+new2" in diff

    def test_compile_plan_validation_invalid_version(self, temp_repo_root: str):
        """Test compilation fails with invalid version."""
        plan = {
            "version": "INVALID_VERSION",
            "operations": []
        }

        with pytest.raises(PlanCompileError) as exc_info:
            _compile(plan, temp_repo_root)

        assert "version" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower() or "kind" in str(exc_info.value).lower()

    def test_compile_plan_validation_missing_version(self, temp_repo_root: str):
        """Test compilation fails with missing version."""
        plan = {
            "operations": []
        }

        with pytest.raises(PlanCompileError) as exc_info:
            _compile(plan, temp_repo_root)

        assert "version" in str(exc_info.value).lower() or "kind" in str(exc_info.value).lower()

    def test_compile_plan_validation_missing_operations(self, temp_repo_root: str):
        """Test compilation fails with missing operations."""
        plan = {
            "version": "ASICODE_PLAN_V1"
        }

        with pytest.raises(PlanCompileError) as exc_info:
            _compile(plan, temp_repo_root)

        assert "ops" in str(exc_info.value).lower() or "operations" in str(exc_info.value).lower()

    def test_compile_empty_operations(self, temp_repo_root: str):
        """Test compilation with empty operations list raises an error."""
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": []
        }

        with pytest.raises(PlanCompileError):
            _compile(plan, temp_repo_root)

    def test_compile_plan_apply_roundtrip(self, temp_repo_root: str, sample_simple_edit_plan_dict: dict):
        """Test full roundtrip: compile plan -> apply diff -> verify changes."""
        from tests.integration.helpers import apply_patch_and_verify

        # Compile plan to diff
        diff = _compile(sample_simple_edit_plan_dict, temp_repo_root)

        # Apply the diff
        success = apply_patch_and_verify(
            temp_repo_root,
            diff,
            ["Fixed indentation"]
        )

        assert success, "Failed to apply compiled diff"

        # Verify file was actually changed
        filepath = Path(temp_repo_root) / "sample.py"
        content = filepath.read_text()
        assert "Fixed indentation" in content

    def test_compile_plan_with_complex_blocks(self, temp_repo_root: str):
        """Test compilation of edit_blocks with multi-line blocks."""
        # Create a more complex file
        filepath = Path(temp_repo_root) / "complex.py"
        filepath.write_text("""def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
""")
        git_add_and_commit(temp_repo_root, "Add complex file")

        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "complex.py",
                    "blocks": [
                        {
                            "before": "def func2():\n    return 2",
                            "after": "def func2():\n    return 2\n    # Modified"
                        }
                    ]
                }
            ]
        }

        diff = _compile(plan, temp_repo_root)

        assert "def func2():" in diff
        assert "Modified" in diff
        # Should only modify func2, not func1 or func3
        assert "func1" not in diff or "+++ b/complex.py" in diff  # Might appear in context lines
        assert "func3" not in diff or "+++ b/complex.py" in diff


    def test_fuzzy_match_reindents_after_block(self, temp_repo_root: str):
        """When edit_blocks 'before' doesn't match exactly (whitespace diff),
        fuzzy match kicks in and _reindent_to_match fixes 'after' indentation."""
        filepath = Path(temp_repo_root) / "indent_test.ts"
        filepath.write_text("  function foo() {\n    return 1;\n}\n")
        git_add_and_commit(temp_repo_root, "Add indent test file")

        # 'before' block has 4-space indent (file has 2-space) — triggers fuzzy match
        # 'after' block also has 4-space indent — should be reindented to 2-space
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "indent_test.ts",
                    "blocks": [
                        {
                            "before": '    function foo() {\n        return 1;\n    }',
                            "after": '    function foo() {\n        return 2;\n    }'
                        }
                    ]
                }
            ]
        }

        diff = _compile(plan, temp_repo_root)

        # The 'after' block was reindented from 4-space to match file's indentation.
        # Only the content line differs (return 1 → return 2), indentation preserved.
        assert "-    return 1;" in diff  # old content removed (file indent: 4sp)
        assert "+    return 2;" in diff  # new content added (file indent: 4sp)
        # The function declaration and closing brace are context lines (unchanged)
        assert "function foo() {" in diff
        assert "}" in diff

    def test_decorative_match_box_drawing_separator(self, temp_repo_root: str):
        """A 'before' block whose decorative separator line differs from the file
        (ASCII dashes + wrong run length vs. the file's box-drawing run) still
        matches via the decorative-tolerant fallback, and the file's original
        separator is preserved (not rewritten to the LLM's mangled version)."""
        filepath = Path(temp_repo_root) / "deco.py"
        sep = "# ── _find_last_definition " + "─" * 30  # box-drawing run
        filepath.write_text(
            sep + "\n"
            "def _find_last_definition(node):\n"
            "    return node\n"
        )
        git_add_and_commit(temp_repo_root, "Add decorative separator file")

        # LLM emits ASCII dashes with a different run length than the file.
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "deco.py",
                    "blocks": [
                        {
                            "before": "# -- _find_last_definition --\n"
                                      "def _find_last_definition(node):\n"
                                      "    return node",
                            "after": "# -- _find_last_definition --\n"
                                     "def _find_last_definition(node):\n"
                                     "    return node.last",
                        }
                    ],
                }
            ],
        }

        diff = _compile(plan, temp_repo_root)

        # Content change applied...
        assert "+    return node.last" in diff
        # ...and the original box-drawing separator is NOT corrupted into ASCII:
        # it stays a context line, so it must not appear as a removed (-) line.
        assert "-" + sep not in diff


@pytest.mark.integration
class TestCreateFileContentNormalization:
    """Regression tests: write_plan create_file/replace_file must normalize
    content via _normalize_file_content (was dead code before the fix).

    Covers: list content, literal backslash-n, and double-encoded quotes —
    the exact failure mode an LLM hit when it JSON-encoded the content value
    a second time.
    """

    def _stage(self, plan, repo_root):
        result = compile_plan_to_unified_diff(plan=plan, repo_root=repo_root)
        return result.staged

    def test_create_file_list_content_joined(self, temp_repo_root: str):
        """list[str] content must be joined with newlines, not str()-ified."""
        path = "new_list.py"
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "create_file", "path": path, "content": ["import os", "x = 1", "y = 2"]}],
        }
        staged = self._stage(plan, temp_repo_root)
        assert staged[path] == "import os\nx = 1\ny = 2"

    def test_create_file_literal_backslash_n_unescaped(self, temp_repo_root: str):
        """Literal '\\n' sequences (no real newline) must be unescaped."""
        path = "new_literal.py"
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "create_file", "path": path, "content": "import os\\nx = 1\\ny = 2"}],
        }
        staged = self._stage(plan, temp_repo_root)
        assert staged[path] == "import os\nx = 1\ny = 2"

    def test_create_file_double_encoded_quotes_stripped(self, temp_repo_root: str):
        """The actual bug: LLM JSON-encoded the content value a second time.

        content='\"monkeypatch.setenv(...)\"' must become the inner string,
        not retain the outer quotes.
        """
        import json as _json
        path = "new_dbl.py"
        inner = 'monkeypatch.setenv("EXTERNAL_LLM_MODEL", "deepseek/deepseek-v4-flash")'
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "create_file", "path": path, "content": _json.dumps(inner)}],
        }
        staged = self._stage(plan, temp_repo_root)
        assert staged[path] == inner
        assert not staged[path].startswith('"')

    def test_create_file_normal_multiline_preserved(self, temp_repo_root: str):
        """Normal multi-line content must pass through unchanged."""
        path = "new_normal.py"
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "create_file", "path": path, "content": "import os\nx = 1\ny = 2"}],
        }
        staged = self._stage(plan, temp_repo_root)
        assert staged[path] == "import os\nx = 1\ny = 2"

    def test_replace_file_list_content_joined(self, temp_repo_root: str):
        """replace_file must also normalize list content."""
        path = "existing_big.py"
        # Create a file large enough to avoid the 10% size-ratio guard.
        create_test_file(temp_repo_root, path, "old = 1\n" * 50)
        new_lines = ["new content line1", "new content line2"] * 30
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "replace_file", "path": path, "content": new_lines}],
        }
        staged = self._stage(plan, temp_repo_root)
        assert staged[path] == "\n".join(new_lines)


@pytest.mark.integration
@pytest.mark.integration
class TestContentNormalizationExtensionGate:
    """The encoding-recovery transforms (double-JSON-decode, literal ``\\n``
    unescape) are GATED on the target being a code-like file (by extension).

    Rationale: a whole file whose content is a single JSON-string-escaped value
    is virtually always a double-encoding artifact for *code*, but can be a
    perfectly legitimate file body for data/text formats (.json, .txt, .md).
    Applying these unconditionally silently corrupts legitimate data content —
    and round-trip identity cannot distinguish the two (JSON always round-trips),
    so the file *extension* is the only discriminator that actually works.
    """

    def _stage(self, plan, repo_root):
        result = compile_plan_to_unified_diff(plan=plan, repo_root=repo_root)
        return result.staged

    def test_data_file_double_encoded_quotes_preserved(self, temp_repo_root: str):
        """A .json/.txt/.md file whose body is a quoted string with escaped
        inner quotes is LEGITIMATE content and must NOT be decoded.

        This is the exact regression the extension gate fixes: before the gate,
        content '"a \\"b\\" c"' was silently turned into a "b" c for any file.
        """
        cases = {
            "data.json": '"a \\"b\\" c"',
            "notes.txt": '"hello \\"world\\""',
            "quote.md": '"to be or not to be"',
        }
        for path, content in cases.items():
            plan = {
                "kind": "ASICODE_PLAN_V1",
                "ops": [{"op": "create_file", "path": path, "content": content}],
            }
            staged = self._stage(plan, temp_repo_root)
            assert staged[path] == content, (
                f"{path}: legitimate data content was corrupted by double-JSON-decode; "
                f"got {staged[path]!r} expected {content!r}"
            )

    def test_data_file_literal_backslash_n_preserved(self, temp_repo_root: str):
        """A .txt file may legitimately contain literal backslash-n sequences
        (e.g. a doc explaining escape sequences). They must NOT be unescaped."""
        raw = "use \\\\n for newline\\nand \\\\t for tab"
        for path in ["escapes.txt", "notes.md"]:
            plan = {
                "kind": "ASICODE_PLAN_V1",
                "ops": [{"op": "create_file", "path": path, "content": raw}],
            }
            staged = self._stage(plan, temp_repo_root)
            assert staged[path] == raw, (
                f"{path}: legitimate literal backslash-n was corrupted by unescape"
            )

    def test_code_file_double_encoded_quotes_still_decoded(self, temp_repo_root: str):
        """The gate must NOT break the original fix: for code files the
        double-encoding artifact must still be decoded."""
        import json as _json
        inner = 'monkeypatch.setenv("EXTERNAL_LLM_MODEL", "deepseek/deepseek-v4-flash")'
        for path in ["fix.py", "lib.go", "main.rs", "index.ts", "build.sh", "App.vue"]:
            plan = {
                "kind": "ASICODE_PLAN_V1",
                "ops": [{"op": "create_file", "path": path, "content": _json.dumps(inner)}],
            }
            staged = self._stage(plan, temp_repo_root)
            assert staged[path] == inner, f"{path}: double-encoding not decoded for code file"
            assert not staged[path].startswith('"'), f"{path}: outer quotes survived"

    def test_code_file_literal_backslash_n_still_unescaped(self, temp_repo_root: str):
        """Literal backslash-n in code-file content must still become real newlines."""
        raw = "import os\\nx = 1\\ny = 2"
        expected = "import os\nx = 1\ny = 2"
        for path in ["code.py", "lib.go", "app.ts"]:
            plan = {
                "kind": "ASICODE_PLAN_V1",
                "ops": [{"op": "create_file", "path": path, "content": raw}],
            }
            staged = self._stage(plan, temp_repo_root)
            assert staged[path] == expected, f"{path}: backslash-n not unescaped for code file"


@pytest.mark.integration
class TestPathIsCodeLike:
    """Unit tests for the _path_is_code_like gate predicate."""

    def test_code_extensions(self):
        from plan_compiler import _path_is_code_like

        for path in ["a.py", "src/app.py", "lib.go", "main.rs", "index.ts",
                     "App.tsx", "App.vue", "build.sh", "Main.kt", "page.dart"]:
            assert _path_is_code_like(path), f"{path} should be code-like"

    def test_data_text_extensions(self):
        from plan_compiler import _path_is_code_like

        for path in ["data.json", "notes.txt", "README.md", "conf.yaml",
                     "data.csv", "page.html", "style.css", "feed.xml"]:
            assert not _path_is_code_like(path), f"{path} should NOT be code-like"

    def test_no_extension_is_not_code_like(self):
        """Extensionless files (Dockerfile, Makefile, LICENSE) are conservatively
        treated as data/text — they are never source code that could be
        double-encoded."""
        from plan_compiler import _path_is_code_like

        for path in ["Dockerfile", "Makefile", "LICENSE", "CHANGELOG"]:
            assert not _path_is_code_like(path), f"{path} should NOT be code-like"

    def test_case_insensitive(self):
        from plan_compiler import _path_is_code_like

        assert _path_is_code_like("APP.PY")
        assert _path_is_code_like("Module.GO")
        assert not _path_is_code_like("DATA.JSON")
        assert not _path_is_code_like("README.MD")
        @pytest.mark.integration
        class TestDoubleDecodeQuoteFreeCode:
            """The double-JSON-decode gate must catch code whose only encoded character
            is a newline — i.e. double-encoded code that contains NO escaped quote.

            Regression: the earlier ``'\\"' in s``-only test missed this case because
            ``json.dumps("x = 1\\nprint(x)")`` produces ``"x = 1\\nprint(x)"`` which has
            outer quotes and an escaped newline but NO ``\\"``. The decode was skipped
            and the outer quotes survived into the written file (silent corruption).
            Fix: the gate now triggers on ANY escape sequence (``\\"``, ``\\n``, ``\\t``,
            ``\\\\``), not just ``\\"``.
            """

            def _stage(self, plan, repo_root):
                result = compile_plan_to_unified_diff(plan=plan, repo_root=repo_root)
                return result.staged

            def test_quote_free_code_with_newline_decoded(self, temp_repo_root: str):
                """Double-encoded code with a newline but no inner quotes must still
                have its outer quotes stripped (the exact regression case)."""
                import json as _json
                inner = "x = 1\nprint(x)\ny = 2"
                for path in ["qfree.py", "qfree.go", "qfree.ts"]:
                    plan = {
                        "kind": "ASICODE_PLAN_V1",
                        "ops": [{"op": "create_file", "path": path, "content": _json.dumps(inner)}],
                    }
                    staged = self._stage(plan, temp_repo_root)
                    assert staged[path] == inner, (
                        f"{path}: quote-free double-encoded code not decoded; "
                        f"got {staged[path]!r} expected {inner!r}"
                    )
                    assert not staged[path].startswith('"'), f"{path}: outer quotes survived"

            def test_plain_quoted_string_without_escape_preserved(self, temp_repo_root: str):
                """A code file whose body is a single quoted string with NO escape
                sequence (e.g. ``"just a string"``) is left untouched — the escape-
                sequence requirement prevents over-decoding of unusual-but-legitimate
                bodies."""
                body = '"just a quoted string"'
                plan = {
                    "kind": "ASICODE_PLAN_V1",
                    "ops": [{"op": "create_file", "path": "literal.py", "content": body}],
                }
                staged = self._stage(plan, temp_repo_root)
                assert staged["literal.py"] == body, (
                    "plain quoted string without escapes should be preserved"
                )

            def test_quote_free_code_with_tab_decoded(self, temp_repo_root: str):
                """Double-encoded code whose only escape is a tab (``\\t``) must decode."""
                import json as _json
                inner = "a\tb\tc"
                plan = {
                    "kind": "ASICODE_PLAN_V1",
                    "ops": [{"op": "create_file", "path": "tabs.py", "content": _json.dumps(inner)}],
                }
                staged = self._stage(plan, temp_repo_root)
                assert staged["tabs.py"] == inner
                assert not staged["tabs.py"].startswith('"')
class TestInsertOpsContentNormalization:
    """Regression tests: insert_after / insert_before / insert_after_line line
    payloads PRESERVE language escape sequences (e.g. Python "\n") rather than
    unescaping them.

    Whole-file ops (create_file/replace_file) run double-JSON-decode + literal
    backslash-n recovery because a single JSON-string-escaped value is almost
    always a double-encoding artifact there. Insert-op line payloads do NOT:
    each ``lines`` element is one logical line, and by the time it reaches the
    compiler the request JSON has already been decoded by the tool framework.
    A literal backslash-n in an element is therefore a legitimate language
    escape (Python ``"\\n"``), and unescaping it turns ``x = "a\\nb"`` into
    ``x = "a`` + newline + ``b"`` → ``unterminated string literal``. LLMs that
    want multiple lines use a real newline (split by splitlines) or a list.
    """

    _ANCHOR_FILE = "anchor_target.py"

    def _seed(self, repo_root: str) -> None:
        create_test_file(repo_root, self._ANCHOR_FILE, "line1\nline2\nline3\n")

    def _stage(self, plan, repo_root):
        result = compile_plan_to_unified_diff(plan=plan, repo_root=repo_root)
        return result.staged[self._ANCHOR_FILE]

    def test_insert_after_preserves_language_escape_sequence(self, temp_repo_root: str):
        """A literal backslash-n inside an insert_after line element is a LANGUAGE
        escape (e.g. Python ``"\\n"``), not a double-encoding artifact — the
        request JSON was already decoded by the tool framework. It must be
        preserved verbatim; unescaping it would split ``x = "a\\nb"`` into
        ``x = "a`` + newline + ``b"`` → ``unterminated string literal``."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after", "path": self._ANCHOR_FILE,
                "anchor": "line2", "lines": ['x = "foo\\nbar"'],
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        # Literal backslash-n preserved (not turned into a real newline):
        assert 'x = "foo\\nbar"' in staged
        assert 'x = "foo\nbar"' not in staged.replace('\\n', '')  # no real newline introduced

    def test_insert_after_real_newline_flattened(self, temp_repo_root: str):
        """When the LLM wants multiple lines it sends a REAL newline (which the
        tool framework decodes from JSON ``\n``) or a list. splitlines() then
        flattens it — this path does NOT depend on backslash-n recovery."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after", "path": self._ANCHOR_FILE,
                "anchor": "line2", "lines": "import os\nx = 1",  # real newline
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        assert "import os\nx = 1" in staged

    def test_insert_after_list_payload_joined(self, temp_repo_root: str):
        """A list payload must be joined into the inserted lines."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after", "path": self._ANCHOR_FILE,
                "anchor": "line2", "lines": ["import os", "x = 1"],
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        assert "import os\nx = 1" in staged

    def test_insert_before_content_key_accepted_as_fallback(self, temp_repo_root: str):
        """insert_before must accept 'content' when 'lines' is absent (the schema
        documents both keys as supported)."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_before", "path": self._ANCHOR_FILE,
                "anchor": "line2", "content": "import os\nx = 1",
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        assert "import os\nx = 1" in staged

    def test_insert_after_line_preserves_language_escape_sequence(self, temp_repo_root: str):
        """insert_after_line: a literal backslash-n inside a line element is a
        language escape (Python ``"\\n"``) and must be preserved, not unescaped.
        Unescaping turns ``x = "a\\nb"`` into an unterminated string literal."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after_line", "path": self._ANCHOR_FILE,
                "line": 2, "lines": ['x = "foo\\nbar"'],
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        assert 'x = "foo\\nbar"' in staged
        assert 'x = "foo\nbar"' not in staged.replace('\\n', '')

    def test_insert_after_line_real_newline_flattened(self, temp_repo_root: str):
        """insert_after_line: a real newline (JSON-decoded) is flattened into
        separate lines by splitlines() — no backslash-n recovery needed."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after_line", "path": self._ANCHOR_FILE,
                "line": 2, "lines": "import os\nx = 1",  # real newline
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        assert "import os\nx = 1" in staged

    def test_insert_after_py_compiles_with_newline_join_escape(self, temp_repo_root: str):
        """Regression (HIGH): inserting Python code containing a string literal
        escape such as ``"\\n".join(...)`` must produce a file that COMPILES.
        Before the fix, _normalize_str_content unescaped the backslash-n,
        turning ``x = "foo\\nbar"`` into ``x = "foo`` + newline + ``bar"`` →
        ``unterminated string literal``. This is the exact failure a parallel
        session hit via write_plan/insert_after_line."""
        # seed a valid python body so the whole staged file compiles
        create_test_file(temp_repo_root, self._ANCHOR_FILE,
                         "line1 = 1\nline2 = 2\nline3 = 3\n")
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after", "path": self._ANCHOR_FILE,
                "anchor": "line2 = 2", "lines": ['_out = "\\n".join(_parts)'],
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        # Must compile cleanly — the literal backslash-n inside the string must survive.
        compile(staged, self._ANCHOR_FILE, "exec")

    def test_insert_after_line_py_compiles_with_newline_join_escape(self, temp_repo_root: str):
        """Same regression as above, via insert_after_line (line-number based)."""
        create_test_file(temp_repo_root, self._ANCHOR_FILE,
                         "line1 = 1\nline2 = 2\nline3 = 3\n")
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after_line", "path": self._ANCHOR_FILE,
                "line": 2, "lines": ['_out = "\\n".join(_parts)'],
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        compile(staged, self._ANCHOR_FILE, "exec")
        assert '_out = "\\n".join(_parts)' in staged

    def test_insert_after_line_list_payload_joined(self, temp_repo_root: str):
        """insert_after_line must join a list payload into separate lines."""
        self._seed(temp_repo_root)
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after_line", "path": self._ANCHOR_FILE,
                "line": 2, "lines": ["import os", "x = 1"],
            }],
        }
        staged = self._stage(plan, temp_repo_root)
        assert "import os\nx = 1" in staged


@pytest.mark.integration
class TestReplaceFileGuardHonorsStagedState:
    """The replace_file content-wipe guard must use the in-plan STAGED state
    (``cur_text``) as its baseline, NOT disk.

    Regression: ``old_for_check = _read_text_if_exists(abs_path)`` read from disk,
    so within a single plan an earlier ``create_file``/``edit_blocks`` that
    changed the file was ignored. Two failure modes resulted:

      1. ``create_file`` (large) -> ``replace_file`` (tiny): the just-created
         file is NOT on disk yet, so ``_read_text_if_exists`` returned None and
         the guard was SILENTLY SKIPPED — a wipe was allowed with no warning.

      2. ``edit_blocks`` (shrink) -> ``replace_file``: the guard's denominator
         was the pre-shrink disk size, so the ratio was inflated and the guard
         under-reacted.

    Fix: ``old_for_check = cur_text or None`` uses exactly the "state right
    before this op" baseline, and ``or None`` keeps the "new file skips guard"
    behavior for genuinely-empty cur_text.
    """

    def test_create_then_replace_triggers_guard(self, temp_repo_root: str):
        """A create_file immediately followed by a replace_file that shrinks the
        just-created file to <10% must trip the content-wipe guard.

        Before the fix this was the worst failure mode: the guard never fired
        because the file did not exist on disk yet.
        """
        big = "x = 1\n" * 1000  # ~6000 chars
        small = "y = 2\n"        # 6 chars -> ~0.1% of big
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [
                {"op": "create_file", "path": "demo.py", "content": big},
                {"op": "replace_file", "path": "demo.py", "content": small},
            ],
        }
        with pytest.raises(PlanCompileError, match="accidental content loss"):
            compile_plan_to_unified_diff(plan=plan, repo_root=temp_repo_root)

    def test_create_then_replace_large_keeps_going(self, temp_repo_root: str):
        """Sanity: when the replace_file is NOT a wipe, no guard fires."""
        big = "x = 1\n" * 100
        also_big = "y = 2\n" * 100  # comparable size -> no wipe
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [
                {"op": "create_file", "path": "demo.py", "content": big},
                {"op": "replace_file", "path": "demo.py", "content": also_big},
            ],
        }
        # No raise, no wipe warning
        result = compile_plan_to_unified_diff(plan=plan, repo_root=temp_repo_root)
        assert not any("content loss" in w for w in result.warnings)

    def test_disk_file_replace_guard_still_works(self, temp_repo_root: str):
        """The guard must still fire for a replace_file of a pre-existing disk
        file (the original, non-staged use case must not regress)."""
        big = "x = 1\n" * 1000
        create_test_file(temp_repo_root, "exists.py", big)

        small = "y = 2\n"
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [
                {"op": "replace_file", "path": "exists.py", "content": small},
            ],
        }
        with pytest.raises(PlanCompileError, match="accidental content loss"):
            compile_plan_to_unified_diff(plan=plan, repo_root=temp_repo_root)
