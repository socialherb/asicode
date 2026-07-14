"""
write_plan per-op normalization tests.

Tests cover _normalize_plan_op transformations (path, action→op, field aliases,
line→anchor, etc.) that run before plan_compiler. Pre-compile validation rejects
malformed plan structure (missing kind, string plan, etc.) with clear errors.

Run: pytest tests/unit/test_write_plan_tolerant.py -v
"""
from __future__ import annotations

import subprocess

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)

    sample = repo / "main.py"
    sample.write_text(
        "import os\n"
        "import sys\n"
        "\n"
        "def main():\n"
        "    print('hello')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    return repo


@pytest.fixture
def registry(git_repo):
    cfg = AgentConfig(
        max_turns=5,
        run_tests=False,
        run_lint=False,
        auto_test_on_patch=False,
        planning_enabled=False,
        self_review_enabled=False,
        rag_enabled=False,
                            parallel_tool_execution_enabled=False)
    return ToolRegistry(str(git_repo), cfg)


# ── Integration: _tool_write_plan with valid plans ──────────────────────────────

class TestWritePlanTolerant:

    def test_create_file_plan(self, registry, git_repo):
        """write_plan accepts a valid create_file op."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "create_file",
                "path": "autofile.py",
                "content": "# auto\n",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert (git_repo / "autofile.py").exists()

    def test_edit_blocks_before_after_shorthand(self, registry, git_repo):
        """write_plan: edit_blocks with before/after at op level (no blocks wrapper)."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": "main.py",
                "before": "def main():\n    print('hello')",
                "after": "def main():\n    print('hello world')",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "hello world" in (git_repo / "main.py").read_text()

    def test_action_edit_with_before_after(self, registry, git_repo):
        """write_plan: action='edit' + before/after → edit_blocks."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "action": "edit",
                "path": "main.py",
                "before": "import os",
                "after": "import os  # used",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "import os  # used" in (git_repo / "main.py").read_text()

    def test_line_range_edit_blocks(self, registry, git_repo):
        """write_plan: start_line/end_line → edit_blocks via file read."""
        # line 1 is "import os", replace it with "import os  # patched"
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": "main.py",
                "start_line": 1,
                "end_line": 1,
                "content": "import os  # patched",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "import os  # patched" in (git_repo / "main.py").read_text()

    def test_line_range_insert_after(self, registry, git_repo):
        """write_plan: insert_after with start_line → resolved to anchor text."""
        # Insert after line 2 ("import sys")
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after",
                "path": "main.py",
                "start_line": 2,
                "content": "import json",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "import json" in (git_repo / "main.py").read_text()

    def test_action_create_with_action_field(self, registry, git_repo):
        """write_plan: action='create' is mapped to op='create_file' by normalize_plan_op."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "action": "create",
                "path": "top_level.py",
                "content": "# top level op\n",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert (git_repo / "top_level.py").exists()

    def test_screenshot_case_action_insert_with_line_range(self, registry, git_repo):
        """Full reproduction: action='insert' with line range → insert_after via anchor."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "action": "insert",
                "path": "main.py",
                "start_line": 2,
                "end_line": 2,
                "content": "# TESTEST",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "# TESTEST" in (git_repo / "main.py").read_text()


# ── New: edit_blocks with problematic blocks (screenshot case 2) ─────────────

class TestEditBlocksTolerrant:

    # --- Unit: _normalize_plan_op ---

    def test_blocks_dict_wrapped_in_list(self, tmp_path):
        """blocks as object (not list) → wrapped in list."""
        r = ToolRegistry(str(tmp_path), AgentConfig(max_turns=1))
        repairs = []
        op = r._normalize_plan_op({
            "op": "edit_blocks",
            "path": "x.py",
            "blocks": {"before": "old", "after": "new"},
        }, repairs)
        assert isinstance(op.get("blocks"), list)
        assert op["blocks"][0] == {"before": "old", "after": "new"}

    def test_block_field_aliases_old_new(self, tmp_path):
        """blocks with 'old'/'new' fields → renamed to 'before'/'after'."""
        r = ToolRegistry(str(tmp_path), AgentConfig(max_turns=1))
        repairs = []
        op = r._normalize_plan_op({
            "op": "edit_blocks",
            "path": "x.py",
            "blocks": [{"old": "hello", "new": "world"}],
        }, repairs)
        blk = op["blocks"][0]
        assert blk.get("before") == "hello"
        assert blk.get("after") == "world"

    def test_block_field_aliases_search_replace(self, tmp_path):
        """blocks with 'search'/'replacement' fields → renamed."""
        r = ToolRegistry(str(tmp_path), AgentConfig(max_turns=1))
        repairs = []
        op = r._normalize_plan_op({
            "op": "edit_blocks",
            "path": "x.py",
            "blocks": [{"search": "find_me", "replacement": "replace_me"}],
        }, repairs)
        blk = op["blocks"][0]
        assert blk.get("before") == "find_me"
        assert blk.get("after") == "replace_me"

    def test_insert_after_lines_as_string(self, tmp_path):
        """insert_after with lines as string → converted to list."""
        r = ToolRegistry(str(tmp_path), AgentConfig(max_turns=1))
        repairs = []
        op = r._normalize_plan_op({
            "op": "insert_after",
            "path": "x.py",
            "anchor": "def foo():",
            "lines": "    # comment",
        }, repairs)
        assert isinstance(op.get("lines"), list)
        assert "    # comment" in op["lines"]

    # --- Integration ---

    def test_edit_blocks_old_new_field_names(self, registry, git_repo):
        """write_plan: edit_blocks with 'old'/'new' instead of 'before'/'after'."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": "main.py",
                "blocks": [{"old": "def main():", "new": "def main():  # patched"}],
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "def main():  # patched" in (git_repo / "main.py").read_text()

    def test_insert_after_lines_string_integration(self, registry, git_repo):
        """write_plan: insert_after with lines as string (not list)."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "insert_after",
                "path": "main.py",
                "anchor": "import os",
                "lines": "# INSERTED",
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "# INSERTED" in (git_repo / "main.py").read_text()

    def test_create_file_existing_gives_clear_error(self, registry, git_repo):
        """write_plan: create_file on existing file gives actionable error."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "create_file", "path": "main.py", "content": "x"}]
        }})
        assert not result.ok
        # Should mention replace_file or edit_blocks
        assert "replace_file" in (result.error or "") or "edit_blocks" in (result.error or "")

    def test_error_contains_file_context_for_missing_before(self, registry, git_repo):
        """When edit_blocks fails, error includes file content for context."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": "main.py",
                "blocks": [{"before": "THIS TEXT DOES NOT EXIST IN FILE", "after": "x"}],
            }]
        }})
        assert not result.ok
        # Error should contain file content context
        assert "import os" in (result.error or "") or "HINT" in (result.error or "")


# ── Absolute path normalization ───────────────────────────────────────────────

class TestAbsolutePathNormalization:

    def _reg(self, tmp_path):
        cfg = AgentConfig(max_turns=1)
        return ToolRegistry(str(tmp_path), cfg)

    def test_abs_path_with_repo_prefix_stripped(self, tmp_path):
        """path starts with repo_root → strip to relative."""
        r = self._reg(tmp_path)
        abs_path = f"{tmp_path}/src/main.py"
        repairs = []
        result = r._normalize_op_path(abs_path, repairs)
        assert result == "src/main.py"
        assert any("prefix" in x for x in repairs)

    def test_abs_path_without_repo_prefix_strips_slashes(self, tmp_path):
        """path is absolute but different root → strip leading /."""
        r = self._reg(tmp_path)
        repairs = []
        result = r._normalize_op_path("/some/other/path/file.py", repairs)
        assert result == "some/other/path/file.py"

    def test_relative_path_unchanged(self, tmp_path):
        """relative path → not modified."""
        r = self._reg(tmp_path)
        repairs = []
        result = r._normalize_op_path("src/main.py", repairs)
        assert result == "src/main.py"
        assert not repairs

    def test_full_abs_path_in_op_dict(self, tmp_path):
        """_normalize_plan_op normalizes path field in op."""
        r = self._reg(tmp_path)
        repairs = []
        op = r._normalize_plan_op({
            "op": "edit_blocks",
            "path": f"{tmp_path}/main.py",
            "blocks": [{"before": "old", "after": "new"}],
        }, repairs)
        assert op["path"] == "main.py"

    def test_integration_abs_path_write_plan(self, registry, git_repo):
        """write_plan with absolute path succeeds after normalization."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": str(git_repo / "main.py"),  # absolute path
                "blocks": [{"before": "def main():", "after": "def main():  # fixed"}],
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "def main():  # fixed" in (git_repo / "main.py").read_text()

    def test_screenshot_case4_absolute_path(self, registry, git_repo):
        """Full reproduction of screenshot case 4: qwen7b uses absolute path."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": str(git_repo / "main.py"),
                "blocks": [{
                    "before": "import os",
                    "after": "import os  # patched",
                }],
            }]
        }})
        assert result.ok, f"Expected ok=True, got error: {result.error}"
        assert "import os  # patched" in (git_repo / "main.py").read_text()


# ── Before-text indent enrichment (screenshot case 6) ────────────────────────

class TestBeforeTextIndentEnrichment:
    """
    Small models strip leading whitespace from 'before' text, causing
    out.count(before) > 1 when the un-indented string appears as a
    substring of multiple indented file lines.

    _normalize_plan_op must restore the proper indentation from the file.
    """

    @pytest.fixture
    def indent_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)

        # File with TWO occurrences of the same identifier at DIFFERENT indentation levels.
        # Line 5: `  const cancelBtn = ...` (2-space indent)
        # Line 11: `    const cancelBtn = ...` (4-space indent)
        # The stripped string "const cancelBtn = ..." is a substring of BOTH lines.
        (repo / "ui.js").write_text(
            "function setup() {\n"                                     # 1
            "  // outer\n"                                             # 2
            "  function inner() {\n"                                   # 3
            "    // inner setup\n"                                     # 4
            "  const cancelBtn = document.getElementById('btn');\n"   # 5 (2-space)
            "  }\n"                                                    # 6
            "}\n"                                                      # 7
            "\n"                                                       # 8
            "function teardown() {\n"                                  # 9
            "  // teardown\n"                                          # 10
            "    const cancelBtn = document.getElementById('btn');\n"  # 11 (4-space)
            "}\n",                                                     # 12
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
        return repo

    @pytest.fixture
    def indent_registry(self, indent_repo):
        cfg = AgentConfig(
            max_turns=1,
            run_tests=False, run_lint=False, auto_test_on_patch=False,
            planning_enabled=False, self_review_enabled=False,
            rag_enabled=False,
            parallel_tool_execution_enabled=False,
        )
        return ToolRegistry(str(indent_repo), cfg)

    def test_single_match_unique_indent_restored(self, tmp_path):
        """
        before='def main()' (no indent) → file has unique '    def main():' →
        enriched to '    def main():' so out.count() == 1.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.py").write_text(
            "class Foo:\n"
            "    def main():\n"
            "        pass\n",
            encoding="utf-8",
        )
        cfg = AgentConfig(max_turns=1, run_tests=False, run_lint=False,
                          auto_test_on_patch=False, planning_enabled=False,
                          self_review_enabled=False, rag_enabled=False,
                          parallel_tool_execution_enabled=False,
                        )
        reg = ToolRegistry(str(repo), cfg)
        repairs: list = []
        op = reg._normalize_plan_op({
            "op": "edit_blocks",
            "path": "a.py",
            "blocks": [{"before": "def main():", "after": "def renamed():"}],
        }, repairs)
        assert op["blocks"][0]["before"] == "    def main():"
        assert any("indent" in r for r in repairs)

    def test_already_indented_before_not_modified(self, indent_registry, indent_repo):
        """
        before text already has leading whitespace → not touched.
        """
        repairs: list = []
        op = indent_registry._normalize_plan_op({
            "op": "edit_blocks",
            "path": "ui.js",
            # Already has the proper 2-space indent
            "blocks": [{
                "before": "  const cancelBtn = document.getElementById('btn');",
                "after": "  const cancelBtn = document.getElementById('btn'); // ok",
            }],
        }, repairs)
        before = op["blocks"][0]["before"]
        assert before.startswith("  const"), f"Should not change already-indented text: {before!r}"
        assert not any("indent" in r for r in repairs)

    def test_screenshot_case6_abs_path_stripped_unique_before_ok(self, indent_registry, indent_repo):
        """
        Screenshot case 6 (updated): abs-path normalization still fires.
        When before text is ambiguous (multiple matches), write_plan returns a clear
        'matched N locations' error — multi-match context prepend was removed to prevent
        silent distortion of LLM intent. The LLM must provide a unique before block.
        """
        # Ambiguous before → expect clear error, not silent context injection
        result_ambiguous = indent_registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": str(indent_repo / "ui.js"),  # absolute path → gets stripped
                "blocks": [{
                    "before": "const cancelBtn = document.getElementById('btn');",
                    "after": "const cancelBtn = document.getElementById('btn'); // 취소",
                }],
            }]
        }})
        assert not result_ambiguous.ok
        assert "matched 2" in (result_ambiguous.error or "").lower() or \
               "matched" in (result_ambiguous.error or "")

        # With unique before (4-space indent → unique, 2-space line doesn't contain 4-space prefix)
        result_ok = indent_registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": str(indent_repo / "ui.js"),  # absolute path → gets stripped
                "blocks": [{
                    "before": "    const cancelBtn = document.getElementById('btn');",
                    "after": "    const cancelBtn = document.getElementById('btn'); // 취소",
                }],
            }]
        }})
        assert result_ok.ok, f"write_plan failed: {result_ok.error}"
        assert "// 취소" in (indent_repo / "ui.js").read_text()


# ── Regression: existing correct formats still work ───────────────────────────

class TestWritePlanRegressions:

    def test_valid_plan_v1_still_works(self, registry, git_repo):
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "create_file", "path": "reg.py", "content": ""}]
        }})
        assert result.ok

    def test_version_field_alias_still_works(self, registry, git_repo):
        result = registry.dispatch("write_plan", {"plan": {
            "version": "ASICODE_PLAN_V1",
            "operations": [{"op": "create_file", "path": "reg2.py", "content": ""}]
        }})
        assert result.ok

    def test_ops_key_in_args_still_works(self, registry, git_repo):
        result = registry.dispatch("write_plan", {
            "ops": [{"op": "create_file", "path": "reg3.py", "content": ""}]
        })
        assert result.ok


# ── Placeholder detection ────────────────────────────────────────────────────

class TestPlaceholderDetection:
    """write_plan should reject 'OLD TEXT' / 'NEW TEXT' template placeholders."""

    def test_old_text_placeholder_rejected(self, registry, git_repo):
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "edit_blocks", "path": "main.py",
                     "blocks": [{"before": "OLD TEXT", "after": "new text"}]}],
        }})
        assert not result.ok
        assert "placeholder" in (result.error or "").lower()

    def test_new_text_placeholder_rejected(self, registry, git_repo):
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "edit_blocks", "path": "main.py",
                     "blocks": [{"before": "def main():", "after": "NEW TEXT"}]}],
        }})
        assert not result.ok
        assert "placeholder" in (result.error or "").lower()

    def test_real_text_not_rejected(self, registry, git_repo):
        """Actual code text should pass the placeholder check."""
        result = registry.dispatch("write_plan", {"plan": {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "edit_blocks", "path": "main.py",
                     "blocks": [{"before": "    print('hello')", "after": "    print('world')"}]}],
        }})
        assert result.ok, result.error


class TestLineNumberPrefixStripping:
    """_normalize_plan_op must strip read_file line-number prefixes from before/after."""

    def test_line_numbers_stripped_from_before(self, registry, git_repo, tmp_path):
        """Model copies '836:     def foo():' from read_file — strip the prefix."""
        # Create a test file
        target = git_repo / "calc.py"
        target.write_text("def add(a, b):\n    return a + b\n")

        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": "calc.py",
                "blocks": [{
                    "before": "1: def add(a, b):\n2:     return a + b",
                    "after": "def add(a, b):\n    return a + b + 0",
                }],
            }],
        }
        result = registry.dispatch("write_plan", {"plan": plan})
        assert result.ok, f"Expected success but got error: {result.error}"
        assert "add" in target.read_text()

    def test_line_numbers_stripped_only_from_prefix(self, registry, git_repo, tmp_path):
        """Numbers inside the code (e.g. '0' in 'return 0') must NOT be stripped."""
        target = git_repo / "util.py"
        target.write_text("def zero():\n    return 0\n")

        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{
                "op": "edit_blocks",
                "path": "util.py",
                "blocks": [{
                    "before": "1: def zero():\n2:     return 0",
                    "after": "def zero():\n    return 1",
                }],
            }],
        }
        result = registry.dispatch("write_plan", {"plan": plan})
        assert result.ok, f"Expected success but got error: {result.error}"
        assert "return 1" in target.read_text()
