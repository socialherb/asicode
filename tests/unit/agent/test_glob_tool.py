"""Tests for the `glob` read tool and its glob→regex translator.

`glob` exists so "what files are here?" stops routing through `bash ls`/`find`,
which leaves the repo boundary, returns unbounded output, and cannot be
result-cached. That only holds if the pattern semantics are the ones every
model already assumes, so the translator is pinned separately from the tool.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from external_llm.agent.tool_handlers.read_tools import _glob_to_regex
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


class TestGlobToRegex:
    @pytest.mark.parametrize(
        "pattern,path,expected",
        [
            # `*` must not cross a separator — the reason fnmatch.translate is unusable
            ("src/*.py", "src/a.py", True),
            ("src/*.py", "src/pkg/a.py", False),
            ("*.py", "a.py", True),
            ("*.py", "a.txt", False),
            # `**` spans directories; `**/x` must also match a bare `x`
            ("src/**/*.py", "src/a.py", True),
            ("src/**/*.py", "src/pkg/deep/a.py", True),
            ("**/*.ts", "a.ts", True),
            ("**/*.ts", "x/y/a.ts", True),
            ("**/*.ts", "a.tsx", False),
            # single-char and classes
            ("a?.py", "ab.py", True),
            ("a?.py", "a/b.py", False),
            ("[abc].py", "b.py", True),
            ("[abc].py", "d.py", False),
            ("[!abc].py", "d.py", True),
            ("[!abc].py", "a.py", False),
            # anchored at both ends
            ("test_*.py", "xtest_a.py", False),
            ("test_*.py", "test_a.pyc", False),
            # regex metacharacters in the pattern are literals
            ("a.b.py", "a.b.py", True),
            ("a.b.py", "axbxpy", False),
            ("v1+2.txt", "v1+2.txt", True),
        ],
    )
    def test_matching(self, pattern: str, path: str, expected: bool):
        assert bool(_glob_to_regex(pattern).match(path)) is expected

    def test_unterminated_class_is_literal(self):
        """Must not raise — an LLM will eventually send this."""
        assert _glob_to_regex("a[bc.py").match("a[bc.py")

    @pytest.mark.parametrize(
        "pattern",
        [
            "[z-a]*.py",  # reversed range
            "[9-0].py",  # reversed digit range
        ],
    )
    def test_invalid_class_raises_value_error(self, pattern: str):
        """A broken class must surface as ValueError in glob coordinates, not
        as a raw re.error naming positions inside the translated regex."""
        with pytest.raises(ValueError, match="invalid glob pattern"):
            _glob_to_regex(pattern)

    def test_invalid_class_error_names_the_class(self):
        with pytest.raises(ValueError, match=r"\[z-a\]"):
            _glob_to_regex("[z-a]*.py")

    def test_leading_bracket_member_is_literal(self):
        """`]` right after `[` is a member (POSIX), not the closer."""
        assert _glob_to_regex("[]]").match("]")
        assert not _glob_to_regex("[]]").match("a")
        assert _glob_to_regex("[!]]").match("x")
        assert not _glob_to_regex("[!]]").match("]")

    def test_backslash_in_class_is_literal(self):
        """A backslash inside a class is a literal member — `[\\d]` is the
        chars backslash and 'd', not a digit class (POSIX glob)."""
        assert _glob_to_regex("[\\d]").match("d")
        assert _glob_to_regex("[\\d]").match("\\")
        assert not _glob_to_regex("[\\d]").match("3")

    def test_bang_only_class_is_literal(self):
        """`[!]` has no closing bracket for its negated member — literal."""
        assert _glob_to_regex("[!]").match("[!]")

    def test_posix_named_class_is_not_a_digit_class(self):
        """`[[:alpha:]]` must not crash; members are literal (no POSIX classes)."""
        assert _glob_to_regex("[[:alpha:]]").match("a]")
        assert not _glob_to_regex("[[:alpha:]]").match("b]")

    def test_leading_bracket_in_class_is_literal_member(self):
        """`[[]` is the class {`[`} in glob terms — a leading `[[` must not be
        reinterpreted as a Python 3.13+ nested set (FutureWarning today, silent
        semantics change in a future Python)."""
        import warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rx = _glob_to_regex("[[]")
        assert not any("nested set" in str(w.message) for w in rec), rec
        assert rx.match("[")
        assert not rx.match("a")
        # `[` anywhere in the body is a literal member, not a nested-set opener
        assert _glob_to_regex("[a[b]").match("[")
        assert _glob_to_regex("[a[b]").match("b")

    def test_ampersand_pair_in_class_is_literal(self):
        """`[a&&b]` is the class {a, &, b} in glob terms — the `&&` must not be
        reinterpreted as the Python 3.13+ set-intersection operator."""
        import warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rx = _glob_to_regex("[a&&b]")
        assert not any("intersection" in str(w.message) for w in rec), rec
        assert rx.match("&")
        assert rx.match("a")
        assert rx.match("b")
        assert not rx.match("c")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    for rel in ("a.py", "b.txt", "src/c.py", "src/deep/d.py", "tests/test_e.py", "한글파일.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _registry(repo: Path) -> ToolRegistry:
    return ToolRegistry(str(repo), AgentConfig(rag_enabled=False))


def _paths(result) -> set[str]:
    return {line.strip().split("  (")[0] for line in (result.content or "").splitlines()[1:] if line.strip()}


class TestGlobTool:
    def test_bare_pattern_matches_basename_anywhere(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "*.py"})
        assert result.ok
        assert _paths(result) == {
            "a.py",
            "src/c.py",
            "src/deep/d.py",
            "tests/test_e.py",
            "한글파일.py",
        }

    def test_pattern_with_separator_matches_full_path(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "src/*.py"})
        assert _paths(result) == {"src/c.py"}

    def test_double_star_spans_directories(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "src/**/*.py"})
        assert _paths(result) == {"src/c.py", "src/deep/d.py"}

    def test_non_ascii_paths_survive(self, repo: Path):
        """git ls-files without -z C-quotes non-ASCII names into oblivion."""
        result = _registry(repo).dispatch("glob", {"pattern": "한글*.py"})
        assert _paths(result) == {"한글파일.py"}

    def test_gitignored_files_are_excluded(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "ignored.py"})
        assert result.ok
        assert _paths(result) == set()

    def test_path_scopes_the_search(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "*.py", "path": "src"})
        assert _paths(result) == {"src/c.py", "src/deep/d.py"}

    def test_path_outside_repo_is_rejected(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "*.py", "path": "../"})
        assert result.ok is False
        assert "outside the repository" in (result.error or "")

    def test_no_match_is_success_not_error(self, repo: Path):
        """An empty result is an answer, not a failure — an error would push the
        model into a retry loop over a question already answered."""
        result = _registry(repo).dispatch("glob", {"pattern": "*.rs"})
        assert result.ok
        assert "No files match" in result.content

    def test_missing_pattern_is_an_error(self, repo: Path):
        result = _registry(repo).dispatch("glob", {})
        assert result.ok is False
        assert "required" in (result.error or "")

    def test_invalid_class_returns_actionable_error(self, repo: Path):
        """The tool must answer in glob coordinates, not with a translated-
        regex re.error that the model cannot act on."""
        result = _registry(repo).dispatch("glob", {"pattern": "[z-a]*.py"})
        assert result.ok is False
        assert "invalid glob pattern" in (result.error or "")
        assert "[z-a]" in (result.error or "")

    def test_max_results_truncates_and_says_so(self, repo: Path):
        result = _registry(repo).dispatch("glob", {"pattern": "*.py", "max_results": 2})
        assert result.ok
        assert len(_paths(result)) == 2
        assert "showing the first 2" in result.content

    def test_newest_first(self, repo: Path):
        """'What did I just touch?' is the question a glob usually stands in for."""
        import os
        import time

        now = time.time()
        # Back-date every match, not just some: files left at creation time
        # would interleave and make the assertion about nothing.
        order = ["src/deep/d.py", "src/c.py", "a.py", "tests/test_e.py", "한글파일.py"]
        for i, rel in enumerate(order):
            os.utime(repo / rel, (now - i * 3600, now - i * 3600))
        result = _registry(repo).dispatch("glob", {"pattern": "*.py"})
        listed = [line.strip().split("  (")[0] for line in result.content.splitlines()[1:] if line.strip()]
        assert listed == order

    def test_works_outside_a_git_checkout(self, tmp_path: Path):
        """_repo_file_index falls back to a pruned walk when git is unusable."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "m.py").write_text("x = 1\n")
        result = _registry(tmp_path).dispatch("glob", {"pattern": "*.py"})
        assert result.ok
        assert _paths(result) == {"pkg/m.py"}


class TestGlobRegistration:
    def test_exposed_to_the_model(self):
        from external_llm.agent.tool_schemas import AGENT_TOOL_SCHEMAS

        names = {s.get("function", s).get("name") for s in AGENT_TOOL_SCHEMAS}
        assert "glob" in names

    def test_is_result_cacheable(self, repo: Path):
        """The whole point over `bash ls` — a repeated listing must not re-run."""
        assert "glob" in ToolRegistry._READ_ONLY_TOOLS

    def test_unscoped_glob_reports_unknown_cache_scope(self, repo: Path):
        """A glob goes stale the moment any matching file appears or vanishes,
        so an unscoped call must not claim a narrow invalidation scope."""
        reg = _registry(repo)
        assert reg._extract_read_scope_paths("glob", {"pattern": "*.py"}) is None
        scoped = reg._extract_read_scope_paths("glob", {"pattern": "*.py", "path": "src"})
        assert scoped is not None and len(scoped) == 1
