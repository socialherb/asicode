"""Regression tests for SuperContextBuilder (external_llm/super_context_builder.py).

Coverage before this file: 35% — only the P25-2 bounded-read guards (2 tests)
exercised the module. ``build_context``, ``_build_enhanced_file_context``,
``_build_dependency_context``, ``_build_git_context``, ``_select_important_lines``,
``_extract_type_info`` and the entire git collaboration-metadata path were at 0%.
These tests lock in behavior contracts and exercise every branch.

Git is fully mocked (``subprocess.run``, ``get_git_snapshot``, ``_cached_git_log``)
so the tests never touch the real repository.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from external_llm import super_context_builder as scb
from external_llm.super_context_builder import SuperContextBuilder

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class _FakeProc:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _git_run(author_subjects=(), oneline=""):
    """Fake subprocess.run answering the module's two git-log invocations."""

    def _fake(cmd, *a, **kw):
        # cmd is a list; match the pretty-format string as a substring (it is
        # never a standalone element — always part of "--pretty=format:...").
        joined = " ".join(cmd)
        if "%an%x09%s" in joined:
            return _FakeProc("\n".join(f"{au}\t{su}" for au, su in author_subjects))
        if "--oneline" in cmd:
            return _FakeProc(oneline)
        return _FakeProc("", returncode=1)

    return _fake


@pytest.fixture
def builder(tmp_path, monkeypatch):
    # Neuter all git so tests never touch the real repo.
    monkeypatch.setattr(scb.subprocess, "run", _git_run())
    monkeypatch.setattr(scb, "get_git_snapshot", lambda repo: {"status": ""})
    # Bypass the TTL cache and call the real fetch (covers _fetch_recent_commits).
    monkeypatch.setattr(scb, "_cached_git_log", lambda repo, count, fetch: fetch(count))
    return SuperContextBuilder(str(tmp_path))


# --------------------------------------------------------------------------- #
# _bounded_head_text — incomplete-multibyte trim branch (L48)
# --------------------------------------------------------------------------- #

def test_bounded_head_text_trims_incomplete_multibyte(tmp_path):
    p = tmp_path / "u.txt"
    p.write_bytes(b"abc\xea\xb0")  # "abc" + an incomplete 3-byte leader
    out = scb._bounded_head_text(p, max_bytes=64)
    assert out == "abc"          # incomplete tail trimmed, no replacement char


def test_bounded_head_text_complete_passthrough(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("hello \ud55c", encoding="utf-8")
    assert scb._bounded_head_text(p, max_bytes=64) == "hello \ud55c"


# --------------------------------------------------------------------------- #
# _fetch_commit_subjects_authors (L245-277)
# --------------------------------------------------------------------------- #

def test_fetch_commits_parses_tab_separated(builder, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run",
        _git_run(author_subjects=[("Alice", "fix: #1"), ("Bob", "refactor")]),
    )
    authors, subjects = builder._fetch_commit_subjects_authors(commits=2)
    assert authors == ["Alice", "Bob"]
    assert subjects == ["fix: #1", "refactor"]


def test_fetch_commits_line_without_tab(builder, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run", lambda cmd, *a, **k: _FakeProc("NoTab", returncode=0)
    )
    authors, subjects = builder._fetch_commit_subjects_authors(commits=1)
    assert authors == ["NoTab"]
    assert subjects == [""]


def test_fetch_commits_nonzero_returncode(builder, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run", lambda cmd, *a, **k: _FakeProc("", returncode=128)
    )
    assert builder._fetch_commit_subjects_authors() == ([], [])


def test_fetch_commits_exception_returns_empty(builder, monkeypatch):
    monkeypatch.setattr(scb.subprocess, "run", lambda cmd, *a, **k: (_ for _ in ()).throw(OSError()))
    assert builder._fetch_commit_subjects_authors() == ([], [])


# --------------------------------------------------------------------------- #
# _get_recent_contributors (L278-289)
# --------------------------------------------------------------------------- #

def test_recent_contributors_prefetched_dedup_ordered(builder):
    assert builder._get_recent_contributors(
        commits=5, authors=["A", "A", "B", "", "C"]
    ) == ["A", "B", "C"]


def test_recent_contributors_fetches_on_demand(builder, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run",
        _git_run(author_subjects=[("Alice", "x"), ("Alice", "y"), ("Bob", "z")]),
    )
    assert builder._get_recent_contributors(commits=3) == ["Alice", "Bob"]


# --------------------------------------------------------------------------- #
# _detect_review_patterns (L291-307)
# --------------------------------------------------------------------------- #

def test_review_patterns_counts_keywords(builder):
    subjects = ["fix: a", "fix: b", "refactor: c", "feat: d"]
    out = builder._detect_review_patterns(commits=4, subjects=subjects)
    assert "fix:" in out and "refactor:" in out
    assert "feat" not in out  # not a tracked review keyword


def test_review_patterns_empty(builder):
    assert builder._detect_review_patterns(commits=2, subjects=["feat: x"]) == ""


# --------------------------------------------------------------------------- #
# _detect_team_conventions (L309-328)
# --------------------------------------------------------------------------- #

def test_team_conventions_files(builder, tmp_path):
    (tmp_path / ".editorconfig").write_text("x")
    (tmp_path / "pyproject.toml").write_text("")
    out = builder._detect_team_conventions()
    assert ".editorconfig" in out and "pyproject.toml" in out


def test_team_conventions_ci_linting(builder, tmp_path):
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    assert "CI linting" in builder._detect_team_conventions()


def test_team_conventions_empty(builder):
    assert builder._detect_team_conventions() == ""


# --------------------------------------------------------------------------- #
# _extract_issue_references (L330-348)
# --------------------------------------------------------------------------- #

def test_issue_references_github_and_jira(builder):
    subjects = ["fix: bug #123", "refactor PROJ-45", "plain"]
    refs = builder._extract_issue_references(commits=3, subjects=subjects)
    assert "123" in refs and "PROJ-45" in refs


def test_issue_references_empty(builder):
    assert builder._extract_issue_references(commits=1, subjects=["nothing"]) == []


# --------------------------------------------------------------------------- #
# _extract_collaboration_metadata (L206-243) — integration
# --------------------------------------------------------------------------- #

def test_collaboration_metadata_end_to_end(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run",
        _git_run(author_subjects=[("Alice", "fix: #100 PROJ-9"), ("Bob", "docs: readme")]),
    )
    (tmp_path / "pyproject.toml").write_text("")
    out = builder._extract_collaboration_metadata()
    assert "Alice" in out           # contributor
    assert "fix:" in out            # review pattern
    assert "pyproject.toml" in out  # convention
    assert "100" in out             # issue ref


def test_collaboration_metadata_empty(builder):
    assert builder._extract_collaboration_metadata() == ""


# --------------------------------------------------------------------------- #
# _build_project_metadata (L169-204)
# --------------------------------------------------------------------------- #

def test_project_metadata_readme_reqs_pyproject(builder, tmp_path):
    (tmp_path / "README.md").write_text("First para.\n\nSecond para.")
    (tmp_path / "requirements.txt").write_text("requests==2.0\nnumpy\n# c\nflask")
    (tmp_path / "pyproject.toml").write_text("")
    out = builder._build_project_metadata()
    assert "README.md" in out
    assert "First para" in out
    assert "requests" in out and "flask" in out
    assert "pyproject.toml" in out


def test_project_metadata_empty(builder):
    assert builder._build_project_metadata() == ""


# --------------------------------------------------------------------------- #
# _build_enhanced_file_context (L350-450)
# --------------------------------------------------------------------------- #

def test_enhanced_file_context_summary_and_type_info(builder, tmp_path):
    f = tmp_path / "m.py"
    f.write_text('"""Module purpose."""\nMAX = 1\ndef f(a):\n    return a\n')
    out = builder._build_enhanced_file_context(f, max_lines=500)
    assert "File Summary" in out
    assert "Module Purpose" in out
    assert "Key Functions" in out
    assert "def f(a)" in out
    assert "MAX = 1" in out          # type info (UPPERCASE global)
    assert "Functions**: 1 defined" in out


def test_enhanced_file_context_smart_snippet(builder, tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line{i}" for i in range(50)))
    out = builder._build_enhanced_file_context(f, max_lines=10)
    assert "showing important sections" in out


def test_enhanced_file_context_unparseable_still_embeds(builder, tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def (\n")  # analysis None -> no summary, but content still shown
    out = builder._build_enhanced_file_context(f, max_lines=500)
    assert "File Content" in out
    assert "File Summary" not in out


# --------------------------------------------------------------------------- #
# _build_dependency_context (L452-486)
# --------------------------------------------------------------------------- #

def _stub_graph(file_imports=None):
    return SimpleNamespace(file_imports=file_imports or {})


def test_dependency_context_imports_and_calls(builder, tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f():\n    g()\n")
    builder.dependency_builder.build_graph = lambda target_file, max_depth=1: _stub_graph({"m.py": ["os", "sys"]})
    builder.dependency_builder.format_call_graph = lambda graph, key, max_items=5: "calls -> g, h"
    out = builder._build_dependency_context(f)
    assert "This file imports" in out
    assert "`os`" in out
    assert "f()" in out            # function header
    assert "calls -> g, h" in out


def test_dependency_context_no_data(builder, tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("x = 1\n")
    builder.dependency_builder.build_graph = lambda target_file, max_depth=1: _stub_graph({})
    assert builder._build_dependency_context(f) == ""


def test_dependency_context_no_call_info_filtered(builder, tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f():\n    pass\n")
    builder.dependency_builder.build_graph = lambda target_file, max_depth=1: _stub_graph({"m.py": ["os"]})
    builder.dependency_builder.format_call_graph = lambda graph, key, max_items=5: "No call information for f"
    out = builder._build_dependency_context(f)
    assert "This file imports" in out
    assert "f()" not in out        # filtered out


def test_dependency_context_non_repo_file_caught(builder, tmp_path):
    foreign = Path("/__scb_foreign__/m.py")  # not under repo_root
    builder.dependency_builder.build_graph = lambda target_file, max_depth=1: _stub_graph()
    assert builder._build_dependency_context(foreign) == ""


# --------------------------------------------------------------------------- #
# _build_git_context (L489-507)
# --------------------------------------------------------------------------- #

def test_git_context_status_and_commits(builder, monkeypatch):
    monkeypatch.setattr(scb, "get_git_snapshot", lambda r: {"status": "M a.py"})
    monkeypatch.setattr(scb.subprocess, "run", _git_run(oneline="abc123 fix"))
    monkeypatch.setattr(scb, "_cached_git_log", lambda repo, count, fetch: fetch(count))
    out = builder._build_git_context()
    assert "M a.py" in out
    assert "abc123 fix" in out


def test_git_context_empty(builder):
    assert builder._build_git_context() == ""


# --------------------------------------------------------------------------- #
# _select_important_lines (L509-546)
# --------------------------------------------------------------------------- #

def test_select_important_lines_small_file(builder):
    sel = builder._select_important_lines(["a", "b", "c"], analysis=None, max_lines=500)
    assert sel == [(1, "a"), (2, "b"), (3, "c")]


def test_select_important_lines_function_and_class_windows(builder):
    body = [f"line{i}" for i in range(40)]
    analysis = SimpleNamespace(
        functions=[SimpleNamespace(line_number=15)],
        classes=[SimpleNamespace(line_number=25)],
    )
    nums = [n for n, _ in builder._select_important_lines(body, analysis, max_lines=500)]
    assert 1 in nums
    assert 15 in nums and 24 in nums   # function window (lines 15..24)
    assert 25 in nums                   # class window
    assert 40 not in nums               # beyond class window


def test_select_important_lines_respects_max_lines(builder):
    lines = [str(i) for i in range(100)]
    sel = builder._select_important_lines(lines, analysis=None, max_lines=10)
    assert len(sel) == 10
    assert sel[0] == (1, "0")


# --------------------------------------------------------------------------- #
# _extract_type_info (L548-557)
# --------------------------------------------------------------------------- #

def test_type_info_only_uppercase(builder):
    analysis = SimpleNamespace(global_vars={"MAX": "5", "lower": "x", "_PRIV": "1"})
    out = builder._extract_type_info(analysis)
    assert "MAX = 5" in out
    assert "lower" not in out
    assert "_PRIV" not in out          # underscore prefix is not .isupper()


def test_type_info_empty(builder):
    assert builder._extract_type_info(SimpleNamespace(global_vars={})) == ""


# --------------------------------------------------------------------------- #
# _find_first_existing (L582-588)
# --------------------------------------------------------------------------- #

def test_find_first_existing(builder, tmp_path):
    (tmp_path / "requirements.txt").write_text("x")
    found = builder._find_first_existing("README.md", "requirements.txt")
    assert found is not None and found.name == "requirements.txt"


def test_find_first_existing_none(builder):
    assert builder._find_first_existing("NOPE.md", "ALSO.md") is None


# --------------------------------------------------------------------------- #
# _get_enhanced_instructions (L590-631)
# --------------------------------------------------------------------------- #

def test_enhanced_instructions_with_target(builder):
    assert "for `app.py`" in builder._get_enhanced_instructions("app.py")


def test_enhanced_instructions_without_target(builder):
    assert "for `" not in builder._get_enhanced_instructions(None)


# --------------------------------------------------------------------------- #
# build_context (L101-167) — end-to-end
# --------------------------------------------------------------------------- #

def test_build_context_with_target(builder, tmp_path):
    f = tmp_path / "app.py"
    f.write_text('"""App."""\ndef main():\n    return 1\n')
    out = builder.build_context("fix the bug", target_file="app.py")
    assert "PROJECT CONTEXT" in out
    assert "Target File" in out
    assert "def main" in out
    assert "User Request" in out and "fix the bug" in out
    assert "Instructions" in out


def test_build_context_without_target(builder):
    out = builder.build_context("hello")
    assert "PROJECT CONTEXT" in out
    assert "User Request" in out and "hello" in out
    assert "Target File" not in out


def test_build_context_nonexistent_target_skipped(builder):
    out = builder.build_context("r", target_file="ghost.py")
    assert "Target File" not in out
    assert "User Request" in out


def test_build_context_no_git_no_deps(builder, tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    out = builder.build_context(
        "r", target_file="a.py", include_git_context=False, include_dependencies=False
    )
    assert "Git Status" not in out
    assert "Dependencies" not in out


# --------------------------------------------------------------------------- #
# additional branch coverage
# --------------------------------------------------------------------------- #

def test_build_context_emits_metadata_git_and_dep_sections(builder, tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("Project desc.")
    monkeypatch.setattr(scb, "get_git_snapshot", lambda r: {"status": "M a.py"})
    monkeypatch.setattr(scb.subprocess, "run", _git_run(oneline="abc123 fix"))
    monkeypatch.setattr(scb, "_cached_git_log", lambda repo, count, fetch: fetch(count))
    f = tmp_path / "app.py"
    f.write_text("def main():\n    return 1\n")
    monkeypatch.setattr(
        SuperContextBuilder, "_build_dependency_context", lambda self, p: "## deps here"
    )
    out = builder.build_context("do it", target_file="app.py")
    assert "Project Metadata" in out    # L113-116
    assert "Git Status" in out          # L122-125
    assert "Dependencies" in out        # L146-149


def test_enhanced_file_context_imports_and_classes(builder, tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        '"""doc"""\nimport os\nimport sys\n'
        'class Foo(Base):\n    """a class。"""\n    def m(self):\n        return 1\n'
    )
    out = builder._build_enhanced_file_context(f, max_lines=500)
    assert "Imports" in out             # L368-370
    assert "Key Classes" in out         # L387-396
    assert "class Foo(Base)" in out


def test_enhanced_file_context_read_error(builder, tmp_path, monkeypatch):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")

    def _boom(self, *a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_text", _boom)
    out = builder._build_enhanced_file_context(f, max_lines=500)
    assert "Error reading file" in out  # L447-448


def test_fetch_recent_commits_failure_returns_empty(builder, monkeypatch):
    monkeypatch.setattr(scb.subprocess, "run", lambda cmd, *a, **k: _FakeProc("", returncode=1))
    monkeypatch.setattr(scb, "_cached_git_log", lambda repo, count, fetch: fetch(count))
    assert builder._get_recent_commits(count=3) == ""  # L580


# --------------------------------------------------------------------------- #
# on-demand fetch branches (subjects=None) + metadata edge branches
# --------------------------------------------------------------------------- #

def test_review_patterns_on_demand_fetch(builder, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run",
        _git_run(author_subjects=[("A", "fix: x"), ("B", "fix: y")]),
    )
    out = builder._detect_review_patterns(commits=2)  # subjects=None -> fetches (L298)
    assert "fix:" in out


def test_issue_references_on_demand_fetch(builder, monkeypatch):
    monkeypatch.setattr(
        scb.subprocess, "run", _git_run(author_subjects=[("A", "fix: #7")]),
    )
    refs = builder._extract_issue_references(commits=1)  # subjects=None -> fetches (L337)
    assert "7" in refs


def test_project_metadata_long_readme_first_para_skipped(builder, tmp_path):
    (tmp_path / "README.md").write_text("x" * 600)  # first para >= 500 chars (L181->185)
    out = builder._build_project_metadata()
    assert "README.md" in out
    assert "x" * 600 not in out   # oversized first paragraph NOT appended


def test_project_metadata_requirements_only_comments(builder, tmp_path):
    (tmp_path / "requirements.txt").write_text("# comment\n\n# another\n")  # L190->194
    assert builder._build_project_metadata() == ""   # main_reqs empty -> nothing emitted
