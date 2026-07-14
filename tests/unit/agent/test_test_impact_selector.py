"""Tests for impact-based test selection (the closed-loop verification glue)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from external_llm.agent.test_impact_selector import (
    select_affected_tests,
    _defined_names,
    is_test_file,
)


# ── _defined_names ───────────────────────────────────────────────────────────

def test_defined_names_extracts_functions_classes_methods(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(
        "def top_fn():\n    pass\n"
        "class Klass:\n"
        "    def method(self):\n        pass\n"
        "    async def amethod(self):\n        pass\n",
        encoding="utf-8",
    )
    names = _defined_names(f)
    assert "top_fn" in names
    assert "Klass" in names
    assert "method" in names
    assert "amethod" in names
    assert "Klass.method" in names  # qualified form for call-graph lookup


def test_defined_names_tolerates_syntax_error(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def broken(:\n", encoding="utf-8")
    # Must not raise — a half-edited file must never break selection.
    assert _defined_names(f) == set()


# ── _is_test_file ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("tests/unit/test_foo.py", True),
    ("tests/test_bar.py", True),
    ("tests/unit/baz_test.py", True),
    ("external_llm/agent/foo.py", False),
    ("tests/conftest.py", False),
])
def test_is_test_file(path, expected):
    assert is_test_file(path) is expected


# ── select_affected_tests: naming convention (primary signal) ────────────────

def _make_repo(tmp_path):
    """Create a minimal repo with a source module and matching test."""
    (tmp_path / "external_llm" / "agent").mkdir(parents=True)
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    src = tmp_path / "external_llm" / "agent" / "tool_safety.py"
    src.write_text("def repair():\n    pass\n", encoding="utf-8")
    test = tmp_path / "tests" / "unit" / "test_tool_safety.py"
    test.write_text("def test_repair():\n    pass\n", encoding="utf-8")
    # An unrelated test that must NOT be selected.
    (tmp_path / "tests" / "unit" / "test_orchestrator.py").write_text(
        "def test_x():\n    pass\n", encoding="utf-8"
    )
    return src, test


def test_import_graph_matches_full_module_path(tmp_path):
    # Downstream consumer found via import-graph (signal 3b) even when the
    # edited file has no name-matched test. Module name deliberately ends in
    # "p" ("agent_loop") — a str.rstrip(".py") bug would derive "agent_loo"
    # and never match the import_index key.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "pkg" / "agent_loop.py").write_text(
        "def run():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "unit" / "test_flow.py").write_text(
        "import pkg.agent_loop\n\ndef test_flow():\n    pass\n", encoding="utf-8"
    )
    result = select_affected_tests(str(tmp_path), ["pkg/agent_loop.py"])
    assert "tests/unit/test_flow.py" in result


def test_naming_convention_maps_source_to_test(tmp_path):
    src, test = _make_repo(tmp_path)
    result = select_affected_tests(str(tmp_path), ["external_llm/agent/tool_safety.py"])
    assert "tests/unit/test_tool_safety.py" in result
    assert "tests/unit/test_orchestrator.py" not in result


def test_editing_a_test_file_runs_it_directly(tmp_path):
    src, test = _make_repo(tmp_path)
    result = select_affected_tests(
        str(tmp_path), ["tests/unit/test_tool_safety.py"]
    )
    assert result == ["tests/unit/test_tool_safety.py"]


def test_editing_a_suffix_convention_test_file_runs_it_directly(tmp_path):
    """foo_test.py (suffix convention) must select itself, same as test_foo.py.

    Regression guard: signal 1 used ``stem.startswith("test_")`` to detect
    "the edit IS a test file", so suffix-convention tests fell through to the
    stem-index lookup ("foo_test" is not a key — the index stores "foo") and
    were never selected at all.
    """
    _make_repo(tmp_path)
    suffix_test = tmp_path / "tests" / "unit" / "orchestrator_test.py"
    suffix_test.write_text("def test_y():\n    pass\n", encoding="utf-8")
    result = select_affected_tests(str(tmp_path), ["tests/unit/orchestrator_test.py"])
    assert result == ["tests/unit/orchestrator_test.py"]


def test_empty_when_no_matching_test(tmp_path):
    # Source with no corresponding test → empty (caller falls back to full suite).
    (tmp_path / "external_llm").mkdir()
    (tmp_path / "external_llm" / "lonely.py").write_text("def f():\n    pass\n", encoding="utf-8")
    result = select_affected_tests(str(tmp_path), ["external_llm/lonely.py"])
    assert result == []


def test_leading_slash_normalized(tmp_path):
    src, test = _make_repo(tmp_path)
    # Callers sometimes pass absolute or slash-prefixed paths.
    result = select_affected_tests(str(tmp_path), ["/external_llm/agent/tool_safety.py"])
    assert "tests/unit/test_tool_safety.py" in result


# ── call-graph integration (secondary signal) ────────────────────────────────

class _FakeEdge(SimpleNamespace):
    pass


class _FakeCallGraph:
    """Minimal stand-in for CallGraphIndexer exercising get_callers()."""
    def __init__(self, callers_map):
        # callers_map: symbol -> list of {caller_file, caller_symbol}
        self._map = callers_map

    def get_callers(self, symbol):
        edges = []
        for entry in self._map.get(symbol, []):
            edges.append(
                _FakeEdge(caller_file=entry["caller_file"],
                          caller_symbol=entry.get("caller_symbol"))
            )
        return edges


def test_call_graph_adds_cross_module_test(tmp_path):
    src, test = _make_repo(tmp_path)
    # A test in a NON-matching name calls repair() — naming convention alone
    # would miss it, but the call graph must surface it.
    (tmp_path / "tests" / "test_repair_flow.py").write_text(
        "from external_llm.agent.tool_safety import repair\n"
        "def test_flow():\n    repair()\n", encoding="utf-8"
    )
    cg = _FakeCallGraph({
        "repair": [{"caller_file": "tests/test_repair_flow.py",
                    "caller_symbol": "test_flow"}],
    })
    result = select_affected_tests(
        str(tmp_path), ["external_llm/agent/tool_safety.py"], call_graph=cg
    )
    assert "tests/test_repair_flow.py" in result
    # And the naming-convention hit is still present (union).
    assert "tests/unit/test_tool_safety.py" in result


def test_call_graph_non_test_callers_ignored(tmp_path):
    src, test = _make_repo(tmp_path)
    cg = _FakeCallGraph({
        "repair": [{"caller_file": "external_llm/agent/other.py",
                    "caller_symbol": "caller_fn"}],
    })
    result = select_affected_tests(
        str(tmp_path), ["external_llm/agent/tool_safety.py"], call_graph=cg
    )
    # Non-test caller filtered out; only naming-convention hit remains.
    assert result == ["tests/unit/test_tool_safety.py"]


def test_call_graph_exception_does_not_break_selection(tmp_path):
    src, test = _make_repo(tmp_path)

    class _Boom:
        def get_callers(self, symbol):
            raise RuntimeError("indexer exploded")

    result = select_affected_tests(
        str(tmp_path), ["external_llm/agent/tool_safety.py"], call_graph=_Boom()
    )
    # Naming-convention signal still works; indexer failure is contained.
    assert "tests/unit/test_tool_safety.py" in result


# ── capping ──────────────────────────────────────────────────────────────────

def test_result_capped(tmp_path):
    (tmp_path / "external_llm").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "external_llm" / "shared.py").write_text("def f():\n    pass\n", encoding="utf-8")
    for i in range(10):
        (tmp_path / "tests" / f"test_shared_{i:02d}.py").write_text(
            "def t():\n    pass\n", encoding="utf-8"
        )
    # stem "shared" → test_shared.py (0) plus test_shared_NN.py (none match exactly),
    # but the call-graph path is unused here; verify cap is honored when exceeded.
    result = select_affected_tests(
        str(tmp_path), ["external_llm/shared.py"], max_tests=2
    )
    assert len(result) <= 2


def test_dedup(tmp_path):
    src, test = _make_repo(tmp_path)
    # Same file touched twice → no duplicate entries.
    result = select_affected_tests(
        str(tmp_path), ["external_llm/agent/tool_safety.py"] * 3
    )
    assert result.count("tests/unit/test_tool_safety.py") == 1


# ── git_status_test_files ────────────────────────────────────────────────────
# Regression guards for the porcelain-v1 -z parser. Empirically verified git
# output for the scenarios below: "RM <new>\0<old>\0D  <deleted>\0??  <new>\0".
# Earlier inline parser bugs:
#   * rename detected only for status "R " — "RM"/"RD"/" R" fell through, so
#     the origin-path field was parsed as a record and rec[3:] chopped the
#     first 3 chars off the path (garbage like "ts/test_a.py" selected);
#   * deleted files and rename-origin paths (nonexistent) were selected,
#     which aborts pytest (exit 4) and false-fails the quality gate.

import subprocess as _sp

from external_llm.agent.test_impact_selector import git_status_test_files


def _git(repo, *args):
    _sp.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True,
    )


@pytest.fixture()
def git_repo(tmp_path):
    _git(tmp_path, "init", "-q", ".")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a(): pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_del.py").write_text("def test_d(): pass\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_git_status_rename_modified_parses_cleanly(git_repo):
    """An 'RM' entry (rename + worktree modification) must yield the intact
    NEW path only — no 3-char-chopped garbage, no nonexistent origin path."""
    _git(git_repo, "mv", "tests/test_a.py", "tests/test_b.py")
    with open(git_repo / "tests" / "test_b.py", "a", encoding="utf-8") as fh:
        fh.write("# modified after rename\n")

    result = git_status_test_files(git_repo)

    assert "tests/test_b.py" in result
    assert "tests/test_a.py" not in result       # origin path no longer exists
    assert all((git_repo / p).is_file() for p in result), (
        f"nonexistent path selected (would abort pytest): {result}"
    )
    assert not any(p.startswith("ts/") for p in result), (
        f"3-char-chopped garbage path selected: {result}"
    )


def test_git_status_deleted_test_not_selected(git_repo):
    """A deleted test file must not be selected — passing it to pytest aborts
    the whole run (exit 4) and false-fails the quality gate."""
    _git(git_repo, "rm", "-q", "tests/test_del.py")
    result = git_status_test_files(git_repo)
    assert "tests/test_del.py" not in result


def test_git_status_untracked_new_dir_files_visible(git_repo):
    """Files inside a brand-new directory must be listed individually
    (--untracked-files=all), not folded into a single '?? dir/' entry."""
    newdir = git_repo / "tests" / "newpkg"
    newdir.mkdir()
    (newdir / "test_fresh.py").write_text("def test_f(): pass\n", encoding="utf-8")
    result = git_status_test_files(git_repo)
    assert "tests/newpkg/test_fresh.py" in result


def test_git_status_non_ascii_path_not_cquoted(git_repo):
    """-z output carries raw UTF-8 — a non-ASCII test filename must round-trip
    without C-quoting (which would break is_test_file matching)."""
    (git_repo / "tests" / "test_한글.py").write_text("def test_k(): pass\n", encoding="utf-8")
    result = git_status_test_files(git_repo)
    assert "tests/test_한글.py" in result


def test_git_status_non_repo_returns_empty(tmp_path):
    """Outside a git repo the helper degrades to [] instead of raising."""
    assert git_status_test_files(tmp_path / "not_a_repo") == []
