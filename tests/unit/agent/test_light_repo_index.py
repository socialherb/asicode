"""Tests for LightRepoIndex — AST-based lightweight repo index."""
import os
import tempfile
import textwrap

import pytest

from external_llm.agent.light_repo_index import LightRepoIndex


@pytest.fixture()
def repo(tmp_path):
    """Minimal fake repo with 3 Python files."""
    (tmp_path / "core.py").write_text(textwrap.dedent("""\
        class CoreService:
            def process(self, data):
                return helper(data)
    """))
    (tmp_path / "helpers.py").write_text(textwrap.dedent("""\
        from core import CoreService

        def helper(x):
            return x * 2

        def another_helper(x):
            return x + 1
    """))
    (tmp_path / "main.py").write_text(textwrap.dedent("""\
        from core import CoreService
        from helpers import helper

        def run():
            svc = CoreService()
            svc.process(42)
    """))
    return tmp_path


def test_build_indexes_all_files(repo):
    idx = LightRepoIndex(str(repo)).build()
    files = set(idx._files.keys())
    assert "core.py" in files
    assert "helpers.py" in files
    assert "main.py" in files


def test_build_is_idempotent(repo):
    idx = LightRepoIndex(str(repo))
    idx.build()
    idx.build()  # second call should not re-parse
    assert len(idx._files) == 3


def test_find_by_identifier_exact(repo):
    idx = LightRepoIndex(str(repo)).build()
    result = idx.find_by_identifier("helper")
    assert "helpers.py" in result


def test_find_by_identifier_class(repo):
    idx = LightRepoIndex(str(repo)).build()
    result = idx.find_by_identifier("CoreService")
    assert "core.py" in result


def test_find_by_identifier_token_overlap(repo):
    # "core_service" shares token "core" with nothing useful here
    # but "another_helper" shares "helper" token with "helper"
    idx = LightRepoIndex(str(repo)).build()
    result = idx.find_by_identifier("another_helper")
    assert "helpers.py" in result


def test_find_by_identifier_no_match(repo):
    idx = LightRepoIndex(str(repo)).build()
    result = idx.find_by_identifier("nonexistent_function_xyz")
    assert result == []


def test_score_files_returns_sorted_list(repo):
    idx = LightRepoIndex(str(repo)).build()
    scores = idx.score_files()
    assert len(scores) == 3
    # sorted descending
    vals = [s for _, s in scores]
    assert vals == sorted(vals, reverse=True)
    # all scores non-negative
    assert all(s >= 0 for _, s in scores)


def test_score_files_hub_file_ranks_higher(repo):
    # core.py is imported by helpers.py and main.py → highest fan-in
    idx = LightRepoIndex(str(repo)).build()
    scores = dict(idx.score_files())
    assert scores["core.py"] > scores["main.py"]


def test_bfs_from_forward(repo):
    # main.py imports core and helpers → BFS should reach them
    idx = LightRepoIndex(str(repo)).build()
    result = idx.bfs_from(["main.py"], depth=1)
    assert "main.py" in result
    # BFS at depth=1 starts from main.py, adds it + forward/reverse imports
    # core.py and helpers.py should appear eventually
    result_d2 = idx.bfs_from(["main.py"], depth=2)
    assert "core.py" in result_d2 or "helpers.py" in result_d2


def test_bfs_from_empty_seeds(repo):
    idx = LightRepoIndex(str(repo)).build()
    result = idx.bfs_from([], depth=2)
    assert result == []


def test_bfs_from_unknown_seed(repo):
    idx = LightRepoIndex(str(repo)).build()
    result = idx.bfs_from(["does_not_exist.py"], depth=2)
    assert result == []


def test_bfs_result_capped_at_20():
    """Even with many files, bfs_from never exceeds 20."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 30 files that all import "hub"
        (os.path.join(tmpdir), None)
        hub = os.path.join(tmpdir, "hub.py")
        with open(hub, "w") as f:
            f.write("def hub_func(): pass\n")
        for i in range(30):
            path = os.path.join(tmpdir, f"file_{i}.py")
            with open(path, "w") as f:
                f.write("from hub import hub_func\ndef fn(): hub_func()\n")

        idx = LightRepoIndex(tmpdir).build()
        result = idx.bfs_from(["hub.py"], depth=3)
        assert len(result) <= 20


def test_skip_dirs_excluded(tmp_path):
    """venv/ and __pycache__/ are skipped during collection."""
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "lib.py").write_text("def venv_func(): pass\n")
    (tmp_path / "real.py").write_text("def real_func(): pass\n")

    idx = LightRepoIndex(str(tmp_path)).build()
    files = set(idx._files.keys())
    assert "real.py" in files
    assert not any("venv" in f for f in files)


def test_large_file_skipped(tmp_path):
    """Files over MAX_FILE_SIZE are skipped."""
    big = tmp_path / "big.py"
    big.write_bytes(b"x = 1\n" * 20_000)  # ~120 KB > 100 KB limit
    small = tmp_path / "small.py"
    small.write_text("def fn(): pass\n")

    idx = LightRepoIndex(str(tmp_path)).build()
    assert "small.py" in idx._files
    assert "big.py" not in idx._files


def test_syntax_error_file_skipped(tmp_path):
    """Files with syntax errors don't crash build."""
    (tmp_path / "bad.py").write_text("def bad(:\n    pass\n")
    (tmp_path / "good.py").write_text("def good(): pass\n")

    idx = LightRepoIndex(str(tmp_path)).build()
    assert "good.py" in idx._files
    assert "bad.py" not in idx._files


def test_reverse_imports_built(repo):
    """Reverse import index: 'core' → files importing it."""
    idx = LightRepoIndex(str(repo)).build()
    # helpers.py and main.py import 'core'
    importers = idx._reverse_imports.get("core", set())
    assert "helpers.py" in importers or "main.py" in importers


def test_max_files_cap():
    """MAX_FILES cap prevents indexing more than 300 files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(350):
            with open(os.path.join(tmpdir, f"f{i}.py"), "w") as fh:
                fh.write(f"def fn_{i}(): pass\n")
        idx = LightRepoIndex(tmpdir).build()
        assert len(idx._files) <= LightRepoIndex.MAX_FILES


class TestBfsFromEdgeCases:
    """Coverage for bfs_from edge cases: visited skip, missing info, cyclic deps."""

    def test_cyclic_dependency_visited_skip(self, tmp_path):
        """Line 142: cyclic import causes a file to be in frontier but already visited."""
        (tmp_path / "a.py").write_text("import b\n")
        (tmp_path / "b.py").write_text("import a\n")
        idx = LightRepoIndex(str(tmp_path)).build()
        result = idx.bfs_from(["a.py"], depth=3)
        assert "a.py" in result
        assert "b.py" in result

    def test_bfs_from_skips_missing_file_in_frontier(self, tmp_path):
        """Line 147: a file in frontier but not in _files is skipped (via _reverse_imports mutation)."""
        (tmp_path / "a.py").write_text("def fn(): pass\n")
        idx = LightRepoIndex(str(tmp_path)).build()
        # Manually inject a non-existent file into _reverse_imports so BFS
        # tries to add it to the frontier. Since it's not in _files,
        # _files.get(f) returns None and the `if not info: continue` guard fires.
        idx._reverse_imports.setdefault("a", set()).add("nonexistent.py")
        # BFS should not crash — the missing file in frontier is skipped
        result = idx.bfs_from(["a.py"], depth=2)
        assert "a.py" in result

    def test_visited_guard_fires_with_manipulated_seed(self, tmp_path):
        """Force line 142 by placing an already-visited file into frontier."""
        (tmp_path / "a.py").write_text("def fn(): pass\n")
        idx = LightRepoIndex(str(tmp_path)).build()
        # Seed with "a.py" (valid) and also inject it into _reverse_imports
        # via a different stem so it re-appears in frontier at depth 2
        # while already visited.
        idx._reverse_imports.setdefault("a", set()).add("a.py")
        result = idx.bfs_from(["a.py"], depth=3)
        assert "a.py" in result

    def test_import_dotted_style(self, tmp_path):
        """Lines 215-217: 'import foo.bar.baz' style imports are handled."""
        (tmp_path / "outer.py").write_text("import os.path.join\nimport sys\n")
        (tmp_path / "consumer.py").write_text("from outer import something\n")
        idx = LightRepoIndex(str(tmp_path)).build()
        # outer.py should have "join" and "sys" in its imports
        outer_info = idx._files.get("outer.py")
        assert outer_info is not None
        assert "join" in outer_info.imports
        assert "sys" in outer_info.imports
        # BFS should work correctly
        result = idx.bfs_from(["consumer.py"], depth=3)
        assert "consumer.py" in result
        assert "outer.py" in result
