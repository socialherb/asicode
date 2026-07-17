"""Unit tests for the shared SSOT ``git_list_repo_files``.

Guarantees three properties a separate ``os.walk`` + hardcoded skip-set cannot:
  * .gitignore is respected automatically (no skip-set drift);
  * non-ASCII (Korean/CJK) paths survive unmangled (the ``-z`` guarantee);
  * a non-checkout returns ``None`` (distinct from 'empty repo' = []).

These underpin the duplicate-definition guard in ``symbol_index``: a symbol
defined only in a gitignored vendored copy must NOT leak into the index.
"""
import os
import subprocess

from external_llm.common.repo_files import git_list_repo_files


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args} failed: {r.stderr.decode('utf-8','replace')}")


def _make_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "test")
    return repo


def test_non_git_dir_returns_none(tmp_path):
    """Non-checkout → None (NOT []): callers must walk as fallback."""
    repo = tmp_path / "notgit"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    assert git_list_repo_files(str(repo)) is None


def test_respects_gitignore(tmp_path):
    """gitignored files are excluded; untracked-but-not-ignored still listed."""
    repo = _make_git_repo(tmp_path)
    (repo / "tracked.py").write_text("x = 1\n")
    (repo / "vendor").mkdir()
    (repo / "vendor" / "vendored.py").write_text("y = 2\n")
    (repo / ".gitignore").write_text("vendor/\n*_pb2.py\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    # An untracked-but-not-ignored file must still appear (--others --exclude-standard)
    (repo / "untracked.py").write_text("z = 3\n")
    paths = git_list_repo_files(str(repo))
    assert paths is not None
    names = {os.path.basename(p) for p in paths}
    assert "tracked.py" in names
    assert "untracked.py" in names
    assert "vendored.py" not in names  # gitignored → excluded


def test_non_ascii_path_survives_unmangled(tmp_path):
    """Regression: Korean/CJK path round-trips exactly (no C-quoting).

    ``git ls-files`` default output C-quotes non-ASCII as ``"\\303\\..."``;
    ``-z`` emits raw NUL-separated bytes so membership tests downstream match.
    """
    repo = _make_git_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "모듈.py").write_text("x = 1\n")
    (repo / "src" / "クラス.py").write_text("y = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    paths = git_list_repo_files(str(repo))
    assert paths is not None
    assert "src/모듈.py" in paths
    assert "src/クラス.py" in paths


def test_sorted_output(tmp_path):
    """Output is sorted so downstream dicts are reproducible across machines."""
    repo = _make_git_repo(tmp_path)
    for name in ("zeta.py", "alpha.py", "mid.py"):
        (repo / name).write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    paths = git_list_repo_files(str(repo))
    assert paths == sorted(paths)
