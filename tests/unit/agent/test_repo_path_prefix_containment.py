"""Path containment must end at a separator, not at a shared text prefix.

Two sites compared a resolved path against repo_root with plain string
prefixing. A sibling directory that merely starts with the same characters
(``/a/repo`` vs ``/a/repo-evil``) satisfies that test, which is exactly the trap
``path_security._repo_within_allowlist`` and ``resolve_under_repo_subdir``
already document for their own callers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from external_llm.agent.lint_runner import LintRunner


@pytest.fixture()
def sibling_tree(tmp_path: Path) -> Path:
    """/root/repo (the repo) beside /root/repo-evil (must stay unreachable)."""
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "inside.py").write_text("x = 1\n")
    (tmp_path / "repo-evil").mkdir()
    (tmp_path / "repo-evil" / "secret.py").write_text("SECRET = 1\n")
    return tmp_path


class TestLintRunnerResolvePath:
    def test_sibling_sharing_the_prefix_is_rejected(self, sibling_tree: Path):
        runner = LintRunner(repo_root=str(sibling_tree / "repo"))
        assert runner._resolve_path("../repo-evil/secret.py") is None

    def test_absolute_sibling_is_rejected(self, sibling_tree: Path):
        runner = LintRunner(repo_root=str(sibling_tree / "repo"))
        assert runner._resolve_path(str(sibling_tree / "repo-evil" / "secret.py")) is None

    def test_file_inside_repo_still_resolves(self, sibling_tree: Path):
        runner = LintRunner(repo_root=str(sibling_tree / "repo"))
        resolved = runner._resolve_path("inside.py")
        assert resolved == (sibling_tree / "repo" / "inside.py").resolve()

    @pytest.mark.parametrize("whole_repo", ["", "."])
    def test_whole_repo_arguments_still_resolve(self, sibling_tree: Path, whole_repo: str):
        """run_ruff documents ""/"." as "lint the entire repository" — the
        containment check must not reject the root itself."""
        runner = LintRunner(repo_root=str(sibling_tree / "repo"))
        assert runner._resolve_path(whole_repo) == (sibling_tree / "repo").resolve()

    def test_missing_file_inside_repo_is_none(self, sibling_tree: Path):
        runner = LintRunner(repo_root=str(sibling_tree / "repo"))
        assert runner._resolve_path("nope.py") is None


class TestApplyPatchRepoRootStrip:
    """apply_patch turns an absolute path into a repo-relative one by stripping
    repo_root. Stripping at a non-separator boundary silently renamed the file."""

    @staticmethod
    def _strip(repo_root: str, path: str) -> str:
        """Mirror of the normalisation in _tool_apply_patch_impl."""
        repo_root_str = repo_root.rstrip("/")
        if path == repo_root_str:
            path = ""
        elif path.startswith(repo_root_str + "/"):
            path = path[len(repo_root_str) :].lstrip("/")
        if path.startswith("/"):
            path = path.lstrip("/")
        return path

    def test_path_inside_repo_becomes_relative(self):
        assert self._strip("/home/dev/repo", "/home/dev/repo/a.py") == "a.py"

    def test_nested_path_inside_repo(self):
        assert self._strip("/home/dev/repo", "/home/dev/repo/pkg/mod/a.py") == "pkg/mod/a.py"

    @pytest.mark.parametrize(
        "path",
        [
            "/home/dev/repository/a.py",
            "/home/dev/repo-backup/x.py",
        ],
    )
    def test_sibling_prefix_is_left_alone(self, path: str):
        """Previously produced 'sitory/a.py' / '-backup/x.py'."""
        assert self._strip("/home/dev/repo", path) == path.lstrip("/")

    def test_trailing_slash_repo_root(self):
        assert self._strip("/home/dev/repo/", "/home/dev/repo/a.py") == "a.py"


def test_apply_patch_impl_uses_the_separator_aware_strip():
    """Guard the mirror above against drift from the real implementation."""
    import inspect

    from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin

    src = inspect.getsource(WriteToolsMixin._tool_apply_patch_impl)
    assert 'path.startswith(repo_root_str + "/")' in src, (
        "apply_patch no longer strips repo_root at a separator boundary — "
        "update TestApplyPatchRepoRootStrip._strip to match"
    )
