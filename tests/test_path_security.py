"""
Tests for path_security.py — path normalization and validation.
"""
from __future__ import annotations

import pytest

from path_security import (
    normalize_rel_path,
    resolve_inside_repo,
    resolve_under_repo_subdir,
    validate_repo_root,
)


class TestNormalizeRelPath:
    """Tests for normalize_rel_path."""

    def test_normal_path(self):
        assert normalize_rel_path("foo/bar.py") == "foo/bar.py"

    def test_strip_quotes(self):
        assert normalize_rel_path('"foo/bar.py"') == "foo/bar.py"
        assert normalize_rel_path("'foo/bar.py'") == "foo/bar.py"

    def test_strip_whitespace(self):
        assert normalize_rel_path("  foo/bar.py  ") == "foo/bar.py"

    def test_strip_git_prefix_a(self):
        assert normalize_rel_path("a/foo/bar.py") == "foo/bar.py"

    def test_strip_git_prefix_b(self):
        assert normalize_rel_path("b/foo/bar.py") == "foo/bar.py"

    def test_strip_dot_slash(self):
        assert normalize_rel_path("./foo/bar.py") == "foo/bar.py"

    def test_strip_multiple_dot_slash(self):
        assert normalize_rel_path("././foo/bar.py") == "foo/bar.py"

    def test_backslash_to_forward_slash(self):
        assert normalize_rel_path("foo\\bar.py") == "foo/bar.py"

    def test_leading_slash_removed(self):
        assert normalize_rel_path("/foo/bar.py") == "foo/bar.py"

    def test_reject_absolute_windows(self):
        assert normalize_rel_path("C:\\foo\\bar.py") == ""
        assert normalize_rel_path("D:\\foo.txt") == ""

    def test_reject_traversal(self):
        assert normalize_rel_path("../foo/bar.py") == ""
        assert normalize_rel_path("foo/../../etc/passwd") == ""

    def test_empty_string(self):
        assert normalize_rel_path("") == ""

    def test_none_input(self):
        assert normalize_rel_path(None) == ""

    def test_only_dot(self):
        # "." is a valid relative path (current directory)
        assert normalize_rel_path(".") == "."

    def test_only_dot_slash(self):
        assert normalize_rel_path("./") == ""

    def test_valid_path_with_dots(self):
        assert normalize_rel_path("some.path/file.name") == "some.path/file.name"

    def test_git_prefix_and_dot_slash(self):
        assert normalize_rel_path("a/./foo.py") == "foo.py"

    def test_mixed_backslash_and_forward(self):
        assert normalize_rel_path("foo\\bar/baz.py") == "foo/bar/baz.py"


class TestResolveInsideRepo:
    """Tests for resolve_inside_repo."""

    def test_resolve_normal(self, tmp_path):
        (tmp_path / "foo").mkdir()
        result = resolve_inside_repo(str(tmp_path), "foo")
        assert result == (tmp_path / "foo").resolve()

    def test_resolve_nested(self, tmp_path):
        (tmp_path / "sub" / "dir").mkdir(parents=True)
        result = resolve_inside_repo(str(tmp_path), "sub/dir")
        assert result == (tmp_path / "sub" / "dir").resolve()

    def test_resolve_invalid_rel_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="path_invalid"):
            resolve_inside_repo(str(tmp_path), "../outside")

    def test_resolve_outside_repo_raises(self, tmp_path):
        # normalize_rel_path rejects ".." paths before resolve_inside_repo checks
        with pytest.raises(ValueError, match="path_invalid"):
            resolve_inside_repo(str(tmp_path), "foo/../../outside")

    def test_resolve_empty_string_raises(self, tmp_path):
        with pytest.raises(ValueError, match="path_invalid"):
            resolve_inside_repo(str(tmp_path), "")

    def test_resolve_absolute_path_normalized(self, tmp_path):
        # normalize_rel_path strips leading /, making it a relative path
        result = resolve_inside_repo(str(tmp_path), "/etc/passwd")
        assert result == (tmp_path / "etc" / "passwd").resolve()

    def test_resolve_symlink_inside_repo(self, tmp_path):
        """Symlink target inside repo should resolve successfully."""
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to("real", target_is_directory=True)
        result = resolve_inside_repo(str(tmp_path), "link")
        assert result == (tmp_path / "real").resolve()


class TestValidateRepoRoot:
    """Tests for validate_repo_root."""

    def test_valid_git_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        result = validate_repo_root(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_missing_directory_raises(self, tmp_path):
        non_existent = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="not found"):
            validate_repo_root(str(non_existent))

    def test_not_a_git_repo_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a git repo"):
            validate_repo_root(str(tmp_path))

    def test_allowed_roots_allows(self, tmp_path):
        (tmp_path / ".git").mkdir()
        result = validate_repo_root(str(tmp_path), allowed_roots=[str(tmp_path)])
        assert result == tmp_path.resolve()

    def test_allowed_roots_rejects(self, tmp_path):
        (tmp_path / ".git").mkdir()
        allowed = str(tmp_path / "other")
        with pytest.raises(ValueError, match="not in allowed list"):
            validate_repo_root(str(tmp_path), allowed_roots=[allowed])

    def test_allowed_roots_rejects_sibling_prefix_bypass(self, tmp_path):
        """Regression: allowlist entry must NOT match a sibling that merely
        shares a string prefix.

        A naive ``str.startswith`` check accepted ``/x/projects-evil`` when the
        allowlist held ``/x/projects`` (the actual CVE-class bypass this guard
        closes). Path-boundary-correct comparison (relative_to) must reject it.
        """
        (tmp_path / "projects" / ".git").mkdir(parents=True)
        (tmp_path / "projects-evil" / ".git").mkdir(parents=True)
        allowed = str(tmp_path / "projects")  # allowlist the *real* projects dir
        # sibling sharing only a string prefix must be rejected
        with pytest.raises(ValueError, match="not in allowed list"):
            validate_repo_root(str(tmp_path / "projects-evil"), allowed_roots=[allowed])

    def test_allowed_roots_allows_subdirectory_of_root(self, tmp_path):
        """A repo nested under an allowed root must still be accepted."""
        nested = tmp_path / "dev" / "myrepo"
        nested.mkdir(parents=True)
        (nested / ".git").mkdir()
        # allowlist the parent; a deeper repo must pass
        result = validate_repo_root(str(nested), allowed_roots=[str(tmp_path)])
        assert result == nested.resolve()

    def test_allowed_roots_empty_list_treated_as_none(self, tmp_path):
        """Empty list means allow all."""
        (tmp_path / ".git").mkdir()
        result = validate_repo_root(str(tmp_path), allowed_roots=[])
        assert result == tmp_path.resolve()

    def test_allowed_roots_none_allows_all(self, tmp_path):
        (tmp_path / ".git").mkdir()
        result = validate_repo_root(str(tmp_path), allowed_roots=None)
        assert result == tmp_path.resolve()

    def test_resolve_symlink_repo_root(self, tmp_path):
        """Repo root via symlink should resolve to real path."""
        real = tmp_path / "real_repo"
        real.mkdir()
        (real / ".git").mkdir()
        link = tmp_path / "link_to_repo"
        link.symlink_to("real_repo", target_is_directory=True)
        result = validate_repo_root(str(link))
        assert result == real.resolve()


class TestResolveUnderRepoSubdir:
    """Tests for resolve_under_repo_subdir — constrains an attacker-controlled
    path (e.g. the continuation_path query param) to repo_root/<subdir>.

    Regression coverage for the arbitrary-file-read surface where a raw query
    param was passed straight to open().
    """

    SUBDIR = ".asicode/continuation"

    def test_valid_absolute_path_under_dir(self, tmp_path):
        cont = tmp_path / self.SUBDIR
        cont.mkdir(parents=True)
        f = cont / "sess.json"
        f.write_text("{}")
        result = resolve_under_repo_subdir(str(tmp_path), self.SUBDIR, str(f))
        assert result == f.resolve()

    def test_valid_relative_path_under_dir(self, tmp_path):
        cont = tmp_path / self.SUBDIR
        cont.mkdir(parents=True)
        (cont / "sess.json").write_text("{}")
        # relative candidate is anchored at repo_root, not process CWD
        result = resolve_under_repo_subdir(
            str(tmp_path), self.SUBDIR, f"{self.SUBDIR}/sess.json"
        )
        assert result == (cont / "sess.json").resolve()

    def test_reject_absolute_path_outside_repo(self, tmp_path):
        # /etc/passwd style arbitrary read — must be rejected even if it exists
        with pytest.raises(ValueError, match="path_outside_allowed"):
            resolve_under_repo_subdir(str(tmp_path), self.SUBDIR, "/etc/passwd")

    def test_reject_sibling_prefix_bypass(self, tmp_path):
        """Regression: /repo/.asicode-evil/x.json must NOT be accepted as
        'under' /repo/.asicode. A str.startswith check would wrongly accept it.
        """
        evil = tmp_path / ".asicode-evil"
        evil.mkdir(parents=True)
        (evil / "x.json").write_text("{}")
        with pytest.raises(ValueError, match="path_outside_allowed"):
            resolve_under_repo_subdir(
                str(tmp_path), self.SUBDIR, str(evil / "x.json")
            )

    def test_reject_relative_traversal(self, tmp_path):
        # relative candidate that escapes via .. — anchored at repo_root, then
        # resolves outside the allowed dir
        with pytest.raises(ValueError, match="path_outside_allowed"):
            resolve_under_repo_subdir(
                str(tmp_path), self.SUBDIR, "../../etc/passwd"
            )

    def test_reject_path_under_repo_but_not_subdir(self, tmp_path):
        """A path inside repo_root but outside the continuation subdir must be
        rejected — containment is scoped to the subdir, not the whole repo.
        """
        (tmp_path / "package.json").write_text("{}")
        with pytest.raises(ValueError, match="path_outside_allowed"):
            resolve_under_repo_subdir(
                str(tmp_path), self.SUBDIR, str(tmp_path / "package.json")
            )
