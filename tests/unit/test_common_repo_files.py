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
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {args} failed: {r.stderr.decode('utf-8', 'replace')}")


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


def test_non_git_walk_fallback_skips_vendored(tmp_path):
    """F7: the os.walk fallback (git unavailable / non-checkout) delegates
    directory pruning to walk_policy._walk_should_skip_dir, so vendored trees
    (vendor/ site-packages/ venv*/ *.egg-info) are excluded even from a non-git
    tree — the former _FILE_INDEX_SKIP_DIRS used exact match only and descended
    into them."""
    from external_llm.common.repo_files import _FILE_INDEX_CACHE, cached_repo_file_list

    repo = tmp_path / "notgit"
    repo.mkdir()
    (repo / "real.py").write_text("x = 1\n")
    for d in ("vendor", "site-packages", "venv310", "pkg.egg-info"):
        vdir = repo / d
        vdir.mkdir()
        (vdir / "v.py").write_text("y = 2\n")
    _FILE_INDEX_CACHE.clear()
    try:
        paths = cached_repo_file_list(str(repo))
    finally:
        _FILE_INDEX_CACHE.clear()
    assert paths == ["real.py"], f"vendored trees leaked into non-git walk: {paths}"


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


# ── P5-2: TTL-cached listing + write-through invalidation ─────────────────


def _make_git_repo_with(tmp_path, files):
    repo = _make_git_repo(tmp_path)
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def test_cached_listing_returns_same_object_within_ttl(tmp_path):
    """Two calls within the TTL return the SAME cached list object."""
    from external_llm.common.repo_files import _FILE_INDEX_CACHE, cached_repo_file_list

    repo = _make_git_repo_with(tmp_path, {"a.py": "x = 1\n", "b/c.py": "y = 2\n"})
    _FILE_INDEX_CACHE.clear()
    try:
        first = cached_repo_file_list(str(repo))
        second = cached_repo_file_list(str(repo))
        assert first is second, "TTL cache hit must return the same list object"
        assert "a.py" in first and "b/c.py" in first
    finally:
        _FILE_INDEX_CACHE.clear()


def test_invalidate_for_written_path_clears_containing_repo(tmp_path):
    """A write anywhere under a cached repo root clears that repo's entry."""
    from external_llm.common.repo_files import (
        _FILE_INDEX_CACHE,
        cached_repo_file_list,
        invalidate_for_written_path,
    )

    repo = _make_git_repo_with(tmp_path, {"a.py": "x = 1\n"})
    _FILE_INDEX_CACHE.clear()
    try:
        cached_repo_file_list(str(repo))
        assert _FILE_INDEX_CACHE, "listing must be cached"
        invalidate_for_written_path(str(repo / "a.py"))
        assert not _FILE_INDEX_CACHE, "write inside repo must invalidate"
    finally:
        _FILE_INDEX_CACHE.clear()


def test_invalidate_for_written_path_outside_repo_is_noop(tmp_path):
    """A write outside every cached repo root leaves the cache untouched."""
    from external_llm.common.repo_files import (
        _FILE_INDEX_CACHE,
        cached_repo_file_list,
        invalidate_for_written_path,
    )

    repo = _make_git_repo_with(tmp_path, {"a.py": "x = 1\n"})
    _FILE_INDEX_CACHE.clear()
    try:
        cached_repo_file_list(str(repo))
        invalidate_for_written_path(str(tmp_path / "unrelated" / "x.txt"))
        assert _FILE_INDEX_CACHE, "outside-repo write must not invalidate"
    finally:
        _FILE_INDEX_CACHE.clear()


def test_atomic_write_makes_new_file_visible_immediately(tmp_path):
    """A file written via atomic_write_text is visible to the next listing.

    This is the P5-2 freshness contract: editor-lane writes funnel through
    common.atomic_io, which invalidates the shared cache, so a file created
    this turn never waits out the 60 s TTL.
    """
    from external_llm.common.atomic_io import atomic_write_text
    from external_llm.common.repo_files import (
        _FILE_INDEX_CACHE,
        cached_repo_file_list,
    )

    repo = _make_git_repo_with(tmp_path, {"a.py": "x = 1\n"})
    _FILE_INDEX_CACHE.clear()
    try:
        cached_repo_file_list(str(repo))  # warm the cache
        assert "brand_new.py" not in cached_repo_file_list(str(repo))
        atomic_write_text(str(repo / "brand_new.py"), "z = 3\n")
        assert "brand_new.py" in cached_repo_file_list(str(repo)), (
            "write-through invalidation must make the new file visible immediately"
        )
    finally:
        _FILE_INDEX_CACHE.clear()


# ── _capped_put LRU refresh (P0-1) ─────────────────────────────────────────


def test_capped_put_refreshes_existing_key_position():
    """Re-inserting an existing key must move it to the back of the dict (LRU).

    The docstring's claim that "the most-recently-inserted path is the current
    repo" was FALSE for re-inserted keys — dict keeps the ORIGINAL insertion
    position on reassignment — so a hot repo refreshed past its TTL stayed at
    the front and became the FIRST eviction candidate when a 9th repo arrived.
    """
    from external_llm.common.repo_files import _capped_put

    cache: dict = {}
    cap = 4
    for i in range(cap):
        _capped_put(cache, f"r{i}", i, cap)
    # Refresh the oldest key many times (TTL-expiry → re-store pattern).
    for _ in range(50):
        _capped_put(cache, "r0", 0, cap)
    _capped_put(cache, "new", 99, cap)  # over cap → evicts the LRU entry
    assert "r0" in cache, "hot key must survive — LRU refresh failed"
    assert len(cache) == cap
    assert "r1" not in cache, "the least-recently-used entry must be evicted"
