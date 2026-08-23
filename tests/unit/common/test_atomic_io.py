"""Unit tests for the atomic write primitives (common.atomic_io).

Focus: atomic_write_bytes — the bytes analogue of atomic_write_text (content
replacement, existing-mode preservation, repo-file-index invalidation through
the atomic funnel). The text/json writers are exercised end-to-end elsewhere.
"""

from __future__ import annotations

import os

from external_llm.common.atomic_io import atomic_write_bytes


def test_atomic_write_bytes_replaces_content(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"\x00\x01old")
    atomic_write_bytes(str(p), b"\xffnew\x00")
    assert p.read_bytes() == b"\xffnew\x00"
    # No stray temp files left behind.
    assert [x.name for x in tmp_path.iterdir()] == ["f.bin"]


def test_atomic_write_bytes_preserves_existing_mode(tmp_path):
    p = tmp_path / "tool.sh"
    p.write_bytes(b"#!/bin/sh\n")
    os.chmod(p, 0o755)
    atomic_write_bytes(str(p), b"#!/bin/sh\necho hi\n")
    assert p.read_bytes() == b"#!/bin/sh\necho hi\n"
    assert (p.stat().st_mode & 0o777) == 0o755


def test_atomic_write_bytes_creates_parent_dir(tmp_path):
    p = tmp_path / "a" / "b" / "f.bin"
    atomic_write_bytes(str(p), b"data")
    assert p.read_bytes() == b"data"


def test_atomic_write_bytes_invalidates_repo_file_index(tmp_path, monkeypatch):
    """The atomic funnel (atomic_io -> invalidate_for_written_path) must fire
    for the bytes writer too — the non-UTF-8 edit_text write depends on it."""
    import external_llm.common.repo_files as common_rf

    key = common_rf.canonical_repo_key(str(tmp_path))
    common_rf._FILE_INDEX_CACHE.pop(key, None)

    def fake_listing(root):
        return ["app.py"]

    monkeypatch.setattr(common_rf, "git_list_repo_files", fake_listing)
    assert common_rf.cached_repo_file_list(str(tmp_path)) == ["app.py"]
    assert key in common_rf._FILE_INDEX_CACHE

    atomic_write_bytes(str(tmp_path / "app.py"), b"x = 2\n")
    assert key not in common_rf._FILE_INDEX_CACHE


def test_atomic_write_bytes_new_target_respects_umask(tmp_path):
    p = tmp_path / "fresh.bin"
    old_umask = os.umask(0o022)
    try:
        atomic_write_bytes(str(p), b"data")
    finally:
        os.umask(old_umask)
    assert (p.stat().st_mode & 0o777) == 0o644
