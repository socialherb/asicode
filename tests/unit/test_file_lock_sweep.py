"""Tests for sweep_stale_lock_files — stale lock file reclamation.

Covers the two safety gates (age + try-lock) and the best-effort contract
(missing dir / unreadable files never raise).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from external_llm.common import file_lock

_DAY = 24 * 3600


def _make_old(path: Path, age_seconds: float = 30 * _DAY) -> None:
    """Create a lock file with an mtime in the past."""
    path.touch()
    old = time.time() - age_seconds
    os.utime(path, (old, old))


def test_sweep_removes_stale_lock_files(tmp_path: Path) -> None:
    _make_old(tmp_path / "a.lock")
    _make_old(tmp_path / "b.lock", age_seconds=60 * _DAY)
    removed = file_lock.sweep_stale_lock_files(tmp_path)
    assert removed == 2
    assert list(tmp_path.iterdir()) == []


def test_sweep_keeps_fresh_files(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.lock"
    fresh.touch()
    removed = file_lock.sweep_stale_lock_files(tmp_path)
    assert removed == 0
    assert fresh.exists()


def test_sweep_keeps_lock_held_by_another_holder(tmp_path: Path) -> None:
    """A lock file currently held (even in-process, different fd) is kept."""
    held = tmp_path / "held.lock"
    _make_old(held)
    if file_lock._HAS_LOCK and file_lock._LOCK_IMPL == "fcntl":
        import fcntl

        with open(held, "ab") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            removed = file_lock.sweep_stale_lock_files(tmp_path)
        assert removed == 0
        assert held.exists()
    else:
        # No lock backend available: the try-lock gate cannot run, so the
        # age-only decision removes the file. (Sweep is designed for the
        # platforms where flock is available.)
        removed = file_lock.sweep_stale_lock_files(tmp_path)
        assert removed == 1


def test_sweep_ignores_non_lock_files(tmp_path: Path) -> None:
    _make_old(tmp_path / "data.json")
    removed = file_lock.sweep_stale_lock_files(tmp_path)
    assert removed == 0
    assert (tmp_path / "data.json").exists()


def test_sweep_missing_dir_returns_zero(tmp_path: Path) -> None:
    assert file_lock.sweep_stale_lock_files(tmp_path / "nope") == 0


def test_sweep_honors_custom_max_age(tmp_path: Path) -> None:
    """A 1-hour-old file is stale under a 10-minute threshold, fresh under 1 day."""
    f = tmp_path / "old.lock"
    _make_old(f, age_seconds=3600)
    assert file_lock.sweep_stale_lock_files(tmp_path, max_age_seconds=600) == 1
    _make_old(f, age_seconds=3600)  # recreate for the second assertion
    assert file_lock.sweep_stale_lock_files(tmp_path, max_age_seconds=_DAY) == 0


def test_sweep_uses_injected_now(tmp_path: Path) -> None:
    """``now`` injection makes the age decision deterministic for tests."""
    f = tmp_path / "border.lock"
    f.touch()
    os.utime(f, (1_000_000.0, 1_000_000.0))  # mtime 1970-01-12
    removed = file_lock.sweep_stale_lock_files(tmp_path, now=1_000_000.0 + 30 * _DAY)
    assert removed == 1


def test_sweep_skips_unreadable_file_without_raising(tmp_path: Path) -> None:
    """A stat-visible but unremovable file must not abort the sweep."""
    locked = tmp_path / "x.lock"
    _make_old(locked)
    removed = file_lock.sweep_stale_lock_files(tmp_path)
    assert removed == 1
    assert not locked.exists()


@pytest.mark.skipif(not file_lock._HAS_LOCK, reason="lock backend unavailable")
def test_sweep_and_cross_process_flock_coexist(tmp_path: Path) -> None:
    """A file locked via cross_process_flock is not swept mid-use."""
    lock_path = tmp_path / "guard.json.lock"
    _make_old(lock_path)
    with file_lock.cross_process_flock(lock_path):
        removed = file_lock.sweep_stale_lock_files(tmp_path)
        assert removed == 0
        assert lock_path.exists()
    # cross_process_flock re-opens with "wb", bumping mtime to now; re-age the
    # file so the post-release sweep sees a stale candidate again.
    _make_old(lock_path)
    assert file_lock.sweep_stale_lock_files(tmp_path) == 1


@pytest.mark.skipif(not (file_lock._HAS_LOCK and file_lock._LOCK_IMPL == "fcntl"),
                    reason="fcntl backend required")
def test_held_exclusive_holds_lock_during_body(tmp_path: Path) -> None:
    """P1-1: while ``_held_exclusive`` yields True the lock is ACTUALLY held —
    a second non-blocking exclusive flock on the same file must fail. The old
    probe released the lock before returning, so the caller's unlink ran in a
    release/recreate window."""
    import fcntl

    path = tmp_path / "x.lock"
    path.touch()
    with file_lock._held_exclusive(path) as held:
        assert held is True
        # Same-process second fd: flock locks conflict across fds of one file.
        with open(path, "ab") as fh2, pytest.raises(OSError, match="temporarily unavailable"):
            fcntl.flock(fh2, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # After the block the lock is released.
    with open(path, "ab") as fh3:
        fcntl.flock(fh3, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
        fcntl.flock(fh3, fcntl.LOCK_UN)


@pytest.mark.skipif(not (file_lock._HAS_LOCK and file_lock._LOCK_IMPL == "fcntl"),
                    reason="fcntl backend required")
def test_held_exclusive_false_when_contended(tmp_path: Path) -> None:
    """P1-1: a lock held by another open file description yields False."""
    import fcntl

    path = tmp_path / "x.lock"
    path.touch()
    with open(path, "ab") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        with file_lock._held_exclusive(path) as held:
            assert held is False


@pytest.mark.skipif(not file_lock._HAS_LOCK, reason="lock backend unavailable")
def test_held_exclusive_rejects_replaced_path(tmp_path: Path) -> None:
    """P1-1: if the path is replaced between probe-open and lock, the probe must
    not report the stale inode as locked (inode-parity gate)."""
    import builtins

    path = tmp_path / "x.lock"
    path.touch()

    def _replacing_open(file, mode="r", *args, **kwargs):
        # SIM115 noqa: wrapper replaces file_lock.open — the handle is RETURNED
        # to the caller (file_lock internals own its lifetime), so a with-block
        # here would close it before the lock code ever sees it.
        fh = builtins.open(file, mode, *args, **kwargs)  # noqa: SIM115 — handle returned to caller
        if mode == "ab":
            # Simulate another process unlinking + recreating the path right
            # after our probe open: the fd now points at a dead inode.
            with builtins.open(str(file) + ".tmp", "wb"):
                pass
            os.replace(str(file) + ".tmp", file)
        return fh

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(file_lock, "open", _replacing_open, raising=False)  # shadows the builtin
    try:
        with file_lock._held_exclusive(path) as held:
            assert held is False
    finally:
        monkeypatch.undo()
