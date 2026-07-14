"""Tests for cross_process_flock's no-crash contract on acquisition failure.

Regression for B1: ``msvcrt.locking(LK_LOCK)`` raises ``OSError`` after
exhausting its ~10x retry budget, but the lock context manager previously
had no ``except`` for that acquisition path — the ``OSError`` propagated to
the caller, breaking the module's documented promise that a transient lock
failure degrades to a no-op instead of crashing the protected work.

These tests force the msvcrt backend (monkeypatched, since msvcrt is
Windows-only) and assert the degrade-to-no-op behavior.
"""

from __future__ import annotations

import sys
import types

import pytest

from external_llm.common import file_lock


def _force_msvcrt_backend(monkeypatch: pytest.MonkeyPatch, *, locking_raises: bool = False) -> dict[str, int]:
    """Install a fake msvcrt module and force the msvcrt backend.

    Returns a dict counting lock/unlock calls so tests can assert whether
    unlock was (correctly) skipped after a failed acquire.
    """
    fake = types.ModuleType("msvcrt")
    # file_lock.py calls msvcrt.locking() for BOTH acquire (LK_LOCK) and
    # release (LK_UNLCK); distinguish them by mode to assert counts precisely.
    calls = {"acquire": 0, "release": 0}

    LK_LOCK = 1
    LK_UNLCK = 2

    def _locking(fd: int, mode: int, nbytes: int) -> None:
        if mode == LK_LOCK:
            calls["acquire"] += 1
            if locking_raises:
                raise OSError("locking contention")
        elif mode == LK_UNLCK:
            calls["release"] += 1

    fake.LK_LOCK = LK_LOCK
    fake.LK_UNLCK = LK_UNLCK
    fake.locking = _locking
    monkeypatch.setattr(file_lock, "_LOCK_IMPL", "msvcrt")
    monkeypatch.setattr(file_lock, "_HAS_LOCK", True)
    # msvcrt doesn't exist on POSIX (ImportError), so raising=False to create it.
    monkeypatch.setattr(file_lock, "msvcrt", fake, raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    return calls


def test_msvcrt_locking_oserror_degrades_to_noop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """OSError from msvcrt.locking must NOT propagate; body still runs, no unlock."""
    calls = _force_msvcrt_backend(monkeypatch, locking_raises=True)

    body_ran = False
    with file_lock.cross_process_flock(tmp_path / "x.lock"):
        body_ran = True

    assert body_ran, "protected body did not run after OSError degradation"
    assert calls["acquire"] == 1, "msvcrt.locking acquire must be attempted"
    assert calls["release"] == 0, "must not unlock when lock was never acquired"


def test_msvcrt_locking_success_unlocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Happy path: lock acquired then released on exit."""
    calls = _force_msvcrt_backend(monkeypatch, locking_raises=False)

    with file_lock.cross_process_flock(tmp_path / "x.lock"):
        pass

    assert calls["acquire"] == 1
    assert calls["release"] == 1


def test_posix_fcntl_path_still_works(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Sanity: fcntl backend exercises the locked=True path on POSIX."""
    monkeypatch.setattr(file_lock, "_LOCK_IMPL", "fcntl")
    monkeypatch.setattr(file_lock, "_HAS_LOCK", True)

    with file_lock.cross_process_flock(tmp_path / "x.lock"):
        pass
