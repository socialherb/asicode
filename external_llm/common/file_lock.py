"""Cross-process file locking — single source of truth.

Provides an exclusive cross-process lock context manager for protecting
read-modify-write cycles against concurrent processes. Uses ``fcntl.flock``
on POSIX and ``msvcrt.locking`` on Windows. Previously this
pattern was duplicated (with subtle, inconsistent differences) across:

  - external_llm/design_session.py        (_flock)
  - webapp/run_store.py                    (_file_lock)
  - external_llm/agent/checkpoint_store.py (_flock)

The canonical implementation lives here. It unifies the safest behavior
of the three copies:

  * On POSIX → ``fcntl.flock`` (exclusive, blocking).
  * On Windows → ``msvcrt.locking`` (``LK_LOCK`` — blocking with retry).
  * On platforms with neither (rare embedded / sandboxed environments) the
    lock is a no-op — the atomic-rename + merge logic of callers still
    mitigates most races, though append-only writers (e.g. JSONL) remain
    exposed to torn lines.
  * If the lock file cannot be opened (OSError — disk full, permission
    denied, read-only FS), we degrade to no-op instead of propagating the
    error. This prevents a transient/locking failure from crashing the
    caller's main work (which is what a lock *protects*, not *gates*).
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

_HAS_LOCK = False
_LOCK_IMPL: str | None = None  # "fcntl", "msvcrt", or None

try:
    import fcntl  # type: ignore[import-not-found]

    _HAS_LOCK = True
    _LOCK_IMPL = "fcntl"
except ImportError:
    with suppress(ImportError):
        import msvcrt  # type: ignore[import-not-found]

        _HAS_LOCK = True
        _LOCK_IMPL = "msvcrt"

logger = logging.getLogger(__name__)


@contextmanager
def _held_exclusive(path: Path) -> Iterator[bool]:
    """Yield True while a non-blocking exclusive lock on ``path`` is held.

    Used by :func:`sweep_stale_lock_files` as a liveness probe: a lock file
    we can acquire right now is not held by any other process, so it is safe
    to reclaim. The caller must perform the reclaim (e.g. ``unlink``) INSIDE
    the ``with`` block, while the lock is still held — releasing first would
    open a window in which another process acquires the same file and the
    sweep unlinks a live lock (the remove/recreate race this module warns
    about).

    Yields False when the lock cannot be acquired: held elsewhere, open
    failure, or the path was replaced between open and lock (the lock would
    then guard a dead inode — inode parity with ``fstat`` rejects that case).
    ``ab`` mode avoids truncating the file (mtime stays put).
    """
    if not _HAS_LOCK or _LOCK_IMPL is None:
        yield True  # no lock backend — fall back to the age-only decision
        return
    try:
        with open(path, "ab") as fh:
            acquired = False
            try:
                if _LOCK_IMPL == "fcntl":
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif _LOCK_IMPL == "msvcrt":
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as _err:
                logger.debug("stale-lock probe: %s is in use (%s)", path, _err)
            if acquired:
                # The path may have been replaced (unlink + recreate) between
                # open() and the lock acquisition; the lock would then guard an
                # inode no one else will ever contend on.
                try:
                    _st = path.stat()
                except OSError:
                    _st = None
                if _st is None or _st.st_ino != os.fstat(fh.fileno()).st_ino:
                    acquired = False
            yield acquired
    except OSError:
        yield False
        return


def sweep_stale_lock_files(
    lock_dir: Path,
    *,
    max_age_seconds: float = 7 * 24 * 3600,
    now: float | None = None,
) -> int:
    """Delete stale ``*.lock`` files from a dedicated lock directory.

    Intended for *orphan* lock directories (e.g. ``.asicode/locks``) that no
    live code path writes to any more. Safety is enforced twice:

    1. **Age gate** — only files whose mtime is older than ``max_age_seconds``
       are candidates. An actively used lock file is re-opened recently.
    2. **Try-lock gate** — a candidate is deleted only **while we hold** a
       non-blocking exclusive lock on it: ``_held_exclusive`` acquires the
       lock and ``path.unlink()`` runs inside the ``with`` block, before the
       lock is released. If another process holds the lock — or the path was
       replaced between probe-open and lock (inode mismatch) — acquisition
       fails and the file is left untouched.

    Returns the number of files removed. Best-effort: a missing directory or
    an unreadable/unremovable file is skipped, never raised.

    NOTE: do NOT run this against directories holding live
    ``cross_process_flock`` files (e.g. ``.asicode/*.lock`` siblings of active
    JSON stores). Unlinking a file another process may open next introduces
    the classic remove/recreate race; this function exists to reclaim *dead*
    directories only.
    """
    now = time.time() if now is None else now
    if not lock_dir.is_dir():
        return 0
    removed = 0
    for path in sorted(lock_dir.glob("*.lock")):
        try:
            st = path.stat()
        except OSError as _err:
            logger.debug("stale-lock sweep: stat failed for %s (%s)", path, _err)
            continue
        try:
            if now - st.st_mtime < max_age_seconds:
                continue
            with _held_exclusive(path) as _held:
                if _held:
                    path.unlink()  # still holding the lock — no release/unlink gap
                    removed += 1
        except OSError as _err:
            logger.debug("stale-lock sweep: remove failed for %s (%s)", path, _err)
    return removed


@contextmanager
def cross_process_flock(lock_path: Path) -> Iterator[None]:
    """Exclusive cross-process flock context manager.

    Acquires an exclusive lock (blocking) on ``lock_path`` for the duration
    of the ``with`` block, releasing it on exit. Implementation:

    * **fcntl** (POSIX) — ``LOCK_EX`` / ``LOCK_UN`` on the whole file.
    * **msvcrt** (Windows) — ``LK_LOCK`` / ``LK_UNLCK`` on the first byte
      (binary mode required).
    * **no-op** — yields immediately when neither backend is available, or
      when the lock file cannot be opened.

    The ``.lock`` file is harmless to leave behind after use and is *not*
    removed, avoiding a remove/recreate race between two processes.
    """
    if not _HAS_LOCK:
        yield
        return

    if _LOCK_IMPL is None:
        yield
        return
    # Ensure the parent directory exists (best-effort) so that opening the
    # lock file does not fail solely due to a missing parent.
    with suppress(OSError):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Binary read-write, non-truncating. "wb" truncates the lock file
        # BEFORE the lock is acquired; on Windows the truncate can collide
        # with a byte-range lock another process holds on the file
        # (ERROR_LOCK_VIOLATION at open) and silently degrade this lock to a
        # no-op. "a+b" never truncates. msvcrt.locking needs binary mode;
        # fcntl.flock also works on binary-mode files.
        # SIM115 noqa: open/seek/flock failures are handled DIFFERENTLY below
        # (open/seek → no-op, flock → propagate); a with-block would collapse
        # that distinction and swallow flock OSErrors as a no-op lock.
        fh = open(lock_path, "a+b")  # noqa: SIM115 — per-stage failure modes must stay distinct
    except OSError:
        # Cannot create/open the lock file (disk full, permissions, ...
        # read-only FS). Degrade to no-op rather than crashing the caller's
        # protected work — the lock guards correctness, not availability.
        yield
        return

    # Write a placeholder byte ONLY while the file is empty so
    # msvcrt.locking has a region to lock. fcntl.flock ignores content and
    # locks the whole file descriptor, but the write is harmless and keeps
    # the paths uniform. In "a+b" every write appends, so writing on every
    # acquisition would grow the lock file by one byte per lock; the size
    # check bounds it at 1-2 bytes.
    try:
        if fh.seek(0, 2) == 0:
            fh.write(b" ")
            fh.flush()
        fh.seek(0)  # Lock from offset 0 (first byte); see docstring note.
    except OSError:
        # Degrade to a no-op lock. fh must still be closed even when the
        # with-body raises — the old bare ``yield; fh.close()`` leaked the
        # descriptor on body exceptions.
        try:
            yield
        finally:
            fh.close()
        return

    # POSIX fcntl.flock(LOCK_EX) blocks indefinitely until acquired. Windows
    # msvcrt.locking(LK_LOCK) retries ~10x then raises OSError on persistent
    # contention — that must NOT propagate to the caller: this module's
    # docstring promises the lock guards correctness, not availability. On
    # OSError we degrade to a no-op (yield without holding the lock).
    locked = False
    try:
        if _LOCK_IMPL == "fcntl":
            fcntl.flock(fh, fcntl.LOCK_EX)
            locked = True
        elif _LOCK_IMPL == "msvcrt":
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                locked = True
            except OSError:
                logger.warning(
                    "msvcrt.locking failed on %s; degrading to no-op lock",
                    lock_path,
                )
        yield
    finally:
        if locked:
            with suppress(OSError):  # unlock on already-closed fd / EINVAL
                if _LOCK_IMPL == "fcntl":
                    fcntl.flock(fh, fcntl.LOCK_UN)
                elif _LOCK_IMPL == "msvcrt":
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        fh.close()
