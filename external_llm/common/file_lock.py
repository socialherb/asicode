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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_HAS_LOCK = False
_LOCK_IMPL: str | None = None  # "fcntl", "msvcrt", or None

try:
    import fcntl  # type: ignore[import-not-found]

    _HAS_LOCK = True
    _LOCK_IMPL = "fcntl"
except ImportError:
    try:
        import msvcrt  # type: ignore[import-not-found]

        _HAS_LOCK = True
        _LOCK_IMPL = "msvcrt"
    except ImportError:
        pass  # no-op fallback

logger = logging.getLogger(__name__)


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
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        # Binary mode is required by msvcrt.locking; fcntl.flock also works
        # on binary-mode files, so we use "wb" uniformly.
        fh = open(lock_path, "wb")
    except OSError:
        # Cannot create/open the lock file (disk full, permissions, ...
        # read-only FS). Degrade to no-op rather than crashing the caller's
        # protected work — the lock guards correctness, not availability.
        yield
        return

    # Write a placeholder byte so msvcrt.locking has a region to lock.
    # fcntl.flock ignores content and locks the whole file descriptor, but
    # the write is harmless and keeps the paths uniform.
    try:
        fh.write(b" ")
        fh.flush()
        fh.seek(0)  # Lock from offset 0 (first byte); see docstring note.
    except OSError:
        yield
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
            try:
                if _LOCK_IMPL == "fcntl":
                    fcntl.flock(fh, fcntl.LOCK_UN)
                elif _LOCK_IMPL == "msvcrt":
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        fh.close()
