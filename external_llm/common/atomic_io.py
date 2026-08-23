"""atomic_io.py -- Atomic JSON persistence helpers (single source of truth).

Provides three primitives for crash-safe JSON writes:

* :func:`atomic_write_json` -- replace an entire JSON file atomically.
* :func:`atomic_write_jsonl` -- replace an entire JSONL (line-delimited JSON)
  file atomically, serializing one object per line.
* :func:`write_namespace_json` -- read-merge-write a single key of a shared
  multi-namespace JSON file atomically, preserving other top-level keys.

All three writers share one pipeline (:func:`_atomic_replace`): sibling temp
file + fsync + ``os.replace`` (POSIX atomic rename), which was previously
duplicated -- with subtle, inconsistent differences -- across:

  - external_llm/agent/checkpoint_store.py          (whole-file, index)
  - external_llm/editor/learning/strategy_state.py  (namespace merge)

``os.replace`` is atomic on the same filesystem: readers see either the old
file or the fully-written new one, never a truncated/partial file. The temp
file is created in the SAME directory as the target (so the rename stays on one
filesystem) and is always removed on failure.

These helpers do NOT serialize concurrent read-modify-write cycles across
processes; callers needing that should additionally hold
:func:`external_llm.common.file_lock.cross_process_flock`. The atomic rename
alone already prevents the crash-corruption (truncation) class of bugs that
motivated this module.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from typing import Any, TextIO

from .repo_files import invalidate_for_written_path

logger = logging.getLogger(__name__)

# ── Crash-leftover sweep ──────────────────────────────────────────────────
# The ``except BaseException: os.unlink(tmp_path)`` in each writer covers the
# EXCEPTION path only. A SIGKILL, an OOM kill or a power loss runs no Python
# handler at all, so the half-written ``.atomic_*.tmp`` survives forever —
# observed in the wild as a 96 MB orphan in .asicode/vector_cache/ left by an
# interrupted 124 MB metadata dump, still present a day later because nothing
# in the codebase ever looked for one.
#
# Sweep once per directory per process: orphans are created by a process that
# already died, so re-scanning on every write would be pure overhead.
_ATOMIC_TMP_PREFIX = ".atomic_"
# Generous enough that a legitimate in-flight write is never a candidate (the
# largest dump in this repo takes ~2 s), small enough to reclaim promptly.
_STALE_TMP_AGE_S = 3600.0
_swept_dirs: set[str] = set()


def sweep_stale_temp_files(base_dir: str, max_age_s: float = _STALE_TMP_AGE_S) -> int:
    """Delete ``.atomic_*`` leftovers in *base_dir* older than *max_age_s*.

    Returns the number of files removed. Never raises: a sweep failure must not
    prevent the write it was called from.

    Age-gated rather than unconditional because a *concurrent* writer in
    another process has its own live temp file in this directory, and deleting
    that would corrupt its rename. Nothing legitimate holds one for an hour.
    """
    removed = 0
    now = time.time()
    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if not entry.name.startswith(_ATOMIC_TMP_PREFIX):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if (now - st.st_mtime) < max_age_s:
                        continue
                    os.unlink(entry.path)
                    removed += 1
                    logger.info(
                        "Reclaimed stale atomic-write leftover: %s (%.1f MB, age %.1f h)",
                        entry.path,
                        st.st_size / 1e6,
                        (now - st.st_mtime) / 3600.0,
                    )
                except OSError as exc:
                    # Raced with another sweeper, or not ours to remove.
                    logger.debug("Could not reclaim %s: %s", entry.path, exc)
    except OSError as exc:
        logger.debug("Stale-temp sweep skipped for %s: %s", base_dir, exc)
    return removed


def _sweep_once(base_dir: str) -> None:
    """Run :func:`sweep_stale_temp_files` at most once per directory per process."""
    key = os.path.abspath(base_dir)
    if key in _swept_dirs:
        return
    _swept_dirs.add(key)
    sweep_stale_temp_files(base_dir)


def _atomic_replace(
    path: Any,
    suffix: str,
    write_body: Callable[[Any], None],
    *,
    finalize: Callable[[str, str], None] | None = None,
    binary: bool = False,
) -> None:
    """Shared crash-safe write pipeline behind every public atomic writer.

    Sibling temp file (same directory, so the rename stays on one filesystem)
    -> *write_body* -> flush + fsync -> optional *finalize* (tmp_path, target)
    -> ``os.replace`` -> repo-cache invalidation.  On ANY failure the temp file
    is removed and the exception is re-raised, so the target is never left
    truncated/partial if the process is interrupted mid-write (SIGKILL, disk
    full, power loss).  Creates the parent directory if missing.

    Args:
        path: Target file path (``str`` or :class:`~pathlib.Path`).
        suffix: Suffix for the temp file (``".tmp"``, ``".jsonl"``, ...).
        write_body: Serializes the payload into the open temp handle.
        finalize: Optional post-write hook called with ``(tmp_path, target)``
            before the rename (e.g. mode preservation in
            :func:`atomic_write_text`).
        binary: Open the temp in binary mode (``"wb"``) and pass a bytes
            payload to *write_body* (used by :func:`atomic_write_bytes`).
    """
    file_path = os.fspath(path)
    base_dir = os.path.dirname(file_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    _sweep_once(base_dir)  # reclaim leftovers from a previously killed process
    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=".atomic_", suffix=suffix)
    try:
        if binary:
            with os.fdopen(fd, "wb") as fh:
                write_body(fh)
                fh.flush()
                os.fsync(fh.fileno())  # durability: ensure data is on disk before rename
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                write_body(fh)
                fh.flush()
                os.fsync(fh.fileno())  # durability: ensure data is on disk before rename
        if finalize is not None:
            finalize(tmp_path, file_path)
        os.replace(tmp_path, file_path)
        invalidate_for_written_path(file_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def atomic_write_json(
    path: Any,
    data: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default: Any = None,
) -> None:
    """Atomically write ``data`` as JSON to ``path`` (whole-file replacement).

    Uses the shared :func:`_atomic_replace` pipeline — sibling temp file +
    fsync + atomic rename — so the target is never left truncated/partial if
    the process is interrupted mid-write. Creates the parent directory if
    missing.

    Args:
        path: Target JSON file path (``str`` or :class:`~pathlib.Path`).
        data: JSON-serializable object.
        indent: Passed to :func:`json.dump` (default 2).
        ensure_ascii: Passed to :func:`json.dump` (default False).
        default: Passed to :func:`json.dump` as the non-serializable fallback
            (default None -> raise on non-serializable types).

    Raises:
        OSError/IOError: on write or rename failure (temp file is cleaned up).
    """

    # NOTE: intentional near-duplicate of atomic_write_text's wrapper below
    # (structural scanner: sim 0.85 shared-prefix -- both are the 3-line
    # "def _write_body + _atomic_replace(path, .tmp, ...)" skeleton). Merging
    # further would push format-specific serialization (json.dump kwargs vs
    # plain write) into the shared core as knobs, for zero behavior gain.
    def _write_body(fh: TextIO) -> None:
        json.dump(data, fh, indent=indent, ensure_ascii=ensure_ascii, default=default)

    _atomic_replace(path, ".tmp", _write_body)


def atomic_write_jsonl(
    path: Any,
    records: Any,
    *,
    ensure_ascii: bool = False,
    default: Any = None,
) -> None:
    """Atomically write ``records`` as JSONL to ``path`` (whole-file replacement).

    Like :func:`atomic_write_json`, but for line-delimited JSON where each line
    is a separate JSON object and the file as a whole is *not* a single JSON
    value (e.g. ``run_history.jsonl``). Each record is serialized on its own
    line via the shared :func:`_atomic_replace` pipeline.

    Args:
        path: Target JSONL file path (``str`` or :class:`~pathlib.Path`).
        records: Iterable of JSON-serializable objects, each written on its own
            line. Consumed exactly once (a generator is fine).
        ensure_ascii: Passed to :func:`json.dumps` (default False).
        default: Passed to :func:`json.dumps` as the non-serializable fallback
            (default None -> raise on non-serializable types).

    Raises:
        OSError/IOError: on write or rename failure (temp file is cleaned up).
    """

    def _write_body(fh: TextIO) -> None:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=ensure_ascii, default=default) + "\n")

    _atomic_replace(path, ".jsonl", _write_body)


def atomic_write_text(path: Any, content: str, *, mode: Any = None) -> None:
    """Atomically replace ``path`` with ``content`` (UTF-8 text, whole-file).

    Uses the shared :func:`_atomic_replace` pipeline — sibling temp file +
    fsync + atomic rename — so a crash / SIGKILL / disk-full mid-write never
    leaves the target truncated or partially written. Creates the parent
    directory if missing, and the temp is always removed on failure.

    This is the plain-text analogue of :func:`atomic_write_json` and a faithful
    drop-in for ``open(path, "w")``-then-``write``: callers that rewrite a file's
    full contents (e.g. ``modify_symbol``, ``write_plan`` direct-write, output
    normalization, quality-gate auto-fix/revert) get crash-safety without giving
    up the truncating-write semantics. Writing via ``open(path, "w")`` truncates
    the target BEFORE the new bytes land, so an interrupt between open and
    write-completion corrupts the file; this helper closes that window.

    Permission handling:

    * Existing target — the original mode (exec bit, group/world perms) is
      preserved, so an executable script or a shared file keeps its bits.
    * New target — the mode mirrors ``open(path, "w")``: ``0o666 & ~umask``. Pass
      ``mode`` to force specific bits (e.g. ``0o600`` for a secrets file)
      regardless of the process umask.

    Does NOT serialize concurrent read-modify-write across processes; callers
    needing that should additionally hold
    :func:`external_llm.common.file_lock.cross_process_flock`.

    Args:
        path: Target file path (``str`` or :class:`~pathlib.Path`).
        content: Full replacement text (UTF-8).
        mode: Optional permission bits for a NEWLY CREATED target (ignored when
            the target already exists). When None, a new file gets
            ``0o666 & ~umask`` to match ``open(path, "w")``.

    Raises:
        OSError/IOError: on write or rename failure (temp file is cleaned up).
    """

    # Same intentional wrapper skeleton as atomic_write_json's (see the note
    # there): identical structure, different body (plain write vs json.dump).
    def _write_body(fh: TextIO) -> None:
        fh.write(content)

    def _finalize(tmp_path: str, target_path: str) -> None:
        if os.path.exists(target_path):
            # Existing target: preserve its mode (exec bit, group/world perms):
            # see the permission note in the docstring above.
            os.chmod(tmp_path, os.stat(target_path).st_mode)
        elif mode is None:
            # New target: mirror open(path,"w") = 0o666 & ~umask so this is a
            # faithful drop-in for the truncating write it replaces.
            _um = os.umask(0)
            os.umask(_um)
            os.chmod(tmp_path, 0o666 & ~_um)
        else:
            os.chmod(tmp_path, mode)

    _atomic_replace(path, ".tmp", _write_body, finalize=_finalize)


def atomic_write_bytes(path: Any, data: bytes, *, mode: Any = None) -> None:
    """Atomically replace ``path`` with ``data`` (raw bytes, whole-file).

    The bytes analogue of :func:`atomic_write_text` — sibling temp file +
    fsync + atomic rename — so a crash / SIGKILL / disk-full mid-write never
    leaves the target truncated or partially written. Creates the parent
    directory if missing, and the temp is always removed on failure.

    Use when the caller already holds the exact bytes (e.g. text re-encoded
    with the file's detected non-UTF-8 encoding) and must not round-trip them
    through a UTF-8 text writer, which would alter every non-ASCII byte.
    Encoding is the CALLER's responsibility: unlike :func:`atomic_write_text`
    there is no implicit encode, so an encode failure happens before any file
    I/O and never touches the target.

    Permission handling mirrors :func:`atomic_write_text`:

    * Existing target — the original mode (exec bit, group/world perms) is
      preserved, so an executable script or a shared file keeps its bits.
    * New target — ``0o666 & ~umask`` (pass ``mode`` to force specific bits).

    Args:
        path: Target file path (``str`` or :class:`~pathlib.Path`).
        data: Full replacement payload (raw bytes).
        mode: Optional permission bits for a NEWLY CREATED target (ignored when
            the target already exists).

    Raises:
        OSError/IOError: on write or rename failure (temp file is cleaned up).
    """

    # Same intentional wrapper skeleton as atomic_write_text's (see the note
    # there): identical structure, different body (raw bytes vs text).
    def _write_body(fh) -> None:
        fh.write(data)

    def _finalize(tmp_path: str, target_path: str) -> None:
        if os.path.exists(target_path):
            # Existing target: preserve its mode (exec bit, group/world perms).
            os.chmod(tmp_path, os.stat(target_path).st_mode)
        elif mode is None:
            # New target: mirror open(path,"wb") = 0o666 & ~umask so this is a
            # faithful drop-in for the truncating write it replaces.
            _um = os.umask(0)
            os.umask(_um)
            os.chmod(tmp_path, 0o666 & ~_um)
        else:
            os.chmod(tmp_path, mode)

    _atomic_replace(path, ".tmp", _write_body, finalize=_finalize, binary=True)


def write_namespace_json(
    path: Any,
    namespace: str,
    value: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Any = None,
) -> None:
    """Atomically merge ``value`` under ``namespace`` in the JSON file at
    ``path``, preserving other top-level keys.

    For components that share one JSON file across namespaces (e.g.
    ``failure_memory.json`` holds both ``"graph"`` and ``"repair"`` namespaces).
    Performs a full read-merge-atomic-write so one writer's crash can never
    truncate another namespace's data.

    If the existing file is missing or its top level is not an object, a fresh
    object is started.

    Args:
        path: Target JSON file path (``str`` or :class:`~pathlib.Path`).
        namespace: Top-level key to write.
        value: Value to store under ``namespace``.
        indent, ensure_ascii, default: passed through to :func:`json.dump`.

    Raises:
        OSError/IOError: on write or rename failure.
        json.JSONDecodeError: if the existing file exists but is corrupt.
    """
    file_path = os.fspath(path)
    data: dict = {}
    if os.path.isfile(file_path):
        with open(file_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            data = loaded
    data[namespace] = value
    atomic_write_json(
        file_path,
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
    )
