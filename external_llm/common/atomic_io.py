"""atomic_io.py -- Atomic JSON persistence helpers (single source of truth).

Provides three primitives for crash-safe JSON writes:

* :func:`atomic_write_json` -- replace an entire JSON file atomically.
* :func:`atomic_write_jsonl` -- replace an entire JSONL (line-delimited JSON)
  file atomically, serializing one object per line.
* :func:`write_namespace_json` -- read-merge-write a single key of a shared
  multi-namespace JSON file atomically, preserving other top-level keys.

Both use the tempfile + ``os.replace`` (POSIX atomic rename) pattern that was
previously duplicated -- with subtle, inconsistent differences -- across:

  - external_llm/agent/session_state.py            (whole-file)
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

import json
import os
import tempfile
from typing import Any


def atomic_write_json(
    path: Any,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Any = None,
) -> None:
    """Atomically write ``data`` as JSON to ``path`` (whole-file replacement).

    Writes to a sibling temp file then ``os.replace``-s it into place, so the
    target is never left truncated/partial if the process is interrupted
    mid-write (SIGKILL, disk full, power loss). Creates the parent directory if
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
    file_path = os.fspath(path)
    base_dir = os.path.dirname(file_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=".atomic_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                data, fh, indent=indent, ensure_ascii=ensure_ascii, default=default,
            )
            fh.flush()
            os.fsync(fh.fileno())  # durability: ensure data is on disk before rename
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
    line into a sibling temp file, which is then ``os.replace``-d into place, so
    a crash mid-write (SIGKILL, disk full, power loss) never leaves the target
    truncated/partial. Creates the parent directory if missing.

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
    file_path = os.fspath(path)
    base_dir = os.path.dirname(file_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=".atomic_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(
                    json.dumps(rec, ensure_ascii=ensure_ascii, default=default) + "\n"
                )
            fh.flush()
            os.fsync(fh.fileno())  # durability: ensure data is on disk before rename
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: Any, content: str, *, mode: Any = None) -> None:
    """Atomically replace ``path`` with ``content`` (UTF-8 text, whole-file).

    Sibling temp file → ``os.replace`` (POSIX atomic rename): a crash / SIGKILL /
    disk-full mid-write never leaves the target truncated or partially written.
    The temp is created in the SAME directory as the target (so the rename stays
    on one filesystem), the parent directory is created if missing, and the temp
    is always removed on failure.

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
    file_path = os.fspath(path)
    base_dir = os.path.dirname(file_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=".atomic_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())  # durability: data on disk before rename
        if os.path.exists(file_path):
            # Existing target: preserve its mode (exec bit, group/world perms):
            # see the permission note in the docstring above.
            os.chmod(tmp_path, os.stat(file_path).st_mode)
        else:
            # New target: caller mode, or mirror open(path,"w") = 0o666 & ~umask
            # so this is a faithful drop-in for the truncating write it replaces.
            if mode is None:
                _um = os.umask(0)
                os.umask(_um)
                mode = 0o666 & ~_um
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
        file_path, data, indent=indent, ensure_ascii=ensure_ascii, default=default,
    )
