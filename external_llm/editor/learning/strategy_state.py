"""strategy_state.py — Namespace-based strategy state persistence.

All strategy-related state is consolidated into a single JSON file at
``~/.asicode/learning/strategy_state.json``.  Each module reads/writes its
own namespace within that file using the same load-modify-save pattern that
``RepairMemory`` / ``GraphFailureMemory`` already established for
``failure_memory.json``.

Namespaces
    experience_store       list    ExperienceStore records
    primitive_learning     dict    PrimitiveLearningStore data
    transferable_knowledge dict    Shared policy knowledge (cross-model)
    policy/{model}         dict    PolicyLearner state (model-keyed)
    weights/{model}        dict    WeightLearner state (model-keyed)
    adaptive_hub/{model}   dict    AdaptiveLearnerHub state (model-keyed)
    execution_state/{model} dict   ExecutionLearner state (model-keyed)
    fallback_scores        dict    FallbackScoreStore strategy scores
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from pathlib import Path

from external_llm.common.atomic_io import (
    write_namespace_json,
    atomic_write_json,
)
from external_llm.common.file_lock import cross_process_flock

logger = logging.getLogger(__name__)

# Overridable via ``ASICODE_STRATEGY_STATE``, matching ``ASICODE_RUNS_DIR`` /
# ``ASICODE_WRITE_TOOL_FAILURE_LOG`` / ``ASICODE_CONTEXT_OVERRIDE_CACHE``.
#
# Redirecting this one path is enough for the sidecars too: ``_path_for`` derives
# them from ``os.path.dirname(_STRATEGY_STATE_PATH)``.
#
# An env var, not just a patchable attribute, because the writers that leaked
# were in test SUBPROCESSES — a child re-imports this module and rebuilds the
# constant, so patching it in the parent cannot reach them. Measured over a full
# suite run with a before/after snapshot of ~/.asicode: this file was CREATED
# carrying adaptive_hub Q-values and counts fabricated by tests, and the
# experience_store sidecar was rewritten, losing 17 of its FIFO-capped 200
# records to make room for 15 test ones.
_STRATEGY_STATE_PATH = os.environ.get("ASICODE_STRATEGY_STATE") or os.path.join(
    os.path.expanduser("~"), ".asicode", "learning", "strategy_state.json",
)

# Namespaces that live in their OWN file instead of the consolidated one.
#
# Every write here is a read-merge-atomic-write of the WHOLE file under a
# machine-global flock, so one bulky namespace taxes every other namespace's
# write. ``experience_store`` is 89 KB of the 123 KB file on a real profile and
# is written only from the disabled PLANNER lane, while ``adaptive_hub`` (3 KB)
# is written by the live agent loop every few seconds. Measured cost of one hub
# flush, same machine: 3.93 ms together vs 0.63 ms apart, and 15.71 ms vs
# 2.37 ms with four concurrent sessions serialising on the shared lock.
#
# Keyed by the namespace ROOT segment, so model-keyed forms ("weights/{model}")
# route by their family. Splitting also gives each file its own lock.
_NAMESPACE_FILES: dict[str, str] = {
    "experience_store": "experience_store.json",
}


def get_path() -> str:
    """Return the absolute path to the consolidated strategy state file."""
    return _STRATEGY_STATE_PATH


def _path_for(namespace: str, path: str = "") -> str:
    """File backing *namespace*.

    An explicit *path* disables routing entirely: those callers (tests, the
    exploration run-store, the fallback-score store) own a private file and
    expect every namespace they write to land in it.
    """
    if path:
        return path
    sidecar = _NAMESPACE_FILES.get(namespace.split("/", 1)[0])
    if not sidecar:
        return _STRATEGY_STATE_PATH
    return os.path.join(os.path.dirname(_STRATEGY_STATE_PATH), sidecar)


# Keyed by consolidated-file path, not a bare flag: tests point
# ``_STRATEGY_STATE_PATH`` at a tmpdir, and a process-wide bool would make the
# first test's migration suppress every later one's.
_migrated: set[str] = set()


def _migrate_split_namespaces() -> None:
    """Move routed namespaces out of the consolidated file, once per process.

    Existing installs have the bulky namespace inside strategy_state.json.
    Reads already fall back to the legacy location, so this is purely about
    making the hot file small — which is the entire point of the split, and
    would otherwise never happen for a user whose only writer of the routed
    namespace is the disabled lane.

    Lock order is main-file-then-sidecar and nothing acquires them in the other
    order. A crash between the two writes leaves the value in BOTH files; the
    sidecar wins on read, so the outcome is a stale duplicate rather than data
    loss, and the next run re-runs the move.
    """
    if _STRATEGY_STATE_PATH in _migrated:
        return
    _migrated.add(_STRATEGY_STATE_PATH)
    try:
        if not os.path.isfile(_STRATEGY_STATE_PATH):
            return
        with cross_process_flock(Path(f"{_STRATEGY_STATE_PATH}.lock")):
            with open(_STRATEGY_STATE_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return
            moved = {k: data[k] for k in list(data)
                     if k.split("/", 1)[0] in _NAMESPACE_FILES}
            if not moved:
                return
            for ns, value in moved.items():
                target = _path_for(ns)
                with cross_process_flock(Path(f"{target}.lock")):
                    write_namespace_json(target, ns, value, default=str)
                data.pop(ns, None)
            atomic_write_json(_STRATEGY_STATE_PATH, data, indent=2,
                              ensure_ascii=False, default=str)
        logger.info(
            "strategy_state: moved %s out of the consolidated file",
            ", ".join(sorted(moved)),
        )
    except Exception:
        logger.debug("strategy_state: split migration failed", exc_info=True)


def read_namespace(namespace: str, path: str = "") -> Optional[Any]:
    """Read a single namespace from the consolidated strategy state file.

    Args:
        namespace: namespace key to read.
        path: optional custom path; uses default when empty.

    Returns the stored value (whatever type the caller wrote) or ``None``
    when the namespace is absent, the file is missing, or a read error
    occurs.  Callers should treat ``None`` as "no data" and substitute
    their own default (``{}``, ``[]``, etc.).
    """
    file_path = _path_for(namespace, path)
    value = _read_from(file_path, namespace)
    if value is None and not path and file_path != _STRATEGY_STATE_PATH:
        # Routed namespace not in its own file yet: an install that predates
        # the split still has it in the consolidated one. Reading through keeps
        # the move invisible to callers and independent of when it runs.
        value = _read_from(_STRATEGY_STATE_PATH, namespace)
    return value


def _read_from(file_path: str, namespace: str) -> Optional[Any]:
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get(namespace)
    except Exception:
        logger.debug("strategy_state: read_namespace(%s) failed", namespace, exc_info=True)
        return None


def write_namespace(namespace: str, value: Any, path: str = "") -> bool:
    """Atomically write one namespace into the consolidated state file.

    Args:
        namespace: namespace key to write.
        value: value to store under the namespace.
        path: optional custom path; uses default when empty.

    Reads existing data, merges ``data[namespace] = value``, and atomically
    rewrites via tempfile + ``os.replace`` so the file is never left
    partially written.  Other namespaces are preserved.

    Returns ``True`` on success, ``False`` on failure (never raises).
    """
    try:
        if not path:
            _migrate_split_namespaces()
        file_path = _path_for(namespace, path)
        lock_path = Path(f"{file_path}.lock")
        with cross_process_flock(lock_path):
            write_namespace_json(file_path, namespace, value, default=str)
        return True
    except Exception:
        logger.debug("strategy_state: write_namespace(%s) failed", namespace, exc_info=True)
        return False


def batch_write_namespaces(
    namespace_value_map: dict[str, Any],
    path: str = "",
) -> bool:
    """Atomically write multiple namespaces in one read-merge-write cycle.

    Args:
        namespace_value_map: ``{namespace: value}`` pairs to write.
        path: optional custom path; uses default when empty.

    Like :func:`write_namespace` but batches *N* namespaces into a single
    file read and a single file write, preserving all other top-level keys.
    Namespaces routed to their own file (see ``_NAMESPACE_FILES``) are grouped
    per target, so a batch still costs one read+write per FILE rather than one
    per namespace — and never writes a routed namespace into the shared file,
    which would resurrect the very blob the split removes.

    Returns ``True`` on success, ``False`` on failure (never raises).
    """
    try:
        if not path:
            _migrate_split_namespaces()
        groups: dict[str, dict[str, Any]] = {}
        for ns, value in namespace_value_map.items():
            groups.setdefault(_path_for(ns, path), {})[ns] = value
        for file_path, group in groups.items():
            lock_path = Path(f"{file_path}.lock")
            with cross_process_flock(lock_path):
                # Single read
                data: dict = {}
                if os.path.isfile(file_path):
                    with open(file_path, encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        data = loaded
                # Multi-update
                for ns, value in group.items():
                    data[ns] = value
                # Single write
                atomic_write_json(file_path, data, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception:
        logger.debug("strategy_state: batch_write_namespaces failed", exc_info=True)
        return False
