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

_STRATEGY_STATE_PATH = os.path.join(
    os.path.expanduser("~"), ".asicode", "learning", "strategy_state.json",
)


def get_path() -> str:
    """Return the absolute path to the consolidated strategy state file."""
    return _STRATEGY_STATE_PATH


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
    file_path = path or _STRATEGY_STATE_PATH
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
        file_path = path or _STRATEGY_STATE_PATH
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
    Returns ``True`` on success, ``False`` on failure (never raises).
    """
    try:
        file_path = path or _STRATEGY_STATE_PATH
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
            for ns, value in namespace_value_map.items():
                data[ns] = value
            # Single write
            atomic_write_json(file_path, data, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception:
        logger.debug("strategy_state: batch_write_namespaces failed", exc_info=True)
        return False
