"""exploration_state.py — Consolidated exploration/bandit state persistence.

All exploration-related state is consolidated into a single JSON file at
``~/.asicode/learning/exploration.json``.  Each component (CandidateMemory,
RewardNormalizer, RunStore) reads/writes its own namespace.

Namespaces
    candidate_memory      dict  repo_id → {goal_key: {file: hit_count}}
    reward_normalizer     dict  repo_id → {n, mean, M2}
    exploration_runs      dict  repo_id → [run records]
    weights               dict  repo_id → {graph, source, depth, ...}
    weights_history       dict  repo_id → [{graph, source, ...}]
    strategy_history      dict  repo_id → [{strategy, layer, success, ...}]
    joint_strategy_history dict repo_id → [{l1, l2, success, ...}]
    source_weights        dict  repo_id → {source_name: weight}
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_EXPLORATION_PATH = os.path.join(
    os.path.expanduser("~"), ".asicode", "learning", "exploration.json",
)
_EXPLORATION_DIR = os.path.dirname(_EXPLORATION_PATH)


def get_path() -> str:
    """Return the absolute path to the consolidated exploration state file."""
    return _EXPLORATION_PATH


def read_namespace(namespace: str, path: str = "") -> Optional[Any]:
    """Read a single namespace from the consolidated exploration file.

    Args:
        namespace: namespace key to read.
        path: optional custom path; uses default when empty.

    Returns the stored value or ``None`` when absent / on error.
    """
    file_path = path or _EXPLORATION_PATH
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get(namespace)
    except Exception:
        logger.debug("exploration_state: read_namespace(%s) failed", namespace, exc_info=True)
        return None


def read_namespaces_by_prefix(prefix: str, path: str = "") -> Dict[str, Any]:
    """Return all namespaces whose key starts with ``prefix``.

    Useful for cross-repo scans where data is stored as e.g.
    ``"exploration_runs/repo1"``, ``"exploration_runs/repo2"``.

    Returns a dict mapping full namespace key → value.
    Returns empty dict when file does not exist or on error.
    """
    file_path = path or _EXPLORATION_PATH
    if not os.path.isfile(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: v for k, v in data.items() if k.startswith(prefix)}
    except Exception:
        logger.debug(
            "exploration_state: read_namespaces_by_prefix(%s) failed", prefix, exc_info=True,
        )
        return {}


def write_namespace(namespace: str, value: Any, path: str = "") -> bool:
    """Atomically write one namespace into the consolidated exploration file.

    Args:
        namespace: namespace key to write.
        value: value to store under the namespace.
        path: optional custom path; uses default when empty.

    Reads existing data, merges ``data[namespace] = value``, and atomically
    rewrites via tempfile + ``os.replace``.  Other namespaces are preserved.
    Returns ``True`` on success, ``False`` on failure (never raises).
    """
    try:
        file_path = path or _EXPLORATION_PATH
        base_dir = os.path.dirname(file_path) or _EXPLORATION_DIR
        os.makedirs(base_dir, exist_ok=True)

        data: dict = {}
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        data[namespace] = value

        fd, tmp_path = tempfile.mkstemp(
            dir=base_dir, prefix=".exploration_state_", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, file_path)
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        logger.debug(
            "exploration_state: write_namespace(%s) failed", namespace, exc_info=True,
        )
        return False
