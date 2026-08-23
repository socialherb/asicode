"""
Semantic lint using ruff F-codes. Graceful skip if ruff unavailable.

Phase 1 keystone: ruff F401/F811/F821/F841 findings against snapshot-diff
(on-disk content vs pre-write content). Designed as a soft signal — warnings
only, no rollbacks. Pre-existing lint debt is filtered out via pre/post diff.
"""

import json
import logging
import os
import subprocess
from collections import OrderedDict
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# ruff exit codes: 0 = clean, 1 = findings, other = error
_RUFF_EXIT_OK = {0, 1}


_RUFF_AVAILABLE: bool | None = None


def _check_ruff_available() -> bool:
    """Ruff availability check. Cache only the positive result.

    Negative results are deliberately NOT cached: this agent installs
    tools at runtime (playwright bootstrap, etc.), so ruff may appear
    mid-session. Re-probing a missing binary is sub-millisecond, so an
    uncached negative costs nothing per call.
    """
    global _RUFF_AVAILABLE
    if _RUFF_AVAILABLE is True:
        return True
    try:
        proc = subprocess.run(
            ["ruff", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            _RUFF_AVAILABLE = True
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as e:
        logger.debug("ruff availability probe failed: %s", e)
    return False


def ruff_findings(
    content: str,
    path: str | None = None,
    select: str = "F401,F811,F821,F841",
) -> list[dict]:
    """Run ruff --select F... on content via stdin. Returns list of findings.

    Each finding dict: {"code": str, "line": int, "message": str}
    Returns [] on any failure (ruff missing, parse error, timeout, etc.)

    Uses --isolated to ignore user/project config files.

    Results are memoized (LRU, maxsize=64, key = path/select/content) so a
    session that repeatedly edits the same files — where this turn's
    pre-content is byte-identical to the last turn's post-content — reuses
    the previous scan instead of re-spawning ruff. Only non-empty results
    are cached: an empty result may mean ruff is missing, and ruff can be
    installed mid-session (see _check_ruff_available).
    """
    if not _check_ruff_available():
        return []

    key = (path or "", select, content)
    cached = _findings_cache_lookup(key)
    if cached is not None:
        return cached

    cmd = [
        "ruff",
        "check",
        "--isolated",
        f"--select={select}",
        "--output-format=json",
        "-",  # read from stdin
    ]
    if path:
        cmd.extend(["--stdin-filename", path])

    try:
        result = subprocess.run(
            cmd,
            input=content,
            capture_output=True,
            timeout=15,
            text=True,
            check=False,
        )
        if result.returncode not in _RUFF_EXIT_OK:
            logger.debug("ruff returned %d: %s", result.returncode, result.stderr[:200])
            return []
        if not result.stdout.strip():
            return []
        raw_findings = json.loads(result.stdout)

        normalized = [_normalize_finding(f) for f in raw_findings]
        if normalized:
            _findings_cache_store(key, normalized)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        logger.debug("ruff error: %s", e)
        return []
    else:
        return normalized


def ruff_findings_many(
    paths: Sequence[str],
    select: str = "F401,F811,F821,F841",
) -> dict[str, list[dict]]:
    """Run ruff ONCE over multiple on-disk files. Returns {path: findings}.

    Same semantics as ruff_findings (--isolated, same select, same
    normalization, []-per-file on any failure). Paths that do not exist or
    do not end in ``.py`` are skipped. This is the batching counterpart to
    ruff_findings: spawning one ruff process per file costs ~8ms each in
    process startup alone (measured: 10 files = 66ms one-by-one vs 9ms in
    one call), which dominates per-file linting of a multi-file write.

    Result paths are the exact strings passed in (ruff echoes them back in
    the JSON ``filename`` field), so callers can key on their own path form.
    """
    if not _check_ruff_available():
        return {}
    py_paths = [p for p in paths if p.endswith(".py") and os.path.exists(p)]
    if not py_paths:
        return {}

    cmd = [
        "ruff",
        "check",
        "--isolated",
        f"--select={select}",
        "--output-format=json",
        *py_paths,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            text=True,
            check=False,
        )
        if result.returncode not in _RUFF_EXIT_OK:
            logger.debug("ruff returned %d: %s", result.returncode, result.stderr[:200])
            return {}
        by_path: dict[str, list[dict]] = {p: [] for p in py_paths}
        if not result.stdout.strip():
            return by_path
        for f in json.loads(result.stdout):
            by_path.setdefault(f.get("filename", ""), []).append(_normalize_finding(f))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        logger.debug("ruff error: %s", e)
        return {}
    else:
        return by_path


def _normalize_finding(f: dict) -> dict:
    """Normalize a raw ruff JSON finding to {"code", "line", "message"}.

    ruff 0.x uses "location.row", ruff 1.x may differ.
    """
    loc = f.get("location", f)
    line = loc.get("row", loc.get("line", 0))
    return {
        "code": f.get("code", ""),
        "line": int(line),
        "message": f.get("message", ""),
    }


# LRU cache shared by ruff_findings (stdin) — NOT by ruff_findings_many,
# which reads the disk and must see the current on-disk state.
_FINDINGS_CACHE: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_FINDINGS_CACHE_MAX = 64


def _findings_cache_lookup(key: tuple) -> list[dict] | None:
    cached = _FINDINGS_CACHE.get(key)
    if cached is None:
        return None
    _FINDINGS_CACHE.move_to_end(key)
    return list(cached)  # copy — callers may mutate the result


def _findings_cache_store(key: tuple, findings: list[dict]) -> None:
    _FINDINGS_CACHE[key] = list(findings)
    _FINDINGS_CACHE.move_to_end(key)
    while len(_FINDINGS_CACHE) > _FINDINGS_CACHE_MAX:
        _FINDINGS_CACHE.popitem(last=False)
