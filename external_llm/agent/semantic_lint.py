"""
Semantic lint using ruff F-codes. Graceful skip if ruff unavailable.

Phase 1 keystone: ruff F401/F811/F821/F841 findings against snapshot-diff
(on-disk content vs pre-write content). Designed as a soft signal — warnings
only, no rollbacks. Pre-existing lint debt is filtered out via pre/post diff.
"""

import json
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# ruff exit codes: 0 = clean, 1 = findings, other = error
_RUFF_EXIT_OK = {0, 1}



_RUFF_AVAILABLE: Optional[bool] = None


def _check_ruff_available() -> bool:
    """One-time ruff availability check. Cache result."""
    global _RUFF_AVAILABLE
    if _RUFF_AVAILABLE is None:
        try:
            subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                timeout=5,
            )
            _RUFF_AVAILABLE = True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            _RUFF_AVAILABLE = False
    return _RUFF_AVAILABLE


def ruff_findings(
    content: str,
    path: Optional[str] = None,
    select: str = "F401,F811,F821,F841",
) -> list[dict]:
    """Run ruff --select F... on content via stdin. Returns list of findings.

    Each finding dict: {"code": str, "line": int, "message": str}
    Returns [] on any failure (ruff missing, parse error, timeout, etc.)

    Uses --isolated to ignore user/project config files.
    """
    if not _check_ruff_available():
        return []

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
        )
        if result.returncode not in _RUFF_EXIT_OK:
            logger.debug("ruff returned %d: %s", result.returncode, result.stderr[:200])
            return []
        if not result.stdout.strip():
            return []
        raw_findings = json.loads(result.stdout)

        # Normalize: ruff 0.x uses "location.row", ruff 1.x may differ
        normalized: list[dict] = []
        for f in raw_findings:
            loc = f.get("location", f)
            line = loc.get("row", loc.get("line", 0))
            normalized.append({
                "code": f.get("code", ""),
                "line": int(line),
                "message": f.get("message", ""),
            })
        return normalized
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        logger.debug("ruff error: %s", e)
        return []
