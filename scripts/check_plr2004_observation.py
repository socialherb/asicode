#!/usr/bin/env python3
"""Observe PLR2004 (magic-value-comparison) drift — REPORT-ONLY, never gates.

Usage:
    python scripts/check_plr2004_observation.py             # report drift vs baseline
    python scripts/check_plr2004_observation.py --write-baseline  # re-snapshot baseline

The 1836-site backlog (int/float literal comparisons: exit codes, HTTP
statuses, timeouts, test expectations) is intentionally NOT converted — a
manual sweep of that size would cost far more than it returns, and int/float
exemption would silently neuter the rule. Instead the rule stays unselected
in pyproject.toml while this script tracks NET-NEW drift on every CI run.

Baseline key: ``<file_path>::<line>``. This observation hook always exits 0 —
it reports a trend, it does not block. Re-baseline deliberately with
``--write-baseline`` (e.g. after a rule-level decision), not as a reflex.

Contrast with the *gating* baseline-diff checks (check_f401_no_new.py etc.):
those exit non-zero on net-new violations. This one exists to keep the
decision "hold PLR2004, watch it" honest — a blind ignore would let the
backlog grow undetected until the next audit.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "plr2004_baseline.txt"


def _scan() -> set[str]:
    """Current PLR2004 sites as ``<path>::<line>`` keys (repo-relative).

    --no-cache: a stale ruff cache (parallel-session edits) would silently
    under-report — for a trend monitor that is the one failure mode that
    matters.
    """
    try:
        result = subprocess.run(
            ["ruff", "check", "--no-cache", "--select=PLR2004", "--output-format=concise", "."],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("❌ ruff PLR2004 scan timed out after 300s — observation degraded for this run.", file=sys.stderr)
        return set()
    keys: set[str] = set()
    for line in result.stdout.splitlines():
        if "PLR2004" not in line:
            continue
        parts = line.split(":", 3)
        if len(parts) >= 3:
            keys.add(f"{parts[0].strip()}::{parts[1].strip()}")
    return keys


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {ln.strip() for ln in BASELINE.read_text(encoding="utf-8").splitlines() if ln.strip()}


def main() -> int:
    if "--write-baseline" in sys.argv:
        cur = _scan()
        BASELINE.write_text("\n".join(sorted(cur)) + ("\n" if cur else ""), encoding="utf-8")
        print(f"plr2004_baseline.txt written: {len(cur)} sites")
        return 0

    cur = _scan()
    base = _load_baseline()
    new = cur - base
    resolved = base - cur
    print(f"PLR2004 observation: total {len(cur)} sites (baseline {len(base)})")
    for key in sorted(new):
        print(f"  ⬆ NEW  {key}")
    for key in sorted(resolved):
        print(f"  ⬇ gone {key} (re-baseline with --write-baseline when intended)")
    if not new and not resolved:
        print("  no drift vs baseline — backlog stable")
    print("report-only observation — exit 0 (trend monitor, not a gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
