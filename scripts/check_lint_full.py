#!/usr/bin/env python3
"""Check the repo is clean under ruff's FULL configured rule set (``ruff check .``).

Zero-tolerance gate with **no baseline** — the default select set in
pyproject.toml IS the gate.  A violation here means new code slipped past both
review and the rule-specific baseline-diff hooks (F821/F401/F811/F823), so it
must be fixed, never baselined.

Why this gate exists
--------------------
The older hooks are baseline-diff *by design*: F821/F401/F811/silent-except
grandfather pre-existing violations so a PR only fails on NET-NEW ones.  But a
full ``ruff check .`` under the default rule set was long impossible — the
comment in pyproject.toml recorded a ~1377-violation backlog.  That backlog was
cleared rule-by-rule over time; by 2026-08-07 only **2** violations remained,
both in code written after the last clearing round (a PLW1510 ``subprocess.run``
missing explicit ``check=`` in a test, and a RUF003 EN DASH in a comment), and
neither gate caught them.  Full-scan enforcement is what keeps that from
happening again: the default select set is now the floor, and this script is
the first line that catches anything it flags.

What it does NOT cover
----------------------
- Preview rules: the gate runs the STABLE default set only — no ``--select``
  override, because an override would silently disable everything else (the
  RUF100 audit trap: ``ruff check --select RUF100`` re-enables only RUF100 and
  misreports every other active rule's noqa as "non-enabled").

Usage:
    python scripts/check_lint_full.py
    python scripts/check_lint_full.py <file>.py ...  # check only given files

Explicit file args (pre-commit per-file mode) scan only those files — the
full-repo always_run scans were dropped from the hook config because they
created a multi-second window where pre-commit's run-start ``git diff`` vs
post-hook diff comparison false-positives on parallel-session writes.  No args
(lint.yml CI) still scans the whole repo.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    """Normalize explicit file args to repo-relative ``*.py`` paths.

    Returns ``None`` when no file args survive (or none were given) — the
    caller then scans the whole repo, preserving the no-args (lint.yml CI)
    behaviour.  pre-commit passes absolute paths; lint.yml passes none — both
    normalize to the same repo-relative key space as the ``.`` full scan.
    """
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(".py") and not rel.startswith(".."):
            out.append(rel)
    return out or None


def _get_current_errors(paths: list[str] | None = None) -> list[str]:
    # NO --select: the whole point is the default configured rule set.  An
    # override would disable every other active rule (see the RUF100 trap in
    # the docstring).
    # timeout= so a hung ruff can never stall the hook/CI forever. A gate must
    # FAIL on timeout (fail-closed), not silently pass on empty output — which
    # is why we do NOT use common/subprocess_utils.run_bounded_subprocess here
    # (that helper swallows timeouts into returncode=-9, a fail-open semantic).
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=concise"] + (paths or ["."]),
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            "❌ ruff full scan timed out after 180s — failing closed rather than risk a silent pass.", file=sys.stderr
        )
        sys.exit(1)
    except FileNotFoundError:
        print("❌ ruff not found on PATH — failing closed rather than silently passing.", file=sys.stderr)
        sys.exit(1)
    return [line for line in result.stdout.splitlines() if ": " in line]


def _get_format_errors(paths: list[str] | None = None) -> list[str]:
    """Check ``ruff format --check`` — zero-tolerance, no baseline.

    Formatting is now a hard gate: the 848-file backlog was cleared in one
    round (commits 3de8992d..dfa61d1a), so any file that would be reformatted
    is a NEW drift.  ``--diff`` output is reported so the developer sees
    exactly what would change, matching the ``ruff check`` concise-report
    philosophy.  Like ``_get_current_errors``, this runs with a hard timeout
    and fails closed.
    """
    try:
        result = subprocess.run(
            ["ruff", "format", "--check", "--diff"] + (paths or ["."]),
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            "❌ ruff format check timed out after 180s — failing closed rather than risk a silent pass.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print("❌ ruff not found on PATH — failing closed rather than silently passing.", file=sys.stderr)
        sys.exit(1)
    # --diff output: diff body goes to stdout; a nonzero exit code means at
    # least one file would be reformatted.  The per-file summary line on
    # stderr ("N file(s) would be reformatted") varies in wording between
    # single-file and full-repo modes, so key off returncode + the diff
    # headers ("+++ path") instead of parsing the summary text.
    if result.returncode == 0:
        return []
    out = []
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            out.append("Would reformat: " + line[4:].strip())
    if not out:
        # No diff headers (e.g. an empty diff) — fall back to the summary.
        for line in result.stderr.splitlines():
            if "would be reformatted" in line:
                out.append(line.strip())
    return out


def main() -> int:
    paths = _resolve_scan_paths([a for a in sys.argv[1:] if not a.startswith("--")])
    errors = _get_current_errors(paths)
    fmt_errors = _get_format_errors(paths)
    if not errors and not fmt_errors:
        print("✅ ruff check . — 0 violations under the full default rule set (no baseline)")
        if paths:
            print("✅ ruff format --check — 0 files would be reformatted")
        else:
            print("✅ ruff format --check (full repo) — 0 files would be reformatted")
        return 0

    if errors:
        print(f"❌ {len(errors)} violation(s) under the full default rule set:\n")
        for err in errors:
            print(f"  {err}")
        print(
            "\nThis gate has NO baseline — the default select set in pyproject.toml is the floor."
            "\nA violation here means code slipped past review AND the baseline-diff hooks."
            "\nFix the code — do NOT add an ignore entry or a --select override."
        )
    if fmt_errors:
        print(f"\n❌ {len(fmt_errors)} file(s) would be reformatted by `ruff format` (zero-tolerance):\n")
        for f in fmt_errors[:50]:
            print(f"  {f}")
        if len(fmt_errors) > 50:
            print(f"  … and {len(fmt_errors) - 50} more")
        print(
            "\nThe 848-file format backlog was cleared in one round; any drift is NEW."
            "\nRun `ruff format <file>` (or `ruff format .` for the whole repo) and commit."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
