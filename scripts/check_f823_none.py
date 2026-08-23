#!/usr/bin/env python3
"""Check there are ZERO F823 (local variable referenced before assignment) errors.

Unlike its F401/F811/F821 siblings this gate has **no baseline** — the repo was
at zero when it was introduced, so there is nothing to grandfather and any hit
is a new bug. Keep it that way: F823 has no legitimate use, so a violation
should be fixed, never baselined.

Why this gate exists
--------------------
``webapp/ui/ui_tools.py`` shipped a function that assigned a module global
without declaring ``global``, making the name function-local throughout and
turning its first *read* into an ``UnboundLocalError``:

    _rg_dirty: bool = False          # module level

    def _save(...):
        if not _rg_dirty:            # <-- UnboundLocalError, every call
            return
        ...
        _rg_dirty = False            # <-- makes the whole name local

The raise landed *before* the function's ``try:``, so its ``except Exception``
never caught it: ``GET /stats/rg-fallback`` returned 500 and the ``atexit``
flush died. Ruff already reported it as F823 — but the gate only enforced
F821/F401/F811, so it sailed through review and CI.

Usage:
    python scripts/check_f823_none.py
    python scripts/check_f823_none.py <file>.py ...  # check only given files

Explicit file args (pre-commit per-file mode) scan only those files — the
full-repo always_run scans were dropped from the hook config because they
created a multi-second window where pre-commit's run-start `git diff` vs
post-hook diff comparison false-positives on parallel-session writes.  No
args (lint.yml CI) still scans the whole repo.
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
    # timeout= so a hung ruff can never stall the hook/CI forever. A gate must
    # FAIL on timeout (fail-closed), not silently pass on empty output — which is
    # why we do NOT use common/subprocess_utils.run_bounded_subprocess here (that
    # helper swallows timeouts into returncode=-9, a fail-open semantic).
    try:
        result = subprocess.run(
            ["ruff", "check", "--select=F823", "--output-format=concise"] + (paths or ["."]),
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            "❌ ruff F823 scan timed out after 180s — failing closed rather than risk a silent pass.", file=sys.stderr
        )
        sys.exit(1)
    except FileNotFoundError:
        print("❌ ruff not found on PATH — failing closed rather than silently passing.", file=sys.stderr)
        sys.exit(1)
    return [line for line in result.stdout.splitlines() if "F823" in line]


def main() -> int:
    paths = _resolve_scan_paths([a for a in sys.argv[1:] if not a.startswith("--")])
    errors = _get_current_errors(paths)
    if not errors:
        print("✅ No F823 errors (0 tolerated — this gate has no baseline)")
        return 0

    print(f"❌ {len(errors)} F823 error(s) — local variable referenced before assignment:\n")
    for err in errors:
        print(f"  {err}")
    print(
        "\nThis is almost always a missing `global <name>` (or `nonlocal <name>`)"
        "\ndeclaration in a function that assigns to a name from an outer scope."
        "\nAssigning anywhere in a function makes the name local for the WHOLE body,"
        "\nso every earlier read raises UnboundLocalError at runtime."
        "\n\nFix the code — do NOT add a baseline for F823."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
