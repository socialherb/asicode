#!/usr/bin/env bash
# cov.sh — xdist-safe coverage measurement wrapper.
#
# WHY this wrapper exists (measured 2026-08-16, coverage 7.14.3 + pytest 9.0):
#   1. `coverage run -m pytest` under the repo's default `-n auto` addopts
#      records ONLY the main process. pytest-xdist workers are separate
#      interpreters, so a plain coverage run reports ~0% (measured: 0% vs
#      62% with -n 0) — a silently wrong number.
#   2. `parallel = true` (pyproject [tool.coverage.run]) writes per-process
#      `.coverage.<suffix>` files, and a stale base `.coverage` shadows them
#      (`coverage report` reads the base and ignores suffix files), so a
#      re-run can report an old snapshot instead of the fresh data.
#
# Fix: COVERAGE_PROCESS_START=pyproject.toml makes the installed
# a1_coverage.pth (site-packages) inject coverage into EVERY python process
# (main, xdist workers, test-spawned subprocesses); each writes its own
# parallel data file into a PRIVATE tmp dir (no interference with parallel
# sessions' `.coverage*` files); `coverage combine` merges them and
# `coverage report` shows the true picture.
#
# pty child drivers (tests/unit/repl_stage2_child.py / repl_stage3_child.py,
# spawned via SpawnPtySession) are the one deliberate exception: pty_driver
# strips COVERAGE_PROCESS_START/CONFIG from the child env, so the .pth hook
# does NOT start there. The child starts its OWN Coverage instance instead,
# writing into the inherited COVERAGE_FILE (this script's private COV_DIR) or
# /tmp/covstage*-{pid}; the child passes data_suffix=True and pins config_file
# to the repo pyproject, so its data carries a parallel suffix beside the main
# process's files in COV_DIR whatever cwd it inherits (a base-name write would
# be REPLACED by `coverage combine`, losing those lines silently — measured
# 2026-09-12), so the `coverage combine` below merges the pty session's lines
# too (verified 2026-08-25: repl_impl.py 8% → 33% when the child data is
# included).
#
# Usage:
#   ./scripts/cov.sh [pytest args...]      # default: tests/unit -q
#   PYTHON=/path/to/python ./scripts/cov.sh ...
set -euo pipefail
cd "$(dirname "$0")/.."    # repo root regardless of invocation cwd

PYTHON="${PYTHON:-python3}"

# --- sanity: coverage + the .pth must be visible to $PYTHON ---
if ! "$PYTHON" -c 'import coverage' >/dev/null 2>&1; then
  echo "cov.sh: error: 'coverage' is not importable via $PYTHON" >&2
  echo "          pip install coverage   (or set PYTHON=/path/to/venv-python)" >&2
  exit 2
fi
SITE_PKGS=$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')
if [ ! -f "$SITE_PKGS/a1_coverage.pth" ]; then
  # Fresh environments (new dev machine, CI runner) get the .pth created
  # automatically — otherwise xdist workers and test-spawned subprocesses are
  # silently unmeasured (~0% under the repo's default `-n auto`). The hook is
  # inert unless COVERAGE_PROCESS_START/COVERAGE_PROCESS_CONFIG is set, so it
  # cannot perturb non-coverage runs. COV_NO_PTH_INSTALL=1 keeps the old
  # fail-with-hint behavior.
  if [ -n "${COV_NO_PTH_INSTALL:-}" ]; then
    echo "cov.sh: error: $SITE_PKGS/a1_coverage.pth not found (COV_NO_PTH_INSTALL=1)" >&2
    exit 2
  fi
  echo "cov.sh: creating $SITE_PKGS/a1_coverage.pth"
  cat > "$SITE_PKGS/a1_coverage.pth" <<'PTH'
import sys; exec('import os\n\nif os.getenv("COVERAGE_PROCESS_START") or os.getenv("COVERAGE_PROCESS_CONFIG"):\n try:\n  import coverage\n except:\n  pass\n else:\n  coverage.process_startup(slug="pth")')
PTH
  echo "          (xdist workers + subprocesses are measured via this hook;"
  echo "           inert when COVERAGE_PROCESS_START is unset)"
fi

# --- private data dir (override with COV_DIR for CI artifact staging) ---
# COV_DIR is normalized to an ABSOLUTE path before it reaches COVERAGE_FILE.
# coverage resolves a relative data-file path against the cwd of EVERY process
# it instruments, so a relative COV_DIR (lint.yml passes `COV_DIR=coverage-data`)
# scattered parallel data into whatever cwd a test-spawned child happened to
# have. Measured 2026-09-12: one `tests/unit` run left 99 data files / 5.3MB
# across 69 directories under pytest's tmp dirs — none of them readable by the
# `coverage combine` below (which resolves relative to THIS cwd, the repo root),
# so the gate silently lost those lines; the stray directory in the child's cwd
# also failed test_export_cli_args.py::test_real_cli_help_is_instant_and_clean
# in CI. `mkdir -p` must precede the `cd` because a fresh COV_DIR (CI's
# `coverage-data`) does not exist yet and `set -e` would abort. `cd … && pwd`
# rather than realpath: POSIX, and it keeps the caller's spelling so a relative
# COV_DIR still resolves against the repo root (lint.yml uploads
# `coverage-data/coverage`, workspace-relative).
COV_DIR="${COV_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/asicode-cov.XXXXXX")}"
mkdir -p "$COV_DIR"
COV_DIR="$(cd "$COV_DIR" && pwd)"
export COVERAGE_FILE="$COV_DIR/coverage"
# NOTE: COVERAGE_PROCESS_START is interpreted as a CONFIG FILE PATH by
# coverage.process_startup() (an absolute path is required; `=1` silently
# disables child coverage).
export COVERAGE_PROCESS_START="$PWD/pyproject.toml"

if [ "$#" -eq 0 ]; then
  set -- tests/unit -q
fi

echo "cov.sh: $PYTHON -m pytest $*   (data: $COV_DIR)"
PYTEST_RC=0
"$PYTHON" -m pytest "$@" || PYTEST_RC=$?
# Always combine+report, even on test failures — the report (and the
# fail_under gate) is the point of this wrapper, and the failed suite's
# coverage picture is the most useful diagnostic.
#
# Every step's rc is captured explicitly instead of being left to `set -e`:
# an aborted combine/report used to kill the script AT THAT LINE, masking the
# pytest rc (measured: lint run 34689020064 exited 1 from the report while the
# suite was 16425 passed / 0 failed) and skipping the footer, so a red log
# could not say WHICH step failed.
echo "cov.sh: pytest rc=$PYTEST_RC; combining parallel data files -> $COV_DIR/coverage"
COMBINE_RC=0
"$PYTHON" -m coverage combine || COMBINE_RC=$?

# COV_FAIL_UNDER overrides [tool.coverage.report] fail_under — e.g.
# COV_FAIL_UNDER=0 for release.yml's report-only run (a publish pipeline must
# not be blocked by the threshold tuned to lint's deselected suite).
REPORT_RC=0
if [ -n "${COV_FAIL_UNDER:-}" ]; then
  "$PYTHON" -m coverage report -m --fail-under="$COV_FAIL_UNDER" || REPORT_RC=$?
else
  "$PYTHON" -m coverage report -m || REPORT_RC=$?
fi
echo "cov.sh: done (pytest rc=$PYTEST_RC, combine rc=$COMBINE_RC, report rc=$REPORT_RC). Re-report with:"
echo "  COVERAGE_FILE=$COV_DIR/coverage $PYTHON -m coverage report -m"
# Exit contract: ANY failed step reds the run — a failed suite must not be
# masked by a clean report, and a failed coverage step must not be masked by a
# green suite. pytest's rc wins when several steps failed (it is the number a
# human acts on first).
if [ "$PYTEST_RC" -ne 0 ]; then
  exit "$PYTEST_RC"
fi
if [ "$COMBINE_RC" -ne 0 ]; then
  exit "$COMBINE_RC"
fi
exit "$REPORT_RC"
