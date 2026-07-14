#!/usr/bin/env python3
"""Export the public (CLI-only) snapshot of this repo.

The private repo is the single source of truth; the public GitHub repo is a
filtered subset with fresh history. This script materializes that subset.

Excluded from the public snapshot:
  - external_llm/editor/_editor_core/lane/  (PLANNER lane — permanently
    disabled at routing; kept private, see planner_lane_facade)
  - webapp/          (FastAPI server/UI — not deployed)
  - tools/           (legacy verification scripts)
  - tasks/, screenshots/, .vscode/, CLAUDE.md  (internal artifacts)
  - .github/workflows/p11-ci.yml  (runs a tools/ script)
  - tests that import lane/webapp/tools (recomputed on every export, so
    newly added coupled tests are excluded automatically)

Lint-baseline files (scripts/*_baseline.txt) are copied then pruned: any
entry keyed by ``<path>::...`` whose path is itself excluded from the
export (e.g. a lane/ module, or a coupled test) is dropped, so the public
snapshot's baseline never references or names a file that isn't there.

Usage:
    python3 scripts/export_public.py <target-dir>          # export
    python3 scripts/export_public.py <target-dir> --list   # dry-run listing
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

EXCLUDE_PREFIXES = (
    "external_llm/editor/_editor_core/lane/",
    # lane-only machinery outside lane/ (sole importers are lane executor mixins)
    "external_llm/editor/operation_handlers/",
    "external_llm/editor/refactor/",
    "external_llm/editor/safety/",
    "external_llm/editor/semantic_lineage/",
    "webapp/",
    "tools/",
    "tasks/",
    "screenshots/",
    ".vscode/",
)
EXCLUDE_FILES = {
    "CLAUDE.md",
    ".github/workflows/p11-ci.yml",
    # private development history (references lane/planner internals);
    # the public repo starts its own CHANGELOG at the first release
    "CHANGELOG.md",
    # lane-internal design doc (planner_agent/operation_executor key map)
    "docs/design/stage_context_key_map.md",
    # lane-only lazy-constant shims (sole consumers live in lane/)
    "external_llm/agent/_lazy_constants.py",
    "tests/unit/agent/test_lazy_constants.py",
    # repo-shape guards over lane/tools content — meaningless in the snapshot
    "tests/unit/agent/test_scanner_registry_coverage.py",
    "tests/unit/agent/test_skip_reason_classification.py",
    "tests/unit/test_config_flag_reachable.py",
}

# A test file is excluded when it imports (or patches into) an excluded area.
_COUPLED_TEST_PAT = re.compile(
    r"(_editor_core\.lane|_editor_core/lane"
    r"|editor\.operation_handlers|editor/operation_handlers"
    r"|editor\.refactor|editor\.safety|editor\.semantic_lineage"
    r"|^from webapp|^import webapp\b|from webapp import|from webapp\."
    r"|^from tools|^import tools\b|from tools import|from tools\."
    # path-string loading of excluded dirs (importlib.spec_from_file_location,
    # subprocess script invocations): REPO / "tools" / "x.py", "webapp/..." etc.
    r"|[\"']tools[\"'] */|[\"']webapp[\"'] */|tools/[A-Za-z_]+\.py|webapp/[A-Za-z_]+\.py)",
    re.M,
)


def tracked_files() -> list[str]:
    # -z: NUL-separated so non-ASCII (e.g. Korean) filenames are exact,
    # never C-quoted (see git ls-files quoting semantics).
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    return [p.decode("utf-8") for p in out.split(b"\0") if p]


def is_excluded(rel: str) -> str | None:
    """Return the exclusion reason, or None if the file ships."""
    if rel in EXCLUDE_FILES:
        return "internal"
    for pref in EXCLUDE_PREFIXES:
        if rel.startswith(pref):
            return pref.rstrip("/")
    if rel.startswith("tests/") and rel.endswith(".py"):
        try:
            src = (REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return None
        if _COUPLED_TEST_PAT.search(src):
            return "coupled-test"
    return None


def _prune_baseline_file(path: Path) -> int:
    """Drop ``<excluded-path>::...`` entries from a copied lint-baseline file.

    Baseline entries are keyed ``<file_path>::...``; flag-only baselines
    (e.g. config_flag_unreachable_baseline.txt) have no ``::`` and pass
    through untouched. Returns the number of entries dropped.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)
    kept: list[str] = []
    dropped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "::" not in stripped:
            kept.append(line)
            continue
        ref_path = stripped.split("::", 1)[0]
        if is_excluded(ref_path):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        newline = "\n" if text.endswith("\n") else ""
        path.write_text("\n".join(kept) + newline, encoding="utf-8")
    return dropped


def prune_baseline_files(target: Path, shipped: list[str]) -> None:
    for rel in shipped:
        if not (rel.startswith("scripts/") and rel.endswith("_baseline.txt")):
            continue
        dropped = _prune_baseline_file(target / rel)
        if dropped:
            print(f"pruned {dropped} excluded-path entries from {rel}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1]).resolve()
    dry_run = "--list" in sys.argv[2:]

    shipped: list[str] = []
    excluded: dict[str, int] = {}
    for rel in tracked_files():
        reason = is_excluded(rel)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            shipped.append(rel)

    if dry_run:
        for rel in shipped:
            print(rel)
    else:
        if target.exists() and any(target.iterdir()):
            print(f"error: target {target} exists and is not empty", file=sys.stderr)
            return 1
        for rel in shipped:
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)
        prune_baseline_files(target, shipped)

    print(f"\nshipped: {len(shipped)} files -> {target}", file=sys.stderr)
    for reason, n in sorted(excluded.items(), key=lambda kv: -kv[1]):
        print(f"excluded [{reason}]: {n} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
