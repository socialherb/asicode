"""Pin the pre-commit gate config: incremental per-file hooks + CI mirror.

Regression pin for the parallel-session write race fix: the 7 local gate hooks
must stay NON-always_run (per-file).  An always_run full-repo scan recreates
the multi-second window where pre-commit's run-start `git diff` vs post-hook
diff comparison (run.py: files_modified = diff_before != diff_after)
false-positives with "files were modified by this hook" whenever a parallel
session writes a tracked file mid-run.  lint.yml invoking the same scripts
with no args is what keeps the full-repo scan alive.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

GATE_IDS = {
    "baseline-f821",
    "baseline-f401",
    "baseline-f811",
    "no-f823",
    "no-new-silent-except",
    "open-encoding",
    "no-new-unguarded-subprocess",
}


def _local_gate_hooks():
    cfg = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [h for repo in cfg["repos"] if repo.get("repo") == "local" for h in repo["hooks"]]
    return [h for h in hooks if h["id"] in GATE_IDS]


def test_gate_hooks_are_per_file_not_always_run():
    for h in _local_gate_hooks():
        assert h.get("always_run") is not True, (
            f"{h['id']} must not be always_run: a full-repo scan recreates the "
            "multi-second diff-comparison window where a parallel session "
            "writing a tracked file trips a false files-modified failure"
        )
        assert h.get("pass_filenames") is not False, f"{h['id']} must receive filenames (per-file incremental scan)"
        assert "python" in h.get("types", []), f"{h['id']} must be typed [python]"


def test_lint_yml_still_mirrors_all_gate_scripts():
    """lint.yml invokes the same scripts with no args = full-repo backstop."""
    lint = (REPO / ".github/workflows/lint.yml").read_text(encoding="utf-8")
    for h in _local_gate_hooks():
        script = next(part for part in h["entry"].split() if part.endswith(".py"))
        assert script in lint, f"{script} missing from lint.yml — full-repo gate lost"
