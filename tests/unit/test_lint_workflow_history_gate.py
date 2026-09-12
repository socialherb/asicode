"""Pin the history-deepening contract lint.yml's unit-tests job depends on.

`tests/unit/test_sse_emit_consume_gate.py` re-proves the silent subagent-event
drift by reading a blob from a PAST commit (`git show <sha>^:...`), so a depth-1
checkout fails it with exit 128 (measured red in run 34680698084). lint.yml
deepens history in a dedicated step gated on `hashFiles(<that test>)` — a
tree-shape marker, never a repo name.

That condition is only meaningful while the marker is EXCLUDED from the public
export: present => the full private tree (whose history owns the sha);
absent => the snapshot (rebuilt from releases, where the sha never resolves).
This module pins both halves plus the step's shell semantics, so an export-rule
change or a quoting slip cannot silently flip the condition's truth value in
the public tree — where the step would then run and fail.

This file's own source deliberately contains no path-join string over an
excluded tree: the export's coupled-test pattern excludes test files that load
such paths, and this gate must ship in BOTH trees (in the snapshot the marker
half degrades to a skip, matching the workflow condition going false there).
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "lint.yml"
GATE_TEST = "tests/unit/test_sse_emit_consume_gate.py"


def _unit_tests_steps() -> list[dict]:
    cfg = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return cfg["jobs"]["unit-tests"]["steps"]


def _history_step_index() -> int:
    """Index of the single unit-tests step whose run unshallows the checkout."""
    hits = [i for i, step in enumerate(_unit_tests_steps()) if "unshallow" in str(step.get("run", ""))]
    assert len(hits) == 1, f"expected exactly one history-deepening step, found {len(hits)}"
    return hits[0]


def _git(args: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git; probes that assert on the return code pass check=False."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120, check=check)


def _load_export_public():
    """Load scripts/export_public.py (a script, not a package) for its rules."""
    spec = importlib.util.spec_from_file_location("_export_public_history_gate", REPO / "scripts" / "export_public.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_history_step_unshallows_behind_the_private_only_marker() -> None:
    steps = _unit_tests_steps()
    index = _history_step_index()
    step = steps[index]
    assert step["if"] == f"hashFiles('{GATE_TEST}') != ''", (
        "the history step must be gated on the export-excluded gate test itself "
        "(present <=> full private tree): a repo-name guard would leak the private "
        "repo name into a public-shipped file, and an unconditional fetch would pay "
        "a full-history clone in the snapshot for a sha that cannot resolve there"
    )
    # `--unshallow` is FATAL on a complete repository: the step must fetch only
    # when the checkout is actually shallow, or it fails the job it exists to unblock.
    assert "rev-parse --is-shallow-repository" in step["run"]
    assert "git fetch --unshallow" in step["run"]
    # History must exist BEFORE the suite runs, or the step is decoration.
    suite = next(i for i, s in enumerate(steps) if "tests/unit" in str(s.get("run", "")))
    assert index < suite, "the history step must run before the step that runs tests/unit"


def test_the_step_recipe_deepens_a_shallow_clone_and_is_idempotent(tmp_path: Path) -> None:
    """Execute the workflow's own run: text against a real shallow clone.

    Sourcing the recipe from the YAML (instead of paraphrasing its commands)
    keeps this test honest when the step's shell changes.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(["init", "-q"], origin)
    _git(["config", "user.email", "gate@example.com"], origin)
    _git(["config", "user.name", "Gate"], origin)
    (origin / "data.txt").write_text("old\n", encoding="utf-8")
    _git(["add", "data.txt"], origin)
    _git(["commit", "-qm", "old"], origin)
    prev = _git(["rev-parse", "HEAD"], origin).stdout.strip()
    (origin / "data.txt").write_text("new\n", encoding="utf-8")
    _git(["commit", "-qam", "new"], origin)

    clone = tmp_path / "clone"
    _git(["clone", "--depth", "1", origin.as_uri(), str(clone)], tmp_path)
    assert _git(["rev-parse", "--is-shallow-repository"], clone).stdout.strip() == "true", (
        "fixture precondition: the clone must start shallow or this test proves nothing"
    )
    before = _git(["show", f"{prev}:data.txt"], clone, check=False)
    assert before.returncode != 0, "fixture precondition: the parent blob must be unreachable before the recipe"

    recipe = _unit_tests_steps()[_history_step_index()]["run"]
    first = subprocess.run(
        ["bash", "-e", "-c", recipe], cwd=clone, capture_output=True, text=True, timeout=120, check=False
    )
    assert first.returncode == 0, first.stderr
    assert _git(["rev-parse", "--is-shallow-repository"], clone).stdout.strip() == "false"
    after = _git(["show", f"{prev}:data.txt"], clone, check=False)
    assert after.returncode == 0, after.stderr
    assert after.stdout == "old\n"

    again = subprocess.run(
        ["bash", "-e", "-c", recipe], cwd=clone, capture_output=True, text=True, timeout=120, check=False
    )
    assert again.returncode == 0, (
        "the recipe must be idempotent: GitHub runs the step with `bash -e`, and a bare "
        f"`git fetch --unshallow` on a complete repository is fatal ({again.stderr})"
    )


def test_marker_file_is_private_only_and_still_needs_history() -> None:
    """The marker's export exclusion is what makes the hashFiles() condition tree-shaped."""
    marker = REPO / GATE_TEST
    if not marker.exists():
        pytest.skip("public snapshot: the marker is absent, so the condition is false here by construction")
    assert _load_export_public().is_excluded(GATE_TEST) is not None, (
        "the marker file ships publicly, but the condition means 'full private tree' only "
        "while the export excludes it — the snapshot's history cannot resolve the sha it reads"
    )
    src = marker.read_text(encoding="utf-8")
    assert re.search(r'"[0-9a-f]{7,40}\^:', src), (
        "the marker no longer reads a past commit; the workflow step and this contract should be removed with it"
    )
    assert re.search(r'"git",\s*"show"', src), (
        "the marker's past-blob helper no longer invokes `git show`; the history step exists for that read"
    )


def test_sha_pinned_past_blob_reads_live_only_in_the_marker() -> None:
    """Enumeration ratchet: a sha-pinned past blob may live only in the marker file.

    A spec like "<7-40 hex>^:path" resolves only where history reaches that
    commit; lint.yml pays for it with a tree-gated unshallow conditioned on THIS
    one marker.  A second reader appearing (especially in the public snapshot,
    where the step is skipped) would red the job with exit 128 again — so a new
    history-dependent test must be a deliberate decision here (export exclusion +
    workflow story), not a silent add.
    """
    pattern = re.compile(r'"[0-9a-f]{7,40}\^:')
    hits = {
        p.relative_to(REPO).as_posix()
        for p in sorted((REPO / "tests").rglob("*.py"))
        if pattern.search(p.read_text(encoding="utf-8"))
    }
    expected = {GATE_TEST} if (REPO / GATE_TEST).exists() else set()
    assert hits == expected, (
        f"sha-pinned past-blob reads changed: {sorted(hits)} != {sorted(expected)} — "
        f"a new history-dependent test needs the same treatment as {GATE_TEST} "
        "(export exclusion + workflow history step), or this ratchet must be updated deliberately"
    )
