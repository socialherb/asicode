#!/usr/bin/env python3
"""Main-push guard: skip heavy CI jobs when the pushed tree is already green.

P46-A (2026-09-16). Billed Actions minutes exist only on the private mirror
(the public snapshot's runs are free), and a squash-merge push re-runs a tree
the PR run already gated: with the base an ancestor of the PR head, the merged
commit's tree is byte-identical to the PR head tree, so the same checks on the
same bytes re-decide nothing. Measured model: a main-push run costs ~24.9
billed minutes (lint ~22.0 + private-tests ~2.9); the two guard jobs cost ~2.

This script emits ``skip=true`` only when ALL of the following hold:

* the pushed commit's subject carries GitHub's squash-merge ``(#N)`` suffix
  (end-anchored -- a title merely citing another PR cannot select it);
* the pushed tree owns the export-EXCLUDED private-tree marker -- a snapshot
  checkout always fail-opens: its pushes are release snapshots, not merges;
* the pushed tree EQUALS the PR head's tree (main did not move before the
  merge -- a mismatch means these bytes were never the PR gate's subject);
* every ``--required`` check name -- folded to its LATEST attempt per name --
  is SUCCESS on the PR head's ``statusCheckRollup``.

Anything else -- no reference, marker absent, tree mismatch, missing / failed /
unfinished check, API error, timeout, exception, empty ``--required`` -- is
fail-open: ``skip=false`` and the heavy jobs run. The guard JOB is also
``continue-on-error`` and the dependent jobs' condition is
``!cancelled() && needs.guard.outputs.skip != 'true'``. Both halves are
load-bearing, measured on a throwaway public branch (2026-09-16):

* a SKIPPED needs-job blocks a dependent whose ``if`` carries no status
  function -- the plain ``needs.X.outputs.skip != 'true'`` form FAILS CLOSED
  on every pull_request event -- while ``!cancelled() && ...`` runs;
* a continue-on-error failure reports ``needs.<job>.result == success`` and
  still exports outputs, so every failure path stays open;
* on a concurrency supersede, ``!cancelled()`` dependents never start (no
  spend), while ``always()`` dependents strand ``queued`` after the cancel --
  the run then never concludes and keeps holding its concurrency group.

``--replay [PR ...]`` (optionally ``--since/--until`` instead of numbers)
re-adjudicates past merged PRs against the CURRENT rules and prints one
verdict per PR -- the harness for reviewing a rules change without CI.

``--probe-token`` is the real-run scope check for the guard job's
``permissions:`` block: under the same ``GH_TOKEN`` decide() runs with, it
performs the two token-scoped reads the decision depends on (the commit tree
and ``gh pr view --json statusCheckRollup``), prints one annotation per
surface and returns non-zero when a required read fails. The workflow step is
``continue-on-error`` and the probe never writes step outputs, so the
decision path is untouched -- without this, a scope gap is silent (the guard
merely fail-opens forever and the skip savings never materialise).

Exit code is always 0 in guard mode; the decision lives in the step outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

#: Export-EXCLUDED private-tree marker (the same idiom as lint.yml's
#: hashFiles() gates): present <=> this checkout is the full private tree.
TREE_SHAPE_MARKER = "tests/unit/test_sse_emit_consume_gate.py"

#: GitHub appends " (#123)" to a squash-merged commit's subject.
PR_REFERENCE_RE = re.compile(r"\(#(\d+)\)\s*$")

_SUBPROCESS_TIMEOUT_S = 60


class GuardError(RuntimeError):
    """A fact could not be gathered. Callers must fail open (run the jobs)."""


def _run(cmd: Sequence[str]) -> str:
    """Run *cmd*, returning stdout; every failure mode raises GuardError."""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except OSError as exc:
        raise GuardError(f"cannot execute {cmd[0]!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GuardError(f"{cmd[0]!r} timed out after {_SUBPROCESS_TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        raise GuardError(
            f"{' '.join(str(part) for part in cmd[:2])} failed (rc={proc.returncode}): {' '.join(proc.stderr.split())[:300]}"
        )
    return proc.stdout


def _gh(*args: str) -> str:
    return _run(("gh", *args))


def _git(*args: str) -> str:
    return _run(("git", *args))


def commit_subject(sha: str) -> str:
    """The pushed commit's subject line."""
    return _git("log", "-1", "--format=%s", sha).strip()


def tree_contains_marker(sha: str, marker: str = TREE_SHAPE_MARKER) -> bool:
    """True when *sha*'s own tree owns the export-excluded private-tree marker.

    ``git cat-file -e`` resolves the path inside the commit's tree (not the
    working directory), so a depth-1 checkout answers about the pushed tree.
    A probe failure prints and answers False -- absent => fail-open run.
    """
    try:
        _git("cat-file", "-e", f"{sha}:{marker}")
    except GuardError as exc:
        print(f"marker probe: {marker} not in {sha[:12]} ({exc})", file=sys.stderr)
        return False
    return True


def commit_tree(repo: str, sha: str) -> str:
    """The git tree object id of *sha*."""
    return _gh("api", f"repos/{repo}/commits/{sha}", "--jq", ".commit.tree.sha").strip()


def pr_view(repo: str, pr: int) -> dict:
    """``{headRefOid, mergeCommit, statusCheckRollup}`` for PR *pr*."""
    raw = _gh("pr", "view", str(pr), "--repo", repo, "--json", "headRefOid,mergeCommit,statusCheckRollup")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise GuardError(f"gh pr view {pr} returned non-object JSON")
    return data


@dataclass(frozen=True)
class GuardFacts:
    """Everything decide() may look at -- gathered by gather_facts()."""

    subject: str
    marker_present: bool
    main_tree: str | None = None
    pr_head_tree: str | None = None
    rollup: tuple[dict, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class Decision:
    skip: bool
    reason: str


def extract_pr_number(subject: str) -> int | None:
    """The PR number GitHub's squash-merge subject ends with, or None."""
    match = PR_REFERENCE_RE.search(subject)
    return int(match.group(1)) if match else None


def latest_conclusion_by_name(rollup: Iterable[dict]) -> dict[str, str]:
    """Fold a statusCheckRollup to {check name: LATEST attempt's conclusion}.

    Retries (``gh run rerun``) leave several check runs per name; only the
    newest attempt may decide. Folding by max(startedAt/createdAt) is
    order-independent and does not rely on the API surfacing a stale flag.
    """
    newest: dict[str, tuple[str, str]] = {}
    for item in rollup:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("context")
        if not isinstance(name, str) or not name:
            continue
        conclusion = item.get("conclusion") or item.get("state") or ""
        started = item.get("startedAt") or item.get("createdAt") or item.get("completedAt") or ""
        previous = newest.get(name)
        if previous is None or str(started) > previous[0]:
            newest[name] = (str(started), str(conclusion))
    return {name: conclusion for name, (_started, conclusion) in newest.items()}


def decide(facts: GuardFacts, required: Sequence[str]) -> Decision:
    """Pure adjudication; every uncertain path returns skip=False."""
    if not required:
        return Decision(False, "no --required checks configured - refusing to skip on an empty contract")
    if facts.error is not None:
        return Decision(False, f"fail-open: {facts.error}")
    pr = extract_pr_number(facts.subject)
    if pr is None:
        return Decision(False, "commit subject carries no ' (#N)' squash-merge reference")
    if not facts.marker_present:
        return Decision(False, f"tree-shape marker {TREE_SHAPE_MARKER} absent (snapshot checkout)")
    if not facts.main_tree or not facts.pr_head_tree:
        return Decision(False, "fail-open: commit-tree data missing")
    if facts.main_tree != facts.pr_head_tree:
        return Decision(
            False,
            f"pushed tree {facts.main_tree[:12]} != PR #{pr} head tree {facts.pr_head_tree[:12]}: "
            "main moved before the merge, so these bytes were never the PR gate's subject",
        )
    latest = latest_conclusion_by_name(facts.rollup)
    missing = [name for name in required if name not in latest]
    if missing:
        return Decision(False, f"PR #{pr} head rollup lacks required checks: {', '.join(missing)}")
    not_green = {name: latest[name] for name in required if latest[name] != "SUCCESS"}
    if not_green:
        rendered = ", ".join(f"{name}={value or 'INCOMPLETE'}" for name, value in not_green.items())
        return Decision(False, f"PR #{pr} required checks not SUCCESS: {rendered}")
    return Decision(
        True, f"tree {facts.pr_head_tree[:12]} of PR #{pr} already passed all {len(required)} required checks"
    )


def gather_facts(repo: str, sha: str, marker: str = TREE_SHAPE_MARKER) -> GuardFacts:
    """Collect adjudication facts; any failure becomes an error fact (fail-open)."""
    try:
        subject = commit_subject(sha)
        marker_present = tree_contains_marker(sha, marker)
    except (GuardError, KeyError, TypeError, ValueError) as exc:
        return GuardFacts(subject="", marker_present=False, error=str(exc))
    facts = GuardFacts(subject=subject, marker_present=marker_present)
    pr = extract_pr_number(subject)
    if pr is None or not marker_present:
        return facts
    try:
        view = pr_view(repo, pr)
        head_sha = view.get("headRefOid")
        if not isinstance(head_sha, str) or not head_sha:
            raise GuardError(f"PR #{pr} view carries no headRefOid")
        rollup = view.get("statusCheckRollup") or []
        if not isinstance(rollup, list):
            raise GuardError(f"PR #{pr} statusCheckRollup is not a list")
        main_tree = commit_tree(repo, sha)
        head_tree = commit_tree(repo, head_sha)
    except (GuardError, KeyError, TypeError, ValueError) as exc:
        return replace(facts, error=str(exc))
    kept = tuple(item for item in rollup if isinstance(item, dict))
    return replace(facts, main_tree=main_tree, pr_head_tree=head_tree, rollup=kept)


def _append_output(path: str, decision: Decision) -> None:
    """Write single-line step outputs and a workflow notice."""
    skip_line = f"skip={'true' if decision.skip else 'false'}"
    reason = " ".join(decision.reason.split())
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(f"{skip_line}\nreason={reason}\n")
    print(skip_line)
    print(f"::notice ::main-push guard: {'SKIP' if decision.skip else 'RUN'} - {reason}")


def merged_pr_numbers(repo: str, since: str, until: str | None = None) -> list[int]:
    """Merged PR numbers with mergedAt on/after *since* (and before *until*)."""
    rows = json.loads(
        _gh("pr", "list", "--repo", repo, "--state", "merged", "--limit", "200", "--json", "number,mergedAt")
    )
    if not isinstance(rows, list):
        raise GuardError("gh pr list returned non-list JSON")
    picked: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        number = row.get("number")
        day = str(row.get("mergedAt") or "")[:10]
        if not isinstance(number, int) or not day:
            continue
        if day < since or (until is not None and day >= until):
            continue
        picked.append(number)
    return sorted(picked)


def replay(repo: str, numbers: Sequence[int], required: Sequence[str]) -> int:
    """Re-adjudicate merged PRs with the CURRENT rules (no CI needed)."""
    skipped = 0
    total = 0
    for pr in numbers:
        total += 1
        try:
            view = pr_view(repo, pr)
        except (GuardError, ValueError) as exc:
            print(f"PR #{pr:>4}: ERROR {exc}")
            continue
        merge = view.get("mergeCommit")
        sha = merge.get("oid") if isinstance(merge, dict) else None
        if not isinstance(sha, str) or not sha:
            print(f"PR #{pr:>4}: no merge commit recorded (not merged?)")
            continue
        facts = gather_facts(repo, sha)
        decision = decide(facts, required)
        if decision.skip:
            skipped += 1
        tree = facts.main_tree[:9] if facts.main_tree else "?"
        print(f"PR #{pr:>4} merge {sha[:9]} tree {tree:<9} -> {'SKIP' if decision.skip else 'RUN'}  {decision.reason}")
    print(f"replay: {skipped}/{total} adjudicated merges would have skipped")
    return 0


def probe_pr_number(repo: str, subject: str, explicit: int | None = None) -> tuple[int | None, str]:
    """The PR a token probe should read: ``--pr``, else the subject, else the newest."""
    if explicit is not None:
        return explicit, "explicit --pr"
    referenced = extract_pr_number(subject)
    if referenced is not None:
        return referenced, "commit subject ' (#N)' reference"
    rows = json.loads(_gh("pr", "list", "--repo", repo, "--state", "all", "--limit", "1", "--json", "number"))
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("number"), int):
                return row["number"], "newest PR in the repository"
    return None, ""


def probe_token(repo: str, sha: str, pr: int | None = None) -> int:
    """Exercise, with the workflow's own token, the exact reads decide() performs.

    Diagnostic only: never writes step outputs and cannot influence the
    decision. Prints one annotation per token-scoped surface (commit read,
    PR lookup, PR view with statusCheckRollup) and returns 0 when the required
    reads succeeded, 1 otherwise -- the workflow step is continue-on-error,
    so even 1 cannot block decide() or the dependent heavy jobs.
    """
    failures: list[str] = []
    try:
        subject = commit_subject(sha)
    except GuardError as exc:
        subject = ""
        print(f"::warning::guard token probe: cannot read the commit subject ({exc})")
    try:
        tree = commit_tree(repo, sha)
    except GuardError as exc:
        failures.append(f"commit read (contents: read): {exc}")
        print(f"::error::guard token probe: commit read FAILED - {exc}")
    else:
        print(f"::notice::guard token probe: commit read OK - tree {tree[:12]}")
    try:
        number, origin = probe_pr_number(repo, subject, pr)
    except (GuardError, ValueError) as exc:
        failures.append(f"PR lookup (pull-requests: read): {exc}")
        print(f"::error::guard token probe: PR lookup FAILED - {exc}")
        number, origin = None, ""
    if number is None:
        print(f"::warning::guard token probe: no pull request in {repo} to exercise the rollup read")
    else:
        try:
            view = pr_view(repo, number)
        except (GuardError, ValueError) as exc:
            failures.append(f"gh pr view #{number} (pull-requests + checks + statuses read): {exc}")
            print(f"::error::guard token probe: gh pr view {number} FAILED - {exc}")
        else:
            rollup = [item for item in (view.get("statusCheckRollup") or []) if isinstance(item, dict)]
            folded = latest_conclusion_by_name(rollup)
            contexts = sum(1 for item in rollup if "context" in item)
            green = sum(1 for value in folded.values() if value == "SUCCESS")
            print(
                f"::notice::guard token probe: gh pr view {number} ({origin}) OK - rollup {len(rollup)} entries "
                f"({len(rollup) - contexts} check runs, {contexts} status contexts), "
                f"{green} SUCCESS of {len(folded)} names"
            )
    if failures:
        print(f"::error::guard token probe: {len(failures)} required read(s) failed - " + "; ".join(failures))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P46-A main-push tree-reuse guard (see the module docstring).")
    parser.add_argument(
        "--required",
        action="append",
        default=[],
        metavar="CHECK",
        help="exact check-run name that must be SUCCESS on the PR head; repeat per check",
    )
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""), metavar="OWNER/NAME")
    parser.add_argument(
        "--sha", default=os.environ.get("GITHUB_SHA", ""), help="pushed commit to adjudicate (default: $GITHUB_SHA)"
    )
    parser.add_argument("--marker", default=TREE_SHAPE_MARKER, help="export-excluded tree-shape marker path")
    parser.add_argument(
        "--output", default=os.environ.get("GITHUB_OUTPUT", ""), help="step-output file (default: $GITHUB_OUTPUT)"
    )
    parser.add_argument(
        "--probe-token",
        action="store_true",
        help="diagnose the workflow token's ability to perform decide()'s reads (see probe_token)",
    )
    parser.add_argument(
        "--pr", type=int, default=None, metavar="N", help="with --probe-token: read this PR instead of deriving one"
    )
    parser.add_argument(
        "--replay", nargs="*", type=int, default=None, metavar="PR", help="re-adjudicate these merged PRs"
    )
    parser.add_argument(
        "--since", metavar="YYYY-MM-DD", help="with --replay and no numbers: merged PRs on/after this day"
    )
    parser.add_argument("--until", metavar="YYYY-MM-DD", help="... and before this day")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.replay is not None:
        numbers = list(args.replay)
        if not numbers and args.since:
            numbers = merged_pr_numbers(args.repo, args.since, args.until)
        if not numbers:
            print("replay: pass PR numbers or --since to select past merges", file=sys.stderr)
            return 2
        return replay(args.repo, numbers, args.required)
    if not args.repo or not args.sha:
        raise SystemExit("--repo and --sha are required (or set GITHUB_REPOSITORY and GITHUB_SHA)")
    if args.probe_token:
        return probe_token(args.repo, args.sha, args.pr)
    _append_output(args.output, decide(gather_facts(args.repo, args.sha, args.marker), args.required))
    return 0


if __name__ == "__main__":
    sys.exit(main())
