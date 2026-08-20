#!/usr/bin/env python3
"""Release the public (CLI-only) snapshot into the public GitHub repo.

One command for the whole release step:
  1. Export the filtered snapshot (scripts/export_public.py rules).
  2. Sync it into the public repo working tree — overwrite changed files AND
     remove files that no longer ship (rsync --delete semantics).
  3. Commit in the public repo with a traceable message; optionally tag/push.

The private repo (this one) stays the single source of truth. This script is
the ONLY sanctioned path to publish: never push the private repo itself to a
public remote — its history contains lane/, webapp/, tools/.

PyPI publishing is automatic: the `.github/workflows/release.yml` workflow
triggers on the `v*` tag push and uploads to PyPI via Trusted Publishing (OIDC)
— no API token is stored anywhere. After the one-time publisher registration
(https://pypi.org/manage/project/asicode/settings/publishing/), the entire
release is: bump version → commit → `release_public.py <pub> --tag --push`.
The tag push both publishes the snapshot and drives the PyPI upload.

Usage:
    python3 scripts/release_public.py <public-repo-path> [--tag] [--push] [--allow-dirty]

    <public-repo-path>  existing git repo (git init it once, first release
                        creates the initial commit). Defaults to
                        $ASICODE_PUBLIC_REPO when omitted.
    --tag           tag the release commit v<version> (version from pyproject.toml)
    --push          push branch (and tag, with --tag) to the public repo's origin
    --verify[=fast|full]
                    before committing, run the public-CI gate mirror on the
                    STAGED release inside the public repo (fast: script gates
                    + tree-policy unit subset; full: + the whole snapshot unit
                    suite). Any failure aborts BEFORE commit/tag/push.
    --allow-dirty   skip the clean-working-tree check (testing only — a dirty
                    tree means uncommitted edits of tracked files get published)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import export_public  # noqa: E402  (reuse the exclusion rules verbatim)


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a subprocess with a 120s timeout. On timeout, abort immediately."""
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        cmd = " ".join(str(a) for a in args)
        print(
            f"fatal: '{cmd}' timed out after 120s — aborting release.",
            file=sys.stderr,
        )
        sys.exit(1)


# First-party packages ship in the wheel. The definition — and the memoized
# per-file import scan — lives in export_public, which (unlike this script) is
# not re-executed between gate invocations, so its caches survive.
FIRST_PARTY_PREFIXES = export_public.FIRST_PARTY_PREFIXES


def _version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "0.0.0"


def _check_untracked_imports() -> bool:
    """Fail-fast gate: every first-party module imported by a *shipping* file
    must itself be tracked by git.  Prevents the 'untracked module silently
    missing from wheel' class of release bug (0.2.6 version_check, 0.2.14
    rich_markdown).

    Two properties this gate lives or dies by:

    * **``ast.walk``, not ``ast.iter_child_nodes``.**  Only walking top-level
      statements would miss function-level imports — and this codebase imports
      lazily by convention (``asi._rich_markdown_cls`` and
      ``streaming_display._markdown_lines`` are both function-level).  A
      top-level-only scan passes clean on the exact bug this gate cites as its
      reason to exist, which is worse than no gate: it certifies the release.
      The traversal lives in ``export_public._first_party_imports`` (memoized
      per path — the tree is immutable during a run), and the gate delegates
      to it.  Mutation guard: revert to ``iter_child_nodes`` → the
      ``rich_markdown``-untracked case in test_release_untracked_import_gate
      FAILS.
    * **Scoped to what ships.**  ``is_excluded`` drops lane/, webapp/, tools/
      etc., whose imports are irrelevant to the wheel and which legitimately
      reference private-only modules.  Scanning them produced 4 pre-existing
      false positives that would have made the gate un-turn-on-able.

    Returns True if all imports resolve to tracked files, False otherwise.
    """
    tracked = set(export_public.tracked_files())
    shipped = [
        rel for rel in tracked
        if rel.endswith(".py") and export_public.is_excluded(rel) is None
    ]

    errors: list[str] = []
    for pyfile in sorted(shipped):
        # The scan is memoized in export_public (keyed by path — repo files are
        # immutable during a run), so repeated invocations — this gate, the
        # ignored-.py gate, the export itself — pay read/parse/walk once per
        # file.  Missing or malformed files yield no imports (skipped, as
        # before).
        for module in export_public._first_party_imports(pyfile):
            _check_import(module, pyfile, tracked, errors)
    if errors:
        print("error: release blocked — untracked first-party imports detected:", file=sys.stderr)
        for line in sorted(set(errors)):
            print(f"  {line}", file=sys.stderr)
        print("  → git add the module(s), or the wheel ships an ImportError.", file=sys.stderr)
        return False
    return True


def _check_import(module: str, importer: str, tracked: set[str], errors: list[str]) -> None:
    """Check if a single imported module resolves to a tracked file."""
    # Only check first-party modules (those under our packages).
    if not any(module.startswith(p) for p in FIRST_PARTY_PREFIXES):
        # Not a first-party module — skip (stdlib, third-party, or top-level
        # module like "asi" which lives in the repo root and is always tracked).
        return
    # Convert module name to file path: external_llm.foo.bar → external_llm/foo/bar.py
    parts = module.split(".")
    # The module file could be a directory with __init__.py or a .py file
    # Try module.py first, then module/__init__.py
    candidate_py = "/".join(parts) + ".py"
    candidate_init = "/".join(parts) + "/__init__.py"
    if candidate_py in tracked or candidate_init in tracked:
        return
    # Namespace package: a package directory WITHOUT __init__.py (e.g.
    # external_llm/graph) still ships and imports fine as long as at least one
    # tracked module lives under it — export copies tracked files, and Python
    # resolves the namespace at import time.  Treat it as tracked; a package
    # whose EVERY member is untracked stays a hard error (nothing ships).
    prefix = "/".join(parts) + "/"
    if any(p.startswith(prefix) for p in tracked):
        return
    errors.append(f"{module} (imported by {importer}) → not tracked by git")


def _ignored_shipping_py(ignored_lines: list[str], shipped: list[str]) -> list[str]:
    """Return gitignored ``.py`` files that would silently vanish from the wheel.

    ``git status --porcelain`` does not report ignored files, so the clean-tree
    check passes while a gitignored module is absent from the export (which
    copies tracked files only) — the 0.2.6 ``version_check`` class of bug. The
    hazard scope is: any ignored ``.py`` under a top-level directory that
    contains shipped ``.py`` files, plus any ignored root-level ``.py`` (root
    modules ship too).
    """
    shipped_dirs = {rel.split("/", 1)[0] for rel in shipped if "/" in rel and rel.endswith(".py")}
    bad: list[str] = []
    for line in ignored_lines:
        stripped = line.strip()
        if not stripped.startswith("!! "):
            continue
        rel = stripped[3:].strip().strip('"')
        if not rel.endswith(".py"):
            continue
        if "/" in rel:
            if rel.split("/", 1)[0] in shipped_dirs:
                bad.append(rel)
        else:
            # Root .py files ship too; any ignored root module is suspicious.
            bad.append(rel)
    return bad


def _check_ignored_shipping_py(shipped: list[str]) -> bool:
    """Fail-fast gate: no gitignored .py under a shipping location.

    Machine-enforces the CLAUDE.md guidance ("release checks must use
    ``git status --ignored``, not ``git status``) so the 0.2.6 class of
    release bug cannot silently recur.
    """
    out = _run(["git", "status", "--porcelain", "--ignored"], REPO).stdout
    bad = _ignored_shipping_py(out.splitlines(), shipped)
    if bad:
        print(
            "error: release blocked — gitignored .py files under shipping locations:\n"
            + "".join(f"  {rel}\n" for rel in sorted(bad))
            + "  These are silently absent from the wheel (export copies tracked files\n"
            "  only). Fix with: git check-ignore -v <file> → remove the rule and\n"
            "  git add <file>, or delete the file.",
            file=sys.stderr,
        )
        return False
    return True


def _changelog_has_version(version: str) -> bool:
    """True if CHANGELOG.md contains a ``## [<version>]`` section header.

    Catches the recurring 'bumped version, forgot CHANGELOG' gap — the log had
    been stuck at an old release across several version bumps because nothing
    enforced the entry.  Versions are digits/dots, so a plain substring match
    on ``## [0.2.12]`` is exact and unambiguous.
    """
    cl = REPO / "CHANGELOG.md"
    if not cl.exists():
        return False
    return f"## [{version}]" in cl.read_text(encoding="utf-8", errors="replace")


# ── --verify: the public-CI gate mirror, run on the STAGED release ─────────
#
# Why this exists (v0.2.25 hotfix, first-attempt push): the release flow
# verified the structural baseline but nothing else about the SNAPSHOT —
# export_public.py:402 carried errors='ignore', a violation of the P26-3
# policy unit gate that ships in the public tree, and the failure only
# surfaced in public CI *after* the push (a second push fixed it). The gates
# below mirror the public repo's lint job one-to-one: they run with
# cwd=<public repo> AFTER `git add -A` and BEFORE the commit, so the
# --index-only forms scan exactly the bytes this release would push — the
# same content, the same semantics, at the cheapest possible point in the
# chain (a local abort costs seconds; a CI round-trip after a push costs a
# red main plus a forced follow-up commit).
#
# NOT mirrored here (deliberately):
#   * PLR2004 / silent-except --observe — report-only steps, exit 0 by design.
#   * JS coverage suite — tree-shape guarded in CI (no webapp/ in the
#     snapshot), nothing to gate locally.
#   * unit/integration jobs — that is what --verify=full (unit) and CI are
#     for; fast stays in the "seconds per gate" band.
_VERIFY_GATES: list[tuple[str, list[str]]] = [
    ("f821-baseline", ["scripts/check_f821_no_new.py"]),
    ("f401-baseline", ["scripts/check_f401_no_new.py"]),
    ("f811-baseline", ["scripts/check_f811_no_new.py"]),
    ("f823-zero", ["scripts/check_f823_none.py"]),
    ("missing-global-zero", ["scripts/check_missing_global.py"]),
    ("silent-except-baseline", ["scripts/check_no_new_silent_except.py", "--index-only"]),
    ("open-encoding-zero", ["scripts/check_open_encoding.py"]),
    ("unguarded-subprocess-baseline", ["scripts/check_no_new_unguarded_subprocess.py"]),
    ("standalone-imports-baseline", ["scripts/check_standalone_imports.py", "--index-only"]),
    ("first-party-fallback-baseline", ["scripts/check_no_new_first_party_import_fallback.py", "--index-only"]),
    ("discarded-signal-baseline", ["scripts/check_discarded_signal.py"]),
    ("dead-params-zero", ["scripts/check_dead_params.py"]),
    ("select-redundant-zero", ["scripts/check_select_not_redundant.py"]),
    ("ruff-full-zero", ["scripts/check_lint_full.py"]),
    ("structural-scanners-zero", ["scripts/check_structural_scanners.py", "--gate-only"]),
]

# Tree-policy unit tests for --verify=fast: tests that statically scan the
# SHIPPED tree and can therefore fail on the exported subset even when the
# private tree is green — exactly the v0.2.25 errors='ignore' failure class
# (test_file_read_errors_ignore_gate). Enumerated policy, like a baseline:
# extend this list when adding a new tree-scanning policy test.
#
# CONSTRAINT (machine-checked in test_release_verify_mode.py): every entry
# must itself SHIP — is_excluded(path) is None. A coupled-test entry (the
# sse/tools-git/ui-route gates read webapp/ or tools/, so they are excluded
# from the snapshot) would make fast mode fail with pytest exit 4 "file not
# found" on every release: a false red that blocks shipping.
_FAST_POLICY_TESTS: list[str] = [
    "tests/unit/test_file_read_errors_ignore_gate.py",
    "tests/unit/test_network_boundary_get_policy.py",
    "tests/unit/test_provider_rate_limit_contract.py",
]

_VERIFY_GATE_TIMEOUT_S = 600    # structural gate cold-boots ~69s; 600 is headroom
_VERIFY_UNIT_TIMEOUT_S = 2400

# Durations evidence: the unit steps run with ``-p no:cacheprovider`` (the
# public tree must not grow a .pytest_cache the next release would stage), so
# pytest's own duration cache never exists — timing was argued from total wall
# time only. ``--durations`` is computed in-session (cache-independent) and the
# step output is persisted as an artifact (see _write_verify_artifact).
_VERIFY_DURATIONS_COUNT = 40
_VERIFY_ARTIFACT_DIRNAME = ".verify_artifacts"
_VERIFY_ARTIFACT_KEEP = 12   # full snapshot unit suite: ~5-6 min in CI


def _run_verify_step(args: list[str], cwd: Path, timeout: float) -> tuple[bool, str, str]:
    """Run one verify step. Returns (ok, output-tail, full-output) — never
    raises, never sys.exits: the caller decides what an abort looks like.
    The full output feeds the durations artifact; the tail is for the abort
    message."""
    try:
        proc = subprocess.run(
            [sys.executable, *args], cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"timed out after {exc.timeout:.0f}s", ""
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return True, "", out
    tail = "\n".join(out.strip().splitlines()[-30:])
    return False, f"exit {proc.returncode}\n{tail}", out


def _write_verify_artifact(
    mode: str, dt: float, failed: int, total: int, chunks: list[str]
) -> Path | None:
    """Persist per-step output (pytest --durations) OUTSIDE the public tree.

    Two placement constraints, both machine-checked in test_release_verify_mode:

    * **Not in *public*** — verify runs after ``git add -A``, so any file
      created there would be swept into the NEXT release's staging and then
      deleted by the snapshot sync: leaking into shipped bytes or thrashing
      the committed tree.
    * **Gitignored in the private repo** — the clean-tree preflight reads
      ``git status --porcelain``, which lists untracked files; an un-ignored
      artifact dir would block the next release (the same abort-dead-end
      class as the stash guidance fixed for the public tree).

    Observability must never abort a release: any OSError degrades to None.
    """
    try:
        art_dir = REPO / _VERIFY_ARTIFACT_DIRNAME
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / f"verify-durations-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        status = f"{total - failed}/{total} steps green" if not failed else f"{failed}/{total} steps FAILED"
        header = (
            f"# asicode release --verify={mode} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# {status}, {dt:.0f}s total\n"
        )
        path.write_text(header + "".join(chunks), encoding="utf-8")
    except OSError:
        return None
    # Retention cap: `olds[:-KEEP]` is empty until the cap is exceeded, so the
    # slice alone implements "keep newest N". Prune failure is non-fatal.
    try:
        for old in sorted(art_dir.glob("verify-durations-*.txt"))[:-_VERIFY_ARTIFACT_KEEP]:
            old.unlink(missing_ok=True)
    except OSError:
        pass
    return path


def _verify_release(public: Path, mode: str) -> bool:
    """Run the gate mirror on the staged release in *public*. True = green."""
    steps: list[tuple[str, list[str], float]] = [
        (name, args, _VERIFY_GATE_TIMEOUT_S) for name, args in _VERIFY_GATES
    ]
    durations = [f"--durations={_VERIFY_DURATIONS_COUNT}"]
    if mode == "fast":
        if _FAST_POLICY_TESTS:
            steps.append((
                "policy-unit-subset",
                ["-m", "pytest", "-q", "-p", "no:cacheprovider", *durations,
                 *_FAST_POLICY_TESTS],
                _VERIFY_UNIT_TIMEOUT_S,
            ))
    else:  # full — fast gates plus the whole snapshot unit suite
        steps.append((
            "unit-suite-full",
            ["-m", "pytest", "-q", "-p", "no:cacheprovider", *durations,
             "tests/unit", "-m", "not slow"],
            _VERIFY_UNIT_TIMEOUT_S,
        ))

    failed = 0
    t_start = time.monotonic()
    chunks: list[str] = []
    for i, (name, args, timeout) in enumerate(steps, 1):
        print(f"verify [{i}/{len(steps)}] {name}")
        sys.stdout.flush()
        ok, detail, out = _run_verify_step(args, public, timeout)
        verdict = "ok" if ok else "FAILED"
        chunks.append(f"\n===== [{i}/{len(steps)}] {name} — {verdict} =====\n{out.rstrip()}\n")
        if not ok:
            failed += 1
            print(f"    FAILED: {detail}", file=sys.stderr)
    dt = time.monotonic() - t_start
    artifact = _write_verify_artifact(mode, dt, failed, len(steps), chunks)
    if artifact:
        print(f"verify: durations artifact → {artifact}")
    if failed:
        print(f"verify: {failed}/{len(steps)} step(s) failed in {dt:.0f}s", file=sys.stderr)
        return False
    print(f"verify: all {len(steps)} steps green in {dt:.0f}s")
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="release_public.py",
        description="Export the filtered public snapshot and release it into the public git repo.",
    )
    parser.add_argument(
        "public_repo", nargs="?", default=None, metavar="public-repo-path",
        help="existing git repo (git init it once; defaults to $ASICODE_PUBLIC_REPO)",
    )
    parser.add_argument("--tag", action="store_true",
                        help="tag the release commit v<version> (version from pyproject.toml)")
    parser.add_argument("--push", action="store_true",
                        help="push branch (and tag, with --tag) to the public repo's origin")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="skip the clean-working-tree check (testing only — a dirty tree "
                             "means uncommitted edits of tracked files get published)")
    parser.add_argument(
        "--verify", nargs="?", const="fast", default=None, choices=("fast", "full"),
        metavar="{fast,full}",
        help="run the public-CI gate mirror on the STAGED release before committing "
             "(fast: script gates + tree-policy unit subset; full: + the whole snapshot "
             "unit suite). Any failure aborts BEFORE commit/tag/push. Pass the repo path "
             "BEFORE --verify: '--verify <path>' is rejected as an invalid mode "
             "(fail-closed, never misparsed).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    target_arg = args.public_repo or os.environ.get("ASICODE_PUBLIC_REPO", "")
    if not target_arg:
        print(__doc__)
        return 2
    public = Path(target_arg).resolve()

    # ── Preflight ──────────────────────────────────────────────────────────
    if not (public / ".git").is_dir():
        print(f"error: {public} is not a git repo — create it once with:\n"
              f"  mkdir -p {public} && git -C {public} init -b main", file=sys.stderr)
        return 1
    if public == REPO or REPO.is_relative_to(public):
        print("error: target must not be the private repo itself", file=sys.stderr)
        return 1

    dirty = _run(["git", "status", "--porcelain"], REPO).stdout.strip()
    if dirty and not args.allow_dirty:
        print("error: private repo has uncommitted changes — the export copies\n"
              "working-tree contents of tracked files, so a dirty tree would\n"
              "publish uncommitted edits. Commit first (or --allow-dirty for tests).",
              file=sys.stderr)
        return 1

    pub_dirty = _run(["git", "status", "--porcelain"], public).stdout.strip()
    if pub_dirty:
        print(f"error: public repo {public} has uncommitted changes — resolve first.",
              file=sys.stderr)
        return 1

    # ── CHANGELOG gate: the version being released must have an entry ───────
    # Computed once here and reused for the commit message below.
    version = _version()
    if not _changelog_has_version(version):
        import datetime as _dt
        print(
            f"error: CHANGELOG.md has no '## [{version}]' section for this release.\n"
            "A bumped version without a changelog entry is exactly the gap this gate "
            "exists for. Add a section, e.g.:\n"
            f'  ## [{version}] — {_dt.date.today().isoformat()}\n'
            "    ### Added / ### Changed / ### Fixed ...\n"
            "Draft from recent commits:  git log --oneline $(git tag | sort -V | tail -1)..HEAD\n",
            file=sys.stderr,
        )
        return 1

    # ── 1) Check: no untracked first-party imports ──────────────────────────
    # A tracked file importing an untracked module means the module is
    # silently absent from the wheel (export copies tracked files only).
    # This gate catches the 0.2.6 (version_check) class of release bug.
    if not _check_untracked_imports():
        return 1

    # ── 1b) Check: no gitignored .py under a shipping location ──────────────
    # `git status --porcelain` is blind to ignored files; a gitignored module
    # passes the clean-tree check and silently vanishes from the wheel.
    # The same single pass classifies the tracked tree for the export below.
    shipped: list[str] = []
    excluded_paths: list[str] = []
    for rel in export_public.tracked_files():
        if export_public.is_excluded(rel) is None:
            shipped.append(rel)
        else:
            excluded_paths.append(rel)
    if not _check_ignored_shipping_py(shipped):
        return 1

    # ── 2) Build the snapshot via the SHARED sequence and sync it ───────────
    # export_public.build_snapshot = copy + baseline-prune + structural-baseline
    # regeneration. The hand-rolled copy that used to live here skipped the
    # regeneration: the baseline is a machine-generated export artifact (it
    # must NOT exist in the private tree), so the next release through this
    # script would have deleted it from the public repo and reddened the
    # structural gate. The sync mirrors the BUILT snapshot exactly — whatever
    # tmp contains is what ships (including generated artifacts).
    with tempfile.TemporaryDirectory(prefix="asicode-release-") as td:
        tmp = Path(td)
        if not export_public.build_snapshot(tmp, shipped, excluded_paths):
            print("error: snapshot build failed (structural baseline generation) — "
                  "release aborted", file=sys.stderr)
            return 1
        snapshot = {p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file()}

        removed = 0
        for p in sorted(public.rglob("*"), reverse=True):
            rel = p.relative_to(public).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                continue
            if p.is_file() and rel not in snapshot:
                p.unlink()
                removed += 1
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        for rel in sorted(snapshot):
            dst = public / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp / rel, dst)

    # ── 3) Stage → verify the staged bytes → commit (+ tag/push) ────────────
    # --verify runs between `git add -A` and the commit: the --index-only
    # gates then scan exactly the content this release would push (same
    # semantics as the public CI lint job), and a failure aborts before
    # anything irreversible happens.
    _run(["git", "add", "-A"], public)
    if not _run(["git", "status", "--porcelain"], public).stdout.strip():
        print("nothing to release: public repo already matches the snapshot")
        return 0

    if args.verify and not _verify_release(public, args.verify):
        print(
            f"error: --verify={args.verify} failed — release aborted BEFORE commit/tag/push.\n"
            f"  The staged release is preserved in {public} for inspection:\n"
            f"    git -C {public} diff --cached --stat        (what would have shipped)\n"
            f"  To retry after fixing the failure, clean the tree first — a bare\n"
            f"  `git reset` only unstages: the synced snapshot edits stay in the\n"
            f"  working tree and the dirty-tree preflight blocks the next run:\n"
            f"    git -C {public} stash push -u                (the snapshot regenerates\n"
            f"                                                 deterministically — safe to drop)",
            file=sys.stderr,
        )
        return 1

    src_head = _run(["git", "rev-parse", "--short", "HEAD"], REPO).stdout.strip()
    msg = f"release: v{version} (source snapshot {src_head})"
    r = _run(["git", "commit", "-m", msg], public)
    if r.returncode != 0:
        print(f"error: public commit failed:\n{r.stderr}", file=sys.stderr)
        return 1
    print(f"committed: {msg}  ({len(shipped)} files shipped, {removed} stale files removed)")

    if args.tag:
        t = _run(["git", "tag", f"v{version}"], public)
        print(f"tagged v{version}" if t.returncode == 0
              else f"tag failed (exists?): {t.stderr.strip()}")

    if args.push:
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], public).stdout.strip()
        p = _run(["git", "push", "origin", branch], public)
        if p.returncode != 0:
            print(f"error: push failed:\n{p.stderr}", file=sys.stderr)
            return 1
        if args.tag:
            t = _run(["git", "push", "origin", f"v{version}"], public)
            if t.returncode != 0:
                print(f"error: tag push failed:\n{t.stderr}", file=sys.stderr)
                return 1
        print(f"pushed {branch} to origin")
    else:
        print(f"not pushed — review with:  git -C {public} show --stat HEAD\n"
              f"then push with:            git -C {public} push origin main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
