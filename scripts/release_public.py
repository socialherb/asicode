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
    --allow-dirty   skip the clean-working-tree check (testing only — a dirty
                    tree means uncommitted edits of tracked files get published)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
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


# Root-level ignored .py files that are known personal/dev-only scripts. Root
# .py files ship (asi.py, config.py, ...), so an ignored root module is the
# same wheel-vanishing hazard as an ignored module under a shipping package —
# except these, which have zero shipping references (verified 2026-07-30).
_IGNORED_ROOT_ALLOWLIST = frozenset({"radio.py"})


def _ignored_shipping_py(ignored_lines: list[str], shipped: list[str]) -> list[str]:
    """Return gitignored ``.py`` files that would silently vanish from the wheel.

    ``git status --porcelain`` does not report ignored files, so the clean-tree
    check passes while a gitignored module is absent from the export (which
    copies tracked files only) — the 0.2.6 ``version_check`` class of bug. The
    hazard scope is: any ignored ``.py`` under a top-level directory that
    contains shipped ``.py`` files, plus any ignored root-level ``.py`` (root
    modules ship too), minus the documented allowlist.
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
        elif rel not in _IGNORED_ROOT_ALLOWLIST:
            # Root .py files ship too; any ignored root module is suspicious
            # except the documented personal-script allowlist.
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
            "  git add <file>, or delete the file. Known-benign root allowlist: "
            + ", ".join(sorted(_IGNORED_ROOT_ALLOWLIST)),
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


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    target_arg = args[0] if args else os.environ.get("ASICODE_PUBLIC_REPO", "")
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
    if dirty and "--allow-dirty" not in flags:
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
    shipped = [rel for rel in export_public.tracked_files()
               if export_public.is_excluded(rel) is None]
    if not _check_ignored_shipping_py(shipped):
        return 1

    # ── 2) Export snapshot to a temp dir ───────────────────────────────────
    shipped_set = set(shipped)

    with tempfile.TemporaryDirectory(prefix="asicode-release-") as td:
        tmp = Path(td)
        for rel in shipped:
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)

        # Prune excluded-path entries from lint baselines so the public
        # snapshot stays self-consistent (no dangling references to excluded
        # modules). Mirrors export_public.main(); without this, baselines
        # copied verbatim still reference modules that don't ship.
        export_public.prune_baseline_files(tmp, shipped)

        # ── 3) Sync into public working tree (delete + overwrite) ──────────
        removed = 0
        for p in sorted(public.rglob("*"), reverse=True):
            rel = p.relative_to(public).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                continue
            if p.is_file() and rel not in shipped_set:
                p.unlink()
                removed += 1
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        for rel in shipped:
            dst = public / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp / rel, dst)

    # ── 4) Commit (+ tag/push) in the public repo ──────────────────────────
    _run(["git", "add", "-A"], public)
    if not _run(["git", "status", "--porcelain"], public).stdout.strip():
        print("nothing to release: public repo already matches the snapshot")
        return 0

    src_head = _run(["git", "rev-parse", "--short", "HEAD"], REPO).stdout.strip()
    msg = f"release: v{version} (source snapshot {src_head})"
    r = _run(["git", "commit", "-m", msg], public)
    if r.returncode != 0:
        print(f"error: public commit failed:\n{r.stderr}", file=sys.stderr)
        return 1
    print(f"committed: {msg}  ({len(shipped)} files shipped, {removed} stale files removed)")

    if "--tag" in flags:
        t = _run(["git", "tag", f"v{version}"], public)
        print(f"tagged v{version}" if t.returncode == 0
              else f"tag failed (exists?): {t.stderr.strip()}")

    if "--push" in flags:
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], public).stdout.strip()
        p = _run(["git", "push", "origin", branch], public)
        if p.returncode != 0:
            print(f"error: push failed:\n{p.stderr}", file=sys.stderr)
            return 1
        if "--tag" in flags:
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
