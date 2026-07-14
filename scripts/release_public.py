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

import re
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import export_public  # noqa: E402  (reuse the exclusion rules verbatim)


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def _version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else "0.0.0"


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

    # ── 1) Export snapshot to a temp dir ───────────────────────────────────
    shipped: list[str] = []
    for rel in export_public.tracked_files():
        if export_public.is_excluded(rel) is None:
            shipped.append(rel)
    shipped_set = set(shipped)

    with tempfile.TemporaryDirectory(prefix="asicode-release-") as td:
        tmp = Path(td)
        for rel in shipped:
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)

        # ── 2) Sync into public working tree (delete + overwrite) ──────────
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

    # ── 3) Commit (+ tag/push) in the public repo ──────────────────────────
    _run(["git", "add", "-A"], public)
    if not _run(["git", "status", "--porcelain"], public).stdout.strip():
        print("nothing to release: public repo already matches the snapshot")
        return 0

    version = _version()
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
            _run(["git", "push", "origin", f"v{version}"], public)
        print(f"pushed {branch} to origin")
    else:
        print(f"not pushed — review with:  git -C {public} show --stat HEAD\n"
              f"then push with:            git -C {public} push origin main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
