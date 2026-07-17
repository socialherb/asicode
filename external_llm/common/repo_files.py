"""Shared SSOT for listing repo files via ``git ls-files``.

Both ``write_tools._repo_file_index`` (did-you-mean suggestions) and
``symbol_index._collect_mtimes`` (import-vs-create resolution) must enumerate
the *same* set of repo files. Sourcing from a single ``git ls-files -z`` call
guarantees three properties a separate ``os.walk`` + hardcoded skip-set cannot:

  * ``.gitignore`` is respected automatically — no skip-set drift (the legacy
    ``_SKIP_DIRS`` had ``.venv`` but not ``venv``/``vendor/``/``third_party/``);
  * non-ASCII (Korean/CJK) paths survive unmangled (``-z`` — porcelain output
    C-quotes them by default, which would then fail membership tests);
  * vendored / generated copies (``vendor/``, ``*_pb2.py``) do NOT leak into
    the symbol index — a leaked symbol would flip a correct ``"create"`` into
    a wrong ``"import"`` and risk a DUPLICATE DEFINITION.

Constraint: ``common`` must NOT import ``agent`` (design-insight invariant).
This module is pure stdlib (``os`` + ``subprocess``), so it sits at the bottom
of the dependency graph and is safe for any layer to import.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional


def git_list_repo_files(repo_root: str) -> Optional[list[str]]:
    """Return sorted repo-relative paths via ``git ls-files`` (NUL-separated).

    Uses ``-z`` (REQUIRED — porcelain output C-quotes non-ASCII paths like
    Korean, which would then fail membership tests downstream) + ``--cached``
    (tracked) + ``--others --exclude-standard`` (untracked but NOT gitignored)
    = every file visible in the work tree, with ``.gitignore`` respected
    automatically — so a hardcoded skip-set is not needed on the git path.

    Returns ``None`` when git is unavailable, the path is not a git checkout,
    or the call fails — callers fall back to ``os.walk`` in that case. A
    ``None`` return (NOT an empty list) is what distinguishes "git unusable,
    please walk" from "git OK, repo has zero files".
    """
    if not os.path.exists(os.path.join(repo_root, ".git")):
        return None
    try:
        r = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            timeout=15,
        )
        if r.returncode != 0:
            return None
        out = r.stdout.decode("utf-8", "replace")
        paths = [_p for _p in out.split("\0") if _p]
        paths.sort()
        return paths
    except Exception:
        return None
