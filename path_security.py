"""
Unified path security utilities for asicode.

All repo-relative path validation and resolution goes through this module.
Prevents path traversal, absolute paths, and escape from repo root.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_rel_path(p: str) -> str:
    """
    Normalize a repo-relative file path.

    - Strips quotes, whitespace
    - Removes a/ or b/ prefixes (git diff convention)
    - Removes leading ./
    - Rejects absolute paths, drive letters, and traversal (..)
    - Returns empty string if invalid
    """
    p = (p or "").strip().strip('"').strip("'")
    p = p.replace("\\", "/")
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")

    if not p:
        return ""
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":" and p[0].isalpha()):
        return ""
    parts = [x for x in p.split("/") if x]
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def resolve_inside_repo(repo_root: str, rel_path: str) -> Path:
    """
    Resolve a repo-relative path and ensure it stays inside repo_root.

    Raises ValueError if the path escapes the repo boundary.
    """
    repo = Path(repo_root).resolve()
    rel = normalize_rel_path(rel_path)
    if not rel:
        raise ValueError(f"path_invalid: '{rel_path}'")
    p = (repo / rel).resolve()
    try:
        p.relative_to(repo)
    except ValueError:
        raise ValueError(f"path_outside_repo: '{rel_path}' resolves outside '{repo}'")
    return p


def resolve_under_repo_subdir(repo_root: str, subdir: str, candidate: str) -> Path:
    """Resolve *candidate* and ensure it is contained within ``repo_root/subdir``.

    Unlike :func:`resolve_inside_repo` (which takes a repo-relative path and
    rejects absolute paths), this accepts an already-absolute path — e.g. the
    design-chat continuation producer writes an absolute path under
    ``repo_root/.asicode/continuation/``. It constrains a raw, attacker-
    controlled path (such as a query param) *before* it reaches ``open()``,
    closing a path-traversal / arbitrary-file-read surface.

    A relative *candidate* is anchored at *repo_root* (not the process CWD) so
    it cannot escape via a relative traversal either.

    Uses ``Path.resolve()`` + ``relative_to`` for boundary-correct containment
    — a naive ``str.startswith`` would wrongly accept ``/repo/.asicode-evil``
    as being "under" ``/repo/.asicode``.

    Raises ``ValueError`` if *candidate* resolves outside the allowed dir.
    """
    repo = Path(repo_root).resolve()
    allowed_dir = (repo / subdir).resolve()
    cand = Path(candidate)
    if not cand.is_absolute():
        cand = repo / cand
    p = cand.resolve()
    try:
        p.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(
            f"path_outside_allowed: {candidate!r} resolves to {p}, not under {allowed_dir}"
        )
    return p


def _repo_within_allowlist(repo: Path, root: str) -> bool:
    """True if *repo* (already resolved) is *root* itself or below it.

    Uses ``relative_to`` for path-boundary-correct semantics so that an
    allowlist entry ``/home/dev/projects`` does NOT match a sibling like
    ``/home/dev/projects-evil`` — which a naive ``str.startswith`` would
    wrongly accept. This is a real security boundary reachable from the webapp
    via the attacker-controlled ``req.repo_root``.

    Both sides are canonicalized via ``Path.resolve()`` so symlinks and
    trailing slashes are handled consistently, matching ``resolve_inside_repo``.
    """
    try:
        resolved_root = Path(root).resolve()
    except OSError:
        return False
    if resolved_root == repo:
        return True
    try:
        repo.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def validate_repo_root(repo_root: str, allowed_roots: list[str] | None = None) -> Path:
    """
    Validate that repo_root exists, is a git repo, and optionally
    falls within the allowed root prefixes.

    Returns the resolved Path.
    Raises ValueError with descriptive messages.
    """
    repo = Path(repo_root).resolve()
    if not repo.exists():
        raise ValueError(f"repo_root not found: {repo}")
    if not (repo / ".git").exists():
        raise ValueError(f"repo_root is not a git repo: {repo}")

    # If allowed_roots is an empty list, treat it as None (allow all)
    if allowed_roots is not None and len(allowed_roots) == 0:
        allowed_roots = None

    if allowed_roots:
        if not any(_repo_within_allowlist(repo, root) for root in allowed_roots):
            logger.warning("Rejected repo_root outside allowlist: %s", repo)
            raise ValueError(f"repo_root not in allowed list: {repo}")

    return repo
