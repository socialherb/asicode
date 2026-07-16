"""
Common utility functions for asicode.

Eliminates duplicated utility functions across modules.
"""
from __future__ import annotations


def unique_keep_order(items: list) -> list:
    """Deduplicate items while preserving insertion order.

    Previously duplicated in run_helpers.py and context_collector.py.
    """
    seen: set = set()
    out: list = []
    for x in items or []:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def safe_filename(name: str, max_len: int = 128) -> str:
    """Sanitize a string to be a valid filename.

    Replaces characters invalid on Windows/Linux (\\ / : * ? " < > |) with underscore,
    compresses consecutive underscores, strips leading/trailing spaces and dots,
    truncates to max_len, and returns 'unnamed' if the result is empty.
    """
    # Replace invalid characters with underscore
    result = name.translate(str.maketrans({c: '_' for c in '\\/:*?"<>|'}))
    # Compress consecutive underscores
    while '__' in result:
        result = result.replace('__', '_')
    # Strip leading/trailing spaces and dots
    result = result.strip(' .')
    # Truncate to max_len
    if len(result) > max_len:
        result = result[:max_len].rstrip('_. ')
    # Return 'unnamed' if empty
    return result if result else 'unnamed'


def norm_ws(s: str) -> str:
    """Normalize whitespace: strip leading/trailing spaces."""
    return (s or "").strip()


def ensure_trailing_newline(s: str) -> str:
    """Ensure the string ends with a newline, normalizing line endings."""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return s if s.endswith("\n") else (s + "\n")


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    if items is None:
        return []
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def normalize_rel_path_fast(rel_path: str) -> str:
    """Quick normalization of repo-relative path.

    - Strips surrounding whitespace
    - Replaces backslashes with forward slashes
    - Removes repeated leading ``./`` (loop, not lstrip — avoids eating ``.gitignore``)
    - Removes leading ``/``
    - Returns empty string for empty/None input

    This is the canonical single-source helper for path normalization.
    All previous callers that used the defective inline chain
    ``.strip().removeprefix("/").removeprefix("./")`` have been migrated to call
    this function (or ``path_security.normalize_rel_path``, which additionally
    strips quotes, removes ``a/``/``b/`` prefixes, and rejects traversal).
    No defective chain remains in the repo as of 2026-07.
    """
    p = (rel_path or "").strip()
    p = p.replace("\\", "/")
    # Use loop instead of lstrip to avoid character-set stripping
    # (lstrip("./") would turn .gitignore into gitignore)
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    return p
