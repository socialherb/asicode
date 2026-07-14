"""
Diff cleaning, validation, and application for asicode.

Handles:
- LLM output cleaning (strip fences, noise, recount hunks)
- git apply with 3-way fallback
- Conflict marker detection and rollback
- Safety guards (large files, binary files, path traversal)
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import normalize_rel_path_fast
from config import (
    BINARY_SNIFF_BYTES,
    CONFLICT_MARKER_MAX_BYTES,
    LARGE_FILE_MAX_BYTES,
    STRICT_CLEAN,
)
from patch_synth import synthesize_append_line_unified_diff
from path_security import normalize_rel_path

logger = logging.getLogger(__name__)


# =========================================================
# Diff change counting
# =========================================================
# Diff cleaning utilities
# =========================================================

# Accept any fenced code block opener (```diff / ```patch / ```text / ```python ...)
_FENCE_RE = re.compile(r"^\s*```[^\n\r]*\s*$", re.IGNORECASE)
_END_FENCE_RE = re.compile(r"^\s*```\s*$")
_DIFF_START_RE = re.compile(r"^\s*diff\s+--git\s+", re.IGNORECASE)
_HUNK_HEADER_RE = re.compile(r"^\s*@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")
_PATH_PREFIX_RE = re.compile(r"^(?:a/|b/)")

# Static regex patterns (module-level so they are compiled once, not per call).
# See _extract_files_from_git_apply_output and _has_conflict_markers.
_GIT_APPLY_FILE_PATTERNS = (
    re.compile(r"patch failed:\s+(.+?):\d+", re.IGNORECASE),
    re.compile(r"Checking patch\s+(.+?)\.\.\.", re.IGNORECASE),
    re.compile(r"Applying patch\s+(.+?)\.\.\.", re.IGNORECASE),
    re.compile(r"error:\s+(.+?):\s+(No such file|does not exist|cannot|can't|failed)", re.IGNORECASE),
)


def _clean_diff_lines(text: str, strict: bool) -> list[str]:
    """Remove noise from LLM output, returning candidate diff lines."""
    if not text:
        return []

    lines: list[str] = []
    in_fence = False

    for raw in str(text).splitlines():
        ln = raw

        # --- strip agent hints / chain hints injected by LLM ---
        if ln.startswith("[CHAIN-HINT]") or ln.startswith("[TOOL CHAIN HINT]"):
            continue
        if ln.startswith("Typical next steps"):
            continue
        # --- end hint strip ---
        # Check end fence first when inside a fence (bare ``` closes it)
        if in_fence and _END_FENCE_RE.match(ln):
            in_fence = False
            continue
        # Opening fence (``` with optional language tag like ```diff)
        if _FENCE_RE.match(ln):
            in_fence = True
            continue
        # Drop git "index ..." lines (often bogus from LLM)
        if ln.startswith("index "):
            continue
        lines.append(ln)

    # Trim empty edges
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return []

    if strict:
        has_header = any(_DIFF_START_RE.match(_item_) for _item_ in lines)
        has_hunk = any(_HUNK_HEADER_RE.match(_item_) for _item_ in lines)
        if not (has_header or has_hunk):
            return []

        # Skip leading non-diff junk
        i = 0
        while i < len(lines):
            if (
                _DIFF_START_RE.match(lines[i])
                or _HUNK_HEADER_RE.match(lines[i])
                or lines[i].startswith("--- ")
                or lines[i].startswith("+++ ")
            ):
                break
            i += 1
        lines = lines[i:] if i < len(lines) else []

    # Truncate trailing non-diff content (strict)
    #
    # COUNTING-BASED PARSER (Design 11):
    # Instead of heuristic noise-budget or regex header checks, use hunk
    # line-count validation to determine content boundaries. A line starting
    # with "--- " inside a hunk body is hunk content (deleted line), not a
    # diff header — the counting approach eliminates this ambiguity.
    #
    # - Hunk starts with @@ header
    # - Inside hunk body: only +, -, space, empty, and \ lines are valid
    # - Hunk ends when its claimed line counts are consumed OR a next
    #   hunk/diff header is encountered
    # - Non-diff lines outside hunk are dropped (no noise budget needed)
    if strict and lines:
        kept: list[str] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            # Diff/file header: always kept
            if _is_diff_header_line(ln):
                kept.append(ln)
                i += 1
                continue
            # Hunk header: parse and count body
            if _HUNK_HEADER_RE.match(ln):
                kept.append(ln)
                result = _count_hunk_body(lines, i)
                if result is not None:
                    end, actual_old, actual_new, claimed_old, claimed_new = result
                    # Accept hunk body up to end (may be truncated LLM output)
                    for j in range(i + 1, end):
                        kept.append(lines[j])
                    i = end
                else:
                    i += 1
                continue
            # ---/+++ file header pair WITHOUT a/b/ prefix (common in LLM output).
            # _is_diff_header_line requires "--- a/" or "--- /dev/null", but LLMs
            # frequently emit "--- f.py" / "+++ f.py". Detect these as a consecutive
            # pair — a lone "--- " line could be an SQL/Lua comment deletion inside
            # a hunk, but the hunk-body counter already consumed those before we get
            # here (they start with "-"), so pair detection outside hunks is safe.
            if ln.startswith("--- ") and not ln.startswith(("--- a/", "--- /dev/null")):
                if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                    kept.append(ln)
                    kept.append(lines[i + 1])
                    i += 2
                else:
                    i += 1  # lone "--- " without matching "+++ " → skip
                continue
            # "+++ " without b/ prefix: keep if the preceding kept line is a "--- "
            # file header (either canonical "--- a/" or bare "--- f.py" from pair
            # detection above). Otherwise skip (standalone "+++ " is not a valid header).
            if ln.startswith("+++ ") and not ln.startswith(("+++ b/", "+++ /dev/null")):
                if kept and kept[-1].startswith("--- "):
                    kept.append(ln)
                i += 1
                continue
            # Before first diff/hunk header: skip (non-diff preamble)
            if not kept:
                i += 1
                continue
            # After diff/hunk content: trailing noise → skip
            i += 1
        lines = kept

    return lines


def _parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    """Parse @@ -a,b +c,d @@ → (old_start, old_count, new_start, new_count)."""
    m = re.match(r"^\s*@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or "1"), int(m.group(3)), int(m.group(4) or "1")


def _is_diff_header_line(ln: str) -> bool:
    """Check if line is a diff/file header (not hunk content).

    In standard unified-diff format:
      - ``--- a/<path>`` or ``--- /dev/null``  (old file)
      - ``+++ b/<path>`` or ``+++ /dev/null``  (new file)
    We require ``a/``/``b/`` prefix (or ``/dev/null``) to disambiguate
    from hunk content lines starting with ``--- `` (e.g. SQL/Lua comments).
    """
    if not ln:
        return False
    if ln.startswith("diff --git "):
        return True
    if ln.startswith("--- a/") or ln.startswith("--- /dev/null"):
        return True
    if ln.startswith("+++ b/") or ln.startswith("+++ /dev/null"):
        return True
    if ln.startswith(("new file mode ", "deleted file mode ", "old mode ", "new mode ")):
        return True
    if ln.startswith(("similarity index ", "rename from ", "rename to ")):
        return True
    return False


def _count_hunk_body(lines: list[str], start: int) -> tuple[int, int, int, int, int] | None:
    """
    From a hunk header at lines[start], count actual hunk body lines
    and validate against the header's claimed counts.

    Returns (end_idx, actual_old, actual_new, claimed_old, claimed_new)
    or None if the header can't be parsed.
    A hunk body may be shorter than claimed (trailing context can be elided
    in LLM output); actual counts are clamped at claimed.

    After claimed counts are met, only context lines (`` ``-prefix, ``\\ ``)
    continue to be absorbed — blank lines, ``-``/``+`` lines after that
    point are considered a new hunk/file boundary.  This prevents:
      - Markdown bullets (``- ...``) from being absorbed.
      - Multi-file bare headers (``--- f2.py``) from being counted as hunk
        deletions.
      - Blank separator lines between hunks from being counted as phantom
        trailing context.
    """
    parsed = _parse_hunk_header(lines[start])
    if not parsed:
        return None
    _, claimed_old, _, claimed_new = parsed

    i = start + 1
    actual_old = 0
    actual_new = 0

    while i < len(lines):
        ln = lines[i]
        # Next hunk header or diff header → hunk boundary
        if _HUNK_HEADER_RE.match(ln) or _is_diff_header_line(ln):
            break
        # Bare ---/+++ pair without a/b prefix → file header boundary
        # (needed since LLM often omits the a/b prefix).
        # Single "--- comment" without matching "+++" is NOT a boundary
        # (safe: SQL/Lua comments are consumed as deletion lines if they
        # appear inside a hunk body; at hunk boundaries they stay noise).
        if ln.startswith("--- ") and not ln.startswith(("--- a/", "--- /dev/null")):
            if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                break  # bare file header pair → hunk boundary
        # No-newline-at-EOF marker (\) — meta, not counted
        if ln.startswith("\\ "):
            i += 1
            continue
        if ln.startswith(" "):
            actual_old += 1
            actual_new += 1
        elif ln.startswith("-"):
            actual_old += 1
        elif ln.startswith("+"):
            actual_new += 1
        elif ln == "" or ln.strip() == "":
            # Blank/empty lines are valid context lines (git apply counts them).
            # Not counting them causes recount to produce headers that misalign
            # with git apply's own counting, resulting in "patch does not apply".
            actual_old += 1
            actual_new += 1
        else:
            # Non-hunk content → hunk boundary (truncated LLM output)
            break

        i += 1

        # After counts are met, only absorb trailing context lines
        # (space-prefixed, \ no-newline).  Stop on blank, -/+, or
        # anything else: blank lines are separators between hunks/files,
        # not legitimate trailing context (legitimate context is always
        # space-prefixed in unified diff format).
        if (actual_old >= claimed_old and actual_new >= claimed_new) and (
            i >= len(lines) or not lines[i].startswith((" ", "\\ "))
        ):
            break

    return i, actual_old, actual_new, claimed_old, claimed_new


def _recount_hunks(lines: list[str]) -> list[str]:
    """Recompute line counts for each hunk header using counting-based parser."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.lstrip().startswith("@@"):
            out.append(ln)
            i += 1
            continue

        parsed = _parse_hunk_header(ln)
        if not parsed:
            out.append(ln)
            i += 1
            continue

        old_a, _, new_a, _ = parsed

        result = _count_hunk_body(lines, i)
        if result is not None:
            end, actual_old, actual_new, _, _ = result
        else:
            end = i + 1
            actual_old = actual_new = 0

        out.append(f"@@ -{old_a},{actual_old} +{new_a},{actual_new} @@")
        for j in range(i + 1, end):
            out.append(lines[j])
        i = end

    return out


def _rewrite_patch_paths(lines: list[str], target_rel: str) -> list[str]:
    """Enforce patch paths to target file, preserving /dev/null for new files.

    Uses counting-based parser to avoid rewriting --- /+++ inside hunk bodies
    (e.g., SQL/Lua comment deletions like "--- old comment").
    """
    if not target_rel:
        return lines
    t = normalize_rel_path(target_rel)
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Hunk header: use _count_hunk_body to determine hunk extent
        if _HUNK_HEADER_RE.match(ln):
            out.append(ln)
            parsed = _count_hunk_body(lines, i)
            if parsed:
                end, _, _, _, _ = parsed
                # Copy hunk body as-is (no path rewriting inside hunks)
                for j in range(i + 1, end):
                    out.append(lines[j])
                i = end
            else:
                i += 1
        # Diff/file headers: rewrite paths
        elif ln.startswith("diff --git "):
            out.append(f"diff --git a/{t} b/{t}")
            i += 1
        elif ln.startswith("--- "):
            out.append("--- /dev/null" if ln.strip() == "--- /dev/null" else f"--- a/{t}")
            i += 1
        elif ln.startswith("+++ "):
            out.append("+++ /dev/null" if ln.strip() == "+++ /dev/null" else f"+++ b/{t}")
            i += 1
        else:
            out.append(ln)
            i += 1
    return out


def _upgrade_hunk_fragment(lines: list[str], target_rel: str) -> list[str]:
    """Wrap naked hunk(s) into a minimal unified diff header."""
    t = normalize_rel_path(target_rel)
    if not t or not lines:
        return lines
    if any(_item_.startswith("diff --git ") for _item_ in lines):
        return lines
    if not any(_item_.startswith("@@") for _item_ in lines):
        return lines
    return [f"diff --git a/{t} b/{t}", f"--- a/{t}", f"+++ b/{t}", *lines]


def _clean_diff(diff_text: str, repo_root: str, file_path_hint: str | None = None) -> str:
    """Full cleaning pipeline: parse, upgrade, rewrite paths, recount."""
    lines = _clean_diff_lines(diff_text, strict=STRICT_CLEAN)
    if not lines:
        return ""

    if file_path_hint:
        lines = _upgrade_hunk_fragment(lines, file_path_hint)

    if file_path_hint and any(_item_.startswith("diff --git ") for _item_ in lines):
        lines = _rewrite_patch_paths(lines, file_path_hint)

    lines = _recount_hunks(lines)
    # IMPORTANT:
    # Do NOT use `.rstrip()` here because it strips spaces/tabs from the final
    # hunk lines, which can invalidate blank-line entries in hunks and cause:
    #   git apply --check -> "corrupt patch"
    return "\n".join(lines).rstrip("\n") + "\n"


# =========================================================
# Apply failure reason typing
# =========================================================

REASON_OK = "OK"
REASON_CONFLICT = "CONFLICT"
REASON_PATCH_MALFORMED = "PATCH_MALFORMED"
REASON_PATH_INVALID = "PATH_INVALID"
REASON_REPO_NOT_FOUND = "REPO_NOT_FOUND"
REASON_EMPTY_DIFF = "EMPTY_DIFF"
REASON_UNKNOWN = "UNKNOWN"
REASON_CONFLICT_MARKERS = "CONFLICT_MARKERS"
REASON_SKIPPED_LARGE_FILE = "SKIPPED_LARGE_FILE"
REASON_SKIPPED_BINARY_FILE = "SKIPPED_BINARY_FILE"


def _classify_git_apply_output(out: str) -> str:
    s = (out or "").lower()
    if "corrupt patch" in s or "malformed" in s or "patch fragment" in s or "lacks the necessary blob" in s:
        return REASON_PATCH_MALFORMED
    if "no such file or directory" in s or "can't find file" in s:
        return REASON_PATH_INVALID
    if "patch failed" in s or "hunk failed" in s or "does not apply" in s:
        return REASON_CONFLICT
    return REASON_UNKNOWN


# =========================================================
# Public helpers
# =========================================================

def extract_touched_files_from_diff(diff_text: str) -> list[str]:
    """Parse touched files from unified diff."""
    touched: list[str] = []
    for ln in (diff_text or "").splitlines():
        if ln.startswith("diff --git "):
            parts = ln.split()
            if len(parts) >= 4:
                a_path = _PATH_PREFIX_RE.sub("", parts[2])
                b_path = _PATH_PREFIX_RE.sub("", parts[3])
                rel = (b_path or a_path).strip().replace("\\", "/").lstrip("/")
                if rel and rel not in touched:
                    touched.append(rel)
    return touched


# =========================================================
# Git apply execution
# =========================================================

def _run_git_apply(repo: Path, args: list[str], input_text: str | None = None) -> tuple[int, str]:
    cmd = ["git", "apply", *args]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(repo),
            input=(input_text.encode("utf-8") if input_text is not None else None),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,  # Prevent hanging on large patches
        )
        return p.returncode, p.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return -1, "git apply timeout after 30 seconds"
    except Exception as e:
        return -1, f"git apply exception: {e}"

def _git_status_porcelain(repo: Path, *, include_untracked: bool = True) -> str:
    """Return `git status --porcelain` output (best-effort)."""
    try:
        cmd = ["git", "status", "--porcelain"]
        if not include_untracked:
            cmd.append("--untracked-files=no")
        p = subprocess.run(
            cmd,
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,  # bound HTTP request; TimeoutExpired caught below
        )
        return (p.stdout or b"").decode("utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired:
        # Distinguish a hung git from a genuinely clean tree: returning "" makes
        # _is_worktree_clean() report True, so a timeout would silently masquerade
        # as "no changes" and let _rollback conclude there is nothing to restore.
        logger.warning(
            "git status timed out (repo=%s); returning empty — rollback/clean "
            "checks may misjudge a timeout as 'no changes'",
            repo,
        )
        return ""
    except Exception:
        return ""


def _is_worktree_clean(repo: Path) -> bool:
    """True if there are no uncommitted *tracked* changes (best-effort).

    NOTE:
    - Untracked files (??) are ignored, because they don't affect `git apply --3way`
      safety for tracked file merges, and blocking on them hurts UX.
    """
    return _git_status_porcelain(repo, include_untracked=False).strip() == ""



def _cleanup_reject_files(repo: Path, touched_files: list[str]) -> None:
    """Clean up .rej and .orig files for touched files only (no global rglob)."""
    for rel in touched_files:
        p = repo / rel
        for suffix in (".rej", ".orig"):
            try:
                candidate = Path(str(p) + suffix)
                if candidate.exists():
                    candidate.unlink()
            except Exception as e:
                logger.debug("Failed to clean %s: %s", candidate, e)


def _extract_files_from_git_apply_output(stderr_text: str) -> list[str]:
    """Best-effort extraction of file paths from git apply output."""
    if not stderr_text:
        return []

    files: set[str] = set()

    for ln in stderr_text.splitlines():
        s = ln.strip()
        for pat in _GIT_APPLY_FILE_PATTERNS:
            m = pat.search(s)
            if m:
                files.add(m.group(1).strip())
                break

    normed: set[str] = set()
    for f in files:
        f = f.strip().strip('"').strip("'")
        if f.startswith("a/") or f.startswith("b/"):
            f = f[2:]
        f = f.rstrip(".,;:")
        if f and f not in ("dev/null", "/dev/null"):
            normed.add(f)
    return sorted(normed)


def _is_probably_binary_file(path: Path, sniff_bytes: int) -> bool:
    try:
        with open(path, "rb") as f:
            b = f.read(max(1, sniff_bytes))
        return b"\x00" in b
    except Exception:
        return False


def _has_conflict_markers(path: Path, max_bytes: int) -> bool:
    """
    Detect git conflict markers (<<<<<<<, =======, >>>>>>>).
    Requires a complete conflict block (open → separator → close, in order)
    to avoid false positives from Markdown setext headings (=======) or RST
    dividers.

    Uses ALL match positions per marker (not just the first) so that a stray
    separator-like line (e.g. a setext '=======' heading) appearing BEFORE a
    real conflict block does not cause a false negative: the first-match
    approach would pick the heading as the separator, fail the open<sep<close
    ordering, and miss the genuine conflict block that follows.
    """
    try:
        size = path.stat().st_size
        to_read = min(size, max_bytes) if max_bytes > 0 else size
        with open(path, "rb") as f:
            b = f.read(to_read)
        text = b.decode("utf-8", errors="ignore")

        # Collect line-anchored start positions per marker type (ascending).
        opens = [m.start() for m in re.finditer(r"^<{7}", text, re.MULTILINE)]
        if not opens:
            return False
        seps = [m.start() for m in re.finditer(r"^={7}", text, re.MULTILINE)]
        closes = [m.start() for m in re.finditer(r"^>{7}", text, re.MULTILINE)]
        if not seps or not closes:
            return False

        # A real conflict block exists iff some open < separator < close triple
        # can be formed. Positions are ascending, so the first separator after
        # each open and the first close after that separator settle it.
        for o in opens:
            s = next((x for x in seps if x > o), None)
            if s is None:
                continue
            if any(c > s for c in closes):
                return True
        return False
    except Exception:
        return False


def _resolve_inside_repo_path(repo: Path, rel: str) -> Path:
    """Resolve repo/rel and ensure result stays inside repo."""
    base = repo.resolve()
    p = (base / rel).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        raise ValueError(f"path_outside_repo: {rel}")
    return p


# =========================================================
# 3-way fallback logic (deduplicated)
# =========================================================

def _git_status_untracked(repo: Path) -> set[str]:
    """
    Return current untracked file paths (relative, normalized with '/').
    Uses porcelain -z for robustness.
    """
    try:
        p = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,  # bound HTTP request; TimeoutExpired caught below
        )
        out = (p.stdout or b"").decode("utf-8", errors="replace")
        items = [x for x in out.split("\x00") if x]
        untracked: set[str] = set()

        i = 0
        while i < len(items):
            rec = items[i]
            if len(rec) >= 3 and rec[:2] == "??" and rec[2] == " ":
                path = rec[3:].strip().replace("\\", "/").lstrip("/")
                if path:
                    untracked.add(path)
                i += 1
                continue

            # Renames in -z have: "R  old\0new\0" (or similar). Skip the extra path if present.
            if len(rec) >= 3 and rec[2] == " " and rec[:2].strip() and rec[:2] != "??":
                # If it's a rename/copy, git emits an additional path item.
                if rec[0] in ("R", "C") or rec[1] in ("R", "C"):
                    i += 2
                else:
                    i += 1
                continue

            i += 1

        return untracked
    except subprocess.TimeoutExpired:
        # An empty set means "nothing was pre-existing untracked", so on a timeout
        # _rollback could wrongly delete files it considers newly-created. Log so a
        # hung git is observable instead of silently degrading rollback safety.
        logger.warning(
            "git status -z timed out (repo=%s); returning empty set — rollback "
            "snapshot may be incomplete (nothing recorded as pre-existing untracked)",
            repo,
        )
        return set()
    except Exception:
        return set()


def _capture_rollback_snapshot(repo: Path, touched_files: list[str]) -> dict[str, Any]:
    """
    Capture enough info to rollback *completely*:
    - untracked files before apply attempt (so we don't delete user-owned untracked files)
    - existence of touched paths before apply (so we can delete newly-created touched files/dirs)
    """
    pre_untracked = _git_status_untracked(repo)
    pre_exists: dict[str, bool] = {}
    for rel in (touched_files or []):
        try:
            p = _resolve_inside_repo_path(repo, rel)
            pre_exists[rel] = p.exists()
        except Exception:
            # touched_files are already path-safety filtered before snapshot is used
            pre_exists[rel] = False
    return {"pre_untracked": pre_untracked, "pre_exists": pre_exists}


def _delete_path_best_effort(p: Path) -> None:
    try:
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)  # py>=3.8
        elif p.is_dir():
            # remove dir tree
            for child in sorted(p.rglob("*"), reverse=True):
                try:
                    if child.is_symlink() or child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                except Exception:
                    pass
            try:
                p.rmdir()
            except Exception:
                pass
    except Exception:
        pass


def _parse_porcelain_z_paths(raw: str) -> list[str]:
    """Parse ``git status --porcelain -z`` output into a list of dirty file paths.

    Handles non-ASCII filenames (no C-quoting with -z) and rename entries
    (``R  old\\0new\\0`` where the extra path must be skipped).
    """
    items = [x for x in (raw or "").split("\x00") if x]
    paths: list[str] = []
    i = 0
    while i < len(items):
        rec = items[i]
        if len(rec) >= 4 and rec[2] == " ":
            path = rec[3:]
            if path:
                paths.append(path)
            # Renames/copies have an additional path entry in the next item
            if rec[0] in ("R", "C") or rec[1] in ("R", "C"):
                i += 2
            else:
                i += 1
        else:
            i += 1
    return paths


def _rollback(repo: Path, touched_files: list[str], snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Best-effort *complete* rollback:
    - restore tracked changes for touched files (only pre-existing ones)
    - delete newly-created touched files/dirs that didn't exist before
    - delete newly-created untracked files that appeared during apply attempt
      (without touching pre-existing untracked files)
    Returns a report dict with verification results.
    """
    report: dict[str, Any] = {
        "attempted": False,
        "verified": False,
        "restore_failed": [],
        "delete_failed": [],
        "remaining_dirty": [],
    }
    pre_exists = (snapshot or {}).get("pre_exists") or {}

    try:
        # 1) Restore tracked state for pre-existing touched files only
        # New files (pre_exists[rel] is False) would cause git to abort with
        # "pathspec did not match", leaving ALL files unrestored.
        if touched_files:
            existing_files = [rel for rel in touched_files if pre_exists.get(rel) is not False]
            if existing_files:
                report["attempted"] = True
                result = subprocess.run(
                    ["git", "restore", "--staged", "--worktree", "--", *existing_files],
                    cwd=str(repo),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,  # bound HTTP request; TimeoutExpired caught below
                )
                if result.returncode != 0:
                    logger.warning("git restore failed (rc=%d): %s", result.returncode, result.stderr)
                    report["restore_failed"] = existing_files
    except Exception as e:
        logger.warning("Rollback restore failed: %s", e)

    # 2) Delete newly-created paths among touched files (covers new files and dirs)
    try:
        for rel in (touched_files or []):
            try:
                if pre_exists.get(rel) is False:
                    p = _resolve_inside_repo_path(repo, rel)
                    if p.exists():
                        report["attempted"] = True
                        _delete_path_best_effort(p)
            except Exception:
                continue
    except Exception as e:
        logger.warning("Rollback delete-new-touched failed: %s", e)

    # 3) Delete newly-created untracked files (delta from pre_untracked)
    try:
        pre_untracked: set[str] = set((snapshot or {}).get("pre_untracked") or set())
        post_untracked = _git_status_untracked(repo)
        created = sorted(p for p in (post_untracked - pre_untracked) if p)

        if created:
            report["attempted"] = True
            # Use git clean with explicit pathspecs (safer than global clean)
            # Batch to avoid argv limits
            BATCH = 50
            for i in range(0, len(created), BATCH):
                batch = created[i : i + BATCH]
                subprocess.run(
                    ["git", "clean", "-fd", "--", *batch],
                    cwd=str(repo),
                    check=False,
                    timeout=30,  # bound HTTP request; TimeoutExpired caught below
                )
            # Extra safety: best-effort delete if git clean didn't remove something
            for rel in created:
                try:
                    p = _resolve_inside_repo_path(repo, rel)
                    if p.exists():
                        _delete_path_best_effort(p)
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Rollback clean-new-untracked failed: %s", e)

    # 4) Verification: check if touched files are actually clean
    # Use -z (NUL-separated) to robustly handle non-ASCII filenames (Korean, CJK)
    # and rename entries — plain --porcelain C-quotes non-ASCII paths and
    # renames use "R  old -> new" which breaks naive line[3:] parsing.
    if report["attempted"] and touched_files:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "-z", "--", *touched_files],
                cwd=str(repo),
                check=False,
                capture_output=True,
                text=False,  # bytes for safe NUL-split
                timeout=30,  # bound HTTP request; TimeoutExpired caught below
            )
            if result.returncode == 0:
                raw = (result.stdout or b"").decode("utf-8", errors="replace")
                dirty_paths = _parse_porcelain_z_paths(raw)
                if dirty_paths:
                    report["remaining_dirty"] = dirty_paths
                    report["verified"] = False
                else:
                    report["verified"] = True
            else:
                logger.warning("git status verification failed (rc=%d): %s",
                               result.returncode,
                               (result.stderr or b"").decode("utf-8", errors="replace"))
        except Exception as e:
            logger.warning("Rollback verification failed: %s", e)

    return report


def _rollback_warning(report: dict[str, Any]) -> str:
    """Return a user-facing warning suffix if rollback left dirty state."""
    if not report.get("attempted"):
        return ""
    if report.get("verified"):
        return ""
    dirty = report.get("remaining_dirty", [])
    if dirty:
        return f" [WARNING: repository was left in a dirty state. Remaining files: {len(dirty)}]"
    return " [WARNING: repository was left in a dirty state (unable to verify)]"


def _try_3way_fallback(
    repo: Path,
    cleaned: str,
    touched_files: list[str],
    snapshot: dict[str, Any] | None = None,
) -> tuple[bool, str, str, dict[str, Any]]:
    """
    Attempt git apply --3way. If it succeeds but leaves conflict markers, roll back.
    Returns (ok, msg, reason, details).
    """
    rc3, out3 = _run_git_apply(
        repo, ["--3way", "--recount", "--whitespace=nowarn"], input_text=cleaned,
    )
    if rc3 != 0:
        return False, out3.strip() or "3way also failed", _classify_git_apply_output(out3), {
            "touched_files": touched_files,
            "used_strategy": "git-apply-3way-failed",
            "returncode": rc3,
        }

    # Check for conflict markers
    marker_hits: list[str] = []
    for rel in touched_files:
        fp = repo / rel
        if fp.exists() and fp.is_file() and _has_conflict_markers(fp, CONFLICT_MARKER_MAX_BYTES):
            marker_hits.append(rel)

    if marker_hits:
        rollback_report = _rollback(repo, touched_files, snapshot=snapshot)
        _cleanup_reject_files(repo, touched_files)
        msg = f"3way produced conflict markers in: {', '.join(marker_hits[:8])}"
        return False, msg, REASON_CONFLICT_MARKERS, {
            "touched_files": touched_files,
            "failed_files": sorted(set(touched_files)),
            "rollback_performed": rollback_report["attempted"],
            "rollback_verified": rollback_report["verified"],
            "rollback_remaining_dirty": rollback_report["remaining_dirty"],
            "used_strategy": "git-apply-3way-marker-guard",
            "marker_files": marker_hits,
        }

    return True, "applied", REASON_OK, {
        "touched_files": touched_files,
        "failed_files": [],
        "rollback_performed": False,
        "used_strategy": "git-apply-3way",
    }


# =========================================================
# Patch apply (main entry point)
# =========================================================

def apply_patch(
    repo_root: str, diff_text: str, file_path_hint: str | None = None,
    skip_3way: bool = False,
) -> tuple[bool, str, str, dict[str, Any]]:
    """
    Apply a unified diff patch to repo_root.

    - Preflight skip for large/binary files
    - git apply with --3way fallback on conflicts
    - Conflict marker detection and rollback

    Args:
        skip_3way: When True, do NOT attempt `git apply --3way` on conflict. Set this for
            targets known to lack a pre-image blob (untracked / gitignored / freshly-edited
            files) — `--3way` would fail with "repository lacks the necessary blob to
            perform a 3-way merge" anyway, so skipping it avoids a wasted subprocess and
            surfaces the real conflict error directly. Plain `git apply` (non-3way) is
            still attempted, so well-formed patches on such files still succeed.
    """
    repo = Path(repo_root).resolve()
    execution_steps: list[dict[str, Any]] = []
    if not repo.exists():
        msg = f"repo_root not found: {repo}"
        return False, msg, REASON_REPO_NOT_FOUND, {"last_error": msg, "execution_steps": execution_steps}

    raw = "" if diff_text is None else str(diff_text)

    # Detect and wrap hunk‑only diffs (no file headers)
    lines = raw.strip().splitlines()
    first_non_empty = None
    for line in lines:
        if line.strip():
            first_non_empty = line
            break
    is_hunk_only = (
        first_non_empty is not None
        and first_non_empty.startswith("@@")
        and "diff --git" not in raw
        and "--- a/" not in raw
        and "+++ b/" not in raw
    )
    if is_hunk_only:
        if not file_path_hint:
            msg = "Hunk‑only patch requires the 'path' parameter to determine the target file"
            return False, msg, "MISSING_PATH_HINT", {"execution_steps": execution_steps}
        # Normalize path: ensure it's relative and inside repo_root
        norm = normalize_rel_path(file_path_hint)
        if not norm:
            msg = f"Invalid or unsafe path in hunk‑only patch: {file_path_hint}"
            return False, msg, "PATH_INVALID", {"execution_steps": execution_steps}
        # Construct full unified diff headers
        header = f"diff --git a/{norm} b/{norm}\n--- a/{norm}\n+++ b/{norm}\n"
        raw = header + raw

    cleaned = _clean_diff(raw, repo_root, file_path_hint=file_path_hint)
    if not cleaned.strip():
        return False, "empty diff after cleaning", REASON_EMPTY_DIFF, {"execution_steps": execution_steps}

    touched_files = extract_touched_files_from_diff(cleaned)

    # Guard: path safety
    safe_touched: list[str] = []
    unsafe: list[str] = []
    for rel in touched_files:
        norm = normalize_rel_path(rel)
        if not norm or norm != rel:
            unsafe.append(rel)
            continue
        try:
            _resolve_inside_repo_path(repo, norm)
            safe_touched.append(norm)
        except Exception:
            unsafe.append(rel)

    touched_files = safe_touched
    if unsafe:
        msg = f"unsafe path in patch: {', '.join(unsafe[:8])}"
        return False, msg, REASON_PATH_INVALID, {"unsafe_paths": unsafe, "execution_steps": execution_steps}

    # Guard: large / binary files
    for rel in touched_files:
        fp = _resolve_inside_repo_path(repo, rel)
        if not fp.exists():
            continue
        if fp.is_file() and fp.stat().st_size > LARGE_FILE_MAX_BYTES:
            msg = f"skipped: large file {rel} exceeds {LARGE_FILE_MAX_BYTES} bytes"
            return False, msg, REASON_SKIPPED_LARGE_FILE, {"skipped_large": [rel], "execution_steps": execution_steps}
        if fp.is_file() and _is_probably_binary_file(fp, BINARY_SNIFF_BYTES):
            msg = f"skipped: binary file detected: {rel}"
            return False, msg, REASON_SKIPPED_BINARY_FILE, {"skipped_binary": [rel], "execution_steps": execution_steps}

    # Capture snapshot for complete rollback (including newly-created untracked files)
    rollback_snapshot = _capture_rollback_snapshot(repo, touched_files)

    # 1) Pre-check
    # Use --recount --whitespace=nowarn to match what tolerant apply variants use.
    # Without these, a patch with minor hunk-counter drift or trailing whitespace
    # fails the pre-check even though the actual apply (or tolerant fallback)
    # would succeed — causing unnecessary rollback and retry (P3, HIGH).
    step_check = {"step": "git_apply_check", "status": "pending"}
    t0_check = time.monotonic()
    rc, out = _run_git_apply(
        repo, ["--check", "--recount", "--whitespace=nowarn"], input_text=cleaned,
    )
    step_check["duration_ms"] = int((time.monotonic() - t0_check) * 1000)
    if rc == 0:
        step_check["status"] = "ok"
    else:
        step_check["status"] = "failed"
        step_check["error"] = {"returncode": int(rc), "stdout_clip": (out or "")[:2000]}
    execution_steps.append(step_check)
    if rc != 0:
        reason = _classify_git_apply_output(out)
        if reason == REASON_CONFLICT:
            # Pre-apply gate: if caller flagged the target as lacking a pre-image blob
            # (untracked / gitignored / freshly-edited), skip 3-way entirely — it is
            # guaranteed to fail with "repository lacks the necessary blob", and the
            # autostash dance below would only churn the index for nothing. Fall through
            # to the generic conflict-failure return so patch_engine can run its repair
            # ladder (reanchor / AST / symbol) instead.
            if skip_3way:
                # Fix 2 (user-WIP protection): `git apply --check` is a NON-MUTATING
                # dry-run, so the working tree here is still in its pre-check state —
                # including any user work-in-progress (uncommitted edits). Calling
                # _rollback() would run `git restore --staged --worktree`, silently
                # reverting user-WIP to HEAD on a path that never mutated the tree.
                # _rollback's new-file cleanup (steps 2/3) is also a no-op here since
                # --check creates nothing. So: skip _rollback, only sweep stale .rej.
                _cleanup_reject_files(repo, touched_files)
                return False, out.strip() or "git apply --check failed (3way skipped: no pre-image blob)", reason, {
                    "touched_files": touched_files,
                    "failed_files": sorted(set(touched_files) | set(_extract_files_from_git_apply_output(out))),
                    "rollback_performed": False,
                    "rollback_skipped_reason": "check_is_dryrun_worktree_unchanged",
                    "used_strategy": "git-apply-check-3way-skipped",
                    "execution_steps": execution_steps,
                }

            # SAFETY: refuse 3-way merge when working tree is dirty.
            # Otherwise conflict markers may be written into files and left behind.
            #
            # Cursor/Claude-style UX: auto-stash tracked changes, attempt 3-way, then pop.
            autostash_used = False
            autostash_pop_ok = False
            autostash_pop_error = ""
            dirty_before = _git_status_porcelain(repo)[:2000]

            if not _is_worktree_clean(repo):
                try:
                    # Stash tracked changes (untracked are ignored by our cleanliness policy anyway)
                    p = subprocess.run(
                        ["git", "stash", "push", "-m", "asicode-autostash"],
                        cwd=str(repo),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=30,  # bound HTTP request; TimeoutExpired caught below
                    )
                    out_s = (p.stdout or b"").decode("utf-8", errors="replace")
                    if p.returncode == 0:
                        autostash_used = True
                        execution_steps.append({"step": "autostash_push", "status": "ok"})
                    else:
                        execution_steps.append({"step": "autostash_push", "status": "failed", "stdout_clip": out_s[:800]})
                except Exception as e:
                    execution_steps.append({"step": "autostash_push", "status": "exception", "error": str(e)})

            ok, msg, r, d = _try_3way_fallback(repo, cleaned, touched_files, snapshot=rollback_snapshot)

            if autostash_used:
                try:
                    p2 = subprocess.run(
                        ["git", "stash", "pop"],
                        cwd=str(repo),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=30,  # bound HTTP request; TimeoutExpired caught below
                    )
                    out_p = (p2.stdout or b"").decode("utf-8", errors="replace")
                    if p2.returncode == 0:
                        autostash_pop_ok = True
                        execution_steps.append({"step": "autostash_pop", "status": "ok"})
                    else:
                        autostash_pop_error = out_p[:800]
                        execution_steps.append({"step": "autostash_pop", "status": "failed", "stdout_clip": autostash_pop_error})
                except Exception as e:
                    autostash_pop_error = str(e)
                    execution_steps.append({"step": "autostash_pop", "status": "exception", "error": autostash_pop_error})

            try:
                if isinstance(d, dict):
                    d["autostash_used"] = bool(autostash_used)
                    d["autostash_pop_ok"] = bool(autostash_pop_ok)
                    d["autostash_pop_error"] = str(autostash_pop_error or "")
                    d["git_status_porcelain_before"] = dirty_before
                    d["git_status_porcelain_after"] = _git_status_porcelain(repo)[:2000]
                    d["used_strategy"] = "git-apply-3way-autostash" if autostash_used else (d.get("used_strategy") or "")
            except Exception:
                pass

            if ok:
                return ok, msg, r, d

            # 3way failed too; DO NOT fall through to generic failure path,
            # because that path overwrites details and drops autostash metadata.
            try:
                if isinstance(d, dict):
                    # Ensure execution_steps is present for UI/debug consistency
                    if not d.get("execution_steps"):
                        d["execution_steps"] = execution_steps
            except Exception:
                pass

            return False, msg, r, d

        # Fix 2 (user-WIP protection): same principle as the skip_3way branch above —
        # `git apply --check` is a non-mutating dry-run, so the worktree (and any
        # user-WIP) is unchanged here. _rollback() would `git restore --worktree`
        # and silently revert user-WIP to HEAD. Skip it; only sweep stale .rej.
        _cleanup_reject_files(repo, touched_files)
        return False, out.strip() or "git apply --check failed", reason, {
            "touched_files": touched_files,
            "failed_files": sorted(set(touched_files) | set(_extract_files_from_git_apply_output(out))),
            "rollback_performed": False,
            "rollback_skipped_reason": "check_is_dryrun_worktree_unchanged",
            "used_strategy": "git-apply-check",
            "execution_steps": execution_steps,
        }

    # 2) Apply
    step_apply = {"step": "git_apply", "status": "pending"}
    t0 = time.monotonic()
    # Use --recount --whitespace=nowarn to match pre-check flags (P3).
    # Without them, a patch that passed pre-check with recount could fail
    # here because the actual apply uses raw line numbers.
    rc2, out2 = _run_git_apply(
        repo, ["--recount", "--whitespace=nowarn"], input_text=cleaned,
    )
    step_apply["duration_ms"] = int((time.monotonic() - t0) * 1000)
    if rc2 == 0:
        step_apply["status"] = "ok"
    else:
        step_apply["status"] = "failed"
        step_apply["error"] = {
            "returncode": int(rc2),
            "stdout_clip": (out2 or "")[:2000],
        }
    execution_steps.append(step_apply)
    if rc2 != 0:
        reason = _classify_git_apply_output(out2)
        # Why no autostash here (unlike the --check-failure branch, Fix 2 ~L815)?
        #  (a) It would be a NON-FIX: autostash would pop user-WIP back AFTER the 3-way
        #      attempt, only for the `_rollback()` at L922 to `git restore --staged
        #      --worktree` it right back to HEAD. Protecting WIP on this path would
        #      require SKIPPING L922's rollback (mirroring Fix 2), not adding autostash.
        #  (b) Reachability is TOCTOU-only: `--check` (L769) and `git apply` (L902) use
        #      identical flags (`--recount --whitespace=nowarn`) and equivalent logic on a
        #      static tree, so check-pass + apply-fail implies a concurrent edit in the
        #      ms-window between the two subprocess calls. Rare in practice (intra-process
        #      writes are serialized by the write lock; only external/IPC-subagent edits
        #      can race it).
        #  (c) Plain `git apply` (no --3way) is transactional on failure — it leaves the
        #      worktree UNMUTATED (verified: multi-file partial patches apply nothing) —
        #      so L922's rollback is effectively a no-op that guarantees a clean known
        #      state (asserted by test_rollback_on_failure: "file == original"). Skipping
        #      it would trade a rare TOCTOU WIP-loss for a rare corruption risk; not
        #      clearly better, hence the deliberate asymmetry. Do NOT "symmetrize" by
        #      adding autostash — it is defeated by L922.
        if reason == REASON_CONFLICT and not skip_3way:
            ok, msg, r, d = _try_3way_fallback(repo, cleaned, touched_files, snapshot=rollback_snapshot)
            if ok:
                return ok, msg, r, d

        rollback_report = _rollback(repo, touched_files, snapshot=rollback_snapshot)
        _cleanup_reject_files(repo, touched_files)
        msg = (out2.strip() or "git apply failed") + _rollback_warning(rollback_report)
        return False, msg, reason, {
            "touched_files": touched_files,
            "failed_files": sorted(set(touched_files) | set(_extract_files_from_git_apply_output(out2))),
            "rollback_performed": rollback_report["attempted"],
            "rollback_verified": rollback_report["verified"],
            "rollback_remaining_dirty": rollback_report["remaining_dirty"],
            "used_strategy": "git-apply",
            "execution_steps": execution_steps,
        }

    # 3) Post-apply guard: python syntax check (py_compile) for touched .py files
    py_files = [p for p in (touched_files or []) if str(p).endswith(".py")]
    if py_files:
        try:
            cmd = [sys.executable, "-m", "py_compile", *py_files]
            t0_pyc = time.monotonic()
            p = subprocess.run(
                cmd,
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=60,  # bound HTTP request; TimeoutExpired caught below
            )
            out3 = (p.stdout or b"").decode("utf-8", errors="replace")
            pyc_duration_ms = int((time.monotonic() - t0_pyc) * 1000)

            if p.returncode != 0:
                excerpt = (out3 or "").strip()
                if len(excerpt) > 500:
                    excerpt = excerpt[:500] + "..."

                # -------------------------------------------------
                # Indent-heal (best-effort):
                # If patch broke Python indentation, try to repair locally
                # and re-run py_compile once.
                # -------------------------------------------------
                #
                # SECURITY NOTE: indent-heal can silently pull legitimate top-level
                # statements (e.g. `MODULE_VAR = compute()`, bare imports) into a
                # preceding def body when the heuristic mis-judges what counts as a
                # "top-level start" (only def/class/@/if __name__/async are recognized).
                # If the healed output happens to compile, the corruption is committed
                # silently. Therefore indent-heal is OPT-IN: it only runs when the
                # environment variable ASICODE_INDENT_HEAL=1 is set explicitly.
                # Otherwise we skip straight to the rollback path (no silent data loss).
                indent_heal_enabled = os.environ.get("ASICODE_INDENT_HEAL") == "1"
                healed = False
                try:
                    if indent_heal_enabled and (
                        ("IndentationError" in (out3 or "")) or ("TabError" in (out3 or ""))
                    ):

                        def _indent_len(s: str) -> int:
                            """Return indentation width (tab=8)."""
                            raw = s.rstrip(" \t")
                            return len(raw) - len(raw.lstrip()) if raw else 0

                        def _looks_like_toplevel_start(s: str) -> bool:
                            t = (s or "").lstrip()
                            return t.startswith(("def ", "class ", "@", "if __name__", "async "))

                        for rel in py_files:
                            fp = repo / rel
                            if not fp.exists() or (not fp.is_file()):
                                continue

                            file_lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
                            if not file_lines:
                                continue

                            # Find a likely broken def/class: next non-empty line not indented deeper
                            candidate_def_idx = -1
                            for k in range(len(file_lines)):
                                t = file_lines[k].lstrip()
                                # Cover def/class/async def (but not async for/async with)
                                if not (t.startswith(("def ", "class ")) or t.startswith("async def ")):
                                    continue
                                j = k + 1
                                while j < len(file_lines) and file_lines[j].strip() == "":
                                    j += 1
                                if j < len(file_lines) and _indent_len(file_lines[j]) <= _indent_len(file_lines[k]):
                                    candidate_def_idx = k
                                    break

                            if candidate_def_idx < 0:
                                continue

                            def_idx = candidate_def_idx
                            def_indent = _indent_len(file_lines[def_idx])
                            body_indent = def_indent + 4

                            # first statement line (skip blanks)
                            j = def_idx + 1
                            while j < len(file_lines) and file_lines[j].strip() == "":
                                j += 1
                            if j >= len(file_lines):
                                continue

                            # If first statement is not indented, indent a contiguous run until next top-level start.
                            if _indent_len(file_lines[j]) <= def_indent:
                                k = j
                                while k < len(file_lines):
                                    if file_lines[k].strip() == "":
                                        k += 1
                                        continue
                                    if _indent_len(file_lines[k]) <= def_indent and _looks_like_toplevel_start(file_lines[k]):
                                        break
                                    if _indent_len(file_lines[k]) <= def_indent:
                                        file_lines[k] = (" " * body_indent) + file_lines[k].lstrip(" ")
                                        healed = True
                                    k += 1

                                if healed:
                                    fp.write_text("\n".join(file_lines) + "\n", encoding="utf-8", errors="replace")

                    if healed:
                        cmd2 = [sys.executable, "-m", "py_compile", *py_files]
                        p2 = subprocess.run(
                            cmd2,
                            cwd=str(repo),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                            timeout=60,  # bound HTTP request; TimeoutExpired caught below
                        )
                        out_heal = (p2.stdout or b"").decode("utf-8", errors="replace")
                        if p2.returncode == 0:
                            return True, "applied", REASON_OK, {
                                "touched_files": touched_files,
                                "failed_files": [],
                                "rollback_performed": False,
                                "used_strategy": "git-apply+pycompile-guard+indent-heal",
                                "py_files": py_files,
                                "pycompile_returncode": int(p2.returncode),
                                "indent_heal_performed": True,
                            }

                        # Still failing: use latest output for excerpt below
                        out3 = out_heal
                        excerpt = (out3 or "").strip()
                        if len(excerpt) > 500:
                            excerpt = excerpt[:500] + "..."

                except Exception:
                    healed = False  # best-effort only

                rollback_report = _rollback(repo, touched_files, snapshot=rollback_snapshot)
                _cleanup_reject_files(repo, touched_files)
                msg_heal = "py_compile failed"
                if healed:
                    msg_heal = "py_compile failed (indent healed, still broken)"
                msg_heal += _rollback_warning(rollback_report)

                # --- structured pycompile error (BUNDLE v3) ---
                raw = out3 or ""
                err = {
                    "type": None,
                    "file": None,
                    "line": None,
                    "column": None,
                    "message": None,
                    "stderr_clip": raw[:2000],
                }

                m = re.search(r'File\s+"([^"]+)",\s+line\s+(\d+)', raw)
                if m:
                    err["file"] = m.group(1)
                    try:
                        err["line"] = int(m.group(2))
                    except Exception:
                        pass

                for ln in raw.splitlines():
                    if "^" in ln:
                        try:
                            err["column"] = ln.index("^") + 1
                        except Exception:
                            pass
                        break

                m2 = re.search(r'(\w+(?:Error|Exception)):\s*(.+?)(?:\n|$)', raw)
                if m2:
                    err["type"] = m2.group(1)
                    err["message"] = m2.group(2).strip()

                if not err.get("message"):
                    err["message"] = (excerpt or "py_compile failed").strip()

                # code excerpt (best-effort)
                try:
                    if err.get("file") and err.get("line"):
                        fp = (repo / err["file"]).resolve()
                        if fp.exists() and fp.is_file():
                            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
                            L = int(err["line"])
                            start = max(0, L - 4)
                            end = min(len(lines), L + 3)
                            out_lines = []
                            for i in range(start, end):
                                marker = ">>>" if (i + 1) == L else "   "
                                out_lines.append(f"{marker} {i+1:4d} | {lines[i]}")
                            err["excerpt"] = "\n".join(out_lines)
                except Exception:
                    pass

                execution_steps.append({
                    "step": "pycompile",
                    "status": "failed",
                    "duration_ms": pyc_duration_ms,
                    "checked_files": py_files,
                    "error": err,
                })

                return False, msg_heal, "PYCOMPILE_FAILED", {
                    "touched_files": touched_files,
                    "failed_files": py_files,
                    "rollback_performed": rollback_report["attempted"],
                    "rollback_verified": rollback_report["verified"],
                    "rollback_remaining_dirty": rollback_report["remaining_dirty"],
                    "used_strategy": "git-apply+pycompile-guard",
                    "py_files": py_files,
                    "pycompile_returncode": int(p.returncode),
                    "pycompile_output_excerpt": excerpt,
                    "indent_heal_performed": bool(healed),
                    "pycompile_error": err,
                    "execution_steps": execution_steps,
                }

            execution_steps.append({
                "step": "pycompile",
                "status": "ok",
                "duration_ms": pyc_duration_ms,
                "checked_files": py_files,
            })
            return True, "applied", REASON_OK, {
                "touched_files": touched_files,
                "failed_files": [],
                "rollback_performed": False,
                "used_strategy": "git-apply+pycompile-guard",
                "py_files": py_files,
                "pycompile_returncode": int(p.returncode),
                "execution_steps": execution_steps,
            }

        except Exception as e:
            # Even if the guard itself crashes, we MUST rollback to preserve atomicity.
            rollback_report = _rollback(repo, touched_files, snapshot=rollback_snapshot)
            _cleanup_reject_files(repo, touched_files)
            execution_steps.append({
                "step": "pycompile",
                "status": "failed",
                "duration_ms": 0,
                "checked_files": py_files,
                "error": {"type": type(e).__name__, "message": str(e)[:240]},
            })
            return False, "py_compile failed" + _rollback_warning(rollback_report), "PYCOMPILE_FAILED", {
                "touched_files": touched_files,
                "failed_files": py_files,
                "rollback_performed": rollback_report["attempted"],
                "rollback_verified": rollback_report["verified"],
                "rollback_remaining_dirty": rollback_report["remaining_dirty"],
                "used_strategy": "git-apply+pycompile-guard",
                "py_files": py_files,
                "pycompile_returncode": -1,
                "pycompile_output_excerpt": f"exception:{type(e).__name__}:{str(e)[:240]}",
                "execution_steps": execution_steps,
            }

    return True, "applied", REASON_OK, {
        "touched_files": touched_files,
        "failed_files": [],
        "rollback_performed": False,
        "used_strategy": "git-apply",
        "execution_steps": execution_steps,
    }


# =========================================================
# Salvage: non-diff LLM outputs -> synthesize minimal diff
# =========================================================

def salvage_unified_diff_from_llm_output(
    llm_text: str,
    repo_root: str | None,
    file_path_hint: str | None,
    insert_line_hint: str | None,
) -> str:
    """
    If LLM output is not a diff, try to salvage a single-line insert
    by delegating to patch_synth.synthesize_append_line_unified_diff.

    Hardening:
    - Prefer candidates found inside fenced Markdown code blocks only (reduces false positives).
    - Avoid treating real code lines like '+++variable' as diff headers (header is '+++ b/<path>' with a space).
    """
    if not llm_text or not file_path_hint:
        return ""

    text = str(llm_text).strip()

    # Already a real diff
    if "diff --git " in text or text.lstrip().startswith("@@"):
        return ""

    def _extract_fenced_code_blocks(t: str) -> list[str]:
        blocks: list[str] = []
        in_block = False
        buf: list[str] = []
        for ln in (t or "").splitlines():
            s = ln
            if s.strip().startswith("```"):
                if in_block:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                    in_block = False
                else:
                    in_block = True
                    buf = []
                continue
            if in_block:
                buf.append(s)
        if in_block and buf:
            blocks.append("\n".join(buf).strip())
        return [b for b in blocks if b]

    code_blocks = _extract_fenced_code_blocks(text)
    body = "\n\n".join(code_blocks).strip() if code_blocks else "\n".join(
        [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    ).strip()

    line = (insert_line_hint or "").strip().strip('"').strip("'")
    if not line:
        cands: list[str] = []
        for ln in body.splitlines():
            s = ln.strip()
            if not s.startswith("+"):
                continue
            # unified diff header is '+++ b/<path>' (space required)
            if s.startswith("+++ "):
                continue
            cand = s[1:].strip()
            if cand:
                cands.append(cand)

        # Stay conservative: only salvage if we have exactly one unambiguous candidate
        if len(cands) == 1:
            line = cands[0]

    if not line:
        return ""

    rel = normalize_rel_path_fast(file_path_hint or "")
    return synthesize_append_line_unified_diff(repo_root or ".", rel, line) if rel else ""
# autostash test
# dirty-for-autostash
# DIRTY_FOR_AUTOSTASH
# DIRTY_FOR_AUTOSTASH
# DIRTY_FOR_AUTOSTASH
