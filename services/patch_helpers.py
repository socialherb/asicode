# services/patch_helpers.py
from __future__ import annotations

import subprocess
from pathlib import Path


def normalize_patch_text(p: str) -> str:
    """
    Normalize patch text for git-apply-check.

    IMPORTANT:
    - Do NOT strip spaces/tabs from line ends.
      Unified diff hunks can legitimately contain "blank line" entries represented as:
        " "   (context blank line)
        "+"   (added blank line)
        "-"   (removed blank line)
      If we strip trailing whitespace from the patch text, those can turn into empty lines
      without a prefix and git will report: "corrupt patch".

    We only:
    - normalize newlines to LF
    - trim leading/trailing *newlines* (not spaces/tabs)
    - ensure exactly ONE trailing newline
    """
    if p is None:
        return ""
    s = str(p)
    s = s.replace('\r\n', '\n').replace('\r', '\n')

    # Trim ONLY newlines around the patch (keep spaces/tabs intact)
    s = s.strip("\n")
    if not s:
        return ""

    # Ensure EXACTLY one trailing newline
    s = s.rstrip("\n") + "\n"
    return s


def _is_effectively_empty_patch(patch_text: str) -> bool:
    return not (patch_text or "").strip()


def _map_git_apply_output_to_taxonomy(output: str) -> str:
    o = (output or "").strip().lower()

    if not o:
        return "UNKNOWN"

    if "no valid patches in input" in o:
        return "EMPTY_PATCH"

    if "corrupt patch" in o or "corrupt" in o:
        return "CORRUPT_PATCH"

    if "patch format" in o or "malformed patch" in o:
        return "MALFORMED_PATCH"

    if "patch failed" in o:
        return "PATCH_FAILED"

    if "does not apply" in o:
        return "DOES_NOT_APPLY"

    if "already exists in working directory" in o:
        return "ALREADY_EXISTS"

    if "does not exist in index" in o or "does not exist in working tree" in o:
        return "PATH_NOT_FOUND"

    if "unrecognized input" in o:
        return "UNRECOGNIZED_INPUT"

    if "not a git repository" in o:
        return "NOT_A_GIT_REPO"

    if "permission denied" in o:
        return "PERMISSION_DENIED"

    return "GIT_APPLY_CHECK_FAILED"


def _normalize_git_output(s: str) -> str:
    if s is None:
        return ""
    out = str(s)
    out = out.replace('\r\n', '\n').replace('\r', '\n')
    out = out.strip()
    if len(out) > 32000:
        out = out[:32000] + "\n...[truncated]..."
    return out


def git_apply_check_only(repo_root: str, patch_text: str) -> tuple[bool, str, str]:
    patch_norm = normalize_patch_text(patch_text)

    if _is_effectively_empty_patch(patch_norm):
        return False, "empty patch (blocked)", "EMPTY_PATCH"

    try:
        proc = subprocess.run(
            ["git", "apply", "--check"],
            cwd=str(Path(repo_root).resolve()),
            input=patch_norm.encode("utf-8", errors="ignore"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,  # Bound the HTTP request: a hung git (lock contention,
            # NFS, pathological repo) must not stall /edit/run indefinitely.
            # TimeoutExpired is a SubprocessError→Exception subclass, so the
            # bare ``except Exception`` below catches and maps it to CHECK_EXCEPTION.
        )

        out = _normalize_git_output((proc.stdout or b"").decode("utf-8", errors="replace"))

        if proc.returncode == 0:
            return True, (out or "ok"), "OK"

        raw_taxonomy = _map_git_apply_output_to_taxonomy(out)
        return False, (out or "git apply --check failed"), raw_taxonomy

    except Exception as e:
        msg = _normalize_git_output(f"check-exception: {type(e).__name__}: {e}")
        return False, msg, "CHECK_EXCEPTION"


def prepare_patch_for_apply(
    repo_root: str,
    diff_patch: str,
    file_hint: str | None = None,
) -> tuple[str, list[str]]:
    """
    Shared "apply prep" helper used by:
    - main.py /edit/apply endpoint
    - IntelligentLLMService Agent Mode (apply loop)

    Returns:
      (cleaned_patch, touched_files)

    Notes:
    - Keeps patch whitespace semantics intact via normalize_patch_text().
    - Uses the same diff cleaner as the server apply endpoint.
    """
    # Local import to avoid heavy imports for callers that only need git_apply_check_only.
    from diff_apply import _clean_diff, extract_touched_files_from_diff

    patch_raw = normalize_patch_text(diff_patch or "")
    patch_clean = _clean_diff(patch_raw, repo_root, file_path_hint=file_hint)
    touched = extract_touched_files_from_diff(patch_clean) if patch_clean.strip() else []
    return patch_clean, touched
