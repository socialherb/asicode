from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Optional

from common import normalize_rel_path_fast
from path_security import resolve_inside_repo
from external_llm.common.indent_utils import detect_indent_char, min_indent, shift_block

logger = logging.getLogger(__name__)


# ============================================================
# Prompt artifact sanitation (defense-in-depth)
# ============================================================
# Accept optional ":" to tolerate variants like "ASICODE_SELECTED_INDICES:".
_PROMPT_TAIL_MARKER_RE = re.compile(r"(?im)^\s*ASICODE_SELECTED_[A-Z0-9_]+\s*:?\s*$")


def _strip_prompt_tail_markers(block: str) -> str:
    """Strip engine-owned prompt metadata accidentally included in a block.

    If UI routing metadata is placed between AFTER and END (e.g. ASICODE_SELECTED_INDICES),
    and the parser fails to exclude it, the unified diff would insert those lines into the file.
    This function removes that tail starting at the first marker line.
    """
    if not block:
        return block
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if _PROMPT_TAIL_MARKER_RE.match(ln):
            return "\n".join(lines[:i]).rstrip("\n")
    return block


# ============================================================
# File IO
# ============================================================
def _read_text_lines(repo_root: str, rel_path: str) -> Optional[list[str]]:
    """
    Read file lines safely, returning the line list only.

    Thin wrapper over ``_read_text_lines_with_eof`` (which also reports whether
    the file ends with ``\\n``). Callers that need the EOF flag should call
    ``_read_text_lines_with_eof`` directly to avoid a double read.

    Returns:
        List[str]  -> success (including legitimately empty file)
        None       -> read failure / path violation
    """
    info = _read_text_lines_with_eof(repo_root, rel_path)
    return info[0] if info is not None else None


def _split_diff_lines(raw: str) -> tuple[list[str], bool]:
    """Split text into lines the way ``git apply`` counts them: LF only.

    ``str.splitlines()`` ALSO splits on U+2028 (LINE SEPARATOR), U+0085 (NEL),
    U+000C (FF), U+000B (VT). Those extra splits disagree with git's own line
    counting, misaligning the ``@@ -N,M`` hunk offsets and making the generated
    patch fail to apply. Splitting on ``\\n`` matches git.

    CRLF / bare CR are normalized to LF so the trailing ``\\r`` does not linger
    in line text (which would otherwise break exact anchor matching).

    Returns ``(lines, ends_with_newline)``. A file ending in ``\\n`` does NOT
    produce a phantom trailing empty line.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if raw == "":
        return [], False
    ends_with_newline = raw.endswith("\n")
    parts = raw.split("\n")
    if ends_with_newline:
        parts = parts[:-1]  # drop the phantom '' after the final '\n'
    return parts, ends_with_newline


def _read_text_lines_with_eof(
    repo_root: str, rel_path: str
) -> Optional[tuple[list[str], bool]]:
    """
    Read file lines safely, also reporting whether the file ends with ``\\n``.

    The EOF-newline flag is required by every unified-diff synth path: when a
    file lacks a trailing newline, the diff must carry a ``\\ No newline at end
    of file`` marker (or re-state the last line) or ``git apply`` rejects it.

    Returns:
        (List[str], bool)  -> success (lines, ends_with_newline); empty file -> ([], False)
        None               -> read failure / path violation
    """
    rel = normalize_rel_path_fast(rel_path)
    if not rel:
        return None

    # Path safety: keep reads strictly inside repo_root
    try:
        fp = resolve_inside_repo(repo_root, rel)
    except ValueError:
        return None

    try:
        try:
            # Strict UTF-8 first (avoid silent data loss).
            raw = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fall back to ignoring non-UTF8 bytes (backward compat: all callers
            # are matching/diff-creation paths; U+FFFD from errors="replace" would
            # break exact-block anchor matching since _norm_line_for_match does
            # not strip U+FFFD). The warning below flags the encoding issue.
            raw = fp.read_bytes().decode("utf-8", errors="ignore")
            logger.warning(
                "non_utf8_bytes_ignored: rel_path=%r abs_path=%s",
                rel_path,
                str(fp),
            )
    except (FileNotFoundError, IsADirectoryError):
        return None
    except OSError:
        logger.warning(
            "read_text failed: rel_path=%r abs_path=%s",
            rel_path,
            str(fp),
            exc_info=True,
        )
        return None

    return _split_diff_lines(raw)


# ============================================================
# Matching helpers (accuracy improvements)
# ============================================================



def _norm_line_for_match(s: str) -> str:
    """
    Normalization for robust anchor matching:
    - strip leading/trailing spaces
    - collapse all whitespace runs to single space
    """
    if s is None:
        return ""
    s = s.replace("\r", "")
    s = ' '.join(s.split())
    return s


def _find_line_matches(lines: list[str], rx: re.Pattern, *, normalized: bool = False) -> list[int]:
    """
    Return indices where regex matches a single line.
    If normalized=True, run regex on normalized line text.
    """
    out: list[int] = []
    for i, ln in enumerate(lines):
        s = _norm_line_for_match(ln) if normalized else ln
        if rx.search(s):
            out.append(i)
    return out


def _find_multiline_matches(
    lines: list[str],
    rx: re.Pattern,
    *,
    window_max_lines: int = 12,
) -> list[int]:
    """
    Multi-line anchor matching:
    - Slide a window of up to window_max_lines lines
    - Join with '\n' and search with DOTALL-enabled regex
    - Return start indices (first line of match window)
    """
    if not lines:
        return []
    n = len(lines)
    out: list[int] = []

    # cap to reasonable
    wmax = max(2, int(window_max_lines))
    for i in range(n):
        # grow window progressively (2..wmax)
        acc = []
        for k in range(i, min(n, i + wmax)):
            acc.append(lines[k])
            blob = "\n".join(acc)
            if rx.search(blob):
                out.append(i)
                break
    return out


def _pick_unique_or_fail(matches: list[int], require_unique: bool) -> Optional[int]:
    if not matches:
        return None
    if require_unique and len(matches) != 1:
        return None
    return matches[0]


def _compile_anchor_regex(pattern: str, *, dotall: bool = False) -> Optional[re.Pattern]:
    """Compile regex with MULTILINE; add DOTALL when dotall=True (for multi-line windows)."""
    if not pattern:
        return None
    try:
        flags = re.MULTILINE | re.DOTALL if dotall else re.MULTILINE
        return re.compile(pattern, flags=flags)
    except Exception:
        logger.warning("anchor_regex_compile_failed: pattern=%r", pattern, exc_info=True)
        return None


def _resolve_anchor_index(
    lines: list[str],
    pattern: str,
    *,
    require_unique: bool = True,
    multiline_window_max_lines: int = 12,
) -> Optional[int]:
    """
    Resolve anchor match index with fallbacks:
      1) direct line regex on raw lines
      2) line regex on normalized lines (whitespace-insensitive)
      3) multi-line window regex (DOTALL) to catch split signatures
    """
    rx = _compile_anchor_regex(pattern)
    if rx is None:
        return None

    # (1) raw line match
    m1 = _find_line_matches(lines, rx, normalized=False)
    idx = _pick_unique_or_fail(m1, require_unique)
    if idx is not None:
        return idx

    # (2) normalized line match (helps with whitespace drift)
    m2 = _find_line_matches(lines, rx, normalized=True)
    idx = _pick_unique_or_fail(m2, require_unique)
    if idx is not None:
        return idx

    # (3) multi-line match
    rx2 = _compile_anchor_regex(pattern, dotall=True)
    if rx2 is None:
        return None
    m3 = _find_multiline_matches(lines, rx2, window_max_lines=multiline_window_max_lines)
    idx = _pick_unique_or_fail(m3, require_unique)
    return idx

# ============================================================
# EOF-no-newline helpers (shared across all synth/difflib paths)
#
# Every function that constructs a hunk for a file lacking a trailing "\n" must
# emit "\ No newline at end of file" (or re-state the last line), or `git apply`
# rejects the patch with "patch failed". See design insight:
# "[architecture] 2026-07-08 12:42 — EOF-no-newline marker is systemic".
# ============================================================
_NO_NL_MARKER = "\\ No newline at end of file"


def _emit_hunk_replace_body(
    body_lines: list[str],
    *,
    old_eof_no_newline: bool,
    new_eof_no_newline: bool,
) -> list[str]:
    """Add EOF-no-newline markers to an already-prefixed replace/delete hunk body.

    ``body_lines`` are the hunk body lines AFTER the ``@@`` header, each already
    prefixed with ``' '`` (context), ``'-'`` (old), or ``'+'`` (new) — i.e. the
    exact lines the synth caller would have emitted before this fix. This helper
    returns a NEW list with the ``\\ No newline at end of file`` markers inserted
    (and, where git requires it, re-states the trailing context line) so the
    patch applies to files lacking a trailing newline.

    Marker rules (all verified against ``git diff`` / ``git apply``):

    * The file's final line can only be represented by the LAST ``' '``/``'-'``
      line (old EOF) and the LAST ``' '``/``'+'`` line (new EOF).
    * Case A — both EOF resolve to the SAME trailing context line, both
      no-newline: a SINGLE marker after it covers both sides.
    * Case B — old EOF is a ``'-'`` line (removed) and new EOF is a trailing
      context line whose terminator CHANGED (it used to have a newline, now it
      is the last line without one): the context line must be RE-STATED as
      ``-line`` (old, had newline → no marker) followed by ``+line`` + marker
      (new, no newline). This is the delete-last / delete-block-at-EOF case.
    * Case C — otherwise: old-side marker follows the last old-side line,
      new-side marker follows the last new-side line.
    """
    if not body_lines:
        return list(body_lines)
    if not old_eof_no_newline and not new_eof_no_newline:
        return list(body_lines)

    # Last old-side line (' ' or '-') and last new-side line (' ' or '+').
    old_idx = None
    new_idx = None
    for i in range(len(body_lines) - 1, -1, -1):
        c = body_lines[i][:1]
        if old_idx is None and c in (" ", "-"):
            old_idx = i
        if new_idx is None and c in (" ", "+"):
            new_idx = i
        if old_idx is not None and new_idx is not None:
            break

    if old_idx is None and new_idx is None:
        return list(body_lines)

    same_line = (
        old_idx is not None
        and new_idx is not None
        and old_idx == new_idx
        and body_lines[old_idx][:1] == " "
    )

    out: list[str] = []

    # Case A: shared trailing context line, both no-newline -> one marker.
    if same_line and old_eof_no_newline and new_eof_no_newline:
        for i, l in enumerate(body_lines):
            out.append(l)
            if i == old_idx:
                out.append(_NO_NL_MARKER)
        return out

    # Case B: delete-at-EOF. Old EOF is a '-' line and new EOF is a context line
    # whose terminator changed (newline -> no-newline). Re-state that context
    # line: emit '-line' (old, had newline) then '+line' + marker (new).
    if (
        old_idx is not None
        and new_idx is not None
        and body_lines[old_idx][:1] == "-"
        and body_lines[new_idx][:1] == " "
        and new_eof_no_newline
    ):
        for i, l in enumerate(body_lines):
            if i == new_idx:
                content = l[1:]
                out.append("-" + content)            # old: had newline (no marker)
                out.append("+" + content)            # new: no newline
                out.append(_NO_NL_MARKER)
            else:
                out.append(l)
        # old-side marker for the removed EOF line, if it was no-newline
        if old_eof_no_newline:
            _insert_marker_after_last_minus(out)
        return out

    # Case C: general — place markers after the last old-side and last new-side.
    need = set()
    if old_idx is not None and old_eof_no_newline:
        need.add(old_idx)
    if new_idx is not None and new_eof_no_newline:
        need.add(new_idx)
    if not need:
        return list(body_lines)
    for i, l in enumerate(body_lines):
        out.append(l)
        if i in need:
            out.append(_NO_NL_MARKER)
    return out


def _emit_insert_at_eof_restate(
    out: list[str], body: list[str], old_last: str, line: str
) -> None:
    """Insert AT EOF on a no-trailing-newline file: re-state the old last line.

    Used by the insert synth paths (core helper + the line-no wrapper) when the
    insertion lands after the file's final line AND the file lacks a trailing
    newline. The old last line GAINS a newline in the result, so git requires it
    to be re-stated as ``-old_last`` + ``\\ No newline at end of file`` (old: no
    newline) then ``+old_last`` (new: has newline), followed by the ``+line``
    insert. At the call site ``body`` is laid out as
    ``[...prefix ctx, " old_last", "+line"]``; this re-emits everything except
    those last two entries, then the restate, then re-appends ``+line``.

    Centralized so the insert-at-EOF restate logic (and the marker constant)
    lives in exactly one place — previously duplicated verbatim across
    ``_synthesize_insert_with_context_unified_diff`` and the line-no wrapper in
    ``llm_execution`` (which hard-coded the marker literal, inviting drift).
    """
    out.extend(body[:-2])
    out.append("-" + old_last)
    out.append(_NO_NL_MARKER)
    out.append("+" + old_last)
    out.append("+" + line)


def _insert_marker_after_last_minus(out: list[str]) -> None:
    """Insert the EOF marker immediately after the last '-' line, if absent."""
    for i in range(len(out) - 1, -1, -1):
        if out[i].startswith("-"):
            if i + 1 < len(out) and out[i + 1] == _NO_NL_MARKER:
                return
            out.insert(i + 1, _NO_NL_MARKER)
            return


def _difflib_apply_eof_markers(
    body: list[str], *, old_eof_no_newline: bool, new_eof_no_newline: bool
) -> list[str]:
    """Post-process a difflib unified-diff body to add EOF-no-newline markers.

    ``difflib.unified_diff`` NEVER emits ``\\ No newline at end of file``, so for
    files lacking a trailing newline git rejects the patch. The EOF line can only
    appear in the LAST hunk (hunks are ascending), so we only inspect that one.

    For the last hunk we find the representation of the file's final line on each
    side (``-``/`` `` for old, ``+``/`` `` for new) and insert a marker after it.
    A shared context final line gets a single marker.
    """
    if not old_eof_no_newline and not new_eof_no_newline:
        return body

    # Locate the last hunk header.
    hunk_starts = [i for i, l in enumerate(body) if l.startswith("@@")]
    if not hunk_starts:
        return body
    hs = hunk_starts[-1]
    hunk = body[hs:]

    # Find the last old-side line (' ' or '-') and last new-side line (' ' or '+')
    # within the hunk body (skip the @@ header at index 0).
    old_idx = None
    new_idx = None
    for i in range(len(hunk) - 1, 0, -1):
        c = hunk[i][:1]
        if old_idx is None and c in (" ", "-"):
            old_idx = i
        if new_idx is None and c in (" ", "+"):
            new_idx = i
        if old_idx is not None and new_idx is not None:
            break

    need = set()
    if old_idx is not None and old_eof_no_newline:
        need.add(old_idx)
    if new_idx is not None and new_eof_no_newline:
        need.add(new_idx)

    if not need:
        return body

    # Rebuild the hunk inserting markers. Because the same absolute hunk index
    # can be shared (shared context), the set dedups -> one marker.
    new_hunk: list[str] = []
    for i, l in enumerate(hunk):
        new_hunk.append(l)
        if i in need:
            new_hunk.append(_NO_NL_MARKER)
    return body[:hs] + new_hunk

# ============================================================
# Diff synth
# ============================================================
def synthesize_append_line_unified_diff(repo_root: str, rel_path: str, line: str) -> str:
    """Append exactly one line to EOF (deterministic unified diff)."""
    rel = normalize_rel_path_fast(rel_path)
    line = (line or "").rstrip("\r\n")
    if not rel or line == "":
        return ""

    # Resolve the canonical path once (path-safety + reuse for the exists() check).
    try:
        fp = resolve_inside_repo(repo_root, rel)
    except ValueError:
        return ""

    # Read once with the EOF flag. The helper returns None for BOTH "absent file"
    # and "present but unreadable" (PermissionError/IOError). For append those two
    # must diverge: an absent file is a legitimate "new file" target, but an
    # unreadable present file must NOT be turned into a destructive "new file"
    # diff (it would overwrite on apply). So on None we check existence.
    info = _read_text_lines_with_eof(repo_root, rel)
    if info is None:
        if fp.exists():
            # present but unreadable -> refuse to synthesize
            return ""
        # legitimately absent -> fall through to the empty/new-file branch below
        existing_lines: list[str] = []
        ends_with_newline = True
    else:
        existing_lines, ends_with_newline = info

    out: list[str] = []
    out.append(f"diff --git a/{rel} b/{rel}")

    if not existing_lines:
        out.append("new file mode 100644")
        out.append("index 0000000..1111111 100644")
        out.append("--- /dev/null")
        out.append(f"+++ b/{rel}")
        out.append("@@ -0,0 +1,1 @@")
        out.append(f"+{line}")
        return "\n".join(out) + "\n"

    out.append(f"--- a/{rel}")
    out.append(f"+++ b/{rel}")

    n = len(existing_lines)
    last = existing_lines[-1].rstrip("\r")
    out.append(f"@@ -{n},1 +{n},2 @@")
    if ends_with_newline:
        # Normal file: last line is plain context, +line appends at EOF.
        out.append(f" {last}")
        out.append(f"+{line}")
    else:
        # No-trailing-newline file: appending gives the old last line a newline,
        # so git requires it to be re-stated as -last (no-nl) / +last (has nl).
        out.append(f"-{last}")
        out.append(_NO_NL_MARKER)
        out.append(f"+{last}")
        out.append(f"+{line}")
    return "\n".join(out) + "\n"


def _synthesize_insert_with_context_unified_diff(
    repo_root: str,
    rel_path: str,
    *,
    insert_pos: int,          # 0..len(lines)
    insert_line: str,
    context_lines: int = 12,
    lines: Optional[list[str]] = None,  # pre-fetched lines (avoid double read)
    ends_with_newline: Optional[bool] = None,  # pre-fetched EOF flag
) -> str:
    """
    Core helper: create a robust unified diff hunk with +/- context_lines around insert_pos.
    - No deletions, only one insertion.
    - Hunk includes enough context so `git apply` can match reliably.

    ``ends_with_newline`` is the file's trailing-newline flag; when ``lines`` is
    None it is read from disk alongside the content. Required so the hunk can
    emit the ``\\ No newline at end of file`` marker when the context reaches a
    no-trailing-newline EOF.
    """
    rel = normalize_rel_path_fast(rel_path)
    line = (insert_line or "").rstrip("\r\n")
    if not rel or line == "":
        return ""

    if lines is None or ends_with_newline is None:
        info = _read_text_lines_with_eof(repo_root, rel)
        if info is None:
            return ""
        if lines is None:
            lines = info[0]
        if ends_with_newline is None:
            ends_with_newline = info[1]
    if not lines:
        return ""

    n = len(lines)
    pos = int(insert_pos)
    if pos < 0:
        pos = 0
    if pos > n:
        pos = n

    # Auto-indent (quality-of-life):
    # If insert_line has no leading whitespace, copy indentation from neighbor line
    # so inserted lines align with surrounding code in many common cases.
    if line and (not line.startswith((" ", "\t"))):
        ref_after = lines[pos] if pos < n else ""
        ref_before = lines[pos - 1] if pos > 0 else ""

        def _lead_ws(s: str) -> str:
            s = s or ""
            return s[:len(s) - len(s.lstrip("\t "))]

        ia = _lead_ws(ref_after)
        ib = _lead_ws(ref_before)

        # prefer indent that is consistent between neighbors when possible
        indent = ia if (ia and (ia == ib or not ib)) else ib

        if indent:
            line = indent + line

    ctx = max(1, int(context_lines))
    lo = max(0, pos - ctx)
    hi = min(n, pos + ctx)

    old_chunk = lines[lo:hi]
    # new chunk is old chunk with one extra line at (pos-lo)
    new_chunk = [*lines[lo:pos], line, *lines[pos:hi]]

    old_start_1 = lo + 1
    new_start_1 = lo + 1
    old_count = len(old_chunk)
    new_count = len(new_chunk)

    out: list[str] = []
    out.append(f"diff --git a/{rel} b/{rel}")
    out.append(f"--- a/{rel}")
    out.append(f"+++ b/{rel}")
    out.append(f"@@ -{old_start_1},{old_count} +{new_start_1},{new_count} @@")

    # emit: context up to pos, then +line, then remaining context
    body: list[str] = []
    for ln in lines[lo:pos]:
        body.append(" " + ln.rstrip("\r"))
    body.append("+" + line)
    for ln in lines[pos:hi]:
        body.append(" " + ln.rstrip("\r"))

    # EOF markers only matter when the hunk reaches the old file's final line.
    if hi == n and not ends_with_newline:
        if pos == n:
            # Insert AT EOF: the inserted +line becomes the new last line (with a
            # newline). The old last line (now second-to-last) GAINED a newline, so
            # it must be re-stated: emit `-last` + marker (old: no newline) then
            # `+last` (new: has newline). The inserted line needs no marker.
            # body layout at pos==n: [...prefix ctx including " old_last", "+line"]
            old_last = lines[n - 1].rstrip("\r")
            _emit_insert_at_eof_restate(out, body, old_last, line)
        else:
            # Insert BEFORE the EOF line: the trailing context (the old last line)
            # keeps its no-newline state on both sides -> a single marker after it.
            out.extend(_emit_hunk_replace_body(
                body,
                old_eof_no_newline=True,
                new_eof_no_newline=True,
            ))
    else:
        out.extend(body)

    return "\n".join(out) + "\n"


def synthesize_insert_line_before_first_match_unified_diff(
    repo_root: str,
    rel_path: str,
    insert_line: str,
    before_regex: str,
    require_unique: bool = True,
    *,
    multiline_window_max_lines: int = 12,
    context_lines: int = 12,
    lines: Optional[list[str]] = None,  # pre-fetched lines (avoid double read)
    ends_with_newline: Optional[bool] = None,  # pre-fetched EOF flag (avoid double read)
) -> str:
    """
    Insert exactly one line BEFORE the first line matching before_regex.

    Robustness:
    - robust anchor resolution: raw -> normalized -> multiline window
    - thick context hunk (±context_lines) to make `git apply` reliable

    ``ends_with_newline`` is threaded to the core helper so the EOF-no-newline
    marker is emitted without a re-read when ``lines`` is pre-fetched.
    """
    rel = normalize_rel_path_fast(rel_path)
    line = (insert_line or "").rstrip("\r\n")
    if not rel or line == "" or not before_regex:
        return ""

    if lines is None or ends_with_newline is None:
        info = _read_text_lines_with_eof(repo_root, rel)
        if info is None:
            return ""
        if lines is None:
            lines = info[0]
        if ends_with_newline is None:
            ends_with_newline = info[1]
    if not lines:
        return ""

    idx = _resolve_anchor_index(
        lines,
        before_regex,
        require_unique=require_unique,
        multiline_window_max_lines=multiline_window_max_lines,
    )
    if idx is None:
        return ""

    # insert before anchor line => insert_pos = idx
    return _synthesize_insert_with_context_unified_diff(
        repo_root,
        rel,
        insert_pos=idx,
        insert_line=line,
        lines=lines,
        ends_with_newline=ends_with_newline,
        context_lines=context_lines,
    )


def synthesize_insert_line_after_first_match_unified_diff(
    repo_root: str,
    rel_path: str,
    insert_line: str,
    after_regex: str,
    require_unique: bool = True,
    *,
    multiline_window_max_lines: int = 12,
    context_lines: int = 12,
    lines: Optional[list[str]] = None,  # pre-fetched lines (avoid double read)
    ends_with_newline: Optional[bool] = None,  # pre-fetched EOF flag (avoid double read)
) -> str:
    """
    Insert exactly one line AFTER the first line matching after_regex.

    Robustness:
    - robust anchor resolution: raw -> normalized -> multiline window
    - thick context hunk (±context_lines) to make `git apply` reliable

    ``ends_with_newline`` is threaded to the core helper so the EOF-no-newline
    marker is emitted without a re-read when ``lines`` is pre-fetched.
    """
    rel = normalize_rel_path_fast(rel_path)
    line = (insert_line or "").rstrip("\r\n")
    if not rel or line == "" or not after_regex:
        return ""

    if lines is None or ends_with_newline is None:
        info = _read_text_lines_with_eof(repo_root, rel)
        if info is None:
            return ""
        if lines is None:
            lines = info[0]
        if ends_with_newline is None:
            ends_with_newline = info[1]
    if not lines:
        return ""

    idx = _resolve_anchor_index(
        lines,
        after_regex,
        require_unique=require_unique,
        multiline_window_max_lines=multiline_window_max_lines,
    )
    if idx is None:
        return ""

    # insert after anchor line => insert_pos = idx+1
    return _synthesize_insert_with_context_unified_diff(
        repo_root,
        rel,
        insert_pos=idx + 1,
        insert_line=line,
        lines=lines,
        ends_with_newline=ends_with_newline,
        context_lines=context_lines,
    )


def synthesize_replace_first_exact_block_unified_diff(
    repo_root: str,
    rel_path: str,
    before_block: str,
    after_block: str,
    *,
    require_unique: bool = True,
    context_lines: int = 6,
    # NEW (optional): ambiguity resolver
    # - selected_start_line_1: 1-based start line number to force-pick among multiple hits
    # - selected_indices_0: pick Nth hit (0-based index into "hits" list)
    # - prompt_lines: if provided, we will also try to parse "start_line_1: N" / "start_line: N"
    selected_start_line_1: Optional[int] = None,
    selected_indices_0: Optional[list[int]] = None,
    prompt_lines: Optional[list[str]] = None,
    **_ignored: object,
) -> str:
    """
    Deterministically generate a unified diff that replaces the FIRST exact block match.

    Key improvement:
    - Include +/- `context_lines` around the replaced region to make `git apply` robust.
      (thin hunks like @@ -N,2 +N,2 @@ often fail to apply)

    Notes:
    - before_block / after_block are split on LF only via _split_block_lines
      (NOT str.splitlines(), which would over-split on U+2028/NEL/FF and disagree
      with git's line counting). Matching is exact on line sequences.
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel or before_block is None:
        return ""

    info = _read_text_lines_with_eof(repo_root, rel)
    if info is None:
        return ""
    lines, ends_with_newline = info

    before_block = _strip_prompt_tail_markers(before_block or "")
    after_block = _strip_prompt_tail_markers(after_block or "")
    before_lines = _split_block_lines(before_block)
    after_lines = _split_block_lines(after_block)

    if not before_lines:
        return ""


    def _extract_start_line_hint_1(plines: Optional[list[str]]) -> Optional[int]:
        if not plines:
            return None

        # 1) Preferred legacy hint: start_line_1: N (scan from tail)
        for s in reversed(plines):
            ln_stripped = str(s or "").strip()
            ln_lower = ln_stripped.lower()
            if ln_lower.startswith("start_line_1:") or ln_lower.startswith("start_line:"):
                num_part = ln_stripped.split(":", 1)[1].strip()
                try:
                    n = int(num_part)
                    return n if n > 0 else None
                except Exception:
                    return None

        # 2) UI marker hint: ASICODE_SELECTED_INDICES then parse first positive integer after it
        seen_marker = False
        for s in plines:
            ln = str(s or "").strip()
            if not ln:
                continue
            if not seen_marker:
                if ln.lower().startswith("asicode_selected_indices"):
                    rest = ln[26:].strip()
                    if rest == "" or rest == ":":
                        seen_marker = True
                continue
            # stop if another ASICODE_* marker appears
            if ln.upper().startswith("ASICODE_") and len(ln) > 9:
                after = ln[9:].strip(": ") if ln[9:] else ""
                if not after:
                    break
            num_str = ""
            for ch in ln:
                if ch.isdigit():
                    num_str += ch
                elif num_str:
                    break
            if num_str:
                try:
                    n = int(num_str)
                    return n if n > 0 else None
                except Exception:
                    continue

        return None

    # If caller didn't pass selected_start_line_1 explicitly, try to parse from prompt_lines
    if selected_start_line_1 is None:
        selected_start_line_1 = _extract_start_line_hint_1(prompt_lines)


    # --- find all exact matches of before_lines in lines ---
    hits = []
    n = len(lines)
    m = len(before_lines)
    if m == 0:
        return ""

    for i in range(0, n - m + 1):
        if lines[i : i + m] == before_lines:
            hits.append(i)

    # -------------------------------------------------
    # Fallback 1: tolerant whitespace-insensitive match
    # -------------------------------------------------
    if not hits:
        tolerant_hits = _find_all_block_occurrences_tolerant(lines, before_lines)
        if tolerant_hits:
            hits = tolerant_hits

    # -------------------------------------------------
    # Fallback 2: similarity-based fuzzy match
    # -------------------------------------------------
    best_score = 0.0
    best_idx = None

    if not hits:
        def _block_similarity(a: list[str], b: list[str]) -> float:
            sa = "\n".join(a)
            sb = "\n".join(b)
            return SequenceMatcher(None, sa, sb).ratio()

        for i in range(0, n - m + 1):
            candidate = lines[i : i + m]
            score = _block_similarity(candidate, before_lines)
            if score > best_score:
                best_score = score
                best_idx = i

        # >= 0.80 → auto accept
        if best_idx is not None and best_score >= 0.80:
            hits = [best_idx]

        # 0.60~0.80 → allow ambiguity resolution upstream
        elif best_idx is not None and best_score >= 0.60:
            # treat as ambiguous but still expose candidate
            hits = [best_idx]

    if not hits:
        return ""

    if require_unique and len(hits) != 1:
        # 1) strongest: explicit start_line_1 (1-based absolute line)
        if selected_start_line_1 is not None:
            try:
                forced0 = int(selected_start_line_1) - 1
            except Exception:
                forced0 = None
            if forced0 is not None and forced0 in hits:
                hits = [forced0]
            else:
                return ""

        # 2) next: selected_indices_0 can be either:
        #    - absolute start_line_1 values (1-based, same semantics as match_lines_1), OR
        #    - hit positions (0-based index into the "hits" list)
        elif selected_indices_0:
            try:
                raw0 = int(selected_indices_0[0])
            except Exception:
                return ""

            # Prefer interpreting as absolute start_line_1 (1-based): raw0-1 must be in hits
            forced0 = raw0 - 1
            if forced0 in hits:
                hits = [forced0]
            else:
                # Fallback: treat as 0-based hit index into hits[]
                if 0 <= raw0 < len(hits):
                    hits = [hits[raw0]]
                else:
                    return ""

        else:
            return ""


    start_idx = hits[0]
    end_idx = start_idx + m  # exclusive

    # --- context window ---
    ctx = max(0, int(context_lines))
    lo = max(0, start_idx - ctx)
    hi = min(n, end_idx + ctx)

    old_chunk = lines[lo:hi]
    new_chunk = lines[lo:start_idx] + after_lines + lines[end_idx:hi]

    old_count = len(old_chunk)
    new_count = len(new_chunk)

    # 1-based line numbers in unified diff header
    old_start_1 = lo + 1
    new_start_1 = lo + 1

    out = []
    out.append(f"diff --git a/{rel} b/{rel}")
    out.append(f"--- a/{rel}")
    out.append(f"+++ b/{rel}")
    out.append(f"@@ -{old_start_1},{old_count} +{new_start_1},{new_count} @@")

    body: list[str] = []
    for ln in lines[lo:start_idx]:
        body.append(" " + ln)
    for ln in lines[start_idx:end_idx]:
        body.append("-" + ln)
    for ln in after_lines:
        body.append("+" + ln)
    for ln in lines[end_idx:hi]:
        body.append(" " + ln)

    # EOF markers when the hunk reaches the file's final line (hi == n).
    # See synthesize_replace_line_range_unified_diff: an in-place block replace
    # preserves the file's trailing-newline state on both sides; the marker only
    # differs in the delete-at-EOF case, which _emit_hunk_replace_body handles.
    if hi == n:
        old_eof_no_newline = not ends_with_newline
        out.extend(_emit_hunk_replace_body(
            body,
            old_eof_no_newline=old_eof_no_newline,
            new_eof_no_newline=old_eof_no_newline,
        ))
    else:
        out.extend(body)

    return "\n".join(out) + "\n"


def synthesize_replace_selected_matching_blocks_unified_diff(
    repo_root: str,
    rel_path: str,
    before_block: str,
    after_block: str,
    *,
    selected_start_lines_1: list[int],
    max_replacements: int = 50,
) -> str:
    """
    Replace ONLY selected matching blocks (tolerant whitespace) by rewriting the file and emitting a unified diff.

    selected_start_lines_1:
      - list of 1-based start line numbers (same space as meta.match_lines_1).
      - must be a subset of the tolerant match hits; otherwise returns "".

    Includes the same per-occurrence indent-heal behavior as replace-all.
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel or before_block is None:
        return ""
    if not selected_start_lines_1:
        return ""

    lines = _read_text_lines(repo_root, rel)
    if lines is None:
        return ""

    before_lines = _split_block_lines(before_block)
    after_lines_base = _split_block_lines(after_block)

    if not before_lines:
        return ""

    hits = _find_all_block_occurrences_tolerant(lines, before_lines)
    if not hits:
        return ""
    if len(hits) > int(max_replacements):
        return ""

    hit_lines_1 = [(i + 1) for i in hits]
    wanted = []
    try:
        wanted = [int(x) for x in selected_start_lines_1]
    except Exception:
        return ""

    # Validate: must be subset (no unknown indices)
    wanted_set = set(wanted)
    if not wanted_set.issubset(set(hit_lines_1)):
        return ""

    # Map wanted start lines -> start_idx
    start_indices = [ln1 - 1 for ln1 in wanted if ln1 >= 1]

    m = len(before_lines)
    after_min = min_indent(after_lines_base)

    new_lines = list(lines)
    for start_idx in sorted(set(start_indices), reverse=True):
        end_idx = start_idx + m
        if start_idx < 0 or end_idx > len(new_lines):
            return ""
        before_slice = new_lines[start_idx:end_idx]
        before_min = min_indent(before_slice)
        file_indent_char = detect_indent_char(before_slice)
        local_after = shift_block(after_lines_base, before_min, after_min, file_indent_char)
        new_lines[start_idx:end_idx] = local_after

    after_text = "\n".join(new_lines) + "\n"
    return synthesize_replace_file_unified_diff(repo_root, rel, after_text)

def synthesize_replace_line_range_unified_diff(
    repo_root: str,
    rel_path: str,
    start_line_1: int,
    end_line_1: int,
    after_block: str,
    *,
    context_lines: int = 1,
    lines: Optional[list[str]] = None,
    ends_with_newline: Optional[bool] = None,
) -> str:
    """
    Deterministically replace an inclusive 1-based line range [start_line_1, end_line_1]
    with after_block (multi-line allowed), returning a unified diff.

    Returns empty string on invalid range or missing file.

    If ``lines`` is provided (pre-read file lines), skips the internal file read.
    ``ends_with_newline`` is the pre-fetched trailing-newline flag for ``lines``;
    when ``lines`` is None it is read from disk along with the content.
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel:
        return ""

    if lines is None or ends_with_newline is None:
        info = _read_text_lines_with_eof(repo_root, rel)
        if info is None:
            return ""
        if lines is None:
            lines = info[0]
        if ends_with_newline is None:
            ends_with_newline = info[1]
    if not lines:
        return ""

    try:
        s1 = int(start_line_1)
        e1 = int(end_line_1)
    except Exception:
        return ""

    n = len(lines)
    if s1 <= 0 or e1 <= 0 or e1 < s1 or e1 > n:
        return ""

    start_idx = s1 - 1
    end_idx = e1  # exclusive

    after_lines = _split_block_lines(after_block)

    ctx = max(0, int(context_lines))
    lo = max(0, start_idx - ctx)
    hi = min(n, end_idx + ctx)

    old_start_1 = lo + 1
    old_count = hi - lo

    new_start_1 = old_start_1
    new_count = (start_idx - lo) + len(after_lines) + (hi - end_idx)

    out: list[str] = []
    out.append(f"diff --git a/{rel} b/{rel}")
    out.append(f"--- a/{rel}")
    out.append(f"+++ b/{rel}")
    out.append(f"@@ -{old_start_1},{old_count} +{new_start_1},{new_count} @@")

    # Build the body with prefixes, then add EOF markers via the shared helper.
    body: list[str] = []
    for ln in lines[lo:start_idx]:
        body.append(" " + ln)
    for ln in lines[start_idx:end_idx]:
        body.append("-" + ln)
    for ln in after_lines:
        body.append("+" + ln)
    for ln in lines[end_idx:hi]:
        body.append(" " + ln)

    # EOF markers only matter when the hunk reaches the file's final line (hi==n).
    if hi == n:
        # For an in-place line/range replace, the new file's trailing-newline
        # state EQUALS the old file's: the final '\n' (or its absence) is a
        # property of the file position, and the replacement text is always
        # newline-stripped before emission. The marker therefore matches on
        # both sides except in the delete-at-EOF case, which _emit_hunk_replace_body
        # detects and re-states. (Whole-file rewrite via replace_file is a
        # separate difflib path and is handled there.)
        old_eof_no_newline = not ends_with_newline
        out.extend(_emit_hunk_replace_body(
            body,
            old_eof_no_newline=old_eof_no_newline,
            new_eof_no_newline=old_eof_no_newline,
        ))
    else:
        out.extend(body)

    return "\n".join(out) + "\n"


def synthesize_delete_line_unified_diff(
    repo_root: str,
    rel_path: str,
    line_text: str,
    *,
    require_unique: bool = True,
    context_lines: int = 1,
) -> str:
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel:
        return ""

    info = _read_text_lines_with_eof(repo_root, rel)
    if info is None:
        return ""
    lines, ends_with_newline = info
    if not lines:
        return ""

    # âœ… í•µì‹¬: ë¬¸ìžì—´ì„ "ì •ê·œì‹"ìœ¼ë¡œ ì»´íŒŒì¼í•´ì„œ _find_line_matchesì— ë„˜ê¹€
    needle = _norm_line_for_match(line_text or "")
    if not needle:
        return ""

    rx = re.compile(rf"^\s*{re.escape(needle)}\s*$")
    hits0 = _find_line_matches(lines, rx, normalized=True)

    if not hits0:
        return ""
    if require_unique and len(hits0) != 1:
        return ""

    idx0 = hits0[0]
    line_no_1 = idx0 + 1

    return synthesize_replace_line_range_unified_diff(
        repo_root=repo_root,
        rel_path=rel,
        start_line_1=line_no_1,
        end_line_1=line_no_1,
        after_block="",
        context_lines=context_lines,
        lines=lines,
        ends_with_newline=ends_with_newline,
    )


def synthesize_replace_line_unified_diff(
    repo_root: str,
    rel_path: str,
    before_line: str,
    after_line: str,
    *,
    require_unique: bool = True,
    context_lines: int = 3,
    lines: Optional[list[str]] = None,
    ends_with_newline: Optional[bool] = None,
) -> str:
    """
    Replace exactly one line (before_line) with after_line.
    Whitespace-insensitive match (normalized).

    If ``lines`` is provided (pre-read file lines), skips the internal file read.
    ``ends_with_newline`` is the pre-fetched trailing-newline flag; threaded to
    the range synth so EOF-no-newline markers are emitted without a re-read.
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel:
        return ""

    if lines is None or ends_with_newline is None:
        info = _read_text_lines_with_eof(repo_root, rel)
        if info is None:
            return ""
        if lines is None:
            lines = info[0]
        if ends_with_newline is None:
            ends_with_newline = info[1]
    if not lines:
        return ""

    needle = _norm_line_for_match(before_line or "")
    if not needle:
        return ""

    rx = re.compile(rf"^\s*{re.escape(needle)}\s*$")
    hits0 = _find_line_matches(lines, rx, normalized=True)

    if not hits0:
        return ""
    if require_unique and len(hits0) != 1:
        return ""

    idx0 = hits0[0]
    line_no_1 = idx0 + 1
    return synthesize_replace_line_range_unified_diff(
        repo_root=repo_root,
        rel_path=rel,
        start_line_1=line_no_1,
        end_line_1=line_no_1,
        after_block=(after_line or "").rstrip("\r\n"),
        context_lines=context_lines,
        lines=lines,
        ends_with_newline=ends_with_newline,
    )



def synthesize_delete_first_exact_block_unified_diff(
    repo_root: str,
    rel_path: str,
    before_block: str,
    require_unique: bool = True,
    context_lines: int = 1,
) -> str:
    """Delete an exact literal block. Match must be unique if require_unique."""
    return synthesize_replace_first_exact_block_unified_diff(
        repo_root=repo_root,
        rel_path=rel_path,
        before_block=before_block,
        after_block="",
        require_unique=require_unique,
        context_lines=context_lines,
    )


def _norm_ws_line_for_block_match(s: str) -> str:
    # Normalize for tolerant block matching:
    # - ignore leading indentation
    # - ignore trailing whitespace
    # - treat tabs as spaces
    # - collapse internal whitespace runs to a single space
    s = (s or "").rstrip().lstrip().replace("\t", " ")
    s = " ".join(s.split())
    return s


def _split_block_lines(block_text: str) -> list[str]:
    block_text = (block_text or "").replace("\r\n", "\n").replace("\r", "\n")
    block_text = block_text.strip("\n")
    if not block_text:
        return []
    return block_text.split("\n")


def _find_all_block_occurrences_tolerant(file_lines: list[str], block_lines: list[str]) -> list[int]:
    """Return 0-based start indices where block_lines matches file_lines, with tolerant whitespace rules."""
    if not file_lines or not block_lines:
        return []
    n = len(file_lines)
    m = len(block_lines)
    if m > n:
        return []
    needle = [_norm_ws_line_for_block_match(x) for x in block_lines]
    file_norm = [_norm_ws_line_for_block_match(x) for x in file_lines]
    starts: list[int] = []
    for i in range(0, n - m + 1):
        ok = True
        for j in range(m):
            if file_norm[i + j] != needle[j]:
                ok = False
                break
        if ok:
            starts.append(i)
    return starts


def find_replace_block_hits_tolerant(
    repo_root: str,
    rel_path: str,
    before_block: str,
) -> list[int]:
    """
    Return 0-based start indices where before_block matches (tolerant whitespace).
    UI/diagnostics use (shows locations when multiple matches occur).
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel or before_block is None:
        return []
    file_lines = _read_text_lines(repo_root, rel)
    if file_lines is None:
        return []
    block_lines = _split_block_lines(before_block)
    return _find_all_block_occurrences_tolerant(file_lines, block_lines)


def find_closest_replace_block_candidates(
    repo_root: str,
    rel_path: str,
    before_block: str,
    *,
    max_candidates: int = 3,
    window_pad_lines: int = 2,
    max_excerpt_lines: int = 40,
) -> list[dict]:
    """
    Find top-N closest windows in target file for the given before_block.

    Purpose: help LLM/UI diagnose "block_not_found" due to context drift/whitespace/version skew.
    Returns list of dict:
      {
        "rank": 1,
        "similarity": 0.83,
        "start_line_1": 123,
        "end_line_1": 140,
        "excerpt": "  121 | ...\\n>>>123 | ...",
      }
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel:
        return []
    if before_block is None:
        return []

    file_lines = _read_text_lines(repo_root, rel)
    if file_lines is None:
        return []

    before_lines = _split_block_lines(before_block)

    if not file_lines or not before_lines:
        return []

    m = len(before_lines)
    n = len(file_lines)
    if m <= 0 or m > n:
        return []

    # Normalize lines for similarity (tolerant-ish) and cache once for reuse
    needle_norm = "\n".join(_norm_ws_line_for_block_match(x) for x in before_lines)
    file_norm = [_norm_ws_line_for_block_match(x) for x in file_lines]  # pre-normalize once

    scored: list[tuple[float, int]] = []

    # Slide fixed-size window and score (O(1) per window after pre-normalization)
    for i in range(0, n - m + 1):
        cand_norm = "\n".join(file_norm[i : i + m])
        # autojunk=False prevents SequenceMatcher from discarding common lines as "junk"
        # when the sequence length exceeds 200, which would distort similarity scores
        score = SequenceMatcher(None, needle_norm, cand_norm, autojunk=False).ratio()
        scored.append((float(score), int(i)))

    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)

    out: list[dict] = []
    used_starts: set[int] = set()

    pad = max(0, int(window_pad_lines))
    cap_lines = max(10, int(max_excerpt_lines))

    for score, start0 in scored:
        if len(out) >= max(1, int(max_candidates)):
            break
        if start0 in used_starts:
            continue
        used_starts.add(start0)

        end0 = start0 + m  # exclusive
        lo = max(0, start0 - pad)
        hi = min(n, end0 + pad)

        # Build numbered excerpt with ">>>" mark at start line
        excerpt_lines: list[str] = []
        for ln0 in range(lo, hi):
            ln1 = ln0 + 1
            prefix = ">>> " if ln0 == start0 else "    "
            excerpt_lines.append(f"{prefix}{ln1:4d} | {file_lines[ln0]}")

        # Hard clip excerpt line count
        if len(excerpt_lines) > cap_lines:
            excerpt_lines = [*excerpt_lines[:cap_lines], "... (clipped)"]

        out.append(
            {
                "rank": int(len(out) + 1),
                "similarity": float(round(score, 4)),
                "start_line_1": int(start0 + 1),
                "end_line_1": int(end0),
                "excerpt": "\n".join(excerpt_lines),
            }
        )

    return out


def synthesize_replace_all_matching_blocks_unified_diff(
    repo_root: str,
    rel_path: str,
    before_block: str,
    after_block: str,
    *,
    max_replacements: int = 50,
) -> str:
    """
    Replace ALL matching blocks (tolerant whitespace) by rewriting the file and emitting a unified diff.
    - Safety guard: returns "" when matches exceed max_replacements
    - indent-heal: shift the AFTER block's min indent to match each BEFORE slice's min indent
    """
    rel = normalize_rel_path_fast(rel_path)
    if not repo_root or not rel or before_block is None:
        return ""

    info = _read_text_lines_with_eof(repo_root, rel)
    if info is None:
        return ""
    old_lines, old_ends_with_newline = info

    before_lines = _split_block_lines(before_block)
    after_lines_base = _split_block_lines(after_block)
    if not before_lines:
        return ""

    hits = _find_all_block_occurrences_tolerant(old_lines, before_lines)
    if not hits:
        return ""
    if len(hits) > int(max_replacements):
        return ""

    m = len(before_lines)
    after_min = min_indent(after_lines_base)

    new_lines = list(old_lines)
    # Replace in reverse order (avoids index invalidation)
    for start_idx in sorted(hits, reverse=True):
        end_idx = start_idx + m
        before_slice = new_lines[start_idx:end_idx]
        before_min = min_indent(before_slice)
        file_indent_char = detect_indent_char(before_slice)
        local_after = shift_block(after_lines_base, before_min, after_min, file_indent_char)
        new_lines[start_idx:end_idx] = local_after

    # Build the unified diff (using difflib: stable patch-hunk numbering)
    import difflib as _difflib

    diff_lines = list(
        _difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    if not diff_lines:
        return ""

    # difflib never emits "\ No newline at end of file"; post-process the last
    # hunk. An in-place block replace preserves the file's trailing-newline
    # state on both sides, so old/new EOF flags are equal.
    diff_lines = _difflib_apply_eof_markers(
        diff_lines,
        old_eof_no_newline=not old_ends_with_newline,
        new_eof_no_newline=not old_ends_with_newline,
    )

    out: list[str] = []
    out.append(f"diff --git a/{rel} b/{rel}")
    out.extend(diff_lines)
    return "\n".join(out) + "\n"






# ============================================================
# Whole-file rewrite (difflib)
# ============================================================
def synthesize_replace_file_unified_diff(
    repo_root: str,
    rel_path: str,
    after_text: str,
) -> str:
    """Rewrite an entire file by computing a unified diff via difflib.

    This is deterministic and avoids asking the LLM to format unified diff hunks.
    Returns an empty string if no changes are needed or if file path is invalid.
    """
    rel = normalize_rel_path_fast(rel_path)
    if not rel:
        return ""

    # Read old (with EOF flag for the difflib post-processor).
    info = _read_text_lines_with_eof(repo_root, rel)
    if info is None:
        # IMPORTANT: distinguish "no changes" from "cannot read original file".
        # main.py instruction-mode catches ValueError("failed_to_read_file:...") and returns
        # a safe FAILED response instead of attempting destructive rewrites.
        raise ValueError(f"failed_to_read_file:{rel}")
    old_lines, old_ends_with_newline = info
    # split on LF only (NOT splitlines, which would over-split on U+2028/NEL/FF
    # and disagree with git's line counting); also capture the new content's
    # trailing-newline state.
    new_lines, new_ends_with_newline = _split_diff_lines(after_text or "")

    if old_lines == new_lines:
        return ""

    import difflib as _difflib

    diff_lines = list(
        _difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    if not diff_lines:
        return ""

    # difflib never emits "\ No newline at end of file"; post-process the last
    # hunk. A whole-file rewrite respects BOTH the original and the new content's
    # trailing-newline state independently.
    diff_lines = _difflib_apply_eof_markers(
        diff_lines,
        old_eof_no_newline=not old_ends_with_newline,
        new_eof_no_newline=not new_ends_with_newline,
    )

    out: list[str] = []
    out.append(f"diff --git a/{rel} b/{rel}")
    out.extend(diff_lines)
    return "\n".join(out) + "\n"
