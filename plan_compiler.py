"""
PLAN -> multi-file unified diff compiler for asicode.

Goal
- Take a Cursor-style structured plan (ASICODE_PLAN_V1) and deterministically
  compile it into a git-apply compatible unified diff that can touch multiple files.

Design notes
- Does NOT call an LLM. Pure deterministic compilation.
- Safe path handling: all paths are validated to stay inside repo_root.
- Operations supported (V1):
  - create_file(path, content)
  - replace_file(path, content)
  - edit_blocks(path, edits=[{before, after}, ...], expect_unique=True)
  - insert_after(path, anchor, lines=[...], expect_unique=True)
  - insert_before(path, anchor, lines=[...], expect_unique=True)

Expected integration
- External LLM (or local LLM) outputs ASICODE_PLAN_V1 JSON.
- Server loads plan, calls compile_plan_to_unified_diff(), then proceeds as usual
  (diff cleaning / git-apply check-only / run_store, etc.)

Patch correctness notes
- difflib.unified_diff is sensitive to whether input lines include trailing "\\n".
  We feed it newline-stripped lines (keepends=False) and set lineterm="\\n".
  Then we join the generated diff lines with "" (not "\\n") to avoid double blank lines.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from common import ensure_trailing_newline
from external_llm.common.indent_utils import detect_indent_char, indent_unit, reindent_to_match
from path_security import resolve_inside_repo

logger = logging.getLogger(__name__)

_SUPPORTED_OPS = [
    "create_file",
    "replace_file",
    "edit_blocks",
    "insert_after",
    "insert_before",
    "insert_after_line",  # line-number based insert; no text matching required
]


# ============================================================
# Errors
# ============================================================


class PlanCompileError(RuntimeError):
    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


# ============================================================
# Helpers
# ============================================================


def _norm_rel_path(p: str) -> str:
    rel = (p or "").strip().lstrip("/").replace("\\", "/")
    if not rel:
        raise PlanCompileError("invalid path: empty", details={"path": p})
    # forbid obvious traversal even before resolve_inside_repo
    if ".." in rel.split("/"):
        raise PlanCompileError("invalid path: contains '..'", details={"path": p})
    return rel


def _norm_op_type(op_type_raw: str) -> str:
    """
    Normalize operation type names from external LLMs.

    Accepts variants like:
      - CREATE_FILE / create_file / create-file / createFile
      - REPLACE_FILE / replaceFile / replace-file
      - EDIT_BLOCKS / editBlocks / edit-blocks
      - INSERT_AFTER / insertAfter / insert-after
      - INSERT_BEFORE / insertBefore / insert-before
    """
    s = str(op_type_raw or "").strip()
    if not s:
        return ""

    # lowercase, normalize separators, strip weird chars (keep a-z0-9 and underscore)
    s = s.lower().replace("-", "_").replace(" ", "_")
    s = "".join(ch for ch in s if (ch.isalnum() or ch == "_"))

    # handle camelCase after lowercase (e.g., createfile)
    aliases = {
        "createfile": "create_file",
        "replacefile": "replace_file",
        "replace": "replace_file",
        "editblocks": "edit_blocks",
        "insertafter": "insert_after",
        "insertbefore": "insert_before",
    }
    return aliases.get(s, s)


def _read_text_if_exists(abs_path: Path) -> Optional[str]:
    try:
        if abs_path.exists():
            return abs_path.read_text(encoding="utf-8")
    except Exception as e:
        raise PlanCompileError(
            "failed to read file",
            details={"path": str(abs_path), "error": str(e)},
        ) from e
    return None


# ============================================================
# Content-normalization gates
# ============================================================

# Extensions whose file bodies are "code-like": a whole file that is a single
# JSON-string-escaped value is, for these, virtually always a double-encoding
# artifact (safe to decode). Used to GATE encoding-recovery transforms in
# _normalize_str_content so that data/text formats are not corrupted.
_CODE_LIKE_EXTS = frozenset(
    {
        # general-purpose languages
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".go", ".rs", ".java", ".kt", ".kts", ".scala", ".c", ".h", ".cpp",
        ".hpp", ".cc", ".cxx", ".mm", ".swift", ".rb", ".php", ".cs", ".m",
        # jvm
        ".clj", ".cljc",
        # scripting / dynamic
        ".pl", ".pm", ".lua", ".tcl", ".r", ".jl", ".ex", ".exs",
        ".dart", ".groovy", ".gradle",
        # shell
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
        # functional / lisp
        ".el", ".scm", ".ss", ".lisp",
        # web markup that embeds code
        ".vue", ".svelte", ".astro",
    }
)


def _path_is_code_like(path: str) -> bool:
    """True if ``path``'s extension marks it as source code, i.e. a file whose
    entire body being a single JSON-string-escaped value is almost certainly a
    double-encoding artifact (and therefore safe to decode).

    Data/text formats (.json, .txt, .md, .yaml, .csv, .html, .xml, .css, ...)
    return False: the same string shape is legitimate content there.
    """
    p = str(path).lower().rsplit("/", 1)[-1]
    dot = p.rfind(".")
    if dot < 0:
        return False
    return p[dot:] in _CODE_LIKE_EXTS


def _normalize_str_content(s: str, *, path: str, recover_encoding: bool = True) -> str:
    """Apply per-string normalization to a single content value.

    Args:
        s:    raw content string from an LLM plan.
        path: target file path (relative). Used as the safety GATE for the
              encoding-recovery transforms (see below).
        recover_encoding:
              When True (default — whole-file content) the LLM-encoding-recovery
              transforms below run. When False (insert-op line payloads) they are
              SKIPPED entirely and ``s`` is returned unchanged.

    Two classes of transform (only when ``recover_encoding``):
      (1) '++' diff-prefix recovery — content-shape based (not encoding),
          applied to ALL file types.
      (2) Double-JSON-decode + literal ``\\n``/``\\t`` unescape — GATED on
          the target being a source-code file (``_path_is_code_like``).

    Why the gate: a whole file whose content is a single JSON-string-escaped
    value (outer quotes + ``\\"``) is virtually always a double-encoding
    artifact for *code*, but can be a legitimate file body for data/text
    formats (.json, .txt, .md, .yaml, .csv). Applying these to data files
    silently corrupts user content. Round-trip identity does NOT discriminate
    the two (JSON always round-trips); the file extension is the only
    discriminator that actually works.

    Why ``recover_encoding=False`` for insert-op lines: by the time an insert
    op (insert_after / insert_before / insert_after_line) reaches the compiler,
    the JSON request has *already* been decoded by the tool framework. Each
    ``lines`` element is one logical line. A literal backslash-n inside an
    element is therefore NOT a double-encoding artifact — it is a legitimate
    escape sequence in the target language (e.g. Python ``"\\n"``).
    Unescaping it splits the line and, for Python, turns ``x = "foo\\nbar"``
    into ``x = "foo`` + newline + ``bar"`` → ``unterminated string literal``.
    Whole-file content keeps the recovery because there the element boundary
    does not encode a line boundary.
    """
    if not recover_encoding:
        return s

    # (1) '++' diff-prefix heuristic — shape-based, not encoding-based;
    # applies to every file type.
    if "\n" not in s:
        plusplus = s.count("++")
        if plusplus >= 20 and ("from __future__" in s or "import " in s or "def " in s or "class " in s):
            s2 = s.replace("++", "\n")
            lines = s2.splitlines()
            starts_plus = sum(1 for ln in lines if ln.startswith("+"))
            if lines and (starts_plus / max(1, len(lines))) >= 0.7:
                lines = [ln[1:] if ln.startswith("+") else ln for ln in lines]
                s = "\n".join(lines)

    # (2) Encoding-recovery transforms — gated on a code-like target.
    # For data/text files the same string shape is legitimate content,
    # not a double-encoding artifact; decoding it would corrupt user data.
    if not _path_is_code_like(path):
        return s

    # Undo accidental double JSON-encoding: whole content is a JSON string
    # literal (outer quotes + escaped inner chars). Require len > 1 so a
    # legitimate quote-only 1-char file is left alone.
    #
    # Require *some* escape sequence inside (``\\"``, ``\\n``, ``\\t``, ``\\``):
    # a plain ``"foo"`` with no escapes is a legitimate (if unusual) body and is
    # left untouched, whereas any escape present means the value was JSON-encoded.
    # The earlier ``'\\"' in s``-only test missed quote-free code whose only
    # encoded char was a newline (e.g. ``json.dumps("x = 1\\nprint(x)")`` ->
    # ``"x = 1\\nprint(x)"`` has no ``\\"``), leaving the outer quotes in the file.
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and (
        '\\"' in s or '\\n' in s or '\\t' in s or '\\\\' in s
    ):
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                s = decoded
        except Exception:
            pass

    # Literal backslash-n / backslash-t unescape when no real newline present.
    if "\n" not in s and "\\n" in s:
        s = s.replace("\\n", "\n").replace("\\t", "\t")

    return s


def _normalize_file_content(op_obj: dict[str, Any], *, path: str, content_key: str = "content") -> str:
    """
    Normalize WHOLE-FILE content coming from LLM plans (create_file / replace_file).

    Reads the payload from ``op_obj[content_key]`` (falling back to the
    canonical ``"content"`` key when absent) and returns a single string.
    A list payload is joined with newlines (one file's worth of text).
    """
    raw = op_obj.get(content_key)
    if raw is None and content_key != "content":
        raw = op_obj.get("content")
    if isinstance(raw, list):
        return "\n".join([_normalize_str_content(str(x), path=path) for x in raw])
    return _normalize_str_content(str(raw or ""), path=path)


def _normalize_insert_lines_payload(op_obj: dict[str, Any], *, path: str, content_key: str = "lines") -> list[str]:
    """
    Normalize INSERT-OP line payloads (insert_after / insert_before / insert_after_line).

    Unlike whole-file content, an insert op's ``lines`` list treats EACH
    element as a single line (trailing newline handled downstream by
    ``_normalize_insert_lines``/``_strip_lno_prefix_no_space``). So a list
    is NOT joined-then-split — each element is normalized in place and any
    newlines inside an element are flattened into separate lines, which
    avoids spurious empty lines (e.g. ``["x\\n"]`` must become ``["x"]``,
    not ``["x", ""]``).

    Returns a list of line strings (no trailing newline).
    """
    raw = op_obj.get(content_key)
    if raw is None and content_key != "content":
        raw = op_obj.get("content")
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            # recover_encoding=False: see _normalize_str_content docstring.
            # An element here is one logical line; its embedded backslash-n is a
            # language escape sequence (Python "\n"), not a double-encoding
            # artifact. splitlines() still flattens any *real* newline the tool
            # framework already decoded from JSON.
            s = _normalize_str_content(str(x), path=path, recover_encoding=False)
            out.extend(s.splitlines() if s else [])
        return out
    s = _normalize_str_content(str(raw or ""), path=path, recover_encoding=False)
    return s.splitlines()


def _unified_diff_for_file(rel_path: str, old: Optional[str], new: str) -> str:
    """
    git-apply compatible:
      diff --git a/x b/x
      --- a/x (or /dev/null)
      +++ b/x (or /dev/null)
      @@ ...

    Newline-accurate: lines are split with keepends=True so the diff faithfully
    represents whether old/new end with a trailing newline.  A content line that
    lacks a terminator emits the standard "\\ No newline at end of file" marker.

    This matters because git-apply matches context EXACTLY, including the final
    line's newline.  Forcing a trailing newline onto a file that lacks one (the
    previous behavior) produced a diff whose old-side context did not match the
    real file, so git-apply rejected it ("patch failed" → --3way → "lacks blob").
    """
    # NOTE: For create_file, we MUST use "/dev/null" (with leading slash), otherwise
    # git-apply may treat it as a literal path "dev/null".
    fromfile = f"a/{rel_path}" if old is not None else "/dev/null"
    tofile = f"b/{rel_path}"

    a_lines = (old or "").splitlines(keepends=True) if old is not None else []
    b_lines = (new or "").splitlines(keepends=True)

    hunks = list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=fromfile,
            tofile=tofile,
            n=3,
        )
    )

    # difflib may produce no hunks if identical.
    if not hunks:
        # Creating an empty file: "".splitlines() == [] so difflib yields no
        # hunks for [] → []. git apply still creates the file from the
        # "new file mode" header alone, so emit a header-only create diff.
        if old is None:
            return f"diff --git a/{rel_path} b/{rel_path}\nnew file mode 100644\n"
        return ""

    out_lines: list[str] = []
    for ln in hunks:
        if ln.startswith(("--- ", "+++ ", "@@")):
            # difflib's control lines; ensure each ends with exactly one newline.
            out_lines.append(ln if ln.endswith("\n") else ln + "\n")
        elif ln.endswith("\n"):
            out_lines.append(ln)
        else:
            # Last line of old/new without a terminator (keepends preserved this).
            out_lines.append(ln + "\n")
            out_lines.append("\\ No newline at end of file\n")
    diff_body = "".join(out_lines)

    # Prepend git diff header (required by your pipeline / strict mode)
    if old is None:
        header = f"diff --git a/{rel_path} b/{rel_path}\nnew file mode 100644\n"
    else:
        header = f"diff --git a/{rel_path} b/{rel_path}\n"
    return header + diff_body


def _has_lno_prefix(line: str) -> bool:
    """Check if line has a line-number prefix like '  28: ' or '28:\t'."""
    s = line.lstrip()
    colon_idx = s.find(":")
    return colon_idx > 0 and s[:colon_idx].isdigit()


def _strip_lno_prefix(line: str) -> str:
    """Strip prefix: leading ws + digits + colon + optional tab + optional space."""
    s = line.lstrip()
    colon_idx = s.find(":")
    if colon_idx <= 0 or not s[:colon_idx].isdigit():
        return line
    start = colon_idx + 1
    if start < len(s) and s[start] == "\t":
        start += 1
    if start < len(s) and s[start] == " ":
        start += 1
    return s[start:]


def _strip_lno_prefix_no_space(line: str) -> str:
    """Strip prefix: leading ws + digits + colon + optional tab only (no space)."""
    s = line.lstrip()
    colon_idx = s.find(":")
    if colon_idx <= 0 or not s[:colon_idx].isdigit():
        return line
    start = colon_idx + 1
    if start < len(s) and s[start] == "\t":
        start += 1
    return s[start:]


def _strip_lno_prefixes(text: str) -> str:
    """Strip line-number prefixes (e.g. '  28: code') from each line.

    Only strips when the majority (≥50%) of non-empty lines match the pattern,
    indicating the text was pasted verbatim from a read output.
    """
    if not text:
        return text
    lines = text.split("\n")
    non_empty = [_item_ for _item_ in lines if _item_.strip()]
    if not non_empty:
        return text
    prefix_count = sum(1 for _item_ in non_empty if _has_lno_prefix(_item_))
    if prefix_count / len(non_empty) >= 0.5:
        return "\n".join(_strip_lno_prefix(_item_) for _item_ in lines)
    return text


# _reindent_to_match → imported from external_llm.common.indent_utils.reindent_to_match


# Map decorative Unicode (box-drawing, em/en-dash) to ASCII equivalents so a
# `before` block that uses a different separator glyph than the file still
# matches in the decorative-tolerant fallback below.
_DECORATIVE_TRANSLATION = str.maketrans({
    "─": "-", "━": "-", "—": "-", "–": "-", "⎯": "-", "⏤": "-",
    "═": "=",
    "│": "|", "┃": "|", "┌": "|", "┐": "|", "└": "|", "┘": "|",
    "├": "|", "┤": "|", "┬": "|", "┴": "|", "┼": "|",
})
# Collapse runs (2+) of horizontal separator chars to a single char, so that
# "# ── name ─────" and "# - name -------" normalize identically. LLMs almost
# never reproduce the exact length of a decorative separator run.
_DECORATIVE_RUN_RE = re.compile(r"([-=_~*])\1+")


def _normalize_decorative(s: str) -> str:
    """NFC-normalize, fold decorative Unicode to ASCII, and collapse separator runs.

    Used only as a last-resort fallback in edit_blocks matching, after exact and
    whitespace-normalized matching have both failed. Mirrors the Unicode-tolerant
    fallback in the edit_text tool's matcher.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_DECORATIVE_TRANSLATION)
    s = _DECORATIVE_RUN_RE.sub(r"\1", s)
    return s


def _restore_decorative_lines(after: str, actual_before: str) -> str:
    """Restore the file's exact decorative separators in the ``after`` block.

    When a decorative-normalized match succeeds, the LLM's ``after`` text may
    carry a mangled separator (e.g. ASCII ``----`` where the file has box-drawing
    ``────``). For any ``after`` line that is decoratively-equivalent to a file
    line but rendered differently, restore the file's original bytes so a fuzzy
    match doesn't rewrite a separator the LLM never meant to change.
    """
    before_lines = actual_before.split("\n")
    norm_to_orig: dict[str, str] = {}
    for bl in before_lines:
        norm_to_orig.setdefault(_normalize_decorative(bl.strip()), bl)

    result = []
    for al in after.split("\n"):
        orig = norm_to_orig.get(_normalize_decorative(al.strip()))
        # Only restore when the line differs but is decoratively equivalent —
        # i.e. the LLM mangled a separator rather than editing real content.
        result.append(orig if orig is not None and orig != al else al)
    return "\n".join(result)


def _apply_edit_blocks(
    *,
    text: str,
    edits: list[dict[str, str]],
    expect_unique: bool,
    path: str,
) -> str:
    out = text
    # Detect the file's per-level indent width once.  A flat (single-level)
    # match site cannot reveal it, so without this hint reindent_to_match
    # assumes the file shares the snippet's unit — over-indenting a 2-space
    # file when the LLM snippet is 4-space.  The file's unit is stable across
    # edits, so one detection up front suffices.
    try:
        _file_unit = indent_unit(text, detect_indent_char(text.split("\n"))) or None
    except Exception:
        _file_unit = None
    for i, e in enumerate(edits or []):
        before = _strip_lno_prefixes(str(e.get("before") or ""))
        after = _strip_lno_prefixes(str(e.get("after") or ""))

        if not before:
            raise PlanCompileError(
                "edit_blocks: missing 'before'",
                details={"path": path, "edit_index": i},
            )

        count = out.count(before)
        if expect_unique and count != 1 and count != 0:
            # Collect line numbers of every match so the LLM can see exactly
            # which locations are ambiguous and add more surrounding context.
            out_lines = out.split("\n")
            match_locations: list[dict[str, object]] = []
            search_start = 0
            before_line_count = before.count("\n") + 1
            while True:
                pos = out.find(before, search_start)
                if pos == -1:
                    break
                line_no = out[:pos].count("\n") + 1
                ctx_start = max(0, line_no - 2)
                ctx_end = min(len(out_lines), line_no - 1 + before_line_count + 2)
                snippet = "\n".join(
                    f"{ctx_start + j + 1:4d}: {out_lines[ctx_start + j]}"
                    for j in range(ctx_end - ctx_start)
                )
                match_locations.append({"line": line_no, "snippet": snippet})
                search_start = pos + len(before)

            location_summary = "; ".join(f"line {m['line']}" for m in match_locations)
            snippets_text = "\n\n".join(
                f"--- Match {idx + 1} (line {m['line']}) ---\n```\n{m['snippet']}\n```"
                for idx, m in enumerate(match_locations)
            )
            error_msg = (
                f"edit_blocks: 'before' block matched {count} locations "
                f"({location_summary}). Add more surrounding lines to make it unique.\n\n"
                f"{snippets_text}"
            )
            raise PlanCompileError(
                error_msg,
                details={
                    "path": path,
                    "edit_index": i,
                    "match_count": count,
                    "match_locations": [m["line"] for m in match_locations],
                },
            )
        if count == 0:
            # --- Fuzzy fallback: try whitespace-normalized matching ---
            before_lines = before.split("\n")
            out_lines = out.split("\n")
            fuzzy_start = -1
            blen = len(before_lines)
            if blen > 0:
                for si in range(len(out_lines) - blen + 1):
                    segment = out_lines[si : si + blen]
                    if all(a.strip() == b.strip() for a, b in zip(segment, before_lines, strict=False)):
                        fuzzy_start = si
                        break

            if fuzzy_start >= 0:
                # Rebuild actual_before from file lines to preserve indentation
                actual_before = "\n".join(out_lines[fuzzy_start : fuzzy_start + blen])
                logger.debug(
                    "edit_blocks fuzzy match at line %d (whitespace-normalized) for path=%s",
                    fuzzy_start + 1, path,
                )
                after_fixed = reindent_to_match(after, actual_before, file_unit=_file_unit)
                if after_fixed != after:
                    logger.debug(
                        "edit_blocks reindented 'after' via fuzzy match for path=%s",
                        path,
                    )
                out = out.replace(actual_before, after_fixed, 1)
                continue  # successfully applied via fuzzy match

            # --- Decorative-tolerant fallback: normalize Unicode separators ---
            # Whitespace-normalized matching failed. LLM-generated `before`
            # blocks frequently mangle decorative separator lines (box-drawing
            # runs like "# ── name ─────"): wrong dash count, or em-dash vs
            # box-drawing glyph. Normalize both sides (NFC + decorative→ASCII +
            # collapse runs) and retry line-wise before giving up.
            if blen > 0:
                norm_before = [_normalize_decorative(b.strip()) for b in before_lines]
                for si in range(len(out_lines) - blen + 1):
                    segment = out_lines[si : si + blen]
                    if all(
                        _normalize_decorative(a.strip()) == nb
                        for a, nb in zip(segment, norm_before, strict=False)
                    ):
                        fuzzy_start = si
                        break

            if fuzzy_start >= 0:
                actual_before = "\n".join(out_lines[fuzzy_start : fuzzy_start + blen])
                logger.debug(
                    "edit_blocks decorative-normalized match at line %d for path=%s",
                    fuzzy_start + 1, path,
                )
                after_fixed = reindent_to_match(after, actual_before, file_unit=_file_unit)
                after_fixed = _restore_decorative_lines(after_fixed, actual_before)
                out = out.replace(actual_before, after_fixed, 1)
                continue  # successfully applied via decorative-normalized match

            # No exact or fuzzy match — build helpful error with file context
            before_first_line = before.split("\n")[0] if before else ""
            before_preview = before_first_line[:60] + ("..." if len(before_first_line) > 60 else "")

            similar_lines = []
            if before_first_line:
                similar_lines = difflib.get_close_matches(
                    before_first_line, out_lines, n=3, cutoff=0.6
                )

            error_msg = (
                f"edit_blocks: block not found. First line: '{before_preview}'"
                if before_preview
                else "edit_blocks: block not found (empty before block)"
            )
            if similar_lines:
                suggestions = ", ".join(
                    f"'{ln[:60]}{'...' if len(ln) > 60 else ''}'" for ln in similar_lines[:2]
                )
                error_msg += f"\nDid you mean: {suggestions}"

            # Find best-matching location in the file via SequenceMatcher.
            # This gives the LLM the *actual* file content near the expected location,
            # so it can copy the exact text for the 'before' block on retry.
            best_ratio = 0.0
            best_start = -1
            before_sample = before[:300]  # limit comparison to first 300 chars for speed
            blen = max(1, len(before_lines))
            for si in range(max(0, len(out_lines) - blen + 1)):
                segment = "\n".join(out_lines[si : si + blen])
                ratio = difflib.SequenceMatcher(None, before_sample, segment[:300]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = si

            file_context: Optional[str] = None
            if best_start >= 0 and best_ratio > 0.35:
                ctx_start = max(0, best_start - 2)
                ctx_end = min(len(out_lines), best_start + blen + 3)
                ctx_lines = [
                    f"{ctx_start + j + 1:4d}: {out_lines[ctx_start + j]}"
                    for j in range(ctx_end - ctx_start)
                ]
                file_context = "\n".join(ctx_lines)
                error_msg += (
                    f"\n\nFile content near best match (line {best_start + 1}, "
                    f"similarity {best_ratio:.0%}) — copy the exact text you need:\n"
                    f"```\n{file_context}\n```"
                )

            raise PlanCompileError(
                error_msg,
                details={
                    "path": path,
                    "edit_index": i,
                    "before_first_line": before_preview,
                    "similar_lines": similar_lines[:2],
                    "file_context_near_match": file_context,
                    "best_match_line": best_start + 1 if best_start >= 0 else None,
                    "best_match_ratio": round(best_ratio, 2),
                },
            )

        # Default: replace first occurrence (even if multiple and expect_unique=False)
        out = out.replace(before, after, 1)

    return out


def _find_anchor_line_index(lines: list[str], anchor: str, expect_unique: bool, *, path: str) -> int:
    hits = [idx for idx, ln in enumerate(lines) if anchor in ln]
    if not hits:
        # Extract anchor preview (max 60 chars)
        anchor_preview = anchor[:60] + ('...' if len(anchor) > 60 else '')

        # Find similar lines in the file
        similar_lines = []
        if anchor:
            similar_lines = difflib.get_close_matches(
                anchor,
                lines,
                n=3,
                cutoff=0.6
            )

        # Build error message with suggestions
        if anchor_preview:
            error_msg = f"anchor not found: '{anchor_preview}'"
        else:
            error_msg = "anchor not found (empty anchor)"

        if similar_lines:
            suggestions = ', '.join(f"'{line[:60]}{'...' if len(line) > 60 else ''}'" for line in similar_lines[:2])
            error_msg += f"\nDid you mean: {suggestions}"

        raise PlanCompileError(
            error_msg,
            details={
                "path": path,
                "anchor": anchor_preview,
                "similar_lines": similar_lines[:2]
            },
        )
    if expect_unique and len(hits) != 1:
        raise PlanCompileError(
            "anchor match count is not 1",
            details={"path": path, "anchor": anchor, "match_count": len(hits)},
        )
    return hits[0]


def _apply_insert_at(
    *,
    text: str,
    anchor: str,
    insert_lines: list[str],
    expect_unique: bool,
    path: str,
    after: bool = True,
) -> str:
    """Insert lines after (after=True) or before (after=False) anchor match."""
    src = ensure_trailing_newline(text)
    lines = src.splitlines(keepends=True)

    idx = _find_anchor_line_index(lines, anchor, expect_unique, path=path)
    _ins = _normalize_insert_lines(insert_lines)
    cut = idx + 1 if after else idx
    return "".join(lines[:cut] + _ins + lines[cut:])


def _normalize_insert_lines(insert_lines: list[str]) -> list[str]:
    """Ensure every insert line has a trailing newline."""
    ins = [
        ensure_trailing_newline(ln).splitlines(keepends=True)[0] if "\n" not in ln else ln
        for ln in insert_lines
    ]
    ins = [ln if ln.endswith("\n") else (ln + "\n") for ln in ins]

    return ins


def _apply_insert_after(
    *,
    text: str,
    anchor: str,
    insert_lines: list[str],
    expect_unique: bool,
    path: str,
) -> str:
    return _apply_insert_at(
        text=text, anchor=anchor, insert_lines=insert_lines,
        expect_unique=expect_unique, path=path, after=True,
    )


def _apply_insert_before(
    *,
    text: str,
    anchor: str,
    insert_lines: list[str],
    expect_unique: bool,
    path: str,
) -> str:
    return _apply_insert_at(
        text=text, anchor=anchor, insert_lines=insert_lines,
        expect_unique=expect_unique, path=path, after=False,
    )


# ============================================================
# Public API
# ============================================================


@dataclass
class PlanCompileResult:
    diff_patch: str
    picked_files: list[str]
    warnings: list[str]
    meta: dict[str, Any]
    # Final post-edit content per touched file (rel_path -> text). Lets callers
    # apply the plan by writing content directly instead of round-tripping the
    # diff through `git apply` (which can reject a correctly-computed result over
    # context fuzz, line endings, untracked paths, or a missing --3way blob).
    staged: dict[str, str] = field(default_factory=dict)


def compile_plan_to_unified_diff(
    *,
    repo_root: str,
    plan: dict[str, Any],
    allow_empty: bool = False,
) -> PlanCompileResult:
    """
    Compile ASICODE_PLAN_V1 -> unified diff.

    allow_empty:
      - False (default): raises if the plan results in no diff (all no-ops).
      - True: returns empty diff_patch.
    """
    if not isinstance(plan, dict):
        raise PlanCompileError("plan must be an object", details={"type": str(type(plan))})

    # Accept both canonical ("kind"/"ops") and common alias keys ("version"/"operations")
    kind = str(plan.get("kind") or plan.get("version") or "").strip()
    if kind != "ASICODE_PLAN_V1":
        # NOTE: do not reference per-op variables here (idx/op/rel/etc). This runs before op loop.
        raise PlanCompileError(
            "unsupported plan kind",
            details={
                "kind": kind,
                "expected": "ASICODE_PLAN_V1",
                "available_ops": _SUPPORTED_OPS,
                "available_keys": sorted(list(plan.keys())),
            },
        )

    ops = plan.get("ops")
    if ops is None:
        ops = plan.get("operations")
    if ops is None:
        ops = []
    if not isinstance(ops, list) or not ops:
        raise PlanCompileError(
            "ops must be a non-empty list",
            details={"kind": kind, "available_keys": sorted(list(plan.keys()))},
        )

    repo_root_p = Path(str(repo_root))
    warnings: list[str] = []
    picked_files_set: set[str] = set()

    # Stage final file contents (relative path -> text)
    staged: dict[str, str] = {}

    for idx, op in enumerate(ops):
        if not isinstance(op, dict):
            raise PlanCompileError("op must be an object", details={"op_index": idx})

        op_type_raw = op.get("op") if ("op" in op) else op.get("type")
        op_type = _norm_op_type(str(op_type_raw or ""))

        if op_type not in _SUPPORTED_OPS:
            raise PlanCompileError(
                "unsupported op type",
                details={
                    "op_index": idx,
                    "op_type_raw": op_type_raw,
                    "op_type": op_type,
                    "supported": _SUPPORTED_OPS,
                    "available_keys": sorted(list(op.keys())),
                },
            )

        rel = _norm_rel_path(str(op.get("path") or ""))
        abs_path = resolve_inside_repo(repo_root_p, rel)

        # Load current staged base: staged -> repo file -> empty (for create/replace)
        if rel in staged:
            cur_text = staged[rel]
        else:
            cur_text = _read_text_if_exists(abs_path) or ""

        if op_type == "create_file":
            if abs_path.exists():
                raise PlanCompileError(
                    "create_file: file already exists",
                    details={"op_index": idx, "path": rel},
                )
            content = op.get("content")
            if content is None:
                raise PlanCompileError(
                    "create_file: content is required",
                    details={"op_index": idx, "path": rel},
                )
            staged[rel] = _normalize_file_content(op, path=rel)
            picked_files_set.add(rel)
            continue

        if op_type == "replace_file":
            content = op.get("content")
            if content is None:
                raise PlanCompileError(
                    "replace_file: content is required",
                    details={"op_index": idx, "path": rel},
                )
            new_content = _normalize_file_content(op, path=rel)
            # Safety: guard against accidental content wipe.
            # Use the in-plan staged state (``cur_text``) as the baseline, NOT disk:
            # within a single plan an earlier create_file/edit_blocks may have changed
            # the content, so reading disk here would (a) miss the just-staged size and
            # (b) return None for create_file'd files not yet on disk, silently
            # disabling the guard. ``cur_text`` is exactly "state right before this op".
            old_for_check = cur_text or None
            if old_for_check and len(old_for_check) > 500:
                ratio = len(new_content) / len(old_for_check)
                if ratio < 0.10:
                    raise PlanCompileError(
                        f"replace_file: new content is {ratio:.0%} of original size — "
                        "this looks like accidental content loss. "
                        "Use edit_blocks for partial edits, or confirm the full replacement is intentional.",
                        details={
                            "op_index": idx, "path": rel,
                            "old_chars": len(old_for_check),
                            "new_chars": len(new_content),
                            "ratio": round(ratio, 3),
                        },
                    )
                elif ratio < 0.30:
                    warnings.append(
                        f"replace_file '{rel}': new content is {ratio:.0%} of original "
                        f"({len(new_content)} vs {len(old_for_check)} chars). "
                        "Verify the full replacement is intentional."
                    )
            staged[rel] = new_content
            picked_files_set.add(rel)
            continue

        if op_type == "edit_blocks":
            # Accept both "edits" (preferred) and "blocks" (legacy / agent output)
            edits = op.get("edits")
            if edits is None:
                edits = op.get("blocks")
            if not isinstance(edits, list) or not edits:
                raise PlanCompileError(
                    "edit_blocks.edits must be a non-empty list",
                    details={"op_index": idx, "path": rel, "available_keys": sorted(list(op.keys()))},
                )
            expect_unique = bool(op.get("expect_unique", True))
            staged[rel] = _apply_edit_blocks(text=cur_text, edits=edits, expect_unique=expect_unique, path=rel)
            picked_files_set.add(rel)
            continue

        if op_type == "insert_after_line":
            line_no = op.get("line")
            try:
                line_no = int(line_no)
            except (TypeError, ValueError):
                raise PlanCompileError(
                    "insert_after_line: 'line' must be an integer (1-based line number)",
                    details={"op_index": idx, "path": rel},
                )
            # Normalize the payload (handles list, literal \n, double-encoded quotes)
            # via the same path as create_file/replace_file. 'lines' is the canonical
            # key, but 'content' is accepted as a fallback.
            if op.get("lines") is None and op.get("content") is None:
                raise PlanCompileError(
                    "insert_after_line: 'lines' or 'content' is required",
                    details={"op_index": idx, "path": rel},
                )
            lines_to_insert = _normalize_insert_lines_payload(op, path=rel, content_key="lines")
            # Strip line-number prefixes that small models copy verbatim
            # e.g. "302:     api_version: str" → "    api_version: str"
            lines_to_insert = [_strip_lno_prefix_no_space(ln) for ln in lines_to_insert]
            # Ensure trailing newline after stripping
            lines_to_insert = [ln if ln.endswith("\n") else ln + "\n" for ln in lines_to_insert]
            src = ensure_trailing_newline(cur_text)
            src_lines = src.splitlines(keepends=True)
            # line_no is 1-based; clamp to valid range
            insert_idx = max(0, min(line_no, len(src_lines)))
            staged[rel] = "".join(src_lines[:insert_idx] + lines_to_insert + src_lines[insert_idx:])
            picked_files_set.add(rel)
            continue

        if op_type in ("insert_after", "insert_before"):
            anchor = str(op.get("anchor") or "")
            if not anchor:
                raise PlanCompileError(
                    f"{op_type}: anchor is required",
                    details={"op_index": idx, "path": rel},
                )
            # Normalize the payload (handles list, literal \n, double-encoded quotes).
            # 'lines' is the canonical key, but 'content' is accepted as a fallback.
            if op.get("lines") is None and op.get("content") is None:
                raise PlanCompileError(
                    f"{op_type}.lines must be a non-empty list",
                    details={"op_index": idx, "path": rel},
                )
            lines = _normalize_insert_lines_payload(op, path=rel, content_key="lines")
            if not lines:
                raise PlanCompileError(
                    f"{op_type}.lines must be a non-empty list",
                    details={"op_index": idx, "path": rel},
                )
            expect_unique = bool(op.get("expect_unique", True))
            if op_type == "insert_after":
                staged[rel] = _apply_insert_after(
                    text=cur_text,
                    anchor=anchor,
                    insert_lines=[str(x) for x in lines],
                    expect_unique=expect_unique,
                    path=rel,
                )
            else:
                staged[rel] = _apply_insert_before(
                    text=cur_text,
                    anchor=anchor,
                    insert_lines=[str(x) for x in lines],
                    expect_unique=expect_unique,
                    path=rel,
                )
            picked_files_set.add(rel)
            continue

        # Should be unreachable due to supported ops check.
        raise PlanCompileError("internal: op not handled", details={"op_index": idx, "op_type": op_type})

    # Now compute unified diff across all touched files (deterministic order)
    diffs: list[str] = []
    picked_files = sorted(picked_files_set)

    for rel in picked_files:
        abs_path = resolve_inside_repo(Path(str(repo_root)), rel)
        old = _read_text_if_exists(abs_path)  # None if missing
        new = staged.get(rel)

        # If a file was in picked_files but not in staged (shouldn't happen), treat as no-op
        if new is None:
            warnings.append(f"warning: picked file not staged (no-op): {rel}")
            continue

        d = _unified_diff_for_file(rel, old, new)
        if d:
            diffs.append(d)

    diff_patch = "".join(diffs)

    if not diff_patch:
        if allow_empty:
            return PlanCompileResult(
                diff_patch="",
                picked_files=picked_files,
                warnings=[*warnings, "allow_empty: true; produced empty diff"],
                meta={"plan_kind": kind, "touched_files": picked_files},
                staged=dict(staged),
            )
        raise PlanCompileError(
            "plan resulted in empty diff (no-op)",
            details={"touched_files": picked_files, "warnings": warnings},
        )

    return PlanCompileResult(
        diff_patch=diff_patch,
        picked_files=picked_files,
        warnings=warnings,
        meta={"plan_kind": kind, "touched_files": picked_files},
        staged=dict(staged),
    )


def parse_plan_json(text: str) -> dict[str, Any]:
    """
    Strict-ish plan JSON loader (helper).

    This is intentionally minimal; the server can decide how strict to be.
    """
    try:
        obj = json.loads(text)
    except Exception as e:
        raise PlanCompileError("invalid JSON", details={"error": str(e)}) from e
    if not isinstance(obj, dict):
        raise PlanCompileError("plan must be a JSON object")
    return obj
