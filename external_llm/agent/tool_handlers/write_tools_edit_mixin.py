"""edit-family write tool handlers (P2-2 split).

``WriteToolsEditMixin`` — edit_text, edit_file, create_file, modify_symbol and
the syntax/rollback gate helpers. Split out of ``WriteToolsMixin`` in
``write_tools.py``; recombined there via ``class WriteToolsMixin(...,
WriteToolsEditMixin, ...)``.
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...common.atomic_io import atomic_write_bytes, atomic_write_text
from ...common.text_reading import read_text_with_encoding_fallback
from ...languages import LanguageId
from .._shared_utils import compile_quiet, extract_files_from_patch
from ..write_targets import recover_args_from_raw as _recover_args_from_raw
from ..write_targets import try_repair_truncated_json as _try_repair_truncated_json
from .write_tools_core import (
    _detect_file_unit,
    _find_block_end_line,
    _leading_indent_width,
    _reindent_to_match,
    _repo_file_index,
)

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)


class WriteToolsEditMixin:
    """Edit-family write handlers: edit_text, edit_file, modify_symbol."""

    # Host contracts (P29 pattern): ToolRegistry / WriteToolsPatchMixin own
    # these at runtime via MRO combination; annotations only — never assign.
    repo_root: str
    _effective_repo_root: Any
    defer_semantic_check: Any
    _make_result: Any
    _secure_path: Any
    _should_soft_fail_verify: Any
    _refuse_foreign_leased: Any
    _acquire_edit_leases: Any
    _append_applied_patch: Any
    _text_edited_files: set[str]

    def _resolve_edit_anchor(
        self,
        modified: str,
        anchor: str,
        line: int | None = None,
    ) -> tuple[int, str, float]:
        """Find anchor in modified text with exact-match fallback strategies.

        Returns (position, actual_anchor_text, match_ratio).
        match_ratio is always 0.0 (exact/line-hint/first-line matches only;
        fuzzy matching is disabled to avoid false-positive anchor resolution).
        Raises ValueError with close-match suggestions on complete failure.
        """
        import difflib

        _lines = modified.splitlines(True)

        # 1. Exact match — but the contract requires a UNIQUE anchor. Silently
        #    taking the first of several matches is the tool's worst failure mode
        #    (it edits an unintended, often mid-line, location). So:
        #      • exactly one match  → use it
        #      • multiple matches   → disambiguate via the line hint, else fail loud
        _count = modified.count(anchor)
        if _count == 1:
            return modified.find(anchor), anchor, 0.0
        if _count > 1:
            if line is not None and 1 <= line <= len(_lines):
                _target_byte = sum(len(_lines[i]) for i in range(line - 1))
                _positions: list[int] = []
                _p = modified.find(anchor)
                while _p != -1:
                    _positions.append(_p)
                    _p = modified.find(anchor, _p + 1)
                _best = min(_positions, key=lambda q: abs(q - _target_byte))
                logger.info(
                    "edit_file: anchor not unique (%d matches) — line hint %d → byte %d",
                    _count,
                    line,
                    _best,
                )
                return _best, anchor, 0.0
            raise ValueError(
                f"anchor not unique: found {_count} occurrences of {anchor[:60]!r}. "
                "Include 2-3 surrounding lines to make it unique, or pass a 'line' hint."
            )

        # 2. Line hint → use actual content at that line
        if line is not None and 1 <= line <= len(_lines):
            _line_content = _lines[line - 1].rstrip("\n\r")
            if _line_content:
                # Compute exact byte offset from line number (handles duplicates correctly)
                _byte_pos = sum(len(_lines[i]) for i in range(line - 1))
                if modified[_byte_pos : _byte_pos + len(_line_content)] == _line_content:
                    logger.info(
                        "edit_file: line hint %d → exact byte offset",
                        line,
                    )
                    return _byte_pos, _line_content, 0.0

        # 3. First anchor line → strip-match in search window
        _first_line = anchor.split("\n", maxsplit=1)[0].strip()
        if _first_line:
            _search_start = 0
            _search_end = len(_lines)
            if line is not None:
                _search_start = max(0, line - 1 - 5)
                _search_end = min(len(_lines), line + 5)

            # Single-line anchor: enforce the same uniqueness contract as the
            # exact-match path. Silently taking the first of several stripped
            # matches is exactly the wrong-location failure mode the exact
            # path fails loud on — the fallback must not reintroduce it.
            if "\n" not in anchor:
                _strip_matches = [
                    _smi for _smi in range(_search_start, _search_end) if _lines[_smi].strip() == _first_line
                ]
                if not _strip_matches:
                    pass  # fall through to the error path below
                else:
                    if len(_strip_matches) == 1:
                        _li = _strip_matches[0]
                    elif line is not None:
                        _li = min(_strip_matches, key=lambda q: abs(q - (line - 1)))
                        logger.info(
                            "edit_file: %d stripped matches — line hint %d → line %d",
                            len(_strip_matches),
                            line,
                            _li + 1,
                        )
                    else:
                        raise ValueError(
                            f"anchor matches {len(_strip_matches)} lines when ignoring "
                            f"indentation ({_first_line[:60]!r}). Include 2-3 surrounding "
                            "lines to make it unique, or pass a 'line' hint."
                        )
                    _raw = _lines[_li].rstrip("\n\r")
                    _li_byte_pos = sum(len(_lines[i]) for i in range(_li))
                    logger.info("edit_file: first-line match → line %d", _li + 1)
                    return _li_byte_pos, _raw, 0.0

            for _li in range(_search_start, _search_end):
                if _lines[_li].strip() == _first_line:
                    _raw = _lines[_li].rstrip("\n\r")
                    pos = modified.find(_raw)
                    # _raw IS a line of ``modified`` (the strip match succeeded on
                    # the same line), so find() always succeeds.
                    if pos == -1:  # pragma: no cover
                        continue
                    # Multi-line anchor: reconstruct from actual file content
                    _anchor_line_count = anchor.count("\n") + 1
                    if _anchor_line_count > 1 and _li + _anchor_line_count <= len(_lines):
                        _actual = "".join(_lines[_li : _li + _anchor_line_count]).rstrip("\n\r")
                        if modified.count(_actual) == 1:
                            _actual_pos = modified.find(_actual)
                            if _actual_pos != -1:
                                logger.info(
                                    "edit_file: first-line match → line %d (reconstructed)",
                                    _li + 1,
                                )
                                return _actual_pos, _actual, 0.0
                        # Progressive fallback: try shorter anchor suffix (2+ lines).
                        # Uniqueness must be verified for each variant before returning;
                        # otherwise a common prefix could match the wrong location.
                        if _anchor_line_count > 2:
                            for _try_lines in range(_anchor_line_count - 1, 1, -1):
                                _partial = "".join(_lines[_li : _li + _try_lines]).rstrip("\n\r")
                                # A duplicated full block implies duplicated
                                # prefixes (same text), so this is unreachable.
                                if _partial and modified.count(_partial) == 1:  # pragma: no cover
                                    _actual_pos = modified.find(_partial)
                                    if _actual_pos != -1:
                                        logger.info(
                                            "edit_file: first-line match → progressive %d/%d lines at line %d",
                                            _try_lines,
                                            _anchor_line_count,
                                            _li + 1,
                                        )
                                        return _actual_pos, _partial, 0.0
                        # Multi-line reconstruction failed (duplicate/out-of-bounds/progressive)
                        # → continue searching; do NOT fall through to single-line return
                        continue
                    # Use the byte offset of THIS matched line, not modified.find(_raw):
                    # _raw can occur earlier (as a substring or a duplicate line), and
                    # find() would resolve to that wrong location. Mirrors the multi-line
                    # path's _li_byte_pos fix above.
                    _li_byte_pos = sum(len(_lines[i]) for i in range(_li))
                    logger.info("edit_file: first-line match → line %d", _li + 1)
                    return _li_byte_pos, _raw, 0.0

        # (step 4 — fuzzy match disabled: caused false-positive anchor resolution
        #  leading to syntax errors when applied at the wrong location)

        # 5. Build helpful error
        _suggestions = ""
        if _first_line:
            _close = difflib.get_close_matches(
                _first_line,
                [ln.strip() for ln in modified.splitlines()],
                n=3,
                cutoff=0.4,
            )
            if _close:
                _suggestions = f". Did you mean: {_close}"

        raise ValueError(f"anchor text not found: {anchor[:80]!r}{_suggestions}")

    def _tool_edit_file(self, args: dict[str, Any]) -> ToolResult:
        """Edit a single file using anchor-based text operations.

        No diff/patch syntax needed.  Supports:
          replace       -- find *anchor* text and replace it with *content*
          insert_after  -- insert *content* after *anchor*
          insert_before -- insert *content* before *anchor*

        Operations are applied **sequentially** in order.  If one fails the
        whole call is rolled back and an error is returned.
        """
        import time as _time

        start_time = _time.monotonic()
        args = self._recover_args_from_raw(args, ("path",))
        file_path = (args.get("path") or "").strip()
        ops = args.get("operations") or args.get("ops") or []
        # If operations is empty but __raw_arguments has serialized JSON, try to recover
        if not ops and "__raw_arguments" in args:
            _raw_ops = self._extract_ops_from_raw(args["__raw_arguments"])
            if _raw_ops:
                ops = _raw_ops
        if not file_path:
            # If __raw_arguments is present, the JSON was likely truncated during streaming
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(
                ok=False,
                error=f"path is required{_raw_hint}",
                execution_time=0,
            )
        if not ops:
            return self._make_result(
                ok=False, error=f"operations list cannot be empty for {file_path}", execution_time=0
            )

        # Repo-boundary check — see _tool_edit_text. Placed at the resolution
        # point rather than next to the arg read, so it covers the path whichever
        # key supplied it (`path`, or `file_path` recovered from raw arguments).
        # `Path(repo_root) / file_path` below happily walks out of the repo on a
        # leading `../`, and .exists() then confirms the victim file.
        # The RESULT is reused as `_norm`: resolved + bias-corrected, so a
        # repo-internal symlink stays a symlink (atomic os.replace on an
        # unresolved path would replace it with a regular file).
        _secured = self._secure_path(file_path, confine=True)
        if _secured is None:
            return self._make_result(
                ok=False,
                error=f"Path blocked (outside repo): {file_path}",
                execution_time=0,
            )

        # F1 cross-process edit-lease guard.
        _lease_refused = self._refuse_foreign_leased([file_path], start_time)
        if _lease_refused is not None:
            return _lease_refused

        _norm = _secured
        if not _norm.exists():
            return self._make_result(
                ok=False, error=f"File not found: {file_path}{self._suggest_missing_paths(file_path)}", execution_time=0
            )

        try:
            original = _norm.read_text(encoding="utf-8")
        except Exception as e:
            return self._make_result(ok=False, error=f"Failed to read {file_path}: {e}", execution_time=0)

        # Detect file line ending style so insert operations preserve it
        _file_newline = "\r\n" if "\r\n" in original else "\n"
        modified = original
        _op_type_counts: dict[str, int] = {}
        _edit_warnings: list[str] = []
        for i, op in enumerate(ops):
            op_type = op.get("type", "replace")
            anchor = op.get("anchor", "")
            content = op.get("content", "")
            if not anchor:
                return self._make_result(ok=False, error=f"Operation {i}: 'anchor' is required")

            try:
                _pos, _actual_anchor, _fuzzy_ratio = self._resolve_edit_anchor(
                    modified,
                    anchor,
                    op.get("line"),
                )
            except ValueError as e:
                return self._make_result(
                    ok=False,
                    error=f"Operation {i}: {e}",
                )

            # The resolver returns 0.0 for every strategy (fuzzy matching is
            # disabled), so this branch is unreachable.
            if _fuzzy_ratio > 0.0:  # pragma: no cover
                _warn = f"op {i}: anchor fuzzy-matched at ratio={_fuzzy_ratio:.2f} (anchor={anchor[:60]!r}, matched={_actual_anchor[:60]!r})"
                logger.warning("edit_file %s", _warn)
                _edit_warnings.append(_warn)

            if op_type == "replace":
                _op_type_counts["replace"] = _op_type_counts.get("replace", 0) + 1
                # Defensive: warn if content contains the anchor (likely copy-paste error)
                if anchor in content:
                    _warn = f"op {i}: content contains anchor text — possible content/anchor inversion (anchor_len={len(anchor)}, content_len={len(content)})"
                    logger.warning("edit_file %s", _warn)
                    _edit_warnings.append(_warn)
                # Heuristic: warn if content >> anchor suggests whole-file intent
                if len(content) > len(anchor) * 20 and len(content) > 500:
                    _warn_content_anchor_ratio = (
                        f"op {i}: content ({len(content)} chars) is much larger than "
                        f"anchor ({len(anchor)} chars). Did you mean to replace the whole file? "
                        f"If so, use write_plan's replace_file or create_file(overwrite=true) instead."
                    )
                    logger.warning("edit_file %s", _warn_content_anchor_ratio)
                    _edit_warnings.append(_warn_content_anchor_ratio)
                modified = modified[:_pos] + content + modified[_pos + len(_actual_anchor) :]

            elif op_type == "insert_after":
                _op_type_counts["insert_after"] = _op_type_counts.get("insert_after", 0) + 1
                _eol = modified.find("\n", _pos + len(_actual_anchor))

                # ── Block-end auto-correction (mirrors anchor_edit) ────────
                # If the anchor line is a block header (def/class/if/... in
                # Python, a '{'-opening line in brace languages), the EOL
                # computed above points at the END of the header line — so the
                # insert would nest the content INSIDE the body. Find the
                # block's real end (language-agnostic via _find_block_end_line)
                # and move the insertion point past it so the content becomes a
                # sibling. Installing a grammar enables this for that language
                # with no code change.
                if _eol != -1:
                    _ef_anchor_lineno = modified[:_pos].count("\n")
                    _ef_lines = modified.splitlines(True)
                    _ef_block_end = _find_block_end_line(
                        modified,
                        LanguageId.from_path(str(_norm)).value,
                        _ef_anchor_lineno,
                        _ef_lines,
                    )
                    if _ef_block_end is not None and _ef_block_end > _ef_anchor_lineno:
                        _ef_new_eol = sum(len(_l) for _l in _ef_lines[: _ef_block_end + 1]) - 1
                        if 0 <= _ef_new_eol < len(modified) and modified[_ef_new_eol] == "\n":
                            _eol = _ef_new_eol
                            logger.info(
                                "edit_file op %d (insert_after): anchor L%d is a "
                                "%d-line block header — inserting after block end "
                                "L%d instead of into the body",
                                i,
                                _ef_anchor_lineno + 1,
                                _ef_block_end - _ef_anchor_lineno + 1,
                                _ef_block_end + 1,
                            )

                if _eol == -1:
                    # Anchor line is the last line of a file with no trailing
                    # newline — terminate it first, otherwise the slice below
                    # glues the inserted content onto the anchor line.
                    modified += _file_newline
                    _eol = len(modified) - 1  # index of the '\n' just added
                # Idempotency check: skip if content already exists after the anchor.
                # Second clause covers the EOF case (no trailing newline) — exact
                # match only, since a permissive startswith would false-positive on
                # any prefix (e.g. content="x = 1" vs after_text="x = 123\n...").
                _after_text = modified[_eol + 1 :]
                _normalized_content = content.rstrip("\r\n") + _file_newline
                _already_present = _after_text.startswith(_normalized_content) or _after_text == content.rstrip("\r\n")
                if _already_present:
                    logger.info(
                        "edit_file op %d (insert_after): content already present after anchor — skipping (idempotent)",
                        i,
                    )
                else:
                    modified = modified[: _eol + 1] + _normalized_content + modified[_eol + 1 :]

            elif op_type == "insert_before":
                _op_type_counts["insert_before"] = _op_type_counts.get("insert_before", 0) + 1
                # Idempotency check: skip if content (normalized) already exists before anchor.
                # Checks the text immediately preceding _pos (after the last newline before anchor).
                _before_text = modified[:_pos]
                _candidate_end = _before_text.rfind("\n", 0, _pos)
                _candidate = _before_text[:_pos] if _candidate_end == -1 else _before_text[_candidate_end + 1 : _pos]
                _candidate_norm = _candidate.rstrip("\r\n")
                _content_norm = content.rstrip("\r\n")
                _already_present = _candidate_norm == _content_norm
                # Also check multi-line: content might span multiple lines before anchor
                if not _already_present:
                    _content_with_newline = _content_norm + _file_newline
                    _already_present = _before_text.endswith(_content_with_newline)
                if _already_present:
                    logger.info(
                        "edit_file op %d (insert_before): content already present before anchor -- skipping (idempotent)",
                        i,
                    )
                else:
                    modified = modified[:_pos] + content.rstrip("\r\n") + _file_newline + modified[_pos:]

            else:
                return self._make_result(
                    ok=False,
                    error=f"Operation {i}: unknown type '{op_type}' (expected replace, insert_after, insert_before)",
                )

        try:
            atomic_write_text(str(_norm), modified)
        except Exception as e:
            return self._make_result(ok=False, error=f"Failed to write {file_path}: {e}")

        _orig_lines = original.splitlines()
        _mod_lines = modified.splitlines()
        _delta = len(_mod_lines) - len(_orig_lines)
        # Compute separate add/remove counts via a simple line-diff
        _added_lines = _removed_lines = 0
        for line in difflib.unified_diff(_orig_lines, _mod_lines, n=0):
            if line.startswith(("+++", "---")):
                continue
            if line.startswith("+"):
                _added_lines += 1
            elif line.startswith("-"):
                _removed_lines += 1
        # Build op-type breakdown for the summary
        _op_type_breakdown = " + ".join(
            f"{_op_type_counts[k]} {k}"
            for k in ("replace", "insert_after", "insert_before")
            if _op_type_counts.get(k, 0)
        )
        _line_detail = f" (+{_added_lines}, -{_removed_lines})" if _added_lines or _removed_lines else ""
        # Structural validation: warn if replace op type resulted in zero removed lines
        _replace_count = _op_type_counts.get("replace", 0)
        if _replace_count > 0 and _added_lines > 0 and _removed_lines == 0:
            _warn = (
                f"{_replace_count} replace op(s) resulted in +{_added_lines}, -0 — "
                "replace structurally always removes the anchor text. "
                "This suggests the intended operation was insert_after or insert_before, not replace."
            )
            logger.warning("edit_file op-type mismatch: %s", _warn)
            _edit_warnings.append(_warn)
        _meta: dict[str, Any] = {}
        if _edit_warnings:
            _meta["edit_warnings"] = _edit_warnings
        _syn = self._run_syntax_check_for_file(file_path)
        if not _syn.get("skipped"):
            _meta["syntax_check"] = _syn
            if not _syn.get("ok"):
                # Syntax error detected — rollback file to original and return error.
                # This prevents the LLM from operating on a broken file and avoids
                # downstream verify_after_write failures that cause repeated 100K+
                # token edit_file retry loops (observed on asi.py, a large file).
                atomic_write_text(str(_norm), original)
                _error_details = "; ".join(
                    f"line {e.get('line')}:{e.get('col')} \u2014 {e.get('message', '').strip()}"
                    for e in (_syn.get("errors") or [])
                )
                _exec = _time.monotonic() - start_time
                _meta["rollback_reason"] = "syntax_error"
                return self._make_result(
                    ok=False,
                    error=(
                        f"Syntax error after editing {file_path}. Rolled back to original.\nErrors: {_error_details}\n"
                    ),
                    metadata=_meta,
                    execution_time=_exec,
                )
        _exec = _time.monotonic() - start_time
        # Track applied patch so agent_loop can detect successful writes
        self._append_applied_patch(f"edit_file:{file_path}:{_op_type_breakdown}:{_added_lines:+}/{-_removed_lines:-}")
        # F1: stake our lease (edit_file does not call _record_text_edit, so
        # the acquire lives here explicitly).
        self._acquire_edit_leases([file_path])
        return self._make_result(
            ok=True,
            content=f"File updated: {file_path} ({_op_type_breakdown} →{_line_detail}) [{_exec:.1f}s]",
            metadata=_meta,
            execution_time=_exec,
        )

    def _current_file_head_snippet(self, content: str, max_lines: int = 25, max_chars: int = 2000) -> str:
        """Head of the CURRENT file content, read_file-gutter style.

        Attached to search_string_mismatch errors when near-match hinting
        found nothing, so the LLM sees what the file actually looks like now
        (it may have changed since its last read) instead of guessing blindly.
        Returns "" for empty content or when the snippet would exceed the
        char budget (degrade gracefully - the caller keeps the plain error).
        """
        lines = content.splitlines()
        if not lines:
            return ""
        shown: list[str] = []
        _chars = 0
        for _i, _ln in enumerate(lines[:max_lines]):
            _line = f"│{_i + 1}│ {_ln}"
            _chars += len(_line) + 1
            if _chars > max_chars:
                break
            shown.append(_line)
        if not shown:
            return ""
        _tail = "" if len(lines) <= len(shown) else f"\n... ({len(lines) - len(shown)} more lines)"
        return (
            "--- current file content (re-read from disk; the file may have "
            f"changed since you last read it - first {len(shown)} lines) ---\n" + "\n".join(shown) + _tail + "\n---\n"
        )

    def _patch_failure_snippet(self, patch_text: str, path_hint) -> str:
        """Current-file head for a failed apply_patch (stale-target diagnosis).

        The patch failed against the file as it exists now; if the agent wrote
        the patch from an earlier read, the file may have changed. Attach the
        first lines of the primary target so the LLM can rewrite the patch
        against what is actually on disk. Returns "" when no target can be
        determined or read (degrade gracefully - caller keeps the plain error).
        """
        _target = None
        if isinstance(path_hint, str) and path_hint.strip():
            _p = Path(path_hint)
            _target = _p if _p.is_absolute() else Path(self.repo_root) / path_hint
        else:
            try:
                _files = extract_files_from_patch(patch_text)
                if _files:
                    _p = Path(_files[0])
                    _target = _p if _p.is_absolute() else Path(self.repo_root) / _p
            except (TypeError, ValueError):  # malformed patch → no target
                return ""
        if _target is None or not _target.exists():
            return ""
        _content = None
        for _enc in ("utf-8", "latin-1"):
            with contextlib.suppress(UnicodeDecodeError, UnicodeError):
                _content = _target.read_text(encoding=_enc)
                break
        # latin-1 decodes every byte sequence, so the loop always breaks with
        # content set.
        if _content is None:  # pragma: no cover
            return ""
        return self._current_file_head_snippet(_content)

    @staticmethod
    def _raw_repr(text: str, max_lines: int = 3) -> str:
        """Show raw character representation of a text snippet.

        Makes invisible differences (trailing whitespace, unusual Unicode,
        tabs vs spaces, CRLF vs LF) immediately visible.
        Returns an empty string if text is empty.
        """
        if not text:
            return ""
        lines = text.splitlines(keepends=True)
        target = text if len(lines) <= max_lines else "".join(lines[:max_lines]) + f"\n... ({len(lines)} total lines)"
        raw = repr(target)
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1]
        return f"Raw old_string (repr): {raw}\n"

    def _near_match_hint(self, content: str, old_string: str, max_window_lines: int = 200) -> str:
        """Best-effort 'did you mean' hint for a failed exact-match edit.

        When old_string does not match verbatim (the dominant cause being
        leading-whitespace / indentation drift in LLM-reconstructed code),
        locate the file region most similar to old_string and return:
          1. a line-numbered, copyable snippet of the real text, and
          2. a unified diff (your old_string → actual file) so the model
             sees exactly which characters differ instead of blindly
             re-guessing or falling back to fragile shell here-docs.

        Returns "" if no plausible candidate is found or on any error —
        the caller appends the result to the failure message, so a blank
        hint degrades gracefully to the original behaviour.
        """
        try:
            import difflib

            file_lines = content.splitlines()
            old_lines = old_string.splitlines()
            if not file_lines or not old_lines:
                return ""
            window = len(old_lines)
            # Anchor on the first non-blank line of old_string to build a
            # cheap candidate set, then score full windows around each
            # candidate by similarity. Skip the window scan for pathologically
            # large old_strings — the anchor block alone is still useful.
            old_first = next((_item_.strip() for _item_ in old_lines if _item_.strip()), "")
            if not old_first:
                return ""
            stripped = [_item_.strip() for _item_ in file_lines]
            close = difflib.get_close_matches(old_first, stripped, n=5, cutoff=0.4)
            if not close:
                return ""
            close_set = set(close)
            cand_idxs = [i for i, _item_ in enumerate(stripped) if _item_ in close_set]

            old_blob = "\n".join(old_lines)
            best_ratio, best_start = 0.0, -1
            if window <= max_window_lines:
                # P24-1: autojunk=False. The scorer compares the joined
                # old_string at CHARACTER level, so autojunk (default) purges
                # every char appearing in >1% of a >=200-char old_string —
                # e/t/s/space/'\n'/... — from b2j. When the drift sits at a
                # non-popular char (a changed digit/letter: the normal content
                # edit), the aligned anchor vanishes and ratio() collapses to
                # ~0.46-0.84 for a block that is 99.9% identical, dropping the
                # hint below the 0.88 note threshold.
                _sm = difflib.SequenceMatcher(autojunk=False)
                _sm.set_seq2(old_blob)
                # Window scan bounded: score at most 3 starts per candidate
                # line (ci-2..ci, clamped) instead of every start in the full
                # window. old_string matches at or near its first-line anchor,
                # so starts further than 2 lines from the anchor add nothing
                # but O(window·n·m) SequenceMatcher work (a 60-line old_string
                # with 5 candidates used to score up to 300 full-blob
                # comparisons — ~19s for one hint on the P24-1 regression).
                for ci in cand_idxs:
                    lo = max(0, ci - 2)
                    hi = min(len(file_lines) - window + 1, ci + 1)
                    for start in range(lo, hi):
                        region = file_lines[start : start + window]
                        # start <= ci < len(file_lines) and window >= 1, so the
                        # slice is never empty.
                        if not region:  # pragma: no cover
                            continue
                        _sm.set_seq1("\n".join(region))
                        r = _sm.ratio()
                        if r > best_ratio:
                            best_ratio, best_start = r, start
            if best_start < 0:
                # Window scan skipped or inconclusive — anchor on the single
                # best candidate line so the model at least gets a location.
                best_start = cand_idxs[0]
                best_ratio = difflib.SequenceMatcher(None, old_first, stripped[best_start], autojunk=False).ratio()

            ctx_start = max(0, best_start - 2)
            ctx_end = min(len(file_lines), best_start + window + 2)
            numbered = "\n".join(
                f"{ctx_start + j + 1:5d}| {file_lines[ctx_start + j]}" for j in range(ctx_end - ctx_start)
            )
            region = file_lines[best_start : best_start + window]
            diff = "\n".join(
                difflib.unified_diff(
                    old_lines,
                    region,
                    fromfile="your old_string",
                    tofile="actual file",
                    lineterm="",
                )
            )
            # Cap diff size so a wildly-wrong old_string can't bloat the result.
            if len(diff) > 2000:
                diff = diff[:2000] + "\n… (diff truncated)"

            def _decor_norm(s: str) -> str:
                """Strip markdown decoration chars (` * _) for content comparison.

                Equality after normalization means the ONLY difference is
                decoration characters — whitespace and surrounding text are
                identical. Used to distinguish a markdown-decoration drift
                (``x`` vs `x`, **b** vs *b*) from a real content mismatch, so
                the hint can name the cause precisely instead of leaving the
                model to eyeball a 97%-similar diff.
                """
                return s.replace("`", "").replace("*", "").replace("_", "")

            ws_note = ""
            decor_note = ""
            # High similarity + exact-match failure ⇒ the difference is subtle.
            # Classify the first differing line into one of two causes so the
            # hint names it unambiguously:
            #   - whitespace/indent drift (ol.strip() == rl.strip())
            #   - markdown decoration drift (decor-normalized equal, content differs)
            # Whitespace is checked first (it is the dominant failure mode); a
            # decoration note never fires when whitespace is the cause, and vice
            # versa — the two causes are mutually exclusive by construction
            # (decoration chars are non-whitespace, so a whitespace-only diff
            # normalizes away under .strip() while a decoration diff does not).
            if best_ratio >= 0.88:
                for ol, rl in zip(old_lines, region, strict=False):
                    if ol == rl:
                        continue
                    if ol.strip() == rl.strip():

                        def _vis(s):
                            lead = s[: len(s) - len(s.lstrip())]
                            return lead.replace(" ", "·").replace("\t", "⇥") + s.lstrip()

                        ws_note = (
                            f"\nWhitespace differs (· = space, ⇥ = tab):\n  yours: {_vis(ol)!r}\n  file : {_vis(rl)!r}"
                        )
                        break
                    if _decor_norm(ol) == _decor_norm(rl):
                        decor_note = (
                            "\nMarkdown decoration differs (backticks `` ` ``/asterisks "
                            "`*`/underscores `_`). The surrounding text is identical — "
                            "only the decoration characters differ. Copy the EXACT "
                            "decoration from the file (shown in the diff/numbered block)."
                        )
                        break
            return (
                f"\nClosest match (~{best_ratio:.0%} similar) near line {best_start + 1}:\n"
                f"```\n{numbered}\n```\n"
                f"Diff (your old_string vs file):\n```diff\n{diff}\n```"
                f"{ws_note}{decor_note}\n"
                "Copy the EXACT text — including indentation — from the numbered "
                "block above into old_string."
            )
        except (IndexError, ValueError, TypeError):
            return ""

    def _suggest_missing_paths(self, file_path: str) -> str:
        """Return a ". Did you mean: ..." suffix for a path that doesn't exist.

        Mirrors the close-match hints already given for anchor-text and
        symbol resolution failures (the #1 write-tool failure signal here is
        a missing *file*). Ranks repo files by exact-basename > same-stem >
        close-name match, refining ties by path proximity so the closest
        directory wins among same-named files (e.g. many ``__init__.py``).

        Returns "" when nothing plausible is found or on any error, so the
        caller appends the result and degrades to the original message.
        """
        import difflib

        try:
            _missing = (file_path or "").strip()
            if not _missing:
                return ""
            # _effective_repo_root, matching BOTH the other reader (glob) and
            # the two invalidation routines. Reading a different root's index
            # than the invalidator clears is the exact intermittency 1dd10ddb's
            # sibling fix (5a1c405f) called out — "the invalidator could clear a
            # key the reader never used". Latent today (_repo_root_override is
            # never assigned, so the two expressions agree) but free to align.
            _paths = _repo_file_index(str(self._effective_repo_root))
            if not _paths:
                return ""
            _tgt_base = os.path.basename(_missing).lower()
            _tgt_stem = Path(_missing).stem.lower()
            _tgt_full = _missing.replace("\\", "/").lower()

            def _score(rel: str) -> tuple[float, float]:
                rb = os.path.basename(rel).lower()
                if rb == _tgt_base:
                    tier = 1.0
                elif _tgt_stem and Path(rel).stem.lower() == _tgt_stem:
                    tier = 0.8
                else:
                    tier = difflib.SequenceMatcher(None, _tgt_base, rb, autojunk=False).ratio() * 0.7
                # P24-C: autojunk=False for consistency — short basenames sit
                # below the 200-char autojunk threshold, but the full-path
                # comparison can exceed it, and a purged '/' or common char
                # would skew the proximity ranking.
                prox = difflib.SequenceMatcher(None, _tgt_full, rel.lower(), autojunk=False).ratio()
                return tier, prox

            # Stable sort: alphabetical first, then by score descending, so
            # equal-score ties keep deterministic path ordering.
            scored = [(_score(r), r) for r in _paths]
            scored.sort(key=lambda kv: kv[1])
            scored.sort(key=lambda kv: kv[0], reverse=True)

            def _keep(rel: str) -> bool:
                rb = os.path.basename(rel).lower()
                if rb == _tgt_base:
                    return True
                if _tgt_stem and Path(rel).stem.lower() == _tgt_stem:
                    return True
                # 0.75 leans to precision: short common names (conftest.py,
                # __init__.py) false-match around ~0.66, while real typos sit
                # ≥0.77. Exact-basename/stem tiers above are unaffected.
                return difflib.SequenceMatcher(None, _tgt_base, rb, autojunk=False).ratio() >= 0.75

            top = [r for (_s, r) in scored if _keep(r)][:3]
            if not top:
                return ""
            return ". Did you mean: " + ", ".join(top)
        except (OSError, ValueError, TypeError):  # repo walk / path ops
            return ""

    def _ast_fail_hint(self, source: str, ops: list[dict[str, Any]], symbol: str) -> str:
        """'Did you mean' hint for a failed edit_ast call.

        edit_ast fails in two distinct ways the model can't self-diagnose
        from the bare ``no match found`` string:

          1. Symbol resolution — ops like add_guard target a function/method
             by name; if ``symbol`` doesn't resolve, suggest the closest
             defined names (the add_guard failure seen in the wild).
          2. Text search — replace_expr.old / delete_stmt.pattern look for
             existing text; reuse _near_match_hint to surface the real span.

        Returns "" on no candidate or any error — appended to the failure
        message so a blank hint preserves the original behaviour.
        """
        try:
            import ast as _ast
            import difflib

            parts: list[str] = []

            # 1. Symbol resolution suggestions.
            sym = (symbol or "").strip()
            if sym:
                try:
                    tree = _ast.parse(source)
                except SyntaxError:
                    tree = None
                if tree is not None:
                    defined: set[str] = set()
                    qualified: set[str] = set()

                    class _V(_ast.NodeVisitor):
                        def __init__(self):
                            self.stack: list[str] = []

                        def _record(self, name: str):
                            defined.add(name)
                            if self.stack:
                                qualified.add(".".join([*self.stack, name]))

                        def visit_ClassDef(self, node):
                            self._record(node.name)
                            self.stack.append(node.name)
                            self.generic_visit(node)
                            self.stack.pop()

                        def visit_FunctionDef(self, node):
                            self._record(node.name)
                            self.stack.append(node.name)
                            self.generic_visit(node)
                            self.stack.pop()

                        def visit_AsyncFunctionDef(self, node):
                            self._record(node.name)
                            self.stack.append(node.name)
                            self.generic_visit(node)
                            self.stack.pop()

                    _V().visit(tree)
                    bare = sym.split(".")[-1]
                    if bare not in defined and sym not in qualified:
                        pool = sorted(qualified | defined)
                        close = difflib.get_close_matches(bare, pool, n=5, cutoff=0.4)
                        if close:
                            parts.append(f"symbol {sym!r} not found in this file. Did you mean: {close}?")
                        else:
                            _avail = sorted(defined)[:20]
                            if _avail:
                                parts.append(f"symbol {sym!r} not found. Defined here: {_avail}")

            # 2. Text-search op suggestions — only when the searched text is
            #    genuinely absent verbatim (a reliable proxy for "this op is
            #    the one that failed on a mismatch", avoiding misleading hints
            #    for ops that failed for other reasons).
            for op in ops:
                if not isinstance(op, dict):
                    continue
                t = (op.get("type") or "").strip()
                search = ""
                if t == "replace_expr":
                    search = op.get("old") or ""
                elif t == "delete_stmt":
                    search = op.get("pattern") or ""
                if search and search not in source:
                    h = self._near_match_hint(source, search)
                    if h:
                        parts.append(f"[{t}] {h}")

            return ("\n" + "\n".join(parts)) if parts else ""
        except (KeyError, TypeError, ValueError, IndexError):
            return ""

    def _resolve_with_fallback(self, content: str, old_string: str) -> tuple[str, int, list | None, list | None]:
        """Resolve old_string via exact → whitespace-tolerant → unicode-tolerant matching.

        Returns (resolved_old_string, count, fallback_matches, orig_split).

        - resolved_old_string: exact text from file to use in content.replace()/content.index()
        - count: number of occurrences found after all fallback attempts
        - fallback_matches: None if exact match succeeded; [(line_idx, text), ...] for fallback matches
        - orig_split: content.splitlines(keepends=True) from fallback path (None for exact match)
        """
        count = content.count(old_string)
        if count > 0:
            return old_string, count, None, None

        # ── Fallback matching setup ───────────────────────────────────
        _orig_split = content.splitlines(keepends=True)
        _norm_content_lines = [_item_.rstrip() for _item_ in _orig_split]
        _norm_old_lines = [_item_.rstrip() for _item_ in old_string.splitlines()]

        # ── Unicode decoration map ──
        _uni_decorative = str.maketrans(
            {
                "─": "-",
                "━": "-",  # box-drawing horizontal
                "—": "-",
                "\u2013": "-",  # em-dash, en-dash
                "│": "|",
                "┃": "|",  # box-drawing vertical
                "┌": "|",
                "┐": "|",  # box-drawing corners
                "└": "|",
                "┘": "|",  # box-drawing corners
            }
        )

        # ── Whitespace-tolerant fallback ──────────────────────────────
        if _norm_old_lines:
            _ws_matches: list[tuple[int, str]] = []
            for _s_idx in range(len(_norm_content_lines) - len(_norm_old_lines) + 1):
                if _norm_content_lines[_s_idx : _s_idx + len(_norm_old_lines)] == _norm_old_lines:
                    _recon = "".join(_orig_split[_s_idx : _s_idx + len(_norm_old_lines)])
                    # Honor caller's trailing-newline intent
                    if not old_string.endswith(("\n", "\r")):
                        _recon = _recon.rstrip("\r\n")
                    _ws_matches.append((_s_idx, _recon))
            if _ws_matches:
                count = len(_ws_matches)
                resolved = _ws_matches[0][1] if count == 1 else old_string
                return resolved, count, _ws_matches, _orig_split

        # ── Indent-tolerant fallback ──────────────────────────────────
        # Normalize both leading AND trailing whitespace — catches
        # indentation differences (reindent, tab↔space, wrong indent level).
        # An empty/whitespace old_string is rejected by _apply_one_edit_text
        # before reaching here; without that guard, an all-empty
        # _indent_norm_old would spuriously match every line position.
        _indent_norm_content = [_item_.strip() for _item_ in _orig_split]
        _indent_norm_old = [_item_.strip() for _item_ in old_string.splitlines()]
        _indent_matches: list[tuple[int, str]] = []
        for _s_idx in range(len(_indent_norm_content) - len(_indent_norm_old) + 1):
            if _indent_norm_content[_s_idx : _s_idx + len(_indent_norm_old)] == _indent_norm_old:
                _recon = "".join(_orig_split[_s_idx : _s_idx + len(_indent_norm_old)])
                if not old_string.endswith(("\n", "\r")):
                    _recon = _recon.rstrip("\r\n")
                _indent_matches.append((_s_idx, _recon))
        if _indent_matches:
            count = len(_indent_matches)
            resolved = _indent_matches[0][1] if count == 1 else old_string
            return resolved, count, _indent_matches, _orig_split

        # ── Unicode-tolerant fallback ─────────────────────────────────
        import unicodedata

        def _unorm(s):
            s = unicodedata.normalize("NFC", s)
            return s.translate(_uni_decorative)

        _uni_norm_lines = [_unorm(_item_.rstrip()) for _item_ in _orig_split]
        _uni_old_lines = [_unorm(_item_.rstrip()) for _item_ in old_string.splitlines()]
        if _uni_old_lines:
            _uni_matches: list[tuple[int, str]] = []
            for _s_idx in range(len(_uni_norm_lines) - len(_uni_old_lines) + 1):
                if _uni_norm_lines[_s_idx : _s_idx + len(_uni_old_lines)] == _uni_old_lines:
                    _recon = "".join(_orig_split[_s_idx : _s_idx + len(_uni_old_lines)])
                    if not old_string.endswith(("\n", "\r")):
                        _recon = _recon.rstrip("\r\n")
                    _uni_matches.append((_s_idx, _recon))
            if _uni_matches:
                count = len(_uni_matches)
                resolved = _uni_matches[0][1] if count == 1 else old_string
                return resolved, count, _uni_matches, _orig_split

        # ── No match found ──
        return old_string, 0, None, _orig_split

    @staticmethod
    def _edited_line_regions(
        original: str,
        modified: str,
        lineno_1based: int,
        context: int = 1,
    ):
        """Determine whether ``lineno_1based`` falls inside an edited region.

        Used by the edit_text syntax gate to give a scope-aware diagnosis:
        Python reports an INDENTATION error on the line where the parser
        *notices* the inconsistency, which is often several lines below the
        line whose indentation was actually wrong. Without knowing whether
        the reported line was touched by the edit, the LLM cannot tell its
        own indentation mistake from a cascade caused elsewhere.

        Compares ``original`` (pre-edit content) against ``modified``
        (post-edit content) with :class:`difflib.SequenceMatcher` and returns
        a tuple ``(in_edited_region, changed_regions)`` where
        ``changed_regions`` is a list of ``(start_1based, end_1based)``
        inclusive line ranges (in ``modified``) that differ from ``original``.
        A line counts as "edited" if it is within ``context`` lines of any
        differing block (default 1) — this matches how indentation errors
        typically surface one line off.

        Robustness: any difflib/line-split failure degrades gracefully to
        ``(True, [])`` so the gate still refuses (safe default) without
        crashing the tool call.
        """
        try:
            import difflib

            _orig_lines = original.splitlines(keepends=True)
            _mod_lines = modified.splitlines(keepends=True)
            _sm = difflib.SequenceMatcher(a=_orig_lines, b=_mod_lines, autojunk=False)
            regions = []
            for _tag, _i1, _i2, _j1, _j2 in _sm.get_opcodes():
                if _tag == "equal":
                    continue
                if _j1 >= _j2:
                    # pure deletion — no lines in modified. Anchor the region at
                    # the position the deletion happened so a cascade landing
                    # there is still flagged.
                    _start = max(1, _j1)
                    _end = _start
                else:
                    _start = max(1, _j1 + 1)
                    _end = min(len(_mod_lines), _j2)
                regions.append((_start, _end))
            if not regions:
                return True, []  # couldn't find a change but content differs — be safe
            in_region = any(max(1, s - context) <= lineno_1based <= e + context for (s, e) in regions)
        except Exception:
            return True, []  # safe default: assume in-region so gate still refuses
        else:
            return in_region, regions

    @staticmethod
    def _indentation_hint(content: str, lineno_1based: int, msg: str) -> str:
        """Build a concrete indentation suggestion for a SyntaxError.

        Python's indentation messages are generic ("unexpected indent",
        "unindent does not match", "expected an indented block"). The LLM
        often retries 2-3 times guessing the right column because the message
        never states *how many* spaces are needed. This helper inspects the
        surrounding lines of ``content`` (the post-edit source) and produces a
        specific hint, e.g.::

            Indentation: line 122 has 8 leading spaces; nearby statements use 4.
            Reduce this line to 4 spaces.

        Returns ``""`` when no confident suggestion can be derived (the caller
        then omits the hint). All arithmetic is defensive against tabs,
        blank lines, and boundary indices.
        """
        try:
            lines = content.split("\n")
            if not (1 <= lineno_1based <= len(lines)):
                return ""
            err_idx = lineno_1based - 1
            err_raw = lines[err_idx]
            err_lead = len(err_raw) - len(err_raw.lstrip(" "))

            def _lead_of(idx):
                # Callers only pass indices from range(err_idx-1, ..., -1) and
                # range(err_idx+1, len(lines)) — both in bounds.
                if idx < 0 or idx >= len(lines):  # pragma: no cover
                    return None
                _l = lines[idx]
                stripped = _l.lstrip(" ")
                if not stripped or stripped.startswith("#"):
                    return None
                return len(_l) - len(stripped)

            prev_leads = []
            for j in range(err_idx - 1, -1, -1):
                lv = _lead_of(j)
                if lv is not None:
                    prev_leads.append((j + 1, lv))
                    if len(prev_leads) >= 4:
                        break
            next_leads = []
            for j in range(err_idx + 1, len(lines)):
                lv = _lead_of(j)
                if lv is not None:
                    next_leads.append((j + 1, lv))
                    if len(next_leads) >= 3:
                        break

            m_lower = (msg or "").lower()
            from collections import Counter

            if "unexpected indent" in m_lower:
                candidates = [lv for (_ln, lv) in (prev_leads + next_leads) if lv < err_lead]
                if not candidates:
                    return ""
                target = Counter(candidates).most_common(1)[0][0]
                return (
                    f"Indentation: line {lineno_1based} has {err_lead} leading "
                    f"spaces; nearby statements use {target}. Reduce this line "
                    f"to {target} spaces."
                )

            if "unindent does not match" in m_lower or "unexpected unindent" in m_lower:
                outer = sorted({lv for (_ln, lv) in prev_leads if lv < err_lead})
                if not outer:
                    return ""
                levels = ", ".join(f"{x} spaces" for x in outer)
                return (
                    f"Indentation: line {lineno_1based} dedents to {err_lead} "
                    f"spaces, but no enclosing block uses that width. Valid "
                    f"outer indentation level(s): {levels}."
                )

            if "expected an indented block" in m_lower:
                opener_idx = None
                for j in range(err_idx - 1, -1, -1):
                    _l = lines[j].rstrip()
                    if _l and not _l.startswith("#") and _l.endswith(":"):
                        opener_idx = j
                        break
                if opener_idx is None:
                    return ""
                opener_lead = _lead_of(opener_idx) or 0
                target = opener_lead + 4
                return (
                    f"Indentation: line {opener_idx + 1} ends with ':' and opens "
                    f"a block, so line {lineno_1based} (and the rest of its body) "
                    f"must be indented deeper — use {target} spaces (opener is "
                    f"at {opener_lead})."
                )

        # All operations above are bounds-checked; nothing in the body raises.
        except (IndexError, ValueError, TypeError):  # pragma: no cover
            return ""
        else:
            return ""

    @staticmethod
    def _structural_imbalance_hint(msg: str) -> str:
        """Turn a structural SyntaxError message into an actionable hint.

        Python reports structural imbalances ("expected 'except' or 'finally'",
        "'(' was never closed", ...) on the line where the parser GIVES UP, not
        the line that opened the unbalanced construct. The generic diagnosis
        above (indentation/cascade) rarely names WHICH structural token got
        dropped, so the LLM has to re-read the file to find it. This helper maps
        the exact error substring ast.parse already raised to a concise hint
        naming the missing element. No heuristic guessing is performed, so false
        positives are impossible — the hint only fires when Python itself
        pinpointed the missing token.

        Returns ``""`` when ``msg`` does not match a known structural pattern;
        the caller then omits the hint and falls back to indentation/cascade.
        """
        m_lower = (msg or "").lower()
        if "expected 'except' or 'finally'" in m_lower or "has no 'except' or 'finally'" in m_lower:
            return (
                "Structural imbalance: new_string opens a `try:` block but is "
                "missing its matching `except`/`finally` clause — a truncated "
                "replacement block is the usual cause. Include the complete "
                "try/except/finally structure in new_string."
            )
        if "'(' was never closed" in m_lower:
            return (
                "Structural imbalance: an opening `(` is never closed — "
                "new_string likely dropped the closing parenthesis. Balance "
                "all parentheses in new_string."
            )
        if "'[' was never closed" in m_lower:
            return (
                "Structural imbalance: an opening `[` is never closed — new_string likely dropped the closing bracket."
            )
        if "'{' was never closed" in m_lower:
            return "Structural imbalance: an opening `{` is never closed — new_string likely dropped the closing brace."
        if "unexpected eof while parsing" in m_lower:
            return (
                "Structural imbalance: the source ends while a bracket or block "
                "is still open — new_string likely truncated the closing token(s)."
            )
        return ""

    def _apply_scoped_replacement(self, content, file_path, old_string, new_string, scope):
        """Scope-restricted old_string->new_string replacement.

        Uniqueness is measured WITHIN the (start, end) 1-based inclusive line
        range only. Occurrences OUTSIDE the range are ignored. The replacement
        is a POSITION-BASED splice (not content.replace(...,1)) so the correct
        in-scope occurrence is replaced even when an identical block exists
        earlier out-of-scope — and critically, this holds for BOTH the exact
        and the fallback (whitespace/indent/unicode-tolerant) matching paths.

        Bug history: the fallback path previously stored ``_char_pos = None``
        and fell back to ``content.replace(resolved, ..., 1)``. But when the
        fallback matched 2+ sites, ``resolved == old_string`` (the caller's
        non-matching original), so that replace was a NO-OP yet returned
        ``{"ok": True}`` — a silent success with an unmodified file. The fix
        stores the precise (char_pos, recon) from the fallback's line index so
        the splice is always position-based, mirroring the replace_all path.
        """
        scope_start, scope_end = scope
        if not old_string.strip():
            return {
                "ok": False,
                "error": "old_string is empty or whitespace only; cannot perform a meaningful replacement.",
                "metadata": {},
            }

        _resolved, total_count, fallback_matches, _orig_split = self._resolve_with_fallback(content, old_string)
        if _orig_split is None:
            _orig_split = []

        in_scope_matches = []
        if fallback_matches is not None:
            # Build per-line char offsets so we can splice the EXACT in-scope
            # site. _orig_split uses keepends=True, so offset[i] is the char
            # position of line i (0-based). Mirrors the replace_all path.
            _offset_by_line = [0]
            for _l in _orig_split:
                _offset_by_line.append(_offset_by_line[-1] + len(_l))
            for _m_idx, _m_recon in fallback_matches:
                _start_line = _m_idx + 1
                _n = len(_m_recon.splitlines()) or 1
                if scope_start <= _start_line <= scope_end:
                    _pos = _offset_by_line[_m_idx]
                    in_scope_matches.append((_start_line, _start_line + _n - 1, _pos, _m_recon))
        else:
            _search = 0
            _sl = len(old_string)
            while True:
                _pos = content.find(old_string, _search)
                if _pos < 0:
                    break
                _start_line = content[:_pos].count("\n") + 1
                if scope_start <= _start_line <= scope_end:
                    in_scope_matches.append((_start_line, _start_line + old_string.count("\n"), _pos, old_string))
                _search = _pos + _sl

        in_scope_count = len(in_scope_matches)

        if in_scope_count == 0:
            if total_count == 0:
                _hint = self._near_match_hint(content, old_string)
                _raw = self._raw_repr(old_string)
                return {
                    "ok": False,
                    "error": (
                        f"old_string not found in {file_path}\n{_raw}{_hint}\nTo fix this, re-read the file and include 2-3 lines of surrounding context as old_string."
                    ),
                    "metadata": {
                        "matched": False,
                        "near_match": bool(_hint),
                        "failure_class": "search_string_mismatch",
                    },
                }
            return {
                "ok": False,
                "error": (
                    f"old_string not found WITHIN scope L{scope_start}-{scope_end} in {file_path}, but {total_count} occurrence(s) exist OUTSIDE the scope. Adjust scope_start_line/scope_end_line to cover the intended occurrence."
                ),
                "metadata": {
                    "matched": False,
                    "in_scope_count": 0,
                    "out_of_scope_count": total_count,
                    "scope": list(scope),
                    "failure_class": "search_string_mismatch",
                },
            }

        if in_scope_count > 1:
            return {
                "ok": False,
                "error": (
                    f"Found {in_scope_count} occurrences of old_string WITHIN scope L{scope_start}-{scope_end} in {file_path}. Narrow the scope or make old_string more unique (include 2-3 lines of context)."
                ),
                "metadata": {
                    "matched": False,
                    "in_scope_count": in_scope_count,
                    "scope": list(scope),
                    "failure_class": "search_string_mismatch",
                },
            }

        _ms, _me, _char_pos, _recon = in_scope_matches[0]
        # Position-based splice using the char offset we recorded. For the
        # fallback path, reindent new_string to the match's actual indentation
        # (same as replace_all); for the exact path _recon == old_string so
        # _reindent_to_match is a no-op.
        _reindented_new = _reindent_to_match(new_string, _recon, file_unit=_detect_file_unit(content))
        new_content = content[:_char_pos] + _reindented_new + content[_char_pos + len(_recon) :]
        return {
            "ok": True,
            "new_content": new_content,
            "occurrences": 1,
            "high_count_warning": "",
            "match_line": _ms,
            "match_indent": _leading_indent_width(_recon),
            "reindent_applied": (_reindented_new != new_string),
        }

    def _apply_one_edit_text(
        self,
        content: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
        scope=None,
    ) -> dict[str, Any]:
        """Apply ONE old_string→new_string replacement to ``content``.

        Pure transformation — no disk I/O, no encoding, no syntax gate.
        Used by both the single-edit path and the batch (``edits``) path so
        both share identical matching/reindent/disambiguation behaviour.

        Returns a dict:
          * failure: ``{"ok": False, "error": str, "metadata": dict}``
          * success: ``{"ok": True, "new_content": str, "occurrences": int,
                        "high_count_warning": str}``

        ``scope`` — when not None, a ``(start, end)`` 1-based inclusive line
        range. Matching & uniqueness are restricted to that range: occurrences
        OUTSIDE it are ignored. This is a position-based splice (not
        ``content.replace(..., 1)``) so the correct in-scope occurrence is
        replaced even when an earlier identical block exists out-of-scope.
        Mutually exclusive with ``replace_all`` (validated by the caller).
        """
        _max_replace_all_matches = 20

        # ── Scoped replacement: restrict matching to a line range ──
        # Early-return branch: when scope is set we delegate to a dedicated
        # helper. The scope=None path below is COMPLETELY UNTOUCHED, so no
        # existing caller can regress.
        if scope is not None:
            return self._apply_scoped_replacement(content, file_path, old_string, new_string, scope)

        # No minimum-length gate: uniqueness is measured by occurrence count
        # below (count==0 → not found; count>1 → not unique; count==1 → safe).
        # Length is a poor proxy for uniqueness ("if __name__" is 13 chars but
        # appears dozens of times; a rare constant can be 4 chars but unique).
        # The ONLY hard rejection here is an empty/whitespace old_string:
        # content.count("") returns len(content)+1, so it would pass the exact
        # match path and content.replace("", x, 1) would prepend x nonsensically.
        # NOTE: the single-edit path also guards this in _tool_edit_text, but
        # the batch path does NOT pre-check each edit's old_string, so this is
        # the authoritative guard for batch edits.
        if not old_string.strip():
            return {
                "ok": False,
                "error": ("old_string is empty or whitespace only; cannot perform a meaningful replacement."),
                "metadata": {},
            }

        # ── Resolve old_string via fallback matching ──
        _orig_old_string = old_string
        _orig_new_string = new_string
        old_string, count, fallback_matches, _orig_split = self._resolve_with_fallback(content, old_string)
        # Contract: fallback_matches non-None implies _orig_split non-None; exact
        # match path returns None but no fallback code touches it there. Bind to
        # [] (not None) so all later fallback branches see a list.
        if _orig_split is None:
            _orig_split = []

        # ── Re-indent new_string when fallback resolved with diff indent ──
        _reindent_applied = False
        if count > 0 and old_string != _orig_old_string:
            new_string = _reindent_to_match(new_string, old_string, file_unit=_detect_file_unit(content))
            _reindent_applied = new_string != _orig_new_string

        _high_count_warning = ""
        if not replace_all:
            if count == 0:
                _hint = self._near_match_hint(content, old_string)
                _raw = self._raw_repr(old_string)
                if not _hint and _raw:
                    _extra = (
                        "\nNo close match found in the file. The old_string you sent "
                        "may be completely different from the actual file content, or "
                        "the file has changed since you last read it.\n"
                    )
                else:
                    _extra = ""
                return {
                    "ok": False,
                    "error": (
                        f"old_string not found in {file_path}\n{_raw}{_hint}{_extra}\n"
                        "To fix this, re-read the file and include 2-3 lines of "
                        "surrounding context as old_string (not just the line you "
                        "want to change)."
                    ),
                    "metadata": {
                        "matched": False,
                        "near_match": bool(_hint),
                        "failure_class": "search_string_mismatch",
                    },
                }
            if count > 1:
                # Build disambiguation context for each occurrence
                if fallback_matches is not None:
                    _contexts = []
                    _n_lines = len(old_string.splitlines())
                    for _mi, (_m_idx, _m_recon) in enumerate(fallback_matches):
                        _start_line = max(0, _m_idx - 2)
                        _end_line = min(len(_orig_split), _m_idx + _n_lines + 2)
                        _snippet = "".join(_orig_split[_start_line:_end_line]).rstrip()
                        _contexts.append(f"  [match {_mi + 1}] around line {_m_idx + 1}:\n{_snippet}")
                else:
                    lines = content.splitlines(keepends=True)
                    _contexts = []
                    _search_start = 0
                    for _occ_idx in range(count):
                        _pos = content.index(old_string, _search_start)
                        _line_no = content[:_pos].count("\n")
                        _start_line = max(0, _line_no - 2)
                        _end_line = min(len(lines), _line_no + 3)
                        _snippet = "".join(lines[_start_line:_end_line]).rstrip()
                        _contexts.append(f"  [match {_occ_idx + 1}] around line {_line_no + 1}:\n{_snippet}")
                        _search_start = _pos + len(old_string)
                return {
                    "ok": False,
                    "error": (
                        f"Found {count} occurrences of old_string in {file_path}. "
                        f"Make old_string more unique (include 2-3 lines of surrounding context).\n"
                        + "\n---\n".join(_contexts)
                    ),
                    "metadata": {"occurrences": count, "matched": False, "failure_class": "search_string_mismatch"},
                }
            if fallback_matches is not None:
                # Position-based splice: the fallback matched a specific LINE, but
                # content.replace(resolved, ..., 1) replaces the first SUBSTRING
                # occurrence, which may sit inside an earlier longer line that the
                # line-based fallback did NOT match — a silent wrong-site edit that
                # still parses as valid Python (so the syntax gate can't catch it).
                # Mirror _apply_scoped_replacement: splice at the matched line's char
                # offset. (_reindent_to_match is idempotent, so re-applying it here to
                # the already-reindented new_string from the earlier `_reindent_to_match` is a no-op.)
                _offset_by_line = [0]
                for _l in _orig_split:
                    _offset_by_line.append(_offset_by_line[-1] + len(_l))
                _m_idx, _m_recon = fallback_matches[0]
                _first_pos = _offset_by_line[_m_idx]
                _reindented_new = _reindent_to_match(new_string, _m_recon, file_unit=_detect_file_unit(content))
                new_content = content[:_first_pos] + _reindented_new + content[_first_pos + len(_m_recon) :]
                _match_line = _m_idx + 1
                _match_indent = _leading_indent_width(_m_recon)
            else:
                new_content = content.replace(old_string, new_string, 1)
                # ── Capture edit location for Enot/inside metadata ──
                _first_pos = content.find(old_string)
                _match_line = content[:_first_pos].count("\n") + 1 if _first_pos >= 0 else 0
                _match_indent = _leading_indent_width(old_string)
            _occurrences_replaced = 1
        else:
            if count > _max_replace_all_matches:
                _high_count_warning = (
                    f" ⚠️ old_string matched {count} times "
                    f"(max recommended: {_max_replace_all_matches}). "
                    f"Use a more specific old_string if this was unintentional."
                )
                logger.warning(
                    "%s [REPLACE_ALL_HIGH_COUNT] in %s",
                    _high_count_warning,
                    file_path,
                )
            if count == 0:
                _hint = self._near_match_hint(content, old_string)
                _raw = self._raw_repr(old_string)
                return {
                    "ok": False,
                    "error": (
                        f"old_string not found in {file_path}\n{_raw}{_hint}\n"
                        "To fix this, re-read the file and include 2-3 lines of "
                        "surrounding context as old_string (not just the line you "
                        "want to change)."
                    ),
                    "metadata": {
                        "matched": False,
                        "near_match": bool(_hint),
                        "failure_class": "search_string_mismatch",
                    },
                }
            # ── replace_all: when the fallback matched (count == 1 OR > 1),
            #    old_string may not actually exist in the file verbatim — for
            #    count == 1 it was reassigned to the reconstructed line text, which
            #    can appear as a substring of an earlier longer line that the
            #    line-based fallback did NOT match, so content.replace() would edit
            #    the wrong site (same wrong-site bug as the single-edit path). Use
            #    fallback_matches line indices + _orig_split offsets for a precise
            #    position-based splice in ALL fallback cases. ──
            if fallback_matches is not None:
                _repl = content
                _offset_by_line = [0]
                for _l in _orig_split:
                    _offset_by_line.append(_offset_by_line[-1] + len(_l))
                for _m_idx, _m_recon in reversed(fallback_matches):
                    _pos = _offset_by_line[_m_idx]
                    # Reindent new_string to match this particular match's indent
                    _reindented_new = _reindent_to_match(new_string, _m_recon, file_unit=_detect_file_unit(content))
                    _repl = _repl[:_pos] + _reindented_new + _repl[_pos + len(_m_recon) :]
                new_content = _repl
                # First match location for metadata
                _match_line = (fallback_matches[0][0] + 1) if fallback_matches else 0
                _match_indent = _leading_indent_width(fallback_matches[0][1]) if fallback_matches else 0
            else:
                new_content = content.replace(old_string, new_string)
                _first_pos = content.find(old_string)
                _match_line = content[:_first_pos].count("\n") + 1 if _first_pos >= 0 else 0
                _match_indent = _leading_indent_width(old_string)
            _occurrences_replaced = count

        return {
            "ok": True,
            "new_content": new_content,
            "occurrences": _occurrences_replaced,
            "high_count_warning": _high_count_warning,
            "match_line": _match_line,
            "match_indent": _match_indent,
            "reindent_applied": _reindent_applied,
        }

    def _tool_edit_text(self, args: dict[str, Any], *, _reread_retry: bool = False) -> ToolResult:
        """Edit a file by replacing exact strings — mirrors Claude Code's Edit tool.

        ``_reread_retry`` is an internal recursion guard (private): when a
        context-mismatch failure detects the file changed between our entry
        read and the failure, the whole edit list is retried ONCE against a
        fresh disk read. The flag suppresses further retries so the recursion
        is strictly depth-1 (no retry loops).


        Pure string replacement with two safety nets, but NO disk rollback:

        * **Blocking syntax gate** — for Python, ``compile()`` runs in-memory on
          the post-edit content *before* any byte touches disk; if the original
          file parsed and the edit would break parsing, the write is refused
          (file left untouched). This catches the classic indent-mismatch case.
        * **Non-blocking semantic diagnostics** — after a successful write,
          pyright/tsc/go diagnostics (type/undefined-name/import issues) are
          collected and surfaced as ``metadata.syntax_check`` for LLM
          self-healing, mirroring apply_patch/edit_file/modify_symbol/anchor_edit.

        Because there is no rollback (unlike apply_patch/edit_file), the syntax
        gate is the only write-time safety net — semantic findings are advisory.

        Two modes (mutually exclusive):

        1. **Single** (default): replace one ``old_string``→``new_string``,
           with optional ``replace_all``.

        2. **Batch (MultiEdit)**: pass ``edits`` — a list of objects
           ``[{"old_string": ..., "new_string": ..., "replace_all"?: false}, ...]``.
           Edits apply in order; each later edit sees the result of earlier ones.
           The whole batch is **atomic**: if any edit fails to match, the file is
           left untouched and the failing edit's index + error is returned with
           ``partial_failure`` metadata. The file is written exactly once.
           Use batch mode to make several unrelated substitutions in a single
           tool call instead of N round-trips.
        """
        import time as _time

        start_time = _time.monotonic()
        # --- Fix ①: Recover args from truncated streaming JSON ---
        args = self._recover_args_from_raw(args, ("file_path", "old_string", "new_string", "edits"))
        file_path = (args.get("file_path") or "").strip()

        if not file_path:
            return self._make_result(ok=False, error="file_path is required", execution_time=0)

        # Repo-boundary check. `confine=True` because _secure_path's contract is
        # that unrestricted_read is a READ capability and "writes can never
        # escape repo_root even on a trusted CLI" — but edit_text had no check at
        # all, so `../outside/f` and absolute paths both wrote through, at the
        # DEFAULT config (unrestricted_read=False), which is the webapp's, where
        # that same docstring notes repo_root is attacker-controlled. Mirrors
        # _tool_modify_symbol / _tool_edit_ast, which were already guarded.
        # The RESULT is reused below as `_norm`: it is fully resolved, so a
        # symlink inside the repo stays a symlink (the atomic write lands on
        # its target), mirroring create_file — a non-resolved path would make
        # os.replace in the atomic funnel replace the symlink with a regular
        # file.
        _secured = self._secure_path(file_path, confine=True)
        if _secured is None:
            return self._make_result(
                ok=False,
                error=f"Path blocked (outside repo): {file_path}",
                execution_time=0,
            )

        # F1 cross-process edit-lease guard.
        _lease_refused = self._refuse_foreign_leased([file_path], start_time)
        if _lease_refused is not None:
            return _lease_refused

        # ── Determine mode: batch (edits) vs single ──
        raw_edits = args.get("edits")
        is_batch = isinstance(raw_edits, list) and len(raw_edits) > 0
        # Reject ambiguous mixed-mode calls up front.
        if is_batch and (args.get("old_string") or args.get("new_string")):
            return self._make_result(
                ok=False,
                error=(
                    "Cannot mix 'edits' (batch mode) with 'old_string'/'new_string' "
                    "(single mode). Use one mode or the other."
                ),
                execution_time=0,
            )

        # ── scope_start_line / scope_end_line: range-restricted matching ──
        # When provided, uniqueness is measured WITHIN the [start, end] line
        # range only — occurrences outside it are ignored. This lets edit_text
        # target one of several identical blocks. Validation rules:
        #   * both must be provided together (one without the other is rejected)
        #   * start <= end
        #   * mutually exclusive with replace_all
        # The scope applies per-edit in batch mode (each edit may carry its own).
        def _parse_scope(d, allow_replace_all_field=True):
            """Extract & validate a (start, end) 1-based scope tuple or None.

            Returns (scope_tuple_or_None, error_str_or_None). On error the
            caller must abort with the message.
            """
            _ssl = d.get("scope_start_line")
            _sel = d.get("scope_end_line")
            _has_start = _ssl is not None
            _has_end = _sel is not None
            if _has_start != _has_end:
                return None, ("scope_start_line and scope_end_line must be provided together (got only one of them).")
            if not _has_start:
                return None, None
            try:
                _s = int(_ssl)
                _e = int(_sel)
            except (TypeError, ValueError):
                return None, (f"scope_start_line/scope_end_line must be integers (got start={_ssl!r}, end={_sel!r}).")
            if _s > _e:
                return None, (f"scope_start_line ({_s}) must be <= scope_end_line ({_e}).")
            if _s < 1:
                return None, "scope_start_line must be >= 1."
            if allow_replace_all_field and d.get("replace_all"):
                return None, (
                    "scope_start_line/scope_end_line cannot be combined with "
                    "replace_all (scope targets a single occurrence; replace_all "
                    "targets all)."
                )
            return (_s, _e), None

        if is_batch:
            edits = []
            assert isinstance(raw_edits, list)  # is_batch guarantees this
            for i, e in enumerate(raw_edits):
                if not isinstance(e, dict):
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}] must be an object with old_string/new_string",
                        execution_time=0,
                    )
                _old = e.get("old_string")
                _new = e.get("new_string")
                if _old is None:
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}] is missing old_string",
                        execution_time=0,
                    )
                if _new is None:
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}] is missing new_string",
                        execution_time=0,
                    )
                _e_scope, _e_scope_err = _parse_scope(e)
                if _e_scope_err is not None:
                    return self._make_result(
                        ok=False,
                        error=f"edits[{i}]: {_e_scope_err}",
                        execution_time=0,
                    )
                edits.append(
                    {
                        "old_string": _old,
                        "new_string": _new,
                        "replace_all": bool(e.get("replace_all", False)),
                        "scope": _e_scope,
                    }
                )
        else:
            old_string = args.get("old_string") or ""
            new_string = args.get("new_string") or ""
            replace_all = args.get("replace_all", False)
            if not old_string:
                return self._make_result(ok=False, error="old_string is required", execution_time=0)
            _single_scope, _single_scope_err = _parse_scope(args)
            if _single_scope_err is not None:
                return self._make_result(ok=False, error=_single_scope_err, execution_time=0)
            edits = [
                {"old_string": old_string, "new_string": new_string, "replace_all": replace_all, "scope": _single_scope}
            ]

        # Resolved path from the confine check above (symlink-preserving, bias-
        # corrected) — same as create_file. Never re-derive from repo_root: an
        # unresolved path would let the atomic funnel's os.replace replace a
        # repo-internal symlink with a regular file.
        _norm = _secured

        if not _norm.exists():
            return self._make_result(
                ok=False, error=f"File not found: {_norm}{self._suggest_missing_paths(file_path)}", execution_time=0
            )

        # Strict UTF-8 first, then latin-1. latin-1 decodes ANY byte sequence
        # losslessly (1:1 byte↔char), so untouched regions round-trip exactly
        # when written back with the SAME encoding. The previous
        # errors="replace" fallback baked U+FFFD over every undecodable byte
        # and then rewrote the whole file as UTF-8 — silently corrupting
        # regions far from the edit.
        content = None
        _read_encoding = "utf-8"
        for _enc in ("utf-8", "latin-1"):
            with contextlib.suppress(UnicodeDecodeError, UnicodeError):
                content = _norm.read_text(encoding=_enc)
                _read_encoding = _enc
                break
        # latin-1 decodes every byte sequence, so content is always set.
        if content is None:  # pragma: no cover
            return self._make_result(
                ok=False, error=f"Failed to read {file_path}: unsupported encoding", execution_time=0
            )

        # ── Apply all edits in-memory. Atomic: a failing edit aborts ALL ──
        _cur_content = content
        _total_occurrences = 0
        _high_count_warnings = []
        # Capture edit-site location from the FIRST edit (for single-mode Enot/inside
        # metadata: matched_line / matched_indent / reindent_applied). In batch
        # mode only the first edit's location is surfaced — batch callers get
        # per-edit detail via the diff, not metadata.
        _first_match_line = 0
        _first_match_indent = 0
        _first_reindent_applied = False
        for i, e in enumerate(edits):
            _res = self._apply_one_edit_text(
                _cur_content,
                file_path,
                e["old_string"],
                e["new_string"],
                e["replace_all"],
                scope=e.get("scope"),
            )
            if not _res["ok"]:
                # ── Auto re-read + bounded retry for CONTEXT-MISMATCH ──────
                # A search_string_mismatch means old_string did not match the
                # content we read at entry. The SAME args can only succeed if
                # the file changed between that read and now (a parallel
                # editor / another session on the same checkout), so before
                # surfacing the failure we re-read the file from disk once:
                #   * fresh != entry content → the file moved under us →
                #     retry the whole edit list ONCE against the fresh content
                #     (bounded: _reread_retry=True suppresses further retries)
                #   * fresh == entry content → unchanged; a retry would be
                #     deterministic and pointless, so fall through to the
                #     normal error path below.
                _fc = _res.get("metadata", {}).get("failure_class")
                if not _reread_retry and _fc == "search_string_mismatch" and _norm.exists():
                    _fresh_content = None
                    for _enc2 in ("utf-8", "latin-1"):
                        with contextlib.suppress(UnicodeDecodeError, UnicodeError):
                            _fresh_content = _norm.read_text(encoding=_enc2)
                            break
                    if _fresh_content is not None and _fresh_content != content:
                        _retry_args = dict(args)
                        _retry_result = self._tool_edit_text(_retry_args, _reread_retry=True)
                        _retry_meta = dict(_retry_result.metadata or {})
                        _retry_meta["reread_retried"] = True
                        if _retry_result.ok:
                            _retry_meta["reread_retry_success"] = True
                        _retry_result.metadata = _retry_meta
                        return _retry_result
                # Single mode: return the raw error verbatim (preserves the
                # exact message existing tests/callers depend on). Batch mode:
                # annotate with the failing edit's index.
                if is_batch:
                    _error = (
                        f"edit_text refused (file NOT modified): edit #{i + 1} "
                        f"(edits[{i}]) failed to match — no edits were applied "
                        f"(atomic batch).\n" + _res["error"]
                    )
                else:
                    _error = _res["error"]
                _meta = dict(_res.get("metadata", {}))
                if is_batch:
                    _meta["failed_edit_index"] = i
                    _meta["applied_edits"] = i  # edits before this one were computed in-memory only
                # ── Fresh-content snippet for dead-end mismatches ──
                # When near-match hinting found nothing, old_string shares no
                # similar line with the file - the file likely changed since
                # the agent last read it. Attach the current file head
                # (read_file-gutter style) so the LLM can craft a correct
                # old_string without an extra read round-trip.
                if _fc == "search_string_mismatch" and not _meta.get("near_match"):
                    _snip = self._current_file_head_snippet(content)
                    if _snip:
                        _error = _error.rstrip("\n") + "\n\n" + _snip
                        _meta["reread_snippet"] = True
                return self._make_result(
                    ok=False,
                    error=_error,
                    metadata=_meta,
                    execution_time=_time.monotonic() - start_time,
                )
            _cur_content = _res["new_content"]
            _total_occurrences += _res["occurrences"]
            if _res["high_count_warning"]:
                _high_count_warnings.append(_res["high_count_warning"])
            if i == 0:
                _first_match_line = _res.get("match_line", 0)
                _first_match_indent = _res.get("match_indent", 0)
                _first_reindent_applied = _res.get("reindent_applied", False)
        new_content = _cur_content

        # ── Syntax gate: refuse to write a .py edit that would BREAK parsing ──
        # edit_text does pure string replacement plus an indent-tolerant
        # fallback; a reindent that lands content at the wrong column (tab/space
        # mismatch, non-uniform old_string indent) silently corrupts the file —
        # the one write tool with no rollback.  Catch it here, in memory, BEFORE
        # touching disk.  Only gate when the ORIGINAL file parsed, so we never
        # block an edit that is fixing a pre-existing syntax error.
        if LanguageId.from_path(file_path) is LanguageId.PYTHON:
            _orig_parses = True
            try:
                compile_quiet(content, file_path, "exec")
            except SyntaxError:
                _orig_parses = False
            except Exception:
                _orig_parses = True  # non-SyntaxError → don't block on it
            if _orig_parses:
                try:
                    compile_quiet(new_content, file_path, "exec")
                except SyntaxError as _se:
                    # ── Scope-aware diagnosis ──
                    # Python reports an INDENTATION/structure error on the line
                    # where the parser NOTICES it, not necessarily the line whose
                    # indentation is wrong. We compare pre/post-edit content to
                    # tell whether ``_se.lineno`` was actually touched by this
                    # edit, so the LLM gets an actionable diagnosis instead of
                    # guessing whether it broke its own block or a cascade from
                    # elsewhere surfaced lines away.
                    _err_line = _se.lineno or 0
                    _in_edited, _regions = self._edited_line_regions(content, new_content, _err_line)
                    _region_str = ", ".join(f"L{s}-{e}" for (s, e) in _regions[:6]) or "unknown"
                    if _in_edited:
                        _diagnosis = (
                            "The error line is INSIDE the block you just edited "
                            f"(edited regions: {_region_str}). This is almost always "
                            "an indentation mistake in new_string — copy the exact "
                            "indentation (including comment lines) from the file. "
                            "Note Python may report the error one or two lines "
                            "BELOW the actually-misindented line."
                        )
                    else:
                        _diagnosis = (
                            "The error line was NOT directly edited (edited regions: "
                            f"{_region_str}), so this is likely a CASCADE: new_string "
                            "changed a block's structure (indentation, dedent, or an "
                            "unbalanced bracket/colon) whose effect the parser only "
                            "notices here. Check that new_string preserves the "
                            "surrounding block's indentation and that you didn't "
                            "accidentally drop a line or close a block early."
                        )
                    # ── Structural-imbalance hint ──
                    # ast.parse pinpoints the missing structural token
                    # ("expected 'except' or 'finally'", "'(' was never
                    # closed", ...). Surface it up-front so the LLM knows WHICH
                    # token got truncated — no file re-read required. Reuses the
                    # exact error Python raised, so no false positives.
                    _structure_hint = self._structural_imbalance_hint(_se.msg or "")
                    if _structure_hint:
                        _diagnosis = _structure_hint + " " + _diagnosis
                    # ── Concrete indentation hint ──
                    # Generic messages ("unexpected indent") never state the
                    # column count, so the LLM retries guessing. Compute the
                    # actual expected width from neighbouring lines.
                    _indent_hint = self._indentation_hint(new_content, _err_line, _se.msg or "")
                    if _indent_hint:
                        _diagnosis += " " + _indent_hint
                    return self._make_result(
                        ok=False,
                        error=(
                            f"edit_text refused (file NOT modified): the replacement would "
                            f"introduce a Python syntax error in {file_path}: "
                            f"{_se.msg} at line {_se.lineno}.\n" + _diagnosis
                        ),
                        metadata={
                            "syntax_error": str(_se),
                            "syntax_error_line": _err_line,
                            "error_in_edited_region": _in_edited,
                            "edited_regions": _regions,
                            "indentation_hint": _indent_hint,
                            "written": False,
                            "matched": True,
                            "failure_class": "syntax_invalid_after_edit",
                        },
                        execution_time=_time.monotonic() - start_time,
                    )

        # ── Language-neutral syntax gate (non-Python) ──────────────────────
        # edit_text is excluded from dispatch's snapshot+verify+rollback cycle
        # (tool_registry.py's `_write_snapshots` gate) because it has no rollback path, and the
        # Python ``compile()`` gate above only covers .py. For every OTHER
        # language we run the SAME provider.validate_syntax the dispatch path
        # uses — in memory, BEFORE writing — so a broken new_string never reaches
        # disk. The gate mirrors dispatch exactly: only GENUINE syntax errors
        # (FailureType.SYNTAX_ERROR) are refused; soft-fail errors that may
        # resolve cross-file (Go "undefined:", Java "cannot find symbol" →
        # UNKNOWN_SYMBOL) are KEPT, so edit_text is neither stricter nor looser
        # than apply_patch/edit_file for the same file+edit. Skip when the
        # ORIGINAL already failed parsing (we never block an edit fixing a
        # pre-existing error), matching the Python branch above.
        # P1: whether the non-Python gate below produced a CLEAN verdict on
        # new_content. The success return used to re-run validate_syntax via
        # _run_syntax_check_for_file (post-apply), spawning the backing tool
        # (npx tsc / go build / ...) a second time on every successful non-
        # Python edit. When this flag is set and the bytes on disk still match
        # what the gate validated, the post-check reuses the gate's verdict and
        # only runs the (separate) semantic check.
        _et_gate_ok = False
        _et_lang = LanguageId.from_path(file_path)
        if _et_lang is not LanguageId.PYTHON and _et_lang is not LanguageId.UNKNOWN:
            try:
                from ...languages import LanguageRegistry

                _et_provider = LanguageRegistry.instance().get(file_path)
            except Exception:
                _et_provider = None
            if _et_provider is not None and _et_provider.capabilities().has_syntax_validator:
                try:
                    _et_orig_ok = _et_provider.validate_syntax(file_path, content).ok
                except Exception:
                    _et_orig_ok = True  # validator crash → don't block the edit
                if _et_orig_ok:
                    try:
                        _et_new_val = _et_provider.validate_syntax(file_path, new_content)
                    except Exception:
                        _et_new_val = None
                    if _et_new_val is not None and _et_new_val.ok:
                        # Clean gate verdict on new_content — the post-check
                        # can reuse it after verifying disk bytes.
                        _et_gate_ok = True
                    elif _et_new_val is not None and not _et_new_val.ok:
                        _et_errs = _et_new_val.errors or []
                        if _et_errs:
                            _e0 = _et_errs[0]
                            _et_detail = f"{_e0.file}:{_e0.line}:{_e0.col}: {_e0.message}"
                            for _e in _et_errs[1:3]:
                                _et_detail += f"; L{_e.line}:{_e.col} {_e.message}"
                            if len(_et_errs) > 3:
                                _et_detail += f" (+{len(_et_errs) - 3} more syntax errors)"
                        else:
                            _et_detail = f"syntax error in {file_path}"
                        # Mirror dispatch soft-fail: keep cross-file-resolvable
                        # errors; refuse only genuine syntax errors.
                        if not self._should_soft_fail_verify(_et_detail, {file_path: content}):
                            return self._make_result(
                                ok=False,
                                error=(
                                    f"edit_text refused (file NOT modified): the "
                                    f"replacement would introduce a syntax error "
                                    f"in {file_path}: {_et_detail}"
                                ),
                                metadata={
                                    "syntax_error": _et_detail,
                                    "written": False,
                                    "matched": True,
                                    "failure_class": "syntax_invalid_after_edit",
                                },
                                execution_time=_time.monotonic() - start_time,
                            )
                        # soft-fail → fall through and write (dispatch keeps these)
        # Write back with the encoding the file was read with — re-encoding a
        # latin-1 file as UTF-8 would alter every non-ASCII byte. Encode BEFORE
        # any file I/O: the atomic bytes writer opens its temp file only after
        # encoding succeeds, so an encode failure never touches the target
        # (a plain open("wb") would truncate first). The atomic funnel
        # (atomic_write_bytes -> invalidate_for_written_path) keeps cached
        # consumers fresh, same as every other write tool.
        try:
            _encoded = new_content.encode(_read_encoding)
        except UnicodeEncodeError as e:
            return self._make_result(
                ok=False,
                error=(
                    f"Failed to write {file_path}: new_string contains characters "
                    f"not representable in the file's encoding ({_read_encoding}): {e}"
                ),
                execution_time=0,
            )
        try:
            atomic_write_bytes(str(_norm), _encoded)
        except Exception as e:
            return self._make_result(ok=False, error=f"Failed to write {file_path}: {e}", execution_time=0)

        _added = len(new_content) - len(content)
        _exec = _time.monotonic() - start_time
        # Track applied patch so agent_loop can detect successful writes
        self._append_applied_patch(f"edit_text:{file_path}:replace:{is_batch}")
        self._record_text_edit(file_path)
        _enc_detail = f" [enc: {_read_encoding}]" if _read_encoding != "utf-8" else ""
        _high_warn = "".join(_high_count_warnings)
        if is_batch:
            _content_msg = (
                f"Edited {file_path} ({len(edits)} edits, "
                f"{_total_occurrences} occurrence"
                f"{'s' if _total_occurrences != 1 else ''} replaced, "
                f"{_added:+d} chars{_enc_detail}){_high_warn} [{_exec:.1f}s]"
            )
        else:
            # Preserve the exact single-edit success wording.
            _count_detail = f"{_total_occurrences} occurrence{'s' if _total_occurrences > 1 else ''}"
            _content_msg = (
                f"Edited {file_path} (replaced {_count_detail}, "
                f"{_added:+d} chars{_enc_detail}){_high_warn} [{_exec:.1f}s]"
            )
        # Semantic feedback (non-blocking): pyright/tsc/go diagnostics for
        # type/undefined-name/import issues. Mirrors apply_patch/edit_file/
        # modify_symbol/anchor_edit. The blocking syntax gate above already
        # refused syntactically-broken Python edits; this catches semantic
        # problems (undefined names, type mismatches) and surfaces them for
        # LLM self-healing. _norm is on disk at this point.
        _meta: dict[str, Any] = {}
        if _high_count_warnings:
            _meta["high_count_warnings"] = _high_count_warnings
        # ── Enot/inside: surface the edit site's actual indentation ──
        # In single mode, matched_line/matched_indent let the LLM verify it hit
        # the intended location at the intended depth — paired with read_file's
        # │N│ gutter, this closes the indent-guessing loop. reindent_applied
        # warns that new_string's indentation was auto-corrected to old_string's
        # base indent (the LLM's original new_string indent did not match the
        # file). Batch mode omits per-edit detail (see comment in the loop).
        if not is_batch:
            if _first_match_line:
                _meta["matched_line"] = _first_match_line
            _meta["matched_indent"] = _first_match_indent
            if _first_reindent_applied:
                _meta["reindent_applied"] = True
        # P1: reuse the non-Python syntax gate's clean verdict (when it ran and the
        # disk bytes still match) instead of re-spawning the backing tool. The
        # semantic check below is never skipped — it is the part the gate did
        # not run and the reason this post-apply call still exists.
        _syn = self._run_syntax_check_for_file(
            str(_norm), reuse_gate_syntax=_et_gate_ok, gate_content=new_content if _et_gate_ok else None
        )
        if not _syn.get("skipped"):
            _meta["syntax_check"] = _syn
        return self._make_result(
            ok=True,
            content=_content_msg,
            metadata=_meta,
            execution_time=_exec,
        )

    def _tool_create_file(self, args: dict[str, Any]) -> ToolResult:
        """Create or overwrite a file with the given content.

        Creates parent directories automatically. Fails if the file already
        exists (use ``overwrite=True`` to replace).

        Reachable via the apply_patch create_file / multi-symbol fallbacks
        (``_try_apply_patch_create_file_fallback`` /
        ``_try_apply_patch_multi_symbol_fallback``), so this handler MUST carry
        the same safety every other write handler does — it was the lone
        exception, which made those fallback paths LESS safe than the main
        PatchEngine path they recover from:

        * **Repo confinement** — ``_secure_path(confine=True)`` rejects any path
          escaping repo_root (absolute paths, ``..`` traversal). Without it,
          ``Path(repo_root) / abs_path`` collapses to ``abs_path`` and the
          fallback could write outside the repo. ``unrestricted_read`` is a READ
          capability only — mirrors edit_text / modify_symbol / edit_ast /
          anchor_edit.
        * **Blocking syntax gate** — content is validated in memory BEFORE any
          byte touches disk: Python via ``compile()``; every other registered
          language via its provider's ``validate_syntax``, refusing genuine
          SYNTAX_ERROR and soft-failing cross-file-resolvable errors (mirrors
          dispatch's ``_should_soft_fail_verify``). New files have no
          pre-snapshot, so the dispatch-level ``_verify_after_write`` (which has
          nothing to compare against) can't catch a malformed creation — this
          in-handler gate does.
        * **Atomic write** — ``atomic_write_text`` (mkstemp + fsync +
          os.replace), so a crash / SIGKILL / disk-full never leaves a partial
          or empty file.
        """
        import time as _time

        start_time = _time.monotonic()
        args = self._recover_args_from_raw(args, ("path",))
        file_path = (args.get("path") or args.get("file_path") or "").strip()
        content = args.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        description = args.get("description", "")
        overwrite = bool(args.get("overwrite", False))

        if not file_path:
            # If __raw_arguments is present, the JSON was likely truncated during streaming
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(ok=False, error=f"path is required{_raw_hint}", execution_time=0)

        # ── Repo confinement (defense in depth — matches every other write handler) ──
        _secured = self._secure_path(file_path, confine=True)
        if _secured is None:
            return self._make_result(
                ok=False,
                error=f"Path blocked (outside repo): {file_path}",
                execution_time=0,
            )
        _norm = _secured

        _existed = _norm.exists()
        if _existed and not overwrite:
            return self._make_result(
                ok=False,
                error=f"File already exists: {file_path} (use overwrite=True to replace)",
                execution_time=0,
            )

        # F1 cross-process edit-lease guard (covers the freshly-created case too:
        # a live foreign lease means a parallel session created/edited this
        # path moments ago — overwriting it would clobber its WIP).
        _lease_refused = self._refuse_foreign_leased([file_path], start_time)
        if _lease_refused is not None:
            return _lease_refused

        # ── Blocking syntax gate (in memory, before disk) ──
        # Python: compile() catches parse errors only (no undefined-name cascade
        # at compile time → no soft-fail needed). Non-Python: provider
        # validate_syntax + _should_soft_fail_verify, mirroring edit_text/dispatch.
        # _gate_snapshots keys the pre-write content under the SAME path the
        # provider reports (file_path) so origin-skip matches; a None value (new
        # file) makes _should_soft_fail_verify skip the origin guard and apply
        # pure FailureType classification — exactly what dispatch does for a
        # _MISSING_SNAP new-file entry.
        _orig_content = None
        if _existed:
            try:
                _orig_content = _norm.read_text(encoding="utf-8", errors="replace")
            except Exception:
                _orig_content = None
        _gate_snapshots = {file_path: _orig_content}
        _gate_soft_failed = False

        _lang = LanguageId.from_path(file_path)
        if _lang is LanguageId.PYTHON:
            try:
                compile_quiet(content, file_path, "exec")
            except SyntaxError as _se:
                return self._make_result(
                    ok=False,
                    error=(
                        f"create_file refused (file NOT written): content has a Python "
                        f"syntax error in {file_path}: {_se.msg} at line {_se.lineno}"
                    ),
                    metadata={
                        "file_path": file_path,
                        "failure_class": "syntax_invalid_after_edit",
                        "written": False,
                    },
                    execution_time=_time.monotonic() - start_time,
                )
            # CPython 3.14 raises SyntaxError for every source-level malformation
            # (NUL bytes, too many nested parens, unterminated strings) — any
            # other exception here is environmental (MemoryError etc.).
            except Exception as _ce:  # pragma: no cover
                # CPython 3.14 raises SyntaxError for every source-level
                # malformation (NUL bytes, too many nested parens, unterminated
                # strings), so any non-SyntaxError exception here is
                # environmental (MemoryError etc.) and says nothing about the
                # content being malformed — the gate opens rather than refusing
                # a legitimate write. Logged because a silent skip here is
                # indistinguishable from a gate that ran and passed.
                logger.debug(
                    "create_file: Python syntax gate skipped for %s (%s: %s)",
                    file_path,
                    type(_ce).__name__,
                    _ce,
                )
        elif _lang is not LanguageId.UNKNOWN:
            try:
                from ...languages import LanguageRegistry

                _provider = LanguageRegistry.instance().get(file_path)
            except Exception:
                _provider = None
            if _provider is not None and _provider.capabilities().has_syntax_validator:
                try:
                    _val = _provider.validate_syntax(file_path, content)
                except Exception:
                    _val = None
                if _val is not None and not _val.ok:
                    _errs = _val.errors or []
                    if _errs:
                        _detail = f"{_errs[0].file}:{_errs[0].line}:{_errs[0].col}: {_errs[0].message}"
                        for _e in _errs[1:3]:
                            _detail += f"; L{_e.line}:{_e.col} {_e.message}"
                        if len(_errs) > 3:
                            _detail += f" (+{len(_errs) - 3} more syntax errors)"
                    else:
                        _detail = f"syntax error in {file_path}"
                    # Mirror dispatch soft-fail: keep cross-file-resolvable errors;
                    # refuse only genuine syntax errors.
                    if not self._should_soft_fail_verify(_detail, _gate_snapshots):
                        return self._make_result(
                            ok=False,
                            error=(
                                f"create_file refused (file NOT written): content would "
                                f"introduce a syntax error in {file_path}: {_detail}"
                            ),
                            metadata={
                                "file_path": file_path,
                                "syntax_error": _detail,
                                "failure_class": "syntax_invalid_after_edit",
                                "written": False,
                            },
                            execution_time=_time.monotonic() - start_time,
                        )
                    _gate_soft_failed = True  # soft-fail → fall through and write

        # ── Atomic write ──
        try:
            atomic_write_text(str(_norm), content)
        except Exception as e:
            return self._make_result(
                ok=False,
                error=f"Failed to create {file_path}: {e}",
                execution_time=0,
            )

        _exec = _time.monotonic() - start_time
        # F1: stake our lease so parallel sessions see this file as actively WIP
        # (create_file has no _record_text_edit call — its own acquire).
        self._acquire_edit_leases([file_path])
        _verb = "Overwrote" if _existed else "Created"
        _desc = f" ({description})" if description else ""
        _size = len(content)
        _meta: dict[str, Any] = {"file_path": file_path}
        if _gate_soft_failed:
            _meta["syntax_gate"] = "soft_fail"
        return self._make_result(
            ok=True,
            content=f"{_verb}: {file_path}{_desc} ({_size} chars) [{_exec:.1f}s]",
            metadata=_meta,
            execution_time=_exec,
        )

    def _extract_ops_from_raw(self, raw: str) -> list[dict[str, Any]]:
        """Try to extract ``operations`` list from truncated raw JSON string.

        Stream truncation can cut the JSON before the outer ``}``, leaving a
        complete ``"operations": [...]`` inside the partial string.  Extract
        the array via bracket matching instead of full JSON parsing.
        """
        import json as _json

        _m = re.search(r'"operations"\s*:\s*(\[)', raw)
        if not _m:
            return []
        _start = _m.start(1)
        _depth = 0
        _end = -1
        for _i, _c in enumerate(raw[_start:], start=_start):
            if _c == "[":
                _depth += 1
            elif _c == "]":
                _depth -= 1
                if _depth == 0:
                    _end = _i + 1
                    break
        if _end == -1:
            return []  # truncated inside the array — cannot recover
        with contextlib.suppress(_json.JSONDecodeError):
            _parsed = _json.loads(raw[_start:_end])
            if isinstance(_parsed, list):
                return _parsed
        return []

    def _recover_args_from_raw(
        self,
        args: dict[str, Any],
        required_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        """Recover args from __raw_arguments when required keys are missing.

        Delegates to :func:`external_llm.agent.write_targets.recover_args_from_raw`,
        which is the single source of truth. The implementation moved there
        because the write-safety gates (checkpoint, rollback snapshot, file
        lock) resolve their target paths BEFORE this handler runs: while the
        recovery lived only here, a tool call whose ``path`` existed solely
        inside ``__raw_arguments`` wrote normally but was invisible to all
        three, so the run silently got no Undo point and no file lock.
        """
        return _recover_args_from_raw(args, required_keys)

    @staticmethod
    def _try_repair_truncated_json(raw: str) -> dict[str, Any] | None:
        """Repair and parse a truncated JSON object string.

        Delegates to :func:`external_llm.agent.write_targets.try_repair_truncated_json`
        — see :meth:`_recover_args_from_raw` for why it lives there.
        """
        return _try_repair_truncated_json(raw)

    def _run_syntax_check_for_file(
        self, rel_or_abs_path: str, *, reuse_gate_syntax: bool = False, gate_content: str | None = None
    ) -> dict:
        """Run post-apply syntax validation for *rel_or_abs_path* if a provider exists.

        ``reuse_gate_syntax`` lets a caller (edit_text's non-Python success
        path) skip the post-apply syntax re-spawn when it already validated the
        written content in-memory via the same provider: pass
        ``gate_content=`` the validated content and this method compares disk
        bytes against it, reusing the clean verdict on match and falling back
        to a full ``validate_syntax`` run on mismatch.

        Always returns a dict.  ``skipped=True`` means no provider is registered
        for this file type — callers should omit the result from metadata rather
        than treating it as an error.

        When syntax is OK and the provider supports semantic validation
        (``has_semantic_validator``), an additional ``semantic_diagnostics``
        field is populated with type/undefined-name/import diagnostics collected
        by running the backing tool (pyright/tsc/go build) against the real
        project. These are **non-blocking** — surfaced for LLM self-healing.

        If the backing tool never ran — not installed, timed out, no project
        config — the result carries ``semantic_check_skipped`` with the reason
        INSTEAD of ``semantic_diagnostics``. The two must not be conflated: an
        empty diagnostics list is a clean verdict, and reporting one for a check
        that never happened tells the model the file was verified.
        """
        try:
            import os

            from ...languages.registry import LanguageRegistry

            abs_path = (
                rel_or_abs_path
                if os.path.isabs(rel_or_abs_path)
                else os.path.join(str(self.repo_root), rel_or_abs_path)
            )
            provider = LanguageRegistry.instance().get(abs_path)
            if provider is None:
                return {"ok": True, "skipped": True, "reason": "no_provider"}

            try:
                content, _ = read_text_with_encoding_fallback(abs_path)
            except OSError:
                return {"ok": True, "skipped": True, "reason": "file_read_error"}
            # read_text_with_encoding_fallback tries utf-8 then latin-1, and
            # latin-1 decodes every byte sequence — content is never None here.
            if content is None:  # pragma: no cover
                # Non-UTF-8 source that even latin-1 cannot decode.  Previously
                # UnicodeDecodeError escaped the OSError guard and was swallowed
                # by the outer except Exception → the gate reported a clean
                # "exception" skip.  A distinct reason makes the skip observable
                # (callers branch on `skipped` and would otherwise bypass the
                # syntax check AND the rollback silently).
                logger.warning(
                    "Post-apply syntax check skipped for %s: undecodable (utf-8/latin-1)",
                    abs_path,
                )
                return {"ok": True, "skipped": True, "reason": "decode_error"}

            # P1: when the caller already ran a clean non-Python syntax gate on
            # gate_content (the exact new_content that was about to be written),
            # and the bytes that actually landed on disk are identical to that
            # content, reuse the gate's verdict instead of re-spawning the
            # backing tool (npx tsc / go build / javac ...) a second time. The
            # disk comparison is the safety net: any drift (concurrent writer,
            # encoding round-trip mismatch) falls back to a fresh full run.
            if reuse_gate_syntax and gate_content is not None and gate_content == content:
                result = type("ReusedGateVerdict", (), {"ok": True, "language": provider.language_id(), "errors": []})()
            else:
                result = provider.validate_syntax(abs_path, content)
            out = {
                "ok": result.ok,
                "language": result.language.value if result.language else None,
                "errors": [{"line": e.line, "col": e.col, "message": e.message} for e in (result.errors or [])],
            }
            # Only run semantic check on syntactically-valid files to avoid
            # cascading-error noise from the backing tool.
            if result.ok and provider.capabilities().has_semantic_validator:
                # Inside an agent turn the check is coalesced to once per
                # (turn, file) and run at turn end against the FINAL content —
                # see ToolRegistry.begin_semantic_turn for the cost measurement
                # and for why it must be the last write, not the first.
                #
                # Deferring writes NO ``semantic_diagnostics`` key. That matters:
                # an empty list here is rendered by
                # ``agent_loop._append_semantic_diagnostics`` as "nothing to
                # report", so a deferred check would read as a clean one. The
                # marker below is what the turn pipeline matches on to fill in
                # the real result before the message reaches the model.
                _sem_key = os.path.normpath(os.path.abspath(abs_path))
                if self.defer_semantic_check(_sem_key, rel_or_abs_path):
                    out["semantic_deferred"] = True
                    out["semantic_deferred_path"] = _sem_key
                else:
                    try:
                        sem = provider.validate_semantics(abs_path)
                        if not getattr(sem, "checked", True):
                            # Nothing examined the file (toolchain missing,
                            # timed out, no project config). Reporting an empty
                            # diagnostics list here would read as "checked,
                            # clean" — see ToolRegistry.SemanticOutcome.
                            out["semantic_check_skipped"] = getattr(sem, "skip_reason", "") or "the checker did not run"
                        else:
                            out["semantic_diagnostics"] = [
                                {
                                    "file_path": abs_path,
                                    "line": e.line,
                                    "col": e.col,
                                    "message": e.message,
                                    "severity": getattr(e, "severity", "error"),
                                    "code": getattr(e, "code", ""),
                                }
                                for e in (sem.errors or [])
                            ]
                    except Exception as sem_exc:
                        logger.debug("Semantic check failed for %s: %s", abs_path, sem_exc)
                        out["semantic_check_skipped"] = "the semantic checker raised before reporting"
        except Exception as exc:
            logger.debug("Post-apply syntax check failed: %s", exc)
            return {"ok": True, "skipped": True, "reason": "exception"}
        else:
            return out

    def _norm_repo_rel(self, p: str) -> str:
        """Normalize a path (absolute or relative) to repo-root-relative form."""
        if not p:
            return ""
        rr = str(getattr(self, "_effective_repo_root", None) or getattr(self, "repo_root", ""))
        if rr and p.startswith(rr):
            p = p[len(rr) :]
        return p.lstrip("/")

    def _record_text_edit(self, file_path: str) -> None:
        """Record that a text-editing tool wrote ``file_path`` this session.

        Tracked so apply_patch can refuse to clobber a working-tree edit it cannot
        safely merge: apply_patch / diff_apply reconstructs hunk context from HEAD,
        and on a freshly-edited target PatchEngine uses skip_3way=True whose
        _rollback() reverts the working tree to HEAD — silently deleting the edit.
        See the session-edit guards in _tool_apply_patch and _apply_patch_text.
        """
        with contextlib.suppress(ValueError, AttributeError):  # path outside root / partial-mixin harness
            rel = self._norm_repo_rel(file_path)
            if rel:
                self._text_edited_files.add(rel)
        # F1: record the cross-process edit lease (fail-open; no-op without a
        # repo_root) so parallel asicode sessions see this file as actively WIP.
        try:
            self._acquire_edit_leases([file_path])
        except Exception:  # partial-mixin harness without the patch mixin
            logger.debug(
                "<module>::WriteToolsEditMixin::_record_text_edit:1 edit-lease acquire suppressed",
                exc_info=True,
            )

    def _tool_modify_symbol(self, args: dict[str, Any]) -> ToolResult:
        """Modify a symbol in a file deterministically — no LLM call.

        Fallback chain: AST precise (Python) → surgical edit (any language) → text replacement.
        Each fallback is deterministic — no Developer LLM calls.
        """
        from external_llm.agent.symbol_modify_tool import modify_symbol as _do_modify

        args = self._recover_args_from_raw(args, ("file_path", "symbol", "code"))
        file_path = str(args.get("file_path", "")).strip()
        symbol = str(args.get("symbol", "")).strip()
        code = str(args.get("code", "")).strip("\n")
        dry_run = args.get("dry_run", False)

        if not file_path:
            return self._make_result(ok=False, content="", error="'file_path' is required")
        if not symbol:
            return self._make_result(ok=False, content="", error="'symbol' is required")
        if not code:
            return self._make_result(ok=False, content="", error="'code' is required")

        sec = self._secure_path(file_path, confine=True)
        if sec is None:
            return self._make_result(ok=False, content="", error=f"Path traversal blocked: {file_path}")
        abs_path = str(sec)
        if not os.path.isfile(abs_path):
            return self._make_result(
                ok=False, content="", error=f"File not found: {file_path}{self._suggest_missing_paths(file_path)}"
            )

        # F1 cross-process edit-lease guard.
        _lease_refused = self._refuse_foreign_leased([abs_path])
        if _lease_refused is not None:
            return _lease_refused

        rel_path = os.path.relpath(abs_path, self.repo_root)

        # dry_run snapshot: _do_modify writes the file on every success path,
        # so a preview REQUIRES a snapshot to restore from. Refuse the dry run
        # if the snapshot cannot be taken — otherwise the "preview" would
        # silently mutate the file.
        original_source: str | None = None
        if dry_run:
            try:
                with open(abs_path, encoding="utf-8") as f:
                    original_source = f.read()
            except Exception as e:
                return self._make_result(
                    ok=False,
                    content="",
                    error=f"[DRY RUN] cannot snapshot {rel_path} for preview: {e}",
                )

        success, diff_or_error, new_content = _do_modify(abs_path, symbol, code, repo_root=self.repo_root)

        if dry_run:
            # Restore the pre-edit content — _do_modify already wrote the file.
            if success and original_source is not None:
                try:
                    atomic_write_text(abs_path, original_source)
                except Exception as e:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"[DRY RUN] modify succeeded but restoring {rel_path} failed: {e} "
                            f"— the file HAS BEEN MODIFIED on disk"
                        ),
                        metadata={"file_path": rel_path, "symbol": symbol, "dry_run": True, "restore_failed": True},
                    )
            if success:
                return self._make_result(
                    ok=True,
                    content=(f"[DRY RUN] modify_symbol preview for {rel_path}@{symbol}\nDiff:\n{diff_or_error}"),
                    metadata={
                        "file_path": rel_path,
                        "symbol": symbol,
                        "dry_run": True,
                        "diff_preview": diff_or_error[:25000] if diff_or_error else "",
                    },
                )
            preview = f"[DRY RUN] Preview for {rel_path}@{symbol} (apply skipped: {diff_or_error})\n"
            preview += f"New code:\n{code}"
            return self._make_result(
                ok=True,
                content=preview,
                metadata={"file_path": rel_path, "symbol": symbol, "dry_run": True, "preview_only": True},
            )

        if success:
            self._record_text_edit(rel_path)
            self._append_applied_patch(f"modify_symbol:{rel_path}:{symbol}")
            _meta = {
                "file_path": rel_path,
                "symbol": symbol,
                "diff_preview": diff_or_error[:25000] if diff_or_error else "",
                "changed": True,
            }
            # ── Enot/inside: surface the replaced symbol's definition indent ──
            # new_content is the post-edit file in memory (3rd tuple element of
            # _do_modify). Locate the symbol's def/decorator start line and
            # report its leading-whitespace column count, mirroring read_file's
            # │N│ gutter. This lets the LLM verify the replacement landed at the
            # intended nesting depth (esp. for body-only mode, where the LLM
            # must guess the body indent). Best-effort: any failure is swallowed.
            if new_content:
                with contextlib.suppress(IndexError, TypeError, ValueError):  # best-effort metadata enrichment
                    from external_llm.agent.symbol_modify_tool import _find_symbol_line_range as _find_range

                    _rng = _find_range(new_content, symbol, rel_path)
                    if _rng is not None:
                        _nc_lines = new_content.splitlines()
                        _def_idx = _rng[0]
                        if 0 <= _def_idx < len(_nc_lines):
                            _meta["symbol_def_line"] = _def_idx + 1
                            _meta["symbol_def_indent"] = _leading_indent_width(_nc_lines[_def_idx])
            # Semantic feedback (non-blocking): pyright/tsc/go diagnostics for
            # type/undefined-name/import issues. Mirrors apply_patch/edit_file.
            _syn = self._run_syntax_check_for_file(abs_path)
            if not _syn.get("skipped"):
                _meta["syntax_check"] = _syn
            return self._make_result(
                ok=True,
                content=(f"Modified symbol '{symbol}' in {rel_path}\nDiff:\n{diff_or_error}"),
                metadata=_meta,
            )
        return self._make_result(
            ok=False, content="", error=f"modify_symbol failed for {rel_path}@{symbol}: {diff_or_error}"
        )
