"""apply_patch-family write tool handlers (P2-2 split).

``WriteToolsPatchMixin`` — write_plan, apply_patch, apply_patch_text (git apply)
and anchor_edit. Split out of ``WriteToolsMixin`` in ``write_tools.py``;
recombined there via ``class WriteToolsMixin(WriteToolsPatchMixin, ...)``.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from diff_apply import _clean_diff, extract_touched_files_from_diff
from services.patch_helpers import normalize_patch_text

from ...common.atomic_io import atomic_write_text
from ...common.indent_utils import format_numbered_line
from ...languages import LanguageId
from ...languages._normalize import normalize_key
from ...languages.comment_syntax import comment_syntax_for
from ...patch_engine import PatchContext, PatchEngine
from .._shared_utils import (
    _net_bracket_delta,
    _scan_line_brackets_delta,
    _scan_to_line_state,
    compile_quiet,
    extract_files_from_patch,
)
from ..write_targets import normalize_plan
from .write_tools_core import (
    _check_block_introducer_nesting,
    _detect_enclosing_scope,
    _detect_fragment_duplication,
    _extract_truncated_op_path,
    _find_block_end_line,
    _repair_plan_json,
)

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

# P25-4: plan normalization / error-enrichment read whole target files to
# extract one line, a line window, or a short head. plan_compiler's stat gate
# (P24-2) bounds the APPLY path, but these helpers run BEFORE it — an
# unbounded read there defeated the gate downstream (a multi-hundred-MB target
# was fully materialised up to 4x per op during normalization). Everything
# below is streaming: O(needed) memory, not O(file).


def _iter_file_lines(fp: Path):
    """Yield lines of ``fp`` (UTF-8, errors=replace, ``\n`` stripped) without
    loading the file whole. ``\r\n`` is normalized by universal-newline mode,
    matching the previous ``read_text().splitlines()`` behavior."""
    with fp.open(encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            yield ln.rstrip("\n")


def _file_line_window(fp: Path, start: int, count: int) -> list[str]:
    """Lines ``[start, start+count)`` of ``fp`` (0-based) — O(count) memory."""
    return list(itertools.islice(_iter_file_lines(fp), start, start + count))


logger = logging.getLogger(__name__)


# ── Precompiled regexes used on the hot apply_patch path ─────────────────────
# Module-level so each call reuses the compiled pattern instead of recompiling.
_PATCH_PATH_PREFIX_RE = re.compile(r"^(?:a/|b/)")  # strip git diff a// b/ prefixes
_HUNK_HEADER_RE = re.compile(r"^@@ -([0-9]+),([0-9]+) \+([0-9]+),([0-9]+) @@")
_RE_LINE_NUMBER_PREFIX = re.compile(r"^\d+:\s?")


def _resolve_ast_anchor_line(
    anchor_ast_lineno: Any,
    lines: list[str],
    anchor_pattern: str | None,
) -> int | None:
    """Resolve a caller-supplied 1-indexed ``anchor_ast_lineno`` to a 0-indexed
    line, or None when absent/out of range (P5-3 extraction from
    ``_tool_anchor_edit`` — pure computation, no self)."""
    if not (isinstance(anchor_ast_lineno, int) and anchor_ast_lineno > 0):
        return None
    _candidate = anchor_ast_lineno - 1  # 1-indexed → 0-indexed
    if 0 <= _candidate < len(lines):
        logger.info(
            "[ANCHOR_AST] tool anchor_edit using caller-supplied line %d (pattern=%r bypassed)",
            anchor_ast_lineno,
            (anchor_pattern or "")[:40],
        )
        return _candidate
    logger.warning(
        "[ANCHOR_AST] tool anchor_edit line %d out of range (file=%d lines) — falling back to string search",
        anchor_ast_lineno,
        len(lines),
    )
    return None


class WriteToolsPatchMixin:
    """Patch-family write handlers: write_plan, apply_patch, anchor_edit."""

    # ── Host-class attributes (provided by ToolRegistry, not set here) ──
    # Class-level annotations give pyright the host contract WITHOUT runtime
    # assignment: ToolRegistry.__init__ owns the real values, so these are pure
    # typing scaffolding (no setattr, no __getattr__). Mirrors the docstring
    # contracts in the sibling mixins; keep the two in sync.
    repo_root: str
    _make_result: Any
    _effective_repo_root: Any
    _secure_path: Any
    _invalidate_cache_after_write: Any
    _run_syntax_check_for_file: Any
    _should_soft_fail_verify: Any
    _tool_create_file: Any
    _recover_args_from_raw: Any
    _suggest_missing_paths: Any
    _record_text_edit: Any
    _norm_repo_rel: Any
    _text_edited_files: set[str]
    _applied_patches: list[str]
    _patch_failure_snippet: Any

    # ── Plan normalizer class variables ─────────────────────────────── #
    _ACTION_TO_OP: ClassVar[dict[str, str]] = {
        "insert": "insert_after",
        "append": "insert_after",
        "prepend": "insert_before",
        "add": "insert_after",
        "replace": "replace_file",
        "edit": "edit_blocks",
        "modify": "edit_blocks",
        "update": "edit_blocks",
        "patch": "edit_blocks",
        "change": "edit_blocks",
        "create": "create_file",
        "write": "create_file",
        "new": "create_file",
        "overwrite": "replace_file",
        "rewrite": "replace_file",
        "delete_content": "edit_blocks",
    }

    _WRITE_PLAN_OP_TYPES = frozenset(
        {
            "create_file",
            "replace_file",
            "edit_blocks",
            "insert_after",
            "insert_before",
            "insert_after_line",
        }
    )

    _OP_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "createfile": "create_file",
        "replacefile": "replace_file",
        "replace": "replace_file",
        "editblocks": "edit_blocks",
        "insertafter": "insert_after",
        "insertbefore": "insert_before",
    }

    # NOTE: short/lowercase tokens that can legitimately appear in real code
    # (e.g. "..." Python ellipsis, "old_code"/"new_code" as actual identifiers,
    # "old text"/"new text"/"new code" inside docstrings) are excluded to avoid
    # rejecting valid edits.
    _PLACEHOLDER_BEFORE = frozenset(
        {
            "OLD TEXT",
            "ORIGINAL CODE",
            "EXISTING CODE",
            "CURRENT CODE",
            "YOUR CODE HERE",
            "REPLACE THIS",
            "...\n...",
            "old code",
            "existing code",
            "current content",
            "put old code here",
        }
    )
    _PLACEHOLDER_AFTER = frozenset(
        {
            "NEW TEXT",
            "UPDATED CODE",
            "NEW CODE",
            "REPLACEMENT CODE",
            "YOUR NEW CODE HERE",
            "updated content",
            "put new code here",
        }
    )

    # ── Path normalizer helpers ──────────────────────────────────────── #

    def _normalize_op_path(self, raw_path: str, repairs: list[str]) -> str:
        if not raw_path:
            return raw_path
        p = raw_path.replace("\\", "/")
        repo = str(self.repo_root).rstrip("/") + "/"
        if p.startswith(repo):
            rel = p[len(repo) :]
            repairs.append(f"abs path stripped repo prefix→{rel!r}")
            return rel
        if p.startswith("/"):
            rel = p.lstrip("/")
            repairs.append(f"abs path→rel: {rel!r}")
            return rel
        return p

    def _normalize_plan_op(self, op: dict, repairs: list[str]) -> dict:
        """Normalize a single plan op dict for small-model compat."""
        op = dict(op)

        if "path" in op:
            op["path"] = self._normalize_op_path(str(op["path"]), repairs)

        if "action" in op and "op" not in op and "type" not in op:
            action = str(op.pop("action")).lower().strip()
            mapped = self._ACTION_TO_OP.get(action, action)
            op["op"] = mapped
            repairs.append(f"action:{action!r}→op:{mapped!r}")

        op_type = normalize_key(str(op.get("op") or op.get("type") or ""))

        if op_type in ("insert_after", "insert_before"):
            if "anchor" not in op and ("start_line" in op or "line" in op):
                path = str(op.get("path") or "")
                line_no = int(op.get("start_line") or op.get("line") or 1)
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        idx = max(0, line_no - 1)
                        anchor_line = next(itertools.islice(_iter_file_lines(fp), idx, idx + 1), None)
                        if anchor_line is not None:
                            op["anchor"] = anchor_line
                            repairs.append(f"line {line_no}→anchor:{op['anchor']!r}")
                except Exception:
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_normalize_plan_op:0 suppressed Exception", exc_info=True
                    )
                op.pop("start_line", None)
                op.pop("end_line", None)
                op.pop("line", None)

            if "content" in op and "lines" not in op:
                content = op.pop("content")
                if isinstance(content, str):
                    op["lines"] = content.splitlines() or [""]
                elif isinstance(content, list):
                    op["lines"] = content
                repairs.append("content→lines")

        if op_type in ("insert_after", "insert_before") and "lines" in op and isinstance(op["lines"], str):
            op["lines"] = op["lines"].splitlines() or [op["lines"]]
            repairs.append("lines string→list")

        if op_type == "insert_after_line":
            if "content" in op and "lines" not in op:
                content = op.pop("content")
                if isinstance(content, str):
                    op["lines"] = content.splitlines() or [""]
                elif isinstance(content, list):
                    op["lines"] = content
                repairs.append("insert_after_line: content→lines")
            if "line" not in op and "start_line" in op:
                op["line"] = int(op.pop("start_line"))
                repairs.append("insert_after_line: start_line→line")
            _ins_lines = op.get("lines")
            if isinstance(_ins_lines, str):
                op["lines"] = _ins_lines.splitlines() or [_ins_lines]
                repairs.append("insert_after_line: lines string→list")

        if op_type == "edit_blocks" and "before" in op and "blocks" not in op and "edits" not in op:
            op["blocks"] = [{"before": op.pop("before"), "after": op.pop("after", "")}]
            repairs.append("before/after→blocks")

        if op_type == "edit_blocks":
            # 1. blocks dict→list
            raw_blocks = op.get("blocks") or op.get("edits")
            if isinstance(raw_blocks, dict):
                op["blocks"] = [raw_blocks]
                op.pop("edits", None)
                repairs.append("blocks dict→list")

            # 2. blocks alias normalization + line→before
            _before_aliases = ("old", "original", "from", "search", "replace_this", "find", "source", "existing")
            _after_aliases = (
                "new",
                "new_content",
                "replacement",
                "to",
                "with",
                "replace_with",
                "substitute",
                "target",
                "updated",
            )
            blocks = op.get("blocks") or op.get("edits") or []
            if isinstance(blocks, list) and blocks:
                new_blocks = []
                for blk in blocks:
                    if not isinstance(blk, dict):
                        new_blocks.append(blk)
                        continue
                    blk = dict(blk)
                    if not blk.get("before"):
                        for alias in _before_aliases:
                            if alias in blk:
                                blk["before"] = blk.pop(alias)
                                repairs.append(f"{alias}→before")
                                break
                    if blk.get("after") is None:
                        for alias in _after_aliases:
                            if alias in blk:
                                blk["after"] = blk.pop(alias)
                                repairs.append(f"{alias}→after")
                                break
                    # Block-level start_line/end_line → before (file read)
                    if not blk.get("before") and ("start_line" in blk or "line" in blk):
                        _path = str(op.get("path") or "")
                        _start = int(blk.get("start_line") or blk.get("line") or 1)
                        _end = int(blk.get("end_line") or _start)
                        try:
                            _fp = Path(self.repo_root) / _path
                            if _fp.exists():
                                blk["before"] = "\n".join(_file_line_window(_fp, _start - 1, _end - _start + 1))
                                blk.pop("start_line", None)
                                blk.pop("end_line", None)
                                blk.pop("line", None)
                                repairs.append(f"blk line {_start}-{_end}→before")
                        except Exception:
                            logger.debug(
                                "<module>::WriteToolsPatchMixin::_normalize_plan_op:1 suppressed Exception",
                                exc_info=True,
                            )
                    new_blocks.append(blk)
                op["blocks"] = new_blocks
                op.pop("edits", None)

            # 3. strip line-number prefixes
            blocks = op.get("blocks") or []
            if isinstance(blocks, list) and blocks:
                new_blocks = []
                any_stripped = False
                for blk in blocks:
                    if not isinstance(blk, dict):
                        new_blocks.append(blk)
                        continue
                    blk = dict(blk)
                    for field in ("before", "after"):
                        val = blk.get(field)
                        if isinstance(val, str):
                            lines_in = val.splitlines()
                            lines_out = [_RE_LINE_NUMBER_PREFIX.sub("", ln) for ln in lines_in]
                            if lines_out != lines_in:
                                blk[field] = "\n".join(lines_out)
                                any_stripped = True
                    new_blocks.append(blk)
                if any_stripped:
                    op["blocks"] = new_blocks
                    repairs.append("stripped line-number prefixes from blocks")

            # 4. before+indent enrichment
            blocks = op.get("blocks") or []
            path = str(op.get("path") or "")
            if isinstance(blocks, list) and blocks and path:
                fp = Path(self.repo_root) / path
                try:
                    if fp.exists():
                        any_enriched = False
                        new_blocks = []
                        for blk in blocks:
                            if not isinstance(blk, dict):
                                new_blocks.append(blk)
                                continue
                            before = blk.get("before", "")
                            if before and isinstance(before, str):
                                before_lines = before.splitlines()
                                if len(before_lines) == 1 and before[:1] not in (" ", "\t"):
                                    stripped = before.strip()
                                    matches: list[tuple[int, str]] = []
                                    for _i, _ln in enumerate(_iter_file_lines(fp)):
                                        if _ln.strip() == stripped:
                                            matches.append((_i, _ln))
                                    if len(matches) == 1:
                                        blk = dict(blk)
                                        blk["before"] = matches[0][1]
                                        repairs.append(f"before+indent (unique): {blk['before']!r:.60}")
                                        any_enriched = True
                                    # multi-match: do NOT silently prepend context — let LLM
                                    # provide a more specific before block via error feedback
                            new_blocks.append(blk)
                        if any_enriched:
                            op["blocks"] = new_blocks
                except Exception:
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_normalize_plan_op:2 suppressed Exception", exc_info=True
                    )
                    # non-critical enrichment — never block patch application

        if (
            op_type == "edit_blocks"
            and "blocks" not in op
            and "edits" not in op
            and "before" not in op
            and ("start_line" in op or "end_line" in op)
        ):
            path = str(op.get("path") or "")
            start = int(op.get("start_line") or 1)
            end = int(op.get("end_line") or start)
            content = op.get("content", "")
            try:
                fp = Path(self.repo_root) / path
                if fp.exists():
                    before_lines = _file_line_window(fp, start - 1, end - start + 1)
                    before_text = "\n".join(before_lines)
                    after_text = content if isinstance(content, str) else "\n".join(content)
                    op["blocks"] = [{"before": before_text, "after": after_text}]
                    op.pop("start_line", None)
                    op.pop("end_line", None)
                    op.pop("content", None)
                    repairs.append(f"line_range {start}-{end}→edit_blocks")
            except Exception:
                logger.debug("<module>::WriteToolsPatchMixin::_normalize_plan_op:3 suppressed Exception", exc_info=True)
                # non-critical line range → edit_blocks conversion

        # NOTE: empty edit_blocks (all blocks missing 'before') are left as-is so that
        # validation below rejects them with a clear error message — do NOT silently
        # convert to insert_before with an auto-chosen anchor, as that distorts LLM intent.

        return op

    def _detect_placeholder_op(self, op: dict[str, Any]) -> str | None:
        if not isinstance(op, dict):
            return None
        op_type = normalize_key(str(op.get("op") or op.get("type") or ""))
        if op_type != "edit_blocks":
            return None
        for blk in op.get("blocks") or []:
            if not isinstance(blk, dict):
                continue
            before = str(blk.get("before") or "").strip()
            after = str(blk.get("after") or "").strip()
            if before in self._PLACEHOLDER_BEFORE:
                return (
                    f"'before' contains a placeholder value ({before!r}). "
                    "You must read the target file first and copy the EXACT text "
                    "you want to replace into 'before'."
                )
            if after in self._PLACEHOLDER_AFTER:
                return (
                    f"'after' contains a placeholder value ({after!r}). "
                    "Replace 'after' with the actual new content you want."
                )
        return None

    def _enrich_plan_error(self, plan: Any, error_str: str) -> str:
        if not isinstance(plan, dict):
            return ""
        ops = plan.get("ops") or plan.get("operations") or []
        if not isinstance(ops, list) or not ops:
            return ""

        hints: list[str] = []

        for op in ops:
            if not isinstance(op, dict):
                continue
            op_type = normalize_key(str(op.get("op") or op.get("type") or ""))
            path = str(op.get("path") or "")

            if op_type == "edit_blocks" and "missing" in error_str.lower() and "before" in error_str.lower():
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        with fp.open(encoding="utf-8", errors="replace") as fh:
                            _chunk = fh.read(1501)
                        preview = _chunk[:1500]
                        if len(_chunk) > 1500:
                            preview += "\n... (truncated)"
                        hints.append(
                            f"HINT: For edit_blocks on '{path}', the 'before' field must contain "
                            f"exact text from the file. Current file content:\n```\n{preview}\n```\n"
                            f"Copy the exact lines you want to replace into 'before', and put "
                            f"the replacement text in 'after'.\n"
                            f"To INSERT a new line without replacing anything, use "
                            f"op='insert_before' or op='insert_after' with an 'anchor' line and "
                            f"a 'lines' list instead."
                        )
                except Exception:
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_enrich_plan_error:0 suppressed Exception", exc_info=True
                    )
                    # non-critical: error message building must not block

            if op_type == "edit_blocks" and "not found" in error_str.lower():
                try:
                    import difflib

                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        before_text = ""
                        for blk in op.get("blocks") or []:
                            if isinstance(blk, dict) and blk.get("before"):
                                before_text = str(blk["before"])
                                break
                        before_first = before_text.splitlines()[0].strip() if before_text else ""

                        ctx_hint = ""
                        if before_first:
                            # Streaming scan replaces get_close_matches over the
                            # whole file — same ratio metric, with autojunk=False
                            # (P24-D gate class): a >200-char before_first must
                            # not have popular chars purged from the comparison.
                            best_idx = -1
                            best_ratio = 0.0
                            for _i, _ln in enumerate(_iter_file_lines(fp)):
                                _r = difflib.SequenceMatcher(None, before_first, _ln, autojunk=False).ratio()
                                if _r > best_ratio:
                                    best_ratio = _r
                                    best_idx = _i
                            if best_idx >= 0 and best_ratio >= 0.5:
                                _start = max(0, best_idx - 2)
                                _w = _file_line_window(fp, _start, best_idx + 8 - _start)
                                # Gutter format, not "NNN: line": this block
                                # is immediately followed by "copy the EXACT
                                # text", and a bare listing hides the one
                                # column the mismatch is usually about.
                                ctx = "\n".join(format_numbered_line(_start + j + 1, _w[j]) for j in range(len(_w)))
                                ctx_hint = (
                                    f"\nClosest match found near line {best_idx + 1} "
                                    f"(│N│ = leading-whitespace count; copy the code after it, "
                                    f"not the gutter):\n```\n{ctx}\n```\n"
                                    f"Copy the EXACT text from this block into 'before'."
                                )

                        if ctx_hint:
                            hints.append(
                                f"HINT: 'before' text not found in '{path}'. "
                                f"Your text did not match the file.{ctx_hint}"
                            )
                        else:
                            preview = "\n".join(
                                format_numbered_line(i + 1, _item_)
                                for i, _item_ in enumerate(itertools.islice(_iter_file_lines(fp), 60))
                            )
                            hints.append(
                                f"HINT: 'before' text not found in '{path}'. "
                                f"Use find_symbol to locate the target, then read_file with "
                                f"start_line/end_line around it — NOT bash cat, which omits the "
                                f"│N│ indent gutter that this mismatch is usually about. "
                                f"First 60 lines:\n```\n{preview}\n```"
                            )
                except Exception:
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_enrich_plan_error:1 suppressed Exception", exc_info=True
                    )
                    # non-critical: error message building must not block

            if op_type == "create_file" and "already exists" in error_str.lower():
                hints.append(
                    f"HINT: '{path}' already exists. Use op='replace_file' to overwrite it, "
                    f"or op='edit_blocks' to make partial changes."
                )

            if op_type in ("insert_after", "insert_before") and "anchor" in error_str.lower():
                try:
                    fp = Path(self.repo_root) / path
                    if fp.exists():
                        first_lines = list(itertools.islice(_iter_file_lines(fp), 10))
                        preview = "\n".join(first_lines)
                        hints.append(
                            f"HINT: For {op_type} on '{path}', 'anchor' must be an exact line from "
                            f"the file. First 10 lines:\n```\n{preview}\n```"
                        )
                except Exception:
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_enrich_plan_error:2 suppressed Exception", exc_info=True
                    )
                    # non-critical: error message building must not block

        return "\n".join(hints)

    def _looks_like_unified_diff(self, text: str) -> bool:
        t = str(text or "")
        if not t.strip():
            return False
        has_header = any(s in t for s in ("diff --git ", "--- a/", "+++ b/")) or t.lstrip().startswith("--- ")
        has_hunk = "@@ " in t
        # Allow hunk-only patches (starting with @@, no header) — git apply handles them
        return bool(has_hunk and (has_header or t.lstrip().startswith("@@")))

    def _write_staged_files_directly(
        self,
        staged: dict[str, str],
        picked_files: list[str],
    ) -> ToolResult:
        """Apply a compiled plan by writing each file's final content directly.

        plan_compiler already computed the exact post-edit content of every
        touched file; this writes it to disk without going through `git apply`,
        sidestepping that path's failure modes (context fuzz, trailing-newline
        mismatch, untracked/gitignored paths, missing --3way blob).

        Safety mirrors the git-apply path: snapshot existing files, write, run
        py_compile on touched .py files, and roll everything back (restore
        snapshots, delete created files) if any syntax error is introduced.
        Files whose content is unchanged are skipped (not touched, not counted).
        """
        import os as _os

        repo = str(self._effective_repo_root)
        snapshots: dict[str, str] = {}  # abs_path -> original content (existing files)
        created: list[str] = []  # abs_paths that did not exist before
        written: list[str] = []  # rel_paths actually written (changed)

        def _rollback() -> None:
            for _ap, _orig in snapshots.items():
                try:
                    atomic_write_text(_ap, _orig)
                except OSError as _re:
                    logger.debug("write_plan rollback restore failed for %s: %s", _ap, _re)
            for _ap in created:
                try:
                    _os.remove(_ap)
                except OSError as _re:
                    logger.debug("write_plan rollback remove failed for %s: %s", _ap, _re)

        try:
            for rel in picked_files or list(staged.keys()):
                if rel not in staged:
                    continue
                new_content = staged[rel]
                ap = _os.path.join(repo, rel)
                if _os.path.isfile(ap):
                    try:
                        with open(ap, encoding="utf-8", errors="replace") as _fh:
                            cur = _fh.read()
                    except OSError as _re:
                        # The file exists but we cannot read it (permission denied,
                        # race, etc.). We therefore cannot snapshot it, so
                        # overwriting would break the rollback contract (a later
                        # syntax error could not restore it). Abort; the outer
                        # handler rolls back everything written so far.
                        raise OSError(f"cannot read existing file {rel} for snapshot: {_re}") from _re
                    if cur == new_content:
                        continue  # unchanged — don't touch
                    snapshots[ap] = cur
                else:
                    _parent = _os.path.dirname(ap)
                    if _parent and not _os.path.isdir(_parent):
                        _os.makedirs(_parent, exist_ok=True)
                    created.append(ap)
                atomic_write_text(ap, new_content)
                written.append(rel)
        except Exception as exc:
            _rollback()
            return self._make_result(
                ok=False,
                content="",
                error=f"Direct write failed: {type(exc).__name__}: {exc}",
                metadata={"rolled_back": True},
            )

        # Post-write syntax validation for .py (mirrors git-apply rollback gate).
        _syntax_errors: list[str] = []
        for rel in written:
            if LanguageId.from_path(rel) is LanguageId.PYTHON:
                try:
                    compile_quiet(staged[rel], rel, "exec")
                except SyntaxError as _se:
                    _syntax_errors.append(f"{rel}: {_se}")
                except Exception as _exc:
                    logger.debug("write_plan direct-write compile() non-SyntaxError for %s: %s", rel, _exc)
        if _syntax_errors:
            _rollback()
            return self._make_result(
                ok=False,
                content="",
                error=f"Plan introduced syntax errors (rolled back): {'; '.join(_syntax_errors)}",
                metadata={"syntax_errors": _syntax_errors, "rolled_back": True},
            )

        if written:
            try:
                self._invalidate_cache_after_write(written)
            except Exception as _exc:
                logger.debug("write_plan cache invalidation failed: %s", _exc)
        return self._make_result(
            ok=True,
            content=f"Wrote {len(written)} file(s) directly",
            metadata={"touched_files": written},
        )

    # ── Main write tools ─────────────────────────────────────────────── #

    def _tool_write_plan(self, args: dict[str, Any]) -> ToolResult:
        if "__raw_arguments" in args and "plan" not in args:
            import json as _json

            _raw = args["__raw_arguments"]
            if isinstance(_raw, str):
                try:
                    _parsed = _json.loads(_raw)
                    if isinstance(_parsed, dict):
                        args = _parsed
                        logger.info("write_plan: recovered args from __raw_arguments")
                except (ValueError, _json.JSONDecodeError):
                    _repaired = _repair_plan_json(_raw)
                    _start = -1
                    try:
                        _parsed = _json.loads(_repaired)
                        if isinstance(_parsed, dict):
                            args = _parsed
                            logger.info("write_plan: recovered args from __raw_arguments (repaired)")
                    except (ValueError, _json.JSONDecodeError):
                        _start = _raw.find("{")
                    _end = _raw.rfind("}")
                    if _start != -1 and _end > _start:
                        _sub = _raw[_start : _end + 1]
                        try:
                            _parsed = _json.loads(_sub)
                            if isinstance(_parsed, dict):
                                args = _parsed
                                logger.info("write_plan: recovered args from __raw_arguments (substring)")
                        except (ValueError, _json.JSONDecodeError):
                            # Try repair on substring too
                            _sub_repaired = _repair_plan_json(_sub)
                            try:
                                _parsed = _json.loads(_sub_repaired)
                                if isinstance(_parsed, dict):
                                    args = _parsed
                                    logger.info("write_plan: recovered args from __raw_arguments (substring repaired)")
                            except (ValueError, _json.JSONDecodeError):
                                logger.debug(
                                    "<module>::WriteToolsPatchMixin::_tool_write_plan:3 suppressed (ValueError, _json.JSONDecodeError)",
                                    exc_info=True,
                                )

        plan = args.get("plan")
        if not plan:
            ops = args.get("ops")
            if ops is None:
                ops = args.get("operations")
            if ops is None:
                _raw_hint = ""
                _raw = args.get("__raw_arguments", "")
                if isinstance(_raw, str) and len(_raw) > 10:
                    # Detect truncation: unclosed braces indicate the tool_call
                    # arguments were cut off mid-stream (common with large content fields).
                    _trimmed = _raw.strip()
                    _open_br = _trimmed.count("{")
                    _close_br = _trimmed.count("}")
                    if _open_br > _close_br:
                        # Best-effort: identify which op/path was being written
                        # when the stream was cut, by string-boundary-aware scan
                        # of the (unparseable) raw payload. Purely diagnostic.
                        _target = _extract_truncated_op_path(_raw)
                        _target_hint = f" Truncated target: {_target}." if _target else ""
                        return self._make_result(
                            ok=False,
                            content="",
                            error=(
                                f"write_plan: tool_call arguments were truncated "
                                f"({_open_br - _close_br} unclosed braces, {len(_trimmed)} chars)."
                                f"{_target_hint} "
                                f"For large file creation/edits, use bash (python3/cat) to write "
                                f"the file directly, then use write_plan to update other files."
                            ),
                        )
                    _raw_hint = f" (raw args: {_raw[:120]})"
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"plan is required{_raw_hint}. Correct format:\n"
                        'write_plan({"plan": {"kind": "ASICODE_PLAN_V1", "ops": ['
                        '{"op": "create_file", "path": "path/to/file.py", "content": "file content here"}'
                        "]}})\n"
                        "For new files use op='create_file'. For patches use op='patch' with unified diff."
                    ),
                )
            plan = {"kind": "ASICODE_PLAN_V1", "ops": ops}

        # Normalize string/JSON/list/top-level-ops plan to dict (LLM may pass a
        # raw JSON string, a markdown block, or a bare list). Delegated to
        # write_targets.normalize_plan, which is the single source of truth:
        # the pre-write gates (Undo checkpoint, rollback snapshot, file lock)
        # resolve their targets BEFORE this handler runs, and while they carried
        # their own copy of this normalisation every shape below except the
        # documented dict was invisible to them — the plan wrote normally and
        # the run silently got no Undo point and no lock.
        plan = normalize_plan(args)
        if isinstance(plan, str):
            # Survived normalisation as text → not JSON at all. Quote the
            # (fence-stripped) input back so the model can see what we received.
            _sample = plan[:200].replace("\n", "\\n")
            return self._make_result(
                ok=False, content="", error=(f"plan must be a valid JSON object. Received: {_sample}")
            )

        # Guard BEFORE any plan.get(...) call below: json.loads above can yield a
        # non-dict scalar (int/float/bool/str/None from inputs like {"plan": "42"}
        # or {"plan": true}), and plan.get(...) on such a value raises
        # AttributeError — surfacing as a generic dispatch error instead of the
        # intended "plan must be a JSON object" rejection.
        if not isinstance(plan, dict):
            return self._make_result(ok=False, content="", error="plan must be a JSON object")

        # Normalize ops for plan_compiler compatibility (path, field aliases, etc.)
        repairs: list[str] = []
        ops = plan.get("ops") or plan.get("operations") or []
        for i, op in enumerate(ops):
            if isinstance(op, dict):
                ops[i] = self._normalize_plan_op(op, repairs)
        if repairs:
            logger.info("write_plan normalized ops: %s", "; ".join(repairs))

        kind = plan.get("kind") or plan.get("version")
        if not kind or str(kind).strip() != "ASICODE_PLAN_V1":
            return self._make_result(
                ok=False, content="", error="plan must have 'kind' or 'version' field set to 'ASICODE_PLAN_V1'"
            )

        ops = plan.get("ops") or plan.get("operations")
        if not ops:
            return self._make_result(ok=False, content="", error="plan must have non-empty 'ops' or 'operations' list")
        if not isinstance(ops, list):
            return self._make_result(ok=False, content="", error="'ops' or 'operations' must be a list")
        if len(ops) == 0:
            return self._make_result(ok=False, content="", error="'ops' or 'operations' list must not be empty")

        for op_idx, op in enumerate(ops or []):
            # ── Phase 1: op must be a dict ───────────────────────────────
            if not isinstance(op, dict):
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: ops[{op_idx}] is not a JSON object "
                        f"(type={type(op).__name__!r}).\n"
                        "ACTION: Each op must be a JSON object with 'op', 'path', "
                        "and type-specific fields."
                    ),
                )

            # ── Phase 2: Placeholder content detection ──────────────────
            ph_err = self._detect_placeholder_op(op)
            if ph_err:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: {ph_err}\n"
                        "ACTION: Use read_file on the target file first, then use the actual "
                        "text from the file in 'before', and your desired replacement in 'after'. "
                        "read_file, not bash cat: its │N│ gutter shows each line's exact "
                        "leading-whitespace count, which 'before' has to match."
                    ),
                )

            # ── Phase 3: Path validation ────────────────────────────────
            path_val = op.get("path")
            if path_val is None or str(path_val).strip() == "":
                op_type_info = str(op.get("op") or op.get("type") or "(unknown)")
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: ops[{op_idx}] (type={op_type_info!r}) has missing or empty 'path'.\n"
                        "ACTION: Every op must include a 'path' field with the relative file path "
                        "(e.g. 'external_llm/agent/example.py'). Add the correct path and retry write_plan."
                    ),
                )
            if ".." in str(path_val).split("/"):
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: ops[{op_idx}] has path traversal ('..') in "
                        f"path={path_val!r}.\n"
                        "ACTION: Use a relative path within the repository, without '..' segments."
                    ),
                )

            # ── Phase 4: Op type validation ─────────────────────────────
            raw_op_type = str(op.get("op") or op.get("type") or "").strip()
            if not raw_op_type:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: ops[{op_idx}] is missing 'op' or 'type' field.\n"
                        f"ACTION: Add an 'op' field. Supported types: "
                        f"{', '.join(sorted(self._WRITE_PLAN_OP_TYPES))}."
                    ),
                )
            # Normalize the same way plan_compiler._norm_op_type does
            op_type = normalize_key(raw_op_type)
            op_type = "".join(ch for ch in op_type if (ch.isalnum() or ch == "_"))
            op_type = self._OP_TYPE_ALIASES.get(op_type, op_type)
            if op_type not in self._WRITE_PLAN_OP_TYPES:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: ops[{op_idx}] has unsupported op type "
                        f"{raw_op_type!r} (normalized={op_type!r}).\n"
                        f"ACTION: Use one of the supported types: "
                        f"{', '.join(sorted(self._WRITE_PLAN_OP_TYPES))}."
                    ),
                )

            # ── Phase 5: Per-op required field validation ───────────────
            if op_type in ("create_file", "replace_file") and ("content" not in op or op.get("content") is None):
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"write_plan rejected: ops[{op_idx}] ({op_type}) is missing "
                        f"'content' field.\n"
                        "ACTION: Add a 'content' field with the full file content."
                    ),
                )

            if op_type == "edit_blocks":
                edits = op.get("edits") or op.get("blocks")
                if not isinstance(edits, list) or not edits:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"write_plan rejected: ops[{op_idx}] (edit_blocks) is missing "
                            f"non-empty 'edits' or 'blocks' list.\n"
                            "ACTION: Add an 'edits' list with 'before'/'after' pairs."
                        ),
                    )

            if op_type in ("insert_after", "insert_before"):
                if not op.get("anchor"):
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"write_plan rejected: ops[{op_idx}] ({op_type}) is missing "
                            f"'anchor' field.\n"
                            "ACTION: Add an 'anchor' field with an exact line from the target "
                            "file. Then add 'lines' with the text to insert."
                        ),
                    )
                lines = op.get("lines")
                if not isinstance(lines, list) or not lines:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"write_plan rejected: ops[{op_idx}] ({op_type}) has missing or "
                            f"non-list 'lines' field.\n"
                            "ACTION: Add a 'lines' list with the text to insert."
                        ),
                    )

            if op_type == "insert_after_line":
                op_line = op.get("line")
                if not isinstance(op_line, int) or op_line < 1:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"write_plan rejected: ops[{op_idx}] (insert_after_line) is missing "
                            f"or invalid 'line' field. Must be a positive integer.\n"
                            "ACTION: Add a 'line' field with the 1-based line number."
                        ),
                    )
                lines = op.get("lines")
                if not isinstance(lines, list) or not lines:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"write_plan rejected: ops[{op_idx}] (insert_after_line) has missing or "
                            f"non-list 'lines' field.\n"
                            "ACTION: Add a 'lines' list with the text to insert."
                        ),
                    )

        # F1 cross-process edit-lease guard: refuse when a parallel session
        # holds a live lease on any op target (pre-check before compile/apply).
        _lease_paths = [str(op.get("path") or "") for op in (ops or []) if isinstance(op, dict) and op.get("path")]
        _lease_refused = self._refuse_foreign_leased(_lease_paths)
        if _lease_refused is not None:
            return _lease_refused

        # plan_compiler is a first-party root module — import cannot fail.
        from plan_compiler import compile_plan_to_unified_diff

        def _compile_and_apply(p: dict[str, Any]) -> ToolResult:
            try:
                result = compile_plan_to_unified_diff(
                    repo_root=str(self._effective_repo_root),
                    plan=p,
                    allow_empty=False,
                )
            except Exception as exc:
                return self._make_result(
                    ok=False,
                    content="",
                    error=f"Plan compilation failed: {type(exc).__name__}: {exc}",
                )
            patch = result.diff_patch or ""
            warnings = result.warnings or []

            if not patch.strip():
                return self._make_result(ok=False, content="", error="Plan compiled to empty patch")

            # ── Apply by writing compiler-staged content directly (no git apply) ──
            # The compiler already produced the exact final content of every touched
            # file (result.staged). Writing it directly avoids the git-apply failure
            # modes the diff round-trip introduced — context fuzz, trailing-newline
            # mismatch, untracked/gitignored paths, missing --3way blob — none of
            # which reflect a real problem with the computed result. The diff is
            # still kept for the no-op gate above, line-count display, and the
            # applied_patches record. Snapshot + py_compile + rollback preserve the
            # same safety the git-apply path provided.
            apply_result = self._write_staged_files_directly(
                result.staged,
                result.picked_files,
            )
            if not apply_result.ok:
                _err_content = f"Plan compiled successfully but apply failed: {apply_result.error}"
                _err_metadata = {"patch": patch, "warnings": warnings, "files": result.picked_files}
                if apply_result.metadata:
                    for _k, _v in apply_result.metadata.items():
                        _err_metadata.setdefault(_k, _v)
                return self._make_result(
                    ok=False,
                    content=_err_content,
                    error=apply_result.error,
                    metadata=_err_metadata,
                )

            # Line counts come from the diff (display only).
            added_lines = removed_lines = 0
            try:
                for line in patch.split("\n"):
                    if line.startswith(("+++ ", "--- ")):
                        continue
                    if line.startswith("+"):
                        added_lines += 1
                    elif line.startswith("-"):
                        removed_lines += 1
            except (AttributeError, TypeError):
                logger.debug(
                    "<module>::WriteToolsPatchMixin::_tool_write_plan::_compile_and_apply:1 suppressed (AttributeError, TypeError)",
                    exc_info=True,
                )

            touched_files = (apply_result.metadata or {}).get("touched_files") or result.picked_files or []
            display_file = touched_files[0] if touched_files else "unknown"
            if added_lines > 0 or removed_lines > 0:
                line_info = f" (+{added_lines} lines, -{removed_lines} lines)"
            else:
                line_info = ""

            # Record the diff so agent_loop detects a real edit happened.
            if patch:
                self._append_applied_patch(str(patch))

            content_parts = [f"Plan applied. Touched: {display_file}{line_info}"]
            if warnings:
                content_parts.append("Warnings: " + "; ".join(warnings))
            return self._make_result(
                ok=True,
                content="\n".join(content_parts),
                metadata={
                    "patch": patch,
                    "warnings": warnings,
                    "files": result.picked_files,
                    "touched_files": touched_files,
                },
            )

        try:
            compile_result = _compile_and_apply(plan)
            if compile_result.ok:
                return compile_result

            first_error = compile_result.error or ""
            error_msg = first_error
            enriched = self._enrich_plan_error(plan, first_error)
            if enriched:
                error_msg = f"{error_msg}\n\n{enriched}"

            compile_result.error = error_msg

        except Exception as e:
            error_msg = f"Plan compilation failed: {type(e).__name__}: {e}"
            enriched = self._enrich_plan_error(plan, str(e))
            if enriched:
                error_msg = f"{error_msg}\n\n{enriched}"
            return self._make_result(ok=False, content="", error=error_msg)
        else:
            return compile_result

    # ── apply_patch → modify_symbol auto-fallback ──────────────────────────
    # When PatchEngine fails (e.g. context mismatch on a freshly-edited/
    # untracked file where `git apply --3way` has no pre-image blob to merge
    # against), inspect the unified diff: if it touches a SINGLE Python file
    # and replaces exactly ONE top-level def/class symbol, route it to
    # modify_symbol, which is AST-based and needs no git blob. This eliminates
    # the manual LLM retry loop for the most common single-symbol patch failure.

    _FALLBACK_PATCH_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*$")

    def _parse_unified_diff_files(self, patch_text: str) -> list[dict[str, Any]]:
        """Split a unified diff into per-file hunks.

        Returns list of dicts: {file, hunks: [{old_start, old_count, new_start,
        new_count, lines: [(kind, text), ...]}]} where kind is '+', '-', or ' '.
        Conservative: returns [] on any structural ambiguity (binary patches,
        rename/copy headers, no recognizable hunks).
        """
        if not patch_text:
            return []
        # Reject rename/copy/mode-only patches — out of scope for symbol edit.
        if re.search(
            r"^(rename from|rename to|copy from|copy to|similarity index|new file mode|deleted file mode|old mode|new mode)\b",
            patch_text,
            re.MULTILINE,
        ):
            return []

        files: list[dict[str, Any]] = []
        cur_file: dict[str, Any] | None = None
        cur_hunk: dict[str, Any] | None = None

        for raw_line in patch_text.splitlines():
            if raw_line.startswith("diff --git "):
                # New file boundary. Defer file path to +++ header.
                cur_file = {"file": None, "hunks": []}
                files.append(cur_file)
                cur_hunk = None
                continue
            m_path = re.match(r"^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$", raw_line)
            if m_path and raw_line != "+++ /dev/null":
                if cur_file is None:
                    cur_file = {"file": None, "hunks": []}
                    files.append(cur_file)
                cur_file["file"] = m_path.group(1).strip()
                cur_hunk = None
                continue
            m_hunk = self._FALLBACK_PATCH_HUNK_RE.match(raw_line)
            if m_hunk:
                if cur_file is None:
                    cur_file = {"file": None, "hunks": []}
                    files.append(cur_file)
                cur_hunk = {
                    "old_start": int(m_hunk.group(1)),
                    "old_count": int(m_hunk.group(2)) if m_hunk.group(2) else 1,
                    "new_start": int(m_hunk.group(3)),
                    "new_count": int(m_hunk.group(4)) if m_hunk.group(4) else 1,
                    "lines": [],
                }
                cur_file["hunks"].append(cur_hunk)
                continue
            if cur_hunk is not None and (raw_line.startswith(("+", "-", " "))):
                # Context/added/removed line. Skip '--- '/'+++ ' file headers
                # (marker + space); content lines like '---x'/'+++x' (no space)
                # are real body lines and must NOT be dropped (WP-B1).
                if raw_line.startswith(("--- ", "+++ ")):
                    continue
                cur_hunk["lines"].append((raw_line[0], raw_line[1:]))
                continue
            # Lines outside any hunk (e.g. 'Index:', diff metadata) — ignore.

        # Drop files with no resolved path or no hunks.
        return [f for f in files if f["file"] and f["hunks"]]

    def _extract_new_file_target(self, patch_text: str, path_hint: str | None) -> dict[str, Any] | None:
        """Detect a new-file unified diff and extract its full content.

        A creation patch has no pre-image (``--- /dev/null`` and/or a
        ``new file mode`` header) and every body line is an addition, so the
        '+' lines ARE the complete file content — no context matching and no
        git blob are needed. Such a patch can be routed to create_file safely.

        Conservative: requires exactly ONE created file, a /dev/null or
        new-file-mode signal, and a body of pure '+' additions (any '-' or
        non-marker context line disqualifies it).

        Returns None (not a clean creation) or {file_path, content}.
        """
        if not patch_text:
            return None
        lines = patch_text.splitlines()
        is_creation_signal = any(
            _item_.strip() == "--- /dev/null" or _item_.startswith("new file mode") for _item_ in lines
        )
        if not is_creation_signal:
            return None

        # Resolve the created path from the +++ header (ignore /dev/null).
        new_path: str | None = None
        plus_headers = 0
        for raw in lines:
            m = re.match(r"^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$", raw)
            if m and raw != "+++ /dev/null":
                plus_headers += 1
                new_path = m.group(1).strip()
        if plus_headers != 1:
            return None  # zero or multiple files — out of scope
        file_path = new_path or path_hint
        if not file_path:
            return None

        # Collect body: must be a pure-addition hunk set.
        content_lines: list[str] = []
        in_hunk = False
        for raw in lines:
            # Skip '+++ '/'--- ' file headers only; '+++x' (content starting
            # with '++') is a real added line and must survive (WP-B1).
            if raw.startswith(("+++ ", "--- ")):
                continue
            if self._FALLBACK_PATCH_HUNK_RE.match(raw):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if raw.startswith("\\"):
                # "\ No newline at end of file" marker — note and skip.
                continue
            if raw.startswith("+"):
                content_lines.append(raw[1:])
                continue
            if raw.strip() == "":
                # Blank line inside a creation hunk is an unprefixed empty
                # context line in some diff dialects — treat as empty content.
                content_lines.append("")
                continue
            # Any '-' (deletion) or ' ' context line means this is NOT a pure
            # creation; bail out so we don't fabricate content.
            return None

        if not content_lines:
            return None
        content = "\n".join(content_lines)
        # Preserve trailing newline unless a "no newline" marker was present.
        no_newline = any(_item_.startswith("\\") for _item_ in lines)
        if not no_newline:
            content += "\n"
        return {"file_path": file_path, "content": content}

    def _analyze_patch_symbol_change(
        self,
        patch_text: str,
        path_hint: str | None = None,
    ) -> dict[str, Any] | None:
        """Apply a failed patch to the on-disk file in memory and diff symbols.

        Shared core for the single- and multi-symbol fallbacks. Requires exactly
        ONE supported file whose hunks anchor cleanly against the current disk
        content (blob-free — this survives a missing git pre-image blob). Because
        the new source is built by splicing hunks into the real disk source,
        untouched lines are always preserved (no truncation/data-loss).

        Headerless ``@@``-only patches (a small-model dialect) carry no file
        path; when ``path_hint`` is given and no header resolves, the hint is
        synthesized into a ``+++ b/<hint>`` header so the shared hunk-anchoring
        logic stays identical. A resolvable header always wins over the hint —
        a conflicting hint never redirects the patch.

        Python files are located via the stdlib ``ast`` (PythonAstLocator); other
        languages via the tree-sitter ``SyntaxProvider`` registry
        (ProviderLocator), so the fallback is multi-language.

        Returns None (ineligible) or a dict:
          {file_path, abs_path, language, is_python, old_lines, new_lines,
           new_src, old_by_name, new_by_name, changed}
        where *_by_name map a top-level symbol's qualname to its source text and
        `changed` is the set of qualnames whose source text differs.
        """
        try:
            files = self._parse_unified_diff_files(patch_text)
            if len(files) != 1 and path_hint:
                # Headerless @@-only patch: synthesize the missing header from
                # the caller's path argument (the twin of _extract_new_file_target's
                # ``new_path or path_hint`` — a fallback, never an override).
                files = self._parse_unified_diff_files(
                    f"--- a/{path_hint}\n+++ b/{path_hint}\n" + patch_text,
                )
        except Exception:
            logger.debug(
                "<module>::WriteToolsPatchMixin::_analyze_patch_symbol_change:0 suppressed Exception", exc_info=True
            )
            return None
        if len(files) != 1:
            return None
        f = files[0]
        file_path = f["file"]

        # ── Resolve a locator: Python via ast, others via tree-sitter provider ──
        is_python = file_path.endswith(".py")
        language = "python"
        locator: Any
        if is_python:
            from external_llm.agent.symbol_locator import PythonAstLocator

            locator = PythonAstLocator()
        else:
            try:
                from external_llm.agent.symbol_locator import ProviderLocator
                from external_llm.languages.models import LanguageId
                from external_llm.languages.registry import LanguageRegistry
            except Exception:
                logger.debug(
                    "<module>::WriteToolsPatchMixin::_analyze_patch_symbol_change:1 suppressed Exception", exc_info=True
                )
                return None
            lang_id = LanguageId.from_path(file_path)
            if lang_id == LanguageId.UNKNOWN:
                return None  # unsupported language — surface original error
            language = lang_id.value
            provider = LanguageRegistry.instance().get(file_path)
            if provider is None:
                return None
            locator = ProviderLocator(provider)

        try:
            sec = self._secure_path(file_path)
            if sec is None:
                return None
            abs_path = str(sec)
            if not os.path.isfile(abs_path):
                return None
            with open(abs_path, encoding="utf-8") as fh:
                existing = fh.read()
        except Exception:
            logger.debug(
                "<module>::WriteToolsPatchMixin::_analyze_patch_symbol_change:2 suppressed Exception", exc_info=True
            )
            return None

        old_lines = existing.split("\n")
        new_lines = self._apply_hunks_in_memory(old_lines, f["hunks"])
        if new_lines is None:
            return None  # a hunk couldn't be anchored — bail (no fabrication)
        new_src = "\n".join(new_lines)

        old_spans = [s for s in locator.locate(existing) if s.top_level]
        new_spans = [s for s in locator.locate(new_src) if s.top_level]
        if not new_spans and not old_spans:
            return None  # unparseable / no symbols on both sides → not safe

        def _src(lines: list[str], span) -> str:
            return "\n".join(lines[span.start_line - 1 : span.end_line])

        old_by_name = {s.qualname: _src(old_lines, s) for s in old_spans}
        new_by_name = {s.qualname: _src(new_lines, s) for s in new_spans}
        changed = {
            name for name in (set(old_by_name) | set(new_by_name)) if old_by_name.get(name) != new_by_name.get(name)
        }
        return {
            "file_path": file_path,
            "abs_path": abs_path,
            "language": language,
            "is_python": is_python,
            "old_lines": old_lines,
            "new_lines": new_lines,
            "new_src": new_src,
            "old_by_name": old_by_name,
            "new_by_name": new_by_name,
            "changed": changed,
        }

    def _extract_modify_symbol_target(self, patch_text: str, path_hint: str | None) -> dict[str, Any] | None:
        """Analyze a failed patch to see if it can route to modify_symbol.

        Eligible when the patch changes EXACTLY ONE top-level symbol that already
        exists in the file (a modify, not an add). See
        ``_analyze_patch_symbol_change`` for the shared, data-loss-free core.

        Returns None (ineligible) or a dict: {file_path, symbol, code, reason}
        """
        info = self._analyze_patch_symbol_change(patch_text, path_hint)
        if info is None:
            return None
        if not info["is_python"]:
            return None  # modify_symbol is Python-AST-only; others use rewrite
        changed = info["changed"]
        if len(changed) != 1:
            return None  # zero or multiple symbols changed (multi-symbol → None)
        symbol = next(iter(changed))
        # Must be a MODIFY (symbol exists in both): an add/remove is out of scope.
        if symbol not in info["old_by_name"] or symbol not in info["new_by_name"]:
            return None

        code = info["new_by_name"][symbol]
        if not code or not code.strip():
            return None
        return {
            "file_path": info["file_path"],
            "symbol": symbol,
            "code": code,
            "reason": "single_python_symbol",
        }

    def _extract_multi_symbol_rewrite(self, patch_text: str, path_hint: str | None) -> dict[str, Any] | None:
        """See if a failed patch can be applied as a multi-symbol rewrite.

        Eligible when the patch changes TWO OR MORE top-level symbols (the
        single-symbol case is handled by modify_symbol). The complete new file
        was already reconstructed by ``_analyze_patch_symbol_change`` via
        content-anchored hunk application, so the only remaining safety bar is
        that it parses. Application is an atomic whole-file write (all-or-nothing
        by construction — no partial per-symbol state).

        Returns None (ineligible) or a dict:
          {file_path, abs_path, new_src, symbols}
        """
        info = self._analyze_patch_symbol_change(patch_text, path_hint)
        if info is None:
            return None
        changed = info["changed"]
        # Python single-symbol edits are handled by modify_symbol (nicer diffs);
        # for other languages there is no modify_symbol, so the rewrite path owns
        # single edits too.
        min_changed = 2 if info["is_python"] else 1
        if len(changed) < min_changed:
            return None
        # The new file must be syntactically valid before we write it wholesale.
        if info["is_python"]:
            try:
                import ast as _ast

                _ast.parse(info["new_src"])
            except SyntaxError:
                logger.debug(
                    "<module>::WriteToolsPatchMixin::_extract_multi_symbol_rewrite:0 suppressed SyntaxError",
                    exc_info=True,
                )
                return None
        else:
            try:
                from external_llm.languages.tree_sitter_utils import has_error

                err = has_error(info["new_src"], info["language"])
            except Exception:
                err = None
            # True → syntax error; None → tree-sitter unavailable (can't verify).
            # Bail in both cases; only a clean parse (False) is safe to write.
            if err is not False:
                return None
        return {
            "file_path": info["file_path"],
            "abs_path": info["abs_path"],
            "new_src": info["new_src"],
            "symbols": sorted(changed),
        }

    def _find_block(self, lines: list[str], block: list[str], hint: int = 0) -> int | None:
        """Locate a contiguous ``block`` within ``lines``; return its start index.

        Tries exact match, then trailing-whitespace-insensitive, then fully
        whitespace-stripped — to tolerate the kind of drift that makes git apply
        fail. When several positions match, picks the one nearest ``hint`` (the
        hunk's expected location) to disambiguate repeated blocks. Returns None
        if the block cannot be found (caller must treat as ineligible).
        """
        if not block:
            return max(0, min(hint, len(lines)))
        n = len(block)
        if n > len(lines):
            return None
        for key in (lambda s: s, lambda s: s.rstrip(), lambda s: s.strip()):
            keyed = [key(b) for b in block]
            matches = [i for i in range(0, len(lines) - n + 1) if [key(lines[j]) for j in range(i, i + n)] == keyed]
            if matches:
                return min(matches, key=lambda i: abs(i - hint))
        return None

    def _apply_hunks_in_memory(self, file_lines: list[str], hunks: list[dict[str, Any]]) -> list[str] | None:
        """Apply unified-diff hunks to ``file_lines`` by content matching.

        Pure-Python, blob-free: each hunk's old-side block (context + deleted
        lines) is located in the evolving file by content (not line number, so
        it survives line drift) and spliced with the new-side block (context +
        added lines). Hunks are assumed ascending; a running delta keeps the
        search hint aligned as earlier splices shift line numbers.

        Returns the new file lines, or None if any hunk cannot be anchored.
        """
        result = list(file_lines)
        delta = 0
        for hunk in hunks:
            old_block = [t for k, t in hunk["lines"] if k in (" ", "-")]
            new_block = [t for k, t in hunk["lines"] if k in (" ", "+")]
            hint = hunk["old_start"] - 1 + delta
            idx = self._find_block(result, old_block, hint=hint)
            if idx is None:
                return None
            result = result[:idx] + new_block + result[idx + len(old_block) :]
            delta += len(new_block) - len(old_block)
        return result

    def _try_apply_patch_create_file_fallback(
        self,
        patch_text: str,
        path_hint: str | None,
        original_error: str,
        start_time: float,
    ) -> ToolResult | None:
        """Route a failed new-file patch to create_file.

        Returns None when the patch is NOT a clean creation (so the caller can
        try the modify_symbol path). Otherwise returns a ToolResult: ok on a
        successful create, or the ORIGINAL error enriched with the create
        failure (e.g. the file already exists, which means it wasn't really a
        creation and the LLM should retry as a modify).
        """
        nf = self._extract_new_file_target(patch_text, path_hint)
        if nf is None:
            return None
        import time as _time

        try:
            result = self._tool_create_file(
                {
                    "path": nf["file_path"],
                    "content": nf["content"],
                }
            )
        except Exception as e:
            logger.warning("apply_patch create_file fallback raised: %s", e, exc_info=True)
            return self._make_result(
                ok=False,
                content="",
                error=original_error,
                execution_time=_time.monotonic() - start_time,
                metadata={
                    "auto_fallback_attempted": "create_file",
                    "auto_fallback_exception": f"{type(e).__name__}: {e}",
                },
            )
        if result.ok:
            logger.info(
                "apply_patch auto-fallback to create_file succeeded: %s",
                nf["file_path"],
            )
            _meta = dict(result.metadata) if result.metadata else {}
            _meta.update(
                {
                    "auto_fallback_attempted": "create_file",
                    "auto_fallback_reason": "new_file_patch",
                    "file_path": nf["file_path"],
                }
            )
            return self._make_result(
                ok=True,
                content=(
                    f"Patch applied via create_file fallback (new file {nf['file_path']}).\n"
                    f"Original apply_patch error: {original_error}"
                ),
                execution_time=_time.monotonic() - start_time,
                metadata=_meta,
            )
        logger.info(
            "apply_patch auto-fallback to create_file failed: %s",
            result.error,
        )
        return self._make_result(
            ok=False,
            content="",
            error=(
                f"{original_error}\n\n[auto-fallback create_file also failed for {nf['file_path']}: {result.error}]"
            ),
            execution_time=_time.monotonic() - start_time,
            metadata={
                "auto_fallback_attempted": "create_file",
                "auto_fallback_failed": True,
                "auto_fallback_error": str(result.error)[:2000],
            },
        )

    def _try_apply_patch_multi_symbol_fallback(
        self,
        patch_text: str,
        path_hint: str | None,
        original_error: str,
        start_time: float,
    ) -> ToolResult | None:
        """Apply a multi-symbol patch as an atomic whole-file rewrite.

        Returns None when the patch is NOT a clean multi-symbol change (so the
        caller falls through to the single-symbol modify_symbol path). Otherwise
        writes the fully-reconstructed file (all-or-nothing) and returns the
        ToolResult.
        """
        ms = self._extract_multi_symbol_rewrite(patch_text, path_hint)
        if ms is None:
            return None
        import time as _time

        try:
            result = self._tool_create_file(
                {
                    "path": ms["file_path"],
                    "content": ms["new_src"],
                    "overwrite": True,
                }
            )
        except Exception as e:
            logger.warning("apply_patch multi-symbol fallback raised: %s", e, exc_info=True)
            return self._make_result(
                ok=False,
                content="",
                error=original_error,
                execution_time=_time.monotonic() - start_time,
                metadata={
                    "auto_fallback_attempted": "multi_symbol_rewrite",
                    "auto_fallback_exception": f"{type(e).__name__}: {e}",
                },
            )
        if result.ok:
            logger.info(
                "apply_patch auto-fallback to multi_symbol_rewrite succeeded: %s (%s)",
                ms["file_path"],
                ", ".join(ms["symbols"]),
            )
            _meta = dict(result.metadata) if result.metadata else {}
            _meta.update(
                {
                    "auto_fallback_attempted": "multi_symbol_rewrite",
                    "auto_fallback_reason": "multi_symbol_patch",
                    "file_path": ms["file_path"],
                    "symbols": ms["symbols"],
                }
            )
            _syn = self._run_syntax_check_for_file(ms["abs_path"])
            if not _syn.get("skipped"):
                _meta["syntax_check"] = _syn
            return self._make_result(
                ok=True,
                content=(
                    f"Patch applied via multi-symbol rewrite "
                    f"(symbols {', '.join(ms['symbols'])} in {ms['file_path']}).\n"
                    f"Original apply_patch error: {original_error}"
                ),
                execution_time=_time.monotonic() - start_time,
                metadata=_meta,
            )
        return self._make_result(
            ok=False,
            content="",
            error=(
                f"{original_error}\n\n"
                f"[auto-fallback multi-symbol rewrite also failed for "
                f"{ms['file_path']}: {result.error}]"
            ),
            execution_time=_time.monotonic() - start_time,
            metadata={
                "auto_fallback_attempted": "multi_symbol_rewrite",
                "auto_fallback_failed": True,
                "auto_fallback_error": str(result.error)[:2000],
            },
        )

    def _patch_path_resolvable(self, patch_text: str, path_hint: str | None) -> bool:
        """True when the patch text itself carries a resolvable target path.

        Distinguishes "no target path at all" (headerless @@-only patch with no
        caller hint) from the other modify_symbol ineligibility reasons, so the
        skip metadata doesn't misattribute a path-resolution failure to
        ``not_single_python_symbol``.
        """
        if path_hint:
            return True
        try:
            return any(f["file"] for f in self._parse_unified_diff_files(patch_text))
        except Exception:
            return True  # parse failure ≠ missing path — keep the generic reason

    def _try_apply_patch_modify_symbol_fallback(
        self,
        patch_text: str,
        path_hint: str | None,
        original_error: str,
        start_time: float,
    ) -> ToolResult:
        """Attempt modify_symbol as a fallback for a failed unified-diff patch.

        Returns a ToolResult. On success, metadata.auto_fallback_attempted marks
        the path taken. On ineligibility or failure, returns the ORIGINAL error
        so the caller (LLM) sees why the patch failed.
        """
        # ── New-file patch → create_file (no symbol to modify) ──
        nf = self._try_apply_patch_create_file_fallback(
            patch_text,
            path_hint,
            original_error,
            start_time,
        )
        if nf is not None:
            return nf

        # ── Multi-symbol patch → atomic whole-file rewrite ──
        ms = self._try_apply_patch_multi_symbol_fallback(
            patch_text,
            path_hint,
            original_error,
            start_time,
        )
        if ms is not None:
            return ms

        target = self._extract_modify_symbol_target(patch_text, path_hint)
        if target is None:
            # Not eligible — return original error with a skip marker.
            skip_reason = (
                "not_single_python_symbol" if self._patch_path_resolvable(patch_text, path_hint) else "no_target_path"
            )
            return self._make_result(
                ok=False,
                content="",
                error=original_error,
                metadata={"auto_fallback_attempted": None, "auto_fallback_skipped_reason": skip_reason},
            )

        try:
            from external_llm.agent.symbol_modify_tool import modify_symbol as _do_modify

            sec = self._secure_path(target["file_path"], confine=True)
            if sec is None:
                return self._make_result(ok=False, content="", error=f"Path traversal blocked: {target['file_path']}")
            abs_path = str(sec)
            success, diff_or_error, _new_content = _do_modify(
                abs_path,
                target["symbol"],
                target["code"],
                repo_root=str(self._effective_repo_root),
            )
            import time as _time

            execution_time = _time.monotonic() - start_time
            if success:
                rel_path = os.path.relpath(abs_path, str(self._effective_repo_root))
                _meta = {
                    "file_path": rel_path,
                    "symbol": target["symbol"],
                    "auto_fallback_attempted": "modify_symbol",
                    "auto_fallback_reason": target["reason"],
                    "diff_preview": diff_or_error[:25000] if diff_or_error else "",
                    "changed": True,
                }
                _syn = self._run_syntax_check_for_file(abs_path)
                if not _syn.get("skipped"):
                    _meta["syntax_check"] = _syn
                logger.info(
                    "apply_patch auto-fallback to modify_symbol succeeded: %s@%s",
                    rel_path,
                    target["symbol"],
                )
                return self._make_result(
                    ok=True,
                    content=(
                        f"Patch applied via modify_symbol fallback (symbol '{target['symbol']}' in {rel_path}).\n"
                        f"Original apply_patch error: {original_error}\n"
                        f"Diff:\n{diff_or_error}"
                    ),
                    execution_time=execution_time,
                    metadata=_meta,
                )
            logger.info(
                "apply_patch auto-fallback to modify_symbol failed: %s",
                diff_or_error,
            )
            return self._make_result(
                ok=False,
                content="",
                error=(
                    f"{original_error}\n\n"
                    f"[auto-fallback modify_symbol also failed for "
                    f"{target['file_path']}@{target['symbol']}: {diff_or_error}]"
                ),
                execution_time=execution_time,
                metadata={
                    "auto_fallback_attempted": "modify_symbol",
                    "auto_fallback_failed": True,
                    "auto_fallback_error": str(diff_or_error)[:2000],
                },
            )
        except Exception as e:
            logger.warning("apply_patch auto-fallback raised: %s", e, exc_info=True)
            import time as _time

            return self._make_result(
                ok=False,
                content="",
                error=original_error,
                execution_time=_time.monotonic() - start_time,
                metadata={
                    "auto_fallback_attempted": "modify_symbol",
                    "auto_fallback_exception": f"{type(e).__name__}: {e}",
                },
            )

    def _tool_apply_patch(self, args: dict[str, Any]) -> ToolResult:
        """Thin wrapper adding contract metadata to EVERY apply_patch success.

        Two stamps, one place:

        1. ``touched_files`` — the auto-observation pipeline
           (agent_turn_pipeline._process_post_tool_turn) scopes its follow-up
           ``git diff`` to the files a patch touched, reading this metadata key
           (documented at that call site as the apply_patch/write_plan
           contract). The internal success paths (git-apply, synthesize-and-
           apply, legacy text apply, modify_symbol fallback) return different
           metadata shapes, so stamping at this single post-impl point is the
           only way all of them are covered. setdefault semantics: an inner
           path that already recorded a more precise list keeps it.
        2. unverifiable-hunk notice (context-free hunks placed on trust).
        """
        result = self._tool_apply_patch_impl(args)
        if not result.ok:
            return result
        try:
            _touched = extract_files_from_patch(args.get("patch", "") or "")
            _path_arg = args.get("path")
            if not _touched and isinstance(_path_arg, str) and _path_arg.strip():
                _touched = [_path_arg.strip()]
            if _touched and not (result.metadata or {}).get("touched_files"):
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["touched_files"] = list(dict.fromkeys(_touched))
        except Exception as _e:
            logger.debug("touched_files stamp failed: %s", _e)
        try:
            from external_llm.patch_engine import PatchEngine as _PE  # noqa: N814 — private lazy-import alias

            unverifiable = _PE.context_free_hunks(args.get("patch", "") or "")
        except Exception as _e:
            logger.debug("context-free hunk scan failed: %s", _e)
            return result
        if not unverifiable:
            return result
        # Report, don't refuse — see PatchEngine.context_free_hunks. The agent
        # is the only thing left that can notice a misplaced insert here, so it
        # has to be told which hunks were placed on trust.
        if result.metadata is None:
            result.metadata = {}
        result.metadata["unverifiable_hunks"] = list(unverifiable)
        result.content = (result.content or "") + (
            "\n\nNOTE: {n} hunk(s) carried no context lines, so they were placed "
            "by line number alone and their location could NOT be verified: {h}. "
            "If the line numbers were stale the change landed somewhere else and "
            "may still parse — re-read the affected region to confirm."
        ).format(n=len(unverifiable), h=", ".join(unverifiable))
        return result

    def _refuse_session_edited(self, touched: list[str], start_time: float) -> ToolResult | None:
        """Opt D session-edit guard — shared by all three apply entry points.

        Returns a refusal ToolResult (ok=False) when at least one path in
        ``touched`` was already written this session by a text-editing tool
        (edit_text / modify_symbol / edit_ast / anchor_edit, tracked in
        ``_text_edited_files``); returns None when nothing is session-edited so
        the caller proceeds with the apply.

        WHY refuse: apply_patch / diff_apply reconstructs hunk context from HEAD,
        not the working tree. On a freshly-edited target (skip_3way=True) the
        apply chain reverts the file to HEAD — silently deleting the session edit
        while returning ok=False. The guard MUST run before any apply so the
        working tree is never mutated; post-apply detection is meaningless (the
        revert itself makes the file match HEAD again). Keeping the message +
        metadata here guarantees the main entry, the diff_apply path and the pure
        git apply path all refuse identically.
        """
        import time as _time

        _session = [p for p in touched if self._norm_repo_rel(p) in self._text_edited_files]
        if not _session:
            return None
        return self._make_result(
            ok=False,
            content="",
            error=(
                "apply_patch refused: target file(s) were already edited this session "
                "via edit_text / modify_symbol / edit_ast / anchor_edit — those edits "
                "live in the working tree but apply_patch would revert to HEAD on "
                "conflict (silently losing them): "
                + ", ".join(_session)
                + ". Continue editing these files with a text-editing tool instead"
                " — edit_text (exact string match) always works here, and is the"
                " right retry when modify_symbol has just failed on the same file."
            ),
            execution_time=_time.monotonic() - start_time,
            metadata={
                "refused_dirty_files": _session,
                "reason": "session_text_edit_overwrite_risk",
            },
        )

    def _acquire_edit_leases(self, paths) -> None:
        """F1: stake this session's cross-process edit lease on ``paths``.

        Called after a successful write so parallel asicode sessions (other
        terminals / subagent workers) see these files as actively WIP before
        their own write tools touch them — see
        :mod:`external_llm.common.edit_lease`. Never raises, never blocks: a
        lease guards visibility, not availability (the acquire is fail-open
        and a no-op without a repo_root).
        """
        rr = str(getattr(self, "repo_root", "") or "")
        if not rr or not paths:
            return
        try:
            from external_llm.common.edit_lease import acquire_edit_lease

            for p in paths:
                acquire_edit_lease(rr, p)
        except Exception:
            logger.debug(
                "<module>::WriteToolsPatchMixin::_acquire_edit_leases suppressed",
                exc_info=True,
            )

    def _refuse_foreign_leased(self, touched, start_time: float | None = None) -> ToolResult | None:
        """F1 cross-process edit-lease guard — sibling of _refuse_session_edited.

        Returns a refusal ToolResult (ok=False) when at least one path in
        ``touched`` carries a LIVE edit lease owned by a DIFFERENT asicode
        process (parallel terminal session or subagent worker) — i.e. that
        session has recent uncommitted WIP on the file. Documented failure
        modes this prevents: apply_patch's HEAD-context revert silently
        deleting the other session's edits, AUTO-REPAIR re-attaching code it
        deleted, ruff/AUTO-fix reformatting a mid-edit file.

        Returns None when nothing conflicts so the caller proceeds. Fail-open:
        any lease read/parse failure, empty repo_root, or
        ``ASICODE_EDIT_LEASES=0`` means "no conflict". The escape hatch is
        deliberate and stated in the error: delete the lease file (path in
        metadata.foreign_lease_conflicts[].lease_file) once the other session
        is confirmed done, or disable the guard via the env var.
        """
        import time as _time

        rr = str(getattr(self, "repo_root", "") or "")
        if not rr:
            return None
        try:
            from external_llm.common.edit_lease import find_live_foreign_leases

            conflicts = find_live_foreign_leases(rr, touched)
        except Exception:
            logger.debug(
                "<module>::WriteToolsPatchMixin::_refuse_foreign_leased suppressed",
                exc_info=True,
            )
            return None
        if not conflicts:
            return None
        _files = ", ".join(str(c.get("path", "?")) for c in conflicts)
        _owners = "; ".join(
            "pid {pid} on {host} (last edit {age:.0f}s ago)".format(
                pid=c.get("pid", "?"), host=c.get("host", "?"), age=c.get("age_s", 0.0)
            )
            for c in conflicts
        )
        return self._make_result(
            ok=False,
            content="",
            error=(
                "Write refused: live edit lease from another asicode session on: "
                + _files
                + ". Owner: "
                + _owners
                + ". That session likely has uncommitted WIP on these files; writing "
                "now risks silently clobbering its edits (HEAD-context revert, "
                "AUTO-REPAIR resurrecting deleted code, mid-edit reformatting). Let "
                "the other session commit or finish first. If you have CONFIRMED it "
                "is done with these files, remove the lease file(s) listed in "
                "metadata.foreign_lease_conflicts[].lease_file, or disable the guard "
                "with ASICODE_EDIT_LEASES=0."
            ),
            execution_time=(_time.monotonic() - start_time) if start_time else 0.0,
            metadata={
                "foreign_lease_conflicts": conflicts,
                "reason": "foreign_edit_lease",
            },
        )

    def _append_applied_patch(self, record: str) -> None:
        """Guarded append to ``self._applied_patches`` — shared by ALL write entry points.

        write_plan / apply_patch / anchor_edit (this mixin) and edit_text /
        edit_file / modify_symbol (edit mixin) all record successful writes here so
        agent_loop can detect a real edit happened (non-empty list). Some hosts and
        test harnesses have no ``_applied_patches`` (or a read-only one), so the
        append must never fail the write tool itself. Single SSOT — never inline a
        raw append again; route every recording through this helper.
        """
        try:
            self._applied_patches.append(record)
        except (AttributeError, TypeError):
            logger.debug(
                "<module>::WriteToolsPatchMixin::_append_applied_patch:0 suppressed (AttributeError, TypeError)",
                exc_info=True,
            )

    def _tool_apply_patch_impl(self, args: dict[str, Any]) -> ToolResult:
        import time as _time

        start_time = _time.monotonic()
        args = self._recover_args_from_raw(args, ("patch", "path"))
        patch_text = args.get("patch", "")
        if not patch_text.strip():
            execution_time = _time.monotonic() - start_time
            _raw_hint = ""
            _raw = args.get("__raw_arguments", "")
            if isinstance(_raw, str) and len(_raw) > 10:
                _raw_hint = f" (raw args: {_raw[:120]})"
            return self._make_result(
                ok=False, content="", error=f"patch is empty{_raw_hint}", execution_time=execution_time
            )
        path = args.get("path")

        if isinstance(path, str):
            # Strip a leading repo_root so an absolute path from the model
            # becomes repo-relative. The prefix must end at a path separator:
            # a bare startswith also matched siblings that merely share the
            # text, mangling "/dev/repository/a.py" into "sitory/a.py" and
            # "/dev/repo-backup/x.py" into "-backup/x.py" — paths that then
            # fail with a "File not found" naming a file nobody asked for.
            repo_root_str = str(self._effective_repo_root).rstrip("/")
            if path == repo_root_str:
                path = ""
            elif path.startswith(repo_root_str + "/"):
                path = path[len(repo_root_str) :].lstrip("/")
            if path.startswith("/"):
                path = path.lstrip("/")
        if path is not None and not path.strip():
            path = None

        # ── Hard guard (MAIN entry point): reject BEFORE PatchEngine / diff_apply mutates
        # the working tree. The main path (_tool_apply_patch → PatchEngine.apply_patch →
        # diff_apply with skip_3way=True) reverts a DIRTY file to HEAD ("git checkout --")
        # while returning ok=False — silently DELETING uncommitted edits (e.g. just made
        # via edit_text / modify_symbol / anchor_edit this session). The fallback guard in
        # _apply_patch_text does NOT protect this path: PatchEngine is reached first.
        # Detect BEFORE any apply so the working tree is never mutated; post-apply
        # detection is meaningless (the revert itself makes the file match HEAD again).
        try:
            _mp_touched = extract_files_from_patch(patch_text)
        except Exception:
            _mp_touched = []
        if isinstance(path, str) and path and path not in _mp_touched:
            _mp_touched.append(path)
        # Opt D: refuse only files THIS SESSION already edited via a text-editing
        # tool (edit_text / modify_symbol / edit_ast / anchor_edit), tracked in
        # _text_edited_files. Those edits live in the working tree, but apply_patch /
        # diff_apply reconstructs hunk context from HEAD and, on a freshly-edited
        # target (skip_3way=True), _rollback() reverts the file to HEAD — silently
        # deleting the session edit. Unlike the prior git-dirty check, this does NOT
        # block user pre-existing uncommitted edits or unrelated dirty files in a
        # multi-file patch (less friction); it targets the agent's own consecutive
        # edits precisely.
        _refused = self._refuse_session_edited(_mp_touched, start_time)
        if _refused is not None:
            return _refused
        # F1 cross-process edit-lease guard: a parallel session's live lease on
        # any touched file means uncommitted WIP we must not clobber.
        _lease_refused = self._refuse_foreign_leased(_mp_touched, start_time)
        if _lease_refused is not None:
            return _lease_refused

        try:
            engine = PatchEngine(self._effective_repo_root)

            if not self._looks_like_unified_diff(patch_text):
                if path is None:
                    execution_time = _time.monotonic() - start_time
                    _raw_hint = ""
                    _raw = args.get("__raw_arguments", "")
                    if isinstance(_raw, str) and len(_raw) > 10:
                        _raw_hint = f" (raw args: {_raw[:120]})"
                    return self._make_result(
                        ok=False,
                        content="",
                        error=f"Non-diff patch input requires 'path' so PatchEngine can synthesize and apply it{_raw_hint}",
                        execution_time=execution_time,
                        metadata={
                            "reason": "missing_path_for_non_diff_input",
                            "source": "agent_apply_patch",
                        },
                    )

                patch_result = engine.synthesize_and_apply(patch_text, path, output_mode="auto")
                execution_time = _time.monotonic() - start_time

                if patch_result.success:
                    patch_record = (patch_result.metadata or {}).get("patch") or patch_text
                    if patch_record:
                        self._append_applied_patch(str(patch_record))
                    # F1: stake our lease on the synthesized-patch target so
                    # parallel sessions see it as actively WIP.
                    self._acquire_edit_leases(list(_mp_touched) + ([str(path)] if path else []))

                    _meta = dict(patch_result.metadata) if patch_result.metadata else {}
                    # P26-1: content-loss guard on the LIVE synthesize path —
                    # score the APPLIED patch (patch_record), not the raw
                    # non-diff input: the input has no hunks, so scoring it
                    # would always pass silently.
                    _applied_text = str(patch_record) if patch_record else patch_text
                    _ratio_warn = self._check_patch_content_ratio(_applied_text)
                    if _ratio_warn:
                        _meta["content_ratio_warning"] = _ratio_warn
                    _content = patch_result.patch_applied or "Patch applied successfully"
                    if _ratio_warn:
                        _content += f"\n{_ratio_warn}"
                    _syn = self._run_syntax_check_for_file(path)
                    if not _syn.get("skipped"):
                        _meta["syntax_check"] = _syn
                    return self._make_result(ok=True, content=_content, execution_time=execution_time, metadata=_meta)

                logger.warning(
                    "PatchEngine synthesize_and_apply failed for non-diff input; "
                    "falling back to legacy apply path. error=%s",
                    patch_result.error,
                )
                legacy_result = self._apply_patch_text(patch_text, path_hint=path)
                legacy_result.metadata.setdefault("fallback_from_patch_engine", True)
                legacy_result.metadata.setdefault("patch_engine_error", patch_result.error)
                if legacy_result.execution_time < 1e-9:
                    legacy_result.execution_time = execution_time
                return legacy_result

            context = PatchContext(
                original_request=None,
                file_content=None,
                llm_output=None,
                output_mode="auto",
                metadata={"source": "agent_apply_patch"},
            )
            patch_result = engine.apply_patch(patch_text, target_file=path, context=context)
            execution_time = _time.monotonic() - start_time

            if patch_result.success:
                patch_record = (patch_result.metadata or {}).get("patch") or patch_text
                if patch_record:
                    self._append_applied_patch(str(patch_record))
                # F1: stake our lease on every file this patch touched so
                # parallel sessions see them as actively WIP.
                self._acquire_edit_leases(list(_mp_touched) + ([str(path)] if path else []))

                _meta2 = dict(patch_result.metadata) if patch_result.metadata else {}
                # P26-1: content-loss guard on the MAIN PatchEngine branch —
                # before this the guard was only wired to the legacy
                # git-apply fallback, so the branch every apply_patch tool
                # call takes applied large removals silently.
                _ratio_warn = self._check_patch_content_ratio(patch_text)
                if _ratio_warn:
                    _meta2["content_ratio_warning"] = _ratio_warn
                _check_path = path or (patch_result.metadata.get("file") if patch_result.metadata else None)
                if _check_path:
                    _syn2 = self._run_syntax_check_for_file(_check_path)
                    if not _syn2.get("skipped"):
                        _meta2["syntax_check"] = _syn2
                _content_msg = patch_result.patch_applied or "Patch applied successfully"
                if _ratio_warn:
                    _content_msg += f"\n{_ratio_warn}"
                return self._make_result(ok=True, content=_content_msg, execution_time=execution_time, metadata=_meta2)
            # ── Auto-fallback: try modify_symbol for single-symbol patches ──
            # PatchEngine exhausted its repair ladder (plain apply, --3way,
            # tolerant, re-anchor, AST/symbol repair). For untracked or
            # freshly-edited files, `git apply --3way` cannot find the
            # pre-image blob, so AST-based modify_symbol is the only path
            # that works. Route single-Python-symbol patches there before
            # surfacing the failure to the LLM.
            _fb = self._try_apply_patch_modify_symbol_fallback(
                patch_text,
                path,
                patch_result.error or "Patch application failed",
                start_time,
            )
            if _fb.ok:
                self._append_applied_patch(str(patch_text))
                return _fb
            # Fallback ineligible or failed — return its enriched result
            # (preserves original error + skip/attempt metadata).
            _fb.metadata = {**(patch_result.metadata or {}), **_fb.metadata}
            _fb.metadata.setdefault("failure_class", "patch_apply_failed")
            # ── Stale-target diagnosis ──
            # The patch failed against the file as it is NOW. If the
            # agent wrote it from an earlier read, the target may have
            # changed (parallel editor / own earlier edits this
            # session). Attach the current file head so the LLM can
            # rewrite the patch against what is actually on disk
            # without an extra read round-trip. (No auto-retry here:
            # apply_patch has already exhausted its repair ladder and
            # a retry could double-apply a partially-landed patch.)
            if not _fb.ok:
                _snip = self._patch_failure_snippet(patch_text, path)
                if _snip:
                    _fb.error = (_fb.error or "").rstrip("\n") + "\n\n" + _snip
                    _fb.metadata["reread_snippet"] = True
        except Exception as e:
            logger.exception("Unexpected error in patch engine")
            # Same monotonic discipline as the ImportError branch above.
            execution_time = _time.monotonic() - start_time
            return self._make_result(
                ok=False,
                content="",
                error=f"Patch engine error: {type(e).__name__}: {e}",
                execution_time=execution_time,
                metadata={"error_type": "patch_engine_exception"},
            )
        else:
            return _fb

    def _apply_patch_text(self, patch_text: str, path_hint: str | None = None) -> ToolResult:
        """Shared internal method to apply a patch via git apply chain."""
        import time as _time

        start_time = _time.monotonic()

        # ── Hard guard: capture uncommitted-changes state BEFORE any apply ──
        # `git apply` (and diff_apply, which wraps it) reconstructs hunk context
        # from the HEAD blob, so a patch whose context matches HEAD can silently
        # overwrite pre-existing working-tree edits on a DIRTY file (e.g. one
        # just edited via edit_text / modify_symbol / anchor_edit this session).
        # We detect this BEFORE applying — post-apply detection is meaningless
        # (the patch itself makes the file differ from HEAD) — and REJECT the
        # patch outright so the working tree is never mutated. The caller must
        # continue editing such files with edit_text / modify_symbol.
        try:
            _pre_touched_files = extract_files_from_patch(patch_text)
        except Exception:
            _pre_touched_files = []
        # Opt D session-edit guard (mirrors the main entry guard in _tool_apply_patch).
        _refused = self._refuse_session_edited(_pre_touched_files, start_time)
        if _refused is not None:
            return _refused
        # F1 cross-process edit-lease guard (mirrors the main entry guard).
        _lease_refused = self._refuse_foreign_leased(_pre_touched_files, start_time)
        if _lease_refused is not None:
            return _lease_refused

        # diff_apply is a first-party root module — the import cannot fail.
        # The None guard stays: tests patch diff_apply.apply_patch to None to
        # exercise the pure git-apply chain below.
        from diff_apply import apply_patch

        if apply_patch is not None:
            ok, msg, _reason, details = apply_patch(self._effective_repo_root, patch_text, file_path_hint=path_hint)
            execution_time = _time.monotonic() - start_time
            if ok:
                # Track applied patch so agent_loop can detect successful writes
                # (applied_patches non-empty = "real edit happened", avoids false-success nudge)
                self._append_applied_patch(str(patch_text))
                # F1: stake our lease on every file this diff_apply touched.
                self._acquire_edit_leases(_pre_touched_files)
                # P26-1: content-loss guard on the diff_apply fast path too —
                # the legacy git-apply fallback below had the guard, but this
                # earlier return bypassed it entirely.
                ratio_warning = self._check_patch_content_ratio(patch_text)
                _meta = dict(details) if details else {}
                if ratio_warning:
                    _meta["content_ratio_warning"] = ratio_warning
                    msg = f"{msg}\n{ratio_warning}"
                return self._make_result(ok=True, content=msg, execution_time=execution_time, metadata=_meta)
            return self._make_result(ok=False, content="", error=msg, execution_time=execution_time, metadata=details)

        patch_norm = normalize_patch_text(patch_text)

        if "diff --git a/.." in patch_norm or "diff --git b/.." in patch_norm:
            return self._make_result(ok=False, content="", error="Unsafe path in patch (path traversal detected)")

        patch_clean = patch_norm
        try:
            patch_clean = _clean_diff(patch_norm, self.repo_root, file_path_hint=path_hint)
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Patch cleanup failed: {e}")

        # Private (mode 0o600), unpredictably-named temp file for the patch.
        # A fixed /tmp path created with default umask leaks source-code diffs
        # to other users on shared/multi-user systems. mkstemp creates the file
        # atomically with restrictive perms; subsequent open(patch_file, "w")
        # calls truncate and rewrite while preserving the 0o600 mode (open never
        # relaxes permissions of an existing file).
        try:
            _fd, patch_file = tempfile.mkstemp(suffix=".patch", prefix="asicode.")
            os.close(_fd)
        except OSError as _e:
            return self._make_result(ok=False, content="", error=f"Failed to create temp patch file: {_e}")
        try:
            if not patch_clean.strip():
                try:
                    synthesized = None
                    if PatchEngine is not None and path_hint:
                        try:
                            engine = PatchEngine(self._effective_repo_root)
                            synthesized = engine._salvage_small_model_output(patch_text, path_hint)
                        except Exception as e:
                            logger.debug("PatchEngine salvage failed: %s", e)
                            synthesized = None
                    if synthesized and synthesized.strip():
                        try:
                            with open(patch_file, "w", encoding="utf-8") as fh:
                                fh.write(synthesized)
                        except OSError as e:
                            return self._make_result(
                                ok=False, content="", error=f"Failed to write synthesized patch file: {e}"
                            )
                        # Validate synthesized patch with git apply --check BEFORE accepting it
                        check = subprocess.run(
                            ["git", "apply", "--check", patch_file],
                            cwd=self._effective_repo_root,
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        if check.returncode == 0:
                            patch_clean = synthesized
                            logger.info("Recovered empty normalized patch via small-model synthesizer")
                        else:
                            return self._make_result(
                                ok=False,
                                content="",
                                error="empty diff after cleaning (salvage failed git apply --check)",
                                metadata={
                                    "check_stderr": (check.stderr or "").strip(),
                                },
                            )
                    else:
                        return self._make_result(ok=False, content="", error="empty diff after cleaning")
                except Exception as e:
                    logger.debug("Pre-check synthesizer failed: %s", e)
                    return self._make_result(ok=False, content="", error="empty diff after cleaning")
            patch_sha256 = hashlib.sha256(patch_clean.encode("utf-8")).hexdigest()[:16]
            patch_len = len(patch_clean)
            try:
                with open(patch_file, "w", encoding="utf-8") as fh:
                    fh.write(patch_clean)
            except OSError as e:
                return self._make_result(ok=False, content="", error=f"Failed to write patch file: {e}")

            _head_lines = patch_clean.lstrip().splitlines()[:8] if patch_clean else []
            _has_git_header = "diff --git " in patch_clean
            _looks_like_ab_paths = any(s.startswith("--- a/") for s in _head_lines) and any(
                s.startswith("+++ b/") for s in _head_lines
            )
            _needs_p1 = (not _has_git_header) and _looks_like_ab_paths

            _apply_base = ["git", "apply"]
            if _needs_p1:
                _apply_base.append("-p1")

            use_ignore_ws = False
            try:
                check = subprocess.run(
                    [*_apply_base, "--check", patch_file],
                    cwd=self._effective_repo_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if check.returncode != 0:
                    check_ws = subprocess.run(
                        [*_apply_base, "--check", "--ignore-whitespace", patch_file],
                        cwd=self._effective_repo_root,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if check_ws.returncode == 0:
                        use_ignore_ws = True
                        logger.info("Patch check passed with --ignore-whitespace; will apply with flag")
                    else:
                        # --check failed even with --ignore-whitespace. Last resort:
                        # small-model diff synthesizer. If it can't produce a patch
                        # that passes --check, fail HERE. (A previous regression put
                        # this return inside the synthesizer's `except` block, so a
                        # normally-failing check fell through to the real `git apply`;
                        # it also NameError'd when the ws-check had passed.)
                        failure_analysis = self._analyze_patch_failure(patch_clean, check.stderr or "")
                        _salvaged = False
                        try:
                            synthesized = None
                            if PatchEngine is not None and path_hint:
                                try:
                                    engine = PatchEngine(self._effective_repo_root)
                                    synthesized = engine._salvage_small_model_output(patch_text, path_hint)
                                except Exception as e:
                                    logger.debug("PatchEngine salvage failed: %s", e)
                                    synthesized = None
                            if synthesized:
                                logger.info("Patch synthesizer activated for small-model diff repair")
                                with open(patch_file, "w", encoding="utf-8") as fh:
                                    fh.write(synthesized)
                                retry = subprocess.run(
                                    [*_apply_base, "--check", patch_file],
                                    cwd=self._effective_repo_root,
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                    check=False,
                                )
                                if retry.returncode == 0:
                                    patch_clean = synthesized
                                    _salvaged = True
                                    logger.info("Synthesized diff accepted by git apply")
                                else:
                                    logger.debug(
                                        "Synthesized diff still invalid: %s",
                                        (retry.stderr or retry.stdout or "").strip(),
                                    )
                        except Exception as e:
                            logger.debug("Small-model diff synthesizer failed: %s", e)

                        if not _salvaged:
                            return self._make_result(
                                ok=False,
                                content="",
                                error=failure_analysis.get("error_message", "git apply --check failed"),
                                metadata={
                                    "patch_file": patch_file,
                                    "patch_sha256": patch_sha256,
                                    "patch_len": patch_len,
                                    "check_stderr": (check.stderr or "").strip(),
                                    "failure_analysis": failure_analysis,
                                },
                            )
            except subprocess.TimeoutExpired:
                return self._make_result(
                    ok=False,
                    content="",
                    error="git apply --check timeout after 30 seconds",
                    metadata={"patch_file": patch_file, "timeout": True},
                )
            except Exception as e:
                return self._make_result(
                    ok=False, content="", error=f"git apply --check error: {e}", metadata={"patch_file": patch_file}
                )

            # ── Pre-apply snapshot for rollback ────────────────────────────
            # Extract file paths from diff BEFORE apply, snapshot their content.
            import os as _os_snap

            _pre_touched: list[str] = extract_files_from_patch(patch_text)

            _pre_apply_snapshot: dict[str, str] = {}
            for _tf_snap in _pre_touched:
                _abs_snap = _os_snap.path.join(self._effective_repo_root, _tf_snap)
                if _os_snap.path.isfile(_abs_snap):
                    try:
                        with open(_abs_snap, encoding="utf-8", errors="replace") as _fsnap:
                            _pre_apply_snapshot[_abs_snap] = _fsnap.read()
                    except OSError:
                        logger.debug(
                            "<module>::WriteToolsPatchMixin::_apply_patch_text:15 suppressed OSError", exc_info=True
                        )

            # Hard guard: capture uncommitted-changes state BEFORE apply (post-apply
            # detection is meaningless — the patch itself makes files differ from
            # HEAD) and REJECT so the working tree is never mutated. Mirrors the
            # diff_apply main path.
            # Opt D session-edit guard (mirrors the main entry guard in _tool_apply_patch).
            # NB: this pure `git apply` path has no _rollback, so it is safe even for dirty
            # files — but we still refuse session-edited targets to give a clear "use
            # edit_text instead" message rather than a confusing context-mismatch failure.
            _refused = self._refuse_session_edited(_pre_touched, start_time)
            if _refused is not None:
                return _refused
            # F1 cross-process edit-lease guard (mirrors the main entry guard).
            _lease_refused = self._refuse_foreign_leased(_pre_touched, start_time)
            if _lease_refused is not None:
                return _lease_refused

            apply_cmd = list(_apply_base)
            if use_ignore_ws:
                apply_cmd.append("--ignore-whitespace")
            apply_cmd.append(patch_file)
            try:
                apply_proc = subprocess.run(
                    apply_cmd,
                    cwd=self._effective_repo_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if apply_proc.returncode != 0:
                    failure_analysis = self._analyze_patch_failure(patch_clean, apply_proc.stderr or "")
                    return self._make_result(
                        ok=False,
                        content="",
                        error=failure_analysis.get("error_message", "git apply failed"),
                        metadata={
                            "patch_file": patch_file,
                            "patch_sha256": patch_sha256,
                            "patch_len": patch_len,
                            "apply_stderr": (apply_proc.stderr or "").strip(),
                            "failure_analysis": failure_analysis,
                        },
                    )
            except subprocess.TimeoutExpired:
                return self._make_result(
                    ok=False,
                    content="",
                    error="git apply timeout after 30 seconds",
                    metadata={"patch_file": patch_file, "timeout": True},
                )
            except Exception as e:
                return self._make_result(
                    ok=False, content="", error=f"git apply error: {e}", metadata={"patch_file": patch_file}
                )

            ratio_warning = self._check_patch_content_ratio(patch_clean)

            touched: list[str] = []
            try:
                touched = extract_touched_files_from_diff(patch_clean)
            except (ValueError, TypeError, AttributeError):
                logger.debug(
                    "<module>::WriteToolsPatchMixin::_apply_patch_text:18 suppressed (ValueError, TypeError, AttributeError)",
                    exc_info=True,
                )

            if not touched:
                try:
                    for line in patch_clean.splitlines():
                        if line.startswith("diff --git "):
                            parts = line.split()
                            if len(parts) >= 4:
                                a_path = _PATCH_PATH_PREFIX_RE.sub("", parts[2])
                                b_path = _PATCH_PATH_PREFIX_RE.sub("", parts[3])
                                rel = (b_path or a_path).strip().replace("\\", "/").lstrip("/")
                                if rel and rel not in touched:
                                    touched.append(rel)
                        elif line.startswith(("+++ b/", "--- a/")):
                            rel = line[6:].strip().replace("\\", "/").lstrip("/")
                            if rel and rel not in touched:
                                touched.append(rel)
                except (AttributeError, TypeError):
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_apply_patch_text:19 suppressed (AttributeError, TypeError)",
                        exc_info=True,
                    )

            # ── Post-apply syntax validation + snapshot-based rollback ────────
            _syntax_errors: list[str] = []
            for _tf_chk in touched or _pre_touched:
                _abs_chk = _os_snap.path.join(self._effective_repo_root, _tf_chk)
                if LanguageId.from_path(_abs_chk) is LanguageId.PYTHON and _os_snap.path.isfile(_abs_chk):
                    try:
                        with open(_abs_chk, encoding="utf-8", errors="replace") as _fchk:
                            _post_content = _fchk.read()
                        compile_quiet(_post_content, _tf_chk, "exec")
                    except SyntaxError as _se:
                        _syntax_errors.append(f"{_tf_chk}: {_se}")
                    except Exception as _exc:
                        logger.debug("Post-apply compile() check raised non-SyntaxError: %s", _exc)

            if _syntax_errors:
                # Rollback using pre-apply snapshot (git-independent)
                for _snap_path, _snap_content in _pre_apply_snapshot.items():
                    try:
                        with open(_snap_path, "w", encoding="utf-8") as _froll:
                            _froll.write(_snap_content)
                    except Exception as _roll_exc:
                        logger.debug("Rollback write failed for %s: %s", _snap_path, _roll_exc)
                logger.warning(
                    "Patch introduced syntax errors — rolled back from snapshot: %s",
                    _syntax_errors,
                )
                return self._make_result(
                    ok=False,
                    content="",
                    error=f"Patch introduced syntax errors (rolled back): {'; '.join(_syntax_errors)}",
                    metadata={"syntax_errors": _syntax_errors, "rolled_back": True},
                )

            self._append_applied_patch(patch_clean)
            # F1: stake our lease on every file the pure git-apply path touched.
            self._acquire_edit_leases(_pre_touched)
            if touched:
                self._invalidate_cache_after_write(touched)
            content_msg = f"Patch applied successfully. Touched files: {', '.join(touched) or 'unknown'}"
            if ratio_warning:
                content_msg += f"\n{ratio_warning}"
            _fallback_meta = {"touched_files": touched, "patch": patch_clean, "content_ratio_warning": ratio_warning}
            return self._make_result(
                ok=True,
                content=content_msg,
                metadata=_fallback_meta,
            )
        finally:
            try:
                os.unlink(patch_file)
            except OSError:
                logger.debug("<module>::WriteToolsPatchMixin::_apply_patch_text:23 suppressed OSError", exc_info=True)

    def _analyze_patch_failure(self, patch_text: str, git_error: str) -> dict[str, Any]:
        import re

        # Lazy import: analysis.parse_cache is a heavy module (tree-sitter
        # grammars); only failure paths pay the cost.
        from ...analysis import parse_cache
        from ..failure_context import analyze_failure

        read_source = parse_cache.read_source

        failure_ctx = analyze_failure(
            stage="git_apply_check",
            raw_text=git_error,
        )

        file_path = None
        hunks = []
        # Reused across the two context-mismatch analysis branches below; the
        # process-wide parse_cache makes the second branch a cache hit instead
        # of a second disk read.
        file_source: str | None = None

        lines = patch_text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("--- a/"):
                file_path = line[6:].split("\t")[0].strip()
                if file_path == "/dev/null":
                    file_path = None
                if i + 1 < len(lines) and lines[i + 1].startswith("+++ b/"):
                    new_file_path = lines[i + 1][6:].split("\t")[0].strip()
                    if new_file_path != "/dev/null":
                        file_path = new_file_path
                break
            if line.startswith("+++ b/"):
                file_path = line[6:].split("\t")[0].strip()
                if file_path == "/dev/null":
                    file_path = None
                break
            if line.startswith("diff --git a/"):
                parts = line.split()
                if len(parts) >= 4:
                    b_path = parts[3]
                    file_path = b_path[2:] if b_path.startswith("b/") else b_path
                    break

        current_hunk = None

        for line in lines:
            hunk_match = _HUNK_HEADER_RE.match(line)
            if hunk_match:
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = {
                    "old_start": int(hunk_match.group(1)),
                    "old_lines": int(hunk_match.group(2)),
                    "new_start": int(hunk_match.group(3)),
                    "new_lines": int(hunk_match.group(4)),
                    "context_lines": [],
                    "original_lines": [],
                }
            elif current_hunk:
                if line.startswith(" "):
                    content = line[1:]
                    current_hunk["context_lines"].append(content)
                    current_hunk["original_lines"].append(("context", content))
                elif line.startswith("-"):
                    content = line[1:]
                    current_hunk["original_lines"].append(("remove", content))
                elif line.startswith("+"):
                    content = line[1:]
                    current_hunk["original_lines"].append(("add", content))

        if current_hunk:
            hunks.append(current_hunk)

        error_lower = git_error.lower()
        reason = "unknown"
        hint = ""
        conflicting_lines = []

        if "already applied" in error_lower or "already exists" in error_lower or "file exists" in error_lower:
            reason = "already_applied"
            hint = "Patch appears to be already applied to the file."
        elif "corrupt patch" in error_lower or "patch format" in error_lower or "unrecognized input" in error_lower:
            reason = "offset_error"
            hint = "Patch format is corrupt or malformed."
        elif "does not apply" in error_lower or "patch failed" in error_lower or "hunk failed" in error_lower:
            reason = "context_mismatch"
            hint = "Patch context does not match the current file content."

            line_match = re.search(r"at line\s+(\d+)", git_error)
            if not line_match:
                line_match = re.search(r"line\s+(\d+)", git_error)
            if not line_match:
                line_match = re.search(r"hunk\s+(\d+)", git_error)

            if line_match:
                conflicting_line = int(line_match.group(1))
                conflicting_lines.append(conflicting_line)

                if file_path:
                    # Reuse the process-wide parse_cache (single-stat, cached
                    # read) instead of re-reading the target file from disk.
                    # read_source() decodes UTF-8 with errors="replace" — for
                    # well-formed UTF-8 files this is identical to the old
                    # strict-utf8 read, and for undecodable files it surfaces
                    # U+FFFD markers in the hint instead of failing silently.
                    file_source = read_source(str(Path(self.repo_root) / file_path))
                    if file_source is not None:
                        file_lines_list = file_source.split("\n")

                        if 0 <= conflicting_line - 1 < len(file_lines_list):
                            actual_line = file_lines_list[conflicting_line - 1]

                            expected_line = None
                            for hunk in hunks:
                                if hunk["old_start"] <= conflicting_line <= hunk["old_start"] + hunk["old_lines"]:
                                    offset = conflicting_line - hunk["old_start"]
                                    context_counter = 0
                                    for line_type, content in hunk["original_lines"]:
                                        if line_type == "context":
                                            if context_counter == offset:
                                                expected_line = content
                                                break
                                            context_counter += 1
                                        elif line_type == "remove":
                                            context_counter += 1
                                    break

                            if expected_line:
                                if len(expected_line) > 100:
                                    expected_line = expected_line[:97] + "..."
                                if len(actual_line) > 100:
                                    actual_line = actual_line[:97] + "..."
                                hint = f"Context mismatch at line ~{conflicting_line}. Expected: '{expected_line}' but found: '{actual_line}'"
                            else:
                                ctx_start = max(0, conflicting_line - 3)
                                ctx_end = min(len(file_lines_list), conflicting_line + 2)
                                ctx = "\n".join(f"{i + 1}: {file_lines_list[i]}" for i in range(ctx_start, ctx_end))
                                hint = f"Patch failed at line {conflicting_line}. File context around line:\n{ctx}"
                else:
                    file_source = None
        elif "no such file" in error_lower or "cannot stat" in error_lower:
            reason = "file_not_found"
            hint = f"Target file not found: {file_path or 'unknown'}{self._suggest_missing_paths(file_path or '')}"

        file_context_snippet: str | None = None
        if reason == "context_mismatch" and file_path and hunks:
            # The line_match branch above populated file_source; without a
            # line number the read is done here (single parse_cache lookup).
            if file_source is None:
                file_source = read_source(str(Path(self.repo_root) / file_path))
            if file_source is not None:
                try:
                    # Reuse the source read in the line_match branch above (parse_cache
                    # guarantees both reads observe the same file version).
                    file_lines_list = file_source.splitlines()
                    hunk_start = hunks[0]["old_start"]
                    ctx_start = max(0, hunk_start - 5)
                    ctx_end = min(len(file_lines_list), hunk_start + hunks[0].get("old_lines", 10) + 5)
                    ctx_lines = [
                        f"{ctx_start + j + 1:4d}: {file_lines_list[ctx_start + j]}" for j in range(ctx_end - ctx_start)
                    ]
                    file_context_snippet = "\n".join(ctx_lines)
                except (IndexError, TypeError):
                    logger.debug(
                        "<module>::WriteToolsPatchMixin::_analyze_patch_failure:2 suppressed (IndexError, TypeError)",
                        exc_info=True,
                    )

        parts = [f"Patch failed ({reason}): {hint or git_error.strip()[:200]}"]

        if reason in ["context_mismatch", "offset_error", "unknown"]:
            parts.append("\n**Patch guidance**:")
            parts.append("- Provide exact context lines from the file (copy-paste, don't paraphrase)")
            parts.append("- For simple changes, use format: `-old line\\n+new line`")
            parts.append("\n**Example of correct unified diff format**:")
            parts.append("```diff")
            parts.append("diff --git a/path/to/file.py b/path/to/file.py")
            parts.append("--- a/path/to/file.py")
            parts.append("+++ b/path/to/file.py")
            parts.append("@@ -10,7 +10,7 @@")
            parts.append(" def old_function():")
            parts.append("     print('old')")
            parts.append("-    return 1")
            parts.append("+    return 2")
            parts.append("```")

            if "```" in patch_text and "diff --git" not in patch_text:
                parts.append(
                    "\n**Detected issue**: Your patch contains markdown code fences but not proper diff format."
                )
                parts.append("**Try this instead**: Remove the ``` markers and use unified diff format above.")

            if "before:" in patch_text.lower() or "after:" in patch_text.lower():
                parts.append("\n**Detected issue**: Your patch uses 'before:/after:' notation.")
                parts.append("**Try this instead**: Convert to unified diff format with exact context lines.")

        if file_context_snippet:
            parts.append(
                f"\n**Actual file content at patch location** (copy exact text for retry):\n"
                f"```\n{file_context_snippet}\n```"
            )
        elif reason == "file_not_found":
            parts = [
                f"Patch failed: {hint}. "
                f"For new files use the 'create_file' tool, or start the patch with "
                f"'--- /dev/null' to indicate new file creation."
            ]

        if not patch_text.strip().startswith("diff --git") and not patch_text.strip().startswith("---"):
            parts.append("\n**Patch format issue**: Your patch doesn't start with standard diff headers.")
            parts.append("**Try this**: Start with `diff --git a/file/path b/file/path` or `--- a/file/path`")

        error_message = "\n".join(parts)

        result = {
            "reason": reason,
            "hint": hint,
            "conflicting_lines": conflicting_lines,
            "error_message": error_message,
            "file_path": file_path,
            "hunk_count": len(hunks),
            "file_context_snippet": file_context_snippet,
        }

        if failure_ctx:
            result["failure_context"] = {
                "stage": failure_ctx.stage,
                "type": failure_ctx.type,
                "message": failure_ctx.message,
                "tags": failure_ctx.tags,
                "fingerprint": failure_ctx.fingerprint,
            }

        return result

    def _check_patch_content_ratio(self, patch_text: str) -> str | None:
        """Detect accidental whole-file wipes in LLM-generated patches.

        File attribution resolves from ``diff --git`` OR the ``--- a/`` /
        ``+++ b/`` header pair (plain unified diffs — patch_synthesizer /
        _salvage_small_model_output output — have no ``diff --git`` line,
        P26-2).  The pre-image line total is accumulated from hunk headers
        (``@@ -S[,C] @@`` → C, default 1) instead of the post-apply disk
        size, so a wipe that leaves < 20 lines on disk is still caught.
        Removed lines are counted hunk-aware: inside a hunk every ``-``
        line is content — SQL comments / CSS custom properties removed as
        ``--…`` render as ``---…`` diff lines and must NOT be mistaken for
        file headers (that misparse silently undercounts a wipe into
        silence).
        """
        warnings_out: list[str] = []
        current_file: str | None = None
        pre_image: dict[str, int] = {}
        removals: dict[str, int] = {}
        additions: dict[str, int] = {}
        in_hunk = False

        for line in patch_text.splitlines():
            if line.startswith("@@"):
                in_hunk = True
                if current_file is not None:
                    m = re.match(r"^@@ -(\d+)(?:,(\d+))?", line)
                    if m:
                        pre_image[current_file] = pre_image.get(current_file, 0) + (
                            int(m.group(2)) if m.group(2) else 1
                        )
                continue
            if line.startswith("diff --git "):
                in_hunk = False
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                    pre_image.setdefault(current_file, 0)
                    removals.setdefault(current_file, 0)
                    additions.setdefault(current_file, 0)
                continue
            if line.startswith(("--- a/", "--- b/", "--- /dev/null")):
                in_hunk = False
                _p = line[4:]
                if not _p.startswith("/dev/null"):
                    current_file = _p[2:] if _p.startswith(("a/", "b/")) else _p
                    pre_image.setdefault(current_file, 0)
                    removals.setdefault(current_file, 0)
                    additions.setdefault(current_file, 0)
                continue
            if line.startswith(("+++ a/", "+++ b/")):
                in_hunk = False
                continue
            if in_hunk and current_file is not None:
                if line.startswith("-"):
                    removals[current_file] = removals.get(current_file, 0) + 1
                elif line.startswith("+"):
                    additions[current_file] = additions.get(current_file, 0) + 1

        for fpath, removed in removals.items():
            if removed < 10:
                continue
            total = pre_image.get(fpath, 0)
            if total < 20:
                continue
            added = additions.get(fpath, 0)
            abs_fp = Path(self.repo_root) / fpath
            if not abs_fp.is_file():
                continue
            ratio = removed / total
            if ratio > 0.7 and added < removed * 0.3:
                warnings_out.append(
                    f"CONTENT LOSS WARNING: {fpath} — removing {removed}/{total} lines "
                    f"({ratio:.0%}) but only adding {added}. "
                    "Verify this is intentional (not an accidental wipe)."
                )

        return "\n".join(warnings_out) if warnings_out else None

    # ═══════════════════════════════════════════════════════════════════════
    # anchor_edit — pattern-based sub-symbol insertion/deletion
    # ═══════════════════════════════════════════════════════════════════════

    def _tool_anchor_edit(self, args: dict[str, Any]) -> ToolResult:
        """Pattern-based file editing for precise sub-symbol insertion/deletion.

        Uses anchor_pattern (substring-first, regex-fallback) to locate the
        target line.  Supports occurrence selection, context-before/after
        disambiguation, and fuzzy fallback.  Deterministic — no LLM call;
        the calling LLM provides code_snippet directly.
        """
        import time as _time

        start_time = _time.monotonic()

        args = self._recover_args_from_raw(args, ("file_path",))
        file_path = (args.get("file_path") or "").strip()
        anchor_pattern = (args.get("anchor_pattern") or "").strip()
        edit_mode = (args.get("edit_mode") or "insert_before").strip()
        code_snippet = str(args.get("code_snippet") or "").strip()
        occurrence = args.get("occurrence", -1)
        context_before = (args.get("context_before") or "").strip() or None
        context_after = (args.get("context_after") or "").strip() or None
        # anchor_ast_lineno: caller-supplied 1-indexed line that bypasses string
        # search (mirrors editor path's 2a strategy). Optional — when present,
        # anchor_pattern becomes a readability hint and is not searched.
        anchor_ast_lineno = args.get("anchor_ast_lineno")

        # ── Validate required fields ──────────────────────────────────────
        if not file_path:
            return self._make_result(
                ok=False,
                content="",
                error="'file_path' is required",
                execution_time=0,
            )
        # Repo-boundary check — see _tool_edit_text. anchor_edit is exposed to the
        # LLM in tool_schemas, so a bare `../` in a model-emitted file_path wrote
        # outside the repo with nothing logged.
        # The RESULT is reused as `_norm` below: resolved + bias-corrected, so a
        # repo-internal symlink stays a symlink (atomic os.replace on an
        # unresolved path would replace it with a regular file).
        _secured = self._secure_path(file_path, confine=True)
        if _secured is None:
            return self._make_result(
                ok=False,
                content="",
                error=f"Path blocked (outside repo): {file_path}",
                execution_time=0,
            )
        # F1 cross-process edit-lease guard.
        _lease_refused = self._refuse_foreign_leased([file_path])
        if _lease_refused is not None:
            return _lease_refused
        # anchor_pattern OR anchor_ast_lineno — exactly one locating strategy required.
        if not anchor_pattern and anchor_ast_lineno is None:
            return self._make_result(
                ok=False,
                content="",
                error="'anchor_pattern' or 'anchor_ast_lineno' is required (one of them)",
                execution_time=0,
            )
        if edit_mode not in ("insert_before", "insert_after", "replace_line", "delete"):
            return self._make_result(
                ok=False,
                content="",
                error=f"Invalid edit_mode: {edit_mode!r} (expected insert_before, insert_after, replace_line, or delete)",
                execution_time=0,
            )
        if edit_mode != "delete" and not code_snippet:
            return self._make_result(
                ok=False,
                content="",
                error=f"'code_snippet' is required for edit_mode={edit_mode!r}",
                execution_time=0,
            )

        # ── Resolve file path ──────────────────────────────────────────────
        # Resolved path from the confine check above (symlink-preserving,
        # bias-corrected) — same as edit_text/create_file.
        _norm = _secured
        if not _norm.exists():
            return self._make_result(
                ok=False,
                content="",
                error=f"File not found: {file_path}{self._suggest_missing_paths(file_path)}",
                execution_time=0,
            )

        try:
            original = _norm.read_text(encoding="utf-8")
        except Exception as e:
            return self._make_result(
                ok=False,
                content="",
                error=f"Failed to read {file_path}: {e}",
                execution_time=0,
            )

        lines = original.splitlines(True)
        lang_id = LanguageId.from_path(str(_norm))

        # ── Anchor matching — import shared helpers ────────────────────────
        from external_llm.agent.anchor_shared import (
            _find_anchor_line,
            _fuzzy_find_anchor_line,
            _inherit_anchor_indent_if_bare,
            _match_anchor,
            resolve_multiline_anchor,
        )
        from external_llm.common.indent_utils import detect_indent_char, indent_unit

        # Destination file's chars-per-level — so a tab/4-space snippet rebased
        # into this file maps each level to the file's real unit, not hardcoded 4.
        _wt_dest_unit = indent_unit(original, detect_indent_char(lines))
        from external_llm.agent.operation_models import FailureClass as _FC  # noqa: N814 — private lazy-import alias

        # ── anchor_ast_lineno: direct line bypasses string search ──────────
        # Mirrors the editor path's direct-line anchor strategy.
        # When the caller supplies an exact 1-indexed line (e.g. right after a
        # read_file/read_symbol), skip the string/regex/fuzzy search entirely —
        # eliminates anchor_miss and anchor_not_unique failures. Falls back to
        # the string path if the number is out of range (stale after a prior edit).
        _ast_anchor = _resolve_ast_anchor_line(anchor_ast_lineno, lines, anchor_pattern)

        # ── Multiline anchor: resolve to a block range instead of rejecting ─
        # A '\n'-containing anchor_pattern (the recurring failure mode where
        # the LLM concatenates several lines) is now *resolved*: the first
        # non-empty line locates the anchor, and every subsequent non-empty
        # line must strip-match the corresponding file line. This turns the
        # previous fail-fast rejection into a fail-tolerant auto-resolve,
        # eliminating the read→retry debug loop. The resolved inclusive range
        # is consumed below by delete / insert / replace per their semantics.
        # Skipped when _ast_anchor is set (lineno is authoritative).
        _multiline_range = None  # (anchor, end) when resolved; else single-line
        if _ast_anchor is None and "\n" in (anchor_pattern or ""):
            _ml = resolve_multiline_anchor(
                lines,
                anchor_pattern,
                occurrence,
                ctx_before=context_before,
                ctx_after=context_after,
            )
            if not _ml["ok"]:
                return self._make_result(
                    ok=False,
                    content="",
                    error=_ml["error"],
                    metadata={
                        "file_path": file_path,
                        "failure_class": _ml.get("failure_class") or _FC.ANCHOR_MULTILINE_PATTERN.value,
                    },
                )
            _multiline_range = (_ml["anchor"], _ml["end"])

        # ── Delete mode: deterministic, no code_snippet needed ─────────────
        if edit_mode == "delete":
            if _multiline_range is not None:
                # Multiline path: range already resolved + verified by
                # resolve_multiline_anchor() above. No fuzzy, no re-verify.
                _del_anchor, _del_end_inclusive = _multiline_range
                _del_count = _del_end_inclusive - _del_anchor + 1
                _del_search_pat = anchor_pattern.split("\n", 1)[0]
                _del_fuzzy_match = False
            else:
                _del_lines_raw = anchor_pattern.split("\n") if anchor_pattern else []
                _del_pat_lines = [pl for pl in _del_lines_raw if pl.strip()]
                if _ast_anchor is not None:
                    # AST lineno path — line is authoritative; pattern (if any)
                    # only supplies the line count for multi-line delete. The
                    # verify-all-pattern-lines block below still runs for count>1.
                    _del_search_pat = _del_pat_lines[0] if _del_pat_lines else lines[_ast_anchor].strip()
                    _del_count = len(_del_pat_lines) if _del_pat_lines else 1
                    _del_anchor = _ast_anchor
                    _del_fuzzy_match = False
                else:
                    if not _del_pat_lines:
                        return self._make_result(
                            ok=False,
                            content="",
                            error="anchor_edit(delete): empty anchor pattern",
                        )
                    _del_search_pat = _del_pat_lines[0]
                    _del_count = len(_del_pat_lines)

                    _del_anchor = _find_anchor_line(
                        lines,
                        _del_search_pat,
                        occurrence,
                        ctx_before=context_before,
                        ctx_after=context_after,
                    )

                    # ── Uniqueness guard — mirrors insert/replace path ──
                    # delete is the MOST DANGEROUS mode (irreversible), so it
                    # MUST fail loudly on multiple matches with default
                    # occurrence=-1 and no context — never silently delete the
                    # last match. The insert/replace path has this guard; the
                    # delete path was the lone hold-out, leaving the most
                    # destructive mode outside the fail-loud contract. Matches
                    # edit_text's "old_string must be UNIQUE" contract.
                    if (
                        _del_anchor is not None
                        and occurrence in (-1, None)
                        and not context_before
                        and not context_after
                    ):
                        _del_match_count = sum(1 for _item_ in lines if _match_anchor(_del_search_pat, _item_))
                        if _del_match_count > 1:
                            return self._make_result(
                                ok=False,
                                content="",
                                error=(
                                    f"anchor_edit(delete): pattern {_del_search_pat!r} "
                                    f"matched {_del_match_count} times in {file_path}. "
                                    f"delete is irreversible — the default occurrence=-1 "
                                    f"(last match) is ambiguous. Specify `occurrence` "
                                    f"or context_before/context_after to disambiguate."
                                ),
                                metadata={
                                    "file_path": file_path,
                                    "failure_class": "anchor_not_unique",
                                    "match_count": _del_match_count,
                                },
                            )

                # Fuzzy fallback — conservative (margin gate, no indent gate for delete)
                _del_fuzzy_match = False
                if _del_anchor is None:
                    _fz_lineno, _fz_score = _fuzzy_find_anchor_line(
                        lines,
                        _del_search_pat,
                        snippet_lines=None,
                        edit_mode="delete",
                    )
                    if _fz_lineno is not None:
                        _del_anchor = _fz_lineno
                        _del_fuzzy_match = True

                _del_fuzzy_match = _del_fuzzy_match if _del_anchor is not None else False

                if _del_anchor is None:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"anchor_edit(delete): pattern {_del_search_pat!r} not found "
                            f"in {file_path} (searched {len(lines)} lines)"
                        ),
                    )

                # ── Verify ALL pattern lines match before deleting ──────────
                _pi = 0  # bound even when _del_count == 1 (loop body never runs)
                _del_mismatch = False
                for _pi in range(1, _del_count):
                    _file_lineno = _del_anchor + _pi
                    if _file_lineno >= len(lines):
                        # Pattern extends beyond file end — only an issue if pattern line is non-empty
                        if _del_pat_lines[_pi].strip():
                            _del_mismatch = True
                        break
                    _pat_stripped = _del_pat_lines[_pi].strip()
                    _file_stripped = lines[_file_lineno].strip()
                    if _pat_stripped and _pat_stripped not in _file_stripped:
                        _del_mismatch = True
                        break
                if _del_mismatch:
                    _del_end_mismatch = min(_del_anchor + _del_count, len(lines))
                    _actual_lines = "".join(lines[_del_anchor:_del_end_mismatch])[:500]
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"anchor_edit(delete): pattern line {_pi + 1} mismatch after anchor "
                            f"at line {_del_anchor + 1} in {file_path}. "
                            f"The remaining {_del_count - 1} pattern line(s) do not match file content. "
                            f"Read the file and provide the exact text to delete."
                        ),
                        metadata={
                            "file_path": file_path,
                            "failure_class": "delete_mismatch",
                        },
                    )
                _del_end_inclusive = min(_del_anchor + _del_count - 1, len(lines) - 1)

            _del_end = _del_end_inclusive + 1
            _deleted_text = "".join(lines[_del_anchor:_del_end])
            lines = lines[:_del_anchor] + lines[_del_end:]
            new_content = "".join(lines)

            if new_content == original:
                return self._make_result(
                    ok=True,
                    content="The content is already as requested — nothing to delete (already_equal)",
                    error="",
                )

            # Syntax validation
            from ...languages.syntax_validator import SyntaxValidator

            _sv = SyntaxValidator.validate_syntax(new_content, lang_id, file_path=file_path)
            _gate_soft_failed = False
            if not _sv.ok:
                _sv_err_msg = _sv.errors[0].message if _sv.errors else "unknown"
                _sv_err_line = _sv.errors[0].line if _sv.errors else 0
                # Soft-fail / origin-skip — mirrors edit_text + dispatch and the
                # insert/replace gate above. Keep edits whose pre-edit content also
                # fails isolated-compile (cascade noise) or whose errors are
                # cross-file-resolvable; refuse only genuine syntax errors.
                _gate_refuse = True
                if lang_id is not LanguageId.PYTHON:
                    _sv_detail = f"{file_path}:{_sv_err_line or 0}:0: {_sv_err_msg}"
                    _gate_refuse = not self._should_soft_fail_verify(_sv_detail, {file_path: original})
                if _gate_refuse:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=f"anchor_edit(delete) produced invalid syntax: {_sv_err_msg}",
                        metadata={"file_path": file_path, "failure_class": "syntax_invalid_after_edit"},
                    )
                # soft-fail → fall through to write
                _gate_soft_failed = True

            atomic_write_text(str(_norm), new_content)
            self._record_text_edit(file_path)
            _exec_time = _time.monotonic() - start_time

            _anchor_meta = {
                "file_path": file_path,
                "mode": "delete",
                "deleted_lines": _del_count,
                "anchor_line": _del_anchor + 1,
                "deleted_text": _deleted_text[:2000],
                "execution_time": _exec_time,
                "fuzzy_match": _del_fuzzy_match,
            }
            _syn = self._run_syntax_check_for_file(str(_norm))
            if not _syn.get("skipped"):
                _anchor_meta["syntax_check"] = _syn
            if _gate_soft_failed:
                _anchor_meta["syntax_gate"] = "soft_fail"
            logger.info(
                "anchor_edit(delete): removed %d lines (L%d-L%d) matching %r from %s",
                _del_count,
                _del_anchor + 1,
                _del_end,
                _del_search_pat[:60],
                file_path,
            )
            return self._make_result(
                ok=True,
                content=f"Deleted {_del_count} line(s) from {file_path} (lines {_del_anchor + 1}-{_del_end})",
                metadata=_anchor_meta,
            )

        # ── Find anchor for insert/replace modes ──────────────────────────
        # When the pattern was multiline, the inclusive range was already
        # resolved + verified above (_multiline_range). Use the block's END for
        # insert_after (insert past the block) and its START for insert_before
        # / replace_line semantics — see the edit-mode branches below.
        _fuzzy_match = False  # set True only via fuzzy path below (multiline: always False)
        _anchor_end = None  # inclusive end of the matched block (0-indexed)
        if _ast_anchor is not None:
            # AST lineno path — authoritative line; bypass string/regex/fuzzy.
            anchor_lineno = _ast_anchor
        elif _multiline_range is not None:
            anchor_lineno, _anchor_end = _multiline_range
        else:
            anchor_lineno = _find_anchor_line(
                lines,
                anchor_pattern,
                occurrence,
                ctx_before=context_before,
                ctx_after=context_after,
            )

            # ── Too-many-matches guard (mirrors editor path's ANCHOR_MAX_MATCHES) ──
            # When occurrence=-1 (default) and no context hints, _find_anchor_line
            # silently picks the LAST match even if the pattern matches many lines,
            # leading to wrong-target edits (e.g. inserting inside the wrong
            # try/except block). Fail loudly instead so the caller supplies
            # `occurrence` or `context_before`/`context_after` to disambiguate.
            # This matches edit_text's "old_string must be UNIQUE" contract.
            if anchor_lineno is not None and occurrence in (-1, None) and not context_before and not context_after:
                _match_count = sum(1 for _item_ in lines if _match_anchor(anchor_pattern, _item_))
                if _match_count > 1:
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            f"anchor_pattern {anchor_pattern!r} matched {_match_count} "
                            f"times in {file_path}. The default occurrence=-1 (last match) "
                            f"is ambiguous with multiple matches. Specify `occurrence` "
                            f"(1=first, 2=second, ...) or context_before/context_after to "
                            f"disambiguate."
                        ),
                        metadata={
                            "file_path": file_path,
                            "failure_class": "anchor_not_unique",
                            "match_count": _match_count,
                        },
                    )

            # Fuzzy fallback — conservative (margin gate + indent compatibility gate)
            if anchor_lineno is None:
                _fz_lineno, _fz_score = _fuzzy_find_anchor_line(
                    lines,
                    anchor_pattern,
                    snippet_lines=code_snippet.splitlines() if code_snippet else None,
                    edit_mode=edit_mode,
                )
                if _fz_lineno is not None:
                    _fuzzy_match = True
                    anchor_lineno = _fz_lineno

        if anchor_lineno is None:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    f"anchor_pattern {anchor_pattern!r} not found in {file_path} "
                    f"(searched {len(lines)} lines) — read the file first and use exact text"
                ),
                metadata={"file_path": file_path, "failure_class": "anchor_miss"},
            )

        # ── Compute anchor indent ──────────────────────────────────────────
        anchor_line_text = lines[anchor_lineno].rstrip("\n\r")
        anchor_indent = len(lines[anchor_lineno]) - len(lines[anchor_lineno].lstrip())
        # Track indent correction for structural feedback metadata. Populated
        # by the Python block-introducer correction below (insert/replace path);
        # stays None for delete mode or non-Python files.
        _indent_correction_info = None

        # ── Collection-literal indentation fix ─────────────────────────────
        if edit_mode == "insert_before" and anchor_line_text.strip() in ("}", "};", "},", "})", "});"):
            _entry_indent = None
            _brace_depth = 0
            for _bi in range(anchor_lineno - 1, max(anchor_lineno - 200, -1), -1):
                _bl_stripped = lines[_bi].strip()
                for _ch in lines[_bi]:
                    if _ch == "}":
                        _brace_depth += 1
                    elif _ch == "{":
                        _brace_depth -= 1
                if _brace_depth < 0:
                    break
                if (
                    _bl_stripped
                    and not _bl_stripped.startswith(("//", "#", "/*", "*"))
                    and _bl_stripped not in ("{", "}", "};", "},", "})", "});")
                ):
                    _detected = len(lines[_bi]) - len(lines[_bi].lstrip())
                    if _detected > anchor_indent:
                        _entry_indent = _detected
                        break
            if _entry_indent is not None:
                anchor_indent = _entry_indent

        # ── Apply the edit ─────────────────────────────────────────────────
        new_code = code_snippet
        orig_content = original
        # (insert_start_line, insert_end_line) of a block-introducer snippet,
        # populated only for insert_before/insert_after on Python — feeds the
        # post-insert AST nesting gate below.
        _introducer_insert_range = None

        if edit_mode == "replace_line":
            # Multiline anchor: replace the WHOLE matched block [anchor, end]
            # with the snippet. Single-line path (the common case) retains the
            # bracket-balance guard below.
            if _anchor_end is not None:
                _old_lines = lines[anchor_lineno : _anchor_end + 1]
                _replace_block = new_code.splitlines(True)
                if _replace_block:
                    _adj_block = _inherit_anchor_indent_if_bare(
                        _replace_block,
                        anchor_line_text,
                        _wt_dest_unit,
                    )
                    _block_text = "".join(_adj_block)
                    if not _block_text.endswith("\n"):
                        _block_text += "\n"
                else:
                    _block_text = "\n"
                lines[anchor_lineno : _anchor_end + 1] = [_block_text]
            else:
                _old_line = lines[anchor_lineno]
                # Indent bare snippet to anchor depth — the earlier `.strip()` already
                # removed all leading whitespace, so snippet is always "bare".
                _replace_lines = new_code.splitlines(True)
                if _replace_lines:
                    _adj_lines = _inherit_anchor_indent_if_bare(
                        _replace_lines,
                        anchor_line_text,
                        _wt_dest_unit,
                    )
                    _new_line = "".join(_adj_lines)
                    if not _new_line.endswith("\n"):
                        _new_line += "\n"
                else:
                    _new_line = "\n"

                # Bracket-balance guard (single-line replace only).
                # _net_bracket_delta is the SSOT per-line bracket tally
                # (_shared_utils): comment/string-aware via a typed CommentSyntax
                # policy (comment_syntax_for(lang_id)), so an unbalanced '(' in a
                # '// note (' (JS/TS/Go/C/...), a '# note (' (Python/Ruby/Bash/PHP),
                # or a '-- note (' (Lua) comment no longer falsely trips the guard
                # nor triggers a spurious multi-line expansion that could delete
                # real code. Replaces the prior binary ``is not PYTHON`` flag which
                # mis-classified Ruby/Bash/PHP (real '#'-comment languages) as
                # C-style and mis-counted brackets inside their '#' comments.
                _comment = comment_syntax_for(lang_id)
                # Seed the per-line tally with the prior-line state (via
                # _scan_to_line_state) so an anchor sitting INSIDE a multi-line
                # block comment or triple-quoted string is counted correctly —
                # its brackets are literal/comment content, not real code.
                # Without this seed a replace_line inside a /* */ block would
                # mis-trigger the F2 expansion and del real code after the
                # comment (the stateless tally saw the anchor's brackets as
                # real, and the forward scan started from empty state).
                _s0, _t0, _bc0 = _scan_to_line_state(lines, anchor_lineno, _comment)
                _old_delta = _net_bracket_delta(_old_line, _comment, in_str=_s0, in_triple=_t0, block_close=_bc0)
                _new_delta = _net_bracket_delta(_new_line, _comment, in_str=_s0, in_triple=_t0, block_close=_bc0)

                if _old_delta != _new_delta:
                    # Guard: snippet starts with '}' → continuation fragment
                    if _new_line.strip().startswith("}"):
                        return self._make_result(
                            ok=False,
                            content="",
                            error=(
                                f"anchor_edit(replace_line): snippet starts with '}}' "
                                f"at {file_path}:{anchor_lineno + 1} — continuation fragment "
                                f"cannot replace a top-level construct."
                            ),
                            metadata={"file_path": file_path, "failure_class": "structural_gate_violation"},
                        )

                    # Attempt bracket-balance expansion
                    _needed_balance = _new_delta
                    _scan_balance = _old_delta
                    _close_line = None
                    # Stateful comment-aware scan: thread string/block-comment
                    # state across lines via _scan_line_brackets_delta (SSOT in
                    # _shared_utils) so a bracket inside a Python '#' comment, a
                    # C-family '/* */' block comment, or a triple-quoted string is
                    # never mis-counted. The prior inline scanner only handled '//'
                    # line comments and could mis-identify the close line — then
                    # `del` real code (e.g. a ')' in a Python '#' comment
                    # terminated the scan early, deleting the function's real
                    # arguments). ``_block_close`` carries the open block comment's
                    # CLOSE token (e.g. '*/', ']]') across lines.
                    #
                    # The scan starts from the state AFTER the new anchor line
                    # (seeded with the same prior-line context), so a multi-line
                    # construct opened on/before the anchor is tracked correctly —
                    # not from an empty state that would mis-identify the close.
                    _, _in_str, _in_triple, _block_close = _scan_line_brackets_delta(
                        _new_line, _s0, _t0, _bc0, _comment
                    )
                    for _scan_i in range(anchor_lineno + 1, min(len(lines), anchor_lineno + 500)):
                        _ld, _in_str, _in_triple, _block_close = _scan_line_brackets_delta(
                            lines[_scan_i], _in_str, _in_triple, _block_close, _comment
                        )
                        _scan_balance += _ld
                        if _scan_balance == _needed_balance:
                            _close_line = _scan_i
                            break

                    if _close_line is not None:
                        logger.warning(
                            "anchor_edit(replace_line): bracket delta mismatch — "
                            "expanding replace from 1 line to %d lines",
                            _close_line - anchor_lineno + 1,
                        )
                        lines[anchor_lineno] = _new_line
                        del lines[anchor_lineno + 1 : _close_line + 1]
                    else:
                        return self._make_result(
                            ok=False,
                            content="",
                            error=(
                                f"anchor_edit(replace_line): bracket imbalance "
                                f"(old={_old_delta:+d}, new={_new_delta:+d}) at "
                                f"{file_path}:{anchor_lineno + 1} — cannot safely replace"
                            ),
                            metadata={"file_path": file_path, "failure_class": "structural_gate_violation"},
                        )
                else:
                    lines[anchor_lineno] = _new_line

        else:
            # insert_before or insert_after
            if edit_mode == "insert_before":
                insert_idx = anchor_lineno
            # Multiline anchor: insert past END of matched block (semantics A).
            # def/class multi-line signature skip is single-line-anchor only —
            # a multiline block is already fully resolved so no skip is needed.
            elif _anchor_end is not None:
                insert_idx = _anchor_end + 1
            else:
                insert_idx = anchor_lineno + 1

                # ── Block-end auto-correction ────────────────────────────
                # If the anchor line is a block HEADER (def/class/if/... in
                # Python, a '{'-opening line in brace languages), inserting
                # at lineno+1 lands INSIDE the body — the classic
                # "insert_after on a def/{ line nests the snippet" bug.
                # Find the block's real END (language-agnostic, tree-sitter
                # first with brace/indent fallbacks) and insert past it so
                # the new construct becomes a sibling. Installing a grammar
                # enables the correction for that language with no code
                # change. Replaces the old def-skip logic that only scanned
                # to the signature colon (which still landed in the body).
                _block_end = _find_block_end_line(
                    original,
                    lang_id.value,
                    anchor_lineno,
                    lines,
                )
                if _block_end is not None and _block_end > anchor_lineno:
                    logger.info(
                        "anchor_edit(insert_after): anchor L%d is a %d-line "
                        "block header — inserting after block end L%d "
                        "instead of into the body",
                        anchor_lineno + 1,
                        _block_end - anchor_lineno + 1,
                        _block_end + 1,
                    )
                    insert_idx = _block_end + 1
                    # anchor_indent already reflects the header's own indent,
                    # so the new construct is placed as a sibling at the same
                    # depth. (The old def-skip path bumped indent to the BODY
                    # level, which is what caused the nesting bug.)

            # ── Python block-introducer indent correction ────────────────
            # When the snippet STARTS a new def/class (a block introducer) but
            # the anchor matched a *body* line (deeper than its enclosing
            # header), blindly inheriting anchor_indent would land the new
            # block as a nested function/class — a silent structural bug.
            # Re-derive anchor_indent from the nearest enclosing def/class
            # header so the new block is inserted as a sibling instead.
            # (Indentation has no structural meaning in brace-languages, so
            # this correction is Python-only — see system prompt rule 7.)
            _snip_is_block_introducer: bool = False  # pre-bound; the PYTHON-only branch overrides
            if lang_id is LanguageId.PYTHON:
                # Detect block-introducer using the snippet's MINIMUM-indent
                # line (not the first non-empty line). When the LLM prepends a
                # fragment of existing code (e.g. "    finally:\n
                # rec.cleanup()\n") before the new block, the first non-empty
                # line is the fragment — not the introducer — so the old
                # "_snip_first_line" check silently failed to correct, landing
                # the new def/class at the anchor's deep indent. Using the
                # min-indent line reliably finds the new top-level construct.
                _snip_lines_for_intro = new_code.splitlines()
                _snip_min_indent = None
                for _ln in _snip_lines_for_intro:
                    if _ln.strip():
                        _ind = len(_ln) - len(_ln.lstrip())
                        if _snip_min_indent is None or _ind < _snip_min_indent:
                            _snip_min_indent = _ind
                # Check ALL min-indent lines, not just the first one found. A
                # comment banner, module-level constant, or decorator at the
                # same min-indent as the def/class it introduces would
                # otherwise be picked as "the" introducer line, fail the
                # startswith check, and silently skip this correction —
                # exactly the gap that let a cache-section snippet (banner +
                # constants + defs) inherit a nested anchor's indent.
                _snip_is_block_introducer = (
                    any(
                        _item_.strip()
                        and (len(_item_) - len(_item_.lstrip())) == _snip_min_indent
                        and _item_.lstrip().startswith(("def ", "async def ", "class ", "@"))
                        for _item_ in _snip_lines_for_intro
                    )
                    if _snip_min_indent is not None
                    else False
                )
                # Only correct when the anchor is NOT itself a def/class header:
                # a header anchor already sits at the right sibling level, and
                # lifting it further (e.g. first-method-of-class) would corrupt
                # the enclosing class body by de-indenting to the class level.
                _anchor_is_header = lines[anchor_lineno].strip().startswith(("def ", "async def ", "class "))
                if _snip_is_block_introducer and not _anchor_is_header:
                    for _up in range(anchor_lineno - 1, -1, -1):
                        _up_text = lines[_up]
                        _up_stripped = _up_text.strip()
                        if _up_stripped.startswith(("def ", "async def ", "class ")):
                            _up_indent = len(_up_text) - len(_up_text.lstrip())
                            if _up_indent < anchor_indent:
                                _prev = anchor_indent
                                anchor_indent = _up_indent
                                logger.debug(
                                    "anchor_edit(indent-correct): snippet is a block "
                                    "introducer; re-anchored indent %d→%d (nearest enclosing header)",
                                    _prev,
                                    _up_indent,
                                )
                                _indent_correction_info = {
                                    "snippet_base_indent": _snip_min_indent if _snip_min_indent is not None else 0,
                                    "original_anchor_indent": _prev,
                                    "corrected_anchor_indent": _up_indent,
                                    "reason": "block_introducer_at_nested_anchor",
                                }
                            break

            # ── Fragment-duplication pre-guard (insert_before/insert_after) ──
            # If code_snippet duplicates existing code around the anchor, the
            # insert would land a dangling block that only fails the POST-write
            # syntax check with an opaque message. Detect it HERE (before
            # indentation normalization touches the snippet) so we can reject
            # with an actionable, file-preserving error. replace_line/delete
            # are exempt — they legitimately overlap existing code.
            _dup = _detect_fragment_duplication(lines, insert_idx, new_code)
            if _dup is not None:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"anchor_edit({edit_mode}): code_snippet duplicates "
                        f"{_dup['matched']}/{_dup['content_lines']} non-trivial "
                        f"lines already present near {file_path}:L{anchor_lineno + 1} "
                        f"(ratio {_dup['ratio']:.2f}). code_snippet must contain "
                        f"ONLY the new lines to insert — not a copy of the anchor "
                        f"or its surrounding context. Duplicated lines:\n"
                        f"{_dup['dup_lines']}"
                    ),
                    metadata={
                        "file_path": file_path,
                        "failure_class": "fragment_duplication",
                        "anchor_line": anchor_lineno + 1,
                        "mode": edit_mode,
                        "dup_ratio": _dup["ratio"],
                        "dup_content_lines": _dup["content_lines"],
                    },
                )

            # ── Indentation normalization ───────────────────────────────
            _llm_lines = new_code.splitlines()
            _min_indent = None
            for _ln in _llm_lines:
                if _ln.strip():
                    _ind = len(_ln) - len(_ln.lstrip())
                    if _min_indent is None or _ind < _min_indent:
                        _min_indent = _ind
            if _min_indent is None:
                _min_indent = 0
            indented_lines = []
            for ln in _llm_lines:
                if ln.strip():
                    current_indent = len(ln) - len(ln.lstrip())
                    rel_indent = current_indent - _min_indent
                    rel_indent = max(rel_indent, 0)
                    indented_lines.append(" " * (anchor_indent + rel_indent) + ln.lstrip())
                else:
                    indented_lines.append("")
            indented_code = "\n".join(indented_lines) + "\n"

            # ── Newline guard: last-line insert on non-\n-terminated files ──
            if insert_idx > 0 and insert_idx == len(lines) and not lines[-1].endswith("\n"):
                lines[-1] += "\n"

            lines.insert(insert_idx, indented_code)

            if lang_id is LanguageId.PYTHON and _snip_is_block_introducer:
                _introducer_insert_range = (insert_idx, insert_idx + len(indented_lines))

        new_content = "".join(lines)

        # ── Already-equal guard: no-op success ──────────────────────────────
        if new_content == orig_content:
            return self._make_result(
                ok=True,
                content="The content is already as requested — no change needed (already_equal)",
                error="",
                metadata={"file_path": file_path, "failure_class": "already_equal"},
            )

        # ── Structural gate: block-introducer nested inside a function ──────
        # Defense-in-depth AST check behind the text-based indent-correction
        # above. If the snippet introduces a def/class and it still ended up
        # lexically inside a FunctionDef/AsyncFunctionDef body, reject before
        # writing — a silent nesting bug that the syntax gate below cannot
        # catch (nested defs/constants are syntactically valid Python).
        if _introducer_insert_range is not None:
            _nest_err = _check_block_introducer_nesting(
                new_content, _introducer_insert_range[0], _introducer_insert_range[1]
            )
            if _nest_err is not None:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(f"anchor_edit({edit_mode}): {_nest_err} file={file_path}, anchor_line={anchor_lineno + 1}"),
                    metadata={
                        "file_path": file_path,
                        "failure_class": "structural_gate_violation",
                        "anchor_line": anchor_lineno + 1,
                        "mode": edit_mode,
                    },
                )

        # ── Syntax validation + write ──────────────────────────────────────
        from ...languages.syntax_validator import SyntaxValidator

        _sv = SyntaxValidator.validate_syntax(new_content, lang_id, file_path=file_path)
        _gate_soft_failed = False
        if not _sv.ok:
            _sv_err_msg = _sv.errors[0].message if _sv.errors else "unknown"
            _sv_err_line = getattr(_sv.errors[0], "line", None) if _sv.errors else None
            # ── Soft-fail / origin-skip (mirrors edit_text + dispatch) ──
            # Non-Python compiled languages may emit cascade noise (missing
            # deps/SDK — e.g. an Android ViewModel without the SDK) or
            # cross-file-resolvable errors under isolated-compile. Keep such
            # edits; refuse only GENUINE syntax errors on a valid baseline.
            # Python compile() is self-contained, so it keeps the strict refuse.
            # See _should_soft_fail_verify (origin-skip guard).
            _gate_refuse = True
            if lang_id is not LanguageId.PYTHON:
                _sv_detail = f"{file_path}:{_sv_err_line or 0}:0: {_sv_err_msg}"
                _gate_refuse = not self._should_soft_fail_verify(_sv_detail, {file_path: original})
            if _gate_refuse:
                # Build an actionable hint: show the region around the error and
                # the anchor context so the LLM can see WHY the edit broke syntax.
                # The most common cause of "invalid syntax" in insert mode is the
                # snippet accidentally including a copy of existing code (a
                # "fragment duplication"), which then gets re-indented to the
                # anchor level and produces a duplicate/dangling block.
                _hint_parts = [f"anchor_edit introduced syntax error (file unchanged): {_sv_err_msg}"]
                _hint_parts.append(f"file={file_path}, anchor_line={anchor_lineno + 1}")
                if _sv_err_line:
                    _hint_parts.append(f"syntax_error_at_line={_sv_err_line}")
                if edit_mode in ("insert_before", "insert_after"):
                    _hint_parts.append(
                        "Likely cause: code_snippet accidentally includes a copy of "
                        "existing code around the anchor (fragment duplication). The "
                        "snippet should contain ONLY the new code to insert, not the "
                        "anchor line or its surrounding context. Re-read the file, "
                        "then provide only the new lines in code_snippet."
                    )
                _hint_parts.append(
                    "If inserting a top-level construct (def/class) at file scope, "
                    "prefer apply_patch (which uses exact line ranges) over anchor_edit."
                )
                return self._make_result(
                    ok=False,
                    content="",
                    error=" ".join(_hint_parts),
                    metadata={
                        "file_path": file_path,
                        "failure_class": "syntax_invalid_after_edit",
                        "anchor_line": anchor_lineno + 1,
                        "syntax_error_line": _sv_err_line,
                        "mode": edit_mode,
                    },
                )
            # soft-fail → fall through to write
            _gate_soft_failed = True

        atomic_write_text(str(_norm), new_content)
        self._record_text_edit(file_path)
        _exec_time = _time.monotonic() - start_time

        _orig_lines = orig_content.splitlines()
        _mod_lines = new_content.splitlines()
        _delta = len(_mod_lines) - len(_orig_lines)

        _anchor_meta = {
            "file_path": file_path,
            "mode": edit_mode,
            "anchor_line": anchor_lineno + 1,
            "line_delta": _delta,
            "execution_time": _exec_time,
            "fuzzy_match": _fuzzy_match,
        }
        if _anchor_end is not None:
            _anchor_meta["anchor_end"] = _anchor_end + 1
            _anchor_meta["multiline_anchor"] = True
        if _gate_soft_failed:
            _anchor_meta["syntax_gate"] = "soft_fail"
        # ── Structural feedback (Python only) ────────────────────────────────
        # Expose the indent the snippet was inserted at, the enclosing scope it
        # landed in, and (when applicable) whether anchor_indent was corrected
        # to lift a block-introducer snippet to its proper sibling level. This
        # lets the LLM self-verify the structural correctness of an insert
        # without a separate read_file round-trip (see _detect_enclosing_scope).
        if lang_id is LanguageId.PYTHON:
            try:
                _scope = _detect_enclosing_scope(orig_content.splitlines(), anchor_lineno)
                _tl = _scope.get("top_level")
                _il = _scope.get("innermost")
                if _tl and _tl[0] is not None:
                    _anchor_meta["enclosing_scope"] = {
                        "kind": _tl[0],
                        "name": _tl[1],
                        "indent": _tl[2],
                    }
                    if _il and _il[0] is not None and _il[1] != _tl[1]:
                        _anchor_meta["enclosing_scope"]["innermost"] = {
                            "kind": _il[0],
                            "name": _il[1],
                            "indent": _il[2],
                        }
                _anchor_meta["inserted_at_indent"] = (
                    _indent_correction_info["corrected_anchor_indent"]
                    if _indent_correction_info
                    else _scope.get("anchor_indent", anchor_indent)
                )
            except Exception:
                logger.debug("<module>::WriteToolsPatchMixin::_tool_anchor_edit:1 suppressed Exception", exc_info=True)
            if _indent_correction_info is not None:
                _anchor_meta["indent_correction"] = _indent_correction_info
        # Semantic feedback (non-blocking): pyright/tsc/go diagnostics for
        # type/undefined-name/import issues. Mirrors apply_patch/edit_file.
        _syn = self._run_syntax_check_for_file(str(_norm))
        if not _syn.get("skipped"):
            _anchor_meta["syntax_check"] = _syn

        _line_desc = (
            f"lines {anchor_lineno + 1}-{_anchor_end + 1}" if _anchor_end is not None else f"line {anchor_lineno + 1}"
        )
        logger.info(
            "anchor_edit: %s in %s at %s (mode=%s)",
            edit_mode,
            file_path,
            _line_desc,
            edit_mode,
        )

        return self._make_result(
            ok=True,
            content=(
                f"anchor_edit ({'[fuzzy] ' if _fuzzy_match else ''}{edit_mode}) applied to {file_path} "
                f"at {_line_desc} (delta: {_delta:+d} lines)"
                + (" ⚠️ fuzzy match — verify result with read_file" if _fuzzy_match else "")
            ),
            metadata=_anchor_meta,
        )
