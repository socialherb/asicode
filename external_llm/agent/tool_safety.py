"""
Write Safety Manager for asicode Agent

Provides file snapshot/verify/restore safety for write operations,
and approval gating for dangerous tool calls.
Extracted from tool_registry.py to reduce its size and improve SRP.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from typing import Any, Optional

from external_llm.languages.tree_sitter_utils import grammar_key_for_ext

logger = logging.getLogger(__name__)


# Sentinel marking "file did not exist before this write" in a snapshot,
# so the restore step can ``os.remove`` files the tool created from scratch.
_MISSING_SNAP = object()


# ── Post-edit declaration-loss guard ─────────────────────────────────────
# Detects symbols/imports that silently disappeared in an edit — the failure
# mode where a full-block rewrite (replace_file, modify_symbol full mode)
# drops existing functions. Self-contained AST walk on purpose: this is the
# portable kernel of the planner lane's SYMBOL_NOT_REMOVED / IMPORT_EXISTS
# intent assertions, re-implemented here WITHOUT importing planner code
# (the planner lane may be removed later). Unlike intent assertions, this
# needs no declared intent — it is computed purely from pre/post snapshots.

def _python_decl_sets(source: str):
    """Extract (symbols, import bindings) declared at module/class level.

    Symbols: top-level functions/classes + class methods as 'Class.method'.
    Imports: module-level binding names (alias-aware).
    Returns None when the source does not parse (caller skips the check —
    syntax breakage is handled by the separate verify/rollback path).
    """
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    symbols: set = set()
    imports: set = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}.{sub.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imports.add(a.asname or a.name)
    return symbols, imports



def _treesitter_symbol_set(source: str, language: str):
    """Symbol names via tree-sitter for non-Python languages.

    Returns None (caller skips the check) when tree-sitter/parser is
    unavailable or the source has syntax errors — tree-sitter parses broken
    code tolerantly, which would surface phantom "removed" symbols, so a
    has_error tree is treated like Python's parse failure.
    Names are FLAT (method 'update', not 'Svc.update') — a same-named method
    in another class masks a removal; acceptable for a warning heuristic.
    """
    try:
        from external_llm.languages.tree_sitter_utils import find_all_symbols, get_parser
    except ImportError:
        return None
    parser = get_parser(language)
    if parser is None:
        return None
    try:
        tree = parser.parse(source.encode("utf-8"))
        if tree.root_node.has_error:
            return None
    except Exception:
        return None
    return {name for name, _kind, _s, _e in find_all_symbols(source, language)}


def _cap_list(items: list, cap: int = 8) -> str:
    shown = ", ".join(items[:cap])
    return shown + (f" (+{len(items) - cap} more)" if len(items) > cap else "")


def summarize_decl_losses(original: str, current: str, suffix: str = ".py") -> str:
    """One warning line when an edit dropped top-level symbols (and, for
    Python, module-level imports).

    Returns "" when nothing was lost, either side does not parse, or the
    language is unsupported. Deliberately NO rename heuristics (substring
    matching mis-fires) — when symbols were removed, newly added names are
    listed so the model can judge rename-vs-deletion itself.
    """
    removed_imps: list = []
    if suffix == ".py":
        pre = _python_decl_sets(original)
        post = _python_decl_sets(current)
        if pre is None or post is None:
            return ""
        pre_syms, post_syms = pre[0], post[0]
        removed_imps = sorted(pre[1] - post[1])
    else:
        language = grammar_key_for_ext(suffix)
        if not language:
            return ""
        pre_syms = _treesitter_symbol_set(original, language)
        post_syms = _treesitter_symbol_set(current, language)
        if pre_syms is None or post_syms is None:
            return ""

    removed_syms = sorted(pre_syms - post_syms)
    if not removed_syms and not removed_imps:
        return ""
    parts = []
    if removed_syms:
        parts.append(f"symbols [{_cap_list(removed_syms)}]")
    if removed_imps:
        parts.append(f"imports [{_cap_list(removed_imps)}]")
    line = "    ⚠️ removed " + " and ".join(parts) + " — verify this was intended"
    added_syms = sorted(post_syms - pre_syms)
    if removed_syms and added_syms:
        line += f"; newly added: [{_cap_list(added_syms, 6)}] (rename?)"
    return line


class WriteSafetyManager:
    """Manages write safety: file snapshots before writes, syntax verification
    after writes, and rollback on failure.  Also handles approval gating."""

    _PATCH_FILE_THRESHOLD = 3

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        # Memo for _resolve_missing_import — (name, current_file) → import
        # line or None. Reset per auto_repair_semantic call; initialized here
        # so direct calls to _resolve_missing_import never hit AttributeError.
        self._f821_import_cache: dict = {}

    # ------------------------------------------------------------------
    # Approval gate
    # ------------------------------------------------------------------

    @staticmethod
    def count_patch_files(patch_text: str) -> int:
        """Count number of files affected by a patch.

        Counts ``diff --git`` headers (primary) AND ``--- a/`` / ``--- b/``
        lines (secondary, for patches that omit the git header).  Match the
        extraction logic of ``snapshot_target_files`` so the approval gate
        and the safety snapshot see the same file set — otherwise a bare
        ``--- a/`` / ``+++ b/`` patch (no ``diff --git`` prefix) bypasses
        the multi-file approval threshold entirely.
        """
        _seen: set = set()
        for _line in patch_text.splitlines():
            if _line.startswith("diff --git "):
                # Extract b/ path: "diff --git a/foo.py b/foo.py"
                _parts = _line.split()
                if len(_parts) >= 4:
                    _seen.add(_parts[3])
                else:
                    _seen.add(_line)  # fallback: count the whole line once
            elif (
                _line.startswith("--- a/")
                or _line.startswith("--- b/")
                or _line.startswith("+++ b/")
            ):
                # "+++ b/" covers new-file hunks whose old side is
                # "--- /dev/null" — same as snapshot_target_files.
                _path = _line[6:].strip()
                if _path and _path != "/dev/null":
                    _seen.add("b/" + _path)
        return len(_seen)

    def approval_preview(self, tool_name: str, args: dict) -> tuple[str, bool]:
        """Generate a preview for approval gating.

        Returns:
            (preview_text, needs_approval) tuple.
        """
        if tool_name == "apply_patch":
            patch = args.get("patch", "")
            if self.count_patch_files(patch) >= self._PATCH_FILE_THRESHOLD:
                return patch[:4000], True
            return "", False
        if tool_name == "delete_file":
            path = args.get("path", "")
            return f"DELETE FILE: {path}", True
        if tool_name == "write_plan":
            try:
                preview = json.dumps(args.get("plan", {}), indent=2, ensure_ascii=False)[:4000]
            except (TypeError, ValueError):
                preview = str(args.get("plan", ""))[:4000]
            return f"WRITE PLAN:\n{preview}", True
        return "", False

    def gate_check(
        self,
        tool_name: str,
        args: dict,
        approval_callback: Optional[Callable],
    ) -> Optional[dict[str, Any]]:
        """Check if tool call needs approval and if it was granted.

        Returns:
            None if approved or no approval needed.
            A dict with error info if rejected (caller should construct ToolResult).
        """
        if approval_callback is None:
            return None
        preview, needs = self.approval_preview(tool_name, args)
        if not needs:
            return None
        approved = approval_callback(tool_name, args, preview)
        if approved:
            return None
        return {
            "error": (
                f"[User rejected] {tool_name!r} was blocked by the user. "
                "Try a different, safer approach (e.g. split into single-file patches, "
                "reduce plan scope, or confirm deletion path with the user)."
            ),
            "metadata": {"gate": "rejected", "tool": tool_name},
        }

    # ------------------------------------------------------------------
    # Write safety: snapshot + verify + rollback
    # ------------------------------------------------------------------

    def snapshot_target_files(self, tool_name: str, args: dict) -> dict:
        """Capture file contents before a write operation.

        For apply_patch / write_plan, captures ALL files mentioned in the patch
        (diff --git headers), not just the first one.  This ensures that multi-file
        patches are fully restorable on syntax-error rollback.
        """
        snapshots: dict = {}
        try:
            if tool_name in ("apply_patch", "write_plan"):
                raw_plan = args.get("patch") or args.get("plan") or ""
                # write_plan accepts a dict plan; only use string-typed values
                patch = raw_plan if isinstance(raw_plan, str) else ""

                targets: list = []
                # Secondary: use "--- a/…" / "+++ b/…" lines when no diff --git
                # headers present. "+++ b/…" captures new-file targets (--- /dev/null).
                for _line in patch.splitlines():
                    if _line.startswith('--- a/') or _line.startswith('--- b/'):
                        _path = _line[6:].split('\t')[0].strip()
                        if _path and _path != "/dev/null":
                            targets.append(_path)
                    elif _line.startswith('+++ b/'):
                        _path = _line[6:].split('\t')[0].strip()
                        if _path and _path != "/dev/null":
                            targets.append(_path)
                # For write_plan with dict plan: extract paths from ops
                if not targets and isinstance(raw_plan, dict):
                    plan_ops = (
                        raw_plan.get("ops") or raw_plan.get("operations") or []
                    )
                    if not plan_ops and "path" in raw_plan:
                        plan_ops = [raw_plan]
                    targets = [str(op["path"]) for op in plan_ops if op.get("path")]
                # Fallback: explicit file_path / path arg (e.g. apply_patch called
                # without a unified diff header, or for direct single-file targeting)
                if not targets:
                    _explicit = args.get("file_path") or args.get("path") or ""
                    if _explicit and isinstance(_explicit, str):
                        targets = [_explicit]
                for target in targets:
                    full_path = (
                        target if os.path.isabs(target)
                        else os.path.join(self.repo_root, target)
                    )
                    if os.path.isfile(full_path) and full_path not in snapshots:
                        with open(full_path, encoding="utf-8", errors="replace") as f:
                            snapshots[full_path] = f.read()
                    elif full_path not in snapshots:
                        # New file: store sentinel so restore can remove it on rollback
                        snapshots[full_path] = _MISSING_SNAP
            else:
                target = args.get("file_path") or args.get("path") or ""
                if target and isinstance(target, str):
                    full_path = (
                        target if os.path.isabs(target)
                        else os.path.join(self.repo_root, target)
                    )
                    if os.path.isfile(full_path):
                        with open(full_path, encoding="utf-8", errors="replace") as f:
                            snapshots[full_path] = f.read()
                    elif full_path not in snapshots:
                        snapshots[full_path] = _MISSING_SNAP
        except OSError:
            pass
        return snapshots

    def verify_after_write(self, snapshots: dict, _post_contents: dict | None = None) -> tuple[bool, str]:
        """Basic syntax check on files that were modified.

        Returns (True, "") if all files pass syntax validation (or have no validator),
        or (False, "error detail") with the first syntax error message.

        When *_post_contents* is provided (path → post-write content), the file
        is validated from memory instead of being re-read from disk — an I/O
        optimisation for call sites that already hold the just-written content.
        """
        from ..languages import LanguageRegistry
        for path in snapshots:
            _vaw_provider = LanguageRegistry.instance().get(path)
            if _vaw_provider and _vaw_provider.capabilities().has_syntax_validator and os.path.isfile(path):
                try:
                    _content = (
                        _post_contents.get(path)
                        if _post_contents
                        else None
                    )
                    if _content is None:
                        with open(path, encoding="utf-8", errors="replace") as f:
                            _content = f.read()
                    _val = _vaw_provider.validate_syntax(path, _content)
                    if not _val.ok:
                        # A single structural break (e.g. an unbalanced brace)
                        # makes the parser emit a long cascade of follow-on
                        # errors. Surface only the first few — the root cause is
                        # at the top — and summarise the rest so the message
                        # stays short for the LLM/user. The first entry keeps the
                        # full ``file:line:col:`` shape that downstream rollback
                        # context (tool_registry) parses out of this detail.
                        _errs = _val.errors or []
                        _MAX_SHOWN = 3
                        if _errs:
                            _head = _errs[0]
                            _detail = f"{_head.file}:{_head.line}:{_head.col}: {_head.message}"
                            for _e in _errs[1:_MAX_SHOWN]:
                                _detail += f"; L{_e.line}:{_e.col} {_e.message}"
                            if len(_errs) > _MAX_SHOWN:
                                _detail += f" (+{len(_errs) - _MAX_SHOWN} more syntax errors)"
                        else:
                            _detail = f"syntax error in {path}"
                        logger.warning(
                            "verify_after_write: syntax error in %s — %s", path, _detail
                        )
                        return False, _detail
                except OSError:
                    return False, f"OS error reading {path}"
        return True, ""

    @staticmethod
    def restore_snapshots(snapshots: dict) -> list[str]:
        """Restore files from pre-write snapshot.

        Returns list of file paths whose restoration failed (empty on success).

        Uses mkstemp + os.replace for atomic writes (crash-safe — no partial
        truncation). Files that did not exist before (``_MISSING_SNAP`` sentinel)
        are removed.
        """
        _failed: list[str] = []
        for path, content in snapshots.items():
            try:
                if content is _MISSING_SNAP:
                    if os.path.exists(path):
                        os.remove(path)
                    continue
                # Atomic write: mkstemp + os.replace so a crash/SIGKILL/disk-full
                # mid-restore never leaves the file truncated/partial.
                _dir = os.path.dirname(path) or "."
                _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=".asi-revert-")
                try:
                    with os.fdopen(_fd, "w", encoding="utf-8") as _f:
                        _f.write(content)
                    # Preserve the original file's permission bits: mkstemp
                    # creates the temp with mode 0600 and os.replace keeps the
                    # temp's mode. Without this, restoring an executable script
                    # (+x) or a group/world-readable file would silently strip it
                    # to owner-only — "restore" must never mutate metadata.
                    if os.path.exists(path):
                        os.chmod(_tmp, os.stat(path).st_mode)
                    os.replace(_tmp, path)
                except BaseException:
                    try:
                        os.unlink(_tmp)
                    except OSError:
                        pass
                    raise
            except OSError:
                logger.error(
                    "Write safety: rollback failed for %s — file may be corrupted",
                    path, exc_info=True,
                )
                _failed.append(path)
        return _failed

    # ------------------------------------------------------------------
    # Deterministic post-write change summary (no intent inference)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_regions(regions: list, max_shown: int = 6) -> str:
        """Format a list of (start, end) 1-based inclusive line ranges."""
        if not regions:
            return "(no line-range info)"
        parts = [f"L{s}" if s == e else f"L{s}-{e}" for (s, e) in regions]
        if len(parts) > max_shown:
            extra = len(parts) - max_shown
            parts = [*parts[:max_shown], f"(+{extra} more)"]
        return ", ".join(parts)

    def summarize_change(self, snapshots: dict) -> Optional[str]:
        """Deterministic post-write diff summary for write tools.

        Diffs each pre-write snapshot against the file's current on-disk
        content and reports added/removed line counts plus the changed line
        regions (1-based, in the NEW file). Byte-identical writes are flagged
        prominently: a write tool reporting success while changing nothing is
        the dominant NO_EFFECTIVE_PROGRESS failure mode, and the LLM otherwise
        has no signal that its "successful" edit was a no-op.

        Pure diff over already-captured content + one re-read per file. No LLM,
        no intent inference. Returns a compact summary, or None if there is
        nothing to report (no snapshots / unreadable files).
        """
        import difflib

        if not snapshots:
            return None

        lines_out: list = []
        for path, original in snapshots.items():
            # New file (did not exist before): treat original as empty
            if original is _MISSING_SNAP:
                original = ""
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    current = f.read()
            except OSError:
                continue

            try:
                rel = os.path.relpath(path, self.repo_root)
            except ValueError:
                rel = path

            if current == original:
                lines_out.append(
                    f"  {rel}: ⚠️ NO CHANGE — file is byte-identical to "
                    "before this edit (the tool reported success but nothing was "
                    "modified). Re-check your anchor/symbol/patch target."
                )
                continue

            a = original.splitlines()
            b = current.splitlines()
            added = removed = 0
            regions: list = []
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                if tag in ("replace", "insert"):
                    added += (j2 - j1)
                if tag in ("replace", "delete"):
                    removed += (i2 - i1)
                if j2 > j1:
                    regions.append((j1 + 1, j2))
                elif tag == "delete":
                    # pure deletion — point to the gap in the new file
                    regions.append((j1 + 1, j1 + 1))

            lines_out.append(
                f"  {rel}: +{added}/-{removed} lines; changed "
                f"{self._format_regions(regions)}"
            )

            # Declaration-loss guard: flag symbols/imports that disappeared
            # (Python via ast; JS/TS/Go/Java/Kotlin via tree-sitter — both
            # sides must parse cleanly; see summarize_decl_losses)
            _loss = summarize_decl_losses(
                original, current, os.path.splitext(path)[1].lower()
            )
            if _loss:
                lines_out.append(_loss)

        if not lines_out:
            return None
        return "[POST-EDIT DIFF]\n" + "\n".join(lines_out)

    @staticmethod
    def all_files_unchanged(snapshots: dict) -> bool:
        """True iff ``snapshots`` is non-empty and every file is byte-identical to disk.

        Targets the NO_EFFECTIVE_PROGRESS failure mode: a write tool (notably
        ``apply_patch``) can report ``ok=True`` while its patch matched content
        that was already present, so nothing on disk changed. ``summarize_change``
        surfaces a "⚠️ NO CHANGE" *warning* in the result text, but the ``ok``
        flag stays True — so progress/retry heuristics treat the no-op as a
        successful step. Callers that want a hard signal use this to downgrade
        ``ok`` to False.

        Returns False for an empty dict (nothing was snapshotted, so the
        post-hoc check is meaningless), for any new-file (``_MISSING_SNAP``)
        entry (a created file IS a change), or if any file differs from / cannot
        be read from disk.

        Known limitation: both the snapshot and the re-read here decode via
        ``utf-8`` with ``errors="replace"`` (consistent with
        :meth:`snapshot_target_files` / :meth:`summarize_change`). A change
        confined to *invalid* UTF-8 bytes would map to the same replacement
        char on both sides and read as "unchanged". apply_patch targets text,
        so this is near-impossible in practice; the consistency across the
        three methods keeps them in lock-step either way.
        """
        if not snapshots:
            return False
        for _path, original in snapshots.items():
            if original is _MISSING_SNAP:
                return False
            try:
                with open(_path, encoding="utf-8", errors="replace") as _f:
                    if _f.read() != original:
                        return False
            except OSError:
                return False
        return True

    def new_semantic_warnings(self, snapshots: dict) -> Optional[str]:
        """Compare pre-snapshot vs current content for new ruff F-code findings.

        The keystone: ruff F401/F811/F821/F841 findings that appeared AFTER the
        edit (filtered against pre-snapshot content) are surfaced as a soft
        signal. No rollbacks — just a [SEMANTIC LINT] warning in the tool result.

        Graceful degradation: returns None if ruff is unavailable or no .py files.
        """
        from external_llm.agent.semantic_lint import ruff_findings

        new_findings: list = []
        for path, pre_content in snapshots.items():
            if not path.endswith(".py"):
                continue
            # New file (did not exist before): no pre-existing findings
            if pre_content is _MISSING_SNAP:
                pre_content = ""
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    post_content = f.read()
            except (OSError, FileNotFoundError):
                continue

            pre = ruff_findings(pre_content, path=path)
            post = ruff_findings(post_content, path=path)

            if not post:
                continue

            pre_set = {(f["code"], f["line"]) for f in pre}
            for f in post:
                if (f["code"], f["line"]) not in pre_set:
                    try:
                        rel = os.path.relpath(path, self.repo_root)
                    except ValueError:
                        rel = path
                    new_findings.append({
                        "code": f["code"],
                        "line": f["line"],
                        "message": f.get("message", ""),
                        "path": rel,
                    })

        if not new_findings:
            return None
        return "[SEMANTIC LINT]\n" + "\n".join(
            "  {}:L{} {} {}".format(
                f["path"], f["line"], f["code"], f["message"]
            )
            for f in new_findings
        )

    # ── Phase 2: deterministic semantic auto-repair ──────────────────────
    # Runs AFTER syntax verify passes (or after soft-fail preserves changes).
    # F821: project-wide import search → insert missing import line
    # Non-fatal: any failure here degrades gracefully to Phase 1 warning.
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_missing_import(
        self, name: str, repo_root: str, current_file: str
    ) -> Optional[str]:
        """Search project files for *name* as an imported symbol.

        Standalone version of planner lane's repair_f821._find_import_for_name.
        Searches repo_root/external_llm/agent, repo_root/external_llm, and
        repo_root for any file that imports ``name``. Returns the first matching
        import line, or None.

        Results are memoized via ``self._f821_import_cache`` (keyed on
        ``(name, current_file)`` to honour the cross-directory relative-import
        guard). The cache is reset per file in ``auto_repair_semantic``.
        """
        _cache_key = (name, current_file)
        # Membership test, not .get() — a None result (name not found anywhere)
        # must also be memoized: the not-found case is the most expensive one
        # (full scan of every search dir before giving up).
        if _cache_key in self._f821_import_cache:
            return self._f821_import_cache[_cache_key]

        import ast as _ast

        search_dirs = [
            os.path.join(repo_root, "external_llm", "agent"),
            os.path.join(repo_root, "external_llm"),
            repo_root,
        ]
        current_abs = os.path.join(repo_root, current_file)
        found: list = []

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            try:
                for fname in os.listdir(search_dir):
                    if not fname.endswith(".py") or fname.startswith("_"):
                        continue
                    fpath = os.path.join(search_dir, fname)
                    if os.path.abspath(fpath) == os.path.abspath(current_abs):
                        continue
                    try:
                        with open(fpath, encoding="utf-8", errors="replace") as f:
                            source = f.read()
                        tree = _ast.parse(source)
                        for node in tree.body:
                            if isinstance(node, _ast.ImportFrom):
                                for alias in node.names:
                                    actual = alias.asname or alias.name
                                    if actual == name:
                                        # Cross-directory relative-import guard: a relative
                                        # import (level > 0) copied from a source file in a
                                        # DIFFERENT directory than the target silently changes
                                        # its binding — `from .utils import X` in
                                        # `agent/foo.py` means `agent.utils`, but pasted into
                                        # `external_llm/bar.py` it means `external_llm.utils`
                                        # (wrong module / ImportError at runtime).
                                        # _import_line_resolves cannot catch this (relative
                                        # imports are unverifiable without package context).
                                        # Skip so the search continues to a same-directory or
                                        # absolute match.
                                        if (node.level or 0) > 0:
                                            _src_dir = os.path.dirname(os.path.abspath(fpath))
                                            _dst_dir = os.path.dirname(os.path.abspath(current_abs))
                                            if _src_dir != _dst_dir:
                                                break
                                        # Preserve relative-import level (node.level) — the
                                        # proven "." * level + module pattern used by
                                        # ast_op_executor / symbol_handlers_apply. Dropping
                                        # level rewrites a sibling's `from .pkg import X` into
                                        # the broken absolute `from pkg import X`
                                        # (ModuleNotFoundError at import time — the F821
                                        # silent-corruption failure mode).
                                        module = "." * (node.level or 0) + (node.module or "")
                                        if alias.asname:
                                            found.append(
                                                f"from {module} import {alias.name} as {alias.asname}"
                                            )
                                        else:
                                            found.append(f"from {module} import {name}")
                                        break
                            elif isinstance(node, _ast.Import):
                                for alias in node.names:
                                    actual = alias.asname or alias.name.split(".")[0]
                                    if actual == name:
                                        if alias.asname:
                                            found.append(
                                                f"import {alias.name} as {alias.asname}"
                                            )
                                        else:
                                            found.append(f"import {alias.name}")
                                        break
                        if found:
                            break
                    except (SyntaxError, OSError, AttributeError):
                        continue
            except (OSError, AttributeError):
                continue
            if found:
                break

        _result = found[0] if found else None
        self._f821_import_cache[_cache_key] = _result
        return _result

    @staticmethod
    def _insert_import_line(content: str, import_line: str) -> str:
        """Insert *import_line* at the correct position in *content*.

        Inserts after the last MODULE-LEVEL import statement, or after the
        module docstring, or at line 0. Uses AST for precision — substring
        matching avoided. Returns the modified content.

        Only module-level imports (direct children of ``tree.body``) are
        considered as insertion anchors. Walking *all* nesting levels
        (``ast.walk``) would pick a nested import inside ``if TYPE_CHECKING:``
        or a function body as the anchor; inserting a top-level import line at
        that nested ``end_lineno`` splits the block and produces
        ``unexpected indent`` SyntaxError — the dominant cause of the
        "Phase 2 F821 repair produced invalid syntax" warnings.

        If *content* has syntax errors (cannot be parsed by AST), returns
        *content* unchanged — unsafe string fallback is avoided because it
        cannot distinguish module-level imports from inline imports inside
        function bodies (the most common cause of F821 repair creating
        invalid syntax).
        """
        import ast as _ast

        try:
            tree = _ast.parse(content)
        except SyntaxError:
            logger.debug(
                "AUTO-REPAIR _insert_import_line: content has syntax errors — "
                "skipping import insertion (unsafe fallback would risk "
                "inserting into the middle of a function body)"
            )
            return content

        lines = content.split("\n")
        # Only module-level imports are valid anchors. Nested imports (inside
        # ``if TYPE_CHECKING:`` or function/class bodies) must NOT be used:
        # inserting a non-indented import_line there splits the block.
        last_import = None
        for node in tree.body:
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                last_import = node
        insert_idx = last_import.end_lineno if last_import else -1

        if insert_idx == -1:
            # No module-level imports — insert after module docstring, or at line 0
            insert_idx = 0
            if tree.body and isinstance(tree.body[0], _ast.Expr):
                first = tree.body[0]
                if isinstance(first.value, _ast.Constant) and isinstance(
                    first.value.value, str
                ):
                    insert_idx = first.end_lineno
            lines.insert(insert_idx, import_line)
        else:
            lines.insert(insert_idx, import_line)

        return "\n".join(lines)

    @staticmethod
    def _validate_python_syntax(content: str) -> bool:
        """Check that *content* is valid Python syntax.

        Used as safety net after Phase 2/3 repairs. Returns True if content
        compiles without SyntaxError. Does NOT execute the code.
        """
        try:
            compile(content, "<safety-net>", "exec")
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _import_line_resolves(import_line: str) -> bool:
        """Check whether *import_line* refers to an importable module.

        Validates ABSOLUTE imports via importlib.util.find_spec. RELATIVE
        imports (level > 0) cannot be validated without the importing
        module's package context and are accepted as-is. Returns False when
        an absolute import line names a module that cannot be found —
        inserting it would trade a loud F821 for a silent ModuleNotFoundError
        at import time (the silent-corruption failure mode of F821 repair).
        """
        import ast as _ast
        import importlib.util as _ilu
        try:
            _tree = _ast.parse(import_line)
        except SyntaxError:
            return False
        for _node in _tree.body:
            if isinstance(_node, _ast.ImportFrom):
                if getattr(_node, "level", 0) and _node.level > 0:
                    continue  # relative — unverifiable without package context
                _mod = _node.module or ""
                if not _mod:
                    continue
                try:
                    if _ilu.find_spec(_mod) is None:
                        return False
                except (ValueError, ImportError, ModuleNotFoundError):
                    return False
            elif isinstance(_node, _ast.Import):
                for _alias in _node.names:
                    _top = _alias.name.split(".")[0]
                    try:
                        if _ilu.find_spec(_top) is None:
                            return False
                    except (ValueError, ImportError, ModuleNotFoundError):
                        return False
        return True

    def auto_repair_semantic(self, snapshots: dict) -> int:
        """Deterministic auto-repair for ruff F-code findings.

        Phase 2+3 entry point. Called from tool_registry.dispatch() after
        syntax verify passes (or after soft-fail). For each .py file:
        - F821: resolves import via project search, inserts it
        - Phase 3: decl-loss symbol that now triggers F821 → auto-restore
        Note: F401 (unused import) auto-fix is intentionally disabled —
              removing a seemingly-unused import that was deliberately added
              is too aggressive. F401 is surfaced as a soft warning only.

        Returns count of files where at least one repair was applied.
        Non-fatal: any failure degrades gracefully to Phase 1 warning.
        """
        from external_llm.agent.semantic_lint import ruff_findings

        repaired_count = 0
        # Reset the _resolve_missing_import memo (see __init__) so this call
        # sees the repo's current on-disk import state.
        self._f821_import_cache = {}
        for path in snapshots:
            if not path.endswith(".py"):
                continue

            pre_content = snapshots[path]
            # New file (did not exist before edit): treat as empty content.
            # Required so _python_decl_sets / rollback paths don't choke on
            # the _MISSING_SNAP sentinel (cf. new_semantic_warnings L476-477).
            if pre_content is _MISSING_SNAP:
                pre_content = ""

            # Read current content (handler may have modified it)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    current = f.read()
            except (OSError, FileNotFoundError):
                continue

            post = ruff_findings(current, path=path)
            if not post:
                continue

            # Precondition: file must already parse cleanly.
            # If _ast.parse(current) fails the file has pre-existing syntax
            # errors and _insert_import_line can't find the right insertion
            # point — skip F821 repair entirely to avoid inserting imports
            # into the middle of a function body via unsafe string fallback.
            if not self._validate_python_syntax(current):
                logger.debug(
                    "AUTO-REPAIR skip F821 for %s — file has syntax errors",
                    path,
                )
                # Fall through to Phase 3 which has its own SyntaxError guard.
            else:
                # --- F821: resolve import ---
                # Filter from the 'post' list already computed above (L779)
                # instead of re-spawning ruff — current hasn't changed.
                f821_findings = [f for f in post if f.get("code") == "F821"]
                for finding in f821_findings:
                    # Extract the undefined name from the message
                    msg = finding.get("message", "")
                    # Ruff uses backticks: "Undefined name `Optional`"
                    # Try backticks first, fall back to single quotes
                    for _sep in ("`", "'"):
                        _parts = msg.split(_sep)
                        if len(_parts) >= 3:
                            missing_name = _parts[1]
                            break
                    else:
                        continue

                    # Compute relative path for resolver
                    try:
                        rel = os.path.relpath(path, self.repo_root)
                    except ValueError:
                        rel = path

                    import_line = self._resolve_missing_import(
                        missing_name, self.repo_root, rel
                    )
                    if not import_line:
                        continue
                    if import_line in current:
                        continue
                    if not self._import_line_resolves(import_line):
                        logger.debug(
                            "AUTO-REPAIR F821: skip import '%s' in %s — module does "
                            "not resolve (would trade a loud F821 for a silent "
                            "ModuleNotFoundError at import time)",
                            import_line, path,
                        )
                        continue

                    # Snapshot pre-import state for selective rollback: if the import
                    # insertion breaks syntax, we roll back ONLY the import, not the
                    # entire edit (which was already validated at L768).
                    _pre_import = current
                    current = self._insert_import_line(current, import_line)
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(current)
                    except OSError:
                        continue
                    # Safety net: validate syntax after repair, rollback on failure
                    if not self._validate_python_syntax(current):
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(_pre_import)
                        current = _pre_import  # Reset in-memory state after rollback
                        logger.warning("Phase 2 F821 repair produced invalid syntax in %s — rolled back", path)
                        continue
                    repaired_count += 1
                    # F821 protection: typing imports resolved by AST search may
                    # target symbols used only in deferred string annotations,
                    # which the import_normalizer's AST pass cannot detect.
                    # Mark them so the normalizer preserves them — otherwise the
                    # two passes oscillate (normalizer strips it → F821 returns).
                    if import_line.startswith("from typing import "):
                        try:
                            from external_llm.editor._editor_core.common.import_normalizer import mark_f821_protected
                            mark_f821_protected(path, missing_name)
                            logger.debug(
                                "AUTO-REPAIR marked '%s' as F821-protected in %s",
                                missing_name, path,
                            )
                        except Exception:
                            pass  # non-critical — worst case normalizer strips it once

            # --- Phase 3: decl-loss symbol → F821 → auto-restore ---
            # If a top-level symbol was removed AND that symbol is now flagged
            # as F821, it was an accidental deletion — restore from pre-snapshot.
            # Symbols are APPENDED at file-end to avoid line-number conflicts
            # after F401 fix (which removes import lines) shifted the document.
            pre_sets = _python_decl_sets(pre_content)
            post_sets = _python_decl_sets(current)
            if pre_sets is not None and post_sets is not None:
                pre_syms = pre_sets[0]
                post_syms = post_sets[0]
                removed_syms = pre_syms - post_syms
                if removed_syms:
                    remaining_f821 = ruff_findings(current, path=path)
                    # Ruff uses backticks: "Undefined name `parse_config`"
                    f821_undefined = set()
                    for _f in remaining_f821:
                        if _f.get("code") != "F821":
                            continue
                        _msg = _f.get("message", "")
                        for _sep in ("`", "'"):
                            _parts = _msg.split(_sep)
                            if len(_parts) >= 3:
                                f821_undefined.add(_parts[1])
                                break
                    to_restore = removed_syms & f821_undefined
                    if to_restore:
                        import ast as _ast
                        pre_lines = pre_content.split("\n")
                        try:
                            pre_tree = _ast.parse(pre_content)
                            # Build {name: (start_line, end_line)} from pre-tree
                            sym_ranges: dict = {}
                            for node in pre_tree.body:
                                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                                    sym_ranges[node.name] = (node.lineno, node.end_lineno)
                            # Collect symbol definitions in pre-snapshot line order
                            to_restore_sorted = sorted(
                                [s for s in to_restore if s in sym_ranges],
                                key=lambda s: sym_ranges[s][0],
                            )
                            # Build restore blocks
                            restore_blocks: list = []
                            for sym in to_restore_sorted:
                                start, end = sym_ranges[sym]
                                block_text = "\n".join(pre_lines[start - 1:end])
                                restore_blocks.append(block_text)
                            if restore_blocks:
                                # Append at file end (before trailing whitespace)
                                current = current.rstrip("\n") + "\n\n" + "\n\n".join(restore_blocks) + "\n"
                                # Safety net: validate syntax before writing Phase 3 restore
                                if not self._validate_python_syntax(current):
                                    logger.warning("Phase 3 decl-loss restore would produce invalid syntax in %s — skipped", path)
                                    continue
                                with open(path, "w", encoding="utf-8") as f:
                                    f.write(current)
                                repaired_count += 1
                                logger.info(
                                    "Phase 3: auto-restored symbols %s from pre-snapshot in %s",
                                    sorted(to_restore_sorted), path,
                                )
                        except SyntaxError:
                            continue

        return repaired_count
