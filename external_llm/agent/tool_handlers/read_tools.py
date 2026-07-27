"""Read-only tool handlers for ToolRegistry."""
from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.thresholds import config as _cfg
from ..rag_configs import CodeTokenizer
from ..rag_searcher import _bm25_score as _bm25

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)


# ── Indent gutter for read_file output ─────────────────────────────────────
# The agent's write tools (edit_text/anchor_edit/modify_symbol) frequently fail
# or trigger retry loops because the LLM cannot reliably count leading
# whitespace from a plain ``"  NNN  code"`` dump — the line-number padding
# visually merges with the code's own indentation. We inject an explicit
# ``│N│`` gutter (leading-whitespace column count) between the line number and
# the code so the exact indentation is a readable number. The U+2502 box-drawing
# bar never appears at column 0 of a real source line, so a naive LLM copy of
# the line (which starts at the code, past the gutter) cannot accidentally
# include it — the format is copy-safe by construction. See design insight:
# expose indent as structured metadata, not something to be inferred.
_INDENT_GUTTER_BAR = "│"  # U+2502 — box-drawing vertical, never a valid code prefix

# Method names listed per class in read_file's over-cap outline. Matches the
# get_file_outline tool's own cap so the two views of a file agree.
_METHODS_PER_CLASS = 15


# ── File-extension → language-label map (shared by read_file / read_symbol) ──
# Extracted to a module-level constant so the hot path (read_file, the most
# frequently called tool) does not allocate a fresh dict literal on every call.
_EXT_LANG_MAP = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "go": "go", "java": "java", "kt": "kotlin", "rs": "rust",
    "md": "markdown", "yaml": "yaml", "yml": "yaml",
    "json": "json", "css": "css", "html": "html",
    "sh": "bash", "bash": "bash", "zsh": "bash",
    "sql": "sql", "xml": "xml", "svg": "xml",
}


def _format_numbered_line(lineno: int, line: str) -> str:
    """Format one source line as ``"  NNN │N│ code"`` with an indent gutter.

    The gutter value ``N`` is the count of leading whitespace characters
    (spaces + tabs counted as width 1 each — the same metric write tools use to
    compute ``min_indent``/``detect_indent_char`` in common/indent_utils). Empty
    lines show ``0``. The bar is U+2502 so it is visually distinct from ASCII
    ``|`` used in code (e.g. type unions, bitwise-or) and uncopyable as a line
    prefix.
    """
    indent = len(line) - len(line.lstrip()) if line.strip() else 0
    return f"{lineno:>6} {_INDENT_GUTTER_BAR}{indent:>2}{_INDENT_GUTTER_BAR} {line}"


def _split_source_lines(text: str) -> list[str]:
    r"""Split ``text`` into lines using ``\n`` only — matching ``ast.lineno`` /
    ``ast.end_lineno`` and git/unified-diff line numbering.

    ``str.splitlines()`` additionally treats ``\f`` (form-feed), ``\v``,
    ``\x1c``–``\x1e``, ``\x85``, ``\u2028`` (line separator) and ``\u2029``
    (paragraph separator) as line breaks. ``read_symbol`` indexes the resulting
    list with ``sym.line`` / ``sym.end_line``, which originate from
    ``ast.lineno`` (``\n``-only). For a source file containing any of those
    extra characters the two models disagree, so read_symbol would slice and
    DISPLAY THE WRONG LINES. Splitting on ``\n`` and dropping the trailing
    empty element (from a final ``\n``) keeps the line count aligned with the
    AST/git model.

    NOTE: ``read_file`` intentionally keeps ``str.splitlines()`` because its
    line numbers are consumed by anchor_edit's ``anchor_ast_lineno`` mode,
    which builds its own ``splitlines()`` array — changing one without the
    other would desync them. read_symbol's line numbers come from the AST,
    not from a caller, so it is safe (and correct) to align it here.
    """
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


@functools.lru_cache(maxsize=256)
def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate a glob into a separator-aware regex.

    ``fnmatch.translate`` is unusable here: its ``*`` also matches ``/``, so
    ``src/*.py`` would wrongly match ``src/a/b/c.py``. ``PurePath.match`` does
    not support recursive ``**`` before 3.13 and ``glob.translate`` is 3.13+,
    while this package supports 3.10 — hence a local translator.

    Semantics (POSIX glob, the shape every agent already knows):
      ``**/`` zero or more directories · ``**`` any characters incl. ``/``
      ``*`` any run of non-``/`` · ``?`` one non-``/`` · ``[abc]`` a class

    Memoised: one glob call matches the pattern against every path in the repo
    index, and agents re-issue the same handful of patterns across turns.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")       # `**/x` must also match a bare `x`
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:                    # unterminated class → literal '['
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1:j]
                if body.startswith(("!", "^")):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


class ReadToolsMixin:
    """Mixin providing read-only tool implementations for ToolRegistry."""

    # Above this many matches the mtime sort is skipped (it stats every hit).
    _GLOB_MTIME_SORT_LIMIT = 2000

    def _tool_glob(self, args: dict[str, Any]) -> "ToolResult":
        """List repository files matching a glob pattern, newest first.

        Fills the gap that made ``bash ls``/``find`` the only way to answer
        "what files are here?" — a path that leaves the repo boundary, returns
        unbounded output, and cannot be result-cached. The file set comes from
        ``git ls-files`` (``.gitignore``-aware, NUL-separated so non-ASCII paths
        survive), falling back to a pruned walk outside a git checkout.
        """
        import os
        import time

        pattern = str(args.get("pattern", "") or "").strip()
        if not pattern:
            return self._make_result(ok=False, content="", error="'pattern' is required")

        scope = str(args.get("path", "") or "").strip()
        max_results = max(1, min(int(args.get("max_results", 200) or 200), 1000))

        root = Path(self._effective_repo_root)
        if scope:
            scope = self._correct_bias_path(scope)
            scoped = self._secure_path(scope, confine=True)
            if scoped is None:
                return self._make_result(
                    ok=False, content="",
                    error=f"path {scope!r} is outside the repository",
                )
            try:
                prefix = scoped.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                return self._make_result(
                    ok=False, content="",
                    error=f"path {scope!r} is outside the repository",
                )
            prefix = "" if prefix == "." else prefix.rstrip("/") + "/"
        else:
            prefix = ""

        # Reuse the TTL-cached repo index the write tools already maintain,
        # so a glob costs a dict lookup rather than another `git ls-files`.
        from .write_tools import _repo_file_index
        paths = _repo_file_index(str(root))

        # A pattern with no separator matches the BASENAME anywhere ("*.py"
        # finds every .py file), which is what both humans and models mean by
        # it. Patterns containing "/" are matched against the full repo-
        # relative path.
        rx = _glob_to_regex(pattern)
        basename_only = "/" not in pattern

        matches: list[str] = []
        for rel in paths:
            if prefix and not rel.startswith(prefix):
                continue
            target = os.path.basename(rel) if basename_only else rel
            if rx.match(target):
                matches.append(rel)

        if not matches:
            _scope_note = f" under {prefix.rstrip('/')!r}" if prefix else ""
            return self._make_result(
                ok=True,
                content=f"No files match {pattern!r}{_scope_note}.",
            )

        truncated = False
        if len(matches) <= self._GLOB_MTIME_SORT_LIMIT:
            # Newest first: "what was touched recently" is the question a glob
            # is usually standing in for.
            def _mtime(rel: str) -> float:
                try:
                    return (root / rel).stat().st_mtime
                except OSError:
                    return 0.0
            matches.sort(key=_mtime, reverse=True)
        # else: already path-sorted by _repo_file_index — deterministic, and
        # stat()ing thousands of files to order a list nobody will read whole
        # is not worth it.

        if len(matches) > max_results:
            truncated = True
            shown = matches[:max_results]
        else:
            shown = matches

        header = f"{len(matches)} file(s) match {pattern!r}"
        if prefix:
            header += f" under {prefix.rstrip('/')!r}"
        if truncated:
            header += f" — showing the first {len(shown)}"
        _now = time.time()
        lines = [header]
        for rel in shown:
            try:
                age_days = (_now - (root / rel).stat().st_mtime) / 86400.0
                lines.append(f"  {rel}  ({age_days:.0f}d)")
            except OSError:
                lines.append(f"  {rel}")
        return self._make_result(ok=True, content="\n".join(lines))

    def _tool_read_file(self, args: dict[str, Any]) -> "ToolResult":
        """Read a file by path with optional line range.

        Output prefixes each line with its 1-based number AND an indent gutter
        ``│N│`` (leading-whitespace column count) so the exact indentation of
        every line is readable at a glance — eliminating the guesswork that
        causes indent mismatches in edit_text/anchor_edit/modify_symbol.
        Example: ``   121 │ 4│     return x``  (4 leading spaces).
        """
        path = args.get("path", "").strip()
        if not path:
            return self._make_result(ok=False, content="", error="'path' is required")

        abs_path = self._secure_path(path)
        if abs_path is None:
            return self._make_result(ok=False, content="", error=f"Path not found or outside repo: {path!r}")
        if not abs_path.is_file():
            return self._make_result(ok=False, content="", error=f"Not a file: {path!r}")

        try:
            lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to read {path!r}: {e}")

        start_line = args.get("start_line")
        end_line = args.get("end_line")

        if start_line is None and end_line is None:
            if len(lines) > _cfg.lines.READ_FILE_FULL_LINES:
                return self._make_result(
                    ok=True,
                    content=self._over_cap_guidance(path, len(lines)),
                    metadata={"over_line_cap": True, "line_count": len(lines)},
                )
            s, e = 1, len(lines)
        else:
            s = max(1, int(start_line or 1))
            e = min(len(lines), int(end_line or len(lines)))
            if s > len(lines) or s > e:
                return self._make_result(
                    ok=True,
                    content=f"Line range {s}–{e} is out of range (file has {len(lines)} lines).",
                )

        numbered_lines = [_format_numbered_line(i, ln) for i, ln in enumerate(lines[s - 1 : e], start=s)]
        # Char budget. Applied here rather than only on the no-range path
        # because an explicit range is the documented way around the line cap,
        # so it is the path most able to overrun the context window.  Truncate
        # on a line boundary and name the resumption line, so continuing is one
        # unambiguous call rather than another guess.
        truncated_at: int | None = None
        partial_line: int | None = None  # line emitted only as a prefix (mid-line truncation)
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        if sum(len(ln) + 1 for ln in numbered_lines) > budget:
            kept: list[str] = []
            used = 0
            for i, ln in enumerate(numbered_lines):
                used += len(ln) + 1
                if used > budget:
                    truncated_at = s + i  # first line NOT emitted
                    break
                kept.append(ln)
            # A single line wider than the whole budget would otherwise emit
            # nothing and report a resumption line that never advances. Emit a
            # prefix of it and advance past it, but flag that its tail was
            # dropped mid-line: start_line is line-granular, so re-reading
            # cannot recover the rest. Without this signal a caller mistakes
            # the partial line for the full line and edits on a truncated view
            # (minified JS/CSS, single-line JSON, base64 blobs).
            if not kept:
                partial_line = s
                kept = [numbered_lines[0][:budget]]
                truncated_at = s + 1
            numbered_lines = kept
            e = truncated_at - 1
        content = "\n".join(numbered_lines)

        lang = path.split(".")[-1] if "." in path else ""
        lang_label = _EXT_LANG_MAP.get(lang, "")

        result_content = f"`{path}` ({len(lines)} lines) — `│N│` = leading-indent column count"
        if start_line is not None or end_line is not None:
            result_content += f" lines {s}–{e}"
        if lang_label:
            result_content += f"\n```{lang_label}\n{content}\n```"
        else:
            result_content += f"\n```\n{content}\n```"
        if truncated_at is not None:
            if partial_line is not None:
                result_content += (
                    f"\n\n[Truncated at the {budget:,}-char output budget. "
                    f"Line {partial_line} alone exceeds it, so only its first "
                    f"{budget:,} chars were returned; the REST OF THAT LINE was "
                    f"dropped and is NOT recoverable by re-reading (start_line is "
                    f"line-granular). Continue at start_line={truncated_at} for the "
                    f"next line.]"
                )
            else:
                result_content += (
                    f"\n\n[Truncated at the {budget:,}-char output budget. "
                    f"Lines {truncated_at}–{len(lines)} were not returned — "
                    f"call read_file again with start_line={truncated_at}.]"
                )

        meta: dict[str, Any] = {}
        if truncated_at is not None:
            meta = {"truncated": True, "resume_line": truncated_at, "line_count": len(lines)}
            if partial_line is not None:
                meta["partial_line"] = partial_line
        return self._make_result(ok=True, content=result_content, metadata=meta)

    def _over_cap_guidance(self, path: str, line_count: int) -> str:
        """Message for a no-range read of a file past ``READ_FILE_FULL_LINES``.

        Carries the outline, not just the line count.  The bare count made the
        model choose ``start_line``/``end_line`` blind, so it typically spent
        two or three reads homing in; with the symbol map it can name the range
        it wants on the next call.  Falls back to the plain count when the
        outline is empty (unsupported language, parse failure) so this path can
        never be worse than what it replaces.
        """
        head = (
            f"`{path}` has {line_count} lines — too long to return whole "
            f"(cap {_cfg.lines.READ_FILE_FULL_LINES})."
        )
        try:
            symbols = self._symbol_searcher.get_file_outline(path)
        except Exception:
            logger.debug("read_file: outline failed for %s", path, exc_info=True)
            symbols = []
        if not symbols:
            return head + " Call read_file again with start_line and end_line."

        cap = _cfg.lines.READ_FILE_OUTLINE_MAX_SYMBOLS
        rows: list[str] = []
        for s in symbols[:cap]:
            rows.append(f"  line {s.line:>5}  [{s.kind}] {s.name}")
            # Methods carry no line of their own in the outline, so listing the
            # NAMES is what makes them reachable: read_symbol takes a name, and
            # a file that is one 2000-line class would otherwise offer nothing
            # between "class at line 350" and a blind range.
            methods = s.methods or []
            if methods:
                shown = ", ".join(methods[:_METHODS_PER_CLASS])
                more = f" (+{len(methods) - _METHODS_PER_CLASS} more)" if len(methods) > _METHODS_PER_CLASS else ""
                rows.append(f"           methods: {shown}{more}")
        if len(symbols) > cap:
            rows.append(f"  … {len(symbols) - cap} more symbols (get_file_outline for the full map)")

        return (
            head
            + "\n\nOutline:\n"
            + "\n".join(rows)
            + "\n\nNext: read_symbol with a name above (exact, no range needed), "
              "or read_file with start_line/end_line."
        )

    def _tool_grep(self, args: dict[str, Any]) -> "ToolResult":
        """Search for a pattern across files using grep (or ripgrep if available)."""
        import shutil
        import subprocess

        # Safety limit: ~33k tokens max per result (prevent token explosion from context+N on long lines)
        # Match bash tool's BASH_OUTPUT_MAX_CHARS threshold for consistency.
        from ..config.thresholds import config as _thresholds
        _MAX_RESULT_CHARS = _thresholds.tokens.BASH_OUTPUT_MAX_CHARS

        pattern = args.get("pattern", "").strip()
        if not pattern:
            return self._make_result(ok=False, content="", error="'pattern' is required")

        search_path = args.get("path", "").strip() or "."
        search_path = self._correct_bias_path(search_path)
        max_results = min(int(args.get("max_results", 200)), 500)
        context = int(args.get("context", 0))
        ignore_case = args.get("ignore_case", False)
        include = args.get("include", "").strip()

        # Detect regex special chars — safe patterns use -F (fixed string)
        _re = __import__("re")
        _has_regex = bool(_re.search(r"[.+*?\[\]{}()|\\^$]", pattern))
        use_fixed = not _has_regex

        # ── Prefer ripgrep (rg) over system grep ──
        _rg = shutil.which("rg")
        use_rg = _rg is not None

        for _attempt in range(2):
            if use_rg:
                cmd = [_rg, "-n", "--no-heading"]
                if ignore_case:
                    cmd.append("-i")
                if context > 0:
                    cmd.extend(["-C", str(context)])
                if include:
                    cmd.extend(["--glob", include])
                if search_path in (".", self.repo_root):
                    cmd.extend(["--glob", "!.asicode/**", "--glob", "!design_sessions/**", "--glob", "!logs/**"])
                if use_fixed:
                    cmd.append("-F")
                cmd.append("--")
                cmd.append(pattern)
                cmd.append(search_path)
            else:
                cmd = ["grep", "-rn"]
                if ignore_case:
                    cmd.append("-i")
                if context > 0:
                    cmd.extend(["-C", str(context)])
                if include:
                    cmd.extend(["--include", include])
                if search_path in (".", self.repo_root):
                    cmd.extend(["--exclude-dir=.asicode", "--exclude-dir=design_sessions", "--exclude-dir=logs"])
                if use_fixed:
                    cmd.append("-F")
                else:
                    cmd.append("-E")
                cmd.append("--")
                cmd.append(pattern)
                cmd.append(search_path)

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.repo_root,
                    capture_output=True, text=True, timeout=120,
                )
            except subprocess.TimeoutExpired:
                return self._make_result(ok=True, content=f"grep timed out (pattern={pattern!r})")
            except Exception as e:
                return self._make_result(ok=False, content="", error=f"grep failed: {e}")

            if proc.returncode != 2 or use_fixed:
                break  # success or non-regex error — done
            # Exit code 2 = regex syntax error → retry as fixed string
            use_fixed = True

        if proc.returncode == 0 or (proc.returncode == 1 and proc.stdout.strip()):
            lines = proc.stdout.splitlines()

            # BM25 ranking: re-rank FLAT match-lines (context==0) by relevance to
            # the search pattern. Each match line is treated as a pseudo-document
            # and scored against the query tokens, so lines with richer token
            # overlap rank higher and survive the cap, rather than filesystem-order.
            #
            # CRITICAL: only rank when context==0. With context>0 the grep/rg
            # output is spatially grouped — match lines (path:line:), context
            # lines (path-line-), and group separators (--) — whose meaning is
            # entirely positional. Re-ordering each line independently by score
            # DESTROYS that grouping: context lines detach from their match, line
            # numbers shuffle out of order, and separators float to meaningless
            # spots (the more context requested, the worse the scramble). Native
            # group order must be preserved. See test_grep_context_* regression.
            if len(lines) > 1 and context == 0:
                from collections import Counter
                _tok = CodeTokenizer()
                _qtokens = _tok.tokenize(pattern)
                if _qtokens:
                    # Pre-truncate to bound BM25 cost for pathological match sets.
                    # BM25 on 50k+ lines is O(n*q) CPU — pre-cutting to a sensible
                    # cap keeps ranking quality (top N out of shuffled filesystem
                    # order ≅ top N out of K*N) while bounding worst-case time.
                    # We take only the first max_results*20 or 5000 (whichever is
                    # larger) lines, which comfortably covers the top max_results
                    # (≤500) after re-ranking.  The tail is appended unsorted at the
                    # end — it will be truncated away by the max_results cap below.
                    _bm25_cap = max(max_results * 20, 5000)
                    if len(lines) > _bm25_cap:
                        _tail = lines[_bm25_cap:]
                        lines = lines[:_bm25_cap]
                    else:
                        _tail = []
                    _tokenized = [_tok.tokenize(_item_) for _item_ in lines]
                    _doc_tc: list[dict[str, int]] = [dict(Counter(t)) for t in _tokenized]
                    _doc_lens = [len(t) for t in _tokenized]
                    _n = len(lines)
                    _avgdl = sum(_doc_lens) / _n
                    _df: dict[str, int] = {}
                    for qt in _qtokens:
                        _df[qt] = sum(1 for tc in _doc_tc if qt in tc)
                    _scores = [
                        _bm25(_qtokens, _doc_tc[i], _doc_lens[i], _df, _n, _avgdl)
                        for i in range(_n)
                    ]
                    lines = [lines[i] for i in sorted(range(_n), key=lambda i: _scores[i], reverse=True)]
                    if _tail:
                        lines.extend(_tail)

            truncated = len(lines) > max_results
            total = len(lines)

            # --- Character-based truncation guard: prevent token explosion ---
            # context=N + long-line files (logs, JSON, stacktraces) can produce
            # massive output even with few matches.  Enforce a hard char limit.
            display_chars = 0
            display_lines = []
            for _item_ in lines[:max_results]:
                display_chars += len(_item_) + 1  # +1 for newline
                if display_chars > _MAX_RESULT_CHARS:
                    # Include this line but stop; next loop break is informational
                    display_lines.append(_item_)
                    break
                display_lines.append(_item_)
            display = "\n".join(display_lines)
            char_truncated = display_chars > _MAX_RESULT_CHARS

            tool_name = "rg" if use_rg else "grep"
            result = f"{tool_name}: {pattern!r} in {search_path} ({total} match{'es' if total != 1 else ''})"
            if context > 0:
                result += f" ({context} context lines)"
            result += f"\n{display}"
            if char_truncated:
                result += f"\n... (truncated at {_MAX_RESULT_CHARS:,} characters — {len(display_lines)} of {total} matches shown). For log files, use `bash grep -n 'pattern' file` then `read_file` with exact line range — drastically reduces tokens."
            elif truncated:
                result += f"\n... (truncated to {max_results} of {total} matches — refine your pattern)"

            return self._make_result(ok=True, content=result)
        elif proc.returncode == 1:
            tool_name = "rg" if use_rg else "grep"
            return self._make_result(
                ok=True,
                content=f"{tool_name}: {pattern!r} in {search_path} — no matches.",
            )
        else:
            stderr = (proc.stderr or "").strip()[:500]
            return self._make_result(
                ok=False, content="",
                error=f"grep failed (exit={proc.returncode}): {stderr}",
            )

    def _tool_read_symbol(self, args: dict[str, Any]) -> "ToolResult":
        """Read a symbol definition (function, class, or variable) by name.

        When SymbolDef.end_line is available (AST end_lineno), read the full
        symbol body — not just a fixed ±context_lines window — so the result
        covers the whole definition even for long functions/classes.
        """
        name = args.get("name", "")
        if not name:
            return self._make_result(ok=True, content="Symbol name is required.")
        file_path = args.get("file_path") or None
        context_lines = int(args.get("context_lines", 10))

        defs = self._symbol_searcher.find_symbol(name, search_path=file_path)
        if not defs:
            return self._make_result(ok=True, content=f"Symbol '{name}' not found.")
        sym = defs[0]

        abs_path = Path(self.repo_root) / sym.file
        if not abs_path.exists():
            return self._make_result(ok=True, content=f"File '{sym.file}' not found.")

        lines = _split_source_lines(abs_path.read_text(encoding="utf-8", errors="replace"))
        if sym.end_line and sym.end_line >= sym.line:
            # Full body: leading context (covers decorators) + trailing context.
            start = max(0, sym.line - 1 - context_lines)
            end = min(len(lines), sym.end_line + context_lines)
        else:
            # Fallback: fixed window around the definition line.
            start = max(0, sym.line - 1 - context_lines)
            end = min(len(lines), sym.line + context_lines)
        context = "\n".join(
            _format_numbered_line(i, ln)
            for i, ln in enumerate(lines[start:end], start=start + 1)
        )

        lang = sym.file.split(".")[-1] if "." in sym.file else ""
        lang_label = _EXT_LANG_MAP.get(lang, lang)

        loc = f"{sym.file}:{sym.line}"
        if sym.end_line and sym.end_line > sym.line:
            loc += f"-{sym.end_line}"
        content = (f"**{sym.kind}** `{name}` defined in `{loc}` — `│N│` = leading-indent column count\n"
                   f"```{lang_label}\n{context}\n```")
        return self._make_result(ok=True, content=content)

    def _tool_find_symbol(self, args: dict[str, Any]) -> "ToolResult":
        name = args.get("name", "").strip()
        if not name:
            return self._make_result(ok=False, content="", error="'name' is required")
        kind = args.get("kind", "any")
        search_path = args.get("search_path")
        include_inheritance = bool(args.get("include_inheritance", False))

        defs = self._symbol_searcher.find_symbol(name, kind=kind, search_path=search_path)
        if not defs:
            # Distinguish "symbol genuinely absent" from "the file index was
            # truncated at the cap, so the definition may live in an
            # un-indexed file". Without this note the agent treats a silent
            # truncation as proof of absence (fail-silent → re-creating an
            # existing symbol or giving up).
            _trunc = self._symbol_searcher.index_was_truncated(search_path)
            _note = ""
            if _trunc:
                _note = (
                    " NOTE: the file index was truncated at its cap, so this "
                    "symbol may exist in an un-indexed file — try grep/bash, "
                    "narrow search_path, or the cap may need raising."
                )
            return self._make_result(
                ok=True, content=f"No definitions found for '{name}'.{_note}"
            )

        lines: list[str] = [f"Found {len(defs)} definition(s) for '{name}':\n"]
        for d in defs:
            lines.append(f"  [{d.kind}] {d.file}:{d.line}")
            if d.signature:
                lines.append(f"    signature : {d.signature}")
            if d.docstring:
                lines.append(f"    docstring : {d.docstring[:100]}")
            if d.bases:
                lines.append(f"    bases     : {', '.join(d.bases)}")
            if d.methods:
                methods_str = ", ".join(d.methods[:10])
                suffix = f" (+{len(d.methods)-10} more)" if len(d.methods) > 10 else ""
                lines.append(f"    methods   : {methods_str}{suffix}")
            if d.decorators:
                lines.append(f"    decorators: {', '.join(d.decorators)}")
            lines.append("")

        # include_inheritance: enrich first result with subclasses + references
        if include_inheritance and defs:
            info = self._symbol_searcher.get_symbol_info(
                name, file_path=search_path, kind=kind, defs=defs
            )
            if info:
                if "subclasses" in info:
                    lines.append(f"Subclasses : {', '.join(info['subclasses'])}")
                lines.append(f"References : {info.get('reference_count', 0)}")
                if "referenced_in" in info:
                    lines.append(f"Used in    : {', '.join(info['referenced_in'])}")
                if "sample_references" in info:
                    lines.append("\nSample references:")
                    for sr in info["sample_references"]:
                        lines.append(f"  {sr['file']}:{sr['line']}  {sr['context'][:80]}")
                if "other_definitions" in info:
                    lines.append("\nOther definitions:")
                    for od in info["other_definitions"]:
                        lines.append(f"  [{od['kind']}] {od['file']}:{od['line']}")

        return self._make_result(ok=True, content="\n".join(lines))

    def _tool_find_references(self, args: dict[str, Any]) -> "ToolResult":
        name = (args.get("name") or args.get("symbol") or "").strip()
        if not name:
            return self._make_result(ok=False, content="", error="'name' (or 'symbol') is required")
        search_path = args.get("search_path")
        include_definitions = bool(args.get("include_definitions", False))

        refs = self._symbol_searcher.find_references(
            name, search_path=search_path, include_definitions=include_definitions
        )
        if not refs:
            return self._make_result(ok=True, content=f"No references found for '{name}'.")

        lines: list[str] = [f"Found {len(refs)} reference(s) for '{name}':\n"]
        for r in refs:
            lines.append(f"  {r.file}:{r.line}:{r.col}  {r.context}")

        return self._make_result(ok=True, content="\n".join(lines))

    def _tool_get_file_outline(self, args: dict[str, Any]) -> "ToolResult":
        path = args.get("path", "").strip()
        if not path:
            return self._make_result(ok=False, content="", error="'path' is required")

        abs_path = self._secure_path(path)
        if abs_path is None:
            return self._make_result(ok=False, content="", error=f"Path not found or outside repo: {path!r}")

        symbols = self._symbol_searcher.get_file_outline(path)
        if not symbols:
            return self._make_result(ok=True, content=f"No symbols found in '{path}' (file may be empty or unsupported language).")

        lines: list[str] = [f"File outline: {path} ({len(symbols)} symbols)\n"]
        for s in symbols:
            prefix = f"  [{s.kind}] {s.name}"
            loc = f"(line {s.line})"
            if s.kind == "class":
                detail = ""
                if s.bases:
                    detail += f" — bases: {', '.join(s.bases)}"
                lines.append(f"{prefix} {loc}{detail}")
                if s.methods:
                    m_str = ", ".join(s.methods[:15])
                    suffix = f" (+{len(s.methods)-15} more)" if len(s.methods) > 15 else ""
                    lines.append(f"    methods: {m_str}{suffix}")
            elif s.kind in ("function", "async_function"):
                sig = f"({s.signature})" if s.signature else ""
                lines.append(f"{prefix}{sig} {loc}")
            elif s.kind == "variable":
                sig = f" — {s.signature}" if s.signature else ""
                lines.append(f"{prefix} {loc}{sig}")
            else:
                sig = f" — {s.signature}" if s.signature else ""
                lines.append(f"{prefix} {loc}{sig}")

        # Point at the read tools, not at `cat`/`sed`. Their output carries the
        # `│N│` indent gutter that write tools depend on for correct
        # old_string/indentation, and it goes through _secure_path; a raw shell
        # dump has neither, so steering here was training the model out of the
        # repo's own safety net.
        lines.append("\nUse read_symbol to pull one of these by name, or read_file with start_line/end_line.")
        return self._make_result(
            ok=True, content="\n".join(lines),
            metadata={"path": path, "symbol_count": len(symbols)},
        )

    def _tool_find_relevant_files(self, args: dict[str, Any]) -> "ToolResult":
        query = args.get("query", "").strip()
        if not query:
            return self._make_result(ok=False, content="", error="'query' is required")
        top_k = max(1, min(int(args.get("top_k", 5)), 15))
        file_glob = args.get("file_glob") or None

        results = self._rag_searcher.find_relevant_files(query, top_k=top_k, file_glob=file_glob)
        logger.debug(
            "RAG search invoked: query=%s results=%d",
            query,
            len(results)
        )
        if not results:
            return self._make_result(ok=True, content="No relevant files found for the given query.")

        lines: list[str] = [f"Top {len(results)} relevant file(s) for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.file}  (score: {r.score:.2f}, line ~{r.line})")
            if r.snippet.strip():
                lines.append(f"     {r.snippet[:110]}")
        lines.append("\nUse read_file to inspect these, or get_file_outline first if a file is large.")
        return self._make_result(
            ok=True, content="\n".join(lines),
            metadata={"files_found": [r.file for r in results], "result_count": len(results)},
        )

    def _tool_read_image(self, args: dict[str, Any]) -> "ToolResult":
        """Read text from an image file using OCR."""
        path = args.get("path", "").strip()
        if not path:
            return self._make_result(ok=False, content="", error="'path' is required")

        abs_path = self._secure_path(path)
        if abs_path is None:
            return self._make_result(ok=False, content="", error=f"Path not found or outside repo: {path!r}")
        if not abs_path.is_file():
            return self._make_result(ok=False, content="", error=f"Not a file: {path!r}")

        try:
            import base64 as _b64
            data = _b64.b64encode(abs_path.read_bytes()).decode("utf-8")
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to read image file {path!r}: {e}")

        try:
            from external_llm.providers import _try_ocr_base64 as _ocr_fn
            ocr_text = _ocr_fn(data)
        except ImportError:
            return self._make_result(
                ok=True,
                content="OCR libraries (pytesseract or Pillow) are not installed. "
                        "Install with: pip install pytesseract Pillow",
            )

        if ocr_text:
            return self._make_result(
                ok=True,
                content=f"[Image OCR — {abs_path.name}]\n{ocr_text}",
            )
        else:
            return self._make_result(
                ok=True,
                content=f"[Image OCR — {abs_path.name}] No text detected in the image. "
                        "The image may contain only graphics without text, or OCR could not read it.",
            )

