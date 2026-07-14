"""Read-only tool handlers for ToolRegistry."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


class ReadToolsMixin:
    """Mixin providing read-only tool implementations for ToolRegistry."""

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
            if len(lines) > 200:
                return self._make_result(
                    ok=True,
                    content=(
                        f"`{path}` has {len(lines)} lines. "
                        f"Specify start_line and end_line to read a specific range."
                    ),
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

        return self._make_result(ok=True, content=result_content)

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
                    lines = [_item_ for _, _item_ in sorted(zip(_scores, lines, strict=False), reverse=True)]

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
            return self._make_result(ok=True, content=f"No definitions found for '{name}'.")

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

        lines.append("\nUse bash with cat or sed to examine specific symbols (e.g. `sed -n 'X,Yp' <file>`).")
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
        lines.append("\nUse bash with cat or sed to inspect these files.")
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

