"""
StreamingDisplay — Terminal UI for real-time collaboration visualization.

Renders Claude Code Agent activity in asicode's native CLI style:

  [1] ○ read_file  ui_routes.py          ← running (in-place, no newline)
  [  0.9s] ✓ read_file  ui_routes.py     ← overwritten on completion
  ▌ UI file location confirmed — ...          ← Claude's own words (mauve gutter bar)

  ✓ UI files are in the ui/ directory…            ← structured verdict
     confidence 97%

Colors mirror asi's Catppuccin Mocha palette (truecolor ANSI).
The mauve gutter bar (▌) marks every line spoken by the Claude Code Agent,
so its words are visually distinct from asicode's own output.
"""
from __future__ import annotations

import logging
import re
import shutil
import sys
import threading
import time
import unicodedata
from typing import Any, Optional

from external_llm.agent.terminal_coordination import TERM_WRITE_LOCK, set_row_pending

from .claude_session import SessionEvent
from .collaboration_orchestrator import _STRIP_XML_RE

# Internal SDK tools that should be hidden from display
_INTERNAL_TOOLS: set[str] = {"StructuredOutput", "output", "output_json"}

logger = logging.getLogger(__name__)



def _fg(hex_color: str) -> str:
    """Hex color → truecolor ANSI foreground escape."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


# Catppuccin Mocha — same hex as asi._C
_MAUVE = _fg("#cba6f7")   # Claude Code Agent identification color
_GREEN = _fg("#a6e3a1")
_RED = _fg("#f38ba8")
_YELLOW = _fg("#f9e2af")
_TEXT = _fg("#cdd6f4")
_MUTED = _fg("#6c7086")
_BORDER = _fg("#313244")
_BOLD = "\033[1m"
_RESET = "\033[0m"

_RULE_WIDTH = 56

# Spinner chars for long-running display — same as asi._ProgressPrinter
_SPIN_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StreamingDisplay:
    """Real-time terminal display for collaboration sessions.

    Native asicode tool format: a single in-place live line shows ALL
    currently running tools ("[2,3] ○ read_file · grep") and is cleared
    before any permanent line ("[  1.2s] ✓ tool", text, verdict) prints,
    then re-rendered if tools are still pending. One live line handles
    both sequential and parallel tool calls without orphaned ○ lines.
    Claude Code Agent text is marked with a mauve ▌ gutter bar.
    """

    def __init__(self, verbose: bool = False, output_file: Optional[str] = None):
        self._verbose = verbose
        self._output_file = output_file
        self._line_buffer: list[str] = []
        self._start_time = time.perf_counter()
        self._verdict_summary: Optional[str] = None
        self._call_seq: int = 0
        # Text already printed as body — used for verdict details dedup
        self._text_seen: list[str] = []
        #progress during/middle tool: tool_id → {"seq", "name", "hint", "t0"}
        self._pending: dict[str, dict[str, Any]] = {}
        # Whether the last terminal line is a live ○ line (can be overwritten)
        self._live: bool = False
        # ── Long-running ticker (same UX as design chat _ProgressPrinter) ──
        # The event callback (asyncio thread) and ticker thread both operate on the
        # same live line, so all stdout rendering is done inside _lock only.
        self._lock = threading.Lock()
        # Serialize _start_ticker's check-and-start — handle_event (asyncio callback
        # thread, *before* acquiring self._lock) and print_header (orchestrator thread,
        # *outside* self._lock) can both call _start_ticker concurrently. Without
        # serialization, both threads observe "no ticker" and each spawns a thread;
        # the second one overwrites self._ticker_thread, causing the first to detach
        # from join. The orphaned ticker (daemon) keeps overwriting the live line
        # every 0.25s until process exit.
        self._ticker_launch_lock = threading.Lock()
        self._spin_i: int = 0  # Spinner frame — ticker increments, render always uses current value
        self._ticker_stop: Optional[threading.Event] = None
        self._ticker_thread: Optional[threading.Thread] = None
        self._last_out_t: float = time.perf_counter()  # Last permanent output time (idle detection)

    # ── event handling ───────────────────────────────────────────────────

    def handle_event(self, event: SessionEvent) -> None:
        """Process a session event and update the display."""
        self._start_ticker()
        with self._lock:
            self._handle_event_locked(event)

    def _handle_event_locked(self, event: SessionEvent) -> None:
        if event.type == "text":
            if event.metadata.get("partial"):
                # Partial tokens are ignored — only complete TextBlock is printed once.
                # (printing both partial + complete would double the same paragraph)
                return
            self._handle_text(event)

        elif event.type == "tool_call":
            self._handle_tool_call(event)

        elif event.type == "tool_result":
            self._handle_tool_result(event)

        elif event.type == "verdict":
            self._handle_verdict(event)

        elif event.type == "error":
            self._clear_live()
            segs = _wrap_cells(event.content.replace("\n", " | "), _term_cols() - 5)
            self._print(f"  {_RED}✗ {segs[0]}{_RESET}")
            for seg in segs[1:]:
                self._print(f"    {_RED}{seg}{_RESET}")
            self._render_live()

        elif event.type == "status":
            self._clear_live()
            self._print(f"  {_MUTED}[{_truncate(event.content, _term_cols() - 5)}]{_RESET}")
            self._render_live()

        # Write to output file if configured (partial content already returned above)
        if self._output_file:
            self._line_buffer.append(
                f"[{time.strftime('%H:%M:%S')}] [{event.type}] {event.content}"
            )

    def _handle_text(self, event: SessionEvent) -> None:
        """Claude Code Agent's words — mauve gutter bar per line."""
        self._clear_live()
        cleaned = _STRIP_XML_RE.sub("", event.content)
        self._text_seen.append(cleaned)
        # Partial tokens are already filtered in _handle_event_locked; by the time
        # we get here, it's always a complete TextBlock — headers/tables/bold render as markdown.
        self._print_gutter_block(cleaned, markdown=True)
        self._render_live()

    def _print_gutter_block(self, text: str, markdown: bool = False) -> None:
        """Print text with the mauve ▌ gutter, wrapping to terminal width.

        markdown=True renders the final answer with rich markdown (headings/bold/inline
        code/code blocks/lists/horizontal rules). Streaming intermediate text uses partial
        tokens, so it stays as plain text (markdown=False) to avoid broken code fences etc.

        Terminal hard-wrap would drop the gutter on continuation rows
        (left edge breaks visually) — so wrap each logical line ourselves
        and re-apply the gutter prefix on every physical row.
        """
        body_cells = _term_cols() - 5  # "  ▌ " prefix + 1 margin
        if markdown:
            md_lines = _markdown_lines(text, body_cells)
            if md_lines is not None:
                for line in md_lines:
                    self._print(f"  {_BOLD}{_MAUVE}▌{_RESET} {line}{_RESET}")
                return
        for line in text.splitlines() or [""]:
            for seg in _wrap_cells(line, body_cells):
                self._print(f"  {_BOLD}{_MAUVE}▌{_RESET} {_TEXT}{seg}{_RESET}")

    def _handle_tool_call(self, event: SessionEvent) -> None:
        meta = event.metadata
        tool_name = meta.get("tool_name", "?")
        if tool_name in _INTERNAL_TOOLS:
            return
        disp_name = _display_name(tool_name)  # mcp__asr__read_file → read_file
        tool_id = meta.get("tool_id") or f"_anon_{self._call_seq + 1}"

        if meta.get("event") == "start":
            self._call_seq += 1
            self._pending[tool_id] = {
                "seq": self._call_seq,
                "name": disp_name,
                "hint": "",
                "t0": time.perf_counter(),
            }
            self._render_live()

        elif meta.get("delta_type") == "input_json":
            pass  # partial JSON input — too verbose

        else:
            # complete: args confirmed — fill pending hint and update live line
            info = self._pending.get(tool_id)
            if info is None:
                # Arrived without 'start' event (include_partial=False path)
                self._call_seq += 1
                info = {
                    "seq": self._call_seq,
                    "name": disp_name,
                    "hint": "",
                    "t0": time.perf_counter(),
                }
                self._pending[tool_id] = info
            info["hint"] = _format_tool_input(tool_name, meta.get("input", {}))
            self._render_live()

    def _handle_tool_result(self, event: SessionEvent) -> None:
        meta = event.metadata
        tool_id = meta.get("tool_use_id", "")
        info = self._pending.pop(tool_id, None)
        name = (info or {}).get("name") or _display_name(meta.get("tool_name", "?"))
        hint = (info or {}).get("hint", "")
        elapsed = time.perf_counter() - info["t0"] if info else 0.0
        is_error = meta.get("is_error", False)

        mark = "✗" if is_error else "✓"
        name_color = _RED if is_error else _TEXT
        prefix = f"  [{elapsed:5.1f}s] {mark} "
        # Truncate to fit one line — terminal hard-wrap breaks left alignment.
        # Name in body color, hint (path:line etc args) in muted for readability.
        avail = _term_cols() - 1 - _disp_width(prefix)
        disp_name = _truncate(name, avail)
        hint_avail = avail - _disp_width(disp_name) - 2
        hint_sfx = f"  {_truncate(hint, hint_avail)}" if hint and hint_avail >= 8 else ""

        # Overwrite the live ○ line with a ✓/✗ confirmed line, and if tools are
        # still running, redraw the live line — no orphaned ○ even with parallel calls
        self._clear_live()
        self._print(f"{name_color}{prefix}{disp_name}{_RESET}{_MUTED}{hint_sfx}{_RESET}")
        if is_error:
            err_snip = event.content.replace("\n", " | ")[:300]
            indent = " " * _disp_width(prefix)
            for seg in _wrap_cells(err_snip, _term_cols() - 1 - len(indent)):
                self._print(f"{indent}{_RED}{seg}{_RESET}")
        self._render_live()

    def _handle_verdict(self, event: SessionEvent) -> None:
        self._verdict_summary = event.content
        self._clear_live()
        # The verdict block is the only output path for the analysis body (summary/details)
        # — print_summary only prints a one-liner status, so omitting it here would cause
        # Claude's analysis in StructuredOutput.details to vanish from the screen entirely
        # (it IS recorded in the session, visible only to the next-turn LLM). Duplication
        # with content already streamed as body text is handled by _print_verdict's heuristic.
        verdict = event.metadata.get("verdict", {})
        if isinstance(verdict, dict) and verdict:
            self._print_verdict(verdict)
        elif event.content:
            self._print_gutter_block(event.content, markdown=True)

    # ── live line (in-place ○) helpers ───────────────────────────────────

    def _start_ticker(self) -> None:
        """Start a ticker that periodically re-renders the live ○ line with spinner+elapsed.

        Same UX as design chat _ProgressPrinter._tool_ticker_worker — eliminates the
        "stuck" feeling during long-running tools (including MCP Claude Code Agent
        running ASR tools). Does NOT start on non-tty (piped output).
        """
        if not sys.stdout.isatty():
            return
        with self._ticker_launch_lock:
            if self._ticker_thread is not None and self._ticker_thread.is_alive():
                return
            stop = threading.Event()
            self._ticker_stop = stop
            self._ticker_thread = threading.Thread(
                target=self._ticker_worker, args=(stop,), daemon=True,
            )
            self._ticker_thread.start()

    def stop(self) -> None:
        """Stop ticker + clean up live line (idempotent — safe in finally)."""
        with self._ticker_launch_lock:
            if self._ticker_stop is not None:
                self._ticker_stop.set()
                self._ticker_stop = None
            t = self._ticker_thread
            self._ticker_thread = None
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.5)
        with self._lock:
            self._clear_live()

    def _ticker_worker(self, stop: threading.Event) -> None:
        """Every 0.25s: if tools are running, re-render spinner+elapsed; if no tools but
        2+ seconds since last output, show idle spinner (agent thinking)."""
        while not stop.wait(0.25):
            with self._lock:
                if stop.is_set():
                    continue
                self._spin_i += 1
                if self._pending:
                    self._render_live()
                else:
                    idle = time.perf_counter() - self._last_out_t
                    if idle >= 2.0:
                        ch = _SPIN_CHARS[self._spin_i % len(_SPIN_CHARS)]
                        line = f"  {_MUTED}{ch} claude code …  ·  {int(idle)}s{_RESET}"
                        with TERM_WRITE_LOCK:
                            sys.stdout.write(f"\r\x1b[2K{line}")
                            sys.stdout.flush()
                            set_row_pending(True)
                        self._live = True

    def _render_live(self) -> None:
        """Render the single in-place line for ALL currently running tools.

        Runs under 1s → ○ (static); after that, ticker rotates spinner + `· Ns` elapsed
        time (parallel mode uses longest-running tool as reference).

        MUST fit one physical row: if it wraps, \\r\\x1b[2K only clears the
        last wrapped row and every re-render leaves an orphan line behind.
        """
        if not self._pending:
            return
        infos = sorted(self._pending.values(), key=lambda i: i["seq"])
        seqs = ",".join(str(i["seq"]) for i in infos)
        names = " · ".join(i["name"] for i in infos)
        ch, tail = "○", ""
        run_secs = time.perf_counter() - min(i["t0"] for i in infos)
        if run_secs >= 1.0:
            ch = _SPIN_CHARS[self._spin_i % len(_SPIN_CHARS)]
            tail = f"  ·  {int(run_secs)}s"
        prefix = f"  [{seqs}] {ch} "
        avail = _term_cols() - 1 - _disp_width(prefix) - _disp_width(tail)
        names = _truncate(names, avail)
        # hint shown only for single-tool — parallel just lists names to keep line short
        hint = infos[0]["hint"] if len(infos) == 1 else ""
        hint_avail = avail - _disp_width(names) - 2
        hint_sfx = f"  {_truncate(hint, hint_avail)}" if hint and hint_avail >= 8 else ""
        line = f"  {_YELLOW}[{seqs}] {ch} {names}{_RESET}{_MUTED}{hint_sfx}{tail}{_RESET}"
        with TERM_WRITE_LOCK:
            sys.stdout.write(f"\r\x1b[2K{line}")
            sys.stdout.flush()
            set_row_pending(True)
        self._live = True

    def _clear_live(self) -> None:
        """Erase the live ○ line so a permanent line can print in its place."""
        if self._live:
            with TERM_WRITE_LOCK:
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()
                set_row_pending(False)
            self._live = False

    # ── header / summary ─────────────────────────────────────────────────

    def print_header(self, task: str, model: str | None = None) -> None:
        """Print the session header (native banner style)."""
        with self._lock:
            self._print("")
            self._print(f"  {_BOLD}{_MAUVE}▌ Claude Code{_RESET}")
            self._print(f"  {_MUTED}task   {_RESET}{_TEXT}{_truncate(task, 72)}{_RESET}")
            self._print(f"  {_MUTED}model  {_RESET}{_TEXT}{model or '(sdk default)'}{_RESET}")
            self._print(f"  {_BORDER}{'─' * _RULE_WIDTH}{_RESET}")
        # SDK connection~first event gap covered by idle spinner too
        self._start_ticker()

    def print_summary(self, result: Any = None) -> None:
        """Print a compact one-line session summary (no verdict repetition)."""
        self.stop()
        self._clear_live()
        elapsed = time.perf_counter() - self._start_time
        status = ""
        if result is not None:
            status = getattr(getattr(result, "verdict", None), "status", "") or ""
        _display_label = {"success": "completed", "failure": "failed"}.get(status, status)
        icon, color = {
            "success": ("✓", _GREEN),
            "failure": ("✗", _RED),
            "needs_review": ("△", _YELLOW),
        }.get(status, ("·", _MUTED))

        parts = [f"{elapsed:.1f}s"]
        if self._call_seq:
            parts.append(f"{self._call_seq} tool{'' if self._call_seq == 1 else 's'}")
        if result is not None:
            cost = getattr(result, "total_cost_usd", 0.0) or 0.0
            if cost:
                parts.append(f"${cost:.4f}")

        self._print(f"  {_BORDER}{'─' * _RULE_WIDTH}{_RESET}")
        label = _display_label or "done"
        self._print(
            f"  {_BOLD}{color}{icon} {label}{_RESET}"
            f"{_MUTED}  ·  {'  ·  '.join(parts)}{_RESET}"
        )

    # ── verdict rendering ────────────────────────────────────────────────

    def _print_verdict(self, verdict: dict) -> None:
        """Print a compact verdict block matching asicode's result style."""
        status = verdict.get("status", "?")
        summary = verdict.get("summary", "")
        # Defense-in-depth: from_result_message already normalizes confidence,
        # but any future emitter that builds the dict without the dataclass
        # would hit ':.0%' → ValueError on a str. Coerce+clamp here too.
        try:
            confidence = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        suggestions = verdict.get("suggestions") or []

        status_color = {
            "success": _GREEN,
            "failure": _RED,
            "needs_review": _YELLOW,
            "insufficient_info": _MAUVE,
        }.get(status, _RESET)
        icon = {
            "success": "✓", "failure": "✗",
            "needs_review": "△", "insufficient_info": "?",
        }.get(status, "?")

        self._print("")
        head_prefix = f"{icon} "
        head_segs = _wrap_cells(summary, _term_cols() - 3 - _disp_width(head_prefix))
        self._print(f"  {_BOLD}{status_color}{head_prefix}{head_segs[0]}{_RESET}")
        head_indent = " " * (2 + _disp_width(head_prefix))
        for seg in head_segs[1:]:
            self._print(f"{head_indent}{_BOLD}{status_color}{seg}{_RESET}")
        self._print(f"     {_MUTED}confidence {confidence:.0%}{_RESET}")
        for s in suggestions[:3]:
            segs = _wrap_cells(str(s), _term_cols() - 8)  # "     ↳ " + margin
            self._print(f"     {_MUTED}↳ {segs[0]}{_RESET}")
            for seg in segs[1:]:
                self._print(f"       {_MUTED}{seg}{_RESET}")
        # details carries the analysis body (system append instructs Claude to put
        # analysis in verdict.details). Always skipping it makes the body vanish from
        # screen (budget truncation case); always printing it doubles the answer (model
        # disobeyed and streamed the analysis as body too). Heuristic:
        #   - Body short (just progress notes) → details is the only analysis → print
        #   - Body long → details likely restatement → skip; but if details is much
        #     longer (2x+) than body, it has new content → print
        details = str(verdict.get("details") or "").strip()
        # Strip Claude Code SDK internal XML tool call tag leakage
        details = _STRIP_XML_RE.sub("", details)
        # The model sometimes wraps details in <details>/<summary> HTML — even though
        # the system prompt forbids it, multi-block cases slip through. Strip ALL such
        # tags regardless of position (summary title text is preserved). Removing only
        # the outermost pair leaves `</details><details>` boundaries visible in the terminal.
        details = re.sub(r"<summary>(.*?)</summary>", r"\1", details, flags=re.S)
        details = re.sub(r"</?details>", "", details)
        details = re.sub(r"\n{3,}", "\n\n", details).strip()
        body = "\n".join(self._text_seen)
        show_details = (
            details
            and details not in body
            and (len(body) < 600 or len(details) > 2 * len(body))
        )
        if show_details:
            self._print("")
            self._print_gutter_block(details, markdown=True)

    # ── low-level output ─────────────────────────────────────────────────

    def flush_log(self) -> None:
        """Flush buffered log lines to output file."""
        if self._output_file and self._line_buffer:
            try:
                with open(self._output_file, "a", encoding="utf-8") as f:
                    for line in self._line_buffer:
                        f.write(line + "\n")
                self._line_buffer.clear()
            except OSError as ex:
                logger.warning("Failed to write session log: %s", ex)

    def _print(self, text: str) -> None:
        """Print a permanent line to stdout (ends with newline → row not pending)."""
        self._last_out_t = time.perf_counter()
        with TERM_WRITE_LOCK:
            print(text, file=sys.stdout, flush=True)
            set_row_pending(False)


def _display_name(tool_name: str) -> str:
    """Strip ``mcp__<server>__`` prefix from MCP tool names for concise display.

    E.g. ``mcp__asr__read_file`` → ``read_file``. Returns as-is if no prefix.
    Splits only on ``__`` boundaries, so a single underscore in the tool name
    (``read_file``) is preserved.
    """
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return tool_name


# Same _out_console theme (Catppuccin Mocha) as asi.py — colors match exactly
# so markdown rendering looks identical in both places.
_MARKDOWN_THEME_COLORS = {
    "markdown.h1":          "bold #89b4fa",
    "markdown.h1.border":   "#313244",
    "markdown.h2":          "bold #89dceb",
    "markdown.h3":          "bold #94e2d5",
    "markdown.h4":          "bold #cdd6f4",
    "markdown.code":        "#89dceb",
    "markdown.code_block":  "#cdd6f4",
    "markdown.link":        "underline #89b4fa",
    "markdown.link_url":    "#6c7086",
    "markdown.item.bullet": "#fab387",
    "markdown.item.number": "#fab387",
    "markdown.hr":          "#313244",
    "markdown.block_quote": "italic #6c7086",
}


def _markdown_lines(text: str, width: int) -> Optional[list[str]]:
    """Render markdown as ANSI line list (caller attaches the gutter).

    Uses ``rich.markdown.Markdown`` (same as asi) — headings/bold/inline
    code/code blocks/lists/horizontal rules are rendered. Console width is fixed
    by ``width`` so all physical rows fit within width cells, preventing left
    gutter misalignment (rich handles CJK width too). Returns ``None`` if rich
    is unavailable or rendering fails — caller falls back to plain text.
    """
    if not text.strip():
        return None
    try:
        import io as _io

        from rich.console import Console
        from rich.markdown import Markdown
        from rich.theme import Theme
    except Exception:
        return None
    try:
        buf = _io.StringIO()
        Console(
            file=buf, width=max(width, 8), force_terminal=True,
            color_system="truecolor", theme=Theme(_MARKDOWN_THEME_COLORS),
        ).print(Markdown(text))
    except Exception:
        return None
    lines = buf.getvalue().split("\n")
    while lines and lines[-1].strip() == "":  # Strip trailing blank lines left by rich
        lines.pop()
    return lines or None


def _format_tool_input(tool_name: str, inp: dict) -> str:
    """Format tool input for compact display — matches asicode's native style.

    Returns the FULL argument value with NO fixed cell-budget cap. Width
    adaptation (truncation to fit exactly one terminal row) is the renderer's
    job — see the ``_truncate(hint, hint_avail)`` call in both ``_render_live``
    and ``_handle_tool_result``, where ``hint_avail`` is derived from
    ``_term_cols()``. Pre-truncating here with a fixed budget (the old 72/80
    caps) capped the command *short of* the real terminal width on wide
    terminals: the command showed '…' even when plenty of room remained.
    """
    if not inp:
        return ""
    tool_name = _display_name(tool_name)
    # read_file: append line range to path (e.g. asi.py:120-340).
    # If neither start nor end is set, it's a full read so show path only.
    if tool_name == "read_file" and "path" in inp:
        base = str(inp["path"])
        _s, _e = inp.get("start_line"), inp.get("end_line")
        if _s is not None or _e is not None:
            base += f":{_s if _s is not None else 1}-{_e if _e is not None else 'end'}"
        return base
    # Key args to highlight for common tools
    key_args = {
        "read_file": "path",
        "read_symbol": "name",
        "grep": "pattern",
        "find_symbol": "name",
        "find_references": "name",
        "find_relevant_files": "query",
        "get_file_outline": "path",
        "search_code": "query",
        "list_directory": "path",
    }
    key = key_args.get(tool_name)
    if key and key in inp:
        return str(inp[key])
    # Fallback: first meaningful arg
    for k in ("query", "path", "name", "pattern", "command", "file_path"):
        if k in inp:
            return str(inp[k])
    # Just show first 2 args
    parts = [f"{k}={v}" for k, v in list(inp.items())[:2]]
    return ", ".join(parts)


def _term_cols() -> int:
    """Current terminal width in cells (fallback 100)."""
    try:
        return shutil.get_terminal_size().columns
    except (ValueError, OSError):
        return 100


def _ch_width(ch: str) -> int:
    """Display cell width of a single character (CJK wide = 2)."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _disp_width(s: str) -> int:
    """Display width of a string in terminal cells — len() undercounts CJK."""
    return sum(_ch_width(ch) for ch in s)


def _truncate(s: str, max_cells: int = 60) -> str:
    """Truncate string to a display-cell budget (CJK-aware)."""
    if _disp_width(s) <= max_cells:
        return s
    out, w = [], 0
    for ch in s:
        cw = _ch_width(ch)
        if w + cw > max_cells - 1:
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


def _wrap_cells(s: str, max_cells: int) -> list[str]:
    """Word-wrap a string into segments of at most max_cells display cells.

    Terminal hard-wrap loses the left gutter/indent on continuation rows —
    wrapping here lets every segment get its own prefix. Long unbreakable
    tokens (paths, identifiers) are hard-split.
    """
    max_cells = max(max_cells, 8)
    lines: list[str] = []
    cur, cur_w = "", 0
    for word in s.split(" "):
        ww = _disp_width(word)
        sep = 1 if cur else 0
        if cur_w + sep + ww <= max_cells:
            cur = f"{cur} {word}" if cur else word
            cur_w += sep + ww
            continue
        if cur:
            lines.append(cur)
            cur, cur_w = "", 0
        while _disp_width(word) > max_cells:
            piece, w = [], 0
            for ch in word:
                cw = _ch_width(ch)
                if w + cw > max_cells:
                    break
                piece.append(ch)
                w += cw
            lines.append("".join(piece))
            word = word[len(piece):]
        cur, cur_w = word, _disp_width(word)
    if cur or not lines:
        lines.append(cur)
    return lines
