"""REPL implementation — extracted from asi.py (P6-2).

Owns the interactive REPL loop (``run_repl`` / ``_run_repl_impl``), the JSON
single-shot paths (``run_once`` / ``run_subagent_worker``), the progress printer
(``_ProgressPrinter``) and the shared REPL module state (auto-continue, ghost
suggestion, prompt session, ollama cache, ...).

asi.py remains the public barrel: it re-exports every symbol defined here, so
``import asi; asi.run_repl`` and all ``from asi import X`` call sites keep working.

Import-cycle contract: this module does ``import asi`` for the CLI scaffolding
helpers (_print, _C, console setup, git helpers, ...). asi.py imports this module
ONLY at its very bottom (after every helper is defined), so ``import asi`` here
always sees a fully initialized module. Never import this module before ``asi``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import select as _select
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.agent.terminal_coordination import (
    TERM_WRITE_LOCK as _TERM_WRITE_LOCK,
)
from external_llm.common.atomic_io import atomic_write_json
from external_llm.image_utils import _check_clipboard_image, _extract_images_from_input
from external_llm.model_catalog import (
    KNOWN_MODELS as _KNOWN_MODELS,
)

# ─── REPL shared state (moved from asi.py; re-exported by the asi.py barrel) ──────
# prompt_toolkit session reused across prompts for persistent history
_prompt_session: Any = None  # runtime type bound lazily by asi._load_prompt_toolkit()
# Draw an underline (separator) below the input box only for the main input prompt.
# Toggled per _collect_input call; referenced by ConditionalContainer/menu filters in the layout.
_input_underline: bool = False
_prompt_history_path: str = ""  # run_repl sets to <repo>/.asicode/cli_history
_ctrlc_armed: bool = False  # True after 1st Ctrl+C on empty prompt — reset by typing/sending etc.
# ── Ghost suggestion for next action (filled by background LLM after turn end) ──
_next_prompt_suggestion: str = ""  # Suggestion text shown dimmed on empty prompt
_next_suggestion_gen: int = 0      # Incremented each new turn — invalidates stale suggestions
# ── Auto-continue (/auto): countdown-submit the ghost suggestion (self-improve loop) ──
_auto_continue_state: dict = {"on": False, "cap": 5, "depth": 0}
_auto_submit_gen: int = 0           # Incremented to cancel a pending countdown (typing/Esc/new turn)
_auto_countdown_active: bool = False  # True while a countdown is pending (Enter = run now)
_last_input_was_auto: bool = False  # Set by the countdown submit; read+cleared by run_repl
_REPO_ROOT: str = ""  # Set by run_repl/run_once — shortens absolute paths to relative for tool hints

# Current provider/model for /think /model autocompletion (set by run_repl)
_completer_provider: str = ""
_completer_model: str = ""

# Current sub-agent slots for /model dev_N autocompletion (synced by run_repl).
# run_repl sets _completer_dev_models to point to the same dict as _dev_models, so in-place
# mutations (.pop / []=) are automatically reflected.
_completer_dev_models: dict = {}


def _set_completer_context(provider: str, model: str) -> None:
    """Single writer for the completer's provider/model view.

    Exists as a function rather than a bare assignment because the assignment
    has to happen from a NESTED scope (``_dispatch_command`` lives inside
    ``_run_repl_impl``), and a plain ``x = ...`` there silently binds a local:
    the module global keeps its startup value and the completer never learns
    about a ``/model`` switch. That is exactly what happened — all six
    assignments in ``_dispatch_command`` were no-ops, so ``/think`` kept
    offering the STARTING model's reasoning values (deepseek's ``max`` missing,
    openai's ``none``/``medium`` wrongly offered) for the rest of the session.

    Routing every write through here means the ``global`` declaration exists in
    exactly one place and cannot be forgotten at a call site. The write-only
    shape is invisible to ruff — F823 (gated at zero by
    scripts/check_f823_none.py) only fires when the name is also READ before
    assignment; scripts/check_missing_global.py covers this other half.
    """
    global _completer_provider, _completer_model
    _completer_provider = provider or ""
    _completer_model = model or ""


# Ollama model list cache (optimizes 3 calls within run_repl). None = not yet
# fetched — a pre-populated [] would pass the ``is not None`` TTL guard and
# make the FIRST call (within 10s of process start) return an empty list.
_ollama_cache: Optional[list[str]] = None
_ollama_cache_ts: float = 0.0

# ESC watcher pause event — prevents _run_esc_watcher from intercepting input
# while the checkpoint callback reads stdin.
_esc_watcher_pause = threading.Event()
# The printer currently showing a spinner — exposed at module level so _cli_checkpoint_cb can
# stop the spinner before outputting a question (stream callback and checkpoint cb are called
# from the same engine thread, so no lock needed).
_active_spinner_printer: Optional["_ProgressPrinter"] = None

class _ProgressPrinter:
    """Simple handler that prints stream_callback events to the terminal."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._total_prompt: int = 0
        self._total_completion: int = 0
        self._call_seq: int = 0  # Tool call sequence number (assigned 1,2,3… in completion order)
        # Track concurrently running (ThreadPoolExecutor) tools by call_id.
        # design_tool_call events may lack call_id (provider-specific tool-call id
        # missing), falling back to anon-key. {{call_id: {tool, hint, t0}}}, insertion order = start order.
        self._inflight: "dict[str, dict]" = {}
        self._inflight_ctr: int = 0  # Monotonically increasing counter to make anon keys unique when call_id is missing
        self._live_drawn: bool = False  # Whether a pending live line (no trailing newline) is at the bottom of screen
        self._pending_llm_tokens: int = 0  # Cache miss tokens from the last LLM call (billing accounting; ↑ display sums with hits via total_input_tokens())
        self._pending_llm_cache_read: int = 0  # Cache-read (hit) tokens from last LLM call (↑ sum + hit% numerator)
        self._pending_llm_provider: str = ""   # Provider of last LLM call (used for ↑/hit% denominator branching)
        self._pending_llm_cache_creation: int = 0  # Cache-creation (write) tokens from last LLM call (separate accounting only >0; ↑ sum + hit% numerator)
        # Long-running tool ticker: periodically re-render the ○ line with spinner+elapsed time (prevents perceived freeze)
        self._ticker_stop: Optional[threading.Event] = None
        self._ticker_thread: Optional[threading.Thread] = None
        # thinking ticker: re-renders spinner as "<spinner> thinking · Ns" every 0.1s during LLM calls.
        self._think_tick_stop: Optional[threading.Event] = None
        self._think_tick_thread: Optional[threading.Thread] = None
        self._think_tick_t0: float = 0.0
        self._preview_active: bool = False  # tool_call_preview emitted but tool_call hasn't arrived yet
        self._preview_suffix: str = ""
        self._thinking_buffer: str = ""  # Accumulated reasoning text
        self._thinking_displayed: bool = False  # Whether the current chunk has been displayed
        # Block output from workers that moved to background after ESC (ignore all events after mute())
        self._muted: bool = False
        # ── Spinner (displayed while waiting for LLM response) ──
        self._spinner_running: bool = False
        self._spinner_thread: Optional[threading.Thread] = None
        self._spinner_live: Optional[Any] = None   # Rich Live instance
        self._spinner_obj: Optional[Any] = None    # Rich Spinner instance (preserves start_time)
        self._spinner_msg: str = ""
        self._spinner_chars = "◴◷◶◵"



    # Strip rich markup for log file.
    @staticmethod
    def _plain(msg: str) -> str:
        # Strip [tag], [/tag], [tag word], [hex_color] rich markup
        # e.g. [bold #89b4fa]...[/bold], [#89b4fa]...[/#89b4fa], [dim]...[/dim]
        import re
        return re.sub(r'\[/?[^\[\]\n]*\]', '', msg)

    def _log(self, msg: str) -> None:
        logging.getLogger(__name__).info("[cli] %s", self._plain(msg))

    # ── Spinner control ──

    @staticmethod
    def _spinner_safe(msg: str, max_cols: int) -> str:
        """Collapse newlines and truncate to max_cols visual columns (CJK = 2 cols each).

        When truncating, frees up space from the end — including the ellipsis
        "…" (1 cell) — so the result never exceeds max_cols. This keeps a
        1-cell overflow from pushing the right border onto the next line and
        breaking renders where the width must be exact, like box-border alignment.
        """
        import unicodedata as _ud
        if max_cols <= 0:
            return ""
        flat = " ".join(msg.split()) if msg else ""
        cols = 0
        out: list[str] = []
        for ch in flat:
            w = 2 if _ud.east_asian_width(ch) in ("W", "F") else 1
            if cols + w > max_cols:
                # Drop trailing characters as needed to make room for "…" (1 cell).
                while out and cols + 1 > max_cols:
                    _last = out.pop()
                    cols -= 2 if _ud.east_asian_width(_last) in ("W", "F") else 1
                out.append("…")
                break
            out.append(ch)
            cols += w
        return "".join(out)

    def _start_tool_ticker(self) -> None:
        """Start the ticker that periodically re-renders the currently running tool's ○ line with a spinner + elapsed time."""
        if not sys.stdout.isatty():
            return
        if self._ticker_thread is not None and self._ticker_thread.is_alive():
            return  # Already running — just swap _running_* state
        stop = threading.Event()
        self._ticker_stop = stop
        self._ticker_thread = threading.Thread(
            target=self._tool_ticker_worker, args=(stop,), daemon=True,
        )
        self._ticker_thread.start()

    def _stop_tool_ticker(self) -> None:
        if self._ticker_stop is not None:
            self._ticker_stop.set()
            self._ticker_stop = None
        self._ticker_thread = None

    def _render_live_line(self, marker: str) -> str:
        """Build a single live-line string representing the currently in-flight tools.

        - While in progress the completion seq is unknown, so the number slot is
          **left empty** (the old "[·]" placeholder is no longer shown). The
          empty width matches the icon column of the completion line
          "  [N]…✓ tool" exactly, so the number/✓ fills that same spot naturally
          on completion.
        - Shows the tool that started first (= waited longest) as the
          representative; if several are in flight at once, `(+N)` notes the
          rest. Elapsed time is based on the representative tool's start time.
        """
        if not self._inflight:
            return ""
        import shutil as _sh_tk
        _tw = _sh_tk.get_terminal_size((80, 24)).columns
        first = next(iter(self._inflight.values()))
        _tool = first["tool"]
        _hint = first["hint"]
        run_secs = time.perf_counter() - first["t0"]
        n = len(self._inflight)
        _extra = f"  (+{n - 1})" if n > 1 else ""
        _tail = f"  ·  {int(run_secs)}s" if run_secs >= 1.0 else ""
        # While in progress, seq is unknown → leave number slot empty ([·] not shown). Width matches
        # the icon column of completion line "  [N]…✓" (= 2 + (_SEQ_W+2)). Without ANSI,
        # _fit_row only cuts the right hint, which is safe (width calculation stays plain).
        _lead = " " * (2 + (_SEQ_W + 2))
        _base = f"{_lead}{marker} {_tool}"
        _base_plain = _base
        # Prevent elapsed time/count from being clipped — truncate hint by display cell count (same as ✓ line)
        _avail = _tw - len(_base_plain) - len(_extra) - len(_tail) - 3
        _suffix = f"  {self._spinner_safe(_hint, max(0, _avail))}" if _hint and _avail > 4 else ""
        return f"{_base}{_suffix}{_extra}{_tail}"

    def _pop_inflight(self, call_id: Optional[str]) -> Optional[dict]:
        """Match a completion/error event to an in-flight entry and pop it.

        - Exact call_id match takes priority (the normal path, when the
          provider supplies a tool-call id).
        - If call_id is None (a provider with no id), pop the oldest anon-key
          entry FIFO — this prevents two completions from collapsing into one
          slot (a lost completion) or a live line lingering forever, in
          concurrent runs where exact matching is impossible without an id.
        - If call_id is present but doesn't match anything (e.g. an internal
          tool whose "running" event was never seen), return None instead of
          stealing another tool's slot — the caller still stamps ✓/✗ with a
          fresh seq regardless.
        """
        if call_id is not None and call_id in self._inflight:
            return self._inflight.pop(call_id)
        if call_id is None:
            for k in self._inflight:
                if k.startswith("anon-"):
                    return self._inflight.pop(k)
            if len(self._inflight) == 1:
                return self._inflight.pop(next(iter(self._inflight)))
        return None

    def _refresh_live_line(self) -> None:
        """Call this *after* all finalized ✓/✗ lines (+preview) have been printed.

        If in-flight tools remain, redraws the live line as the last line on
        screen; if none remain, lifts the terminal-log suppression and drops
        the live state. The entry point that maintains the invariant 'the live
        line is always last, and never has a trailing newline'."""
        with _TERM_WRITE_LOCK:
            if self._inflight:
                sys.stdout.write(f"\r\x1b[2K{self._shimmer_row(self._fit_row(self._render_live_line('○')), time.perf_counter())}")
                sys.stdout.flush()
                _tool_running_filter.active = True
                _tool_running_filter.row_pending = True
                self._live_drawn = True
                # _stop_spinner() at design_tool_call entry killed the ticker, so
                # revive it if there are still running tools — otherwise the live line
                # freezes with a stale elapsed time until the next event (async slow-tool scenario).
                self._start_tool_ticker()
            else:
                _tool_running_filter.active = False
                _tool_running_filter.row_pending = False
                self._live_drawn = False

    def _tool_ticker_worker(self, stop: threading.Event) -> None:
        """Re-render the live line every 0.25s with a ◴◷◶◵ spinner and elapsed time.

        - Doesn't draw if the representative (oldest) tool has run under 1s —
          keeps short tools clean as a plain ○→✓.
        - Suppressed while _esc_watcher_pause is set (an ask_user checkpoint is
          reading stdin).
        - Only writes inside self._lock, so there's no race with the ✓
          overwrite done by the complete/error handlers (once the live line
          goes down, _live_drawn=False stops further redraws).
        """
        i = 0
        while not stop.wait(0.25):
            i += 1
            with self._lock:
                if stop.is_set() or not self._inflight or not self._live_drawn:
                    continue
                if _esc_watcher_pause.is_set():
                    continue
                first = next(iter(self._inflight.values()))
                if time.perf_counter() - first["t0"] < 1.0:
                    continue
                ch = self._spinner_chars[i % len(self._spinner_chars)]
                # _TERM_WRITE_LOCK: serialize with log handler emit (_RowSafeEmitMixin)
                # — without it, re-render would interleave during WARNING output, mangling lines.
                with _TERM_WRITE_LOCK:
                    sys.stdout.write(f"\r\x1b[2K{self._shimmer_row(self._fit_row(self._render_live_line(ch)), time.perf_counter())}")
                    sys.stdout.flush()
                    _tool_running_filter.row_pending = True

    def _start_thinking_ticker(self) -> None:
        """Start of an LLM call — updates the spinner to "<spinner> thinking · Ns"
        every 0.1s to show it's thinking, along with the live elapsed time.
        When the call finishes, _stop_spinner() (→ _stop_thinking_ticker) stops
        it, and the design_thinking handler finalizes it into a
        'thought for Ns' panel."""
        self._stop_thinking_ticker()
        # Spinner may have been stopped when the last tool line was printed — re-spawn if needed.
        if not self._spinner_running:
            self._start_spinner("thinking")
        else:
            self._update_spinner("thinking")
        self._think_tick_t0 = time.perf_counter()
        self._think_tick_stop = threading.Event()
        self._think_tick_thread = threading.Thread(
            target=self._thinking_ticker_worker,
            args=(self._think_tick_stop,), daemon=True,
        )
        self._think_tick_thread.start()

    def _stop_thinking_ticker(self) -> None:
        if self._think_tick_stop is not None:
            self._think_tick_stop.set()
            self._think_tick_stop = None
        self._think_tick_thread = None

    def _thinking_ticker_worker(self, stop: threading.Event) -> None:
        while not stop.wait(0.1):
            if stop.is_set() or not self._spinner_running:
                return
            secs = time.perf_counter() - self._think_tick_t0
            # Hide timer under 1s — short calls show just clean "thinking".
            # Remove leading "…": _ShimmerSpinner.render puts the rotating glyph at the front,
            # so that glyph replaces the old "…" and serves as the spinner.
            _t = f"  ·  {secs:.1f}s" if secs >= 1.0 else ""
            self._update_spinner(f"thinking{_t}")

    def _consume_llm_tokens_str(self) -> str:
        """Display the last LLM call's 'total input context size' + cache hit% (consumed once, dim).

        Shows the total input size accounting for provider-specific billing
        (_CACHE_TOKENS_SEPARATE) — this is the 'context window currently
        occupied' regardless of cache hits, so it monotonically increases
        across turns. The billed amount (cumulative sum) is shown separately
        on the turn-end summary line.

        Note: ``_pending_llm_tokens`` is the 'cache-miss portion' for billing
        accounting. Displaying this value directly as ↑ would create the
        illusion that ↑ shrinks as the cache hit rate rises (this is exactly
        what 'Z.AI cache: N cached (5116% of prompt)' in the logs reflects).
        So the display uses ``total_input_tokens()`` to sum miss+hit
        (zai/anthropic), or miss only (openai/deepseek subset).

        If one call issues multiple tools, this is attached only to the first
        ✓ line — repeating the same call's tokens per tool would misread as a
        sum. Next to the context size, cache hit rate is shown as 'N% cached'
        (e.g. ↑48k · 84% cached).
        """
        n = self._pending_llm_tokens        # Cache miss tokens (billing accounting)
        crt = self._pending_llm_cache_read  # Cache hit portion
        cc = self._pending_llm_cache_creation  # cache-creation(write) portion (separate accounting only >0)
        prov = self._pending_llm_provider
        if not n and not crt and not cc:
            return ""
        self._pending_llm_tokens = 0
        self._pending_llm_cache_read = 0
        self._pending_llm_cache_creation = 0
        self._pending_llm_provider = ""
        try:
            from external_llm.agent._shared_utils import cache_hit_pct, total_input_tokens
            _total = total_input_tokens(prov, n, crt, cc)
            if not _total:
                return ""
            if crt and prov:
                _pct = cache_hit_pct(prov, n, crt, cc)
                if _pct > 0:
                    return f" \x1b[2m· ↑{_abbrev_tokens(_total)} · {_pct:.0f}% cached\x1b[22m"
            return f" \x1b[2m· ↑{_abbrev_tokens(_total)}\x1b[22m"
        except Exception:
            return f" \x1b[2m· ↑{_abbrev_tokens(n + crt + cc)}\x1b[22m"

    @staticmethod
    def _fit_row(text: str) -> str:
        """Flatten embedded newlines/tabs and truncate to the terminal width so the
        line never spans more than one terminal row. This keeps the in-place
        \\r\\x1b[2K overwrite valid — that escape only clears a single row, so a
        preview that wraps onto a 2nd row would leave the 1st row stranded.
        Unlike _spinner_safe this preserves intentional spacing (indent, gaps)."""
        import shutil as _sh_fr
        import unicodedata as _ud
        # leave 1 col of slack to avoid terminals auto-wrapping at the last column
        max_cols = max(20, _sh_fr.get_terminal_size((80, 24)).columns - 1)
        flat = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        cols = 0
        out: list[str] = []
        for ch in flat:
            w = 2 if _ud.east_asian_width(ch) in ("W", "F") else 1
            if cols + w > max_cols:
                out.append("…")
                break
            out.append(ch)
            cols += w
        return "".join(out)

    # ── Common event line rendering ─────────────────────────────────────────────
    # All observation lines follow a single unified form: "dim [elapsed] · color icon · body"
    # Timestamps are always dimmed to give the body visual hierarchy precedence.

    @staticmethod
    def _hex_ansi(hex_color: str) -> tuple[str, str]:
        """#rrggbb → (truecolor SGR on, reset-fg off). For coloring raw stdout lines."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"\x1b[38;2;{r};{g};{b}m", "\x1b[39m"

    @classmethod
    def _style_row(cls, s: str, icon_hex: Optional[str] = None) -> str:
        """Colorize a plain line already width-fitted by _fit_row (width unchanged).

        Dims the leading "[…]" token, then colors the first non-space
        character after it (the icon) with icon_hex. Since several _seq_pad
        padding spaces may follow "]", the character after skipping all
        spaces is treated as the icon. ANSI is injected after truncation, so
        it doesn't break the width calculation. (The leading token/icon is
        never a position that gets truncated.)
        """
        i = s.find("[")
        j = s.find("]", i) if i >= 0 else -1
        if i < 0 or j < 0:
            return s
        out = s[:i] + "\x1b[2m" + s[i:j + 1] + "\x1b[22m"
        rest = s[j + 1:]
        if icon_hex:
            k = 0
            while k < len(rest) and rest[k] == " ":
                k += 1  # Skip padding space after "]"
            if k < len(rest) and rest[k] != "…":
                on, off = cls._hex_ansi(icon_hex)
                rest = rest[:k] + on + rest[k] + off + rest[k + 1:]
        return out + rest

    @classmethod
    def _shimmer_row(cls, s: str, elapsed: float) -> str:
        """Apply a Claude Code-style beam shimmer to the running live tool line ("◴ bash …").

        Leaves the leading indent + rotating glyph + one space untouched, and
        colors only the body after it (tool label + command/elapsed) with a
        per-char interpolated color. As with _style_row, ANSI is injected
        *after* _fit_row truncation, so it doesn't break the width calculation
        (symmetric with the completed ✓ line's _style_row). _shimmer_beam /
        _shimmer_style_for are shared with the thinking spinner."""
        if not s:
            return s
        k = 0
        while k < len(s) and s[k] == " ":
            k += 1                       # Leading indent
        if k >= len(s):
            return s
        k += 1                           # Skip rotating glyph 1 char (◴◷◶◵/○)
        if k < len(s) and s[k] == " ":
            k += 1                       # Space after glyph
        prefix, body = s[:k], s[k:]
        n = len(body)
        if n < 4 or not body.strip():
            return s
        center, beam_w = _shimmer_beam(n, elapsed)
        out = [prefix]
        for i, ch in enumerate(body):
            if ch == " ":
                out.append(" ")
                continue
            on, off = cls._hex_ansi(_shimmer_style_for(i, center, beam_w))
            out.append(on + ch + off)
        return "".join(out)

    def _event(self, elapsed: float, icon: str, msg: str,
               color: str = "text", icon_color: Optional[str] = None) -> None:
        """Print one observation line: dim [elapsed] · colored icon · body.

        color/icon_color are either _C palette keys or a style/hex string to use as-is.
        """
        _ensure_out_console_imported()
        _style = _C.get(color, color)
        _istyle = _C.get(icon_color or color, color)
        if _RICH and asi._out_console:
            from rich.text import Text
            _f = asi._out_console.file
            if hasattr(_f, "reset_bol"):
                _f.reset_bol()
            t = Text("  ")
            t.append(f"[{elapsed:5.1f}s]", style=_C["muted"])
            t.append(" ")
            if icon:
                t.append(icon, style=_istyle)
                t.append(" ")
            t.append(msg, style=_style)
            asi._out_console.print(t)
        else:
            _ic = f"{icon} " if icon else ""
            print(f"  [{elapsed:5.1f}s] {_ic}{msg}")

    @staticmethod
    def _make_spinner(text: str, style: str):
        """Build a Claude Code-style shimmer spinner.

        Adds a shimmer lighting effect — a beam scanning left↔right over the
        body text — to the same family of rotating glyphs as the tool ticker
        (◴◷◶◵ circle quadrants); see _ShimmerSpinner.render. The rotating
        glyph uses Rich Spinner's frames/interval, and the body is colored
        per-char with an interpolated beam color every frame.
        """
        return _ShimmerSpinner(text, style, frames=["◴", "◷", "◶", "◵"], interval=130)

    # ── update_plan specific rendering: full checklist + changes from previous plan ──

    _PLAN_STATUS_UI: ClassVar[dict] = {
        "done":        ("✓", "green"),
        "in_progress": ("▸", "sky"),
        "pending":     ("○", "muted"),
        "skipped":     ("⊘", "muted"),
        "blocked":     ("✖", "red"),
    }

    def _emit_plan_line(self, parts: list) -> None:
        """Emit (text, palette_key|style_str|None) segments on one line.

        Palette keys ("green" etc.) are looked up in _C for a color; other
        strings ("bold #89dceb" etc.) are used as-is as rich styles. Without
        rich, use plain.
        """
        if _RICH and asi._out_console:
            from rich.text import Text
            _f = asi._out_console.file
            if hasattr(_f, "reset_bol"):
                _f.reset_bol()
            _t = Text("  ")
            for _txt, _ckey in parts:
                _t.append(_txt, style=(_C.get(_ckey) or _ckey) if _ckey else None)
            asi._out_console.print(_t)
        else:
            print("  " + "".join(_txt for _txt, _ in parts))

    def _render_plan_update(self, plan: dict, prev_statuses: Optional[dict]) -> None:
        """Render the full plan as a bordered box below the update_plan ✓ line.

        Layout (a frame visually separated from the conversation flow):
          ╭─ Plan ──────────────── done/total ─╮
          │ Goal  <goal>                        │   (only when goal is set + divider)
          ├─────────────────────────────────────┤
          │ ✓ <item>                            │
          │ ▸ <item>                            │
          ╰─ ▰▰▱▱… pct% · open·skipped·… ──────╯

        - Per-status icon/color: ✓ done · ▸ in_progress · ○ pending · ⊘ skipped · ✖ blocked
        - Changes are expressed via color instead of text annotations: done
          items always render with a green body, but only items that just
          changed (or were just added) in this update get a brighter bold tone
          to distinguish "what just changed" (skipped=muted).
          Removed items are tracked cumulatively across the session but not
          listed individually — they surface only as the '· N removed' count
          in the bottom border (this signals, via count alone, that the
          denominator quietly shrank — inflating the completion percentage —
          without adding list noise).
        - At 16+ items, unchanged done items collapse into a single line to
          cut noise (items that just changed are still shown individually).
        - Bottom border: a segmented progress bar (done=green▰·skipped=peach▰·
          blocked=red✖·open=▱) + pct% + a breakdown count of
          open·skipped·blocked·removed. The bar fills fully green only when
          everything is done.
        - Right-border alignment pads each row to content_w based on display
          width (_cjk_width, CJK=2 cells). Body/counts are truncated by
          display width if they exceed the box width.
        """
        import shutil as _sh_pl
        items = [it for it in plan.get("items", []) if isinstance(it, dict)]
        if not items:
            return
        prev = prev_statuses if isinstance(prev_statuses, dict) else None
        created = prev is None
        # Track removed items accumulatively across the session — prevents a deferred task that disappeared
        # from silently evaporating with its denominator in the next update, making it look "100% complete".
        # Reset accumulated removals on a fresh plan (no prev).
        removed_acc = getattr(self, "_plan_removed_acc", None)
        if removed_acc is None:
            removed_acc = {}
            self._plan_removed_acc = removed_acc
        if created:
            removed_acc.clear()

        # ── Box dimensions ──
        # Width reference must be asi._out_console.width, which rich uses for actual line wrapping.
        # shutil terminal width and rich console width may differ (pipe/margin handling),
        # so measuring with shutil would make zero-pad rows overflow the console width, leaking │ to the next line.
        # Screen width = 2 (left indent) + box_w. Leave 2 extra cells to absorb micro-discrepancies between
        # _cjk_width and rich cell counting (ambiguous-width glyphs, etc.).
        _term_w = getattr(asi._out_console, "width", None) if (_RICH and asi._out_console) else None
        if not _term_w:
            _term_w = _sh_pl.get_terminal_size((80, 24)).columns - _CONSOLE_MARGIN * 2
        box_w = max(30, _term_w - 4)  # 2 indent + 2 safety margin
        content_w = box_w - 4  # "│ " (2) + content + " │" (2)

        def _box_row(inner: list) -> None:
            """Wrap inner=[(text,color)...] with left/right borders, padded to content_w.

            Width is measured via display width (_cjk_width), so the right │ stays aligned even with CJK mixed in.
            """
            w = sum(_cjk_width(_t) for _t, _ in inner)
            pad = max(0, content_w - w)
            self._emit_plan_line(
                [("│ ", "muted"), *inner, (" " * pad, None), (" │", "muted")]
            )

        total = len(items)
        done = sum(1 for it in items if it.get("status") == "done")
        n_skipped = sum(1 for it in items if it.get("status") == "skipped")
        n_blocked = sum(1 for it in items if it.get("status") == "blocked")
        n_open = total - done - n_skipped - n_blocked

        # ── Top border: ╭─ Plan ──…── done/total ─╮ ──
        _hl = "╭─ Plan "
        _hr = f" {done}/{total} ─╮"
        _hdash = max(0, box_w - _cjk_width(_hl) - _cjk_width(_hr))
        self._emit_plan_line([(_hl + "─" * _hdash + _hr, "muted")])

        # ── Goal row + separator (only when goal exists) ──
        goal = str(plan.get("goal", "")).strip()
        if goal:
            _glabel = "Goal  "
            _box_row([
                (_glabel, "muted"),
                (self._spinner_safe(goal, max(4, content_w - _cjk_width(_glabel))), "mauve"),
            ])
            self._emit_plan_line([("├" + "─" * (box_w - 2) + "┤", "muted")])

        collapse = len(items) > 15
        collapsed_done = 0
        seen_titles = set()
        # Highlight color for items changed in this update (brighten the status color)
        _changed_body_c = {
            "done": f"bold {_C['green']}",
            "in_progress": f"bold {_C['sky']}",
            "pending": "yellow",
            "skipped": "peach",
            "blocked": "red",
        }
        for it in items:
            title = str(it.get("title", ""))
            status = str(it.get("status", "pending"))
            seen_titles.add(title)
            old = prev.get(title) if prev else None
            is_new = (not created) and old is None
            changed = (old is not None) and old != status
            if collapse and status == "done" and not changed and not is_new:
                collapsed_done += 1
                continue
            icon, icon_c = self._PLAN_STATUS_UI.get(status, ("○", "muted"))
            note = str(it.get("note", "")).strip()
            body = title + (f" — {note}" if note else "")
            if is_new or changed:
                body_c = _changed_body_c.get(status, "yellow")
            else:
                body_c = (
                    "sky" if status == "in_progress"
                    else "green" if status == "done"
                    else "muted" if status == "skipped" else "text"
                )
            _icon_seg = f"{icon} "
            _bmax = max(4, content_w - _cjk_width(_icon_seg))
            _box_row([(_icon_seg, icon_c),
                      (self._spinner_safe(body, _bmax), body_c)])
        if collapsed_done:
            _box_row([("✓ ", "green"),
                      (f"… {collapsed_done} done (unchanged)", "muted")])
        # Reflect items removed since previous plan in the accumulated set (reappearance clears them).
        # Only expose as a count (n_removed), not individual lines — prevents denominator from silently
        # shrinking and inflating completion percentage. The only hint is the bottom '· N removed'.
        if prev:
            for title in prev:
                if title not in seen_titles:
                    removed_acc[title] = prev.get(title) or "removed"
        for title in seen_titles:
            removed_acc.pop(title, None)

        n_removed = len(removed_acc)

        # ── Bottom border: ╰─ <progress bar> pct% · breakdown count ──…──╯ ──
        # Segment progress bar: done=green▰ · skipped=peach▰ · blocked=red✖ · open=▱.
        # Blocked items are distinguished by both color and a dedicated glyph (✖, same as status icon)
        # for quick visual scanning. Width of 10 cells is proportionally allocated (largest remainder, exact sum of 10),
        # so the bar never fills entirely green when skipped/blocked items exist.
        def _alloc(counts: list, denom: int, width: int = 10) -> list:
            if denom <= 0:
                return [0] * len(counts)
            raw = [c * width / denom for c in counts]
            cells = [int(x) for x in raw]
            for idx in sorted(range(len(counts)),
                              key=lambda i: raw[i] - cells[i],
                              reverse=True)[:width - sum(cells)]:
                cells[idx] += 1
            return cells
        d_cells, s_cells, b_cells, o_cells = _alloc([done, n_skipped, n_blocked, n_open], total)
        bar = [
            ("▰" * d_cells, "green"),
            ("▰" * s_cells, "peach"),
            ("✖" * b_cells, "red"),
            ("▱" * o_cells, "muted"),
        ]
        pct = round(done * 100 / total) if total else 0
        _stat = f"{pct}%"
        if n_open:
            _stat += f" · {n_open} open"
        if n_skipped:
            _stat += f" · {n_skipped} skipped"
        if n_blocked:
            _stat += f" · {n_blocked} blocked"
        if n_removed:
            _stat += f" · {n_removed} removed"
        # Narrow terminal: if count exceeds box_w, the bottom border would pass the top-right corner,
        # appearing "broken". Clip by subtracting the width taken by progress bar, corner, and separator spaces.
        # Separator spaces are not placed in _spinner_safe (which strips leading/trailing spaces via .split())
        # but attached as an outer segment to guarantee one space between progress bar and pct.
        _bar_w = sum(_cjk_width(_t) for _t, _ in bar)
        # "╰─ "(3) + bar + " "(1) + _stat + " "(1) + dashes + "╯"(1) == box_w
        _stat_budget = max(0, box_w - 3 - _bar_w - 3)
        _stat = self._spinner_safe(_stat, _stat_budget)
        foot = [("╰─ ", "muted"), *bar, (f" {_stat} ", "muted")]
        _fw = sum(_cjk_width(_t) for _t, _ in foot)
        _ffill = max(0, box_w - _fw - 1)  # 1 = ╯
        foot += [("─" * _ffill, "muted"), ("╯", "muted")]
        self._emit_plan_line(foot)
        _log_tail = f"{done}/{total} done"
        if n_open:
            _log_tail += f" · {n_open} open"
        if n_skipped:
            _log_tail += f" · {n_skipped} skipped"
        if n_blocked:
            _log_tail += f" · {n_blocked} blocked"
        if n_removed:
            _log_tail += f" · {n_removed} removed"
        self._log(f"plan: {_log_tail}")

    def _spinner_indent(self) -> str:
        """Leading whitespace to align the spinner glyph with the ✓ column of the completed line "  [N]…✓ tool".

        The icon column is fixed at "  " + "[…]"(_SEQ_W+2) = 2 + (_SEQ_W+2),
        since the icon is aligned purely by _seq_pad with no literal space
        after "]". _ShimmerSpinner.render moves this leading whitespace in
        front of the rotating glyph, so the glyph lands exactly on the icon
        (✓/○) column and the body naturally shifts to its right (the tool-name column).
        """
        return " " * (2 + (_SEQ_W + 2))

    def _start_spinner(self, msg: str = "processing") -> None:
        global _active_spinner_printer
        self._stop_spinner()
        _active_spinner_printer = self
        import shutil as _sh_sp
        _indent = self._spinner_indent()
        _max = max(20, _sh_sp.get_terminal_size((80, 24)).columns - _CONSOLE_MARGIN * 2 - 4 - len(_indent))
        _safe_msg = self._spinner_safe(msg, _max)
        self._spinner_msg = _safe_msg
        self._spinner_running = True
        _ensure_console_imported()
        if _RICH and asi._console:
            self._spawn_rich_live(_safe_msg)
        else:
            self._spinner_thread = threading.Thread(target=self._spinner_worker, daemon=True)
            self._spinner_thread.start()

    def _spawn_rich_live(self, safe_msg: str) -> None:
        """Create/start the Rich Live spinner object. Shared by _start_spinner's
        initial creation and _update_spinner's re-creation (recovery after a
        log emit brings the Live down)."""
        from rich.live import Live
        _indent = self._spinner_indent()
        _spin_text = f"{_indent}{safe_msg}" if safe_msg else ""
        _spinner = self._make_spinner(_spin_text, _C["blue"])
        self._spinner_obj = _spinner   # _update_spinner updates via update() preserving start_time
        self._spinner_live = Live(
            _spinner,
            console=asi._console,
            refresh_per_second=12,
            # See banner Live: dynamic _MarginIO + Rich's default stream redirect
            # form a self-referential FileProxy loop. Interleaving with log emits
            # is handled by _suspend_live_for_log/_TERM_WRITE_LOCK, not Rich reflow.
            redirect_stdout=False, redirect_stderr=False,
            transient=True,
        )
        self._spinner_live.start()

    def _suspend_live_for_log(self) -> None:
        """Bring down the Rich Live spinner row right before a log emit
        (called from _RowSafeEmitMixin, inside _TERM_WRITE_LOCK), so the log
        doesn't overlap the spinner row. Live.stop() joins the refresh thread
        (removing the concurrent-write race) and, since transient=True, erases
        the row. _spinner_running is left as-is so the thinking ticker
        recreates it on the next tick via _update_spinner→_spawn_rich_live
        (spinner returns)."""
        if _RICH and self._spinner_live is not None:
            try:
                self._spinner_live.stop()
            except Exception:
                logging.getLogger(__name__).debug("spinner live stop failed", exc_info=True)
            self._spinner_live = None
            self._spinner_obj = None
            if asi._margin_stderr:
                try:
                    asi._margin_stderr.reset_bol()
                except Exception:
                    logging.getLogger(__name__).debug("margin reset_bol failed", exc_info=True)

    def _update_spinner(self, msg: str) -> None:
        # Collapse newlines and truncate to fit in one terminal line (CJK-aware)
        import shutil as _sh_sp2
        _indent = self._spinner_indent()
        _max = max(20, _sh_sp2.get_terminal_size((80, 24)).columns - _CONSOLE_MARGIN * 2 - 4 - len(_indent))
        _safe_msg = self._spinner_safe(msg, _max)
        self._spinner_msg = _safe_msg
        if not _RICH:
            return
        # If log emit (_suspend_live_for_log) lowered Live (_spinner_running is
        # preserved), recreate with the same message. Wrap recreation/update in _TERM_WRITE_LOCK to
        # serialize with log handler emit's stop — overlapping would cause Live to double-start or
        # leave a broken row.
        with _TERM_WRITE_LOCK:
            if self._spinner_running and self._spinner_live is None:
                self._spawn_rich_live(_safe_msg)
            elif self._spinner_live and self._spinner_live.is_started:
                # Creating a new Spinner object resets start_time → frame_no≈0 → always
                # displays only the first frame (◴) without rotation. The existing instance's
                # .update(text=...) must be used to preserve start_time so ◴◷◶◵ animation works.
                if self._spinner_obj is not None:
                    self._spinner_obj.update(text=f"{_indent}{_safe_msg}")
                else:
                    self._spinner_live.update(self._make_spinner(f"{_indent}{_safe_msg}", _C["blue"]))

    def _stop_spinner(self) -> None:
        global _active_spinner_printer
        if _active_spinner_printer is self:
            _active_spinner_printer = None
        # Execution end/display reset point — prevent filter from staying on when cancel suppresses complete events,
        # which would hide all subsequent terminal logs (a leak).
        _tool_running_filter.active = False
        self._stop_tool_ticker()
        self._stop_thinking_ticker()
        self._spinner_running = False
        if _RICH and self._spinner_live:
            try:
                self._spinner_live.stop()
            except Exception:
                logging.getLogger(__name__).debug("spinner live stop failed", exc_info=True)
            self._spinner_live = None
            self._spinner_obj = None
            if asi._margin_stderr:
                asi._margin_stderr.reset_bol()
        if self._spinner_thread:
            self._spinner_thread.join(timeout=0.5)
            self._spinner_thread = None
        if not _RICH:
            import shutil as _sh
            _tw = _sh.get_terminal_size((80, 24)).columns
            sys.stderr.write("\r" + " " * _tw + "\r")
            sys.stderr.flush()

    def _spinner_worker(self) -> None:
        import shutil as _sh
        i = 0
        _t0 = time.perf_counter()
        while self._spinner_running:
            ch = self._spinner_chars[i % len(self._spinner_chars)]
            _tw = _sh.get_terminal_size((80, 24)).columns
            _max = max(20, _tw - _CONSOLE_MARGIN * 2 - 4)  # leave room for "  ⠇ " prefix
            _msg = self._spinner_msg[:_max]
            # ── Claude Code shimmer: beam scans left↔right, brightening characters ──
            # Rotating glyph is fixed blue, body is per-char painted with beam interpolated colors.
            _elapsed = time.perf_counter() - _t0
            _center, _beam_w = _shimmer_beam(len(_msg), _elapsed)
            _ron, _roff = self._hex_ansi(_C["blue"])
            _buf = [f"\r  {_ron}{ch}{_roff} "]
            for _i, _c in enumerate(_msg):
                if _c == " ":
                    _buf.append(" ")
                    continue
                _on, _off = self._hex_ansi(_shimmer_style_for(_i, _center, _beam_w))
                _buf.append(f"{_on}{_c}{_off}")
            sys.stderr.write("".join(_buf))
            sys.stderr.write("\x1b[K")  # clear rest of line (handles partial wrap)
            sys.stderr.flush()
            time.sleep(0.12)
            i += 1

    def mute(self) -> None:
        """Ignore all events from here on — ESC returned the user to the prompt,
        but while the worker thread keeps receiving the in-flight LLM response
        in the background, this keeps its output from bleeding into the new
        prompt/input screen."""
        self._muted = True
        self._stop_spinner()

    def __call__(self, event_name: str, data: dict) -> None:
        if self._muted:
            return
        # ── Spinner control (runs before SILENT_EVENTS) ──
        if event_name == "routing_intent":
            self._start_spinner("")
        elif event_name == "route_decision":
            self._update_spinner("routing")
        elif event_name == "token_usage":
            self._update_spinner("")
        elif event_name in ("done", "session_end", "error", "cancelled"):
            self._stop_spinner()

        if event_name in _SILENT_EVENTS:
            return

        with self._lock:
            elapsed = time.perf_counter() - self._t0
            label = _EVENT_LABELS.get(event_name, event_name)

            if event_name == "tool_call_preview":
                self._call_seq += 1
                tool = data.get("tool", "?")
                # Show key arg in preview for read/search tools
                _preview_suffix = ""
                _args = data.get("args") or {}
                _preview_arg_keys = {
                    "read_file": "path",
                    "read_symbol": "name",
                    "get_project_info": "path",
                    "list_directory": "path",
                    "grep": "pattern",
                    "glob": "pattern",
                    "find_symbol": "name",
                    "find_references": "name",
                    "find_tests_for_symbol": "symbol",
                    "find_relevant_files": "query",
                    "search_code": "query",
                }
                _arg_key = _preview_arg_keys.get(tool)
                if _arg_key:
                    _val = _args.get(_arg_key, "")
                    if _val:
                        _val = _relativize_repo_paths(str(_val))
                        if tool == "grep":
                            _gpath = _relativize_repo_paths(_args.get("path", "") or ".")
                            _preview_suffix = f"  {_val!r} in {_gpath}"
                        elif tool == "read_symbol":
                            _spath = _relativize_repo_paths(_args.get("file_path", "") or "")
                            _preview_suffix = f"  {_val} in {_spath}" if _spath else f"  {_val}"
                        else:
                            _preview_suffix = f"  {_val}"
                # In-place overwrite using \r (like design_tool_call handler)
                _itag = f"[{self._call_seq}]"
                _preview_line = self._style_row(
                    self._fit_row(f"  {_itag}{_seq_pad(_itag)}○ {tool}{_preview_suffix}"),
                    _C["muted"],
                )
                with _TERM_WRITE_LOCK:
                    if self._preview_active:
                        sys.stdout.write("\n")
                    sys.stdout.write(f"\r\x1b[2K{_preview_line}")
                    sys.stdout.flush()
                    _tool_running_filter.active = True   # suppress terminal logs during tool run
                    _tool_running_filter.row_pending = True
                self._preview_active = True
                self._preview_suffix = _preview_suffix
                self._log(f"[#{self._call_seq}] tool_call: {tool}")
                return

            if event_name == "tool_call":
                tool = data.get("tool", "?")
                ok = data.get("result", {}).get("ok", False)
                err = data.get("result", {}).get("error", "")
                if not ok and err:
                    # bash: informational stderr (file-not-found, etc.) is not displayed as error
                    if tool == "bash" and any(kw in err for kw in ["No such file", "No such file or directory", "cannot access", "does not exist", "No such"]):
                        _suffix = ""
                        _rc = data.get("result", {}).get("content", "")
                        _first_line = _relativize_repo_paths((_rc or err).split("\n", 1)[0].strip())
                        if _first_line:
                            _suffix = f"  {_first_line[:200]}"
                        if self._preview_active:
                            with _TERM_WRITE_LOCK:
                                _tool_running_filter.active = False
                                sys.stdout.write(f"\r\x1b[2K{self._style_row(self._fit_row(f'  [{elapsed:5.1f}s] ✓ {tool}{_suffix}'), _C['green'])}\n")
                                sys.stdout.flush()
                            self._preview_active = False
                        else:
                            self._event(elapsed, "✓", f"{tool}{_suffix}", "text", "green")
                        self._log(f"tool_ok: {tool}{_suffix}")
                    else:
                        short_err = _relativize_repo_paths(err[:2000].replace("\n", " | "))
                        if self._preview_active:
                            with _TERM_WRITE_LOCK:
                                _tool_running_filter.active = False
                                sys.stdout.write(f"\r\x1b[2K{self._style_row(self._fit_row(f'  [{elapsed:5.1f}s] ✗ {tool}: {short_err}'), _C['red'])}\n")
                                sys.stdout.flush()
                            self._preview_active = False
                        else:
                            self._event(elapsed, "✗", f"{tool}: {short_err}", "red", "red")
                        self._log(f"tool_fail: {tool}: {short_err}")
                else:
                    # Extract summary from result content's first line
                    _suffix = ""
                    if ok:
                        _result_content = data.get("result", {}).get("content", "")
                        if _result_content:
                            _first_line = _relativize_repo_paths(_result_content.split("\n", 1)[0])
                            # read_file: "`path` (N lines) lines S-E" → "(N lines) lines S-E"
                            if tool == "read_file":
                                _info = _first_line.replace("`", "").strip()
                                # Strip the leading path (it's in the tool call line already)
                                _space = _info.find(" ")
                                if _space > 0:
                                    _info = _info[_space:].strip()
                            # grep: "grep: 'pattern' in path (N matches) (context)" → "(N matches) (context)"
                            elif tool == "grep":
                                # Extract "(N matches)" and optional "(context)" — string ops instead of regex
                                _info = _first_line
                                _open = _first_line.find('(')
                                if _open != -1:
                                    _close = _first_line.find(')', _open)
                                    if _close != -1:
                                        _match_part = _first_line[_open:_close + 1]
                                        if ' match' in _match_part:
                                            _info = _match_part
                                            _rest = _first_line[_close + 1:].strip()
                                            if _rest.startswith('(') and _rest.endswith(')'):
                                                _info += ' ' + _rest
                            elif tool in ("read_symbol", "find_symbol",
                                          "find_references",
                                          "get_project_info", "find_relevant_files",
                                          "search_code", "find_tests_for_symbol"):
                                _info = _first_line
                            else:
                                _info = ""
                            if _info:
                                _suffix = f"  {_info}"
                    _completion_msg = f"  [{elapsed:5.1f}s] ✓ {tool}{_suffix}"
                    if self._preview_active:
                        with _TERM_WRITE_LOCK:
                            _tool_running_filter.active = False
                            sys.stdout.write(f"\r\x1b[2K{self._style_row(self._fit_row(_completion_msg), _C['green'])}\n")
                            sys.stdout.flush()
                        self._preview_active = False
                    elif self._verbose:
                        self._event(elapsed, "✓", f"{tool}{_suffix}", "text", "green")
                    elif _suffix:
                        # Non-verbose: at least show the file info line
                        self._event(elapsed, "✓", f"{tool}{_suffix}", "blue", "green")
                    self._log(f"tool_ok: {tool}{_suffix}")
                return

            if event_name == "route_applied":
                lane = data.get("lane", "?")
                kind = data.get("task_kind", "")
                conf = data.get("confidence", 0)
                self._event(elapsed, "»", f"{lane} · {kind} · {conf:.2f}", "muted", "muted")
                self._log(f"route: lane={lane} kind={kind} conf={conf:.2f}")
                return

            if event_name == "rate_limit_retry":
                delay = data.get("delay", "?")
                attempt = data.get("attempt", "?")
                max_r = data.get("max_retries", "?")
                self._event(elapsed, "↻", f"rate limit — retry in {delay}s ({attempt}/{max_r})", "yellow", "yellow")
                self._log(f"rate_limit: retry {attempt}/{max_r} in {delay}s")
                return

            if event_name in ("tdd_cycle_pass", "tdd_cycle_fail"):
                ok_ev = "pass" in event_name
                icon = "✓" if ok_ev else "✗"
                _ck = "green" if ok_ev else "red"
                self._event(elapsed, icon, label, _ck, _ck)
                self._log(f"{event_name}")
                return

            if event_name == "fail_loop_detected":
                self._event(elapsed, "▲", "fail loop — switching strategy", "yellow", "yellow")
                self._log("fail_loop_detected")
                return

            if event_name in ("error", "cancelled"):
                msg = data.get("message") or data.get("error") or ""
                self._event(elapsed, "✗", label + (f": {msg[:4000]}" if msg else ""), "red", "red")
                self._log(f"{event_name}: {msg[:4000]}")
                return

            # ── Main agent observability ───────────────────────────────────────
            if event_name == "reasoning":
                text = data.get("text", "")
                if text:
                    self._thinking_buffer += text
                    # Reasoning preview is not shown on the spinner.
                    # The spinner Live(asi._console) and log handler(_log_console) share stderr,
                    # so if INFO logs flow during thinking, the transient spinner
                    # frame (🧠 …) would be duplicated to scrollback on every log line.
                    self._thinking_displayed = True
                return

            if event_name == "agent_thinking":
                content = data.get("content", "")
                if content:
                    flat = " ".join(
                        ln.strip() for ln in content.strip().split("\n") if ln.strip()
                    )
                    # Spinner text: strip markdown symbols, keep as a single line.
                    # Strip markdown syntax — string ops instead of regex
                    _markdown_chars = str.maketrans({'`': ' ', '*': ' ', '#': ' ', '_': ' ', '~': ' '})
                    plain = flat.translate(_markdown_chars)
                    # Collapse multiple dashes and spaces
                    while '---' in plain:
                        plain = plain.replace('---', ' ')
                    plain = ' '.join(plain.split())
                    self._update_spinner(f"  [{elapsed:5.1f}s] … {plain[:120]}")
                return

            if event_name == "turn_start":
                turn = data.get("turn", 0)
                if turn > 1:
                    model = data.get("model", "") or ""
                    model_str = f" ({model})" if model else ""
                    self._event(elapsed, "↻", f"Turn {turn}{model_str}", "peach", "peach")
                    self._log(f"turn {turn}{model_str}")
                # Show remaining thinking summary if truncated
                if self._thinking_displayed and len(self._thinking_buffer) > 800:
                    self._event(elapsed, "…", f"(thinking {len(self._thinking_buffer)} chars total)", "muted", "muted")
                # Reset thinking state for new turn
                self._thinking_buffer = ""
                self._thinking_displayed = False
                return

            if event_name == "token_usage":
                pt = data.get("prompt_tokens", 0)
                ct = data.get("completion_tokens", 0)
                tpt = data.get("total_prompt_tokens", 0)
                tct = data.get("total_completion_tokens", 0)
                tcost = data.get("total_cost_usd", 0)
                # Cache-read tokens — this turn + session cumulative
                crt = data.get("cache_read_tokens", 0) or 0
                tcrt = data.get("total_cache_read_tokens", 0) or 0
                _prov = data.get("provider", "") or ""
                self._total_prompt = tpt or self._total_prompt + pt
                self._total_completion = tct or self._total_completion + ct
                # Show cache-adjusted cost if available and different
                actual_cost = data.get("total_actual_cost_usd")
                cost_display = f"${tcost:.4f}"
                if actual_cost is not None and abs(actual_cost - tcost) > 0.000001:
                    cost_display = f"${tcost:.4f} (actual: ${actual_cost:.4f}, cache savings)"
                # Dollar amount intentionally omitted from the user-facing token line
                # (cost is an estimate, not a precise charge). Debug _log keeps it.
                # Per-turn + session cumulative cache hit% (DeepSeek-style)
                _hit_suffix = ""
                if _prov and (crt or tcrt):
                    try:
                        from external_llm.agent._shared_utils import cache_hit_pct
                        # cache_creation MUST be in the denominator for separate-
                        # accounting providers (Anthropic/z.ai): a cache-WRITE turn
                        # (cold start / post-eviction prefix rebuild) spends cc
                        # tokens, so omitting it overstates the hit%. DeepSeek/
                        # OpenAI send cc=0 and are unaffected. (design-chat lane
                        # already does this via _cache_hit_ratio — keep parity.)
                        _cct = data.get("cache_creation_tokens", 0) or 0
                        _tcct = data.get("total_cache_creation_tokens", 0) or 0
                        _turn_hit = cache_hit_pct(_prov, pt, crt, _cct) if crt else 0.0
                        _sess_hit = cache_hit_pct(_prov, tpt, tcrt, _tcct) if tcrt else 0.0
                        if _sess_hit > 0:
                            _hit_suffix = f"  ·  {_turn_hit:.0f}% cached / {_sess_hit:.0f}% sess"
                    except Exception:
                        logging.getLogger(__name__).debug("cache hit pct failed", exc_info=True)
                _print(
                    f"  tok ↑{_abbrev_tokens(pt)} ↓{_abbrev_tokens(ct)}  ·  total ↑{_abbrev_tokens(tpt)} ↓{_abbrev_tokens(tct)}{_hit_suffix}",
                    _C["muted"],
                )
                _log_hit = ""
                if _hit_suffix:
                    _log_hit = f" hit: turn {_turn_hit:.1f}% session {_sess_hit:.1f}%"
                self._log(f"tokens: ↑{pt:,} ↓{ct:,} total ↑{tpt:,} ↓{tct:,} {cost_display}{_log_hit}")
                return

            # ── Design chat observability ──────────────────────────────────────────────
            if event_name == "design_llm_call":
                # Tokens from the last single LLM call (separate cache miss/hit accounting) — next ✓ line sums via total_input_tokens()
                self._pending_llm_tokens = data.get("prompt_tokens", 0) or 0
                self._pending_llm_cache_read = data.get("cache_read_tokens", 0) or 0
                self._pending_llm_provider = data.get("provider", "") or ""
                self._pending_llm_cache_creation = data.get("cache_creation_tokens", 0) or 0
                return

            if event_name == "design_tool_call":
                self._stop_spinner()
                tool = data.get("tool", "?")
                status = data.get("status", "")
                preview = data.get("preview", "")
                args = data.get("args") or {}
                # Extract actual command from shell_exec, etc.
                cmd_hint = _extract_tool_cmd(args)
                if status == "running":
                    # Tools can run concurrently via ThreadPoolExecutor — running/complete
                    # events arrive interleaved, so track in-flight state by call_id.
                    # If call_id is missing (provider-specific omission), fall back to anon key — prevents
                    # two concurrent tools from merging into the same empty key, losing one completion.
                    self._inflight_ctr += 1
                    cid = data.get("call_id") or f"anon-{self._inflight_ctr}"
                    self._inflight[cid] = {
                        "tool": tool, "hint": cmd_hint or "", "t0": time.perf_counter(),
                    }
                    # Draw live line (representative tool + concurrent count) immediately at the bottom of screen.
                    # No number attached — assigned [1][2][3]… on the ✓ line in completion order.
                    # Synchronous render means even sub-1s tools briefly show ○ (ticker animates after 1s).
                    with _TERM_WRITE_LOCK:
                        _tool_running_filter.active = True   # suppress terminal logs during run
                        sys.stdout.write(f"\r\x1b[2K{self._shimmer_row(self._fit_row(self._render_live_line('○')), time.perf_counter())}")
                        sys.stdout.flush()
                        _tool_running_filter.row_pending = True
                    self._live_drawn = True
                    # Tools taking 1s+ get a live line that animates with spinner+elapsed time
                    self._start_tool_ticker()
                elif status in ("complete", "ok"):
                    if _RICH and asi._margin_stderr:
                        asi._margin_stderr.reset_bol()
                    # Pop the in-flight item matching by call_id. Even without a match, still
                    # print ✓ — a completion event must never vanish silently.
                    _info = self._pop_inflight(data.get("call_id"))
                    _hint = _info["hint"] if _info else (cmd_hint or "")
                    import shutil as _shutil2
                    _tw = _shutil2.get_terminal_size((80, 24)).columns
                    # Duration of this tool (not session cumulative)
                    _tool_secs = (time.perf_counter() - _info["t0"]) if _info else elapsed
                    # seq assigned in completion order → screen numbers always ascending [1][2][3]…
                    self._call_seq += 1
                    _seq = self._call_seq
                    _elapsed_str = f"  {_tool_secs:.1f}s{self._consume_llm_tokens_str()}"
                    _gon, _goff = self._hex_ansi(_C["green"])
                    _stag = f"[{_seq}]"
                    _spad = _seq_pad(_stag)
                    _base = f"  \x1b[2m{_stag}\x1b[22m{_spad}{_gon}✓{_goff} {tool}"
                    # Hint width calculation uses plain length (no ANSI) — prevents color/dim codes from
                    # eating available width. 3 = "  " before hint + 1 col wrap slack.
                    _base_plain = f"  {_stag}{_spad}✓ {tool}"
                    _avail = _tw - len(_base_plain) - len(_elapsed_str) - 3
                    _suffix = f"  {self._spinner_safe(_hint, max(0, _avail))}" if _hint and _avail > 4 else ""
                    # Clear live line with \r\x1b[2K and finalize
                    # Remaining in-flight live line is after preview output, by _refresh_live_line
                    # redrawn at the bottom (invariant: 'live line is always last').
                    with _TERM_WRITE_LOCK:
                        sys.stdout.write(f"\r\x1b[2K{_base}{_suffix}{_elapsed_str}\n")
                        sys.stdout.flush()
                        _tool_running_filter.row_pending = False
                    self._live_drawn = False
                    # update_plan: if event carries a structured plan, render the full checklist + change annotations
                    # instead of text preview (legacy events fall back to
                    # _select_preview_lines's 2-line summary below)
                    _plan_data = data.get("plan") if tool == "update_plan" else None
                    if isinstance(_plan_data, dict) and _plan_data.get("items"):
                        self._render_plan_update(_plan_data, data.get("plan_prev"))
                    elif preview:
                        _col_w = _shutil2.get_terminal_size((80, 24)).columns - _CONSOLE_MARGIN * 2 - 6
                        # _hint: cmd_hint from the in-flight item popped above (no recalculation)
                        _preview_lines = _relativize_repo_paths(_strip_ansi(preview)).split("\n")
                        _preview_lines = [
                            ln for ln in _preview_lines
                            if ln.strip() and not ln.strip().startswith("```")
                        ]
                        # first line: strip repeated cmd hint
                        if _preview_lines:
                            _fl = _preview_lines[0]
                            if _hint and _hint in _fl:
                                _fl = _fl.replace(_hint, "", 1).strip()
                                _fl = _fl.lstrip('`').lstrip()  # strip leading backticks/whitespace — no regex
                                for _pfx in ("grep:", "in ", "— ", ": ", "—"):
                                    if _fl.startswith(_pfx):
                                        _fl = _fl[len(_pfx):].strip()
                            if _fl:
                                _preview_lines[0] = _fl
                            else:
                                _preview_lines.pop(0)
                        import textwrap as _textwrap2
                        def _wrap_preview(text: str) -> str:
                            """Pre-wrap preview line so Rich continuation gets subsequent_indent."""
                            return _textwrap2.fill(
                                text.strip(), width=_col_w,
                                initial_indent="  ", subsequent_indent="  ",
                            )
                        if tool == "read_file":
                            # Show line count info only: strip path to avoid
                            # truncating the line range at narrow terminal widths.
                            _raw_preview = _relativize_repo_paths(_strip_ansi(preview))
                            _plines = _raw_preview.split("\n")
                            if _plines:
                                _info = _plines[0].replace("`", "").strip()
                                _space = _info.find(" ")
                                if _space > 0:
                                    _info = _info[_space:].strip()
                                if _info:
                                    _print(_wrap_preview(_info), _C["muted"])
                        elif tool == "read_symbol" and _preview_lines:
                            # First line: "**kind** `name` defined in `file.py:line`"
                            # Show only "[kind] file.py:line" — skip code body lines
                            _fl = _preview_lines[0]
                            _loc_info = _fl.replace("`", "")
                            _def_tok = "defined in "
                            _di = _fl.find(_def_tok)
                            if _di != -1:
                                _loc_raw = _fl[_di + len(_def_tok):].replace("`", "").strip()
                                # Extract kind from **kind**
                                _kind = ""
                                if _fl.startswith("**"):
                                    _ke = _fl.find("**", 2)
                                    if _ke != -1:
                                        _kind = _fl[2:_ke]
                                _loc_info = f"[{_kind}] {_loc_raw}" if _kind else _loc_raw
                            _print(f"  {_loc_info[:_col_w]}", _C["muted"])
                        else:
                            for _result_ln in _select_preview_lines(tool, _preview_lines):
                                _print(_wrap_preview(_result_ln), _C["muted"])
                    # If concurrent tools are still running, redraw live line at bottom.
                    self._refresh_live_line()
                elif status == "error":
                    if _RICH and asi._margin_stderr:
                        asi._margin_stderr.reset_bol()
                    import shutil as _shutil3
                    import textwrap as _textwrap
                    _tw = _shutil3.get_terminal_size((80, 24)).columns
                    _einfo = self._pop_inflight(data.get("call_id"))
                    _tool_secs = (time.perf_counter() - _einfo["t0"]) if _einfo else elapsed
                    self._call_seq += 1
                    _seq = self._call_seq
                    _elapsed_str = f"  {_tool_secs:.1f}s{self._consume_llm_tokens_str()}"
                    _ron, _roff = self._hex_ansi(_C["red"])
                    _stag = f"[{_seq}]"
                    _spad = _seq_pad(_stag)
                    _base = f"  \x1b[2m{_stag}\x1b[22m{_spad}{_ron}✗{_roff} {tool}"
                    _base_plain = f"  {_stag}{_spad}✗ {tool}"
                    _avail = _tw - len(_base_plain) - len(_elapsed_str) - 3
                    _hint = _einfo["hint"] if _einfo else (cmd_hint or "")
                    _suffix = f"  {self._spinner_safe(_hint, max(0, _avail))}" if _hint and _avail > 4 else ""
                    _preview_lines = (preview or "").split("\n")
                    _err_first_line = _strip_ansi(_preview_lines[0].strip())
                    # "[stderr]" alone is just a label — use the next non-empty line as the real message
                    if _err_first_line.lower() == "[stderr]" and len(_preview_lines) > 1:
                        _err_first_line = _strip_ansi(
                            next((ln.strip() for ln in _preview_lines[1:] if ln.strip()), "")
                        )
                    err = _err_first_line[:400] if _err_first_line else _strip_ansi((preview or "")[:400].replace("\n", " "))
                    err = _relativize_repo_paths(err)
                    # ✗ header: same raw write as ✓ success header (consistent indent)
                    # error detail: textwrap so continuation lines align (consistent with ✓ preview)
                    # \r\x1b[2K  to clear live line and finalize ✗ line (newline). Remaining in-flight
                    # live line is redrawn by _refresh_live_line after error detail output.
                    with _TERM_WRITE_LOCK:
                        sys.stdout.write(f"\r\x1b[2K{_base}{_suffix}{_elapsed_str}\n")
                        sys.stdout.flush()
                        _tool_running_filter.row_pending = False
                    self._live_drawn = False
                    if err:
                        # _out_console width = _tw - _CONSOLE_MARGIN*2; match it so Rich never re-wraps
                        _detail_w = _tw - _CONSOLE_MARGIN * 2
                        _wrapped = _textwrap.fill(err, width=_detail_w,
                                                  initial_indent="  ",
                                                  subsequent_indent="  ")
                        _print(_wrapped, _C["red"])
                    self._refresh_live_line()
                elif status == "switching":
                    # Join preview's newlines with spaces for single-line output (prevents line breakage)
                    _sw_flat = " ".join(ln.strip() for ln in (preview or "").split("\n") if ln.strip())
                    self._event(elapsed, "⇄", "agent switch" + (f": {_sw_flat[:4000]}" if _sw_flat else ""), "blue", "blue")
                    # Log only first line — prevents duplication since full content goes through logging handler to terminal
                    self._log(f"agent_switch: {_sw_flat[:600]}")
                return

            if event_name == "design_plan_gate":
                # Work-plan completion gate fired — the model tried to end the
                # turn with open plan items. Surface it so the user can see the
                # nudge (and how many items remain), rather than silently
                # looping. Only present in design chat (no spinner interaction).
                _open_titles = data.get("open_items", []) or []
                _nudge = data.get("nudge", 1)
                _max_nudges = data.get("max_nudges") or _nudge
                self._stop_spinner()
                _summary = "; ".join(t for t in _open_titles[:3] if t)
                if len(_open_titles) > 3:
                    _summary += f"; +{len(_open_titles) - 3} more"
                self._event(
                    elapsed, "↻",
                    f"plan gate — {len(_open_titles)} open item(s), nudge {_nudge}/{_max_nudges}"
                    + (f": {_summary}" if _summary else ""),
                    "yellow", "yellow",
                )
                return

            if event_name == "design_thinking_start":
                # LLM call starts — start "… thinking · Ns" live ticker.
                # Stopped at call end (design_thinking / design_tool_call → _stop_spinner).
                # Ticker stop on all return/exception paths is guaranteed by the
                # design_thinking_stop event emitted by design_chat_loop.respond()'s finally block.
                self._start_thinking_ticker()
                return

            if event_name == "design_thinking_stop":
                # Always stop ticker/spinner — unlike design_thinking (content render + stop),
                # this event has the sole purpose of "cleanly stopping the ticker without content."
                # Emitted exactly once from design_chat_loop.respond()'s finally across all
                # return/exception paths, preventing ticker-stuck bugs during final message rendering.
                self._stop_spinner()
                return

            if event_name == "design_thinking":
                content = data.get("content", "")
                # Per-call thinking duration (time this LLM call took) — used instead of cumulative elapsed.
                _think_secs = data.get("elapsed")
                if content:
                    self._stop_spinner()
                    # Live "… thinking" spinner has ended — switch to completion label.
                    # Title: 💭 thought for 6.2s. Body is markdown-rendered.
                    if _RICH and asi._out_console:
                        _RichMD = _rich_markdown_cls()
                        from rich.segment import Segment as _Seg
                        from rich.segment import Segments as _Segs
                        from rich.style import Style as _Style
                        from rich.text import Text as _RichTxt
                        _f = asi._out_console.file
                        if hasattr(_f, "reset_bol"):
                            _f.reset_bol()
                        _hdr = (f"💭 thought for {_think_secs:.1f}s"
                                if _think_secs is not None else "💭 thinking")
                        # Header is NOT placed as panel title — Rich's title, even left-aligned,
                        # sits at a fixed offset (first char at body column), pushing 'thought' rightward
                        # by "💭 ". Instead, directly output a left-aligned header on the gutter baseline.
                        # This way "💭 " (3 display cells) occupies the left gutter (▌+2 spaces=3 cols),
                        # so 'thought''s 't' aligns vertically with the body's first character column.
                        # Body uses color gutter (▌  ) attached directly to each line (instead of a panel,
                        # whose top empty border line creates a gap between header and body) for tight
                        # rendering right below the header. Markdown style is preserved via render_lines.
                        asi._out_console.print()
                        asi._out_console.print(_RichTxt(_hdr, style=f"bold {_C['mauve']}"))
                        _md_opts = asi._out_console.options.update(
                            width=max(10, asi._out_console.width - 3))
                        _md_lines = asi._out_console.render_lines(
                            _RichMD(content.strip()), _md_opts, pad=False)
                        def _is_blank(_l):
                            return all(not _s.text.strip() for _s in _l)
                        while _md_lines and _is_blank(_md_lines[0]):
                            _md_lines.pop(0)
                        while _md_lines and _is_blank(_md_lines[-1]):
                            _md_lines.pop()
                        _bar = _Seg("▌", _Style.parse(_C["mauve"]))
                        _gap = _Seg("  ")
                        _nl = _Seg.line()
                        _segs: list = []
                        for _l in _md_lines:
                            _segs += [_bar, _gap, *_l, _nl]
                        if _segs:
                            asi._out_console.print(_Segs(_segs))
                        asi._out_console.print()
                    else:
                        import shutil as _sh_dt
                        import textwrap as _tw_dt
                        _dt_tw = _sh_dt.get_terminal_size((80, 24)).columns
                        # Plain fallback: left ▌ separates LLM utterance from tool lines
                        _think_hdr = (f"thought for {_think_secs:.1f}s"
                                      if _think_secs is not None else "thinking")
                        sys.stdout.write(f"\n 💭 {_think_hdr}\n")
                        _ind = "  ▌ "
                        for _ln in content.strip().split("\n"):
                            _ln = _ln.strip()
                            if not _ln:
                                sys.stdout.write("  ▌\n")
                                continue
                            for _s in _tw_dt.wrap(_ln, width=max(40, _dt_tw - 2),
                                                  initial_indent=_ind, subsequent_indent=_ind) or [_ind]:
                                sys.stdout.write(_s + "\n")
                        sys.stdout.flush()
                return

            if event_name == "self_review":
                _sr_status = data.get("status", "")
                if _sr_status == "start":
                    _files = data.get("files_changed", 0)
                    _chars = data.get("diff_chars", 0)
                    _scope = (f"{_files} file{'' if _files == 1 else 's'}"
                              if _files else f"{_chars} chars")
                    self._update_spinner(f"  [{elapsed:5.1f}s] ⊙ self-review — checking diff ({_scope})")
                    return
                self._stop_spinner()
                _sr_secs = data.get("elapsed")
                _sr_meta = f"  ·  {_sr_secs:.1f}s" if _sr_secs is not None else ""
                if _sr_status == "clean":
                    self._event(elapsed, "⊙", f"self-review ✓ LGTM{_sr_meta}", "green", "green")
                    self._log("self-review: LGTM")
                elif _sr_status == "failed":
                    _err = data.get("error", "")
                    self._event(elapsed, "⊙",
                                f"self-review skipped (reviewer unavailable){_sr_meta}",
                                "muted", "muted")
                    self._log(f"self-review failed: {_err}")
                else:  # "issues"
                    _content = (data.get("content", "") or "").strip()
                    if _RICH and asi._out_console:
                        _RichMD = _rich_markdown_cls()
                        from rich.text import Text as _RichTxt
                        _f = asi._out_console.file
                        if hasattr(_f, "reset_bol"):
                            _f.reset_bol()
                        _title = _RichTxt(" ⊙ self-review ", style=f"bold {_C['yellow']}")
                        _title.append(f" issues found{_sr_meta}", style=_C["muted"])
                        asi._out_console.print()
                        asi._out_console.print(_bar_panel(
                            _RichMD(_content) if _content else _RichTxt("(no detail)"),
                            title=_title, color=_C["yellow"],
                        ))
                    else:
                        import shutil as _sh_sr
                        import textwrap as _tw_sr
                        _sr_tw = _sh_sr.get_terminal_size((80, 24)).columns
                        sys.stdout.write(f"\n  [{elapsed:5.1f}s] ⊙ self-review — issues found{_sr_meta}\n")
                        _ind = "  ▌ "
                        for _ln in _content.split("\n"):
                            _ln = _ln.rstrip()
                            if not _ln:
                                sys.stdout.write("  ▌\n")
                                continue
                            for _s in _tw_sr.wrap(_ln, width=max(40, _sr_tw - 2),
                                                  initial_indent=_ind, subsequent_indent=_ind) or [_ind]:
                                sys.stdout.write(_s + "\n")
                        sys.stdout.flush()
                    self._log(f"self-review issues: {_content[:300]}")
                return

            if event_name == "scope_violation":
                _err = data.get("error", "")
                _src = data.get("source", "")
                self._event(elapsed, "\u25b2", "scope violation" + (f" ({_src})" if _src else "") + (f": {_err[:300]}" if _err else ""), "yellow", "yellow")
                self._log(f"scope_violation: {_err[:300]}")
                return

            # Other events: verbose mode only
            if self._verbose:
                self._event(elapsed, "·", f"{label}: {str(data)[:2000]}", "muted", "muted")


# ─── Direct engine execution ─────────────────────────────────────────────────────────

def _cjk_width(s: str) -> int:
    """Display width of a string, counting CJK/full-width chars as 2 columns.

    textwrap measures every code point as width 1, so CJK-heavy lines overflow
    the terminal and get re-wrapped at column 0 — losing their left margin.
    Mirrors the East-Asian-Width convention already used by _spinner_safe.
    """
    import unicodedata as _ud
    w = 0
    for ch in s:
        if _ud.combining(ch):
            continue
        w += 2 if _ud.east_asian_width(ch) in ("W", "F") else 1
    return w


def _wrap_cjk(text: str, avail: int, initial_indent: str = "", subsequent_indent: str = "") -> list[str]:
    """Word-wrap `text` to `avail` display columns, CJK-aware.

    `avail` is the total terminal width budget (including indent). Breaks on
    whitespace; a single token wider than the budget is hard-split. Returns the
    fully-indented lines ready to print.
    """
    out: list[str] = []
    indent = initial_indent
    cur = ""
    cur_w = 0
    for word in text.split():
        ww = _cjk_width(word)
        ind_w = _cjk_width(indent)
        sep_w = 1 if cur else 0
        # Start a new line if the word won't fit on the current one.
        if cur and ind_w + cur_w + sep_w + ww > avail:
            out.append(indent + cur)
            indent = subsequent_indent
            cur, cur_w, sep_w = "", 0, 0
            ind_w = _cjk_width(indent)
        # Hard-split a token that is wider than a whole line on its own.
        while ind_w + ww > avail:
            budget = max(1, avail - ind_w)
            piece, piece_w = "", 0
            rest = word
            for i, ch in enumerate(word):
                cw = _cjk_width(ch)
                if piece_w + cw > budget:
                    rest = word[i:]
                    break
                piece += ch
                piece_w += cw
            else:
                rest = ""
            out.append(indent + piece)
            indent = subsequent_indent
            ind_w = _cjk_width(indent)
            word = rest
            ww = _cjk_width(word)
            cur, cur_w = "", 0
        if word:
            if cur:
                cur += " "
                cur_w += 1
            cur += word
            cur_w += ww
    if cur or not out:
        out.append(indent + cur)
    return out


def _cli_checkpoint_cb(question_data: dict) -> dict:
    """Stdin-based user checkpoint callback for CLI mode.

    Blocks and waits for user input from terminal.
    Times out after `timeout` seconds, returning the default answer.

    Before reading stdin, pauses the ESC watcher and temporarily restores
    terminal echo so the user can see what they type (ESC watcher runs stdin
    in non-canonical/noecho mode, which would swallow typed characters).
    """
    import copy as _copy
    import select as _sel
    import sys as _sys
    import termios as _termios
    import time as _time

    # Pause ESC watcher — stop it from consuming stdin bytes while we read
    _esc_watcher_pause.set()

    # Stop any live spinner — its stderr repaints would interleave with the
    # stdout question text below. Same engine thread as stream callbacks.
    _spinner_owner = _active_spinner_printer
    if _spinner_owner is not None:
        try:
            _spinner_owner._stop_spinner()
        except Exception:
            logging.getLogger(__name__).debug("checkpoint: spinner stop failed", exc_info=True)

    # Save terminal state and restore echo+canonical for user input
    _saved_tio = None
    try:
        _fd = _sys.stdin.fileno()
        if _sys.stdin.isatty():
            _saved_tio = _copy.deepcopy(_termios.tcgetattr(_fd))
            _restored = _termios.tcgetattr(_fd)
            _restored[3] |= _termios.ECHO | _termios.ICANON
            # Restoring ICANON|ECHO (c_lflag) alone is not enough: prompt_toolkit's
            # raw mode / the ESC watcher can leave ICRNL cleared in c_iflag. Without
            # CR→NL translation, Enter delivers a bare \r — which canonical mode does
            # NOT treat as a line terminator, so readline() never completes and each
            # Enter echoes as a literal ^M (ECHOCTL). Force ICRNL on (and clear IGNCR
            # so CR isn't dropped entirely) to guarantee Enter submits the line.
            _restored[0] |= _termios.ICRNL
            _restored[0] &= ~_termios.IGNCR
            _termios.tcsetattr(_fd, _termios.TCSADRAIN, _restored)
    except Exception:
        logging.getLogger(__name__).debug("checkpoint: terminal state save failed", exc_info=True)

    # Pre-initialise so finally block can safely reference these even on ValueError
    _M2 = " " * (_CONSOLE_MARGIN + 2)  # "      " (nested items)
    answer = ""
    try:
        question = question_data.get("question", "")

        options = question_data.get("options") or []
        default = str(question_data.get("default", ""))
        try:
            timeout = int(question_data.get("timeout", 120))
        except (ValueError, TypeError):
            timeout = 120

        # Print question to terminal — indent to match _CONSOLE_MARGIN (4 spaces)
        _M = " " * _CONSOLE_MARGIN       # "    "
        _SEP_W = 60 - _CONSOLE_MARGIN    # separator width (fits within standard margin)
        print("\n" + _M + "─" * _SEP_W)
        print(_M + "🔔 [User Checkpoint] LLM has a question:")
        print(_M + "─" * _SEP_W)
        # Wrap long lines (e.g. dangerous shell commands) to the terminal width
        # so soft-wrapped continuation rows keep a left margin instead of falling
        # flush against the terminal edge. _M2 (hanging indent) marks continuations.
        import shutil as _sh_q
        # Leave 1 col of right slack so a full-width CJK glyph never tips the
        # line over the terminal edge (which would re-wrap it at column 0).
        _wrap_w = max(20, _sh_q.get_terminal_size((80, 24)).columns - _CONSOLE_MARGIN - 1)
        # Question body: markdown-rendered (**bold**, `code`, # header, - bullet).
        # _out_console auto-injects left _CONSOLE_MARGIN margin via _MarginIO, so manual _M
        # prefix is unnecessary — aligns at same column (4) as header/option lines (same pattern as
        # design_thinking·self-review). reset_bol() syncs _MarginIO's _bol state to BOL after a raw print.
        # Falls back to existing _wrap_cjk plain-text loop when Rich is unavailable.
        if _RICH and asi._out_console:
            _RichMD = _rich_markdown_cls()
            _f = asi._out_console.file
            if hasattr(_f, "reset_bol"):
                _f.reset_bol()
            asi._out_console.print(_RichMD(question))
        else:
            for _qline in question.splitlines():
                if not _qline.strip():
                    print(_M)
                    continue
                for _wline in _wrap_cjk(_qline, _wrap_w, initial_indent=_M, subsequent_indent=_M2):
                    print(_wline)
        if options:
            print(_M)
            print(_M + "Options:")
            for i, opt in enumerate(options, 1):
                # Wrap long options to prevent falling to col 0 — continuation lines get hanging indent
                # of (_M2 + number width) for alignment.
                _pfx = f"{i}. "
                for _wline in _wrap_cjk(f"{_pfx}{opt}", _wrap_w,
                                        initial_indent=_M2,
                                        subsequent_indent=_M2 + " " * len(_pfx)):
                    print(_wline)
        if default:
            print(_M)
            # [Default: …] (auto-applied in Ns) is long; if the terminal soft-wraps,
            # the continuation "(auto-applied …)" drops to col 0. Wrap with _wrap_cjk
            # (same as body) to maintain left margin (_M2) on continuation lines.
            for _wline in _wrap_cjk(f"[Default: {default}] (auto-applied in {timeout}s)",
                                    _wrap_w, initial_indent=_M, subsequent_indent=_M2):
                print(_wline)
        print(_M + "─" * _SEP_W)

        # Show a prompt indicator so the user knows to type something
        _sys.stdout.write(f"{_M}❯ ")
        _sys.stdout.flush()

        # Wait for input with timeout
        answer = default
        _user_responded = False
        deadline = _time.monotonic() + timeout
        # Remaining-time reminder timing (in canonical mode, typing detection is impossible,
        # so instead of re-rendering every second, notify via newline only at milestones — minimizes
        # visual clutterring while ensuring auto-applied defaults don't pass silently).
        _milestones = [m for m in (60, 30, 10) if m < timeout]
        while _time.monotonic() < deadline:
            remaining = int(deadline - _time.monotonic())
            if _milestones and remaining <= _milestones[0]:
                while _milestones and remaining <= _milestones[0]:
                    _milestones.pop(0)  # Consume already-passed milestones in a batch
                _sys.stdout.write(
                    f"\n{_M}⏳ default auto-applies in {max(remaining, 1)}s\n{_M}❯ ")
                _sys.stdout.flush()
            timeout_sec = max(0, min(1, remaining))
            rlist, _, _ = _sel.select([_sys.stdin], [], [], timeout_sec)
            if rlist:
                line = _sys.stdin.readline().strip()
                _user_responded = True
                if line:
                    answer = line
                break
            if remaining <= 0:
                break

        if options:
            try:
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    answer = options[idx]
                elif _user_responded:
                    # Number entered but out of option range — silently passing the original text
                    # would send the wrong answer to the LLM. State that it's treated as free text.
                    print(f"{_M2}(no option #{answer} — passing it through as a free-form answer)")
            except (ValueError, IndexError):
                # Non-numeric free-text answers are the normal path — pass through
                logging.getLogger(__name__).debug("checkpoint: non-numeric free-text answer")
    finally:
        # Restore terminal echo state if we changed it
        if _saved_tio is not None:
            try:
                _termios.tcsetattr(_sys.stdin.fileno(), _termios.TCSADRAIN, _saved_tio)
            except Exception:
                logging.getLogger(__name__).debug("checkpoint: terminal state restore failed", exc_info=True)
        # Drain any leftover ESC / input bytes from canonical mode buffer
        _drain_stdin(0.05)
        # Resume ESC watcher
        _esc_watcher_pause.clear()

    print(f"{_M2}→ Answer: {answer}\n")
    # Restart the spinner we stopped — the run continues after the answer
    if _spinner_owner is not None:
        try:
            _spinner_owner._start_spinner("")
        except Exception:
            logging.getLogger(__name__).debug("checkpoint: spinner restart failed", exc_info=True)
    return {"status": "answered" if _user_responded else "timeout", "answer": answer or default}


@dataclass
class EngineConfig:
    """Parameter bundle for :func:`_build_engine`.

    Mirrors the former 13-parameter signature: the first eight fields are
    required (they correspond to the old positional parameters), the rest
    are optional and default to the same values the old keyword-only
    parameters used. ``svc``/``route_decision`` are reuse hooks — when left
    ``None``, :func:`_build_engine` creates fresh ones.
    """

    repo_root: str
    request_text: str
    provider: str
    model: str
    api_key: Optional[str]
    max_turns: int
    stream_cb: Callable[..., None]
    cancel_event: Optional[threading.Event]
    svc: Any = None
    route_decision: Any = None
    thinking_mode: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    scoped_verification: bool = True


def _build_engine(config: EngineConfig):
    """Build and return an AgentLoop instance from an :class:`EngineConfig`.

    Optionally reuses an existing svc (LLM service) and route_decision;
    creates a new one when not given.
    """
    from external_llm.agent.task_router import TaskRouter
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.intelligent_service import create_intelligent_service_from_env

    svc = config.svc
    if svc is None:
        svc = create_intelligent_service_from_env(
            config.provider or None, config.model or None, api_key=config.api_key or None
        )
    if svc is None:
        raise RuntimeError(
            "failed to initialize LLM service.\n"
            "check EXTERNAL_LLM_PROVIDER / OPENAI_API_KEY etc.\n"
            f"  --provider {config.provider or '(unset)'} --model {config.model or '(unset)'}"
        )

    route_decision = config.route_decision
    if route_decision is None:
        router = TaskRouter(
            llm_client=svc.llm_service.client,
            model=svc.model,
        )
        route_decision = router.route(config.request_text)

    agent_config = AgentConfig(
        model_name=svc.model or "",
        max_turns=config.max_turns,
        stream_callback=config.stream_cb,
        consume_content_events=False,
        run_lint=False,
        run_tests=False,
        cancel_event=config.cancel_event,
        unrestricted_read=True,   # trusted local CLI: read tools may cross the repo boundary (bash already can)
        user_checkpoint_callback=_cli_checkpoint_cb,
        scoped_verification=config.scoped_verification,
    )
    agent_config.route_decision = route_decision
    agent_config.intent_result = route_decision.intent_result  # Reused by SpecResolver
    if config.thinking_mode is not None:
        agent_config.thinking_mode = config.thinking_mode
    if config.reasoning_effort is not None:
        agent_config.reasoning_effort = config.reasoning_effort

    registry = ToolRegistry(config.repo_root, agent_config)

    from external_llm.agent.agent_loop import AgentLoop
    return AgentLoop(
        llm_client=svc.llm_service.client,
        registry=registry,
        config=agent_config,
        model=svc.model,
    )


def _run_with_cancel(loop, request: str, context: str, cancel_event: threading.Event,
                     esc_watcher_stop: Optional[threading.Event] = None,
                     stream_callback: Optional[Callable] = None):
    """Run loop.run() in a separate thread and return the result.

    If the ESC watcher is running, sends a stop signal via esc_watcher_stop.
    If stream_callback is given, emits a "done" event after execution completes (to end the spinner).
    """
    result_box: list = [None]
    exc_box: list = [None]

    def _worker():
        try:
            result_box[0] = loop.run(request, context=context)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while t.is_alive():
        t.join(timeout=0.2)
        if cancel_event.is_set():
            break

    # ESC watcher abort signal
    if esc_watcher_stop is not None:
        esc_watcher_stop.set()

    t.join(timeout=1)  # cancellation path: minimal wait (daemon thread will be reclaimed)
    _drain_stdin()

    # Emit "done" event after agent execution completes (spinner termination)
    if stream_callback:
        try:
            stream_callback("done", {})
        except Exception:
            logging.getLogger(__name__).debug("done event callback failed", exc_info=True)

    if exc_box[0]:
        raise exc_box[0]
    return result_box[0]


def _run_esc_watcher(cancel_event: threading.Event, stop_event: threading.Event) -> None:
    """Background ESC-key detection thread.

    Switches stdin to non-canonical mode; when ESC (\x1b) input is detected,
    calls cancel_event.set(). Terminal settings are always restored on exit.
    ISIG is kept, so Ctrl+C (SIGINT) still works normally.
    """
    import copy as _copy
    import termios as _termios
    import time as _time
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return
    try:
        old = _copy.deepcopy(_termios.tcgetattr(fd))
        try:
            # Non-canonical + no echo, but KEEP ISIG (Ctrl+C → SIGINT).
            # VMIN=0 so os.read never blocks; the select() with 0.3s timeout
            # ensures we only read when data is available, and the loop can
            # check stop_event promptly.
            new = _termios.tcgetattr(fd)
            new[3] &= ~(_termios.ICANON | _termios.ECHO)
            new[6][_termios.VMIN] = 0
            new[6][_termios.VTIME] = 0
            _termios.tcsetattr(fd, _termios.TCSADRAIN, new)

            while not stop_event.is_set():
                if cancel_event.is_set():
                    break
                if _esc_watcher_pause.is_set():
                    _time.sleep(0.1)
                    continue
                r, _, _ = _select.select([fd], [], [], 0.3)
                if r:
                    ch = os.read(fd, 1)
                    if ch == b'\x1b':
                        cancel_event.set()
                        # Drain remaining bytes in the ESC sequence (e.g. arrow keys)
                        _drain_stdin(0.02)
                        break
        finally:
            try:
                _termios.tcsetattr(fd, _termios.TCSADRAIN, old)
            except Exception:
                logging.getLogger(__name__).debug("esc watcher: terminal restore failed", exc_info=True)
    except Exception:
        logging.getLogger(__name__).debug("esc watcher failed", exc_info=True)


def _split_work_state(content: str) -> tuple[str, str]:
    """Split off the [WORK STATE …] block (and everything after it) from the final response.

    Sometimes the model follows the context's work-state digest convention
    and writes a [WORK STATE …] block directly at the end of its response —
    since this is work metadata rather than body content, it's a candidate
    for dim styling when displayed.

    The [WORK STATE …] block is always at the *very end* of the response and
    must start at the *beginning* of a line (not a mention appearing mid-sentence).
    So this checks lines from the end and finds the last line starting with
    [WORK STATE to split there. Returns: (body, work_state block — "" if none).
    """
    text = (content or "").strip()
    lines = text.splitlines(keepends=False)
    # Find the last line *starting with* [WORK STATE from the end
    ws_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("[WORK STATE"):
            ws_idx = i
            break
    if ws_idx < 0:
        return text, ""
    body = "\n".join(lines[:ws_idx])
    work = "\n".join(lines[ws_idx:])
    return body.rstrip(), work.strip()


def _build_turn_digest(chat_result) -> str:
    """Build a work-state digest from a DesignChatResult's tool-loop record.

    Tool messages are discarded at turn end, so this deterministically
    summarizes which files were read/edited and which commands ran, and
    preserves that alongside the session turn (no LLM call). Failure never
    blocks the turn record itself — the digest is supplementary information.
    """
    if chat_result is None:
        return ""
    try:
        from external_llm.agent.work_state_digest import build_work_state_digest
        return build_work_state_digest(getattr(chat_result, "tool_results", None) or [])
    except (TypeError, ValueError, KeyError, AttributeError):  # result-shape tolerance
        return ""


def _build_orchestrator_digest(orch_result) -> str:
    """Build a work-state digest from an OrchestratorResult.

    Attaches the same [WORK STATE] metadata used for design-chat turns when
    persisting an orchestrator turn. OrchestratorResult differs structurally
    from DesignChatResult — instead of tool_results it has
    subtask_results (each an AgentResult) + summary — so this deterministically
    extracts each sub-agent's status/patches/summary into a concise digest
    (no LLM call). Failure never blocks the turn record.
    """
    if orch_result is None:
        return ""
    try:
        _status = getattr(orch_result, "status", "")
        _total = getattr(orch_result, "total_turns", 0)
        _subs = getattr(orch_result, "subtask_results", None) or []
        _lines: list[str] = []
        _lines.append(f"orchestration status: {_status}")
        if _total:
            _lines.append(f"total sub-agent turns: {_total}")
        for _i, _sr in enumerate(_subs, 1):
            if _sr is None:
                continue
            _ss = getattr(_sr, "status", "") or ""
            # applied_patches is the most useful signal of what a sub-agent changed
            _patches = getattr(_sr, "applied_patches", None) or []
            _final = (getattr(_sr, "final_message", "") or "").strip().split("\n", 1)[0][:80]
            if _patches:
                _patch_names = [
                    (p.get("file") or str(p)) if isinstance(p, dict) else str(p)
                    for p in _patches[:5]
                ]
                _lines.append(f"  sub-{_i} [{_ss}]: patched {', '.join(_patch_names)}")
            elif _final:
                _lines.append(f"  sub-{_i} [{_ss}]: {_final}")
            else:
                _lines.append(f"  sub-{_i} [{_ss}]")
        return "\n".join(_lines) if len(_lines) > 1 else ""
    except (TypeError, ValueError, AttributeError, KeyError):  # result-shape tolerance
        return ""


# ─── Next-task ghost suggestion ─────────────────────────────────────────────────────
# After turn end, give helper (lightweight) model [user request + final answer + work digest],
# generate a "natural next task" line in background, display as
# dim ghost text on empty prompt (accept with →). Same as Claude Code input suggestion.

_NEXT_SUGGEST_SYSTEM = (
    "You predict the user's next request in an interactive coding CLI.\n"
    "Given the user's last request, the assistant's final answer, and a work log,\n"
    "suggest ONE natural follow-up task the user is most likely to type next.\n"
    "Rules:\n"
    "- Reply with the suggestion text only — no quotes, no preamble, no numbering.\n"
    "- One line, imperative, under 100 characters.\n"
    "- Write it in the same language the user used (Korean request → Korean suggestion).\n"
    "- Ground it in the work log / final answer: loose ends mentioned, tests to run,\n"
    "  adjacent files touched, natural verification steps.\n"
    "- If there is no clearly useful next step, reply with exactly: NONE"
)

# Auto-continue variant: the suggestion is not a hint the user may accept — it is
# fed back as the next turn's instruction without a human reading it first. So the
# contract is deliberately asymmetric: NONE is the default, and a continuation is
# only allowed for REQUIRED follow-up work, stated self-contained (the next turn
# starts with fresh context — no "it/that" references to this conversation).
_AUTO_NEXT_SUGGEST_SYSTEM = (
    "You decide whether an autonomous coding loop should run one more step, and with\n"
    "what instruction. Given the user's last request, the assistant's final answer,\n"
    "and a work log:\n"
    "- Reply with the next instruction ONLY IF the final answer shows follow-up work\n"
    "  is REQUIRED: unfinished steps, changes not yet verified, failing checks, or an\n"
    "  explicitly stated mandatory next step.\n"
    "- Optional ideas, nice-to-haves, and speculative improvements are NOT required\n"
    "  follow-up — reply with exactly: NONE\n"
    "- If the work is complete and verified, reply with exactly: NONE\n"
    "- The instruction is executed in a fresh session: make it self-contained (name\n"
    "  the files/symbols/commands), and start with verifying the previous step's\n"
    "  result when it was not verified yet.\n"
    "- One line, imperative, under 250 characters, same language as the user\n"
    "  (Korean request → Korean instruction).\n"
    "- Reply with the instruction text only — no quotes, no preamble, no numbering."
)

# Countdown before an auto-continue submit fires (seconds). The window exists so a
# watching user can veto (typing/Esc) — too short removes the veto, too long makes
# the loop crawl when unattended.
def _parse_auto_continue_delay() -> float:
    """Parse ``ASICODE_AUTO_CONTINUE_DELAY``; fall back to 8s on any misconfig.

    Same contract as webapp.llm_execution._parse_edit_run_deadline_s: a typo
    must not silently become a boundary value. ``ValueError`` alone does not
    catch non-finite floats — ``float("inf")`` passes the try and makes the
    auto-continue Timer below never fire, silently disabling the feature.
    """
    _raw = (os.environ.get("ASICODE_AUTO_CONTINUE_DELAY") or "").strip()
    _default = 8.0
    if not _raw:
        return _default
    try:
        _v = float(_raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "ASICODE_AUTO_CONTINUE_DELAY=%r is not a number; using %s default",
            _raw, _default,
        )
        return _default
    if not math.isfinite(_v):
        logging.getLogger(__name__).warning(
            "ASICODE_AUTO_CONTINUE_DELAY=%r is not finite; using %s default",
            _raw, _default,
        )
        return _default
    # Sub-floor values clamp up to the 2s veto floor (existing behavior): the
    # floor exists so a watching user can veto — short is usable, not fatal.
    return max(2.0, _v)


_AUTO_CONTINUE_DELAY: float = _parse_auto_continue_delay()
# Auto instructions must be self-contained, so they get a longer budget than the
# 100-char display hint (system prompt says 250; buffer for tokenization overshoot).
_AUTO_SUGGESTION_MAX_LEN = 300


def _parse_auto_arg(arg: str, cur_on: bool) -> tuple[Optional[bool], Optional[int], Optional[str]]:
    """Parse the ``/auto`` argument → ``(new_on, new_cap, error)``.

    ``new_cap`` is ``None`` when the cap is unchanged. Pure — unit-tested in
    ``tests/unit/test_auto_continue.py``.
    """
    a = arg.strip().lower()
    if not a:
        return (not cur_on, None, None)
    if a in ("off", "stop", "no"):
        return (False, None, None)
    if a in ("on", "yes"):
        return (True, None, None)
    if a.isdigit() and int(a) > 0:
        return (True, int(a), None)
    return (None, None, "usage: /auto [N | on | off]  (N = max consecutive auto steps)")


def _auto_continue_should_arm(on: bool, depth: int, cap: int, suggestion: str) -> tuple[bool, str]:
    """Decide whether a countdown auto-submit may be armed. Pure.

    Returns ``(arm, reason)`` where ``reason`` names the blocking condition
    (``"off" | "no_suggestion" | "cap_reached"``) — callers use it to notify
    the cap stop exactly once instead of silently going quiet.
    """
    if not on:
        return (False, "off")
    if not suggestion:
        return (False, "no_suggestion")
    if depth >= cap:
        return (False, "cap_reached")
    return (True, "")


def _text_has_hangul(s: str) -> bool:
    """Return True if *s* contains a Hangul syllable or Jamo.

    Character-class based (no regex/keywords): AC00-D7A3 syllables, 1100-11FF
    Jamo, 3130-318F compat-Jamo. Used to enforce "same language as user".
    """
    for ch in s:
        o = ord(ch)
        if (0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF
                or 0x3130 <= o <= 0x318F):
            return True
    return False


def _validate_next_suggestion(text: str, user_request: str,
                              max_len: int = 140) -> Optional[str]:
    """Enforce the rules stated in ``_NEXT_SUGGEST_SYSTEM`` structurally.

    The system prompt already tells the model: one short imperative line, same
    language as the user, no preamble, NONE if no step. A model that ignores
    those rules must not leak noise onto the prompt (e.g. an English
    restatement of a Korean request). Returns the cleaned text, or ``None`` to
    suppress. Character-class based — no keyword/regex heuristics.

    ``max_len``: 140 for the display-hint contract ("under 100 characters" +
    tokenization-overshoot buffer); auto-continue instructions pass
    ``_AUTO_SUGGESTION_MAX_LEN`` (self-contained → longer budget).
    """
    # Self-contained: don't rely on the caller having stripped.
    text = text.strip()
    if not text:
        return None
    if re.match(r"(?i)^none[.!?]?$", text):
        return None
    if len(text) > max_len:
        return None
    # Language/script guard: a Korean request (Hangul) must yield a Korean
    # suggestion. All-ASCII reply to Hangul input == rule violation
    # (preamble / translation / meta-commentary) → suppress.
    if _text_has_hangul(user_request) and not _text_has_hangul(text):
        return None
    # Verbatim-echo guard: a genuine next step never quotes the user's own
    # request back. Catches "First, the user said: '<request>' …".
    _ureq = user_request.strip()
    if len(_ureq) >= 8 and _ureq[:48] in text:
        return None
    return text


def _invalidate_next_suggestion() -> None:
    """Called on new turn start — clears the displayed suggestion and invalidates any in-flight generation result."""
    global _next_prompt_suggestion, _next_suggestion_gen
    _next_suggestion_gen += 1
    _next_prompt_suggestion = ""
    _cancel_auto_submit()


def _cancel_auto_submit() -> None:
    """Cancel any pending auto-continue countdown (typing / Esc / new turn / mode off)."""
    global _auto_submit_gen, _auto_countdown_active
    _auto_submit_gen += 1
    _auto_countdown_active = False


def _maybe_arm_auto_submit() -> None:
    """Arm the countdown auto-submit for the currently displayed suggestion.

    Called from inside the prompt app context (``_deliver_next_suggestion``'s
    apply, or ``_collect_input``'s pre_run seeding), only for the main prompt
    (``_input_underline``). A ``threading.Timer`` fires after
    ``_AUTO_CONTINUE_DELAY``; the submit re-checks every liveness condition at
    fire time, so cancellation is generation-based — no timer bookkeeping.
    """
    global _auto_submit_gen, _auto_countdown_active
    st = _auto_continue_state
    arm, reason = _auto_continue_should_arm(
        st["on"], st["depth"], st["cap"], _next_prompt_suggestion)
    if not arm:
        if reason == "cap_reached":
            # Notify once, then stand down until the user re-anchors (manual
            # input resets depth) or re-arms via /auto.
            st["depth"] = 0
            _notify_above_prompt(
                f"  🔁 auto-continue: cap reached ({st['cap']} consecutive steps) — "
                "paused until your next input", _C["yellow"])
        return
    _auto_submit_gen += 1
    _auto_countdown_active = True
    gen = _auto_submit_gen
    sug_gen = _next_suggestion_gen
    _notify_above_prompt(
        f"  🔁 auto-continue {st['depth'] + 1}/{st['cap']} in "
        f"{_AUTO_CONTINUE_DELAY:.0f}s — type/Esc to cancel · Enter to run now",
        _C["muted"])

    def _fire() -> None:
        _sess = _prompt_session
        _app = getattr(_sess, "app", None) if _sess is not None else None
        if _app is None or not _app.is_running:
            return
        _loop = getattr(_app, "loop", None)
        if _loop is None:
            return
        _loop.call_soon_threadsafe(lambda: _auto_submit_now(gen, sug_gen))

    _t = threading.Timer(_AUTO_CONTINUE_DELAY, _fire)
    _t.daemon = True
    _t.start()


def _auto_submit_now(gen: int, sug_gen: int) -> None:
    """Submit the ghost suggestion as this prompt's input (runs in the app loop).

    Every condition is re-validated here — a stale timer must be a no-op:
    countdown cancelled (gen), a new turn started (sug_gen), auto turned off,
    an auxiliary y/N prompt is showing (``_input_underline`` off), or the user
    started typing (non-empty buffer).
    """
    global _last_input_was_auto, _auto_countdown_active
    if gen != _auto_submit_gen or sug_gen != _next_suggestion_gen:
        return
    if not _auto_continue_state["on"] or not _input_underline:
        return
    _sess = _prompt_session
    _app = getattr(_sess, "app", None) if _sess is not None else None
    if _app is None or not _app.is_running:
        return
    try:
        _buf = _app.current_buffer
        if _buf.text or not _next_prompt_suggestion:
            return
        _auto_countdown_active = False
        _last_input_was_auto = True
        _buf.insert_text(_next_prompt_suggestion)
        _buf.validate_and_handle()
    except Exception:
        _last_input_was_auto = False
        logging.getLogger(__name__).debug("auto-continue submit failed", exc_info=True)


def _notify_above_prompt(text: str, color: str) -> None:
    """Print a one-line notice above the live prompt without corrupting it.

    A plain ``_print`` suffices: the live prompt always runs inside
    ``patch_stdout(raw=True)``, whose proxy reflows writes from ANY thread
    above the prompt. ``run_in_terminal`` must NOT be used here — it calls
    ``ensure_future()`` and this is reached from the suggestion worker thread,
    which has no running event loop (RuntimeError + a leaked never-awaited
    coroutine warning).
    """
    try:
        _print(text, color)
    except Exception:
        logging.getLogger(__name__).debug("notify above prompt failed", exc_info=True)


def _deliver_next_suggestion(text: str, gen: int) -> None:
    """Store the generated suggestion, and render it as a ghost immediately if the prompt is already showing.

    Called from a worker thread — prompt_toolkit app operations are handed
    off to the app's event loop via call_soon_threadsafe. Discarded on a gen
    mismatch (a new turn started in the meantime).
    """
    global _next_prompt_suggestion
    if gen != _next_suggestion_gen:
        return
    _next_prompt_suggestion = text
    try:
        from prompt_toolkit.auto_suggest import Suggestion
        _sess = _prompt_session
        _app = getattr(_sess, "app", None) if _sess is not None else None
        if _app is None or not _app.is_running or not _input_underline:
            return  # Displayed in the next prompt's pre_run seeding
        def _apply() -> None:
            try:
                _buf = _app.current_buffer
                if (not _buf.text and gen == _next_suggestion_gen
                        and _next_prompt_suggestion):
                    _buf.suggestion = Suggestion(_next_prompt_suggestion)
                    _app.invalidate()
                    _maybe_arm_auto_submit()
            except Exception:
                logging.getLogger(__name__).debug("suggestion apply failed", exc_info=True)
        _loop = getattr(_app, "loop", None)
        if _loop is not None:
            _loop.call_soon_threadsafe(_apply)
    except Exception:
        logging.getLogger(__name__).debug(
            "suggestion display failed — will be seeded in the next prompt", exc_info=True)


def _kick_next_prompt_suggestion(llm_client, model: str, user_request: str,
                                 final_message: str, digest: str,
                                 auto_mode: bool = False) -> None:
    """Generate a "next task" suggestion in the background after turn end (fire-and-forget).

    ``auto_mode``: the suggestion will be countdown-submitted as the next turn's
    instruction (``/auto``), so use the stricter ``_AUTO_NEXT_SUGGEST_SYSTEM``
    contract (required-follow-up only, self-contained, NONE default) and the
    longer length budget. A NONE reply is the loop's natural stop — notify it.

    Failures/delays are silently ignored — a supplementary feature must never
    block the main flow. While this call runs, external_llm INFO/WARNING logs
    are suppressed the same way as for background compress, so they don't
    bleed into the prompt screen.
    """
    gen = _next_suggestion_gen

    def _worker() -> None:
        # If gen changed before we even started (user already typed next
        # input), bail immediately without any LLM call.
        if gen != _next_suggestion_gen:
            return
        _llm_log = logging.getLogger("external_llm")

        class _Quiet(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.levelno >= logging.ERROR

        _filt = _Quiet()
        _llm_log.addFilter(_filt)
        try:
            from external_llm.client import LLMMessage, effective_content
            parts = [f"[user request]\n{user_request[:1500]}"]
            if final_message:
                parts.append(f"[assistant final answer]\n{final_message[:3000]}")
            if digest:
                parts.append(f"[work log]\n{digest[:1000]}")
            # Early gen check: if user started a new turn while we were
            # setting up, don't waste the LLM call.
            if gen != _next_suggestion_gen:
                return
            resp = llm_client.chat(
                messages=[
                    LLMMessage(role="system", content=(
                        _AUTO_NEXT_SUGGEST_SYSTEM if auto_mode
                        else _NEXT_SUGGEST_SYSTEM)),
                    LLMMessage(role="user", content="\n\n".join(parts)),
                ],
                model=model, temperature=0.3, max_tokens=1024,
            )
            _lines = effective_content(resp).strip().strip('"\'`').splitlines()
            text = _validate_next_suggestion(
                _lines[0].strip() if _lines else "", user_request,
                max_len=_AUTO_SUGGESTION_MAX_LEN if auto_mode else 140)
            # System prompt rule (language match·length·preamble forbid·NONE) violations are suppressed
            if text is None:
                if auto_mode and gen == _next_suggestion_gen and _auto_continue_state["on"]:
                    # NONE (or rule violation) = the loop's natural stop — say so
                    # instead of going silent at the prompt.
                    _d = _auto_continue_state["depth"]
                    _notify_above_prompt(
                        "  🔁 auto-continue: no required follow-up — stopped"
                        + (f" after {_d} auto step(s)" if _d else ""), _C["muted"])
                return
            _deliver_next_suggestion(text, gen)
        except Exception:
            logging.getLogger(__name__).debug(
                "next-prompt suggestion failed", exc_info=True)
        finally:
            _llm_log.removeFilter(_filt)

    threading.Thread(target=_worker, daemon=True).start()


def _finalize_pending_design_chat(pending: dict, session_mgr, session_id: str, model: str) -> None:
    """Reclaim a design-chat worker left running in the background by ESC and record it to the session.

    UX policy: on ESC the user returns to the prompt immediately, but
    internally the in-flight LLM call is not cut off — it waits for the
    response and stops at the next checkpoint. This function must be called
    *before* the next user turn is added to the session — the interrupt note
    (or the completed response) must be recorded before the new user turn to
    preserve conversation order in the session context.

    If the worker is still receiving its response, this waits for completion
    (usually it has already finished by the time the user types their next
    input, so no wait actually happens).
    """
    t = pending["thread"]
    if t.is_alive():
        _sp = _ProgressPrinter()
        _sp._start_spinner("finalizing previous task (awaiting in-progress response…)")
        _deadline = time.monotonic() + 60.0
        try:
            while t.is_alive():
                remaining = _deadline - time.monotonic()
                if remaining <= 0:
                    _log = logging.getLogger(__name__)
                    _log.warning("pending design chat thread did not finish in 60s — proceeding")
                    pending["box"]["_timeout"] = True
                    break
                t.join(timeout=min(0.2, remaining))
        finally:
            _sp._stop_spinner()
    # Worker termination confirmed — now safely reset shared config's cancel_event
    _dc_config = pending.get("design_config")
    if _dc_config is not None:
        _dc_config.cancel_event = None

    box = pending["box"]
    err, res = box.get("error"), box.get("result")
    content = (getattr(res, "content", "") or "").strip() if res is not None else ""
    # Only preserve full tool-loop results on ESC interrupt (incomplete response) — right after resume,
    # the turn has all read/edit details from before interruption. Completed turns
    # discard tool_results like a normal completed turn (tool messages are discarded at turn end).
    interrupt_tool_results = None
    if content:
        # If response completed despite ESC — user may not have seen it, but preserve as context
        # for the next turn (enables "show me that content" without regeneration).
        note = (
            "[This response completed in the background after the user pressed ESC. "
            "The user has NOT seen it — do not assume they read it.]\n\n" + content
        )
        digest_src = res
    elif box.get("_timeout"):
        note = (
            "[The previous task was interrupted after 60s timeout — "
            "the response did not complete in time. Please handle the next "
            "user input as a fresh request or continuation as appropriate.]\n\n"
            + _INTERRUPT_RESUME_INSTRUCTION
        )
        digest_src = None
    else:
        partial = getattr(err, "partial_result", None) if err is not None else res
        note = _build_interrupt_note(partial)
        digest_src = partial
        # Actual ESC interrupt — preserve full tool_results.
        # The 300-char summary is replaced/supplemented by full tool_results rendering at display time.
        interrupt_tool_results = list(getattr(partial, "tool_results", None) or [])
    session_mgr.add_turn(
        session_id, "assistant", note, model=model,
        digest=_build_turn_digest(digest_src),
        tool_results=interrupt_tool_results or None,
    )


def _graceful_join_design_chat(
    thread: threading.Thread | None,
    cancel_event: threading.Event | None,
    timeout: float = 3.0,
) -> bool:
    """Signal a design-chat worker thread to stop, then join it for a bounded window.

    Ctrl+C tears down the REPL while the design-chat worker (a daemon thread)
    may be mid-flight in an LLM / tool / RAG-embedding call that creates
    multiprocessing primitives (semaphores registered with resource_tracker via
    joblib/loky). Abrupt interpreter shutdown then orphans those semaphores →
    ``resource_tracker: leaked semaphore objects`` warning after exit.

    Setting ``cancel_event`` makes the worker unwind at its next checkpoint
    (``AgentCancelled``) and release resources normally; the timeout keeps
    Ctrl+C snappy when a long call is still in flight — the daemon thread then
    dies with the process, but only after the grace window.

    Returns True when the thread is no longer alive after the wait.
    """
    if thread is None or not thread.is_alive():
        return True
    if cancel_event is not None:
        cancel_event.set()
    try:
        thread.join(timeout=timeout)
    except KeyboardInterrupt:
        # Second Ctrl+C — the user wants out immediately; skip the wait.
        return False
    return not thread.is_alive()


# ─── REPL ────────────────────────────────────────────────────────────────────

def _format_result(result) -> str:
    """Build the dim token line shown under the result panel.

    Returns only the token line: the former ``(main_text, token_line)`` tuple
    computed a status/patches/turns summary + ``textwrap.fill`` of up to 8000
    chars that its only caller (``_show_result``) discarded — ``_show_result``
    builds its own meta line for both the Rich and plain branches.
    """
    token_line = ""
    if hasattr(result, "metadata") and result.metadata and "tokens" in result.metadata:
        t = result.metadata["tokens"]
        pt = t.get("prompt", 0)
        ct = t.get("completion", 0)
        total = t.get("total", pt + ct)
        lpt = t.get("last_call_prompt", 0)
        lct = t.get("last_call_completion", 0)
        if lpt:
            token_line = f"  tok ↑{_abbrev_tokens(pt)} ↓{_abbrev_tokens(ct)}  ·  last ↑{_abbrev_tokens(lpt)} ↓{_abbrev_tokens(lct)}"
        else:
            token_line = f"  tok ↑{_abbrev_tokens(pt)} ↓{_abbrev_tokens(ct)}  ·  total {_abbrev_tokens(total)}"
    return token_line


def _eval_ctrlc_armed(
    current_armed: bool, is_main_prompt: bool, buffer_has_text: bool,
) -> tuple[bool, bool]:
    """Pure Ctrl+C state machine — returns ``(new_armed, should_raise)``.

    The two-button ``Ctrl+C → arm → Ctrl+C → exit`` pattern is implemented
    as a deterministic state machine so it can be unit-tested without
    prompt_toolkit fixtures.  See ``_handle_ctrlc`` (inside
    ``_collect_input``) for the keybinding usage with side effects
    (buffer reset, hint print).

    Transitions:

    ============= ============ ============== =========== =============
    current_armed is_main_      buffer_has     new_armed   should_raise
                   prompt        _text
    ============= ============ ============== =========== =============
    any           False        any            False       True
    any           True         True           False       False
    False         True         False          True        False
    True          True         False          False       True
    ============= ============ ============== =========== =============
    """
    if not is_main_prompt:
        return (False, True)   # y/N etc → always raise immediately
    if buffer_has_text:
        return (False, False)  # clear buffer + disarm
    if current_armed:
        return (False, True)   # second Ctrl+C → raise
    return (True, False)       # first Ctrl+C → arm


def _collect_input(prompt: str, bottom_toolbar: bool = False) -> str:
    """Read user input using prompt_toolkit.

    Wraps PromptSession for history, paste, IME/hangul, wide chars.
    Ctrl+C → KeyboardInterrupt, Ctrl+D → EOFError.

    Detects an already-running asyncio event loop and runs the prompt
    in a separate thread to avoid prompt_toolkit's internal
    ``asyncio.run()`` raising ``RuntimeError("cannot be called from a
    running event loop")`` (observed under python3.14+ / certain
    shell frameworks that leave an event loop active).

    ``bottom_toolbar``: draw a thin underline rule directly beneath the
    prompt input area (a "textarea" feel) by enabling the session layout's
    separator/filler (see ``_input_underline``). Only meaningful when
    prompt_toolkit is active; ignored on the input() fallback path.
    """
    if not _load_prompt_toolkit():
        try:
            return input(prompt)
        except EOFError:
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise
        except KeyboardInterrupt:
            sys.stdout.write("^C\n")
            sys.stdout.flush()
            raise
    try:
        fd = sys.stdin.fileno()
    except (OSError, AttributeError):
        fd = None            # Non-file stdin → prompt_toolkit fallback
    if fd is not None and not os.isatty(fd):
        line = sys.stdin.readline()
        if not line:          # EOF (pipe/redirect)
            raise EOFError
        return line.rstrip("\n").strip()

    global _prompt_session, _input_underline
    if _prompt_session is None:
        # ── WINCH redraw debounce patch ─────────────────────────────────
        # prompt_toolkit does erase+redraw on every WINCH event (Application.
        # _on_resize). When width shrinks, scrollback reflows → if screen height is exceeded,
        # the terminal scrolls up, but pt always sees prompt as 1 row (cursor_pos.y=0)
        # and only erases "from current line downwards", so the previous prompt pushed *above*
        # cannot be erased, leaving one "❯" ghost per redraw.  # noqa: RUF003
        # Measurements (ASICODE_RESIZE_DEBUG): one slow drag produced 72 WINCH events →
        # debounce reduced redraws to 7 (ghosts ∝ redraw count), but not zero.
        # Eliminating all ghosts would require screen clear on every settle, which would make
        # the conversation disappear from viewport — more costly (cosmetic limit). Instead, set
        # a generous debounce window (0.5s) so micro-pauses during dragging don't trigger redraw,
        # reducing ghosts to 1-2. Remaining ghosts disappear on the next full repaint.
        try:
            from prompt_toolkit.application import Application as _PtApp
            if not getattr(_PtApp, "_asr_resize_debounced", False):
                _orig_on_resize = _PtApp._on_resize

                def _debounced_on_resize(self):
                    try:
                        import asyncio
                        _loop = asyncio.get_running_loop()
                    except RuntimeError:
                        _orig_on_resize(self)
                        return
                    _h = getattr(self, "_asr_resize_handle", None)
                    if _h is not None:
                        _h.cancel()
                    self._asr_resize_handle = _loop.call_later(
                        0.5,
                        lambda: self.is_running and _orig_on_resize(self),
                    )

                _PtApp._on_resize = _debounced_on_resize
                _PtApp._asr_resize_debounced = True
        except Exception:
            logging.getLogger(__name__).debug("pt on_resize debounce patch failed", exc_info=True)

        kb = asi.KeyBindings()
        @kb.add('escape')
        def _handle_escape(event):
            # Lone ESC: veto a pending auto-continue countdown (this turn only —
            # /auto mode itself stays on). Otherwise silently ignored (matches
            # previous behavior).
            if _auto_countdown_active:
                _cancel_auto_submit()
                _notify_above_prompt(
                    "  🔁 auto-continue: cancelled this step (mode stays on — /auto off to disable)",
                    _C["muted"])
        # Ctrl+C: clear buffer only if input is in progress, require double press on empty prompt
        # to propagate KeyboardInterrupt (session exit) — prevents accidental
        # loss of a long session from a habitual single Ctrl+C (paste cancel, etc.). Ctrl+D exits immediately.
        # No time limit (past 2s expiry reset while reading hints, requiring
        # 3-4 presses to exit) — instead, typing/sending etc. disarms.
        @kb.add('c-c')
        def _handle_ctrlc(event):
            global _ctrlc_armed
            new_armed, should_raise = _eval_ctrlc_armed(
                _ctrlc_armed, _input_underline, bool(event.current_buffer.text),
            )
            if not should_raise and event.current_buffer.text:
                # Buffer reset is a side effect not captured by the pure
                # state machine — it clears the user's current input.
                event.current_buffer.reset()
            _ctrlc_armed = new_armed
            if should_raise:
                # raise 금지: 키바인딩은 이벤트 루프 콜백 안에서 실행되므로
                # 여기서 raise하면 prompt()로 전파되지 않고 loop의 예외 핸들러로
                # 새어나가 "Unhandled exception in event loop" 트레이스백이 찍힌다.
                # app.exit(exception=...)가 prompt() 호출 지점에서 raise되는 정석 경로.
                event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")
                return
            if new_armed:
                # First Ctrl+C on main prompt: show "press again" hint.
                try:
                    from prompt_toolkit.application import run_in_terminal
                    run_in_terminal(
                        lambda: _print("  press Ctrl+C again to exit", _C["muted"]))
                except Exception:
                    logging.getLogger(__name__).debug("ctrl+c hint print failed", exc_info=True)
        # Enter: immediate submit. Newline via Option(Alt)+Enter or Ctrl+J.
        # multiline=True means basic Enter inserts newline, so explicitly bind to submit.
        # (Korean IME uses first Enter during syllable composition to confirm syllable, not sending Return
        #  to the app — Terminal.app limitation. The double-Enter approach, due to this,
        #  requires an extra press in Korean, so submit stays as single Enter.)
        @kb.add('enter')
        def _handle_enter(event):
            global _ctrlc_armed, _last_input_was_auto
            _ctrlc_armed = False
            _buf = event.current_buffer
            # Enter on an empty buffer while an auto-continue countdown is
            # pending = "run now": accept the ghost as this auto step (still
            # counted against the depth cap).
            if (_auto_countdown_active and not _buf.text
                    and _next_prompt_suggestion and _input_underline):
                _cancel_auto_submit()
                _last_input_was_auto = True
                _buf.insert_text(_next_prompt_suggestion)
            _buf.validate_and_handle()
        # Option(Alt)+Enter: insert newline
        @kb.add('escape', 'enter')
        def _force_newline_meta_enter(event):
            event.current_buffer.insert_text("\n")
        # Ctrl+J: insert newline (multiline default behavior, but explicitly bound)
        @kb.add('c-j')
        def _newline_ctrl_j(event):
            event.current_buffer.insert_text("\n")
        # run_repl sets _prompt_history_path; if unset (edge case), use in-memory.
        _history = asi.InMemoryHistory()
        if _prompt_history_path:
            try:
                from prompt_toolkit.history import FileHistory
                Path(_prompt_history_path).parent.mkdir(parents=True, exist_ok=True)
                # Cap history growth BEFORE FileHistory reads it all into memory.
                _rotate_cli_history_if_needed(_prompt_history_path)
                _history = FileHistory(_prompt_history_path)
            except Exception:
                logging.getLogger(__name__).debug(
                    "cli history persistence failed — falling back to in-memory", exc_info=True)
        # Input underline (separator) is a thin "─" line in border color.
        # NO_COLOR convention (https://no-color.org) — Rich detects it automatically, but
        # prompt_toolkit styles must be emptied manually.
        _pt_app_style = asi._PtStyle.from_dict(
            {} if os.environ.get("NO_COLOR") else {
                "separator": "fg:" + _C["border"],
                "auto-suggestion": "fg:" + _C["muted"],
            })
        # Ghost-text suggestion: history prefix matching (fish style) while typing,,
        # "next task" suggestion by LLM on empty buffer after turn end. Accept with → key.
        from prompt_toolkit.auto_suggest import (
            AutoSuggest,
            AutoSuggestFromHistory,
            Suggestion,
        )

        class _NextTaskAutoSuggest(AutoSuggest):
            def __init__(self) -> None:
                self._hist = AutoSuggestFromHistory()

            def get_suggestion(self, buffer, document):
                if not document.text:
                    # Main prompt (_input_underline) only — y/N etc.
                    # shouldn't show task suggestion on auxiliary prompts' empty buffer.
                    if _next_prompt_suggestion and _input_underline:
                        return Suggestion(_next_prompt_suggestion)
                    return None
                return self._hist.get_suggestion(buffer, document)

        _prompt_session = asi.PromptSession(
            history=_history,
            key_bindings=kb,
            style=_pt_app_style,
            auto_suggest=_NextTaskAutoSuggest(),
            completer=_SlashCommandCompleter(
                get_provider_fn=lambda: _completer_provider,
                get_model_fn=lambda: _completer_model,
                get_dev_models_fn=lambda: _completer_dev_models,
            ),
            complete_while_typing=True,
            # multiline=True: allow Buffer.newline() to work normally.
            # Without it, b.newline() rejects with Bell sound (single-line mode).
            multiline=True,
            # Completion menu reserve space 0 (default 8 rows): reserved rows carve space between input and underline,
            # so 0 is required for underline to attach directly below input.
            # Fewer reserved rows reduce cursor-row calculation mismatches (❯ ghost)  # noqa: RUF003
            # during drag resize. Menu is a Float, so it dynamically
            # allocates space when shown — unaffected for slash command completion.
            reserve_space_for_menu=0,
        )
        # Typing (text change) disarms the Ctrl+C double-confirm armed state —
        # prevents accidental "habitual single Ctrl+C after reading hint" exit.
        def _disarm_ctrlc(_buf):
            global _ctrlc_armed
            _ctrlc_armed = False
            # Typing vetoes a pending auto-continue countdown (the auto submit
            # itself inserts text, but it deactivates the countdown first, so
            # this only fires for real user keystrokes).
            if _auto_countdown_active:
                _cancel_auto_submit()
        _prompt_session.default_buffer.on_text_changed += _disarm_ctrlc
        # ── Input underline layout ────────────────────────────
        # Default bottom_toolbar is always fixed at screen bottom (large gap of cursor-above margin),
        # cannot serve as "input underline". Instead, place a separator window right after
        # the input container, and let a high-weight filler below absorb all remaining height,
        # making the underline attach directly below the input (multiline included).
        # However, filler forces the input container to 1 row, losing vertical space
        # for the completion menu Float inside it. Re-host the completion menu to an outer
        # FloatContainer so it floats above the filler area. separator/filler/external menu are
        # active only for main prompt (_input_underline=True); y/N etc.
        # prompts use ConditionalContainer folded to 0 rows, keeping original behavior.
        try:
            from prompt_toolkit.filters import (
                Condition as _Condition,
            )
            from prompt_toolkit.filters import (
                has_focus as _has_focus,
            )
            from prompt_toolkit.filters import (
                is_done as _is_done,
            )
            from prompt_toolkit.layout import (
                ConditionalContainer as _Cond,
            )
            from prompt_toolkit.layout import (
                Float as _Float,
            )
            from prompt_toolkit.layout import (
                FloatContainer as _FloatContainer,
            )
            from prompt_toolkit.layout import (
                HSplit as _HSplit,
            )
            from prompt_toolkit.layout import (
                Layout as _Layout,
            )
            from prompt_toolkit.layout import (
                Window as _Window,
            )
            from prompt_toolkit.layout.menus import (
                CompletionsMenu as _CompletionsMenu,
            )

            # ~is_done: prompt_toolkit draws the last frame at preferred height
            # and leaves it in scrollback. Without ~is_done, that frame would include
            # the underline, embedding "────" separators in history every turn
            # (default bottom_toolbar also uses ~is_done for the same reason). Hide when confirmed.
            _underline_on = _Condition(lambda: _input_underline) & ~_is_done
            _sep = _Cond(
                _Window(height=1, char="─", style="class:separator"),
                filter=_underline_on,
            )
            _orig_root = _prompt_session.app.layout.container
            _buf = _prompt_session.default_buffer
            # Make input buffer window non-greedy only in underline mode. Default buffer
            # window has max=∞, eating all remaining height and pushing separator to screen bottom.
            # To prevent this, a large filler would (a) O(max_weight) render explosion in weight mode,
            # (b) preferred mode would balloon layout preferred to screen height,
            # reserving full screen → scroll away existing output — both impossible.
            # Fixing buffer to content height keeps separator right below,
            # and preferred stays small with no screen/scroll issues (multiline
            # grows with content). Regular prompts (_input_underline=False)
            # have the filter off, keeping original greedy behavior.
            try:
                _prompt_session.app.layout.current_window.dont_extend_height = (
                    _underline_on
                )
            except Exception:
                logging.getLogger(__name__).debug(
                    "pt current_window patch failed — skip underline only", exc_info=True)
            _new_root = _FloatContainer(
                _HSplit([_orig_root, _sep]),
                floats=[
                    # When buffer is fixed to 1 row, the root internal completion menu Float loses
                    # vertical space, so re-host the menu to an outer FloatContainer to float it
                    # above the empty area below the separator.
                    _Float(
                        xcursor=True, ycursor=True, transparent=True,
                        content=_CompletionsMenu(
                            max_height=16, scroll_offset=1,
                            extra_filter=_has_focus(_buf) & _underline_on,
                        ),
                    ),
                ],
            )
            _prompt_session.app.layout = _Layout(_new_root)
        except Exception:
            logging.getLogger(__name__).debug("pt layout API patch failed — no underline", exc_info=True)

    # ── running event-loop guard ──────────────────────────────────
    _ev_loop_running = False
    try:
        import asyncio
        _ev_loop_running = asyncio.get_running_loop().is_running()
    except RuntimeError:
        logging.getLogger(__name__).debug("no running asyncio loop")

    _MAX_INPUT_CHARS = 50000  # Default limit, generously above _cfg.lines.DESIGN_TURN_MAX_CHARS
    # ── Input underline: bottom_toolbar=True enables session layout's separator/filler/
    #    external completion menu flag (layout ConditionalContainer
    #    references this global every render). Regular prompts (False) have flag off,
    #    no underline, existing behavior.
    _input_underline = bool(bottom_toolbar)
    _prompt_kwargs: dict = {}
    # If next-task suggestion already arrived before prompt start, show as ghost from first render
    # — auto_suggest only recalculates on text changes, so the initial empty
    # buffer must be directly seeded via pre_run.
    if bottom_toolbar and _next_prompt_suggestion:
        from prompt_toolkit.auto_suggest import Suggestion as _Sugg

        def _seed_suggestion() -> None:
            try:
                _buf = _prompt_session.app.current_buffer
                if not _buf.text and _next_prompt_suggestion:
                    _buf.suggestion = _Sugg(_next_prompt_suggestion)
                    _maybe_arm_auto_submit()
            except Exception:
                logging.getLogger(__name__).debug("suggestion seed failed", exc_info=True)

        _prompt_kwargs["pre_run"] = _seed_suggestion
    # patch_stdout coordinates writes from other threads (LLM-retry warnings,
    # background compress notices, etc.) with the live prompt: while active,
    # sys.stdout/sys.stderr are proxied so such writes print cleanly above
    # the prompt and the prompt redraws afterward, instead of landing at the
    # terminal's raw cursor position and desyncing prompt_toolkit's redraw
    # (see _MarginIO / _RowSafeEmitMixin — logging/print already look up
    # sys.stdout/stderr dynamically so they pick up this proxy).
    # raw=True: our output (rich _print, RichHandler logs) is already rendered
    # to VT100/ANSI (Console force_terminal=True). patch_stdout's default
    # raw=False routes writes through output.write(), which *sanitizes* escape
    # sequences — showing e.g. the muted-gray "press Ctrl+C again to exit" hint
    # as a literal "?[38;2;…m …?[0m". raw=True uses write_raw() so those escapes
    # are passed through and interpreted (color preserved), still reflowed above
    # the live prompt.
    try:
        if not _ev_loop_running:
            with asi.patch_stdout(raw=True):
                text = _prompt_session.prompt(prompt, **_prompt_kwargs)
        else:
            # prompt_toolkit internally calls asyncio.run() which fails
            # when a loop is already active — run in a dedicated thread.
            #
            # A plain daemon thread is required instead of a
            # ThreadPoolExecutor: when the user hits Ctrl+C the worker may
            # still be blocked inside prompt() waiting for input.  Executor
            # workers are non-daemon and are joined at interpreter shutdown
            # (`concurrent.futures.thread._python_exit` → t.join()), which
            # would deadlock the process.  A daemon thread is abandoned at
            # exit, so the main thread's KeyboardInterrupt can propagate and
            # the process terminates.  A single queue carries either the
            # result or the captured exception (no two-queue race).
            import queue as _queue
            _done_q = _queue.Queue()

            def _prompt_in_thread() -> None:
                try:
                    # patch_stdout must be entered on the thread that owns the
                    # event loop prompt_toolkit's asyncio.run() creates (see
                    # its docstring warning) — that's this worker thread, not
                    # the caller, so the context manager is opened here too.
                    # raw=True: pass pre-rendered ANSI through (see sync path).
                    with asi.patch_stdout(raw=True):
                        _done_q.put(("ok", _prompt_session.prompt(prompt, **_prompt_kwargs)))
                except BaseException as _exc:  # incl. KeyboardInterrupt/EOFError
                    _done_q.put(("err", _exc))

            _worker = threading.Thread(target=_prompt_in_thread, daemon=True)
            _worker.start()
            _tag, _payload = _done_q.get()
            if _tag == "err":
                raise _payload
            text = _payload
    except KeyboardInterrupt:
        sys.stdout.write("^C\n")
        sys.stdout.flush()
        raise
    except EOFError:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise
    finally:
        # prompt_toolkit installs its own SIGWINCH handler during prompt() and
        # resets to SIG_DFL on exit (does not restore previous handler). Without re-registration,
        # Rich console width would never update after the first prompt,
        # wrapping subsequent output at old width — re-register after each prompt,
        # and also pick up resizes that occurred while waiting for input.
        if hasattr(signal, "SIGWINCH"):
            try:
                signal.signal(signal.SIGWINCH, _handle_terminal_resize)
            except (ValueError, OSError):
                logging.getLogger(__name__).debug(
                    "SIGWINCH registration failed — non-main thread", exc_info=True)
            _handle_terminal_resize()

    if len(text) > _MAX_INPUT_CHARS:
        _print(f"  ▲ input truncated to {_MAX_INPUT_CHARS} chars.", _C["yellow"])
        text = text[:_MAX_INPUT_CHARS]
    # After input confirm, leave one separator line in scrollback. prompt_toolkit
    # clears the live underline on is_done (prevents history pollution each turn), so submitting
    # removes the underline and tool output appears right after the prompt. Here, emit the same
    # "─" rule directly to maintain visual separation between the submitted prompt and subsequent output.
    if bottom_toolbar and text.strip() and _RICH and asi._out_console:
        asi._out_console.rule(style=_C["border"])
    return text.strip()


def _prompt_input(chat_mode: str = "code", status: str = "") -> str:
    """Display a chat-style input prompt and return user input.

    `status` is a short dim line (e.g. "claude / sonnet · thinking ON") shown
    above the prompt so the active model + thinking state are always visible.
    """
    _ensure_out_console_imported()
    _mode_tag = "[General] " if chat_mode == "general" else ("[Orchestrator] " if chat_mode == "orchestrator" else "")
    _display_mode = "General mode" if chat_mode == "general" else ("Orchestrator mode" if chat_mode == "orchestrator" else "Code mode")
    if _RICH and asi._out_console:
        asi._out_console.rule(style=_C["border"])
        if status:
            from rich.text import Text as _RichText
            asi._out_console.print(_RichText(f"  {status}", style=_C["muted"]))
        asi._out_console.print(
            f"  [{_C['border']}]/help for commands  ·  Enter send  ·  Ctrl+C exit  · Alt+Enter new line · {_display_mode}[/{_C['border']}]"
        )
        # bottom_toolbar: draw a thin underline rule right below the ❯ prompt input area,  # noqa: RUF003
        # forming the "input box" UI. Ignored in fallback(input()) path.
        return _collect_input(f"{_mode_tag}❯ ", bottom_toolbar=True)
    print("━" * 50)
    if status:
        print(f"  {status}")
    print(f"/help for commands  ·  Enter send  ·  Ctrl+C exit  · Alt+Enter new line · {_display_mode}")
    return _collect_input(f"{_mode_tag}> ")


def _wrap_preserve_code(text: str, width: int = 72) -> list[str]:
    """Wrap text to width, preserving code blocks (```...```) as-is."""
    result = []
    # Split on code fences — string ops instead of regex
    raw = text.split('```')
    # Reconstruct: odd-indexed segments are code blocks (wrap with fences)
    segments = []
    for i, seg in enumerate(raw):
        if i % 2 == 1:
            segments.append(f'```{seg}```')
        else:
            segments.append(seg)
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            # Code block: keep intact
            result.extend(seg.split('\n'))
        else:
            result.extend(textwrap.wrap(seg, width=width) if seg.strip() else [])
    return result


def _show_result(
    result, elapsed: float, *,
    repo_root: Optional[str] = None, baseline: Optional[dict] = None,
) -> None:
    """Display execution result, then a colored diff of what the run changed."""
    status_icon = {"success": "✓", "already_satisfied": "=", "max_turns": "▲", "error": "✗", "cancelled": "▲", "clarification_needed": "?"}.get(result.status, "·")
    status_color = {
        "success": _C["green"],
        "already_satisfied": _C["green"],
        "max_turns": _C["yellow"],
        "error": _C["red"],
        "cancelled": _C["yellow"],
        "clarification_needed": _C["yellow"],
    }.get(result.status, _C["text"])

    patches = len(result.applied_patches) if result.applied_patches else 0
    turns = len(result.turns) if result.turns else 0
    msg = result.final_message or ""
    token_line = _format_result(result)

    if _RICH and asi._out_console:
        from rich.console import Group
        _RichMD = _rich_markdown_cls()
        from rich.text import Text

        renderables = []
        if msg:
            renderables.append(_RichMD(msg.strip()))
        if result.error:
            renderables.append(Text(f"  {result.error[:1000]}", style=_C["red"]))
        if token_line:
            renderables.append(Text(f"  {token_line.strip()}", style=_C["muted"]))

        if not renderables:
            renderables.append(Text("", style=""))
        content = Group(*renderables) if len(renderables) > 1 else renderables[0]
        _meta = f"  ·  {elapsed:.1f}s  ·  {turns} turn{'' if turns == 1 else 's'}"
        if patches:
            _meta += f"  ·  {patches} patch{'' if patches == 1 else 'es'}"
        title = Text(f" {status_icon} {result.status} ", style=f"bold {status_color}")
        title.append(_meta, style=_C["muted"])
        asi._out_console.print(_bar_panel(content, title=title, color=status_color))
    else:
        print("─" * 60)
        _meta = f"  {status_icon} {result.status}  ·  {elapsed:.1f}s  ·  {turns} turn{'' if turns == 1 else 's'}"
        if patches:
            _meta += f"  ·  {patches} patch{'' if patches == 1 else 'es'}"
        print(_meta)
        if msg:
            print()
            for _wrapped in _wrap_preserve_code(msg[:4000], width=78):
                print(f"  {_wrapped}")
        if result.error:
            print(f"  ✗ {result.error[:1000]}")
        if token_line:
            print(token_line)

    # ── Changes diff (files changed by this run only) ──
    # Off by default (enable with ASICODE_RUN_DIFF=1). Even when off, show /diff hint if changes exist.
    if repo_root and baseline and result.status not in ("error", "cancelled", "clarification_needed"):
        if _cfg.display.RUN_DIFF:
            try:
                _render_run_diff(repo_root, baseline)
            except Exception:
                logging.getLogger(__name__).debug("run diff render failed", exc_info=True)
        else:
            # Even when diff rendering is off, always show what changed.
            try:
                if _print_run_change_summary(repo_root, baseline):
                    _print("  /diff to view the full diff", _C["muted"])
            except Exception:
                logging.getLogger(__name__).debug("run change summary failed", exc_info=True)


async def _run_collaborate_session(
    registry: Any,
    config: Any,
    task: str,
    context: Optional[str] = None,
) -> Any:
    """Run a single collaboration session (async wrapper for REPL /claude)."""
    from external_llm.repl.collaborate import CollaborationOrchestrator
    async with CollaborationOrchestrator(registry, config) as orch:
        return await orch.run(task=task, context=context, enable_preprocessing=True)


def _save_key_to_dotenv(repo_root: str, key: str, value: str) -> None:
    """Write ``KEY=VALUE`` to the ``.env`` file (updates it if the key already exists)."""
    dotenv_path = os.path.join(repo_root, ".env")
    try:
        with open(dotenv_path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.split("=", 1)[0].strip() == key:
            lines[i] = f'{key}="{value}"\n'
            found = True
            break

    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f'{key}="{value}"\n')

    # Best-effort persistence: the API key already lives in os.environ for this
    # session, so a write failure (read-only mount, full disk, permission denied)
    # must NOT crash the caller — it still has to return the live service object.
    tmp = dotenv_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp, dotenv_path)
        os.chmod(dotenv_path, 0o600)  # API keys — owner-only
        _print(f"  ✓ saved to {dotenv_path}", _C["muted"])
    except OSError as _e:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                logging.getLogger(__name__).debug("dotenv tmp unlink failed", exc_info=True)
        _print(
            f"  ! could not persist {dotenv_path} ({_e.strerror or _e}); "
            f"the key is active for this session only.",
            _C["yellow"],
        )


def _list_provider_model_choices() -> list[tuple[str, str]]:
    """Return a flat list of provider/model combos shown by ``/model``.

    Appends local ollama models, if available, to every combination in
    ``_KNOWN_MODELS``. Each entry is a ``(provider, model)`` tuple.
    """
    choices: list[tuple[str, str]] = []
    for _prov, _models in _KNOWN_MODELS.items():
        choices.extend((_prov, _m) for _m in _models)
    # Local ollama models via the shared cached fetcher (single parse + 10s TTL
    # — identical parsing, no duplicate subprocess/error handling).
    choices.extend(("ollama", _nm) for _nm in _get_ollama_models(timeout=5))
    return choices


def _interactive_provider_setup(repo_root: Optional[str]) -> Optional[tuple[str, str]]:
    """Interactively pick a provider/model when none is configured.

    Shows the same provider/model combinations as ``/model`` as a numbered
    list and lets the user pick one. The selection is written to the
    ``EXTERNAL_LLM_PROVIDER`` / ``EXTERNAL_LLM_MODEL`` env vars and the
    ``.env`` file, so it won't be asked again on the next run. Returns
    ``None`` if not a TTY or cancelled.

    Returns: the selected ``(provider, model)``, or ``None``.
    """
    if not sys.stdin.isatty():
        _print("  no LLM provider configured.", _C["yellow"])
        _print("  ═ set EXTERNAL_LLM_PROVIDER, or pass --provider", _C["muted"])
        return None

    choices = _list_provider_model_choices()
    if not choices:
        _print("  no known provider/model combinations available.", _C["yellow"])
        _print("  ═ set EXTERNAL_LLM_PROVIDER, or pass --provider", _C["muted"])
        return None

    _print("  no LLM provider configured — pick a model to get started:", _C["yellow"])
    _last_prov = ""
    for _i, (_prov, _m) in enumerate(choices, 1):
        if _prov != _last_prov:
            _print(f"    {_prov}:", _C["muted"])
            _last_prov = _prov
        _print(f"      {_i:>2}) {_m}", _C["text"])
    _print("  enter a number (or press Enter to cancel):", _C["muted"])

    try:
        _sel = _collect_input("  >> ").strip()
    except (EOFError, KeyboardInterrupt):
        _print("\n  cancelled.", _C["muted"])
        return None
    if not _sel:
        _print("  cancelled.", _C["muted"])
        return None
    if not _sel.isdigit() or not (1 <= int(_sel) <= len(choices)):
        _print(f"  invalid selection: {_sel}", _C["red"])
        return None

    provider, model = choices[int(_sel) - 1]
    os.environ["EXTERNAL_LLM_PROVIDER"] = provider
    os.environ["EXTERNAL_LLM_MODEL"] = model
    if repo_root:
        _save_key_to_dotenv(repo_root, "EXTERNAL_LLM_PROVIDER", provider)
        _save_key_to_dotenv(repo_root, "EXTERNAL_LLM_MODEL", model)
    _print(f"  ✓ selected {provider} / {model}", _C["green"])
    return provider, model


def _retry_create_svc_with_api_key_prompt(
    factory: Callable,
    provider: Optional[str],
    model: Optional[str],
    *,
    api_key: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Any:
    """Create the LLM service — if no API key is set, prompt the user and retry.

    On success, also saves the API key to the ``.env`` file so it isn't asked again on the next run.
    """
    svc = factory(provider, model, api_key=api_key)
    if svc is not None:
        return svc

    prov = (provider or os.getenv("EXTERNAL_LLM_PROVIDER", "") or "").strip().lower()
    if not prov:
        # provider not set — interactively choose without exiting.
        _picked = _interactive_provider_setup(repo_root)
        if _picked is None:
            return None
        provider, model = _picked
        prov = provider
        # Key already in environment or ollama (no key needed) — created immediately.
        svc = factory(provider, model, api_key=api_key)
        if svc is not None:
            return svc

    _ak_var = _API_KEY_ENV_MAP.get(prov)
    if _ak_var and prov != "ollama":
        existing = os.getenv(_ak_var, "") or ""
        if not existing:
            _print(f"  {_ak_var} is not set.", _C["yellow"])
            _print(f"  enter API key for {prov} (or press Enter to cancel):", _C["muted"])
            try:
                _input_key = _collect_input("  >> ").strip()
            except (EOFError, KeyboardInterrupt):
                _input_key = ""
            if _input_key:
                os.environ[_ak_var] = _input_key
                svc = factory(provider, model, api_key=_input_key)
                if svc is not None:
                    _print(f"  ✓ API key set for {prov}", _C["green"])
                    # ── Also save to .env ──
                    if repo_root:
                        _save_key_to_dotenv(repo_root, _ak_var, _input_key)
                    return svc
                _print("  ✗ API key did not resolve — check the key", _C["red"])
                return None
            _print("  cancelled.", _C["muted"])
            return None
    return None


def _get_ollama_models(timeout: int = 5) -> list[str]:
    """Return local Ollama model names via ``ollama list`` (cached, 10s TTL).

    Returns empty list on any error (not installed, timeout, no models).
    """
    global _ollama_cache, _ollama_cache_ts
    now = time.monotonic()
    if now - _ollama_cache_ts < 10.0 and _ollama_cache is not None:
        return _ollama_cache
    try:
        _r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        if _r.returncode != 0 or not _r.stdout.strip():
            _ollama_cache = []
            _ollama_cache_ts = now
            return []
        _names: list[str] = []
        for _l in _r.stdout.strip().split("\n")[1:]:
            _parts = _l.strip().split()
            if _parts and _parts[0]:
                _names.append(_parts[0])
        _ollama_cache = _names
        _ollama_cache_ts = now
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _ollama_cache = []
        _ollama_cache_ts = now
        return []
    else:
        return _names


def _resolve_repo_root(repo_arg: Optional[str]) -> str:
    """Resolve the repository root directory.

    Priority:
    1. ``--repo`` CLI argument (explicit user choice) — returned as-is.
    2. Git repository root (``git rev-parse --show-toplevel``) from cwd.
       This means ``asi`` works correctly regardless of which
       subdirectory under the repo you run it from.
    3. Current working directory (fallback for non-git directories).
    """
    if repo_arg:
        return str(Path(repo_arg).resolve())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            cwd=os.getcwd(),
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logging.getLogger(__name__).debug("git rev-parse --show-toplevel failed", exc_info=True)
    return str(Path(os.getcwd()).resolve())


def _terminal_config_path(repo_root: str) -> Optional[str]:
    """Per-terminal config path for isolated /model switches, or None if no TTY.

    Each terminal (TTY) gets its own config file under
    ``.asicode/terminals/<tty>.json`` so that ``/model`` switches in one
    terminal don't affect others. When no TTY is attached (piped input,
    non-interactive ``--prompt``), returns ``None`` and callers fall back to
    the shared ``.asicode/config.json``. This keeps the isolation strictly
    structural — every config read/write flows through a single path, so
    pointing that path at a per-terminal file isolates /model, /think and
    /helper all at once.
    """
    try:
        _tty_name = os.ttyname(sys.stdin.fileno())
    except (OSError, ValueError, AttributeError):
        logging.getLogger(__name__).debug("ttyname failed — no per-terminal config")
        return None
    if not _tty_name:
        return None
    _base = os.path.basename(_tty_name)  # /dev/ttys004 → ttys004
    if not _base or "/" in _base:  # guard against odd names
        return None
    return os.path.join(repo_root, ".asicode", "terminals", _base + ".json")


def _seed_terminal_config(term_path: str, shared_path: str) -> None:
    """Seed a terminal config from the shared config.json on first use.

    Copies the shared config so initial provider/model/thinking/helper
    settings are inherited, after which the terminal config evolves
    independently — the shared ``config.json`` is never touched again.
    No-op if the terminal config already exists. Never raises.
    """
    if os.path.exists(term_path):
        return
    try:
        os.makedirs(os.path.dirname(term_path), exist_ok=True)
        _tmp = term_path + ".tmp"
        if os.path.exists(shared_path):
            with open(shared_path, encoding="utf-8", errors="replace") as _sf:
                _data = _sf.read()
        else:
            _data = "{}"
        with open(_tmp, "w", encoding="utf-8") as _tf:
            _tf.write(_data)
        os.replace(_tmp, term_path)
    except OSError:
        logging.getLogger(__name__).debug("terminal config seed failed: %s", term_path, exc_info=True)


# ── Pure helpers for insights compact (extracted for testability) ────────────

def _insights_compact_is_noop(
    result_text: str, content_text: str,
    result_ents: list, orig_ents: list,
) -> bool:
    """True when the compactor returned effectively the same content.

    Compares normalized text (collapse all whitespace runs to single space)
    AND entry count — header-set comparison would false-positive when the
    compactor legitimately shortens body text while preserving headers.
    """
    import re as _re

    def _norm(s: str) -> str:
        return _re.sub(r"\s+", " ", s).strip()
    return _norm(result_text) == _norm(content_text) and len(result_ents) == len(orig_ents)


def _size_compact_budget(content: str) -> int:
    """Return the base ``max_tokens`` for a compact LLM call.

    Uses byte-based estimate (UTF-8 ~2 bytes/token), which covers both
    ASCII (1 byte/char) and CJK (3 bytes/char) correctly.  The char-based
    estimate (len/3.5) would always be smaller for UTF-8, so byte/2 alone
    is sufficient.  Slack (2048) is added and the floor is 8192 so that
    even tiny inputs get a reasonable output window.

    Callers may further raise the budget for reasoning models.
    """
    _tokens = len(content.encode("utf-8")) // 2
    return max(8192, _tokens + 2048)


def _dropped_entries(orig_ents: list, result_ents: list) -> list:
    """Return entries in *orig_ents* whose ``header_line`` is absent from *result_ents*.

    Uses ``header_line`` as the stable identity — the compact prompt
    instructs the LLM to preserve ``### [category] timestamp`` header
    lines verbatim on surviving entries.
    """
    _after = {e.header_line for e in result_ents}
    return [e for e in orig_ents if e.header_line not in _after]


def _init_session_state(repo_root: str, svc, design_config) -> dict:
    """Initialize session/context state (P1-1 Phase C).

    Extracted from the ``_run_repl_impl`` monolith: /think·/model·/helper
    persistence, dev-model slots, terminal config seeding, and bracketed-paste
    setup. Returns a dict unpacked into locals by ``_run_repl_impl`` so the
    closure helpers keep their capture semantics.
    """
    global _completer_dev_models
    _session_tokens: dict = {"prompt": 0, "completion": 0, "cost": 0.0, "actual_cost": 0.0}
    _session_t0: float = time.monotonic()  # For session summary (_print_session_summary)
    # For /diff · /undo. "baseline" is the git path; "checkpoint" is the undo
    # point used outside a git work tree, where _git_baseline returns None and
    # this surface used to disappear entirely. At most one is ever set.
    _last_run_diff: dict = {"repo_root": None, "baseline": None, "checkpoint": None}
    _last_final_msg: str = ""  # /copy target — most recent final answer body
    # ── /think state persistence (survives CLI restart) ──
    _shared_state_path = os.path.join(repo_root, ".asicode", "config.json")
    _thinking_state_path = _shared_state_path
    # Per-terminal isolation: a TTY-attached terminal uses its own config file
    # (seeded from the shared one) so /model & /think stay isolated per terminal.
    _term_state_cfg = _terminal_config_path(repo_root)
    if _term_state_cfg:
        _seed_terminal_config(_term_state_cfg, _shared_state_path)
        _thinking_state_path = _term_state_cfg
    _thinking_state: Optional[bool] = None  # /think on/off (None=not explicitly set; provider decides)
    _reasoning_effort: Optional[str] = None  # /think high|max (None=provider default)
    # ── /code·/general mode — per-terminal independence (persisted to terminal config, not shared session) ──
    # Putting chat_mode in session (shared via _session_id=repo_root) would let one terminal's /code
    # propagate to other terminals via _adopt_from_disk. So, like /model·/think·/helper,
    # use per-terminal config (_thinking_state_path) as the single source of truth.
    _current_chat_mode: str = "code"
    # ── /helper: Context compression-only model (None → use main model) ──
    # helper uses a separate provider/model/client — so compression runs independently
    # so it operates independently of the main model's rate-limit/overload. Client is lazy-created and cached.
    _helper_provider_str: str = ""
    _helper_model_str: str = ""
    _helper_client = None  # Cached helper LLM client; None=not yet created
    # ── /model dev_N: per-subagent model slots (orchestrate mode) ──
    # {"1": (provider, model), "2": ...}; api_key is resolved from env at
    # orchestration time (never persisted). dev_1 is the canonical fallback
    # for unconfigured slots.
    _dev_models: dict[str, tuple[str, str]] = {}
    try:
        with open(_thinking_state_path, encoding="utf-8") as _tf:
            _tsd = json.load(_tf)
        if "thinking_state" in _tsd:
            _thinking_state = _tsd["thinking_state"]
        if "reasoning_effort" in _tsd:
            _reasoning_effort = _tsd["reasoning_effort"]
        if _tsd.get("helper_provider") and _tsd.get("helper_model"):
            _helper_provider_str = _tsd["helper_provider"]
            _helper_model_str = _tsd["helper_model"]
        if _tsd.get("chat_mode") in ("code", "general", "orchestrator"):
            _current_chat_mode = _tsd["chat_mode"]
        _dm = _tsd.get("dev_models") or {}
        if isinstance(_dm, dict):
            _dev_models = {
                str(k): (str(v[0]), str(v[1]))
                for k, v in _dm.items()
                if isinstance(v, (list, tuple)) and len(v) >= 2 and v[0] and v[1]
            }
    except (FileNotFoundError, json.JSONDecodeError):
        logging.getLogger(__name__).debug("no saved terminal state — defaults apply")
    # Point completer to the same dict reference so it reads current dev slot state.
    # Subsequent in-place changes to _dev_models (.pop / []=) are auto-reflected in completer.
    _completer_dev_models = _dev_models

    # ── Immediately reflect thinking_state loaded from config.json into design_config / svc ──
    design_config.thinking_mode = _thinking_state
    design_config.reasoning_effort = _reasoning_effort
    if hasattr(svc, "llm_service") and svc.llm_service is not None:
        svc.llm_service.thinking_mode = _thinking_state
        svc.llm_service.reasoning_effort = _reasoning_effort

    import atexit as _atexit
    _enable_bracketed_paste()
    _atexit.register(_disable_bracketed_paste)
    # Clean up baseline ref at session end — don't pollute user's stash list.
    _atexit.register(lambda rr=repo_root: _git(rr, "update-ref", "-d", "refs/asicode/baseline"))

    # Design chat worker that was ESC-interrupted and is finishing in background
    # (collected just before next input processing to guarantee session turn order)
    _pending_dc: Optional[dict] = None

    return {
        "session_tokens": _session_tokens,
        "session_t0": _session_t0,
        "last_run_diff": _last_run_diff,
        "last_final_msg": _last_final_msg,
        "shared_state_path": _shared_state_path,
        "thinking_state_path": _thinking_state_path,
        "term_state_cfg": _term_state_cfg,
        "thinking_state": _thinking_state,
        "reasoning_effort": _reasoning_effort,
        "current_chat_mode": _current_chat_mode,
        "helper_provider_str": _helper_provider_str,
        "helper_model_str": _helper_model_str,
        "helper_client": _helper_client,
        "dev_models": _dev_models,
        "pending_dc": _pending_dc,
    }


def _init_repl_engine(args: argparse.Namespace, repo_root: str) -> Optional[dict]:
    """Initialize the LLM engine/provider and design-chat core (P1-1 Phase B).

    Extracted from the ``_run_repl_impl`` monolith. Returns a dict of the
    objects the REPL loop needs; ``_run_repl_impl`` unpacks them into locals
    so the closure helpers keep their capture semantics. Returns ``None``
    when service creation failed (caller should return).
    """
    global _prompt_history_path, _REPO_ROOT
    _REPO_ROOT = repo_root
    _prompt_history_path = str(Path(repo_root) / ".asicode" / "cli_history")

    if not _load_prompt_toolkit():
        _print(
            "  ⚠ prompt_toolkit not found — REPL with basic input() (no history, no autocomplete).",
            _C["yellow"],
        )
        _print("    Activate venv →  source .venv/bin/activate", _C["muted"])
        try:
            answer = _collect_input("    Install now? (pip install prompt_toolkit) [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() in ("y", "yes") and _pip_install(["prompt_toolkit"], timeout=120):
            _restart_cli()

    _print_banner(repo_root)
    # ── Tier 2: design_insights accumulation nudge (non-blocking, threshold-only) ──
    try:
        from external_llm.agent.insights_manager import compute_stats, should_nudge
        _ins_nudge_fire, _ins_nudge_msg = should_nudge(compute_stats(repo_root))
        if _ins_nudge_fire:
            _parts = _ins_nudge_msg.split(" /insights", 1)
            if len(_parts) == 2:
                _print(_parts[0] + ".", _C["yellow"])
                _print(" /insights" + _parts[1], _C["muted"])
            else:
                _print(_ins_nudge_msg, _C["yellow"])
    except Exception:
        logging.getLogger(__name__).debug("insights nudge check failed", exc_info=True)

    _provider_str = args.provider or os.getenv("EXTERNAL_LLM_PROVIDER", "(env)")
    _model_str = args.model or os.getenv("EXTERNAL_LLM_MODEL", "(env)")

    # NOTE: per-terminal model restore already done in main() L8521-8547
    # (_terminal_config_path → _seed_terminal_config → args.provider/model applied).
    # run_repl is only called from main(), no need to re-execute here.

    # ── Dependency status + install prompt ──
    # ToolRegistry → RAGSearcher → VectorCacheManager immediately loads the embedding model,
    # so ask beforehand. This way, if the model is missing, we can show a download
    # y/n prompt instead of leaking a "fell back" warning.
    _print_dep_status(repo_root, no_deps_check=getattr(args, "no_deps_check", False))

    # ── Embedding model background warmup ──
    # ToolRegistry creation below goes through RAGSearcher → VectorCacheManager,
    # calling get_global_embedding_model() synchronously, blocking ~2s. Before that,
    # start background loading to parallelize with LLM service creation and design-chat setup.
    # Run in parallel. The loader uses lock+double-check, so when ToolRegistry
    # it reuses immediately; if in progress, waits briefly then gets the same
    # singleton — no duplicate loading or worsened latency.
    _kick_embedding_model_warmup()

    # ── Shared LLM service creation ──
    from external_llm.intelligent_service import create_intelligent_service_from_env
    svc = _retry_create_svc_with_api_key_prompt(
        create_intelligent_service_from_env,
        _provider_str if _provider_str != "(env)" else None,
        _model_str if _model_str != "(env)" else None,
        api_key=args.api_key or None, repo_root=repo_root,
    )
    if svc is None:
        _print("failed to initialize LLM service.", _C["red"])
        return None

    # ── Status bar sync ──
    # When provider/model was determined via "(env)" or interactive selection,
    # update to the actual created service values for correct /model display.
    _provider_str = getattr(svc, "provider", "") or _provider_str
    _model_str = getattr(svc, "model", "") or _model_str

    # ── Register context for /think /model autocompletion ──
    _set_completer_context(getattr(svc, "provider", ""), getattr(svc, "model", ""))

    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.design_session import DesignSessionManager

    # ── Design Chat init (uses DesignSessionManager, same as Web) ──
    design_config = AgentConfig(
        model_name=svc.model or "",
        user_checkpoint_callback=_cli_checkpoint_cb,
        unrestricted_read=True,   # trusted local CLI (also inherited by _orch_cfg via dataclasses.replace)
    )
    design_config.thinking_mode = None  # overridden by saved config below
    design_registry = ToolRegistry(repo_root, design_config)
    _session_mgr = DesignSessionManager(repo_root)
    import hashlib as _hashlib
    _session_id = f"cli-{_hashlib.md5(repo_root.encode(), usedforsecurity=False).hexdigest()[:16]}"
    _session_mgr.get_or_create(_session_id)

    # Background notification deferred output queue (prevents mixing with prompt bottom)
    _pending_notifications: list[str] = []
    _notifications_lock = threading.Lock()

    return {
        "svc": svc,
        "design_config": design_config,
        "design_registry": design_registry,
        "session_mgr": _session_mgr,
        "session_id": _session_id,
        "pending_notifications": _pending_notifications,
        "notifications_lock": _notifications_lock,
        "provider_str": _provider_str,
        "model_str": _model_str,
    }


def _stderr_spinner(stop: threading.Event, t0: float, msg: str) -> None:
    """Braille spinner on stderr while a synchronous op runs (0.1s cadence).

    Shared by ``/clear`` conversation compaction and insights compact — the
    frame loop + elapsed-time line were duplicated verbatim in both closures.
    The caller owns the stop event, thread object (for join), and the
    ``\r\033[K`` line-clear after stop.
    """
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _i = 0
    while not stop.wait(0.1):
        _el = time.monotonic() - t0
        sys.stderr.write(f"\r\033[K  {frames[_i % len(frames)]}  {msg}  {_el:0.1f}s")
        sys.stderr.flush()
        _i += 1
def _run_repl_impl(args: argparse.Namespace) -> None:
    """REPL main loop (implementation).

    Split out of the original monolithic ``run_repl`` (P1-1); the public
    ``run_repl`` entry point delegates here so ``main()`` callers and the
    CLI contract are unchanged.
    """
    global _last_input_was_auto
    repo_root = _resolve_repo_root(args.repo)
    # Startup stale-lock sweep (best-effort): reclaim orphaned lock files in
    # .asicode/locks left behind by retired code paths. Age + try-lock gated
    # inside sweep_stale_lock_files; failure must never block REPL startup.
    try:
        from external_llm.common.file_lock import sweep_stale_lock_files

        _swept = sweep_stale_lock_files(Path(repo_root) / ".asicode" / "locks")
        if _swept:
            logging.getLogger(__name__).debug(
                "stale-lock sweep removed %d file(s) from .asicode/locks", _swept
            )
    except Exception as _sweep_err:
        logging.getLogger(__name__).debug(
            "stale-lock sweep skipped: %s", _sweep_err
        )
    _init = _init_repl_engine(args, repo_root)
    if _init is None:
        return
    svc = _init["svc"]
    design_config = _init["design_config"]
    design_registry = _init["design_registry"]
    _session_mgr = _init["session_mgr"]
    _session_id = _init["session_id"]
    _pending_notifications = _init["pending_notifications"]
    _notifications_lock = _init["notifications_lock"]
    _provider_str = _init["provider_str"]
    _model_str = _init["model_str"]

    # Imports consumed by the main loop below (kept local like the original
    # monolith; not needed by _init_repl_engine).
    from external_llm.agent.design_chat_loop import DesignChatLoop, DesignChatResult
    from external_llm.client import LLMMessage

    def _deferred_notify(msg: str) -> None:
        """Queue a background-compress completion message."""
        with _notifications_lock:
            _pending_notifications.append(msg)

    def _drain_notifications() -> None:
        """Flush all queued deferred messages before showing the prompt."""
        with _notifications_lock:
            msgs = list(_pending_notifications)
            _pending_notifications.clear()
        for m in msgs:
            _print(m, "")

    def _update_terminal_config(updates: dict, drops: tuple = ()) -> None:
        """Read-modify-write the per-terminal config atomically — single source
        for ``/model``·``/think``·``/helper``·``/dev``·``/code`` persistence.

        Write is the shared crash-safe pipeline (``atomic_write_json``: sibling
        mkstemp tmp + fsync + atomic rename). When a TTY is attached the path
        is per-terminal (TTY-keyed) and the REPL handles these commands
        synchronously, so there is no concurrent writer. When stdin is *not* a
        TTY (piped/non-interactive) the path collapses to the shared
        ``.asicode/config.json`` and two concurrent piped-REPL processes
        against the same repo would share it — mkstemp gives each writer its
        own unique tmp file, so they can never corrupt each other's; the
        residual inter-process race degrades to a benign last-writer-wins lost
        update rather than JSON corruption.
        """
        try:
            with open(_thinking_state_path, encoding="utf-8") as _tf:
                _cfg = json.load(_tf)
        except (FileNotFoundError, json.JSONDecodeError):
            _cfg = {}
        _cfg.update(updates)
        for _k in drops:
            _cfg.pop(_k, None)
        # Same never-crash contract as the old bespoke tmp+replace: a write
        # failure (read-only mount, permission denied) must not kill the
        # command — the in-memory state already reflects the change.
        try:
            atomic_write_json(_thinking_state_path, _cfg)
        except OSError:
            logging.getLogger(__name__).debug("terminal config write failed", exc_info=True)

    def _persist_helper(provider: str, model: str) -> None:
        """Persist the ``/helper`` setting to config.json (restored on CLI restart)."""
        if provider and model:
            _update_terminal_config({"helper_provider": provider, "helper_model": model})
        else:
            _update_terminal_config({}, drops=("helper_provider", "helper_model"))

    def _persist_dev_models(dev_models: dict) -> None:
        """Persist ``/model dev_N`` slots (provider/model only; api_key resolved from env at orchestration time)."""
        if dev_models:
            _update_terminal_config(
                {"dev_models": {k: [v[0], v[1]] for k, v in dev_models.items()}}
            )
        else:
            _update_terminal_config({}, drops=("dev_models",))

    def _persist_chat_mode(mode: str) -> None:
        """Persist ``/code``·``/general`` mode to the per-terminal config.

        ``chat_mode`` is isolated per terminal — saved to the per-terminal
        config (``_thinking_state_path``) rather than the shared session file
        (which is keyed by repo_root and thus shared across terminals, so a
        ``/code`` in one terminal would otherwise bleed into others via
        ``_adopt_from_disk``). Same isolation path as ``/model``·``/think``·
        ``/helper``: every read/write flows through this one path.
        """
        _update_terminal_config({"chat_mode": mode})
    # Sentinel: marks that helper client creation already failed once.
    # Prevents re-attempting (and re-logging) on every compress call.
    _HELPER_CREATION_FAILED = object()

    def _get_compress_llm():
        """Return ``(client, model_name)`` for context compression.

        Uses the dedicated helper model when set (lazy-creating + caching its
        client); otherwise falls back to the main model's client/model. Keeping
        this in one place means both ``schedule_background_compress`` and
        ``/clear``'s ``compact_now`` route through the same resolution.
        """
        nonlocal _helper_client
        if _helper_model_str:
            if _helper_client is None:
                # Lazy-create + cache. If creation fails (e.g. missing API key),
                # log once and fall back to the main model rather than crash.
                _h = _create_llm_client_for(_helper_provider_str)
                if _h is not None:
                    _helper_client = _h
                else:
                    logging.getLogger(__name__).warning(
                        "helper client creation failed for %s — using main model",
                        _helper_provider_str,
                    )
                    _helper_client = _HELPER_CREATION_FAILED  # don't retry
            if _helper_client is not _HELPER_CREATION_FAILED:
                return _helper_client, _helper_model_str
        return svc.llm_service.client, (svc.model or "")

    def _compact_insights_interactive() -> bool:
        """design_insights LLM compact. Returns True on successful rewrite.

        Spins while the synchronous LLM call runs, then writes the
        result atomically. Never raises — logs + returns False on any failure.
        """
        from external_llm.agent.insights_manager import (
            COMPACT_BUDGET_BYTES,
            atomic_write_text,
            build_compact_messages,
            load_insights_file,
            parse_insights,
        )
        _ci_content = load_insights_file(repo_root)
        if not _ci_content.strip():
            _print("  nothing to compact.", _C["muted"])
            return False
        _ci_pre, _ci_ents = parse_insights(_ci_content)
        _ci_b_n, _ci_b_b = len(_ci_ents), len(_ci_content.encode("utf-8"))
        # Over-budget ⇒ instruct the compactor to MERGE+tighten to the budget
        # (see COMPACT_BUDGET_BYTES). Without this, an all-durable file is echoed
        # unchanged and the size nudge's warned condition is never remedied.
        _ci_over_budget = _ci_b_b > COMPACT_BUDGET_BYTES
        _ci_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        _ci_stop = threading.Event()
        _ci_t0 = time.monotonic()

        def _ci_spin() -> None:
            _stderr_spinner(_ci_stop, _ci_t0, "Compacting design insights…")

        _ci_spinner = threading.Thread(target=_ci_spin, daemon=True)
        if _ci_tty:
            _ci_spinner.start()
        else:
            _print("  Compacting design insights…", "dim")
        _ci_result = ""
        _ci_finish_reason = None  # for truncation detection
        # Suppress the provider logger during this synchronous LLM call, mirroring
        # the background-compress path (context_manager.compress_old_turns). Without
        # this, a persistent helper-model auth/quota failure surfaces as
        # ``ERROR DeepSeek authentication failed (401)`` on the terminal every time
        # insights compaction runs (and it re-runs each turn the file is over
        # budget). The filter attaches to the "external_llm" parent logger — the
        # one providers.py emits ``logger.error`` on — so the ERROR spam is
        # suppressed here; a single user-facing notice is routed via
        # ``_compress_failure_notice`` below instead. See context_manager.py:660-717
        # for the same pattern in the background path.
        from external_llm.agent.context_manager import (
            _compress_failure_notice,
            _SuppressInfoFilter,
        )
        _ci_llm_logger = logging.getLogger("external_llm")
        _ci_suppress = _SuppressInfoFilter()
        _ci_notice: Optional[str] = None
        try:
            from external_llm.client import LLMMessage
            _ci_client, _ci_model = _get_compress_llm()
            # DEBUG (not INFO): compaction runs automatically every turn the file
            # is over budget — INFO-level "insights compact: client/model" would
            # spam the terminal on every turn. Demoted to debug so it is silent
            # by default but recoverable via --debug.
            logging.getLogger(__name__).debug(
                "insights compact: client=%s, model=%s",
                type(_ci_client).__name__, _ci_model,
            )
            _ci_msgs = [
                LLMMessage(role=_m["role"], content=_m["content"])
                for _m in build_compact_messages(
                    _ci_content, budget_bytes=COMPACT_BUDGET_BYTES)
            ]
            # Compaction is deterministic curation, not a reasoning task: disable
            # thinking so the whole token budget goes to the rewritten file.
            # (Reasoning tokens were eating max_tokens and truncating the output —
            # finish_reason=length.) Size the budget to the input + slack so the
            # full rewrite always fits; the compacted file is never larger than
            # the input.
            # Token budget: delegate to module-level helper for testability.
            _ci_max_tokens = _size_compact_budget(_ci_content)
            # Reasoning-capable models (DeepSeek v4 on OpenCode Go / OpenRouter)
            # share the ``max_tokens`` budget between reasoning + content.  On
            # OpenCode the endpoint does NOT support ``max_completion_tokens``
            # (it is silently ignored), so the full reasoning trace + rewritten
            # file must fit within ``max_tokens``.  Observed reasoning traces for
            # a 6 KB insights file range from 5K to 15K tokens depending on
            # budget — the model generates MORE reasoning when given more room.
            # A generous fixed budget of 32k ensures the trace never exhausts
            # the budget while keeping the response time under ~60s.  On native
            # OpenAI (which routes to ``max_completion_tokens``) the extra headroom
            # is harmless — the model simply doesn't fill it.
            from external_llm.openai_client import _is_reasoning_model as _ci_is_reasoning
            if _ci_is_reasoning(_ci_model):
                _ci_max_tokens = max(_ci_max_tokens, 32000)
            _ci_llm_logger.addFilter(_ci_suppress)
            from external_llm.agent._response_utils import _TRUNCATION_REASONS
            from external_llm.agent._shared_utils import coerce_token_count
            # Retry on finish_reason=length/truncated with doubled budget (capped at 128k).
            # Reasoning models can consume the entire shared budget on reasoning trace,
            # leaving nothing for content — a larger budget gives reasoning+content
            # room to co-exist.
            _ci_pt_total = _ci_ct_total = _ci_crt_total = 0
            _ci_max_attempts = 2  # total attempts (initial + 1 retry on length)
            for _ci_attempt in range(_ci_max_attempts):
                _ci_resp = _ci_client.chat(
                    messages=_ci_msgs, model=_ci_model,
                    temperature=0.1, max_tokens=_ci_max_tokens,
                    thinking_mode=False,
                )
                _ci_result = (_ci_resp.content or "").strip()
                _ci_finish_reason = getattr(_ci_resp, "finish_reason", None)
                _ci_pt_total += coerce_token_count(getattr(_ci_resp, "prompt_tokens", 0))
                _ci_ct_total += coerce_token_count(getattr(_ci_resp, "completion_tokens", 0))
                _ci_crt_total += coerce_token_count(getattr(_ci_resp, "cache_read_input_tokens", 0))
                if _ci_finish_reason not in _TRUNCATION_REASONS:
                    break  # success or non-truncation failure
                # finish_reason=length/truncated: retry with doubled budget regardless of
                # whether content is partial or empty. Partial truncation is the
                # common case for non-reasoning models; empty content is the
                # common case for reasoning models (trace ate the budget).
                # Truncated with empty content — double budget for retry
                _ci_max_tokens = min(_ci_max_tokens * 2, 128000)
                logging.getLogger(__name__).debug(
                    "insights compact: truncated (length), retrying with "
                    "max_tokens=%d (attempt %d/%d)",
                    _ci_max_tokens, _ci_attempt + 1, _ci_max_attempts,
                )
        except Exception as _cie:
            logging.getLogger(__name__).warning(
                "insights compact LLM call failed: %s: %s",
                type(_cie).__name__, _cie)
            # Route ONE user-facing notice per failure-class per session — same
            # once-latch as the background path. ``_compress_failure_notice``
            # branches on provider exception types (LLMAuthenticationError etc.),
            # so pass the original exception through (do not broaden/re-wrap).
            # Keyed by a stable session-scoped id so this synchronous path and the
            # background path share the same latch (a notice from one suppresses
            # the duplicate from the other for the same failure class).
            try:
                _ci_notice = _compress_failure_notice(
                    "insights-compact", _ci_model if "_ci_model" in locals() else "",
                    _cie,
                    use_latch=False,  # interactive path: report every failure, not once
                    context="interactive",
                )
            except Exception as _cfn_err:
                # Last-resort: if notice generation itself fails, surface the
                # original error directly so the user can diagnose the problem.
                _ci_notice = (
                    f"⚠ insights compact failed: {type(_cie).__name__}: {_cie}"
                )
        except KeyboardInterrupt:
            # Ctrl+C during the blocking LLM call. ``except Exception`` above does
            # NOT catch BaseException subclasses (KeyboardInterrupt/SystemExit), so
            # without this the interrupt escaped the documented "never raises"
            # contract and crashed the whole REPL with a raw traceback. Route through
            # the same ``_ci_notice`` path: ``finally`` cleans up the spinner, then
            # the post-try block prints the cancel notice and returns False (the
            # gitignored file is left untouched — no partial write ever happened).
            _ci_notice = "⏹ insights compact cancelled."
        finally:
            _ci_llm_logger.removeFilter(_ci_suppress)
            _ci_stop.set()
            if _ci_tty:
                _ci_spinner.join(timeout=1.0)
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()
        # Log response details AFTER the suppress filter is removed so the
        # record actually reaches the handler.  This is the only reliable
        # place to capture what the LLM returned (or why it failed).
        if _ci_finish_reason:
            logging.getLogger(__name__).debug(
                "insights compact: finish_reason=%s, result_len=%d, model=%s",
                _ci_finish_reason, len(_ci_result),
                _ci_model if "_ci_model" in locals() else "?",
            )
        if _ci_notice:
            _print(f"  {_ci_notice}", _C["yellow"])
        if not _ci_result:
            if not _ci_notice:
                # The LLM call returned empty content without raising an exception.
                # Surface diagnostic info so the user can identify the root cause
                # (e.g. finish_reason=length, no choices, API returned empty body).
                _ci_diag = (
                    f"finish_reason={_ci_finish_reason}, "
                    f"model={_ci_model if '_ci_model' in locals() else '?'}"
                )
                _print(
                    f"  ✗ compaction failed — LLM returned empty content ({_ci_diag}). "
                    f"File unchanged.", _C["yellow"])
            return False
        # ── sanity gates (before overwriting the gitignored single source of truth) ──
        if _ci_finish_reason in _TRUNCATION_REASONS:
            _print("  ✗ compaction refused: LLM response truncated even after retry "
                   "(finish_reason=length/truncated). File unchanged.", _C["yellow"])
            return False
        _, _ci_a_ents = parse_insights(_ci_result)
        _ci_a_n, _ci_a_b = len(_ci_a_ents), len(_ci_result.encode("utf-8"))
        if _ci_b_n > 0 and _ci_a_n == 0:
            _print("  ✗ compaction refused: all entries were dropped "
                   f"({_ci_b_n}→0). File unchanged.", _C["yellow"])
            return False
        # No-op: the model echoed the same content without meaningful changes.
        # Delegating to the module-level helper for testability.
        _ci_is_noop = _insights_compact_is_noop(_ci_result, _ci_content, _ci_a_ents, _ci_ents)
        if _ci_is_noop:
            if _ci_over_budget:
                # The file was over budget yet the compactor couldn't reduce it
                # — exactly the "all durable invariants, nothing to compact" case.
                # The HARD budget backstop (enforce_budget_by_demotion) still
                # applies: demote oldest timestamped entries to the archive rather
                # than leaving the warned condition unaddressed.
                _ci_n_demoted = 0
                try:
                    from external_llm.agent.insights_manager import (
                        enforce_budget_by_demotion as _ci_ebd,
                    )
                    _ci_n_demoted, _ci_a_b = _ci_ebd(repo_root, COMPACT_BUDGET_BYTES)
                except Exception:
                    _ci_a_b = _ci_b_b
                if _ci_n_demoted:
                    _print(
                        f"  ✓ already compact — but over budget ({_ci_b_b:,} > "
                        f"{COMPACT_BUDGET_BYTES:,}); demoted {_ci_n_demoted} oldest "
                        f"entr{'y' if _ci_n_demoted == 1 else 'ies'} to "
                        f"design_insights_archive.md (not deleted; "
                        f"/insights archive list|restore). Active now {_ci_a_b:,} bytes.", "dim")
                else:
                    _print(
                        f"  ⚠ over budget ({_ci_b_b:,} > {COMPACT_BUDGET_BYTES:,} bytes) "
                        f"and no reduction possible (all entries protected principles). "
                        f"Manual /insights drop <n> or /insights prune <days> to remove.",
                        _C["yellow"])
            else:
                _print(f"  ✓ already compact — no changes ({_ci_b_n} entries · {_ci_b_b:,} bytes).", "dim")
            return True
        _print(f"  before: {_ci_b_n} entries · {_ci_b_b:,} bytes", _C["muted"])
        _print(f"  after:  {_ci_a_n} entries · {_ci_a_b:,} bytes", _C["muted"])
        # ── P2: partial entry loss guard ────────────────────────────────────────
        # Sanity gate above only blocks total deletion (N→0).  Partial loss
        # (>50%) on a non-over-budget path likely means the model hallucinated
        # away durable invariants.  Since the file is gitignored single-source-
        # of-truth and .bak is overwritten each compact, warn the user.
        if _ci_b_n > 0 and not _ci_over_budget and _ci_a_n < _ci_b_n / 2:
            _dropped = _dropped_entries(_ci_ents, _ci_a_ents)
            _dropped_hdrs = " ".join(
                f"[{e.category or '?'}]" for e in _dropped
            )
            _print(
                f"  ⚠ {len(_dropped)} entries dropped ({_ci_b_n}→{_ci_a_n}) "
                f"on non-over-budget compact — possible hallucinated deletion. "
                f"Dropped: {_dropped_hdrs}",
                _C["yellow"],
            )
        # Write a backup before overwriting (the file is gitignored so git can't recover it)
        from external_llm.agent.insights_manager import insights_path as _ci_ip
        _ci_path = _ci_ip(repo_root)
        _ci_bak = _ci_path + ".bak"
        try:
            import shutil
            shutil.copy2(_ci_path, _ci_bak)
        except OSError:
            logging.getLogger(__name__).debug("insights backup failed", exc_info=True)
        atomic_write_text(_ci_path, _ci_result)
        # ── Layer 1: HARD budget backstop ──────────────────────────────────────
        # The LLM compact is best-effort tightening. If the result is STILL over
        # budget (the steady state where every entry is a durable invariant —
        # nothing legitimate to DROP), mechanically demote the oldest timestamped
        # entries to design_insights_archive.md. NOT deletion: demoted entries stay
        # retrievable (/insights archive restore) and may auto-promote by relevance.
        _ci_n_demoted = 0
        # Gate on the post-write size, not the pre-write _ci_over_budget flag —
        # a file that started under budget can still be rewritten over budget
        # by the LLM (e.g. merged entries with expanded rationale), and that
        # case must hit the backstop too.
        if _ci_a_b > COMPACT_BUDGET_BYTES:
            try:
                from external_llm.agent.insights_manager import (
                    enforce_budget_by_demotion as _ci_ebd,
                )
                _ci_n_demoted, _ci_a_b = _ci_ebd(repo_root, COMPACT_BUDGET_BYTES)
                if _ci_n_demoted:
                    _, _ci_a_ents = parse_insights(load_insights_file(repo_root))
                    _ci_a_n = len(_ci_a_ents)
            except Exception:
                logging.getLogger(__name__).debug("budget backstop demotion failed", exc_info=True)
        if _ci_n_demoted:
            _ci_msg = (
                f"✓ design_insights compacted ({_ci_b_n}→{_ci_a_n} entries) + "
                f"{_ci_n_demoted} oldest entr{'y' if _ci_n_demoted == 1 else 'ies'} "
                f"demoted to design_insights_archive.md (not deleted; "
                f"/insights archive list|restore). Active now {_ci_a_b:,} bytes ≤ "
                f"{COMPACT_BUDGET_BYTES:,} budget."
            )
            # Pre-wrap so every wrapped line carries the same 2-space literal indent as
            # `before:`/`after:` above. Rich's auto-wrap preserves the leading whitespace
            # of the first line only — wrapped continuation lines would start at col 4
            # (_MarginIO left margin alone) instead of col 6 (margin + literal indent),
            # leaving `deleted`/`Active now` misaligned with `before`/`after`.
            _print(textwrap.fill(_ci_msg, width=max(40, _console_width),
                                 initial_indent="  ", subsequent_indent="  "), "dim")
        elif _ci_a_b > COMPACT_BUDGET_BYTES:
            # Backstop could not reach budget: every remaining entry is a protected
            # timestamp-less principle (never demoted). Report honestly.
            _ci_warn = (
                f"⚠ over budget ({_ci_a_b:,} > {COMPACT_BUDGET_BYTES:,} bytes); "
                f"all remaining entries are protected principles — no demotion "
                f"possible. Manual /insights drop <n> to remove."
            )
            _print(textwrap.fill(_ci_warn, width=max(40, _console_width),
                                 initial_indent="  ", subsequent_indent="  "), _C["yellow"])
        else:
            _print(f"  ✓ design_insights compacted ({_ci_b_n}→{_ci_a_n} entries).", "dim")
        # ── Token accounting (interactive compact) ───────────────────────────
        # The LLM call consumes real tokens but the interactive path never
        # accumulated them, making them invisible in the session cost line;
        # this log is the only record of what the compact consumed.
        try:
            _ci_pt = _ci_pt_total
            _ci_ct = _ci_ct_total
            _ci_crt = _ci_crt_total
            if _ci_pt or _ci_ct:
                logging.getLogger(__name__).debug(
                    "insights compact: tokens prompt=%d completion=%d cache_read=%d",
                    _ci_pt, _ci_ct, _ci_crt,
                )
        except Exception:
            logging.getLogger(__name__).debug("insights compact token accounting failed", exc_info=True)
        return True

    def _maybe_auto_compact_insights(chat_result) -> None:
        """After a design-chat turn ends: auto-compact if save_insight pushed the file over budget.

        Why after run() ends rather than inside the handler — the save_insight
        handler (design_chat_loop.py) runs inside an agent turn, so a
        synchronous LLM call there would become a re-entrant LLM call within
        the agent turn. Instead, this reuses the existing
        ``_compact_insights_interactive`` (the single source of truth
        encapsulating spinner · sanity gates · atomic write) here, outside the turn.

        Kept near-zero cost via a double gate:
          1. If ``chat_result.tool_calls_made`` has no ``save_insight``, return
             immediately (skips even the compute_stats file read).
          2. Even if present, skip if the file is still within budget
             (``COMPACT_BUDGET_BYTES``).
        save_insight is a design-chat-only tool, so this has no effect on
        regular code/general turns. Never raises — a compact failure must
        never block the turn.
        """
        if chat_result is None:
            return
        _names = {tc.get("tool") for tc in (getattr(chat_result, "tool_calls_made", None) or [])}
        if "save_insight" not in _names:
            return
        try:
            from external_llm.agent.insights_manager import (
                COMPACT_BUDGET_BYTES,
                compute_stats,
            )
            _stats = compute_stats(repo_root)
        except Exception:
            logging.getLogger(__name__).debug("auto-compact stats failed", exc_info=True)
            return
        if _stats.bytes_size <= COMPACT_BUDGET_BYTES:
            return
        _print("  💡 design_insights over budget after save_insight — auto-compacting…", _C["muted"])
        try:
            _compact_insights_interactive()
        except Exception:
            logging.getLogger(__name__).debug("auto-compact failed", exc_info=True)

    def _verify_insights_interactive() -> bool:
        """design_insights LLM verify against the codebase via a design-chat tool loop.

        Unlike ``_compact_insights_interactive`` (a single blind LLM call that only
        summarizes), this runs the design-chat agent with its read/write tools so it
        can confirm each entry against the *current* codebase before rewriting the
        file in place. The agent edits design_insights.md directly (edit_text /
        apply_patch), so there is no separate atomic write here. Returns True on a
        completed run; never raises.
        """
        from external_llm.agent.design_chat_loop import DesignChatLoop
        from external_llm.agent.insights_manager import (
            build_verify_messages,
            load_insights_file,
            parse_insights,
        )
        from external_llm.client import LLMMessage

        _v_content = load_insights_file(repo_root)
        if not _v_content.strip():
            _print("  nothing to verify.", _C["muted"])
            return False
        _, _v_ents = parse_insights(_v_content)
        _v_b_n = len(_v_ents)

        _v_cb = _ProgressPrinter(verbose=args.verbose)
        _v_loop = DesignChatLoop(svc.llm_service.client, design_registry, svc.model)
        _v_msgs = [
            LLMMessage(role=_m["role"], content=_m["content"])
            for _m in build_verify_messages(_v_content)
        ]
        _v_cb._start_spinner("Verifying design insights…")
        _v_result = None
        try:
            _v_result = _v_loop.respond(
                _v_msgs,
                stream_callback=_v_cb,
                max_tool_iterations=_cfg.counts.DESIGN_CHAT_MAX_TOOL_ITERATIONS,
                session_id=_session_id,
                session_mgr=_session_mgr,
                mode="code",
                thinking_mode=_thinking_state,
                reasoning_effort=_reasoning_effort,
            )
        except Exception as _ve:
            logging.getLogger(__name__).debug(
                "insights verify tool loop failed: %s", _ve)
        except KeyboardInterrupt:
            # Ctrl+C during the blocking agent tool loop. ``except Exception`` does
            # not catch BaseException subclasses, so without this the interrupt
            # escaped the documented "never raises" contract and crashed the REPL
            # with a raw traceback. The agent loop edits design_insights.md in place
            # mid-run, so a partial edit is possible — surface that caveat (matching
            # the tool-loop-error path) and return False, skipping the post-try
            # ``_v_result is None`` branch which would print a misleading
            # "verify failed (tool loop error)" for a user-initiated cancel.
            _print("  ⏹ insights verify cancelled (file may be partially edited).", _C["yellow"])
            return False
        finally:
            _v_cb._stop_spinner()

        # design_insights.md was modified directly by the loop, so re-read result file for stats
        _v_after = load_insights_file(repo_root)
        _, _v_a_ents = parse_insights(_v_after)
        _v_a_n = len(_v_a_ents)
        if _v_result is None or getattr(_v_result, "is_error", False):
            _print("  ✗ verify failed (tool loop error). File may be partially edited.", _C["yellow"])
            return False
        _print(f"  before: {_v_b_n} entries", _C["muted"])
        _print(f"  after:  {_v_a_n} entries", _C["muted"])
        _print(f"  ✓ design_insights verified ({_v_b_n}→{_v_a_n} entries).", "dim")
        _v_txt = (getattr(_v_result, "content", "") or "").strip()
        if _v_txt:
            _print(f"  {_v_txt[:300]}", _C["muted"])
        return True

    _st = _init_session_state(repo_root, svc, design_config)
    _session_tokens = _st["session_tokens"]
    _session_t0 = _st["session_t0"]
    _last_run_diff = _st["last_run_diff"]
    _last_final_msg = _st["last_final_msg"]
    _shared_state_path = _st["shared_state_path"]
    _thinking_state_path = _st["thinking_state_path"]
    _term_state_cfg = _st["term_state_cfg"]
    _thinking_state = _st["thinking_state"]
    _reasoning_effort = _st["reasoning_effort"]
    _current_chat_mode = _st["current_chat_mode"]
    _helper_provider_str = _st["helper_provider_str"]
    _helper_model_str = _st["helper_model_str"]
    _helper_client = _st["helper_client"]
    _dev_models = _st["dev_models"]
    _pending_dc = _st["pending_dc"]
    _input_truncated = False  # set by _dispatch_command (input-length limit); read after dispatch

    # Design chat worker that was ESC-interrupted and is finishing in background
    # (collected just before next input processing to guarantee session turn order)

    def _dispatch_command(user_input: str) -> tuple:
        """Slash-command & mode-switch dispatch.

        Returns ``(action, user_input)`` where ``action`` is one of:
          "break"    - end the session (caller must ``break`` the main loop)
          "continue" - handled; caller must ``continue`` the main loop
          "chat"     - not a command; proceed to the chat turn
        The returned ``user_input`` may have been rewritten (mode-switch
        prefixes stripped, truncation applied, inline task extracted).
        """
        nonlocal _current_chat_mode, _helper_client, _helper_model_str, _helper_provider_str, _model_str, _provider_str, _reasoning_effort, _thinking_state, _input_truncated, _pending_dc
        # ── Slash commands (utilities) ──
        _cmd_tok = user_input.strip().split(None, 1)
        _cmd_name = _SLASH_ALIASES.get(_cmd_tok[0].lower()) if _cmd_tok else None

        # Typo/unsupported slash command guard — prevents wasted LLM calls.
        # Allow paths like '/Users/...' or sentences starting with hangul through,
        # only treat as command typo when it's a single '/' + short ASCII alpha token.
        if _cmd_name is None and _cmd_tok:
            _tok0 = _cmd_tok[0]
            _word = _tok0[1:]
            if (_tok0.startswith("/") and 0 < len(_word) <= 15
                    and _word.isascii() and _word.isalpha()):
                import difflib as _difflib
                _sugg = _difflib.get_close_matches(
                    _tok0.lower(), list(_SLASH_ALIASES), n=1, cutoff=0.6)
                _hint = f" — did you mean {_sugg[0]}?" if _sugg else ""
                _print(f"  unknown command: {_tok0}{_hint}  (/help for commands)", _C["yellow"])
                return ("continue", user_input)

        if user_input.lower() in (":q", "quit", "exit") or _cmd_name == "/quit":
            # ── Tier 3: suggest cleanup right before session end if design_insights exceeds threshold ──
            try:
                from external_llm.agent.insights_manager import compute_stats, should_nudge
                _end_fire, _end_msg = should_nudge(compute_stats(repo_root))
                if _end_fire:
                    _parts = _end_msg.split(" /insights", 1)
                    if len(_parts) == 2:
                        _print(_parts[0] + ".", _C["yellow"])
                        _print(" /insights" + _parts[1], _C["muted"])
                    else:
                        _print(_end_msg, _C["yellow"])
                    try:
                        _end_ans = input("  compact now? (y/N) ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        _end_ans = ""
                    if _end_ans in ("y", "yes"):
                        _compact_insights_interactive()
            except Exception:
                logging.getLogger(__name__).debug("session-end insights nudge failed", exc_info=True)

            _print_session_summary(_session_tokens, _session_t0)
            _print("session ended.", "")
            return ("break", user_input)
        if _cmd_name in ("/help", "/diff", "/undo", "/status", "/model", "/helper", "/clear", "/insights", "/copy"):
            _cur_mode = _current_chat_mode
            if _cmd_name == "/help":
                _render_help()
            elif _cmd_name == "/diff":
                if not (_last_run_diff.get("baseline") and
                        _render_run_diff(_last_run_diff["repo_root"], _last_run_diff["baseline"])):
                    _print("  no changes recorded yet.", _C["muted"])
            elif _cmd_name == "/undo":
                _u_base = _last_run_diff.get("baseline")
                _u_root = _last_run_diff.get("repo_root") or repo_root
                _u_cp = _last_run_diff.get("checkpoint")
                _u_stats = _run_changed_stats(_u_root, _u_base) if _u_base else []
                if not _u_stats and _u_cp:
                    # Checkpoint path (non-git directory). No per-file +N -M is
                    # available — there is no baseline tree to diff against —
                    # so the confirmation lists the paths instead, including
                    # the ones restore() will DELETE because the run created
                    # them.
                    _cp_files = _checkpoint_changed_files(_u_root, _u_cp)
                    if not _cp_files:
                        _print("  nothing to undo — no run changes recorded.", _C["muted"])
                    else:
                        for _p in _cp_files:
                            _print(f"  · {_p}", _C["text"])
                        _print("  ▲ reverts these files to their pre-run state — "
                               "files the run created are deleted, and manual edits "
                               "made after the run are lost too.", _C["yellow"])
                        try:
                            _u_ans = input(f"  revert {len(_cp_files)} file(s)? (y/N) ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            _u_ans = ""
                        if _u_ans in ("y", "yes"):
                            if _undo_via_checkpoint(_u_root, _u_cp):
                                _print(f"  ✓ reverted {len(_cp_files)} file(s)", _C["green"])
                                _last_run_diff["checkpoint"] = None
                            else:
                                _print("  ✗ revert failed — see logs for the "
                                       "files that could not be restored.", _C["red"])
                        else:
                            _print("  cancelled.", _C["muted"])
                elif not _u_stats:
                    _print("  nothing to undo — no run changes recorded.", _C["muted"])
                else:
                    # Use actual file count (unlimited) for the prompt, not max_files=20 from stats
                    _u_total = len(_changed_files_since(_u_root, _u_base)) if _u_base else 0
                    _print_run_change_summary(_u_root, _u_base)
                    _print("  ▲ reverts these files to their pre-run state — "
                           "manual edits made after the run are lost too.", _C["yellow"])
                    try:
                        _u_ans = input(f"  revert {_u_total} file(s)? (y/N) ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        _u_ans = ""
                    if _u_ans in ("y", "yes"):
                        _undone, _u_failed = _undo_run_changes(_u_root, _u_base)
                        for _p in _undone:
                            _print(f"  ✓ reverted {_p}", _C["green"])
                        for _p in _u_failed:
                            _print(f"  ✗ failed to revert {_p}", _C["red"])
                        if not _u_failed:
                            # If all reverted, also clear /diff·/undo re-execution targets
                            _last_run_diff["baseline"] = None
                    else:
                        _print("  cancelled.", _C["muted"])
            elif _cmd_name == "/model":
                _model_arg = _cmd_tok[1].strip() if len(_cmd_tok) > 1 else ""
                if not _model_arg:
                    # ── No args: show current model + available model list ──
                    _print(f"  current: {_provider_str} / {_model_str}", _C["text"])
                    if _term_state_cfg:
                        _tty_short = os.path.basename(_term_state_cfg)[:-5]  # strip .json
                        _print(f"  isolated to this terminal: {_tty_short}", _C["muted"])
                    if _helper_model_str:
                        _print(f"  helper:  {_helper_provider_str} / {_helper_model_str}  (context compression)", _C["text"])
                    if _dev_models:
                        _print("  sub-agent slots (orchestrate):", _C["sky"])
                        for _dk in sorted(_dev_models, key=lambda x: int(x) if x.isdigit() else 999):
                            _dp, _dm = _dev_models[_dk]
                            _fb = "  ← fallback" if _dk == min(
                                (k for k in _dev_models if k.isdigit()), key=int, default=""
                            ) else ""
                            _print(f"    dev_{_dk}: {_dp} / {_dm}{_fb}", _C["text"])
                    _print("", "")
                    _print("  available models:", _C["sky"])
                    for _prov, _models in _KNOWN_MODELS.items():
                        _print(f"    {_prov}:", _C["muted"])
                        for _m in _models:
                            _mark = " ←" if _prov == _provider_str and _m == _model_str else ""
                            _print(f"      {_m}{_mark}", _C["text"])
                    # ── ollama local model list ──
                    _ollama_names = _get_ollama_models(timeout=5)
                    if _ollama_names:
                        _print("    ollama (local):", _C["muted"])
                        for _m in _ollama_names:
                            _mark = " ←" if _provider_str == "ollama" and _m == _model_str else ""
                            _print(f"      {_m}{_mark}", _C["text"])
                    _print("", "")
                    _print("  usage: /model <name>  or  /model <provider>/<name>", _C["muted"])
                    _print("  e.g.:  /model gpt-4o  or  /model anthropic/claude-sonnet-4-6", _C["muted"])
                    _print("  sub-agent: /model dev_N <model>  (orchestrate; dev_1 = fallback default)", _C["muted"])
                else:
                    # ── /model dev_N [model|off]: per-subagent model slot ──
                    # Sets the model for the Nth spawned sub-agent in /orchestrate.
                    # dev_1 is the canonical fallback for unconfigured slots.
                    _dev_head = _model_arg.split(None, 1)
                    if _dev_head and _dev_head[0].lower().startswith("dev_"):
                        _slot_str = _dev_head[0][4:]
                        if not _slot_str.isdigit() or int(_slot_str) < 1:
                            _print("  invalid slot: use /model dev_N <model>  (N≥1, e.g. /model dev_1 qwen2.5-coder:3b)", _C["yellow"])
                            return ("continue", user_input)
                        _slot_key = str(int(_slot_str))
                        _dev_rest = _dev_head[1].strip() if len(_dev_head) > 1 else ""
                        if not _dev_rest:
                            # show current slot config
                            if _slot_key in _dev_models:
                                _p, _m = _dev_models[_slot_key]
                                _print(f"  dev_{_slot_key}: {_p} / {_m}", _C["text"])
                            else:
                                _print(f"  dev_{_slot_key}: (not set — falls back to lowest configured slot, else orchestrator model)", _C["muted"])
                            _print("  usage: /model dev_N <model>  or  /model dev_N off", _C["muted"])
                            return ("continue", user_input)
                        if _dev_rest.lower() == "off":
                            _dev_models.pop(_slot_key, None)
                            _persist_dev_models(_dev_models)
                            _print(f"  dev_{_slot_key} cleared.", _C["green"])
                            return ("continue", user_input)
                        _resolved_dev = _resolve_model_interactive(
                            _dev_rest, usage_hint="/model dev_N"
                        )
                        if _resolved_dev is None:
                            return ("continue", user_input)
                        _dv_provider, _dv_model = _resolved_dev
                        _dev_models[_slot_key] = (_dv_provider, _dv_model)
                        _persist_dev_models(_dev_models)
                        _print(f"  dev_{_slot_key} set: {_dv_provider} / {_dv_model}", _C["green"])
                        if _slot_key != "1" and "1" not in _dev_models:
                            _print("  (tip: unconfigured slots fall back to the lowest configured slot; set dev_1 to define a default)", _C["muted"])
                        return ("continue", user_input)
                    _resolved = _resolve_model_interactive(_model_arg, usage_hint="/model")
                    if _resolved is None:
                        return ("continue", user_input)
                    _new_provider, _new_model = _resolved
                    # Warn if model is not in _KNOWN_MODELS (typo prevention)
                    _known_models_for_provider = _KNOWN_MODELS.get(_new_provider, [])
                    if _known_models_for_provider and _new_model not in _known_models_for_provider:
                        _print(f"  ⚠ model '{_new_model}' is not in the known list for {_new_provider}", _C["yellow"])
                        _print(f"    (possible typo? known models: {', '.join(_known_models_for_provider)})", _C["muted"])
                    if _new_model == _model_str and _new_provider == _provider_str:
                        _print(f"  already using {_new_provider} / {_new_model}", _C["muted"])
                    else:
                        _old_provider, _old_model = _provider_str, _model_str
                        _provider_str, _model_str = _new_provider, _new_model
                        svc.model = _new_model
                        svc.llm_service.model = _new_model
                        design_config.model_name = _new_model
                        if _new_provider != _old_provider:
                            svc.provider = _new_provider
                            svc.llm_service.provider = _new_provider
                            _ak_var = _API_KEY_ENV_MAP.get(_new_provider.lower())
                            _ak = os.getenv(_ak_var, "") if _ak_var else ""
                            # If API key missing, show hint + input prompt
                            if not _ak and _new_provider.lower() != "ollama":
                                if _ak_var:
                                    _print(
                                        f"  {_ak_var} not set in environment.",
                                        _C["yellow"],
                                    )
                                else:
                                    _print(
                                        f"  unknown provider '{_new_provider}' — no API key env var known.",
                                        _C["yellow"],
                                    )
                                _print(
                                    "  enter API key (or press Enter to cancel):",
                                    _C["muted"],
                                )
                                try:
                                    _input_key = _collect_input("  API key: ").strip()
                                except (EOFError, KeyboardInterrupt):
                                    _input_key = ""
                                if _input_key:
                                    _ak = _input_key
                                    # Persist the key so it survives a re-switch in the same
                                    # session AND a CLI restart — mirrors the initial-setup
                                    # path (which sets os.environ + writes .env). Without this
                                    # the key lives only in the local `_ak` var and the prompt
                                    # re-appears on every /model switch / restart.
                                    if _ak_var:
                                        os.environ[_ak_var] = _input_key
                                        if repo_root:
                                            _save_key_to_dotenv(repo_root, _ak_var, _input_key)
                                    _print(f"  using inline API key for {_new_provider}", _C["green"])
                                else:
                                    # Cancel: restore original provider/model
                                    _provider_str, _model_str = _old_provider, _old_model
                                    svc.model = _old_model
                                    svc.llm_service.model = _old_model
                                    design_config.model_name = _old_model
                                    svc.provider = _old_provider
                                    svc.llm_service.provider = _old_provider
                                    _print("  cancelled — no API key provided.", _C["muted"])
                                    _set_completer_context(_old_provider, _old_model)
                                    return ("continue", user_input)
                            try:
                                from external_llm.client import create_llm_client as _create_llm
                                from external_llm.client import resolve_provider_base_url
                                svc.llm_service.client = _create_llm(
                                    provider=_new_provider,
                                    api_key=_ak,
                                    base_url=resolve_provider_base_url(_new_provider),
                                )
                            except Exception as _ce:
                                _print(f"  ✗ failed to create LLM client: {_ce}", _C["red"])
                                # Rollback: restore original provider/model
                                _provider_str, _model_str = _old_provider, _old_model
                                svc.model = _old_model
                                svc.llm_service.model = _old_model
                                design_config.model_name = _old_model
                                svc.provider = _old_provider
                                svc.llm_service.provider = _old_provider
                                _set_completer_context(_old_provider, _old_model)
                                return ("continue", user_input)
                        _print(
                            f"  model switched: {_old_provider} / {_old_model} → {_provider_str} / {_model_str}",
                            _C["green"],
                        )
                        # ── Context-window / eviction visibility ──
                        # Show the resolved context limit so the user can see whether
                        # eviction is active for this model. Models not in
                        # _CONTEXT_LIMITS fall back to 1M (eviction effectively off —
                        # the hard-cap trim is the only backstop); registered models
                        # show their real window so eviction's 75% gate is reachable.
                        try:
                            from external_llm.agent.context_budget import _resolve_context_limit as _rcl
                            _ctx = _rcl(_model_str)
                            if _ctx >= 1_000_000:
                                _print(
                                    f"  context: {_ctx:,} tokens (eviction: off — unknown model, 1M fallback; hard-cap only)",
                                    _C["muted"],
                                )
                            else:
                                _print(
                                    f"  context: {_ctx:,} tokens (eviction: active at ~{_ctx * 3 // 4:,} tokens)",
                                    _C["muted"],
                                )
                        except Exception:
                            logging.getLogger(__name__).debug("context-limit resolve failed", exc_info=True)
                        # ── Autocomplete context sync ──
                        _set_completer_context(_provider_str, _model_str)
                        # ── Model state persistence (restored on CLI restart) ──
                        _update_terminal_config(
                            {"provider": _provider_str, "model": _model_str}
                        )
            elif _cmd_name == "/helper":
                # Context compression-only model. When set, compression (schedule_background_
                # compress / /clear's compact_now) uses this model's
                # dedicated client — independent of the main model's rate-limit/overload.
                _helper_arg = _cmd_tok[1].strip() if len(_cmd_tok) > 1 else ""
                if not _helper_arg:
                    # ── No args: show current helper status ──
                    if _helper_model_str:
                        _print(f"  compression helper: {_helper_provider_str} / {_helper_model_str}", _C["text"])
                        _print("  (use '/helper off' to fall back to the main model)", _C["muted"])
                    else:
                        _print(f"  compression helper: (none — using main model {_provider_str} / {_model_str})", _C["muted"])
                    _print("", "")
                    _print("  usage: /helper <name>  or  /helper off", _C["muted"])
                    _print("  e.g.:  /helper gpt-4o-mini  or  /helper deepseek/deepseek-chat", _C["muted"])
                elif _helper_arg.lower() == "off":
                    _helper_provider_str = ""
                    _helper_model_str = ""
                    _helper_client = None
                    _persist_helper(_helper_provider_str, _helper_model_str)
                    _print(f"  compression helper cleared — using main model {_provider_str} / {_model_str}", _C["green"])
                else:
                    # ── Model name resolution: provider/name or autocomplete ──
                    _resolved = _resolve_model_interactive(
                        _helper_arg, usage_hint="/helper"
                    )
                    if _resolved is None:
                        return ("continue", user_input)
                    _new_provider, _new_model = _resolved
                    # ── Dedicated client creation (separate from main svc) ──
                    _h_client = _create_llm_client_for(_new_provider)
                    if _h_client is None:
                        _print(f"  ✗ failed to create helper client for {_new_provider} (missing API key?)", _C["red"])
                        return ("continue", user_input)
                    _helper_provider_str = _new_provider
                    _helper_model_str = _new_model
                    _helper_client = _h_client
                    _persist_helper(_helper_provider_str, _helper_model_str)
                    _print(
                        f"  compression helper set: {_helper_provider_str} / {_helper_model_str}",
                        _C["green"],
                    )
                    _print("  (context compression will use this model instead of the main model)", _C["muted"])
            elif _cmd_name == "/status":
                _helper_str = f"{_helper_provider_str} / {_helper_model_str}" if _helper_model_str else ""
                _render_status(repo_root, _provider_str, _model_str, _cur_mode, _session_tokens, _thinking_state, _reasoning_effort, helper=_helper_str)
            elif _cmd_name == "/copy":
                if not _last_final_msg.strip():
                    _print("  no final message to copy yet.", _C["muted"])
                else:
                    _via = _copy_to_clipboard(_last_final_msg)
                    if _via:
                        _print(
                            f"  ✓ copied final message to clipboard "
                            f"({len(_last_final_msg)} chars · {_via})",
                            _C.get("green", "green"),
                        )
                    else:
                        _print(
                            "  ✗ clipboard copy failed — no pbcopy/xclip/wl-copy "
                            "and terminal lacks OSC 52 support.",
                            _C["yellow"],
                        )
            elif _cmd_name == "/clear":
                # 1. Conversation context compression — merge up to recent raw turns into compressed_summary,
                #    clear raw window (recent_keep=0). Old conversations stay in archive,
                #    searchable via search_design_history. Must be reflected in next turn's context
                #    immediately, so run synchronously.
                _compacted = False
                try:
                    _ds_clear = _session_mgr.get_or_create(_session_id)
                    if getattr(_ds_clear, "turns", None):
                        # Compression is a sync LLM call, can take seconds — prevent screen freeze by
                        # running the work on main thread and dynamic spinner on a separate thread.
                        _clr_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
                        _clr_stop = threading.Event()
                        _clr_t0 = time.monotonic()

                        def _clr_spin() -> None:
                            _stderr_spinner(_clr_stop, _clr_t0, "Compacting conversation into summary…")

                        _clr_spinner = threading.Thread(target=_clr_spin, daemon=True)
                        if _clr_tty:
                            _clr_spinner.start()
                        else:
                            _print("  Compacting conversation into summary…", "dim")
                        try:
                            _clr_client, _clr_model = _get_compress_llm()
                            _compacted = _session_mgr.compact_now(
                                _ds_clear, _clr_model,
                                _clr_client, recent_keep=0,
                            )
                        finally:
                            _clr_stop.set()
                            if _clr_tty:
                                _clr_spinner.join(timeout=1.0)
                                sys.stderr.write("\r\033[K")
                                sys.stderr.flush()
                except Exception as _ce:
                    logging.getLogger(__name__).debug(
                        "/clear compaction failed: %s", _ce)
                # 2. Screen clear + banner
                if sys.stdout.isatty():
                    sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
                    sys.stdout.flush()
                if asi._margin_stderr:
                    asi._margin_stderr.reset_bol()
                _print_banner(repo_root)
                if _compacted:
                    _print("  ✓ Conversation compacted into summary — context window cleared.", "dim")
            elif _cmd_name == "/insights":
                # design_insights.md management: list / compact(LLM) / drop <n>.
                # insights_manager provides pure functions (parsing/stats/thresholds/atomic writes),
                # LLM calls are delegated to the _compact_insights_interactive helper above.
                from external_llm.agent.insights_manager import (
                    COMPACT_BUDGET_BYTES,
                    InsightEntry,
                    atomic_write_text,
                    compute_stats,
                    drop_entry,
                    entry_age_days,
                    load_insights_file,
                    parse_insights,
                    select_entries_older_than,
                    serialize_insights,
                )
                _ins_rest = _cmd_tok[1].strip() if len(_cmd_tok) > 1 else ""
                _ins_sub = (_ins_rest.split(None, 1)[0].lower() if _ins_rest else "list")
                _ins_path = os.path.join(repo_root, ".asicode", "design_insights.md")

                if _ins_sub in ("", "list", "ls"):
                    _ins_stats = compute_stats(repo_root)
                    if not _ins_stats.exists:
                        _print("  no design_insights file yet.", _C["muted"])
                    else:
                        _ins_over = _ins_stats.bytes_size > COMPACT_BUDGET_BYTES
                        _ins_budget_s = (
                            f"  ⚠ over budget ({COMPACT_BUDGET_BYTES:,} bytes) — /insights compact will merge+tighten"
                            if _ins_over else "")
                        _print(
                            f"  design_insights: {_ins_stats.count} entries · "
                            f"{_ins_stats.bytes_size:,} bytes"
                            + ("  ⚠ over budget" if _ins_over else ""),
                            _C["yellow"] if _ins_over else _C["text"])
                        if _ins_over:
                            _print(_ins_budget_s, _C["yellow"])
                        _, _ins_ents = parse_insights(load_insights_file(repo_root))
                        if not _ins_ents:
                            _print("  (preamble only — no ### entries)", _C["muted"])
                        else:
                            for _ii, _ie in enumerate(_ins_ents, 1):
                                _icat = f"[{_ie.category}]" if _ie.category else "[—]"
                                _iage = entry_age_days(_ie)
                                _iage_s = f"{int(_iage)}d" if _iage is not None else "—"
                                _iprev = (_ie.body.strip().split("\n", 1)[0])[:64]
                                _print(f"    {_ii}. {_icat} ({_iage_s}) {_iprev}", _C["muted"])
                        # Archive count (two-tier): archived entries are durable
                        # insights demoted to keep this file within budget.
                        try:
                            from external_llm.agent.insights_manager import load_archive_file
                            _ins_arch = parse_insights(load_archive_file(repo_root))[1]
                            if _ins_arch:
                                _print(f"  archive: {len(_ins_arch)} demoted entries (not deleted; /insights archive list|restore <n>).", _C["muted"])
                        except Exception:
                            logging.getLogger(__name__).debug("insights archive count failed", exc_info=True)
                        _print("  /insights {list|archive|compact|verify|prune <days>|drop <n>|edit <n> <body>}", _C["muted"])

                elif _ins_sub == "drop":
                    _ins_rest_toks = _ins_rest.split()
                    if len(_ins_rest_toks) < 2:
                        _print("  usage: /insights drop <n>  (n from /insights list)", _C["yellow"])
                    else:
                        try:
                            _ins_idx = int(_ins_rest_toks[1])
                        except ValueError:
                            _ins_idx = 0
                        _ins_content = load_insights_file(repo_root)
                        _ins_pre, _ins_ents = parse_insights(_ins_content)
                        if not (1 <= _ins_idx <= len(_ins_ents)):
                            _print(f"  no entry #{_ins_idx}  (valid: 1-{len(_ins_ents)})", _C["yellow"])
                        else:
                            _ins_dropped = _ins_ents[_ins_idx - 1].category or "—"
                            atomic_write_text(_ins_path, serialize_insights(_ins_pre, drop_entry(_ins_ents, _ins_idx)))
                            _print(f"  ✓ dropped #{_ins_idx} [{_ins_dropped}]", "dim")

                elif _ins_sub == "edit":
                    _ins_rest_toks = _ins_rest.split(None, 2)
                    if len(_ins_rest_toks) < 3:
                        _print("  usage: /insights edit <n> <new_body>  (n from /insights list)", _C["yellow"])
                    else:
                        try:
                            _ins_idx = int(_ins_rest_toks[1])
                        except ValueError:
                            _ins_idx = 0
                        _ins_new_body = _ins_rest_toks[2]
                        _ins_content = load_insights_file(repo_root)
                        _ins_pre, _ins_ents = parse_insights(_ins_content)
                        if not (1 <= _ins_idx <= len(_ins_ents)):
                            _print(f"  no entry #{_ins_idx}  (valid: 1-{len(_ins_ents)})", _C["yellow"])
                        else:
                            _ins_old = _ins_ents[_ins_idx - 1]
                            # Preserve the original header line verbatim (with its \n)
                            # so the round-trip invariant holds.  body= is NOT passed —
                            # it is a @property, not a dataclass field.
                            _hdr = _ins_old.lines[0] if _ins_old.lines else _ins_old.header_line + "\n"
                            if not _hdr.endswith("\n"):
                                _hdr += "\n"
                            _ins_new_ent = InsightEntry(
                                lines=[_hdr, _ins_new_body.rstrip("\n") + "\n"],
                                header_line=_ins_old.header_line,
                                category=_ins_old.category,
                            )
                            _ins_ents[_ins_idx - 1] = _ins_new_ent
                            atomic_write_text(_ins_path, serialize_insights(_ins_pre, _ins_ents))
                            _print(f"  ✓ edited #{_ins_idx} [{_ins_old.category or '—'}]", "dim")

                elif _ins_sub == "prune":
                    # Age-based pruning: list entries older than <days>, confirm,
                    # then drop them. NEVER auto-deletes — entries are shown and
                    # require explicit y/N. Untimestamped entries are never
                    # selected (select_entries_older_than guards that), so
                    # hand-written principles are safe.
                    _ins_rest_toks = _ins_rest.split()
                    _ins_days = None
                    if len(_ins_rest_toks) >= 2:
                        try:
                            _ins_days = int(_ins_rest_toks[1])
                        except ValueError:
                            _ins_days = None
                    if _ins_days is None or _ins_days < 0:
                        _print("  usage: /insights prune <days>  (drop entries older than <days>, with confirmation)", _C["yellow"])
                    else:
                        _ins_content = load_insights_file(repo_root)
                        _ins_pre, _ins_ents = parse_insights(_ins_content)
                        _ins_old_idx = select_entries_older_than(_ins_ents, _ins_days)
                        if not _ins_old_idx:
                            _print(f"  no entries older than {_ins_days} days.", _C["muted"])
                        else:
                            _print(f"  {len(_ins_old_idx)} entr{'y' if len(_ins_old_idx)==1 else 'ies'} older than {_ins_days} days:", _C["yellow"])
                            for _pi in _ins_old_idx:
                                _pe = _ins_ents[_pi - 1]
                                _page = entry_age_days(_pe)
                                _page_s = f"{int(_page)}d" if _page is not None else "—"
                                _pcat = f"[{_pe.category}]" if _pe.category else "[—]"
                                _pprev = (_pe.body.strip().split("\n", 1)[0])[:60]
                                _print(f"    {_pi}. {_pcat} ({_page_s}) {_pprev}", _C["muted"])
                            _ans = _collect_input(f"    Drop these {len(_ins_old_idx)} entries? [y/N] ").strip().lower()
                            if _ans in ("y", "yes"):
                                _ins_drop_set = set(_ins_old_idx)
                                _ins_keep = [_e for _i, _e in enumerate(_ins_ents, 1) if _i not in _ins_drop_set]
                                # Backup before overwrite (the file is gitignored)
                                try:
                                    import shutil
                                    shutil.copy2(_ins_path, _ins_path + ".bak")
                                except OSError:
                                    logging.getLogger(__name__).debug("insights prune backup failed", exc_info=True)
                                atomic_write_text(_ins_path, serialize_insights(_ins_pre, _ins_keep))
                                _print(f"  ✓ pruned {len(_ins_old_idx)} entries older than {_ins_days} days ({len(_ins_ents)}→{len(_ins_keep)}).", "dim")
                            else:
                                _print("  prune cancelled. File unchanged.", _C["muted"])

                elif _ins_sub == "archive":
                    _handle_insights_archive(repo_root, _ins_rest)
                elif _ins_sub == "compact":
                    _compact_insights_interactive()

                elif _ins_sub == "verify":
                    _verify_insights_interactive()

                elif _ins_sub not in _INSIGHTS_SUBCOMMANDS and _ins_sub != "ls":
                    _print(f"  unknown subcommand '{_ins_sub}' — use: /insights [{'|'.join(_INSIGHTS_SUBCOMMANDS)}]", _C["yellow"])
                    return ("continue", user_input)

            elif _cmd_name == "/failure-patterns":
                # Show or modify the persistent failure-pattern store.
                _fps_rest = _cmd_tok[1].strip() if len(_cmd_tok) > 1 else ""
                _fps_cmd = _fps_rest.split(None, 1)[0].lower() if _fps_rest else ""
                try:
                    from external_llm.agent.failure_pattern_store import get_store
                    _fps = get_store(repo_root)

                    if _fps_cmd == "clear":
                        _fps_size = _fps.store_size()
                        _fps_size_str = (
                            f" ({_fps_size} pattern{'' if _fps_size == 1 else 's'})"
                            if _fps_size > 0 else ""
                        )
                        _ans = _collect_input(f"    Clear the failure-pattern store{_fps_size_str}? [y/N] ").strip().lower()
                        if _ans in ("y", "yes"):
                            _fps.clear()
                            _print("  ✓ failure-pattern store cleared.", "dim")
                        else:
                            _print("  clear cancelled.", _C["muted"])

                    elif _fps_cmd == "prune":
                        _fps_prune_toks = _fps_rest.split()
                        _fps_prune_threshold = 1.0
                        if len(_fps_prune_toks) >= 2:
                            try:
                                _fps_prune_threshold = float(_fps_prune_toks[1])
                            except ValueError:
                                _print("  usage: /failure-patterns prune [threshold]  (threshold default: 1.0)", _C["yellow"])
                                return ("continue", user_input)
                        _fps_prune_count = _fps.prune(threshold=_fps_prune_threshold)
                        if _fps_prune_count > 0:
                            _print(f"  ✓ pruned {_fps_prune_count} pattern{'' if _fps_prune_count == 1 else 's'} (effective < {_fps_prune_threshold})", "dim")
                        else:
                            _print(f"  no patterns below threshold {_fps_prune_threshold} to prune.", _C["muted"])

                    elif _fps_cmd == "drop":
                        _fps_drop_toks = _fps_rest.split()
                        if len(_fps_drop_toks) < 2:
                            _print("  usage: /failure-patterns drop <n> [or drop <substr>]", _C["yellow"])
                        else:
                            _fps_drop_arg = _fps_drop_toks[1]
                            _fps_was_substring = False
                            try:
                                _fps_drop_idx = int(_fps_drop_arg)
                                _fps_drop_result = _fps.drop(_fps_drop_idx)
                            except ValueError:
                                # Substring match: find all matching patterns by tool/reason.
                                _fps_was_substring = True
                                _fps_top_200 = _fps.top_patterns(limit=200)
                                _fps_sub = _fps_drop_arg.lower()
                                _fps_matches = [
                                    _p for _p in _fps_top_200
                                    if _fps_sub in _p.get("tool", "").lower()
                                    or _fps_sub in _p.get("reason", "").lower()
                                ]
                                if not _fps_matches:
                                    _print(f"  no pattern matching '{_fps_drop_arg}'", _C["yellow"])
                                    return ("continue", user_input)
                                if len(_fps_matches) > 1:
                                    _print(f"  {len(_fps_matches)} patterns match '{_fps_drop_arg}':", _C["yellow"])
                                    for _fps_mi, _fps_mp in enumerate(_fps_matches, 1):
                                        _fps_mt = _fps_mp.get("tool", "?")
                                        _fps_mr = (_fps_mp.get("reason", "") or "")[:80]
                                        _print(f"    {_fps_mi}. [{_fps_mt}] {_fps_mr}", _C["muted"])
                                    _print("  (use numeric index or a more specific substring)", _C["muted"])
                                    return ("continue", user_input)
                                _fps_drop_result = _fps.drop_key(_fps_matches[0]["key"])
                            if _fps_drop_result is not None:
                                _fps_drop_tool, _fps_drop_reason = _fps_drop_result
                                _print(f"  ✓ dropped [{_fps_drop_tool}] {_fps_drop_reason}", "dim")
                            elif _fps_was_substring:
                                _print(f"  pattern '{_fps_drop_arg}' disappeared before drop (race)", _C["yellow"])
                            else:
                                _print(f"  no pattern #{_fps_drop_arg}  (valid: 1-{_fps.store_size()})", _C["yellow"])

                    elif _fps_cmd and _fps_cmd not in _FAILURE_PATTERNS_SUBCOMMANDS:
                        _print(f"  unknown subcommand '{_fps_cmd}' — use: /failure-patterns [{'|'.join(_FAILURE_PATTERNS_SUBCOMMANDS)}]", _C["yellow"])
                        return ("continue", user_input)

                    else:
                        # Default: list top patterns.
                        _fps_top = _fps.top_patterns(limit=20)
                        if not _fps_top:
                            _print("  failure-pattern store: empty (no patterns recorded yet).", _C["muted"])
                        else:
                            _print(f"  failure-pattern store: {_fps.store_size()} patterns (top shown):", _C["text"])
                            for _i, _p in enumerate(_fps_top, 1):
                                _eff = _p.get("effective", 0)
                                _raw = _p.get("count", 0)
                                _tool = _p.get("tool", "?")
                                _reason = (_p.get("reason", "") or "")[:80]
                                _print(f"    {_i}. [{_tool}] eff={_eff} raw={_raw}  {_reason}", _C["muted"])
                            _print("  (use /failure-patterns drop <n|substr> or /failure-patterns clear)", _C["muted"])
                except Exception as _fps_e:
                    _print(f"  failure-pattern store error: {_fps_e}", _C["red"])
                return ("continue", user_input)

            # Catch-all: any command in the dispatch tuple above that didn't
            # already `continue` falls through here. (/orchestrate is NOT in that
            # tuple — it is a persistent mode handled by the mode-switch parsing
            # block further below, like /code and /general.)
            return ("continue", user_input)

        # ── /claude: Claude Code Agent collaboration ──
        if _cmd_name == "/claude":
            _claude_raw = _cmd_tok[1].strip() if len(_cmd_tok) > 1 else ""
            # Parse `/claude [--fresh] [--model <name> | -m <name>] <task>`
            _claude_model = None  # None → DEFAULT_COLLAB_MODEL (sonnet) — SDK default is user's default model (Opus-class), banned
            _claude_fresh = False  # True → do not share design session conversation context (blind verification)
            _rest = _claude_raw
            while True:
                if _rest.startswith("--fresh"):
                    _claude_fresh = True
                    _rest = _rest[len("--fresh"):].strip()
                elif _rest.startswith("--model "):
                    _sp_name, _, _rest = _rest[len("--model "):].strip().partition(" ")
                    _claude_model = _sp_name.strip() or None
                    _rest = _rest.strip()
                elif _rest.startswith("-m "):
                    _sp_name, _, _rest = _rest[len("-m "):].strip().partition(" ")
                    _claude_model = _sp_name.strip() or None
                    _rest = _rest.strip()
                else:
                    break
            _claude_task = _rest
            if not _claude_task:
                _print("  usage: /claude [--fresh] [--model <name>] <task description>", _C["yellow"])
                return ("continue", user_input)
            from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
            from external_llm.repl.collaborate import (
                DEFAULT_COLLAB_MODEL,
                CollaborationOrchestratorConfig,
                build_collaborate_install_spec,
                build_session_handoff,
                format_verdict_for_session,
                is_claude_sdk_installed,
            )
            from external_llm.repl.collaborate.streaming_display import StreamingDisplay

            # ── Ensure optional claude_agent_sdk is available; offer one-shot install ──
            # The SDK is an optional dependency. Rather than abort on a bare
            # ImportError, prompt to install it now (typed availability gate — never
            # key off error message text). The install command is derived from the
            # asicode distribution metadata so it never runs in the user's project
            # repo_root by mistake.
            if not is_claude_sdk_installed():
                _print(
                    "  claude_agent_sdk (optional dependency for Claude Code collaboration) is not installed.",
                    _C["yellow"],
                )
                try:
                    _claude_install_ans = input("  Install it now? (y/N) ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    _claude_install_ans = ""
                if _claude_install_ans in ("y", "yes"):
                    _claude_spec = build_collaborate_install_spec()
                    _print("  running: pip install " + " ".join(_claude_spec), "dim")
                    # Reuse the shared _pip_install so the PEP 668 externally-managed
                    # retry (Homebrew/system Python) and the live install spinner are
                    # identical to every other in-REPL pip install — no bespoke
                    # subprocess path that silently skips the retry.
                    try:
                        _claude_ok = _pip_install(_claude_spec, label="claude_agent_sdk")
                    except KeyboardInterrupt:
                        _print("\n  cancelled.", _C["yellow"])
                        return ("continue", user_input)
                    if _claude_ok and is_claude_sdk_installed():
                        _print("  \u2713 installed \u2014 starting collaboration session.", _C["green"])
                    else:
                        _print(
                            "  \u2717 installation did not complete successfully. Install manually:\n"
                            "      pip install '.[collaborate]'",
                            _C["red"],
                        )
                        return ("continue", user_input)
                else:
                    _print("  Install with:  pip install '.[collaborate]'", "dim")
                    return ("continue", user_input)

            # ASR→Claude: handoff design session conversation context (compressed_summary +
            # verbatim recent turns truncated — no LLM call). --fresh: no sharing.
            _claude_context = None
            if not _claude_fresh:
                try:
                    _handoff_session = _session_mgr.get_or_create(_session_id)
                    _claude_context = build_session_handoff(_handoff_session) or None
                except Exception:
                    _claude_context = None  # Handoff failure does not prevent session execution

            _claude_model = _claude_model or DEFAULT_COLLAB_MODEL
            _claude_display = StreamingDisplay(verbose=args.verbose)
            _claude_display.print_header(_claude_task, model=_claude_model)
            _claude_config = CollaborationOrchestratorConfig(
                event_callback=_claude_display.handle_event,
                repo_root=repo_root,
                model=_claude_model,
            )
            _claude_registry = ToolRegistry(repo_root, AgentConfig(unrestricted_read=True))  # trusted local CLI

            # Suppress INFO logs during session to prevent breaking in-place ○→✓ lines
            # (WARNING+ passes, file handlers unaffected)
            _tool_running_filter.active = True
            try:
                import asyncio
                _claude_result = asyncio.run(_run_collaborate_session(
                    _claude_registry, _claude_config, _claude_task,
                    context=_claude_context,
                ))
                _claude_display.print_summary(_claude_result)
                _claude_display.flush_log()
                # Claude→ASR: record verdict as a clearly source-labeled turn in the session —
                # so design LLM can reference "the analysis just now" starting next turn.
                # (appended to conversation tail, cache-safe)
                if _claude_result is not None and not _claude_result.error:
                    try:
                        _session_mgr.add_turn(
                            _session_id, "assistant",
                            format_verdict_for_session(_claude_result, _claude_task),
                            model="claude-code", exclude_from_compression=True,
                        )
                    except Exception:
                        logging.getLogger(__name__).debug("claude verdict record failed", exc_info=True)
            except KeyboardInterrupt:
                _print("\ncancelled.", _C["yellow"])
            except Exception as _claude_err:
                _print(f"  ✗ collaboration error: {_claude_err}", _C["red"])
                if args.verbose:
                    import traceback as _tb_c
                    _tb_c.print_exc()
            finally:
                # Stop ticker on cancel/error paths too — otherwise idle spinner keeps drawing
                # on top of the REPL prompt
                _claude_display.stop()
                _tool_running_filter.active = False
            return ("continue", user_input)

        # Resume paused work: no special branch — all input goes to design chat,
        # and LLM decides whether to resume based on the interrupt note (tool-loop record) in session.

        # ── Collect previous worker left in background by ESC ──
        # Interrupt note must be recorded before new user turn is added to session
        # to maintain conversation order. If worker is still receiving response, wait here.
        if _pending_dc is not None:
            try:
                _finalize_pending_design_chat(
                    _pending_dc, _session_mgr, _session_id, svc.model or "")
            except KeyboardInterrupt:
                _print("\nsession ended.", "")
                return ("break", user_input)
            except Exception as _fe:
                logging.getLogger(__name__).warning(
                    "pending design chat finalize failed: %s", _fe)
            finally:
                _pending_dc = None

        # ── Input length limit (prevents terminal/LLM overload) ──
        _input_truncated = False
        if len(user_input) > _cfg.lines.DESIGN_TURN_MAX_CHARS:
            user_input = user_input[:_cfg.lines.DESIGN_TURN_MAX_CHARS] + "\n\n[TRUNCATED]"
            _input_truncated = True

        # ── Mode switch prefix parsing: /general · /code · /orchestrate ──
        # /orchestrate is now a PERSISTENT mode (same model as /code and /general):
        # the user enters orchestrator mode, runs tasks in a loop that inherits the
        # shared design-chat session context, and each task + its result is persisted
        # as turns — so the orchestrator and any later design-chat turn stay continuous.
        # Exit orchestrator mode by switching to /code or /general.
        _mode_switched = False
        _stripped = user_input.strip()
        if _stripped.startswith("/general ") or _stripped == "/general":
            if _current_chat_mode != "general":
                _current_chat_mode = "general"
                _ds = _session_mgr.get_or_create(_session_id)
                _ds.chat_mode = "general"  # in-process sync only; NOT persisted to shared session
                _persist_chat_mode("general")
                _invalidate_next_suggestion()
                _print("  switched to [General Chat] mode — no code context loaded.", _C["blue"])
                _mode_switched = True
            elif not _stripped[len("/general"):].strip():
                _print("  already in [General Chat] mode.", _C["sky"])
                return ("continue", user_input)
            user_input = _stripped[len("/general"):].lstrip()
        elif _stripped.startswith(("/orchestrate ", "/orch ")) or _stripped in {"/orchestrate", "/orch"}:
            # Extract the inline task (if any) after /orchestrate or /orch
            _tok0 = _cmd_tok[0].lower() if _cmd_tok else ""
            _orch_inline = _stripped[len(_tok0):].lstrip() if _tok0 in ("/orchestrate", "/orch") else ""
            if _current_chat_mode != "orchestrator":
                _current_chat_mode = "orchestrator"
                _ds = _session_mgr.get_or_create(_session_id)
                _ds.chat_mode = "orchestrator"  # in-process sync only; NOT persisted to shared session
                _persist_chat_mode("orchestrator")
                _print("  switched to [Orchestrator] mode — decompose and dispatch tasks to sub-agents.  (/code to exit)", _C["blue"])
                _mode_switched = True
            elif not _orch_inline:
                # already in orchestrator mode AND no inline task → just nudge
                _print("  already in [Orchestrator] mode.", _C["sky"])
                # bare /orchestrate → wait for a task on the next prompt
                return ("continue", user_input)
            # /orchestrate <task> → set user_input to the task so the orchestrator
            # execution path (below) runs it this same iteration.
            user_input = _orch_inline
        elif _stripped.startswith("/code ") or _stripped == "/code":
            if _current_chat_mode != "code":
                _current_chat_mode = "code"
                _ds = _session_mgr.get_or_create(_session_id)
                _ds.chat_mode = "code"  # in-process sync only; NOT persisted to shared session
                _persist_chat_mode("code")
                _invalidate_next_suggestion()
                _print("  switched to [Code Chat] mode — full context loaded.", _C["blue"])
                _mode_switched = True
            elif not _stripped[len("/code"):].strip():
                _print("  already in [Code Chat] mode.", _C["sky"])
                return ("continue", user_input)
            user_input = _stripped[len("/code"):].lstrip()

        # ── /think on | off (thinking/reasoning toggle) ──
        if _cmd_name == "/think":
            _think_arg = _cmd_tok[1].lower().strip() if len(_cmd_tok) > 1 else ""
            if _think_arg in ("on", "1", "true", "yes"):
                _thinking_state = True
                _reasoning_effort = None
            elif _think_arg in ("off", "0", "false", "no", "none"):
                _thinking_state = False
                _reasoning_effort = None
            elif _think_arg in ("high", "max", "low", "medium", "minimal", "xhigh"):
                # thinking ON + effort level
                _thinking_state = True
                _reasoning_effort = _think_arg
            elif _think_arg in ("auto", "default", "provider"):
                # reset to provider auto-decide
                _thinking_state = None
                _reasoning_effort = None
            else:
                # toggle: None → True, True → False, False → True
                _thinking_state = _thinking_state is not True
                _reasoning_effort = None
            if hasattr(svc, "llm_service") and svc.llm_service is not None:
                svc.llm_service.thinking_mode = _thinking_state
                svc.llm_service.reasoning_effort = _reasoning_effort
            design_config.thinking_mode = _thinking_state
            design_config.reasoning_effort = _reasoning_effort
            if _thinking_state is True:
                _state_label = "ON"
            elif _thinking_state is False:
                _state_label = "OFF"
            else:
                _state_label = "auto (provider decides)"
            if _reasoning_effort:
                _state_label += f" (effort={_reasoning_effort})"
            # ── State persistence (restored on CLI restart) ──
            _update_terminal_config(
                {"thinking_state": _thinking_state, "reasoning_effort": _reasoning_effort}
            )
            _print(f"  thinking/reasoning → {_state_label}", _C["sky"])
            return ("continue", user_input)

        # ── /auto [N|on|off] — auto-continue the suggested next step (self-improve loop) ──
        if _cmd_name == "/auto":
            _auto_arg = _cmd_tok[1].strip() if len(_cmd_tok) > 1 else ""
            _new_on, _new_cap, _auto_err = _parse_auto_arg(
                _auto_arg, _auto_continue_state["on"])
            if _auto_err:
                _print(f"  {_auto_err}", _C["yellow"])
                return ("continue", user_input)
            _auto_continue_state["on"] = _new_on
            if _new_cap is not None:
                _auto_continue_state["cap"] = _new_cap
            _auto_continue_state["depth"] = 0  # every /auto re-anchors the loop
            if _new_on:
                _print(
                    f"  🔁 auto-continue ON — after each turn, a REQUIRED next step is "
                    f"auto-run after {_AUTO_CONTINUE_DELAY:.0f}s (max {_auto_continue_state['cap']} "
                    "consecutive; type/Esc cancels, Enter runs now, /auto off disables)",
                    _C["sky"])
                if not _cfg.display.NEXT_SUGGEST:
                    _print("  ▲ next-step suggestion is disabled in config "
                           "(display.NEXT_SUGGEST) — auto-continue will never fire.",
                           _C["yellow"])
            else:
                _cancel_auto_submit()
                _print("  🔁 auto-continue OFF", _C["sky"])
            return ("continue", user_input)

        # Pure mode switch with no message → wait for next input
        if _mode_switched and not user_input:
            return ("continue", user_input)
        return ("chat", user_input)

    def _run_chat_turn(user_input: str) -> str:
        """Run the chat turn — orchestrator mode, design-chat loop and all
        post-processing (token display, compression, change summary, next-task
        suggestion).  Extracted from the main loop (P1-2 continue/break refactor);
        returns an action code consumed by the caller:

          "break"    — end the session
          "continue" — handled; caller continues the main loop
          "ok"       — turn completed; fall through to the next iteration
        """
        nonlocal _pending_dc, _last_final_msg
        # ════════════════════════════════════════════════════════════════════
        #  Orchestrator mode execution (persistent — mirrors design-chat path)
        # ════════════════════════════════════════════════════════════════════
        # When _current_chat_mode == "orchestrator", each non-empty user_input is
        # an orchestration task.  It inherits the shared design-chat session
        # context (repo / project.md / insights / prior turns — orchestrator tasks
        # AND design-chat exchanges share one session), runs via the orchestrator
        # tool loop, and the task + final summary are persisted as turns so the
        # NEXT orchestrator turn and any later design-chat turn stay continuous.
        if _current_chat_mode == "orchestrator" and user_input:
            _orch_task = user_input
            # Persist the task as a USER turn BEFORE running, so the orchestrator's
            # build_context_messages picks it up as the current request (same
            # contract as the design-chat path).  in_progress guards parallel
            # terminals sharing this session.
            _session_mgr.add_turn(_session_id, "user", _orch_task, in_progress=True, auto=_was_auto_input)
            from external_llm.agent.orchestrator import (
                OrchestratorAgent,
                OrchestratorConfig,
            )
            from external_llm.agent.tool_registry import ToolRegistry
            from external_llm.client import create_llm_client

            _orch_printer = _ProgressPrinter(verbose=args.verbose)
            _orch_cancel = threading.Event()
            _orch_esc_stop = threading.Event()

            # Reuse the interactive design-chat config as the SINGLE SOURCE OF
            # TRUTH (dataclasses.replace → shallow copy).  Inherits EVERY design-
            # chat callback (user_checkpoint_callback / approval_callback) +
            # thinking_mode + thresholds, overriding only the per-task fields.
            import dataclasses as _dc
            _orch_cfg = _dc.replace(
                design_config,
                cancel_event=_orch_cancel,
                stream_callback=_orch_printer,
                _user_checkpoint_count=0,   # fresh question budget per task
            )
            _orch_registry = ToolRegistry(repo_root, _orch_cfg)
            _orch_client = create_llm_client(
                provider=_provider_str,
                api_key=os.getenv(_API_KEY_ENV_MAP.get(_provider_str.lower(), ""), ""),
            )
            _orch_sub_provider = _provider_str
            _orch_sub_model = _model_str
            _orch_sub_api_key = os.getenv(_API_KEY_ENV_MAP.get(_provider_str.lower(), ""), "")

            _orch_subagent_models: dict[str, tuple[str, str, str]] = {}
            for _dk, (_dp, _dm) in _dev_models.items():
                _dk_key = os.getenv(_API_KEY_ENV_MAP.get(_dp.lower(), ""), "")
                _orch_subagent_models[str(_dk)] = (_dp, _dm, _dk_key or _orch_sub_api_key)

            _orch_config = OrchestratorConfig(
                tool_loop_enabled=True,
                subagent_mode="ipc",
                # macOS: a VISIBLE Terminal.app window per worker (watchable).
                auto_launch_terminal=True,
                # Cross-platform: headless background subprocess worker.  On macOS
                # this is the fallback if Terminal.app launch fails; on Linux/
                # Windows it is the primary path — so IPC now works everywhere
                # instead of force-downgrading to in_process off macOS.
                auto_spawn_worker=True,
                cancel_event=_orch_cancel,
                thinking_mode=_thinking_state,
                reasoning_effort=_reasoning_effort,
                session_mgr=_session_mgr,
                subagent_provider=_orch_sub_provider,
                subagent_model=_orch_sub_model,
                subagent_api_key=_orch_sub_api_key,
                subagent_models=_orch_subagent_models,
            )
            if sys.platform != "darwin":
                _print(
                    "  ℹ non-macOS: sub-agents run as headless background workers "
                    "(logs in .asicode/subagents/<id>/worker.log)",
                    _C["muted"],
                )
            _orch_agent = OrchestratorAgent(
                llm_client=_orch_client,
                registry=_orch_registry,
                orch_config=_orch_config,
                model=_model_str,
                callback=_orch_printer,
                design_stream_callback=_orch_printer,
            )
            _orch_watcher = threading.Thread(
                target=_run_esc_watcher,
                args=(_orch_cancel, _orch_esc_stop),
                daemon=True,
            )
            _orch_watcher.start()
            _orch_printer._start_spinner("")
            _invalidate_next_suggestion()  # New orchestrator turn — discard previous suggestion
            _orch_result = None
            try:
                _orch_result = _orch_agent.run(_orch_task, session_id=_session_id)
            except KeyboardInterrupt:
                _orch_cancel.set()
                _print("\n  cancelled.", _C["yellow"])
            except Exception as _orch_err:
                _print(f"  ✗ orchestrator error: {_orch_err}", _C["red"])
                if args.verbose:
                    import traceback as _tb_o
                    _tb_o.print_exc()
            finally:
                _orch_printer._stop_spinner()
                _orch_esc_stop.set()
                _orch_watcher.join(timeout=1.0)
                _drain_stdin()
            if _orch_result:
                _print("", "")
                _print(f"  status: {_orch_result.status}", _C["green"] if _orch_result.status == "success" else _C["yellow"])
                if _orch_result.summary:
                    _orch_body, _orch_ws = _split_work_state(_orch_result.summary)
                    if _orch_body.strip():
                        _last_final_msg = _orch_body  # Update /copy target
                    if _RICH and asi._out_console:
                        try:
                            from rich.console import Group as _RichGroup
                            _RichMD = _rich_markdown_cls()
                            from rich.text import Text as _RichTxt
                            _f = asi._out_console.file
                            if hasattr(_f, "reset_bol"):
                                _f.reset_bol()
                            _orch_rend: list = []
                            if _orch_body:
                                _orch_rend.append(_RichMD(_orch_body))
                            if _orch_ws:
                                _orch_rend.append(_RichTxt(_orch_ws, style=_C["muted"]))
                            asi._out_console.print()
                            asi._out_console.print(_bar_panel(
                                _RichGroup(*_orch_rend) if len(_orch_rend) > 1 else (_orch_rend[0] if _orch_rend else _RichTxt("")),
                                title=_RichTxt(" ✦ orchestrator ", style=f"bold {_C['blue']}"),
                                color=_C["blue"],
                            ))
                        except Exception:
                            _print(f"\n{_orch_result.summary}", "")
                    else:
                        _print(f"\n{_orch_body}", "")
                _st_count = getattr(_orch_result, "total_turns", 0)
                if _st_count:
                    _print(f"  total sub-agent turns: {_st_count}", _C["muted"])
                # ── Persist the result as an ASSISTANT turn ──────────────────
                # This closes the user/assistant pair so the NEXT turn (orchestrator
                # OR design-chat) sees both the task and its outcome in the shared
                # session — making orchestrator mode continuous with design-chat.
                # Also releases the in_progress flag on the matching user turn.
                _orch_summary = _orch_result.summary or f"[Orchestration status: {_orch_result.status}]"
                try:
                    _session_mgr.add_turn(
                        _session_id, "assistant", _orch_summary,
                        model=svc.model or "",
                        digest=_build_orchestrator_digest(_orch_result),
                        auto=_was_auto_input,
                    )
                    # Schedule background compression so orchestrator history does
                    # not grow unbounded (same safeguard as design-chat).
                    _comp_client, _comp_model = _get_compress_llm()
                    _session_mgr.schedule_background_compress(
                        _session_mgr.get_or_create(_session_id), _comp_model,
                        _comp_client, notify=_deferred_notify)
                except Exception:
                    logging.getLogger(__name__).debug("orchestrator turn persist failed", exc_info=True)
            else:
                # No result (cancelled / error) — still close the in_progress user
                # turn so the session doesn't leave it dangling.
                try:
                    _session_mgr.add_turn(
                        _session_id, "assistant", "[Orchestration produced no result.]",
                        model=svc.model or "",
                        auto=_was_auto_input,
                    )
                except Exception:
                    logging.getLogger(__name__).debug("orchestrator no-result turn close failed", exc_info=True)
            _print("")
            return "continue"

        # ── Design Chat execution (all requests go through Design Chat first) ──
        if _input_truncated:
            _print(f"  ▲ input truncated to {_cfg.lines.DESIGN_TURN_MAX_CHARS} chars.", _C["yellow"])

        # Ctrl+C graceful-exit hooks — declared before the try so the
        # KeyboardInterrupt handler below can reference them even when the
        # interrupt lands during setup (before the worker thread exists).
        _dc_thread: threading.Thread | None = None
        _dc_cancel: threading.Event | None = None

        try:
            # When image detected: save [Image attached] marker + text in session
            if _current_user_images:
                _store_content = f"[Image attached: {len(_current_user_images)} image(s)]\n{user_input}"
            else:
                _store_content = user_input
            # in_progress: marks this request as being processed until tool-loop ends and assistant turn recorded
            # — prevents other terminals sharing the same session from
            # mistaking it for their own request and duplicating execution
            _session_mgr.add_turn(_session_id, "user", _store_content, in_progress=True, auto=_was_auto_input)
            _ds = _session_mgr.get_or_create(_session_id)
            _context_msgs = _session_mgr.build_context_messages(_ds, current_model=svc.model or "", mode=_current_chat_mode)
            # Attach images to the LAST USER message — the context builder may
            # append trailing system messages (current-request marker, mode
            # notice) after it, so "last message" is not necessarily the request.
            _last_user_idx = max(
                (_i for _i, _cm in enumerate(_context_msgs) if _cm["role"] == "user"),
                default=-1,
            )
            _messages_for_llm = [
                LLMMessage(role=_cm["role"], content=_cm["content"],
                          images=(_current_user_images or None) if _i == _last_user_idx else None)
                for _i, _cm in enumerate(_context_msgs)
            ]

            # 3c. OCR enrichment: extract text from images for non-vision contexts
            if _current_user_images:
                try:
                    from external_llm.providers import _images_to_text
                    _ocr_text = _images_to_text(_current_user_images)
                    if _ocr_text.strip():
                        _messages_for_llm.append(LLMMessage(
                            role="system",
                            content=f"=== ATTACHED IMAGE(S) OCR TEXT ===\n{_ocr_text}\n=== END OCR TEXT ===",
                        ))
                except Exception:
                    logging.getLogger(__name__).debug("OCR enrichment failed", exc_info=True)

            # ── ESC watcher for the Design Chat phase ──
            # The design-chat tool loop (git/file tools etc.) runs here directly.
            # Without a watcher the terminal stays in canonical mode, so ESC just
            # echoes "^[" and is never captured. Wire cancel_event into the design
            # registry's config so ToolRegistry.dispatch + the loop's per-iteration
            # check both honor it.
            # New turn starts — previous turn's "next task" ghost suggestion is now stale
            _invalidate_next_suggestion()

            # Pre-run git snapshot — track files changed by this turn (/diff · /undo · change summary).
            # Previously REPL did not capture baseline, so /diff was always empty.
            _turn_baseline = _git_baseline(repo_root)
            # Outside a git work tree _git_baseline is None and the whole
            # /diff · /undo surface used to vanish with it. The agent's write
            # gate records a checkpoint either way, so remember which one was
            # newest BEFORE the turn — anything newer after it belongs to this
            # run and can serve as the undo point git cannot provide.
            _turn_cp_before = None if _turn_baseline else _newest_checkpoint_id(repo_root)

            _dc_cancel = threading.Event()
            _dc_esc_stop = threading.Event()
            design_config.cancel_event = _dc_cancel
            _dc_watcher = threading.Thread(
                target=_run_esc_watcher,
                args=(_dc_cancel, _dc_esc_stop),
                daemon=True,
            )
            _dc_watcher.start()

            design_cb = _ProgressPrinter(verbose=args.verbose)
            design_loop = DesignChatLoop(svc.llm_service.client, design_registry, svc.model)
            design_cb._start_spinner("")

            # ── Worker thread execution ──
            # On ESC, UI returns to prompt immediately; worker continues receiving
            # the ongoing LLM response without interruption, then stops at next checkpoint (AgentCancelled).
            # Result collection/session recording happens in _finalize_pending_design_chat()
            # right before next input — ensures interrupt note is recorded before new user turn.
            _dc_box: dict = {"result": None, "error": None, "elapsed": 0.0}

            def _dc_worker(_box=_dc_box, _loop=design_loop, _msgs=_messages_for_llm,
                           _cb=design_cb, _mode=_current_chat_mode,
                           _thinking=_thinking_state, _effort=_reasoning_effort):
                try:
                    # NOTE: token_callback not passed — CLI does not use token-by-token streaming.
                    # Uses only design_thinking_start/stop events
                    # to control spinner, and renders final response at once.
                    # webapp/design_chat.py does incremental rendering via design_token, but
                    # CLI path has no content handler in _ProgressPrinter, so enabling
                    # streaming would only add overhead.
                    _t_dc0 = time.monotonic()
                    _box["result"] = _loop.respond(
                        _msgs,
                        stream_callback=_cb,
                        max_tool_iterations=_cfg.counts.DESIGN_CHAT_MAX_TOOL_ITERATIONS,
                        session_id=_session_id,
                        session_mgr=_session_mgr,
                        mode=_mode,
                        thinking_mode=_thinking,
                        reasoning_effort=_effort,
                    )
                    _box["elapsed"] = time.monotonic() - _t_dc0
                except Exception as _we:
                    _box["error"] = _we
                    # Record partial elapsed so interrupted/failed turns still
                    # show how long the loop ran in the status line.
                    try:
                        _box["elapsed"] = time.monotonic() - _t_dc0
                    except NameError:
                        logging.getLogger(__name__).debug("dc worker: t0 unbound (early failure)")

            _dc_thread = threading.Thread(target=_dc_worker, daemon=True)
            _dc_thread.start()
            try:
                while _dc_thread.is_alive():
                    _dc_thread.join(timeout=0.15)
                    if _dc_cancel.is_set():
                        # Immediately block worker output — prevents interference with prompt screen
                        design_cb.mute()
                        break
            finally:
                design_cb._stop_spinner()
                _dc_esc_stop.set()
                _dc_watcher.join(timeout=1.0)
                _drain_stdin()

            if _dc_cancel.is_set():
                # ESC — return to prompt immediately even if worker is still running.
                # design_config.cancel_event must remain available for worker checkpoints
                # so don't reset here (reset in finalize).
                _print(f"\n{_PAUSED_HINT}", _C["yellow"])
                _pending_dc = {
                    "thread": _dc_thread,
                    "box": _dc_box,
                    "design_config": design_config,
                }
                return "continue"

            design_config.cancel_event = None
            if _dc_box["error"] is not None:
                raise _dc_box["error"]
            chat_result = _dc_box["result"]
            if chat_result is None:
                _print("  ▲ design chat returned no result — please retry.", _C["yellow"])
                _session_mgr.add_turn(
                    _session_id, "assistant", "[No result — please retry.]", model=svc.model or "",
                    auto=_was_auto_input,
                )
                return "continue"
        except KeyboardInterrupt:
            # ── Graceful design-chat worker shutdown before exit ──
            # The worker is a daemon thread; killing it mid-flight at interpreter
            # shutdown orphans multiprocessing semaphores (embedding/RAG paths via
            # joblib/loky) → resource_tracker "leaked semaphore objects" warning.
            # Signal cancel (worker unwinds at its next checkpoint via
            # AgentCancelled) and join with a bounded timeout; a second Ctrl+C
            # during the wait skips it (helper returns False immediately).
            _graceful_join_design_chat(_dc_thread, _dc_cancel)
            _print("", "")
            _print_session_summary(_session_tokens, _session_t0)
            _print("session ended.", "")
            return "break"
        except Exception as _de:
            _print(f"  ✗ design chat error: {_de}", _C["red"])
            if args.verbose:
                import traceback as _tb
                _tb.print_exc()
            _session_mgr.add_turn(
                _session_id, "assistant", f"[Error: {_de}]", model=svc.model or "",
                auto=_was_auto_input,
            )
            return "continue"
        # Design chat token display (design chat + cache + session cumulative on one line)
        _dt = chat_result.tokens_used or 0
        # Initialize context-occupancy trackers here (not only inside `if _dt:`) so
        # the /general-mode compression gate below always sees defined values, even
        # on turns that report no token usage.
        _lpt = chat_result.last_call_prompt_tokens or 0
        # Context window via the single source of truth (context_budget.py):
        # covers GLM/Qwen/Ollama (exact-match table + Ollama /api/show dynamic
        # query) and falls back to 1M for unknowns, so the /general occupancy
        # gate and ctx display work for EVERY model. Resolved outside `if _dt:`
        # so the gate still sees a valid budget on turns that report no tokens.
        from external_llm.agent.context_budget import _resolve_context_limit
        _ctx_budget = _resolve_context_limit(
            svc.model or chat_result.provider or args.provider or "",
            base_url=getattr(getattr(svc.llm_service, 'client', None), 'base_url', None))
        if _dt:
            from external_llm.agent._shared_utils import cache_cost_summary
            _provider = chat_result.provider or args.provider or ""
            _pt = chat_result.prompt_tokens or 0
            _ct = chat_result.completion_tokens or 0
            _lct = chat_result.last_call_completion_tokens or 0
            _crt = chat_result.cache_read_tokens or 0
            _cct = getattr(chat_result, "cache_creation_tokens", 0) or 0
            # full_cost = no-cache counterfactual; _actual = true billed cost.
            # Provider-aware: Anthropic reports prompt EXCLUDING cache, so the cost
            # math and hit% denominator differ from DeepSeek (which includes it).
            _zai_base_url = getattr(getattr(svc.llm_service, 'client', None), 'base_url', '') or ''
            _cost, _actual, _hit_pct = cache_cost_summary(
                _provider, _pt, _ct, _crt, _cct, model=svc.model or "", base_url=_zai_base_url,
            )
            _session_tokens["prompt"] += _pt
            _session_tokens["completion"] += _ct
            _session_tokens["cost"] += _cost
            _session_tokens["actual_cost"] += _actual
            # One-line integration: ctx | tok | cache — ctx (context usage %) is the most important
            # metric for current state, so it comes first. Large numbers are K/M abbreviated.
            _groups: list[str] = []
            if _ctx_budget and _lpt:
                _ctx_pct = _lpt * 100 // _ctx_budget
                # Show 1 decimal place for sub-1K values to avoid "0K" (e.g., 0.4K)
                if _ctx_budget >= 1_000_000:
                    _groups.append(f"ctx {_lpt/1000:.1f}K / {_ctx_budget/1_000_000:.0f}M ({_ctx_pct}%)")
                else:
                    _groups.append(f"ctx {_lpt/1000:.1f}K / {_ctx_budget//1000}K ({_ctx_pct}%)")
            # Per-turn ambient status line — money is intentionally excluded: the
            # dollar amount is an estimate, not exact billing, so it is not shown on
            # any CLI surface (debug _log only). Token counts and the cache-hit ratio
            # are kept (usage/efficiency, not "cost"). _cost/_actual are still
            # accumulated into _session_tokens (below) for debug logging.
            if _pt or _ct:
                _groups.append(f"tok ↑{_abbrev_tokens(_pt)}  ↓{_abbrev_tokens(_ct)}")
            if _crt and _pt:
                _groups.append(f"cache {_hit_pct:.0f}%")
            # Tool-loop wall-clock duration — per-turn latency (measured inside the
            # design-chat worker around loop.respond(), so it excludes thread
            # spawn/join overhead). Always present on a completed turn.
            _dc_elapsed = _dc_box.get("elapsed", 0.0) or 0.0
            if _dc_elapsed > 0:
                _groups.append(_fmt_elapsed(_dc_elapsed))
            # Session on separate line — wrapping past 80 chars breaks left alignment
            if _groups:
                _print(f"  {' │ '.join(_groups)}", _C["muted"])
            _print(
                f"  session  ↑{_abbrev_tokens(_session_tokens['prompt'])}"
                f"  ↓{_abbrev_tokens(_session_tokens['completion'])}",
                _C["muted"],
            )

        # ════════════════════════════════════════════════════════════════
        #  Design Chat response (text)
        # ════════════════════════════════════════════════════════════════
        # ── Auth error → new API key input then immediate retry ──
        # This runs BEFORE the render branch below, not inside it. The retry
        # replaces ``chat_result`` wholesale, and a successful one has to reach
        # the normal render path — when this lived inside the ``is_error``
        # branch, the retry's answer was fully computed (whole tool loop, real
        # tokens, session turn recorded) and then never displayed, so the REPL
        # dropped straight back to the prompt with no final summary.
        _dc_error_shown = False
        if chat_result.is_error and chat_result.error_type == "auth":
            # No separate "✗ error" label — content already starts with ⚠️
            # user-facing error message and printed in red, so it would be redundant.
            # The preceding terminal log (WARNING Design chat LLM call failed …) is on its own line.
            _print(f"\n{chat_result.content}", _C["red"])
            _dc_error_shown = True
            if _prompt_auth_retry_key(
                svc.llm_service.provider, svc,
                error_message=chat_result.content,
            ):
                design_loop = DesignChatLoop(
                    svc.llm_service.client, design_registry, svc.model
                )
                _retry_cb = _ProgressPrinter(verbose=args.verbose)
                _retry_cb._start_spinner("  🔄 retrying with new API key")
                try:
                    chat_result = design_loop.respond(
                        _messages_for_llm,
                        stream_callback=_retry_cb,
                        max_tool_iterations=_cfg.counts.DESIGN_CHAT_MAX_TOOL_ITERATIONS,
                        session_id=_session_id,
                        session_mgr=_session_mgr,
                        mode=_current_chat_mode,
                        thinking_mode=_thinking_state,
                        reasoning_effort=_reasoning_effort,
                    )
                except Exception as _retry_exc:
                    chat_result = DesignChatResult(
                        content=f"⚠️ retry failed: {_retry_exc}",
                        is_error=True,
                    )
                # chat_result is a fresh outcome now — let the branches below
                # render it (success) or report it (a NEW error), instead of
                # suppressing them because the ORIGINAL error was printed.
                _dc_error_shown = False
                # Only a retry the provider actually accepted proves the key
                # is good; persist it to .env at that point and not before.
                if not chat_result.is_error:
                    asi._commit_verified_api_key()

        if chat_result.is_error:
            if not _dc_error_shown:
                _print(f"\n{chat_result.content}", _C["red"])
        else:
            # Content after [WORK STATE …] (turn's tool usage record) is not body but
            # metadata — separate and dim it. Full original stored in session.
            _dc_body, _dc_work_state = _split_work_state(chat_result.content)
            if _dc_body.strip():
                _last_final_msg = _dc_body  # Update /copy target
            if not chat_result.content:
                _print("  ▲ design chat returned an empty response — please retry.", _C["yellow"])
            elif _RICH and asi._out_console:
                try:
                    from rich.console import Group as _RichGroup
                    _RichMD = _rich_markdown_cls()
                    from rich.text import Text as _RichTxt
                    # Final answer also gets left gutter bar — same family as … thinking (mid-utterance),
                    # with title ✦(blue) to distinguish as "this turn's final answer"
                    _f = asi._out_console.file
                    if hasattr(_f, "reset_bol"):
                        _f.reset_bol()
                    _renderables: list = []
                    if _dc_body:
                        _renderables.append(_RichMD(_dc_body))
                    asi._out_console.print()
                    asi._out_console.print(_bar_panel(
                        _RichGroup(*_renderables),
                        title=_RichTxt(" ✦ ", style=f"bold {_C['blue']}"),
                        color=_C["blue"],
                    ))
                except Exception:
                    _print(f"\n{chat_result.content}", "")
            elif _dc_body:
                _print(f"\n{_dc_body}", "")
        _turn_digest = _build_turn_digest(chat_result)
        _session_mgr.add_turn(
            _session_id, "assistant", chat_result.content, model=svc.model or "",
            digest=_turn_digest,
            auto=_was_auto_input,
        )
        if not chat_result.is_error:
            _comp_client, _comp_model = _get_compress_llm()
            if _current_chat_mode == "general":
                # /general mode disables the periodic turn-count auto-compression
                # (MIN_RECENT_TURNS_KEEP + COMPRESS_BATCH_MIN): turns accumulate verbatim
                # so the stable prefix — and its prompt cache — survives across many
                # turns. Compression (summarize) fires only once the live context window
                # nears its limit, preempting the lossy hard-cap front-trim. force=True
                # bypasses the turn-count gate in needs_compression().
                if _ctx_budget and _lpt and (
                    _lpt / _ctx_budget >= _cfg.compression.GENERAL_MODE_COMPRESS_OCCUPANCY
                ):
                    _session_mgr.schedule_background_compress(
                        _session_mgr.get_or_create(_session_id), _comp_model,
                        _comp_client, notify=_deferred_notify, force=True)
            else:
                _session_mgr.schedule_background_compress(
                    _session_mgr.get_or_create(_session_id), _comp_model,
                    _comp_client, notify=_deferred_notify)

        # ── This turn's change summary (per-file +N -M) — enables /diff · /undo ──
        if _turn_baseline:
            try:
                if _print_run_change_summary(repo_root, _turn_baseline):
                    _last_run_diff["repo_root"] = repo_root
                    _last_run_diff["baseline"] = _turn_baseline
                    _last_run_diff["checkpoint"] = None
                    if _cfg.display.RUN_DIFF:
                        _render_run_diff(repo_root, _turn_baseline)
                    else:
                        _print("  /diff full diff  ·  /undo revert", _C["muted"])
            except Exception:
                logging.getLogger(__name__).debug("turn change summary failed", exc_info=True)
        else:
            # Non-git directory: no baseline to diff against, but the write
            # gate's checkpoint still knows exactly what this run touched.
            # /undo becomes available; /diff does not, since there is no stored
            # "after" side to render a unified diff from.
            try:
                _turn_cp = _newest_checkpoint_id(repo_root)
                if _turn_cp and _turn_cp != _turn_cp_before:
                    _cp_files = _checkpoint_changed_files(repo_root, _turn_cp)
                    if _cp_files:
                        _last_run_diff["repo_root"] = repo_root
                        _last_run_diff["baseline"] = None
                        _last_run_diff["checkpoint"] = _turn_cp
                        _print(f"  {len(_cp_files)} file(s) changed  ·  /undo revert",
                               _C["muted"])
            except Exception:
                logging.getLogger(__name__).debug(
                    "turn checkpoint summary failed", exc_info=True
                )

        # ── Next-task ghost suggestion generation (background, helper model) ──
        if _cfg.display.NEXT_SUGGEST and not chat_result.is_error:
            try:
                _sug_client, _sug_model = _get_compress_llm()
                _kick_next_prompt_suggestion(
                    _sug_client, _sug_model, user_input,
                    chat_result.content or "", _turn_digest,
                    auto_mode=_auto_continue_state["on"])
            except Exception:
                logging.getLogger(__name__).debug("next-task suggestion kick failed", exc_info=True)
        elif chat_result.is_error and _auto_continue_state["on"]:
            # An error turn generates no suggestion, so the auto loop stops here —
            # say so instead of going silent at the prompt.
            _print("  🔁 auto-continue: turn ended with an error — stopped", _C["yellow"])

        _print("")  # blank line
        # Auto compact if save_insight exceeded budget (after run ends, outside agent turn)
        _maybe_auto_compact_insights(chat_result)
        return "ok"

    while True:
        # Output delayed notifications first (background compress complete messages, etc.)
        _drain_notifications()

        # ── Prompt input ──
        try:
            _current_mode = _current_chat_mode
            if _thinking_state is True:
                _think_label = "think ON"
            elif _thinking_state is False:
                _think_label = "think OFF"
            else:
                _think_label = "think (auto)"
            if _reasoning_effort and _thinking_state is not False:
                _think_label = f"think ON ({_reasoning_effort})"
            _status_bits = [f"{_provider_str} / {_model_str}", _think_label]
            if _auto_continue_state["on"]:
                # Persistent mode indicator — the countdown fires turns later,
                # so the user must be able to see the mode is armed.
                _status_bits.append(
                    f"🔁 auto-continue {_auto_continue_state['depth']}/{_auto_continue_state['cap']}")
            user_input = _prompt_input(chat_mode=_current_mode, status="  ·  ".join(_status_bits))
            # ── Auto-continue depth bookkeeping ──
            # Countdown-submitted input deepens the chain (capped); any manual
            # input re-anchors the loop (depth reset) — the user took over.
            _was_auto_input = _last_input_was_auto
            _last_input_was_auto = False
            if _was_auto_input:
                _auto_continue_state["depth"] += 1
                _print(
                    f"  🔁 auto-continue step {_auto_continue_state['depth']}/{_auto_continue_state['cap']}",
                    _C["muted"])
            else:
                _auto_continue_state["depth"] = 0

            # ── Image file drag/clipboard/Data URL detection → base64 conversion ──
            _current_user_images: list[dict[str, str]] = []
            _clean_input, _detected = _extract_images_from_input(user_input)
            if not _detected:
                # If no file path/data URL, check clipboard (clipboard screenshot)
                _clip_images = _check_clipboard_image()
                if _clip_images:
                    _detected = _clip_images
            if _detected:
                _img_hint = "  📷 " + ", ".join(
                    f"{img['media_type']} ({len(img['data'])}B base64)"
                    for img in _detected
                )
                _print(_img_hint, _C["teal"])
                user_input = _clean_input
                _current_user_images = _detected
        except EOFError:
            print()
            _print_session_summary(_session_tokens, _session_t0)
            print("session ended.")
            break
        except KeyboardInterrupt:
            print()
            _print_session_summary(_session_tokens, _session_t0)
            print("session ended.")
            break

        if not user_input:
            continue

        # ── Slash commands & mode switches — dispatched to _dispatch_command ──
        _act, user_input = _dispatch_command(user_input)
        if _act == "break":
            break
        if _act == "continue":
            continue

        # ── Chat turn: orchestrator / design-chat execution + post-processing ──
        _turn_act = _run_chat_turn(user_input)
        if _turn_act == "break":
            break
        if _turn_act == "continue":
            continue


def run_repl(args: argparse.Namespace) -> None:
    """Public REPL entry point (called from ``main()``).

    The implementation lives in :func:`_run_repl_impl` (split out of the
    original monolith — P1-1, docs/improvement_proposals.md).
    """
    _run_repl_impl(args)
def _extract_patched_file(p) -> str:
    """Extract a repo-relative file path from one ``applied_patches`` entry.

    ``applied_patches`` is heterogeneous by source, so this normalises it into a
    clean path list for the JSON ``patched_files`` field:

    * IPC workers (``derive_applied_patches``) emit ``{"file": ...}`` dicts.
    * The in-process write path (``write_tools.py``) emits structured
      ``edit_file:PATH:...`` / ``edit_text:PATH:...`` op strings.
    * The patch engine emits raw unified-diff / patch text.

    Returns the path, or ``""`` if none can be reliably determined (callers drop
    empties). Centralising extraction keeps ``patched_files`` a clean list of
    paths regardless of which execution path produced the run.
    """
    if isinstance(p, dict):
        return str(p.get("file") or p.get("path") or "")
    s = str(p or "")
    if not s:
        return ""
    # Structured tool-op prefixes: edit_file:PATH:..., edit_text:PATH:...
    for _prefix in ("edit_file:", "edit_text:", "modify_symbol:"):
        if s.startswith(_prefix):
            return s[len(_prefix):].split(":", 1)[0].strip()
    # Raw unified-diff / patch text: prefer '+++ b/PATH', then 'diff --git a/PATH b/PATH'.
    import re
    _m = re.search(r"^\+\+\+ b/(.+)$", s, re.MULTILINE)
    if _m:
        return _m.group(1).strip()
    _m = re.search(r"^diff --git a/(.+?) b/", s, re.MULTILINE)
    if _m:
        return _m.group(1).strip()
    return ""


def _turns_to_int(turns) -> int:
    """Normalize a ``turns`` value to an int, tolerating BOTH shapes that reach
    ``_result_output_dict``:

    * a real ``AgentResult.turns`` is ``list[AgentTurn]``  -> ``len(list)``
    * the ``--orchestrate`` adapter (``_orchestrator_result_to_agent_like``) and
      IPC ``SubagentResult.turns`` are ``int``                          -> the int

    Calling ``len()`` on an int raises ``TypeError``, which previously crashed the
    final ``result`` event of a SUCCESSFUL orchestration (turn 13106 verification:
    "object of type 'int' has no len()"). Centralizing the normalization here makes
    the output builder robust to whichever shape a caller supplies.
    """
    if isinstance(turns, list):
        return len(turns)
    if isinstance(turns, bool):  # bool is an int subclass — guard before the int check
        return 0
    if isinstance(turns, (int, float)):
        return int(turns)
    return 0
def _result_output_dict(result, elapsed: float) -> dict:
    """Build the machine-readable result dict shared by --json and --json-stream.

    Kept separate from printing so the NDJSON streaming path (``--json-stream``)
    can emit the same payload as a final ``result`` event line, and so a stray
    metadata schema can never leave consumers without parseable output.
    """
    tokens = {}
    if hasattr(result, "metadata") and result.metadata and "tokens" in result.metadata:
        tokens = result.metadata["tokens"]
    # Structured clarification questions for automation (Tenet etc.). Populated
    # only on clarification_needed — extracted from the result metadata
    # (required_clarifications: [{field, reason, suggestion}, ...]) with a
    # free-text fallback to final_message. Always present (empty list) so
    # consumers can read one stable field regardless of status. Exit code 2
    # already signals clarification_needed; the body now carries the questions.
    questions: list = []
    if result.status == "clarification_needed":
        _md = getattr(result, "metadata", None) or {}
        _rc = _md.get("required_clarifications") or []
        if _rc:
            for _r in _rc:
                if isinstance(_r, dict):
                    _f = _r.get("field") or "?"
                    _reason = _r.get("reason") or ""
                    questions.append(f"{_f}: {_reason}".strip())
                else:
                    questions.append(str(_r))
        elif getattr(result, "final_message", None):
            questions.append(result.final_message)
    return {
        "status": result.status,
        "output": result.final_message or "",
        # F7: a cancelled result always carries a reason so consumers (Tenet etc.)
        # can read ONE field for "why did it fail" instead of special-casing the
        # status. The engine surfaces a cancel as result.error="" on some paths;
        # fill the canonical reason there so error is never null on cancelled.
        "error": (
            result.error
            or ("Request cancelled by user" if result.status == "cancelled" else None)
        ),
        "duration_ms": int(elapsed * 1000),
        "tokens_in": tokens.get("prompt", 0),
        "tokens_out": tokens.get("completion", 0),
        "total_tokens": tokens.get("total", 0),
        "cost_usd": tokens.get("cost_usd", 0),
        "patches": len(result.applied_patches) if result.applied_patches else 0,
        "patched_files": [
            _f for _f in (_extract_patched_file(p) for p in (result.applied_patches or [])) if _f
        ],
        # result.turns is ``list[AgentTurn]`` on the AgentResult path but an
        # ``int`` on the --orchestrate adapter path (see ``_turns_to_int``). When
        # it's falsy (empty list / 0), fall back to metadata["turns_used"], which
        # is where the normal MAIN_AGENT tool loop records the real turn count.
        "turns": (
            _turns_to_int(result.turns)
            or int((getattr(result, "metadata", None) or {}).get("turns_used") or 0)
        ),
        "questions": questions,
    }


def _build_json_output(result, elapsed: float) -> None:
    """Print machine-readable JSON result to stdout (for --json flag / Tenet integration).

    The output builder itself is guarded so a stray metadata schema (or any
    other attribute-access surprise on a *successful* run) can never leave
    stdout with NO JSON — automation consumers (Tenet etc.) require parseable
    output even then. On failure it falls back to _json_error_output.
    """
    try:
        print(json.dumps(_result_output_dict(result, elapsed), ensure_ascii=False), flush=True)
    except Exception as _bo_exc:
        logging.getLogger(__name__).exception("_build_json_output failed")
        _json_error_output(
            "unexpected_error", f"output build failed: {_bo_exc}",
            duration_ms=int(elapsed * 1000),
        )


def _json_error_output(status: str, error: str, duration_ms: int = 0) -> None:
    """Print minimal JSON error output to stdout (for --json flag / Tenet integration)."""
    print(json.dumps({
        "status": status,
        "output": "",
        "error": error,
        "duration_ms": duration_ms,
        "tokens_in": 0,
        "tokens_out": 0,
        "total_tokens": 0,
        "cost_usd": 0,
        "patches": 0,
        "turns": 0,
        # Stable schema: every status (success AND error) carries a questions
        # list so consumers can read one field unconditionally without KeyError.
        "questions": [],
    }, ensure_ascii=False), flush=True)


def _json_stream_emit(event: str, payload=None, **extra) -> None:
    """Emit one NDJSON line to stdout for --json-stream (Tenet progress feed).

    Each agent-loop stream event becomes a self-contained JSON object on its own
    line (newline-delimited JSON), so a long-running run streams turn/tool events
    as they happen instead of only emitting a single final blob. ``default=str``
    keeps serialization robust against event payloads carrying non-JSON values
    (Path, Exception, etc.). Never raises — streaming is advisory.
    """
    try:
        line = {"event": str(event)}
        if isinstance(payload, dict):
            line.update(payload)
        elif payload is not None:
            line["payload"] = payload
        if extra:
            line.update(extra)
        sys.stdout.write(json.dumps(line, default=str, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        logging.getLogger(__name__).debug("json stream emit failed", exc_info=True)


def _orchestrator_result_to_agent_like(orch_result):
    """F5: adapt an OrchestratorResult to the AgentResult-like shape
    :func:`_result_output_dict` expects (status / final_message / error /
    applied_patches / turns / metadata).

    The orchestrator runs sub-agents and aggregates their results; for the JSON
    output we flatten the sub-task patches + turn counts onto the top-level
    result so a single ``--json`` / ``--json-stream`` payload carries the full
    attribution. ``applied_patches`` keeps whatever form the sub-agents emitted
    (IPC ``{"file": ...}`` dicts or in-process structured strings); the output
    builder's ``_extract_patched_file`` normalizes both.
    """
    from types import SimpleNamespace
    _patches: list = []
    _turns = 0
    for _sr in (getattr(orch_result, "subtask_results", None) or []):
        if _sr is None:
            continue
        _patches.extend(getattr(_sr, "applied_patches", None) or [])
        # SubagentResult.turns is an int (the worker's turn count), NOT a list —
        # so ``len()`` would raise TypeError once a sub-agent reports a non-zero
        # count. Use ``_turns_to_int`` to tolerate both int and list shapes.
        _turns += _turns_to_int(getattr(_sr, "turns", 0))
    _status = getattr(orch_result, "status", "success")
    _summary = getattr(orch_result, "summary", "") or ""
    return SimpleNamespace(
        status=_status,
        final_message=_summary,
        error=None if _status in ("success", "already_satisfied") else (_summary or None),
        applied_patches=_patches,
        turns=_turns,
        metadata=getattr(orch_result, "metadata", None) or {},
    )


def _run_orchestrate_single_shot(
    args: argparse.Namespace, repo_root: str, prompt: str, _stream_cb, cancel_event,
) -> "Any":
    """F5: build and run a single-shot OrchestratorAgent (``--orchestrate``).

    Mirrors the interactive REPL's orchestrator construction but for the
    one-shot ``run_once`` path. The orchestrator decomposes *prompt* into sub-
    tasks and dispatches them to IPC sub-agent workers. ``_stream_cb`` is wired
    as BOTH the orchestrator event callback (subagent_start / complete / waiting)
    and the design-chat stream callback (the orchestrator's own tool calls), so a
    ``--json-stream`` consumer sees the full multi-agent progress live.
    """
    from external_llm.agent.orchestrator import OrchestratorAgent, OrchestratorConfig
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.intelligent_service import create_intelligent_service_from_env

    svc = create_intelligent_service_from_env(
        args.provider or None, args.model or None, api_key=args.api_key or None,
    )
    if svc is None:
        raise RuntimeError(
            "failed to initialize LLM service for orchestration.\n"
            f"  --provider {args.provider or '(unset)'} --model {args.model or '(unset)'}"
        )
    _cfg = AgentConfig(
        model_name=svc.model or "",
        stream_callback=_stream_cb,
        consume_content_events=False,
        cancel_event=cancel_event,
        unrestricted_read=True,   # trusted local CLI
    )
    if getattr(args, "thinking_mode", None) is not None:
        _cfg.thinking_mode = args.thinking_mode
    if getattr(args, "reasoning_effort", None) is not None:
        _cfg.reasoning_effort = args.reasoning_effort
    _registry = ToolRegistry(repo_root, _cfg)
    _orch_config = OrchestratorConfig(
        tool_loop_enabled=True,
        subagent_mode="ipc",
        # macOS: a VISIBLE Terminal.app window per worker (watchable). Elsewhere,
        # the headless background spawn is the primary IPC path.
        auto_launch_terminal=(sys.platform == "darwin"),
        auto_spawn_worker=True,
        cancel_event=cancel_event,
        thinking_mode=getattr(args, "thinking_mode", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        subagent_provider=args.provider or "",
        subagent_model=args.model or "",
        subagent_api_key=args.api_key or "",
    )
    _orch_agent = OrchestratorAgent(
        llm_client=svc.llm_service.client,
        registry=_registry,
        orch_config=_orch_config,
        model=svc.model or "",
        callback=_stream_cb,
        design_stream_callback=_stream_cb,
    )
    return _orch_agent.run(prompt)


def run_once(args: argparse.Namespace, prompt: str) -> int:
    """Execute a single request and return the exit code."""
    global _REPO_ROOT
    repo_root = _resolve_repo_root(args.repo)
    _REPO_ROOT = repo_root

    cancel_event = threading.Event()
    printer = _ProgressPrinter(verbose=args.verbose)
    # --json-stream: stream every agent-loop event as an NDJSON line (turn/tool
    # progress) instead of the human printer, so Tenet-style automation can
    # observe a long run in progress rather than only the final blob. The final
    # result is emitted as a ``result`` event line at the end of run_once.
    _use_json_stream = bool(getattr(args, "json_stream", False))
    # --json (single final blob): stdout must carry ONLY the final JSON line so a
    # consumer can json.loads(stdout) directly. The human _ProgressPrinter writes
    # ANSI progress to stdout, which would pollute the single-line contract — so
    # suppress it entirely here. (Progress is still available via --json-stream.)
    # Safe because the engine treats stream_callback purely as a callable
    # (stream_callback(event, data)); it never reaches for printer-only methods.
    _use_json_blob = bool(getattr(args, "json", False)) and not _use_json_stream
    if _use_json_stream:
        def _stream_cb(event, payload=None, **kw):
            _json_stream_emit(event, payload, **kw)
    elif _use_json_blob:
        def _stream_cb(*_a, **_k):
            pass
    else:
        _stream_cb = printer

    def _sigint_handler(sig, frame):
        if cancel_event.is_set():
            if not _use_json_stream and not _use_json_blob:
                _print("\nforcing exit…", _C["red"])
            sys.exit(130)
        cancel_event.set()
        # In JSON modes keep human text off stdout — the JSON error/cancel line
        # is the only thing a consumer should see there.
        if not _use_json_stream and not _use_json_blob:
            _print("\ncancelling…", _C["yellow"])

    signal.signal(signal.SIGINT, _sigint_handler)

    # t0 is set BEFORE the try so the error/cancel branches can always report
    # a meaningful duration_ms (previously they hardcoded 0, so automation could
    # never tell how long a failed/cancelled run actually took).
    t0 = time.perf_counter()
    elapsed = 0.0
    try:
        if getattr(args, "orchestrate", False):
            # F5: single-shot multi-agent orchestration. The orchestrator
            # decomposes the prompt and dispatches sub-tasks to IPC workers;
            # events flow through _stream_cb (NDJSON under --json-stream).
            _run_baseline = _git_baseline(repo_root)
            _orch_result = _run_orchestrate_single_shot(
                args, repo_root, prompt, _stream_cb, cancel_event,
            )
            result = _orchestrator_result_to_agent_like(_orch_result)
            elapsed = time.perf_counter() - t0
        else:
            engine_config = EngineConfig(
                repo_root=repo_root,
                request_text=prompt,
                provider=args.provider or "",
                model=args.model or "",
                api_key=args.api_key or None,
                max_turns=args.max_turns,
                stream_cb=_stream_cb,
                cancel_event=cancel_event,
                thinking_mode=getattr(args, "thinking_mode", None),
                reasoning_effort=getattr(args, "reasoning_effort", None),
                scoped_verification=(
                    getattr(args, "scoped_verification", True)
                    or os.environ.get("ASICODE_SCOPED_VERIFICATION", "").strip().lower()
                    in ("1", "true", "yes", "on")
                ),
            )
            loop = _build_engine(config=engine_config)
            _run_baseline = _git_baseline(repo_root)
            result = _run_with_cancel(loop, prompt, "", cancel_event, stream_callback=_stream_cb)
            elapsed = time.perf_counter() - t0
    except RuntimeError as e:
        elapsed = time.perf_counter() - t0
        if _use_json_stream:
            _json_stream_emit("error", {"status": "error", "error": str(e), "duration_ms": int(elapsed * 1000)})
        elif args.json:
            _json_error_output("error", str(e), duration_ms=int(elapsed * 1000))
        else:
            _print(f"error: {e}", _C["red"])
        return 1
    except Exception as e:
        elapsed = time.perf_counter() - t0
        if _use_json_stream:
            _json_stream_emit("error", {"status": "unexpected_error", "error": str(e), "duration_ms": int(elapsed * 1000)})
        elif args.json:
            _json_error_output("unexpected_error", str(e), duration_ms=int(elapsed * 1000))
        else:
            _print(f"unexpected error: {e}", _C["red"])
            if args.verbose:
                import traceback
                traceback.print_exc()
        return 1

    if result is None:
        elapsed = time.perf_counter() - t0
        if _use_json_stream:
            _json_stream_emit("cancelled", {"status": "cancelled", "error": "Request cancelled by user", "duration_ms": int(elapsed * 1000)})
        elif args.json:
            _json_error_output("cancelled", "Request cancelled by user", duration_ms=int(elapsed * 1000))
        else:
            _print("cancelled.", _C["yellow"])
        return 130

    # Exit-code helper shared by the JSON / NDJSON final-event paths.
    # cancelled → 130 matches the result-is-None cancel branch above, so
    # automation sees ONE cancel exit code regardless of whether the engine
    # surfaced the cancel as None or as a result with status="cancelled".
    _exit_for_status = (
        0 if result.status in ("success", "already_satisfied")
        else 2 if result.status == "clarification_needed"
        else 130 if result.status == "cancelled"
        else 1
    )

    # --json-stream: emit the final result as the last NDJSON line (same payload
    # as --json) and return. Progress events already streamed during the run.
    if _use_json_stream:
        _json_stream_emit("result", _result_output_dict(result, elapsed))
        return _exit_for_status

    # --json: machine-readable JSON output (for Tenet integration)
    if args.json:
        _build_json_output(result, elapsed)
        return _exit_for_status

    _show_result(result, elapsed, repo_root=repo_root, baseline=_run_baseline)

    # Handle clarification_needed: show questions and exit with specific code
    if result.status == "clarification_needed" and result.final_message:
        _print("\n  💡 The assistant needs additional information:", _C["yellow"])
        _print(f"\n{result.final_message}", _C["text"])
        _print("\n  Use --prompt (REPL mode) to provide an answer.", _C["muted"])
        return 2  # exit code 2 = clarification needed

    if result.status == "cancelled":
        return 130  # same cancel exit code as the result-is-None branch

    return 0 if result.status in ("success", "already_satisfied") else 1


# ─── Entry point ──────────────────────────────────────────────────────────────────


def run_subagent_worker(args: argparse.Namespace) -> None:
    """Sub-agent worker mode: poll task.json → run → write result.json.

    Watches ``.asicode/subagents/<subagent_id>/task.json``; when a task arrives,
    runs it via DesignChatLoop and writes ``result.json`` back.  Stays alive in a
    loop so the orchestrator can reuse the worker for subsequent tasks (Ctrl-C
    to exit).

    Launched automatically by ``/orchestrate`` (auto_launch_terminal on macOS)
    or manually: ``asi --subagent --subagent-id <id> --provider ... --model ...``
    """
    global _REPO_ROOT
    repo_root = _resolve_repo_root(args.repo)
    _REPO_ROOT = repo_root

    agent_id = args.subagent_id
    if not agent_id:
        _print("--subagent-id is required with --subagent", _C["red"])
        sys.exit(1)

    # Capture the spawning parent PID ONCE at worker start. If the orchestrator
    # (headless path: this worker is its direct child) is SIGKILL'd,
    # poll_for_task's orphan check (_is_process_alive, cross-platform) detects
    # that this pid is dead and self-exits. Captured here (not per-call)
    # because the pid to watch never changes.
    #
    # getppid() is only correct on the direct-child launch path (headless
    # background spawn). On the macOS Terminal.app path (osascript → Terminal
    # → login shell → this worker) the parent is the login shell, NOT the
    # orchestrator — getppid() never changes when the orchestrator itself
    # dies, so orphan self-exit silently never fires and the worker idles up
    # to max_poll_s (24h). --orch-pid carries the orchestrator's actual PID
    # explicitly (set by asr_subagent_argv callers) so poll_for_task can
    # probe it directly instead of trusting getppid(). Falls back to the old
    # getppid()-based check when absent (manual launch, older orchestrator).
    _origin_ppid = os.getppid()
    _orch_pid = getattr(args, "orch_pid", 0) or None

    # Idle heartbeat writer: proves the WORKER PROCESS is alive while it is
    # between tasks (polling), independent of any task dir. The orchestrator's
    # _claim_reusable_worker reads this to judge a Terminal-launched worker's
    # (no PID handle) liveness BEFORE reuse — closing the gap where a hung
    # Terminal worker was optimistically reused and burned ipc_timeout_s. See
    # write_worker_idle_heartbeat / read_worker_idle_heartbeat_age.
    from external_llm.agent.subagent_ipc import (
        _IDLE_HEARTBEAT_INTERVAL_S,
        HEARTBEAT_INTERVAL_S,
        SubagentResult,
        build_subagent_prompt,
        partition_changed_files,
        poll_for_task,
        write_heartbeat,
        write_result,
        write_worker_exited_heartbeat,
        write_worker_idle_heartbeat,
    )

    _print(
        f"Sub-agent worker [{agent_id}] started, watching "
        f"{repo_root}/.asicode/subagents/{agent_id}/task.json",
        _C["teal"],
    )

    cancel_event = threading.Event()        # TASK-scope: abort the current task only
    shutdown_event = threading.Event()      # PROCESS-scope: exit the worker loop

    def _sigint_handler(sig, frame):
        # Ctrl-C is a process-level intent: abort the in-flight task AND exit.
        _print(f"\nSub-agent [{agent_id}] shutting down…", _C["yellow"])
        cancel_event.set()        # abort in-flight task immediately
        shutdown_event.set()      # exit the poll loop after the task unwinds

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Idle heartbeat: a long-lived daemon that writes worker.heartbeat.json
    # into the worker's OWN poll directory every _IDLE_HEARTBEAT_INTERVAL_S for
    # the ENTIRE process lifetime. This covers BOTH the between-tasks idle window
    # AND task execution (redundant with the per-task heartbeat, but harmless —
    # the idle heartbeat proves process liveness regardless of task state).
    # _claim_reusable_worker reads it to reject a hung Terminal-launched worker
    # (no PID handle) before reuse, so a dead worker is never re-dispatched.
    _idle_hb_stop = threading.Event()
    # Observability state for the idle heartbeat (Bug 4): mutated only by the
    # main poll loop below (single-threaded), read by this daemon thread. Not
    # lock-protected — advisory display data, not a correctness signal, and
    # simple int/str assignment is atomic enough under the GIL.
    _worker_start_ts = time.monotonic()
    _worker_stats = {"tasks_served": 0, "last_task_id": ""}

    def _write_idle_hb() -> None:
        write_worker_idle_heartbeat(
            repo_root, agent_id, pid=os.getpid(),
            tasks_served=_worker_stats["tasks_served"],
            last_task_id=_worker_stats["last_task_id"],
            uptime_s=time.monotonic() - _worker_start_ts,
        )

    def _idle_heartbeat_writer() -> None:
        while not _idle_hb_stop.wait(_IDLE_HEARTBEAT_INTERVAL_S):
            try:
                _write_idle_hb()
            except Exception:
                logging.getLogger(__name__).debug("idle heartbeat write failed", exc_info=True)

    # Write one immediately so a fresh heartbeat exists before the first poll.
    try:
        _write_idle_hb()
    except Exception:
        logging.getLogger(__name__).debug("initial idle heartbeat write failed", exc_info=True)
    threading.Thread(
        target=_idle_heartbeat_writer,
        name=f"ipc-idle-heartbeat-{agent_id}",
        daemon=True,
    ).start()

    # ── External cancel watcher ────────────────────────────────────────────
    # The orchestrator signals mid-task cancellation by writing a ``cancel.json``
    # sentinel into this worker's poll directory.  This PER-TASK daemon thread
    # polls for that sentinel WHILE a task runs and sets the task-scope
    # ``cancel_event`` — DesignChatLoop already checks ``cancel_event`` at the top
    # of every iteration (turn boundary) and aborts via ``AgentCancelled``.
    #
    # Lifecycle (B6): ``cancel.json`` is TASK-scoped, NOT process-scoped.  After
    # the task aborts and writes its error result, ``cancel_event`` is cleared and
    # the worker loops back to poll for the NEXT task — a single cancel no longer
    # kills a reusable worker (saving the ~5-8s respawn cost per cancelled task).
    # Only SIGINT, the ``shutdown.json`` sentinel, or orphan-detection exit the
    # worker.  The watcher is started PER-TASK (not as one long-lived thread) so a
    # stale sentinel cannot fire during idle polling between tasks (write_task also
    # clears a stale cancel.json as a belt-and-braces measure).
    from external_llm.agent.subagent_ipc import check_cancel_sentinel

    def _cancel_watcher(stop_flag: threading.Event) -> None:
        _cancel_path = os.path.join(
            repo_root, ".asicode", "subagents", agent_id, "cancel.json",
        )
        while not stop_flag.is_set() and not cancel_event.is_set():
            try:
                if check_cancel_sentinel(repo_root, agent_id):
                    _print(
                        f"\nSub-agent [{agent_id}] cancel signal received "
                        f"— aborting current task at the next turn boundary.",
                        _C["yellow"],
                    )
                    try:
                        os.unlink(_cancel_path)  # fire-once
                    except OSError:
                        logging.getLogger(__name__).debug("cancel sentinel unlink failed", exc_info=True)
                    cancel_event.set()
                    return
            except Exception:
                logging.getLogger(__name__).debug("cancel watcher poll failed", exc_info=True)
            # P27-2: stop_flag.wait(0.5) — wake immediately on shutdown instead of
            # sleeping the full poll interval (cancelable-sleep pattern; keeps the
            # 0.5s cancel-sentinel cadence while a task runs).
            stop_flag.wait(0.5)

    # ── Per-process LLM service cache (worker reuse optimization).
    # create_intelligent_service_from_env builds a client (auth handshake, model
    # resolution) every call — a few seconds each. A reused worker serves MANY
    # tasks, often with the SAME (provider, model, api_key), so re-creating it per
    # task defeats the worker-reuse (P3) goal of saving the ~5-8s respawn cost.
    # Cache the service keyed by (provider, model, api_key); a task that changes
    # any of these misses the cache and re-initializes. The cache holds AT MOST
    # one entry (the common case: all tasks use the same provider) — a different
    # key evicts the prior entry. api_key is normalized to "" so None/"" collide.
    _svc_cache: dict = {}  # {(provider, model, api_key): svc} — single-slot cache

    # Run tasks in a loop.  The worker stays alive across tasks so the orchestrator
    # can reuse it (no new Terminal/process per task).  Exit ONLY on: SIGINT
    # (shutdown_event), the shutdown.json sentinel, or orphan-detection — NOT on a
    # task-level cancel.json (B6: that aborts only the current task, then the
    # worker loops back to serve the next one).
    while not shutdown_event.is_set():
        # Reset the task-scope cancel before polling so a previous task's cancel
        # does not leak into the next poll (poll_for_task checks cancel_event and
        # returns None if it is set).
        cancel_event.clear()
        _print(f"[{agent_id}] Polling for task… (Ctrl-C to exit)", _C["muted"])
        task = poll_for_task(
            repo_root=repo_root,
            agent_id=agent_id,
            poll_interval_s=1.0,
            timeout_s=None,  # infinite — worker stays alive until killed
            cancel_event=cancel_event,
            expected_parent_pid=_origin_ppid,
            orchestrator_pid=_orch_pid,
        )
        if task is None or shutdown_event.is_set():
            break

        # Start THIS task's cancel watcher (per-task; stopped after the result is
        # written so it cannot fire during the next idle poll).
        _watcher_stop = threading.Event()
        threading.Thread(
            target=_cancel_watcher, args=(_watcher_stop,),
            name=f"ipc-cancel-{agent_id}-{task.task_id}", daemon=True,
        ).start()

        _print(f"[{agent_id}] Received task: {task.title}", _C["green"])
        _print(f"  files: {task.assigned_files}", _C["muted"])
        if task.description:
            _print(f"  description: {task.description[:200]}", _C["muted"])

        # Task payload can override provider/model/api_key; else use CLI args.
        provider = task.provider or getattr(args, "provider", "") or ""
        model = task.model or getattr(args, "model", "") or ""
        api_key = task.api_key or getattr(args, "api_key", "") or None
        max_turns = task.max_turns or getattr(args, "max_turns", 12) or 12

        printer = _ProgressPrinter(verbose=getattr(args, "verbose", False))
        ipc_result = None

        # ── Heartbeat: prove liveness so the orchestrator's wait_for_result
        # can distinguish a BUSY worker (long LLM/tool turn) from a DEAD one
        # (OOM/segfault) instead of burning the full ipc_timeout_s. A daemon
        # thread writes heartbeat.json (wall-clock ts) every HEARTBEAT_INTERVAL_S
        # into the task's OWN dir (same as result.json) so the orchestrator can
        # read it back. The thread is stopped once the result is written below.
        # Uses pid so a diagnostic can identify the heartbeating process.
        _hb_stop = threading.Event()
        # Shared progress state, updated by the wrapped stream_callback below and
        # read by the heartbeat thread so heartbeats carry "turn N, <tool>" hints
        # (F3). Plain dict writes/reads are GIL-atomic for independent keys; the
        # heartbeat is advisory, so a torn read across keys is harmless.
        _hb_state = {"turn": 0, "last_tool": ""}

        # Bind the task id NOW: the worker loop reassigns ``task`` for the next
        # task, and the heartbeat thread may still be mid-write after
        # _hb_stop.set() (wait→False race). Late-binding task.task_id would then
        # stamp a heartbeat with the NEXT task's id into its dir. A plain local
        # rebinding cannot pin the value (the closures read the CELL, rebound on
        # the next iteration) — so _heartbeat_writer/_hb_stream_cb bind
        # _hb_stop/_task_id/_hb_state/_printer as def-time default args (B023):
        # each iteration's thread/callback is pinned to THIS iteration's
        # objects, and a straggler thread heartbeats its own task dir instead
        # of stamping the next task's id/state.
        _task_id = task.task_id

        def _heartbeat_writer(_hb_stop=_hb_stop, _task_id=_task_id, _hb_state=_hb_state) -> None:
            while not _hb_stop.wait(HEARTBEAT_INTERVAL_S):
                try:
                    write_heartbeat(
                        repo_root, _task_id, pid=os.getpid(),
                        turn=_hb_state.get("turn", 0),
                        last_tool=_hb_state.get("last_tool", ""),
                    )
                except Exception:
                    logging.getLogger(__name__).debug("task heartbeat write failed", exc_info=True)

        # Write one immediately so a heartbeat exists before the first poll gap.
        try:
            write_heartbeat(repo_root, task.task_id, pid=os.getpid())
        except Exception:
            logging.getLogger(__name__).debug("initial task heartbeat write failed", exc_info=True)
        threading.Thread(
            target=_heartbeat_writer,
            name=f"ipc-heartbeat-{agent_id}-{task.task_id}",
            daemon=True,
        ).start()

        try:
            # ── DesignChatLoop-based execution (lighter than AgentLoop/router) ──
            from external_llm.intelligent_service import (
                create_intelligent_service_from_env,
            )
            # Reuse the LLM service across tasks when (provider, model, api_key)
            # is unchanged (see _svc_cache decl above). A reused worker serves
            # many tasks with the same provider, so re-initializing the service
            # per task would burn seconds of auth/handshake each time — defeating
            # the P3 reuse goal. Cache key is the resolved triple; api_key is
            # normalized so None/"" collide.
            _svc_key = (provider or "", model or "", api_key or "")
            _cached = _svc_cache.get(_svc_key)
            if _cached is not None:
                svc = _cached
            else:
                svc = create_intelligent_service_from_env(
                    provider or None, model or None, api_key=api_key or None,
                )
                if svc is not None:
                    # Evict any prior entry (single-slot cache: the common case
                    # is one provider per worker; a provider switch is rare).
                    _svc_cache.clear()
                    _svc_cache[_svc_key] = svc
            if svc is None:
                raise RuntimeError(
                    f"failed to initialize LLM service for sub-agent {agent_id}\n"
                    f"  --provider {provider or '(unset)'} --model {model or '(unset)'}"
                )

            from external_llm.agent.agent_loop_types import AgentCancelled
            from external_llm.agent.design_chat_loop import DesignChatLoop
            from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
            from external_llm.client import LLMMessage

            config = AgentConfig(
                model_name=svc.model or "",
                max_turns=max_turns,
                stream_callback=printer,
                consume_content_events=False,
                run_lint=True,
                run_tests=True,
                cancel_event=cancel_event,
                unrestricted_read=True,   # trusted local CLI
            )

            registry = ToolRegistry(repo_root, config)
            design_loop = DesignChatLoop(svc.llm_service.client, registry, svc.model)
            printer._start_spinner("")

            t0 = time.perf_counter()

            # Build the initial user message — mirroring the in-process path
            # (orchestrator._run_subagent). ``build_subagent_prompt`` prefers
            # ``task.predecessor_context`` (the richly-built task_text with
            # predecessor results + shared memory) over bare ``task.description``
            # and wraps it with ``task.original_request`` (the overall goal), so
            # dependent IPC subtasks no longer run "blind".
            messages = [LLMMessage(role="user", content=build_subagent_prompt(task))]

            # Wrap the printer so design-loop stream events also feed the heartbeat
            # progress state (F3): each LLM call bumps the turn counter, and a
            # tool_call start records the tool name. The orchestrator's wait_for_result
            # then surfaces "turn N, <tool>" instead of only elapsed time.
            _printer = printer
            def _hb_stream_cb(event: str, payload, _hb_state=_hb_state, _printer=_printer):
                try:
                    if event == "design_llm_call":
                        _hb_state["turn"] = int(_hb_state.get("turn", 0)) + 1
                    elif (
                        event == "design_tool_call" and isinstance(payload, dict) and payload.get("status") == "running"
                    ):
                        _hb_state["last_tool"] = str(payload.get("tool") or "")
                except Exception:
                    logging.getLogger(__name__).debug("heartbeat stream state update failed", exc_info=True)
                if _printer is not None:
                    try:
                        _printer(event, payload)
                    except Exception:
                        logging.getLogger(__name__).debug("subagent printer callback failed", exc_info=True)

            dc_result = design_loop.respond(
                messages,
                stream_callback=_hb_stream_cb,
                max_tool_iterations=max_turns,
            )

            elapsed = time.perf_counter() - t0

            # Collect diff via git for the result payload.
            diff = ""
            try:
                import subprocess as _sp
                diff = _sp.run(
                    ["git", "diff", "--stat", "HEAD", "--", *task.assigned_files],
                    cwd=repo_root, capture_output=True, text=True, timeout=10,
                    check=False,
                ).stdout.strip()
            except Exception:
                logging.getLogger(__name__).debug("subagent git diff failed", exc_info=True)

            # Map DesignChatResult → SubagentResult fields.
            # is_error takes PRECEDENCE over hit_max_iterations: a task that both
            # exhausted its turn budget AND then failed to generate its final
            # response must surface as an error (not "max_turns"), otherwise the
            # orchestrator misreads a generation failure as "budget exhausted /
            # partial progress" and picks the wrong retry strategy. When
            # DesignChatLoop reaches the max-iterations tail it leaves is_error
            # False unless the final-response call itself failed (in which case
            # it now sets is_error=True) — so a clean budget-exhaustion still
            # reports "max_turns".
            if dc_result.is_error:
                status = "error"
            elif dc_result.hit_max_iterations:
                status = "max_turns"
            else:
                status = "success"
            final_message = dc_result.content or ""
            turns = dc_result.total_llm_calls or len(dc_result.tool_calls_made) or 1
            # DesignChatLoop does not itself track applied patches — derive the
            # file list from an UNSCOPED ``git status`` and partition into
            # in-scope (applied_patches) vs out-of-scope (unassigned_changes)
            # so the orchestrator's diff cross-verification AND scope-violation
            # review both have full visibility (B5). Previously the call was
            # scoped to assigned_files, hiding any out-of-scope write.
            patches, unassigned = partition_changed_files(repo_root, task.assigned_files)
            error_msg = dc_result.content if dc_result.is_error else ""

            ipc_result = SubagentResult(
                task_id=task.task_id,
                status=status,
                final_message=final_message,
                diff=diff,
                turns=turns,
                applied_patches=patches,
                error=error_msg,
                epoch=task.epoch,
                unassigned_changes=unassigned,
            )

            printer._stop_spinner()
            _print(
                f"[{agent_id}] Task complete: {status} ({turns} turns, {elapsed:.1f}s)",
                _C["green"] if status == "success" else _C["yellow"],
            )

        except AgentCancelled:
            # Task-level cancel (cancel.json sentinel, B6): abort ONLY this task,
            # write a cancelled result, then loop back to poll for the next one.
            # The worker process stays alive (shutdown_event is NOT set by a
            # task-scope cancel) so the orchestrator can reuse it.
            _cancel_msg = f"Sub-agent task '{task.task_id}' cancelled by orchestrator"
            logging.getLogger(__name__).info(
                "Sub-agent %s task %s cancelled (task-scope; worker stays alive)",
                agent_id, task.task_id,
            )
            # Report partial edits even when cancelled: a mid-task abort can
            # leave half-applied changes on disk. Without partition_changed_files
            # here the result carries applied_patches=[]/unassigned_changes=[],
            # leaving the orchestrator's diff cross-verification AND the B5 scope
            # signal blind to whatever the worker wrote before it stopped — and
            # hiding the exact set the orchestrator must revert (turn 13114 bug 1).
            try:
                patches, unassigned = partition_changed_files(repo_root, task.assigned_files)
            except Exception:
                patches, unassigned = [], []
            ipc_result = SubagentResult(
                task_id=task.task_id,
                status="cancelled",
                final_message=_cancel_msg,
                applied_patches=patches,
                error=_cancel_msg,
                epoch=task.epoch,
                unassigned_changes=unassigned,
            )
            try:
                printer._stop_spinner()
            except Exception:
                logging.getLogger(__name__).debug("subagent cancel spinner stop failed", exc_info=True)
            _print(f"[{agent_id}] Task cancelled (worker staying alive).", _C["yellow"])
        except Exception as e:
            logging.getLogger(__name__).exception(
                "Sub-agent %s execution failed", agent_id,
            )
            # Drop the cached LLM service: an exception may mean it is broken
            # (expired auth, dropped connection after a long idle period) rather
            # than the task itself being bad. Without this, every subsequent
            # reused-worker task would hit the same dead client and fail
            # immediately instead of reinitializing. Cheap and safe — the next
            # task just re-creates it (a few seconds, only once).
            _svc_cache.clear()
            # Same rationale as the cancelled branch: a task that crashed mid-run
            # may have partial edits on disk. Report them so the orchestrator can
            # attribute, cross-verify, and revert as needed.
            try:
                patches, unassigned = partition_changed_files(repo_root, task.assigned_files)
            except Exception:
                patches, unassigned = [], []
            ipc_result = SubagentResult(
                task_id=task.task_id,
                status="error",
                applied_patches=patches,
                error=str(e),
                epoch=task.epoch,
                unassigned_changes=unassigned,
            )
            try:
                printer._stop_spinner()
            except Exception:
                logging.getLogger(__name__).debug("subagent error spinner stop failed", exc_info=True)
            _print(f"[{agent_id}] Task failed: {e}", _C["red"])

        # Always write a result so the orchestrator's wait_for_result unblocks.
        # write_result itself can raise (I/O error, serialization failure); if it
        # does, the exception would propagate out of the poll loop and KILL the
        # worker — result.json would never appear and the orchestrator would burn
        # the full ipc_timeout_s before failing. Guard it so the "always writes a
        # result" contract holds even then: retry once with a minimal error result.
        if ipc_result is not None:
            try:
                write_result(repo_root, ipc_result)
                _print(f"[{agent_id}] Result written.", _C["muted"])
            except Exception as _wr_exc:
                logging.getLogger(__name__).exception(
                    "Sub-agent %s: result write failed (%s); retrying with a "
                    "minimal error result", agent_id, _wr_exc,
                )
                try:
                    write_result(repo_root, SubagentResult(
                        task_id=ipc_result.task_id,
                        status="error",
                        final_message=f"result write failed: {_wr_exc}",
                        error=f"result write failed: {_wr_exc}",
                        epoch=ipc_result.epoch,
                    ))
                    _print(f"[{agent_id}] Minimal error result written.", _C["muted"])
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Sub-agent %s: minimal error-result write also failed; "
                        "orchestrator will time out.", agent_id,
                    )

        # Record this task for the idle heartbeat's observability fields (Bug
        # 4) before the worker goes back to idle polling — reflects in the
        # NEXT periodic write (or the following task's completion), same as
        # the pre-existing idle-heartbeat cadence.
        _worker_stats["tasks_served"] += 1
        _worker_stats["last_task_id"] = task.task_id

        # Stop this task's heartbeat thread now that a result has been written
        # (or the write is irrecoverably failed). Prevents the daemon thread
        # from carrying a stale heartbeat into the next idle poll cycle.
        _hb_stop.set()
        # Stop this task's cancel watcher (B6): the task is done (or aborted), so
        # there is nothing to cancel. The worker loops back to poll; a fresh
        # watcher is started for the next task.
        try:
            _watcher_stop.set()
        except Exception:
            logging.getLogger(__name__).debug("subagent watcher stop failed", exc_info=True)

    # Stop the idle-heartbeat daemon and mark the heartbeat "exited" BEFORE the
    # process actually terminates. Without this, the last "idle" heartbeat (up
    # to _IDLE_HEARTBEAT_INTERVAL_S stale) still reads as fresh to
    # _claim_reusable_worker for up to ipc_heartbeat_stale_s (120s default)
    # after this process is gone, so a dead worker gets re-claimed and burns
    # the orchestrator's full ipc_timeout_s on the next dispatch.
    _idle_hb_stop.set()
    try:
        write_worker_exited_heartbeat(repo_root, agent_id, pid=os.getpid())
    except Exception:
        logging.getLogger(__name__).debug("subagent exited-heartbeat write failed", exc_info=True)

    _print(f"Sub-agent [{agent_id}] stopped.", _C["teal"])

# ─── Cross-module cycle with asi.py (bottom-of-file imports, symmetric) ─────
# asi.py imports THIS module at the very bottom of asi.py (barrel re-export),
# and this module imports asi HERE — also at the bottom of the file. Because
# both imports sit at the bottom, the cycle resolves in EITHER import order:
# whichever module loads first is fully initialized (every name used below is
# defined above this point in the other file) by the time the other module's
# bottom import runs. Moving this block up (e.g. to the top, as it was before)
# broke `import external_llm.repl.repl_impl` first with a circular-import
# ImportError (`cannot import name '_AUTO_CONTINUE_DELAY' from partially
# initialized module 'external_llm.repl.repl_impl'`). The names below are used
# only at runtime (function bodies) — nothing between the top and here depends
# on them. Guarded by scripts/check_standalone_imports.py.
import asi  # noqa: E402 — bottom-of-file (symmetric cycle, see comment above)
from asi import (  # noqa: E402 — bottom-of-file (symmetric cycle)
    _API_KEY_ENV_MAP,
    _C,
    _CONSOLE_MARGIN,
    _EVENT_LABELS,
    _FAILURE_PATTERNS_SUBCOMMANDS,
    _INSIGHTS_SUBCOMMANDS,
    _INTERRUPT_RESUME_INSTRUCTION,
    _PAUSED_HINT,
    _RICH,
    _SEQ_W,
    _SILENT_EVENTS,
    _SLASH_ALIASES,
    _abbrev_tokens,
    _bar_panel,
    _build_interrupt_note,
    _changed_files_since,
    _checkpoint_changed_files,
    _console_width,
    _copy_to_clipboard,
    _create_llm_client_for,
    _disable_bracketed_paste,
    _drain_stdin,
    _enable_bracketed_paste,
    _ensure_console_imported,
    _ensure_out_console_imported,
    _extract_tool_cmd,
    _fmt_elapsed,
    _git,
    _git_baseline,
    _handle_insights_archive,
    _handle_terminal_resize,
    _kick_embedding_model_warmup,
    _load_prompt_toolkit,
    _newest_checkpoint_id,
    _pip_install,
    _print,
    _print_banner,
    _print_dep_status,
    _print_run_change_summary,
    _print_session_summary,
    _prompt_auth_retry_key,
    _relativize_repo_paths,
    _render_help,
    _render_run_diff,
    _render_status,
    _resolve_model_interactive,
    _restart_cli,
    _rich_markdown_cls,
    _rotate_cli_history_if_needed,
    _run_changed_stats,
    _select_preview_lines,
    _seq_pad,
    _shimmer_beam,
    _shimmer_style_for,
    _ShimmerSpinner,
    _SlashCommandCompleter,
    _strip_ansi,
    _tool_running_filter,
    _undo_run_changes,
    _undo_via_checkpoint,
)
