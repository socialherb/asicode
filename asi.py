#!/usr/bin/env python3
"""
asicode Interactive CLI

Interactive CLI that connects directly to the engine without a FastAPI server.

Usage:
    python asi.py                        # Start REPL in the current directory
    python asi.py --repo /path/to/repo   # Use a specific repository
    python asi.py -p "fix the bug"       # Run a single request then exit
    python asi.py --provider anthropic --model claude-sonnet-4-6

Environment variables:
    EXTERNAL_LLM_PROVIDER  = anthropic / openai / google / deepseek / ollama
    EXTERNAL_LLM_MODEL     = model name (optional)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / ...
"""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import logging
import logging.handlers
import os
import select as _select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The runtime import is lazy (inside _load_checkpoint_store) — this exists
    # only so that helper's return annotation can name the class.
    from external_llm.agent.checkpoint_store import CheckpointStore

# When launched directly (``python asi.py``) the file runs as ``__main__``, so
# ``import asi`` inside ``external_llm/repl/repl_impl.py`` would re-execute this
# file as a *separate* module ``asi`` and hit a circular-import ImportError on the
# barrel re-export at the bottom. Alias the running module object under ``asi``
# early so the one-way cycle resolves to THIS object (no re-execution). A no-op
# for the normal entry point ``from asi import main`` where ``__name__ == "asi"``.
if __name__ == "__main__":
    sys.modules.setdefault("asi", sys.modules[__name__])

import contextlib

from external_llm.agent.config.thresholds import config as _cfg

# Terminal row-level write serialization: the ticker thread's \r\x1b[2K re-render and the log handler
# emit to the same tty concurrently would mix lines (WARNING attaches to the right of the spinner row,
# row-erase truncates in-flight log). Both acquire this lock when writing.
# The lock/row_pending flag lives in a shared module — asi runs as __main__, while the
# collaborate StreamingDisplay loads as a library, so a neutral module is needed for both to see
# the same object. This ensures that the live row on the collaborate screen does not get
# WARNING interleaved.
from external_llm.agent.terminal_coordination import (
    TERM_WRITE_LOCK as _TERM_WRITE_LOCK,
)
from external_llm.agent.terminal_coordination import (
    row_pending as _term_row_pending,
)
from external_llm.agent.terminal_coordination import (
    set_row_pending as _set_term_row_pending,
)

# ── Per-provider known model list ──────────────────────────────────────────────
# Single source: external_llm/model_catalog.py — shared with the kp verify
# tool and the webapp picker endpoint (three hand-synced copies drifted apart
# before it existed). Edit the catalog module, not these names.
from external_llm.model_catalog import (
    KNOWN_MODELS as _KNOWN_MODELS,
)
from external_llm.model_catalog import (
    MODEL_ALIASES as _MODEL_ALIASES,
)


def _silence_socks_dependency_warning() -> None:
    """Pre-empt urllib3's "PySocks is not installed" warning leaking into the banner.

    ``requests`` already silences it — ``requests/__init__.py`` installs
    ``simplefilter("ignore", DependencyWarning)`` before importing
    ``requests.adapters``, whose ``from urllib3.contrib.socks import ...`` is what
    emits the warning (PySocks is an optional extra; urllib3 warns, then re-raises
    ImportError, which requests catches). But that filter is only live for the
    ~10ms between the two lines, and ``warnings.catch_warnings()`` is *not*
    thread-safe: it snapshots the global filter list on entry and restores that
    snapshot on exit, so any thread leaving such a block inside the window drops
    requests' filter and the warning goes straight to stderr — landing inside the
    startup banner.

    asicode is exactly that shape: ``_kick_embedding_model_warmup()`` starts the
    emb-warmup thread (importing sentence_transformers/torch runs ~20
    ``catch_warnings`` blocks over several seconds) three lines before the main
    thread imports ``requests`` via ``external_llm.client``. Measured collision
    rate ~4% per startup, so it surfaces occasionally and confusingly.

    Installing the filter here — at import time, before any thread exists — makes
    it part of every later snapshot, so a restore can no longer drop it. Matched
    by message rather than by ``urllib3.exceptions.DependencyWarning``
    deliberately: importing urllib3 only to name the class costs ~80ms on the
    cold-start path, more than asi's own module import.
    """
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="SOCKS support in urllib3 requires",
        category=Warning,
    )


_silence_socks_dependency_warning()

# ── prompt_toolkit: optional dependency — deferred import, only for REPL/interactive input ──────
# Non-interactive paths (--subagent, --prompt, --json) never load it, saving ~77ms
# (half of the module import cost) on cold start. _load_prompt_toolkit() binds on first entry.
# Sub-agent workers are spawned per-process (orchestrator.py), so this saving compounds.
_PROMPT_TOOLKIT_AVAILABLE = False
PromptSession = None  # type: ignore[assignment,misc]   # bound by _load_prompt_toolkit()
Completion: Any = None  # used by _SlashCommandCompleter methods; bound by _load_prompt_toolkit()
KeyBindings = None  # type: ignore[assignment,misc]   # used when configuring _collect_input session
InMemoryHistory = None  # type: ignore[assignment,misc]
_PtStyle = None  # type: ignore[assignment,misc]   # (previously missing fallback — latent NameError fixed)
patch_stdout = None  # type: ignore[assignment,misc]   # used by _collect_input to coordinate background writes with the live prompt


def _load_prompt_toolkit() -> bool:
    """Lazy-import prompt_toolkit and bind module globals.

    Called once on first entry into REPL or _collect_input. On success,
    _PROMPT_TOOLKIT_AVAILABLE becomes True and PromptSession/Completion/KeyBindings/InMemoryHistory/_PtStyle/patch_stdout
    globals are filled with real classes. On failure, False (callers fall back to input()).
    Idempotent — returns True immediately if already loaded.
    """
    global _PROMPT_TOOLKIT_AVAILABLE, PromptSession, Completion, KeyBindings, InMemoryHistory, _PtStyle, patch_stdout
    if _PROMPT_TOOLKIT_AVAILABLE:
        return True
    try:
        from prompt_toolkit import PromptSession as _PS  # noqa: N814 — private lazy-import alias
        from prompt_toolkit.completion import Completion as _Cmpl
        from prompt_toolkit.history import InMemoryHistory as _IMH  # noqa: N814 — private lazy-import alias
        from prompt_toolkit.key_binding import KeyBindings as _KB  # noqa: N814 — private lazy-import alias
        from prompt_toolkit.patch_stdout import patch_stdout as _PatchStdout  # noqa: N812 — private lazy-import alias
        from prompt_toolkit.styles import Style as _Style
    except ModuleNotFoundError:  # pragma: no cover — prompt_toolkit absent (optional dep; REPL tests run with it)
        return False
    PromptSession = _PS
    Completion = _Cmpl
    KeyBindings = _KB
    InMemoryHistory = _IMH
    _PtStyle = _Style
    patch_stdout = _PatchStdout
    _PROMPT_TOOLKIT_AVAILABLE = True
    return True


# CLI history rotation: FileHistory loads the whole file at startup, so cap it.
_CLI_HISTORY_ROTATE_AT = 20000  # lines — trigger rotation above this
_CLI_HISTORY_KEEP = 10000  # lines — how many recent lines to keep after rotation

# ── CLI color theme (Catppuccin Mocha — same as web UI palette) ──────────────────
_C: dict[str, str] = {
    "blue": "#89b4fa",  # primary / read series
    "sky": "#89dceb",  # search / git series
    "mauve": "#cba6f7",  # plan / PLANNER section
    "peach": "#fab387",  # patch / turn indicator
    "green": "#a6e3a1",  # success / test
    "red": "#f38ba8",  # error / exec series
    "yellow": "#f9e2af",  # warning / rate-limit
    "teal": "#94e2d5",  # teal / misc
    "text": "#cdd6f4",  # default text
    "muted": "#6c7086",  # secondary / dim info
    "border": "#313244",  # separator (very fine)
}


def _rotate_cli_history_if_needed(path: str) -> None:
    """Cap unbounded CLI history growth.

    prompt_toolkit's ``FileHistory`` loads the ENTIRE file into memory at
    startup, so a long-lived session's ``.asicode/cli_history`` grows without
    bound (observed 40k+ lines / 2.6MB).  When the line count exceeds
    ``_CLI_HISTORY_ROTATE_AT``, rewrite the file keeping only the most recent
    ``_CLI_HISTORY_KEEP`` lines, snapped forward to the next ``# <ts>`` entry
    boundary so no multi-line ``+...`` entry is split (FileHistory treats any
    non-``+`` line as an entry separator; starting mid-entry would orphan lines
    into a spurious first entry).  Non-critical: any failure is swallowed and
    the full history is used as-is.
    """
    import os as _os

    with contextlib.suppress(OSError):  # history rotate is best-effort file I/O
        if not path or not _os.path.exists(path):
            return
        with open(path, "rb") as f:
            lines = f.readlines()
        if len(lines) <= _CLI_HISTORY_ROTATE_AT:
            return
        tail = lines[-_CLI_HISTORY_KEEP:]
        # Snap to the first entry boundary ('# <ts>' header) so we never start
        # inside a multi-line entry.
        start = 0
        for i, ln in enumerate(tail):
            if ln.startswith(b"# "):
                start = i
                break
        kept = tail[start:]
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.writelines(kept)
        _os.replace(tmp, path)


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate two #rrggbb colors at t∈[0,1], return #rrggbb."""
    h1, h2 = c1.lstrip("#"), c2.lstrip("#")
    r = round(int(h1[0:2], 16) * (1 - t) + int(h2[0:2], 16) * t)
    g = round(int(h1[2:4], 16) * (1 - t) + int(h2[2:4], 16) * t)
    b = round(int(h1[4:6], 16) * (1 - t) + int(h2[4:6], 16) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Claude Code style shimmer (light beam scans left↔right over text) ──────────
_SHIMMER_BASE = _C.get("text", "#cdd6f4")  # base color outside beam
_SHIMMER_HI = "#ffffff"  # beam center (brightest)
_SHIMMER_SPEED = 7.0  # beam round-trip speed factor (higher = faster)


def _shimmer_beam(n: int, elapsed: float) -> tuple[int, int]:
    """Return (beam center index, beam radius) for text of length n.

    Uses a triangle wave to sweep left→right→left — the beam changes direction at
    both ends of the text for a smooth scan. Shared computation between Rich and non-Rich paths."""
    if n < 4:
        return n // 2, 0
    beam_w = max(3, n // 3)  # beam width (1/3 of text, minimum 3)
    span = n + beam_w
    phase = (elapsed * _SHIMMER_SPEED) % (span * 2)
    # Triangle wave: sweep left→right on the first half, right→left on the second
    center = (phase - beam_w / 2) if phase < span else (span - (phase - span) - beam_w / 2)
    return round(center), beam_w


def _shimmer_style_for(idx: int, center: int, beam_w: int) -> str:
    """Character color (#rrggbb) based on beam distance from index — Rich style string."""
    d = abs(idx - center)
    if d >= beam_w:
        return _SHIMMER_BASE
    t = 1.0 - (d / beam_w)  # 1=center(bright), 0=edge(base)
    return _lerp_color(_SHIMMER_BASE, _SHIMMER_HI, t * t)  # squared for smooth edge falloff


def _render_shimmer_text(text: str, elapsed: float):
    """Return Rich Text with beam shimmer applied (body text only, no spin glyph).

    Used to apply shimmer to static lines (banner titles, etc.). Shares the body-painting
    logic with _ShimmerSpinner.render — both call this helper for DRY.
    Short/blank text skips beam calculation and returns base color directly."""
    from rich.text import Text

    n = len(text)
    if n < 4 or not text.strip():
        return Text(text, style=_SHIMMER_BASE)
    center, beam_w = _shimmer_beam(n, elapsed)
    out = Text()
    for i, ch in enumerate(text):
        if ch == " ":
            out.append(" ")
            continue
        out.append(ch, style=_shimmer_style_for(i, center, beam_w))
    return out


# Per-model context windows: resolved via context_budget._resolve_context_limit
# (single source of truth — exact-match table for GLM/Qwen/Kimi/MiniMax/etc. +
# Ollama /api/show dynamic query + 1M fallback). Do NOT re-introduce a local
# prefix table here: a previous _CLOUD_CONTEXT_TOKENS dict omitted glm/qwen/
# ollama, leaving _ctx_budget=0 and silently disabling /general-mode compression
# for exactly the models that need it most.


def _enable_bracketed_paste() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?2004h")
        sys.stdout.flush()


def _disable_bracketed_paste() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()


# Pause/Resume: keyword matching approach is deprecated.
# On ESC interrupt, tool-loop history is preserved as an interrupt note in the session (_build_interrupt_note),
# and the next input is passed directly to the design chat LLM, which decides whether to resume.


# _check_clipboard_image and _extract_images_from_input have been moved to
# external_llm/image_utils.py.


def _drain_stdin(timeout: float = 0.05) -> None:
    """Drain all remaining bytes from stdin. Switch to non-canonical mode, drain, then restore."""
    import copy as _copy
    import termios as _termios

    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return
    with contextlib.suppress(OSError, ValueError):  # termios.error is an OSError subclass
        old = _copy.deepcopy(_termios.tcgetattr(fd))
        new = _termios.tcgetattr(fd)
        new[3] &= ~(_termios.ICANON | _termios.ECHO)
        new[6][_termios.VMIN] = 0
        new[6][_termios.VTIME] = 0
        _termios.tcsetattr(fd, _termios.TCSANOW, new)
        try:
            while _select.select([fd], [], [], timeout)[0]:
                os.read(fd, 4096)
        finally:
            _termios.tcsetattr(fd, _termios.TCSANOW, old)


# ── rich is optional: plain text fallback when absent ──────────────────────────
_CONSOLE_MARGIN = 4  # left/right whitespace (spaces)
# Left margin for INFO/WARNING logs (_log_console). Uses the same value as _CONSOLE_MARGIN
# so timestamps/level labels align vertically with body output (separator, tree-sitter, eslint, etc.).
_LOG_MARGIN = _CONSOLE_MARGIN
# Tool-call sequence number column width. Numbers inside brackets are rendered as-is
# "[1]"/"[10]"/"[100]" (no padding between brackets), and missing width is filled with _seq_pad
# **after** the "]". This means "]" shifts with digit count, but the ✓/✗/○ icon column
# always stays at a fixed alignment. 3 = alignment holds up to 999 (beyond that, 1 char shift).
_SEQ_W = 3


def _seq_pad(plain_tag: str) -> str:
    """ "[N]"/"[·]" token padding that fills after ']' to keep the icon (✓/✗/○) column at fixed width.

    Numbers inside brackets are left untouched (=[1]/[10]/[100]), only the space after "]"
    is padded, avoiding extra spaces around the number. plain_tag is the raw token without
    color/dim codes, e.g. "[1]".
    """
    return " " * max(0, (_SEQ_W + 2) - len(plain_tag))


class _MarginIO:
    """Stream wrapper that prepends `margin` spaces at the start of each line.

    Looks up ``sys.<stream_name>`` fresh on every write instead of capturing
    the stream object at construction time. This matters because
    prompt_toolkit's ``patch_stdout()`` works by reassigning the ``sys.stdout``/
    ``sys.stderr`` *names* to a proxy for the duration of an active prompt —
    a wrapper that captured the original stream object at import time would
    keep writing straight past that proxy, silently defeating patch_stdout
    for every _print()/log call made from a background thread while a prompt
    is being read.
    """

    def __init__(self, stream_name: str, margin: int = _CONSOLE_MARGIN):
        self._stream_name = stream_name
        self._pad = " " * margin
        self._bol = True  # beginning-of-line flag

    @property
    def _s(self):
        return getattr(sys, self._stream_name)

    def reset_bol(self) -> None:
        """Force beginning-of-line state (call after spinner/live display stops)."""
        self._bol = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        out: list[str] = []
        for line in data.splitlines(keepends=True):
            if self._bol and line and line[0] not in ("\n", "\r"):
                out.append(self._pad)
                self._bol = False
            out.append(line)
            if line and line[-1] == "\n":
                self._bol = True
        return self._s.write("".join(out))

    def flush(self):
        self._s.flush()

    def fileno(self):
        return self._s.fileno()

    def isatty(self):
        return getattr(self._s, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self._s, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._s, "errors", "strict")


# Lazy rich.console import: detect availability WITHOUT loading the ~29-40ms stack.
# Console instances are created on first use via the three _ensure_*_console_imported()
# builders, sharing the import/fallback ceremony through _import_rich_console().
# This preserves the --version/--help fast path (no rich.console loaded) while keeping
# the spinner/Live, RichHandler, and stdout-Markdown consoles intact.

_RICH = _importlib_util.find_spec("rich.console") is not None
_console_width = 0  # lazy: set by _ensure_console_widths()
_log_console_width = 0  # lazy: set by _ensure_console_widths()
_console = None  # lazy: _ensure_console_imported() creates on first use
_margin_stderr = None  # lazy: _ensure_log_console_imported() creates on first use
_log_console = None  # lazy: _ensure_log_console_imported() creates on first use
_console_widths_ready = False


def _ensure_console_widths() -> None:
    """Seed both module-level console widths from the terminal size (idempotent).

    MUST run before any Console(...) construction. Rich treats ``width=0`` as a
    zero-column viewport and renders NOTHING — not merely narrow output, but
    zero lines — so a Console built while these are still 0 silently swallows
    everything printed through it.

    This used to be free: both widths were computed at module import, next to the
    two eager ``Console(...)`` calls. Once Console creation went lazy, each
    ``_ensure_*_imported()`` became a possible FIRST toucher, so the widths can
    no longer live in only one of them. (Regression this guards: the stdout
    ``_out_console`` is normally built first — banner/help/status/_print — and
    ``_ensure_console_imported()``, which owned the width, only runs when the
    first spinner starts. Every startup line vanished.)

    ``_SIGWINCH`` also assigns ``_console_width`` directly; recomputing here
    afterwards just re-reads the same live terminal size, so the two agree.
    """
    global _console_width, _log_console_width, _console_widths_ready
    if _console_widths_ready:
        return
    import shutil as _shutil

    _cols = _shutil.get_terminal_size().columns
    _console_width = max(40, _cols - _CONSOLE_MARGIN * 2)
    # _log_console_width: MarginIO only adds left _CONSOLE_MARGIN, so right margin removal is unnecessary.
    # → terminal_width - left_margin makes the log line fill the terminal exactly.
    _log_console_width = max(40, _cols - _LOG_MARGIN)
    _console_widths_ready = True


def _import_rich_console() -> type | None:
    """Import the rich ``Console`` class, or return None when Rich is unavailable.

    Single source for the lazy-import ceremony shared by the three
    ``_ensure_*_console_imported()`` builders: the ``_RICH`` pre-check, the width
    seeding (width=0 would make a Console render nothing at all), and the
    broken-install fallback (clears ``_RICH`` so the ``_RICH and console`` call
    sites take the plain-text branch instead of dereferencing None — this
    restores the pre-lazy meaning of _RICH ("import actually succeeded"), which
    find_spec alone cannot promise).
    """
    global _RICH
    if not _RICH:
        return None
    _ensure_console_widths()
    try:
        from rich.console import Console

        return Console  # noqa: TRY300
    except ImportError:
        _RICH = False
        logging.getLogger(__name__).debug("rich.console import failed — falling back to plain output")
        return None


def _make_rich_console(file, width: int):
    """Create a rich Console bound to *file*, or None when Rich is unavailable.

    Single source for the console-construction step shared by the
    ``_ensure_*_console_imported()`` builders (after ``_import_rich_console()``
    has already decided Rich availability).  Returns None when Rich is not
    importable, so callers keep their module-level console slot untouched.
    """
    console_cls = _import_rich_console()
    if console_cls is None:
        return None
    return console_cls(file=file, width=width, force_terminal=True)


def _ensure_console_imported() -> None:
    """Lazily create _console (spinner/Live) on first use. No-op if Rich unavailable."""
    global _console
    if _console is not None:
        return
    # _console: for spinner/Live only — uses cursor-movement ANSI escapes, so MarginIO is not applicable.
    _console = _make_rich_console(sys.stderr, _console_width)


def _ensure_log_console_imported() -> None:
    """Lazily create _log_console (RichHandler) on first use. No-op if Rich unavailable."""
    global _log_console, _margin_stderr
    if _log_console is not None:
        return
    # _log_console: for RichHandler only — wrapped in _margin_stderr(MarginIO) to align INFO logs
    # from col 0 → col _LOG_MARGIN. Unlike spinner/Live, it does not use cursor-movement escapes,
    # so left-margin injection is safe. (_margin_stderr.reset_bol() on spinner→log transition.)
    # _margin_stderr is committed only when Rich actually imported, so the no-Rich
    # no-op contract (both slots stay None) is preserved.
    margin = _MarginIO("stderr", _LOG_MARGIN)
    console = _make_rich_console(margin, _log_console_width)
    if console is not None:
        _margin_stderr = margin
        _log_console = console


RichHandler = None  # type: ignore[assignment] — set lazily in _setup_logging()


class _ShimmerSpinner:
    """Claude Code style shimmer spinner — beam scans left↔right over text.

    Does not load in environments without Rich (guarded by _RICH in __init__),
    so this class is Rich-dependent. In addition to the spinning glyph (◴◷◶◵ circle quadrants, clockwise),
    it computes the beam position via triangle wave each render frame and paints text in per-char
    interpolated colors. Rich's Live(refresh_per_second=12) calls __rich_console__ → render(time)
    each frame, so animation runs without a separate thread.
    """

    def __init__(self, text: str, style: str, frames=None, interval: float = 130.0):
        from rich.spinner import Spinner
        from rich.text import Text

        self._Text = Text
        self._spinner = Spinner("dots", text=text, style=style)
        self._spinner.frames = frames or ["◴", "◷", "◶", "◵"]
        self._spinner.interval = interval
        self._spin_style = style

    # Rich renderable protocol: Live calls this every frame
    def __rich_console__(self, console, options):
        yield self.render(console.get_time())

    def __rich_measure__(self, console, options):
        from rich.measure import Measurement

        text = self.render(0)
        return Measurement.get(console, options, text)

    def render(self, time: float):
        sp = self._spinner
        if sp.start_time is None:
            sp.start_time = time
        frame_no = (time - sp.start_time) / (sp.interval / 1000.0)
        glyph = sp.frames[int(frame_no) % len(sp.frames)]
        _rich_text_cls = self._Text
        # ── Rotating glyph + body text ──
        frame = _rich_text_cls(glyph, style=self._spin_style or "")
        body = sp.text
        if not body:
            return frame
        plain = body.plain if isinstance(body, _rich_text_cls) else str(body)
        if not plain.strip():
            return _rich_text_cls.assemble(frame, " ", body)  # type: ignore[arg-type]  # RenderableType includes Text; assemble accepts Text
        # ── Leading spaces go before the glyph ──
        # If the body has indent like "      thinking", move those spaces before the rotating
        # glyph so the glyph aligns with the indent column (the ✓ column of completion lines "  [N] ✓").
        # Shimmer interpolation applies only to the body stripped of leading spaces.
        stripped = plain.lstrip(" ")
        indent = plain[: len(plain) - len(stripped)]
        out = _render_shimmer_text(stripped, time)
        if indent:
            return _rich_text_cls.assemble(_rich_text_cls(indent), frame, " ", out)
        return _rich_text_cls.assemble(frame, " ", out)

    # _update_spinner delegates to update() to preserve start_time while replacing text
    def update(self, *, text: str = "", style=None):
        if text:
            self._spinner.text = self._Text.from_markup(text) if isinstance(text, str) else text
        if style:
            self._spin_style = style
            self._spinner.style = style

    @property
    def start_time(self):
        return self._spinner.start_time

    @start_time.setter
    def start_time(self, v):
        self._spinner.start_time = v


# Add project root to sys.path (works regardless of where it's executed from)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _handle_terminal_resize(_signum=None, _frame=None) -> None:
    """SIGWINCH: Update Rich console width to new terminal size.

    Console width is fixed at import time; without a handler, resizing the window
    mid-session causes wrapping at the old width. Updating the width property alone
    applies the new width to subsequent prints (the input line itself is handled by prompt_toolkit).

    Additionally: when the width shrinks, an active Rich Live spinner message can wrap from
    1 line to 2, but transient=True Live only moves the cursor up by the "previous line count (1)",
    leaving residual lines that accumulate and overlap. Stopping the active spinner's Live after
    the width update makes transient clear the phantom lines and joins the refresh thread
    (eliminating contention). The instance is set to None while _spinner_running stays True —
    the next _update_spinner tick re-calls _spawn_rich_live with the new width for a clean restart.
    Reuses the _TERM_WRITE_LOCK serialization pattern from _suspend_live_for_log
    (log handler emit ↔ Live stop contention prevention) to also prevent signal handler ↔ refresh thread contention.
    """
    with contextlib.suppress(OSError):  # get_terminal_size on a broken tty
        import shutil as _sh_wz

        cols = _sh_wz.get_terminal_size((80, 24)).columns
        if _console is not None:
            _console.width = max(40, cols - _CONSOLE_MARGIN * 2)
        if _log_console is not None:
            _log_console.width = max(40, cols - _LOG_MARGIN)
        if _out_console is not None:
            _out_console.width = max(40, cols - _CONSOLE_MARGIN * 2)
        # Global width is only referenced by _build_interrupt_note(textwrap.fill), but
        # without updating it, an interrupt note after resize wraps at the old width.
        global _console_width
        _console_width = max(40, cols - _CONSOLE_MARGIN * 2)
        # Prevent residual lines when spinner wraps due to width reduction: stop
        # the active Live to clear transient remnants; the next ticker tick re-creates it at new width.
        _sp = _active_spinner_printer
        if _sp is not None and _sp._spinner_live is not None:
            with _TERM_WRITE_LOCK:
                with contextlib.suppress(OSError):  # spinner teardown on closed stream
                    _sp._spinner_live.stop()
                _sp._spinner_live = None
                _sp._spinner_obj = None
                if _margin_stderr:
                    with contextlib.suppress(OSError):  # reset_bol on closed stream
                        _margin_stderr.reset_bol()


class _FsyncedFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler with safe periodic fsync — not per-emit.

    Calling os.fsync() on every emit() blocks on APFS CoW state right after
    shutil.copytree, holding the handler lock and blocking all subsequent log writes
    (RichHandler writes directly to stderr, so it's unaffected).

    The flush() path only flushes Python I/O buffers to the OS kernel (stream.flush);
    fsync() is called only at close() time (on normal process exit).
    Default rotation: 10MB / 3 backups.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        maxBytes: int = 10 * 1024 * 1024,  # noqa: N803 — logging.RotatingFileHandler signature parity
        backupCount: int = 3,  # noqa: N803 — logging.RotatingFileHandler signature parity
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay, errors)

    def flush(self) -> None:
        # Only call stream.flush() — flushes to OS kernel buffer.
        # os.fsync() is NOT called here (avoids CoW contention after staging copytree).
        super().flush()

    def close(self) -> None:
        # flush then fsync before close — guarantees disk sync on normal exit.
        try:
            self.flush()
            if self.stream and hasattr(self.stream, "fileno"):
                os.fsync(self.stream.fileno())
        except (OSError, ValueError):
            print("log fsync failed — last log entries may be lost", file=sys.stderr)
        super().close()


_LOG_FILE_HANDLER: logging.FileHandler | None = None


class _ToolRunningFilter(logging.Filter):
    """Suppresses terminal log output while a design_tool_call is in-flight.

    Prevents log lines from interleaving with the ○→✓ in-place overwrite:
    stdout has no trailing newline while a tool is running, so any stderr
    write (RichHandler) would land on the same terminal row and break \\r\\x1b[2K.
    File handlers are NOT affected — they still capture every record.

    WARNING+ always passes — a broken in-place row is cosmetic, a hidden
    error is not (and `active` can linger if a run is cancelled mid-tool).
    Row break (row_pending handling) is NOT done here — it's done in _RowSafeEmitMixin.emit
    inside _TERM_WRITE_LOCK. Breaking in the filter would let ticker re-rendering slip between
    the newline and the actual log emit, causing WARNING to attach to the spinner row.
    """

    def __init__(self) -> None:
        super().__init__()
        self._active: bool = False

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value
        if not value:
            # Done (✓/✗) lines always end with "\n" — no row break needed
            self.row_pending = False

    @property
    def row_pending(self) -> bool:
        """An in-place row is drawn without a trailing newline. Shared between asi ticker and
        collaborate StreamingDisplay — stored in the terminal_coordination module."""
        return _term_row_pending()

    @row_pending.setter
    def row_pending(self, value: bool) -> None:
        _set_term_row_pending(value)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or not self._active


_tool_running_filter = _ToolRunningFilter()


class _RowSafeEmitMixin:
    """Make terminal log handler emit atomic with ticker re-render.

    If an in-flight ○/spinner row is drawn without a trailing newline (row_pending),
    break the row first, then emit the log — all inside _TERM_WRITE_LOCK so the ticker
    cannot interleave. The broken row is redrawn by the ticker on its next tick (≤0.25s).

    Also coordinates with a *third* rendering surface: the main
    prompt_toolkit input prompt. ``_collect_input`` wraps its
    ``PromptSession.prompt()`` calls in ``patch_stdout()``, which redirects
    ``sys.stdout``/``sys.stderr`` to a proxy that prints cleanly above the
    live prompt and reflows it. But a background thread's log record can
    still land in the brief window between prompt draws, or via a handler
    whose stream reference was captured before the swap (the non-Rich
    ``logging.StreamHandler`` fallback stores ``self.stream`` at
    construction). ``self.stream = sys.stderr`` re-targets that fallback on
    every emit so it always honors whichever stream is currently installed
    (patched or not) — the Rich path doesn't need this because its console
    writes through ``_MarginIO``, which already looks up ``sys.stderr``
    dynamically. As insurance beyond patch_stdout, explicitly invalidate the
    running prompt_toolkit Application (if any) after emit, forcing a clean
    redraw instead of relying solely on the proxy's own scheduling.
    """

    def emit(self, record: logging.LogRecord) -> None:
        with _TERM_WRITE_LOCK:
            if _tool_running_filter.row_pending:
                _tool_running_filter.row_pending = False
                with contextlib.suppress(OSError):  # broken pipe on closed stdout
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            # Rich Live spinner (thinking etc.) occupies its own Live area, not a raw \r row,
            # so row_pending cannot break it. Moreover, Live uses stdout while logs go to stderr
            # (_log_console), so Rich's "print over live" coordinate system does not work —
            # emitting directly would overlay WARNING on top of the spinner row.
            # Therefore, we lower the Live before emit (transient=True → row deletion) so the log
            # lands on a clean new line. Live.stop() joins the refresh thread, eliminating concurrent
            # write contention. _spinner_running is preserved, so the thinking ticker recreates Live
            # on its next tick (≤0.1s) (spinner resumes on non-critical retry warnings).
            _sp = _active_spinner_printer
            if _sp is not None:
                _sp._suspend_live_for_log()
            if hasattr(self, "stream"):
                # non-Rich StreamHandler fallback: re-target to the current
                # sys.stderr (may be patch_stdout's proxy) instead of the
                # object captured at handler-construction time.
                self.stream = sys.stderr  # type: ignore[attr-defined]
            super().emit(record)  # type: ignore[misc]
            # Insurance beyond patch_stdout's own redraw scheduling: force the
            # active prompt (if the user is currently at one) to redraw now
            # rather than leaving a possibly-stale frame on screen.
            _sess = _prompt_session
            _app = getattr(_sess, "app", None) if _sess is not None else None
            if _app is not None and getattr(_app, "is_running", False):
                with contextlib.suppress(RuntimeError, OSError):  # prompt redraw on closed loop
                    _app.invalidate()


class _TerminalInfoFilter(logging.Filter):
    """Suppress INFO-level records from noisy internal loggers on terminal.

    WARNING+ always passes. INFO from the progress printer's own named logger
    ("asi.progress") and from internal pipeline loggers ("external_llm.*"
    via getLogger(__name__), "asicode.*" via explicit names) is suppressed on
    the terminal handler — file handlers are NOT affected. The progress printer
    already renders all important events visually, so the duplicate INFO lines
    add noise without adding information.

    "torch.*" is included because torch (pulled in by sentence-transformers)
    registers an atexit handler in torch._subclasses.fake_tensor that emits
    log.info("FakeTensor cache stats: ...") at interpreter shutdown — otherwise
    printed on the terminal right after "session ended.". Filtering here (rather
    than raising the torch logger level at startup) is robust regardless of
    when/whether torch re-inits its own logger; the file handler still records it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        name = record.name
        return not (
            name == "asi.progress" or name.startswith(("external_llm.", "asicode.", "torch.")) or name == "torch"
        )


_terminal_info_filter = _TerminalInfoFilter()


class _SafeRichFormatter(logging.Formatter):
    """Escapes log message content so Rich markup doesn't break on user/LLM text.

    RichHandler(markup=True) parses all log output as Rich markup.
    If a log message contains ``[/...]`` (e.g. raw LLM output with ``[/{/]``),
    the markup parser crashes with "closing tag ... doesn't match any open tag".

    This formatter escapes ``record.msg`` before formatting, so the
    ``[dim]...[/dim]`` style wrappers in the format string stay intact
    while message content is safe from markup parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        from rich.markup import escape as _escape

        original_msg = record.msg
        original_args = record.args
        if isinstance(record.msg, str):
            record.msg = _escape(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_escape(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: (_escape(v) if isinstance(v, str) else v) for k, v in record.args.items()}
        elif isinstance(record.args, str):  # pragma: no cover — logging.LogRecord normalizes args to tuple/dict
            record.args = _escape(record.args)  # type: ignore[attr-defined]  # LogRecord.args is writeable at runtime
        try:
            result = super().format(record)
            # WARNING+ logs align vertically with tool-call detail lines (Edited/SEMANTIC LINT, etc., col 6 =
            # MarginIO 4 + "  " indent) by indenting 2 more spaces.
            # INFO aligns with start-phase body (tree-sitter/eslint, col 4) — no extra indent.
            if record.levelno >= logging.WARNING:
                result = "  " + result
            return result
        finally:
            record.msg = original_msg
            record.args = original_args


def _setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the Python root logger.

    - Terminal (stderr): always output (RichHandler or StreamHandler)
    - File (log_file): when given, also append plain-text logs. {date}, {time} usable in the filename.
    """
    global _LOG_FILE_HANDLER
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # ── Terminal handler ──
    _rh_class = RichHandler  # module-level None; lazy-import below
    _ensure_log_console_imported()
    if _RICH and _log_console and _rh_class is None:
        try:
            from rich.logging import (
                RichHandler as _rh_class,  # type: ignore[assignment]  # noqa: N813 — lazy-import slot variable
            )
        except ImportError:
            logging.getLogger(__name__).debug("RichHandler not available — fall back to StreamHandler")
    if _RICH and _log_console and _rh_class is not None:

        class _RowSafeRichHandler(_RowSafeEmitMixin, _rh_class):  # type: ignore[name-defined]
            def render_message(self, record, message):  # type: ignore[override]
                # Terminal logs are always cropped to one line. In narrow terminals, long logs
                # would soft-wrap into indented continuation lines, breaking the spinner row,
                # so we prevent wrapping and truncate overflow with …. Full content is preserved
                # (separate formatter, no crop) by the file handler. Tracebacks are rendered
                # through a separate path and are unaffected by this method.
                text = super().render_message(record, message)
                text.no_wrap = True  # noqa: V101 — rich.text.Text API attribute, not ours  # type: ignore[attr-defined]
                text.overflow = "ellipsis"  # type: ignore[attr-defined]
                return text

        term_handler = _RowSafeRichHandler(
            console=_log_console,  # MarginIO-wrapped stderr — aligns col 0 → col _LOG_MARGIN
            show_time=False,  # Terminal logs omit [HH:MM:SS] timestamps → level/message aligns vertically with body (timestamps preserved by file handler)
            show_path=False,
            show_level=False,
            markup=True,
            rich_tracebacks=True,
        )
        term_handler.setFormatter(_SafeRichFormatter("[dim]%(levelname)-5s %(message)s[/dim]", datefmt="[%X]"))
    else:

        class _RowSafeStreamHandler(_RowSafeEmitMixin, logging.StreamHandler):
            pass

        term_handler = _RowSafeStreamHandler(sys.stderr)
        term_handler.setFormatter(
            logging.Formatter("  %(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
    term_handler.addFilter(_tool_running_filter)
    term_handler.addFilter(_terminal_info_filter)
    handlers.append(term_handler)

    # ── File handler ──
    if log_file:
        import datetime

        now = datetime.datetime.now()
        resolved = log_file.replace("{date}", now.strftime("%Y%m%d")).replace("{time}", now.strftime("%H%M%S"))
        log_path = Path(resolved)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # ── logs/ retention: delete files older than 30 days, keep at most 50 ──
        with contextlib.suppress(OSError):  # retention failure is non-critical
            _log_dir = log_path.parent
            if _log_dir.exists():
                _now_ts = time.time()
                _day_secs = 86400
                _log_files = sorted(
                    [f for f in _log_dir.iterdir() if f.is_file() and f.suffix == ".log"],
                    key=lambda f: f.stat().st_mtime,
                )
                # Delete files older than 30 days
                for _lf in _log_files:
                    if _now_ts - _lf.stat().st_mtime > 30 * _day_secs:
                        _lf.unlink(missing_ok=True)
                # Keep at most 50 files (oldest among remaining)
                _log_files = sorted(
                    [f for f in _log_dir.iterdir() if f.is_file() and f.suffix == ".log"],
                    key=lambda f: f.stat().st_mtime,
                )
                while len(_log_files) > 50:
                    _log_files[0].unlink(missing_ok=True)
                    _log_files = _log_files[1:]
        file_handler = _FsyncedFileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _LOG_FILE_HANDLER = file_handler
        handlers.append(file_handler)
        # Notify log file path on terminal (direct print since logging not yet set up)
        print(f"[log] saved → {log_path.resolve()}", file=sys.stderr)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    # ── Suppress third-party library logs ──
    # faiss.loader: prints logger.info("Loading faiss.") / "Successfully loaded faiss." on import
    logging.getLogger("faiss").setLevel(logging.WARNING)
    # (torch's atexit FakeTensor-cache-stats INFO is suppressed on the terminal
    # by _TerminalInfoFilter — emit-time, ordering-immune; file logs keep it.)


# ─── Output helpers (uses stdout-only console) ────────────────────────────────────

_out_console = None  # lazy: _ensure_out_console_imported() creates on first use


def _ensure_out_console_imported() -> None:
    """Lazily create _out_console (stdout Markdown output) on first use. No-op if Rich unavailable."""
    global _out_console
    if _out_console is not None:
        return
    _console_cls = _import_rich_console()
    if _console_cls is not None:
        # rich.theme is imported by rich.console itself (console.py:57), so this
        # cannot fail once _import_rich_console() succeeded.
        from rich.theme import Theme as _RichTheme

        _out_console = _console_cls(
            file=_MarginIO("stdout"),
            width=_console_width,
            force_terminal=True,
            theme=_RichTheme(
                {
                    # headings — blue/sky/teal series, purple removed
                    "markdown.h1": f"bold {_C['blue']}",
                    "markdown.h1.border": _C["border"],
                    "markdown.h2": f"bold {_C['sky']}",
                    "markdown.h3": f"bold {_C['teal']}",
                    "markdown.h4": f"bold {_C['text']}",
                    # inline code — sky text, no background
                    "markdown.code": _C["sky"],
                    # code block
                    "markdown.code_block": _C["text"],
                    # links
                    "markdown.link": f"underline {_C['blue']}",
                    "markdown.link_url": _C["muted"],
                    # bullets/numbers
                    "markdown.item.bullet": _C["peach"],
                    "markdown.item.number": _C["peach"],
                    # horizontal rule
                    "markdown.hr": _C["border"],
                    # blockquote
                    "markdown.block_quote": f"italic {_C['muted']}",
                }
            ),
        )


def _rich_markdown_cls():
    """Lazy ``rich.markdown.Markdown`` accessor (delegates to shared module).

    Importing ``rich.markdown`` costs ~16ms; lazy import defers cost to the
    first actual Markdown render.  Idempotent — cached inside the shared module.
    """
    from external_llm.common.rich_markdown import markdown_cls

    return markdown_cls()


def _strip_ansi(text: str) -> str:
    """Strip terminal ANSI escape sequences (for cleaning up grep/bash output previews) — string ops instead of regex."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\x1b" and i + 1 < n and text[i + 1] == "[":
            j = i + 2
            while j < n and text[j] in "0123456789;":
                j += 1
            if j < n and text[j] in "mABCDEFGHJKSTfilu":
                i = j + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


# ─── Git-based change (diff) rendering ──────────────────────────────────────────
#
# The engine writes files directly to the working tree, so a working tree snapshot *before* execution
# lets git diff *after* execution extract exactly "what this run changed".
#   · ref           = git stash create (freezes tracked modifications as a dangling commit)
#   · untracked set = pre-execution untracked file list (to distinguish newly created files)


def _git(repo_root: str, *args: str, timeout: float = 12.0) -> tuple[int, str]:
    """Run `git -C repo_root <args>`. Returns (returncode, stdout). Never raises."""
    try:
        p = subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return 1, ""
    else:
        return p.returncode, p.stdout


def _git_baseline(repo_root: str) -> dict | None:
    """Snapshot the working tree state before a run. None if not a git repo.

    Notes:
      - ``git stash create`` captures tracked modifications as a dangling
        commit.  ``git update-ref`` in ``refs/asicode/baseline`` anchors it
        so ``git gc`` does not silently delete it, without polluting the
        user's stash list (``git stash list`` stays clean).
      - Untracked files are **not** captured by ``stash create``, so changes
        to pre-existing untracked files inside the run are invisible to
        ``/diff`` and ``/undo``.  This is a known limitation; only files
        that were *tracked at baseline* can be reliably restored.
    """
    rc, _ = _git(repo_root, "rev-parse", "--is-inside-work-tree")
    if rc != 0:
        return None
    # tracked: stash create captures current modifications without touching the tree;
    # empty (clean tree) → fall back to HEAD.
    _, out = _git(repo_root, "stash", "create")
    ref = out.strip()
    if ref:
        # Anchor with a custom ref (not stash) so git gc doesn't collect it
        # and the user's stash list stays clean.
        _git(repo_root, "update-ref", "refs/asicode/baseline", ref)
    else:
        _, out = _git(repo_root, "rev-parse", "HEAD")
        ref = out.strip()
    if not ref:
        return None
    _, untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    return {"ref": ref, "untracked": frozenset(p for p in untracked.split("\x00") if p)}


def _load_checkpoint_store(repo_root: str) -> CheckpointStore | None:
    """Instantiate the agent checkpoint store for *repo_root*, or None.

    The store is an optional dependency of the CLI (the /undo fallback outside
    git work trees). Import/construction failure is never fatal — a store that
    cannot be read simply means the checkpoint path is not offered. Shared by
    the three checkpoint helpers so the guarded import stays in one place.
    """
    try:
        from external_llm.agent.checkpoint_store import CheckpointStore

        return CheckpointStore(repo_root)
    except Exception:
        logging.getLogger(__name__).debug("could not read checkpoint store at %s", repo_root, exc_info=True)
        return None


def _newest_checkpoint_id(repo_root: str) -> str | None:
    """Id of the most recent Undo checkpoint for *repo_root*, or None.

    The CLI's ``/diff`` and ``/undo`` are built on ``_git_baseline``, which
    returns None outside a git work tree — so in a plain directory the CLI had
    no undo at all, while the agent had been recording a perfectly good
    checkpoint of every file it touched on every write. Nothing outside
    ``webapp/`` read those checkpoints, and webapp/ is excluded from the public
    export, so the shipped CLI wrote them and no code could ever read them back.

    Git stays the preferred path where it exists: it undoes a run against the
    real history the user already trusts. This is the fallback for the case git
    cannot serve.
    """
    store = _load_checkpoint_store(repo_root)
    if store is None:
        return None
    try:
        entries = store.list()
    except Exception:
        # Never fatal: this only decides whether /undo is offered, and a store
        # that cannot be read simply means it is not.
        logging.getLogger(__name__).debug("could not read checkpoint store at %s", repo_root, exc_info=True)
        return None
    return entries[0]["id"] if entries else None


def _undo_via_checkpoint(repo_root: str, checkpoint_id: str) -> bool:
    """Restore *checkpoint_id*, returning whether it fully succeeded."""
    store = _load_checkpoint_store(repo_root)
    if store is None:
        return False
    try:
        return store.restore(checkpoint_id)
    except Exception:
        logging.getLogger(__name__).debug("checkpoint undo failed", exc_info=True)
        return False


def _checkpoint_changed_files(repo_root: str, checkpoint_id: str) -> list[str]:
    """Repo-relative paths a checkpoint would touch on restore.

    Both kinds: files it would rewrite, and files it would DELETE because they
    did not exist when the run started. Used to show the user what ``/undo`` is
    about to do, which for the checkpoint path is the only preview available —
    there is no git diff to render.
    """
    store = _load_checkpoint_store(repo_root)
    if store is None:
        return []
    try:
        entry = next((c for c in store.checkpoints if c["id"] == checkpoint_id), None)
        if entry is None:
            return []
        with open(store.checkpoint_dir / entry["path"], encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, KeyError):
        return []
    return sorted(set(data.get("file_hashes", {})) | set(data.get("absent", [])))


def _changed_files_since(repo_root: str, baseline: dict) -> list[str]:
    """Files changed by the run: tracked diffs vs baseline + newly-created untracked.

    Uses ``-z`` (NUL-separated) git output to handle non-ASCII filenames
    (Korean, CJK, etc.) correctly instead of C-quoted ``\\xxx`` paths.
    """
    _, names = _git(repo_root, "diff", "--no-renames", "--name-only", "-z", baseline["ref"])
    tracked = [p for p in names.split("\x00") if p.strip()]
    _, cur = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    new_untracked = [p for p in cur.split("\x00") if p.strip() and p not in baseline["untracked"]]
    # stable order, deduped
    seen: dict[str, None] = {}
    for f in tracked + new_untracked:
        seen.setdefault(f, None)
    return list(seen)


def _file_diff_text(repo_root: str, baseline: dict, path: str) -> tuple[str, bool]:
    """Return (unified_diff_body, is_new_file) for one file relative to baseline."""
    _, out = _git(repo_root, "diff", "--no-color", baseline["ref"], "--", path)
    if out.strip():
        return out, False
    # untracked / freshly created → diff against /dev/null
    _, out = _git(repo_root, "diff", "--no-color", "--no-index", "/dev/null", path)
    return out, bool(out.strip())


def _parse_diff_stats(diff_body: str) -> tuple[int, int]:
    """Count added / removed lines (excluding the +++/--- file headers)."""
    add = rem = 0
    for ln in diff_body.split("\n"):
        if ln.startswith("+") and not ln.startswith("+++ "):
            add += 1
        elif ln.startswith("-") and not ln.startswith("--- "):
            rem += 1
    return add, rem


_DIFF_HEADER_PREFIXES = (
    "diff --git",
    "index ",
    "--- ",
    "+++ ",
    "new file",
    "deleted file",
    "old mode",
    "new mode",
    "similarity ",
    "rename ",
    "copy ",
    "Binary files",
)


def _build_file_diff_renderable(
    rel_path: str,
    diff_body: str,
    is_new: bool,
    *,
    max_lines: int = 60,
):
    """Build a Rich renderable for one file's diff with line numbers + color."""
    from rich.text import Text

    add, rem = _parse_diff_stats(diff_body)
    rows: list[tuple[str, str, str, str]] = []  # (gutter, sign, text, style)
    # old_ln (deleted-line numbers) intentionally unused: removed lines render
    # with an empty gutter slot so the diff stays visually aligned. Only new_ln
    # is shown.
    new_ln = 0
    shown = 0
    truncated = 0
    pending_hunk_gap = False
    for ln in diff_body.split("\n"):
        if any(ln.startswith(p) for p in _DIFF_HEADER_PREFIXES):
            continue
        if ln.startswith("@@"):
            with contextlib.suppress(IndexError, ValueError):  # malformed hunk header → keep previous line number
                seg = ln.split("@@", 2)[1].strip()
                plus = seg.split(" ")[1]
                new_ln = int(plus[1:].split(",")[0])
            if shown:
                pending_hunk_gap = True
            continue
        if ln.startswith("\\"):  # "\ No newline at end of file"
            continue
        if shown >= max_lines:
            truncated += 1
            continue
        if pending_hunk_gap:
            rows.append(("⋯", "", "", _C["border"]))
            pending_hunk_gap = False
        if ln.startswith("+"):
            rows.append((str(new_ln), "+", ln[1:], _C["green"]))
            new_ln += 1
            shown += 1
        elif ln.startswith("-"):
            rows.append(("", "-", ln[1:], _C["red"]))
            shown += 1
        elif ln.startswith(" ") or ln == "":
            rows.append((str(new_ln), " ", ln[1:] if ln else "", _C["muted"]))
            new_ln += 1
            shown += 1

    # trim trailing blank context rows (artifact of split on final newline)
    while rows and rows[-1][1] == " " and rows[-1][2] == "":
        rows.pop()

    gutter_w = max((len(g) for g, *_ in rows if g not in ("", "⋯")), default=1)

    body = Text()
    # file header — path left, +add -rem right
    tag = "  new" if is_new else ""
    body.append(f"  {rel_path}", style=f"bold {_C['text']}")
    body.append(tag, style=_C["teal"])
    body.append("   ")
    if add:
        body.append(f"+{add}", style=_C["green"])
    if add and rem:
        body.append(" ")
    if rem:
        body.append(f"−{rem}", style=_C["red"])
    body.append("\n")

    for gutter, sign, text, style in rows:
        if gutter == "⋯":
            body.append(f"  {'⋯':>{gutter_w}}\n", style=_C["border"])
            continue
        body.append(f"  {gutter:>{gutter_w}} ", style=_C["border"])
        body.append(f"{sign or ' '} ", style=style)
        body.append(text, style=style)
        body.append("\n")
    if truncated:
        body.append(f"  {'':>{gutter_w}} … {truncated} more lines\n", style=_C["muted"])
    return body


def _render_run_diff(
    repo_root: str,
    baseline: dict | None,
    *,
    max_files: int = 20,
    max_lines_per_file: int = 60,
) -> bool:
    """Render colored diffs for every file the run changed. Returns True if shown."""
    _ensure_out_console_imported()
    if not baseline:
        return False
    files = _changed_files_since(repo_root, baseline)
    if not files:
        return False
    rendered: list[tuple[str, str, bool]] = []
    for path in files[:max_files]:
        body, is_new = _file_diff_text(repo_root, baseline, path)
        if body.strip():
            rendered.append((path, body, is_new))
    if not rendered:
        return False

    total_add = total_rem = 0
    for _, body, _new in rendered:
        a, r = _parse_diff_stats(body)
        total_add += a
        total_rem += r
    extra = len(files) - len(rendered)

    if _RICH and _out_console:
        from rich.console import Group
        from rich.text import Text

        title = Text("  changes  ", style=f"bold {_C['peach']}")
        title.append(f"{len(rendered)} file{'s' if len(rendered) != 1 else ''}", style=_C["muted"])
        title.append("   ")
        title.append(f"+{total_add}", style=_C["green"])
        title.append(" ")
        title.append(f"−{total_rem}", style=_C["red"])
        _ensure_out_console_imported()
        _out_console.print()
        _out_console.print(title)
        _out_console.rule(style=_C["border"])
        blocks = [_build_file_diff_renderable(p, b, n, max_lines=max_lines_per_file) for p, b, n in rendered]
        _out_console.print(Group(*blocks))
        if extra > 0:
            _out_console.print(f"  [{_C['muted']}]… {extra} more file(s) changed[/{_C['muted']}]")
    else:
        print()
        print(f"  changes — {len(rendered)} file(s)  +{total_add} -{total_rem}")
        for p, b, n in rendered:
            a, r = _parse_diff_stats(b)
            print(f"  {p}{'  (new)' if n else ''}  +{a} -{r}")
            for ln in b.split("\n"):
                if any(ln.startswith(x) for x in _DIFF_HEADER_PREFIXES) or ln.startswith("@@"):
                    continue
                if ln and ln[0] in "+- ":
                    print(f"    {ln}")
    return True


def _run_changed_stats(
    repo_root: str,
    baseline: dict | None,
    max_files: int = 20,
) -> list[tuple[str, int, int, bool]]:
    """Per-file (path, added, removed, is_new) summary stats of what the run changed.

    Uses ``git diff --numstat -z`` for a single batch call instead of
    per-file ``git diff`` calls (was N+1, now 2 at most).
    """
    if not baseline:
        return []

    # Batch: parse numstat for ALL tracked file changes in one git call.
    stats_map: dict[str, tuple[int, int]] = {}
    rc, numstat = _git(repo_root, "diff", "--numstat", "-z", baseline["ref"])
    if rc == 0 and numstat.strip():
        for field in numstat.split("\x00"):
            if not field or "\t" not in field:
                continue
            parts = field.split("\t", 2)
            if len(parts) < 3:
                continue
            add_s, rem_s, path = parts[0], parts[1], parts[2]
            add = int(add_s) if add_s.isdigit() else 0
            rem = int(rem_s) if rem_s.isdigit() else 0
            stats_map[path] = (add, rem)

    out: list[tuple[str, int, int, bool]] = []
    for path in _changed_files_since(repo_root, baseline)[:max_files]:
        if path in stats_map:
            add, rem = stats_map[path]
            out.append((path, add, rem, False))
        else:
            # New/untracked file: fall back to per-file diff
            body, is_new = _file_diff_text(repo_root, baseline, path)
            if body.strip():
                add, rem = _parse_diff_stats(body)
                out.append((path, add, rem, is_new))
    return out


def _print_run_change_summary(repo_root: str, baseline: dict | None) -> bool:
    """Print a one-line stat (+N -M) per file the run changed. Returns False if nothing changed.

    Prints only this lightweight summary so it's always visible even when the
    full diff (RUN_DIFF) is off — details via /diff, revert via /undo.
    """
    _ensure_out_console_imported()
    stats = _run_changed_stats(repo_root, baseline)
    if not stats:
        return False
    if _RICH and _out_console:
        from rich.text import Text

        for path, add, rem, is_new in stats:
            line = Text("  ")
            line.append("A" if is_new else "M", style=_C["peach"])
            line.append(f" {path}  ", style=_C["text"])
            line.append(f"+{add}", style=_C["green"])
            line.append(" ")
            line.append(f"−{rem}", style=_C["red"])
            _out_console.print(line)
    else:
        for path, add, rem, is_new in stats:
            print(f"  {'A' if is_new else 'M'} {path}  +{add} -{rem}")
    return True


def _undo_run_changes(repo_root: str, baseline: dict) -> tuple[list[str], list[str]]:
    """Revert files the run changed back to their baseline (pre-run) state.

    - Files present in the baseline ref → `git restore --source` (leaves the
      index untouched; falls back to checkout on older git, which also
      restores the index).
    - New files not present in the baseline → deletion is the revert.
    Returns (reverted paths, failed paths).
    """
    undone: list[str] = []
    failed: list[str] = []
    for path in _changed_files_since(repo_root, baseline):
        rc, _ = _git(repo_root, "cat-file", "-e", f"{baseline['ref']}:{path}")
        if rc == 0:
            rc2, _ = _git(repo_root, "restore", "--source", baseline["ref"], "--", path)
            if rc2 != 0:  # fallback for git < 2.23
                rc2, _ = _git(repo_root, "checkout", baseline["ref"], "--", path)
            (undone if rc2 == 0 else failed).append(path)
        else:
            try:
                os.remove(os.path.join(repo_root, path))
                undone.append(path)
            except OSError:
                failed.append(path)
    return undone, failed


def _fmt_elapsed(elapsed: float) -> str:
    """Format a wall-clock duration — shared duration formatter.

    The single source for every CLI-facing duration: the per-turn status line
    and ``_print_session_summary`` both call this.  Sub-second precision for the
    < 60s case (e.g. ``8.2s``); minutes use ``1m 12s`` and hours zero-pad the
    minutes field (``1h 02m``).
    """
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}h {mins:02d}m"
    return f"{mins}m {secs}s"


def _print_session_summary(session_tokens: dict, t0: float) -> None:
    """One-line summary right before session end (elapsed · ↑↓ tokens). Silent if no usage.

    Dollar amounts are intentionally excluded — cost is an estimate, not an exact
    bill, so it is never surfaced on any CLI-facing surface (only logged to the
    debug _log line). Token counts / elapsed time are objective usage metrics, so
    they're kept. This principle applies uniformly to every run-summary token line
    and the session-end summary.
    """
    pt = session_tokens.get("prompt", 0)
    ct = session_tokens.get("completion", 0)
    if not (pt or ct):
        return
    dur = _fmt_elapsed(time.monotonic() - t0)
    _print(
        f"  session  {dur}  ·  ↑{_abbrev_tokens(pt)} ↓{_abbrev_tokens(ct)} tokens",
        _C["muted"],
    )


# ─── Slash commands (interactive utilities) ──────────────────────────────────────

# (command, aliases, argument hint, one-line description)
_SLASH_COMMANDS: list[tuple[str, tuple[str, ...], str, str]] = [
    ("/help", ("/?",), "", "show this command list"),
    ("/diff", (), "", "re-show the last run's file changes"),
    ("/undo", (), "", "revert files changed by the last run to their pre-run state"),
    ("/status", ("/info",), "", "repo · model · mode · session usage"),
    (
        "/model",
        (),
        "[name]",
        "show or switch model: /model <name> · /model <provider>/<name> · /model <provider> <name>",
    ),
    ("/helper", (), "[name]", "model for context-compression: /helper <name> or /helper off (= use main model)"),
    ("/clear", ("/cls",), "", "clear screen + compact conversation into summary"),
    # arg hint is concise one-liner — detailed usage printed when command runs alone
    # (e.g., /insights → subcommands in the `/insights` handler, /think → tab completion).
    (
        "/insights",
        (),
        "[subcommand]",
        "manage design_insights.md: list, compact, verify, archive, prune, drop, or edit",
    ),
    ("/failure-patterns", (), "[subcommand]", "failure-pattern store: list (default), clear, drop <n>"),
    ("/copy", ("/yank",), "", "copy the last final message to the clipboard"),
    ("/code", (), "[msg]", "switch to Code Chat (full context)"),
    ("/general", (), "[msg]", "switch to General Chat (no code context)"),
    ("/think", ("/thinking",), "[mode]", "toggle thinking/reasoning mode (tab for suggestions)"),
    (
        "/auto",
        (),
        "[N|off]",
        "auto-continue: countdown-run the suggested next step after each turn (N = max consecutive steps)",
    ),
    ("/claude", (), "[--fresh] <task>", "ask Claude Code Agent (--fresh: don't share conversation context)"),
    (
        "/orchestrate",
        ("/orch",),
        "<task>",
        "enter Orchestrator mode (persistent — inherits session context; /code to exit)",
    ),
    ("/quit", (":q", "/exit"), "", "end the session"),
]

# Subcommand lists shared between the completer and the command handler so that
# adding a new subcommand updates both automatically. Defining them once prevents
# desync (e.g. a subcommand having tab completion but no handler, or vice versa).
_INSIGHTS_SUBCOMMANDS: list[str] = ["list", "compact", "verify", "archive", "prune", "drop", "edit"]
_FAILURE_PATTERNS_SUBCOMMANDS: list[str] = ["list", "clear", "drop", "prune"]

# Section groups for /help rendering — a flat list of 15 is slow to scan. Commands not listed here
# are gathered into the "other" section by _render_help, so omissions still display.
_SLASH_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("session", ("/help", "/status", "/clear", "/quit")),
    ("model", ("/model", "/helper", "/think")),
    ("mode", ("/code", "/general", "/orchestrate", "/claude", "/auto")),
    ("output", ("/diff", "/undo", "/copy")),
    ("project", ("/insights", "/failure-patterns")),
]

# alias → canonical name
_SLASH_ALIASES: dict[str, str] = {}

# Per-provider API key environment variable names
# Env keys present before .env was consulted (i.e. exported by the shell).
_SHELL_PROVIDED_ENV_KEYS: set[str] = set()
# Key entered at the auth prompt, awaiting proof that it actually works.
_PENDING_API_KEY: dict[str, str] = {}

_API_KEY_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "zai": "ZAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "opencode": "OPENCODE_API_KEY",
}
for _name, _aliases, _arg, _desc in _SLASH_COMMANDS:
    _SLASH_ALIASES[_name] = _name
    for _al in _aliases:
        _SLASH_ALIASES[_al] = _name


def _model_candidates(prefix: str, ollama_timeout: int = 3) -> list[tuple[str, str]]:
    """(provider, model) pairs whose name starts with *prefix* (case-insensitive).

    Known providers are scanned first, then local ollama models. Exact-name
    matches are included implicitly — an exact name always satisfies the prefix
    scan. Shared by ``_resolve_model_arg`` and ``_resolve_model_interactive``
    so the candidate scan stays in one place.
    """
    candidates: list[tuple[str, str]] = []
    for prov, models in _KNOWN_MODELS.items():
        candidates.extend((prov, m) for m in models if m.lower().startswith(prefix))
    candidates.extend(
        ("ollama", nm) for nm in _get_ollama_models(timeout=ollama_timeout) if nm.lower().startswith(prefix)
    )
    return candidates


def _resolve_model_arg(arg: str, ollama_timeout: int = 3):
    """Resolve a model argument into (provider, model_name) or None.

    Accepts ``provider/name`` (explicit) or ``name`` (auto-resolved across
    known providers + ollama local models). Returns ``None`` if ambiguous
    (multiple matches — caller should prompt) or unknown. Empty → None.
    Used by /model and /helper so resolution logic stays in one place.
    """
    arg = (arg or "").strip()
    if not arg:
        return None
    if "/" in arg:
        prov, _, model = arg.partition("/")
        prov, model = prov.strip(), model.strip()
        if not model:
            return None
        return (prov, model)
    if " " in arg:
        # /model <provider> <model_name>  space-separated format supported
        prov, _, model = arg.partition(" ")
        prov, model = prov.strip(), model.strip()
        if model and (prov.lower() in _KNOWN_MODELS or prov.lower() in _API_KEY_ENV_MAP):
            return (prov, model)
    prefix = arg.lower()
    candidates = _model_candidates(prefix, ollama_timeout=ollama_timeout)
    if len(candidates) == 1:
        return candidates[0]
    # Exact name match in a single known provider wins even if prefix is short.
    exact = [(p, m) for (p, m) in candidates if m == arg]
    if len(exact) == 1:
        return exact[0]
    return None


def _resolve_model_interactive(
    arg: str,
    *,
    usage_hint: str = "/model",
    warn_unknown: bool = True,
) -> tuple[str, str] | None:
    """Resolve a model argument interactively into (provider, model).

    Handles three formats:
    1. ``provider/name`` (explicit slash)
    2. ``provider model_name`` (space-separated, known provider only)
    3. ``model_name`` (prefix search across all known providers + ollama)

    For format 3, if multiple candidates match, shows a numbered list and
    prompts for selection.  Returns ``(provider, model)`` on success, or
    ``None`` on cancel / unknown.

    Shared by ``/model`` and ``/helper`` so the resolution + selection UI
    stays in one place.
    """
    arg = (arg or "").strip()
    if not arg:
        return None

    new_provider = ""
    new_model = arg

    # ── 1. slash-separated: provider/name ──
    if "/" in arg:
        parts = arg.split("/", 1)
        new_provider = parts[0].strip()
        new_model = parts[1].strip()
        if not new_model:
            _print(f"  model name required after '/'  (e.g. {usage_hint} anthropic/claude-sonnet-4-6)", _C["yellow"])
            return None
        # provider identifier must be a single token without spaces. If spaces are present,
        # it means natural language input contained '/' (e.g., "qwen3.7-max bug/feature/perf …")
        # which falsely triggered slash parsing — reject to prevent model name pollution.
        if len(new_provider.split()) > 1:
            _print(
                f"  invalid provider '{new_provider}' — provider name must not contain spaces",
                _C["yellow"],
            )
            _print(
                f"  (looks like natural language got mixed in — use only the {usage_hint} <provider>/<model> format)",
                _C["muted"],
            )
            return None

    # ── 2. space-separated: provider model_name (known provider only) ──
    elif " " in arg:
        parts = arg.split(None, 1)
        prov_cand = parts[0].strip().lower()
        if prov_cand in _KNOWN_MODELS or prov_cand in _API_KEY_ENV_MAP:
            new_provider = prov_cand
            new_model = parts[1].strip()
            # Model name must be a single token without spaces. If spaces are present,
            # mixed natural language input (e.g., "opencode qwen3.7-max bug/feature/perf …")
            # would pollute the model name — reject. (provider is already validated as known)
            if len(new_model.split()) > 1:
                _print(
                    f"  invalid model name '{new_model}' — model name must not contain spaces",
                    _C["yellow"],
                )
                _print(
                    f"  (looks like natural language got mixed in — use only the {usage_hint} {prov_cand}/<model> format)",
                    _C["muted"],
                )
                return None
        # else: first token not a known provider, fall back to prefix search below

    # ── 3. no provider — prefix search + interactive selection ──
    if not new_provider:
        candidates = _model_candidates(arg.lower())

        if len(candidates) == 1:
            new_provider, new_model = candidates[0]
        elif len(candidates) > 1:
            _print("", "")
            _print(f"  models matching '{arg}':", _C["sky"])
            for i, (prov, m) in enumerate(candidates, 1):
                _print(f"    {i}. {prov}/{m}", _C["text"])
            _print("", "")
            _print("  select model (number) or Enter to cancel:", _C["muted"], end=" ")
            sys.stdout.flush()
            try:
                sel = _collect_input("").strip()
            except (EOFError, KeyboardInterrupt):
                _print("  cancelled.", _C["yellow"])
                return None
            if sel.isdigit() and 1 <= int(sel) <= len(candidates):
                new_provider, new_model = candidates[int(sel) - 1]
            else:
                _print("  cancelled.", _C["yellow"])
                return None
        elif arg:
            # no prefix match — exact name lookup
            for prov, models in _KNOWN_MODELS.items():
                if new_model in models:
                    new_provider = prov
                    break
            if not new_provider and new_model in _get_ollama_models(timeout=3):
                new_provider = "ollama"
            if not new_provider:
                if warn_unknown:
                    _print(f"  unknown model: {new_model}  ({usage_hint} to list available models)", _C["yellow"])
                return None

    # ── alias conversion (old/typo model names → correct ones) ──
    canonical = _MODEL_ALIASES.get(new_model)
    if canonical:
        _print(f"  ↪ '{new_model}' → '{canonical}' (auto-corrected)", _C["muted"])
        new_model = canonical

    return (new_provider, new_model)


def _prompt_auth_retry_key(provider: str, svc, *, error_message: str = "") -> bool:
    """Prompt for a new API key on auth failure; recreate client if provided.

    Returns True if a new key was entered and the LLM client was successfully
    recreated, False if the user skipped or the provider has no env-var mapping.

    Some providers (e.g. opencode) return HTTP 401 for an *unsupported model
    name* rather than a bad key. In that case re-entering the key never fixes
    the error — detect the "not supported" signal in the error body and steer
    the user to ``/model`` instead of prompting for a key.
    """
    # ── 401 but actual cause is "unsupported model name" ──
    # opencode server returns 401 for unknown models. Re-entering the key at this point
    # would create an infinite 401 loop, so branch to model name verification.
    _emsg = (error_message or "").lower()
    if "not supported" in _emsg or "is not supported" in _emsg:
        _print(
            f"\n  ⚡ {provider} server responded that it does not support the current model ({svc.model}).",
            _C["yellow"],
        )
        _print(
            "  This isn't an API key problem — switch to a supported model with /model.",
            _C["muted"],
        )
        return False

    env_var = _API_KEY_ENV_MAP.get(provider.lower(), "")
    hint = f" (${env_var})" if env_var else ""
    _print(
        f"\n  ⚡ {provider} API key is expired or invalid.{hint}",
        _C["yellow"],
    )
    _print("  Enter a new API key (empty line = skip):", _C["muted"])
    new_key = input("  ▸ ").strip()
    if not new_key:
        _print("  ↪ skipped — showing the original error.", _C["muted"])
        return False
    if env_var:
        os.environ[env_var] = new_key
    try:
        from external_llm.client import create_llm_client as _mk_client

        new_client = _mk_client(provider=provider, api_key=new_key)
        svc.llm_service.client = new_client
        _print(
            f"  ✅ {provider} client recreated.",
            _C["green"],
        )
        # Persistence is DEFERRED until a real call proves the key works —
        # see _commit_verified_api_key. create_llm_client only constructs an
        # object (no network round-trip), so "client recreated" is not evidence
        # of a valid key: eagerly writing here put typos and placeholders into
        # .env permanently, and the next run then failed auth with the bad key
        # already persisted.
        _PENDING_API_KEY.clear()
        _PENDING_API_KEY.update({"env_var": env_var, "key": new_key, "provider": provider})
    except Exception as exc:
        _print(f"  ❌ client recreation failed: {exc}", _C["red"])
        return False
    else:
        return True


def _commit_verified_api_key() -> None:
    """Persist the key from the auth prompt, now that a live call has accepted it.

    Called by the retry site only on success. Nothing to do when the prompt was
    skipped or the retry failed — that is the point: an unverified key never
    reaches .env.
    """
    if not _PENDING_API_KEY:
        return
    env_var = _PENDING_API_KEY.get("env_var") or ""
    new_key = _PENDING_API_KEY.get("key") or ""
    _PENDING_API_KEY.clear()
    if not env_var or not new_key:
        return
    try:
        _save_key_to_dotenv(_resolve_repo_root(None), env_var, new_key)
    except Exception:
        # Non-critical: the key is already live in os.environ for this session.
        # Logged rather than swallowed — a persist that fails every time would
        # otherwise look identical to one that works, and the user would just
        # get re-prompted forever with no explanation.
        logging.getLogger(__name__).warning("could not persist %s to .env", env_var, exc_info=True)
        return
    if env_var in _SHELL_PROVIDED_ENV_KEYS:
        # .env is only consulted for keys the shell did NOT already export, so
        # this save is inert until the stale shell value goes away. Without this
        # warning the "saved" message is a lie and every later run re-prompts.
        _print(
            f"  ! your shell already exports ${env_var}; it overrides .env, so the "
            f"next run will reuse the OLD key. Update your shell config or run "
            f"`unset {env_var}`.",
            _C["yellow"],
        )


def _handle_insights_archive(repo_root: str, rest: str) -> None:
    """Handle ``/insights archive {list|restore <n>|drop <n>}`` subcommands.

    Extracted from the monolithic ``run_repl`` to reduce nesting from
    36 indent levels down to ~4.
    """
    from external_llm.agent.insights_manager import (
        COMPACT_BUDGET_BYTES,
    )
    from external_llm.agent.insights_manager import (
        _archive_invalidate as _ins_arch_invalidate,
    )
    from external_llm.agent.insights_manager import (
        atomic_write_text as _ins_atomic_write,
    )
    from external_llm.agent.insights_manager import (
        entry_age_days as _ins_age,
    )
    from external_llm.agent.insights_manager import (
        insights_archive_path as _ins_arch_path_fn,
    )
    from external_llm.agent.insights_manager import (
        load_archive_file as _ins_load_arch,
    )
    from external_llm.agent.insights_manager import (
        load_insights_file as _ins_load_active,
    )
    from external_llm.agent.insights_manager import (
        parse_insights as _ins_parse,
    )
    from external_llm.agent.insights_manager import (
        serialize_insights as _ins_serialize,
    )

    arch_rest = rest.split(None, 1)[1].strip() if len(rest.split(None, 1)) > 1 else ""
    arch_sub = arch_rest.split(None, 1)[0].lower() if arch_rest else "list"
    arch_path = _ins_arch_path_fn(repo_root)
    arch_content = _ins_load_arch(repo_root)
    if arch_content.strip():
        arch_pre, arch_ents = _ins_parse(arch_content)
    else:
        arch_pre, arch_ents = [], []

    if arch_sub in ("", "list", "ls"):
        if not arch_ents:
            _print("  no archived insights.", _C["muted"])
        else:
            _print(
                f"  archived insights: {len(arch_ents)} (demoted to keep active within budget — NOT deleted):",
                _C["muted"],
            )
            for ai, ae in enumerate(arch_ents, 1):
                acat = f"[{ae.category}]" if ae.category else "[—]"
                aage = _ins_age(ae)
                aage_s = f"{int(aage)}d" if aage is not None else "—"
                aprev = (ae.body.strip().split("\n", 1)[0])[:64]
                _print(f"    {ai}. {acat} ({aage_s}) {aprev}", _C["muted"])
            _print("  /insights archive {list|restore <n>|drop <n>}", _C["muted"])

    elif arch_sub in ("restore", "promote"):
        arch_toks = rest.split()
        if len(arch_toks) < 3:
            _print("  usage: /insights archive restore <n>  (n from /insights archive list)", _C["yellow"])
        else:
            try:
                arch_n = int(arch_toks[2])
            except ValueError:
                arch_n = 0
            if not (1 <= arch_n <= len(arch_ents)):
                _print(f"  no archive entry #{arch_n}  (valid: 1-{len(arch_ents)})", _C["yellow"])
            else:
                restored = arch_ents[arch_n - 1]
                # DURABILITY: re-promote into the ACTIVE file BEFORE removing
                # from the archive. A crash between the two writes leaves the
                # entry in BOTH (recoverable duplicate) — never in NEITHER.
                act_content = _ins_load_active(repo_root)
                if act_content.strip():
                    act_pre, act_ents = _ins_parse(act_content)
                else:
                    act_pre, act_ents = ["# Design Chat Insights\n\n"], []
                act_ents.append(restored)
                act_path = os.path.join(repo_root, ".asicode", "design_insights.md")
                _ins_atomic_write(act_path, _ins_serialize(act_pre, act_ents))
                # Now safe to remove from the archive
                arch_kept = [e for i, e in enumerate(arch_ents, 1) if i != arch_n]
                _ins_atomic_write(arch_path, _ins_serialize(arch_pre, arch_kept))
                _ins_arch_invalidate(repo_root)
                _print(f"  ✓ restored archive #{arch_n} [{restored.category or '—'}] to active insights.", "dim")
                new_b = len(_ins_serialize(act_pre, act_ents).encode("utf-8"))
                if new_b > COMPACT_BUDGET_BYTES:
                    _print(
                        f"  ⚠ active now over budget ({new_b:,} > {COMPACT_BUDGET_BYTES:,}); next /insights compact re-demotes oldest.",
                        _C["yellow"],
                    )

    elif arch_sub == "drop":
        arch_toks = rest.split()
        if len(arch_toks) < 3:
            _print(
                "  usage: /insights archive drop <n>  (PERMANENTLY delete; n from /insights archive list)", _C["yellow"]
            )
        else:
            try:
                arch_n = int(arch_toks[2])
            except ValueError:
                arch_n = 0
            if not (1 <= arch_n <= len(arch_ents)):
                _print(f"  no archive entry #{arch_n}  (valid: 1-{len(arch_ents)})", _C["yellow"])
            else:
                arch_kept = [e for i, e in enumerate(arch_ents, 1) if i != arch_n]
                _ins_atomic_write(arch_path, _ins_serialize(arch_pre, arch_kept))
                _ins_arch_invalidate(repo_root)
                _print(f"  ✓ permanently deleted archive #{arch_n}.", "dim")

    else:
        _print("  usage: /insights archive {list|restore <n>|drop <n>}", _C["muted"])


def _create_llm_client_for(provider: str, api_key: str = ""):
    """Create an LLM client for ``provider`` with env/inline API key.

    Returns the client, or None on failure. Reused by /model and /helper so
    client-creation (env lookup, base_url, error handling) stays consistent.
    """
    from external_llm.client import create_llm_client as _create_llm
    from external_llm.client import resolve_provider_base_url

    if not api_key and provider.lower() != "ollama":
        ak_var = _API_KEY_ENV_MAP.get(provider.lower())
        api_key = os.getenv(ak_var, "") if ak_var else ""
    try:
        return _create_llm(
            provider=provider,
            api_key=api_key or "",
            base_url=resolve_provider_base_url(provider),
        )
    except Exception as _exc:
        # Honor the "None on failure" contract so callers' fail-open paths
        # (compress-helper → fall back to main model; /helper → error line)
        # actually trigger instead of the exception crashing the REPL. This
        # notably covers ModuleNotFoundError from create_llm_client's lazy
        # per-provider imports (`from .openai_client import ...`) — a corrupt
        # or partial install missing an optional provider module must degrade
        # gracefully, not abort startup in _get_compress_llm.
        logging.getLogger(__name__).warning(
            "LLM client creation failed for provider %r: %s",
            provider,
            _exc,
        )
        return None


def _copy_to_clipboard(text: str) -> str:
    """Copy text to the system clipboard.

    Tries native tools in order — macOS pbcopy → Linux wl-copy/xclip/xsel →
    Windows clip — and falls back to the OSC 52 escape sequence if none are
    available (also works over SSH/tmux). Returns the method label used on
    success, or an empty string on failure.
    """
    if not text:
        return ""
    if sys.platform == "darwin":
        _cmds = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        _cmds = [["clip"]]
    else:
        _cmds = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]]
    _payload = text.encode("utf-8")
    for _cmd in _cmds:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                _cmd, input=_payload, check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return _cmd[0]
    # OSC 52 fallback — writes to clipboard if terminal supports it, even without native tools.
    with contextlib.suppress(OSError):  # OSC52 write on broken pipe
        if sys.stdout.isatty():
            import base64

            _b64 = base64.b64encode(_payload).decode("ascii")
            sys.stdout.write(f"\x1b]52;c;{_b64}\x07")
            sys.stdout.flush()
            return "OSC 52"
    return ""


def _get_think_suggestions(provider: str, model: str) -> list[str]:
    """Return /think argument completions based on provider and model.

    Each provider has different thinking/reasoning effort values, so this
    returns the candidate list matching the current model.
    """
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()

    if p == "openai":
        # o-series: minimum effort="low"; GPT-5.2+: "none" available
        if any(x in m for x in ("o1", "o3", "o4")):
            return ["low", "medium", "high"]
        return ["on", "off", "none", "low", "medium", "high"]

    if p == "anthropic":
        # always-thinking models (Fable 5, Mythos 5, Opus 4.8/4.7) cannot be turned off
        _always = m.startswith(("claude-fable-5", "claude-mythos-5", "claude-opus-4-8", "claude-opus-4-7"))
        if _always:
            return ["low", "medium", "high"]
        return ["on", "off", "low", "medium", "high"]

    if p == "deepseek":
        return ["on", "off", "high", "max"]

    if p == "google":
        # Gemini 2.5: no reasoning effort concept → on/off only
        if m.startswith("gemini-2.5"):
            return ["on", "off"]
        # Gemini 3+: supports thinkingLevel
        return ["on", "off", "minimal", "low", "medium", "high"]

    if p == "zai":
        # GLM-5.2+ reasoning_effort: max | xhigh | high | medium | low | minimal | none
        # xhigh maps to max, minimal/none effectively disable thinking
        return ["on", "off", "max", "xhigh", "high", "medium", "low"]

    if p == "ollama":
        return ["on", "off"]

    # Unknown provider → generic list
    return ["on", "off", "high", "medium", "low", "minimal", "max", "none"]


class _SlashCommandCompleter:
    """Slash-command autocomplete when '/' is typed at the start of the prompt.

    Duck-typed completer — does not subclass prompt_toolkit's Completer.
    PromptSession doesn't isinstance-check the completer, it only calls
    get_completions, so this class definition doesn't depend on the
    prompt_toolkit import (→ saves cold-start time on the non-interactive
    path). The Completion symbol is bound to the module globals by
    _load_prompt_toolkit() on REPL entry, so it's available by the time
    the methods run.

    Doesn't interfere with plain text input — candidates are only offered
    when the entire buffer before the cursor is a single token starting
    with '/' (no spaces/newlines). A path like '/Users/...' stops matching
    any command's prefix the moment the second '/' appears, closing the menu.

    After '/model ' it autocompletes from the provider/name list in
    _KNOWN_MODELS. After '/think ' it autocompletes from the thinking/reasoning
    value list matching the current model.
    """

    def __init__(self, get_provider_fn=None, get_model_fn=None, get_dev_models_fn=None):
        # Ensure Completion global is bound by the time the instance is created.
        # production (_collect_input) already loaded it, so this is an idempotent no-op;
        # also makes direct instantiation paths (tests, etc.) self-sufficient.
        _load_prompt_toolkit()
        self._get_provider = get_provider_fn or (lambda: "")
        self._get_model = get_model_fn or (lambda: "")
        self._get_dev_models = get_dev_models_fn or (lambda: {})

    def get_completions(self, document, _complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        # If space present, try command argument autocompletion
        if " " in text:
            yield from self._try_arg_completions(text)
            return
        low = text.lower()
        for name, aliases, arg, desc in _SLASH_COMMANDS:
            for cand in (name, *(a for a in aliases if a.startswith("/"))):
                if cand.startswith(low):
                    yield Completion(
                        cand,
                        start_position=-len(text),
                        display=f"{cand} {arg}".strip() if cand == name and arg else cand,
                        display_meta=desc,
                    )
                    break

    async def get_completions_async(self, document, complete_event):
        """Async completion — prompt_toolkit 3.x calls get_completions_async first.

        This duck-typed completer does NOT inherit from prompt_toolkit's Completer
        base (kept that way for cold-start savings on non-interactive paths), so
        the default async implementation is not inherited either. Mirror the base
        class exactly: iterate the synchronous get_completions and yield. No extra
        import is needed and cold-start is unaffected. get_completions here is
        pure in-memory iteration (fast, no blocking I/O), so running it inline in
        the event loop is safe.
        """
        for item in self.get_completions(document, complete_event):
            yield item

    def _try_arg_completions(self, text):
        """Command-argument autocomplete — /model (model name), /think (thinking value)."""
        cmd_part, _, after = text.partition(" ")
        cmd_low = cmd_part.lower()
        # Identify which command
        cmd_name = None
        for name, aliases, _arg, _desc in _SLASH_COMMANDS:
            cands = [name] + [a for a in aliases if a.startswith("/")]
            if cmd_low in (c.lower() for c in cands):
                cmd_name = name
                break
        if cmd_name == "/think":
            yield from self._yield_think_completions(after)
        elif cmd_name == "/model":
            yield from self._yield_model_completions(after)
        elif cmd_name == "/helper":
            yield from self._yield_model_completions(after)
            # 'off' clears the helper → falls back to the main model.
            if "off".startswith(after.lower()):
                yield Completion(
                    "off",
                    start_position=-len(after),
                    display="off",
                    display_meta="use main model for compression",
                )
        elif cmd_name == "/failure-patterns":
            yield from self._yield_subcommand_completions(after, _FAILURE_PATTERNS_SUBCOMMANDS)
        elif cmd_name == "/insights":
            yield from self._yield_subcommand_completions(after, _INSIGHTS_SUBCOMMANDS)

    def _yield_model_completions(self, prefix):
        """Model-name autocomplete for the /model command.

        /model dev_N <model>: assign a model to a sub-agent slot.
          - "dev" / "dev_1"   → suggest dev_1..dev_8 slot tokens (shows whether set)
          - "dev_1 qwen"      → suggest model names (including off)
        """
        # ── /model dev_N <model>: per-subagent slot ──
        # (1) Slot token completion: "dev" / "dev_" / "dev_1" → suggest dev_1..dev_8
        #     Regex ^dev_?\d*$ matches slot prefixes only (avoids collision with model name typos).
        import re as _re

        _low = prefix.strip()
        if " " not in prefix and _re.match(r"^dev_?\d*$", _low.lower()):
            _cfg_slots = self._get_dev_models() or {}
            for _i in range(1, 9):
                _cand = f"dev_{_i}"
                if _cand.startswith(_low.lower()):
                    _meta = "✓ set" if str(_i) in _cfg_slots else "sub-agent model slot"
                    yield Completion(
                        _cand,
                        start_position=-len(_low),
                        display=_cand,
                        display_meta=_meta,
                    )
            return
        # (2) dev_N <model>: first token is dev_<digit> → complete model name (off included)
        _tok, _sep, _rest = prefix.partition(" ")
        if _sep and _tok.lower().startswith("dev_") and _tok[4:].isdigit():
            _model_part = _rest.strip()
            if _model_part and "off".startswith(_model_part.lower()):
                yield Completion(
                    "off",
                    start_position=-len(_model_part),
                    display="off",
                    display_meta="clear slot → fall back",
                )
            _seen = set()
            for _provider, _models in _KNOWN_MODELS.items():
                for _model in _models:
                    _qualified = f"{_provider}/{_model}"
                    if (_qualified.startswith(_model_part) or _model.startswith(_model_part)) and _model not in _seen:
                        _seen.add(_model)
                        yield Completion(
                            _qualified,
                            start_position=-len(_model_part),
                            display=f"{_provider}/{_model}",
                            display_meta=f"set dev slot → {_provider}/{_model}",
                        )
            return
        # ── Regular /model <model> ──
        seen = set()
        for provider, models in _KNOWN_MODELS.items():
            for model in models:
                qualified = f"{provider}/{model}"
                if (qualified.startswith(prefix) or model.startswith(prefix)) and model not in seen:
                    seen.add(model)
                    yield Completion(
                        qualified,
                        start_position=-len(prefix),
                        display=f"{provider}/{model}",
                        display_meta=f"switch to {provider}/{model}",
                    )

    def _yield_subcommand_completions(self, prefix, subcommands):
        """Generic subcommand completion for /failure-patterns, /insights, etc."""
        _low = prefix.lower()
        for cmd in subcommands:
            if cmd.startswith(_low):
                yield Completion(
                    cmd,
                    start_position=-len(prefix),
                    display=cmd,
                )

    def _yield_think_completions(self, prefix):
        """Thinking-value autocomplete for /think, based on the current model."""
        provider = self._get_provider()
        model = self._get_model()
        suggestions = _get_think_suggestions(provider, model)
        for val in suggestions:
            if val.startswith(prefix):
                yield Completion(
                    val,
                    start_position=-len(prefix),
                    display=val,
                    display_meta=f"set thinking mode ({provider}/{model})",
                )


def _grouped_slash_commands() -> list[tuple[str, list[tuple]]]:
    """(section name, command tuples) in _SLASH_GROUPS order. Ungrouped commands go to "other"."""
    by_name = {c[0]: c for c in _SLASH_COMMANDS}
    grouped: list[tuple[str, list[tuple]]] = []
    seen: set[str] = set()
    for title, names in _SLASH_GROUPS:
        cmds = [by_name[n] for n in names if n in by_name]
        seen.update(n for n in names if n in by_name)
        if cmds:
            grouped.append((title, cmds))
    leftover = [c for c in _SLASH_COMMANDS if c[0] not in seen]
    if leftover:  # pragma: no cover — all commands are grouped in _SLASH_GROUPS; kept as safety net
        grouped.append(("other", leftover))
    return grouped


def _render_help() -> None:
    """Render the slash-command palette, sectioned by _SLASH_GROUPS."""
    _ensure_out_console_imported()
    if _RICH and _out_console:
        from rich import box
        from rich.table import Table
        from rich.text import Text as _Txt

        _out_console.print()
        for title, cmds in _grouped_slash_commands():
            _out_console.print(f"  [bold {_C['blue']}]{title}[/bold {_C['blue']}]")
            tbl = Table(
                box=box.SIMPLE_HEAD, show_header=False, pad_edge=False, padding=(0, 2, 0, 0), border_style=_C["border"]
            )
            # Fixed width: even when sections split, the description column start column aligns across sections
            tbl.add_column(style=_C["sky"], no_wrap=True, width=30)
            # ratio=1: remaining terminal width all goes to description column → no_wrap label does not monopolize width
            tbl.add_column(style=_C["muted"], ratio=1)
            for name, aliases, arg, desc in cmds:
                label = f"{name} {arg}".strip()
                if aliases:
                    label += f"  ({', '.join(aliases)})"
                tbl.add_row(_Txt(label), _Txt(desc))
            _out_console.print(tbl)
        _out_console.print(
            f"  [{_C['muted']}]Enter send · Alt+Enter newline · Ctrl+C exit · drag an image to attach[/{_C['muted']}]"
        )
        _out_console.print()
    else:
        print()
        for title, cmds in _grouped_slash_commands():
            print(f"  {title}")
            for name, aliases, arg, desc in cmds:
                label = f"{name} {arg}".strip()
                if aliases:
                    label += f"  ({', '.join(aliases)})"
                print(f"    {label:<32} {desc}")
        print()


def _render_status(
    repo_root: str,
    provider: str,
    model: str,
    mode: str,
    session_tokens: dict,
    thinking_state: bool | None = None,
    reasoning_effort: str | None = None,
    helper: str = "",
) -> None:
    """Render a compact session status block."""
    _ensure_out_console_imported()
    pt = session_tokens.get("prompt", 0)
    ct = session_tokens.get("completion", 0)
    # Cost (dollars) is an estimate, not an exact bill, so it's not shown in /status.
    # Token count is an objective usage metric, so it's kept.
    _session_str = f"↑{_abbrev_tokens(pt)}  ↓{_abbrev_tokens(ct)} tokens"
    mode_label = "General Chat" if mode == "general" else "Code Chat"
    if thinking_state is True:
        think_label = "thinking ON"
        if reasoning_effort:
            think_label += f" ({reasoning_effort})"
    elif thinking_state is False:
        think_label = "thinking OFF"
    else:
        think_label = "thinking (auto)"
    if _RICH and _out_console:
        from rich.text import Text

        _out_console.print()
        rows = [
            ("repo", repo_root),
            ("model", f"{provider} / {model}" if provider else model),
            ("mode", mode_label),
            ("think", think_label),
        ]
        if helper:
            rows.append(("helper", helper))
        rows.append(("session", _session_str))
        for k, v in rows:
            line = Text(f"  {k:<8} ", style=_C["muted"])
            line.append(v, style=_C["text"])
            _out_console.print(line)
        _out_console.print()
    else:
        print(f"\n  repo     {repo_root}")
        print(f"  model    {provider} / {model}" if provider else f"  model    {model}")
        print(f"  mode     {mode_label}")
        print(f"  think    {think_label}")
        if helper:
            print(f"  helper   {helper}")
        print(f"  session  {_session_str}\n")


_BAR_BOX = None


def _bar_box():
    """A Rich Box that draws only a left gutter bar (▌) — instead of a border on
    all four sides, a single thin colored bar sits to the left of the content,
    making a light, modern block instead of a heavy panel.
    (8-line x 4-char convention: each line is [left, fill, divider, right] —
    only the left side gets ▌, the rest is blank)"""
    global _BAR_BOX
    if _BAR_BOX is None:
        from rich.box import Box

        _BAR_BOX = Box("    \n▌   \n    \n▌   \n    \n    \n▌   \n    \n")
    return _BAR_BOX


def _bar_panel(content, title=None, color: str = "", padding=(0, 2)):
    """Gutter-bar panel based on _bar_box — title is left-aligned on the top (barless) line."""
    from rich.panel import Panel

    return Panel(
        content,
        box=_bar_box(),
        title=title,
        title_align="left",
        border_style=color or _C["border"],
        padding=padding,
    )


def _print(msg: str, style: str = "", end: str = "\n") -> None:
    _ensure_out_console_imported()
    if _RICH and _out_console:
        # Sync MarginIO BOL state in case a direct sys.stdout.write() call happened before.
        # _out_console writes via _MarginIO(sys.stdout); direct writes bypass it and can
        # leave _bol=False, which would suppress the margin on the next _print() line.
        _f = _out_console.file
        if hasattr(_f, "reset_bol"):
            _f.reset_bol()
        from rich.text import Text

        t = Text(msg)
        if style:
            s = _C.get(style, style)
            t.stylize(s)
        _out_console.print(t, end=end)
    else:
        print(msg, end=end)


def _print_banner(repo_root: str = "") -> None:
    """Print the startup banner.

    The title line animates a one-shot beam shimmer *in place* (~0.7s) when
    printed, then settles to its static color. Help line (with the repo path
    right-aligned on the same line) and rule follow.
    No separate ghost title is ever rendered.
    """
    _ensure_out_console_imported()
    if _RICH and _out_console:
        import time as _bt

        from rich.live import Live
        from rich.text import Text

        _out_console.print()
        _word = "asicode"

        # ▌ gutter aligns with col 4 (body/separator/INFO left column). Live's \r re-draw
        # drops the MarginIO left margin (final frame has no margin), leaving only literal indent.
        # So instead of relying on margin(4), directly produce col 4 with a literal 4-space indent.
        def _title_at(el: float) -> Text:
            t = Text("    ▌ ", style=_C["blue"])
            t.append(_render_shimmer_text(_word, el))
            return t

        def _static_title() -> Text:
            t = Text("    ▌ ", style=_C["blue"])
            t.append(_word, style=f"bold {_C['text']}")
            return t

        _dur = 0.7
        if os.environ.get("NO_COLOR"):
            # NO_COLOR convention: color-interpolated shimmer is meaningless — static title only.
            _out_console.print(_static_title())
        else:
            try:
                with Live(
                    _title_at(0.0),
                    console=_out_console,
                    refresh_per_second=24,
                    # Rich's redirect_stdout/stderr swaps sys.stdout for a
                    # FileProxy(_out_console); since _MarginIO (this console's
                    # file) now resolves sys.stdout dynamically, that proxy
                    # points straight back here → infinite recursion. Disable
                    # the redirect: margin-console writes always went to the
                    # real stream anyway, and log/spinner interleaving is
                    # coordinated via _TERM_WRITE_LOCK, not Rich's reflow.
                    redirect_stdout=False,
                    redirect_stderr=False,
                    transient=False,
                ) as live:
                    _t0 = _bt.monotonic()
                    while True:
                        el = _bt.monotonic() - _t0
                        if el >= _dur:
                            live.update(_static_title())
                            break
                        live.update(_title_at(el))
                        _bt.sleep(1 / 24)
            except Exception:
                _out_console.print(_static_title())

        # Literal indent 2 + MarginIO margin(4) = col 6 → bottom status line (zai / ... · /help ...)
        # and input prompt (❯) vertical alignment. Title ▌/separator/body(INFO) stay at col 4.  # noqa: RUF003
        _help = Text("  /help for commands  ·  Ctrl+C exit", style=_C["muted"])
        # Append repo path at end of help line in same color (muted), joined by separator (·).
        if repo_root:
            _help.append(f"  ·  {repo_root}", style=_C["muted"])
        _out_console.print(_help)
        _out_console.rule(style=_C["border"])
    else:
        print("─" * 60)
        print("  ▌ asicode")
        _suffix = f"  ·  {repo_root}" if repo_root else ""
        print(f"    /help for commands  ·  Ctrl+C exit{_suffix}")
        print("─" * 60)


# ─── Dependency status check ─────────────────────────────────────────────────────


def _check_dep_status(tools) -> dict[str, str]:
    """Return a dict of optional dependency → 'ON' / 'OFF' / 'skip'.

    *tools* is a list of already-resolved ``_Tool`` instances (produced by
    ``_check_tools_with_state``) carrying their ``found`` / ``skipped`` /
    ``use_npx`` state for this run.  This renderer therefore:

    * never re-polls ``$PATH`` (avoids flapping with npx-based tools), and
    * never mislabels a user-skipped tool as 'OFF' (it shows 'skip' instead).

    Always-present infrastructure (tree-sitter, vector) is appended here.
    """
    from external_llm.languages.tree_sitter_utils import is_available

    ts = "ON" if is_available() else "OFF"

    result: dict[str, str] = {"tree-sitter": ts}
    for t in tools:
        if t.cmd in result:
            continue
        if t.skipped:
            result[t.cmd] = "skip"
        elif t.found:
            result[t.cmd] = "ON"
        else:
            result[t.cmd] = "OFF"

    # vector (semantic search) — reflects actual availability in 3 tiers:
    #   OFF      : packages (faiss/numpy/sentence-transformers) not installed
    #   no-model : packages installed but embedding model not downloaded → BM25 only
    #   ON       : packages + model (preferred or fallback) ready
    from external_llm.agent.vector_cache import (
        FALLBACK_EMBEDDING_MODELS,
        HAS_FAISS,
        HAS_NUMPY,
        HAS_SENTENCE_TRANSFORMERS,
        get_configured_embedding_model_name,
    )

    if not (HAS_FAISS and HAS_NUMPY and HAS_SENTENCE_TRANSFORMERS):
        vector = "OFF"
    else:
        _models = [get_configured_embedding_model_name(), *FALLBACK_EMBEDDING_MODELS]
        vector = "ON" if any(_is_embedding_model_cached(m) for m in _models) else "no-model"
    result["vector"] = vector

    return result


# tree-sitter language → short display label
_LANG_LABEL: dict[str, str] = {
    "python": "py",
    "typescript": "ts",
    "javascript": "js",
    "go": "go",
    "java": "java",
    "kotlin": "kt",
    "html": "html",
    "rust": "rs",
    "c": "c",
    "cpp": "cp",
    "ruby": "rb",
    "php": "php",
    "c_sharp": "cs",
    "swift": "sw",
    "scala": "sc",
    "lua": "lua",
    "bash": "sh",
    "css": "css",
}


def _git_ls_files(repo_root: str) -> list[str]:
    """Repo-relative file paths in *repo_root*; [] on any failure.

    Delegates to the ``common.repo_files`` SSOT rather than running
    ``git ls-files`` here. The local version omitted ``-z``, and porcelain
    output C-quotes non-ASCII paths — ``한글파일.py`` arrives as
    ``"\\355\\225\\234...2.py"``, whose suffix is ``.py"`` (quote included), so
    ``LanguageId.from_path`` returns UNKNOWN and every Korean/CJK-named source
    file was invisible to :func:`_detect_repo_ts_languages`. That is precisely
    the trap ``git_list_repo_files`` documents ``-z`` as REQUIRED for.

    Also widens the set from tracked-only to tracked + untracked-not-ignored,
    which is the right answer for "which languages are in this repo": a
    just-created ``.ts`` file counts. Matches what ``glob`` already lists.
    """
    from external_llm.common.repo_files import git_list_repo_files

    return git_list_repo_files(repo_root) or []


def _detect_repo_ts_languages(files: list[str]) -> set[str]:
    """Curated (AST-supported) languages present in the repo file list."""
    from external_llm.languages.models import LanguageId
    from external_llm.languages.tree_sitter_utils import _LANG_MODULE_MAP

    langs: set[str] = set()
    for path in files:
        lang = LanguageId.from_path(path).value
        if lang in _LANG_MODULE_MAP:
            langs.add(lang)
    return langs


def _print_dep_status(repo_root: str, *, no_deps_check: bool = False) -> None:
    """Print dependency status line (non-blocking, <20 ms overhead).

    A single pass through :func:`_check_tools_with_state` both prompts for
    missing tools (when interactive and not suppressed) *and* yields the
    ``found``/``skipped`` state consumed by :func:`_check_dep_status`.  This
    avoids the previous double-call (prompt loop + separate ``which()`` pass)
    and its misleading 'OFF' label for tools the user deliberately skipped.

    *no_deps_check* propagates the ``--no-deps-check`` CLI flag so that the
    REPL status line honors it the same way ``main()`` does.
    """
    # ── 1. Detect repo languages ──
    from external_llm.languages.dependency_checker import (
        _check_tools_with_state,
        detect_repo_languages,
    )

    detected = detect_repo_languages(repo_root)

    # ── 2. Single check + interactive install (returns rich _Tool state) ──
    tools = _check_tools_with_state(detected, no_prompt=no_deps_check)

    # ── 3. Final status (after any installs) — reflects skipped/found ──
    tool_status = _check_dep_status(tools)

    # ── 4. Tree-sitter status (ON/OFF — 0ms, no grammar import) ──
    from external_llm.languages.tree_sitter_utils import is_available

    ts_summary = "ON" if is_available() else "OFF"

    # Build dynamic tool status line (only tools relevant to detected langs)
    tool_parts = []
    for key in sorted(tool_status):
        if key in ("tree-sitter", "vector"):
            continue
        tool_parts.append(f"{key}: {tool_status[key]}")

    # Merge tree-sitter + tools + vector into a single line
    if _RICH and _out_console:
        from rich.text import Text

        # Leading space aligns this status line with the "/insights" nudge
        # continuation line printed above it in run_repl (col 6 = _CONSOLE_MARGIN
        # 4 + 1 literal space; the nudge is split at " /insights" and printed with
        # one literal leading space). The non-Rich branch below mirrors this.
        line = Text(" tree-sitter: ", style="dim")
        line.append(ts_summary)
        if tool_parts:
            line.append("  ", style="dim")
            line.append("  ".join(tool_parts), style="dim")
        line.append("  ", style="dim")
        line.append(f"vector: {tool_status.get('vector', 'OFF')}", style="dim")
        _out_console.print(line)
    else:
        parts = [f"tree-sitter: {ts_summary}"]
        if tool_parts:
            parts.append("  ".join(tool_parts))
        parts.append(f"vector: {tool_status.get('vector', 'OFF')}")
        _print(" " + "  ".join(parts))

    # ── (a) Tree-sitter grammar missing warning (repo-filtered lazy check) ──
    repo_files = _git_ls_files(repo_root)
    repo_langs = _detect_repo_ts_languages(repo_files)
    if repo_langs and is_available():
        from external_llm.languages.tree_sitter_utils import _get_language

        ts_available = {lang for lang in repo_langs if _get_language(lang) is not None}
    else:
        ts_available = set()
    missing = sorted(repo_langs - ts_available)
    if missing:
        labels = ", ".join(_LANG_LABEL.get(_item_, _item_) for _item_ in missing)
        # Recommend the single language-pack (covers every repo language at
        # once) rather than per-grammar packages — matches the core dependency.
        pkgs = ["tree-sitter-language-pack"]
        if _RICH and _out_console:
            from rich.text import Text

            warn = Text("  ⚠ ", style=_C.get("yellow", "yellow"))
            warn.append(
                f"This repo contains {labels}, but the tree-sitter grammar is "
                f"not installed — AST-based analysis is disabled.",
                style=_C.get("text", ""),
            )
            _out_console.print(warn)
        else:
            _print(
                f"  ⚠ This repo contains {labels}, but the tree-sitter grammar is "
                f"not installed — AST-based analysis is disabled."
            )

        # Y/N prompt → Y for automatic install
        try:
            answer = _collect_input(f"    Install now? (pip install {' '.join(pkgs)}) [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            _install_tree_sitter_grammars(pkgs)
        else:
            _print(f"    Skipped. To install later: pip install {' '.join(pkgs)}", _C.get("muted", ""))

    # ── vector (embedding) dependencies: prompt to install/download if packages or model missing ──
    _maybe_prompt_vector_install()

    # If restart-required deps were newly installed, restart once after all prompts finish
    if _DEPS_RESTART_PENDING:
        _restart_cli()


# Whether restart-required deps (flags fixed at import time) were installed
_DEPS_RESTART_PENDING = False


def _restart_cli() -> None:
    """Replace the current process with a fresh interpreter run (same argv).

    Used after installing deps whose import-time flags can't be refreshed live
    (tree-sitter core, vector packages)."""
    _print(
        "  ↻ Restarting asi to load newly installed dependencies ...",
        _C.get("green", "green"),
    )
    with contextlib.suppress(OSError, ValueError):  # flush on closed/broken streams
        sys.stdout.flush()
        sys.stderr.flush()
    argv = [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    try:
        os.execv(sys.executable, argv)
    except OSError as _e:
        # execv failed (broken/missing interpreter, exec-permission). Don't crash —
        # the newly installed deps simply won't be live this session; the CLI can
        # continue degraded and the user can restart manually when convenient.
        _print(
            f"  ! auto-restart failed ({_e.strerror or _e}); please restart asi manually to load the new dependencies.",
            _C["yellow"],
        )


def _pip_install(pkgs: list[str], *, timeout: int = 600, _force_break: bool = False, label: str | None = None) -> bool:
    """pip-install *pkgs* into the current interpreter's env. Returns success.

    If the first attempt fails due to PEP 668 (externally-managed-environment),
    retries automatically with --break-system-packages.

    Shows a single live status line (spinner + elapsed) while pip runs, since
    `capture_output` otherwise leaves long installs (e.g. sentence-transformers
    pulling torch) looking frozen for minutes. The line is stopped and cleared
    synchronously before any result message is printed.
    """
    import threading as _threading
    import time as _time

    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    if _force_break:
        # PEP 668 retry — target the user site (never the managed system tree)
        # via the shared decision helper. These are all import-packages
        # (tree-sitter / vector / prompt_toolkit / claude SDK), so --user is
        # safe (contrast dependency_checker, which installs CLI tools). On an
        # externally-managed env this yields --user --break-system-packages;
        # elsewhere it degrades to plain (the retry only fires post-PEP-668).
        from external_llm.pip_env import pip_install_flags

        cmd += pip_install_flags() or ["--break-system-packages"]
    if not label:
        label = pkgs[0] + (f" (+{len(pkgs) - 1})" if len(pkgs) > 1 else "")

    _tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    _stop = _threading.Event()
    _t0 = _time.monotonic()

    def _spin() -> None:
        frames = "▖▘▝▗"
        i = 0
        while not _stop.wait(0.15):
            el = _time.monotonic() - _t0
            sys.stderr.write(f"\r\033[K  Installing {label} … {frames[i % 4]}  {el:0.0f}s")
            sys.stderr.flush()
            i += 1

    _spinner = _threading.Thread(target=_spin, daemon=True)

    proc = None
    err: BaseException | None = None
    timed_out = False
    timed_out_tail = ""
    try:
        if _tty:
            _spinner.start()
        else:
            _print(f"  Installing {label} …", _C.get("muted", ""))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as te:
            timed_out = True
            # TimeoutExpired carries the output captured so far — surface its
            # tail so a stalled install is diagnosable (where did it stop?).
            _partial = getattr(te, "stdout", None)
            if _partial is None:
                _partial = getattr(te, "output", None)
            if isinstance(_partial, bytes):
                _partial = _partial.decode("utf-8", "replace")
            timed_out_tail = "\n".join((_partial or "").strip().splitlines()[-2:])
        except (OSError, subprocess.SubprocessError) as e:
            err = e
    finally:
        _stop.set()
        if _tty:
            _spinner.join(timeout=1.0)
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    if timed_out:
        _print(f"  ✗ Install timed out after {timeout}s.", _C.get("yellow", "yellow"))
        for _line in timed_out_tail.splitlines():
            _print(f"    {_line}", _C.get("muted", ""))
        return False
    if err is not None:
        _print(f"  ✗ Install failed: {err}", _C.get("yellow", "yellow"))
        return False
    assert proc is not None  # timed_out/err handled above; subprocess.run succeeded

    if proc.returncode != 0:
        combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
        for line in combined.strip().splitlines()[-3:]:
            _print(f"    {line}", _C.get("muted", ""))
        # PEP 668: system/Homebrew Python is externally-managed, pip refuses.
        if "externally-managed-environment" in combined and not _force_break:
            _print(
                "  ↳ Python externally managed (PEP 668) — retrying with --break-system-packages …",
                _C.get("yellow", "yellow"),
            )
            # Thread `label` into the retry so the explicit display label
            # (e.g. "claude_agent_sdk") survives — otherwise it reverts to the
            # pkgs[0] default ("-e"), which is the regression this retry path
            # is most often the *normal* path (Homebrew/externally-managed).
            return _pip_install(pkgs, timeout=timeout, _force_break=True, label=label)
        _print(f"  ✗ Install failed (exit {proc.returncode}).", _C.get("yellow", "yellow"))
        return False
    # A successful in-process install wrote packages into site-packages; clear
    # the import finder cache so a subsequent import/find_spec sees them. The
    # FileFinder caches directory listings by mtime, which can be stale right
    # after a write — a real install may otherwise read back as "still missing"
    # (the failure mode this guards: find_spec() returning None immediately
    # after pip just installed the package). Benefits every caller
    # (tree-sitter / prompt_toolkit / vector / claude SDK), not only the one
    # that re-checks via find_spec right after installing.
    import importlib

    from external_llm.pip_env import ensure_user_site_importable

    # A --user install may land in a user-site dir absent at startup; make it
    # importable before the caller re-imports the just-installed package.
    ensure_user_site_importable()
    importlib.invalidate_caches()
    return True


def _install_tree_sitter_grammars(pkgs: list[str]) -> None:
    """pip-install the given tree-sitter grammar packages into the current env,
    then refresh the grammar cache so they take effect without a restart."""
    global _DEPS_RESTART_PENDING
    if not _pip_install(pkgs, timeout=300):
        return

    import external_llm.languages.tree_sitter_utils as _ts_utils

    # If core (tree-sitter) is not in this process, installing grammar alone won't work → restart
    if not _ts_utils.is_available():
        _print("  ✓ Installed.", _C.get("green", "green"))
        _DEPS_RESTART_PENDING = True
        return

    # If core is present, grammar is live-reflected via cache invalidation
    try:
        _ts_utils.invalidate_caches()
        now_available = _ts_utils.get_available_languages()
    except Exception:
        now_available = set()

    labels = " ".join(sorted(_LANG_LABEL.get(_item_, _item_) for _item_ in now_available)) if now_available else "OFF"
    _print(f"  ✓ Installed. tree-sitter: {labels}", _C.get("green", "green"))


# vector (semantic search) Python packages — listed explicitly since pyproject has no extra
_VECTOR_PKGS: list[str] = ["sentence-transformers", "faiss-cpu", "numpy"]


def _embedding_cache_roots() -> list[str]:
    """Candidate HF/ST cache hub roots, in priority order."""
    roots: list[str] = []
    for env in ("HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        val = os.environ.get(env)
        if val:
            roots.append(val)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    return roots


def _embedding_model_folder(model_name: str) -> str:
    """The `models--…` cache folder name for a model."""
    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return "models--" + repo_id.replace("/", "--")


def _is_embedding_model_cached(model_name: str) -> bool:
    """Best-effort: is the SentenceTransformer model already in the HF/ST cache?

    Fast filesystem check (no network, no model load). False negatives are
    harmless — a Y just re-loads from cache quickly."""
    folder = _embedding_model_folder(model_name)
    for root in _embedding_cache_roots():
        snap = os.path.join(root, folder, "snapshots")
        with contextlib.suppress(OSError):
            if os.path.isdir(snap) and any(os.scandir(snap)):
                return True
    return False


def _embedding_cache_bytes(model_name: str) -> int:
    """Bytes downloaded so far for a model (sum of files under its cache folder).

    Used to drive a live download progress line; cheap to poll for ~15 files."""
    folder = _embedding_model_folder(model_name)
    for root in _embedding_cache_roots():
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        total = 0
        for dirpath, _dirs, files in os.walk(base):
            for fname in files:
                with contextlib.suppress(OSError):
                    total += os.path.getsize(os.path.join(dirpath, fname))
        return total
    return 0


def _maybe_prompt_vector_install() -> None:
    """Prompt to install vector deps / download the embedding model when missing."""
    from external_llm.agent.vector_cache import (
        FALLBACK_EMBEDDING_MODELS,
        HAS_FAISS,
        HAS_NUMPY,
        HAS_SENTENCE_TRANSFORMERS,
        get_configured_embedding_model_name,
    )

    deps_ok = HAS_FAISS and HAS_NUMPY and HAS_SENTENCE_TRANSFORMERS
    model_name = get_configured_embedding_model_name()
    fallback_name = FALLBACK_EMBEDDING_MODELS[0] if FALLBACK_EMBEDDING_MODELS else None

    # 1) Python package missing → suggest pip install (needs restart to take effect)
    if not deps_ok:
        _print(
            "  ⚠ Semantic (vector) search is disabled — sentence-transformers is not installed.",
            _C.get("yellow", "yellow"),
        )
        try:
            answer = _collect_input(f"    Install now? (pip install {' '.join(_VECTOR_PKGS)}) [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            global _DEPS_RESTART_PENDING
            # sentence-transformers pulls in torch (hundreds of MB), use generous timeout.
            if _pip_install(_VECTOR_PKGS, timeout=1800):
                _print("  ✓ Installed.", _C.get("green", "green"))
                _DEPS_RESTART_PENDING = True  # HAS_* flags are only set after restart
            else:
                _print(
                    f"    To install manually: pip install {' '.join(_VECTOR_PKGS)}",
                    _C.get("muted", ""),
                )
        else:
            _print(
                f"    Skipped. To install later: pip install {' '.join(_VECTOR_PKGS)}",
                _C.get("muted", ""),
            )
        return

    # 2) Package installed but embedding model not downloaded → suggest download (live reflection)
    # If either preferred (multilingual) or fallback (lightweight) is cached, semantic search
    # already works — don't ask again (avoids nagging users who only have a fallback).
    def _yes(s: str) -> bool:
        return s.strip().lower() in ("y", "yes")

    if _is_embedding_model_cached(model_name):
        return
    if fallback_name and _is_embedding_model_cached(fallback_name):
        return

    # No model at all → suggest preferred first
    _print(
        f"  ⚠ Embedding model '{model_name}' is not downloaded — semantic "
        "search will be unavailable until a model is fetched.",
        _C.get("yellow", "yellow"),
    )
    try:
        answer = _collect_input("    Download now? (~470MB, multilingual, one-time) [y/N] ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if _yes(answer):
        _download_embedding_model(model_name)
        return

    # preferred declined → ask if they want the lightweight fallback model
    if fallback_name:
        try:
            answer = _collect_input(
                f"    Install the lighter fallback '{fallback_name}' instead? (~90MB, English-leaning) [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if _yes(answer):
            _download_embedding_model(fallback_name)
            return

    _print(
        "    Skipped. A model will be downloaded automatically on first use.",
        _C.get("muted", ""),
    )


def _download_embedding_model(model_name: str) -> None:
    """Download and load the SentenceTransformer model with a single, clean status line.

    We render our OWN progress line and suppress all third-party logging/tqdm
    output for the duration. tqdm bars from huggingface_hub write to the terminal
    from worker threads and `leave=True`, so their final flush can land *after*
    we return — landing in the freshly-drawn prompt. Driving the line ourselves
    and stopping it synchronously before returning removes any late async writes.
    """
    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"

    import threading as _threading
    import time as _time

    from huggingface_hub import constants as hf_constants
    from huggingface_hub import snapshot_download

    from external_llm.agent.vector_cache import _suppress_hf_progress

    # ── 1) Bypass HF_HUB_OFFLINE ──────────────────────────────────
    # huggingface_hub.constants.HF_HUB_OFFLINE reads os.environ at import time and caches it,
    # so changing os.environ alone is useless. Modify the module constant directly.
    _old_hf_offline = getattr(hf_constants, "HF_HUB_OFFLINE", None)
    if _old_hf_offline:
        hf_constants.HF_HUB_OFFLINE = False

    # ── 2) Single progress line (spinner + download size + elapsed) ─────────────
    # Log/progress bar suppression is handled by _suppress_hf_progress(). The progress display is
    # exclusively our spinner line. (_print is not logging, so it's unaffected.)
    _tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    _stop = _threading.Event()
    _t0 = _time.monotonic()

    def _spin() -> None:
        frames = "▖▘▝▗"
        i = 0
        while not _stop.wait(0.15):
            mb = _embedding_cache_bytes(model_name) / 1_000_000
            el = _time.monotonic() - _t0
            sys.stderr.write(f"\r\033[K  Downloading embedding model… {frames[i % 4]}  {mb:,.0f}MB  {el:0.0f}s")
            sys.stderr.flush()
            i += 1

    _spinner = _threading.Thread(target=_spin, daemon=True)

    try:
        with _suppress_hf_progress():
            if _tty:
                _spinner.start()
            else:
                _print(
                    f"  Downloading embedding model '{model_name}' (one-time) …",
                    _C.get("muted", ""),
                )

            try:
                # Exclude onnx/openvino variants → ~470MB, torch weights only. (Default backend is
                # torch, so onnx export is unnecessary and would bloat to 2GB+.)
                snapshot_download(
                    repo_id,
                    ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "*.onnx_data"],
                )
            finally:
                # Synchronously stop and clear the progress line — prevents late flicker after return.
                _stop.set()
                if _tty:
                    _spinner.join(timeout=1.0)
                    sys.stderr.write("\r\033[K")
                    sys.stderr.flush()

            # Verify cache
            if not _is_embedding_model_cached(model_name):
                _print("  ✗ Model files not found in cache after download.", _C.get("yellow", "yellow"))
                return

            # ── Load: activate the model we just downloaded (snapshot_download is called again,
            # so OFFLINE bypass is still needed). set_active_embedding_model bypasses
            # the preferred→fallback order, so if the user chose fallback, it won't
            # silently download preferred instead.
            from external_llm.agent.vector_cache import set_active_embedding_model

            model = set_active_embedding_model(model_name)
            if model is None:
                _print("  ✗ Could not load the embedding model.", _C.get("yellow", "yellow"))
                return
            _print(f"  ✓ Embedding model ready ({model_name}).", _C.get("green", "green"))
    except Exception as e:
        _stop.set()
        _print(f"  ✗ Failed: {e}", _C.get("yellow", "yellow"))
        return
    finally:
        # ── Restore ─────────────────────────────────────────────────
        if _old_hf_offline:
            hf_constants.HF_HUB_OFFLINE = _old_hf_offline


def _kick_embedding_model_warmup() -> None:
    """Start a background thread pre-loading the embedding model.

    The ``SentenceTransformer`` load (~2 s) blocks ``ToolRegistry`` construction
    via ``RAGSearcher`` → ``VectorCacheManager`` → ``get_global_embedding_model``.
    We start the load here so it overlaps with subsequent main-thread startup
    work (LLM service creation, design-chat setup, prompt UI init). The loader
    is lock-guarded with a double-check, so the eventual ``ToolRegistry`` call
    either reuses the now-loaded singleton or briefly blocks until the warmup
    finishes — never loading twice, never worse than the status quo.

    Guarded to run only when deps are already present AND a model is cached:
    missing deps need a restart (handled by the install prompt), and a missing
    model needs a download decision (the user Y/N prompt), so neither should be
    silently triggered from a daemon thread. ``warmup_embedding_model`` itself
    short-circuits when deps are absent or the model is already loaded.
    """
    from external_llm.agent.vector_cache import (
        FALLBACK_EMBEDDING_MODELS,
        HAS_FAISS,
        HAS_NUMPY,
        HAS_SENTENCE_TRANSFORMERS,
        get_configured_embedding_model_name,
        warmup_embedding_model,
    )

    if not (HAS_FAISS and HAS_NUMPY and HAS_SENTENCE_TRANSFORMERS):
        return
    model_name = get_configured_embedding_model_name()
    fallback = FALLBACK_EMBEDDING_MODELS[0] if FALLBACK_EMBEDDING_MODELS else None
    # Only warm up when a model is already on disk — otherwise the background
    # thread would either stall on a network fetch or race the interactive
    # download prompt. The dep prompt above handles fetching.
    if not _is_embedding_model_cached(model_name) and (not fallback or not _is_embedding_model_cached(fallback)):
        return
    t = threading.Thread(target=warmup_embedding_model, name="emb-warmup", daemon=True)
    t.start()


# ─── Stream callback → user-friendly message conversion ────────────────────────────────

_EVENT_LABELS: dict[str, str] = {
    "routing_intent": "analyzing",
    "route_decision": "routing",
    "route_applied": "route applied",
    "tool_call_preview": "tool",
    "tool_call": "tool done",
    "tdd_cycle_start": "running tests",
    "tdd_cycle_pass": "tests pass",
    "tdd_cycle_fail": "tests fail",
    "budget_warning": "context limit warning",
    "fail_loop_detected": "fail loop detected",
    "complete": "done",
    "error": "error",
    "cancelled": "cancelled",
    "rate_limit_retry": "rate limit — retrying",
    "agent_thinking": "thinking",
    "turn_start": "turn",
    "design_tool_call": "design tool",
    "design_thinking": "design thinking",
    "self_review": "self-review",
}

_SILENT_EVENTS = {
    "session_start",
    "session_end",
    "done",
    "llm_input",
    "llm_output",
    "routing_intent",  # internal classification noise
    "auto_observation",
    "performance_metrics",
    "small_model_complexity_warning",
}


def _relativize_repo_paths(text: str) -> str:
    """Shorten the repo root's absolute path to a relative one — reclaims width for the one-line hint.

    The 'cd <repo> && ' prefix is stripped entirely (every command already runs
    from the repo root); other occurrences of '<repo>/' become empty, and a
    bare '<repo>' becomes '.'.
    """
    rr = _REPO_ROOT.rstrip("/")
    if not rr or rr == "/":
        return text
    for _q in ("", "'", '"'):
        _pfx = f"cd {_q}{rr}{_q} && "
        if text.startswith(_pfx):
            text = text[len(_pfx) :].lstrip()
            break
    return text.replace(rr + "/", "").replace(rr, ".")


def _extract_tool_cmd(args: dict) -> str:
    """Extract a CLI-displayable command hint from design_tool_call args."""
    if not args:
        return ""
    # shell_exec / bash / git_* series
    cmd = args.get("command") or args.get("cmd") or ""
    if cmd:
        # normalize newlines/tabs so inline hint stays on a single line
        return _relativize_repo_paths(" ".join(cmd.split()))[:200]
    # grep: 'pattern' in path format
    pattern = args.get("pattern") or ""
    fpath = args.get("file_path") or args.get("path") or ""
    if fpath:
        fpath = _relativize_repo_paths(fpath)
    if pattern:
        hint = f"'{pattern[:80]}'" + (f" in {fpath}" if fpath else "")
        return hint[:200]
    symname = args.get("name") or ""
    if symname:
        hint = f"'{symname[:80]}'" + (f" in {fpath}" if fpath else "")
        return hint[:200]
    if fpath:
        return fpath[:200]
    # find_symbol / rag_search etc. query series
    query = args.get("query") or ""
    if query:
        return f"'{query[:80]}'"
    return ""


# Write tools to show in preview — the [POST-EDIT DIFF] block at the end of results is the key signal
_WRITE_PREVIEW_TOOLS = frozenset(
    {
        "apply_patch",
        "modify_symbol",
        "edit_text",
        "anchor_edit",
        "edit_ast",
        "write_plan",
    }
)

# Read/analysis tools whose result structure is "item listing" — 3 lines is more useful
_THREE_LINE_PREVIEW_TOOLS = frozenset(
    {
        "grep",
        "glob",
        "find_relevant_files",
        "find_references",
        "search_web",
        "analyze_change_impact",
        "run_structural_scan",
        "get_project_info",
        "query_dependency_graph",
        "get_file_outline",
        "find_symbol",
    }
)


def _select_preview_lines(tool: str, lines: list) -> list:
    """Select preview lines per tool — pick the most informative line(s) from each result.

    2 lines by default, 3 for listing-style tools. Write tools prioritize the
    [POST-EDIT DIFF] block at the end of the result (per-path +N/-M, NO CHANGE
    warnings).
    """
    # grep-type noise: remove pycache and other binary match lines
    lines = [ln for ln in lines if not ln.strip().startswith("Binary file ")]

    if tool == "find_relevant_files":
        # Header ("Top N relevant file(s) for: ...") duplicates cmd hint — keep result items only
        lines = [ln for ln in lines if not ln.strip().startswith("Top ")]
        return lines[:3]

    if tool == "bash":
        # `ls -la`'s "total N" line carries no information
        if lines:
            _first = lines[0].strip()
            if _first.startswith("total ") and _first[6:].strip().isdigit():
                lines = lines[1:]
        return lines[:3]

    if tool in _WRITE_PREVIEW_TOOLS:
        # Key signal is [POST-EDIT DIFF] block: what changed where / NO CHANGE warnings
        for _i, ln in enumerate(lines):
            if ln.strip().startswith("[POST-EDIT DIFF]"):
                _head = lines[:1] if _i > 0 else []
                return (_head + lines[_i + 1 : _i + 4])[:4]
        return lines[:2]

    if tool == "update_plan":
        # Line 1: status change summary (diff). Line 2: in-progress items ([~]) or Goal
        out = lines[:1]
        _cur = next((ln for ln in lines[1:] if ln.strip().startswith("[~]")), None)
        _second = _cur or next((ln for ln in lines[1:] if ln.strip().startswith("Goal:")), None)
        if _second:
            out.append(_second)
        return out

    if tool in _THREE_LINE_PREVIEW_TOOLS:
        return lines[:3]
    return lines[:2]


_INTERRUPT_RESUME_INSTRUCTION = (
    "(The user paused this task with ESC. If the next user input intends to "
    "continue this task — whether phrased like '계속'/'이어서'/'continue' or "
    "implied by context — resume from where it was interrupted, using the "
    "records above instead of repeating searches/reads that were already done. "
    "If it is an unrelated new request, handle that request instead.)"
)

_PAUSED_HINT = '⏸ paused — to resume, just ask naturally in your next input (e.g. "continue").'


def _abbrev_tokens(n: int) -> str:
    """Abbreviate token counts for display: 690 → '690', 43,606 → '43.6K', 11,377,708 → '11.38M'."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K"
    return f"{n / 1_000_000:.2f}M"


def _build_interrupt_note(partial_res) -> str:
    """Build the assistant interrupt note to leave in the session when ESC interrupts a turn.

    The note body carries only the partial response and resume instructions —
    the *full detail* of the tool-loop results is persisted separately via
    add_turn's tool_results argument, and build_context_messages fully renders
    it within the budget cap (see _TOOL_RESULTS_HEADER below). So if this
    function put a 300-char summary in the body, it would duplicate the full
    render; instead it leaves only the tool count as a guide. A short summary
    fallback is also provided for legacy/edge cases where tool_results
    persistence fails (only meaningful when the full render is empty).
    """
    content = (getattr(partial_res, "content", "") or "").strip()
    tool_results = list(getattr(partial_res, "tool_results", None) or [])
    parts: list[str] = []

    if tool_results:
        # Full tool_results are persisted and fully rendered, so body only gets a count guide.
        parts.append(
            f"[Interrupted during tool loop — {len(tool_results)} tool call(s) "
            f"executed before interruption; full results are attached below.]"
        )

    if content:
        parts.append(f"[Partial response at interruption]\n{content[:2000]}")

    parts.append(_INTERRUPT_RESUME_INSTRUCTION)
    return "\n\n".join(parts)


def _load_dotenv(repo_root: str) -> None:
    """Load .env file from *repo_root* into os.environ (manual parser, no dependency).

    Only sets keys not already set in the environment, so existing env vars win.
    """
    dotenv_path = os.path.join(repo_root, ".env")
    # Anything already exported by the shell wins over .env (below). Remember
    # which keys those were: a key persisted to .env can never take effect while
    # the shell still exports a stale value, and the user has to be told.
    _SHELL_PROVIDED_ENV_KEYS.update(os.environ)
    # .env missing is fine
    with contextlib.suppress(FileNotFoundError), open(dotenv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key.startswith("export "):  # export KEY=val format
                key = key[7:].strip()
            # strip inline comment and outer quotes
            if val.startswith(('"', "'")):
                q = val[0]
                close = val.find(q, 1)
                if close > 0:
                    after = val[close + 1 :].strip()
                    if after.startswith("#"):
                        logging.getLogger(__name__).debug(
                            "_load_dotenv: inline comment stripped in %s (was %r → %r)",
                            key,
                            val,
                            val[: close + 1],
                        )
                    # strip outer quotes (backslash-escaped quote guard)
                    _n_bs = 0
                    _i = close - 1
                    while _i >= 0 and val[_i] == "\\":
                        _n_bs += 1
                        _i -= 1
                    if _n_bs % 2 == 0:  # closing quote is not escaped
                        val = val[1:close]
                # else: malformed — keep as-is
            else:
                # Inline comment = '#' preceded by whitespace (python-dotenv
                # semantics). A '#' without preceding whitespace is part of
                # the value — e.g. KEY=https://host/path#frag must keep its
                # fragment, not be truncated at it.
                _cut = 0 if val.startswith("#") else -1
                if _cut < 0:
                    for _i, _ch in enumerate(val[:-1]):
                        if _ch in " \t" and val[_i + 1] == "#":
                            _cut = _i
                            break
                if _cut >= 0:
                    logging.getLogger(__name__).debug(
                        "_load_dotenv: inline comment stripped in %s (was %r → %r)",
                        key,
                        val,
                        val[:_cut].rstrip(),
                    )
                    val = val[:_cut].rstrip()
            if key and key not in os.environ:
                os.environ[key] = val


def _maybe_show_update_notice() -> None:
    """Show a non-blocking PyPI update hint to stderr (interactive modes only).

    Runs the check once per day (rate-limited via an on-disk cache) on a daemon
    thread so it never blocks startup. The notice goes to **stderr** — never
    stdout — so machine consumers (--json/--json-stream) are unaffected even if
    this were reached by mistake. Any error is swallowed: the update check must
    never impair the CLI itself.
    """
    with contextlib.suppress(Exception):  # fail-open: update check must never break the CLI
        from utils.version_check import start_update_check

        handle = start_update_check()
        notice = handle.collect(wait_s=0.0)  # fully non-blocking: cached-only
        if notice:
            sys.stderr.write(notice + "\n")
            sys.stderr.flush()


def main() -> None:
    # ── Collaboration subcommands (collaborate/mcp) ────────────────────────
    if len(sys.argv) > 1 and sys.argv[1] in ("collaborate", "mcp"):
        from external_llm.repl.collaborate.cli import main as collaborate_main

        collaborate_main()
        return

    parser = argparse.ArgumentParser(
        description="asicode Interactive CLI — direct engine connection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    from utils.version_check import get_current_version as _get_ver

    _ver = _get_ver()
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"asicode {_ver}"
            if _ver != "0.0.0"
            else "asicode 0.0.0 (uninstalled source checkout — pip install for a real version)"
        ),
    )
    parser.add_argument("--repo", "-r", metavar="PATH", help="Repository root path (default: current directory)")
    parser.add_argument("--prompt", "-p", metavar="TEXT", help="Single request text (default: REPL mode)")
    parser.add_argument("--prompt-file", metavar="FILE", help="Read request text from file")
    parser.add_argument(
        "--json", action="store_true", help="Output result as JSON (for machine consumption / Tenet integration)"
    )
    parser.add_argument(
        "--json-stream",
        dest="json_stream",
        action="store_true",
        help="Stream turn/tool events as newline-delimited JSON (NDJSON) during the "
        "run, ending with a 'result' event line (for Tenet live progress). "
        "Implies machine-readable stdout; the final line carries the same "
        "payload as --json.",
    )
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Run in single-shot multi-agent orchestration mode (F5): decompose the "
        "request into sub-tasks and dispatch them to sub-agent workers (IPC), "
        "instead of the default single AgentLoop. Combined with --json-stream, "
        "subagent_start / subagent_complete / heartbeat progress events stream "
        "as NDJSON so automation (Tenet) can watch the multi-agent run live.",
    )
    parser.add_argument("--prompt-stdin", action="store_true", help="Read prompt from stdin (for Tenet integration)")
    parser.add_argument("--provider", metavar="NAME", help="LLM provider (CLI arg > EXTERNAL_LLM_PROVIDER)")
    parser.add_argument("--model", "-m", metavar="NAME", help="LLM model name (CLI arg > EXTERNAL_LLM_MODEL)")
    parser.add_argument("--api-key", metavar="KEY", help="API key (CLI arg > env var)")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_cfg.counts.AGENT_MAX_TURNS_DEFAULT,
        help=f"Max agent turns (default: {_cfg.counts.AGENT_MAX_TURNS_DEFAULT})",
    )
    parser.add_argument(
        "--scoped-verification",
        action="store_true",
        help=(
            "After edits, run only tests likely affected by changed files "
            "(naming-convention + call-graph) instead of the full suite. "
            "Empty selection falls back to the full suite (safe). "
            "Also set via ASICODE_SCOPED_VERIFICATION=1."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose log output")
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "NONE"],
        help="Engine log level (DEBUG/INFO/WARNING/ERROR/NONE, default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default="logs/run_{date}_{time}.log",
        help=(
            "Path to save log file. "
            "{date}→YYYYMMDD, {time}→HHMMSS substitution supported. "
            "Default: logs/run_{date}_{time}.log"
        ),
    )
    parser.add_argument(
        "--no-deps-check",
        action="store_true",
        help="Skip the interactive semantic-validation tool check at startup",
    )
    parser.add_argument(
        "--subagent",
        action="store_true",
        help=(
            "Run as a sub-agent worker: poll .asicode/subagents/<id>/task.json, "
            "execute the task, write result.json back. "
            "Use with --subagent-id to set the worker identity. "
            "Launched automatically by /orchestrate on macOS."
        ),
    )
    parser.add_argument(
        "--subagent-id",
        metavar="ID",
        help="Sub-agent worker ID (used with --subagent). Polls .asicode/subagents/<ID>/task.json.",
    )
    parser.add_argument(
        "--orch-pid",
        type=int,
        default=0,
        metavar="PID",
        help="PID of the orchestrator process that spawned this worker (used "
        "with --subagent). Enables a direct liveness probe for orphan "
        "self-exit instead of relying on getppid(), which does not "
        "reflect the orchestrator on the macOS Terminal.app launch path "
        "(parent there is the login shell, not the orchestrator).",
    )

    args = parser.parse_args()

    # ── .env auto-loading (direct parsing, no python-dotenv) ──
    _repo_root = _resolve_repo_root(args.repo)
    _load_dotenv(_repo_root)

    # Terminal resize handling — update Rich console width (skipped on no-SIGWINCH platforms)
    if hasattr(signal, "SIGWINCH") and sys.stdout.isatty():
        # non-main thread registration etc. — operates without resize handling
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGWINCH, _handle_terminal_resize)

    # Logging config — output engine internal logs to terminal, same as unicorn server
    if args.log_level != "NONE":
        _setup_logging(args.log_level, log_file=args.log_file or None)
    elif args.log_file:
        # Allow file save even if NONE (only disable terminal output, file is saved)
        _setup_logging("DEBUG", log_file=args.log_file)

    # ── Load model from config.json (CLI args > config.json > env vars) ──
    # Common to all modes (subagent included): CLI --model/provider takes precedence.
    # Read from git toplevel (_repo_root), NOT cwd — this MUST match run_repl()'s
    # write path so /model (and /think, /helper, /dev, /code) persistence survives
    # launches from a subdirectory under the repo. See _resolve_repo_root.
    _shared_cfg_path = os.path.join(_repo_root, ".asicode", "config.json")
    _saved_cfg_path = _shared_cfg_path
    # Per-terminal isolation: a TTY-attached terminal reads/writes its own
    # config file (seeded from the shared one) so /model switches stay local.
    _term_cfg = _terminal_config_path(_repo_root)
    if _term_cfg:
        _seed_terminal_config(_term_cfg, _shared_cfg_path)
        _saved_cfg_path = _term_cfg
    try:
        with open(_saved_cfg_path, encoding="utf-8") as _cf:
            _saved_cfg = json.load(_cf)
    except (FileNotFoundError, json.JSONDecodeError):
        _saved_cfg = {}
    if not args.provider:
        args.provider = _saved_cfg.get("provider", "")
    if not args.model:
        args.model = _saved_cfg.get("model", "")
    # thinking_state / reasoning_effort (no CLI flag — config.json only)
    args.thinking_mode = _saved_cfg.get("thinking_state")
    args.reasoning_effort = _saved_cfg.get("reasoning_effort")
    # Final fallback: environment variables
    if not args.provider:
        args.provider = os.getenv("EXTERNAL_LLM_PROVIDER", "")
    if not args.model:
        args.model = os.getenv("EXTERNAL_LLM_MODEL", "")

    # ── Sub-agent worker mode: poll task.json, run, write result.json ──
    # Launched by /orchestrate (auto_launch_terminal) or manually:
    #   asi --subagent --subagent-id <id> --provider ... --model ...
    # NOTE: config.json/env var resolution is above, so manual testing
    # without --provider/--model also works.
    if args.subagent:
        run_subagent_worker(args)
        return

    # --prompt-file
    if args.prompt_file and not args.prompt:
        try:
            args.prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        except OSError as e:
            _print(f"file read error: {e}", _C["red"])
            sys.exit(1)

    # --prompt-stdin: read prompt from stdin (for Tenet integration)
    if args.prompt_stdin:
        if args.prompt:
            _print("error: --prompt-stdin is mutually exclusive with --prompt / --prompt-file", _C["red"])
            sys.exit(1)
        try:
            args.prompt = sys.stdin.read().strip()
        except OSError as e:
            _print(f"stdin read error: {e}", _C["red"])
            sys.exit(1)
        if not args.prompt:
            _print("error: --prompt-stdin: empty input from stdin", _C["red"])
            sys.exit(1)

    # NOTE: The language-aware dependency check + install prompt now happens
    # exactly once, inside run_repl() → _print_dep_status(), so that the
    # resolved tool state (found/skipped) feeds directly into the status line.
    # Doing it here too would double-prompt the user.  --no-deps-check is
    # forwarded via args and honored by _print_dep_status.

    # --json / --json-stream require a prompt (single-shot mode)
    if (args.json or getattr(args, "json_stream", False)) and not args.prompt:
        _print("error: --json/--json-stream requires --prompt, --prompt-file, or --prompt-stdin", _C["red"])
        sys.exit(1)

    # ── Non-blocking PyPI update check (interactive modes only) ────────────
    # Skipped for machine consumers (--json/--json-stream: stdout must stay
    # clean) and sub-agent workers (no human to read it). The notice is written
    # to stderr regardless, but we avoid even starting the check here.
    if not (args.json or getattr(args, "json_stream", False) or getattr(args, "subagent", False)):
        _maybe_show_update_notice()

    if args.prompt:
        sys.exit(run_once(args, args.prompt))
    else:
        run_repl(args)

    # Restore default SIGINT handler before interpreter shutdown.
    # Python 3.14+ asyncio installs a SIGINT handler that raises
    # KeyboardInterrupt during atexit → ThreadPoolExecutor join,
    # which conflicts with the threading shutdown sequence and
    # produces a spurious traceback.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# ─── P6-2: REPL implementation moved to external_llm/repl/repl_impl.py ───────────
# Barrel re-export: keeps ``import asi; asi.run_repl`` and ``from asi import X``
# call sites working. Imported at the very bottom so repl_impl's ``import asi``
# always observes a fully initialized module (deliberate one-way cycle).
from external_llm.repl.repl_impl import (  # noqa: E402 — bottom-of-file barrel re-export (one-way cycle)
    _AUTO_CONTINUE_DELAY,
    _AUTO_NEXT_SUGGEST_SYSTEM,
    _AUTO_SUGGESTION_MAX_LEN,
    _NEXT_SUGGEST_SYSTEM,
    _REPO_ROOT,
    _active_spinner_printer,
    _auto_continue_should_arm,
    _auto_continue_state,
    _auto_countdown_active,
    _auto_submit_gen,
    _auto_submit_now,
    _build_engine,
    _build_json_output,
    _build_orchestrator_digest,
    _build_turn_digest,
    _cancel_auto_submit,
    _cjk_width,
    _cli_checkpoint_cb,
    _collect_input,
    _completer_dev_models,
    _completer_model,
    _completer_provider,
    _deliver_next_suggestion,
    _dropped_entries,
    _esc_watcher_pause,
    _eval_ctrlc_armed,
    _extract_patched_file,
    _finalize_pending_design_chat,
    _format_result,
    _get_ollama_models,
    _init_repl_engine,
    _init_session_state,
    _input_underline,
    _insights_compact_is_noop,
    _interactive_provider_setup,
    _invalidate_next_suggestion,
    _json_error_output,
    _json_stream_emit,
    _kick_next_prompt_suggestion,
    _last_input_was_auto,
    _list_provider_model_choices,
    _maybe_arm_auto_submit,
    _next_prompt_suggestion,
    _next_suggestion_gen,
    _notify_above_prompt,
    _ollama_cache,
    _ollama_cache_ts,
    _orchestrator_result_to_agent_like,
    _parse_auto_arg,
    _ProgressPrinter,
    _prompt_history_path,
    _prompt_input,
    _prompt_session,
    _resolve_repo_root,
    _result_output_dict,
    _retry_create_svc_with_api_key_prompt,
    _run_collaborate_session,
    _run_esc_watcher,
    _run_orchestrate_single_shot,
    _run_repl_impl,
    _run_with_cancel,
    _save_key_to_dotenv,
    _seed_terminal_config,
    _show_result,
    _size_compact_budget,
    _split_work_state,
    _terminal_config_path,
    _text_has_hangul,
    _turns_to_int,
    _validate_next_suggestion,
    _wrap_cjk,
    _wrap_preserve_code,
    run_once,
    run_repl,
    run_subagent_worker,
)

__all__ = [
    "_AUTO_CONTINUE_DELAY",
    "_AUTO_NEXT_SUGGEST_SYSTEM",
    "_AUTO_SUGGESTION_MAX_LEN",
    "_NEXT_SUGGEST_SYSTEM",
    "_REPO_ROOT",
    "_ProgressPrinter",
    "_active_spinner_printer",
    "_auto_continue_should_arm",
    "_auto_continue_state",
    "_auto_countdown_active",
    "_auto_submit_gen",
    "_auto_submit_now",
    "_build_engine",
    "_build_json_output",
    "_build_orchestrator_digest",
    "_build_turn_digest",
    "_cancel_auto_submit",
    "_cjk_width",
    "_cli_checkpoint_cb",
    "_collect_input",
    "_completer_dev_models",
    "_completer_model",
    "_completer_provider",
    "_deliver_next_suggestion",
    "_dropped_entries",
    "_esc_watcher_pause",
    "_eval_ctrlc_armed",
    "_extract_patched_file",
    "_finalize_pending_design_chat",
    "_format_result",
    "_get_ollama_models",
    "_init_repl_engine",
    "_init_session_state",
    "_input_underline",
    "_insights_compact_is_noop",
    "_interactive_provider_setup",
    "_invalidate_next_suggestion",
    "_json_error_output",
    "_json_stream_emit",
    "_kick_next_prompt_suggestion",
    "_last_input_was_auto",
    "_list_provider_model_choices",
    "_maybe_arm_auto_submit",
    "_next_prompt_suggestion",
    "_next_suggestion_gen",
    "_notify_above_prompt",
    "_ollama_cache",
    "_ollama_cache_ts",
    "_orchestrator_result_to_agent_like",
    "_parse_auto_arg",
    "_prompt_history_path",
    "_prompt_input",
    "_prompt_session",
    "_resolve_repo_root",
    "_result_output_dict",
    "_retry_create_svc_with_api_key_prompt",
    "_run_collaborate_session",
    "_run_esc_watcher",
    "_run_orchestrate_single_shot",
    "_run_repl_impl",
    "_run_with_cancel",
    "_save_key_to_dotenv",
    "_seed_terminal_config",
    "_show_result",
    "_size_compact_budget",
    "_split_work_state",
    "_terminal_config_path",
    "_text_has_hangul",
    "_turns_to_int",
    "_validate_next_suggestion",
    "_wrap_cjk",
    "_wrap_preserve_code",
    "run_once",
    "run_repl",
    "run_subagent_worker",
]


if __name__ == "__main__":
    main()
