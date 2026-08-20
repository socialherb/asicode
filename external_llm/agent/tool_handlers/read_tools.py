"""Read-only tool handlers for ToolRegistry."""
from __future__ import annotations

import codecs
import functools
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...common.indent_utils import INDENT_GUTTER_BAR, format_numbered_line
from ...common.subprocess_utils import CANCEL_POLL_INTERVAL as _CANCEL_POLL_INTERVAL
from ...common.subprocess_utils import cancel_probe as _cancel_probe
from ...common.text_reading import read_line_window as _read_line_window
from ..bm25 import bm25_rank
from ..cancel_scope import current_cancel_event
from ..config.thresholds import config as _cfg
from ..rag_configs import CodeTokenizer

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
# Re-exported under the historical private names: the renderer moved to
# common/indent_utils so the write tools' failure previews can use it too
# without read_tools <-> write_tools becoming an import cycle (read_tools
# already imports _repo_file_index from write_tools).
_INDENT_GUTTER_BAR = INDENT_GUTTER_BAR
_format_numbered_line = format_numbered_line

# Method names listed per class in read_file's over-cap outline. Matches the
# get_file_outline tool's own cap so the two views of a file agree.
_METHODS_PER_CLASS = 15


def _outline_extent(sym: Any) -> str:
    """Render a symbol's line extent for an outline row.

    ``"120-450"`` when the end line is known, a bare ``"120"`` when it is not.
    Both forms occur: every AST-backed outline (Python, TS/JS, tree-sitter for
    the rest) carries ``end_line``, but ``_outline_ripgrep`` — the fallback for
    a language with no installed grammar — matches a declaration by regex and
    has no extent to report. Degrading to the start alone keeps that path
    exactly as useful as it was rather than printing a fabricated range.

    ``end_line > line`` rather than ``!= None``: a one-line symbol renders as a
    bare line number, since "300-300" reads like a mistake and says no more.
    """
    end = getattr(sym, "end_line", None)
    if end and end > sym.line:
        return f"{sym.line}–{end}"
    return str(sym.line)


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


# ── Binary detection for read_file ─────────────────────────────────────────
# read_file decoded every file as UTF-8 with errors="replace", so a .png/.pyc/
# .so came back as thousands of U+FFFD replacement characters. That reads as
# *content*: the model spends a turn interpreting it, and the write tools will
# happily accept an edit against that garbled view. Sniff a prefix instead and
# say what the file is. The prefix is read from the same handle that then reads
# the body, so a text file costs no extra I/O and a 2 GB pack file is never
# read in full just to be rejected.
_BINARY_SNIFF_BYTES = 8192

# Bytes a text file may legitimately contain: printable ASCII, every byte a
# UTF-8 multibyte sequence can use (0x80-0xFF, which also keeps legacy cp949 /
# latin-1 sources out of the binary bucket), and the control codes that really
# do appear in sources and captured terminal output.
_TEXTUAL_BYTES = bytes(
    sorted(
        {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1B}  # BS TAB LF VT FF CR ESC
        | set(range(0x20, 0x7F))                     # printable ASCII
        | set(range(0x80, 0x100))                    # UTF-8 lead/continuation
    )
)

# Share of the sniffed prefix allowed to be non-textual control bytes before
# the file is called binary. Real text sits at ~0%; this only has to separate
# that from formats carrying no NUL in their first 8 KiB.
_BINARY_CONTROL_RATIO = 0.10

# UTF-16/32 text is mostly NUL bytes, so the NUL rule alone would call it
# "binary" — true of its UTF-8 decode, but useless guidance. A BOM names the
# real encoding, so report that instead. The UTF-32 BOMs must be tested before
# the UTF-16 ones they begin with.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, ""),  # "" = genuine UTF-8 text; not binary
    (codecs.BOM_UTF32_LE, "UTF-32 LE"),
    (codecs.BOM_UTF32_BE, "UTF-32 BE"),
    (codecs.BOM_UTF16_LE, "UTF-16 LE"),
    (codecs.BOM_UTF16_BE, "UTF-16 BE"),
)

# Extensions read_image can OCR — the alternative worth naming by tool.
_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"})

_BINARY = "binary"

# How long a search process may take to EXIT after it has closed stdout. Kept
# apart from the search timeout because they bound different things: the search
# budget bounds how long the tool may look, this bounds teardown after it has
# already answered. See _run_search_bounded.
_EXIT_GRACE = 5.0


# Above this size read_file streams instead of materialising. Below it the whole
# file is read and split in one go, which is what read_file has always done and
# is measurably the faster of the two at that scale — the streaming path exists
# for the case where materialising costs hundreds of megabytes, not to replace a
# working fast path. See _stream_split_window.
_STREAM_ABOVE_BYTES = 1 << 20
# Bytes per read while streaming.
_STREAM_CHUNK = 1 << 16
# Stands in for "to the end of the file" when the caller gave a start_line
# but no end_line — the retained window still has to be an integer bound.
_MAX_LINE_INDEX = 1 << 62


class SearchCancelled(Exception):  # noqa: N818 — Cancelled-suffix convention
    """The user cancelled a running search. Distinct from a timeout.

    They mean different things to the caller and must not share a type: a
    timeout is the tool giving up and is reported as a (successful) empty
    answer, while a cancel is the user withdrawing the request and has to come
    back as `ok=False` — reported as success it reads as "asked and answered"
    and the model moves on as if the search had found nothing.
    """


# Per-file access failures. Their presence means the search RAN and skipped
# individual files — the tree was still searched, so whatever came back is a
# real answer.
_ACCESS_ERROR_MARKERS = (
    "permission denied",
    "no such file or directory",
    "is a directory",
    "operation not permitted",
    "too many levels of symbolic links",
    "input/output error",
    "device or resource busy",
)
# Pattern-compile failures. These happen BEFORE any file is opened, so the
# search never ran and retrying as a fixed string is the right move.
_PATTERN_ERROR_MARKERS = (
    "regex parse error",
    "invalid regular expression",
    "unclosed group",
    "unmatched",
    "unterminated",
    "repetition operator",
    "trailing backslash",
    "brackets ([ ]) not balanced",
    "parentheses not balanced",
)


def _search_ran_despite_errors(stderr: str) -> bool:
    """True when exit 2 came from unreadable FILES, not from a bad pattern.

    Both ripgrep and grep exit 2 for "an error occurred", which conflates two
    opposite outcomes: a pattern that never compiled (nothing was searched) and
    a tree that was searched fine except for a file the process could not open.
    Treating the second as the first discarded real matches — measured against
    the live handler, one ``chmod 000`` file next to a matching one turned the
    whole call into ``ok=False, "grep failed (exit=2): Permission denied"`` and
    the match was never reported. Root-owned files under a Docker bind mount
    make that an ordinary repo state, not a corner case.

    The markers are checked on the ERROR class rather than the success class
    because ripgrep emits pattern errors before it opens anything, so the two
    sets cannot co-occur; requiring an access marker AND no pattern marker
    stays correct even when stderr was truncated by the 64 KB drain cap.
    """
    low = (stderr or "").lower()
    if any(m in low for m in _PATTERN_ERROR_MARKERS):
        return False
    return any(m in low for m in _ACCESS_ERROR_MARKERS)


def _unsupported_flag(stderr: str) -> str | None:
    """The rejected long flag when the child rejected one, else None.

    ``--max-columns-preview`` is ripgrep >= 12.0 (2020-03); Ubuntu 20.04 still
    ships 11.0.2. An unknown flag is an exit 2 with no output, which lands in
    the same conflated bucket as everything else above — so without this the
    whole grep tool dies on those hosts, and never falls back to the system
    grep whose code path is sitting right there.
    """
    low = (stderr or "").lower()
    if "unrecognized flag" not in low and "unknown flag" not in low:
        return None
    _m = re.search(r"(--[a-z0-9-]+)", stderr or "")
    return _m.group(1) if _m else ""


def _stream_split_window(
    fh, prefix: bytes, first: int, last: int, retain_chars: int, *,
    stop_after_last: bool = False, cut_overflow_line: bool = False,
):
    """``(total_lines, lines[first-1:last])`` without holding the whole file.

    read_file materialised every byte of a file to answer questions that need
    almost none of it. Measured on a 108 MB log: +509 MB of peak RSS to produce
    a "too long, use start_line" refusal, and the same again for the ranged read
    the refusal invites. The bytes, the decoded string and a list of two million
    line objects all existed at once.

    The split has to stay ``str.splitlines()``, exactly — read_file's line
    numbers are its own, and anchor_edit's ``anchor_ast_lineno`` mode builds a
    matching array, so the semantics are load-bearing across tools. That rules
    out splitting on ``\\n``: ``splitlines`` also breaks on ``\\v \\f \\x1c \\x1d
    \\x1e \\x85 \\u2028 \\u2029``, and a file containing any of them would
    silently renumber.

    So the chunk boundary is the whole problem, and there are three of them:
    a line straddling two reads, a multibyte sequence straddling two reads, and
    ``\\r\\n`` straddling two reads — the last being the only ambiguous one,
    since a trailing ``\\r`` is a complete break until an ``\\n`` arrives and
    makes it half of one. A part is therefore carried forward when it is
    unterminated OR ends with a lone ``\\r``.

    Verified rather than argued: 20,000 random texts built from every break
    character, multibyte codepoints and invalid UTF-8, at five chunk sizes,
    against ``raw.decode("utf-8", errors="replace").splitlines()``.

    Only the NEW chunk is ever split. The carried part is held as a list of
    pieces and joined once, when the line completes — because prepending it to
    the next chunk and re-splitting the result is quadratic in the length of a
    line, and a file with few newlines is a real input, not a hypothetical one
    (a minified bundle, a .map, one-line JSON). Measured on a 34 MB single-line
    file: 11.10 s, 9.81 s of it inside 1,042 ``splitlines()`` calls over an
    ever-growing buffer, ~1.1 GB of transient strings. Splitting the chunk
    alone also keeps the ``\\r``-boundary probe bounded by the chunk, since it
    is that probe re-scanning the accumulated carry that produced half the
    calls. The only break that can straddle a boundary is ``\\r\\n``, so it is
    the only case the seam has to reason about.

    ``retain_chars`` bounds the window itself, because "give me lines 1 to
    2000000" is a request the caller can make and the char budget downstream
    will cut long before that. The count keeps going after retention stops, so
    the total is still exact.

    A line wider than ``retain_chars`` is DROPPED by default and cut only under
    ``cut_overflow_line``, because whether a prefix may be passed off as a line
    depends on whether the caller can say that it is one. read_file can: the
    budget that decides its output is :func:`_apply_char_budget` downstream,
    which emits a prefix and reports ``partial_line`` — so dropping here just
    starved it, and on a 34 MB one-liner the bulk path returned 60,355 chars
    plus that metadata while this path returned an empty code block and ``{}``
    (indistinguishable from "the file is empty"). The context-snippet caller
    cannot: ``retain_chars`` IS its final budget and its numbered output has no
    field to announce a partial line, so a silent prefix would read as the
    whole line. Truncation is safe exactly where it can be declared.

    ``stop_after_last`` (opt-in, default False) stops reading once the window
    is final — line *last* fully emitted (or retention full), so anything
    still carried is already beyond the window — and callers that discard
    ``total_lines`` (e.g. a bounded context snippet) get O(last) I/O instead
    of one full pass. With it, the returned total is only exact up to the stop
    point; read_file's refusal guidance keeps the default and its
    count-to-EOF contract.
    """
    import codecs as _codecs

    _dec = _codecs.getincrementaldecoder("utf-8")("replace")
    # Pieces of the line in progress. Holds no break character except a
    # possible trailing lone '\r' (_carry_cr), because a part is only carried
    # when it is unterminated or '\r'-terminated.
    _carry: list[str] = []
    _carry_cr = False
    _total = 0
    _window: list[str] = []
    _kept = 0
    _full = False

    def _keep(part: str) -> None:
        nonlocal _kept, _full
        _stripped = part.splitlines()
        _line = _stripped[0] if _stripped else ""
        if _kept + len(_line) > retain_chars:
            # Cut to what is left, for the caller that can declare the cut.
            # Dropping made read_file answer a 34 MB one-line file with an
            # empty code block — indistinguishable from "this file is empty" —
            # while the bulk path emitted a prefix and set partial_line.
            _room = retain_chars - _kept
            if cut_overflow_line and _room > 0:
                _window.append(_line[:_room])
                _kept = retain_chars
            _full = True
            return
        _window.append(_line)
        _kept += len(_line)

    def _emit(part: str) -> None:
        nonlocal _total
        _total += 1
        if _full or not (first <= _total <= last):
            return
        _keep(part)

    def _flush_carry() -> None:
        """Complete the carried line: count it, and join it only if kept."""
        nonlocal _carry, _carry_cr, _total
        if not _carry:
            _carry_cr = False
            return
        _total += 1
        # The join is what materialises a wide line, so an out-of-window line
        # is counted without ever being built — the case that made a whole-file
        # carry expensive even when nothing from it could be returned.
        if not _full and first <= _total <= last:
            _keep("".join(_carry))
        _carry = []
        _carry_cr = False

    _raw = prefix
    while True:
        if _raw:
            _chunk = _dec.decode(_raw)
            if _carry_cr:
                # The one ambiguous seam: a trailing '\r' is already a break
                # unless this chunk opens with the '\n' that makes it a CRLF.
                if _chunk[:1] == "\n":
                    _carry.append("\n")
                    _chunk = _chunk[1:]
                _flush_carry()
            _parts = _chunk.splitlines(keepends=True)
            _tail_carry = ""
            if _parts:
                _tail = _parts[-1]
                # Bounded by one chunk, never by the accumulated carry.
                if _tail.splitlines()[0] == _tail or _tail.endswith("\r"):
                    _tail_carry = _parts.pop()
            if _carry:
                if _parts:
                    # The carried line ends at the first complete line here.
                    _carry.append(_parts.pop(0))
                    _flush_carry()
                elif _tail_carry:
                    # No break anywhere in this chunk — the line goes on.
                    _carry.append(_tail_carry)
                    _tail_carry = ""
            if _parts:
                if _full or _total >= last or _total + len(_parts) < first:
                    # Nothing in this chunk can be retained — it is entirely
                    # before the window, or past it, or retention is already
                    # full. Only the count is still owed, and a per-line Python
                    # call to produce it is the whole cost at scale: a 108 MB
                    # file is 2M calls, ~0.9 s against a 0.13 s bulk split.
                    # (Counting break characters instead of splitting was
                    # measured and is 3x SLOWER — ten passes with str.count
                    # lose to one optimised split.)
                    _total += len(_parts)
                else:
                    for _part in _parts:
                        _emit(_part)
            if _tail_carry:
                _carry.append(_tail_carry)
            _carry_cr = bool(_carry) and _carry[-1].endswith("\r")
        if stop_after_last and (_full or _total >= last):
            break
        _raw = fh.read(_STREAM_CHUNK)
        if not _raw:
            break
    # The decoder's final flush yields at most a replacement character, never a
    # break, so the carry is still at most one logical line.
    _rest = _dec.decode(b"", True)
    if _rest:
        _carry.append(_rest)
    _flush_carry()
    return _total, _window


def _classify_binary(head: bytes) -> str | None:
    """Why ``head`` is not UTF-8 text, or ``None`` if it is.

    Returns the BOM-declared encoding name ("UTF-16 LE") when the file is text
    in another encoding, or ``_BINARY`` when it is not text at all. ``head`` is
    only a prefix, so this is a heuristic — the one git uses (a NUL in the
    first 8 KiB), widened to read a BOM and to catch headers carrying no NUL.
    """
    if not head:
        return None
    for bom, label in _BOM_ENCODINGS:
        if head.startswith(bom):
            return label or None
    if b"\x00" in head:
        return _BINARY
    non_text = len(head.translate(None, _TEXTUAL_BYTES))
    return _BINARY if non_text / len(head) > _BINARY_CONTROL_RATIO else None


def _binary_guidance(path: str, size: int, verdict: str) -> str:
    """Explain the refusal and name the tool that *can* read ``path``.

    Both branches state that no content follows, so the message is never
    mistaken for a short file.
    """
    if verdict != _BINARY:
        codec = verdict.replace(" ", "").lower()
        return (
            f"`{path}` is text encoded as {verdict}, not UTF-8 ({size:,} bytes) — "
            f"no content is shown, because decoding it as UTF-8 returns mojibake. "
            f"Convert a copy first: bash `iconv -f {codec} -t utf-8 {path}`."
        )
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    hint = (
        "Use read_image to OCR it."
        if ext in _IMAGE_EXTS
        else "Identify or inspect it with bash (`file`, `xxd -l 256`)."
    )
    return (
        f"`{path}` is a binary file ({size:,} bytes) — no content is shown, "
        f"because reading it as UTF-8 returns replacement characters rather than "
        f"anything editable. {hint}"
    )


def _read_symbol_window(path: Path, start: int, count: int) -> list[str]:
    """Stream lines ``[start, start+count)`` of ``path`` (0-based).

    Replaces the previous whole-file ``read_text`` + ``split("\n")`` in
    ``_tool_read_symbol`` (P25-1): the AST index already knows the symbol's
    exact line span, so loading the entire file — including every line before
    the window — was pure waste on files with symbols near the end.

    Thin wrapper over :func:`common.text_reading.read_line_window` — the
    canonical implementation moved there (P26-3) so symbol_search's Go
    signature-line read shares the exact same window semantics instead of a
    drifting copy.
    """
    return _read_line_window(path, start, count)


def _count_file_lines(path: Path) -> int:
    """Total line count of ``path`` via a streaming pass (O(1) memory).

    Only used on the read_symbol truncation path, where the model is told
    ``line_count`` (a file line number) so it can resume with read_file.
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def _apply_char_budget(
    numbered_lines: list[str], first_line: int, budget: int
) -> tuple[list[str], int | None, int | None]:
    """Cut ``numbered_lines`` to ``budget`` chars on a line boundary.

    ``first_line`` is the **1-based** source line number of ``numbered_lines[0]``
    — the same origin the lines were numbered with. Returns
    ``(kept, truncated_at, partial_line)``:

    * ``truncated_at`` — 1-based number of the first line NOT emitted, i.e. the
      exact ``start_line=`` a caller passes to continue. ``None`` if all fit.
    * ``partial_line`` — 1-based number of a line emitted as a PREFIX only
      (that line alone exceeded the budget), else ``None``. Its tail cannot be
      recovered by re-reading, because ``start_line`` is line-granular, so
      callers MUST say so instead of offering a plain "continue here".

    Shared by read_file and read_symbol. Duplicating this arithmetic is exactly
    how read_symbol reintroduced both bugs 64a36e1c had already fixed for
    read_file: an origin off by one (naming a line that WAS emitted, so the
    caller re-reads it) and a resumption line that does not advance past an
    over-wide line (naming that same line forever). Both are only correct when
    derived from the same 1-based origin as the numbering, so the derivation
    lives here once.
    """
    if sum(len(ln) + 1 for ln in numbered_lines) <= budget:
        return numbered_lines, None, None

    kept: list[str] = []
    used = 0
    # The prefix sum exceeds the budget, so the loop always breaks; the
    # initialiser only keeps `truncated_at` bound for a type checker.
    truncated_at = first_line + len(numbered_lines)
    for i, ln in enumerate(numbered_lines):
        used += len(ln) + 1
        if used > budget:
            truncated_at = first_line + i  # first line NOT emitted
            break
        kept.append(ln)

    if not kept:
        # One line wider than the whole budget: emitting nothing and naming
        # this line as the resumption point is a loop (the retry returns the
        # same over-wide line). Emit a prefix and advance PAST it instead.
        return [numbered_lines[0][:budget]], first_line + 1, first_line
    return kept, truncated_at, None


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

    Raises ``ValueError`` for a syntactically invalid character class (e.g. a
    reversed range like ``[z-a]``), naming the pattern and the offending class
    in glob coordinates — never a raw ``re.error`` over the translated regex.
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
                negated = body.startswith(("!", "^"))
                if negated:
                    body = body[1:]
                # POSIX glob: backslash is literal inside a class, and a
                # leading `]` (even after `!`/`^`) is a member, not the
                # closer. Make both regex-safe so `[]]`/`[a\]`/`[!]]`
                # compile instead of surfacing a raw re.error. Literal `[`
                # and `&` are escaped too: Python 3.13+ nested sets and set
                # intersections would reinterpret a leading `[[` / any `&&`
                # (FutureWarning today, silent semantics change later).
                body = body.replace("\\", "\\\\")
                body = body.replace("[", "\\[").replace("&", "\\&")
                if body.startswith("]"):
                    body = "\\]" + body[1:]
                cls = ("^" if negated else "") + body
                try:
                    re.compile(f"[{cls}]")
                except re.error as exc:
                    raise ValueError(
                        f"invalid glob pattern {pattern!r}: bad character class "
                        f"[{pattern[i + 1:j]}] ({exc.msg})"
                    ) from exc
                out.append(f"[{cls}]")
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
        try:
            rx = _glob_to_regex(pattern)
        except ValueError as exc:
            # The translator raises in glob coordinates (original pattern +
            # offending class) so the model can see what it did wrong — a raw
            # re.error would reference the translated regex and send it into
            # a retry loop.
            return self._make_result(ok=False, content="", error=str(exc))
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

        start_line = args.get("start_line")
        end_line = args.get("end_line")

        # The window worth keeping, decided BEFORE the file is read. Without a
        # range that is the line cap — anything past it is refused, so reading it
        # is pure cost. With one it is the range asked for, bounded by twice the
        # char budget that will cut it downstream (numbered lines are longer than
        # raw ones, so the budget always binds first and `truncated_at` stays
        # exact).
        _first = max(1, int(start_line or 1)) if (start_line or end_line) else 1
        if start_line is None and end_line is None:
            _last = _cfg.lines.READ_FILE_FULL_LINES
        elif end_line is not None:
            _last = int(end_line)
        else:
            _last = _MAX_LINE_INDEX
        _retain = _cfg.lines.READ_FILE_MAX_CHARS * 2

        # Sniff a prefix before committing to the whole file, so a binary is
        # rejected without ever being read past its header. The body is read
        # from the same handle, so the text path pays for one open, not two.
        try:
            _size = abs_path.stat().st_size
            with abs_path.open("rb") as fh:
                head = fh.read(_BINARY_SNIFF_BYTES)
                verdict = _classify_binary(head)
                if verdict is not None:
                    return self._make_result(
                        ok=True,
                        content=_binary_guidance(path, _size, verdict),
                        metadata={"binary": True, "reason": verdict, "byte_size": _size},
                    )
                if _size > _STREAM_ABOVE_BYTES:
                    # Streaming answers the same questions off a bounded window.
                    # Measured on a 108 MB log, one cold process per row:
                    #   refusal   511.4 MB / 0.25 s  ->  0.8 MB / 0.18 s
                    #   lines 100-120   511.4 MB / 0.16 s  ->  0.2 MB / 0.12 s
                    # Identical output, and no slower — the file is read once
                    # either way, and the decoded copy is what cost the time.
                    _total, lines = _stream_split_window(
                        fh, head, _first, _last, _retain,
                        # _apply_char_budget below cuts and reports
                        # partial_line, so a line wider than _retain must
                        # arrive as a prefix, not as nothing — that is what
                        # the non-streaming branch hands it.
                        cut_overflow_line=True,
                    )
                else:
                    lines = (head + fh.read()).decode(
                        "utf-8", errors="replace",
                    ).splitlines()
                    _total = len(lines)
                    lines = lines[_first - 1:_last]
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to read {path!r}: {e}")

        if start_line is None and end_line is None:
            if _total > _cfg.lines.READ_FILE_FULL_LINES:
                return self._make_result(
                    ok=True,
                    content=self._over_cap_guidance(path, _total),
                    metadata={"over_line_cap": True, "line_count": _total},
                )
            s, e = 1, _total
        else:
            s = _first
            _requested_end = int(end_line) if end_line is not None else _total
            e = min(_total, _requested_end)
            # Two different mistakes, which the single "out of range" message
            # conflated — and it named the CLAMPED end, so asking for 9999-10005
            # of a 200-line file was reported as "range 9999-200", a range the
            # caller never asked for. Naming the wrong problem costs a turn: the
            # model reads "file has 200 lines" after an inverted range whose
            # bounds are both inside the file, and has nothing to act on.
            #
            # ok=False, not ok=True: a range that yields no lines is a failed
            # call, not an answer. Reported as success it reads as "asked and
            # answered" and the retry never happens — the same reasoning
            # _tool_read_symbol records for its own missing-argument case.
            # A zero or negative bound is malformed, not merely inverted, and
            # "swap them" would be useless advice for it (swapping end_line=0
            # gives start_line=0, equally wrong). Line numbers are 1-based.
            if end_line is not None and _requested_end < 1:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"end_line must be 1 or greater (got {_requested_end}); "
                        f"line numbers are 1-based. Omit end_line to read to the "
                        f"end of {path!r} ({_total} lines)."
                    ),
                )
            # Guarded on end_line being SUPPLIED: with it omitted the default is
            # the file's last line, and blaming an "end_line" the caller never
            # sent is the same misdirection this block exists to remove
            # (start_line=201 on a 200-line file is a past-the-end error, not an
            # inverted range).
            if end_line is not None and s > _requested_end:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"start_line {s} is after end_line {_requested_end} "
                        f"in {path!r} ({_total} lines). Swap them."
                    ),
                )
            if s > _total:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"start_line {s} is past the end of {path!r}, which has "
                        f"{_total} lines."
                    ),
                )

        numbered_lines = [_format_numbered_line(i, ln) for i, ln in enumerate(lines[: e - s + 1], start=s)]
        # Char budget. Applied here rather than only on the no-range path
        # because an explicit range is the documented way around the line cap,
        # so it is the path most able to overrun the context window.  Truncate
        # on a line boundary and name the resumption line, so continuing is one
        # unambiguous call rather than another guess.
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        # partial_line: emitted as a prefix only (mid-line truncation) — the
        # caller must be told its tail is unrecoverable, see _apply_char_budget.
        numbered_lines, truncated_at, partial_line = _apply_char_budget(
            numbered_lines, s, budget
        )
        if truncated_at is not None:
            e = truncated_at - 1
        content = "\n".join(numbered_lines)

        lang = path.split(".")[-1] if "." in path else ""
        lang_label = _EXT_LANG_MAP.get(lang, "")

        result_content = f"`{path}` ({_total} lines) — `│N│` = leading-indent column count"
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
                    f"Lines {truncated_at}–{_total} were not returned — "
                    f"call read_file again with start_line={truncated_at}.]"
                )

        meta: dict[str, Any] = {}
        if truncated_at is not None:
            meta = {"truncated": True, "resume_line": truncated_at, "line_count": _total}
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
        shown_symbols = symbols[:cap]
        # The END line is the half that was missing. This outline exists to make
        # the follow-up read_file exact, but printing only a start left the model
        # to invent end_line — and inventing it is where malformed ranges come
        # from, including inverted ones (start_line=600, end_line=460) that fail
        # the call outright and cost a turn. The extent is already computed by
        # every AST-backed outline, so surfacing it is free.
        extents = [_outline_extent(s) for s in shown_symbols]
        width = max((len(x) for x in extents), default=1)
        # Methods hang under the symbol they belong to, so the continuation
        # indent has to track the (now variable-width) range column.
        method_indent = " " * (len("  lines ") + width + 2)

        rows: list[str] = []
        for s, extent in zip(shown_symbols, extents, strict=True):
            rows.append(f"  lines {extent:>{width}}  [{s.kind}] {s.name}")
            # Methods carry no line of their own in the outline, so listing the
            # NAMES is what makes them reachable: read_symbol takes a name, and
            # a file that is one 2000-line class would otherwise offer nothing
            # between "class at line 350" and a blind range.
            methods = s.methods or []
            if methods:
                shown = ", ".join(methods[:_METHODS_PER_CLASS])
                more = f" (+{len(methods) - _METHODS_PER_CLASS} more)" if len(methods) > _METHODS_PER_CLASS else ""
                rows.append(f"{method_indent}methods: {shown}{more}")
        if len(symbols) > cap:
            rows.append(f"  … {len(symbols) - cap} more symbols (get_file_outline for the full map)")

        return (
            head
            + "\n\nOutline — each range is exact; pass it as start_line/end_line:\n"
            + "\n".join(rows)
            + "\n\nNext: read_symbol with a name above (exact, no range needed), "
              "or read_file with start_line/end_line."
        )

    @staticmethod
    def _run_search_bounded(cmd, cwd, timeout, retain_lines, cancelled=None,
                            max_line_chars=None):
        """Run a line-oriented search, keeping at most *retain_lines* of stdout.

        Returns ``(returncode, lines, total_lines, stderr)`` where *lines* is a
        prefix of the output and *total_lines* is the true count.

        ``capture_output=True`` materialises everything the tool prints before
        any cap can be applied, and the cap here is ~24 KB of content: ripgrep
        over a 108 MB log with a match on every line cost 522 MB of peak RSS to
        produce it (measured). The tool's own budget was already applied twice —
        ``max_results`` and a char cap — but both ran on a list that had to
        exist first. Nothing about a search needs the whole output resident, so
        it is streamed and only the retained prefix is kept.

        The count is still exact. That matters: reporting the DISPLAYED count as
        the match count was a real bug once (50 reported for 29,871), and the
        model reads it as "I have seen everything".

        stderr is drained on its own thread — reading stdout to exhaustion while
        the child fills the stderr pipe is the classic deadlock ``communicate``
        exists to avoid — and bounded, since a permission-denied storm is
        exactly the run that also produces the most stdout.

        stdout gets a thread of its own for a different reason: *timeout*.
        Iterating the pipe on this thread blocks in ``read()`` until a line
        arrives, so a deadline checked between lines is only ever checked when
        the search is PRODUCING — and the slow search this budget exists to
        bound is the silent one (a rare pattern over a huge tree emits nothing
        for minutes). Measured against a 1.0 s budget: ``sleep 5`` returned at
        5.01 s and ``echo hi; sleep 5`` at 5.03 s, i.e. no timeout at all.
        Waiting on an Event instead makes the deadline independent of whether
        any output ever shows up.

        ``max_line_chars`` bounds the retained line's WIDTH, the axis
        ``retain_lines`` says nothing about. rg is told the same number via
        ``--max-columns`` so the truncation happens in the child and the wide
        line never crosses the pipe; the system-grep fallback has no such flag,
        so the clamp here is what bounds it — the transient line is already
        decoded by then, but the retained list is not.

        *cancelled* is a zero-arg predicate polled on the same wait. ESC reached
        `bash` and stopped at the tool next door: a search is the other call
        that can hold a turn for two minutes, and until now nothing observed a
        cancel while it ran. Raises :class:`SearchCancelled` after tearing the
        process group down — an abandoned search must not outlive the request
        that asked for it.
        """
        import os
        import signal
        import subprocess
        import threading
        import time

        _STDERR_CAP = 64 * 1024
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            # A missing binary is an OSError, not a SubprocessError. Named here
            # rather than left to the caller's generic handler, which would
            # report "grep failed: [Errno 2]" without saying what was missing.
            raise RuntimeError(f"could not start {cmd[0]!r}: {exc}") from exc
        err_chunks: list[str] = []

        def _drain_err():
            try:
                for chunk in iter(lambda: proc.stderr.read(8192), ""):
                    if sum(map(len, err_chunks)) < _STDERR_CAP:
                        err_chunks.append(chunk)
            except (OSError, ValueError) as exc:
                # The pipe closed under us (kill path) — stdout is the result
                # that matters, so this degrades to "no stderr", never fails.
                logger.debug("search stderr drain ended early: %s", exc)

        err_thread = threading.Thread(target=_drain_err, daemon=True)
        err_thread.start()

        lines: list[str] = []
        counted = [0]
        read_done = threading.Event()

        def _drain_out():
            try:
                for line in proc.stdout:
                    counted[0] += 1
                    if len(lines) < retain_lines:
                        _kept = line.rstrip("\n")
                        if max_line_chars is not None and len(_kept) > max_line_chars:
                            _kept = _kept[:max_line_chars] + " [... long line truncated]"
                        lines.append(_kept)
            except (OSError, ValueError) as exc:
                # Same degradation as stderr: the pipe closed under us on the
                # kill path, and the prefix read so far is still the answer.
                logger.debug("search stdout drain ended early: %s", exc)
            finally:
                read_done.set()

        out_thread = threading.Thread(target=_drain_out, daemon=True)
        out_thread.start()

        deadline = time.monotonic() + timeout
        try:
            while not read_done.is_set():
                _remaining = deadline - time.monotonic()
                if _remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if read_done.wait(timeout=min(_CANCEL_POLL_INTERVAL, _remaining)):
                    break
                if cancelled is not None and cancelled():
                    raise SearchCancelled(cmd)
            # EOF on stdout means the search is done producing, so exiting is
            # not part of the search budget and gets its own grace. Charging it
            # the deadline's remainder meant a search that used its budget had
            # 0.1 s to exit — and a process killed at 0.1 s after delivering
            # every match was reported as "grep timed out", discarding a
            # complete result.
            proc.wait(timeout=_EXIT_GRACE)
        except (subprocess.TimeoutExpired, SearchCancelled):
            # The process GROUP, not the process: rg is spawned in its own
            # session, so nothing on this side reaps it and a search abandoned
            # by ESC would otherwise keep walking the tree unowned — the same
            # shape the bash cancel path exists to prevent.
            try:
                # start_new_session=True makes the child its own session
                # leader, so pgid == proc.pid. Kill the stored pid, not a
                # re-resolved getpgid(): the leader may have exited and been
                # reaped while the search tree kept walking (rg itself is the
                # leader here, but a wrapped command could exit early) —
                # getpgid() would then raise ProcessLookupError and skip the
                # kill, orphaning the very grandchildren this teardown exists
                # for. killpg() targets the GROUP, which survives its leader.
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError) as exc:
                logger.debug("search process group already gone: %s", exc)
            proc.wait(timeout=5)
            raise
        finally:
            # Joined before the pipes close: the drain threads are reading these
            # very handles, and closing under them turns an orderly EOF into the
            # ValueError the drains only tolerate as a degradation.
            out_thread.join(timeout=2)
            err_thread.join(timeout=2)
            for _stream in (proc.stdout, proc.stderr):
                try:
                    _stream.close()
                except (OSError, ValueError) as exc:
                    logger.debug("search pipe close failed: %s", exc)
        return proc.returncode, lines, counted[0], "".join(err_chunks)

    def _search_cancel_requested(self) -> bool:
        """Zero-arg cancel predicate for ``_run_search_bounded``'s poll loop.

        Live per-poll observation of BOTH sources: ``config.cancel_event`` is
        read FRESH each call (the design-chat REPL swaps it per turn — a value
        captured once goes stale and the poll then watches an event nobody
        will ever set), OR'd with the innermost per-call cancel scope so a
        search abandoned by its caller — MCP ``wait_for`` timeout, aborted
        parallel batch — tears its process group down instead of walking the
        tree to the 120 s bound unowned. Mirrors test_tools' runner check.
        """
        if _cancel_probe(getattr(self, "config", None))():
            return True
        _scope = current_cancel_event()
        return _scope is not None and _scope.is_set()

    def _tool_grep(self, args: dict[str, Any]) -> "ToolResult":
        """Search for a pattern across files using grep (or ripgrep if available)."""
        import shutil
        import subprocess

        # Safety limit: ~33k tokens max per result (prevent token explosion from context+N on long lines)
        # Match bash tool's BASH_OUTPUT_MAX_CHARS threshold for consistency.
        from ..config.thresholds import config as _thresholds
        _MAX_RESULT_CHARS = _thresholds.tokens.BASH_OUTPUT_MAX_CHARS
        # Width companion to the cap above. Without it the cap bounded only how
        # many lines were emitted, and one match in a minified file returned
        # 34,000,257 chars against this 60,000-char "hard limit".
        _MAX_LINE_CHARS = _thresholds.lines.SEARCH_MAX_LINE_CHARS

        pattern = args.get("pattern", "").strip()
        if not pattern:
            return self._make_result(ok=False, content="", error="'pattern' is required")

        search_path = args.get("path", "").strip() or "."
        search_path = self._correct_bias_path(search_path)
        # Security gate: webapp path must be confined to repo_root (mirrors glob's gate).
        # confine=False respects unrestricted_read: CLI can grep outside, webapp cannot.
        #
        # The RESOLVED path is deliberately discarded rather than substituted for
        # search_path, because it is ABSOLUTE and rg echoes back whatever path it
        # was given: searching "." prints "./a.py:1:...", searching the resolved
        # root prints "/private/tmp/repo/a.py:1:..." (measured). Every consumer of
        # this output — the model included — reads repo-relative paths, so
        # substituting would reformat every grep result to no benefit. This is a
        # gate only; the raw path is what gets searched.
        if self._secure_path(search_path) is None:
            return self._make_result(ok=False, content="", error=f"path {search_path!r} is outside the repository")
        # Floor at 1, matching glob / find_relevant_files: a correct-type 0 is
        # left intact by the argument-repair layer (only null is dropped), so it
        # reaches the handler — and without the floor grep rendered "truncated to
        # 0 of N matches", a useless answer that reads as "found nothing to show".
        # (null is handled upstream by ArgumentRepairer → key dropped → default.)
        max_results = max(1, min(int(args.get("max_results", 200)), 500))
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

        # ``--max-columns-preview`` needs rg >= 12.0. Dropped and retried once
        # if the installed rg rejects it — see _unsupported_flag. The retained
        # width clamp below is independent of the flag, so the drop costs
        # nothing but the in-child truncation.
        _rg_preview = True
        # Three attempts, because there are now two independent reasons to
        # retry (unsupported flag, uncompilable pattern) and they can both fire
        # on the same call. Each one flips a latch that cannot flip back, so the
        # loop still terminates well inside the bound.
        for _attempt in range(3):
            if use_rg:
                cmd = [_rg, "-n", "--no-heading",
                       # Truncate wide lines in the CHILD, so a match inside a
                       # minified bundle never crosses the pipe at all. The
                       # match COUNT is unaffected — rg still counts the line.
                       "--max-columns", str(_MAX_LINE_CHARS)]
                if _rg_preview:
                    # The preview form keeps the head of the line; the plain
                    # flag replaces it with a bare "omitted" marker, which
                    # tells the model nothing about what matched.
                    cmd.append("--max-columns-preview")
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

            # Retained prefix: what ranking and display can possibly need. The
            # BM25 pass below used to do this cut itself, AFTER the whole output
            # was already a list in memory; doing it while reading bounds the
            # memory too, and the exact match count still comes back separately.
            _retain = max(max_results * 20, 5000)
            try:
                _rc, _lines, _total, _stderr = self._run_search_bounded(
                    cmd, self.repo_root, 120, _retain,
                    cancelled=self._search_cancel_requested,
                    # Redundant for rg (already clamped in the child by
                    # --max-columns) and load-bearing for the system-grep
                    # fallback, which has no equivalent flag.
                    max_line_chars=_MAX_LINE_CHARS,
                )
            except SearchCancelled:
                # ok=False, mirroring the bash cancel: the user withdrew the
                # request, so this is not an answer.
                return self._make_result(
                    ok=False, content="", error="Operation cancelled",
                    retryable=False, metadata={"cancelled": True},
                )
            except subprocess.TimeoutExpired:
                return self._make_result(ok=True, content=f"grep timed out (pattern={pattern!r})")
            except Exception as e:
                return self._make_result(ok=False, content="", error=f"grep failed: {e}")

            if _rc != 2 or _lines:
                # Not an error, or an error that still produced matches: exit 2
                # with output means the tree WAS searched and only some files
                # were skipped. Retrying that as a fixed string would replace a
                # correct regex answer with a literal one.
                break
            _bad_flag = _unsupported_flag(_stderr) if use_rg else None
            if _bad_flag is not None and _rg_preview:
                # Old ripgrep. Drop the flag it does not know and search again;
                # the width clamp on retention still bounds the output.
                logger.debug("grep: rg rejected %s — retrying without it", _bad_flag or "a flag")
                _rg_preview = False
                continue
            if not use_fixed and not _search_ran_despite_errors(_stderr):
                # Nothing matched and the failure is not per-file access noise
                # → the pattern is the suspect. Retry it as a literal.
                use_fixed = True
                continue
            break

        # Exit 2 conflates "bad pattern" with "one file was unreadable". When
        # the search demonstrably ran, re-map it onto the code its OUTPUT says
        # it earned, so the branches below report matches / no-matches instead
        # of failing the call outright.
        _skipped_note = ""
        if _rc == 2 and (_lines or _search_ran_despite_errors(_stderr)):
            _first_err = next(
                (ln.strip() for ln in (_stderr or "").splitlines() if ln.strip()), "",
            )
            _skipped_note = (
                f"\n(note: some files could not be read and were skipped — {_first_err[:200]})"
            )
            _rc = 0 if _lines else 1

        if _rc == 0 or (_rc == 1 and _lines):
            lines = _lines
            # Captured BEFORE the ranking block, which SELECTS the top
            # ``max_results`` and discards the rest — after it, ``len(lines)`` is
            # the displayed count, not the match count. Deriving the header and
            # the truncation notice from the post-selection list reported
            # "50 matches" for a pattern with 29,871 and suppressed the "refine
            # your pattern" hint entirely, so the agent believed it had seen
            # every hit and stopped searching. Only stop-word patterns
            # (tokenize("import") == []) escaped, because they skip ranking.
            total = _total

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
            if len(lines) > max_results and context == 0:
                _tok = CodeTokenizer()
                _qtokens = _tok.tokenize(pattern)
                if _qtokens:
                    # Pre-truncate to bound BM25 cost for pathological match sets.
                    # BM25 on 50k+ lines is O(n*q) CPU — pre-cutting to a sensible
                    # cap keeps ranking quality (top N out of shuffled filesystem
                    # order ≅ top N out of K*N) while bounding worst-case time.
                    # (The pre-cut this comment describes now happens while
                    # the output is read — see _run_search_bounded's `retain`.)
                    # Ranking itself is single-sourced in agent/bm25.py's
                    # bm25_rank — this setup used to be a copy of the
                    # symbol_search twin; scores are bit-identical.
                    _n = len(lines)
                    _scores = bm25_rank(_qtokens, [_tok.tokenize(_item_) for _item_ in lines])
                    # Select top-N by relevance, then restore file/line order for
                    # readability. Ranking's job is which results survive the cap,
                    # not the display order — BM25-scrambled order within a single
                    # file (lines 136,112,36) is confusing and was never the intent.
                    _top = sorted(range(_n), key=lambda i: _scores[i], reverse=True)[:max_results]
                    _top.sort()
                    lines = [lines[i] for i in _top]

            truncated = total > max_results

            # --- Character-based truncation guard: prevent token explosion ---
            # context=N + long-line files (logs, JSON, stacktraces) can produce
            # massive output even with few matches.  Enforce a hard char limit.
            display_chars = 0
            display_lines = []
            for _item_ in lines[:max_results]:
                display_chars += len(_item_) + 1  # +1 for newline
                if display_chars > _MAX_RESULT_CHARS:
                    # Include this line but stop. It is CUT to what the budget
                    # had left: appending it whole made this "hard char limit"
                    # a limit plus one line of unbounded width, and a single
                    # match in a minified bundle returned 34,000,257 chars —
                    # 566x the cap, into the conversation history. The two
                    # clamps upstream (rg --max-columns, the retention clamp)
                    # keep a normal line intact here; this is the backstop for
                    # the one line that still straddles the budget.
                    _left = _MAX_RESULT_CHARS - (display_chars - len(_item_) - 1)
                    display_lines.append(_item_[:_left] if _left > 0 else "")
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
            result += _skipped_note

            return self._make_result(
                ok=True, content=result,
                metadata={"files_skipped": True} if _skipped_note else {},
            )
        if _rc == 1:
            tool_name = "rg" if use_rg else "grep"
            return self._make_result(
                ok=True,
                content=f"{tool_name}: {pattern!r} in {search_path} — no matches.{_skipped_note}",
                metadata={"files_skipped": True} if _skipped_note else {},
            )
        stderr = (_stderr or "").strip()[:500]
        return self._make_result(
            ok=False, content="",
            error=f"grep failed (exit={_rc}): {stderr}",
        )

    def _tool_read_symbol(self, args: dict[str, Any]) -> "ToolResult":
        """Read a symbol definition (function, class, or variable) by name.

        When SymbolDef.end_line is available (AST end_lineno), read the full
        symbol body — not just a fixed ±context_lines window — so the result
        covers the whole definition even for long functions/classes.
        """
        name = args.get("name", "")
        if not name:
            # ok=False: a missing required argument is a failed call, not an
            # answer. Reported as success it reads to the model as "asked and
            # answered", so the retry it needs never happens.
            return self._make_result(
                ok=False, content="", error="'name' is required (the symbol to read).",
            )
        file_path = args.get("file_path") or None
        context_lines = int(args.get("context_lines", 10))

        defs = self._symbol_searcher.find_symbol(name, search_path=file_path)
        if not defs:
            return self._make_result(ok=True, content=f"Symbol '{name}' not found.")
        sym = defs[0]

        # When the name matches N definitions and the agent didn't pass
        # file_path=, it picks the first silently.  Tell it how many there are
        # and where the rest live — symmetry with _tool_find_symbol which
        # already reports the count.
        multi_header = ""
        if len(defs) > 1:
            _others = ", ".join(f"`{d.file}:{d.line}`" for d in defs[1:])
            multi_header = (
                f"Showing 1st of {len(defs)} definitions of `{name}`"
                f" — others at {_others}"
                f" (pass file_path= to narrow).\n\n"
            )

        abs_path = Path(self.repo_root) / sym.file
        if not abs_path.exists():
            return self._make_result(ok=True, content=f"File '{sym.file}' not found.")

        # P25-1: this read the WHOLE file and only then sliced the window —
        # the one read-tool path bypassing the READ_FILE_MAX_CHARS SSOT. The
        # output budget below bounded what the model SAW, but never what was
        # loaded into memory; a 100 MB file with the symbol near the end was
        # fully materialised on every call. The AST index already carries the
        # exact line span, so only [start, start+count) is read — O(window)
        # memory for any file size.
        context_lines = max(0, min(context_lines, 100))
        start = max(0, sym.line - 1 - context_lines)
        if sym.end_line and sym.end_line >= sym.line:
            # Full body: leading context (covers decorators) + trailing context.
            count = (sym.end_line + context_lines) - start
        else:
            # Fallback: fixed window around the definition line.
            count = (sym.line + context_lines) - start
        window = _read_symbol_window(abs_path, start, count)
        actual_end = start + len(window)

        numbered_lines = [
            _format_numbered_line(i, ln)
            for i, ln in enumerate(window, start=start + 1)
        ]

        lang = sym.file.split(".")[-1] if "." in sym.file else ""
        lang_label = _EXT_LANG_MAP.get(lang, lang)

        loc = f"{sym.file}:{sym.line}"
        if sym.end_line and sym.end_line > sym.line:
            loc += f"-{sym.end_line}"

        # ── Char budget ─────────────────────────────────────────────────────
        # read_symbol was the ONLY read tool without one. A single symbol on a
        # long file (WriteToolsMixin ≈ 350k chars / 89k tokens) could swallow the
        # whole context window: context_budget.fit_messages() leaves it alone
        # (deliberately) and SlidingWindowContext is message-count based — one
        # giant message survives until it ages out.
        #
        # READ_FILE_MAX_CHARS is the SSOT cap shared by read_file / bash / grep,
        # and _apply_char_budget is the SSOT for the arithmetic — `start` is a
        # 0-based index, so the origin handed over is `start + 1`, matching the
        # 1-based numbering above.
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        kept, truncated_at, partial_line = _apply_char_budget(
            numbered_lines, start + 1, budget
        )
        context = "\n".join(kept)

        content = multi_header + (
            f"**{sym.kind}** `{name}` defined in `{loc}`"
            f" — `│N│` = leading-indent column count\n"
            f"```{lang_label}\n{context}\n```")
        if truncated_at is not None:
            if partial_line is not None:
                content += (
                    f"\n\n[Truncated at the {budget:,}-char output budget. "
                    f"Line {partial_line} alone exceeds it, so only its first "
                    f"{budget:,} chars were returned; the REST OF THAT LINE was "
                    f"dropped and is NOT recoverable by re-reading (start_line is "
                    f"line-granular). Continue at start_line={truncated_at} for "
                    f"the next line.]"
                )
            else:
                content += (
                    f"\n\n[Truncated at the {budget:,}-char output budget. "
                    f"Lines {truncated_at}–{actual_end} were not returned — "
                    f"call read_file with start_line={truncated_at} to continue.]"
                )

        # Same metadata shape read_file emits, for the same reason: the prose
        # notice above is for the model, but the agent loop and telemetry read
        # the metadata. Emitting the notice WITHOUT these keys made read_symbol
        # truncation the one form of dropped output no consumer could detect
        # programmatically. ``line_count``/``resume_line`` are FILE line numbers
        # (not symbol-relative) because the resumption call is a read_file.
        meta: dict[str, Any] = {}
        if truncated_at is not None:
            # Streaming pass — only paid on the rare truncation path.
            meta = {"truncated": True, "resume_line": truncated_at, "line_count": _count_file_lines(abs_path)}
            if partial_line is not None:
                meta["partial_line"] = partial_line
        return self._make_result(ok=True, content=content, metadata=meta)

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
                    lines.extend(f"  {sr['file']}:{sr['line']}  {sr['context'][:80]}" for sr in info["sample_references"])
                if "other_definitions" in info:
                    lines.append("\nOther definitions:")
                    lines.extend(f"  [{od['kind']}] {od['file']}:{od['line']}" for od in info["other_definitions"])

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
        lines.extend(f"  {r.file}:{r.line}:{r.col}  {r.context}" for r in refs)

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
            # Same reasoning as read_file's over-cap outline (_outline_extent):
            # this tool's own closing line sends the caller to read_file with a
            # range, so withholding the end line makes it guess one.
            _extent = _outline_extent(s)
            loc = f"(lines {_extent})" if "–" in _extent else f"(line {_extent})"
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
        # The corpus walk caps at RAG_MAX_FILES; when the cap is hit the index is
        # incomplete, so "no results" does NOT mean "absent in repo". Surface the
        # truncation flag in content + metadata so the agent can fall back to
        # grep/glob (a full-repo scan the cap does not bind). getattr guards
        # against test doubles that lack the property.
        index_truncated = bool(getattr(self._rag_searcher, "index_truncated", False))
        logger.debug(
            "RAG search invoked: query=%s results=%d truncated=%s",
            query,
            len(results),
            index_truncated,
        )
        if not results:
            if index_truncated:
                content = (
                    "No relevant files found in the indexed corpus, but the RAG index "
                    "hit its file cap and is incomplete — a match may exist in files "
                    "beyond the cap. Retry with grep or glob, which scan the full repo."
                )
            else:
                content = "No relevant files found for the given query."
            return self._make_result(
                ok=True, content=content,
                metadata={"files_found": [], "result_count": 0, "index_truncated": index_truncated},
            )

        lines: list[str] = [f"Top {len(results)} relevant file(s) for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.file}  (score: {r.score:.2f}, line ~{r.line})")
            if r.snippet.strip():
                lines.append(f"     {r.snippet[:110]}")
        lines.append("\nUse read_file to inspect these, or get_file_outline first if a file is large.")
        if index_truncated:
            lines.append(
                "\nNote: the RAG index hit its file cap and is incomplete — a more "
                "relevant file may exist beyond the cap. Use grep or glob to scan "
                "the full repository."
            )
        return self._make_result(
            ok=True, content="\n".join(lines),
            metadata={
                "files_found": [r.file for r in results],
                "result_count": len(results),
                "index_truncated": index_truncated,
            },
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

        # providers is first-party (requests is a hard dependency) and
        # _try_ocr_base64 degrades internally when OCR deps are missing —
        # the old except ImportError fallback was dead code.
        from external_llm.providers import _try_ocr_base64 as _ocr_fn
        ocr_text = _ocr_fn(data)

        if ocr_text:
            return self._make_result(
                ok=True,
                content=f"[Image OCR — {abs_path.name}]\n{ocr_text}",
            )
        return self._make_result(
            ok=True,
            content=f"[Image OCR — {abs_path.name}] No text detected in the image. "
                    "The image may contain only graphics without text, or OCR could not read it.",
        )

