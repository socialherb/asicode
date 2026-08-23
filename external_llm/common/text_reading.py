"""text_reading.py -- Shared text-file reading with encoding fallback (SSOT).

The write-tools family must read a file back with the SAME bytes it was
written with.  A strict ``utf-8`` read (the historical behaviour in
write_tools_edit_mixin) raised UnicodeDecodeError on cp949/latin-1 sources,
which escaped the ``except OSError`` guard and fell into the outer
``except Exception`` — silently skipping the post-write syntax gate
(``ok:True``/``skipped:True`` with no observable reason).  This module is
the single reader both mixins use so the fallback chain cannot diverge again.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "latin-1")


def read_text_with_encoding_fallback(
    abs_path: str,
    encodings: tuple[str, ...] = _DEFAULT_ENCODINGS,
) -> tuple[str | None, str | None]:
    """Read a text file trying each encoding in order.

    Returns ``(content, encoding)`` on success.  Returns ``(None, None)``
    when the file cannot be decoded with any of ``encodings``.  Raises
    ``OSError`` when the file cannot be opened (missing file, permission,
    directory, ...) — callers decide how to surface that.
    """
    for enc in encodings:
        try:
            with open(abs_path, encoding=enc) as fh:
                return fh.read(), enc
        except (UnicodeDecodeError, UnicodeError):
            logger.debug("read failed with %s, retry next encoding", enc, exc_info=True)
            continue
    return None, None


def read_line_window(path: Path, start: int, count: int) -> list[str]:
    """Stream lines ``[start, start+count)`` of ``path`` (0-based).

    Added P26-3: the canonical implementation of the windowed line read first
    introduced in read_tools (P25-1, ``_read_symbol_window``) so symbol
    search's Go signature-line read shares the exact same window semantics
    instead of a drifting copy.

    * ``\\n``-only splitting aligned with ``ast.lineno`` / git numbering
      (``\\f``/``\\v``/``\\x85``/``\\u2028`` stay inside a line, exactly as
      the AST model expects),
    * universal-newline translation matching ``Path.read_text``
      (``\\r\\n`` / lone ``\\r`` become ``\\n``),
    * each line's trailing ``\\n`` dropped the same way the old whole-file
      ``split("\\n")`` reads did.

    Returns fewer than *count* lines at EOF, or ``[]`` when the file has no
    line at index *start* (EOF was hit while skipping).  O(count) memory
    regardless of file size.
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        for _ in range(start):
            if fh.readline() == "":
                return []
        out: list[str] = []
        for _ in range(count):
            line = fh.readline()
            if line == "":
                break
            out.append(line.rstrip("\n"))
        return out
