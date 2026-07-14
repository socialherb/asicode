"""Shared terminal in-place write coordination.

Two independent components draw live, in-place status rows on stdout using
``\\r\\x1b[2K`` WITHOUT a trailing newline:

  * asi's ``_ProgressPrinter`` (the design-agent tool ticker), and
  * the collaboration ``StreamingDisplay`` (Claude Code agent activity).

Both must serialize those writes against the root logger's terminal handler so
a WARNING/ERROR never glues onto the right edge of a live row. The lock and the
"a row is pending (drawn without a newline)" flag live here — not in asi —
because asi runs as ``__main__`` while ``StreamingDisplay`` is imported as a
library module; only a shared importable module guarantees both sides see the
SAME objects (importing asi from the library would create a second copy).

Protocol:
  * Hold ``TERM_WRITE_LOCK`` around every in-place stdout write AND around the
    log handler's emit, so a ticker re-render can't interleave with a log line.
  * Call ``set_row_pending(True)`` after drawing an in-place row, and
    ``set_row_pending(False)`` after any write that ends at the beginning of a
    line (a cleared row, or a permanent line ending in ``\\n``).
  * The log handler (asi ``_RowSafeEmitMixin``) checks ``row_pending()`` and,
    if set, writes one newline to break the live row before emitting the record.
"""
from __future__ import annotations

import threading

# Held around every in-place stdout write AND around log-handler emit.
TERM_WRITE_LOCK = threading.RLock()

_row_pending = False


def set_row_pending(value: bool) -> None:
    """Mark whether an in-place row is currently on screen without a newline."""
    global _row_pending
    _row_pending = bool(value)


def row_pending() -> bool:
    """True when a live in-place row is pending (must be broken before a log)."""
    return _row_pending
